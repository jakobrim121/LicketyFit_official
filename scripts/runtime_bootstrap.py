#!/usr/bin/env python3
"""Prepare one shared runtime cache before a multiprocessing fit.

The production driver must not initialize OpenMP in a parent that will later
fork. Historically it therefore ran the complete warm-up independently in
every event worker. With many workers and clean cosmic generations that turns
one cache-construction cost into dozens or hundreds of simultaneous warm-ups.

This module runs one disposable serial child when the source-addressed cache is
genuinely empty, then disables the redundant worker warm-ups. No fit result from
the disposable child is retained.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Iterable, Iterator, Mapping

from LicketyFit.runtime_cache import runtime_cache_content_tag


_SIGNATURE_PREFIXES = (
    "AUTO_",
    "BOUNDARY_",
    "COMPOUND_",
    "COSMIC_",
    "DETECTOR_",
    "EMITTER_",
    "FIT_",
    "GLOBAL_",
    "JOINT_",
    "LF_EXACT_",
    "LF_VALIDATED_",
    "MCS_",
    "PMT_",
    "T0_",
    "WCSIM_",
    "WCTE_",
)

_SIGNATURE_EXCLUDED = {
    "AUTO_PREPARE_MULTIPROCESS_RUNTIME_CACHE",
    "BEAM_P",
    "CONFIG_ROOT_FILE",
    "ENERGY_TRUE",
    "EVENT_SOURCE",
    "LF_COSMIC_CHILD_QUIET",
    "LF_COSMIC_SUPERVISED_CHILD",
    "LF_EVENT_COUNT",
    "LF_EVENT_START_INDEX",
    "LF_OUTPUT_FILE",
    "LF_PUBLIC_DRIVER_RELEASE",
    "LF_RESOLVED_RUNTIME_CACHE_DIR",
    "LF_RUN_CONFIG_FILE",
    "LF_RUN_CONFIG_SHA256",
    "LF_RUNTIME_CACHE_DIR",
    "LF_RUNTIME_CACHE_PROGRESS_INTERVAL_SECONDS",
    "MAX_EVENTS_TO_FIT",
    "N_EVENTS",
    "N_ROOT_ENTRIES",
    "NPROC",
    "N_EVENTS_PER_BATCH",
    "PRINT_BATCH_PROGRESS",
    "PRINT_CHECKPOINT_MESSAGES",
    "PRINT_EVENT_RESULTS",
    "SAVE_AFTER_EACH_BATCH",
    "SAVE_DETAILED_EVENT_RESULTS",
    "TOT_EVENTS",
    "RUN",
    "USER_EVENT_FILE",
    "USER_EVENT_KEY",
    "VERBOSE_SETUP",
    "WARM_FIT_KERNELS",
    "WCSIM_INPUT_FILE",
    "WCSIM_TRUTH_EVENT_ID_OFFSET",
    "WCSIM_TRUTH_EVENT_INDEX_STEP_SIZE",
    "WCSIM_TRUTH_INCLUDE_OPTIONAL_DETAILS",
    "WCSIM_TRUTH_INDEX_CACHE_DIR",
    "WCSIM_TRUTH_IO_WORKERS",
    "WCSIM_TRUTH_POSITION_OFFSET_MM",
    "WCSIM_TRUTH_PRIMARY_TRACK_ID",
    "WCSIM_TRUTH_ROOT_FILE",
    "WCSIM_TRUTH_TREE",
    "WCSIM_TRUTH_UPROOT_STEP_SIZE",
    "WCSIM_TRUTH_USE_EVENT_INDEX_CACHE",
    "WCSIM_USE_TRUTH_ROOT",
    "WCTE_EXPECTED_KE_MEV",
}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _runtime_root(project_root: Path, environment: Mapping[str, str]) -> Path:
    cache_base = (
        str(environment.get("LF_RUNTIME_CACHE_DIR", "")).strip()
        or str(environment.get("TMPDIR", "")).strip()
        or "/tmp"
    )
    project_tag = runtime_cache_content_tag(project_root)
    return Path(cache_base).expanduser() / (
        f"licketyfit-{os.getuid()}-{project_tag}-"
        f"py{sys.version_info.major}{sys.version_info.minor}"
    )


def _cache_inventory(runtime_root: Path) -> dict[str, int]:
    """Return the compiled artifacts a completed bootstrap depends on."""
    inventory: dict[str, int] = {}
    for directory, suffixes in (
        (runtime_root / "numba", {".nbi", ".nbc"}),
        (runtime_root / "native", {".so", ".dylib", ".dll"}),
    ):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            try:
                inventory[path.relative_to(runtime_root).as_posix()] = int(
                    path.stat().st_size
                )
            except OSError:
                continue
    return dict(sorted(inventory.items()))


def _ready_marker_is_valid(
    marker: Path,
    *,
    runtime_root: Path,
    source: str,
    mode: str,
    signature: str,
) -> bool:
    """Reject interrupted, stale, or partially deleted cache markers."""
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        if int(record.get("schema", -1)) != 2:
            return False
        if str(record.get("source", "")) != str(source):
            return False
        if str(record.get("mode", "")) != str(mode):
            return False
        if str(record.get("signature", "")) != str(signature):
            return False
        if str(record.get("python", "")) != sys.version.split()[0]:
            return False
        artifacts = record.get("compiled_artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            return False
        for relative_text, expected_size in artifacts.items():
            relative = Path(str(relative_text))
            if relative.is_absolute() or ".." in relative.parts:
                return False
            artifact = runtime_root / relative
            if not artifact.is_file():
                return False
            actual_size = int(artifact.stat().st_size)
            # A later mode/configuration may append another overload to a
            # shared Numba index. That is a valid cache extension. Compiled
            # overload payloads and native libraries themselves are immutable.
            if relative.suffix.lower() == ".nbi":
                if actual_size < int(expected_size):
                    return False
            elif actual_size != int(expected_size):
                return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _warm_signature(
    environment: Mapping[str, str],
    *,
    source: str,
    mode: str,
    configuration_names: Iterable[str] | None = None,
    project_root: str | Path | None = None,
) -> str:
    configured = (
        None
        if configuration_names is None
        else {str(name) for name in configuration_names}
    )
    selected = {
        str(name): str(value)
        for name, value in environment.items()
        if str(name) not in _SIGNATURE_EXCLUDED
        and (
            (configured is not None and str(name) in configured)
            or (
                configured is None
                and (
                    str(name).startswith(_SIGNATURE_PREFIXES)
                    or str(name) in {
                        "DATA_SOURCE", "FIT_MODE", "LIKELIHOOD_MODE"
                    }
                )
            )
        )
    }
    payload = {
        "schema": 2,
        "source": str(source),
        "mode": str(mode),
        # Numba's on-disk locator contains a hash of the source directory even
        # though LicketyFit's outer runtime root is content-addressed. Moving
        # an extraction therefore requires a distinct readiness certificate.
        "project_root": (
            None if project_root is None else str(Path(project_root).resolve())
        ),
        "machine": _machine_identity(),
        "configuration": dict(sorted(selected.items())),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _machine_identity() -> dict[str, str]:
    """Describe the CPU target used by Numba/native cached machine code."""
    identity = {
        "platform": str(sys.platform),
        "machine": str(platform.machine()),
        "processor": str(platform.processor()),
    }
    try:
        import llvmlite.binding as llvm

        identity["llvm_cpu_name"] = str(llvm.get_host_cpu_name())
        identity["llvm_cpu_features"] = str(
            llvm.get_host_cpu_features().flatten()
        )
    except (AttributeError, ImportError, RuntimeError):
        pass
    return identity


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Serialize cold-cache preparation when two launchers start together."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        stream.close()


def _bootstrap_environment(
    environment: Mapping[str, str],
    *,
    source: str,
    output_path: Path,
) -> dict[str, str]:
    prepared = dict(environment)
    prepared.update({
        "NPROC": "1",
        "N_EVENTS_PER_BATCH": "1",
        "WARM_FIT_KERNELS": "1",
        "SAVE_AFTER_EACH_BATCH": "0",
        "SAVE_DETAILED_EVENT_RESULTS": "0",
        "PRINT_EVENT_RESULTS": "0",
        "PRINT_BATCH_PROGRESS": "0",
        "PRINT_CHECKPOINT_MESSAGES": "0",
        "VERBOSE_SETUP": "0",
        "LF_OUTPUT_FILE": str(output_path),
        "LF_COSMIC_SUPERVISED_CHILD": "1",
        "LF_COSMIC_CHILD_QUIET": "1",
        "LF_RUNTIME_BOOTSTRAP_CHILD": "1",
        "PYTHONUNBUFFERED": "1",
    })
    if str(source) == "wcsim":
        prepared["TOT_EVENTS"] = "1"
        prepared["WCSIM_USE_TRUTH_ROOT"] = "0"
        prepared.pop("WCSIM_TRUTH_ROOT_FILE", None)
    else:
        # In the WCTE driver this is a count *after* LF_EVENT_START_INDEX,
        # unlike a stop index. Keep the disposable job to exactly one selected
        # muon even when the production request starts later in the selection.
        prepared["MAX_EVENTS_TO_FIT"] = "1"
        prepared["TOT_EVENTS"] = "1"
    return prepared


def _run_bootstrap_process(
    *,
    driver_path: Path,
    environment: Mapping[str, str],
    heartbeat_seconds: float,
) -> tuple[int, str, float]:
    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-u", str(driver_path)],
        cwd=str(driver_path.parent),
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = ""
    interval = max(5.0, float(heartbeat_seconds))
    while True:
        try:
            output, _ = process.communicate(timeout=interval)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.perf_counter() - started
            print(
                "  shared runtime preparation is still active "
                f"({elapsed:.0f} s elapsed)",
                flush=True,
            )
    return int(process.returncode), str(output or ""), float(
        time.perf_counter() - started
    )


def prepare_multiprocess_runtime(
    environment: Mapping[str, str],
    *,
    project_root: str | Path,
    driver_path: str | Path,
    source: str,
    configuration_names: Iterable[str] | None = None,
    enabled: bool = True,
    heartbeat_seconds: float = 15.0,
) -> dict[str, str]:
    """Return a production environment without replicated worker warm-ups."""
    production = dict(environment)
    try:
        workers = max(1, int(float(str(production.get("NPROC", "1")))))
    except (TypeError, ValueError):
        workers = 1
    if workers <= 1 or not enabled or not _truthy(
        production.get("WARM_FIT_KERNELS", "1")
    ):
        return production

    # These are driver-resolved internals, not supported launcher inputs. An
    # inherited shell value could otherwise send the serial and production
    # processes to a cache outside the verified source-addressed runtime root.
    # LF_RUNTIME_CACHE_DIR remains the supported way to choose the cache base.
    for internal in (
        "NUMBA_CACHE_DIR", "LF_NATIVE_CACHE_DIR",
        "LF_RESOLVED_RUNTIME_CACHE_DIR",
    ):
        production.pop(internal, None)

    project = Path(project_root).resolve()
    driver = Path(driver_path).resolve()
    mode = str(production.get("FIT_MODE", "general")).strip().lower()
    runtime_root = _runtime_root(project, production)
    signature = _warm_signature(
        production,
        source=str(source),
        mode=mode,
        configuration_names=configuration_names,
        project_root=project,
    )
    state_dir = runtime_root / "bootstrap"
    marker = state_dir / f"{source}-{mode}-{signature}.ready.json"
    lock = state_dir / f"{source}-{mode}-{signature}.lock"
    bootstrap_output = state_dir / f"{source}-{mode}-{signature}.discarded.dict"

    state_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(lock):
        if _ready_marker_is_valid(
            marker,
            runtime_root=runtime_root,
            source=str(source),
            mode=mode,
            signature=signature,
        ):
            print(
                "LicketyFit startup: shared compiled runtime cache is ready; "
                f"starting {workers} workers without repeated warm-up.",
                flush=True,
            )
        else:
            print(
                "LicketyFit startup: preparing the shared compiled runtime "
                "cache once in a serial disposable process.",
                flush=True,
            )
            print(
                "  This is a one-time source/Python-ABI cost; event results "
                "from the disposable process are discarded.",
                flush=True,
            )
            bootstrap_env = _bootstrap_environment(
                production, source=str(source), output_path=bootstrap_output
            )
            try:
                bootstrap_output.unlink(missing_ok=True)
            except OSError:
                pass
            returncode, output, elapsed = _run_bootstrap_process(
                driver_path=driver,
                environment=bootstrap_env,
                heartbeat_seconds=float(heartbeat_seconds),
            )
            try:
                bootstrap_output.unlink(missing_ok=True)
            except OSError:
                pass
            if returncode != 0:
                tail = "\n".join(output.splitlines()[-40:])
                raise RuntimeError(
                    "Serial runtime-cache preparation failed before production "
                    f"workers were created (exit status {returncode})."
                    + (f"\nChild output:\n{tail}" if tail else "")
                )
            artifacts = _cache_inventory(runtime_root)
            if not artifacts:
                raise RuntimeError(
                    "Serial runtime-cache preparation completed but produced "
                    "no compiled Numba/native artifacts; refusing to start "
                    "multiprocess workers with an unverified cache."
                )
            record = {
                "schema": 2,
                "source": str(source),
                "mode": mode,
                "signature": signature,
                "elapsed_seconds": float(elapsed),
                "python": sys.version.split()[0],
                "project_root": str(project),
                "machine": _machine_identity(),
                "compiled_artifacts": artifacts,
            }
            temporary = marker.with_name(marker.name + f".tmp.{os.getpid()}")
            temporary.write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(temporary, marker)
            print(
                f"  shared runtime cache prepared in {elapsed:.1f} s.",
                flush=True,
            )

    production["WARM_FIT_KERNELS"] = "0"
    production["LF_SHARED_RUNTIME_CACHE_POLICY"] = (
        "serial_once_then_no_worker_warmup_v1"
    )
    production["LF_SHARED_RUNTIME_CACHE_SIGNATURE"] = signature
    return production
