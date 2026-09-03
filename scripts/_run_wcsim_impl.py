"""Private implementation loaded by ``run_wcsim.py``.

Keep user-editable choices in the public launcher; this module owns validation, environment translation, and process orchestration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
from pathlib import Path
from pathlib import PurePosixPath
import pickle
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time


_DRIVER = Path(__file__).resolve().with_name("batch_fit_driver.py")
# CPython never caches bytecode for the script named on the command line, so
# launching the 44k-line driver directly recompiles it on every process start
# (~0.3 s, and once per bootstrap/production/chunk process).  This stub runs
# the same module through the import system, which maintains
# scripts/__pycache__.  __file__, sys.argv and sys.path[0] are unchanged.
_DRIVER_ENTRY = _DRIVER.with_name("lf_driver_main.py")
if not _DRIVER_ENTRY.is_file():
    _DRIVER_ENTRY = _DRIVER
_PROJECT_ROOT = _DRIVER.parent.parent
_GEOMETRY_SUBMODULE = _PROJECT_ROOT / "Geometry"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_RUNTIME_DEPS = _PROJECT_ROOT / "runtime_deps"
if _RUNTIME_DEPS.is_dir() and str(_RUNTIME_DEPS) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DEPS))
_CHARGE_LIKELIHOODS = {
    "poisson_pe", "compound_spe_profile", "compound_spe_profile_reference",
    "compound_spe_calibrated",
}
_REMOVED_MODEL_ENV_SWITCHES = {
    "MCS_RECONSTRUCTION_MODE",
    "EMITTER_ENABLE_PRIMARY_MCS",
    "ENABLE_PRIMARY_MCS",
    "COSMIC_FERMI_EYGES_ENABLED",
    "COSMIC_COHERENT_PROFILE_ENABLED",
    "COSMIC_JOINT_ENERGY_RANGE_ENABLED",
}

from LicketyFit.absolute_light_calibration import resolve_manifest_calibration
from LicketyFit.fixed_parameters import resolve_fixed_parameter_environment
from LicketyFit.runtime_cache import resolve_runtime_cache_location
from LicketyFit.run_configuration import (
    FIT_PARTICLES as _FIT_PARTICLES,
    LIKELIHOOD_MODES as _LIKELIHOODS,
    resolve_reconstruction_configuration,
)
from LicketyFit.run_console import (
    print_details,
    print_goodbye,
    print_preparation_notice,
    print_welcome,
    run_with_simple_console,
)


_PUBLIC_DRIVER_RELEASE = "2026-09-03-v1.45.1-adaptive-wcte-pid-fallback"
_RUNTIME_BOOTSTRAP_PROTOCOL = 3
_MULTIPROCESS_RUNTIME_ENV = {
    "BLIS_NUM_THREADS": "1",
    "EMITTER_PHOTON_SCATTER_NATIVE_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
    "NUMBA_THREADING_LAYER": "forksafe",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_DEDICATED_RECONSTRUCTION_ENV = {
    "FIT_MODE",
    "LF_SEEDING_MODE",
    "LF_INTERACTION_MODE",
    "LF_PUBLIC_FIT_MODE",
    "EMITTER_PRIMARY_MCS_MODEL",
    "MCS_COHERENT_IMPLEMENTATION",
    "EMITTER_COSMIC_MCS_CONTINUATION",
    "EMITTER_COSMIC_JOINT_INFERENCE_METHOD",
}


def _runtime_dependency_available(module_name: str) -> bool:
    """Return whether a required execution helper imports successfully."""
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError):
        return False
    if module_name == "threadpoolctl":
        return callable(getattr(module, "threadpool_limits", None))
    return True


def _available_cpu_count() -> int:
    """Return the CPUs this job is actually allowed to schedule on."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def _internal_threads_per_worker() -> int:
    """Give every event worker a bounded share of the process CPU affinity."""
    workers = max(1, int(NPROC))
    if workers > 1:
        return 1
    available_per_worker = max(1, _available_cpu_count() // workers)
    return max(
        1,
        min(int(MAX_INTERNAL_THREADS_PER_WORKER), available_per_worker),
    )


def _validate_runtime_dependencies() -> None:
    """Fail in the launcher, before any event workers are created."""
    if _runtime_dependency_available("threadpoolctl"):
        return
    requirements = _PROJECT_ROOT / "requirements.txt"
    raise RuntimeError(
        "The validated small-KL accelerator requires threadpoolctl, but it "
        "could not be imported. From the LicketyFit repository root run:\n  "
        f"{sys.executable} -m pip install -r {requirements}\n"
        "Then start this launcher again."
    )


def _resolved_reconstruction_configuration():
    return resolve_reconstruction_configuration(
        seeding_mode=SEEDING_MODE,
        interaction_mode=INTERACTION_MODE,
        likelihood_mode=LIKELIHOOD_MODE,
        enable_mcs=ENABLE_MCS,
        seed_mode=COSMIC_MULTILATERATION_SEED_MODE,
        primary_mcs_model=PRIMARY_MCS_MODEL,
        coherent_mcs_implementation=COHERENT_MCS_IMPLEMENTATION,
        cosmic_mcs_continuation=COSMIC_MCS_CONTINUATION,
        cosmic_joint_inference_method=COSMIC_JOINT_INFERENCE_METHOD,
    )


def _resolved_fixed_parameter_items() -> list[tuple[str, float | None]]:
    return list(
        resolve_fixed_parameter_environment(
            FIXED_PARAMETERS,
            seeding_mode=SEEDING_MODE,
            interaction_mode=INTERACTION_MODE,
            extra_driver_env=EXTRA_DRIVER_ENV,
        ).items()
    )


def _resolved_charge_configuration() -> tuple[
    str,
    str,
    float | None,
    str,
    str | None,
    str | None,
    bool | None,
    str | None,
]:
    """Resolve the one-switch shape-versus-absolute charge contract."""
    shape_mode = str(CHARGE_LIKELIHOOD).strip().lower().replace("-", "_")
    if shape_mode not in _CHARGE_LIKELIHOODS:
        raise ValueError(
            f"CHARGE_LIKELIHOOD must be one of {sorted(_CHARGE_LIKELIHOODS)}"
    )
    if bool(USE_ABSOLUTE_LIGHT_YIELD):
        source = str(ABSOLUTE_LIGHT_YIELD_SOURCE).strip().lower().replace(
            "-", "_"
        )
        if source not in {"wcsim_calibration", "mathematical"}:
            raise ValueError(
                "ABSOLUTE_LIGHT_YIELD_SOURCE must be 'wcsim_calibration' or "
                "'mathematical'"
            )
        manifest_value = str(
            MATHEMATICAL_CHARGE_CALIBRATION_MANIFEST
            if source == "mathematical"
            else GLOBAL_CHARGE_CALIBRATION_MANIFEST
        ).strip()
        if not manifest_value:
            raise ValueError(
                "USE_ABSOLUTE_LIGHT_YIELD=True requires "
                "GLOBAL_CHARGE_CALIBRATION_MANIFEST for the "
                "wcsim_calibration source or the packaged mathematical manifest"
            )
        scale, calibration_id, path, manifest_sha256, manifest = (
            resolve_manifest_calibration(
                manifest_value,
                expected_detector="wcsim",
                expected_particle=FIT_PARTICLE,
                project_root=_PROJECT_ROOT,
                manual_scale=GLOBAL_CHARGE_SCALE,
                manual_calibration_id=GLOBAL_CHARGE_CALIBRATION_ID,
            )
        )
        response = manifest.get("pmt_charge_response", {})
        if response.get("model") != str(SPE_RESPONSE_MODEL):
            raise ValueError(
                "WCSim absolute-light manifest PMT response does not match "
                f"SPE_RESPONSE_MODEL={SPE_RESPONSE_MODEL!r}"
            )
        basis = str(manifest.get("calibration_basis", "")).strip().lower()
        if source == "mathematical":
            if basis != "ground_up_mathematical":
                raise ValueError(
                    "the mathematical WCSim option requires a "
                    "ground_up_mathematical manifest"
                )
            ground_up = manifest.get("ground_up_light_model", {})
            if ground_up.get("include_wcsim_qe_boosts") is not True:
                raise ValueError(
                    "the WCSim mathematical manifest must include both "
                    "audited WCSim QE boosts"
                )
        elif basis == "ground_up_mathematical":
            raise ValueError(
                "the wcsim_calibration option cannot select a mathematical "
                "manifest"
            )
        model_contract = manifest.get("model_contract", {})
        reflection_in_charge = model_contract.get(
            "analytic_reflection_in_charge"
        )
        if not isinstance(reflection_in_charge, bool):
            raise ValueError(
                "WCSim absolute-light manifest must declare boolean "
                "model_contract.analytic_reflection_in_charge"
            )
        reflection_charge_policy = str(
            model_contract.get(
                "analytic_reflection_charge_policy", "unconditional"
            )
        ).strip().lower().replace("-", "_")
        if reflection_charge_policy not in {
            "unconditional",
            "prompt_group_gated",
        }:
            raise ValueError(
                "WCSim absolute-light manifest reflection charge policy must "
                "be unconditional or prompt_group_gated"
            )
        if reflection_charge_policy == "prompt_group_gated" and not reflection_in_charge:
            raise ValueError(
                "prompt_group_gated reflection requires "
                "analytic_reflection_in_charge=true"
            )
        return (
            "compound_spe_calibrated",
            "global_scale",
            scale,
            calibration_id,
            path,
            manifest_sha256,
            reflection_in_charge,
            reflection_charge_policy,
        )
    if shape_mode == "compound_spe_calibrated":
        raise ValueError(
            "Set USE_ABSOLUTE_LIGHT_YIELD=True to use compound_spe_calibrated"
        )
    return shape_mode, "event_mean", None, "", None, None, None, None


def _validate_coherent_numerics() -> None:
    if (
        not math.isfinite(float(LONGITUDINAL_GAUGE_MAX_ABS_T0_NS))
        or float(LONGITUDINAL_GAUGE_MAX_ABS_T0_NS) <= 0.0
    ):
        raise ValueError(
            "LONGITUDINAL_GAUGE_MAX_ABS_T0_NS must be finite and positive"
        )
    if COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY not in {
        "straight_conditioned", "free"
    }:
        raise ValueError(
            "COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY must be "
            "'straight_conditioned' or 'free'"
        )
    if int(COHERENT_TRACK_CYCLES) < 1:
        raise ValueError("COHERENT_TRACK_CYCLES must be positive")
    if int(COHERENT_ADAPTIVE_MAX_CYCLES) < int(COHERENT_TRACK_CYCLES):
        raise ValueError(
            "COHERENT_ADAPTIVE_MAX_CYCLES must be at least "
            "COHERENT_TRACK_CYCLES"
        )
    if int(COHERENT_ADAPTIVE_STOP_PATIENCE) < 1:
        raise ValueError("COHERENT_ADAPTIVE_STOP_PATIENCE must be positive")
    if int(COHERENT_BASIN_CANDIDATES) < 1:
        raise ValueError("COHERENT_BASIN_CANDIDATES must be positive")
    if int(COHERENT_JOINT_TIMING_RANGE_CYCLES) < 1:
        raise ValueError(
            "COHERENT_JOINT_TIMING_RANGE_CYCLES must be positive"
        )
    if int(COHERENT_JOINT_TIMING_GLOBAL_CYCLES) < 1:
        raise ValueError(
            "COHERENT_JOINT_TIMING_GLOBAL_CYCLES must be positive"
        )
    if int(COHERENT_JOINT_TIMING_LATENT_ITERATIONS) < 0:
        raise ValueError(
            "COHERENT_JOINT_TIMING_LATENT_ITERATIONS must be nonnegative"
        )
    if int(COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS) < 0:
        raise ValueError(
            "COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS must be "
            "nonnegative"
        )
    thresholds = {
        "COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL": (
            COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL
        ),
        "COHERENT_ADAPTIVE_STOP_GAIN_NLL": COHERENT_ADAPTIVE_STOP_GAIN_NLL,
        "COHERENT_BASIN_GATE_NLL": COHERENT_BASIN_GATE_NLL,
        "COHERENT_JOINT_TIMING_RANGE_STEP_MM": (
            COHERENT_JOINT_TIMING_RANGE_STEP_MM
        ),
        "COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM": (
            COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM
        ),
    }
    for name, value in thresholds.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if float(COHERENT_JOINT_TIMING_RANGE_STEP_MM) <= 0.0:
        raise ValueError(
            "COHERENT_JOINT_TIMING_RANGE_STEP_MM must be positive"
        )
    if not (
        0.0 < float(COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM)
        <= float(COHERENT_JOINT_TIMING_RANGE_STEP_MM)
    ):
        raise ValueError(
            "COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM must be positive and "
            "no larger than COHERENT_JOINT_TIMING_RANGE_STEP_MM"
        )


def _resolved_visible_range_manifest() -> str | None:
    if not bool(APPLY_WCSIM_VISIBLE_RANGE_CONVENTION):
        return None
    value = str(WCSIM_VISIBLE_RANGE_CONVENTION_MANIFEST).strip()
    if not value:
        raise ValueError(
            "APPLY_WCSIM_VISIBLE_RANGE_CONVENTION=True requires "
            "WCSIM_VISIBLE_RANGE_CONVENTION_MANIFEST"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path.resolve())


def _require_file(value: str, label: str) -> None:
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")


def _require_geometry_submodule() -> None:
    required = (
        _GEOMETRY_SUBMODULE / "Geometry" / "Device.py",
        _GEOMETRY_SUBMODULE / "Geometry" / "WCD.py",
        _GEOMETRY_SUBMODULE / "examples" / "wcte_bldg157.geo",
    )
    if not all(path.is_file() for path in required):
        raise ValueError(
            "The Geometry submodule is missing or uninitialized at "
            f"{_GEOMETRY_SUBMODULE}. From LicketyFit_official run: "
            "git submodule update --init Geometry"
        )


def _validate(*, check_paths: bool) -> None:
    if FIT_PARTICLE not in _FIT_PARTICLES:
        raise ValueError(f"FIT_PARTICLE must be one of {sorted(_FIT_PARTICLES)}")
    if LIKELIHOOD_MODE not in _LIKELIHOODS:
        raise ValueError(f"LIKELIHOOD_MODE must be one of {sorted(_LIKELIHOODS)}")
    _resolved_fixed_parameter_items()
    _resolved_reconstruction_configuration()
    _resolved_charge_configuration()
    if (
        bool(USE_ABSOLUTE_LIGHT_YIELD)
        and "EMITTER_REFLECTION_IN_CHARGE" in EXTRA_DRIVER_ENV
    ):
        raise ValueError(
            "EMITTER_REFLECTION_IN_CHARGE is fixed by the selected WCSim "
            "absolute-light calibration manifest; remove the override from "
            "EXTRA_DRIVER_ENV"
        )
    if (
        bool(USE_ABSOLUTE_LIGHT_YIELD)
        and "EMITTER_REFLECTION_CHARGE_POLICY" in EXTRA_DRIVER_ENV
    ):
        raise ValueError(
            "EMITTER_REFLECTION_CHARGE_POLICY is fixed by the selected WCSim "
            "absolute-light calibration manifest; remove the override from "
            "EXTRA_DRIVER_ENV"
        )
    _validate_coherent_numerics()
    range_manifest = _resolved_visible_range_manifest()
    if range_manifest is not None and FIT_PARTICLE != "muon":
        raise ValueError(
            "the supplied WCSim visible-range convention is calibrated for muons"
        )
    if int(N_EVENTS) < 1 or int(EVENT_START_INDEX) < 0 or int(NPROC) < 1:
        raise ValueError("N_EVENTS and NPROC must be positive; EVENT_START_INDEX must be nonnegative")
    if int(MAX_INTERNAL_THREADS_PER_WORKER) < 1:
        raise ValueError("MAX_INTERNAL_THREADS_PER_WORKER must be positive")
    child_stall_timeout = float(CHILD_STALL_TIMEOUT_SECONDS)
    result_stall_timeout = float(EVENT_RESULT_STALL_TIMEOUT_SECONDS)
    if not math.isfinite(child_stall_timeout) or child_stall_timeout < 0.0:
        raise ValueError(
            "CHILD_STALL_TIMEOUT_SECONDS must be finite and nonnegative"
        )
    if not math.isfinite(result_stall_timeout) or result_stall_timeout <= 0.0:
        raise ValueError(
            "EVENT_RESULT_STALL_TIMEOUT_SECONDS must be finite and positive"
        )
    if (
        child_stall_timeout > 0.0
        and result_stall_timeout > 0.0
        and result_stall_timeout >= child_stall_timeout
    ):
        raise ValueError(
            "EVENT_RESULT_STALL_TIMEOUT_SECONDS must be shorter than "
            "CHILD_STALL_TIMEOUT_SECONDS so the child can report unresolved events"
        )
    if bool(USE_WCTE_GEOMETRY) == bool(USE_IWCD_GEOMETRY):
        raise ValueError("Select exactly one of USE_WCTE_GEOMETRY and USE_IWCD_GEOMETRY")
    if USE_IWCD_GEOMETRY and not str(GEOMETRY_FILE).strip():
        raise ValueError(
            "USE_IWCD_GEOMETRY=True requires an explicit GEOMETRY_FILE. The "
            "Geometry submodule does not contain a validated serialized IWCD .geo file."
        )
    if (PROMPT_TIME_MIN_NS is None) != (PROMPT_TIME_MAX_NS is None):
        raise ValueError("Set both PROMPT_TIME_MIN_NS and PROMPT_TIME_MAX_NS, or neither")
    if PROMPT_TIME_MIN_NS is not None and float(PROMPT_TIME_MAX_NS) <= float(PROMPT_TIME_MIN_NS):
        raise ValueError("PROMPT_TIME_MAX_NS must exceed PROMPT_TIME_MIN_NS")
    if (
        not math.isfinite(float(PRIMARY_PROMPT_PEAK_SEARCH_MAX_NS))
        or float(PRIMARY_PROMPT_PEAK_SEARCH_MAX_NS) <= 5.0
    ):
        raise ValueError(
            "PRIMARY_PROMPT_PEAK_SEARCH_MAX_NS must be finite and exceed 5 ns"
        )
    if TRUTH_POSITION_OFFSET_MM is not None and len(tuple(TRUTH_POSITION_OFFSET_MM)) != 3:
        raise ValueError("TRUTH_POSITION_OFFSET_MM must be None or three numbers")
    if INACTIVE_SLOTS is not None:
        for value in INACTIVE_SLOTS:
            int(value)
    if "GEOMETRY_PATH" in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV["GEOMETRY_PATH"]).strip():
        raise ValueError(
            "GEOMETRY_PATH overrides are no longer supported; remove "
            "GEOMETRY_PATH from EXTRA_DRIVER_ENV and use the Geometry submodule"
        )
    forbidden_model_overrides = _REMOVED_MODEL_ENV_SWITCHES.intersection(
        map(str, EXTRA_DRIVER_ENV)
    )
    if forbidden_model_overrides:
        raise ValueError(
            "Removed reconstruction-model environment switches are not "
            "supported; use ENABLE_MCS, PRIMARY_MCS_MODEL, "
            "COHERENT_MCS_IMPLEMENTATION, and COSMIC_MCS_CONTINUATION at the "
            "top of this launcher, and remove "
            + ", ".join(sorted(forbidden_model_overrides))
            + " from EXTRA_DRIVER_ENV"
        )
    duplicate_reconstruction_overrides = (
        _DEDICATED_RECONSTRUCTION_ENV.intersection(map(str, EXTRA_DRIVER_ENV))
    )
    if duplicate_reconstruction_overrides:
        raise ValueError(
            "Use the dedicated reconstruction settings at the top of this file "
            "instead of duplicating them in EXTRA_DRIVER_ENV: "
            + ", ".join(sorted(duplicate_reconstruction_overrides))
        )
    if check_paths:
        _require_geometry_submodule()
        if not str(INPUT_FILE).strip():
            raise ValueError("Set INPUT_FILE to the WCSim NPZ file you want to fit")
        if USE_TRUTH_ROOT and not str(TRUTH_ROOT_FILE).strip():
            raise ValueError("USE_TRUTH_ROOT=True requires TRUTH_ROOT_FILE")
        _require_file(INPUT_FILE, "INPUT_FILE")
        if USE_TRUTH_ROOT:
            _require_file(TRUTH_ROOT_FILE, "TRUTH_ROOT_FILE")
        if str(GEOMETRY_FILE).strip():
            _require_file(GEOMETRY_FILE, "GEOMETRY_FILE")
        if str(WCSIM_WCTE_MAPPING_FILE).strip():
            _require_file(WCSIM_WCTE_MAPPING_FILE, "WCSIM_WCTE_MAPPING_FILE")
        if range_manifest is not None:
            _require_file(
                range_manifest, "WCSIM_VISIBLE_RANGE_CONVENTION_MANIFEST"
            )
        _require_file(str(_DRIVER), "batch_fit_driver.py")


def _encode(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value)
    return str(value)


def _configuration_items() -> list[tuple[str, object]]:
    _validate_coherent_numerics()
    reconstruction = _resolved_reconstruction_configuration()
    duplicate_reconstruction_overrides = (
        _DEDICATED_RECONSTRUCTION_ENV.intersection(map(str, EXTRA_DRIVER_ENV))
    )
    if duplicate_reconstruction_overrides:
        raise ValueError(
            "Use the dedicated reconstruction settings at the top of this file "
            "instead of duplicating them in EXTRA_DRIVER_ENV: "
            + ", ".join(sorted(duplicate_reconstruction_overrides))
        )
    if "GEOMETRY_PATH" in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV["GEOMETRY_PATH"]).strip():
        raise ValueError(
            "GEOMETRY_PATH overrides are no longer supported; remove "
            "GEOMETRY_PATH from EXTRA_DRIVER_ENV"
        )
    geometry_file = str(GEOMETRY_FILE).strip() or None
    mapping_file = str(WCSIM_WCTE_MAPPING_FILE).strip() or None
    output_file = str(OUTPUT_FILE).strip() or None
    inactive_slots = None if INACTIVE_SLOTS is None else tuple(INACTIVE_SLOTS)
    (
        charge_likelihood,
        normalization_mode,
        global_scale,
        calibration_id,
        calibration_manifest,
        calibration_manifest_sha256,
        calibrated_reflection_in_charge,
        calibrated_reflection_charge_policy,
    ) = _resolved_charge_configuration()
    range_manifest = _resolved_visible_range_manifest()
    config_file = Path(__file__).resolve()
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    items = [
        ("LF_DATA_SOURCE", "wcsim"), ("DATA_SOURCE", "wcsim"),
        # Remove an inherited v1.43 FIT_MODE; the authoritative two axes are
        # emitted by reconstruction.environment() below.
        ("FIT_MODE", None), ("WCSIM_INPUT_FILE", INPUT_FILE),
        ("TOT_EVENTS", N_EVENTS), ("LF_EVENT_START_INDEX", EVENT_START_INDEX),
        ("FIT_PARTICLE", FIT_PARTICLE), ("WCSIM_PARTICLE_LABEL", FIT_PARTICLE),
        ("LIKELIHOOD_MODE", LIKELIHOOD_MODE), ("NPROC", NPROC),
        ("EMITTER_ENABLE_DELTA_E", ENABLE_DELTA_ELECTRONS),
        ("EMITTER_ENABLE_MCS", ENABLE_MCS),
        ("EMITTER_ENABLE_REFLECTION", ENABLE_REFLECTION),
        ("EMITTER_ENABLE_RAYLEIGH", ENABLE_PHOTON_SCATTERING),
        (
            "COSMIC_MULTILATERATION_SEED_MODE",
            reconstruction.navigation_mode,
        ),
        (
            "WCTE_INCLUDE_ORIENTATION_GUARD",
            True if reconstruction.seeding_mode == "general"
            else BEAM_USE_GLOBAL_SEED_GUARD,
        ),
        (
            "WCTE_INCLUDE_DETECTOR_GLOBAL",
            True if reconstruction.seeding_mode == "general"
            else BEAM_USE_GLOBAL_SEED_GUARD,
        ),
        *reconstruction.environment().items(),
        ("PMT_CHARGE_LIKELIHOOD", charge_likelihood),
        ("PMT_SPE_RESPONSE_MODEL", SPE_RESPONSE_MODEL),
        ("PMT_COMPOUND_PROFILE_MAX_ITER", COMPOUND_PROFILE_MAX_ITER),
        ("PMT_COMPOUND_PROFILE_TOL", COMPOUND_PROFILE_TOL),
        ("PMT_COMPOUND_PROFILE_N_CAP", COMPOUND_PROFILE_N_CAP),
        ("WCSIM_USE_ABSOLUTE_LIGHT_YIELD", USE_ABSOLUTE_LIGHT_YIELD),
        (
            "WCSIM_ABSOLUTE_LIGHT_YIELD_SOURCE",
            ABSOLUTE_LIGHT_YIELD_SOURCE if USE_ABSOLUTE_LIGHT_YIELD else "shape_only",
        ),
        ("WCSIM_CHARGE_NORMALIZATION_MODE", normalization_mode),
        ("WCSIM_GLOBAL_CHARGE_SCALE", global_scale),
        ("WCSIM_GLOBAL_CHARGE_CALIBRATION_ID", calibration_id),
        ("WCSIM_GLOBAL_CHARGE_CALIBRATION_MANIFEST", calibration_manifest),
        (
            "WCSIM_GLOBAL_CHARGE_CALIBRATION_MANIFEST_SHA256",
            calibration_manifest_sha256,
        ),
        (
            "EMITTER_REFLECTION_IN_CHARGE",
            calibrated_reflection_in_charge,
        ),
        (
            "EMITTER_REFLECTION_CHARGE_POLICY",
            calibrated_reflection_charge_policy,
        ),
        ("WCSIM_VISIBLE_RANGE_CONVENTION_MANIFEST", range_manifest),
        ("WCSIM_LONGITUDINAL_GAUGE_ENABLED", LONGITUDINAL_GAUGE_ENABLED),
        (
            "WCSIM_LONGITUDINAL_GAUGE_MAX_ABS_T0_NS",
            LONGITUDINAL_GAUGE_MAX_ABS_T0_NS,
        ),
        ("COSMIC_COHERENT_TRACK_CYCLES", COHERENT_TRACK_CYCLES),
        (
            "COSMIC_COHERENT_ADAPTIVE_MAX_CYCLES",
            COHERENT_ADAPTIVE_MAX_CYCLES,
        ),
        (
            "COSMIC_COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL",
            COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL,
        ),
        (
            "COSMIC_COHERENT_ADAPTIVE_STOP_GAIN_NLL",
            COHERENT_ADAPTIVE_STOP_GAIN_NLL,
        ),
        (
            "COSMIC_COHERENT_ADAPTIVE_STOP_PATIENCE",
            COHERENT_ADAPTIVE_STOP_PATIENCE,
        ),
        ("COSMIC_COHERENT_BASIN_CANDIDATES", COHERENT_BASIN_CANDIDATES),
        ("COSMIC_COHERENT_BASIN_GATE_NLL", COHERENT_BASIN_GATE_NLL),
        (
            "COSMIC_COHERENT_JOINT_TIMING_RANGE_STEP_MM",
            COHERENT_JOINT_TIMING_RANGE_STEP_MM,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM",
            COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_RANGE_CYCLES",
            COHERENT_JOINT_TIMING_RANGE_CYCLES,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_GLOBAL_CYCLES",
            COHERENT_JOINT_TIMING_GLOBAL_CYCLES,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY",
            COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_LATENT_ITERATIONS",
            COHERENT_JOINT_TIMING_LATENT_ITERATIONS,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS",
            COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS,
        ),
        ("LF_OUTPUT_FILE", output_file),
        ("LF_WCTE", USE_WCTE_GEOMETRY), ("LF_IWCD", USE_IWCD_GEOMETRY),
        ("WCD_GEOMETRY_FILE", geometry_file), ("WCTE_GEOMETRY_FILE", geometry_file),
        ("GEOMETRY_PATH", None),
        ("LF_GEOMETRY_POLICY", "required_repository_submodule"),
        ("ENERGY_TRUE", ENERGY_LABEL_MEV),
        ("WCSIM_USE_TRUTH_ROOT", USE_TRUTH_ROOT),
        ("WCSIM_TRUTH_ROOT_FILE", TRUTH_ROOT_FILE),
        ("WCSIM_TRUTH_TREE", TRUTH_TREE),
        ("WCSIM_TRUTH_EVENT_ID_OFFSET", TRUTH_EVENT_ID_OFFSET),
        ("WCSIM_TRUTH_POSITION_OFFSET_MM", TRUTH_POSITION_OFFSET_MM),
        ("WCSIM_TRUTH_PRIMARY_TRACK_ID", TRUTH_PRIMARY_TRACK_ID),
        ("WCSIM_TRUTH_UPROOT_STEP_SIZE", TRUTH_UPROOT_STEP_SIZE),
        ("WCSIM_TRUTH_EVENT_INDEX_STEP_SIZE", TRUTH_EVENT_INDEX_STEP_SIZE),
        ("WCSIM_TRUTH_IO_WORKERS", TRUTH_IO_WORKERS),
        ("WCSIM_TRUTH_USE_EVENT_INDEX_CACHE", TRUTH_USE_EVENT_INDEX_CACHE),
        ("WCSIM_TRUTH_INDEX_CACHE_DIR", TRUTH_INDEX_CACHE_DIR),
        ("WCSIM_TRUTH_INCLUDE_OPTIONAL_DETAILS", TRUTH_INCLUDE_OPTIONAL_DETAILS),
        ("LF_WCSIM_TRUTH_USE_SOURCE", "run_wcsim.py"),
        ("LF_WCSIM_TRUTH_ROOT_FILE_SOURCE", "run_wcsim.py"),
        ("WCSIM_PROMPT_TIME_MIN_NS", PROMPT_TIME_MIN_NS),
        ("WCSIM_PROMPT_TIME_MAX_NS", PROMPT_TIME_MAX_NS),
        (
            "WCSIM_PRIMARY_PROMPT_PEAK_SEARCH_MAX_NS",
            PRIMARY_PROMPT_PEAK_SEARCH_MAX_NS,
        ),
        ("WCSIM_PMT_ID_MODE", PMT_ID_MODE), ("WCSIM_PMT_ID_OFFSET", PMT_ID_OFFSET),
        ("WCSIM_WCTE_MAPPING_PATH", mapping_file),
        ("INACTIVE_SLOTS", inactive_slots),
        ("N_EVENTS_PER_BATCH", N_EVENTS_PER_BATCH),
        ("WARM_FIT_KERNELS", WARM_FIT_KERNELS),
        (
            "LF_RUNTIME_CACHE_DIR",
            str(RUNTIME_CACHE_DIR).strip() or None,
        ),
        (
            "LF_NODE_LOCAL_RUNTIME_CACHE_DIR",
            str(NODE_LOCAL_RUNTIME_CACHE_DIR).strip() or None,
        ),
        ("LF_COSMIC_CHILD_HEARTBEAT_SECONDS", CHILD_HEARTBEAT_SECONDS),
        ("LF_COSMIC_CHILD_STALL_TIMEOUT_SECONDS", CHILD_STALL_TIMEOUT_SECONDS),
        (
            "LF_EVENT_RESULT_STALL_TIMEOUT_SECONDS",
            EVENT_RESULT_STALL_TIMEOUT_SECONDS,
        ),
        ("SAVE_AFTER_EACH_BATCH", SAVE_AFTER_EACH_BATCH),
        ("SAVE_DETAILED_EVENT_RESULTS", SAVE_DETAILED_EVENT_RESULTS),
        ("MCS_RECORD_EVENT_FAILURES", CONTINUE_AFTER_EVENT_FAILURE),
        ("MCS_RETAIN_STRAIGHT_ON_FAILURE", RETAIN_STRAIGHT_ON_MCS_FAILURE),
        ("PRINT_EVENT_RESULTS", PRINT_EVENT_RESULTS),
        ("PRINT_BATCH_PROGRESS", PRINT_BATCH_PROGRESS),
        ("PRINT_LIVE_EVENT_PROGRESS", PRINT_LIVE_EVENT_PROGRESS),
        ("PRINT_CHECKPOINT_MESSAGES", PRINT_CHECKPOINT_MESSAGES),
        ("VERBOSE_SETUP", VERBOSE_SETUP),
        ("LF_RUN_CONFIG_KIND", "wcsim"),
        ("LF_RUN_CONFIG_FILE", str(config_file)),
        ("LF_RUN_CONFIG_SHA256", config_hash),
        ("LF_PUBLIC_DRIVER_RELEASE", _PUBLIC_DRIVER_RELEASE),
    ]
    if ENABLE_EXACT_EXECUTION_ACCELERATORS:
        items.extend(
            (str(name), value)
            for name, value in EXACT_EXECUTION_ENV.items()
        )
    if ENABLE_VALIDATED_COSMIC_LATENCY_PROFILE:
        items.extend(
            (str(name), value)
            for name, value in VALIDATED_COSMIC_LATENCY_ENV.items()
        )
    internal_threads = _internal_threads_per_worker()
    items.extend(
        (
            ("NUMBA_NUM_THREADS", internal_threads),
            ("EMITTER_PHOTON_SCATTER_NATIVE_THREADS", internal_threads),
            ("LF_INTERNAL_THREADS_PER_WORKER", internal_threads),
        )
    )
    items.extend(_resolved_fixed_parameter_items())
    items.extend((str(name), value) for name, value in EXTRA_DRIVER_ENV.items())
    return items


def build_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return the exact environment handed to batch_fit_driver.py."""
    env = dict(os.environ if base is None else base)
    for internal in (
        "LF_COSMIC_SUPERVISED_CHILD", "LF_COSMIC_CHILD_QUIET",
        "LF_RUNTIME_BOOTSTRAP_READY", "LF_RUNTIME_BOOTSTRAP_SIGNATURE",
        "LF_RUNTIME_BOOTSTRAP_VERIFIED", "LF_RUNTIME_CACHE_STAGED",
        "LF_RUNTIME_CACHE_STAGE_DIR", "LF_RUNTIME_CANONICAL_CACHE_DIR",
        "LF_RUNTIME_BOOTSTRAP_WORKER_READY_DIR",
        "LF_EVENT_COUNT", "LF_WCTE_INTERNAL_PREPARED_EVENT_FILE",
        "LF_WCTE_PREPARE_EVENTS_ONLY", "LF_WCTE_PREPARED_EVENT_FILE",
    ):
        env.pop(internal, None)
    for name in _REMOVED_MODEL_ENV_SWITCHES:
        env.pop(name, None)
    for name, value in _configuration_items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = _encode(value)
    nested = str(env.get("ALLOW_NESTED_PARALLELISM", "0")).strip().lower() in {
        "1", "true", "yes", "y", "on",
    }
    if int(NPROC) > 1 and not nested:
        # These must be present in the environment handed to execve.  NumPy
        # loads OpenBLAS while importing the batch driver, before the driver's
        # own runtime setup function can run.  Setting the values there is too
        # late: the parent and every forked worker otherwise retain the
        # host-sized BLAS thread pool.
        env.update(_MULTIPROCESS_RUNTIME_ENV)
    if _RUNTIME_DEPS.is_dir():
        inherited = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_RUNTIME_DEPS)] + ([inherited] if inherited else [])
        )
    return env


_BOOTSTRAP_SIGNATURE_IGNORES = {
    "LF_OUTPUT_FILE",
    "LF_EVENT_START_INDEX",
    "LF_RUN_CONFIG_FILE",
    "LF_RUN_CONFIG_SHA256",
    "N_EVENTS_PER_BATCH",
    "LF_COSMIC_CHILD_HEARTBEAT_SECONDS",
    "LF_COSMIC_CHILD_STALL_TIMEOUT_SECONDS",
    "LF_EVENT_RESULT_STALL_TIMEOUT_SECONDS",
    "LF_NODE_LOCAL_RUNTIME_CACHE_DIR",
    "LF_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS",
    "LF_RUNTIME_BOOTSTRAP_WORKER_READY_DIR",
    "PRINT_BATCH_PROGRESS",
    "PRINT_CHECKPOINT_MESSAGES",
    "PRINT_EVENT_RESULTS",
    "PRINT_LIVE_EVENT_PROGRESS",
    "SAVE_AFTER_EACH_BATCH",
    "SAVE_DETAILED_EVENT_RESULTS",
    "TOT_EVENTS",
    "VERBOSE_SETUP",
    "WCSIM_INPUT_FILE",
    "WCSIM_TRUTH_ROOT_FILE",
}


def _runtime_bootstrap_signature(environment: dict[str, str]) -> str:
    """Identify a compiled execution contract, independent of input/output."""
    configured_names = {
        str(name) for name, _value in _configuration_items()
    }
    configured_names.update(_MULTIPROCESS_RUNTIME_ENV)
    contract = {
        name: environment.get(name)
        for name in sorted(configured_names - _BOOTSTRAP_SIGNATURE_IGNORES)
    }
    payload = {
        # Numba's cache locator includes the absolute source location. A moved
        # extraction must therefore bootstrap again even when sources match.
        "project_root": str(_PROJECT_ROOT.resolve()),
        "python": str(Path(sys.executable).resolve()),
        "cpu": _runtime_cpu_identity(),
        "release": _PUBLIC_DRIVER_RELEASE,
        "contract": contract,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _runtime_cpu_identity() -> dict[str, object]:
    """Return the model/features that govern native and Numba codegen."""
    identity: dict[str, object] = {
        "machine": platform.machine().strip().lower(),
        "processor": platform.processor().strip().lower(),
    }
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        )
        first = cpuinfo.split("\n\n", 1)[0]
        selected: dict[str, str] = {}
        for raw in first.splitlines():
            if ":" not in raw:
                continue
            key, value = raw.split(":", 1)
            normalized_key = key.strip().lower()
            if normalized_key in {
                "vendor_id", "cpu family", "model", "model name",
                "stepping", "flags", "features",
            }:
                normalized_value = " ".join(value.strip().lower().split())
                if normalized_key in {"flags", "features"}:
                    normalized_value = " ".join(sorted(normalized_value.split()))
                selected[normalized_key] = normalized_value
        identity["cpuinfo"] = selected
    except OSError:
        identity["cpuinfo"] = {}
    return identity


def _bootstrap_marker_is_valid(
    marker: Path,
    cache_root: Path,
    *,
    expected_signature: str | None = None,
) -> bool:
    """Require every content-hashed cache artifact recorded by bootstrap."""
    try:
        if marker.is_symlink():
            return False
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if expected_signature is not None:
            if payload.get("verified") is not True:
                return False
            if int(payload.get("protocol", -1)) != _RUNTIME_BOOTSTRAP_PROTOCOL:
                return False
            if str(payload.get("release", "")) != _PUBLIC_DRIVER_RELEASE:
                return False
            if str(payload.get("signature", "")) != expected_signature:
                return False
            runtime_contract = payload.get("runtime_contract")
            phases = payload.get("bootstrap_phases")
            if not isinstance(runtime_contract, dict) or not isinstance(phases, list):
                return False
            contract_nproc = max(1, int(runtime_contract.get("NPROC", "1")))
            serial_verified = any(
                isinstance(row, dict)
                and row.get("phase") == "serial-cache-build"
                and int(row.get("nproc", 0)) == 1
                and int(row.get("events_completed", 0)) == 1
                and row.get("forced_full_worker_pool") is False
                for row in phases
            )
            if not serial_verified:
                return False
            if contract_nproc > 1:
                fork_verified = any(
                    isinstance(row, dict)
                    and row.get("phase") == "production-fork-check"
                    and int(row.get("nproc", 0)) == contract_nproc
                    and int(row.get("events_completed", 0)) == 1
                    and row.get("forced_full_worker_pool") is True
                    and int(row.get("workers_initialized_observed", 0))
                    >= contract_nproc
                    for row in phases
                )
                if not fork_verified:
                    return False
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, dict) or not artifacts:
            return False
        for relative, expected in artifacts.items():
            if (
                expected_signature is not None
                and _runtime_cache_payload_path(relative) is None
            ):
                return False
            pure = PurePosixPath(str(relative))
            if (
                pure.is_absolute()
                or not pure.parts
                or any(part in {"", ".", ".."} for part in pure.parts)
            ):
                return False
            path = cache_root.joinpath(*pure.parts)
            current = cache_root
            for part in pure.parts:
                current = current / part
                if current.is_symlink():
                    return False
            if not path.is_file():
                return False
            if isinstance(expected, dict):
                expected_size = int(expected["size"])
                expected_sha256 = str(expected["sha256"])
                if len(expected_sha256) != 64:
                    return False
            elif expected_signature is None:
                # Size-only records are accepted solely by the legacy helper
                # form used without a production signature. Protocol-v3 cache
                # activation always requires a SHA-256 record.
                expected_size = int(expected)
                expected_sha256 = ""
            else:
                return False
            if path.stat().st_size != expected_size:
                return False
            if expected_sha256:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected_sha256:
                    return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


_RUNTIME_CACHE_PAYLOAD_SUFFIXES = {
    "numba": frozenset({".nbc", ".nbi"}),
    "native": frozenset({".so", ".dylib", ".dll", ".pyd"}),
    "reflection": frozenset({".npz"}),
    "response": frozenset({".npz"}),
}

_RUNTIME_RESPONSE_PAYLOAD_NAME = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_-]*\.npz\Z"
)
_RUNTIME_RESPONSE_LOCK_NAME = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_-]*\.lock\Z"
)


def _runtime_cache_payload_path(relative: object) -> PurePosixPath | None:
    """Return one canonical, allowlisted runtime-payload path."""
    text = str(relative)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.as_posix() != text
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(part.startswith(".") for part in pure.parts)
        or len(pure.parts) < 2
    ):
        return None
    allowed_suffixes = _RUNTIME_CACHE_PAYLOAD_SUFFIXES.get(pure.parts[0])
    if allowed_suffixes is None:
        return None
    if pure.parts[0] == "response" and (
        len(pure.parts) != 2
        or _RUNTIME_RESPONSE_PAYLOAD_NAME.fullmatch(pure.name) is None
    ):
        return None
    name = pure.name
    if ".tmp." in name or name.endswith(".tmp"):
        return None
    if pure.suffix.lower() not in allowed_suffixes:
        return None
    return pure


def _cache_artifact_manifest(cache_root: Path) -> dict[str, dict[str, object]]:
    artifacts: dict[str, dict[str, object]] = {}
    for directory_name in _RUNTIME_CACHE_PAYLOAD_SUFFIXES:
        directory = cache_root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(cache_root).as_posix()
            if _runtime_cache_payload_path(relative) is None:
                # Locks and interrupted writer temporaries are deliberately
                # excluded. They are mutable coordination state, not payload.
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            artifacts[relative] = {
                "size": int(path.stat().st_size),
                "sha256": digest.hexdigest(),
            }
    return artifacts


def _manifest_runtime_payload_paths(marker: Path) -> tuple[PurePosixPath, ...]:
    """Load and validate the allowlisted payload paths recorded by a marker."""
    payload = json.loads(marker.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("runtime-cache marker has no artifact manifest")
    paths = []
    for relative in sorted(artifacts):
        pure = _runtime_cache_payload_path(relative)
        if pure is None:
            raise ValueError(f"disallowed runtime-cache payload path: {relative!r}")
        paths.append(pure)
    return tuple(paths)


def _staged_cache_contains_only_manifest_payloads(
    stage_root: Path,
    manifest_paths: tuple[PurePosixPath, ...],
) -> bool:
    """Reject a staged cache containing unverified payload or transient data."""
    if any(
        _runtime_cache_payload_path(path.as_posix()) != path
        for path in manifest_paths
    ):
        return False
    expected = {path.as_posix() for path in manifest_paths}
    observed: set[str] = set()
    try:
        if stage_root.is_symlink() or not stage_root.is_dir():
            return False
        for path in stage_root.rglob("*"):
            if path.is_symlink():
                return False
            relative = path.relative_to(stage_root).as_posix()
            pure = PurePosixPath(relative)
            if not pure.parts or pure.parts[0] not in _RUNTIME_CACHE_PAYLOAD_SUFFIXES:
                return False
            if path.is_dir():
                # The response cache has one deliberately flat namespace. This
                # prevents a marker or stale stage from broadening its payload
                # boundary through a nested directory tree.
                if pure.parts[0] == "response" and len(pure.parts) != 1:
                    return False
                continue
            if not path.is_file():
                return False
            if relative in expected:
                observed.add(relative)
                continue
            # Runtime loaders may create lock files after activation. They are
            # inert coordination state; no other unmanifested file is allowed.
            if pure.name.endswith(".lock"):
                if pure.parts[0] == "response" and (
                    len(pure.parts) != 2
                    or _RUNTIME_RESPONSE_LOCK_NAME.fullmatch(pure.name) is None
                    or pure.with_suffix(".npz").as_posix() not in expected
                ):
                    return False
                continue
            return False
    except OSError:
        return False
    return observed == expected


def _runtime_contract(environment: dict[str, str]) -> dict[str, str | None]:
    """Return the process/threading contract certified by the bootstrap."""
    names = {"NPROC", *_MULTIPROCESS_RUNTIME_ENV}
    return {name: environment.get(name) for name in sorted(names)}


def _stage_manifest_key(marker: Path) -> str:
    """Identify one immutable set of verified compiled cache payloads."""
    payload = json.loads(marker.read_text(encoding="utf-8"))
    encoded = json.dumps(
        payload["artifacts"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _safe_node_local_stage_base(
    canonical_root: Path,
    environment: dict[str, str],
) -> Path | None:
    """Choose a private, distinct temporary filesystem for cache staging."""
    explicit = str(
        environment.get("LF_NODE_LOCAL_RUNTIME_CACHE_DIR", "")
    ).strip()
    temporary = explicit or str(environment.get("TMPDIR", "")).strip()
    base = Path(temporary or tempfile.gettempdir()).expanduser().resolve()
    try:
        base.mkdir(parents=True, exist_ok=True)
        canonical = canonical_root.resolve()
        if (
            base == canonical
            or base in canonical.parents
            or canonical in base.parents
        ):
            return None
        # Copying between two directories on the same filesystem supplies no
        # AFS/NFS isolation and only wastes startup I/O.  An explicit staging
        # directory is an expert assertion that staging is still intentional.
        if not explicit and base.stat().st_dev == canonical.stat().st_dev:
            return None
        uid = int(os.getuid())
        private = base / f"licketyfit-{uid}"
        private.mkdir(mode=0o700, parents=False, exist_ok=True)
        private_stat = private.lstat()
        if (
            stat.S_ISLNK(private_stat.st_mode)
            or int(private_stat.st_uid) != uid
        ):
            return None
        if stat.S_IMODE(private_stat.st_mode) != 0o700:
            private.chmod(0o700)
            private_stat = private.lstat()
            if stat.S_IMODE(private_stat.st_mode) != 0o700:
                return None
        stage_base = private / "runtime-stage"
        stage_base.mkdir(mode=0o700, parents=False, exist_ok=True)
        stage_stat = stage_base.lstat()
        if (
            stat.S_ISLNK(stage_stat.st_mode)
            or int(stage_stat.st_uid) != uid
        ):
            return None
        if stat.S_IMODE(stage_stat.st_mode) != 0o700:
            stage_base.chmod(0o700)
            stage_stat = stage_base.lstat()
            if stat.S_IMODE(stage_stat.st_mode) != 0o700:
                return None
        probe = stage_base / f".stage-probe-{os.getpid()}"
        probe.touch(mode=0o600, exist_ok=False)
        probe.unlink()
    except OSError:
        return None
    return stage_base


def _stage_verified_runtime_cache(
    *,
    canonical_root: Path,
    marker: Path,
    signature: str,
    environment: dict[str, str],
) -> Path | None:
    """Copy verified runtime payloads off network storage before forking."""
    stage_base = _safe_node_local_stage_base(canonical_root, environment)
    if stage_base is None:
        return None
    try:
        if not _bootstrap_marker_is_valid(
            marker, canonical_root, expected_signature=signature
        ):
            return None
        manifest_paths = _manifest_runtime_payload_paths(marker)
        stage_base.mkdir(parents=True, exist_ok=True)
        stage_name = (
            f"{canonical_root.name}-{signature}-{_stage_manifest_key(marker)}"
        )
        stage_root = stage_base / stage_name
        lock_path = stage_base / f"{stage_name}.lock"
        import fcntl

        with lock_path.open("a+", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            if (
                _bootstrap_marker_is_valid(
                    marker, stage_root, expected_signature=signature
                )
                and _staged_cache_contains_only_manifest_payloads(
                    stage_root, manifest_paths
                )
            ):
                return stage_root

            temporary = stage_base / f".{stage_name}.tmp-{os.getpid()}"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            for relative in manifest_paths:
                source = canonical_root.joinpath(*relative.parts)
                current = canonical_root
                for part in relative.parts:
                    current = current / part
                    if current.is_symlink():
                        raise ValueError(
                            f"symlink in runtime-cache payload path: {relative}"
                        )
                if not stat.S_ISREG(source.lstat().st_mode):
                    raise ValueError(
                        f"runtime-cache payload is not a regular file: {relative}"
                    )
                destination = temporary.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination, follow_symlinks=False)
            if (
                not _bootstrap_marker_is_valid(
                    marker, temporary, expected_signature=signature
                )
                or not _staged_cache_contains_only_manifest_payloads(
                    temporary, manifest_paths
                )
            ):
                shutil.rmtree(temporary, ignore_errors=True)
                return None
            if stage_root.exists():
                # A corrupt or interrupted cache staging area is never used.
                # Keep the freshly verified process-unique directory instead of
                # deleting a path another job may still have open.
                return temporary
            os.replace(temporary, stage_root)
            return stage_root
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _activate_verified_runtime_cache(
    *,
    location,
    marker: Path,
    signature: str,
    environment: dict[str, str],
) -> None:
    """Expose only verified canonical or safely staged caches to production."""
    simple_console = environment.get("LF_SIMPLE_CONSOLE", "0") == "1"
    environment["LF_RUNTIME_BOOTSTRAP_VERIFIED"] = "1"
    environment["LF_RUNTIME_BOOTSTRAP_READY"] = "1"
    environment["LF_RUNTIME_BOOTSTRAP_SIGNATURE"] = signature
    environment["LF_RUNTIME_CANONICAL_CACHE_DIR"] = str(location.root)
    stage_root = None
    if location.persistent:
        stage_root = _stage_verified_runtime_cache(
            canonical_root=location.root,
            marker=marker,
            signature=signature,
            environment=environment,
        )
    active_root = location.root if stage_root is None else stage_root
    environment["NUMBA_CACHE_DIR"] = str(active_root / "numba")
    environment["LF_NATIVE_CACHE_DIR"] = str(active_root / "native")
    environment["LF_RESOLVED_RUNTIME_CACHE_DIR"] = str(active_root)
    environment["LF_RUNTIME_CACHE_STAGED"] = "1" if stage_root is not None else "0"
    if stage_root is not None:
        environment["LF_RUNTIME_CACHE_STAGE_DIR"] = str(stage_root)
        environment["LF_RUNTIME_CACHE_PERSISTENT"] = "0"
        environment["LF_RUNTIME_CACHE_LOCATION_SOURCE"] = "temporary-mirror"
        if not simple_console:
            print(f"Production runtime cache staged locally: {stage_root}", flush=True)
    else:
        environment.pop("LF_RUNTIME_CACHE_STAGE_DIR", None)
        environment["LF_RUNTIME_CACHE_PERSISTENT"] = (
            "1" if location.persistent else "0"
        )
        environment["LF_RUNTIME_CACHE_LOCATION_SOURCE"] = str(location.source)
        if location.persistent and not simple_console:
            print(
                "Runtime cache local staging was not used; production workers "
                f"will read the canonical cache at {location.root}. If this is "
                "AFS/NFS, set NODE_LOCAL_RUNTIME_CACHE_DIR in this launcher "
                "to writable batch scratch and rerun (the launcher maps it "
                "to LF_NODE_LOCAL_RUNTIME_CACHE_DIR for the child).",
                flush=True,
            )


def _terminate_bootstrap_process_group(
    process: subprocess.Popen,
    *,
    grace_seconds: float = 5.0,
) -> None:
    """Stop a failed bootstrap process and every fork worker it created."""
    if os.name == "posix":
        # ``start_new_session=True`` makes the child PID its process-group ID.
        # Fork workers can outlive a failed leader, so always sweep the group
        # on a failure even when ``poll()`` has already reaped that leader.
        process_group = int(process.pid)
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return

        deadline = time.monotonic() + max(0.0, float(grace_seconds))
        while True:
            process.poll()
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait()
        return

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0.1, float(grace_seconds)))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        return
    process.wait()


def _bootstrap_log_tail(log_path: Path, *, lines: int = 30) -> str:
    try:
        return "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                -max(1, int(lines)):
            ]
        )
    except OSError:
        return "<bootstrap log unavailable>"


def _run_runtime_bootstrap_phase(
    child_environment: dict[str, str],
    *,
    phase_label: str,
    log_path: Path,
    timeout_seconds: float,
) -> float:
    """Run one bounded bootstrap phase with visible compile heartbeats."""
    simple_console = child_environment.get("LF_SIMPLE_CONSOLE", "0") == "1"
    started = time.monotonic()
    next_notice = 30.0
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            [sys.executable, str(_DRIVER_ENTRY)],
            cwd=str(_PROJECT_ROOT),
            env=child_environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if timeout_seconds > 0.0 and elapsed >= timeout_seconds:
                    raise RuntimeError(
                        f"Disposable runtime bootstrap {phase_label} exceeded "
                        f"{timeout_seconds:.0f} s and was stopped with its worker "
                        f"process group. Log: {log_path}\n"
                        f"{_bootstrap_log_tail(log_path)}"
                    )
                if not simple_console and elapsed >= next_notice:
                    print(
                        f"  runtime bootstrap {phase_label} still running "
                        f"({elapsed:.0f} s)",
                        flush=True,
                    )
                    next_notice += 30.0
                time.sleep(1.0)
        except BaseException:
            # The bootstrap child starts a new session, so terminal Ctrl-C is
            # delivered only to this launcher. Always sweep the detached child
            # and any pool workers before propagating an interrupt or failure.
            _terminate_bootstrap_process_group(process)
            raise
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        _terminate_bootstrap_process_group(process)
        raise RuntimeError(
            f"Disposable runtime bootstrap {phase_label} failed with exit "
            f"code {process.returncode}. Log: {log_path}\n"
            f"{_bootstrap_log_tail(log_path)}"
        )
    return elapsed


def _ensure_runtime_cache_ready(environment: dict[str, str]) -> None:
    """Compile cold kernels in a disposable process before production starts."""
    simple_console = environment.get("LF_SIMPLE_CONSOLE", "0") == "1"
    location = resolve_runtime_cache_location(
        _PROJECT_ROOT,
        environ=environment,
        create=True,
    )
    environment["LF_RESOLVED_RUNTIME_CACHE_DIR"] = str(location.root)
    # Ignore arbitrary inherited cache paths.  Bootstrap and validation must
    # refer to the same source-guarded canonical root.
    environment["NUMBA_CACHE_DIR"] = str(location.root / "numba")
    environment["LF_NATIVE_CACHE_DIR"] = str(location.root / "native")
    for internal_name in (
        "LF_RUNTIME_BOOTSTRAP_CHILD",
        "LF_RUNTIME_BOOTSTRAP_FORCE_ALL_WORKERS",
        "LF_RUNTIME_BOOTSTRAP_PHASE",
        "LF_RUNTIME_BOOTSTRAP_WORKER_READY_DIR",
    ):
        environment.pop(internal_name, None)
    environment.pop("LF_RUNTIME_BOOTSTRAP_VERIFIED", None)
    environment.pop("LF_RUNTIME_BOOTSTRAP_READY", None)
    environment.pop("LF_RUNTIME_BOOTSTRAP_SIGNATURE", None)
    environment.pop("LF_RUNTIME_CACHE_STAGED", None)
    environment.pop("LF_RUNTIME_CACHE_STAGE_DIR", None)
    environment["LF_RUNTIME_CACHE_PERSISTENT"] = (
        "1" if location.persistent else "0"
    )
    environment["LF_RUNTIME_CACHE_LOCATION_SOURCE"] = str(location.source)
    persistence = "persistent" if location.persistent else "node-local"
    if not simple_console:
        print(
            f"Runtime cache: {location.root} ({persistence}, {location.source})",
            flush=True,
        )
    if not location.persistent and not simple_console:
        print(
            "WARNING: no persistent user cache was writable; this job may need "
            "to compile again on another batch node.",
            flush=True,
        )
    if not bool(AUTO_BOOTSTRAP_RUNTIME_CACHE):
        if not simple_console:
            print(
                "Runtime cache bootstrap disabled; the first production event may "
                "pay compilation cost.",
                flush=True,
            )
        return

    import fcntl

    signature = _runtime_bootstrap_signature(environment)
    bootstrap_dir = location.root / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    marker = bootstrap_dir / f"ready-{signature}.json"
    lock_path = bootstrap_dir / f"ready-{signature}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        wait_start = time.monotonic()
        next_notice = 30.0
        while True:
            try:
                fcntl.flock(
                    lock_stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                break
            except BlockingIOError:
                elapsed = time.monotonic() - wait_start
                if not simple_console and elapsed >= next_notice:
                    print(
                        "Waiting for another LicketyFit process to finish the "
                        f"same runtime bootstrap ({elapsed:.0f} s elapsed).",
                        flush=True,
                    )
                    next_notice += 30.0
                time.sleep(1.0)

        if _bootstrap_marker_is_valid(
            marker, location.root, expected_signature=signature
        ):
            if not simple_console:
                print("Runtime cache is ready; starting production fit.", flush=True)
            _activate_verified_runtime_cache(
                location=location,
                marker=marker,
                signature=signature,
                environment=environment,
            )
            return

        try:
            bootstrap_timeout = max(
                0.0,
                float(environment.get("LF_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS", "1800")),
            )
        except ValueError as exc:
            raise ValueError(
                "LF_RUNTIME_BOOTSTRAP_TIMEOUT_SECONDS must be a nonnegative number"
            ) from exc

        common_child_environment = dict(environment)
        common_child_environment.update(
            {
                "LF_COSMIC_SUPERVISED_CHILD": "1",
                "LF_RUNTIME_BOOTSTRAP_CHILD": "1",
                "MCS_RECORD_EVENT_FAILURES": "0",
                "N_EVENTS_PER_BATCH": "1",
                "PRINT_BATCH_PROGRESS": "0",
                "PRINT_CHECKPOINT_MESSAGES": "0",
                "PRINT_EVENT_RESULTS": "0",
                "PRINT_LIVE_EVENT_PROGRESS": "0",
                "SAVE_AFTER_EACH_BATCH": "0",
                "SAVE_DETAILED_EVENT_RESULTS": "0",
                "TOT_EVENTS": "1",
                "VERBOSE_SETUP": "0",
            }
        )
        requested_nproc = max(1, int(environment.get("NPROC", "1")))
        phases: list[tuple[str, dict[str, str]]] = [
            (
                "serial cache build",
                {
                    "NPROC": "1",
                    "WARM_FIT_KERNELS": "1",
                    "LF_RUNTIME_BOOTSTRAP_PHASE": "serial-cache-build",
                },
            )
        ]
        if requested_nproc > 1:
            phases.append(
                (
                    f"{requested_nproc}-worker fork check",
                    {
                        "NPROC": str(requested_nproc),
                        "WARM_FIT_KERNELS": "0",
                        "LF_RUNTIME_BOOTSTRAP_PHASE": "production-fork-check",
                        "LF_RUNTIME_BOOTSTRAP_FORCE_ALL_WORKERS": "1",
                    },
                )
            )

        if not simple_console:
            print(
                "Runtime cache is cold for this fitter configuration. Building "
                "the cache in one disposable process, then checking the exact "
                "production worker contract; production event 0 will not be a "
                "first-JIT fit.",
                flush=True,
            )
        phase_records: list[dict[str, object]] = []
        result_paths: list[Path] = []
        ready_dirs: list[Path] = []
        for phase_index, (phase_label, phase_overrides) in enumerate(phases, start=1):
            phase_slug = str(phase_overrides["LF_RUNTIME_BOOTSTRAP_PHASE"])
            result_path = bootstrap_dir / (
                f"bootstrap-{signature}-{phase_slug}-{os.getpid()}.dict"
            )
            log_path = bootstrap_dir / f"bootstrap-{signature}-{phase_slug}.log"
            child_environment = dict(common_child_environment)
            child_environment.update(phase_overrides)
            child_environment["LF_OUTPUT_FILE"] = str(result_path)
            ready_dir = None
            if child_environment.get(
                "LF_RUNTIME_BOOTSTRAP_FORCE_ALL_WORKERS", "0"
            ) == "1":
                ready_dir = bootstrap_dir / (
                    f"workers-{signature}-{phase_slug}-{os.getpid()}"
                )
                if ready_dir.exists():
                    shutil.rmtree(ready_dir)
                ready_dir.mkdir(mode=0o700)
                child_environment[
                    "LF_RUNTIME_BOOTSTRAP_WORKER_READY_DIR"
                ] = str(ready_dir)
            if not simple_console:
                print(
                    f"  bootstrap phase {phase_index}/{len(phases)}: {phase_label}",
                    flush=True,
                )
            phase_elapsed = _run_runtime_bootstrap_phase(
                child_environment,
                phase_label=phase_label,
                log_path=log_path,
                timeout_seconds=bootstrap_timeout,
            )
            try:
                with result_path.open("rb") as stream:
                    bootstrap_result = pickle.load(stream)
                bootstrap_completed = int(
                    bootstrap_result.get("metadata", {}).get(
                        "n_events_completed", 0
                    )
                )
            except (
                AttributeError,
                EOFError,
                OSError,
                pickle.UnpicklingError,
                TypeError,
                ValueError,
            ) as exc:
                raise RuntimeError(
                    f"Runtime bootstrap {phase_label} exited successfully but "
                    "did not produce a readable one-event result at "
                    f"{result_path}"
                ) from exc
            if bootstrap_completed != 1:
                raise RuntimeError(
                    f"Runtime bootstrap {phase_label} did not complete its "
                    f"one-event fit (completed={bootstrap_completed}). Log: "
                    f"{log_path}"
                )
            initialized_pids: set[int] = set()
            if ready_dir is not None:
                for ready_path in ready_dir.glob("*.ready"):
                    try:
                        if ready_path.is_symlink() or not ready_path.is_file():
                            continue
                        pid = int(ready_path.stem)
                        if int(ready_path.read_text(encoding="utf-8")) == pid:
                            initialized_pids.add(pid)
                    except (OSError, TypeError, ValueError):
                        continue
                expected_workers = int(child_environment["NPROC"])
                if len(initialized_pids) < expected_workers:
                    raise RuntimeError(
                        f"Runtime bootstrap {phase_label} observed only "
                        f"{len(initialized_pids)}/{expected_workers} worker "
                        f"initializers. Cache certification refused. Log: "
                        f"{log_path}"
                    )
                ready_dirs.append(ready_dir)
            phase_records.append(
                {
                    "phase": phase_slug,
                    "nproc": int(child_environment["NPROC"]),
                    "forced_full_worker_pool": bool(
                        child_environment.get(
                            "LF_RUNTIME_BOOTSTRAP_FORCE_ALL_WORKERS", "0"
                        ) == "1"
                    ),
                    "events_completed": bootstrap_completed,
                    "workers_initialized_observed": len(initialized_pids),
                    "elapsed_s": phase_elapsed,
                }
            )
            result_paths.append(result_path)

        artifacts = _cache_artifact_manifest(location.root)
        if not artifacts:
            raise RuntimeError(
                "Runtime bootstrap exited successfully but produced no cache "
                f"artifacts under {location.root}"
            )
        marker_payload = {
            "protocol": _RUNTIME_BOOTSTRAP_PROTOCOL,
            "verified": True,
            "release": _PUBLIC_DRIVER_RELEASE,
            "signature": signature,
            "elapsed_s": float(sum(
                float(row["elapsed_s"]) for row in phase_records
            )),
            "bootstrap_phases": phase_records,
            "runtime_contract": _runtime_contract(environment),
            "artifacts": artifacts,
        }
        temporary_marker = marker.with_suffix(f".tmp-{os.getpid()}")
        temporary_marker.write_text(
            json.dumps(marker_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_marker, marker)
        for result_path in result_paths:
            result_path.unlink(missing_ok=True)
        for ready_dir in ready_dirs:
            shutil.rmtree(ready_dir, ignore_errors=True)
        elapsed = float(sum(float(row["elapsed_s"]) for row in phase_records))
        if not simple_console:
            print(
                f"Runtime bootstrap completed in {elapsed:.1f} s; starting a clean "
                "production process.",
                flush=True,
            )
        _activate_verified_runtime_cache(
            location=location,
            marker=marker,
            signature=signature,
            environment=environment,
        )


def _print_configuration() -> None:
    print(f"Launcher: {Path(__file__).resolve()}")
    print(f"Driver:   {_DRIVER}")
    print(f"Geometry submodule: {_GEOMETRY_SUBMODULE}")
    for name, value in _configuration_items():
        shown = "<unset>" if value is None else _encode(value)
        print(f"{name}={shown}")


def _resolved_output_file() -> str:
    """Return the exact explicit/default output path used by the driver."""
    requested = str(OUTPUT_FILE).strip()
    if requested:
        return str(Path(requested).expanduser())
    reconstruction = _resolved_reconstruction_configuration()
    detector = "wcte" if bool(USE_WCTE_GEOMETRY) else "iwcd"
    return str(
        _PROJECT_ROOT
        / "outputs"
        / (
            f"estimates_{detector}_{FIT_PARTICLE}_{int(float(ENERGY_LABEL_MEV))}MeV_"
            f"{reconstruction.public_mode_label}_{LIKELIHOOD_MODE}.dict"
        )
    )


def _print_run_summary(output_file: str) -> None:
    print_welcome(SEEDING_MODE, INTERACTION_MODE)
    print_details(
        (
            ("Input NPZ", Path(str(INPUT_FILE)).expanduser()),
            ("Particle", FIT_PARTICLE),
            ("Energy", f"{float(ENERGY_LABEL_MEV):g} MeV"),
            ("Likelihood", LIKELIHOOD_MODE),
            ("Events", f"{int(N_EVENTS)} from index {int(EVENT_START_INDEX)}"),
            ("Workers", int(NPROC)),
            ("Output", output_file),
        )
    )
    print_preparation_notice()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure and run LicketyFit on WCSim data."
    )
    parser.add_argument(
        "--show-config", action="store_true",
        help="print the driver environment generated by this file and exit",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="validate choices and required local paths without running a fit",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="show the legacy detailed setup, timing, and checkpoint diagnostics",
    )
    args = parser.parse_args()
    try:
        _validate_runtime_dependencies()
        _validate(check_paths=bool(args.check or not args.show_config))
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.show_config:
        _print_configuration()
        return
    if args.check:
        print("WCSim run configuration is valid.")
        return
    environment = build_environment()
    output_file = _resolved_output_file()
    environment["LF_OUTPUT_FILE"] = output_file
    environment["LF_SIMPLE_CONSOLE"] = "0" if args.verbose else "1"
    environment["LF_PROGRESS_INTERVAL"] = "50"
    if args.verbose:
        print(
            f"LicketyFit launch: {N_EVENTS} event"
            f"{'s' if int(N_EVENTS) != 1 else ''} from index {EVENT_START_INDEX}, "
            f"{NPROC} worker{'s' if int(NPROC) != 1 else ''}, "
            f"{_internal_threads_per_worker()} internal thread"
            f"{'s' if _internal_threads_per_worker() != 1 else ''} per worker.",
            flush=True,
        )
    else:
        _print_run_summary(output_file)
    _ensure_runtime_cache_ready(environment)
    if args.verbose:
        os.execve(sys.executable, [sys.executable, str(_DRIVER_ENTRY)], environment)
    return_code = run_with_simple_console(
        [sys.executable, str(_DRIVER_ENTRY)],
        environment=environment,
    )
    if return_code != 0:
        raise SystemExit(return_code)
    print_goodbye(output_file)


if __name__ == "__main__":
    main()
