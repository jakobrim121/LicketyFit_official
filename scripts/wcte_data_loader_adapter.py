"""Thin real-WCTE input adapter built on the shared ``analysis_tools`` API.

The fitter owns reconstruction, detector geometry, prompt conditioning, charge
conversion, calibration, and output.  This module owns only the production ROOT
input layer:

* resolve/import the external ``analysis_tools`` checkout (or an installed/
  submodule copy);
* use :class:`analysis_tools.DataLoader` for ROOT access and data-quality cuts;
* use :class:`analysis_tools.BeamSelection` for the run-derived particle PID;
* convert selected jagged hit records into LicketyFit's compact ``N x 5`` hit
  table ``[global_pmt_id, charge_adc, calibrated_time_ns,
  root_entry_index, event_number]``.

No reconstruction code and no WCSim truth is used here.  The returned calibrated
PMT times are deliberately left in the production-file convention; the batch
fitter applies its existing prompt window and event-time reference afterwards.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import inspect
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_ANALYSIS_TOOLS_PATH = (
    "/eos/user/j/jrimmer/SWAN_projects/beam/data_production_v1/analysis_tools"
)
DEFAULT_PRODUCTION_TEMPLATE = (
    "/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/"
    "production_v1_0/{run}/WCTE_merged_production_R{run}.root"
)

_PARTICLE_ALIASES = {
    "mu": "muon", "mu-": "muon", "mu+": "muon", "muon": "muon",
    "pi": "pion", "pi-": "pion", "pi+": "pion", "pion": "pion",
    "e": "electron", "e-": "electron", "e+": "electron", "electron": "electron",
    "p": "proton", "p+": "proton", "proton": "proton",
}

_BASE_REQUIRED_BRANCHES = (
    "event_number",
    "hit_mpmt_slot_ids",
    "hit_pmt_position_ids",
    "hit_pmt_charges",
    "hit_pmt_calibrated_times",
    "vme_act_eveto",
    "vme_act_tagger",
    "vme_tof_corr",
)
_OPTIONAL_ID_BRANCHES = ("run_id", "sub_run_id", "spill_counter", "readout_number")
_MPMT_DQ_BRANCHES = ("window_data_quality_mask", "hit_pmt_readout_mask")
_VME_DQ_BRANCHES = ("vme_digi_issues_bitmask", "vme_evt_quality_bitmask")
_T5_DQ_BRANCHES = (
    "T5_HasValidHit",
    "T5_HasMultipleScintillatorsHit",
    "T5_HasInTimeWindow",
)
_ROOT_ENTRY_FIELD = "__licketyfit_root_entry_index"


@dataclass(frozen=True)
class WCTESelectionConfig:
    run: int
    root_file: str
    particle: str = "muon"
    max_root_entries: int | None = 50_000
    max_selected_events: int | None = None
    step_size: int | str = 1000

    apply_mpmt_data_quality_cuts: bool = True
    apply_vme_event_quality_cuts: bool = True
    apply_t5_event_quality_cuts: bool = True
    require_muon_tagger: bool = False
    tof_fallback_when_zero_ns: float = 999.0

    # Additional *hit-level* selection used by the historical LicketyFit loader
    # and by the supplied analysis-tools example.  These are independent of the
    # fitter's later narrow prompt window.
    use_t5_hit_time_cut: bool = True
    t5_peak_window_ns: float = 200.0
    t5_peak_bin_width_ns: float = 50.0
    t5_peak_time_min_ns: float = -2000.0
    t5_peak_time_max_ns: float = 4000.0

    use_calibrated_peak_time_cut: bool = False
    calibrated_peak_window_ns: float = 100.0
    calibrated_peak_bin_width_ns: float = 50.0
    calibrated_peak_time_min_ns: float = 0.0
    calibrated_peak_time_max_ns: float = 10000.0

    peak_sample_events: int = 2000
    peak_sample_hits: int | None = None
    analysis_tools_path: str | None = None
    verbose: bool = True


class AnalysisToolsImportError(RuntimeError):
    """Raised when the shared analysis-tools package cannot be resolved."""


def _canonical_particle(value: str) -> str:
    key = str(value).strip().lower()
    try:
        return _PARTICLE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported WCTE particle selection {value!r}; expected one of "
            "muon, pion, electron, or proton."
        ) from exc


def _sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception:
        return None


def _candidate_import_roots(
    *, explicit_path: str | os.PathLike[str] | None, project_root: Path | None
) -> list[Path]:
    candidates: list[Path] = []

    def add(value: str | os.PathLike[str] | None) -> None:
        if value is None or not str(value).strip():
            return
        path = Path(value).expanduser()
        try:
            path = path.resolve()
        except Exception:
            path = path.absolute()
        if path not in candidates:
            candidates.append(path)

    add(explicit_path)
    add(os.environ.get("WCTE_ANALYSIS_TOOLS_PATH"))
    add(os.environ.get("ANALYSIS_TOOLS_PATH"))

    if project_root is not None:
        root = Path(project_root)
        # Current external checkout, plus conventional future submodule homes.
        for relative in (
            "analysis_tools",
            "external/analysis_tools",
            "extern/analysis_tools",
            "third_party/analysis_tools",
            "vendor/analysis_tools",
            "../analysis_tools",
            "../data_production_v1/analysis_tools",
        ):
            add(root / relative)

    add(DEFAULT_ANALYSIS_TOOLS_PATH)
    return candidates


def _sys_path_root_for_candidate(candidate: Path) -> Path | None:
    """Return the path that must be on ``sys.path`` for ``import analysis_tools``."""
    # Repository root: <candidate>/analysis_tools/__init__.py
    if (candidate / "analysis_tools" / "__init__.py").is_file():
        return candidate
    # Package directory itself: <candidate>/__init__.py
    if candidate.name == "analysis_tools" and (candidate / "__init__.py").is_file():
        return candidate.parent
    return None


def import_analysis_tools(
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> tuple[type, type, dict[str, Any]]:
    """Resolve and import ``DataLoader`` and ``BeamSelection``.

    Normal Python import is attempted first, which makes a future installed or
    submodule-packaged dependency work without path configuration.  Explicit and
    conventional checkout paths are then tried in deterministic order.
    """
    attempts: list[str] = []

    def load() -> tuple[type, type, dict[str, Any]]:
        module = importlib.import_module("analysis_tools")
        DataLoader = getattr(module, "DataLoader")
        BeamSelection = getattr(module, "BeamSelection")
        module_file = Path(inspect.getfile(module)).resolve()
        metadata = {
            "module": "analysis_tools",
            "module_file": str(module_file),
            "repository_root": str(module_file.parent.parent),
            "data_loader_file": str(Path(inspect.getfile(DataLoader)).resolve()),
            "beam_selection_file": str(Path(inspect.getfile(BeamSelection)).resolve()),
            "data_loader_sha256": _sha256_file(Path(inspect.getfile(DataLoader)).resolve()),
            "beam_selection_sha256": _sha256_file(Path(inspect.getfile(BeamSelection)).resolve()),
        }
        return DataLoader, BeamSelection, metadata

    try:
        return load()
    except Exception as exc:
        attempts.append(f"normal import: {exc!r}")

    root_path = Path(project_root).resolve() if project_root is not None else None
    for candidate in _candidate_import_roots(
        explicit_path=explicit_path, project_root=root_path
    ):
        import_root = _sys_path_root_for_candidate(candidate)
        if import_root is None:
            attempts.append(f"{candidate}: no analysis_tools package found")
            continue
        text = str(import_root)
        if text not in sys.path:
            sys.path.insert(0, text)
        # An earlier failed import can leave the package or one of its imported
        # submodules behind. Purge the complete namespace before retrying another
        # checkout so DataLoader and BeamSelection cannot come from mixed roots.
        for module_name in tuple(sys.modules):
            if module_name == "analysis_tools" or module_name.startswith("analysis_tools."):
                sys.modules.pop(module_name, None)
        try:
            DataLoader, BeamSelection, metadata = load()
            metadata["resolved_from_candidate"] = str(candidate)
            metadata["sys_path_root"] = text
            return DataLoader, BeamSelection, metadata
        except Exception as exc:
            attempts.append(f"{candidate}: {exc!r}")

    raise AnalysisToolsImportError(
        "Could not import the WCTE analysis_tools package. Set "
        "WCTE_ANALYSIS_TOOLS_PATH to the repository root (the directory that "
        "contains analysis_tools/__init__.py). Attempts:\n  - "
        + "\n  - ".join(attempts)
    )


def production_root_file(run: int, override: str | None = None) -> str:
    return str(override).strip() if override and str(override).strip() else (
        DEFAULT_PRODUCTION_TEMPLATE.format(run=int(run))
    )


def _scalar(value: Any, label: str) -> float:
    try:
        import awkward as ak
        array = np.asarray(ak.to_numpy(value), dtype=np.float64).ravel()
    except Exception:
        try:
            array = np.asarray(value, dtype=np.float64).ravel()
        except Exception as exc:
            raise ValueError(f"Could not convert {label} to a scalar") from exc
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"{label} contains no finite scalar value")
    return float(finite[0])


def _selection_thresholds(loader: Any, tof_fallback: float) -> dict[str, float]:
    values = loader.get_vme_analysis_scalar_results()
    tof = _scalar(values["proton_tof_cut"], "proton_tof_cut")
    if tof == 0.0:
        tof = float(tof_fallback)
    return {
        "proton_tof_cut_ns": float(tof),
        "act_eveto_cut_pe": _scalar(values["act_eveto_cut"], "act_eveto_cut"),
        "act_tagger_cut_pe": _scalar(values["act_tagger_cut"], "act_tagger_cut"),
        "mu_tag_cut": _scalar(values["mu_tag_cut"], "mu_tag_cut"),
    }


def _build_selection(BeamSelection: type, particle: str, cuts: Mapping[str, float], *, require_muon_tagger: bool):
    tof = float(cuts["proton_tof_cut_ns"])
    eveto = float(cuts["act_eveto_cut_pe"])
    tagger = float(cuts["act_tagger_cut_pe"])
    mu_tag = float(cuts["mu_tag_cut"])
    if particle == "muon":
        specs: list[list[Any]] = [
            ["vme_act_eveto", "<", eveto],
            ["vme_act_tagger", ">", tagger],
            ["vme_tof_corr", "<", tof],
        ]
        if require_muon_tagger:
            specs.append(["vme_mu_tag_total", ">", mu_tag])
    elif particle == "pion":
        specs = [
            ["vme_act_eveto", "<", eveto],
            ["vme_act_tagger", "<", tagger],
            ["vme_tof_corr", "<", tof],
        ]
    elif particle == "electron":
        specs = [
            ["vme_act_eveto", ">", eveto],
            ["vme_tof_corr", "<", tof],
        ]
    elif particle == "proton":
        specs = [["vme_tof_corr", "between", [tof, tof + 10.0]]]
    else:  # pragma: no cover - guarded by canonicalization
        raise AssertionError(particle)
    return BeamSelection.selection(particle, *specs)


def _configure_dq(loader: Any, cfg: WCTESelectionConfig) -> None:
    if cfg.apply_mpmt_data_quality_cuts:
        loader.apply_mPMT_data_quality_cuts()
    if cfg.apply_vme_event_quality_cuts:
        loader.apply_vme_event_quality_cuts()
    if cfg.apply_t5_event_quality_cuts:
        loader.apply_t5_event_quality_cuts()


def _loader_with_branches(
    DataLoader: type,
    cfg: WCTESelectionConfig,
    *,
    require_t5_time: bool,
) -> tuple[Any, list[str], set[str]]:
    probe = DataLoader(cfg.root_file, branches_to_load=[])
    try:
        available = set(str(x) for x in probe.file["WCTEReadoutWindows"].keys())
    finally:
        probe.file.close()

    required = list(_BASE_REQUIRED_BRANCHES)
    if cfg.require_muon_tagger:
        required.append("vme_mu_tag_total")
    if require_t5_time:
        required.append("T5_hit_time")
    # DataLoader intentionally adds these masks only when they exist.  When a
    # corresponding DQ operation is requested, however, absence is a hard input
    # error rather than something that should fail later with an opaque KeyError
    # inside DataLoader._apply_all_data_quality_cuts().
    if cfg.apply_mpmt_data_quality_cuts:
        required.extend(_MPMT_DQ_BRANCHES)
    if cfg.apply_vme_event_quality_cuts:
        required.extend(_VME_DQ_BRANCHES)
    if cfg.apply_t5_event_quality_cuts:
        required.extend(_T5_DQ_BRANCHES)
    required = list(dict.fromkeys(required))

    missing = [name for name in required if name not in available]
    if missing:
        raise RuntimeError(
            f"WCTE production file {cfg.root_file!r} is missing required branches: "
            + ", ".join(missing)
        )

    branches = required + [name for name in _OPTIONAL_ID_BRANCHES if name in available]
    loader = DataLoader(cfg.root_file, branches_to_load=list(dict.fromkeys(branches)))
    _configure_dq(loader, cfg)
    return loader, list(loader.branches_to_load or []), available


def _to_numpy_1d(value: Any, dtype: Any) -> np.ndarray:
    try:
        import awkward as ak
        return np.asarray(ak.to_numpy(value), dtype=dtype).ravel()
    except Exception:
        return np.asarray(value, dtype=dtype).ravel()


def _first_finite(value: Any) -> float:
    array = _to_numpy_1d(value, np.float64)
    finite = array[np.isfinite(array)]
    return float(finite[0]) if finite.size else math.nan


def _histogram_peak(
    chunks: Sequence[np.ndarray], *, bin_width: float, low: float, high: float
) -> float | None:
    if not chunks:
        return None
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    values = values[np.isfinite(values) & (values >= low) & (values < high)]
    if values.size == 0:
        return None
    if not (bin_width > 0.0 and high > low):
        raise ValueError("Invalid WCTE peak-histogram configuration")
    edges = np.arange(float(low), float(high) + float(bin_width), float(bin_width))
    counts, edges = np.histogram(values, bins=edges)
    if counts.size == 0 or int(np.max(counts)) == 0:
        return None
    index = int(np.argmax(counts))
    return float(0.5 * (edges[index] + edges[index + 1]))


def _iterate_selected(
    loader: Any,
    selection: Any,
    cfg: WCTESelectionConfig,
) -> Iterable[tuple[Any, np.ndarray]]:
    """Yield selected batches together with original ROOT entry indices.

    ``DataLoader.iterate`` intentionally returns only the post-DQ awkward
    batch.  The analysis-tools LicketyFit example therefore attaches the raw
    ``WCTEReadoutWindows`` entry index before calling DataLoader's common DQ
    implementation.  We follow that example here so event identity survives
    event-level DQ cuts exactly rather than being replaced by a selected-list
    counter.

    The private method is used narrowly and guarded explicitly because the
    current analysis-tools public iterator does not expose uproot reports or
    raw entry indices.  A future public indexed iterator can replace this
    block without changing the adapter output contract.
    """
    try:
        import awkward as ak
    except Exception as exc:  # pragma: no cover - real ROOT operation requires ak
        raise RuntimeError(
            "Indexed WCTE DataLoader iteration requires awkward, which is also "
            "a dependency of analysis_tools.DataLoader."
        ) from exc

    apply_dq = getattr(loader, "_apply_all_data_quality_cuts", None)
    if not callable(apply_dq):
        raise RuntimeError(
            "The installed analysis_tools.DataLoader does not provide the "
            "indexed-DQ hook used by the supplied LF_data_loader example "
            "(_apply_all_data_quality_cuts). Update the adapter for the new "
            "analysis_tools API rather than silently losing ROOT entry identity."
        )

    tree = loader.file["WCTEReadoutWindows"]
    kwargs: dict[str, Any] = {
        "expressions": loader.branches_to_load,
        "step_size": cfg.step_size,
        "library": "ak",
    }
    if cfg.max_root_entries is not None:
        kwargs["entry_stop"] = max(0, int(cfg.max_root_entries))

    raw_entry_cursor = 0
    for raw_batch in tree.iterate(**kwargs):
        raw_length = int(len(raw_batch))
        if raw_length == 0:
            continue
        raw_indices = np.arange(
            raw_entry_cursor,
            raw_entry_cursor + raw_length,
            dtype=np.int64,
        )
        raw_entry_cursor += raw_length
        indexed = ak.with_field(raw_batch, raw_indices, _ROOT_ENTRY_FIELD)
        batch = apply_dq(indexed, False)
        if len(batch) == 0:
            continue
        selected = batch[selection.mask(batch)]
        if len(selected) == 0:
            continue
        indices = np.asarray(
            ak.to_numpy(selected[_ROOT_ENTRY_FIELD]), dtype=np.int64
        ).ravel()
        if indices.size != len(selected):
            raise RuntimeError(
                "DataLoader selection returned a ROOT-index array with the "
                "wrong length."
            )
        yield selected, indices


def _estimate_optional_peaks(
    DataLoader: type,
    BeamSelection: type,
    cfg: WCTESelectionConfig,
    cuts: Mapping[str, float],
) -> tuple[float | None, float | None, dict[str, int]]:
    if not (cfg.use_t5_hit_time_cut or cfg.use_calibrated_peak_time_cut):
        return None, None, {"events": 0, "hits": 0}

    loader, _, _ = _loader_with_branches(
        DataLoader, cfg, require_t5_time=cfg.use_t5_hit_time_cut
    )
    selection = _build_selection(
        BeamSelection,
        cfg.particle,
        cuts,
        require_muon_tagger=cfg.require_muon_tagger,
    )
    t5_chunks: list[np.ndarray] = []
    calibrated_chunks: list[np.ndarray] = []
    n_events = 0
    n_hits = 0
    try:
        stop = False
        for batch, _root_indices in _iterate_selected(loader, selection, cfg):
            for record in batch:
                times = _to_numpy_1d(record["hit_pmt_calibrated_times"], np.float64)
                times = times[np.isfinite(times)]
                if times.size == 0:
                    continue
                if cfg.peak_sample_hits is not None:
                    remaining = int(cfg.peak_sample_hits) - n_hits
                    if remaining <= 0:
                        stop = True
                        break
                    times = times[:remaining]
                if cfg.use_calibrated_peak_time_cut:
                    calibrated_chunks.append(times)
                if cfg.use_t5_hit_time_cut:
                    t5 = _first_finite(record["T5_hit_time"])
                    if np.isfinite(t5):
                        t5_chunks.append(times - t5)
                n_events += 1
                n_hits += int(times.size)
                if n_events >= max(1, int(cfg.peak_sample_events)):
                    stop = True
                    break
                if cfg.peak_sample_hits is not None and n_hits >= int(cfg.peak_sample_hits):
                    stop = True
                    break
            if stop:
                break
    finally:
        loader.file.close()

    t5_peak = _histogram_peak(
        t5_chunks,
        bin_width=cfg.t5_peak_bin_width_ns,
        low=cfg.t5_peak_time_min_ns,
        high=cfg.t5_peak_time_max_ns,
    ) if cfg.use_t5_hit_time_cut else None
    calibrated_peak = _histogram_peak(
        calibrated_chunks,
        bin_width=cfg.calibrated_peak_bin_width_ns,
        low=cfg.calibrated_peak_time_min_ns,
        high=cfg.calibrated_peak_time_max_ns,
    ) if cfg.use_calibrated_peak_time_cut else None

    if cfg.use_t5_hit_time_cut and t5_peak is None:
        raise RuntimeError(
            "Could not estimate the selected-event PMT-minus-T5 timing peak "
            f"from {cfg.root_file!r}."
        )
    if cfg.use_calibrated_peak_time_cut and calibrated_peak is None:
        raise RuntimeError(
            "Could not estimate the selected-event calibrated PMT timing peak "
            f"from {cfg.root_file!r}."
        )
    return t5_peak, calibrated_peak, {"events": n_events, "hits": n_hits}


def _record_to_event(
    record: Any,
    *,
    root_entry_index: int,
    fallback_event_number: int,
    cfg: WCTESelectionConfig,
    t5_peak: float | None,
    calibrated_peak: float | None,
) -> np.ndarray:
    slots = _to_numpy_1d(record["hit_mpmt_slot_ids"], np.int64)
    positions = _to_numpy_1d(record["hit_pmt_position_ids"], np.int64)
    charges = _to_numpy_1d(record["hit_pmt_charges"], np.float64)
    times = _to_numpy_1d(record["hit_pmt_calibrated_times"], np.float64)
    if not (slots.size == positions.size == charges.size == times.size):
        raise RuntimeError(
            "DataLoader returned inconsistent WCTE hit-array lengths: "
            f"slot={slots.size}, position={positions.size}, charge={charges.size}, "
            f"time={times.size}."
        )

    keep = (
        np.isfinite(charges)
        & np.isfinite(times)
        & (slots >= 0)
        & (positions >= 0)
    )
    if cfg.use_t5_hit_time_cut:
        t5 = _first_finite(record["T5_hit_time"])
        if not (np.isfinite(t5) and t5_peak is not None):
            keep &= False
        else:
            keep &= np.abs((times - t5) - float(t5_peak)) <= float(cfg.t5_peak_window_ns)
    if cfg.use_calibrated_peak_time_cut:
        if calibrated_peak is None:
            keep &= False
        else:
            keep &= np.abs(times - float(calibrated_peak)) <= float(
                cfg.calibrated_peak_window_ns
            )

    slots = slots[keep]
    positions = positions[keep]
    charges = charges[keep]
    times = times[keep]
    event_number = _first_finite(record["event_number"])
    if not np.isfinite(event_number):
        event_number = float(fallback_event_number)
    ids = slots * 100 + positions
    root_identity = np.full(ids.size, int(root_entry_index), dtype=np.float64)
    event_identity = np.full(ids.size, int(event_number), dtype=np.float64)
    if ids.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    return np.column_stack((
        ids.astype(np.float64),
        charges.astype(np.float64),
        times.astype(np.float64),
        root_identity,
        event_identity,
    ))


def load_selected_events(
    config: WCTESelectionConfig,
    *,
    project_root: str | os.PathLike[str] | None = None,
    return_metadata: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict[str, Any]]:
    """Load beam-selected real-WCTE events through ``analysis_tools``."""
    particle = _canonical_particle(config.particle)
    cfg = WCTESelectionConfig(**{**asdict(config), "particle": particle})
    DataLoader, BeamSelection, import_metadata = import_analysis_tools(
        explicit_path=cfg.analysis_tools_path,
        project_root=project_root,
    )

    # Read run-derived PID thresholds once through DataLoader.
    threshold_loader, branches, available = _loader_with_branches(
        DataLoader, cfg, require_t5_time=cfg.use_t5_hit_time_cut
    )
    try:
        cuts = _selection_thresholds(
            threshold_loader, cfg.tof_fallback_when_zero_ns
        )
        run_info = None
        try:
            info = threshold_loader.get_vme_analysis_run_info()
            run_info = {
                key: _scalar(info[key], key)
                for key in ("run_momentum", "n_eveto", "n_tagger")
                if key in info.fields
            }
        except Exception:
            run_info = None
    finally:
        threshold_loader.file.close()

    t5_peak, calibrated_peak, peak_stats = _estimate_optional_peaks(
        DataLoader, BeamSelection, cfg, cuts
    )

    loader, branches, _ = _loader_with_branches(
        DataLoader, cfg, require_t5_time=cfg.use_t5_hit_time_cut
    )
    selection = _build_selection(
        BeamSelection,
        particle,
        cuts,
        require_muon_tagger=cfg.require_muon_tagger,
    )
    events: list[np.ndarray] = []
    selected_before_hit_cut = 0
    empty_after_hit_cut = 0
    fallback_id = 0
    try:
        for batch, root_indices in _iterate_selected(loader, selection, cfg):
            selected_before_hit_cut += int(len(batch))
            for record, root_entry_index in zip(batch, root_indices):
                event = _record_to_event(
                    record,
                    root_entry_index=int(root_entry_index),
                    fallback_event_number=fallback_id,
                    cfg=cfg,
                    t5_peak=t5_peak,
                    calibrated_peak=calibrated_peak,
                )
                fallback_id += 1
                if event.shape[0] == 0:
                    empty_after_hit_cut += 1
                    continue
                events.append(event)
                if (
                    cfg.max_selected_events is not None
                    and len(events) >= int(cfg.max_selected_events)
                ):
                    break
            if (
                cfg.max_selected_events is not None
                and len(events) >= int(cfg.max_selected_events)
            ):
                break
    finally:
        loader.file.close()

    metadata: dict[str, Any] = {
        "adapter": "LicketyFit.scripts.wcte_data_loader_adapter",
        "analysis_tools": import_metadata,
        "config": asdict(cfg),
        "root_file": cfg.root_file,
        "particle": particle,
        "branches_requested": branches,
        "available_branch_count": int(len(available)),
        "selection_thresholds": dict(cuts),
        "run_info": run_info,
        "t5_corrected_peak_ns": t5_peak,
        "calibrated_peak_ns": calibrated_peak,
        "peak_sample": peak_stats,
        "selected_before_hit_time_cut": int(selected_before_hit_cut),
        "empty_after_hit_time_cut": int(empty_after_hit_cut),
        "events_returned": int(len(events)),
        "root_entry_limit_semantics": (
            "WCTEReadoutWindows.iterate(entry_stop=max_root_entries), followed by "
            "DataLoader._apply_all_data_quality_cuts and BeamSelection"
        ),
        "hit_table_columns": [
            "global_wcte_pmt_id",
            "hit_pmt_charges_adc",
            "hit_pmt_calibrated_times_ns",
            "WCTEReadoutWindows_root_entry_index",
            "WCTEReadoutWindows_event_number",
        ],
        "root_entry_index_column": 3,
        "event_identity_column": 4,
        "times_returned": "hit_pmt_calibrated_times",
        "charges_returned": "hit_pmt_charges (calibrated ADC; fitter converts to PE)",
    }
    if cfg.verbose:
        print("WCTE analysis_tools DataLoader selection")
        print("  ROOT file:", cfg.root_file)
        print("  particle:", particle)
        print("  raw entry limit:", cfg.max_root_entries)
        print("  selected-event cap:", cfg.max_selected_events)
        print("  DQ cuts (mPMT/VME/T5):", (
            cfg.apply_mpmt_data_quality_cuts,
            cfg.apply_vme_event_quality_cuts,
            cfg.apply_t5_event_quality_cuts,
        ))
        print("  run-derived cuts:", cuts)
        print("  PMT-minus-T5 peak [ns]:", t5_peak)
        print("  calibrated-time peak [ns]:", calibrated_peak)
        print("  selected / returned / empty:", (
            selected_before_hit_cut, len(events), empty_after_hit_cut
        ))
        print("  analysis_tools source:", import_metadata.get("repository_root"))

    return (events, metadata) if return_metadata else events


def load_good_wcte_pmts(
    root_file: str,
    *,
    analysis_tools_path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> tuple[set[int], dict[str, Any]]:
    """Read ``Configuration/good_wcte_pmts`` through ``DataLoader``."""
    DataLoader, _, import_metadata = import_analysis_tools(
        explicit_path=analysis_tools_path,
        project_root=project_root,
    )
    loader = DataLoader(str(root_file), branches_to_load=[])
    try:
        slots, positions = loader.get_good_wcte_pmts()
        slots = _to_numpy_1d(slots, np.int64)
        positions = _to_numpy_1d(positions, np.int64)
        if slots.size != positions.size:
            raise RuntimeError("DataLoader returned misaligned good-PMT slot/position arrays")
        ids = slots * 100 + positions
        good = set(int(value) for value in ids)
        if not good:
            raise RuntimeError("DataLoader Configuration/good_wcte_pmts is empty")
    finally:
        loader.file.close()
    return good, {
        "root_file": str(root_file),
        "good_pmt_count": int(len(good)),
        "analysis_tools": import_metadata,
    }


__all__ = [
    "AnalysisToolsImportError",
    "DEFAULT_ANALYSIS_TOOLS_PATH",
    "DEFAULT_PRODUCTION_TEMPLATE",
    "WCTESelectionConfig",
    "import_analysis_tools",
    "load_good_wcte_pmts",
    "load_selected_events",
    "production_root_file",
]
