"""Real-WCTE input adapter built on the shared ``analysis_tools`` API.

The fitter owns reconstruction, geometry, calibration, prompt conditioning and
output.  This module owns only the production ROOT input layer:

* resolve/import ``analysis_tools`` only from the repository submodule at
  ``LicketyFit_official/analysis_tools``;
* use :class:`analysis_tools.DataLoader` for ROOT access and data-quality cuts;
* use :class:`analysis_tools.BeamSelection` for nominal or custom beam PID;
* resolve good-PMT masks from a user list or a run DQ/merged ROOT without
  treating a standalone DQ product as an event stream;
* preserve both the raw ``WCTEReadoutWindows`` ROOT entry and production
  ``event_number``;
* convert selected jagged hits to LicketyFit's compact ``N x 5`` table
  ``[global_pmt_id, charge_adc, calibrated_time_ns, root_entry, event_number]``.

The nominal light-particle selections support three explicit PID policies:

* ``act``: reproduce the supplied analysis-tools ACT examples, with the
  optional proton/fast-particle TOF veto;
* ``act_tof``: retain those ACT requirements and add every usable
  run-calibrated, hypothesis-specific electron/muon/pion TOF boundary. If a
  boundary is not physically defined for a run, retain the ACT identity and
  the remaining usable TOF cuts instead of inventing a threshold;
* ``tof``: use only the hypothesis-specific light-particle TOF interval.

The adjacent light-particle boundaries are taken from explicit scalar cut
values when a future production supplies them. Otherwise they are calculated
as the equal-density intersections of the run's fitted ``tof_mean_*`` and
``tof_std_*`` populations. They are never copied between runs. ``tof`` mode is
strict because it has no ACT identity fallback; it reports an actionable error
when a required boundary is unavailable.

The remaining nominal selection is:

* proton: TOF in ``[proton_tof_cut, proton_tof_cut + window]``.

The examples explicitly say these nominal cuts must be adapted to the run.  The
configuration therefore exposes independent ACT/TOF toggles, scalar overrides,
a proton-window width, and arbitrary additional/custom ``BeamSelection`` cuts.
No nominal kaon selection is defined by analysis_tools; kaon-labelled data must
use ``selection_mode='custom'`` with explicit cuts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# Retained as a compatibility export for code that imported the old constant.
# External-path overrides are deliberately unsupported: the pinned repository
# submodule is the single authoritative analysis_tools source.
DEFAULT_ANALYSIS_TOOLS_PATH: None = None
ANALYSIS_TOOLS_SUBMODULE_DIRNAME = "analysis_tools"
DEFAULT_PRODUCTION_TEMPLATE = (
    "/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/"
    "production_v1_0/{run}/WCTE_merged_production_R{run}.root"
)
DEFAULT_GOOD_PMT_ROOT_SEARCH_BASES = (
    "/eos/experiment/wcte/data/2025_commissioning/processed_offline_data",
)

_GOOD_PMT_FILE_KEYS = (
    "good_pmt_ids",
    "good_wcte_pmts",
    "pmt_ids",
    "ids",
    "arr_0",
)

_PARTICLE_ALIASES = {
    "mu": "muon", "mu-": "muon", "mu+": "muon", "muon": "muon",
    "pi": "pion", "pi-": "pion", "pi+": "pion", "pion": "pion",
    "e": "electron", "e-": "electron", "e+": "electron", "electron": "electron",
    "k": "kaon", "k-": "kaon", "k+": "kaon", "kaon": "kaon",
    "p": "proton", "p+": "proton", "proton": "proton",
}
_NOMINAL_PARTICLES = frozenset({"muon", "pion", "electron", "proton"})
_SELECTION_MODES = frozenset({"nominal", "custom"})
_TOF_CUT_MODES = frozenset({"auto", "require", "disable"})
_LIGHT_PARTICLE_PID_MODES = frozenset({"act", "act_tof", "tof"})
_LIGHT_PARTICLES = frozenset({"electron", "muon", "pion"})
_SUPPORTED_OPERATORS = frozenset({">", "<", ">=", "<=", "==", "!=", "between"})

_BASE_HIT_BRANCHES = (
    "event_number",
    "hit_mpmt_slot_ids",
    "hit_pmt_position_ids",
    "hit_pmt_charges",
    "hit_pmt_calibrated_times",
)
_OPTIONAL_ID_BRANCHES = ("run_id", "sub_run_id", "spill_counter", "readout_number")
_MPMT_DQ_BRANCHES = ("window_data_quality_mask", "hit_pmt_readout_mask")
_VME_DQ_BRANCHES = ("vme_digi_issues_bitmask", "vme_evt_quality_bitmask")
_T5_DQ_BRANCHES = (
    "T5_HasValidHit",
    "T5_HasMultipleScintillatorsHit",
    "T5_HasInTimeWindow",
)
_SELECTION_VARIABLE_ALIASES = {
    "act_eveto": "vme_act_eveto",
    "act_tagger": "vme_act_tagger",
    "tof": "vme_tof_corr",
    "mu_tag_total": "vme_mu_tag_total",
    "act_0_charge": "vme_act0_l_charge",
    "T5_particle_nr": "T5_particle_nr",
}
_ROOT_ENTRY_FIELD = "__licketyfit_root_entry_index"


@dataclass(frozen=True)
class WCTESelectionConfig:
    run: int
    root_file: str
    particle: str = "muon"
    max_root_entries: int | None = 50_000
    max_selected_events: int | None = None
    step_size: int | str = "100 MB"

    apply_mpmt_data_quality_cuts: bool = True
    apply_vme_event_quality_cuts: bool = True
    apply_t5_event_quality_cuts: bool = True

    # PID construction. ``nominal`` follows the repository examples. ``custom``
    # applies only ``extra_selection_cuts`` and is required for a kaon label.
    selection_mode: str = "nominal"
    # act: legacy ACT identity with only the optional proton/fast TOF veto;
    # act_tof: ACT identity plus every usable run-calibrated TOF boundary;
    #           an unusable boundary falls back to ACT for that separation;
    # tof: run-calibrated e/mu/pi TOF boundaries without ACT identity cuts.
    light_particle_pid_mode: str = "act_tof"
    use_act_eveto_cut: bool = True
    use_act_tagger_cut: bool = True
    # auto: use run scalar when >0, otherwise omit for fast particles;
    # require: fail if no positive scalar/override; disable: omit.
    tof_cut_mode: str = "auto"
    proton_tof_window_ns: float = 10.0
    require_muon_tagger: bool = False

    # Optional run-scalar overrides. None means use vme_analysis_scalar_results.
    act_eveto_cut_override_pe: float | None = None
    act_tagger_cut_override_pe: float | None = None
    proton_tof_cut_override_ns: float | None = None
    muon_tag_cut_override: float | None = None
    electron_muon_tof_boundary_override_ns: float | None = None
    muon_pion_tof_boundary_override_ns: float | None = None

    # Each item is [variable, operator, value], exactly as accepted by
    # BeamSelection.selection. In nominal mode these are appended; in custom mode
    # they are the complete selection.
    extra_selection_cuts: Sequence[Sequence[Any]] = ()
    print_selection_description: bool = True
    print_cherenkov_thresholds: bool = True

    # Optional legacy/example hit-level timing preselection. This is distinct
    # from DataLoader.apply_t5_event_quality_cuts(), which operates on event-level
    # T5 validity flags. Disabled by default so the official DataLoader DQ is the
    # only T5 selection before LicketyFit's later event-specific prompt window.
    use_t5_hit_time_cut: bool = False
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
            "muon, pion, electron, kaon, or proton."
        ) from exc


def _canonical_selection_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in _SELECTION_MODES:
        raise ValueError("WCTE selection_mode must be 'nominal' or 'custom'")
    return mode


def _canonical_tof_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {"on": "require", "true": "require", "off": "disable", "false": "disable"}
    mode = aliases.get(mode, mode)
    if mode not in _TOF_CUT_MODES:
        raise ValueError("WCTE tof_cut_mode must be auto, require, or disable")
    return mode


def _canonical_light_particle_pid_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "legacy": "act",
        "combined": "act_tof",
        "tof_only": "tof",
    }
    mode = aliases.get(mode, mode)
    if mode not in _LIGHT_PARTICLE_PID_MODES:
        raise ValueError(
            "WCTE light_particle_pid_mode must be 'act', 'act_tof', or 'tof'"
        )
    return mode


def _normalise_cut_spec(spec: Sequence[Any]) -> list[Any]:
    if not isinstance(spec, (list, tuple)) or len(spec) != 3:
        raise ValueError(
            "Every WCTE selection cut must be [variable, operator, value]; "
            f"got {spec!r}"
        )
    variable, operator, value = spec
    variable = str(variable).strip()
    operator = str(operator).strip()
    if not variable:
        raise ValueError(f"WCTE selection cut has an empty variable: {spec!r}")
    if operator not in _SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unsupported WCTE selection operator {operator!r}; expected one of "
            + ", ".join(sorted(_SUPPORTED_OPERATORS))
        )
    if operator == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"'between' requires [low, high], got {value!r}")
        value = [value[0], value[1]]
    return [variable, operator, value]


def _selection_branch_name(variable: str) -> str:
    name = str(variable).strip()
    if name in _SELECTION_VARIABLE_ALIASES:
        return _SELECTION_VARIABLE_ALIASES[name]
    if name.startswith("vme_") or name.startswith("T5_"):
        return name
    # This matches analysis_tools.beam_selection._parse_cut_spec.
    return f"vme_{name}"


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
    if explicit_path is not None and str(explicit_path).strip():
        raise AnalysisToolsImportError(
            "External analysis_tools paths are no longer supported. Initialize "
            "the pinned submodule at LicketyFit_official/analysis_tools instead."
        )
    if project_root is None:
        raise AnalysisToolsImportError(
            "The LicketyFit repository root is required to resolve the "
            "analysis_tools submodule."
        )
    return [
        (Path(project_root).resolve() / ANALYSIS_TOOLS_SUBMODULE_DIRNAME).resolve()
    ]


def _sys_path_root_for_candidate(candidate: Path) -> Path | None:
    if (candidate / "analysis_tools" / "__init__.py").is_file():
        return candidate
    if candidate.name == "analysis_tools" and (candidate / "__init__.py").is_file():
        return candidate.parent
    return None


def _component_source_files(candidate: Path) -> tuple[Path, Path] | None:
    """Return the two lightweight source modules needed by LicketyFit.

    The public ``analysis_tools`` package initializer imports calibration-DB,
    waveform-processing and geometry helpers in addition to DataLoader/PID.  Some
    of those unrelated helpers have optional dependencies such as ``requests``.
    LicketyFit must not require those dependencies merely to read a merged ROOT
    file, so a component-only fallback loads the actual repository source files
    directly when the public package import fails.
    """
    package_dir: Path | None = None
    if (candidate / "analysis_tools" / "data_loader.py").is_file():
        package_dir = candidate / "analysis_tools"
    elif candidate.name == "analysis_tools" and (candidate / "data_loader.py").is_file():
        package_dir = candidate
    if package_dir is None:
        return None
    data_loader_file = package_dir / "data_loader.py"
    beam_selection_file = package_dir / "beam_selection.py"
    if not (data_loader_file.is_file() and beam_selection_file.is_file()):
        return None
    return data_loader_file, beam_selection_file


def _load_source_module(source_file: Path, role: str):
    digest = hashlib.sha256(str(source_file.resolve()).encode("utf-8")).hexdigest()[:16]
    module_name = f"_licketyfit_analysis_tools_{role}_{digest}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {source_file}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and similar runtime inspection require the module to be visible
    # while its body executes.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _import_analysis_tools_components(
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> tuple[type, type, Any, dict[str, Any]]:
    """Resolve only DataLoader, BeamSelection and the threshold printer.

    Resolve the pinned repository submodule and prefer its public package API.
    If its broad ``__init__`` fails because an unrelated optional dependency is
    absent, use the same submodule's *actual* ``data_loader.py`` and
    ``beam_selection.py`` modules directly.  This is not a reimplementation of
    their behavior, and an installed/global package is never accepted.
    """
    attempts: list[str] = []

    root_path = Path(project_root).resolve() if project_root is not None else None
    candidates = _candidate_import_roots(
        explicit_path=explicit_path, project_root=root_path
    )
    candidate = candidates[0]

    def is_within(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    def load_public() -> tuple[type, type, Any, dict[str, Any]]:
        module = importlib.import_module("analysis_tools")
        DataLoader = getattr(module, "DataLoader")
        BeamSelection = getattr(module, "BeamSelection")
        print_thresholds = getattr(module, "print_cherenkov_thresholds", None)
        module_file = Path(inspect.getfile(module)).resolve()
        data_file = Path(inspect.getfile(DataLoader)).resolve()
        beam_file = Path(inspect.getfile(BeamSelection)).resolve()
        for label, source_file in (
            ("analysis_tools", module_file),
            ("DataLoader", data_file),
            ("BeamSelection", beam_file),
        ):
            if not is_within(source_file, candidate):
                raise ImportError(
                    f"{label} resolved outside the required submodule: {source_file}"
                )
        metadata = {
            "module": "analysis_tools",
            "import_mode": "public_package_api",
            "module_file": str(module_file),
            "repository_root": str(module_file.parent.parent),
            "data_loader_file": str(data_file),
            "beam_selection_file": str(beam_file),
            "data_loader_sha256": _sha256_file(data_file),
            "beam_selection_sha256": _sha256_file(beam_file),
            "source_policy": "required_repository_submodule",
            "submodule_root": str(candidate),
        }
        return DataLoader, BeamSelection, print_thresholds, metadata

    import_root = _sys_path_root_for_candidate(candidate)
    components = _component_source_files(candidate)
    if import_root is None or components is None:
        raise AnalysisToolsImportError(
            "The required analysis_tools submodule is missing or uninitialized at "
            f"{candidate}. From LicketyFit_official run: "
            "git submodule update --init analysis_tools"
        )

    text = str(import_root)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)
    for module_name in tuple(sys.modules):
        if module_name == "analysis_tools" or module_name.startswith("analysis_tools."):
            sys.modules.pop(module_name, None)
    try:
        DataLoader, BeamSelection, print_thresholds, metadata = load_public()
        metadata["resolved_from_candidate"] = str(candidate)
        metadata["sys_path_root"] = text
        return DataLoader, BeamSelection, print_thresholds, metadata
    except Exception as exc:
        attempts.append(f"{candidate} public-package import: {exc!r}")

    data_file, beam_file = components
    try:
        data_module = _load_source_module(data_file, "data_loader")
        beam_module = _load_source_module(beam_file, "beam_selection")
        DataLoader = getattr(data_module, "DataLoader")
        BeamSelection = getattr(beam_module, "BeamSelection")
        print_thresholds = getattr(beam_module, "print_cherenkov_thresholds", None)
        metadata = {
            "module": "analysis_tools component sources",
            "import_mode": "direct_component_files",
            "repository_root": str(data_file.parent.parent),
            "resolved_from_candidate": str(candidate),
            "submodule_root": str(candidate),
            "source_policy": "required_repository_submodule",
            "data_loader_file": str(data_file.resolve()),
            "beam_selection_file": str(beam_file.resolve()),
            "data_loader_sha256": _sha256_file(data_file),
            "beam_selection_sha256": _sha256_file(beam_file),
            "public_package_import_bypassed": True,
            "reason": (
                "The package initializer imports unrelated optional subsystems; "
                "LicketyFit loaded the submodule's DataLoader/BeamSelection "
                "source modules directly."
            ),
        }
        return DataLoader, BeamSelection, print_thresholds, metadata
    except Exception as exc:
        attempts.append(f"{candidate} direct-component import: {exc!r}")

    raise AnalysisToolsImportError(
        "Could not import DataLoader/BeamSelection from the required "
        f"analysis_tools submodule at {candidate}. Ensure its Python "
        "dependencies are installed, or reinitialize it with "
        "'git submodule update --init analysis_tools'. Attempts:\n  - "
        + "\n  - ".join(attempts)
    )


def import_analysis_tools(
    *,
    explicit_path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> tuple[type, type, dict[str, Any]]:
    """Backward-compatible public resolver for DataLoader and BeamSelection."""
    DataLoader, BeamSelection, _print_thresholds, metadata = (
        _import_analysis_tools_components(
            explicit_path=explicit_path,
            project_root=project_root,
        )
    )
    return DataLoader, BeamSelection, metadata

def production_root_file(run: int, override: str | None = None) -> str:
    return str(override).strip() if override and str(override).strip() else (
        DEFAULT_PRODUCTION_TEMPLATE.format(run=int(run))
    )


def _integer_values(values: Any, *, label: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64).ravel()
    except Exception as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if array.size == 0:
        raise ValueError(f"{label} is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    rounded = np.rint(array)
    if not np.all(array == rounded):
        examples = array[array != rounded][:5].tolist()
        raise ValueError(f"{label} must be integer-valued; examples: {examples}")
    return rounded.astype(np.int64)


def _good_pmt_ids_from_payload(payload: Any) -> tuple[set[int], str]:
    """Normalize global IDs or ``[slot, position]`` pairs."""
    if isinstance(payload, Mapping):
        if "slots" in payload and "positions" in payload:
            slots = _integer_values(payload["slots"], label="good-PMT slots")
            positions = _integer_values(
                payload["positions"], label="good-PMT positions"
            )
            if slots.size != positions.size:
                raise ValueError("good-PMT slots and positions have different lengths")
            array = np.column_stack((slots, positions))
        else:
            selected = next((key for key in _GOOD_PMT_FILE_KEYS if key in payload), None)
            if selected is None:
                raise KeyError(
                    "Could not choose a good-PMT array; expected one of "
                    f"{list(_GOOD_PMT_FILE_KEYS)} or slots+positions, found "
                    f"{list(payload.keys())}"
                )
            payload = payload[selected]
            array = np.asarray(payload)
    else:
        array = np.asarray(payload)

    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 1:
        ids = _integer_values(array, label="good-PMT global IDs")
        layout = "global_pmt_id"
    elif array.ndim == 2 and array.shape[1] == 1:
        ids = _integer_values(array[:, 0], label="good-PMT global IDs")
        layout = "global_pmt_id"
    elif array.ndim == 2 and array.shape[1] == 2:
        slots = _integer_values(array[:, 0], label="good-PMT slots")
        positions = _integer_values(array[:, 1], label="good-PMT positions")
        if np.any(slots < 0):
            raise ValueError("good-PMT slots must be non-negative")
        if np.any((positions < 0) | (positions > 18)):
            raise ValueError("good-PMT positions must be in the inclusive range 0..18")
        ids = slots * 100 + positions
        layout = "slot_position"
    else:
        raise ValueError(
            "A good-PMT payload must be a one-column global-ID list or a "
            "two-column [mPMT slot, PMT position] table"
        )

    if np.any(ids < 0):
        raise ValueError("good-PMT global IDs must be non-negative")
    positions = np.mod(ids, 100)
    if np.any(positions > 18):
        examples = ids[positions > 18][:5].tolist()
        raise ValueError(
            "Global WCTE PMT IDs must use slot*100+position with position 0..18; "
            f"examples: {examples}"
        )
    good = set(int(value) for value in ids)
    if not good:
        raise ValueError("The good-PMT list is empty")
    return good, layout


def _text_good_pmt_payload(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    width: int | None = None
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        fields = text.replace(",", " ").split()
        try:
            values = [float(field) for field in fields]
        except ValueError:
            # Permit exactly one ordinary header before any numeric rows.
            if not rows:
                continue
            raise ValueError(
                f"Non-numeric good-PMT row at {path}:{line_number}: {raw!r}"
            )
        if len(values) not in {1, 2}:
            raise ValueError(
                f"Good-PMT row {line_number} must contain one global ID or "
                "two slot/position values"
            )
        if width is None:
            width = len(values)
        elif width != len(values):
            raise ValueError("Good-PMT text rows must all have the same column count")
        rows.append(values)
    if not rows:
        raise ValueError(f"Good-PMT text file has no numeric rows: {path}")
    return np.asarray(rows, dtype=np.float64)


def load_good_wcte_pmts_file(
    path: str | os.PathLike[str], *, key: str | None = None
) -> tuple[set[int], dict[str, Any]]:
    """Load a validated user good-PMT list from a small, safe data file."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    selected_key: str | None = None
    if suffix == ".npy":
        payload: Any = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(source, allow_pickle=False)
        try:
            files = list(archive.files)
            if key:
                selected_key = str(key)
                if selected_key not in files:
                    raise KeyError(
                        f"WCTE_GOOD_PMT_FILE_KEY={selected_key!r} is absent; "
                        f"available keys are {files}"
                    )
            else:
                selected_key = next(
                    (name for name in _GOOD_PMT_FILE_KEYS if name in files), None
                )
                if selected_key is None and len(files) == 1:
                    selected_key = files[0]
                if selected_key is None:
                    raise KeyError(
                        "Could not choose a good-PMT NPZ array; available keys "
                        f"are {files}. Set WCTE_GOOD_PMT_FILE_KEY."
                    )
            payload = np.asarray(archive[selected_key])
        finally:
            archive.close()
    elif suffix == ".json":
        payload = json.loads(source.read_text())
        if key:
            if not isinstance(payload, Mapping) or key not in payload:
                raise KeyError(f"WCTE_GOOD_PMT_FILE_KEY={key!r} is absent")
            payload = payload[key]
            selected_key = str(key)
    elif suffix in {".txt", ".csv"}:
        payload = _text_good_pmt_payload(source)
    else:
        raise ValueError(
            f"Unsupported WCTE_GOOD_PMT_FILE suffix {suffix!r}; use npy, npz, "
            "txt, csv, or json"
        )

    good, layout = _good_pmt_ids_from_payload(payload)
    stat = source.stat()
    return good, {
        "source_requested": "file",
        "source_resolved": "user_file",
        "file": str(source),
        "file_key": selected_key,
        "file_layout": layout,
        "file_size_bytes": int(stat.st_size),
        "file_sha256": _sha256_file(source),
        "good_pmt_count": int(len(good)),
    }


def _good_pmt_root_candidates(
    run: int,
    *,
    explicit_root_file: str | os.PathLike[str] | None = None,
    selection_root_file: str | os.PathLike[str] | None = None,
    search_bases: Sequence[str | os.PathLike[str]] = (),
) -> list[Path]:
    """Return bounded EOS candidates, preferring standalone DQ products."""
    candidates: list[Path] = []

    def add(value: str | os.PathLike[str] | None) -> None:
        if value is None or not str(value).strip():
            return
        path = Path(value).expanduser()
        if path not in candidates:
            candidates.append(path)

    if explicit_root_file and str(explicit_root_file).strip():
        add(explicit_root_file)
        return candidates

    bases = [Path(value).expanduser() for value in search_bases if str(value).strip()]
    for default in DEFAULT_GOOD_PMT_ROOT_SEARCH_BASES:
        path = Path(default)
        if path not in bases:
            bases.append(path)
    patterns = (
        f"dq_flags/{int(run)}/*.root",
        f"dq_flags/R{int(run)}/*.root",
        f"dq_flags/*{int(run)}*.root",
        f"dq_flags/**/*{int(run)}*.root",
        f"dq_flags/**/{int(run)}/*.root",
        f"production_v1_0/{int(run)}/*dq*.root",
        f"production_v1_0/{int(run)}/*quality*.root",
        f"*/{int(run)}/*dq*.root",
        f"*/R{int(run)}/*dq*.root",
    )
    for base in bases:
        if not base.exists():
            continue
        for pattern in patterns:
            try:
                for match in sorted(base.glob(pattern)):
                    if match.is_file():
                        add(match)
            except OSError:
                continue
    add(selection_root_file)
    add(production_root_file(int(run)))
    return candidates


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


def _optional_override(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} override must be finite or None")
    return number


def _equal_pdf_boundary(
    left_name: str,
    left_mean: float,
    left_sigma: float,
    right_name: str,
    right_mean: float,
    right_sigma: float,
) -> float:
    """Return the equal-density Gaussian boundary between adjacent peaks.

    A boundary is accepted only when the fitted means are ordered and the two
    densities cross between them. Silently falling back to an arbitrary
    midpoint would conceal a failed or incompatible run calibration.
    """
    numbers = (left_mean, left_sigma, right_mean, right_sigma)
    if not all(math.isfinite(float(value)) for value in numbers):
        raise RuntimeError(
            f"The {left_name}/{right_name} TOF calibration contains non-finite values"
        )
    left_mean = float(left_mean)
    right_mean = float(right_mean)
    left_sigma = float(left_sigma)
    right_sigma = float(right_sigma)
    if left_sigma <= 0.0 or right_sigma <= 0.0:
        raise RuntimeError(
            f"The {left_name}/{right_name} TOF widths must both be positive"
        )
    if left_mean >= right_mean:
        raise RuntimeError(
            f"The run TOF means are not ordered for {left_name}/{right_name}: "
            f"{left_mean:.6g} ns >= {right_mean:.6g} ns"
        )

    def log_density_difference(value: float) -> float:
        left = -math.log(left_sigma) - 0.5 * (
            (value - left_mean) / left_sigma
        ) ** 2
        right = -math.log(right_sigma) - 0.5 * (
            (value - right_mean) / right_sigma
        ) ** 2
        return left - right

    lo, hi = left_mean, right_mean
    f_lo, f_hi = log_density_difference(lo), log_density_difference(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        raise RuntimeError(
            f"The fitted {left_name}/{right_name} TOF Gaussians do not have an "
            "equal-density crossing between their means"
        )
    for _ in range(80):
        midpoint = 0.5 * (lo + hi)
        f_mid = log_density_difference(midpoint)
        if f_mid == 0.0:
            return midpoint
        if f_lo * f_mid > 0.0:
            lo, f_lo = midpoint, f_mid
        else:
            hi = midpoint
    return 0.5 * (lo + hi)


def _selection_thresholds(
    loader: Any, cfg: WCTESelectionConfig, particle: str
) -> dict[str, Any]:
    """Resolve only the run scalars required by the selected nominal PID.

    The production scalar record normally contains every key, but a proton-only
    or custom analysis must not fail because an unrelated ACT/muon-tagger scalar
    is absent. User overrides remove the corresponding metadata dependency.
    """
    values = loader.get_vme_analysis_scalar_results()
    mode = _canonical_selection_mode(cfg.selection_mode)
    tof_mode = _canonical_tof_mode(cfg.tof_cut_mode)
    light_pid_mode = _canonical_light_particle_pid_mode(
        cfg.light_particle_pid_mode
    )

    eveto_override = _optional_override(
        cfg.act_eveto_cut_override_pe, "act_eveto_cut_pe"
    )
    tagger_override = _optional_override(
        cfg.act_tagger_cut_override_pe, "act_tagger_cut_pe"
    )
    tof_override = _optional_override(
        cfg.proton_tof_cut_override_ns, "proton_tof_cut_ns"
    )
    mu_override = _optional_override(cfg.muon_tag_cut_override, "mu_tag_cut")
    electron_muon_override = _optional_override(
        cfg.electron_muon_tof_boundary_override_ns,
        "electron_muon_tof_boundary_ns",
    )
    muon_pion_override = _optional_override(
        cfg.muon_pion_tof_boundary_override_ns,
        "muon_pion_tof_boundary_ns",
    )

    def read_scalar(key: str, label: str, *, required: bool) -> float | None:
        try:
            return _scalar(values[key], label)
        except Exception:
            if required:
                raise RuntimeError(
                    f"The nominal {particle} selection requires {key!r} in "
                    "vme_analysis_scalar_results, but it is unavailable and no "
                    "corresponding override was supplied."
                )
            return None

    def read_first_scalar(
        keys: Sequence[str], label: str, *, required: bool
    ) -> tuple[float | None, str | None]:
        errors: list[str] = []
        for key in keys:
            try:
                return _scalar(values[key], label), key
            except Exception as exc:
                errors.append(f"{key}: {exc}")
        if required:
            choices = ", ".join(repr(key) for key in keys)
            raise RuntimeError(
                f"The nominal {particle} selection requires one of {choices} "
                f"for {label}, but none is usable. " + "; ".join(errors)
            )
        return None, None

    nominal = mode == "nominal"
    uses_act_identity = bool(
        particle in _LIGHT_PARTICLES and light_pid_mode in {"act", "act_tof"}
    )
    uses_light_tof = bool(
        nominal and particle in _LIGHT_PARTICLES and light_pid_mode != "act"
    )
    need_eveto = bool(
        nominal and uses_act_identity and cfg.use_act_eveto_cut
        and particle in {"muon", "pion", "electron"}
        and eveto_override is None
    )
    need_tagger = bool(
        nominal and uses_act_identity and cfg.use_act_tagger_cut
        and particle in {"muon", "pion"}
        and tagger_override is None
    )
    need_mu_tag = bool(
        nominal and particle == "muon" and cfg.require_muon_tagger
        and mu_override is None
    )
    need_tof = bool(
        nominal and tof_mode != "disable" and tof_override is None
        and (
            particle == "proton"
            or (particle == "pion" and uses_light_tof)
            or tof_mode == "require"
        )
    )

    raw_tof = read_scalar(
        "proton_tof_cut", "proton_tof_cut", required=need_tof
    )
    raw_eveto = read_scalar(
        "act_eveto_cut", "act_eveto_cut", required=need_eveto
    )
    raw_tagger = read_scalar(
        "act_tagger_cut", "act_tagger_cut", required=need_tagger
    )
    raw_mu_tag = read_scalar(
        "mu_tag_cut", "mu_tag_cut", required=need_mu_tag
    )

    if tof_mode == "disable":
        resolved_tof = None
        tof_source = "disabled"
    elif tof_override is not None:
        resolved_tof = float(tof_override)
        tof_source = "user_override"
    elif raw_tof is not None and math.isfinite(raw_tof) and raw_tof > 0.0:
        resolved_tof = float(raw_tof)
        tof_source = "vme_analysis_scalar_results"
    elif tof_mode == "require":
        raise RuntimeError(
            "WCTE TOF selection was required, but proton_tof_cut is zero/nonpositive "
            "or unavailable and no WCTE_PROTON_TOF_CUT_OVERRIDE_NS was supplied."
        )
    else:
        # The analysis-tools examples explicitly instruct users to omit the TOF
        # cut when proton_tof_cut is zero. Do not silently turn 999 ns into a
        # pseudo-threshold; record the omission explicitly.
        resolved_tof = None
        tof_source = "omitted_because_run_scalar_is_zero_or_unavailable"

    def resolve_value(
        raw: float | None, override: float | None, label: str, required: bool
    ) -> tuple[float | None, str]:
        if override is not None:
            return float(override), "user_override"
        if raw is not None and math.isfinite(raw):
            return float(raw), "vme_analysis_scalar_results"
        if required:
            raise RuntimeError(f"Required WCTE selection scalar {label!r} is unavailable")
        return None, "not_required_or_unavailable"

    eveto, eveto_source = resolve_value(
        raw_eveto, eveto_override, "act_eveto_cut", need_eveto
    )
    tagger, tagger_source = resolve_value(
        raw_tagger, tagger_override, "act_tagger_cut", need_tagger
    )
    mu_tag, mu_source = resolve_value(
        raw_mu_tag, mu_override, "mu_tag_cut", need_mu_tag
    )
    boundary_metadata: dict[str, Any] = {
        "raw_electron_muon_tof_boundary_ns": None,
        "raw_muon_pion_tof_boundary_ns": None,
        "electron_muon_tof_boundary_error": None,
        "muon_pion_tof_boundary_error": None,
        "electron_muon_tof_peak_separation_sigma": None,
        "muon_pion_tof_peak_separation_sigma": None,
        "tof_mean_electron_ns": None,
        "tof_std_electron_ns": None,
        "tof_mean_muon_ns": None,
        "tof_std_muon_ns": None,
        "tof_mean_pion_ns": None,
        "tof_std_pion_ns": None,
    }

    def resolve_light_boundary(
        *,
        boundary_name: str,
        left_name: str,
        right_name: str,
        explicit_keys: Sequence[str],
        override: float | None,
        required: bool,
    ) -> tuple[float | None, str]:
        if not required:
            return None, "not_required"
        if override is not None:
            return float(override), "user_override"

        explicit, explicit_key = read_first_scalar(
            explicit_keys, boundary_name, required=False
        )
        boundary_metadata[f"raw_{boundary_name}_ns"] = explicit
        if explicit is not None:
            return float(explicit), f"vme_analysis_scalar_results:{explicit_key}"

        try:
            left_mean, left_mean_key = read_first_scalar(
                (f"tof_mean_{left_name}", f"{left_name}_tof_mean"),
                f"{left_name} TOF mean",
                required=True,
            )
            left_sigma, left_sigma_key = read_first_scalar(
                (f"tof_std_{left_name}", f"{left_name}_tof_std"),
                f"{left_name} TOF width",
                required=True,
            )
            right_mean, right_mean_key = read_first_scalar(
                (f"tof_mean_{right_name}", f"{right_name}_tof_mean"),
                f"{right_name} TOF mean",
                required=True,
            )
            right_sigma, right_sigma_key = read_first_scalar(
                (f"tof_std_{right_name}", f"{right_name}_tof_std"),
                f"{right_name} TOF width",
                required=True,
            )
            boundary_metadata[f"tof_mean_{left_name}_ns"] = left_mean
            boundary_metadata[f"tof_std_{left_name}_ns"] = left_sigma
            boundary_metadata[f"tof_mean_{right_name}_ns"] = right_mean
            boundary_metadata[f"tof_std_{right_name}_ns"] = right_sigma
            pooled_sigma = math.sqrt(
                0.5 * (float(left_sigma) ** 2 + float(right_sigma) ** 2)
            )
            if pooled_sigma > 0.0:
                boundary_metadata[
                    f"{left_name}_{right_name}_tof_peak_separation_sigma"
                ] = abs(float(right_mean) - float(left_mean)) / pooled_sigma
            boundary = _equal_pdf_boundary(
                left_name,
                float(left_mean),
                float(left_sigma),
                right_name,
                float(right_mean),
                float(right_sigma),
            )
        except Exception as exc:
            message = str(exc)
            boundary_metadata[f"{boundary_name}_error"] = message
            if light_pid_mode == "tof":
                raise RuntimeError(
                    f"TOF-only {particle} PID cannot construct the required "
                    f"{left_name}/{right_name} boundary for run {cfg.run}: "
                    f"{message}. Use LIGHT_PARTICLE_PID_MODE='act_tof' to "
                    "retain ACT identity and fall back only for this unusable "
                    "boundary, or supply a validated boundary override."
                ) from exc
            return None, "unavailable_run_calibration"
        scalar_keys = ",".join(
            (
                str(left_mean_key),
                str(left_sigma_key),
                str(right_mean_key),
                str(right_sigma_key),
            )
        )
        return boundary, f"equal_pdf_intersection:{scalar_keys}"

    need_electron_muon_boundary = bool(
        uses_light_tof and particle in {"electron", "muon"}
    )
    need_muon_pion_boundary = bool(
        uses_light_tof and particle in {"muon", "pion"}
    )
    electron_muon_boundary, electron_muon_source = resolve_light_boundary(
        boundary_name="electron_muon_tof_boundary",
        left_name="electron",
        right_name="muon",
        explicit_keys=(
            "tof_cut_electron_muon",
            "electron_muon_tof_cut",
            "muon_tof_cut",
        ),
        override=electron_muon_override,
        required=need_electron_muon_boundary,
    )
    muon_pion_boundary, muon_pion_source = resolve_light_boundary(
        boundary_name="muon_pion_tof_boundary",
        left_name="muon",
        right_name="pion",
        explicit_keys=(
            "tof_cut_muon_pion",
            "muon_pion_tof_cut",
            "pion_tof_cut",
        ),
        override=muon_pion_override,
        required=need_muon_pion_boundary,
    )
    if (
        electron_muon_boundary is not None
        and muon_pion_boundary is not None
        and electron_muon_boundary >= muon_pion_boundary
    ):
        message = (
            "The resolved light-particle TOF boundaries are not ordered: "
            f"electron/muon={electron_muon_boundary:.6g} ns, "
            f"muon/pion={muon_pion_boundary:.6g} ns"
        )
        if (
            light_pid_mode == "tof"
            or electron_muon_override is not None
            or muon_pion_override is not None
        ):
            raise RuntimeError(message)
        boundary_metadata["electron_muon_tof_boundary_error"] = message
        boundary_metadata["muon_pion_tof_boundary_error"] = message
        electron_muon_boundary = None
        muon_pion_boundary = None
        electron_muon_source = "unavailable_run_calibration"
        muon_pion_source = "unavailable_run_calibration"
    if (
        particle == "pion"
        and uses_light_tof
        and resolved_tof is not None
        and muon_pion_boundary is not None
        and muon_pion_boundary >= resolved_tof
    ):
        message = (
            "The resolved muon/pion TOF boundary must be below proton_tof_cut: "
            f"{muon_pion_boundary:.6g} ns >= {resolved_tof:.6g} ns"
        )
        if light_pid_mode == "tof" or muon_pion_override is not None:
            raise RuntimeError(message)
        boundary_metadata["muon_pion_tof_boundary_error"] = message
        muon_pion_boundary = None
        muon_pion_source = "unavailable_run_calibration"

    unavailable_boundaries = []
    if need_electron_muon_boundary and electron_muon_boundary is None:
        unavailable_boundaries.append("electron/muon")
    if need_muon_pion_boundary and muon_pion_boundary is None:
        unavailable_boundaries.append("muon/pion")
    return {
        "light_particle_pid_mode": light_pid_mode,
        "raw_proton_tof_cut_ns": raw_tof,
        "raw_act_eveto_cut_pe": raw_eveto,
        "raw_act_tagger_cut_pe": raw_tagger,
        "raw_mu_tag_cut": raw_mu_tag,
        "proton_tof_cut_ns": resolved_tof,
        "proton_tof_cut_source": tof_source,
        "act_eveto_cut_pe": eveto,
        "act_eveto_cut_source": eveto_source,
        "act_tagger_cut_pe": tagger,
        "act_tagger_cut_source": tagger_source,
        "mu_tag_cut": mu_tag,
        "mu_tag_cut_source": mu_source,
        "electron_muon_tof_boundary_ns": electron_muon_boundary,
        "electron_muon_tof_boundary_source": electron_muon_source,
        "muon_pion_tof_boundary_ns": muon_pion_boundary,
        "muon_pion_tof_boundary_source": muon_pion_source,
        "light_particle_tof_fallback_used": bool(
            light_pid_mode == "act_tof" and unavailable_boundaries
        ),
        "light_particle_tof_unavailable_boundaries": unavailable_boundaries,
        **boundary_metadata,
    }


def _selection_specs(
    particle: str, cuts: Mapping[str, Any], cfg: WCTESelectionConfig
) -> list[list[Any]]:
    mode = _canonical_selection_mode(cfg.selection_mode)
    extra = [_normalise_cut_spec(spec) for spec in cfg.extra_selection_cuts]
    if mode == "custom":
        if not extra:
            raise ValueError(
                "WCTE selection_mode='custom' requires at least one item in "
                "WCTE_EXTRA_SELECTION_CUTS."
            )
        return extra

    if particle not in _NOMINAL_PARTICLES:
        raise ValueError(
            f"analysis_tools defines no nominal {particle!r} BeamSelection in the "
            "supplied examples. Use WCTE_SELECTION_MODE='custom' and provide "
            "WCTE_EXTRA_SELECTION_CUTS explicitly."
        )
    if cfg.require_muon_tagger and particle != "muon":
        raise ValueError(
            "WCTE_REQUIRE_MUON_TAGGER is only part of the nominal muon selection. "
            "For another particle, express the desired cut explicitly in "
            "WCTE_EXTRA_SELECTION_CUTS."
        )

    eveto = cuts.get("act_eveto_cut_pe")
    tagger = cuts.get("act_tagger_cut_pe")
    tof = cuts.get("proton_tof_cut_ns")
    mu_tag = cuts.get("mu_tag_cut")
    electron_muon_tof = cuts.get("electron_muon_tof_boundary_ns")
    muon_pion_tof = cuts.get("muon_pion_tof_boundary_ns")
    light_pid_mode = _canonical_light_particle_pid_mode(
        cfg.light_particle_pid_mode
    )
    uses_act_identity = light_pid_mode in {"act", "act_tof"}
    specs: list[list[Any]] = []

    if particle == "muon":
        if uses_act_identity and cfg.use_act_eveto_cut:
            specs.append(["vme_act_eveto", "<", float(eveto)])
        if uses_act_identity and cfg.use_act_tagger_cut:
            specs.append(["vme_act_tagger", ">", float(tagger)])
        if light_pid_mode == "tof":
            if electron_muon_tof is None or muon_pion_tof is None:
                raise RuntimeError(
                    "TOF-only muon PID requires both the electron/muon and "
                    "muon/pion TOF boundaries"
                )
            specs.append([
                "vme_tof_corr",
                "between",
                [float(electron_muon_tof), float(muon_pion_tof)],
            ])
        elif light_pid_mode == "act_tof":
            lower = electron_muon_tof
            upper = muon_pion_tof if muon_pion_tof is not None else tof
            if lower is not None and upper is not None:
                if float(lower) >= float(upper):
                    raise RuntimeError(
                        "The nominal muon TOF bounds are not ordered: "
                        f"{float(lower):.6g} ns >= {float(upper):.6g} ns"
                    )
                specs.append([
                    "vme_tof_corr",
                    "between",
                    [float(lower), float(upper)],
                ])
            elif lower is not None:
                specs.append(["vme_tof_corr", ">", float(lower)])
            elif upper is not None:
                specs.append(["vme_tof_corr", "<", float(upper)])
        elif tof is not None:
            specs.append(["vme_tof_corr", "<", float(tof)])
        if cfg.require_muon_tagger:
            specs.append(["vme_mu_tag_total", ">", float(mu_tag)])
    elif particle == "pion":
        if uses_act_identity and cfg.use_act_eveto_cut:
            specs.append(["vme_act_eveto", "<", float(eveto)])
        if uses_act_identity and cfg.use_act_tagger_cut:
            specs.append(["vme_act_tagger", "<", float(tagger)])
        if light_pid_mode == "tof":
            if muon_pion_tof is None:
                raise RuntimeError(
                    "TOF-only pion PID requires the muon/pion TOF boundary"
                )
            if tof is None:
                raise RuntimeError(
                    "TOF-only pion PID requires a "
                    "positive proton_tof_cut. Use WCTE_TOF_CUT_MODE='auto' or "
                    "'require' with a valid scalar/override, or select "
                    "LIGHT_PARTICLE_PID_MODE='act'."
                )
            specs.append([
                "vme_tof_corr",
                "between",
                [float(muon_pion_tof), float(tof)],
            ])
        elif light_pid_mode == "act_tof":
            if muon_pion_tof is not None and tof is not None:
                specs.append([
                    "vme_tof_corr",
                    "between",
                    [float(muon_pion_tof), float(tof)],
                ])
            elif muon_pion_tof is not None:
                specs.append(["vme_tof_corr", ">", float(muon_pion_tof)])
            elif tof is not None:
                specs.append(["vme_tof_corr", "<", float(tof)])
        elif tof is not None:
            specs.append(["vme_tof_corr", "<", float(tof)])
    elif particle == "electron":
        if uses_act_identity and cfg.use_act_eveto_cut:
            specs.append(["vme_act_eveto", ">", float(eveto)])
        # The nominal example does not use act_tagger for electrons.
        if light_pid_mode == "tof":
            if electron_muon_tof is None:
                raise RuntimeError(
                    "TOF-only electron PID requires the electron/muon TOF boundary"
                )
            specs.append(["vme_tof_corr", "<", float(electron_muon_tof)])
        elif light_pid_mode == "act_tof":
            upper = electron_muon_tof if electron_muon_tof is not None else tof
            if upper is not None:
                specs.append(["vme_tof_corr", "<", float(upper)])
        elif tof is not None:
            specs.append(["vme_tof_corr", "<", float(tof)])
    elif particle == "proton":
        if tof is None:
            raise RuntimeError(
                "The nominal proton selection requires a positive proton_tof_cut. "
                "This run reports no TOF separation. Supply "
                "WCTE_PROTON_TOF_CUT_OVERRIDE_NS, or use a custom selection."
            )
        window = float(cfg.proton_tof_window_ns)
        if not math.isfinite(window) or window <= 0.0:
            raise ValueError("WCTE_PROTON_TOF_WINDOW_NS must be positive")
        specs.append(["vme_tof_corr", "between", [float(tof), float(tof) + window]])

    specs.extend(extra)
    if not specs:
        raise ValueError(
            f"The nominal {particle} selection contains no enabled cuts. Enable a "
            "nominal cut or provide WCTE_EXTRA_SELECTION_CUTS."
        )
    return [_normalise_cut_spec(spec) for spec in specs]


def _build_selection(BeamSelection: type, particle: str, specs: Sequence[Sequence[Any]]):
    return BeamSelection.selection(particle, *[list(spec) for spec in specs])


def _configure_dq(loader: Any, cfg: WCTESelectionConfig) -> None:
    if cfg.apply_mpmt_data_quality_cuts:
        loader.apply_mPMT_data_quality_cuts()
    if cfg.apply_vme_event_quality_cuts:
        loader.apply_vme_event_quality_cuts()
    if cfg.apply_t5_event_quality_cuts:
        loader.apply_t5_event_quality_cuts()


def _available_branches(DataLoader: type, root_file: str) -> set[str]:
    probe = DataLoader(root_file, branches_to_load=[])
    try:
        return set(str(x) for x in probe.file["WCTEReadoutWindows"].keys())
    finally:
        probe.file.close()


def _loader_with_branches(
    DataLoader: type,
    cfg: WCTESelectionConfig,
    *,
    selection_specs: Sequence[Sequence[Any]],
    require_t5_time: bool,
    available_branches: set[str] | None = None,
) -> tuple[Any, list[str], set[str]]:
    available = (
        _available_branches(DataLoader, cfg.root_file)
        if available_branches is None
        else set(available_branches)
    )
    required = list(_BASE_HIT_BRANCHES)
    required.extend(_selection_branch_name(str(spec[0])) for spec in selection_specs)
    if require_t5_time:
        required.append("T5_hit_time")
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


def _selection_scan_branches(
    available: set[str],
    cfg: WCTESelectionConfig,
    selection_specs: Sequence[Sequence[Any]],
) -> list[str]:
    """Return the event-level branches needed to identify selected entries.

    PMT hit arrays are deliberately absent.  In particular,
    ``hit_pmt_readout_mask`` is a hit-level mask: it can remove individual hits
    but cannot change whether an event passes the beam/DQ selection.  Reading
    every jagged hit array while looking for a small selected population makes
    remote production ROOT files unnecessarily expensive to scan.
    """
    required = [_selection_branch_name(str(spec[0])) for spec in selection_specs]
    if cfg.apply_mpmt_data_quality_cuts:
        required.append("window_data_quality_mask")
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
    return required


def _event_level_dq_mask(batch: Any, cfg: WCTESelectionConfig) -> Any:
    """Reproduce DataLoader's event-level DQ mask without touching hit arrays."""
    mask: Any = np.ones(len(batch), dtype=bool)
    if cfg.apply_mpmt_data_quality_cuts:
        mask = mask & (batch["window_data_quality_mask"] == 0)
    if cfg.apply_vme_event_quality_cuts:
        mask = mask & (
            (batch["vme_digi_issues_bitmask"] == 0)
            & (batch["vme_evt_quality_bitmask"] == 0)
        )
    if cfg.apply_t5_event_quality_cuts:
        mask = mask & (
            (batch["T5_HasValidHit"] == True)  # noqa: E712 - Awkward elementwise comparison
            & (batch["T5_HasMultipleScintillatorsHit"] == False)  # noqa: E712
            & (batch["T5_HasInTimeWindow"] == True)  # noqa: E712
        )
    return mask


def _scan_selected_root_indices(
    loader: Any,
    selection: Any,
    cfg: WCTESelectionConfig,
    *,
    branches: Sequence[str],
    maximum: int | None,
) -> list[int]:
    """Scan only scalar event fields and return original selected ROOT entries."""
    try:
        import awkward as ak
    except Exception as exc:
        raise RuntimeError(
            "Indexed WCTE DataLoader selection requires awkward, which is also "
            "a dependency of analysis_tools.DataLoader."
        ) from exc

    tree = loader.file["WCTEReadoutWindows"]
    kwargs: dict[str, Any] = {
        "expressions": list(branches),
        "step_size": cfg.step_size,
        "library": "ak",
    }
    if cfg.max_root_entries is not None:
        kwargs["entry_stop"] = max(0, int(cfg.max_root_entries))

    selected_indices: list[int] = []
    raw_entry_cursor = 0
    for raw_batch in tree.iterate(**kwargs):
        raw_length = int(len(raw_batch))
        if raw_length == 0:
            continue
        raw_indices = np.arange(
            raw_entry_cursor, raw_entry_cursor + raw_length, dtype=np.int64
        )
        raw_entry_cursor += raw_length
        event_dq = _event_level_dq_mask(raw_batch, cfg)
        kept = raw_batch[event_dq]
        kept_indices = raw_indices[np.asarray(ak.to_numpy(event_dq), dtype=bool)]
        if len(kept) == 0:
            continue
        selected = np.asarray(
            ak.to_numpy(selection.mask(kept)), dtype=bool
        ).ravel()
        selected_indices.extend(int(x) for x in kept_indices[selected])
        if maximum is not None and len(selected_indices) >= int(maximum):
            return selected_indices[: int(maximum)]
    return selected_indices


def _load_selected_records_by_index(
    loader: Any,
    cfg: WCTESelectionConfig,
    root_indices: Sequence[int],
) -> Iterable[tuple[Any, int]]:
    """Load full hit payloads only for already-selected ROOT entries.

    DataLoader's complete DQ implementation is reapplied as an exactness check.
    This both performs ``hit_pmt_readout_mask`` filtering and protects the
    two-phase scan against future changes in the pinned DataLoader semantics.
    """
    apply_dq = getattr(loader, "_apply_all_data_quality_cuts", None)
    if not callable(apply_dq):
        raise RuntimeError(
            "The analysis_tools DataLoader lacks _apply_all_data_quality_cuts; "
            "the indexed WCTE loading contract cannot be verified."
        )
    tree = loader.file["WCTEReadoutWindows"]
    for root_index in root_indices:
        index = int(root_index)
        raw = tree.arrays(
            expressions=loader.branches_to_load,
            entry_start=index,
            entry_stop=index + 1,
            library="ak",
        )
        checked = apply_dq(raw, False)
        if len(checked) != 1:
            raise RuntimeError(
                "A WCTE event passed the scalar DQ scan but failed the pinned "
                f"DataLoader's full DQ pass at ROOT entry {index}."
            )
        yield checked[0], index


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
    """Yield selected batches together with original ROOT entry indices."""
    try:
        import awkward as ak
    except Exception as exc:
        raise RuntimeError(
            "Indexed WCTE DataLoader iteration requires awkward, which is also "
            "a dependency of analysis_tools.DataLoader."
        ) from exc
    apply_dq = getattr(loader, "_apply_all_data_quality_cuts", None)
    if not callable(apply_dq):
        raise RuntimeError(
            "The analysis_tools submodule's DataLoader does not provide the "
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
            raw_entry_cursor, raw_entry_cursor + raw_length, dtype=np.int64
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
                "DataLoader selection returned a ROOT-index array with the wrong length."
            )
        yield selected, indices


def _estimate_optional_peaks(
    DataLoader: type,
    cfg: WCTESelectionConfig,
    selection: Any,
    selection_specs: Sequence[Sequence[Any]],
) -> tuple[float | None, float | None, dict[str, int]]:
    if not (cfg.use_t5_hit_time_cut or cfg.use_calibrated_peak_time_cut):
        return None, None, {"events": 0, "hits": 0}
    loader, _, _ = _loader_with_branches(
        DataLoader,
        cfg,
        selection_specs=selection_specs,
        require_t5_time=cfg.use_t5_hit_time_cut,
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


def _try_run_info(loader: Any) -> tuple[Any | None, dict[str, float] | None]:
    try:
        record = loader.get_vme_analysis_run_info()
        fields = set(getattr(record, "fields", ()))
        metadata = {
            key: _scalar(record[key], key)
            for key in ("run_momentum", "n_eveto", "n_tagger")
            if key in fields
        }
        return record, metadata
    except Exception:
        return None, None


def load_selected_events(
    config: WCTESelectionConfig,
    *,
    project_root: str | os.PathLike[str] | None = None,
    return_metadata: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict[str, Any]]:
    """Load beam-selected real-WCTE events through ``analysis_tools``."""
    particle = _canonical_particle(config.particle)
    mode = _canonical_selection_mode(config.selection_mode)
    tof_mode = _canonical_tof_mode(config.tof_cut_mode)
    light_pid_mode = _canonical_light_particle_pid_mode(
        config.light_particle_pid_mode
    )
    normalised_extra = tuple(
        tuple(_normalise_cut_spec(spec)) for spec in config.extra_selection_cuts
    )
    cfg = WCTESelectionConfig(**{
        **asdict(config),
        "particle": particle,
        "selection_mode": mode,
        "tof_cut_mode": tof_mode,
        "light_particle_pid_mode": light_pid_mode,
        "extra_selection_cuts": normalised_extra,
    })
    DataLoader, BeamSelection, print_thresholds, import_metadata = (
        _import_analysis_tools_components(
            explicit_path=cfg.analysis_tools_path,
            project_root=project_root,
        )
    )

    threshold_loader = DataLoader(cfg.root_file, branches_to_load=[])
    try:
        available = set(
            str(x) for x in threshold_loader.file["WCTEReadoutWindows"].keys()
        )
        if mode == "nominal":
            cuts = _selection_thresholds(threshold_loader, cfg, particle)
        else:
            # Custom selection has no hidden dependency on the scalar-results
            # tree. Retain optional scalar metadata when available.
            try:
                cuts = _selection_thresholds(threshold_loader, cfg, particle)
            except Exception as exc:
                cuts = {"custom_selection_scalar_metadata_error": repr(exc)}
        run_info_record, run_info = _try_run_info(threshold_loader)
    finally:
        threshold_loader.file.close()

    specs = _selection_specs(particle, cuts, cfg)
    selection = _build_selection(BeamSelection, particle, specs)

    unavailable_boundaries = cuts.get(
        "light_particle_tof_unavailable_boundaries", ()
    )
    if cfg.verbose and unavailable_boundaries:
        joined = ", ".join(str(item) for item in unavailable_boundaries)
        print(
            f"WCTE PID fallback: run {cfg.run} has no usable {joined} TOF "
            "boundary; ACT identity and the remaining valid TOF cuts will be "
            "used."
        )
        print("")

    if cfg.verbose and cfg.print_cherenkov_thresholds and run_info_record is not None:
        try:
            if print_thresholds is None:
                raise RuntimeError("analysis_tools beam_selection module has no threshold printer")
            print("WCTE ACT Cherenkov-threshold diagnostic")
            print_thresholds(run_info_record)
            print("")
        except Exception as exc:
            print(f"WARNING: could not print ACT Cherenkov thresholds: {exc!r}")
    if cfg.verbose and cfg.print_selection_description:
        selection.describe()
        print("")

    t5_peak, calibrated_peak, peak_stats = _estimate_optional_peaks(
        DataLoader, cfg, selection, specs
    )
    scan_branches = _selection_scan_branches(available, cfg, specs)
    scan_loader = DataLoader(cfg.root_file, branches_to_load=list(scan_branches))
    # The pinned DataLoader eagerly appends hit-level DQ branches whenever a
    # branch list is supplied.  The scalar pass intentionally does not need
    # those arrays; full DataLoader DQ is reapplied to each selected event below.
    scan_loader.branches_to_load = list(scan_branches)
    try:
        scan_limit = (
            None
            if (cfg.use_t5_hit_time_cut or cfg.use_calibrated_peak_time_cut)
            else cfg.max_selected_events
        )
        selected_root_indices = _scan_selected_root_indices(
            scan_loader,
            selection,
            cfg,
            branches=scan_branches,
            maximum=scan_limit,
        )
    finally:
        scan_loader.file.close()

    loader, branches, available = _loader_with_branches(
        DataLoader,
        cfg,
        selection_specs=specs,
        require_t5_time=cfg.use_t5_hit_time_cut,
        available_branches=available,
    )

    events: list[np.ndarray] = []
    selected_before_hit_cut = len(selected_root_indices)
    empty_after_hit_cut = 0
    fallback_id = 0
    try:
        for record, root_entry_index in _load_selected_records_by_index(
            loader, cfg, selected_root_indices
        ):
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
    finally:
        loader.file.close()

    metadata: dict[str, Any] = {
        "adapter": "LicketyFit.scripts.wcte_data_loader_adapter",
        "analysis_tools": import_metadata,
        "config": asdict(cfg),
        "root_file": cfg.root_file,
        "particle": particle,
        "selection_mode": mode,
        "light_particle_pid_mode": light_pid_mode,
        "light_particle_pid_mode_applied": (
            light_pid_mode
            if mode == "nominal" and particle in _LIGHT_PARTICLES
            else None
        ),
        "selection_cut_specs": [list(spec) for spec in specs],
        "selection_cut_branches": [_selection_branch_name(spec[0]) for spec in specs],
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
        "root_loading_strategy": "event_scalar_scan_then_selected_hit_payloads",
        "selection_scan_branches": list(scan_branches),
        "root_entry_limit_semantics": (
            "scalar WCTEReadoutWindows.iterate(entry_stop=max_root_entries), "
            "followed by the pinned DataLoader event-level DQ predicates and "
            "BeamSelection; full DataLoader DQ and hit-readout masking are then "
            "reapplied only to selected ROOT entries"
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
        print("  particle label:", particle)
        print("  selection mode:", mode)
        print("  raw entry limit:", cfg.max_root_entries)
        print("  selected-event cap:", cfg.max_selected_events)
        print("  DQ cuts (mPMT/VME/T5):", (
            cfg.apply_mpmt_data_quality_cuts,
            cfg.apply_vme_event_quality_cuts,
            cfg.apply_t5_event_quality_cuts,
        ))
        print("  applied PID cuts:", specs)
        print("  PMT-minus-T5 peak [ns]:", t5_peak)
        print("  calibrated-time peak [ns]:", calibrated_peak)
        print("  selected / returned / empty:", (
            selected_before_hit_cut, len(events), empty_after_hit_cut
        ))
        print("  analysis_tools source:", import_metadata.get("repository_root"))

    return (events, metadata) if return_metadata else events


def _unwrap_single_configuration_entry(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1 and isinstance(
        value[0], (list, tuple, Mapping)
    ):
        value = value[0]
    return value


def _mapping_good_pmt_payload(value: Mapping[str, Any]) -> Any:
    simplified = {
        str(key).split("/")[-1].split(".")[-1].lower(): payload
        for key, payload in value.items()
    }
    slot_key = next(
        (key for key in simplified if "slot" in key or key in {"first", "ffirst"}),
        None,
    )
    position_key = next(
        (
            key for key in simplified
            if "position" in key or key in {"second", "fsecond", "pmt"}
        ),
        None,
    )
    if slot_key is not None and position_key is not None:
        return {
            "slots": simplified[slot_key],
            "positions": simplified[position_key],
        }
    if len(value) == 1:
        return next(iter(value.values()))
    return value


def _load_good_wcte_pmts_direct_uproot(
    root_file: str | os.PathLike[str],
) -> tuple[set[int], dict[str, Any]]:
    """Read the Configuration mask without requiring an event TTree.

    This path is needed for standalone ``dq_flags`` products, which contain the
    Configuration tree but deliberately do not contain WCTEReadoutWindows.
    """
    try:
        import awkward as ak
        import uproot
    except Exception as exc:
        raise RuntimeError(
            "Direct standalone-DQ good-PMT loading requires uproot and awkward"
        ) from exc

    with uproot.open(str(root_file)) as root:
        try:
            configuration = root["Configuration"]
        except Exception as exc:
            raise RuntimeError(
                f"ROOT file {root_file!r} has no Configuration object"
            ) from exc
        keys = [str(key).split(";")[0] for key in configuration.keys()]
        relevant = [key for key in keys if "good_wcte_pmts" in key.lower()]
        if not relevant:
            raise RuntimeError(
                "Configuration contains no branch matching good_wcte_pmts; "
                f"available branches are {keys}"
            )

        errors: list[str] = []
        # A single unsplit vector branch commonly becomes a list of global IDs,
        # slot/position pairs, or first/second records after ``ak.to_list``.
        unsplit = [
            key for key in relevant
            if key.split("/")[-1].split(".")[-1].lower() == "good_wcte_pmts"
        ]
        for key in unsplit:
            try:
                payload = ak.to_list(configuration[key].array(entry_stop=1))
                payload = _unwrap_single_configuration_entry(payload)
                if isinstance(payload, Mapping):
                    payload = _mapping_good_pmt_payload(payload)
                elif isinstance(payload, list) and payload and isinstance(
                    payload[0], Mapping
                ):
                    rows = [_mapping_good_pmt_payload(row) for row in payload]
                    if all(
                        isinstance(row, Mapping)
                        and "slots" in row and "positions" in row
                        for row in rows
                    ):
                        payload = [
                            [row["slots"], row["positions"]] for row in rows
                        ]
                good, layout = _good_pmt_ids_from_payload(payload)
                return good, {
                    "source_resolved": "run_root",
                    "root_file": str(root_file),
                    "root_loader": "direct_uproot_configuration",
                    "configuration_branch": key,
                    "configuration_layout": layout,
                    "good_pmt_count": int(len(good)),
                }
            except Exception as exc:
                errors.append(f"{key}: {exc!r}")

        # Split C++ record branches are easier to interpret together.
        try:
            arrays = configuration.arrays(relevant, entry_stop=1, library="ak")
            payload = ak.to_list(arrays)
            payload = _unwrap_single_configuration_entry(payload)
            if isinstance(payload, Mapping):
                payload = _mapping_good_pmt_payload(payload)
            good, layout = _good_pmt_ids_from_payload(payload)
            return good, {
                "source_resolved": "run_root",
                "root_file": str(root_file),
                "root_loader": "direct_uproot_configuration",
                "configuration_branches": relevant,
                "configuration_layout": layout,
                "good_pmt_count": int(len(good)),
            }
        except Exception as exc:
            errors.append(f"combined branches: {exc!r}")
    raise RuntimeError(
        "Could not decode Configuration/good_wcte_pmts with direct Uproot:\n  - "
        + "\n  - ".join(errors)
    )


def load_good_wcte_pmts(
    root_file: str,
    *,
    analysis_tools_path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> tuple[set[int], dict[str, Any]]:
    """Read ``Configuration/good_wcte_pmts`` from merged or DQ ROOT."""
    # Resolving the pinned submodule is mandatory even when the standalone DQ
    # file later needs the direct-Configuration Uproot fallback. This prevents a
    # missing or profile-local analysis_tools checkout from being hidden.
    DataLoader, _, import_metadata = import_analysis_tools(
        explicit_path=analysis_tools_path,
        project_root=project_root,
    )
    try:
        loader = DataLoader(str(root_file), branches_to_load=[])
        try:
            slots, positions = loader.get_good_wcte_pmts()
            slots = _to_numpy_1d(slots, np.int64)
            positions = _to_numpy_1d(positions, np.int64)
            if slots.size != positions.size:
                raise RuntimeError(
                    "DataLoader returned misaligned good-PMT slot/position arrays"
                )
            ids = slots * 100 + positions
            good = set(int(value) for value in ids)
            if not good:
                raise RuntimeError("DataLoader Configuration/good_wcte_pmts is empty")
        finally:
            loader.file.close()
        return good, {
            "source_resolved": "run_root",
            "root_file": str(root_file),
            "root_loader": "analysis_tools.DataLoader.get_good_wcte_pmts",
            "good_pmt_count": int(len(good)),
            "analysis_tools": import_metadata,
        }
    except Exception as data_loader_error:
        good, metadata = _load_good_wcte_pmts_direct_uproot(root_file)
        metadata["analysis_tools"] = import_metadata
        metadata["analysis_tools_loader_error"] = (
            f"{type(data_loader_error).__name__}: "
            + str(data_loader_error).splitlines()[0]
        )
        return good, metadata


def resolve_good_wcte_pmts(
    *,
    source: str,
    run: int,
    good_pmt_file: str | os.PathLike[str] | None = None,
    good_pmt_file_key: str | None = None,
    good_pmt_root_file: str | os.PathLike[str] | None = None,
    selection_root_file: str | os.PathLike[str] | None = None,
    root_search_bases: Sequence[str | os.PathLike[str]] = (),
    analysis_tools_path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> tuple[set[int], dict[str, Any]]:
    """Resolve a user mask or a run-derived ROOT mask with full provenance.

    ``source='auto'`` selects the user file when one is supplied and otherwise
    performs run-root discovery. Geometry-only and missing-mask fallbacks are
    deliberately rejected: real-WCTE fitting requires an authoritative user or
    run/DQ list.
    """
    requested = str(source).strip().lower().replace("-", "_") or "auto"
    aliases = {
        "user": "file",
        "user_file": "file",
        "mask": "file",
        "root": "run",
        "run_root": "run",
        "dq": "run",
        "dq_root": "run",
    }
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "file", "run"}:
        raise ValueError(
            "WCTE_GOOD_PMT_SOURCE must be auto, file, or run; real-WCTE fits "
            "require an authoritative user or run/DQ good-PMT list"
        )
    resolved_request = requested
    if requested == "auto":
        resolved_request = "file" if good_pmt_file and str(good_pmt_file).strip() else "run"
    if resolved_request == "file":
        if not good_pmt_file or not str(good_pmt_file).strip():
            raise ValueError(
                "WCTE_GOOD_PMT_SOURCE=file requires WCTE_GOOD_PMT_FILE"
            )
        good, metadata = load_good_wcte_pmts_file(
            good_pmt_file, key=good_pmt_file_key
        )
        metadata.update({
            "source_requested": requested,
            "run": int(run),
        })
        return good, metadata

    candidates = _good_pmt_root_candidates(
        int(run),
        explicit_root_file=good_pmt_root_file,
        selection_root_file=selection_root_file,
        search_bases=root_search_bases,
    )
    attempts: list[dict[str, str]] = []
    for candidate in candidates:
        if not candidate.is_file():
            attempts.append({"path": str(candidate), "status": "not_found"})
            continue
        try:
            good, metadata = load_good_wcte_pmts(
                str(candidate),
                analysis_tools_path=analysis_tools_path,
                project_root=project_root,
            )
        except Exception as exc:
            attempts.append({
                "path": str(candidate),
                "status": "rejected",
                "reason": repr(exc),
            })
            continue
        attempts.append({"path": str(candidate), "status": "selected"})
        metadata.update({
            "source_requested": requested,
            "source_resolved": "run_root",
            "run": int(run),
            "root_discovery_candidates": attempts,
            "explicit_root_requested": (
                str(good_pmt_root_file)
                if good_pmt_root_file and str(good_pmt_root_file).strip()
                else None
            ),
        })
        return good, metadata
    detail = "\n  - ".join(
        f"{row['path']}: {row['status']}"
        + (f" ({row['reason']})" if "reason" in row else "")
        for row in attempts
    )
    raise RuntimeError(
        f"Could not resolve Configuration/good_wcte_pmts for WCTE run {int(run)}. "
        "Set WCTE_GOOD_PMT_ROOT_FILE explicitly or provide "
        "WCTE_GOOD_PMT_FILE. Candidates:\n  - " + (detail or "none")
    )


def authoritative_active_wcte_pmts(
    good_pmt_ids: Iterable[int],
    geometry_pmt_ids: Iterable[int],
) -> set[int]:
    """Validate and return the authoritative real-WCTE active-channel set.

    The run/DQ or user mask is the complete active set.  In particular, no
    WCSim inactive-slot policy is intersected with it.  Geometry is only a
    consistency requirement: a requested channel must exist in the detector
    model because the likelihood needs its position and orientation.
    """
    requested = {int(value) for value in good_pmt_ids}
    geometry = {int(value) for value in geometry_pmt_ids}
    if not requested:
        raise ValueError("The authoritative real-WCTE good-PMT list is empty")
    absent = sorted(requested - geometry)
    if absent:
        preview = absent[:20]
        suffix = "" if len(absent) <= len(preview) else " ..."
        raise ValueError(
            f"The authoritative real-WCTE good-PMT list contains {len(absent)} "
            "channel(s) absent from the loaded detector geometry: "
            f"{preview}{suffix}"
        )
    return requested


__all__ = [
    "AnalysisToolsImportError",
    "DEFAULT_ANALYSIS_TOOLS_PATH",
    "DEFAULT_GOOD_PMT_ROOT_SEARCH_BASES",
    "DEFAULT_PRODUCTION_TEMPLATE",
    "WCTESelectionConfig",
    "authoritative_active_wcte_pmts",
    "import_analysis_tools",
    "load_good_wcte_pmts",
    "load_good_wcte_pmts_file",
    "load_selected_events",
    "production_root_file",
    "resolve_good_wcte_pmts",
]
