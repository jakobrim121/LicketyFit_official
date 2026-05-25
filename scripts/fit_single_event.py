# CORRECTED VERSION: WCTE workflow explicitly loads selected events from raw ROOT entries before fitting.
# Public helper: load_wcte_selected_events(cfg, n_root_entries=..., max_selected_events=...).
"""fit_single_event.py

Notebook-friendly single-event front end for the LicketyFit batch drivers.

The goal of this file is to let users exercise the same configuration knobs used
by ``batch_fit_driver_wcte.py`` and ``batch_fit_driver_wcsim.py`` without running
an entire batch job.  It intentionally reuses the driver modules for the
likelihood, seed scan, fixed-parameter handling, retry logic, and detector/event
conversion.  This keeps one-event studies consistent with production fits.

Typical notebook usage
----------------------

from fit_single_event import (
    SingleEventConfig,
    load_wcte_selected_events,
    fit_single_event,
    summarize_result,
)

cfg = SingleEventConfig(
    data_kind="wcte",
    run=1589,
    fit_particle="proton",
    fit_mode="absorption",
    likelihood_mode="charge_time",
    fixed_params={"ke0_mev": 1500.0},
    ncall_migrad=20000,
    max_fit_attempts=2,
)

selection = load_wcte_selected_events(
    cfg,
    n_root_entries=5000,
    max_selected_events=50,
)

print(selection["n_selected"])
print(selection["root_entry_indices"][:10])

event_array = selection["events"][0]
out = fit_single_event(cfg, event=event_array)
summarize_result(out)

The returned dictionary contains the fitted Minuit result plus arrays useful for
fast diagnostics: ``obs_pes``, ``obs_ts``, ``exp_pes``, ``exp_ts``, ``pmt_ids``,
``p_locations``, ``direction_zs``, ``mpmt_slots``, and seed/attempt metadata.

Notes
-----
* This module should live next to the batch drivers, usually in ``scripts/``.
* The batch-driver modules are reloaded for each fit so that environment-style
  configuration mirrors the production workflow and stale notebook settings are
  avoided.
* No multiprocessing is used here; the selected event is fitted in-process.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

FloatList = Optional[Sequence[float]]
DirectionList = Optional[Sequence[Tuple[float, float]]]


@dataclass
class SingleEventConfig:
    """Configuration for a one-event LicketyFit run.

    Most fields map directly to settings in ``batch_fit_driver_wcte.py`` or
    ``batch_fit_driver_wcsim.py``.  Leave a field as ``None`` to use the batch
    driver's default value.
    """

    # Which batch driver to reuse.
    data_kind: str = "wcte"  # "wcte" or "wcsim"

    # Main physics choices.
    fit_particle: Optional[str] = None
    fit_mode: Optional[str] = None  # "full_length" or "absorption"
    likelihood_mode: Optional[str] = None  # "charge_time", "charge_only", "timing_only"
    fixed_params: Dict[str, Optional[float]] = field(default_factory=dict)

    # Event choice.
    # For WCTE selection mode this is the index in the selected-event list, not
    # necessarily the raw ROOT entry number.  Prefer load_wcte_selected_events()
    # in notebooks when you want to see how many events pass cuts before fitting.
    event_index: int = 0

    # WCTE selected-event loading controls.  These make the selected-event
    # semantics explicit for notebooks: scan n raw ROOT entries, count/return the
    # events that pass cuts, then fit one event from that selected list.
    wcte_n_root_entries_to_scan: Optional[int] = None
    wcte_max_selected_events: Optional[int] = None
    wcte_loader_step_size: Optional[int] = None
    wcte_selection_sample_fraction: Optional[float] = 1.0
    wcte_selection_max_sample_events: Optional[int] = None
    wcte_use_t5_hit_time: Optional[bool] = None

    # WCTE / real-data-style event input.
    run: Optional[int] = None
    beam_p: Optional[float] = None
    event_source: Optional[str] = None  # "selection" or "file"; ignored if event=... is supplied
    user_event_file: Optional[Union[str, Path]] = None
    user_event_key: Optional[str] = None
    user_event_apply_peak_window: Optional[bool] = None

    # WCTE selection controls, used only when event_source="selection".
    particle_selection_label: Optional[str] = None
    selection_tof_ns: Optional[float] = None
    selection_tof_window_ns: Optional[float] = None
    selection_tof_field: Optional[str] = None
    selection_momentum_field: Optional[str] = None
    selection_t5_particle_nr: Optional[int] = None
    use_peak_time_cut: Optional[bool] = None
    peak_window_ns: Optional[float] = None
    peak_bin_width_ns: Optional[float] = None
    config_root_file: Optional[Union[str, Path]] = None
    allow_missing_good_pmts: Optional[bool] = None

    # WCSim event input.
    wcsim_input_file: Optional[Union[str, Path]] = None
    energy_true: Optional[float] = None
    wcsim_particle_label: Optional[str] = None
    wcsim_particle_dir: Optional[str] = None
    wcsim_pmt_id_mode: Optional[str] = None  # "auto", "mapping", or "wcte"
    wcsim_wcte_mapping_path: Optional[Union[str, Path]] = None
    wcsim_pmt_id_offset: Optional[int] = None

    # Detector/table paths.
    geometry_path: Optional[Union[str, Path]] = None
    geometry_file: Optional[Union[str, Path]] = None
    table_dir: Optional[Union[str, Path]] = None
    other_mpmt_info_path: Optional[Union[str, Path]] = None
    rel_mpmt_eff_path: Optional[Union[str, Path]] = None
    delta_e_angular_pdf_path: Optional[Union[str, Path]] = None

    # Event observable controls.
    apply_peak_time_window: Optional[bool] = None
    ring_mask_mode: Optional[str] = None  # "none", "pes", "ts", or "both"
    inactive_slots: Optional[Sequence[int]] = None

    # Fit controls and retry/rescue behavior.
    fcn_retry_threshold: Optional[float] = None
    max_fit_attempts: Optional[int] = None
    ncall_migrad: Optional[int] = None
    ncall_simplex: Optional[int] = None
    m_strat: Optional[int] = None
    t0_limits: Optional[Tuple[float, float]] = None
    t_min: Optional[float] = None
    use_t0_prior: Optional[bool] = None
    enable_stage2_migrad_first: Optional[bool] = None
    enable_stage3_adaptive_rescue: Optional[bool] = None
    enable_stage4_length_profile: Optional[bool] = None
    visible_length_retry_threshold: Optional[float] = None
    z_seed_eps: Optional[float] = None
    visible_length_seed_eps: Optional[float] = None
    full_range_seed_eps: Optional[float] = None

    # Initial seed grid.  These are the same names as the batch drivers.
    fast_seed_x0: FloatList = None
    fast_seed_y0: FloatList = None
    fast_seed_z0: FloatList = None
    fast_seed_visible_lengths: FloatList = None
    fast_seed_ke0_mev: FloatList = None
    fast_seed_full_ranges_mm: FloatList = None
    fast_seed_directions: DirectionList = None
    fast_seed_full_cartesian: Optional[bool] = None

    # Debug payloads.  For notebooks, top-N seed storage is often useful.
    save_attempt_results: Optional[bool] = None
    save_seed_scan: Optional[bool] = None
    save_top_n_seeds: Optional[int] = None

    # Import/location behavior.
    driver_dir: Optional[Union[str, Path]] = None
    force_reload_driver: bool = True
    clear_driver_environment: bool = True
    verbose: bool = True


_KNOWN_ENV_KEYS = {
    # Main controls
    "FIT_PARTICLE", "FIT_MODE", "TRACK_END_MODE", "LIKELIHOOD_MODE", "FIT_TYPE",
    "USE_CHARGE_LIKELIHOOD", "USE_TIMING_LIKELIHOOD", "USE_T0_PRIOR",
    # WCTE event input
    "RUN", "BEAM_P", "N_EVENTS", "EVENT_SOURCE", "USER_EVENT_FILE", "USER_EVENT_KEY",
    "USER_EVENT_APPLY_PEAK_WINDOW", "PARTICLE_SELECTION_LABEL", "SELECTION_TOF_NS",
    "SELECTION_TOF_WINDOW_NS", "SELECTION_TOF_FIELD", "SELECTION_MOMENTUM_FIELD",
    "SELECTION_T5_PARTICLE_NR", "USE_PEAK_TIME_CUT", "PEAK_WINDOW_NS", "PEAK_BIN_WIDTH_NS",
    "CONFIG_ROOT_FILE", "ALLOW_MISSING_GOOD_PMTS",
    # WCSim input
    "ENERGY_TRUE", "TOT_EVENTS", "WCSIM_INPUT_FILE", "DEFAULT_WCSIM_INPUT_FILE",
    "WCSIM_PARTICLE_LABEL", "WCSIM_PARTICLE_DIR", "WCSIM_PMT_ID_MODE",
    "WCSIM_WCTE_MAPPING_PATH", "WCSIM_PMT_ID_OFFSET",
    # Paths
    "GEOMETRY_PATH", "WCTE_GEOMETRY_FILE", "TABLE_DIR", "LF_TABLE_DIR",
    "LF_MULTIPARTICLES_TABLE_DIR", "LF_OFFICIAL_TABLE_DIR", "OTHER_MPMT_INFO_PATH",
    "REL_MPMT_EFF_PATH", "DELTA_E_ANGULAR_PDF_PATH",
    # Detector/event settings
    "RING_MASK_MODE", "INACTIVE_SLOTS",
    # Fit controls
    "FCN_RETRY_THRESHOLD", "MAX_FIT_ATTEMPTS", "NCALL_MIGRAD", "NCALL_SIMPLEX",
    "M_STRAT", "T0_MIN", "T0_MAX", "T_MIN", "ENABLE_STAGE2_MIGRAD_FIRST",
    "ENABLE_STAGE3_ADAPTIVE_RESCUE", "ENABLE_STAGE4_LENGTH_PROFILE",
    "VISIBLE_LENGTH_RETRY_THRESHOLD", "Z_SEED_EPS", "VISIBLE_LENGTH_SEED_EPS",
    "FULL_RANGE_SEED_EPS",
    # Seed grid
    "FAST_SEED_X0", "FAST_SEED_Y0", "FAST_SEED_Z0", "FAST_SEED_VISIBLE_LENGTHS",
    "FAST_SEED_KE0_MEV", "FAST_SEED_FULL_RANGES_MM", "FAST_SEED_FULL_CARTESIAN",
    # Fixed params
    "FIX_X0", "FIX_Y0", "FIX_Z0", "FIX_CX", "FIX_CY", "FIX_LENGTH",
    "FIX_VISIBLE_LENGTH", "FIX_FULL_RANGE", "FIX_KE0_MEV", "FIXED_KE0_MEV", "FIX_T0",
    # Debug payloads
    "SAVE_ATTEMPT_RESULTS", "SAVE_SEED_SCAN", "SAVE_TOP_N_SEEDS",
}


def _as_bool_env(value: bool) -> str:
    return "1" if bool(value) else "0"


def _as_csv(values: Iterable[Any]) -> str:
    return ",".join(str(x) for x in values)


def _path_str(value: Optional[Union[str, Path]]) -> Optional[str]:
    if value is None:
        return None
    return str(Path(value).expanduser())


def _maybe_set(env: Dict[str, str], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        env[key] = _as_bool_env(value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        env[key] = _as_csv(value)
    else:
        env[key] = str(value)


def _env_from_config(cfg: SingleEventConfig) -> Dict[str, str]:
    env: Dict[str, str] = {}

    # Main physics controls.
    _maybe_set(env, "FIT_PARTICLE", cfg.fit_particle)
    _maybe_set(env, "FIT_MODE", cfg.fit_mode)
    _maybe_set(env, "LIKELIHOOD_MODE", cfg.likelihood_mode)
    _maybe_set(env, "USE_T0_PRIOR", cfg.use_t0_prior)

    # WCTE controls.
    _maybe_set(env, "RUN", cfg.run)
    _maybe_set(env, "BEAM_P", cfg.beam_p)
    _maybe_set(env, "N_EVENTS", cfg.wcte_n_root_entries_to_scan)
    _maybe_set(env, "EVENT_SOURCE", cfg.event_source)
    _maybe_set(env, "USER_EVENT_FILE", _path_str(cfg.user_event_file))
    _maybe_set(env, "USER_EVENT_KEY", cfg.user_event_key)
    _maybe_set(env, "USER_EVENT_APPLY_PEAK_WINDOW", cfg.user_event_apply_peak_window)
    _maybe_set(env, "PARTICLE_SELECTION_LABEL", cfg.particle_selection_label)
    _maybe_set(env, "SELECTION_TOF_NS", cfg.selection_tof_ns)
    _maybe_set(env, "SELECTION_TOF_WINDOW_NS", cfg.selection_tof_window_ns)
    _maybe_set(env, "SELECTION_TOF_FIELD", cfg.selection_tof_field)
    _maybe_set(env, "SELECTION_MOMENTUM_FIELD", cfg.selection_momentum_field)
    _maybe_set(env, "SELECTION_T5_PARTICLE_NR", cfg.selection_t5_particle_nr)
    _maybe_set(env, "USE_PEAK_TIME_CUT", cfg.use_peak_time_cut)
    _maybe_set(env, "PEAK_WINDOW_NS", cfg.peak_window_ns)
    _maybe_set(env, "PEAK_BIN_WIDTH_NS", cfg.peak_bin_width_ns)
    _maybe_set(env, "CONFIG_ROOT_FILE", _path_str(cfg.config_root_file))
    _maybe_set(env, "ALLOW_MISSING_GOOD_PMTS", cfg.allow_missing_good_pmts)

    # WCSim controls.
    _maybe_set(env, "ENERGY_TRUE", cfg.energy_true)
    _maybe_set(env, "WCSIM_INPUT_FILE", _path_str(cfg.wcsim_input_file))
    _maybe_set(env, "DEFAULT_WCSIM_INPUT_FILE", _path_str(cfg.wcsim_input_file))
    _maybe_set(env, "WCSIM_PARTICLE_LABEL", cfg.wcsim_particle_label)
    _maybe_set(env, "WCSIM_PARTICLE_DIR", cfg.wcsim_particle_dir)
    _maybe_set(env, "WCSIM_PMT_ID_MODE", cfg.wcsim_pmt_id_mode)
    _maybe_set(env, "WCSIM_WCTE_MAPPING_PATH", _path_str(cfg.wcsim_wcte_mapping_path))
    _maybe_set(env, "WCSIM_PMT_ID_OFFSET", cfg.wcsim_pmt_id_offset)

    # Paths.
    _maybe_set(env, "GEOMETRY_PATH", _path_str(cfg.geometry_path))
    _maybe_set(env, "WCTE_GEOMETRY_FILE", _path_str(cfg.geometry_file))
    _maybe_set(env, "TABLE_DIR", _path_str(cfg.table_dir))
    _maybe_set(env, "LF_TABLE_DIR", _path_str(cfg.table_dir))
    _maybe_set(env, "LF_MULTIPARTICLES_TABLE_DIR", _path_str(cfg.table_dir))
    _maybe_set(env, "LF_OFFICIAL_TABLE_DIR", _path_str(cfg.table_dir))
    _maybe_set(env, "OTHER_MPMT_INFO_PATH", _path_str(cfg.other_mpmt_info_path))
    _maybe_set(env, "REL_MPMT_EFF_PATH", _path_str(cfg.rel_mpmt_eff_path))
    _maybe_set(env, "DELTA_E_ANGULAR_PDF_PATH", _path_str(cfg.delta_e_angular_pdf_path))

    # Detector/event settings.
    _maybe_set(env, "RING_MASK_MODE", cfg.ring_mask_mode)
    if cfg.inactive_slots is not None:
        env["INACTIVE_SLOTS"] = _as_csv(int(x) for x in cfg.inactive_slots)

    # Fit controls.
    _maybe_set(env, "FCN_RETRY_THRESHOLD", cfg.fcn_retry_threshold)
    _maybe_set(env, "MAX_FIT_ATTEMPTS", cfg.max_fit_attempts)
    _maybe_set(env, "NCALL_MIGRAD", cfg.ncall_migrad)
    _maybe_set(env, "NCALL_SIMPLEX", cfg.ncall_simplex)
    _maybe_set(env, "M_STRAT", cfg.m_strat)
    if cfg.t0_limits is not None:
        env["T0_MIN"] = str(float(cfg.t0_limits[0]))
        env["T0_MAX"] = str(float(cfg.t0_limits[1]))
    _maybe_set(env, "T_MIN", cfg.t_min)
    _maybe_set(env, "ENABLE_STAGE2_MIGRAD_FIRST", cfg.enable_stage2_migrad_first)
    _maybe_set(env, "ENABLE_STAGE3_ADAPTIVE_RESCUE", cfg.enable_stage3_adaptive_rescue)
    _maybe_set(env, "ENABLE_STAGE4_LENGTH_PROFILE", cfg.enable_stage4_length_profile)
    _maybe_set(env, "VISIBLE_LENGTH_RETRY_THRESHOLD", cfg.visible_length_retry_threshold)
    _maybe_set(env, "Z_SEED_EPS", cfg.z_seed_eps)
    _maybe_set(env, "VISIBLE_LENGTH_SEED_EPS", cfg.visible_length_seed_eps)
    _maybe_set(env, "FULL_RANGE_SEED_EPS", cfg.full_range_seed_eps)

    # Seed grid.
    _maybe_set(env, "FAST_SEED_X0", cfg.fast_seed_x0)
    _maybe_set(env, "FAST_SEED_Y0", cfg.fast_seed_y0)
    _maybe_set(env, "FAST_SEED_Z0", cfg.fast_seed_z0)
    _maybe_set(env, "FAST_SEED_VISIBLE_LENGTHS", cfg.fast_seed_visible_lengths)
    _maybe_set(env, "FAST_SEED_KE0_MEV", cfg.fast_seed_ke0_mev)
    _maybe_set(env, "FAST_SEED_FULL_RANGES_MM", cfg.fast_seed_full_ranges_mm)
    _maybe_set(env, "FAST_SEED_FULL_CARTESIAN", cfg.fast_seed_full_cartesian)

    # Fixed parameters.
    fixed_env = {
        "x0": "FIX_X0",
        "y0": "FIX_Y0",
        "z0": "FIX_Z0",
        "cx": "FIX_CX",
        "cy": "FIX_CY",
        "length": "FIX_LENGTH",
        "visible_length": "FIX_VISIBLE_LENGTH",
        "full_range": "FIX_FULL_RANGE",
        "ke0_mev": "FIXED_KE0_MEV",
        "t0": "FIX_T0",
    }
    for name, value in (cfg.fixed_params or {}).items():
        key = fixed_env.get(name)
        if key is None:
            raise ValueError(f"Unknown fixed parameter {name!r}. Known names: {sorted(fixed_env)}")
        if value is not None:
            env[key] = str(float(value))

    # Debug payloads.
    _maybe_set(env, "SAVE_ATTEMPT_RESULTS", cfg.save_attempt_results)
    _maybe_set(env, "SAVE_SEED_SCAN", cfg.save_seed_scan)
    _maybe_set(env, "SAVE_TOP_N_SEEDS", cfg.save_top_n_seeds)

    return env


def _apply_environment(cfg: SingleEventConfig) -> None:
    if cfg.clear_driver_environment:
        for key in _KNOWN_ENV_KEYS:
            os.environ.pop(key, None)

    env = _env_from_config(cfg)
    for key, value in env.items():
        os.environ[key] = value


def _canonical_data_kind(kind: str) -> str:
    key = str(kind).strip().lower()
    if key in {"wcte", "real", "data", "wcte_data", "wcte-real"}:
        return "wcte"
    if key in {"wcsim", "sim", "simulation", "mc"}:
        return "wcsim"
    raise ValueError("data_kind must be 'wcte' or 'wcsim'.")


def _driver_module_name(data_kind: str) -> str:
    return "batch_fit_driver_wcte" if data_kind == "wcte" else "batch_fit_driver_wcsim"


def _configure_sys_path(cfg: SingleEventConfig) -> Path:
    if cfg.driver_dir is None:
        driver_dir = Path(__file__).resolve().parent
    else:
        driver_dir = Path(cfg.driver_dir).expanduser().resolve()

    repo_root = driver_dir.parent
    candidates = [
        driver_dir,
        repo_root,
        repo_root / "LicketyFit",
        repo_root / "tables",
    ]
    for path in candidates:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return driver_dir


def _load_driver(cfg: SingleEventConfig):
    data_kind = _canonical_data_kind(cfg.data_kind)
    _configure_sys_path(cfg)
    _apply_environment(cfg)

    module_name = _driver_module_name(data_kind)
    if module_name in sys.modules and cfg.force_reload_driver:
        driver = importlib.reload(sys.modules[module_name])
    else:
        driver = importlib.import_module(module_name)

    # FAST_SEED_DIRECTIONS is not environment-configurable in the batch drivers,
    # so update it explicitly if requested.
    if cfg.fast_seed_directions is not None:
        driver.FAST_SEED_DIRECTIONS = [(float(cx), float(cy)) for cx, cy in cfg.fast_seed_directions]
        driver.FAST_SEED_GEOMETRY_VARIANTS = driver.build_sparse_geometry_variants()

    return driver, data_kind


# -----------------------------------------------------------------------------
# Runtime setup copied from the batch-driver main() setup phase
# -----------------------------------------------------------------------------

def _print_if(verbose: bool, *args: Any) -> None:
    if verbose:
        print(*args)


def _setup_driver_runtime(driver, data_kind: str, cfg: SingleEventConfig):
    verbose = bool(cfg.verbose)

    driver.RANGE_LOOKUP = driver.ParticleRangeLookup(
        driver.FIT_PARTICLE_CANONICAL,
        table_dirs=[str(driver.TABLE_DIR)],
    )
    driver.RESOLVED_FIXED_FIT_PARAMS = driver.normalize_fixed_fit_params(
        driver.FIXED_FIT_PARAMS,
        driver.RANGE_LOOKUP,
    )

    driver.configure_truth_params()
    init_param_sets = driver.build_fast_seed_grid(
        driver.RANGE_LOOKUP,
        fixed_params=driver.RESOLVED_FIXED_FIT_PARAMS,
    )
    if not init_param_sets:
        raise RuntimeError("Seed grid is empty. Check the FAST_SEED_* settings and fixed parameters.")

    for i, seed in enumerate(init_param_sets):
        missing = [k for k in driver.PARAM_NAMES if k not in seed]
        if missing:
            raise ValueError(f"Seed {i} is missing keys: {missing}")

    driver.set_active_particle(driver.FIT_PARTICLE_CANONICAL)
    driver.OVERALL_DISTANCES, driver.INIT_ENERGY_TABLE, _distance_rows = driver.get_energy_distance_tables(
        driver.FIT_PARTICLE_CANONICAL
    )

    if data_kind == "wcte":
        driver.GOOD_WCTE_PMTS_SET = driver.load_good_wcte_pmts()
    else:
        driver.SIM_WCTE_MAPPING = driver.load_wcsim_to_wcte_mapping()

    hall = driver.Device.open_file(driver.GEOMETRY_FILE)
    driver.WCD = hall.wcds[0]

    # In absorption mode fixed_initial_KE is overwritten per FCN call from
    # full_range -> ke0.  This seed only needs to be valid for construction.
    initial_ke_seed = float(
        driver.RANGE_LOOKUP.range_mm_to_energy(
            min(1000.0, float(driver.RANGE_LOOKUP.overall_distances_mm[-1]))
        )
    )

    emitter_model = driver.Emitter(
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.96,
        500.0,
        18.0,
        particle=driver.FIT_PARTICLE_CANONICAL,
        track_end_mode=driver.EMITTER_TRACK_END_MODE,
        fixed_initial_KE=initial_ke_seed if driver.IS_ABSORPTION_MODE else None,
    )

    delta_pdf_path = None
    if data_kind == "wcsim":
        delta_pdf_path = getattr(driver, "DELTA_E_ANGULAR_PDF_PATH", None)
    if delta_pdf_path is None:
        maybe = Path(driver.TABLE_DIR) / "delta_e_angular_pdf_table.npz"
        delta_pdf_path = maybe if maybe.exists() else None

    if delta_pdf_path is not None and hasattr(emitter_model, "load_delta_e_angular_pdf_table"):
        emitter_model.load_delta_e_angular_pdf_table(str(delta_pdf_path))

    driver.PMT_MODEL = driver.PMT(1.0, 0.3, 1.0, 40.0, 0.2, 0.0)
    driver.EMITTER_TEMPLATE = emitter_model.copy()
    driver.CORR_POS = None
    driver.P_LOCATIONS = None
    driver.DIRECTION_ZS = None
    driver.RING_KEEP_MASK = None
    driver.OBS_PES_ALL = []
    driver.OBS_TS_ALL = []
    if data_kind == "wcte":
        driver.MPMT_TYPE_CODES_ALL = []
        driver.MPMT_SLOTS_ALL = []

    _print_if(verbose, "LicketyFit single-event setup")
    _print_if(verbose, "  data_kind:", data_kind)
    _print_if(verbose, "  likelihood mode:", driver.LIKELIHOOD_MODE)
    _print_if(verbose, "  fit particle:", driver.FIT_PARTICLE_CANONICAL)
    _print_if(verbose, "  fit mode:", driver.TRACK_END_MODE)
    _print_if(verbose, "  fit parameters:", driver.FIT_PARAMETER_NAMES)
    _print_if(verbose, "  fixed parameters:", driver.RESOLVED_FIXED_FIT_PARAMS or "none")
    _print_if(verbose, "  number of seeds:", len(init_param_sets))

    return init_param_sets


# -----------------------------------------------------------------------------
# Event loading and observable construction
# -----------------------------------------------------------------------------

def _apply_wcte_peak_window(event: np.ndarray) -> np.ndarray:
    if event.size == 0:
        return event
    time_hist = np.histogram(event[:, 2], bins=np.arange(0, 4000))
    if len(time_hist[0]) == 0 or np.max(time_hist[0]) == 0:
        return event
    max_idx = int(np.argmax(time_hist[0]))
    lo_idx = max(0, max_idx - 20)
    hi_idx = min(len(time_hist[1]) - 1, max_idx + 5)
    min_time = time_hist[1][lo_idx]
    cut_time = time_hist[1][hi_idx]
    return event[(event[:, 2] > min_time) & (event[:, 2] < cut_time)]


def _apply_wcsim_peak_window(hit_pmts, hit_times, hit_charges):
    hit_pmts = np.asarray(hit_pmts, dtype=int)
    hit_times = np.asarray(hit_times, dtype=np.float64)
    hit_charges = np.asarray(hit_charges, dtype=np.float64)
    if hit_times.size == 0:
        keep = np.zeros_like(hit_times, dtype=bool)
    else:
        time_hist = np.histogram(hit_times, bins=np.arange(0, 2000))
        max_idx = int(np.argmax(time_hist[0]))
        min_time = 0.0
        cut_idx = min(max_idx + 5, len(time_hist[1]) - 1)
        cut_time = time_hist[1][cut_idx]
        keep = (hit_times > min_time) & (hit_times < cut_time)
    return hit_pmts[keep], hit_times[keep], hit_charges[keep]


def _coerce_wcte_event(driver, event: Any) -> np.ndarray:
    if hasattr(driver, "_coerce_event_array"):
        return driver._coerce_event_array(event, event_label="event")
    arr = np.asarray(event)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("WCTE event must be a 2D array with columns [pmt_id, charge, time].")
    return np.asarray(arr[:, :4] if arr.shape[1] >= 4 else arr[:, :3], dtype=np.float64)


def _cfg_from_any(
    config: Optional[Union[SingleEventConfig, Mapping[str, Any]]] = None,
    **overrides: Any,
) -> SingleEventConfig:
    """Normalize config-like inputs to a SingleEventConfig."""
    if config is None:
        cfg = SingleEventConfig()
    elif isinstance(config, SingleEventConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = SingleEventConfig(**dict(config))
    else:
        raise TypeError("config must be None, a SingleEventConfig, or a mapping/dict.")
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def _root_entry_indices_from_events(events: Sequence[np.ndarray]) -> np.ndarray:
    """Return one ROOT entry index per selected WCTE event when available."""
    out = []
    for ev in events:
        arr = np.asarray(ev)
        if arr.ndim == 2 and arr.shape[1] >= 4 and arr.shape[0] > 0:
            vals = np.unique(arr[:, 3].astype(np.int64))
            out.append(int(vals[0]) if vals.size else -1)
        else:
            out.append(-1)
    return np.asarray(out, dtype=np.int64)


def _load_wcte_selected_events_with_driver(
    driver,
    cfg: SingleEventConfig,
    *,
    n_root_entries: Optional[int] = None,
    max_selected_events: Optional[int] = None,
) -> Dict[str, Any]:
    """Internal selected-event loader using an already-imported WCTE driver."""
    if n_root_entries is None:
        n_root_entries = cfg.wcte_n_root_entries_to_scan
    if n_root_entries is None:
        n_root_entries = getattr(driver, "N_EVENTS", None)
    if n_root_entries is None:
        raise ValueError(
            "Set n_root_entries=... or cfg.wcte_n_root_entries_to_scan before "
            "loading selected WCTE events from a ROOT file."
        )

    n_root_entries = int(n_root_entries)
    if n_root_entries < 0:
        raise ValueError("n_root_entries must be non-negative or None.")

    if max_selected_events is None:
        max_selected_events = cfg.wcte_max_selected_events
    if max_selected_events is not None:
        max_selected_events = int(max_selected_events)
        if max_selected_events < 0:
            raise ValueError("max_selected_events must be non-negative or None.")

    kwargs = dict(
        particle=driver.PARTICLE_SELECTION_LABEL,
        root_file=driver.CONFIG_ROOT_FILE,
        max_selected_events=max_selected_events,
        use_peak_time_cut=driver.USE_PEAK_TIME_CUT,
        peak_window=driver.PEAK_WINDOW_NS,
        peak_bin_width=driver.PEAK_BIN_WIDTH_NS,
        tof_primary=driver.SELECTION_TOF_NS,
        tof_window=driver.SELECTION_TOF_WINDOW_NS,
        tof_scalar_field=driver.SELECTION_TOF_FIELD,
        momentum_scalar_field=driver.SELECTION_MOMENTUM_FIELD,
        t5_particle_nr=driver.SELECTION_T5_PARTICLE_NR,
        verbose=bool(cfg.verbose),
    )
    if cfg.wcte_loader_step_size is not None:
        kwargs["step_size"] = int(cfg.wcte_loader_step_size)
    if cfg.wcte_selection_sample_fraction is not None:
        kwargs["sample_fraction"] = float(cfg.wcte_selection_sample_fraction)
    if cfg.wcte_selection_max_sample_events is not None:
        kwargs["max_sample_events"] = int(cfg.wcte_selection_max_sample_events)
    if cfg.wcte_use_t5_hit_time is not None:
        kwargs["use_t5_hit_time"] = bool(cfg.wcte_use_t5_hit_time)

    try:
        events = driver.get_selected_events(driver.RUN, n_root_entries, **kwargs)
    except RuntimeError as exc:
        msg = str(exc)
        if "Could not estimate T5 delta peak time" in msg:
            raise RuntimeError(
                msg
                + "\n\nSingle-event helper note: this usually means the scan used too few "
                  "raw ROOT entries for the T5 reference sample. Try a larger "
                  "n_root_entries, e.g. 5000 or 20000, or pass "
                  "wcte_selection_sample_fraction=1.0. If this file does not have "
                  "usable T5_hit_time information, set wcte_use_t5_hit_time=False."
            ) from exc
        raise

    events = [_coerce_wcte_event(driver, ev) for ev in events]
    root_entry_indices = _root_entry_indices_from_events(events)
    result = {
        "events": events,
        "n_selected": len(events),
        "root_entry_indices": root_entry_indices,
        "n_root_entries_requested": n_root_entries,
        "max_selected_events": max_selected_events,
        "run": int(driver.RUN),
        "root_file": str(driver.CONFIG_ROOT_FILE),
        "particle_selection_label": str(driver.PARTICLE_SELECTION_LABEL),
        "tof_primary_ns": getattr(driver, "SELECTION_TOF_NS", None),
        "tof_window_ns": getattr(driver, "SELECTION_TOF_WINDOW_NS", None),
        "use_peak_time_cut": bool(getattr(driver, "USE_PEAK_TIME_CUT", False)),
        "use_t5_hit_time": bool(kwargs.get("use_t5_hit_time", True)),
        "driver_module": driver,
    }
    if cfg.verbose:
        print("Single-event WCTE selection summary")
        print("-----------------------------------")
        print(f"Raw ROOT entries scanned:      {n_root_entries}")
        print(f"Selected events returned:      {len(events)}")
        if root_entry_indices.size:
            print(f"First ROOT entry indices:      {root_entry_indices[:10].tolist()}")
        print("")
    return result


def load_wcte_selected_events(
    config: Optional[Union[SingleEventConfig, Mapping[str, Any]]] = None,
    *,
    n_root_entries: Optional[int] = None,
    max_selected_events: Optional[int] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Load selected WCTE events before fitting one of them.

    This is the recommended notebook workflow for WCTE ROOT data.  The user
    chooses how many raw ROOT entries to inspect, this function reports how many
    events passed the WCTE/event-loader cuts, and then one selected event can be
    passed directly to fit_single_event(..., event=selection["events"][i]).

    Parameters
    ----------
    config:
        SingleEventConfig or dict.  The data kind is forced/validated as WCTE.
    n_root_entries:
        Number of raw WCTEReadoutWindows ROOT entries to inspect before cuts.
        This is not the same as the number of selected events returned.
    max_selected_events:
        Optional cap on the selected events returned after cuts.
    overrides:
        Any SingleEventConfig field, for convenience.

    Returns
    -------
    dict
        Keys include events, n_selected, root_entry_indices, n_root_entries_requested,
        run, root_file, and selection metadata.
    """
    cfg = _cfg_from_any(config, **overrides)
    cfg = replace(cfg, data_kind="wcte")
    driver, data_kind = _load_driver(cfg)
    if data_kind != "wcte":
        raise RuntimeError("Internal error: load_wcte_selected_events loaded a non-WCTE driver.")
    return _load_wcte_selected_events_with_driver(
        driver,
        cfg,
        n_root_entries=n_root_entries,
        max_selected_events=max_selected_events,
    )


def _load_wcte_raw_event(driver, cfg: SingleEventConfig, event: Any = None) -> np.ndarray:
    if event is not None:
        return _coerce_wcte_event(driver, event)

    idx = int(cfg.event_index)
    if cfg.user_event_file is not None or str(getattr(driver, "EVENT_SOURCE", "")).lower() == "file":
        if not getattr(driver, "USER_EVENT_FILE", ""):
            raise ValueError("Set user_event_file when using WCTE EVENT_SOURCE='file'.")
        events = driver.load_user_event_file(driver.USER_EVENT_FILE, max_events=idx + 1)
        if idx >= len(events):
            raise IndexError(f"Requested event_index={idx}, but the user file only supplied {len(events)} events.")
        return _coerce_wcte_event(driver, events[idx])

    # Selected events from the run.  event_index is the selected-event index
    # after cuts, while wcte_n_root_entries_to_scan/n_root_entries controls how
    # many raw ROOT entries are inspected before cuts.
    needed_selected = idx + 1 if cfg.wcte_max_selected_events is None else cfg.wcte_max_selected_events
    needed_selected = max(needed_selected, idx + 1)
    selection = _load_wcte_selected_events_with_driver(
        driver,
        cfg,
        n_root_entries=cfg.wcte_n_root_entries_to_scan,
        max_selected_events=needed_selected,
    )
    events = selection["events"]
    if idx >= len(events):
        raise IndexError(
            f"Requested selected event_index={idx}, but only {len(events)} events passed cuts "
            f"after scanning {selection['n_root_entries_requested']} raw ROOT entries. "
            "Increase wcte_n_root_entries_to_scan/n_root_entries or choose a smaller selected index."
        )
    return _coerce_wcte_event(driver, events[idx])


def _prepare_wcte_observables(driver, cfg: SingleEventConfig, event: Any = None) -> Dict[str, Any]:
    raw_event = _load_wcte_raw_event(driver, cfg, event=event)

    apply_window = cfg.apply_peak_time_window
    if apply_window is None:
        apply_window = True if event is not None else (
            getattr(driver, "EVENT_SOURCE", "selection") == "selection"
            or bool(getattr(driver, "USER_EVENT_APPLY_PEAK_WINDOW", True))
        )
    if apply_window:
        raw_event = _apply_wcte_peak_window(raw_event)

    ev, pmt_ids = driver.sim_to_event(raw_event, driver.WCD, n_mpmt_total=106, pe_scale=143)

    if driver.P_LOCATIONS is None or driver.DIRECTION_ZS is None:
        p_locations, direction_zs, mpmt_slots = driver.EMITTER_TEMPLATE.get_pmt_placements(ev, driver.WCD, "est")
        driver.P_LOCATIONS = np.asarray(p_locations, dtype=np.float64)
        driver.DIRECTION_ZS = np.asarray(direction_zs, dtype=np.float64)
        driver.MPMT_SLOTS = np.asarray(mpmt_slots, dtype=int)
        driver.RING_KEEP_MASK = np.isin(driver.MPMT_SLOTS, driver.ALL_RING)

    obs_pes, obs_ts = driver.build_observables_from_event(ev, pe_scale=143)
    obs_pes, obs_ts = driver.apply_ring_mask_to_observables(
        obs_pes,
        obs_ts,
        driver.RING_KEEP_MASK,
        mode=driver.RING_MASK_MODE,
    )

    mpmt_type_codes = driver.get_mpmt_slot_type_codes(driver.MPMT_SLOTS)
    driver.OBS_PES_ALL = [obs_pes]
    driver.OBS_TS_ALL = [obs_ts]
    driver.MPMT_TYPE_CODES_ALL = [mpmt_type_codes]
    driver.MPMT_SLOTS_ALL = [driver.MPMT_SLOTS]

    return {
        "raw_event": raw_event,
        "event_object": ev,
        "pmt_ids": np.asarray(pmt_ids, dtype=int),
        "obs_pes": obs_pes,
        "obs_ts": obs_ts,
        "mpmt_slots": np.asarray(driver.MPMT_SLOTS, dtype=int),
        "mpmt_type_codes": mpmt_type_codes,
    }


def _coerce_wcsim_event(event: Any) -> Dict[str, np.ndarray]:
    if isinstance(event, Mapping):
        required = ["digi_hit_pmt", "digi_hit_time", "digi_hit_charge"]
        missing = [k for k in required if k not in event]
        if missing:
            raise KeyError(f"WCSim event dictionary is missing keys: {missing}")
        return {
            "digi_hit_pmt": np.asarray(event["digi_hit_pmt"], dtype=int),
            "digi_hit_time": np.asarray(event["digi_hit_time"], dtype=np.float64),
            "digi_hit_charge": np.asarray(event["digi_hit_charge"], dtype=np.float64),
        }

    arr = np.asarray(event)
    if arr.ndim == 2 and arr.shape[1] >= 3:
        # Convenience form: [pmt_id, charge, time].
        return {
            "digi_hit_pmt": np.asarray(arr[:, 0], dtype=int),
            "digi_hit_charge": np.asarray(arr[:, 1], dtype=np.float64),
            "digi_hit_time": np.asarray(arr[:, 2], dtype=np.float64),
        }
    raise ValueError(
        "WCSim event must be either a dict with digi_hit_pmt/time/charge keys "
        "or a 2D [pmt_id, charge, time] array."
    )


def _load_wcsim_raw_event(driver, cfg: SingleEventConfig, event: Any = None) -> Dict[str, np.ndarray]:
    if event is not None:
        return _coerce_wcsim_event(event)

    input_file = getattr(driver, "INPUT_FILE", None)
    if not input_file:
        raise ValueError("Set wcsim_input_file or pass event=... for WCSim fits.")

    data_raw = driver.read_sim_data(input_file)
    idx = int(cfg.event_index)
    n_available = len(data_raw["digi_hit_time"])
    if idx < 0 or idx >= n_available:
        raise IndexError(f"event_index={idx} is outside the WCSim file range [0, {n_available}).")

    return {
        "digi_hit_pmt": np.asarray(data_raw["digi_hit_pmt"][idx], dtype=int),
        "digi_hit_time": np.asarray(data_raw["digi_hit_time"][idx], dtype=np.float64),
        "digi_hit_charge": np.asarray(data_raw["digi_hit_charge"][idx], dtype=np.float64),
    }


def _prepare_wcsim_observables(driver, cfg: SingleEventConfig, event: Any = None) -> Dict[str, Any]:
    raw = _load_wcsim_raw_event(driver, cfg, event=event)

    hit_pmts = raw["digi_hit_pmt"]
    hit_times = raw["digi_hit_time"]
    hit_charges = raw["digi_hit_charge"]

    apply_window = cfg.apply_peak_time_window
    if apply_window is None:
        apply_window = True
    if apply_window:
        hit_pmts, hit_times, hit_charges = _apply_wcsim_peak_window(hit_pmts, hit_times, hit_charges)

    sim_data = {
        "digi_hit_pmt": hit_pmts,
        "digi_hit_time": hit_times,
        "digi_hit_charge": hit_charges,
    }

    ev, pmt_ids = driver.sim_to_event(sim_data, driver.WCD, n_mpmt_total=106, pe_scale=1.0)

    if driver.P_LOCATIONS is None or driver.DIRECTION_ZS is None:
        p_locations, direction_zs, mpmt_slots = driver.EMITTER_TEMPLATE.get_pmt_placements(ev, driver.WCD, "design")
        driver.P_LOCATIONS = np.asarray(p_locations, dtype=np.float64)
        driver.DIRECTION_ZS = np.asarray(direction_zs, dtype=np.float64)
        driver.MPMT_SLOTS = np.asarray(mpmt_slots, dtype=int)
        driver.RING_KEEP_MASK = np.isin(driver.MPMT_SLOTS, driver.ALL_RING)

    obs_pes, obs_ts = driver.build_observables_from_event(ev, pe_scale=1.0)
    obs_pes, obs_ts = driver.apply_ring_mask_to_observables(
        obs_pes,
        obs_ts,
        driver.RING_KEEP_MASK,
        mode=driver.RING_MASK_MODE,
    )

    driver.OBS_PES_ALL = [obs_pes]
    driver.OBS_TS_ALL = [obs_ts]

    return {
        "raw_event": sim_data,
        "event_object": ev,
        "pmt_ids": np.asarray(pmt_ids, dtype=int),
        "obs_pes": obs_pes,
        "obs_ts": obs_ts,
        "mpmt_slots": np.asarray(driver.MPMT_SLOTS, dtype=int),
        "mpmt_type_codes": None,
    }


def _prepare_observables(driver, data_kind: str, cfg: SingleEventConfig, event: Any = None) -> Dict[str, Any]:
    if data_kind == "wcte":
        return _prepare_wcte_observables(driver, cfg, event=event)
    return _prepare_wcsim_observables(driver, cfg, event=event)


# -----------------------------------------------------------------------------
# Prediction and output formatting
# -----------------------------------------------------------------------------

def predict_from_params(driver, params: Mapping[str, float], obs_pes: np.ndarray, mpmt_types=None) -> Dict[str, Any]:
    """Evaluate expected PEs/times for a set of fit parameters.

    Parameters are interpreted using the active driver's current fit mode.  This
    is useful for plotting a final fit, a seed, or a hand-edited hypothesis.
    """
    emitter = driver.EMITTER_TEMPLATE.copy()

    x0 = float(params["x0"])
    y0 = float(params["y0"])
    z0 = float(params["z0"])
    cx = float(params["cx"])
    cy = float(params["cy"])

    if driver.IS_ABSORPTION_MODE:
        visible_length = float(params["visible_length"])
        full_range = float(params["full_range"])
        t0 = float(params["t0"])
        ke0 = float(driver.RANGE_LOOKUP.range_mm_to_energy(full_range))
        emitter.fixed_initial_KE = ke0
        track_length_for_emission = visible_length
    else:
        length = float(params["length"])
        t0 = float(params["t0"])
        full_range = length
        ke0 = float(driver.RANGE_LOOKUP.range_mm_to_energy(full_range))
        emitter.fixed_initial_KE = None
        track_length_for_emission = length

    cz2 = 1.0 - cx * cx - cy * cy
    if cz2 <= 0.0:
        raise ValueError("Invalid direction: cx^2 + cy^2 must be < 1.")
    cz = float(np.sqrt(cz2))

    emitter.start_coord = (x0, y0, z0)
    emitter.starting_time = t0
    emitter.direction = (cx, cy, cz)

    init_ke = emitter.refresh_kinematics_from_length(track_length_for_emission)
    if hasattr(emitter, "visible_length_is_physical") and not emitter.visible_length_is_physical():
        raise ValueError("Parameter point is not physical: visible length exceeds allowed range.")

    s = emitter.get_emission_points(driver.P_LOCATIONS, init_ke)
    try:
        exp_pes, exp_ts = emitter.get_expected_pes_ts(
            driver.WCD,
            s,
            driver.P_LOCATIONS,
            driver.DIRECTION_ZS,
            mpmt_types,
            obs_pes,
            need_times=driver.USE_TIMING_LIKELIHOOD,
        )
    except TypeError:
        exp_pes, exp_ts = emitter.get_expected_pes_ts(
            driver.WCD,
            s,
            driver.P_LOCATIONS,
            driver.DIRECTION_ZS,
            mpmt_types,
            obs_pes,
        )

    return {
        "exp_pes": np.asarray(exp_pes, dtype=np.float64),
        "exp_ts": np.asarray(exp_ts, dtype=np.float64),
        "emitter": emitter,
        "emission_points": np.asarray(s),
        "init_ke_mev": float(init_ke),
        "ke0_mev": float(ke0),
        "track_direction": np.asarray([cx, cy, cz], dtype=np.float64),
        "track_length_for_emission_mm": float(track_length_for_emission),
        "components": getattr(emitter, "_last_expected_components", None),
    }


def _metadata_from_driver(driver, data_kind: str, cfg: SingleEventConfig, init_param_sets) -> Dict[str, Any]:
    return {
        "data_kind": data_kind,
        "event_index": int(cfg.event_index),
        "wcte_n_root_entries_to_scan": cfg.wcte_n_root_entries_to_scan,
        "wcte_max_selected_events": cfg.wcte_max_selected_events,
        "fit_particle": driver.FIT_PARTICLE_CANONICAL,
        "particle_mass_mev": float(driver.FIT_PARTICLE_MASS_MEV),
        "particle_threshold_mev": float(driver.FIT_PARTICLE_THRESHOLD_MEV),
        "track_end_mode": driver.TRACK_END_MODE,
        "fit_parameters": list(driver.FIT_PARAMETER_NAMES),
        "likelihood_mode": driver.LIKELIHOOD_MODE,
        "fixed_params": dict(driver.RESOLVED_FIXED_FIT_PARAMS),
        "free_params": list(driver.free_parameter_names(driver.RESOLVED_FIXED_FIT_PARAMS)),
        "geometry_file": str(driver.GEOMETRY_FILE),
        "table_dir": str(driver.TABLE_DIR),
        "n_initial_seeds": len(init_param_sets),
        "fcn_retry_threshold": float(driver.FCN_RETRY_THRESHOLD) if driver.FCN_RETRY_THRESHOLD is not None else None,
        "max_fit_attempts": int(driver.MAX_FIT_ATTEMPTS),
        "ncall_migrad": int(driver.NCALL_MIGRAD),
        "ring_mask_mode": str(driver.RING_MASK_MODE),
    }


def fit_single_event(
    config: Optional[Union[SingleEventConfig, Mapping[str, Any]]] = None,
    *,
    event: Any = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Fit one WCTE or WCSim event and return fit + diagnostic arrays.

    Parameters
    ----------
    config:
        ``SingleEventConfig`` or a dictionary of fields accepted by that
        dataclass.  Keyword overrides are applied after this object.
    event:
        Optional event payload supplied directly from a notebook.

        * WCTE: 2D array with columns ``[wcte_pmt_id, charge, time]`` and an
          optional 4th event-number column.
        * WCSim: dictionary with keys ``digi_hit_pmt``, ``digi_hit_time``, and
          ``digi_hit_charge``.  A convenience 2D ``[pmt_id, charge, time]`` array
          is also accepted.

    overrides:
        Any ``SingleEventConfig`` field can be passed here directly.

    Returns
    -------
    dict
        Contains ``result``/``values``/``errors`` from Minuit plus ``obs_pes``,
        ``obs_ts``, ``exp_pes``, ``exp_ts``, ``pmt_ids``, ``p_locations``,
        ``direction_zs``, ``mpmt_slots``, and metadata.
    """
    cfg = _cfg_from_any(config, **overrides)

    driver, data_kind = _load_driver(cfg)
    init_param_sets = _setup_driver_runtime(driver, data_kind, cfg)
    event_payload = _prepare_observables(driver, data_kind, cfg, event=event)

    result = driver.fit_one_event_by_index((
        0,
        init_param_sets,
        driver.FCN_RETRY_THRESHOLD,
        driver.MAX_FIT_ATTEMPTS,
        driver.NCALL_MIGRAD,
    ))

    pred = predict_from_params(
        driver,
        result["values"],
        event_payload["obs_pes"],
        mpmt_types=event_payload["mpmt_type_codes"],
    )

    exp_pes = pred["exp_pes"]
    exp_ts = pred["exp_ts"]
    nll_check = float(driver.evaluate_pmt_nll(
        exp_pes,
        event_payload["obs_pes"],
        exp_ts,
        event_payload["obs_ts"],
    ))
    if driver.USE_TIMING_LIKELIHOOD and driver.USE_T0_PRIOR:
        sigma_t0 = driver.get_t0_prior_sigma(event_payload["obs_pes"], event_payload["obs_ts"])
        nll_check += abs(0.5 * (float(result["values"]["t0"]) / sigma_t0) ** 2)

    output = {
        "result": result,
        "values": dict(result.get("values", {})),
        "errors": dict(result.get("errors", {})),
        "fval": float(result.get("fval", np.nan)),
        "nll_check": nll_check,
        "valid": bool(result.get("valid", False)),
        "metadata": _metadata_from_driver(driver, data_kind, cfg, init_param_sets),
        "config": asdict(cfg),
        "obs_pes": np.asarray(event_payload["obs_pes"], dtype=np.float64),
        "obs_ts": np.asarray(event_payload["obs_ts"], dtype=np.float64),
        "exp_pes": exp_pes,
        "exp_ts": exp_ts,
        "pmt_ids": np.asarray(event_payload["pmt_ids"], dtype=int),
        "p_locations": np.asarray(driver.P_LOCATIONS, dtype=np.float64),
        "direction_zs": np.asarray(driver.DIRECTION_ZS, dtype=np.float64),
        "mpmt_slots": np.asarray(event_payload["mpmt_slots"], dtype=int),
        "mpmt_type_codes": event_payload["mpmt_type_codes"],
        "raw_event": event_payload["raw_event"],
        "event_object": event_payload["event_object"],
        "emitter": pred["emitter"],
        "emission_points": pred["emission_points"],
        "emitter_components": pred["components"],
        "init_ke_mev": pred["init_ke_mev"],
        "ke0_mev": pred["ke0_mev"],
        "track_direction": pred["track_direction"],
        "track_length_for_emission_mm": pred["track_length_for_emission_mm"],
        "driver_module": driver,
        "initial_seeds": init_param_sets,
    }
    return output


# Convenience wrappers.

def fit_wcte_event(event: Any = None, **kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("data_kind", "wcte")
    return fit_single_event(SingleEventConfig(**kwargs), event=event)


def fit_wcsim_event(event: Any = None, **kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("data_kind", "wcsim")
    return fit_single_event(SingleEventConfig(**kwargs), event=event)


def summarize_result(out: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a compact dictionary of the most important fit outputs."""
    values = dict(out.get("values", {}))
    meta = dict(out.get("metadata", {}))
    summary = {
        "data_kind": meta.get("data_kind"),
        "fit_particle": meta.get("fit_particle"),
        "fit_mode": meta.get("track_end_mode"),
        "likelihood_mode": meta.get("likelihood_mode"),
        "fcn": out.get("fval"),
        "valid": out.get("valid"),
        "attempts": out.get("result", {}).get("attempts"),
        "chosen_seed_idx": out.get("result", {}).get("chosen_seed_index"),
        "chosen_seed_fcn": out.get("result", {}).get("chosen_seed_fcn"),
        "x0": values.get("x0"),
        "y0": values.get("y0"),
        "z0": values.get("z0"),
        "cx": values.get("cx"),
        "cy": values.get("cy"),
        "cz": float(np.sqrt(max(0.0, 1.0 - values.get("cx", np.nan) ** 2 - values.get("cy", np.nan) ** 2)))
        if "cx" in values and "cy" in values else np.nan,
        "t0": values.get("t0"),
        "visible_length_mm": values.get("visible_length", values.get("length")),
        "full_range_mm": values.get("full_range", values.get("length")),
        "ke0_mev": out.get("ke0_mev"),
        "n_observed_pes": float(np.nansum(out.get("obs_pes", np.array([])))),
        "n_expected_pes": float(np.nansum(out.get("exp_pes", np.array([])))),
    }
    return summary


def result_dataframe(out: Mapping[str, Any]):
    """Return a pandas DataFrame with one row per fitted PMT."""
    import pandas as pd

    obs_pes = np.asarray(out["obs_pes"])
    exp_pes = np.asarray(out["exp_pes"])
    obs_ts = np.asarray(out["obs_ts"])
    exp_ts = np.asarray(out["exp_ts"])
    pmt_ids = np.asarray(out.get("pmt_ids", np.arange(obs_pes.size)))
    mpmt_slots = np.asarray(out.get("mpmt_slots", np.full(obs_pes.size, -1)))

    return pd.DataFrame({
        "pmt_id": pmt_ids[: obs_pes.size],
        "mpmt_slot": mpmt_slots[: obs_pes.size],
        "obs_pes": obs_pes,
        "exp_pes": exp_pes,
        "obs_ts": obs_ts,
        "exp_ts": exp_ts,
        "charge_residual": obs_pes - exp_pes,
        "charge_ratio_exp_over_obs": np.divide(exp_pes, obs_pes, out=np.full_like(exp_pes, np.nan), where=obs_pes > 0),
        "time_residual_ns": obs_ts - exp_ts,
    })


def plot_observed_vs_expected(out: Mapping[str, Any], *, sort_by: str = "pmt_id", log: bool = False):
    """Quick charge comparison plot for notebooks."""
    import matplotlib.pyplot as plt

    df = result_dataframe(out)
    if sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, df["obs_pes"].to_numpy(), marker="o", linestyle="", label="observed")
    ax.plot(x, df["exp_pes"].to_numpy(), marker=".", linestyle="", label="expected")
    ax.set_xlabel(f"PMT index sorted by {sort_by}")
    ax.set_ylabel("PE")
    if log:
        ax.set_yscale("log")
    ax.legend()
    return fig, ax


if __name__ == "__main__":
    # Minimal command-line smoke test.  For serious use, prefer the notebook API.
    cfg = SingleEventConfig(
        data_kind=os.environ.get("LF_SINGLE_DATA_KIND", "wcte"),
        event_index=int(os.environ.get("LF_SINGLE_EVENT_INDEX", "0")),
        fit_particle=os.environ.get("FIT_PARTICLE"),
        fit_mode=os.environ.get("FIT_MODE"),
        likelihood_mode=os.environ.get("LIKELIHOOD_MODE"),
        user_event_file=os.environ.get("USER_EVENT_FILE") or None,
        wcsim_input_file=os.environ.get("WCSIM_INPUT_FILE") or None,
    )
    out = fit_single_event(cfg)
    print(summarize_result(out))
