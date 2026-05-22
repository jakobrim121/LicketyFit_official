# 7,8-parameter abrupt-endpoint batch driver.
#
# Fit parameters:
#   x0, y0, z0, cx, cy, visible_length, full_range, t0
#
# Meaning:
#   visible_length = actual visible primary-Cherenkov length before abrupt cutoff [mm]
#   full_range     = dE/dx-only range to Cherenkov threshold if no abrupt interaction [mm]
#   ke0            = inferred initial kinetic energy from full_range using particle range table [MeV]
#
# This driver assumes an abrupt endpoint model.  It does not use fixed_initial_KE
# as a fit setting; fixed_initial_KE is overwritten inside every FCN call using
# full_range -> ke0.

"""Toggleable multi-stage batch driver for the 8-parameter Minuit Cherenkov fit on WCTE/real-data-style events.

This is the selected-event/input-array version of the driver.  It preserves:
  - get_selected_events(RUN, N_EVENTS) event loading
  - run configuration GOOD_WCTE_PMTS masking from the ROOT Configuration tree
  - pe_scale=143
  - estimated geometry placement "est"
  - mPMT type/relative efficiency corrections when tables are available

The 8 fitted parameters are:
  x0, y0, z0, cx, cy, visible_length, full_range, t0
"""

import os
import sys
import pickle
import multiprocessing as mp
from pathlib import Path

import numpy as np
import uproot
from iminuit import Minuit

# =============================================================================
# SELF-CONTAINED PATH SETUP
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LICKETYFIT_DIR = PROJECT_ROOT / "LicketyFit"
TABLE_DIR = PROJECT_ROOT / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

geometry_path = os.environ.get("GEOMETRY_PATH", "/eos/user/j/jrimmer/Geometry")
GEOMETRY_FILE = os.environ.get(
    "WCTE_GEOMETRY_FILE",
    str(Path(geometry_path) / "examples" / "wcte_bldg157.geo"),
)

for _path in (str(LICKETYFIT_DIR), str(PROJECT_ROOT), str(SCRIPT_DIR), str(TABLE_DIR), geometry_path):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Force local tables before importing lookup/collapse helpers.
os.environ["LF_TABLE_DIR"] = str(TABLE_DIR)
os.environ["LF_MULTIPARTICLES_TABLE_DIR"] = str(TABLE_DIR)

from Geometry.Device import Device
from LicketyFit.Event import Event
from LicketyFit.PMT import PMT
from LicketyFit.Emitter import Emitter
from particle_cherenkov_model import (
    get_energy_distance_tables,
    set_active_particle,
    canonical_particle_name,
    particle_mass_mev,
    cherenkov_threshold_kinetic_mev,
)
try:
    from event_loader import get_selected_events
except Exception:
    get_selected_events = None
from particle_range_lookup import ParticleRangeLookup


# =============================================================================
# ENV HELPERS
# =============================================================================
def _env_float(name, default=None):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _env_optional_float(name, default=None):
    """Parse an optional float from the environment.

    Empty strings and common None-like strings mean "not fixed".
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default

    key = str(raw).strip().lower()
    if key in {"none", "null", "nan", "float", "free", "false"}:
        return None

    return float(raw)


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float_list_env(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return list(default)
    return [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]


# =============================================================================
# TOP-LEVEL CONFIGURATION
# =============================================================================
N_EVENTS_PER_BATCH = _env_int("N_EVENTS_PER_BATCH", 200)
NPROC = _env_int("NPROC", 16)
M_STRAT = _env_int("M_STRAT", 1)
#M_STRAT = _env_int("M_STRAT", 0)

Z_SEED_EPS = 20.0
VISIBLE_LENGTH_SEED_EPS = 40.0
FULL_RANGE_SEED_EPS = 80.0
T_MIN = -8.0

FCN_RETRY_THRESHOLD = 1100.0
VISIBLE_LENGTH_RETRY_THRESHOLD = _env_float("VISIBLE_LENGTH_RETRY_THRESHOLD", 2700.0)
MAX_FIT_ATTEMPTS = _env_int("MAX_FIT_ATTEMPTS", 4)
NCALL_MIGRAD = _env_int("NCALL_MIGRAD", 70000)
NCALL_SIMPLEX = _env_int("NCALL_SIMPLEX", NCALL_MIGRAD)

RUN = _env_int("RUN", 2079)
BEAM_P = _env_float("BEAM_P", 430)
N_EVENTS = _env_int("N_EVENTS", 60000)
print('N_EVENTS',N_EVENTS)

# =============================================================================
# PARTICLE HYPOTHESIS / 8-PARAMETER MODE
# =============================================================================
FIT_PARTICLE = os.environ.get("FIT_PARTICLE", "muon")
FIT_PARTICLE_CANONICAL = canonical_particle_name(FIT_PARTICLE)
FIT_PARTICLE_MASS_MEV = particle_mass_mev(FIT_PARTICLE_CANONICAL)
FIT_PARTICLE_THRESHOLD_MEV = cherenkov_threshold_kinetic_mev(FIT_PARTICLE_MASS_MEV)
set_active_particle(FIT_PARTICLE_CANONICAL)

# Fit mode:
#   full_length -> original 7-parameter fit:
#                  x0, y0, z0, cx, cy, length, t0
#                  length is the dE/dx range to Cherenkov threshold, so ke0 is inferred from length.
#   absorption  -> 8-parameter abrupt-endpoint fit:
#                  x0, y0, z0, cx, cy, visible_length, full_range, t0
#                  visible_length is the abrupt cutoff; full_range determines ke0.
_FIT_MODE_RAW = os.environ.get("FIT_MODE", os.environ.get("TRACK_END_MODE", "full_length")).strip().lower()

if _FIT_MODE_RAW in {"absorption", "absorbed", "abrupt", "abrupt_8param", "interaction", "truncated"}:
    FIT_MODE = "absorption"
elif _FIT_MODE_RAW in {"full_length", "full-length", "full", "threshold", "range", "csda", "old", "7param", "7_parameter"}:
    FIT_MODE = "full_length"
else:
    raise ValueError("FIT_MODE/TRACK_END_MODE must be 'full_length' or 'absorption'")

TRACK_END_MODE = FIT_MODE
IS_ABSORPTION_MODE = FIT_MODE == "absorption"
IS_FULL_LENGTH_MODE = FIT_MODE == "full_length"
EMITTER_TRACK_END_MODE = "abrupt" if IS_ABSORPTION_MODE else "threshold"
FULL_LENGTH_PARAM_NAMES = ("x0", "y0", "z0", "cx", "cy", "length", "t0")
ABSORPTION_PARAM_NAMES = ("x0", "y0", "z0", "cx", "cy", "visible_length", "full_range", "t0")
FIT_PARAMETER_NAMES = ABSORPTION_PARAM_NAMES if IS_ABSORPTION_MODE else FULL_LENGTH_PARAM_NAMES
ALL_FIT_PARAM_NAMES = tuple(dict.fromkeys(FULL_LENGTH_PARAM_NAMES + ABSORPTION_PARAM_NAMES))

# -----------------------------------------------------------------------------
# Fixed-parameter configuration
# -----------------------------------------------------------------------------
# Put a number in FIXED_FIT_PARAMS for any parameter you want held fixed.
# Leave the value as None to let Minuit float it normally.
#
# Environment variable equivalents are FIX_X0, FIX_Y0, FIX_Z0, FIX_CX, FIX_CY,
# FIX_LENGTH, FIX_VISIBLE_LENGTH, FIX_FULL_RANGE, and FIX_T0.
#
# In absorption mode you may also fix the initial kinetic energy directly using
# FIXED_KE0_MEV or FIX_KE0_MEV.  That value is converted to full_range only
# after the particle-specific range table is loaded, so it uses the same table
# as the likelihood and output KE conversion.  Do not set both FIX_FULL_RANGE
# and FIXED_KE0_MEV unless they intentionally describe the same range.
FIXED_KE0_MEV = _env_optional_float("FIXED_KE0_MEV", _env_optional_float("FIX_KE0_MEV", None))

FIXED_FIT_PARAMS = {
    "x0": _env_optional_float("FIX_X0", None),
    "y0": _env_optional_float("FIX_Y0", None),
    "z0": _env_optional_float("FIX_Z0", None),
    "cx": _env_optional_float("FIX_CX", None),
    "cy": _env_optional_float("FIX_CY", None),

    # full_length mode only
    "length": _env_optional_float("FIX_LENGTH", None),

    # absorption mode only
    "visible_length": _env_optional_float("FIX_VISIBLE_LENGTH", None),
    "full_range": _env_optional_float("FIX_FULL_RANGE", None),

    # derived absorption-mode fixed parameter; converted to full_range later
    "ke0_mev": FIXED_KE0_MEV,

    "t0": _env_optional_float("FIX_T0", None),
}

# This is resolved in main(), after RANGE_LOOKUP exists, and is shared by forked
# worker processes.  It never contains ke0_mev; fixed ke0 is represented as the
# corresponding fixed full_range.
RESOLVED_FIXED_FIT_PARAMS = {}

# Output/debug controls.  The full seed scan is large in the 8-parameter
# fit because every event can have thousands of seed dictionaries.  Keep these
# off for production output; enable only when debugging seed selection.
SAVE_ATTEMPT_RESULTS = _env_bool("SAVE_ATTEMPT_RESULTS", False)
SAVE_SEED_SCAN = _env_bool("SAVE_SEED_SCAN", False)
SAVE_TOP_N_SEEDS = _env_int("SAVE_TOP_N_SEEDS", 0)

# Likelihood toggles.
USE_CHARGE_LIKELIHOOD = _env_bool("USE_CHARGE_LIKELIHOOD", True)
USE_TIMING_LIKELIHOOD = _env_bool("USE_TIMING_LIKELIHOOD", True)
USE_T0_PRIOR = _env_bool("USE_T0_PRIOR", False)

if (not USE_CHARGE_LIKELIHOOD) and (not USE_TIMING_LIKELIHOOD):
    raise ValueError("At least one likelihood term must be enabled.")

if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
    LIKELIHOOD_MODE = "charge_time"
elif USE_CHARGE_LIKELIHOOD:
    LIKELIHOOD_MODE = "charge_only"
else:
    LIKELIHOOD_MODE = "timing_only"

OUTPUT_FILE = os.environ.get(
    "LF_OUTPUT_FILE",
    str(OUTPUT_DIR / f"estimates_run{RUN}_{BEAM_P:g}p_{FIT_PARTICLE_CANONICAL}_{TRACK_END_MODE}_mpmtEff_{LIKELIHOOD_MODE}.dict"),
)

RING_MASK_MODE = os.environ.get("RING_MASK_MODE", "both").strip().lower()

# Data configuration.
CONFIG_ROOT_FILE = os.environ.get(
    "CONFIG_ROOT_FILE",
    f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/{RUN}/WCTE_merged_production_R{RUN}.root",
)

# Event source.
#   "selection" / "internal" : use event_loader.get_selected_events(...) with particle-specific TOF cuts.
#   "file" / "user" / "custom" : load already-selected user events from USER_EVENT_FILE.
#
# USER_EVENT_FILE can be .npy, .npz, .pkl, or .pickle.  Supported payloads:
#   - list/object array of event arrays, each with columns [pmt_id, charge, time] or
#     [pmt_id, charge, time, event_number]
#   - a single 2D array [pmt_id, charge, time] for one event
#   - a single 2D array [pmt_id, charge, time, event_number], which is grouped by
#     event_number
EVENT_SOURCE = os.environ.get("EVENT_SOURCE", "selection").strip().lower()
if EVENT_SOURCE in {"selected", "internal", "event_loader", "auto"}:
    EVENT_SOURCE = "selection"
elif EVENT_SOURCE in {"file", "user", "custom", "user_file", "provided"}:
    EVENT_SOURCE = "file"
if EVENT_SOURCE not in {"selection", "file"}:
    raise ValueError("EVENT_SOURCE must be 'selection' or 'file'")

USER_EVENT_FILE = os.environ.get("USER_EVENT_FILE", "").strip()
USER_EVENT_KEY = os.environ.get("USER_EVENT_KEY", "").strip() or None
USER_EVENT_APPLY_PEAK_WINDOW = _env_bool("USER_EVENT_APPLY_PEAK_WINDOW", True)

# Event-selection configuration for event_loader.get_selected_events().
# Defaults reproduce the historical muon-like WCTE selection.  For non-muon
# beam selections, set PARTICLE_SELECTION_LABEL plus either SELECTION_TOF_NS
# or SELECTION_TOF_FIELD/T5_PARTICLE_NR as needed for your production ROOT file.
PARTICLE_SELECTION_LABEL = os.environ.get("PARTICLE_SELECTION_LABEL", FIT_PARTICLE_CANONICAL)
SELECTION_TOF_NS = _env_float("SELECTION_TOF_NS", None)
SELECTION_TOF_WINDOW_NS = _env_float("SELECTION_TOF_WINDOW_NS", 0.2)
SELECTION_TOF_FIELD = os.environ.get("SELECTION_TOF_FIELD", "") or None
SELECTION_MOMENTUM_FIELD = os.environ.get("SELECTION_MOMENTUM_FIELD", "") or None
SELECTION_T5_PARTICLE_NR = _env_int("SELECTION_T5_PARTICLE_NR", 1)
USE_PEAK_TIME_CUT = _env_bool("USE_PEAK_TIME_CUT", True)
PEAK_WINDOW_NS = _env_float("PEAK_WINDOW_NS", 100.0)
PEAK_BIN_WIDTH_NS = _env_float("PEAK_BIN_WIDTH_NS", 50.0)

# =============================================================================
# DETECTOR CONFIGURATION
# =============================================================================
DEFAULT_INACTIVE_SLOTS = [27, 32, 45, 74, 77, 79, 85, 91, 99, 9, 67]
INACTIVE_SLOTS = [int(x) for x in os.environ.get(
    "INACTIVE_SLOTS",
    ",".join(str(x) for x in DEFAULT_INACTIVE_SLOTS),
).replace(";", ",").split(",") if x.strip()]
INACTIVE_SLOTS_SET = set(int(s) for s in INACTIVE_SLOTS)

OUTER_RING = np.array([0, 7, 19, 34, 50, 66, 82, 83, 105, 94, 95, 71, 72, 56, 40, 24, 11, 3, 18])
INNER_RING = np.array([1, 8, 35, 51, 67, 84, 69, 70, 55, 39, 23, 10, 2, 20, 36, 52, 68, 53, 54, 38, 22, 21, 37, 9])
OUTSIDE_RING = np.array([12, 13, 4, 5, 6, 17, 33, 49, 65, 81, 82, 104, 93, 86, 87, 72, 57, 41, 25])
ALL_RING = np.arange(0, 106)

# Optional truth diagnostics.  If not supplied, true_fcn is NaN.
TRUE_LENGTH_MM = _env_float("TRUE_LENGTH_MM", None)
TRUE_VISIBLE_LENGTH_MM = _env_float("TRUE_VISIBLE_LENGTH_MM", None)
TRUE_FULL_RANGE_MM = _env_float("TRUE_FULL_RANGE_MM", None)
TRUE_INITIAL_KE_MEV = _env_float("TRUE_INITIAL_KE_MEV", None)

TRUE_PARAMS = {
    "x0": _env_float("TRUE_X0", 0.0),
    "y0": _env_float("TRUE_Y0", 0.0),
    "z0": _env_float("TRUE_Z0", -1348.0),
    "cx": _env_float("TRUE_CX", 0.0),
    "cy": _env_float("TRUE_CY", 0.0),
    "visible_length": np.nan,
    "full_range": np.nan,
    "t0": _env_float("TRUE_T0", 0.0),
}

# =============================================================================
# GLOBAL FIT-SEARCH STAGE TOGGLES
# =============================================================================
ENABLE_STAGE1_SEED_GRID = _env_bool("ENABLE_STAGE1_SEED_GRID", True)
ENABLE_STAGE2_MIGRAD_FIRST = _env_bool("ENABLE_STAGE2_MIGRAD_FIRST", False)
ENABLE_STAGE3_ADAPTIVE_RESCUE = _env_bool("ENABLE_STAGE3_ADAPTIVE_RESCUE", False)
ENABLE_STAGE4_LENGTH_PROFILE = _env_bool("ENABLE_STAGE4_LENGTH_PROFILE", False)

# Keep the full seed ranking only when it is actually needed.  This avoids
# sorting and returning thousands of seed dictionaries for normal production fits.
NEED_FULL_SEED_SCAN = (
    ENABLE_STAGE3_ADAPTIVE_RESCUE
    or ENABLE_STAGE4_LENGTH_PROFILE
    or SAVE_SEED_SCAN
    or SAVE_TOP_N_SEEDS > 0
)

# =============================================================================
# INITIAL SEED CONFIGURATION
# =============================================================================
FAST_SEED_X0 = _parse_float_list_env("FAST_SEED_X0", [-100.0, -50, 0.0, 50, 100.0])
FAST_SEED_Y0 = _parse_float_list_env("FAST_SEED_Y0", [-100.0, -50, 0.0, 50, 100.0])
FAST_SEED_Z0 = _parse_float_list_env("FAST_SEED_Z0", [-1500.0, -1400.0, -1300.0, -1350, -1200.0, -1100.0, -1000.0])

FAST_SEED_VISIBLE_LENGTHS = _parse_float_list_env(
    "FAST_SEED_VISIBLE_LENGTHS",
    [100.0, 150, 200, 250, 300.0, 350, 400, 450, 500.0, 700.0, 900.0, 1100.0, 1300.0, 1400.0, 1500.0, 1700.0, 1900.0],
)

FAST_SEED_KE0_MEV = _parse_float_list_env(
    "FAST_SEED_KE0_MEV",
    [600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0],
)
FAST_SEED_FULL_RANGES_MM = _parse_float_list_env(
    "FAST_SEED_FULL_RANGES_MM",
    [300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0],
)
FAST_SEED_DIRECTIONS = [
    (0.0, 0.0),
    (0.04, 0.0),
    (-0.04, 0.0),
    (0.0, 0.04),
    (0.0, -0.04),
]
FAST_SEED_FULL_CARTESIAN = _env_bool("FAST_SEED_FULL_CARTESIAN", False)


def active_parameter_names():
    return ABSORPTION_PARAM_NAMES if IS_ABSORPTION_MODE else FULL_LENGTH_PARAM_NAMES


def fixed_parameter_limits(range_lookup=None):
    max_range = 4000.0
    if range_lookup is not None:
        try:
            max_range = float(range_lookup.overall_distances_mm[-1])
        except Exception:
            pass

    return {
        "x0": (-2000.0, 2000.0),
        "y0": (-2000.0, 2000.0),
        "z0": (-2000.0, 2000.0),
        "cx": (-0.5, 0.5),
        "cy": (-0.5, 0.5),
        "t0": (-8.0, 8.0),
        "length": (0.0, max_range),
        "visible_length": (0.0, max_range),
        "full_range": (1.0, max_range),
    }


def normalize_fixed_fit_params(fixed_params=None, range_lookup=None):
    """Return the finite fixed parameters that are active for this fit mode.

    The returned dictionary contains only Minuit parameter names.  In absorption
    mode, a supplied ke0_mev is converted to full_range using range_lookup.
    """
    fixed_params = {} if fixed_params is None else dict(fixed_params)
    if not isinstance(fixed_params, dict):
        raise TypeError("fixed_params must be a dictionary mapping parameter names to values or None")

    active = set(active_parameter_names())
    all_known = set(ALL_FIT_PARAM_NAMES) | {"ke0_mev"}

    cleaned = {}

    fixed_ke0 = fixed_params.get("ke0_mev", None)
    if fixed_ke0 is not None:
        if not IS_ABSORPTION_MODE:
            raise ValueError("ke0_mev can only be used as a fixed parameter in absorption mode.")
        if range_lookup is None:
            raise ValueError("range_lookup is required to convert fixed ke0_mev into full_range.")

        fixed_ke0 = float(fixed_ke0)
        if not np.isfinite(fixed_ke0):
            raise ValueError(f"Fixed ke0_mev must be finite, got {fixed_ke0}")
        if fixed_ke0 <= FIT_PARTICLE_THRESHOLD_MEV:
            raise ValueError(
                f"Fixed ke0_mev={fixed_ke0} is below the Cherenkov threshold "
                f"{FIT_PARTICLE_THRESHOLD_MEV:.6g} MeV for {FIT_PARTICLE_CANONICAL}."
            )

        fixed_full_range_from_ke = float(range_lookup.energy_to_range_mm(fixed_ke0))
        if not np.isfinite(fixed_full_range_from_ke) or fixed_full_range_from_ke <= 0.0:
            raise ValueError(f"Could not convert fixed ke0_mev={fixed_ke0} to a valid full_range.")

        supplied_full_range = fixed_params.get("full_range", None)
        if supplied_full_range is not None:
            supplied_full_range = float(supplied_full_range)
            if not np.isclose(supplied_full_range, fixed_full_range_from_ke, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    "Both full_range and ke0_mev were fixed, but they disagree: "
                    f"full_range={supplied_full_range}, "
                    f"range_from_ke0={fixed_full_range_from_ke}."
                )

        fixed_params["full_range"] = fixed_full_range_from_ke

    limits = fixed_parameter_limits(range_lookup)

    for name, value in fixed_params.items():
        if value is None or name == "ke0_mev":
            continue

        if name not in all_known:
            raise ValueError(f"Unknown fixed parameter {name!r}. Known parameters are {sorted(all_known)}")
        if name not in active:
            raise ValueError(
                f"Fixed parameter {name!r} is not valid for FIT_MODE={FIT_MODE!r}. "
                f"Valid parameters are {active_parameter_names()}."
            )

        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Fixed parameter {name!r} must be finite, got {value}")

        lo, hi = limits[name]
        if value < lo or value > hi:
            raise ValueError(
                f"Fixed parameter {name!r}={value} is outside its allowed range [{lo}, {hi}]."
            )

        cleaned[name] = value

    if IS_ABSORPTION_MODE and {"visible_length", "full_range"}.issubset(cleaned):
        if cleaned["visible_length"] > cleaned["full_range"]:
            raise ValueError("In absorption mode, fixed visible_length cannot exceed fixed full_range.")

    cx = cleaned.get("cx", None)
    cy = cleaned.get("cy", None)
    if cx is not None and cy is not None and (cx * cx + cy * cy >= 1.0):
        raise ValueError("Fixed cx and cy imply cx^2 + cy^2 >= 1, so the direction is invalid.")

    return cleaned


def apply_fixed_params_to_seed(seed, fixed_params=None):
    out = dict(seed)
    for name, value in (fixed_params or {}).items():
        if name in out:
            out[name] = float(value)
    return out


def free_parameter_names(fixed_params=None):
    fixed = set((fixed_params or {}).keys())
    return tuple(name for name in active_parameter_names() if name not in fixed)


def build_sparse_geometry_variants():
    variants = []
    for x0 in FAST_SEED_X0:
        variants.append({"x0": float(x0), "y0": 0.0, "cx": 0.0, "cy": 0.0})
    for y0 in FAST_SEED_Y0:
        variants.append({"x0": 0.0, "y0": float(y0), "cx": 0.0, "cy": 0.0})
    for cx, cy in FAST_SEED_DIRECTIONS:
        variants.append({"x0": 0.0, "y0": 0.0, "cx": float(cx), "cy": float(cy)})

    unique = []
    seen = set()
    for v in variants:
        sig = (float(v["x0"]), float(v["y0"]), float(v["cx"]), float(v["cy"]))
        if sig not in seen:
            seen.add(sig)
            unique.append(v)
    return unique


FAST_SEED_GEOMETRY_VARIANTS = build_sparse_geometry_variants()


def build_full_range_seed_values(range_lookup):
    values = []
    for ke0 in FAST_SEED_KE0_MEV:
        if ke0 <= range_lookup.threshold_mev:
            continue
        r = range_lookup.energy_to_range_mm(float(ke0))
        if np.isfinite(r) and r > 0:
            values.append(float(r))

    values.extend(float(r) for r in FAST_SEED_FULL_RANGES_MM)

    max_r = float(range_lookup.overall_distances_mm[-1])
    values = [r for r in values if np.isfinite(r) and 0.0 < r <= max_r]

    unique = []
    seen = set()
    for r in values:
        sig = round(float(r), 6)
        if sig not in seen:
            seen.add(sig)
            unique.append(float(r))
    return unique


def build_fast_seed_grid(range_lookup, fixed_params=None):
    fixed_params = normalize_fixed_fit_params(fixed_params, range_lookup)
    seeds = []

    if IS_FULL_LENGTH_MODE:
        # 7-parameter original/full-length mode: scan only one length-like parameter.
        z0_values = [fixed_params["z0"]] if "z0" in fixed_params else FAST_SEED_Z0
        length_values = [fixed_params["length"]] if "length" in fixed_params else FAST_SEED_VISIBLE_LENGTHS
        max_r = float(range_lookup.overall_distances_mm[-1])
        length_values = [float(x) for x in length_values if 0.0 <= float(x) <= max_r]

        if not length_values:
            raise RuntimeError("No valid full_length seed values remain after applying fixed parameters.")

        if FAST_SEED_FULL_CARTESIAN:
            for x0 in FAST_SEED_X0:
                for y0 in FAST_SEED_Y0:
                    for z0 in z0_values:
                        for length in length_values:
                            for cx, cy in FAST_SEED_DIRECTIONS:
                                seed = {
                                    "x0": float(x0),
                                    "y0": float(y0),
                                    "z0": float(z0),
                                    "cx": float(cx),
                                    "cy": float(cy),
                                    "length": float(length),
                                    "t0": 0.0,
                                }
                                seeds.append(apply_fixed_params_to_seed(seed, fixed_params))
        else:
            for z0 in z0_values:
                for length in length_values:
                    for geom in FAST_SEED_GEOMETRY_VARIANTS:
                        seed = {
                            "x0": float(geom["x0"]),
                            "y0": float(geom["y0"]),
                            "z0": float(z0),
                            "cx": float(geom["cx"]),
                            "cy": float(geom["cy"]),
                            "length": float(length),
                            "t0": 0.0,
                        }
                        seeds.append(apply_fixed_params_to_seed(seed, fixed_params))
        keys = FULL_LENGTH_PARAM_NAMES
    else:
        # 8-parameter absorption mode: scan visible cutoff length and full CSDA range separately.
        full_range_seeds = build_full_range_seed_values(range_lookup)
        max_r = float(range_lookup.overall_distances_mm[-1])

        z0_values = [fixed_params["z0"]] if "z0" in fixed_params else FAST_SEED_Z0
        visible_length_values = (
            [fixed_params["visible_length"]]
            if "visible_length" in fixed_params
            else FAST_SEED_VISIBLE_LENGTHS
        )
        full_range_values = (
            [fixed_params["full_range"]]
            if "full_range" in fixed_params
            else full_range_seeds
        )

        visible_length_values = [float(x) for x in visible_length_values if 0.0 <= float(x) <= max_r]
        full_range_values = [float(x) for x in full_range_values if 0.0 < float(x) <= max_r]

        if not visible_length_values:
            raise RuntimeError("No valid visible_length seed values remain after applying fixed parameters.")
        if not full_range_values:
            raise RuntimeError("No valid full_range seed values remain after applying fixed parameters.")

        if FAST_SEED_FULL_CARTESIAN:
            for x0 in FAST_SEED_X0:
                for y0 in FAST_SEED_Y0:
                    for z0 in z0_values:
                        for visible_length in visible_length_values:
                            for full_range in full_range_values:
                                if visible_length > full_range:
                                    continue
                                for cx, cy in FAST_SEED_DIRECTIONS:
                                    seed = {
                                        "x0": float(x0),
                                        "y0": float(y0),
                                        "z0": float(z0),
                                        "cx": float(cx),
                                        "cy": float(cy),
                                        "visible_length": float(visible_length),
                                        "full_range": float(full_range),
                                        "t0": 0.0,
                                    }
                                    seed = apply_fixed_params_to_seed(seed, fixed_params)
                                    if seed["visible_length"] <= seed["full_range"]:
                                        seeds.append(seed)
        else:
            for z0 in z0_values:
                for visible_length in visible_length_values:
                    for full_range in full_range_values:
                        if visible_length > full_range:
                            continue
                        for geom in FAST_SEED_GEOMETRY_VARIANTS:
                            seed = {
                                "x0": float(geom["x0"]),
                                "y0": float(geom["y0"]),
                                "z0": float(z0),
                                "cx": float(geom["cx"]),
                                "cy": float(geom["cy"]),
                                "visible_length": float(visible_length),
                                "full_range": float(full_range),
                                "t0": 0.0,
                            }
                            seed = apply_fixed_params_to_seed(seed, fixed_params)
                            if seed["visible_length"] <= seed["full_range"]:
                                seeds.append(seed)
        keys = ABSORPTION_PARAM_NAMES

    unique = []
    seen = set()
    for seed in seeds:
        sig = tuple(float(seed[k]) for k in keys)
        if sig not in seen:
            seen.add(sig)
            unique.append(seed)

    if not unique:
        raise RuntimeError("Seed grid is empty after applying fixed parameters.")

    return unique


PARAM_NAMES = FIT_PARAMETER_NAMES

# =============================================================================
# GLOBALS SHARED BY WORKERS
# =============================================================================
OVERALL_DISTANCES = None
INIT_ENERGY_TABLE = None
RANGE_LOOKUP = None

WCD = None
PMT_MODEL = None
EMITTER_TEMPLATE = None
P_LOCATIONS = None
DIRECTION_ZS = None
RING_KEEP_MASK = None
CORR_POS = None
MPMT_SLOTS_ALL = None
MPMT_TYPE_CODES_ALL = None

OBS_PES_ALL = None
OBS_TS_ALL = None
GOOD_WCTE_PMTS_SET = None
# Resolved finite fixed parameters, shared with forked workers.
RESOLVED_FIXED_FIT_PARAMS = {}

# =============================================================================
# mPMT INFO / EFFICIENCY TABLES
# =============================================================================
other_mpmt_info_path = Path(os.environ.get("OTHER_MPMT_INFO_PATH", str(TABLE_DIR / "other_mpmt_info_v2.dict")))
if other_mpmt_info_path.exists():
    with open(other_mpmt_info_path, "rb") as f:
        mpmt_info = pickle.load(f)
else:
    mpmt_info = {}

rel_mpmt_eff_path = Path(os.environ.get("REL_MPMT_EFF_PATH", str(TABLE_DIR / "rel_mpmt_eff.dict")))
if rel_mpmt_eff_path.exists():
    with open(rel_mpmt_eff_path, "rb") as f:
        rel_mpmt_eff = pickle.load(f)
else:
    unity = np.ones(200, dtype=np.float64)
    rel_mpmt_eff = {
        "tri_exsitu": unity,
        "tri_insitu": unity,
        "wut_insitu": unity,
        "wut_exsitu": unity,
    }

tri_exsitu = rel_mpmt_eff["tri_exsitu"]
tri_insitu = rel_mpmt_eff["tri_insitu"]
wut_insitu = rel_mpmt_eff["wut_insitu"]
wut_exsitu = rel_mpmt_eff["wut_exsitu"]


def get_mpmt_slot_type(mpmt_slots):
    slot_type = []
    for slot in mpmt_slots:
        slot = int(slot)
        try:
            if mpmt_info[slot]["mpmt_site"] == "TRI":
                if mpmt_info[slot]["mpmt_type"] == "In-situ":
                    slot_type.append("tri_insitu")
                else:
                    slot_type.append("tri_exsitu")
            else:
                if mpmt_info[slot]["mpmt_type"] == "In-situ":
                    slot_type.append("wut_insitu")
                else:
                    slot_type.append("wut_exsitu")
        except Exception:
            slot_type.append("empty")
    return slot_type



_MPMT_TYPE_TO_CODE_LOCAL = {
    "tri_exsitu": 0,
    "tri_insitu": 1,
    "wut_exsitu": 2,
    "wut_insitu": 3,
}

def get_mpmt_slot_type_codes(mpmt_slots):
    """Same information as get_mpmt_slot_type(), but encoded once as int8.

    Emitter treats integer mPMT types directly, avoiding repeated string
    comparisons in seed scans and the first FCN call of each Minuit attempt.
    """
    types = get_mpmt_slot_type(mpmt_slots)
    return np.asarray([_MPMT_TYPE_TO_CODE_LOCAL.get(t, -1) for t in types], dtype=np.int8)

# =============================================================================
# CONFIG / TRUTH HELPERS
# =============================================================================
def load_good_wcte_pmts():
    try:
        with uproot.open(CONFIG_ROOT_FILE) as f:
            t_c = f["Configuration"]
            arr_config = t_c.arrays(library="ak")
        good = np.asarray(arr_config["good_wcte_pmts"][0], dtype=int)
        print("Loaded GOOD_WCTE_PMTS from:", CONFIG_ROOT_FILE)
        return set(good.tolist())
    except Exception as exc:
        # For selected ROOT input, missing the run Configuration tree is usually a
        # real problem.  For user-provided event files, allow a self-contained
        # fallback by default and turn on every non-inactive PMT.
        allow_missing_default = EVENT_SOURCE == "file"
        if _env_bool("ALLOW_MISSING_GOOD_PMTS", allow_missing_default):
            print("WARNING: could not load GOOD_WCTE_PMTS; using all non-inactive PMTs.")
            print("Reason:", repr(exc))
            all_ids = []
            for slot in range(106):
                if slot in INACTIVE_SLOTS_SET:
                    continue
                for pmt_pos in range(19):
                    all_ids.append(slot * 100 + pmt_pos)
            return set(all_ids)
        raise


def configure_truth_params():
    if IS_ABSORPTION_MODE:
        if TRUE_FULL_RANGE_MM is not None:
            TRUE_PARAMS["full_range"] = float(TRUE_FULL_RANGE_MM)
        elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
            TRUE_PARAMS["full_range"] = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))

        if TRUE_VISIBLE_LENGTH_MM is not None:
            TRUE_PARAMS["visible_length"] = float(TRUE_VISIBLE_LENGTH_MM)
        elif TRUE_LENGTH_MM is not None:
            TRUE_PARAMS["visible_length"] = float(TRUE_LENGTH_MM)
        else:
            TRUE_PARAMS["visible_length"] = np.nan

        if np.isfinite(TRUE_PARAMS["visible_length"]):
            TRUE_PARAMS["length"] = TRUE_PARAMS["visible_length"]
        return

    # Full-length mode: the single fitted length is also the full CSDA range.
    if TRUE_LENGTH_MM is not None:
        length = float(TRUE_LENGTH_MM)
    elif TRUE_FULL_RANGE_MM is not None:
        length = float(TRUE_FULL_RANGE_MM)
    elif TRUE_VISIBLE_LENGTH_MM is not None:
        length = float(TRUE_VISIBLE_LENGTH_MM)
    elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
        length = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))
    else:
        length = np.nan

    TRUE_PARAMS["length"] = length
    TRUE_PARAMS["visible_length"] = length
    TRUE_PARAMS["full_range"] = length


# =============================================================================
# EVENT / OBSERVABLE HELPERS
# =============================================================================
def sim_to_event(
    sim_data,
    WCD,
    n_mpmt_total=106,
    pe_scale=143,
    shift_times=True,
    n_earliest_for_t0=10,
):
    vw = 223.0598645833333  # mm/ns

    ev = Event(0, 0, n_mpmt_total)
    ev.set_mpmt_status(list(range(n_mpmt_total)), False)

    active_wcte_pmt_ids = []

    for slot in range(n_mpmt_total):
        if slot in INACTIVE_SLOTS_SET:
            continue

        slot_has_good_pmt = False
        for pmt_pos_id in range(ev.npmt_per_mpmt):
            wcte_pmt = int(slot * 100 + pmt_pos_id)
            if wcte_pmt in GOOD_WCTE_PMTS_SET:
                ev.set_pmt_status(slot, [pmt_pos_id], True)
                slot_has_good_pmt = True
                active_wcte_pmt_ids.append(wcte_pmt)

        if slot_has_good_pmt:
            ev.set_mpmt_status([slot], True)

    for i in range(len(sim_data[:, 0])):
        wcte_pmt = int(sim_data[i, 0])
        slot = int(wcte_pmt // 100)
        pmt_pos_id = int(wcte_pmt % 100)

        if slot < 0 or slot >= ev.n_mpmt:
            continue
        if pmt_pos_id < 0 or pmt_pos_id >= ev.npmt_per_mpmt:
            continue
        if not ev.mpmt_status[slot]:
            continue
        if not ev.pmt_status[slot][pmt_pos_id]:
            continue

        ev.hit_charges[slot][pmt_pos_id].append(float(sim_data[i, 1]))
        ev.hit_times[slot][pmt_pos_id].append(float(sim_data[i, 2]))

    if shift_times:
        bp_loc = np.array([0.0, 0.0, -1350.0])
        early_hits = []

        for i_mpmt in range(ev.n_mpmt):
            if not ev.mpmt_status[i_mpmt]:
                continue
            for i_pmt in range(ev.npmt_per_mpmt):
                if not ev.pmt_status[i_mpmt][i_pmt]:
                    continue
                if len(ev.hit_times[i_mpmt][i_pmt]) == 0:
                    continue

                pmt_loc = WCD.mpmts[i_mpmt].pmts[i_pmt].get_placement("est")["location"]
                r = np.linalg.norm(pmt_loc - bp_loc)

                for t in ev.hit_times[i_mpmt][i_pmt]:
                    early_hits.append({
                        "time": float(t),
                        "t0_est": float(t) - r / vw,
                    })

        if len(early_hits) > 0:
            early_hits = sorted(early_hits, key=lambda x: x["time"])
            n_use = min(n_earliest_for_t0, len(early_hits))
            time_offset = np.median([hit["t0_est"] for hit in early_hits[:n_use]])

            for i_mpmt in range(ev.n_mpmt):
                for i_pmt in range(ev.npmt_per_mpmt):
                    ev.hit_times[i_mpmt][i_pmt] = [
                        t - time_offset for t in ev.hit_times[i_mpmt][i_pmt]
                    ]

            ev.global_time_offset = time_offset

    return ev, np.asarray(active_wcte_pmt_ids, dtype=int)


def build_observables_from_event(ev, pe_scale=143):
    obs_pes = []
    obs_ts = []

    for i_mpmt in range(ev.n_mpmt):
        if not ev.mpmt_status[i_mpmt]:
            continue
        for i_pmt in range(ev.npmt_per_mpmt):
            if not ev.pmt_status[i_mpmt][i_pmt]:
                continue

            q = np.asarray(ev.hit_charges[i_mpmt][i_pmt], dtype=np.float64)
            t = np.asarray(ev.hit_times[i_mpmt][i_pmt], dtype=np.float64)

            if q.size == 0:
                obs_pes.append(0.0)
                obs_ts.append(np.nan)
            else:
                obs_pes.append(float(np.sum(q)) / pe_scale)
                obs_ts.append(float(np.sum(q * t) / np.sum(q)))

    return np.asarray(obs_pes, dtype=np.float64), np.asarray(obs_ts, dtype=np.float64)


def apply_ring_mask_to_observables(obs_pes, obs_ts, ring_keep_mask, mode="both"):
    obs_pes = obs_pes.copy()
    obs_ts = obs_ts.copy()

    if mode not in {"none", "pes", "ts", "both"}:
        raise ValueError("RING_MASK_MODE must be one of: none, pes, ts, both")

    if mode in {"pes", "both"}:
        obs_pes[~ring_keep_mask] = 0.0
    if mode in {"ts", "both"}:
        obs_ts[~ring_keep_mask] = np.nan

    return obs_pes, obs_ts


def get_t0_prior_sigma(obs_pes, obs_ts):
    n_timed = np.count_nonzero(np.isfinite(obs_ts))
    total_pe = np.sum(obs_pes)

    if (n_timed < 250) or (total_pe < 300):
        return 0.1
    elif (n_timed < 275) or (total_pe < 350):
        return 0.2
    elif (n_timed < 300) or (total_pe < 400):
        return 0.3
    elif (n_timed < 325) or (total_pe < 450):
        return 0.4
    elif (n_timed < 350) or (total_pe < 500):
        return 0.5
    elif (n_timed < 375) or (total_pe < 550):
        return 0.6
    elif (n_timed < 400) or (total_pe < 600):
        return 0.7
    elif (n_timed < 425) or (total_pe < 650):
        return 0.8
    elif (n_timed < 450) or (total_pe < 700):
        return 1.0
    elif (n_timed < 475) or (total_pe < 750):
        return 1.2
    elif (n_timed < 500) or (total_pe < 800):
        return 1.4
    elif (n_timed < 525) or (total_pe < 850):
        return 1.6
    elif (n_timed < 550) or (total_pe < 900):
        return 1.8
    else:
        return 2.0


# =============================================================================
# LIKELIHOOD EVALUATION
# =============================================================================
def get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts):
    exp_pes = np.asarray(exp_pes, dtype=np.float64)
    obs_pes = np.asarray(obs_pes, dtype=np.float64)
    exp_ts = np.asarray(exp_ts, dtype=np.float64)
    obs_ts = np.asarray(obs_ts, dtype=np.float64)

    mask = (
        (exp_pes > 0.0)
        & (obs_pes > 0.0)
        & np.isfinite(exp_ts)
        & np.isfinite(obs_ts)
    )

    if not np.any(mask):
        return 1e30

    sigma_t = PMT_MODEL.single_pe_time_std / np.sqrt(obs_pes[mask])
    dt = (obs_ts[mask] - exp_ts[mask]) / sigma_t
    return float(0.5 * np.sum(dt * dt))


def evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts):
    if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
        return PMT_MODEL.get_neg_log_likelihood_npe_t(exp_pes, obs_pes, exp_ts, obs_ts)
    if USE_CHARGE_LIKELIHOOD:
        return PMT_MODEL.get_neg_log_likelihood_npe(exp_pes, obs_pes)
    return get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts)


def evaluate_neg_log_likelihood(
    obs_pes,
    obs_ts,
    emitter,
    mpmt_types,
    x0,
    y0,
    z0,
    cx,
    cy,
    length_or_visible,
    full_range_or_t0,
    t0=None,
):
    """Evaluate the selected fit mode.

    full_length mode receives: length_or_visible=length, full_range_or_t0=t0.
    absorption mode receives:  length_or_visible=visible_length, full_range_or_t0=full_range, t0=t0.
    """
    if IS_ABSORPTION_MODE:
        visible_length = float(length_or_visible)
        full_range = float(full_range_or_t0)
        t0 = float(t0)

        if not np.isfinite(visible_length) or not np.isfinite(full_range):
            return 1e30
        if visible_length < 0.0 or full_range <= 0.0:
            return 1e30
        if visible_length > full_range:
            return 1e30
        if full_range > float(RANGE_LOOKUP.overall_distances_mm[-1]):
            return 1e30

        ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))
        if (not np.isfinite(ke0)) or ke0 <= FIT_PARTICLE_THRESHOLD_MEV:
            return 1e30
        emitter.fixed_initial_KE = ke0
        track_length_for_emission = visible_length
    else:
        length = float(length_or_visible)
        t0 = float(full_range_or_t0)
        if not np.isfinite(length) or length < 0.0:
            return 1e30
        if length > float(RANGE_LOOKUP.overall_distances_mm[-1]):
            return 1e30
        emitter.fixed_initial_KE = None
        track_length_for_emission = length

    cz2 = 1.0 - cx * cx - cy * cy
    if cz2 <= 0.0:
        return 1e30

    cz = np.sqrt(cz2)
    emitter.start_coord = (float(x0), float(y0), float(z0))
    emitter.starting_time = float(t0)
    emitter.direction = (float(cx), float(cy), float(cz))

    init_ke = emitter.refresh_kinematics_from_length(track_length_for_emission)

    if hasattr(emitter, "visible_length_is_physical"):
        if not emitter.visible_length_is_physical():
            return 1e30
    elif getattr(emitter, "last_visible_length_exceeds_range", False):
        return 1e30

    s = emitter.get_emission_points(P_LOCATIONS, init_ke)
    exp_pes, exp_ts = emitter.get_expected_pes_ts(
        WCD,
        s,
        P_LOCATIONS,
        DIRECTION_ZS,
        mpmt_types,
        obs_pes,
        need_times=USE_TIMING_LIKELIHOOD,
    )

    nll = evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts)
    if not np.isfinite(nll):
        return 1e30

    if USE_TIMING_LIKELIHOOD and USE_T0_PRIOR:
        sigma_t0 = get_t0_prior_sigma(obs_pes, obs_ts)
        nll += abs(0.5 * (float(t0) / sigma_t0) ** 2)

    return float(nll)


def _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed):
    if IS_ABSORPTION_MODE:
        return evaluate_neg_log_likelihood(
            obs_pes, obs_ts, emitter, mpmt_types,
            seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
            seed["visible_length"], seed["full_range"], seed["t0"],
        )
    return evaluate_neg_log_likelihood(
        obs_pes, obs_ts, emitter, mpmt_types,
        seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
        seed["length"], seed["t0"],
    )


def select_best_initial_seed(obs_pes, obs_ts, init_param_sets, mpmt_types=None):
    """
    Cheap deterministic seed prescan.

    Always retains the top MAX_FIT_ATTEMPTS seeds (sorted best-first) so that
    the FCN retry loop in fit_one_event_by_index can step through the next-best
    untried seed on each attempt.  When NEED_FULL_SEED_SCAN is True the full
    sorted scan is also kept for rescue/debug output.
    """
    best_info = None
    seed_scan = [] if NEED_FULL_SEED_SCAN else None

    # Always keep at least MAX_FIT_ATTEMPTS seeds for the retry loop regardless
    # of NEED_FULL_SEED_SCAN.  We maintain a small sorted list as we go; this
    # is O(N * k log k) where k = MAX_FIT_ATTEMPTS which is negligible.
    top_n_for_retry = int(MAX_FIT_ATTEMPTS)
    top_seeds_buffer = []  # kept sorted best-first, length <= top_n_for_retry

    for i, seed in enumerate(init_param_sets):
        emitter = EMITTER_TEMPLATE.copy()

        fval = _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed)

        if not np.isfinite(fval):
            fval = np.inf

        info = {
            "seed_index": int(i),
            "fval": float(fval),
            "params": dict(seed),
        }

        if seed_scan is not None:
            seed_scan.append(info)

        if best_info is None or fval < best_info["fval"]:
            best_info = info

        # Maintain a compact sorted buffer of the top-N seeds for retries.
        top_seeds_buffer.append(info)
        top_seeds_buffer.sort(key=lambda x: x["fval"])
        if len(top_seeds_buffer) > top_n_for_retry:
            top_seeds_buffer = top_seeds_buffer[:top_n_for_retry]

    if best_info is None or not np.isfinite(best_info["fval"]):
        raise RuntimeError("All seed FCNs were non-finite. Check ranges, PMT ordering, and event selection.")

    if seed_scan is not None:
        seed_scan_sorted = sorted(seed_scan, key=lambda x: x["fval"])
    else:
        # Use only the retained top-N seeds; already sorted best-first.
        seed_scan_sorted = top_seeds_buffer

    best = seed_scan_sorted[0]
    return dict(best["params"]), int(best["seed_index"]), float(best["fval"]), seed_scan_sorted


def compute_true_fcn_for_event(event_index):
    if IS_ABSORPTION_MODE:
        if not np.isfinite(TRUE_PARAMS.get("visible_length", np.nan)):
            return np.nan
        if not np.isfinite(TRUE_PARAMS.get("full_range", np.nan)):
            return np.nan
        length_args = (TRUE_PARAMS["visible_length"], TRUE_PARAMS["full_range"], TRUE_PARAMS["t0"])
    else:
        if not np.isfinite(TRUE_PARAMS.get("length", np.nan)):
            return np.nan
        length_args = (TRUE_PARAMS["length"], TRUE_PARAMS["t0"])

    mpmt_types = MPMT_TYPE_CODES_ALL[event_index]
    emitter = EMITTER_TEMPLATE.copy()
    return evaluate_neg_log_likelihood(
        OBS_PES_ALL[event_index],
        OBS_TS_ALL[event_index],
        emitter,
        mpmt_types,
        TRUE_PARAMS["x0"],
        TRUE_PARAMS["y0"],
        TRUE_PARAMS["z0"],
        TRUE_PARAMS["cx"],
        TRUE_PARAMS["cy"],
        *length_args,
    )


# =============================================================================
# MINUIT HELPERS
# =============================================================================
def make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types=None, fixed_params=None):
    if fixed_params is None:
        fixed_params = RESOLVED_FIXED_FIT_PARAMS
    fixed_params = normalize_fixed_fit_params(fixed_params, RANGE_LOOKUP)
    start_params = apply_fixed_params_to_seed(start_params, fixed_params)

    emitter = EMITTER_TEMPLATE.copy()

    if IS_ABSORPTION_MODE:
        def nll(x0, y0, z0, cx, cy, visible_length, full_range, t0):
            return evaluate_neg_log_likelihood(
                obs_pes, obs_ts, emitter, mpmt_types,
                x0, y0, z0, cx, cy, visible_length, full_range, t0,
            )
    else:
        def nll(x0, y0, z0, cx, cy, length, t0):
            return evaluate_neg_log_likelihood(
                obs_pes, obs_ts, emitter, mpmt_types,
                x0, y0, z0, cx, cy, length, t0,
            )

    m = Minuit(nll, **start_params)

    max_range = float(RANGE_LOOKUP.overall_distances_mm[-1])

    m.limits["x0"] = (-2000, 2000)
    m.limits["y0"] = (-2000, 2000)
    m.limits["z0"] = (-2000, 2000)
    m.limits["cx"] = (-0.5, 0.5)
    m.limits["cy"] = (-0.5, 0.5)
    m.limits["t0"] = (-8.0, 8.0)

    m.errors["x0"] = 30.0
    m.errors["y0"] = 30.0
    m.errors["z0"] = 30.0
    m.errors["cx"] = 0.01
    m.errors["cy"] = 0.01
    m.errors["t0"] = 0.1

    if IS_ABSORPTION_MODE:
        m.limits["visible_length"] = (0.0, max_range)
        m.limits["full_range"] = (1.0, max_range)
        m.errors["visible_length"] = 60.0
        m.errors["full_range"] = 100.0
    else:
        m.limits["length"] = (0.0, max_range)
        m.errors["length"] = 60.0

    if not USE_TIMING_LIKELIHOOD:
        m.fixed["t0"] = True

    # Apply user-requested fixed parameters after limits/errors are defined.
    # The same fixed values were already applied to the seed scan, so this keeps
    # Minuit and the deterministic prescan consistent.
    for name, value in fixed_params.items():
        if name not in m.parameters:
            raise ValueError(f"Cannot fix {name!r}; it is not a parameter in this fit mode.")
        m.values[name] = float(value)
        m.fixed[name] = True

    m.errordef = Minuit.LIKELIHOOD
    m.strategy = M_STRAT

    return m


def is_bad_minuit_result(m, *, edm_max=1e10):
    if (m.fval is None) or (not np.isfinite(m.fval)):
        return True
    # Do not use m.valid as a bad-result criterion.
    try:
        if (m.fmin is not None) and np.isfinite(m.fmin.edm) and (m.fmin.edm > edm_max):
            return True
    except Exception:
        pass
    return False


def run_minuit_attempt(m, ncall):
    # Minuit cannot improve anything if every active parameter is fixed.
    if not any(not m.fixed[name] for name in m.parameters):
        try:
            m.migrad(ncall=0)
        except Exception:
            pass
        return m

    if not ENABLE_STAGE2_MIGRAD_FIRST:
        m.strategy = M_STRAT
        m.simplex(ncall=ncall)
        m.migrad(ncall=ncall)
        return m

    ncall_fast = max(2000, int(0.35 * ncall))
    ncall_simplex = max(2000, int(0.25 * ncall))

    m.strategy = 0
    m.migrad(ncall=ncall_fast)

    if is_bad_minuit_result(m):
        m.simplex(ncall=ncall_simplex)
        m.strategy = M_STRAT
        m.migrad(ncall=ncall)

    return m


# =============================================================================
# ADAPTIVE RESCUE
# =============================================================================
ENABLE_ADAPTIVE_RESCUE = ENABLE_STAGE3_ADAPTIVE_RESCUE
RESCUE_MAX_SEEDS = 6
RESCUE_LENGTH_BINS = [
    (0.0, 1000.0),
    (1000.0, 1250.0),
    (1250.0, 1400.0),
    (1400.0, 1700.0),
    (1700.0, 3000.0),
]


def result_length_value(values):
    return float(values["visible_length"] if IS_ABSORPTION_MODE else values["length"])


def result_full_range_value(values):
    return float(values["full_range"] if IS_ABSORPTION_MODE else values["length"])


def seed_length_value(params):
    return float(params["visible_length"] if IS_ABSORPTION_MODE else params["length"])


def result_ke0_from_values(values):
    try:
        return float(RANGE_LOOKUP.range_mm_to_energy(result_full_range_value(values)))
    except Exception:
        return np.nan


def needs_rescue_result(result):
    if result is None:
        return True
    if not np.isfinite(result.get("fval", np.inf)):
        return True
    values = result.get("values", {})
    try:
        fitted_length = result_length_value(values)
        fitted_full = result_full_range_value(values)
    except Exception:
        return True
    if (not np.isfinite(fitted_length)) or (not np.isfinite(fitted_full)):
        return True
    if fitted_length <= 10.0 or fitted_length >= VISIBLE_LENGTH_RETRY_THRESHOLD:
        return True
    if IS_ABSORPTION_MODE and fitted_length > fitted_full:
        return True
    if result.get("seed_stuck", False):
        return True
    if USE_TIMING_LIKELIHOOD and result.get("below_t_min", False):
        return True
    return False


def needs_fcn_retry_result(result, fcn_threshold):
    """Return True when the default, non-adaptive FCN retry should run.

    This intentionally does not use m.valid.  A retry is triggered only by a
    non-finite FCN or by FCN exceeding the user-configured threshold.
    """
    if result is None:
        return True
    fval = float(result.get("fval", np.inf))
    if not np.isfinite(fval):
        return True
    if fcn_threshold is None:
        return False
    try:
        threshold = float(fcn_threshold)
    except Exception:
        return False
    return np.isfinite(threshold) and (fval > threshold)


def next_untried_seed_info(seed_scan_sorted, tried_seed_indices):
    """Return the best remaining seed from the prescan, or None if exhausted."""
    for seed_info in seed_scan_sorted:
        idx = int(seed_info["seed_index"])
        if idx not in tried_seed_indices:
            return seed_info
    return None


def choose_diverse_rescue_seed_infos(seed_scan_sorted, already_tried_seed_indices=None, max_total=RESCUE_MAX_SEEDS):
    already = set() if already_tried_seed_indices is None else set(already_tried_seed_indices)
    chosen = []

    for lo, hi in RESCUE_LENGTH_BINS:
        candidates = [
            s for s in seed_scan_sorted
            if int(s["seed_index"]) not in already
            and lo <= seed_length_value(s["params"]) < hi
        ]
        if candidates:
            chosen.append(candidates[0])
            already.add(int(candidates[0]["seed_index"]))
        if len(chosen) >= max_total:
            return chosen

    for s in seed_scan_sorted:
        idx = int(s["seed_index"])
        if idx in already:
            continue
        chosen.append(s)
        already.add(idx)
        if len(chosen) >= max_total:
            break

    return chosen


def compact_seed_scan(seed_scan_sorted):
    """Return the configured seed-scan payload for output/debugging."""
    if SAVE_SEED_SCAN:
        return seed_scan_sorted
    if SAVE_TOP_N_SEEDS > 0:
        return seed_scan_sorted[:SAVE_TOP_N_SEEDS]
    return []


def build_result_from_minuit(m, attempt, start_params, chosen_seed_idx, chosen_seed_fcn, seed_scan_sorted):
    current_fval = float(m.fval) if (m.fval is not None and np.isfinite(m.fval)) else np.inf
    current_values = m.values.to_dict()

    fitted_z0 = float(current_values["z0"])
    fitted_length = result_length_value(current_values)
    fitted_full = result_full_range_value(current_values)
    fitted_ke0 = result_ke0_from_values(current_values)

    visible_too_large = fitted_length > VISIBLE_LENGTH_RETRY_THRESHOLD
    z_near_seed = abs(fitted_z0 - float(start_params["z0"])) <= Z_SEED_EPS
    length_near_seed = abs(fitted_length - seed_length_value(start_params)) <= VISIBLE_LENGTH_SEED_EPS
    if IS_ABSORPTION_MODE:
        full_near_seed = abs(fitted_full - float(start_params["full_range"])) <= FULL_RANGE_SEED_EPS
    else:
        full_near_seed = False
    seed_stuck = z_near_seed and length_near_seed and (full_near_seed if IS_ABSORPTION_MODE else True)
    below_t_min = USE_TIMING_LIKELIHOOD and (current_values["t0"] < T_MIN)

    # Always expose consistent aliases in the result dictionary.
    visible_length_mm = fitted_length
    full_range_mm = fitted_full
    current_values.setdefault("length", fitted_length)
    current_values.setdefault("visible_length", visible_length_mm)
    current_values.setdefault("full_range", full_range_mm)

    return {
        "values": current_values,
        "errors": m.errors.to_dict(),
        "fval": current_fval,
        "valid": bool(m.valid),
        "fixed_params": dict(RESOLVED_FIXED_FIT_PARAMS),
        "free_params": free_parameter_names(RESOLVED_FIXED_FIT_PARAMS),
        "attempts": attempt,
        "visible_length_too_large": bool(visible_too_large),
        "length_too_large": bool(visible_too_large),
        "seed_stuck": bool(seed_stuck),
        "z_near_seed": bool(z_near_seed),
        "visible_length_near_seed": bool(length_near_seed),
        "full_range_near_seed": bool(full_near_seed),
        "length_near_seed": bool(length_near_seed),
        "below_t_min": bool(below_t_min),
        "chosen_seed_index": int(chosen_seed_idx),
        "chosen_seed_fcn": float(chosen_seed_fcn) if np.isfinite(chosen_seed_fcn) else np.nan,
        "chosen_seed_params": dict(start_params),
        "seed_scan": compact_seed_scan(seed_scan_sorted),
        "visible_length_mm": visible_length_mm,
        "full_range_mm": full_range_mm,
        "length_mm": fitted_length,
        "ke0_mev": fitted_ke0,
        "edm": (
            float(m.fmin.edm)
            if (getattr(m, "fmin", None) is not None and m.fmin.edm is not None)
            else np.nan
        ),
    }


def result_sort_key(result):
    if result is None:
        return (999, np.inf)
    fval = float(result.get("fval", np.inf))
    penalty = 0
    if not np.isfinite(fval):
        penalty += 100
    if result.get("visible_length_too_large", False) or result.get("length_too_large", False):
        penalty += 10
    if result.get("seed_stuck", False):
        penalty += 5
    if result.get("below_t_min", False):
        penalty += 5
    return (penalty, fval)


# =============================================================================
# HARD-EVENT VISIBLE-LENGTH PROFILE RESCUE
# =============================================================================
ENABLE_LENGTH_PROFILE_RESCUE = ENABLE_STAGE4_LENGTH_PROFILE
LENGTH_PROFILE_GRID = list(FAST_SEED_VISIBLE_LENGTHS)
LENGTH_PROFILE_MAX_POINTS = 6


def run_length_profile_rescue(obs_pes, obs_ts, mpmt_types, seed_scan_sorted, ncall, starting_attempt_index=100):
    profile_results = []
    base_seed = dict(seed_scan_sorted[0]["params"])
    length_key = "visible_length" if IS_ABSORPTION_MODE else "length"

    # Stage 4 is a visible-length profiling rescue.  If the user explicitly fixed
    # that same parameter, do not override their fixed value.
    if length_key in RESOLVED_FIXED_FIT_PARAMS:
        return None

    for j, profile_length in enumerate(LENGTH_PROFILE_GRID[:LENGTH_PROFILE_MAX_POINTS]):
        start_params = dict(base_seed)
        start_params[length_key] = float(profile_length)
        if IS_ABSORPTION_MODE and start_params["visible_length"] > start_params["full_range"]:
            continue

        m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
        m.fixed[length_key] = True
        run_minuit_attempt(m, max(5000, int(0.5 * ncall)))

        prof_result = build_result_from_minuit(
            m,
            attempt=starting_attempt_index + j,
            start_params=start_params,
            chosen_seed_idx=-1000 - j,
            chosen_seed_fcn=np.nan,
            seed_scan_sorted=seed_scan_sorted,
        )
        prof_result["profile_fixed_length"] = float(profile_length)
        prof_result["profile_fixed_visible_length"] = float(profile_length)
        profile_results.append(prof_result)

    if not profile_results:
        return None

    best_profile = min(profile_results, key=result_sort_key)

    polish_params = dict(best_profile["values"])
    # Keep only parameters actually used by this mode; Minuit will reject extras.
    polish_params = {k: polish_params[k] for k in PARAM_NAMES if k in polish_params}
    m = make_minuit_for_event(obs_pes, obs_ts, polish_params, mpmt_types)
    m.fixed[length_key] = False
    run_minuit_attempt(m, ncall)

    polish_result = build_result_from_minuit(
        m,
        attempt=starting_attempt_index + len(profile_results),
        start_params=polish_params,
        chosen_seed_idx=-2000,
        chosen_seed_fcn=float(best_profile["fval"]),
        seed_scan_sorted=seed_scan_sorted,
    )
    polish_result["length_profile_rescue_used"] = True
    polish_result["length_profile_results"] = profile_results
    polish_result["length_profile_best_fixed"] = best_profile
    return polish_result


def fit_one_event_by_index(args):
    event_index, init_param_sets, fcn_threshold, max_attempts, ncall = args

    obs_pes = OBS_PES_ALL[event_index]
    obs_ts = OBS_TS_ALL[event_index]
    mpmt_types = MPMT_TYPE_CODES_ALL[event_index]

    best_seed, best_seed_idx, best_seed_fval, seed_scan_sorted = select_best_initial_seed(
        obs_pes,
        obs_ts,
        init_param_sets,
        mpmt_types,
    )

    attempt_results = []
    tried_seed_indices = set()

    primary_info = seed_scan_sorted[0]
    start_params = dict(primary_info["params"])
    chosen_seed_idx = int(primary_info["seed_index"])
    chosen_seed_fcn = float(primary_info["fval"])
    tried_seed_indices.add(chosen_seed_idx)

    m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
    run_minuit_attempt(m, ncall)
    primary_result = build_result_from_minuit(
        m,
        attempt=1,
        start_params=start_params,
        chosen_seed_idx=chosen_seed_idx,
        chosen_seed_fcn=chosen_seed_fcn,
        seed_scan_sorted=seed_scan_sorted,
    )
    attempt_results.append(primary_result)

    # ------------------------------------------------------------------
    # Default FCN retry path.  This is intentionally independent of the
    # adaptive-rescue stage: if the best result so far has FCN above the
    # configured threshold, try the next-best prescan seeds up to
    # MAX_FIT_ATTEMPTS.  Do not use m.valid as a bad-result criterion.
    # ------------------------------------------------------------------
    while len(attempt_results) < max(1, int(max_attempts)):
        best_so_far = min(attempt_results, key=result_sort_key)
        if not needs_fcn_retry_result(best_so_far, fcn_threshold):
            break

        seed_info = next_untried_seed_info(seed_scan_sorted, tried_seed_indices)
        if seed_info is None:
            break

        start_params = dict(seed_info["params"])
        chosen_seed_idx = int(seed_info["seed_index"])
        chosen_seed_fcn = float(seed_info["fval"])
        tried_seed_indices.add(chosen_seed_idx)

        m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
        run_minuit_attempt(m, ncall)
        result = build_result_from_minuit(
            m,
            attempt=len(attempt_results) + 1,
            start_params=start_params,
            chosen_seed_idx=chosen_seed_idx,
            chosen_seed_fcn=chosen_seed_fcn,
            seed_scan_sorted=seed_scan_sorted,
        )
        result["fcn_retry_used"] = True
        attempt_results.append(result)

    best_result = min(attempt_results, key=result_sort_key)

    if ENABLE_ADAPTIVE_RESCUE and needs_rescue_result(best_result):
        rescue_seed_infos = choose_diverse_rescue_seed_infos(
            seed_scan_sorted,
            already_tried_seed_indices=tried_seed_indices,
            max_total=RESCUE_MAX_SEEDS,
        )

        for seed_info in rescue_seed_infos:
            start_params = dict(seed_info["params"])
            chosen_seed_idx = int(seed_info["seed_index"])
            chosen_seed_fcn = float(seed_info["fval"])
            tried_seed_indices.add(chosen_seed_idx)

            m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
            run_minuit_attempt(m, ncall)
            result = build_result_from_minuit(
                m,
                attempt=len(attempt_results) + 1,
                start_params=start_params,
                chosen_seed_idx=chosen_seed_idx,
                chosen_seed_fcn=chosen_seed_fcn,
                seed_scan_sorted=seed_scan_sorted,
            )
            result["adaptive_rescue_attempt"] = True
            attempt_results.append(result)

    best_result = min(attempt_results, key=result_sort_key)

    if ENABLE_LENGTH_PROFILE_RESCUE and needs_rescue_result(best_result):
        profile_result = run_length_profile_rescue(
            obs_pes,
            obs_ts,
            mpmt_types,
            seed_scan_sorted,
            ncall,
            starting_attempt_index=100 + len(attempt_results),
        )
        if profile_result is not None:
            attempt_results.append(profile_result)
            best_result = min(attempt_results, key=result_sort_key)

    best_result["attempts"] = len(attempt_results)
    if SAVE_ATTEMPT_RESULTS:
        best_result["attempt_results"] = attempt_results
    else:
        best_result["attempt_results"] = []
    best_result["adaptive_rescue_used"] = bool(len(attempt_results) > 1)
    best_result["length_profile_rescue_considered"] = bool(ENABLE_LENGTH_PROFILE_RESCUE)
    best_result["length_profile_rescue_used"] = bool(
        best_result.get("length_profile_rescue_used", False)
        or any(r.get("length_profile_rescue_used", False) for r in attempt_results)
    )
    return best_result


def run_batch(event_indices, init_param_sets, nproc, fcn_threshold, max_attempts, ncall):
    args = [(idx, init_param_sets, fcn_threshold, max_attempts, ncall) for idx in event_indices]

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    with ctx.Pool(processes=nproc) as pool:
        return pool.map(fit_one_event_by_index, args)


# =============================================================================
# USER-PROVIDED EVENT FILE HELPERS
# =============================================================================
def _coerce_event_array(event, *, event_label="event"):
    arr = np.asarray(event)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f"{event_label} must be a 2D array with at least 3 columns: "
            "[pmt_id, charge, time]."
        )
    # Keep optional event-number columns, but the fitter only consumes columns 0:3.
    return np.asarray(arr[:, :4] if arr.shape[1] >= 4 else arr[:, :3], dtype=np.float64)


def _events_from_loaded_object(obj):
    """Normalize npy/npz/pickle payloads into a list of event arrays."""
    if isinstance(obj, dict):
        if USER_EVENT_KEY is not None:
            obj = obj[USER_EVENT_KEY]
        elif "events" in obj:
            obj = obj["events"]
        elif "data" in obj:
            obj = obj["data"]
        elif "arr_0" in obj:
            obj = obj["arr_0"]
        else:
            keys = ", ".join(map(str, obj.keys()))
            raise KeyError(
                "Could not choose an event array from the dict payload. "
                f"Available keys: {keys}. Set USER_EVENT_KEY."
            )

    if isinstance(obj, np.lib.npyio.NpzFile):
        if USER_EVENT_KEY is not None:
            key = USER_EVENT_KEY
        elif "events" in obj.files:
            key = "events"
        elif "data" in obj.files:
            key = "data"
        elif "arr_0" in obj.files:
            key = "arr_0"
        elif len(obj.files) == 1:
            key = obj.files[0]
        else:
            raise KeyError(
                "Could not choose an event array from the npz payload. "
                f"Available keys: {obj.files}. Set USER_EVENT_KEY."
            )
        obj = obj[key]

    if isinstance(obj, (list, tuple)):
        return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(obj)]

    arr = np.asarray(obj, dtype=object if getattr(obj, "dtype", None) == object else None)

    # Object arrays are normally lists of variable-length events.
    if arr.dtype == object and arr.ndim == 1:
        return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(arr)]

    # A 3D numeric array is N_events x N_hits x N_columns.
    if arr.ndim == 3:
        return [_coerce_event_array(arr[i], event_label=f"event[{i}]") for i in range(arr.shape[0])]

    # A 2D array with a 4th column is interpreted as a concatenated event table
    # grouped by event number.  A 2D array with only 3 columns is one event.
    if arr.ndim == 2:
        arr2 = np.asarray(arr, dtype=np.float64)
        if arr2.shape[1] >= 4:
            events = []
            event_numbers = arr2[:, 3].astype(np.int64)
            for evnum in np.unique(event_numbers):
                events.append(_coerce_event_array(arr2[event_numbers == evnum], event_label=f"event_number={evnum}"))
            return events
        return [_coerce_event_array(arr2, event_label="single_event")]

    raise ValueError(
        "Unsupported USER_EVENT_FILE payload. Expected a list/object array of "
        "event arrays, a 3D event array, or a 2D [pmt_id, charge, time] table."
    )


def load_user_event_file(path, *, max_events=None):
    """Load user-provided, already-selected events from npy/npz/pickle files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"USER_EVENT_FILE does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        loaded = np.load(path, allow_pickle=True)
    elif suffix == ".npy":
        loaded = np.load(path, allow_pickle=True)
    elif suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            loaded = pickle.load(f)
    else:
        raise ValueError(
            f"Unsupported USER_EVENT_FILE suffix {suffix!r}. Use .npy, .npz, .pkl, or .pickle."
        )

    events = _events_from_loaded_object(loaded)
    if max_events is not None:
        events = events[: int(max_events)]
    print(f"Loaded {len(events)} user-provided events from: {path}")
    return events


# =============================================================================
# MAIN DRIVER
# =============================================================================
def main():
    global OVERALL_DISTANCES, INIT_ENERGY_TABLE, RANGE_LOOKUP
    global WCD, PMT_MODEL, EMITTER_TEMPLATE, P_LOCATIONS, DIRECTION_ZS, RING_KEEP_MASK, CORR_POS
    global OBS_PES_ALL, OBS_TS_ALL, MPMT_SLOTS_ALL, MPMT_TYPE_CODES_ALL, GOOD_WCTE_PMTS_SET
    global RESOLVED_FIXED_FIT_PARAMS

    print("Likelihood mode:", LIKELIHOOD_MODE)
    print("Fit particle:", FIT_PARTICLE_CANONICAL)
    print("Particle mass [MeV]:", FIT_PARTICLE_MASS_MEV)
    print("Cherenkov threshold [MeV]:", FIT_PARTICLE_THRESHOLD_MEV)
    print("Fit mode:", TRACK_END_MODE)
    print("Fit parameters:", FIT_PARAMETER_NAMES)
    print("Output file:", OUTPUT_FILE)

    if EVENT_SOURCE == "selection" and get_selected_events is None:
        raise ImportError(
            "event_loader.py was not found. Copy it into LF_multiParticles/scripts "
            "or add its directory to PYTHONPATH, or set EVENT_SOURCE=file and USER_EVENT_FILE."
        )

    if EVENT_SOURCE == "file" and not USER_EVENT_FILE:
        raise ValueError("EVENT_SOURCE=file requires USER_EVENT_FILE=/path/to/events.npy|npz|pkl")

    GOOD_WCTE_PMTS_SET = load_good_wcte_pmts()

    RANGE_LOOKUP = ParticleRangeLookup(FIT_PARTICLE_CANONICAL, table_dirs=[str(TABLE_DIR)])
    print("Range table max KE [MeV]:", float(RANGE_LOOKUP.initial_energies_mev[-1]))
    print("Range table max full_range [mm]:", float(RANGE_LOOKUP.overall_distances_mm[-1]))

    RESOLVED_FIXED_FIT_PARAMS = normalize_fixed_fit_params(FIXED_FIT_PARAMS, RANGE_LOOKUP)
    if RESOLVED_FIXED_FIT_PARAMS:
        print("Fixed parameters:", RESOLVED_FIXED_FIT_PARAMS)
        print("Free parameters:", free_parameter_names(RESOLVED_FIXED_FIT_PARAMS))
        if IS_ABSORPTION_MODE and FIXED_KE0_MEV is not None:
            print("Fixed KE0 [MeV]:", float(FIXED_KE0_MEV))
            print("Fixed full_range from KE0 [mm]:", RESOLVED_FIXED_FIT_PARAMS.get("full_range"))
    else:
        print("Fixed parameters: none")

    configure_truth_params()
    if IS_ABSORPTION_MODE:
        truth_ready = np.isfinite(TRUE_PARAMS["visible_length"]) and np.isfinite(TRUE_PARAMS["full_range"])
        if truth_ready:
            print("Truth visible length [mm]:", TRUE_PARAMS["visible_length"])
            print("Truth full range [mm]:", TRUE_PARAMS["full_range"])
            print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["full_range"]))
        else:
            print("Truth FCN disabled: set TRUE_VISIBLE_LENGTH_MM and TRUE_FULL_RANGE_MM/TRUE_INITIAL_KE_MEV.")
    else:
        truth_ready = np.isfinite(TRUE_PARAMS["length"])
        if truth_ready:
            print("Truth length [mm]:", TRUE_PARAMS["length"])
            print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["length"]))
        else:
            print("Truth FCN disabled: set TRUE_LENGTH_MM or TRUE_INITIAL_KE_MEV.")

    init_param_sets = build_fast_seed_grid(RANGE_LOOKUP, fixed_params=RESOLVED_FIXED_FIT_PARAMS)
    if not init_param_sets:
        raise RuntimeError("Seed grid is empty. Check FAST_SEED_VISIBLE_LENGTHS and FAST_SEED_KE0_MEV/FULL_RANGES.")
    print("Number of initial seeds:", len(init_param_sets))

    for i, seed in enumerate(init_param_sets):
        missing = [k for k in PARAM_NAMES if k not in seed]
        if missing:
            raise ValueError(f"Seed {i} is missing keys: {missing}")

    set_active_particle(FIT_PARTICLE_CANONICAL)
    OVERALL_DISTANCES, INIT_ENERGY_TABLE, _distance_rows = get_energy_distance_tables(
        FIT_PARTICLE_CANONICAL
    )

    hall = Device.open_file(GEOMETRY_FILE)
    WCD = hall.wcds[0]

    initial_ke_seed = float(RANGE_LOOKUP.range_mm_to_energy(
        min(1000.0, float(RANGE_LOOKUP.overall_distances_mm[-1]))
    ))

    emitter_model = Emitter(
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.96,
        500.0,
        18.0,
        particle=FIT_PARTICLE_CANONICAL,
        track_end_mode=EMITTER_TRACK_END_MODE,
        fixed_initial_KE=initial_ke_seed if IS_ABSORPTION_MODE else None,
    )

    delta_pdf_path = TABLE_DIR / "delta_e_angular_pdf_table.npz"
    if delta_pdf_path.exists() and hasattr(emitter_model, "load_delta_e_angular_pdf_table"):
        emitter_model.load_delta_e_angular_pdf_table(str(delta_pdf_path))

    PMT_MODEL = PMT(1.0,0.3, 1.0, 40.0, 0.2, 0.0)
    EMITTER_TEMPLATE = emitter_model.copy()
    CORR_POS = None

    print("Building event observables...")

    obs_pes_all = []
    obs_ts_all = []
    mpmt_slots_all = []

    if EVENT_SOURCE == "selection":
        print("N_EVENTS AHHHHH", N_EVENTS)
        events = get_selected_events(
            RUN,
            N_EVENTS,
            particle=PARTICLE_SELECTION_LABEL,
            root_file=CONFIG_ROOT_FILE,
            use_peak_time_cut=USE_PEAK_TIME_CUT,
            peak_window=PEAK_WINDOW_NS,
            peak_bin_width=PEAK_BIN_WIDTH_NS,
            tof_primary=SELECTION_TOF_NS,
            tof_window=SELECTION_TOF_WINDOW_NS,
            tof_scalar_field=SELECTION_TOF_FIELD,
            momentum_scalar_field=SELECTION_MOMENTUM_FIELD,
            t5_particle_nr=SELECTION_T5_PARTICLE_NR,
        )
    else:
        events = load_user_event_file(USER_EVENT_FILE, max_events=N_EVENTS)

    tot_events = len(events)
    print("Total Events to Fit:", tot_events)

    for i in range(tot_events):
        event = np.asarray(events[i])
        if event.size == 0:
            continue

        apply_event_time_window = (EVENT_SOURCE == "selection") or USER_EVENT_APPLY_PEAK_WINDOW
        if apply_event_time_window:
            time_hist = np.histogram(event[:, 2], bins=np.arange(0, 4000))
            max_idx = int(np.argmax(time_hist[0]))
            lo_idx = max(0, max_idx - 20)
            hi_idx = min(len(time_hist[1]) - 1, max_idx + 5)
            min_time = time_hist[1][lo_idx]
            cut_time = time_hist[1][hi_idx]
            time_mask = (event[:, 2] > min_time) & (event[:, 2] < cut_time)
            event = event[time_mask]

        ev, pmt_ids = sim_to_event(event, WCD, n_mpmt_total=106, pe_scale=143)

        if P_LOCATIONS is None or DIRECTION_ZS is None:
            P_LOCATIONS, DIRECTION_ZS, MPMT_SLOTS = EMITTER_TEMPLATE.get_pmt_placements(ev, WCD, "est")
            MPMT_SLOTS = np.asarray(MPMT_SLOTS, dtype=int)
            RING_KEEP_MASK = np.isin(MPMT_SLOTS, ALL_RING)

        obs_pes, obs_ts = build_observables_from_event(ev, pe_scale=143)
        obs_pes, obs_ts = apply_ring_mask_to_observables(
            obs_pes,
            obs_ts,
            RING_KEEP_MASK,
            mode=RING_MASK_MODE,
        )

        obs_pes_all.append(obs_pes)
        obs_ts_all.append(obs_ts)
        mpmt_slots_all.append(MPMT_SLOTS)

    OBS_PES_ALL = obs_pes_all
    OBS_TS_ALL = obs_ts_all
    MPMT_SLOTS_ALL = mpmt_slots_all
    MPMT_TYPE_CODES_ALL = [get_mpmt_slot_type_codes(slots) for slots in MPMT_SLOTS_ALL]
    tot_events = len(OBS_PES_ALL)

    print("Computing truth FCNs...")
    true_fcn_all = [compute_true_fcn_for_event(i) for i in range(tot_events)]

    est_dict = {
        "metadata": {
            "fit_particle": FIT_PARTICLE_CANONICAL,
            "particle_mass_mev": FIT_PARTICLE_MASS_MEV,
            "particle_threshold_mev": FIT_PARTICLE_THRESHOLD_MEV,
            "beam_p": BEAM_P,
            "track_end_mode": TRACK_END_MODE,
            "fit_parameters": list(FIT_PARAMETER_NAMES),
            "fixed_params": dict(RESOLVED_FIXED_FIT_PARAMS),
            "free_params": list(free_parameter_names(RESOLVED_FIXED_FIT_PARAMS)),
            "fixed_ke0_mev": float(FIXED_KE0_MEV) if FIXED_KE0_MEV is not None else None,
            "truth_params": dict(TRUE_PARAMS),
            "geometry_file": GEOMETRY_FILE,
            "config_root_file": CONFIG_ROOT_FILE,
            "event_source": EVENT_SOURCE,
            "user_event_file": USER_EVENT_FILE if EVENT_SOURCE == "file" else None,
            "user_event_key": USER_EVENT_KEY if EVENT_SOURCE == "file" else None,
            "user_event_apply_peak_window": bool(USER_EVENT_APPLY_PEAK_WINDOW) if EVENT_SOURCE == "file" else None,
            "particle_selection_label": PARTICLE_SELECTION_LABEL,
            "selection_tof_ns": SELECTION_TOF_NS,
            "selection_tof_window_ns": SELECTION_TOF_WINDOW_NS,
            "selection_tof_field": SELECTION_TOF_FIELD,
            "selection_t5_particle_nr": SELECTION_T5_PARTICLE_NR,
            "range_table_max_full_range_mm": float(RANGE_LOOKUP.overall_distances_mm[-1]),
            "save_seed_scan": bool(SAVE_SEED_SCAN),
            "save_top_n_seeds": int(SAVE_TOP_N_SEEDS),
            "save_attempt_results": bool(SAVE_ATTEMPT_RESULTS),
        },
        "minimum_found": [],
        "x": [],
        "y": [],
        "z": [],
        "visible_length": [],
        "full_range": [],
        "ke0": [],
        "length": [],  # legacy alias for visible_length
        "t": [],
        "est_fcn": [],
        "true_fcn": [],
        "cx": [],
        "cy": [],
        "n_attempts": [],
        "chosen_seed_idx": [],
        "chosen_seed_fcn": [],
        "chosen_seed_params": [],
        "adaptive_rescue_used": [],
        "length_profile_rescue_considered": [],
        "length_profile_rescue_used": [],
        "fcn_retry_used": [],
        "edm": [],
    }
    if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
        est_dict["seed_scan"] = []

    if SAVE_ATTEMPT_RESULTS:
        est_dict["attempt_results"] = []

    n_events_per_batch = min(N_EVENTS_PER_BATCH, max(1, tot_events))

    for batch_start in range(0, tot_events, n_events_per_batch):
        batch_end = min(batch_start + n_events_per_batch, tot_events)
        event_indices = list(range(batch_start, batch_end))

        print(f"Starting event number {batch_start}")

        results = run_batch(
            event_indices=event_indices,
            init_param_sets=init_param_sets,
            nproc=NPROC,
            fcn_threshold=FCN_RETRY_THRESHOLD,
            max_attempts=MAX_FIT_ATTEMPTS,
            ncall=NCALL_MIGRAD,
        )

        for local_i, result in enumerate(results):
            event_index = event_indices[local_i]
            vals = result["values"]

            if IS_ABSORPTION_MODE:
                visible_length = float(vals["visible_length"])
                full_range = float(vals["full_range"])
            else:
                visible_length = float(vals["length"])
                full_range = visible_length
            ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))

            est_dict["minimum_found"].append(int(result["valid"]))
            est_dict["x"].append(vals["x0"])
            est_dict["y"].append(vals["y0"])
            est_dict["z"].append(vals["z0"])
            est_dict["visible_length"].append(visible_length)
            est_dict["full_range"].append(full_range)
            est_dict["ke0"].append(ke0)
            est_dict["length"].append(visible_length)
            est_dict["t"].append(vals["t0"])
            est_dict["cx"].append(vals["cx"])
            est_dict["cy"].append(vals["cy"])
            est_dict["est_fcn"].append(result["fval"])
            est_dict["true_fcn"].append(true_fcn_all[event_index])
            est_dict["n_attempts"].append(result["attempts"])
            est_dict["chosen_seed_idx"].append(result["chosen_seed_index"])
            est_dict["chosen_seed_fcn"].append(result["chosen_seed_fcn"])
            est_dict["chosen_seed_params"].append(result["chosen_seed_params"])
            if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
                est_dict["seed_scan"].append(result.get("seed_scan", []))
            est_dict["adaptive_rescue_used"].append(result.get("adaptive_rescue_used", False))
            est_dict["length_profile_rescue_considered"].append(result.get("length_profile_rescue_considered", False))
            est_dict["length_profile_rescue_used"].append(result.get("length_profile_rescue_used", False))
            est_dict["fcn_retry_used"].append(bool(result.get("fcn_retry_used", False) or result.get("attempts", 1) > 1))
            est_dict["edm"].append(result.get("edm", np.nan))
            if SAVE_ATTEMPT_RESULTS:
                est_dict["attempt_results"].append(result.get("attempt_results", []))

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(est_dict, f)

    print("Done.")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()










# # 7,8-parameter abrupt-endpoint batch driver.
# #
# # Fit parameters:
# #   x0, y0, z0, cx, cy, visible_length, full_range, t0
# #
# # Meaning:
# #   visible_length = actual visible primary-Cherenkov length before abrupt cutoff [mm]
# #   full_range     = dE/dx-only range to Cherenkov threshold if no abrupt interaction [mm]
# #   ke0            = inferred initial kinetic energy from full_range using particle range table [MeV]
# #
# # This driver assumes an abrupt endpoint model.  It does not use fixed_initial_KE
# # as a fit setting; fixed_initial_KE is overwritten inside every FCN call using
# # full_range -> ke0.

# """Toggleable multi-stage batch driver for the 8-parameter Minuit Cherenkov fit on WCTE/real-data-style events.

# This is the selected-event/input-array version of the driver.  It preserves:
#   - get_selected_events(RUN, N_EVENTS) event loading
#   - run configuration GOOD_WCTE_PMTS masking from the ROOT Configuration tree
#   - pe_scale=143
#   - estimated geometry placement "est"
#   - mPMT type/relative efficiency corrections when tables are available

# The 8 fitted parameters are:
#   x0, y0, z0, cx, cy, visible_length, full_range, t0
# """

# import os
# import sys
# import pickle
# import multiprocessing as mp
# from pathlib import Path

# import numpy as np
# import uproot
# from iminuit import Minuit

# # =============================================================================
# # SELF-CONTAINED PATH SETUP
# # =============================================================================
# SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = SCRIPT_DIR.parent
# LICKETYFIT_DIR = PROJECT_ROOT / "LicketyFit"
# TABLE_DIR = PROJECT_ROOT / "tables"
# OUTPUT_DIR = PROJECT_ROOT / "outputs"
# OUTPUT_DIR.mkdir(exist_ok=True)

# geometry_path = os.environ.get("GEOMETRY_PATH", "/eos/user/j/jrimmer/Geometry")
# GEOMETRY_FILE = os.environ.get(
#     "WCTE_GEOMETRY_FILE",
#     str(Path(geometry_path) / "examples" / "wcte_bldg157.geo"),
# )

# for _path in (str(LICKETYFIT_DIR), str(PROJECT_ROOT), str(SCRIPT_DIR), str(TABLE_DIR), geometry_path):
#     if _path not in sys.path:
#         sys.path.insert(0, _path)

# # Force local tables before importing lookup/collapse helpers.
# os.environ["LF_TABLE_DIR"] = str(TABLE_DIR)
# os.environ["LF_MULTIPARTICLES_TABLE_DIR"] = str(TABLE_DIR)

# from Geometry.Device import Device
# from LicketyFit.Event import Event
# from LicketyFit.PMT import PMT
# from LicketyFit.Emitter import Emitter
# from particle_cherenkov_model import (
#     get_energy_distance_tables,
#     set_active_particle,
#     canonical_particle_name,
#     particle_mass_mev,
#     cherenkov_threshold_kinetic_mev,
# )
# try:
#     from event_loader import get_selected_events
# except Exception:
#     get_selected_events = None
# from particle_range_lookup import ParticleRangeLookup


# # =============================================================================
# # ENV HELPERS
# # =============================================================================
# def _env_float(name, default=None):
#     raw = os.environ.get(name)
#     if raw is None or str(raw).strip() == "":
#         return default
#     return float(raw)


# def _env_int(name, default):
#     raw = os.environ.get(name)
#     if raw is None or str(raw).strip() == "":
#         return int(default)
#     return int(raw)


# def _env_bool(name, default=False):
#     raw = os.environ.get(name)
#     if raw is None:
#         return bool(default)
#     return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


# def _parse_float_list_env(name, default):
#     raw = os.environ.get(name)
#     if raw is None or str(raw).strip() == "":
#         return list(default)
#     return [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]


# # =============================================================================
# # TOP-LEVEL CONFIGURATION
# # =============================================================================
# N_EVENTS_PER_BATCH = _env_int("N_EVENTS_PER_BATCH", 200)
# NPROC = _env_int("NPROC", 16)
# M_STRAT = _env_int("M_STRAT", 1)
# #M_STRAT = _env_int("M_STRAT", 0)

# Z_SEED_EPS = 20.0
# VISIBLE_LENGTH_SEED_EPS = 40.0
# FULL_RANGE_SEED_EPS = 80.0
# T_MIN = -8.0

# FCN_RETRY_THRESHOLD = 1100.0
# VISIBLE_LENGTH_RETRY_THRESHOLD = _env_float("VISIBLE_LENGTH_RETRY_THRESHOLD", 2700.0)
# MAX_FIT_ATTEMPTS = _env_int("MAX_FIT_ATTEMPTS", 4)
# NCALL_MIGRAD = _env_int("NCALL_MIGRAD", 70000)
# NCALL_SIMPLEX = _env_int("NCALL_SIMPLEX", NCALL_MIGRAD)

# RUN = _env_int("RUN", 2079)
# BEAM_P = _env_float("BEAM_P", 430)
# N_EVENTS = _env_int("N_EVENTS", 60000)
# print('N_EVENTS',N_EVENTS)

# # =============================================================================
# # PARTICLE HYPOTHESIS / 8-PARAMETER MODE
# # =============================================================================
# FIT_PARTICLE = os.environ.get("FIT_PARTICLE", "muon")
# FIT_PARTICLE_CANONICAL = canonical_particle_name(FIT_PARTICLE)
# FIT_PARTICLE_MASS_MEV = particle_mass_mev(FIT_PARTICLE_CANONICAL)
# FIT_PARTICLE_THRESHOLD_MEV = cherenkov_threshold_kinetic_mev(FIT_PARTICLE_MASS_MEV)
# set_active_particle(FIT_PARTICLE_CANONICAL)

# # Fit mode:
# #   full_length -> original 7-parameter fit:
# #                  x0, y0, z0, cx, cy, length, t0
# #                  length is the dE/dx range to Cherenkov threshold, so ke0 is inferred from length.
# #   absorption  -> 8-parameter abrupt-endpoint fit:
# #                  x0, y0, z0, cx, cy, visible_length, full_range, t0
# #                  visible_length is the abrupt cutoff; full_range determines ke0.
# _FIT_MODE_RAW = os.environ.get("FIT_MODE", os.environ.get("TRACK_END_MODE", "full_length")).strip().lower()

# if _FIT_MODE_RAW in {"absorption", "absorbed", "abrupt", "abrupt_8param", "interaction", "truncated"}:
#     FIT_MODE = "absorption"
# elif _FIT_MODE_RAW in {"full_length", "full-length", "full", "threshold", "range", "csda", "old", "7param", "7_parameter"}:
#     FIT_MODE = "full_length"
# else:
#     raise ValueError("FIT_MODE/TRACK_END_MODE must be 'full_length' or 'absorption'")

# TRACK_END_MODE = FIT_MODE
# IS_ABSORPTION_MODE = FIT_MODE == "absorption"
# IS_FULL_LENGTH_MODE = FIT_MODE == "full_length"
# EMITTER_TRACK_END_MODE = "abrupt" if IS_ABSORPTION_MODE else "threshold"
# FIT_PARAMETER_NAMES = (
#     ("x0", "y0", "z0", "cx", "cy", "visible_length", "full_range", "t0")
#     if IS_ABSORPTION_MODE
#     else ("x0", "y0", "z0", "cx", "cy", "length", "t0")
# )

# # Output/debug controls.  The full seed scan is large in the 8-parameter
# # fit because every event can have thousands of seed dictionaries.  Keep these
# # off for production output; enable only when debugging seed selection.
# SAVE_ATTEMPT_RESULTS = _env_bool("SAVE_ATTEMPT_RESULTS", False)
# SAVE_SEED_SCAN = _env_bool("SAVE_SEED_SCAN", False)
# SAVE_TOP_N_SEEDS = _env_int("SAVE_TOP_N_SEEDS", 0)

# # Likelihood toggles.
# USE_CHARGE_LIKELIHOOD = _env_bool("USE_CHARGE_LIKELIHOOD", True)
# USE_TIMING_LIKELIHOOD = _env_bool("USE_TIMING_LIKELIHOOD", True)
# USE_T0_PRIOR = _env_bool("USE_T0_PRIOR", False)

# if (not USE_CHARGE_LIKELIHOOD) and (not USE_TIMING_LIKELIHOOD):
#     raise ValueError("At least one likelihood term must be enabled.")

# if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
#     LIKELIHOOD_MODE = "charge_time"
# elif USE_CHARGE_LIKELIHOOD:
#     LIKELIHOOD_MODE = "charge_only"
# else:
#     LIKELIHOOD_MODE = "timing_only"

# OUTPUT_FILE = os.environ.get(
#     "LF_OUTPUT_FILE",
#     str(OUTPUT_DIR / f"estimates_run{RUN}_{BEAM_P:g}p_{FIT_PARTICLE_CANONICAL}_{TRACK_END_MODE}_mpmtEff_{LIKELIHOOD_MODE}.dict"),
# )

# RING_MASK_MODE = os.environ.get("RING_MASK_MODE", "both").strip().lower()

# # Data configuration.
# CONFIG_ROOT_FILE = os.environ.get(
#     "CONFIG_ROOT_FILE",
#     f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/{RUN}/WCTE_merged_production_R{RUN}.root",
# )

# # Event source.
# #   "selection" / "internal" : use event_loader.get_selected_events(...) with particle-specific TOF cuts.
# #   "file" / "user" / "custom" : load already-selected user events from USER_EVENT_FILE.
# #
# # USER_EVENT_FILE can be .npy, .npz, .pkl, or .pickle.  Supported payloads:
# #   - list/object array of event arrays, each with columns [pmt_id, charge, time] or
# #     [pmt_id, charge, time, event_number]
# #   - a single 2D array [pmt_id, charge, time] for one event
# #   - a single 2D array [pmt_id, charge, time, event_number], which is grouped by
# #     event_number
# EVENT_SOURCE = os.environ.get("EVENT_SOURCE", "selection").strip().lower()
# if EVENT_SOURCE in {"selected", "internal", "event_loader", "auto"}:
#     EVENT_SOURCE = "selection"
# elif EVENT_SOURCE in {"file", "user", "custom", "user_file", "provided"}:
#     EVENT_SOURCE = "file"
# if EVENT_SOURCE not in {"selection", "file"}:
#     raise ValueError("EVENT_SOURCE must be 'selection' or 'file'")

# USER_EVENT_FILE = os.environ.get("USER_EVENT_FILE", "").strip()
# USER_EVENT_KEY = os.environ.get("USER_EVENT_KEY", "").strip() or None
# USER_EVENT_APPLY_PEAK_WINDOW = _env_bool("USER_EVENT_APPLY_PEAK_WINDOW", True)

# # Event-selection configuration for event_loader.get_selected_events().
# # Defaults reproduce the historical muon-like WCTE selection.  For non-muon
# # beam selections, set PARTICLE_SELECTION_LABEL plus either SELECTION_TOF_NS
# # or SELECTION_TOF_FIELD/T5_PARTICLE_NR as needed for your production ROOT file.
# PARTICLE_SELECTION_LABEL = os.environ.get("PARTICLE_SELECTION_LABEL", FIT_PARTICLE_CANONICAL)
# SELECTION_TOF_NS = _env_float("SELECTION_TOF_NS", None)
# SELECTION_TOF_WINDOW_NS = _env_float("SELECTION_TOF_WINDOW_NS", 0.2)
# SELECTION_TOF_FIELD = os.environ.get("SELECTION_TOF_FIELD", "") or None
# SELECTION_MOMENTUM_FIELD = os.environ.get("SELECTION_MOMENTUM_FIELD", "") or None
# SELECTION_T5_PARTICLE_NR = _env_int("SELECTION_T5_PARTICLE_NR", 1)
# USE_PEAK_TIME_CUT = _env_bool("USE_PEAK_TIME_CUT", True)
# PEAK_WINDOW_NS = _env_float("PEAK_WINDOW_NS", 100.0)
# PEAK_BIN_WIDTH_NS = _env_float("PEAK_BIN_WIDTH_NS", 50.0)

# # =============================================================================
# # DETECTOR CONFIGURATION
# # =============================================================================
# DEFAULT_INACTIVE_SLOTS = [27, 32, 45, 74, 77, 79, 85, 91, 99, 9, 67]
# INACTIVE_SLOTS = [int(x) for x in os.environ.get(
#     "INACTIVE_SLOTS",
#     ",".join(str(x) for x in DEFAULT_INACTIVE_SLOTS),
# ).replace(";", ",").split(",") if x.strip()]
# INACTIVE_SLOTS_SET = set(int(s) for s in INACTIVE_SLOTS)

# OUTER_RING = np.array([0, 7, 19, 34, 50, 66, 82, 83, 105, 94, 95, 71, 72, 56, 40, 24, 11, 3, 18])
# INNER_RING = np.array([1, 8, 35, 51, 67, 84, 69, 70, 55, 39, 23, 10, 2, 20, 36, 52, 68, 53, 54, 38, 22, 21, 37, 9])
# OUTSIDE_RING = np.array([12, 13, 4, 5, 6, 17, 33, 49, 65, 81, 82, 104, 93, 86, 87, 72, 57, 41, 25])
# ALL_RING = np.arange(0, 106)

# # Optional truth diagnostics.  If not supplied, true_fcn is NaN.
# TRUE_LENGTH_MM = _env_float("TRUE_LENGTH_MM", None)
# TRUE_VISIBLE_LENGTH_MM = _env_float("TRUE_VISIBLE_LENGTH_MM", None)
# TRUE_FULL_RANGE_MM = _env_float("TRUE_FULL_RANGE_MM", None)
# TRUE_INITIAL_KE_MEV = _env_float("TRUE_INITIAL_KE_MEV", None)

# TRUE_PARAMS = {
#     "x0": _env_float("TRUE_X0", 0.0),
#     "y0": _env_float("TRUE_Y0", 0.0),
#     "z0": _env_float("TRUE_Z0", -1348.0),
#     "cx": _env_float("TRUE_CX", 0.0),
#     "cy": _env_float("TRUE_CY", 0.0),
#     "visible_length": np.nan,
#     "full_range": np.nan,
#     "t0": _env_float("TRUE_T0", 0.0),
# }

# # =============================================================================
# # GLOBAL FIT-SEARCH STAGE TOGGLES
# # =============================================================================
# ENABLE_STAGE1_SEED_GRID = _env_bool("ENABLE_STAGE1_SEED_GRID", True)
# ENABLE_STAGE2_MIGRAD_FIRST = _env_bool("ENABLE_STAGE2_MIGRAD_FIRST", False)
# ENABLE_STAGE3_ADAPTIVE_RESCUE = _env_bool("ENABLE_STAGE3_ADAPTIVE_RESCUE", False)
# ENABLE_STAGE4_LENGTH_PROFILE = _env_bool("ENABLE_STAGE4_LENGTH_PROFILE", False)

# # Keep the full seed ranking only when it is actually needed.  This avoids
# # sorting and returning thousands of seed dictionaries for normal production fits.
# NEED_FULL_SEED_SCAN = (
#     ENABLE_STAGE3_ADAPTIVE_RESCUE
#     or ENABLE_STAGE4_LENGTH_PROFILE
#     or SAVE_SEED_SCAN
#     or SAVE_TOP_N_SEEDS > 0
# )

# # =============================================================================
# # INITIAL SEED CONFIGURATION
# # =============================================================================
# FAST_SEED_X0 = _parse_float_list_env("FAST_SEED_X0", [-100.0, -50, 0.0, 50, 100.0])
# FAST_SEED_Y0 = _parse_float_list_env("FAST_SEED_Y0", [-100.0, -50, 0.0, 50, 100.0])
# FAST_SEED_Z0 = _parse_float_list_env("FAST_SEED_Z0", [-1500.0, -1400.0, -1300.0, -1350, -1200.0, -1100.0, -1000.0])

# FAST_SEED_VISIBLE_LENGTHS = _parse_float_list_env(
#     "FAST_SEED_VISIBLE_LENGTHS",
#     [100.0, 150, 200, 250, 300.0, 350, 400, 450, 500.0, 700.0, 900.0, 1100.0, 1300.0, 1400.0, 1500.0, 1700.0, 1900.0],
# )

# FAST_SEED_KE0_MEV = _parse_float_list_env(
#     "FAST_SEED_KE0_MEV",
#     [600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0],
# )
# FAST_SEED_FULL_RANGES_MM = _parse_float_list_env(
#     "FAST_SEED_FULL_RANGES_MM",
#     [300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0],
# )
# FAST_SEED_DIRECTIONS = [
#     (0.0, 0.0),
#     (0.04, 0.0),
#     (-0.04, 0.0),
#     (0.0, 0.04),
#     (0.0, -0.04),
# ]
# FAST_SEED_FULL_CARTESIAN = _env_bool("FAST_SEED_FULL_CARTESIAN", False)


# def build_sparse_geometry_variants():
#     variants = []
#     for x0 in FAST_SEED_X0:
#         variants.append({"x0": float(x0), "y0": 0.0, "cx": 0.0, "cy": 0.0})
#     for y0 in FAST_SEED_Y0:
#         variants.append({"x0": 0.0, "y0": float(y0), "cx": 0.0, "cy": 0.0})
#     for cx, cy in FAST_SEED_DIRECTIONS:
#         variants.append({"x0": 0.0, "y0": 0.0, "cx": float(cx), "cy": float(cy)})

#     unique = []
#     seen = set()
#     for v in variants:
#         sig = (float(v["x0"]), float(v["y0"]), float(v["cx"]), float(v["cy"]))
#         if sig not in seen:
#             seen.add(sig)
#             unique.append(v)
#     return unique


# FAST_SEED_GEOMETRY_VARIANTS = build_sparse_geometry_variants()


# def build_full_range_seed_values(range_lookup):
#     values = []
#     for ke0 in FAST_SEED_KE0_MEV:
#         if ke0 <= range_lookup.threshold_mev:
#             continue
#         r = range_lookup.energy_to_range_mm(float(ke0))
#         if np.isfinite(r) and r > 0:
#             values.append(float(r))

#     values.extend(float(r) for r in FAST_SEED_FULL_RANGES_MM)

#     max_r = float(range_lookup.overall_distances_mm[-1])
#     values = [r for r in values if np.isfinite(r) and 0.0 < r <= max_r]

#     unique = []
#     seen = set()
#     for r in values:
#         sig = round(float(r), 6)
#         if sig not in seen:
#             seen.add(sig)
#             unique.append(float(r))
#     return unique


# def build_fast_seed_grid(range_lookup):
#     seeds = []

#     if IS_FULL_LENGTH_MODE:
#         # 7-parameter original/full-length mode: scan only one length-like parameter.
#         if FAST_SEED_FULL_CARTESIAN:
#             for x0 in FAST_SEED_X0:
#                 for y0 in FAST_SEED_Y0:
#                     for z0 in FAST_SEED_Z0:
#                         for length in FAST_SEED_VISIBLE_LENGTHS:
#                             for cx, cy in FAST_SEED_DIRECTIONS:
#                                 seeds.append({
#                                     "x0": float(x0),
#                                     "y0": float(y0),
#                                     "z0": float(z0),
#                                     "cx": float(cx),
#                                     "cy": float(cy),
#                                     "length": float(length),
#                                     "t0": 0.0,
#                                 })
#         else:
#             for z0 in FAST_SEED_Z0:
#                 for length in FAST_SEED_VISIBLE_LENGTHS:
#                     for geom in FAST_SEED_GEOMETRY_VARIANTS:
#                         seeds.append({
#                             "x0": float(geom["x0"]),
#                             "y0": float(geom["y0"]),
#                             "z0": float(z0),
#                             "cx": float(geom["cx"]),
#                             "cy": float(geom["cy"]),
#                             "length": float(length),
#                             "t0": 0.0,
#                         })
#         keys = ("x0", "y0", "z0", "cx", "cy", "length", "t0")
#     else:
#         # 8-parameter absorption mode: scan visible cutoff length and full CSDA range separately.
#         full_range_seeds = build_full_range_seed_values(range_lookup)
#         if FAST_SEED_FULL_CARTESIAN:
#             for x0 in FAST_SEED_X0:
#                 for y0 in FAST_SEED_Y0:
#                     for z0 in FAST_SEED_Z0:
#                         for visible_length in FAST_SEED_VISIBLE_LENGTHS:
#                             for full_range in full_range_seeds:
#                                 if visible_length > full_range:
#                                     continue
#                                 for cx, cy in FAST_SEED_DIRECTIONS:
#                                     seeds.append({
#                                         "x0": float(x0),
#                                         "y0": float(y0),
#                                         "z0": float(z0),
#                                         "cx": float(cx),
#                                         "cy": float(cy),
#                                         "visible_length": float(visible_length),
#                                         "full_range": float(full_range),
#                                         "t0": 0.0,
#                                     })
#         else:
#             for z0 in FAST_SEED_Z0:
#                 for visible_length in FAST_SEED_VISIBLE_LENGTHS:
#                     for full_range in full_range_seeds:
#                         if visible_length > full_range:
#                             continue
#                         for geom in FAST_SEED_GEOMETRY_VARIANTS:
#                             seeds.append({
#                                 "x0": float(geom["x0"]),
#                                 "y0": float(geom["y0"]),
#                                 "z0": float(z0),
#                                 "cx": float(geom["cx"]),
#                                 "cy": float(geom["cy"]),
#                                 "visible_length": float(visible_length),
#                                 "full_range": float(full_range),
#                                 "t0": 0.0,
#                             })
#         keys = ("x0", "y0", "z0", "cx", "cy", "visible_length", "full_range", "t0")

#     unique = []
#     seen = set()
#     for seed in seeds:
#         sig = tuple(float(seed[k]) for k in keys)
#         if sig not in seen:
#             seen.add(sig)
#             unique.append(seed)
#     return unique


# PARAM_NAMES = FIT_PARAMETER_NAMES

# # =============================================================================
# # GLOBALS SHARED BY WORKERS
# # =============================================================================
# OVERALL_DISTANCES = None
# INIT_ENERGY_TABLE = None
# RANGE_LOOKUP = None

# WCD = None
# PMT_MODEL = None
# EMITTER_TEMPLATE = None
# P_LOCATIONS = None
# DIRECTION_ZS = None
# RING_KEEP_MASK = None
# CORR_POS = None
# MPMT_SLOTS_ALL = None
# MPMT_TYPE_CODES_ALL = None

# OBS_PES_ALL = None
# OBS_TS_ALL = None
# GOOD_WCTE_PMTS_SET = None

# # =============================================================================
# # mPMT INFO / EFFICIENCY TABLES
# # =============================================================================
# other_mpmt_info_path = Path(os.environ.get("OTHER_MPMT_INFO_PATH", str(TABLE_DIR / "other_mpmt_info_v2.dict")))
# if other_mpmt_info_path.exists():
#     with open(other_mpmt_info_path, "rb") as f:
#         mpmt_info = pickle.load(f)
# else:
#     mpmt_info = {}

# rel_mpmt_eff_path = Path(os.environ.get("REL_MPMT_EFF_PATH", str(TABLE_DIR / "rel_mpmt_eff.dict")))
# if rel_mpmt_eff_path.exists():
#     with open(rel_mpmt_eff_path, "rb") as f:
#         rel_mpmt_eff = pickle.load(f)
# else:
#     unity = np.ones(200, dtype=np.float64)
#     rel_mpmt_eff = {
#         "tri_exsitu": unity,
#         "tri_insitu": unity,
#         "wut_insitu": unity,
#         "wut_exsitu": unity,
#     }

# tri_exsitu = rel_mpmt_eff["tri_exsitu"]
# tri_insitu = rel_mpmt_eff["tri_insitu"]
# wut_insitu = rel_mpmt_eff["wut_insitu"]
# wut_exsitu = rel_mpmt_eff["wut_exsitu"]


# def get_mpmt_slot_type(mpmt_slots):
#     slot_type = []
#     for slot in mpmt_slots:
#         slot = int(slot)
#         try:
#             if mpmt_info[slot]["mpmt_site"] == "TRI":
#                 if mpmt_info[slot]["mpmt_type"] == "In-situ":
#                     slot_type.append("tri_insitu")
#                 else:
#                     slot_type.append("tri_exsitu")
#             else:
#                 if mpmt_info[slot]["mpmt_type"] == "In-situ":
#                     slot_type.append("wut_insitu")
#                 else:
#                     slot_type.append("wut_exsitu")
#         except Exception:
#             slot_type.append("empty")
#     return slot_type



# _MPMT_TYPE_TO_CODE_LOCAL = {
#     "tri_exsitu": 0,
#     "tri_insitu": 1,
#     "wut_exsitu": 2,
#     "wut_insitu": 3,
# }

# def get_mpmt_slot_type_codes(mpmt_slots):
#     """Same information as get_mpmt_slot_type(), but encoded once as int8.

#     Emitter treats integer mPMT types directly, avoiding repeated string
#     comparisons in seed scans and the first FCN call of each Minuit attempt.
#     """
#     types = get_mpmt_slot_type(mpmt_slots)
#     return np.asarray([_MPMT_TYPE_TO_CODE_LOCAL.get(t, -1) for t in types], dtype=np.int8)

# # =============================================================================
# # CONFIG / TRUTH HELPERS
# # =============================================================================
# def load_good_wcte_pmts():
#     try:
#         with uproot.open(CONFIG_ROOT_FILE) as f:
#             t_c = f["Configuration"]
#             arr_config = t_c.arrays(library="ak")
#         good = np.asarray(arr_config["good_wcte_pmts"][0], dtype=int)
#         print("Loaded GOOD_WCTE_PMTS from:", CONFIG_ROOT_FILE)
#         return set(good.tolist())
#     except Exception as exc:
#         # For selected ROOT input, missing the run Configuration tree is usually a
#         # real problem.  For user-provided event files, allow a self-contained
#         # fallback by default and turn on every non-inactive PMT.
#         allow_missing_default = EVENT_SOURCE == "file"
#         if _env_bool("ALLOW_MISSING_GOOD_PMTS", allow_missing_default):
#             print("WARNING: could not load GOOD_WCTE_PMTS; using all non-inactive PMTs.")
#             print("Reason:", repr(exc))
#             all_ids = []
#             for slot in range(106):
#                 if slot in INACTIVE_SLOTS_SET:
#                     continue
#                 for pmt_pos in range(19):
#                     all_ids.append(slot * 100 + pmt_pos)
#             return set(all_ids)
#         raise


# def configure_truth_params():
#     if IS_ABSORPTION_MODE:
#         if TRUE_FULL_RANGE_MM is not None:
#             TRUE_PARAMS["full_range"] = float(TRUE_FULL_RANGE_MM)
#         elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
#             TRUE_PARAMS["full_range"] = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))

#         if TRUE_VISIBLE_LENGTH_MM is not None:
#             TRUE_PARAMS["visible_length"] = float(TRUE_VISIBLE_LENGTH_MM)
#         elif TRUE_LENGTH_MM is not None:
#             TRUE_PARAMS["visible_length"] = float(TRUE_LENGTH_MM)
#         else:
#             TRUE_PARAMS["visible_length"] = np.nan

#         if np.isfinite(TRUE_PARAMS["visible_length"]):
#             TRUE_PARAMS["length"] = TRUE_PARAMS["visible_length"]
#         return

#     # Full-length mode: the single fitted length is also the full CSDA range.
#     if TRUE_LENGTH_MM is not None:
#         length = float(TRUE_LENGTH_MM)
#     elif TRUE_FULL_RANGE_MM is not None:
#         length = float(TRUE_FULL_RANGE_MM)
#     elif TRUE_VISIBLE_LENGTH_MM is not None:
#         length = float(TRUE_VISIBLE_LENGTH_MM)
#     elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
#         length = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))
#     else:
#         length = np.nan

#     TRUE_PARAMS["length"] = length
#     TRUE_PARAMS["visible_length"] = length
#     TRUE_PARAMS["full_range"] = length


# # =============================================================================
# # EVENT / OBSERVABLE HELPERS
# # =============================================================================
# def sim_to_event(
#     sim_data,
#     WCD,
#     n_mpmt_total=106,
#     pe_scale=143,
#     shift_times=True,
#     n_earliest_for_t0=10,
# ):
#     vw = 223.0598645833333  # mm/ns

#     ev = Event(0, 0, n_mpmt_total)
#     ev.set_mpmt_status(list(range(n_mpmt_total)), False)

#     active_wcte_pmt_ids = []

#     for slot in range(n_mpmt_total):
#         if slot in INACTIVE_SLOTS_SET:
#             continue

#         slot_has_good_pmt = False
#         for pmt_pos_id in range(ev.npmt_per_mpmt):
#             wcte_pmt = int(slot * 100 + pmt_pos_id)
#             if wcte_pmt in GOOD_WCTE_PMTS_SET:
#                 ev.set_pmt_status(slot, [pmt_pos_id], True)
#                 slot_has_good_pmt = True
#                 active_wcte_pmt_ids.append(wcte_pmt)

#         if slot_has_good_pmt:
#             ev.set_mpmt_status([slot], True)

#     for i in range(len(sim_data[:, 0])):
#         wcte_pmt = int(sim_data[i, 0])
#         slot = int(wcte_pmt // 100)
#         pmt_pos_id = int(wcte_pmt % 100)

#         if slot < 0 or slot >= ev.n_mpmt:
#             continue
#         if pmt_pos_id < 0 or pmt_pos_id >= ev.npmt_per_mpmt:
#             continue
#         if not ev.mpmt_status[slot]:
#             continue
#         if not ev.pmt_status[slot][pmt_pos_id]:
#             continue

#         ev.hit_charges[slot][pmt_pos_id].append(float(sim_data[i, 1]))
#         ev.hit_times[slot][pmt_pos_id].append(float(sim_data[i, 2]))

#     if shift_times:
#         bp_loc = np.array([0.0, 0.0, -1350.0])
#         early_hits = []

#         for i_mpmt in range(ev.n_mpmt):
#             if not ev.mpmt_status[i_mpmt]:
#                 continue
#             for i_pmt in range(ev.npmt_per_mpmt):
#                 if not ev.pmt_status[i_mpmt][i_pmt]:
#                     continue
#                 if len(ev.hit_times[i_mpmt][i_pmt]) == 0:
#                     continue

#                 pmt_loc = WCD.mpmts[i_mpmt].pmts[i_pmt].get_placement("est")["location"]
#                 r = np.linalg.norm(pmt_loc - bp_loc)

#                 for t in ev.hit_times[i_mpmt][i_pmt]:
#                     early_hits.append({
#                         "time": float(t),
#                         "t0_est": float(t) - r / vw,
#                     })

#         if len(early_hits) > 0:
#             early_hits = sorted(early_hits, key=lambda x: x["time"])
#             n_use = min(n_earliest_for_t0, len(early_hits))
#             time_offset = np.median([hit["t0_est"] for hit in early_hits[:n_use]])

#             for i_mpmt in range(ev.n_mpmt):
#                 for i_pmt in range(ev.npmt_per_mpmt):
#                     ev.hit_times[i_mpmt][i_pmt] = [
#                         t - time_offset for t in ev.hit_times[i_mpmt][i_pmt]
#                     ]

#             ev.global_time_offset = time_offset

#     return ev, np.asarray(active_wcte_pmt_ids, dtype=int)


# def build_observables_from_event(ev, pe_scale=143):
#     obs_pes = []
#     obs_ts = []

#     for i_mpmt in range(ev.n_mpmt):
#         if not ev.mpmt_status[i_mpmt]:
#             continue
#         for i_pmt in range(ev.npmt_per_mpmt):
#             if not ev.pmt_status[i_mpmt][i_pmt]:
#                 continue

#             q = np.asarray(ev.hit_charges[i_mpmt][i_pmt], dtype=np.float64)
#             t = np.asarray(ev.hit_times[i_mpmt][i_pmt], dtype=np.float64)

#             if q.size == 0:
#                 obs_pes.append(0.0)
#                 obs_ts.append(np.nan)
#             else:
#                 obs_pes.append(float(np.sum(q)) / pe_scale)
#                 obs_ts.append(float(np.sum(q * t) / np.sum(q)))

#     return np.asarray(obs_pes, dtype=np.float64), np.asarray(obs_ts, dtype=np.float64)


# def apply_ring_mask_to_observables(obs_pes, obs_ts, ring_keep_mask, mode="both"):
#     obs_pes = obs_pes.copy()
#     obs_ts = obs_ts.copy()

#     if mode not in {"none", "pes", "ts", "both"}:
#         raise ValueError("RING_MASK_MODE must be one of: none, pes, ts, both")

#     if mode in {"pes", "both"}:
#         obs_pes[~ring_keep_mask] = 0.0
#     if mode in {"ts", "both"}:
#         obs_ts[~ring_keep_mask] = np.nan

#     return obs_pes, obs_ts


# def get_t0_prior_sigma(obs_pes, obs_ts):
#     n_timed = np.count_nonzero(np.isfinite(obs_ts))
#     total_pe = np.sum(obs_pes)

#     if (n_timed < 250) or (total_pe < 300):
#         return 0.1
#     elif (n_timed < 275) or (total_pe < 350):
#         return 0.2
#     elif (n_timed < 300) or (total_pe < 400):
#         return 0.3
#     elif (n_timed < 325) or (total_pe < 450):
#         return 0.4
#     elif (n_timed < 350) or (total_pe < 500):
#         return 0.5
#     elif (n_timed < 375) or (total_pe < 550):
#         return 0.6
#     elif (n_timed < 400) or (total_pe < 600):
#         return 0.7
#     elif (n_timed < 425) or (total_pe < 650):
#         return 0.8
#     elif (n_timed < 450) or (total_pe < 700):
#         return 1.0
#     elif (n_timed < 475) or (total_pe < 750):
#         return 1.2
#     elif (n_timed < 500) or (total_pe < 800):
#         return 1.4
#     elif (n_timed < 525) or (total_pe < 850):
#         return 1.6
#     elif (n_timed < 550) or (total_pe < 900):
#         return 1.8
#     else:
#         return 2.0


# # =============================================================================
# # LIKELIHOOD EVALUATION
# # =============================================================================
# def get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts):
#     exp_pes = np.asarray(exp_pes, dtype=np.float64)
#     obs_pes = np.asarray(obs_pes, dtype=np.float64)
#     exp_ts = np.asarray(exp_ts, dtype=np.float64)
#     obs_ts = np.asarray(obs_ts, dtype=np.float64)

#     mask = (
#         (exp_pes > 0.0)
#         & (obs_pes > 0.0)
#         & np.isfinite(exp_ts)
#         & np.isfinite(obs_ts)
#     )

#     if not np.any(mask):
#         return 1e30

#     sigma_t = PMT_MODEL.single_pe_time_std / np.sqrt(obs_pes[mask])
#     dt = (obs_ts[mask] - exp_ts[mask]) / sigma_t
#     return float(0.5 * np.sum(dt * dt))


# def evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts):
#     if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
#         return PMT_MODEL.get_neg_log_likelihood_npe_t(exp_pes, obs_pes, exp_ts, obs_ts)
#     if USE_CHARGE_LIKELIHOOD:
#         return PMT_MODEL.get_neg_log_likelihood_npe(exp_pes, obs_pes)
#     return get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts)


# def evaluate_neg_log_likelihood(
#     obs_pes,
#     obs_ts,
#     emitter,
#     mpmt_types,
#     x0,
#     y0,
#     z0,
#     cx,
#     cy,
#     length_or_visible,
#     full_range_or_t0,
#     t0=None,
# ):
#     """Evaluate the selected fit mode.

#     full_length mode receives: length_or_visible=length, full_range_or_t0=t0.
#     absorption mode receives:  length_or_visible=visible_length, full_range_or_t0=full_range, t0=t0.
#     """
#     if IS_ABSORPTION_MODE:
#         visible_length = float(length_or_visible)
#         full_range = float(full_range_or_t0)
#         t0 = float(t0)

#         if not np.isfinite(visible_length) or not np.isfinite(full_range):
#             return 1e30
#         if visible_length < 0.0 or full_range <= 0.0:
#             return 1e30
#         if visible_length > full_range:
#             return 1e30
#         if full_range > float(RANGE_LOOKUP.overall_distances_mm[-1]):
#             return 1e30

#         ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))
#         if (not np.isfinite(ke0)) or ke0 <= FIT_PARTICLE_THRESHOLD_MEV:
#             return 1e30
#         emitter.fixed_initial_KE = ke0
#         track_length_for_emission = visible_length
#     else:
#         length = float(length_or_visible)
#         t0 = float(full_range_or_t0)
#         if not np.isfinite(length) or length < 0.0:
#             return 1e30
#         if length > float(RANGE_LOOKUP.overall_distances_mm[-1]):
#             return 1e30
#         emitter.fixed_initial_KE = None
#         track_length_for_emission = length

#     cz2 = 1.0 - cx * cx - cy * cy
#     if cz2 <= 0.0:
#         return 1e30

#     cz = np.sqrt(cz2)
#     emitter.start_coord = (float(x0), float(y0), float(z0))
#     emitter.starting_time = float(t0)
#     emitter.direction = (float(cx), float(cy), float(cz))

#     init_ke = emitter.refresh_kinematics_from_length(track_length_for_emission)

#     if hasattr(emitter, "visible_length_is_physical"):
#         if not emitter.visible_length_is_physical():
#             return 1e30
#     elif getattr(emitter, "last_visible_length_exceeds_range", False):
#         return 1e30

#     s = emitter.get_emission_points(P_LOCATIONS, init_ke)
#     exp_pes, exp_ts = emitter.get_expected_pes_ts(
#         WCD,
#         s,
#         P_LOCATIONS,
#         DIRECTION_ZS,
#         mpmt_types,
#         obs_pes,
#         need_times=USE_TIMING_LIKELIHOOD,
#     )

#     nll = evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts)
#     if not np.isfinite(nll):
#         return 1e30

#     if USE_TIMING_LIKELIHOOD and USE_T0_PRIOR:
#         sigma_t0 = get_t0_prior_sigma(obs_pes, obs_ts)
#         nll += abs(0.5 * (float(t0) / sigma_t0) ** 2)

#     return float(nll)


# def _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed):
#     if IS_ABSORPTION_MODE:
#         return evaluate_neg_log_likelihood(
#             obs_pes, obs_ts, emitter, mpmt_types,
#             seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
#             seed["visible_length"], seed["full_range"], seed["t0"],
#         )
#     return evaluate_neg_log_likelihood(
#         obs_pes, obs_ts, emitter, mpmt_types,
#         seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
#         seed["length"], seed["t0"],
#     )


# def select_best_initial_seed(obs_pes, obs_ts, init_param_sets, mpmt_types=None):
#     """
#     Cheap deterministic seed prescan.

#     Always retains the top MAX_FIT_ATTEMPTS seeds (sorted best-first) so that
#     the FCN retry loop in fit_one_event_by_index can step through the next-best
#     untried seed on each attempt.  When NEED_FULL_SEED_SCAN is True the full
#     sorted scan is also kept for rescue/debug output.
#     """
#     best_info = None
#     seed_scan = [] if NEED_FULL_SEED_SCAN else None

#     # Always keep at least MAX_FIT_ATTEMPTS seeds for the retry loop regardless
#     # of NEED_FULL_SEED_SCAN.  We maintain a small sorted list as we go; this
#     # is O(N * k log k) where k = MAX_FIT_ATTEMPTS which is negligible.
#     top_n_for_retry = int(MAX_FIT_ATTEMPTS)
#     top_seeds_buffer = []  # kept sorted best-first, length <= top_n_for_retry

#     for i, seed in enumerate(init_param_sets):
#         emitter = EMITTER_TEMPLATE.copy()

#         fval = _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed)

#         if not np.isfinite(fval):
#             fval = np.inf

#         info = {
#             "seed_index": int(i),
#             "fval": float(fval),
#             "params": dict(seed),
#         }

#         if seed_scan is not None:
#             seed_scan.append(info)

#         if best_info is None or fval < best_info["fval"]:
#             best_info = info

#         # Maintain a compact sorted buffer of the top-N seeds for retries.
#         top_seeds_buffer.append(info)
#         top_seeds_buffer.sort(key=lambda x: x["fval"])
#         if len(top_seeds_buffer) > top_n_for_retry:
#             top_seeds_buffer = top_seeds_buffer[:top_n_for_retry]

#     if best_info is None or not np.isfinite(best_info["fval"]):
#         raise RuntimeError("All seed FCNs were non-finite. Check ranges, PMT ordering, and event selection.")

#     if seed_scan is not None:
#         seed_scan_sorted = sorted(seed_scan, key=lambda x: x["fval"])
#     else:
#         # Use only the retained top-N seeds; already sorted best-first.
#         seed_scan_sorted = top_seeds_buffer

#     best = seed_scan_sorted[0]
#     return dict(best["params"]), int(best["seed_index"]), float(best["fval"]), seed_scan_sorted


# def compute_true_fcn_for_event(event_index):
#     if IS_ABSORPTION_MODE:
#         if not np.isfinite(TRUE_PARAMS.get("visible_length", np.nan)):
#             return np.nan
#         if not np.isfinite(TRUE_PARAMS.get("full_range", np.nan)):
#             return np.nan
#         length_args = (TRUE_PARAMS["visible_length"], TRUE_PARAMS["full_range"], TRUE_PARAMS["t0"])
#     else:
#         if not np.isfinite(TRUE_PARAMS.get("length", np.nan)):
#             return np.nan
#         length_args = (TRUE_PARAMS["length"], TRUE_PARAMS["t0"])

#     mpmt_types = MPMT_TYPE_CODES_ALL[event_index]
#     emitter = EMITTER_TEMPLATE.copy()
#     return evaluate_neg_log_likelihood(
#         OBS_PES_ALL[event_index],
#         OBS_TS_ALL[event_index],
#         emitter,
#         mpmt_types,
#         TRUE_PARAMS["x0"],
#         TRUE_PARAMS["y0"],
#         TRUE_PARAMS["z0"],
#         TRUE_PARAMS["cx"],
#         TRUE_PARAMS["cy"],
#         *length_args,
#     )


# # =============================================================================
# # MINUIT HELPERS
# # =============================================================================
# def make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types=None):
#     emitter = EMITTER_TEMPLATE.copy()

#     if IS_ABSORPTION_MODE:
#         def nll(x0, y0, z0, cx, cy, visible_length, full_range, t0):
#             return evaluate_neg_log_likelihood(
#                 obs_pes, obs_ts, emitter, mpmt_types,
#                 x0, y0, z0, cx, cy, visible_length, full_range, t0,
#             )
#     else:
#         def nll(x0, y0, z0, cx, cy, length, t0):
#             return evaluate_neg_log_likelihood(
#                 obs_pes, obs_ts, emitter, mpmt_types,
#                 x0, y0, z0, cx, cy, length, t0,
#             )

#     m = Minuit(nll, **start_params)

#     max_range = float(RANGE_LOOKUP.overall_distances_mm[-1])

#     m.limits["x0"] = (-2000, 2000)
#     m.limits["y0"] = (-2000, 2000)
#     m.limits["z0"] = (-2000, 2000)
#     m.limits["cx"] = (-0.5, 0.5)
#     m.limits["cy"] = (-0.5, 0.5)
#     m.limits["t0"] = (-8.0,8.0) #T0_LIMITS

#     m.errors["x0"] = 30.0
#     m.errors["y0"] = 30.0
#     m.errors["z0"] = 30.0
#     m.errors["cx"] = 0.01
#     m.errors["cy"] = 0.01
#     m.errors["t0"] = 0.1

#     if IS_ABSORPTION_MODE:
#         m.limits["visible_length"] = (0.0, max_range)
#         m.limits["full_range"] = (1.0, max_range)
#         m.errors["visible_length"] = 60.0
#         m.errors["full_range"] = 100.0
#     else:
#         m.limits["length"] = (0.0, max_range)
#         m.errors["length"] = 60.0

#     if not USE_TIMING_LIKELIHOOD:
#         m.fixed["t0"] = True

#     m.errordef = Minuit.LIKELIHOOD
#     m.strategy = M_STRAT

#     return m


# def is_bad_minuit_result(m, *, edm_max=1e10):
#     if (m.fval is None) or (not np.isfinite(m.fval)):
#         return True
#     # Do not use m.valid as a bad-result criterion.
#     try:
#         if (m.fmin is not None) and np.isfinite(m.fmin.edm) and (m.fmin.edm > edm_max):
#             return True
#     except Exception:
#         pass
#     return False


# def run_minuit_attempt(m, ncall):
#     if not ENABLE_STAGE2_MIGRAD_FIRST:
#         m.strategy = M_STRAT
#         m.simplex(ncall=ncall)
#         m.migrad(ncall=ncall)
#         return m

#     ncall_fast = max(2000, int(0.35 * ncall))
#     ncall_simplex = max(2000, int(0.25 * ncall))

#     m.strategy = 0
#     m.migrad(ncall=ncall_fast)

#     if is_bad_minuit_result(m):
#         m.simplex(ncall=ncall_simplex)
#         m.strategy = M_STRAT
#         m.migrad(ncall=ncall)

#     return m


# # =============================================================================
# # ADAPTIVE RESCUE
# # =============================================================================
# ENABLE_ADAPTIVE_RESCUE = ENABLE_STAGE3_ADAPTIVE_RESCUE
# RESCUE_MAX_SEEDS = 6
# RESCUE_LENGTH_BINS = [
#     (0.0, 1000.0),
#     (1000.0, 1250.0),
#     (1250.0, 1400.0),
#     (1400.0, 1700.0),
#     (1700.0, 3000.0),
# ]


# def result_length_value(values):
#     return float(values["visible_length"] if IS_ABSORPTION_MODE else values["length"])


# def result_full_range_value(values):
#     return float(values["full_range"] if IS_ABSORPTION_MODE else values["length"])


# def seed_length_value(params):
#     return float(params["visible_length"] if IS_ABSORPTION_MODE else params["length"])


# def result_ke0_from_values(values):
#     try:
#         return float(RANGE_LOOKUP.range_mm_to_energy(result_full_range_value(values)))
#     except Exception:
#         return np.nan


# def needs_rescue_result(result):
#     if result is None:
#         return True
#     if not np.isfinite(result.get("fval", np.inf)):
#         return True
#     values = result.get("values", {})
#     try:
#         fitted_length = result_length_value(values)
#         fitted_full = result_full_range_value(values)
#     except Exception:
#         return True
#     if (not np.isfinite(fitted_length)) or (not np.isfinite(fitted_full)):
#         return True
#     if fitted_length <= 10.0 or fitted_length >= VISIBLE_LENGTH_RETRY_THRESHOLD:
#         return True
#     if IS_ABSORPTION_MODE and fitted_length > fitted_full:
#         return True
#     if result.get("seed_stuck", False):
#         return True
#     if USE_TIMING_LIKELIHOOD and result.get("below_t_min", False):
#         return True
#     return False


# def needs_fcn_retry_result(result, fcn_threshold):
#     """Return True when the default, non-adaptive FCN retry should run.

#     This intentionally does not use m.valid.  A retry is triggered only by a
#     non-finite FCN or by FCN exceeding the user-configured threshold.
#     """
#     if result is None:
#         return True
#     fval = float(result.get("fval", np.inf))
#     if not np.isfinite(fval):
#         return True
#     if fcn_threshold is None:
#         return False
#     try:
#         threshold = float(fcn_threshold)
#     except Exception:
#         return False
#     return np.isfinite(threshold) and (fval > threshold)


# def next_untried_seed_info(seed_scan_sorted, tried_seed_indices):
#     """Return the best remaining seed from the prescan, or None if exhausted."""
#     for seed_info in seed_scan_sorted:
#         idx = int(seed_info["seed_index"])
#         if idx not in tried_seed_indices:
#             return seed_info
#     return None


# def choose_diverse_rescue_seed_infos(seed_scan_sorted, already_tried_seed_indices=None, max_total=RESCUE_MAX_SEEDS):
#     already = set() if already_tried_seed_indices is None else set(already_tried_seed_indices)
#     chosen = []

#     for lo, hi in RESCUE_LENGTH_BINS:
#         candidates = [
#             s for s in seed_scan_sorted
#             if int(s["seed_index"]) not in already
#             and lo <= seed_length_value(s["params"]) < hi
#         ]
#         if candidates:
#             chosen.append(candidates[0])
#             already.add(int(candidates[0]["seed_index"]))
#         if len(chosen) >= max_total:
#             return chosen

#     for s in seed_scan_sorted:
#         idx = int(s["seed_index"])
#         if idx in already:
#             continue
#         chosen.append(s)
#         already.add(idx)
#         if len(chosen) >= max_total:
#             break

#     return chosen


# def compact_seed_scan(seed_scan_sorted):
#     """Return the configured seed-scan payload for output/debugging."""
#     if SAVE_SEED_SCAN:
#         return seed_scan_sorted
#     if SAVE_TOP_N_SEEDS > 0:
#         return seed_scan_sorted[:SAVE_TOP_N_SEEDS]
#     return []


# def build_result_from_minuit(m, attempt, start_params, chosen_seed_idx, chosen_seed_fcn, seed_scan_sorted):
#     current_fval = float(m.fval) if (m.fval is not None and np.isfinite(m.fval)) else np.inf
#     current_values = m.values.to_dict()

#     fitted_z0 = float(current_values["z0"])
#     fitted_length = result_length_value(current_values)
#     fitted_full = result_full_range_value(current_values)
#     fitted_ke0 = result_ke0_from_values(current_values)

#     visible_too_large = fitted_length > VISIBLE_LENGTH_RETRY_THRESHOLD
#     z_near_seed = abs(fitted_z0 - float(start_params["z0"])) <= Z_SEED_EPS
#     length_near_seed = abs(fitted_length - seed_length_value(start_params)) <= VISIBLE_LENGTH_SEED_EPS
#     if IS_ABSORPTION_MODE:
#         full_near_seed = abs(fitted_full - float(start_params["full_range"])) <= FULL_RANGE_SEED_EPS
#     else:
#         full_near_seed = False
#     seed_stuck = z_near_seed and length_near_seed and (full_near_seed if IS_ABSORPTION_MODE else True)
#     below_t_min = USE_TIMING_LIKELIHOOD and (current_values["t0"] < T_MIN)

#     # Always expose consistent aliases in the result dictionary.
#     visible_length_mm = fitted_length
#     full_range_mm = fitted_full
#     current_values.setdefault("length", fitted_length)
#     current_values.setdefault("visible_length", visible_length_mm)
#     current_values.setdefault("full_range", full_range_mm)

#     return {
#         "values": current_values,
#         "errors": m.errors.to_dict(),
#         "fval": current_fval,
#         "valid": bool(m.valid),
#         "attempts": attempt,
#         "visible_length_too_large": bool(visible_too_large),
#         "length_too_large": bool(visible_too_large),
#         "seed_stuck": bool(seed_stuck),
#         "z_near_seed": bool(z_near_seed),
#         "visible_length_near_seed": bool(length_near_seed),
#         "full_range_near_seed": bool(full_near_seed),
#         "length_near_seed": bool(length_near_seed),
#         "below_t_min": bool(below_t_min),
#         "chosen_seed_index": int(chosen_seed_idx),
#         "chosen_seed_fcn": float(chosen_seed_fcn) if np.isfinite(chosen_seed_fcn) else np.nan,
#         "chosen_seed_params": dict(start_params),
#         "seed_scan": compact_seed_scan(seed_scan_sorted),
#         "visible_length_mm": visible_length_mm,
#         "full_range_mm": full_range_mm,
#         "length_mm": fitted_length,
#         "ke0_mev": fitted_ke0,
#         "edm": (
#             float(m.fmin.edm)
#             if (getattr(m, "fmin", None) is not None and m.fmin.edm is not None)
#             else np.nan
#         ),
#     }


# def result_sort_key(result):
#     if result is None:
#         return (999, np.inf)
#     fval = float(result.get("fval", np.inf))
#     penalty = 0
#     if not np.isfinite(fval):
#         penalty += 100
#     if result.get("visible_length_too_large", False) or result.get("length_too_large", False):
#         penalty += 10
#     if result.get("seed_stuck", False):
#         penalty += 5
#     if result.get("below_t_min", False):
#         penalty += 5
#     return (penalty, fval)


# # =============================================================================
# # HARD-EVENT VISIBLE-LENGTH PROFILE RESCUE
# # =============================================================================
# ENABLE_LENGTH_PROFILE_RESCUE = ENABLE_STAGE4_LENGTH_PROFILE
# LENGTH_PROFILE_GRID = list(FAST_SEED_VISIBLE_LENGTHS)
# LENGTH_PROFILE_MAX_POINTS = 6


# def run_length_profile_rescue(obs_pes, obs_ts, mpmt_types, seed_scan_sorted, ncall, starting_attempt_index=100):
#     profile_results = []
#     base_seed = dict(seed_scan_sorted[0]["params"])
#     length_key = "visible_length" if IS_ABSORPTION_MODE else "length"

#     for j, profile_length in enumerate(LENGTH_PROFILE_GRID[:LENGTH_PROFILE_MAX_POINTS]):
#         start_params = dict(base_seed)
#         start_params[length_key] = float(profile_length)
#         if IS_ABSORPTION_MODE and start_params["visible_length"] > start_params["full_range"]:
#             continue

#         m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
#         m.fixed[length_key] = True
#         run_minuit_attempt(m, max(5000, int(0.5 * ncall)))

#         prof_result = build_result_from_minuit(
#             m,
#             attempt=starting_attempt_index + j,
#             start_params=start_params,
#             chosen_seed_idx=-1000 - j,
#             chosen_seed_fcn=np.nan,
#             seed_scan_sorted=seed_scan_sorted,
#         )
#         prof_result["profile_fixed_length"] = float(profile_length)
#         prof_result["profile_fixed_visible_length"] = float(profile_length)
#         profile_results.append(prof_result)

#     if not profile_results:
#         return None

#     best_profile = min(profile_results, key=result_sort_key)

#     polish_params = dict(best_profile["values"])
#     # Keep only parameters actually used by this mode; Minuit will reject extras.
#     polish_params = {k: polish_params[k] for k in PARAM_NAMES if k in polish_params}
#     m = make_minuit_for_event(obs_pes, obs_ts, polish_params, mpmt_types)
#     m.fixed[length_key] = False
#     run_minuit_attempt(m, ncall)

#     polish_result = build_result_from_minuit(
#         m,
#         attempt=starting_attempt_index + len(profile_results),
#         start_params=polish_params,
#         chosen_seed_idx=-2000,
#         chosen_seed_fcn=float(best_profile["fval"]),
#         seed_scan_sorted=seed_scan_sorted,
#     )
#     polish_result["length_profile_rescue_used"] = True
#     polish_result["length_profile_results"] = profile_results
#     polish_result["length_profile_best_fixed"] = best_profile
#     return polish_result


# def fit_one_event_by_index(args):
#     event_index, init_param_sets, fcn_threshold, max_attempts, ncall = args

#     obs_pes = OBS_PES_ALL[event_index]
#     obs_ts = OBS_TS_ALL[event_index]
#     mpmt_types = MPMT_TYPE_CODES_ALL[event_index]

#     best_seed, best_seed_idx, best_seed_fval, seed_scan_sorted = select_best_initial_seed(
#         obs_pes,
#         obs_ts,
#         init_param_sets,
#         mpmt_types,
#     )

#     attempt_results = []
#     tried_seed_indices = set()

#     primary_info = seed_scan_sorted[0]
#     start_params = dict(primary_info["params"])
#     chosen_seed_idx = int(primary_info["seed_index"])
#     chosen_seed_fcn = float(primary_info["fval"])
#     tried_seed_indices.add(chosen_seed_idx)

#     m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
#     run_minuit_attempt(m, ncall)
#     primary_result = build_result_from_minuit(
#         m,
#         attempt=1,
#         start_params=start_params,
#         chosen_seed_idx=chosen_seed_idx,
#         chosen_seed_fcn=chosen_seed_fcn,
#         seed_scan_sorted=seed_scan_sorted,
#     )
#     attempt_results.append(primary_result)

#     # ------------------------------------------------------------------
#     # Default FCN retry path.  This is intentionally independent of the
#     # adaptive-rescue stage: if the best result so far has FCN above the
#     # configured threshold, try the next-best prescan seeds up to
#     # MAX_FIT_ATTEMPTS.  Do not use m.valid as a bad-result criterion.
#     # ------------------------------------------------------------------
#     while len(attempt_results) < max(1, int(max_attempts)):
#         best_so_far = min(attempt_results, key=result_sort_key)
#         if not needs_fcn_retry_result(best_so_far, fcn_threshold):
#             break

#         seed_info = next_untried_seed_info(seed_scan_sorted, tried_seed_indices)
#         if seed_info is None:
#             break

#         start_params = dict(seed_info["params"])
#         chosen_seed_idx = int(seed_info["seed_index"])
#         chosen_seed_fcn = float(seed_info["fval"])
#         tried_seed_indices.add(chosen_seed_idx)

#         m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
#         run_minuit_attempt(m, ncall)
#         result = build_result_from_minuit(
#             m,
#             attempt=len(attempt_results) + 1,
#             start_params=start_params,
#             chosen_seed_idx=chosen_seed_idx,
#             chosen_seed_fcn=chosen_seed_fcn,
#             seed_scan_sorted=seed_scan_sorted,
#         )
#         result["fcn_retry_used"] = True
#         attempt_results.append(result)

#     best_result = min(attempt_results, key=result_sort_key)

#     if ENABLE_ADAPTIVE_RESCUE and needs_rescue_result(best_result):
#         rescue_seed_infos = choose_diverse_rescue_seed_infos(
#             seed_scan_sorted,
#             already_tried_seed_indices=tried_seed_indices,
#             max_total=RESCUE_MAX_SEEDS,
#         )

#         for seed_info in rescue_seed_infos:
#             start_params = dict(seed_info["params"])
#             chosen_seed_idx = int(seed_info["seed_index"])
#             chosen_seed_fcn = float(seed_info["fval"])
#             tried_seed_indices.add(chosen_seed_idx)

#             m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
#             run_minuit_attempt(m, ncall)
#             result = build_result_from_minuit(
#                 m,
#                 attempt=len(attempt_results) + 1,
#                 start_params=start_params,
#                 chosen_seed_idx=chosen_seed_idx,
#                 chosen_seed_fcn=chosen_seed_fcn,
#                 seed_scan_sorted=seed_scan_sorted,
#             )
#             result["adaptive_rescue_attempt"] = True
#             attempt_results.append(result)

#     best_result = min(attempt_results, key=result_sort_key)

#     if ENABLE_LENGTH_PROFILE_RESCUE and needs_rescue_result(best_result):
#         profile_result = run_length_profile_rescue(
#             obs_pes,
#             obs_ts,
#             mpmt_types,
#             seed_scan_sorted,
#             ncall,
#             starting_attempt_index=100 + len(attempt_results),
#         )
#         if profile_result is not None:
#             attempt_results.append(profile_result)
#             best_result = min(attempt_results, key=result_sort_key)

#     best_result["attempts"] = len(attempt_results)
#     if SAVE_ATTEMPT_RESULTS:
#         best_result["attempt_results"] = attempt_results
#     else:
#         best_result["attempt_results"] = []
#     best_result["adaptive_rescue_used"] = bool(len(attempt_results) > 1)
#     best_result["length_profile_rescue_considered"] = bool(ENABLE_LENGTH_PROFILE_RESCUE)
#     best_result["length_profile_rescue_used"] = bool(
#         best_result.get("length_profile_rescue_used", False)
#         or any(r.get("length_profile_rescue_used", False) for r in attempt_results)
#     )
#     return best_result


# def run_batch(event_indices, init_param_sets, nproc, fcn_threshold, max_attempts, ncall):
#     args = [(idx, init_param_sets, fcn_threshold, max_attempts, ncall) for idx in event_indices]

#     try:
#         ctx = mp.get_context("fork")
#     except ValueError:
#         ctx = mp.get_context()

#     with ctx.Pool(processes=nproc) as pool:
#         return pool.map(fit_one_event_by_index, args)


# # =============================================================================
# # USER-PROVIDED EVENT FILE HELPERS
# # =============================================================================
# def _coerce_event_array(event, *, event_label="event"):
#     arr = np.asarray(event)
#     if arr.ndim != 2 or arr.shape[1] < 3:
#         raise ValueError(
#             f"{event_label} must be a 2D array with at least 3 columns: "
#             "[pmt_id, charge, time]."
#         )
#     # Keep optional event-number columns, but the fitter only consumes columns 0:3.
#     return np.asarray(arr[:, :4] if arr.shape[1] >= 4 else arr[:, :3], dtype=np.float64)


# def _events_from_loaded_object(obj):
#     """Normalize npy/npz/pickle payloads into a list of event arrays."""
#     if isinstance(obj, dict):
#         if USER_EVENT_KEY is not None:
#             obj = obj[USER_EVENT_KEY]
#         elif "events" in obj:
#             obj = obj["events"]
#         elif "data" in obj:
#             obj = obj["data"]
#         elif "arr_0" in obj:
#             obj = obj["arr_0"]
#         else:
#             keys = ", ".join(map(str, obj.keys()))
#             raise KeyError(
#                 "Could not choose an event array from the dict payload. "
#                 f"Available keys: {keys}. Set USER_EVENT_KEY."
#             )

#     if isinstance(obj, np.lib.npyio.NpzFile):
#         if USER_EVENT_KEY is not None:
#             key = USER_EVENT_KEY
#         elif "events" in obj.files:
#             key = "events"
#         elif "data" in obj.files:
#             key = "data"
#         elif "arr_0" in obj.files:
#             key = "arr_0"
#         elif len(obj.files) == 1:
#             key = obj.files[0]
#         else:
#             raise KeyError(
#                 "Could not choose an event array from the npz payload. "
#                 f"Available keys: {obj.files}. Set USER_EVENT_KEY."
#             )
#         obj = obj[key]

#     if isinstance(obj, (list, tuple)):
#         return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(obj)]

#     arr = np.asarray(obj, dtype=object if getattr(obj, "dtype", None) == object else None)

#     # Object arrays are normally lists of variable-length events.
#     if arr.dtype == object and arr.ndim == 1:
#         return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(arr)]

#     # A 3D numeric array is N_events x N_hits x N_columns.
#     if arr.ndim == 3:
#         return [_coerce_event_array(arr[i], event_label=f"event[{i}]") for i in range(arr.shape[0])]

#     # A 2D array with a 4th column is interpreted as a concatenated event table
#     # grouped by event number.  A 2D array with only 3 columns is one event.
#     if arr.ndim == 2:
#         arr2 = np.asarray(arr, dtype=np.float64)
#         if arr2.shape[1] >= 4:
#             events = []
#             event_numbers = arr2[:, 3].astype(np.int64)
#             for evnum in np.unique(event_numbers):
#                 events.append(_coerce_event_array(arr2[event_numbers == evnum], event_label=f"event_number={evnum}"))
#             return events
#         return [_coerce_event_array(arr2, event_label="single_event")]

#     raise ValueError(
#         "Unsupported USER_EVENT_FILE payload. Expected a list/object array of "
#         "event arrays, a 3D event array, or a 2D [pmt_id, charge, time] table."
#     )


# def load_user_event_file(path, *, max_events=None):
#     """Load user-provided, already-selected events from npy/npz/pickle files."""
#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(f"USER_EVENT_FILE does not exist: {path}")

#     suffix = path.suffix.lower()
#     if suffix == ".npz":
#         loaded = np.load(path, allow_pickle=True)
#     elif suffix == ".npy":
#         loaded = np.load(path, allow_pickle=True)
#     elif suffix in {".pkl", ".pickle"}:
#         with open(path, "rb") as f:
#             loaded = pickle.load(f)
#     else:
#         raise ValueError(
#             f"Unsupported USER_EVENT_FILE suffix {suffix!r}. Use .npy, .npz, .pkl, or .pickle."
#         )

#     events = _events_from_loaded_object(loaded)
#     if max_events is not None:
#         events = events[: int(max_events)]
#     print(f"Loaded {len(events)} user-provided events from: {path}")
#     return events


# # =============================================================================
# # MAIN DRIVER
# # =============================================================================
# def main():
#     global OVERALL_DISTANCES, INIT_ENERGY_TABLE, RANGE_LOOKUP
#     global WCD, PMT_MODEL, EMITTER_TEMPLATE, P_LOCATIONS, DIRECTION_ZS, RING_KEEP_MASK, CORR_POS
#     global OBS_PES_ALL, OBS_TS_ALL, MPMT_SLOTS_ALL, MPMT_TYPE_CODES_ALL, GOOD_WCTE_PMTS_SET

#     print("Likelihood mode:", LIKELIHOOD_MODE)
#     print("Fit particle:", FIT_PARTICLE_CANONICAL)
#     print("Particle mass [MeV]:", FIT_PARTICLE_MASS_MEV)
#     print("Cherenkov threshold [MeV]:", FIT_PARTICLE_THRESHOLD_MEV)
#     print("Fit mode:", TRACK_END_MODE)
#     print("Fit parameters:", FIT_PARAMETER_NAMES)
#     print("Output file:", OUTPUT_FILE)

#     if EVENT_SOURCE == "selection" and get_selected_events is None:
#         raise ImportError(
#             "event_loader.py was not found. Copy it into LF_multiParticles/scripts "
#             "or add its directory to PYTHONPATH, or set EVENT_SOURCE=file and USER_EVENT_FILE."
#         )

#     if EVENT_SOURCE == "file" and not USER_EVENT_FILE:
#         raise ValueError("EVENT_SOURCE=file requires USER_EVENT_FILE=/path/to/events.npy|npz|pkl")

#     GOOD_WCTE_PMTS_SET = load_good_wcte_pmts()

#     RANGE_LOOKUP = ParticleRangeLookup(FIT_PARTICLE_CANONICAL, table_dirs=[str(TABLE_DIR)])
#     print("Range table max KE [MeV]:", float(RANGE_LOOKUP.initial_energies_mev[-1]))
#     print("Range table max full_range [mm]:", float(RANGE_LOOKUP.overall_distances_mm[-1]))

#     configure_truth_params()
#     if IS_ABSORPTION_MODE:
#         truth_ready = np.isfinite(TRUE_PARAMS["visible_length"]) and np.isfinite(TRUE_PARAMS["full_range"])
#         if truth_ready:
#             print("Truth visible length [mm]:", TRUE_PARAMS["visible_length"])
#             print("Truth full range [mm]:", TRUE_PARAMS["full_range"])
#             print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["full_range"]))
#         else:
#             print("Truth FCN disabled: set TRUE_VISIBLE_LENGTH_MM and TRUE_FULL_RANGE_MM/TRUE_INITIAL_KE_MEV.")
#     else:
#         truth_ready = np.isfinite(TRUE_PARAMS["length"])
#         if truth_ready:
#             print("Truth length [mm]:", TRUE_PARAMS["length"])
#             print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["length"]))
#         else:
#             print("Truth FCN disabled: set TRUE_LENGTH_MM or TRUE_INITIAL_KE_MEV.")

#     init_param_sets = build_fast_seed_grid(RANGE_LOOKUP)
#     if not init_param_sets:
#         raise RuntimeError("Seed grid is empty. Check FAST_SEED_VISIBLE_LENGTHS and FAST_SEED_KE0_MEV/FULL_RANGES.")
#     print("Number of initial seeds:", len(init_param_sets))

#     for i, seed in enumerate(init_param_sets):
#         missing = [k for k in PARAM_NAMES if k not in seed]
#         if missing:
#             raise ValueError(f"Seed {i} is missing keys: {missing}")

#     set_active_particle(FIT_PARTICLE_CANONICAL)
#     OVERALL_DISTANCES, INIT_ENERGY_TABLE, _distance_rows = get_energy_distance_tables(
#         FIT_PARTICLE_CANONICAL
#     )

#     hall = Device.open_file(GEOMETRY_FILE)
#     WCD = hall.wcds[0]

#     initial_ke_seed = float(RANGE_LOOKUP.range_mm_to_energy(
#         min(1000.0, float(RANGE_LOOKUP.overall_distances_mm[-1]))
#     ))

#     emitter_model = Emitter(
#         0.0,
#         (0.0, 0.0, 0.0),
#         (0.0, 0.0, 1.0),
#         0.96,
#         500.0,
#         18.0,
#         particle=FIT_PARTICLE_CANONICAL,
#         track_end_mode=EMITTER_TRACK_END_MODE,
#         fixed_initial_KE=initial_ke_seed if IS_ABSORPTION_MODE else None,
#     )

#     delta_pdf_path = TABLE_DIR / "delta_e_angular_pdf_table.npz"
#     if delta_pdf_path.exists() and hasattr(emitter_model, "load_delta_e_angular_pdf_table"):
#         emitter_model.load_delta_e_angular_pdf_table(str(delta_pdf_path))

#     PMT_MODEL = PMT(1.0,0.3, 1.0, 40.0, 0.2, 0.0)
#     EMITTER_TEMPLATE = emitter_model.copy()
#     CORR_POS = None

#     print("Building event observables...")

#     obs_pes_all = []
#     obs_ts_all = []
#     mpmt_slots_all = []

#     if EVENT_SOURCE == "selection":
#         print("N_EVENTS AHHHHH", N_EVENTS)
#         events = get_selected_events(
#             RUN,
#             N_EVENTS,
#             particle=PARTICLE_SELECTION_LABEL,
#             root_file=CONFIG_ROOT_FILE,
#             use_peak_time_cut=USE_PEAK_TIME_CUT,
#             peak_window=PEAK_WINDOW_NS,
#             peak_bin_width=PEAK_BIN_WIDTH_NS,
#             tof_primary=SELECTION_TOF_NS,
#             tof_window=SELECTION_TOF_WINDOW_NS,
#             tof_scalar_field=SELECTION_TOF_FIELD,
#             momentum_scalar_field=SELECTION_MOMENTUM_FIELD,
#             t5_particle_nr=SELECTION_T5_PARTICLE_NR,
#         )
#     else:
#         events = load_user_event_file(USER_EVENT_FILE, max_events=N_EVENTS)

#     tot_events = len(events)
#     print("Total Events to Fit:", tot_events)

#     for i in range(tot_events):
#         event = np.asarray(events[i])
#         if event.size == 0:
#             continue

#         apply_event_time_window = (EVENT_SOURCE == "selection") or USER_EVENT_APPLY_PEAK_WINDOW
#         if apply_event_time_window:
#             time_hist = np.histogram(event[:, 2], bins=np.arange(0, 4000))
#             max_idx = int(np.argmax(time_hist[0]))
#             lo_idx = max(0, max_idx - 20)
#             hi_idx = min(len(time_hist[1]) - 1, max_idx + 5)
#             min_time = time_hist[1][lo_idx]
#             cut_time = time_hist[1][hi_idx]
#             time_mask = (event[:, 2] > min_time) & (event[:, 2] < cut_time)
#             event = event[time_mask]

#         ev, pmt_ids = sim_to_event(event, WCD, n_mpmt_total=106, pe_scale=143)

#         if P_LOCATIONS is None or DIRECTION_ZS is None:
#             P_LOCATIONS, DIRECTION_ZS, MPMT_SLOTS = EMITTER_TEMPLATE.get_pmt_placements(ev, WCD, "est")
#             MPMT_SLOTS = np.asarray(MPMT_SLOTS, dtype=int)
#             RING_KEEP_MASK = np.isin(MPMT_SLOTS, ALL_RING)

#         obs_pes, obs_ts = build_observables_from_event(ev, pe_scale=143)
#         obs_pes, obs_ts = apply_ring_mask_to_observables(
#             obs_pes,
#             obs_ts,
#             RING_KEEP_MASK,
#             mode=RING_MASK_MODE,
#         )

#         obs_pes_all.append(obs_pes)
#         obs_ts_all.append(obs_ts)
#         mpmt_slots_all.append(MPMT_SLOTS)

#     OBS_PES_ALL = obs_pes_all
#     OBS_TS_ALL = obs_ts_all
#     MPMT_SLOTS_ALL = mpmt_slots_all
#     MPMT_TYPE_CODES_ALL = [get_mpmt_slot_type_codes(slots) for slots in MPMT_SLOTS_ALL]
#     tot_events = len(OBS_PES_ALL)

#     print("Computing truth FCNs...")
#     true_fcn_all = [compute_true_fcn_for_event(i) for i in range(tot_events)]

#     est_dict = {
#         "metadata": {
#             "fit_particle": FIT_PARTICLE_CANONICAL,
#             "particle_mass_mev": FIT_PARTICLE_MASS_MEV,
#             "particle_threshold_mev": FIT_PARTICLE_THRESHOLD_MEV,
#             "beam_p": BEAM_P,
#             "track_end_mode": TRACK_END_MODE,
#             "fit_parameters": list(FIT_PARAMETER_NAMES),
#             "truth_params": dict(TRUE_PARAMS),
#             "geometry_file": GEOMETRY_FILE,
#             "config_root_file": CONFIG_ROOT_FILE,
#             "event_source": EVENT_SOURCE,
#             "user_event_file": USER_EVENT_FILE if EVENT_SOURCE == "file" else None,
#             "user_event_key": USER_EVENT_KEY if EVENT_SOURCE == "file" else None,
#             "user_event_apply_peak_window": bool(USER_EVENT_APPLY_PEAK_WINDOW) if EVENT_SOURCE == "file" else None,
#             "particle_selection_label": PARTICLE_SELECTION_LABEL,
#             "selection_tof_ns": SELECTION_TOF_NS,
#             "selection_tof_window_ns": SELECTION_TOF_WINDOW_NS,
#             "selection_tof_field": SELECTION_TOF_FIELD,
#             "selection_t5_particle_nr": SELECTION_T5_PARTICLE_NR,
#             "range_table_max_full_range_mm": float(RANGE_LOOKUP.overall_distances_mm[-1]),
#             "save_seed_scan": bool(SAVE_SEED_SCAN),
#             "save_top_n_seeds": int(SAVE_TOP_N_SEEDS),
#             "save_attempt_results": bool(SAVE_ATTEMPT_RESULTS),
#         },
#         "minimum_found": [],
#         "x": [],
#         "y": [],
#         "z": [],
#         "visible_length": [],
#         "full_range": [],
#         "ke0": [],
#         "length": [],  # legacy alias for visible_length
#         "t": [],
#         "est_fcn": [],
#         "true_fcn": [],
#         "cx": [],
#         "cy": [],
#         "n_attempts": [],
#         "chosen_seed_idx": [],
#         "chosen_seed_fcn": [],
#         "chosen_seed_params": [],
#         "adaptive_rescue_used": [],
#         "length_profile_rescue_considered": [],
#         "length_profile_rescue_used": [],
#         "fcn_retry_used": [],
#         "edm": [],
#     }
#     if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
#         est_dict["seed_scan"] = []

#     if SAVE_ATTEMPT_RESULTS:
#         est_dict["attempt_results"] = []

#     n_events_per_batch = min(N_EVENTS_PER_BATCH, max(1, tot_events))

#     for batch_start in range(0, tot_events, n_events_per_batch):
#         batch_end = min(batch_start + n_events_per_batch, tot_events)
#         event_indices = list(range(batch_start, batch_end))

#         print(f"Starting event number {batch_start}")

#         results = run_batch(
#             event_indices=event_indices,
#             init_param_sets=init_param_sets,
#             nproc=NPROC,
#             fcn_threshold=FCN_RETRY_THRESHOLD,
#             max_attempts=MAX_FIT_ATTEMPTS,
#             ncall=NCALL_MIGRAD,
#         )

#         for local_i, result in enumerate(results):
#             event_index = event_indices[local_i]
#             vals = result["values"]

#             if IS_ABSORPTION_MODE:
#                 visible_length = float(vals["visible_length"])
#                 full_range = float(vals["full_range"])
#             else:
#                 visible_length = float(vals["length"])
#                 full_range = visible_length
#             ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))

#             est_dict["minimum_found"].append(int(result["valid"]))
#             est_dict["x"].append(vals["x0"])
#             est_dict["y"].append(vals["y0"])
#             est_dict["z"].append(vals["z0"])
#             est_dict["visible_length"].append(visible_length)
#             est_dict["full_range"].append(full_range)
#             est_dict["ke0"].append(ke0)
#             est_dict["length"].append(visible_length)
#             est_dict["t"].append(vals["t0"])
#             est_dict["cx"].append(vals["cx"])
#             est_dict["cy"].append(vals["cy"])
#             est_dict["est_fcn"].append(result["fval"])
#             est_dict["true_fcn"].append(true_fcn_all[event_index])
#             est_dict["n_attempts"].append(result["attempts"])
#             est_dict["chosen_seed_idx"].append(result["chosen_seed_index"])
#             est_dict["chosen_seed_fcn"].append(result["chosen_seed_fcn"])
#             est_dict["chosen_seed_params"].append(result["chosen_seed_params"])
#             if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
#                 est_dict["seed_scan"].append(result.get("seed_scan", []))
#             est_dict["adaptive_rescue_used"].append(result.get("adaptive_rescue_used", False))
#             est_dict["length_profile_rescue_considered"].append(result.get("length_profile_rescue_considered", False))
#             est_dict["length_profile_rescue_used"].append(result.get("length_profile_rescue_used", False))
#             est_dict["fcn_retry_used"].append(bool(result.get("fcn_retry_used", False) or result.get("attempts", 1) > 1))
#             est_dict["edm"].append(result.get("edm", np.nan))
#             if SAVE_ATTEMPT_RESULTS:
#                 est_dict["attempt_results"].append(result.get("attempt_results", []))

#     Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
#     with open(OUTPUT_FILE, "wb") as f:
#         pickle.dump(est_dict, f)

#     print("Done.")
#     print("Saved:", OUTPUT_FILE)


# if __name__ == "__main__":
#     main()




# # # 7,8-parameter abrupt-endpoint batch driver.
# # #
# # # Fit parameters:
# # #   x0, y0, z0, cx, cy, visible_length, full_range, t0
# # #
# # # Meaning:
# # #   visible_length = actual visible primary-Cherenkov length before abrupt cutoff [mm]
# # #   full_range     = dE/dx-only range to Cherenkov threshold if no abrupt interaction [mm]
# # #   ke0            = inferred initial kinetic energy from full_range using particle range table [MeV]
# # #
# # # This driver assumes an abrupt endpoint model.  It does not use fixed_initial_KE
# # # as a fit setting; fixed_initial_KE is overwritten inside every FCN call using
# # # full_range -> ke0.

# # """Toggleable multi-stage batch driver for the 8-parameter Minuit Cherenkov fit on WCTE/real-data-style events.

# # This is the selected-event/input-array version of the driver.  It preserves:
# #   - get_selected_events(RUN, N_EVENTS) event loading
# #   - run configuration GOOD_WCTE_PMTS masking from the ROOT Configuration tree
# #   - pe_scale=143
# #   - estimated geometry placement "est"
# #   - mPMT type/relative efficiency corrections when tables are available

# # The 8 fitted parameters are:
# #   x0, y0, z0, cx, cy, visible_length, full_range, t0
# # """

# # import os
# # import sys
# # import pickle
# # import multiprocessing as mp
# # from pathlib import Path

# # import numpy as np
# # import uproot
# # from iminuit import Minuit

# # # =============================================================================
# # # SELF-CONTAINED PATH SETUP
# # # =============================================================================
# # SCRIPT_DIR = Path(__file__).resolve().parent
# # PROJECT_ROOT = SCRIPT_DIR.parent
# # LICKETYFIT_DIR = PROJECT_ROOT / "LicketyFit"
# # TABLE_DIR = PROJECT_ROOT / "tables"
# # OUTPUT_DIR = PROJECT_ROOT / "outputs"
# # OUTPUT_DIR.mkdir(exist_ok=True)

# # geometry_path = os.environ.get("GEOMETRY_PATH", "/eos/user/j/jrimmer/Geometry")
# # GEOMETRY_FILE = os.environ.get(
# #     "WCTE_GEOMETRY_FILE",
# #     str(Path(geometry_path) / "examples" / "wcte_bldg157.geo"),
# # )

# # for _path in (str(LICKETYFIT_DIR), str(PROJECT_ROOT), str(SCRIPT_DIR), str(TABLE_DIR), geometry_path):
# #     if _path not in sys.path:
# #         sys.path.insert(0, _path)

# # # Force local tables before importing lookup/collapse helpers.
# # os.environ["LF_TABLE_DIR"] = str(TABLE_DIR)
# # os.environ["LF_MULTIPARTICLES_TABLE_DIR"] = str(TABLE_DIR)

# # from Geometry.Device import Device
# # from LicketyFit.Event import Event
# # from LicketyFit.PMT import PMT
# # from LicketyFit.Emitter import Emitter
# # from particle_cherenkov_model import (
# #     get_energy_distance_tables,
# #     set_active_particle,
# #     canonical_particle_name,
# #     particle_mass_mev,
# #     cherenkov_threshold_kinetic_mev,
# # )
# # try:
# #     from event_loader import get_selected_events
# # except Exception:
# #     get_selected_events = None
# # from particle_range_lookup import ParticleRangeLookup


# # # =============================================================================
# # # ENV HELPERS
# # # =============================================================================
# # def _env_float(name, default=None):
# #     raw = os.environ.get(name)
# #     if raw is None or str(raw).strip() == "":
# #         return default
# #     return float(raw)


# # def _env_int(name, default):
# #     raw = os.environ.get(name)
# #     if raw is None or str(raw).strip() == "":
# #         return int(default)
# #     return int(raw)


# # def _env_bool(name, default=False):
# #     raw = os.environ.get(name)
# #     if raw is None:
# #         return bool(default)
# #     return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


# # def _parse_float_list_env(name, default):
# #     raw = os.environ.get(name)
# #     if raw is None or str(raw).strip() == "":
# #         return list(default)
# #     return [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]


# # # =============================================================================
# # # TOP-LEVEL CONFIGURATION
# # # =============================================================================
# # N_EVENTS_PER_BATCH = _env_int("N_EVENTS_PER_BATCH", 200)
# # NPROC = _env_int("NPROC", 16)
# # #M_STRAT = _env_int("M_STRAT", 1)
# # M_STRAT = _env_int("M_STRAT", 0)

# # Z_SEED_EPS = 20.0
# # VISIBLE_LENGTH_SEED_EPS = 40.0
# # FULL_RANGE_SEED_EPS = 80.0
# # T_MIN = -8.0

# # FCN_RETRY_THRESHOLD = 1100.0
# # VISIBLE_LENGTH_RETRY_THRESHOLD = _env_float("VISIBLE_LENGTH_RETRY_THRESHOLD", 2700.0)
# # MAX_FIT_ATTEMPTS = _env_int("MAX_FIT_ATTEMPTS", 4)
# # NCALL_MIGRAD = _env_int("NCALL_MIGRAD", 70000)
# # NCALL_SIMPLEX = _env_int("NCALL_SIMPLEX", NCALL_MIGRAD)

# # RUN = _env_int("RUN", 2079)
# # BEAM_P = _env_float("BEAM_P", 430)
# # N_EVENTS = _env_int("N_EVENTS", 30000)
# # print('N_EVENTS',N_EVENTS)

# # # =============================================================================
# # # PARTICLE HYPOTHESIS / 8-PARAMETER MODE
# # # =============================================================================
# # FIT_PARTICLE = os.environ.get("FIT_PARTICLE", "muon")
# # FIT_PARTICLE_CANONICAL = canonical_particle_name(FIT_PARTICLE)
# # FIT_PARTICLE_MASS_MEV = particle_mass_mev(FIT_PARTICLE_CANONICAL)
# # FIT_PARTICLE_THRESHOLD_MEV = cherenkov_threshold_kinetic_mev(FIT_PARTICLE_MASS_MEV)
# # set_active_particle(FIT_PARTICLE_CANONICAL)

# # # Fit mode:
# # #   full_length -> original 7-parameter fit:
# # #                  x0, y0, z0, cx, cy, length, t0
# # #                  length is the dE/dx range to Cherenkov threshold, so ke0 is inferred from length.
# # #   absorption  -> 8-parameter abrupt-endpoint fit:
# # #                  x0, y0, z0, cx, cy, visible_length, full_range, t0
# # #                  visible_length is the abrupt cutoff; full_range determines ke0.
# # _FIT_MODE_RAW = os.environ.get("FIT_MODE", os.environ.get("TRACK_END_MODE", "full_length")).strip().lower()

# # if _FIT_MODE_RAW in {"absorption", "absorbed", "abrupt", "abrupt_8param", "interaction", "truncated"}:
# #     FIT_MODE = "absorption"
# # elif _FIT_MODE_RAW in {"full_length", "full-length", "full", "threshold", "range", "csda", "old", "7param", "7_parameter"}:
# #     FIT_MODE = "full_length"
# # else:
# #     raise ValueError("FIT_MODE/TRACK_END_MODE must be 'full_length' or 'absorption'")

# # TRACK_END_MODE = FIT_MODE
# # IS_ABSORPTION_MODE = FIT_MODE == "absorption"
# # IS_FULL_LENGTH_MODE = FIT_MODE == "full_length"
# # EMITTER_TRACK_END_MODE = "abrupt" if IS_ABSORPTION_MODE else "threshold"
# # FIT_PARAMETER_NAMES = (
# #     ("x0", "y0", "z0", "cx", "cy", "visible_length", "full_range", "t0")
# #     if IS_ABSORPTION_MODE
# #     else ("x0", "y0", "z0", "cx", "cy", "length", "t0")
# # )

# # # Output/debug controls.  The full seed scan is large in the 8-parameter
# # # fit because every event can have thousands of seed dictionaries.  Keep these
# # # off for production output; enable only when debugging seed selection.
# # SAVE_ATTEMPT_RESULTS = _env_bool("SAVE_ATTEMPT_RESULTS", False)
# # SAVE_SEED_SCAN = _env_bool("SAVE_SEED_SCAN", False)
# # SAVE_TOP_N_SEEDS = _env_int("SAVE_TOP_N_SEEDS", 0)

# # # Likelihood toggles.
# # USE_CHARGE_LIKELIHOOD = _env_bool("USE_CHARGE_LIKELIHOOD", True)
# # USE_TIMING_LIKELIHOOD = _env_bool("USE_TIMING_LIKELIHOOD", True)
# # USE_T0_PRIOR = _env_bool("USE_T0_PRIOR", False)

# # if (not USE_CHARGE_LIKELIHOOD) and (not USE_TIMING_LIKELIHOOD):
# #     raise ValueError("At least one likelihood term must be enabled.")

# # if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
# #     LIKELIHOOD_MODE = "charge_time"
# # elif USE_CHARGE_LIKELIHOOD:
# #     LIKELIHOOD_MODE = "charge_only"
# # else:
# #     LIKELIHOOD_MODE = "timing_only"

# # OUTPUT_FILE = os.environ.get(
# #     "LF_OUTPUT_FILE",
# #     str(OUTPUT_DIR / f"estimates_run{RUN}_{BEAM_P:g}p_{FIT_PARTICLE_CANONICAL}_{TRACK_END_MODE}_mpmtEff_{LIKELIHOOD_MODE}.dict"),
# # )

# # RING_MASK_MODE = os.environ.get("RING_MASK_MODE", "both").strip().lower()

# # # Data configuration.
# # CONFIG_ROOT_FILE = os.environ.get(
# #     "CONFIG_ROOT_FILE",
# #     f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/{RUN}/WCTE_merged_production_R{RUN}.root",
# # )

# # # Event source.
# # #   "selection" / "internal" : use event_loader.get_selected_events(...) with particle-specific TOF cuts.
# # #   "file" / "user" / "custom" : load already-selected user events from USER_EVENT_FILE.
# # #
# # # USER_EVENT_FILE can be .npy, .npz, .pkl, or .pickle.  Supported payloads:
# # #   - list/object array of event arrays, each with columns [pmt_id, charge, time] or
# # #     [pmt_id, charge, time, event_number]
# # #   - a single 2D array [pmt_id, charge, time] for one event
# # #   - a single 2D array [pmt_id, charge, time, event_number], which is grouped by
# # #     event_number
# # EVENT_SOURCE = os.environ.get("EVENT_SOURCE", "selection").strip().lower()
# # if EVENT_SOURCE in {"selected", "internal", "event_loader", "auto"}:
# #     EVENT_SOURCE = "selection"
# # elif EVENT_SOURCE in {"file", "user", "custom", "user_file", "provided"}:
# #     EVENT_SOURCE = "file"
# # if EVENT_SOURCE not in {"selection", "file"}:
# #     raise ValueError("EVENT_SOURCE must be 'selection' or 'file'")

# # USER_EVENT_FILE = os.environ.get("USER_EVENT_FILE", "").strip()
# # USER_EVENT_KEY = os.environ.get("USER_EVENT_KEY", "").strip() or None
# # USER_EVENT_APPLY_PEAK_WINDOW = _env_bool("USER_EVENT_APPLY_PEAK_WINDOW", True)

# # # Event-selection configuration for event_loader.get_selected_events().
# # # Defaults reproduce the historical muon-like WCTE selection.  For non-muon
# # # beam selections, set PARTICLE_SELECTION_LABEL plus either SELECTION_TOF_NS
# # # or SELECTION_TOF_FIELD/T5_PARTICLE_NR as needed for your production ROOT file.
# # PARTICLE_SELECTION_LABEL = os.environ.get("PARTICLE_SELECTION_LABEL", FIT_PARTICLE_CANONICAL)
# # SELECTION_TOF_NS = _env_float("SELECTION_TOF_NS", None)
# # SELECTION_TOF_WINDOW_NS = _env_float("SELECTION_TOF_WINDOW_NS", 0.2)
# # SELECTION_TOF_FIELD = os.environ.get("SELECTION_TOF_FIELD", "") or None
# # SELECTION_MOMENTUM_FIELD = os.environ.get("SELECTION_MOMENTUM_FIELD", "") or None
# # SELECTION_T5_PARTICLE_NR = _env_int("SELECTION_T5_PARTICLE_NR", 1)
# # USE_PEAK_TIME_CUT = _env_bool("USE_PEAK_TIME_CUT", True)
# # PEAK_WINDOW_NS = _env_float("PEAK_WINDOW_NS", 100.0)
# # PEAK_BIN_WIDTH_NS = _env_float("PEAK_BIN_WIDTH_NS", 50.0)

# # # =============================================================================
# # # DETECTOR CONFIGURATION
# # # =============================================================================
# # DEFAULT_INACTIVE_SLOTS = [27, 32, 45, 74, 77, 79, 85, 91, 99, 9, 67]
# # INACTIVE_SLOTS = [int(x) for x in os.environ.get(
# #     "INACTIVE_SLOTS",
# #     ",".join(str(x) for x in DEFAULT_INACTIVE_SLOTS),
# # ).replace(";", ",").split(",") if x.strip()]
# # INACTIVE_SLOTS_SET = set(int(s) for s in INACTIVE_SLOTS)

# # OUTER_RING = np.array([0, 7, 19, 34, 50, 66, 82, 83, 105, 94, 95, 71, 72, 56, 40, 24, 11, 3, 18])
# # INNER_RING = np.array([1, 8, 35, 51, 67, 84, 69, 70, 55, 39, 23, 10, 2, 20, 36, 52, 68, 53, 54, 38, 22, 21, 37, 9])
# # OUTSIDE_RING = np.array([12, 13, 4, 5, 6, 17, 33, 49, 65, 81, 82, 104, 93, 86, 87, 72, 57, 41, 25])
# # ALL_RING = np.arange(0, 106)

# # # Optional truth diagnostics.  If not supplied, true_fcn is NaN.
# # TRUE_LENGTH_MM = _env_float("TRUE_LENGTH_MM", None)
# # TRUE_VISIBLE_LENGTH_MM = _env_float("TRUE_VISIBLE_LENGTH_MM", None)
# # TRUE_FULL_RANGE_MM = _env_float("TRUE_FULL_RANGE_MM", None)
# # TRUE_INITIAL_KE_MEV = _env_float("TRUE_INITIAL_KE_MEV", None)

# # TRUE_PARAMS = {
# #     "x0": _env_float("TRUE_X0", 0.0),
# #     "y0": _env_float("TRUE_Y0", 0.0),
# #     "z0": _env_float("TRUE_Z0", -1348.0),
# #     "cx": _env_float("TRUE_CX", 0.0),
# #     "cy": _env_float("TRUE_CY", 0.0),
# #     "visible_length": np.nan,
# #     "full_range": np.nan,
# #     "t0": _env_float("TRUE_T0", 0.0),
# # }

# # # =============================================================================
# # # GLOBAL FIT-SEARCH STAGE TOGGLES
# # # =============================================================================
# # ENABLE_STAGE1_SEED_GRID = _env_bool("ENABLE_STAGE1_SEED_GRID", True)
# # ENABLE_STAGE2_MIGRAD_FIRST = _env_bool("ENABLE_STAGE2_MIGRAD_FIRST", False)
# # ENABLE_STAGE3_ADAPTIVE_RESCUE = _env_bool("ENABLE_STAGE3_ADAPTIVE_RESCUE", False)
# # ENABLE_STAGE4_LENGTH_PROFILE = _env_bool("ENABLE_STAGE4_LENGTH_PROFILE", False)

# # # Keep the full seed ranking only when it is actually needed.  This avoids
# # # sorting and returning thousands of seed dictionaries for normal production fits.
# # NEED_FULL_SEED_SCAN = (
# #     ENABLE_STAGE3_ADAPTIVE_RESCUE
# #     or ENABLE_STAGE4_LENGTH_PROFILE
# #     or SAVE_SEED_SCAN
# #     or SAVE_TOP_N_SEEDS > 0
# # )

# # # =============================================================================
# # # INITIAL SEED CONFIGURATION
# # # =============================================================================
# # FAST_SEED_X0 = _parse_float_list_env("FAST_SEED_X0", [-50.0, 0.0, 50.0])
# # FAST_SEED_Y0 = _parse_float_list_env("FAST_SEED_Y0", [-50.0, 0.0, 50.0])
# # FAST_SEED_Z0 = _parse_float_list_env("FAST_SEED_Z0", [-1500.0, -1400.0, -1300.0, -1350, -1200.0, -1100.0, -1000.0])

# # FAST_SEED_VISIBLE_LENGTHS = _parse_float_list_env(
# #     "FAST_SEED_VISIBLE_LENGTHS",
# #     [100.0, 150, 200, 250, 300.0, 350, 400, 450, 500.0, 700.0, 900.0, 1100.0, 1300.0, 1400.0, 1500.0, 1700.0, 1900.0],
# # )

# # FAST_SEED_KE0_MEV = _parse_float_list_env(
# #     "FAST_SEED_KE0_MEV",
# #     [600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0],
# # )
# # FAST_SEED_FULL_RANGES_MM = _parse_float_list_env(
# #     "FAST_SEED_FULL_RANGES_MM",
# #     [300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0],
# # )
# # FAST_SEED_DIRECTIONS = [
# #     (0.0, 0.0),
# #     (0.04, 0.0),
# #     (-0.04, 0.0),
# #     (0.0, 0.04),
# #     (0.0, -0.04),
# # ]
# # FAST_SEED_FULL_CARTESIAN = _env_bool("FAST_SEED_FULL_CARTESIAN", False)


# # def build_sparse_geometry_variants():
# #     variants = []
# #     for x0 in FAST_SEED_X0:
# #         variants.append({"x0": float(x0), "y0": 0.0, "cx": 0.0, "cy": 0.0})
# #     for y0 in FAST_SEED_Y0:
# #         variants.append({"x0": 0.0, "y0": float(y0), "cx": 0.0, "cy": 0.0})
# #     for cx, cy in FAST_SEED_DIRECTIONS:
# #         variants.append({"x0": 0.0, "y0": 0.0, "cx": float(cx), "cy": float(cy)})

# #     unique = []
# #     seen = set()
# #     for v in variants:
# #         sig = (float(v["x0"]), float(v["y0"]), float(v["cx"]), float(v["cy"]))
# #         if sig not in seen:
# #             seen.add(sig)
# #             unique.append(v)
# #     return unique


# # FAST_SEED_GEOMETRY_VARIANTS = build_sparse_geometry_variants()


# # def build_full_range_seed_values(range_lookup):
# #     values = []
# #     for ke0 in FAST_SEED_KE0_MEV:
# #         if ke0 <= range_lookup.threshold_mev:
# #             continue
# #         r = range_lookup.energy_to_range_mm(float(ke0))
# #         if np.isfinite(r) and r > 0:
# #             values.append(float(r))

# #     values.extend(float(r) for r in FAST_SEED_FULL_RANGES_MM)

# #     max_r = float(range_lookup.overall_distances_mm[-1])
# #     values = [r for r in values if np.isfinite(r) and 0.0 < r <= max_r]

# #     unique = []
# #     seen = set()
# #     for r in values:
# #         sig = round(float(r), 6)
# #         if sig not in seen:
# #             seen.add(sig)
# #             unique.append(float(r))
# #     return unique


# # def build_fast_seed_grid(range_lookup):
# #     seeds = []

# #     if IS_FULL_LENGTH_MODE:
# #         # 7-parameter original/full-length mode: scan only one length-like parameter.
# #         if FAST_SEED_FULL_CARTESIAN:
# #             for x0 in FAST_SEED_X0:
# #                 for y0 in FAST_SEED_Y0:
# #                     for z0 in FAST_SEED_Z0:
# #                         for length in FAST_SEED_VISIBLE_LENGTHS:
# #                             for cx, cy in FAST_SEED_DIRECTIONS:
# #                                 seeds.append({
# #                                     "x0": float(x0),
# #                                     "y0": float(y0),
# #                                     "z0": float(z0),
# #                                     "cx": float(cx),
# #                                     "cy": float(cy),
# #                                     "length": float(length),
# #                                     "t0": 0.0,
# #                                 })
# #         else:
# #             for z0 in FAST_SEED_Z0:
# #                 for length in FAST_SEED_VISIBLE_LENGTHS:
# #                     for geom in FAST_SEED_GEOMETRY_VARIANTS:
# #                         seeds.append({
# #                             "x0": float(geom["x0"]),
# #                             "y0": float(geom["y0"]),
# #                             "z0": float(z0),
# #                             "cx": float(geom["cx"]),
# #                             "cy": float(geom["cy"]),
# #                             "length": float(length),
# #                             "t0": 0.0,
# #                         })
# #         keys = ("x0", "y0", "z0", "cx", "cy", "length", "t0")
# #     else:
# #         # 8-parameter absorption mode: scan visible cutoff length and full CSDA range separately.
# #         full_range_seeds = build_full_range_seed_values(range_lookup)
# #         if FAST_SEED_FULL_CARTESIAN:
# #             for x0 in FAST_SEED_X0:
# #                 for y0 in FAST_SEED_Y0:
# #                     for z0 in FAST_SEED_Z0:
# #                         for visible_length in FAST_SEED_VISIBLE_LENGTHS:
# #                             for full_range in full_range_seeds:
# #                                 if visible_length > full_range:
# #                                     continue
# #                                 for cx, cy in FAST_SEED_DIRECTIONS:
# #                                     seeds.append({
# #                                         "x0": float(x0),
# #                                         "y0": float(y0),
# #                                         "z0": float(z0),
# #                                         "cx": float(cx),
# #                                         "cy": float(cy),
# #                                         "visible_length": float(visible_length),
# #                                         "full_range": float(full_range),
# #                                         "t0": 0.0,
# #                                     })
# #         else:
# #             for z0 in FAST_SEED_Z0:
# #                 for visible_length in FAST_SEED_VISIBLE_LENGTHS:
# #                     for full_range in full_range_seeds:
# #                         if visible_length > full_range:
# #                             continue
# #                         for geom in FAST_SEED_GEOMETRY_VARIANTS:
# #                             seeds.append({
# #                                 "x0": float(geom["x0"]),
# #                                 "y0": float(geom["y0"]),
# #                                 "z0": float(z0),
# #                                 "cx": float(geom["cx"]),
# #                                 "cy": float(geom["cy"]),
# #                                 "visible_length": float(visible_length),
# #                                 "full_range": float(full_range),
# #                                 "t0": 0.0,
# #                             })
# #         keys = ("x0", "y0", "z0", "cx", "cy", "visible_length", "full_range", "t0")

# #     unique = []
# #     seen = set()
# #     for seed in seeds:
# #         sig = tuple(float(seed[k]) for k in keys)
# #         if sig not in seen:
# #             seen.add(sig)
# #             unique.append(seed)
# #     return unique


# # PARAM_NAMES = FIT_PARAMETER_NAMES

# # # =============================================================================
# # # GLOBALS SHARED BY WORKERS
# # # =============================================================================
# # OVERALL_DISTANCES = None
# # INIT_ENERGY_TABLE = None
# # RANGE_LOOKUP = None

# # WCD = None
# # PMT_MODEL = None
# # EMITTER_TEMPLATE = None
# # P_LOCATIONS = None
# # DIRECTION_ZS = None
# # RING_KEEP_MASK = None
# # CORR_POS = None
# # MPMT_SLOTS_ALL = None
# # MPMT_TYPE_CODES_ALL = None

# # OBS_PES_ALL = None
# # OBS_TS_ALL = None
# # GOOD_WCTE_PMTS_SET = None

# # # =============================================================================
# # # mPMT INFO / EFFICIENCY TABLES
# # # =============================================================================
# # other_mpmt_info_path = Path(os.environ.get("OTHER_MPMT_INFO_PATH", str(TABLE_DIR / "other_mpmt_info_v2.dict")))
# # if other_mpmt_info_path.exists():
# #     with open(other_mpmt_info_path, "rb") as f:
# #         mpmt_info = pickle.load(f)
# # else:
# #     mpmt_info = {}

# # rel_mpmt_eff_path = Path(os.environ.get("REL_MPMT_EFF_PATH", str(TABLE_DIR / "rel_mpmt_eff.dict")))
# # if rel_mpmt_eff_path.exists():
# #     with open(rel_mpmt_eff_path, "rb") as f:
# #         rel_mpmt_eff = pickle.load(f)
# # else:
# #     unity = np.ones(200, dtype=np.float64)
# #     rel_mpmt_eff = {
# #         "tri_exsitu": unity,
# #         "tri_insitu": unity,
# #         "wut_insitu": unity,
# #         "wut_exsitu": unity,
# #     }

# # tri_exsitu = rel_mpmt_eff["tri_exsitu"]
# # tri_insitu = rel_mpmt_eff["tri_insitu"]
# # wut_insitu = rel_mpmt_eff["wut_insitu"]
# # wut_exsitu = rel_mpmt_eff["wut_exsitu"]


# # def get_mpmt_slot_type(mpmt_slots):
# #     slot_type = []
# #     for slot in mpmt_slots:
# #         slot = int(slot)
# #         try:
# #             if mpmt_info[slot]["mpmt_site"] == "TRI":
# #                 if mpmt_info[slot]["mpmt_type"] == "In-situ":
# #                     slot_type.append("tri_insitu")
# #                 else:
# #                     slot_type.append("tri_exsitu")
# #             else:
# #                 if mpmt_info[slot]["mpmt_type"] == "In-situ":
# #                     slot_type.append("wut_insitu")
# #                 else:
# #                     slot_type.append("wut_exsitu")
# #         except Exception:
# #             slot_type.append("empty")
# #     return slot_type



# # _MPMT_TYPE_TO_CODE_LOCAL = {
# #     "tri_exsitu": 0,
# #     "tri_insitu": 1,
# #     "wut_exsitu": 2,
# #     "wut_insitu": 3,
# # }

# # def get_mpmt_slot_type_codes(mpmt_slots):
# #     """Same information as get_mpmt_slot_type(), but encoded once as int8.

# #     Emitter treats integer mPMT types directly, avoiding repeated string
# #     comparisons in seed scans and the first FCN call of each Minuit attempt.
# #     """
# #     types = get_mpmt_slot_type(mpmt_slots)
# #     return np.asarray([_MPMT_TYPE_TO_CODE_LOCAL.get(t, -1) for t in types], dtype=np.int8)

# # # =============================================================================
# # # CONFIG / TRUTH HELPERS
# # # =============================================================================
# # def load_good_wcte_pmts():
# #     try:
# #         with uproot.open(CONFIG_ROOT_FILE) as f:
# #             t_c = f["Configuration"]
# #             arr_config = t_c.arrays(library="ak")
# #         good = np.asarray(arr_config["good_wcte_pmts"][0], dtype=int)
# #         print("Loaded GOOD_WCTE_PMTS from:", CONFIG_ROOT_FILE)
# #         return set(good.tolist())
# #     except Exception as exc:
# #         # For selected ROOT input, missing the run Configuration tree is usually a
# #         # real problem.  For user-provided event files, allow a self-contained
# #         # fallback by default and turn on every non-inactive PMT.
# #         allow_missing_default = EVENT_SOURCE == "file"
# #         if _env_bool("ALLOW_MISSING_GOOD_PMTS", allow_missing_default):
# #             print("WARNING: could not load GOOD_WCTE_PMTS; using all non-inactive PMTs.")
# #             print("Reason:", repr(exc))
# #             all_ids = []
# #             for slot in range(106):
# #                 if slot in INACTIVE_SLOTS_SET:
# #                     continue
# #                 for pmt_pos in range(19):
# #                     all_ids.append(slot * 100 + pmt_pos)
# #             return set(all_ids)
# #         raise


# # def configure_truth_params():
# #     if IS_ABSORPTION_MODE:
# #         if TRUE_FULL_RANGE_MM is not None:
# #             TRUE_PARAMS["full_range"] = float(TRUE_FULL_RANGE_MM)
# #         elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
# #             TRUE_PARAMS["full_range"] = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))

# #         if TRUE_VISIBLE_LENGTH_MM is not None:
# #             TRUE_PARAMS["visible_length"] = float(TRUE_VISIBLE_LENGTH_MM)
# #         elif TRUE_LENGTH_MM is not None:
# #             TRUE_PARAMS["visible_length"] = float(TRUE_LENGTH_MM)
# #         else:
# #             TRUE_PARAMS["visible_length"] = np.nan

# #         if np.isfinite(TRUE_PARAMS["visible_length"]):
# #             TRUE_PARAMS["length"] = TRUE_PARAMS["visible_length"]
# #         return

# #     # Full-length mode: the single fitted length is also the full CSDA range.
# #     if TRUE_LENGTH_MM is not None:
# #         length = float(TRUE_LENGTH_MM)
# #     elif TRUE_FULL_RANGE_MM is not None:
# #         length = float(TRUE_FULL_RANGE_MM)
# #     elif TRUE_VISIBLE_LENGTH_MM is not None:
# #         length = float(TRUE_VISIBLE_LENGTH_MM)
# #     elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
# #         length = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))
# #     else:
# #         length = np.nan

# #     TRUE_PARAMS["length"] = length
# #     TRUE_PARAMS["visible_length"] = length
# #     TRUE_PARAMS["full_range"] = length


# # # =============================================================================
# # # EVENT / OBSERVABLE HELPERS
# # # =============================================================================
# # def sim_to_event(
# #     sim_data,
# #     WCD,
# #     n_mpmt_total=106,
# #     pe_scale=143,
# #     shift_times=True,
# #     n_earliest_for_t0=10,
# # ):
# #     vw = 223.0598645833333  # mm/ns

# #     ev = Event(0, 0, n_mpmt_total)
# #     ev.set_mpmt_status(list(range(n_mpmt_total)), False)

# #     active_wcte_pmt_ids = []

# #     for slot in range(n_mpmt_total):
# #         if slot in INACTIVE_SLOTS_SET:
# #             continue

# #         slot_has_good_pmt = False
# #         for pmt_pos_id in range(ev.npmt_per_mpmt):
# #             wcte_pmt = int(slot * 100 + pmt_pos_id)
# #             if wcte_pmt in GOOD_WCTE_PMTS_SET:
# #                 ev.set_pmt_status(slot, [pmt_pos_id], True)
# #                 slot_has_good_pmt = True
# #                 active_wcte_pmt_ids.append(wcte_pmt)

# #         if slot_has_good_pmt:
# #             ev.set_mpmt_status([slot], True)

# #     for i in range(len(sim_data[:, 0])):
# #         wcte_pmt = int(sim_data[i, 0])
# #         slot = int(wcte_pmt // 100)
# #         pmt_pos_id = int(wcte_pmt % 100)

# #         if slot < 0 or slot >= ev.n_mpmt:
# #             continue
# #         if pmt_pos_id < 0 or pmt_pos_id >= ev.npmt_per_mpmt:
# #             continue
# #         if not ev.mpmt_status[slot]:
# #             continue
# #         if not ev.pmt_status[slot][pmt_pos_id]:
# #             continue

# #         ev.hit_charges[slot][pmt_pos_id].append(float(sim_data[i, 1]))
# #         ev.hit_times[slot][pmt_pos_id].append(float(sim_data[i, 2]))

# #     if shift_times:
# #         bp_loc = np.array([0.0, 0.0, -1350.0])
# #         early_hits = []

# #         for i_mpmt in range(ev.n_mpmt):
# #             if not ev.mpmt_status[i_mpmt]:
# #                 continue
# #             for i_pmt in range(ev.npmt_per_mpmt):
# #                 if not ev.pmt_status[i_mpmt][i_pmt]:
# #                     continue
# #                 if len(ev.hit_times[i_mpmt][i_pmt]) == 0:
# #                     continue

# #                 pmt_loc = WCD.mpmts[i_mpmt].pmts[i_pmt].get_placement("est")["location"]
# #                 r = np.linalg.norm(pmt_loc - bp_loc)

# #                 for t in ev.hit_times[i_mpmt][i_pmt]:
# #                     early_hits.append({
# #                         "time": float(t),
# #                         "t0_est": float(t) - r / vw,
# #                     })

# #         if len(early_hits) > 0:
# #             early_hits = sorted(early_hits, key=lambda x: x["time"])
# #             n_use = min(n_earliest_for_t0, len(early_hits))
# #             time_offset = np.median([hit["t0_est"] for hit in early_hits[:n_use]])

# #             for i_mpmt in range(ev.n_mpmt):
# #                 for i_pmt in range(ev.npmt_per_mpmt):
# #                     ev.hit_times[i_mpmt][i_pmt] = [
# #                         t - time_offset for t in ev.hit_times[i_mpmt][i_pmt]
# #                     ]

# #             ev.global_time_offset = time_offset

# #     return ev, np.asarray(active_wcte_pmt_ids, dtype=int)


# # def build_observables_from_event(ev, pe_scale=143):
# #     obs_pes = []
# #     obs_ts = []

# #     for i_mpmt in range(ev.n_mpmt):
# #         if not ev.mpmt_status[i_mpmt]:
# #             continue
# #         for i_pmt in range(ev.npmt_per_mpmt):
# #             if not ev.pmt_status[i_mpmt][i_pmt]:
# #                 continue

# #             q = np.asarray(ev.hit_charges[i_mpmt][i_pmt], dtype=np.float64)
# #             t = np.asarray(ev.hit_times[i_mpmt][i_pmt], dtype=np.float64)

# #             if q.size == 0:
# #                 obs_pes.append(0.0)
# #                 obs_ts.append(np.nan)
# #             else:
# #                 obs_pes.append(float(np.sum(q)) / pe_scale)
# #                 obs_ts.append(float(np.sum(q * t) / np.sum(q)))

# #     return np.asarray(obs_pes, dtype=np.float64), np.asarray(obs_ts, dtype=np.float64)


# # def apply_ring_mask_to_observables(obs_pes, obs_ts, ring_keep_mask, mode="both"):
# #     obs_pes = obs_pes.copy()
# #     obs_ts = obs_ts.copy()

# #     if mode not in {"none", "pes", "ts", "both"}:
# #         raise ValueError("RING_MASK_MODE must be one of: none, pes, ts, both")

# #     if mode in {"pes", "both"}:
# #         obs_pes[~ring_keep_mask] = 0.0
# #     if mode in {"ts", "both"}:
# #         obs_ts[~ring_keep_mask] = np.nan

# #     return obs_pes, obs_ts


# # def get_t0_prior_sigma(obs_pes, obs_ts):
# #     n_timed = np.count_nonzero(np.isfinite(obs_ts))
# #     total_pe = np.sum(obs_pes)

# #     if (n_timed < 250) or (total_pe < 300):
# #         return 0.1
# #     elif (n_timed < 275) or (total_pe < 350):
# #         return 0.2
# #     elif (n_timed < 300) or (total_pe < 400):
# #         return 0.3
# #     elif (n_timed < 325) or (total_pe < 450):
# #         return 0.4
# #     elif (n_timed < 350) or (total_pe < 500):
# #         return 0.5
# #     elif (n_timed < 375) or (total_pe < 550):
# #         return 0.6
# #     elif (n_timed < 400) or (total_pe < 600):
# #         return 0.7
# #     elif (n_timed < 425) or (total_pe < 650):
# #         return 0.8
# #     elif (n_timed < 450) or (total_pe < 700):
# #         return 1.0
# #     elif (n_timed < 475) or (total_pe < 750):
# #         return 1.2
# #     elif (n_timed < 500) or (total_pe < 800):
# #         return 1.4
# #     elif (n_timed < 525) or (total_pe < 850):
# #         return 1.6
# #     elif (n_timed < 550) or (total_pe < 900):
# #         return 1.8
# #     else:
# #         return 2.0


# # # =============================================================================
# # # LIKELIHOOD EVALUATION
# # # =============================================================================
# # def get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts):
# #     exp_pes = np.asarray(exp_pes, dtype=np.float64)
# #     obs_pes = np.asarray(obs_pes, dtype=np.float64)
# #     exp_ts = np.asarray(exp_ts, dtype=np.float64)
# #     obs_ts = np.asarray(obs_ts, dtype=np.float64)

# #     mask = (
# #         (exp_pes > 0.0)
# #         & (obs_pes > 0.0)
# #         & np.isfinite(exp_ts)
# #         & np.isfinite(obs_ts)
# #     )

# #     if not np.any(mask):
# #         return 1e30

# #     sigma_t = PMT_MODEL.single_pe_time_std / np.sqrt(obs_pes[mask])
# #     dt = (obs_ts[mask] - exp_ts[mask]) / sigma_t
# #     return float(0.5 * np.sum(dt * dt))


# # def evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts):
# #     if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
# #         return PMT_MODEL.get_neg_log_likelihood_npe_t(exp_pes, obs_pes, exp_ts, obs_ts)
# #     if USE_CHARGE_LIKELIHOOD:
# #         return PMT_MODEL.get_neg_log_likelihood_npe(exp_pes, obs_pes)
# #     return get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts)


# # def evaluate_neg_log_likelihood(
# #     obs_pes,
# #     obs_ts,
# #     emitter,
# #     mpmt_types,
# #     x0,
# #     y0,
# #     z0,
# #     cx,
# #     cy,
# #     length_or_visible,
# #     full_range_or_t0,
# #     t0=None,
# # ):
# #     """Evaluate the selected fit mode.

# #     full_length mode receives: length_or_visible=length, full_range_or_t0=t0.
# #     absorption mode receives:  length_or_visible=visible_length, full_range_or_t0=full_range, t0=t0.
# #     """
# #     if IS_ABSORPTION_MODE:
# #         visible_length = float(length_or_visible)
# #         full_range = float(full_range_or_t0)
# #         t0 = float(t0)

# #         if not np.isfinite(visible_length) or not np.isfinite(full_range):
# #             return 1e30
# #         if visible_length < 0.0 or full_range <= 0.0:
# #             return 1e30
# #         if visible_length > full_range:
# #             return 1e30
# #         if full_range > float(RANGE_LOOKUP.overall_distances_mm[-1]):
# #             return 1e30

# #         ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))
# #         if (not np.isfinite(ke0)) or ke0 <= FIT_PARTICLE_THRESHOLD_MEV:
# #             return 1e30
# #         emitter.fixed_initial_KE = ke0
# #         track_length_for_emission = visible_length
# #     else:
# #         length = float(length_or_visible)
# #         t0 = float(full_range_or_t0)
# #         if not np.isfinite(length) or length < 0.0:
# #             return 1e30
# #         if length > float(RANGE_LOOKUP.overall_distances_mm[-1]):
# #             return 1e30
# #         emitter.fixed_initial_KE = None
# #         track_length_for_emission = length

# #     cz2 = 1.0 - cx * cx - cy * cy
# #     if cz2 <= 0.0:
# #         return 1e30

# #     cz = np.sqrt(cz2)
# #     emitter.start_coord = (float(x0), float(y0), float(z0))
# #     emitter.starting_time = float(t0)
# #     emitter.direction = (float(cx), float(cy), float(cz))

# #     init_ke = emitter.refresh_kinematics_from_length(track_length_for_emission)

# #     if hasattr(emitter, "visible_length_is_physical"):
# #         if not emitter.visible_length_is_physical():
# #             return 1e30
# #     elif getattr(emitter, "last_visible_length_exceeds_range", False):
# #         return 1e30

# #     s = emitter.get_emission_points(P_LOCATIONS, init_ke)
# #     exp_pes, exp_ts = emitter.get_expected_pes_ts(
# #         WCD,
# #         s,
# #         P_LOCATIONS,
# #         DIRECTION_ZS,
# #         mpmt_types,
# #         obs_pes,
# #         need_times=USE_TIMING_LIKELIHOOD,
# #     )

# #     nll = evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts)
# #     if not np.isfinite(nll):
# #         return 1e30

# #     if USE_TIMING_LIKELIHOOD and USE_T0_PRIOR:
# #         sigma_t0 = get_t0_prior_sigma(obs_pes, obs_ts)
# #         nll += abs(0.5 * (float(t0) / sigma_t0) ** 2)

# #     return float(nll)


# # def _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed):
# #     if IS_ABSORPTION_MODE:
# #         return evaluate_neg_log_likelihood(
# #             obs_pes, obs_ts, emitter, mpmt_types,
# #             seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
# #             seed["visible_length"], seed["full_range"], seed["t0"],
# #         )
# #     return evaluate_neg_log_likelihood(
# #         obs_pes, obs_ts, emitter, mpmt_types,
# #         seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
# #         seed["length"], seed["t0"],
# #     )


# # def select_best_initial_seed(obs_pes, obs_ts, init_param_sets, mpmt_types=None):
# #     """
# #     Cheap deterministic seed prescan.

# #     For production fits, only the best seed is retained.  This is faster and
# #     avoids moving a huge seed-scan list between multiprocessing workers.  When
# #     rescue/debug output is enabled, the full sorted scan is kept.
# #     """
# #     best_info = None
# #     seed_scan = [] if NEED_FULL_SEED_SCAN else None

# #     for i, seed in enumerate(init_param_sets):
# #         emitter = EMITTER_TEMPLATE.copy()

# #         fval = _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed)

# #         if not np.isfinite(fval):
# #             fval = np.inf

# #         info = {
# #             "seed_index": int(i),
# #             "fval": float(fval),
# #             "params": dict(seed),
# #         }

# #         if seed_scan is not None:
# #             seed_scan.append(info)

# #         if best_info is None or fval < best_info["fval"]:
# #             best_info = info

# #     if best_info is None or not np.isfinite(best_info["fval"]):
# #         raise RuntimeError("All seed FCNs were non-finite. Check ranges, PMT ordering, and event selection.")

# #     if seed_scan is not None:
# #         seed_scan_sorted = sorted(seed_scan, key=lambda x: x["fval"])
# #     else:
# #         seed_scan_sorted = [best_info]

# #     best = seed_scan_sorted[0]
# #     return dict(best["params"]), int(best["seed_index"]), float(best["fval"]), seed_scan_sorted


# # def compute_true_fcn_for_event(event_index):
# #     if IS_ABSORPTION_MODE:
# #         if not np.isfinite(TRUE_PARAMS.get("visible_length", np.nan)):
# #             return np.nan
# #         if not np.isfinite(TRUE_PARAMS.get("full_range", np.nan)):
# #             return np.nan
# #         length_args = (TRUE_PARAMS["visible_length"], TRUE_PARAMS["full_range"], TRUE_PARAMS["t0"])
# #     else:
# #         if not np.isfinite(TRUE_PARAMS.get("length", np.nan)):
# #             return np.nan
# #         length_args = (TRUE_PARAMS["length"], TRUE_PARAMS["t0"])

# #     mpmt_types = MPMT_TYPE_CODES_ALL[event_index]
# #     emitter = EMITTER_TEMPLATE.copy()
# #     return evaluate_neg_log_likelihood(
# #         OBS_PES_ALL[event_index],
# #         OBS_TS_ALL[event_index],
# #         emitter,
# #         mpmt_types,
# #         TRUE_PARAMS["x0"],
# #         TRUE_PARAMS["y0"],
# #         TRUE_PARAMS["z0"],
# #         TRUE_PARAMS["cx"],
# #         TRUE_PARAMS["cy"],
# #         *length_args,
# #     )


# # # =============================================================================
# # # MINUIT HELPERS
# # # =============================================================================
# # def make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types=None):
# #     emitter = EMITTER_TEMPLATE.copy()

# #     if IS_ABSORPTION_MODE:
# #         def nll(x0, y0, z0, cx, cy, visible_length, full_range, t0):
# #             return evaluate_neg_log_likelihood(
# #                 obs_pes, obs_ts, emitter, mpmt_types,
# #                 x0, y0, z0, cx, cy, visible_length, full_range, t0,
# #             )
# #     else:
# #         def nll(x0, y0, z0, cx, cy, length, t0):
# #             return evaluate_neg_log_likelihood(
# #                 obs_pes, obs_ts, emitter, mpmt_types,
# #                 x0, y0, z0, cx, cy, length, t0,
# #             )

# #     m = Minuit(nll, **start_params)

# #     max_range = float(RANGE_LOOKUP.overall_distances_mm[-1])

# #     m.limits["x0"] = (-2000, 2000)
# #     m.limits["y0"] = (-2000, 2000)
# #     m.limits["z0"] = (-2000, 2000)
# #     m.limits["cx"] = (-0.5, 0.5)
# #     m.limits["cy"] = (-0.5, 0.5)
# #     m.limits["t0"] = (-8.0,8.0) #T0_LIMITS

# #     m.errors["x0"] = 30.0
# #     m.errors["y0"] = 30.0
# #     m.errors["z0"] = 30.0
# #     m.errors["cx"] = 0.01
# #     m.errors["cy"] = 0.01
# #     m.errors["t0"] = 0.1

# #     if IS_ABSORPTION_MODE:
# #         m.limits["visible_length"] = (0.0, max_range)
# #         m.limits["full_range"] = (1.0, max_range)
# #         m.errors["visible_length"] = 60.0
# #         m.errors["full_range"] = 100.0
# #     else:
# #         m.limits["length"] = (0.0, max_range)
# #         m.errors["length"] = 60.0

# #     if not USE_TIMING_LIKELIHOOD:
# #         m.fixed["t0"] = True

# #     m.errordef = Minuit.LIKELIHOOD
# #     m.strategy = M_STRAT

# #     return m


# # def is_bad_minuit_result(m, *, edm_max=1e10):
# #     if (m.fval is None) or (not np.isfinite(m.fval)):
# #         return True
# #     # Do not use m.valid as a bad-result criterion.
# #     try:
# #         if (m.fmin is not None) and np.isfinite(m.fmin.edm) and (m.fmin.edm > edm_max):
# #             return True
# #     except Exception:
# #         pass
# #     return False


# # def run_minuit_attempt(m, ncall):
# #     if not ENABLE_STAGE2_MIGRAD_FIRST:
# #         m.strategy = M_STRAT
# #         m.simplex(ncall=ncall)
# #         m.migrad(ncall=ncall)
# #         return m

# #     ncall_fast = max(2000, int(0.35 * ncall))
# #     ncall_simplex = max(2000, int(0.25 * ncall))

# #     m.strategy = 0
# #     m.migrad(ncall=ncall_fast)

# #     if is_bad_minuit_result(m):
# #         m.simplex(ncall=ncall_simplex)
# #         m.strategy = M_STRAT
# #         m.migrad(ncall=ncall)

# #     return m


# # # =============================================================================
# # # ADAPTIVE RESCUE
# # # =============================================================================
# # ENABLE_ADAPTIVE_RESCUE = ENABLE_STAGE3_ADAPTIVE_RESCUE
# # RESCUE_MAX_SEEDS = 6
# # RESCUE_LENGTH_BINS = [
# #     (0.0, 1000.0),
# #     (1000.0, 1250.0),
# #     (1250.0, 1400.0),
# #     (1400.0, 1700.0),
# #     (1700.0, 3000.0),
# # ]


# # def result_length_value(values):
# #     return float(values["visible_length"] if IS_ABSORPTION_MODE else values["length"])


# # def result_full_range_value(values):
# #     return float(values["full_range"] if IS_ABSORPTION_MODE else values["length"])


# # def seed_length_value(params):
# #     return float(params["visible_length"] if IS_ABSORPTION_MODE else params["length"])


# # def result_ke0_from_values(values):
# #     try:
# #         return float(RANGE_LOOKUP.range_mm_to_energy(result_full_range_value(values)))
# #     except Exception:
# #         return np.nan


# # def needs_rescue_result(result):
# #     if result is None:
# #         return True
# #     if not np.isfinite(result.get("fval", np.inf)):
# #         return True
# #     values = result.get("values", {})
# #     try:
# #         fitted_length = result_length_value(values)
# #         fitted_full = result_full_range_value(values)
# #     except Exception:
# #         return True
# #     if (not np.isfinite(fitted_length)) or (not np.isfinite(fitted_full)):
# #         return True
# #     if fitted_length <= 10.0 or fitted_length >= VISIBLE_LENGTH_RETRY_THRESHOLD:
# #         return True
# #     if IS_ABSORPTION_MODE and fitted_length > fitted_full:
# #         return True
# #     if result.get("seed_stuck", False):
# #         return True
# #     if USE_TIMING_LIKELIHOOD and result.get("below_t_min", False):
# #         return True
# #     return False


# # def needs_fcn_retry_result(result, fcn_threshold):
# #     """Return True when the default, non-adaptive FCN retry should run.

# #     This intentionally does not use m.valid.  A retry is triggered only by a
# #     non-finite FCN or by FCN exceeding the user-configured threshold.
# #     """
# #     if result is None:
# #         return True
# #     fval = float(result.get("fval", np.inf))
# #     if not np.isfinite(fval):
# #         return True
# #     if fcn_threshold is None:
# #         return False
# #     try:
# #         threshold = float(fcn_threshold)
# #     except Exception:
# #         return False
# #     return np.isfinite(threshold) and (fval > threshold)


# # def next_untried_seed_info(seed_scan_sorted, tried_seed_indices):
# #     """Return the best remaining seed from the prescan, or None if exhausted."""
# #     for seed_info in seed_scan_sorted:
# #         idx = int(seed_info["seed_index"])
# #         if idx not in tried_seed_indices:
# #             return seed_info
# #     return None


# # def choose_diverse_rescue_seed_infos(seed_scan_sorted, already_tried_seed_indices=None, max_total=RESCUE_MAX_SEEDS):
# #     already = set() if already_tried_seed_indices is None else set(already_tried_seed_indices)
# #     chosen = []

# #     for lo, hi in RESCUE_LENGTH_BINS:
# #         candidates = [
# #             s for s in seed_scan_sorted
# #             if int(s["seed_index"]) not in already
# #             and lo <= seed_length_value(s["params"]) < hi
# #         ]
# #         if candidates:
# #             chosen.append(candidates[0])
# #             already.add(int(candidates[0]["seed_index"]))
# #         if len(chosen) >= max_total:
# #             return chosen

# #     for s in seed_scan_sorted:
# #         idx = int(s["seed_index"])
# #         if idx in already:
# #             continue
# #         chosen.append(s)
# #         already.add(idx)
# #         if len(chosen) >= max_total:
# #             break

# #     return chosen


# # def compact_seed_scan(seed_scan_sorted):
# #     """Return the configured seed-scan payload for output/debugging."""
# #     if SAVE_SEED_SCAN:
# #         return seed_scan_sorted
# #     if SAVE_TOP_N_SEEDS > 0:
# #         return seed_scan_sorted[:SAVE_TOP_N_SEEDS]
# #     return []


# # def build_result_from_minuit(m, attempt, start_params, chosen_seed_idx, chosen_seed_fcn, seed_scan_sorted):
# #     current_fval = float(m.fval) if (m.fval is not None and np.isfinite(m.fval)) else np.inf
# #     current_values = m.values.to_dict()

# #     fitted_z0 = float(current_values["z0"])
# #     fitted_length = result_length_value(current_values)
# #     fitted_full = result_full_range_value(current_values)
# #     fitted_ke0 = result_ke0_from_values(current_values)

# #     visible_too_large = fitted_length > VISIBLE_LENGTH_RETRY_THRESHOLD
# #     z_near_seed = abs(fitted_z0 - float(start_params["z0"])) <= Z_SEED_EPS
# #     length_near_seed = abs(fitted_length - seed_length_value(start_params)) <= VISIBLE_LENGTH_SEED_EPS
# #     if IS_ABSORPTION_MODE:
# #         full_near_seed = abs(fitted_full - float(start_params["full_range"])) <= FULL_RANGE_SEED_EPS
# #     else:
# #         full_near_seed = False
# #     seed_stuck = z_near_seed and length_near_seed and (full_near_seed if IS_ABSORPTION_MODE else True)
# #     below_t_min = USE_TIMING_LIKELIHOOD and (current_values["t0"] < T_MIN)

# #     # Always expose consistent aliases in the result dictionary.
# #     visible_length_mm = fitted_length
# #     full_range_mm = fitted_full
# #     current_values.setdefault("length", fitted_length)
# #     current_values.setdefault("visible_length", visible_length_mm)
# #     current_values.setdefault("full_range", full_range_mm)

# #     return {
# #         "values": current_values,
# #         "errors": m.errors.to_dict(),
# #         "fval": current_fval,
# #         "valid": bool(m.valid),
# #         "attempts": attempt,
# #         "visible_length_too_large": bool(visible_too_large),
# #         "length_too_large": bool(visible_too_large),
# #         "seed_stuck": bool(seed_stuck),
# #         "z_near_seed": bool(z_near_seed),
# #         "visible_length_near_seed": bool(length_near_seed),
# #         "full_range_near_seed": bool(full_near_seed),
# #         "length_near_seed": bool(length_near_seed),
# #         "below_t_min": bool(below_t_min),
# #         "chosen_seed_index": int(chosen_seed_idx),
# #         "chosen_seed_fcn": float(chosen_seed_fcn) if np.isfinite(chosen_seed_fcn) else np.nan,
# #         "chosen_seed_params": dict(start_params),
# #         "seed_scan": compact_seed_scan(seed_scan_sorted),
# #         "visible_length_mm": visible_length_mm,
# #         "full_range_mm": full_range_mm,
# #         "length_mm": fitted_length,
# #         "ke0_mev": fitted_ke0,
# #         "edm": (
# #             float(m.fmin.edm)
# #             if (getattr(m, "fmin", None) is not None and m.fmin.edm is not None)
# #             else np.nan
# #         ),
# #     }


# # def result_sort_key(result):
# #     if result is None:
# #         return (999, np.inf)
# #     fval = float(result.get("fval", np.inf))
# #     penalty = 0
# #     if not np.isfinite(fval):
# #         penalty += 100
# #     if result.get("visible_length_too_large", False) or result.get("length_too_large", False):
# #         penalty += 10
# #     if result.get("seed_stuck", False):
# #         penalty += 5
# #     if result.get("below_t_min", False):
# #         penalty += 5
# #     return (penalty, fval)


# # # =============================================================================
# # # HARD-EVENT VISIBLE-LENGTH PROFILE RESCUE
# # # =============================================================================
# # ENABLE_LENGTH_PROFILE_RESCUE = ENABLE_STAGE4_LENGTH_PROFILE
# # LENGTH_PROFILE_GRID = list(FAST_SEED_VISIBLE_LENGTHS)
# # LENGTH_PROFILE_MAX_POINTS = 6


# # def run_length_profile_rescue(obs_pes, obs_ts, mpmt_types, seed_scan_sorted, ncall, starting_attempt_index=100):
# #     profile_results = []
# #     base_seed = dict(seed_scan_sorted[0]["params"])
# #     length_key = "visible_length" if IS_ABSORPTION_MODE else "length"

# #     for j, profile_length in enumerate(LENGTH_PROFILE_GRID[:LENGTH_PROFILE_MAX_POINTS]):
# #         start_params = dict(base_seed)
# #         start_params[length_key] = float(profile_length)
# #         if IS_ABSORPTION_MODE and start_params["visible_length"] > start_params["full_range"]:
# #             continue

# #         m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# #         m.fixed[length_key] = True
# #         run_minuit_attempt(m, max(5000, int(0.5 * ncall)))

# #         prof_result = build_result_from_minuit(
# #             m,
# #             attempt=starting_attempt_index + j,
# #             start_params=start_params,
# #             chosen_seed_idx=-1000 - j,
# #             chosen_seed_fcn=np.nan,
# #             seed_scan_sorted=seed_scan_sorted,
# #         )
# #         prof_result["profile_fixed_length"] = float(profile_length)
# #         prof_result["profile_fixed_visible_length"] = float(profile_length)
# #         profile_results.append(prof_result)

# #     if not profile_results:
# #         return None

# #     best_profile = min(profile_results, key=result_sort_key)

# #     polish_params = dict(best_profile["values"])
# #     # Keep only parameters actually used by this mode; Minuit will reject extras.
# #     polish_params = {k: polish_params[k] for k in PARAM_NAMES if k in polish_params}
# #     m = make_minuit_for_event(obs_pes, obs_ts, polish_params, mpmt_types)
# #     m.fixed[length_key] = False
# #     run_minuit_attempt(m, ncall)

# #     polish_result = build_result_from_minuit(
# #         m,
# #         attempt=starting_attempt_index + len(profile_results),
# #         start_params=polish_params,
# #         chosen_seed_idx=-2000,
# #         chosen_seed_fcn=float(best_profile["fval"]),
# #         seed_scan_sorted=seed_scan_sorted,
# #     )
# #     polish_result["length_profile_rescue_used"] = True
# #     polish_result["length_profile_results"] = profile_results
# #     polish_result["length_profile_best_fixed"] = best_profile
# #     return polish_result


# # def fit_one_event_by_index(args):
# #     event_index, init_param_sets, fcn_threshold, max_attempts, ncall = args

# #     obs_pes = OBS_PES_ALL[event_index]
# #     obs_ts = OBS_TS_ALL[event_index]
# #     mpmt_types = MPMT_TYPE_CODES_ALL[event_index]

# #     best_seed, best_seed_idx, best_seed_fval, seed_scan_sorted = select_best_initial_seed(
# #         obs_pes,
# #         obs_ts,
# #         init_param_sets,
# #         mpmt_types,
# #     )

# #     attempt_results = []
# #     tried_seed_indices = set()

# #     primary_info = seed_scan_sorted[0]
# #     start_params = dict(primary_info["params"])
# #     chosen_seed_idx = int(primary_info["seed_index"])
# #     chosen_seed_fcn = float(primary_info["fval"])
# #     tried_seed_indices.add(chosen_seed_idx)

# #     m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# #     run_minuit_attempt(m, ncall)
# #     primary_result = build_result_from_minuit(
# #         m,
# #         attempt=1,
# #         start_params=start_params,
# #         chosen_seed_idx=chosen_seed_idx,
# #         chosen_seed_fcn=chosen_seed_fcn,
# #         seed_scan_sorted=seed_scan_sorted,
# #     )
# #     attempt_results.append(primary_result)

# #     # ------------------------------------------------------------------
# #     # Default FCN retry path.  This is intentionally independent of the
# #     # adaptive-rescue stage: if the best result so far has FCN above the
# #     # configured threshold, try the next-best prescan seeds up to
# #     # MAX_FIT_ATTEMPTS.  Do not use m.valid as a bad-result criterion.
# #     # ------------------------------------------------------------------
# #     while len(attempt_results) < max(1, int(max_attempts)):
# #         best_so_far = min(attempt_results, key=result_sort_key)
# #         if not needs_fcn_retry_result(best_so_far, fcn_threshold):
# #             break

# #         seed_info = next_untried_seed_info(seed_scan_sorted, tried_seed_indices)
# #         if seed_info is None:
# #             break

# #         start_params = dict(seed_info["params"])
# #         chosen_seed_idx = int(seed_info["seed_index"])
# #         chosen_seed_fcn = float(seed_info["fval"])
# #         tried_seed_indices.add(chosen_seed_idx)

# #         m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# #         run_minuit_attempt(m, ncall)
# #         result = build_result_from_minuit(
# #             m,
# #             attempt=len(attempt_results) + 1,
# #             start_params=start_params,
# #             chosen_seed_idx=chosen_seed_idx,
# #             chosen_seed_fcn=chosen_seed_fcn,
# #             seed_scan_sorted=seed_scan_sorted,
# #         )
# #         result["fcn_retry_used"] = True
# #         attempt_results.append(result)

# #     best_result = min(attempt_results, key=result_sort_key)

# #     if ENABLE_ADAPTIVE_RESCUE and needs_rescue_result(best_result):
# #         rescue_seed_infos = choose_diverse_rescue_seed_infos(
# #             seed_scan_sorted,
# #             already_tried_seed_indices=tried_seed_indices,
# #             max_total=RESCUE_MAX_SEEDS,
# #         )

# #         for seed_info in rescue_seed_infos:
# #             start_params = dict(seed_info["params"])
# #             chosen_seed_idx = int(seed_info["seed_index"])
# #             chosen_seed_fcn = float(seed_info["fval"])
# #             tried_seed_indices.add(chosen_seed_idx)

# #             m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# #             run_minuit_attempt(m, ncall)
# #             result = build_result_from_minuit(
# #                 m,
# #                 attempt=len(attempt_results) + 1,
# #                 start_params=start_params,
# #                 chosen_seed_idx=chosen_seed_idx,
# #                 chosen_seed_fcn=chosen_seed_fcn,
# #                 seed_scan_sorted=seed_scan_sorted,
# #             )
# #             result["adaptive_rescue_attempt"] = True
# #             attempt_results.append(result)

# #     best_result = min(attempt_results, key=result_sort_key)

# #     if ENABLE_LENGTH_PROFILE_RESCUE and needs_rescue_result(best_result):
# #         profile_result = run_length_profile_rescue(
# #             obs_pes,
# #             obs_ts,
# #             mpmt_types,
# #             seed_scan_sorted,
# #             ncall,
# #             starting_attempt_index=100 + len(attempt_results),
# #         )
# #         if profile_result is not None:
# #             attempt_results.append(profile_result)
# #             best_result = min(attempt_results, key=result_sort_key)

# #     best_result["attempts"] = len(attempt_results)
# #     if SAVE_ATTEMPT_RESULTS:
# #         best_result["attempt_results"] = attempt_results
# #     else:
# #         best_result["attempt_results"] = []
# #     best_result["adaptive_rescue_used"] = bool(len(attempt_results) > 1)
# #     best_result["length_profile_rescue_considered"] = bool(ENABLE_LENGTH_PROFILE_RESCUE)
# #     best_result["length_profile_rescue_used"] = bool(
# #         best_result.get("length_profile_rescue_used", False)
# #         or any(r.get("length_profile_rescue_used", False) for r in attempt_results)
# #     )
# #     return best_result


# # def run_batch(event_indices, init_param_sets, nproc, fcn_threshold, max_attempts, ncall):
# #     args = [(idx, init_param_sets, fcn_threshold, max_attempts, ncall) for idx in event_indices]

# #     try:
# #         ctx = mp.get_context("fork")
# #     except ValueError:
# #         ctx = mp.get_context()

# #     with ctx.Pool(processes=nproc) as pool:
# #         return pool.map(fit_one_event_by_index, args)


# # # =============================================================================
# # # USER-PROVIDED EVENT FILE HELPERS
# # # =============================================================================
# # def _coerce_event_array(event, *, event_label="event"):
# #     arr = np.asarray(event)
# #     if arr.ndim != 2 or arr.shape[1] < 3:
# #         raise ValueError(
# #             f"{event_label} must be a 2D array with at least 3 columns: "
# #             "[pmt_id, charge, time]."
# #         )
# #     # Keep optional event-number columns, but the fitter only consumes columns 0:3.
# #     return np.asarray(arr[:, :4] if arr.shape[1] >= 4 else arr[:, :3], dtype=np.float64)


# # def _events_from_loaded_object(obj):
# #     """Normalize npy/npz/pickle payloads into a list of event arrays."""
# #     if isinstance(obj, dict):
# #         if USER_EVENT_KEY is not None:
# #             obj = obj[USER_EVENT_KEY]
# #         elif "events" in obj:
# #             obj = obj["events"]
# #         elif "data" in obj:
# #             obj = obj["data"]
# #         elif "arr_0" in obj:
# #             obj = obj["arr_0"]
# #         else:
# #             keys = ", ".join(map(str, obj.keys()))
# #             raise KeyError(
# #                 "Could not choose an event array from the dict payload. "
# #                 f"Available keys: {keys}. Set USER_EVENT_KEY."
# #             )

# #     if isinstance(obj, np.lib.npyio.NpzFile):
# #         if USER_EVENT_KEY is not None:
# #             key = USER_EVENT_KEY
# #         elif "events" in obj.files:
# #             key = "events"
# #         elif "data" in obj.files:
# #             key = "data"
# #         elif "arr_0" in obj.files:
# #             key = "arr_0"
# #         elif len(obj.files) == 1:
# #             key = obj.files[0]
# #         else:
# #             raise KeyError(
# #                 "Could not choose an event array from the npz payload. "
# #                 f"Available keys: {obj.files}. Set USER_EVENT_KEY."
# #             )
# #         obj = obj[key]

# #     if isinstance(obj, (list, tuple)):
# #         return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(obj)]

# #     arr = np.asarray(obj, dtype=object if getattr(obj, "dtype", None) == object else None)

# #     # Object arrays are normally lists of variable-length events.
# #     if arr.dtype == object and arr.ndim == 1:
# #         return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(arr)]

# #     # A 3D numeric array is N_events x N_hits x N_columns.
# #     if arr.ndim == 3:
# #         return [_coerce_event_array(arr[i], event_label=f"event[{i}]") for i in range(arr.shape[0])]

# #     # A 2D array with a 4th column is interpreted as a concatenated event table
# #     # grouped by event number.  A 2D array with only 3 columns is one event.
# #     if arr.ndim == 2:
# #         arr2 = np.asarray(arr, dtype=np.float64)
# #         if arr2.shape[1] >= 4:
# #             events = []
# #             event_numbers = arr2[:, 3].astype(np.int64)
# #             for evnum in np.unique(event_numbers):
# #                 events.append(_coerce_event_array(arr2[event_numbers == evnum], event_label=f"event_number={evnum}"))
# #             return events
# #         return [_coerce_event_array(arr2, event_label="single_event")]

# #     raise ValueError(
# #         "Unsupported USER_EVENT_FILE payload. Expected a list/object array of "
# #         "event arrays, a 3D event array, or a 2D [pmt_id, charge, time] table."
# #     )


# # def load_user_event_file(path, *, max_events=None):
# #     """Load user-provided, already-selected events from npy/npz/pickle files."""
# #     path = Path(path)
# #     if not path.exists():
# #         raise FileNotFoundError(f"USER_EVENT_FILE does not exist: {path}")

# #     suffix = path.suffix.lower()
# #     if suffix == ".npz":
# #         loaded = np.load(path, allow_pickle=True)
# #     elif suffix == ".npy":
# #         loaded = np.load(path, allow_pickle=True)
# #     elif suffix in {".pkl", ".pickle"}:
# #         with open(path, "rb") as f:
# #             loaded = pickle.load(f)
# #     else:
# #         raise ValueError(
# #             f"Unsupported USER_EVENT_FILE suffix {suffix!r}. Use .npy, .npz, .pkl, or .pickle."
# #         )

# #     events = _events_from_loaded_object(loaded)
# #     if max_events is not None:
# #         events = events[: int(max_events)]
# #     print(f"Loaded {len(events)} user-provided events from: {path}")
# #     return events


# # # =============================================================================
# # # MAIN DRIVER
# # # =============================================================================
# # def main():
# #     global OVERALL_DISTANCES, INIT_ENERGY_TABLE, RANGE_LOOKUP
# #     global WCD, PMT_MODEL, EMITTER_TEMPLATE, P_LOCATIONS, DIRECTION_ZS, RING_KEEP_MASK, CORR_POS
# #     global OBS_PES_ALL, OBS_TS_ALL, MPMT_SLOTS_ALL, MPMT_TYPE_CODES_ALL, GOOD_WCTE_PMTS_SET

# #     print("Likelihood mode:", LIKELIHOOD_MODE)
# #     print("Fit particle:", FIT_PARTICLE_CANONICAL)
# #     print("Particle mass [MeV]:", FIT_PARTICLE_MASS_MEV)
# #     print("Cherenkov threshold [MeV]:", FIT_PARTICLE_THRESHOLD_MEV)
# #     print("Fit mode:", TRACK_END_MODE)
# #     print("Fit parameters:", FIT_PARAMETER_NAMES)
# #     print("Output file:", OUTPUT_FILE)

# #     if EVENT_SOURCE == "selection" and get_selected_events is None:
# #         raise ImportError(
# #             "event_loader.py was not found. Copy it into LF_multiParticles/scripts "
# #             "or add its directory to PYTHONPATH, or set EVENT_SOURCE=file and USER_EVENT_FILE."
# #         )

# #     if EVENT_SOURCE == "file" and not USER_EVENT_FILE:
# #         raise ValueError("EVENT_SOURCE=file requires USER_EVENT_FILE=/path/to/events.npy|npz|pkl")

# #     GOOD_WCTE_PMTS_SET = load_good_wcte_pmts()

# #     RANGE_LOOKUP = ParticleRangeLookup(FIT_PARTICLE_CANONICAL, table_dirs=[str(TABLE_DIR)])
# #     print("Range table max KE [MeV]:", float(RANGE_LOOKUP.initial_energies_mev[-1]))
# #     print("Range table max full_range [mm]:", float(RANGE_LOOKUP.overall_distances_mm[-1]))

# #     configure_truth_params()
# #     if IS_ABSORPTION_MODE:
# #         truth_ready = np.isfinite(TRUE_PARAMS["visible_length"]) and np.isfinite(TRUE_PARAMS["full_range"])
# #         if truth_ready:
# #             print("Truth visible length [mm]:", TRUE_PARAMS["visible_length"])
# #             print("Truth full range [mm]:", TRUE_PARAMS["full_range"])
# #             print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["full_range"]))
# #         else:
# #             print("Truth FCN disabled: set TRUE_VISIBLE_LENGTH_MM and TRUE_FULL_RANGE_MM/TRUE_INITIAL_KE_MEV.")
# #     else:
# #         truth_ready = np.isfinite(TRUE_PARAMS["length"])
# #         if truth_ready:
# #             print("Truth length [mm]:", TRUE_PARAMS["length"])
# #             print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["length"]))
# #         else:
# #             print("Truth FCN disabled: set TRUE_LENGTH_MM or TRUE_INITIAL_KE_MEV.")

# #     init_param_sets = build_fast_seed_grid(RANGE_LOOKUP)
# #     if not init_param_sets:
# #         raise RuntimeError("Seed grid is empty. Check FAST_SEED_VISIBLE_LENGTHS and FAST_SEED_KE0_MEV/FULL_RANGES.")
# #     print("Number of initial seeds:", len(init_param_sets))

# #     for i, seed in enumerate(init_param_sets):
# #         missing = [k for k in PARAM_NAMES if k not in seed]
# #         if missing:
# #             raise ValueError(f"Seed {i} is missing keys: {missing}")

# #     set_active_particle(FIT_PARTICLE_CANONICAL)
# #     OVERALL_DISTANCES, INIT_ENERGY_TABLE, _distance_rows = get_energy_distance_tables(
# #         FIT_PARTICLE_CANONICAL
# #     )

# #     hall = Device.open_file(GEOMETRY_FILE)
# #     WCD = hall.wcds[0]

# #     initial_ke_seed = float(RANGE_LOOKUP.range_mm_to_energy(
# #         min(1000.0, float(RANGE_LOOKUP.overall_distances_mm[-1]))
# #     ))

# #     emitter_model = Emitter(
# #         0.0,
# #         (0.0, 0.0, 0.0),
# #         (0.0, 0.0, 1.0),
# #         0.96,
# #         500.0,
# #         18.0,
# #         particle=FIT_PARTICLE_CANONICAL,
# #         track_end_mode=EMITTER_TRACK_END_MODE,
# #         fixed_initial_KE=initial_ke_seed if IS_ABSORPTION_MODE else None,
# #     )

# #     delta_pdf_path = TABLE_DIR / "delta_e_angular_pdf_table.npz"
# #     if delta_pdf_path.exists() and hasattr(emitter_model, "load_delta_e_angular_pdf_table"):
# #         emitter_model.load_delta_e_angular_pdf_table(str(delta_pdf_path))

# #     PMT_MODEL = PMT(1.0, 0.3, 1.0, 40.0, 0.2, 0.0)
# #     EMITTER_TEMPLATE = emitter_model.copy()
# #     CORR_POS = None

# #     print("Building event observables...")

# #     obs_pes_all = []
# #     obs_ts_all = []
# #     mpmt_slots_all = []

# #     if EVENT_SOURCE == "selection":
# #         print("N_EVENTS AHHHHH", N_EVENTS)
# #         events = get_selected_events(
# #             RUN,
# #             N_EVENTS,
# #             particle=PARTICLE_SELECTION_LABEL,
# #             root_file=CONFIG_ROOT_FILE,
# #             use_peak_time_cut=USE_PEAK_TIME_CUT,
# #             peak_window=PEAK_WINDOW_NS,
# #             peak_bin_width=PEAK_BIN_WIDTH_NS,
# #             tof_primary=SELECTION_TOF_NS,
# #             tof_window=SELECTION_TOF_WINDOW_NS,
# #             tof_scalar_field=SELECTION_TOF_FIELD,
# #             momentum_scalar_field=SELECTION_MOMENTUM_FIELD,
# #             t5_particle_nr=SELECTION_T5_PARTICLE_NR,
# #         )
# #     else:
# #         events = load_user_event_file(USER_EVENT_FILE, max_events=N_EVENTS)

# #     tot_events = len(events)
# #     print("Total Events to Fit:", tot_events)

# #     for i in range(tot_events):
# #         event = np.asarray(events[i])
# #         if event.size == 0:
# #             continue

# #         apply_event_time_window = (EVENT_SOURCE == "selection") or USER_EVENT_APPLY_PEAK_WINDOW
# #         if apply_event_time_window:
# #             time_hist = np.histogram(event[:, 2], bins=np.arange(0, 4000))
# #             max_idx = int(np.argmax(time_hist[0]))
# #             lo_idx = max(0, max_idx - 20)
# #             hi_idx = min(len(time_hist[1]) - 1, max_idx + 5)
# #             min_time = time_hist[1][lo_idx]
# #             cut_time = time_hist[1][hi_idx]
# #             time_mask = (event[:, 2] > min_time) & (event[:, 2] < cut_time)
# #             event = event[time_mask]

# #         ev, pmt_ids = sim_to_event(event, WCD, n_mpmt_total=106, pe_scale=143)

# #         if P_LOCATIONS is None or DIRECTION_ZS is None:
# #             P_LOCATIONS, DIRECTION_ZS, MPMT_SLOTS = EMITTER_TEMPLATE.get_pmt_placements(ev, WCD, "est")
# #             MPMT_SLOTS = np.asarray(MPMT_SLOTS, dtype=int)
# #             RING_KEEP_MASK = np.isin(MPMT_SLOTS, ALL_RING)

# #         obs_pes, obs_ts = build_observables_from_event(ev, pe_scale=143)
# #         obs_pes, obs_ts = apply_ring_mask_to_observables(
# #             obs_pes,
# #             obs_ts,
# #             RING_KEEP_MASK,
# #             mode=RING_MASK_MODE,
# #         )

# #         obs_pes_all.append(obs_pes)
# #         obs_ts_all.append(obs_ts)
# #         mpmt_slots_all.append(MPMT_SLOTS)

# #     OBS_PES_ALL = obs_pes_all
# #     OBS_TS_ALL = obs_ts_all
# #     MPMT_SLOTS_ALL = mpmt_slots_all
# #     MPMT_TYPE_CODES_ALL = [get_mpmt_slot_type_codes(slots) for slots in MPMT_SLOTS_ALL]
# #     tot_events = len(OBS_PES_ALL)

# #     print("Computing truth FCNs...")
# #     true_fcn_all = [compute_true_fcn_for_event(i) for i in range(tot_events)]

# #     est_dict = {
# #         "metadata": {
# #             "fit_particle": FIT_PARTICLE_CANONICAL,
# #             "particle_mass_mev": FIT_PARTICLE_MASS_MEV,
# #             "particle_threshold_mev": FIT_PARTICLE_THRESHOLD_MEV,
# #             "beam_p": BEAM_P,
# #             "track_end_mode": TRACK_END_MODE,
# #             "fit_parameters": list(FIT_PARAMETER_NAMES),
# #             "truth_params": dict(TRUE_PARAMS),
# #             "geometry_file": GEOMETRY_FILE,
# #             "config_root_file": CONFIG_ROOT_FILE,
# #             "event_source": EVENT_SOURCE,
# #             "user_event_file": USER_EVENT_FILE if EVENT_SOURCE == "file" else None,
# #             "user_event_key": USER_EVENT_KEY if EVENT_SOURCE == "file" else None,
# #             "user_event_apply_peak_window": bool(USER_EVENT_APPLY_PEAK_WINDOW) if EVENT_SOURCE == "file" else None,
# #             "particle_selection_label": PARTICLE_SELECTION_LABEL,
# #             "selection_tof_ns": SELECTION_TOF_NS,
# #             "selection_tof_window_ns": SELECTION_TOF_WINDOW_NS,
# #             "selection_tof_field": SELECTION_TOF_FIELD,
# #             "selection_t5_particle_nr": SELECTION_T5_PARTICLE_NR,
# #             "range_table_max_full_range_mm": float(RANGE_LOOKUP.overall_distances_mm[-1]),
# #             "save_seed_scan": bool(SAVE_SEED_SCAN),
# #             "save_top_n_seeds": int(SAVE_TOP_N_SEEDS),
# #             "save_attempt_results": bool(SAVE_ATTEMPT_RESULTS),
# #         },
# #         "minimum_found": [],
# #         "x": [],
# #         "y": [],
# #         "z": [],
# #         "visible_length": [],
# #         "full_range": [],
# #         "ke0": [],
# #         "length": [],  # legacy alias for visible_length
# #         "t": [],
# #         "est_fcn": [],
# #         "true_fcn": [],
# #         "cx": [],
# #         "cy": [],
# #         "n_attempts": [],
# #         "chosen_seed_idx": [],
# #         "chosen_seed_fcn": [],
# #         "chosen_seed_params": [],
# #         "adaptive_rescue_used": [],
# #         "length_profile_rescue_considered": [],
# #         "length_profile_rescue_used": [],
# #         "fcn_retry_used": [],
# #         "edm": [],
# #     }
# #     if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
# #         est_dict["seed_scan"] = []

# #     if SAVE_ATTEMPT_RESULTS:
# #         est_dict["attempt_results"] = []

# #     n_events_per_batch = min(N_EVENTS_PER_BATCH, max(1, tot_events))

# #     for batch_start in range(0, tot_events, n_events_per_batch):
# #         batch_end = min(batch_start + n_events_per_batch, tot_events)
# #         event_indices = list(range(batch_start, batch_end))

# #         print(f"Starting event number {batch_start}")

# #         results = run_batch(
# #             event_indices=event_indices,
# #             init_param_sets=init_param_sets,
# #             nproc=NPROC,
# #             fcn_threshold=FCN_RETRY_THRESHOLD,
# #             max_attempts=MAX_FIT_ATTEMPTS,
# #             ncall=NCALL_MIGRAD,
# #         )

# #         for local_i, result in enumerate(results):
# #             event_index = event_indices[local_i]
# #             vals = result["values"]

# #             if IS_ABSORPTION_MODE:
# #                 visible_length = float(vals["visible_length"])
# #                 full_range = float(vals["full_range"])
# #             else:
# #                 visible_length = float(vals["length"])
# #                 full_range = visible_length
# #             ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))

# #             est_dict["minimum_found"].append(int(result["valid"]))
# #             est_dict["x"].append(vals["x0"])
# #             est_dict["y"].append(vals["y0"])
# #             est_dict["z"].append(vals["z0"])
# #             est_dict["visible_length"].append(visible_length)
# #             est_dict["full_range"].append(full_range)
# #             est_dict["ke0"].append(ke0)
# #             est_dict["length"].append(visible_length)
# #             est_dict["t"].append(vals["t0"])
# #             est_dict["cx"].append(vals["cx"])
# #             est_dict["cy"].append(vals["cy"])
# #             est_dict["est_fcn"].append(result["fval"])
# #             est_dict["true_fcn"].append(true_fcn_all[event_index])
# #             est_dict["n_attempts"].append(result["attempts"])
# #             est_dict["chosen_seed_idx"].append(result["chosen_seed_index"])
# #             est_dict["chosen_seed_fcn"].append(result["chosen_seed_fcn"])
# #             est_dict["chosen_seed_params"].append(result["chosen_seed_params"])
# #             if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
# #                 est_dict["seed_scan"].append(result.get("seed_scan", []))
# #             est_dict["adaptive_rescue_used"].append(result.get("adaptive_rescue_used", False))
# #             est_dict["length_profile_rescue_considered"].append(result.get("length_profile_rescue_considered", False))
# #             est_dict["length_profile_rescue_used"].append(result.get("length_profile_rescue_used", False))
# #             est_dict["fcn_retry_used"].append(bool(result.get("fcn_retry_used", False) or result.get("attempts", 1) > 1))
# #             est_dict["edm"].append(result.get("edm", np.nan))
# #             if SAVE_ATTEMPT_RESULTS:
# #                 est_dict["attempt_results"].append(result.get("attempt_results", []))

# #     Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
# #     with open(OUTPUT_FILE, "wb") as f:
# #         pickle.dump(est_dict, f)

# #     print("Done.")
# #     print("Saved:", OUTPUT_FILE)


# # if __name__ == "__main__":
# #     main()












# # # # 7,8-parameter abrupt-endpoint batch driver.
# # # #
# # # # Fit parameters:
# # # #   x0, y0, z0, cx, cy, visible_length, full_range, t0
# # # #
# # # # Meaning:
# # # #   visible_length = actual visible primary-Cherenkov length before abrupt cutoff [mm]
# # # #   full_range     = dE/dx-only range to Cherenkov threshold if no abrupt interaction [mm]
# # # #   ke0            = inferred initial kinetic energy from full_range using particle range table [MeV]
# # # #
# # # # This driver assumes an abrupt endpoint model.  It does not use fixed_initial_KE
# # # # as a fit setting; fixed_initial_KE is overwritten inside every FCN call using
# # # # full_range -> ke0.

# # # """Toggleable multi-stage batch driver for the 8-parameter Minuit Cherenkov fit on WCTE/real-data-style events.

# # # This is the selected-event/input-array version of the driver.  It preserves:
# # #   - get_selected_events(RUN, N_EVENTS) event loading
# # #   - run configuration GOOD_WCTE_PMTS masking from the ROOT Configuration tree
# # #   - pe_scale=143
# # #   - estimated geometry placement "est"
# # #   - mPMT type/relative efficiency corrections when tables are available

# # # The 8 fitted parameters are:
# # #   x0, y0, z0, cx, cy, visible_length, full_range, t0
# # # """

# # # import os
# # # import sys
# # # import pickle
# # # import multiprocessing as mp
# # # from pathlib import Path

# # # import numpy as np
# # # import uproot
# # # from iminuit import Minuit

# # # # =============================================================================
# # # # SELF-CONTAINED PATH SETUP
# # # # =============================================================================
# # # SCRIPT_DIR = Path(__file__).resolve().parent
# # # PROJECT_ROOT = SCRIPT_DIR.parent
# # # LICKETYFIT_DIR = PROJECT_ROOT / "LicketyFit"
# # # TABLE_DIR = PROJECT_ROOT / "tables"
# # # OUTPUT_DIR = PROJECT_ROOT / "outputs"
# # # OUTPUT_DIR.mkdir(exist_ok=True)

# # # geometry_path = os.environ.get("GEOMETRY_PATH", "/eos/user/j/jrimmer/Geometry")
# # # GEOMETRY_FILE = os.environ.get(
# # #     "WCTE_GEOMETRY_FILE",
# # #     str(Path(geometry_path) / "examples" / "wcte_bldg157.geo"),
# # # )

# # # for _path in (str(LICKETYFIT_DIR), str(PROJECT_ROOT), str(SCRIPT_DIR), str(TABLE_DIR), geometry_path):
# # #     if _path not in sys.path:
# # #         sys.path.insert(0, _path)

# # # # Force local tables before importing lookup/collapse helpers.
# # # os.environ["LF_TABLE_DIR"] = str(TABLE_DIR)
# # # os.environ["LF_MULTIPARTICLES_TABLE_DIR"] = str(TABLE_DIR)

# # # from Geometry.Device import Device
# # # from LicketyFit.Event import Event
# # # from LicketyFit.PMT import PMT
# # # from LicketyFit.Emitter import Emitter
# # # from particle_cherenkov_model import (
# # #     get_energy_distance_tables,
# # #     set_active_particle,
# # #     canonical_particle_name,
# # #     particle_mass_mev,
# # #     cherenkov_threshold_kinetic_mev,
# # # )
# # # try:
# # #     from event_loader import get_selected_events
# # # except Exception:
# # #     get_selected_events = None
# # # from particle_range_lookup import ParticleRangeLookup


# # # # =============================================================================
# # # # ENV HELPERS
# # # # =============================================================================
# # # def _env_float(name, default=None):
# # #     raw = os.environ.get(name)
# # #     if raw is None or str(raw).strip() == "":
# # #         return default
# # #     return float(raw)


# # # def _env_int(name, default):
# # #     raw = os.environ.get(name)
# # #     if raw is None or str(raw).strip() == "":
# # #         return int(default)
# # #     return int(raw)


# # # def _env_bool(name, default=False):
# # #     raw = os.environ.get(name)
# # #     if raw is None:
# # #         return bool(default)
# # #     return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


# # # def _parse_float_list_env(name, default):
# # #     raw = os.environ.get(name)
# # #     if raw is None or str(raw).strip() == "":
# # #         return list(default)
# # #     return [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]


# # # # =============================================================================
# # # # TOP-LEVEL CONFIGURATION
# # # # =============================================================================
# # # N_EVENTS_PER_BATCH = _env_int("N_EVENTS_PER_BATCH", 100)
# # # NPROC = _env_int("NPROC", 16)
# # # M_STRAT = _env_int("M_STRAT", 1)

# # # Z_SEED_EPS = 20.0
# # # VISIBLE_LENGTH_SEED_EPS = 40.0
# # # FULL_RANGE_SEED_EPS = 80.0
# # # T_MIN = -8.0

# # # FCN_RETRY_THRESHOLD = 1100.0
# # # VISIBLE_LENGTH_RETRY_THRESHOLD = _env_float("VISIBLE_LENGTH_RETRY_THRESHOLD", 2700.0)
# # # MAX_FIT_ATTEMPTS = 4
# # # NCALL_MIGRAD = _env_int("NCALL_MIGRAD", 70000)
# # # NCALL_SIMPLEX = _env_int("NCALL_SIMPLEX", NCALL_MIGRAD)

# # # RUN = _env_int("RUN", 2079)
# # # BEAM_P = _env_float("BEAM_P", 430)
# # # #N_EVENTS = _env_int("N_EVENTS", 7000)
# # # N_EVENTS = 50000
# # # # =============================================================================
# # # # PARTICLE HYPOTHESIS / 8-PARAMETER MODE
# # # # =============================================================================
# # # FIT_PARTICLE = os.environ.get("FIT_PARTICLE", "muon")
# # # FIT_PARTICLE_CANONICAL = canonical_particle_name(FIT_PARTICLE)
# # # FIT_PARTICLE_MASS_MEV = particle_mass_mev(FIT_PARTICLE_CANONICAL)
# # # FIT_PARTICLE_THRESHOLD_MEV = cherenkov_threshold_kinetic_mev(FIT_PARTICLE_MASS_MEV)
# # # set_active_particle(FIT_PARTICLE_CANONICAL)

# # # # Fit mode:
# # # #   full_length -> original 7-parameter fit:
# # # #                  x0, y0, z0, cx, cy, length, t0
# # # #                  length is the dE/dx range to Cherenkov threshold, so ke0 is inferred from length.
# # # #   absorption  -> 8-parameter abrupt-endpoint fit:
# # # #                  x0, y0, z0, cx, cy, visible_length, full_range, t0
# # # #                  visible_length is the abrupt cutoff; full_range determines ke0.
# # # _FIT_MODE_RAW = os.environ.get("FIT_MODE", os.environ.get("TRACK_END_MODE", "full_length")).strip().lower()

# # # if _FIT_MODE_RAW in {"absorption", "absorbed", "abrupt", "abrupt_8param", "interaction", "truncated"}:
# # #     FIT_MODE = "absorption"
# # # elif _FIT_MODE_RAW in {"full_length", "full-length", "full", "threshold", "range", "csda", "old", "7param", "7_parameter"}:
# # #     FIT_MODE = "full_length"
# # # else:
# # #     raise ValueError("FIT_MODE/TRACK_END_MODE must be 'full_length' or 'absorption'")

# # # TRACK_END_MODE = FIT_MODE
# # # IS_ABSORPTION_MODE = FIT_MODE == "absorption"
# # # IS_FULL_LENGTH_MODE = FIT_MODE == "full_length"
# # # EMITTER_TRACK_END_MODE = "abrupt" if IS_ABSORPTION_MODE else "threshold"
# # # FIT_PARAMETER_NAMES = (
# # #     ("x0", "y0", "z0", "cx", "cy", "visible_length", "full_range", "t0")
# # #     if IS_ABSORPTION_MODE
# # #     else ("x0", "y0", "z0", "cx", "cy", "length", "t0")
# # # )

# # # # Output/debug controls.  The full seed scan is large in the 8-parameter
# # # # fit because every event can have thousands of seed dictionaries.  Keep these
# # # # off for production output; enable only when debugging seed selection.
# # # SAVE_ATTEMPT_RESULTS = _env_bool("SAVE_ATTEMPT_RESULTS", False)
# # # SAVE_SEED_SCAN = _env_bool("SAVE_SEED_SCAN", False)
# # # SAVE_TOP_N_SEEDS = _env_int("SAVE_TOP_N_SEEDS", 0)

# # # # Likelihood toggles.
# # # USE_CHARGE_LIKELIHOOD = _env_bool("USE_CHARGE_LIKELIHOOD", True)
# # # USE_TIMING_LIKELIHOOD = _env_bool("USE_TIMING_LIKELIHOOD", True)
# # # USE_T0_PRIOR = _env_bool("USE_T0_PRIOR", False)

# # # if (not USE_CHARGE_LIKELIHOOD) and (not USE_TIMING_LIKELIHOOD):
# # #     raise ValueError("At least one likelihood term must be enabled.")

# # # if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
# # #     LIKELIHOOD_MODE = "charge_time"
# # # elif USE_CHARGE_LIKELIHOOD:
# # #     LIKELIHOOD_MODE = "charge_only"
# # # else:
# # #     LIKELIHOOD_MODE = "timing_only"

# # # OUTPUT_FILE = os.environ.get(
# # #     "LF_OUTPUT_FILE",
# # #     str(OUTPUT_DIR / f"estimates_run{RUN}_{BEAM_P:g}p_{FIT_PARTICLE_CANONICAL}_{TRACK_END_MODE}_mpmtEff_{LIKELIHOOD_MODE}.dict"),
# # # )

# # # RING_MASK_MODE = os.environ.get("RING_MASK_MODE", "both").strip().lower()

# # # # Data configuration.
# # # CONFIG_ROOT_FILE = os.environ.get(
# # #     "CONFIG_ROOT_FILE",
# # #     f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/{RUN}/WCTE_merged_production_R{RUN}.root",
# # # )

# # # # Event source.
# # # #   "selection" / "internal" : use event_loader.get_selected_events(...) with particle-specific TOF cuts.
# # # #   "file" / "user" / "custom" : load already-selected user events from USER_EVENT_FILE.
# # # #
# # # # USER_EVENT_FILE can be .npy, .npz, .pkl, or .pickle.  Supported payloads:
# # # #   - list/object array of event arrays, each with columns [pmt_id, charge, time] or
# # # #     [pmt_id, charge, time, event_number]
# # # #   - a single 2D array [pmt_id, charge, time] for one event
# # # #   - a single 2D array [pmt_id, charge, time, event_number], which is grouped by
# # # #     event_number
# # # EVENT_SOURCE = os.environ.get("EVENT_SOURCE", "selection").strip().lower()
# # # if EVENT_SOURCE in {"selected", "internal", "event_loader", "auto"}:
# # #     EVENT_SOURCE = "selection"
# # # elif EVENT_SOURCE in {"file", "user", "custom", "user_file", "provided"}:
# # #     EVENT_SOURCE = "file"
# # # if EVENT_SOURCE not in {"selection", "file"}:
# # #     raise ValueError("EVENT_SOURCE must be 'selection' or 'file'")

# # # USER_EVENT_FILE = os.environ.get("USER_EVENT_FILE", "").strip()
# # # USER_EVENT_KEY = os.environ.get("USER_EVENT_KEY", "").strip() or None
# # # USER_EVENT_APPLY_PEAK_WINDOW = _env_bool("USER_EVENT_APPLY_PEAK_WINDOW", True)

# # # # Event-selection configuration for event_loader.get_selected_events().
# # # # Defaults reproduce the historical muon-like WCTE selection.  For non-muon
# # # # beam selections, set PARTICLE_SELECTION_LABEL plus either SELECTION_TOF_NS
# # # # or SELECTION_TOF_FIELD/T5_PARTICLE_NR as needed for your production ROOT file.
# # # PARTICLE_SELECTION_LABEL = os.environ.get("PARTICLE_SELECTION_LABEL", FIT_PARTICLE_CANONICAL)
# # # SELECTION_TOF_NS = _env_float("SELECTION_TOF_NS", None)
# # # SELECTION_TOF_WINDOW_NS = _env_float("SELECTION_TOF_WINDOW_NS", 0.2)
# # # SELECTION_TOF_FIELD = os.environ.get("SELECTION_TOF_FIELD", "") or None
# # # SELECTION_MOMENTUM_FIELD = os.environ.get("SELECTION_MOMENTUM_FIELD", "") or None
# # # SELECTION_T5_PARTICLE_NR = _env_int("SELECTION_T5_PARTICLE_NR", 1)
# # # USE_PEAK_TIME_CUT = _env_bool("USE_PEAK_TIME_CUT", True)
# # # PEAK_WINDOW_NS = _env_float("PEAK_WINDOW_NS", 100.0)
# # # PEAK_BIN_WIDTH_NS = _env_float("PEAK_BIN_WIDTH_NS", 50.0)

# # # # =============================================================================
# # # # DETECTOR CONFIGURATION
# # # # =============================================================================
# # # DEFAULT_INACTIVE_SLOTS = [27, 32, 45, 74, 77, 79, 85, 91, 99, 9, 67]
# # # INACTIVE_SLOTS = [int(x) for x in os.environ.get(
# # #     "INACTIVE_SLOTS",
# # #     ",".join(str(x) for x in DEFAULT_INACTIVE_SLOTS),
# # # ).replace(";", ",").split(",") if x.strip()]
# # # INACTIVE_SLOTS_SET = set(int(s) for s in INACTIVE_SLOTS)

# # # OUTER_RING = np.array([0, 7, 19, 34, 50, 66, 82, 83, 105, 94, 95, 71, 72, 56, 40, 24, 11, 3, 18])
# # # INNER_RING = np.array([1, 8, 35, 51, 67, 84, 69, 70, 55, 39, 23, 10, 2, 20, 36, 52, 68, 53, 54, 38, 22, 21, 37, 9])
# # # OUTSIDE_RING = np.array([12, 13, 4, 5, 6, 17, 33, 49, 65, 81, 82, 104, 93, 86, 87, 72, 57, 41, 25])
# # # ALL_RING = np.arange(0, 106)

# # # # Optional truth diagnostics.  If not supplied, true_fcn is NaN.
# # # TRUE_LENGTH_MM = _env_float("TRUE_LENGTH_MM", None)
# # # TRUE_VISIBLE_LENGTH_MM = _env_float("TRUE_VISIBLE_LENGTH_MM", None)
# # # TRUE_FULL_RANGE_MM = _env_float("TRUE_FULL_RANGE_MM", None)
# # # TRUE_INITIAL_KE_MEV = _env_float("TRUE_INITIAL_KE_MEV", None)

# # # TRUE_PARAMS = {
# # #     "x0": _env_float("TRUE_X0", 0.0),
# # #     "y0": _env_float("TRUE_Y0", 0.0),
# # #     "z0": _env_float("TRUE_Z0", -1348.0),
# # #     "cx": _env_float("TRUE_CX", 0.0),
# # #     "cy": _env_float("TRUE_CY", 0.0),
# # #     "visible_length": np.nan,
# # #     "full_range": np.nan,
# # #     "t0": _env_float("TRUE_T0", 0.0),
# # # }

# # # # =============================================================================
# # # # GLOBAL FIT-SEARCH STAGE TOGGLES
# # # # =============================================================================
# # # ENABLE_STAGE1_SEED_GRID = _env_bool("ENABLE_STAGE1_SEED_GRID", True)
# # # ENABLE_STAGE2_MIGRAD_FIRST = _env_bool("ENABLE_STAGE2_MIGRAD_FIRST", False)
# # # ENABLE_STAGE3_ADAPTIVE_RESCUE = _env_bool("ENABLE_STAGE3_ADAPTIVE_RESCUE", False)
# # # ENABLE_STAGE4_LENGTH_PROFILE = _env_bool("ENABLE_STAGE4_LENGTH_PROFILE", False)

# # # # Keep the full seed ranking only when it is actually needed.  This avoids
# # # # sorting and returning thousands of seed dictionaries for normal production fits.
# # # NEED_FULL_SEED_SCAN = (
# # #     MAX_FIT_ATTEMPTS > 1
# # #     or ENABLE_STAGE3_ADAPTIVE_RESCUE
# # #     or ENABLE_STAGE4_LENGTH_PROFILE
# # #     or SAVE_SEED_SCAN
# # #     or SAVE_TOP_N_SEEDS > 0
# # # )

# # # # =============================================================================
# # # # INITIAL SEED CONFIGURATION
# # # # =============================================================================
# # # FAST_SEED_X0 = _parse_float_list_env("FAST_SEED_X0", [-150.0, -100, -50, 0.0, 50, 100, 150.0])
# # # FAST_SEED_Y0 = _parse_float_list_env("FAST_SEED_Y0", [-50.0, 0.0, 50.0])
# # # FAST_SEED_Z0 = _parse_float_list_env("FAST_SEED_Z0", [-1500.0, -1400.0, -1300.0, -1350, -1200.0, -1100.0, -1000.0])

# # # FAST_SEED_VISIBLE_LENGTHS = _parse_float_list_env(
# # #     "FAST_SEED_VISIBLE_LENGTHS",
# # #     [100.0, 200, 300.0, 400, 450, 500.0, 700.0, 900.0, 1100.0, 1300.0, 1400.0, 1500.0, 1700.0, 1900.0],
# # # )

# # # FAST_SEED_KE0_MEV = _parse_float_list_env(
# # #     "FAST_SEED_KE0_MEV",
# # #     [600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0],
# # # )
# # # FAST_SEED_FULL_RANGES_MM = _parse_float_list_env(
# # #     "FAST_SEED_FULL_RANGES_MM",
# # #     [300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0],
# # # )
# # # FAST_SEED_DIRECTIONS = [
# # #     (0.0, 0.0),
# # #     (0.04, 0.0),
# # #     (-0.04, 0.0),
# # #     (0.0, 0.04),
# # #     (0.0, -0.04),
# # # ]
# # # FAST_SEED_FULL_CARTESIAN = _env_bool("FAST_SEED_FULL_CARTESIAN", False)


# # # def build_sparse_geometry_variants():
# # #     variants = []
# # #     for x0 in FAST_SEED_X0:
# # #         variants.append({"x0": float(x0), "y0": 0.0, "cx": 0.0, "cy": 0.0})
# # #     for y0 in FAST_SEED_Y0:
# # #         variants.append({"x0": 0.0, "y0": float(y0), "cx": 0.0, "cy": 0.0})
# # #     for cx, cy in FAST_SEED_DIRECTIONS:
# # #         variants.append({"x0": 0.0, "y0": 0.0, "cx": float(cx), "cy": float(cy)})

# # #     unique = []
# # #     seen = set()
# # #     for v in variants:
# # #         sig = (float(v["x0"]), float(v["y0"]), float(v["cx"]), float(v["cy"]))
# # #         if sig not in seen:
# # #             seen.add(sig)
# # #             unique.append(v)
# # #     return unique


# # # FAST_SEED_GEOMETRY_VARIANTS = build_sparse_geometry_variants()


# # # def build_full_range_seed_values(range_lookup):
# # #     values = []
# # #     for ke0 in FAST_SEED_KE0_MEV:
# # #         if ke0 <= range_lookup.threshold_mev:
# # #             continue
# # #         r = range_lookup.energy_to_range_mm(float(ke0))
# # #         if np.isfinite(r) and r > 0:
# # #             values.append(float(r))

# # #     values.extend(float(r) for r in FAST_SEED_FULL_RANGES_MM)

# # #     max_r = float(range_lookup.overall_distances_mm[-1])
# # #     values = [r for r in values if np.isfinite(r) and 0.0 < r <= max_r]

# # #     unique = []
# # #     seen = set()
# # #     for r in values:
# # #         sig = round(float(r), 6)
# # #         if sig not in seen:
# # #             seen.add(sig)
# # #             unique.append(float(r))
# # #     return unique


# # # def build_fast_seed_grid(range_lookup):
# # #     seeds = []

# # #     if IS_FULL_LENGTH_MODE:
# # #         # 7-parameter original/full-length mode: scan only one length-like parameter.
# # #         if FAST_SEED_FULL_CARTESIAN:
# # #             for x0 in FAST_SEED_X0:
# # #                 for y0 in FAST_SEED_Y0:
# # #                     for z0 in FAST_SEED_Z0:
# # #                         for length in FAST_SEED_VISIBLE_LENGTHS:
# # #                             for cx, cy in FAST_SEED_DIRECTIONS:
# # #                                 seeds.append({
# # #                                     "x0": float(x0),
# # #                                     "y0": float(y0),
# # #                                     "z0": float(z0),
# # #                                     "cx": float(cx),
# # #                                     "cy": float(cy),
# # #                                     "length": float(length),
# # #                                     "t0": 0.0,
# # #                                 })
# # #         else:
# # #             for z0 in FAST_SEED_Z0:
# # #                 for length in FAST_SEED_VISIBLE_LENGTHS:
# # #                     for geom in FAST_SEED_GEOMETRY_VARIANTS:
# # #                         seeds.append({
# # #                             "x0": float(geom["x0"]),
# # #                             "y0": float(geom["y0"]),
# # #                             "z0": float(z0),
# # #                             "cx": float(geom["cx"]),
# # #                             "cy": float(geom["cy"]),
# # #                             "length": float(length),
# # #                             "t0": 0.0,
# # #                         })
# # #         keys = ("x0", "y0", "z0", "cx", "cy", "length", "t0")
# # #     else:
# # #         # 8-parameter absorption mode: scan visible cutoff length and full CSDA range separately.
# # #         full_range_seeds = build_full_range_seed_values(range_lookup)
# # #         if FAST_SEED_FULL_CARTESIAN:
# # #             for x0 in FAST_SEED_X0:
# # #                 for y0 in FAST_SEED_Y0:
# # #                     for z0 in FAST_SEED_Z0:
# # #                         for visible_length in FAST_SEED_VISIBLE_LENGTHS:
# # #                             for full_range in full_range_seeds:
# # #                                 if visible_length > full_range:
# # #                                     continue
# # #                                 for cx, cy in FAST_SEED_DIRECTIONS:
# # #                                     seeds.append({
# # #                                         "x0": float(x0),
# # #                                         "y0": float(y0),
# # #                                         "z0": float(z0),
# # #                                         "cx": float(cx),
# # #                                         "cy": float(cy),
# # #                                         "visible_length": float(visible_length),
# # #                                         "full_range": float(full_range),
# # #                                         "t0": 0.0,
# # #                                     })
# # #         else:
# # #             for z0 in FAST_SEED_Z0:
# # #                 for visible_length in FAST_SEED_VISIBLE_LENGTHS:
# # #                     for full_range in full_range_seeds:
# # #                         if visible_length > full_range:
# # #                             continue
# # #                         for geom in FAST_SEED_GEOMETRY_VARIANTS:
# # #                             seeds.append({
# # #                                 "x0": float(geom["x0"]),
# # #                                 "y0": float(geom["y0"]),
# # #                                 "z0": float(z0),
# # #                                 "cx": float(geom["cx"]),
# # #                                 "cy": float(geom["cy"]),
# # #                                 "visible_length": float(visible_length),
# # #                                 "full_range": float(full_range),
# # #                                 "t0": 0.0,
# # #                             })
# # #         keys = ("x0", "y0", "z0", "cx", "cy", "visible_length", "full_range", "t0")

# # #     unique = []
# # #     seen = set()
# # #     for seed in seeds:
# # #         sig = tuple(float(seed[k]) for k in keys)
# # #         if sig not in seen:
# # #             seen.add(sig)
# # #             unique.append(seed)
# # #     return unique


# # # PARAM_NAMES = FIT_PARAMETER_NAMES

# # # # =============================================================================
# # # # GLOBALS SHARED BY WORKERS
# # # # =============================================================================
# # # OVERALL_DISTANCES = None
# # # INIT_ENERGY_TABLE = None
# # # RANGE_LOOKUP = None

# # # WCD = None
# # # PMT_MODEL = None
# # # EMITTER_TEMPLATE = None
# # # P_LOCATIONS = None
# # # DIRECTION_ZS = None
# # # RING_KEEP_MASK = None
# # # CORR_POS = None
# # # MPMT_SLOTS_ALL = None

# # # OBS_PES_ALL = None
# # # OBS_TS_ALL = None
# # # GOOD_WCTE_PMTS_SET = None

# # # # =============================================================================
# # # # mPMT INFO / EFFICIENCY TABLES
# # # # =============================================================================
# # # other_mpmt_info_path = Path(os.environ.get("OTHER_MPMT_INFO_PATH", str(TABLE_DIR / "other_mpmt_info_v2.dict")))
# # # if other_mpmt_info_path.exists():
# # #     with open(other_mpmt_info_path, "rb") as f:
# # #         mpmt_info = pickle.load(f)
# # # else:
# # #     mpmt_info = {}

# # # rel_mpmt_eff_path = Path(os.environ.get("REL_MPMT_EFF_PATH", str(TABLE_DIR / "rel_mpmt_eff.dict")))
# # # if rel_mpmt_eff_path.exists():
# # #     with open(rel_mpmt_eff_path, "rb") as f:
# # #         rel_mpmt_eff = pickle.load(f)
# # # else:
# # #     unity = np.ones(200, dtype=np.float64)
# # #     rel_mpmt_eff = {
# # #         "tri_exsitu": unity,
# # #         "tri_insitu": unity,
# # #         "wut_insitu": unity,
# # #         "wut_exsitu": unity,
# # #     }

# # # tri_exsitu = rel_mpmt_eff["tri_exsitu"]
# # # tri_insitu = rel_mpmt_eff["tri_insitu"]
# # # wut_insitu = rel_mpmt_eff["wut_insitu"]
# # # wut_exsitu = rel_mpmt_eff["wut_exsitu"]


# # # def get_mpmt_slot_type(mpmt_slots):
# # #     slot_type = []
# # #     for slot in mpmt_slots:
# # #         slot = int(slot)
# # #         try:
# # #             if mpmt_info[slot]["mpmt_site"] == "TRI":
# # #                 if mpmt_info[slot]["mpmt_type"] == "In-situ":
# # #                     slot_type.append("tri_insitu")
# # #                 else:
# # #                     slot_type.append("tri_exsitu")
# # #             else:
# # #                 if mpmt_info[slot]["mpmt_type"] == "In-situ":
# # #                     slot_type.append("wut_insitu")
# # #                 else:
# # #                     slot_type.append("wut_exsitu")
# # #         except Exception:
# # #             slot_type.append("empty")
# # #     return slot_type


# # # # =============================================================================
# # # # CONFIG / TRUTH HELPERS
# # # # =============================================================================
# # # def load_good_wcte_pmts():
# # #     try:
# # #         with uproot.open(CONFIG_ROOT_FILE) as f:
# # #             t_c = f["Configuration"]
# # #             arr_config = t_c.arrays(library="ak")
# # #         good = np.asarray(arr_config["good_wcte_pmts"][0], dtype=int)
# # #         print("Loaded GOOD_WCTE_PMTS from:", CONFIG_ROOT_FILE)
# # #         return set(good.tolist())
# # #     except Exception as exc:
# # #         # For selected ROOT input, missing the run Configuration tree is usually a
# # #         # real problem.  For user-provided event files, allow a self-contained
# # #         # fallback by default and turn on every non-inactive PMT.
# # #         allow_missing_default = EVENT_SOURCE == "file"
# # #         if _env_bool("ALLOW_MISSING_GOOD_PMTS", allow_missing_default):
# # #             print("WARNING: could not load GOOD_WCTE_PMTS; using all non-inactive PMTs.")
# # #             print("Reason:", repr(exc))
# # #             all_ids = []
# # #             for slot in range(106):
# # #                 if slot in INACTIVE_SLOTS_SET:
# # #                     continue
# # #                 for pmt_pos in range(19):
# # #                     all_ids.append(slot * 100 + pmt_pos)
# # #             return set(all_ids)
# # #         raise


# # # def configure_truth_params():
# # #     if IS_ABSORPTION_MODE:
# # #         if TRUE_FULL_RANGE_MM is not None:
# # #             TRUE_PARAMS["full_range"] = float(TRUE_FULL_RANGE_MM)
# # #         elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
# # #             TRUE_PARAMS["full_range"] = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))

# # #         if TRUE_VISIBLE_LENGTH_MM is not None:
# # #             TRUE_PARAMS["visible_length"] = float(TRUE_VISIBLE_LENGTH_MM)
# # #         elif TRUE_LENGTH_MM is not None:
# # #             TRUE_PARAMS["visible_length"] = float(TRUE_LENGTH_MM)
# # #         else:
# # #             TRUE_PARAMS["visible_length"] = np.nan

# # #         if np.isfinite(TRUE_PARAMS["visible_length"]):
# # #             TRUE_PARAMS["length"] = TRUE_PARAMS["visible_length"]
# # #         return

# # #     # Full-length mode: the single fitted length is also the full CSDA range.
# # #     if TRUE_LENGTH_MM is not None:
# # #         length = float(TRUE_LENGTH_MM)
# # #     elif TRUE_FULL_RANGE_MM is not None:
# # #         length = float(TRUE_FULL_RANGE_MM)
# # #     elif TRUE_VISIBLE_LENGTH_MM is not None:
# # #         length = float(TRUE_VISIBLE_LENGTH_MM)
# # #     elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
# # #         length = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))
# # #     else:
# # #         length = np.nan

# # #     TRUE_PARAMS["length"] = length
# # #     TRUE_PARAMS["visible_length"] = length
# # #     TRUE_PARAMS["full_range"] = length


# # # # =============================================================================
# # # # EVENT / OBSERVABLE HELPERS
# # # # =============================================================================
# # # def sim_to_event(
# # #     sim_data,
# # #     WCD,
# # #     n_mpmt_total=106,
# # #     pe_scale=143,
# # #     shift_times=True,
# # #     n_earliest_for_t0=10,
# # # ):
# # #     vw = 223.0598645833333  # mm/ns

# # #     ev = Event(0, 0, n_mpmt_total)
# # #     ev.set_mpmt_status(list(range(n_mpmt_total)), False)

# # #     active_wcte_pmt_ids = []

# # #     for slot in range(n_mpmt_total):
# # #         if slot in INACTIVE_SLOTS_SET:
# # #             continue

# # #         slot_has_good_pmt = False
# # #         for pmt_pos_id in range(ev.npmt_per_mpmt):
# # #             wcte_pmt = int(slot * 100 + pmt_pos_id)
# # #             if wcte_pmt in GOOD_WCTE_PMTS_SET:
# # #                 ev.set_pmt_status(slot, [pmt_pos_id], True)
# # #                 slot_has_good_pmt = True
# # #                 active_wcte_pmt_ids.append(wcte_pmt)

# # #         if slot_has_good_pmt:
# # #             ev.set_mpmt_status([slot], True)

# # #     for i in range(len(sim_data[:, 0])):
# # #         wcte_pmt = int(sim_data[i, 0])
# # #         slot = int(wcte_pmt // 100)
# # #         pmt_pos_id = int(wcte_pmt % 100)

# # #         if slot < 0 or slot >= ev.n_mpmt:
# # #             continue
# # #         if pmt_pos_id < 0 or pmt_pos_id >= ev.npmt_per_mpmt:
# # #             continue
# # #         if not ev.mpmt_status[slot]:
# # #             continue
# # #         if not ev.pmt_status[slot][pmt_pos_id]:
# # #             continue

# # #         ev.hit_charges[slot][pmt_pos_id].append(float(sim_data[i, 1]))
# # #         ev.hit_times[slot][pmt_pos_id].append(float(sim_data[i, 2]))

# # #     if shift_times:
# # #         bp_loc = np.array([0.0, 0.0, -1350.0])
# # #         early_hits = []

# # #         for i_mpmt in range(ev.n_mpmt):
# # #             if not ev.mpmt_status[i_mpmt]:
# # #                 continue
# # #             for i_pmt in range(ev.npmt_per_mpmt):
# # #                 if not ev.pmt_status[i_mpmt][i_pmt]:
# # #                     continue
# # #                 if len(ev.hit_times[i_mpmt][i_pmt]) == 0:
# # #                     continue

# # #                 pmt_loc = WCD.mpmts[i_mpmt].pmts[i_pmt].get_placement("est")["location"]
# # #                 r = np.linalg.norm(pmt_loc - bp_loc)

# # #                 for t in ev.hit_times[i_mpmt][i_pmt]:
# # #                     early_hits.append({
# # #                         "time": float(t),
# # #                         "t0_est": float(t) - r / vw,
# # #                     })

# # #         if len(early_hits) > 0:
# # #             early_hits = sorted(early_hits, key=lambda x: x["time"])
# # #             n_use = min(n_earliest_for_t0, len(early_hits))
# # #             time_offset = np.median([hit["t0_est"] for hit in early_hits[:n_use]])

# # #             for i_mpmt in range(ev.n_mpmt):
# # #                 for i_pmt in range(ev.npmt_per_mpmt):
# # #                     ev.hit_times[i_mpmt][i_pmt] = [
# # #                         t - time_offset for t in ev.hit_times[i_mpmt][i_pmt]
# # #                     ]

# # #             ev.global_time_offset = time_offset

# # #     return ev, np.asarray(active_wcte_pmt_ids, dtype=int)


# # # def build_observables_from_event(ev, pe_scale=143):
# # #     obs_pes = []
# # #     obs_ts = []

# # #     for i_mpmt in range(ev.n_mpmt):
# # #         if not ev.mpmt_status[i_mpmt]:
# # #             continue
# # #         for i_pmt in range(ev.npmt_per_mpmt):
# # #             if not ev.pmt_status[i_mpmt][i_pmt]:
# # #                 continue

# # #             q = np.asarray(ev.hit_charges[i_mpmt][i_pmt], dtype=np.float64)
# # #             t = np.asarray(ev.hit_times[i_mpmt][i_pmt], dtype=np.float64)

# # #             if q.size == 0:
# # #                 obs_pes.append(0.0)
# # #                 obs_ts.append(np.nan)
# # #             else:
# # #                 obs_pes.append(float(np.sum(q)) / pe_scale)
# # #                 obs_ts.append(float(np.sum(q * t) / np.sum(q)))

# # #     return np.asarray(obs_pes, dtype=np.float64), np.asarray(obs_ts, dtype=np.float64)


# # # def apply_ring_mask_to_observables(obs_pes, obs_ts, ring_keep_mask, mode="both"):
# # #     obs_pes = obs_pes.copy()
# # #     obs_ts = obs_ts.copy()

# # #     if mode not in {"none", "pes", "ts", "both"}:
# # #         raise ValueError("RING_MASK_MODE must be one of: none, pes, ts, both")

# # #     if mode in {"pes", "both"}:
# # #         obs_pes[~ring_keep_mask] = 0.0
# # #     if mode in {"ts", "both"}:
# # #         obs_ts[~ring_keep_mask] = np.nan

# # #     return obs_pes, obs_ts


# # # def get_t0_prior_sigma(obs_pes, obs_ts):
# # #     n_timed = np.count_nonzero(np.isfinite(obs_ts))
# # #     total_pe = np.sum(obs_pes)

# # #     if (n_timed < 250) or (total_pe < 300):
# # #         return 0.1
# # #     elif (n_timed < 275) or (total_pe < 350):
# # #         return 0.2
# # #     elif (n_timed < 300) or (total_pe < 400):
# # #         return 0.3
# # #     elif (n_timed < 325) or (total_pe < 450):
# # #         return 0.4
# # #     elif (n_timed < 350) or (total_pe < 500):
# # #         return 0.5
# # #     elif (n_timed < 375) or (total_pe < 550):
# # #         return 0.6
# # #     elif (n_timed < 400) or (total_pe < 600):
# # #         return 0.7
# # #     elif (n_timed < 425) or (total_pe < 650):
# # #         return 0.8
# # #     elif (n_timed < 450) or (total_pe < 700):
# # #         return 1.0
# # #     elif (n_timed < 475) or (total_pe < 750):
# # #         return 1.2
# # #     elif (n_timed < 500) or (total_pe < 800):
# # #         return 1.4
# # #     elif (n_timed < 525) or (total_pe < 850):
# # #         return 1.6
# # #     elif (n_timed < 550) or (total_pe < 900):
# # #         return 1.8
# # #     else:
# # #         return 2.0


# # # # =============================================================================
# # # # LIKELIHOOD EVALUATION
# # # # =============================================================================
# # # def get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts):
# # #     exp_pes = np.asarray(exp_pes, dtype=np.float64)
# # #     obs_pes = np.asarray(obs_pes, dtype=np.float64)
# # #     exp_ts = np.asarray(exp_ts, dtype=np.float64)
# # #     obs_ts = np.asarray(obs_ts, dtype=np.float64)

# # #     mask = (
# # #         (exp_pes > 0.0)
# # #         & (obs_pes > 0.0)
# # #         & np.isfinite(exp_ts)
# # #         & np.isfinite(obs_ts)
# # #     )

# # #     if not np.any(mask):
# # #         return 1e30

# # #     sigma_t = PMT_MODEL.single_pe_time_std / np.sqrt(obs_pes[mask])
# # #     dt = (obs_ts[mask] - exp_ts[mask]) / sigma_t
# # #     return float(0.5 * np.sum(dt * dt))


# # # def evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts):
# # #     if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
# # #         return PMT_MODEL.get_neg_log_likelihood_npe_t(exp_pes, obs_pes, exp_ts, obs_ts)
# # #     if USE_CHARGE_LIKELIHOOD:
# # #         return PMT_MODEL.get_neg_log_likelihood_npe(exp_pes, obs_pes)
# # #     return get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts)


# # # def evaluate_neg_log_likelihood(
# # #     obs_pes,
# # #     obs_ts,
# # #     emitter,
# # #     mpmt_types,
# # #     x0,
# # #     y0,
# # #     z0,
# # #     cx,
# # #     cy,
# # #     length_or_visible,
# # #     full_range_or_t0,
# # #     t0=None,
# # # ):
# # #     """Evaluate the selected fit mode.

# # #     full_length mode receives: length_or_visible=length, full_range_or_t0=t0.
# # #     absorption mode receives:  length_or_visible=visible_length, full_range_or_t0=full_range, t0=t0.
# # #     """
# # #     if IS_ABSORPTION_MODE:
# # #         visible_length = float(length_or_visible)
# # #         full_range = float(full_range_or_t0)
# # #         t0 = float(t0)

# # #         if not np.isfinite(visible_length) or not np.isfinite(full_range):
# # #             return 1e30
# # #         if visible_length < 0.0 or full_range <= 0.0:
# # #             return 1e30
# # #         if visible_length > full_range:
# # #             return 1e30
# # #         if full_range > float(RANGE_LOOKUP.overall_distances_mm[-1]):
# # #             return 1e30

# # #         ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))
# # #         if (not np.isfinite(ke0)) or ke0 <= FIT_PARTICLE_THRESHOLD_MEV:
# # #             return 1e30
# # #         emitter.fixed_initial_KE = ke0
# # #         track_length_for_emission = visible_length
# # #     else:
# # #         length = float(length_or_visible)
# # #         t0 = float(full_range_or_t0)
# # #         if not np.isfinite(length) or length < 0.0:
# # #             return 1e30
# # #         if length > float(RANGE_LOOKUP.overall_distances_mm[-1]):
# # #             return 1e30
# # #         emitter.fixed_initial_KE = None
# # #         track_length_for_emission = length

# # #     cz2 = 1.0 - cx * cx - cy * cy
# # #     if cz2 <= 0.0:
# # #         return 1e30

# # #     cz = np.sqrt(cz2)
# # #     emitter.start_coord = (float(x0), float(y0), float(z0))
# # #     emitter.starting_time = float(t0)
# # #     emitter.direction = (float(cx), float(cy), float(cz))

# # #     init_ke = emitter.refresh_kinematics_from_length(track_length_for_emission)

# # #     if hasattr(emitter, "visible_length_is_physical"):
# # #         if not emitter.visible_length_is_physical():
# # #             return 1e30
# # #     elif getattr(emitter, "last_visible_length_exceeds_range", False):
# # #         return 1e30

# # #     s = emitter.get_emission_points(P_LOCATIONS, init_ke)
# # #     exp_pes, exp_ts = emitter.get_expected_pes_ts(
# # #         WCD,
# # #         s,
# # #         P_LOCATIONS,
# # #         DIRECTION_ZS,
# # #         mpmt_types,
# # #         obs_pes,
# # #     )

# # #     nll = evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts)
# # #     if not np.isfinite(nll):
# # #         return 1e30

# # #     if USE_TIMING_LIKELIHOOD and USE_T0_PRIOR:
# # #         sigma_t0 = get_t0_prior_sigma(obs_pes, obs_ts)
# # #         nll += abs(0.5 * (float(t0) / sigma_t0) ** 2)

# # #     return float(nll)


# # # def _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed):
# # #     if IS_ABSORPTION_MODE:
# # #         return evaluate_neg_log_likelihood(
# # #             obs_pes, obs_ts, emitter, mpmt_types,
# # #             seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
# # #             seed["visible_length"], seed["full_range"], seed["t0"],
# # #         )
# # #     return evaluate_neg_log_likelihood(
# # #         obs_pes, obs_ts, emitter, mpmt_types,
# # #         seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
# # #         seed["length"], seed["t0"],
# # #     )


# # # def select_best_initial_seed(obs_pes, obs_ts, init_param_sets, mpmt_types=None):
# # #     """
# # #     Cheap deterministic seed prescan.

# # #     For production fits, only the best seed is retained.  This is faster and
# # #     avoids moving a huge seed-scan list between multiprocessing workers.  When
# # #     rescue/debug output is enabled, the full sorted scan is kept.
# # #     """
# # #     best_info = None
# # #     seed_scan = [] if NEED_FULL_SEED_SCAN else None

# # #     for i, seed in enumerate(init_param_sets):
# # #         emitter = EMITTER_TEMPLATE.copy()

# # #         fval = _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed)

# # #         if not np.isfinite(fval):
# # #             fval = np.inf

# # #         info = {
# # #             "seed_index": int(i),
# # #             "fval": float(fval),
# # #             "params": dict(seed),
# # #         }

# # #         if seed_scan is not None:
# # #             seed_scan.append(info)

# # #         if best_info is None or fval < best_info["fval"]:
# # #             best_info = info

# # #     if best_info is None or not np.isfinite(best_info["fval"]):
# # #         raise RuntimeError("All seed FCNs were non-finite. Check ranges, PMT ordering, and event selection.")

# # #     if seed_scan is not None:
# # #         seed_scan_sorted = sorted(seed_scan, key=lambda x: x["fval"])
# # #     else:
# # #         seed_scan_sorted = [best_info]

# # #     best = seed_scan_sorted[0]
# # #     return dict(best["params"]), int(best["seed_index"]), float(best["fval"]), seed_scan_sorted


# # # def compute_true_fcn_for_event(event_index):
# # #     if IS_ABSORPTION_MODE:
# # #         if not np.isfinite(TRUE_PARAMS.get("visible_length", np.nan)):
# # #             return np.nan
# # #         if not np.isfinite(TRUE_PARAMS.get("full_range", np.nan)):
# # #             return np.nan
# # #         length_args = (TRUE_PARAMS["visible_length"], TRUE_PARAMS["full_range"], TRUE_PARAMS["t0"])
# # #     else:
# # #         if not np.isfinite(TRUE_PARAMS.get("length", np.nan)):
# # #             return np.nan
# # #         length_args = (TRUE_PARAMS["length"], TRUE_PARAMS["t0"])

# # #     mpmt_types = None
# # #     emitter = EMITTER_TEMPLATE.copy()
# # #     return evaluate_neg_log_likelihood(
# # #         OBS_PES_ALL[event_index],
# # #         OBS_TS_ALL[event_index],
# # #         emitter,
# # #         mpmt_types,
# # #         TRUE_PARAMS["x0"],
# # #         TRUE_PARAMS["y0"],
# # #         TRUE_PARAMS["z0"],
# # #         TRUE_PARAMS["cx"],
# # #         TRUE_PARAMS["cy"],
# # #         *length_args,
# # #     )


# # # # =============================================================================
# # # # MINUIT HELPERS
# # # # =============================================================================
# # # def make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types=None):
# # #     emitter = EMITTER_TEMPLATE.copy()

# # #     if IS_ABSORPTION_MODE:
# # #         def nll(x0, y0, z0, cx, cy, visible_length, full_range, t0):
# # #             return evaluate_neg_log_likelihood(
# # #                 obs_pes, obs_ts, emitter, mpmt_types,
# # #                 x0, y0, z0, cx, cy, visible_length, full_range, t0,
# # #             )
# # #     else:
# # #         def nll(x0, y0, z0, cx, cy, length, t0):
# # #             return evaluate_neg_log_likelihood(
# # #                 obs_pes, obs_ts, emitter, mpmt_types,
# # #                 x0, y0, z0, cx, cy, length, t0,
# # #             )

# # #     m = Minuit(nll, **start_params)

# # #     max_range = float(RANGE_LOOKUP.overall_distances_mm[-1])

# # #     m.limits["x0"] = (-2000, 2000)
# # #     m.limits["y0"] = (-2000, 2000)
# # #     m.limits["z0"] = (-2000, 2000)
# # #     m.limits["cx"] = (-0.5, 0.5)
# # #     m.limits["cy"] = (-0.5, 0.5)
# # #     m.limits["t0"] = (-8.0,8.0) #T0_LIMITS

# # #     m.errors["x0"] = 100.0
# # #     m.errors["y0"] = 100.0
# # #     m.errors["z0"] = 100.0
# # #     m.errors["cx"] = 0.01
# # #     m.errors["cy"] = 0.01
# # #     m.errors["t0"] = 0.1

# # #     if IS_ABSORPTION_MODE:
# # #         m.limits["visible_length"] = (0.0, 5000)
# # #         m.limits["full_range"] = (1.0, 5000)
# # #         m.errors["visible_length"] = 60.0
# # #         m.errors["full_range"] = 100.0
# # #     else:
# # #         m.limits["length"] = (0.0, 3000)
# # #         m.errors["length"] = 100.0

# # #     if not USE_TIMING_LIKELIHOOD:
# # #         m.fixed["t0"] = True

# # #     m.errordef = Minuit.LIKELIHOOD
# # #     m.strategy = M_STRAT

# # #     return m


# # # def is_bad_minuit_result(m, *, edm_max=1e10):
# # #     if (m.fval is None) or (not np.isfinite(m.fval)):
# # #         return True
# # #     # Do not use m.valid as a bad-result criterion.
# # #     try:
# # #         if (m.fmin is not None) and np.isfinite(m.fmin.edm) and (m.fmin.edm > edm_max):
# # #             return True
# # #     except Exception:
# # #         pass
# # #     return False


# # # def run_minuit_attempt(m, ncall):
# # #     if not ENABLE_STAGE2_MIGRAD_FIRST:
# # #         m.strategy = M_STRAT
# # #         m.simplex(ncall=ncall)
# # #         m.migrad(ncall=ncall)
# # #         return m

# # #     ncall_fast = max(2000, int(0.35 * ncall))
# # #     ncall_simplex = max(2000, int(0.25 * ncall))

# # #     m.strategy = 0
# # #     m.migrad(ncall=ncall_fast)

# # #     if is_bad_minuit_result(m):
# # #         m.simplex(ncall=ncall_simplex)
# # #         m.strategy = M_STRAT
# # #         m.migrad(ncall=ncall)

# # #     return m


# # # # =============================================================================
# # # # ADAPTIVE RESCUE
# # # # =============================================================================
# # # ENABLE_ADAPTIVE_RESCUE = ENABLE_STAGE3_ADAPTIVE_RESCUE
# # # RESCUE_MAX_SEEDS = 6
# # # RESCUE_LENGTH_BINS = [
# # #     (0.0, 1000.0),
# # #     (1000.0, 1250.0),
# # #     (1250.0, 1400.0),
# # #     (1400.0, 1700.0),
# # #     (1700.0, 3000.0),
# # # ]


# # # def result_length_value(values):
# # #     return float(values["visible_length"] if IS_ABSORPTION_MODE else values["length"])


# # # def result_full_range_value(values):
# # #     return float(values["full_range"] if IS_ABSORPTION_MODE else values["length"])


# # # def seed_length_value(params):
# # #     return float(params["visible_length"] if IS_ABSORPTION_MODE else params["length"])


# # # def result_ke0_from_values(values):
# # #     try:
# # #         return float(RANGE_LOOKUP.range_mm_to_energy(result_full_range_value(values)))
# # #     except Exception:
# # #         return np.nan


# # # def needs_rescue_result(result, fcn_threshold=None):
# # #     """
# # #     Decide whether a fit result should be retried.

# # #     This function is used by the default FCN retry logic and by the optional
# # #     rescue stages. A retry is triggered if the FCN is non-finite, if it is
# # #     above fcn_threshold, or if one of the existing physical/seed-stuck checks
# # #     fails.
# # #     """
# # #     if result is None:
# # #         return True

# # #     fval = float(result.get("fval", np.inf))
# # #     if not np.isfinite(fval):
# # #         return True

# # #     # Default FCN retry condition.
# # #     if fcn_threshold is not None and fval > float(fcn_threshold):
# # #         return True

# # #     values = result.get("values", {})
# # #     try:
# # #         fitted_length = result_length_value(values)
# # #         fitted_full = result_full_range_value(values)
# # #     except Exception:
# # #         return True

# # #     if (not np.isfinite(fitted_length)) or (not np.isfinite(fitted_full)):
# # #         return True
# # #     if fitted_length <= 10.0 or fitted_length >= VISIBLE_LENGTH_RETRY_THRESHOLD:
# # #         return True
# # #     if IS_ABSORPTION_MODE and fitted_length > fitted_full:
# # #         return True
# # #     if result.get("seed_stuck", False):
# # #         return True
# # #     if USE_TIMING_LIKELIHOOD and result.get("below_t_min", False):
# # #         return True
# # #     return False


# # # def choose_diverse_rescue_seed_infos(seed_scan_sorted, already_tried_seed_indices=None, max_total=RESCUE_MAX_SEEDS):
# # #     already = set() if already_tried_seed_indices is None else set(already_tried_seed_indices)
# # #     chosen = []

# # #     for lo, hi in RESCUE_LENGTH_BINS:
# # #         candidates = [
# # #             s for s in seed_scan_sorted
# # #             if int(s["seed_index"]) not in already
# # #             and lo <= seed_length_value(s["params"]) < hi
# # #         ]
# # #         if candidates:
# # #             chosen.append(candidates[0])
# # #             already.add(int(candidates[0]["seed_index"]))
# # #         if len(chosen) >= max_total:
# # #             return chosen

# # #     for s in seed_scan_sorted:
# # #         idx = int(s["seed_index"])
# # #         if idx in already:
# # #             continue
# # #         chosen.append(s)
# # #         already.add(idx)
# # #         if len(chosen) >= max_total:
# # #             break

# # #     return chosen


# # # def compact_seed_scan(seed_scan_sorted):
# # #     """Return the configured seed-scan payload for output/debugging."""
# # #     if SAVE_SEED_SCAN:
# # #         return seed_scan_sorted
# # #     if SAVE_TOP_N_SEEDS > 0:
# # #         return seed_scan_sorted[:SAVE_TOP_N_SEEDS]
# # #     return []


# # # def build_result_from_minuit(m, attempt, start_params, chosen_seed_idx, chosen_seed_fcn, seed_scan_sorted):
# # #     current_fval = float(m.fval) if (m.fval is not None and np.isfinite(m.fval)) else np.inf
# # #     current_values = m.values.to_dict()

# # #     fitted_z0 = float(current_values["z0"])
# # #     fitted_length = result_length_value(current_values)
# # #     fitted_full = result_full_range_value(current_values)
# # #     fitted_ke0 = result_ke0_from_values(current_values)

# # #     visible_too_large = fitted_length > VISIBLE_LENGTH_RETRY_THRESHOLD
# # #     z_near_seed = abs(fitted_z0 - float(start_params["z0"])) <= Z_SEED_EPS
# # #     length_near_seed = abs(fitted_length - seed_length_value(start_params)) <= VISIBLE_LENGTH_SEED_EPS
# # #     if IS_ABSORPTION_MODE:
# # #         full_near_seed = abs(fitted_full - float(start_params["full_range"])) <= FULL_RANGE_SEED_EPS
# # #     else:
# # #         full_near_seed = False
# # #     seed_stuck = z_near_seed and length_near_seed and (full_near_seed if IS_ABSORPTION_MODE else True)
# # #     below_t_min = USE_TIMING_LIKELIHOOD and (current_values["t0"] < T_MIN)

# # #     # Always expose consistent aliases in the result dictionary.
# # #     visible_length_mm = fitted_length
# # #     full_range_mm = fitted_full
# # #     current_values.setdefault("length", fitted_length)
# # #     current_values.setdefault("visible_length", visible_length_mm)
# # #     current_values.setdefault("full_range", full_range_mm)

# # #     return {
# # #         "values": current_values,
# # #         "errors": m.errors.to_dict(),
# # #         "fval": current_fval,
# # #         "valid": bool(m.valid),
# # #         "attempts": attempt,
# # #         "visible_length_too_large": bool(visible_too_large),
# # #         "length_too_large": bool(visible_too_large),
# # #         "seed_stuck": bool(seed_stuck),
# # #         "z_near_seed": bool(z_near_seed),
# # #         "visible_length_near_seed": bool(length_near_seed),
# # #         "full_range_near_seed": bool(full_near_seed),
# # #         "length_near_seed": bool(length_near_seed),
# # #         "below_t_min": bool(below_t_min),
# # #         "chosen_seed_index": int(chosen_seed_idx),
# # #         "chosen_seed_fcn": float(chosen_seed_fcn) if np.isfinite(chosen_seed_fcn) else np.nan,
# # #         "chosen_seed_params": dict(start_params),
# # #         "seed_scan": compact_seed_scan(seed_scan_sorted),
# # #         "visible_length_mm": visible_length_mm,
# # #         "full_range_mm": full_range_mm,
# # #         "length_mm": fitted_length,
# # #         "ke0_mev": fitted_ke0,
# # #         "edm": (
# # #             float(m.fmin.edm)
# # #             if (getattr(m, "fmin", None) is not None and m.fmin.edm is not None)
# # #             else np.nan
# # #         ),
# # #     }


# # # def result_sort_key(result):
# # #     if result is None:
# # #         return (999, np.inf)
# # #     fval = float(result.get("fval", np.inf))
# # #     penalty = 0
# # #     if not np.isfinite(fval):
# # #         penalty += 100
# # #     if result.get("visible_length_too_large", False) or result.get("length_too_large", False):
# # #         penalty += 10
# # #     if result.get("seed_stuck", False):
# # #         penalty += 5
# # #     if result.get("below_t_min", False):
# # #         penalty += 5
# # #     return (penalty, fval)


# # # # =============================================================================
# # # # HARD-EVENT VISIBLE-LENGTH PROFILE RESCUE
# # # # =============================================================================
# # # ENABLE_LENGTH_PROFILE_RESCUE = ENABLE_STAGE4_LENGTH_PROFILE
# # # LENGTH_PROFILE_GRID = list(FAST_SEED_VISIBLE_LENGTHS)
# # # LENGTH_PROFILE_MAX_POINTS = 6


# # # def run_length_profile_rescue(obs_pes, obs_ts, mpmt_types, seed_scan_sorted, ncall, starting_attempt_index=100):
# # #     profile_results = []
# # #     base_seed = dict(seed_scan_sorted[0]["params"])
# # #     length_key = "visible_length" if IS_ABSORPTION_MODE else "length"

# # #     for j, profile_length in enumerate(LENGTH_PROFILE_GRID[:LENGTH_PROFILE_MAX_POINTS]):
# # #         start_params = dict(base_seed)
# # #         start_params[length_key] = float(profile_length)
# # #         if IS_ABSORPTION_MODE and start_params["visible_length"] > start_params["full_range"]:
# # #             continue

# # #         m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# # #         m.fixed[length_key] = True
# # #         run_minuit_attempt(m, max(5000, int(0.5 * ncall)))

# # #         prof_result = build_result_from_minuit(
# # #             m,
# # #             attempt=starting_attempt_index + j,
# # #             start_params=start_params,
# # #             chosen_seed_idx=-1000 - j,
# # #             chosen_seed_fcn=np.nan,
# # #             seed_scan_sorted=seed_scan_sorted,
# # #         )
# # #         prof_result["profile_fixed_length"] = float(profile_length)
# # #         prof_result["profile_fixed_visible_length"] = float(profile_length)
# # #         profile_results.append(prof_result)

# # #     if not profile_results:
# # #         return None

# # #     best_profile = min(profile_results, key=result_sort_key)

# # #     polish_params = dict(best_profile["values"])
# # #     # Keep only parameters actually used by this mode; Minuit will reject extras.
# # #     polish_params = {k: polish_params[k] for k in PARAM_NAMES if k in polish_params}
# # #     m = make_minuit_for_event(obs_pes, obs_ts, polish_params, mpmt_types)
# # #     m.fixed[length_key] = False
# # #     run_minuit_attempt(m, ncall)

# # #     polish_result = build_result_from_minuit(
# # #         m,
# # #         attempt=starting_attempt_index + len(profile_results),
# # #         start_params=polish_params,
# # #         chosen_seed_idx=-2000,
# # #         chosen_seed_fcn=float(best_profile["fval"]),
# # #         seed_scan_sorted=seed_scan_sorted,
# # #     )
# # #     polish_result["length_profile_rescue_used"] = True
# # #     polish_result["length_profile_results"] = profile_results
# # #     polish_result["length_profile_best_fixed"] = best_profile
# # #     return polish_result


# # # def fit_one_event_by_index(args):
# # #     event_index, init_param_sets, fcn_threshold, max_attempts, ncall = args

# # #     obs_pes = OBS_PES_ALL[event_index]
# # #     obs_ts = OBS_TS_ALL[event_index]
# # #     mpmt_types = get_mpmt_slot_type(MPMT_SLOTS_ALL[event_index])

# # #     best_seed, best_seed_idx, best_seed_fval, seed_scan_sorted = select_best_initial_seed(
# # #         obs_pes,
# # #         obs_ts,
# # #         init_param_sets,
# # #         mpmt_types,
# # #     )

# # #     attempt_results = []
# # #     tried_seed_indices = set()

# # #     # -------------------------------------------------------------------------
# # #     # Default FCN retry logic.
# # #     #
# # #     # Attempt 1 uses the best seed from the deterministic prescan. If the final
# # #     # Minuit FCN is above FCN_RETRY_THRESHOLD, retry from additional seed points.
# # #     # This does NOT require ENABLE_STAGE3_ADAPTIVE_RESCUE.
# # #     #
# # #     # MAX_FIT_ATTEMPTS is the total number of Minuit fits, including attempt 1.
# # #     # For example, MAX_FIT_ATTEMPTS = 4 means one primary attempt plus up to
# # #     # three retries.
# # #     # -------------------------------------------------------------------------
# # #     max_attempts = int(max_attempts)
# # #     if max_attempts < 1:
# # #         max_attempts = 1

# # #     if len(seed_scan_sorted) == 0:
# # #         raise RuntimeError("seed_scan_sorted is empty. Cannot fit event.")

# # #     seeds_to_try = []

# # #     primary_info = seed_scan_sorted[0]
# # #     seeds_to_try.append(primary_info)
# # #     tried_seed_indices.add(int(primary_info["seed_index"]))

# # #     if max_attempts > 1:
# # #         retry_seed_infos = choose_diverse_rescue_seed_infos(
# # #             seed_scan_sorted,
# # #             already_tried_seed_indices=tried_seed_indices,
# # #             max_total=max_attempts - 1,
# # #         )
# # #         seeds_to_try.extend(retry_seed_infos)

# # #     for attempt_number, seed_info in enumerate(seeds_to_try, start=1):
# # #         if attempt_number > max_attempts:
# # #             break

# # #         start_params = dict(seed_info["params"])
# # #         chosen_seed_idx = int(seed_info["seed_index"])
# # #         chosen_seed_fcn = float(seed_info["fval"])
# # #         tried_seed_indices.add(chosen_seed_idx)

# # #         m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# # #         run_minuit_attempt(m, ncall)

# # #         result = build_result_from_minuit(
# # #             m,
# # #             attempt=attempt_number,
# # #             start_params=start_params,
# # #             chosen_seed_idx=chosen_seed_idx,
# # #             chosen_seed_fcn=chosen_seed_fcn,
# # #             seed_scan_sorted=seed_scan_sorted,
# # #         )
# # #         attempt_results.append(result)

# # #         # Stop as soon as an acceptable result is found. This includes the
# # #         # explicit FCN threshold check.
# # #         if not needs_rescue_result(result, fcn_threshold=fcn_threshold):
# # #             break

# # #     best_result = min(attempt_results, key=result_sort_key)

# # #     # -------------------------------------------------------------------------
# # #     # Optional adaptive rescue stage. This remains available, but it is no
# # #     # longer required for ordinary FCN-based retries. It can only use remaining
# # #     # attempts up to MAX_FIT_ATTEMPTS.
# # #     # -------------------------------------------------------------------------
# # #     if ENABLE_ADAPTIVE_RESCUE and needs_rescue_result(best_result, fcn_threshold=fcn_threshold):
# # #         remaining_attempts = max_attempts - len(attempt_results)

# # #         if remaining_attempts > 0:
# # #             rescue_seed_infos = choose_diverse_rescue_seed_infos(
# # #                 seed_scan_sorted,
# # #                 already_tried_seed_indices=tried_seed_indices,
# # #                 max_total=remaining_attempts,
# # #             )

# # #             for seed_info in rescue_seed_infos:
# # #                 if len(attempt_results) >= max_attempts:
# # #                     break

# # #                 attempt_number = len(attempt_results) + 1

# # #                 start_params = dict(seed_info["params"])
# # #                 chosen_seed_idx = int(seed_info["seed_index"])
# # #                 chosen_seed_fcn = float(seed_info["fval"])
# # #                 tried_seed_indices.add(chosen_seed_idx)

# # #                 m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# # #                 run_minuit_attempt(m, ncall)

# # #                 result = build_result_from_minuit(
# # #                     m,
# # #                     attempt=attempt_number,
# # #                     start_params=start_params,
# # #                     chosen_seed_idx=chosen_seed_idx,
# # #                     chosen_seed_fcn=chosen_seed_fcn,
# # #                     seed_scan_sorted=seed_scan_sorted,
# # #                 )
# # #                 attempt_results.append(result)

# # #                 if result_sort_key(result) < result_sort_key(best_result):
# # #                     best_result = result

# # #                 if not needs_rescue_result(result, fcn_threshold=fcn_threshold):
# # #                     break

# # #     best_result = min(attempt_results, key=result_sort_key)

# # #     if ENABLE_LENGTH_PROFILE_RESCUE and needs_rescue_result(best_result, fcn_threshold=fcn_threshold):
# # #         profile_result = run_length_profile_rescue(
# # #             obs_pes,
# # #             obs_ts,
# # #             mpmt_types,
# # #             seed_scan_sorted,
# # #             ncall,
# # #             starting_attempt_index=100 + len(attempt_results),
# # #         )
# # #         if profile_result is not None:
# # #             attempt_results.append(profile_result)
# # #             best_result = min(attempt_results, key=result_sort_key)

# # #     best_result["attempts"] = len(attempt_results)
# # #     if SAVE_ATTEMPT_RESULTS:
# # #         best_result["attempt_results"] = attempt_results
# # #     else:
# # #         best_result["attempt_results"] = []

# # #     # Kept for backward compatibility with existing output readers. In this
# # #     # version, this flag means any extra fit attempt was used, whether from the
# # #     # default FCN retry or from adaptive rescue.
# # #     best_result["adaptive_rescue_used"] = bool(len(attempt_results) > 1)
# # #     best_result["length_profile_rescue_considered"] = bool(ENABLE_LENGTH_PROFILE_RESCUE)
# # #     best_result["length_profile_rescue_used"] = bool(
# # #         best_result.get("length_profile_rescue_used", False)
# # #         or any(r.get("length_profile_rescue_used", False) for r in attempt_results)
# # #     )
# # #     return best_result


# # # def run_batch(event_indices, init_param_sets, nproc, fcn_threshold, max_attempts, ncall):
# # #     args = [(idx, init_param_sets, fcn_threshold, max_attempts, ncall) for idx in event_indices]

# # #     try:
# # #         ctx = mp.get_context("fork")
# # #     except ValueError:
# # #         ctx = mp.get_context()

# # #     with ctx.Pool(processes=nproc) as pool:
# # #         return pool.map(fit_one_event_by_index, args)


# # # # =============================================================================
# # # # USER-PROVIDED EVENT FILE HELPERS
# # # # =============================================================================
# # # def _coerce_event_array(event, *, event_label="event"):
# # #     arr = np.asarray(event)
# # #     if arr.ndim != 2 or arr.shape[1] < 3:
# # #         raise ValueError(
# # #             f"{event_label} must be a 2D array with at least 3 columns: "
# # #             "[pmt_id, charge, time]."
# # #         )
# # #     # Keep optional event-number columns, but the fitter only consumes columns 0:3.
# # #     return np.asarray(arr[:, :4] if arr.shape[1] >= 4 else arr[:, :3], dtype=np.float64)


# # # def _events_from_loaded_object(obj):
# # #     """Normalize npy/npz/pickle payloads into a list of event arrays."""
# # #     if isinstance(obj, dict):
# # #         if USER_EVENT_KEY is not None:
# # #             obj = obj[USER_EVENT_KEY]
# # #         elif "events" in obj:
# # #             obj = obj["events"]
# # #         elif "data" in obj:
# # #             obj = obj["data"]
# # #         elif "arr_0" in obj:
# # #             obj = obj["arr_0"]
# # #         else:
# # #             keys = ", ".join(map(str, obj.keys()))
# # #             raise KeyError(
# # #                 "Could not choose an event array from the dict payload. "
# # #                 f"Available keys: {keys}. Set USER_EVENT_KEY."
# # #             )

# # #     if isinstance(obj, np.lib.npyio.NpzFile):
# # #         if USER_EVENT_KEY is not None:
# # #             key = USER_EVENT_KEY
# # #         elif "events" in obj.files:
# # #             key = "events"
# # #         elif "data" in obj.files:
# # #             key = "data"
# # #         elif "arr_0" in obj.files:
# # #             key = "arr_0"
# # #         elif len(obj.files) == 1:
# # #             key = obj.files[0]
# # #         else:
# # #             raise KeyError(
# # #                 "Could not choose an event array from the npz payload. "
# # #                 f"Available keys: {obj.files}. Set USER_EVENT_KEY."
# # #             )
# # #         obj = obj[key]

# # #     if isinstance(obj, (list, tuple)):
# # #         return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(obj)]

# # #     arr = np.asarray(obj, dtype=object if getattr(obj, "dtype", None) == object else None)

# # #     # Object arrays are normally lists of variable-length events.
# # #     if arr.dtype == object and arr.ndim == 1:
# # #         return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(arr)]

# # #     # A 3D numeric array is N_events x N_hits x N_columns.
# # #     if arr.ndim == 3:
# # #         return [_coerce_event_array(arr[i], event_label=f"event[{i}]") for i in range(arr.shape[0])]

# # #     # A 2D array with a 4th column is interpreted as a concatenated event table
# # #     # grouped by event number.  A 2D array with only 3 columns is one event.
# # #     if arr.ndim == 2:
# # #         arr2 = np.asarray(arr, dtype=np.float64)
# # #         if arr2.shape[1] >= 4:
# # #             events = []
# # #             event_numbers = arr2[:, 3].astype(np.int64)
# # #             for evnum in np.unique(event_numbers):
# # #                 events.append(_coerce_event_array(arr2[event_numbers == evnum], event_label=f"event_number={evnum}"))
# # #             return events
# # #         return [_coerce_event_array(arr2, event_label="single_event")]

# # #     raise ValueError(
# # #         "Unsupported USER_EVENT_FILE payload. Expected a list/object array of "
# # #         "event arrays, a 3D event array, or a 2D [pmt_id, charge, time] table."
# # #     )


# # # def load_user_event_file(path, *, max_events=None):
# # #     """Load user-provided, already-selected events from npy/npz/pickle files."""
# # #     path = Path(path)
# # #     if not path.exists():
# # #         raise FileNotFoundError(f"USER_EVENT_FILE does not exist: {path}")

# # #     suffix = path.suffix.lower()
# # #     if suffix == ".npz":
# # #         loaded = np.load(path, allow_pickle=True)
# # #     elif suffix == ".npy":
# # #         loaded = np.load(path, allow_pickle=True)
# # #     elif suffix in {".pkl", ".pickle"}:
# # #         with open(path, "rb") as f:
# # #             loaded = pickle.load(f)
# # #     else:
# # #         raise ValueError(
# # #             f"Unsupported USER_EVENT_FILE suffix {suffix!r}. Use .npy, .npz, .pkl, or .pickle."
# # #         )

# # #     events = _events_from_loaded_object(loaded)
# # #     if max_events is not None:
# # #         events = events[: int(max_events)]
# # #     print(f"Loaded {len(events)} user-provided events from: {path}")
# # #     return events


# # # # =============================================================================
# # # # MAIN DRIVER
# # # # =============================================================================
# # # def main():
# # #     global OVERALL_DISTANCES, INIT_ENERGY_TABLE, RANGE_LOOKUP
# # #     global WCD, PMT_MODEL, EMITTER_TEMPLATE, P_LOCATIONS, DIRECTION_ZS, RING_KEEP_MASK, CORR_POS
# # #     global OBS_PES_ALL, OBS_TS_ALL, MPMT_SLOTS_ALL, GOOD_WCTE_PMTS_SET

# # #     print("Likelihood mode:", LIKELIHOOD_MODE)
# # #     print("Fit particle:", FIT_PARTICLE_CANONICAL)
# # #     print("Particle mass [MeV]:", FIT_PARTICLE_MASS_MEV)
# # #     print("Cherenkov threshold [MeV]:", FIT_PARTICLE_THRESHOLD_MEV)
# # #     print("Fit mode:", TRACK_END_MODE)
# # #     print("Fit parameters:", FIT_PARAMETER_NAMES)
# # #     print("Output file:", OUTPUT_FILE)

# # #     if EVENT_SOURCE == "selection" and get_selected_events is None:
# # #         raise ImportError(
# # #             "event_loader.py was not found. Copy it into LF_multiParticles/scripts "
# # #             "or add its directory to PYTHONPATH, or set EVENT_SOURCE=file and USER_EVENT_FILE."
# # #         )

# # #     if EVENT_SOURCE == "file" and not USER_EVENT_FILE:
# # #         raise ValueError("EVENT_SOURCE=file requires USER_EVENT_FILE=/path/to/events.npy|npz|pkl")

# # #     GOOD_WCTE_PMTS_SET = load_good_wcte_pmts()

# # #     RANGE_LOOKUP = ParticleRangeLookup(FIT_PARTICLE_CANONICAL, table_dirs=[str(TABLE_DIR)])
# # #     print("Range table max KE [MeV]:", float(RANGE_LOOKUP.initial_energies_mev[-1]))
# # #     print("Range table max full_range [mm]:", float(RANGE_LOOKUP.overall_distances_mm[-1]))

# # #     configure_truth_params()
# # #     if IS_ABSORPTION_MODE:
# # #         truth_ready = np.isfinite(TRUE_PARAMS["visible_length"]) and np.isfinite(TRUE_PARAMS["full_range"])
# # #         if truth_ready:
# # #             print("Truth visible length [mm]:", TRUE_PARAMS["visible_length"])
# # #             print("Truth full range [mm]:", TRUE_PARAMS["full_range"])
# # #             print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["full_range"]))
# # #         else:
# # #             print("Truth FCN disabled: set TRUE_VISIBLE_LENGTH_MM and TRUE_FULL_RANGE_MM/TRUE_INITIAL_KE_MEV.")
# # #     else:
# # #         truth_ready = np.isfinite(TRUE_PARAMS["length"])
# # #         if truth_ready:
# # #             print("Truth length [mm]:", TRUE_PARAMS["length"])
# # #             print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["length"]))
# # #         else:
# # #             print("Truth FCN disabled: set TRUE_LENGTH_MM or TRUE_INITIAL_KE_MEV.")

# # #     init_param_sets = build_fast_seed_grid(RANGE_LOOKUP)
# # #     if not init_param_sets:
# # #         raise RuntimeError("Seed grid is empty. Check FAST_SEED_VISIBLE_LENGTHS and FAST_SEED_KE0_MEV/FULL_RANGES.")
# # #     print("Number of initial seeds:", len(init_param_sets))

# # #     for i, seed in enumerate(init_param_sets):
# # #         missing = [k for k in PARAM_NAMES if k not in seed]
# # #         if missing:
# # #             raise ValueError(f"Seed {i} is missing keys: {missing}")

# # #     set_active_particle(FIT_PARTICLE_CANONICAL)
# # #     OVERALL_DISTANCES, INIT_ENERGY_TABLE, _distance_rows = get_energy_distance_tables(
# # #         FIT_PARTICLE_CANONICAL
# # #     )

# # #     hall = Device.open_file(GEOMETRY_FILE)
# # #     WCD = hall.wcds[0]

# # #     initial_ke_seed = float(RANGE_LOOKUP.range_mm_to_energy(
# # #         min(1000.0, float(RANGE_LOOKUP.overall_distances_mm[-1]))
# # #     ))

# # #     emitter_model = Emitter(
# # #         0.0,
# # #         (0.0, 0.0, 0.0),
# # #         (0.0, 0.0, 1.0),
# # #         0.96,
# # #         500.0,
# # #         18.0,
# # #         particle=FIT_PARTICLE_CANONICAL,
# # #         track_end_mode=EMITTER_TRACK_END_MODE,
# # #         fixed_initial_KE=initial_ke_seed if IS_ABSORPTION_MODE else None,
# # #     )

# # #     delta_pdf_path = TABLE_DIR / "delta_e_angular_pdf_table.npz"
# # #     if delta_pdf_path.exists() and hasattr(emitter_model, "load_delta_e_angular_pdf_table"):
# # #         emitter_model.load_delta_e_angular_pdf_table(str(delta_pdf_path))

# # #     PMT_MODEL = PMT(1.0, 0.3, 1.0, 40.0, 0.2, 0.0)
# # #     EMITTER_TEMPLATE = emitter_model.copy()
# # #     CORR_POS = None

# # #     print("Building event observables...")

# # #     obs_pes_all = []
# # #     obs_ts_all = []
# # #     mpmt_slots_all = []

# # #     if EVENT_SOURCE == "selection":
# # #         events = get_selected_events(
# # #             RUN,
# # #             N_EVENTS,
# # #             particle=PARTICLE_SELECTION_LABEL,
# # #             root_file=CONFIG_ROOT_FILE,
# # #             use_peak_time_cut=USE_PEAK_TIME_CUT,
# # #             peak_window=PEAK_WINDOW_NS,
# # #             peak_bin_width=PEAK_BIN_WIDTH_NS,
# # #             tof_primary=SELECTION_TOF_NS,
# # #             tof_window=SELECTION_TOF_WINDOW_NS,
# # #             tof_scalar_field=SELECTION_TOF_FIELD,
# # #             momentum_scalar_field=SELECTION_MOMENTUM_FIELD,
# # #             t5_particle_nr=SELECTION_T5_PARTICLE_NR,
# # #         )
# # #     else:
# # #         events = load_user_event_file(USER_EVENT_FILE, max_events=N_EVENTS)

# # #     tot_events = len(events)
# # #     print("Total Events to Fit:", tot_events)

# # #     for i in range(tot_events):
# # #         event = np.asarray(events[i])
# # #         if event.size == 0:
# # #             continue

# # #         apply_event_time_window = (EVENT_SOURCE == "selection") or USER_EVENT_APPLY_PEAK_WINDOW
# # #         if apply_event_time_window:
# # #             time_hist = np.histogram(event[:, 2], bins=np.arange(0, 4000))
# # #             max_idx = int(np.argmax(time_hist[0]))
# # #             lo_idx = max(0, max_idx - 20)
# # #             hi_idx = min(len(time_hist[1]) - 1, max_idx + 5)
# # #             min_time = time_hist[1][lo_idx]
# # #             cut_time = time_hist[1][hi_idx]
# # #             time_mask = (event[:, 2] > min_time) & (event[:, 2] < cut_time)
# # #             event = event[time_mask]

# # #         ev, pmt_ids = sim_to_event(event, WCD, n_mpmt_total=106, pe_scale=143)

# # #         if P_LOCATIONS is None or DIRECTION_ZS is None:
# # #             P_LOCATIONS, DIRECTION_ZS, MPMT_SLOTS = EMITTER_TEMPLATE.get_pmt_placements(ev, WCD, "est")
# # #             MPMT_SLOTS = np.asarray(MPMT_SLOTS, dtype=int)
# # #             RING_KEEP_MASK = np.isin(MPMT_SLOTS, ALL_RING)

# # #         obs_pes, obs_ts = build_observables_from_event(ev, pe_scale=143)
# # #         obs_pes, obs_ts = apply_ring_mask_to_observables(
# # #             obs_pes,
# # #             obs_ts,
# # #             RING_KEEP_MASK,
# # #             mode=RING_MASK_MODE,
# # #         )

# # #         obs_pes_all.append(obs_pes)
# # #         obs_ts_all.append(obs_ts)
# # #         mpmt_slots_all.append(MPMT_SLOTS)

# # #     OBS_PES_ALL = obs_pes_all
# # #     OBS_TS_ALL = obs_ts_all
# # #     MPMT_SLOTS_ALL = mpmt_slots_all
# # #     tot_events = len(OBS_PES_ALL)

# # #     print("Computing truth FCNs...")
# # #     true_fcn_all = [compute_true_fcn_for_event(i) for i in range(tot_events)]

# # #     est_dict = {
# # #         "metadata": {
# # #             "fit_particle": FIT_PARTICLE_CANONICAL,
# # #             "particle_mass_mev": FIT_PARTICLE_MASS_MEV,
# # #             "particle_threshold_mev": FIT_PARTICLE_THRESHOLD_MEV,
# # #             "beam_p": BEAM_P,
# # #             "track_end_mode": TRACK_END_MODE,
# # #             "fit_parameters": list(FIT_PARAMETER_NAMES),
# # #             "truth_params": dict(TRUE_PARAMS),
# # #             "geometry_file": GEOMETRY_FILE,
# # #             "config_root_file": CONFIG_ROOT_FILE,
# # #             "event_source": EVENT_SOURCE,
# # #             "user_event_file": USER_EVENT_FILE if EVENT_SOURCE == "file" else None,
# # #             "user_event_key": USER_EVENT_KEY if EVENT_SOURCE == "file" else None,
# # #             "user_event_apply_peak_window": bool(USER_EVENT_APPLY_PEAK_WINDOW) if EVENT_SOURCE == "file" else None,
# # #             "particle_selection_label": PARTICLE_SELECTION_LABEL,
# # #             "selection_tof_ns": SELECTION_TOF_NS,
# # #             "selection_tof_window_ns": SELECTION_TOF_WINDOW_NS,
# # #             "selection_tof_field": SELECTION_TOF_FIELD,
# # #             "selection_t5_particle_nr": SELECTION_T5_PARTICLE_NR,
# # #             "range_table_max_full_range_mm": float(RANGE_LOOKUP.overall_distances_mm[-1]),
# # #             "save_seed_scan": bool(SAVE_SEED_SCAN),
# # #             "save_top_n_seeds": int(SAVE_TOP_N_SEEDS),
# # #             "save_attempt_results": bool(SAVE_ATTEMPT_RESULTS),
# # #         },
# # #         "minimum_found": [],
# # #         "x": [],
# # #         "y": [],
# # #         "z": [],
# # #         "visible_length": [],
# # #         "full_range": [],
# # #         "ke0": [],
# # #         "length": [],  # legacy alias for visible_length
# # #         "t": [],
# # #         "est_fcn": [],
# # #         "true_fcn": [],
# # #         "cx": [],
# # #         "cy": [],
# # #         "n_attempts": [],
# # #         "chosen_seed_idx": [],
# # #         "chosen_seed_fcn": [],
# # #         "chosen_seed_params": [],
# # #         "adaptive_rescue_used": [],
# # #         "length_profile_rescue_considered": [],
# # #         "length_profile_rescue_used": [],
# # #         "edm": [],
# # #     }
# # #     if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
# # #         est_dict["seed_scan"] = []

# # #     if SAVE_ATTEMPT_RESULTS:
# # #         est_dict["attempt_results"] = []

# # #     n_events_per_batch = min(N_EVENTS_PER_BATCH, max(1, tot_events))

# # #     for batch_start in range(0, tot_events, n_events_per_batch):
# # #         batch_end = min(batch_start + n_events_per_batch, tot_events)
# # #         event_indices = list(range(batch_start, batch_end))

# # #         print(f"Starting event number {batch_start}")

# # #         results = run_batch(
# # #             event_indices=event_indices,
# # #             init_param_sets=init_param_sets,
# # #             nproc=NPROC,
# # #             fcn_threshold=FCN_RETRY_THRESHOLD,
# # #             max_attempts=MAX_FIT_ATTEMPTS,
# # #             ncall=NCALL_MIGRAD,
# # #         )

# # #         for local_i, result in enumerate(results):
# # #             event_index = event_indices[local_i]
# # #             vals = result["values"]

# # #             if IS_ABSORPTION_MODE:
# # #                 visible_length = float(vals["visible_length"])
# # #                 full_range = float(vals["full_range"])
# # #             else:
# # #                 visible_length = float(vals["length"])
# # #                 full_range = visible_length
# # #             ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))

# # #             est_dict["minimum_found"].append(int(result["valid"]))
# # #             est_dict["x"].append(vals["x0"])
# # #             est_dict["y"].append(vals["y0"])
# # #             est_dict["z"].append(vals["z0"])
# # #             est_dict["visible_length"].append(visible_length)
# # #             est_dict["full_range"].append(full_range)
# # #             est_dict["ke0"].append(ke0)
# # #             est_dict["length"].append(visible_length)
# # #             est_dict["t"].append(vals["t0"])
# # #             est_dict["cx"].append(vals["cx"])
# # #             est_dict["cy"].append(vals["cy"])
# # #             est_dict["est_fcn"].append(result["fval"])
# # #             est_dict["true_fcn"].append(true_fcn_all[event_index])
# # #             est_dict["n_attempts"].append(result["attempts"])
# # #             est_dict["chosen_seed_idx"].append(result["chosen_seed_index"])
# # #             est_dict["chosen_seed_fcn"].append(result["chosen_seed_fcn"])
# # #             est_dict["chosen_seed_params"].append(result["chosen_seed_params"])
# # #             if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
# # #                 est_dict["seed_scan"].append(result.get("seed_scan", []))
# # #             est_dict["adaptive_rescue_used"].append(result.get("adaptive_rescue_used", False))
# # #             est_dict["length_profile_rescue_considered"].append(result.get("length_profile_rescue_considered", False))
# # #             est_dict["length_profile_rescue_used"].append(result.get("length_profile_rescue_used", False))
# # #             est_dict["edm"].append(result.get("edm", np.nan))
# # #             if SAVE_ATTEMPT_RESULTS:
# # #                 est_dict["attempt_results"].append(result.get("attempt_results", []))

# # #     Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
# # #     with open(OUTPUT_FILE, "wb") as f:
# # #         pickle.dump(est_dict, f)

# # #     print("Done.")
# # #     print("Saved:", OUTPUT_FILE)


# # # if __name__ == "__main__":
# # #     main()








# # # # # 7,8-parameter abrupt-endpoint batch driver.
# # # # #
# # # # # Fit parameters:
# # # # #   x0, y0, z0, cx, cy, visible_length, full_range, t0
# # # # #
# # # # # Meaning:
# # # # #   visible_length = actual visible primary-Cherenkov length before abrupt cutoff [mm]
# # # # #   full_range     = dE/dx-only range to Cherenkov threshold if no abrupt interaction [mm]
# # # # #   ke0            = inferred initial kinetic energy from full_range using particle range table [MeV]
# # # # #
# # # # # This driver assumes an abrupt endpoint model.  It does not use fixed_initial_KE
# # # # # as a fit setting; fixed_initial_KE is overwritten inside every FCN call using
# # # # # full_range -> ke0.

# # # # """Toggleable multi-stage batch driver for the 8-parameter Minuit Cherenkov fit on WCTE/real-data-style events.

# # # # This is the selected-event/input-array version of the driver.  It preserves:
# # # #   - get_selected_events(RUN, N_EVENTS) event loading
# # # #   - run configuration GOOD_WCTE_PMTS masking from the ROOT Configuration tree
# # # #   - pe_scale=143
# # # #   - estimated geometry placement "est"
# # # #   - mPMT type/relative efficiency corrections when tables are available

# # # # The 8 fitted parameters are:
# # # #   x0, y0, z0, cx, cy, visible_length, full_range, t0
# # # # """

# # # # import os
# # # # import sys
# # # # import pickle
# # # # import multiprocessing as mp
# # # # from pathlib import Path

# # # # import numpy as np
# # # # import uproot
# # # # from iminuit import Minuit

# # # # # =============================================================================
# # # # # SELF-CONTAINED PATH SETUP
# # # # # =============================================================================
# # # # SCRIPT_DIR = Path(__file__).resolve().parent
# # # # PROJECT_ROOT = SCRIPT_DIR.parent
# # # # LICKETYFIT_DIR = PROJECT_ROOT / "LicketyFit"
# # # # TABLE_DIR = PROJECT_ROOT / "tables"
# # # # OUTPUT_DIR = PROJECT_ROOT / "outputs"
# # # # OUTPUT_DIR.mkdir(exist_ok=True)

# # # # geometry_path = os.environ.get("GEOMETRY_PATH", "/eos/user/j/jrimmer/Geometry")
# # # # GEOMETRY_FILE = os.environ.get(
# # # #     "WCTE_GEOMETRY_FILE",
# # # #     str(Path(geometry_path) / "examples" / "wcte_bldg157.geo"),
# # # # )

# # # # for _path in (str(LICKETYFIT_DIR), str(PROJECT_ROOT), str(SCRIPT_DIR), str(TABLE_DIR), geometry_path):
# # # #     if _path not in sys.path:
# # # #         sys.path.insert(0, _path)

# # # # # Force local tables before importing lookup/collapse helpers.
# # # # os.environ["LF_TABLE_DIR"] = str(TABLE_DIR)
# # # # os.environ["LF_MULTIPARTICLES_TABLE_DIR"] = str(TABLE_DIR)

# # # # from Geometry.Device import Device
# # # # from LicketyFit.Event import Event
# # # # from LicketyFit.PMT import PMT
# # # # from LicketyFit.Emitter import Emitter
# # # # from particle_cherenkov_model import (
# # # #     get_energy_distance_tables,
# # # #     set_active_particle,
# # # #     canonical_particle_name,
# # # #     particle_mass_mev,
# # # #     cherenkov_threshold_kinetic_mev,
# # # # )
# # # # try:
# # # #     from event_loader import get_selected_events
# # # # except Exception:
# # # #     get_selected_events = None
# # # # from particle_range_lookup import ParticleRangeLookup


# # # # # =============================================================================
# # # # # ENV HELPERS
# # # # # =============================================================================
# # # # def _env_float(name, default=None):
# # # #     raw = os.environ.get(name)
# # # #     if raw is None or str(raw).strip() == "":
# # # #         return default
# # # #     return float(raw)


# # # # def _env_int(name, default):
# # # #     raw = os.environ.get(name)
# # # #     if raw is None or str(raw).strip() == "":
# # # #         return int(default)
# # # #     return int(raw)


# # # # def _env_bool(name, default=False):
# # # #     raw = os.environ.get(name)
# # # #     if raw is None:
# # # #         return bool(default)
# # # #     return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


# # # # def _parse_float_list_env(name, default):
# # # #     raw = os.environ.get(name)
# # # #     if raw is None or str(raw).strip() == "":
# # # #         return list(default)
# # # #     return [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]


# # # # # =============================================================================
# # # # # TOP-LEVEL CONFIGURATION
# # # # # =============================================================================
# # # # N_EVENTS_PER_BATCH = _env_int("N_EVENTS_PER_BATCH", 100)
# # # # NPROC = _env_int("NPROC", 16)
# # # # M_STRAT = _env_int("M_STRAT", 1)

# # # # Z_SEED_EPS = 20.0
# # # # VISIBLE_LENGTH_SEED_EPS = 40.0
# # # # FULL_RANGE_SEED_EPS = 80.0
# # # # T_MIN = -8.0

# # # # FCN_RETRY_THRESHOLD = 1100.0
# # # # VISIBLE_LENGTH_RETRY_THRESHOLD = _env_float("VISIBLE_LENGTH_RETRY_THRESHOLD", 2700.0)
# # # # MAX_FIT_ATTEMPTS = 4
# # # # NCALL_MIGRAD = _env_int("NCALL_MIGRAD", 70000)
# # # # NCALL_SIMPLEX = _env_int("NCALL_SIMPLEX", NCALL_MIGRAD)

# # # # RUN = _env_int("RUN", 2079)
# # # # BEAM_P = _env_float("BEAM_P", 430)
# # # # #N_EVENTS = _env_int("N_EVENTS", 7000)
# # # # N_EVENTS = 50000
# # # # # =============================================================================
# # # # # PARTICLE HYPOTHESIS / 8-PARAMETER MODE
# # # # # =============================================================================
# # # # FIT_PARTICLE = os.environ.get("FIT_PARTICLE", "muon")
# # # # FIT_PARTICLE_CANONICAL = canonical_particle_name(FIT_PARTICLE)
# # # # FIT_PARTICLE_MASS_MEV = particle_mass_mev(FIT_PARTICLE_CANONICAL)
# # # # FIT_PARTICLE_THRESHOLD_MEV = cherenkov_threshold_kinetic_mev(FIT_PARTICLE_MASS_MEV)
# # # # set_active_particle(FIT_PARTICLE_CANONICAL)

# # # # # Fit mode:
# # # # #   full_length -> original 7-parameter fit:
# # # # #                  x0, y0, z0, cx, cy, length, t0
# # # # #                  length is the dE/dx range to Cherenkov threshold, so ke0 is inferred from length.
# # # # #   absorption  -> 8-parameter abrupt-endpoint fit:
# # # # #                  x0, y0, z0, cx, cy, visible_length, full_range, t0
# # # # #                  visible_length is the abrupt cutoff; full_range determines ke0.
# # # # _FIT_MODE_RAW = os.environ.get("FIT_MODE", os.environ.get("TRACK_END_MODE", "full_length")).strip().lower()

# # # # if _FIT_MODE_RAW in {"absorption", "absorbed", "abrupt", "abrupt_8param", "interaction", "truncated"}:
# # # #     FIT_MODE = "absorption"
# # # # elif _FIT_MODE_RAW in {"full_length", "full-length", "full", "threshold", "range", "csda", "old", "7param", "7_parameter"}:
# # # #     FIT_MODE = "full_length"
# # # # else:
# # # #     raise ValueError("FIT_MODE/TRACK_END_MODE must be 'full_length' or 'absorption'")

# # # # TRACK_END_MODE = FIT_MODE
# # # # IS_ABSORPTION_MODE = FIT_MODE == "absorption"
# # # # IS_FULL_LENGTH_MODE = FIT_MODE == "full_length"
# # # # EMITTER_TRACK_END_MODE = "abrupt" if IS_ABSORPTION_MODE else "threshold"
# # # # FIT_PARAMETER_NAMES = (
# # # #     ("x0", "y0", "z0", "cx", "cy", "visible_length", "full_range", "t0")
# # # #     if IS_ABSORPTION_MODE
# # # #     else ("x0", "y0", "z0", "cx", "cy", "length", "t0")
# # # # )

# # # # # Output/debug controls.  The full seed scan is large in the 8-parameter
# # # # # fit because every event can have thousands of seed dictionaries.  Keep these
# # # # # off for production output; enable only when debugging seed selection.
# # # # SAVE_ATTEMPT_RESULTS = _env_bool("SAVE_ATTEMPT_RESULTS", False)
# # # # SAVE_SEED_SCAN = _env_bool("SAVE_SEED_SCAN", False)
# # # # SAVE_TOP_N_SEEDS = _env_int("SAVE_TOP_N_SEEDS", 0)

# # # # # Likelihood toggles.
# # # # USE_CHARGE_LIKELIHOOD = _env_bool("USE_CHARGE_LIKELIHOOD", True)
# # # # USE_TIMING_LIKELIHOOD = _env_bool("USE_TIMING_LIKELIHOOD", True)
# # # # USE_T0_PRIOR = _env_bool("USE_T0_PRIOR", False)

# # # # if (not USE_CHARGE_LIKELIHOOD) and (not USE_TIMING_LIKELIHOOD):
# # # #     raise ValueError("At least one likelihood term must be enabled.")

# # # # if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
# # # #     LIKELIHOOD_MODE = "charge_time"
# # # # elif USE_CHARGE_LIKELIHOOD:
# # # #     LIKELIHOOD_MODE = "charge_only"
# # # # else:
# # # #     LIKELIHOOD_MODE = "timing_only"

# # # # OUTPUT_FILE = os.environ.get(
# # # #     "LF_OUTPUT_FILE",
# # # #     str(OUTPUT_DIR / f"estimates_run{RUN}_{BEAM_P:g}p_{FIT_PARTICLE_CANONICAL}_{TRACK_END_MODE}_mpmtEff_{LIKELIHOOD_MODE}.dict"),
# # # # )

# # # # RING_MASK_MODE = os.environ.get("RING_MASK_MODE", "both").strip().lower()

# # # # # Data configuration.
# # # # CONFIG_ROOT_FILE = os.environ.get(
# # # #     "CONFIG_ROOT_FILE",
# # # #     f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/{RUN}/WCTE_merged_production_R{RUN}.root",
# # # # )

# # # # # Event source.
# # # # #   "selection" / "internal" : use event_loader.get_selected_events(...) with particle-specific TOF cuts.
# # # # #   "file" / "user" / "custom" : load already-selected user events from USER_EVENT_FILE.
# # # # #
# # # # # USER_EVENT_FILE can be .npy, .npz, .pkl, or .pickle.  Supported payloads:
# # # # #   - list/object array of event arrays, each with columns [pmt_id, charge, time] or
# # # # #     [pmt_id, charge, time, event_number]
# # # # #   - a single 2D array [pmt_id, charge, time] for one event
# # # # #   - a single 2D array [pmt_id, charge, time, event_number], which is grouped by
# # # # #     event_number
# # # # EVENT_SOURCE = os.environ.get("EVENT_SOURCE", "selection").strip().lower()
# # # # if EVENT_SOURCE in {"selected", "internal", "event_loader", "auto"}:
# # # #     EVENT_SOURCE = "selection"
# # # # elif EVENT_SOURCE in {"file", "user", "custom", "user_file", "provided"}:
# # # #     EVENT_SOURCE = "file"
# # # # if EVENT_SOURCE not in {"selection", "file"}:
# # # #     raise ValueError("EVENT_SOURCE must be 'selection' or 'file'")

# # # # USER_EVENT_FILE = os.environ.get("USER_EVENT_FILE", "").strip()
# # # # USER_EVENT_KEY = os.environ.get("USER_EVENT_KEY", "").strip() or None
# # # # USER_EVENT_APPLY_PEAK_WINDOW = _env_bool("USER_EVENT_APPLY_PEAK_WINDOW", True)

# # # # # Event-selection configuration for event_loader.get_selected_events().
# # # # # Defaults reproduce the historical muon-like WCTE selection.  For non-muon
# # # # # beam selections, set PARTICLE_SELECTION_LABEL plus either SELECTION_TOF_NS
# # # # # or SELECTION_TOF_FIELD/T5_PARTICLE_NR as needed for your production ROOT file.
# # # # PARTICLE_SELECTION_LABEL = os.environ.get("PARTICLE_SELECTION_LABEL", FIT_PARTICLE_CANONICAL)
# # # # SELECTION_TOF_NS = _env_float("SELECTION_TOF_NS", None)
# # # # SELECTION_TOF_WINDOW_NS = _env_float("SELECTION_TOF_WINDOW_NS", 0.2)
# # # # SELECTION_TOF_FIELD = os.environ.get("SELECTION_TOF_FIELD", "") or None
# # # # SELECTION_MOMENTUM_FIELD = os.environ.get("SELECTION_MOMENTUM_FIELD", "") or None
# # # # SELECTION_T5_PARTICLE_NR = _env_int("SELECTION_T5_PARTICLE_NR", 1)
# # # # USE_PEAK_TIME_CUT = _env_bool("USE_PEAK_TIME_CUT", True)
# # # # PEAK_WINDOW_NS = _env_float("PEAK_WINDOW_NS", 100.0)
# # # # PEAK_BIN_WIDTH_NS = _env_float("PEAK_BIN_WIDTH_NS", 50.0)

# # # # # =============================================================================
# # # # # DETECTOR CONFIGURATION
# # # # # =============================================================================
# # # # DEFAULT_INACTIVE_SLOTS = [27, 32, 45, 74, 77, 79, 85, 91, 99, 9, 67]
# # # # INACTIVE_SLOTS = [int(x) for x in os.environ.get(
# # # #     "INACTIVE_SLOTS",
# # # #     ",".join(str(x) for x in DEFAULT_INACTIVE_SLOTS),
# # # # ).replace(";", ",").split(",") if x.strip()]
# # # # INACTIVE_SLOTS_SET = set(int(s) for s in INACTIVE_SLOTS)

# # # # OUTER_RING = np.array([0, 7, 19, 34, 50, 66, 82, 83, 105, 94, 95, 71, 72, 56, 40, 24, 11, 3, 18])
# # # # INNER_RING = np.array([1, 8, 35, 51, 67, 84, 69, 70, 55, 39, 23, 10, 2, 20, 36, 52, 68, 53, 54, 38, 22, 21, 37, 9])
# # # # OUTSIDE_RING = np.array([12, 13, 4, 5, 6, 17, 33, 49, 65, 81, 82, 104, 93, 86, 87, 72, 57, 41, 25])
# # # # ALL_RING = np.arange(0, 106)

# # # # # Optional truth diagnostics.  If not supplied, true_fcn is NaN.
# # # # TRUE_LENGTH_MM = _env_float("TRUE_LENGTH_MM", None)
# # # # TRUE_VISIBLE_LENGTH_MM = _env_float("TRUE_VISIBLE_LENGTH_MM", None)
# # # # TRUE_FULL_RANGE_MM = _env_float("TRUE_FULL_RANGE_MM", None)
# # # # TRUE_INITIAL_KE_MEV = _env_float("TRUE_INITIAL_KE_MEV", None)

# # # # TRUE_PARAMS = {
# # # #     "x0": _env_float("TRUE_X0", 0.0),
# # # #     "y0": _env_float("TRUE_Y0", 0.0),
# # # #     "z0": _env_float("TRUE_Z0", -1348.0),
# # # #     "cx": _env_float("TRUE_CX", 0.0),
# # # #     "cy": _env_float("TRUE_CY", 0.0),
# # # #     "visible_length": np.nan,
# # # #     "full_range": np.nan,
# # # #     "t0": _env_float("TRUE_T0", 0.0),
# # # # }

# # # # # =============================================================================
# # # # # GLOBAL FIT-SEARCH STAGE TOGGLES
# # # # # =============================================================================
# # # # ENABLE_STAGE1_SEED_GRID = _env_bool("ENABLE_STAGE1_SEED_GRID", True)
# # # # ENABLE_STAGE2_MIGRAD_FIRST = _env_bool("ENABLE_STAGE2_MIGRAD_FIRST", False)
# # # # ENABLE_STAGE3_ADAPTIVE_RESCUE = _env_bool("ENABLE_STAGE3_ADAPTIVE_RESCUE", False)
# # # # ENABLE_STAGE4_LENGTH_PROFILE = _env_bool("ENABLE_STAGE4_LENGTH_PROFILE", False)

# # # # # Keep the full seed ranking only when it is actually needed.  This avoids
# # # # # sorting and returning thousands of seed dictionaries for normal production fits.
# # # # NEED_FULL_SEED_SCAN = (
# # # #     ENABLE_STAGE3_ADAPTIVE_RESCUE
# # # #     or ENABLE_STAGE4_LENGTH_PROFILE
# # # #     or SAVE_SEED_SCAN
# # # #     or SAVE_TOP_N_SEEDS > 0
# # # # )

# # # # # =============================================================================
# # # # # INITIAL SEED CONFIGURATION
# # # # # =============================================================================
# # # # FAST_SEED_X0 = _parse_float_list_env("FAST_SEED_X0", [-150.0, -100, -50, 0.0, 50, 100, 150.0])
# # # # FAST_SEED_Y0 = _parse_float_list_env("FAST_SEED_Y0", [-50.0, 0.0, 50.0])
# # # # FAST_SEED_Z0 = _parse_float_list_env("FAST_SEED_Z0", [-1500.0, -1400.0, -1300.0, -1350, -1200.0, -1100.0, -1000.0])

# # # # FAST_SEED_VISIBLE_LENGTHS = _parse_float_list_env(
# # # #     "FAST_SEED_VISIBLE_LENGTHS",
# # # #     [100.0, 200, 300.0, 400, 450, 500.0, 700.0, 900.0, 1100.0, 1300.0, 1400.0, 1500.0, 1700.0, 1900.0],
# # # # )

# # # # FAST_SEED_KE0_MEV = _parse_float_list_env(
# # # #     "FAST_SEED_KE0_MEV",
# # # #     [600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0],
# # # # )
# # # # FAST_SEED_FULL_RANGES_MM = _parse_float_list_env(
# # # #     "FAST_SEED_FULL_RANGES_MM",
# # # #     [300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0],
# # # # )
# # # # FAST_SEED_DIRECTIONS = [
# # # #     (0.0, 0.0),
# # # #     (0.04, 0.0),
# # # #     (-0.04, 0.0),
# # # #     (0.0, 0.04),
# # # #     (0.0, -0.04),
# # # # ]
# # # # FAST_SEED_FULL_CARTESIAN = _env_bool("FAST_SEED_FULL_CARTESIAN", False)


# # # # def build_sparse_geometry_variants():
# # # #     variants = []
# # # #     for x0 in FAST_SEED_X0:
# # # #         variants.append({"x0": float(x0), "y0": 0.0, "cx": 0.0, "cy": 0.0})
# # # #     for y0 in FAST_SEED_Y0:
# # # #         variants.append({"x0": 0.0, "y0": float(y0), "cx": 0.0, "cy": 0.0})
# # # #     for cx, cy in FAST_SEED_DIRECTIONS:
# # # #         variants.append({"x0": 0.0, "y0": 0.0, "cx": float(cx), "cy": float(cy)})

# # # #     unique = []
# # # #     seen = set()
# # # #     for v in variants:
# # # #         sig = (float(v["x0"]), float(v["y0"]), float(v["cx"]), float(v["cy"]))
# # # #         if sig not in seen:
# # # #             seen.add(sig)
# # # #             unique.append(v)
# # # #     return unique


# # # # FAST_SEED_GEOMETRY_VARIANTS = build_sparse_geometry_variants()


# # # # def build_full_range_seed_values(range_lookup):
# # # #     values = []
# # # #     for ke0 in FAST_SEED_KE0_MEV:
# # # #         if ke0 <= range_lookup.threshold_mev:
# # # #             continue
# # # #         r = range_lookup.energy_to_range_mm(float(ke0))
# # # #         if np.isfinite(r) and r > 0:
# # # #             values.append(float(r))

# # # #     values.extend(float(r) for r in FAST_SEED_FULL_RANGES_MM)

# # # #     max_r = float(range_lookup.overall_distances_mm[-1])
# # # #     values = [r for r in values if np.isfinite(r) and 0.0 < r <= max_r]

# # # #     unique = []
# # # #     seen = set()
# # # #     for r in values:
# # # #         sig = round(float(r), 6)
# # # #         if sig not in seen:
# # # #             seen.add(sig)
# # # #             unique.append(float(r))
# # # #     return unique


# # # # def build_fast_seed_grid(range_lookup):
# # # #     seeds = []

# # # #     if IS_FULL_LENGTH_MODE:
# # # #         # 7-parameter original/full-length mode: scan only one length-like parameter.
# # # #         if FAST_SEED_FULL_CARTESIAN:
# # # #             for x0 in FAST_SEED_X0:
# # # #                 for y0 in FAST_SEED_Y0:
# # # #                     for z0 in FAST_SEED_Z0:
# # # #                         for length in FAST_SEED_VISIBLE_LENGTHS:
# # # #                             for cx, cy in FAST_SEED_DIRECTIONS:
# # # #                                 seeds.append({
# # # #                                     "x0": float(x0),
# # # #                                     "y0": float(y0),
# # # #                                     "z0": float(z0),
# # # #                                     "cx": float(cx),
# # # #                                     "cy": float(cy),
# # # #                                     "length": float(length),
# # # #                                     "t0": 0.0,
# # # #                                 })
# # # #         else:
# # # #             for z0 in FAST_SEED_Z0:
# # # #                 for length in FAST_SEED_VISIBLE_LENGTHS:
# # # #                     for geom in FAST_SEED_GEOMETRY_VARIANTS:
# # # #                         seeds.append({
# # # #                             "x0": float(geom["x0"]),
# # # #                             "y0": float(geom["y0"]),
# # # #                             "z0": float(z0),
# # # #                             "cx": float(geom["cx"]),
# # # #                             "cy": float(geom["cy"]),
# # # #                             "length": float(length),
# # # #                             "t0": 0.0,
# # # #                         })
# # # #         keys = ("x0", "y0", "z0", "cx", "cy", "length", "t0")
# # # #     else:
# # # #         # 8-parameter absorption mode: scan visible cutoff length and full CSDA range separately.
# # # #         full_range_seeds = build_full_range_seed_values(range_lookup)
# # # #         if FAST_SEED_FULL_CARTESIAN:
# # # #             for x0 in FAST_SEED_X0:
# # # #                 for y0 in FAST_SEED_Y0:
# # # #                     for z0 in FAST_SEED_Z0:
# # # #                         for visible_length in FAST_SEED_VISIBLE_LENGTHS:
# # # #                             for full_range in full_range_seeds:
# # # #                                 if visible_length > full_range:
# # # #                                     continue
# # # #                                 for cx, cy in FAST_SEED_DIRECTIONS:
# # # #                                     seeds.append({
# # # #                                         "x0": float(x0),
# # # #                                         "y0": float(y0),
# # # #                                         "z0": float(z0),
# # # #                                         "cx": float(cx),
# # # #                                         "cy": float(cy),
# # # #                                         "visible_length": float(visible_length),
# # # #                                         "full_range": float(full_range),
# # # #                                         "t0": 0.0,
# # # #                                     })
# # # #         else:
# # # #             for z0 in FAST_SEED_Z0:
# # # #                 for visible_length in FAST_SEED_VISIBLE_LENGTHS:
# # # #                     for full_range in full_range_seeds:
# # # #                         if visible_length > full_range:
# # # #                             continue
# # # #                         for geom in FAST_SEED_GEOMETRY_VARIANTS:
# # # #                             seeds.append({
# # # #                                 "x0": float(geom["x0"]),
# # # #                                 "y0": float(geom["y0"]),
# # # #                                 "z0": float(z0),
# # # #                                 "cx": float(geom["cx"]),
# # # #                                 "cy": float(geom["cy"]),
# # # #                                 "visible_length": float(visible_length),
# # # #                                 "full_range": float(full_range),
# # # #                                 "t0": 0.0,
# # # #                             })
# # # #         keys = ("x0", "y0", "z0", "cx", "cy", "visible_length", "full_range", "t0")

# # # #     unique = []
# # # #     seen = set()
# # # #     for seed in seeds:
# # # #         sig = tuple(float(seed[k]) for k in keys)
# # # #         if sig not in seen:
# # # #             seen.add(sig)
# # # #             unique.append(seed)
# # # #     return unique


# # # # PARAM_NAMES = FIT_PARAMETER_NAMES

# # # # # =============================================================================
# # # # # GLOBALS SHARED BY WORKERS
# # # # # =============================================================================
# # # # OVERALL_DISTANCES = None
# # # # INIT_ENERGY_TABLE = None
# # # # RANGE_LOOKUP = None

# # # # WCD = None
# # # # PMT_MODEL = None
# # # # EMITTER_TEMPLATE = None
# # # # P_LOCATIONS = None
# # # # DIRECTION_ZS = None
# # # # RING_KEEP_MASK = None
# # # # CORR_POS = None
# # # # MPMT_SLOTS_ALL = None

# # # # OBS_PES_ALL = None
# # # # OBS_TS_ALL = None
# # # # GOOD_WCTE_PMTS_SET = None

# # # # # =============================================================================
# # # # # mPMT INFO / EFFICIENCY TABLES
# # # # # =============================================================================
# # # # other_mpmt_info_path = Path(os.environ.get("OTHER_MPMT_INFO_PATH", str(TABLE_DIR / "other_mpmt_info_v2.dict")))
# # # # if other_mpmt_info_path.exists():
# # # #     with open(other_mpmt_info_path, "rb") as f:
# # # #         mpmt_info = pickle.load(f)
# # # # else:
# # # #     mpmt_info = {}

# # # # rel_mpmt_eff_path = Path(os.environ.get("REL_MPMT_EFF_PATH", str(TABLE_DIR / "rel_mpmt_eff.dict")))
# # # # if rel_mpmt_eff_path.exists():
# # # #     with open(rel_mpmt_eff_path, "rb") as f:
# # # #         rel_mpmt_eff = pickle.load(f)
# # # # else:
# # # #     unity = np.ones(200, dtype=np.float64)
# # # #     rel_mpmt_eff = {
# # # #         "tri_exsitu": unity,
# # # #         "tri_insitu": unity,
# # # #         "wut_insitu": unity,
# # # #         "wut_exsitu": unity,
# # # #     }

# # # # tri_exsitu = rel_mpmt_eff["tri_exsitu"]
# # # # tri_insitu = rel_mpmt_eff["tri_insitu"]
# # # # wut_insitu = rel_mpmt_eff["wut_insitu"]
# # # # wut_exsitu = rel_mpmt_eff["wut_exsitu"]


# # # # def get_mpmt_slot_type(mpmt_slots):
# # # #     slot_type = []
# # # #     for slot in mpmt_slots:
# # # #         slot = int(slot)
# # # #         try:
# # # #             if mpmt_info[slot]["mpmt_site"] == "TRI":
# # # #                 if mpmt_info[slot]["mpmt_type"] == "In-situ":
# # # #                     slot_type.append("tri_insitu")
# # # #                 else:
# # # #                     slot_type.append("tri_exsitu")
# # # #             else:
# # # #                 if mpmt_info[slot]["mpmt_type"] == "In-situ":
# # # #                     slot_type.append("wut_insitu")
# # # #                 else:
# # # #                     slot_type.append("wut_exsitu")
# # # #         except Exception:
# # # #             slot_type.append("empty")
# # # #     return slot_type


# # # # # =============================================================================
# # # # # CONFIG / TRUTH HELPERS
# # # # # =============================================================================
# # # # def load_good_wcte_pmts():
# # # #     try:
# # # #         with uproot.open(CONFIG_ROOT_FILE) as f:
# # # #             t_c = f["Configuration"]
# # # #             arr_config = t_c.arrays(library="ak")
# # # #         good = np.asarray(arr_config["good_wcte_pmts"][0], dtype=int)
# # # #         print("Loaded GOOD_WCTE_PMTS from:", CONFIG_ROOT_FILE)
# # # #         return set(good.tolist())
# # # #     except Exception as exc:
# # # #         # For selected ROOT input, missing the run Configuration tree is usually a
# # # #         # real problem.  For user-provided event files, allow a self-contained
# # # #         # fallback by default and turn on every non-inactive PMT.
# # # #         allow_missing_default = EVENT_SOURCE == "file"
# # # #         if _env_bool("ALLOW_MISSING_GOOD_PMTS", allow_missing_default):
# # # #             print("WARNING: could not load GOOD_WCTE_PMTS; using all non-inactive PMTs.")
# # # #             print("Reason:", repr(exc))
# # # #             all_ids = []
# # # #             for slot in range(106):
# # # #                 if slot in INACTIVE_SLOTS_SET:
# # # #                     continue
# # # #                 for pmt_pos in range(19):
# # # #                     all_ids.append(slot * 100 + pmt_pos)
# # # #             return set(all_ids)
# # # #         raise


# # # # def configure_truth_params():
# # # #     if IS_ABSORPTION_MODE:
# # # #         if TRUE_FULL_RANGE_MM is not None:
# # # #             TRUE_PARAMS["full_range"] = float(TRUE_FULL_RANGE_MM)
# # # #         elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
# # # #             TRUE_PARAMS["full_range"] = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))

# # # #         if TRUE_VISIBLE_LENGTH_MM is not None:
# # # #             TRUE_PARAMS["visible_length"] = float(TRUE_VISIBLE_LENGTH_MM)
# # # #         elif TRUE_LENGTH_MM is not None:
# # # #             TRUE_PARAMS["visible_length"] = float(TRUE_LENGTH_MM)
# # # #         else:
# # # #             TRUE_PARAMS["visible_length"] = np.nan

# # # #         if np.isfinite(TRUE_PARAMS["visible_length"]):
# # # #             TRUE_PARAMS["length"] = TRUE_PARAMS["visible_length"]
# # # #         return

# # # #     # Full-length mode: the single fitted length is also the full CSDA range.
# # # #     if TRUE_LENGTH_MM is not None:
# # # #         length = float(TRUE_LENGTH_MM)
# # # #     elif TRUE_FULL_RANGE_MM is not None:
# # # #         length = float(TRUE_FULL_RANGE_MM)
# # # #     elif TRUE_VISIBLE_LENGTH_MM is not None:
# # # #         length = float(TRUE_VISIBLE_LENGTH_MM)
# # # #     elif TRUE_INITIAL_KE_MEV is not None and RANGE_LOOKUP is not None:
# # # #         length = float(RANGE_LOOKUP.energy_to_range_mm(TRUE_INITIAL_KE_MEV))
# # # #     else:
# # # #         length = np.nan

# # # #     TRUE_PARAMS["length"] = length
# # # #     TRUE_PARAMS["visible_length"] = length
# # # #     TRUE_PARAMS["full_range"] = length


# # # # # =============================================================================
# # # # # EVENT / OBSERVABLE HELPERS
# # # # # =============================================================================
# # # # def sim_to_event(
# # # #     sim_data,
# # # #     WCD,
# # # #     n_mpmt_total=106,
# # # #     pe_scale=143,
# # # #     shift_times=True,
# # # #     n_earliest_for_t0=10,
# # # # ):
# # # #     vw = 223.0598645833333  # mm/ns

# # # #     ev = Event(0, 0, n_mpmt_total)
# # # #     ev.set_mpmt_status(list(range(n_mpmt_total)), False)

# # # #     active_wcte_pmt_ids = []

# # # #     for slot in range(n_mpmt_total):
# # # #         if slot in INACTIVE_SLOTS_SET:
# # # #             continue

# # # #         slot_has_good_pmt = False
# # # #         for pmt_pos_id in range(ev.npmt_per_mpmt):
# # # #             wcte_pmt = int(slot * 100 + pmt_pos_id)
# # # #             if wcte_pmt in GOOD_WCTE_PMTS_SET:
# # # #                 ev.set_pmt_status(slot, [pmt_pos_id], True)
# # # #                 slot_has_good_pmt = True
# # # #                 active_wcte_pmt_ids.append(wcte_pmt)

# # # #         if slot_has_good_pmt:
# # # #             ev.set_mpmt_status([slot], True)

# # # #     for i in range(len(sim_data[:, 0])):
# # # #         wcte_pmt = int(sim_data[i, 0])
# # # #         slot = int(wcte_pmt // 100)
# # # #         pmt_pos_id = int(wcte_pmt % 100)

# # # #         if slot < 0 or slot >= ev.n_mpmt:
# # # #             continue
# # # #         if pmt_pos_id < 0 or pmt_pos_id >= ev.npmt_per_mpmt:
# # # #             continue
# # # #         if not ev.mpmt_status[slot]:
# # # #             continue
# # # #         if not ev.pmt_status[slot][pmt_pos_id]:
# # # #             continue

# # # #         ev.hit_charges[slot][pmt_pos_id].append(float(sim_data[i, 1]))
# # # #         ev.hit_times[slot][pmt_pos_id].append(float(sim_data[i, 2]))

# # # #     if shift_times:
# # # #         bp_loc = np.array([0.0, 0.0, -1350.0])
# # # #         early_hits = []

# # # #         for i_mpmt in range(ev.n_mpmt):
# # # #             if not ev.mpmt_status[i_mpmt]:
# # # #                 continue
# # # #             for i_pmt in range(ev.npmt_per_mpmt):
# # # #                 if not ev.pmt_status[i_mpmt][i_pmt]:
# # # #                     continue
# # # #                 if len(ev.hit_times[i_mpmt][i_pmt]) == 0:
# # # #                     continue

# # # #                 pmt_loc = WCD.mpmts[i_mpmt].pmts[i_pmt].get_placement("est")["location"]
# # # #                 r = np.linalg.norm(pmt_loc - bp_loc)

# # # #                 for t in ev.hit_times[i_mpmt][i_pmt]:
# # # #                     early_hits.append({
# # # #                         "time": float(t),
# # # #                         "t0_est": float(t) - r / vw,
# # # #                     })

# # # #         if len(early_hits) > 0:
# # # #             early_hits = sorted(early_hits, key=lambda x: x["time"])
# # # #             n_use = min(n_earliest_for_t0, len(early_hits))
# # # #             time_offset = np.median([hit["t0_est"] for hit in early_hits[:n_use]])

# # # #             for i_mpmt in range(ev.n_mpmt):
# # # #                 for i_pmt in range(ev.npmt_per_mpmt):
# # # #                     ev.hit_times[i_mpmt][i_pmt] = [
# # # #                         t - time_offset for t in ev.hit_times[i_mpmt][i_pmt]
# # # #                     ]

# # # #             ev.global_time_offset = time_offset

# # # #     return ev, np.asarray(active_wcte_pmt_ids, dtype=int)


# # # # def build_observables_from_event(ev, pe_scale=143):
# # # #     obs_pes = []
# # # #     obs_ts = []

# # # #     for i_mpmt in range(ev.n_mpmt):
# # # #         if not ev.mpmt_status[i_mpmt]:
# # # #             continue
# # # #         for i_pmt in range(ev.npmt_per_mpmt):
# # # #             if not ev.pmt_status[i_mpmt][i_pmt]:
# # # #                 continue

# # # #             q = np.asarray(ev.hit_charges[i_mpmt][i_pmt], dtype=np.float64)
# # # #             t = np.asarray(ev.hit_times[i_mpmt][i_pmt], dtype=np.float64)

# # # #             if q.size == 0:
# # # #                 obs_pes.append(0.0)
# # # #                 obs_ts.append(np.nan)
# # # #             else:
# # # #                 obs_pes.append(float(np.sum(q)) / pe_scale)
# # # #                 obs_ts.append(float(np.sum(q * t) / np.sum(q)))

# # # #     return np.asarray(obs_pes, dtype=np.float64), np.asarray(obs_ts, dtype=np.float64)


# # # # def apply_ring_mask_to_observables(obs_pes, obs_ts, ring_keep_mask, mode="both"):
# # # #     obs_pes = obs_pes.copy()
# # # #     obs_ts = obs_ts.copy()

# # # #     if mode not in {"none", "pes", "ts", "both"}:
# # # #         raise ValueError("RING_MASK_MODE must be one of: none, pes, ts, both")

# # # #     if mode in {"pes", "both"}:
# # # #         obs_pes[~ring_keep_mask] = 0.0
# # # #     if mode in {"ts", "both"}:
# # # #         obs_ts[~ring_keep_mask] = np.nan

# # # #     return obs_pes, obs_ts


# # # # def get_t0_prior_sigma(obs_pes, obs_ts):
# # # #     n_timed = np.count_nonzero(np.isfinite(obs_ts))
# # # #     total_pe = np.sum(obs_pes)

# # # #     if (n_timed < 250) or (total_pe < 300):
# # # #         return 0.1
# # # #     elif (n_timed < 275) or (total_pe < 350):
# # # #         return 0.2
# # # #     elif (n_timed < 300) or (total_pe < 400):
# # # #         return 0.3
# # # #     elif (n_timed < 325) or (total_pe < 450):
# # # #         return 0.4
# # # #     elif (n_timed < 350) or (total_pe < 500):
# # # #         return 0.5
# # # #     elif (n_timed < 375) or (total_pe < 550):
# # # #         return 0.6
# # # #     elif (n_timed < 400) or (total_pe < 600):
# # # #         return 0.7
# # # #     elif (n_timed < 425) or (total_pe < 650):
# # # #         return 0.8
# # # #     elif (n_timed < 450) or (total_pe < 700):
# # # #         return 1.0
# # # #     elif (n_timed < 475) or (total_pe < 750):
# # # #         return 1.2
# # # #     elif (n_timed < 500) or (total_pe < 800):
# # # #         return 1.4
# # # #     elif (n_timed < 525) or (total_pe < 850):
# # # #         return 1.6
# # # #     elif (n_timed < 550) or (total_pe < 900):
# # # #         return 1.8
# # # #     else:
# # # #         return 2.0


# # # # # =============================================================================
# # # # # LIKELIHOOD EVALUATION
# # # # # =============================================================================
# # # # def get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts):
# # # #     exp_pes = np.asarray(exp_pes, dtype=np.float64)
# # # #     obs_pes = np.asarray(obs_pes, dtype=np.float64)
# # # #     exp_ts = np.asarray(exp_ts, dtype=np.float64)
# # # #     obs_ts = np.asarray(obs_ts, dtype=np.float64)

# # # #     mask = (
# # # #         (exp_pes > 0.0)
# # # #         & (obs_pes > 0.0)
# # # #         & np.isfinite(exp_ts)
# # # #         & np.isfinite(obs_ts)
# # # #     )

# # # #     if not np.any(mask):
# # # #         return 1e30

# # # #     sigma_t = PMT_MODEL.single_pe_time_std / np.sqrt(obs_pes[mask])
# # # #     dt = (obs_ts[mask] - exp_ts[mask]) / sigma_t
# # # #     return float(0.5 * np.sum(dt * dt))


# # # # def evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts):
# # # #     if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
# # # #         return PMT_MODEL.get_neg_log_likelihood_npe_t(exp_pes, obs_pes, exp_ts, obs_ts)
# # # #     if USE_CHARGE_LIKELIHOOD:
# # # #         return PMT_MODEL.get_neg_log_likelihood_npe(exp_pes, obs_pes)
# # # #     return get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts)


# # # # def evaluate_neg_log_likelihood(
# # # #     obs_pes,
# # # #     obs_ts,
# # # #     emitter,
# # # #     mpmt_types,
# # # #     x0,
# # # #     y0,
# # # #     z0,
# # # #     cx,
# # # #     cy,
# # # #     length_or_visible,
# # # #     full_range_or_t0,
# # # #     t0=None,
# # # # ):
# # # #     """Evaluate the selected fit mode.

# # # #     full_length mode receives: length_or_visible=length, full_range_or_t0=t0.
# # # #     absorption mode receives:  length_or_visible=visible_length, full_range_or_t0=full_range, t0=t0.
# # # #     """
# # # #     if IS_ABSORPTION_MODE:
# # # #         visible_length = float(length_or_visible)
# # # #         full_range = float(full_range_or_t0)
# # # #         t0 = float(t0)

# # # #         if not np.isfinite(visible_length) or not np.isfinite(full_range):
# # # #             return 1e30
# # # #         if visible_length < 0.0 or full_range <= 0.0:
# # # #             return 1e30
# # # #         if visible_length > full_range:
# # # #             return 1e30
# # # #         if full_range > float(RANGE_LOOKUP.overall_distances_mm[-1]):
# # # #             return 1e30

# # # #         ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))
# # # #         if (not np.isfinite(ke0)) or ke0 <= FIT_PARTICLE_THRESHOLD_MEV:
# # # #             return 1e30
# # # #         emitter.fixed_initial_KE = ke0
# # # #         track_length_for_emission = visible_length
# # # #     else:
# # # #         length = float(length_or_visible)
# # # #         t0 = float(full_range_or_t0)
# # # #         if not np.isfinite(length) or length < 0.0:
# # # #             return 1e30
# # # #         if length > float(RANGE_LOOKUP.overall_distances_mm[-1]):
# # # #             return 1e30
# # # #         emitter.fixed_initial_KE = None
# # # #         track_length_for_emission = length

# # # #     cz2 = 1.0 - cx * cx - cy * cy
# # # #     if cz2 <= 0.0:
# # # #         return 1e30

# # # #     cz = np.sqrt(cz2)
# # # #     emitter.start_coord = (float(x0), float(y0), float(z0))
# # # #     emitter.starting_time = float(t0)
# # # #     emitter.direction = (float(cx), float(cy), float(cz))

# # # #     init_ke = emitter.refresh_kinematics_from_length(track_length_for_emission)

# # # #     if hasattr(emitter, "visible_length_is_physical"):
# # # #         if not emitter.visible_length_is_physical():
# # # #             return 1e30
# # # #     elif getattr(emitter, "last_visible_length_exceeds_range", False):
# # # #         return 1e30

# # # #     s = emitter.get_emission_points(P_LOCATIONS, init_ke)
# # # #     exp_pes, exp_ts = emitter.get_expected_pes_ts(
# # # #         WCD,
# # # #         s,
# # # #         P_LOCATIONS,
# # # #         DIRECTION_ZS,
# # # #         mpmt_types,
# # # #         obs_pes,
# # # #     )

# # # #     nll = evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts)
# # # #     if not np.isfinite(nll):
# # # #         return 1e30

# # # #     if USE_TIMING_LIKELIHOOD and USE_T0_PRIOR:
# # # #         sigma_t0 = get_t0_prior_sigma(obs_pes, obs_ts)
# # # #         nll += abs(0.5 * (float(t0) / sigma_t0) ** 2)

# # # #     return float(nll)


# # # # def _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed):
# # # #     if IS_ABSORPTION_MODE:
# # # #         return evaluate_neg_log_likelihood(
# # # #             obs_pes, obs_ts, emitter, mpmt_types,
# # # #             seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
# # # #             seed["visible_length"], seed["full_range"], seed["t0"],
# # # #         )
# # # #     return evaluate_neg_log_likelihood(
# # # #         obs_pes, obs_ts, emitter, mpmt_types,
# # # #         seed["x0"], seed["y0"], seed["z0"], seed["cx"], seed["cy"],
# # # #         seed["length"], seed["t0"],
# # # #     )


# # # # def select_best_initial_seed(obs_pes, obs_ts, init_param_sets, mpmt_types=None):
# # # #     """
# # # #     Cheap deterministic seed prescan.

# # # #     For production fits, only the best seed is retained.  This is faster and
# # # #     avoids moving a huge seed-scan list between multiprocessing workers.  When
# # # #     rescue/debug output is enabled, the full sorted scan is kept.
# # # #     """
# # # #     best_info = None
# # # #     seed_scan = [] if NEED_FULL_SEED_SCAN else None

# # # #     for i, seed in enumerate(init_param_sets):
# # # #         emitter = EMITTER_TEMPLATE.copy()

# # # #         fval = _evaluate_seed_fcn(obs_pes, obs_ts, emitter, mpmt_types, seed)

# # # #         if not np.isfinite(fval):
# # # #             fval = np.inf

# # # #         info = {
# # # #             "seed_index": int(i),
# # # #             "fval": float(fval),
# # # #             "params": dict(seed),
# # # #         }

# # # #         if seed_scan is not None:
# # # #             seed_scan.append(info)

# # # #         if best_info is None or fval < best_info["fval"]:
# # # #             best_info = info

# # # #     if best_info is None or not np.isfinite(best_info["fval"]):
# # # #         raise RuntimeError("All seed FCNs were non-finite. Check ranges, PMT ordering, and event selection.")

# # # #     if seed_scan is not None:
# # # #         seed_scan_sorted = sorted(seed_scan, key=lambda x: x["fval"])
# # # #     else:
# # # #         seed_scan_sorted = [best_info]

# # # #     best = seed_scan_sorted[0]
# # # #     return dict(best["params"]), int(best["seed_index"]), float(best["fval"]), seed_scan_sorted


# # # # def compute_true_fcn_for_event(event_index):
# # # #     if IS_ABSORPTION_MODE:
# # # #         if not np.isfinite(TRUE_PARAMS.get("visible_length", np.nan)):
# # # #             return np.nan
# # # #         if not np.isfinite(TRUE_PARAMS.get("full_range", np.nan)):
# # # #             return np.nan
# # # #         length_args = (TRUE_PARAMS["visible_length"], TRUE_PARAMS["full_range"], TRUE_PARAMS["t0"])
# # # #     else:
# # # #         if not np.isfinite(TRUE_PARAMS.get("length", np.nan)):
# # # #             return np.nan
# # # #         length_args = (TRUE_PARAMS["length"], TRUE_PARAMS["t0"])

# # # #     mpmt_types = None
# # # #     emitter = EMITTER_TEMPLATE.copy()
# # # #     return evaluate_neg_log_likelihood(
# # # #         OBS_PES_ALL[event_index],
# # # #         OBS_TS_ALL[event_index],
# # # #         emitter,
# # # #         mpmt_types,
# # # #         TRUE_PARAMS["x0"],
# # # #         TRUE_PARAMS["y0"],
# # # #         TRUE_PARAMS["z0"],
# # # #         TRUE_PARAMS["cx"],
# # # #         TRUE_PARAMS["cy"],
# # # #         *length_args,
# # # #     )


# # # # # =============================================================================
# # # # # MINUIT HELPERS
# # # # # =============================================================================
# # # # def make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types=None):
# # # #     emitter = EMITTER_TEMPLATE.copy()

# # # #     if IS_ABSORPTION_MODE:
# # # #         def nll(x0, y0, z0, cx, cy, visible_length, full_range, t0):
# # # #             return evaluate_neg_log_likelihood(
# # # #                 obs_pes, obs_ts, emitter, mpmt_types,
# # # #                 x0, y0, z0, cx, cy, visible_length, full_range, t0,
# # # #             )
# # # #     else:
# # # #         def nll(x0, y0, z0, cx, cy, length, t0):
# # # #             return evaluate_neg_log_likelihood(
# # # #                 obs_pes, obs_ts, emitter, mpmt_types,
# # # #                 x0, y0, z0, cx, cy, length, t0,
# # # #             )

# # # #     m = Minuit(nll, **start_params)

# # # #     max_range = float(RANGE_LOOKUP.overall_distances_mm[-1])

# # # #     m.limits["x0"] = (-2000, 2000)
# # # #     m.limits["y0"] = (-2000, 2000)
# # # #     m.limits["z0"] = (-2000, 2000)
# # # #     m.limits["cx"] = (-0.5, 0.5)
# # # #     m.limits["cy"] = (-0.5, 0.5)
# # # #     m.limits["t0"] = (-8.0,8.0) #T0_LIMITS

# # # #     m.errors["x0"] = 100.0
# # # #     m.errors["y0"] = 100.0
# # # #     m.errors["z0"] = 100.0
# # # #     m.errors["cx"] = 0.01
# # # #     m.errors["cy"] = 0.01
# # # #     m.errors["t0"] = 0.1

# # # #     if IS_ABSORPTION_MODE:
# # # #         m.limits["visible_length"] = (0.0, 5000)
# # # #         m.limits["full_range"] = (1.0, 5000)
# # # #         m.errors["visible_length"] = 60.0
# # # #         m.errors["full_range"] = 100.0
# # # #     else:
# # # #         m.limits["length"] = (0.0, 3000)
# # # #         m.errors["length"] = 100.0

# # # #     if not USE_TIMING_LIKELIHOOD:
# # # #         m.fixed["t0"] = True

# # # #     m.errordef = Minuit.LIKELIHOOD
# # # #     m.strategy = M_STRAT

# # # #     return m


# # # # def is_bad_minuit_result(m, *, edm_max=1e10):
# # # #     if (m.fval is None) or (not np.isfinite(m.fval)):
# # # #         return True
# # # #     # Do not use m.valid as a bad-result criterion.
# # # #     try:
# # # #         if (m.fmin is not None) and np.isfinite(m.fmin.edm) and (m.fmin.edm > edm_max):
# # # #             return True
# # # #     except Exception:
# # # #         pass
# # # #     return False


# # # # def run_minuit_attempt(m, ncall):
# # # #     if not ENABLE_STAGE2_MIGRAD_FIRST:
# # # #         m.strategy = M_STRAT
# # # #         m.simplex(ncall=ncall)
# # # #         m.migrad(ncall=ncall)
# # # #         return m

# # # #     ncall_fast = max(2000, int(0.35 * ncall))
# # # #     ncall_simplex = max(2000, int(0.25 * ncall))

# # # #     m.strategy = 0
# # # #     m.migrad(ncall=ncall_fast)

# # # #     if is_bad_minuit_result(m):
# # # #         m.simplex(ncall=ncall_simplex)
# # # #         m.strategy = M_STRAT
# # # #         m.migrad(ncall=ncall)

# # # #     return m


# # # # # =============================================================================
# # # # # ADAPTIVE RESCUE
# # # # # =============================================================================
# # # # ENABLE_ADAPTIVE_RESCUE = ENABLE_STAGE3_ADAPTIVE_RESCUE
# # # # RESCUE_MAX_SEEDS = 6
# # # # RESCUE_LENGTH_BINS = [
# # # #     (0.0, 1000.0),
# # # #     (1000.0, 1250.0),
# # # #     (1250.0, 1400.0),
# # # #     (1400.0, 1700.0),
# # # #     (1700.0, 3000.0),
# # # # ]


# # # # def result_length_value(values):
# # # #     return float(values["visible_length"] if IS_ABSORPTION_MODE else values["length"])


# # # # def result_full_range_value(values):
# # # #     return float(values["full_range"] if IS_ABSORPTION_MODE else values["length"])


# # # # def seed_length_value(params):
# # # #     return float(params["visible_length"] if IS_ABSORPTION_MODE else params["length"])


# # # # def result_ke0_from_values(values):
# # # #     try:
# # # #         return float(RANGE_LOOKUP.range_mm_to_energy(result_full_range_value(values)))
# # # #     except Exception:
# # # #         return np.nan


# # # # def needs_rescue_result(result):
# # # #     if result is None:
# # # #         return True
# # # #     if not np.isfinite(result.get("fval", np.inf)):
# # # #         return True
# # # #     values = result.get("values", {})
# # # #     try:
# # # #         fitted_length = result_length_value(values)
# # # #         fitted_full = result_full_range_value(values)
# # # #     except Exception:
# # # #         return True
# # # #     if (not np.isfinite(fitted_length)) or (not np.isfinite(fitted_full)):
# # # #         return True
# # # #     if fitted_length <= 10.0 or fitted_length >= VISIBLE_LENGTH_RETRY_THRESHOLD:
# # # #         return True
# # # #     if IS_ABSORPTION_MODE and fitted_length > fitted_full:
# # # #         return True
# # # #     if result.get("seed_stuck", False):
# # # #         return True
# # # #     if USE_TIMING_LIKELIHOOD and result.get("below_t_min", False):
# # # #         return True
# # # #     return False


# # # # def choose_diverse_rescue_seed_infos(seed_scan_sorted, already_tried_seed_indices=None, max_total=RESCUE_MAX_SEEDS):
# # # #     already = set() if already_tried_seed_indices is None else set(already_tried_seed_indices)
# # # #     chosen = []

# # # #     for lo, hi in RESCUE_LENGTH_BINS:
# # # #         candidates = [
# # # #             s for s in seed_scan_sorted
# # # #             if int(s["seed_index"]) not in already
# # # #             and lo <= seed_length_value(s["params"]) < hi
# # # #         ]
# # # #         if candidates:
# # # #             chosen.append(candidates[0])
# # # #             already.add(int(candidates[0]["seed_index"]))
# # # #         if len(chosen) >= max_total:
# # # #             return chosen

# # # #     for s in seed_scan_sorted:
# # # #         idx = int(s["seed_index"])
# # # #         if idx in already:
# # # #             continue
# # # #         chosen.append(s)
# # # #         already.add(idx)
# # # #         if len(chosen) >= max_total:
# # # #             break

# # # #     return chosen


# # # # def compact_seed_scan(seed_scan_sorted):
# # # #     """Return the configured seed-scan payload for output/debugging."""
# # # #     if SAVE_SEED_SCAN:
# # # #         return seed_scan_sorted
# # # #     if SAVE_TOP_N_SEEDS > 0:
# # # #         return seed_scan_sorted[:SAVE_TOP_N_SEEDS]
# # # #     return []


# # # # def build_result_from_minuit(m, attempt, start_params, chosen_seed_idx, chosen_seed_fcn, seed_scan_sorted):
# # # #     current_fval = float(m.fval) if (m.fval is not None and np.isfinite(m.fval)) else np.inf
# # # #     current_values = m.values.to_dict()

# # # #     fitted_z0 = float(current_values["z0"])
# # # #     fitted_length = result_length_value(current_values)
# # # #     fitted_full = result_full_range_value(current_values)
# # # #     fitted_ke0 = result_ke0_from_values(current_values)

# # # #     visible_too_large = fitted_length > VISIBLE_LENGTH_RETRY_THRESHOLD
# # # #     z_near_seed = abs(fitted_z0 - float(start_params["z0"])) <= Z_SEED_EPS
# # # #     length_near_seed = abs(fitted_length - seed_length_value(start_params)) <= VISIBLE_LENGTH_SEED_EPS
# # # #     if IS_ABSORPTION_MODE:
# # # #         full_near_seed = abs(fitted_full - float(start_params["full_range"])) <= FULL_RANGE_SEED_EPS
# # # #     else:
# # # #         full_near_seed = False
# # # #     seed_stuck = z_near_seed and length_near_seed and (full_near_seed if IS_ABSORPTION_MODE else True)
# # # #     below_t_min = USE_TIMING_LIKELIHOOD and (current_values["t0"] < T_MIN)

# # # #     # Always expose consistent aliases in the result dictionary.
# # # #     visible_length_mm = fitted_length
# # # #     full_range_mm = fitted_full
# # # #     current_values.setdefault("length", fitted_length)
# # # #     current_values.setdefault("visible_length", visible_length_mm)
# # # #     current_values.setdefault("full_range", full_range_mm)

# # # #     return {
# # # #         "values": current_values,
# # # #         "errors": m.errors.to_dict(),
# # # #         "fval": current_fval,
# # # #         "valid": bool(m.valid),
# # # #         "attempts": attempt,
# # # #         "visible_length_too_large": bool(visible_too_large),
# # # #         "length_too_large": bool(visible_too_large),
# # # #         "seed_stuck": bool(seed_stuck),
# # # #         "z_near_seed": bool(z_near_seed),
# # # #         "visible_length_near_seed": bool(length_near_seed),
# # # #         "full_range_near_seed": bool(full_near_seed),
# # # #         "length_near_seed": bool(length_near_seed),
# # # #         "below_t_min": bool(below_t_min),
# # # #         "chosen_seed_index": int(chosen_seed_idx),
# # # #         "chosen_seed_fcn": float(chosen_seed_fcn) if np.isfinite(chosen_seed_fcn) else np.nan,
# # # #         "chosen_seed_params": dict(start_params),
# # # #         "seed_scan": compact_seed_scan(seed_scan_sorted),
# # # #         "visible_length_mm": visible_length_mm,
# # # #         "full_range_mm": full_range_mm,
# # # #         "length_mm": fitted_length,
# # # #         "ke0_mev": fitted_ke0,
# # # #         "edm": (
# # # #             float(m.fmin.edm)
# # # #             if (getattr(m, "fmin", None) is not None and m.fmin.edm is not None)
# # # #             else np.nan
# # # #         ),
# # # #     }


# # # # def result_sort_key(result):
# # # #     if result is None:
# # # #         return (999, np.inf)
# # # #     fval = float(result.get("fval", np.inf))
# # # #     penalty = 0
# # # #     if not np.isfinite(fval):
# # # #         penalty += 100
# # # #     if result.get("visible_length_too_large", False) or result.get("length_too_large", False):
# # # #         penalty += 10
# # # #     if result.get("seed_stuck", False):
# # # #         penalty += 5
# # # #     if result.get("below_t_min", False):
# # # #         penalty += 5
# # # #     return (penalty, fval)


# # # # # =============================================================================
# # # # # HARD-EVENT VISIBLE-LENGTH PROFILE RESCUE
# # # # # =============================================================================
# # # # ENABLE_LENGTH_PROFILE_RESCUE = ENABLE_STAGE4_LENGTH_PROFILE
# # # # LENGTH_PROFILE_GRID = list(FAST_SEED_VISIBLE_LENGTHS)
# # # # LENGTH_PROFILE_MAX_POINTS = 6


# # # # def run_length_profile_rescue(obs_pes, obs_ts, mpmt_types, seed_scan_sorted, ncall, starting_attempt_index=100):
# # # #     profile_results = []
# # # #     base_seed = dict(seed_scan_sorted[0]["params"])
# # # #     length_key = "visible_length" if IS_ABSORPTION_MODE else "length"

# # # #     for j, profile_length in enumerate(LENGTH_PROFILE_GRID[:LENGTH_PROFILE_MAX_POINTS]):
# # # #         start_params = dict(base_seed)
# # # #         start_params[length_key] = float(profile_length)
# # # #         if IS_ABSORPTION_MODE and start_params["visible_length"] > start_params["full_range"]:
# # # #             continue

# # # #         m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# # # #         m.fixed[length_key] = True
# # # #         run_minuit_attempt(m, max(5000, int(0.5 * ncall)))

# # # #         prof_result = build_result_from_minuit(
# # # #             m,
# # # #             attempt=starting_attempt_index + j,
# # # #             start_params=start_params,
# # # #             chosen_seed_idx=-1000 - j,
# # # #             chosen_seed_fcn=np.nan,
# # # #             seed_scan_sorted=seed_scan_sorted,
# # # #         )
# # # #         prof_result["profile_fixed_length"] = float(profile_length)
# # # #         prof_result["profile_fixed_visible_length"] = float(profile_length)
# # # #         profile_results.append(prof_result)

# # # #     if not profile_results:
# # # #         return None

# # # #     best_profile = min(profile_results, key=result_sort_key)

# # # #     polish_params = dict(best_profile["values"])
# # # #     # Keep only parameters actually used by this mode; Minuit will reject extras.
# # # #     polish_params = {k: polish_params[k] for k in PARAM_NAMES if k in polish_params}
# # # #     m = make_minuit_for_event(obs_pes, obs_ts, polish_params, mpmt_types)
# # # #     m.fixed[length_key] = False
# # # #     run_minuit_attempt(m, ncall)

# # # #     polish_result = build_result_from_minuit(
# # # #         m,
# # # #         attempt=starting_attempt_index + len(profile_results),
# # # #         start_params=polish_params,
# # # #         chosen_seed_idx=-2000,
# # # #         chosen_seed_fcn=float(best_profile["fval"]),
# # # #         seed_scan_sorted=seed_scan_sorted,
# # # #     )
# # # #     polish_result["length_profile_rescue_used"] = True
# # # #     polish_result["length_profile_results"] = profile_results
# # # #     polish_result["length_profile_best_fixed"] = best_profile
# # # #     return polish_result


# # # # def fit_one_event_by_index(args):
# # # #     event_index, init_param_sets, fcn_threshold, max_attempts, ncall = args

# # # #     obs_pes = OBS_PES_ALL[event_index]
# # # #     obs_ts = OBS_TS_ALL[event_index]
# # # #     mpmt_types = get_mpmt_slot_type(MPMT_SLOTS_ALL[event_index])

# # # #     best_seed, best_seed_idx, best_seed_fval, seed_scan_sorted = select_best_initial_seed(
# # # #         obs_pes,
# # # #         obs_ts,
# # # #         init_param_sets,
# # # #         mpmt_types,
# # # #     )

# # # #     attempt_results = []
# # # #     tried_seed_indices = set()

# # # #     primary_info = seed_scan_sorted[0]
# # # #     start_params = dict(primary_info["params"])
# # # #     chosen_seed_idx = int(primary_info["seed_index"])
# # # #     chosen_seed_fcn = float(primary_info["fval"])
# # # #     tried_seed_indices.add(chosen_seed_idx)

# # # #     m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# # # #     run_minuit_attempt(m, ncall)
# # # #     primary_result = build_result_from_minuit(
# # # #         m,
# # # #         attempt=1,
# # # #         start_params=start_params,
# # # #         chosen_seed_idx=chosen_seed_idx,
# # # #         chosen_seed_fcn=chosen_seed_fcn,
# # # #         seed_scan_sorted=seed_scan_sorted,
# # # #     )
# # # #     attempt_results.append(primary_result)

# # # #     if ENABLE_ADAPTIVE_RESCUE and needs_rescue_result(primary_result):
# # # #         rescue_seed_infos = choose_diverse_rescue_seed_infos(
# # # #             seed_scan_sorted,
# # # #             already_tried_seed_indices=tried_seed_indices,
# # # #             max_total=RESCUE_MAX_SEEDS,
# # # #         )

# # # #         for rescue_i, seed_info in enumerate(rescue_seed_infos, start=2):
# # # #             start_params = dict(seed_info["params"])
# # # #             chosen_seed_idx = int(seed_info["seed_index"])
# # # #             chosen_seed_fcn = float(seed_info["fval"])
# # # #             tried_seed_indices.add(chosen_seed_idx)

# # # #             m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)
# # # #             run_minuit_attempt(m, ncall)
# # # #             result = build_result_from_minuit(
# # # #                 m,
# # # #                 attempt=rescue_i,
# # # #                 start_params=start_params,
# # # #                 chosen_seed_idx=chosen_seed_idx,
# # # #                 chosen_seed_fcn=chosen_seed_fcn,
# # # #                 seed_scan_sorted=seed_scan_sorted,
# # # #             )
# # # #             attempt_results.append(result)

# # # #     best_result = min(attempt_results, key=result_sort_key)

# # # #     if ENABLE_LENGTH_PROFILE_RESCUE and needs_rescue_result(best_result):
# # # #         profile_result = run_length_profile_rescue(
# # # #             obs_pes,
# # # #             obs_ts,
# # # #             mpmt_types,
# # # #             seed_scan_sorted,
# # # #             ncall,
# # # #             starting_attempt_index=100 + len(attempt_results),
# # # #         )
# # # #         if profile_result is not None:
# # # #             attempt_results.append(profile_result)
# # # #             best_result = min(attempt_results, key=result_sort_key)

# # # #     best_result["attempts"] = len(attempt_results)
# # # #     if SAVE_ATTEMPT_RESULTS:
# # # #         best_result["attempt_results"] = attempt_results
# # # #     else:
# # # #         best_result["attempt_results"] = []
# # # #     best_result["adaptive_rescue_used"] = bool(len(attempt_results) > 1)
# # # #     best_result["length_profile_rescue_considered"] = bool(ENABLE_LENGTH_PROFILE_RESCUE)
# # # #     best_result["length_profile_rescue_used"] = bool(
# # # #         best_result.get("length_profile_rescue_used", False)
# # # #         or any(r.get("length_profile_rescue_used", False) for r in attempt_results)
# # # #     )
# # # #     return best_result


# # # # def run_batch(event_indices, init_param_sets, nproc, fcn_threshold, max_attempts, ncall):
# # # #     args = [(idx, init_param_sets, fcn_threshold, max_attempts, ncall) for idx in event_indices]

# # # #     try:
# # # #         ctx = mp.get_context("fork")
# # # #     except ValueError:
# # # #         ctx = mp.get_context()

# # # #     with ctx.Pool(processes=nproc) as pool:
# # # #         return pool.map(fit_one_event_by_index, args)


# # # # # =============================================================================
# # # # # USER-PROVIDED EVENT FILE HELPERS
# # # # # =============================================================================
# # # # def _coerce_event_array(event, *, event_label="event"):
# # # #     arr = np.asarray(event)
# # # #     if arr.ndim != 2 or arr.shape[1] < 3:
# # # #         raise ValueError(
# # # #             f"{event_label} must be a 2D array with at least 3 columns: "
# # # #             "[pmt_id, charge, time]."
# # # #         )
# # # #     # Keep optional event-number columns, but the fitter only consumes columns 0:3.
# # # #     return np.asarray(arr[:, :4] if arr.shape[1] >= 4 else arr[:, :3], dtype=np.float64)


# # # # def _events_from_loaded_object(obj):
# # # #     """Normalize npy/npz/pickle payloads into a list of event arrays."""
# # # #     if isinstance(obj, dict):
# # # #         if USER_EVENT_KEY is not None:
# # # #             obj = obj[USER_EVENT_KEY]
# # # #         elif "events" in obj:
# # # #             obj = obj["events"]
# # # #         elif "data" in obj:
# # # #             obj = obj["data"]
# # # #         elif "arr_0" in obj:
# # # #             obj = obj["arr_0"]
# # # #         else:
# # # #             keys = ", ".join(map(str, obj.keys()))
# # # #             raise KeyError(
# # # #                 "Could not choose an event array from the dict payload. "
# # # #                 f"Available keys: {keys}. Set USER_EVENT_KEY."
# # # #             )

# # # #     if isinstance(obj, np.lib.npyio.NpzFile):
# # # #         if USER_EVENT_KEY is not None:
# # # #             key = USER_EVENT_KEY
# # # #         elif "events" in obj.files:
# # # #             key = "events"
# # # #         elif "data" in obj.files:
# # # #             key = "data"
# # # #         elif "arr_0" in obj.files:
# # # #             key = "arr_0"
# # # #         elif len(obj.files) == 1:
# # # #             key = obj.files[0]
# # # #         else:
# # # #             raise KeyError(
# # # #                 "Could not choose an event array from the npz payload. "
# # # #                 f"Available keys: {obj.files}. Set USER_EVENT_KEY."
# # # #             )
# # # #         obj = obj[key]

# # # #     if isinstance(obj, (list, tuple)):
# # # #         return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(obj)]

# # # #     arr = np.asarray(obj, dtype=object if getattr(obj, "dtype", None) == object else None)

# # # #     # Object arrays are normally lists of variable-length events.
# # # #     if arr.dtype == object and arr.ndim == 1:
# # # #         return [_coerce_event_array(ev, event_label=f"event[{i}]") for i, ev in enumerate(arr)]

# # # #     # A 3D numeric array is N_events x N_hits x N_columns.
# # # #     if arr.ndim == 3:
# # # #         return [_coerce_event_array(arr[i], event_label=f"event[{i}]") for i in range(arr.shape[0])]

# # # #     # A 2D array with a 4th column is interpreted as a concatenated event table
# # # #     # grouped by event number.  A 2D array with only 3 columns is one event.
# # # #     if arr.ndim == 2:
# # # #         arr2 = np.asarray(arr, dtype=np.float64)
# # # #         if arr2.shape[1] >= 4:
# # # #             events = []
# # # #             event_numbers = arr2[:, 3].astype(np.int64)
# # # #             for evnum in np.unique(event_numbers):
# # # #                 events.append(_coerce_event_array(arr2[event_numbers == evnum], event_label=f"event_number={evnum}"))
# # # #             return events
# # # #         return [_coerce_event_array(arr2, event_label="single_event")]

# # # #     raise ValueError(
# # # #         "Unsupported USER_EVENT_FILE payload. Expected a list/object array of "
# # # #         "event arrays, a 3D event array, or a 2D [pmt_id, charge, time] table."
# # # #     )


# # # # def load_user_event_file(path, *, max_events=None):
# # # #     """Load user-provided, already-selected events from npy/npz/pickle files."""
# # # #     path = Path(path)
# # # #     if not path.exists():
# # # #         raise FileNotFoundError(f"USER_EVENT_FILE does not exist: {path}")

# # # #     suffix = path.suffix.lower()
# # # #     if suffix == ".npz":
# # # #         loaded = np.load(path, allow_pickle=True)
# # # #     elif suffix == ".npy":
# # # #         loaded = np.load(path, allow_pickle=True)
# # # #     elif suffix in {".pkl", ".pickle"}:
# # # #         with open(path, "rb") as f:
# # # #             loaded = pickle.load(f)
# # # #     else:
# # # #         raise ValueError(
# # # #             f"Unsupported USER_EVENT_FILE suffix {suffix!r}. Use .npy, .npz, .pkl, or .pickle."
# # # #         )

# # # #     events = _events_from_loaded_object(loaded)
# # # #     if max_events is not None:
# # # #         events = events[: int(max_events)]
# # # #     print(f"Loaded {len(events)} user-provided events from: {path}")
# # # #     return events


# # # # # =============================================================================
# # # # # MAIN DRIVER
# # # # # =============================================================================
# # # # def main():
# # # #     global OVERALL_DISTANCES, INIT_ENERGY_TABLE, RANGE_LOOKUP
# # # #     global WCD, PMT_MODEL, EMITTER_TEMPLATE, P_LOCATIONS, DIRECTION_ZS, RING_KEEP_MASK, CORR_POS
# # # #     global OBS_PES_ALL, OBS_TS_ALL, MPMT_SLOTS_ALL, GOOD_WCTE_PMTS_SET

# # # #     print("Likelihood mode:", LIKELIHOOD_MODE)
# # # #     print("Fit particle:", FIT_PARTICLE_CANONICAL)
# # # #     print("Particle mass [MeV]:", FIT_PARTICLE_MASS_MEV)
# # # #     print("Cherenkov threshold [MeV]:", FIT_PARTICLE_THRESHOLD_MEV)
# # # #     print("Fit mode:", TRACK_END_MODE)
# # # #     print("Fit parameters:", FIT_PARAMETER_NAMES)
# # # #     print("Output file:", OUTPUT_FILE)

# # # #     if EVENT_SOURCE == "selection" and get_selected_events is None:
# # # #         raise ImportError(
# # # #             "event_loader.py was not found. Copy it into LF_multiParticles/scripts "
# # # #             "or add its directory to PYTHONPATH, or set EVENT_SOURCE=file and USER_EVENT_FILE."
# # # #         )

# # # #     if EVENT_SOURCE == "file" and not USER_EVENT_FILE:
# # # #         raise ValueError("EVENT_SOURCE=file requires USER_EVENT_FILE=/path/to/events.npy|npz|pkl")

# # # #     GOOD_WCTE_PMTS_SET = load_good_wcte_pmts()

# # # #     RANGE_LOOKUP = ParticleRangeLookup(FIT_PARTICLE_CANONICAL, table_dirs=[str(TABLE_DIR)])
# # # #     print("Range table max KE [MeV]:", float(RANGE_LOOKUP.initial_energies_mev[-1]))
# # # #     print("Range table max full_range [mm]:", float(RANGE_LOOKUP.overall_distances_mm[-1]))

# # # #     configure_truth_params()
# # # #     if IS_ABSORPTION_MODE:
# # # #         truth_ready = np.isfinite(TRUE_PARAMS["visible_length"]) and np.isfinite(TRUE_PARAMS["full_range"])
# # # #         if truth_ready:
# # # #             print("Truth visible length [mm]:", TRUE_PARAMS["visible_length"])
# # # #             print("Truth full range [mm]:", TRUE_PARAMS["full_range"])
# # # #             print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["full_range"]))
# # # #         else:
# # # #             print("Truth FCN disabled: set TRUE_VISIBLE_LENGTH_MM and TRUE_FULL_RANGE_MM/TRUE_INITIAL_KE_MEV.")
# # # #     else:
# # # #         truth_ready = np.isfinite(TRUE_PARAMS["length"])
# # # #         if truth_ready:
# # # #             print("Truth length [mm]:", TRUE_PARAMS["length"])
# # # #             print("Truth KE0 [MeV]:", RANGE_LOOKUP.range_mm_to_energy(TRUE_PARAMS["length"]))
# # # #         else:
# # # #             print("Truth FCN disabled: set TRUE_LENGTH_MM or TRUE_INITIAL_KE_MEV.")

# # # #     init_param_sets = build_fast_seed_grid(RANGE_LOOKUP)
# # # #     if not init_param_sets:
# # # #         raise RuntimeError("Seed grid is empty. Check FAST_SEED_VISIBLE_LENGTHS and FAST_SEED_KE0_MEV/FULL_RANGES.")
# # # #     print("Number of initial seeds:", len(init_param_sets))

# # # #     for i, seed in enumerate(init_param_sets):
# # # #         missing = [k for k in PARAM_NAMES if k not in seed]
# # # #         if missing:
# # # #             raise ValueError(f"Seed {i} is missing keys: {missing}")

# # # #     set_active_particle(FIT_PARTICLE_CANONICAL)
# # # #     OVERALL_DISTANCES, INIT_ENERGY_TABLE, _distance_rows = get_energy_distance_tables(
# # # #         FIT_PARTICLE_CANONICAL
# # # #     )

# # # #     hall = Device.open_file(GEOMETRY_FILE)
# # # #     WCD = hall.wcds[0]

# # # #     initial_ke_seed = float(RANGE_LOOKUP.range_mm_to_energy(
# # # #         min(1000.0, float(RANGE_LOOKUP.overall_distances_mm[-1]))
# # # #     ))

# # # #     emitter_model = Emitter(
# # # #         0.0,
# # # #         (0.0, 0.0, 0.0),
# # # #         (0.0, 0.0, 1.0),
# # # #         0.96,
# # # #         500.0,
# # # #         18.0,
# # # #         particle=FIT_PARTICLE_CANONICAL,
# # # #         track_end_mode=EMITTER_TRACK_END_MODE,
# # # #         fixed_initial_KE=initial_ke_seed if IS_ABSORPTION_MODE else None,
# # # #     )

# # # #     delta_pdf_path = TABLE_DIR / "delta_e_angular_pdf_table.npz"
# # # #     if delta_pdf_path.exists() and hasattr(emitter_model, "load_delta_e_angular_pdf_table"):
# # # #         emitter_model.load_delta_e_angular_pdf_table(str(delta_pdf_path))

# # # #     PMT_MODEL = PMT(1.0, 0.3, 1.0, 40.0, 0.2, 0.0)
# # # #     EMITTER_TEMPLATE = emitter_model.copy()
# # # #     CORR_POS = None

# # # #     print("Building event observables...")

# # # #     obs_pes_all = []
# # # #     obs_ts_all = []
# # # #     mpmt_slots_all = []

# # # #     if EVENT_SOURCE == "selection":
# # # #         events = get_selected_events(
# # # #             RUN,
# # # #             N_EVENTS,
# # # #             particle=PARTICLE_SELECTION_LABEL,
# # # #             root_file=CONFIG_ROOT_FILE,
# # # #             use_peak_time_cut=USE_PEAK_TIME_CUT,
# # # #             peak_window=PEAK_WINDOW_NS,
# # # #             peak_bin_width=PEAK_BIN_WIDTH_NS,
# # # #             tof_primary=SELECTION_TOF_NS,
# # # #             tof_window=SELECTION_TOF_WINDOW_NS,
# # # #             tof_scalar_field=SELECTION_TOF_FIELD,
# # # #             momentum_scalar_field=SELECTION_MOMENTUM_FIELD,
# # # #             t5_particle_nr=SELECTION_T5_PARTICLE_NR,
# # # #         )
# # # #     else:
# # # #         events = load_user_event_file(USER_EVENT_FILE, max_events=N_EVENTS)

# # # #     tot_events = len(events)
# # # #     print("Total Events to Fit:", tot_events)

# # # #     for i in range(tot_events):
# # # #         event = np.asarray(events[i])
# # # #         if event.size == 0:
# # # #             continue

# # # #         apply_event_time_window = (EVENT_SOURCE == "selection") or USER_EVENT_APPLY_PEAK_WINDOW
# # # #         if apply_event_time_window:
# # # #             time_hist = np.histogram(event[:, 2], bins=np.arange(0, 4000))
# # # #             max_idx = int(np.argmax(time_hist[0]))
# # # #             lo_idx = max(0, max_idx - 20)
# # # #             hi_idx = min(len(time_hist[1]) - 1, max_idx + 5)
# # # #             min_time = time_hist[1][lo_idx]
# # # #             cut_time = time_hist[1][hi_idx]
# # # #             time_mask = (event[:, 2] > min_time) & (event[:, 2] < cut_time)
# # # #             event = event[time_mask]

# # # #         ev, pmt_ids = sim_to_event(event, WCD, n_mpmt_total=106, pe_scale=143)

# # # #         if P_LOCATIONS is None or DIRECTION_ZS is None:
# # # #             P_LOCATIONS, DIRECTION_ZS, MPMT_SLOTS = EMITTER_TEMPLATE.get_pmt_placements(ev, WCD, "est")
# # # #             MPMT_SLOTS = np.asarray(MPMT_SLOTS, dtype=int)
# # # #             RING_KEEP_MASK = np.isin(MPMT_SLOTS, ALL_RING)

# # # #         obs_pes, obs_ts = build_observables_from_event(ev, pe_scale=143)
# # # #         obs_pes, obs_ts = apply_ring_mask_to_observables(
# # # #             obs_pes,
# # # #             obs_ts,
# # # #             RING_KEEP_MASK,
# # # #             mode=RING_MASK_MODE,
# # # #         )

# # # #         obs_pes_all.append(obs_pes)
# # # #         obs_ts_all.append(obs_ts)
# # # #         mpmt_slots_all.append(MPMT_SLOTS)

# # # #     OBS_PES_ALL = obs_pes_all
# # # #     OBS_TS_ALL = obs_ts_all
# # # #     MPMT_SLOTS_ALL = mpmt_slots_all
# # # #     tot_events = len(OBS_PES_ALL)

# # # #     print("Computing truth FCNs...")
# # # #     true_fcn_all = [compute_true_fcn_for_event(i) for i in range(tot_events)]

# # # #     est_dict = {
# # # #         "metadata": {
# # # #             "fit_particle": FIT_PARTICLE_CANONICAL,
# # # #             "particle_mass_mev": FIT_PARTICLE_MASS_MEV,
# # # #             "particle_threshold_mev": FIT_PARTICLE_THRESHOLD_MEV,
# # # #             "beam_p": BEAM_P,
# # # #             "track_end_mode": TRACK_END_MODE,
# # # #             "fit_parameters": list(FIT_PARAMETER_NAMES),
# # # #             "truth_params": dict(TRUE_PARAMS),
# # # #             "geometry_file": GEOMETRY_FILE,
# # # #             "config_root_file": CONFIG_ROOT_FILE,
# # # #             "event_source": EVENT_SOURCE,
# # # #             "user_event_file": USER_EVENT_FILE if EVENT_SOURCE == "file" else None,
# # # #             "user_event_key": USER_EVENT_KEY if EVENT_SOURCE == "file" else None,
# # # #             "user_event_apply_peak_window": bool(USER_EVENT_APPLY_PEAK_WINDOW) if EVENT_SOURCE == "file" else None,
# # # #             "particle_selection_label": PARTICLE_SELECTION_LABEL,
# # # #             "selection_tof_ns": SELECTION_TOF_NS,
# # # #             "selection_tof_window_ns": SELECTION_TOF_WINDOW_NS,
# # # #             "selection_tof_field": SELECTION_TOF_FIELD,
# # # #             "selection_t5_particle_nr": SELECTION_T5_PARTICLE_NR,
# # # #             "range_table_max_full_range_mm": float(RANGE_LOOKUP.overall_distances_mm[-1]),
# # # #             "save_seed_scan": bool(SAVE_SEED_SCAN),
# # # #             "save_top_n_seeds": int(SAVE_TOP_N_SEEDS),
# # # #             "save_attempt_results": bool(SAVE_ATTEMPT_RESULTS),
# # # #         },
# # # #         "minimum_found": [],
# # # #         "x": [],
# # # #         "y": [],
# # # #         "z": [],
# # # #         "visible_length": [],
# # # #         "full_range": [],
# # # #         "ke0": [],
# # # #         "length": [],  # legacy alias for visible_length
# # # #         "t": [],
# # # #         "est_fcn": [],
# # # #         "true_fcn": [],
# # # #         "cx": [],
# # # #         "cy": [],
# # # #         "n_attempts": [],
# # # #         "chosen_seed_idx": [],
# # # #         "chosen_seed_fcn": [],
# # # #         "chosen_seed_params": [],
# # # #         "adaptive_rescue_used": [],
# # # #         "length_profile_rescue_considered": [],
# # # #         "length_profile_rescue_used": [],
# # # #         "edm": [],
# # # #     }
# # # #     if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
# # # #         est_dict["seed_scan"] = []

# # # #     if SAVE_ATTEMPT_RESULTS:
# # # #         est_dict["attempt_results"] = []

# # # #     n_events_per_batch = min(N_EVENTS_PER_BATCH, max(1, tot_events))

# # # #     for batch_start in range(0, tot_events, n_events_per_batch):
# # # #         batch_end = min(batch_start + n_events_per_batch, tot_events)
# # # #         event_indices = list(range(batch_start, batch_end))

# # # #         print(f"Starting event number {batch_start}")

# # # #         results = run_batch(
# # # #             event_indices=event_indices,
# # # #             init_param_sets=init_param_sets,
# # # #             nproc=NPROC,
# # # #             fcn_threshold=FCN_RETRY_THRESHOLD,
# # # #             max_attempts=MAX_FIT_ATTEMPTS,
# # # #             ncall=NCALL_MIGRAD,
# # # #         )

# # # #         for local_i, result in enumerate(results):
# # # #             event_index = event_indices[local_i]
# # # #             vals = result["values"]

# # # #             if IS_ABSORPTION_MODE:
# # # #                 visible_length = float(vals["visible_length"])
# # # #                 full_range = float(vals["full_range"])
# # # #             else:
# # # #                 visible_length = float(vals["length"])
# # # #                 full_range = visible_length
# # # #             ke0 = float(RANGE_LOOKUP.range_mm_to_energy(full_range))

# # # #             est_dict["minimum_found"].append(int(result["valid"]))
# # # #             est_dict["x"].append(vals["x0"])
# # # #             est_dict["y"].append(vals["y0"])
# # # #             est_dict["z"].append(vals["z0"])
# # # #             est_dict["visible_length"].append(visible_length)
# # # #             est_dict["full_range"].append(full_range)
# # # #             est_dict["ke0"].append(ke0)
# # # #             est_dict["length"].append(visible_length)
# # # #             est_dict["t"].append(vals["t0"])
# # # #             est_dict["cx"].append(vals["cx"])
# # # #             est_dict["cy"].append(vals["cy"])
# # # #             est_dict["est_fcn"].append(result["fval"])
# # # #             est_dict["true_fcn"].append(true_fcn_all[event_index])
# # # #             est_dict["n_attempts"].append(result["attempts"])
# # # #             est_dict["chosen_seed_idx"].append(result["chosen_seed_index"])
# # # #             est_dict["chosen_seed_fcn"].append(result["chosen_seed_fcn"])
# # # #             est_dict["chosen_seed_params"].append(result["chosen_seed_params"])
# # # #             if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
# # # #                 est_dict["seed_scan"].append(result.get("seed_scan", []))
# # # #             est_dict["adaptive_rescue_used"].append(result.get("adaptive_rescue_used", False))
# # # #             est_dict["length_profile_rescue_considered"].append(result.get("length_profile_rescue_considered", False))
# # # #             est_dict["length_profile_rescue_used"].append(result.get("length_profile_rescue_used", False))
# # # #             est_dict["edm"].append(result.get("edm", np.nan))
# # # #             if SAVE_ATTEMPT_RESULTS:
# # # #                 est_dict["attempt_results"].append(result.get("attempt_results", []))

# # # #     Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
# # # #     with open(OUTPUT_FILE, "wb") as f:
# # # #         pickle.dump(est_dict, f)

# # # #     print("Done.")
# # # #     print("Saved:", OUTPUT_FILE)


# # # # if __name__ == "__main__":
# # # #     main()

