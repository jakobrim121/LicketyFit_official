# 7/8-parameter batch driver for LicketyFit particle-track fits on WCSim data.
#
# This version keeps all user-facing configuration in one block near the top.
# The fitting mechanics below are intentionally kept separate from the settings
# users are most likely to change.

"""Toggleable multi-stage batch driver for Minuit Cherenkov fits on WCSim data.

WCSim-specific pieces kept:
  - INPUT_FILE is read with read_sim_data(INPUT_FILE)
  - WCSim PMT IDs are converted to WCTE PMT IDs if a mapping file is provided
  - WCSim charges use pe_scale=1.0
  - PMT placements use the design geometry
  - no real-data mPMT efficiency correction is applied; mpmt_types remains None

Supported track-end modes:
  - full_length : original 7-parameter fit
      x0, y0, z0, cx, cy, length, t0

  - absorption  : abrupt-endpoint 8-parameter fit
      x0, y0, z0, cx, cy, visible_length, full_range, t0

In absorption mode:
  visible_length = actual visible primary-Cherenkov length before abrupt cutoff [mm]
  full_range     = dE/dx-only range to Cherenkov threshold if no abrupt interaction [mm]
  ke0            = inferred initial kinetic energy from full_range [MeV]
"""

import os
import sys
import pickle
import multiprocessing as mp
from pathlib import Path

import numpy as np
from iminuit import Minuit


# =============================================================================
# ENVIRONMENT PARSING HELPERS
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


def _parse_int_list_env(name, default):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return list(default)
    return [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]


# =============================================================================
# USER CONFIGURATION: EDIT THIS BLOCK FIRST
# =============================================================================
# The values below can be changed directly in this file.  Almost every setting
# also has an environment-variable override, which is useful for batch jobs:
#
#   WCSIM_INPUT_FILE=/path/to/events.npz FIT_PARTICLE=proton FIT_MODE=absorption python batch_fit_driver_wcsim.py
#
# If you are handing this script to someone else, this is the only section they
# should normally need to inspect.

# -----------------------------------------------------------------------------
# 1) WCSim input, event count, and output
# -----------------------------------------------------------------------------
# Nominal kinetic energy of the WCSim sample.  This is used for default filenames,
# truth diagnostics, and the default fixed KE0 in absorption mode.
ENERGY_TRUE = _env_float("ENERGY_TRUE", 1000.0)

# Maximum number of WCSim events to fit from the input file.
TOT_EVENTS = _env_int("TOT_EVENTS", 1000)

# Number of fitted events per multiprocessing batch.
N_EVENTS_PER_BATCH = _env_int("N_EVENTS_PER_BATCH", 100)

# WCSim file-label settings.  FIT_PARTICLE controls the physics hypothesis;
# WCSIM_PARTICLE_LABEL controls the default file/directory label only.
WCSIM_PARTICLE_LABEL = os.environ.get(
    "WCSIM_PARTICLE_LABEL",
    os.environ.get("FIT_PARTICLE_STR", "p+"),
).strip()
WCSIM_PARTICLE_DIR = os.environ.get("WCSIM_PARTICLE_DIR", WCSIM_PARTICLE_LABEL).strip()

# Base WCSim data area used only if you choose the generated default input path.
WCSIM_DATA_PATH = os.environ.get(
    "WCSIM_DATA_PATH",
    "/eos/user/j/jrimmer/sim_work_dir/WCSim/sim_data/",
).strip()

# Default input file.  
# To use the generated path instead, set DEFAULT_WCSIM_INPUT_FILE to something like:
#   str(Path(WCSIM_DATA_PATH) / WCSIM_PARTICLE_DIR / f"1k{WCSIM_PARTICLE_LABEL}_{int(ENERGY_TRUE)}MeV_x0y0zn1350.npz")
DEFAULT_WCSIM_INPUT_FILE = os.environ.get(
    "DEFAULT_WCSIM_INPUT_FILE",
    "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit_official/work_dir/"
    "sample_data/1kp+_1000MeV_x0y0zn1350_fixed_single_ring_218events.npz",
).strip()

# Leave blank to use DEFAULT_WCSIM_INPUT_FILE.
INPUT_FILE_OVERRIDE = os.environ.get("WCSIM_INPUT_FILE", "").strip()

# Leave blank to use the automatic output filename under outputs/.
OUTPUT_FILE_OVERRIDE = os.environ.get("LF_OUTPUT_FILE", "").strip()


# -----------------------------------------------------------------------------
# 2) Main physics choices
# -----------------------------------------------------------------------------
# Particle hypothesis used by the likelihood and range table.
# Common choices depend on your particle_cherenkov_model.py support, e.g.:
#   "muon", "pion", "kaon", "proton", "p+"
FIT_PARTICLE = os.environ.get("FIT_PARTICLE", WCSIM_PARTICLE_LABEL).strip()

# Track-end / parameterization mode:
#   "full_length" : original 7-parameter fit; length determines initial KE
#   "absorption"  : 8-parameter fit; visible_length and full_range float separately
FIT_MODE_REQUEST = os.environ.get(
    "FIT_MODE",
    os.environ.get("TRACK_END_MODE", "absorption"),
).strip().lower()

# Likelihood mode:
#   "charge_time" : use both charge and timing
#   "charge_only" : use charge likelihood only
#   "timing_only" : use timing likelihood only
#
# Legacy environment variables USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD
# are still supported and override this single-string setting if provided.
LIKELIHOOD_MODE_REQUEST = os.environ.get(
    "LIKELIHOOD_MODE",
    os.environ.get("FIT_TYPE", "charge_time"),
).strip().lower()

USE_T0_PRIOR = _env_bool("USE_T0_PRIOR", False)

# Minuit t0 bounds and retry lower-bound diagnostic.
T0_LIMITS = (
    _env_float("T0_MIN", -3.0),
    _env_float("T0_MAX", 3.0),
)
T_MIN = _env_float("T_MIN", 0.0)


# -----------------------------------------------------------------------------
# 3) Fixed parameters
# -----------------------------------------------------------------------------
# Put a number next to any parameter you want Minuit to hold fixed.
# Leave a value as None to let Minuit float it normally.
#
# Full-length mode parameters:
#   x0, y0, z0, cx, cy, length, t0
#
# Absorption mode parameters:
#   x0, y0, z0, cx, cy, visible_length, full_range, t0
#
# Absorption mode also supports fixing the initial kinetic energy through
# ke0_mev.  This is converted internally to full_range using the same particle
# range table used by the likelihood.  Do not fix both full_range and ke0_mev
# unless they intentionally describe the same range.
#
# WCSim default here preserves your current behavior: z0 is fixed at -1350 mm,
# and ke0_mev is fixed to ENERGY_TRUE in absorption mode.  In full_length mode,
# ke0_mev defaults to None because length itself determines the initial KE.
_DEFAULT_FIXED_KE0_MEV = (
    ENERGY_TRUE
    if FIT_MODE_REQUEST in {"absorption", "absorbed", "abrupt", "abrupt_8param", "interaction", "truncated"}
    else None
)

FIXED_FIT_PARAMS = {
    "x0": _env_optional_float("FIX_X0", None),
    "y0": _env_optional_float("FIX_Y0", None),
    "z0": _env_optional_float("FIX_Z0", -1350.0),
    "cx": _env_optional_float("FIX_CX", None),
    "cy": _env_optional_float("FIX_CY", None),

    # full_length mode only
    "length": _env_optional_float("FIX_LENGTH", None),

    # absorption mode only
    "visible_length": _env_optional_float("FIX_VISIBLE_LENGTH", None),
    "full_range": _env_optional_float("FIX_FULL_RANGE", None),
    "ke0_mev": _env_optional_float(
        "FIXED_KE0_MEV",
        _env_optional_float("FIX_KE0_MEV", _DEFAULT_FIXED_KE0_MEV),
    ),

    "t0": _env_optional_float("FIX_T0", None),
}

# Convenience alias used later for printing and metadata.
FIXED_KE0_MEV = FIXED_FIT_PARAMS["ke0_mev"]


# -----------------------------------------------------------------------------
# 4) Fit controls and retry/rescue behavior
# -----------------------------------------------------------------------------
NPROC = _env_int("NPROC", 16)
M_STRAT = _env_int("M_STRAT", 1)

# The default retry path runs when the current best FCN is non-finite or above
# this threshold.  It does not depend on m.valid.
FCN_RETRY_THRESHOLD = _env_float("FCN_RETRY_THRESHOLD", 1000.0)
MAX_FIT_ATTEMPTS = _env_int("MAX_FIT_ATTEMPTS", 3)
NCALL_MIGRAD = _env_int("NCALL_MIGRAD", 70000)
NCALL_SIMPLEX = _env_int("NCALL_SIMPLEX", NCALL_MIGRAD)

# Optional additional search stages.
ENABLE_STAGE1_SEED_GRID = _env_bool("ENABLE_STAGE1_SEED_GRID", True)
ENABLE_STAGE2_MIGRAD_FIRST = _env_bool("ENABLE_STAGE2_MIGRAD_FIRST", False)
ENABLE_STAGE3_ADAPTIVE_RESCUE = _env_bool("ENABLE_STAGE3_ADAPTIVE_RESCUE", False)
ENABLE_STAGE4_LENGTH_PROFILE = _env_bool("ENABLE_STAGE4_LENGTH_PROFILE", False)

# Retry diagnostics / heuristics.
VISIBLE_LENGTH_RETRY_THRESHOLD = _env_float("VISIBLE_LENGTH_RETRY_THRESHOLD", 3000.0)
Z_SEED_EPS = _env_float("Z_SEED_EPS", 20.0)
VISIBLE_LENGTH_SEED_EPS = _env_float("VISIBLE_LENGTH_SEED_EPS", 40.0)
FULL_RANGE_SEED_EPS = _env_float("FULL_RANGE_SEED_EPS", 80.0)


# -----------------------------------------------------------------------------
# 5) Initial seed grid
# -----------------------------------------------------------------------------
FAST_SEED_X0 = _parse_float_list_env("FAST_SEED_X0", [-150.0, 0.0, 150.0])
FAST_SEED_Y0 = _parse_float_list_env("FAST_SEED_Y0", [-150.0, 0.0, 150.0])
FAST_SEED_Z0 = _parse_float_list_env(
    "FAST_SEED_Z0",
    [-1500.0, -1400.0, -1300.0, -1200.0, -1100.0, -1000.0],
)

# Used as "length" in full_length mode and "visible_length" in absorption mode.
FAST_SEED_VISIBLE_LENGTHS = _parse_float_list_env(
    "FAST_SEED_VISIBLE_LENGTHS",
    [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0,
     500.0, 550.0, 600.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0,
     1700.0, 1900.0],
)

# Absorption mode only: full_range seeds can be given directly or generated from KE.
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

# False = compact/sparse geometry scan.  True = full x0*y0*z0*direction Cartesian scan.
FAST_SEED_FULL_CARTESIAN = _env_bool("FAST_SEED_FULL_CARTESIAN", False)


# -----------------------------------------------------------------------------
# 6) WCSim PMT IDs, masks, and detector settings
# -----------------------------------------------------------------------------
# WCSim PMT ID handling:
#   "mapping" : require tables/wcsim_wcte_mapping.txt
#   "wcte"    : assume digi_hit_pmt already stores WCTE IDs like slot*100+pmt
#   "auto"    : use mapping if present; otherwise assume WCTE IDs
WCSIM_PMT_ID_MODE = os.environ.get("WCSIM_PMT_ID_MODE", "auto").strip().lower()
WCSIM_PMT_ID_OFFSET = _env_int("WCSIM_PMT_ID_OFFSET", 1)

# Leave blank to use <table dir>/wcsim_wcte_mapping.txt.
WCSIM_WCTE_MAPPING_PATH_OVERRIDE = os.environ.get("WCSIM_WCTE_MAPPING_PATH", "").strip()

# Ring mask applied after event observables are built:
#   "none" : no ring masking
#   "pes"  : mask charge only
#   "ts"   : mask timing only
#   "both" : mask charge and timing
RING_MASK_MODE = os.environ.get("RING_MASK_MODE", "both").strip().lower()

DEFAULT_INACTIVE_SLOTS = [27, 32, 45, 74, 77, 79, 85, 91, 99]
INACTIVE_SLOTS = _parse_int_list_env("INACTIVE_SLOTS", DEFAULT_INACTIVE_SLOTS)
INACTIVE_SLOTS_SET = set(int(s) for s in INACTIVE_SLOTS)


# -----------------------------------------------------------------------------
# 7) Paths and lookup tables
# -----------------------------------------------------------------------------
GEOMETRY_PATH = os.environ.get("GEOMETRY_PATH", "/eos/user/j/jrimmer/Geometry")
GEOMETRY_FILE = os.environ.get(
    "WCTE_GEOMETRY_FILE",
    str(Path(GEOMETRY_PATH) / "examples" / "wcte_bldg157.geo"),
)

# Leave blank to use <project root>/tables.
TABLE_DIR_OVERRIDE = os.environ.get(
    "TABLE_DIR",
    os.environ.get("LF_TABLE_DIR", ""),
).strip()

# Leave blank to use <table dir>/delta_e_angular_pdf_table.npz when it exists.
DELTA_E_ANGULAR_PDF_PATH_OVERRIDE = os.environ.get("DELTA_E_ANGULAR_PDF_PATH", "").strip()


# -----------------------------------------------------------------------------
# 8) Output/debug payload controls
# -----------------------------------------------------------------------------
# The full seed scan can be huge.  Keep these off for production output.
SAVE_ATTEMPT_RESULTS = _env_bool("SAVE_ATTEMPT_RESULTS", False)
SAVE_SEED_SCAN = _env_bool("SAVE_SEED_SCAN", False)
SAVE_TOP_N_SEEDS = _env_int("SAVE_TOP_N_SEEDS", 0)


# -----------------------------------------------------------------------------
# 9) Optional truth diagnostics
# -----------------------------------------------------------------------------
# If these are not supplied, true_fcn is stored as NaN.  By default this WCSim
# driver uses ENERGY_TRUE for TRUE_INITIAL_KE_MEV, matching your current script.
TRUE_VISIBLE_LENGTH_MM = _env_float("TRUE_VISIBLE_LENGTH_MM", None)
TRUE_FULL_RANGE_MM = _env_float("TRUE_FULL_RANGE_MM", None)
TRUE_LENGTH_MM = _env_float("TRUE_LENGTH_MM", None)
TRUE_INITIAL_KE_MEV = _env_float("TRUE_INITIAL_KE_MEV", ENERGY_TRUE)

TRUE_PARAMS = {
    "x0": _env_float("TRUE_X0", 0.0),
    "y0": _env_float("TRUE_Y0", 0.0),
    "z0": _env_float("TRUE_Z0", -1350.0),
    "cx": _env_float("TRUE_CX", 0.0),
    "cy": _env_float("TRUE_CY", 0.0),
    "length": np.nan,
    "visible_length": np.nan,
    "full_range": np.nan,
    "t0": _env_float("TRUE_T0", 0.0),
}


# =============================================================================
# END USER CONFIGURATION
# Everything below is derived/internal.  In normal use, do not edit below here.
# =============================================================================


# =============================================================================
# SELF-CONTAINED PATH SETUP
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LICKETYFIT_DIR = PROJECT_ROOT / "LicketyFit"
TABLE_DIR = Path(TABLE_DIR_OVERRIDE).expanduser() if TABLE_DIR_OVERRIDE else PROJECT_ROOT / "tables"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

WCSIM_WCTE_MAPPING_PATH = (
    Path(WCSIM_WCTE_MAPPING_PATH_OVERRIDE).expanduser()
    if WCSIM_WCTE_MAPPING_PATH_OVERRIDE
    else TABLE_DIR / "wcsim_wcte_mapping.txt"
)

DELTA_E_ANGULAR_PDF_PATH = (
    Path(DELTA_E_ANGULAR_PDF_PATH_OVERRIDE).expanduser()
    if DELTA_E_ANGULAR_PDF_PATH_OVERRIDE
    else TABLE_DIR / "delta_e_angular_pdf_table.npz"
)
if not DELTA_E_ANGULAR_PDF_PATH.exists():
    DELTA_E_ANGULAR_PDF_PATH = None

for _path in (str(LICKETYFIT_DIR), str(PROJECT_ROOT), str(SCRIPT_DIR), str(TABLE_DIR), GEOMETRY_PATH):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Force local tables before importing lookup/collapse helpers.
os.environ["LF_TABLE_DIR"] = str(TABLE_DIR)
os.environ["LF_MULTIPARTICLES_TABLE_DIR"] = str(TABLE_DIR)

from Geometry.Device import Device
from LicketyFit.Event import Event
from LicketyFit.PMT import PMT
from LicketyFit.Emitter import Emitter
from read_sim_data import read_sim_data
from particle_cherenkov_model import (
    get_energy_distance_tables,
    set_active_particle,
    canonical_particle_name,
    particle_mass_mev,
    cherenkov_threshold_kinetic_mev,
)
from particle_range_lookup import ParticleRangeLookup


# =============================================================================
# DERIVED CONFIGURATION AND VALIDATION
# =============================================================================
FIT_PARTICLE_STR = WCSIM_PARTICLE_LABEL  # legacy alias used in output metadata/prints
FIT_PARTICLE_CANONICAL = canonical_particle_name(FIT_PARTICLE)
FIT_PARTICLE_MASS_MEV = particle_mass_mev(FIT_PARTICLE_CANONICAL)
FIT_PARTICLE_THRESHOLD_MEV = cherenkov_threshold_kinetic_mev(FIT_PARTICLE_MASS_MEV)
set_active_particle(FIT_PARTICLE_CANONICAL)

if FIT_MODE_REQUEST in {"absorption", "absorbed", "abrupt", "abrupt_8param", "interaction", "truncated"}:
    FIT_MODE = "absorption"
elif FIT_MODE_REQUEST in {"full_length", "full-length", "full", "threshold", "range", "csda", "old", "7param", "7_parameter"}:
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

if "USE_CHARGE_LIKELIHOOD" in os.environ or "USE_TIMING_LIKELIHOOD" in os.environ:
    USE_CHARGE_LIKELIHOOD = _env_bool("USE_CHARGE_LIKELIHOOD", True)
    USE_TIMING_LIKELIHOOD = _env_bool("USE_TIMING_LIKELIHOOD", True)
else:
    _like = LIKELIHOOD_MODE_REQUEST.replace("-", "_").strip().lower()
    if _like in {"charge_time", "both", "joint", "charge_and_time"}:
        USE_CHARGE_LIKELIHOOD = True
        USE_TIMING_LIKELIHOOD = True
    elif _like in {"charge_only", "charge"}:
        USE_CHARGE_LIKELIHOOD = True
        USE_TIMING_LIKELIHOOD = False
    elif _like in {"timing_only", "time_only", "timing", "time"}:
        USE_CHARGE_LIKELIHOOD = False
        USE_TIMING_LIKELIHOOD = True
    else:
        raise ValueError(
            "LIKELIHOOD_MODE/FIT_TYPE must be 'charge_time', 'charge_only', or 'timing_only'."
        )

if (not USE_CHARGE_LIKELIHOOD) and (not USE_TIMING_LIKELIHOOD):
    raise ValueError("At least one likelihood term must be enabled.")

if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
    LIKELIHOOD_MODE = "charge_time"
elif USE_CHARGE_LIKELIHOOD:
    LIKELIHOOD_MODE = "charge_only"
else:
    LIKELIHOOD_MODE = "timing_only"

INPUT_FILE = INPUT_FILE_OVERRIDE or DEFAULT_WCSIM_INPUT_FILE

if OUTPUT_FILE_OVERRIDE:
    OUTPUT_FILE = OUTPUT_FILE_OVERRIDE
else:
    OUTPUT_FILE = str(
        OUTPUT_DIR
        / f"estimates_wcsim_{FIT_PARTICLE_CANONICAL}_{int(ENERGY_TRUE)}MeV_{TRACK_END_MODE}_{LIKELIHOOD_MODE}.dict"
    )

NEED_FULL_SEED_SCAN = (
    ENABLE_STAGE3_ADAPTIVE_RESCUE
    or ENABLE_STAGE4_LENGTH_PROFILE
    or SAVE_SEED_SCAN
    or SAVE_TOP_N_SEEDS > 0
)

OUTER_RING = np.array([0, 7, 19, 34, 50, 66, 82, 83, 105, 94, 95, 71, 72, 56, 40, 24, 11, 3, 18])
INNER_RING = np.array([1, 8, 35, 51, 67, 84, 69, 70, 55, 39, 23, 10, 2, 20, 36, 52, 68, 53, 54, 38, 22, 21, 37, 9])
ALL_RING = np.arange(0, 106)


# ---------------------------- #
# END OF CONFIGURATION
# NO NEED TO TOUCH CODE BELOW
# ---------------------------- #


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
        "t0": tuple(float(x) for x in T0_LIMITS),
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
        z0_values = [fixed_params["z0"]] if "z0" in fixed_params else FAST_SEED_Z0
        visible_length_values = (
            [fixed_params["visible_length"]]
            if "visible_length" in fixed_params
            else FAST_SEED_VISIBLE_LENGTHS
        )
        full_range_seeds = build_full_range_seed_values(range_lookup)
        full_range_values = (
            [fixed_params["full_range"]]
            if "full_range" in fixed_params
            else full_range_seeds
        )

        max_r = float(range_lookup.overall_distances_mm[-1])
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
SIM_WCTE_MAPPING = None
OVERALL_DISTANCES = None
INIT_ENERGY_TABLE = None
RANGE_LOOKUP = None
RESOLVED_FIXED_FIT_PARAMS = {}

WCD = None
PMT_MODEL = None
EMITTER_TEMPLATE = None
P_LOCATIONS = None
DIRECTION_ZS = None
RING_KEEP_MASK = None
CORR_POS = None

OBS_PES_ALL = None
OBS_TS_ALL = None


# =============================================================================
# LOCAL FILE HELPERS
# =============================================================================
def load_wcsim_to_wcte_mapping():
    mode = WCSIM_PMT_ID_MODE
    if mode not in {"mapping", "wcte", "auto"}:
        raise ValueError("WCSIM_PMT_ID_MODE must be one of: mapping, wcte, auto")

    if mode == "wcte":
        print("WCSim PMT ID mode: assuming digi_hit_pmt already contains WCTE PMT IDs.")
        return None

    if not WCSIM_WCTE_MAPPING_PATH.exists():
        if mode == "auto":
            print(
                "No wcsim_wcte_mapping.txt found in tables/. "
                "Falling back to WCTE-ID mode."
            )
            return None
        raise FileNotFoundError(
            f"WCSim mapping file not found: {WCSIM_WCTE_MAPPING_PATH}"
        )

    wcte_mapping = np.loadtxt(WCSIM_WCTE_MAPPING_PATH)
    mapping = {}
    for row in np.atleast_2d(wcte_mapping):
        mapping[int(row[0])] = int(row[1] * 100 + row[2] - 1)
    print(f"Loaded WCSim -> WCTE PMT mapping: {WCSIM_WCTE_MAPPING_PATH}")
    return mapping


def map_raw_pmt_to_wcte(raw_pmt):
    raw = int(raw_pmt)
    if SIM_WCTE_MAPPING is None:
        return raw
    return SIM_WCTE_MAPPING.get(raw + WCSIM_PMT_ID_OFFSET, None)


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
    pe_scale=1.0,
    shift_times=False,
    n_earliest_for_t0=10,
):
    slots = []
    pmt_pos_ids = []
    charges = []
    times = []
    vw = 223.0598645833333  # mm/ns

    for i in range(len(sim_data["digi_hit_pmt"])):
        wcte_pmt = map_raw_pmt_to_wcte(sim_data["digi_hit_pmt"][i])
        if wcte_pmt is None:
            continue
        slot = int(wcte_pmt // 100)
        pmt_pos = int(wcte_pmt % 100)
        if slot < 0 or slot >= n_mpmt_total or pmt_pos < 0 or pmt_pos >= 19:
            continue
        slots.append(slot)
        pmt_pos_ids.append(pmt_pos)
        charges.append(float(sim_data["digi_hit_charge"][i]))
        times.append(float(sim_data["digi_hit_time"][i]))

    ev = Event(0, 0, n_mpmt_total)
    ev.set_mpmt_status(list(range(n_mpmt_total)), True)

    wcte_pmt_ids = []
    for i_mpmt in range(n_mpmt_total):
        if i_mpmt in INACTIVE_SLOTS:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), False)
        else:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), True)
            for i_pmt in range(19):
                wcte_pmt_ids.append(i_mpmt * 100 + i_pmt)

    for s, p, q, t in zip(slots, pmt_pos_ids, charges, times):
        ev.hit_times[s][p].append(t)
        ev.hit_charges[s][p].append(q)

    if shift_times:
        bp_loc = np.array([0.0, 0.0, -1350.0])
        early_hits = []

        for i_mpmt in range(ev.n_mpmt):
            for i_pmt in range(ev.npmt_per_mpmt):
                if len(ev.hit_times[i_mpmt][i_pmt]) == 0:
                    continue
                pmt_loc = WCD.mpmts[i_mpmt].pmts[i_pmt].get_placement("est")["location"]
                r = np.linalg.norm(pmt_loc - bp_loc)
                for t in ev.hit_times[i_mpmt][i_pmt]:
                    early_hits.append({"time": float(t), "t0_est": float(t) - r / vw})

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

    return ev, np.asarray(wcte_pmt_ids, dtype=int)


def build_observables_from_event(ev, pe_scale=1.0):
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

    top_n_for_retry = max(1, int(MAX_FIT_ATTEMPTS))
    top_seeds_buffer = []

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

        top_seeds_buffer.append(info)
        top_seeds_buffer.sort(key=lambda x: x["fval"])
        if len(top_seeds_buffer) > top_n_for_retry:
            top_seeds_buffer = top_seeds_buffer[:top_n_for_retry]

    if best_info is None or not np.isfinite(best_info["fval"]):
        raise RuntimeError("All seed FCNs were non-finite. Check ranges, PMT ordering, and event selection.")

    if seed_scan is not None:
        seed_scan_sorted = sorted(seed_scan, key=lambda x: x["fval"])
    else:
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

    mpmt_types = None
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
    m.limits["t0"] = T0_LIMITS

    m.errors["x0"] = 30.0
    m.errors["y0"] = 30.0
    m.errors["z0"] = 30.0
    m.errors["cx"] = 0.01
    m.errors["cy"] = 0.01
    m.errors["t0"] = 0.1

    if IS_ABSORPTION_MODE:
        m.limits["visible_length"] = (0.0, 5000)
        m.limits["full_range"] = (1.0, 5000)
        m.errors["visible_length"] = 60.0
        m.errors["full_range"] = 100.0
    else:
        m.limits["length"] = (0.0, 5000)
        m.errors["length"] = 60.0

    if not USE_TIMING_LIKELIHOOD:
        m.fixed["t0"] = True

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
    """Default FCN retry criterion, independent of adaptive rescue.

    Do not use m.valid here.  Retry only for non-finite FCN or FCN above the
    configured threshold.
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
    """Return either the full seed scan, the top-N seeds, or nothing.

    Keeping the full seed_scan for every event can make the output pickle
    gigabytes large.  By default, production output stores no seed_scan.
    """
    if SAVE_SEED_SCAN:
        return seed_scan_sorted
    if SAVE_TOP_N_SEEDS > 0:
        return seed_scan_sorted[:SAVE_TOP_N_SEEDS]
    return None


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
    mpmt_types = None

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

    # Default FCN retry path, independent of adaptive rescue.  This lets the
    # WCSim driver behave like the real-data batch driver: if the current best
    # FCN is too high, try the next-best prescan seeds up to MAX_FIT_ATTEMPTS.
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
        best_result.pop("length_profile_results", None)
        best_result.pop("length_profile_best_fixed", None)

    best_result["adaptive_rescue_used"] = bool(
        any(r.get("adaptive_rescue_attempt", False) for r in attempt_results)
    )
    best_result["fcn_retry_used"] = bool(
        any(r.get("fcn_retry_used", False) for r in attempt_results)
    )
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
# MAIN DRIVER
# =============================================================================
def main():
    global SIM_WCTE_MAPPING, OVERALL_DISTANCES, INIT_ENERGY_TABLE, RANGE_LOOKUP, RESOLVED_FIXED_FIT_PARAMS
    global WCD, PMT_MODEL, EMITTER_TEMPLATE, P_LOCATIONS, DIRECTION_ZS, RING_KEEP_MASK, CORR_POS
    global OBS_PES_ALL, OBS_TS_ALL

    print("Likelihood mode:", LIKELIHOOD_MODE)
    print("Fit particle:", FIT_PARTICLE_CANONICAL)
    print("Raw FIT_PARTICLE:", FIT_PARTICLE)
    print("WCSim particle label:", FIT_PARTICLE_STR)
    print("Particle mass [MeV]:", FIT_PARTICLE_MASS_MEV)
    print("Cherenkov threshold [MeV]:", FIT_PARTICLE_THRESHOLD_MEV)
    print("Fit mode:", TRACK_END_MODE)
    print("Fit parameters:", FIT_PARAMETER_NAMES)
    print("Input file:", INPUT_FILE)
    print("Output file:", OUTPUT_FILE)

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
        print("Free parameters:", free_parameter_names(RESOLVED_FIXED_FIT_PARAMS))

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

    data_raw = read_sim_data(INPUT_FILE)

    set_active_particle(FIT_PARTICLE_CANONICAL)
    OVERALL_DISTANCES, INIT_ENERGY_TABLE, _distance_rows = get_energy_distance_tables(
        FIT_PARTICLE_CANONICAL
    )

    SIM_WCTE_MAPPING = load_wcsim_to_wcte_mapping()

    hall = Device.open_file(GEOMETRY_FILE)
    WCD = hall.wcds[0]

    # In absorption mode fixed_initial_KE is overwritten per FCN call from full_range -> ke0.
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
    if DELTA_E_ANGULAR_PDF_PATH is not None and hasattr(emitter_model, "load_delta_e_angular_pdf_table"):
        emitter_model.load_delta_e_angular_pdf_table(str(DELTA_E_ANGULAR_PDF_PATH))

    PMT_MODEL = PMT(1.0, 0.3, 1.0, 40.0, 0.2, 0.0)
    EMITTER_TEMPLATE = emitter_model.copy()
    CORR_POS = None

    print("Building event observables...")

    obs_pes_all = []
    obs_ts_all = []

    n_available = len(data_raw["digi_hit_time"])
    n_events_to_process = min(int(TOT_EVENTS), int(n_available))
    print("Total events to fit:", n_events_to_process)

    for evt_num in range(n_events_to_process):
        hit_times = np.asarray(data_raw["digi_hit_time"][evt_num], dtype=np.float64)
        hit_pmts = np.asarray(data_raw["digi_hit_pmt"][evt_num], dtype=int)
        hit_charges = np.asarray(data_raw["digi_hit_charge"][evt_num], dtype=np.float64)

        if hit_times.size == 0:
            keep = np.zeros_like(hit_times, dtype=bool)
        else:
            time_hist = np.histogram(hit_times, bins=np.arange(0, 2000))
            max_idx = int(np.argmax(time_hist[0]))
            min_time = 0.0
            cut_idx = min(max_idx + 5, len(time_hist[1]) - 1)
            cut_time = time_hist[1][cut_idx]
            keep = (hit_times > min_time) & (hit_times < cut_time)

        sim_data = {
            "digi_hit_pmt": hit_pmts[keep],
            "digi_hit_time": hit_times[keep],
            "digi_hit_charge": hit_charges[keep],
        }

        ev, pmt_ids = sim_to_event(sim_data, WCD, n_mpmt_total=106, pe_scale=1.0)

        if P_LOCATIONS is None or DIRECTION_ZS is None:
            P_LOCATIONS, DIRECTION_ZS, MPMT_SLOTS = EMITTER_TEMPLATE.get_pmt_placements(ev, WCD, "design")
            RING_KEEP_MASK = np.isin(np.asarray(MPMT_SLOTS, dtype=int), ALL_RING)

        obs_pes, obs_ts = build_observables_from_event(ev, pe_scale=1.0)
        obs_pes, obs_ts = apply_ring_mask_to_observables(
            obs_pes,
            obs_ts,
            RING_KEEP_MASK,
            mode=RING_MASK_MODE,
        )

        obs_pes_all.append(obs_pes)
        obs_ts_all.append(obs_ts)

    OBS_PES_ALL = obs_pes_all
    OBS_TS_ALL = obs_ts_all

    print("Computing truth FCNs...")
    true_fcn_all = [compute_true_fcn_for_event(i) for i in range(n_events_to_process)]

    est_dict = {
        "metadata": {
            "fit_particle": FIT_PARTICLE_CANONICAL,
            "fit_particle_raw": FIT_PARTICLE,
            "wcsim_particle_label": FIT_PARTICLE_STR,
            "wcsim_particle_dir": WCSIM_PARTICLE_DIR,
            "particle_mass_mev": FIT_PARTICLE_MASS_MEV,
            "particle_threshold_mev": FIT_PARTICLE_THRESHOLD_MEV,
            "energy_true_mev": ENERGY_TRUE,
            "track_end_mode": TRACK_END_MODE,
            "fit_parameters": list(FIT_PARAMETER_NAMES),
            "fixed_params": dict(RESOLVED_FIXED_FIT_PARAMS),
            "free_params": list(free_parameter_names(RESOLVED_FIXED_FIT_PARAMS)),
            "fixed_ke0_mev": float(FIXED_KE0_MEV) if FIXED_KE0_MEV is not None else None,
            "truth_params": dict(TRUE_PARAMS),
            "input_file": INPUT_FILE,
            "geometry_file": GEOMETRY_FILE,
            "wcsim_pmt_id_mode": WCSIM_PMT_ID_MODE,
            "mapping_path": str(WCSIM_WCTE_MAPPING_PATH),
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
        "fcn_retry_used": [],
        "length_profile_rescue_considered": [],
        "length_profile_rescue_used": [],
        "edm": [],
    }
    if SAVE_SEED_SCAN or SAVE_TOP_N_SEEDS > 0:
        est_dict["seed_scan"] = []

    if SAVE_ATTEMPT_RESULTS:
        est_dict["attempt_results"] = []

    for batch_start in range(0, n_events_to_process, N_EVENTS_PER_BATCH):
        batch_end = min(batch_start + N_EVENTS_PER_BATCH, n_events_to_process)
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
            est_dict["fcn_retry_used"].append(result.get("fcn_retry_used", False))
            est_dict["length_profile_rescue_considered"].append(result.get("length_profile_rescue_considered", False))
            est_dict["length_profile_rescue_used"].append(result.get("length_profile_rescue_used", False))
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
