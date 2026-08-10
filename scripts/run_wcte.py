#!/usr/bin/env python3
"""Edit this short configuration and run LicketyFit on real WCTE data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


# =============================================================================
# WCTE RUN CONFIGURATION -- EDIT THIS SECTION
# Settings are ordered from the most commonly changed to the least commonly
# changed. This file overrides matching values left in your shell environment.
# =============================================================================

# --- 1. Fit and computing -----------------------------------------------------

# "cosmic" handles all inside/outside start/stop topologies.
# "full_length" fits an internal start and remaining stopping range.
# "absorption" fits separate visible length and full range.
FIT_MODE = "cosmic"

# Supported fit hypotheses: "muon", "pion", "kaon", or "proton".
FIT_PARTICLE = "muon"

# Choose your likelihood mode - "charge_time", "charge_only", or "time_only"
# "charge_time" is recommended
LIKELIHOOD_MODE = "charge_time"

# Number of event-level worker processes.
NPROC = 16

# Blank lets the driver construct its normal output filename.
OUTPUT_FILE = ""


# --- 2. Event source ----------------------------------------------------------

# "selection": load the collaboration ROOT and run DataLoader/BeamSelection.
# "file": load already-selected events from USER_EVENT_FILE.
EVENT_SOURCE = "file"

# In selection mode this chooses the collaboration run. In file mode it does
# not alter the user events, but it still chooses the run mask when
# GOOD_PMT_SOURCE is "run" or resolves to "run" through "auto".
RUN = 2079

# Blank uses the standard merged production ROOT for RUN. This is the event
# ROOT in selection mode, not a standalone dq_flags file.
COLLABORATION_ROOT_FILE = ""

# Required only when EVENT_SOURCE = "file". Supported: NPY, NPZ, PKL, PICKLE.
USER_EVENT_FILE = "/eos/user/j/jrimmer/SWAN_projects/beam/data_production_v1/r2079.npy"

# Optional NPZ/pickle key when a user event container is ambiguous.
USER_EVENT_KEY = ""

# Selection mode: maximum raw ROOT entries inspected before event selection.
N_ROOT_ENTRIES = 5000

# Maximum selected events actually fitted. None means all selected events.
MAX_EVENTS_TO_FIT = None

# Skip this many already-selected events before fitting. Usually zero.
EVENT_START_INDEX = 0


# --- 3. Authoritative active-PMT source ---------------------------------------

# "auto": use GOOD_PMT_FILE when nonblank, otherwise discover the mask for RUN.
# "file": require GOOD_PMT_FILE.
# "run": read Configuration/good_wcte_pmts from a DQ/merged ROOT for RUN.
GOOD_PMT_SOURCE = "run"

# NPY/NPZ/TXT/CSV/JSON list of active WCTE PMTs. Required for source "file".
GOOD_PMT_FILE = "NPY"
GOOD_PMT_FILE_KEY = ""  # Usually blank; selects an array in an ambiguous NPZ.

# Optional exact standalone DQ or merged ROOT override for source "run".
# This file supplies only good_wcte_pmts; it never replaces USER_EVENT_FILE.
GOOD_PMT_ROOT_FILE = ""

# Optional additional directories searched for a run ROOT before built-in EOS
# locations, for example ("/my/production", "/another/location").
GOOD_PMT_ROOT_SEARCH_BASES = ()


# --- 4. Beam population and event selection ----------------------------------
# These settings apply only when EVENT_SOURCE = "selection".

# Nominal analysis_tools populations: muon, pion, electron, proton.
# Kaon requires SELECTION_MODE="custom" and explicit cuts.
PARTICLE_SELECTION_LABEL = "muon"
SELECTION_MODE = "nominal"  # "nominal" or "custom"

# Each custom/extra cut is (branch, operator, value). In nominal mode cuts are
# appended; in custom mode they are the complete selection.
# Example: (("vme_act0_l_charge", ">", 10.0), ("T5_particle_nr", "==", 1))
EXTRA_SELECTION_CUTS = ()

APPLY_MPMT_DATA_QUALITY_CUTS = True
APPLY_VME_EVENT_QUALITY_CUTS = True
APPLY_T5_EVENT_QUALITY_CUTS = True

USE_ACT_EVETO_CUT = True
USE_ACT_TAGGER_CUT = True
TOF_CUT_MODE = "auto"  # "auto", "require", or "disable"
PROTON_TOF_WINDOW_NS = 10.0
REQUIRE_MUON_TAGGER = False

# None uses run-derived selection constants.
ACT_EVETO_CUT_OVERRIDE_PE = None
ACT_TAGGER_CUT_OVERRIDE_PE = None
PROTON_TOF_CUT_OVERRIDE_NS = None
MUON_TAG_CUT_OVERRIDE = None

# Nominal beam metadata and optional seed guidance; neither is event truth.
BEAM_MOMENTUM_MEV_C = 430.0
EXPECTED_KINETIC_ENERGY_MEV = 300.0
USE_EXPECTED_ENERGY_SEED_HINT = False


# --- 5. Real-data preparation and calibration --------------------------------

RELATIVE_EFFICIENCY_MODE = "slot"  # "slot", "type", or "none"
GEOMETRY_PLACEMENT = "est"         # surveyed "est" or nominal "design"
CHARGE_ADC_PER_PE = 143.0

# "event_mean" profiles out total event charge. "global_scale" requires an
# independently calibrated GLOBAL_CHARGE_SCALE.
CHARGE_NORMALIZATION_MODE = "event_mean"
GLOBAL_CHARGE_SCALE = None

PROMPT_WINDOW_MODE = "peak_relative"  # "peak_relative", "fixed", or "none"
PROMPT_TIME_MIN_NS = None              # Required only for "fixed" mode.
PROMPT_TIME_MAX_NS = None
USER_EVENT_APPLY_PROMPT_WINDOW = True
STRICT_USER_EVENT_VALIDATION = True

# Established real-data timing reference.
TIME_REFERENCE_MODE = "beam_corrected_peak"
TIME_REFERENCE_BIN_WIDTH_NS = 0.5
TIME_REFERENCE_LOCAL_HALF_WIDTH_NS = 1.0


# --- 6. External paths --------------------------------------------------------

# Blank GEOMETRY_FILE uses the geometry_repository bundled with this package.
GEOMETRY_FILE = ""

# DataLoader/BeamSelection checkout. An installed analysis_tools package is also
# detected; this path is the CERN compatibility fallback.
ANALYSIS_TOOLS_PATH = (
    "/eos/user/j/jrimmer/SWAN_projects/beam/"
    "data_production_v1/analysis_tools"
)
SELECTION_STEP_SIZE = "100 MB"


# --- 7. Checkpointing, performance, and output verbosity ---------------------

N_EVENTS_PER_BATCH = 100
WARM_FIT_KERNELS = True
SAVE_AFTER_EACH_BATCH = True
SAVE_DETAILED_EVENT_RESULTS = False

PRINT_EVENT_RESULTS = False
PRINT_BATCH_PROGRESS = True
PRINT_CHECKPOINT_MESSAGES = False
VERBOSE_SETUP = False
PRINT_SELECTION_DESCRIPTION = True
PRINT_CHERENKOV_THRESHOLDS = True

# Expert escape hatch for driver environment settings not listed above.
# Applied last, so matching names here override the settings above.
# Example: {"FIX_X0": 0.0, "FIX_Y0": 0.0, "FIX_Z0": -1350.0}
EXTRA_DRIVER_ENV = {}

# =============================================================================
# END WCTE RUN CONFIGURATION -- USERS NORMALLY DO NOT EDIT BELOW THIS LINE
# =============================================================================


_DRIVER = Path(__file__).resolve().with_name("batch_fit_driver.py")
_FIT_MODES = {"full_length", "absorption", "cosmic"}
_FIT_PARTICLES = {"muon", "pion", "kaon", "proton"}
_LIKELIHOODS = {"charge_only", "charge_time", "timing_only"}


def _require_file(value: str, label: str) -> None:
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")


def _validate(*, check_paths: bool) -> None:
    if FIT_MODE not in _FIT_MODES:
        raise ValueError(f"FIT_MODE must be one of {sorted(_FIT_MODES)}")
    if FIT_PARTICLE not in _FIT_PARTICLES:
        raise ValueError(f"FIT_PARTICLE must be one of {sorted(_FIT_PARTICLES)}")
    if LIKELIHOOD_MODE not in _LIKELIHOODS:
        raise ValueError(f"LIKELIHOOD_MODE must be one of {sorted(_LIKELIHOODS)}")
    if EVENT_SOURCE not in {"selection", "file"}:
        raise ValueError("EVENT_SOURCE must be 'selection' or 'file'")
    if GOOD_PMT_SOURCE not in {"auto", "file", "run"}:
        raise ValueError("GOOD_PMT_SOURCE must be 'auto', 'file', or 'run'")
    if int(RUN) < 0 or int(NPROC) < 1 or int(N_ROOT_ENTRIES) < 1:
        raise ValueError("RUN must be nonnegative; NPROC and N_ROOT_ENTRIES must be positive")
    if MAX_EVENTS_TO_FIT is not None and int(MAX_EVENTS_TO_FIT) < 1:
        raise ValueError("MAX_EVENTS_TO_FIT must be positive or None")
    if int(EVENT_START_INDEX) < 0:
        raise ValueError("EVENT_START_INDEX must be nonnegative")
    if SELECTION_MODE not in {"nominal", "custom"}:
        raise ValueError("SELECTION_MODE must be 'nominal' or 'custom'")
    if TOF_CUT_MODE not in {"auto", "require", "disable"}:
        raise ValueError("TOF_CUT_MODE must be 'auto', 'require', or 'disable'")
    if GEOMETRY_PLACEMENT not in {"est", "design"}:
        raise ValueError("GEOMETRY_PLACEMENT must be 'est' or 'design'")
    if RELATIVE_EFFICIENCY_MODE not in {"slot", "type", "none"}:
        raise ValueError("RELATIVE_EFFICIENCY_MODE must be 'slot', 'type', or 'none'")
    if PROMPT_WINDOW_MODE not in {"peak_relative", "fixed", "none"}:
        raise ValueError("PROMPT_WINDOW_MODE must be 'peak_relative', 'fixed', or 'none'")
    if PROMPT_WINDOW_MODE == "fixed" and (
        PROMPT_TIME_MIN_NS is None or PROMPT_TIME_MAX_NS is None
    ):
        raise ValueError("Fixed prompt mode requires PROMPT_TIME_MIN_NS and PROMPT_TIME_MAX_NS")
    if (
        PROMPT_TIME_MIN_NS is not None and PROMPT_TIME_MAX_NS is not None
        and float(PROMPT_TIME_MAX_NS) <= float(PROMPT_TIME_MIN_NS)
    ):
        raise ValueError("PROMPT_TIME_MAX_NS must exceed PROMPT_TIME_MIN_NS")
    if TIME_REFERENCE_MODE not in {
        "beam_earliest", "beam_corrected_peak", "beam_corrected_local_median",
        "beam_all_median", "none",
    }:
        raise ValueError("Unsupported TIME_REFERENCE_MODE")
    if CHARGE_NORMALIZATION_MODE not in {"event_mean", "global_scale"}:
        raise ValueError("CHARGE_NORMALIZATION_MODE must be 'event_mean' or 'global_scale'")
    if CHARGE_NORMALIZATION_MODE == "global_scale" and (
        GLOBAL_CHARGE_SCALE is None or float(GLOBAL_CHARGE_SCALE) <= 0.0
    ):
        raise ValueError("A positive GLOBAL_CHARGE_SCALE is required in global_scale mode")
    if float(CHARGE_ADC_PER_PE) <= 0.0:
        raise ValueError("CHARGE_ADC_PER_PE must be positive")
    if float(TIME_REFERENCE_BIN_WIDTH_NS) <= 0.0 or float(TIME_REFERENCE_LOCAL_HALF_WIDTH_NS) < 0.0:
        raise ValueError("Time-reference bin width must be positive and local half-width nonnegative")
    if EVENT_SOURCE == "file" and not str(USER_EVENT_FILE).strip():
        raise ValueError("EVENT_SOURCE='file' requires USER_EVENT_FILE")
    if GOOD_PMT_SOURCE == "file" and not str(GOOD_PMT_FILE).strip():
        raise ValueError("GOOD_PMT_SOURCE='file' requires GOOD_PMT_FILE")
    if EVENT_SOURCE == "selection" and SELECTION_MODE == "custom" and not EXTRA_SELECTION_CUTS:
        raise ValueError("Custom selection mode requires EXTRA_SELECTION_CUTS")
    if EVENT_SOURCE == "selection" and PARTICLE_SELECTION_LABEL == "kaon" and SELECTION_MODE != "custom":
        raise ValueError("Kaon selection requires SELECTION_MODE='custom'")
    for cut in EXTRA_SELECTION_CUTS:
        if not isinstance(cut, (tuple, list)) or len(cut) != 3:
            raise ValueError(f"Malformed selection cut: {cut!r}")
    json.dumps([list(cut) for cut in EXTRA_SELECTION_CUTS])
    if check_paths:
        if EVENT_SOURCE == "file":
            _require_file(USER_EVENT_FILE, "USER_EVENT_FILE")
        if EVENT_SOURCE == "selection" and str(COLLABORATION_ROOT_FILE).strip():
            _require_file(COLLABORATION_ROOT_FILE, "COLLABORATION_ROOT_FILE")
        use_user_mask = GOOD_PMT_SOURCE == "file" or (
            GOOD_PMT_SOURCE == "auto" and bool(str(GOOD_PMT_FILE).strip())
        )
        if use_user_mask:
            _require_file(GOOD_PMT_FILE, "GOOD_PMT_FILE")
        if not use_user_mask and str(GOOD_PMT_ROOT_FILE).strip():
            _require_file(GOOD_PMT_ROOT_FILE, "GOOD_PMT_ROOT_FILE")
        if str(GEOMETRY_FILE).strip():
            _require_file(GEOMETRY_FILE, "GEOMETRY_FILE")
        _require_file(str(_DRIVER), "batch_fit_driver.py")


def _encode(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value)
    return str(value)


def _configuration_items() -> list[tuple[str, object]]:
    root_file = str(COLLABORATION_ROOT_FILE).strip() or None
    geometry_file = str(GEOMETRY_FILE).strip() or None
    output_file = str(OUTPUT_FILE).strip() or None
    config_file = Path(__file__).resolve()
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    items = [
        ("LF_DATA_SOURCE", "wcte"), ("DATA_SOURCE", "wcte"),
        ("FIT_MODE", FIT_MODE), ("FIT_PARTICLE", FIT_PARTICLE),
        ("LIKELIHOOD_MODE", LIKELIHOOD_MODE), ("NPROC", NPROC),
        ("LF_OUTPUT_FILE", output_file),
        ("EVENT_SOURCE", EVENT_SOURCE), ("RUN", RUN),
        ("CONFIG_ROOT_FILE", root_file), ("USER_EVENT_FILE", USER_EVENT_FILE),
        ("USER_EVENT_KEY", USER_EVENT_KEY), ("N_ROOT_ENTRIES", N_ROOT_ENTRIES),
        ("N_EVENTS", N_ROOT_ENTRIES), ("MAX_EVENTS_TO_FIT", MAX_EVENTS_TO_FIT),
        ("TOT_EVENTS", MAX_EVENTS_TO_FIT), ("LF_EVENT_START_INDEX", EVENT_START_INDEX),
        ("WCTE_GOOD_PMT_SOURCE", GOOD_PMT_SOURCE),
        ("WCTE_GOOD_PMT_FILE", GOOD_PMT_FILE),
        ("WCTE_GOOD_PMT_FILE_KEY", GOOD_PMT_FILE_KEY),
        ("WCTE_GOOD_PMT_ROOT_FILE", GOOD_PMT_ROOT_FILE),
        ("WCTE_GOOD_PMT_ROOT_SEARCH_BASES", os.pathsep.join(map(str, GOOD_PMT_ROOT_SEARCH_BASES))),
        ("PARTICLE_SELECTION_LABEL", PARTICLE_SELECTION_LABEL),
        ("WCTE_SELECTION_MODE", SELECTION_MODE),
        ("WCTE_EXTRA_SELECTION_CUTS_JSON", json.dumps([list(cut) for cut in EXTRA_SELECTION_CUTS])),
        ("WCTE_APPLY_MPMT_DATA_QUALITY_CUTS", APPLY_MPMT_DATA_QUALITY_CUTS),
        ("WCTE_APPLY_VME_EVENT_QUALITY_CUTS", APPLY_VME_EVENT_QUALITY_CUTS),
        ("WCTE_APPLY_T5_EVENT_QUALITY_CUTS", APPLY_T5_EVENT_QUALITY_CUTS),
        ("WCTE_USE_ACT_EVETO_CUT", USE_ACT_EVETO_CUT),
        ("WCTE_USE_ACT_TAGGER_CUT", USE_ACT_TAGGER_CUT),
        ("WCTE_TOF_CUT_MODE", TOF_CUT_MODE),
        ("WCTE_PROTON_TOF_WINDOW_NS", PROTON_TOF_WINDOW_NS),
        ("WCTE_REQUIRE_MUON_TAGGER", REQUIRE_MUON_TAGGER),
        ("WCTE_ACT_EVETO_CUT_OVERRIDE_PE", ACT_EVETO_CUT_OVERRIDE_PE),
        ("WCTE_ACT_TAGGER_CUT_OVERRIDE_PE", ACT_TAGGER_CUT_OVERRIDE_PE),
        ("WCTE_PROTON_TOF_CUT_OVERRIDE_NS", PROTON_TOF_CUT_OVERRIDE_NS),
        ("WCTE_MUON_TAG_CUT_OVERRIDE", MUON_TAG_CUT_OVERRIDE),
        ("BEAM_P", BEAM_MOMENTUM_MEV_C),
        ("WCTE_EXPECTED_KE_MEV", EXPECTED_KINETIC_ENERGY_MEV),
        ("WCTE_USE_EXPECTED_ENERGY_HINT", USE_EXPECTED_ENERGY_SEED_HINT),
        ("REL_EFF_MODE", RELATIVE_EFFICIENCY_MODE),
        ("WCTE_PLACEMENT_KEY", GEOMETRY_PLACEMENT),
        ("CHARGE_ADC_PER_PE", CHARGE_ADC_PER_PE),
        ("WCTE_CHARGE_NORMALIZATION_MODE", CHARGE_NORMALIZATION_MODE),
        ("WCTE_GLOBAL_CHARGE_SCALE", GLOBAL_CHARGE_SCALE),
        ("WCTE_PROMPT_WINDOW_MODE", PROMPT_WINDOW_MODE),
        ("WCTE_PROMPT_TIME_MIN_NS", PROMPT_TIME_MIN_NS),
        ("WCTE_PROMPT_TIME_MAX_NS", PROMPT_TIME_MAX_NS),
        ("USER_EVENT_APPLY_PROMPT_WINDOW", USER_EVENT_APPLY_PROMPT_WINDOW),
        ("WCTE_STRICT_USER_EVENT_VALIDATION", STRICT_USER_EVENT_VALIDATION),
        ("WCTE_TIME_REFERENCE_MODE", TIME_REFERENCE_MODE),
        ("WCTE_TIME_REFERENCE_BIN_WIDTH_NS", TIME_REFERENCE_BIN_WIDTH_NS),
        ("WCTE_TIME_REFERENCE_LOCAL_HALF_WIDTH_NS", TIME_REFERENCE_LOCAL_HALF_WIDTH_NS),
        ("WCD_GEOMETRY_FILE", geometry_file), ("WCTE_GEOMETRY_FILE", geometry_file),
        ("WCTE_ANALYSIS_TOOLS_PATH", ANALYSIS_TOOLS_PATH),
        ("SELECTION_STEP_SIZE", SELECTION_STEP_SIZE),
        ("N_EVENTS_PER_BATCH", N_EVENTS_PER_BATCH),
        ("WARM_FIT_KERNELS", WARM_FIT_KERNELS),
        ("SAVE_AFTER_EACH_BATCH", SAVE_AFTER_EACH_BATCH),
        ("SAVE_DETAILED_EVENT_RESULTS", SAVE_DETAILED_EVENT_RESULTS),
        ("PRINT_EVENT_RESULTS", PRINT_EVENT_RESULTS),
        ("PRINT_BATCH_PROGRESS", PRINT_BATCH_PROGRESS),
        ("PRINT_CHECKPOINT_MESSAGES", PRINT_CHECKPOINT_MESSAGES),
        ("VERBOSE_SETUP", VERBOSE_SETUP),
        ("WCTE_PRINT_SELECTION_DESCRIPTION", PRINT_SELECTION_DESCRIPTION),
        ("WCTE_PRINT_CHERENKOV_THRESHOLDS", PRINT_CHERENKOV_THRESHOLDS),
        ("WCSIM_USE_TRUTH_ROOT", False),
        ("ALLOW_MISSING_GOOD_PMTS", False),
        ("LF_RUN_CONFIG_KIND", "wcte"),
        ("LF_RUN_CONFIG_FILE", str(config_file)),
        ("LF_RUN_CONFIG_SHA256", config_hash),
        ("LF_PUBLIC_DRIVER_RELEASE", None),
    ]
    items.extend((str(name), value) for name, value in EXTRA_DRIVER_ENV.items())
    return items


def build_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return the exact environment handed to batch_fit_driver.py."""
    env = dict(os.environ if base is None else base)
    for internal in (
        "LF_COSMIC_SUPERVISED_CHILD", "LF_COSMIC_CHILD_QUIET",
        "LF_WCTE_INTERNAL_PREPARED_EVENT_FILE", "LF_WCTE_PREPARE_EVENTS_ONLY",
        "LF_WCTE_PREPARED_EVENT_FILE", "LF_EVENT_COUNT",
    ):
        env.pop(internal, None)
    for name, value in _configuration_items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = _encode(value)
    return env


def _print_configuration() -> None:
    print(f"Launcher: {Path(__file__).resolve()}")
    print(f"Driver:   {_DRIVER}")
    for name, value in _configuration_items():
        shown = "<unset>" if value is None else _encode(value)
        print(f"{name}={shown}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure and run LicketyFit on real WCTE data."
    )
    parser.add_argument(
        "--show-config", action="store_true",
        help="print the driver environment generated by this file and exit",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="validate choices and required local paths without running a fit",
    )
    args = parser.parse_args()
    try:
        _validate(check_paths=bool(args.check or not args.show_config))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.show_config:
        _print_configuration()
        return
    if args.check:
        print("WCTE run configuration is valid.")
        return
    environment = build_environment()
    os.execve(sys.executable, [sys.executable, str(_DRIVER)], environment)


if __name__ == "__main__":
    main()
