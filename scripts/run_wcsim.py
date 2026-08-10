#!/usr/bin/env python3
"""Edit this short configuration and run LicketyFit on WCSim data."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys


# =============================================================================
# WCSIM RUN CONFIGURATION -- EDIT THIS SECTION
# Settings are ordered from the most commonly changed to the least commonly
# changed. This file overrides matching values left in your shell environment.
# =============================================================================

# --- 1. Input, fit, and computing --------------------------------------------

# "cosmic" handles all inside/outside start/stop topologies.
# "full_length" fits an internal start and remaining stopping range.
# "absorption" fits separate visible length and full range.
FIT_MODE = "cosmic"

# Digitized WCSim NPZ input.
INPUT_FILE = ""  # Required: set this to your WCSim NPZ file.

# Number of events to fit and the first input event index.
N_EVENTS = 100
EVENT_START_INDEX = 0

# Supported fit hypotheses: "muon", "pion", "kaon", or "proton".
FIT_PARTICLE = "muon"

# "charge_time", "charge_only", or "timing_only".
LIKELIHOOD_MODE = "charge_time"

# Number of event-level worker processes.
NPROC = 16

# Blank lets the driver construct its normal output filename.
OUTPUT_FILE = ""


# --- 2. Detector geometry and seed label -------------------------------------

# Select exactly one detector mode.
USE_WCTE_GEOMETRY = True
USE_IWCD_GEOMETRY = False

# WCTE: blank uses Geometry/examples/wcte_bldg157.geo from the pinned Geometry
# submodule. IWCD: an explicit, independently validated .geo file is currently
# required; the upstream Geometry repository does not ship a serialized IWCD
# geometry and LicketyFit does not silently generate one.
GEOMETRY_FILE = ""

# Metadata and optional seed guidance only. It is not truth and does not
# constrain the full_length/cosmic fit.
ENERGY_LABEL_MEV = 300.0


# --- 3. Optional per-step WCSim truth diagnostics ----------------------------

# Truth is diagnostic only and never enters the likelihood.
USE_TRUTH_ROOT = False
TRUTH_ROOT_FILE = ""  # Required only when USE_TRUTH_ROOT = True.
TRUTH_TREE = "AllSecondaries"
TRUTH_EVENT_ID_OFFSET = 0

# None automatically applies the WCSim-to-geometry y offset for WCTE. Supply a
# three-number tuple only for a deliberately different coordinate convention.
TRUTH_POSITION_OFFSET_MM = None

# None infers the primary track deterministically.
TRUTH_PRIMARY_TRACK_ID = None

TRUTH_UPROOT_STEP_SIZE = "64 MB"
TRUTH_EVENT_INDEX_STEP_SIZE = "16 MB"
TRUTH_IO_WORKERS = 4
TRUTH_USE_EVENT_INDEX_CACHE = True
TRUTH_INDEX_CACHE_DIR = ""  # Blank uses the normal user cache.
TRUTH_INCLUDE_OPTIONAL_DETAILS = False


# --- 4. Optional input interpretation ----------------------------------------

# None uses the normal detector-specific prompt interval. Set both bounds for
# an explicit prompt window.
PROMPT_TIME_MIN_NS = None
PROMPT_TIME_MAX_NS = None

# Normally leave these mapping controls unchanged.
PMT_ID_MODE = "auto"
PMT_ID_OFFSET = 1
WCSIM_WCTE_MAPPING_FILE = ""  # Blank uses tables/wcsim_wcte_mapping.txt.

# None uses the historical WCSim detector-mode inactive-slot list. This setting
# applies only to WCSim; it is never used for real WCTE data.
INACTIVE_SLOTS = None


# --- 5. Checkpointing, performance, and output verbosity ---------------------

N_EVENTS_PER_BATCH = 100
WARM_FIT_KERNELS = True
SAVE_AFTER_EACH_BATCH = True
SAVE_DETAILED_EVENT_RESULTS = False

PRINT_EVENT_RESULTS = False
PRINT_BATCH_PROGRESS = True
PRINT_CHECKPOINT_MESSAGES = False
VERBOSE_SETUP = False

# Expert escape hatch for driver environment settings not listed above.
# Applied last, so matching names here override the settings above.
# Example: {"FIX_X0": 0.0, "FIX_Y0": 424.0, "FIX_Z0": -1350.0}
EXTRA_DRIVER_ENV = {}

# =============================================================================
# END WCSIM RUN CONFIGURATION -- USERS NORMALLY DO NOT EDIT BELOW THIS LINE
# =============================================================================


_DRIVER = Path(__file__).resolve().with_name("batch_fit_driver.py")
_PROJECT_ROOT = _DRIVER.parent.parent
_GEOMETRY_SUBMODULE = _PROJECT_ROOT / "Geometry"
_FIT_MODES = {"full_length", "absorption", "cosmic"}
_FIT_PARTICLES = {"muon", "pion", "kaon", "proton"}
_LIKELIHOODS = {"charge_only", "charge_time", "timing_only"}


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
    if FIT_MODE not in _FIT_MODES:
        raise ValueError(f"FIT_MODE must be one of {sorted(_FIT_MODES)}")
    if FIT_PARTICLE not in _FIT_PARTICLES:
        raise ValueError(f"FIT_PARTICLE must be one of {sorted(_FIT_PARTICLES)}")
    if LIKELIHOOD_MODE not in _LIKELIHOODS:
        raise ValueError(f"LIKELIHOOD_MODE must be one of {sorted(_LIKELIHOODS)}")
    if int(N_EVENTS) < 1 or int(EVENT_START_INDEX) < 0 or int(NPROC) < 1:
        raise ValueError("N_EVENTS and NPROC must be positive; EVENT_START_INDEX must be nonnegative")
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
        _require_file(str(_DRIVER), "batch_fit_driver.py")


def _encode(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value)
    return str(value)


def _configuration_items() -> list[tuple[str, object]]:
    if "GEOMETRY_PATH" in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV["GEOMETRY_PATH"]).strip():
        raise ValueError(
            "GEOMETRY_PATH overrides are no longer supported; remove "
            "GEOMETRY_PATH from EXTRA_DRIVER_ENV"
        )
    geometry_file = str(GEOMETRY_FILE).strip() or None
    mapping_file = str(WCSIM_WCTE_MAPPING_FILE).strip() or None
    output_file = str(OUTPUT_FILE).strip() or None
    inactive_slots = None if INACTIVE_SLOTS is None else tuple(INACTIVE_SLOTS)
    config_file = Path(__file__).resolve()
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    items = [
        ("LF_DATA_SOURCE", "wcsim"), ("DATA_SOURCE", "wcsim"),
        ("FIT_MODE", FIT_MODE), ("WCSIM_INPUT_FILE", INPUT_FILE),
        ("TOT_EVENTS", N_EVENTS), ("LF_EVENT_START_INDEX", EVENT_START_INDEX),
        ("FIT_PARTICLE", FIT_PARTICLE), ("WCSIM_PARTICLE_LABEL", FIT_PARTICLE),
        ("LIKELIHOOD_MODE", LIKELIHOOD_MODE), ("NPROC", NPROC),
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
        ("WCSIM_PMT_ID_MODE", PMT_ID_MODE), ("WCSIM_PMT_ID_OFFSET", PMT_ID_OFFSET),
        ("WCSIM_WCTE_MAPPING_PATH", mapping_file),
        ("INACTIVE_SLOTS", inactive_slots),
        ("N_EVENTS_PER_BATCH", N_EVENTS_PER_BATCH),
        ("WARM_FIT_KERNELS", WARM_FIT_KERNELS),
        ("SAVE_AFTER_EACH_BATCH", SAVE_AFTER_EACH_BATCH),
        ("SAVE_DETAILED_EVENT_RESULTS", SAVE_DETAILED_EVENT_RESULTS),
        ("PRINT_EVENT_RESULTS", PRINT_EVENT_RESULTS),
        ("PRINT_BATCH_PROGRESS", PRINT_BATCH_PROGRESS),
        ("PRINT_CHECKPOINT_MESSAGES", PRINT_CHECKPOINT_MESSAGES),
        ("VERBOSE_SETUP", VERBOSE_SETUP),
        ("LF_RUN_CONFIG_KIND", "wcsim"),
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
        "LF_EVENT_COUNT", "LF_WCTE_INTERNAL_PREPARED_EVENT_FILE",
        "LF_WCTE_PREPARE_EVENTS_ONLY", "LF_WCTE_PREPARED_EVENT_FILE",
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
    print(f"Geometry submodule: {_GEOMETRY_SUBMODULE}")
    for name, value in _configuration_items():
        shown = "<unset>" if value is None else _encode(value)
        print(f"{name}={shown}")


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
    args = parser.parse_args()
    try:
        _validate(check_paths=bool(args.check or not args.show_config))
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.show_config:
        _print_configuration()
        return
    if args.check:
        print("WCSim run configuration is valid.")
        return
    environment = build_environment()
    os.execve(sys.executable, [sys.executable, str(_DRIVER)], environment)


if __name__ == "__main__":
    main()
