#!/usr/bin/env python3
"""Edit this short configuration and run LicketyFit on real WCTE data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


# =============================================================================
# WCTE RUN CONFIGURATION -- EDIT THIS SECTION
# Settings are ordered from the most commonly changed to the least commonly
# changed. This file overrides matching values left in your shell environment.
# Shared physics, MCS, and seeding choices are all exposed below.  The launcher
# validates their compatibility before data loading, geometry, or workers start.
# =============================================================================

# --- 1. Data and event range -------------------------------------------------

# "selection": read the collaboration ROOT and apply DataLoader/BeamSelection.
# "file": read an already-selected NPY/NPZ/PKL/PICKLE event container.
EVENT_SOURCE = "selection"

# Collaboration run and optional exact merged-production ROOT override.
RUN = 2079
COLLABORATION_ROOT_FILE = ""

# Required only for EVENT_SOURCE="file". USER_EVENT_KEY can disambiguate a
# multi-array NPZ or mapping-like pickle.
USER_EVENT_FILE = ""
USER_EVENT_KEY = ""

# Selection mode scans at most N_ROOT_ENTRIES raw windows, then fits at most
# MAX_EVENTS_TO_FIT selected events after skipping EVENT_START_INDEX of them.
N_ROOT_ENTRIES = 500
MAX_EVENTS_TO_FIT = None
EVENT_START_INDEX = 0

# Nominal analysis_tools populations: muon, pion, electron, proton. Kaon
# requires SELECTION_MODE="custom" and explicit cuts below.
PARTICLE_SELECTION_LABEL = "muon"
SELECTION_MODE = "nominal"

# Blank lets the driver construct its normal output filename.
OUTPUT_FILE = ""

# Number of independent event workers.
NPROC = 8


# --- 2. Reconstruction -------------------------------------------------------

# "general" handles all inside/outside start/stop topologies.
# "beam" uses the compact beam-pipe seed bank and fits an internal start plus
# the remaining stopping range.
# "absorption" fits separate visible length and full range.
FIT_MODE = "general"

# Beam mode is fast because it normally searches only the calibrated cluster
# around the beam pipe. Set this True only for an intentional off-axis beam-mode
# study; it restores the expensive orientation and detector-global guard banks.
BEAM_USE_GLOBAL_SEED_GUARD = False

# Supported fit hypotheses: "muon", "pion", "kaon", or "proton".
FIT_PARTICLE = "muon"

# "charge_time" uses both prompt charge and first-arrival timing. Other
# choices are "charge_only" and "timing_only".
LIKELIHOOD_MODE = "charge_time"

# Optional exact physical constraints. Leave empty to fit every coordinate.
# Set any mode-compatible subset using these names and units:
#   x0_mm, y0_mm, z0_mm, direction=(x,y,z), t0_ns
#   general:     full_range_mm OR kinetic_energy_mev
#   beam:        length_mm
#   absorption:  visible_length_mm plus full_range_mm OR kinetic_energy_mev
# Direction is fixed as one complete vector and is normalized automatically;
# individual direction components cannot be fixed independently without
# violating the unit-vector constraint.
FIXED_PARAMETERS = {}

# --- Physics processes -------------------------------------------------------

# These four switches are independent and default to the validated all-physics
# configuration. MCS is a single master switch routed to exactly one
# mode-specific implementation. Reflection always participates in first-arrival
# timing when enabled; charge inclusion is fixed by the selected absolute-light
# manifest and is off in event-normalized shape-only operation.
ENABLE_DELTA_ELECTRONS = True
ENABLE_MCS = True
ENABLE_REFLECTION = True
ENABLE_PHOTON_SCATTERING = True

# Beam/absorption primary-MCS model: "coherent_fisher",
# "fermi_eyges_process", or "legacy".  The coherent implementation can be
# "auto", "physics_reference", "fast12_profile", or "legacy_fisher".  Auto
# selects physics_reference for beam and fast12_profile for absorption or
# general operation. Both primary-MCS selectors are inactive in general mode;
# general-mode scattering is selected independently below.
PRIMARY_MCS_MODEL = "coherent_fisher"
COHERENT_MCS_IMPLEMENTATION = "auto"

# General-mode MCS continuation: "off", "linear_fermi_eyges",
# "coherent_fermi_eyges", "joint_k0_range_gaussian_fe", or the experimental
# "joint_k0_range_mixed_mcs".  "auto" inference chooses Laplace/cubature except
# for the mixed model, where it selects the required reference SMC engine.
COSMIC_MCS_CONTINUATION = "coherent_fermi_eyges"
COSMIC_JOINT_INFERENCE_METHOD = "auto"

# Same truth-blind general navigator choices as run_wcsim.py. "hybrid" uses
# causal timing and point multilateration only when both physical start
# hypotheses are represented; otherwise the complete immutable bank is
# restored. WCTE's peak-relative timing preparation remains unchanged.
COSMIC_MULTILATERATION_SEED_MODE = "hybrid"

# Every non-off general-mode MCS continuation requires charge. A timing_only
# general-mode run must use COSMIC_MCS_CONTINUATION="off" (or ENABLE_MCS=False).

# False uses charge shape only: each event's overall light normalization is
# profiled out. True retains absolute light and automatically selects the
# calibrated threshold-censored compound-SPE likelihood. This is the only
# switch a normal user needs to change between the two policies.
USE_ABSOLUTE_LIGHT_YIELD = False

# When absolute light is enabled, select either an in-situ/beam measurement or
# the packaged ground-up optical estimate:
#   "measured"     -- user supplies GLOBAL_CHARGE_CALIBRATION_MANIFEST;
#   "mathematical" -- Frank--Tamm, active geometry, water/gel transmission,
#                     and unboosted R14374 QE/collection efficiency.
# Unlike the WCSim mathematical option, this deliberately does NOT apply
# WCSim's 1/0.73 and 1/(1-0.25) QE multipliers.
ABSOLUTE_LIGHT_YIELD_SOURCE = "measured"

# Charge-response model used when USE_ABSOLUTE_LIGHT_YIELD is False.
# ``poisson_pe`` preserves the incoming fitter default;
# ``compound_spe_profile`` models the threshold-censored detector response.
CHARGE_LIKELIHOOD = "poisson_pe"
COMPOUND_PROFILE_MAX_ITER = 12
COMPOUND_PROFILE_TOL = 1.0e-9
COMPOUND_PROFILE_N_CAP = 256

# --- 3. Authoritative active-PMT source ---------------------------------------

# "auto": use GOOD_PMT_FILE when nonblank, otherwise discover the mask for RUN.
# "file": require GOOD_PMT_FILE.
# "run": read Configuration/good_wcte_pmts from a DQ/merged ROOT for RUN.
GOOD_PMT_SOURCE = "auto"

# NPY/NPZ/TXT/CSV/JSON list of active WCTE PMTs. Required for source "file".
GOOD_PMT_FILE = ""
GOOD_PMT_FILE_KEY = ""  # Usually blank; selects an array in an ambiguous NPZ.

# Optional exact standalone DQ or merged ROOT override for source "run".
# This file supplies only good_wcte_pmts; it never replaces USER_EVENT_FILE.
GOOD_PMT_ROOT_FILE = ""

# Optional additional directories searched for a run ROOT before built-in EOS
# locations, for example ("/my/production", "/another/location").
GOOD_PMT_ROOT_SEARCH_BASES = ()


# --- 4. Beam population and event selection ----------------------------------
# These settings apply only when EVENT_SOURCE = "selection".

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

# Real-data charge is converted to PE with CHARGE_ADC_PER_PE.  Its SPE response
# is calibrated independently of WCSim; the manifest binds these Gaussian
# response parameters as well as the mPMT efficiency corrections below.
SPE_RESPONSE_MODEL = "gaussian_censored"

# Required only when USE_ABSOLUTE_LIGHT_YIELD is True.  A WCTE manifest is
# calibrated after the active-PMT mask, ADC-to-PE conversion, geometry, and
# per-mPMT relative-efficiency correction have been applied.  It is therefore
# detector/run-context-specific and a WCSim manifest is rejected.
GLOBAL_CHARGE_CALIBRATION_MANIFEST = ""

# Packaged, data-independent nominal-PMT estimate. The realized active-PMT mask
# and selected slot/type efficiency corrections are still applied explicitly by
# the WCTE runtime and are recorded in every output.
MATHEMATICAL_CHARGE_CALIBRATION_MANIFEST = (
    "tables/wcte_muon_absolute_light_ground_up_v1.json"
)

# Optional duplicated values for old configuration files. Leave these blank;
# when supplied they must exactly match the immutable WCTE manifest.
GLOBAL_CHARGE_SCALE = None
GLOBAL_CHARGE_CALIBRATION_ID = ""

# Coherent-MCS numerical completion. This is independent of the WCTE
# absolute-light calibration above: it controls only truth-blind optimization
# over the coherent path and nearly tied global-line basins.
COHERENT_TRACK_CYCLES = 2
COHERENT_ADAPTIVE_MAX_CYCLES = 12
COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL = 5.0
COHERENT_ADAPTIVE_STOP_GAIN_NLL = 0.5
COHERENT_ADAPTIVE_STOP_PATIENCE = 2
COHERENT_BASIN_CANDIDATES = 2
COHERENT_BASIN_GATE_NLL = 1.0
# Keep the accepted straight charge+time entrance line while the coherent
# stage jointly updates random path, range, and t0.  "free" is an experimental
# diagnostic that also lets the random-path stage reprofile x/y and direction.
COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY = "straight_conditioned"
COHERENT_JOINT_TIMING_RANGE_STEP_MM = 30.0
COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM = 2.0
COHERENT_JOINT_TIMING_RANGE_CYCLES = 8
COHERENT_JOINT_TIMING_GLOBAL_CYCLES = 4
COHERENT_JOINT_TIMING_LATENT_ITERATIONS = 2
COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS = 1

# The WCSim latency profile is now available on equal functional footing, but
# remains disabled for real data until it has its own WCTE validation sample.
# Enabling it changes navigation/CPU budgets only, never the optical objective,
# PMT mask, ADC conversion, prompt definition, or relative efficiencies.
ENABLE_VALIDATED_COSMIC_LATENCY_PROFILE = False
VALIDATED_COSMIC_LATENCY_ENV = {
    "LF_VALIDATED_COSMIC_LATENCY_PROFILE": "v1.35.0",
    "COSMIC_DIRECTION_FAN_ENABLED": 0,
    "AUTO_ENABLE_RANGE_PROFILE": 0,
    "AUTO_INTERNAL_PROXY_CANDIDATES": 3,
    "AUTO_BOUNDARY_PROXY_CANDIDATES": 2,
    "AUTO_EXACT_RERANK_PER_TOPOLOGY": 2,
    "COSMIC_TOURNAMENT_MAX_PROBES": 2,
    "COSMIC_TOURNAMENT_MAX_CONTINUATIONS": 2,
    "COSMIC_TOURNAMENT_CAUSAL_SUBSTITUTION_GAIN_NLL": 150.0,
    "COSMIC_TOURNAMENT_CERTIFIED_INTERNAL_ONLY": 1,
    "COSMIC_SAME_TOPOLOGY_CHALLENGES": 0,
    "T0_PROFILE_MAX_GLOBAL_POINTS": 3,
    "T0_PROFILE_SEED_HALF_WIDTH_NS": 0.5,
    "T0_PROFILE_REFINE_LEVELS": 1,
    "COSMIC_COHERENT_JOINT_TIMING_HARD_LATENT_BONUS": 1,
    "COSMIC_CAUSAL_LAZY_BEAM_DOMINANCE_NLL": 100.0,
    "COSMIC_FE_CHARGE_SWEEPS": 1,
    "COSMIC_TWO_ANCHOR_LAZY_PROBE_CERTIFICATE": 1,
    "COSMIC_TWO_ANCHOR_LAZY_MAX_PROBE_GAIN_NLL": 0.75,
    "COSMIC_TWO_ANCHOR_LAZY_MAX_CONTINUATION_GAIN_NLL": 8.0,
    "COSMIC_TWO_ANCHOR_STATIONARITY_PROBE": 1,
    "COSMIC_TWO_ANCHOR_STATIONARITY_MAX_GAIN_NLL": 0.05,
    "COSMIC_TOURNAMENT_ANCHOR_PROBE_MIN_CONTINUATION_GAIN_NLL": 0.0,
    "LF_COHERENT_ANALYTIC_CHARGE_SCORE": 1,
    "LF_COHERENT_LOCAL_T0_CERTIFICATE": 1,
    "LF_COHERENT_TRACK_ONE_SIDED_HALF_STEP": 1,
    "LF_COHERENT_PIPELINE_WARMUP": 1,
    "EMITTER_PHOTON_SCATTER_NATIVE_THREADS": 6,
}

PROMPT_WINDOW_MODE = "peak_relative"  # "peak_relative", "fixed", or "none"
PROMPT_TIME_MIN_NS = None              # Required only for "fixed" mode.
PROMPT_TIME_MAX_NS = None
USER_EVENT_APPLY_PROMPT_WINDOW = True
STRICT_USER_EVENT_VALIDATION = True

# Established real-data timing reference.
TIME_REFERENCE_MODE = "beam_corrected_peak"
TIME_REFERENCE_BIN_WIDTH_NS = 0.5
TIME_REFERENCE_LOCAL_HALF_WIDTH_NS = 1.0


# --- 6. Geometry and selection I/O -------------------------------------------

# Blank uses Geometry/examples/wcte_bldg157.geo from the pinned Geometry
# submodule. Set a file only when deliberately using another serialized detector.
GEOMETRY_FILE = ""

# Geometry and DataLoader/BeamSelection always come from the two pinned Git
# submodules at LicketyFit_official/Geometry and
# LicketyFit_official/analysis_tools. There are intentionally no source-path
# settings here. Initialize both once with:
# git submodule update --init analysis_tools Geometry
SELECTION_STEP_SIZE = "100 MB"


# --- 7. Checkpointing, performance, and output verbosity ---------------------

N_EVENTS_PER_BATCH = 100
WARM_FIT_KERNELS = True
SAVE_AFTER_EACH_BATCH = True
SAVE_DETAILED_EVENT_RESULTS = False

# Save a failed event as an explicit rejected row and continue independent
# events. Set False for stop-at-first-error debugging.
CONTINUE_AFTER_EVENT_FAILURE = True

# If the straight fit succeeds but optional coherent MCS fails, retain the
# valid straight solution and record the MCS failure instead of rejecting the
# entire event.
RETAIN_STRAIGHT_ON_MCS_FAILURE = True

PRINT_EVENT_RESULTS = False
PRINT_BATCH_PROGRESS = True
PRINT_CHECKPOINT_MESSAGES = False
VERBOSE_SETUP = False
PRINT_SELECTION_DESCRIPTION = True
PRINT_CHERENKOV_THRESHOLDS = True

# Expert escape hatch for non-physical driver settings not listed above.
# Applied last. Put physical track constraints in FIXED_PARAMETERS instead.
EXTRA_DRIVER_ENV = {}

# =============================================================================
# END WCTE RUN CONFIGURATION -- USERS NORMALLY DO NOT EDIT BELOW THIS LINE
# =============================================================================


_DRIVER = Path(__file__).resolve().with_name("batch_fit_driver.py")
_PROJECT_ROOT = _DRIVER.parent.parent
_ANALYSIS_TOOLS_SUBMODULE = _PROJECT_ROOT / "analysis_tools"
_GEOMETRY_SUBMODULE = _PROJECT_ROOT / "Geometry"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
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
from LicketyFit.run_configuration import (
    FIT_PARTICLES as _FIT_PARTICLES,
    LIKELIHOOD_MODES as _LIKELIHOODS,
    resolve_reconstruction_configuration,
)


_PUBLIC_DRIVER_RELEASE = "2026-08-31-v1.43.0-mode-rename-beam-speed"
_DEDICATED_RECONSTRUCTION_ENV = {
    "EMITTER_PRIMARY_MCS_MODEL",
    "MCS_COHERENT_IMPLEMENTATION",
    "EMITTER_COSMIC_MCS_CONTINUATION",
    "EMITTER_COSMIC_JOINT_INFERENCE_METHOD",
}


def _resolved_reconstruction_configuration():
    return resolve_reconstruction_configuration(
        fit_mode=FIT_MODE,
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
            fit_mode=FIT_MODE,
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
    dict | None,
]:
    """Resolve and bind the WCTE shape-versus-absolute charge contract."""
    shape_mode = str(CHARGE_LIKELIHOOD).strip().lower().replace("-", "_")
    if shape_mode not in _CHARGE_LIKELIHOODS:
        raise ValueError(
            f"CHARGE_LIKELIHOOD must be one of {sorted(_CHARGE_LIKELIHOODS)}"
    )
    if bool(USE_ABSOLUTE_LIGHT_YIELD):
        source = str(ABSOLUTE_LIGHT_YIELD_SOURCE).strip().lower().replace(
            "-", "_"
        )
        if source not in {"measured", "mathematical"}:
            raise ValueError(
                "ABSOLUTE_LIGHT_YIELD_SOURCE must be 'measured' or "
                "'mathematical'"
            )
        manifest_value = str(
            MATHEMATICAL_CHARGE_CALIBRATION_MANIFEST
            if source == "mathematical"
            else GLOBAL_CHARGE_CALIBRATION_MANIFEST
        ).strip()
        if not manifest_value:
            raise ValueError(
                "USE_ABSOLUTE_LIGHT_YIELD=True requires the selected "
                "WCTE GLOBAL_CHARGE_CALIBRATION_MANIFEST or the packaged "
                "mathematical manifest"
            )
        scale, calibration_id, path, manifest_sha256, manifest = (
            resolve_manifest_calibration(
                manifest_value,
                expected_detector="wcte",
                expected_particle=FIT_PARTICLE,
                project_root=_PROJECT_ROOT,
                manual_scale=GLOBAL_CHARGE_SCALE,
                manual_calibration_id=GLOBAL_CHARGE_CALIBRATION_ID,
            )
        )
        basis = str(manifest.get("calibration_basis", "")).strip().lower()
        if source == "mathematical":
            if basis != "ground_up_mathematical":
                raise ValueError(
                    "the mathematical WCTE option requires a "
                    "ground_up_mathematical manifest"
                )
            ground_up = manifest.get("ground_up_light_model", {})
            if ground_up.get("include_wcsim_qe_boosts") is not False:
                raise ValueError(
                    "the WCTE mathematical manifest must use the unboosted "
                    "R14374 QE convention"
                )
        elif basis == "ground_up_mathematical":
            raise ValueError(
                "the measured option cannot select a mathematical manifest"
            )
        context = manifest.get("wcte_calibration_context")
        if not isinstance(context, dict):
            raise ValueError(
                "the WCTE absolute-light manifest lacks "
                "wcte_calibration_context"
            )
        relative_context = context.get("relative_efficiency")
        if not isinstance(relative_context, dict):
            raise ValueError(
                "the WCTE absolute-light manifest lacks its mPMT "
                "relative-efficiency context"
            )
        active_rel_eff = str(RELATIVE_EFFICIENCY_MODE).strip().lower().replace(
            "-", "_"
        )
        if source == "mathematical":
            if context.get("runtime_active_mask_policy") != "per_sensor":
                raise ValueError(
                    "the mathematical WCTE manifest must declare a per-sensor "
                    "runtime active-mask policy"
                )
            allowed = relative_context.get("allowed_runtime_modes", [])
            if active_rel_eff not in set(map(str, allowed)):
                raise ValueError(
                    "the active WCTE relative-efficiency mode is not allowed "
                    "by the mathematical manifest"
                )
        else:
            calibrated_rel_eff = str(relative_context.get("mode", "")).strip().lower().replace(
                "-", "_"
            )
            if calibrated_rel_eff != active_rel_eff:
                raise ValueError(
                    "The measured absolute-light calibration was made with "
                    f"RELATIVE_EFFICIENCY_MODE={calibrated_rel_eff!r}, but the "
                    f"active mode is {active_rel_eff!r}"
                )
            if not math.isclose(
                float(context.get("charge_adc_per_pe", math.nan)),
                float(CHARGE_ADC_PER_PE),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "The measured absolute-light calibration CHARGE_ADC_PER_PE "
                    "does not match the active WCTE conversion"
                )
            if str(context.get("geometry_placement", "")) != str(GEOMETRY_PLACEMENT):
                raise ValueError(
                    "The measured absolute-light calibration geometry placement "
                    "does not match the active WCTE geometry"
                )
        response = manifest.get("pmt_charge_response", {})
        if response.get("model") != str(SPE_RESPONSE_MODEL):
            raise ValueError(
                "The WCTE absolute-light calibration PMT charge response does "
                f"not match SPE_RESPONSE_MODEL={SPE_RESPONSE_MODEL!r}"
            )
        reflection_in_charge = manifest.get("model_contract", {}).get(
            "analytic_reflection_in_charge"
        )
        if not isinstance(reflection_in_charge, bool):
            raise ValueError(
                "the WCTE absolute-light manifest must declare boolean "
                "model_contract.analytic_reflection_in_charge"
            )
        reflection_charge_policy = str(
            manifest.get("model_contract", {}).get(
                "analytic_reflection_charge_policy", "unconditional"
            )
        ).strip().lower().replace("-", "_")
        if reflection_charge_policy not in {
            "unconditional",
            "prompt_group_gated",
        }:
            raise ValueError(
                "the WCTE absolute-light reflection charge policy must be "
                "unconditional or prompt_group_gated"
            )
        return (
            "compound_spe_calibrated",
            "global_scale",
            scale,
            calibration_id,
            path,
            manifest_sha256,
            manifest,
        )
    if shape_mode == "compound_spe_calibrated":
        raise ValueError(
            "Set USE_ABSOLUTE_LIGHT_YIELD=True to use compound_spe_calibrated"
        )
    return shape_mode, "event_mean", None, "", None, None, None


def _validate_coherent_numerics() -> None:
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


def _require_file(value: str, label: str) -> None:
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")


def _require_analysis_tools_submodule() -> None:
    required = (
        _ANALYSIS_TOOLS_SUBMODULE / "analysis_tools" / "data_loader.py",
        _ANALYSIS_TOOLS_SUBMODULE / "analysis_tools" / "beam_selection.py",
    )
    if not all(path.is_file() for path in required):
        raise ValueError(
            "The analysis_tools submodule is missing or uninitialized at "
            f"{_ANALYSIS_TOOLS_SUBMODULE}. From LicketyFit_official run: "
            "git submodule update --init analysis_tools"
        )


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
            "EMITTER_REFLECTION_IN_CHARGE is fixed by the selected WCTE "
            "absolute-light calibration manifest; remove the override from "
            "EXTRA_DRIVER_ENV"
        )
    if (
        bool(USE_ABSOLUTE_LIGHT_YIELD)
        and "EMITTER_REFLECTION_CHARGE_POLICY" in EXTRA_DRIVER_ENV
    ):
        raise ValueError(
            "EMITTER_REFLECTION_CHARGE_POLICY is fixed by the selected WCTE "
            "absolute-light calibration manifest; remove the override from "
            "EXTRA_DRIVER_ENV"
        )
    _validate_coherent_numerics()
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
    if float(CHARGE_ADC_PER_PE) <= 0.0:
        raise ValueError("CHARGE_ADC_PER_PE must be positive")
    if float(TIME_REFERENCE_BIN_WIDTH_NS) <= 0.0 or float(TIME_REFERENCE_LOCAL_HALF_WIDTH_NS) < 0.0:
        raise ValueError("Time-reference bin width must be positive and local half-width nonnegative")
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
    forbidden_path_overrides = {
        name for name in ("WCTE_ANALYSIS_TOOLS_PATH", "ANALYSIS_TOOLS_PATH")
        if name in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV[name]).strip()
    }
    if forbidden_path_overrides:
        raise ValueError(
            "analysis_tools path overrides are no longer supported; remove "
            + ", ".join(sorted(forbidden_path_overrides))
            + " from EXTRA_DRIVER_ENV and use the repository submodule"
        )
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
        if EVENT_SOURCE == "file":
            _require_file(USER_EVENT_FILE, "USER_EVENT_FILE")
        if EVENT_SOURCE == "selection" and str(COLLABORATION_ROOT_FILE).strip():
            _require_file(COLLABORATION_ROOT_FILE, "COLLABORATION_ROOT_FILE")
        use_user_mask = GOOD_PMT_SOURCE == "file" or (
            GOOD_PMT_SOURCE == "auto" and bool(str(GOOD_PMT_FILE).strip())
        )
        if EVENT_SOURCE == "selection" or not use_user_mask:
            _require_analysis_tools_submodule()
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
    forbidden_path_overrides = {
        name for name in ("WCTE_ANALYSIS_TOOLS_PATH", "ANALYSIS_TOOLS_PATH")
        if name in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV[name]).strip()
    }
    if forbidden_path_overrides:
        raise ValueError(
            "analysis_tools path overrides are no longer supported; remove "
            + ", ".join(sorted(forbidden_path_overrides))
            + " from EXTRA_DRIVER_ENV"
        )
    if "GEOMETRY_PATH" in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV["GEOMETRY_PATH"]).strip():
        raise ValueError(
            "GEOMETRY_PATH overrides are no longer supported; remove "
            "GEOMETRY_PATH from EXTRA_DRIVER_ENV"
        )
    root_file = str(COLLABORATION_ROOT_FILE).strip() or None
    geometry_file = str(GEOMETRY_FILE).strip() or None
    output_file = str(OUTPUT_FILE).strip() or None
    (
        charge_likelihood,
        normalization_mode,
        global_scale,
        calibration_id,
        calibration_manifest,
        calibration_manifest_sha256,
        calibration_payload,
    ) = _resolved_charge_configuration()
    calibration_context = (
        calibration_payload.get("wcte_calibration_context", {})
        if calibration_payload is not None
        else {}
    )
    mathematical_calibration = bool(
        calibration_payload is not None
        and calibration_payload.get("calibration_basis")
        == "ground_up_mathematical"
    )
    calibration_rel_eff = (
        {"mode": str(RELATIVE_EFFICIENCY_MODE)}
        if mathematical_calibration
        else calibration_context.get("relative_efficiency", {})
    )
    calibration_adc_per_pe = (
        float(CHARGE_ADC_PER_PE)
        if mathematical_calibration
        else calibration_context.get("charge_adc_per_pe")
    )
    calibration_geometry_placement = (
        str(GEOMETRY_PLACEMENT)
        if mathematical_calibration
        else calibration_context.get("geometry_placement")
    )
    calibrated_reflection_in_charge = (
        calibration_payload.get("model_contract", {}).get(
            "analytic_reflection_in_charge"
        )
        if calibration_payload is not None
        else None
    )
    calibrated_reflection_charge_policy = (
        str(
            calibration_payload.get("model_contract", {}).get(
                "analytic_reflection_charge_policy", "unconditional"
            )
        ).strip().lower().replace("-", "_")
        if calibration_payload is not None
        else None
    )
    config_file = Path(__file__).resolve()
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    items = [
        ("LF_DATA_SOURCE", "wcte"), ("DATA_SOURCE", "wcte"),
        ("FIT_MODE", reconstruction.fit_mode), ("FIT_PARTICLE", FIT_PARTICLE),
        ("LIKELIHOOD_MODE", LIKELIHOOD_MODE), ("NPROC", NPROC),
        ("EMITTER_ENABLE_DELTA_E", ENABLE_DELTA_ELECTRONS),
        ("EMITTER_ENABLE_MCS", ENABLE_MCS),
        ("EMITTER_ENABLE_REFLECTION", ENABLE_REFLECTION),
        ("EMITTER_ENABLE_RAYLEIGH", ENABLE_PHOTON_SCATTERING),
        (
            "COSMIC_MULTILATERATION_SEED_MODE",
            reconstruction.seed_mode,
        ),
        (
            "WCTE_INCLUDE_ORIENTATION_GUARD",
            BEAM_USE_GLOBAL_SEED_GUARD
            if reconstruction.fit_mode == "beam" else None,
        ),
        (
            "WCTE_INCLUDE_DETECTOR_GLOBAL",
            BEAM_USE_GLOBAL_SEED_GUARD
            if reconstruction.fit_mode == "beam" else None,
        ),
        *reconstruction.environment().items(),
        ("PMT_CHARGE_LIKELIHOOD", charge_likelihood),
        ("PMT_SPE_RESPONSE_MODEL", SPE_RESPONSE_MODEL),
        ("PMT_COMPOUND_PROFILE_MAX_ITER", COMPOUND_PROFILE_MAX_ITER),
        ("PMT_COMPOUND_PROFILE_TOL", COMPOUND_PROFILE_TOL),
        ("PMT_COMPOUND_PROFILE_N_CAP", COMPOUND_PROFILE_N_CAP),
        ("WCTE_USE_ABSOLUTE_LIGHT_YIELD", USE_ABSOLUTE_LIGHT_YIELD),
        (
            "WCTE_ABSOLUTE_LIGHT_YIELD_SOURCE",
            ABSOLUTE_LIGHT_YIELD_SOURCE if USE_ABSOLUTE_LIGHT_YIELD else "shape_only",
        ),
        ("WCTE_GLOBAL_CHARGE_CALIBRATION_ID", calibration_id),
        ("WCTE_GLOBAL_CHARGE_CALIBRATION_MANIFEST", calibration_manifest),
        (
            "WCTE_GLOBAL_CHARGE_CALIBRATION_MANIFEST_SHA256",
            calibration_manifest_sha256,
        ),
        (
            "WCTE_GLOBAL_CHARGE_CALIBRATION_REL_EFF_MODE",
            calibration_rel_eff.get("mode"),
        ),
        (
            "WCTE_GLOBAL_CHARGE_CALIBRATION_ADC_PER_PE",
            calibration_adc_per_pe,
        ),
        (
            "WCTE_GLOBAL_CHARGE_CALIBRATION_GEOMETRY_PLACEMENT",
            calibration_geometry_placement,
        ),
        (
            "EMITTER_REFLECTION_IN_CHARGE",
            calibrated_reflection_in_charge,
        ),
        (
            "EMITTER_REFLECTION_CHARGE_POLICY",
            calibrated_reflection_charge_policy,
        ),
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
        ("WCTE_CHARGE_NORMALIZATION_MODE", normalization_mode),
        ("WCTE_GLOBAL_CHARGE_SCALE", global_scale),
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
        ("WCTE_PROMPT_WINDOW_MODE", PROMPT_WINDOW_MODE),
        ("WCTE_PROMPT_TIME_MIN_NS", PROMPT_TIME_MIN_NS),
        ("WCTE_PROMPT_TIME_MAX_NS", PROMPT_TIME_MAX_NS),
        ("USER_EVENT_APPLY_PROMPT_WINDOW", USER_EVENT_APPLY_PROMPT_WINDOW),
        ("WCTE_STRICT_USER_EVENT_VALIDATION", STRICT_USER_EVENT_VALIDATION),
        ("WCTE_TIME_REFERENCE_MODE", TIME_REFERENCE_MODE),
        ("WCTE_TIME_REFERENCE_BIN_WIDTH_NS", TIME_REFERENCE_BIN_WIDTH_NS),
        ("WCTE_TIME_REFERENCE_LOCAL_HALF_WIDTH_NS", TIME_REFERENCE_LOCAL_HALF_WIDTH_NS),
        ("WCD_GEOMETRY_FILE", geometry_file), ("WCTE_GEOMETRY_FILE", geometry_file),
        ("GEOMETRY_PATH", None),
        ("LF_GEOMETRY_POLICY", "required_repository_submodule"),
        ("WCTE_ANALYSIS_TOOLS_PATH", None),
        ("ANALYSIS_TOOLS_PATH", None),
        ("LF_ANALYSIS_TOOLS_POLICY", "required_repository_submodule"),
        ("SELECTION_STEP_SIZE", SELECTION_STEP_SIZE),
        ("N_EVENTS_PER_BATCH", N_EVENTS_PER_BATCH),
        ("WARM_FIT_KERNELS", WARM_FIT_KERNELS),
        ("SAVE_AFTER_EACH_BATCH", SAVE_AFTER_EACH_BATCH),
        ("SAVE_DETAILED_EVENT_RESULTS", SAVE_DETAILED_EVENT_RESULTS),
        ("MCS_RECORD_EVENT_FAILURES", CONTINUE_AFTER_EVENT_FAILURE),
        ("MCS_RETAIN_STRAIGHT_ON_FAILURE", RETAIN_STRAIGHT_ON_MCS_FAILURE),
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
        ("LF_PUBLIC_DRIVER_RELEASE", _PUBLIC_DRIVER_RELEASE),
    ]
    if ENABLE_VALIDATED_COSMIC_LATENCY_PROFILE:
        items.extend(
            (str(name), value)
            for name, value in VALIDATED_COSMIC_LATENCY_ENV.items()
        )
    items.extend(_resolved_fixed_parameter_items())
    items.extend((str(name), value) for name, value in EXTRA_DRIVER_ENV.items())
    return items


def build_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return the exact environment handed to batch_fit_driver.py."""
    env = dict(os.environ if base is None else base)
    # This calibration converts the fitter's CSDA coordinate to a WCSim-only
    # reporting convention.  Never allow an inherited shell setting to enter
    # a real-data WCTE run.
    env.pop("WCSIM_VISIBLE_RANGE_CONVENTION_MANIFEST", None)
    for internal in (
        "LF_COSMIC_SUPERVISED_CHILD", "LF_COSMIC_CHILD_QUIET",
        "LF_WCTE_INTERNAL_PREPARED_EVENT_FILE", "LF_WCTE_PREPARE_EVENTS_ONLY",
        "LF_WCTE_PREPARED_EVENT_FILE", "LF_EVENT_COUNT",
    ):
        env.pop(internal, None)
    for name in _REMOVED_MODEL_ENV_SWITCHES:
        env.pop(name, None)
    for name, value in _configuration_items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = _encode(value)
    runtime_deps = _PROJECT_ROOT / "runtime_deps"
    if runtime_deps.is_dir():
        inherited = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(runtime_deps)] + ([inherited] if inherited else [])
        )
    return env


def _print_configuration() -> None:
    print(f"Launcher: {Path(__file__).resolve()}")
    print(f"Driver:   {_DRIVER}")
    print(f"analysis_tools submodule: {_ANALYSIS_TOOLS_SUBMODULE}")
    print(f"Geometry submodule:       {_GEOMETRY_SUBMODULE}")
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
