#!/usr/bin/env python3
"""Edit this short configuration and run LicketyFit on WCSim data."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import sys


# =============================================================================
# WCSIM RUN CONFIGURATION -- EDIT THIS SECTION
# Settings are ordered from the most commonly changed to the least commonly
# changed. This file overrides matching values left in your shell environment.
# Shared physics, MCS, and seeding choices are all exposed below.  The launcher
# validates their compatibility before geometry, tables, or worker processes are
# initialized.
# =============================================================================

# --- 1. Data and event range -------------------------------------------------

# Digitized WCSim NPZ input.
INPUT_FILE = ""  # Required: set this to your WCSim NPZ file.

# Number of events to fit and the first input event index.
N_EVENTS = 100
EVENT_START_INDEX = 0

# Blank lets the driver construct its normal output filename.
OUTPUT_FILE = ""

# Number of independent event workers. One reproduces validated latency;
# increase this for production throughput after checking available memory.
NPROC = 1


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

# "charge_time", "charge_only", or "timing_only".
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

# Initial navigation for general charge+time fits. "off" retains the complete
# calibrated general seed bank. "hybrid" uses one compact, balanced family from
# each of two independent timing navigators (causal first-arrival timing and
# point multilateration); if either navigator cannot cover both physical start
# hypotheses, it restores the complete bank.  "guided" restricts the immutable
# bank only when its independent proxy certificate is decisive.  ``hybrid`` is
# the production default: it uses the compact independent navigators when both
# physical start hypotheses are covered and restores the complete calibrated
# bank whenever that safety certificate is absent.
COSMIC_MULTILATERATION_SEED_MODE = "hybrid"

# Every non-off general-mode MCS continuation requires charge. A timing_only
# general-mode run must use COSMIC_MCS_CONTINUATION="off" (or ENABLE_MCS=False).

# False uses charge shape only: each event's overall light normalization is
# profiled out. True retains absolute light and automatically selects the
# calibrated threshold-censored compound-SPE likelihood. This is the only
# switch a normal user needs to change between the two policies.
USE_ABSOLUTE_LIGHT_YIELD = True

# When absolute light is enabled, choose where its one detector-wide scale
# comes from:
#   "wcsim_calibration" -- the existing value calibrated on WCSim events;
#   "mathematical"      -- a ground-up Frank--Tamm/geometry/transmission/QE
#                          estimate that intentionally preserves both QE boosts
#                          present in the audited WCSim source.
ABSOLUTE_LIGHT_YIELD_SOURCE = "wcsim_calibration"

# Charge-response model used when USE_ABSOLUTE_LIGHT_YIELD is False.
# ``poisson_pe`` preserves the incoming fitter default;
# ``compound_spe_profile`` models the threshold-censored detector response.
CHARGE_LIKELIHOOD = "poisson_pe"
# This is fixed by the WCSim PMT/digitizer configuration used to make the NPZ
# files, not a fit tuning parameter.  The implementation reproduces the
# R14374-WCTE qpe table and SK-I stochastic threshold.
SPE_RESPONSE_MODEL = "wcsim_r14374_ski"
COMPOUND_PROFILE_MAX_ITER = 12
COMPOUND_PROFILE_TOL = 1.0e-9
COMPOUND_PROFILE_N_CAP = 256

# Required only when USE_ABSOLUTE_LIGHT_YIELD is True.  Point this at the
# immutable WCSim calibration manifest produced by
# analysis/calibrate_muon_absolute_light.py.  The manifest supplies the scale,
# split identity, optical configuration, and source hashes as one atomic unit.
# The packaged v3 calibration preserves the original fitter's direct charge
# shape: blacksheet reflection remains in first-arrival timing, while its mean
# charge yield is absorbed by the independently fitted global light scale.
GLOBAL_CHARGE_CALIBRATION_MANIFEST = (
    "tables/wcsim_muon_absolute_light_direct_shape_v3.json"
)

# Packaged, data-independent ground-up estimate. Normal users select this with
# ABSOLUTE_LIGHT_YIELD_SOURCE rather than editing the path.
MATHEMATICAL_CHARGE_CALIBRATION_MANIFEST = (
    "tables/wcsim_muon_absolute_light_ground_up_v2.json"
)

# Optional duplicated values for batch bookkeeping/backward configuration
# files. Leave these blank; when supplied they must exactly match the manifest.
GLOBAL_CHARGE_SCALE = None
GLOBAL_CHARGE_CALIBRATION_ID = ""

# Optional reporting-only conversion from the fitter's CSDA range convention
# to WCSim's above-Cherenkov-threshold visible-length convention.  It never
# changes the likelihood or MCS fit, and it is never used by run_wcte.py.
# The supplied v1 calibration is independently measured over 1,000 muons at
# each of 200, 300, and 400 MeV and has an operational domain of 180--425 MeV.
APPLY_WCSIM_VISIBLE_RANGE_CONVENTION = False
WCSIM_VISIBLE_RANGE_CONVENTION_MANIFEST = (
    "tables/wcsim_visible_range_convention_muon_200_400_v1.json"
)

# Appendix I of the physical-longitudinal handoff removes the v1.35 post-fit
# reporting gauge.  Event time remains a free fitted coordinate, and public
# fitted coordinates are the optimizer coordinates.  Keep these compatibility
# settings only so older launch automation receives an explicit disabled state;
# the production driver no longer implements the coordinate translation.
LONGITUDINAL_GAUGE_ENABLED = False
LONGITUDINAL_GAUGE_MAX_ABS_T0_NS = 0.10

# Coherent-MCS numerical completion. Two mandatory global updates keep routine
# events fast. A large exact-NLL gain opens a bounded adaptive continuation;
# a second nearly tied straight-line basin is tested only inside the narrow NLL
# gate, then selected by the coherent FE objective rather than by event truth.
COHERENT_TRACK_CYCLES = 2
COHERENT_ADAPTIVE_MAX_CYCLES = 12
COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL = 5.0
COHERENT_ADAPTIVE_STOP_GAIN_NLL = 0.5
COHERENT_ADAPTIVE_STOP_PATIENCE = 2
COHERENT_BASIN_CANDIDATES = 2
COHERENT_BASIN_GATE_NLL = 1.0

# In charge_time mode, the default keeps the sharply localized straight
# charge+time entrance line and jointly updates coherent path, continuous
# range, and event time.  This avoids aliasing the event-specific random MCS
# path into x/y and direction while timing still directly constrains range.
# "free" also reprofiles the entrance line and is retained as an experimental
# diagnostic until every nonprimary optical component follows the curved path.
COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY = "straight_conditioned"
COHERENT_JOINT_TIMING_RANGE_STEP_MM = 15.0
COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM = 2.0
COHERENT_JOINT_TIMING_RANGE_CYCLES = 1
COHERENT_JOINT_TIMING_GLOBAL_CYCLES = 4
COHERENT_JOINT_TIMING_LATENT_ITERATIONS = 2
COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS = 1

# Validated single-process latency policy.  These controls reduce redundant
# basin navigation and t0 sampling while preserving the exact charge+time
# objective, all PMTs, the 24-mode coherent-MCS state, and the absolute-light
# calibration.  A proxy dominance certificate skips the causal Hough guard
# only for an unmistakable nominal WCSim beam-cluster event; ambiguous events
# retain the complete guard.  EXTRA_DRIVER_ENV below can override any item.
ENABLE_VALIDATED_COSMIC_LATENCY_PROFILE = True
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

# --- 3. Detector geometry and seed label -------------------------------------

# Select exactly one detector mode.
USE_WCTE_GEOMETRY = True
USE_IWCD_GEOMETRY = False

# WCTE: blank uses Geometry/examples/wcte_bldg157.geo from the pinned Geometry
# submodule. IWCD: an explicit, independently validated .geo file is currently
# required; the upstream Geometry repository does not ship a serialized IWCD
# geometry and LicketyFit does not silently generate one.
GEOMETRY_FILE = ""

# Metadata and optional seed guidance only. It is not truth and does not
# constrain the beam/general fit.
ENERGY_LABEL_MEV = 200.0


# --- 4. Optional per-step WCSim truth diagnostics ----------------------------

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


# --- 5. Optional input interpretation ----------------------------------------

# None preserves the production peak-relative prompt selection implemented by
# the batch driver.  Set both bounds only for a deliberately fixed-window run,
# such as the separate 0--17 ns validation analyses.
PROMPT_TIME_MIN_NS = None
PROMPT_TIME_MAX_NS = None

# Normally leave these mapping controls unchanged.
PMT_ID_MODE = "auto"
PMT_ID_OFFSET = 1
WCSIM_WCTE_MAPPING_FILE = ""  # Blank uses tables/wcsim_wcte_mapping.txt.

# None uses the historical WCSim detector-mode inactive-slot list. This setting
# applies only to WCSim; it is never used for real WCTE data.
INACTIVE_SLOTS = None


# --- 6. Checkpointing, performance, and output verbosity ---------------------

N_EVENTS_PER_BATCH = 100
WARM_FIT_KERNELS = True
SAVE_AFTER_EACH_BATCH = True
SAVE_DETAILED_EVENT_RESULTS = False

# A failed per-event optimizer/physics certificate is saved as an explicit
# rejected row (NaN fitted coordinates plus its exception and traceback) while
# independent events continue. Set False only when debugging, where stopping at
# the first exception is more useful than completing the production ensemble.
CONTINUE_AFTER_EVENT_FAILURE = True

# If the ordinary straight-track fit succeeds but the optional coherent-MCS
# stage fails, retain the straight fit and mark MCS as failed/not applied.
# This is independent of CONTINUE_AFTER_EVENT_FAILURE because the event itself
# still has a valid reconstruction.
RETAIN_STRAIGHT_ON_MCS_FAILURE = True

PRINT_EVENT_RESULTS = False
PRINT_BATCH_PROGRESS = True
PRINT_CHECKPOINT_MESSAGES = False
VERBOSE_SETUP = False

# Expert escape hatch for non-physical driver settings not listed above.
# Applied last. Put physical track constraints in FIXED_PARAMETERS instead.
EXTRA_DRIVER_ENV = {}

# =============================================================================
# END WCSIM RUN CONFIGURATION -- USERS NORMALLY DO NOT EDIT BELOW THIS LINE
# =============================================================================


_DRIVER = Path(__file__).resolve().with_name("batch_fit_driver.py")
_PROJECT_ROOT = _DRIVER.parent.parent
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
        ("FIT_MODE", reconstruction.fit_mode), ("WCSIM_INPUT_FILE", INPUT_FILE),
        ("TOT_EVENTS", N_EVENTS), ("LF_EVENT_START_INDEX", EVENT_START_INDEX),
        ("FIT_PARTICLE", FIT_PARTICLE), ("WCSIM_PARTICLE_LABEL", FIT_PARTICLE),
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
        ("WCSIM_PMT_ID_MODE", PMT_ID_MODE), ("WCSIM_PMT_ID_OFFSET", PMT_ID_OFFSET),
        ("WCSIM_WCTE_MAPPING_PATH", mapping_file),
        ("INACTIVE_SLOTS", inactive_slots),
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
        ("LF_RUN_CONFIG_KIND", "wcsim"),
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
    for internal in (
        "LF_COSMIC_SUPERVISED_CHILD", "LF_COSMIC_CHILD_QUIET",
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
