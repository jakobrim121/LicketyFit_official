#!/usr/bin/env python3
"""Edit this short configuration and run LicketyFit on real WCTE data."""

from __future__ import annotations

# =============================================================================
# WCTE RUN CONFIGURATION -- EDIT THIS SECTION
# Settings are ordered from the most commonly changed to the least commonly
# changed. This file overrides matching values left in your shell environment.
# Shared physics, MCS, and seeding choices are all exposed below.  The launcher
# validates their compatibility before data loading, geometry, or workers start.
# =============================================================================

# --- 1. Essential run settings ----------------------------------------------

# "selection": read the collaboration ROOT and apply DataLoader/BeamSelection.
# "file": read an already-selected NPY/NPZ/PKL/PICKLE event container.
EVENT_SOURCE = "selection"

# Collaboration run and optional exact merged-production ROOT override.
RUN = 1855
COLLABORATION_ROOT_FILE = ""

# Nominal analysis_tools populations: muon, pion, electron, proton. Kaon
# requires SELECTION_MODE="custom" and explicit cuts below.
PARTICLE_SELECTION_LABEL = "muon"
SELECTION_MODE = "nominal"

# Light-particle PID policy for nominal electron/muon/pion samples:
# "act_tof" combines the usual ACT identity with run-calibrated TOF boundaries;
# "tof" uses those TOF boundaries alone; "act" preserves the legacy ACT cuts.
# Proton selection remains TOF-based and kaons still require custom cuts.
LIGHT_PARTICLE_PID_MODE = "act_tof"

# Supported fit hypotheses: "muon", "pion", "kaon", or "proton".
FIT_PARTICLE = "muon"

# Seed coverage and endpoint physics are independent choices.
# "general" searches detector-wide; "beam" uses the compact beam-pipe bank.
# "full_length" fits one threshold-range endpoint; "absorption" also fits a
# separate abrupt visible endpoint.
SEEDING_MODE = "general"          # "general" or "beam"
INTERACTION_MODE = "full_length"  # "full_length" or "absorption"

# "charge_time" uses prompt charge and first-arrival timing. Other choices are
# "charge_only" and "timing_only".
LIKELIHOOD_MODE = "charge_time"

# Beam metadata and optional seed guidance; neither value is event truth.
BEAM_MOMENTUM_MEV_C = 780
EXPECTED_KINETIC_ENERGY_MEV = 300.0
USE_EXPECTED_ENERGY_SEED_HINT = False

# Required only for EVENT_SOURCE="file". USER_EVENT_KEY can disambiguate a
# multi-array NPZ or mapping-like pickle.
USER_EVENT_FILE = ""
USER_EVENT_KEY = ""

# Selection mode scans at most N_ROOT_ENTRIES raw windows, then fits at most
# MAX_EVENTS_TO_FIT selected events after skipping EVENT_START_INDEX of them.
N_ROOT_ENTRIES = 5000
MAX_EVENTS_TO_FIT = None
EVENT_START_INDEX = 0

# Blank lets the driver construct its normal output filename.
OUTPUT_FILE = ""

# Number of independent event workers.
NPROC = 16


# --- 2. Physical constraints and models -------------------------------------

# Beam mode is fast because it normally searches only the calibrated cluster
# around the beam pipe. Set this True only for an intentional off-axis beam-mode
# study; it restores the expensive orientation and detector-global guard banks.
BEAM_USE_GLOBAL_SEED_GUARD = False

# Optional exact physical constraints. Leave empty to fit every coordinate.
# Set any mode-compatible subset using these names and units:
#   x0_mm, y0_mm, z0_mm, direction=(x,y,z), t0_ns
#   full_length + beam:     length_mm
#   full_length + general:  length_mm OR full_range_mm OR kinetic_energy_mev
#   absorption: visible_length_mm plus full_range_mm OR kinetic_energy_mev
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

# Standard-engine primary-MCS model: "coherent_fisher",
# "fermi_eyges_process", or "legacy".  The coherent implementation can be
# "auto", "physics_reference", "fast12_profile", or "legacy_fisher".  Auto
# selects physics_reference for beam+full_length and fast12_profile for either
# absorption combination. Both selectors are inactive only in
# general+full_length; its scattering is selected independently below.
PRIMARY_MCS_MODEL = "coherent_fisher"
COHERENT_MCS_IMPLEMENTATION = "auto"

# General+full_length MCS continuation: "off", "linear_fermi_eyges",
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
# Availability policy for the proton/fast-particle boundary. Light e/mu/pi
# separation is controlled independently by LIGHT_PARTICLE_PID_MODE above.
TOF_CUT_MODE = "auto"  # "auto", "require", or "disable"
PROTON_TOF_WINDOW_NS = 10.0
REQUIRE_MUON_TAGGER = False

# None uses run-derived selection constants.
ACT_EVETO_CUT_OVERRIDE_PE = None
ACT_TAGGER_CUT_OVERRIDE_PE = None
PROTON_TOF_CUT_OVERRIDE_NS = None
MUON_TAG_CUT_OVERRIDE = None
ELECTRON_MUON_TOF_BOUNDARY_OVERRIDE_NS = None
MUON_PION_TOF_BOUNDARY_OVERRIDE_NS = None

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

# Exact execution accelerators.  These batch the complete coherent timing
# and local t0 stencils through compiled likelihood calls, continue a scheduled
# basin with its existing objective cache, reuse the geometry-only charge NLL
# across t0 probes, keep tiny KL eigensolves off a host-wide BLAS team, and
# evaluate the unchanged robust causal score without repeated NumPy dispatch.
# Independent PMTs in an additive-t0 grid use the same compiled thread team;
# their final likelihood is still reduced once in historical PMT-major order.
# They do not change PMTs, quadrature, physics terms, likelihood values,
# optimizer steps, or stopping criteria.
ENABLE_EXACT_EXECUTION_ACCELERATORS = True
EXACT_EXECUTION_ENV = {
    "LF_COHERENT_BATCHED_TIMING_RESPONSE": 1,
    "LF_COHERENT_DEFERRED_RESPONSE_BATCH": 1,
    "LF_BATCH_T0_BLOCK_STENCIL": 1,
    "LF_EXACT_OBJECTIVE_REUSE": 1,
    "LF_EXACT_CHARGE_NLL_REUSE": 1,
    "LF_EXACT_SMALL_KL_SINGLE_THREAD": 1,
    "LF_EXACT_COMPILED_CAUSAL_SCORE": 1,
    "LF_EXACT_PARALLEL_T0_GRID": 1,
}

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

# Abort a multiprocessing batch if no worker returns any result for this long.
# The deadline resets after every completed event, so slow overall throughput is
# allowed while a genuinely lost native-worker result cannot block forever.
EVENT_RESULT_STALL_TIMEOUT_SECONDS = 540

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


# Load validation/execution helpers while keeping this file focused on user choices.
try:
    from scripts.launcher_loader import install_launcher as _install_launcher
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from launcher_loader import install_launcher as _install_launcher
_install_launcher(globals(), 'wcte')
