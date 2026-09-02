#!/usr/bin/env python3
"""Edit this short configuration and run LicketyFit on WCSim data."""

from __future__ import annotations

# =============================================================================
# WCSIM RUN CONFIGURATION -- EDIT THIS SECTION
# Settings are ordered from the most commonly changed to the least commonly
# changed. This file overrides matching values left in your shell environment.
# Shared physics, MCS, and seeding choices are all exposed below.  The launcher
# validates their compatibility before geometry, tables, or worker processes are
# initialized.
# =============================================================================

# --- 1. Essential run settings ----------------------------------------------

# Digitized WCSim NPZ input.
INPUT_FILE = "/eos/user/j/jrimmer/sim_work_dir/WCSim/sim_data/mu-/1kmu-_400MeV_bp.npz"  # Required: set this to your WCSim NPZ file.

# Supported fit hypotheses: "muon", "pion", "kaon", or "proton".
FIT_PARTICLE = "muon"

# Describes the simulated sample and seeds the beam-mode range. It does not
# constrain a general-mode fit.
ENERGY_LABEL_MEV = 200.0

# Seed coverage and endpoint physics are independent choices.
# "general" searches detector-wide; "beam" uses the compact beam-pipe bank.
# "full_length" fits one threshold-range endpoint; "absorption" also fits a
# separate abrupt visible endpoint.
SEEDING_MODE = "general"          # "general" or "beam"
INTERACTION_MODE = "full_length"  # "full_length" or "absorption"

# "charge_time", "charge_only", or "timing_only".
LIKELIHOOD_MODE = "charge_time"

# Number of events to fit and the first input event index.
N_EVENTS = 500
EVENT_START_INDEX = 0

# Blank lets the driver construct its normal output filename.
OUTPUT_FILE = ""

# Number of independent event workers. One reproduces validated latency;
# increase this for production throughput after checking available memory.
NPROC = 16

# Bound the internal Numba/native-scatter teams used by each event worker. The
# launcher uses this cap for one event worker and one internal thread per worker
# when NPROC is larger than one, preventing nested oversubscription.
MAX_INTERNAL_THREADS_PER_WORKER = 4


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

# With automatic selection, search for the primary prompt peak only in this
# early window.  The subsequent 0..peak+5 ns hit window is unchanged.  This
# prevents a later, unrelated pulse from replacing the primary track.  The hit
# data alone do not identify the physical origin of that later pulse.
PRIMARY_PROMPT_PEAK_SEARCH_MAX_NS = 100.0

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

# The first execution of newly compiled Numba/native kernels can be both very
# slow and numerically different from a cache-loaded process. On a cold cache,
# run one disposable fit first and start production only after it exits cleanly.
AUTO_BOOTSTRAP_RUNTIME_CACHE = True

# Blank uses XDG_CACHE_HOME (or ~/.cache), which persists across lxplus jobs.
# Set an explicit shared, persistent directory if home-cache storage is limited.
RUNTIME_CACHE_DIR = ""

# The persistent cache above is the canonical cross-job copy. Production
# mirrors its small verified payload to TMPDIR or /tmp before workers start.
# Set this only when the batch system exposes a different node-local path.
NODE_LOCAL_RUNTIME_CACHE_DIR = ""

# Clean bounded process generations report a heartbeat at this cadence. If no
# atomic checkpoint advances for the timeout, the complete child/worker process
# group is stopped and its log is retained. Set the timeout to 0 to disable.
CHILD_HEARTBEAT_SECONDS = 30
CHILD_STALL_TIMEOUT_SECONDS = 600
# The worker-pool collector must diagnose a lost result before the outer
# process-group watchdog fires.  Keep a full minute for the child to checkpoint
# completed peers, terminate its pool, and publish the unresolved event IDs.
EVENT_RESULT_STALL_TIMEOUT_SECONDS = 540

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
# With one worker, report each completed event immediately instead of waiting
# for a whole N_EVENTS_PER_BATCH batch to finish.
PRINT_LIVE_EVENT_PROGRESS = True
PRINT_CHECKPOINT_MESSAGES = False
VERBOSE_SETUP = False

# Expert escape hatch for non-physical driver settings not listed above.
# Applied last. Put physical track constraints in FIXED_PARAMETERS instead.
EXTRA_DRIVER_ENV = {}

# =============================================================================
# END WCSIM RUN CONFIGURATION -- USERS NORMALLY DO NOT EDIT BELOW THIS LINE
# =============================================================================


# Load validation/execution helpers while keeping this file focused on user choices.
try:
    from scripts.launcher_loader import install_launcher as _install_launcher
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from launcher_loader import install_launcher as _install_launcher
_install_launcher(globals(), 'wcsim')
