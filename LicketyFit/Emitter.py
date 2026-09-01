# -----------------------------------------------------------------------------
# LicketyFit Emitter.py -- production/diagnostic layout (July 2026)
#
# Main physics defaults and environment-variable overrides are centralized in
# the USER-FACING EMITTER DEFAULTS block immediately after the imports.  Do not
# hunt through __init__ to decide whether deltas, primary MCS, or Rayleigh are on.
# -----------------------------------------------------------------------------
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import zipfile

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux/CERN production provides fcntl.
    fcntl = None

import numpy as np
from typing import List, Tuple
from numba import njit, prange


# =============================================================================
# USER-FACING EMITTER DEFAULTS
# =============================================================================
# Edit the DEFAULT_* values here, or set the matching environment variable before
# importing/constructing the Emitter. Environment variables override the optical
# process defaults and the two validated reconstruction-model selectors below.
# The public launchers record every resolved selector so the optical model and
# post-fit continuation cannot silently disagree. Accepted boolean values are:
#     on:  1, true, yes, y, on
#     off: 0, false, no, n, off
#
# Main physics switches:
#   EMITTER_ENABLE_DELTA_E=1|0
#   EMITTER_ENABLE_MCS=1|0
#   EMITTER_ENABLE_RAYLEIGH=1|0
#   EMITTER_ENABLE_REFLECTION=1|0
#
# Package defaults in this file:
#   delta electrons: ON
#   primary-particle MCS: ON (routed to exactly one mode-specific path)
#   cosmic MCS continuation: coherent Fermi--Eyges when MCS is ON
#   molecular Rayleigh/Raman scattering: ON
#   analytic one-bounce WCTE blacksheet reflection: ON for timing
#
# ``DEFAULT_ENABLE_PRIMARY_MCS`` is the historical name of the single MCS
# master switch. In contained/full-length operation it activates the ordinary
# primary-MCS implementation. In cosmic operation it leaves that ordinary path
# off and activates exactly one implementation selected by
# ``DEFAULT_COSMIC_MCS_CONTINUATION``. Turning the master switch off resolves the
# effective cosmic continuation to ``off`` as well. This routing prevents the
# ordinary and cosmic continuations from double-counting the same scattering.
# ``DEFAULT_COSMIC_JOINT_INFERENCE_METHOD`` selects the optional joint engine.
# ``DEFAULT_COSMIC_JOINT_INFERENCE_METHOD`` selects its numerical engine.  A run
# may override only these two selectors with the validated string settings
# ``EMITTER_COSMIC_MCS_CONTINUATION`` and
# ``EMITTER_COSMIC_JOINT_INFERENCE_METHOD``; the resolved values are recorded in
# every run configuration, avoiding source edits between comparison jobs.
# =============================================================================
TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
FALSE_STRINGS = {"0", "false", "no", "n", "off"}

# ---- reconstruction-model switches: edit these first -------------------------
# Historical public name retained for compatibility. This is now the single MCS
# master default for every fit mode; ``EMITTER_ENABLE_MCS`` is the per-run
# override forwarded by both public launchers.
DEFAULT_ENABLE_PRIMARY_MCS = True

# Optional geometry-clipped general-mode continuation used by BOTH run_wcte.py and
# run_wcsim.py when public FIT_MODE="general". Choose exactly one string:
#
#   "off"
#       Original straight-track general result; no post-fit MCS continuation.
#   "linear_fermi_eyges"
#       Historical linear Fermi--Eyges process continuation.
#   "coherent_fermi_eyges"
#       Nonlinear coherent FE path profile with deterministic inverse-range
#       energy unless a separately documented fixed-energy validation is used.
#   "joint_k0_range_gaussian_fe"
#       New charge-only continuous posterior over independent K0, stopping-range
#       straggling z_R, and the Gaussian coherent FE path.
#   "joint_k0_range_mixed_mcs"
#       Experimental charge-only reference SMC over K0, z_R, a soft coherent
#       Wentzel FE path, and explicit marked-Poisson hard scatters.  It is not a
#       production fast path and currently supports contained trajectories only.
# The joint choices require public FIT_MODE="general", a supported charge
# likelihood, and the mode-routed ordinary primary path to be inactive. They support
# charge_only and charge_time;
# timing_only is rejected because it has no charge-shape constraint. The driver
# validates these contracts.
# Production general-mode default: the nonlinear coherent Fermi--Eyges path model
# validated with the complete charge/time likelihood.  The ordinary straight
# fitter keeps primary MCS off because it supplies only the seed; enabling both
# would count the same scattering twice.
DEFAULT_COSMIC_MCS_CONTINUATION = "coherent_fermi_eyges"

# Numerical inference engine for the continuous joint model above:
#   "laplace_cubature"  production deterministic continuous fit;
#   "reference_smc"     slow annealed-SMC validation reference.
# Both choices infer free K0, continuous range straggling, and all configured
# Gaussian FE path modes with the same physical priors and optical likelihood.
DEFAULT_COSMIC_JOINT_INFERENCE_METHOD = "laplace_cubature"


# ---- optical-process on/off switches -----------------------------------------
DEFAULT_ENABLE_DELTA_E = True
DEFAULT_ENABLE_RAYLEIGH = True
DEFAULT_ENABLE_BLACKSHEET_REFLECTION = True

# ---- secondary-electron / delta model defaults -------------------------------
DEFAULT_DELTA_E_SCALE = 1.0
DEFAULT_ANALYTIC_DELTA_SCALE = 1.0
DEFAULT_DELTA_E_DISTANCE_PMT_RADIUS_MM = 45.0
DEFAULT_DELTA_E_COST_SOFT = 0.0

# ---- primary cone / endpoint defaults ----------------------------------------
# Effective phase index used by the single-cone primary-light approximation.
# The range tables retain their documented historical n=1.344 threshold; this
# optical setting is deliberately explicit so a wavelength-integrated detector
# response can be validated without silently rebuilding the range coordinate.
DEFAULT_WATER_PHASE_INDEX = 1.344
DEFAULT_PRIMARY_ENDPOINT_MODEL = "root_overlap_weight_only"
DEFAULT_PRIMARY_ENDPOINT_APERTURE_RADIUS_MM = 45.0
DEFAULT_PRIMARY_ENDPOINT_SCOPE = "start"
DEFAULT_PRIMARY_EDGE_MODEL = "legacy"
DEFAULT_PRIMARY_MCS_MODEL = "coherent_fisher"
DEFAULT_PRIMARY_MCS_ENERGY_MODE = "initial"
DEFAULT_PRIMARY_MCS_ENERGY_SAMPLES = 24
DEFAULT_PRIMARY_COST_SOFT = 0.02
DEFAULT_PRIMARY_COST_SOFT_CENTERED = True
DEFAULT_PRIMARY_SOFT_CONE_SIGMA_RAD = 0.009

# ---- detector-obstacle defaults ---------------------------------------------
# WCTE WCSim contains a polished stainless-steel central deployment system
# (CDS) that is not represented in the standalone Geometry package.  The
# generic Emitter default is deliberately OFF so historical contained-track
# modes and non-WCTE detectors remain unchanged.  Cosmic WCTE reconstruction
# enables the fixed detector geometry explicitly.
DEFAULT_ENABLE_WCTE_CDS_OCCLUSION = False
DEFAULT_WCTE_CDS_AXIS_X_MM = 0.0
DEFAULT_WCTE_CDS_AXIS_Z_MM = 0.0
DEFAULT_WCTE_CDS_INNER_RADIUS_MM = 72.40
DEFAULT_WCTE_CDS_OUTER_RADIUS_MM = 93.42
DEFAULT_WCTE_CDS_Y_MIN_MM = 1556.63
DEFAULT_WCTE_CDS_Y_MAX_MM = 1777.59
DEFAULT_WCTE_CDS_PMT_APERTURE_RADIUS_MM = 45.0
# One polished-metal reflection from the water-facing inner shaft wall.  This
# is separate from the direct-visibility switch because hard occlusion is a
# useful diagnostic, while a physical WCSim/IWCD model must restore the photons
# that reflect from the configured metal surface.  Generic/non-WCTE default is
# OFF; the WCSim cosmic driver enables it explicitly after detector checks.
DEFAULT_ENABLE_WCTE_CDS_SPECULAR_REFLECTION = False
DEFAULT_WCTE_CDS_SPECULAR_REFLECTIVITY = 0.90
DEFAULT_WCTE_CDS_SPECULAR_PHI_BINS = 72
DEFAULT_WCTE_CDS_SPECULAR_Y_BINS = 32
DEFAULT_WCTE_CDS_SPECULAR_TIMING_BINS = 12

# ---- fast analytic blacksheet-reflection defaults -----------------------------
# One primary-photon Lambertian bounce from the exact 16-sided WCTE blacksheet.
# The default keeps reflection in the TIMING source mixture only, matching the
# validated conditional-likelihood studies.  Charge inclusion is an explicit
# future/joint-likelihood option.
DEFAULT_REFLECTION_IN_CHARGE = False
# Charge policies when reflected light is enabled:
#
# ``unconditional`` preserves the historical analytic component.
# ``prompt_group_gated`` applies the exact Poisson probability that at least
# one non-blacksheet prompt PE opens a digitizer integration group on that PMT.
# It is the data-independent WCSim SK-I response for the late one-bounce term
# under the documented ordering approximation that prompt non-reflected light
# opens the selected group.  Reflected light remains fully available to
# first-arrival timing in both cases.
DEFAULT_REFLECTION_CHARGE_POLICY = "unconditional"
DEFAULT_REFLECTION_BSRFF = 2.5
DEFAULT_REFLECTION_PMT_APERTURE_RADIUS_MM = 45.0
DEFAULT_REFLECTION_TANGENT_BINS = 2
DEFAULT_REFLECTION_Y_BINS = 3
DEFAULT_REFLECTION_CAP_RADIAL_BINS = 3
DEFAULT_USE_FIRST_ARRIVAL_TIMING = True
DEFAULT_REFLECTION_FIRST_ARRIVAL_NODES = 24

# ---- photon molecular-scattering defaults ------------------------------------
# ``first_interaction`` is the physical Rayleigh + water-Raman transport.
DEFAULT_PHOTON_SCATTER_MODEL = "first_interaction"
DEFAULT_PHOTON_SCATTER_SPECTRAL_MODE = "moment"
DEFAULT_PHOTON_SCATTER_N_TRACK = 5
DEFAULT_PHOTON_SCATTER_N_AZIMUTH = 12
DEFAULT_PHOTON_SCATTER_N_INTERACTION = 4
DEFAULT_PHOTON_SCATTER_N_WAVELENGTH = 6
DEFAULT_PHOTON_SCATTER_N_RAMAN_SHIFT = 8
DEFAULT_PHOTON_SCATTER_N_TIMING_BINS = 16
DEFAULT_PHOTON_SCATTER_ENABLE_RAMAN = True
DEFAULT_PHOTON_SCATTER_DIRECT_SURVIVAL = True
DEFAULT_PHOTON_SCATTER_INCLUDE_MPMT_DOMES = True
DEFAULT_PHOTON_SCATTER_PMT_APERTURE_RADIUS_MM = 45.0
DEFAULT_PHOTON_SCATTER_PMT_FACING_SOFT_WIDTH = 0.02
DEFAULT_PHOTON_SCATTER_PARALLEL_PMT_LOOP = False
DEFAULT_PHOTON_SCATTER_RECEIVER_MODE = "sparse_moment"
DEFAULT_PHOTON_SCATTER_RECEIVER_TABLE = ""
DEFAULT_PHOTON_SCATTER_RECEIVER_TABLE_REQUIRED = False
DEFAULT_PHOTON_SCATTER_NATIVE_RECEIVER = True
DEFAULT_PHOTON_SCATTER_NATIVE_THREADS = 1
DEFAULT_PHOTON_SCATTER_NATIVE_REQUIRED = False
DEFAULT_PHOTON_SCATTER_BOUNDARY_MODEL = "auto"

# Legacy midpoint-Rayleigh settings, retained only for explicit A/B tests.
DEFAULT_RAYLEIGH_CACHE_MODE = "quantized"
DEFAULT_RAYLEIGH_SCATTER_LENGTH_MM = 120000.0
DEFAULT_RAYLEIGH_N_SOURCES = 6
DEFAULT_RAYLEIGH_N_PHI = 10
DEFAULT_RAYLEIGH_TIMING_CUT_NS = 5.0
DEFAULT_RAYLEIGH_TIMING_SOFT_MM = 150.0
DEFAULT_RAYLEIGH_USE_PARALLEL_ACCUMULATOR = False

# ---- likelihood / numerical smoothness defaults ------------------------------
# The charge floor is ONLY a charge-likelihood protection.  The timing likelihood
# must use the unfloored physical expectation saved by get_expected_pes_ts as
# _last_expected_pes_for_timing; otherwise the floor makes every observed PMT
# eligible for timing pulls even when the physical model predicts zero light.
DEFAULT_CHARGE_FLOOR_PE = 1e-4
DEFAULT_EVENT_MEAN_CONTAMINATION_MODEL = "off"
DEFAULT_EVENT_MEAN_CONTAMINATION_MAX_FRACTION = 0.50
DEFAULT_SMOOTH_TABLES = True
DEFAULT_USE_FUSED_PRIMARY = True


def _env_bool_switch(name, default, *aliases):
    """Read a boolean switch with a clear default and optional legacy aliases."""
    raw = None
    for key in (name, *aliases):
        val = os.environ.get(key)
        if val is not None and str(val).strip() != "":
            raw = val
            break
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in TRUE_STRINGS:
        return True
    if value in FALSE_STRINGS:
        return False
    raise ValueError(f"Environment variable {name} must be boolean-like, got {raw!r}")


def _env_float_switch(name, default, *aliases):
    """Read a float-valued Emitter setting with optional legacy aliases."""
    for key in (name, *aliases):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return float(raw)
    return float(default)


def _env_int_switch(name, default, *aliases):
    """Read an integer-valued Emitter setting with optional legacy aliases."""
    for key in (name, *aliases):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return int(float(raw))
    return int(default)


def _env_str_switch(name, default, *aliases):
    """Read a string-valued Emitter setting with optional legacy aliases."""
    for key in (name, *aliases):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return str(default)


def prompt_group_open_probability(expected_prompt_pe):
    """Return the Poisson probability that a PMT opens a prompt group.

    For a non-reflected prompt expectation ``mu``, the WCSim SK-I digitizer
    opens an integration group exactly when at least one accepted PE exists.
    Thus ``P(open) = 1 - exp(-mu)``.  The ``expm1`` form remains accurate for
    dim PMTs, where an unconditional reflected-charge term is most harmful.
    """

    values = np.asarray(expected_prompt_pe, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("expected prompt PE must be finite and nonnegative")
    return -np.expm1(-values)


def profile_event_mean_uniform_contamination(
    raw_signal,
    observed_charge,
    *,
    max_fraction=DEFAULT_EVENT_MEAN_CONTAMINATION_MAX_FRACTION,
    iterations=48,
):
    """Profile a broad channel-uniform contamination fraction.

    With event-total normalization, the charge shape is multinomial.  This
    profiles the convex one-parameter mixture

        p_i(epsilon) = (1-epsilon) s_i/sum(s) + epsilon/N,

    which gives isolated unmodelled PMTs bounded leverage without using event
    truth or an absolute light-yield calibration.  The broad component is a
    charge-only nuisance and deliberately supplies no timing prediction.
    """

    signal = np.asarray(raw_signal, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed_charge, dtype=np.float64).reshape(-1)
    if (
        signal.shape != observed.shape
        or signal.size == 0
        or np.any(~np.isfinite(signal))
        or np.any(signal < 0.0)
        or np.any(~np.isfinite(observed))
        or np.any(observed < 0.0)
    ):
        raise ValueError("contamination profile requires matching finite PE arrays")
    upper = float(max_fraction)
    if not math.isfinite(upper) or upper < 0.0 or upper >= 1.0:
        raise ValueError("contamination max_fraction must lie in [0,1)")
    signal_total = float(np.sum(signal))
    observed_total = float(np.sum(observed))
    if upper == 0.0 or signal_total <= 0.0 or observed_total <= 0.0:
        return 0.0

    p_signal = signal / signal_total
    p_uniform = 1.0 / float(signal.size)
    delta = p_uniform - p_signal
    active = observed > 0.0
    q = observed[active]
    p0 = p_signal[active]
    d = delta[active]

    def derivative(epsilon):
        probability = np.maximum(p0 + float(epsilon) * d, 1.0e-300)
        return -float(np.sum(q * d / probability))

    derivative_zero = derivative(0.0)
    if derivative_zero >= 0.0:
        return 0.0
    derivative_upper = derivative(upper)
    if derivative_upper <= 0.0:
        return upper
    lo, hi = 0.0, upper
    for _ in range(max(1, int(iterations))):
        mid = 0.5 * (lo + hi)
        if derivative(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def fermi_eyges_bridge_variance(
    s_mm, emission_weight, scattering_power_rad2_per_mm
):
    """Return projected FE variance about the fitted mean direction.

    A free straight direction absorbs the emission-weighted mean tangent of a
    scattered trajectory.  An angular increment at coordinate ``u`` is present
    for the downstream fraction ``a(u)`` of the light, hence its mean-subtracted
    variance is ``T(u) a(u) (1-a(u)) du``.  The second return value is the same
    integral without ``T``; for constant light and scattering power it is
    exactly ``L/6``.
    """

    s = np.asarray(s_mm, dtype=np.float64).reshape(-1)
    weight = np.asarray(emission_weight, dtype=np.float64).reshape(-1)
    power = np.asarray(
        scattering_power_rad2_per_mm, dtype=np.float64
    ).reshape(-1)
    if (
        s.size < 2
        or s.shape != weight.shape
        or s.shape != power.shape
        or np.any(~np.isfinite(s))
        or np.any(~np.isfinite(weight))
        or np.any(~np.isfinite(power))
        or np.any(weight < 0.0)
        or np.any(power < 0.0)
        or np.any(np.diff(s) <= 0.0)
    ):
        raise ValueError(
            "FE bridge requires matching finite arrays on a strictly increasing grid"
        )
    ds = np.diff(s)
    emission_intervals = 0.5 * (weight[:-1] + weight[1:]) * ds
    total_emission = float(np.sum(emission_intervals))
    if total_emission <= 0.0 or not np.isfinite(total_emission):
        return 0.0, 0.0
    downstream = np.zeros_like(s)
    downstream[:-1] = np.cumsum(emission_intervals[::-1])[::-1]
    fraction_downstream = np.clip(downstream / total_emission, 0.0, 1.0)
    bridge_weight = fraction_downstream * (1.0 - fraction_downstream)
    variance = float(
        np.sum(
            0.5
            * (power[:-1] * bridge_weight[:-1]
               + power[1:] * bridge_weight[1:])
            * ds
        )
    )
    equivalent_thickness = float(
        np.sum(0.5 * (bridge_weight[:-1] + bridge_weight[1:]) * ds)
    )
    return max(variance, 0.0), max(equivalent_thickness, 0.0)


def _resolved_mcs_enabled():
    """Return the one mode-independent MCS master switch."""
    return _env_bool_switch(
        "EMITTER_ENABLE_MCS",
        DEFAULT_ENABLE_PRIMARY_MCS,
        "ENABLE_MCS",
    )


def _is_cosmic_fit_mode(fit_mode=None):
    """Return whether the current public fit mode uses the cosmic engine."""
    configured = fit_mode
    if configured is None:
        configured = os.environ.get("FIT_MODE", os.environ.get("LF_FIT_MODE", ""))
    value = str(configured or "").strip().lower().replace("-", "_")
    return value in {
        "general",
        "cosmic",
        "auto_clipped",
        "range_clipped",
        "universal",
    }


def _resolved_cosmic_mcs_continuation(*, mcs_enabled=None):
    """Return the validated source-default or per-run cosmic selector."""
    if mcs_enabled is None:
        mcs_enabled = _resolved_mcs_enabled()
    if not bool(mcs_enabled):
        return "off"
    configured = _env_str_switch(
        "EMITTER_COSMIC_MCS_CONTINUATION",
        DEFAULT_COSMIC_MCS_CONTINUATION,
    )
    value = str(configured).strip().lower().replace("-", "_")
    aliases = {
        "none": "off",
        "straight": "off",
        "straight_track": "off",
        "linear_fe": "linear_fermi_eyges",
        "fermi_eyges": "linear_fermi_eyges",
        "coherent_fe": "coherent_fermi_eyges",
        "coherent_profile": "coherent_fermi_eyges",
        "joint": "joint_k0_range_gaussian_fe",
        "joint_k0": "joint_k0_range_gaussian_fe",
        "continuous_joint": "joint_k0_range_gaussian_fe",
        "mixed": "joint_k0_range_mixed_mcs",
        "mixed_mcs": "joint_k0_range_mixed_mcs",
        "soft_hard": "joint_k0_range_mixed_mcs",
    }
    value = aliases.get(value, value)
    allowed = {
        "off",
        "linear_fermi_eyges",
        "coherent_fermi_eyges",
        "joint_k0_range_gaussian_fe",
        "joint_k0_range_mixed_mcs",
    }
    if value not in allowed:
        raise ValueError(
            "DEFAULT_COSMIC_MCS_CONTINUATION must be one of "
            + ", ".join(sorted(allowed))
            + f"; got {configured!r}"
        )
    return value


def _resolved_cosmic_joint_inference_method(cosmic_continuation=None):
    raw_override = os.environ.get("EMITTER_COSMIC_JOINT_INFERENCE_METHOD")
    configured_override = (
        None
        if raw_override is None or not str(raw_override).strip()
        else raw_override
    )
    configured = (
        DEFAULT_COSMIC_JOINT_INFERENCE_METHOD
        if configured_override is None
        else configured_override
    )
    value = str(configured).strip().lower().replace(
        "-", "_"
    )
    aliases = {
        "laplace": "laplace_cubature",
        "deterministic": "laplace_cubature",
        "smc": "reference_smc",
        "annealed_smc": "reference_smc",
    }
    value = aliases.get(value, value)
    if cosmic_continuation == "joint_k0_range_mixed_mcs":
        if configured_override is not None and value != "reference_smc":
            raise ValueError(
                "joint_k0_range_mixed_mcs requires the reference_smc engine"
            )
        return "reference_smc"
    allowed = {"laplace_cubature", "reference_smc"}
    if value not in allowed:
        raise ValueError(
            "DEFAULT_COSMIC_JOINT_INFERENCE_METHOD must be one of "
            + ", ".join(sorted(allowed))
            + f"; got {configured!r}"
        )
    return value


def emitter_switch_summary_from_env(*, fit_mode=None):
    """Return resolved top-level physics switches without building an Emitter.

    Both reconstruction selectors are resolved only from this file. The
    effective mean-cone smearing flag is deliberately separate: the validated
    Fermi--Eyges process model keeps a sharp mean cone and applies MCS through
    its path/process continuation, whereas the legacy model uses local Highland
    broadening.
    """
    mcs_enabled = _resolved_mcs_enabled()
    primary_mcs_model = _env_str_switch(
        "EMITTER_PRIMARY_MCS_MODEL", DEFAULT_PRIMARY_MCS_MODEL
    ).strip().lower().replace("-", "_")
    primary_mcs_enabled = bool(
        mcs_enabled and not _is_cosmic_fit_mode(fit_mode)
    )
    cosmic_continuation = _resolved_cosmic_mcs_continuation(
        mcs_enabled=mcs_enabled
    )
    cosmic_joint_inference = _resolved_cosmic_joint_inference_method(
        cosmic_continuation
    )
    legacy_smearing = primary_mcs_model in {
        "legacy", "cone_broadening", "local_highland"
    }
    return {
        "water_phase_index": _env_float_switch(
            "EMITTER_WATER_PHASE_INDEX", DEFAULT_WATER_PHASE_INDEX
        ),
        "event_mean_contamination_model": _env_str_switch(
            "EMITTER_EVENT_MEAN_CONTAMINATION_MODEL",
            DEFAULT_EVENT_MEAN_CONTAMINATION_MODEL,
        ).strip().lower().replace("-", "_"),
        "event_mean_contamination_max_fraction": _env_float_switch(
            "EMITTER_EVENT_MEAN_CONTAMINATION_MAX_FRACTION",
            DEFAULT_EVENT_MEAN_CONTAMINATION_MAX_FRACTION,
        ),
        "enable_delta_e": _env_bool_switch("EMITTER_ENABLE_DELTA_E", DEFAULT_ENABLE_DELTA_E, "ENABLE_DELTA_E"),
        "enable_mcs": bool(mcs_enabled),
        "enable_primary_mcs": primary_mcs_enabled,
        "enable_primary_mcs_smearing": primary_mcs_enabled and legacy_smearing,
        "primary_mcs_model": primary_mcs_model,
        "cosmic_mcs_continuation": cosmic_continuation,
        "cosmic_joint_inference_method": cosmic_joint_inference,
        "enable_cosmic_linear_fermi_eyges": (
            cosmic_continuation == "linear_fermi_eyges"
        ),
        "enable_cosmic_coherent_fermi_eyges": cosmic_continuation in {
            "coherent_fermi_eyges",
        },
        "enable_cosmic_joint_k0_range_gaussian_fe": cosmic_continuation in {
            "joint_k0_range_gaussian_fe",
        },
        "enable_cosmic_joint_k0_range_mixed_mcs": cosmic_continuation in {
            "joint_k0_range_mixed_mcs",
        },
        "enable_rayleigh_scatter": _env_bool_switch("EMITTER_ENABLE_RAYLEIGH", DEFAULT_ENABLE_RAYLEIGH, "ENABLE_RAYLEIGH"),
        "enable_blacksheet_reflection": _env_bool_switch("EMITTER_ENABLE_REFLECTION", DEFAULT_ENABLE_BLACKSHEET_REFLECTION, "ENABLE_REFLECTION"),
        "reflection_in_charge": _env_bool_switch("EMITTER_REFLECTION_IN_CHARGE", DEFAULT_REFLECTION_IN_CHARGE),
        "reflection_charge_policy": _env_str_switch(
            "EMITTER_REFLECTION_CHARGE_POLICY",
            DEFAULT_REFLECTION_CHARGE_POLICY,
        ).strip().lower().replace("-", "_"),
        "reflection_bsrff": _env_float_switch("EMITTER_REFLECTION_BSRFF", DEFAULT_REFLECTION_BSRFF),
        "use_first_arrival_timing": _env_bool_switch("EMITTER_USE_FIRST_ARRIVAL_TIMING", DEFAULT_USE_FIRST_ARRIVAL_TIMING),
        "reflection_first_arrival_nodes": _env_int_switch("EMITTER_REFLECTION_FIRST_ARRIVAL_NODES", DEFAULT_REFLECTION_FIRST_ARRIVAL_NODES),
        "delta_e_distance_pmt_radius_mm": _env_float_switch("EMITTER_DELTA_E_DISTANCE_PMT_RADIUS_MM", DEFAULT_DELTA_E_DISTANCE_PMT_RADIUS_MM, "DELTA_E_DISTANCE_PMT_RADIUS_MM"),
        "primary_endpoint_model": _env_str_switch("EMITTER_PRIMARY_ENDPOINT_MODEL", DEFAULT_PRIMARY_ENDPOINT_MODEL, "PRIMARY_ENDPOINT_MODEL").lower(),
        "primary_endpoint_aperture_radius_mm": _env_float_switch("EMITTER_PRIMARY_ENDPOINT_APERTURE_RADIUS_MM", DEFAULT_PRIMARY_ENDPOINT_APERTURE_RADIUS_MM),
        "primary_endpoint_scope": _env_str_switch("EMITTER_PRIMARY_ENDPOINT_SCOPE", DEFAULT_PRIMARY_ENDPOINT_SCOPE).lower(),
        "enable_wcte_cds_occlusion": _env_bool_switch(
            "EMITTER_ENABLE_WCTE_CDS_OCCLUSION", DEFAULT_ENABLE_WCTE_CDS_OCCLUSION
        ),
        "wcte_cds_inner_radius_mm": _env_float_switch(
            "EMITTER_WCTE_CDS_INNER_RADIUS_MM", DEFAULT_WCTE_CDS_INNER_RADIUS_MM
        ),
        "wcte_cds_outer_radius_mm": _env_float_switch(
            "EMITTER_WCTE_CDS_OUTER_RADIUS_MM", DEFAULT_WCTE_CDS_OUTER_RADIUS_MM
        ),
        "wcte_cds_y_min_mm": _env_float_switch(
            "EMITTER_WCTE_CDS_Y_MIN_MM", DEFAULT_WCTE_CDS_Y_MIN_MM
        ),
        "wcte_cds_y_max_mm": _env_float_switch(
            "EMITTER_WCTE_CDS_Y_MAX_MM", DEFAULT_WCTE_CDS_Y_MAX_MM
        ),
        "enable_wcte_cds_specular_reflection": _env_bool_switch(
            "EMITTER_ENABLE_WCTE_CDS_SPECULAR_REFLECTION",
            DEFAULT_ENABLE_WCTE_CDS_SPECULAR_REFLECTION,
        ),
        "wcte_cds_specular_reflectivity": _env_float_switch(
            "EMITTER_WCTE_CDS_SPECULAR_REFLECTIVITY",
            DEFAULT_WCTE_CDS_SPECULAR_REFLECTIVITY,
        ),
        "wcte_cds_specular_phi_bins": _env_int_switch(
            "EMITTER_WCTE_CDS_SPECULAR_PHI_BINS",
            DEFAULT_WCTE_CDS_SPECULAR_PHI_BINS,
        ),
        "wcte_cds_specular_y_bins": _env_int_switch(
            "EMITTER_WCTE_CDS_SPECULAR_Y_BINS",
            DEFAULT_WCTE_CDS_SPECULAR_Y_BINS,
        ),
        "wcte_cds_specular_timing_bins": _env_int_switch(
            "EMITTER_WCTE_CDS_SPECULAR_TIMING_BINS",
            DEFAULT_WCTE_CDS_SPECULAR_TIMING_BINS,
        ),
        "rayleigh_cache_mode": _env_str_switch("EMITTER_RAYLEIGH_CACHE_MODE", DEFAULT_RAYLEIGH_CACHE_MODE).lower(),
        "photon_scatter_model": _env_str_switch(
            "EMITTER_PHOTON_SCATTER_MODEL", DEFAULT_PHOTON_SCATTER_MODEL
        ).lower(),
        "photon_scatter_enable_raman": _env_bool_switch(
            "EMITTER_PHOTON_SCATTER_ENABLE_RAMAN", DEFAULT_PHOTON_SCATTER_ENABLE_RAMAN
        ),
        "photon_scatter_parallel_pmt_loop": _env_bool_switch(
            "EMITTER_PHOTON_SCATTER_PARALLEL_PMT_LOOP",
            DEFAULT_PHOTON_SCATTER_PARALLEL_PMT_LOOP,
        ),
        "photon_scatter_threads": _env_int_switch(
            "EMITTER_PHOTON_SCATTER_THREADS", 1
        ),
        "photon_scatter_boundary_model": _env_str_switch(
            "EMITTER_PHOTON_SCATTER_BOUNDARY_MODEL",
            DEFAULT_PHOTON_SCATTER_BOUNDARY_MODEL,
        ).lower(),
    }

try:
    from .particle_cherenkov_model import (
        find_scale_for_pmts,
        get_cerenkov_angle_table,
        get_energy_distance_tables,
        theta_c_func,
        get_rel_mpmt_eff_tables,
        set_active_particle,
        canonical_particle_name,
        particle_mass_mev,
        particle_subthreshold_range_mm,
        cherenkov_threshold_kinetic_mev,
    )
    from .n_model_wrapper import *
    from .photon_scattering_transport import (
        PhotonScatteringTransportConfig,
        WCTEScatteringGeometry,
        build_photon_scatter_nodes,
        accumulate_photon_scatter_prediction,
        direct_zero_interaction_survival,
        direct_surviving_group_index,
        direct_survival_and_group_index,
        receiver_moment_table_status,
    )
    from .optical_obstacles import (
        annular_cylinder_aperture_visibility_numba,
        build_inner_cylinder_surface,
        trace_specular_inner_cylinder_to_pmt_disks_numba,
        accumulate_sparse_patch_receivers_numba,
    )
except ImportError:
    # Support the historical usage pattern where LicketyFit/ itself is placed
    # directly on sys.path and modules are imported as top-level files.
    from particle_cherenkov_model import (
        find_scale_for_pmts,
        get_cerenkov_angle_table,
        get_energy_distance_tables,
        theta_c_func,
        get_rel_mpmt_eff_tables,
        set_active_particle,
        canonical_particle_name,
        particle_mass_mev,
        particle_subthreshold_range_mm,
        cherenkov_threshold_kinetic_mev,
    )
    from n_model_wrapper import *
    from photon_scattering_transport import (
        PhotonScatteringTransportConfig,
        WCTEScatteringGeometry,
        build_photon_scatter_nodes,
        accumulate_photon_scatter_prediction,
        direct_zero_interaction_survival,
        direct_surviving_group_index,
        direct_survival_and_group_index,
        receiver_moment_table_status,
    )
    from optical_obstacles import (
        annular_cylinder_aperture_visibility_numba,
        build_inner_cylinder_surface,
        trace_specular_inner_cylinder_to_pmt_disks_numba,
        accumulate_sparse_patch_receivers_numba,
    )


_TABLE_CACHE = {}

def _get_tables(particle=None):
    """Load and cache lookup tables once per Python process and particle."""
    pname = canonical_particle_name(particle)
    cached = _TABLE_CACHE.get(pname)
    if cached is None:
        c_ang, energy_for_angle = get_cerenkov_angle_table(pname)
        overall_distances, energy_rows, distance_rows = get_energy_distance_tables(pname)
        tri_exsitu, tri_insitu, wut_insitu, wut_exsitu = get_rel_mpmt_eff_tables()
        cached = (
            c_ang, energy_for_angle, overall_distances, energy_rows, distance_rows,
            tri_exsitu, tri_insitu, wut_insitu, wut_exsitu,
        )
        _TABLE_CACHE[pname] = cached
    return cached


# -----------------------------------------------------------------------------
# Hot-loop helper caches
# -----------------------------------------------------------------------------
# These are intentionally module-level.  In a Minuit fit the Emitter may be
# constructed many times, but the detector geometry tables and mPMT response
# tables are fixed.  Caching here avoids repeated string handling, table stacking,
# and scalar normalization work in every FCN call.

_MPMT_TYPE_TO_CODE = {
    "tri_exsitu": 0,
    "tri_insitu": 1,
    "wut_exsitu": 2,
    "wut_insitu": 3,
}

# Relative mPMT efficiency mode.
#   "type": historical 4-curve per-mPMT-type model (tri/wut x in/ex-situ).
#   "slot": per-slot data/MC ratio curves, one row per mPMT slot.
# Default is "type" so all existing behavior (and WCSim, which passes no mPMT
# types) is bit-identical unless a caller explicitly opts into slot mode via
# set_rel_eff_mode("slot", slot_stack=...).
_REL_EFF_MODE = "type"
_REL_EFF_STACK_CACHE = None
_REL_EFF_SLOT_STACK_CACHE = None
_PRIMARY_NGEO_NORM_CACHE = {}
_PARTICLE_STOPPING_POWER_CACHE = {}
_PMT_RADIUS_CACHE = {}
_RANGE_FROM_ENERGY_CACHE = {}


def _initial_energies_from_table_rows(energy_rows):
    """Return K0 without expanding compact lazy trajectory rows."""
    direct = getattr(energy_rows, "initial_energies_mev", None)
    if direct is not None:
        return np.asarray(direct, dtype=np.float64)
    return np.asarray([float(row[0]) for row in energy_rows], dtype=np.float64)


def _get_range_from_energy_arrays(particle):
    """Return cached, ascending (initial_energies, overall_distances) arrays.

    Depends only on the particle's fixed range table, so it is computed once
    per particle.  Used by the absorption-mode FCN hot path.
    """
    pname = canonical_particle_name(particle)
    cached = _RANGE_FROM_ENERGY_CACHE.get(pname)
    if cached is None:
        tables = _get_tables(pname)
        overall_distances = np.asarray(tables[2], dtype=np.float64)
        energy_rows = tables[3]
        initial_energies = _initial_energies_from_table_rows(energy_rows)
        order = np.argsort(initial_energies)
        initial_energies = np.ascontiguousarray(initial_energies[order])
        overall_distances = np.ascontiguousarray(overall_distances[order])
        cached = (initial_energies, overall_distances)
        _RANGE_FROM_ENERGY_CACHE[pname] = cached
    return cached


def _get_type_rel_eff_stack():
    """
    Return relative mPMT efficiency curves in the code order

        0: tri_exsitu
        1: tri_insitu
        2: wut_exsitu
        3: wut_insitu

    The raw table order returned by get_rel_mpmt_eff_tables() is
    tri_exsitu, tri_insitu, wut_insitu, wut_exsitu, so wut entries are swapped
    here to match the string labels used throughout the Emitter.
    """
    global _REL_EFF_STACK_CACHE
    if _REL_EFF_STACK_CACHE is None:
        tables = _get_tables()
        tri_exsitu = np.asarray(tables[5], dtype=np.float64)
        tri_insitu = np.asarray(tables[6], dtype=np.float64)
        wut_insitu = np.asarray(tables[7], dtype=np.float64)
        wut_exsitu = np.asarray(tables[8], dtype=np.float64)
        _REL_EFF_STACK_CACHE = np.ascontiguousarray(
            np.vstack([tri_exsitu, tri_insitu, wut_exsitu, wut_insitu])
        )
    return _REL_EFF_STACK_CACHE


def _get_rel_eff_stack():
    """
    Return the active relative mPMT efficiency stack for the current mode.

    type mode -> 4 per-mPMT-type rows; slot mode -> per-slot rows.  In both
    cases the per-PMT integer codes passed to the kernels index rows of this
    stack, so the hot kernels are unchanged: only the table contents and the
    meaning of the codes differ between modes.
    """
    if _REL_EFF_MODE == "slot":
        if _REL_EFF_SLOT_STACK_CACHE is None:
            raise RuntimeError(
                "rel-eff mode is 'slot' but no slot stack has been set; "
                "call set_rel_eff_mode('slot', slot_stack=...) first"
            )
        return _REL_EFF_SLOT_STACK_CACHE
    return _get_type_rel_eff_stack()


def set_rel_eff_mode(mode, slot_stack=None):
    """Select the relative mPMT efficiency model used by the hot kernels.

    mode="type" (default): historical 4-curve per-mPMT-type model.
    mode="slot": per-slot data/MC ratio curves.  slot_stack must be a
        (n_slots, n_grid) float64 array, one relative-efficiency curve per mPMT
        slot, sampled on the uniform cos-incidence grid linspace(0, 1, n_grid).
        The per-PMT "codes" handed to the kernels are then slot numbers.

    Slot codes are carried in int16 by the callers.  This supports up to
    32,768 slots (0..32767), well beyond foreseeable IWCD layouts.
    """
    global _REL_EFF_MODE, _REL_EFF_SLOT_STACK_CACHE
    mode = str(mode).strip().lower()
    if mode not in ("type", "slot"):
        raise ValueError("rel-eff mode must be 'type' or 'slot' (got %r)" % (mode,))

    if mode == "slot":
        if slot_stack is None:
            raise ValueError("slot mode requires slot_stack=(n_slots, n_grid) array")
        slot_stack = np.ascontiguousarray(np.asarray(slot_stack, dtype=np.float64))
        if slot_stack.ndim != 2 or slot_stack.shape[0] < 1 or slot_stack.shape[1] < 2:
            raise ValueError("slot_stack must have shape (n_slots>=1, n_grid>=2)")
        if slot_stack.shape[0] > np.iinfo(np.int16).max + 1:
            raise ValueError(
                "slot mode supports at most 32768 slots because codes are int16 "
                "(got %d rows)" % (slot_stack.shape[0],)
            )
        _REL_EFF_SLOT_STACK_CACHE = slot_stack

    _REL_EFF_MODE = mode
    return _REL_EFF_MODE


def reset_rel_eff_mode():
    """Reset to the historical per-mPMT-type relative-efficiency model."""
    global _REL_EFF_MODE, _REL_EFF_SLOT_STACK_CACHE
    _REL_EFF_MODE = "type"
    _REL_EFF_SLOT_STACK_CACHE = None
    return _REL_EFF_MODE


def get_rel_eff_mode():
    """Return the active relative mPMT efficiency mode ('type' or 'slot')."""
    return _REL_EFF_MODE


def _encode_mpmt_types(mpmt_types):
    """
    Convert mPMT type strings to small integer codes once.

    Unknown/empty types get code -1 and are treated as fill_empty in the
    interpolation helper.  This replaces repeated string masks in the fit loop.
    """
    arr = np.asarray(mpmt_types)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int16, copy=False)

    codes = np.full(arr.shape, -1, dtype=np.int16)
    for typ, code in _MPMT_TYPE_TO_CODE.items():
        codes[arr == typ] = code
    return codes


def _interp_rel_mpmt_eff_from_codes(cost, mpmt_type_codes, fill_empty=1.0):
    """
    Fast relative mPMT efficiency interpolation on the fixed uniform cost grid.

    This is equivalent to np.interp(cost, linspace(0,1,N), yvals,
    left=yvals[0], right=yvals[-1]) for each mPMT type, but avoids building
    four boolean string masks and avoids np.tile() for the secondary-electron
    source grid.
    """
    cost = np.asarray(cost, dtype=np.float64)
    codes = np.asarray(mpmt_type_codes)

    # Broadcast PMT codes over a source x PMT cost grid without allocating a
    # tiled string array.  For 1D cost this is a no-op.
    if cost.ndim == 2 and codes.ndim == 1:
        codes = np.broadcast_to(codes[None, :], cost.shape)
    else:
        codes = np.broadcast_to(codes, cost.shape)

    table = _get_rel_eff_stack()
    n_rows = table.shape[0]
    n_grid = table.shape[1]

    out = np.full(cost.shape, fill_empty, dtype=np.float64)
    valid = np.isfinite(cost) & (codes >= 0) & (codes < n_rows)
    if not np.any(valid):
        return out

    # np.interp with x-grid linspace(0,1,N) is just linear interpolation in
    # fractional index space.  Clipping reproduces left/right edge behavior.
    x = np.clip(cost[valid], 0.0, 1.0) * (n_grid - 1)
    i0 = np.floor(x).astype(np.int64)
    i0 = np.clip(i0, 0, n_grid - 2)
    t = x - i0

    c = codes[valid].astype(np.int64, copy=False)
    y0 = table[c, i0]
    y1 = table[c, i0 + 1]
    out[valid] = y0 + t * (y1 - y0)
    return out


# -----------------------------------------------------------------------------
# Numba-compiled hot-path helpers
# -----------------------------------------------------------------------------
# These helpers are deliberately standalone rather than methods so that Numba can
# compile the source x PMT loops.  They keep the same algebra as the vectorized
# Python implementation, but avoid allocating large intermediate matrices such as
# dx, dy, dz, r, cost, optical_corr, forward_kernel, and delta_contrib.

@njit(cache=True)
def _power_law_scalar_numba(x):
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    xn = x ** 3.0777000000000001
    return (0.1209 + (1.6396999999999999 - 0.1209) * (xn / (xn + 0.79428866592713121))) / 1.002379253316015

@njit(cache=True)
def _power_law_lut_scalar_numba(x, lut):
    """Linear-interpolated power_law via a precomputed LUT on x in [0, 1].

    Replaces the per-(PMT,source) pow(x, 3.0777) in the delta accumulator with a
    clamped table lookup.  Accuracy vs the exact form (4096-pt LUT): max rel err
    ~1.3e-7 -- far below the model's numerical noise.  Caller passes lut.size==0
    to fall back to the exact scalar.
    """
    n = lut.shape[0]
    if n < 2:
        return _power_law_scalar_numba(x)
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    f = x * (n - 1)
    i0 = int(math.floor(f))
    if i0 > n - 2:
        i0 = n - 2
    t = f - i0
    return lut[i0] + t * (lut[i0 + 1] - lut[i0])


def _build_power_law_lut(n=4096):
    """Build the angular-response LUT once (module-level cache)."""
    x = np.linspace(0.0, 1.0, n)
    xn = np.power(x, 3.0777000000000001)
    y = (0.1209 + (1.6396999999999999 - 0.1209) * (xn / (xn + 0.79428866592713121))) / 1.002379253316015
    return np.ascontiguousarray(y, dtype=np.float64)


# Module-level LUT, built once.  Empty array disables the LUT path.
_POWER_LAW_LUT = _build_power_law_lut(4096)
_POWER_LAW_LUT_EMPTY = np.empty(0, dtype=np.float64)

# Rayleigh finite-disk solid-angle constants (a=60mm, R0=1000mm) hoisted out of
# the per-ray inner loop -- these depend only on fixed constants.
_RL_A2 = 3600.0
_RL_INV_RF = 557.0551063637078



# -----------------------------------------------------------------------------
# Minuit policy hooks (read by batch drivers; not used inside the Emitter FCN).
#
# The updated WCSim driver reads these values after importing this module, unless
# explicit environment variables were supplied.  This lets a tested Emitter
# version carry its preferred optimizer policy while preserving command-line
# overrideability.
#
# Environment override examples:
#   MINUIT_OPTIMIZER_POLICY=reference     # old simplex -> migrad behavior
#   MINUIT_OPTIMIZER_POLICY=fast_migrad   # strategy=0, tol=10, no simplex unless bad
#   MINUIT_FAST_TOL=10
#   MINUIT_FAST_STRATEGY=0
#
# These optimizer defaults were measured in the July 2026 MCS speed scan;
# the present optical-process defaults are defined separately at the top of
# this file.  The strategy-0/tol10 fast Migrad policy gave ~3.6x speedup on the
# local z,cx,cy,L diagnostic with negligible FCN/parameter change; use the
# driver env vars above to switch back to the old behavior for cross-checks.
# -----------------------------------------------------------------------------
def _lf_env_optional_float(name, default=None):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    key = str(raw).strip().lower()
    if key in {"none", "null", "nan", "free", "false"}:
        return None
    return float(raw)


def _lf_env_optional_int(name, default=None):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    key = str(raw).strip().lower()
    if key in {"none", "null", "nan", "free", "false"}:
        return None
    return int(raw)


MINUIT_OPTIMIZER_POLICY = os.environ.get(
    "EMITTER_MINUIT_OPTIMIZER_POLICY",
    os.environ.get("MINUIT_OPTIMIZER_POLICY", "fast_migrad"),
).strip().lower()
MINUIT_TOL = _lf_env_optional_float("EMITTER_MINUIT_TOL", _lf_env_optional_float("MINUIT_FAST_TOL", 10.0))
MINUIT_STRATEGY = _lf_env_optional_int("EMITTER_MINUIT_STRATEGY", _lf_env_optional_int("MINUIT_FAST_STRATEGY", 0))

# -----------------------------------------------------------------------------
# SPEED NOTES (July 2026 audit; production numbers from the user's own logs):
# * Compiled FCN ~2.7 ms/eval without Rayleigh; Rayleigh added ~12 ms/eval of
#   which the dominant cost was a scalar pow (x**3.0777) per ray-PMT pair.
#   All three Rayleigh kernels now use the same 4096-entry power-law LUT as
#   the delta path (max rel. change 1.3e-7; scatter-field totals agree to
#   4.5e-9).  UNMEASURED compiled speedup -- the sandbox cannot compile numba;
#   expect roughly 3-5x on the Rayleigh term, to be confirmed by one timing
#   run (enable_rayleigh_scatter True vs False).
# * First call per process pays ~49 s of numba JIT even with cache=True if the
#   on-disk cache is cold (any edit to this file invalidates it) or if
#   NUMBA_CACHE_DIR is not persistent/writable on the batch node.  The largest
#   job-level win available: set a persistent shared NUMBA_CACHE_DIR, or warm
#   the model with ONE throwaway FCN call in the parent process and FORK
#   workers afterwards so children inherit the compiled code.
# * Remaining per-call lever (NOT implemented): the delta accumulator's outer
#   loop is over sources with mu[i] accumulation inside, so PMT-parallelizing
#   it requires inverting the loop nest (reorders FP summation; physically
#   negligible but not bit-identical).  It is the dominant compiled cost of
#   the default config and the next candidate if more speed is needed.
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
class TimingPrediction(np.ndarray):
    """1D expected-time array carrying source-resolved first-arrival nodes.

    Existing callers can treat this exactly like a normal NumPy time array.  The
    updated PMT likelihood detects the attached node amplitudes/times and evaluates
    the conditional first-photoelectron likelihood without any driver change.
    """
    def __new__(
        cls, nominal, *, node_mu=None, node_t=None, active_indices=None,
        node_weight=None, weight_output_efficiency=None,
        deferred_base_mu=None, deferred_base_t=None,
        reflection_u=None, reflection_tbase=None,
        reflection_transfer_active=None, reflection_time_offset_active=None,
        reflection_patch_min_time_offset=None, reflection_patch_max_time_offset=None,
        reflection_n_bins=None,
        node_pe_scale=None,
    ):
        obj = np.asarray(nominal, dtype=np.float64).view(cls)
        obj.first_arrival_node_mu = node_mu
        obj.first_arrival_node_t = node_t
        obj.first_arrival_node_weight = node_weight
        obj.first_arrival_weight_output_efficiency = weight_output_efficiency
        obj.first_arrival_active_indices = active_indices
        obj.first_arrival_deferred_base_mu = deferred_base_mu
        obj.first_arrival_deferred_base_t = deferred_base_t
        obj.first_arrival_reflection_u = reflection_u
        obj.first_arrival_reflection_tbase = reflection_tbase
        obj.first_arrival_reflection_transfer_active = reflection_transfer_active
        obj.first_arrival_reflection_time_offset_active = reflection_time_offset_active
        obj.first_arrival_reflection_patch_min_time_offset = reflection_patch_min_time_offset
        obj.first_arrival_reflection_patch_max_time_offset = reflection_patch_max_time_offset
        obj.first_arrival_reflection_n_bins = reflection_n_bins
        obj.first_arrival_node_pe_scale = node_pe_scale
        _has_deferred = (
            deferred_base_mu is not None and deferred_base_t is not None
            and reflection_u is not None and reflection_tbase is not None
            and reflection_transfer_active is not None
            and reflection_time_offset_active is not None
        )
        obj.first_arrival_model = (
            (node_t is not None and (node_mu is not None or node_weight is not None))
            or _has_deferred
        )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.first_arrival_node_mu = getattr(obj, "first_arrival_node_mu", None)
        self.first_arrival_node_t = getattr(obj, "first_arrival_node_t", None)
        self.first_arrival_node_weight = getattr(obj, "first_arrival_node_weight", None)
        self.first_arrival_weight_output_efficiency = getattr(
            obj, "first_arrival_weight_output_efficiency", None
        )
        self.first_arrival_active_indices = getattr(obj, "first_arrival_active_indices", None)
        self.first_arrival_deferred_base_mu = getattr(obj, "first_arrival_deferred_base_mu", None)
        self.first_arrival_deferred_base_t = getattr(obj, "first_arrival_deferred_base_t", None)
        self.first_arrival_reflection_u = getattr(obj, "first_arrival_reflection_u", None)
        self.first_arrival_reflection_tbase = getattr(obj, "first_arrival_reflection_tbase", None)
        self.first_arrival_reflection_transfer_active = getattr(obj, "first_arrival_reflection_transfer_active", None)
        self.first_arrival_reflection_time_offset_active = getattr(obj, "first_arrival_reflection_time_offset_active", None)
        self.first_arrival_reflection_patch_min_time_offset = getattr(obj, "first_arrival_reflection_patch_min_time_offset", None)
        self.first_arrival_reflection_patch_max_time_offset = getattr(obj, "first_arrival_reflection_patch_max_time_offset", None)
        self.first_arrival_reflection_n_bins = getattr(obj, "first_arrival_reflection_n_bins", None)
        self.first_arrival_node_pe_scale = getattr(obj, "first_arrival_node_pe_scale", None)
        self.first_arrival_model = getattr(obj, "first_arrival_model", False)


def shift_timing_prediction(prediction, delta_t_ns):
    """Return an arrival-time prediction shifted by one global time offset.

    Track start time enters every direct, delta, molecular-scatter and reflected
    arrival node additively.  Re-running the optical model while profiling only
    ``t0`` is therefore unnecessary.  This helper shifts all time-bearing fields
    of :class:`TimingPrediction` while sharing immutable amplitudes, weights,
    transfer matrices and PMT indices.

    A normal array is supported as a backwards-compatible fallback.  ``inf``
    sentinel entries remain ``inf`` under the shift.
    """
    delta = float(delta_t_ns)
    if not math.isfinite(delta):
        raise ValueError(f"delta_t_ns must be finite, got {delta_t_ns!r}")
    if delta == 0.0:
        return prediction

    def shifted(value, *, dtype=None):
        """Shift finite arrival times while preserving NaN/inf sentinels.

        Some timing predictions use ``inf`` for absent nodes and ``nan`` for
        PMTs with no physical mean time.  A masked finite-only addition avoids
        spurious NumPy ``invalid value encountered in add`` warnings while
        retaining those sentinels exactly.
        """
        if value is None:
            return None
        source = np.asarray(value, dtype=dtype)
        out = np.array(source, copy=True)
        finite = np.isfinite(source)
        if np.any(finite):
            out[finite] = source[finite] + delta
        return out

    nominal = shifted(prediction, dtype=np.float64)
    if not isinstance(prediction, TimingPrediction):
        return nominal

    return TimingPrediction(
        nominal,
        node_mu=getattr(prediction, "first_arrival_node_mu", None),
        node_t=shifted(getattr(prediction, "first_arrival_node_t", None)),
        active_indices=getattr(prediction, "first_arrival_active_indices", None),
        node_weight=getattr(prediction, "first_arrival_node_weight", None),
        weight_output_efficiency=getattr(
            prediction, "first_arrival_weight_output_efficiency", None
        ),
        deferred_base_mu=getattr(
            prediction, "first_arrival_deferred_base_mu", None
        ),
        deferred_base_t=shifted(getattr(
            prediction, "first_arrival_deferred_base_t", None
        )),
        reflection_u=getattr(prediction, "first_arrival_reflection_u", None),
        reflection_tbase=shifted(getattr(
            prediction, "first_arrival_reflection_tbase", None
        )),
        reflection_transfer_active=getattr(
            prediction, "first_arrival_reflection_transfer_active", None
        ),
        reflection_time_offset_active=getattr(
            prediction, "first_arrival_reflection_time_offset_active", None
        ),
        reflection_patch_min_time_offset=getattr(
            prediction, "first_arrival_reflection_patch_min_time_offset", None
        ),
        reflection_patch_max_time_offset=getattr(
            prediction, "first_arrival_reflection_patch_max_time_offset", None
        ),
        reflection_n_bins=getattr(
            prediction, "first_arrival_reflection_n_bins", None
        ),
        node_pe_scale=getattr(prediction, "first_arrival_node_pe_scale", None),
    )


# Fast analytic WCTE blacksheet reflection
# -----------------------------------------------------------------------------
# Physics model:
#   * exact 16-sided WCTE inner blacksheet prism and polygonal endcaps;
#   * physical mPMT vessel openings removed from the reflecting surface;
#   * primary-muon Cherenkov illumination from the same cone-collapse and
#     analytic N_geo machinery as the direct component;
#   * one groundfrontpainted/Lambertian reflection;
#   * 45 mm receiving aperture and the existing PMT angular response;
#   * wavelength-integrated reflectivity, QE and water absorption;
#   * spectrally weighted photon group velocity.
#
# Speed strategy (no event-truth information):
#   1. Build a fine deterministic surface quadrature once per WCD geometry.
#   2. Remove mPMT openings on that fine surface.
#   3. Compress the fine surface into 192 area-preserving macro-patches.  The
#      PMT transfer of all fine patches is integrated into each macro-patch, so
#      openings and receiving geometry are not approximated by a bare centroid.
#   4. The wavelength integral F(D) is represented by its first-order physical
#      exponential about D0=3 m.  Across 0-9 m this differs from the full
#      wavelength integral by <= about 1.3%; this is not a fitted correction.
#   5. Build the current track-dependent source field in one compiled pass.
#   6. Preserve reflected PE mass in 24 global arrival-time quadrature bins.
#      Exact 192-node timing is retained as an opt-in validation mode.
#
# The constants below are derived from the uploaded WCSim WCTE material/QE
# configuration, not from any event-level light distribution:
#   L_eff = [<1/L_abs>]^{-1} at a 3 m reflected path,
#   n_g,eff = detected-spectrum weighted group index at 3 m,
#   C_spec = reflected detected spectrum relative to direct light normalized at
#            the existing 1 m N_geo convention, for BSRFF=2.5.
_WCTE_REFLECTION_N_SIDES = 16
_WCTE_REFLECTION_APOTHEM_MM = 3075.926 / 2.0
_WCTE_REFLECTION_HEIGHT_MM = 2714.235
_WCTE_REFLECTION_Y_CENTER_MM = 424.763
_WCTE_REFLECTION_DOME_OUTER_RADIUS_MM = 347.0
_WCTE_REFLECTION_DOME_CUT_MM = 235.0
_WCTE_REFLECTION_DOME_CYL_HEIGHT_MM = 2.0 * 77.785
_WCTE_REFLECTION_LEFF_MM = 88996.09802858661
_WCTE_REFLECTION_GROUP_INDEX = 1.384730463081079
_WCTE_REFLECTION_SPECTRAL_C_BSRFF2P5 = 0.12560474934849003
_WCTE_REFLECTION_REFERENCE_BSRFF = 2.5
_WCTE_REFLECTION_FINE_TANGENT = 10
_WCTE_REFLECTION_FINE_Y = 36
_WCTE_REFLECTION_FINE_CAP_SUBDIV = 10
_WCTE_REFLECTION_TRANSFER_CACHE_SCHEMA = 1
_WCTE_FAST_REFLECTION_CACHE = {}
_WCTE_CDS_SPECULAR_SURFACE_CACHE = {}
_PARTICLE_TOF_ANTIDERIVATIVE_CACHE = {}
_WCTE_TIMING_NODE_ORDER_CACHE = {}
_WCTE_REFLECTION_IMPLEMENTATION_SHA256 = None


def _get_particle_tof_antiderivative(emitter):
    """Return master range and cumulative primary flight time arrays.

    Let R(K) be the table range remaining above Cherenkov threshold.  Define

        A(R) = integral_0^R dR' / [beta(K(R')) c].

    A depends only on the particle range table.  Therefore the flight time from
    the track start to source coordinate s is exactly

        t_mu(s) = A(R0) - A(max(R0-s, 0)),

    where R0 is the current range to threshold.  This removes repeated
    Gauss-Legendre integrations from every likelihood call while retaining the
    continuous table-based beta(K) model.
    """
    pname = canonical_particle_name(getattr(emitter, "particle_name", "muon"))
    mass = float(getattr(emitter, "particle_mass", particle_mass_mev(pname)))
    key = (pname, mass)
    cached = _PARTICLE_TOF_ANTIDERIVATIVE_CACHE.get(key)
    if cached is not None:
        return cached

    tables = _get_tables(pname)
    rr = np.asarray(tables[2], dtype=np.float64)
    energy_rows = tables[3]
    kk = _initial_energies_from_table_rows(energy_rows)
    order = np.argsort(rr)
    rr = rr[order]
    kk = kk[order]
    keep = np.isfinite(rr) & np.isfinite(kk)
    rr = rr[keep]
    kk = kk[keep]
    rr, unique_idx = np.unique(rr, return_index=True)
    kk = kk[unique_idx]

    # The first generated table row lies slightly above threshold.  Add the
    # physical R=0 endpoint using the analytic threshold kinetic energy.
    if rr.size == 0 or rr[0] > 0.0:
        kth = cherenkov_threshold_kinetic_mev(mass, n=float(getattr(emitter, "n", 1.344)))
        rr = np.concatenate(([0.0], rr))
        kk = np.concatenate(([kth], kk))

    gamma = 1.0 + kk / mass
    beta = np.sqrt(np.clip(1.0 - 1.0 / np.maximum(gamma * gamma, 1.0), 1.0e-15, 1.0))
    inv_v = 1.0 / (beta * 299.792458)  # ns/mm
    dR = np.diff(rr)
    dA = 0.5 * (inv_v[1:] + inv_v[:-1]) * dR
    aa = np.empty_like(rr)
    aa[0] = 0.0
    aa[1:] = np.cumsum(dA)
    cached = (np.ascontiguousarray(rr), np.ascontiguousarray(aa))
    _PARTICLE_TOF_ANTIDERIVATIVE_CACHE[key] = cached
    return cached

# Fixed 8-point Gauss-Legendre rule mapped from [-1,1] to [0,1].
_WCTE_GL8_U = np.array([
    0.0198550717512319, 0.1016667612931866, 0.2372337950418355,
    0.4082826787521751, 0.5917173212478249, 0.7627662049581645,
    0.8983332387068134, 0.9801449282487681,
], dtype=np.float64)
_WCTE_GL8_W = np.array([
    0.0506142681451881, 0.1111905172266872, 0.1568533229389436,
    0.1813418916891810, 0.1813418916891810, 0.1568533229389436,
    0.1111905172266872, 0.0506142681451881,
], dtype=np.float64)


def _wcte_reflection_power_law_vector(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    xn = x ** 3.0777000000000001
    return (
        0.1209
        + (1.6396999999999999 - 0.1209)
        * (xn / (xn + 0.79428866592713121))
    ) / 1.002379253316015


def _wcte_regular_polygon_vertices():
    n = _WCTE_REFLECTION_N_SIDES
    rc = _WCTE_REFLECTION_APOTHEM_MM / math.cos(math.pi / n)
    ang = (np.arange(n, dtype=np.float64) + 0.5) * 2.0 * math.pi / n
    return np.column_stack([rc * np.sin(ang), rc * np.cos(ang)])


def _wcte_triangle_centroids(b, c, n):
    pts = []
    for i in range(n):
        for j in range(n - i):
            pts.append(((i + 1.0/3.0) * b + (j + 1.0/3.0) * c) / n)
            if i + j < n - 1:
                pts.append(((i + 2.0/3.0) * b + (j + 2.0/3.0) * c) / n)
    pts = np.asarray(pts, dtype=np.float64)
    area = 0.5 * abs(b[0]*c[1] - b[1]*c[0]) / (n*n)
    return pts, np.full(len(pts), area, dtype=np.float64)


def _wcte_build_fine_blacksheet_surface(wcd):
    """Return fine surface xyz, outward normal, area and surface code.

    Surface codes: 0 barrel, 1 bottom, 2 top.  mPMT openings are removed using
    the physical dome envelope before any speed compression is performed.
    """
    xyz = []
    normal = []
    area = []
    code = []
    nside = _WCTE_REFLECTION_N_SIDES
    ap = _WCTE_REFLECTION_APOTHEM_MM
    height = _WCTE_REFLECTION_HEIGHT_MM
    yc = _WCTE_REFLECTION_Y_CENTER_MM
    nt = _WCTE_REFLECTION_FINE_TANGENT
    ny = _WCTE_REFLECTION_FINE_Y
    dphi = 2.0 * math.pi / nside
    half_width = ap * math.tan(math.pi / nside)
    dy = height / ny
    dt = 2.0 * half_width / nt
    ybottom = yc - 0.5 * height

    for jf in range(nside):
        phi = jf * dphi
        nvec = np.array([math.sin(phi), 0.0, math.cos(phi)], dtype=np.float64)
        tvec = np.array([math.cos(phi), 0.0, -math.sin(phi)], dtype=np.float64)
        centre = ap * nvec
        for iy in range(ny):
            y = ybottom + (iy + 0.5) * dy
            for it in range(nt):
                q = -half_width + (it + 0.5) * dt
                p = centre + q * tvec
                xyz.append((p[0], y, p[2]))
                normal.append(nvec)
                area.append(dt * dy)
                code.append(0)

    verts = _wcte_regular_polygon_vertices()
    nc = _WCTE_REFLECTION_FINE_CAP_SUBDIV
    for sign, scode in ((-1.0, 1), (1.0, 2)):
        y = yc + sign * 0.5 * height
        nvec = np.array([0.0, sign, 0.0], dtype=np.float64)
        for j in range(nside):
            pts2, aa = _wcte_triangle_centroids(verts[j], verts[(j + 1) % nside], nc)
            for p2, ar in zip(pts2, aa):
                xyz.append((p2[0], y, p2[1]))
                normal.append(nvec)
                area.append(ar)
                code.append(scode)

    xyz = np.asarray(xyz, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    code = np.asarray(code, dtype=np.int8)

    dome_centres = []
    dome_axes = []
    for mpmt in getattr(wcd, "mpmts", []):
        if mpmt is None:
            continue
        try:
            pl = mpmt.get_placement("design", wcd)
            axis = np.asarray(pl["direction_z"], dtype=np.float64)
            axis /= max(float(np.linalg.norm(axis)), 1e-30)
            loc = np.asarray(pl["location"], dtype=np.float64)
            centre = loc + (
                _WCTE_REFLECTION_DOME_CYL_HEIGHT_MM
                - _WCTE_REFLECTION_DOME_CUT_MM
            ) * axis
            dome_centres.append(centre)
            dome_axes.append(axis)
        except Exception:
            continue

    if dome_centres:
        dc = np.asarray(dome_centres, dtype=np.float64)
        da = np.asarray(dome_axes, dtype=np.float64)
        d = xyz[:, None, :] - dc[None, :, :]
        within_sphere = np.sum(d*d, axis=2) < _WCTE_REFLECTION_DOME_OUTER_RADIUS_MM**2
        within_cap = np.einsum("pmi,mi->pm", d, da) >= _WCTE_REFLECTION_DOME_CUT_MM
        hole = np.any(within_sphere & within_cap, axis=1)
    else:
        hole = np.zeros(len(xyz), dtype=bool)

    return (
        np.ascontiguousarray(xyz[~hole]),
        np.ascontiguousarray(normal[~hole]),
        np.ascontiguousarray(area[~hole]),
        np.ascontiguousarray(code[~hole]),
        int(np.sum(hole)),
    )


def _wcte_group_surface(xyz, normal, area, code, ntan, ny, nrad):
    """Area-preserving deterministic macro-patch grouping."""
    nside = _WCTE_REFLECTION_N_SIDES
    ap = _WCTE_REFLECTION_APOTHEM_MM
    height = _WCTE_REFLECTION_HEIGHT_MM
    yc = _WCTE_REFLECTION_Y_CENTER_MM
    y0 = yc - 0.5 * height
    half_width = ap * math.tan(math.pi / nside)
    dphi = 2.0 * math.pi / nside
    keys = []
    for x, n, scode in zip(xyz, normal, code):
        if scode == 0:
            phi = math.atan2(n[0], n[2])
            jf = int(round(phi / dphi)) % nside
            tvec = np.array([math.cos(jf*dphi), 0.0, -math.sin(jf*dphi)])
            q = float(np.dot(x, tvec))
            it = min(ntan - 1, max(0, int((q + half_width) / (2.0*half_width) * ntan)))
            iy = min(ny - 1, max(0, int((x[1] - y0) / height * ny)))
            key = (0, jf, it, iy)
        else:
            ph = math.atan2(x[0], x[2]) % (2.0*math.pi)
            jf = int(ph / dphi) % nside
            face_phi = round(ph / dphi) * dphi
            r_boundary = ap / max(math.cos(ph - face_phi), 1e-9)
            frac = math.hypot(x[0], x[2]) / r_boundary
            ir = min(nrad - 1, max(0, int(frac * nrad)))
            key = (int(scode), jf, ir)
        keys.append(key)

    unique = sorted(set(keys))
    mapping = {k: i for i, k in enumerate(unique)}
    gid = np.asarray([mapping[k] for k in keys], dtype=np.int32)
    ng = len(unique)
    gx = np.zeros((ng, 3), dtype=np.float64)
    gn = np.zeros((ng, 3), dtype=np.float64)
    ga = np.zeros(ng, dtype=np.float64)
    gc = np.zeros(ng, dtype=np.int8)
    for g, key in enumerate(unique):
        idx = np.flatnonzero(gid == g)
        aa = area[idx]
        atot = float(np.sum(aa))
        ga[g] = atot
        gx[g] = np.sum(xyz[idx] * aa[:, None], axis=0) / atot
        nn = np.sum(normal[idx] * aa[:, None], axis=0)
        gn[g] = nn / max(float(np.linalg.norm(nn)), 1e-30)
        gc[g] = code[idx[0]]
    return gx, gn, ga, gc, gid


@njit(cache=True)
def _wcte_compress_sorted_reflection_nodes_equal_mass(node_mu, node_t, n_out):
    """Compress ordered reflection nodes into equal-PE CDF quadrature bins.

    Each PMT column is split into n_out consecutive intervals containing equal
    reflected expectation.  A bin retains its exact PE mass and PE-weighted
    arrival time.  This is a deterministic quadrature compression of the
    analytic reflected arrival-time CDF; no WCSim event distribution or fitted
    correction enters the construction.
    """
    n_in, n_col = node_mu.shape
    if n_out <= 0 or n_out >= n_in:
        return node_mu.copy(), node_t.copy()
    out_mu = np.zeros((n_out, n_col), dtype=np.float32)
    out_t = np.empty((n_out, n_col), dtype=np.float32)
    for g in range(n_out):
        for i in range(n_col):
            out_t[g, i] = np.inf

    for i in range(n_col):
        total = 0.0
        for j in range(n_in):
            m = float(node_mu[j, i])
            if math.isfinite(m) and m > 0.0:
                total += m
        if total <= 0.0:
            continue
        target = total / n_out
        g = 0
        filled = 0.0
        tsum = 0.0
        for j in range(n_in):
            remaining_m = float(node_mu[j, i])
            tt = float(node_t[j, i])
            if remaining_m <= 0.0 or (not math.isfinite(remaining_m)) or (not math.isfinite(tt)):
                continue
            while remaining_m > 0.0 and g < n_out:
                if g == n_out - 1:
                    take = remaining_m
                else:
                    room = target - filled
                    if room <= 1.0e-14 * max(target, 1.0):
                        if filled > 0.0:
                            out_mu[g, i] = filled
                            out_t[g, i] = tsum / filled
                        g += 1
                        filled = 0.0
                        tsum = 0.0
                        continue
                    take = remaining_m if remaining_m < room else room
                filled += take
                tsum += take * tt
                remaining_m -= take
                if g < n_out - 1 and filled >= target * (1.0 - 1.0e-12):
                    out_mu[g, i] = filled
                    out_t[g, i] = tsum / filled
                    g += 1
                    filled = 0.0
                    tsum = 0.0
        if g < n_out and filled > 0.0:
            out_mu[g, i] = filled
            out_t[g, i] = tsum / filled
    return out_mu, out_t


@njit(cache=True, fastmath=True)
def _wcte_interp_scalar(x, grid_x, grid_y):
    n = grid_x.size
    if n == 0:
        return 0.0
    if x <= grid_x[0]:
        return grid_y[0]
    if x >= grid_x[n - 1]:
        return grid_y[n - 1]
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if grid_x[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    x0 = grid_x[lo - 1]
    x1 = grid_x[lo]
    y0 = grid_y[lo - 1]
    y1 = grid_y[lo]
    if x1 <= x0:
        return y0
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


@njit(cache=True, fastmath=True)
def _wcte_reflection_source_field_numba(
    surface_xyz, surface_normal, surface_area,
    start, direction, length, scale, sroot, energy,
    intensity, ngeo_norm, ngeo_radius, n_water, particle_mass,
    dedx_E, dedx_val, tof_range, tof_A, range_to_threshold,
    starting_time, group_index_over_c, absorption_length,
):
    """Build reflected source amplitude and base time in one compiled pass.

    This is the same analytic incident-flux expression used by the vectorized
    implementation: cone-root position, analytic N_geo, projected blacksheet
    incidence, Beer--Lambert loss on the source-to-surface leg, integrated
    particle flight time, and photon group propagation.  The compilation only
    removes temporary arrays and Python dispatch; it does not alter the model.
    """
    nsrc = surface_xyz.shape[0]
    u = np.zeros(nsrc, dtype=np.float32)
    tbase = np.zeros(nsrc, dtype=np.float32)
    tx = direction[0]; ty = direction[1]; tz = direction[2]
    a0 = _wcte_interp_scalar(range_to_threshold, tof_range, tof_A)
    for i in range(nsrc):
        ss = sroot[i]
        if ss < 0.0:
            ss = 0.0
        elif ss > length:
            ss = length
        ex = start[0] + ss * tx
        ey = start[1] + ss * ty
        ez = start[2] + ss * tz
        dx = surface_xyz[i, 0] - ex
        dy = surface_xyz[i, 1] - ey
        dz = surface_xyz[i, 2] - ez
        r2 = dx*dx + dy*dy + dz*dz
        r = math.sqrt(max(r2, 1.0e-30))
        invr = 1.0 / r
        cos_inc = (
            dx * surface_normal[i, 0]
            + dy * surface_normal[i, 1]
            + dz * surface_normal[i, 2]
        ) * invr
        inc = 0.0
        E = energy[i]
        amp = scale[i]
        if cos_inc > 0.0 and amp > 0.0 and E > 0.0:
            gamma = 1.0 + E / particle_mass
            beta2 = 1.0 - 1.0 / max(gamma*gamma, 1.0e-30)
            if beta2 > 0.0:
                beta = math.sqrt(beta2)
                if n_water * beta > 1.0:
                    cos_tc = 1.0 / (n_water * beta)
                    sin2 = max(1.0 - cos_tc*cos_tc, 0.0)
                    dedx = _wcte_interp_scalar(E, dedx_E, dedx_val)
                    dc_ds = dedx / (
                        n_water * particle_mass * beta**3 * gamma**3
                    )
                    reff = math.sqrt(r*r + ngeo_radius*ngeo_radius)
                    denom = reff * sin2 + reff*reff * dc_ds
                    if denom > 0.0 and math.isfinite(denom):
                        inc = (
                            intensity * ngeo_norm * (1.0 / denom) * amp
                            * cos_inc * surface_area[i]
                            * math.exp(-r / absorption_length)
                        )
        if inc > 0.0 and math.isfinite(inc):
            u[i] = inc
        rem = range_to_threshold - ss
        if rem < 0.0:
            rem = 0.0
        arem = _wcte_interp_scalar(rem, tof_range, tof_A)
        tof = a0 - arem
        if tof < 0.0:
            tof = 0.0
        tbase[i] = starting_time + tof + r * group_index_over_c
    return u, tbase


@njit(cache=True, fastmath=True)
def _wcte_fast_reflection_accumulate(u, tbase, transfer, transfer_r2, ng_over_c):
    npatch, npmts = transfer.shape
    mu = np.zeros(npmts, dtype=np.float64)
    tnum = np.zeros(npmts, dtype=np.float64)
    for i in range(npmts):
        sm = 0.0
        st = 0.0
        for p in range(npatch):
            up = float(u[p])
            m = up * float(transfer[p, i])
            sm += m
            st += float(tbase[p]) * m + ng_over_c * up * float(transfer_r2[p, i])
        mu[i] = sm
        tnum[i] = st
    return mu, tnum


@njit(cache=True, fastmath=True)
def _wcte_fast_reflection_accumulate_selected(
    u, tbase, transfer, transfer_r2, ng_over_c, active_indices
):
    npatch = transfer.shape[0]
    nsel = active_indices.size
    mu = np.zeros(nsel, dtype=np.float64)
    tnum = np.zeros(nsel, dtype=np.float64)
    for j in range(nsel):
        i = active_indices[j]
        sm = 0.0
        st = 0.0
        for p in range(npatch):
            up = float(u[p])
            m = up * float(transfer[p, i])
            sm += m
            st += float(tbase[p]) * m + ng_over_c * up * float(transfer_r2[p, i])
        mu[j] = sm
        tnum[j] = st
    return mu, tnum


@njit(cache=True, fastmath=True)
def _wcte_fast_reflection_region_nodes_selected(
    u, tbase, transfer, transfer_r2, ng_over_c,
    active_indices, region_id, n_regions,
):
    """Accumulate macro-patches into fixed physical timing regions.

    The default 80 regions are 16 prism faces x 3 vertical bands on the
    barrel, plus 16 azimuthal wedges on each endcap. Fine opening masks and PMT
    receiving geometry remain integrated in the static transfer matrix. This
    retains a parameter-dependent analytic arrival field while avoiding a
    192-node likelihood evaluation on every FCN call.
    """
    npatch = transfer.shape[0]
    nsel = active_indices.size
    node_mu = np.zeros((n_regions, nsel), dtype=np.float64)
    node_tnum = np.zeros((n_regions, nsel), dtype=np.float64)
    mu = np.zeros(nsel, dtype=np.float64)
    tnum = np.zeros(nsel, dtype=np.float64)
    for j in range(nsel):
        i = active_indices[j]
        sm = 0.0
        st = 0.0
        for p in range(npatch):
            up = float(u[p])
            m = up * float(transfer[p, i])
            if m <= 0.0:
                continue
            g = int(region_id[p])
            term_t = float(tbase[p]) * m + ng_over_c * up * float(transfer_r2[p, i])
            node_mu[g, j] += m
            node_tnum[g, j] += term_t
            sm += m
            st += term_t
        mu[j] = sm
        tnum[j] = st
    node_t = np.empty((n_regions, nsel), dtype=np.float64)
    for g in range(n_regions):
        for j in range(nsel):
            mm = node_mu[g, j]
            if mm > 0.0:
                node_t[g, j] = node_tnum[g, j] / mm
            else:
                node_t[g, j] = np.inf
    return mu, tnum, node_mu, node_t


@njit(cache=True)
def _wcte_fast_reflection_time_bins_selected(
    u, tbase, transfer, transfer_r2, ng_over_c, active_indices, n_bins,
):
    """Accumulate reflected light into arrival-time bins for selected PMTs.

    For each PMT, the exact analytic macro-patch arrival range is divided into
    n_bins with quadratic edge spacing,

        t_k = t_min + (t_max-t_min) (k/n_bins)^2.

    The denser early-time spacing is motivated by the first-photoelectron order
    statistic: the likelihood is most sensitive to the leading edge.  Each bin
    preserves its exact reflected PE mass and PE-weighted mean arrival time.
    This is deterministic quadrature of the analytic field; no WCSim event
    distribution is used.
    """
    npatch = transfer.shape[0]
    nsel = active_indices.size
    node_mu = np.zeros((n_bins, nsel), dtype=np.float64)
    node_tnum = np.zeros((n_bins, nsel), dtype=np.float64)
    mu = np.zeros(nsel, dtype=np.float64)
    tnum = np.zeros(nsel, dtype=np.float64)

    for j in range(nsel):
        i = active_indices[j]
        tmin = 1.0e300
        tmax = -1.0e300
        sm = 0.0
        st = 0.0
        # First pass: exact total and arrival range.
        for p in range(npatch):
            up = float(u[p])
            tr = float(transfer[p, i])
            m = up * tr
            if m <= 0.0:
                continue
            mean_r2 = float(transfer_r2[p, i]) / tr if tr > 0.0 else 0.0
            tt = float(tbase[p]) + ng_over_c * mean_r2
            if tt < tmin:
                tmin = tt
            if tt > tmax:
                tmax = tt
            sm += m
            st += m * tt
        mu[j] = sm
        tnum[j] = st
        if sm <= 0.0 or tmax < tmin:
            continue
        span = tmax - tmin
        if span < 1.0e-12:
            span = 1.0e-12

        # Second pass: quadratic-time quadrature bins.  Inverting the quadratic
        # edge map gives bin = floor(n_bins*sqrt(x)).
        for p in range(npatch):
            up = float(u[p])
            tr = float(transfer[p, i])
            m = up * tr
            if m <= 0.0:
                continue
            mean_r2 = float(transfer_r2[p, i]) / tr if tr > 0.0 else 0.0
            tt = float(tbase[p]) + ng_over_c * mean_r2
            x = (tt - tmin) / span
            if x < 0.0:
                x = 0.0
            elif x > 1.0:
                x = 1.0
            b = int(math.sqrt(x) * n_bins)
            if b >= n_bins:
                b = n_bins - 1
            node_mu[b, j] += m
            node_tnum[b, j] += m * tt

    node_t = np.empty((n_bins, nsel), dtype=np.float64)
    for b in range(n_bins):
        for j in range(nsel):
            mm = node_mu[b, j]
            if mm > 0.0:
                node_t[b, j] = node_tnum[b, j] / mm
            else:
                node_t[b, j] = np.inf
    return mu, tnum, node_mu, node_t


@njit(cache=True, fastmath=True)
def _wcte_fast_reflection_global_time_bins_selected(
    u, tbase, transfer, transfer_r2, ng_over_c, active_indices, n_bins,
    patch_min_mean_r2, patch_max_mean_r2,
):
    """One-pass reflected-light accumulation in global arrival-time bins.

    The static detector-to-PMT transfer stores, for each blacksheet macro-patch,
    the minimum and maximum receiving-leg distance over all physically visible
    PMTs.  Combining those exact geometry bounds with the current source time
    gives a conservative current-hypothesis arrival interval.  All reflected
    patch-to-PMT contributions are then accumulated once into uniform bins over
    that interval.  Bin PE mass and PE-weighted mean time are preserved exactly.

    This is numerical quadrature of the analytic field, not a fit to WCSim.
    The global interval makes the operation one pass over patch-PMT pairs rather
    than the two per-PMT passes required by adaptive local bins.
    """
    npatch = transfer.shape[0]
    nsel = active_indices.size
    node_mu = np.zeros((n_bins, nsel), dtype=np.float32)
    node_tnum = np.zeros((n_bins, nsel), dtype=np.float64)
    mu = np.zeros(nsel, dtype=np.float64)
    tnum = np.zeros(nsel, dtype=np.float64)

    tmin = 1.0e300
    tmax = -1.0e300
    for p in range(npatch):
        if float(u[p]) <= 0.0:
            continue
        lo = float(tbase[p]) + ng_over_c * float(patch_min_mean_r2[p])
        hi = float(tbase[p]) + ng_over_c * float(patch_max_mean_r2[p])
        if lo < tmin:
            tmin = lo
        if hi > tmax:
            tmax = hi
    if tmax < tmin:
        node_t = np.empty((n_bins, nsel), dtype=np.float32)
        for b in range(n_bins):
            for j in range(nsel):
                node_t[b, j] = np.inf
        return mu, tnum, node_mu, node_t
    span = tmax - tmin
    if span < 1.0e-12:
        span = 1.0e-12
    inv_span_bins = float(n_bins) / span

    for j in range(nsel):
        i = active_indices[j]
        sm = 0.0
        st = 0.0
        for p in range(npatch):
            up = float(u[p])
            tr = float(transfer[p, i])
            m = up * tr
            if m <= 0.0:
                continue
            mean_r2 = float(transfer_r2[p, i]) / tr
            tt = float(tbase[p]) + ng_over_c * mean_r2
            b = int((tt - tmin) * inv_span_bins)
            if b < 0:
                b = 0
            elif b >= n_bins:
                b = n_bins - 1
            node_mu[b, j] += m
            node_tnum[b, j] += m * tt
            sm += m
            st += m * tt
        mu[j] = sm
        tnum[j] = st

    node_t = np.empty((n_bins, nsel), dtype=np.float32)
    for b in range(n_bins):
        for j in range(nsel):
            mm = float(node_mu[b, j])
            if mm > 0.0:
                node_t[b, j] = node_tnum[b, j] / mm
            else:
                node_t[b, j] = np.inf
    return mu, tnum, node_mu, node_t


@njit(cache=True, fastmath=True)
def _wcte_fast_reflection_global_time_bins_selected_pmt_major(
    u, tbase, transfer_pmt, time_offset_pmt, active_indices, n_bins,
    patch_min_time_offset, patch_max_time_offset,
):
    npatch = transfer_pmt.shape[1]
    nsel = active_indices.size
    node_mu = np.zeros((n_bins, nsel), dtype=np.float32)
    node_tnum = np.zeros((n_bins, nsel), dtype=np.float64)
    mu = np.zeros(nsel, dtype=np.float64)
    tnum = np.zeros(nsel, dtype=np.float64)
    tmin = 1.0e300; tmax = -1.0e300
    for pp in range(npatch):
        if float(u[pp]) <= 0.0: continue
        lo = float(tbase[pp]) + float(patch_min_time_offset[pp])
        hi = float(tbase[pp]) + float(patch_max_time_offset[pp])
        if lo < tmin: tmin = lo
        if hi > tmax: tmax = hi
    if tmax < tmin:
        node_t = np.empty((n_bins,nsel),np.float32)
        for b in range(n_bins):
            for j in range(nsel): node_t[b,j]=np.inf
        return mu,tnum,node_mu,node_t
    span=max(tmax-tmin,1e-12); inv_span_bins=float(n_bins)/span
    for j in range(nsel):
        i=active_indices[j]; sm=0.0; st=0.0
        for pp in range(npatch):
            m=float(u[pp])*float(transfer_pmt[i,pp])
            if m<=0.0: continue
            tt=float(tbase[pp])+float(time_offset_pmt[i,pp])
            b=int((tt-tmin)*inv_span_bins)
            if b<0:b=0
            elif b>=n_bins:b=n_bins-1
            node_mu[b,j]+=m; node_tnum[b,j]+=m*tt; sm+=m; st+=m*tt
        mu[j]=sm;tnum[j]=st
    node_t=np.empty((n_bins,nsel),np.float32)
    for b in range(n_bins):
        for j in range(nsel):
            mm=float(node_mu[b,j]);node_t[b,j]=node_tnum[b,j]/mm if mm>0 else np.inf
    return mu,tnum,node_mu,node_t



@njit(cache=True, fastmath=True)
def _wcte_fast_reflection_global_time_bins_compact(
    u, tbase, transfer_active, time_offset_active, n_bins,
    patch_min_time_offset, patch_max_time_offset,
):
    """PMT-compact equivalent of the analytic reflection-bin accumulator."""
    nsel = transfer_active.shape[0]
    npatch = transfer_active.shape[1]
    node_mu = np.zeros((n_bins, nsel), dtype=np.float32)
    node_tnum = np.zeros((n_bins, nsel), dtype=np.float64)
    mu = np.zeros(nsel, dtype=np.float64)
    tnum = np.zeros(nsel, dtype=np.float64)
    tmin = 1.0e300; tmax = -1.0e300
    for p in range(npatch):
        if float(u[p]) <= 0.0:
            continue
        lo = float(tbase[p]) + float(patch_min_time_offset[p])
        hi = float(tbase[p]) + float(patch_max_time_offset[p])
        if lo < tmin: tmin = lo
        if hi > tmax: tmax = hi
    if tmax < tmin:
        node_t = np.empty((n_bins, nsel), dtype=np.float32)
        for b in range(n_bins):
            for j in range(nsel): node_t[b,j] = np.inf
        return mu, tnum, node_mu, node_t
    span = tmax-tmin
    if span < 1.0e-12: span = 1.0e-12
    inv_span_bins = float(n_bins)/span
    for j in range(nsel):
        sm=0.0; st=0.0
        for p in range(npatch):
            m = float(u[p]) * float(transfer_active[j,p])
            if m <= 0.0: continue
            tt = float(tbase[p]) + float(time_offset_active[j,p])
            b = int((tt-tmin)*inv_span_bins)
            if b < 0: b=0
            elif b >= n_bins: b=n_bins-1
            node_mu[b,j] += m
            node_tnum[b,j] += m*tt
            sm += m; st += m*tt
        mu[j]=sm; tnum[j]=st
    node_t=np.empty((n_bins,nsel),dtype=np.float32)
    for b in range(n_bins):
        for j in range(nsel):
            mm=float(node_mu[b,j])
            node_t[b,j]=node_tnum[b,j]/mm if mm>0.0 else np.inf
    return mu,tnum,node_mu,node_t


def _wcte_get_compact_active_transfer(cache, active):
    active = np.ascontiguousarray(active, dtype=np.int32)
    key = active.tobytes()
    compact_cache = getattr(cache, "active_transfer_cache", None)
    if compact_cache is None:
        compact_cache = {}
        cache.active_transfer_cache = compact_cache
    out = compact_cache.get(key)
    if out is None:
        out = (
            np.ascontiguousarray(cache.transfer_pmt[active], dtype=np.float32),
            np.ascontiguousarray(cache.time_offset_pmt[active], dtype=np.float32),
        )
        if len(compact_cache) >= 8:
            compact_cache.clear()
        compact_cache[key] = out
    return out


def _wcte_get_compact_patch_transfer(cache, active, patch_indices):
    """Cache the exact nonzero-illumination submatrix for an event support."""
    active = np.ascontiguousarray(active, dtype=np.int32)
    patch_indices = np.ascontiguousarray(patch_indices, dtype=np.int16)
    key = (active.tobytes(), patch_indices.tobytes())
    subcache = getattr(cache, "active_patch_transfer_cache", None)
    if subcache is None:
        subcache = {}
        cache.active_patch_transfer_cache = subcache
    out = subcache.get(key)
    if out is None:
        tr_active, to_active = _wcte_get_compact_active_transfer(cache, active)
        out = (
            np.ascontiguousarray(tr_active[:, patch_indices], dtype=np.float32),
            np.ascontiguousarray(to_active[:, patch_indices], dtype=np.float32),
        )
        if len(subcache) >= 16:
            subcache.clear()
        subcache[key] = out
    return out


class _WCTEFastReflectionTransfer:
    pass


_WCTE_REFLECTION_TRANSFER_ARRAY_FIELDS = (
    "surface_xyz",
    "surface_normal",
    "surface_area",
    "surface_code",
    "timing_region_id",
    "transfer",
    "transfer_r2",
    "patch_min_mean_r2",
    "patch_max_mean_r2",
    "transfer_pmt",
    "time_offset_pmt",
    "patch_min_time_offset",
    "patch_max_time_offset",
)
_WCTE_REFLECTION_TRANSFER_SCALAR_FIELDS = (
    "n_timing_regions",
    "group_index_over_c",
    "bsrff",
    "pmt_radius_mm",
    "n_fine_surface_patches",
    "n_macro_patches",
    "n_removed_opening_patches",
    "memory_bytes",
)


def _sha256_contiguous_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(x) for x in array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _wcte_design_opening_geometry(wcd):
    """Return the exact mPMT geometry consumed by the opening-mask builder."""
    centres = []
    axes = []
    for mpmt in getattr(wcd, "mpmts", []):
        if mpmt is None:
            continue
        try:
            placement = mpmt.get_placement("design", wcd)
            axis = np.asarray(placement["direction_z"], dtype=np.float64)
            axis /= max(float(np.linalg.norm(axis)), 1.0e-30)
            location = np.asarray(placement["location"], dtype=np.float64)
            centre = location + (
                _WCTE_REFLECTION_DOME_CYL_HEIGHT_MM
                - _WCTE_REFLECTION_DOME_CUT_MM
            ) * axis
            centres.append(centre)
            axes.append(axis)
        except Exception:
            continue
    if centres:
        return (
            np.ascontiguousarray(centres, dtype=np.float64),
            np.ascontiguousarray(axes, dtype=np.float64),
        )
    return (
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
    )


def _wcte_reflection_implementation_sha256():
    global _WCTE_REFLECTION_IMPLEMENTATION_SHA256
    if _WCTE_REFLECTION_IMPLEMENTATION_SHA256 is None:
        try:
            source = Path(__file__).resolve().read_bytes()
        except OSError:
            source = b""
        _WCTE_REFLECTION_IMPLEMENTATION_SHA256 = hashlib.sha256(source).hexdigest()
    return str(_WCTE_REFLECTION_IMPLEMENTATION_SHA256)


def _wcte_reflection_cpuinfo_sha256():
    try:
        payload = Path("/proc/cpuinfo").read_bytes()
    except OSError:
        payload = (platform.machine() + "\0" + platform.processor()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _wcte_reflection_persistent_metadata(
    wcd,
    p_locations,
    direction_zs,
    *,
    bsrff,
    pmt_radius_mm,
    ntan,
    ny,
    nrad,
):
    p = np.ascontiguousarray(p_locations, dtype=np.float64)
    pn = np.ascontiguousarray(direction_zs, dtype=np.float64)
    opening_centres, opening_axes = _wcte_design_opening_geometry(wcd)
    constants = {
        "reflection_n_sides": int(_WCTE_REFLECTION_N_SIDES),
        "reflection_apothem_mm": float(_WCTE_REFLECTION_APOTHEM_MM),
        "reflection_height_mm": float(_WCTE_REFLECTION_HEIGHT_MM),
        "reflection_y_center_mm": float(_WCTE_REFLECTION_Y_CENTER_MM),
        "reflection_dome_outer_radius_mm": float(_WCTE_REFLECTION_DOME_OUTER_RADIUS_MM),
        "reflection_dome_cut_mm": float(_WCTE_REFLECTION_DOME_CUT_MM),
        "reflection_dome_cyl_height_mm": float(_WCTE_REFLECTION_DOME_CYL_HEIGHT_MM),
        "reflection_leff_mm": float(_WCTE_REFLECTION_LEFF_MM),
        "reflection_group_index": float(_WCTE_REFLECTION_GROUP_INDEX),
        "reflection_spectral_c_bsrff2p5": float(_WCTE_REFLECTION_SPECTRAL_C_BSRFF2P5),
        "reflection_reference_bsrff": float(_WCTE_REFLECTION_REFERENCE_BSRFF),
        "reflection_fine_tangent": int(_WCTE_REFLECTION_FINE_TANGENT),
        "reflection_fine_y": int(_WCTE_REFLECTION_FINE_Y),
        "reflection_fine_cap_subdiv": int(_WCTE_REFLECTION_FINE_CAP_SUBDIV),
        "reflection_power_law": [
            3.0777000000000001,
            0.1209,
            1.6396999999999999,
            0.79428866592713121,
            1.002379253316015,
        ],
        "speed_of_light_mm_per_ns": 299.792458,
    }
    return {
        "schema_version": int(_WCTE_REFLECTION_TRANSFER_CACHE_SCHEMA),
        "table_kind": "licketyfit_wcte_analytic_reflection_transfer",
        "implementation_sha256": _wcte_reflection_implementation_sha256(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy_version": str(np.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpuinfo_sha256": _wcte_reflection_cpuinfo_sha256(),
        "byteorder": sys.byteorder,
        "p_locations_shape": list(p.shape),
        "p_locations_sha256": _sha256_contiguous_array(p),
        "direction_zs_shape": list(pn.shape),
        "direction_zs_sha256": _sha256_contiguous_array(pn),
        "opening_centres_shape": list(opening_centres.shape),
        "opening_centres_sha256": _sha256_contiguous_array(opening_centres),
        "opening_axes_shape": list(opening_axes.shape),
        "opening_axes_sha256": _sha256_contiguous_array(opening_axes),
        "bsrff": float(bsrff),
        "pmt_radius_mm": float(pmt_radius_mm),
        "tangent_bins": int(ntan),
        "y_bins": int(ny),
        "cap_radial_bins": int(nrad),
        "constants": constants,
    }


def _wcte_reflection_persistent_cache_dir():
    # The production driver resolves LF_RUNTIME_CACHE_DIR to a per-project,
    # per-Python runtime root before importing this module.  Keep generated
    # tables there; never write them to the release's source or tables tree.
    runtime = (
        os.environ.get("LF_RESOLVED_RUNTIME_CACHE_DIR", "").strip()
        or os.environ.get("LF_RUNTIME_CACHE_DIR", "").strip()
    )
    if not runtime:
        return None
    root = Path(runtime).expanduser() / "reflection"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return root


def _wcte_reflection_persistent_cache_path(metadata):
    root = _wcte_reflection_persistent_cache_dir()
    if root is None:
        return None
    payload = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return root / f"wcte_reflection_transfer_{digest}_v{_WCTE_REFLECTION_TRANSFER_CACHE_SCHEMA}.npz"


@contextmanager
def _wcte_reflection_cache_lock(lock_path):
    handle = None
    try:
        handle = open(lock_path, "a+b")
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if handle is not None:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()


def _wcte_reflection_array_manifest(cache):
    return {
        name: {
            "dtype": np.asarray(getattr(cache, name)).dtype.str,
            "shape": list(np.asarray(getattr(cache, name)).shape),
            "sha256": _sha256_contiguous_array(getattr(cache, name)),
        }
        for name in _WCTE_REFLECTION_TRANSFER_ARRAY_FIELDS
    }


def _wcte_load_persistent_reflection_transfer(path, expected_metadata):
    try:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
            if metadata != expected_metadata:
                return None
            manifest = json.loads(str(np.asarray(payload["array_manifest_json"]).item()))
            if set(manifest) != set(_WCTE_REFLECTION_TRANSFER_ARRAY_FIELDS):
                return None
            out = _WCTEFastReflectionTransfer()
            for name in _WCTE_REFLECTION_TRANSFER_ARRAY_FIELDS:
                array = np.ascontiguousarray(payload[name])
                expected = manifest[name]
                if (
                    array.dtype.str != str(expected["dtype"])
                    or list(array.shape) != list(expected["shape"])
                    or _sha256_contiguous_array(array) != str(expected["sha256"])
                ):
                    return None
                setattr(out, name, array)
            scalars = json.loads(str(np.asarray(payload["scalar_json"]).item()))
            if set(scalars) != set(_WCTE_REFLECTION_TRANSFER_SCALAR_FIELDS):
                return None
            for name in _WCTE_REFLECTION_TRANSFER_SCALAR_FIELDS:
                setattr(out, name, scalars[name])
            out.active_transfer_cache = {}
            out.active_patch_transfer_cache = {}
            out.persistent_cache_path = str(path)
            out.persistent_cache_hit = True
            return out
    except (
        OSError,
        EOFError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        return None


def _wcte_save_persistent_reflection_transfer(path, metadata, cache):
    manifest = _wcte_reflection_array_manifest(cache)
    scalars = {
        name: getattr(cache, name)
        for name in _WCTE_REFLECTION_TRANSFER_SCALAR_FIELDS
    }
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}.npz")
    arrays = {
        name: np.ascontiguousarray(getattr(cache, name))
        for name in _WCTE_REFLECTION_TRANSFER_ARRAY_FIELDS
    }
    try:
        np.savez(
            temporary,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            array_manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
            scalar_json=np.asarray(json.dumps(scalars, sort_keys=True)),
            **arrays,
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _wcte_build_fast_reflection_transfer(
    wcd,
    p_locations,
    direction_zs,
    *,
    bsrff,
    pmt_radius_mm,
    ntan,
    ny,
    nrad,
):
    fine_x, fine_n, fine_a, fine_code, n_removed = _wcte_build_fine_blacksheet_surface(wcd)
    gx, gn, ga, gc, gid = _wcte_group_surface(
        fine_x, fine_n, fine_a, fine_code, ntan, ny, nrad
    )
    p = np.asarray(p_locations, dtype=np.float64)
    pn = np.asarray(direction_zs, dtype=np.float64)
    ngroup = len(gx)
    npmts = len(p)
    transfer = np.zeros((ngroup, npmts), dtype=np.float64)
    transfer_r2 = np.zeros_like(transfer)
    spectral_c = (
        _WCTE_REFLECTION_SPECTRAL_C_BSRFF2P5
        * float(bsrff) / _WCTE_REFLECTION_REFERENCE_BSRFF
    )
    a_pmt = float(pmt_radius_mm)
    pmt_area = math.pi * a_pmt * a_pmt

    # Integrate the receiving transfer of every fine patch into its macro-patch.
    # This is why the coarse per-FCN source grid still preserves exact openings
    # and the fine receiving geometry.
    chunk = 128
    for lo in range(0, len(fine_x), chunk):
        hi = min(lo + chunk, len(fine_x))
        xx = fine_x[lo:hi]
        nn = fine_n[lo:hi]
        aa = fine_a[lo:hi]
        v = p[None, :, :] - xx[:, None, :]
        r2 = np.linalg.norm(v, axis=2)
        wo = v / np.maximum(r2[:, :, None], 1e-30)
        cos_out = np.einsum("fmi,fi->fm", wo, -nn)
        pmt_cost = -np.einsum("fmi,mi->fm", wo, pn)
        valid = (cos_out > 0.0) & (pmt_cost > 0.0)
        lambert = np.maximum(cos_out, 0.0) / math.pi
        omega = 2.0 * math.pi * (
            1.0 - r2 / np.sqrt(r2*r2 + a_pmt*a_pmt)
        )
        geom = (
            lambert
            * (omega / pmt_area)
            * _wcte_reflection_power_law_vector(pmt_cost)
            * spectral_c
            * np.exp(-r2 / _WCTE_REFLECTION_LEFF_MM)
        )
        geom[~valid] = 0.0
        for local, fidx in enumerate(range(lo, hi)):
            g = gid[fidx]
            weighted = aa[local] * geom[local]
            transfer[g] += weighted
            transfer_r2[g] += weighted * r2[local]

    transfer /= ga[:, None]
    transfer_r2 /= ga[:, None]

    # Exact static receiving-leg distance bounds for the one-pass global time
    # quadrature.  Only physically visible nonzero transfer entries contribute.
    mean_r2 = np.divide(
        transfer_r2, transfer, out=np.zeros_like(transfer_r2), where=(transfer > 0.0)
    )
    patch_min_mean_r2 = np.zeros(ngroup, dtype=np.float64)
    patch_max_mean_r2 = np.zeros(ngroup, dtype=np.float64)
    for ipatch in range(ngroup):
        vals = mean_r2[ipatch, transfer[ipatch] > 0.0]
        if vals.size:
            patch_min_mean_r2[ipatch] = float(np.min(vals))
            patch_max_mean_r2[ipatch] = float(np.max(vals))

    # Fixed physical timing regions, derived only from the blacksheet geometry.
    dphi_region = 2.0 * math.pi / _WCTE_REFLECTION_N_SIDES
    y0_region = _WCTE_REFLECTION_Y_CENTER_MM - 0.5 * _WCTE_REFLECTION_HEIGHT_MM
    region_keys = []
    for xx, nn, sc in zip(gx, gn, gc):
        if int(sc) == 0:
            face = int(round(math.atan2(nn[0], nn[2]) / dphi_region)) % _WCTE_REFLECTION_N_SIDES
            # Two physical vertical barrel bands plus one azimuthal bin per
            # blacksheet face.  The 20-event full-node profile test changes the
            # profiled Delta-NLL shape by at most 0.153 while preserving the
            # same sampled length minimum; three bands cost 25% more timing
            # nodes without a meaningful accuracy gain.
            iy = min(1, max(0, int((xx[1] - y0_region) / _WCTE_REFLECTION_HEIGHT_MM * 2.0)))
            key = (0, face, iy)
        else:
            ph = math.atan2(xx[0], xx[2]) % (2.0 * math.pi)
            face = int(ph / dphi_region) % _WCTE_REFLECTION_N_SIDES
            key = (int(sc), face)
        region_keys.append(key)
    region_unique = sorted(set(region_keys))
    region_lookup = {key: i for i, key in enumerate(region_unique)}
    timing_region_id = np.asarray([region_lookup[key] for key in region_keys], dtype=np.int32)

    out = _WCTEFastReflectionTransfer()
    out.surface_xyz = np.ascontiguousarray(gx, dtype=np.float64)
    out.surface_normal = np.ascontiguousarray(gn, dtype=np.float64)
    out.surface_area = np.ascontiguousarray(ga, dtype=np.float64)
    out.surface_code = np.ascontiguousarray(gc, dtype=np.int8)
    out.timing_region_id = np.ascontiguousarray(timing_region_id, dtype=np.int32)
    out.n_timing_regions = int(len(region_unique))
    out.transfer = np.ascontiguousarray(transfer.astype(np.float32))
    out.transfer_r2 = np.ascontiguousarray(transfer_r2.astype(np.float32))
    out.patch_min_mean_r2 = np.ascontiguousarray(patch_min_mean_r2.astype(np.float32))
    out.patch_max_mean_r2 = np.ascontiguousarray(patch_max_mean_r2.astype(np.float32))
    _ngoc = _WCTE_REFLECTION_GROUP_INDEX / 299.792458
    out.transfer_pmt = np.ascontiguousarray(transfer.T.astype(np.float32))
    # Single-precision static time offsets retain sub-picosecond resolution
    # over WCTE path lengths while halving the hot reflection-operator traffic.
    out.time_offset_pmt = np.ascontiguousarray((_ngoc * mean_r2.T).astype(np.float32))
    out.patch_min_time_offset = np.ascontiguousarray((_ngoc * patch_min_mean_r2).astype(np.float32))
    out.patch_max_time_offset = np.ascontiguousarray((_ngoc * patch_max_mean_r2).astype(np.float32))
    out.group_index_over_c = _ngoc
    out.bsrff = float(bsrff)
    out.pmt_radius_mm = float(pmt_radius_mm)
    out.n_fine_surface_patches = int(len(fine_x))
    out.n_macro_patches = int(len(gx))
    out.n_removed_opening_patches = int(n_removed)
    out.memory_bytes = int(out.transfer.nbytes + out.transfer_r2.nbytes + out.patch_min_mean_r2.nbytes + out.patch_max_mean_r2.nbytes)
    out.active_transfer_cache = {}
    out.active_patch_transfer_cache = {}
    out.persistent_cache_path = None
    out.persistent_cache_hit = False
    return out


def _wcte_reflection_cache_key(wcd, p_locations, bsrff, pmt_radius, ntan, ny, nrad):
    p = np.asarray(p_locations)
    first = tuple(np.round(p[0], 3)) if len(p) else (0.0, 0.0, 0.0)
    last = tuple(np.round(p[-1], 3)) if len(p) else (0.0, 0.0, 0.0)
    return (
        id(wcd), p.shape, first, last,
        float(bsrff), float(pmt_radius), int(ntan), int(ny), int(nrad),
    )


def _get_wcte_fast_reflection_transfer(emitter, wcd, p_locations, direction_zs):
    bsrff = float(getattr(emitter, "reflection_bsrff", DEFAULT_REFLECTION_BSRFF))
    pmt_radius = float(getattr(
        emitter, "reflection_pmt_aperture_radius_mm",
        DEFAULT_REFLECTION_PMT_APERTURE_RADIUS_MM,
    ))
    ntan = int(getattr(emitter, "reflection_tangent_bins", DEFAULT_REFLECTION_TANGENT_BINS))
    ny = int(getattr(emitter, "reflection_y_bins", DEFAULT_REFLECTION_Y_BINS))
    nrad = int(getattr(emitter, "reflection_cap_radial_bins", DEFAULT_REFLECTION_CAP_RADIAL_BINS))
    key = _wcte_reflection_cache_key(wcd, p_locations, bsrff, pmt_radius, ntan, ny, nrad)
    cached = _WCTE_FAST_REFLECTION_CACHE.get(key)
    if cached is None:
        resolved_ntan = max(1, ntan)
        resolved_ny = max(1, ny)
        resolved_nrad = max(1, nrad)
        metadata = _wcte_reflection_persistent_metadata(
            wcd,
            p_locations,
            direction_zs,
            bsrff=bsrff,
            pmt_radius_mm=pmt_radius,
            ntan=resolved_ntan,
            ny=resolved_ny,
            nrad=resolved_nrad,
        )
        persistent_path = _wcte_reflection_persistent_cache_path(metadata)
        if persistent_path is None:
            cached = _wcte_build_fast_reflection_transfer(
                wcd, p_locations, direction_zs,
                bsrff=bsrff,
                pmt_radius_mm=pmt_radius,
                ntan=resolved_ntan,
                ny=resolved_ny,
                nrad=resolved_nrad,
            )
        else:
            lock_path = persistent_path.with_suffix(persistent_path.suffix + ".lock")
            with _wcte_reflection_cache_lock(lock_path):
                cached = _wcte_load_persistent_reflection_transfer(
                    persistent_path, metadata
                )
                if cached is None:
                    cached = _wcte_build_fast_reflection_transfer(
                        wcd, p_locations, direction_zs,
                        bsrff=bsrff,
                        pmt_radius_mm=pmt_radius,
                        ntan=resolved_ntan,
                        ny=resolved_ny,
                        nrad=resolved_nrad,
                    )
                    try:
                        _wcte_save_persistent_reflection_transfer(
                            persistent_path, metadata, cached
                        )
                        cached.persistent_cache_path = str(persistent_path)
                    except OSError:
                        pass
        if len(_WCTE_FAST_REFLECTION_CACHE) >= 4:
            _WCTE_FAST_REFLECTION_CACHE.clear()
        _WCTE_FAST_REFLECTION_CACHE[key] = cached
    return cached


def _wcte_integrated_primary_tof_fast(emitter, s_mm):
    """Fast table-exact integrated primary flight time [ns].

    Uses the cached antiderivative A(R), so the cost is two vectorized linear
    interpolations rather than an 8-point beta integral for every source.
    """
    s = np.asarray(s_mm, dtype=np.float64)
    rr, aa = _get_particle_tof_antiderivative(emitter)
    R0 = max(float(getattr(emitter, "range_to_threshold_mm", emitter.length)), 0.0)
    coordinate_scale = float(
        getattr(emitter, "stopping_range_coordinate_scale", 1.0)
    )
    coordinate_scale = max(coordinate_scale, 1.0e-30)
    rem = np.maximum(R0 - coordinate_scale * np.maximum(s, 0.0), 0.0)
    a0 = float(np.interp(R0, rr, aa, left=aa[0], right=aa[-1]))
    arem = np.interp(rem, rr, aa, left=aa[0], right=aa[-1])
    # A is integrated in the mean-loss coordinate u.  Since u=c*s for a
    # straggled track, physical flight time is integral(ds/beta c0)=dA/c.
    out = (a0 - arem) / coordinate_scale
    return np.maximum(out, 0.0)



@njit(cache=True, fastmath=True)
def _wcte_direct_node_times_active_numba(
    active, p_locations, start, direction, s_raw, scale, sroot, length,
    tof_range, tof_A, range_to_threshold, energy_distance_scale,
    starting_time, group_index_over_c,
):
    n = active.size
    out = np.empty(n, dtype=np.float32)
    a0 = _wcte_interp_scalar(range_to_threshold, tof_range, tof_A)
    tx=direction[0];ty=direction[1];tz=direction[2]
    for j in range(n):
        i=active[j]
        ss=sroot[i] if scale[i]>0.0 else s_raw[i]
        if ss<0.0:ss=0.0
        elif ss>length:ss=length
        ex=start[0]+ss*tx;ey=start[1]+ss*ty;ez=start[2]+ss*tz
        dx=p_locations[i,0]-ex;dy=p_locations[i,1]-ey;dz=p_locations[i,2]-ez
        r=math.sqrt(dx*dx+dy*dy+dz*dz)+0.01
        rem=range_to_threshold-energy_distance_scale*ss
        if rem<0.0:rem=0.0
        tof=(a0-_wcte_interp_scalar(rem,tof_range,tof_A))/energy_distance_scale
        if tof<0.0:tof=0.0
        out[j]=starting_time+tof+r*group_index_over_c
    return out


def _prepare_wcte_fast_reflection_source(emitter, transfer_cache):
    """Evaluate only the hypothesis-dependent blacksheet illumination field."""
    start = np.asarray(emitter.start_coord, dtype=np.float64)
    direction = np.asarray(emitter.direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-30)
    length = max(float(emitter.length), 0.0)
    x = transfer_cache.surface_xyz
    scale, sroot, energy = find_scale_for_pmts(
        pmt_pos=x, start_pos=start, track_dir=direction,
        s_a_mm=0.001, s_max_mm=length, theta_c_func=theta_c_func,
        range_stop_mm=float(getattr(emitter, "range_to_threshold_mm", length)),
        energy_distance_scale=float(
            getattr(emitter, "stopping_range_coordinate_scale", 1.0)
        ),
        n_scan=150,
        near_cross_tol=float(emitter.effective_primary_soft_cone_sigma_rad(None)),
        edge_model=str(getattr(emitter, "primary_edge_model", "legacy")),
        particle=emitter.particle_name,
        particle_mass=float(emitter.particle_mass), n_water=float(emitter.n),
        subgrid_refine=bool(getattr(emitter, "use_subgrid_refine", True)),
        legacy_grid=not bool(getattr(emitter, "smooth_tables", True)),
    )
    dedx_E, dedx_val = _get_particle_stopping_power_table(emitter.particle_name)
    tof_range, tof_A = _get_particle_tof_antiderivative(emitter)
    u, tbase = _wcte_reflection_source_field_numba(
        np.ascontiguousarray(transfer_cache.surface_xyz, dtype=np.float64),
        np.ascontiguousarray(transfer_cache.surface_normal, dtype=np.float64),
        np.ascontiguousarray(transfer_cache.surface_area, dtype=np.float64),
        np.ascontiguousarray(start, dtype=np.float64),
        np.ascontiguousarray(direction, dtype=np.float64), float(length),
        np.ascontiguousarray(scale, dtype=np.float64),
        np.ascontiguousarray(sroot, dtype=np.float64),
        np.ascontiguousarray(energy, dtype=np.float64),
        float(emitter.intensity), float(emitter.primary_ngeo_normalization()),
        float(emitter.primary_ngeo_pmt_radius_mm), float(emitter.n),
        float(emitter.particle_mass),
        np.ascontiguousarray(dedx_E, dtype=np.float64),
        np.ascontiguousarray(dedx_val, dtype=np.float64),
        np.ascontiguousarray(tof_range, dtype=np.float64),
        np.ascontiguousarray(tof_A, dtype=np.float64),
        float(getattr(emitter, "range_to_threshold_mm", length)),
        float(emitter.starting_time), float(transfer_cache.group_index_over_c),
        float(_WCTE_REFLECTION_LEFF_MM),
    )
    return np.ascontiguousarray(u, dtype=np.float64), np.ascontiguousarray(tbase, dtype=np.float64)


def _wcte_cds_specular_surface_key(emitter):
    return (
        round(float(getattr(emitter, "wcte_cds_axis_x_mm", DEFAULT_WCTE_CDS_AXIS_X_MM)), 9),
        round(float(getattr(emitter, "wcte_cds_axis_z_mm", DEFAULT_WCTE_CDS_AXIS_Z_MM)), 9),
        round(float(getattr(emitter, "wcte_cds_inner_radius_mm", DEFAULT_WCTE_CDS_INNER_RADIUS_MM)), 9),
        round(float(getattr(emitter, "wcte_cds_y_min_mm", DEFAULT_WCTE_CDS_Y_MIN_MM)), 9),
        round(float(getattr(emitter, "wcte_cds_y_max_mm", DEFAULT_WCTE_CDS_Y_MAX_MM)), 9),
        max(8, int(getattr(emitter, "wcte_cds_specular_phi_bins", DEFAULT_WCTE_CDS_SPECULAR_PHI_BINS))),
        max(1, int(getattr(emitter, "wcte_cds_specular_y_bins", DEFAULT_WCTE_CDS_SPECULAR_Y_BINS))),
    )


def _get_wcte_cds_specular_surface(emitter):
    """Return cached deterministic quadrature for the CDS water-facing wall."""
    key = _wcte_cds_specular_surface_key(emitter)
    cached = _WCTE_CDS_SPECULAR_SURFACE_CACHE.get(key)
    if cached is None:
        axis_x, axis_z, radius, y_min, y_max, n_phi, n_y = key
        cached = build_inner_cylinder_surface(
            axis_x=axis_x,
            axis_z=axis_z,
            radius=radius,
            y_min=y_min,
            y_max=y_max,
            n_phi=n_phi,
            n_y=n_y,
        )
        if len(_WCTE_CDS_SPECULAR_SURFACE_CACHE) >= 8:
            _WCTE_CDS_SPECULAR_SURFACE_CACHE.clear()
        _WCTE_CDS_SPECULAR_SURFACE_CACHE[key] = cached
    return cached


def _evaluate_wcte_cds_specular_reflection(
    emitter,
    p_locations,
    direction_zs,
    *,
    active_pmt_indices=None,
    return_nodes=False,
):
    """Evaluate one polished-metal reflection from the WCTE CDS inner shaft.

    The incident field is generated by the same shrinking-cone root and analytic
    N_geo calculation as the primary and blacksheet components.  Each surface
    patch is reflected according to the exact specular law and accepted only if
    the outgoing ray intersects a finite 45-mm PMT disk without a second CDS
    encounter.  The calculation contains detector geometry and configured
    material reflectivity only; no event-truth light map or fitted scale enters.
    """
    p_locations = np.ascontiguousarray(p_locations, dtype=np.float64)
    direction_zs = np.ascontiguousarray(direction_zs, dtype=np.float64)
    npmts = int(p_locations.shape[0])
    zeros = np.zeros(npmts, dtype=np.float64)
    if npmts == 0:
        if return_nodes:
            return zeros, np.full(0, np.inf), np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
        return zeros, np.full(0, np.inf)

    surface_xyz, surface_normal, surface_area = _get_wcte_cds_specular_surface(emitter)
    start = np.asarray(emitter.start_coord, dtype=np.float64)
    direction = np.asarray(emitter.direction, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
    length = max(float(emitter.length), 0.0)
    if length <= 0.0:
        if return_nodes:
            active = np.arange(npmts, dtype=np.int64) if active_pmt_indices is None else np.asarray(active_pmt_indices, dtype=np.int64)
            return zeros, np.full(npmts, np.inf), np.zeros((0, active.size), np.float32), np.zeros((0, active.size), np.float32)
        return zeros, np.full(npmts, np.inf)

    scale, sroot, energy = find_scale_for_pmts(
        pmt_pos=surface_xyz,
        start_pos=start,
        track_dir=direction,
        s_a_mm=0.001,
        s_max_mm=length,
        theta_c_func=theta_c_func,
        range_stop_mm=float(getattr(emitter, "range_to_threshold_mm", length)),
        energy_distance_scale=float(
            getattr(emitter, "stopping_range_coordinate_scale", 1.0)
        ),
        n_scan=150,
        near_cross_tol=float(emitter.effective_primary_soft_cone_sigma_rad(None)),
        edge_model=str(getattr(emitter, "primary_edge_model", "legacy")),
        particle=emitter.particle_name,
        particle_mass=float(emitter.particle_mass),
        n_water=float(emitter.n),
        subgrid_refine=bool(getattr(emitter, "use_subgrid_refine", True)),
        legacy_grid=not bool(getattr(emitter, "smooth_tables", True)),
    )
    dedx_E, dedx_val = _get_particle_stopping_power_table(emitter.particle_name)
    tof_range, tof_A = _get_particle_tof_antiderivative(emitter)
    incident_mu, incident_tbase = _wcte_reflection_source_field_numba(
        np.ascontiguousarray(surface_xyz, dtype=np.float64),
        np.ascontiguousarray(surface_normal, dtype=np.float64),
        np.ascontiguousarray(surface_area, dtype=np.float64),
        np.ascontiguousarray(start, dtype=np.float64),
        np.ascontiguousarray(direction, dtype=np.float64),
        float(length),
        np.ascontiguousarray(scale, dtype=np.float64),
        np.ascontiguousarray(sroot, dtype=np.float64),
        np.ascontiguousarray(energy, dtype=np.float64),
        float(emitter.intensity),
        float(emitter.primary_ngeo_normalization()),
        float(emitter.primary_ngeo_pmt_radius_mm),
        float(emitter.n),
        float(emitter.particle_mass),
        np.ascontiguousarray(dedx_E, dtype=np.float64),
        np.ascontiguousarray(dedx_val, dtype=np.float64),
        np.ascontiguousarray(tof_range, dtype=np.float64),
        np.ascontiguousarray(tof_A, dtype=np.float64),
        float(getattr(emitter, "range_to_threshold_mm", length)),
        float(emitter.starting_time),
        float(_WCTE_REFLECTION_GROUP_INDEX / 299.792458),
        float(_WCTE_REFLECTION_LEFF_MM),
    )
    hit_index, hit_mu, hit_time, _ = trace_specular_inner_cylinder_to_pmt_disks_numba(
        np.ascontiguousarray(surface_xyz, dtype=np.float64),
        np.ascontiguousarray(surface_normal, dtype=np.float64),
        np.ascontiguousarray(incident_mu, dtype=np.float64),
        np.ascontiguousarray(incident_tbase, dtype=np.float64),
        np.ascontiguousarray(sroot, dtype=np.float64),
        np.ascontiguousarray(start, dtype=np.float64),
        np.ascontiguousarray(direction, dtype=np.float64),
        float(length),
        p_locations,
        direction_zs,
        float(getattr(emitter, "wcte_cds_axis_x_mm", DEFAULT_WCTE_CDS_AXIS_X_MM)),
        float(getattr(emitter, "wcte_cds_axis_z_mm", DEFAULT_WCTE_CDS_AXIS_Z_MM)),
        float(getattr(emitter, "wcte_cds_inner_radius_mm", DEFAULT_WCTE_CDS_INNER_RADIUS_MM)),
        float(getattr(emitter, "wcte_cds_outer_radius_mm", DEFAULT_WCTE_CDS_OUTER_RADIUS_MM)),
        float(getattr(emitter, "wcte_cds_y_min_mm", DEFAULT_WCTE_CDS_Y_MIN_MM)),
        float(getattr(emitter, "wcte_cds_y_max_mm", DEFAULT_WCTE_CDS_Y_MAX_MM)),
        float(getattr(emitter, "wcte_cds_pmt_aperture_radius_mm", DEFAULT_WCTE_CDS_PMT_APERTURE_RADIUS_MM)),
        float(getattr(emitter, "wcte_cds_specular_reflectivity", DEFAULT_WCTE_CDS_SPECULAR_REFLECTIVITY)),
        float(_WCTE_REFLECTION_LEFF_MM),
        float(_WCTE_REFLECTION_GROUP_INDEX / 299.792458),
    )
    nbin = max(0, int(getattr(
        emitter, "wcte_cds_specular_timing_bins",
        DEFAULT_WCTE_CDS_SPECULAR_TIMING_BINS,
    ))) if return_nodes else 0
    mu, tnum, node_mu, node_t = accumulate_sparse_patch_receivers_numba(
        np.ascontiguousarray(hit_index, dtype=np.int32),
        np.ascontiguousarray(hit_mu, dtype=np.float64),
        np.ascontiguousarray(hit_time, dtype=np.float64),
        npmts,
        nbin,
    )
    tmean = np.full(npmts, np.inf, dtype=np.float64)
    good = mu > 0.0
    tmean[good] = tnum[good] / mu[good]
    emitter._last_wcte_cds_specular_surface_patches = int(surface_xyz.shape[0])
    emitter._last_wcte_cds_specular_illuminated_patches = int(np.count_nonzero(incident_mu > 0.0))
    emitter._last_wcte_cds_specular_received_patches = int(np.count_nonzero(hit_mu > 0.0))
    if return_nodes:
        active = np.arange(npmts, dtype=np.int64) if active_pmt_indices is None else np.asarray(active_pmt_indices, dtype=np.int64)
        return (
            np.ascontiguousarray(mu, dtype=np.float64),
            np.ascontiguousarray(tmean, dtype=np.float64),
            np.ascontiguousarray(node_mu[:, active], dtype=np.float32),
            np.ascontiguousarray(node_t[:, active], dtype=np.float32),
        )
    return np.ascontiguousarray(mu, dtype=np.float64), np.ascontiguousarray(tmean, dtype=np.float64)


def _evaluate_wcte_fast_reflection(emitter, transfer_cache, active_pmt_indices=None, return_nodes=False):
    """Return reflected primary PE amplitude and mean arrival time per PMT.

    With return_nodes=True, also return the macro-patch PE/time nodes for the
    selected PMTs.  These nodes feed the exact discrete first-arrival likelihood.
    """
    u, tbase = _prepare_wcte_fast_reflection_source(emitter, transfer_cache)
    if return_nodes:
        if active_pmt_indices is None:
            active = np.arange(transfer_cache.transfer.shape[1], dtype=np.int64)
        else:
            active = np.ascontiguousarray(active_pmt_indices, dtype=np.int64)
        requested = int(getattr(
            emitter, "reflection_first_arrival_nodes",
            DEFAULT_REFLECTION_FIRST_ARRIVAL_NODES,
        ))
        if requested > 0:
            _tr_active, _to_active = _wcte_get_compact_active_transfer(
                transfer_cache, active
            )
            mu_sel, tnum_sel, node_mu, node_t = (
                _wcte_fast_reflection_global_time_bins_compact(
                    u, tbase, _tr_active, _to_active, requested,
                    transfer_cache.patch_min_time_offset,
                    transfer_cache.patch_max_time_offset,
                )
            )
        else:
            # Full 192 macro-patch node path for validation.
            mu_sel, tnum_sel = _wcte_fast_reflection_accumulate_selected(
                u, tbase, transfer_cache.transfer, transfer_cache.transfer_r2,
                float(transfer_cache.group_index_over_c), active,
            )
            tr = np.asarray(transfer_cache.transfer[:, active], dtype=np.float64)
            trr = np.asarray(transfer_cache.transfer_r2[:, active], dtype=np.float64)
            node_mu = u[:, None].astype(np.float64) * tr
            mean_r2 = np.divide(trr, tr, out=np.zeros_like(trr), where=(tr > 0.0))
            node_t = tbase[:, None].astype(np.float64) + float(transfer_cache.group_index_over_c) * mean_r2
        mu = np.zeros(transfer_cache.transfer.shape[1], dtype=np.float64)
        tmean = np.full(transfer_cache.transfer.shape[1], np.nan, dtype=np.float64)
        mu[active] = mu_sel
        tmean[active] = np.divide(
            tnum_sel, mu_sel, out=np.full_like(mu_sel, np.nan), where=(mu_sel > 0.0)
        )
        mu[~np.isfinite(mu)] = 0.0
        mu[mu < 0.0] = 0.0
        node_mu = np.where(np.isfinite(node_mu) & (node_mu > 0.0), node_mu, 0.0)
        node_t = np.where(node_mu > 0.0, node_t, np.inf)
        return (
            mu, tmean,
            np.asarray(node_mu, dtype=np.float32),
            np.asarray(node_t, dtype=np.float32),
            active,
        )

    if active_pmt_indices is None:
        mu, tnum = _wcte_fast_reflection_accumulate(
            u, tbase, transfer_cache.transfer, transfer_cache.transfer_r2,
            float(transfer_cache.group_index_over_c),
        )
        tmean = np.divide(
            tnum, mu, out=np.full_like(mu, np.nan), where=(mu > 0.0)
        )
    else:
        active = np.ascontiguousarray(active_pmt_indices, dtype=np.int64)
        mu_sel, tnum_sel = _wcte_fast_reflection_accumulate_selected(
            u, tbase, transfer_cache.transfer, transfer_cache.transfer_r2,
            float(transfer_cache.group_index_over_c), active,
        )
        mu = np.zeros(transfer_cache.transfer.shape[1], dtype=np.float64)
        tmean = np.full(transfer_cache.transfer.shape[1], np.nan, dtype=np.float64)
        mu[active] = mu_sel
        tmean[active] = np.divide(
            tnum_sel, mu_sel, out=np.full_like(mu_sel, np.nan), where=(mu_sel > 0.0)
        )
    mu[~np.isfinite(mu)] = 0.0
    mu[mu < 0.0] = 0.0
    return mu, tmean


@njit(cache=True)
def _wcte_compact_first_arrival_weights_numba(
    node_mu, node_t, observed_q, output_efficiency
):
    """Convert ordered source intensities to exact conditional first-PE weights.

    The observed charge is already available to Emitter.get_expected_pes_ts.
    Computing the order-statistic weights once here avoids repeating powers and
    cumulative-intensity algebra inside the PMT likelihood.  Zero-intensity
    source rows are removed column-by-column and the rectangular result is
    truncated to the largest actual count.  No probability threshold or
    empirical pruning is applied.
    """
    n_nodes, n_cols = node_mu.shape
    out_w = np.zeros((n_nodes, n_cols), dtype=np.float32)
    out_t = np.empty((n_nodes, n_cols), dtype=np.float32)
    for j in range(n_nodes):
        for i in range(n_cols):
            out_t[j, i] = np.inf
    max_count = 0
    for i in range(n_cols):
        total = 0.0
        for j in range(n_nodes):
            m = float(node_mu[j, i])
            if math.isfinite(m) and m > 0.0:
                total += m
        q = float(observed_q[i])
        if total <= 0.0 or (not math.isfinite(total)) or q <= 0.0:
            continue
        neff = q / output_efficiency
        if neff < 1.0e-6:
            neff = 1.0e-6
        remaining = 1.0
        k = 0
        wsum = 0.0
        for j in range(n_nodes):
            m = float(node_mu[j, i])
            if (not math.isfinite(m)) or m <= 0.0:
                continue
            p = m / total
            next_remaining = remaining - p
            if next_remaining < 0.0:
                next_remaining = 0.0
            w = remaining ** neff - next_remaining ** neff
            remaining = next_remaining
            tt = float(node_t[j, i])
            if w > 0.0 and math.isfinite(w) and math.isfinite(tt):
                out_w[k, i] = w
                out_t[k, i] = tt
                wsum += w
                k += 1
        if wsum > 0.0:
            inv = 1.0 / wsum
            for j in range(k):
                out_w[j, i] *= inv
        if k > max_count:
            max_count = k
    return out_w[:max_count], out_t[:max_count]


@njit(cache=True, fastmath=True)
def _wcte_sort_or_merge_timing_nodes_numba(base_mu, base_t, ref_mu, ref_t, observed_q, output_efficiency):
    """Exactly sort compact direct/delta nodes and merge ordered reflection bins.

    Direct plus delta contains at most about eleven nodes in the current model,
    so an in-column insertion sort is cheaper than a general 2-D argsort.
    Global reflection time bins are already ordered by construction.  A linear
    merge then produces the exact current-hypothesis node order for every PMT.
    No node is dropped and no frozen ordering approximation is used.
    """
    nb, nc = base_mu.shape
    nr = ref_mu.shape[0]
    out_mu = np.zeros((nb + nr, nc), dtype=np.float32)
    out_weight = np.zeros((nb + nr, nc), dtype=np.float32)
    out_t = np.empty((nb + nr, nc), dtype=np.float32)
    for j in range(nb + nr):
        for i in range(nc):
            out_t[j, i] = np.inf

    # Small per-column insertion sort of the base nodes, performed into local
    # rectangular arrays to keep Numba memory access contiguous.
    sbm = np.zeros((nb, nc), dtype=np.float32)
    sbt = np.empty((nb, nc), dtype=np.float32)
    for j in range(nb):
        for i in range(nc):
            sbt[j, i] = np.inf
    for i in range(nc):
        nvalid = 0
        for j in range(nb):
            m = float(base_mu[j, i])
            tt = float(base_t[j, i])
            if m <= 0.0 or (not math.isfinite(m)) or (not math.isfinite(tt)):
                continue
            k = nvalid
            while k > 0 and tt < float(sbt[k - 1, i]):
                sbt[k, i] = sbt[k - 1, i]
                sbm[k, i] = sbm[k - 1, i]
                k -= 1
            sbt[k, i] = tt
            sbm[k, i] = m
            nvalid += 1

        total = 0.0
        for k in range(nvalid):
            total += float(sbm[k, i])
        for k in range(nr):
            rm = float(ref_mu[k, i])
            rt = float(ref_t[k, i])
            if rm > 0.0 and math.isfinite(rm) and math.isfinite(rt):
                total += rm
        q = float(observed_q[i])
        neff = q / output_efficiency if output_efficiency > 0.0 else q
        if neff < 1.0e-6:
            neff = 1.0e-6
        remaining = 1.0
        remaining_power = 1.0

        ib = 0
        ir = 0
        io = 0
        while ib < nvalid or ir < nr:
            while ir < nr:
                rm = float(ref_mu[ir, i])
                rt = float(ref_t[ir, i])
                if rm > 0.0 and math.isfinite(rm) and math.isfinite(rt):
                    break
                ir += 1
            if ib >= nvalid and ir >= nr:
                break
            take_base = False
            if ir >= nr:
                take_base = True
            elif ib < nvalid and float(sbt[ib, i]) <= float(ref_t[ir, i]):
                take_base = True
            if take_base:
                m_out = float(sbm[ib, i])
                out_mu[io, i] = m_out
                out_t[io, i] = sbt[ib, i]
                ib += 1
            else:
                m_out = float(ref_mu[ir, i])
                out_mu[io, i] = m_out
                out_t[io, i] = ref_t[ir, i]
                ir += 1
            if total > 0.0:
                p_out = m_out / total
                next_remaining = remaining - p_out
                if next_remaining < 0.0:
                    next_remaining = 0.0
                next_power = next_remaining ** neff
                out_weight[io, i] = remaining_power - next_power
                remaining = next_remaining
                remaining_power = next_power
            io += 1
    return out_mu, out_weight, out_t


def _wcte_adaptive_sort_timing_nodes(node_mu, node_t, active_indices):
    """Sort source nodes exactly while reusing the previous smooth ordering.

    Source identities (direct, longitudinal delta bins, and fixed blacksheet
    regions) do not change during a fit, and their arrival-time ordering is
    highly stable under millimetre-scale parameter steps.  A cached order is
    applied first.  Every column is then checked for an inversion; only columns
    whose current times violate monotonicity are re-sorted.  Consequently the
    returned arrays are exactly ordered for the current hypothesis, while the
    usual FCN call avoids a full 2-D argsort.
    """
    mu = np.asarray(node_mu, dtype=np.float32)
    tt = np.asarray(node_t, dtype=np.float32)
    active = np.ascontiguousarray(active_indices, dtype=np.int32)
    if mu.ndim != 2 or tt.shape != mu.shape:
        raise ValueError("timing-node arrays must have matching 2-D shapes")
    if mu.shape[1] != active.size:
        raise ValueError("active PMT count does not match timing-node columns")

    # The active support is event-specific but constant throughout one fit.
    # A bytes key is deterministic and collision-free within the process.
    key = (int(mu.shape[0]), active.tobytes())
    order = _WCTE_TIMING_NODE_ORDER_CACHE.get(key)
    if order is None or order.shape != mu.shape:
        order = np.argsort(tt, axis=0).astype(np.int16, copy=False)
        if len(_WCTE_TIMING_NODE_ORDER_CACHE) >= 64:
            _WCTE_TIMING_NODE_ORDER_CACHE.clear()
        _WCTE_TIMING_NODE_ORDER_CACHE[key] = order.copy()
        return (
            np.take_along_axis(mu, order, axis=0),
            np.take_along_axis(tt, order, axis=0),
        )

    sorted_t = np.take_along_axis(tt, order, axis=0)
    # Any incorrect permutation must contain at least one adjacent inversion.
    bad = np.any(sorted_t[1:] < sorted_t[:-1], axis=0)
    if np.any(bad):
        local_order = np.argsort(tt[:, bad], axis=0).astype(np.int16, copy=False)
        order = order.copy()
        order[:, bad] = local_order
        sorted_t[:, bad] = np.take_along_axis(
            tt[:, bad], local_order, axis=0
        )
        _WCTE_TIMING_NODE_ORDER_CACHE[key] = order.copy()
    sorted_mu = np.take_along_axis(mu, order, axis=0)
    return sorted_mu, sorted_t


def clear_fast_reflection_cache():
    """Clear process-local reflection and timing-order caches."""
    _WCTE_FAST_REFLECTION_CACHE.clear()
    _WCTE_CDS_SPECULAR_SURFACE_CACHE.clear()
    _WCTE_TIMING_NODE_ORDER_CACHE.clear()


# -----------------------------------------------------------------------------
# Static detector geometry and helpers for the physical first-interaction model.
_PHOTON_SCATTER_GEOMETRY_CACHE = {}

def _get_photon_scatter_geometry(wcd):
    key = id(wcd)
    cached = _PHOTON_SCATTER_GEOMETRY_CACHE.get(key)
    if cached is None:
        cached = WCTEScatteringGeometry.from_wcd(wcd)
        if len(_PHOTON_SCATTER_GEOMETRY_CACHE) >= 4:
            _PHOTON_SCATTER_GEOMETRY_CACHE.clear()
        _PHOTON_SCATTER_GEOMETRY_CACHE[key] = cached
    return cached

def _photon_scatter_transport_config(emitter):
    return PhotonScatteringTransportConfig(
        n_track_nodes=max(1, int(getattr(emitter, "photon_scatter_n_track", DEFAULT_PHOTON_SCATTER_N_TRACK))),
        n_azimuth_nodes=max(1, int(getattr(emitter, "photon_scatter_n_azimuth", DEFAULT_PHOTON_SCATTER_N_AZIMUTH))),
        n_scatter_nodes=max(1, int(getattr(emitter, "photon_scatter_n_interaction", DEFAULT_PHOTON_SCATTER_N_INTERACTION))),
        n_wavelength_nodes=max(2, int(getattr(emitter, "photon_scatter_n_wavelength", DEFAULT_PHOTON_SCATTER_N_WAVELENGTH))),
        n_raman_shift_nodes=max(2, int(getattr(emitter, "photon_scatter_n_raman_shift", DEFAULT_PHOTON_SCATTER_N_RAMAN_SHIFT))),
        n_timing_bins=max(1, int(getattr(emitter, "photon_scatter_n_timing_bins", DEFAULT_PHOTON_SCATTER_N_TIMING_BINS))),
        spectral_mode=str(getattr(emitter, "photon_scatter_spectral_mode", DEFAULT_PHOTON_SCATTER_SPECTRAL_MODE)).strip().lower(),
        pmt_aperture_radius_mm=float(getattr(emitter, "photon_scatter_pmt_aperture_radius_mm", DEFAULT_PHOTON_SCATTER_PMT_APERTURE_RADIUS_MM)),
        pmt_facing_soft_width=float(getattr(emitter, "photon_scatter_pmt_facing_soft_width", DEFAULT_PHOTON_SCATTER_PMT_FACING_SOFT_WIDTH)),
        pmt_response_model="legacy_power",
        enable_rayleigh=True,
        enable_raman=bool(getattr(emitter, "photon_scatter_enable_raman", DEFAULT_PHOTON_SCATTER_ENABLE_RAMAN)),
        parallel_pmt_loop=bool(getattr(
            emitter, "photon_scatter_parallel_pmt_loop",
            DEFAULT_PHOTON_SCATTER_PARALLEL_PMT_LOOP,
        )),
        receiver_mode=str(getattr(
            emitter, "photon_scatter_receiver_mode",
            DEFAULT_PHOTON_SCATTER_RECEIVER_MODE,
        )).strip().lower(),
        receiver_moment_table_path=str(getattr(
            emitter, "photon_scatter_receiver_table",
            DEFAULT_PHOTON_SCATTER_RECEIVER_TABLE,
        )),
        receiver_moment_table_required=bool(getattr(
            emitter, "photon_scatter_receiver_table_required",
            DEFAULT_PHOTON_SCATTER_RECEIVER_TABLE_REQUIRED,
        )),
        native_receiver=bool(getattr(
            emitter, "photon_scatter_native_receiver",
            DEFAULT_PHOTON_SCATTER_NATIVE_RECEIVER,
        )),
        native_receiver_threads=max(1, int(getattr(
            emitter, "photon_scatter_native_threads",
            DEFAULT_PHOTON_SCATTER_NATIVE_THREADS,
        ))),
        native_receiver_required=bool(getattr(
            emitter, "photon_scatter_native_required",
            DEFAULT_PHOTON_SCATTER_NATIVE_REQUIRED,
        )),
        boundary_model=str(getattr(
            emitter, "photon_scatter_boundary_model",
            DEFAULT_PHOTON_SCATTER_BOUNDARY_MODEL,
        )).strip().lower(),
        include_mpmt_domes=bool(getattr(emitter, "photon_scatter_include_mpmt_domes", DEFAULT_PHOTON_SCATTER_INCLUDE_MPMT_DOMES)),
        enforce_receiver_dome_visibility=False,
    )


def get_photon_scatter_receiver_status(emitter, p_locations, direction_zs):
    """Return the effective sparse/exact receiver decision for one geometry.

    This is intentionally a small public diagnostic wrapper around the transport
    configuration used by :meth:`Emitter.get_expected_pes_ts`.  Production
    drivers can fail early or emit metadata when a geometry-specific sparse
    table is unavailable instead of discovering an expensive exact fallback
    only through timing measurements.
    """
    return receiver_moment_table_status(
        _photon_scatter_transport_config(emitter),
        np.asarray(p_locations, dtype=np.float64),
        np.asarray(direction_zs, dtype=np.float64),
    )

def _primary_direct_molecular_state(emitter, p_locations, start_pos, track_dir, raw_s, scale, sroot, energy, config):
    s_eff = np.where(np.asarray(scale) > 0.0, np.asarray(sroot), np.asarray(raw_s))
    s_eff = np.clip(np.asarray(s_eff, dtype=np.float64), 0.0, max(float(emitter.length), 0.0))
    source = np.asarray(start_pos, dtype=np.float64)[None, :] + s_eff[:, None] * np.asarray(track_dir, dtype=np.float64)[None, :]
    path = np.linalg.norm(np.asarray(p_locations, dtype=np.float64) - source, axis=1) + 0.01
    E = np.asarray(energy, dtype=np.float64)
    gamma = 1.0 + np.maximum(E, 0.0) / float(emitter.particle_mass)
    beta2 = np.maximum(1.0 - 1.0 / np.maximum(gamma * gamma, 1.0), 0.0)
    beta = np.sqrt(beta2)
    fallback = float(getattr(emitter, "beta", 1.0))
    beta = np.where(np.isfinite(beta) & (beta > 0.0), beta, fallback)
    survival, group_index = direct_survival_and_group_index(beta, path, config=config)
    survival = np.where(np.isfinite(survival), np.clip(survival, 0.0, 1.0), 1.0)
    group_index = np.where(np.isfinite(group_index) & (group_index > 0.0), group_index, _WCTE_REFLECTION_GROUP_INDEX)
    return (np.ascontiguousarray(survival), np.ascontiguousarray(group_index),
            np.ascontiguousarray(path), np.ascontiguousarray(s_eff))

def _evaluate_photon_scatter_transport(emitter, wcd, p_locations, direction_zs, start_pos, track_dir, timing_active_indices=None, charge_active_indices=None):
    config = _photon_scatter_transport_config(emitter)
    geometry = _get_photon_scatter_geometry(wcd)
    R0 = float(getattr(emitter, "range_to_threshold_mm", emitter.length))
    mass = float(emitter.particle_mass)
    def beta_at_s(s_mm):
        K = np.asarray(emitter.muon_energy_at_s_array(s_mm, R0), dtype=np.float64)
        gamma = 1.0 + np.maximum(K, 0.0) / mass
        return np.sqrt(np.maximum(1.0 - 1.0 / np.maximum(gamma * gamma, 1.0), 0.0))
    def particle_time_at_s_ns(s_mm):
        return _wcte_integrated_primary_tof_fast(emitter, s_mm)
    nodes = build_photon_scatter_nodes(
        start_position_mm=np.asarray(start_pos, dtype=np.float64),
        track_direction=np.asarray(track_dir, dtype=np.float64),
        visible_length_mm=max(float(emitter.length), 0.0),
        beta_at_s=beta_at_s, particle_time_at_s_ns=particle_time_at_s_ns,
        intensity=float(emitter.intensity),
        primary_ngeo_normalization=float(emitter.primary_ngeo_normalization()),
        start_time_ns=float(emitter.starting_time), config=config, geometry=geometry,
    )
    prediction = accumulate_photon_scatter_prediction(
        nodes, np.asarray(p_locations, dtype=np.float64), np.asarray(direction_zs, dtype=np.float64),
        timing_active_indices=(None if timing_active_indices is None else np.asarray(timing_active_indices, dtype=np.int32)),
        charge_active_indices=(None if charge_active_indices is None else np.asarray(charge_active_indices, dtype=np.int32)),
        config=config,
    )
    return prediction, nodes, config

def clear_photon_scatter_geometry_cache():
    _PHOTON_SCATTER_GEOMETRY_CACHE.clear()

# -----------------------------------------------------------------------------
# Single-scatter Rayleigh light model (see rayleigh_scatter_model.pdf).
# Physics content: Frank-Tamm-weighted sources on the track, rays on the
# Cherenkov cone, scatter probability T/lambda_s along each ray to the can,
# scatter vertex at the path midpoint, Rayleigh phase (3/16pi)(1+cos^2 Theta),
# the fitter's own PMT angular/solid-angle response, and a timing-cut
# acceptance computed from the extra path vs the peak-cut window.  The single
# constant is lambda_s.  Validated on noE/noScat WCSim mu- (charge-only,
# delta off): restores the truth point to ~the NLL minimum.  NOT yet
# validated with deltas on, free direction, or on the production mu+ sample.
# -----------------------------------------------------------------------------
_RAYLEIGH_CAN_CACHE = {}


def _rayleigh_can_params(p_locations):
    key = (id(p_locations), p_locations.shape[0])
    c = _RAYLEIGH_CAN_CACHE.get(key)
    if c is None:
        rho = np.sqrt(p_locations[:, 0] ** 2 + p_locations[:, 2] ** 2)
        c = (float(np.percentile(rho, 98.0)),
             float(p_locations[:, 1].min()), float(p_locations[:, 1].max()))
        if len(_RAYLEIGH_CAN_CACHE) > 64:
            _RAYLEIGH_CAN_CACHE.clear()
        _RAYLEIGH_CAN_CACHE[key] = c
    return c



@njit(cache=True)
def _rayleigh_accumulate_numba(p_locations, direction_zs, Mx, My, Mz, Ox, Oy, Oz,
                               X0x, X0y, X0z, W, r_direct, dcut, soft):
    """Compiled ray x PMT accumulator for the single-scatter field.
    Mirrors _accumulate_refined_delta_numba's structure so it compiles under
    numba in production (runs as pure python under the sandbox shim).
    Geometry/phase/acceptance identical to the numpy path."""
    n_pmts = p_locations.shape[0]
    n_ray = Mx.shape[0]
    mu = np.zeros(n_pmts, dtype=np.float64)
    wsum = np.zeros(n_pmts, dtype=np.float64)   # sum of W*kern (no tacc) for Pbar_eff
    inv3_16pi = 3.0 / (16.0 * math.pi)
    for i in range(n_pmts):
        px = p_locations[i, 0]; py = p_locations[i, 1]; pz = p_locations[i, 2]
        nx = direction_zs[i, 0]; ny = direction_zs[i, 1]; nz = direction_zs[i, 2]
        acc = 0.0; accw = 0.0
        for j in range(n_ray):
            dx = px - Mx[j]; dy = py - My[j]; dz = pz - Mz[j]
            r = math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01
            inv_r = 1.0 / r
            cosT = (Ox[j]*dx + Oy[j]*dy + Oz[j]*dz) * inv_r
            phase = inv3_16pi * (1.0 + cosT*cosT)
            cost = -(nx*dx + ny*dy + nz*dz) * inv_r
            if cost <= 0.0:
                continue
            _xc = cost if cost < 1.0 else 1.0
            _fp = _xc * 4095.0
            _ip = int(_fp)
            if _ip >= 4095:
                pw = _POWER_LAW_LUT[4095]
            else:
                _fr = _fp - _ip
                pw = (
                    _POWER_LAW_LUT[_ip] * (1.0 - _fr)
                    + _POWER_LAW_LUT[_ip + 1] * _fr
                )
            # finite-disk solid-angle (a=60, R0=1000), inline
            omg = (1.0 - r / math.sqrt(r*r + _RL_A2)) * _RL_INV_RF
            if omg < 0.0:
                omg = 0.0
            kern = phase * omg * pw
            if kern <= 0.0:
                continue
            # extra path vs direct -> timing acceptance
            dxr = px - X0x[j]; dyr = py - X0y[j]; dzr = pz - X0z[j]
            rdir = math.sqrt(dxr*dxr + dyr*dyr + dzr*dzr) + 0.01
            extra = r_direct[j] + r - rdir
            _z = (extra - dcut) / soft
            if _z > 8.0:
                tacc = 0.0
            elif _z < -8.0:
                tacc = 1.0
            else:
                tacc = 0.5 * (1.0 - math.tanh(_z))
            wk = W[j] * kern
            acc += wk * tacc
            accw += wk
        mu[i] = acc; wsum[i] = accw
    return mu, wsum


@njit(cache=True)
def _rayleigh_accumulate_numba_2d(p_locations, direction_zs, M, OM, W, X0,
                                  r_direct, dcut, soft, pw_lut):
    """Same Rayleigh accumulator as _rayleigh_accumulate_numba, but accepts
    compact 2D ray arrays directly.

    This avoids making six strided column copies (Mx/My/Mz/Ox/Oy/Oz) and three
    more X0 column copies on every FCN call.  The physics and arithmetic order
    inside each PMT's ray loop are intentionally unchanged.
    """
    n_pmts = p_locations.shape[0]
    n_ray = M.shape[0]
    mu = np.zeros(n_pmts, dtype=np.float64)
    wsum = np.zeros(n_pmts, dtype=np.float64)
    inv3_16pi = 3.0 / (16.0 * math.pi)
    for i in range(n_pmts):
        px = p_locations[i, 0]; py = p_locations[i, 1]; pz = p_locations[i, 2]
        nx = direction_zs[i, 0]; ny = direction_zs[i, 1]; nz = direction_zs[i, 2]
        acc = 0.0; accw = 0.0
        for j in range(n_ray):
            dx = px - M[j, 0]; dy = py - M[j, 1]; dz = pz - M[j, 2]
            r = math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01
            inv_r = 1.0 / r
            cosT = (OM[j, 0]*dx + OM[j, 1]*dy + OM[j, 2]*dz) * inv_r
            phase = inv3_16pi * (1.0 + cosT*cosT)
            cost = -(nx*dx + ny*dy + nz*dz) * inv_r
            if cost <= 0.0:
                continue
            _xc = cost if cost < 1.0 else 1.0
            _fp = _xc * 4095.0
            _ip = int(_fp)
            if _ip >= 4095:
                pw = pw_lut[4095]
            else:
                _fr = _fp - _ip
                pw = pw_lut[_ip] * (1.0 - _fr) + pw_lut[_ip + 1] * _fr
            omg = (1.0 - r / math.sqrt(r*r + _RL_A2)) * _RL_INV_RF
            if omg < 0.0:
                omg = 0.0
            kern = phase * omg * pw
            if kern <= 0.0:
                continue
            dxr = px - X0[j, 0]; dyr = py - X0[j, 1]; dzr = pz - X0[j, 2]
            rdir = math.sqrt(dxr*dxr + dyr*dyr + dzr*dzr) + 0.01
            extra = r_direct[j] + r - rdir
            _z = (extra - dcut) / soft
            if _z > 8.0:
                tacc = 0.0
            elif _z < -8.0:
                tacc = 1.0
            else:
                tacc = 0.5 * (1.0 - math.tanh(_z))
            wk = W[j] * kern
            acc += wk * tacc
            accw += wk
        mu[i] = acc; wsum[i] = accw
    return mu, wsum


@njit(cache=True, parallel=True)
def _rayleigh_accumulate_numba_2d_parallel(p_locations, direction_zs, M, OM, W, X0,
                                           r_direct, dcut, soft, pw_lut):
    """Parallel PMT-loop version of _rayleigh_accumulate_numba_2d.

    Each PMT is independent and keeps the same serial ray summation order, so
    this changes only scheduling across PMTs.  It is intended for single-process
    per-event Minuit fits.  If using forked multiprocessing after import, set
    emitter.rayleigh_use_parallel_accumulator=False to avoid OpenMP/fork issues.
    """
    n_pmts = p_locations.shape[0]
    n_ray = M.shape[0]
    mu = np.zeros(n_pmts, dtype=np.float64)
    wsum = np.zeros(n_pmts, dtype=np.float64)
    inv3_16pi = 3.0 / (16.0 * math.pi)
    for i in prange(n_pmts):
        px = p_locations[i, 0]; py = p_locations[i, 1]; pz = p_locations[i, 2]
        nx = direction_zs[i, 0]; ny = direction_zs[i, 1]; nz = direction_zs[i, 2]
        acc = 0.0; accw = 0.0
        for j in range(n_ray):
            dx = px - M[j, 0]; dy = py - M[j, 1]; dz = pz - M[j, 2]
            r = math.sqrt(dx*dx + dy*dy + dz*dz) + 0.01
            inv_r = 1.0 / r
            cosT = (OM[j, 0]*dx + OM[j, 1]*dy + OM[j, 2]*dz) * inv_r
            phase = inv3_16pi * (1.0 + cosT*cosT)
            cost = -(nx*dx + ny*dy + nz*dz) * inv_r
            if cost <= 0.0:
                continue
            _xc = cost if cost < 1.0 else 1.0
            _fp = _xc * 4095.0
            _ip = int(_fp)
            if _ip >= 4095:
                pw = pw_lut[4095]
            else:
                _fr = _fp - _ip
                pw = pw_lut[_ip] * (1.0 - _fr) + pw_lut[_ip + 1] * _fr
            omg = (1.0 - r / math.sqrt(r*r + _RL_A2)) * _RL_INV_RF
            if omg < 0.0:
                omg = 0.0
            kern = phase * omg * pw
            if kern <= 0.0:
                continue
            dxr = px - X0[j, 0]; dyr = py - X0[j, 1]; dzr = pz - X0[j, 2]
            rdir = math.sqrt(dxr*dxr + dyr*dyr + dzr*dzr) + 0.01
            extra = r_direct[j] + r - rdir
            _z = (extra - dcut) / soft
            if _z > 8.0:
                tacc = 0.0
            elif _z < -8.0:
                tacc = 1.0
            else:
                tacc = 0.5 * (1.0 - math.tanh(_z))
            wk = W[j] * kern
            acc += wk * tacc
            accw += wk
        mu[i] = acc; wsum[i] = accw
    return mu, wsum


def _rayleigh_scatter_field(em, p_locations, direction_zs, start_pos, track_dir,
                            visible_L):
    """Return (mu_scat_shape, Pbar_eff) for the single-scatter Rayleigh model.

    The default ``rayleigh_cache_mode='quantized'`` uses deterministic binned Rayleigh reuse for speed.
    Faster modes keep the same single-scatter formula but avoid recomputing it
    at every near-identical Minuit hypothesis:

      quantized
          Default. Evaluate at binned x/y/z/length/direction.  This turns the smooth
          Rayleigh field into a controlled interpolation-by-nearest-bin and can
          reuse many FCN calls while keeping the result close to exact.
      exact / smooth
          Historical smooth/exact behavior with exact parameters in the cache key.
      tolerant_last
          Reuse the previous exact field while the track moves less than small
          physical tolerances; otherwise recompute exactly.
      frozen_init
          Always evaluate at the track used when the Emitter was constructed.
          This is the fastest mode and is useful because prompt Rayleigh is a
          small, slowly varying correction, but it must be validated against
          exact per-event fits before production use.
    """
    lam = float(getattr(em, "rayleigh_scatter_length_mm", 45000.0))
    n_s = int(getattr(em, "rayleigh_n_sources", 6))
    n_phi = int(getattr(em, "rayleigh_n_phi", 10))
    dt_ns = float(getattr(em, "rayleigh_timing_cut_ns", 5.0))
    soft = max(float(getattr(em, "rayleigh_timing_soft_mm", 150.0)), 1.0)
    if lam <= 0.0 or visible_L <= 0.0:
        return np.zeros(p_locations.shape[0]), 0.0

    p_locations = np.ascontiguousarray(p_locations, dtype=np.float64)
    direction_zs = np.ascontiguousarray(direction_zs, dtype=np.float64)
    start_pos_arr = np.asarray(start_pos, dtype=np.float64)
    d = np.asarray(track_dir, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-30)
    visible_L = float(visible_L)

    mode = str(getattr(em, "rayleigh_cache_mode", "quantized")).strip().lower()
    if mode in {"none", "off", "0"}:
        mode = "quantized"
    if mode in {"smooth", "smoothed", "smooth_exact", "historical"}:
        mode = "exact"

    # Fast reuse of the previous exact result within a physically small step.
    # This is checked before the exact-cache path because it intentionally allows
    # approximate reuse across nearby floating-point hypotheses.
    if mode in {"tolerant", "tolerant_last", "reuse_last"}:
        last = getattr(em, "_rayleigh_tolerant_last", None)
        if last is not None:
            last_x, last_d, last_L, last_result = last
            tol_xyz = float(getattr(em, "rayleigh_reuse_tolerance_xyz_mm", 20.0))
            tol_L = float(getattr(em, "rayleigh_reuse_tolerance_length_mm", 20.0))
            tol_dir = float(getattr(em, "rayleigh_reuse_tolerance_dir", 0.005))
            if (np.max(np.abs(start_pos_arr - last_x)) <= tol_xyz and
                abs(visible_L - last_L) <= tol_L and
                np.max(np.abs(d - last_d)) <= tol_dir):
                return last_result

    # Choose the track at which to evaluate the Rayleigh field.  For exact and
    # tolerant_last this is the current track.  Quantized/frozen modes deliberately
    # evaluate a nearby representative track so cache hits occur during fits.
    eval_start = start_pos_arr.copy()
    eval_d = d.copy()
    eval_L = float(visible_L)

    if mode in {"frozen", "frozen_init", "init", "static"}:
        eval_start = np.asarray(getattr(em, "_rayleigh_init_start_coord", start_pos_arr), dtype=np.float64)
        eval_d = np.asarray(getattr(em, "_rayleigh_init_direction", d), dtype=np.float64)
        eval_d = eval_d / max(float(np.linalg.norm(eval_d)), 1e-30)
        eval_L = float(getattr(em, "_rayleigh_init_length", visible_L))
    elif mode in {"quantized", "quantised", "binned"}:
        qxyz = max(float(getattr(em, "rayleigh_quantize_xyz_mm", 20.0)), 1e-9)
        qL = max(float(getattr(em, "rayleigh_quantize_length_mm", 20.0)), 1e-9)
        qd = max(float(getattr(em, "rayleigh_quantize_dir", 0.005)), 1e-9)
        eval_start = np.round(eval_start / qxyz) * qxyz
        eval_L = float(round(eval_L / qL) * qL)
        # Quantize the two transverse direction cosines while preserving the
        # current z hemisphere.  The historical implementation always rebuilt
        # +sqrt(1-cx^2-cy^2), which silently flipped negative-cz hypotheses in
        # the quantized legacy-Rayleigh branch.
        qx = round(float(eval_d[0]) / qd) * qd
        qy = round(float(eval_d[1]) / qd) * qd
        r2 = qx*qx + qy*qy
        if r2 < 0.999999:
            z_sign = -1.0 if float(eval_d[2]) < 0.0 else 1.0
            eval_d = np.array(
                [qx, qy, z_sign * math.sqrt(max(1.0 - r2, 1e-12))],
                dtype=np.float64,
            )
        else:
            eval_d = d.copy()

    if eval_L <= 0.0:
        return np.zeros(p_locations.shape[0]), 0.0

    if bool(getattr(em, "rayleigh_enable_exact_cache", True)):
        cache_key = (
            str(mode), id(p_locations), id(direction_zs), p_locations.shape, direction_zs.shape,
            float(eval_L), float(lam), int(n_s), int(n_phi), float(dt_ns), float(soft),
            float(eval_start[0]), float(eval_start[1]), float(eval_start[2]),
            float(eval_d[0]), float(eval_d[1]), float(eval_d[2]),
            bool(getattr(em, "rayleigh_use_parallel_accumulator", True)),
        )
        cache = getattr(em, "_rayleigh_exact_cache", None)
        if cache is not None and cache_key in cache:
            result = cache[cache_key]
            if mode in {"tolerant", "tolerant_last", "reuse_last"}:
                em._rayleigh_tolerant_last = (start_pos_arr.copy(), d.copy(), visible_L, result)
            return result
    else:
        cache_key = None

    R_can, y_bot, y_top = _rayleigh_can_params(p_locations)
    h = np.array([1.0, 0.0, 0.0]) if abs(eval_d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(eval_d, h); u /= np.linalg.norm(u); v = np.cross(eval_d, u)

    s_c = (np.arange(n_s) + 0.5) * float(eval_L) / n_s
    K = np.asarray(em.muon_energy_at_s_array(
        s_c, float(getattr(em, "range_to_threshold_mm", eval_L))), float)
    ft = np.maximum(np.asarray(em.frank_tamm_factor(K, em.particle_mass), float), 0.0)
    if ft.sum() <= 0.0:
        return np.zeros(p_locations.shape[0]), 0.0
    gam = 1.0 + K / em.particle_mass
    beta = np.sqrt(np.clip(1.0 - 1.0 / gam ** 2, 1e-9, 1.0))
    cth = np.clip(1.0 / (em.n * beta), -1.0, 1.0); sth = np.sqrt(1.0 - cth ** 2)
    phi = (np.arange(n_phi) + 0.5) * 2.0 * np.pi / n_phi
    om = (cth[:, None, None] * eval_d[None, None, :]
          + (sth[:, None] * np.cos(phi)[None, :])[..., None] * u[None, None, :]
          + (sth[:, None] * np.sin(phi)[None, :])[..., None] * v[None, None, :])
    x0 = eval_start[None, None, :] + s_c[:, None, None] * eval_d[None, None, :]
    x0 = np.broadcast_to(x0, om.shape).copy()

    a2 = om[..., 0] ** 2 + om[..., 2] ** 2
    b2 = 2.0 * (x0[..., 0] * om[..., 0] + x0[..., 2] * om[..., 2])
    c2 = x0[..., 0] ** 2 + x0[..., 2] ** 2 - R_can ** 2
    disc = np.maximum(b2 * b2 - 4.0 * a2 * c2, 0.0)
    tside = np.where(a2 > 1e-12, (-b2 + np.sqrt(disc)) / (2.0 * np.maximum(a2, 1e-12)), np.inf)
    wy = om[..., 1]
    tt = np.where(wy > 1e-12, (y_top - x0[..., 1]) / np.where(np.abs(wy) > 1e-12, wy, 1.0), np.inf)
    tb = np.where(wy < -1e-12, (y_bot - x0[..., 1]) / np.where(np.abs(wy) > 1e-12, wy, 1.0), np.inf)
    T = np.clip(np.minimum(tside, np.minimum(np.where(tt > 0, tt, np.inf),
                                             np.where(tb > 0, tb, np.inf))), 0.0, 4000.0)

    w = (ft[:, None] / n_phi) * (T / lam)
    Pbar = float(w.sum() / ft.sum())
    m = x0 + 0.5 * T[..., None] * om
    M = np.ascontiguousarray(m.reshape(-1, 3), dtype=np.float64)
    OM = np.ascontiguousarray(om.reshape(-1, 3), dtype=np.float64)
    W = np.ascontiguousarray(w.reshape(-1), dtype=np.float64)
    X0 = np.ascontiguousarray(x0.reshape(-1, 3), dtype=np.float64)
    r_direct = np.ascontiguousarray(np.linalg.norm(M - X0, axis=1), dtype=np.float64)  # = 0.5*T
    dcut = dt_ns * (em.c / em.n)
    if bool(getattr(em, "rayleigh_use_parallel_accumulator", True)):
        mu, wsum = _rayleigh_accumulate_numba_2d_parallel(
            p_locations, direction_zs, M, OM, W, X0,
            r_direct, float(dcut), float(soft), _POWER_LAW_LUT,
        )
    else:
        mu, wsum = _rayleigh_accumulate_numba_2d(
            p_locations, direction_zs, M, OM, W, X0,
            r_direct, float(dcut), float(soft), _POWER_LAW_LUT,
        )
    mu_sum = float(mu.sum()); w_sum = float(wsum.sum())
    Pbar_eff = Pbar * (mu_sum / w_sum) if w_sum > 0.0 else 0.0
    result = (mu, Pbar_eff)

    if mode in {"tolerant", "tolerant_last", "reuse_last"}:
        em._rayleigh_tolerant_last = (start_pos_arr.copy(), d.copy(), visible_L, result)

    if cache_key is not None:
        cache = getattr(em, "_rayleigh_exact_cache", None)
        if cache is None:
            cache = {}
            em._rayleigh_exact_cache = cache
        cache[cache_key] = result
        max_entries = int(getattr(em, "rayleigh_exact_cache_max_entries", 16))
        if max_entries > 0 and len(cache) > max_entries:
            for _k in list(cache.keys())[:len(cache) - max_entries]:
                del cache[_k]
    return result

@njit(cache=True)
def _rel_mpmt_eff_scalar_numba(cost, code, table):
    if code < 0 or code >= table.shape[0]:
        return 1.0
    n_grid = table.shape[1]
    if n_grid < 2:
        return 1.0
    x = cost
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    x *= (n_grid - 1)
    i0 = int(math.floor(x))
    if i0 < 0:
        i0 = 0
    elif i0 > n_grid - 2:
        i0 = n_grid - 2
    t = x - i0
    y0 = table[code, i0]
    y1 = table[code, i0 + 1]
    return y0 + t * (y1 - y0)


@njit(cache=True)
def _finite_disk_rel_scalar_numba(r, pmt_radius_mm, ref_r_mm):
    r_safe = r
    if r_safe < 1e-9:
        r_safe = 1e-9
    a = pmt_radius_mm
    R0 = ref_r_mm
    if a <= 0.0:
        return (R0 / r_safe) * (R0 / r_safe)
    omega_shape = 1.0 - r_safe / math.sqrt(r_safe * r_safe + a * a)
    omega_ref = 1.0 - R0 / math.sqrt(R0 * R0 + a * a)
    if (not math.isfinite(omega_ref)) or omega_ref <= 0.0:
        return (R0 / r_safe) * (R0 / r_safe)
    out = omega_shape / omega_ref
    if (not math.isfinite(out)) or out < 0.0:
        return 0.0
    return out


@njit(cache=True)
def _refined_delta_dSdu_scalar_numba(K, u, K_grid, u_grid, table):
    """Scalar bilinear interpolation of dS_delta/du(K,u)."""
    if (not math.isfinite(K)) or (not math.isfinite(u)):
        return 0.0
    if K < K_grid[0] or u < -1.0 or u > 1.0:
        return 0.0

    # Clip high values exactly like the vectorized implementation.
    Kc = K
    if Kc > K_grid[K_grid.size - 1]:
        Kc = K_grid[K_grid.size - 1]
    uc = u
    if uc < u_grid[0]:
        uc = u_grid[0]
    elif uc > u_grid[u_grid.size - 1]:
        uc = u_grid[u_grid.size - 1]

    # Uniform grids in current table builder, so use direct index math.
    dK = K_grid[1] - K_grid[0]
    iK = int(math.floor((Kc - K_grid[0]) / dK))
    if iK < 0:
        iK = 0
    elif iK > K_grid.size - 2:
        iK = K_grid.size - 2
    K0 = K_grid[iK]
    K1 = K_grid[iK + 1]
    tK = (Kc - K0) / (K1 - K0 + 1e-300)
    if tK < 0.0:
        tK = 0.0
    elif tK > 1.0:
        tK = 1.0

    du = u_grid[1] - u_grid[0]
    iu = int(math.floor((uc - u_grid[0]) / du))
    if iu < 0:
        iu = 0
    elif iu > u_grid.size - 2:
        iu = u_grid.size - 2
    u0 = u_grid[iu]
    u1 = u_grid[iu + 1]
    tu = (uc - u0) / (u1 - u0 + 1e-300)
    if tu < 0.0:
        tu = 0.0
    elif tu > 1.0:
        tu = 1.0

    p00 = table[iK, iu]
    p01 = table[iK, iu + 1]
    p10 = table[iK + 1, iu]
    p11 = table[iK + 1, iu + 1]
    p0 = p00 + tu * (p01 - p00)
    p1 = p10 + tu * (p11 - p10)
    out = p0 + tK * (p1 - p0)
    if (not math.isfinite(out)) or out < 0.0:
        return 0.0
    return out


@njit(cache=True, fastmath=True)
def _accumulate_refined_delta_numba(
    p_locations,
    direction_zs,
    start_pos,
    track_dir,
    s_centers,
    ds_cm,
    K_mu,
    K_grid,
    u_grid,
    table,
    mpmt_codes,
    rel_eff_table,
    apply_mpmt_eff,
    use_finite_disk,
    pmt_radius_mm,
    ref_r_mm,
    distance_power,
    analytic_delta_scale,
    source_k_power,
    source_k_ref,
    source_k_floor,
    intensity,
    starting_time,
    v,
    n_water,
    c_light,
    delta_e_time_offset_ns,
    return_times,
    return_nodes,
    node_col_for_pmt,
    n_node_cols,
    source_tof,
    node_group_index_over_c,
    return_node_times,
    pl_lut,
    cost_soft,
    src_lo,
    src_hi,
    use_seg_gate,
):
    """
    Fast PMT-parallel secondary-electron accumulator.

    This keeps the same physics as the original source x PMT loop, but removes
    repeated source-only work from the hot PMT loop:

      * source positions, source times, source weights are precomputed once;
      * K-grid interpolation indices/fractions are precomputed once per source;
      * each Numba thread accumulates one PMT, avoiding write conflicts;
      * impossible contributions are rejected before expensive optical factors.

    The table is still interpreted as dS_delta/du(K_mu, u), where u is the
    photon direction cosine relative to the primary muon direction.
    """
    n_src = s_centers.size
    n_pmts = p_locations.shape[0]

    mu = np.zeros(n_pmts, dtype=np.float64)
    tnum = np.zeros(n_pmts, dtype=np.float64)
    if return_nodes:
        node_mu = np.zeros((n_src, n_node_cols), dtype=np.float64)
        node_t = np.empty((n_src, n_node_cols), dtype=np.float64)
        for jj in range(n_src):
            for ii in range(n_node_cols):
                node_t[jj, ii] = np.inf
    else:
        node_mu = np.empty((0, 0), dtype=np.float64)
        node_t = np.empty((0, 0), dtype=np.float64)

    if n_src == 0 or n_pmts == 0:
        if return_times:
            t_empty = np.empty(n_pmts, dtype=np.float64)
            for i in range(n_pmts):
                t_empty[i] = np.nan
            return mu, t_empty, node_mu, node_t
        return mu, tnum, node_mu, node_t

    # ------------------------------------------------------------------
    # Precompute source-only quantities.
    # ------------------------------------------------------------------
    src_x = np.empty(n_src, dtype=np.float64)
    src_y = np.empty(n_src, dtype=np.float64)
    src_z = np.empty(n_src, dtype=np.float64)
    src_t = np.empty(n_src, dtype=np.float64)
    src_w = np.empty(n_src, dtype=np.float64)
    src_iK = np.empty(n_src, dtype=np.int64)
    src_tK = np.empty(n_src, dtype=np.float64)
    src_valid = np.zeros(n_src, dtype=np.uint8)

    K_min = K_grid[0]
    K_max = K_grid[K_grid.size - 1]
    dK = K_grid[1] - K_grid[0]
    inv_dK = 1.0 / dK
    nK = K_grid.size

    for j in range(n_src):
        K = K_mu[j]
        ds = ds_cm[j]

        if (not math.isfinite(K)) or K <= 0.0:
            continue
        if (not math.isfinite(ds)) or ds <= 0.0:
            continue

        if source_k_power == 0.0:
            source_weight = 1.0
        else:
            K_for_weight = K
            if K_for_weight < source_k_floor:
                K_for_weight = source_k_floor

            if source_k_ref <= 0.0:
                source_weight = 1.0
            else:
                source_weight = (K_for_weight / source_k_ref) ** source_k_power

        w_src = analytic_delta_scale * source_weight * ds
        if (not math.isfinite(w_src)) or w_src <= 0.0:
            continue

        s = s_centers[j]
        src_x[j] = start_pos[0] + s * track_dir[0]
        src_y[j] = start_pos[1] + s * track_dir[1]
        src_z[j] = start_pos[2] + s * track_dir[2]
        src_w[j] = w_src

        if return_times:
            src_t[j] = starting_time + s / v + delta_e_time_offset_ns
        else:
            src_t[j] = 0.0

        # The original scalar interpolation returned zero for K below the
        # table minimum.  In this model K_grid[0] is 0, and nonpositive K has
        # already been filtered, so the lower clip only protects roundoff.
        Kc = K
        if Kc < K_min:
            Kc = K_min
        elif Kc > K_max:
            Kc = K_max

        iK = int(math.floor((Kc - K_min) * inv_dK))
        if iK < 0:
            iK = 0
        elif iK > nK - 2:
            iK = nK - 2

        K0 = K_grid[iK]
        K1 = K_grid[iK + 1]
        tK = (Kc - K0) / (K1 - K0 + 1e-300)
        if tK < 0.0:
            tK = 0.0
        elif tK > 1.0:
            tK = 1.0

        src_iK[j] = iK
        src_tK[j] = tK
        src_valid[j] = 1

    # ------------------------------------------------------------------
    # Constants for u-grid interpolation.
    # ------------------------------------------------------------------
    u_min = u_grid[0]
    u_max = u_grid[u_grid.size - 1]
    du = u_grid[1] - u_grid[0]
    inv_du = 1.0 / du
    nU = u_grid.size

    # omega_ref is loop-invariant (depends only on pmt_radius_mm and ref_r_mm),
    # so compute it once instead of inside _finite_disk_rel_scalar_numba on every
    # (PMT, source) pair.  The per-pair sqrt(r^2 + a^2) is left untouched so this
    # is bit-identical to calling the helper.
    fd_a = pmt_radius_mm
    fd_R0 = ref_r_mm
    fd_use_ratio = use_finite_disk and (fd_a > 0.0)
    fd_omega_ref = 1.0 - fd_R0 / math.sqrt(fd_R0 * fd_R0 + fd_a * fd_a)
    fd_omega_ref_ok = fd_use_ratio and math.isfinite(fd_omega_ref) and (fd_omega_ref > 0.0)

    # ------------------------------------------------------------------
    # PMT accumulation.
    # IMPORTANT: this is intentionally not parallelized with prange/OpenMP.
    # Some batch drivers fork worker processes after importing/compiling this
    # module, and GNU OpenMP aborts on fork-after-OpenMP.  Keeping this loop
    # serial preserves multiprocessing compatibility while retaining the
    # source precomputation and fast interpolation optimizations.
    # ------------------------------------------------------------------
    for i in range(n_pmts):
        px = p_locations[i, 0]
        py = p_locations[i, 1]
        pz = p_locations[i, 2]

        nx = direction_zs[i, 0]
        ny = direction_zs[i, 1]
        nz = direction_zs[i, 2]

        mpmt_code = mpmt_codes[i]
        node_col = node_col_for_pmt[i] if return_nodes else -1

        mu_i = 0.0
        tnum_i = 0.0

        for j in range(n_src):
            if src_valid[j] == 0:
                continue

            dx = px - src_x[j]
            dy = py - src_y[j]
            dz = pz - src_z[j]

            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0:
                continue

            r = math.sqrt(r2) + 0.01
            inv_r = 1.0 / r

            # Direction cosine of photon direction relative to the muon.
            # The refined table now spans the full range -1 <= u <= 1, so
            # backward-hemisphere PMTs receive the (physical) backward knock-on
            # light instead of being dropped.  Values slightly outside [-1, 1]
            # from roundoff are clamped to the nearest edge below.
            u = (dx * track_dir[0] + dy * track_dir[1] + dz * track_dir[2]) * inv_r
            if (not math.isfinite(u)) or u < -1.0:
                continue
            if u > 1.0:
                u = 1.0

            # PMT-facing factor.  This is checked before the optical response;
            # negative values cannot contribute.
            cost = -(dx * nx + dy * ny + dz * nz) * inv_r
            if not math.isfinite(cost):
                continue

            # SMOOTH-NLL: segment-averaged visibility gate.
            # Each secondary source represents a finite track segment, but its
            # light is lumped at the segment center.  With a hard point gate,
            # a PMT whose horizon (cost=0) lies inside the segment turns fully
            # on/off as the fitted parameters slide the center across it -- a
            # step of up to several NLL units (verified: one crossing tripled
            # a tube's delta charge within 0.1 mm of L).  Here the gate weight
            # is instead the fraction of the segment on the visible side,
            # linear from the sign of the (unnormalized) facing product at the
            # segment endpoints: continuous in the parameters (kink, not step),
            # and light-conserving.  use_seg_gate=0 restores the point gate.
            vis_seg = 1.0
            if use_seg_gate != 0:
                a_lo = -((px - src_lo[j, 0]) * nx + (py - src_lo[j, 1]) * ny + (pz - src_lo[j, 2]) * nz)
                a_hi = -((px - src_hi[j, 0]) * nx + (py - src_hi[j, 1]) * ny + (pz - src_hi[j, 2]) * nz)
                if a_lo <= 0.0 and a_hi <= 0.0:
                    continue
                if a_lo > 0.0 and a_hi > 0.0:
                    vis_seg = 1.0
                elif a_hi > 0.0:
                    vis_seg = a_hi / (a_hi - a_lo)
                else:
                    vis_seg = a_lo / (a_lo - a_hi)
                if cost <= 0.0:
                    # partial-visibility segment whose center is just behind the
                    # horizon: evaluate the optical factors at grazing incidence.
                    cost = 0.0
            elif cost <= 0.0:
                continue

            # SMOOTH-NLL: soft visibility ramp at the PMT-facing gate.
            # The hard cost<=0 cutoff combined with power_law(0+)~0.125 makes
            # each (source, PMT) contribution a step function of the track
            # parameters; as sources slide with the fitted length, these steps
            # imprint discontinuities of up to several NLL units on the FCN
            # (verified: a single edge-on crossing tripled a tube's expected
            # delta charge within 0.1 mm of L).  The smoothstep below takes the
            # contribution to zero continuously over cost in [0, cost_soft],
            # mirroring the primary path's cost_soft option.  cost_soft<=0
            # restores the historical hard gate exactly.
            if cost_soft > 0.0:
                if cost >= cost_soft:
                    vis_d = 1.0
                else:
                    uu_d = cost / cost_soft
                    vis_d = uu_d * uu_d * (3.0 - 2.0 * uu_d)
            else:
                vis_d = 1.0

            # ----------------------------------------------------------
            # Fast bilinear interpolation of dS_delta/du(K, u).
            # K interpolation terms are source-only and were precomputed.
            # ----------------------------------------------------------
            uc = u
            if uc < u_min:
                uc = u_min
            elif uc > u_max:
                uc = u_max

            iu = int(math.floor((uc - u_min) * inv_du))
            if iu < 0:
                iu = 0
            elif iu > nU - 2:
                iu = nU - 2

            u0 = u_grid[iu]
            u1 = u_grid[iu + 1]
            tu = (uc - u0) / (u1 - u0 + 1e-300)
            if tu < 0.0:
                tu = 0.0
            elif tu > 1.0:
                tu = 1.0

            iK = src_iK[j]
            tK = src_tK[j]

            p00 = table[iK, iu]
            p01 = table[iK, iu + 1]
            p10 = table[iK + 1, iu]
            p11 = table[iK + 1, iu + 1]

            p0 = p00 + tu * (p01 - p00)
            p1 = p10 + tu * (p11 - p10)
            kernel = p0 + tK * (p1 - p0)

            if (not math.isfinite(kernel)) or kernel <= 0.0:
                continue

            pwr = _power_law_lut_scalar_numba(cost, pl_lut)

            if use_finite_disk:
                # Faithful inline of _finite_disk_rel_scalar_numba with the
                # loop-invariant omega_ref hoisted out (see above).  r already
                # includes the +0.01 used by the helper's r_safe (r > 1e-9 here).
                if fd_omega_ref_ok:
                    fd_shape = 1.0 - r / math.sqrt(r * r + fd_a * fd_a)
                    fd_out = fd_shape / fd_omega_ref
                    if (not math.isfinite(fd_out)) or fd_out < 0.0:
                        fd_out = 0.0
                    optical = fd_out * pwr
                else:
                    optical = (fd_R0 / r) * (fd_R0 / r) * pwr
            else:
                R0 = 1000.0
                optical = (R0 / r) ** distance_power * pwr

            if apply_mpmt_eff:
                optical *= _rel_mpmt_eff_scalar_numba(cost, mpmt_code, rel_eff_table)

            if (not math.isfinite(optical)) or optical <= 0.0:
                continue

            contrib = src_w[j] * optical * kernel * vis_d * vis_seg
            mu_i += contrib
            if node_col >= 0:
                node_mu[j, node_col] = contrib * intensity
                if return_node_times:
                    node_t[j, node_col] = (
                        starting_time + source_tof[j]
                        + r * node_group_index_over_c
                    )

            if return_times:
                t_delta = src_t[j] + r * n_water / c_light
                tnum_i += contrib * t_delta

        mu_i *= intensity
        tnum_i *= intensity
        mu[i] = mu_i
        tnum[i] = tnum_i

    if return_times:
        t = np.empty(n_pmts, dtype=np.float64)
        for i in range(n_pmts):
            if mu[i] > 0.0 and math.isfinite(mu[i]) and math.isfinite(tnum[i]):
                t[i] = tnum[i] / mu[i]
            else:
                t[i] = np.nan
        return mu, t, node_mu, node_t

    return mu, tnum, node_mu, node_t



@njit(cache=True)
def _disk_cdf_unit_numba(z):
    """CDF of one coordinate for a uniformly filled unit disk."""
    if z <= -1.0:
        return 0.0
    if z >= 1.0:
        return 1.0
    zz = z
    root = math.sqrt(max(0.0, 1.0 - zz * zz))
    return 0.5 + (math.asin(zz) + zz * root) / math.pi


@njit(cache=True)
def _disk_moment_z_numba(zlo, zhi):
    """Integral of z * p_disk(z) dz over zlo..zhi for a unit disk coordinate."""
    if zlo < -1.0:
        zlo = -1.0
    elif zlo > 1.0:
        zlo = 1.0
    if zhi < -1.0:
        zhi = -1.0
    elif zhi > 1.0:
        zhi = 1.0
    if zhi <= zlo:
        return 0.0
    a_lo = max(0.0, 1.0 - zlo * zlo)
    a_hi = max(0.0, 1.0 - zhi * zhi)
    return (2.0 / (3.0 * math.pi)) * (a_lo ** 1.5 - a_hi ** 1.5)


@njit(cache=True)
def _endpoint_rootdisk_weight_and_mean_s_numba(s_center, track_length, h, scope_code):
    """Finite-aperture overlap fraction and mean source coordinate.

    scope_code: 0 start edge only, 1 end edge only, 2 full finite interval.
    """
    L = track_length
    if L < 0.0:
        L = 0.0
    if (not math.isfinite(s_center)) or (not math.isfinite(L)):
        return 0.0, 0.0
    if (not math.isfinite(h)) or h <= 1.0e-12:
        inside = False
        if scope_code == 0:
            inside = s_center >= 0.0
        elif scope_code == 1:
            inside = s_center <= L
        else:
            inside = (s_center >= 0.0 and s_center <= L)
        if inside:
            sm = s_center
            if sm < 0.0:
                sm = 0.0
            elif sm > L:
                sm = L
            return 1.0, sm
        return 0.0, 0.0

    if scope_code == 0:
        zlo_raw = (-s_center) / h
        zhi_raw = 1.0
    elif scope_code == 1:
        zlo_raw = -1.0
        zhi_raw = (L - s_center) / h
    else:
        zlo_raw = (-s_center) / h
        zhi_raw = (L - s_center) / h

    w = _disk_cdf_unit_numba(zhi_raw) - _disk_cdf_unit_numba(zlo_raw)
    if (not math.isfinite(w)) or w <= 0.0:
        return 0.0, 0.0

    zlo = zlo_raw
    if zlo < -1.0:
        zlo = -1.0
    elif zlo > 1.0:
        zlo = 1.0
    zhi = zhi_raw
    if zhi < -1.0:
        zhi = -1.0
    elif zhi > 1.0:
        zhi = 1.0

    moment_z = _disk_moment_z_numba(zlo, zhi)
    s_mean = s_center + h * moment_z / w
    if s_mean < 0.0:
        s_mean = 0.0
    elif s_mean > L:
        s_mean = L
    return w, s_mean


@njit(cache=True)
def _interp_dedx_scalar_for_endpoint_numba(E, dedx_E_grid, dedx_grid):
    """np.interp equivalent for the stopping-power table, scalar/numba."""
    n = dedx_E_grid.shape[0]
    if n <= 0:
        return 0.0
    if E <= dedx_E_grid[0]:
        return dedx_grid[0]
    if E >= dedx_E_grid[n - 1]:
        return dedx_grid[n - 1]
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if dedx_E_grid[mid] < E:
            lo = mid + 1
        else:
            hi = mid
    x0 = dedx_E_grid[lo - 1]
    x1 = dedx_E_grid[lo]
    y0 = dedx_grid[lo - 1]
    y1 = dedx_grid[lo]
    if x1 <= x0:
        return y0
    return y0 + (E - x0) / (x1 - x0) * (y1 - y0)


@njit(cache=True, fastmath=True)
def _fused_primary_kernel(
    p_locations,        # n_pmts x 3
    direction_zs,       # n_pmts x 3
    s,                  # n_pmts (raw emission coordinate)
    scale_in,           # n_pmts (from find_scale; 0 outside collapse set)
    s_b,                # n_pmts (collapsed emission coord)
    E_b,                # n_pmts (energy at emission point)
    start_pos,          # 3
    track_dir,          # 3 (already normalized)
    mpmt_codes,         # n_pmts int16
    rel_eff_table,      # 4 x n_grid
    dedx_E_grid,        # stopping-power table energies
    dedx_grid,          # stopping-power table values
    ngeo_norm,          # scalar primary_ngeo normalization
    ngeo_a,             # primary_ngeo_pmt_radius_mm
    pmt_radius,         # legacy front-face radius (collapse pmt_radius)
    endpoint_mode,      # 0 legacy, 1 root-overlap weight-only, 2 root-overlap weight+mean
    endpoint_scope,     # 0 start, 1 end, 2 both
    endpoint_aperture_radius,
    track_length,
    n_water,
    mu_mass,
    intensity,
    starting_time,
    v,
    c_light,
    need_times,
    cost_soft,
    cost_soft_centered,
    cds_occlusion_enabled,
    cds_axis_x,
    cds_axis_z,
    cds_inner_radius,
    cds_outer_radius,
    cds_y_min,
    cds_y_max,
    cds_aperture_radius,
):
    """Single-pass fusion of the primary-light path in get_expected_pes_ts.

    Reproduces, per PMT, the exact algebra of the vectorized implementation:
      s_eff/front-face/valid-s gating, geometry (dx,dy,dz,r,cost), power_law
      angular response, analytic primary_ngeo cone-density falloff (with inline
      dedx interpolation), and uniform-grid rel-mPMT efficiency.  Produces
      mu_primary and (optionally) t_primary with no intermediate full-length
      arrays.

    Numerically equivalent to the original to ~1e-12 (the only differences are
    floating-point summation/order); NOT bit-identical by construction.
    """
    n_pmts = p_locations.shape[0]
    mu_primary = np.zeros(n_pmts, dtype=np.float64)
    if need_times:
        t_primary = np.empty(n_pmts, dtype=np.float64)
    else:
        t_primary = np.empty(0, dtype=np.float64)

    n_grid = rel_eff_table.shape[1]
    nde = dedx_E_grid.shape[0]

    tdx = track_dir[0]
    tdy = track_dir[1]
    tdz = track_dir[2]
    two_pr = 2.0 * pmt_radius

    # angular response constants (old_power_y0p1209)
    pl_y0 = 0.1209
    pl_yinf = 1.6396999999999999
    pl_nfit = 3.0777000000000001
    pl_x50n = 0.79428866592713121
    pl_max = 1.002379253316015

    for i in range(n_pmts):
        scale = scale_in[i]
        # s_eff = collapsed coord where scale>0, else raw s.
        if scale > 0.0:
            s_eff = s_b[i]
        else:
            s_eff = s[i]

        if endpoint_mode > 0:
            if scale <= 0.0:
                valid_s = False
                s_eff = 0.0
            else:
                # Physics-based finite-aperture endpoint model.  The endpoint
                # overlap is centered on the collapsed cone-root coordinate sb,
                # not the rough one-cone raw s.  This preserves the existing
                # cone-collapse assignment while replacing the heuristic front
                # gate by an aperture fraction at the track start.
                s_center_endpoint = s_eff

                # Geometry for implicit derivative of the cone-root equation
                # G(s,P)=u(P)-s-rho(P)cot(theta_c(s))=0.
                yx = p_locations[i, 0] - start_pos[0]
                yy = p_locations[i, 1] - start_pos[1]
                yz = p_locations[i, 2] - start_pos[2]
                u_line = yx * tdx + yy * tdy + yz * tdz
                bx = yx - u_line * tdx
                by = yy - u_line * tdy
                bz = yz - u_line * tdz
                rho = math.sqrt(bx * bx + by * by + bz * bz)

                Eep = E_b[i]
                gamma_ep = 1.0 + Eep / mu_mass
                beta2_ep = 1.0 - 1.0 / max(gamma_ep * gamma_ep, 1.0e-30)
                if beta2_ep < 0.0:
                    beta2_ep = 0.0
                beta_ep = math.sqrt(beta2_ep)

                if n_water * beta_ep > 1.0:
                    cos_tc_ep = 1.0 / (n_water * beta_ep)
                    if cos_tc_ep > 1.0:
                        cos_tc_ep = 1.0
                    elif cos_tc_ep < -1.0:
                        cos_tc_ep = -1.0
                    sin2_tc_ep = 1.0 - cos_tc_ep * cos_tc_ep
                    if sin2_tc_ep < 1.0e-18:
                        sin2_tc_ep = 1.0e-18
                    sin_tc_ep = math.sqrt(sin2_tc_ep)
                    cot_ep = cos_tc_ep / sin_tc_ep
                    dEdx_ep = _interp_dedx_scalar_for_endpoint_numba(Eep, dedx_E_grid, dedx_grid)
                    dc_ds_ep = dEdx_ep / (n_water * mu_mass * beta_ep ** 3 * gamma_ep ** 3)
                    d_cot_ds_ep = dc_ds_ep / (sin2_tc_ep * sin_tc_ep)
                else:
                    cot_ep = 0.0
                    d_cot_ds_ep = 0.0

                if rho > 1.0e-12 and math.isfinite(cot_ep):
                    denom_root = 1.0 + rho * d_cot_ds_ep
                    if (not math.isfinite(denom_root)) or denom_root <= 1.0e-12:
                        denom_root = 1.0
                    gx = (tdx - cot_ep * bx / rho) / denom_root
                    gy = (tdy - cot_ep * by / rho) / denom_root
                    gz = (tdz - cot_ep * bz / rho) / denom_root
                else:
                    gx = tdx
                    gy = tdy
                    gz = tdz

                # Project ds/dP into the PMT aperture plane.
                nx = direction_zs[i, 0]
                ny = direction_zs[i, 1]
                nz = direction_zs[i, 2]
                n2 = nx * nx + ny * ny + nz * nz
                if n2 > 1.0e-18:
                    gd = (gx * nx + gy * ny + gz * nz) / n2
                    gpx = gx - gd * nx
                    gpy = gy - gd * ny
                    gpz = gz - gd * nz
                else:
                    gpx = gx
                    gpy = gy
                    gpz = gz
                h_endpoint = endpoint_aperture_radius * math.sqrt(gpx * gpx + gpy * gpy + gpz * gpz)

                w_endpoint, s_mean_endpoint = _endpoint_rootdisk_weight_and_mean_s_numba(
                    s_center_endpoint, track_length, h_endpoint, endpoint_scope
                )
                scale = scale * w_endpoint
                if w_endpoint > 0.0:
                    valid_s = True
                    if endpoint_mode == 2:
                        s_eff = s_mean_endpoint
                    else:
                        if s_eff < 0.0:
                            s_eff = 0.0
                        elif s_eff > track_length:
                            s_eff = track_length
                else:
                    valid_s = False
                    scale = 0.0
                    s_eff = 0.0
        else:
            # Legacy heuristic front/start gate.
            s_gate = s_eff
            if s_gate < pmt_radius:
                scale = scale * (s_gate + pmt_radius) / two_pr

            if s_gate >= -pmt_radius:
                valid_s = True
            else:
                valid_s = False
                scale = 0.0
                s_eff = 0.0

        if need_times:
            t_primary[i] = np.nan  # set below if active; matches t_primary array shape

        # geometry at the effective emission point
        ex = start_pos[0] + s_eff * tdx
        ey = start_pos[1] + s_eff * tdy
        ez = start_pos[2] + s_eff * tdz
        dx = p_locations[i, 0] - ex
        dy = p_locations[i, 1] - ey
        dz = p_locations[i, 2] - ez
        r = math.sqrt(dx * dx + dy * dy + dz * dz) + 0.01

        cost = -(dx * direction_zs[i, 0] + dy * direction_zs[i, 1] + dz * direction_zs[i, 2]) / r
        # Visibility gate.  Hard (cost_soft<=0): contribution is on for cost>0,
        # off otherwise -> a discontinuity at cost=0 (power_law(0+)~0.125).
        # Soft (cost_soft>0): smoothstep ramp over cost in [0, cost_soft] so the
        # PMT turns on/off continuously, removing the per-PMT cliff that roughens
        # the FCN and can trap optimizer seeds.
        if cost_soft > 0.0 and cost_soft_centered != 0:
            # SMOOTH-NLL (opt-in): visibility ramp CENTERED on cost=0, over
            # [-cost_soft, +cost_soft].  The hard gate at the collapsed
            # emission point switches a tube's full primary contribution in
            # one step when its facing cosine crosses zero (verified on a
            # production event: cost(s_b) sign flip within 0.02 mm of L
            # toggled 0 <-> 0.59 PE).  Unlike the one-sided ramp below, this
            # form gives HALF the light at exact grazing -- the finite PMT
            # face is half-illuminated by a source on its horizon -- so tubes
            # at small positive cost are not starved of expected light (the
            # failure mode of the one-sided ramp in the delta path).
            valid_cost = math.isfinite(cost) and (cost > -cost_soft)
            if cost >= cost_soft:
                vis = 1.0
            elif cost <= -cost_soft:
                vis = 0.0
            else:
                u_s = (cost + cost_soft) / (2.0 * cost_soft)
                vis = u_s * u_s * (3.0 - 2.0 * u_s)
        else:
            valid_cost = math.isfinite(cost) and (cost > 0.0)
            if cost_soft > 0.0:
                if cost <= 0.0:
                    vis = 0.0
                elif cost >= cost_soft:
                    vis = 1.0
                else:
                    u_s = cost / cost_soft
                    vis = u_s * u_s * (3.0 - 2.0 * u_s)
            else:
                vis = 1.0
        if not valid_cost:
            scale = 0.0

        # primary time uses s_eff and r regardless of activity (matches vectorized
        # t_primary which is computed for all PMTs); the caller only consumes it
        # where mu>0, but we fill it to preserve identical values where finite.
        if need_times:
            t_primary[i] = starting_time + s_eff / v + r * n_water / c_light

        active = (scale > 0.0) and valid_cost
        if not active:
            continue

        cds_visibility = 1.0
        if cds_occlusion_enabled != 0:
            cds_visibility = annular_cylinder_aperture_visibility_numba(
                ex, ey, ez,
                p_locations[i, 0], p_locations[i, 1], p_locations[i, 2],
                direction_zs[i, 0], direction_zs[i, 1], direction_zs[i, 2],
                cds_axis_x, cds_axis_z,
                cds_inner_radius, cds_outer_radius, cds_y_min, cds_y_max,
                cds_aperture_radius,
            )
            if cds_visibility <= 0.0:
                continue

        # angular response power_law(cost)
        x = cost
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        xn = x ** pl_nfit
        pwr = (pl_y0 + (pl_yinf - pl_y0) * (xn / (xn + pl_x50n))) / pl_max

        # primary_ngeo_falloff_raw(E_b, r) * ngeo_norm
        E = E_b[i]
        gamma = 1.0 + E / mu_mass
        g2 = gamma * gamma
        beta2 = 1.0 - 1.0 / g2
        if beta2 < 0.0:
            beta2 = 0.0
        beta = math.sqrt(beta2)
        corr = 0.0
        if n_water * beta > 1.0:
            cos_tc = 1.0 / (n_water * beta)
            sin2_tc = 1.0 - cos_tc * cos_tc
            # inline dedx interp (np.interp with edge clamp)
            if E <= dedx_E_grid[0]:
                dEdx = dedx_grid[0]
            elif E >= dedx_E_grid[nde - 1]:
                dEdx = dedx_grid[nde - 1]
            else:
                lo = 0
                hi = nde
                while lo < hi:
                    mid = (lo + hi) // 2
                    if dedx_E_grid[mid] < E:
                        lo = mid + 1
                    else:
                        hi = mid
                # lo is first index with grid >= E
                x0 = dedx_E_grid[lo - 1]
                x1 = dedx_E_grid[lo]
                y0 = dedx_grid[lo - 1]
                y1 = dedx_grid[lo]
                dEdx = y0 + (E - x0) / (x1 - x0) * (y1 - y0)
            dc_ds = dEdx / (n_water * mu_mass * beta ** 3 * gamma ** 3)
            r_eff = math.sqrt(r * r + ngeo_a * ngeo_a)
            denom = r_eff * sin2_tc + r_eff * r_eff * dc_ds
            if math.isfinite(denom) and denom > 0.0:
                corr = (1.0 / denom) * ngeo_norm * pwr

        # rel-mPMT efficiency (uniform grid, code-based)
        rel = 1.0
        code = mpmt_codes[i]
        if code >= 0 and code < rel_eff_table.shape[0] and n_grid >= 2:
            xc = cost
            if xc < 0.0:
                xc = 0.0
            elif xc > 1.0:
                xc = 1.0
            xc *= (n_grid - 1)
            i0 = int(math.floor(xc))
            if i0 < 0:
                i0 = 0
            elif i0 > n_grid - 2:
                i0 = n_grid - 2
            t = xc - i0
            rel = rel_eff_table[code, i0] + t * (rel_eff_table[code, i0 + 1] - rel_eff_table[code, i0])

        mu_primary[i] = intensity * corr * scale * rel * vis * cds_visibility

    return mu_primary, t_primary


def _get_pmt_radius_cached(wcd):
    """Cache the PMT radius lookup from the WCD object."""
    key = id(wcd)
    val = _PMT_RADIUS_CACHE.get(key)
    if val is None:
        val = float(wcd.mpmts[0].pmts[0].get_properties("design")["size"]) / 2.0
        _PMT_RADIUS_CACHE[key] = val
    return val

def _finite_disk_solid_angle_rel(r_mm, pmt_radius_mm=37.0, ref_r_mm=1000.0):
    """
    Relative face-on solid angle of a circular PMT disk.

    Exact face-on solid angle:

        Omega(r) = 2*pi * (1 - r / sqrt(r^2 + a^2))

    where:
        r = source-to-PMT distance
        a = PMT radius

    This function returns Omega(r) / Omega(ref_r_mm), so the factor is
    dimensionless and equals 1 at the reference distance.

    This replaces the arbitrary (R0/r)^p distance law with the finite-aperture
    point-source collection law.  It does NOT include PMT angular response,
    because that is already handled by pwr_corr.
    """
    r = np.asarray(r_mm, dtype=np.float64)
    r_safe = np.maximum(r, 1e-9)

    a = float(pmt_radius_mm)
    R0 = float(ref_r_mm)

    if a <= 0.0:
        # Far-field point-aperture limit.
        return (R0 / r_safe) ** 2

    omega_shape = 1.0 - r_safe / np.sqrt(r_safe * r_safe + a * a)
    omega_ref = 1.0 - R0 / np.sqrt(R0 * R0 + a * a)

    if (not np.isfinite(omega_ref)) or omega_ref <= 0.0:
        return (R0 / r_safe) ** 2

    out = omega_shape / omega_ref
    out[~np.isfinite(out)] = 0.0
    out[out < 0.0] = 0.0

    return out

def _primary_ngeo_raw_static(E_MeV, r_mm, *, n=1.344, mu_mass=105.658, pmt_radius_mm=37.0):
    """Static version of primary_ngeo_falloff_raw used for cached normalization."""
    E = np.asarray(E_MeV, dtype=np.float64)
    r = np.asarray(r_mm, dtype=np.float64)

    gamma = 1.0 + E / mu_mass
    beta2 = np.clip(1.0 - 1.0 / np.maximum(gamma, 1e-30) ** 2, 0.0, None)
    beta = np.sqrt(beta2)

    above = n * beta > 1.0

    cos_tc = np.zeros_like(E, dtype=np.float64)
    cos_tc[above] = 1.0 / (n * beta[above])

    sin2_tc = np.zeros_like(E, dtype=np.float64)
    sin2_tc[above] = 1.0 - cos_tc[above] ** 2

    dEdx = _interp_muon_dedx_positive(E)
    dc_ds = np.zeros_like(E, dtype=np.float64)
    dc_ds[above] = dEdx[above] / (
        n * mu_mass * beta[above] ** 3 * gamma[above] ** 3
    )

    r_eff = np.sqrt(r * r + pmt_radius_mm * pmt_radius_mm)
    denom = r_eff * sin2_tc + r_eff * r_eff * dc_ds

    out = np.zeros(np.broadcast(E, r).shape, dtype=np.float64)
    good = above & np.isfinite(denom) & (denom > 0.0)
    out[good] = 1.0 / denom[good]
    return out

def _electron_cherenkov_threshold_MeV(n=1.344, m_e=0.51099895):
    """
    Electron kinetic-energy Cherenkov threshold in MeV.

        beta_thr = 1/n
        gamma_thr = 1 / sqrt(1 - beta_thr^2)
        T_thr = m_e (gamma_thr - 1)
    """
    beta_thr = 1.0 / float(n)
    gamma_thr = 1.0 / np.sqrt(1.0 - beta_thr * beta_thr)
    return float(m_e * (gamma_thr - 1.0))


def _electron_range_cm_water_approx(T_MeV):
    """
    Approximate electron CSDA range in water [cm].

    This is the same empirical relation already used in your secondary-electron
    model.  For water, density ~= 1 g/cm^3, so g/cm^2 and cm are numerically
    equivalent to good approximation.

    T_MeV can be scalar or array.
    """
    T = np.asarray(T_MeV, dtype=np.float64)
    T_safe = np.maximum(T, 1e-12)

    R_cm = np.zeros_like(T_safe, dtype=np.float64)

    low = T_safe <= 2.5
    R_cm[low] = 0.412 * T_safe[low] ** (
        1.265 - 0.0954 * np.log(T_safe[low])
    )
    R_cm[~low] = 0.530 * T_safe[~low] - 0.106

    R_cm = np.maximum(R_cm, 0.0)
    return R_cm


def _electron_stopping_power_MeV_per_cm_water_approx(T_MeV):
    """
    Effective electron stopping power in water [MeV/cm].

    Uses the derivative of the same range relation:

        R = R(T)
        dR/dT = cm / MeV
        S(T) = dT/dR = 1 / (dR/dT)

    This is not yet as good as an ESTAR table, but it is already better than
    treating the full electron range as if it emitted at the initial T0.
    """
    T = np.asarray(T_MeV, dtype=np.float64)
    T_safe = np.maximum(T, 1e-8)

    # Relative finite-difference step.  Keep it small but not catastrophically
    # small near threshold.
    dT = np.maximum(1e-4 * T_safe, 1e-6)

    T_lo = np.maximum(T_safe - dT, 1e-8)
    T_hi = T_safe + dT

    R_lo = _electron_range_cm_water_approx(T_lo)
    R_hi = _electron_range_cm_water_approx(T_hi)

    dR_dT = (R_hi - R_lo) / np.maximum(T_hi - T_lo, 1e-30)

    # Avoid division by zero or negative numerical artifacts.
    dR_dT = np.where(np.isfinite(dR_dT) & (dR_dT > 0.0), dR_dT, np.nan)

    S = 1.0 / dR_dT
    S = np.where(np.isfinite(S) & (S > 0.0), S, 1e30)

    return S


def _electron_frank_tamm_factor(T_MeV, n=1.344, m_e=0.51099895):
    """
    Electron Frank--Tamm factor:

        F(T) = 1 - 1 / (n^2 beta(T)^2)

    Returns zero below Cherenkov threshold.
    """
    T = np.asarray(T_MeV, dtype=np.float64)

    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / np.maximum(gamma, 1e-30) ** 2
    beta2 = np.clip(beta2, 0.0, None)

    ft = 1.0 - 1.0 / (float(n) ** 2 * np.maximum(beta2, 1e-30))
    ft = np.where(beta2 * float(n) ** 2 > 1.0, np.maximum(ft, 0.0), 0.0)

    return ft


def _electron_cherenkov_cos_alpha(T_MeV, n=1.344, m_e=0.51099895):
    """
    cos(alpha_e) for an electron of kinetic energy T_MeV.

        cos(alpha_e) = 1 / (n beta_e)

    Values below threshold are returned as nan.
    """
    T = np.asarray(T_MeV, dtype=np.float64)

    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / np.maximum(gamma, 1e-30) ** 2
    beta2 = np.clip(beta2, 0.0, None)
    beta = np.sqrt(beta2)

    above = float(n) * beta > 1.0

    cos_alpha = np.full_like(T, np.nan, dtype=np.float64)
    cos_alpha[above] = 1.0 / (float(n) * beta[above])
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

    return cos_alpha

def _get_particle_stopping_power_table(particle="muon"):
    """
    Build and cache a smooth positive stopping-power table for a primary particle.

    Returns
    -------
    E_grid : ndarray
        Kinetic energies [MeV].
    dEdx_grid : ndarray
        Positive stopping power, -dE/ds [MeV/mm].

    Notes
    -----
    The range table stores total above-threshold range versus initial kinetic
    energy. Differentiating range with respect to kinetic energy gives dR/dE, so

        -dE/ds = 1 / (dR/dE).

    This is the same range-table information used by the collapse solver, just
    rearranged into the derivative needed by the analytic N_geo formula.
    """
    pname = canonical_particle_name(particle)
    cached = _PARTICLE_STOPPING_POWER_CACHE.get(pname)
    if cached is not None:
        return cached

    overall_distances = np.asarray(_get_tables(pname)[2], dtype=np.float64)  # mm
    energy_rows = _get_tables(pname)[3]

    # Initial kinetic energy for each stopping range.
    E0 = _initial_energies_from_table_rows(energy_rows)

    order = np.argsort(E0)
    E0 = E0[order]
    ranges = overall_distances[order]

    keep = np.isfinite(E0) & np.isfinite(ranges)
    E0 = E0[keep]
    ranges = ranges[keep]

    unique_E, unique_idx = np.unique(E0, return_index=True)
    E0 = unique_E
    ranges = ranges[unique_idx]

    dR_dE = np.gradient(ranges, E0)  # mm / MeV
    dEdx = 1.0 / np.maximum(dR_dE, 1e-30)  # MeV / mm

    good = np.isfinite(E0) & np.isfinite(dEdx) & (dEdx > 0.0)
    cached = (E0[good], dEdx[good])
    _PARTICLE_STOPPING_POWER_CACHE[pname] = cached
    return cached


def _interp_particle_dedx_positive(E_MeV, particle="muon"):
    """
    Interpolate positive primary-particle stopping power -dE/ds [MeV/mm].
    """
    E_grid, dEdx_grid = _get_particle_stopping_power_table(particle)
    E = np.asarray(E_MeV, dtype=np.float64)
    return np.interp(E, E_grid, dEdx_grid, left=dEdx_grid[0], right=dEdx_grid[-1])


# Backward-compatible names.
def _get_muon_stopping_power_table():
    return _get_particle_stopping_power_table("muon")


def _interp_muon_dedx_positive(E_MeV):
    return _interp_particle_dedx_positive(E_MeV, "muon")


_REFINED_ANALYTIC_DELTA_CACHE = {}
_REFINED_ANALYTIC_DELTA_CACHE_INFO = {}
_REFINED_ANALYTIC_DELTA_TABLE_KIND = "licketyfit_refined_analytic_delta_v1"
_REFINED_ANALYTIC_DELTA_TABLE_SCHEMA_VERSION = 1
_REFINED_ANALYTIC_DELTA_BUILD_CONFIG = {
    "K_min": 0.0,
    "K_max": 1000.0,
    "n_K": 180,
    "n_u": 240,
    "n_T0": 120,
    "n_T_slow": 60,
    "n_psi": 12,
}


def _refined_delta_float_tag(value):
    """Filesystem-safe, deterministic tag for a floating-point table parameter."""
    return (f"{float(value):.9g}".replace("-", "m").replace(".", "p").replace("+", "p"))


def _refined_delta_default_filename(particle, n, projectile_mass):
    pname = canonical_particle_name(particle)
    return (
        f"refined_analytic_delta_table_{pname}"
        f"_n{_refined_delta_float_tag(n)}"
        f"_m{_refined_delta_float_tag(projectile_mass)}"
        f"_v{_REFINED_ANALYTIC_DELTA_TABLE_SCHEMA_VERSION}.npz"
    )


def _refined_delta_table_dirs():
    """Search directories for the exact precomputed refined-delta table."""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    dirs = []

    cache_dir = os.environ.get("EMITTER_REFINED_DELTA_CACHE_DIR", "").strip()
    if cache_dir:
        dirs.append(cache_dir)

    for env_name in ("LF_TABLE_DIR", "LF_MULTIPARTICLES_TABLE_DIR"):
        raw = os.environ.get(env_name, "")
        if raw:
            dirs.extend([x for x in raw.split(os.pathsep) if x])

    dirs.extend([
        os.path.join(project_root, "tables"),
        os.path.join(module_dir, "tables"),
        os.path.join(os.getcwd(), "tables"),
        module_dir,
        os.getcwd(),
    ])

    out = []
    seen = set()
    for directory in dirs:
        directory = os.path.abspath(os.path.expanduser(directory))
        if directory not in seen:
            seen.add(directory)
            out.append(directory)
    return out


def _refined_delta_candidate_paths(particle, n, projectile_mass):
    explicit = os.environ.get("EMITTER_REFINED_DELTA_TABLE_PATH", "").strip()
    if explicit:
        return [os.path.abspath(os.path.expanduser(explicit))]
    filename = _refined_delta_default_filename(particle, n, projectile_mass)
    return [os.path.join(directory, filename) for directory in _refined_delta_table_dirs()]


def _validate_refined_delta_disk_table(payload, *, particle, n, projectile_mass):
    """Validate that a disk table is the exact production refined-delta table."""
    required = {"K_grid", "u_grid", "table", "table_kind", "schema_version"}
    if not required.issubset(set(payload.files)):
        return None

    table_kind = str(np.asarray(payload["table_kind"]).item())
    schema_version = int(np.asarray(payload["schema_version"]).item())
    if table_kind != _REFINED_ANALYTIC_DELTA_TABLE_KIND:
        return None
    if schema_version != _REFINED_ANALYTIC_DELTA_TABLE_SCHEMA_VERSION:
        return None

    if "particle" in payload.files:
        disk_particle = canonical_particle_name(str(np.asarray(payload["particle"]).item()))
        if disk_particle != canonical_particle_name(particle):
            return None
    if "n_water" in payload.files and not np.isclose(
        float(np.asarray(payload["n_water"]).item()), float(n), rtol=0.0, atol=1.0e-12
    ):
        return None
    if "projectile_mass_mev" in payload.files and not np.isclose(
        float(np.asarray(payload["projectile_mass_mev"]).item()),
        float(projectile_mass), rtol=0.0, atol=1.0e-9,
    ):
        return None

    for key, expected in _REFINED_ANALYTIC_DELTA_BUILD_CONFIG.items():
        if key in payload.files:
            value = np.asarray(payload[key]).item()
            if isinstance(expected, int):
                if int(value) != int(expected):
                    return None
            elif not np.isclose(float(value), float(expected), rtol=0.0, atol=1.0e-12):
                return None

    K_grid = np.ascontiguousarray(np.asarray(payload["K_grid"], dtype=np.float64))
    u_grid = np.ascontiguousarray(np.asarray(payload["u_grid"], dtype=np.float64))
    table = np.ascontiguousarray(np.asarray(payload["table"], dtype=np.float64))

    if K_grid.ndim != 1 or u_grid.ndim != 1 or table.ndim != 2:
        return None
    if table.shape != (K_grid.size, u_grid.size):
        return None
    if K_grid.size != _REFINED_ANALYTIC_DELTA_BUILD_CONFIG["n_K"]:
        return None
    if u_grid.size != _REFINED_ANALYTIC_DELTA_BUILD_CONFIG["n_u"]:
        return None
    if not (np.all(np.isfinite(K_grid)) and np.all(np.isfinite(u_grid)) and np.all(np.isfinite(table))):
        return None
    if np.any(np.diff(K_grid) <= 0.0) or np.any(np.diff(u_grid) <= 0.0):
        return None
    if np.any(table < 0.0):
        return None

    return K_grid, u_grid, table


def _save_refined_delta_disk_table(path, *, particle, n, projectile_mass, cached):
    """Atomically save the exact table. Failure to save never changes the physics."""
    path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.npz"
    K_grid, u_grid, table = cached
    try:
        np.savez(
            tmp,
            K_grid=np.asarray(K_grid, dtype=np.float64),
            u_grid=np.asarray(u_grid, dtype=np.float64),
            table=np.asarray(table, dtype=np.float64),
            table_kind=np.asarray(_REFINED_ANALYTIC_DELTA_TABLE_KIND),
            schema_version=np.asarray(_REFINED_ANALYTIC_DELTA_TABLE_SCHEMA_VERSION, dtype=np.int64),
            particle=np.asarray(canonical_particle_name(particle)),
            n_water=np.asarray(float(n), dtype=np.float64),
            projectile_mass_mev=np.asarray(float(projectile_mass), dtype=np.float64),
            **{
                key: np.asarray(value, dtype=np.int64 if isinstance(value, int) else np.float64)
                for key, value in _REFINED_ANALYTIC_DELTA_BUILD_CONFIG.items()
            },
        )
        os.replace(tmp, path)
        return path
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return None


def get_refined_analytic_delta_cache_info(particle=None, n=1.344, projectile_mass=None):
    """Return how the process obtained its refined-delta table."""
    pname = canonical_particle_name(particle)
    if projectile_mass is None:
        projectile_mass = particle_mass_mev(pname)
    key = (pname, float(n), float(projectile_mass))
    return dict(_REFINED_ANALYTIC_DELTA_CACHE_INFO.get(key, {}))

@njit(cache=True)
def _electron_cherenkov_threshold_numba(n, m_e):
    beta_thr = 1.0 / n
    gamma_thr = 1.0 / math.sqrt(1.0 - beta_thr * beta_thr)
    return m_e * (gamma_thr - 1.0)


@njit(cache=True)
def _electron_range_cm_water_approx_scalar_numba(T):
    T_safe = T
    if T_safe < 1e-12:
        T_safe = 1e-12

    if T_safe <= 2.5:
        R = 0.412 * T_safe ** (1.265 - 0.0954 * math.log(T_safe))
    else:
        R = 0.530 * T_safe - 0.106

    if R < 0.0 or not math.isfinite(R):
        return 0.0
    return R


@njit(cache=True)
def _electron_stopping_power_MeV_per_cm_scalar_numba(T):
    """
    Effective electron stopping power in water [MeV/cm].

    This is the scalar compiled equivalent of
    _electron_stopping_power_MeV_per_cm_water_approx().  It uses the derivative
    of the same empirical range relation already used in the model:

        S(T) = dT/dR = 1 / (dR/dT).
    """
    T_safe = T
    if T_safe < 1e-8:
        T_safe = 1e-8

    dT = 1e-4 * T_safe
    if dT < 1e-6:
        dT = 1e-6

    T_lo = T_safe - dT
    if T_lo < 1e-8:
        T_lo = 1e-8

    T_hi = T_safe + dT

    R_lo = _electron_range_cm_water_approx_scalar_numba(T_lo)
    R_hi = _electron_range_cm_water_approx_scalar_numba(T_hi)

    dR_dT = (R_hi - R_lo) / (T_hi - T_lo)

    if (not math.isfinite(dR_dT)) or dR_dT <= 0.0:
        return 1e30

    S = 1.0 / dR_dT

    if (not math.isfinite(S)) or S <= 0.0:
        return 1e30

    return S


@njit(cache=True)
def _electron_frank_tamm_factor_scalar_numba(T, n, m_e):
    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / (gamma * gamma)

    if beta2 <= 0.0:
        return 0.0

    if beta2 * n * n <= 1.0:
        return 0.0

    ft = 1.0 - 1.0 / (n * n * beta2)

    if ft < 0.0 or not math.isfinite(ft):
        return 0.0

    return ft


@njit(cache=True)
def _electron_cherenkov_cos_alpha_scalar_numba(T, n, m_e):
    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / (gamma * gamma)

    if beta2 <= 0.0:
        return np.nan

    beta = math.sqrt(beta2)

    if n * beta <= 1.0:
        return np.nan

    c = 1.0 / (n * beta)

    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0

    return c


@njit(cache=True)
def _electron_recoil_cos_theta_numba(K_mu, T_e0, m_mu, m_e):
    E_mu = K_mu + m_mu
    p_mu2 = E_mu * E_mu - m_mu * m_mu
    if p_mu2 <= 0.0:
        return 1.0

    p_e2 = T_e0 * (T_e0 + 2.0 * m_e)
    if p_e2 <= 0.0:
        return 1.0

    p_mu = math.sqrt(p_mu2)
    p_e = math.sqrt(p_e2)

    c = T_e0 * (E_mu + m_e) / (p_mu * p_e)

    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0

    return c


@njit(cache=True)
def _add_arcsine_kernel_to_row_numba(row, u_centers, du, A, B, weight):
    """
    Add weight * p(u) to row, where

        u = A + B cos(phi),

    and p(u) is the bin-averaged arcsine density.

    This avoids allocating a full kernel array for every T0/T step.
    """
    n_u = u_centers.size

    if weight <= 0.0:
        return

    if (not math.isfinite(A)) or (not math.isfinite(B)) or (not math.isfinite(weight)):
        return

    if B < 0.0:
        B = -B

    u_min_edge = u_centers[0] - 0.5 * du
    u_max_edge = u_centers[n_u - 1] + 0.5 * du

    # Collapsed-cone limit: p(u) = delta(u - A).
    if B <= 1e-12:
        if A < u_min_edge or A > u_max_edge:
            return

        idx = int(math.floor((A - u_min_edge) / du))
        if idx < 0:
            idx = 0
        elif idx >= n_u:
            idx = n_u - 1

        row[idx] += weight / du
        return

    support_lo = A - B
    support_hi = A + B

    if support_hi < u_min_edge or support_lo > u_max_edge:
        return

    j0 = int(math.floor((support_lo - u_min_edge) / du))
    j1 = int(math.floor((support_hi - u_min_edge) / du))

    if j0 < 0:
        j0 = 0
    if j1 >= n_u:
        j1 = n_u - 1

    inv_pi = 1.0 / math.pi

    for j in range(j0, j1 + 1):
        u = u_centers[j]

        lo = (u - 0.5 * du - A) / B
        hi = (u + 0.5 * du - A) / B

        if hi < -1.0 or lo > 1.0:
            continue

        if lo < -1.0:
            lo = -1.0
        elif lo > 1.0:
            lo = 1.0

        if hi < -1.0:
            hi = -1.0
        elif hi > 1.0:
            hi = 1.0

        prob = (math.asin(hi) - math.asin(lo)) * inv_pi

        if prob > 0.0 and math.isfinite(prob):
            row[j] += weight * prob / du


@njit(cache=True)
def _inv_lambda_tr_electron_numba(T, m_e):
    """
    First-principles transport inverse-mean-free-path 1/lambda_tr(T) [1/cm]
    for an electron of kinetic energy T (MeV) in water.

    This is the screened-Rutherford elastic cross section with the relativistic
    (Mott) spin correction, screened by the Thomas-Fermi atomic potential, summed
    over the atoms of H2O.  No empirical normalization or tuning factor is used;
    every quantity is a physical constant or a kinematic function of T.

        1/lambda_tr = N_mol * sum_atoms nat * Z(Z+1) * (r_e^2 / (4 gamma^2 beta^4))
                                      * 8 pi * (L0 - beta^2 L2),

    with screening angle parameter

        eta_screen = 0.25 * chi^2 * (1.13 + 3.76 (alpha Z / beta)^2),
        chi        = hbar c / (p a_TF),   a_TF = 0.885 a0 Z^(-1/3),

    and the transport integrals

        L0 = ln(1 + 1/eta) - 1/(1+eta),
        L2 = 1 + eta - 2 eta ln((1+eta)/eta) - eta^2/(1+eta)   (clipped at 0).

    The l=1 (transport) coefficient returned here sets the mean cosine of the
    accumulated multiple scattering through <cos> = exp(-Integral 1/lambda_tr ds),
    i.e. the small-angle multiple-scattering (diffusion) limit, which is the
    correct regime for the sub-MeV knock-on electrons that carry the light.
    """
    if T <= 0.0:
        return 0.0

    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    if beta2 <= 0.0:
        return 0.0
    beta = math.sqrt(beta2)
    p = math.sqrt(T * (T + 2.0 * m_e))  # MeV/c

    r_e_cm = 2.8179403262e-13
    N_A = 6.02214076e23
    N_mol = N_A / 18.01528  # water molecules per cm^3 at rho = 1 g/cm^3

    alpha = 1.0 / 137.036
    hbarc = 197.3269804     # MeV fm
    a0_fm = 0.529177e5      # Bohr radius in fm

    inv = 0.0
    for ia in range(2):
        if ia == 0:
            Z = 8.0
            nat = 1.0       # one O per H2O
        else:
            Z = 1.0
            nat = 2.0       # two H per H2O

        a_TF = 0.885 * a0_fm * Z ** (-1.0 / 3.0)
        chi = hbarc / (p * a_TF)
        eta_s = 0.25 * chi * chi * (1.13 + 3.76 * (alpha * Z / beta) ** 2)
        if eta_s <= 0.0:
            continue

        L0 = math.log(1.0 + 1.0 / eta_s) - 1.0 / (1.0 + eta_s)
        L2 = (
            1.0
            + eta_s
            - 2.0 * eta_s * math.log((1.0 + eta_s) / eta_s)
            - eta_s * eta_s / (1.0 + eta_s)
        )
        if L2 < 0.0:
            L2 = 0.0

        sig_tr = (
            nat
            * Z * (Z + 1.0)
            * r_e_cm * r_e_cm / (4.0 * gamma * gamma * beta2 * beta2)
            * 8.0 * math.pi
            * (L0 - beta2 * L2)
        )
        if math.isfinite(sig_tr) and sig_tr > 0.0:
            inv += sig_tr

    out = N_mol * inv
    if (not math.isfinite(out)) or out < 0.0:
        return 0.0
    return out


@njit(cache=True)
def _fill_refined_analytic_delta_table_numba(
    K_grid,
    u_centers,
    table,
    n,
    n_T0,
    n_T_slow,
    projectile_mass,
    n_psi,
):
    """
    Compiled version of the slowing-down secondary-electron table builder.

    Same physics as the slow Python version:

        dS_delta/du(K,u)
        =
        integral dT0 dN/dT0
        integral dT [F_e(T)/S_e(T)] p(u | T0,T),

    but without Python loops or repeated kernel-array allocations.
    """
    m_e = 0.51099895
    m_mu = projectile_mass

    r_e_cm = 2.8179403262e-13
    N_A = 6.02214076e23
    rho_water = 1.0

    n_e = rho_water * N_A * (10.0 / 18.01528)
    ft_sat_mu = 1.0 - 1.0 / (n * n)

    T_thr = _electron_cherenkov_threshold_numba(n, m_e)

    n_K = K_grid.size
    n_u = u_centers.size
    du = u_centers[1] - u_centers[0]

    T0_min = T_thr * 1.0001
    log_T0_min = math.log(T0_min)

    for iK in range(n_K):
        K_mu = K_grid[iK]

        gamma_mu = 1.0 + K_mu / m_mu
        beta2_mu = 1.0 - 1.0 / (gamma_mu * gamma_mu)

        if beta2_mu <= 0.0:
            continue

        T_max = (
            2.0 * m_e * beta2_mu * gamma_mu * gamma_mu
            / (1.0 + 2.0 * gamma_mu * m_e / m_mu + (m_e / m_mu) * (m_e / m_mu))
        )

        if (not math.isfinite(T_max)) or T_max <= T_thr:
            continue

        log_T0_max = math.log(T_max)
        dlog_T0 = (log_T0_max - log_T0_min) / n_T0

        prefactor = n_e * 2.0 * math.pi * r_e_cm * r_e_cm * m_e / beta2_mu

        for iT0 in range(n_T0):
            T0_lo = math.exp(log_T0_min + iT0 * dlog_T0)
            T0_hi = math.exp(log_T0_min + (iT0 + 1) * dlog_T0)

            T0 = math.sqrt(T0_lo * T0_hi)
            dT0 = T0_hi - T0_lo

            if T0 <= T_thr:
                continue

            dN_dx_dT0 = (
                prefactor
                * (1.0 / (T0 * T0))
                * (1.0 - beta2_mu * T0 / T_max)
            )

            if dN_dx_dT0 <= 0.0 or not math.isfinite(dN_dx_dT0):
                continue

            prod_weight = dN_dx_dT0 * dT0

            cos_te = _electron_recoil_cos_theta_numba(K_mu, T0, m_mu, m_e)
            sin2_te = 1.0 - cos_te * cos_te
            if sin2_te < 0.0:
                sin2_te = 0.0
            sin_te = math.sqrt(sin2_te)

            if T0 <= T_thr * 1.0002:
                continue

            log_T_min = math.log(T_thr * 1.0001)
            log_T_max = math.log(T0)
            dlog_T = (log_T_max - log_T_min) / n_T_slow

            # Accumulate the electron multiple-scattering transport optical depth
            #   eta(T) = Integral_T^T0 (1/lambda_tr) dT'/S_e(T')
            # while the electron slows from its birth energy T0 down to each
            # emission energy T.  The mean cosine of the accumulated scattering
            # relative to the birth (recoil) direction is exp(-eta(T)) in the
            # small-angle multiple-scattering (diffusion) limit, which is the
            # correct regime for these sub-MeV knock-on electrons.  We therefore
            # march iT from high T (just born, eta ~ 0) down to low T (most
            # scattered).  Light emitted at energy T is Cherenkov-coned about the
            # scattered electron direction, which is itself distributed about the
            # recoil axis with mean cosine exp(-eta).  There is no tuning factor
            # anywhere in this chain.
            two_pi = 2.0 * math.pi
            inv_n_psi = 1.0 / n_psi
            run_eta = 0.0

            for iT in range(n_T_slow - 1, -1, -1):
                T_lo = math.exp(log_T_min + iT * dlog_T)
                T_hi = math.exp(log_T_min + (iT + 1) * dlog_T)

                T = math.sqrt(T_lo * T_hi)
                dT = T_hi - T_lo

                S = _electron_stopping_power_MeV_per_cm_scalar_numba(T)
                if S <= 0.0 or not math.isfinite(S):
                    continue

                # Path length traversed in this energy bin [cm] and the transport
                # optical depth accrued over it (mid-bin value of the cumulative).
                ds_path = dT / S
                inv_ltr = _inv_lambda_tr_electron_numba(T, m_e)
                eta_mid = run_eta + 0.5 * inv_ltr * ds_path
                run_eta += inv_ltr * ds_path

                ft = _electron_frank_tamm_factor_scalar_numba(T, n, m_e)
                if ft <= 0.0:
                    continue

                # dT/S is path length in cm.
                dY_equiv = ft * dT / S / ft_sat_mu
                if dY_equiv <= 0.0 or not math.isfinite(dY_equiv):
                    continue

                cos_alpha = _electron_cherenkov_cos_alpha_scalar_numba(T, n, m_e)
                if not math.isfinite(cos_alpha):
                    continue

                sin2_alpha = 1.0 - cos_alpha * cos_alpha
                if sin2_alpha < 0.0:
                    sin2_alpha = 0.0
                sin_alpha = math.sqrt(sin2_alpha)

                # Mean cosine of the accumulated multiple scattering at this
                # emission energy (diffusion limit).
                cos_ms = math.exp(-eta_mid)
                if cos_ms > 1.0:
                    cos_ms = 1.0
                elif cos_ms < -1.0:
                    cos_ms = -1.0
                sin2_ms = 1.0 - cos_ms * cos_ms
                if sin2_ms < 0.0:
                    sin2_ms = 0.0
                sin_ms = math.sqrt(sin2_ms)

                weight = prod_weight * dY_equiv
                w_psi = weight * inv_n_psi

                # Compose recoil (cos_te) (x) multiple scattering (cos_ms) by
                # averaging over the scattering azimuth psi, then add the
                # Cherenkov cone (cos_alpha) about the resulting electron
                # direction through the azimuthal arcsine kernel.  This is the
                # full recoil (x) MS (x) cone angular composition; with cos_ms = 1
                # it reduces exactly to the previous recoil (x) cone model.
                for ip in range(n_psi):
                    psi = (ip + 0.5) * two_pi * inv_n_psi
                    cpsi = math.cos(psi)

                    cos_e = cos_te * cos_ms + sin_te * sin_ms * cpsi
                    if cos_e > 1.0:
                        cos_e = 1.0
                    elif cos_e < -1.0:
                        cos_e = -1.0
                    sin2_e = 1.0 - cos_e * cos_e
                    if sin2_e < 0.0:
                        sin2_e = 0.0
                    sin_e = math.sqrt(sin2_e)

                    A = cos_e * cos_alpha
                    B = sin_e * sin_alpha

                    _add_arcsine_kernel_to_row_numba(
                        table[iK],
                        u_centers,
                        du,
                        A,
                        B,
                        w_psi,
                    )

    # Safety cleanup.
    for iK in range(n_K):
        for iu in range(n_u):
            val = table[iK, iu]
            if (not math.isfinite(val)) or val < 0.0:
                table[iK, iu] = 0.0


def _build_refined_analytic_delta_table(
    n=1.344,
    K_min=0.0,
    K_max=1000.0,
    n_K=180,
    n_u=240,
    n_T0=120,
    n_T_slow=60,
    n_T=None,
    projectile_mass=105.6583755,
    n_psi=12,
):
    """
    Fast compiled builder for dS_delta/du(K_mu, u).

    Physics is the same as the slow Python slowing-down version:

        dS_delta/du
        =
        integral dT0 [dN_delta/(ds dT0)]
        integral dT [F_e(T)/S_e(T)] p(u | T0,T).

    The speedup comes from:
      - no Python loop over K/T0/T/u,
      - no repeated kernel-array allocation,
      - direct bin accumulation into table[iK, iu].
    """
    if n_T is not None:
        n_T0 = int(n_T)

    K_grid = np.linspace(K_min, K_max, int(n_K), dtype=np.float64)

    # Full physical range of the photon direction cosine relative to the muon.
    # The knock-on light is no longer forward-only: once first-principles
    # multiple scattering is included, a physical ~14% of the light is emitted
    # into the backward hemisphere (u < 0), so the table must cover [-1, 1].
    u_centers = np.linspace(
        -1.0 + 1.0 / int(n_u),
        1.0 - 1.0 / int(n_u),
        int(n_u),
        dtype=np.float64,
    )

    table = np.zeros((int(n_K), int(n_u)), dtype=np.float64)

    _fill_refined_analytic_delta_table_numba(
        K_grid,
        u_centers,
        table,
        float(n),
        int(n_T0),
        int(n_T_slow),
        float(projectile_mass),
        int(n_psi),
    )

    return K_grid, u_centers, table




def get_refined_analytic_delta_cache(
    n=1.344,
    projectile_mass=105.6583755,
    particle="muon",
    *,
    force_rebuild=False,
):
    """Return the exact refined analytic secondary-electron table.

    Production behavior is now:

      1. reuse the process-local array cache;
      2. load a validated float64 table from ``tables/`` (or the explicit
         ``EMITTER_REFINED_DELTA_TABLE_PATH``);
      3. if no validated table exists, build the unchanged analytic table once
         and atomically persist it for later jobs.

    The table contents and interpolation used by the FCN are unchanged.  This
    removes the expensive repeated K/T0/T/u construction from new batch-driver
    processes without introducing an empirical or WCSim-derived input.
    """
    global _REFINED_ANALYTIC_DELTA_CACHE, _REFINED_ANALYTIC_DELTA_CACHE_INFO

    pname = canonical_particle_name(particle)
    key = (pname, float(n), float(projectile_mass))
    if not force_rebuild:
        cached = _REFINED_ANALYTIC_DELTA_CACHE.get(key)
        if cached is not None:
            return cached

    use_disk = _env_bool_switch(
        "EMITTER_REFINED_DELTA_DISK_CACHE", True
    )
    candidate_paths = _refined_delta_candidate_paths(pname, n, projectile_mass)

    if use_disk and not force_rebuild:
        for path in candidate_paths:
            if not os.path.isfile(path):
                continue
            try:
                with np.load(path, allow_pickle=False) as payload:
                    cached = _validate_refined_delta_disk_table(
                        payload,
                        particle=pname,
                        n=n,
                        projectile_mass=projectile_mass,
                    )
            except Exception:
                cached = None
            if cached is not None:
                _REFINED_ANALYTIC_DELTA_CACHE[key] = cached
                _REFINED_ANALYTIC_DELTA_CACHE_INFO[key] = {
                    "source": "disk",
                    "path": path,
                    "table_kind": _REFINED_ANALYTIC_DELTA_TABLE_KIND,
                    "schema_version": _REFINED_ANALYTIC_DELTA_TABLE_SCHEMA_VERSION,
                }
                return cached

    cached = _build_refined_analytic_delta_table(
        n=n,
        projectile_mass=float(projectile_mass),
        **_REFINED_ANALYTIC_DELTA_BUILD_CONFIG,
    )
    cached = tuple(np.ascontiguousarray(x, dtype=np.float64) for x in cached)
    _REFINED_ANALYTIC_DELTA_CACHE[key] = cached

    saved_path = None
    if use_disk:
        for path in candidate_paths:
            parent = os.path.dirname(path)
            try:
                if os.path.isdir(parent):
                    writable = os.access(parent, os.W_OK)
                else:
                    ancestor = parent
                    while ancestor and not os.path.exists(ancestor):
                        next_ancestor = os.path.dirname(ancestor)
                        if next_ancestor == ancestor:
                            break
                        ancestor = next_ancestor
                    writable = bool(ancestor and os.access(ancestor, os.W_OK))
                if not writable:
                    continue
                saved_path = _save_refined_delta_disk_table(
                    path,
                    particle=pname,
                    n=n,
                    projectile_mass=projectile_mass,
                    cached=cached,
                )
                if saved_path is not None:
                    break
            except Exception:
                continue

    _REFINED_ANALYTIC_DELTA_CACHE_INFO[key] = {
        "source": "built_and_saved" if saved_path is not None else "built_in_memory",
        "path": saved_path,
        "table_kind": _REFINED_ANALYTIC_DELTA_TABLE_KIND,
        "schema_version": _REFINED_ANALYTIC_DELTA_TABLE_SCHEMA_VERSION,
    }
    return cached



def _highland_projected_mcs_sigma_rad(K_MeV, mass_MeV, thickness_mm, *, radiation_length_mm=360.8, charge_number=1.0):
    """Projected RMS multiple-scattering angle from the Highland formula.

    Parameters
    ----------
    K_MeV : float
        Kinetic energy of the primary particle at the local emission region.
    mass_MeV : float
        Particle rest mass.
    thickness_mm : float
        Local material thickness over which the unresolved scattering is accumulated.
    radiation_length_mm : float
        Radiation length of the medium. For water, X0 ~= 36.08 cm = 360.8 mm.
    charge_number : float
        Particle charge magnitude in units of e. The current supported primaries are singly charged.

    Notes
    -----
    The returned theta0 is the one-dimensional projected-plane RMS angle.
    This is intentionally a *local* width, not the total scattering over the whole track:
    long-wavelength deflections are absorbed by the fitted vertex/direction, while unresolved
    deflections across the finite PMT-cone acceptance broaden the local cone edge.
    """
    K = float(K_MeV)
    m = float(mass_MeV)
    x = float(thickness_mm)
    X0 = float(radiation_length_mm)
    z = abs(float(charge_number))
    if K <= 0.0 or m <= 0.0 or x <= 0.0 or X0 <= 0.0 or z <= 0.0:
        return 0.0
    p2 = K * (K + 2.0 * m)
    if p2 <= 0.0:
        return 0.0
    p = math.sqrt(p2)
    beta = p / (K + m)
    xx0 = max(x / X0, 1e-12)
    corr = 1.0 + 0.038 * math.log(xx0)
    # For extremely tiny x/X0 the logarithmic correction can go negative; in the
    # physical operating range here (PMT-scale cm in water) it remains positive,
    # but guard anyway.
    corr = max(corr, 0.0)
    return float((13.6 * z / max(beta * p, 1e-30)) * math.sqrt(xx0) * corr)


def _cherenkov_weight_from_energy(K_MeV, mass_MeV, n_water):
    """Frank-Tamm angular yield factor, up to a common constant.

    This is used only as a weight for averaging local MCS widths along the
    visible track.  It is zero below threshold and proportional to
    sin^2(theta_c) above threshold.
    """
    K = np.asarray(K_MeV, dtype=np.float64)
    m = float(mass_MeV)
    n = float(n_water)
    gamma = 1.0 + K / max(m, 1e-30)
    beta2 = 1.0 - 1.0 / np.maximum(gamma * gamma, 1e-30)
    beta2 = np.maximum(beta2, 0.0)
    denom = n * n * beta2
    w = np.zeros_like(K, dtype=np.float64)
    mask = denom > 1.0
    w[mask] = 1.0 - 1.0 / denom[mask]
    return w


_DELTA_E_CACHE = None

class Emitter:
    """
    Optimized Cherenkov emitter model used by the fitter.

    The public fit-facing API is preserved, but the hot methods avoid:
      - pickle-based copying
      - debug prints in the fit loop
      - repeated temporary allocations when not needed
    """

    def __init__(
        self,
        starting_time,
        start_coord,
        direction,
        beta,
        length,
        intensity,
        particle="muon",
        track_end_mode="threshold",
        fixed_initial_KE=None,
    ):
        if not isinstance(starting_time, (int, float)):
            raise TypeError("starting_time must be a number")
        if not (
            isinstance(start_coord, tuple)
            and len(start_coord) == 3
            and all(isinstance(c, (int, float)) for c in start_coord)
        ):
            raise TypeError("start_coord must be a tuple of three numbers")
        if not (
            isinstance(direction, tuple)
            and len(direction) == 3
            and all(isinstance(c, (int, float)) for c in direction)
        ):
            raise TypeError("direction must be a tuple of three numbers")
        if not isinstance(beta, (int, float)) or not (0 < beta < 1):
            raise ValueError("beta must be a number between 0 and 1")
        if not isinstance(length, (int, float)) or length <= 0:
            raise ValueError("length must be a positive number")
        if not isinstance(intensity, (int, float)) or intensity <= 0:
            raise ValueError("intensity must be a positive number")

        self.starting_time = float(starting_time)
        self.start_coord = tuple(float(c) for c in start_coord)
        self.direction = tuple(float(c) for c in direction)
        self.length = float(length)
        self.intensity = float(intensity)
        # Charge shape is the backward-compatible default.  A calibrated
        # absolute-light analysis may instead set ``global_scale`` together
        # with a positive detector scale measured independently of the fitted
        # event.  The scale is never inferred inside an event likelihood.
        self.charge_normalization_mode = "event_mean"
        self.global_charge_scale = None
        self.event_mean_contamination_model = _env_str_switch(
            "EMITTER_EVENT_MEAN_CONTAMINATION_MODEL",
            DEFAULT_EVENT_MEAN_CONTAMINATION_MODEL,
        ).strip().lower().replace("-", "_")
        if self.event_mean_contamination_model not in {
            "off", "none", "uniform_profile"
        }:
            raise ValueError(
                "EMITTER_EVENT_MEAN_CONTAMINATION_MODEL must be off or "
                "uniform_profile"
            )
        self.event_mean_contamination_max_fraction = _env_float_switch(
            "EMITTER_EVENT_MEAN_CONTAMINATION_MAX_FRACTION",
            DEFAULT_EVENT_MEAN_CONTAMINATION_MAX_FRACTION,
        )
        if not 0.0 <= self.event_mean_contamination_max_fraction < 1.0:
            raise ValueError(
                "event-mean contamination maximum must lie in [0,1)"
            )
        self._last_event_mean_contamination_fraction = 0.0

        self.particle_name = canonical_particle_name(particle)
        set_active_particle(self.particle_name)
        self.particle_mass = particle_mass_mev(self.particle_name)
        self.mu_mass = self.particle_mass  # Backward-compatible alias used by older helper names.

        # ------------------------------------------------------------------
        # Primary-track endpoint model.
        #
        # threshold:
        #     Old behavior.  The fitted parameter ``length`` is interpreted as
        #     the dE/dx-only range to Cherenkov threshold, so it determines the
        #     initial kinetic energy through the range table.
        #
        # abrupt:
        #     The fitted parameter ``length`` is interpreted as the visible
        #     primary-Cherenkov length before an abrupt cutoff/interaction.
        #     The initial kinetic energy is fixed independently by
        #     ``fixed_initial_KE``.  Internally, ``range_to_threshold_mm`` still
        #     comes from the dE/dx table and is used to evaluate K(s), theta_c(s),
        #     beta, and dE/dx along the visible part of the track.
        #
        # straggled_threshold:
        #     ``fixed_initial_KE`` determines the mean CSDA energy profile while
        #     ``fixed_realized_range_mm`` is the event's independently realised
        #     stopping range.  The mean loss coordinate is mapped continuously
        #     onto that physical arc length.  This is the non-centred
        #     stopping-range model; it is not an abrupt interaction.
        # ------------------------------------------------------------------
        self.track_end_mode = "threshold"
        self.fixed_initial_KE = None
        self.fixed_realized_range_mm = None
        self.range_to_threshold_mm = float(length)
        self.realized_range_to_threshold_mm = float(length)
        self.stopping_range_coordinate_scale = 1.0
        self.last_visible_length_exceeds_range = False

        self.n = _env_float_switch(
            "EMITTER_WATER_PHASE_INDEX", DEFAULT_WATER_PHASE_INDEX
        )
        if not math.isfinite(self.n) or self.n <= 1.0:
            raise ValueError("EMITTER_WATER_PHASE_INDEX must be finite and greater than one")
        self.c = 299.792458  # mm/ns

        self.beta = float(beta)
        self.v = self.beta * self.c
        self.cos_tq = None
        self.cot_tq = None
        self.interp_E_init = None

        # Per-instance caches for quantities that are repeatedly needed in the
        # Minuit FCN hot loop.
        self._energy_main_idx = None
        self._energy_dist_row = None
        self._energy_energy_row = None
        self._last_geometry_cache_key = None
        self._last_mpmt_type_codes = None
        self._delta_src_grid_cache = {}
        # Exact, process-local source-grid cache.  A bounded cache prevents
        # long production jobs from accumulating thousands of one-off Minuit
        # length hypotheses in every worker.
        self.delta_source_cache_max_entries = _env_int_switch(
            "EMITTER_DELTA_SOURCE_CACHE_MAX", 256
        )
        self._timing_active_cache_key = None
        self._timing_active_cache = None

        # Production fits do not need to copy large diagnostic arrays on every
        # FCN call.  Set this True only for residual/component debugging.
        self.store_expected_component_diagnostics = False

        # Global numerical policy switches resolved from the top-level defaults.
        # These were previously implicit getattr(..., True) fallbacks scattered
        # through the file; setting them here makes the active policy explicit.
        self.smooth_tables = _env_bool_switch("EMITTER_SMOOTH_TABLES", DEFAULT_SMOOTH_TABLES)
        self.use_fused_primary = _env_bool_switch("EMITTER_USE_FUSED_PRIMARY", DEFAULT_USE_FUSED_PRIMARY)

        # Fixed detector-obstacle geometry.  This affects direct source-to-PMT
        # transport only; it is disabled unless the detector/driver explicitly
        # identifies the WCTE CDS configuration.
        self.enable_wcte_cds_occlusion = _env_bool_switch(
            "EMITTER_ENABLE_WCTE_CDS_OCCLUSION",
            DEFAULT_ENABLE_WCTE_CDS_OCCLUSION,
        )
        self.wcte_cds_axis_x_mm = _env_float_switch(
            "EMITTER_WCTE_CDS_AXIS_X_MM", DEFAULT_WCTE_CDS_AXIS_X_MM
        )
        self.wcte_cds_axis_z_mm = _env_float_switch(
            "EMITTER_WCTE_CDS_AXIS_Z_MM", DEFAULT_WCTE_CDS_AXIS_Z_MM
        )
        self.wcte_cds_inner_radius_mm = _env_float_switch(
            "EMITTER_WCTE_CDS_INNER_RADIUS_MM", DEFAULT_WCTE_CDS_INNER_RADIUS_MM
        )
        self.wcte_cds_outer_radius_mm = _env_float_switch(
            "EMITTER_WCTE_CDS_OUTER_RADIUS_MM", DEFAULT_WCTE_CDS_OUTER_RADIUS_MM
        )
        self.wcte_cds_y_min_mm = _env_float_switch(
            "EMITTER_WCTE_CDS_Y_MIN_MM", DEFAULT_WCTE_CDS_Y_MIN_MM
        )
        self.wcte_cds_y_max_mm = _env_float_switch(
            "EMITTER_WCTE_CDS_Y_MAX_MM", DEFAULT_WCTE_CDS_Y_MAX_MM
        )
        self.wcte_cds_pmt_aperture_radius_mm = _env_float_switch(
            "EMITTER_WCTE_CDS_PMT_APERTURE_RADIUS_MM",
            DEFAULT_WCTE_CDS_PMT_APERTURE_RADIUS_MM,
        )
        self.enable_wcte_cds_specular_reflection = _env_bool_switch(
            "EMITTER_ENABLE_WCTE_CDS_SPECULAR_REFLECTION",
            DEFAULT_ENABLE_WCTE_CDS_SPECULAR_REFLECTION,
        )
        self.wcte_cds_specular_reflectivity = _env_float_switch(
            "EMITTER_WCTE_CDS_SPECULAR_REFLECTIVITY",
            DEFAULT_WCTE_CDS_SPECULAR_REFLECTIVITY,
        )
        self.wcte_cds_specular_phi_bins = _env_int_switch(
            "EMITTER_WCTE_CDS_SPECULAR_PHI_BINS",
            DEFAULT_WCTE_CDS_SPECULAR_PHI_BINS,
        )
        self.wcte_cds_specular_y_bins = _env_int_switch(
            "EMITTER_WCTE_CDS_SPECULAR_Y_BINS",
            DEFAULT_WCTE_CDS_SPECULAR_Y_BINS,
        )
        self.wcte_cds_specular_timing_bins = _env_int_switch(
            "EMITTER_WCTE_CDS_SPECULAR_TIMING_BINS",
            DEFAULT_WCTE_CDS_SPECULAR_TIMING_BINS,
        )

        self.primary_subthreshold_range_mm = particle_subthreshold_range_mm(
            self.particle_name
        )
        # Backward-compatible attribute name used by the secondary-electron
        # implementation.  Its value is now particle-aware.
        self.muon_subthreshold_range_mm = self.primary_subthreshold_range_mm

        # ---- resolved top-level physics switches (see USER-FACING DEFAULTS above) ----
        self.enable_delta_e = _env_bool_switch("EMITTER_ENABLE_DELTA_E", DEFAULT_ENABLE_DELTA_E, "ENABLE_DELTA_E")
        self.delta_e_scale = _env_float_switch("EMITTER_DELTA_E_SCALE", DEFAULT_DELTA_E_SCALE)

        # Number of source bins along the above-threshold, Cherenkov-visible muon path.
        self.n_delta_steps = 4

        # Force the below-threshold tail to be sampled separately.
        # This prevents the 110 mm tail from disappearing when n_delta_steps is small.
        self.delta_e_tail_step_mm = 20.0
        self.delta_e_tail_min_steps = 3

        # ------------------------------------------------------------------
        # Secondary-electron timing model.
        #
        # The observed times in the current batch driver are charge-weighted
        # mean hit times per PMT, so the expected time should also be a
        # PE-weighted mixture of primary-muon light and secondary-electron
        # light.  The secondary-electron emission time is approximated as the
        # time for the muon to reach the secondary source point plus the photon
        # time of flight from that source point to the PMT.  Any explicit
        # electron-propagation delay can be added with delta_e_time_offset_ns.
        # ------------------------------------------------------------------
        self.use_delta_e_timing = False
        self.delta_e_time_offset_ns = 0

        # Secondary electrons are treated as localized light sources.
        # Their geometric collection factor is therefore projected PMT area / r^2,
        # rather than the primary muon cone/line-source-like 1/r factor.
        self.delta_e_point_source_geometry = True

        # ------------------------------------------------------------------
        # Analytic primary-muon falloff replacement for n_from_E_r.
        #
        # This replaces the WCSim-derived empirical falloff surface with
        #
        #   N_geo(E,r) = C / [r_eff sin^2(theta_c)
        #                    + r_eff^2 d cos(theta_c)/ds]
        #
        # where r_eff = sqrt(r^2 + a^2), a ~= 37 mm is the PMT radius, and
        # d cos(theta_c)/ds is computed from the muon stopping power table.
        #
        # This term is only the geometric/cone-density falloff.  The
        # Frank-Tamm yield factor, PMT angular response, and relative mPMT
        # efficiency remain separate, as in the old model.
        # ------------------------------------------------------------------
        self.use_analytic_primary_ngeo = True
        self.primary_ngeo_pmt_radius_mm = 60.0
        self.primary_ngeo_ref_energy_MeV = 304.0
        self.primary_ngeo_ref_r_mm = 1000.0

        # The original reference energy (304 MeV) is safely above the muon
        # Cherenkov threshold, but it is below the proton threshold in water.
        # If the normalization reference is below threshold,
        # primary_ngeo_falloff_raw() is zero and the old code silently used
        # norm = 1.0, suppressing primary proton light by orders of magnitude.
        # Keep the old muon/pion behavior, but automatically move the reference
        # energy above threshold for heavy particles such as protons.
        self.primary_ngeo_auto_ref_above_threshold = True
        self.primary_ngeo_ref_threshold_factor = 2.0
        self.primary_ngeo_ref_threshold_margin_MeV = 25.0

        # Apply relative mPMT efficiency using each secondary source point's
        # actual incidence angle, not the primary-muon emission angle.
        self.delta_e_apply_mpmt_eff_by_source = True


        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # FUNCTIONAL SECONDARY-ELECTRON MODEL
        # ------------------------------------------------------------------
        # Best data-matching secondary-electron option from the analytic tests.
        #
        # When enabled, the secondary-electron angular/yield model uses a
        # physically motivated dS_delta/du(K_mu, u) table built from:
        #   knock-on electron production,
        #   electron range * Frank-Tamm light yield,
        #   recoil angle + electron Cherenkov cone kinematics,
        #   bin-integrated forward-endpoint handling,
        #   modest electron-transport / multiple-scattering broadening.
        #
        # It replaces the old factorized model:
        #   S_delta(K_mu) * external p(u | K_mu).
        # ------------------------------------------------------------------
        self.use_refined_analytic_delta_e = True


        # ------------------------------------------------------------------
        # Secondary-electron distance falloff.
        #
        # The refined secondary-electron table already contains the energy/yield
        # and angular distribution dS_delta/du(K_mu, u).  The remaining geometric
        # distance factor should be the finite-disk solid-angle falloff of the PMT,
        # normalized to a reference distance.
        #
        # Since pwr_corr already represents the angular detection efficiency of
        # the PMT relative to a face-on PMT at the same distance, do NOT add an
        # extra cos(eta) projected-area factor here.
        # ------------------------------------------------------------------
        self.delta_e_use_finite_disk_solid_angle = True
        self.delta_e_distance_ref_r_mm = 1000.0
        # Production default: use the physical in-situ reflector-mouth radius.
        # Earlier diagnostic work used 60 mm as an inflated effective disk.
        # The WCSim in-situ reflector with the correct ~31.7 degree side angle has
        # an upper inner radius of 45.0 mm, so that is the analytic default.
        # Use EMITTER_DELTA_E_DISTANCE_PMT_RADIUS_MM=60.0 only for legacy A/B tests.
        self.delta_e_distance_pmt_radius_mm = _env_float_switch(
            "EMITTER_DELTA_E_DISTANCE_PMT_RADIUS_MM",
            DEFAULT_DELTA_E_DISTANCE_PMT_RADIUS_MM,
            "DELTA_E_DISTANCE_PMT_RADIUS_MM",
        )

        # Kept only for backward-compatible fallback when finite-disk distance
        # falloff is explicitly disabled.
        self.delta_e_distance_power = 2

        self.delta_e_source_k_power = 0 #-2.5 #-0.5
        self.delta_e_source_k_ref_MeV = 100.0
        self.delta_e_source_k_floor_MeV = 25.0


        # Overall secondary-electron strength for the refined analytic table.
        # After fixing the electron-energy dT integration and the forward-u
        # endpoint handling, the best low+high joint value was about 3.4.
        self.analytic_delta_scale = _env_float_switch("EMITTER_ANALYTIC_DELTA_SCALE", DEFAULT_ANALYTIC_DELTA_SCALE)
        
        # SMOOTH-NLL CANDIDATE CONFIG (pending 1000-event A/B validation):
        # small expected-charge floor.  Bounds the -obs*ln(mu) term for tubes
        # with observed charge but ~zero expected charge, whose numerical dust
        # otherwise dominates NLL curvature once the gate steps are fixed
        # (production event 1: 1.28 -> 0.062 max|d2| at 1 mm).  This changes
        # the penalty for unexplained observed charge, i.e. likelihood policy:
        # verify fitted-L vs truth in the batch A/B before trusting.
        # Set to 0 for the historical likelihood.
        self.charge_floor_pe = _env_float_switch("EMITTER_CHARGE_FLOOR_PE", DEFAULT_CHARGE_FLOOR_PE)
        
        # ==================================================================
        # DIAGNOSTIC / EXPERIMENTAL KNOBS -- off by default.
        # These exist for controlled A/B tests and should not be considered
        # production model choices unless a dedicated validation promotes them.
        # ==================================================================
        # Diagnostic only:
        # Artificially shift secondary-electron emission points downstream
        # along the muon direction. This tests whether collapsing electron
        # range back to the muon point is causing the central-light deficit.
        self.delta_e_test_forward_shift_mm = 0
        
        # ------------------------------------------------------------------
        # Diagnostic only:
        # Reweight the refined secondary-electron angular table toward high u,
        # while optionally preserving each K-row's total integrated yield.
        #
        # u_power = 0 -> original table
        # u_power = 1 -> multiply by u
        # u_power = 2 -> multiply by u^2
        # u_power = 4 -> strongly forward-weighted
        # ------------------------------------------------------------------
        self.delta_e_debug_u_power = 0
        self.delta_e_debug_u_min = 0.0
        self.delta_e_debug_preserve_yield = True
        
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # FUNCTIONAL PRIMARY CONE / EDGE MODEL
        # ------------------------------------------------------------------
        # Primary-muon cone softening.
        #
        # The collapse solver compares the PMT direction alpha(s) to the
        # shrinking Cherenkov angle theta_c(s).  If there is no exact crossing,
        # this width gives a soft nonzero contribution based on the closest
        # angular mismatch:
        #
        #   exp[-0.5 * (min|alpha - theta_c| / sigma)^2]
        #
        # Units: radians.  Set to 0.0 to recover hard zero behavior for
        # no-crossing PMTs.
        # ------------------------------------------------------------------
        # Paired with the scatter-model calibration: 0.009 rad approximates the
        # multiple-scattering ring-edge smearing at 300 MeV.  Was 0.005 pre-scatter;
        # note this widens the primary soft cone even when scatter is disabled.
        self.primary_soft_cone_sigma_rad = _env_float_switch("EMITTER_PRIMARY_SOFT_CONE_SIGMA_RAD", DEFAULT_PRIMARY_SOFT_CONE_SIGMA_RAD)

        # ------------------------------------------------------------------
        # Physics-based unresolved multiple-scattering broadening of the primary
        # cone edge.
        #
        # A straight-track fit can absorb long-wavelength MCS by moving the
        # fitted vertex and direction, but it cannot represent the local
        # curvature across the finite source region that feeds one PMT.  The
        # longitudinal size of that unresolved source patch is estimated from
        # the PMT photocathode radius projected onto the track by the Cherenkov
        # cone,
        #
        #     ell_local ~= r_pmt / tan(theta_c(K0)),
        #
        # and Highland's formula gives the projected RMS angular diffusion over
        # that local thickness in water.  This is then added in quadrature to
        # the finite-geometry cone width.  Enabled by default for standard
        # WCSim with primary multiple scattering.  Disable explicitly for
        # no-scattering control MC.
        # ------------------------------------------------------------------
        # MCS on/off state is controlled by the one master switch at the top of
        # this file (or EMITTER_ENABLE_MCS from a public launcher). Cosmic mode
        # routes that master to the cosmic continuation and leaves this ordinary
        # contained-track path inactive, so the two implementations cannot run
        # together.
        # Evidence (July 2026 session, sandbox scale): applying sigma_eff=0.0148
        # as a wider ONE-SIDED soft cone (a) shifts the length minimum short via
        # edge-band (40-45 deg) over-fill, and (b) softens the per-event NLL(L)
        # surface so per-event fitted lengths scatter (std ~90 mm) and skew far
        # short of even the ensemble minimum (median-ensemble gap -48 mm at
        # 0.0148 vs -16 mm at 0.009; per-event free-(z,L) medians: -1 mm at
        # sigma=0.009 vs -40 mm at 0.0148 on the same events).  Ensemble-level
        # validations (fixed-z profiles, stacked fits) do NOT see this and can
        # look unbiased while per-event fits underestimate severely.
        # Production MCS policy. ``DEFAULT_ENABLE_PRIMARY_MCS`` is the source
        # default for the sole master switch. In Fermi--Eyges process mode the ordinary likelihood
        # retains the validated sharp mean cone; MCS enters through the
        # correlated post-fit process update. Only the explicitly selected
        # legacy model uses deterministic local Highland cone broadening.
        self.primary_mcs_model = _env_str_switch(
            "EMITTER_PRIMARY_MCS_MODEL", DEFAULT_PRIMARY_MCS_MODEL
        ).strip().lower().replace("-", "_")
        self.enable_primary_mcs = bool(
            _resolved_mcs_enabled() and not _is_cosmic_fit_mode()
        )
        self.enable_primary_mcs_smearing = bool(
            self.enable_primary_mcs
            and self.primary_mcs_model in {
                "legacy", "cone_broadening", "local_highland"
            }
        )
        self.primary_mcs_process_modes_per_plane = _env_int_switch(
            "EMITTER_PRIMARY_MCS_PROCESS_MODES_PER_PLANE", 4
        )
        self.primary_mcs_process_grid_points = _env_int_switch(
            "EMITTER_PRIMARY_MCS_PROCESS_GRID_POINTS", 41
        )
        # This flag is intentionally false in every ordinary FCN.  It is set on
        # a temporary Emitter copy for the single post-fit process evaluation.
        self.compute_primary_mcs_process_jacobian = False
        self._last_mcs_charge_jacobian = None
        self._last_mcs_basis_explained_fraction = None

        # Edge model for the primary cone boundary:
        #   "legacy": full brightness inside + one-sided Gaussian tail outside
        #             (the historical behavior; NOT charge-conserving, and the
        #             mechanism behind the MCS short-length overshoot).
        #   "erf":    symmetric charge-conserving smeared edge,
        #             w = Phi(-f_lo/sigma)*Phi(f_hi/sigma)  (UNVALIDATED beyond
        #             small sandbox checks -- see session notes).
        self.primary_edge_model = _env_str_switch("EMITTER_PRIMARY_EDGE_MODEL", DEFAULT_PRIMARY_EDGE_MODEL).lower()
        # ------------------------------------------------------------------
        # Primary endpoint/front-gate model.
        #
        #   "root_overlap_weight_only" (default): physics-based, fast start-edge
        #       correction.  For PMTs whose collapsed cone root is near the fitted
        #       track start, multiply the primary PE expectation by the fraction
        #       of a circular reflector/PMT aperture whose local cone-root
        #       coordinate lies inside the physical track interval.  The source
        #       coordinate itself remains the collapsed root, clipped into [0,L].
        #
        #   "legacy": original heuristic front gate, using a linear ramp over
        #       pmt_radius = PMT_size/2 + 20 mm.
        #
        #   "root_overlap_weight_mean": diagnostic only; uses the mean in-track
        #       source coordinate over the aperture overlap.  Not the default.
        #
        # Environment switches:
        #   EMITTER_PRIMARY_ENDPOINT_MODEL=legacy | root_overlap_weight_only | root_overlap_weight_mean
        #   EMITTER_PRIMARY_ENDPOINT_APERTURE_RADIUS_MM=45.0
        #   EMITTER_PRIMARY_ENDPOINT_SCOPE=start | end | both
        #
        # Speed: implemented inside the numba fused primary kernel; no Python
        # per-PMT loop is added in the normal fit path.
        # ------------------------------------------------------------------
        self.primary_endpoint_model = _env_str_switch("EMITTER_PRIMARY_ENDPOINT_MODEL", DEFAULT_PRIMARY_ENDPOINT_MODEL, "PRIMARY_ENDPOINT_MODEL").lower()
        self.primary_endpoint_aperture_radius_mm = _env_float_switch("EMITTER_PRIMARY_ENDPOINT_APERTURE_RADIUS_MM", DEFAULT_PRIMARY_ENDPOINT_APERTURE_RADIUS_MM)
        self.primary_endpoint_scope = _env_str_switch("EMITTER_PRIMARY_ENDPOINT_SCOPE", DEFAULT_PRIMARY_ENDPOINT_SCOPE).lower()
        self.primary_mcs_radiation_length_mm = 360.8  # water X0 = 36.08 cm
        self.primary_mcs_charge_number = 1.0
        self.primary_mcs_use_local_pmt_patch = True
        self.primary_mcs_min_thickness_mm = 1.0
        self.primary_mcs_max_thickness_fraction = 0.25
        self.primary_mcs_sigma_cap_rad = 0.025
        # Energy treatment for the local MCS width.  The historical model used
        # the initial KE only ("initial").  The physics-motivated variants below
        # keep the same Highland formula but evaluate/average it over the actual
        # K(s) curve from the range table, so the effective unresolved MCS changes
        # with particle energy and track length without any per-energy tuning.
        #   initial           : historical K0 local-patch model
        #   midpoint          : local patch at s = 0.5 L_visible
        #   emission_weighted : RMS-average local patch over the visible track,
        #                       weighted by the Frank-Tamm sin^2(theta_c) factor
        self.primary_mcs_energy_mode = _env_str_switch("EMITTER_PRIMARY_MCS_ENERGY_MODE", DEFAULT_PRIMARY_MCS_ENERGY_MODE).lower()
        self.primary_mcs_energy_samples = _env_int_switch("EMITTER_PRIMARY_MCS_ENERGY_SAMPLES", DEFAULT_PRIMARY_MCS_ENERGY_SAMPLES)

        # ------------------------------------------------------------------
        # Fast analytic one-bounce WCTE blacksheet reflection.
        #
        # Default behavior matches the validated conditional timing study:
        # reflected PRIMARY light participates in timing eligibility and in the
        # primary/delta/reflection expected-time mixture, but it is not added to
        # the already-normalized charge marginal unless REFLECTION_IN_CHARGE=1.
        # No event-truth light distribution or fitted reflection scale is used.
        # ------------------------------------------------------------------
        self.enable_blacksheet_reflection = _env_bool_switch(
            "EMITTER_ENABLE_REFLECTION",
            DEFAULT_ENABLE_BLACKSHEET_REFLECTION,
            "ENABLE_REFLECTION",
        )
        self.reflection_in_charge = _env_bool_switch(
            "EMITTER_REFLECTION_IN_CHARGE",
            DEFAULT_REFLECTION_IN_CHARGE,
        )
        self.reflection_charge_policy = _env_str_switch(
            "EMITTER_REFLECTION_CHARGE_POLICY",
            DEFAULT_REFLECTION_CHARGE_POLICY,
        ).strip().lower().replace("-", "_")
        if self.reflection_charge_policy not in {
            "unconditional",
            "prompt_group_gated",
        }:
            raise ValueError(
                "EMITTER_REFLECTION_CHARGE_POLICY must be unconditional or "
                "prompt_group_gated"
            )
        self.reflection_bsrff = _env_float_switch(
            "EMITTER_REFLECTION_BSRFF",
            DEFAULT_REFLECTION_BSRFF,
        )
        self.reflection_pmt_aperture_radius_mm = _env_float_switch(
            "EMITTER_REFLECTION_PMT_APERTURE_RADIUS_MM",
            DEFAULT_REFLECTION_PMT_APERTURE_RADIUS_MM,
        )
        self.reflection_tangent_bins = _env_int_switch(
            "EMITTER_REFLECTION_TANGENT_BINS",
            DEFAULT_REFLECTION_TANGENT_BINS,
        )
        self.reflection_y_bins = _env_int_switch(
            "EMITTER_REFLECTION_Y_BINS",
            DEFAULT_REFLECTION_Y_BINS,
        )
        self.reflection_cap_radial_bins = _env_int_switch(
            "EMITTER_REFLECTION_CAP_RADIAL_BINS",
            DEFAULT_REFLECTION_CAP_RADIAL_BINS,
        )
        self.use_first_arrival_timing = _env_bool_switch(
            "EMITTER_USE_FIRST_ARRIVAL_TIMING",
            DEFAULT_USE_FIRST_ARRIVAL_TIMING,
        )
        self.reflection_first_arrival_nodes = _env_int_switch(
            "EMITTER_REFLECTION_FIRST_ARRIVAL_NODES",
            DEFAULT_REFLECTION_FIRST_ARRIVAL_NODES,
        )

        # ------------------------------------------------------------------
        # Single-scatter Rayleigh light model (rayleigh_scatter_model.pdf).
        # Default is set by DEFAULT_ENABLE_RAYLEIGH at the top of this file and can
        # be overridden with EMITTER_ENABLE_RAYLEIGH.  lambda_s is the single
        # physical constant (effective scattering length; ~30-45 m on the
        # noE/noScat sample; MUST be recalibrated with deltas on / production).
        # ------------------------------------------------------------------
        # CALIBRATION NOTE (physics-rooted, but preliminary; this legacy
        # midpoint-Rayleigh branch is not the default molecular transport):
        # lambda_s = 120 m is set by matching the wide-angle (theta>42 deg) charge
        # fraction of the DIGITIZED WCSim mu- 300 MeV sample WITH secondary electrons
        # on (data 4.59%, model primary+delta+scatter 4.56%); 120 m sits in the
        # physical Rayleigh range for water (~100-200 m).  IMPORTANT decomposition:
        # the delta-ray model already supplies most wide-angle light (4.19% of the
        # 4.59%); Rayleigh scatter fills the small residual (~0.4%) but still
        # contributes ~21 mm to removing the length bias.  The paired soft-cone
        # sigma=0.009 rad (multiple-scattering ring-edge smearing) removes most of
        # the length bias.  At this working point the joint (z,L) NLL minimum on the
        # digitized data sits at (-8, +6) mm from truth with the truth point +0.6
        # NLL/event above the minimum -- NEAR truth, not exactly at it.
        # CAVEATS: calibrated on 18-35 events, 300 MeV, mu-, beam-axis direction
        # fixed, reduced-resolution delta table.  A ~4x disagreement between this
        # lambda_s and the noE-only calibration (30 m) indicates the model absorbs
        # unmodeled wall/PMT REFLECTIONS into an effective lambda_s (it is not a pure
        # Rayleigh constant).  NOT validated for pi/K/p, other energies, free
        # direction, Minuit, or WCTE real data.  Set enable_rayleigh_scatter=False
        # to recover pre-scatter predictions.
        self.enable_rayleigh_scatter = _env_bool_switch("EMITTER_ENABLE_RAYLEIGH", DEFAULT_ENABLE_RAYLEIGH, "ENABLE_RAYLEIGH")
        self.photon_scatter_model = _env_str_switch("EMITTER_PHOTON_SCATTER_MODEL", DEFAULT_PHOTON_SCATTER_MODEL).strip().lower()
        self.photon_scatter_spectral_mode = _env_str_switch("EMITTER_PHOTON_SCATTER_SPECTRAL_MODE", DEFAULT_PHOTON_SCATTER_SPECTRAL_MODE).strip().lower()
        self.photon_scatter_n_track = _env_int_switch("EMITTER_PHOTON_SCATTER_N_TRACK", DEFAULT_PHOTON_SCATTER_N_TRACK)
        self.photon_scatter_n_azimuth = _env_int_switch("EMITTER_PHOTON_SCATTER_N_AZIMUTH", DEFAULT_PHOTON_SCATTER_N_AZIMUTH)
        self.photon_scatter_n_interaction = _env_int_switch("EMITTER_PHOTON_SCATTER_N_INTERACTION", DEFAULT_PHOTON_SCATTER_N_INTERACTION)
        self.photon_scatter_n_wavelength = _env_int_switch("EMITTER_PHOTON_SCATTER_N_WAVELENGTH", DEFAULT_PHOTON_SCATTER_N_WAVELENGTH)
        self.photon_scatter_n_raman_shift = _env_int_switch("EMITTER_PHOTON_SCATTER_N_RAMAN_SHIFT", DEFAULT_PHOTON_SCATTER_N_RAMAN_SHIFT)
        self.photon_scatter_n_timing_bins = _env_int_switch("EMITTER_PHOTON_SCATTER_N_TIMING_BINS", DEFAULT_PHOTON_SCATTER_N_TIMING_BINS)
        self.photon_scatter_enable_raman = _env_bool_switch("EMITTER_PHOTON_SCATTER_ENABLE_RAMAN", DEFAULT_PHOTON_SCATTER_ENABLE_RAMAN)
        self.photon_scatter_direct_survival = _env_bool_switch("EMITTER_PHOTON_SCATTER_DIRECT_SURVIVAL", DEFAULT_PHOTON_SCATTER_DIRECT_SURVIVAL)
        self.photon_scatter_include_mpmt_domes = _env_bool_switch("EMITTER_PHOTON_SCATTER_INCLUDE_MPMT_DOMES", DEFAULT_PHOTON_SCATTER_INCLUDE_MPMT_DOMES)
        self.photon_scatter_pmt_aperture_radius_mm = _env_float_switch("EMITTER_PHOTON_SCATTER_PMT_APERTURE_RADIUS_MM", DEFAULT_PHOTON_SCATTER_PMT_APERTURE_RADIUS_MM)
        self.photon_scatter_pmt_facing_soft_width = _env_float_switch("EMITTER_PHOTON_SCATTER_PMT_FACING_SOFT_WIDTH", DEFAULT_PHOTON_SCATTER_PMT_FACING_SOFT_WIDTH)
        self.photon_scatter_parallel_pmt_loop = _env_bool_switch(
            "EMITTER_PHOTON_SCATTER_PARALLEL_PMT_LOOP",
            DEFAULT_PHOTON_SCATTER_PARALLEL_PMT_LOOP,
        )
        self.photon_scatter_receiver_mode = _env_str_switch(
            "EMITTER_PHOTON_SCATTER_RECEIVER_MODE",
            DEFAULT_PHOTON_SCATTER_RECEIVER_MODE,
        ).strip().lower()
        self.photon_scatter_receiver_table = _env_str_switch(
            "EMITTER_PHOTON_SCATTER_RECEIVER_TABLE",
            DEFAULT_PHOTON_SCATTER_RECEIVER_TABLE,
        )
        self.photon_scatter_receiver_table_required = _env_bool_switch(
            "EMITTER_PHOTON_SCATTER_RECEIVER_TABLE_REQUIRED",
            DEFAULT_PHOTON_SCATTER_RECEIVER_TABLE_REQUIRED,
        )
        self.photon_scatter_native_receiver = _env_bool_switch(
            "EMITTER_PHOTON_SCATTER_NATIVE_RECEIVER",
            DEFAULT_PHOTON_SCATTER_NATIVE_RECEIVER,
        )
        self.photon_scatter_native_threads = _env_int_switch(
            "EMITTER_PHOTON_SCATTER_NATIVE_THREADS",
            DEFAULT_PHOTON_SCATTER_NATIVE_THREADS,
        )
        self.photon_scatter_native_required = _env_bool_switch(
            "EMITTER_PHOTON_SCATTER_NATIVE_REQUIRED",
            DEFAULT_PHOTON_SCATTER_NATIVE_REQUIRED,
        )
        self.photon_scatter_boundary_model = _env_str_switch(
            "EMITTER_PHOTON_SCATTER_BOUNDARY_MODEL",
            DEFAULT_PHOTON_SCATTER_BOUNDARY_MODEL,
        ).strip().lower()
        self.rayleigh_scatter_length_mm = _env_float_switch("EMITTER_RAYLEIGH_SCATTER_LENGTH_MM", DEFAULT_RAYLEIGH_SCATTER_LENGTH_MM)
        self.rayleigh_n_sources = _env_int_switch("EMITTER_RAYLEIGH_N_SOURCES", DEFAULT_RAYLEIGH_N_SOURCES)
        self.rayleigh_n_phi = _env_int_switch("EMITTER_RAYLEIGH_N_PHI", DEFAULT_RAYLEIGH_N_PHI)
        self.rayleigh_timing_cut_ns = _env_float_switch("EMITTER_RAYLEIGH_TIMING_CUT_NS", DEFAULT_RAYLEIGH_TIMING_CUT_NS)
        self.rayleigh_timing_soft_mm = _env_float_switch("EMITTER_RAYLEIGH_TIMING_SOFT_MM", DEFAULT_RAYLEIGH_TIMING_SOFT_MM)
        # Speed-only settings: exact-hypothesis cache and optional parallel PMT loop for
        # Rayleigh. It is OFF by default to avoid OpenMP/fork issues in batch jobs;
        # set rayleigh_use_parallel_accumulator=True only for single-process fits.
        self.rayleigh_enable_exact_cache = True
        self.rayleigh_exact_cache_max_entries = 16
        self.rayleigh_use_parallel_accumulator = _env_bool_switch("EMITTER_RAYLEIGH_USE_PARALLEL_ACCUMULATOR", DEFAULT_RAYLEIGH_USE_PARALLEL_ACCUMULATOR)

        # Rayleigh speed modes.  The exact single-scatter field is smooth and
        # small compared with the direct+delta light, but recomputing it for
        # every Minuit FCN call is expensive.  These modes trade tiny controlled
        # approximations for large speedups:
        #   quantized     : DEFAULT. evaluate Rayleigh at binned
        #                   x/y/z/length/direction and cache the result.
        #   exact/smooth  : historical smooth/exact behavior; exact parameters
        #                   in the cache key. Select explicitly with
        #                   EMITTER_RAYLEIGH_CACHE_MODE=exact or smooth.
        #   tolerant_last : reuse the last exact field while parameters stay
        #                   within small physical tolerances.
        #   frozen_init   : evaluate once at the Emitter construction track and
        #                   reuse it; fastest, best as a diagnostic/production
        #                   option only after bias validation.
        self.rayleigh_cache_mode = _env_str_switch("EMITTER_RAYLEIGH_CACHE_MODE", DEFAULT_RAYLEIGH_CACHE_MODE).lower()
        self.rayleigh_quantize_xyz_mm = 20.0
        self.rayleigh_quantize_length_mm = 20.0
        self.rayleigh_quantize_dir = 0.005
        self.rayleigh_reuse_tolerance_xyz_mm = 20.0
        self.rayleigh_reuse_tolerance_length_mm = 20.0
        self.rayleigh_reuse_tolerance_dir = 0.005
        self._rayleigh_init_start_coord = tuple(self.start_coord)
        _rd0 = np.asarray(self.direction, dtype=np.float64)
        _rd0 = _rd0 / max(float(np.linalg.norm(_rd0)), 1e-30)
        self._rayleigh_init_direction = tuple(float(x) for x in _rd0)
        self._rayleigh_init_length = float(self.length)

        # SMOOTH-NLL (opt-in, default off): centered visibility ramp at the
        # PRIMARY path's facing gate.  Set primary_cost_soft to the half-width
        # (e.g. 0.02-0.05 ~ PMT face radius / typical distance) and
        # primary_cost_soft_centered=True to ramp over [-w, +w] instead of the
        # one-sided [0, w].  Both False/0.0 -> historical hard gate, bit-exact.
        self.primary_cost_soft_centered = _env_bool_switch("EMITTER_PRIMARY_COST_SOFT_CENTERED", DEFAULT_PRIMARY_COST_SOFT_CENTERED)

        # SMOOTH-NLL: independent override for the delta segment-visibility
        # gate.  None (default) -> follow smooth_tables; True/False -> force.
        # Exists so ablation studies can isolate the segment gate's effect on
        # fitted lengths without touching the other smooth-table machinery.
        self.delta_e_segment_gate = None

        # SMOOTH-NLL CANDIDATE CONFIG: half-width of the primary facing-gate
        # ramp.  The hard cost<=0 gate at the collapsed emission point switches
        # whole tubes on/off as parameters vary (verified: cost(s_b) sign flip
        # within 0.02 mm of L toggled 0 <-> 0.59 PE; softening took event 1
        # from max|d2| ~17 to 1.28).  Centered vs one-sided was empirically
        # indistinguishable on the tested event; centered is kept for the
        # half-illuminated-face rationale at grazing.  Set to 0.0 (and
        # primary_cost_soft_centered=False) for the historical hard gate.
        self.primary_cost_soft = _env_float_switch("EMITTER_PRIMARY_COST_SOFT", DEFAULT_PRIMARY_COST_SOFT)

        # SMOOTH-NLL: width of the smoothstep visibility ramp at the delta
        # accumulator's PMT-facing gate (see _accumulate_refined_delta_numba).
        # 0.0 restores the historical hard cost<=0 cutoff.
        self.delta_e_cost_soft = _env_float_switch("EMITTER_DELTA_E_COST_SOFT", DEFAULT_DELTA_E_COST_SOFT)

        # Configure the endpoint mode after all attributes used by the kinematic
        # refresh helpers exist.  For the default threshold mode this is exactly
        # the old behavior.
        self.configure_track_end(track_end_mode, fixed_initial_KE=fixed_initial_KE, refresh=False)

        # Match the original behavior: initialise beta from the length-dependent
        # lookup table rather than trusting the constructor beta argument.
        self.refresh_kinematics_from_length(self.length)

        
        


    def __repr__(self):
        return (
            f"Emitter(starting_time={self.starting_time}, start_coord={self.start_coord}, "
            f"direction={self.direction}, beta={self.beta}, length={self.length}, "
            f"intensity={self.intensity})"
        )

    def copy(self):
        """
        Lightweight copy.

        The original version used pickle for every copy, which is much more
        expensive than needed for this small numeric state.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        return new

    def charge_normalization_factor(self, raw_mean, observed_mean):
        """Return the configured raw-to-predicted PE scale.

        ``event_mean`` conditions on the event total and therefore retains
        charge shape only.  ``global_scale`` keeps absolute light information
        and requires an externally calibrated, event-independent positive
        scale.  There is deliberately no fitted-event or simulation-truth
        fallback for a missing calibration.
        """
        raw = float(raw_mean)
        observed = float(observed_mean)
        mode = str(
            getattr(self, "charge_normalization_mode", "event_mean")
        ).strip().lower().replace("-", "_")
        if mode == "event_mean":
            if raw > 0.0:
                return observed / raw
            if observed <= 0.0:
                return 0.0
            raise ValueError(
                "positive observed charge has zero raw optical prediction"
            )
        if mode == "global_scale":
            scale = getattr(self, "global_charge_scale", None)
            if (
                scale is None
                or not np.isfinite(float(scale))
                or float(scale) <= 0.0
            ):
                raise ValueError(
                    "global_scale charge normalization requires a positive "
                    "externally calibrated global_charge_scale"
                )
            return float(scale)
        raise ValueError(
            "charge_normalization_mode must be event_mean or global_scale"
        )

    def set_particle(self, particle):
        """Set the primary particle species for this emitter."""
        self.particle_name = canonical_particle_name(particle)
        set_active_particle(self.particle_name)
        self.particle_mass = particle_mass_mev(self.particle_name)
        self.mu_mass = self.particle_mass  # compatibility alias
        self.primary_subthreshold_range_mm = particle_subthreshold_range_mm(
            self.particle_name
        )
        self.muon_subthreshold_range_mm = self.primary_subthreshold_range_mm
        self.interp_E_init = None
        self._energy_main_idx = None
        self._energy_dist_row = None
        self._energy_energy_row = None
        self.range_to_threshold_mm = float(self.length)
        self.realized_range_to_threshold_mm = float(self.length)
        self.stopping_range_coordinate_scale = 1.0
        self.fixed_realized_range_mm = None
        self.last_visible_length_exceeds_range = False
        self.refresh_kinematics_from_length(self.length)
        return self

    def calc_constants(self, n):
        self.n = float(n)
        self.cos_tq = 1.0 / (self.beta * self.n)
        self.cos_tq = np.clip(self.cos_tq, -1.0, 1.0)
        sin_tq = np.sqrt(max(1e-15, 1.0 - self.cos_tq**2))
        self.cot_tq = self.cos_tq / sin_tq
        self.c = 299.792458
        self.v = self.beta * self.c

    @staticmethod
    def nearest_main_idx(length_mm, particle=None):
        tables = _get_tables(particle)
        overall = tables[2]
        idx = np.searchsorted(overall, float(length_mm))
        idx = np.clip(idx, 1, len(overall) - 1)
        left = overall[idx - 1]
        right = overall[idx]
        if (float(length_mm) - left) <= (right - float(length_mm)):
            idx -= 1
        return int(idx)

    def configure_track_end(
        self,
        mode="threshold",
        fixed_initial_KE=None,
        fixed_realized_range_mm=None,
        refresh=True,
    ):
        """Configure how the primary Cherenkov track terminates.

        Parameters
        ----------
        mode : {"threshold", "abrupt", "straggled_threshold"}
            ``threshold`` keeps the original behavior: fitted ``length`` is the
            dE/dx range to Cherenkov threshold and therefore determines the
            initial kinetic energy.

            ``abrupt`` makes fitted ``length`` the visible primary-Cherenkov
            length only.  The initial kinetic energy must be supplied separately
            through ``fixed_initial_KE``.  This is intended for protons/hadrons
            whose clean Cherenkov ring can end suddenly before the particle has
            slowed to threshold.

            ``straggled_threshold`` keeps a threshold endpoint but separates
            the mean energy-loss range from the event's realised stopping
            range.  Energy, Cherenkov yield, time of flight, and MCS power use
            the fixed initial kinetic energy while optical support ends at
            ``fixed_realized_range_mm``.
        fixed_initial_KE : float or None
            Fixed initial kinetic energy in MeV for abrupt or straggled mode.
        fixed_realized_range_mm : float or None
            Realised range to Cherenkov threshold for straggled mode.
        refresh : bool
            If True, immediately refresh beta and the cached K(s) lookup row.
        """
        mode = str(mode).strip().lower()
        aliases = {
            "threshold": "threshold",
            "range": "threshold",
            "csda": "threshold",
            "old": "threshold",
            "abrupt": "abrupt",
            "truncated": "abrupt",
            "interaction": "abrupt",
            "absorbed": "abrupt",
            "absorption": "abrupt",
            "straggled_threshold": "straggled_threshold",
            "stochastic_threshold": "straggled_threshold",
            "range_straggling": "straggled_threshold",
        }
        if mode not in aliases:
            raise ValueError(
                "track_end_mode must be 'threshold', 'abrupt', or "
                "'straggled_threshold' "
                f"(got {mode!r})"
            )

        self.track_end_mode = aliases[mode]
        if fixed_initial_KE is None:
            self.fixed_initial_KE = None
        else:
            self.fixed_initial_KE = float(fixed_initial_KE)

        if fixed_realized_range_mm is None:
            self.fixed_realized_range_mm = None
        else:
            self.fixed_realized_range_mm = float(fixed_realized_range_mm)

        if self.track_end_mode in {"abrupt", "straggled_threshold"} and self.fixed_initial_KE is None:
            raise ValueError(
                f"{self.track_end_mode} track-end mode requires fixed_initial_KE in MeV. "
                "Use threshold mode if fitted length should determine energy."
            )
        if self.track_end_mode == "straggled_threshold":
            if self.fixed_realized_range_mm is None:
                self.fixed_realized_range_mm = float(self.length)
            if (
                not math.isfinite(self.fixed_realized_range_mm)
                or self.fixed_realized_range_mm <= 0.0
            ):
                raise ValueError(
                    "straggled_threshold requires a positive finite realised range"
                )

        self.interp_E_init = None
        self._energy_main_idx = None
        self._energy_dist_row = None
        self._energy_energy_row = None

        if refresh:
            self.refresh_kinematics_from_length(self.length)
        return self

    # More explicit aliases for fit-driver code.
    set_track_end_mode = configure_track_end
    set_primary_truncation_mode = configure_track_end

    def configure_stopping_range(
        self,
        initial_kinetic_energy_mev,
        realized_range_mm,
        *,
        refresh=True,
    ):
        """Configure an independently realised threshold-stopping range."""

        return self.configure_track_end(
            "straggled_threshold",
            fixed_initial_KE=float(initial_kinetic_energy_mev),
            fixed_realized_range_mm=float(realized_range_mm),
            refresh=bool(refresh),
        )

    def _range_to_threshold_from_energy(self, initial_KE):
        """Interpolate dE/dx-only Cherenkov-visible range [mm] from K0 [MeV].

        The (initial_energy, overall_distance) lookup arrays depend only on the
        particle's fixed range table, so they are sorted once and cached
        per particle.  Previously this rebuilt a ~3000-element list
        comprehension and argsort on every FCN call in absorption mode.
        """
        initial_energies, overall_distances = _get_range_from_energy_arrays(
            self.particle_name
        )

        return float(
            np.interp(
                float(initial_KE),
                initial_energies,
                overall_distances,
                left=overall_distances[0],
                right=overall_distances[-1],
            )
        )

    def _cache_energy_row_for_range(self, range_mm):
        """Cache the K(s) table row corresponding to a dE/dx range."""
        main_idx = self.nearest_main_idx(float(range_mm), particle=self.particle_name)
        tables = _get_tables(self.particle_name)
        self._energy_main_idx = main_idx
        self._energy_dist_row = tables[4][main_idx]
        self._energy_energy_row = tables[3][main_idx]
        return main_idx

    def visible_length_is_physical(self, tol_mm=1e-9):
        """Return False when visible support exceeds its physical endpoint."""
        if self.track_end_mode == "straggled_threshold":
            return (
                float(self.length)
                <= float(self.realized_range_to_threshold_mm) + float(tol_mm)
            )
        if self.track_end_mode != "abrupt":
            return True
        return float(self.length) <= float(self.range_to_threshold_mm) + float(tol_mm)

    def _get_energy_rows_for_length(self, L_stop_mm):
        """
        Return the range-table row used to map distance along track to muon KE.

        For the common case L_stop_mm == self.length, the row is cached by
        refresh_kinematics_from_length(), avoiding repeated table searches for
        every secondary-electron source calculation.
        """
        cached_range = float(getattr(self, "range_to_threshold_mm", self.length))
        if (
            self._energy_dist_row is not None
            and self._energy_energy_row is not None
            and np.isclose(float(L_stop_mm), cached_range, rtol=0.0, atol=1e-12)
        ):
            return self._energy_dist_row, self._energy_energy_row

        overall_distances, energy_rows, distance_rows = _get_tables(self.particle_name)[2:5]
        main_idx = np.searchsorted(overall_distances, float(L_stop_mm))
        main_idx = np.clip(main_idx, 1, len(overall_distances) - 1)

        left = overall_distances[main_idx - 1]
        right = overall_distances[main_idx]
        if (float(L_stop_mm) - left) <= (right - float(L_stop_mm)):
            main_idx -= 1

        return distance_rows[main_idx], energy_rows[main_idx]

    def muon_energy_at_s(self, s_mm, L_stop_mm):
        """
        Approximate primary-particle kinetic energy at distance s along the physical muon path.

        Uses the same range-table philosophy as the collapse solver.
        """
        if getattr(self, "smooth_tables", True):
            # SMOOTH-NLL: scalar version of the continuous master-curve lookup.
            return float(self.muon_energy_at_s_array(np.asarray([float(s_mm)]), L_stop_mm)[0])
        dist_row, energy_row = self._get_energy_rows_for_length(L_stop_mm)
        scaled_s = float(s_mm)
        if getattr(self, "track_end_mode", "threshold") == "straggled_threshold":
            scaled_s *= float(getattr(self, "stopping_range_coordinate_scale", 1.0))
        idx = np.searchsorted(dist_row, scaled_s)
        idx = np.clip(idx, 0, len(dist_row) - 1)
        return energy_row[idx]

    def muon_energy_at_s_array(self, s_mm, L_stop_mm):
        distance = np.asarray(s_mm, dtype=np.float64)
        if getattr(self, "track_end_mode", "threshold") == "straggled_threshold":
            distance = distance * float(
                getattr(self, "stopping_range_coordinate_scale", 1.0)
            )
        if getattr(self, "smooth_tables", True):
            # SMOOTH-NLL: continuous K(s) from the master KE(range) curve.
            # The row-based lookup swaps the entire K(s) row when the fitted
            # length crosses a table-row boundary (~5 mm) and snaps K to the
            # nearest row point.  With the secondary-electron model enabled
            # this staircase re-enters the FCN through the delta source grid,
            # re-roughening NLL(L) even after the primary-path fixes.
            try:
                from .particle_cherenkov_model import _ensure_tables_loaded as _pcm_tables
            except ImportError:
                from particle_cherenkov_model import _ensure_tables_loaded as _pcm_tables
            _mt = _pcm_tables(self.particle_name)
            rem = np.maximum(float(L_stop_mm) - distance, 0.0)
            return np.interp(rem, _mt["master_range"], _mt["master_ke"],
                             left=_mt["master_ke"][0], right=_mt["master_ke"][-1])
        dist_row, energy_row = self._get_energy_rows_for_length(L_stop_mm)
        idx = np.searchsorted(dist_row, distance)
        idx = np.clip(idx, 0, len(dist_row) - 1)
        return energy_row[idx]

    def refresh_kinematics_from_energy(self, initial_KE):
        initial_KE = float(initial_KE)
        if self.interp_E_init is not None and initial_KE == self.interp_E_init:
            return self.interp_E_init

        self.interp_E_init = initial_KE
        self.beta = np.sqrt(
            1.0 - (self.mu_mass / (self.interp_E_init + self.mu_mass)) ** 2
        )
        self.calc_constants(self.n)
        return self.interp_E_init

    def refresh_kinematics_from_length(self, length_mm):
        """Refresh beta and K(s) tables from the current fit length.

        In threshold mode this is the historical behavior: ``length_mm`` selects
        the range-table row and therefore the initial kinetic energy.

        In abrupt mode, ``length_mm`` is only the visible/truncated Cherenkov
        length.  The dE/dx range row and beta are instead selected from the fixed
        initial kinetic energy.

        In straggled-threshold mode, the fixed initial kinetic energy sets the
        mean CSDA coordinate and ``fixed_realized_range_mm`` sets the physical
        endpoint.  Their ratio maps physical arc length continuously onto the
        mean energy-loss coordinate.
        """
        self.length = float(length_mm)

        if getattr(self, "track_end_mode", "threshold") == "straggled_threshold":
            if self.fixed_initial_KE is None or self.fixed_realized_range_mm is None:
                raise ValueError(
                    "Emitter straggled-threshold mode lacks energy or realised range."
                )
            mean_range = self._range_to_threshold_from_energy(self.fixed_initial_KE)
            realized_range = float(self.fixed_realized_range_mm)
            if not math.isfinite(realized_range) or realized_range <= 0.0:
                raise ValueError("realised stopping range must be positive and finite")
            self.range_to_threshold_mm = float(mean_range)
            self.realized_range_to_threshold_mm = realized_range
            self.stopping_range_coordinate_scale = float(mean_range / realized_range)
            self._cache_energy_row_for_range(mean_range)
            self.last_visible_length_exceeds_range = not self.visible_length_is_physical()
            return self.refresh_kinematics_from_energy(self.fixed_initial_KE)

        if getattr(self, "track_end_mode", "threshold") == "abrupt":
            if self.fixed_initial_KE is None:
                raise ValueError(
                    "Emitter is in abrupt track-end mode but fixed_initial_KE is not set."
                )
            self.range_to_threshold_mm = self._range_to_threshold_from_energy(self.fixed_initial_KE)
            self.realized_range_to_threshold_mm = float(self.range_to_threshold_mm)
            self.stopping_range_coordinate_scale = 1.0
            self._cache_energy_row_for_range(self.range_to_threshold_mm)
            self.last_visible_length_exceeds_range = not self.visible_length_is_physical()
            return self.refresh_kinematics_from_energy(self.fixed_initial_KE)

        # Default/old behavior: fitted length is the range to Cherenkov threshold.
        self.range_to_threshold_mm = float(self.length)
        self.realized_range_to_threshold_mm = float(self.length)
        self.stopping_range_coordinate_scale = 1.0
        self.last_visible_length_exceeds_range = False
        main_idx = self._cache_energy_row_for_range(self.length)
        tables = _get_tables(self.particle_name)
        if getattr(self, "smooth_tables", True):
            # SMOOTH-NLL: initial KE from the continuous master KE(range)
            # curve instead of the nearest table row.  The nearest-row lookup
            # quantizes KE in ~1 MeV steps every ~5 mm of L, which makes
            # cot(theta_c) (and hence every emission point) a staircase
            # function of the fitted length -- one of the two dominant
            # sources of NLL(L) roughness.  Row caching above is retained
            # unchanged for the secondary-electron table paths.
            try:
                from .particle_cherenkov_model import _ensure_tables_loaded as _pcm_tables
            except ImportError:
                from particle_cherenkov_model import _ensure_tables_loaded as _pcm_tables
            _mt = _pcm_tables(self.particle_name)
            ke0 = float(np.interp(self.length, _mt["master_range"], _mt["master_ke"],
                                  left=_mt["master_ke"][0], right=_mt["master_ke"][-1]))
            return self.refresh_kinematics_from_energy(ke0)
        return self.refresh_kinematics_from_energy(tables[3][main_idx][0])

    def set_nominal_track_parameters(self, starting_time, start_coord, direction, length):
        self.starting_time = float(starting_time)
        self.start_coord = tuple(float(c) for c in start_coord)
        self.direction = tuple(float(c) for c in direction)
        self.length = float(length)

    def set_wall_track_parameters(self, starting_time, y_w, phi_w, d_w, w_y, w_phi, length, r, sign_cz=+1):
        """ Set the track parameters of the emitter using "wall" parameters.

        Args:
            starting_time (float): The time that emitter starts emission in nanoseconds.
            y_w (float): y coordinate of the wall intersection point
            phi_w (float): azimuthal angle of the wall intersection point
            d_w (float): distance from start to wall intersection point
            w_y (float): cosine of angle between line direction and y-axis
            w_phi (float): cosine of angle in x-z plane between line direction and tangent to cylinder at wall point
            length (float): The length of the path for the emitter (mm).
            r (float): radius of the cylinder
            sign_cz (int): sign of c_z to choose branch (+1 or -1)
        """
        (x_0, y_0, z_0, c_x, c_y), _ = self.inverse_transform_and_jacobian(y_w, phi_w, d_w, w_y, w_phi, r, sign_cz)
        self.starting_time = float(starting_time)
        self.start_coord = (x_0, y_0, z_0)
        self.direction = (c_x, c_y)
        self.length = float(length)

    def get_wall_parameters_and_jacobian(self, r, sign_cz=+1):
        """
        Forward: (x_0, y_0, z_0, c_x, c_y) -> (y_w, phi_w, d_w, w_y, w_phi), J_f (5x5)
        Cylinder axis is y; wall: x^2 + z^2 = r^2.  phi_w = atan2(x_w, z_w).
        """
        def _safe_sqrt(x):
            return np.sqrt(np.maximum(0.0, x))

        (x_0, y_0, z_0) = self.start_coord
        (c_x, c_y) = self.direction

        # Direction and checks
        beta_xy = c_x ** 2 + c_y ** 2
        c_z = sign_cz * _safe_sqrt(1.0 - beta_xy)
        beta = c_x ** 2 + c_z ** 2  # = 1 - c_y**2 = ||c_perp||^2
        if beta <= 0:
            raise ValueError("Degenerate direction: c_x=c_z=0 (parallel to axis).")

        # Solve (x0 + t cx)^2 + (z0 + t cz)^2 = r^2 for first t>0
        alpha = x_0 * c_x + z_0 * c_z
        rho0_sq = x_0 ** 2 + z_0 ** 2
        disc = alpha ** 2 + beta * (r ** 2 - rho0_sq)
        if disc < 0:
            raise ValueError("No intersection with cylinder (discriminant < 0).")
        d_w = (-alpha + np.sqrt(disc)) / beta

        # Hit point and cylindrical coords
        x_w = x_0 + d_w * c_x
        z_w = z_0 + d_w * c_z
        phi_w = np.arctan2(x_w, z_w)  # φ=0 on +z
        y_w = y_0 + d_w * c_y

        # Cosines
        w_y = c_y
        S, C = np.sin(phi_w), np.cos(phi_w)
        sqrt_beta = np.sqrt(beta)
        w_phi = (c_x * C - c_z * S) / sqrt_beta

        # Jacobian building blocks
        a = c_x * S + c_z * C  # c_perp · n (>=0 for outward hit)

        # ∂d_w/∂(x0,z0,cx,cz) at fixed (cx,cz)
        dd_dx0_ind = -S / a
        dd_dz0_ind = -C / a
        dd_dcx_ind = -d_w * S / a
        dd_dcz_ind = -d_w * C / a

        # c_z depends on (c_x, c_y):  ∂c_z/∂c_x = -c_x/c_z,  ∂c_z/∂c_y = -c_y/c_z
        dcz_dcx = -c_x / (c_z if c_z != 0 else 1e-300)
        dcz_dcy = -c_y / (c_z if c_z != 0 else 1e-300)

        # Chain to (c_x, c_y)
        dd_dx0 = dd_dx0_ind
        dd_dz0 = dd_dz0_ind
        dd_dcx = dd_dcx_ind + dd_dcz_ind * dcz_dcx
        dd_dcy = dd_dcz_ind * dcz_dcy

        # φ partials:  dφ = (-x_w dz_w + z_w dx_w)/r^2  ⇒  at wall: dφ = (C dx - S dz)/r
        dphi_dx0_ind = C / (r * a)
        dphi_dz0_ind = -S / (r * a)
        dphi_dcx_ind = d_w * C / (r * a)
        dphi_dcz_ind = -d_w * S / (r * a)

        dphi_dx0 = dphi_dx0_ind
        dphi_dz0 = dphi_dz0_ind
        dphi_dcx = dphi_dcx_ind + dphi_dcz_ind * dcz_dcx
        dphi_dcy = dphi_dcz_ind * dcz_dcy  # dφ/dc_y via c_z only
        # dφ/dy0 = 0

        # Assemble forward Jacobian J_f
        J = np.zeros((5, 5), dtype=float)

        # (1) y_w = y_0 + d_w c_y
        J[0, 0] = c_y * dd_dx0
        J[0, 1] = 1.0
        J[0, 2] = c_y * dd_dz0
        J[0, 3] = c_y * dd_dcx
        J[0, 4] = d_w + c_y * dd_dcy

        # (2) φ_w
        J[1, 0] = dphi_dx0
        J[1, 1] = 0.0
        J[1, 2] = dphi_dz0
        J[1, 3] = dphi_dcx
        J[1, 4] = dphi_dcy

        # (3) d_w
        J[2, 0] = dd_dx0
        J[2, 1] = 0.0
        J[2, 2] = dd_dz0
        J[2, 3] = dd_dcx
        J[2, 4] = dd_dcy

        # (4) w_y = c_y
        J[3, 0] = 0.0
        J[3, 1] = 0.0
        J[3, 2] = 0.0
        J[3, 3] = 0.0
        J[3, 4] = 1.0

        # (5) w_phi = (c_x C - c_z S)/sqrt_beta
        inv_sqrtb = 1.0 / (sqrt_beta if sqrt_beta != 0 else 1e-300)
        inv_beta = 1.0 / (beta if beta != 0 else 1e-300)

        # φ-coupling factor:  ∂w_phi/∂φ at fixed (cx,cz) equals (-a)/sqrt_beta
        fac = (-a) * inv_sqrtb

        # wrt (x0,y0,z0): only via φ
        J[4, 0] = fac * dphi_dx0
        J[4, 1] = 0.0
        J[4, 2] = fac * dphi_dz0

        # wrt (c_x, c_y) including c_z and φ dependences
        # general: dw = (C dcx - S dcz)/sqrtβ + fac dφ - w_phi/β (c_x dcx + c_z dcz)
        coeff_dcz = (-S) * inv_sqrtb - w_phi * c_z * inv_beta
        coeff_dcx = (C) * inv_sqrtb - w_phi * c_x * inv_beta

        J[4, 3] = coeff_dcx + coeff_dcz * dcz_dcx + fac * dphi_dcx  # ∂/∂c_x
        J[4, 4] = coeff_dcz * dcz_dcy + fac * dphi_dcy  # ∂/∂c_y

        return (y_w, phi_w, d_w, w_y, w_phi), J

    def inverse_transform_and_jacobian(y_w, phi_w, d_w, w_y, w_phi, r, sign_cz=+1):
        """
        Inverse: (y_w, phi_w, d_w, w_y, w_phi) -> (x_0, y_0, z_0, c_x, c_y), J_g (5x5)
        Using t_phi = (cosφ, 0, -sinφ).
        """
        def _safe_sqrt(x):
            return np.sqrt(np.maximum(0.0, x))

        S, C = np.sin(phi_w), np.cos(phi_w)
        s = _safe_sqrt(1.0 - w_phi ** 2)  # = sin(angle to t_phi) in xz-plane
        sb = _safe_sqrt(1.0 - w_y ** 2)  # = ||c_perp||

        # Direction (c_perp = sb*(w_phi t_phi + s n))
        c_y = w_y
        c_x = sb * (w_phi * C + s * S)
        c_z = sb * (-w_phi * S + s * C)

        # Optional: enforce chosen c_z branch sign
        if sign_cz < 0 and c_z > 0: c_z = -c_z
        if sign_cz > 0 and c_z < 0: c_z = -c_z

        # Wall point and start point
        x_w = r * S
        z_w = r * C
        x_0 = x_w - d_w * c_x
        y_0 = y_w - d_w * c_y
        z_0 = z_w - d_w * c_z

        # Inverse Jacobian J_g
        J = np.zeros((5, 5), dtype=float)

        # helpers
        dsb_dwy = -(w_y / (sb if sb != 0 else 1e-300))
        ds_dwp = -(w_phi / (s if s != 0 else 1e-300))

        # Direction partials
        dcx_dphi = sb * (w_phi * (-S) + s * C)
        dcx_dwy = dsb_dwy * (w_phi * C + s * S)
        dcx_dwp = sb * (C + ds_dwp * S)

        dcz_dphi = sb * (-w_phi * C - s * S)
        dcz_dwy = dsb_dwy * (-w_phi * S + s * C)
        dcz_dwp = sb * (-S + ds_dwp * C)

        # Rows for start point: x_0 = r*S - d_w*c_x;  z_0 = r*C - d_w*c_z;  y_0 = y_w - d_w*c_y
        J[0, 0] = 0.0
        J[0, 1] = r * C - d_w * dcx_dphi
        J[0, 2] = -c_x
        J[0, 3] = -d_w * dcx_dwy
        J[0, 4] = -d_w * dcx_dwp

        J[1, 0] = 1.0
        J[1, 1] = 0.0
        J[1, 2] = -c_y
        J[1, 3] = -d_w
        J[1, 4] = 0.0

        J[2, 0] = 0.0
        J[2, 1] = -r * S - d_w * dcz_dphi  # because d(r*C)/dφ = -r*S
        J[2, 2] = -c_z
        J[2, 3] = -d_w * dcz_dwy
        J[2, 4] = -d_w * dcz_dwp

        # Rows for direction (outputs 4,5)
        J[3, 0] = 0.0
        J[3, 1] = dcx_dphi
        J[3, 2] = 0.0
        J[3, 3] = dcx_dwy
        J[3, 4] = dcx_dwp

        J[4, 0] = 0.0
        J[4, 1] = 0.0
        J[4, 2] = 0.0
        J[4, 3] = 1.0
        J[4, 4] = 0.0

        return (x_0, y_0, z_0, c_x, c_y), J

    def get_emission_point(self, pmt_coord, initial_KE):
        """
        Emission point for a single PMT.
        """
        x0, y0, z0 = self.start_coord
        cx, cy, cz = self.direction
        px, py, pz = pmt_coord

        dx = px - x0
        dy = py - y0
        dz = pz - z0

        self.refresh_kinematics_from_energy(initial_KE)

        u = cx * dx + cy * dy + cz * dz
        A = dx**2 + dy**2 + dz**2

        if A <= u**2:
            return u
        return u - self.cot_tq * np.sqrt(A - u**2)

    def get_emission_points(self, p_locations, initial_KE):
        """
        Vectorized Cherenkov emission-point calculation for many PMTs.
        """
        x0, y0, z0 = self.start_coord
        cx, cy, cz = self.direction

        p_locations = np.asarray(p_locations, dtype=np.float64)
        dx = p_locations[:, 0] - x0
        dy = p_locations[:, 1] - y0
        dz = p_locations[:, 2] - z0

        self.refresh_kinematics_from_energy(initial_KE)

        u = cx * dx + cy * dy + cz * dz
        A = dx * dx + dy * dy + dz * dz

        ss = np.empty(p_locations.shape[0], dtype=np.float64)
        valid = A > u * u
        ss[valid] = u[valid] - self.cot_tq * np.sqrt(A[valid] - u[valid] * u[valid])
        ss[~valid] = u[~valid]
        return ss

    def power_law(self, x):
        """Angular response. Variant: old_power_y0p1209."""
        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
        xn = x ** 3.0777000000000001
        return (0.1209 + (1.6396999999999999 - 0.1209) * (xn / (xn + 0.79428866592713121))) / 1.002379253316015

    def wl_corr(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_safe = np.maximum(x, 1e-12)

        ymin_wl = 0.1399
        ymax_wl = 1.0
        x50_wl = 3.7620
        n_wl = 2.1020

        return ymin_wl + (ymax_wl - ymin_wl) / (1.0 + (x50_wl / x_safe) ** n_wl)


    def interp_by_mpmt_type(
        self,
        cost,
        mpmt_types,
        cost_for_fit,
        tri_exsitu,
        tri_insitu,
        wut_exsitu,
        wut_insitu,
        fill_empty=1.0,   # was np.nan
    ):
        """
        Interpolate relative mPMT efficiency by mPMT type.

        This keeps the public method signature intact, but uses the faster coded
        implementation whenever the cost grid is the standard uniform [0, 1]
        grid used by this model.  It falls back to the original np.interp loop
        only for a non-standard grid.
        """
        cost = np.asarray(cost, dtype=np.float64)
        cost_for_fit = np.asarray(cost_for_fit, dtype=np.float64)

        if (
            cost_for_fit.ndim == 1
            and cost_for_fit.size == len(tri_exsitu)
            and cost_for_fit.size >= 2
            and np.isclose(cost_for_fit[0], 0.0)
            and np.isclose(cost_for_fit[-1], 1.0)
            and np.allclose(np.diff(cost_for_fit), cost_for_fit[1] - cost_for_fit[0])
        ):
            return _interp_rel_mpmt_eff_from_codes(
                cost,
                _encode_mpmt_types(mpmt_types),
                fill_empty=fill_empty,
            )

        # Fallback: original generic implementation.
        mpmt_types = np.asarray(mpmt_types)
        out = np.full(cost.shape, fill_empty, dtype=np.float64)
        y_by_type = {
            "tri_exsitu": np.asarray(tri_exsitu, dtype=np.float64),
            "tri_insitu": np.asarray(tri_insitu, dtype=np.float64),
            "wut_exsitu": np.asarray(wut_exsitu, dtype=np.float64),
            "wut_insitu": np.asarray(wut_insitu, dtype=np.float64),
        }
        for typ, yvals in y_by_type.items():
            mask = mpmt_types == typ
            if np.any(mask):
                out[mask] = np.interp(
                    cost[mask],
                    cost_for_fit,
                    yvals,
                    left=yvals[0],
                    right=yvals[-1],
                )
        return out


    def particle_dedx_positive(self, E_MeV):
        """
        Positive primary-particle stopping power, -dE/ds, in MeV/mm.

        This is derived from the same particle range table used by the collapse
        solver. It is needed for the analytic cone-density falloff:

            d cos(theta_c)/ds = (-dE/ds) / (n m beta^3 gamma^3).
        """
        return _interp_particle_dedx_positive(E_MeV, self.particle_name)

    def muon_dedx_positive(self, E_MeV):
        """Backward-compatible alias for particle_dedx_positive()."""
        return self.particle_dedx_positive(E_MeV)


    def primary_ngeo_falloff_raw(self, E_MeV, r_mm):
        """
        Analytic cone-density geometric falloff for primary muon light.

        This is the analytic replacement for n_from_E_r(E, r).  It excludes:
          - Frank-Tamm / Cherenkov light-yield scale
          - PMT angular response
          - relative mPMT efficiency

        Those factors are applied elsewhere in get_expected_pes_ts.

        Formula
        -------
        N_geo(E,r) = 1 / [ r_eff sin^2(theta_c(E))
                           + r_eff^2 d cos(theta_c)/ds ]

        where

            r_eff = sqrt(r^2 + a^2)

        and

            d cos(theta_c)/ds = (-dE/ds) / (n m beta^3 gamma^3).

        Units are arbitrary up to an overall constant; the public
        primary_ngeo_falloff() applies a fixed reference normalization so that
        the result has approximately the same convention as n_from_E_r.
        """
        E = np.asarray(E_MeV, dtype=np.float64)
        r = np.asarray(r_mm, dtype=np.float64)

        gamma = 1.0 + E / self.mu_mass
        beta2 = np.clip(1.0 - 1.0 / np.maximum(gamma, 1e-30)**2, 0.0, None)
        beta = np.sqrt(beta2)

        above = self.n * beta > 1.0

        cos_tc = np.zeros_like(E, dtype=np.float64)
        cos_tc[above] = 1.0 / (self.n * beta[above])

        sin2_tc = np.zeros_like(E, dtype=np.float64)
        sin2_tc[above] = 1.0 - cos_tc[above]**2

        dEdx = self.muon_dedx_positive(E)
        dc_ds = np.zeros_like(E, dtype=np.float64)
        dc_ds[above] = dEdx[above] / (
            self.n * self.mu_mass * beta[above]**3 * gamma[above]**3
        )

        a = float(self.primary_ngeo_pmt_radius_mm)
        r_eff = np.sqrt(r*r + a*a)

        denom = r_eff * sin2_tc + r_eff*r_eff * dc_ds

        out = np.zeros(np.broadcast(E, r).shape, dtype=np.float64)
        good = above & np.isfinite(denom) & (denom > 0.0)
        out[good] = 1.0 / denom[good]

        return out


    def primary_ngeo_reference_energy(self):
        """Return a particle-safe reference energy for N_geo normalization.

        The historical value primary_ngeo_ref_energy_MeV=304 MeV was tuned for
        muons.  For protons in water, 304 MeV is below Cherenkov threshold, so
        primary_ngeo_falloff_raw(304 MeV, r_ref) is exactly zero.  That made the
        normalization fall back to 1.0 and effectively removed the primary
        proton ring.
        """
        E_ref = float(self.primary_ngeo_ref_energy_MeV)
        if getattr(self, "primary_ngeo_auto_ref_above_threshold", True):
            threshold = float(self.get_cherenkov_threshold_kinetic_energy())
            min_ref = max(
                threshold + float(getattr(self, "primary_ngeo_ref_threshold_margin_MeV", 25.0)),
                threshold * float(getattr(self, "primary_ngeo_ref_threshold_factor", 2.0)),
            )
            if E_ref <= min_ref:
                E_ref = min_ref
        return float(E_ref)


    def primary_ngeo_normalization(self):
        """
        Fixed convention factor for N_geo.

        Cached globally because this scalar normalization is otherwise
        recomputed for every Minuit FCN call even though it only depends on the
        optical constants and chosen reference point.
        """
        E_ref = self.primary_ngeo_reference_energy()
        r_ref = float(self.primary_ngeo_ref_r_mm)
        key = (
            self.particle_name,
            float(self.n),
            float(self.particle_mass),
            float(self.primary_ngeo_pmt_radius_mm),
            E_ref,
            r_ref,
        )

        cached = _PRIMARY_NGEO_NORM_CACHE.get(key)
        if cached is not None:
            return cached

        raw_ref = self.primary_ngeo_falloff_raw(
            np.asarray([E_ref], dtype=np.float64),
            np.asarray([r_ref], dtype=np.float64),
        )[0]

        if not np.isfinite(raw_ref) or raw_ref <= 0.0:
            # This should no longer happen for protons because the reference
            # energy is moved above threshold, but keep a clear diagnostic state
            # instead of silently producing a tiny primary component.
            norm = 1.0
        else:
            norm = float(n_from_E_r(E_ref, r_ref) / raw_ref)

        self._last_primary_ngeo_norm = float(norm)
        self._last_primary_ngeo_ref_energy_MeV = float(E_ref)
        self._last_primary_ngeo_raw_ref = float(raw_ref)

        _PRIMARY_NGEO_NORM_CACHE[key] = norm
        return norm


    def primary_ngeo_falloff(self, E_MeV, r_mm):
        """
        Normalized analytic primary-muon falloff.

        Use this in place of n_from_E_r(E_b, r) for the primary muon term.
        """
        return self.primary_ngeo_normalization() * self.primary_ngeo_falloff_raw(E_MeV, r_mm)


    def get_physical_stop_length_from_cherenkov_length(self):
        if getattr(self, "track_end_mode", "threshold") == "abrupt":
            return float(self.length)
        return self.length + self.muon_subthreshold_range_mm

    def get_cherenkov_threshold_kinetic_energy(self):
        return cherenkov_threshold_kinetic_mev(float(self.particle_mass), n=float(self.n))

    def beta2_from_K(self, K, mass):
        K = np.asarray(K, dtype=np.float64)
        gamma = 1.0 + K / mass
        return np.clip(1.0 - 1.0 / gamma**2, 0.0, 1.0)


    def frank_tamm_factor(self, K, mass):
        beta2 = self.beta2_from_K(K, mass)
        out = 1.0 - 1.0 / (self.n**2 * np.maximum(beta2, 1e-30))
        return np.where(beta2 * self.n**2 > 1.0, np.maximum(out, 0.0), 0.0)


    def electron_cherenkov_threshold(self):
        m_e = 0.51099895
        beta_thr = 1.0 / self.n
        gamma_thr = 1.0 / np.sqrt(1.0 - beta_thr**2)
        return m_e * (gamma_thr - 1.0)


    def electron_range_cm(self, T):
        """
        Approximate electron CSDA range in water.

        Returns range in cm. Since liquid water has rho ~= 1 g/cm^3,
        a mass range in g/cm^2 is numerically equal to a length in cm.

        Uses a Katz-Penfold-style empirical approximation.
        For serious production use, replace this with ESTAR interpolation.
        """
        T = np.asarray(T, dtype=np.float64)
        T_safe = np.maximum(T, 1e-12)

        out = np.zeros_like(T_safe)

        low = T_safe <= 2.5

        # Corrected low-energy form:
        out[low] = 0.412 * T_safe[low] ** (
            1.265 - 0.0954 * np.log(T_safe[low])
        )

        # Higher-energy empirical form.
        out[~low] = 0.530 * T_safe[~low] - 0.106

        return np.maximum(out, 0.0)


    def Tmax_delta_e(self, K_mu):
        m_e = 0.51099895
        m_mu = self.mu_mass

        K_mu = np.asarray(K_mu, dtype=np.float64)
        beta2 = self.beta2_from_K(K_mu, m_mu)
        gamma = 1.0 + K_mu / m_mu

        return (
            2.0 * m_e * beta2 * gamma**2
            / (1.0 + 2.0 * gamma * m_e / m_mu + (m_e / m_mu)**2)
        )

    def delta_e_photon_angle_deg(self, K_mu):
        """
        Returns the average photon angle produced by knock-on electrons at a given muon energy
        Should replace this with the actual shape of angular distribution
        """
        theta_max = 55.41
        A = 41.43
        tau = 32.3

        return theta_max - A*np.exp(-K_mu/tau)

    def load_delta_e_angular_pdf_table(self, path):
        self.delta_e_angular_pdf_path = path
        return self

    def evaluate_refined_analytic_delta_dSdu(self, K_mu, cos_forward):
        """
        Evaluate the refined analytic secondary-electron angular/yield model.

        Returns dS_delta/du(K_mu, u), where u = cos_forward is the photon
        direction cosine relative to the primary muon direction.

        Unlike the old factorized model, this table already includes the
        secondary-electron yield and the angular shape together.  It therefore
        replaces

            S_delta(K_mu) * evaluate_delta_e_angular_pdf(K_mu, u)

        in get_delta_e_expected_pes.
        """
        K_mu = np.asarray(K_mu, dtype=np.float64)
        u = np.asarray(cos_forward, dtype=np.float64)

        K_grid, u_grid, table = get_refined_analytic_delta_cache(self.n, projectile_mass=float(self.particle_mass), particle=self.particle_name)

        valid_K = np.isfinite(K_mu)
        # The table now spans the full physical range -1 <= u <= 1 (the knock-on
        # light has a backward hemisphere once multiple scattering is included).
        valid_u = np.isfinite(u) & (u >= -1.0) & (u <= 1.0)

        K_safe = np.where(valid_K, K_mu, K_grid[0])
        u_safe = np.where(np.isfinite(u), u, u_grid[0])

        K_clip = np.clip(K_safe, K_grid[0], K_grid[-1])
        u_clip = np.clip(u_safe, u_grid[0], u_grid[-1])

        iK = np.searchsorted(K_grid, K_clip, side="right") - 1
        iK = np.clip(iK, 0, len(K_grid) - 2)

        K0 = K_grid[iK]
        K1 = K_grid[iK + 1]
        tK = (K_clip - K0) / (K1 - K0 + 1e-300)
        tK = np.clip(tK, 0.0, 1.0)

        du = u_grid[1] - u_grid[0]
        iu = np.floor((u_clip - u_grid[0]) / du).astype(np.int64)
        iu = np.clip(iu, 0, len(u_grid) - 2)

        tu = (u_clip - u_grid[iu]) / (u_grid[iu + 1] - u_grid[iu] + 1e-300)
        tu = np.clip(tu, 0.0, 1.0)

        src_idx = np.arange(K_mu.size)[:, None]

        row0 = table[iK]
        row1 = table[iK + 1]

        p00 = row0[src_idx, iu]
        p01 = row0[src_idx, iu + 1]
        p10 = row1[src_idx, iu]
        p11 = row1[src_idx, iu + 1]

        p0 = p00 + tu * (p01 - p00)
        p1 = p10 + tu * (p11 - p10)
        out = p0 + tK[:, None] * (p1 - p0)

        out[~valid_u] = 0.0
        out[~valid_K, :] = 0.0
        out[~np.isfinite(out)] = 0.0
        out[out < 0.0] = 0.0

        return out


    def _build_delta_source_grid(self):
        """Build the secondary-electron source grid (s_centers, ds_cm, K_mu).

        Extracted verbatim from get_delta_e_expected_pes so the result can be
        cached by length scalars.  Returns (s_centers, ds_cm, K_mu, any_valid)
        with the valid-source mask already applied; any_valid is False when no
        source bin survives (the caller then returns zeros).
        """
        L_ch = max(float(self.length), 0.0)
        if getattr(self, "track_end_mode", "threshold") == "abrupt":
            L_tail = 0.0
        else:
            L_tail = max(float(self.muon_subthreshold_range_mm), 0.0)
        L_stop_for_energy = float(getattr(self, "range_to_threshold_mm", L_ch))
        n_ch = max(1, int(self.n_delta_steps))

        tail_step_mm = max(float(getattr(self, "delta_e_tail_step_mm", 20.0)), 1e-12)
        tail_min_steps = max(1, int(getattr(self, "delta_e_tail_min_steps", 3)))

        if L_ch > 0.0:
            s_edges_ch = np.linspace(0.0, L_ch, n_ch + 1, dtype=np.float64)
        else:
            s_edges_ch = np.array([0.0], dtype=np.float64)

        if L_tail > 0.0:
            n_tail = max(tail_min_steps, int(np.ceil(L_tail / tail_step_mm)))
            s_edges_tail = L_ch + np.linspace(0.0, L_tail, n_tail + 1, dtype=np.float64)[1:]
            s_edges = np.concatenate([s_edges_ch, s_edges_tail])
        else:
            s_edges = s_edges_ch

        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        ds_cm = np.diff(s_edges) / 10.0

        K_mu = np.zeros_like(s_centers, dtype=np.float64)
        above_threshold = s_centers <= L_ch
        below_threshold = ~above_threshold

        if np.any(above_threshold):
            K_mu[above_threshold] = self.muon_energy_at_s_array(s_centers[above_threshold], L_stop_for_energy)

        if np.any(below_threshold):
            K_thr = self.muon_energy_at_s_array(np.array([L_ch], dtype=np.float64), L_stop_for_energy)[0]
            d_post = s_centers[below_threshold] - L_ch
            frac = np.clip(d_post / max(L_tail, 1e-12), 0.0, 1.0)
            K_mu[below_threshold] = K_thr * (1.0 - frac)

        K_mu = np.maximum(K_mu, 0.0)

        valid_src = (
            np.isfinite(K_mu)
            & (K_mu > 0.0)
            & np.isfinite(ds_cm)
            & (ds_cm > 0.0)
        )

        if not np.any(valid_src):
            return None, None, None, False

        s_centers = np.ascontiguousarray(s_centers[valid_src], dtype=np.float64)
        ds_cm = np.ascontiguousarray(ds_cm[valid_src], dtype=np.float64)
        K_mu = np.ascontiguousarray(K_mu[valid_src], dtype=np.float64)
        return s_centers, ds_cm, K_mu, True

    def _fast_delta_first_arrival_nodes(
        self,
        p_locations,
        direction_zs,
        start_pos,
        track_dir,
        mpmt_codes,
        active_indices,
        authoritative_mu_delta,
    ):
        """Return source-resolved analytic delta PE/time nodes for hit PMTs.

        The same analytic dS_delta/du kernel and source grid used by the charge
        model are evaluated one source segment at a time on only the observed PMT
        columns.  Columns are finally rescaled to the authoritative aggregate
        delta PE, so this changes timing resolution but not the validated charge
        allocation.
        """
        active = np.ascontiguousarray(active_indices, dtype=np.int64)
        if active.size == 0:
            return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
        s_centers, ds_cm, K_mu, valid = self._build_delta_source_grid()
        if not valid:
            return np.zeros((0, active.size), np.float32), np.zeros((0, active.size), np.float32)
        psel = np.ascontiguousarray(np.asarray(p_locations, dtype=np.float64)[active])
        nsel = np.ascontiguousarray(np.asarray(direction_zs, dtype=np.float64)[active])
        csel = np.ascontiguousarray(np.asarray(mpmt_codes, dtype=np.int16)[active])
        start = np.ascontiguousarray(start_pos, dtype=np.float64)
        direction = np.ascontiguousarray(track_dir, dtype=np.float64)
        K_grid, u_grid, table = get_refined_analytic_delta_cache(
            self.n, projectile_mass=float(self.particle_mass), particle=self.particle_name
        )
        K_grid = np.ascontiguousarray(K_grid, dtype=np.float64)
        u_grid = np.ascontiguousarray(u_grid, dtype=np.float64)
        table = np.ascontiguousarray(table, dtype=np.float64)
        rel_table = np.ascontiguousarray(_get_rel_eff_stack(), dtype=np.float64)
        nsrc = len(s_centers)
        nodes = np.zeros((nsrc, active.size), dtype=np.float64)
        times = np.full((nsrc, active.size), np.inf, dtype=np.float64)
        tof_src = _wcte_integrated_primary_tof_fast(self, s_centers)
        for j in range(nsrc):
            sc = np.ascontiguousarray(s_centers[j:j+1], dtype=np.float64)
            dc = np.ascontiguousarray(ds_cm[j:j+1], dtype=np.float64)
            kk = np.ascontiguousarray(K_mu[j:j+1], dtype=np.float64)
            lo = np.ascontiguousarray(
                start[None, :] + (sc - 5.0 * dc)[:, None] * direction[None, :],
                dtype=np.float64,
            )
            hi = np.ascontiguousarray(
                start[None, :] + (sc + 5.0 * dc)[:, None] * direction[None, :],
                dtype=np.float64,
            )
            _dummy_map = np.full(psel.shape[0], -1, dtype=np.int32)
            mu_j, _, _, _ = _accumulate_refined_delta_numba(
                psel, nsel, start, direction, sc, dc, kk,
                K_grid, u_grid, table, csel, rel_table,
                bool(getattr(self, "delta_e_apply_mpmt_eff_by_source", True)),
                bool(getattr(self, "delta_e_use_finite_disk_solid_angle", True)),
                float(getattr(self, "delta_e_distance_pmt_radius_mm", 45.0)),
                float(getattr(self, "delta_e_distance_ref_r_mm", 1000.0)),
                float(getattr(self, "delta_e_distance_power", 2.0)),
                float(getattr(self, "analytic_delta_scale", 1.0)),
                float(getattr(self, "delta_e_source_k_power", 0.0)),
                float(getattr(self, "delta_e_source_k_ref_MeV", 100.0)),
                float(getattr(self, "delta_e_source_k_floor_MeV", 25.0)),
                float(self.intensity), float(self.starting_time), float(self.v),
                float(self.n), float(self.c), 0.0, False, False,
                _dummy_map, 0, np.zeros(1, dtype=np.float64),
                float(_WCTE_REFLECTION_GROUP_INDEX / 299.792458), False,
                _POWER_LAW_LUT, float(getattr(self, "delta_e_cost_soft", 0.0)),
                lo, hi,
                int(1 if (getattr(self, "smooth_tables", True)
                          if getattr(self, "delta_e_segment_gate", None) is None
                          else bool(getattr(self, "delta_e_segment_gate"))) else 0),
            )
            mu_j = float(self.delta_e_scale) * np.asarray(mu_j, dtype=np.float64)
            nodes[j] = mu_j
            src = start + float(s_centers[j]) * direction
            path = np.linalg.norm(psel - src[None, :], axis=1) + 0.01
            times[j] = (
                float(self.starting_time) + float(tof_src[j])
                + path * (_WCTE_REFLECTION_GROUP_INDEX / 299.792458)
            )
        target = np.asarray(authoritative_mu_delta, dtype=np.float64)[active]
        source_sum = np.sum(nodes, axis=0)
        fac = np.divide(target, source_sum, out=np.zeros_like(target), where=(source_sum > 0.0))
        nodes *= fac[None, :]
        times = np.where(nodes > 0.0, times, np.inf)
        return np.asarray(nodes, dtype=np.float32), np.asarray(times, dtype=np.float32)

    @staticmethod
    def _empty_delta_result(
        n_pmts,
        *,
        return_times,
        return_source_nodes,
        source_node_active_indices,
        return_source_node_times,
    ):
        """Return zero predictions with the requested delta-result tuple shape."""
        zeros = np.zeros(int(n_pmts), dtype=np.float64)
        times = np.full(int(n_pmts), np.nan, dtype=np.float64)
        if source_node_active_indices is None:
            n_active = int(n_pmts)
        else:
            n_active = int(np.asarray(source_node_active_indices).size)
        source_mu = np.zeros((0, n_active), dtype=np.float64)
        source_times = np.zeros((0, n_active), dtype=np.float64)
        source_s = np.zeros(0, dtype=np.float64)
        if return_source_nodes:
            if return_times and return_source_node_times:
                return zeros, times, source_mu, source_times, source_s
            if return_times:
                return zeros, times, source_mu, source_s
            if return_source_node_times:
                return zeros, source_mu, source_times, source_s
            return zeros, source_mu, source_s
        if return_times:
            return zeros, times
        return zeros

    def get_delta_e_expected_pes(
        self,
        p_locations,
        direction_zs,
        start_pos,
        track_dir,
        mpmt_types=None,
        return_times=False,
        return_source_nodes=False,
        source_node_active_indices=None,
        return_source_node_times=False,
    ):
        """
        Fast secondary-electron expected PE model.

        For the refined analytic model this uses a Numba-compiled source x PMT
        accumulator.  The physics and algebra are the same as the previous
        vectorized implementation, but it avoids materializing large temporary
        matrices for dx/dy/dz/r/cost/optical_corr/forward_kernel/delta_contrib.

        """

        p_locations = np.asarray(p_locations, dtype=np.float64)
        direction_zs = np.asarray(direction_zs, dtype=np.float64)
        start_pos = np.asarray(start_pos, dtype=np.float64)
        track_dir = np.asarray(track_dir, dtype=np.float64)
        track_dir = track_dir / np.linalg.norm(track_dir)

        n_pmts = p_locations.shape[0]

        # The secondary-electron *source* grid (s_centers, ds_cm, K_mu) depends
        # only on the track length scalars and particle, not on the vertex or
        # direction.  During a fit (and especially the seed prescan, which sweeps
        # ~15k vertex/direction combinations over only ~160 unique length pairs)
        # this is recomputed far more often than it changes, so cache it.
        # Bit-identical: identical scalar key -> identical arrays.
        L_ch = max(float(self.length), 0.0)
        L_stop_for_energy = float(getattr(self, "range_to_threshold_mm", L_ch))
        _src_key = (
            self.particle_name,
            L_ch,
            L_stop_for_energy,
            # SMOOTH-NLL: K(s) values depend on the table mode, so the cached
            # source grid must be keyed by it (prevents stale cross-mode reuse
            # when smooth_tables is toggled on a live emitter, e.g. in A/B tests).
            bool(getattr(self, "smooth_tables", True)),
            getattr(self, "track_end_mode", "threshold"),
            int(self.n_delta_steps),
            float(getattr(self, "muon_subthreshold_range_mm", 0.0)),
            float(getattr(self, "delta_e_tail_step_mm", 20.0)),
            int(getattr(self, "delta_e_tail_min_steps", 3)),
        )
        _src_cached = self._delta_src_grid_cache.get(_src_key) if getattr(
            self, "_delta_src_grid_cache", None) is not None else None

        if _src_cached is not None:
            s_centers, ds_cm, K_mu, _any_valid = _src_cached
            if not _any_valid:
                return self._empty_delta_result(
                    n_pmts,
                    return_times=return_times,
                    return_source_nodes=return_source_nodes,
                    source_node_active_indices=source_node_active_indices,
                    return_source_node_times=return_source_node_times,
                )
        else:
            s_centers, ds_cm, K_mu, _any_valid = self._build_delta_source_grid()
            if self._delta_src_grid_cache is None:
                self._delta_src_grid_cache = {}
            # Minuit generates many nearly unique floating-point length values.
            # Retaining thousands of exact source grids gives little reuse and
            # causes every forked worker to grow indefinitely.  Keep a bounded
            # exact FIFO cache instead: no quantization and no physics change.
            _delta_cache_max = max(16, int(getattr(
                self, "delta_source_cache_max_entries", 256
            )))
            if len(self._delta_src_grid_cache) >= _delta_cache_max:
                self._delta_src_grid_cache.pop(next(iter(self._delta_src_grid_cache)))
            self._delta_src_grid_cache[_src_key] = (s_centers, ds_cm, K_mu, _any_valid)
            if not _any_valid:
                return self._empty_delta_result(
                    n_pmts,
                    return_times=return_times,
                    return_source_nodes=return_source_nodes,
                    source_node_active_indices=source_node_active_indices,
                    return_source_node_times=return_source_node_times,
                )

        # ------------------------------------------------------------
        # DIAGNOSTIC ONLY:
        # Keep the secondary-electron yield and muon-energy assignment
        # fixed, but project the light as if it were emitted downstream
        # from the parent muon point.
        #
        # This tests whether spatially collapsing electron range back to
        # the muon point is responsible for underfilling the ring center.
        # ------------------------------------------------------------
#         delta_shift_mm = float(getattr(self, "delta_e_test_forward_shift_mm", 0.0))
#         s_centers_for_projection = np.ascontiguousarray(
#             s_centers + delta_shift_mm,
#             dtype=np.float64,
#         )
        
        ###
        

        K_grid, u_grid, table = get_refined_analytic_delta_cache(self.n, projectile_mass=float(self.particle_mass), particle=self.particle_name)
       
        K_grid = np.ascontiguousarray(K_grid, dtype=np.float64)
        u_grid = np.ascontiguousarray(u_grid, dtype=np.float64)
        table = np.ascontiguousarray(table, dtype=np.float64)
        
        
        # ------------------------------------------------------------------
        # DEBUG / DIAGNOSTIC ONLY:
        # Modify the refined secondary-electron angular table shape.
        # This tests whether the electron light is too broad in u.
        #
        # Important:
        #   If delta_e_debug_preserve_yield=True, each K row is renormalized so
        #   the total secondary-electron yield S_delta(K) stays approximately fixed.
        #   Therefore this tests angular broadness, not total electron intensity.
        # ------------------------------------------------------------------
        u_power = float(getattr(self, "delta_e_debug_u_power", 0.0))
        u_min = float(getattr(self, "delta_e_debug_u_min", 0.0))
        preserve_yield = bool(getattr(self, "delta_e_debug_preserve_yield", True))

        if (u_power != 0.0) or (u_min > 0.0):
            table_mod = np.array(table, dtype=np.float64, copy=True)

            du = float(u_grid[1] - u_grid[0])
            row_yield_before = np.sum(table_mod, axis=1) * du

            if u_min > 0.0:
                table_mod[:, u_grid < u_min] = 0.0

            if u_power != 0.0:
                weights = np.clip(u_grid, 0.0, 1.0) ** u_power
                table_mod *= weights[None, :]

            if preserve_yield:
                row_yield_after = np.sum(table_mod, axis=1) * du
                renorm = np.divide(
                    row_yield_before,
                    row_yield_after,
                    out=np.ones_like(row_yield_before),
                    where=(row_yield_after > 0.0) & np.isfinite(row_yield_after),
                )
                table_mod *= renorm[:, None]

            table = table_mod
        ###

        if mpmt_types is None:
            mpmt_codes = np.full(n_pmts, -1, dtype=np.int16)
        else:
            mpmt_codes = _encode_mpmt_types(mpmt_types)
            mpmt_codes = np.asarray(mpmt_codes, dtype=np.int16)
            if mpmt_codes.ndim != 1 or mpmt_codes.size != n_pmts:
                mpmt_codes = np.broadcast_to(mpmt_codes, (n_pmts,)).astype(np.int16, copy=False)

        rel_table = _get_rel_eff_stack()
        rel_table = np.ascontiguousarray(rel_table, dtype=np.float64)

        if return_source_nodes:
            if source_node_active_indices is None:
                _node_active = np.arange(n_pmts, dtype=np.int32)
            else:
                _node_active = np.ascontiguousarray(source_node_active_indices, dtype=np.int32)
            _node_map = np.full(n_pmts, -1, dtype=np.int32)
            _node_map[_node_active] = np.arange(_node_active.size, dtype=np.int32)
            _source_tof = (
                _wcte_integrated_primary_tof_fast(self, s_centers)
                if return_source_node_times else np.zeros(s_centers.size, dtype=np.float64)
            )
        else:
            _node_active = np.empty(0, dtype=np.int32)
            _node_map = np.full(n_pmts, -1, dtype=np.int32)
            _source_tof = np.zeros(s_centers.size, dtype=np.float64)

        mu_delta, t_delta, source_node_mu, source_node_t = _accumulate_refined_delta_numba(
            np.ascontiguousarray(p_locations, dtype=np.float64),
            np.ascontiguousarray(direction_zs, dtype=np.float64),
            np.ascontiguousarray(start_pos, dtype=np.float64),
            np.ascontiguousarray(track_dir, dtype=np.float64),
            s_centers,
            ds_cm,
            K_mu,
            K_grid,
            u_grid,
            table,
            np.ascontiguousarray(mpmt_codes, dtype=np.int16),
            rel_table,
            bool((mpmt_types is not None) and np.any(mpmt_codes >= 0)
                 and getattr(self, "delta_e_apply_mpmt_eff_by_source", True)),
            bool(getattr(self, "delta_e_use_finite_disk_solid_angle", True)),
            float(getattr(self, "delta_e_distance_pmt_radius_mm", 37.0)),
            float(getattr(self, "delta_e_distance_ref_r_mm", 1000.0)),
            float(getattr(self, "delta_e_distance_power", 2.0)),
            float(getattr(self, "analytic_delta_scale", 1.0)),
            float(getattr(self, "delta_e_source_k_power", 0.0)),
            float(getattr(self, "delta_e_source_k_ref_MeV", 100.0)),
            float(getattr(self, "delta_e_source_k_floor_MeV", 25.0)),
            float(self.intensity),
            float(self.starting_time),
            float(self.v),
            float(self.n),
            float(self.c),
            float(getattr(self, "delta_e_time_offset_ns", 0.0)),
            bool(return_times),
            bool(return_source_nodes),
            np.ascontiguousarray(_node_map, dtype=np.int32),
            int(_node_active.size),
            np.ascontiguousarray(_source_tof, dtype=np.float64),
            float(_WCTE_REFLECTION_GROUP_INDEX / 299.792458),
            bool(return_source_node_times),
            (_POWER_LAW_LUT if getattr(self, "use_delta_power_law_lut", True)
             else _POWER_LAW_LUT_EMPTY),
            float(getattr(self, "delta_e_cost_soft", 0.0)),
            np.ascontiguousarray(
                start_pos[None, :] + (s_centers - 5.0 * ds_cm)[:, None] * track_dir[None, :],
                dtype=np.float64),
            np.ascontiguousarray(
                start_pos[None, :] + (s_centers + 5.0 * ds_cm)[:, None] * track_dir[None, :],
                dtype=np.float64),
            int(1 if (getattr(self, "smooth_tables", True)
                      if getattr(self, "delta_e_segment_gate", None) is None
                      else bool(getattr(self, "delta_e_segment_gate"))) else 0),
        )

        if return_source_nodes:
            # source_node_mu is already intensity-scaled and sums to mu_delta.
            # Return the source coordinates so the caller can assign the
            # physically integrated primary flight time and photon group delay
            # without a second evaluation of the delta kernel.
            if return_times and return_source_node_times:
                return mu_delta, t_delta, source_node_mu, source_node_t, s_centers
            if return_times:
                return mu_delta, t_delta, source_node_mu, s_centers
            if return_source_node_times:
                return mu_delta, source_node_mu, source_node_t, s_centers
            return mu_delta, source_node_mu, s_centers
        if return_times:
            return mu_delta, t_delta
        return mu_delta


    def effective_primary_soft_cone_sigma_rad(self, wcd=None):
        """Return the primary cone-softening width, including optional MCS.

        This version keeps the historical K0-local Highland model available as
        ``primary_mcs_energy_mode="initial"`` but also supports physically
        motivated energy-dependent variants:

        * ``midpoint`` evaluates the local PMT patch at s=0.5 L_visible.
        * ``emission_weighted`` samples K(s) along the visible track and returns
          the RMS local Highland width, weighted by the Frank-Tamm
          sin^2(theta_c) yield.  This gives a deterministic energy/track-length
          dependence without tuning separate values for 300/400/500 MeV.
        * ``bridge_weighted`` uses the Fermi--Eyges scattering power along the
          complete visible support and measures each local tangent relative to
          the Frank--Tamm-weighted mean tangent absorbed by the fitted straight
          direction.  For constant scattering power and light yield this is
          exactly ``sigma_total/sqrt(6)``.  It is the inexpensive marginal
          broadening appropriate after fitting a straight direction, rather
          than the unrelated PMT-patch thickness used by the legacy model.

        The result is a single effective angular width because the current
        collapse solver accepts one ``near_cross_tol`` per call.  The edge model
        itself can be made charge-conserving by setting
        ``primary_edge_model="erf"``; the underlying solver already supports
        that option.
        """
        sigma_geom = max(float(getattr(self, "primary_soft_cone_sigma_rad", 0.0)), 0.0)
        self._last_primary_mcs_sigma_rad = 0.0
        self._last_primary_mcs_local_thickness_mm = 0.0
        self._last_primary_effective_soft_cone_sigma_rad = sigma_geom
        self._last_primary_mcs_energy_mode = str(getattr(self, "primary_mcs_energy_mode", "initial"))
        self._last_primary_mcs_eval_ke_mev = np.nan

        if not bool(getattr(self, "enable_primary_mcs_smearing", False)):
            return sigma_geom

        # The process model keeps the sharp mean response.  MCS enters only as
        # a correlated covariance/update after the charge fit, so no additional
        # Highland cone width is applied here.
        _mcs_model = str(getattr(self, "primary_mcs_model", "")).strip().lower()
        if _mcs_model in {
            "fermi_eyges_process", "fermi-eyges-process", "fe_process",
            "process", "gee",
        }:
            return sigma_geom

        L_visible = max(float(getattr(self, "length", 0.0)), 1e-9)
        L_stop = max(float(getattr(self, "range_to_threshold_mm", L_visible)), 1e-9)

        # Use the physical photocathode radius, not the enlarged collapse radius
        # with reflector padding.  This estimates the source patch subtended by a
        # real PMT face along the Cherenkov cone.
        if wcd is not None:
            try:
                pmt_radius_mm = float(wcd.mpmts[0].pmts[0].get_properties("design")["size"]) / 2.0
            except Exception:
                pmt_radius_mm = float(getattr(self, "primary_mcs_pmt_radius_mm", 37.0))
        else:
            pmt_radius_mm = float(getattr(self, "primary_mcs_pmt_radius_mm", 37.0))

        def _theta_tan_and_ell(K):
            K = float(K)
            if not np.isfinite(K) or K <= 0.0:
                return None
            gamma = 1.0 + K / float(self.particle_mass)
            beta2 = max(0.0, 1.0 - 1.0 / max(gamma * gamma, 1e-30))
            beta = math.sqrt(beta2)
            if float(self.n) * beta <= 1.0:
                return None
            theta_c = math.acos(max(-1.0, min(1.0, 1.0 / (float(self.n) * beta))))
            tan_tc = max(math.tan(theta_c), 1e-6)
            if bool(getattr(self, "primary_mcs_use_local_pmt_patch", True)):
                ell = pmt_radius_mm / tan_tc
            else:
                ell = L_visible
            ell = max(float(getattr(self, "primary_mcs_min_thickness_mm", 1.0)), ell)
            max_frac = float(getattr(self, "primary_mcs_max_thickness_fraction", 0.25))
            if max_frac > 0.0:
                ell = min(ell, max_frac * max(L_visible, 1.0))
            return theta_c, tan_tc, ell

        def _sigma_for_K(K):
            vals = _theta_tan_and_ell(K)
            if vals is None:
                return 0.0, 0.0
            _theta_c, _tan_tc, ell = vals
            sig = _highland_projected_mcs_sigma_rad(
                K,
                float(self.particle_mass),
                ell,
                radiation_length_mm=float(getattr(self, "primary_mcs_radiation_length_mm", 360.8)),
                charge_number=float(getattr(self, "primary_mcs_charge_number", 1.0)),
            )
            cap = float(getattr(self, "primary_mcs_sigma_cap_rad", 0.0))
            if cap > 0.0:
                sig = min(sig, cap)
            return float(sig), float(ell)

        mode = str(getattr(self, "primary_mcs_energy_mode", "initial")).strip().lower()
        if mode in {"k0", "initial", "legacy"}:
            K_eval = float(getattr(self, "interp_E_init", np.nan))
            if not np.isfinite(K_eval) or K_eval <= 0.0:
                try:
                    K_eval = float(self.muon_energy_at_s(0.0, L_stop))
                except Exception:
                    K_eval = 0.0
            sigma_mcs, ell_mm = _sigma_for_K(K_eval)
            self._last_primary_mcs_eval_ke_mev = float(K_eval)

        elif mode in {"mid", "middle", "midpoint", "midtrack"}:
            s_eval = 0.5 * min(L_visible, L_stop)
            K_eval = float(self.muon_energy_at_s(s_eval, L_stop))
            sigma_mcs, ell_mm = _sigma_for_K(K_eval)
            self._last_primary_mcs_eval_ke_mev = float(K_eval)

        elif mode in {"emission", "emission_weighted", "track_weighted", "weighted"}:
            n_samp = max(3, int(getattr(self, "primary_mcs_energy_samples", 24)))
            s_max = min(L_visible, L_stop)
            # Midpoint samples avoid the singular endpoints and better represent
            # a charge integral along the finite visible track.
            ds = s_max / float(n_samp)
            s_vals = (np.arange(n_samp, dtype=np.float64) + 0.5) * ds
            K_vals = np.asarray(self.muon_energy_at_s_array(s_vals, L_stop), dtype=np.float64)
            weights = _cherenkov_weight_from_energy(K_vals, float(self.particle_mass), float(self.n))
            sig2 = []
            ells = []
            good_w = []
            good_K = []
            for K_i, w_i in zip(K_vals, weights):
                if not np.isfinite(w_i) or w_i <= 0.0:
                    continue
                sig_i, ell_i = _sigma_for_K(float(K_i))
                if sig_i > 0.0 and np.isfinite(sig_i):
                    sig2.append(sig_i * sig_i)
                    ells.append(ell_i)
                    good_w.append(float(w_i))
                    good_K.append(float(K_i))
            if len(sig2) == 0:
                sigma_mcs, ell_mm = 0.0, 0.0
                self._last_primary_mcs_eval_ke_mev = np.nan
            else:
                sig2 = np.asarray(sig2, dtype=np.float64)
                good_w = np.asarray(good_w, dtype=np.float64)
                ells = np.asarray(ells, dtype=np.float64)
                good_K = np.asarray(good_K, dtype=np.float64)
                sigma_mcs = float(math.sqrt(np.sum(good_w * sig2) / max(np.sum(good_w), 1e-30)))
                ell_mm = float(np.sum(good_w * ells) / max(np.sum(good_w), 1e-30))
                self._last_primary_mcs_eval_ke_mev = float(np.sum(good_w * good_K) / max(np.sum(good_w), 1e-30))
        elif mode in {
            "bridge", "bridge_weighted", "fermi_eyges_bridge", "fe_bridge"
        }:
            # Let dtheta(u) be an independent projected FE angular increment
            # with variance T(u)du.  If the free straight direction absorbs the
            # Frank--Tamm-weighted mean tangent, an increment at u contributes
            # to the local residual with variance a(u)(1-a(u)), where a(u) is
            # the fraction of visible emission downstream of u.  Thus
            #
            #   sigma_eff^2 = integral T(u) a(u) [1-a(u)] du.
            #
            # This is a true marginal second moment, contains no fitted scale,
            # and remains well behaved for an abruptly terminated track.
            n_samp = max(17, int(getattr(self, "primary_mcs_energy_samples", 24)))
            s_max = min(L_visible, L_stop)
            s_vals = np.linspace(0.0, s_max, n_samp, dtype=np.float64)
            K_vals = np.asarray(
                self.muon_energy_at_s_array(s_vals, L_stop), dtype=np.float64
            )
            weights = _cherenkov_weight_from_energy(
                K_vals, float(self.particle_mass), float(self.n)
            )
            mass = float(self.particle_mass)
            gamma = 1.0 + np.maximum(K_vals, 0.0) / max(mass, 1e-30)
            beta2 = np.maximum(
                1.0 - 1.0 / np.maximum(gamma * gamma, 1.0), 1e-15
            )
            beta = np.sqrt(beta2)
            momentum = np.sqrt(
                np.maximum(K_vals * (K_vals + 2.0 * mass), 1e-30)
            )
            zq = abs(float(getattr(self, "primary_mcs_charge_number", 1.0)))
            X0 = max(
                float(getattr(self, "primary_mcs_radiation_length_mm", 360.8)),
                1e-30,
            )
            scattering_power = (
                13.6 * zq / np.maximum(beta * momentum, 1e-30)
            ) ** 2 / X0

            if s_vals.size < 2 or not np.any(weights > 0.0):
                sigma_mcs, ell_mm = 0.0, 0.0
                self._last_primary_mcs_eval_ke_mev = np.nan
            else:
                ds = np.diff(s_vals)
                emission_intervals = 0.5 * (weights[:-1] + weights[1:]) * ds
                total_emission = float(np.sum(emission_intervals))
                if total_emission <= 0.0 or not np.isfinite(total_emission):
                    sigma_mcs, ell_mm = 0.0, 0.0
                    self._last_primary_mcs_eval_ke_mev = np.nan
                else:
                    sigma2, ell_mm = fermi_eyges_bridge_variance(
                        s_vals, weights, scattering_power
                    )
                    sigma_mcs = math.sqrt(max(sigma2, 0.0))
                    # Do not apply the legacy 25-mrad local-cone cap here.  The
                    # bridge variance is already the fitted-line residual of a
                    # complete FE process, and truncating it selectively removes
                    # the physically largest broadening near threshold.
                    weighted_intervals = 0.5 * (
                        weights[:-1] * K_vals[:-1]
                        + weights[1:] * K_vals[1:]
                    ) * ds
                    self._last_primary_mcs_eval_ke_mev = float(
                        np.sum(weighted_intervals) / total_emission
                    )
        else:
            # Unknown mode: fail safe by preserving legacy behavior rather than
            # silently disabling MCS.
            K_eval = float(getattr(self, "interp_E_init", np.nan))
            if not np.isfinite(K_eval) or K_eval <= 0.0:
                K_eval = float(self.muon_energy_at_s(0.0, L_stop))
            sigma_mcs, ell_mm = _sigma_for_K(K_eval)
            self._last_primary_mcs_eval_ke_mev = float(K_eval)

        sigma_eff = math.sqrt(sigma_geom * sigma_geom + sigma_mcs * sigma_mcs)

        self._last_primary_mcs_sigma_rad = float(sigma_mcs)
        self._last_primary_mcs_local_thickness_mm = float(ell_mm)
        self._last_primary_effective_soft_cone_sigma_rad = float(sigma_eff)
        return float(sigma_eff)

    def _apply_primary_endpoint_gate_fallback(
        self,
        p_locations,
        direction_zs,
        s,
        scale,
        s_b,
        E_b,
        start_pos,
        track_dir,
        dedx_E_grid,
        dedx_grid,
        endpoint_mode,
        endpoint_scope,
        endpoint_aperture_radius,
        track_length,
    ):
        """Non-fused fallback for the root-overlap endpoint model.

        Normal production fits use the numba fused primary kernel.  This Python
        fallback exists for debugging or use_fused_primary=False and mirrors the
        fused endpoint logic as closely as practical.
        """
        s_eff = np.where(scale > 0.0, s_b, s).astype(np.float64, copy=True)
        valid_s = np.zeros_like(scale, dtype=bool)
        tdx, tdy, tdz = (float(track_dir[0]), float(track_dir[1]), float(track_dir[2]))
        L = float(max(float(track_length), 0.0))
        a = float(endpoint_aperture_radius)
        for i in range(scale.size):
            if scale[i] <= 0.0:
                s_eff[i] = 0.0
                continue
            y = p_locations[i] - start_pos
            u_line = float(np.dot(y, track_dir))
            b = y - u_line * track_dir
            rho = float(np.linalg.norm(b))

            Eep = float(E_b[i])
            gamma_ep = 1.0 + Eep / float(self.mu_mass)
            beta2_ep = 1.0 - 1.0 / max(gamma_ep * gamma_ep, 1.0e-30)
            beta_ep = math.sqrt(max(beta2_ep, 0.0))
            if float(self.n) * beta_ep > 1.0:
                cos_tc_ep = 1.0 / (float(self.n) * beta_ep)
                cos_tc_ep = min(1.0, max(-1.0, cos_tc_ep))
                sin2_tc_ep = max(1.0 - cos_tc_ep * cos_tc_ep, 1.0e-18)
                sin_tc_ep = math.sqrt(sin2_tc_ep)
                cot_ep = cos_tc_ep / sin_tc_ep
                dEdx_ep = float(np.interp(Eep, dedx_E_grid, dedx_grid))
                dc_ds_ep = dEdx_ep / (float(self.n) * float(self.mu_mass) * beta_ep ** 3 * gamma_ep ** 3)
                d_cot_ds_ep = dc_ds_ep / (sin2_tc_ep * sin_tc_ep)
            else:
                cot_ep = 0.0
                d_cot_ds_ep = 0.0

            if rho > 1.0e-12 and math.isfinite(cot_ep):
                denom_root = 1.0 + rho * d_cot_ds_ep
                if (not math.isfinite(denom_root)) or denom_root <= 1.0e-12:
                    denom_root = 1.0
                g = (track_dir - cot_ep * b / rho) / denom_root
            else:
                g = track_dir.copy()

            nvec = direction_zs[i]
            n2 = float(np.dot(nvec, nvec))
            if n2 > 1.0e-18:
                g_perp = g - float(np.dot(g, nvec)) / n2 * nvec
            else:
                g_perp = g
            h = a * float(np.linalg.norm(g_perp))
            w_endpoint, s_mean_endpoint = _endpoint_rootdisk_weight_and_mean_s_numba(float(s_eff[i]), L, h, int(endpoint_scope))
            scale[i] *= w_endpoint
            if w_endpoint > 0.0:
                valid_s[i] = True
                if int(endpoint_mode) == 2:
                    s_eff[i] = s_mean_endpoint
                else:
                    s_eff[i] = min(L, max(0.0, s_eff[i]))
            else:
                scale[i] = 0.0
                s_eff[i] = 0.0
        return s_eff, valid_s, scale


    def get_expected_pes_ts(
        self,
        wcd,
        s,
        p_locations,
        direction_zs,
        mpmt_types,
        obs_pes,
        need_times=True,
    ):

        """
        Expected PE and first-hit-time model used by the fit.

        The heavy cone-collapse work is delegated to the optimized solver in
        particle_cherenkov_model.py.
        """
        pmt_radius = _get_pmt_radius_cached(wcd) + 20 # Add 20 mm for additional reflector surface area

        p_locations = np.asarray(p_locations, dtype=np.float64)
        direction_zs = np.asarray(direction_zs, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        obs_pes = np.asarray(obs_pes, dtype=np.float64)
        _first_arrival_mode = bool(
            need_times and getattr(self, "use_first_arrival_timing", True)
        )
        # Event support is intentionally recomputed on every call.  An earlier
        # optimization cached it by id(obs_pes), which is unsafe when event
        # arrays are created/destroyed or a reusable buffer is filled in place:
        # Python can recycle object IDs and the positive-PMT support can change.
        # The flatnonzero pass costs only O(10 microseconds) for WCTE and avoids
        # a silent cross-event correctness failure.
        _timing_active = np.ascontiguousarray(
            np.flatnonzero(np.isfinite(obs_pes) & (obs_pes > 0.0)), dtype=np.int32
        )

        n_pmts = s.size

        # Convert mPMT type strings to integer codes once per geometry object.
        # WCSim deliberately passes mpmt_types=None (no real-data relative-mPMT
        # correction).  A scalar np.asarray(None) must never reach the fused
        # primary Numba kernel: indexing that 0-D array as mpmt_codes[i] performs
        # an unchecked out-of-bounds read and makes the primary prediction
        # process-dependent.  Represent the no-correction case explicitly by one
        # -1 code per PMT, matching the already-correct delta path below.
        geom_key = (id(mpmt_types), np.shape(mpmt_types), int(n_pmts))
        if geom_key == self._last_geometry_cache_key and self._last_mpmt_type_codes is not None:
            mpmt_codes = self._last_mpmt_type_codes
        else:
            if mpmt_types is None:
                mpmt_codes = np.full(n_pmts, -1, dtype=np.int16)
            else:
                mpmt_codes = np.asarray(_encode_mpmt_types(mpmt_types), dtype=np.int16)
                if mpmt_codes.ndim != 1 or mpmt_codes.size != n_pmts:
                    mpmt_codes = np.broadcast_to(mpmt_codes, (n_pmts,)).astype(
                        np.int16, copy=False
                    )
            mpmt_codes = np.ascontiguousarray(mpmt_codes, dtype=np.int16)
            self._last_geometry_cache_key = geom_key
            self._last_mpmt_type_codes = mpmt_codes

        start_pos = np.asarray(self.start_coord, dtype=np.float64)
        track_dir = np.asarray(self.direction, dtype=np.float64)
        track_dir = track_dir / np.linalg.norm(track_dir)

        scale = np.zeros(n_pmts, dtype=np.float64)
        s_b = np.zeros(n_pmts, dtype=np.float64)
        E_b = np.zeros(n_pmts, dtype=np.float64)

        collapse_mask = s > -200.0
        idx = np.flatnonzero(collapse_mask)

        if idx.size:
            scale_sub, s_b_sub, E_b_sub = find_scale_for_pmts(
                pmt_pos=p_locations[idx],
                start_pos=start_pos,
                track_dir=track_dir,
                s_a_mm=0.001,
                s_max_mm=self.length,
                theta_c_func=theta_c_func,
                range_stop_mm=float(getattr(self, "range_to_threshold_mm", self.length)),
                energy_distance_scale=float(
                    getattr(self, "stopping_range_coordinate_scale", 1.0)
                ),
                n_scan=150,
                near_cross_tol=float(self.effective_primary_soft_cone_sigma_rad(wcd)),
                edge_model=str(getattr(self, "primary_edge_model", "legacy")),
                particle=self.particle_name,
                particle_mass=float(self.particle_mass),
                n_water=float(self.n),
                subgrid_refine=bool(getattr(self, "use_subgrid_refine", True)),
                legacy_grid=not bool(getattr(self, "smooth_tables", True)),
            )

            scale[idx] = scale_sub
            s_b[idx] = s_b_sub
            E_b[idx] = E_b_sub

        use_collapse = scale > 0.0

        _primary_endpoint_model = str(getattr(self, "primary_endpoint_model", "root_overlap_weight_only")).strip().lower()
        if _primary_endpoint_model in {"root_overlap_weight_only", "rootdisk_weight_only", "disk_weight_only", "finite_aperture_root"}:
            _primary_endpoint_mode = 1
        elif _primary_endpoint_model in {"root_overlap_weight_mean", "rootdisk_weight_mean", "disk_weight_mean", "finite_aperture_root_mean"}:
            _primary_endpoint_mode = 2
        else:
            _primary_endpoint_mode = 0
        _primary_endpoint_scope_name = str(getattr(self, "primary_endpoint_scope", "start")).strip().lower()
        if _primary_endpoint_scope_name == "end":
            _primary_endpoint_scope = 1
        elif _primary_endpoint_scope_name == "both":
            _primary_endpoint_scope = 2
        else:
            _primary_endpoint_scope = 0
        _primary_endpoint_aperture_radius = float(getattr(self, "primary_endpoint_aperture_radius_mm", 45.0))
        _primary_endpoint_track_length = float(max(float(self.length), 0.0))
        Eg_dedx, dg_dedx = _get_particle_stopping_power_table(self.particle_name)

        if getattr(self, "use_fused_primary", True) and getattr(self, "use_analytic_primary_ngeo", True):
            # Single-pass fused primary path: collapses ~15 full-length numpy
            # operations into one compiled loop with no intermediate arrays.
            # Numerically equivalent to the vectorized path below to ~1e-12.
            _ngeo_norm = self.primary_ngeo_normalization()
            _rel_stack = _get_rel_eff_stack()
            mu_primary, t_primary_fused = _fused_primary_kernel(
                np.ascontiguousarray(p_locations, dtype=np.float64),
                np.ascontiguousarray(direction_zs, dtype=np.float64),
                np.ascontiguousarray(s, dtype=np.float64),
                np.ascontiguousarray(scale, dtype=np.float64),
                np.ascontiguousarray(s_b, dtype=np.float64),
                np.ascontiguousarray(E_b, dtype=np.float64),
                np.ascontiguousarray(start_pos, dtype=np.float64),
                np.ascontiguousarray(track_dir, dtype=np.float64),
                np.ascontiguousarray(mpmt_codes, dtype=np.int16),
                np.ascontiguousarray(_rel_stack, dtype=np.float64),
                np.ascontiguousarray(Eg_dedx, dtype=np.float64),
                np.ascontiguousarray(dg_dedx, dtype=np.float64),
                float(_ngeo_norm),
                float(self.primary_ngeo_pmt_radius_mm),
                float(pmt_radius),
                int(_primary_endpoint_mode),
                int(_primary_endpoint_scope),
                float(_primary_endpoint_aperture_radius),
                float(_primary_endpoint_track_length),
                float(self.n),
                float(self.mu_mass),
                float(self.intensity),
                float(self.starting_time),
                float(self.v),
                float(self.c),
                bool(need_times and not _first_arrival_mode),
                float(getattr(self, "primary_cost_soft", 0.0)),
                int(1 if getattr(self, "primary_cost_soft_centered", False) else 0),
                int(1 if getattr(self, "enable_wcte_cds_occlusion", False) else 0),
                float(getattr(self, "wcte_cds_axis_x_mm", DEFAULT_WCTE_CDS_AXIS_X_MM)),
                float(getattr(self, "wcte_cds_axis_z_mm", DEFAULT_WCTE_CDS_AXIS_Z_MM)),
                float(getattr(self, "wcte_cds_inner_radius_mm", DEFAULT_WCTE_CDS_INNER_RADIUS_MM)),
                float(getattr(self, "wcte_cds_outer_radius_mm", DEFAULT_WCTE_CDS_OUTER_RADIUS_MM)),
                float(getattr(self, "wcte_cds_y_min_mm", DEFAULT_WCTE_CDS_Y_MIN_MM)),
                float(getattr(self, "wcte_cds_y_max_mm", DEFAULT_WCTE_CDS_Y_MAX_MM)),
                float(getattr(self, "wcte_cds_pmt_aperture_radius_mm", DEFAULT_WCTE_CDS_PMT_APERTURE_RADIUS_MM)),
            )
            t_primary = t_primary_fused if bool(need_times and not _first_arrival_mode) else None
        else:
            s_eff = np.where(use_collapse, s_b, s)

            if _primary_endpoint_mode > 0:
                s_eff, valid_s, scale = self._apply_primary_endpoint_gate_fallback(
                    p_locations,
                    direction_zs,
                    s,
                    scale,
                    s_b,
                    E_b,
                    start_pos,
                    track_dir,
                    np.ascontiguousarray(Eg_dedx, dtype=np.float64),
                    np.ascontiguousarray(dg_dedx, dtype=np.float64),
                    int(_primary_endpoint_mode),
                    int(_primary_endpoint_scope),
                    float(_primary_endpoint_aperture_radius),
                    float(_primary_endpoint_track_length),
                )
            else:
                front_mask = s_eff < pmt_radius
                if np.any(front_mask):
                    scale[front_mask] *= (s_eff[front_mask] + pmt_radius) / (2.0 * pmt_radius)

                valid_s = s_eff >= -pmt_radius
                scale *= valid_s
                s_eff = np.where(valid_s, s_eff, 0.0)

            e_pos = start_pos[None, :] + s_eff[:, None] * track_dir[None, :]
            dx = p_locations[:, 0] - e_pos[:, 0]
            dy = p_locations[:, 1] - e_pos[:, 1]
            dz = p_locations[:, 2] - e_pos[:, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz) + 0.01

            cost = -(dx * direction_zs[:, 0] + dy * direction_zs[:, 1] + dz * direction_zs[:, 2]) / r
            valid_cost = np.isfinite(cost) & (cost > 0.0)
            scale *= valid_cost

            active = (scale > 0.0) & valid_cost

            pwr_corr = np.zeros(n_pmts, dtype=np.float64)
            if np.any(active):
                pwr_corr[active] = self.power_law(cost[active])

            corr = np.zeros(n_pmts, dtype=np.float64)
            if np.any(active):
                if getattr(self, "use_analytic_primary_ngeo", True):
                    corr[active] = self.primary_ngeo_falloff(E_b[active], r[active]) * pwr_corr[active]
                else:
                    corr[active] = n_from_E_r(E_b[active], r[active]) * pwr_corr[active]

            rel_mpmt_scaling = _interp_rel_mpmt_eff_from_codes(
                cost,
                mpmt_codes,
                fill_empty=1.0,
            )

            cds_visibility = np.ones(n_pmts, dtype=np.float64)
            if getattr(self, "enable_wcte_cds_occlusion", False):
                _cds_axis_x = float(getattr(self, "wcte_cds_axis_x_mm", DEFAULT_WCTE_CDS_AXIS_X_MM))
                _cds_axis_z = float(getattr(self, "wcte_cds_axis_z_mm", DEFAULT_WCTE_CDS_AXIS_Z_MM))
                _cds_rin = float(getattr(self, "wcte_cds_inner_radius_mm", DEFAULT_WCTE_CDS_INNER_RADIUS_MM))
                _cds_rout = float(getattr(self, "wcte_cds_outer_radius_mm", DEFAULT_WCTE_CDS_OUTER_RADIUS_MM))
                _cds_ymin = float(getattr(self, "wcte_cds_y_min_mm", DEFAULT_WCTE_CDS_Y_MIN_MM))
                _cds_ymax = float(getattr(self, "wcte_cds_y_max_mm", DEFAULT_WCTE_CDS_Y_MAX_MM))
                _cds_aperture = float(getattr(self, "wcte_cds_pmt_aperture_radius_mm", DEFAULT_WCTE_CDS_PMT_APERTURE_RADIUS_MM))
                for _i in np.flatnonzero(active):
                    cds_visibility[_i] = annular_cylinder_aperture_visibility_numba(
                        float(e_pos[_i, 0]), float(e_pos[_i, 1]), float(e_pos[_i, 2]),
                        float(p_locations[_i, 0]), float(p_locations[_i, 1]), float(p_locations[_i, 2]),
                        float(direction_zs[_i, 0]), float(direction_zs[_i, 1]), float(direction_zs[_i, 2]),
                        _cds_axis_x, _cds_axis_z, _cds_rin, _cds_rout,
                        _cds_ymin, _cds_ymax, _cds_aperture,
                    )

            mu_primary = self.intensity * corr * scale * rel_mpmt_scaling * cds_visibility

            if bool(need_times):
                t_light_primary = r * self.n / self.c
                t_emitter_primary = s_eff / self.v
                t_primary = self.starting_time + t_emitter_primary + t_light_primary
            else:
                t_primary = None

        mu_delta = None
        t_delta = None
        _delta_source_node_mu = None
        _delta_source_s = None

        if self.enable_delta_e and self.delta_e_scale != 0.0:
            _need_delta_nodes = _first_arrival_mode
            _delta_source_node_t = None
            if _need_delta_nodes:
                mu_delta, _delta_source_node_mu, _delta_source_node_t, _delta_source_s = (
                    self.get_delta_e_expected_pes(
                        p_locations=p_locations,
                        direction_zs=direction_zs,
                        start_pos=start_pos,
                        track_dir=track_dir,
                        mpmt_types=mpmt_codes,
                        return_times=False,
                        return_source_nodes=True,
                        source_node_active_indices=_timing_active,
                        return_source_node_times=True,
                    )
                )
            elif bool(need_times) and getattr(self, "use_delta_e_timing", True):
                mu_delta, t_delta = self.get_delta_e_expected_pes(
                        p_locations=p_locations,
                        direction_zs=direction_zs,
                        start_pos=start_pos,
                        track_dir=track_dir,
                        mpmt_types=mpmt_codes,
                        return_times=True,
                    )
            else:
                mu_delta = self.get_delta_e_expected_pes(
                        p_locations=p_locations,
                        direction_zs=direction_zs,
                        start_pos=start_pos,
                        track_dir=track_dir,
                        mpmt_types=mpmt_codes,
                        return_times=False,
                    )
            mu_delta_scaled = self.delta_e_scale * mu_delta
            if _delta_source_node_mu is not None:
                _delta_source_node_mu = (
                    float(self.delta_e_scale)
                    * np.asarray(_delta_source_node_mu, dtype=np.float64)
                )
        else:
            mu_delta_scaled = np.zeros_like(mu_primary)

        # Molecular photon scattering. The physical first-interaction transport
        # depletes direct primary light and adds Rayleigh/Raman first interactions
        # in the same raw optical units.
        _photon_scatter_prediction = None
        _photon_scatter_nodes = None
        _photon_scatter_config = None
        _photon_scatter_charge = np.zeros_like(mu_primary)
        _direct_molecular_survival = np.ones_like(mu_primary)
        _direct_surviving_group_index = np.full_like(mu_primary, _WCTE_REFLECTION_GROUP_INDEX, dtype=np.float64)
        _direct_molecular_path_mm = np.zeros_like(mu_primary)
        _direct_molecular_source_s = np.zeros_like(mu_primary)
        _photon_scatter_model = str(getattr(self, "photon_scatter_model", DEFAULT_PHOTON_SCATTER_MODEL)).strip().lower()
        _use_first_interaction_transport = bool(
            getattr(self, "enable_rayleigh_scatter", False)
            and _photon_scatter_model in {"first_interaction", "first-interaction", "transport", "molecular", "physical", "rayleigh_raman", "rayleigh+raman"}
        )
        if _use_first_interaction_transport:
            _photon_scatter_config = _photon_scatter_transport_config(self)
            if bool(getattr(self, "photon_scatter_direct_survival", True)):
                (_direct_molecular_survival, _direct_surviving_group_index,
                 _direct_molecular_path_mm, _direct_molecular_source_s) = _primary_direct_molecular_state(
                    self, p_locations, start_pos, track_dir, s, scale, s_b, E_b, _photon_scatter_config
                )
                mu_primary = mu_primary * _direct_molecular_survival
            _photon_scatter_prediction, _photon_scatter_nodes, _ = _evaluate_photon_scatter_transport(
                self, wcd, p_locations, direction_zs, start_pos, track_dir,
                timing_active_indices=(_timing_active if _first_arrival_mode else None),
                charge_active_indices=_timing_active,
            )
            _photon_scatter_charge = np.asarray(_photon_scatter_prediction.charge, dtype=np.float64)

        # Charge marginal before detector-surface reflections: surviving direct
        # plus delta and molecular scatter.
        mean_pes_raw = mu_primary + mu_delta_scaled + _photon_scatter_charge

        # Polished central-deployment-system reflection.  Direct primary rays
        # blocked by the finite CDS annulus have already been removed above.
        # This source restores the physically reflected subset in the charge
        # marginal and carries source-resolved first-arrival timing nodes.
        mu_cds_reflection = np.zeros_like(mu_primary)
        t_cds_reflection = None
        _cds_reflection_node_mu = _cds_reflection_node_t = None
        _cds_reflection_needed = bool(
            getattr(self, "enable_wcte_cds_specular_reflection", False)
        )
        if _cds_reflection_needed:
            _want_cds_nodes = bool(
                need_times and getattr(self, "use_first_arrival_timing", True)
            )
            _cds_result = _evaluate_wcte_cds_specular_reflection(
                self,
                p_locations,
                direction_zs,
                active_pmt_indices=(_timing_active if _want_cds_nodes else None),
                return_nodes=_want_cds_nodes,
            )
            if _want_cds_nodes:
                (mu_cds_reflection, t_cds_reflection,
                 _cds_reflection_node_mu, _cds_reflection_node_t) = _cds_result
            else:
                mu_cds_reflection, t_cds_reflection = _cds_result
            mean_pes_raw = mean_pes_raw + mu_cds_reflection

        # Fast parameter-dependent one-bounce blacksheet reflection.  The static
        # detector-to-PMT transfer is built once and cached; only macro-patch
        # illumination is recomputed in the FCN.
        reflection_cache = None
        mu_reflection = np.zeros_like(mu_primary)
        t_reflection = None
        _reflection_node_mu = _reflection_node_t = _reflection_node_active = None
        _deferred_reflection = None
        _reflection_needed = (
            getattr(self, "enable_blacksheet_reflection", False)
            and (bool(need_times) or getattr(self, "reflection_in_charge", False))
        )
        if _reflection_needed:
            reflection_cache = _get_wcte_fast_reflection_transfer(
                self, wcd, p_locations, direction_zs
            )
            if bool(need_times) and not getattr(self, "reflection_in_charge", False):
                # Conditional-timing default: unobserved PMTs cannot contribute a
                # timestamp term, so evaluate only observed PMT columns.  This is
                # an exact likelihood-support optimization, not a physics cut.
                _reflection_active_pmts = _timing_active
            else:
                _reflection_active_pmts = None
            _want_reflection_nodes = bool(
                need_times and getattr(self, "use_first_arrival_timing", True)
            )
            # Keep the source identity of the direct row so a coherent curved
            # path can replace that row before the reflection nodes are
            # generated.  Absolute-charge mode still needs reflected charge on
            # every PMT, so evaluate that marginal once while deferring only
            # the active-PMT timing expansion.
            _can_defer = bool(
                _want_reflection_nodes
                and not getattr(self, "store_expected_component_diagnostics", False)
            )
            if _can_defer:
                if getattr(self, "reflection_in_charge", False):
                    mu_reflection, t_reflection = (
                        _evaluate_wcte_fast_reflection(
                            self,
                            reflection_cache,
                            active_pmt_indices=None,
                            return_nodes=False,
                        )
                    )
                _ru, _rtb = _prepare_wcte_fast_reflection_source(
                    self, reflection_cache
                )
                _ractive = np.ascontiguousarray(_timing_active, dtype=np.int32)
                _rpatch = np.ascontiguousarray(
                    np.flatnonzero(np.isfinite(_ru) & (_ru > 0.0)), dtype=np.int16
                )
                _rtr, _rto = _wcte_get_compact_patch_transfer(
                    reflection_cache, _ractive, _rpatch
                )
                _deferred_reflection = (
                    np.ascontiguousarray(_ru[_rpatch], dtype=np.float64),
                    np.ascontiguousarray(_rtb[_rpatch], dtype=np.float64),
                    _rtr, _rto,
                    np.ascontiguousarray(reflection_cache.patch_min_time_offset[_rpatch], dtype=np.float32),
                    np.ascontiguousarray(reflection_cache.patch_max_time_offset[_rpatch], dtype=np.float32),
                    int(getattr(self, "reflection_first_arrival_nodes",
                                DEFAULT_REFLECTION_FIRST_ARRIVAL_NODES)),
                )
            else:
                _reflection_result = _evaluate_wcte_fast_reflection(
                    self, reflection_cache,
                    active_pmt_indices=_reflection_active_pmts,
                    return_nodes=_want_reflection_nodes,
                )
                if _want_reflection_nodes:
                    (mu_reflection, t_reflection, _reflection_node_mu,
                     _reflection_node_t, _reflection_node_active) = _reflection_result
                else:
                    mu_reflection, t_reflection = _reflection_result
                    _reflection_node_mu = _reflection_node_t = _reflection_node_active = None

        # Single-scatter Rayleigh light (opt-in).  Scales off the PRIMARY
        # component (scattering of primary photons; delta-photon scattering is
        # second order) and is added before the per-event normalization.
        if (
            getattr(self, "enable_rayleigh_scatter", False)
            and not _use_first_interaction_transport
        ):
            _mu_sc, _Pbar = _rayleigh_scatter_field(
                self, p_locations, direction_zs, start_pos, track_dir,
                max(float(self.length), 0.0),
            )
            _sc_sum = float(_mu_sc.sum())
            _pr_sum = float(np.sum(mu_primary))
            if _sc_sum > 0.0 and _pr_sum > 0.0 and _Pbar > 0.0:
                mean_pes_raw = mean_pes_raw + _mu_sc * (_Pbar * _pr_sum / _sc_sum)

        # Apply the detector readout response to reflected charge only after all
        # non-blacksheet prompt components are known.  WCSim's SK-I digitizer
        # opens a 144 ns group at the first accepted PE on a PMT.  A later
        # reflected PE contributes to the selected prompt charge only if such a
        # group exists.  With Poisson non-reflected mean mu this opening
        # probability is exactly 1-exp(-mu).  The WCTE dimensions make the
        # one-bounce delay much shorter than 144 ns.  The approximation neglects
        # the small reflection-only subset that can itself start inside the
        # prompt window; no event-derived timing fraction is introduced.
        mean_pes_without_reflection_raw = np.asarray(
            mean_pes_raw, dtype=np.float64
        )
        mu_reflection_charge = np.zeros_like(mu_reflection)
        if getattr(self, "reflection_in_charge", False):
            _reflection_charge_policy = str(getattr(
                self,
                "reflection_charge_policy",
                DEFAULT_REFLECTION_CHARGE_POLICY,
            )).strip().lower().replace("-", "_")
            if _reflection_charge_policy == "unconditional":
                mu_reflection_charge = np.asarray(
                    mu_reflection, dtype=np.float64
                )
            elif _reflection_charge_policy == "prompt_group_gated":
                _normalization_mode = str(getattr(
                    self, "charge_normalization_mode", "event_mean"
                )).strip().lower().replace("-", "_")
                _absolute_scale = getattr(self, "global_charge_scale", None)
                if (
                    _normalization_mode != "global_scale"
                    or _absolute_scale is None
                    or not np.isfinite(float(_absolute_scale))
                    or float(_absolute_scale) <= 0.0
                ):
                    raise ValueError(
                        "prompt_group_gated reflected charge requires a "
                        "positive global absolute-light scale"
                    )
                _open_probability = prompt_group_open_probability(
                    float(_absolute_scale) * mean_pes_without_reflection_raw
                )
                mu_reflection_charge = (
                    np.asarray(mu_reflection, dtype=np.float64)
                    * _open_probability
                )
            else:
                raise ValueError(
                    "reflection_charge_policy must be unconditional or "
                    "prompt_group_gated"
                )
            mean_pes_raw = (
                mean_pes_without_reflection_raw + mu_reflection_charge
            )
        else:
            mean_pes_raw = mean_pes_without_reflection_raw

        # Timing uses the complete reflected source.  Charge gating is a
        # digitizer-selection marginal and must not erase physical photons from
        # the first-arrival source mixture.
        mean_pes_timing_raw = (
            mean_pes_without_reflection_raw
            + np.asarray(mu_reflection, dtype=np.float64)
        )

        # Fail closed on a non-finite optical prediction.  The historical path
        # allowed NaN/inf values to reach ``raw_mean`` and then multiplied them
        # by a normalization factor, producing warnings.  More seriously, the
        # old compiled Poisson kernel could silently skip NaN entries because
        # both ``lam > 0`` and ``lam <= 0`` are false.  Mark the complete
        # hypothesis invalid instead; updated likelihood wrappers return 1e30.
        _obs_array = np.asarray(obs_pes, dtype=np.float64)
        _raw_array = np.asarray(mean_pes_raw, dtype=np.float64)
        _timing_raw_array = np.asarray(mean_pes_timing_raw, dtype=np.float64)
        _prediction_valid = bool(
            _obs_array.shape == _raw_array.shape
            and _timing_raw_array.shape == _raw_array.shape
            and np.all(np.isfinite(_obs_array))
            and np.all(_obs_array >= 0.0)
            and np.all(np.isfinite(_raw_array))
            and np.all(_raw_array >= 0.0)
            and np.all(np.isfinite(_timing_raw_array))
            and np.all(_timing_raw_array >= 0.0)
        )
        if _prediction_valid:
            obs_mean = float(np.mean(_obs_array))
            raw_mean = float(np.mean(_raw_array))
            try:
                norm = self.charge_normalization_factor(
                    raw_mean, obs_mean
                )
            except ValueError:
                _prediction_valid = False
                norm = np.nan
        else:
            obs_mean = np.nan
            raw_mean = np.nan
            norm = np.nan

        if _prediction_valid and np.isfinite(norm) and norm >= 0.0:
            contamination_fraction = 0.0
            contamination_charge = np.zeros_like(_raw_array)
            contamination_model = str(
                getattr(self, "event_mean_contamination_model", "off")
            ).strip().lower().replace("-", "_")
            normalization_mode = str(
                getattr(self, "charge_normalization_mode", "event_mean")
            ).strip().lower().replace("-", "_")
            if (
                normalization_mode == "event_mean"
                and contamination_model == "uniform_profile"
            ):
                contamination_fraction = profile_event_mean_uniform_contamination(
                    _raw_array,
                    _obs_array,
                    max_fraction=float(
                        getattr(
                            self,
                            "event_mean_contamination_max_fraction",
                            DEFAULT_EVENT_MEAN_CONTAMINATION_MAX_FRACTION,
                        )
                    ),
                )
                total_observed = float(np.sum(_obs_array))
                norm *= 1.0 - contamination_fraction
                contamination_charge.fill(
                    contamination_fraction
                    * total_observed
                    / float(max(_raw_array.size, 1))
                )
            mean_pes_unfloored = _raw_array * norm + contamination_charge
            # The contamination component has no source-time prediction.
            mean_pes_timing_unfloored = _timing_raw_array * norm
            mean_pes = np.maximum(mean_pes_unfloored, self.charge_floor_pe)
        else:
            contamination_fraction = np.nan
            contamination_charge = np.full_like(_raw_array, np.nan)
            mean_pes_unfloored = np.full_like(_raw_array, np.nan)
            mean_pes_timing_unfloored = np.full_like(_timing_raw_array, np.nan)
            mean_pes = np.full_like(_raw_array, np.nan)

        # Optional one-shot Fermi--Eyges process Jacobian.  The ordinary charge
        # prediction above remains unchanged.  This branch is invoked only on a
        # temporary Emitter after the sharp charge fit, never inside the broad
        # seed scan or ordinary Minuit FCN.
        self._last_mcs_charge_jacobian = None
        self._last_mcs_basis_explained_fraction = None
        if (
            _prediction_valid
            and bool(getattr(self, "compute_primary_mcs_process_jacobian", False))
        ):
            try:
                from .mcs_process import (
                    build_raw_fe_kl_basis,
                    stable_transverse_basis,
                    primary_charge_jacobian,
                    normalized_charge_jacobian,
                )
            except ImportError:
                from mcs_process import (
                    build_raw_fe_kl_basis,
                    stable_transverse_basis,
                    primary_charge_jacobian,
                    normalized_charge_jacobian,
                )
            _nm = max(1, int(getattr(
                self, "primary_mcs_process_modes_per_plane", 4
            )))
            _ng = max(17, int(getattr(
                self, "primary_mcs_process_grid_points", 41
            )))
            _bs, _bshape, _bslope, _bcurv, _bfrac = build_raw_fe_kl_basis(
                self, _nm, _ng
            )
            _d0, _e1, _e2 = stable_transverse_basis(track_dir)
            _dedx_E, _dedx_S = _get_particle_stopping_power_table(
                self.particle_name
            )
            _master_ke, _master_range = _get_range_from_energy_arrays(
                self.particle_name
            )
            _raw_primary_J = primary_charge_jacobian(
                np.ascontiguousarray(p_locations, dtype=np.float64),
                np.ascontiguousarray(direction_zs, dtype=np.float64),
                np.ascontiguousarray(mu_primary, dtype=np.float64),
                np.ascontiguousarray(s, dtype=np.float64),
                np.ascontiguousarray(scale, dtype=np.float64),
                np.ascontiguousarray(s_b, dtype=np.float64),
                np.ascontiguousarray(E_b, dtype=np.float64),
                np.ascontiguousarray(start_pos, dtype=np.float64),
                _d0, _e1, _e2, _bs, _bshape, _bslope, _bcurv,
                np.ascontiguousarray(mpmt_codes, dtype=np.int16),
                np.ascontiguousarray(_get_rel_eff_stack(), dtype=np.float64),
                np.ascontiguousarray(_dedx_E, dtype=np.float64),
                np.ascontiguousarray(_dedx_S, dtype=np.float64),
                np.ascontiguousarray(_master_range, dtype=np.float64),
                np.ascontiguousarray(_master_ke, dtype=np.float64),
                float(getattr(self, "range_to_threshold_mm", self.length)),
                float(self.length), float(self.n), float(self.particle_mass),
                float(self.primary_ngeo_pmt_radius_mm),
                int(_primary_endpoint_mode), int(_primary_endpoint_scope),
                float(_primary_endpoint_aperture_radius), float(pmt_radius),
                float(getattr(self, "primary_soft_cone_sigma_rad", 0.0)),
                float(getattr(self, "primary_cost_soft", 0.0)),
                int(1 if getattr(
                    self, "primary_cost_soft_centered", False
                ) else 0),
            )
            # The normalized mean contains primary plus delta charge.  The v1
            # process derivative is the dominant primary-photon contribution;
            # the normalization derivative is evaluated against the complete
            # primary+delta mean so total event charge remains conditioned in
            # exactly the same way as the production likelihood.
            self._last_mcs_charge_jacobian = normalized_charge_jacobian(
                mean_pes_raw, _raw_primary_J, norm, self.charge_floor_pe
            )
            self._last_mcs_basis_explained_fraction = np.asarray(
                _bfrac, dtype=np.float64
            ).copy()

        # Lightweight raw-component cache used by the post-fit coherent MCS
        # continuation.  These arrays already exist in this likelihood call;
        # retaining references avoids two complete duplicate optical evaluations
        # for every finite-difference track while leaving the returned prediction
        # and deferred-reflection path unchanged.
        self._last_expected_pes_raw = _raw_array
        self._last_expected_pes_timing_raw = _timing_raw_array
        self._last_mu_primary_raw = np.asarray(mu_primary, dtype=np.float64)
        # Retain the already-scaled analytic delta component separately.  The
        # coherent FE continuation uses this accepted straight field as its
        # literal zero-bend reference and adds only a curved-minus-straight
        # difference.  No extra optical evaluation or truth information is
        # needed to recover the component here.
        self._last_mu_delta_raw = np.asarray(mu_delta_scaled, dtype=np.float64)
        self._last_direct_molecular_survival = np.asarray(
            _direct_molecular_survival, dtype=np.float64
        )
        self._last_expected_pes_unfloored = mean_pes_unfloored
        self._last_expected_pes_for_timing = mean_pes_timing_unfloored
        self._last_expected_pes_charge = mean_pes
        self._last_expected_pes_norm = float(norm)
        self._last_event_mean_contamination_fraction = float(
            contamination_fraction
        )
        self._last_event_mean_contamination_charge = np.asarray(
            contamination_charge, dtype=np.float64
        )
        self._last_charge_floor_pe = float(self.charge_floor_pe)
        self._last_reflection_raw = mu_reflection
        self._last_reflection_charge_raw = mu_reflection_charge
        self._last_reflection_time = t_reflection
        self._last_wcte_cds_specular_raw = mu_cds_reflection
        self._last_wcte_cds_specular_time = t_cds_reflection

        if getattr(self, "store_expected_component_diagnostics", False):
            sum_primary = float(np.sum(mu_primary))
            sum_delta = float(np.sum(mu_delta_scaled))
            sum_reflection = float(np.sum(mu_reflection))
            sum_reflection_charge = float(np.sum(mu_reflection_charge))
            sum_cds_reflection = float(np.sum(mu_cds_reflection))
            sum_total_charge = float(np.sum(mean_pes_raw))
            self._last_expected_components = {
                "mu_primary_raw": mu_primary.copy(),
                "mu_delta_raw": mu_delta_scaled.copy(),
                "mu_reflection_raw": mu_reflection.copy(),
                "mu_reflection_charge_raw": mu_reflection_charge.copy(),
                "mu_wcte_cds_specular_raw": mu_cds_reflection.copy(),
                "t_wcte_cds_specular_raw_ns": (
                    None if t_cds_reflection is None else t_cds_reflection.copy()
                ),
                "t_reflection_raw_ns": (
                    None if t_reflection is None else t_reflection.copy()
                ),
                "mean_pes_raw": mean_pes_raw.copy(),
                "mean_pes_timing_raw": mean_pes_timing_raw.copy(),
                "mean_pes_unfloored": mean_pes_unfloored.copy(),
                "mean_pes_for_timing": mean_pes_timing_unfloored.copy(),
                "mean_pes": mean_pes.copy(),
                "charge_floor_pe": float(self.charge_floor_pe),
                "norm": float(norm),
                "event_mean_contamination_model": str(
                    getattr(self, "event_mean_contamination_model", "off")
                ),
                "event_mean_contamination_fraction": float(
                    contamination_fraction
                ),
                "event_mean_contamination_charge": contamination_charge.copy(),
                "obs_mean": float(obs_mean),
                "raw_mean": float(raw_mean),
                "sum_primary_raw": sum_primary,
                "sum_delta_raw": sum_delta,
                "sum_reflection_raw": sum_reflection,
                "sum_reflection_charge_raw": sum_reflection_charge,
                "sum_wcte_cds_specular_raw": sum_cds_reflection,
                "wcte_cds_specular_fraction_of_direct_delta_raw": (
                    sum_cds_reflection / max(sum_primary + sum_delta, 1e-300)
                ),
                "wcte_cds_specular_enabled": bool(getattr(
                    self, "enable_wcte_cds_specular_reflection", False
                )),
                "wcte_cds_specular_reflectivity": float(getattr(
                    self, "wcte_cds_specular_reflectivity", np.nan
                )),
                "wcte_cds_specular_surface_patches": int(getattr(
                    self, "_last_wcte_cds_specular_surface_patches", 0
                )),
                "wcte_cds_specular_illuminated_patches": int(getattr(
                    self, "_last_wcte_cds_specular_illuminated_patches", 0
                )),
                "wcte_cds_specular_received_patches": int(getattr(
                    self, "_last_wcte_cds_specular_received_patches", 0
                )),
                "reflection_fraction_of_direct_delta_raw": (
                    sum_reflection / max(sum_primary + sum_delta, 1e-300)
                ),
                "reflection_in_charge": bool(getattr(self, "reflection_in_charge", False)),
                "reflection_charge_policy": str(getattr(
                    self,
                    "reflection_charge_policy",
                    DEFAULT_REFLECTION_CHARGE_POLICY,
                )),
                "reflection_bsrff": float(getattr(self, "reflection_bsrff", np.nan)),
                "reflection_macro_patches": (
                    int(reflection_cache.n_macro_patches)
                    if reflection_cache is not None else 0
                ),
                "reflection_transfer_memory_bytes": (
                    int(reflection_cache.memory_bytes)
                    if reflection_cache is not None else 0
                ),
                "reflection_transfer_persistent_cache_hit": bool(
                    getattr(reflection_cache, "persistent_cache_hit", False)
                ) if reflection_cache is not None else False,
                "reflection_transfer_persistent_cache_path": (
                    getattr(reflection_cache, "persistent_cache_path", None)
                    if reflection_cache is not None else None
                ),
                "primary_fraction_raw": (
                    sum_primary / sum_total_charge if sum_total_charge > 0.0 else np.nan
                ),
                "primary_ngeo_ref_energy_MeV": float(getattr(self, "_last_primary_ngeo_ref_energy_MeV", np.nan)),
                "primary_ngeo_norm": float(getattr(self, "_last_primary_ngeo_norm", np.nan)),
                "primary_ngeo_raw_ref": float(getattr(self, "_last_primary_ngeo_raw_ref", np.nan)),
                "primary_soft_cone_sigma_geom_rad": float(getattr(self, "primary_soft_cone_sigma_rad", np.nan)),
                "primary_mcs_sigma_rad": float(getattr(self, "_last_primary_mcs_sigma_rad", np.nan)),
                "primary_mcs_local_thickness_mm": float(getattr(self, "_last_primary_mcs_local_thickness_mm", np.nan)),
                "primary_effective_soft_cone_sigma_rad": float(getattr(self, "_last_primary_effective_soft_cone_sigma_rad", np.nan)),
                "primary_mcs_energy_mode": str(getattr(self, "primary_mcs_energy_mode", "initial")),
                "primary_mcs_eval_ke_mev": float(getattr(self, "_last_primary_mcs_eval_ke_mev", np.nan)),
                "primary_edge_model": str(getattr(self, "primary_edge_model", "legacy")),
                "primary_endpoint_model": str(getattr(self, "primary_endpoint_model", "legacy")),
                "primary_endpoint_aperture_radius_mm": float(getattr(self, "primary_endpoint_aperture_radius_mm", np.nan)),
                "primary_endpoint_scope": str(getattr(self, "primary_endpoint_scope", "start")),
            }

        # Construct the conventional 1D expected time only for the legacy
        # mean-time likelihood. The first-arrival path carries its own active-PMT
        # nodes and never consumes this nominal array.
        if _first_arrival_mode:
            t_hits = np.empty_like(mean_pes)
        elif not bool(need_times):
            t_hits = np.empty_like(mean_pes)
        elif (
            self.enable_delta_e and self.delta_e_scale != 0.0
            and getattr(self, "use_delta_e_timing", True) and t_delta is not None
        ):
            t_hits = t_primary.copy()
            valid_delta_time = np.isfinite(t_delta) & (mu_delta_scaled > 0.0)
            denom = mu_primary + mu_delta_scaled
            mix = valid_delta_time & np.isfinite(t_primary) & (denom > 0.0)
            t_hits[mix] = (
                mu_primary[mix] * t_primary[mix]
                + mu_delta_scaled[mix] * t_delta[mix]
            ) / denom[mix]
        else:
            t_hits = t_primary

        if bool(need_times) and (not _first_arrival_mode) and t_cds_reflection is not None:
            base_mu = mu_primary + mu_delta_scaled
            valid_base = np.isfinite(t_hits) & (base_mu > 0.0)
            valid_cds = np.isfinite(t_cds_reflection) & (mu_cds_reflection > 0.0)
            both = valid_base & valid_cds
            denom = base_mu + mu_cds_reflection
            t_hits[both] = (
                base_mu[both] * t_hits[both]
                + mu_cds_reflection[both] * t_cds_reflection[both]
            ) / denom[both]
            cds_only = (~valid_base) & valid_cds
            t_hits[cds_only] = t_cds_reflection[cds_only]

        if bool(need_times) and (not _first_arrival_mode) and t_reflection is not None:
            base_mu = mu_primary + mu_delta_scaled + mu_cds_reflection
            valid_base = np.isfinite(t_hits) & (base_mu > 0.0)
            valid_ref = np.isfinite(t_reflection) & (mu_reflection > 0.0)
            both = valid_base & valid_ref
            denom = base_mu + mu_reflection
            t_hits[both] = (
                base_mu[both] * t_hits[both]
                + mu_reflection[both] * t_reflection[both]
            ) / denom[both]
            ref_only = (~valid_base) & valid_ref
            t_hits[ref_only] = t_reflection[ref_only]

        # The digit timestamp is a first-photoelectron observable, not a
        # charge-weighted mean.  Attach a compact source-resolved timing field to
        # the otherwise normal 1D array.  The updated PMT class consumes these
        # nodes automatically, so batch drivers need no new arguments or imports.
        if bool(need_times) and getattr(self, "use_first_arrival_timing", True):
            active = _timing_active
            if active.size:
                # Direct primary node with the molecular-surviving detected spectrum.
                if (_use_first_interaction_transport and bool(getattr(self, "photon_scatter_direct_survival", True))):
                    _sda = np.asarray(_direct_molecular_source_s[active], dtype=np.float64)
                    _tda = _wcte_integrated_primary_tof_fast(self, _sda)
                    t_direct_active = (
                        float(self.starting_time) + _tda
                        + np.asarray(_direct_surviving_group_index[active], dtype=np.float64)
                        * np.asarray(_direct_molecular_path_mm[active], dtype=np.float64) / 299.792458
                    ).astype(np.float32, copy=False)
                else:
                    _tof_r, _tof_a = _get_particle_tof_antiderivative(self)
                    t_direct_active = _wcte_direct_node_times_active_numba(
                        np.ascontiguousarray(active, dtype=np.int32),
                        np.ascontiguousarray(p_locations, dtype=np.float64),
                        np.ascontiguousarray(start_pos, dtype=np.float64),
                        np.ascontiguousarray(track_dir, dtype=np.float64),
                        np.ascontiguousarray(s, dtype=np.float64),
                        np.ascontiguousarray(scale, dtype=np.float64),
                        np.ascontiguousarray(s_b, dtype=np.float64),
                        float(self.length),
                        np.ascontiguousarray(_tof_r, dtype=np.float64),
                        np.ascontiguousarray(_tof_a, dtype=np.float64),
                        float(getattr(self, "range_to_threshold_mm", self.length)),
                        float(getattr(self, "stopping_range_coordinate_scale", 1.0)),
                        float(self.starting_time),
                        float(_WCTE_REFLECTION_GROUP_INDEX / 299.792458),
                    )
                node_mu_parts = [np.asarray(mu_primary[active][None, :], dtype=np.float32)]
                node_t_parts = [np.asarray(t_direct_active[None, :], dtype=np.float32)]

                if (
                    self.enable_delta_e and self.delta_e_scale != 0.0
                    and _delta_source_node_mu is not None
                    and _delta_source_s is not None
                ):
                    # Reuse the exact source-by-PMT amplitudes already computed
                    # by the charge model.  This removes the former second pass
                    # through the full analytic delta kernel.
                    dmu = np.asarray(_delta_source_node_mu, dtype=np.float64)
                    dtm = np.asarray(_delta_source_node_t, dtype=np.float64)
                    dsum = np.sum(dmu, axis=0)
                    dtarget = np.asarray(mu_delta_scaled[active], dtype=np.float64)
                    dfac = np.divide(
                        dtarget, dsum, out=np.zeros_like(dtarget),
                        where=(dsum > 0.0),
                    )
                    dmu *= dfac[None, :]
                    dtm = np.where(dmu > 0.0, dtm, np.inf)
                    if dmu.shape[0]:
                        node_mu_parts.append(np.asarray(dmu, dtype=np.float32))
                        node_t_parts.append(np.asarray(dtm, dtype=np.float32))

                if (
                    _photon_scatter_prediction is not None
                    and _photon_scatter_prediction.timing_node_charge is not None
                    and _photon_scatter_prediction.timing_node_time_ns is not None
                ):
                    _smu = np.asarray(_photon_scatter_prediction.timing_node_charge, dtype=np.float32)
                    _st = np.asarray(_photon_scatter_prediction.timing_node_time_ns, dtype=np.float32)
                    if _smu.shape == _st.shape and _smu.shape[1] == active.size:
                        _smu = np.where(np.isfinite(_smu) & (_smu > 0.0), _smu, 0.0).astype(np.float32, copy=False)
                        _st = np.where(_smu > 0.0, _st, np.inf).astype(np.float32, copy=False)
                        if _smu.shape[0]:
                            node_mu_parts.append(_smu)
                            node_t_parts.append(_st)

                if _cds_reflection_node_mu is not None:
                    _cmu = np.asarray(_cds_reflection_node_mu, dtype=np.float32)
                    _ct = np.asarray(_cds_reflection_node_t, dtype=np.float32)
                    if _cmu.shape == _ct.shape and _cmu.shape[1] == active.size:
                        _cmu = np.where(
                            np.isfinite(_cmu) & (_cmu > 0.0), _cmu, 0.0
                        ).astype(np.float32, copy=False)
                        _ct = np.where(_cmu > 0.0, _ct, np.inf).astype(
                            np.float32, copy=False
                        )
                        if _cmu.shape[0]:
                            node_mu_parts.append(_cmu)
                            node_t_parts.append(_ct)

                if _reflection_node_mu is not None:
                    # Default: 24 one-pass global arrival-time quadrature bins.
                    # Each bin preserves its exact analytic reflected PE mass
                    # and PE-weighted mean time. Setting the node count to 0
                    # restores all 192 macro-patch nodes for validation.
                    _rmu = _reflection_node_mu
                    _rt = _reflection_node_t
                    _nref = int(getattr(
                        self, "reflection_first_arrival_nodes",
                        DEFAULT_REFLECTION_FIRST_ARRIVAL_NODES,
                    ))
                    if _nref > 0 and _nref != _rmu.shape[0] and _rmu.shape[0] > _nref:
                        _ro = np.argsort(_rt, axis=0)
                        _rmu = np.take_along_axis(_rmu, _ro, axis=0)
                        _rt = np.take_along_axis(_rt, _ro, axis=0)
                        _rmu, _rt = _wcte_compress_sorted_reflection_nodes_equal_mass(
                            np.ascontiguousarray(_rmu, dtype=np.float32),
                            np.ascontiguousarray(_rt, dtype=np.float32),
                            _nref,
                        )
                    node_mu_parts.append(_rmu)
                    node_t_parts.append(_rt)

                _deferred_base_mu = _deferred_base_t = None
                if _deferred_reflection is not None:
                    _deferred_base_mu = np.ascontiguousarray(
                        np.vstack(node_mu_parts), dtype=np.float32
                    )
                    _deferred_base_t = np.ascontiguousarray(
                        np.vstack(node_t_parts), dtype=np.float32
                    )
                    node_mu = node_t = node_weight = None
                    _first_arrival_eff = float(os.environ.get(
                        "PMT_FIRST_ARRIVAL_OUTPUT_EFFICIENCY", "0.985"
                    ))
                elif _reflection_node_mu is not None:
                    # Direct/delta is tiny and unsorted; reflection bins are
                    # already ordered by their global physical time interval.
                    # Sort the former with insertion sort and merge linearly.
                    base_mu = np.vstack(node_mu_parts[:-1])
                    base_t = np.vstack(node_t_parts[:-1])
                    _first_arrival_eff = float(os.environ.get(
                        "PMT_FIRST_ARRIVAL_OUTPUT_EFFICIENCY", "0.985"
                    ))
                    node_mu, node_weight, node_t = (
                        _wcte_sort_or_merge_timing_nodes_numba(
                            np.ascontiguousarray(base_mu, dtype=np.float32),
                            np.ascontiguousarray(base_t, dtype=np.float32),
                            np.ascontiguousarray(node_mu_parts[-1], dtype=np.float32),
                            np.ascontiguousarray(node_t_parts[-1], dtype=np.float32),
                            np.ascontiguousarray(obs_pes[active], dtype=np.float64),
                            _first_arrival_eff,
                        )
                    )
                else:
                    node_weight = None
                    _first_arrival_eff = None
                    node_mu = np.vstack(node_mu_parts)
                    node_t = np.vstack(node_t_parts)
                    node_mu = np.where(
                        np.isfinite(node_mu) & (node_mu > 0.0), node_mu, 0.0
                    ).astype(np.float32, copy=False)
                    node_t = np.where(node_mu > 0.0, node_t, np.inf).astype(
                        np.float32, copy=False
                    )
                    node_mu, node_t = _wcte_adaptive_sort_timing_nodes(
                        node_mu, node_t, active
                    )
                t_hits = TimingPrediction(
                    t_hits, node_mu=node_mu, node_t=node_t,
                    active_indices=np.asarray(active, dtype=np.int32),
                    node_weight=node_weight,
                    weight_output_efficiency=_first_arrival_eff,
                    deferred_base_mu=_deferred_base_mu,
                    deferred_base_t=_deferred_base_t,
                    reflection_u=(None if _deferred_reflection is None else _deferred_reflection[0]),
                    reflection_tbase=(None if _deferred_reflection is None else _deferred_reflection[1]),
                    reflection_transfer_active=(None if _deferred_reflection is None else _deferred_reflection[2]),
                    reflection_time_offset_active=(None if _deferred_reflection is None else _deferred_reflection[3]),
                    reflection_patch_min_time_offset=(None if _deferred_reflection is None else _deferred_reflection[4]),
                    reflection_patch_max_time_offset=(None if _deferred_reflection is None else _deferred_reflection[5]),
                    reflection_n_bins=(None if _deferred_reflection is None else _deferred_reflection[6]),
                    node_pe_scale=float(norm),
                )

        return mean_pes, t_hits

    def set_primary_endpoint_model(self, model="root_overlap_weight_only", aperture_radius_mm=None, scope=None):
        """Switch the primary endpoint/front-gate model on an existing Emitter.

        Examples
        --------
        emitter.set_primary_endpoint_model("legacy")
        emitter.set_primary_endpoint_model("root_overlap_weight_only", aperture_radius_mm=45.0, scope="start")
        """
        self.primary_endpoint_model = str(model).strip().lower()
        if aperture_radius_mm is not None:
            self.primary_endpoint_aperture_radius_mm = float(aperture_radius_mm)
        if scope is not None:
            self.primary_endpoint_scope = str(scope).strip().lower()
        return self


    @staticmethod
    def get_pmt_placements(event, wcd, place_info):
        """
        Cache-friendly PMT geometry extraction.

        The fitter usually calls this once per detector configuration, so a
        simple straight loop is enough here.
        """
        p_locations = []
        direction_zs = []
        mpmt_slots = []

        for i_mpmt in range(event.n_mpmt):
            if not event.mpmt_status[i_mpmt]:
                continue

            mpmt = wcd.mpmts[i_mpmt]
            if mpmt is None:
                continue

            for i_pmt in range(event.npmt_per_mpmt):
                if not event.pmt_status[i_mpmt][i_pmt]:
                    continue

                pmt = mpmt.pmts[i_pmt]
                if pmt is None:
                    continue

                placement = pmt.get_placement(place_info, wcd)
                p_locations.append(np.asarray(placement["location"], dtype=np.float64))
                direction_zs.append(np.asarray(placement["direction_z"], dtype=np.float64))
                mpmt_slots.append(i_mpmt)

        return np.asarray(p_locations, dtype=np.float64), np.asarray(direction_zs, dtype=np.float64), np.asarray(mpmt_slots, dtype=np.int64)

    def get_cone_can_intersection_points(self,
            r: float,  # cylinder radius
            ht: float, hb: float,  # top and bottom endcap y (ht > hb)
            n: int,  # number of azimuth samples
            flen: float = 0.  # fractional position along cone axis for apex (0=start, 1=end)
    ) -> List[Tuple[float, float, float]]:
        """
        Return n+1 intersection points (last repeats first) of a right circular cone
        (apex at (x0,y0,z0), axis with direction cosines (cx,cy,cz), half-angle q)
        with the finite cylinder (axis = y, radius r, endcaps at y = hb and y = ht).

        For each azimuth ray on the cone, there is exactly one intersection with the
        cylindrical can (your assumption). If the side intersection is outside the
        y-interval, the intersection is on the corresponding endcap.

        Returns: list of (xi, yi, zi), length n+1 with points[0] == points[-1].
        """
        (x0, y0, z0) = self.start_coord + flen * self.length * np.array(self.direction)
        (cx, cy, cz) = self.direction
        q = np.arccos(self.cos_tq)  # half-angle in radians
        if not (0.0 < q < 0.5 * np.pi):
            raise ValueError("Cone half-angle q must be in (0, pi/2) radians.")
        if ht <= hb:
            raise ValueError("Cylinder top ht must be greater than bottom hb.")
        if r <= 0.0:
            raise ValueError("Cylinder radius r must be positive.")
        if n < 3:
            raise ValueError("Number of azimuth samples n must be at least 3.")

        eps = 1e-12

        # Normalize axis c
        c = np.array([cx, cy, cz], dtype=float)
        c_norm = np.linalg.norm(c)
        if c_norm == 0:
            raise ValueError("Axis direction (cx,cy,cz) must be nonzero.")
        c = c / c_norm

        # Build orthonormal basis {u, v, c} with u,v ⟂ c
        # Choose a helper not nearly parallel to c
        helper = np.array([1.0, 0.0, 0.0]) if abs(c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(c, helper)
        u_norm = np.linalg.norm(u)
        if u_norm < eps:
            helper = np.array([0.0, 0.0, 1.0])
            u = np.cross(c, helper)
            u_norm = np.linalg.norm(u)
            if u_norm < eps:
                raise ValueError("Failed to construct basis perpendicular to axis.")
        u = u / u_norm
        v = np.cross(c, u)  # already unit

        # Precompute constants
        cosq = np.cos(q)
        sinq = np.sin(q)
        apex = np.array([x0, y0, z0], dtype=float)

        # Azimuth samples
        theta = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
        ct = np.cos(theta)
        st = np.sin(theta)

        # Generator directions (unit) for each azimuth
        dirs = (cosq * c)[None, :] + (sinq * ct)[:, None] * u[None, :] + (sinq * st)[:, None] * v[None, :]
        dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

        # Quadratic for intersection with infinite cylinder x^2 + z^2 = r^2:
        # a t^2 + b t + c0 = 0 for (x,y,z) = apex + t*dir
        a = dx * dx + dz * dz
        b = 2.0 * (x0 * dx + z0 * dz)
        c0 = x0 * x0 + z0 * z0 - r * r

        # Discriminant and roots (vectorized)
        disc = np.maximum(0.0, b * b - 4.0 * a * c0)  # clamp tiny negatives to 0
        sqrt_disc = np.sqrt(disc)
        denom = 2.0 * a

        # Two roots; pick the smallest positive t
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom

        # Mask out non-forward intersections; choose min positive
        t_candidates = np.stack([np.where(t1 > eps, t1, np.inf),
                                 np.where(t2 > eps, t2, np.inf)], axis=0)
        t_side = np.min(t_candidates, axis=0)

        # Side hit position
        x_side = x0 + t_side * dx
        y_side = y0 + t_side * dy
        z_side = z0 + t_side * dz

        # Decide final intersection:
        # - if hb <= y_side <= ht: keep side hit
        # - if y_side < hb: snap to bottom cap at y=hb
        # - if y_side > ht: snap to top cap at y=ht
        y_plane = np.where(y_side < hb, hb, np.where(y_side > ht, ht, np.nan))

        # For plane hits, recompute t from y = y_plane
        # Guard against |dy| ~ 0 by nudging with eps; your uniqueness guarantee
        # implies this division should be safe, but we keep it numerically stable.
        dy_safe = np.where(np.abs(dy) < eps, np.sign(dy) * eps, dy)
        t_cap = (y_plane - y0) / dy_safe  # valid only where y_plane is not nan

        # Compute cap positions
        x_cap = x0 + t_cap * dx
        z_cap = z0 + t_cap * dz

        # Choose between side and cap per-ray
        use_cap = ~np.isnan(y_plane)
        xi = np.where(use_cap, x_cap, x_side)
        yi = np.where(use_cap, y_plane, y_side)
        zi = np.where(use_cap, z_cap, z_side)

        # Stack and append first point to close the loop
        pts = np.stack([xi, yi, zi], axis=1)
        pts_closed = np.vstack([pts, pts[:1]])

        # Convert to list of tuples
        return [tuple(row) for row in pts_closed]
