"""Ground-up absolute-light estimate for the WCTE 3-inch PMT system.

This module deliberately contains no fitted light-yield parameter and never
opens WCSim or WCTE event data.  It converts a wavelength-resolved
Frank--Tamm photon estimate into an accepted prompt-PE estimate using detector
geometry, water/gel transmission, the R14374_WCTE QE curve, and collection
efficiency.  The WCSim variant also applies the SK-I prompt integration-group
response; the WCTE engineering estimate retains its detector-specific legacy
charge policy because real WCTE data do not use the WCSim digitizer.

The result is an engineering estimate, not a substitute for an in-situ WCTE
calibration.  Its dominant limitations are the representative optical path,
the primary-track approximation to the prompt readout window, and uncertainty
in the installed PMT/QE population.  Those limitations are recorded in the
returned metadata rather than hidden in a fitted correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


GROUND_UP_MODEL_VERSION = 2
WCSIM_SOURCE_COMMIT = "bc5ca65893ee10dc42259ec541690ec09b15facb"
REFERENCE_RESPONSE_FILE = (
    Path(__file__).resolve().parents[1]
    / "tables"
    / "ground_up_reference_response_v2.npz"
)
REFERENCE_RESPONSE_RELATIVE = "tables/ground_up_reference_response_v2.npz"

# Reference optical calculation for a 300 MeV stopping mu- in the packaged
# WCTE-like geometry.  Photon production is the wavelength-resolved
# Frank--Tamm integral over 200--700 nm with the WCSim water dispersion.  The
# all-charged estimate is retained so the primary/all ratio explicitly removes
# delayed secondary light from the 0--17 ns prompt calibration target.
REFERENCE_KINETIC_ENERGY_MEV = 300.0
PRIMARY_FRANK_TAMM_PHOTONS = 68_018.5079
ALL_CHARGED_FRANK_TAMM_PHOTONS = 85_275.486

# Optical factors are QE-spectrum weighted.  The geometric factor is the
# projected active 3-inch photocathode coverage of the active mPMT apertures;
# the transport factors represent the reference direct water and gel paths.
GEOMETRIC_ACCEPTANCE = 0.1871995814
WATER_TRANSMISSION = 0.9023151
GEL_TRANSMISSION = 0.9790935

# Unboosted wavelength-averaged R14374_WCTE QE times the 95% collection
# efficiency.  WCSim's configured value applies both source-code boosts below.
UNBOOSTED_QE_TIMES_COLLECTION_EFFICIENCY = 0.08348307375
WCSIM_PMT_OBJECT_QE_BOOST = 1.0 / 0.73
WCSIM_SENSITIVE_DETECTOR_QE_BOOST = 1.0 / (1.0 - 0.25)

# Historical unconditional scale-one output.  Version 2 loads the per-PMT
# direct and reflected arrays from REFERENCE_RESPONSE_FILE so the SK-I prompt
# integration-group response can be evaluated rather than treating every
# reflected PE as unconditional prompt charge.
REFERENCE_LICKETYFIT_SCALE_ONE_PE = 314_421.51859832066

# This is a deliberately conservative engineering uncertainty, not a fitted
# nuisance.  It covers representative-path/coverage approximations and the
# primary-only prompt-window approximation.
RELATIVE_SYSTEMATIC_UNCERTAINTY = 0.05


@dataclass(frozen=True)
class GroundUpLightEstimate:
    """One completely resolved mathematical absolute-light estimate."""

    include_wcsim_qe_boosts: bool
    reflection_charge_policy: str
    pmt_object_qe_boost: float
    sensitive_detector_qe_boost: float
    total_qe_boost: float
    qe_times_collection_efficiency: float
    all_charged_detected_pe: float
    prompt_primary_fraction: float
    prompt_detected_pe: float
    licketyfit_scale_one_pe: float
    licketyfit_direct_scale_one_pe: float
    licketyfit_reflection_scale_one_pe: float
    licketyfit_accepted_reflection_scale_one_pe: float
    global_charge_scale: float
    relative_systematic_uncertainty: float

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "model_version": GROUND_UP_MODEL_VERSION,
                "wcsim_source_commit": WCSIM_SOURCE_COMMIT,
                "reference_kinetic_energy_mev": REFERENCE_KINETIC_ENERGY_MEV,
                "primary_frank_tamm_photons": PRIMARY_FRANK_TAMM_PHOTONS,
                "all_charged_frank_tamm_photons": (
                    ALL_CHARGED_FRANK_TAMM_PHOTONS
                ),
                "geometric_acceptance": GEOMETRIC_ACCEPTANCE,
                "water_transmission": WATER_TRANSMISSION,
                "gel_transmission": GEL_TRANSMISSION,
                "unboosted_qe_times_collection_efficiency": (
                    UNBOOSTED_QE_TIMES_COLLECTION_EFFICIENCY
                ),
                "calibration_data_used": False,
                "event_truth_used_in_reconstruction": False,
                "prompt_target_policy": (
                    "all-primary detected light is used as a data-independent "
                    "proxy for total prompt-group light; late primary loss and "
                    "prompt secondary gain are not fitted separately"
                ),
                "prompt_target_proxy_is_exact": False,
                "prompt_target_proxy_uncertainty_included_in_systematic": True,
                "reflection_group_gate_assumption": (
                    "the selected prompt integration group is opened by at "
                    "least one non-reflected prompt PE; reflection-only prompt "
                    "group starts are neglected"
                ),
                "reference_response_file": REFERENCE_RESPONSE_RELATIVE,
                "reference_response_sha256": _file_sha256(
                    REFERENCE_RESPONSE_FILE
                ),
            }
        )
        return payload


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _reference_response() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        with np.load(REFERENCE_RESPONSE_FILE, allow_pickle=False) as payload:
            direct = np.asarray(payload["direct_scale_one"], dtype=np.float64)
            reflection = np.asarray(
                payload["reflection_scale_one"], dtype=np.float64
            )
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"cannot load ground-up reference response: {error}"
        ) from error
    if (
        direct.ndim != 1
        or reflection.shape != direct.shape
        or direct.size == 0
        or np.any(~np.isfinite(direct))
        or np.any(~np.isfinite(reflection))
        or np.any(direct < 0.0)
        or np.any(reflection < 0.0)
    ):
        raise RuntimeError("ground-up reference response arrays are invalid")
    if metadata.get("calibration_event_data_used") is not False:
        raise RuntimeError("ground-up reference response used calibration data")
    return direct, reflection, metadata


def _resolve_reference_scale(
    prompt_detected_pe: float,
    *,
    reflection_charge_policy: str,
) -> tuple[float, float, float, float, dict[str, Any]]:
    direct, reflection, metadata = _reference_response()
    direct_sum = float(np.sum(direct))
    reflection_sum = float(np.sum(reflection))
    policy = str(reflection_charge_policy).strip().lower().replace("-", "_")
    if policy == "unconditional":
        accepted_reflection = reflection_sum
        scale_one = direct_sum + accepted_reflection
        scale = float(prompt_detected_pe) / scale_one
    elif policy == "prompt_group_gated":
        # The gate depends on the physical direct PE mean, hence on the desired
        # absolute scale.  Solve the monotonic response equation without a fit:
        #   N_target = s * sum_i[D_i + R_i(1-exp(-s D_i))].
        lower = 0.0
        upper = max(float(prompt_detected_pe) / max(direct_sum, 1.0e-300), 1e-12)
        while upper * float(
            np.sum(direct + reflection * (-np.expm1(-upper * direct)))
        ) < float(prompt_detected_pe):
            upper *= 2.0
        for _ in range(96):
            scale = 0.5 * (lower + upper)
            predicted = scale * float(
                np.sum(direct + reflection * (-np.expm1(-scale * direct)))
            )
            if predicted < float(prompt_detected_pe):
                lower = scale
            else:
                upper = scale
        scale = 0.5 * (lower + upper)
        accepted_reflection = float(
            np.sum(reflection * (-np.expm1(-scale * direct)))
        )
        scale_one = direct_sum + accepted_reflection
    else:
        raise ValueError(
            "reflection_charge_policy must be unconditional or "
            "prompt_group_gated"
        )
    return scale, scale_one, direct_sum, reflection_sum, {
        **metadata,
        "accepted_reflection_scale_one_sum": accepted_reflection,
    }


def derive_ground_up_light_estimate(
    *,
    include_wcsim_qe_boosts: bool,
    reflection_charge_policy: str | None = None,
) -> GroundUpLightEstimate:
    """Return the deterministic ground-up scale for one QE convention."""

    pmt_boost = WCSIM_PMT_OBJECT_QE_BOOST if include_wcsim_qe_boosts else 1.0
    sd_boost = (
        WCSIM_SENSITIVE_DETECTOR_QE_BOOST
        if include_wcsim_qe_boosts
        else 1.0
    )
    total_boost = pmt_boost * sd_boost
    qe_ce = UNBOOSTED_QE_TIMES_COLLECTION_EFFICIENCY * total_boost
    all_detected = (
        ALL_CHARGED_FRANK_TAMM_PHOTONS
        * GEOMETRIC_ACCEPTANCE
        * WATER_TRANSMISSION
        * GEL_TRANSMISSION
        * qe_ce
    )
    prompt_fraction = (
        PRIMARY_FRANK_TAMM_PHOTONS / ALL_CHARGED_FRANK_TAMM_PHOTONS
    )
    prompt_detected = all_detected * prompt_fraction
    policy = (
        "prompt_group_gated"
        if reflection_charge_policy is None and include_wcsim_qe_boosts
        else (
            "unconditional"
            if reflection_charge_policy is None
            else str(reflection_charge_policy)
        )
    )
    (
        scale,
        scale_one,
        direct_scale_one,
        reflection_scale_one,
        response_metadata,
    ) = _resolve_reference_scale(
        prompt_detected,
        reflection_charge_policy=policy,
    )
    accepted_reflection_scale_one = float(
        response_metadata["accepted_reflection_scale_one_sum"]
    )
    values = (
        total_boost,
        qe_ce,
        all_detected,
        prompt_fraction,
        prompt_detected,
        scale_one,
        direct_scale_one,
        reflection_scale_one,
        accepted_reflection_scale_one,
        scale,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("ground-up absolute-light calculation is invalid")
    return GroundUpLightEstimate(
        include_wcsim_qe_boosts=bool(include_wcsim_qe_boosts),
        reflection_charge_policy=str(policy).strip().lower().replace("-", "_"),
        pmt_object_qe_boost=float(pmt_boost),
        sensitive_detector_qe_boost=float(sd_boost),
        total_qe_boost=float(total_boost),
        qe_times_collection_efficiency=float(qe_ce),
        all_charged_detected_pe=float(all_detected),
        prompt_primary_fraction=float(prompt_fraction),
        prompt_detected_pe=float(prompt_detected),
        licketyfit_scale_one_pe=float(scale_one),
        licketyfit_direct_scale_one_pe=float(direct_scale_one),
        licketyfit_reflection_scale_one_pe=float(reflection_scale_one),
        licketyfit_accepted_reflection_scale_one_pe=float(
            accepted_reflection_scale_one
        ),
        global_charge_scale=float(scale),
        relative_systematic_uncertainty=float(
            RELATIVE_SYSTEMATIC_UNCERTAINTY
        ),
    )


__all__ = [
    "GROUND_UP_MODEL_VERSION",
    "GroundUpLightEstimate",
    "RELATIVE_SYSTEMATIC_UNCERTAINTY",
    "WCSIM_SOURCE_COMMIT",
    "derive_ground_up_light_estimate",
]
