"""Shared, fail-closed configuration rules for the public launchers.

``run_wcsim.py`` and ``run_wcte.py`` deliberately differ in their input and
detector-response configuration.  They must not differ in which reconstruction
models are selectable or in the compatibility rules between fit mode,
likelihood, seeding, and multiple-scattering model.  Keeping those rules here
prevents the two editable launchers from drifting independently.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mcs_configuration import normalize_coherent_implementation


# Public reconstruction vocabulary.  ``full_length`` and ``cosmic`` remain
# accepted as input-only compatibility aliases so existing batch scripts do not
# break, but every resolved configuration and newly written output uses the
# clearer ``beam`` / ``general`` names.
FIT_MODES = frozenset({"beam", "absorption", "general"})
FIT_MODE_ALIASES = {
    "full_length": "beam",
    "cosmic": "general",
}
FIT_PARTICLES = frozenset({"muon", "pion", "kaon", "proton"})
LIKELIHOOD_MODES = frozenset({"charge_only", "charge_time", "timing_only"})
COSMIC_SEED_MODES = frozenset({"off", "additive", "primary", "hybrid", "guided"})

PRIMARY_MCS_MODELS = frozenset(
    {"coherent_fisher", "fermi_eyges_process", "legacy"}
)
COSMIC_MCS_CONTINUATIONS = frozenset(
    {
        "off",
        "linear_fermi_eyges",
        "coherent_fermi_eyges",
        "joint_k0_range_gaussian_fe",
        "joint_k0_range_mixed_mcs",
    }
)
COSMIC_JOINT_INFERENCE_METHODS = frozenset(
    {"auto", "laplace_cubature", "reference_smc"}
)

_PRIMARY_MCS_ALIASES = {
    "coherent_fe": "coherent_fisher",
    "coherent_path": "coherent_fisher",
    "fali_fisher": "coherent_fisher",
    "fe_process": "fermi_eyges_process",
    "process": "fermi_eyges_process",
    "gee": "fermi_eyges_process",
    "cone_broadening": "legacy",
    "local_highland": "legacy",
}
_COSMIC_MCS_ALIASES = {
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
_JOINT_INFERENCE_ALIASES = {
    "laplace": "laplace_cubature",
    "deterministic": "laplace_cubature",
    "smc": "reference_smc",
    "annealed_smc": "reference_smc",
}


def _canonical(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def normalize_fit_mode(value: object) -> str:
    """Return the canonical public fit-mode name.

    Historical names are deliberately aliases only at the configuration
    boundary.  Internal optical endpoint names are a separate implementation
    contract and are not rewritten by this function.
    """
    return _choice(
        value,
        name="FIT_MODE",
        allowed=FIT_MODES,
        aliases=FIT_MODE_ALIASES,
    )


def _choice(
    value: object,
    *,
    name: str,
    allowed: frozenset[str],
    aliases: dict[str, str] | None = None,
) -> str:
    result = _canonical(value)
    if aliases is not None:
        result = aliases.get(result, result)
    if result not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}; got {value!r}")
    return result


@dataclass(frozen=True)
class ReconstructionConfiguration:
    """Canonical shared reconstruction settings and their effective routing."""

    fit_mode: str
    likelihood_mode: str
    configured_seed_mode: str
    seed_mode: str
    primary_mcs_model: str
    coherent_mcs_implementation: str
    configured_cosmic_mcs_continuation: str
    effective_cosmic_mcs_continuation: str
    cosmic_joint_inference_method: str

    def environment(self) -> dict[str, str]:
        """Return the authoritative model selectors consumed by the driver."""
        return {
            "EMITTER_PRIMARY_MCS_MODEL": self.primary_mcs_model,
            "MCS_COHERENT_IMPLEMENTATION": self.coherent_mcs_implementation,
            "EMITTER_COSMIC_MCS_CONTINUATION": (
                self.effective_cosmic_mcs_continuation
            ),
            "EMITTER_COSMIC_JOINT_INFERENCE_METHOD": (
                self.cosmic_joint_inference_method
            ),
        }


def resolve_reconstruction_configuration(
    *,
    fit_mode: object,
    likelihood_mode: object,
    enable_mcs: bool,
    seed_mode: object,
    primary_mcs_model: object,
    coherent_mcs_implementation: object,
    cosmic_mcs_continuation: object,
    cosmic_joint_inference_method: object,
) -> ReconstructionConfiguration:
    """Validate and canonicalize every shared reconstruction-model selector.

    The checks describe actual implementation support.  In particular, every
    general-mode MCS continuation uses charge information, and timing-derived
    general-mode navigation cannot supply information to a charge-only fit.
    Rejecting those
    combinations here avoids silently changing the requested likelihood or
    silently falling back to an unrelated seed bank after costly setup.
    """

    mode = normalize_fit_mode(fit_mode)
    likelihood = _choice(
        likelihood_mode,
        name="LIKELIHOOD_MODE",
        allowed=LIKELIHOOD_MODES,
    )
    configured_seed = _choice(
        seed_mode,
        name="COSMIC_MULTILATERATION_SEED_MODE",
        allowed=COSMIC_SEED_MODES,
    )
    seed = configured_seed if mode == "general" else "off"
    primary = _choice(
        primary_mcs_model,
        name="PRIMARY_MCS_MODEL",
        allowed=PRIMARY_MCS_MODELS,
        aliases=_PRIMARY_MCS_ALIASES,
    )
    coherent_requested = _canonical(coherent_mcs_implementation)
    coherent = (
        "physics_reference" if mode == "beam" else "fast12_profile"
    ) if coherent_requested == "auto" else normalize_coherent_implementation(
        coherent_requested
    )
    configured_cosmic = _choice(
        cosmic_mcs_continuation,
        name="COSMIC_MCS_CONTINUATION",
        allowed=COSMIC_MCS_CONTINUATIONS,
        aliases=_COSMIC_MCS_ALIASES,
    )
    inference = _choice(
        cosmic_joint_inference_method,
        name="COSMIC_JOINT_INFERENCE_METHOD",
        allowed=COSMIC_JOINT_INFERENCE_METHODS,
        aliases=_JOINT_INFERENCE_ALIASES,
    )

    effective_cosmic = (
        configured_cosmic
        if bool(enable_mcs) and mode == "general"
        else "off"
    )
    if inference == "auto":
        inference = (
            "reference_smc"
            if effective_cosmic == "joint_k0_range_mixed_mcs"
            else "laplace_cubature"
        )
    elif (
        effective_cosmic == "joint_k0_range_mixed_mcs"
        and inference != "reference_smc"
    ):
        raise ValueError(
            "COSMIC_MCS_CONTINUATION='joint_k0_range_mixed_mcs' requires "
            "COSMIC_JOINT_INFERENCE_METHOD='auto' or 'reference_smc'"
        )

    if mode == "general":
        if likelihood == "charge_only" and seed != "off":
            raise ValueError(
                "General-mode multilateration/causal seeding uses hit timing and is "
                "not active in charge_only fits; set "
                "COSMIC_MULTILATERATION_SEED_MODE='off'"
            )
        if likelihood == "timing_only" and effective_cosmic != "off":
            raise ValueError(
                f"General-mode MCS continuation {effective_cosmic!r} requires charge; "
                "use charge_only/charge_time or set "
                "COSMIC_MCS_CONTINUATION='off' (or ENABLE_MCS=False)"
            )
    elif bool(enable_mcs):
        if primary == "fermi_eyges_process" and mode != "beam":
            raise ValueError(
                "PRIMARY_MCS_MODEL='fermi_eyges_process' supports only "
                "FIT_MODE='beam'"
            )
        if primary == "coherent_fisher" and (
            mode == "absorption" and coherent != "fast12_profile"
        ):
            raise ValueError(
                "Coherent primary MCS in FIT_MODE='absorption' requires "
                "COHERENT_MCS_IMPLEMENTATION='fast12_profile'; the physics "
                "reference and legacy Fisher continuations assume a threshold "
                "endpoint"
            )
        if likelihood == "timing_only" and primary in {
            "coherent_fisher",
            "fermi_eyges_process",
        }:
            raise ValueError(
                f"PRIMARY_MCS_MODEL={primary!r} requires charge; use "
                "charge_only/charge_time, select PRIMARY_MCS_MODEL='legacy', "
                "or set ENABLE_MCS=False"
            )

    return ReconstructionConfiguration(
        fit_mode=mode,
        likelihood_mode=likelihood,
        configured_seed_mode=configured_seed,
        seed_mode=seed,
        primary_mcs_model=primary,
        coherent_mcs_implementation=coherent,
        configured_cosmic_mcs_continuation=configured_cosmic,
        effective_cosmic_mcs_continuation=effective_cosmic,
        cosmic_joint_inference_method=inference,
    )
