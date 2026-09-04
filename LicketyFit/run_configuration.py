"""Shared, fail-closed configuration rules for the public launchers.

``run_wcsim.py`` and ``run_wcte.py`` deliberately differ in their input and
detector-response configuration.  They must not differ in which reconstruction
models are selectable or in the compatibility rules between fit mode,
likelihood, seeding, and multiple-scattering model.  Keeping those rules here
prevents the two editable launchers from drifting independently.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mcs_configuration import (
    STANDARD_FE_PROCESS,
    normalize_coherent_implementation,
)


# Public reconstruction vocabulary.  Seeding coverage and endpoint physics are
# independent choices.  ``FIT_MODE`` remains an input-only compatibility
# boundary for old launch scripts; new launchers always provide both axes.
SEEDING_MODES = frozenset({"beam", "general"})
INTERACTION_MODES = frozenset({"absorption", "full_length"})
LEGACY_FIT_MODE_AXES = {
    "beam": ("beam", "full_length"),
    "full_length": ("beam", "full_length"),
    "general": ("general", "full_length"),
    "cosmic": ("general", "full_length"),
    # The old absorption engine already enabled its global/orientation seed
    # guards, so its exact compatibility mapping is general + absorption.
    "absorption": ("general", "absorption"),
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


def normalize_seeding_mode(value: object) -> str:
    """Return ``general`` or ``beam`` for the public seed-coverage axis."""
    return _choice(value, name="SEEDING_MODE", allowed=SEEDING_MODES)


def normalize_interaction_mode(value: object) -> str:
    """Return the canonical public optical endpoint model."""
    return _choice(
        value,
        name="INTERACTION_MODE",
        allowed=INTERACTION_MODES,
        aliases={"full": "full_length", "abrupt": "absorption"},
    )


def legacy_fit_mode_axes(value: object) -> tuple[str, str]:
    """Resolve one retired ``FIT_MODE`` value into the two public axes."""
    requested = _canonical(value)
    try:
        return LEGACY_FIT_MODE_AXES[requested]
    except KeyError as exc:
        raise ValueError(
            "legacy FIT_MODE must be one of "
            f"{sorted(LEGACY_FIT_MODE_AXES)}; got {value!r}"
        ) from exc


def public_mode_label(seeding_mode: object, interaction_mode: object) -> str:
    """Stable compatibility label used in filenames and old metadata."""
    seeding = normalize_seeding_mode(seeding_mode)
    interaction = normalize_interaction_mode(interaction_mode)
    return {
        ("general", "full_length"): "general",
        ("beam", "full_length"): "beam",
        ("general", "absorption"): "absorption",
        ("beam", "absorption"): "beam_absorption",
    }[(seeding, interaction)]


def internal_engine_mode(seeding_mode: object, interaction_mode: object) -> str:
    """Return the existing engine selector that exactly implements the pair."""
    seeding = normalize_seeding_mode(seeding_mode)
    interaction = normalize_interaction_mode(interaction_mode)
    if (seeding, interaction) == ("general", "full_length"):
        return "cosmic"
    return interaction


def normalize_fit_mode(value: object) -> str:
    """Compatibility helper returning the stable legacy-style mode label."""
    return public_mode_label(*legacy_fit_mode_axes(value))


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

    seeding_mode: str
    interaction_mode: str
    public_mode_label: str
    internal_engine_mode: str
    likelihood_mode: str
    configured_navigation_mode: str
    navigation_mode: str
    primary_mcs_model: str
    coherent_mcs_implementation: str
    configured_cosmic_mcs_continuation: str
    effective_cosmic_mcs_continuation: str
    cosmic_joint_inference_method: str

    # Read-only compatibility properties for external notebooks written for
    # v1.43.x.  They are intentionally not accepted as new configuration axes.
    @property
    def fit_mode(self) -> str:
        return self.public_mode_label

    @property
    def configured_seed_mode(self) -> str:
        return self.configured_navigation_mode

    @property
    def seed_mode(self) -> str:
        return self.navigation_mode

    def environment(self) -> dict[str, str]:
        """Return the authoritative model selectors consumed by the driver."""
        return {
            "LF_SEEDING_MODE": self.seeding_mode,
            "LF_INTERACTION_MODE": self.interaction_mode,
            "LF_PUBLIC_FIT_MODE": self.public_mode_label,
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
    likelihood_mode: object,
    enable_mcs: bool,
    seed_mode: object,
    primary_mcs_model: object,
    coherent_mcs_implementation: object,
    cosmic_mcs_continuation: object,
    cosmic_joint_inference_method: object,
    seeding_mode: object | None = None,
    interaction_mode: object | None = None,
    fit_mode: object | None = None,
) -> ReconstructionConfiguration:
    """Validate and canonicalize every shared reconstruction-model selector.

    The checks describe actual implementation support.  In particular, every
    general-mode MCS continuation uses charge information, and timing-derived
    general-mode navigation cannot supply information to a charge-only fit.
    Rejecting those
    combinations here avoids silently changing the requested likelihood or
    silently falling back to an unrelated seed bank after costly setup.
    """

    supplied_axes = seeding_mode is not None or interaction_mode is not None
    if supplied_axes:
        if seeding_mode is None or interaction_mode is None:
            raise ValueError(
                "SEEDING_MODE and INTERACTION_MODE must be supplied together"
            )
        if fit_mode is not None:
            raise ValueError(
                "Do not combine legacy FIT_MODE with SEEDING_MODE/INTERACTION_MODE"
            )
        seeding = normalize_seeding_mode(seeding_mode)
        interaction = normalize_interaction_mode(interaction_mode)
    else:
        if fit_mode is None:
            raise ValueError(
                "Supply SEEDING_MODE and INTERACTION_MODE (or one legacy FIT_MODE)"
            )
        seeding, interaction = legacy_fit_mode_axes(fit_mode)
    label = public_mode_label(seeding, interaction)
    engine = internal_engine_mode(seeding, interaction)
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
    general_full_length = (
        seeding == "general" and interaction == "full_length"
    )
    navigation = configured_seed if general_full_length else "off"
    primary = _choice(
        primary_mcs_model,
        name="PRIMARY_MCS_MODEL",
        allowed=PRIMARY_MCS_MODELS,
        aliases=_PRIMARY_MCS_ALIASES,
    )
    coherent_requested = _canonical(coherent_mcs_implementation)
    coherent = (
        STANDARD_FE_PROCESS
        if interaction == "absorption"
        else "physics_reference"
        if seeding == "beam"
        else "fast12_profile"
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
        if bool(enable_mcs) and general_full_length
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

    if general_full_length:
        if likelihood == "charge_only" and navigation != "off":
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
        if primary == "fermi_eyges_process" and interaction != "full_length":
            raise ValueError(
                "PRIMARY_MCS_MODEL='fermi_eyges_process' supports only "
                "INTERACTION_MODE='full_length'"
            )
        if primary == "coherent_fisher" and (
            interaction == "absorption" and coherent != STANDARD_FE_PROCESS
        ):
            raise ValueError(
                "Coherent primary MCS in INTERACTION_MODE='absorption' requires "
                "COHERENT_MCS_IMPLEMENTATION='standard_fe_process'. This is the "
                "analytic abrupt-endpoint Fermi--Eyges process continuation; "
                "the other coherent continuations have different endpoint or "
                "inference semantics."
            )
        if primary == "coherent_fisher" and (
            interaction == "full_length" and coherent == STANDARD_FE_PROCESS
        ):
            raise ValueError(
                "COHERENT_MCS_IMPLEMENTATION='standard_fe_process' is the "
                "abrupt-endpoint absorption continuation and cannot be used "
                "with INTERACTION_MODE='full_length'"
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
        seeding_mode=seeding,
        interaction_mode=interaction,
        public_mode_label=label,
        internal_engine_mode=engine,
        likelihood_mode=likelihood,
        configured_navigation_mode=configured_seed,
        navigation_mode=navigation,
        primary_mcs_model=primary,
        coherent_mcs_implementation=coherent,
        configured_cosmic_mcs_continuation=configured_cosmic,
        effective_cosmic_mcs_continuation=effective_cosmic,
        cosmic_joint_inference_method=inference,
    )
