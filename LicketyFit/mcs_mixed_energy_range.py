"""Continuous blind energy/range inference with mixed coherent MCS paths.

The state is ``K0``, non-centred stopping-range ``z_R``, optional event time
``t0``, standardized soft coordinates, and continuous hard marks.  Energy and
range moves transport the same marks at fixed fractional arc length and include
the exact marked-Poisson prior ratio and Jacobian.  Normalized energy/path/time
guides affect proposals only; their complete density enters the annealed bridge
and Metropolis ratios. Randomized stratified mixture allocation prevents a
finite initialization from accidentally omitting a fitted path basin. Disjoint
energy strata can be run independently and recombined with their exact
broad-prior masses and estimated evidences.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Sequence

import numpy as np

from .mcs_joint_energy_range import (
    EnergyPathGuide,
    _joint_guide_logpdf,
    _normalized_guide_weights,
    _sample_joint_guide,
    _stratified_joint_guide_components,
    _truncated_normal_logpdf,
)
from .mcs_mixed_latent import (
    _hard_base_log_ratio,
    _next_temperature,
    _normalize_log_weights,
    _sample_base_hard_scatters,
    _systematic_resample,
    _standard_normal_logpdf,
)
from .mcs_mixed_path import MixedMCSLatent, MixedMCSPathPrior


@dataclass(frozen=True)
class MixedEnergyRangeContext:
    kinetic_energy_mev: float
    z_range: float
    realized_range_mm: float
    path_prior: MixedMCSPathPrior
    log_likelihood: Callable[[MixedMCSLatent], float]
    log_likelihood_t0: Callable[[MixedMCSLatent, float], float] | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.kinetic_energy_mev))
            or float(self.kinetic_energy_mev) <= 0.0
        ):
            raise ValueError("kinetic energy must be positive and finite")
        if not math.isfinite(float(self.z_range)):
            raise ValueError("z_range must be finite")
        if (
            not math.isfinite(float(self.realized_range_mm))
            or float(self.realized_range_mm) <= 0.0
        ):
            raise ValueError("realized range must be positive and finite")


MixedEnergyRangeContextFactory = Callable[
    [float, float], MixedEnergyRangeContext
]


@dataclass(frozen=True)
class MixedEnergyRangeSMCConfig:
    kinetic_energy_bounds_mev: tuple[float, float]
    particles: int = 32
    target_ess_fraction: float = 0.80
    resample_ess_fraction: float = 0.55
    max_temperatures: int = 64
    rejuvenation_steps: int = 3
    posterior_rejuvenation_steps: int = 12
    energy_random_walk_mev: float = 5.0
    energy_independence_probability: float = 0.05
    event_time_bounds_ns: tuple[float, float] | None = None
    event_time_seed_ns: float = 0.0
    event_time_guide_sd_ns: float = 1.0
    event_time_prior_mixture_probability: float = 0.20
    event_time_random_walk_ns: float = 0.20
    range_pcn_rho: float = 0.80
    soft_pcn_rho: float = 0.995
    soft_guide_probability: float = 0.35
    soft_elliptical_slice: bool = True
    soft_elliptical_max_bracket_shrinks: int = 32
    hard_retention_probability: float = 0.72
    hard_local_position_step_fraction: float = 0.08
    hard_local_log_angle_step: float = 0.35
    hard_local_azimuth_step_rad: float = 0.50
    guide_prior_mixture_probability: float = 0.20
    hard_empty_guide_probability: float = 0.70
    stratified_guide_initialization: bool = True
    direct_importance_sampling: bool = False
    initialization_attempts_per_particle: int = 100
    random_seed: int = 41873

    def validate(self) -> None:
        low, high = map(float, self.kinetic_energy_bounds_mev)
        if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
            raise ValueError("energy bounds must satisfy 0 < low < high")
        if int(self.particles) < 2:
            raise ValueError("SMC requires at least two particles")
        for name, value in (
            ("target_ess_fraction", self.target_ess_fraction),
            ("resample_ess_fraction", self.resample_ess_fraction),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must lie in (0,1]")
        for name, value in (
            ("energy_independence_probability", self.energy_independence_probability),
            ("soft_guide_probability", self.soft_guide_probability),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        for name, value in (
            ("range_pcn_rho", self.range_pcn_rho),
            ("soft_pcn_rho", self.soft_pcn_rho),
            ("hard_retention_probability", self.hard_retention_probability),
            ("hard_empty_guide_probability", self.hard_empty_guide_probability),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must lie in [0,1)")
        if not 0.0 < float(self.guide_prior_mixture_probability) < 1.0:
            raise ValueError("guide_prior_mixture_probability must lie in (0,1)")
        if float(self.energy_random_walk_mev) <= 0.0:
            raise ValueError("energy_random_walk_mev must be positive")
        if self.event_time_bounds_ns is not None:
            t0_low, t0_high = map(float, self.event_time_bounds_ns)
            if not (
                math.isfinite(t0_low)
                and math.isfinite(t0_high)
                and t0_low < t0_high
            ):
                raise ValueError("event-time bounds must be finite and ordered")
            if not math.isfinite(float(self.event_time_seed_ns)):
                raise ValueError("event-time seed must be finite")
            if (
                not math.isfinite(float(self.event_time_guide_sd_ns))
                or float(self.event_time_guide_sd_ns) <= 0.0
            ):
                raise ValueError("event-time guide width must be positive")
            if not 0.0 < float(
                self.event_time_prior_mixture_probability
            ) < 1.0:
                raise ValueError(
                    "event-time prior mixture probability must lie in (0,1)"
                )
            if (
                not math.isfinite(float(self.event_time_random_walk_ns))
                or float(self.event_time_random_walk_ns) <= 0.0
            ):
                raise ValueError("event-time random walk must be positive")
        for name, value in (
            (
                "hard_local_position_step_fraction",
                self.hard_local_position_step_fraction,
            ),
            ("hard_local_log_angle_step", self.hard_local_log_angle_step),
            ("hard_local_azimuth_step_rad", self.hard_local_azimuth_step_rad),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if int(self.max_temperatures) < 1:
            raise ValueError("max_temperatures must be positive")
        if int(self.rejuvenation_steps) < 0:
            raise ValueError("rejuvenation_steps must be non-negative")
        if int(self.posterior_rejuvenation_steps) < 0:
            raise ValueError("posterior_rejuvenation_steps must be non-negative")
        if int(self.soft_elliptical_max_bracket_shrinks) < 1:
            raise ValueError(
                "soft_elliptical_max_bracket_shrinks must be positive"
            )


@dataclass(frozen=True)
class MixedEnergyRangeSMCStage:
    beta_before: float
    beta_after: float
    ess_before_resampling: float
    resampled: bool
    energy_acceptance: float
    event_time_acceptance: float
    range_acceptance: float
    soft_acceptance: float
    soft_pcn_acceptance: float
    conditional_guide_acceptance: float
    hard_acceptance: float
    hard_local_acceptance: float
    soft_pcn_attempts: int = 0
    soft_pcn_finite_proposals: int = 0
    soft_pcn_jump_norm_median: float = math.nan
    soft_pcn_loglike_delta_median: float = math.nan
    soft_pcn_log_ratio_median: float = math.nan
    soft_pcn_log_ratio_max: float = math.nan
    conditional_guide_attempts: int = 0
    conditional_guide_finite_proposals: int = 0
    conditional_guide_jump_norm_median: float = math.nan
    conditional_guide_loglike_delta_median: float = math.nan
    conditional_guide_log_ratio_median: float = math.nan
    conditional_guide_log_ratio_max: float = math.nan
    conditional_guide_effective_probability: float = math.nan

    @property
    def path_acceptance(self) -> float:
        """Compatibility aggregate for Gaussian joint-result consumers."""
        return float(self.soft_acceptance)

    @property
    def guide_acceptance(self) -> float:
        """Compatibility alias for the conditional soft-guide move."""
        return float(self.conditional_guide_acceptance)


@dataclass
class MixedEnergyRangeSMCResult:
    kinetic_energy_mev: np.ndarray
    z_range: np.ndarray
    event_time_ns: np.ndarray
    realized_range_mm: np.ndarray
    latents: tuple[MixedMCSLatent, ...]
    log_likelihood: np.ndarray
    log_prior_density: np.ndarray
    weights: np.ndarray
    initial_lineage: np.ndarray
    stages: tuple[MixedEnergyRangeSMCStage, ...]
    likelihood_evaluations: int
    invalid_likelihood_evaluations: int
    context_builds: int
    invalid_context_builds: int
    log_evidence: float
    random_seed: int
    kinetic_energy_bounds_mev: tuple[float, float]
    posterior_trajectory_sweeps: int = 0
    hard_self_transitions: int = 0
    energy_stratum_diagnostics: tuple[dict, ...] = ()

    def weighted_mean(self, values) -> float:
        return float(np.dot(self.weights, np.asarray(values, dtype=np.float64)))

    def weighted_sd(self, values) -> float:
        array = np.asarray(values, dtype=np.float64)
        mean = self.weighted_mean(array)
        return float(math.sqrt(max(np.dot(self.weights, (array - mean) ** 2), 0.0)))

    @property
    def coefficients(self) -> np.ndarray:
        """Soft coordinates in the shape used by the Gaussian joint result."""
        return np.ascontiguousarray(
            np.stack([latent.soft_coefficients for latent in self.latents])
        )

    @property
    def posterior_log_density(self) -> np.ndarray:
        return np.ascontiguousarray(self.log_likelihood + self.log_prior_density)

    @property
    def uses_discrete_range_grid(self) -> bool:
        return False

    @property
    def unique_realized_ranges(self) -> int:
        return int(np.unique(self.realized_range_mm).size)

    @property
    def unique_soft_path_states(self) -> int:
        return int(np.unique(self.coefficients, axis=0).shape[0])

    @staticmethod
    def _category_effective_sample_size(signatures) -> float:
        """Kish ESS after grouping exact duplicate posterior states.

        This is deliberately a conservative duplicate-state diagnostic, not
        an autocorrelation ESS claim.  It exposes particle cloning that the
        nominal uniform weights of recorded post-resampling sweeps conceal.
        """
        counts = {}
        for signature in signatures:
            counts[signature] = counts.get(signature, 0) + 1
        total = int(sum(counts.values()))
        if total == 0:
            return 0.0
        probabilities = np.fromiter(
            (count / total for count in counts.values()), dtype=np.float64
        )
        return float(1.0 / np.sum(probabilities * probabilities))

    @property
    def soft_state_effective_sample_size(self) -> float:
        return self._category_effective_sample_size(
            tuple(map(float, row)) for row in self.coefficients
        )

    @property
    def unique_hard_path_states(self) -> int:
        return int(len({latent.hard_scatters for latent in self.latents}))

    @property
    def hard_state_effective_sample_size(self) -> float:
        return self._category_effective_sample_size(
            latent.hard_scatters for latent in self.latents
        )

    @property
    def unique_joint_states(self) -> int:
        signatures = {
            (
                float(k0),
                float(z0),
                float(t0),
                tuple(map(float, latent.soft_coefficients)),
                latent.hard_scatters,
            )
            for k0, z0, t0, latent in zip(
                self.kinetic_energy_mev,
                self.z_range,
                self.event_time_ns,
                self.latents,
                strict=True,
            )
        }
        return int(len(signatures))

    @property
    def joint_state_effective_sample_size(self) -> float:
        return self._category_effective_sample_size(
            (
                float(k0),
                float(z0),
                float(t0),
                tuple(map(float, latent.soft_coefficients)),
                latent.hard_scatters,
            )
            for k0, z0, t0, latent in zip(
                self.kinetic_energy_mev,
                self.z_range,
                self.event_time_ns,
                self.latents,
                strict=True,
            )
        )

    @property
    def unique_initial_lineages(self) -> int:
        return int(np.unique(self.initial_lineage).size)

    @property
    def initial_lineage_effective_sample_size(self) -> float:
        return self._category_effective_sample_size(
            map(int, self.initial_lineage)
        )

    @property
    def hard_scatter_count(self) -> np.ndarray:
        return np.asarray(
            [len(latent.hard_scatters) for latent in self.latents], dtype=np.int32
        )

    def summary(self) -> dict:
        energy_mean = self.weighted_mean(self.kinetic_energy_mev)
        energy_sd = self.weighted_sd(self.kinetic_energy_mev)
        range_mean = self.weighted_mean(self.realized_range_mm)
        range_sd = self.weighted_sd(self.realized_range_mm)
        z_mean = self.weighted_mean(self.z_range)
        z_sd = self.weighted_sd(self.z_range)
        t0_mean = self.weighted_mean(self.event_time_ns)
        t0_sd = self.weighted_sd(self.event_time_ns)
        covariance = float(
            np.dot(
                self.weights,
                (self.kinetic_energy_mev - energy_mean)
                * (self.realized_range_mm - range_mean),
            )
        )
        denominator = energy_sd * range_sd
        hard_count = self.hard_scatter_count.astype(np.float64)
        map_index = int(np.argmax(self.posterior_log_density))
        nominal_weight_ess = float(
            1.0 / np.sum(np.asarray(self.weights) ** 2)
        )
        return {
            "initial_kinetic_energy_mev_mean": energy_mean,
            "initial_kinetic_energy_mev_sd": energy_sd,
            "realized_range_mm_mean": range_mean,
            "realized_range_mm_sd": range_sd,
            "z_range_mean": z_mean,
            "z_range_sd": z_sd,
            "event_time_ns_mean": t0_mean,
            "event_time_ns_sd": t0_sd,
            "energy_range_covariance_mev_mm": covariance,
            "energy_range_correlation": (
                covariance / denominator if denominator > 0.0 else math.nan
            ),
            "hard_scatter_count_mean": self.weighted_mean(hard_count),
            "hard_scatter_count_sd": self.weighted_sd(hard_count),
            "posterior_probability_any_hard_scatter": self.weighted_mean(
                hard_count > 0.0
            ),
            "map_sample_index_by_full_density": map_index,
            "map_log_posterior_density": float(
                self.posterior_log_density[map_index]
            ),
            "unique_realized_ranges": self.unique_realized_ranges,
            "posterior_sample_count": int(self.weights.size),
            "nominal_posterior_weight_ess": nominal_weight_ess,
            "posterior_weight_ess_fraction": float(
                nominal_weight_ess / max(int(self.weights.size), 1)
            ),
            "maximum_posterior_weight": float(np.max(self.weights)),
            "unique_soft_path_states": self.unique_soft_path_states,
            "unique_hard_path_states": self.unique_hard_path_states,
            "unique_joint_states": self.unique_joint_states,
            "unique_initial_lineages": self.unique_initial_lineages,
            "soft_state_duplicate_ess": self.soft_state_effective_sample_size,
            "hard_state_duplicate_ess": self.hard_state_effective_sample_size,
            "joint_state_duplicate_ess": self.joint_state_effective_sample_size,
            "initial_lineage_ess": self.initial_lineage_effective_sample_size,
            "duplicate_ess_definition": (
                "Kish ESS of exact-state multiplicities; conservative clone "
                "diagnostic, not autocorrelation ESS"
            ),
            "uses_discrete_range_grid": False,
            "range_coordinate": "continuous_noncentered_float64_z_R",
            "energy_coordinate": "continuous_float64_K0_mev",
            "output_length_quantization_mm": None,
            "likelihood_evaluations": int(self.likelihood_evaluations),
            "invalid_likelihood_evaluations": int(
                self.invalid_likelihood_evaluations
            ),
            "context_builds": int(self.context_builds),
            "invalid_context_builds": int(self.invalid_context_builds),
            "temperature_stages": int(len(self.stages)),
            "log_evidence": float(self.log_evidence),
            "random_seed": int(self.random_seed),
            "kinetic_energy_bounds_mev": list(self.kinetic_energy_bounds_mev),
            "posterior_trajectory_sweeps": int(
                self.posterior_trajectory_sweeps
            ),
            "hard_self_transitions": int(self.hard_self_transitions),
            "energy_stratified": bool(
                len({
                    tuple(row.get("bounds_mev", ()))
                    for row in self.energy_stratum_diagnostics
                }) > 1
            ),
            "energy_stratum_replicated": bool(
                any(
                    int(row.get("replicates", 1)) > 1
                    for row in self.energy_stratum_diagnostics
                )
            ),
            "energy_strata": [
                dict(row) for row in self.energy_stratum_diagnostics
            ],
        }


def _reflected_interval(value: float, low: float, high: float) -> float:
    width = high - low
    coordinate = (float(value) - low) % (2.0 * width)
    return float(
        low + (coordinate if coordinate <= width else 2.0 * width - coordinate)
    )


def _sample_conditional_soft_base(
    energy: float,
    z_range: float,
    guides: Sequence[EnergyPathGuide],
    *,
    low: float,
    high: float,
    prior_mixture_probability: float,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``u`` from the exact base conditional ``q(u | K0, z_R)``.

    Holding the physical global coordinates fixed makes the charge-derived
    path guide useful late in annealing.  The normalizing constant of this
    conditional cancels from the independence Metropolis ratio; the complete
    joint base density is still evaluated by :func:`_joint_guide_logpdf`.
    """
    epsilon = float(prior_mixture_probability)
    component_log_weight = [
        math.log(epsilon)
        - math.log(float(high - low))
        + _standard_normal_logpdf(np.asarray((z_range,), dtype=np.float64))
    ]
    guide_weight = _normalized_guide_weights(guides)
    for guide, normalized_weight in zip(guides, guide_weight, strict=True):
        z_sd = float(guide.z_range_proposal_sd)
        component_log_weight.append(
            math.log1p(-epsilon)
            + math.log(float(normalized_weight))
            + _truncated_normal_logpdf(
                float(energy),
                float(guide.kinetic_energy_mev),
                float(guide.energy_proposal_sd_mev),
                float(low),
                float(high),
            )
            - 0.5
            * ((float(z_range) - float(guide.z_range_mean)) / z_sd) ** 2
            - math.log(z_sd)
            - 0.5 * math.log(2.0 * math.pi)
        )
    log_weight = np.asarray(component_log_weight, dtype=np.float64)
    maximum = float(np.max(log_weight))
    probability = np.exp(log_weight - maximum)
    probability /= float(np.sum(probability))
    component = int(rng.choice(len(probability), p=probability))
    if component == 0:
        return np.ascontiguousarray(rng.normal(size=int(dimension)))
    return guides[component - 1].path.sample(rng)


def _event_time_base_logpdf(
    value: float,
    *,
    low: float,
    high: float,
    seed: float,
    guide_sd: float,
    prior_mixture_probability: float,
) -> float:
    """Normalized defensive proposal density for the event-time nuisance."""
    if not low <= float(value) <= high:
        return -math.inf
    epsilon = float(prior_mixture_probability)
    uniform = -math.log(high - low)
    guide = _truncated_normal_logpdf(
        float(value), float(seed), float(guide_sd), float(low), float(high)
    )
    return float(np.logaddexp(
        math.log(epsilon) + uniform,
        math.log1p(-epsilon) + guide,
    ))


def _sample_event_time_base(
    *,
    low: float,
    high: float,
    seed: float,
    guide_sd: float,
    prior_mixture_probability: float,
    rng: np.random.Generator,
) -> float:
    epsilon = float(prior_mixture_probability)
    if rng.random() < epsilon:
        return float(rng.uniform(low, high))
    # Rejection from a narrow normal is both exact and fast for the ordinary
    # timing bounds.  The defensive uniform branch guarantees complete support.
    for _ in range(10_000):
        value = float(seed + guide_sd * rng.normal())
        if low <= value <= high:
            return value
    raise RuntimeError("failed to sample the truncated event-time guide")


def run_mixed_energy_range_smc(
    context_factory: MixedEnergyRangeContextFactory,
    dimension: int,
    energy_path_guides: Sequence[EnergyPathGuide],
    config: MixedEnergyRangeSMCConfig,
) -> MixedEnergyRangeSMCResult:
    """Anneal a normalized broad guide mixture to the exact blind posterior."""
    config.validate()
    dimension = int(dimension)
    if dimension < 2 or dimension % 2:
        raise ValueError("dimension must contain equal non-empty plane blocks")
    guides = tuple(energy_path_guides)
    if not guides:
        raise ValueError("at least one normalized energy/path guide is required")
    if any(guide.path.dimension != dimension for guide in guides):
        raise ValueError("energy/path guide dimension mismatch")
    low, high = map(float, config.kinetic_energy_bounds_mev)
    fit_event_time = config.event_time_bounds_ns is not None
    if fit_event_time:
        t0_low, t0_high = map(float, config.event_time_bounds_ns)
        t0_seed = float(np.clip(config.event_time_seed_ns, t0_low, t0_high))
    else:
        t0_low = t0_high = t0_seed = float(config.event_time_seed_ns)
    epsilon = float(config.guide_prior_mixture_probability)
    hard_empty_probability = float(config.hard_empty_guide_probability)
    rng = np.random.default_rng(int(config.random_seed))
    count = int(config.particles)
    context_builds = 0
    invalid_context_builds = 0
    likelihood_evaluations = 0
    invalid_likelihood_evaluations = 0
    hard_self_transitions = 0

    def make_context(energy: float, z_value: float):
        nonlocal context_builds, invalid_context_builds
        context_builds += 1
        try:
            context = context_factory(float(energy), float(z_value))
        except Exception:
            invalid_context_builds += 1
            return None
        if context.path_prior.dimension != dimension:
            raise ValueError("context path dimension changed")
        return context

    def evaluate(context, latent, event_time):
        nonlocal likelihood_evaluations, invalid_likelihood_evaluations
        likelihood_evaluations += 1
        try:
            if fit_event_time:
                if context.log_likelihood_t0 is None:
                    raise RuntimeError(
                        "event-time SMC requires a t0-aware likelihood"
                    )
                value = float(
                    context.log_likelihood_t0(latent, float(event_time))
                )
            else:
                value = float(context.log_likelihood(latent))
        except Exception:
            value = -math.inf
        if not math.isfinite(value):
            invalid_likelihood_evaluations += 1
            return -math.inf
        return value

    energy_values: list[float] = []
    z_values: list[float] = []
    event_time_values: list[float] = []
    contexts: list[MixedEnergyRangeContext] = []
    latents: list[MixedMCSLatent] = []
    likelihood_values: list[float] = []
    initialization_components = (
        _stratified_joint_guide_components(
            guides,
            prior_mixture_probability=epsilon,
            count=count,
            rng=rng,
        )
        if bool(config.stratified_guide_initialization)
        else np.full(count, -2, dtype=np.int64)
    )
    for _particle in range(count):
        forced_component = int(initialization_components[_particle])
        for _attempt in range(int(config.initialization_attempts_per_particle)):
            energy, z_value, coefficients = _sample_joint_guide(
                guides,
                low=low,
                high=high,
                prior_mixture_probability=epsilon,
                dimension=dimension,
                rng=rng,
                component_index=(
                    forced_component if forced_component >= -1 else None
                ),
            )
            context = make_context(energy, z_value)
            if context is None:
                continue
            latent = MixedMCSLatent(
                coefficients,
                _sample_base_hard_scatters(
                    context.path_prior, hard_empty_probability, rng
                ),
            )
            event_time = (
                _sample_event_time_base(
                    low=t0_low,
                    high=t0_high,
                    seed=t0_seed,
                    guide_sd=float(config.event_time_guide_sd_ns),
                    prior_mixture_probability=float(
                        config.event_time_prior_mixture_probability
                    ),
                    rng=rng,
                )
                if fit_event_time
                else t0_seed
            )
            value = evaluate(context, latent, event_time)
            if math.isfinite(value):
                energy_values.append(float(energy))
                z_values.append(float(z_value))
                event_time_values.append(float(event_time))
                contexts.append(context)
                latents.append(latent)
                likelihood_values.append(value)
                break
        else:
            raise RuntimeError("failed to initialize a finite blind mixed particle")

    energy = np.asarray(energy_values, dtype=np.float64)
    z_range = np.asarray(z_values, dtype=np.float64)
    event_time = np.asarray(event_time_values, dtype=np.float64)
    loglike = np.asarray(likelihood_values, dtype=np.float64)

    def physical_log_prior(k0, z_value, coefficients, t0_value):
        if not low <= float(k0) <= high:
            return -math.inf
        time_prior = 0.0
        if fit_event_time:
            if not t0_low <= float(t0_value) <= t0_high:
                return -math.inf
            time_prior = -math.log(t0_high - t0_low)
        return float(
            -math.log(high - low)
            + _standard_normal_logpdf(np.asarray((z_value,), dtype=np.float64))
            + _standard_normal_logpdf(coefficients)
            + time_prior
        )

    def base_logpdf(k0, z_value, coefficients, t0_value):
        out = float(
            _joint_guide_logpdf(
                k0,
                z_value,
                coefficients,
                guides,
                low=low,
                high=high,
                prior_mixture_probability=epsilon,
            )
        )
        if fit_event_time:
            out += _event_time_base_logpdf(
                t0_value,
                low=t0_low,
                high=t0_high,
                seed=t0_seed,
                guide_sd=float(config.event_time_guide_sd_ns),
                prior_mixture_probability=float(
                    config.event_time_prior_mixture_probability
                ),
            )
        return float(out)

    log_prior = np.asarray(
        [
            physical_log_prior(k0, z0, latent.soft_coefficients, t0)
            for k0, z0, t0, latent in zip(
                energy, z_range, event_time, latents, strict=True
            )
        ],
        dtype=np.float64,
    )
    log_base = np.asarray(
        [
            base_logpdf(k0, z0, latent.soft_coefficients, t0)
            for k0, z0, t0, latent in zip(
                energy, z_range, event_time, latents, strict=True
            )
        ],
        dtype=np.float64,
    )
    hard_base_ratio = np.asarray(
        [
            _hard_base_log_ratio(
                len(latent.hard_scatters),
                context.path_prior.expected_hard_scatter_count,
                hard_empty_probability,
            )
            for latent, context in zip(latents, contexts, strict=True)
        ],
        dtype=np.float64,
    )
    bridge_score = log_prior + loglike - log_base - hard_base_ratio
    weights = np.full(count, 1.0 / count, dtype=np.float64)
    lineage = np.arange(count, dtype=np.int64)

    # The normalized defensive guide mixture is already a valid proposal for
    # the complete physical target.  This branch keeps every independent draw
    # and applies the exact target/base density ratio directly.  It is useful
    # both as a fast inference engine and, more importantly, as an independent
    # check on annealed-SMC evidence estimates: no tempering, resampling, or
    # Markov transition can collapse a guide component or initial lineage.
    if bool(config.direct_importance_sampling):
        weights, log_weight_sum = _normalize_log_weights(bridge_score)
        realized = np.asarray(
            [context.realized_range_mm for context in contexts],
            dtype=np.float64,
        )
        final_log_prior = np.asarray(
            [
                physical_log_prior(k0, z0, latent.soft_coefficients, t0)
                + context.path_prior.hard_log_prior_density(latent)
                for k0, z0, t0, latent, context in zip(
                    energy,
                    z_range,
                    event_time,
                    latents,
                    contexts,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        return MixedEnergyRangeSMCResult(
            kinetic_energy_mev=np.ascontiguousarray(energy),
            z_range=np.ascontiguousarray(z_range),
            event_time_ns=np.ascontiguousarray(event_time),
            realized_range_mm=np.ascontiguousarray(realized),
            latents=tuple(latents),
            log_likelihood=np.ascontiguousarray(loglike),
            log_prior_density=np.ascontiguousarray(final_log_prior),
            weights=np.ascontiguousarray(weights),
            initial_lineage=np.ascontiguousarray(lineage),
            stages=(),
            likelihood_evaluations=int(likelihood_evaluations),
            invalid_likelihood_evaluations=int(
                invalid_likelihood_evaluations
            ),
            context_builds=int(context_builds),
            invalid_context_builds=int(invalid_context_builds),
            log_evidence=float(log_weight_sum - math.log(count)),
            random_seed=int(config.random_seed),
            kinetic_energy_bounds_mev=(low, high),
            posterior_trajectory_sweeps=0,
            hard_self_transitions=0,
        )

    beta = 0.0
    log_evidence = 0.0
    stages: list[MixedEnergyRangeSMCStage] = []

    def accept(log_ratio: float) -> bool:
        return math.log(
            max(float(rng.random()), np.finfo(np.float64).tiny)
        ) < min(0.0, float(log_ratio))

    def transport_move(index, proposed_energy, proposed_z, temperature, proposal_correction):
        proposed_context = make_context(proposed_energy, proposed_z)
        if proposed_context is None:
            return False
        proposed_latent = contexts[index].path_prior.transport_latent_to(
            proposed_context.path_prior, latents[index]
        )
        proposed_loglike = evaluate(
            proposed_context, proposed_latent, event_time[index]
        )
        if not math.isfinite(proposed_loglike):
            return False
        hard_prior_jacobian = (
            contexts[index].path_prior.hard_transport_log_prior_jacobian(
                proposed_context.path_prior, latents[index]
            )
        )
        proposed_prior = physical_log_prior(
            proposed_energy,
            proposed_z,
            proposed_latent.soft_coefficients,
            event_time[index],
        )
        proposed_base = base_logpdf(
            proposed_energy,
            proposed_z,
            proposed_latent.soft_coefficients,
            event_time[index],
        )
        proposed_hard_ratio = _hard_base_log_ratio(
            len(proposed_latent.hard_scatters),
            proposed_context.path_prior.expected_hard_scatter_count,
            hard_empty_probability,
        )
        ratio = (
            (1.0 - temperature) * (proposed_base - log_base[index])
            + temperature
            * (
                proposed_prior
                + proposed_loglike
                - log_prior[index]
                - loglike[index]
            )
            + hard_prior_jacobian
            + (1.0 - temperature)
            * (proposed_hard_ratio - hard_base_ratio[index])
            + float(proposal_correction)
        )
        if not accept(ratio):
            return False
        energy[index] = proposed_energy
        z_range[index] = proposed_z
        contexts[index] = proposed_context
        latents[index] = proposed_latent
        loglike[index] = proposed_loglike
        log_prior[index] = proposed_prior
        log_base[index] = proposed_base
        hard_base_ratio[index] = proposed_hard_ratio
        bridge_score[index] = (
            proposed_prior
            + proposed_loglike
            - proposed_base
            - proposed_hard_ratio
        )
        return True

    def rejuvenate(temperature: float, sweeps: int, *, record_posterior=False):
        nonlocal hard_self_transitions
        attempts = {
            "energy": 0,
            "event_time": 0,
            "range": 0,
            "soft": 0,
            "soft_pcn": 0,
            "conditional_guide": 0,
            "hard": 0,
            "hard_local": 0,
        }
        accepts = {key: 0 for key in attempts}
        move_metrics = {
            "soft_pcn_jump_norm": [],
            "soft_pcn_loglike_delta": [],
            "soft_pcn_log_ratio": [],
            "conditional_guide_jump_norm": [],
            "conditional_guide_loglike_delta": [],
            "conditional_guide_log_ratio": [],
        }
        effective_guide_probability = float(config.soft_guide_probability)
        recorded_energy: list[np.ndarray] = []
        recorded_z: list[np.ndarray] = []
        recorded_event_time: list[np.ndarray] = []
        recorded_range: list[np.ndarray] = []
        recorded_latents: list[tuple[MixedMCSLatent, ...]] = []
        recorded_loglike: list[np.ndarray] = []
        recorded_logprior: list[np.ndarray] = []
        recorded_lineage: list[np.ndarray] = []
        for _ in range(int(sweeps)):
            for index in rng.permutation(count):
                attempts["energy"] += 1
                if rng.random() < float(config.energy_independence_probability):
                    proposed_energy = float(rng.uniform(low, high))
                else:
                    proposed_energy = _reflected_interval(
                        energy[index]
                        + float(config.energy_random_walk_mev) * float(rng.normal()),
                        low,
                        high,
                    )
                if transport_move(
                    index,
                    proposed_energy,
                    float(z_range[index]),
                    temperature,
                    0.0,
                ):
                    accepts["energy"] += 1

                attempts["range"] += 1
                rho = float(config.range_pcn_rho)
                proposed_z = (
                    rho * z_range[index]
                    + math.sqrt(1.0 - rho * rho) * float(rng.normal())
                )
                proposed_z_logprior = _standard_normal_logpdf(
                    np.asarray((proposed_z,), dtype=np.float64)
                )
                current_z_logprior = _standard_normal_logpdf(
                    np.asarray((z_range[index],), dtype=np.float64)
                )
                if transport_move(
                    index,
                    float(energy[index]),
                    proposed_z,
                    temperature,
                    current_z_logprior - proposed_z_logprior,
                ):
                    accepts["range"] += 1

                if fit_event_time:
                    attempts["event_time"] += 1
                    proposed_t0 = _reflected_interval(
                        event_time[index]
                        + float(config.event_time_random_walk_ns)
                        * float(rng.normal()),
                        t0_low,
                        t0_high,
                    )
                    proposed_loglike = evaluate(
                        contexts[index], latents[index], proposed_t0
                    )
                    proposed_prior = physical_log_prior(
                        energy[index],
                        z_range[index],
                        latents[index].soft_coefficients,
                        proposed_t0,
                    )
                    proposed_base = base_logpdf(
                        energy[index],
                        z_range[index],
                        latents[index].soft_coefficients,
                        proposed_t0,
                    )
                    ratio = (
                        (1.0 - temperature)
                        * (proposed_base - log_base[index])
                        + temperature
                        * (
                            proposed_prior
                            + proposed_loglike
                            - log_prior[index]
                            - loglike[index]
                        )
                    )
                    if math.isfinite(proposed_loglike) and accept(ratio):
                        event_time[index] = proposed_t0
                        loglike[index] = proposed_loglike
                        log_prior[index] = proposed_prior
                        log_base[index] = proposed_base
                        bridge_score[index] = (
                            proposed_prior
                            + proposed_loglike
                            - proposed_base
                            - hard_base_ratio[index]
                        )
                        accepts["event_time"] += 1

                attempts["soft"] += 1
                use_guide = bool(rng.random() < effective_guide_probability)
                if use_guide:
                    attempts["conditional_guide"] += 1
                    proposed_coefficients = _sample_conditional_soft_base(
                        float(energy[index]),
                        float(z_range[index]),
                        guides,
                        low=low,
                        high=high,
                        prior_mixture_probability=epsilon,
                        dimension=dimension,
                        rng=rng,
                    )
                    proposed_latent = MixedMCSLatent(
                        proposed_coefficients,
                        latents[index].hard_scatters,
                    )
                    move_metrics["conditional_guide_jump_norm"].append(
                        float(
                            np.linalg.norm(
                                proposed_coefficients
                                - latents[index].soft_coefficients
                            )
                        )
                    )
                    proposed_loglike = evaluate(
                        contexts[index], proposed_latent, event_time[index]
                    )
                    proposed_prior = physical_log_prior(
                        energy[index],
                        z_range[index],
                        proposed_coefficients,
                        event_time[index],
                    )
                    proposed_base = base_logpdf(
                        energy[index],
                        z_range[index],
                        proposed_coefficients,
                        event_time[index],
                    )
                    proposed_bridge = (
                        proposed_prior
                        + proposed_loglike
                        - proposed_base
                        - hard_base_ratio[index]
                    )
                    guide_ratio = float(
                        temperature * (proposed_bridge - bridge_score[index])
                    )
                    if math.isfinite(proposed_loglike):
                        move_metrics["conditional_guide_loglike_delta"].append(
                            float(proposed_loglike - loglike[index])
                        )
                        move_metrics["conditional_guide_log_ratio"].append(
                            guide_ratio
                        )
                    if math.isfinite(proposed_loglike) and accept(guide_ratio):
                        latents[index] = proposed_latent
                        loglike[index] = proposed_loglike
                        log_prior[index] = proposed_prior
                        log_base[index] = proposed_base
                        bridge_score[index] = proposed_bridge
                        accepts["soft"] += 1
                        accepts["conditional_guide"] += 1
                else:
                    attempts["soft_pcn"] += 1
                    if bool(config.soft_elliptical_slice):
                        current_coefficients = np.asarray(
                            latents[index].soft_coefficients,
                            dtype=np.float64,
                        )
                        direction = rng.normal(size=dimension)
                        current_residual = (
                            (1.0 - temperature)
                            * (log_base[index] - log_prior[index])
                            + temperature * loglike[index]
                        )
                        slice_height = current_residual + math.log(
                            max(
                                float(rng.random()),
                                np.finfo(np.float64).tiny,
                            )
                        )
                        angle = float(rng.uniform(0.0, 2.0 * math.pi))
                        angle_min = angle - 2.0 * math.pi
                        angle_max = angle
                        accepted_slice = None
                        last_finite = None
                        for _shrink in range(
                            int(config.soft_elliptical_max_bracket_shrinks)
                        ):
                            proposed_coefficients = np.ascontiguousarray(
                                current_coefficients * math.cos(angle)
                                + direction * math.sin(angle)
                            )
                            proposed_latent = MixedMCSLatent(
                                proposed_coefficients,
                                latents[index].hard_scatters,
                            )
                            proposed_loglike = evaluate(
                                contexts[index], proposed_latent,
                                event_time[index]
                            )
                            proposed_prior = physical_log_prior(
                                energy[index],
                                z_range[index],
                                proposed_coefficients,
                                event_time[index],
                            )
                            proposed_base = base_logpdf(
                                energy[index],
                                z_range[index],
                                proposed_coefficients,
                                event_time[index],
                            )
                            proposed_residual = (
                                (1.0 - temperature)
                                * (proposed_base - proposed_prior)
                                + temperature * proposed_loglike
                            )
                            if math.isfinite(proposed_loglike):
                                last_finite = (
                                    proposed_latent,
                                    proposed_loglike,
                                    proposed_prior,
                                    proposed_base,
                                    proposed_residual,
                                )
                            if (
                                math.isfinite(proposed_residual)
                                and proposed_residual >= slice_height
                            ):
                                accepted_slice = last_finite
                                break
                            if angle < 0.0:
                                angle_min = angle
                            else:
                                angle_max = angle
                            angle = float(rng.uniform(angle_min, angle_max))
                        metric_state = (
                            accepted_slice
                            if accepted_slice is not None else last_finite
                        )
                        if metric_state is not None:
                            metric_latent, metric_loglike, *_metric_rest = metric_state
                            move_metrics["soft_pcn_jump_norm"].append(
                                float(
                                    np.linalg.norm(
                                        metric_latent.soft_coefficients
                                        - current_coefficients
                                    )
                                )
                            )
                            move_metrics["soft_pcn_loglike_delta"].append(
                                float(metric_loglike - loglike[index])
                            )
                            move_metrics["soft_pcn_log_ratio"].append(
                                float(metric_state[4] - current_residual)
                            )
                        if accepted_slice is not None:
                            (
                                proposed_latent,
                                proposed_loglike,
                                proposed_prior,
                                proposed_base,
                                _proposed_residual,
                            ) = accepted_slice
                            latents[index] = proposed_latent
                            loglike[index] = proposed_loglike
                            log_prior[index] = proposed_prior
                            log_base[index] = proposed_base
                            bridge_score[index] = (
                                proposed_prior
                                + proposed_loglike
                                - proposed_base
                                - hard_base_ratio[index]
                            )
                            accepts["soft"] += 1
                            accepts["soft_pcn"] += 1
                    else:
                        proposed_latent = contexts[index].path_prior.pcn_soft(
                            latents[index], float(config.soft_pcn_rho), rng
                        )
                        move_metrics["soft_pcn_jump_norm"].append(
                            float(
                                np.linalg.norm(
                                    proposed_latent.soft_coefficients
                                    - latents[index].soft_coefficients
                                )
                            )
                        )
                        proposed_loglike = evaluate(
                            contexts[index], proposed_latent,
                            event_time[index]
                        )
                        proposed_prior = physical_log_prior(
                            energy[index],
                            z_range[index],
                            proposed_latent.soft_coefficients,
                            event_time[index],
                        )
                        proposed_base = base_logpdf(
                            energy[index],
                            z_range[index],
                            proposed_latent.soft_coefficients,
                            event_time[index],
                        )
                        ratio = (
                            (1.0 - temperature)
                            * (
                                (proposed_base - proposed_prior)
                                - (log_base[index] - log_prior[index])
                            )
                            + temperature * (
                                proposed_loglike - loglike[index]
                            )
                        )
                        if math.isfinite(proposed_loglike):
                            move_metrics["soft_pcn_loglike_delta"].append(
                                float(proposed_loglike - loglike[index])
                            )
                            move_metrics["soft_pcn_log_ratio"].append(
                                float(ratio)
                            )
                        if math.isfinite(proposed_loglike) and accept(ratio):
                            latents[index] = proposed_latent
                            loglike[index] = proposed_loglike
                            log_prior[index] = proposed_prior
                            log_base[index] = proposed_base
                            bridge_score[index] = (
                                proposed_prior
                                + proposed_loglike
                                - proposed_base
                                - hard_base_ratio[index]
                            )
                            accepts["soft"] += 1
                            accepts["soft_pcn"] += 1

                attempts["hard"] += 1
                proposed_latent = contexts[index].path_prior.refresh_hard_scatters(
                    latents[index],
                    float(config.hard_retention_probability),
                    rng,
                )
                if proposed_latent.hard_scatters == latents[index].hard_scatters:
                    # Thinning/superposition proposed the literal current mark
                    # set.  This is an accepted self-transition, so recomputing
                    # the identical physical path cannot affect the chain. Draw
                    # the otherwise automatic acceptance variate to preserve
                    # matched-seed streams across the optimized implementation.
                    accept(0.0)
                    hard_self_transitions += 1
                    accepts["hard"] += 1
                else:
                    proposed_loglike = evaluate(
                        contexts[index], proposed_latent, event_time[index]
                    )
                    proposed_hard_ratio = _hard_base_log_ratio(
                        len(proposed_latent.hard_scatters),
                        contexts[index].path_prior.expected_hard_scatter_count,
                        hard_empty_probability,
                    )
                    ratio = (
                        (1.0 - temperature)
                        * (proposed_hard_ratio - hard_base_ratio[index])
                        + temperature * (proposed_loglike - loglike[index])
                    )
                    if math.isfinite(proposed_loglike) and accept(ratio):
                        latents[index] = proposed_latent
                        loglike[index] = proposed_loglike
                        hard_base_ratio[index] = proposed_hard_ratio
                        bridge_score[index] = (
                            log_prior[index]
                            + proposed_loglike
                            - log_base[index]
                            - proposed_hard_ratio
                        )
                        accepts["hard"] += 1

                if latents[index].hard_scatters:
                    attempts["hard_local"] += 1
                    current_hard_prior = (
                        contexts[index].path_prior.hard_log_prior_density(
                            latents[index]
                        )
                    )
                    proposed_latent, proposal_correction = (
                        contexts[index].path_prior.perturb_hard_scatter(
                            latents[index],
                            rng,
                            position_step_fraction=float(
                                config.hard_local_position_step_fraction
                            ),
                            log_angle_step=float(
                                config.hard_local_log_angle_step
                            ),
                            azimuth_step_rad=float(
                                config.hard_local_azimuth_step_rad
                            ),
                        )
                    )
                    proposed_loglike = evaluate(
                        contexts[index], proposed_latent, event_time[index]
                    )
                    proposed_hard_prior = (
                        contexts[index].path_prior.hard_log_prior_density(
                            proposed_latent
                        )
                    )
                    # Conditional on count/channel, both the defensive hard
                    # proposal and the physical target use the same Wentzel
                    # marked-process density.  It therefore appears with unit
                    # power at every bridge temperature; only the data term is
                    # annealed.  ``proposal_correction`` is the log-angle
                    # coordinate Jacobian returned by the local kernel.
                    local_ratio = (
                        proposed_hard_prior
                        - current_hard_prior
                        + temperature
                        * (proposed_loglike - loglike[index])
                        + float(proposal_correction)
                    )
                    if math.isfinite(proposed_loglike) and accept(local_ratio):
                        latents[index] = proposed_latent
                        loglike[index] = proposed_loglike
                        bridge_score[index] = (
                            log_prior[index]
                            + proposed_loglike
                            - log_base[index]
                            - hard_base_ratio[index]
                        )
                        accepts["hard_local"] += 1
            if record_posterior:
                recorded_energy.append(np.array(energy, copy=True))
                recorded_z.append(np.array(z_range, copy=True))
                recorded_event_time.append(np.array(event_time, copy=True))
                recorded_range.append(
                    np.asarray(
                        [context.realized_range_mm for context in contexts],
                        dtype=np.float64,
                    )
                )
                recorded_latents.append(tuple(latents))
                recorded_loglike.append(np.array(loglike, copy=True))
                recorded_logprior.append(
                    np.asarray(
                        [
                            physical_log_prior(
                                k0, z0, latent.soft_coefficients, t0
                            )
                            + context.path_prior.hard_log_prior_density(latent)
                            for k0, z0, t0, latent, context in zip(
                                energy,
                                z_range,
                                event_time,
                                latents,
                                contexts,
                                strict=True,
                            )
                        ],
                        dtype=np.float64,
                    )
                )
                recorded_lineage.append(np.array(lineage, copy=True))

        def metric(name, operation):
            values = np.asarray(move_metrics[name], dtype=np.float64)
            return float(operation(values)) if values.size else math.nan

        return (
            {
                **{
                    name: accepts[name] / max(attempts[name], 1)
                    for name in attempts
                },
                "soft_pcn_attempts": int(attempts["soft_pcn"]),
                "soft_pcn_finite_proposals": int(
                    len(move_metrics["soft_pcn_log_ratio"])
                ),
                "soft_pcn_jump_norm_median": metric(
                    "soft_pcn_jump_norm", np.median
                ),
                "soft_pcn_loglike_delta_median": metric(
                    "soft_pcn_loglike_delta", np.median
                ),
                "soft_pcn_log_ratio_median": metric(
                    "soft_pcn_log_ratio", np.median
                ),
                "soft_pcn_log_ratio_max": metric(
                    "soft_pcn_log_ratio", np.max
                ),
                "conditional_guide_attempts": int(
                    attempts["conditional_guide"]
                ),
                "conditional_guide_finite_proposals": int(
                    len(move_metrics["conditional_guide_log_ratio"])
                ),
                "conditional_guide_jump_norm_median": metric(
                    "conditional_guide_jump_norm", np.median
                ),
                "conditional_guide_loglike_delta_median": metric(
                    "conditional_guide_loglike_delta", np.median
                ),
                "conditional_guide_log_ratio_median": metric(
                    "conditional_guide_log_ratio", np.median
                ),
                "conditional_guide_log_ratio_max": metric(
                    "conditional_guide_log_ratio", np.max
                ),
                "conditional_guide_effective_probability": float(
                    effective_guide_probability
                ),
            },
            (
                recorded_energy,
                recorded_z,
                recorded_event_time,
                recorded_range,
                recorded_latents,
                recorded_loglike,
                recorded_logprior,
                recorded_lineage,
            ),
        )

    while beta < 1.0 - 1.0e-12:
        if len(stages) >= int(config.max_temperatures):
            raise RuntimeError("blind mixed SMC exceeded max_temperatures")
        next_beta = _next_temperature(
            beta,
            weights,
            bridge_score,
            float(config.target_ess_fraction) * count,
        )
        weights, log_normalizer = _normalize_log_weights(
            np.log(np.maximum(weights, np.finfo(np.float64).tiny))
            + (next_beta - beta) * bridge_score
        )
        log_evidence += log_normalizer
        ess = float(1.0 / np.sum(weights * weights))
        resampled = bool(
            ess < float(config.resample_ess_fraction) * count
            or next_beta >= 1.0 - 1.0e-12
        )
        if resampled:
            indices = _systematic_resample(weights, rng)
            energy = energy[indices].copy()
            z_range = z_range[indices].copy()
            event_time = event_time[indices].copy()
            contexts = [contexts[int(index)] for index in indices]
            latents = [latents[int(index)] for index in indices]
            loglike = loglike[indices].copy()
            log_prior = log_prior[indices].copy()
            log_base = log_base[indices].copy()
            hard_base_ratio = hard_base_ratio[indices].copy()
            bridge_score = bridge_score[indices].copy()
            lineage = lineage[indices].copy()
            weights.fill(1.0 / count)
        acceptance, _recorded = rejuvenate(
            next_beta, int(config.rejuvenation_steps)
        )
        stages.append(
            MixedEnergyRangeSMCStage(
                beta_before=float(beta),
                beta_after=float(next_beta),
                ess_before_resampling=float(ess),
                resampled=resampled,
                energy_acceptance=float(acceptance["energy"]),
                event_time_acceptance=float(acceptance["event_time"]),
                range_acceptance=float(acceptance["range"]),
                soft_acceptance=float(acceptance["soft"]),
                soft_pcn_acceptance=float(acceptance["soft_pcn"]),
                conditional_guide_acceptance=float(
                    acceptance["conditional_guide"]
                ),
                hard_acceptance=float(acceptance["hard"]),
                hard_local_acceptance=float(acceptance["hard_local"]),
                soft_pcn_attempts=int(acceptance["soft_pcn_attempts"]),
                soft_pcn_finite_proposals=int(
                    acceptance["soft_pcn_finite_proposals"]
                ),
                soft_pcn_jump_norm_median=float(
                    acceptance["soft_pcn_jump_norm_median"]
                ),
                soft_pcn_loglike_delta_median=float(
                    acceptance["soft_pcn_loglike_delta_median"]
                ),
                soft_pcn_log_ratio_median=float(
                    acceptance["soft_pcn_log_ratio_median"]
                ),
                soft_pcn_log_ratio_max=float(
                    acceptance["soft_pcn_log_ratio_max"]
                ),
                conditional_guide_attempts=int(
                    acceptance["conditional_guide_attempts"]
                ),
                conditional_guide_finite_proposals=int(
                    acceptance["conditional_guide_finite_proposals"]
                ),
                conditional_guide_jump_norm_median=float(
                    acceptance["conditional_guide_jump_norm_median"]
                ),
                conditional_guide_loglike_delta_median=float(
                    acceptance["conditional_guide_loglike_delta_median"]
                ),
                conditional_guide_log_ratio_median=float(
                    acceptance["conditional_guide_log_ratio_median"]
                ),
                conditional_guide_log_ratio_max=float(
                    acceptance["conditional_guide_log_ratio_max"]
                ),
                conditional_guide_effective_probability=float(
                    acceptance["conditional_guide_effective_probability"]
                ),
            )
        )
        beta = next_beta

    posterior_steps = int(config.posterior_rejuvenation_steps)
    posterior_record = ([], [], [], [], [], [], [], [])
    if posterior_steps:
        acceptance, posterior_record = rejuvenate(
            1.0, posterior_steps, record_posterior=True
        )
        stages.append(
            MixedEnergyRangeSMCStage(
                beta_before=1.0,
                beta_after=1.0,
                ess_before_resampling=float(1.0 / np.sum(weights * weights)),
                resampled=False,
                energy_acceptance=float(acceptance["energy"]),
                event_time_acceptance=float(acceptance["event_time"]),
                range_acceptance=float(acceptance["range"]),
                soft_acceptance=float(acceptance["soft"]),
                soft_pcn_acceptance=float(acceptance["soft_pcn"]),
                conditional_guide_acceptance=float(
                    acceptance["conditional_guide"]
                ),
                hard_acceptance=float(acceptance["hard"]),
                hard_local_acceptance=float(acceptance["hard_local"]),
                soft_pcn_attempts=int(acceptance["soft_pcn_attempts"]),
                soft_pcn_finite_proposals=int(
                    acceptance["soft_pcn_finite_proposals"]
                ),
                soft_pcn_jump_norm_median=float(
                    acceptance["soft_pcn_jump_norm_median"]
                ),
                soft_pcn_loglike_delta_median=float(
                    acceptance["soft_pcn_loglike_delta_median"]
                ),
                soft_pcn_log_ratio_median=float(
                    acceptance["soft_pcn_log_ratio_median"]
                ),
                soft_pcn_log_ratio_max=float(
                    acceptance["soft_pcn_log_ratio_max"]
                ),
                conditional_guide_attempts=int(
                    acceptance["conditional_guide_attempts"]
                ),
                conditional_guide_finite_proposals=int(
                    acceptance["conditional_guide_finite_proposals"]
                ),
                conditional_guide_jump_norm_median=float(
                    acceptance["conditional_guide_jump_norm_median"]
                ),
                conditional_guide_loglike_delta_median=float(
                    acceptance["conditional_guide_loglike_delta_median"]
                ),
                conditional_guide_log_ratio_median=float(
                    acceptance["conditional_guide_log_ratio_median"]
                ),
                conditional_guide_log_ratio_max=float(
                    acceptance["conditional_guide_log_ratio_max"]
                ),
                conditional_guide_effective_probability=float(
                    acceptance["conditional_guide_effective_probability"]
                ),
            )
        )
    if posterior_record[0]:
        energy = np.concatenate(posterior_record[0])
        z_range = np.concatenate(posterior_record[1])
        event_time = np.concatenate(posterior_record[2])
        realized = np.concatenate(posterior_record[3])
        latents = [
            latent for sweep in posterior_record[4] for latent in sweep
        ]
        loglike = np.concatenate(posterior_record[5])
        final_log_prior = np.concatenate(posterior_record[6])
        final_lineage = np.concatenate(posterior_record[7])
        weights = np.full(energy.size, 1.0 / energy.size, dtype=np.float64)
    else:
        realized = np.asarray(
            [context.realized_range_mm for context in contexts], dtype=np.float64
        )
        final_log_prior = np.asarray(
            [
                physical_log_prior(k0, z0, latent.soft_coefficients, t0)
                + context.path_prior.hard_log_prior_density(latent)
                for k0, z0, t0, latent, context in zip(
                    energy,
                    z_range,
                    event_time,
                    latents,
                    contexts,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
        final_lineage = np.asarray(lineage, dtype=np.int64)
    return MixedEnergyRangeSMCResult(
        kinetic_energy_mev=np.ascontiguousarray(energy),
        z_range=np.ascontiguousarray(z_range),
        event_time_ns=np.ascontiguousarray(event_time),
        realized_range_mm=np.ascontiguousarray(realized),
        latents=tuple(latents),
        log_likelihood=np.ascontiguousarray(loglike),
        log_prior_density=np.ascontiguousarray(final_log_prior),
        weights=np.ascontiguousarray(weights),
        initial_lineage=np.ascontiguousarray(final_lineage),
        stages=tuple(stages),
        likelihood_evaluations=int(likelihood_evaluations),
        invalid_likelihood_evaluations=int(invalid_likelihood_evaluations),
        context_builds=int(context_builds),
        invalid_context_builds=int(invalid_context_builds),
        log_evidence=float(log_evidence),
        random_seed=int(config.random_seed),
        kinetic_energy_bounds_mev=(low, high),
        posterior_trajectory_sweeps=int(posterior_steps),
        hard_self_transitions=int(hard_self_transitions),
    )


def run_stratified_mixed_energy_range_smc(
    context_factory: MixedEnergyRangeContextFactory,
    dimension: int,
    energy_path_guides: Sequence[EnergyPathGuide],
    config: MixedEnergyRangeSMCConfig,
    energy_stratum_edges_mev: Sequence[float],
    replicates_per_stratum: int = 1,
) -> MixedEnergyRangeSMCResult:
    """Run exact disjoint energy strata and recombine them by evidence.

    Every sub-run targets the posterior conditional on its energy interval.
    If ``Z_i`` is that conditional evidence and ``p_i`` is the interval's
    mass under the original uniform energy prior, then the full evidence is
    ``sum_i p_i Z_i`` and the posterior mass of stratum ``i`` is proportional
    to ``p_i Z_i``.  This preserves separated energy/path basins without
    changing either the likelihood or the broad physical prior. Independent
    replicate evidence estimates within a stratum are averaged, while their
    posterior samples are mixed using the corresponding evidence weights.
    """
    config.validate()
    full_low, full_high = map(float, config.kinetic_energy_bounds_mev)
    edges = np.asarray(tuple(energy_stratum_edges_mev), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2 or np.any(~np.isfinite(edges)):
        raise ValueError("energy stratum edges must be a finite 1D sequence")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy stratum edges must be strictly increasing")
    tolerance = 1.0e-10 * max(1.0, abs(full_low), abs(full_high))
    if (
        abs(float(edges[0]) - full_low) > tolerance
        or abs(float(edges[-1]) - full_high) > tolerance
    ):
        raise ValueError(
            "energy stratum edges must span the configured energy bounds"
        )
    edges[0] = full_low
    edges[-1] = full_high
    replicates = int(replicates_per_stratum)
    if replicates < 1:
        raise ValueError("replicates_per_stratum must be positive")
    if edges.size == 2 and replicates == 1:
        return run_mixed_energy_range_smc(
            context_factory, dimension, energy_path_guides, config
        )

    guides = tuple(energy_path_guides)
    if not guides:
        raise ValueError("at least one normalized energy/path guide is required")
    total_width = full_high - full_low
    subresults: list[MixedEnergyRangeSMCResult] = []
    log_evidence_terms: list[float] = []
    prior_masses: list[float] = []
    selected_guide_counts: list[int] = []

    run_stratum_indices: list[int] = []
    run_replicate_indices: list[int] = []
    for stratum_index, (low, high) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        low = float(low)
        high = float(high)
        # Very distant Gaussian guide components can have a numerically zero
        # truncation constant.  They are proposal components only, so removing
        # those components leaves the exact target unchanged.  Always retain
        # the nearest component as a fallback alongside the defensive prior.
        selected_guides = []
        for guide in guides:
            probe = float(np.clip(guide.kinetic_energy_mev, low, high))
            if math.isfinite(_truncated_normal_logpdf(
                probe,
                float(guide.kinetic_energy_mev),
                float(guide.energy_proposal_sd_mev),
                low,
                high,
            )):
                selected_guides.append(guide)
        if not selected_guides:
            midpoint = 0.5 * (low + high)
            selected_guides.append(min(
                guides,
                key=lambda guide: abs(
                    float(guide.kinetic_energy_mev) - midpoint
                ),
            ))
        prior_mass = (high - low) / total_width
        for replicate_index in range(replicates):
            selected_guide_counts.append(len(selected_guides))
            subconfig = replace(
                config,
                kinetic_energy_bounds_mev=(low, high),
                random_seed=(
                    int(config.random_seed)
                    + 104729 * stratum_index
                    + 1009 * replicate_index
                ),
            )
            subresult = run_mixed_energy_range_smc(
                context_factory,
                dimension,
                tuple(selected_guides),
                subconfig,
            )
            prior_masses.append(float(prior_mass))
            # Replicate identity is an auxiliary uniform mixture coordinate.
            # Averaging the independent evidence estimates keeps the original
            # physical prior mass for this stratum unchanged.
            log_evidence_terms.append(
                math.log(float(prior_mass))
                - math.log(float(replicates))
                + float(subresult.log_evidence)
            )
            subresults.append(subresult)
            run_stratum_indices.append(int(stratum_index))
            run_replicate_indices.append(int(replicate_index))

    terms = np.asarray(log_evidence_terms, dtype=np.float64)
    maximum = float(np.max(terms))
    if not math.isfinite(maximum):
        raise RuntimeError("all energy strata have non-finite evidence")
    combined_log_evidence = float(
        maximum + math.log(float(np.sum(np.exp(terms - maximum))))
    )
    stratum_mass = np.exp(terms - combined_log_evidence)
    stratum_mass /= float(np.sum(stratum_mass))

    energy = np.concatenate([row.kinetic_energy_mev for row in subresults])
    z_range = np.concatenate([row.z_range for row in subresults])
    event_time = np.concatenate([row.event_time_ns for row in subresults])
    realized = np.concatenate([row.realized_range_mm for row in subresults])
    loglike = np.concatenate([row.log_likelihood for row in subresults])
    # Convert each conditional-uniform energy density back to the density
    # under the original broad uniform prior before MAP comparisons.
    log_prior = np.concatenate([
        row.log_prior_density + math.log(prior_mass)
        for row, prior_mass in zip(subresults, prior_masses, strict=True)
    ])
    weights = np.concatenate([
        float(mass) * row.weights
        for row, mass in zip(subresults, stratum_mass, strict=True)
    ])
    weights /= float(np.sum(weights))
    latents = tuple(
        latent for row in subresults for latent in row.latents
    )
    lineage_parts = []
    lineage_offset = 0
    for row in subresults:
        local = np.asarray(row.initial_lineage, dtype=np.int64)
        if local.size:
            local = local - int(np.min(local)) + lineage_offset
            lineage_offset = int(np.max(local)) + 1
        lineage_parts.append(local)
    lineage = np.concatenate(lineage_parts)

    diagnostics = []
    stratum_posterior_mass = {
        index: float(np.sum(stratum_mass[
            np.asarray(run_stratum_indices, dtype=np.int64) == index
        ]))
        for index in range(edges.size - 1)
    }
    stratum_replication = {}
    run_stratum_array = np.asarray(run_stratum_indices, dtype=np.int64)
    for index in range(edges.size - 1):
        indices = np.flatnonzero(run_stratum_array == index)
        conditional_log_z = np.asarray(
            [subresults[int(i)].log_evidence for i in indices],
            dtype=np.float64,
        )
        maximum_log_z = float(np.max(conditional_log_z))
        replicate_weight = np.exp(conditional_log_z - maximum_log_z)
        replicate_weight /= float(np.sum(replicate_weight))
        energy_means = np.asarray([
            subresults[int(i)].weighted_mean(
                subresults[int(i)].kinetic_energy_mev
            )
            for i in indices
        ])
        range_means = np.asarray([
            subresults[int(i)].weighted_mean(
                subresults[int(i)].realized_range_mm
            )
            for i in indices
        ])
        stratum_replication[index] = {
            "conditional_log_evidence_range": float(
                np.ptp(conditional_log_z)
            ),
            "conditional_log_evidence_sd": float(
                np.std(conditional_log_z, ddof=1)
                if conditional_log_z.size > 1 else math.nan
            ),
            "evidence_weight_effective_replicates": float(
                1.0 / np.sum(replicate_weight * replicate_weight)
            ),
            "replicate_energy_mean_range_mev": float(np.ptp(energy_means)),
            "replicate_range_mean_range_mm": float(np.ptp(range_means)),
        }
    for run_index, (row, prior_mass, posterior_mass, term) in enumerate(zip(
        subresults,
        prior_masses,
        stratum_mass,
        terms,
        strict=True,
    )):
        stratum_index = run_stratum_indices[run_index]
        row_summary = row.summary()
        diagnostics.append({
            "index": int(stratum_index),
            "replicate": int(run_replicate_indices[run_index]),
            "replicates": int(replicates),
            "bounds_mev": [
                float(edges[stratum_index]),
                float(edges[stratum_index + 1]),
            ],
            "prior_mass": float(prior_mass),
            "conditional_log_evidence": float(row.log_evidence),
            "full_prior_log_evidence_contribution": float(term),
            "posterior_mass": float(posterior_mass),
            "stratum_posterior_mass": float(
                stratum_posterior_mass[stratum_index]
            ),
            **stratum_replication[stratum_index],
            "posterior_sample_count": int(row.weights.size),
            "nominal_posterior_weight_ess": float(
                row_summary["nominal_posterior_weight_ess"]
            ),
            "posterior_weight_ess_fraction": float(
                row_summary["posterior_weight_ess_fraction"]
            ),
            "maximum_posterior_weight": float(
                row_summary["maximum_posterior_weight"]
            ),
            "posterior_energy_mean_mev": float(
                row.weighted_mean(row.kinetic_energy_mev)
            ),
            "posterior_range_mean_mm": float(
                row.weighted_mean(row.realized_range_mm)
            ),
            "unique_initial_lineages": int(row.unique_initial_lineages),
            "initial_lineage_ess": float(
                row.initial_lineage_effective_sample_size
            ),
            "unique_soft_path_states": int(row.unique_soft_path_states),
            "unique_joint_states": int(row.unique_joint_states),
            "joint_state_duplicate_ess": float(
                row.joint_state_effective_sample_size
            ),
            "temperature_stages": int(len(row.stages)),
            "likelihood_evaluations": int(row.likelihood_evaluations),
            "selected_guide_count": int(selected_guide_counts[run_index]),
            "random_seed": int(row.random_seed),
        })

    return MixedEnergyRangeSMCResult(
        kinetic_energy_mev=np.ascontiguousarray(energy),
        z_range=np.ascontiguousarray(z_range),
        event_time_ns=np.ascontiguousarray(event_time),
        realized_range_mm=np.ascontiguousarray(realized),
        latents=latents,
        log_likelihood=np.ascontiguousarray(loglike),
        log_prior_density=np.ascontiguousarray(log_prior),
        weights=np.ascontiguousarray(weights),
        initial_lineage=np.ascontiguousarray(lineage),
        stages=tuple(stage for row in subresults for stage in row.stages),
        likelihood_evaluations=int(sum(
            row.likelihood_evaluations for row in subresults
        )),
        invalid_likelihood_evaluations=int(sum(
            row.invalid_likelihood_evaluations for row in subresults
        )),
        context_builds=int(sum(row.context_builds for row in subresults)),
        invalid_context_builds=int(sum(
            row.invalid_context_builds for row in subresults
        )),
        log_evidence=combined_log_evidence,
        random_seed=int(config.random_seed),
        kinetic_energy_bounds_mev=(full_low, full_high),
        posterior_trajectory_sweeps=int(config.posterior_rejuvenation_steps),
        hard_self_transitions=int(sum(
            row.hard_self_transitions for row in subresults
        )),
        energy_stratum_diagnostics=tuple(diagnostics),
    )


__all__ = [
    "MixedEnergyRangeContext",
    "MixedEnergyRangeSMCConfig",
    "MixedEnergyRangeSMCResult",
    "MixedEnergyRangeSMCStage",
    "run_mixed_energy_range_smc",
    "run_stratified_mixed_energy_range_smc",
]
