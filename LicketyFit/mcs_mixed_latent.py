"""Annealed posterior inference for a fixed mixed-MCS trajectory prior.

The target is ``p(path) p(charge | path)^beta``.  Soft pCN and Poisson
thinning/superposition are each reversible under their respective physical
priors, so their Metropolis ratios contain only the tempered charge
likelihood.  This module is the fixed-track gate for the mixed process; energy,
range, and global geometry are deliberately outside its state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .mcs_mixed_path import MixedMCSLatent, MixedMCSPathPrior
from .mcs_joint_energy_range import GaussianPathGuide


MixedLogLikelihood = Callable[[MixedMCSLatent], float]


@dataclass(frozen=True)
class MixedPathSMCConfig:
    particles: int = 32
    target_ess_fraction: float = 0.80
    resample_ess_fraction: float = 0.55
    max_temperatures: int = 64
    rejuvenation_steps: int = 3
    posterior_rejuvenation_steps: int = 12
    soft_pcn_rho: float = 0.92
    hard_retention_probability: float = 0.72
    soft_guide_probability: float = 0.35
    guide_prior_mixture_probability: float = 0.20
    hard_empty_guide_probability: float = 0.0
    random_seed: int = 41873

    def validate(self) -> None:
        if int(self.particles) < 2:
            raise ValueError("SMC requires at least two particles")
        for name, value in (
            ("target_ess_fraction", self.target_ess_fraction),
            ("resample_ess_fraction", self.resample_ess_fraction),
            ("soft_guide_probability", self.soft_guide_probability),
            (
                "guide_prior_mixture_probability",
                self.guide_prior_mixture_probability,
            ),
            ("hard_empty_guide_probability", self.hard_empty_guide_probability),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0.0 < float(self.target_ess_fraction) <= 1.0:
            raise ValueError("target_ess_fraction must lie in (0,1]")
        if not 0.0 < float(self.resample_ess_fraction) <= 1.0:
            raise ValueError("resample_ess_fraction must lie in (0,1]")
        if not 0.0 < float(self.guide_prior_mixture_probability) < 1.0:
            raise ValueError(
                "guide_prior_mixture_probability must lie in (0,1)"
            )
        if not 0.0 <= float(self.hard_empty_guide_probability) < 1.0:
            raise ValueError("hard_empty_guide_probability must lie in [0,1)")
        for name, value in (
            ("soft_pcn_rho", self.soft_pcn_rho),
            ("hard_retention_probability", self.hard_retention_probability),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must lie in [0,1)")
        if int(self.max_temperatures) < 1:
            raise ValueError("max_temperatures must be positive")
        if int(self.rejuvenation_steps) < 0:
            raise ValueError("rejuvenation_steps must be non-negative")
        if int(self.posterior_rejuvenation_steps) < 0:
            raise ValueError("posterior_rejuvenation_steps must be non-negative")


@dataclass(frozen=True)
class MixedPathSMCStage:
    beta_before: float
    beta_after: float
    ess_before_resampling: float
    resampled: bool
    soft_acceptance: float
    hard_acceptance: float


@dataclass
class MixedPathSMCResult:
    latents: tuple[MixedMCSLatent, ...]
    log_likelihood: np.ndarray
    weights: np.ndarray
    stages: tuple[MixedPathSMCStage, ...]
    likelihood_evaluations: int
    invalid_likelihood_evaluations: int
    log_evidence: float
    random_seed: int
    posterior_trajectory_sweeps: int

    @property
    def map_index(self) -> int:
        return int(np.argmax(self.log_likelihood))

    @property
    def map_latent(self) -> MixedMCSLatent:
        return self.latents[self.map_index]

    @property
    def hard_scatter_count(self) -> np.ndarray:
        return np.asarray(
            [len(latent.hard_scatters) for latent in self.latents], dtype=np.int32
        )


def _normalize_log_weights(log_weight: np.ndarray):
    maximum = float(np.max(log_weight))
    if not math.isfinite(maximum):
        raise RuntimeError("all SMC weights are zero")
    scaled = np.exp(log_weight - maximum)
    total = float(np.sum(scaled))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("SMC weight normalization failed")
    return np.ascontiguousarray(scaled / total), float(maximum + math.log(total))


def _standard_normal_logpdf(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(
        -0.5 * array @ array
        - 0.5 * array.size * math.log(2.0 * math.pi)
    )


def _logsumexp(first: float, second: float) -> float:
    maximum = max(float(first), float(second))
    if not math.isfinite(maximum):
        return -math.inf
    return float(
        maximum
        + math.log(math.exp(float(first) - maximum) + math.exp(float(second) - maximum))
    )


def _base_logpdf(
    coefficients: np.ndarray,
    guide: GaussianPathGuide | None,
    prior_probability: float,
) -> float:
    prior = _standard_normal_logpdf(coefficients)
    if guide is None:
        return prior
    return _logsumexp(
        math.log(float(prior_probability)) + prior,
        math.log1p(-float(prior_probability)) + guide.logpdf(coefficients),
    )


def _sample_base_coefficients(
    dimension: int,
    guide: GaussianPathGuide | None,
    prior_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if guide is None or rng.random() < float(prior_probability):
        return np.ascontiguousarray(rng.normal(size=int(dimension)))
    return guide.sample(rng)


def _sample_base_hard_scatters(
    prior: MixedMCSPathPrior,
    empty_probability: float,
    rng: np.random.Generator,
):
    if rng.random() < float(empty_probability):
        return ()
    return prior.sample_hard_scatters(rng)


def _hard_base_log_ratio(
    hard_count: int,
    expected_count: float,
    empty_probability: float,
) -> float:
    """Return log(q_hard / p_Poisson) for the defensive empty mixture."""
    empty = float(empty_probability)
    if empty <= 0.0:
        return 0.0
    if int(hard_count) > 0:
        return math.log1p(-empty)
    # q(empty)/p(empty) = (1-empty) + empty/P_Poisson(empty), and
    # P_Poisson(empty)=exp(-expected_count).
    first = math.log1p(-empty)
    second = math.log(empty) + float(expected_count)
    return _logsumexp(first, second)


def _ess(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.asarray(weights, dtype=np.float64) ** 2))


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator):
    count = int(weights.size)
    positions = (float(rng.random()) + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _conditional_ess(weights, log_likelihood, delta_beta):
    """Conditional ESS of an incremental bridge, independent of current ESS.

    Using the ESS of the fully updated weights here makes an adaptive schedule
    stall whenever the current weights already sit on the requested boundary;
    older code compensated by resampling at every temperature, which rapidly
    destroyed particle lineages.  The standard CESS instead measures only the
    proposed incremental weights under the current particle measure.
    """
    weights = np.asarray(weights, dtype=np.float64)
    score = np.asarray(log_likelihood, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        increment = float(delta_beta) * score
    increment = np.where(np.isfinite(score), increment, -np.inf)
    maximum = float(np.max(increment))
    if not math.isfinite(maximum):
        return 0.0
    ratio = np.exp(np.clip(increment - maximum, -745.0, 0.0))
    first = float(np.dot(weights, ratio))
    second = float(np.dot(weights, ratio * ratio))
    if not math.isfinite(second) or second <= 0.0:
        return 0.0
    return float(weights.size * first * first / second)


def _next_temperature(beta, weights, log_likelihood, target_ess):
    remaining = 1.0 - float(beta)
    if remaining <= 1.0e-12:
        return 1.0
    if _conditional_ess(weights, log_likelihood, remaining) >= target_ess:
        return 1.0
    low = 0.0
    high = remaining
    for _ in range(64):
        middle = 0.5 * (low + high)
        if _conditional_ess(weights, log_likelihood, middle) < target_ess:
            high = middle
        else:
            low = middle
    return float(beta + max(low, np.finfo(np.float64).eps))


def run_mixed_path_smc(
    prior: MixedMCSPathPrior,
    log_likelihood: MixedLogLikelihood,
    config: MixedPathSMCConfig,
    *,
    soft_guide: GaussianPathGuide | None = None,
) -> MixedPathSMCResult:
    """Anneal from a normalized guide mixture to the physical posterior.

    ``soft_guide`` affects proposals only.  Its normalized mixture density is
    included in the bridge weights and every guide Metropolis ratio, so the
    beta=1 target remains the exact mixed-MCS prior times charge likelihood.
    """
    config.validate()
    if soft_guide is not None and soft_guide.dimension != prior.dimension:
        raise ValueError("soft guide dimension does not match mixed path prior")
    rng = np.random.default_rng(int(config.random_seed))
    particle_count = int(config.particles)
    mixture_probability = float(config.guide_prior_mixture_probability)
    hard_empty_probability = float(config.hard_empty_guide_probability)
    latents = [
        MixedMCSLatent(
            _sample_base_coefficients(
                prior.dimension,
                soft_guide,
                mixture_probability,
                rng,
            ),
            _sample_base_hard_scatters(prior, hard_empty_probability, rng),
        )
        for _ in range(particle_count)
    ]
    evaluations = 0
    invalid = 0

    def evaluate(latent: MixedMCSLatent) -> float:
        nonlocal evaluations, invalid
        evaluations += 1
        try:
            value = float(log_likelihood(latent))
        except Exception:
            value = -math.inf
        if not math.isfinite(value):
            invalid += 1
            return -math.inf
        return value

    likelihood = np.asarray([evaluate(latent) for latent in latents], dtype=np.float64)
    if not np.any(np.isfinite(likelihood)):
        raise RuntimeError("no finite mixed-path prior particle")
    weights = np.full(particle_count, 1.0 / particle_count, dtype=np.float64)
    log_prior = np.asarray(
        [_standard_normal_logpdf(latent.soft_coefficients) for latent in latents],
        dtype=np.float64,
    )
    log_base = np.asarray(
        [
            _base_logpdf(
                latent.soft_coefficients,
                soft_guide,
                mixture_probability,
            )
            for latent in latents
        ],
        dtype=np.float64,
    )
    hard_base_ratio = np.asarray(
        [
            _hard_base_log_ratio(
                len(latent.hard_scatters),
                prior.expected_hard_scatter_count,
                hard_empty_probability,
            )
            for latent in latents
        ],
        dtype=np.float64,
    )
    bridge_score = log_prior + likelihood - log_base - hard_base_ratio
    beta = 0.0
    log_evidence = 0.0
    stages: list[MixedPathSMCStage] = []

    def rejuvenate(temperature: float, sweeps: int):
        soft_accepts = 0
        hard_accepts = 0
        soft_attempts = 0
        hard_attempts = 0

        for _ in range(int(sweeps)):
            for index in range(particle_count):
                use_guide = bool(
                    soft_guide is not None
                    and rng.random() < float(config.soft_guide_probability)
                )
                if use_guide:
                    proposed_coefficients = _sample_base_coefficients(
                        prior.dimension,
                        soft_guide,
                        mixture_probability,
                        rng,
                    )
                    proposed = MixedMCSLatent(
                        proposed_coefficients,
                        _sample_base_hard_scatters(
                            prior, hard_empty_probability, rng
                        ),
                    )
                else:
                    proposed = prior.pcn_soft(
                        latents[index], float(config.soft_pcn_rho), rng
                    )
                proposed_likelihood = evaluate(proposed)
                soft_attempts += 1
                proposed_prior = _standard_normal_logpdf(
                    proposed.soft_coefficients
                )
                proposed_base = _base_logpdf(
                    proposed.soft_coefficients,
                    soft_guide,
                    mixture_probability,
                )
                proposed_hard_base_ratio = _hard_base_log_ratio(
                    len(proposed.hard_scatters),
                    prior.expected_hard_scatter_count,
                    hard_empty_probability,
                )
                if use_guide:
                    # This is an independence draw from the complete
                    # normalized soft-guide/physical-prior mixture times the
                    # defensive empty/Poisson hard mixture.  Its exact density
                    # cancels, leaving only the annealed bridge score.
                    proposed_bridge = (
                        proposed_prior
                        + proposed_likelihood
                        - proposed_base
                        - proposed_hard_base_ratio
                    )
                    soft_log_ratio = temperature * (
                        proposed_bridge - bridge_score[index]
                    )
                else:
                    # pCN is reversible with respect to the standard-normal
                    # physical soft prior and retains the hard state.
                    soft_log_ratio = (
                        (1.0 - temperature)
                        * (
                            (proposed_base - proposed_prior)
                            - (log_base[index] - log_prior[index])
                        )
                        + temperature
                        * (proposed_likelihood - likelihood[index])
                    )
                if (
                    math.isfinite(proposed_likelihood)
                    and math.log(
                        max(float(rng.random()), np.finfo(np.float64).tiny)
                    ) < min(0.0, soft_log_ratio)
                ):
                    latents[index] = proposed
                    likelihood[index] = proposed_likelihood
                    log_prior[index] = proposed_prior
                    log_base[index] = proposed_base
                    hard_base_ratio[index] = proposed_hard_base_ratio
                    bridge_score[index] = (
                        proposed_prior
                        + proposed_likelihood
                        - proposed_base
                        - proposed_hard_base_ratio
                    )
                    soft_accepts += 1

                proposed = prior.refresh_hard_scatters(
                    latents[index],
                    float(config.hard_retention_probability),
                    rng,
                )
                proposed_likelihood = evaluate(proposed)
                hard_attempts += 1
                proposed_hard_base_ratio = _hard_base_log_ratio(
                    len(proposed.hard_scatters),
                    prior.expected_hard_scatter_count,
                    hard_empty_probability,
                )
                hard_log_ratio = (
                    (1.0 - temperature)
                    * (proposed_hard_base_ratio - hard_base_ratio[index])
                    + temperature
                    * (proposed_likelihood - likelihood[index])
                )
                if (
                    math.isfinite(proposed_likelihood)
                    and math.log(
                        max(float(rng.random()), np.finfo(np.float64).tiny)
                    ) < min(0.0, hard_log_ratio)
                ):
                    latents[index] = proposed
                    likelihood[index] = proposed_likelihood
                    hard_base_ratio[index] = proposed_hard_base_ratio
                    bridge_score[index] = (
                        log_prior[index]
                        + proposed_likelihood
                        - log_base[index]
                        - proposed_hard_base_ratio
                    )
                    hard_accepts += 1
        return (
            soft_accepts / max(soft_attempts, 1),
            hard_accepts / max(hard_attempts, 1),
        )

    while beta < 1.0 - 1.0e-12:
        if len(stages) >= int(config.max_temperatures):
            raise RuntimeError("mixed-path SMC exceeded max_temperatures")
        next_beta = _next_temperature(
            beta,
            weights,
            bridge_score,
            float(config.target_ess_fraction) * particle_count,
        )
        increment = (next_beta - beta) * bridge_score
        updated, log_normalizer = _normalize_log_weights(
            np.log(np.maximum(weights, np.finfo(np.float64).tiny)) + increment
        )
        # The current normalized weights already integrate to one.
        log_evidence += log_normalizer
        current_ess = _ess(updated)
        # CESS controls only the incremental bridge, so accumulated weights may
        # safely cross several temperatures before an actual-ESS resample.  The
        # final resample supplies equally weighted posterior MCMC chains.
        resampled = bool(
            current_ess < float(config.resample_ess_fraction) * particle_count
            or next_beta >= 1.0 - 1.0e-12
        )
        if resampled:
            indices = _systematic_resample(updated, rng)
            latents = [latents[int(index)] for index in indices]
            likelihood = likelihood[indices].copy()
            log_prior = log_prior[indices].copy()
            log_base = log_base[indices].copy()
            hard_base_ratio = hard_base_ratio[indices].copy()
            bridge_score = bridge_score[indices].copy()
            weights = np.full(particle_count, 1.0 / particle_count)
        else:
            weights = updated
        soft_acceptance, hard_acceptance = rejuvenate(
            next_beta, int(config.rejuvenation_steps)
        )
        stages.append(
            MixedPathSMCStage(
                beta_before=float(beta),
                beta_after=float(next_beta),
                ess_before_resampling=float(current_ess),
                resampled=bool(resampled),
                soft_acceptance=float(soft_acceptance),
                hard_acceptance=float(hard_acceptance),
            )
        )
        beta = next_beta

    posterior_sweeps = int(config.posterior_rejuvenation_steps)
    if posterior_sweeps:
        soft_acceptance, hard_acceptance = rejuvenate(1.0, posterior_sweeps)
        stages.append(
            MixedPathSMCStage(
                beta_before=1.0,
                beta_after=1.0,
                ess_before_resampling=float(_ess(weights)),
                resampled=False,
                soft_acceptance=float(soft_acceptance),
                hard_acceptance=float(hard_acceptance),
            )
        )
    return MixedPathSMCResult(
        latents=tuple(latents),
        log_likelihood=np.ascontiguousarray(likelihood),
        weights=np.ascontiguousarray(weights),
        stages=tuple(stages),
        likelihood_evaluations=int(evaluations),
        invalid_likelihood_evaluations=int(invalid),
        log_evidence=float(log_evidence),
        random_seed=int(config.random_seed),
        posterior_trajectory_sweeps=posterior_sweeps,
    )


__all__ = [
    "MixedPathSMCConfig",
    "MixedPathSMCResult",
    "MixedPathSMCStage",
    "run_mixed_path_smc",
]
