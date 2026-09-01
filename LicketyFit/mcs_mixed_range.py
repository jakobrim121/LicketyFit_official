"""Continuous fixed-energy range inference with mixed coherent MCS paths.

The non-centred state is ``z_R ~ N(0,1)`` and a conditional mixed-MCS path.
For a range move, a new hard marked-Poisson path is drawn from its physical
conditional prior while the standardized soft coordinates are retained.  The
conditional hard-prior densities cancel exactly in the Metropolis ratio; no
range bins, rounded coordinates, or WCSim-derived scales are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .mcs_joint_energy_range import GaussianPathGuide
from .mcs_mixed_latent import (
    _base_logpdf,
    _hard_base_log_ratio,
    _next_temperature,
    _normalize_log_weights,
    _sample_base_coefficients,
    _sample_base_hard_scatters,
    _systematic_resample,
    _standard_normal_logpdf,
)
from .mcs_mixed_path import MixedMCSLatent, MixedMCSPathPrior


@dataclass(frozen=True)
class MixedRangeContext:
    realized_range_mm: float
    path_prior: MixedMCSPathPrior
    log_likelihood: Callable[[MixedMCSLatent], float]

    def __post_init__(self) -> None:
        if (
            not math.isfinite(float(self.realized_range_mm))
            or float(self.realized_range_mm) <= 0.0
        ):
            raise ValueError("realized range must be positive and finite")


MixedRangeContextFactory = Callable[[float], MixedRangeContext]


@dataclass(frozen=True)
class MixedRangeSMCConfig:
    particles: int = 32
    target_ess_fraction: float = 0.80
    resample_ess_fraction: float = 0.55
    max_temperatures: int = 64
    rejuvenation_steps: int = 3
    posterior_rejuvenation_steps: int = 12
    range_pcn_rho: float = 0.80
    soft_pcn_rho: float = 0.995
    hard_retention_probability: float = 0.72
    soft_guide_probability: float = 0.50
    guide_prior_mixture_probability: float = 0.20
    hard_empty_guide_probability: float = 0.70
    initialization_attempts_per_particle: int = 100
    random_seed: int = 41873

    def validate(self) -> None:
        if int(self.particles) < 2:
            raise ValueError("SMC requires at least two particles")
        for name, value in (
            ("target_ess_fraction", self.target_ess_fraction),
            ("resample_ess_fraction", self.resample_ess_fraction),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must lie in (0,1]")
        for name, value in (
            ("range_pcn_rho", self.range_pcn_rho),
            ("soft_pcn_rho", self.soft_pcn_rho),
            ("hard_retention_probability", self.hard_retention_probability),
            ("soft_guide_probability", self.soft_guide_probability),
            ("hard_empty_guide_probability", self.hard_empty_guide_probability),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must lie in [0,1)")
        if not 0.0 < float(self.guide_prior_mixture_probability) < 1.0:
            raise ValueError("guide_prior_mixture_probability must lie in (0,1)")
        if int(self.max_temperatures) < 1:
            raise ValueError("max_temperatures must be positive")
        if int(self.rejuvenation_steps) < 0:
            raise ValueError("rejuvenation_steps must be non-negative")
        if int(self.posterior_rejuvenation_steps) < 0:
            raise ValueError("posterior_rejuvenation_steps must be non-negative")
        if int(self.initialization_attempts_per_particle) < 1:
            raise ValueError("initialization_attempts_per_particle must be positive")


@dataclass(frozen=True)
class MixedRangeSMCStage:
    beta_before: float
    beta_after: float
    ess_before_resampling: float
    resampled: bool
    range_acceptance: float
    soft_acceptance: float
    hard_acceptance: float


@dataclass
class MixedRangeSMCResult:
    z_range: np.ndarray
    realized_range_mm: np.ndarray
    latents: tuple[MixedMCSLatent, ...]
    log_likelihood: np.ndarray
    weights: np.ndarray
    stages: tuple[MixedRangeSMCStage, ...]
    likelihood_evaluations: int
    invalid_likelihood_evaluations: int
    context_builds: int
    invalid_context_builds: int
    log_evidence: float
    random_seed: int

    def weighted_mean(self, values) -> float:
        return float(np.dot(self.weights, np.asarray(values, dtype=np.float64)))

    def weighted_sd(self, values) -> float:
        array = np.asarray(values, dtype=np.float64)
        mean = self.weighted_mean(array)
        return float(math.sqrt(max(np.dot(self.weights, (array - mean) ** 2), 0.0)))

    @property
    def hard_scatter_count(self) -> np.ndarray:
        return np.asarray(
            [len(latent.hard_scatters) for latent in self.latents], dtype=np.int32
        )


def run_mixed_range_smc(
    context_factory: MixedRangeContextFactory,
    dimension: int,
    config: MixedRangeSMCConfig,
    *,
    soft_guide: GaussianPathGuide | None = None,
) -> MixedRangeSMCResult:
    """Anneal the continuous physical range/path prior to the charge posterior."""
    config.validate()
    dimension = int(dimension)
    if dimension < 2 or dimension % 2:
        raise ValueError("dimension must contain equal non-empty plane blocks")
    if soft_guide is not None and soft_guide.dimension != dimension:
        raise ValueError("soft guide dimension does not match mixed path prior")
    rng = np.random.default_rng(int(config.random_seed))
    count = int(config.particles)
    mixture_probability = float(config.guide_prior_mixture_probability)
    hard_empty_probability = float(config.hard_empty_guide_probability)
    context_builds = 0
    invalid_context_builds = 0
    likelihood_evaluations = 0
    invalid_likelihood_evaluations = 0

    def make_context(z_value: float):
        nonlocal context_builds, invalid_context_builds
        context_builds += 1
        try:
            context = context_factory(float(z_value))
        except Exception:
            invalid_context_builds += 1
            return None
        if context.path_prior.dimension != dimension:
            raise ValueError("context path dimension changed across range")
        return context

    def evaluate(context: MixedRangeContext, latent: MixedMCSLatent) -> float:
        nonlocal likelihood_evaluations, invalid_likelihood_evaluations
        likelihood_evaluations += 1
        try:
            value = float(context.log_likelihood(latent))
        except Exception:
            value = -math.inf
        if not math.isfinite(value):
            invalid_likelihood_evaluations += 1
            return -math.inf
        return value

    z_range: list[float] = []
    contexts: list[MixedRangeContext] = []
    latents: list[MixedMCSLatent] = []
    likelihood: list[float] = []
    for _particle in range(count):
        for _attempt in range(int(config.initialization_attempts_per_particle)):
            z_value = float(rng.normal())
            context = make_context(z_value)
            if context is None:
                continue
            latent = MixedMCSLatent(
                _sample_base_coefficients(
                    dimension, soft_guide, mixture_probability, rng
                ),
                _sample_base_hard_scatters(
                    context.path_prior, hard_empty_probability, rng
                ),
            )
            value = evaluate(context, latent)
            if math.isfinite(value):
                z_range.append(z_value)
                contexts.append(context)
                latents.append(latent)
                likelihood.append(value)
                break
        else:
            raise RuntimeError("failed to initialize a finite mixed-range particle")

    z = np.asarray(z_range, dtype=np.float64)
    loglike = np.asarray(likelihood, dtype=np.float64)
    log_prior = np.asarray(
        [_standard_normal_logpdf(latent.soft_coefficients) for latent in latents],
        dtype=np.float64,
    )
    log_base = np.asarray(
        [
            _base_logpdf(
                latent.soft_coefficients, soft_guide, mixture_probability
            )
            for latent in latents
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
    beta = 0.0
    log_evidence = 0.0
    stages: list[MixedRangeSMCStage] = []

    def accept(log_ratio: float) -> bool:
        return math.log(
            max(float(rng.random()), np.finfo(np.float64).tiny)
        ) < min(0.0, float(log_ratio))

    def rejuvenate(temperature: float, sweeps: int):
        attempts = {"range": 0, "soft": 0, "hard": 0}
        accepts = {key: 0 for key in attempts}
        for _ in range(int(sweeps)):
            for index in rng.permutation(count):
                # z_R pCN is reversible under the physical standard-normal
                # straggling prior.  Transport the same hard curvature at fixed
                # fractional arc length and include its exact marked-Poisson
                # prior ratio plus continuous-coordinate Jacobian.
                attempts["range"] += 1
                rho = float(config.range_pcn_rho)
                proposed_z = (
                    rho * z[index]
                    + math.sqrt(1.0 - rho * rho) * float(rng.normal())
                )
                proposed_context = make_context(proposed_z)
                if proposed_context is not None:
                    proposed_latent = contexts[index].path_prior.transport_latent_to(
                        proposed_context.path_prior,
                        latents[index],
                    )
                    proposed_loglike = evaluate(proposed_context, proposed_latent)
                    hard_prior_jacobian = (
                        contexts[index].path_prior.hard_transport_log_prior_jacobian(
                            proposed_context.path_prior,
                            latents[index],
                        )
                    )
                    proposed_hard_ratio = _hard_base_log_ratio(
                        len(proposed_latent.hard_scatters),
                        proposed_context.path_prior.expected_hard_scatter_count,
                        hard_empty_probability,
                    )
                    ratio = (
                        hard_prior_jacobian
                        + (1.0 - temperature)
                        * (proposed_hard_ratio - hard_base_ratio[index])
                        + temperature * (proposed_loglike - loglike[index])
                    )
                    if math.isfinite(proposed_loglike) and accept(ratio):
                        z[index] = proposed_z
                        contexts[index] = proposed_context
                        latents[index] = proposed_latent
                        loglike[index] = proposed_loglike
                        hard_base_ratio[index] = proposed_hard_ratio
                        bridge_score[index] = (
                            log_prior[index]
                            + proposed_loglike
                            - log_base[index]
                            - proposed_hard_ratio
                        )
                        accepts["range"] += 1

                attempts["soft"] += 1
                use_guide = bool(
                    soft_guide is not None
                    and rng.random() < float(config.soft_guide_probability)
                )
                if use_guide:
                    proposed_coefficients = _sample_base_coefficients(
                        dimension, soft_guide, mixture_probability, rng
                    )
                    proposed_latent = MixedMCSLatent(
                        proposed_coefficients,
                        _sample_base_hard_scatters(
                            contexts[index].path_prior,
                            hard_empty_probability,
                            rng,
                        ),
                    )
                else:
                    proposed_latent = contexts[index].path_prior.pcn_soft(
                        latents[index], float(config.soft_pcn_rho), rng
                    )
                proposed_loglike = evaluate(contexts[index], proposed_latent)
                proposed_prior = _standard_normal_logpdf(
                    proposed_latent.soft_coefficients
                )
                proposed_base = _base_logpdf(
                    proposed_latent.soft_coefficients,
                    soft_guide,
                    mixture_probability,
                )
                proposed_hard_ratio = _hard_base_log_ratio(
                    len(proposed_latent.hard_scatters),
                    contexts[index].path_prior.expected_hard_scatter_count,
                    hard_empty_probability,
                )
                if use_guide:
                    proposed_bridge = (
                        proposed_prior
                        + proposed_loglike
                        - proposed_base
                        - proposed_hard_ratio
                    )
                    ratio = temperature * (
                        proposed_bridge - bridge_score[index]
                    )
                else:
                    ratio = (
                        (1.0 - temperature)
                        * (
                            (proposed_base - proposed_prior)
                            - (log_base[index] - log_prior[index])
                        )
                        + temperature * (proposed_loglike - loglike[index])
                    )
                if math.isfinite(proposed_loglike) and accept(ratio):
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
                    accepts["soft"] += 1

                attempts["hard"] += 1
                proposed_latent = contexts[index].path_prior.refresh_hard_scatters(
                    latents[index],
                    float(config.hard_retention_probability),
                    rng,
                )
                proposed_loglike = evaluate(contexts[index], proposed_latent)
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
        return {
            name: accepts[name] / max(attempts[name], 1) for name in attempts
        }

    while beta < 1.0 - 1.0e-12:
        if len(stages) >= int(config.max_temperatures):
            raise RuntimeError("mixed-range SMC exceeded max_temperatures")
        next_beta = _next_temperature(
            beta,
            weights,
            bridge_score,
            float(config.target_ess_fraction) * count,
        )
        increment = (next_beta - beta) * bridge_score
        weights, log_normalizer = _normalize_log_weights(
            np.log(np.maximum(weights, np.finfo(np.float64).tiny)) + increment
        )
        log_evidence += log_normalizer
        ess = float(1.0 / np.sum(weights * weights))
        resampled = bool(
            ess < float(config.resample_ess_fraction) * count
            or next_beta >= 1.0 - 1.0e-12
        )
        if resampled:
            indices = _systematic_resample(weights, rng)
            z = z[indices].copy()
            contexts = [contexts[int(index)] for index in indices]
            latents = [latents[int(index)] for index in indices]
            loglike = loglike[indices].copy()
            log_prior = log_prior[indices].copy()
            log_base = log_base[indices].copy()
            hard_base_ratio = hard_base_ratio[indices].copy()
            bridge_score = bridge_score[indices].copy()
            weights.fill(1.0 / count)
        acceptance = rejuvenate(next_beta, int(config.rejuvenation_steps))
        stages.append(
            MixedRangeSMCStage(
                beta_before=float(beta),
                beta_after=float(next_beta),
                ess_before_resampling=float(ess),
                resampled=resampled,
                range_acceptance=float(acceptance["range"]),
                soft_acceptance=float(acceptance["soft"]),
                hard_acceptance=float(acceptance["hard"]),
            )
        )
        beta = next_beta

    posterior_steps = int(config.posterior_rejuvenation_steps)
    if posterior_steps:
        acceptance = rejuvenate(1.0, posterior_steps)
        stages.append(
            MixedRangeSMCStage(
                beta_before=1.0,
                beta_after=1.0,
                ess_before_resampling=float(1.0 / np.sum(weights * weights)),
                resampled=False,
                range_acceptance=float(acceptance["range"]),
                soft_acceptance=float(acceptance["soft"]),
                hard_acceptance=float(acceptance["hard"]),
            )
        )
    realized = np.asarray(
        [context.realized_range_mm for context in contexts], dtype=np.float64
    )
    return MixedRangeSMCResult(
        z_range=np.ascontiguousarray(z),
        realized_range_mm=np.ascontiguousarray(realized),
        latents=tuple(latents),
        log_likelihood=np.ascontiguousarray(loglike),
        weights=np.ascontiguousarray(weights),
        stages=tuple(stages),
        likelihood_evaluations=int(likelihood_evaluations),
        invalid_likelihood_evaluations=int(invalid_likelihood_evaluations),
        context_builds=int(context_builds),
        invalid_context_builds=int(invalid_context_builds),
        log_evidence=float(log_evidence),
        random_seed=int(config.random_seed),
    )


__all__ = [
    "MixedRangeContext",
    "MixedRangeSMCConfig",
    "MixedRangeSMCResult",
    "MixedRangeSMCStage",
    "run_mixed_range_smc",
]
