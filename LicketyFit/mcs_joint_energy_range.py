"""Continuous non-centred initial-energy/range/path inference.

The physical state is

``K0 ~ Uniform(Kmin, Kmax)``, ``z_R ~ N(0,1)``,
``L = R(K0) + sigma_R(K0) z_R``, and ``u ~ N(0,I)``.

``u`` contains standardized coherent Fermi--Eyges path coordinates.  The
uniform energy distribution is deliberately a broad reconstruction support,
not a beam-energy constraint.  The standard-normal range and path priors are
the published transport model.  Annealed SMC targets the exact nonlinear
charge likelihood; all rejuvenation kernels either preserve these priors or
include their exact proposal-density correction.

There is intentionally no range grid, rounded coordinate, WCSim-derived scale,
or event-truth entry point in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np
from scipy.special import ndtr, ndtri


LogLikelihood = Callable[[float, float, np.ndarray], float]


@dataclass(frozen=True)
class JointEnergyRangeSMCConfig:
    kinetic_energy_bounds_mev: tuple[float, float]
    particles: int = 32
    target_ess_fraction: float = 0.80
    resample_ess_fraction: float = 0.55
    max_temperatures: int = 64
    rejuvenation_steps: int = 3
    posterior_rejuvenation_steps: int = 12
    energy_random_walk_mev: float = 30.0
    energy_independence_probability: float = 0.15
    range_pcn_rho: float = 0.80
    path_pcn_rho: float = 0.92
    path_guide_rho: float = 0.75
    path_guide_probability: float = 0.50
    joint_guide_prior_mixture_probability: float = 0.20
    joint_guide_initial_beta: float = 0.0
    initialization_attempts_per_particle: int = 200
    random_seed: int = 41873

    def validate(self) -> None:
        lo, hi = map(float, self.kinetic_energy_bounds_mev)
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo < hi):
            raise ValueError("kinetic-energy bounds must satisfy 0 < low < high")
        if int(self.particles) < 2:
            raise ValueError("SMC requires at least two particles")
        for name, value in (
            ("target_ess_fraction", self.target_ess_fraction),
            ("resample_ess_fraction", self.resample_ess_fraction),
            ("energy_independence_probability", self.energy_independence_probability),
            ("path_guide_probability", self.path_guide_probability),
            (
                "joint_guide_prior_mixture_probability",
                self.joint_guide_prior_mixture_probability,
            ),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if not 0.0 < float(self.joint_guide_prior_mixture_probability) < 1.0:
            raise ValueError(
                "joint_guide_prior_mixture_probability must lie in (0,1)"
            )
        if float(self.joint_guide_initial_beta) != 0.0:
            raise ValueError(
                "joint_guide_initial_beta must be zero: guided SMC now "
                "anneals continuously from the normalized proposal mixture"
            )
        if not 0.0 < float(self.target_ess_fraction) <= 1.0:
            raise ValueError("target_ess_fraction must lie in (0,1]")
        if not 0.0 < float(self.resample_ess_fraction) <= 1.0:
            raise ValueError("resample_ess_fraction must lie in (0,1]")
        for name, value in (
            ("range_pcn_rho", self.range_pcn_rho),
            ("path_pcn_rho", self.path_pcn_rho),
            ("path_guide_rho", self.path_guide_rho),
        ):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must lie in [0,1)")
        if int(self.max_temperatures) < 1:
            raise ValueError("max_temperatures must be positive")
        if int(self.rejuvenation_steps) < 0:
            raise ValueError("rejuvenation_steps must be non-negative")
        if int(self.posterior_rejuvenation_steps) < 0:
            raise ValueError("posterior_rejuvenation_steps must be non-negative")
        if float(self.energy_random_walk_mev) <= 0.0:
            raise ValueError("energy_random_walk_mev must be positive")
        if int(self.initialization_attempts_per_particle) < 1:
            raise ValueError("initialization_attempts_per_particle must be positive")


@dataclass(frozen=True)
class GaussianPathGuide:
    """Fixed Gaussian independence proposal for standardized path coordinates.

    A guide can be derived from charge data (for example a local FE MAP and
    Fisher covariance).  It is a proposal only: :meth:`logpdf` enters the
    Metropolis ratio, so it cannot alter the posterior target.
    """

    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("guide covariance shape does not match its mean")
        covariance = 0.5 * (covariance + covariance.T)
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("guide covariance must be positive definite") from exc
        logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
        inverse_chol = np.linalg.solve(chol, np.eye(mean.size, dtype=np.float64))
        precision = inverse_chol.T @ inverse_chol
        object.__setattr__(self, "mean", np.ascontiguousarray(mean))
        object.__setattr__(self, "covariance", np.ascontiguousarray(covariance))
        object.__setattr__(self, "_chol", np.ascontiguousarray(chol))
        object.__setattr__(self, "_precision", np.ascontiguousarray(precision))
        object.__setattr__(self, "_logdet", logdet)

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return np.ascontiguousarray(
            self.mean + self._chol @ rng.normal(size=self.dimension)
        )

    def logpdf(self, value: Sequence[float]) -> float:
        delta = np.asarray(value, dtype=np.float64).reshape(self.dimension) - self.mean
        return float(
            -0.5 * delta @ self._precision @ delta
            - 0.5 * self._logdet
            - 0.5 * self.dimension * math.log(2.0 * math.pi)
        )

    def bridge_sample(
        self,
        current: Sequence[float],
        rho: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw an AR(1) bridge toward the guide mean."""
        current_array = np.asarray(current, dtype=np.float64).reshape(self.dimension)
        correlation = float(rho)
        scale = math.sqrt(1.0 - correlation * correlation)
        return np.ascontiguousarray(
            correlation * current_array
            + (1.0 - correlation) * self.mean
            + scale * (self._chol @ rng.normal(size=self.dimension))
        )

    def bridge_logpdf(
        self,
        proposed: Sequence[float],
        current: Sequence[float],
        rho: float,
    ) -> float:
        correlation = float(rho)
        variance_scale = 1.0 - correlation * correlation
        current_array = np.asarray(current, dtype=np.float64).reshape(self.dimension)
        proposed_array = np.asarray(proposed, dtype=np.float64).reshape(self.dimension)
        mean = correlation * current_array + (1.0 - correlation) * self.mean
        delta = proposed_array - mean
        return float(
            -0.5 * (delta @ self._precision @ delta) / variance_scale
            - 0.5 * self._logdet
            - 0.5 * self.dimension * math.log(2.0 * math.pi * variance_scale)
        )


@dataclass(frozen=True)
class EnergyPathGuide:
    """One exact-proposal anchor for a joint energy/path basin move."""

    kinetic_energy_mev: float
    energy_proposal_sd_mev: float
    path: GaussianPathGuide
    mixture_weight: float = 1.0
    z_range_mean: float = 0.0
    z_range_proposal_sd: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.kinetic_energy_mev)):
            raise ValueError("guide kinetic energy must be finite")
        if (
            not math.isfinite(float(self.energy_proposal_sd_mev))
            or float(self.energy_proposal_sd_mev) <= 0.0
        ):
            raise ValueError("guide energy proposal width must be positive")
        if (
            not math.isfinite(float(self.mixture_weight))
            or float(self.mixture_weight) <= 0.0
        ):
            raise ValueError("guide mixture weight must be positive and finite")
        if not math.isfinite(float(self.z_range_mean)):
            raise ValueError("guide z-range mean must be finite")
        if (
            not math.isfinite(float(self.z_range_proposal_sd))
            or float(self.z_range_proposal_sd) <= 0.0
        ):
            raise ValueError("guide z-range proposal width must be positive")


@dataclass(frozen=True)
class SMCStage:
    beta_before: float
    beta_after: float
    ess_before_resampling: float
    resampled: bool
    energy_acceptance: float
    range_acceptance: float
    path_acceptance: float
    guide_acceptance: float


@dataclass
class JointEnergyRangeSMCResult:
    kinetic_energy_mev: np.ndarray
    z_range: np.ndarray
    realized_range_mm: np.ndarray
    coefficients: np.ndarray
    log_likelihood: np.ndarray
    weights: np.ndarray
    stages: tuple[SMCStage, ...]
    likelihood_evaluations: int
    invalid_likelihood_evaluations: int
    log_evidence: float
    random_seed: int
    kinetic_energy_bounds_mev: tuple[float, float]
    initialization_beta: float = 0.0
    initialization_ess: float = math.nan
    posterior_trajectory_sweeps: int = 0

    @property
    def uses_discrete_range_grid(self) -> bool:
        return False

    @property
    def unique_realized_ranges(self) -> int:
        return int(np.unique(self.realized_range_mm).size)

    def weighted_mean(self, values) -> float:
        return float(np.dot(self.weights, np.asarray(values, dtype=np.float64)))

    def weighted_variance(self, values) -> float:
        array = np.asarray(values, dtype=np.float64)
        mean = self.weighted_mean(array)
        return float(np.dot(self.weights, (array - mean) ** 2))

    def weighted_covariance(self, first, second) -> float:
        x = np.asarray(first, dtype=np.float64)
        y = np.asarray(second, dtype=np.float64)
        return float(
            np.dot(
                self.weights,
                (x - self.weighted_mean(x)) * (y - self.weighted_mean(y)),
            )
        )

    def summary(self) -> dict:
        energy_mean = self.weighted_mean(self.kinetic_energy_mev)
        energy_variance = self.weighted_variance(self.kinetic_energy_mev)
        range_mean = self.weighted_mean(self.realized_range_mm)
        range_variance = self.weighted_variance(self.realized_range_mm)
        z_mean = self.weighted_mean(self.z_range)
        z_variance = self.weighted_variance(self.z_range)
        covariance = self.weighted_covariance(
            self.kinetic_energy_mev, self.realized_range_mm
        )
        denominator = math.sqrt(max(energy_variance * range_variance, 0.0))
        return {
            "initial_kinetic_energy_mev_mean": energy_mean,
            "initial_kinetic_energy_mev_sd": math.sqrt(max(energy_variance, 0.0)),
            "realized_range_mm_mean": range_mean,
            "realized_range_mm_sd": math.sqrt(max(range_variance, 0.0)),
            "z_range_mean": z_mean,
            "z_range_sd": math.sqrt(max(z_variance, 0.0)),
            "energy_range_covariance_mev_mm": covariance,
            "energy_range_correlation": (
                covariance / denominator if denominator > 0.0 else math.nan
            ),
            "unique_realized_ranges": self.unique_realized_ranges,
            "uses_discrete_range_grid": False,
            "range_coordinate": "continuous_noncentered_float64_z_R",
            "energy_coordinate": "continuous_float64_K0_mev",
            "output_length_quantization_mm": None,
            "likelihood_evaluations": int(self.likelihood_evaluations),
            "invalid_likelihood_evaluations": int(
                self.invalid_likelihood_evaluations
            ),
            "temperature_stages": int(len(self.stages)),
            "log_evidence": float(self.log_evidence),
            "random_seed": int(self.random_seed),
            "kinetic_energy_bounds_mev": list(self.kinetic_energy_bounds_mev),
            "initialization_beta": float(self.initialization_beta),
            "initialization_ess": float(self.initialization_ess),
            "posterior_trajectory_sweeps": int(
                self.posterior_trajectory_sweeps
            ),
        }


def _normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        raise RuntimeError("every SMC particle has zero likelihood weight")
    maximum = float(np.max(log_weights[finite]))
    scaled = np.zeros_like(log_weights, dtype=np.float64)
    scaled[finite] = np.exp(log_weights[finite] - maximum)
    total = float(np.sum(scaled))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("SMC weight normalization failed")
    weights = scaled / total
    log_normalizer = maximum + math.log(total)
    return np.ascontiguousarray(weights), float(log_normalizer)


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.asarray(weights, dtype=np.float64) ** 2))


def _systematic_resample(
    weights: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n = int(weights.size)
    positions = (float(rng.random()) + np.arange(n, dtype=np.float64)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


def _reflected_interval(value: float, low: float, high: float) -> float:
    width = high - low
    coordinate = (float(value) - low) % (2.0 * width)
    return float(low + (coordinate if coordinate <= width else 2.0 * width - coordinate))


def _standard_normal_logpdf(value: np.ndarray | float) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(-0.5 * np.sum(array * array) - 0.5 * array.size * math.log(2.0 * math.pi))


def _logsumexp(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array))
    if not math.isfinite(maximum):
        return -math.inf
    return float(maximum + math.log(float(np.sum(np.exp(array - maximum)))))


def _normal_cdf(value: float) -> float:
    return float(ndtr(float(value)))


def _truncated_normal_logpdf(
    value: float,
    mean: float,
    sigma: float,
    low: float,
    high: float,
) -> float:
    if value < low or value > high:
        return -math.inf
    alpha = (low - mean) / sigma
    beta = (high - mean) / sigma
    normalizer = _normal_cdf(beta) - _normal_cdf(alpha)
    if normalizer <= 0.0:
        return -math.inf
    standardized = (value - mean) / sigma
    return float(
        -0.5 * standardized * standardized
        - math.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - math.log(normalizer)
    )


def _sample_truncated_normal(
    mean: float,
    sigma: float,
    low: float,
    high: float,
    rng: np.random.Generator,
) -> float:
    cdf_low = float(ndtr((low - mean) / sigma))
    cdf_high = float(ndtr((high - mean) / sigma))
    if not cdf_high > cdf_low:
        raise RuntimeError("truncated energy-guide interval has zero probability")
    probability = float(rng.uniform(cdf_low, cdf_high))
    # Keep ndtri away from infinities in an endpoint-rounding edge case.
    probability = float(np.clip(
        probability,
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    ))
    return float(np.clip(mean + sigma * float(ndtri(probability)), low, high))


def _normalized_guide_weights(
    guides: Sequence[EnergyPathGuide],
) -> np.ndarray:
    weights = np.asarray(
        [float(guide.mixture_weight) for guide in guides], dtype=np.float64
    )
    if (
        weights.size == 0
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("energy/path guide weights must be positive and finite")
    return np.ascontiguousarray(weights / float(np.sum(weights)))


def _joint_guide_logpdf(
    energy: float,
    z_range: float,
    coefficients: np.ndarray,
    guides: Sequence[EnergyPathGuide],
    *,
    low: float,
    high: float,
    prior_mixture_probability: float,
) -> float:
    epsilon = float(prior_mixture_probability)
    components = [
        math.log(epsilon)
        - math.log(high - low)
        + _standard_normal_logpdf(float(z_range))
        + _standard_normal_logpdf(coefficients)
    ]
    guide_weights = _normalized_guide_weights(guides)
    for guide, normalized_weight in zip(guides, guide_weights, strict=True):
        components.append(
            math.log1p(-epsilon)
            + math.log(float(normalized_weight))
            + _truncated_normal_logpdf(
                float(energy),
                float(guide.kinetic_energy_mev),
                float(guide.energy_proposal_sd_mev),
                low,
                high,
            )
            + float(
                -0.5
                * (
                    (float(z_range) - float(guide.z_range_mean))
                    / float(guide.z_range_proposal_sd)
                )
                ** 2
                - math.log(float(guide.z_range_proposal_sd))
                - 0.5 * math.log(2.0 * math.pi)
            )
            + guide.path.logpdf(coefficients)
        )
    return float(_logsumexp(components))


def _sample_joint_guide(
    guides: Sequence[EnergyPathGuide],
    *,
    low: float,
    high: float,
    prior_mixture_probability: float,
    dimension: int,
    rng: np.random.Generator,
    component_index: int | None = None,
) -> tuple[float, float, np.ndarray]:
    """Sample the normalized defensive joint-guide mixture.

    ``component_index`` is a sampling-only control for randomized stratified
    initialization. ``-1`` selects the defensive physical-prior component and
    ``0..len(guides)-1`` selects one fitted guide. The density remains the
    complete mixture in :func:`_joint_guide_logpdf`, so forcing a component
    improves finite-particle basin coverage without changing the target.
    """
    if component_index is not None and not (
        -1 <= int(component_index) < len(guides)
    ):
        raise ValueError("joint-guide component index is out of range")
    choose_prior = (
        int(component_index) == -1
        if component_index is not None
        else rng.random() < float(prior_mixture_probability)
    )
    if choose_prior:
        energy = float(rng.uniform(low, high))
        z_range = float(rng.normal())
        coefficients = np.ascontiguousarray(rng.normal(size=dimension))
    else:
        if component_index is None:
            guide_weights = _normalized_guide_weights(guides)
            guide_index = int(rng.choice(len(guides), p=guide_weights))
        else:
            guide_index = int(component_index)
        guide = guides[guide_index]
        energy = _sample_truncated_normal(
            float(guide.kinetic_energy_mev),
            float(guide.energy_proposal_sd_mev),
            low,
            high,
            rng,
        )
        z_range = float(
            guide.z_range_mean
            + guide.z_range_proposal_sd * rng.normal()
        )
        coefficients = guide.path.sample(rng)
    return energy, z_range, coefficients


def _stratified_joint_guide_components(
    guides: Sequence[EnergyPathGuide],
    *,
    prior_mixture_probability: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Allocate mixture components by randomized systematic sampling.

    A multinomial draw can omit a narrow fitted basin when the particle count
    is comparable to the number of guide components. Systematic allocation has
    the same expected proportions and the usual floor/ceiling coverage. The
    returned convention is ``-1`` for the defensive prior and zero-based guide
    indices otherwise.
    """
    sample_count = int(count)
    if sample_count < 1:
        raise ValueError("stratified guide allocation requires a positive count")
    epsilon = float(prior_mixture_probability)
    if not 0.0 < epsilon < 1.0:
        raise ValueError("prior mixture probability must lie in (0,1)")
    guide_weights = _normalized_guide_weights(guides)
    probabilities = np.concatenate(
        (np.asarray((epsilon,), dtype=np.float64), (1.0 - epsilon) * guide_weights)
    )
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    positions = (float(rng.random()) + np.arange(sample_count)) / sample_count
    allocation = np.searchsorted(cumulative, positions, side="right") - 1
    rng.shuffle(allocation)
    return np.ascontiguousarray(allocation, dtype=np.int64)


def _conditional_ess(
    weights: np.ndarray, log_likelihood: np.ndarray, delta_beta: float
) -> float:
    """Conditional ESS for only the proposed incremental bridge weights."""
    weights = np.asarray(weights, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        increment = float(delta_beta) * log_likelihood
    increment = np.where(np.isfinite(log_likelihood), increment, -np.inf)
    maximum = float(np.max(increment))
    if not math.isfinite(maximum):
        return 0.0
    ratio = np.exp(np.clip(increment - maximum, -745.0, 0.0))
    first = float(np.dot(weights, ratio))
    second = float(np.dot(weights, ratio * ratio))
    if not math.isfinite(second) or second <= 0.0:
        return 0.0
    return float(weights.size * first * first / second)


def _next_temperature(
    beta: float,
    weights: np.ndarray,
    log_likelihood: np.ndarray,
    target_ess: float,
) -> float:
    remaining = 1.0 - float(beta)
    if remaining <= 1.0e-12:
        return 1.0
    if _conditional_ess(weights, log_likelihood, remaining) >= target_ess:
        return 1.0
    low = 0.0
    high = remaining
    for _ in range(60):
        middle = 0.5 * (low + high)
        if _conditional_ess(weights, log_likelihood, middle) < target_ess:
            high = middle
        else:
            low = middle
    # ``high`` is the first side that reaches the requested information gain.
    return float(min(1.0, beta + max(high, 1.0e-8)))


def run_joint_energy_range_smc(
    log_likelihood: LogLikelihood,
    realized_range_from_noncentered: Callable[[float, float], float],
    *,
    n_path_modes: int,
    config: JointEnergyRangeSMCConfig,
    path_guide: (
        GaussianPathGuide
        | EnergyPathGuide
        | Sequence[GaussianPathGuide | EnergyPathGuide]
        | None
    ) = None,
) -> JointEnergyRangeSMCResult:
    """Sample the continuous ``(K0,z_R,u)`` posterior by annealed SMC.

    ``log_likelihood`` must return the exact charge log likelihood (up to an
    additive constant) or ``-inf`` for an invalid physical state.  The callback
    must not include the standard-normal ``z_R`` or path priors; the proposal
    kernels preserve those priors analytically.
    """

    config.validate()
    dimension = int(n_path_modes)
    if dimension < 1:
        raise ValueError("n_path_modes must be positive")
    if path_guide is None:
        supplied_guides: tuple[GaussianPathGuide | EnergyPathGuide, ...] = ()
    elif isinstance(path_guide, (GaussianPathGuide, EnergyPathGuide)):
        supplied_guides = (path_guide,)
    else:
        supplied_guides = tuple(path_guide)
    path_guides = tuple(
        guide for guide in supplied_guides if isinstance(guide, GaussianPathGuide)
    )
    energy_path_guides = tuple(
        guide for guide in supplied_guides if isinstance(guide, EnergyPathGuide)
    )
    if len(path_guides) + len(energy_path_guides) != len(supplied_guides):
        raise TypeError("path_guide contains an unsupported proposal type")
    if any(guide.dimension != dimension for guide in path_guides):
        raise ValueError("path guide dimension does not match n_path_modes")
    if any(guide.path.dimension != dimension for guide in energy_path_guides):
        raise ValueError("energy/path guide dimension does not match n_path_modes")

    rng = np.random.default_rng(int(config.random_seed))
    low, high = map(float, config.kinetic_energy_bounds_mev)
    n = int(config.particles)
    energy = np.empty(n, dtype=np.float64)
    z_range = np.empty(n, dtype=np.float64)
    coefficients = np.empty((n, dimension), dtype=np.float64)
    ranges = np.empty(n, dtype=np.float64)
    loglike = np.full(n, -np.inf, dtype=np.float64)
    log_prior = np.empty(n, dtype=np.float64)
    log_base = np.empty(n, dtype=np.float64)
    bridge_score = np.full(n, -np.inf, dtype=np.float64)
    evaluations = 0
    invalid = 0

    def evaluate(k0: float, z_value: float, path: np.ndarray) -> tuple[float, float]:
        nonlocal evaluations, invalid
        evaluations += 1
        length = float(realized_range_from_noncentered(float(k0), float(z_value)))
        if not math.isfinite(length) or length <= 0.0:
            invalid += 1
            return -math.inf, length
        try:
            value = float(log_likelihood(float(k0), float(z_value), path))
        except (FloatingPointError, ValueError, RuntimeError):
            value = -math.inf
        if not math.isfinite(value):
            invalid += 1
            return -math.inf, length
        return value, length

    def state_log_prior(
        k0: float, z_value: float, path: np.ndarray
    ) -> float:
        if k0 < low or k0 > high:
            return -math.inf
        return float(
            -math.log(high - low)
            + _standard_normal_logpdf(z_value)
            + _standard_normal_logpdf(path)
        )

    def state_log_base(
        k0: float, z_value: float, path: np.ndarray
    ) -> float:
        if not energy_path_guides:
            return state_log_prior(k0, z_value, path)
        return _joint_guide_logpdf(
            k0,
            z_value,
            path,
            energy_path_guides,
            low=low,
            high=high,
            prior_mixture_probability=float(
                config.joint_guide_prior_mixture_probability
            ),
        )

    def tempered_log_target(
        beta_value: float,
        base_value: float,
        prior_value: float,
        likelihood_value: float,
    ) -> float:
        b = float(beta_value)
        if b <= 0.0:
            return float(base_value)
        if not (
            math.isfinite(base_value)
            and math.isfinite(prior_value)
            and math.isfinite(likelihood_value)
        ):
            return -math.inf
        return float(
            (1.0 - b) * base_value
            + b * (prior_value + likelihood_value)
        )

    for index in range(n):
        if energy_path_guides:
            k0, z_value, path = _sample_joint_guide(
                energy_path_guides,
                low=low,
                high=high,
                prior_mixture_probability=float(
                    config.joint_guide_prior_mixture_probability
                ),
                dimension=dimension,
                rng=rng,
            )
        else:
            k0 = float(rng.uniform(low, high))
            z_value = float(rng.normal())
            path = np.ascontiguousarray(rng.normal(size=dimension))
        value, length = evaluate(k0, z_value, path)
        energy[index] = k0
        z_range[index] = z_value
        coefficients[index] = path
        ranges[index] = length
        loglike[index] = value
        log_prior[index] = state_log_prior(k0, z_value, path)
        log_base[index] = state_log_base(k0, z_value, path)
        if math.isfinite(value):
            bridge_score[index] = (
                log_prior[index] + value - log_base[index]
            )
    if not np.any(np.isfinite(loglike)):
        raise RuntimeError("every initial SMC particle has invalid likelihood")

    # Start exactly from the normalized defensive guide mixture (or from the
    # physical prior when no joint guide is supplied) and anneal the complete
    # density ratio q -> prior*likelihood.  This avoids the high-dimensional
    # one-shot importance collapse caused by jumping directly to beta=0.1.
    beta = 0.0
    weights = np.full(n, 1.0 / n, dtype=np.float64)
    initialization_ess = float(n)
    stages: list[SMCStage] = []
    log_evidence = 0.0
    posterior_energy_samples: list[np.ndarray] = []
    posterior_z_samples: list[np.ndarray] = []
    posterior_range_samples: list[np.ndarray] = []
    posterior_path_samples: list[np.ndarray] = []
    posterior_loglike_samples: list[np.ndarray] = []

    for _stage_index in range(int(config.max_temperatures)):
        if beta >= 1.0 - 1.0e-12:
            break
        target_ess = max(2.0, float(config.target_ess_fraction) * n)
        beta_after = _next_temperature(beta, weights, bridge_score, target_ess)
        delta_beta = beta_after - beta
        previous_log_weights = np.log(
            np.maximum(weights, np.finfo(np.float64).tiny)
        )
        increment = np.where(
            np.isfinite(bridge_score), delta_beta * bridge_score, -np.inf
        )
        weights, log_norm = _normalize_log_weights(previous_log_weights + increment)
        log_evidence += float(log_norm)
        ess = _effective_sample_size(weights)

        # CESS measures only incremental bridge degeneracy.  Resample on the
        # actual accumulated ESS, plus once at the final target to initialize
        # equally weighted posterior trajectories.
        resampled = bool(
            ess < float(config.resample_ess_fraction) * n
            or beta_after >= 1.0 - 1.0e-12
        )
        if resampled:
            indices = _systematic_resample(weights, rng)
            energy = np.ascontiguousarray(energy[indices])
            z_range = np.ascontiguousarray(z_range[indices])
            ranges = np.ascontiguousarray(ranges[indices])
            coefficients = np.ascontiguousarray(coefficients[indices])
            loglike = np.ascontiguousarray(loglike[indices])
            log_prior = np.ascontiguousarray(log_prior[indices])
            log_base = np.ascontiguousarray(log_base[indices])
            bridge_score = np.ascontiguousarray(bridge_score[indices])
            weights.fill(1.0 / n)

        attempts = {
            "energy": 0,
            "range": 0,
            "path": 0,
            "guide": 0,
        }
        accepts = {key: 0 for key in attempts}

        def accept_log_ratio(log_ratio: float) -> bool:
            return math.log(max(float(rng.random()), np.finfo(float).tiny)) < min(
                0.0, float(log_ratio)
            )

        moves_this_stage = int(config.rejuvenation_steps)
        if beta_after >= 1.0 - 1.0e-12:
            moves_this_stage += int(config.posterior_rejuvenation_steps)
        for move_index in range(moves_this_stage):
            for index in rng.permutation(n):
                # Energy move in non-centred coordinates: retain z_R and the
                # standardized path, so every prior factor cancels.  A reflected
                # random walk and a uniform independence move are both reversible
                # under the broad uniform energy support.
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
                proposed_loglike, proposed_range = evaluate(
                    proposed_energy, z_range[index], coefficients[index]
                )
                proposed_log_prior = state_log_prior(
                    proposed_energy, z_range[index], coefficients[index]
                )
                proposed_log_base = state_log_base(
                    proposed_energy, z_range[index], coefficients[index]
                )
                proposed_target = tempered_log_target(
                    beta_after,
                    proposed_log_base,
                    proposed_log_prior,
                    proposed_loglike,
                )
                current_target = tempered_log_target(
                    beta_after,
                    log_base[index],
                    log_prior[index],
                    loglike[index],
                )
                if math.isfinite(proposed_loglike) and accept_log_ratio(
                    proposed_target - current_target
                ):
                    energy[index] = proposed_energy
                    ranges[index] = proposed_range
                    loglike[index] = proposed_loglike
                    log_prior[index] = proposed_log_prior
                    log_base[index] = proposed_log_base
                    bridge_score[index] = (
                        proposed_log_prior + proposed_loglike - proposed_log_base
                    )
                    accepts["energy"] += 1

                # pCN preserves the standard-normal stopping-range prior.
                attempts["range"] += 1
                rho = float(config.range_pcn_rho)
                proposed_z = rho * z_range[index] + math.sqrt(1.0 - rho * rho) * float(
                    rng.normal()
                )
                proposed_loglike, proposed_range = evaluate(
                    energy[index], proposed_z, coefficients[index]
                )
                proposed_log_prior = state_log_prior(
                    energy[index], proposed_z, coefficients[index]
                )
                proposed_log_base = state_log_base(
                    energy[index], proposed_z, coefficients[index]
                )
                proposed_target = tempered_log_target(
                    beta_after,
                    proposed_log_base,
                    proposed_log_prior,
                    proposed_loglike,
                )
                current_target = tempered_log_target(
                    beta_after,
                    log_base[index],
                    log_prior[index],
                    loglike[index],
                )
                # pCN is reversible with respect to the z_R prior.
                proposal_correction = log_prior[index] - proposed_log_prior
                if math.isfinite(proposed_loglike) and accept_log_ratio(
                    proposed_target - current_target + proposal_correction
                ):
                    z_range[index] = proposed_z
                    ranges[index] = proposed_range
                    loglike[index] = proposed_loglike
                    log_prior[index] = proposed_log_prior
                    log_base[index] = proposed_log_base
                    bridge_score[index] = (
                        proposed_log_prior + proposed_loglike - proposed_log_base
                    )
                    accepts["range"] += 1

                use_guide = bool(
                    (path_guides or energy_path_guides)
                    and rng.random() < float(config.path_guide_probability)
                )
                label = "guide" if use_guide else "path"
                attempts[label] += 1
                if use_guide and energy_path_guides:
                    epsilon = float(
                        config.joint_guide_prior_mixture_probability
                    )
                    proposed_energy, proposed_z, proposed_path = (
                        _sample_joint_guide(
                            energy_path_guides,
                            low=low,
                            high=high,
                            prior_mixture_probability=epsilon,
                            dimension=dimension,
                            rng=rng,
                        )
                    )
                    log_q_forward = _joint_guide_logpdf(
                        proposed_energy,
                        proposed_z,
                        proposed_path,
                        energy_path_guides,
                        low=low,
                        high=high,
                        prior_mixture_probability=epsilon,
                    )
                    log_q_reverse = _joint_guide_logpdf(
                        energy[index],
                        z_range[index],
                        coefficients[index],
                        energy_path_guides,
                        low=low,
                        high=high,
                        prior_mixture_probability=epsilon,
                    )
                    # The independence proposal is exactly the normalized
                    # defensive base mixture used at beta=0.
                    proposal_correction = log_q_reverse - log_q_forward
                elif use_guide:
                    selected_guide = path_guides[int(rng.integers(len(path_guides)))]
                    guide_rho = float(config.path_guide_rho)
                    proposed_path = selected_guide.bridge_sample(
                        coefficients[index], guide_rho, rng
                    )
                    proposed_energy = float(energy[index])
                    proposed_z = float(z_range[index])
                    proposal_correction = (
                        selected_guide.bridge_logpdf(
                            coefficients[index], proposed_path, guide_rho
                        )
                        - selected_guide.bridge_logpdf(
                            proposed_path, coefficients[index], guide_rho
                        )
                    )
                else:
                    rho = float(config.path_pcn_rho)
                    proposed_path = np.ascontiguousarray(
                        rho * coefficients[index]
                        + math.sqrt(1.0 - rho * rho) * rng.normal(size=dimension)
                    )
                    proposed_energy = float(energy[index])
                    proposed_z = float(z_range[index])
                    # pCN is reversible with respect to the path prior.  The
                    # prior-density correction is filled after evaluating the
                    # complete proposed state below.
                    proposal_correction = math.nan
                proposed_loglike, proposed_range = evaluate(
                    proposed_energy, proposed_z, proposed_path
                )
                proposed_log_prior = state_log_prior(
                    proposed_energy, proposed_z, proposed_path
                )
                proposed_log_base = state_log_base(
                    proposed_energy, proposed_z, proposed_path
                )
                if not use_guide:
                    proposal_correction = (
                        log_prior[index] - proposed_log_prior
                    )
                proposed_target = tempered_log_target(
                    beta_after,
                    proposed_log_base,
                    proposed_log_prior,
                    proposed_loglike,
                )
                current_target = tempered_log_target(
                    beta_after,
                    log_base[index],
                    log_prior[index],
                    loglike[index],
                )
                if math.isfinite(proposed_loglike) and accept_log_ratio(
                    proposed_target - current_target + proposal_correction
                ):
                    energy[index] = proposed_energy
                    z_range[index] = proposed_z
                    coefficients[index] = proposed_path
                    ranges[index] = proposed_range
                    loglike[index] = proposed_loglike
                    log_prior[index] = proposed_log_prior
                    log_base[index] = proposed_log_base
                    bridge_score[index] = (
                        proposed_log_prior + proposed_loglike - proposed_log_base
                    )
                    accepts[label] += 1

            # The final-stage resampling produces equally weighted posterior
            # chains.  Each subsequent beta=1 MCMC sweep preserves that exact
            # posterior.  Retain the already-computed trajectory instead of
            # discarding every state except the last one; this lowers Monte
            # Carlo error without changing the target or adding likelihood
            # evaluations.  The ordinary rejuvenation sweeps are burn-in and
            # are intentionally not retained.
            if (
                beta_after >= 1.0 - 1.0e-12
                and move_index >= int(config.rejuvenation_steps)
            ):
                posterior_energy_samples.append(np.array(energy, copy=True))
                posterior_z_samples.append(np.array(z_range, copy=True))
                posterior_range_samples.append(np.array(ranges, copy=True))
                posterior_path_samples.append(np.array(coefficients, copy=True))
                posterior_loglike_samples.append(np.array(loglike, copy=True))

        def rate(label: str) -> float:
            return (
                float(accepts[label]) / attempts[label]
                if attempts[label] > 0 else math.nan
            )

        stages.append(
            SMCStage(
                beta_before=float(beta),
                beta_after=float(beta_after),
                ess_before_resampling=float(ess),
                resampled=resampled,
                energy_acceptance=rate("energy"),
                range_acceptance=rate("range"),
                path_acceptance=rate("path"),
                guide_acceptance=rate("guide"),
            )
        )
        beta = float(beta_after)

    if beta < 1.0 - 1.0e-10:
        raise RuntimeError(
            "SMC exceeded max_temperatures before reaching the posterior"
        )
    if posterior_energy_samples:
        energy = np.concatenate(posterior_energy_samples, axis=0)
        z_range = np.concatenate(posterior_z_samples, axis=0)
        ranges = np.concatenate(posterior_range_samples, axis=0)
        coefficients = np.concatenate(posterior_path_samples, axis=0)
        loglike = np.concatenate(posterior_loglike_samples, axis=0)
        weights = np.full(energy.size, 1.0 / energy.size, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights /= float(np.sum(weights))
    return JointEnergyRangeSMCResult(
        kinetic_energy_mev=np.ascontiguousarray(energy),
        z_range=np.ascontiguousarray(z_range),
        realized_range_mm=np.ascontiguousarray(ranges),
        coefficients=np.ascontiguousarray(coefficients),
        log_likelihood=np.ascontiguousarray(loglike),
        weights=np.ascontiguousarray(weights),
        stages=tuple(stages),
        likelihood_evaluations=int(evaluations),
        invalid_likelihood_evaluations=int(invalid),
        log_evidence=float(log_evidence),
        random_seed=int(config.random_seed),
        kinetic_energy_bounds_mev=(low, high),
        initialization_beta=0.0,
        initialization_ess=float(initialization_ess),
        posterior_trajectory_sweeps=int(len(posterior_energy_samples)),
    )


__all__ = [
    "EnergyPathGuide",
    "GaussianPathGuide",
    "JointEnergyRangeSMCConfig",
    "JointEnergyRangeSMCResult",
    "SMCStage",
    "run_joint_energy_range_smc",
]
