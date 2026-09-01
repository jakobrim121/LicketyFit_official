"""Reference global-track profiling for the coherent Fermi--Eyges model.

The production continuation historically proposed global-track moves from a
Schur/Fisher linearization but accepted them with a different scalar objective.
That is useful as a fast approximation, but it is not a correctness reference:
the finite-aperture optical active set and the latent Laplace determinant both
change with the global track.

This module deliberately optimizes the *same* scalar at every point,

    min_theta [ min_u {-log p(Q | theta, u) + 0.5 u.T u}
                + 0.5 log det H_F(theta, u_MAP) ],

where ``H_F`` is the positive-definite Poisson Fisher information including the
unit-normal latent prior.  Every eligible global point contains a converged
latent MAP.  The forward field and its latent derivative remain the analytic,
physics-based FALI model; COBYQA is used only to poll the resulting nonlinear
scalar in physical global coordinates.

No event truth, empirical correction, or straight-fit prior appears here.  A
straight fit is only the numerical starting point for ``theta``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import Bounds, minimize

from .mcs_curved_path import MCSPhysicalDomainError
from .mcs_latent_profile import (
    LatentMAPResult,
    _posterior_charge_nll,
    solve_latent_charge_map,
)


DEFAULT_GLOBAL_SCALES = np.asarray(
    [10.0, 10.0, 10.0, 1.0e-2, 1.0e-2, 15.0, 2.0e-2],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ReprofiledEvaluation:
    scaled_coordinates: tuple[float, ...]
    theta: tuple[float, ...]
    laplace_nll: float
    posterior_nll: float
    charge_nll: float
    latent_converged: bool
    latent_gradient_max_abs: float
    latent_newton_decrement: float
    termination_reason: str
    model_valid: bool


@dataclass
class ReprofiledLaplaceResult:
    theta: np.ndarray
    latent: LatentMAPResult
    scaled_coordinates: np.ndarray
    free_indices: tuple[int, ...]
    global_scales: np.ndarray
    optimizer_success: bool
    converged: bool
    message: str
    nfev: int
    nit: int
    evaluations: tuple[ReprofiledEvaluation, ...]
    local_poll_max_downhill: float
    local_poll_radius: float


def _latent_newton_decrement(result: LatentMAPResult) -> float:
    value = getattr(result, "newton_decrement", None)
    if value is not None and math.isfinite(float(value)):
        return float(value)
    gradient = np.asarray(result.final_gradient, dtype=np.float64)
    covariance = np.asarray(result.covariance, dtype=np.float64)
    squared = float(gradient @ covariance @ gradient)
    return math.sqrt(max(squared, 0.0))


class _FullyReprofiledFisherLaplace:
    """Pure scaled-coordinate objective with diagnostic point caching."""

    def __init__(
        self,
        evaluator,
        theta0: Sequence[float],
        *,
        free_indices: Sequence[int],
        global_scales: Sequence[float],
        latent_fd: float | Sequence[float],
        latent_max_iterations: int,
        latent_gradient_tolerance: float,
        latent_trust_max_component: float,
        deterministic_latent_start: bool,
        invalid_penalty: float,
    ):
        self.evaluator = evaluator
        self.theta0 = np.asarray(theta0, dtype=np.float64).reshape(7).copy()
        self.free_indices = tuple(int(i) for i in free_indices)
        if not self.free_indices or len(set(self.free_indices)) != len(self.free_indices):
            raise ValueError("free_indices must contain distinct global coordinates")
        if any(i < 0 or i >= 7 for i in self.free_indices):
            raise ValueError("free_indices are outside the seven-coordinate track")
        self.global_scales = np.asarray(global_scales, dtype=np.float64).reshape(7)
        if np.any(~np.isfinite(self.global_scales)) or np.any(self.global_scales <= 0.0):
            raise ValueError("global_scales must be seven positive finite values")
        self.latent_fd = latent_fd
        self.latent_max_iterations = max(1, int(latent_max_iterations))
        self.latent_gradient_tolerance = float(latent_gradient_tolerance)
        self.latent_trust_max_component = float(latent_trust_max_component)
        self.deterministic_latent_start = bool(deterministic_latent_start)
        self.invalid_penalty = float(invalid_penalty)
        if not math.isfinite(self.invalid_penalty) or self.invalid_penalty <= 0.0:
            raise ValueError("invalid_penalty must be finite and positive")
        self._cache: dict[tuple[float, ...], tuple[float, LatentMAPResult | None]] = {}
        self._evaluations: list[ReprofiledEvaluation] = []

    @property
    def evaluations(self) -> tuple[ReprofiledEvaluation, ...]:
        return tuple(self._evaluations)

    def theta_from_scaled(self, scaled: Sequence[float]) -> np.ndarray:
        x = np.asarray(scaled, dtype=np.float64).reshape(len(self.free_indices))
        theta = self.theta0.copy()
        idx = np.asarray(self.free_indices, dtype=int)
        theta[idx] += x * self.global_scales[idx]
        return np.ascontiguousarray(theta)

    @staticmethod
    def _key(scaled: np.ndarray) -> tuple[float, ...]:
        # COBYQA can revisit a point at roundoff-level differences.  Twelve
        # decimals in standardized global coordinates is far below any stated
        # physical tolerance and makes those evaluations exactly reusable.
        return tuple(np.round(np.asarray(scaled, dtype=np.float64), 12))

    def _nearest_converged_coefficients(self, scaled: np.ndarray):
        if self.deterministic_latent_start:
            return None
        best_distance = float("inf")
        best = None
        for key, (_value, latent) in self._cache.items():
            if latent is None or not latent.converged:
                continue
            distance = float(np.linalg.norm(scaled - np.asarray(key, dtype=np.float64)))
            if distance < best_distance:
                best_distance = distance
                best = latent.coefficients
        return best

    def evaluate(self, scaled: Sequence[float]) -> tuple[float, LatentMAPResult | None]:
        x = np.asarray(scaled, dtype=np.float64).reshape(len(self.free_indices))
        key = self._key(x)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        theta = self.theta_from_scaled(x)
        model = self.evaluator.model(theta)
        if model is None:
            value = self.invalid_penalty * (1.0 + 1.0e-6 * float(x @ x))
            self._cache[key] = (value, None)
            self._evaluations.append(
                ReprofiledEvaluation(
                    tuple(map(float, x)), tuple(map(float, theta)), float(value),
                    float("inf"), float("inf"), False, float("inf"),
                    float("inf"), "invalid_global_track", False,
                )
            )
            return value, None
        initial = self._nearest_converged_coefficients(x)
        try:
            latent = solve_latent_charge_map(
                model,
                initial_coefficients=initial,
                fd_step=self.latent_fd,
                max_iterations=self.latent_max_iterations,
                gradient_tolerance=self.latent_gradient_tolerance,
                trust_max_component=self.latent_trust_max_component,
            )
        except MCSPhysicalDomainError:
            value = self.invalid_penalty * (1.0 + 1.0e-6 * float(x @ x))
            latent = None
        if latent is None or not latent.converged:
            # An under-profiled point is not the marginal objective.  It is
            # deliberately ineligible for global acceptance.
            residual = (
                1.0 if latent is None
                else 1.0 + float(latent.final_gradient_max_abs)
            )
            value = self.invalid_penalty * residual
        else:
            value = float(latent.laplace_nll)
        result = (float(value), latent)
        self._cache[key] = result
        self._evaluations.append(
            ReprofiledEvaluation(
                scaled_coordinates=tuple(map(float, x)),
                theta=tuple(map(float, theta)),
                laplace_nll=float(value),
                posterior_nll=(float("inf") if latent is None else float(latent.posterior_nll)),
                charge_nll=(float("inf") if latent is None else float(latent.charge_nll)),
                latent_converged=bool(latent is not None and latent.converged),
                latent_gradient_max_abs=(
                    float("inf") if latent is None else float(latent.final_gradient_max_abs)
                ),
                latent_newton_decrement=(
                    float("inf") if latent is None else _latent_newton_decrement(latent)
                ),
                termination_reason=(
                    "physical_domain_rejection" if latent is None
                    else str(latent.termination_reason)
                ),
                model_valid=True,
            )
        )
        return result

    def __call__(self, scaled: Sequence[float]) -> float:
        return self.evaluate(scaled)[0]


def _scaled_bounds(evaluator, theta0, free_indices, scales):
    lower = np.full(len(free_indices), -np.inf, dtype=np.float64)
    upper = np.full(len(free_indices), np.inf, dtype=np.float64)
    if getattr(evaluator, "length_limits", None) is not None:
        lo, hi = map(float, evaluator.length_limits)
        for column, index in enumerate(free_indices):
            if index == 5:
                lower[column] = (lo - theta0[5]) / scales[5]
                upper[column] = (hi - theta0[5]) / scales[5]
    if getattr(evaluator, "t0_limits", None) is not None:
        lo, hi = map(float, evaluator.t0_limits)
        for column, index in enumerate(free_indices):
            if index == 6:
                lower[column] = (lo - theta0[6]) / scales[6]
                upper[column] = (hi - theta0[6]) / scales[6]
    return Bounds(lower, upper)


def optimize_reprofiled_fisher_laplace(
    evaluator,
    theta,
    *,
    free_indices: Sequence[int] = (0, 1, 2, 3, 4, 5),
    global_scales: Sequence[float] = DEFAULT_GLOBAL_SCALES,
    latent_fd: float | Sequence[float] = 0.20,
    latent_max_iterations: int = 60,
    latent_gradient_tolerance: float = 1.0e-3,
    latent_trust_max_component: float = 1.0,
    max_global_evaluations: int = 300,
    initial_trust_radius: float = 1.0,
    final_trust_radius: float = 2.0e-2,
    deterministic_latent_start: bool = True,
    local_poll_tolerance: float = 1.0e-4,
) -> ReprofiledLaplaceResult:
    """Optimize a fully reprofiled Fisher--Laplace scalar with COBYQA.

    Global variables are standardized by physical scales, but those scales are
    neither priors nor penalties.  The only finite bound added here is the
    emitter's physical visible-length/time domain; detector containment remains
    enforced by ``CoupledCoherentEvaluator``.
    """
    theta0 = np.asarray(theta, dtype=np.float64).reshape(7)
    free = tuple(int(i) for i in free_indices)
    scales = np.asarray(global_scales, dtype=np.float64).reshape(7)
    if 6 in free and bool(getattr(evaluator, "charge_only", False)):
        raise ValueError("t0 is unidentifiable in a charge-only coherent objective")
    objective = _FullyReprofiledFisherLaplace(
        evaluator,
        theta0,
        free_indices=free,
        global_scales=scales,
        latent_fd=latent_fd,
        latent_max_iterations=latent_max_iterations,
        latent_gradient_tolerance=latent_gradient_tolerance,
        latent_trust_max_component=latent_trust_max_component,
        deterministic_latent_start=deterministic_latent_start,
        invalid_penalty=1.0e20,
    )
    x0 = np.zeros(len(free), dtype=np.float64)
    base_value, base_latent = objective.evaluate(x0)
    if base_latent is None or not base_latent.converged or not math.isfinite(base_value):
        gradient_check = ""
        if base_latent is not None:
            model = evaluator.model(theta0)
            if model is not None:
                u = np.asarray(base_latent.coefficients, dtype=np.float64)
                analytic = np.asarray(base_latent.final_gradient, dtype=np.float64)
                checks = []
                for h in (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4):
                    numeric = np.empty_like(u)
                    for k in range(u.size):
                        plus = u.copy(); minus = u.copy()
                        plus[k] += h; minus[k] -= h
                        fp = float(model.charge_data_nll(plus) + 0.5 * (plus @ plus))
                        fm = float(model.charge_data_nll(minus) + 0.5 * (minus @ minus))
                        numeric[k] = (fp - fm) / (2.0 * h)
                    denominator = max(float(np.linalg.norm(numeric)), 1.0e-300)
                    cosine_denominator = max(
                        float(np.linalg.norm(numeric) * np.linalg.norm(analytic)),
                        1.0e-300,
                    )
                    checks.append(
                        f"h={h:g}:rel={np.linalg.norm(analytic - numeric) / denominator:.5g},"
                        f"cos={float(analytic @ numeric) / cosine_denominator:.7g},"
                        f"max={float(np.max(np.abs(numeric))):.5g}"
                    )
                gradient_check = ", exact-gradient checks=[" + ";".join(checks) + "]"
                if bool(getattr(model, "curved_delta_enabled", False)):
                    saved_enabled = model.curved_delta_enabled
                    saved_charge_cache = dict(model.charge_cache)
                    saved_delta_field_cache = dict(model.delta_field_cache)
                    try:
                        model.curved_delta_enabled = False
                        model.charge_cache.clear()
                        model.delta_field_cache.clear()
                        model.delta_field_cache[
                            tuple(np.zeros(model.n_modes, dtype=np.float64))
                        ] = model.curved_zero_delta
                        direct_mu, direct_jac = model.charge_prediction_and_jacobian(u)
                        direct_gradient = direct_jac.T @ (
                            1.0 - np.asarray(model.obs_pes, dtype=np.float64)
                            / np.maximum(direct_mu, 1.0e-12)
                        ) + u
                        direct_checks = []
                        for h in (1.0e-3, 3.0e-4, 1.0e-4):
                            numeric = np.empty_like(u)
                            for k in range(u.size):
                                plus = u.copy(); minus = u.copy()
                                plus[k] += h; minus[k] -= h
                                fp = float(
                                    model.charge_data_nll(plus)
                                    + 0.5 * (plus @ plus)
                                )
                                fm = float(
                                    model.charge_data_nll(minus)
                                    + 0.5 * (minus @ minus)
                                )
                                numeric[k] = (fp - fm) / (2.0 * h)
                            denominator = max(float(np.linalg.norm(numeric)), 1.0e-300)
                            cosine_denominator = max(
                                float(
                                    np.linalg.norm(numeric)
                                    * np.linalg.norm(direct_gradient)
                                ),
                                1.0e-300,
                            )
                            direct_checks.append(
                                f"h={h:g}:rel="
                                f"{np.linalg.norm(direct_gradient - numeric) / denominator:.5g},"
                                f"cos={float(direct_gradient @ numeric) / cosine_denominator:.7g},"
                                f"max={float(np.max(np.abs(numeric))):.5g}"
                            )
                        gradient_check += (
                            ", direct-only checks=[" + ";".join(direct_checks) + "]"
                        )
                    finally:
                        model.curved_delta_enabled = saved_enabled
                        model.charge_cache.clear()
                        model.charge_cache.update(saved_charge_cache)
                        model.delta_field_cache.clear()
                        model.delta_field_cache.update(saved_delta_field_cache)
        diagnostic = (
            "no latent result" if base_latent is None
            else (
                f"status={base_latent.termination_reason}, "
                f"max|g|={base_latent.final_gradient_max_abs:.6g}, "
                f"newton_decrement={_latent_newton_decrement(base_latent):.6g}, "
                f"iterations={len(base_latent.iterations)}{gradient_check}, "
                f"coefficients={np.asarray(base_latent.coefficients).tolist()}"
            )
        )
        raise RuntimeError(
            "initial global track does not have a converged latent MAP; "
            f"the marginal objective is undefined ({diagnostic})"
        )
    result = minimize(
        objective,
        x0,
        method="COBYQA",
        bounds=_scaled_bounds(evaluator, theta0, free, scales),
        options={
            "maxfev": max(1, int(max_global_evaluations)),
            "initial_tr_radius": float(initial_trust_radius),
            "final_tr_radius": float(final_trust_radius),
            "scale": False,
            "disp": False,
        },
    )
    final_x = np.asarray(result.x, dtype=np.float64)
    final_value, final_latent = objective.evaluate(final_x)
    if final_latent is None:
        raise RuntimeError("COBYQA returned a physical-domain-rejected track")

    # Explicit coordinate poll on the same fully reprofiled objective.  This is
    # an operational stationarity check, independent of COBYQA's status string.
    radius = float(final_trust_radius)
    best_downhill = 0.0
    if radius > 0.0:
        for column in range(len(free)):
            for sign in (-1.0, 1.0):
                trial = final_x.copy()
                trial[column] += sign * radius
                trial_value, _ = objective.evaluate(trial)
                best_downhill = max(best_downhill, float(final_value - trial_value))
    converged = bool(
        result.success
        and final_latent.converged
        and best_downhill <= float(local_poll_tolerance)
    )
    return ReprofiledLaplaceResult(
        theta=objective.theta_from_scaled(final_x),
        latent=final_latent,
        scaled_coordinates=np.ascontiguousarray(final_x),
        free_indices=free,
        global_scales=np.ascontiguousarray(scales),
        optimizer_success=bool(result.success),
        converged=converged,
        message=str(result.message),
        nfev=int(getattr(result, "nfev", len(objective.evaluations))),
        nit=int(getattr(result, "nit", -1)),
        evaluations=objective.evaluations,
        local_poll_max_downhill=float(best_downhill),
        local_poll_radius=radius,
    )


class _ExactPosteriorScalar:
    """Cached exact latent posterior used by both polls and COBYQA.

    Keeping the scalar and coordinate-poll implementation in one place is
    important here: a fast-path certificate must test precisely the same
    finite-aperture likelihood that the fallback optimizer sees.
    """

    def __init__(self, model, n_modes: int):
        self.model = model
        self.n_modes = int(n_modes)
        self.cache = {}
        self.history = []

    def __call__(self, coefficients):
        x = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(x, 12))
        if key in self.cache:
            return self.cache[key]
        try:
            charge, posterior = _posterior_charge_nll(self.model, x)
        except MCSPhysicalDomainError:
            charge, posterior = float("inf"), float("inf")
        if not math.isfinite(posterior):
            value = 1.0e20 * (1.0 + 1.0e-9 * float(x @ x))
        else:
            value = float(posterior)
        self.cache[key] = value
        self.history.append((x.copy(), float(charge), value))
        return value

    def coordinate_poll(self, center, poll_radii):
        centre = np.asarray(center, dtype=np.float64).reshape(self.n_modes)
        value = float(self(centre))
        best_value = value
        best_point = centre.copy()
        rows = []
        maximum_downhill = 0.0
        for radius in poll_radii:
            r = float(radius)
            radius_downhill = 0.0
            for k in range(self.n_modes):
                for sign in (-1.0, 1.0):
                    trial = centre.copy()
                    trial[k] += sign * r
                    trial_value = float(self(trial))
                    downhill = value - trial_value
                    radius_downhill = max(radius_downhill, downhill)
                    if trial_value < best_value:
                        best_value = trial_value
                        best_point = trial.copy()
            rows.append({"radius": r, "max_downhill": float(radius_downhill)})
            maximum_downhill = max(maximum_downhill, radius_downhill)
        return (
            tuple(rows),
            float(maximum_downhill),
            np.ascontiguousarray(best_point),
            float(best_value),
        )

    def charge_at(self, coefficients):
        x = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(x, 12))
        self(x)
        for point, charge, _posterior in reversed(self.history):
            if tuple(np.round(point, 12)) == key:
                return float(charge)
        raise RuntimeError("exact posterior cache lost its centre evaluation")


def exact_posterior_latent_coordinate_poll(
    model,
    coefficients,
    *,
    poll_radii: Sequence[float] = (1.0e-2, 3.0e-3),
    poll_tolerance: float = 1.0e-4,
):
    """Certify stationarity in the original full-rank FE coordinates.

    This is an exact scalar poll, not a gradient approximation.  It evaluates
    both signs of every retained Fermi--Eyges coefficient at every requested
    radius and therefore remains valid across finite-aperture active-set
    changes.  The returned mapping is deliberately compatible with the subset
    of the exact-optimizer result consumed by the continuation controller.
    """
    n_modes = int(getattr(model, "n_modes", 0))
    if n_modes <= 0:
        raise ValueError("model must expose a positive latent dimension")
    centre = np.asarray(coefficients, dtype=np.float64).reshape(n_modes)
    scalar = _ExactPosteriorScalar(model, n_modes)
    poll, max_downhill, best_point, best_value = scalar.coordinate_poll(
        centre, poll_radii
    )
    value = float(scalar(centre))
    centre_charge = scalar.charge_at(centre)
    centre_physical = bool(
        math.isfinite(centre_charge)
        and math.isfinite(value)
        and value < 1.0e20
    )
    return {
        "coefficients": np.ascontiguousarray(centre),
        "charge_nll": centre_charge,
        "posterior_nll": value,
        "converged": bool(
            centre_physical and max_downhill <= float(poll_tolerance)
        ),
        "poll": tuple(poll),
        "max_poll_downhill": float(max_downhill),
        "best_polled_coefficients": np.ascontiguousarray(best_point),
        "best_polled_posterior_nll": float(best_value),
        "evaluations": int(len(scalar.cache)),
        "history": tuple(scalar.history),
    }


def optimize_exact_posterior_latent_cobyqa(
    model,
    *,
    initial_coefficients=None,
    max_evaluations: int = 600,
    initial_trust_radius: float = 0.5,
    final_trust_radius: float = 3.0e-3,
    poll_radii: Sequence[float] = (1.0e-2, 3.0e-3),
    poll_tolerance: float = 1.0e-4,
    restart_evaluations: int | None = None,
    max_restarts: int = 5,
    accept_certified_initial: bool = True,
):
    """Derivative-free reference MAP for a nonsmooth but analytic field.

    The FALI forward integral is piecewise smooth because finite receiver
    support can change topology.  This reference never substitutes a noisy
    local derivative for that exact scalar.  No coefficient bounds are used;
    only the physical FE tangent domain may reject a point.
    """
    n_modes = int(getattr(model, "n_modes", 0))
    if n_modes <= 0:
        raise ValueError("model must expose a positive latent dimension")
    restart_limit = int(max_restarts)
    if restart_limit < 0:
        raise ValueError("max_restarts must be non-negative")
    x0 = (
        np.zeros(n_modes, dtype=np.float64)
        if initial_coefficients is None
        else np.asarray(initial_coefficients, dtype=np.float64).reshape(n_modes)
    )
    scalar = _ExactPosteriorScalar(model, n_modes)
    objective = scalar
    history = scalar.history
    cache = scalar.cache

    def run(start, budget, initial_radius, final_radius):
        return minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="COBYQA",
            options={
                "maxfev": max(1, int(budget)),
                "initial_tr_radius": float(initial_radius),
                "final_tr_radius": float(final_radius),
                "scale": False,
                "disp": False,
            },
        )

    def exact_coordinate_poll(center):
        return scalar.coordinate_poll(center, poll_radii)

    # The exact coordinate poll is already the authoritative stationarity
    # certificate at the end of this routine.  Apply that identical test to a
    # physics-informed initializer before constructing COBYQA's 24-dimensional
    # interpolation set.  When it passes, optimization cannot strengthen the
    # stated certificate and would add at least O(n^2) redundant field calls.
    poll_history = []
    if bool(accept_certified_initial):
        initial_poll, initial_downhill, _initial_best_u, _initial_best_value = (
            exact_coordinate_poll(x0)
        )
        poll_history.append(
            {
                "restart": -1,
                "centre": np.asarray(x0, dtype=float).tolist(),
                "poll": initial_poll,
                "max_downhill": float(initial_downhill),
                "phase": "initial_certificate",
            }
        )
        value = float(objective(x0))
        initial_physical = bool(
            math.isfinite(value)
            and value < 1.0e20
            and math.isfinite(scalar.charge_at(x0))
        )
        if initial_physical and initial_downhill <= float(poll_tolerance):
            return {
                "coefficients": np.ascontiguousarray(x0),
                "charge_nll": float(scalar.charge_at(x0)),
                "posterior_nll": value,
                "converged": True,
                "optimizer_success": True,
                "message": "initial exact coordinate certificate passed",
                "nfev": int(len(cache)),
                "nit": 0,
                "poll": tuple(initial_poll),
                "max_poll_downhill": float(initial_downhill),
                "evaluations": len(cache),
                "history": tuple(history),
                "restart_count": 0,
                "max_restarts": int(restart_limit),
                "optimizer_attempts": (),
                "poll_history": tuple(poll_history),
                "initial_certificate_accepted": True,
            }

    result = run(x0, max_evaluations, initial_trust_radius, final_trust_radius)
    attempts = [result]
    # COBYQA's returned status is useful operational information, but the best
    # exact point can occur before its final interpolation centre.  Always use
    # the best finite physical posterior evaluated so far.
    u = np.asarray(min(history, key=lambda row: row[2])[0], dtype=np.float64)

    # A 24-D interpolation set can consume its budget before resolving a local
    # kink.  Exact coordinate polls both diagnose this and provide a
    # deterministic physical restart point.  No noisy derivative or random
    # perturbation enters the recovery.
    retry_budget = (
        max(200, int(max_evaluations))
        if restart_evaluations is None else max(0, int(restart_evaluations))
    )

    poll, max_downhill, best_polled_u, _best_polled_value = exact_coordinate_poll(u)
    poll_history.append(
        {
            "restart": 0,
            "centre": np.asarray(u, dtype=float).tolist(),
            "poll": poll,
            "max_downhill": float(max_downhill),
        }
    )
    restart_count = 0
    while (
        max_downhill > float(poll_tolerance)
        and restart_count < restart_limit
        and retry_budget > 0
    ):
        # Move first to the best exact polled coordinate, then rebuild a local
        # quadratic model there.  Restarts use deterministic decreasing radii
        # in standardized FE units; ``max_restarts`` is a work limit, never a
        # substitute for the final exact coordinate-poll certificate.
        u = best_polled_u
        local_initial_radius = max(
            10.0 * float(final_trust_radius),
            5.0e-2 / (2.0 ** restart_count),
        )
        retry = run(
            u,
            retry_budget,
            local_initial_radius,
            float(final_trust_radius),
        )
        attempts.append(retry)
        restart_count += 1
        result = retry
        u = np.asarray(min(history, key=lambda row: row[2])[0], dtype=np.float64)
        poll, max_downhill, best_polled_u, _best_polled_value = (
            exact_coordinate_poll(u)
        )
        poll_history.append(
            {
                "restart": int(restart_count),
                "centre": np.asarray(u, dtype=float).tolist(),
                "poll": poll,
                "max_downhill": float(max_downhill),
            }
        )

    value = float(objective(u))
    # This final exact poll is the convergence certificate.  COBYQA success is
    # deliberately not part of the condition: a budget status cannot override
    # either a stationary exact scalar or a demonstrably downhill coordinate.
    final_physical = bool(
        math.isfinite(value)
        and value < 1.0e20
        and math.isfinite(scalar.charge_at(u))
    )
    converged = bool(
        final_physical and max_downhill <= float(poll_tolerance)
    )
    return {
        "coefficients": np.ascontiguousarray(u),
        "charge_nll": float(scalar.charge_at(u)),
        "posterior_nll": value,
        "converged": converged,
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "nfev": int(len(cache)),
        "nit": int(sum(max(int(getattr(row, "nit", 0)), 0) for row in attempts)),
        "poll": tuple(poll),
        "max_poll_downhill": float(max_downhill),
        "evaluations": len(cache),
        "history": tuple(history),
        "restart_count": int(restart_count),
        "max_restarts": int(restart_limit),
        "optimizer_attempts": tuple(
            {
                "success": bool(row.success),
                "message": str(row.message),
                "nfev": int(getattr(row, "nfev", 0)),
                "nit": int(getattr(row, "nit", -1)),
            }
            for row in attempts
        ),
        "poll_history": tuple(poll_history),
        "initial_certificate_accepted": False,
    }


def exact_global_geometry_coordinate_poll(
    evaluator,
    theta,
    coefficients,
    *,
    free_indices: Sequence[int] = (0, 1, 2, 3, 4, 5),
    global_scales: Sequence[float] = DEFAULT_GLOBAL_SCALES,
    poll_radii: Sequence[float] = (2.0e-2, 5.0e-3),
):
    """Poll the exact fixed-latent posterior in physical track coordinates."""
    base_theta = np.asarray(theta, dtype=np.float64).reshape(7)
    u = np.asarray(coefficients, dtype=np.float64).reshape(
        int(getattr(evaluator, "n_modes", np.asarray(coefficients).size))
    )
    free = tuple(int(index) for index in free_indices)
    if not free or len(set(free)) != len(free):
        raise ValueError("free_indices must contain distinct global coordinates")
    if any(index < 0 or index >= 7 for index in free):
        raise ValueError("free_indices are outside the seven-coordinate track")
    if 6 in free:
        raise ValueError("t0 is fixed in the charge-only joint exact stage")
    scales = np.asarray(global_scales, dtype=np.float64).reshape(7)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("global_scales must be seven positive finite values")
    radii = tuple(float(radius) for radius in poll_radii)
    if not radii or any(not math.isfinite(radius) or radius <= 0.0 for radius in radii):
        raise ValueError("poll_radii must contain positive finite values")
    base_value = float(evaluator(base_theta, u, include_prior=True))
    if not math.isfinite(base_value):
        raise ValueError("global coordinate poll requires a physical initial pair")

    best_value = base_value
    best_theta = base_theta.copy()
    maximum_downhill = 0.0
    rows = []
    for radius in radii:
        radius_downhill = 0.0
        physical_trials = 0
        for index in free:
            for sign in (-1.0, 1.0):
                trial = base_theta.copy()
                trial[index] += sign * radius * scales[index]
                value = float(evaluator(trial, u, include_prior=True))
                if not math.isfinite(value):
                    continue
                physical_trials += 1
                downhill = base_value - value
                radius_downhill = max(radius_downhill, downhill)
                if value < best_value:
                    best_value = value
                    best_theta = trial.copy()
        maximum_downhill = max(maximum_downhill, radius_downhill)
        rows.append(
            {
                "radius_scaled": float(radius),
                "max_downhill": float(radius_downhill),
                "physical_trials": int(physical_trials),
            }
        )
    return {
        "base_value": float(base_value),
        "best_value": float(best_value),
        "best_theta": np.ascontiguousarray(best_theta),
        "max_downhill": float(maximum_downhill),
        "poll": tuple(rows),
    }


def optimize_exact_global_geometry_cobyqa(
    evaluator,
    theta,
    coefficients,
    *,
    free_indices: Sequence[int] = (0, 1, 2, 3, 4, 5),
    global_scales: Sequence[float] = DEFAULT_GLOBAL_SCALES,
    max_evaluations: int = 300,
    initial_trust_radius: float = 1.0,
    final_trust_radius: float = 5.0e-3,
    poll_radii: Sequence[float] = (2.0e-2, 5.0e-3),
    poll_tolerance: float = 1.0e-4,
    restart_evaluations: int | None = None,
    max_restarts: int = 3,
):
    """Exact derivative-free global geometry MAP block at fixed FE coefficients.

    The standardized coordinates are numerical conditioning only.  Physical
    containment, the coherent unit-tangent domain, and configured visible-range
    limits are the only restrictions.  Invalid points receive an optimizer
    barrier but can never be selected as a result.
    """
    theta0 = np.asarray(theta, dtype=np.float64).reshape(7).copy()
    u = np.asarray(coefficients, dtype=np.float64).reshape(
        int(getattr(evaluator, "n_modes", np.asarray(coefficients).size))
    )
    free = tuple(int(index) for index in free_indices)
    if not free or len(set(free)) != len(free):
        raise ValueError("free_indices must contain distinct global coordinates")
    if any(index < 0 or index >= 7 for index in free):
        raise ValueError("free_indices are outside the seven-coordinate track")
    if 6 in free:
        raise ValueError("t0 is unidentifiable and fixed in charge-only joint MAP")
    if not bool(getattr(evaluator, "charge_only", False)):
        raise ValueError("joint exact geometry optimization requires charge_only=True")
    scales = np.asarray(global_scales, dtype=np.float64).reshape(7)
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("global_scales must be seven positive finite values")
    restart_limit = int(max_restarts)
    if restart_limit < 0:
        raise ValueError("max_restarts must be non-negative")

    initial_value = float(evaluator(theta0, u, include_prior=True))
    if not math.isfinite(initial_value):
        raise ValueError("initial joint geometry/latent pair is outside the physical domain")
    invalid_penalty = max(1.0e6, abs(initial_value) + 1.0e6)
    cache: dict[tuple[float, ...], tuple[float, float | None, np.ndarray]] = {}
    history = []

    def theta_from_scaled(scaled):
        x = np.asarray(scaled, dtype=np.float64).reshape(len(free))
        out = theta0.copy()
        indices = np.asarray(free, dtype=int)
        out[indices] += x * scales[indices]
        return np.ascontiguousarray(out)

    def objective(scaled):
        x = np.asarray(scaled, dtype=np.float64).reshape(len(free))
        key = tuple(np.round(x, 12))
        cached = cache.get(key)
        if cached is not None:
            return cached[0]
        candidate = theta_from_scaled(x)
        exact = float(evaluator(candidate, u, include_prior=True))
        if math.isfinite(exact):
            value = exact
            physical = True
        else:
            value = invalid_penalty * (1.0 + 1.0e-6 * float(x @ x))
            exact = None
            physical = False
        cache[key] = (float(value), exact, candidate)
        history.append(
            {
                "scaled_coordinates": np.ascontiguousarray(x.copy()),
                "theta": candidate,
                "value": float(value),
                "exact_posterior_nll": exact,
                "physical": bool(physical),
            }
        )
        return float(value)

    bounds = _scaled_bounds(evaluator, theta0, free, scales)

    def run(start, budget, initial_radius):
        return minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="COBYQA",
            bounds=bounds,
            options={
                "maxfev": max(1, int(budget)),
                "initial_tr_radius": float(initial_radius),
                "final_tr_radius": float(final_trust_radius),
                "scale": False,
                "disp": False,
            },
        )

    def best_physical_row():
        physical = [row for row in history if row["physical"]]
        if not physical:
            raise RuntimeError("global optimizer lost every physical exact point")
        return min(physical, key=lambda row: float(row["exact_posterior_nll"]))

    result = run(np.zeros(len(free)), max_evaluations, initial_trust_radius)
    attempts = [result]
    best = best_physical_row()
    best_theta = np.asarray(best["theta"], dtype=np.float64)
    best_scaled = np.asarray(best["scaled_coordinates"], dtype=np.float64)
    poll = exact_global_geometry_coordinate_poll(
        evaluator,
        best_theta,
        u,
        free_indices=free,
        global_scales=scales,
        poll_radii=poll_radii,
    )
    if float(poll["best_value"]) < float(best["exact_posterior_nll"]):
        indices = np.asarray(free, dtype=int)
        polled_theta = np.asarray(poll["best_theta"], dtype=np.float64)
        polled_scaled = (
            polled_theta[indices] - theta0[indices]
        ) / scales[indices]
        objective(polled_scaled)
        best = best_physical_row()
        best_theta = np.asarray(best["theta"], dtype=np.float64)
        best_scaled = np.asarray(best["scaled_coordinates"], dtype=np.float64)
    poll_history = [
        {
            "restart": 0,
            "theta": best_theta.copy(),
            "poll": poll["poll"],
            "max_downhill": float(poll["max_downhill"]),
        }
    ]
    retry_budget = (
        max(100, int(max_evaluations))
        if restart_evaluations is None
        else max(0, int(restart_evaluations))
    )
    restart_count = 0
    while (
        float(poll["max_downhill"]) > float(poll_tolerance)
        and restart_count < restart_limit
        and retry_budget > 0
    ):
        # Express the exact best poll in this block's fixed standardized chart.
        polled_theta = np.asarray(poll["best_theta"], dtype=np.float64)
        indices = np.asarray(free, dtype=int)
        start = (polled_theta[indices] - theta0[indices]) / scales[indices]
        local_radius = max(
            10.0 * float(final_trust_radius),
            5.0e-2 / (2.0 ** restart_count),
        )
        result = run(start, retry_budget, local_radius)
        attempts.append(result)
        restart_count += 1
        best = best_physical_row()
        best_theta = np.asarray(best["theta"], dtype=np.float64)
        best_scaled = np.asarray(best["scaled_coordinates"], dtype=np.float64)
        poll = exact_global_geometry_coordinate_poll(
            evaluator,
            best_theta,
            u,
            free_indices=free,
            global_scales=scales,
            poll_radii=poll_radii,
        )
        if float(poll["best_value"]) < float(best["exact_posterior_nll"]):
            indices = np.asarray(free, dtype=int)
            polled_theta = np.asarray(poll["best_theta"], dtype=np.float64)
            polled_scaled = (
                polled_theta[indices] - theta0[indices]
            ) / scales[indices]
            objective(polled_scaled)
            best = best_physical_row()
            best_theta = np.asarray(best["theta"], dtype=np.float64)
            best_scaled = np.asarray(best["scaled_coordinates"], dtype=np.float64)
        poll_history.append(
            {
                "restart": int(restart_count),
                "theta": best_theta.copy(),
                "poll": poll["poll"],
                "max_downhill": float(poll["max_downhill"]),
            }
        )

    best = best_physical_row()
    return {
        "theta": np.ascontiguousarray(best["theta"]),
        "scaled_coordinates": np.ascontiguousarray(best["scaled_coordinates"]),
        "posterior_nll": float(best["exact_posterior_nll"]),
        "converged": bool(float(poll["max_downhill"]) <= float(poll_tolerance)),
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "nfev": int(len(cache)),
        "nit": int(sum(max(int(getattr(row, "nit", 0)), 0) for row in attempts)),
        "restart_count": int(restart_count),
        "max_restarts": int(restart_limit),
        "max_poll_downhill": float(poll["max_downhill"]),
        "poll": tuple(poll["poll"]),
        "poll_history": tuple(poll_history),
        "history": tuple(history),
        "optimizer_attempts": tuple(
            {
                "success": bool(row.success),
                "message": str(row.message),
                "nfev": int(getattr(row, "nfev", 0)),
                "nit": int(getattr(row, "nit", -1)),
            }
            for row in attempts
        ),
    }


def optimize_alternating_exact_joint_map(
    evaluator,
    theta,
    *,
    initial_latent_result=None,
    initial_coefficients=None,
    free_indices: Sequence[int] = (0, 1, 2, 3, 4, 5),
    global_scales: Sequence[float] = DEFAULT_GLOBAL_SCALES,
    max_cycles: int = 3,
    cycle_tolerance: float = 1.0e-4,
    max_global_evaluations: int = 300,
    initial_global_trust_radius: float = 1.0,
    final_global_trust_radius: float = 5.0e-3,
    global_poll_radii: Sequence[float] = (2.0e-2, 5.0e-3),
    global_poll_tolerance: float = 1.0e-4,
    global_restart_evaluations: int | None = None,
    global_max_restarts: int = 3,
    latent_max_evaluations: int = 600,
    latent_initial_trust_radius: float = 0.5,
    latent_final_trust_radius: float = 3.0e-3,
    latent_poll_radii: Sequence[float] = (1.0e-2, 3.0e-3),
    latent_poll_tolerance: float = 1.0e-4,
    latent_restart_evaluations: int | None = None,
    latent_max_restarts: int = 5,
):
    """Alternate exact fixed-block MAP solves on one joint charge posterior.

    The returned convergence statement is coordinate-block stationarity only;
    it makes no Laplace, covariance, or global-optimum claim.
    """
    if not bool(getattr(evaluator, "charge_only", False)):
        raise ValueError("alternating joint exact MAP requires charge_only=True")
    cycles_limit = int(max_cycles)
    if cycles_limit <= 0:
        raise ValueError("max_cycles must be positive")
    current_theta = np.asarray(theta, dtype=np.float64).reshape(7).copy()
    fixed_t0 = float(current_theta[6])
    free = tuple(int(index) for index in free_indices)
    if 6 in free:
        raise ValueError("t0 must remain fixed in the charge-only joint stage")

    if initial_latent_result is None:
        model = evaluator.model(current_theta)
        if model is None:
            raise ValueError("initial global geometry is outside the physical domain")
        latent = optimize_exact_posterior_latent_cobyqa(
            model,
            initial_coefficients=initial_coefficients,
            max_evaluations=int(latent_max_evaluations),
            initial_trust_radius=float(latent_initial_trust_radius),
            final_trust_radius=float(latent_final_trust_radius),
            poll_radii=latent_poll_radii,
            poll_tolerance=float(latent_poll_tolerance),
            restart_evaluations=latent_restart_evaluations,
            max_restarts=int(latent_max_restarts),
        )
    else:
        latent = initial_latent_result
    if not bool(latent["converged"]):
        raise RuntimeError("initial exact latent block is not coordinate-poll stationary")
    current_u = np.asarray(latent["coefficients"], dtype=np.float64).reshape(
        int(evaluator.n_modes)
    )
    current_value = float(evaluator(current_theta, current_u, include_prior=True))
    if not math.isfinite(current_value):
        raise RuntimeError("initial exact joint posterior is not physical")

    cycle_rows = []
    converged = False
    final_geometry_poll = None
    final_latent = latent
    for cycle_index in range(cycles_limit):
        before = current_value
        geometry = optimize_exact_global_geometry_cobyqa(
            evaluator,
            current_theta,
            current_u,
            free_indices=free,
            global_scales=global_scales,
            max_evaluations=int(max_global_evaluations),
            initial_trust_radius=float(initial_global_trust_radius),
            final_trust_radius=float(final_global_trust_radius),
            poll_radii=global_poll_radii,
            poll_tolerance=float(global_poll_tolerance),
            restart_evaluations=global_restart_evaluations,
            max_restarts=int(global_max_restarts),
        )
        candidate_theta = np.asarray(geometry["theta"], dtype=np.float64)
        if float(candidate_theta[6]) != fixed_t0:
            raise RuntimeError("charge-only global block changed fixed t0")
        after_geometry = float(
            evaluator(candidate_theta, current_u, include_prior=True)
        )
        if after_geometry > before + 1.0e-8:
            raise RuntimeError("exact global geometry block increased the joint posterior")

        model = evaluator.model(candidate_theta)
        if model is None:
            raise RuntimeError("exact global block returned an invalid physical model")
        latent = optimize_exact_posterior_latent_cobyqa(
            model,
            initial_coefficients=current_u,
            max_evaluations=int(latent_max_evaluations),
            initial_trust_radius=float(latent_initial_trust_radius),
            final_trust_radius=float(latent_final_trust_radius),
            poll_radii=latent_poll_radii,
            poll_tolerance=float(latent_poll_tolerance),
            restart_evaluations=latent_restart_evaluations,
            max_restarts=int(latent_max_restarts),
        )
        candidate_u = np.asarray(latent["coefficients"], dtype=np.float64)
        after_latent = float(
            evaluator(candidate_theta, candidate_u, include_prior=True)
        )
        if after_latent > after_geometry + 1.0e-8:
            raise RuntimeError("exact latent block increased the joint posterior")

        # The geometry block was stationary for the previous u.  Poll it again
        # with the newly profiled path before making a joint block-stationarity
        # claim.
        final_geometry_poll = exact_global_geometry_coordinate_poll(
            evaluator,
            candidate_theta,
            candidate_u,
            free_indices=free,
            global_scales=global_scales,
            poll_radii=global_poll_radii,
        )
        improvement = before - after_latent
        cycle_rows.append(
            {
                "cycle": int(cycle_index),
                "posterior_before": float(before),
                "posterior_after_geometry": float(after_geometry),
                "posterior_after_latent": float(after_latent),
                "improvement": float(improvement),
                "theta": candidate_theta.copy(),
                "coefficients": candidate_u.copy(),
                "geometry": geometry,
                "latent": latent,
                "post_latent_geometry_poll": final_geometry_poll,
            }
        )
        current_theta = candidate_theta
        current_u = candidate_u
        current_value = after_latent
        final_latent = latent
        converged = bool(
            geometry["converged"]
            and latent["converged"]
            and float(final_geometry_poll["max_downhill"])
            <= float(global_poll_tolerance)
            and improvement <= float(cycle_tolerance)
        )
        if converged:
            break

    return {
        "theta": np.ascontiguousarray(current_theta),
        "coefficients": np.ascontiguousarray(current_u),
        "posterior_nll": float(current_value),
        "charge_nll": float(final_latent["charge_nll"]),
        "converged": bool(converged),
        "cycles": tuple(cycle_rows),
        "cycle_count": int(len(cycle_rows)),
        "latent": final_latent,
        "post_latent_geometry_poll": final_geometry_poll,
        "free_indices": free,
        "t0_fixed": float(fixed_t0),
    }
