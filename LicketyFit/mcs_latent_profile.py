"""Nonlinear Fermi--Eyges latent-path profiling for coherent charge fits.

This module is deliberately separate from the accepted production continuation.
It implements the statistically appropriate local estimator for a stochastic
Fermi--Eyges trajectory:

* infer standardized KL path coefficients with their unit Gaussian prior;
* retain the exact nonlinear finite-aperture charge prediction in every line
  search;
* use a Fisher--Laplace term when comparing different global tracks, so a track
  is not rewarded merely for opening a larger latent volume;
* construct the profiled global-track curvature with the latent Schur
  complement.

No event truth, empirical MCS scale, detector-location correction, or fitted
bias parameter enters the calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from .mcs_coupled_schur import _psd_inverse
from .mcs_curved_path import MCSPhysicalDomainError


@dataclass(frozen=True)
class LatentIteration:
    iteration: int
    charge_nll: float
    posterior_nll: float
    gradient_norm: float
    gradient_max_abs: float
    newton_decrement: float
    proposed_step_norm: float
    accepted_scale: float
    accepted: bool


@dataclass
class LatentMAPResult:
    coefficients: np.ndarray
    charge_nll: float
    posterior_nll: float
    laplace_nll: float
    information: np.ndarray
    covariance: np.ndarray
    logdet_information: float
    charge_jacobian: np.ndarray
    prediction: np.ndarray
    iterations: tuple[LatentIteration, ...]
    converged: bool
    final_gradient: np.ndarray
    final_gradient_norm: float
    final_gradient_max_abs: float
    final_newton_decrement: float
    termination_reason: str
    objective_evaluations: int
    jacobian_evaluations: int
    solver_method: str = "armijo_fisher_gauss_newton"
    laplace_valid: bool = True
    information_kind: str = "poisson_fisher"
    optimizer_success: bool | None = None
    optimizer_message: str = ""
    optimizer_nfev: int = 0
    local_poll_radii: tuple[float, ...] = ()
    local_poll_max_downhill: tuple[float, ...] = ()
    prediction_score: np.ndarray | None = None


@dataclass
class ProfiledTrackStep:
    delta_theta: np.ndarray
    delta_scaled: np.ndarray
    gradient_scaled: np.ndarray
    information_theta_scaled: np.ndarray
    information_cross_scaled: np.ndarray
    information_latent: np.ndarray
    information_profiled_scaled: np.ndarray
    free_indices: tuple[int, ...]
    finite_difference_steps: np.ndarray
    stencil_schemes: tuple[str, ...]


@dataclass(frozen=True)
class ProfileIteration:
    cycle: int
    theta_before: tuple[float, ...]
    theta_after: tuple[float, ...]
    laplace_before: float
    laplace_after: float
    accepted_scale: float
    accepted: bool
    proposed_delta: tuple[float, ...]


@dataclass
class ProfiledTrackResult:
    theta: np.ndarray
    latent: LatentMAPResult
    iterations: tuple[ProfileIteration, ...]
    converged: bool


@dataclass
class DirectionalProfiledTrackStep:
    """Schur-complement proposal in arbitrary physical track directions.

    ``coordinate_vectors`` are derivatives of the seven local track
    coordinates with respect to named *physical* coordinates.  This permits a
    geometry-general longitudinal start displacement along the fitted track,
    rather than hard-coding the WCTE global ``z0`` coordinate.
    """

    delta_theta: np.ndarray
    delta_scaled: np.ndarray
    gradient_scaled: np.ndarray
    information_theta_scaled: np.ndarray
    information_cross_scaled: np.ndarray
    information_latent: np.ndarray
    information_profiled_scaled: np.ndarray
    coordinate_vectors: np.ndarray
    coordinate_steps: np.ndarray
    coordinate_labels: tuple[str, ...]
    stencil_schemes: tuple[str, ...]


def _as_latent_vector(model, coefficients=None) -> np.ndarray:
    n = int(getattr(model, "n_modes", 0))
    if n <= 0 or n % 2 != 0:
        raise ValueError("coherent model must expose a positive even n_modes")
    if coefficients is None:
        return np.zeros(n, dtype=np.float64)
    out = np.asarray(coefficients, dtype=np.float64).reshape(n)
    if np.any(~np.isfinite(out)):
        raise ValueError("latent coefficients must be finite")
    return np.ascontiguousarray(out)


def _latent_steps(model, step) -> np.ndarray:
    n = int(model.n_modes)
    if np.isscalar(step):
        out = np.full(n, float(step), dtype=np.float64)
    else:
        out = np.asarray(step, dtype=np.float64).reshape(n)
    if np.any(~np.isfinite(out)) or np.any(out <= 0.0):
        raise ValueError("latent finite-difference steps must be positive and finite")
    return out


def finite_difference_charge_jacobian(
    model,
    coefficients,
    *,
    step: float | Sequence[float] = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, dmu/du)`` for the nonlinear coherent charge field.

    Support-tracked FALI exposes an analytic derivative of the same finite-disk
    integral and event-normalized charge marginal.  Use it when available; the
    central finite-difference path remains an exact regression fallback for
    alternate receiver models.
    """
    u = _as_latent_vector(model, coefficients)
    analytic = (
        None
        if bool(getattr(model, "force_finite_difference_charge_jacobian", False))
        else getattr(model, "charge_prediction_and_jacobian", None)
    )
    if analytic is not None:
        try:
            mu0, jac = analytic(u)
            return (
                np.ascontiguousarray(mu0, dtype=np.float64),
                np.ascontiguousarray(jac, dtype=np.float64),
            )
        except (NotImplementedError, FloatingPointError):
            pass
    h = _latent_steps(model, step)
    mu0 = np.asarray(model.charge_prediction(u), dtype=np.float64)
    jac = np.empty((mu0.size, u.size), dtype=np.float64)
    for k in range(u.size):
        up = u.copy(); um = u.copy()
        up[k] += h[k]; um[k] -= h[k]
        mup = np.asarray(model.charge_prediction(up), dtype=np.float64)
        mum = np.asarray(model.charge_prediction(um), dtype=np.float64)
        if mup.shape != mu0.shape or mum.shape != mu0.shape:
            raise RuntimeError("charge support changed across latent finite difference")
        jac[:, k] = (mup - mum) / (2.0 * h[k])
    return np.ascontiguousarray(mu0), np.ascontiguousarray(jac)


def _poisson_score_and_fisher(obs, mu, jacobian, coefficients):
    q = np.asarray(obs, dtype=np.float64)
    m = np.maximum(np.asarray(mu, dtype=np.float64), 1.0e-12)
    j = np.asarray(jacobian, dtype=np.float64)
    u = np.asarray(coefficients, dtype=np.float64)
    residual_factor = 1.0 - q / m
    gradient = j.T @ residual_factor + u
    information = np.eye(u.size, dtype=np.float64) + j.T @ (j / m[:, None])
    information = 0.5 * (information + information.T)
    return gradient, information


def _charge_score_and_information(model, mu, jacobian, coefficients):
    """Exact configured charge score with a positive shape preconditioner.

    Compound-SPE fits obtain their exact profiled score from the detector
    response.  A Poisson-shape Fisher matrix is retained only as a positive
    proposal/preconditioner; exact-posterior Armijo tests decide every step.
    It therefore cannot change the target or the stationarity condition.
    """
    prediction_interface = getattr(
        model, "charge_data_nll_and_score_from_prediction", None
    )
    interface = getattr(model, "charge_data_nll_and_score", None)
    if interface is not None:
        value, prediction_score = interface(coefficients)
    elif prediction_interface is not None:
        value, prediction_score = prediction_interface(mu)
    else:
        gradient, information = _poisson_score_and_fisher(
            model.obs_pes, mu, jacobian, coefficients
        )
        q = np.asarray(model.obs_pes, dtype=np.float64)
        safe_mu = np.maximum(np.asarray(mu, dtype=np.float64), 1.0e-300)
        value = float(np.sum(safe_mu - q * np.log(safe_mu)))
        prediction_score = 1.0 - q / np.maximum(safe_mu, 1.0e-12)
        return value, np.ascontiguousarray(prediction_score), gradient, information
    score = np.asarray(prediction_score, dtype=np.float64).reshape(mu.shape)
    u = np.asarray(coefficients, dtype=np.float64)
    j = np.asarray(jacobian, dtype=np.float64)
    m = np.maximum(np.asarray(mu, dtype=np.float64), 1.0e-12)
    gradient = j.T @ score + u
    information = np.eye(u.size, dtype=np.float64) + j.T @ (j / m[:, None])
    information = 0.5 * (information + information.T)
    return float(value), np.ascontiguousarray(score), gradient, information


def _posterior_charge_nll(model, coefficients) -> tuple[float, float]:
    u = _as_latent_vector(model, coefficients)
    charge = float(model.charge_data_nll(u))
    posterior = charge + 0.5 * float(u @ u)
    if not math.isfinite(posterior):
        return float("inf"), float("inf")
    return charge, posterior


def _laplace_summary(model, coefficients, *, fd_step):
    u = _as_latent_vector(model, coefficients)
    mu, jac = finite_difference_charge_jacobian(model, u, step=fd_step)
    charge, score, gradient, information = _charge_score_and_information(
        model, mu, jac, u
    )
    inverse, eigenvalues, cutoff, keep = _psd_inverse(
        information, relative_floor=1.0e-12, absolute_floor=1.0e-12
    )
    if not np.all(keep):
        # The unit Gaussian prior makes the exact Fisher matrix strictly
        # positive definite. A lost mode therefore indicates numerical failure.
        raise RuntimeError(
            "latent Fisher information lost rank despite the unit Gaussian prior "
            f"(cutoff={cutoff}, eigenvalues={eigenvalues})"
        )
    logdet = float(np.sum(np.log(eigenvalues[keep])))
    return mu, jac, charge, score, gradient, information, inverse, logdet


def solve_latent_charge_map(
    model,
    *,
    initial_coefficients=None,
    fd_step: float | Sequence[float] = 0.20,
    max_iterations: int = 60,
    gradient_tolerance: float = 1.0e-3,
    step_tolerance: float = 2.0e-3,
    trust_max_component: float = 1.0,
    line_search_scales: Iterable[float] | None = None,
    armijo_c1: float = 1.0e-4,
    min_line_search_scale: float = 2.0 ** -16,
) -> LatentMAPResult:
    """Profile the nonlinear coherent charge field over FE coefficients.

    The unit-normal FE coordinates make the Gaussian prior contribution exactly
    ``0.5*u.T@u``.  At every iterate the exact nonlinear Poisson posterior and
    its analytic FALI score are evaluated.  The expected Poisson information is
    used only as a positive-definite preconditioner/proposal; an Armijo search
    on the exact posterior decides whether the proposal is accepted.

    ``step_tolerance`` remains in the public signature for compatibility but is
    deliberately not a convergence condition.  A damped step can be tiny while
    the score is still large.  Convergence is determined only after recomputing
    the final score, using ``max(abs(score)) <= gradient_tolerance``.
    """
    u = _as_latent_vector(model, initial_coefficients)
    history: list[LatentIteration] = []
    termination_reason = "iteration_budget_exhausted"
    # A hard cap prevents a pathological event from monopolizing a worker.  It
    # is a ceiling, not an early-stopping heuristic; ordinary warm starts stop
    # from the final score well before it.
    max_iterations = min(max(0, int(max_iterations)), 200)
    trust = max(float(trust_max_component), 0.0)
    tolerance = float(gradient_tolerance)
    c1 = float(armijo_c1)
    minimum_scale = float(min_line_search_scale)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("gradient_tolerance must be positive and finite")
    if not math.isfinite(c1) or not 0.0 < c1 < 1.0:
        raise ValueError("armijo_c1 must lie strictly between zero and one")
    if not math.isfinite(minimum_scale) or not 0.0 < minimum_scale <= 1.0:
        raise ValueError("min_line_search_scale must lie in (0, 1]")
    # Kept only so callers using the old keyword are not broken.  It must not
    # turn a small damped step into a false convergence flag.
    _ = step_tolerance

    requested_scales = (
        [1.0]
        if line_search_scales is None
        else [float(scale) for scale in line_search_scales]
    )
    scales: list[float] = []
    for scale in requested_scales:
        if math.isfinite(scale) and scale > 0.0 and scale not in scales:
            scales.append(scale)
    if not scales:
        scales = [1.0]
    # The old five-point search stopped at 1/16 and falsely declared many
    # otherwise solvable events stalled.  Extend any caller-supplied schedule by
    # powers of two down to the numerical safeguard.
    tail = min(scales)
    while tail > minimum_scale:
        tail *= 0.5
        if tail < minimum_scale:
            tail = minimum_scale
        if tail not in scales:
            scales.append(tail)
        if tail <= minimum_scale:
            break

    objective_evaluations = 0
    jacobian_evaluations = 0
    last_summary = None

    charge, posterior = _posterior_charge_nll(model, u)
    objective_evaluations += 1
    for iteration in range(max_iterations):
        mu, jac = finite_difference_charge_jacobian(model, u, step=fd_step)
        jacobian_evaluations += 1
        _score_charge, _prediction_score, gradient, information = (
            _charge_score_and_information(model, mu, jac, u)
        )
        inv, eigenvalues, cutoff, keep = _psd_inverse(
            information, relative_floor=1.0e-12, absolute_floor=1.0e-12
        )
        if not np.all(keep):
            raise RuntimeError(
                "latent Fisher information lost rank despite the unit Gaussian "
                f"prior (cutoff={cutoff}, eigenvalues={eigenvalues})"
            )
        # Retain the complete summary at this exact iterate. Gradient exits
        # and failed line searches return the same ``u``; rebuilding its FALI
        # field and Jacobian below would repeat an identical, costly
        # evaluation. Accepted steps replace ``u``, so the exact-equality
        # guard below deliberately refuses reuse after an iteration-budget
        # exit at a newly accepted point.
        last_summary = (
            u.copy(),
            np.ascontiguousarray(mu),
            np.ascontiguousarray(jac),
            float(_score_charge),
            np.ascontiguousarray(_prediction_score),
            np.ascontiguousarray(gradient),
            np.ascontiguousarray(information),
            np.ascontiguousarray(inv),
            float(np.sum(np.log(eigenvalues[keep]))),
        )
        preconditioned_gradient = inv @ gradient
        decrement_sq = max(float(gradient @ preconditioned_gradient), 0.0)
        newton_decrement = math.sqrt(decrement_sq)
        gradient_norm = float(np.linalg.norm(gradient))
        gradient_max_abs = (
            float(np.max(np.abs(gradient))) if gradient.size else 0.0
        )
        if gradient_max_abs <= tolerance:
            history.append(
                LatentIteration(
                    iteration=iteration,
                    charge_nll=float(charge),
                    posterior_nll=float(posterior),
                    gradient_norm=gradient_norm,
                    gradient_max_abs=gradient_max_abs,
                    newton_decrement=newton_decrement,
                    proposed_step_norm=0.0,
                    accepted_scale=0.0,
                    accepted=False,
                )
            )
            termination_reason = "gradient_tolerance"
            break

        raw_step = -preconditioned_gradient
        directional_derivative = float(gradient @ raw_step)
        used_descent_fallback = False
        if (
            np.any(~np.isfinite(raw_step))
            or not math.isfinite(directional_derivative)
            or directional_derivative >= 0.0
        ):
            # This should not occur for the prior-regularized Fisher matrix, but
            # it is safer to retain a guaranteed descent direction than to move
            # uphill because of a numerical eigensolve failure.
            scale_gradient = max(gradient_max_abs, 1.0)
            raw_step = -gradient / scale_gradient
            directional_derivative = float(gradient @ raw_step)
            used_descent_fallback = True

        if trust > 0.0:
            maximum = float(np.max(np.abs(raw_step))) if raw_step.size else 0.0
            if maximum > trust:
                raw_step *= trust / maximum
            directional_derivative = float(gradient @ raw_step)
        proposed_norm = float(np.linalg.norm(raw_step))
        if directional_derivative >= 0.0 or not math.isfinite(directional_derivative):
            raise RuntimeError("latent solver failed to construct a finite descent direction")

        accepted = False
        accepted_scale = 0.0
        best_u = u
        best_charge = charge
        best_posterior = posterior
        for scale in scales:
            candidate = u + scale * raw_step
            c_charge, c_post = _posterior_charge_nll(model, candidate)
            objective_evaluations += 1
            armijo_bound = posterior + c1 * scale * directional_derivative
            if math.isfinite(c_post) and c_post <= armijo_bound:
                accepted = True
                accepted_scale = scale
                best_u = candidate
                best_charge = c_charge
                best_posterior = c_post
                break
        if not accepted and not used_descent_fallback:
            # The Fisher/Gauss--Newton direction is a preconditioned descent
            # direction, but sharp optical active-set transitions can make its
            # local model unhelpful.  Retry once with guaranteed steepest
            # descent before declaring the exact-posterior line search stalled.
            fallback_step = -gradient / max(gradient_max_abs, 1.0)
            if trust > 0.0:
                maximum = (
                    float(np.max(np.abs(fallback_step)))
                    if fallback_step.size else 0.0
                )
                if maximum > trust:
                    fallback_step *= trust / maximum
            fallback_directional = float(gradient @ fallback_step)
            if math.isfinite(fallback_directional) and fallback_directional < 0.0:
                for scale in scales:
                    candidate = u + scale * fallback_step
                    c_charge, c_post = _posterior_charge_nll(model, candidate)
                    objective_evaluations += 1
                    armijo_bound = posterior + c1 * scale * fallback_directional
                    if math.isfinite(c_post) and c_post <= armijo_bound:
                        accepted = True
                        accepted_scale = scale
                        best_u = candidate
                        best_charge = c_charge
                        best_posterior = c_post
                        raw_step = fallback_step
                        proposed_norm = float(np.linalg.norm(fallback_step))
                        break
        history.append(
            LatentIteration(
                iteration=iteration,
                charge_nll=float(charge),
                posterior_nll=float(posterior),
                gradient_norm=gradient_norm,
                gradient_max_abs=gradient_max_abs,
                newton_decrement=newton_decrement,
                proposed_step_norm=proposed_norm,
                accepted_scale=float(accepted_scale),
                accepted=bool(accepted),
            )
        )
        if not accepted:
            termination_reason = "armijo_line_search_failed"
            break
        u = np.ascontiguousarray(best_u)
        charge = float(best_charge)
        posterior = float(best_posterior)

    # Recompute only when the returned nonlinear MAP has not already been
    # summarized. Exact equality is intentional: this optimization must not
    # substitute a nearby point or alter the fitted trajectory in any way.
    if last_summary is not None and np.array_equal(last_summary[0], u):
        (
            _summary_u,
            _mu,
            jac,
            charge,
            final_prediction_score,
            final_gradient,
            information,
            covariance,
            logdet,
        ) = last_summary
    else:
        (
            _mu,
            jac,
            charge,
            final_prediction_score,
            final_gradient,
            information,
            covariance,
            logdet,
        ) = _laplace_summary(model, u, fd_step=fd_step)
        jacobian_evaluations += 1
    final_gradient_norm = float(np.linalg.norm(final_gradient))
    final_gradient_max_abs = float(np.max(np.abs(final_gradient))) if u.size else 0.0
    final_newton_decrement = math.sqrt(
        max(float(final_gradient @ covariance @ final_gradient), 0.0)
    )
    converged = bool(final_gradient_max_abs <= tolerance)
    if converged:
        termination_reason = "gradient_tolerance"
    # Preserve the scalar production-NLL reduction as the authoritative
    # reported objective.  The score kernel is mathematically identical but
    # can differ by a final floating-point rounding in compound-SPE mode.
    charge, posterior = _posterior_charge_nll(model, u)
    objective_evaluations += 1
    laplace = posterior + 0.5 * logdet
    return LatentMAPResult(
        coefficients=np.ascontiguousarray(u),
        charge_nll=float(charge),
        posterior_nll=float(posterior),
        laplace_nll=float(laplace),
        information=np.ascontiguousarray(information),
        covariance=np.ascontiguousarray(covariance),
        logdet_information=float(logdet),
        charge_jacobian=np.ascontiguousarray(jac),
        prediction=np.ascontiguousarray(_mu),
        prediction_score=np.ascontiguousarray(final_prediction_score),
        iterations=tuple(history),
        converged=bool(converged),
        final_gradient=np.ascontiguousarray(final_gradient),
        final_gradient_norm=final_gradient_norm,
        final_gradient_max_abs=final_gradient_max_abs,
        final_newton_decrement=float(final_newton_decrement),
        termination_reason=str(termination_reason),
        objective_evaluations=int(objective_evaluations),
        jacobian_evaluations=int(jacobian_evaluations),
    )


def solve_latent_charge_map_derivative_free(
    model,
    *,
    initial_coefficients=None,
    max_evaluations: int = 600,
    initial_trust_radius: float = 0.5,
    final_trust_radius: float = 3.0e-3,
    poll_radii: Sequence[float] = (1.0e-2, 3.0e-3),
    poll_tolerance: float = 1.0e-4,
) -> LatentMAPResult:
    """Minimize the *exact* latent posterior without using its Jacobian.

    This is the correctness path for a deterministic optical forward model
    whose discretized support topology does not yet possess a numerically
    convergent first derivative.  COBYQA only chooses trial FE coefficients;
    every accepted value is the unchanged physical Poisson charge NLL plus the
    analytic standard-normal FE prior ``0.5*u.T@u``.  There are no coefficient
    bounds.  The sole implicit domain restriction is the unit-tangent physical
    domain enforced by the coherent path model, whose rejected proposals are
    assigned infinite objective.

    Convergence is an operational exact-posterior test, independent of the
    optimizer status: both signs of every latent coordinate are polled at every
    requested radius, and none may improve the posterior by more than
    ``poll_tolerance``.  This is deliberately stricter than treating a small
    trust radius or an exhausted evaluation budget as convergence.

    A Fisher matrix derived from the currently inconsistent analytic optical
    Jacobian would not be a valid local Hessian.  Consequently this function
    returns NaNs for gradient, information, covariance, log determinant, and
    Laplace NLL, with ``laplace_valid=False``.  A future numerical Hessian may
    populate those quantities only after multi-step stability and positive-
    definiteness checks.
    """
    u0 = _as_latent_vector(model, initial_coefficients)
    max_evaluations = int(max_evaluations)
    initial_radius = float(initial_trust_radius)
    final_radius = float(final_trust_radius)
    tolerance = float(poll_tolerance)
    radii = tuple(float(radius) for radius in poll_radii)
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    if (
        not math.isfinite(initial_radius)
        or not math.isfinite(final_radius)
        or initial_radius <= 0.0
        or final_radius <= 0.0
        or initial_radius <= final_radius
    ):
        raise ValueError(
            "COBYQA trust radii must be finite and satisfy initial > final > 0"
        )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("poll_tolerance must be finite and non-negative")
    if not radii or any(
        (not math.isfinite(radius) or radius <= 0.0) for radius in radii
    ):
        raise ValueError("poll_radii must contain positive finite values")

    objective_evaluations = 0
    best_u: np.ndarray | None = None
    best_charge = float("inf")
    best_posterior = float("inf")

    def exact_posterior(coefficients) -> float:
        nonlocal objective_evaluations, best_u, best_charge, best_posterior
        objective_evaluations += 1
        candidate = np.asarray(coefficients, dtype=np.float64).reshape(u0.size)
        if np.any(~np.isfinite(candidate)):
            return float("inf")
        try:
            charge, posterior = _posterior_charge_nll(model, candidate)
        except MCSPhysicalDomainError:
            return float("inf")
        if not math.isfinite(posterior):
            return float("inf")
        if posterior < best_posterior:
            best_u = np.ascontiguousarray(candidate.copy())
            best_charge = float(charge)
            best_posterior = float(posterior)
        return float(posterior)

    initial_value = exact_posterior(u0)
    if not math.isfinite(initial_value):
        raise ValueError(
            "initial latent coefficients lie outside the physical posterior domain"
        )

    optimizer = minimize(
        exact_posterior,
        u0,
        method="COBYQA",
        options={
            "maxfev": max_evaluations,
            "initial_tr_radius": initial_radius,
            "final_tr_radius": final_radius,
            "scale": False,
            "disp": False,
        },
    )
    # COBYQA normally returns its best evaluated point.  Retaining the best
    # finite exact evaluation makes the physical contract explicit even if its
    # final interpolation geometry contains an invalid-domain proposal.
    candidate_u = np.asarray(optimizer.x, dtype=np.float64).reshape(u0.size)
    candidate_value = exact_posterior(candidate_u)
    if math.isfinite(candidate_value) and candidate_value <= best_posterior:
        u = np.ascontiguousarray(candidate_u.copy())
        charge, posterior = _posterior_charge_nll(model, u)
        charge = float(charge)
        posterior = float(posterior)
    else:
        if best_u is None:  # Defensive: the finite initial point sets best_u.
            raise RuntimeError("derivative-free latent solver lost every finite point")
        u = np.ascontiguousarray(best_u.copy())
        charge = float(best_charge)
        posterior = float(best_posterior)

    poll_downhill: list[float] = []
    for radius in radii:
        maximum_downhill = 0.0
        for index in range(u.size):
            for sign in (-1.0, 1.0):
                trial = u.copy()
                trial[index] += sign * radius
                trial_value = exact_posterior(trial)
                if math.isfinite(trial_value):
                    maximum_downhill = max(
                        maximum_downhill, float(posterior - trial_value)
                    )
        poll_downhill.append(maximum_downhill)

    converged = bool(all(value <= tolerance for value in poll_downhill))
    termination_reason = (
        "derivative_free_coordinate_poll_stationary"
        if converged
        else "derivative_free_coordinate_poll_found_downhill"
    )
    prediction = np.asarray(model.charge_prediction(u), dtype=np.float64)
    nan_vector = np.full(u.size, np.nan, dtype=np.float64)
    nan_matrix = np.full((u.size, u.size), np.nan, dtype=np.float64)
    nan_jacobian = np.full((prediction.size, u.size), np.nan, dtype=np.float64)
    return LatentMAPResult(
        coefficients=np.ascontiguousarray(u),
        charge_nll=float(charge),
        posterior_nll=float(posterior),
        laplace_nll=float("nan"),
        information=np.ascontiguousarray(nan_matrix),
        covariance=np.ascontiguousarray(nan_matrix.copy()),
        logdet_information=float("nan"),
        charge_jacobian=np.ascontiguousarray(nan_jacobian),
        prediction=np.ascontiguousarray(prediction),
        prediction_score=None,
        iterations=(),
        converged=converged,
        final_gradient=np.ascontiguousarray(nan_vector),
        final_gradient_norm=float("nan"),
        final_gradient_max_abs=float("nan"),
        final_newton_decrement=float("nan"),
        termination_reason=termination_reason,
        objective_evaluations=int(objective_evaluations),
        jacobian_evaluations=0,
        solver_method="exact_posterior_cobyqa_coordinate_poll",
        laplace_valid=False,
        information_kind="unavailable_until_stable_numerical_hessian",
        optimizer_success=bool(optimizer.success),
        optimizer_message=str(optimizer.message),
        optimizer_nfev=int(getattr(optimizer, "nfev", 0)),
        local_poll_radii=radii,
        local_poll_max_downhill=tuple(map(float, poll_downhill)),
    )


def _adaptive_charge_derivative(evaluator, theta, u, index, step):
    base_model = evaluator.model(theta)
    if base_model is None:
        raise ValueError("invalid reference track")
    base = np.asarray(base_model.charge_prediction(u), dtype=np.float64)

    def prediction(offset):
        candidate = theta.copy()
        candidate[index] += float(offset) * step
        model = evaluator.model(candidate)
        if model is None:
            return None
        try:
            return np.asarray(model.charge_prediction(u), dtype=np.float64)
        except Exception:
            return None

    plus = prediction(+1.0)
    minus = prediction(-1.0)
    if plus is not None and minus is not None:
        return (plus - minus) / (2.0 * step), "central"
    if plus is not None:
        plus2 = prediction(+0.5)
        if plus2 is not None:
            # Second-order forward derivative at x using f(x), f(x+h/2), f(x+h).
            return (-3.0 * base + 4.0 * plus2 - plus) / step, "forward_second"
        return (plus - base) / step, "forward_first"
    if minus is not None:
        minus2 = prediction(-0.5)
        if minus2 is not None:
            return (3.0 * base - 4.0 * minus2 + minus) / step, "backward_second"
        return (base - minus) / step, "backward_first"
    raise RuntimeError(f"no physical finite-difference support for theta index {index}")


def _adaptive_charge_directional_derivative(
    evaluator,
    theta,
    u,
    direction,
    step,
    *,
    one_sided_half_step=False,
):
    """Differentiate charge along one arbitrary physical track coordinate.

    ``direction`` has the same seven-coordinate convention as ``theta`` and
    is normalized by the caller's physical definition, not by its Euclidean
    norm.  For example, a unit longitudinal-start coordinate is
    ``(cx,cy,cz,0,0,0,0)`` and therefore measures millimetres along the fitted
    trajectory in every detector orientation.
    """
    t = np.asarray(theta, dtype=np.float64).reshape(7)
    v = np.asarray(direction, dtype=np.float64).reshape(7)
    h = float(step)
    if np.any(~np.isfinite(v)) or not math.isfinite(h) or h <= 0.0:
        raise ValueError("directional finite difference requires finite direction and step")
    base_model = evaluator.model(t)
    if base_model is None:
        raise ValueError("invalid reference track")
    base = np.asarray(base_model.charge_prediction(u), dtype=np.float64)

    def prediction(offset):
        candidate = t + float(offset) * h * v
        model = evaluator.model(candidate)
        if model is None:
            return None
        try:
            return np.asarray(model.charge_prediction(u), dtype=np.float64)
        except Exception:
            return None

    if bool(one_sided_half_step):
        # A half-step one-sided stencil needs one optical model instead of the
        # two models required by the symmetric stencil.  The derivative is
        # still expressed in the caller's full scaled coordinate.  Every
        # proposed global move remains subject to the unchanged exact
        # posterior line search, so this only accelerates proposal formation.
        plus_half = prediction(+0.5)
        if plus_half is not None:
            return 2.0 * (plus_half - base), "forward_half_first"
        minus_half = prediction(-0.5)
        if minus_half is not None:
            return 2.0 * (base - minus_half), "backward_half_first"
        raise RuntimeError(
            "no physical one-sided finite-difference support for directional coordinate"
        )

    plus = prediction(+1.0)
    minus = prediction(-1.0)
    if plus is not None and minus is not None:
        # Derivative with respect to the *scaled* coordinate, whose unit step
        # is h in the physical coordinate.  This matches the scaled Fisher
        # convention used by profiled_charge_track_step.
        return 0.5 * (plus - minus), "central"
    if plus is not None:
        plus2 = prediction(+0.5)
        if plus2 is not None:
            return -3.0 * base + 4.0 * plus2 - plus, "forward_second"
        return plus - base, "forward_first"
    if minus is not None:
        minus2 = prediction(-0.5)
        if minus2 is not None:
            return 3.0 * base - 4.0 * minus2 + minus, "backward_second"
        return base - minus, "backward_first"
    raise RuntimeError("no physical finite-difference support for directional coordinate")


def profiled_charge_track_step_directions(
    evaluator,
    theta,
    latent_result: LatentMAPResult,
    *,
    coordinate_vectors: Sequence[Sequence[float]],
    coordinate_steps: Sequence[float],
    coordinate_labels: Sequence[str] | None = None,
    coordinate_prior_gradient_scaled: Sequence[float] | None = None,
    coordinate_prior_information_scaled: Sequence[Sequence[float]] | None = None,
    trust_max_scaled_component: float = 1.0,
    one_sided_half_step: bool = False,
) -> DirectionalProfiledTrackStep:
    """Return a latent-profiled proposal in arbitrary physical coordinates.

    Optional prior derivatives are expressed in the same dimensionless scaled
    coordinates as the finite-difference columns.  They support analytic,
    independently specified physics priors (for example stopping-range
    straggling) without folding those priors into the detector response.
    """
    t = np.asarray(theta, dtype=np.float64).reshape(7)
    vectors = np.asarray(coordinate_vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 7 or vectors.shape[0] < 1:
        raise ValueError("coordinate_vectors must have shape (n_coordinates, 7)")
    steps = np.asarray(coordinate_steps, dtype=np.float64).reshape(vectors.shape[0])
    if np.any(~np.isfinite(vectors)) or np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("coordinate vectors and steps must be finite; steps must be positive")
    labels = (
        tuple(f"coordinate_{i}" for i in range(vectors.shape[0]))
        if coordinate_labels is None
        else tuple(str(x) for x in coordinate_labels)
    )
    if len(labels) != vectors.shape[0]:
        raise ValueError("coordinate_labels length must match coordinate_vectors")
    prior_gradient = (
        np.zeros(vectors.shape[0], dtype=np.float64)
        if coordinate_prior_gradient_scaled is None
        else np.asarray(
            coordinate_prior_gradient_scaled, dtype=np.float64
        ).reshape(vectors.shape[0])
    )
    prior_information = (
        np.zeros((vectors.shape[0], vectors.shape[0]), dtype=np.float64)
        if coordinate_prior_information_scaled is None
        else np.asarray(
            coordinate_prior_information_scaled, dtype=np.float64
        ).reshape(vectors.shape[0], vectors.shape[0])
    )
    if np.any(~np.isfinite(prior_gradient)) or np.any(~np.isfinite(prior_information)):
        raise ValueError("coordinate prior derivatives must be finite")
    prior_information = 0.5 * (prior_information + prior_information.T)

    model = evaluator.model(t)
    if model is None:
        raise ValueError("invalid reference track")
    u = _as_latent_vector(model, latent_result.coefficients)
    mu = np.asarray(latent_result.prediction, dtype=np.float64)
    ju = np.asarray(latent_result.charge_jacobian, dtype=np.float64)
    jtheta_scaled = np.empty((mu.size, vectors.shape[0]), dtype=np.float64)
    schemes: list[str] = []
    for column, (vector, step) in enumerate(zip(vectors, steps)):
        derivative_scaled, scheme = _adaptive_charge_directional_derivative(
            evaluator,
            t,
            u,
            vector,
            float(step),
            one_sided_half_step=bool(one_sided_half_step),
        )
        jtheta_scaled[:, column] = derivative_scaled
        schemes.append(scheme)

    m = np.maximum(mu, 1.0e-12)
    # The production charge model is generally compound-SPE, not Poisson.
    # Use its exact score for the global-track stationarity condition, just as
    # the latent MAP solver does.  The Poisson-shape Fisher blocks below are a
    # positive proposal/preconditioner only; exact objective line searches in
    # the caller decide whether a proposed global move is accepted.
    cached_score = getattr(latent_result, "prediction_score", None)
    score_interface = getattr(model, "charge_data_nll_and_score", None)
    if cached_score is not None:
        prediction_score = np.asarray(cached_score, dtype=np.float64).reshape(
            m.shape
        )
    elif score_interface is None:
        q = np.asarray(evaluator.obs_pes, dtype=np.float64)
        prediction_score = 1.0 - q / m
    else:
        _value, prediction_score = score_interface(u)
        prediction_score = np.asarray(
            prediction_score, dtype=np.float64
        ).reshape(m.shape)
    gtheta = jtheta_scaled.T @ prediction_score + prior_gradient
    htt = (
        jtheta_scaled.T @ (jtheta_scaled / m[:, None])
        + prior_information
    )
    htu = jtheta_scaled.T @ (ju / m[:, None])
    huu = np.eye(u.size) + ju.T @ (ju / m[:, None])
    huu = 0.5 * (huu + huu.T)
    huu_inv, _evals, _cutoff, _keep = _psd_inverse(
        huu, relative_floor=1.0e-12, absolute_floor=1.0e-12
    )
    hprofile = htt - htu @ huu_inv @ htu.T
    hprofile = 0.5 * (hprofile + hprofile.T)
    hp_inv, _evalp, _cutp, _keepp = _psd_inverse(
        hprofile, relative_floor=1.0e-10, absolute_floor=1.0e-12
    )
    delta_scaled = -(hp_inv @ gtheta)
    trust = max(float(trust_max_scaled_component), 0.0)
    if trust > 0.0 and delta_scaled.size:
        maximum = float(np.max(np.abs(delta_scaled)))
        if maximum > trust:
            delta_scaled *= trust / maximum
    physical_coordinate_delta = delta_scaled * steps
    delta_theta = physical_coordinate_delta @ vectors
    return DirectionalProfiledTrackStep(
        delta_theta=np.ascontiguousarray(delta_theta),
        delta_scaled=np.ascontiguousarray(delta_scaled),
        gradient_scaled=np.ascontiguousarray(gtheta),
        information_theta_scaled=np.ascontiguousarray(0.5 * (htt + htt.T)),
        information_cross_scaled=np.ascontiguousarray(htu),
        information_latent=np.ascontiguousarray(huu),
        information_profiled_scaled=np.ascontiguousarray(hprofile),
        coordinate_vectors=np.ascontiguousarray(vectors),
        coordinate_steps=np.ascontiguousarray(steps),
        coordinate_labels=labels,
        stencil_schemes=tuple(schemes),
    )


def profiled_charge_track_step(
    evaluator,
    theta,
    latent_result: LatentMAPResult,
    *,
    theta_fd: Sequence[float],
    free_indices: Sequence[int],
    latent_fd: float | Sequence[float] = 0.20,
    trust_max_scaled_component: float = 1.0,
) -> ProfiledTrackStep:
    """Return one Schur-complement Newton proposal for the global track."""
    t = np.asarray(theta, dtype=np.float64).reshape(7)
    steps = np.asarray(theta_fd, dtype=np.float64).reshape(7)
    if np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("theta_fd must contain seven positive finite steps")
    free = tuple(int(i) for i in free_indices)
    if not free or any(i < 0 or i >= 7 for i in free):
        raise ValueError("free_indices must select at least one of seven coordinates")

    model = evaluator.model(t)
    if model is None:
        raise ValueError("invalid reference track")
    u = _as_latent_vector(model, latent_result.coefficients)
    # The latent solver already performed an exact central Jacobian and
    # prediction at its returned MAP for the Fisher--Laplace summary.  Reusing
    # those arrays is algebraically identical and avoids one complete 2*Nmode
    # FALI stencil per global-track cycle.
    mu = np.asarray(latent_result.prediction, dtype=np.float64)
    ju = np.asarray(latent_result.charge_jacobian, dtype=np.float64)
    jtheta_scaled = np.empty((mu.size, len(free)), dtype=np.float64)
    schemes: list[str] = []
    for column, index in enumerate(free):
        derivative, scheme = _adaptive_charge_derivative(
            evaluator, t, u, index, float(steps[index])
        )
        # derivative above is per physical coordinate; scaled coordinate one
        # corresponds to one configured finite-difference step.
        jtheta_scaled[:, column] = derivative * float(steps[index])
        schemes.append(scheme)

    m = np.maximum(mu, 1.0e-12)
    # Match the configured production charge likelihood in the global-track
    # gradient.  The positive Poisson-shape Fisher matrix remains only a
    # preconditioner, and therefore does not redefine the fitted objective.
    cached_score = getattr(latent_result, "prediction_score", None)
    score_interface = getattr(model, "charge_data_nll_and_score", None)
    if cached_score is not None:
        prediction_score = np.asarray(cached_score, dtype=np.float64).reshape(
            m.shape
        )
    elif score_interface is None:
        q = np.asarray(evaluator.obs_pes, dtype=np.float64)
        prediction_score = 1.0 - q / m
    else:
        _value, prediction_score = score_interface(u)
        prediction_score = np.asarray(
            prediction_score, dtype=np.float64
        ).reshape(m.shape)
    gtheta = jtheta_scaled.T @ prediction_score
    htt = jtheta_scaled.T @ (jtheta_scaled / m[:, None])
    htu = jtheta_scaled.T @ (ju / m[:, None])
    huu = np.eye(u.size) + ju.T @ (ju / m[:, None])
    huu = 0.5 * (huu + huu.T)
    huu_inv, _evals, _cutoff, _keep = _psd_inverse(
        huu, relative_floor=1.0e-12, absolute_floor=1.0e-12
    )
    hprofile = htt - htu @ huu_inv @ htu.T
    hprofile = 0.5 * (hprofile + hprofile.T)
    hp_inv, _evalp, _cutp, _keepp = _psd_inverse(
        hprofile, relative_floor=1.0e-10, absolute_floor=1.0e-12
    )
    delta_scaled_free = -(hp_inv @ gtheta)
    trust = max(float(trust_max_scaled_component), 0.0)
    if trust > 0.0 and delta_scaled_free.size:
        maximum = float(np.max(np.abs(delta_scaled_free)))
        if maximum > trust:
            delta_scaled_free *= trust / maximum
    delta_scaled = np.zeros(7, dtype=np.float64)
    delta_scaled[np.asarray(free, dtype=int)] = delta_scaled_free
    delta = delta_scaled * steps
    return ProfiledTrackStep(
        delta_theta=np.ascontiguousarray(delta),
        delta_scaled=np.ascontiguousarray(delta_scaled),
        gradient_scaled=np.ascontiguousarray(gtheta),
        information_theta_scaled=np.ascontiguousarray(0.5 * (htt + htt.T)),
        information_cross_scaled=np.ascontiguousarray(htu),
        information_latent=np.ascontiguousarray(huu),
        information_profiled_scaled=np.ascontiguousarray(hprofile),
        free_indices=free,
        finite_difference_steps=np.ascontiguousarray(steps),
        stencil_schemes=tuple(schemes),
    )


def optimize_profiled_laplace_track(
    evaluator,
    theta,
    *,
    free_indices: Sequence[int],
    theta_fd: Sequence[float],
    latent_fd: float | Sequence[float] = 0.20,
    latent_max_iterations: int = 60,
    candidate_latent_max_iterations: int | None = None,
    track_cycles: int = 2,
    track_trust_max_scaled_component: float = 1.0,
    line_search_scales: Iterable[float] = (1.0, 0.5, 0.25, 0.125),
    initial_coefficients=None,
    final_latent_max_iterations: int | None = None,
) -> ProfiledTrackResult:
    """Alternately profile the coherent path and update selected track coordinates.

    A global-track proposal is accepted only when the re-profiled
    Fisher--Laplace objective decreases. This is the key guard against latent
    MAP overfitting.
    """
    current_theta = np.asarray(theta, dtype=np.float64).reshape(7).copy()
    full_latent_iterations = min(max(1, int(latent_max_iterations)), 60)
    # Retain the old keyword for API compatibility, but never evaluate a
    # candidate with a cheaper/less-converged latent solve than the base point.
    # Comparing partially profiled Laplace values biases outer acceptance.
    _ = candidate_latent_max_iterations
    candidate_iterations = full_latent_iterations
    model = evaluator.model(current_theta)
    if model is None:
        raise ValueError("invalid initial track")
    latent = solve_latent_charge_map(
        model,
        initial_coefficients=initial_coefficients,
        fd_step=latent_fd,
        max_iterations=full_latent_iterations,
    )
    history: list[ProfileIteration] = []
    converged = False

    for cycle in range(max(0, int(track_cycles))):
        step = profiled_charge_track_step(
            evaluator,
            current_theta,
            latent,
            theta_fd=theta_fd,
            free_indices=free_indices,
            latent_fd=latent_fd,
            trust_max_scaled_component=track_trust_max_scaled_component,
        )
        if float(np.linalg.norm(step.delta_theta)) <= 1.0e-6:
            converged = True
            break
        base_theta = current_theta.copy()
        base_laplace = float(latent.laplace_nll)
        accepted = False
        accepted_scale = 0.0
        best_theta = current_theta
        best_latent = latent
        for scale in line_search_scales:
            scale = float(scale)
            if scale <= 0.0:
                continue
            candidate_theta = base_theta + scale * step.delta_theta
            candidate_model = evaluator.model(candidate_theta)
            if candidate_model is None:
                continue
            candidate_latent = solve_latent_charge_map(
                candidate_model,
                initial_coefficients=latent.coefficients,
                fd_step=latent_fd,
                max_iterations=candidate_iterations,
            )
            if not candidate_latent.converged:
                continue
            if candidate_latent.laplace_nll < base_laplace - 1.0e-8:
                accepted = True
                accepted_scale = scale
                best_theta = candidate_theta
                best_latent = candidate_latent
                break
        history.append(
            ProfileIteration(
                cycle=cycle,
                theta_before=tuple(map(float, base_theta)),
                theta_after=tuple(map(float, best_theta)),
                laplace_before=base_laplace,
                laplace_after=float(best_latent.laplace_nll),
                accepted_scale=float(accepted_scale),
                accepted=bool(accepted),
                proposed_delta=tuple(map(float, step.delta_theta)),
            )
        )
        if not accepted:
            break
        current_theta = np.asarray(best_theta, dtype=np.float64)
        latent = best_latent
        if float(np.linalg.norm(accepted_scale * step.delta_theta)) <= 1.0e-3:
            converged = True
            break

    # Every accepted outer point must be returned at a comparably solved latent
    # MAP.  The reported path is always solved to the same full budget; the
    # legacy final-budget keyword cannot lower that standard.
    final_budget = full_latent_iterations
    if final_latent_max_iterations is not None:
        final_budget = min(
            max(final_budget, int(final_latent_max_iterations)), 60
        )
    final_model = evaluator.model(current_theta)
    if final_model is None:
        raise RuntimeError("accepted coherent track became invalid before final profiling")
    latent = solve_latent_charge_map(
        final_model,
        initial_coefficients=latent.coefficients,
        fd_step=latent_fd,
        max_iterations=final_budget,
    )
    return ProfiledTrackResult(
        theta=np.ascontiguousarray(current_theta),
        latent=latent,
        iterations=tuple(history),
        converged=bool(converged and latent.converged),
    )


def optimize_profiled_laplace_track_aligned(
    evaluator,
    theta,
    *,
    longitudinal_step_mm: float = 10.0,
    length_step_mm: float = 15.0,
    latent_fd: float | Sequence[float] = 0.20,
    latent_max_iterations: int = 60,
    candidate_latent_max_iterations: int | None = None,
    latent_trust_max_component: float = 1.0,
    track_cycles: int = 2,
    track_trust_max_scaled_component: float = 1.0,
    line_search_scales: Iterable[float] = (1.0, 0.5, 0.25, 0.125),
    initial_coefficients=None,
    final_latent_max_iterations: int | None = None,
    profile_start_along: bool = True,
) -> ProfiledTrackResult:
    """Profile the coherent path and the two longitudinal endpoint coordinates.

    The global coordinates are a displacement of the start point along the
    *current fitted direction* and an independent visible-length displacement.
    This is the arbitrary-orientation equivalent of the historical ``(z0,L)``
    block used for a +z WCTE beam.  It is derived from the supplied track and
    therefore contains no detector-location special case.
    """
    current_theta = np.asarray(theta, dtype=np.float64).reshape(7).copy()
    full_latent_iterations = min(max(1, int(latent_max_iterations)), 60)
    _ = candidate_latent_max_iterations
    candidate_iterations = full_latent_iterations
    model = evaluator.model(current_theta)
    if model is None:
        raise ValueError("invalid initial track")
    latent = solve_latent_charge_map(
        model,
        initial_coefficients=initial_coefficients,
        fd_step=latent_fd,
        max_iterations=full_latent_iterations,
        trust_max_component=latent_trust_max_component,
    )
    history: list[ProfileIteration] = []
    converged = False

    for cycle in range(max(0, int(track_cycles))):
        direction = evaluator.chart.direction(
            float(current_theta[3]), float(current_theta[4])
        )
        if direction is None:
            raise ValueError("invalid track direction in aligned profile")
        if bool(profile_start_along):
            vectors = np.zeros((2, 7), dtype=np.float64)
            vectors[0, :3] = direction
            vectors[1, 5] = 1.0
            coordinate_steps = (
                float(longitudinal_step_mm),
                float(length_step_mm),
            )
            coordinate_labels = ("start_along_track", "visible_length")
        else:
            vectors = np.zeros((1, 7), dtype=np.float64)
            vectors[0, 5] = 1.0
            coordinate_steps = (float(length_step_mm),)
            coordinate_labels = ("visible_length",)
        step = profiled_charge_track_step_directions(
            evaluator,
            current_theta,
            latent,
            coordinate_vectors=vectors,
            coordinate_steps=coordinate_steps,
            coordinate_labels=coordinate_labels,
            trust_max_scaled_component=track_trust_max_scaled_component,
        )
        if float(np.linalg.norm(step.delta_theta)) <= 1.0e-6:
            converged = True
            break
        base_theta = current_theta.copy()
        base_laplace = float(latent.laplace_nll)
        accepted = False
        accepted_scale = 0.0
        best_theta = current_theta
        best_latent = latent
        for scale in line_search_scales:
            scale = float(scale)
            if scale <= 0.0:
                continue
            candidate_theta = base_theta + scale * step.delta_theta
            candidate_model = evaluator.model(candidate_theta)
            if candidate_model is None:
                continue
            candidate_latent = solve_latent_charge_map(
                candidate_model,
                initial_coefficients=latent.coefficients,
                fd_step=latent_fd,
                max_iterations=candidate_iterations,
                trust_max_component=latent_trust_max_component,
            )
            if not candidate_latent.converged:
                continue
            if candidate_latent.laplace_nll < base_laplace - 1.0e-8:
                accepted = True
                accepted_scale = scale
                best_theta = candidate_theta
                best_latent = candidate_latent
                break
        history.append(
            ProfileIteration(
                cycle=cycle,
                theta_before=tuple(map(float, base_theta)),
                theta_after=tuple(map(float, best_theta)),
                laplace_before=base_laplace,
                laplace_after=float(best_latent.laplace_nll),
                accepted_scale=float(accepted_scale),
                accepted=bool(accepted),
                proposed_delta=tuple(map(float, step.delta_theta)),
            )
        )
        if not accepted:
            break
        current_theta = np.asarray(best_theta, dtype=np.float64)
        latent = best_latent
        if float(np.linalg.norm(accepted_scale * step.delta_theta)) <= 1.0e-3:
            converged = True
            break

    final_budget = full_latent_iterations
    if final_latent_max_iterations is not None:
        final_budget = min(
            max(final_budget, int(final_latent_max_iterations)), 60
        )
    final_model = evaluator.model(current_theta)
    if final_model is None:
        raise RuntimeError("accepted coherent track became invalid before final profiling")
    latent = solve_latent_charge_map(
        final_model,
        initial_coefficients=latent.coefficients,
        fd_step=latent_fd,
        max_iterations=final_budget,
        trust_max_component=latent_trust_max_component,
    )
    return ProfiledTrackResult(
        theta=np.ascontiguousarray(current_theta),
        latent=latent,
        iterations=tuple(history),
        converged=bool(converged and latent.converged),
    )
