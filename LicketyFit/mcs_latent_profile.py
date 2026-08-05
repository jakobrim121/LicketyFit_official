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

from .mcs_coupled_schur import _psd_inverse


@dataclass(frozen=True)
class LatentIteration:
    iteration: int
    charge_nll: float
    posterior_nll: float
    gradient_norm: float
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
    analytic = getattr(model, "charge_prediction_and_jacobian", None)
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
    _gradient, information = _poisson_score_and_fisher(
        model.obs_pes, mu, jac, u
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
    return mu, jac, information, inverse, logdet


def solve_latent_charge_map(
    model,
    *,
    initial_coefficients=None,
    fd_step: float | Sequence[float] = 0.20,
    max_iterations: int = 7,
    gradient_tolerance: float = 2.0e-3,
    step_tolerance: float = 2.0e-3,
    trust_max_component: float = 1.0,
    line_search_scales: Iterable[float] = (1.0, 0.5, 0.25, 0.125, 0.0625),
) -> LatentMAPResult:
    """Profile the nonlinear coherent charge field over FE coefficients.

    Fisher scoring proposes each step, while acceptance is decided by the exact
    nonlinear Poisson-plus-Gaussian posterior. The result is therefore not a
    linearized charge correction.
    """
    u = _as_latent_vector(model, initial_coefficients)
    history: list[LatentIteration] = []
    converged = False
    max_iterations = max(0, int(max_iterations))
    trust = max(float(trust_max_component), 0.0)

    charge, posterior = _posterior_charge_nll(model, u)
    for iteration in range(max_iterations):
        mu, jac = finite_difference_charge_jacobian(model, u, step=fd_step)
        gradient, information = _poisson_score_and_fisher(
            model.obs_pes, mu, jac, u
        )
        inv, _evals, _cutoff, _keep = _psd_inverse(
            information, relative_floor=1.0e-12, absolute_floor=1.0e-12
        )
        raw_step = -(inv @ gradient)
        proposed_norm = float(np.linalg.norm(raw_step))
        if trust > 0.0:
            maximum = float(np.max(np.abs(raw_step))) if raw_step.size else 0.0
            if maximum > trust:
                raw_step *= trust / maximum
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= gradient_tolerance or proposed_norm <= step_tolerance:
            history.append(
                LatentIteration(
                    iteration, charge, posterior, gradient_norm,
                    proposed_norm, 0.0, False,
                )
            )
            converged = True
            break

        accepted = False
        accepted_scale = 0.0
        best_u = u
        best_charge = charge
        best_posterior = posterior
        for scale in line_search_scales:
            scale = float(scale)
            if scale <= 0.0:
                continue
            candidate = u + scale * raw_step
            c_charge, c_post = _posterior_charge_nll(model, candidate)
            if c_post < best_posterior - 1.0e-10:
                accepted = True
                accepted_scale = scale
                best_u = candidate
                best_charge = c_charge
                best_posterior = c_post
                break
        history.append(
            LatentIteration(
                iteration, charge, posterior, gradient_norm,
                proposed_norm, accepted_scale, accepted,
            )
        )
        if not accepted:
            break
        actual_step = float(np.linalg.norm(best_u - u))
        u = np.ascontiguousarray(best_u)
        charge = float(best_charge)
        posterior = float(best_posterior)
        if actual_step <= step_tolerance:
            converged = True
            break

    # Recompute at the returned nonlinear MAP for a self-consistent Laplace term.
    _mu, jac, information, covariance, logdet = _laplace_summary(
        model, u, fd_step=fd_step
    )
    charge, posterior = _posterior_charge_nll(model, u)
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
        iterations=tuple(history),
        converged=bool(converged),
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
    trust_max_scaled_component: float = 1.0,
) -> DirectionalProfiledTrackStep:
    """Return a latent-profiled proposal in arbitrary physical coordinates."""
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
            evaluator, t, u, vector, float(step)
        )
        jtheta_scaled[:, column] = derivative_scaled
        schemes.append(scheme)

    q = np.asarray(evaluator.obs_pes, dtype=np.float64)
    m = np.maximum(mu, 1.0e-12)
    score_factor = 1.0 - q / m
    gtheta = jtheta_scaled.T @ score_factor
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

    q = np.asarray(evaluator.obs_pes, dtype=np.float64)
    m = np.maximum(mu, 1.0e-12)
    score_factor = 1.0 - q / m
    gtheta = jtheta_scaled.T @ score_factor
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
    latent_max_iterations: int = 7,
    candidate_latent_max_iterations: int | None = None,
    track_cycles: int = 2,
    track_trust_max_scaled_component: float = 1.0,
    line_search_scales: Iterable[float] = (1.0, 0.5, 0.25, 0.125),
    initial_coefficients=None,
) -> ProfiledTrackResult:
    """Alternately profile the coherent path and update selected track coordinates.

    A global-track proposal is accepted only when the re-profiled
    Fisher--Laplace objective decreases. This is the key guard against latent
    MAP overfitting.
    """
    current_theta = np.asarray(theta, dtype=np.float64).reshape(7).copy()
    candidate_iterations = (
        int(latent_max_iterations)
        if candidate_latent_max_iterations is None
        else max(0, int(candidate_latent_max_iterations))
    )
    model = evaluator.model(current_theta)
    if model is None:
        raise ValueError("invalid initial track")
    latent = solve_latent_charge_map(
        model,
        initial_coefficients=initial_coefficients,
        fd_step=latent_fd,
        max_iterations=latent_max_iterations,
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

    return ProfiledTrackResult(
        theta=np.ascontiguousarray(current_theta),
        latent=latent,
        iterations=tuple(history),
        converged=bool(converged),
    )


def optimize_profiled_laplace_track_aligned(
    evaluator,
    theta,
    *,
    longitudinal_step_mm: float = 10.0,
    length_step_mm: float = 15.0,
    latent_fd: float | Sequence[float] = 0.20,
    latent_max_iterations: int = 7,
    candidate_latent_max_iterations: int | None = None,
    latent_trust_max_component: float = 1.0,
    track_cycles: int = 2,
    track_trust_max_scaled_component: float = 1.0,
    line_search_scales: Iterable[float] = (1.0, 0.5, 0.25, 0.125),
    initial_coefficients=None,
) -> ProfiledTrackResult:
    """Profile the coherent path and the two longitudinal endpoint coordinates.

    The global coordinates are a displacement of the start point along the
    *current fitted direction* and an independent visible-length displacement.
    This is the arbitrary-orientation equivalent of the historical ``(z0,L)``
    block used for a +z WCTE beam.  It is derived from the supplied track and
    therefore contains no detector-location special case.
    """
    current_theta = np.asarray(theta, dtype=np.float64).reshape(7).copy()
    candidate_iterations = (
        int(latent_max_iterations)
        if candidate_latent_max_iterations is None
        else max(0, int(candidate_latent_max_iterations))
    )
    model = evaluator.model(current_theta)
    if model is None:
        raise ValueError("invalid initial track")
    latent = solve_latent_charge_map(
        model,
        initial_coefficients=initial_coefficients,
        fd_step=latent_fd,
        max_iterations=latent_max_iterations,
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
        vectors = np.zeros((2, 7), dtype=np.float64)
        vectors[0, :3] = direction
        vectors[1, 5] = 1.0
        step = profiled_charge_track_step_directions(
            evaluator,
            current_theta,
            latent,
            coordinate_vectors=vectors,
            coordinate_steps=(float(longitudinal_step_mm), float(length_step_mm)),
            coordinate_labels=("start_along_track", "visible_length"),
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

    return ProfiledTrackResult(
        theta=np.ascontiguousarray(current_theta),
        latent=latent,
        iterations=tuple(history),
        converged=bool(converged),
    )
