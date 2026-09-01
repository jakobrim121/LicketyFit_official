"""Full-rank reduced-response profiling for coherent Fermi--Eyges paths.

The expensive coherent optical operator is nonlinear in the standardized
Fermi--Eyges/KL coordinates.  Around an exact physical state ``u`` this module
uses its analytic tangent in *log PMT rate*,

    log r(u + d) = log r(u) + diag(1 / r(u)) J(u) d,

and solves the resulting event-normalized Poisson posterior in all 24 latent
coordinates.  The log-rate form preserves positive rates; the standard-normal
FE prior remains exact.  Every proposed update is accepted only after an
evaluation of the unchanged nonlinear optical posterior.

This is a deterministic physics-response method, not a learned surrogate.  It
does not truncate KL modes, use event truth, or fit detector/energy-specific
coefficients.  A low optical grid may propose a path, but the production
controller finishes with response corrections and an exact scalar evaluation
on the authoritative G241 model.
"""
from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from .mcs_curved_path import MCSPhysicalDomainError


def posterior_charge_nll(model, coefficients) -> tuple[float, float]:
    """Return the exact nonlinear ``(charge, posterior)`` pair."""
    u = np.asarray(coefficients, dtype=np.float64).reshape(int(model.n_modes))
    charge = float(model.charge_data_nll(u))
    posterior = charge + 0.5 * float(u @ u)
    if not math.isfinite(charge) or not math.isfinite(posterior):
        return float("inf"), float("inf")
    return charge, posterior


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def solve_log_rate_tangent(
    prediction,
    jacobian,
    observed,
    centre,
    *,
    max_iterations: int = 30,
    gradient_tolerance: float = 1.0e-8,
):
    """Solve one convex event-normalized log-rate response model exactly."""
    mu = np.maximum(np.asarray(prediction, dtype=np.float64), 1.0e-12)
    jac = np.asarray(jacobian, dtype=np.float64)
    q = np.asarray(observed, dtype=np.float64)
    u0 = np.asarray(centre, dtype=np.float64)
    if jac.shape != (mu.size, u0.size):
        raise ValueError("response Jacobian has an incompatible shape")
    if q.shape != mu.shape:
        raise ValueError("observed charge and response prediction differ in shape")
    total = float(np.sum(q))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("coherent response profiling requires positive total charge")

    log_mu = np.log(mu)
    response = jac / mu[:, None]
    displacement = np.zeros_like(u0)

    def value(delta):
        logits = log_mu + response @ delta
        shifted = u0 + delta
        return float(
            total * _logsumexp(logits)
            - q @ logits
            + 0.5 * float(shifted @ shifted)
        )

    current = value(displacement)
    iterations = 0
    final_gradient_max_abs = math.inf
    for iteration in range(max(1, int(max_iterations))):
        iterations = iteration + 1
        logits = log_mu + response @ displacement
        maximum = float(np.max(logits))
        weight = np.exp(logits - maximum)
        probability = weight / float(np.sum(weight))
        gradient = (
            response.T @ (total * probability - q) + u0 + displacement
        )
        final_gradient_max_abs = float(np.max(np.abs(gradient)))
        if final_gradient_max_abs <= float(gradient_tolerance):
            break
        weighted_response = total * probability[:, None] * response
        mean_response = np.sum(weighted_response, axis=0)
        hessian = (
            np.eye(u0.size, dtype=np.float64)
            + response.T @ weighted_response
            - np.outer(mean_response, mean_response) / total
        )
        hessian = 0.5 * (hessian + hessian.T)
        eigenvalue, eigenvector = np.linalg.eigh(hessian)
        eigenvalue = np.maximum(eigenvalue, 1.0e-10)
        step = -eigenvector @ ((eigenvector.T @ gradient) / eigenvalue)
        directional = float(gradient @ step)
        accepted = False
        scale = 1.0
        while scale >= 2.0 ** -20:
            trial = displacement + scale * step
            trial_value = value(trial)
            if (
                math.isfinite(trial_value)
                and trial_value <= current + 1.0e-4 * scale * directional
            ):
                displacement = trial
                current = trial_value
                accepted = True
                break
            scale *= 0.5
        if not accepted or float(np.max(np.abs(scale * step))) <= 1.0e-8:
            break
    return {
        "displacement": np.ascontiguousarray(displacement),
        "model_nll": float(current),
        "iterations": int(iterations),
        "gradient_max_abs": float(final_gradient_max_abs),
    }


def solve_sequential_physics_response(
    model,
    *,
    initial_coefficients=None,
    outer_iterations: int = 40,
    trust_max_component: float = 1.0,
    relative_improvement_tolerance: float = 1.0e-7,
    acceptance_scales: Sequence[float] = (
        1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625
    ),
):
    """Sequential log-rate response solve with exact nonlinear acceptance."""
    n_modes = int(model.n_modes)
    if initial_coefficients is None:
        u = np.zeros(n_modes, dtype=np.float64)
    else:
        u = np.asarray(initial_coefficients, dtype=np.float64).reshape(n_modes).copy()
    trust = float(trust_max_component)
    if not math.isfinite(trust) or trust <= 0.0:
        raise ValueError("response trust_max_component must be positive and finite")
    scales = tuple(float(scale) for scale in acceptance_scales)
    if not scales or any((not math.isfinite(x) or x <= 0.0) for x in scales):
        raise ValueError("response acceptance scales must be positive and finite")

    charge, current = posterior_charge_nll(model, u)
    if not math.isfinite(current):
        raise MCSPhysicalDomainError("initial response state is not physical")
    history = []
    objective_evaluations = 1
    jacobian_evaluations = 0
    start = time.perf_counter()
    termination = "iteration_budget"

    for iteration in range(max(0, int(outer_iterations))):
        jacobian_start = time.perf_counter()
        prediction, jacobian = model.charge_prediction_and_jacobian(u)
        jacobian_wall = float(time.perf_counter() - jacobian_start)
        jacobian_evaluations += 1
        local = solve_log_rate_tangent(
            prediction,
            jacobian,
            model.obs_pes,
            u,
        )
        step = np.asarray(local["displacement"], dtype=np.float64)
        maximum = float(np.max(np.abs(step))) if step.size else 0.0
        if maximum > trust:
            step *= trust / maximum

        accepted = False
        accepted_scale = 0.0
        best_charge = charge
        best_posterior = current
        best_u = u
        physical_trials = 0
        for scale in scales:
            trial = u + scale * step
            try:
                trial_charge, trial_posterior = posterior_charge_nll(model, trial)
            except (MCSPhysicalDomainError, FloatingPointError, OverflowError, ValueError):
                continue
            objective_evaluations += 1
            physical_trials += 1
            if trial_posterior < best_posterior - 1.0e-12:
                best_charge = trial_charge
                best_posterior = trial_posterior
                best_u = trial
                accepted_scale = scale
                accepted = True
                break

        improvement = float(current - best_posterior)
        history.append(
            {
                "iteration": int(iteration),
                "charge_nll_before": float(charge),
                "posterior_before": float(current),
                "posterior_after": float(best_posterior),
                "improvement": improvement,
                "proposed_step_norm": float(np.linalg.norm(step)),
                "proposed_step_max_abs": float(np.max(np.abs(step))),
                "accepted": bool(accepted),
                "accepted_scale": float(accepted_scale),
                "physical_trial_count": int(physical_trials),
                "local_model_iterations": int(local["iterations"]),
                "local_model_gradient_max_abs": float(local["gradient_max_abs"]),
                "jacobian_wall_s": jacobian_wall,
            }
        )
        if not accepted:
            termination = "no_exact_nonlinear_improvement"
            break
        u = np.ascontiguousarray(best_u)
        charge = float(best_charge)
        current = float(best_posterior)
        if improvement <= float(relative_improvement_tolerance) * max(
            abs(current), 1.0
        ):
            termination = "relative_exact_improvement"
            break

    return {
        "coefficients": np.ascontiguousarray(u),
        "charge_nll": float(charge),
        "posterior_nll": float(current),
        "wall_s": float(time.perf_counter() - start),
        "objective_evaluations": int(objective_evaluations),
        "jacobian_evaluations": int(jacobian_evaluations),
        "iterations": int(len(history)),
        "termination_reason": str(termination),
        "history": tuple(history),
        "finite_physical": bool(math.isfinite(current)),
    }


def solve_competing_physics_response(
    model,
    *,
    primary_trust: float = 1.0,
    secondary_trust: float = 1.5,
    probe_iterations: int = 12,
    total_iterations: int = 40,
):
    """Probe two prior-standardized trust scales, then continue their exact minimum."""
    begin = time.perf_counter()
    probe = min(max(1, int(probe_iterations)), max(1, int(total_iterations)))
    primary = solve_sequential_physics_response(
        model,
        outer_iterations=probe,
        trust_max_component=float(primary_trust),
    )
    secondary = solve_sequential_physics_response(
        model,
        outer_iterations=probe,
        trust_max_component=float(secondary_trust),
    )
    if primary["posterior_nll"] <= secondary["posterior_nll"]:
        chosen = primary
        chosen_name = "primary"
        chosen_trust = float(primary_trust)
    else:
        chosen = secondary
        chosen_name = "secondary"
        chosen_trust = float(secondary_trust)
    remainder = max(int(total_iterations) - probe, 0)
    continuation = None
    if remainder > 0:
        continuation = solve_sequential_physics_response(
            model,
            initial_coefficients=chosen["coefficients"],
            outer_iterations=remainder,
            trust_max_component=chosen_trust,
        )
        final = continuation
    else:
        final = chosen
    return {
        "coefficients": np.ascontiguousarray(final["coefficients"]),
        "charge_nll": float(final["charge_nll"]),
        "posterior_nll": float(final["posterior_nll"]),
        "wall_s": float(time.perf_counter() - begin),
        "objective_evaluations": int(
            primary["objective_evaluations"]
            + secondary["objective_evaluations"]
            + (0 if continuation is None else continuation["objective_evaluations"])
        ),
        "jacobian_evaluations": int(
            primary["jacobian_evaluations"]
            + secondary["jacobian_evaluations"]
            + (0 if continuation is None else continuation["jacobian_evaluations"])
        ),
        "iterations": int(
            primary["iterations"]
            + secondary["iterations"]
            + (0 if continuation is None else continuation["iterations"])
        ),
        "termination_reason": str(final["termination_reason"]),
        "history": tuple(chosen["history"])
        + (() if continuation is None else tuple(continuation["history"])),
        "finite_physical": bool(final["finite_physical"]),
        "competing": {
            "primary_trust": float(primary_trust),
            "secondary_trust": float(secondary_trust),
            "probe_iterations": int(probe),
            "primary_probe_posterior": float(primary["posterior_nll"]),
            "secondary_probe_posterior": float(secondary["posterior_nll"]),
            "chosen": str(chosen_name),
            "chosen_trust": float(chosen_trust),
        },
    }


def solve_reduced_response_with_authoritative_correction(
    proposal_model,
    authoritative_model,
    *,
    primary_trust: float = 1.0,
    secondary_trust: float = 1.5,
    probe_iterations: int = 12,
    proposal_iterations: int = 40,
    correction_iterations: int = 4,
):
    """Run the full-rank proposal and finish on the authoritative physics."""
    start = time.perf_counter()
    proposal = solve_competing_physics_response(
        proposal_model,
        primary_trust=float(primary_trust),
        secondary_trust=float(secondary_trust),
        probe_iterations=int(probe_iterations),
        total_iterations=int(proposal_iterations),
    )
    correction = solve_sequential_physics_response(
        authoritative_model,
        initial_coefficients=proposal["coefficients"],
        outer_iterations=int(correction_iterations),
        trust_max_component=float(proposal["competing"]["chosen_trust"]),
    )
    return {
        "coefficients": np.ascontiguousarray(correction["coefficients"]),
        "charge_nll": float(correction["charge_nll"]),
        "posterior_nll": float(correction["posterior_nll"]),
        "converged": bool(
            proposal["finite_physical"] and correction["finite_physical"]
        ),
        "wall_s": float(time.perf_counter() - start),
        "objective_evaluations": int(
            proposal["objective_evaluations"]
            + correction["objective_evaluations"]
        ),
        "jacobian_evaluations": int(
            proposal["jacobian_evaluations"]
            + correction["jacobian_evaluations"]
        ),
        "nfev": int(
            proposal["objective_evaluations"]
            + correction["objective_evaluations"]
        ),
        "evaluations": int(
            proposal["objective_evaluations"]
            + correction["objective_evaluations"]
        ),
        "nit": int(proposal["iterations"] + correction["iterations"]),
        "optimizer_success": bool(
            proposal["finite_physical"] and correction["finite_physical"]
        ),
        "message": "full-rank physics-tangent response completed",
        "poll": (),
        "max_poll_downhill": math.nan,
        "restart_count": 0,
        "max_restarts": 0,
        "history": (),
        "proposal": proposal,
        "authoritative_correction": correction,
        "exact_scalar_evaluated": True,
        "exact_coordinate_stationarity_claimed": False,
    }
