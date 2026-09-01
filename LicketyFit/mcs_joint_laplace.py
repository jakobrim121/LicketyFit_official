"""Deterministic continuous K0/range/Gaussian-FE posterior integration.

This module targets the same truth-blind physical state as the reference
annealed SMC implementation: a broad continuous initial kinetic energy, a
standard-normal non-centred stopping-range fluctuation, and the complete
standard-normal coherent Fermi--Eyges coordinate vector.  It replaces random
posterior exploration with a local exact-score/Fisher solve followed by
Gauss--Hermite importance cubature in the two non-Gaussian global coordinates.

Every accepted optimization point and every cubature node is evaluated with
the unchanged nonlinear optical likelihood.  The response derivatives are
used only as a preconditioner and conditional-path transport.  There is no
energy fixing, fitted-range grid, mode truncation, empirical MCS rescaling, or
simulation-derived correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time

import numpy as np
from numpy.polynomial.hermite import hermgauss

from .mcs_joint_energy_range import JointEnergyRangeSMCResult
from .mcs_curved_path import MCSPhysicalDomainError


@dataclass(frozen=True)
class JointLaplaceCubatureConfig:
    energy_step_mev: float = 4.0
    z_range_step: float = 0.20
    maximum_iterations: int = 16
    latent_iterations: int = 3
    energy_trust_mev: float = 20.0
    z_range_trust: float = 0.75
    path_trust: float = 1.0
    gradient_tolerance: float = 2.0e-3
    cubature_order: int = 3
    minimum_line_scale: float = 0.0625
    response_fd_step: float = 0.20
    seed_energy_multipliers: tuple[float, ...] = (
        0.50,
        0.75,
        0.90,
        1.00,
        1.10,
        1.25,
    )
    seed_screen_latent_iterations: int = 5
    profile_t0: bool = True
    t0_profile_coarse_step_ns: float = 0.25
    t0_profile_refine_levels: int = 2
    t0_profile_global_points: int = 9
    t0_profile_seed_half_width_ns: float = 2.0

    def validate(self) -> None:
        for name, value in (
            ("energy_step_mev", self.energy_step_mev),
            ("z_range_step", self.z_range_step),
            ("energy_trust_mev", self.energy_trust_mev),
            ("z_range_trust", self.z_range_trust),
            ("path_trust", self.path_trust),
            ("gradient_tolerance", self.gradient_tolerance),
            ("minimum_line_scale", self.minimum_line_scale),
            ("response_fd_step", self.response_fd_step),
            ("t0_profile_coarse_step_ns", self.t0_profile_coarse_step_ns),
            ("t0_profile_seed_half_width_ns", self.t0_profile_seed_half_width_ns),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if int(self.maximum_iterations) < 1:
            raise ValueError("maximum_iterations must be positive")
        if int(self.latent_iterations) < 0:
            raise ValueError("latent_iterations must be non-negative")
        if int(self.cubature_order) < 3:
            raise ValueError("cubature_order must be at least three")
        if int(self.seed_screen_latent_iterations) < 0:
            raise ValueError("seed_screen_latent_iterations must be non-negative")
        if int(self.t0_profile_refine_levels) < 0:
            raise ValueError("t0_profile_refine_levels must be non-negative")
        if int(self.t0_profile_global_points) < 1:
            raise ValueError("t0_profile_global_points must be positive")
        multipliers = tuple(float(value) for value in self.seed_energy_multipliers)
        if not multipliers or any(
            not math.isfinite(value) or value <= 0.0 for value in multipliers
        ):
            raise ValueError("seed_energy_multipliers must be finite and positive")
        if float(self.minimum_line_scale) > 1.0:
            raise ValueError("minimum_line_scale must not exceed one")


@dataclass
class JointLaplaceCubatureResult:
    posterior: JointEnergyRangeSMCResult
    coefficients_mean: np.ndarray
    coefficients_covariance: np.ndarray
    mode_energy_mev: float
    mode_z_range: float
    mode_range_mm: float
    mode_t0_ns: float
    posterior_t0_mean_ns: float
    mode_coefficients: np.ndarray
    mode_charge_nll: float
    mode_posterior_nll: float
    map_sample_index: int
    map_charge_nll: float
    diagnostics: dict


@dataclass
class _CentralResponseLatent:
    """Exact-posterior latent state using numerical physical response scores.

    The Jacobian is never used as a surrogate likelihood.  It supplies only a
    positive Fisher preconditioner; every line-search point is accepted or
    rejected with the unchanged nonlinear optical prediction and configured
    data likelihood.  In charge-plus-time mode the score itself is the central
    difference of the complete configured likelihood; the charge-response
    Jacobian remains only a positive proposal metric.
    """

    coefficients: np.ndarray
    charge_nll: float
    posterior_nll: float
    information: np.ndarray
    covariance: np.ndarray
    charge_jacobian: np.ndarray
    prediction: np.ndarray
    final_gradient: np.ndarray
    iterations: int
    objective_evaluations: int
    jacobian_evaluations: int
    jacobian_field_evaluations: int
    invalid_physical_evaluations: int
    converged: bool
    prediction_wall_s: float
    profiled_t0_ns: float
    fisher_summary_complete: bool


def _charge_value_and_score(model, prediction):
    """Evaluate the configured charge likelihood from an existing prediction."""
    mu = np.maximum(np.asarray(prediction, dtype=np.float64), 1.0e-300)
    detector_response = getattr(model, "pmt_model", None)
    obs = np.asarray(model.obs_pes, dtype=np.float64)
    score_interface = (
        None if detector_response is None else getattr(
            detector_response, "get_neg_log_likelihood_npe_with_score", None
        )
    )
    if score_interface is not None:
        value, score = score_interface(mu, obs)
        return float(value), np.ascontiguousarray(score, dtype=np.float64)
    return (
        float(np.sum(mu - obs * np.log(mu))),
        np.ascontiguousarray(1.0 - obs / mu),
    )


def _profile_charge_time_t0(
    model,
    coefficients,
    *,
    seed_t0,
    bounds,
    coarse_step_ns,
    refine_levels,
    global_points,
    seed_half_width_ns,
):
    """Profile the additive event time using one cached optical prediction."""
    seed = float(seed_t0)
    if bounds is None:
        return float(model.data_nll(coefficients, t0=seed)), seed
    lo, hi = map(float, bounds)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise ValueError("t0 profile bounds must be finite and ordered")
    seed = float(np.clip(seed, lo, hi))
    if hi == lo:
        return float(model.data_nll(coefficients, t0=lo)), lo

    samples = {}

    def evaluate_many(points):
        pending = []
        keys = []
        for point in np.asarray(points, dtype=np.float64).reshape(-1):
            value = float(np.clip(float(point), lo, hi))
            key = round(value, 12)
            if key in samples or key in keys:
                continue
            pending.append(value)
            keys.append(key)
        if not pending:
            return
        interface = getattr(model, "data_nll_many_t0", None)
        if interface is None:
            values = np.asarray(
                [model.data_nll(coefficients, t0=value) for value in pending],
                dtype=np.float64,
            )
        else:
            values = np.asarray(
                interface(coefficients, pending), dtype=np.float64
            ).reshape(-1)
        if values.size != len(pending):
            raise RuntimeError("batched coherent t0 likelihood returned wrong size")
        for key, value in zip(keys, values, strict=True):
            samples[key] = float(value)

    # The preceding straight charge+time fit supplies a sharply localized t0.
    # Certify that seed with a symmetric exact stencil before paying for a
    # detector-wide scan at every coherent range/path candidate.  A locally
    # convex bracket whose centre beats both guards contains the unique nearby
    # minimum; its parabolic vertex is then scored by the unchanged likelihood.
    # Ambiguous, non-convex, or edge-pointing stencils fall through to the full
    # historical profile below.
    use_local_certificate = str(
        os.environ.get("LF_COHERENT_LOCAL_T0_CERTIFICATE", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if use_local_certificate:
        guard_step = min(
            max(float(coarse_step_ns), 1.0e-4),
            max(float(seed_half_width_ns), 1.0e-4),
            max(seed - lo, 0.0),
            max(hi - seed, 0.0),
        )
        if guard_step > 1.0e-8:
            guard_points = (seed - guard_step, seed, seed + guard_step)
            evaluate_many(guard_points)
            fm = samples.get(round(seed - guard_step, 12), math.inf)
            f0 = samples.get(round(seed, 12), math.inf)
            fp = samples.get(round(seed + guard_step, 12), math.inf)
            curvature = float(fm - 2.0 * f0 + fp)
            if (
                math.isfinite(fm)
                and math.isfinite(f0)
                and math.isfinite(fp)
                and f0 <= fm
                and f0 <= fp
                and math.isfinite(curvature)
                and curvature > 1.0e-10
            ):
                delta = float(
                    0.5 * guard_step * (fm - fp) / curvature
                )
                if abs(delta) < 0.90 * guard_step:
                    vertex = float(np.clip(seed + delta, lo, hi))
                    evaluate_many((vertex,))
                    finite_local = [
                        (float(point), float(value))
                        for point, value in samples.items()
                        if math.isfinite(value)
                    ]
                    if finite_local:
                        best_t0, best_value = min(
                            finite_local, key=lambda item: item[1]
                        )
                        return float(best_value), float(best_t0)

    local_lo = max(lo, seed - float(seed_half_width_ns))
    local_hi = min(hi, seed + float(seed_half_width_ns))
    local_count = max(
        1,
        int(math.ceil((local_hi - local_lo) / float(coarse_step_ns))),
    )
    # These three guards share one optical prediction. Submit them as one
    # ordered likelihood batch so source sorting and deferred-reflection setup
    # are paid once. Concatenation preserves the historical insertion and
    # duplicate-suppression order exactly: seed, global grid, then local grid.
    evaluate_many(np.concatenate((
        np.asarray((seed,), dtype=np.float64),
        np.linspace(lo, hi, int(global_points)),
        np.linspace(local_lo, local_hi, local_count + 1),
    )))
    for _ in range(int(refine_levels)):
        finite = sorted(
            (
                (float(point), float(value))
                for point, value in samples.items()
                if math.isfinite(value)
            ),
            key=lambda item: item[0],
        )
        if not finite:
            break
        best_index = min(range(len(finite)), key=lambda index: finite[index][1])
        best_t0 = finite[best_index][0]
        left = (
            finite[best_index - 1][0]
            if best_index > 0
            else max(lo, best_t0 - float(coarse_step_ns))
        )
        right = (
            finite[best_index + 1][0]
            if best_index + 1 < len(finite)
            else min(hi, best_t0 + float(coarse_step_ns))
        )
        if right <= left + 1.0e-15:
            break
        # Three neighboring exact samples determine the local quadratic
        # interpolant of this one-dimensional profile.  Evaluating its vertex
        # and two narrow guards gives the same 0.1-bracket resolution as the
        # historical eleven-point grid with at most three new likelihood
        # values.  Retain that dense grid whenever the bracket is asymmetric,
        # non-convex, or numerically ill-conditioned.
        use_dense_fallback = not (0 < best_index < len(finite) - 1)
        candidate_points = None
        if not use_dense_fallback:
            x0, f0 = finite[best_index - 1]
            x1, f1 = finite[best_index]
            x2, f2 = finite[best_index + 1]
            left_width = x1 - x0
            right_width = x2 - x1
            width_ratio = max(left_width, right_width) / max(
                min(left_width, right_width), 1.0e-300
            )
            slope_left = (f1 - f0) / left_width
            slope_right = (f2 - f1) / right_width
            curvature = (slope_right - slope_left) / (x2 - x0)
            denominator = (
                (x1 - x0) * (f1 - f2)
                - (x1 - x2) * (f1 - f0)
            )
            numerator = (
                (x1 - x0) ** 2 * (f1 - f2)
                - (x1 - x2) ** 2 * (f1 - f0)
            )
            if (
                math.isfinite(curvature)
                and curvature > 0.0
                and math.isfinite(denominator)
                and abs(denominator) > 1.0e-18
                and width_ratio <= 2.0
            ):
                vertex = float(x1 - 0.5 * numerator / denominator)
                if left < vertex < right:
                    guard = 0.10 * (right - left)
                    candidate_points = (
                        max(left, vertex - guard),
                        vertex,
                        min(right, vertex + guard),
                    )
            use_dense_fallback = candidate_points is None
        evaluate_many(
            np.linspace(left, right, 11)
            if use_dense_fallback
            else candidate_points
        )
    finite = [
        (float(point), float(value))
        for point, value in samples.items()
        if math.isfinite(value)
    ]
    if not finite:
        raise FloatingPointError("coherent t0 profile has no finite point")
    best_t0, best_value = min(finite, key=lambda item: item[1])
    return float(best_value), float(best_t0)


def _solve_latent_with_central_response(
    model,
    *,
    initial_coefficients,
    response_fd_step,
    max_iterations,
    gradient_tolerance,
    trust_max_component,
    minimum_line_scale,
    t0=None,
    profile_t0=False,
    t0_bounds=None,
    t0_profile_coarse_step_ns=0.25,
    t0_profile_refine_levels=2,
    t0_profile_global_points=9,
    t0_profile_seed_half_width_ns=2.0,
    finalize_fisher_summary=True,
    analytic_complete_response=False,
):
    """Profile FE coordinates with exact nonlinear central-difference scores.

    The production analytic 24-column receiver derivative is exceptionally
    slow in a one-thread event worker.  Central differences of the unchanged
    nonlinear field are both faster and the validation reference for that
    derivative.  The Fisher matrix remains only a proposal preconditioner and
    exact nonlinear posterior values accept every step.  Charge-only mode uses
    the configured analytic score with respect to predicted charge.  In
    charge-plus-time mode normally takes central differences directly through
    the complete charge-and-first-hit timing objective.  Disposable range
    candidates may instead use the analytic charge response only to propose a
    nuisance-path step; the complete charge-time posterior still accepts or
    rejects that step, and the selected final state always receives the full
    central-response summary.
    """
    u = np.asarray(initial_coefficients, dtype=np.float64).reshape(-1).copy()
    step_size = float(response_fd_step)
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("response_fd_step must be positive and finite")
    evaluations = 0
    jacobian_evaluations = 0
    jacobian_field_evaluations = 0
    invalid_physical_evaluations = 0
    prediction_wall_s = 0.0
    charge_only = bool(getattr(model, "charge_only", True))

    def state(coefficients, *, profile_event_time=False, fixed_t0=None):
        nonlocal evaluations, prediction_wall_s
        wall0 = time.perf_counter()
        evaluations += 1
        try:
            if charge_only:
                prediction = np.asarray(
                    model.charge_prediction(coefficients), dtype=np.float64
                )
            else:
                complete_prediction = model.prediction(coefficients)
                prediction = np.asarray(
                    complete_prediction[0], dtype=np.float64
                )
        finally:
            prediction_wall_s += float(time.perf_counter() - wall0)
        if np.any(~np.isfinite(prediction)):
            raise FloatingPointError("non-finite coherent FE charge prediction")
        if charge_only:
            data_nll, score = _charge_value_and_score(model, prediction)
            profiled_t0 = float(0.0 if t0 is None else t0)
        elif profile_t0 and profile_event_time:
            data_nll, profiled_t0 = _profile_charge_time_t0(
                model,
                coefficients,
                seed_t0=float(0.0 if t0 is None else t0),
                bounds=t0_bounds,
                coarse_step_ns=float(t0_profile_coarse_step_ns),
                refine_levels=int(t0_profile_refine_levels),
                global_points=int(t0_profile_global_points),
                seed_half_width_ns=float(t0_profile_seed_half_width_ns),
            )
            score = None
        else:
            profiled_t0 = float(
                (0.0 if t0 is None else t0)
                if fixed_t0 is None else fixed_t0
            )
            data_nll = float(
                model.data_nll(coefficients, t0=profiled_t0)
            )
            score = None
        posterior = data_nll + 0.5 * float(coefficients @ coefficients)
        if not (
            math.isfinite(data_nll)
            and math.isfinite(posterior)
            and (score is None or np.all(np.isfinite(score)))
        ):
            raise FloatingPointError("non-finite coherent FE data posterior")
        return prediction, data_nll, posterior, score, profiled_t0

    def physical_state(
        coefficients, *, profile_event_time=False, fixed_t0=None
    ):
        """Return one optical state, or ``None`` outside the FE domain."""
        nonlocal invalid_physical_evaluations
        try:
            return state(
                coefficients,
                profile_event_time=profile_event_time,
                fixed_t0=fixed_t0,
            )
        except (MCSPhysicalDomainError, FloatingPointError, OverflowError):
            invalid_physical_evaluations += 1
            return None

    def central_response(coefficients):
        nonlocal jacobian_evaluations, jacobian_field_evaluations
        centre = physical_state(
            coefficients, profile_event_time=bool(profile_t0 and not charge_only)
        )
        if centre is None:
            raise MCSPhysicalDomainError(
                "central coherent FE response state is outside the physical domain"
            )
        mu, data_nll, posterior, score, centre_t0 = centre
        complete_analytic = getattr(
            model, "charge_time_response_and_gradient", None
        )
        analytic = getattr(model, "charge_prediction_and_jacobian", None)
        if bool(
            getattr(model, "force_finite_difference_complete_response", False)
        ):
            complete_analytic = None
            analytic = None
        if (
            bool(analytic_complete_response)
            and not charge_only
            and callable(complete_analytic)
        ):
            proposal_mu, jacobian, _proposal_nll, data_gradient, _prediction = (
                complete_analytic(
                    coefficients,
                    t0=centre_t0,
                    evaluate_centre_nll=False,
                )
            )
            proposal_mu = np.asarray(proposal_mu, dtype=np.float64)
            jacobian = np.asarray(jacobian, dtype=np.float64)
            data_gradient = np.asarray(data_gradient, dtype=np.float64)
            if (
                proposal_mu.shape != mu.shape
                or jacobian.shape != (mu.size, coefficients.size)
                or data_gradient.shape != coefficients.shape
            ):
                raise RuntimeError(
                    "analytic coherent charge-time response returned the wrong shape"
                )
            exact_gradient = data_gradient + coefficients
            jacobian_evaluations += 1
            jacobian_field_evaluations += 1
            return (
                proposal_mu,
                data_nll,
                posterior,
                np.ascontiguousarray(jacobian),
                np.ascontiguousarray(exact_gradient),
                centre_t0,
            )
        if (
            bool(analytic_complete_response)
            and not charge_only
            and callable(analytic)
        ):
            proposal_mu, jacobian = analytic(coefficients)
            proposal_mu = np.asarray(proposal_mu, dtype=np.float64)
            jacobian = np.asarray(jacobian, dtype=np.float64)
            if proposal_mu.shape != mu.shape or jacobian.shape != (
                mu.size, coefficients.size
            ):
                raise RuntimeError(
                    "analytic coherent charge proposal returned the wrong shape"
                )
            _charge_nll, proposal_score = _charge_value_and_score(
                model, proposal_mu
            )
            exact_gradient = jacobian.T @ proposal_score + coefficients
            jacobian_evaluations += 1
            jacobian_field_evaluations += 1
            return (
                proposal_mu,
                data_nll,
                posterior,
                np.ascontiguousarray(jacobian),
                np.ascontiguousarray(exact_gradient),
                centre_t0,
            )
        jacobian_evaluations += 1
        jacobian_field_evaluations += 1 + 2 * coefficients.size
        jacobian = np.empty((mu.size, coefficients.size), dtype=np.float64)
        exact_gradient = np.empty(coefficients.size, dtype=np.float64)
        for mode in range(coefficients.size):
            scale = 1.0
            one_sided_jacobian = None
            one_sided_gradient = None
            while scale >= float(minimum_line_scale) - 1.0e-15:
                local_step = step_size * scale
                plus = coefficients.copy()
                minus = coefficients.copy()
                plus[mode] += local_step
                minus[mode] -= local_step
                # Envelope theorem: once the central nuisance-time score is
                # zero, holding that optimum fixed gives the derivative of the
                # profiled objective.  Reprofiling every +/- path perturbation
                # is both redundant to first order and very expensive.
                plus_state = physical_state(plus, fixed_t0=centre_t0)
                minus_state = physical_state(minus, fixed_t0=centre_t0)
                if plus_state is not None and minus_state is not None:
                    jacobian[:, mode] = (
                        plus_state[0] - minus_state[0]
                    ) / (2.0 * local_step)
                    exact_gradient[mode] = (
                        plus_state[2] - minus_state[2]
                    ) / (2.0 * local_step)
                    break
                if plus_state is not None:
                    one_sided_jacobian = (
                        plus_state[0] - mu
                    ) / local_step
                    one_sided_gradient = (
                        plus_state[2] - posterior
                    ) / local_step
                elif minus_state is not None:
                    one_sided_jacobian = (
                        mu - minus_state[0]
                    ) / local_step
                    one_sided_gradient = (
                        posterior - minus_state[2]
                    ) / local_step
                scale *= 0.5
            else:
                if one_sided_jacobian is None:
                    raise MCSPhysicalDomainError(
                        "coherent FE response derivative has no physical support"
                    )
                jacobian[:, mode] = one_sided_jacobian
                exact_gradient[mode] = one_sided_gradient
        if charge_only:
            exact_gradient = jacobian.T @ score + coefficients
        return mu, data_nll, posterior, jacobian, exact_gradient, centre_t0

    mu = np.empty(0, dtype=np.float64)
    charge = posterior = math.inf
    score = np.empty(0, dtype=np.float64)
    jacobian = np.empty((0, u.size), dtype=np.float64)
    gradient = np.zeros_like(u)
    information = np.eye(u.size, dtype=np.float64)
    covariance = information.copy()
    completed = 0
    converged = False
    inverse_hessian = None
    previous_u = None
    previous_gradient = None
    returned_state = None
    last_complete_response = None
    if int(max_iterations) > 0:
        trust = float(trust_max_component)
        if not math.isfinite(trust) or trust <= 0.0:
            raise ValueError("trust_max_component must be positive and finite")
        for iteration in range(int(max_iterations)):
            (
                local_mu,
                local_charge,
                local_posterior,
                local_jacobian,
                local_gradient,
                local_t0,
            ) = central_response(u)
            returned_state = (
                local_mu,
                local_charge,
                local_posterior,
                local_jacobian,
                local_gradient,
                local_t0,
            )
            last_complete_response = (u.copy(), returned_state)
            completed = iteration + 1
            if (
                not bool(analytic_complete_response)
                and float(np.max(np.abs(local_gradient)))
                <= float(gradient_tolerance)
            ):
                break
            local_information = np.eye(u.size, dtype=np.float64) + (
                local_jacobian.T
                @ (local_jacobian / np.maximum(local_mu, 1.0e-12)[:, None])
            )
            local_inverse, _local_eigenvalues = _symmetric_psd_inverse(
                local_information
            )
            if charge_only or bool(analytic_complete_response):
                proposal_inverse = local_inverse
            else:
                if inverse_hessian is None:
                    inverse_hessian = local_inverse
                elif previous_u is not None and previous_gradient is not None:
                    displacement = np.asarray(u - previous_u, dtype=np.float64)
                    gradient_change = np.asarray(
                        local_gradient - previous_gradient, dtype=np.float64
                    )
                    curvature = float(displacement @ gradient_change)
                    scale_test = float(
                        np.linalg.norm(displacement)
                        * np.linalg.norm(gradient_change)
                    )
                    if (
                        math.isfinite(curvature)
                        and curvature > max(1.0e-10, 1.0e-8 * scale_test)
                    ):
                        rho = 1.0 / curvature
                        identity = np.eye(u.size, dtype=np.float64)
                        left = identity - rho * np.outer(
                            displacement, gradient_change
                        )
                        inverse_hessian = (
                            left
                            @ inverse_hessian
                            @ left.T
                            + rho * np.outer(displacement, displacement)
                        )
                        inverse_hessian = 0.5 * (
                            inverse_hessian + inverse_hessian.T
                        )
                    else:
                        # A failed curvature condition means the finite-step
                        # timing surface changed basin; restart from the local
                        # positive charge-Fisher metric instead of making an
                        # indefinite quasi-Newton move.
                        inverse_hessian = local_inverse
                proposal_inverse = inverse_hessian
            step = -(proposal_inverse @ local_gradient)
            maximum = float(np.max(np.abs(step))) if step.size else 0.0
            if maximum > trust:
                step *= trust / maximum
            directional = float(local_gradient @ step)
            if not math.isfinite(directional) or directional >= 0.0:
                step = -np.asarray(local_gradient, dtype=np.float64)
                maximum = float(np.max(np.abs(step))) if step.size else 0.0
                if maximum > trust:
                    step *= trust / maximum
                directional = float(local_gradient @ step)

            accepted = False
            scale = 1.0
            while scale >= float(minimum_line_scale) - 1.0e-15:
                candidate = np.ascontiguousarray(u + scale * step)
                candidate_state = physical_state(
                    candidate,
                    profile_event_time=bool(profile_t0 and not charge_only),
                )
                if candidate_state is not None:
                    candidate_posterior = float(candidate_state[2])
                    armijo = (
                        float(local_posterior)
                        + 1.0e-4 * scale * directional
                    )
                    acceptable = (
                        candidate_posterior
                        < float(local_posterior) - 1.0e-8
                        if bool(analytic_complete_response)
                        else candidate_posterior <= armijo
                    )
                    if acceptable:
                        previous_u = u.copy()
                        previous_gradient = local_gradient.copy()
                        u = candidate
                        returned_state = (
                            candidate_state[0],
                            candidate_state[1],
                            candidate_state[2],
                            None,
                            None,
                            candidate_state[4],
                        )
                        accepted = True
                        break
                scale *= 0.5
            if not accepted:
                break
            if float(np.max(np.abs(scale * step))) <= 1.0e-8:
                break

    # A full Fisher summary is required for the accepted final state, but not
    # for the many disposable range-screening states.  Those states have
    # already been evaluated with the exact nonlinear charge-time likelihood;
    # repeating a 2*Nmode central stencil solely to populate unused diagnostic
    # arrays was the dominant production cost.  The fast branch returns that
    # exact cached state and explicit NaN Fisher fields so callers cannot
    # mistake an omitted diagnostic for a computed covariance.
    if bool(finalize_fisher_summary):
        # The final reported covariance and convergence certificate always use
        # a complete charge-time response.  Production support-tracked FALI has
        # an exact analytic optical response and a full data-NLL gradient; old
        # or test models without that interface retain the central reference.
        proposal_mode = bool(analytic_complete_response)
        complete_analytic = getattr(
            model, "charge_time_response_and_gradient", None
        )
        if bool(
            getattr(model, "force_finite_difference_complete_response", False)
        ):
            complete_analytic = None
        analytic_complete_response = bool(
            proposal_mode and callable(complete_analytic)
        )
        if (
            last_complete_response is not None
            and np.array_equal(last_complete_response[0], u)
            # If a caller requested the analytic complete response but its
            # model does not implement that interface, finalization correctly
            # falls back to the central reference and must not reuse a summary
            # produced under the unavailable proposal mode.
            and bool(analytic_complete_response) == bool(proposal_mode)
        ):
            (
                mu,
                charge,
                posterior,
                jacobian,
                gradient,
                profiled_t0,
            ) = last_complete_response[1]
        else:
            (
                mu,
                charge,
                posterior,
                jacobian,
                gradient,
                profiled_t0,
            ) = central_response(u)
        analytic_complete_response = proposal_mode
        information = np.eye(u.size, dtype=np.float64) + jacobian.T @ (
            jacobian / np.maximum(mu, 1.0e-12)[:, None]
        )
        information = 0.5 * (information + information.T)
        covariance, _eigenvalues = _symmetric_psd_inverse(information)
        converged = bool(
            float(np.max(np.abs(gradient))) <= float(gradient_tolerance)
        )
    else:
        if returned_state is None:
            exact_state = physical_state(
                u,
                profile_event_time=bool(profile_t0 and not charge_only),
            )
            if exact_state is None:
                raise MCSPhysicalDomainError(
                    "returned coherent FE state is outside the physical domain"
                )
            returned_state = (
                exact_state[0],
                exact_state[1],
                exact_state[2],
                None,
                None,
                exact_state[4],
            )
        mu, charge, posterior, _jacobian, _gradient, profiled_t0 = returned_state
        jacobian = np.full((mu.size, u.size), np.nan, dtype=np.float64)
        gradient = np.full(u.size, np.nan, dtype=np.float64)
        information = np.full((u.size, u.size), np.nan, dtype=np.float64)
        covariance = np.full((u.size, u.size), np.nan, dtype=np.float64)
        converged = False
    return _CentralResponseLatent(
        coefficients=np.ascontiguousarray(u),
        charge_nll=float(charge),
        posterior_nll=float(posterior),
        information=np.ascontiguousarray(information),
        covariance=np.ascontiguousarray(covariance),
        charge_jacobian=np.ascontiguousarray(jacobian),
        prediction=np.ascontiguousarray(mu),
        final_gradient=np.ascontiguousarray(gradient),
        iterations=int(completed),
        objective_evaluations=int(evaluations),
        jacobian_evaluations=int(jacobian_evaluations),
        jacobian_field_evaluations=int(jacobian_field_evaluations),
        invalid_physical_evaluations=int(invalid_physical_evaluations),
        converged=bool(converged),
        prediction_wall_s=float(prediction_wall_s),
        profiled_t0_ns=float(profiled_t0),
        fisher_summary_complete=bool(finalize_fisher_summary),
    )


def _symmetric_psd_inverse(matrix, *, relative_floor=1.0e-10, absolute_floor=1.0e-10):
    sym = 0.5 * (
        np.asarray(matrix, dtype=np.float64)
        + np.asarray(matrix, dtype=np.float64).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    largest = max(float(np.max(eigenvalues)), 0.0)
    floor = max(float(absolute_floor), float(relative_floor) * largest)
    clipped = np.maximum(eigenvalues, floor)
    inverse = (eigenvectors / clipped[None, :]) @ eigenvectors.T
    return np.ascontiguousarray(0.5 * (inverse + inverse.T)), clipped


def _stable_cholesky(matrix):
    sym = 0.5 * (
        np.asarray(matrix, dtype=np.float64)
        + np.asarray(matrix, dtype=np.float64).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    floor = max(1.0e-12, 1.0e-10 * max(float(np.max(eigenvalues)), 1.0))
    repaired = (eigenvectors * np.maximum(eigenvalues, floor)[None, :]) @ eigenvectors.T
    return np.linalg.cholesky(0.5 * (repaired + repaired.T))


def run_joint_laplace_cubature(
    evaluator,
    theta,
    straggling,
    *,
    kinetic_energy_bounds_mev,
    initial_energy_mev,
    initial_z_range,
    n_path_modes,
    initial_path_coefficients=None,
    config=JointLaplaceCubatureConfig(),
    random_seed=41873,
    t0_bounds=None,
):
    """Return a deterministic approximation to the exact continuous posterior."""
    config.validate()
    base_theta = np.asarray(theta, dtype=np.float64).reshape(7).copy()
    dimension = int(n_path_modes)
    if dimension < 1:
        raise ValueError("n_path_modes must be positive")
    low, high = map(float, kinetic_energy_bounds_mev)
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError("invalid kinetic-energy bounds")
    energy = float(np.clip(float(initial_energy_mev), low, high))
    z_value = float(initial_z_range)
    coefficients = (
        np.zeros(dimension, dtype=np.float64)
        if initial_path_coefficients is None
        else np.asarray(initial_path_coefficients, dtype=np.float64).reshape(dimension).copy()
    )
    energy_step = float(config.energy_step_mev)
    z_step = float(config.z_range_step)
    scales = np.ones(dimension + 2, dtype=np.float64)
    scales[0] = energy_step
    scales[1] = z_step
    model_build_requests = 0
    prediction_evaluations = 0
    invalid_evaluations = 0
    model_wall_s = 0.0
    model_build_wall_rows = []
    prediction_wall_s = 0.0
    accepted_steps = 0
    iteration_rows = []
    charge_only = bool(getattr(evaluator, "charge_only", True))

    def realized_range(k0, z):
        return float(straggling.realized_range_mm(float(k0), float(z)))

    def model_at(k0, z):
        nonlocal model_build_requests, invalid_evaluations, model_wall_s
        length = realized_range(k0, z)
        if not math.isfinite(length) or length <= 0.0:
            invalid_evaluations += 1
            return None
        candidate = base_theta.copy()
        candidate[5] = length
        model_build_requests += 1
        count_before = int(evaluator.optical_model_build_count)
        wall0 = time.perf_counter()
        model = evaluator.model(candidate, initial_kinetic_energy_mev=float(k0))
        elapsed = float(time.perf_counter() - wall0)
        model_wall_s += elapsed
        if int(evaluator.optical_model_build_count) > count_before:
            model_build_wall_rows.append({
                "energy_mev": float(k0),
                "z_range": float(z),
                "wall_s": elapsed,
            })
        if model is None:
            invalid_evaluations += 1
        return model

    def evaluated_state(
        k0, z, path, *, profile_event_time=True, fixed_t0=None
    ):
        nonlocal prediction_evaluations, invalid_evaluations, prediction_wall_s
        if not (low <= float(k0) <= high):
            invalid_evaluations += 1
            return None
        model = model_at(k0, z)
        if model is None:
            return None
        try:
            prediction_evaluations += 1
            wall0 = time.perf_counter()
            if charge_only:
                mu = np.asarray(
                    model.charge_prediction(path), dtype=np.float64
                )
                data_nll = float(model.charge_data_nll(path))
                profiled_t0 = float(base_theta[6])
            else:
                complete_prediction = model.prediction(path)
                mu = np.asarray(
                    complete_prediction[0], dtype=np.float64
                )
                if bool(config.profile_t0) and bool(profile_event_time):
                    data_nll, profiled_t0 = _profile_charge_time_t0(
                        model,
                        path,
                        seed_t0=float(base_theta[6]),
                        bounds=t0_bounds,
                        coarse_step_ns=float(config.t0_profile_coarse_step_ns),
                        refine_levels=int(config.t0_profile_refine_levels),
                        global_points=int(config.t0_profile_global_points),
                        seed_half_width_ns=float(
                            config.t0_profile_seed_half_width_ns
                        ),
                    )
                else:
                    profiled_t0 = float(
                        base_theta[6] if fixed_t0 is None else fixed_t0
                    )
                    data_nll = float(
                        model.data_nll(path, t0=profiled_t0)
                    )
            prediction_wall_s += float(time.perf_counter() - wall0)
        except Exception:
            invalid_evaluations += 1
            return None
        posterior = (
            data_nll
            + 0.5 * float(z) ** 2
            + 0.5 * float(path @ path)
        )
        if (
            np.any(~np.isfinite(mu))
            or not math.isfinite(data_nll)
            or not math.isfinite(posterior)
        ):
            invalid_evaluations += 1
            return None
        return mu, data_nll, posterior, model, profiled_t0

    def exact_state(k0, z, path):
        state = evaluated_state(k0, z, path)
        if state is None:
            return math.inf, math.inf, None, math.nan
        return float(state[1]), float(state[2]), state[3], float(state[4])

    def exact_state_from_profiled_latent(z, latent):
        """Recover the exact central state without re-running its t0 profile.

        A finalized latent result already contains the configured data NLL at
        its profiled event time. Repeating ``exact_state`` at that same path
        performs an identical detector likelihood scan. Build the outer
        posterior in the original operation order so downstream Armijo
        comparisons retain the historical floating-point value as well.
        """
        path = np.asarray(latent.coefficients, dtype=np.float64)
        data_nll = float(latent.charge_nll)
        posterior = (
            data_nll
            + 0.5 * float(z) ** 2
            + 0.5 * float(path @ path)
        )
        return data_nll, posterior, float(latent.profiled_t0_ns)

    def scaled_global_column_and_gradient(
        k0, z, path, which, *, fixed_t0
    ):
        if which == 0:
            plus = evaluated_state(
                k0 + energy_step,
                z,
                path,
                profile_event_time=False,
                fixed_t0=fixed_t0,
            )
            minus = evaluated_state(
                k0 - energy_step,
                z,
                path,
                profile_event_time=False,
                fixed_t0=fixed_t0,
            )
        else:
            plus = evaluated_state(
                k0,
                z + z_step,
                path,
                profile_event_time=False,
                fixed_t0=fixed_t0,
            )
            minus = evaluated_state(
                k0,
                z - z_step,
                path,
                profile_event_time=False,
                fixed_t0=fixed_t0,
            )
        if plus is not None and minus is not None:
            return (
                0.5 * (plus[0] - minus[0]),
                0.5 * float(plus[2] - minus[2]),
                "central",
            )
        center = evaluated_state(
            k0,
            z,
            path,
            profile_event_time=False,
            fixed_t0=fixed_t0,
        )
        if center is None:
            raise RuntimeError("joint response derivative lost its central state")
        if plus is not None:
            return (
                plus[0] - center[0],
                float(plus[2] - center[2]),
                "forward",
            )
        if minus is not None:
            return (
                center[0] - minus[0],
                float(center[2] - minus[2]),
                "backward",
            )
        raise RuntimeError("joint response derivative has no physical support")

    def local_system(k0, z, latent):
        path = np.asarray(latent.coefficients, dtype=np.float64)
        mu = np.asarray(latent.prediction, dtype=np.float64)
        ju = np.asarray(latent.charge_jacobian, dtype=np.float64)
        jk, gradient_k, scheme_k = scaled_global_column_and_gradient(
            k0, z, path, 0, fixed_t0=float(latent.profiled_t0_ns)
        )
        jz, gradient_z, scheme_z = scaled_global_column_and_gradient(
            k0, z, path, 1, fixed_t0=float(latent.profiled_t0_ns)
        )
        jacobian = np.column_stack((jk, jz, ju))
        gradient = np.concatenate((
            np.asarray((gradient_k, gradient_z), dtype=np.float64),
            np.asarray(latent.final_gradient, dtype=np.float64),
        ))
        information = jacobian.T @ (
            jacobian / np.maximum(mu, 1.0e-12)[:, None]
        )
        information[1, 1] += z_step * z_step
        information[2:, 2:] += np.eye(dimension, dtype=np.float64)
        information = 0.5 * (information + information.T)
        inverse, eigenvalues = _symmetric_psd_inverse(information)
        return gradient, information, inverse, eigenvalues, (scheme_k, scheme_z)

    # A straight-track length can be a severely biased energy proxy in exactly
    # the MCS events this continuation is intended to repair.  Screen a small,
    # deterministic scale family on the physical mean-range manifold (z_R=0)
    # before the local solve.  This is data-driven and energy-agnostic: it uses
    # neither generated energy nor event truth, and it retains the original
    # straight-derived (K0,z_R) state as a guard candidate.
    seed_rows = []
    seed_candidates = [(float(energy), float(z_value), coefficients.copy(), "straight_range")]
    zero_path = np.zeros(dimension, dtype=np.float64)
    for multiplier in tuple(config.seed_energy_multipliers):
        candidate_energy = float(np.clip(energy * float(multiplier), low, high))
        seed_candidates.append((candidate_energy, 0.0, zero_path.copy(), "mean_range_scale"))
    unique_candidates = []
    seen_candidates = set()
    for candidate in seed_candidates:
        key = (round(float(candidate[0]), 10), round(float(candidate[1]), 10))
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        unique_candidates.append(candidate)

    selected_seed = None
    for candidate_energy, candidate_z, candidate_path, origin in unique_candidates:
        candidate_model = model_at(candidate_energy, candidate_z)
        if candidate_model is None:
            seed_rows.append({
                "energy_mev": float(candidate_energy),
                "z_range": float(candidate_z),
                "origin": str(origin),
                "valid": False,
                "posterior_nll": math.inf,
            })
            continue
        try:
            candidate_latent = _solve_latent_with_central_response(
                candidate_model,
                initial_coefficients=candidate_path,
                response_fd_step=float(config.response_fd_step),
                max_iterations=int(config.seed_screen_latent_iterations),
                gradient_tolerance=float(config.gradient_tolerance),
                trust_max_component=float(config.path_trust),
                minimum_line_scale=float(config.minimum_line_scale),
                t0=float(base_theta[6]),
                profile_t0=bool(config.profile_t0 and not charge_only),
                t0_bounds=t0_bounds,
                t0_profile_coarse_step_ns=float(config.t0_profile_coarse_step_ns),
                t0_profile_refine_levels=int(config.t0_profile_refine_levels),
                t0_profile_global_points=int(config.t0_profile_global_points),
                t0_profile_seed_half_width_ns=float(
                    config.t0_profile_seed_half_width_ns
                ),
                # Seed candidates are ranked only by their exact nonlinear
                # posterior. Their response Jacobian, Fisher matrix and
                # covariance are never consumed; omit that duplicate optical
                # stencil and retain explicit NaN diagnostics instead.
                finalize_fisher_summary=False,
            )
        except (MCSPhysicalDomainError, FloatingPointError, OverflowError):
            invalid_evaluations += 1
            seed_rows.append({
                "energy_mev": float(candidate_energy),
                "z_range": float(candidate_z),
                "origin": str(origin),
                "valid": False,
                "posterior_nll": math.inf,
            })
            continue
        prediction_evaluations += int(candidate_latent.objective_evaluations)
        prediction_wall_s += float(candidate_latent.prediction_wall_s)
        candidate_posterior = (
            float(candidate_latent.posterior_nll)
            + 0.5 * float(candidate_z) ** 2
        )
        row = {
            "energy_mev": float(candidate_energy),
            "z_range": float(candidate_z),
            "origin": str(origin),
            "valid": True,
            "posterior_nll": float(candidate_posterior),
            "data_nll": float(candidate_latent.charge_nll),
            "path_norm": float(np.linalg.norm(candidate_latent.coefficients)),
            "latent_gradient_max_abs": float(
                np.max(np.abs(candidate_latent.final_gradient))
            ),
            "fisher_summary_complete": bool(
                candidate_latent.fisher_summary_complete
            ),
        }
        seed_rows.append(row)
        if selected_seed is None or candidate_posterior < selected_seed[0]:
            selected_seed = (
                float(candidate_posterior),
                float(candidate_energy),
                float(candidate_z),
                np.asarray(candidate_latent.coefficients, dtype=np.float64).copy(),
                int(len(seed_rows) - 1),
            )
    if selected_seed is None:
        raise RuntimeError("joint Laplace seed screen lost every physical state")
    _seed_score, energy, z_value, coefficients, selected_seed_index = selected_seed

    # Profile the complete FE coordinate vector with central differences of the
    # exact nonlinear field.  This is the derivative-validation reference for
    # the analytic receiver, and is much faster in a one-thread event worker.
    reference_model = model_at(energy, z_value)
    if reference_model is None:
        raise RuntimeError("initial joint Laplace state is physically invalid")
    latent = _solve_latent_with_central_response(
        reference_model,
        initial_coefficients=coefficients,
        response_fd_step=float(config.response_fd_step),
        max_iterations=int(config.latent_iterations),
        gradient_tolerance=float(config.gradient_tolerance),
        trust_max_component=float(config.path_trust),
        minimum_line_scale=float(config.minimum_line_scale),
        t0=float(base_theta[6]),
        profile_t0=bool(config.profile_t0 and not charge_only),
        t0_bounds=t0_bounds,
        t0_profile_coarse_step_ns=float(config.t0_profile_coarse_step_ns),
        t0_profile_refine_levels=int(config.t0_profile_refine_levels),
        t0_profile_global_points=int(config.t0_profile_global_points),
        t0_profile_seed_half_width_ns=float(
            config.t0_profile_seed_half_width_ns
        ),
    )
    prediction_evaluations += int(latent.objective_evaluations)
    prediction_wall_s += float(latent.prediction_wall_s)
    latent_solve_count = 1
    latent_objective_evaluations = int(latent.objective_evaluations)
    latent_jacobian_evaluations = int(latent.jacobian_evaluations)
    latent_jacobian_field_evaluations = int(
        latent.jacobian_field_evaluations
    )
    latent_invalid_physical_evaluations = int(
        latent.invalid_physical_evaluations
    )
    latent_current = True
    joint_inverse_hessian = None
    previous_scaled_state = None
    previous_joint_gradient = None
    for iteration in range(int(config.maximum_iterations)):
        model = model_at(energy, z_value)
        if model is None:
            raise RuntimeError("initial joint Laplace state is physically invalid")
        if iteration > 0 or model is not reference_model:
            latent = _solve_latent_with_central_response(
                model,
                initial_coefficients=coefficients,
                response_fd_step=float(config.response_fd_step),
                # The preceding accepted coupled step already transports the
                # complete FE vector.  Recompute the exact nonlinear score and
                # Fisher matrix once; repeating a path-only optimizer here
                # duplicates work before the next coupled update.
                max_iterations=0,
                gradient_tolerance=float(config.gradient_tolerance),
                trust_max_component=float(config.path_trust),
                minimum_line_scale=float(config.minimum_line_scale),
                t0=float(base_theta[6]),
                profile_t0=bool(config.profile_t0 and not charge_only),
                t0_bounds=t0_bounds,
                t0_profile_coarse_step_ns=float(config.t0_profile_coarse_step_ns),
                t0_profile_refine_levels=int(config.t0_profile_refine_levels),
                t0_profile_global_points=int(config.t0_profile_global_points),
                t0_profile_seed_half_width_ns=float(
                    config.t0_profile_seed_half_width_ns
                ),
            )
            prediction_evaluations += int(latent.objective_evaluations)
            prediction_wall_s += float(latent.prediction_wall_s)
            latent_solve_count += 1
            latent_objective_evaluations += int(latent.objective_evaluations)
            latent_jacobian_evaluations += int(latent.jacobian_evaluations)
            latent_jacobian_field_evaluations += int(
                latent.jacobian_field_evaluations
            )
            latent_invalid_physical_evaluations += int(
                latent.invalid_physical_evaluations
            )
            latent_current = True
        coefficients = np.asarray(latent.coefficients, dtype=np.float64)
        base_charge, base_posterior, base_profiled_t0 = (
            exact_state_from_profiled_latent(z_value, latent)
        )
        gradient, information, inverse, eigenvalues, schemes = local_system(
            energy, z_value, latent
        )
        scaled_state = np.concatenate((
            np.asarray((energy / energy_step, z_value / z_step)),
            coefficients,
        ))
        if joint_inverse_hessian is None:
            joint_inverse_hessian = inverse
        elif (
            previous_scaled_state is not None
            and previous_joint_gradient is not None
        ):
            displacement = scaled_state - previous_scaled_state
            gradient_change = gradient - previous_joint_gradient
            curvature = float(displacement @ gradient_change)
            scale_test = float(
                np.linalg.norm(displacement)
                * np.linalg.norm(gradient_change)
            )
            if (
                math.isfinite(curvature)
                and curvature > max(1.0e-10, 1.0e-8 * scale_test)
            ):
                rho = 1.0 / curvature
                identity = np.eye(dimension + 2, dtype=np.float64)
                left = identity - rho * np.outer(
                    displacement, gradient_change
                )
                joint_inverse_hessian = (
                    left
                    @ joint_inverse_hessian
                    @ left.T
                    + rho * np.outer(displacement, displacement)
                )
                joint_inverse_hessian = 0.5 * (
                    joint_inverse_hessian + joint_inverse_hessian.T
                )
            else:
                joint_inverse_hessian = inverse
        raw = -(joint_inverse_hessian @ gradient)
        physical = raw * scales
        maximum_ratio = max(
            abs(float(physical[0])) / float(config.energy_trust_mev),
            abs(float(physical[1])) / float(config.z_range_trust),
            (
                float(np.max(np.abs(physical[2:]))) / float(config.path_trust)
                if dimension else 0.0
            ),
            1.0,
        )
        physical /= maximum_ratio
        raw = physical / scales
        directional = float(gradient @ raw)
        accepted = False
        accepted_scale = 0.0
        trial_scale = 1.0
        best = (
            energy,
            z_value,
            coefficients,
            base_charge,
            base_posterior,
            base_profiled_t0,
        )
        while trial_scale >= float(config.minimum_line_scale) - 1.0e-15:
            proposed_energy = float(energy + trial_scale * physical[0])
            proposed_z = float(z_value + trial_scale * physical[1])
            proposed_path = np.ascontiguousarray(
                coefficients + trial_scale * physical[2:]
            )
            charge, posterior, _model, proposed_t0 = exact_state(
                proposed_energy, proposed_z, proposed_path
            )
            armijo = base_posterior + 1.0e-4 * trial_scale * directional
            if math.isfinite(posterior) and posterior <= armijo:
                accepted = True
                accepted_scale = float(trial_scale)
                best = (
                    proposed_energy,
                    proposed_z,
                    proposed_path,
                    charge,
                    posterior,
                    proposed_t0,
                )
                break
            trial_scale *= 0.5
        iteration_rows.append({
            "iteration": int(iteration),
            "energy_before_mev": float(energy),
            "z_range_before": float(z_value),
            "charge_nll_before": float(base_charge),
            "posterior_nll_before": float(base_posterior),
            "gradient_max_abs_scaled": float(np.max(np.abs(gradient))),
            "proposed_energy_delta_mev": float(physical[0]),
            "proposed_z_range_delta": float(physical[1]),
            "proposed_path_max_abs": float(np.max(np.abs(physical[2:]))),
            "accepted": bool(accepted),
            "accepted_scale": float(accepted_scale),
            "energy_derivative_scheme": schemes[0],
            "z_range_derivative_scheme": schemes[1],
            "information_min_eigenvalue": float(np.min(eigenvalues)),
        })
        if not accepted:
            break
        previous_scaled_state = scaled_state.copy()
        previous_joint_gradient = gradient.copy()
        energy, z_value, coefficients, _charge, _posterior, _profiled_t0 = best
        latent_current = False
        accepted_steps += 1
        if max(abs(accepted_scale * physical[0]) / energy_step,
               abs(accepted_scale * physical[1]) / z_step,
               float(np.max(np.abs(accepted_scale * physical[2:])))) < 0.05:
            break

    final_model = model_at(energy, z_value)
    if final_model is None:
        raise RuntimeError("accepted joint Laplace mode became invalid")
    if not latent_current:
        latent = _solve_latent_with_central_response(
            final_model,
            initial_coefficients=coefficients,
            response_fd_step=float(config.response_fd_step),
            max_iterations=int(config.latent_iterations),
            gradient_tolerance=float(config.gradient_tolerance),
            trust_max_component=float(config.path_trust),
            minimum_line_scale=float(config.minimum_line_scale),
            t0=float(base_theta[6]),
            profile_t0=bool(config.profile_t0 and not charge_only),
            t0_bounds=t0_bounds,
            t0_profile_coarse_step_ns=float(config.t0_profile_coarse_step_ns),
            t0_profile_refine_levels=int(config.t0_profile_refine_levels),
            t0_profile_global_points=int(config.t0_profile_global_points),
            t0_profile_seed_half_width_ns=float(
                config.t0_profile_seed_half_width_ns
            ),
        )
        prediction_evaluations += int(latent.objective_evaluations)
        prediction_wall_s += float(latent.prediction_wall_s)
        latent_solve_count += 1
        latent_objective_evaluations += int(latent.objective_evaluations)
        latent_jacobian_evaluations += int(latent.jacobian_evaluations)
        latent_jacobian_field_evaluations += int(
            latent.jacobian_field_evaluations
        )
        latent_invalid_physical_evaluations += int(
            latent.invalid_physical_evaluations
        )
    coefficients = np.asarray(latent.coefficients, dtype=np.float64)
    mode_charge, mode_posterior, mode_t0 = exact_state_from_profiled_latent(
        z_value, latent
    )
    gradient, information_scaled, covariance_scaled, eigenvalues, schemes = local_system(
        energy, z_value, latent
    )
    mode_gradient_max_abs = float(np.max(np.abs(gradient)))
    convergence_certified = bool(
        latent.converged
        and mode_gradient_max_abs <= float(config.gradient_tolerance)
    )
    covariance = (scales[:, None] * covariance_scaled) * scales[None, :]
    covariance = 0.5 * (covariance + covariance.T)
    global_covariance = covariance[:2, :2]
    global_inverse, _ = _symmetric_psd_inverse(global_covariance)
    conditional_slope = covariance[2:, :2] @ global_inverse
    conditional_covariance = (
        covariance[2:, 2:]
        - conditional_slope @ covariance[:2, 2:]
    )
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.T
    )
    global_cholesky = _stable_cholesky(global_covariance)
    gh_nodes, gh_weights = hermgauss(int(config.cubature_order))
    sample_energy = []
    sample_z = []
    sample_range = []
    sample_path = []
    sample_loglike = []
    sample_logweight = []
    sample_joint_posterior = []
    sample_t0 = []
    for i, node_first in enumerate(gh_nodes):
        for j, node_second in enumerate(gh_nodes):
            standard = np.asarray((node_first, node_second), dtype=np.float64)
            delta = math.sqrt(2.0) * (global_cholesky @ standard)
            candidate_energy = float(energy + delta[0])
            candidate_z = float(z_value + delta[1])
            candidate_path = np.ascontiguousarray(
                coefficients + conditional_slope @ delta
            )
            charge, posterior_nll, _model, candidate_t0 = exact_state(
                candidate_energy, candidate_z, candidate_path
            )
            if not math.isfinite(posterior_nll):
                continue
            proposal_quadratic = float(standard @ standard)
            base_weight = float(gh_weights[i] * gh_weights[j] / math.pi)
            logweight = math.log(base_weight) - posterior_nll + proposal_quadratic
            sample_energy.append(candidate_energy)
            sample_z.append(candidate_z)
            sample_range.append(realized_range(candidate_energy, candidate_z))
            sample_path.append(candidate_path)
            sample_loglike.append(-charge)
            sample_logweight.append(logweight)
            sample_joint_posterior.append(posterior_nll)
            sample_t0.append(candidate_t0)
    if not sample_energy:
        raise RuntimeError("joint Laplace cubature lost every physical node")
    logweight = np.asarray(sample_logweight, dtype=np.float64)
    logweight -= float(np.max(logweight))
    weights = np.exp(np.clip(logweight, -745.0, 0.0))
    weights /= float(np.sum(weights))
    energies = np.asarray(sample_energy, dtype=np.float64)
    z_values = np.asarray(sample_z, dtype=np.float64)
    ranges = np.asarray(sample_range, dtype=np.float64)
    paths = np.asarray(sample_path, dtype=np.float64)
    loglikes = np.asarray(sample_loglike, dtype=np.float64)
    t0_values = np.asarray(sample_t0, dtype=np.float64)
    posterior_t0_mean = float(weights @ t0_values)
    coefficient_mean = np.asarray(weights @ paths, dtype=np.float64)
    path_centered = paths - coefficient_mean[None, :]
    coefficient_covariance = (
        conditional_covariance
        + (path_centered * weights[:, None]).T @ path_centered
    )
    coefficient_covariance = 0.5 * (
        coefficient_covariance + coefficient_covariance.T
    )
    map_index = int(np.argmin(np.asarray(sample_joint_posterior, dtype=np.float64)))
    posterior = JointEnergyRangeSMCResult(
        kinetic_energy_mev=np.ascontiguousarray(energies),
        z_range=np.ascontiguousarray(z_values),
        realized_range_mm=np.ascontiguousarray(ranges),
        coefficients=np.ascontiguousarray(paths),
        log_likelihood=np.ascontiguousarray(loglikes),
        weights=np.ascontiguousarray(weights),
        stages=(),
        likelihood_evaluations=int(prediction_evaluations),
        invalid_likelihood_evaluations=int(invalid_evaluations),
        log_evidence=math.nan,
        random_seed=int(random_seed),
        kinetic_energy_bounds_mev=(low, high),
        initialization_beta=math.nan,
        initialization_ess=math.nan,
        posterior_trajectory_sweeps=0,
    )
    return JointLaplaceCubatureResult(
        posterior=posterior,
        coefficients_mean=np.ascontiguousarray(coefficient_mean),
        coefficients_covariance=np.ascontiguousarray(coefficient_covariance),
        mode_energy_mev=float(energy),
        mode_z_range=float(z_value),
        mode_range_mm=float(realized_range(energy, z_value)),
        mode_t0_ns=float(mode_t0),
        posterior_t0_mean_ns=float(posterior_t0_mean),
        mode_coefficients=np.ascontiguousarray(coefficients),
        mode_charge_nll=float(mode_charge),
        mode_posterior_nll=float(mode_posterior),
        map_sample_index=int(map_index),
        map_charge_nll=float(-loglikes[map_index]),
        diagnostics={
            "implementation": "continuous_noncentered_energy_range_fe_laplace_gh_v2",
            "inference_method": "laplace_cubature",
            "same_physical_target_as_reference_smc": True,
            "charge_only": bool(charge_only),
            "timing_used": bool(not charge_only),
            "t0_profiled": bool(config.profile_t0 and not charge_only),
            "mode_t0_ns": float(mode_t0),
            "posterior_t0_mean_ns": float(posterior_t0_mean),
            "response_jacobian_likelihood": (
                "configured_charge_score"
                if charge_only
                else "configured_charge_time_scalar_central_difference"
            ),
            "deterministic": True,
            "uses_event_truth": False,
            "uses_empirical_mcs_scale": False,
            "uses_wcsim_range_width": False,
            "uses_discrete_range_grid": False,
            "path_modes": int(dimension),
            "seed_policy": "straight_guard_plus_mean_range_energy_scales",
            "seed_energy_multipliers": [
                float(value) for value in config.seed_energy_multipliers
            ],
            "seed_screen_latent_iterations": int(
                config.seed_screen_latent_iterations
            ),
            "seed_candidates": seed_rows,
            "selected_seed_index": int(selected_seed_index),
            "cubature_order": int(config.cubature_order),
            "cubature_nodes_retained": int(energies.size),
            "accepted_mode_steps": int(accepted_steps),
            "response_jacobian_evaluations": int(
                latent_jacobian_evaluations
            ),
            "response_jacobian_field_evaluations": int(
                latent_jacobian_field_evaluations
            ),
            "response_jacobian_method": "central_exact_optical_field",
            "response_jacobian_step": float(config.response_fd_step),
            "optical_model_wall_s": float(model_wall_s),
            "optical_model_build_wall_rows": model_build_wall_rows,
            "nonlinear_prediction_wall_s": float(prediction_wall_s),
            "response_jacobian_role": "proposal_preconditioner_only",
            "objective_gradient_likelihood": (
                "configured_charge"
                if charge_only else "configured_charge_time"
            ),
            "latent_solve_count": int(latent_solve_count),
            "latent_objective_evaluations": int(
                latent_objective_evaluations
            ),
            "latent_invalid_physical_evaluations": int(
                latent_invalid_physical_evaluations
            ),
            "final_latent_gradient_max_abs": float(
                np.max(np.abs(latent.final_gradient))
            ),
            "final_latent_converged": bool(latent.converged),
            "convergence_certified": bool(convergence_certified),
            "mode_iterations": iteration_rows,
            "mode_gradient_max_abs_scaled": mode_gradient_max_abs,
            "mode_information_min_eigenvalue": float(np.min(eigenvalues)),
            "energy_derivative_scheme": schemes[0],
            "z_range_derivative_scheme": schemes[1],
            "model_build_requests": int(model_build_requests),
            "prediction_evaluations": int(prediction_evaluations),
            "invalid_evaluations": int(invalid_evaluations),
            "mode_energy_mev": float(energy),
            "mode_z_range": float(z_value),
            "mode_range_mm": float(realized_range(energy, z_value)),
            "mode_charge_nll": float(mode_charge),
            "mode_posterior_nll": float(mode_posterior),
            "global_covariance_energy_z": global_covariance.tolist(),
        },
    )


__all__ = [
    "JointLaplaceCubatureConfig",
    "JointLaplaceCubatureResult",
    "run_joint_laplace_cubature",
]
