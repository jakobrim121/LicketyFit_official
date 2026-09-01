"""Physics-reference coherent MCS continuation for contained tracks.

This stage starts from the accepted straight charge--time fit, then jointly
profiles the physical onset line and a conditional Fermi--Eyges path using a
charge-only forward model.  Charge-only is intentional: first-photoelectron
timing must not constrain the path until every enabled source component carries
source-resolved curved timing nodes.

The straight solution is an initializer, not a prior.  The only latent prior is
the standard-normal representation of the analytic Fermi--Eyges covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np

from .mcs_coupled_schur import CoupledCoherentEvaluator
from .mcs_coherent_objective import FixedTrackCoherentMCSObjective
from .mcs_curved_path import MCSPhysicalDomainError, build_arclength_fe_path
from .mcs_fast12_continuation import _thread_context
from .mcs_latent_profile import solve_latent_charge_map
from .mcs_response_profile import (
    solve_reduced_response_with_authoritative_correction,
)
from .mcs_reference_profile import (
    DEFAULT_GLOBAL_SCALES,
    exact_posterior_latent_coordinate_poll,
    optimize_alternating_exact_joint_map,
    optimize_exact_posterior_latent_cobyqa,
    optimize_reprofiled_fisher_laplace,
)
from .track_parameterization import (
    TangentDirectionChart,
    attach_direction_components,
    reanchor_values,
)


_COORDINATE_NAMES = ("x0", "y0", "z0", "dir_u", "dir_v", "length", "t0")
# Event 0 is the limiting supplied geometry: against a 321-point convergence
# reference, 161 points misses the normalized charge-field tolerance slightly
# (1.10e-3), while 241 points closes at 3.32e-4.  The derivative-free reference
# therefore fails closed below 241 rather than silently fitting quadrature error.
MINIMUM_PHYSICS_REFERENCE_GRID_POINTS = 241
# G61 was rejected on the supplied 82-event ensemble because it produced
# multi-nat posterior regressions after the same G241 correction.  G81 is the
# lowest validated proposal grid; it never replaces the authoritative model.
MINIMUM_PHYSICS_RESPONSE_GRID_POINTS = 81


@dataclass
class PhysicsCoherentResult:
    initial_values: dict
    updated_values: dict
    updated_chart: TangentDirectionChart
    coefficients_mean: np.ndarray
    coefficients_covariance: np.ndarray
    charge_nll: float
    posterior_nll: float
    laplace_nll: float
    start_along_track_correction_mm: float
    length_correction_mm: float
    downstream_endpoint_correction_mm: float
    latent_converged: bool
    profile_converged: bool
    wall_s: float
    path_s_mm: np.ndarray
    path_position_mm: np.ndarray
    path_tangent: np.ndarray
    path_energy_mev: np.ndarray
    path_beta: np.ndarray
    diagnostics: dict

    def output_values(self) -> dict:
        return attach_direction_components(self.updated_values, chart=self.updated_chart)


def _free_global_indices(fixed_params: Mapping[str, float] | None):
    fixed = {} if fixed_params is None else dict(fixed_params)
    free = []
    for index, name in enumerate(_COORDINATE_NAMES[:6]):
        if name in fixed:
            continue
        if name in {"dir_u", "dir_v"} and "direction" in fixed:
            continue
        if name == "length" and "visible_length" in fixed:
            continue
        free.append(index)
    if not free:
        raise ValueError("coherent MCS continuation has no free global coordinates")
    return tuple(free)


def _finite_physical_analytic_endpoint(model, result) -> bool:
    """Return whether an analytic stage ended on a finite physical FE path."""
    try:
        coefficients = np.asarray(result.coefficients, dtype=np.float64).reshape(
            int(model.n_modes)
        )
        if np.any(~np.isfinite(coefficients)):
            return False
        charge = float(model.charge_data_nll(coefficients))
        posterior = charge + 0.5 * float(coefficients @ coefficients)
    except (MCSPhysicalDomainError, FloatingPointError, OverflowError, ValueError):
        return False
    return bool(math.isfinite(charge) and math.isfinite(posterior))


def _certified_poll_as_exact_result(poll_result, *, max_restarts: int, message: str):
    """Adapt an authoritative exact poll to the exact-solver result contract."""
    return {
        "coefficients": np.ascontiguousarray(poll_result["coefficients"]),
        "charge_nll": float(poll_result["charge_nll"]),
        "posterior_nll": float(poll_result["posterior_nll"]),
        "converged": True,
        "optimizer_success": True,
        "message": str(message),
        "nfev": int(poll_result["evaluations"]),
        "nit": 0,
        "poll": tuple(poll_result["poll"]),
        "max_poll_downhill": float(poll_result["max_poll_downhill"]),
        "evaluations": int(poll_result["evaluations"]),
        "history": tuple(poll_result["history"]),
        "restart_count": 0,
        "max_restarts": int(max_restarts),
        "optimizer_attempts": (),
        "poll_history": (
            {
                "restart": -1,
                "centre": np.asarray(
                    poll_result["coefficients"], dtype=float
                ).tolist(),
                "poll": tuple(poll_result["poll"]),
                "max_downhill": float(poll_result["max_poll_downhill"]),
                "phase": str(message),
            },
        ),
        "initial_certificate_accepted": True,
    }


def _progressive_direct_budgets(total_iterations: int) -> tuple[int, ...]:
    """Deterministic 40, 40, remainder schedule for the direct G241 fallback."""
    remaining = max(1, int(total_iterations))
    budgets = []
    for checkpoint in (40, 40):
        if remaining <= 0:
            break
        budget = min(checkpoint, remaining)
        budgets.append(budget)
        remaining -= budget
    if remaining > 0:
        budgets.append(remaining)
    return tuple(budgets)


def _run_full_rank_multigrid_exact_controller(
    coarse_model,
    authoritative_model,
    *,
    latent_fd,
    latent_trust_max_component: float,
    coarse_iterations: int,
    fine_iterations: int,
    direct_iterations: int,
    exact_max_evaluations: int,
    exact_initial_trust_radius: float,
    exact_final_trust_radius: float,
    exact_max_restarts: int,
    poll_radii=(1.0e-2, 3.0e-3),
    poll_tolerance: float = 1.0e-4,
    coarse_failure_reason: str = "",
):
    """Full-rank multigrid initializer with exact fail-closed certification.

    The G121 stage changes only the numerical route to an initializer.  Every
    accepted result is certified against the unchanged full-dimensional G241
    scalar.  A failed multigrid/COBYQA route is discarded and recomputed from
    the direct G241 progressive initializer before one final exact solve.
    """
    diagnostics = {
        "used": True,
        "fallback_reason": "",
        "certified_stage": "",
        "fallback_used": False,
        "stages": [],
    }

    def analytic_stage(model, initial, budget, name):
        result = solve_latent_charge_map(
            model,
            initial_coefficients=initial,
            max_iterations=max(1, int(budget)),
            fd_step=latent_fd,
            trust_max_component=float(latent_trust_max_component),
        )
        physical = _finite_physical_analytic_endpoint(model, result)
        diagnostics["stages"].append(
            {
                "name": str(name),
                "max_iterations": int(max(1, int(budget))),
                "iterations": int(len(result.iterations)),
                "termination_reason": str(result.termination_reason),
                "finite_physical_endpoint": bool(physical),
                "posterior_nll": float(result.posterior_nll),
            }
        )
        return result, physical

    def exact_solver(initial, *, accept_certified_initial):
        return optimize_exact_posterior_latent_cobyqa(
            authoritative_model,
            initial_coefficients=initial,
            max_evaluations=int(exact_max_evaluations),
            initial_trust_radius=float(exact_initial_trust_radius),
            final_trust_radius=float(exact_final_trust_radius),
            poll_radii=tuple(poll_radii),
            poll_tolerance=float(poll_tolerance),
            max_restarts=int(exact_max_restarts),
            accept_certified_initial=bool(accept_certified_initial),
        )

    initializer = None
    route_failure = ""
    try:
        if coarse_model is None:
            route_failure = str(coarse_failure_reason) or "coarse_model_unavailable"
            coarse_ok = False
            coarse = None
        else:
            coarse, coarse_ok = analytic_stage(
                coarse_model, None, coarse_iterations, "coarse_g121"
            )
        if not coarse_ok:
            route_failure = route_failure or "nonphysical_coarse_endpoint"
        else:
            fine_first, fine_first_ok = analytic_stage(
                authoritative_model,
                coarse.coefficients,
                fine_iterations,
                "fine_g241_correction_1",
            )
            initializer = fine_first
            if not fine_first_ok:
                route_failure = "nonphysical_first_fine_endpoint"
            else:
                first_poll = exact_posterior_latent_coordinate_poll(
                    authoritative_model,
                    fine_first.coefficients,
                    poll_radii=tuple(poll_radii),
                    poll_tolerance=float(poll_tolerance),
                )
                diagnostics["stages"].append(
                    {
                        "name": "exact_poll_1",
                        "evaluations": int(first_poll["evaluations"]),
                        "max_downhill": float(first_poll["max_poll_downhill"]),
                        "certified": bool(first_poll["converged"]),
                    }
                )
                if first_poll["converged"]:
                    diagnostics["certified_stage"] = "first_exact_poll"
                    return (
                        _certified_poll_as_exact_result(
                            first_poll,
                            max_restarts=exact_max_restarts,
                            message="multigrid first exact coordinate certificate passed",
                        ),
                        fine_first,
                        diagnostics,
                    )

                fine_second, fine_second_ok = analytic_stage(
                    authoritative_model,
                    fine_first.coefficients,
                    fine_iterations,
                    "fine_g241_correction_2",
                )
                initializer = fine_second
                if not fine_second_ok:
                    route_failure = "nonphysical_second_fine_endpoint"
                else:
                    second_poll = exact_posterior_latent_coordinate_poll(
                        authoritative_model,
                        fine_second.coefficients,
                        poll_radii=tuple(poll_radii),
                        poll_tolerance=float(poll_tolerance),
                    )
                    diagnostics["stages"].append(
                        {
                            "name": "exact_poll_2",
                            "evaluations": int(second_poll["evaluations"]),
                            "max_downhill": float(
                                second_poll["max_poll_downhill"]
                            ),
                            "certified": bool(second_poll["converged"]),
                        }
                    )
                    if second_poll["converged"]:
                        diagnostics["certified_stage"] = "second_exact_poll"
                        return (
                            _certified_poll_as_exact_result(
                                second_poll,
                                max_restarts=exact_max_restarts,
                                message=(
                                    "multigrid second exact coordinate certificate "
                                    "passed"
                                ),
                            ),
                            fine_second,
                            diagnostics,
                        )

                    multigrid_exact = exact_solver(
                        fine_second.coefficients,
                        # The just-completed identical exact poll failed; do
                        # not pay for it a second time before COBYQA.
                        accept_certified_initial=False,
                    )
                    diagnostics["stages"].append(
                        {
                            "name": "multigrid_exact_cobyqa",
                            "evaluations": int(multigrid_exact["evaluations"]),
                            "max_downhill": float(
                                multigrid_exact["max_poll_downhill"]
                            ),
                            "certified": bool(multigrid_exact["converged"]),
                        }
                    )
                    if multigrid_exact["converged"]:
                        diagnostics["certified_stage"] = "multigrid_exact_cobyqa"
                        return multigrid_exact, fine_second, diagnostics
                    route_failure = "multigrid_exact_solver_not_certified"
    except (MCSPhysicalDomainError, FloatingPointError, OverflowError, ValueError) as exc:
        route_failure = f"{type(exc).__name__}: {exc}"

    # Fail closed: none of the multigrid state is retained.  Rebuild the
    # initializer solely on the authoritative G241 physics, then run the exact
    # solver (whose own initial poll may avoid an unnecessary COBYQA model).
    diagnostics["fallback_used"] = True
    diagnostics["fallback_reason"] = route_failure or "multigrid_route_failed"
    direct = None
    initial = None
    for index, budget in enumerate(_progressive_direct_budgets(direct_iterations), 1):
        direct, direct_ok = analytic_stage(
            authoritative_model,
            initial,
            budget,
            f"direct_g241_progressive_{index}",
        )
        if not direct_ok:
            raise RuntimeError(
                "direct G241 analytic fallback ended outside the physical domain"
            )
        initial = direct.coefficients
    direct_exact = exact_solver(
        direct.coefficients,
        accept_certified_initial=True,
    )
    diagnostics["stages"].append(
        {
            "name": "direct_g241_exact",
            "evaluations": int(direct_exact["evaluations"]),
            "max_downhill": float(direct_exact["max_poll_downhill"]),
            "certified": bool(direct_exact["converged"]),
        }
    )
    if direct_exact["converged"]:
        diagnostics["certified_stage"] = "direct_g241_exact"
    return direct_exact, direct, diagnostics


def run_physics_coherent_update(
    template_emitter,
    *,
    values: Mapping[str, float],
    chart: TangentDirectionChart,
    detector,
    wcd,
    pmt_model,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    obs_ts: np.ndarray,
    mpmt_types=None,
    fixed_params: Mapping[str, float] | None = None,
    length_limits: tuple[float, float] | None = None,
    t0_limits: tuple[float, float] | None = None,
    modes_per_plane: int = 12,
    process_grid_points: int = MINIMUM_PHYSICS_REFERENCE_GRID_POINTS,
    coherent_grid_points: int = MINIMUM_PHYSICS_REFERENCE_GRID_POINTS,
    latent_fd: float = 0.20,
    latent_max_iterations: int = 160,
    latent_gradient_tolerance: float = 1.0e-3,
    latent_trust_max_component: float = 1.0,
    global_scales: Sequence[float] = DEFAULT_GLOBAL_SCALES,
    max_global_evaluations: int = 300,
    initial_global_trust_radius: float = 1.0,
    final_global_trust_radius: float = 2.0e-2,
    deterministic_latent_start: bool = True,
    global_profile: bool = False,
    global_profile_mode: str = "conditional",
    latent_solver: str = "physics_response",
    latent_exact_max_evaluations: int = 600,
    latent_exact_initial_trust_radius: float = 0.5,
    latent_exact_final_trust_radius: float = 3.0e-4,
    latent_exact_max_restarts: int = 12,
    multigrid_initializer: bool = True,
    multigrid_grid_points: int = 121,
    multigrid_coarse_iterations: int = 40,
    multigrid_fine_iterations: int = 20,
    response_grid_points: int = 81,
    response_primary_trust: float = 1.0,
    response_secondary_trust: float = 1.5,
    response_probe_iterations: int = 12,
    response_proposal_iterations: int = 40,
    response_correction_iterations: int = 4,
    response_fallback_to_exact: bool = True,
    joint_max_cycles: int = 3,
    joint_cycle_tolerance: float = 1.0e-4,
    joint_final_global_trust_radius: float = 5.0e-3,
    joint_global_poll_tolerance: float = 1.0e-4,
    joint_global_max_restarts: int = 3,
    joint_model_cache_size: int = 32,
    numba_threads: int = 4,
) -> PhysicsCoherentResult:
    """Run the all-PMT coherent-charge reference fit.

    The production default is a fixed-global, full-24-dimensional
    physics-response profile.  A G81 analytic log-rate tangent proposes each
    update, but every step is accepted by the unchanged nonlinear optical
    posterior and the final corrections/scalar evaluation use the authoritative
    all-PMT G241 model.  No KL rank, Fermi--Eyges prior, or optical component is
    removed.  The slower ``latent_solver="derivative_free"`` remains the exact
    coordinate-poll reference; ``latent_solver="analytic"`` and the historical
    Fisher--Laplace/global modes remain explicit development paths.
    """
    wall0 = time.perf_counter()
    modes = int(modes_per_plane)
    if modes <= 0:
        raise ValueError("modes_per_plane must be positive")
    if int(process_grid_points) != int(coherent_grid_points):
        raise ValueError("FE and optical path grids must be identical")
    if int(process_grid_points) < MINIMUM_PHYSICS_REFERENCE_GRID_POINTS:
        raise ValueError(
            "physics-reference coherent MCS requires a common FE/optical grid "
            f"of at least {MINIMUM_PHYSICS_REFERENCE_GRID_POINTS} points; "
            "coarser grids remain available only in explicitly legacy paths"
        )
    local_values, profile_chart = reanchor_values(values, chart)
    theta0 = np.asarray(
        [
            float(local_values["x0"]),
            float(local_values["y0"]),
            float(local_values["z0"]),
            0.0,
            0.0,
            float(local_values["length"]),
            float(local_values.get("t0", 0.0)),
        ],
        dtype=np.float64,
    )
    direction0 = profile_chart.anchor.copy()
    emitter = template_emitter.copy()
    emitter.primary_mcs_process_modes_per_plane = modes
    emitter.primary_mcs_process_grid_points = int(process_grid_points)
    free = _free_global_indices(fixed_params)
    profile_mode = str(global_profile_mode).strip().lower().replace("-", "_")
    if profile_mode not in {"conditional", "fisher_laplace", "joint_exact"}:
        raise ValueError(
            "global_profile_mode must be conditional, fisher_laplace, or joint_exact"
        )
    if bool(global_profile):
        if profile_mode != "conditional":
            raise ValueError(
                "global_profile=True cannot be combined with global_profile_mode"
            )
        profile_mode = "fisher_laplace"
    is_fisher_profile = profile_mode == "fisher_laplace"
    is_joint_exact = profile_mode == "joint_exact"

    restore_threads, active_threads = _thread_context(numba_threads)
    try:
        evaluator = CoupledCoherentEvaluator(
            emitter,
            wcd,
            pmt_model,
            p_locations,
            pmt_normals,
            obs_pes,
            obs_ts,
            chart=profile_chart,
            detector=detector,
            mpmt_types=mpmt_types,
            n_modes=2 * modes,
            n_grid=int(coherent_grid_points),
            aperture_radius_mm=float(
                getattr(template_emitter, "primary_endpoint_aperture_radius_mm", 45.0)
            ),
            path_field="fali",
            direct_timing_bins=1,
            sparse_receiver=False,
            charge_only=True,
            require_contained_track=True,
            length_limits=length_limits,
            t0_limits=t0_limits,
            max_cached_models=int(joint_model_cache_size),
        )
        latent_solver_name = str(latent_solver).strip().lower().replace("-", "_")
        if latent_solver_name not in {
            "analytic", "derivative_free", "physics_response"
        }:
            raise ValueError(
                "latent_solver must be physics_response, derivative_free, or analytic"
            )
        if is_joint_exact and latent_solver_name != "derivative_free":
            raise ValueError("joint_exact requires latent_solver='derivative_free'")
        joint_profile = None
        if is_fisher_profile:
            profiled = optimize_reprofiled_fisher_laplace(
                evaluator,
                theta0,
                free_indices=free,
                global_scales=global_scales,
                latent_fd=latent_fd,
                latent_max_iterations=int(latent_max_iterations),
                latent_gradient_tolerance=float(latent_gradient_tolerance),
                latent_trust_max_component=float(latent_trust_max_component),
                max_global_evaluations=int(max_global_evaluations),
                initial_trust_radius=float(initial_global_trust_radius),
                final_trust_radius=float(final_global_trust_radius),
                deterministic_latent_start=bool(deterministic_latent_start),
            )
            theta1 = np.asarray(profiled.theta, dtype=np.float64)
            latent_coefficients = np.asarray(profiled.latent.coefficients)
            latent_charge_nll = float(profiled.latent.charge_nll)
            latent_posterior_nll = float(profiled.latent.posterior_nll)
            latent_laplace_nll = float(profiled.latent.laplace_nll)
            latent_covariance = np.asarray(profiled.latent.covariance)
            latent_converged = bool(profiled.latent.converged)
            profile_converged = bool(profiled.converged)
        else:
            initial_model = evaluator.model(theta0)
            if initial_model is None:
                raise RuntimeError("initial coherent global track is invalid")
            multigrid_diagnostics = {
                "enabled": bool(multigrid_initializer),
                "used": False,
                "fallback_reason": "disabled",
            }
            response_diagnostics = {
                "enabled": latent_solver_name == "physics_response",
                "used": False,
                "fallback_to_exact": False,
                "fallback_reason": "",
            }
            initializer = None
            coarse_model = None
            coarse_failure_reason = ""
            build_reduced_model = bool(multigrid_initializer) or (
                latent_solver_name == "physics_response"
            )
            if build_reduced_model:
                coarse_grid = int(
                    response_grid_points
                    if latent_solver_name == "physics_response"
                    else multigrid_grid_points
                )
                minimum_reduced_grid = (
                    MINIMUM_PHYSICS_RESPONSE_GRID_POINTS
                    if latent_solver_name == "physics_response" else 17
                )
                if coarse_grid < 17 or coarse_grid >= int(coherent_grid_points):
                    raise ValueError(
                        "reduced proposal grid must lie between 17 and the "
                        "authoritative coherent grid"
                    )
                if coarse_grid < minimum_reduced_grid:
                    raise ValueError(
                        "physics-response proposal grid must contain at least "
                        f"{MINIMUM_PHYSICS_RESPONSE_GRID_POINTS} points"
                    )
                coarse_emitter = emitter.copy()
                coarse_emitter.primary_mcs_process_modes_per_plane = modes
                coarse_emitter.primary_mcs_process_grid_points = coarse_grid
                try:
                    coarse_model = FixedTrackCoherentMCSObjective(
                        coarse_emitter,
                        wcd,
                        pmt_model,
                        p_locations,
                        pmt_normals,
                        obs_pes,
                        obs_ts,
                        vertex=theta0[:3],
                        direction=direction0,
                        length=float(theta0[5]),
                        t0=0.0,
                        mpmt_types=mpmt_types,
                        n_grid=coarse_grid,
                        aperture_radius_mm=float(
                            getattr(
                                template_emitter,
                                "primary_endpoint_aperture_radius_mm",
                                45.0,
                            )
                        ),
                        path_field="fali",
                        direct_timing_bins=1,
                        precomputed_base_emitter=initial_model.base_emitter.copy(),
                        precomputed_base_pes=initial_model.base_pes,
                        precomputed_base_timing=None,
                        sparse_receiver=False,
                        charge_only=True,
                        path_validator=initial_model.path_validator,
                    )
                except (
                    MCSPhysicalDomainError,
                    FloatingPointError,
                    OverflowError,
                    ValueError,
                ) as exc:
                    coarse_failure_reason = f"{type(exc).__name__}: {exc}"
            if latent_solver_name == "analytic":
                # The analytic-only development mode keeps its historical
                # authoritative-grid convergence contract.  The production
                # multigrid controller below is exact-posterior certified.
                initializer = solve_latent_charge_map(
                    initial_model,
                    max_iterations=int(latent_max_iterations),
                    fd_step=latent_fd,
                    trust_max_component=float(latent_trust_max_component),
                )
                if not initializer.converged:
                    raise RuntimeError(
                        "resolved-grid analytic latent solver did not converge: "
                        f"{initializer.termination_reason}, "
                        f"max|g|={initializer.final_gradient_max_abs:.6g}"
                    )
                exact_latent = {
                    "coefficients": initializer.coefficients,
                    "charge_nll": initializer.charge_nll,
                    "posterior_nll": initializer.posterior_nll,
                    "converged": True,
                    "optimizer_success": True,
                    "message": initializer.termination_reason,
                    "nfev": initializer.objective_evaluations,
                    "nit": len(initializer.iterations),
                    "poll": (),
                    "max_poll_downhill": 0.0,
                }
            elif latent_solver_name == "physics_response":
                response_failure = str(coarse_failure_reason)
                try:
                    if coarse_model is None:
                        raise RuntimeError(
                            response_failure or "response proposal model is unavailable"
                        )
                    exact_latent = (
                        solve_reduced_response_with_authoritative_correction(
                            coarse_model,
                            initial_model,
                            primary_trust=float(response_primary_trust),
                            secondary_trust=float(response_secondary_trust),
                            probe_iterations=int(response_probe_iterations),
                            proposal_iterations=int(response_proposal_iterations),
                            correction_iterations=int(response_correction_iterations),
                        )
                    )
                    if not bool(exact_latent["converged"]):
                        raise RuntimeError(
                            "physics-response continuation returned a nonphysical state"
                        )
                    proposal = exact_latent["proposal"]
                    correction = exact_latent["authoritative_correction"]
                    response_diagnostics.update(
                        {
                            "used": True,
                            "proposal_grid_points": int(response_grid_points),
                            "authoritative_grid_points": int(
                                coherent_grid_points
                            ),
                            "modes_per_plane": int(modes),
                            "probe_iterations": int(response_probe_iterations),
                            "proposal_iteration_budget": int(
                                response_proposal_iterations
                            ),
                            "authoritative_correction_iteration_budget": int(
                                response_correction_iterations
                            ),
                            "primary_trust": float(response_primary_trust),
                            "secondary_trust": float(response_secondary_trust),
                            "chosen_trust": float(
                                proposal["competing"]["chosen_trust"]
                            ),
                            "chosen_probe": str(
                                proposal["competing"]["chosen"]
                            ),
                            "primary_probe_posterior_nll": float(
                                proposal["competing"][
                                    "primary_probe_posterior"
                                ]
                            ),
                            "secondary_probe_posterior_nll": float(
                                proposal["competing"][
                                    "secondary_probe_posterior"
                                ]
                            ),
                            "proposal_posterior_nll": float(
                                proposal["posterior_nll"]
                            ),
                            "authoritative_posterior_nll": float(
                                correction["posterior_nll"]
                            ),
                            "objective_evaluations": int(
                                exact_latent["objective_evaluations"]
                            ),
                            "jacobian_evaluations": int(
                                exact_latent["jacobian_evaluations"]
                            ),
                            "wall_s": float(exact_latent["wall_s"]),
                            "proposal_termination_reason": str(
                                proposal["termination_reason"]
                            ),
                            "correction_termination_reason": str(
                                correction["termination_reason"]
                            ),
                            "exact_scalar_evaluated": True,
                            "exact_coordinate_stationarity_claimed": False,
                        }
                    )
                    multigrid_diagnostics.update(
                        {
                            "enabled": False,
                            "used": False,
                            "fallback_reason": "physics_response_selected",
                        }
                    )
                except (
                    MCSPhysicalDomainError,
                    FloatingPointError,
                    OverflowError,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    response_failure = f"{type(exc).__name__}: {exc}"
                    if not bool(response_fallback_to_exact):
                        raise
                    response_diagnostics.update(
                        {
                            "used": False,
                            "fallback_to_exact": True,
                            "fallback_reason": response_failure,
                        }
                    )
                    exact_latent, initializer, controller_diagnostics = (
                        _run_full_rank_multigrid_exact_controller(
                            None,
                            initial_model,
                            latent_fd=latent_fd,
                            latent_trust_max_component=float(
                                latent_trust_max_component
                            ),
                            coarse_iterations=int(multigrid_coarse_iterations),
                            fine_iterations=int(multigrid_fine_iterations),
                            direct_iterations=int(latent_max_iterations),
                            exact_max_evaluations=int(
                                latent_exact_max_evaluations
                            ),
                            exact_initial_trust_radius=float(
                                latent_exact_initial_trust_radius
                            ),
                            exact_final_trust_radius=float(
                                latent_exact_final_trust_radius
                            ),
                            exact_max_restarts=int(latent_exact_max_restarts),
                            poll_radii=(1.0e-2, 3.0e-3),
                            poll_tolerance=1.0e-4,
                            coarse_failure_reason=response_failure,
                        )
                    )
                    multigrid_diagnostics.update(controller_diagnostics)
            else:
                if bool(multigrid_initializer):
                    exact_latent, initializer, controller_diagnostics = (
                        _run_full_rank_multigrid_exact_controller(
                            coarse_model,
                            initial_model,
                            latent_fd=latent_fd,
                            latent_trust_max_component=float(
                                latent_trust_max_component
                            ),
                            coarse_iterations=int(multigrid_coarse_iterations),
                            fine_iterations=int(multigrid_fine_iterations),
                            direct_iterations=int(latent_max_iterations),
                            exact_max_evaluations=int(
                                latent_exact_max_evaluations
                            ),
                            exact_initial_trust_radius=float(
                                latent_exact_initial_trust_radius
                            ),
                            exact_final_trust_radius=float(
                                latent_exact_final_trust_radius
                            ),
                            exact_max_restarts=int(latent_exact_max_restarts),
                            poll_radii=(1.0e-2, 3.0e-3),
                            poll_tolerance=1.0e-4,
                            coarse_failure_reason=coarse_failure_reason,
                        )
                    )
                    multigrid_diagnostics.update(controller_diagnostics)
                    multigrid_diagnostics["coarse_grid_points"] = int(
                        multigrid_grid_points
                    )
                    multigrid_diagnostics["authoritative_grid_points"] = int(
                        coherent_grid_points
                    )
                else:
                    initializer = solve_latent_charge_map(
                        initial_model,
                        max_iterations=int(latent_max_iterations),
                        fd_step=latent_fd,
                        trust_max_component=float(latent_trust_max_component),
                    )
                    exact_latent = optimize_exact_posterior_latent_cobyqa(
                        initial_model,
                        initial_coefficients=initializer.coefficients,
                        max_evaluations=int(latent_exact_max_evaluations),
                        initial_trust_radius=float(
                            latent_exact_initial_trust_radius
                        ),
                        final_trust_radius=float(latent_exact_final_trust_radius),
                        poll_radii=(1.0e-2, 3.0e-3),
                        poll_tolerance=1.0e-4,
                        max_restarts=int(latent_exact_max_restarts),
                    )
            theta1 = theta0.copy()
            latent_coefficients = np.asarray(exact_latent["coefficients"])
            latent_charge_nll = float(exact_latent["charge_nll"])
            latent_posterior_nll = float(exact_latent["posterior_nll"])
            latent_laplace_nll = (
                float(initializer.laplace_nll)
                if latent_solver_name == "analytic" else math.nan
            )
            latent_covariance = (
                np.asarray(initializer.covariance)
                if latent_solver_name == "analytic" else
                np.full((2 * modes, 2 * modes), np.nan, dtype=np.float64)
            )
            latent_converged = bool(exact_latent["converged"])
            profile_converged = bool(exact_latent["converged"])
            if not latent_converged:
                raise RuntimeError(
                    "exact derivative-free latent MAP is not coordinate-poll "
                    "stationary: "
                    f"max_downhill={exact_latent['max_poll_downhill']:.6g}, "
                    "tolerance=0.0001, "
                    f"restarts={exact_latent['restart_count']}/"
                    f"{exact_latent['max_restarts']}, "
                    f"evaluations={exact_latent['evaluations']}, "
                    f"optimizer_success={exact_latent['optimizer_success']}"
                )
            if is_joint_exact:
                joint_profile = optimize_alternating_exact_joint_map(
                    evaluator,
                    theta0,
                    initial_latent_result=exact_latent,
                    free_indices=free,
                    global_scales=global_scales,
                    max_cycles=int(joint_max_cycles),
                    cycle_tolerance=float(joint_cycle_tolerance),
                    max_global_evaluations=int(max_global_evaluations),
                    initial_global_trust_radius=float(initial_global_trust_radius),
                    final_global_trust_radius=float(
                        joint_final_global_trust_radius
                    ),
                    global_poll_radii=(2.0e-2, 5.0e-3),
                    global_poll_tolerance=float(joint_global_poll_tolerance),
                    global_max_restarts=int(joint_global_max_restarts),
                    latent_max_evaluations=int(latent_exact_max_evaluations),
                    latent_initial_trust_radius=float(
                        latent_exact_initial_trust_radius
                    ),
                    latent_final_trust_radius=float(
                        latent_exact_final_trust_radius
                    ),
                    latent_poll_radii=(1.0e-2, 3.0e-3),
                    latent_poll_tolerance=1.0e-4,
                    latent_max_restarts=int(latent_exact_max_restarts),
                )
                theta1 = np.asarray(joint_profile["theta"], dtype=np.float64)
                exact_latent = joint_profile["latent"]
                latent_coefficients = np.asarray(
                    joint_profile["coefficients"], dtype=np.float64
                )
                latent_charge_nll = float(joint_profile["charge_nll"])
                latent_posterior_nll = float(joint_profile["posterior_nll"])
                latent_laplace_nll = math.nan
                latent_covariance = np.full(
                    (2 * modes, 2 * modes), np.nan, dtype=np.float64
                )
                latent_converged = bool(exact_latent["converged"])
                profile_converged = bool(joint_profile["converged"])
    finally:
        restore_threads()

    direction1 = profile_chart.direction(float(theta1[3]), float(theta1[4]))
    if direction1 is None:
        raise RuntimeError("reference optimizer returned an invalid direction")
    updated = dict(local_values)
    updated.update(
        {
            "x0": float(theta1[0]),
            "y0": float(theta1[1]),
            "z0": float(theta1[2]),
            "dir_u": float(theta1[3]),
            "dir_v": float(theta1[4]),
            "length": float(theta1[5]),
            # The charge-only stage cannot update t0.
            "t0": float(theta0[6]),
        }
    )

    final_model = evaluator.model(theta1)
    if final_model is None:
        raise RuntimeError("final coherent global track is invalid")
    path = build_arclength_fe_path(
        final_model.path_emitter,
        latent_coefficients,
        n_grid=int(coherent_grid_points),
    )
    vertex_delta = theta1[:3] - theta0[:3]
    start_along = float(vertex_delta @ direction0)
    start_transverse = vertex_delta - start_along * direction0
    length_delta = float(theta1[5] - theta0[5])
    initial_endpoint = theta0[:3] + theta0[5] * direction0
    fitted_endpoint = np.asarray(path["position"][-1], dtype=np.float64)
    endpoint_delta = fitted_endpoint - initial_endpoint
    direction_change = math.acos(
        float(np.clip(direction0 @ direction1, -1.0, 1.0))
    )

    evaluations = [
        {
            "scaled_coordinates": list(row.scaled_coordinates),
            "theta": list(row.theta),
            "laplace_nll": float(row.laplace_nll),
            "posterior_nll": float(row.posterior_nll),
            "latent_converged": bool(row.latent_converged),
            "latent_gradient_max_abs": float(row.latent_gradient_max_abs),
            "latent_newton_decrement": float(row.latent_newton_decrement),
            "termination_reason": str(row.termination_reason),
            "model_valid": bool(row.model_valid),
        }
        for row in profiled.evaluations
    ] if is_fisher_profile else []
    joint_cycles = []
    if is_joint_exact and joint_profile is not None:
        for row in joint_profile["cycles"]:
            geometry = row["geometry"]
            latent_row = row["latent"]
            post_poll = row["post_latent_geometry_poll"]
            joint_cycles.append(
                {
                    "cycle": int(row["cycle"]),
                    "posterior_before": float(row["posterior_before"]),
                    "posterior_after_geometry": float(
                        row["posterior_after_geometry"]
                    ),
                    "posterior_after_latent": float(row["posterior_after_latent"]),
                    "improvement": float(row["improvement"]),
                    "theta": np.asarray(row["theta"], dtype=float).tolist(),
                    "coefficients": np.asarray(
                        row["coefficients"], dtype=float
                    ).tolist(),
                    "geometry_converged": bool(geometry["converged"]),
                    "geometry_optimizer_success": bool(
                        geometry["optimizer_success"]
                    ),
                    "geometry_nfev": int(geometry["nfev"]),
                    "geometry_restart_count": int(geometry["restart_count"]),
                    "geometry_poll_max_downhill": float(
                        geometry["max_poll_downhill"]
                    ),
                    "latent_converged": bool(latent_row["converged"]),
                    "latent_nfev": int(latent_row["nfev"]),
                    "latent_restart_count": int(latent_row["restart_count"]),
                    "latent_poll_max_downhill": float(
                        latent_row["max_poll_downhill"]
                    ),
                    "post_latent_geometry_poll_max_downhill": float(
                        post_poll["max_downhill"]
                    ),
                }
            )
    if is_joint_exact:
        implementation_name = (
            "physics_reference_all_pmt_alternating_exact_joint_map_v1"
        )
        objective_description = (
            "charge Poisson NLL + standard-normal FE prior; alternating exact "
            "derivative-free six-coordinate geometry and latent MAP blocks"
        )
    elif is_fisher_profile:
        implementation_name = (
            "physics_reference_all_pmt_reprofiled_fisher_laplace_v1"
        )
        objective_description = (
            "charge Poisson NLL + standard-normal FE prior at the fully converged "
            "latent MAP + 0.5 logdet Poisson-Fisher latent information"
        )
    elif latent_solver_name == "analytic":
        implementation_name = (
            "physics_reference_all_pmt_resolved_analytic_conditional_path_v1"
        )
        objective_description = (
            "charge Poisson NLL + standard-normal FE prior; resolved-grid "
            "analytic conditional-path MAP at the accepted straight global track"
        )
    elif latent_solver_name == "physics_response":
        implementation_name = (
            "physics_reference_full_rank_log_rate_response_conditional_path_v1"
        )
        objective_description = (
            "charge Poisson NLL + standard-normal FE prior; full-rank analytic "
            "log-rate response proposals with exact nonlinear acceptance and "
            "authoritative all-PMT G241 correction"
        )
    else:
        implementation_name = (
            "physics_reference_all_pmt_exact_posterior_conditional_path_v1"
        )
        objective_description = (
            "charge Poisson NLL + standard-normal FE prior; exact derivative-free "
            "conditional-path MAP at the accepted straight global track"
        )
    diagnostics = {
        "implementation": implementation_name,
        "objective": objective_description,
        "global_profile_mode": profile_mode,
        "global_track_profiled": bool(is_fisher_profile or is_joint_exact),
        "laplace_reported": bool(
            is_fisher_profile
            or (profile_mode == "conditional" and latent_solver_name == "analytic")
        ),
        "covariance_reported": bool(
            is_fisher_profile
            or (profile_mode == "conditional" and latent_solver_name == "analytic")
        ),
        "joint_map_claim_only": bool(is_joint_exact),
        "joint_mixed_direction_stationarity_claimed": False,
        "timing_used_for_mcs": False,
        "straight_fit_used_as_prior": False,
        "uses_event_truth": False,
        "uses_empirical_mcs_scale": False,
        "charge_total_conditioned_to_observation": True,
        "absolute_light_yield_identifiable": False,
        "free_global_indices": list(map(int, free)),
        "free_global_names": [_COORDINATE_NAMES[i] for i in free],
        "global_scales": np.asarray(global_scales, dtype=float).tolist(),
        "modes_per_plane": modes,
        "latent_dimension": 2 * modes,
        "process_grid_points": int(process_grid_points),
        "coherent_grid_points": int(coherent_grid_points),
        "uses_discrete_range_grid": False,
        "range_coordinate": "continuous_float64_length_mm",
        "path_grid_role": "finite_element_quadrature_only",
        "output_length_quantization_mm": None,
        "full_pmt_support": True,
        "physics_response": (
            dict(response_diagnostics)
            if profile_mode == "conditional"
            else {"enabled": False, "used": False, "fallback_reason": "profile_mode"}
        ),
        "multigrid_initializer": (
            dict(multigrid_diagnostics)
            if profile_mode == "conditional"
            else {"enabled": False, "used": False, "fallback_reason": "profile_mode"}
        ),
        "detector_pmt_count": int(np.asarray(p_locations).shape[0]),
        "numba_threads": int(active_threads),
        "latent_gradient_max_abs": (
            float(profiled.latent.final_gradient_max_abs)
            if is_fisher_profile else (
                float(initializer.final_gradient_max_abs)
                if profile_mode == "conditional" and latent_solver_name == "analytic"
                else math.nan
            )
        ),
        "latent_gradient_norm": (
            float(profiled.latent.final_gradient_norm)
            if is_fisher_profile else (
                float(initializer.final_gradient_norm)
                if profile_mode == "conditional" and latent_solver_name == "analytic"
                else math.nan
            )
        ),
        "latent_newton_decrement": (
            float(getattr(profiled.latent, "newton_decrement", math.nan))
            if is_fisher_profile else (
                float(initializer.final_newton_decrement)
                if profile_mode == "conditional" and latent_solver_name == "analytic"
                else math.nan
            )
        ),
        "latent_termination_reason": (
            str(profiled.latent.termination_reason)
            if is_fisher_profile else (
                str(initializer.termination_reason)
                if profile_mode == "conditional" and latent_solver_name == "analytic"
                else (
                    str(
                        exact_latent.get(
                            "message", "physics_response_completed"
                        )
                    )
                    if latent_solver_name == "physics_response"
                    else "exact_coordinate_poll"
                )
            )
        ),
        "global_optimizer_success": (
            bool(profiled.optimizer_success) if is_fisher_profile
            else (
                bool(all(row["geometry_optimizer_success"] for row in joint_cycles))
                if is_joint_exact else bool(exact_latent["optimizer_success"])
            )
        ),
        "latent_exact_max_restarts": int(latent_exact_max_restarts),
        "global_optimizer_message": (
            str(profiled.message) if is_fisher_profile else (
                "alternating_exact_block_coordinate_map"
                if is_joint_exact else str(exact_latent["message"])
            )
        ),
        "global_nfev": (
            int(profiled.nfev) if is_fisher_profile else (
                int(sum(row["geometry_nfev"] for row in joint_cycles))
                if is_joint_exact else int(exact_latent["nfev"])
            )
        ),
        "global_nit": (
            int(profiled.nit) if is_fisher_profile else (
                -1 if is_joint_exact else int(exact_latent["nit"])
            )
        ),
        "local_poll_radius_scaled": (
            float(profiled.local_poll_radius) if is_fisher_profile else math.nan
        ),
        "local_poll_max_downhill_nll": (
            float(profiled.local_poll_max_downhill) if is_fisher_profile
            else (
                float(
                    joint_profile["post_latent_geometry_poll"]["max_downhill"]
                ) if is_joint_exact and joint_profile is not None
                else float(exact_latent["max_poll_downhill"])
            )
        ),
        "latent_coordinate_poll": (
            [] if is_fisher_profile else [dict(x) for x in exact_latent["poll"]]
        ),
        "optical_model_build_count": int(evaluator.optical_model_build_count),
        "resident_optical_model_count": int(
            evaluator.resident_optical_model_count
        ),
        "optical_model_cache_evictions": int(evaluator.model_cache_evictions),
        "coherent_field_evaluation_count": int(
            evaluator.coherent_field_evaluation_count
        ),
        "invalid_model_evaluation_count": int(evaluator.invalid_evaluations),
        "physical_domain_rejection_count": int(evaluator.physical_domain_rejections),
        "curved_path_rejection_count": int(evaluator.curved_path_rejections),
        "model_build_failure_count": int(evaluator.model_build_failures),
        "start_transverse_correction_mm": np.asarray(start_transverse).tolist(),
        "start_transverse_correction_norm_mm": float(np.linalg.norm(start_transverse)),
        "direction_correction_rad": float(direction_change),
        "direction_correction_deg": float(np.degrees(direction_change)),
        "endpoint_correction_vector_mm": endpoint_delta.tolist(),
        "endpoint_correction_norm_mm": float(np.linalg.norm(endpoint_delta)),
        "delta_zero_reference_relative_l1": float(
            getattr(final_model, "delta_zero_reference_relative_l1", math.nan)
        ),
        "global_evaluations": evaluations,
        "joint_cycles": joint_cycles,
    }
    return PhysicsCoherentResult(
        initial_values=dict(local_values),
        updated_values=updated,
        updated_chart=profile_chart,
        coefficients_mean=np.ascontiguousarray(latent_coefficients),
        coefficients_covariance=np.ascontiguousarray(latent_covariance),
        charge_nll=latent_charge_nll,
        posterior_nll=latent_posterior_nll,
        laplace_nll=latent_laplace_nll,
        start_along_track_correction_mm=start_along,
        length_correction_mm=length_delta,
        downstream_endpoint_correction_mm=float(endpoint_delta @ direction0),
        latent_converged=latent_converged,
        profile_converged=profile_converged,
        wall_s=float(time.perf_counter() - wall0),
        path_s_mm=np.ascontiguousarray(path["s"]),
        path_position_mm=np.ascontiguousarray(path["position"]),
        path_tangent=np.ascontiguousarray(path["tangent"]),
        path_energy_mev=np.ascontiguousarray(path["energy"]),
        path_beta=np.ascontiguousarray(path["beta"]),
        diagnostics=diagnostics,
    )
