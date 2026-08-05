"""Fast nonlinear twelve-mode Fermi--Eyges coherent-path continuation.

The accepted straight-track fit remains the basin finder and global estimator.
This module then profiles a standardized Fermi--Eyges trajectory with twelve
Karhunen--Loeve displacement modes in each transverse plane.  The nonlinear
finite-aperture line field is differentiated analytically with respect to all
24 latent coefficients.  The only global coordinates revisited are

* the start position along the fitted trajectory; and
* the visible track length.

Those are geometry-general physical coordinates.  No WCTE beam-axis or event
truth appears in this implementation.  Candidate global points are accepted
only when the re-profiled Fisher--Laplace charge objective decreases.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

import numpy as np

from .mcs_coupled_schur import CoupledCoherentEvaluator
from .mcs_latent_profile import optimize_profiled_laplace_track_aligned
from .track_parameterization import (
    TangentDirectionChart,
    attach_direction_components,
    reanchor_values,
)


@dataclass
class Fast12CoherentResult:
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
    diagnostics: dict

    def output_values(self) -> dict:
        return attach_direction_components(self.updated_values, chart=self.updated_chart)


def _thread_context(requested: int):
    """Return ``(restore, active)`` for the process-global Numba thread count."""
    try:
        from numba import get_num_threads, set_num_threads
        previous = int(get_num_threads())
        active = max(1, int(requested))
        set_num_threads(active)
        return (lambda: set_num_threads(previous)), active
    except Exception:
        return (lambda: None), 1


def run_fast12_coherent_update(
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
    process_grid_points: int = 41,
    coherent_grid_points: int = 41,
    latent_fd: float = 0.20,
    latent_max_iterations: int = 4,
    candidate_latent_max_iterations: int = 2,
    track_cycles: int = 2,
    longitudinal_step_mm: float = 10.0,
    length_step_mm: float = 15.0,
    latent_trust_max_component: float = 1.0,
    track_trust_max_scaled_component: float = 1.0,
    sparse_neighbor_radius_mm: float = 100.0,
    numba_threads: int = 4,
) -> Fast12CoherentResult:
    """Apply the validated high-rank nonlinear charge continuation.

    The latent prior is exactly ``N(0,I)`` because the FE covariance is already
    carried by the KL eigenfunctions.  The rank, grids, finite-difference scales,
    and iteration limits are numerical convergence controls; no reconstruction
    residual or WCSim distribution determines them.
    """
    wall0 = time.perf_counter()
    if int(modes_per_plane) != 12:
        raise ValueError(
            "fast coherent continuation is validated for 12 KL modes per plane"
        )
    if int(process_grid_points) != int(coherent_grid_points):
        # A common path grid avoids hidden interpolation between the FE prior
        # and the nonlinear optical field.
        raise ValueError("process_grid_points and coherent_grid_points must match")
    fixed = {} if fixed_params is None else dict(fixed_params)
    if any(name in fixed for name in ("x0", "y0", "z0")):
        raise NotImplementedError(
            "a fixed Cartesian vertex coordinate is incompatible with a coupled "
            "start-along-track update; disable the continuation for this diagnostic"
        )
    if "length" in fixed or "visible_length" in fixed:
        raise NotImplementedError(
            "fast coherent continuation requires a free visible length"
        )

    # Re-anchor the tangent chart at the accepted direction.  The coherent
    # continuation does not change direction, so the final direction remains a
    # stable chart origin for arbitrary hemispheres and detector orientations.
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
    emitter.primary_mcs_process_modes_per_plane = 12
    emitter.primary_mcs_process_grid_points = int(process_grid_points)

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
            n_modes=24,
            n_grid=int(coherent_grid_points),
            aperture_radius_mm=float(
                getattr(template_emitter, "primary_endpoint_aperture_radius_mm", 45.0)
            ),
            path_field="fali",
            direct_timing_bins=1,
            sparse_receiver=True,
            sparse_neighbor_radius_mm=float(sparse_neighbor_radius_mm),
            charge_only=True,
            require_contained_track=True,
            length_limits=length_limits,
            t0_limits=t0_limits,
        )
        result = optimize_profiled_laplace_track_aligned(
            evaluator,
            theta0,
            longitudinal_step_mm=float(longitudinal_step_mm),
            length_step_mm=float(length_step_mm),
            latent_fd=float(latent_fd),
            latent_max_iterations=int(latent_max_iterations),
            candidate_latent_max_iterations=int(candidate_latent_max_iterations),
            latent_trust_max_component=float(latent_trust_max_component),
            track_cycles=int(track_cycles),
            track_trust_max_scaled_component=float(
                track_trust_max_scaled_component
            ),
        )
    finally:
        restore_threads()

    theta1 = np.asarray(result.theta, dtype=np.float64)
    vertex_delta = theta1[:3] - theta0[:3]
    start_along = float(np.dot(vertex_delta, direction0))
    transverse_leak = float(
        np.linalg.norm(vertex_delta - start_along * direction0)
    )
    length_delta = float(theta1[5] - theta0[5])
    updated = dict(local_values)
    updated.update(
        {
            "x0": float(theta1[0]),
            "y0": float(theta1[1]),
            "z0": float(theta1[2]),
            "dir_u": 0.0,
            "dir_v": 0.0,
            "length": float(theta1[5]),
            "t0": float(theta1[6]),
        }
    )
    first_model = evaluator.model(theta0)
    support_size = (
        None if first_model is None else int(first_model.coherent_active_indices.size)
    )
    diagnostics = {
        "implementation": "analytic_fali_jacobian_fisher_laplace_v1",
        "modes_per_plane": 12,
        "latent_dimension": 24,
        "process_grid_points": int(process_grid_points),
        "coherent_grid_points": int(coherent_grid_points),
        "latent_fd": float(latent_fd),
        "latent_max_iterations": int(latent_max_iterations),
        "candidate_latent_max_iterations": int(candidate_latent_max_iterations),
        "track_cycles": int(track_cycles),
        "longitudinal_step_mm": float(longitudinal_step_mm),
        "length_step_mm": float(length_step_mm),
        "latent_trust_max_component": float(latent_trust_max_component),
        "track_trust_max_scaled_component": float(
            track_trust_max_scaled_component
        ),
        "sparse_neighbor_radius_mm": float(sparse_neighbor_radius_mm),
        "coherent_support_size": support_size,
        "detector_pmt_count": int(np.asarray(p_locations).shape[0]),
        "numba_threads": int(active_threads),
        "optical_model_build_count": int(evaluator.optical_model_build_count),
        "coherent_field_evaluation_count": int(
            evaluator.coherent_field_evaluation_count
        ),
        "invalid_model_evaluation_count": int(evaluator.invalid_evaluations),
        "latent_iterations": [
            {
                "iteration": int(x.iteration),
                "charge_nll": float(x.charge_nll),
                "posterior_nll": float(x.posterior_nll),
                "gradient_norm": float(x.gradient_norm),
                "proposed_step_norm": float(x.proposed_step_norm),
                "accepted_scale": float(x.accepted_scale),
                "accepted": bool(x.accepted),
            }
            for x in result.latent.iterations
        ],
        "profile_iterations": [
            {
                "cycle": int(x.cycle),
                "theta_before": list(x.theta_before),
                "theta_after": list(x.theta_after),
                "laplace_before": float(x.laplace_before),
                "laplace_after": float(x.laplace_after),
                "accepted_scale": float(x.accepted_scale),
                "accepted": bool(x.accepted),
                "proposed_delta": list(x.proposed_delta),
            }
            for x in result.iterations
        ],
        "transverse_vertex_leak_mm": transverse_leak,
        "zero_path_difference_construction": True,
        "uses_event_truth": False,
        "uses_empirical_mcs_scale": False,
    }
    return Fast12CoherentResult(
        initial_values=dict(local_values),
        updated_values=updated,
        updated_chart=profile_chart,
        coefficients_mean=np.ascontiguousarray(result.latent.coefficients),
        coefficients_covariance=np.ascontiguousarray(result.latent.covariance),
        charge_nll=float(result.latent.charge_nll),
        posterior_nll=float(result.latent.posterior_nll),
        laplace_nll=float(result.latent.laplace_nll),
        start_along_track_correction_mm=start_along,
        length_correction_mm=length_delta,
        downstream_endpoint_correction_mm=start_along + length_delta,
        latent_converged=bool(result.latent.converged),
        profile_converged=bool(result.converged),
        wall_s=float(time.perf_counter() - wall0),
        diagnostics=diagnostics,
    )
