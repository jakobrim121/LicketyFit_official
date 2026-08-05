"""Event-specific coherent Fermi--Eyges/KL reconstruction continuation.

This is the production-oriented replacement for the historical charge-only
FE-GEE stage.  It starts from the accepted coupled charge-plus-time straight-
track fit and leaves that successful optimizer untouched.

The continuation has two deliberately separated blocks:

1. Infer a shrunk event-specific FE/KL trajectory from the correlated PMT
   charge residuals with the physics-only Fermi--Eyges covariance.
2. Condition on that inferred coherent path and deconvolve the apparent
   straight-track anchor with the expected Fisher cross-information of the
   complete charge + conditional first-photoelectron likelihood.

The coherent optical derivative uses an arc-length-preserving FE path and the
finite-aperture line integral (FALI).  WCSim truth is never used by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np

from .fast_track_fit import predict_charge_and_process_jacobian
from .mcs_coupled_schur import CoupledCoherentEvaluator, DEFAULT_THETA_FD, _psd_inverse
from .mcs_pmt_scores import finite_difference_expected_fisher_blocks
from .mcs_process import build_raw_fe_kl_basis
from .track_parameterization import (
    TangentDirectionChart,
    attach_direction_components,
    reanchor_values,
)

LOCAL_NAMES = ("x0", "y0", "z0", "dir_u", "dir_v", "length", "t0")
GLOBAL_NAMES = ("x0", "y0", "z0", "cx", "cy", "cz", "length", "t0")


@dataclass
class CoherentFisherResult:
    initial_values: dict
    updated_values: dict
    updated_chart: TangentDirectionChart
    coefficients_mean: np.ndarray
    coefficients_covariance: np.ndarray
    raw_scaled_correction: np.ndarray
    applied_local_correction: np.ndarray
    trust_scale: float
    expected_fisher_theta: np.ndarray
    expected_fisher_cross: np.ndarray
    expected_fisher_latent_data: np.ndarray
    expected_fisher_latent_posterior: np.ndarray
    marginal_information_scaled: np.ndarray
    marginal_covariance_local: np.ndarray
    marginal_covariance_global: np.ndarray
    posterior_mean_contraction_mm: float
    posterior_expected_contraction_mm: float
    coherent_data_nll_initial: float
    coherent_data_nll_updated: float
    coherent_posterior_nll_initial: float
    coherent_posterior_nll_updated: float
    timing_normalization_max_error: float
    timing_information_fraction_theta: float
    timing_information_fraction_cross: float
    basis_explained_fraction: np.ndarray | None
    fixed_local_indices: tuple[int, ...]
    free_local_indices: tuple[int, ...]
    charge_mean_max_abs_difference: float
    wall_s: float
    diagnostics: dict

    def output_values(self) -> dict:
        return attach_direction_components(self.updated_values, chart=self.updated_chart)


def _local_theta(values: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(values[name]) for name in LOCAL_NAMES], dtype=np.float64)


def _fixed_indices(fixed_params: Mapping[str, float] | None) -> tuple[int, ...]:
    fixed = {} if fixed_params is None else fixed_params
    out: list[int] = []
    for i, name in enumerate(LOCAL_NAMES):
        if name in {"dir_u", "dir_v"}:
            if "direction" in fixed:
                out.append(i)
        elif name in fixed:
            out.append(i)
    return tuple(sorted(set(out)))


def _contraction_quadratic(emitter, coefficients, covariance) -> tuple[float, float, np.ndarray]:
    """Return posterior-mean and posterior-expected longitudinal contraction.

    For the arc-length path, ``dz_parallel/ds = sqrt(1-|q|^2)``.  To second
    order the contraction is ``1/2 int |q|^2 ds``.  The returned matrix is the
    exact trapezoidal quadratic form on the active FE grid.
    """
    sg, _shape, slope, _curv, _frac = build_raw_fe_kl_basis(emitter, 4, 81)
    A = np.zeros((4, 4), dtype=np.float64)
    for i, ds in enumerate(np.diff(sg)):
        A += 0.5 * float(ds) * (
            np.outer(slope[i], slope[i]) + np.outer(slope[i + 1], slope[i + 1])
        )
    block = np.zeros((8, 8), dtype=np.float64)
    block[:4, :4] = A
    block[4:, 4:] = A
    u = np.asarray(coefficients, dtype=np.float64).reshape(8)
    C = np.asarray(covariance, dtype=np.float64).reshape(8, 8)
    mean = 0.5 * float(u @ block @ u)
    expected = 0.5 * float(u @ block @ u + np.trace(block @ C))
    return mean, expected, block


def _physical_covariances(
    covariance_scaled: np.ndarray,
    theta_fd: np.ndarray,
    chart: TangentDirectionChart,
    pre_reanchor_values: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Map scaled local covariance to local physical and global coordinates."""
    scale = np.diag(np.asarray(theta_fd, dtype=np.float64))
    local = scale @ covariance_scaled @ scale
    local = 0.5 * (local + local.T)
    transform = np.zeros((8, 7), dtype=np.float64)
    transform[:3, :3] = np.eye(3)
    transform[3:6, 3:5] = chart.direction_jacobian(
        float(pre_reanchor_values.get("dir_u", 0.0)),
        float(pre_reanchor_values.get("dir_v", 0.0)),
    )
    transform[6, 5] = 1.0
    transform[7, 6] = 1.0
    global_cov = transform @ local @ transform.T
    global_cov = 0.5 * (global_cov + global_cov.T)
    return np.ascontiguousarray(local), np.ascontiguousarray(global_cov)


def _valid_step_scale(evaluator, theta0, delta_physical, initial_scale) -> float:
    """Shorten one coupled step only when detector/parameter bounds require it."""
    alpha = float(np.clip(initial_scale, 0.0, 1.0))
    if alpha <= 0.0:
        return 0.0
    if evaluator.model(theta0 + alpha * delta_physical) is not None:
        return alpha
    lo, hi = 0.0, alpha
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if evaluator.model(theta0 + mid * delta_physical) is None:
            hi = mid
        else:
            lo = mid
    return float(lo)


def run_coherent_fisher_update(
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
    modes_per_plane: int = 4,
    process_grid_points: int = 41,
    coherent_grid_points: int = 81,
    timing_quadrature_nodes: int = 32,
    direct_timing_bins: int = 1,
    latent_fd: float = 0.25,
    theta_fd: Sequence[float] = DEFAULT_THETA_FD,
    trust_box: float = 1.0,
    apply_expected_contraction: bool = False,
    straight_prediction_cache: dict | None = None,
) -> CoherentFisherResult:
    """Apply one event-specific coherent-path correction after the joint fit.

    ``trust_box=1`` means that the common coupled step is restricted to the
    finite-difference support used to establish the local response.  This is a
    numerical trust region, not an empirical physics scale.
    """
    wall0 = time.perf_counter()
    if int(modes_per_plane) != 4:
        raise ValueError("coherent Fisher continuation currently requires four modes per plane")
    theta0 = _local_theta(values)
    tfd = np.asarray(theta_fd, dtype=np.float64).reshape(7)
    if np.any(~np.isfinite(theta0)) or np.any(~np.isfinite(tfd)) or np.any(tfd <= 0.0):
        raise ValueError("invalid coherent-Fisher reference point or finite-difference steps")

    # The accepted joint-fit geometry is the reference.  Infer the event path
    # from charge residual correlations without replacing that estimator.
    mu, Ju, explained, base_context_emitter, base_context_timing = (
        predict_charge_and_process_jacobian(
            template_emitter,
            values=values,
            chart=chart,
            wcd=wcd,
            p_locations=p_locations,
            pmt_normals=pmt_normals,
            obs_pes=obs_pes,
            mpmt_types=mpmt_types,
            detector=detector,
            need_process_jacobian=True,
            process_modes_per_plane=int(modes_per_plane),
            process_grid_points=int(process_grid_points),
            need_times=True,
            return_emitter_context=True,
        )
    )
    if Ju is None or Ju.shape[1] != 8:
        raise RuntimeError("Emitter did not provide the expected eight FE/KL charge modes")
    mu = np.asarray(mu, dtype=np.float64)
    q = np.asarray(obs_pes, dtype=np.float64)
    D = np.maximum(mu, float(getattr(template_emitter, "charge_floor_pe", 1.0e-4)))
    Kcharge = np.eye(8) + Ju.T @ (Ju / D[:, None])
    Kcharge_inv, eval_kq, cutoff_kq, keep_kq = _psd_inverse(
        Kcharge, relative_floor=1.0e-12, absolute_floor=1.0e-12
    )
    score = Ju.T @ ((q - mu) / D)
    umean = Kcharge_inv @ score
    ucov = Kcharge_inv

    evaluator = CoupledCoherentEvaluator(
        template_emitter,
        wcd,
        pmt_model,
        p_locations,
        pmt_normals,
        q,
        obs_ts,
        chart=chart,
        detector=detector,
        mpmt_types=mpmt_types,
        n_modes=8,
        n_grid=int(coherent_grid_points),
        aperture_radius_mm=float(getattr(template_emitter, "primary_endpoint_aperture_radius_mm", 45.0)),
        path_field="fali",
        direct_timing_bins=int(direct_timing_bins),
        require_contained_track=True,
        length_limits=length_limits,
        t0_limits=t0_limits,
        straight_prediction_cache=straight_prediction_cache,
        precomputed_base_theta=theta0,
        precomputed_base_emitter=base_context_emitter,
        precomputed_base_pes=mu,
        precomputed_base_timing=base_context_timing,
    )
    blocks = finite_difference_expected_fisher_blocks(
        evaluator,
        theta0,
        theta_fd=tfd,
        latent_fd=float(latent_fd),
        timing_quadrature_nodes=int(timing_quadrature_nodes),
    )

    fixed = _fixed_indices(fixed_params)
    free = tuple(i for i in range(7) if i not in fixed)
    raw_scaled = np.zeros(7, dtype=np.float64)
    A = np.asarray(blocks.information_theta, dtype=np.float64)
    Ccross = np.asarray(blocks.information_cross, dtype=np.float64)
    if free:
        Af = A[np.ix_(free, free)]
        Cf = Ccross[np.ix_(free, tuple(range(8)))]
        Ainv, eval_A, cutoff_A, keep_A = _psd_inverse(
            Af, relative_floor=1.0e-10, absolute_floor=1.0e-12
        )
        raw_scaled[np.asarray(free, dtype=int)] = -(Ainv @ (Cf @ umean))
    else:
        eval_A = np.empty(0); cutoff_A = np.nan; keep_A = np.empty(0, dtype=bool)

    max_scaled = float(np.max(np.abs(raw_scaled))) if raw_scaled.size else 0.0
    trust = max(float(trust_box), 0.0)
    alpha = 1.0 if max_scaled <= trust or max_scaled <= 1.0e-30 else trust / max_scaled
    delta_physical = raw_scaled * tfd
    alpha = _valid_step_scale(evaluator, theta0, delta_physical, alpha)
    theta1 = theta0 + alpha * delta_physical

    # Marginal covariance is reported from the low-rank FE Fisher Schur
    # complement even though the central value uses the stable conditional
    # deconvolution above.
    Kfull = np.asarray(blocks.information_latent_posterior, dtype=np.float64)
    Kinv, eval_K, cutoff_K, keep_K = _psd_inverse(
        Kfull, relative_floor=1.0e-10, absolute_floor=1.0e-12
    )
    S = A - Ccross @ Kinv @ Ccross.T
    Scov = np.zeros((7, 7), dtype=np.float64)
    if free:
        Sf = S[np.ix_(free, free)]
        Sinv, eval_S, cutoff_S, keep_S = _psd_inverse(
            Sf, relative_floor=1.0e-10, absolute_floor=1.0e-12
        )
        Scov[np.ix_(free, free)] = Sinv
    else:
        eval_S = np.empty(0); cutoff_S = np.nan; keep_S = np.empty(0, dtype=bool)

    # Arc-length contraction is an analytically known second-order diagnostic.
    em = template_emitter.copy()
    em.start_coord = tuple(theta0[:3])
    direction0 = chart.direction(float(theta0[3]), float(theta0[4]))
    if direction0 is None:
        raise RuntimeError("invalid reference direction")
    em.direction = tuple(direction0)
    em.refresh_kinematics_from_length(float(theta0[5]))
    cmean, cexpected, contraction_matrix = _contraction_quadratic(em, umean, ucov)
    if apply_expected_contraction and 5 in free:
        theta_candidate = theta1.copy()
        theta_candidate[5] += cexpected
        if evaluator.model(theta_candidate) is not None:
            theta1 = theta_candidate

    before_reanchor = {name: float(theta1[i]) for i, name in enumerate(LOCAL_NAMES)}
    local_cov, global_cov = _physical_covariances(Scov, tfd, chart, before_reanchor)
    updated_values, updated_chart = reanchor_values(before_reanchor, chart)

    f0_data = float(evaluator(theta0, umean, include_prior=False))
    f1_data = float(evaluator(theta1, umean, include_prior=False))
    prior = 0.5 * float(umean @ umean)
    mean_difference = float(np.max(np.abs(
        evaluator.model(theta0).prediction(np.zeros(8))[0] - mu
    )))
    trace_theta = max(float(np.trace(blocks.information_theta)), 1.0e-300)
    norm_cross = max(float(np.linalg.norm(blocks.information_cross)), 1.0e-300)
    diagnostics = {
        "latent_posterior_eigenvalues_charge": np.asarray(eval_kq).tolist(),
        "latent_posterior_rank_charge": int(np.count_nonzero(keep_kq)),
        "latent_posterior_cutoff_charge": float(cutoff_kq),
        "theta_information_eigenvalues_free": np.asarray(eval_A).tolist(),
        "theta_information_rank_free": int(np.count_nonzero(keep_A)),
        "theta_information_cutoff_free": float(cutoff_A),
        "latent_information_eigenvalues": np.asarray(eval_K).tolist(),
        "latent_information_rank": int(np.count_nonzero(keep_K)),
        "latent_information_cutoff": float(cutoff_K),
        "marginal_information_eigenvalues_free": np.asarray(eval_S).tolist(),
        "marginal_information_rank_free": int(np.count_nonzero(keep_S)),
        "marginal_information_cutoff_free": float(cutoff_S),
        "coherent_model_build_count": int(evaluator.optical_model_build_count),
        "straight_prediction_build_count": int(evaluator.straight_prediction_build_count),
        "straight_prediction_external_cache_hits": int(evaluator.external_straight_cache_hits),
        "precomputed_base_context_uses": int(evaluator.precomputed_base_context_uses),
        "coherent_field_evaluation_count": int(evaluator.coherent_field_evaluation_count),
        "exact_coherent_nll_evaluation_count": int(evaluator.exact_evaluations),
        "contraction_quadratic_matrix": contraction_matrix.tolist(),
        "apply_expected_contraction": bool(apply_expected_contraction),
        "theta_fd": tfd.tolist(),
        "theta_fd_minus_fraction": np.asarray(
            blocks.theta_fd_minus_fraction, dtype=np.float64
        ).tolist(),
        "theta_fd_plus_fraction": np.asarray(
            blocks.theta_fd_plus_fraction, dtype=np.float64
        ).tolist(),
        "theta_fd_scheme": list(blocks.theta_fd_scheme),
        "latent_fd": float(latent_fd),
        "trust_box": float(trust_box),
        "timing_quadrature_nodes": int(timing_quadrature_nodes),
        "direct_timing_bins": int(direct_timing_bins),
        "coherent_grid_points": int(coherent_grid_points),
    }
    return CoherentFisherResult(
        initial_values=dict(values),
        updated_values=updated_values,
        updated_chart=updated_chart,
        coefficients_mean=np.ascontiguousarray(umean),
        coefficients_covariance=np.ascontiguousarray(ucov),
        raw_scaled_correction=np.ascontiguousarray(raw_scaled),
        applied_local_correction=np.ascontiguousarray(theta1 - theta0),
        trust_scale=float(alpha),
        expected_fisher_theta=np.ascontiguousarray(A),
        expected_fisher_cross=np.ascontiguousarray(Ccross),
        expected_fisher_latent_data=np.ascontiguousarray(blocks.information_latent_data),
        expected_fisher_latent_posterior=np.ascontiguousarray(Kfull),
        marginal_information_scaled=np.ascontiguousarray(S),
        marginal_covariance_local=local_cov,
        marginal_covariance_global=global_cov,
        posterior_mean_contraction_mm=float(cmean),
        posterior_expected_contraction_mm=float(cexpected),
        coherent_data_nll_initial=f0_data,
        coherent_data_nll_updated=f1_data,
        coherent_posterior_nll_initial=f0_data + prior,
        coherent_posterior_nll_updated=f1_data + prior,
        timing_normalization_max_error=float(np.max(np.abs(blocks.timing_normalization - 1.0))),
        timing_information_fraction_theta=float(np.trace(blocks.information_theta_timing) / trace_theta),
        timing_information_fraction_cross=float(np.linalg.norm(blocks.information_cross_timing) / norm_cross),
        basis_explained_fraction=None if explained is None else np.asarray(explained, dtype=np.float64),
        fixed_local_indices=fixed,
        free_local_indices=free,
        charge_mean_max_abs_difference=mean_difference,
        wall_s=float(time.perf_counter() - wall0),
        diagnostics=diagnostics,
    )
