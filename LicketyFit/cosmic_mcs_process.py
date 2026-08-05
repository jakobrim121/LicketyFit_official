"""Geometry-clipped Fermi--Eyges process update for cosmic tracks.

The ordinary contained-track FE helpers assume that the fitted ``length`` is
both the visible detector segment and the remaining range to Cherenkov
threshold.  Cosmic mode deliberately separates those quantities.  This module
keeps the public cosmic coordinates

    (line reference, direction chart, full range)

while resolving the physical active-water onset and visible segment at every
prediction.  It then evaluates the same parameter-free Fermi--Eyges/KL process
Jacobian used by the contained fitter and performs a low-rank correlated-charge
update in track-aligned coordinates.

No WCSim trajectory, event-truth template, fitted MCS scale, or empirical
smearing width enters the calculation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np

from .cosmic_track_fit import (
    AlignedPriorObjective,
    ConvexDetectorVolume,
    resolve_range_clipped_track,
)
from .track_parameterization import TangentDirectionChart, reanchor_values
from .mcs_process import fermi_eyges_process_update, woodbury_apply

ALIGNED_NAMES = (
    "longitudinal",
    "transverse_1",
    "transverse_2",
    "dir_u",
    "dir_v",
    "full_range",
)


@dataclass
class CosmicFEPrediction:
    expected_charge: np.ndarray
    process_jacobian: np.ndarray | None
    basis_explained_fraction: np.ndarray | None
    resolved_track: object
    emitter: object


@dataclass
class CosmicFEUpdateResult:
    initial_values: dict[str, float]
    updated_values: dict[str, float]
    updated_chart: TangentDirectionChart
    start_hypothesis: str
    initial_topology: str
    final_topology: str
    initial_quasi_nll: float
    final_quasi_nll: float
    applied_aligned_delta: np.ndarray
    raw_aligned_delta: np.ndarray
    active_aligned_indices: tuple[int, ...]
    process_posterior_mean: np.ndarray
    process_posterior_covariance: np.ndarray
    robust_covariance_aligned: np.ndarray
    robust_covariance_global: np.ndarray
    basis_explained_fraction: np.ndarray | None
    cycles: tuple[dict[str, object], ...]
    prediction_count: int
    wall_s: float

    def output_values(self) -> dict[str, float]:
        out = dict(self.updated_values)
        direction = self.updated_chart.direction(
            float(out.get("dir_u", 0.0)), float(out.get("dir_v", 0.0))
        )
        if direction is None:
            direction = np.full(3, np.nan, dtype=np.float64)
        out["cx"], out["cy"], out["cz"] = map(float, direction)
        out["direction_chart"] = self.updated_chart.as_metadata()
        return out


def _normalized_direction_and_basis(
    values: Mapping[str, float], chart: TangentDirectionChart
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = chart.direction(
        float(values.get("dir_u", 0.0)), float(values.get("dir_v", 0.0))
    )
    if direction is None:
        raise ValueError("non-finite cosmic direction")
    # The active chart basis is fixed during one finite-difference/update cycle.
    # Project its two tangent axes into the current tangent plane, then
    # orthonormalize.  This remains stable if the accepted fit has nonzero u/v.
    e1 = np.asarray(chart.e1, dtype=np.float64)
    e1 = e1 - direction * float(np.dot(direction, e1))
    n1 = float(np.linalg.norm(e1))
    if n1 <= 1.0e-14:
        e1 = np.asarray(chart.e2, dtype=np.float64)
        e1 = e1 - direction * float(np.dot(direction, e1))
        n1 = float(np.linalg.norm(e1))
    e1 /= max(n1, 1.0e-30)
    e2 = np.cross(direction, e1)
    e2 /= max(float(np.linalg.norm(e2)), 1.0e-30)
    return (
        np.ascontiguousarray(direction, dtype=np.float64),
        np.ascontiguousarray(e1, dtype=np.float64),
        np.ascontiguousarray(e2, dtype=np.float64),
    )


def apply_aligned_delta(
    values: Mapping[str, float],
    chart: TangentDirectionChart,
    aligned_delta: Sequence[float],
    *,
    scale: float = 1.0,
) -> dict[str, float]:
    """Apply a track-aligned cosmic-coordinate displacement.

    ``length`` is the cosmic full-range coordinate.  The visible detector
    length is intentionally *not* updated independently; it is recomputed from
    the detector clipping at the trial point.
    """
    delta = np.asarray(aligned_delta, dtype=np.float64).reshape(6)
    alpha = float(scale)
    direction, e1, e2 = _normalized_direction_and_basis(values, chart)
    vertex_delta = (
        delta[0] * direction + delta[1] * e1 + delta[2] * e2
    ) * alpha
    out = {k: float(v) for k, v in values.items() if np.isscalar(v)}
    out["x0"] = float(values["x0"]) + float(vertex_delta[0])
    out["y0"] = float(values["y0"]) + float(vertex_delta[1])
    out["z0"] = float(values["z0"]) + float(vertex_delta[2])
    out["dir_u"] = float(values.get("dir_u", 0.0)) + alpha * float(delta[3])
    out["dir_v"] = float(values.get("dir_v", 0.0)) + alpha * float(delta[4])
    out["length"] = float(values["length"]) + alpha * float(delta[5])
    if "t0" in values:
        out["t0"] = float(values["t0"])
    return out


def _topology_label(start_hypothesis: str, exits_detector: bool) -> str:
    prefix = "boundary_entry" if start_hypothesis == "boundary_entry" else "internal_start"
    suffix = "boundary_exit" if bool(exits_detector) else "internal_stop"
    return f"{prefix}_{suffix}"


def predict_cosmic_charge_and_process_jacobian(
    template_emitter,
    *,
    values: Mapping[str, float],
    chart: TangentDirectionChart,
    start_hypothesis: str,
    detector: ConvexDetectorVolume,
    range_lookup,
    particle_threshold_mev: float,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    mpmt_types=None,
    boundary_clip_inset_mm: float = 0.5,
    need_process_jacobian: bool = False,
    process_modes_per_plane: int = 4,
    process_grid_points: int = 41,
    expected_exit_state: bool | None = None,
) -> CosmicFEPrediction | None:
    """Evaluate the sharp cosmic charge mean and optional FE/KL Jacobian."""
    hypothesis = str(start_hypothesis)
    if hypothesis not in {"internal_start", "boundary_entry"}:
        raise ValueError(f"unknown cosmic start hypothesis {hypothesis!r}")
    direction = chart.direction(
        float(values.get("dir_u", 0.0)), float(values.get("dir_v", 0.0))
    )
    if direction is None:
        return None
    reference = np.asarray(
        [values["x0"], values["y0"], values["z0"]], dtype=np.float64
    )
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        return None
    if not detector.contains(reference):
        return None
    full_range = float(values["length"])
    if not math.isfinite(full_range) or full_range <= 0.0:
        return None
    max_range = float(range_lookup.overall_distances_mm[-1])
    if full_range > max_range:
        return None
    resolved = resolve_range_clipped_track(
        detector,
        reference,
        direction,
        full_range,
        starts_at_boundary=(hypothesis == "boundary_entry"),
        inset_mm=float(boundary_clip_inset_mm),
    )
    if resolved is None:
        return None
    if expected_exit_state is not None and bool(resolved.exits_detector) != bool(expected_exit_state):
        return None
    ke0 = float(range_lookup.range_mm_to_energy(full_range))
    if not math.isfinite(ke0) or ke0 <= float(particle_threshold_mev):
        return None

    emitter = template_emitter.copy()
    emitter.enable_primary_mcs_smearing = False
    emitter.primary_mcs_model = "fermi_eyges_process"
    emitter.primary_mcs_process_modes_per_plane = int(process_modes_per_plane)
    emitter.primary_mcs_process_grid_points = int(process_grid_points)
    emitter.compute_primary_mcs_process_jacobian = bool(need_process_jacobian)
    emitter.start_coord = tuple(float(x) for x in resolved.start)
    emitter.direction = tuple(float(x) for x in resolved.direction)
    emitter.starting_time = 0.0
    emitter.track_end_mode = "abrupt"
    emitter.fixed_initial_KE = ke0
    init_ke = emitter.refresh_kinematics_from_length(
        float(resolved.visible_length_mm)
    )
    if hasattr(emitter, "visible_length_is_physical"):
        if not emitter.visible_length_is_physical():
            return None
    elif getattr(emitter, "last_visible_length_exceeds_range", False):
        return None
    emission = emitter.get_emission_points(p_locations, init_ke)
    mu, _ = emitter.get_expected_pes_ts(
        wcd,
        emission,
        p_locations,
        pmt_normals,
        mpmt_types,
        np.asarray(obs_pes, dtype=np.float64),
        need_times=False,
    )
    mu = np.asarray(mu, dtype=np.float64)
    if mu.shape != np.asarray(obs_pes).shape or np.any(~np.isfinite(mu)) or np.any(mu < 0.0):
        return None
    Ju = getattr(emitter, "_last_mcs_charge_jacobian", None)
    frac = getattr(emitter, "_last_mcs_basis_explained_fraction", None)
    if need_process_jacobian:
        if Ju is None:
            return None
        Ju = np.asarray(Ju, dtype=np.float64)
        if Ju.ndim != 2 or Ju.shape[0] != mu.size or np.any(~np.isfinite(Ju)):
            return None
    else:
        Ju = None
    return CosmicFEPrediction(
        np.ascontiguousarray(mu),
        None if Ju is None else np.ascontiguousarray(Ju),
        None if frac is None else np.ascontiguousarray(np.asarray(frac, dtype=np.float64)),
        resolved,
        emitter,
    )


def make_cosmic_aligned_prior_objective(
    base_objective,
    center: Mapping[str, float],
    aligned_covariance: np.ndarray,
    aligned_indices: Sequence[int] = (1, 2, 3, 4),
) -> AlignedPriorObjective:
    """Attach an FE Gaussian prior expressed directly in aligned coordinates.

    The contained-track helper transforms a Cartesian/local covariance into the
    aligned basis.  Cosmic FE already reports its robust covariance in that
    aligned basis, so applying the transform a second time would be incorrect.
    """
    covariance = np.asarray(aligned_covariance, dtype=np.float64)
    if covariance.shape != (6, 6):
        raise ValueError("aligned_covariance must have shape (6, 6)")
    requested = tuple(int(i) for i in aligned_indices)
    indices = tuple(
        i for i in requested
        if 0 <= i < 6
        and np.isfinite(covariance[i, i])
        and covariance[i, i] > 0.0
    )
    if not indices:
        return AlignedPriorObjective(
            base_objective,
            {k: float(v) for k, v in center.items() if np.isscalar(v)},
            (),
            np.empty((0, 0), dtype=np.float64),
        )
    sub = np.asarray(covariance[np.ix_(indices, indices)], dtype=np.float64)
    # Drop rows with missing cross-covariances, which can occur when a user-fixed
    # coordinate was excluded from the FE finite-difference block.
    keep = [
        j for j in range(len(indices))
        if np.all(np.isfinite(sub[j])) and np.all(np.isfinite(sub[:, j]))
    ]
    indices = tuple(indices[j] for j in keep)
    sub = covariance[np.ix_(indices, indices)] if indices else np.empty((0, 0))
    if not indices:
        precision = np.empty((0, 0), dtype=np.float64)
    else:
        sym = 0.5 * (sub + sub.T)
        scale = np.sqrt(np.maximum(np.diag(sym), 1.0e-30))
        corr = sym / scale[:, None] / scale[None, :]
        eig, vec = np.linalg.eigh(0.5 * (corr + corr.T))
        cutoff = max(float(np.max(eig)), 1.0) * 1.0e-10
        inv = np.zeros_like(eig)
        np.divide(1.0, eig, out=inv, where=(eig > cutoff))
        precision = (vec @ np.diag(inv) @ vec.T) / scale[:, None] / scale[None, :]
        precision = np.ascontiguousarray(0.5 * (precision + precision.T))
    return AlignedPriorObjective(
        base_objective,
        {k: float(v) for k, v in center.items() if np.isscalar(v)},
        indices,
        precision,
    )


def process_quasi_nll(
    expected_charge: np.ndarray,
    observed_charge: np.ndarray,
    process_jacobian: np.ndarray,
    *,
    charge_floor_pe: float = 1.0e-4,
) -> float:
    """Local Gaussian-process charge marginal used for safeguarded relinearization.

    The working covariance is ``V = diag(mu) + Ju Ju^T``.  The determinant is
    evaluated with the matrix determinant lemma, so only the low-rank mode
    matrix is factorized.
    """
    mu = np.asarray(expected_charge, dtype=np.float64)
    q = np.asarray(observed_charge, dtype=np.float64)
    Ju = np.asarray(process_jacobian, dtype=np.float64)
    if mu.ndim != 1 or q.shape != mu.shape or Ju.ndim != 2 or Ju.shape[0] != mu.size:
        return math.inf
    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(q)) or np.any(~np.isfinite(Ju)):
        return math.inf
    D = np.maximum(mu, float(charge_floor_pe))
    di = 1.0 / D
    residual = q - mu
    vinv_r = woodbury_apply(di, Ju, residual)
    H = np.eye(Ju.shape[1], dtype=np.float64) + Ju.T @ (di[:, None] * Ju)
    sign, logdet_h = np.linalg.slogdet(0.5 * (H + H.T))
    if sign <= 0.0 or not math.isfinite(float(logdet_h)):
        return math.inf
    value = 0.5 * float(residual @ vinv_r)
    value += 0.5 * float(np.sum(np.log(D)))
    value += 0.5 * float(logdet_h)
    return float(value)


def finite_difference_cosmic_charge_jacobian(
    template_emitter,
    *,
    values: Mapping[str, float],
    chart: TangentDirectionChart,
    start_hypothesis: str,
    detector: ConvexDetectorVolume,
    range_lookup,
    particle_threshold_mev: float,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    mpmt_types=None,
    boundary_clip_inset_mm: float = 0.5,
    active_aligned_indices: Sequence[int] | None = None,
    longitudinal_step_mm: float = 2.0,
    transverse_step_mm: float = 2.0,
    direction_step: float = 2.0e-4,
    full_range_step_mm: float = 2.0,
    process_modes_per_plane: int = 4,
    process_grid_points: int = 41,
) -> tuple[CosmicFEPrediction, np.ndarray, tuple[int, ...], int]:
    base = predict_cosmic_charge_and_process_jacobian(
        template_emitter,
        values=values,
        chart=chart,
        start_hypothesis=start_hypothesis,
        detector=detector,
        range_lookup=range_lookup,
        particle_threshold_mev=particle_threshold_mev,
        wcd=wcd,
        p_locations=p_locations,
        pmt_normals=pmt_normals,
        obs_pes=obs_pes,
        mpmt_types=mpmt_types,
        boundary_clip_inset_mm=boundary_clip_inset_mm,
        need_process_jacobian=True,
        process_modes_per_plane=process_modes_per_plane,
        process_grid_points=process_grid_points,
    )
    if base is None or base.process_jacobian is None:
        raise RuntimeError("invalid cosmic FE reference prediction")
    current_exit = bool(base.resolved_track.exits_detector)
    if active_aligned_indices is None:
        active = (1, 2, 3, 4, 5) if start_hypothesis == "boundary_entry" else (0, 1, 2, 3, 4, 5)
    else:
        active = tuple(int(i) for i in active_aligned_indices)
    if len(set(active)) != len(active) or any(i < 0 or i >= 6 for i in active):
        raise ValueError("active_aligned_indices must be unique members of [0,5]")
    steps = (
        float(longitudinal_step_mm),
        float(transverse_step_mm),
        float(transverse_step_mm),
        float(direction_step),
        float(direction_step),
        float(full_range_step_mm),
    )
    J = np.empty((base.expected_charge.size, len(active)), dtype=np.float64)
    prediction_count = 1

    def predict(trial: Mapping[str, float]):
        nonlocal prediction_count
        result = predict_cosmic_charge_and_process_jacobian(
            template_emitter,
            values=trial,
            chart=chart,
            start_hypothesis=start_hypothesis,
            detector=detector,
            range_lookup=range_lookup,
            particle_threshold_mev=particle_threshold_mev,
            wcd=wcd,
            p_locations=p_locations,
            pmt_normals=pmt_normals,
            obs_pes=obs_pes,
            mpmt_types=mpmt_types,
            boundary_clip_inset_mm=boundary_clip_inset_mm,
            need_process_jacobian=False,
            process_modes_per_plane=process_modes_per_plane,
            process_grid_points=process_grid_points,
            expected_exit_state=current_exit,
        )
        prediction_count += 1
        return None if result is None else result.expected_charge

    for out_col, index in enumerate(active):
        derivative = None
        for shrink in (1.0, 0.5, 0.25, 0.1):
            h = steps[index] * shrink
            displacement = np.zeros(6, dtype=np.float64)
            displacement[index] = h
            plus = apply_aligned_delta(values, chart, displacement)
            minus = apply_aligned_delta(values, chart, -displacement)
            mup = predict(plus)
            mum = predict(minus)
            if mup is not None and mum is not None:
                derivative = (mup - mum) / (2.0 * h)
            elif mup is not None:
                derivative = (mup - base.expected_charge) / h
            elif mum is not None:
                derivative = (base.expected_charge - mum) / h
            if derivative is not None and np.all(np.isfinite(derivative)):
                break
            derivative = None
        if derivative is None:
            raise RuntimeError(f"could not evaluate cosmic FE derivative for {ALIGNED_NAMES[index]}")
        J[:, out_col] = derivative
    return base, np.ascontiguousarray(J), active, int(prediction_count)


def _global_covariance_from_aligned(
    covariance_aligned: np.ndarray,
    values: Mapping[str, float],
    chart: TangentDirectionChart,
) -> np.ndarray:
    cov = np.asarray(covariance_aligned, dtype=np.float64).reshape(6, 6)
    direction, e1, e2 = _normalized_direction_and_basis(values, chart)
    transform = np.zeros((7, 6), dtype=np.float64)
    transform[:3, :3] = np.column_stack((direction, e1, e2))
    transform[3:6, 3:5] = chart.direction_jacobian(
        float(values.get("dir_u", 0.0)), float(values.get("dir_v", 0.0))
    )
    transform[6, 5] = 1.0
    out = transform @ np.nan_to_num(cov, nan=0.0) @ transform.T
    return np.ascontiguousarray(0.5 * (out + out.T), dtype=np.float64)


def run_cosmic_fermi_eyges_update(
    template_emitter,
    *,
    values: Mapping[str, float],
    chart: TangentDirectionChart,
    start_hypothesis: str,
    detector: ConvexDetectorVolume,
    range_lookup,
    particle_threshold_mev: float,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    mpmt_types=None,
    boundary_clip_inset_mm: float = 0.5,
    process_modes_per_plane: int = 4,
    process_grid_points: int = 41,
    longitudinal_step_mm: float = 2.0,
    transverse_step_mm: float = 2.0,
    direction_step: float = 2.0e-4,
    full_range_step_mm: float = 2.0,
    max_cycles: int = 2,
    safeguard_scales: Sequence[float] = (1.0, 0.5, 0.25, 0.125),
    charge_floor_pe: float = 1.0e-4,
) -> CosmicFEUpdateResult:
    """Apply a safeguarded low-rank FE process update to a cosmic track.

    The discrete start and stop/exit topology are held fixed during one update.
    A symmetric finite difference automatically becomes one-sided when a trial
    crosses the topology boundary.  Candidate steps are compared with the
    analytic low-rank process marginal, not with event truth or an ad-hoc MCS
    scale.
    """
    wall0 = time.perf_counter()
    current = {k: float(v) for k, v in values.items() if np.isscalar(v)}
    current_chart = chart
    prediction_count = 0
    cycles: list[dict[str, object]] = []
    total_applied = np.zeros(6, dtype=np.float64)
    last_raw = np.zeros(6, dtype=np.float64)
    last_active: tuple[int, ...] = ()
    last_process_mean = np.empty(0, dtype=np.float64)
    last_process_cov = np.empty((0, 0), dtype=np.float64)
    last_robust = np.full((6, 6), np.nan, dtype=np.float64)
    last_frac = None
    initial_q = math.nan
    initial_topology = "unknown"

    for cycle_index in range(max(1, int(max_cycles))):
        base, Jtheta, active, count = finite_difference_cosmic_charge_jacobian(
            template_emitter,
            values=current,
            chart=current_chart,
            start_hypothesis=start_hypothesis,
            detector=detector,
            range_lookup=range_lookup,
            particle_threshold_mev=particle_threshold_mev,
            wcd=wcd,
            p_locations=p_locations,
            pmt_normals=pmt_normals,
            obs_pes=obs_pes,
            mpmt_types=mpmt_types,
            boundary_clip_inset_mm=boundary_clip_inset_mm,
            longitudinal_step_mm=longitudinal_step_mm,
            transverse_step_mm=transverse_step_mm,
            direction_step=direction_step,
            full_range_step_mm=full_range_step_mm,
            process_modes_per_plane=process_modes_per_plane,
            process_grid_points=process_grid_points,
        )
        prediction_count += int(count)
        Ju = np.asarray(base.process_jacobian, dtype=np.float64)
        q0 = process_quasi_nll(
            base.expected_charge, obs_pes, Ju, charge_floor_pe=charge_floor_pe
        )
        if cycle_index == 0:
            initial_q = float(q0)
            initial_topology = _topology_label(
                start_hypothesis, bool(base.resolved_track.exits_detector)
            )
        update = fermi_eyges_process_update(
            base.expected_charge,
            np.asarray(obs_pes, dtype=np.float64),
            Jtheta,
            Ju,
            update_indices=tuple(range(len(active))),
            charge_floor_pe=float(charge_floor_pe),
        )
        raw_full = np.zeros(6, dtype=np.float64)
        for local, aligned_index in enumerate(active):
            raw_full[int(aligned_index)] = float(update["delta_block"][local])
        # Boundary-entry line references have no physical longitudinal degree.
        if start_hypothesis == "boundary_entry":
            raw_full[0] = 0.0
        last_raw = raw_full.copy()
        last_active = active
        last_process_mean = np.asarray(update["process_posterior_mean"], dtype=np.float64)
        last_process_cov = np.asarray(update["process_posterior_covariance"], dtype=np.float64)
        last_frac = base.basis_explained_fraction
        robust_small = np.asarray(update["robust_covariance"], dtype=np.float64)
        last_robust = np.full((6, 6), np.nan, dtype=np.float64)
        for ia, a in enumerate(active):
            for ib, b in enumerate(active):
                last_robust[a, b] = robust_small[ia, ib]

        best_values = current
        best_chart = current_chart
        best_q = float(q0)
        best_scale = 0.0
        candidate_rows = []
        for scale in safeguard_scales:
            alpha = float(scale)
            if not math.isfinite(alpha) or alpha <= 0.0:
                continue
            trial = apply_aligned_delta(current, current_chart, raw_full, scale=alpha)
            candidate = predict_cosmic_charge_and_process_jacobian(
                template_emitter,
                values=trial,
                chart=current_chart,
                start_hypothesis=start_hypothesis,
                detector=detector,
                range_lookup=range_lookup,
                particle_threshold_mev=particle_threshold_mev,
                wcd=wcd,
                p_locations=p_locations,
                pmt_normals=pmt_normals,
                obs_pes=obs_pes,
                mpmt_types=mpmt_types,
                boundary_clip_inset_mm=boundary_clip_inset_mm,
                need_process_jacobian=True,
                process_modes_per_plane=process_modes_per_plane,
                process_grid_points=process_grid_points,
                expected_exit_state=bool(base.resolved_track.exits_detector),
            )
            prediction_count += 1
            if candidate is None or candidate.process_jacobian is None:
                candidate_rows.append({"scale": alpha, "valid": False})
                continue
            qtrial = process_quasi_nll(
                candidate.expected_charge,
                obs_pes,
                candidate.process_jacobian,
                charge_floor_pe=charge_floor_pe,
            )
            candidate_rows.append({"scale": alpha, "valid": True, "quasi_nll": float(qtrial)})
            if math.isfinite(qtrial) and qtrial < best_q - 1.0e-10:
                best_q = float(qtrial)
                best_values = trial
                best_scale = alpha
        applied = best_scale * raw_full
        cycles.append({
            "cycle": int(cycle_index),
            "topology": _topology_label(start_hypothesis, bool(base.resolved_track.exits_detector)),
            "quasi_nll_before": float(q0),
            "quasi_nll_after": float(best_q),
            "raw_aligned_delta": raw_full.tolist(),
            "accepted_scale": float(best_scale),
            "applied_aligned_delta": applied.tolist(),
            "active_aligned_indices": list(active),
            "active_aligned_names": [ALIGNED_NAMES[i] for i in active],
            "information_condition_scaled": float(update.get("information_condition_scaled", math.nan)),
            "candidates": candidate_rows,
        })
        if best_scale <= 0.0:
            break
        total_applied += applied
        # Make the accepted direction the origin of the next regular chart.
        current, current_chart = reanchor_values(best_values, current_chart)
        current = {k: float(v) for k, v in current.items() if np.isscalar(v)}
        if float(q0) - float(best_q) < 1.0e-5:
            break

    final_prediction = predict_cosmic_charge_and_process_jacobian(
        template_emitter,
        values=current,
        chart=current_chart,
        start_hypothesis=start_hypothesis,
        detector=detector,
        range_lookup=range_lookup,
        particle_threshold_mev=particle_threshold_mev,
        wcd=wcd,
        p_locations=p_locations,
        pmt_normals=pmt_normals,
        obs_pes=obs_pes,
        mpmt_types=mpmt_types,
        boundary_clip_inset_mm=boundary_clip_inset_mm,
        need_process_jacobian=True,
        process_modes_per_plane=process_modes_per_plane,
        process_grid_points=process_grid_points,
    )
    prediction_count += 1
    if final_prediction is None or final_prediction.process_jacobian is None:
        raise RuntimeError("cosmic FE update ended at an invalid prediction")
    final_q = process_quasi_nll(
        final_prediction.expected_charge,
        obs_pes,
        final_prediction.process_jacobian,
        charge_floor_pe=charge_floor_pe,
    )
    final_topology = _topology_label(
        start_hypothesis, bool(final_prediction.resolved_track.exits_detector)
    )
    global_cov = _global_covariance_from_aligned(last_robust, current, current_chart)
    return CosmicFEUpdateResult(
        initial_values={k: float(v) for k, v in values.items() if np.isscalar(v)},
        updated_values=current,
        updated_chart=current_chart,
        start_hypothesis=str(start_hypothesis),
        initial_topology=str(initial_topology),
        final_topology=str(final_topology),
        initial_quasi_nll=float(initial_q),
        final_quasi_nll=float(final_q),
        applied_aligned_delta=np.ascontiguousarray(total_applied),
        raw_aligned_delta=np.ascontiguousarray(last_raw),
        active_aligned_indices=tuple(last_active),
        process_posterior_mean=np.ascontiguousarray(last_process_mean),
        process_posterior_covariance=np.ascontiguousarray(last_process_cov),
        robust_covariance_aligned=np.ascontiguousarray(last_robust),
        robust_covariance_global=np.ascontiguousarray(global_cov),
        basis_explained_fraction=(
            None if last_frac is None else np.ascontiguousarray(last_frac)
        ),
        cycles=tuple(cycles),
        prediction_count=int(prediction_count),
        wall_s=float(time.perf_counter() - wall0),
    )
