"""Fast analytic Fermi--Eyges process update for absorption tracks.

The ordinary absorption fit uses a straight, abruptly terminated Cherenkov
track with two independent longitudinal coordinates: ``length`` is the visible
support and ``full_range`` fixes the initial kinetic energy.  This module adds
the standard small-angle multiple-Coulomb-scattering process after that fit.

The process is completely analytic.  The Fermi--Eyges covariance is represented
by standardized Karhunen--Loeve coordinates supplied by :mod:`mcs_process`, and
the PMT working covariance is

    V = D + J_u J_u.T,

where ``D`` is the Poisson working covariance and ``J_u`` is the analytic
optical derivative with respect to the FE coordinates.  Woodbury inversion
keeps the calculation linear in the PMT count and cubic only in the 24-mode
latent state.  No sampled trajectory, WCSim template, event truth, or empirical
MCS scale enters the model.

The accepted charge-time fit remains the authority for vertex, direction,
initial energy, and event time.  Only the abrupt visible endpoint is updated;
this avoids replacing timing-constrained coordinates with a charge-only local
linearization.  A deterministic exact-optical trust-ratio line search protects
the one-step GEE update when the endpoint response is nonlinear.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .mcs_process import fermi_eyges_process_update, woodbury_apply
from .track_parameterization import (
    TangentDirectionChart,
    attach_direction_components,
    reanchor_values,
)


LOCAL_NAMES = (
    "x0", "y0", "z0", "dir_u", "dir_v", "length", "full_range"
)
GLOBAL_NAMES = (
    "x0", "y0", "z0", "cx", "cy", "cz", "visible_length", "full_range"
)
DEFAULT_FINITE_DIFFERENCE_STEPS = (
    1.0, 1.0, 1.0, 2.0e-4, 2.0e-4, 1.0, 1.0
)


@dataclass
class AbsorptionFEProcessResult:
    """Result of one standard analytic FE/GEE absorption continuation."""

    initial_values: dict[str, float]
    updated_values: dict[str, float]
    updated_chart: TangentDirectionChart
    process_posterior_mean: np.ndarray
    process_posterior_covariance: np.ndarray
    naive_covariance_local: np.ndarray
    robust_covariance_local: np.ndarray
    robust_covariance_global: np.ndarray
    raw_delta_local: np.ndarray
    applied_delta_local: np.ndarray
    physical_step_scale: float
    wall_s: float
    diagnostics: dict[str, object]
    applied: bool = True

    def output_values(self) -> dict[str, object]:
        values = attach_direction_components(
            self.updated_values, chart=self.updated_chart
        )
        values["visible_length"] = float(values["length"])
        values["full_range"] = float(values["full_range"])
        return values


def _canonical_values(values: Mapping[str, object]) -> dict[str, float]:
    """Return the seven local absorption coordinates with stable aliases."""
    out = {str(name): float(value) for name, value in values.items()}
    if "length" not in out:
        if "visible_length" not in out:
            raise KeyError("absorption values require visible_length or length")
        out["length"] = float(out["visible_length"])
    out["visible_length"] = float(out["length"])
    if "full_range" not in out:
        raise KeyError("absorption values require full_range")
    missing = [name for name in LOCAL_NAMES if name not in out]
    if missing:
        raise KeyError(f"missing absorption FE coordinates: {missing}")
    if any(not math.isfinite(float(out[name])) for name in LOCAL_NAMES):
        raise ValueError("absorption FE coordinates must be finite")
    return out


def _local_vector(values: Mapping[str, object]) -> np.ndarray:
    canonical = _canonical_values(values)
    return np.asarray([canonical[name] for name in LOCAL_NAMES], dtype=np.float64)


def _values_from_vector(
    base: Mapping[str, object], vector: Sequence[float]
) -> dict[str, float]:
    vec = np.asarray(vector, dtype=np.float64)
    if vec.shape != (len(LOCAL_NAMES),):
        raise ValueError("absorption FE local vector must have seven entries")
    out = {str(name): float(value) for name, value in base.items()}
    for name, value in zip(LOCAL_NAMES, vec):
        out[name] = float(value)
    out["visible_length"] = float(out["length"])
    return out


def _physical_state(
    values: Mapping[str, object],
    *,
    chart: TangentDirectionChart,
    detector,
    length_limits: tuple[float, float],
    full_range_limits: tuple[float, float] | None,
    max_tangent_radius: float = 2.0,
    endpoint_tolerance_mm: float = 1.0e-9,
) -> bool:
    """Check the coupled absorption and detector domain without clipping."""
    try:
        state = _canonical_values(values)
    except (KeyError, TypeError, ValueError):
        return False
    length = float(state["length"])
    full_range = float(state["full_range"])
    if not (float(length_limits[0]) <= length <= float(length_limits[1])):
        return False
    if full_range_limits is not None and not (
        float(full_range_limits[0]) <= full_range <= float(full_range_limits[1])
    ):
        return False
    if full_range <= 0.0 or length > full_range + float(endpoint_tolerance_mm):
        return False
    u = float(state["dir_u"])
    v = float(state["dir_v"])
    if u * u + v * v > float(max_tangent_radius) ** 2:
        return False
    direction = chart.direction(u, v)
    if direction is None:
        return False
    vertex = np.asarray(
        [state["x0"], state["y0"], state["z0"]], dtype=np.float64
    )
    return bool(detector.segment_contained(vertex, direction, length))


def _predict_charge(
    template_emitter,
    *,
    values: Mapping[str, object],
    chart: TangentDirectionChart,
    detector,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    range_to_energy: Callable[[float], float],
    mpmt_types=None,
    need_process_jacobian: bool = False,
    process_modes_per_plane: int = 12,
    process_grid_points: int = 41,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Evaluate the abrupt optical mean and optional analytic FE tangent."""
    state = _canonical_values(values)
    direction = chart.direction(state["dir_u"], state["dir_v"])
    if direction is None:
        raise RuntimeError("invalid absorption direction")
    vertex = np.asarray(
        [state["x0"], state["y0"], state["z0"]], dtype=np.float64
    )
    length = float(state["length"])
    full_range = float(state["full_range"])
    if not detector.segment_contained(vertex, direction, length):
        raise RuntimeError("absorption track is outside the detector")

    initial_ke = float(range_to_energy(full_range))
    if not math.isfinite(initial_ke) or initial_ke <= 0.0:
        raise RuntimeError("full_range did not map to a positive kinetic energy")

    emitter = template_emitter.copy()
    # The process Jacobian is the stochastic model.  Do not also broaden the
    # mean cone, which would double count MCS and obscure the zero-bend mean.
    emitter.enable_primary_mcs_smearing = False
    emitter.primary_mcs_model = "fermi_eyges_process"
    emitter.primary_mcs_process_modes_per_plane = int(process_modes_per_plane)
    emitter.primary_mcs_process_grid_points = int(process_grid_points)
    emitter.compute_primary_mcs_process_jacobian = bool(need_process_jacobian)
    emitter.configure_track_end(
        "abrupt", fixed_initial_KE=initial_ke, refresh=False
    )
    emitter.start_coord = tuple(float(x) for x in vertex)
    emitter.direction = tuple(float(x) for x in direction)
    # Charge is invariant under t0.  A fixed zero keeps this continuation
    # explicitly independent of the accepted timing coordinate.
    emitter.starting_time = 0.0
    refreshed_ke = float(emitter.refresh_kinematics_from_length(length))
    if not math.isclose(refreshed_ke, initial_ke, rel_tol=0.0, abs_tol=1.0e-9):
        raise RuntimeError("abrupt emitter changed the conditioned kinetic energy")
    represented_range = float(getattr(emitter, "range_to_threshold_mm", math.nan))
    if not math.isclose(
        represented_range, full_range, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise RuntimeError(
            "abrupt emitter range/energy lookup is inconsistent with full_range"
        )
    if not bool(emitter.visible_length_is_physical(tol_mm=1.0e-9)):
        raise RuntimeError("visible absorption length exceeds the conditioned range")

    emission = emitter.get_emission_points(p_locations, initial_ke)
    mu, _ = emitter.get_expected_pes_ts(
        wcd,
        emission,
        p_locations,
        pmt_normals,
        mpmt_types,
        obs_pes,
        need_times=False,
    )
    mean = np.asarray(mu, dtype=np.float64)
    if mean.ndim != 1 or mean.shape != np.asarray(obs_pes).shape:
        raise RuntimeError("absorption optical prediction has the wrong shape")
    if np.any(~np.isfinite(mean)) or np.any(mean < 0.0):
        raise RuntimeError("absorption optical prediction is not finite")
    process = getattr(emitter, "_last_mcs_charge_jacobian", None)
    explained = getattr(emitter, "_last_mcs_basis_explained_fraction", None)
    return (
        np.ascontiguousarray(mean, dtype=np.float64),
        None if process is None else np.ascontiguousarray(process, dtype=np.float64),
        None if explained is None else np.asarray(explained, dtype=np.float64),
    )


def _finite_difference_track_jacobian(
    template_emitter,
    *,
    values: Mapping[str, object],
    chart: TangentDirectionChart,
    detector,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    range_to_energy: Callable[[float], float],
    mpmt_types,
    derivative_indices: Sequence[int],
    steps: Sequence[float],
    length_limits: tuple[float, float],
    full_range_limits: tuple[float, float] | None,
    process_modes_per_plane: int,
    process_grid_points: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    int,
    list[str],
]:
    """Finite-difference track coordinates around one analytic FE tangent."""
    center = _canonical_values(values)
    base, process_jacobian, explained = _predict_charge(
        template_emitter,
        values=center,
        chart=chart,
        detector=detector,
        wcd=wcd,
        p_locations=p_locations,
        pmt_normals=pmt_normals,
        obs_pes=obs_pes,
        range_to_energy=range_to_energy,
        mpmt_types=mpmt_types,
        need_process_jacobian=True,
        process_modes_per_plane=process_modes_per_plane,
        process_grid_points=process_grid_points,
    )
    if process_jacobian is None:
        raise RuntimeError("Emitter did not return an analytic FE process Jacobian")
    prediction_count = 1
    indices = tuple(int(index) for index in derivative_indices)
    step_values = tuple(float(step) for step in steps)
    jacobian = np.empty((base.size, len(indices)), dtype=np.float64)
    schemes: list[str] = []

    def valid(trial: Mapping[str, object]) -> bool:
        return _physical_state(
            trial,
            chart=chart,
            detector=detector,
            length_limits=length_limits,
            full_range_limits=full_range_limits,
        )

    def predict(trial: Mapping[str, object]) -> np.ndarray:
        nonlocal prediction_count
        prediction, _, _ = _predict_charge(
            template_emitter,
            values=trial,
            chart=chart,
            detector=detector,
            wcd=wcd,
            p_locations=p_locations,
            pmt_normals=pmt_normals,
            obs_pes=obs_pes,
            range_to_energy=range_to_energy,
            mpmt_types=mpmt_types,
            need_process_jacobian=False,
            process_modes_per_plane=process_modes_per_plane,
            process_grid_points=process_grid_points,
        )
        prediction_count += 1
        return prediction

    for output_column, index in enumerate(indices):
        if index < 0 or index >= len(LOCAL_NAMES):
            raise IndexError("absorption FE derivative index is out of range")
        name = LOCAL_NAMES[index]
        nominal_step = step_values[index]
        derivative = None
        scheme = ""
        for shrink in (1.0, 0.5, 0.25, 0.1):
            h = nominal_step * float(shrink)
            plus = dict(center)
            minus = dict(center)
            plus[name] += h
            minus[name] -= h
            plus["visible_length"] = plus["length"]
            minus["visible_length"] = minus["length"]
            plus_valid = valid(plus)
            minus_valid = valid(minus)
            try:
                if plus_valid and minus_valid:
                    derivative = (predict(plus) - predict(minus)) / (2.0 * h)
                    scheme = "central"
                elif plus_valid:
                    derivative = (predict(plus) - base) / h
                    scheme = "forward"
                elif minus_valid:
                    derivative = (base - predict(minus)) / h
                    scheme = "backward"
            except (FloatingPointError, RuntimeError, ValueError):
                derivative = None
            if derivative is not None and np.all(np.isfinite(derivative)):
                break
            derivative = None
        if derivative is None:
            raise RuntimeError(f"could not evaluate absorption FE derivative for {name}")
        jacobian[:, output_column] = derivative
        schemes.append(scheme)
    return (
        base,
        np.ascontiguousarray(jacobian, dtype=np.float64),
        np.ascontiguousarray(process_jacobian, dtype=np.float64),
        explained,
        int(prediction_count),
        schemes,
    )


def _maximum_physical_scale(
    center: np.ndarray,
    delta: np.ndarray,
    *,
    base_values: Mapping[str, object],
    chart: TangentDirectionChart,
    detector,
    length_limits: tuple[float, float],
    full_range_limits: tuple[float, float] | None,
) -> float:
    """Return the largest scale in [0,1] on the proposed physical ray."""
    center = np.asarray(center, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)

    def physical(scale: float) -> bool:
        trial = _values_from_vector(base_values, center + float(scale) * delta)
        return _physical_state(
            trial,
            chart=chart,
            detector=detector,
            length_limits=length_limits,
            full_range_limits=full_range_limits,
        )

    if not physical(0.0):
        raise RuntimeError("accepted straight absorption point is not physical")
    if not np.any(delta):
        return 0.0
    if physical(1.0):
        return 1.0
    low = 0.0
    high = 1.0
    for _ in range(64):
        middle = 0.5 * (low + high)
        if physical(middle):
            low = middle
        else:
            high = middle
    return float(max(0.0, low * (1.0 - 1.0e-10)))


def _working_loss(
    expected: np.ndarray,
    observed: np.ndarray,
    *,
    diagonal_inverse: np.ndarray,
    process_jacobian: np.ndarray,
) -> float:
    residual = np.asarray(observed, dtype=np.float64) - np.asarray(
        expected, dtype=np.float64
    )
    return 0.5 * float(
        residual
        @ woodbury_apply(diagonal_inverse, process_jacobian, residual)
    )


def _linearization_trust_step(
    template_emitter,
    *,
    values: Mapping[str, object],
    center: np.ndarray,
    raw_delta: np.ndarray,
    physical_domain_scale: float,
    chart: TangentDirectionChart,
    detector,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    range_to_energy: Callable[[float], float],
    mpmt_types,
    expected_charge: np.ndarray,
    process_jacobian: np.ndarray,
    information: np.ndarray,
    score: np.ndarray,
    update_indices: Sequence[int],
    charge_floor_pe: float,
    process_modes_per_plane: int,
    process_grid_points: int,
    minimum_actual_to_predicted_ratio: float,
    maximum_backtracks: int,
) -> tuple[float, dict[str, object], int]:
    """Backtrack the local GEE proposal against the exact optical mean."""
    diagonal_inverse = 1.0 / np.maximum(
        np.asarray(expected_charge, dtype=np.float64), float(charge_floor_pe)
    )
    base_loss = _working_loss(
        expected_charge,
        obs_pes,
        diagonal_inverse=diagonal_inverse,
        process_jacobian=process_jacobian,
    )
    block_delta = np.asarray(
        [raw_delta[int(index)] for index in update_indices], dtype=np.float64
    )
    linear_gain = float(np.asarray(score, dtype=np.float64) @ block_delta)
    quadratic_cost = float(
        block_delta @ np.asarray(information, dtype=np.float64) @ block_delta
    )
    trials: list[dict[str, object]] = []
    prediction_count = 0
    accepted_scale = 0.0
    accepted_actual = 0.0
    accepted_predicted = 0.0
    accepted_ratio = math.nan

    if physical_domain_scale > 0.0 and np.any(np.abs(block_delta) > 0.0):
        for backtrack in range(int(maximum_backtracks) + 1):
            scale = float(physical_domain_scale) * (0.5 ** backtrack)
            predicted = scale * linear_gain - 0.5 * scale * scale * quadratic_cost
            trial_values = _values_from_vector(values, center + scale * raw_delta)
            actual = -math.inf
            try:
                trial_mu, _, _ = _predict_charge(
                    template_emitter,
                    values=trial_values,
                    chart=chart,
                    detector=detector,
                    wcd=wcd,
                    p_locations=p_locations,
                    pmt_normals=pmt_normals,
                    obs_pes=obs_pes,
                    range_to_energy=range_to_energy,
                    mpmt_types=mpmt_types,
                    need_process_jacobian=False,
                    process_modes_per_plane=process_modes_per_plane,
                    process_grid_points=process_grid_points,
                )
                prediction_count += 1
                actual = base_loss - _working_loss(
                    trial_mu,
                    obs_pes,
                    diagonal_inverse=diagonal_inverse,
                    process_jacobian=process_jacobian,
                )
            except (FloatingPointError, RuntimeError, ValueError):
                pass
            ratio = actual / predicted if predicted > 0.0 else -math.inf
            accepted = bool(
                math.isfinite(actual)
                and predicted > 0.0
                and actual > 0.0
                and ratio >= float(minimum_actual_to_predicted_ratio)
            )
            trials.append({
                "backtrack": int(backtrack),
                "scale": float(scale),
                "actual_improvement": float(actual),
                "predicted_improvement": float(predicted),
                "actual_to_predicted_ratio": float(ratio),
                "accepted": accepted,
            })
            if accepted:
                accepted_scale = float(scale)
                accepted_actual = float(actual)
                accepted_predicted = float(predicted)
                accepted_ratio = float(ratio)
                break

    diagnostics = {
        "method": "fixed_FE_covariance_exact_optical_trust_ratio",
        "minimum_actual_to_predicted_ratio": float(
            minimum_actual_to_predicted_ratio
        ),
        "maximum_backtracks": int(maximum_backtracks),
        "trial_count": int(len(trials)),
        "accepted": bool(accepted_scale > 0.0),
        "accepted_scale": float(accepted_scale),
        "accepted_actual_improvement": float(accepted_actual),
        "accepted_predicted_improvement": float(accepted_predicted),
        "accepted_actual_to_predicted_ratio": float(accepted_ratio),
        "trials": trials,
    }
    return float(accepted_scale), diagnostics, int(prediction_count)


def run_absorption_fermi_eyges_update(
    template_emitter,
    *,
    values: Mapping[str, object],
    chart: TangentDirectionChart,
    detector,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    range_to_energy: Callable[[float], float],
    pmt_model=None,
    mpmt_types=None,
    fixed_params: Mapping[str, object] | None = None,
    update_indices: Sequence[int] = (5,),
    process_modes_per_plane: int = 12,
    process_grid_points: int = 41,
    xyz_step_mm: float = 1.0,
    direction_step: float = 2.0e-4,
    length_step_mm: float = 1.0,
    full_range_step_mm: float = 1.0,
    length_limits: tuple[float, float] = (0.0, 3000.0),
    full_range_limits: tuple[float, float] | None = None,
    charge_floor_pe: float = 1.0e-4,
    minimum_actual_to_predicted_ratio: float = 0.25,
    maximum_backtracks: int = 12,
) -> AbsorptionFEProcessResult:
    """Apply the standard analytic FE process model to an absorption fit.

    The public production route supplies ``update_indices=(5,)``.  The argument
    remains explicit for mathematical tests and controlled diagnostics, but
    fixed coordinates are always removed and the driver never updates the
    accepted vertex/direction/energy block in this charge-only continuation.
    """
    del pmt_model  # Accepted for a stable driver-facing signature.
    wall0 = time.perf_counter()
    initial = _canonical_values(values)
    fixed = {} if fixed_params is None else dict(fixed_params)
    requested = tuple(dict.fromkeys(int(index) for index in update_indices))
    if any(index < 0 or index >= len(LOCAL_NAMES) for index in requested):
        raise IndexError("absorption FE update index is out of range")

    def coordinate_is_fixed(index: int) -> bool:
        name = LOCAL_NAMES[index]
        if name in {"dir_u", "dir_v"}:
            return "direction" in fixed or name in fixed
        if name == "length":
            return "length" in fixed or "visible_length" in fixed
        return name in fixed

    free_indices = tuple(
        index for index in requested if not coordinate_is_fixed(index)
    )
    steps = (
        float(xyz_step_mm), float(xyz_step_mm), float(xyz_step_mm),
        float(direction_step), float(direction_step),
        float(length_step_mm), float(full_range_step_mm),
    )
    mu, track_jacobian, process_jacobian, explained, prediction_count, schemes = (
        _finite_difference_track_jacobian(
            template_emitter,
            values=initial,
            chart=chart,
            detector=detector,
            wcd=wcd,
            p_locations=np.asarray(p_locations, dtype=np.float64),
            pmt_normals=np.asarray(pmt_normals, dtype=np.float64),
            obs_pes=np.asarray(obs_pes, dtype=np.float64),
            range_to_energy=range_to_energy,
            mpmt_types=mpmt_types,
            derivative_indices=free_indices,
            steps=steps,
            length_limits=length_limits,
            full_range_limits=full_range_limits,
            process_modes_per_plane=process_modes_per_plane,
            process_grid_points=process_grid_points,
        )
    )
    gee = fermi_eyges_process_update(
        mu,
        np.asarray(obs_pes, dtype=np.float64),
        track_jacobian,
        process_jacobian,
        update_indices=tuple(range(len(free_indices))),
        charge_floor_pe=float(charge_floor_pe),
    )

    center = _local_vector(initial)
    raw_delta = np.zeros(len(LOCAL_NAMES), dtype=np.float64)
    for output_index, local_index in enumerate(free_indices):
        raw_delta[local_index] = float(gee["delta_block"][output_index])
    physical_domain_scale = _maximum_physical_scale(
        center,
        raw_delta,
        base_values=initial,
        chart=chart,
        detector=detector,
        length_limits=length_limits,
        full_range_limits=full_range_limits,
    )
    accepted_scale, trust, trust_predictions = _linearization_trust_step(
        template_emitter,
        values=initial,
        center=center,
        raw_delta=raw_delta,
        physical_domain_scale=physical_domain_scale,
        chart=chart,
        detector=detector,
        wcd=wcd,
        p_locations=np.asarray(p_locations, dtype=np.float64),
        pmt_normals=np.asarray(pmt_normals, dtype=np.float64),
        obs_pes=np.asarray(obs_pes, dtype=np.float64),
        range_to_energy=range_to_energy,
        mpmt_types=mpmt_types,
        expected_charge=mu,
        process_jacobian=process_jacobian,
        information=np.asarray(gee["information_block"], dtype=np.float64),
        score=np.asarray(gee["score_block"], dtype=np.float64),
        update_indices=free_indices,
        charge_floor_pe=charge_floor_pe,
        process_modes_per_plane=process_modes_per_plane,
        process_grid_points=process_grid_points,
        minimum_actual_to_predicted_ratio=minimum_actual_to_predicted_ratio,
        maximum_backtracks=maximum_backtracks,
    )
    prediction_count += int(trust_predictions)
    applied_delta = accepted_scale * raw_delta
    updated_before_reanchor = _values_from_vector(
        initial, center + applied_delta
    )
    updated, updated_chart = reanchor_values(updated_before_reanchor, chart)
    updated = _canonical_values(updated)

    naive = np.full((len(LOCAL_NAMES), len(LOCAL_NAMES)), np.nan, dtype=np.float64)
    robust = np.full_like(naive, np.nan)
    naive_small = np.asarray(gee["naive_covariance"], dtype=np.float64)
    robust_small = np.asarray(gee["robust_covariance"], dtype=np.float64)
    for i, local_i in enumerate(free_indices):
        for j, local_j in enumerate(free_indices):
            naive[local_i, local_j] = naive_small[i, j]
            robust[local_i, local_j] = robust_small[i, j]

    transform = np.zeros((len(GLOBAL_NAMES), len(LOCAL_NAMES)), dtype=np.float64)
    transform[:3, :3] = np.eye(3)
    transform[3:6, 3:5] = chart.direction_jacobian(
        initial["dir_u"], initial["dir_v"]
    )
    transform[6, 5] = 1.0
    transform[7, 6] = 1.0
    robust_global = transform @ np.nan_to_num(robust, nan=0.0) @ transform.T
    robust_global = 0.5 * (robust_global + robust_global.T)

    score = np.asarray(gee["score_block"], dtype=np.float64)
    information = np.asarray(gee["information_block"], dtype=np.float64)
    block_raw = np.asarray(
        [raw_delta[index] for index in free_indices], dtype=np.float64
    )
    score_after = score - information @ (accepted_scale * block_raw)
    diagnostics: dict[str, object] = {
        "implementation": "standard_analytic_fermi_eyges_process_v1",
        "inference": "poisson_working_covariance_fisher_GEE",
        "track_end_mode": "abrupt",
        "visible_length_independent_of_full_range": True,
        "free_local_indices": list(free_indices),
        "free_local_names": [LOCAL_NAMES[index] for index in free_indices],
        "finite_difference_schemes": list(schemes),
        "finite_difference_steps": list(steps),
        "charge_prediction_count": int(prediction_count),
        "process_modes_per_plane": int(process_modes_per_plane),
        "process_dimension": int(process_jacobian.shape[1]),
        "process_grid_points": int(process_grid_points),
        "basis_explained_fraction": (
            None if explained is None else np.asarray(explained).tolist()
        ),
        "mean_prediction_pe": float(np.mean(mu)),
        "process_response_frobenius_norm": float(np.linalg.norm(process_jacobian)),
        "working_score_norm_before": float(np.linalg.norm(score)),
        "working_score_norm_after_linear_step": float(np.linalg.norm(score_after)),
        "working_quadratic_improvement": float(
            trust["accepted_predicted_improvement"]
        ),
        "information_condition_raw": float(gee["information_condition_raw"]),
        "information_condition_scaled": float(gee["information_condition_scaled"]),
        "physical_domain_step_scale": float(physical_domain_scale),
        "physical_step_scale": float(accepted_scale),
        "linearization_trust": trust,
        "uses_event_truth": False,
        "uses_random_sampling": False,
        "uses_empirical_mcs_scale": False,
    }
    return AbsorptionFEProcessResult(
        initial_values=dict(initial),
        updated_values=dict(updated),
        updated_chart=updated_chart,
        process_posterior_mean=np.asarray(
            gee["process_posterior_mean"], dtype=np.float64
        ),
        process_posterior_covariance=np.asarray(
            gee["process_posterior_covariance"], dtype=np.float64
        ),
        naive_covariance_local=naive,
        robust_covariance_local=robust,
        robust_covariance_global=np.asarray(robust_global, dtype=np.float64),
        raw_delta_local=raw_delta,
        applied_delta_local=applied_delta,
        physical_step_scale=float(accepted_scale),
        wall_s=float(time.perf_counter() - wall0),
        diagnostics=diagnostics,
        applied=True,
    )


__all__ = [
    "AbsorptionFEProcessResult",
    "DEFAULT_FINITE_DIFFERENCE_STEPS",
    "GLOBAL_NAMES",
    "LOCAL_NAMES",
    "run_absorption_fermi_eyges_update",
]
