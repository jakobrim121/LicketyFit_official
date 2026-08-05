"""Nonlinear coherent Fermi--Eyges continuation for clipped cosmic tracks.

This module generalizes the validated coherent-path/Fisher--Laplace machinery
from contained threshold tracks to all four cosmic topologies.  The fitted
remaining CSDA range and the visible in-water support are kept distinct:

* the complete range fixes the entry/start kinetic energy and scattering power;
* detector clipping fixes the path segment on which light can be emitted;
* the start hypothesis and stop/exit branch remain fixed during one local
  continuation, preventing non-differentiable topology hopping;
* boundary-entry lines are represented canonically by detector clipping, so the
  unobservable translation of a line along itself is not included in the
  update coordinates.

WCSim per-step truth is not used by this calculation.  It is only an external
validation target.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np

from .mcs_coherent_objective import FixedTrackCoherentMCSObjective
from .mcs_latent_profile import (
    ProfileIteration,
    profiled_charge_track_step_directions,
    solve_latent_charge_map,
)
from .cosmic_track_fit import resolve_range_clipped_track
from .track_parameterization import (
    TangentDirectionChart,
    attach_direction_components,
    reanchor_values,
)


@dataclass
class CosmicCoherentResult:
    initial_values: dict
    updated_values: dict
    updated_chart: TangentDirectionChart
    coefficients_mean: np.ndarray
    coefficients_covariance: np.ndarray
    initial_resolved: object
    updated_resolved: object
    charge_nll: float
    posterior_nll: float
    laplace_nll: float
    iterations: tuple[ProfileIteration, ...]
    latent_iterations: tuple
    converged: bool
    wall_s: float
    diagnostics: dict

    def output_values(self) -> dict:
        return attach_direction_components(self.updated_values, chart=self.updated_chart)


class CosmicCoherentEvaluator:
    """Cache coherent optical models for nearby clipped cosmic hypotheses.

    The seven local coordinates are ``(x0,y0,z0,dir_u,dir_v,R0,t0)``.  ``R0``
    is the complete remaining water-equivalent range.  Each candidate is
    clipped analytically to obtain the physical active-water start and visible
    path length before the coherent optical objective is constructed.
    """

    def __init__(
        self,
        emitter_template,
        wcd,
        pmt_model,
        pmt_positions,
        pmt_normals,
        obs_pes,
        obs_ts,
        *,
        chart: TangentDirectionChart,
        detector,
        range_lookup,
        starts_at_boundary: bool,
        expected_exits_detector: bool,
        mpmt_types=None,
        inset_mm: float = 0.5,
        range_limits: tuple[float, float] | None = None,
        t0_limits: tuple[float, float] | None = None,
        n_modes: int = 8,
        n_grid: int = 41,
        aperture_radius_mm: float = 45.0,
        sparse_receiver: bool = True,
        sparse_neighbor_radius_mm: float = 100.0,
    ):
        n_modes = int(n_modes)
        if n_modes <= 0 or n_modes % 2:
            raise ValueError("n_modes must be a positive even integer")
        self.n_modes = n_modes
        self.emitter_template = emitter_template
        self.wcd = wcd
        self.pmt_model = pmt_model
        self.pmt_positions = np.ascontiguousarray(pmt_positions, dtype=np.float64)
        self.pmt_normals = np.ascontiguousarray(pmt_normals, dtype=np.float64)
        self.obs_pes = np.asarray(obs_pes, dtype=np.float64)
        self.obs_ts = np.asarray(obs_ts, dtype=np.float64)
        self.chart = chart
        self.detector = detector
        self.range_lookup = range_lookup
        self.starts_at_boundary = bool(starts_at_boundary)
        self.expected_exits_detector = bool(expected_exits_detector)
        self.mpmt_types = mpmt_types
        self.inset_mm = float(inset_mm)
        self.range_limits = range_limits
        self.t0_limits = t0_limits
        self.n_grid = int(n_grid)
        self.aperture_radius_mm = float(aperture_radius_mm)
        self.sparse_receiver = bool(sparse_receiver)
        self.sparse_neighbor_radius_mm = float(sparse_neighbor_radius_mm)
        self._models: dict[tuple[float, ...], FixedTrackCoherentMCSObjective] = {}
        self._resolved: dict[tuple[float, ...], object] = {}
        self.invalid_evaluations = 0
        self.exact_evaluations = 0

    @staticmethod
    def _theta_array(theta: Sequence[float]) -> np.ndarray:
        return np.asarray(theta, dtype=np.float64).reshape(7)

    def _resolve(self, theta: Sequence[float]):
        arr = self._theta_array(theta)
        if np.any(~np.isfinite(arr)):
            return None
        reference = np.ascontiguousarray(arr[:3], dtype=np.float64)
        direction = self.chart.direction(float(arr[3]), float(arr[4]))
        full_range = float(arr[5])
        t0 = float(arr[6])
        if direction is None or not math.isfinite(full_range) or full_range <= 0.0:
            return None
        if not self.detector.contains(reference, tolerance_mm=1.0e-5):
            return None
        if self.range_limits is not None:
            lo, hi = map(float, self.range_limits)
            if full_range < lo or full_range > hi:
                return None
        if self.t0_limits is not None:
            lo, hi = map(float, self.t0_limits)
            if t0 < lo or t0 > hi:
                return None
        resolved = resolve_range_clipped_track(
            self.detector,
            reference,
            direction,
            full_range,
            starts_at_boundary=self.starts_at_boundary,
            inset_mm=self.inset_mm,
            tolerance_mm=1.0e-6,
        )
        if resolved is None:
            return None
        # One Fisher/Laplace continuation must stay on one smooth topology
        # branch.  Stop/exit arbitration is performed outside this module by the
        # exact range tournament.
        if bool(resolved.exits_detector) != self.expected_exits_detector:
            return None
        if resolved.visible_length_mm <= 0.0:
            return None
        return arr, resolved

    @staticmethod
    def _model_key(arr: np.ndarray, resolved) -> tuple[float, ...]:
        return tuple(
            np.round(
                np.concatenate(
                    (
                        np.asarray(resolved.start, dtype=np.float64),
                        np.asarray(resolved.direction, dtype=np.float64),
                        np.asarray(
                            [resolved.visible_length_mm, resolved.full_range_mm],
                            dtype=np.float64,
                        ),
                    )
                ),
                10,
            )
        )

    def resolved_track(self, theta: Sequence[float]):
        valid = self._resolve(theta)
        return None if valid is None else valid[1]

    def model(self, theta: Sequence[float]) -> FixedTrackCoherentMCSObjective | None:
        valid = self._resolve(theta)
        if valid is None:
            self.invalid_evaluations += 1
            return None
        arr, resolved = valid
        key = self._model_key(arr, resolved)
        cached = self._models.get(key)
        if cached is not None:
            return cached
        try:
            ke0 = float(self.range_lookup.range_mm_to_energy(resolved.full_range_mm))
            if not math.isfinite(ke0) or ke0 <= 0.0:
                raise ValueError("invalid kinetic energy from full range")
            emitter = self.emitter_template.copy()
            # The coherent path is the MCS representation.  Never combine it
            # with the historical deterministic cone broadening.
            emitter.enable_primary_mcs = False
            emitter.primary_mcs_process_modes_per_plane = self.n_modes // 2
            emitter.primary_mcs_process_grid_points = self.n_grid
            model = FixedTrackCoherentMCSObjective(
                emitter,
                self.wcd,
                self.pmt_model,
                self.pmt_positions,
                self.pmt_normals,
                self.obs_pes,
                self.obs_ts,
                vertex=resolved.start,
                direction=resolved.direction,
                length=resolved.visible_length_mm,
                full_range_mm=resolved.full_range_mm,
                initial_kinetic_energy_mev=ke0,
                t0=0.0,
                mpmt_types=self.mpmt_types,
                n_grid=self.n_grid,
                aperture_radius_mm=self.aperture_radius_mm,
                path_field="fali",
                direct_timing_bins=1,
                sparse_receiver=self.sparse_receiver,
                sparse_neighbor_radius_mm=self.sparse_neighbor_radius_mm,
                charge_only=True,
            )
        except Exception:
            self.invalid_evaluations += 1
            return None
        self._models[key] = model
        self._resolved[key] = resolved
        return model

    def __call__(
        self,
        theta: Sequence[float],
        coefficients: Sequence[float],
        *,
        include_prior: bool = True,
    ) -> float:
        model = self.model(theta)
        if model is None:
            return float("inf")
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        if np.any(~np.isfinite(u)):
            return float("inf")
        self.exact_evaluations += 1
        value = float(model.charge_data_nll(u))
        if include_prior:
            value += 0.5 * float(u @ u)
        return value if math.isfinite(value) else float("inf")

    @property
    def optical_model_build_count(self) -> int:
        return len(self._models)

    @property
    def coherent_field_evaluation_count(self) -> int:
        return int(sum(model.curved_evaluations for model in self._models.values()))


def run_cosmic_coherent_profile_update(
    template_emitter,
    *,
    values: Mapping[str, float],
    chart: TangentDirectionChart,
    detector,
    range_lookup,
    wcd,
    pmt_model,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    obs_ts: np.ndarray,
    starts_at_boundary: bool,
    mpmt_types=None,
    inset_mm: float = 0.5,
    range_limits: tuple[float, float] | None = None,
    t0_limits: tuple[float, float] | None = None,
    modes_per_plane: int = 4,
    grid_points: int = 41,
    latent_fd: float = 0.20,
    latent_max_iterations: int = 4,
    candidate_latent_max_iterations: int = 2,
    track_cycles: int = 2,
    transverse_step_mm: float = 12.0,
    longitudinal_step_mm: float = 12.0,
    direction_step: float = 0.010,
    range_step_mm: float = 30.0,
    latent_trust_max_component: float = 1.0,
    track_trust_max_scaled_component: float = 1.0,
    sparse_neighbor_radius_mm: float = 100.0,
) -> CosmicCoherentResult:
    """Profile a nonlinear coherent FE path and nearby cosmic track coordinates."""
    wall0 = time.perf_counter()
    local_values, local_chart = reanchor_values(values, chart)
    theta = np.asarray(
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
    initial_resolved = resolve_range_clipped_track(
        detector,
        theta[:3],
        local_chart.anchor,
        theta[5],
        starts_at_boundary=bool(starts_at_boundary),
        inset_mm=float(inset_mm),
        tolerance_mm=1.0e-6,
    )
    if initial_resolved is None:
        raise ValueError("initial cosmic track cannot be resolved")

    evaluator = CosmicCoherentEvaluator(
        template_emitter,
        wcd,
        pmt_model,
        p_locations,
        pmt_normals,
        obs_pes,
        obs_ts,
        chart=local_chart,
        detector=detector,
        range_lookup=range_lookup,
        starts_at_boundary=bool(starts_at_boundary),
        expected_exits_detector=bool(initial_resolved.exits_detector),
        mpmt_types=mpmt_types,
        inset_mm=float(inset_mm),
        range_limits=range_limits,
        t0_limits=t0_limits,
        n_modes=2 * int(modes_per_plane),
        n_grid=int(grid_points),
        aperture_radius_mm=float(
            getattr(template_emitter, "primary_endpoint_aperture_radius_mm", 45.0)
        ),
        sparse_receiver=True,
        sparse_neighbor_radius_mm=float(sparse_neighbor_radius_mm),
    )
    model = evaluator.model(theta)
    if model is None:
        raise ValueError("initial coherent cosmic model is invalid")
    latent = solve_latent_charge_map(
        model,
        fd_step=float(latent_fd),
        max_iterations=int(latent_max_iterations),
        trust_max_component=float(latent_trust_max_component),
    )
    history: list[ProfileIteration] = []
    converged = False

    for cycle in range(max(0, int(track_cycles))):
        direction = local_chart.direction(float(theta[3]), float(theta[4]))
        if direction is None:
            break
        # Reuse the local chart's transverse frame.  The continuation begins
        # re-anchored at u=v=0 and trust regions keep it within the smooth chart.
        vectors: list[np.ndarray] = []
        steps: list[float] = []
        labels: list[str] = []
        if not starts_at_boundary:
            v = np.zeros(7, dtype=np.float64)
            v[:3] = direction
            vectors.append(v)
            steps.append(float(longitudinal_step_mm))
            labels.append("internal_start_along_track")
        for basis, label in ((local_chart.e1, "line_transverse_1"), (local_chart.e2, "line_transverse_2")):
            v = np.zeros(7, dtype=np.float64)
            v[:3] = basis
            vectors.append(v)
            steps.append(float(transverse_step_mm))
            labels.append(label)
        for index, label in ((3, "direction_tangent_1"), (4, "direction_tangent_2")):
            v = np.zeros(7, dtype=np.float64)
            v[index] = 1.0
            vectors.append(v)
            steps.append(float(direction_step))
            labels.append(label)
        v = np.zeros(7, dtype=np.float64)
        v[5] = 1.0
        vectors.append(v)
        steps.append(float(range_step_mm))
        labels.append("full_range")

        proposal = profiled_charge_track_step_directions(
            evaluator,
            theta,
            latent,
            coordinate_vectors=np.asarray(vectors),
            coordinate_steps=np.asarray(steps),
            coordinate_labels=labels,
            trust_max_scaled_component=float(track_trust_max_scaled_component),
        )
        if float(np.linalg.norm(proposal.delta_theta)) <= 1.0e-7:
            converged = True
            break
        base_theta = theta.copy()
        base_laplace = float(latent.laplace_nll)
        accepted = False
        accepted_scale = 0.0
        best_theta = theta
        best_latent = latent
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate_theta = base_theta + float(scale) * proposal.delta_theta
            candidate_model = evaluator.model(candidate_theta)
            if candidate_model is None:
                continue
            candidate_latent = solve_latent_charge_map(
                candidate_model,
                initial_coefficients=latent.coefficients,
                fd_step=float(latent_fd),
                max_iterations=int(candidate_latent_max_iterations),
                trust_max_component=float(latent_trust_max_component),
            )
            if candidate_latent.laplace_nll < base_laplace - 1.0e-8:
                accepted = True
                accepted_scale = float(scale)
                best_theta = candidate_theta
                best_latent = candidate_latent
                break
        history.append(
            ProfileIteration(
                cycle=int(cycle),
                theta_before=tuple(map(float, base_theta)),
                theta_after=tuple(map(float, best_theta)),
                laplace_before=float(base_laplace),
                laplace_after=float(best_latent.laplace_nll),
                accepted_scale=float(accepted_scale),
                accepted=bool(accepted),
                proposed_delta=tuple(map(float, proposal.delta_theta)),
            )
        )
        if not accepted:
            break
        theta = np.asarray(best_theta, dtype=np.float64)
        latent = best_latent
        if float(np.linalg.norm(accepted_scale * proposal.delta_theta)) <= 1.0e-3:
            converged = True
            break

    updated_direction = local_chart.direction(float(theta[3]), float(theta[4]))
    if updated_direction is None:
        raise RuntimeError("coherent update produced an invalid direction")
    updated_resolved = evaluator.resolved_track(theta)
    if updated_resolved is None:
        raise RuntimeError("coherent update produced an invalid clipped track")
    # Return the physical active-water onset.  For a boundary-entry track this
    # removes the arbitrary translation of the internal reference along the line.
    updated_chart = TangentDirectionChart.from_direction(updated_direction)
    updated_values = {
        "x0": float(updated_resolved.start[0]),
        "y0": float(updated_resolved.start[1]),
        "z0": float(updated_resolved.start[2]),
        "dir_u": 0.0,
        "dir_v": 0.0,
        "length": float(updated_resolved.full_range_mm),
        "t0": float(theta[6]),
    }
    diagnostics = {
        "implementation": "cosmic_clipped_nonlinear_fali_fisher_laplace_v1",
        "starts_at_boundary": bool(starts_at_boundary),
        "exits_detector": bool(initial_resolved.exits_detector),
        "initial_topology": str(initial_resolved.topology),
        "updated_topology": str(updated_resolved.topology),
        "modes_per_plane": int(modes_per_plane),
        "latent_dimension": int(2 * modes_per_plane),
        "grid_points": int(grid_points),
        "optical_model_build_count": int(evaluator.optical_model_build_count),
        "coherent_field_evaluation_count": int(evaluator.coherent_field_evaluation_count),
        "invalid_model_evaluation_count": int(evaluator.invalid_evaluations),
        "uses_event_truth": False,
        "uses_empirical_mcs_scale": False,
        "keeps_visible_length_and_full_range_distinct": True,
        "boundary_longitudinal_reference_removed": bool(starts_at_boundary),
    }
    return CosmicCoherentResult(
        initial_values=dict(local_values),
        updated_values=updated_values,
        updated_chart=updated_chart,
        coefficients_mean=np.ascontiguousarray(latent.coefficients),
        coefficients_covariance=np.ascontiguousarray(latent.covariance),
        initial_resolved=initial_resolved,
        updated_resolved=updated_resolved,
        charge_nll=float(latent.charge_nll),
        posterior_nll=float(latent.posterior_nll),
        laplace_nll=float(latent.laplace_nll),
        iterations=tuple(history),
        latent_iterations=tuple(latent.iterations),
        converged=bool(converged),
        wall_s=float(time.perf_counter() - wall0),
        diagnostics=diagnostics,
    )
