"""Local coupled-track evaluator for the coherent Fermi--Eyges continuation.

Only a small interface is required by the current production-oriented test
branch: construct the complete coherent optical objective at a proposed global
track, evaluate it for a set of standardized KL coefficients, and provide a
stable positive-semidefinite inverse.  The accepted straight-track optimizer is
not replaced by this module.

No event truth or WCSim-derived response enters this calculation.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Sequence

import numpy as np

from .mcs_coherent_objective import FixedTrackCoherentMCSObjective
from .mcs_curved_path import MCSPhysicalDomainError
from .mcs_process import (
    parallel_transport_transverse_basis,
    stable_transverse_basis,
)

THETA_NAMES = ("x0", "y0", "z0", "dir_u", "dir_v", "length", "t0")
DEFAULT_THETA_FD = np.asarray(
    [5.0, 5.0, 5.0, 1.0e-3, 1.0e-3, 10.0, 2.0e-2], dtype=np.float64
)


def _psd_inverse(
    matrix: np.ndarray,
    *,
    relative_floor: float = 1.0e-10,
    absolute_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return a symmetric pseudoinverse using a physical PSD rank cut.

    Parameters are first symmetrized.  Only positive eigenvalues above
    ``max(absolute_floor, relative_floor * lambda_max)`` are retained.  The
    eigenvalues, cutoff, and retained-rank mask are returned for diagnostics.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("PSD inverse requires a square matrix")
    a = 0.5 * (a + a.T)
    if np.any(~np.isfinite(a)):
        raise ValueError("PSD inverse received a non-finite matrix")
    if a.size == 0:
        return a.copy(), np.empty(0), float(absolute_floor), np.empty(0, dtype=bool)
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    maximum = max(float(np.max(eigenvalues)), 0.0)
    cutoff = max(float(absolute_floor), float(relative_floor) * maximum)
    keep = eigenvalues > cutoff
    inv = np.zeros_like(a)
    if np.any(keep):
        v = eigenvectors[:, keep]
        inv = (v / eigenvalues[keep][None, :]) @ v.T
    inv = 0.5 * (inv + inv.T)
    return np.ascontiguousarray(inv), eigenvalues, float(cutoff), keep


class CoupledCoherentEvaluator:
    """Cache the complete coherent optical model for nearby global tracks.

    The seven local coordinates are ``(x0,y0,z0,dir_u,dir_v,L,t0)``.  The
    expensive optical object is cached by the first six coordinates; ``t0`` is
    applied analytically by shifting the complete timing prediction.
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
        chart,
        detector,
        mpmt_types=None,
        n_modes: int = 8,
        n_grid: int = 81,
        aperture_radius_mm: float = 45.0,
        path_field: str = "fali",
        direct_timing_bins: int = 1,
        require_contained_track: bool = True,
        length_limits: tuple[float, float] | None = None,
        t0_limits: tuple[float, float] | None = None,
        straight_prediction_cache: dict | None = None,
        precomputed_base_theta: Sequence[float] | None = None,
        precomputed_base_emitter=None,
        precomputed_base_pes=None,
        precomputed_base_timing=None,
        sparse_receiver: bool = True,
        sparse_neighbor_radius_mm: float = 100.0,
        charge_only: bool = False,
        max_cached_models: int = 32,
        track_end_mode: str = "threshold",
        full_range_mm: float | None = None,
        initial_kinetic_energy_mev: float | None = None,
    ):
        n_modes = int(n_modes)
        if n_modes <= 0 or n_modes % 2 != 0:
            raise ValueError(
                "coherent continuation requires a positive even latent dimension"
            )
        self.n_modes = n_modes
        if int(direct_timing_bins) != 1:
            raise NotImplementedError(
                "the supplied test branch validates one coherent direct timing node; "
                "MCS_COHERENT_DIRECT_TIMING_BINS must remain 1"
            )
        self.emitter_template = emitter_template
        self.wcd = wcd
        self.pmt_model = pmt_model
        self.pmt_positions = np.ascontiguousarray(pmt_positions, dtype=np.float64)
        self.pmt_normals = np.ascontiguousarray(pmt_normals, dtype=np.float64)
        self.obs_pes = np.asarray(obs_pes, dtype=np.float64)
        self.obs_ts = np.asarray(obs_ts, dtype=np.float64)
        self.chart = chart
        self.detector = detector
        self.mpmt_types = mpmt_types
        self.n_grid = int(n_grid)
        self.aperture_radius_mm = float(aperture_radius_mm)
        self.path_field = str(path_field)
        self.direct_timing_bins = int(direct_timing_bins)
        self.sparse_receiver = bool(sparse_receiver)
        self.sparse_neighbor_radius_mm = float(sparse_neighbor_radius_mm)
        self.charge_only = bool(charge_only)
        self.max_cached_models = int(max_cached_models)
        if self.max_cached_models <= 0:
            raise ValueError("max_cached_models must be positive")
        self.require_contained_track = bool(require_contained_track)
        end_mode = str(track_end_mode).strip().lower().replace("-", "_")
        if end_mode in {"absorption", "abrupt", "truncated"}:
            end_mode = "abrupt"
        elif end_mode in {"threshold", "full_length", "range"}:
            end_mode = "threshold"
        else:
            raise ValueError(
                "coupled coherent track_end_mode must be threshold or abrupt"
            )
        self.track_end_mode = end_mode
        self.full_range_mm = (
            None if full_range_mm is None else float(full_range_mm)
        )
        self.initial_kinetic_energy_mev = (
            None
            if initial_kinetic_energy_mev is None
            else float(initial_kinetic_energy_mev)
        )
        if self.track_end_mode == "abrupt":
            if (
                self.full_range_mm is None
                or not math.isfinite(self.full_range_mm)
                or self.full_range_mm <= 0.0
                or self.initial_kinetic_energy_mev is None
                or not math.isfinite(self.initial_kinetic_energy_mev)
                or self.initial_kinetic_energy_mev <= 0.0
            ):
                raise ValueError(
                    "abrupt coherent evaluation requires positive full range "
                    "and initial kinetic energy"
                )
        self.length_limits = length_limits
        self.t0_limits = t0_limits
        self._models: OrderedDict[
            tuple[float, ...], FixedTrackCoherentMCSObjective
        ] = OrderedDict()
        self._straight_predictions: dict[tuple[float, ...], tuple] = {}
        self._external_straight_predictions = (
            {} if straight_prediction_cache is None else straight_prediction_cache
        )
        self.external_straight_cache_hits = 0
        self._precomputed_base_key = (
            None if precomputed_base_theta is None
            else self._geometry_key(self._theta_array(precomputed_base_theta))
        )
        self._precomputed_base_emitter = precomputed_base_emitter
        self._precomputed_base_pes = precomputed_base_pes
        self._precomputed_base_timing = precomputed_base_timing
        self.precomputed_base_context_uses = 0
        self.exact_evaluations = 0
        self.invalid_evaluations = 0
        self.physical_domain_rejections = 0
        self.model_build_failures = 0
        self.model_build_count = 0
        self.model_cache_evictions = 0
        self.curved_path_rejections = 0
        self._evicted_coherent_field_evaluations = 0
        anchor_direction, anchor_e1, anchor_e2 = stable_transverse_basis(
            self.chart.anchor
        )
        self._frame_anchor_direction = anchor_direction
        self._frame_anchor_e1 = anchor_e1
        self._frame_anchor_e2 = anchor_e2

    @staticmethod
    def _theta_array(theta: Sequence[float]) -> np.ndarray:
        out = np.asarray(theta, dtype=np.float64).reshape(7)
        return out

    @staticmethod
    def _geometry_key(theta: np.ndarray) -> tuple[float, ...]:
        # The FE grid is numerical quadrature on [0,L], not support for the
        # fitted geometry.  Keep the native float64 hypothesis in the cache key
        # so neither length nor any other fitted coordinate is quantized.
        return tuple(map(float, theta[:6]))

    def _valid_theta(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        if np.any(~np.isfinite(theta)):
            return None
        vertex = np.ascontiguousarray(theta[:3], dtype=np.float64)
        direction = self.chart.direction(float(theta[3]), float(theta[4]))
        length = float(theta[5])
        t0 = float(theta[6])
        if direction is None or not math.isfinite(length) or length < 0.0 or not math.isfinite(t0):
            return None
        if self.length_limits is not None:
            lo, hi = map(float, self.length_limits)
            if length < lo or length > hi:
                return None
        if (
            self.track_end_mode == "abrupt"
            and self.full_range_mm is not None
            and length > float(self.full_range_mm) + 1.0e-7
        ):
            return None
        if self.t0_limits is not None:
            lo, hi = map(float, self.t0_limits)
            if t0 < lo or t0 > hi:
                return None
        if not self.detector.contains(vertex):
            return None
        if self.require_contained_track and not self.detector.segment_contained(vertex, direction, length):
            return None
        return vertex, direction, length, t0

    def model(self, theta: Sequence[float]) -> FixedTrackCoherentMCSObjective | None:
        arr = self._theta_array(theta)
        valid = self._valid_theta(arr)
        if valid is None:
            self.invalid_evaluations += 1
            return None
        key = self._geometry_key(arr)
        cached = self._models.get(key)
        if cached is not None:
            self._models.move_to_end(key)
            return cached
        vertex, direction, length, _t0 = valid
        use_precomputed = (
            self._precomputed_base_key is not None
            and key == self._precomputed_base_key
            and self._precomputed_base_emitter is not None
        )
        try:
            _transported_direction, transported_e1, transported_e2 = (
                parallel_transport_transverse_basis(
                    self._frame_anchor_direction,
                    self._frame_anchor_e1,
                    self._frame_anchor_e2,
                    direction,
                )
            )
            model = FixedTrackCoherentMCSObjective(
                self.emitter_template,
                self.wcd,
                self.pmt_model,
                self.pmt_positions,
                self.pmt_normals,
                self.obs_pes,
                self.obs_ts,
                vertex=vertex,
                direction=direction,
                length=length,
                full_range_mm=(
                    self.full_range_mm
                    if self.track_end_mode == "abrupt"
                    else None
                ),
                initial_kinetic_energy_mev=(
                    self.initial_kinetic_energy_mev
                    if self.track_end_mode == "abrupt"
                    else None
                ),
                track_end_mode=self.track_end_mode,
                t0=0.0,
                mpmt_types=self.mpmt_types,
                n_grid=self.n_grid,
                aperture_radius_mm=self.aperture_radius_mm,
                path_field=self.path_field,
                direct_timing_bins=self.direct_timing_bins,
                precomputed_base_emitter=(
                    self._precomputed_base_emitter if use_precomputed else None
                ),
                precomputed_base_pes=(
                    self._precomputed_base_pes if use_precomputed else None
                ),
                precomputed_base_timing=(
                    self._precomputed_base_timing if use_precomputed else None
                ),
                sparse_receiver=self.sparse_receiver,
                sparse_neighbor_radius_mm=self.sparse_neighbor_radius_mm,
                charge_only=self.charge_only,
                transverse_basis=(transported_e1, transported_e2),
                path_validator=self._curved_path_contained,
            )
            if use_precomputed:
                self.precomputed_base_context_uses += 1
        except MCSPhysicalDomainError:
            self.physical_domain_rejections += 1
            self.invalid_evaluations += 1
            return None
        except (FloatingPointError, OverflowError):
            # Numerical overflow at a legitimate finite-difference proposal is
            # an invalid model point, but it is tracked separately from ordinary
            # detector-boundary rejection.
            self.model_build_failures += 1
            self.invalid_evaluations += 1
            return None
        self.model_build_count += 1
        if len(self._models) >= self.max_cached_models:
            _old_key, old_model = self._models.popitem(last=False)
            self._evicted_coherent_field_evaluations += int(
                old_model.curved_evaluations
            )
            self.model_cache_evictions += 1
        self._models[key] = model
        return model

    def _curved_path_contained(self, path) -> bool:
        """Validate every node of the represented coherent polyline.

        Detector volumes used by this evaluator are convex.  Consequently,
        contained path nodes also certify every linear segment between adjacent
        nodes, which is the same interpolation used by the FALI quadrature.
        """
        try:
            positions = np.asarray(path["position"], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            self.curved_path_rejections += 1
            return False
        contains_many = getattr(self.detector, "contains_many", None)
        if contains_many is not None:
            valid = bool(contains_many(positions, tolerance_mm=1.0e-6))
        else:
            valid = bool(
                positions.ndim == 2
                and positions.shape[1:] == (3,)
                and positions.shape[0] > 0
                and np.all(np.isfinite(positions))
                and all(
                    self.detector.contains(point, tolerance_mm=1.0e-6)
                    for point in positions
                )
            )
        if not valid:
            self.curved_path_rejections += 1
        return valid

    def straight_prediction(
        self, theta: Sequence[float], *, raise_on_model_error: bool = False
    ):
        """Return the accepted straight optical prediction for one geometry.

        Global-track Fisher derivatives are evaluated at ``u=0``, where the
        coherent difference construction is exactly the accepted production
        model.  Building a full FALI objective at every +/- global finite-
        difference point is therefore unnecessary.  This cache performs only
        the single production Emitter evaluation required for that derivative.
        """
        arr = self._theta_array(theta)
        valid = self._valid_theta(arr)
        if valid is None:
            self.invalid_evaluations += 1
            return None
        key = self._geometry_key(arr)
        cached = self._straight_predictions.get(key)
        if cached is not None:
            return cached
        external = self._external_straight_predictions.get(key)
        if external is not None:
            exp_pes, timing, timing_pes = external
            out = (
                np.asarray(exp_pes, dtype=np.float64),
                None if timing_pes is None else np.asarray(timing_pes, dtype=np.float64),
                timing,
            )
            self._straight_predictions[key] = out
            self.external_straight_cache_hits += 1
            return out
        model = self._models.get(key)
        if model is not None:
            out = (model.base_pes, model.base_timing_pes, model.base_timing)
            self._straight_predictions[key] = out
            return out
        vertex, direction, length, _t0 = valid
        try:
            em = self.emitter_template.copy()
            em.store_expected_component_diagnostics = False
            # Full-length points use the proposed threshold range. Absorption
            # points keep the separately fitted/fixed initial energy while the
            # proposed coordinate changes only the abrupt visible support.
            if hasattr(em, "configure_track_end"):
                if self.track_end_mode == "abrupt":
                    em.configure_track_end(
                        "abrupt",
                        fixed_initial_KE=self.initial_kinetic_energy_mev,
                        refresh=False,
                    )
                else:
                    em.configure_track_end(
                        "threshold", fixed_initial_KE=None, refresh=False
                    )
            else:
                em.track_end_mode = self.track_end_mode
                em.fixed_initial_KE = (
                    self.initial_kinetic_energy_mev
                    if self.track_end_mode == "abrupt"
                    else None
                )
            em.start_coord = tuple(vertex)
            em.direction = tuple(direction)
            em.starting_time = 0.0
            kinetic_energy = em.refresh_kinematics_from_length(float(length))
            if self.track_end_mode == "abrupt":
                active_range = float(
                    getattr(em, "range_to_threshold_mm", math.nan)
                )
                if abs(active_range - float(self.full_range_mm)) > max(
                    1.0e-4, 2.0e-6 * float(self.full_range_mm)
                ):
                    raise ValueError(
                        "abrupt coherent emitter range is inconsistent with "
                        "the requested full range"
                    )
            sources = em.get_emission_points(self.pmt_positions, kinetic_energy)
            exp_pes, timing = em.get_expected_pes_ts(
                self.wcd,
                sources,
                self.pmt_positions,
                self.pmt_normals,
                self.mpmt_types,
                self.obs_pes,
                need_times=True,
            )
            out = (
                np.asarray(exp_pes, dtype=np.float64),
                np.asarray(em._last_expected_pes_for_timing, dtype=np.float64),
                timing,
            )
        except (MCSPhysicalDomainError, FloatingPointError, OverflowError):
            self.invalid_evaluations += 1
            if raise_on_model_error:
                raise
            return None
        self._straight_predictions[key] = out
        return out

    def __call__(
        self,
        theta: Sequence[float],
        coefficients: Sequence[float],
        *,
        include_prior: bool = True,
    ) -> float:
        arr = self._theta_array(theta)
        model = self.model(arr)
        if model is None:
            return float("inf")
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        if np.any(~np.isfinite(u)):
            return float("inf")
        self.exact_evaluations += 1
        try:
            value = float(
                model.charge_data_nll(u)
                if self.charge_only
                else model.data_nll(u, t0=float(arr[6]))
            )
        except MCSPhysicalDomainError:
            self.physical_domain_rejections += 1
            return float("inf")
        if include_prior:
            value += 0.5 * float(u @ u)
        return value if math.isfinite(value) else float("inf")

    @property
    def optical_model_build_count(self) -> int:
        return int(self.model_build_count)

    @property
    def resident_optical_model_count(self) -> int:
        return len(self._models)

    @property
    def straight_prediction_build_count(self) -> int:
        return len(self._straight_predictions)

    @property
    def coherent_field_evaluation_count(self) -> int:
        return int(
            self._evicted_coherent_field_evaluations
            + sum(model.curved_evaluations for model in self._models.values())
        )
