"""Local coupled-track evaluator for the coherent Fermi--Eyges continuation.

Only a small interface is required by the current production-oriented test
branch: construct the complete coherent optical objective at a proposed global
track, evaluate it for a set of standardized KL coefficients, and provide a
stable positive-semidefinite inverse.  The accepted straight-track optimizer is
not replaced by this module.

No event truth or WCSim-derived response enters this calculation.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .mcs_coherent_objective import FixedTrackCoherentMCSObjective

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
        self.require_contained_track = bool(require_contained_track)
        self.length_limits = length_limits
        self.t0_limits = t0_limits
        self._models: dict[tuple[float, ...], FixedTrackCoherentMCSObjective] = {}
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

    @staticmethod
    def _theta_array(theta: Sequence[float]) -> np.ndarray:
        out = np.asarray(theta, dtype=np.float64).reshape(7)
        return out

    @staticmethod
    def _geometry_key(theta: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(theta[:6], 12))

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
            return cached
        vertex, direction, length, _t0 = valid
        try:
            use_precomputed = (
                self._precomputed_base_key is not None
                and key == self._precomputed_base_key
                and self._precomputed_base_emitter is not None
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
            )
            if use_precomputed:
                self.precomputed_base_context_uses += 1
        except Exception:
            self.invalid_evaluations += 1
            return None
        self._models[key] = model
        return model

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
            em.start_coord = tuple(vertex)
            em.direction = tuple(direction)
            em.starting_time = 0.0
            kinetic_energy = em.refresh_kinematics_from_length(float(length))
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
        except Exception:
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
        value = float(model.data_nll(u, t0=float(arr[6])))
        if include_prior:
            value += 0.5 * float(u @ u)
        return value if math.isfinite(value) else float("inf")

    @property
    def optical_model_build_count(self) -> int:
        return len(self._models)

    @property
    def straight_prediction_build_count(self) -> int:
        return len(self._straight_predictions)

    @property
    def coherent_field_evaluation_count(self) -> int:
        return int(sum(model.curved_evaluations for model in self._models.values()))
