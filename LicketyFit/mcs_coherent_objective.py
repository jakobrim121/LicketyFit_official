"""Complete optical objective for one coherent Fermi--Eyges/KL trajectory.

The accepted straight-track prediction is retained exactly at ``u=0``.  An
arc-length-preserving coherent trajectory contributes only a nonlinear
finite-aperture direct-primary *difference correction*.  Delta-electron,
molecular-scattering, reflection, event-normalization, and conditional
first-photoelectron terms remain the accepted production prediction in this
first testable implementation.

This construction is deliberately physics-only.  WCSim truth is used only in
external validation and never enters this objective.
"""
from __future__ import annotations

import math
import numpy as np

from .Emitter import TimingPrediction, shift_timing_prediction
from .mcs_curved_path import (
    curved_primary_field,
    curved_primary_finite_disk_interval_field,
    curved_primary_finite_disk_interval_charge_jacobian_field,
    curved_primary_finite_disk_line_field,
)


_SPARSE_NEIGHBOR_CSR_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
_SPARSE_NEIGHBOR_CSR_CACHE_MAX = 4


def _geometry_neighbor_csr(pmt_positions: np.ndarray, radius_mm: float):
    """Return a cached sparse physical-neighbor graph for one geometry.

    The graph depends only on detector geometry and the configured physical
    radius, not on an event or track.  Earlier code rebuilt the same pairwise
    distances for every nearby global-track model, which dominated the fast
    12-mode continuation.  CSR storage keeps the cache suitable for larger
    detectors where a dense N_PMT x N_PMT boolean matrix would be excessive.
    """
    xyz = np.ascontiguousarray(pmt_positions, dtype=np.float64)
    radius = max(float(radius_mm), 0.0)
    pointer = int(xyz.__array_interface__["data"][0])
    key = (
        pointer,
        tuple(xyz.shape),
        tuple(map(float, xyz[0])) if xyz.size else (),
        tuple(map(float, xyz[-1])) if xyz.size else (),
        round(radius, 9),
    )
    cached = _SPARSE_NEIGHBOR_CSR_CACHE.get(key)
    if cached is not None:
        return cached
    n = int(xyz.shape[0])
    r2 = radius * radius
    indptr = np.zeros(n + 1, dtype=np.int64)
    rows: list[np.ndarray] = []
    total = 0
    for first in range(0, n, 256):
        last = min(first + 256, n)
        delta = xyz[first:last, None, :] - xyz[None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        for local in range(last - first):
            neighbors = np.flatnonzero(distance2[local] <= r2).astype(
                np.int32, copy=False
            )
            rows.append(neighbors)
            total += int(neighbors.size)
            indptr[first + local + 1] = total
    indices = (
        np.concatenate(rows).astype(np.int32, copy=False)
        if rows else np.empty(0, dtype=np.int32)
    )
    out = (np.ascontiguousarray(indptr), np.ascontiguousarray(indices))
    if len(_SPARSE_NEIGHBOR_CSR_CACHE) >= _SPARSE_NEIGHBOR_CSR_CACHE_MAX:
        _SPARSE_NEIGHBOR_CSR_CACHE.pop(next(iter(_SPARSE_NEIGHBOR_CSR_CACHE)))
    _SPARSE_NEIGHBOR_CSR_CACHE[key] = out
    return out


def _copy_timing_with_direct(prediction, direct_mu_active, direct_t_active, node_pe_scale):
    """Clone a production TimingPrediction while replacing its direct row."""
    active = np.asarray(prediction.first_arrival_active_indices, dtype=np.int32)
    dbm = getattr(prediction, "first_arrival_deferred_base_mu", None)
    dbt = getattr(prediction, "first_arrival_deferred_base_t", None)
    if dbm is None or dbt is None:
        raise RuntimeError("coherent MCS currently requires deferred reflection timing")
    mu = np.array(dbm, copy=True)
    tt = np.array(dbt, copy=True)
    if mu.shape[0] < 1 or mu.shape[1] != active.size:
        raise RuntimeError("unexpected deferred timing-node shape")
    mu[0] = np.asarray(direct_mu_active, dtype=mu.dtype)
    tt[0] = np.asarray(direct_t_active, dtype=tt.dtype)
    return TimingPrediction(
        np.asarray(prediction, dtype=np.float64),
        node_mu=getattr(prediction, "first_arrival_node_mu", None),
        node_t=getattr(prediction, "first_arrival_node_t", None),
        active_indices=active,
        node_weight=getattr(prediction, "first_arrival_node_weight", None),
        weight_output_efficiency=getattr(
            prediction, "first_arrival_weight_output_efficiency", None
        ),
        deferred_base_mu=np.ascontiguousarray(mu),
        deferred_base_t=np.ascontiguousarray(tt),
        reflection_u=getattr(prediction, "first_arrival_reflection_u", None),
        reflection_tbase=getattr(prediction, "first_arrival_reflection_tbase", None),
        reflection_transfer_active=getattr(
            prediction, "first_arrival_reflection_transfer_active", None
        ),
        reflection_time_offset_active=getattr(
            prediction, "first_arrival_reflection_time_offset_active", None
        ),
        reflection_patch_min_time_offset=getattr(
            prediction, "first_arrival_reflection_patch_min_time_offset", None
        ),
        reflection_patch_max_time_offset=getattr(
            prediction, "first_arrival_reflection_patch_max_time_offset", None
        ),
        reflection_n_bins=getattr(prediction, "first_arrival_reflection_n_bins", None),
        node_pe_scale=float(node_pe_scale),
    )


class FixedTrackCoherentMCSObjective:
    """Full charge-plus-time objective in eight standardized FE coefficients.

    The constructor performs the complete accepted straight optical prediction
    once.  Subsequent calls evaluate only the nonlinear coherent direct field,
    splice its difference into that prediction, reapply event normalization,
    and invoke the unchanged production PMT likelihood.
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
        vertex,
        direction,
        length,
        t0=0.0,
        full_range_mm=None,
        initial_kinetic_energy_mev=None,
        mpmt_types=None,
        n_grid=81,
        aperture_radius_mm=45.0,
        path_field="fali",
        direct_timing_bins: int = 1,
        precomputed_base_emitter=None,
        precomputed_base_pes=None,
        precomputed_base_timing=None,
        sparse_receiver: bool = True,
        sparse_neighbor_radius_mm: float = 100.0,
        charge_only: bool = False,
    ):
        if int(direct_timing_bins) != 1:
            raise NotImplementedError(
                "this test branch validates one coherent direct timing node; "
                "MCS_COHERENT_DIRECT_TIMING_BINS must remain 1"
            )
        self.template = emitter_template
        self.wcd = wcd
        self.pmt_model = pmt_model
        self.pmt_positions = np.ascontiguousarray(pmt_positions, dtype=np.float64)
        self.pmt_normals = np.ascontiguousarray(pmt_normals, dtype=np.float64)
        self.obs_pes = np.asarray(obs_pes, dtype=np.float64)
        self.obs_ts = np.asarray(obs_ts, dtype=np.float64)
        self.vertex = np.asarray(vertex, dtype=np.float64)
        self.direction = np.asarray(direction, dtype=np.float64)
        self.direction /= max(float(np.linalg.norm(self.direction)), 1.0e-30)
        self.length = float(length)
        self.full_range_mm = (
            float(self.length) if full_range_mm is None else float(full_range_mm)
        )
        self.initial_kinetic_energy_mev = (
            None
            if initial_kinetic_energy_mev is None
            else float(initial_kinetic_energy_mev)
        )
        if not math.isfinite(self.full_range_mm) or self.full_range_mm <= 0.0:
            raise ValueError("full_range_mm must be positive and finite")
        if self.full_range_mm + 1.0e-7 < self.length:
            raise ValueError(
                "coherent visible length cannot exceed the remaining full range"
            )
        if (
            self.initial_kinetic_energy_mev is not None
            and (
                not math.isfinite(self.initial_kinetic_energy_mev)
                or self.initial_kinetic_energy_mev <= 0.0
            )
        ):
            raise ValueError("initial_kinetic_energy_mev must be positive and finite")
        self.t0 = float(t0)
        self.mpmt_types = mpmt_types
        self.n_grid = int(n_grid)
        self.aperture_radius_mm = float(aperture_radius_mm)
        self.direct_timing_bins = 1
        self.path_field = str(path_field).strip().lower()
        if self.path_field in {"fali", "finite_disk_interval", "interval", "support_tracked"}:
            self._field_function = curved_primary_finite_disk_interval_field
        elif self.path_field in {"finite_disk_line", "line"}:
            self._field_function = curved_primary_finite_disk_line_field
        elif self.path_field in {"aperture_roots", "roots", "legacy_nonlinear"}:
            self._field_function = curved_primary_field
        else:
            raise ValueError(f"unknown coherent MCS path field {path_field!r}")
        self.calls = 0
        self.curved_evaluations = 0
        self.cache: dict[tuple[float, ...], tuple] = {}
        self.charge_cache: dict[tuple[float, ...], np.ndarray] = {}
        self.cache_max = 16
        self.charge_cache_max = 64
        self.sparse_receiver = bool(sparse_receiver)
        self.sparse_neighbor_radius_mm = max(float(sparse_neighbor_radius_mm), 0.0)
        self.charge_only = bool(charge_only)

        # Complete accepted prediction with deferred reflection timing.  The
        # Emitter now retains the already-computed raw component arrays needed
        # below, so one physical optical evaluation supplies both the accepted
        # prediction and the coherent continuation reference state.
        if precomputed_base_emitter is None:
            em = emitter_template.copy()
            em.store_expected_component_diagnostics = False
            em.start_coord = tuple(self.vertex)
            em.direction = tuple(self.direction)
            em.starting_time = 0.0
            # Cosmic and absorption tracks can leave the active water before
            # reaching Cherenkov threshold.  In that case the visible path
            # length and the remaining CSDA range are independent.  Configure
            # the accepted Emitter in abrupt-end mode so the visible support is
            # clipped by geometry while energy loss, Cherenkov angle and FE
            # scattering power use the complete remaining range.
            if self.initial_kinetic_energy_mev is not None:
                em.configure_track_end(
                    "abrupt",
                    fixed_initial_KE=self.initial_kinetic_energy_mev,
                    refresh=False,
                )
            elif abs(self.full_range_mm - self.length) <= 1.0e-7:
                em.configure_track_end("threshold", refresh=False)
            else:
                raise ValueError(
                    "an independent full range requires initial_kinetic_energy_mev"
                )
            kinetic_energy = em.refresh_kinematics_from_length(self.length)
            if abs(float(getattr(em, "range_to_threshold_mm", self.length)) - self.full_range_mm) > max(1.0e-4, 2.0e-6 * self.full_range_mm):
                raise ValueError(
                    "Emitter range table is inconsistent with requested full_range_mm"
                )
            sources = em.get_emission_points(self.pmt_positions, kinetic_energy)
            base_pes, base_timing = em.get_expected_pes_ts(
                wcd,
                sources,
                self.pmt_positions,
                self.pmt_normals,
                mpmt_types,
                self.obs_pes,
                need_times=not self.charge_only,
            )
            self.precomputed_base_context_used = False
        else:
            em = precomputed_base_emitter
            base_pes = precomputed_base_pes
            base_timing = precomputed_base_timing
            if base_pes is None or (base_timing is None and not self.charge_only):
                raise ValueError("precomputed coherent base context is incomplete")
            self.precomputed_base_context_used = True
        self.base_emitter = em
        cached_full_range = float(getattr(em, "range_to_threshold_mm", self.length))
        if abs(cached_full_range - self.full_range_mm) > max(1.0e-4, 2.0e-6 * self.full_range_mm):
            raise ValueError(
                "precomputed coherent base emitter has the wrong full range"
            )
        self.base_pes = np.asarray(base_pes, dtype=np.float64)
        self.base_timing = base_timing
        self.base_timing_pes = (
            None if self.charge_only else np.asarray(
                em._last_expected_pes_for_timing, dtype=np.float64
            )
        )
        self.base_norm = float(em._last_expected_pes_norm)
        if self.charge_only:
            self.active = np.empty(0, dtype=np.int32)
            self.base_direct_active = np.empty(0, dtype=np.float64)
            self.base_direct_t_active = np.empty(0, dtype=np.float64)
        else:
            self.active = np.asarray(base_timing.first_arrival_active_indices, dtype=np.int32)
            self.base_direct_active = np.asarray(
                base_timing.first_arrival_deferred_base_mu[0], dtype=np.float64
            )
            self.base_direct_t_active = np.asarray(
                base_timing.first_arrival_deferred_base_t[0], dtype=np.float64
            )
        try:
            self.base_raw_charge = np.asarray(
                em._last_expected_pes_raw, dtype=np.float64
            )
            self.base_primary_surviving = np.asarray(
                em._last_mu_primary_raw, dtype=np.float64
            )
            self.base_raw_timing = np.asarray(
                em._last_expected_pes_timing_raw, dtype=np.float64
            )
            self.direct_survival = np.asarray(
                em._last_direct_molecular_survival, dtype=np.float64
            )
        except AttributeError as exc:
            raise RuntimeError(
                "Emitter lacks the lightweight coherent-MCS component cache"
            ) from exc
        self.charge_floor = float(em.charge_floor_pe)
        self.direct_survival = np.clip(self.direct_survival, 0.0, 1.0)

        # FALI consumes only the primary-track kinematics.  Reuse the configured
        # reference emitter rather than running a third full optical prediction.
        ep = em.copy()
        ep.enable_delta_e = False
        ep.enable_rayleigh_scatter = False
        ep.enable_blacksheet_reflection = False
        ep.store_expected_component_diagnostics = False
        self.path_emitter = ep

        self.modes_per_plane = int(getattr(ep, "primary_mcs_process_modes_per_plane", 4))
        self.n_modes = 2 * max(1, self.modes_per_plane)
        # Cache the nonlinear zero path.  The coherent correction is therefore
        # exactly zero at u=0 even if FALI and the accepted collapse model use
        # different absolute approximations.
        (
            curved_zero_mu_full,
            curved_zero_t_full,
            _zero_s,
            self.zero_path,
        ) = self._field_function(
            ep,
            self.pmt_positions,
            self.pmt_normals,
            np.zeros(self.n_modes),
            n_grid=self.n_grid,
            aperture_radius_mm=self.aperture_radius_mm,
        )
        self.curved_evaluations += 1
        self.curved_zero_mu = np.asarray(curved_zero_mu_full, dtype=np.float64)
        self.curved_zero_t = np.asarray(curved_zero_t_full, dtype=np.float64)

        # The coherent direct correction has compact receiver support.  Build a
        # geometry-only safety closure around every observed, accepted-primary,
        # or zero-path FALI PMT.  The neighborhood expansion is expressed in
        # physical millimetres and therefore generalizes to any supplied detector
        # geometry; it is not a WCTE slot or event-location special case.
        if self.sparse_receiver:
            support = (
                (self.obs_pes > 0.0)
                | (self.curved_zero_mu > 0.0)
                | (self.base_primary_surviving > 0.0)
            )
            radius = self.sparse_neighbor_radius_mm
            if radius > 0.0 and np.any(support):
                indptr, indices = _geometry_neighbor_csr(
                    self.pmt_positions, radius
                )
                expanded = support.copy()
                for selected_index in np.flatnonzero(support):
                    expanded[
                        indices[indptr[selected_index]:indptr[selected_index + 1]]
                    ] = True
                support = expanded
            self.coherent_active_indices = np.flatnonzero(support).astype(np.int32)
        else:
            self.coherent_active_indices = np.arange(
                self.pmt_positions.shape[0], dtype=np.int32
            )
        self.coherent_pmt_positions = np.ascontiguousarray(
            self.pmt_positions[self.coherent_active_indices], dtype=np.float64
        )
        self.coherent_pmt_normals = np.ascontiguousarray(
            self.pmt_normals[self.coherent_active_indices], dtype=np.float64
        )
        self.curved_zero_mu_sparse = np.ascontiguousarray(
            self.curved_zero_mu[self.coherent_active_indices], dtype=np.float64
        )
        self.curved_zero_t_sparse = np.ascontiguousarray(
            self.curved_zero_t[self.coherent_active_indices], dtype=np.float64
        )
        # At u=0 the difference correction is identically zero.  Cache the
        # literal accepted prediction so the constructor does not immediately
        # repeat the expensive FALI field evaluation.
        zero_key = tuple(np.zeros(self.n_modes, dtype=np.float64))
        self.cache[zero_key] = (
            self.base_pes,
            self.base_timing_pes,
            self.base_timing,
            self.zero_path,
            self.base_primary_surviving,
            self.base_norm,
        )
        self.charge_cache[zero_key] = self.base_pes
        self.base_nll = (
            self.charge_data_nll(np.zeros(self.n_modes))
            if self.charge_only else self.data_nll(np.zeros(self.n_modes))
        )

    def prediction(self, coefficients):
        if self.charge_only:
            raise RuntimeError(
                "complete timing prediction is unavailable in charge-only coherent mode"
            )
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(u, 10))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        mu_u, t_u, _s_u, path = self._field_function(
            self.path_emitter,
            self.coherent_pmt_positions,
            self.coherent_pmt_normals,
            u,
            n_grid=self.n_grid,
            aperture_radius_mm=self.aperture_radius_mm,
        )
        self.curved_evaluations += 1
        mu_u_full = np.zeros_like(self.curved_zero_mu)
        t_u_full = np.full_like(self.curved_zero_t, np.nan)
        mu_u_full[self.coherent_active_indices] = mu_u
        t_u_full[self.coherent_active_indices] = t_u
        mu_u = mu_u_full
        t_u = t_u_full

        delta_direct = self.direct_survival * (mu_u - self.curved_zero_mu)
        direct_corrected = np.maximum(self.base_primary_surviving + delta_direct, 0.0)
        raw_charge = np.maximum(self.base_raw_charge + delta_direct, 0.0)
        raw_timing = np.maximum(self.base_raw_timing + delta_direct, 0.0)
        raw_mean = float(np.mean(raw_charge))
        observed_mean = float(np.mean(self.obs_pes))
        norm = observed_mean / raw_mean if raw_mean > 0.0 else 0.0
        exp_pes = np.maximum(raw_charge * norm, self.charge_floor)
        timing_pes = raw_timing * norm

        direct_active = direct_corrected[self.active]
        direct_time_active = self.base_direct_t_active.copy()
        zero_mu = self.curved_zero_mu[self.active]
        path_mu = mu_u[self.active]
        zero_t = self.curved_zero_t[self.active]
        path_t = t_u[self.active]
        both = (
            (zero_mu > 0.0)
            & (path_mu > 0.0)
            & np.isfinite(zero_t)
            & np.isfinite(path_t)
        )
        direct_time_active[both] = self.base_direct_t_active[both] + (
            path_t[both] - zero_t[both]
        )
        new_only = (zero_mu <= 0.0) & (path_mu > 0.0) & np.isfinite(path_t)
        direct_time_active[new_only] = path_t[new_only]
        direct_time_active[direct_active <= 0.0] = np.inf
        timing = _copy_timing_with_direct(
            self.base_timing, direct_active, direct_time_active, norm
        )
        out = (
            exp_pes,
            timing_pes,
            timing,
            path,
            direct_corrected,
            norm,
        )
        if len(self.cache) >= self.cache_max:
            zero = tuple(np.zeros(self.n_modes, dtype=np.float64))
            for old_key in tuple(self.cache):
                if old_key != zero:
                    self.cache.pop(old_key, None)
                    break
        self.cache[key] = out
        return out

    def charge_prediction(self, coefficients):
        """Return normalized charge without constructing timing nodes."""
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(u, 10))
        cached = self.charge_cache.get(key)
        if cached is not None:
            return cached
        mu_u, _t_u, _s_u, _path = self._field_function(
            self.path_emitter, self.coherent_pmt_positions,
            self.coherent_pmt_normals, u,
            n_grid=self.n_grid, aperture_radius_mm=self.aperture_radius_mm,
        )
        self.curved_evaluations += 1
        mu_u_full = np.zeros_like(self.curved_zero_mu)
        mu_u_full[self.coherent_active_indices] = mu_u
        mu_u = mu_u_full
        delta_direct = self.direct_survival * (mu_u - self.curved_zero_mu)
        raw_charge = np.maximum(self.base_raw_charge + delta_direct, 0.0)
        raw_mean = float(np.mean(raw_charge))
        observed_mean = float(np.mean(self.obs_pes))
        norm = observed_mean / raw_mean if raw_mean > 0.0 else 0.0
        out = np.maximum(raw_charge * norm, self.charge_floor)
        if len(self.charge_cache) >= self.charge_cache_max:
            zero = tuple(np.zeros(self.n_modes, dtype=np.float64))
            for old_key in tuple(self.charge_cache):
                if old_key != zero:
                    self.charge_cache.pop(old_key, None)
                    break
        self.charge_cache[key] = out
        return out

    def charge_prediction_and_jacobian(self, coefficients):
        """Return normalized charge and the analytic FALI KL Jacobian.

        The derivative includes the coherent finite-disk line integral, direct
        molecular survival, non-negative raw-component guard, and the same
        event-total normalization and charge floor used by the production
        marginal.  It is therefore the derivative of ``charge_prediction``
        itself rather than of an unnormalized proxy field.
        """
        if self._field_function is not curved_primary_finite_disk_interval_field:
            raise NotImplementedError(
                "analytic coherent Jacobian is implemented for support-tracked FALI"
            )
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        mu_sparse, jac_sparse, _path = (
            curved_primary_finite_disk_interval_charge_jacobian_field(
                self.path_emitter,
                self.coherent_pmt_positions,
                self.coherent_pmt_normals,
                u,
                n_grid=self.n_grid,
                aperture_radius_mm=self.aperture_radius_mm,
            )
        )
        self.curved_evaluations += 1
        mu_full = np.zeros_like(self.curved_zero_mu)
        jac_full = np.zeros((mu_full.size, self.n_modes), dtype=np.float64)
        mu_full[self.coherent_active_indices] = mu_sparse
        jac_full[self.coherent_active_indices, :] = jac_sparse

        delta = self.direct_survival * (mu_full - self.curved_zero_mu)
        delta_jac = self.direct_survival[:, None] * jac_full
        raw_unclipped = self.base_raw_charge + delta
        positive = raw_unclipped > 0.0
        raw = np.maximum(raw_unclipped, 0.0)
        draw = delta_jac.copy()
        draw[~positive, :] = 0.0
        raw_mean = float(np.mean(raw))
        observed_mean = float(np.mean(self.obs_pes))
        if raw_mean <= 0.0:
            out = np.full_like(raw, self.charge_floor)
            return out, np.zeros_like(draw)
        norm = observed_mean / raw_mean
        mean_draw = np.mean(draw, axis=0)
        dnorm = -(norm / raw_mean) * mean_draw
        unfloored = raw * norm
        dout = norm * draw + raw[:, None] * dnorm[None, :]
        floor_mask = unfloored <= self.charge_floor
        out = np.maximum(unfloored, self.charge_floor)
        dout[floor_mask, :] = 0.0
        return np.ascontiguousarray(out), np.ascontiguousarray(dout)

    def charge_data_nll(self, coefficients):
        """Poisson charge NLL, matching the production charge marginal."""
        mu = np.maximum(self.charge_prediction(coefficients), 1.0e-300)
        q = np.asarray(self.obs_pes, dtype=np.float64)
        return float(np.sum(mu - q * np.log(mu)))

    def data_nll(self, coefficients, *, t0=None):
        exp_pes, timing_pes, timing, *_ = self.prediction(coefficients)
        dt = self.t0 if t0 is None else float(t0)
        if dt != 0.0:
            timing = shift_timing_prediction(timing, dt)
        return float(
            self.pmt_model.get_neg_log_likelihood_npe_t(
                exp_pes,
                self.obs_pes,
                timing,
                self.obs_ts,
                timing_pes=timing_pes,
            )
        )

    def __call__(self, coefficients):
        self.calls += 1
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        if np.any(~np.isfinite(u)):
            return 1.0e30
        value = self.data_nll(u) + 0.5 * float(u @ u)
        return value if math.isfinite(value) else 1.0e30
