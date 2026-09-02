"""Complete optical objective for one coherent Fermi--Eyges/KL trajectory.

The accepted straight-track prediction is retained exactly at ``u=0``.  An
arc-length-preserving coherent trajectory contributes nonlinear finite-
aperture direct-primary and analytic delta-electron *difference corrections*.
Both the direct field and dense delta field keep exact analytic KL Jacobians of
their evaluated optical transports; no empirical template or learned
parameter enters either derivative.  Charge-plus-time continuation promotes
the direct row and every longitudinal delta source to the same curved path.
Molecular-scattering and reflection timing remain the accepted conditional
production prediction, with reflection source identity retained byte-for-byte.

This construction is deliberately physics-only.  WCSim truth is used only in
external validation and never enters this objective.
"""
from __future__ import annotations

import math
import os
import numpy as np

from .Emitter import TimingPrediction, shift_timing_prediction
from .mcs_curved_path import (
    curved_delta_charge_jacobian_field,
    curved_delta_field,
    curved_delta_source_field,
    curved_delta_source_response_jacobian_field,
    curved_primary_field,
    curved_primary_finite_disk_interval_field,
    curved_primary_finite_disk_interval_path_field,
    curved_primary_finite_disk_interval_charge_jacobian_field,
    curved_primary_finite_disk_interval_response_jacobian_field,
    curved_primary_finite_disk_line_field,
)


_EXACT_CHARGE_NLL_REUSE = str(
    os.environ.get("LF_EXACT_CHARGE_NLL_REUSE", "1")
).strip().lower() not in {"0", "false", "no", "off"}


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


def _copy_timing_with_curved_sources(
    prediction,
    direct_mu_active,
    direct_t_active,
    node_pe_scale,
    *,
    delta_mu_active=None,
    delta_t_active=None,
):
    """Clone timing while replacing only direct and delta-source rows.

    Deferred molecular-scatter/CDS rows and all reflection arrays are copied
    through without alteration.  Delta rows immediately follow the direct row
    by the production :class:`Emitter` first-arrival contract.
    """
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
    if (delta_mu_active is None) != (delta_t_active is None):
        raise ValueError("delta timing amplitudes and times must be supplied together")
    if delta_mu_active is not None:
        dmu = np.asarray(delta_mu_active, dtype=mu.dtype)
        dtt = np.asarray(delta_t_active, dtype=tt.dtype)
        if dmu.ndim != 2 or dmu.shape != dtt.shape:
            raise ValueError("delta timing rows must have matching 2D shapes")
        stop = 1 + int(dmu.shape[0])
        if dmu.shape[1] != active.size or stop > mu.shape[0]:
            raise RuntimeError("delta source rows do not match deferred timing layout")
        mu[1:stop] = dmu
        tt[1:stop] = dtt
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


def _rescale_source_rows_to_aggregate(
    node_mu,
    aggregate_mu,
    *,
    node_mu_jac=None,
    aggregate_mu_jac=None,
):
    """Rescale source rows so each PMT sums to its authoritative aggregate.

    The established aggregate delta kernel remains the scalar authority.  This
    helper changes only the source decomposition used by first-arrival timing.
    If Jacobians are supplied, it differentiates that exact per-PMT rescaling.
    """
    raw_rows = np.asarray(node_mu, dtype=np.float64)
    raw_target = np.asarray(aggregate_mu, dtype=np.float64)
    if np.any(~np.isfinite(raw_rows)) or np.any(~np.isfinite(raw_target)):
        raise FloatingPointError("delta source amplitudes must be finite")
    rows = np.maximum(raw_rows, 0.0)
    target = np.maximum(raw_target, 0.0)
    if rows.ndim != 2 or target.ndim != 1 or rows.shape[1] != target.size:
        raise ValueError("source rows and aggregate delta field have incompatible shapes")
    total = np.sum(rows, axis=0)
    missing = (target > 1.0e-12) & (total <= 0.0)
    if np.any(missing):
        raise FloatingPointError(
            "source-resolved delta support cannot reproduce aggregate charge"
        )
    valid = (target > 0.0) & (total > 0.0)
    factor = np.zeros_like(target)
    factor[valid] = target[valid] / total[valid]
    scaled = rows * factor[None, :]
    if node_mu_jac is None and aggregate_mu_jac is None:
        return np.ascontiguousarray(scaled)
    if node_mu_jac is None or aggregate_mu_jac is None:
        raise ValueError("source and aggregate Jacobians must be supplied together")
    jac = np.asarray(node_mu_jac, dtype=np.float64)
    target_jac = np.asarray(aggregate_mu_jac, dtype=np.float64)
    if (
        jac.ndim != 3
        or jac.shape[:2] != rows.shape
        or target_jac.shape != (target.size, jac.shape[2])
    ):
        raise ValueError("delta source Jacobians have incompatible shapes")
    if np.any(~np.isfinite(jac)) or np.any(~np.isfinite(target_jac)):
        raise FloatingPointError("delta source Jacobians must be finite")
    # Clipped source rows are locally constant on the excluded side.
    jac = np.array(jac, copy=True)
    jac[np.asarray(node_mu) <= 0.0, :] = 0.0
    total_jac = np.sum(jac, axis=0)
    factor_jac = np.zeros_like(target_jac)
    factor_jac[valid, :] = (
        target_jac[valid, :] * total[valid, None]
        - target[valid, None] * total_jac[valid, :]
    ) / (total[valid, None] * total[valid, None])
    scaled_jac = (
        factor[None, :, None] * jac
        + rows[:, :, None] * factor_jac[None, :, :]
    )
    return np.ascontiguousarray(scaled), np.ascontiguousarray(scaled_jac)


class FixedTrackCoherentMCSObjective:
    """Full charge-plus-time objective in standardized FE coefficients.

    The constructor performs the complete accepted straight optical prediction
    once.  Subsequent calls evaluate the nonlinear coherent direct and analytic
    delta fields, splice their differences into that prediction, reapply event
    normalization, and invoke the unchanged production PMT likelihood.
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
        track_end_mode: str | None = None,
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
        delta_jacobian_step: float = 0.001,
        transverse_basis=None,
        path_validator=None,
        range_clipped_track=None,
        range_lookup=None,
        boundary_interface_model=None,
        boundary_interface_timing_policy: str = "mask_module",
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
        # Preserve the historical constructor contract: supplying an
        # independent initial energy implied the cosmic/straggled endpoint.
        # Absorption callers opt into the distinct abrupt energy-loss mapping
        # explicitly.
        end_mode = (
            "straggled_threshold"
            if track_end_mode is None and self.initial_kinetic_energy_mev is not None
            else "threshold"
            if track_end_mode is None
            else str(track_end_mode).strip().lower().replace("-", "_")
        )
        end_aliases = {
            "threshold": "threshold",
            "full_length": "threshold",
            "range": "threshold",
            "straggled_threshold": "straggled_threshold",
            "cosmic": "straggled_threshold",
            "abrupt": "abrupt",
            "absorption": "abrupt",
            "truncated": "abrupt",
        }
        if end_mode not in end_aliases:
            raise ValueError(
                "coherent track_end_mode must be threshold, "
                "straggled_threshold, or abrupt"
            )
        self.track_end_mode = end_aliases[end_mode]
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
        self.mpmt_types = (
            None
            if mpmt_types is None
            else np.ascontiguousarray(np.asarray(mpmt_types))
        )
        if (
            self.mpmt_types is not None
            and (
                self.mpmt_types.ndim != 1
                or self.mpmt_types.size != self.pmt_positions.shape[0]
            )
        ):
            raise ValueError(
                "coherent mPMT efficiency labels must contain one entry per PMT"
            )
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
        if (
            self.mpmt_types is not None
            and self._field_function is not curved_primary_finite_disk_interval_field
        ):
            raise NotImplementedError(
                "real-data mPMT efficiency requires the support-tracked FALI "
                "coherent path field"
            )
        self.calls = 0
        self.curved_evaluations = 0
        self.cache: dict[tuple[float, ...], tuple] = {}
        self.charge_cache: dict[tuple[float, ...], np.ndarray] = {}
        self.charge_data_nll_cache: dict[tuple[float, ...], float] = {}
        self.complete_charge_nll_cache: dict[tuple[float, ...], float] = {}
        self.delta_field_cache: dict[tuple[float, ...], np.ndarray] = {}
        self.delta_source_field_cache: dict[tuple[float, ...], tuple] = {}
        self.response_gradient_cache: dict[tuple[object, ...], tuple] = {}
        self.response_gradient_cache_hits = 0
        self.batched_timing_response_evaluations = 0
        self.cache_max = 16
        self.charge_cache_max = 64
        self.charge_nll_cache_max = 64
        self.delta_field_cache_max = 128
        self.sparse_receiver = bool(sparse_receiver)
        self.sparse_neighbor_radius_mm = max(float(sparse_neighbor_radius_mm), 0.0)
        self.charge_only = bool(charge_only)
        self.delta_jacobian_step = float(delta_jacobian_step)
        if (
            not math.isfinite(self.delta_jacobian_step)
            or self.delta_jacobian_step <= 0.0
        ):
            raise ValueError("delta_jacobian_step must be positive and finite")
        self.curved_delta_evaluations = 0
        self.curved_delta_jacobian_evaluations = 0
        self.curved_delta_source_evaluations = 0
        self.curved_delta_source_jacobian_evaluations = 0
        self.path_validator = path_validator
        self.path_validation_count = 0
        self.range_clipped_track = range_clipped_track
        self.range_lookup = range_lookup
        self.boundary_interface_model = boundary_interface_model
        self.boundary_interface_timing_policy = str(
            boundary_interface_timing_policy
        ).strip().lower()
        if self.boundary_interface_timing_policy not in {
            "mask_module", "baseline"
        }:
            raise ValueError(
                "coherent mPMT timing policy must be mask_module or baseline"
            )
        if self.boundary_interface_model is not None:
            if self.range_clipped_track is None or self.range_lookup is None:
                raise ValueError(
                    "coherent mPMT hardware requires the resolved clipped track "
                    "and particle range lookup"
                )
            if not bool(
                self.boundary_interface_model.validate_track(
                    self.range_clipped_track
                )
            ):
                raise ValueError(
                    "resolved coherent track does not satisfy its explicit "
                    "mPMT interface subclass"
                )
        # Profiling local hardware fractions makes the coherent response
        # piecewise smooth.  The exact finite-difference response is therefore
        # used instead of pretending the water-only analytic Jacobian includes
        # derivatives of the profiled nuisance coordinates.
        self.force_finite_difference_charge_jacobian = bool(
            self.boundary_interface_model is not None
        )
        self.force_finite_difference_complete_response = bool(
            self.boundary_interface_model is not None
        )
        self.boundary_hardware_modes = None
        self.boundary_hardware_profile_calls = 0
        self.last_boundary_hardware_profile = None

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
            # Abrupt absorption keeps the initial-energy/full-range loss
            # coordinate independent of the shorter Cherenkov support.  Cosmic
            # straggled-threshold tracks instead stretch the mean loss
            # coordinate to an independently realised stopping range.
            if self.track_end_mode == "abrupt":
                if self.initial_kinetic_energy_mev is None:
                    raise ValueError(
                        "abrupt coherent tracks require initial kinetic energy"
                    )
                em.configure_track_end(
                    "abrupt",
                    fixed_initial_KE=self.initial_kinetic_energy_mev,
                    refresh=False,
                )
            elif self.track_end_mode == "straggled_threshold":
                if self.initial_kinetic_energy_mev is None:
                    raise ValueError(
                        "straggled coherent tracks require initial kinetic energy"
                    )
                em.configure_stopping_range(
                    self.initial_kinetic_energy_mev,
                    self.full_range_mm,
                    refresh=False,
                )
            elif abs(self.full_range_mm - self.length) <= 1.0e-7:
                em.configure_track_end("threshold", refresh=False)
            else:
                raise ValueError(
                    "an independent full range requires initial_kinetic_energy_mev"
                )
            kinetic_energy = em.refresh_kinematics_from_length(self.length)
            emitter_full_range = float(
                getattr(
                    em,
                    "range_to_threshold_mm"
                    if self.track_end_mode == "abrupt"
                    else "realized_range_to_threshold_mm",
                    self.length,
                )
            )
            if abs(emitter_full_range - self.full_range_mm) > max(1.0e-4, 2.0e-6 * self.full_range_mm):
                raise ValueError(
                    "Emitter realised range is inconsistent with requested full_range_mm"
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
        cached_full_range = float(
            getattr(
                em,
                "range_to_threshold_mm"
                if self.track_end_mode == "abrupt"
                else "realized_range_to_threshold_mm",
                self.length,
            )
        )
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
            self.base_deferred_mu = None
            self.base_deferred_t = None
        else:
            self.active = np.asarray(base_timing.first_arrival_active_indices, dtype=np.int32)
            deferred_mu = getattr(
                base_timing, "first_arrival_deferred_base_mu", None
            )
            deferred_t = getattr(
                base_timing, "first_arrival_deferred_base_t", None
            )
            if deferred_mu is None or deferred_t is None:
                raise RuntimeError(
                    "coherent charge-time continuation requires source-identity-"
                    "preserving deferred reflection timing"
                )
            self.base_deferred_mu = np.asarray(deferred_mu)
            self.base_deferred_t = np.asarray(deferred_t)
            self.base_direct_active = np.asarray(
                deferred_mu[0], dtype=np.float64
            )
            self.base_direct_t_active = np.asarray(
                deferred_t[0], dtype=np.float64
            )
        try:
            self.base_raw_charge = np.asarray(
                em._last_expected_pes_raw, dtype=np.float64
            )
            self.base_primary_surviving = np.asarray(
                em._last_mu_primary_raw, dtype=np.float64
            )
            self.base_delta_raw = np.asarray(
                em._last_mu_delta_raw, dtype=np.float64
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

        # The explicit mPMT interface is a nested nuisance model on top of the
        # complete coherent water prediction.  Its zero-fraction point is the
        # unchanged MCS likelihood.  Hardware geometry and allowed fractions
        # are rebuilt for every global track model, while the convex fractions
        # are reprofiled for every latent path prediction.
        self.water_base_pes = np.ascontiguousarray(
            self.base_pes, dtype=np.float64
        )
        if self.boundary_interface_model is not None:
            if self.initial_kinetic_energy_mev is not None:
                hardware_ke0 = float(self.initial_kinetic_energy_mev)
            else:
                hardware_ke0 = float(
                    self.range_lookup.range_mm_to_energy(self.full_range_mm)
                )
            if not math.isfinite(hardware_ke0) or hardware_ke0 <= 0.0:
                raise ValueError(
                    "coherent mPMT hardware received a nonphysical entry energy"
                )
            self.boundary_hardware_modes = (
                self.boundary_interface_model.predict_track_modes(
                    track=self.range_clipped_track,
                    direction=self.direction,
                    kinetic_energy_at_water_entry_mev=hardware_ke0,
                    range_lookup=self.range_lookup,
                    emitter=self.base_emitter,
                )
            )
            self.base_pes = self._profile_boundary_hardware(
                self.water_base_pes
            )
            if not self.charge_only:
                # Crossed-module timestamps are masked by the caller.  The
                # remaining timing likelihood keeps the ordinary water timing
                # nodes but must use the same profiled charge marginal.
                self.base_timing_pes = np.ascontiguousarray(
                    self.base_pes, dtype=np.float64
                )

        # Curved direct and delta responses use the same source-resolved WCTE
        # slot/type efficiency table as the accepted straight prediction.
        # WCSim passes ``mpmt_types=None`` and takes an exact identity branch.
        self.curved_delta_enabled = bool(
            getattr(em, "enable_delta_e", False)
            and float(getattr(em, "delta_e_scale", 0.0)) != 0.0
        )
        self.delta_scale = (
            float(getattr(em, "delta_e_scale", 0.0))
            if self.curved_delta_enabled else 0.0
        )

        # FALI consumes only the primary-track kinematics.  Reuse the configured
        # reference emitter rather than running a third full optical prediction.
        ep = em.copy()
        ep.enable_delta_e = False
        # Keep the configured molecular-transport switch: the curved FALI
        # field uses it only to evaluate the zero-interaction survival
        # S(beta, r_gamma) at each direct-light quadrature node.  The scattered
        # photon source itself is part of the unchanged accepted base charge
        # and is not evaluated by any curved-primary field function.
        ep.enable_blacksheet_reflection = False
        ep.store_expected_component_diagnostics = False
        if transverse_basis is not None:
            e1, e2 = transverse_basis
            ep.primary_mcs_transverse_e1 = np.ascontiguousarray(
                np.asarray(e1, dtype=np.float64).reshape(3)
            )
            ep.primary_mcs_transverse_e2 = np.ascontiguousarray(
                np.asarray(e2, dtype=np.float64).reshape(3)
            )
        self.path_emitter = ep
        self._delta_source_state = (
            ep._build_delta_source_grid() if self.curved_delta_enabled else None
        )
        self.n_delta_sources = (
            int(np.asarray(self._delta_source_state[0]).size)
            if self._delta_source_state is not None
            and bool(self._delta_source_state[3])
            else 0
        )
        if not self.charge_only and self.curved_delta_enabled:
            stop = 1 + self.n_delta_sources
            if (
                self.base_deferred_mu.ndim != 2
                or self.base_deferred_t.shape != self.base_deferred_mu.shape
                or stop > self.base_deferred_mu.shape[0]
            ):
                raise RuntimeError(
                    "accepted timing prediction lacks the expected delta source rows"
                )
            self.base_delta_node_mu = np.asarray(
                self.base_deferred_mu[1:stop], dtype=np.float64
            )
            self.base_delta_node_t = np.asarray(
                self.base_deferred_t[1:stop], dtype=np.float64
            )
            self.delta_timing_pmt_positions = np.ascontiguousarray(
                self.pmt_positions[self.active], dtype=np.float64
            )
            self.delta_timing_pmt_normals = np.ascontiguousarray(
                self.pmt_normals[self.active], dtype=np.float64
            )
        else:
            self.base_delta_node_mu = np.zeros(
                (0, self.active.size), dtype=np.float64
            )
            self.base_delta_node_t = np.zeros_like(self.base_delta_node_mu)
            self.delta_timing_pmt_positions = np.zeros((0, 3), dtype=np.float64)
            self.delta_timing_pmt_normals = np.zeros((0, 3), dtype=np.float64)

        self.modes_per_plane = int(getattr(ep, "primary_mcs_process_modes_per_plane", 4))
        self.n_modes = 2 * max(1, self.modes_per_plane)
        # Cache the nonlinear zero path.  The coherent correction is therefore
        # exactly zero at u=0 even if FALI and the accepted collapse model use
        # different absolute approximations.
        zero_field_keywords = {}
        if (
            self.charge_only
            and self._field_function is curved_primary_finite_disk_interval_field
        ):
            zero_field_keywords["compute_moments"] = False
        if self._field_function is curved_primary_finite_disk_interval_field:
            zero_field_keywords["mpmt_types"] = self.mpmt_types
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
            **zero_field_keywords,
        )
        self.curved_evaluations += 1
        self._validate_physical_path(self.zero_path)
        self.curved_zero_mu = np.asarray(curved_zero_mu_full, dtype=np.float64)
        self.curved_zero_t = np.asarray(curved_zero_t_full, dtype=np.float64)

        # Delta electrons form a broad field, so unlike the compact direct FALI
        # correction they are always evaluated on every receiver.  Difference
        # matching makes the complete prediction exactly equal to the accepted
        # production model at u=0 even though the two analytic implementations
        # are not required to have bit-identical absolute zero-path fields.
        zero_key = tuple(np.zeros(self.n_modes, dtype=np.float64))
        if self.curved_delta_enabled:
            curved_zero_delta, _zero_delta_t, _zero_delta_path = curved_delta_field(
                self.path_emitter,
                self.pmt_positions,
                self.pmt_normals,
                np.zeros(self.n_modes, dtype=np.float64),
                n_grid=self.n_grid,
                compute_time=False,
                path=self.zero_path,
                source_state=self._delta_source_state,
                mpmt_types=self.mpmt_types,
            )
            self.curved_delta_evaluations += 1
            self.curved_zero_delta = np.ascontiguousarray(
                self.delta_scale * np.asarray(curved_zero_delta, dtype=np.float64)
            )
            if not self.charge_only:
                zero_node_mu, zero_node_t, _zero_source_path = (
                    curved_delta_source_field(
                        self.path_emitter,
                        self.delta_timing_pmt_positions,
                        self.delta_timing_pmt_normals,
                        np.zeros(self.n_modes, dtype=np.float64),
                        n_grid=self.n_grid,
                        path=self.zero_path,
                        source_state=self._delta_source_state,
                        mpmt_types=(
                            None
                            if self.mpmt_types is None
                            else self.mpmt_types[self.active]
                        ),
                    )
                )
                self.curved_delta_source_evaluations += 1
                zero_node_mu = self.delta_scale * np.asarray(
                    zero_node_mu, dtype=np.float64
                )
                self.curved_zero_delta_node_mu = (
                    _rescale_source_rows_to_aggregate(
                        zero_node_mu, self.curved_zero_delta[self.active]
                    )
                )
                self.curved_zero_delta_node_t = np.where(
                    self.curved_zero_delta_node_mu > 0.0,
                    np.asarray(zero_node_t, dtype=np.float64),
                    np.inf,
                )
            else:
                self.curved_zero_delta_node_mu = np.zeros(
                    (0, 0), dtype=np.float64
                )
                self.curved_zero_delta_node_t = np.zeros((0, 0), dtype=np.float64)
        else:
            self.curved_zero_delta = np.zeros_like(self.base_raw_charge)
            self.curved_zero_delta_node_mu = np.zeros(
                (0, self.active.size), dtype=np.float64
            )
            self.curved_zero_delta_node_t = np.zeros_like(
                self.curved_zero_delta_node_mu
            )
        self.delta_field_cache[zero_key] = self.curved_zero_delta
        if self.curved_delta_enabled and not self.charge_only:
            self.delta_source_field_cache[zero_key] = (
                self.curved_zero_delta_node_mu,
                self.curved_zero_delta_node_t,
            )
        delta_denominator = max(
            float(np.sum(np.abs(self.base_delta_raw))), 1.0e-300
        )
        self.delta_zero_reference_relative_l1 = float(
            np.sum(np.abs(self.curved_zero_delta - self.base_delta_raw))
            / delta_denominator
        ) if self.curved_delta_enabled else 0.0

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
        self.coherent_mpmt_types = (
            None
            if self.mpmt_types is None
            else np.ascontiguousarray(
                self.mpmt_types[self.coherent_active_indices]
            )
        )
        self.delta_timing_mpmt_types = (
            None
            if self.mpmt_types is None
            else np.ascontiguousarray(self.mpmt_types[self.active])
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
        self.cache[zero_key] = (
            self.base_pes,
            self.base_timing_pes,
            self.base_timing,
            self.zero_path,
            self.base_primary_surviving,
            self.base_norm,
        )
        self.charge_cache[zero_key] = self.base_pes

    def _validate_physical_path(self, path):
        """Reject a coherent curve outside the configured detector domain."""
        if self.path_validator is None:
            return
        self.path_validation_count += 1
        if not bool(self.path_validator(path)):
            from .mcs_curved_path import MCSPhysicalDomainError

            raise MCSPhysicalDomainError(
                "coherent FE curve leaves the contained detector volume"
            )

    @staticmethod
    def _insert_preserving_zero(cache, key, value, maximum):
        """Insert into a small deterministic cache without evicting u=0."""
        if len(cache) >= int(maximum):
            zero = tuple(np.zeros(len(key), dtype=np.float64))
            for old_key in tuple(cache):
                if old_key != zero:
                    cache.pop(old_key, None)
                    break
        cache[key] = value

    def _charge_normalization(self, raw_mean):
        observed_mean = float(np.mean(self.obs_pes))
        norm = float(
            self.base_emitter.charge_normalization_factor(
                float(raw_mean), observed_mean
            )
        )
        mode = str(
            getattr(
                self.base_emitter,
                "charge_normalization_mode",
                "event_mean",
            )
        ).strip().lower().replace("-", "_")
        return norm, mode == "event_mean"

    def _profile_boundary_hardware(self, water_expected):
        """Profile explicit local hardware modes over one coherent prediction.

        The profile is convex and includes the exact zero-hardware point.  It
        is deliberately evaluated after the coherent water normalization so
        it matches the established straight-track mPMT likelihood contract.
        """
        water = np.ascontiguousarray(water_expected, dtype=np.float64)
        if self.boundary_interface_model is None:
            return water
        profile = self.boundary_interface_model.profile_charge(
            water,
            self.obs_pes,
            self.boundary_hardware_modes,
        )
        expected = np.ascontiguousarray(
            profile.expected_pes, dtype=np.float64
        )
        if (
            expected.shape != water.shape
            or np.any(~np.isfinite(expected))
            or np.any(expected < 0.0)
        ):
            raise FloatingPointError(
                "coherent mPMT profile returned an invalid charge prediction"
            )
        self.boundary_hardware_profile_calls += 1
        fractions = np.asarray(profile.fractions, dtype=np.float64)
        self.last_boundary_hardware_profile = {
            "fractions": fractions.tolist(),
            "total_fraction": float(np.sum(fractions)),
            "charge_nll_improvement": float(profile.improvement),
            "profile_iterations": int(profile.iterations),
            "profile_converged": bool(profile.converged),
            "model": (
                self.boundary_interface_model.metadata()
                if hasattr(self.boundary_interface_model, "metadata")
                else None
            ),
            "timing_policy": str(self.boundary_interface_timing_policy),
        }
        return expected

    def boundary_hardware_metadata(self, coefficients=None):
        """Return an auditable final profile for the requested coherent path."""
        if self.boundary_interface_model is None:
            return {
                "enabled": False,
                "profile_calls": 0,
                "profile": None,
            }
        if coefficients is not None:
            if self.charge_only:
                self.charge_prediction(coefficients)
            else:
                self.prediction(coefficients)
        return {
            "enabled": True,
            "profile_calls": int(self.boundary_hardware_profile_calls),
            "profile": (
                None
                if self.last_boundary_hardware_profile is None
                else dict(self.last_boundary_hardware_profile)
            ),
        }

    def curved_delta_charge_field(self, coefficients, *, path=None):
        """Return the scaled analytic curved-delta field on all PMTs.

        This field is intentionally unnormalized.  Event-total conditioning is
        applied only after it is combined with the direct and unchanged optical
        components, exactly as in the production charge likelihood.
        """
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(u, 10))
        cached = self.delta_field_cache.get(key)
        if cached is not None:
            return cached
        if not self.curved_delta_enabled:
            return self.curved_zero_delta
        mu, _time, _path = curved_delta_field(
            self.path_emitter,
            self.pmt_positions,
            self.pmt_normals,
            u,
            n_grid=self.n_grid,
            compute_time=False,
            path=path,
            source_state=self._delta_source_state,
            mpmt_types=self.mpmt_types,
        )
        self.curved_delta_evaluations += 1
        self._validate_physical_path(_path)
        out = np.ascontiguousarray(
            self.delta_scale * np.asarray(mu, dtype=np.float64)
        )
        self._insert_preserving_zero(
            self.delta_field_cache,
            key,
            out,
            self.delta_field_cache_max,
        )
        return out

    def curved_delta_source_nodes(self, coefficients, *, curved_delta=None, path=None):
        """Return aggregate-matched curved delta nodes on timed PMTs."""
        if self.charge_only:
            raise RuntimeError("delta source timing is unavailable in charge-only mode")
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(u, 10))
        cached = self.delta_source_field_cache.get(key)
        if cached is not None:
            return cached
        if not self.curved_delta_enabled:
            return self.curved_zero_delta_node_mu, self.curved_zero_delta_node_t
        if curved_delta is None:
            curved_delta = self.curved_delta_charge_field(u, path=path)
        node_mu, node_t, checked_path = curved_delta_source_field(
            self.path_emitter,
            self.delta_timing_pmt_positions,
            self.delta_timing_pmt_normals,
            u,
            n_grid=self.n_grid,
            path=path,
            source_state=self._delta_source_state,
            mpmt_types=self.delta_timing_mpmt_types,
        )
        self.curved_delta_source_evaluations += 1
        self._validate_physical_path(checked_path)
        node_mu = _rescale_source_rows_to_aggregate(
            self.delta_scale * np.asarray(node_mu, dtype=np.float64),
            np.asarray(curved_delta, dtype=np.float64)[self.active],
        )
        node_t = np.where(
            node_mu > 0.0, np.asarray(node_t, dtype=np.float64), np.inf
        )
        out = (np.ascontiguousarray(node_mu), np.ascontiguousarray(node_t))
        self._insert_preserving_zero(
            self.delta_source_field_cache,
            key,
            out,
            self.delta_field_cache_max,
        )
        return out

    def _difference_matched_delta_source_nodes(
        self, curved_delta, curved_node_mu, curved_node_t
    ):
        """Splice curved source amplitudes/times into accepted delta rows."""
        if not self.curved_delta_enabled:
            return None, None
        curved_delta = np.asarray(curved_delta, dtype=np.float64)
        node_mu = np.asarray(curved_node_mu, dtype=np.float64)
        node_t = np.asarray(curved_node_t, dtype=np.float64)
        expected_shape = (self.n_delta_sources, self.active.size)
        if node_mu.shape != expected_shape or node_t.shape != expected_shape:
            raise ValueError("curved delta source fields have unexpected shapes")

        # Source labels are fixed longitudinal cells.  Difference-match each
        # cell, then restore the aggregate delta mass exactly on every PMT.
        candidate = (
            self.base_delta_node_mu
            + node_mu
            - self.curved_zero_delta_node_mu
        )
        target = (
            self.base_delta_raw[self.active]
            + curved_delta[self.active]
            - self.curved_zero_delta[self.active]
        )
        # A large support migration can make every cellwise difference row
        # non-positive on one PMT even though the authoritative aggregate
        # difference remains positive.  That is a decomposition degeneracy,
        # not an invalid physical charge state.  Restore support from the
        # current curved source field (whose times follow the current path),
        # or from the accepted base rows when the current curved field is
        # genuinely absent.  The aggregate target and charge kernel remain
        # unchanged; the common rescaler below still closes each PMT exactly.
        positive_candidate = np.maximum(candidate, 0.0)
        target_positive = np.maximum(target, 0.0)
        missing = (
            (target_positive > 1.0e-12)
            & (np.sum(positive_candidate, axis=0) <= 0.0)
        )
        current_fallback = np.zeros(target.shape, dtype=bool)
        base_fallback = np.zeros(target.shape, dtype=bool)
        if np.any(missing):
            current_rows = np.maximum(node_mu, 0.0)
            current_fallback = missing & (
                np.sum(current_rows, axis=0) > 0.0
            )
            candidate[:, current_fallback] = current_rows[:, current_fallback]
            remaining = missing & ~current_fallback
            if np.any(remaining):
                base_rows = np.maximum(self.base_delta_node_mu, 0.0)
                base_fallback = remaining & (
                    np.sum(base_rows, axis=0) > 0.0
                )
                candidate[:, base_fallback] = base_rows[:, base_fallback]
        corrected_mu = _rescale_source_rows_to_aggregate(candidate, target)

        corrected_t = np.array(self.base_delta_node_t, copy=True)
        zero_mu = self.curved_zero_delta_node_mu
        zero_t = self.curved_zero_delta_node_t
        both = (
            (zero_mu > 0.0)
            & (node_mu > 0.0)
            & np.isfinite(zero_t)
            & np.isfinite(node_t)
        )
        corrected_t[both] = self.base_delta_node_t[both] + (
            node_t[both] - zero_t[both]
        )
        new_only = (
            (zero_mu <= 0.0) & (node_mu > 0.0) & np.isfinite(node_t)
        )
        corrected_t[new_only] = node_t[new_only]
        if np.any(current_fallback):
            corrected_t[:, current_fallback] = node_t[:, current_fallback]
        if np.any(base_fallback):
            corrected_t[:, base_fallback] = self.base_delta_node_t[
                :, base_fallback
            ]
        corrected_t[corrected_mu <= 0.0] = np.inf
        return np.ascontiguousarray(corrected_mu), np.ascontiguousarray(corrected_t)

    def _finite_difference_curved_delta_charge_jacobian(self, coefficients, step):
        """Reference central derivative retained only for validation."""
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        h = float(step)
        if not math.isfinite(h) or h <= 0.0:
            raise ValueError("delta Jacobian step must be positive and finite")
        if not self.curved_delta_enabled:
            return np.zeros(
                (self.pmt_positions.shape[0], self.n_modes), dtype=np.float64
            )
        jac = np.empty(
            (self.pmt_positions.shape[0], self.n_modes), dtype=np.float64
        )
        for k in range(self.n_modes):
            plus = u.copy()
            minus = u.copy()
            plus[k] += h
            minus[k] -= h
            jac[:, k] = (
                self.curved_delta_charge_field(plus)
                - self.curved_delta_charge_field(minus)
            ) / (2.0 * h)
        return np.ascontiguousarray(jac)

    def curved_delta_charge_jacobian(self, coefficients, *, step=None, path=None):
        """Return the exact analytic KL Jacobian of curved delta charge.

        Supplying ``step`` explicitly retains the former central derivative as
        a compatibility and validation path.  Normal fitting leaves it unset
        and evaluates all KL columns in one analytic receiver pass.
        """
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        if step is not None:
            return self._finite_difference_curved_delta_charge_jacobian(u, step)
        if not self.curved_delta_enabled:
            return np.zeros(
                (self.pmt_positions.shape[0], self.n_modes), dtype=np.float64
            )
        _reconstructed, jac, _path = curved_delta_charge_jacobian_field(
            self.path_emitter,
            self.pmt_positions,
            self.pmt_normals,
            u,
            n_grid=self.n_grid,
            path=path,
            source_state=self._delta_source_state,
            mpmt_types=self.mpmt_types,
        )
        self.curved_delta_jacobian_evaluations += 1
        self._validate_physical_path(_path)
        return np.ascontiguousarray(self.delta_scale * jac)

    def delta_jacobian_step_halving_diagnostic(self, coefficients, *, step=None):
        """Compare delta Jacobians at h and h/2 without changing the fit.

        The returned relative norms are numerical-stability diagnostics, not
        acceptance cuts or fitted scale factors.  Cached field evaluations make
        this inexpensive when run immediately after a normal Jacobian call.
        """
        h = self.delta_jacobian_step if step is None else float(step)
        coarse = self._finite_difference_curved_delta_charge_jacobian(
            coefficients, h
        )
        fine = self._finite_difference_curved_delta_charge_jacobian(
            coefficients, 0.5 * h
        )
        analytic = self.curved_delta_charge_jacobian(coefficients)
        denominator = max(float(np.linalg.norm(fine)), 1.0e-300)
        column_denominator = np.maximum(
            np.linalg.norm(fine, axis=0), 1.0e-300
        )
        column_relative = np.linalg.norm(fine - coarse, axis=0) / column_denominator
        return {
            "step": float(h),
            "half_step": float(0.5 * h),
            "relative_frobenius": float(
                np.linalg.norm(fine - coarse) / denominator
            ),
            "maximum_relative_column": float(np.max(column_relative)),
            "analytic_relative_frobenius": float(
                np.linalg.norm(analytic - fine) / denominator
            ),
            "coarse": coarse,
            "fine": fine,
            "analytic": analytic,
        }

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
        field_keywords = {}
        if self._field_function is curved_primary_finite_disk_interval_field:
            field_keywords["mpmt_types"] = self.coherent_mpmt_types
        mu_u, t_u, _s_u, path = self._field_function(
            self.path_emitter,
            self.coherent_pmt_positions,
            self.coherent_pmt_normals,
            u,
            n_grid=self.n_grid,
            aperture_radius_mm=self.aperture_radius_mm,
            **field_keywords,
        )
        self.curved_evaluations += 1
        self._validate_physical_path(path)
        mu_u_full = np.zeros_like(self.curved_zero_mu)
        t_u_full = np.full_like(self.curved_zero_t, np.nan)
        mu_u_full[self.coherent_active_indices] = mu_u
        t_u_full[self.coherent_active_indices] = t_u
        if self.curved_delta_enabled:
            curved_delta = self.curved_delta_charge_field(u, path=path)
            delta_node_mu, delta_node_t = self.curved_delta_source_nodes(
                u, curved_delta=curved_delta, path=path
            )
        else:
            curved_delta = self.curved_zero_delta
            delta_node_mu = None
            delta_node_t = None
        out = self._prediction_from_direct_field(
            mu_u_full,
            t_u_full,
            path,
            curved_delta=curved_delta,
            curved_delta_node_mu=delta_node_mu,
            curved_delta_node_t=delta_node_t,
        )
        if len(self.cache) >= self.cache_max:
            zero = tuple(np.zeros(self.n_modes, dtype=np.float64))
            for old_key in tuple(self.cache):
                if old_key != zero:
                    self.cache.pop(old_key, None)
                    break
        self.cache[key] = out
        return out

    def _prediction_from_direct_field(
        self,
        path_mu,
        path_t,
        path,
        *,
        curved_delta=None,
        curved_delta_node_mu=None,
        curved_delta_node_t=None,
    ):
        """Splice coherent direct and delta fields into the base prediction."""
        if self.charge_only:
            raise RuntimeError("complete timing prediction is unavailable in charge-only mode")
        path_mu = np.asarray(path_mu, dtype=np.float64)
        path_t = np.asarray(path_t, dtype=np.float64)

        # Molecular zero-interaction survival is evaluated inside every curved
        # FALI quadrature node.  Both terms below therefore already represent
        # S(path)F(path) and S(0)F(0); applying straight survival again would
        # attenuate the coherent correction twice.
        delta_direct = path_mu - self.curved_zero_mu
        if curved_delta is None:
            curved_delta = self.curved_zero_delta
        curved_delta = np.asarray(curved_delta, dtype=np.float64)
        delta_delta = curved_delta - self.curved_zero_delta
        direct_corrected = np.maximum(
            self.base_primary_surviving + delta_direct, 0.0
        )
        raw_charge = np.maximum(
            self.base_raw_charge + delta_direct + delta_delta, 0.0
        )
        raw_timing = np.maximum(
            self.base_raw_timing + delta_direct + delta_delta, 0.0
        )
        norm, _event_conditioned = self._charge_normalization(
            float(np.mean(raw_charge))
        )
        exp_pes = np.maximum(raw_charge * norm, self.charge_floor)
        timing_pes = raw_timing * norm
        if self.boundary_interface_model is not None:
            exp_pes = self._profile_boundary_hardware(exp_pes)
            # The crossed module's timestamps are masked by the event driver.
            # On all remaining PMTs, the configured timing likelihood must use
            # the same charge marginal that was profiled for this curved path.
            timing_pes = np.ascontiguousarray(exp_pes, dtype=np.float64)

        direct_active = direct_corrected[self.active]
        direct_time_active = self.base_direct_t_active.copy()
        zero_mu = self.curved_zero_mu[self.active]
        curved_mu = path_mu[self.active]
        zero_t = self.curved_zero_t[self.active]
        curved_t = path_t[self.active]
        both = (
            (zero_mu > 0.0)
            & (curved_mu > 0.0)
            & np.isfinite(zero_t)
            & np.isfinite(curved_t)
        )
        direct_time_active[both] = self.base_direct_t_active[both] + (
            curved_t[both] - zero_t[both]
        )
        new_only = (
            (zero_mu <= 0.0) & (curved_mu > 0.0) & np.isfinite(curved_t)
        )
        direct_time_active[new_only] = curved_t[new_only]
        direct_time_active[direct_active <= 0.0] = np.inf
        delta_corrected_mu, delta_corrected_t = (
            self._difference_matched_delta_source_nodes(
                curved_delta, curved_delta_node_mu, curved_delta_node_t
            )
            if self.curved_delta_enabled
            else (None, None)
        )
        timing = _copy_timing_with_curved_sources(
            self.base_timing,
            direct_active,
            direct_time_active,
            norm,
            delta_mu_active=delta_corrected_mu,
            delta_t_active=delta_corrected_t,
        )
        return (
            exp_pes,
            timing_pes,
            timing,
            path,
            direct_corrected,
            norm,
        )

    def _deferred_response_rows_from_direct_field(
        self,
        path_mu,
        path_t,
        *,
        curved_delta,
        curved_delta_node_mu,
        curved_delta_node_t,
    ):
        """Assemble only fields needed by a deferred timing response variant.

        The joint local stencil does not consume nominal time arrays, timing-PE
        marginals, path dictionaries, or complete ``TimingPrediction`` objects.
        Returning its varying rows directly preserves the same normalization,
        difference matching, clipping, and source times while avoiding 48
        short-lived full prediction tuples per response evaluation.
        """
        if self.charge_only or self.boundary_interface_model is not None:
            raise RuntimeError(
                "lightweight deferred rows require an ordinary charge-time model"
            )
        path_mu = np.asarray(path_mu, dtype=np.float64)
        path_t = np.asarray(path_t, dtype=np.float64)
        curved_delta = np.asarray(curved_delta, dtype=np.float64)
        delta_direct = path_mu - self.curved_zero_mu
        delta_delta = curved_delta - self.curved_zero_delta
        direct_corrected = np.maximum(
            self.base_primary_surviving + delta_direct, 0.0
        )
        raw_charge = np.maximum(
            self.base_raw_charge + delta_direct + delta_delta, 0.0
        )
        norm, _event_conditioned = self._charge_normalization(
            float(np.mean(raw_charge))
        )
        exp_pes = np.maximum(raw_charge * norm, self.charge_floor)

        direct_active = direct_corrected[self.active]
        direct_time_active = self.base_direct_t_active.copy()
        zero_mu = self.curved_zero_mu[self.active]
        curved_mu = path_mu[self.active]
        zero_t = self.curved_zero_t[self.active]
        curved_t = path_t[self.active]
        both = (
            (zero_mu > 0.0)
            & (curved_mu > 0.0)
            & np.isfinite(zero_t)
            & np.isfinite(curved_t)
        )
        direct_time_active[both] = self.base_direct_t_active[both] + (
            curved_t[both] - zero_t[both]
        )
        new_only = (
            (zero_mu <= 0.0)
            & (curved_mu > 0.0)
            & np.isfinite(curved_t)
        )
        direct_time_active[new_only] = curved_t[new_only]
        direct_time_active[direct_active <= 0.0] = np.inf
        delta_corrected_mu, delta_corrected_t = (
            self._difference_matched_delta_source_nodes(
                curved_delta,
                curved_delta_node_mu,
                curved_delta_node_t,
            )
            if self.curved_delta_enabled
            else (None, None)
        )
        return (
            np.ascontiguousarray(exp_pes, dtype=np.float64),
            np.ascontiguousarray(direct_active),
            np.ascontiguousarray(direct_time_active),
            (
                None
                if delta_corrected_mu is None
                else np.ascontiguousarray(delta_corrected_mu)
            ),
            (
                None
                if delta_corrected_t is None
                else np.ascontiguousarray(delta_corrected_t)
            ),
            float(norm),
        )

    def prediction_from_path(self, path):
        """Return charge and first-arrival timing for an explicit trajectory.

        This is the timing-capable counterpart to
        :meth:`charge_prediction_from_path`.  Direct primary light and every
        longitudinal delta source follow the supplied coherent trajectory;
        molecular scattering and reflection retain the accepted conditional
        timing state.
        """
        if self.charge_only:
            raise RuntimeError("explicit-path timing requires charge_time mode")
        if self._field_function is not curved_primary_finite_disk_interval_field:
            raise NotImplementedError(
                "explicit-path timing is implemented for support-tracked FALI"
            )
        mu_sparse, time_sparse, _source_s, checked_path = (
            curved_primary_finite_disk_interval_path_field(
                self.path_emitter,
                self.coherent_pmt_positions,
                self.coherent_pmt_normals,
                path,
                aperture_radius_mm=self.aperture_radius_mm,
                compute_moments=True,
                mpmt_types=self.coherent_mpmt_types,
            )
        )
        self.curved_evaluations += 1
        self._validate_physical_path(checked_path)
        mu_full = np.zeros_like(self.curved_zero_mu)
        time_full = np.full_like(self.curved_zero_t, np.nan)
        mu_full[self.coherent_active_indices] = mu_sparse
        time_full[self.coherent_active_indices] = time_sparse
        if self.curved_delta_enabled:
            delta_mu, _delta_time, delta_path = curved_delta_field(
                self.path_emitter,
                self.pmt_positions,
                self.pmt_normals,
                np.zeros(self.n_modes, dtype=np.float64),
                n_grid=self.n_grid,
                compute_time=False,
                path=checked_path,
                source_state=self._delta_source_state,
                mpmt_types=self.mpmt_types,
            )
            self.curved_delta_evaluations += 1
            self._validate_physical_path(delta_path)
            curved_delta = np.ascontiguousarray(
                self.delta_scale * np.asarray(delta_mu, dtype=np.float64)
            )
            source_mu, source_t, source_path = curved_delta_source_field(
                self.path_emitter,
                self.delta_timing_pmt_positions,
                self.delta_timing_pmt_normals,
                np.zeros(self.n_modes, dtype=np.float64),
                n_grid=self.n_grid,
                path=checked_path,
                source_state=self._delta_source_state,
                mpmt_types=self.delta_timing_mpmt_types,
            )
            self.curved_delta_source_evaluations += 1
            self._validate_physical_path(source_path)
            source_mu = _rescale_source_rows_to_aggregate(
                self.delta_scale * np.asarray(source_mu, dtype=np.float64),
                curved_delta[self.active],
            )
            source_t = np.where(
                source_mu > 0.0, np.asarray(source_t, dtype=np.float64), np.inf
            )
        else:
            curved_delta = self.curved_zero_delta
            source_mu = None
            source_t = None
        return self._prediction_from_direct_field(
            mu_full,
            time_full,
            checked_path,
            curved_delta=curved_delta,
            curved_delta_node_mu=source_mu,
            curved_delta_node_t=source_t,
        )

    def charge_prediction(self, coefficients):
        """Return normalized direct-plus-delta charge without timing nodes."""
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(u, 10))
        cached = self.charge_cache.get(key)
        if cached is not None:
            return cached
        charge_field_keywords = {}
        if self._field_function is curved_primary_finite_disk_interval_field:
            charge_field_keywords["compute_moments"] = False
            charge_field_keywords["mpmt_types"] = self.coherent_mpmt_types
        mu_u, _t_u, _s_u, _path = self._field_function(
            self.path_emitter, self.coherent_pmt_positions,
            self.coherent_pmt_normals, u,
            n_grid=self.n_grid, aperture_radius_mm=self.aperture_radius_mm,
            **charge_field_keywords,
        )
        self.curved_evaluations += 1
        self._validate_physical_path(_path)
        mu_u_full = np.zeros_like(self.curved_zero_mu)
        mu_u_full[self.coherent_active_indices] = mu_u
        mu_u = mu_u_full
        delta_direct = mu_u - self.curved_zero_mu
        delta_delta = (
            self.curved_delta_charge_field(u, path=_path) - self.curved_zero_delta
        )
        raw_charge = np.maximum(
            self.base_raw_charge + delta_direct + delta_delta, 0.0
        )
        raw_mean = float(np.mean(raw_charge))
        norm, _event_conditioned = self._charge_normalization(raw_mean)
        out = np.maximum(raw_charge * norm, self.charge_floor)
        if self.boundary_interface_model is not None:
            out = self._profile_boundary_hardware(out)
        if len(self.charge_cache) >= self.charge_cache_max:
            zero = tuple(np.zeros(self.n_modes, dtype=np.float64))
            for old_key in tuple(self.charge_cache):
                if old_key != zero:
                    self.charge_cache.pop(old_key, None)
                    break
        self.charge_cache[key] = out
        return out

    def charge_prediction_from_path(self, path):
        """Return the charge prediction for an explicit physical trajectory.

        The accepted straight prediction, molecular transport, aperture model,
        event normalization, PMT response, and optional analytic delta field
        are identical to :meth:`charge_prediction`.  Only the source trajectory
        is supplied directly, which lets a non-Gaussian scattering prior reuse
        the validated optical likelihood without pretending its hard marks are
        Gaussian KL coefficients.
        """
        if not self.charge_only:
            raise RuntimeError("explicit-path prediction is currently charge-only")
        if self._field_function is not curved_primary_finite_disk_interval_field:
            raise NotImplementedError(
                "explicit-path prediction is implemented for support-tracked FALI"
            )
        mu_sparse, _time, _source_s, checked_path = (
            curved_primary_finite_disk_interval_path_field(
                self.path_emitter,
                self.coherent_pmt_positions,
                self.coherent_pmt_normals,
                path,
                aperture_radius_mm=self.aperture_radius_mm,
                compute_moments=False,
                mpmt_types=self.coherent_mpmt_types,
            )
        )
        self.curved_evaluations += 1
        self._validate_physical_path(checked_path)
        mu_full = np.zeros_like(self.curved_zero_mu)
        mu_full[self.coherent_active_indices] = mu_sparse
        delta_direct = mu_full - self.curved_zero_mu

        if self.curved_delta_enabled:
            delta_mu, _delta_time, delta_path = curved_delta_field(
                self.path_emitter,
                self.pmt_positions,
                self.pmt_normals,
                np.zeros(self.n_modes, dtype=np.float64),
                n_grid=self.n_grid,
                compute_time=False,
                path=checked_path,
                source_state=self._delta_source_state,
                mpmt_types=self.mpmt_types,
            )
            self.curved_delta_evaluations += 1
            self._validate_physical_path(delta_path)
            curved_delta = self.delta_scale * np.asarray(delta_mu, dtype=np.float64)
        else:
            curved_delta = self.curved_zero_delta
        delta_delta = curved_delta - self.curved_zero_delta
        raw_charge = np.maximum(
            self.base_raw_charge + delta_direct + delta_delta, 0.0
        )
        norm, _event_conditioned = self._charge_normalization(
            float(np.mean(raw_charge))
        )
        out = np.maximum(raw_charge * norm, self.charge_floor)
        if self.boundary_interface_model is not None:
            out = self._profile_boundary_hardware(out)
        return out

    def charge_data_nll_from_path(self, path):
        """Configured production charge NLL for an explicit physical path."""
        mu = np.maximum(self.charge_prediction_from_path(path), 1.0e-300)
        if self.pmt_model is not None:
            return float(
                self.pmt_model.get_neg_log_likelihood_npe(mu, self.obs_pes)
            )
        q = np.asarray(self.obs_pes, dtype=np.float64)
        return float(np.sum(mu - q * np.log(mu)))

    def data_nll_from_path(self, path, *, t0=None):
        """Configured charge-plus-time NLL for an explicit physical path."""
        exp_pes, timing_pes, timing, *_ = self.prediction_from_path(path)
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

    def charge_prediction_and_jacobian(self, coefficients):
        """Return normalized charge and the coherent KL Jacobian.

        Both the direct finite-disk and dense delta derivatives are analytic.
        The combined derivative includes molecular survival, the non-negative raw-
        component guard, event-total normalization and the production charge
        floor; it is therefore the derivative of ``charge_prediction`` itself.
        """
        if self.force_finite_difference_charge_jacobian:
            raise NotImplementedError(
                "the profiled mPMT-hardware nuisance response requires the exact "
                "finite-difference coherent Jacobian"
            )
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
                mpmt_types=self.coherent_mpmt_types,
            )
        )
        self.curved_evaluations += 1
        self._validate_physical_path(_path)
        mu_full = np.zeros_like(self.curved_zero_mu)
        jac_full = np.zeros((mu_full.size, self.n_modes), dtype=np.float64)
        mu_full[self.coherent_active_indices] = mu_sparse
        jac_full[self.coherent_active_indices, :] = jac_sparse

        delta_direct = mu_full - self.curved_zero_mu
        direct_jac = jac_full
        delta_delta = (
            self.curved_delta_charge_field(u, path=_path) - self.curved_zero_delta
        )
        delta_jac = self.curved_delta_charge_jacobian(u, path=_path)
        raw_unclipped = (
            self.base_raw_charge + delta_direct + delta_delta
        )
        positive = raw_unclipped > 0.0
        raw = np.maximum(raw_unclipped, 0.0)
        draw = direct_jac + delta_jac
        draw[~positive, :] = 0.0
        raw_mean = float(np.mean(raw))
        if raw_mean <= 0.0:
            out = np.full_like(raw, self.charge_floor)
            return out, np.zeros_like(draw)
        norm, event_conditioned = self._charge_normalization(raw_mean)
        if event_conditioned:
            mean_draw = np.mean(draw, axis=0)
            dnorm = -(norm / raw_mean) * mean_draw
        else:
            dnorm = np.zeros(self.n_modes, dtype=np.float64)
        unfloored = raw * norm
        dout = norm * draw + raw[:, None] * dnorm[None, :]
        floor_mask = unfloored <= self.charge_floor
        out = np.maximum(unfloored, self.charge_floor)
        dout[floor_mask, :] = 0.0
        return np.ascontiguousarray(out), np.ascontiguousarray(dout)

    def charge_time_response_and_gradient(
        self,
        coefficients,
        *,
        t0=None,
        likelihood_fd_step=1.0e-4,
        evaluate_centre_nll=True,
    ):
        """Return the exact local charge response and full data-NLL gradient.

        Curved direct-light moments and source-resolved delta amplitudes/times,
        including derivatives with respect to every standardized Fermi--Eyges
        coordinate, are evaluated in compiled receiver passes.  The existing
        production PMT likelihood is then differentiated through those local
        physical moments with a symmetric numerical directional derivative.
        No optical transport is approximated or repeated for the likelihood
        stencil; accepted optimizer steps are still checked with
        :meth:`data_nll` at the proposed nonlinear path.
        """
        if self.charge_only:
            raise RuntimeError(
                "charge-time response is unavailable in charge-only mode"
            )
        if self._field_function is not curved_primary_finite_disk_interval_field:
            raise NotImplementedError(
                "analytic charge-time response requires support-tracked FALI"
            )
        h = float(likelihood_fd_step)
        if not math.isfinite(h) or h <= 0.0:
            raise ValueError("likelihood_fd_step must be positive and finite")
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        event_t0 = self.t0 if t0 is None else float(t0)
        response_key = (
            u.tobytes(),
            float(event_t0),
            float(h),
            bool(evaluate_centre_nll),
            str(os.environ.get("LF_COHERENT_ANALYTIC_CHARGE_SCORE", "0")),
            str(os.environ.get("LF_COHERENT_BATCHED_TIMING_RESPONSE", "0")),
            str(os.environ.get("LF_COHERENT_DEFERRED_RESPONSE_BATCH", "0")),
            str(os.environ.get("LF_COHERENT_REFLECTION_WORKSPACE", "0")),
        )
        cached_response = self.response_gradient_cache.get(response_key)
        if cached_response is not None:
            self.response_gradient_cache_hits += 1
            return cached_response
        (
            direct_sparse,
            direct_jac_sparse,
            direct_time_sparse,
            direct_time_jac_sparse,
            path,
        ) = curved_primary_finite_disk_interval_response_jacobian_field(
            self.path_emitter,
            self.coherent_pmt_positions,
            self.coherent_pmt_normals,
            u,
            n_grid=self.n_grid,
            aperture_radius_mm=self.aperture_radius_mm,
            mpmt_types=self.coherent_mpmt_types,
        )
        self.curved_evaluations += 1
        self._validate_physical_path(path)

        direct = np.zeros_like(self.curved_zero_mu)
        direct_time = np.full_like(self.curved_zero_t, np.nan)
        direct_jac = np.zeros((direct.size, self.n_modes), dtype=np.float64)
        direct_time_jac = np.zeros_like(direct_jac)
        active = self.coherent_active_indices
        direct[active] = direct_sparse
        direct_time[active] = direct_time_sparse
        direct_jac[active, :] = direct_jac_sparse
        direct_time_jac[active, :] = direct_time_jac_sparse

        if self.curved_delta_enabled:
            curved_delta = self.curved_delta_charge_field(u, path=path)
            curved_delta_jac = self.curved_delta_charge_jacobian(u, path=path)
            (
                delta_node_mu,
                delta_node_t,
                delta_node_mu_jac,
                delta_node_t_jac,
                delta_source_path,
            ) = curved_delta_source_response_jacobian_field(
                self.path_emitter,
                self.delta_timing_pmt_positions,
                self.delta_timing_pmt_normals,
                u,
                n_grid=self.n_grid,
                path=path,
                source_state=self._delta_source_state,
                mpmt_types=self.delta_timing_mpmt_types,
            )
            self.curved_delta_source_jacobian_evaluations += 1
            self._validate_physical_path(delta_source_path)
            delta_node_mu, delta_node_mu_jac = (
                _rescale_source_rows_to_aggregate(
                    self.delta_scale * np.asarray(
                        delta_node_mu, dtype=np.float64
                    ),
                    curved_delta[self.active],
                    node_mu_jac=self.delta_scale * np.asarray(
                        delta_node_mu_jac, dtype=np.float64
                    ),
                    aggregate_mu_jac=curved_delta_jac[self.active, :],
                )
            )
            delta_node_t = np.asarray(delta_node_t, dtype=np.float64)
            delta_node_t_jac = np.asarray(
                delta_node_t_jac, dtype=np.float64
            )
            absent = delta_node_mu <= 0.0
            delta_node_t = np.where(absent, np.inf, delta_node_t)
            delta_node_t_jac = np.array(delta_node_t_jac, copy=True)
            delta_node_t_jac[absent, :] = 0.0
        else:
            curved_delta = self.curved_zero_delta
            curved_delta_jac = np.zeros(
                (curved_delta.size, self.n_modes), dtype=np.float64
            )
            delta_node_mu = None
            delta_node_t = None
            delta_node_mu_jac = None
            delta_node_t_jac = None

        def prediction_nll(
            path_mu,
            path_time,
            delta_mu,
            delta_source_mu,
            delta_source_t,
        ):
            prediction = self._prediction_from_direct_field(
                path_mu,
                path_time,
                path,
                curved_delta=delta_mu,
                curved_delta_node_mu=delta_source_mu,
                curved_delta_node_t=delta_source_t,
            )
            exp_pes, timing_pes, timing = prediction[:3]
            shifted = (
                timing
                if event_t0 == 0.0
                else shift_timing_prediction(timing, event_t0)
            )
            value = float(
                self.pmt_model.get_neg_log_likelihood_npe_t(
                    exp_pes,
                    self.obs_pes,
                    shifted,
                    self.obs_ts,
                    timing_pes=timing_pes,
                )
            )
            return prediction, value

        if bool(evaluate_centre_nll):
            centre, centre_nll = prediction_nll(
                direct,
                direct_time,
                curved_delta,
                delta_node_mu,
                delta_node_t,
            )
        else:
            # The joint solver has already evaluated and profiled the exact
            # centre objective.  It consumes only this response and gradient,
            # so a second detector-likelihood pass would be dead work.
            centre = self._prediction_from_direct_field(
                direct,
                direct_time,
                path,
                curved_delta=curved_delta,
                curved_delta_node_mu=delta_node_mu,
                curved_delta_node_t=delta_node_t,
            )
            centre_nll = float("nan")
        use_analytic_charge_score = str(
            os.environ.get("LF_COHERENT_ANALYTIC_CHARGE_SCORE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        charge_score = None
        if use_analytic_charge_score:
            score_interface = getattr(
                self.pmt_model,
                "get_neg_log_likelihood_npe_with_score",
                None,
            )
            if callable(score_interface):
                _charge_nll, charge_score = score_interface(
                    np.asarray(centre[0], dtype=np.float64),
                    self.obs_pes,
                )
                charge_score = np.asarray(charge_score, dtype=np.float64)
                if charge_score.shape != np.asarray(centre[0]).shape:
                    charge_score = None
        charge_jacobian = np.empty(
            (direct.size, self.n_modes), dtype=np.float64
        )
        gradient = np.empty(self.n_modes, dtype=np.float64)
        batched_timing_enabled = str(
            os.environ.get("LF_COHERENT_BATCHED_TIMING_RESPONSE", "0")
        ).strip().lower() not in {"0", "false", "no", "off"}
        deferred_batch_enabled = str(
            os.environ.get("LF_COHERENT_DEFERRED_RESPONSE_BATCH", "0")
        ).strip().lower() not in {"0", "false", "no", "off"}
        prediction_batch_interface = (
            None
            if charge_score is None or not batched_timing_enabled
            else getattr(
                self.pmt_model,
                "get_neg_log_likelihood_t_many_predictions",
                None,
            )
        )
        deferred_batch_interface = (
            None
            if (
                charge_score is None
                or not batched_timing_enabled
                or not deferred_batch_enabled
                or self.boundary_interface_model is not None
            )
            else getattr(
                self.pmt_model,
                "get_neg_log_likelihood_t_many_deferred_responses",
                None,
            )
        )
        use_deferred_batch = callable(deferred_batch_interface)
        use_prediction_batch = (
            not use_deferred_batch and callable(prediction_batch_interface)
        )
        use_timing_batch = use_deferred_batch or use_prediction_batch
        reflection_workspace_enabled = str(
            os.environ.get("LF_COHERENT_REFLECTION_WORKSPACE", "0")
        ).strip().lower() not in {"0", "false", "no", "off"}
        reflection_workspace_evaluator = getattr(
            self.pmt_model,
            "get_neg_log_likelihood_t_with_reflection_workspace",
            None,
        )
        reflection_workspace = None
        if (
            charge_score is not None
            and reflection_workspace_enabled
            and not use_timing_batch
            and self.boundary_interface_model is None
            and callable(reflection_workspace_evaluator)
        ):
            if not hasattr(self, "_first_arrival_reflection_workspace"):
                prepare_workspace = getattr(
                    self.pmt_model,
                    "prepare_first_arrival_reflection_workspace",
                    None,
                )
                self._first_arrival_reflection_workspace = (
                    prepare_workspace(
                        self.base_timing,
                        self.obs_pes,
                        self.obs_ts,
                    )
                    if callable(prepare_workspace)
                    else None
                )
            reflection_workspace = self._first_arrival_reflection_workspace
        use_reflection_workspace = reflection_workspace is not None
        response_variants = []
        if use_deferred_batch:
            n_variants = 2 * self.n_modes
            deferred_mu_variants = np.array(
                np.broadcast_to(
                    self.base_deferred_mu,
                    (n_variants,) + self.base_deferred_mu.shape,
                ),
                copy=True,
            )
            deferred_t_variants = np.array(
                np.broadcast_to(
                    self.base_deferred_t,
                    (n_variants,) + self.base_deferred_t.shape,
                ),
                copy=True,
            )
            deferred_node_scales = np.empty(n_variants, dtype=np.float64)
        else:
            deferred_mu_variants = None
            deferred_t_variants = None
            deferred_node_scales = None
        for mode in range(self.n_modes):
            plus_direct = direct + h * direct_jac[:, mode]
            plus_time = direct_time + h * direct_time_jac[:, mode]
            minus_direct = direct - h * direct_jac[:, mode]
            minus_time = direct_time - h * direct_time_jac[:, mode]
            plus_delta = curved_delta + h * curved_delta_jac[:, mode]
            minus_delta = curved_delta - h * curved_delta_jac[:, mode]
            if self.curved_delta_enabled:
                plus_delta_node_mu = (
                    delta_node_mu + h * delta_node_mu_jac[:, :, mode]
                )
                minus_delta_node_mu = (
                    delta_node_mu - h * delta_node_mu_jac[:, :, mode]
                )
                plus_delta_node_t = (
                    delta_node_t + h * delta_node_t_jac[:, :, mode]
                )
                minus_delta_node_t = (
                    delta_node_t - h * delta_node_t_jac[:, :, mode]
                )
            else:
                plus_delta_node_mu = None
                minus_delta_node_mu = None
                plus_delta_node_t = None
                minus_delta_node_t = None
            if charge_score is None:
                plus, plus_nll = prediction_nll(
                    plus_direct,
                    plus_time,
                    plus_delta,
                    plus_delta_node_mu,
                    plus_delta_node_t,
                )
                minus, minus_nll = prediction_nll(
                    minus_direct,
                    minus_time,
                    minus_delta,
                    minus_delta_node_mu,
                    minus_delta_node_t,
                )
                plus_exp = plus[0]
                minus_exp = minus[0]
            elif use_deferred_batch:
                plus_rows = self._deferred_response_rows_from_direct_field(
                    plus_direct,
                    plus_time,
                    curved_delta=plus_delta,
                    curved_delta_node_mu=plus_delta_node_mu,
                    curved_delta_node_t=plus_delta_node_t,
                )
                minus_rows = self._deferred_response_rows_from_direct_field(
                    minus_direct,
                    minus_time,
                    curved_delta=minus_delta,
                    curved_delta_node_mu=minus_delta_node_mu,
                    curved_delta_node_t=minus_delta_node_t,
                )
                plus_index = 2 * mode
                minus_index = plus_index + 1
                plus_exp = plus_rows[0]
                minus_exp = minus_rows[0]
                deferred_mu_variants[plus_index, 0] = plus_rows[1]
                deferred_t_variants[plus_index, 0] = plus_rows[2]
                deferred_mu_variants[minus_index, 0] = minus_rows[1]
                deferred_t_variants[minus_index, 0] = minus_rows[2]
                if self.curved_delta_enabled:
                    stop = 1 + self.n_delta_sources
                    deferred_mu_variants[plus_index, 1:stop] = plus_rows[3]
                    deferred_t_variants[plus_index, 1:stop] = plus_rows[4]
                    deferred_mu_variants[minus_index, 1:stop] = minus_rows[3]
                    deferred_t_variants[minus_index, 1:stop] = minus_rows[4]
                deferred_node_scales[plus_index] = plus_rows[5]
                deferred_node_scales[minus_index] = minus_rows[5]
            else:
                plus = self._prediction_from_direct_field(
                    plus_direct,
                    plus_time,
                    path,
                    curved_delta=plus_delta,
                    curved_delta_node_mu=plus_delta_node_mu,
                    curved_delta_node_t=plus_delta_node_t,
                )
                minus = self._prediction_from_direct_field(
                    minus_direct,
                    minus_time,
                    path,
                    curved_delta=minus_delta,
                    curved_delta_node_mu=minus_delta_node_mu,
                    curved_delta_node_t=minus_delta_node_t,
                )
                plus_exp = plus[0]
                minus_exp = minus[0]
                if use_prediction_batch:
                    response_variants.extend((plus, minus))
                elif use_reflection_workspace:
                    plus_nll = reflection_workspace_evaluator(
                        plus[2],
                        reflection_workspace,
                        model_time_shift_ns=event_t0,
                    )
                    minus_nll = reflection_workspace_evaluator(
                        minus[2],
                        reflection_workspace,
                        model_time_shift_ns=event_t0,
                    )
                    if plus_nll is None or minus_nll is None:
                        plus_nll = float(
                            self.pmt_model.get_neg_log_likelihood_t(
                                plus[0],
                                self.obs_pes,
                                plus[2],
                                self.obs_ts,
                                timing_pes=plus[1],
                                model_time_shift_ns=event_t0,
                            )
                        )
                        minus_nll = float(
                            self.pmt_model.get_neg_log_likelihood_t(
                                minus[0],
                                self.obs_pes,
                                minus[2],
                                self.obs_ts,
                                timing_pes=minus[1],
                                model_time_shift_ns=event_t0,
                            )
                        )
                    elif str(
                        os.environ.get(
                            "LF_COHERENT_AUDIT_REFLECTION_WORKSPACE", "0"
                        )
                    ).strip().lower() in {"1", "true", "yes", "on"}:
                        scalar_plus = float(
                            self.pmt_model.get_neg_log_likelihood_t(
                                plus[0],
                                self.obs_pes,
                                plus[2],
                                self.obs_ts,
                                timing_pes=plus[1],
                                model_time_shift_ns=event_t0,
                            )
                        )
                        scalar_minus = float(
                            self.pmt_model.get_neg_log_likelihood_t(
                                minus[0],
                                self.obs_pes,
                                minus[2],
                                self.obs_ts,
                                timing_pes=minus[1],
                                model_time_shift_ns=event_t0,
                            )
                        )
                        if (
                            np.float64(plus_nll).view(np.uint64)
                            != np.float64(scalar_plus).view(np.uint64)
                            or np.float64(minus_nll).view(np.uint64)
                            != np.float64(scalar_minus).view(np.uint64)
                        ):
                            raise AssertionError(
                                "reflection-workspace timing audit failed: "
                                f"mode={mode} plus={plus_nll!r}/"
                                f"{scalar_plus!r} minus={minus_nll!r}/"
                                f"{scalar_minus!r}"
                            )
                else:
                    plus_nll = float(
                        self.pmt_model.get_neg_log_likelihood_t(
                            plus[0],
                            self.obs_pes,
                            plus[2],
                            self.obs_ts,
                            timing_pes=plus[1],
                            model_time_shift_ns=event_t0,
                        )
                    )
                    minus_nll = float(
                        self.pmt_model.get_neg_log_likelihood_t(
                            minus[0],
                            self.obs_pes,
                            minus[2],
                            self.obs_ts,
                            timing_pes=minus[1],
                            model_time_shift_ns=event_t0,
                        )
                    )
            charge_jacobian[:, mode] = (
                np.asarray(plus_exp, dtype=np.float64)
                - np.asarray(minus_exp, dtype=np.float64)
            ) / (2.0 * h)
            if not use_timing_batch:
                gradient[mode] = (plus_nll - minus_nll) / (2.0 * h)
            if charge_score is not None and not use_timing_batch:
                gradient[mode] += float(
                    charge_jacobian[:, mode] @ charge_score
                )
        if use_timing_batch:
            if use_deferred_batch:
                timing_nll = np.asarray(
                    deferred_batch_interface(
                        deferred_mu_variants,
                        deferred_t_variants,
                        self.base_timing,
                        deferred_node_scales,
                        self.obs_pes,
                        self.obs_ts,
                        model_time_shift_ns=event_t0,
                    ),
                    dtype=np.float64,
                )
            else:
                timing_nll = np.asarray(
                    prediction_batch_interface(
                        [prediction[0] for prediction in response_variants],
                        self.obs_pes,
                        [prediction[2] for prediction in response_variants],
                        self.obs_ts,
                        timing_pes_variants=[
                            prediction[1] for prediction in response_variants
                        ],
                        model_time_shift_ns=event_t0,
                    ),
                    dtype=np.float64,
                )
            if timing_nll.shape != (2 * self.n_modes,):
                raise RuntimeError(
                    "batched coherent timing response returned an invalid shape"
                )
            for mode in range(self.n_modes):
                gradient[mode] = (
                    timing_nll[2 * mode] - timing_nll[2 * mode + 1]
                ) / (2.0 * h)
                gradient[mode] += float(
                    charge_jacobian[:, mode] @ charge_score
                )
            self.batched_timing_response_evaluations += 1
        result = (
            np.ascontiguousarray(centre[0], dtype=np.float64),
            np.ascontiguousarray(charge_jacobian),
            float(centre_nll),
            np.ascontiguousarray(gradient),
            centre,
        )
        if len(self.response_gradient_cache) >= 4:
            self.response_gradient_cache.pop(
                next(iter(self.response_gradient_cache)), None
            )
        self.response_gradient_cache[response_key] = result
        return result

    def charge_data_nll(self, coefficients):
        """Configured production charge NLL for one coherent path."""
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(u, 10))
        if _EXACT_CHARGE_NLL_REUSE:
            cached = self.charge_data_nll_cache.get(key)
            if cached is not None:
                return float(cached)
        mu = np.maximum(self.charge_prediction(u), 1.0e-300)
        if self.pmt_model is not None:
            value = float(
                self.pmt_model.get_neg_log_likelihood_npe(mu, self.obs_pes)
            )
        else:
            q = np.asarray(self.obs_pes, dtype=np.float64)
            value = float(np.sum(mu - q * np.log(mu)))
        if _EXACT_CHARGE_NLL_REUSE:
            self._insert_preserving_zero(
                self.charge_data_nll_cache,
                key,
                value,
                self.charge_nll_cache_max,
            )
        return value

    def charge_data_nll_and_score(self, coefficients):
        """Return configured charge NLL and score versus predicted PMT rates."""
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(u, 10))
        mu = np.maximum(self.charge_prediction(u), 1.0e-300)
        value, score = self.charge_data_nll_and_score_from_prediction(mu)
        if _EXACT_CHARGE_NLL_REUSE:
            self._insert_preserving_zero(
                self.charge_data_nll_cache,
                key,
                float(value),
                self.charge_nll_cache_max,
            )
        return value, score

    def charge_data_nll_and_score_from_prediction(self, prediction):
        """Return charge NLL and score for an already evaluated prediction.

        The latent solver obtains the prediction and its coherent-path
        Jacobian together.  Accepting that same prediction here avoids an
        otherwise redundant cache lookup (or field evaluation in alternate
        model implementations) while preserving the configured detector
        likelihood exactly.
        """
        mu = np.maximum(np.asarray(prediction, dtype=np.float64), 1.0e-300)
        score_interface = (
            None if self.pmt_model is None else getattr(
                self.pmt_model, "get_neg_log_likelihood_npe_with_score", None
            )
        )
        if score_interface is not None:
            value, score = score_interface(mu, self.obs_pes)
            return float(value), np.ascontiguousarray(score, dtype=np.float64)
        q = np.asarray(self.obs_pes, dtype=np.float64)
        return (
            float(np.sum(mu - q * np.log(mu))),
            np.ascontiguousarray(1.0 - q / mu),
        )

    def _complete_prediction_charge_nll(self, coefficients, exp_pes) -> float:
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        key = tuple(np.round(u, 10))
        if _EXACT_CHARGE_NLL_REUSE:
            cached = self.complete_charge_nll_cache.get(key)
            if cached is not None:
                return float(cached)
        value = float(
            self.pmt_model.get_neg_log_likelihood_npe(
                exp_pes, self.obs_pes
            )
        )
        if _EXACT_CHARGE_NLL_REUSE:
            self._insert_preserving_zero(
                self.complete_charge_nll_cache,
                key,
                value,
                self.charge_nll_cache_max,
            )
        return value

    def data_nll(self, coefficients, *, t0=None):
        exp_pes, timing_pes, timing, *_ = self.prediction(coefficients)
        dt = self.t0 if t0 is None else float(t0)
        if dt != 0.0:
            timing = shift_timing_prediction(timing, dt)
        if (
            _EXACT_CHARGE_NLL_REUSE
            and hasattr(timing, "first_arrival_active_indices")
        ):
            charge_nll = self._complete_prediction_charge_nll(
                coefficients, exp_pes
            )
            timing_nll = self.pmt_model.get_neg_log_likelihood_t(
                exp_pes,
                self.obs_pes,
                timing,
                self.obs_ts,
                timing_pes=timing_pes,
            )
            return float(charge_nll) + float(timing_nll)
        return float(
            self.pmt_model.get_neg_log_likelihood_npe_t(
                exp_pes,
                self.obs_pes,
                timing,
                self.obs_ts,
                timing_pes=timing_pes,
            )
        )

    def data_nll_many_t0(self, coefficients, t0_values):
        """Exact configured charge-plus-time NLL for several event offsets.

        The curved optical field is evaluated once.  Only the inexpensive
        first-arrival likelihood is repeated, using the PMT model's vectorized
        time-shift interface when available.
        """
        exp_pes, timing_pes, timing, *_ = self.prediction(coefficients)
        shifts = np.ascontiguousarray(t0_values, dtype=np.float64).reshape(-1)
        interface = getattr(
            self.pmt_model, "get_neg_log_likelihood_t_many_t0", None
        )
        if (
            _EXACT_CHARGE_NLL_REUSE
            and interface is not None
            and hasattr(timing, "first_arrival_active_indices")
        ):
            charge_nll = self._complete_prediction_charge_nll(
                coefficients, exp_pes
            )
            timing_values = np.asarray(
                interface(
                    exp_pes,
                    self.obs_pes,
                    timing,
                    self.obs_ts,
                    shifts,
                    timing_pes=timing_pes,
                ),
                dtype=np.float64,
            )
            return timing_values + float(charge_nll)
        combined_interface = getattr(
            self.pmt_model, "get_neg_log_likelihood_npe_t_many_t0", None
        )
        if combined_interface is not None:
            return np.asarray(
                combined_interface(
                    exp_pes,
                    self.obs_pes,
                    timing,
                    self.obs_ts,
                    shifts,
                    timing_pes=timing_pes,
                ),
                dtype=np.float64,
            )
        return np.asarray(
            [self.data_nll(coefficients, t0=float(value)) for value in shifts],
            dtype=np.float64,
        )

    def __call__(self, coefficients):
        self.calls += 1
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        if np.any(~np.isfinite(u)):
            return 1.0e30
        data_value = (
            self.charge_data_nll(u) if self.charge_only else self.data_nll(u)
        )
        value = data_value + 0.5 * float(u @ u)
        return value if math.isfinite(value) else 1.0e30
