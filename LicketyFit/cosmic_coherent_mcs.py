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
import traceback
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from .mcs_coherent_objective import FixedTrackCoherentMCSObjective
from .mcs_latent_profile import (
    ProfileIteration,
    profiled_charge_track_step_directions,
    solve_latent_charge_map,
)
from .mcs_joint_energy_range import (
    EnergyPathGuide,
    GaussianPathGuide,
    JointEnergyRangeSMCConfig,
    JointEnergyRangeSMCResult,
    run_joint_energy_range_smc,
)
from .mcs_joint_laplace import (
    JointLaplaceCubatureConfig,
    _solve_latent_with_central_response,
    run_joint_laplace_cubature,
)
from .stopping_straggling import StoppingRangeStraggling
from .cosmic_track_fit import resolve_range_clipped_track
from .track_parameterization import (
    TangentDirectionChart,
    attach_direction_components,
    reanchor_values,
)


def _screen_prior_path_starts(
    model,
    *,
    dimension: int,
    draws: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Rank antithetic FE-prior draws by the exact latent posterior.

    This is proposal construction only.  Draws come from the model's unchanged
    standard-normal FE prior, and the exact charge likelihood plus analytic
    prior ranks them.  The later density bridge includes the complete normalized
    guide mixture, so screening can change efficiency but never the target.
    """
    dimension = int(dimension)
    draws = max(0, int(draws))
    candidates: list[np.ndarray] = []
    while len(candidates) < draws:
        draw = np.ascontiguousarray(rng.normal(size=dimension), dtype=np.float64)
        candidates.append(draw)
        if len(candidates) < draws:
            candidates.append(np.ascontiguousarray(-draw, dtype=np.float64))
    ranked: list[dict] = []
    for index, coefficients in enumerate(candidates):
        try:
            charge_nll = float(model.charge_data_nll(coefficients))
        except Exception:
            continue
        prior_nll = 0.5 * float(coefficients @ coefficients)
        posterior_nll = charge_nll + prior_nll
        if not math.isfinite(posterior_nll):
            continue
        ranked.append({
            "screen_index": int(index),
            "coefficients": coefficients,
            "charge_nll": charge_nll,
            "prior_nll": prior_nll,
            "posterior_nll": posterior_nll,
        })
    ranked.sort(key=lambda row: float(row["posterior_nll"]))
    return ranked


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
    selection_nll: float
    iterations: tuple[ProfileIteration, ...]
    latent_iterations: tuple
    converged: bool
    wall_s: float
    diagnostics: dict

    def output_values(self) -> dict:
        return attach_direction_components(self.updated_values, chart=self.updated_chart)


@dataclass(frozen=True)
class CoherentJointLengthIteration:
    """One exact charge--time profile step in remaining-range space."""

    cycle: int
    step_mm: float
    length_before_mm: float
    posterior_before: float
    candidate_lengths_mm: tuple[float, ...]
    candidate_posteriors: tuple[float, ...]
    length_after_mm: float
    posterior_after: float
    accepted: bool


@dataclass
class CosmicCoherentJointLengthResult:
    """Local coherent profile of the complete continuous event state.

    The discrete cosmic topology is conditioned on the preceding topology
    tournament, but the continuous global line is not frozen to the
    charge-only seed.  Line position, direction, remaining range, standardized
    Fermi--Eyges coordinates, and additive event time are all updated against
    one curved-path charge-plus-first-arrival posterior.  Every accepted block
    move therefore improves the same exact data likelihood plus physical
    path/range priors.
    """

    initial_values: dict
    updated_values: dict
    updated_chart: TangentDirectionChart
    coefficients_mean: np.ndarray
    coefficients_covariance: np.ndarray
    data_nll: float
    path_prior_nll: float
    range_prior_nll: float
    posterior_nll: float
    profiled_t0_ns: float
    iterations: tuple[CoherentJointLengthIteration, ...]
    converged: bool
    wall_s: float
    diagnostics: dict

    def output_values(self) -> dict:
        return attach_direction_components(
            self.updated_values, chart=self.updated_chart
        )


@dataclass
class CosmicJointEnergyRangeResult:
    """Truth-blind charge posterior for ``(K0,z_R,coherent path)``.

    The line geometry is conditioned on the values supplied by the caller.  In
    the first controlled gate those values are the independently measured beam
    entry and direction; no event truth is accepted by this API.
    """

    initial_values: dict
    updated_values: dict
    updated_chart: TangentDirectionChart
    initial_resolved: object
    updated_resolved: object
    smc: JointEnergyRangeSMCResult
    coefficients_mean: np.ndarray
    coefficients_covariance: np.ndarray
    guide_diagnostics: dict
    map_sample_index: int
    map_charge_nll: float
    wall_s: float
    diagnostics: dict

    def output_values(self) -> dict:
        return attach_direction_components(self.updated_values, chart=self.updated_chart)

    def summary(self) -> dict:
        out = self.smc.summary()
        out.update({
            "map_sample_index": int(self.map_sample_index),
            "map_charge_nll": float(self.map_charge_nll),
            "wall_s": float(self.wall_s),
        })
        return out


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
        fixed_initial_kinetic_energy_mev: float | None = None,
        model_cache_size: int | None = None,
        charge_only: bool = True,
        boundary_interface_model=None,
        boundary_interface_timing_policy: str = "mask_module",
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
        self.fixed_initial_kinetic_energy_mev = (
            None
            if fixed_initial_kinetic_energy_mev is None
            else float(fixed_initial_kinetic_energy_mev)
        )
        self.charge_only = bool(charge_only)
        self.boundary_interface_model = boundary_interface_model
        self.boundary_interface_timing_policy = str(
            boundary_interface_timing_policy
        ).strip().lower()
        self.model_cache_size = (
            None if model_cache_size is None else int(model_cache_size)
        )
        if self.model_cache_size is not None and self.model_cache_size < 1:
            raise ValueError("model_cache_size must be positive or None")
        if (
            self.fixed_initial_kinetic_energy_mev is not None
            and (
                not math.isfinite(self.fixed_initial_kinetic_energy_mev)
                or self.fixed_initial_kinetic_energy_mev <= 0.0
            )
        ):
            raise ValueError("fixed initial kinetic energy must be positive and finite")
        self._models: dict[tuple[float, ...], FixedTrackCoherentMCSObjective] = {}
        self._resolved: dict[tuple[float, ...], object] = {}
        self._model_build_count = 0
        self._model_build_wall_s = 0.0
        self._model_cache_evictions = 0
        self._evicted_coherent_field_evaluations = 0
        self.last_model_build_error: str | None = None
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
    def _model_key(
        arr: np.ndarray,
        resolved,
        initial_kinetic_energy_mev: float,
    ) -> tuple[float, ...]:
        # Preserve the exact continuous (geometry, range, K0) state.  ``n_grid``
        # controls only FE/optical quadrature and must not define posterior
        # support or cache bins for the inferred range or energy.
        return tuple(
            map(
                float,
                np.concatenate(
                    (
                        np.asarray(resolved.start, dtype=np.float64),
                        np.asarray(resolved.direction, dtype=np.float64),
                        np.asarray(
                            [resolved.visible_length_mm, resolved.full_range_mm],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            [initial_kinetic_energy_mev], dtype=np.float64
                        ),
                    )
                ),
            )
        )

    def resolved_track(self, theta: Sequence[float]):
        valid = self._resolve(theta)
        return None if valid is None else valid[1]

    def model(
        self,
        theta: Sequence[float],
        *,
        initial_kinetic_energy_mev: float | None = None,
    ) -> FixedTrackCoherentMCSObjective | None:
        valid = self._resolve(theta)
        if valid is None:
            self.invalid_evaluations += 1
            return None
        arr, resolved = valid
        if initial_kinetic_energy_mev is not None:
            ke0 = float(initial_kinetic_energy_mev)
        elif self.fixed_initial_kinetic_energy_mev is not None:
            ke0 = float(self.fixed_initial_kinetic_energy_mev)
        else:
            ke0 = float(self.range_lookup.range_mm_to_energy(resolved.full_range_mm))
        if not math.isfinite(ke0) or ke0 <= 0.0:
            self.invalid_evaluations += 1
            return None
        key = self._model_key(arr, resolved, ke0)
        cached = self._models.get(key)
        if cached is not None:
            return cached
        build_wall0 = time.perf_counter()
        try:
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
                charge_only=self.charge_only,
                range_clipped_track=resolved,
                range_lookup=self.range_lookup,
                boundary_interface_model=self.boundary_interface_model,
                boundary_interface_timing_policy=(
                    self.boundary_interface_timing_policy
                ),
            )
        except Exception as error:
            self._model_build_wall_s += float(
                time.perf_counter() - build_wall0
            )
            self.last_model_build_error = (
                repr(error) + "\n" + traceback.format_exc(limit=8)
            )
            self.invalid_evaluations += 1
            return None
        self._model_build_wall_s += float(time.perf_counter() - build_wall0)
        if (
            self.model_cache_size is not None
            and len(self._models) >= self.model_cache_size
        ):
            oldest_key = next(iter(self._models))
            oldest = self._models.pop(oldest_key)
            self._resolved.pop(oldest_key, None)
            self._evicted_coherent_field_evaluations += int(
                oldest.curved_evaluations
            )
            self._model_cache_evictions += 1
        self._models[key] = model
        self._resolved[key] = resolved
        self._model_build_count += 1
        return model

    def __call__(
        self,
        theta: Sequence[float],
        coefficients: Sequence[float],
        *,
        include_prior: bool = True,
        initial_kinetic_energy_mev: float | None = None,
    ) -> float:
        model = self.model(
            theta,
            initial_kinetic_energy_mev=initial_kinetic_energy_mev,
        )
        if model is None:
            return float("inf")
        u = np.asarray(coefficients, dtype=np.float64).reshape(self.n_modes)
        if np.any(~np.isfinite(u)):
            return float("inf")
        self.exact_evaluations += 1
        if self.charge_only:
            value = float(model.charge_data_nll(u))
        else:
            theta_array = self._theta_array(theta)
            value = float(model.data_nll(u, t0=float(theta_array[6])))
        if include_prior:
            value += 0.5 * float(u @ u)
        return value if math.isfinite(value) else float("inf")

    @property
    def optical_model_build_count(self) -> int:
        return int(self._model_build_count)

    @property
    def optical_model_build_wall_s(self) -> float:
        return float(self._model_build_wall_s)

    @property
    def model_cache_evictions(self) -> int:
        return int(self._model_cache_evictions)

    @property
    def coherent_field_evaluation_count(self) -> int:
        return int(
            self._evicted_coherent_field_evaluations
            + sum(model.curved_evaluations for model in self._models.values())
        )


def _run_cosmic_joint_energy_range_smc_update(
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
    kinetic_energy_bounds_mev: tuple[float, float],
    random_seed: int,
    mpmt_types=None,
    inset_mm: float = 0.5,
    range_limits: tuple[float, float] | None = None,
    t0_limits: tuple[float, float] | None = None,
    modes_per_plane: int = 12,
    grid_points: int = 41,
    particles: int = 32,
    target_ess_fraction: float = 0.80,
    resample_ess_fraction: float = 0.55,
    max_temperatures: int = 64,
    rejuvenation_steps: int = 3,
    posterior_rejuvenation_steps: int = 12,
    energy_random_walk_mev: float = 30.0,
    energy_independence_probability: float = 0.15,
    range_pcn_rho: float = 0.80,
    path_pcn_rho: float = 0.92,
    path_guide_rho: float = 0.75,
    path_guide_probability: float = 0.50,
    joint_guide_prior_mixture_probability: float = 0.20,
    joint_guide_initial_beta: float = 0.0,
    guide_latent_fd: float = 0.20,
    guide_latent_iterations: int = 80,
    guide_covariance_inflation: float = 4.0,
    guide_prior_screen_draws: int = 128,
    guide_prior_screen_refits: int = 3,
    guide_prior_screen_random_seed: int = 20260816,
    guide_laplace_uniform_mixture_probability: float = 0.10,
    guide_range_profile_cycles: int = 2,
    guide_energy_anchors_mev: Sequence[float] = (100.0, 200.0, 300.0, 450.0, 600.0),
    guide_energy_proposal_sd_mev: float = 30.0,
    sparse_neighbor_radius_mm: float = 100.0,
    initial_path_coefficients: Sequence[float] | None = None,
    charge_only: bool = True,
) -> CosmicJointEnergyRangeResult:
    """Marginalize continuous energy, stopping range, and coherent FE path.

    It deliberately conditions on the input line geometry.  ``charge_only``
    selects either the configured charge likelihood or the exact configured
    charge-plus-first-arrival-time likelihood.  The charge-only latent solves
    below are proposal construction only and do not alter the SMC target.
    """

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

    modes_per_plane = int(modes_per_plane)
    if modes_per_plane < 1:
        raise ValueError("modes_per_plane must be positive")
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
        n_modes=2 * modes_per_plane,
        n_grid=int(grid_points),
        aperture_radius_mm=float(
            getattr(template_emitter, "primary_endpoint_aperture_radius_mm", 45.0)
        ),
        sparse_receiver=True,
        sparse_neighbor_radius_mm=float(sparse_neighbor_radius_mm),
        charge_only=bool(charge_only),
    )
    straggling = StoppingRangeStraggling(
        particle=str(getattr(template_emitter, "particle_name", "muon"))
    )
    energy_low, energy_high = map(float, kinetic_energy_bounds_mev)
    if energy_low <= straggling.threshold_mev:
        raise ValueError(
            "kinetic-energy support must lie above the Cherenkov threshold"
        )

    # The straight charge solution supplies only a data-derived proposal guide.
    # It does not restrict the broad uniform energy support or enter the target.
    seed_energy = float(range_lookup.range_mm_to_energy(theta[5]))
    seed_energy = float(np.clip(seed_energy, energy_low, energy_high))
    seed_range = float(straggling.mean_range_mm(seed_energy))
    seed_z = float(straggling.z_range(seed_energy, theta[5]))
    guides: list[EnergyPathGuide] = []
    guide_log_scores: list[float] = []
    guide_basin_rows: list[dict] = []
    guide_diagnostics = {
        "enabled": False,
        "seed_energy_mev": seed_energy,
        "seed_z_range": seed_z,
        "seed_mean_range_mm": seed_range,
        "target_altering": False,
        "selection_policy": (
            "defensive Laplace-weighted mixture of exact-density proposal "
            "guides on a broad fixed energy ladder plus the data-derived "
            "straight seed"
        ),
        "anchors": [],
    }
    anchor_energies = np.concatenate((
        np.asarray(tuple(guide_energy_anchors_mev), dtype=np.float64),
        np.asarray((seed_energy,), dtype=np.float64),
    ))
    anchor_energies = np.clip(anchor_energies, energy_low, energy_high)
    anchor_energies = np.unique(np.round(anchor_energies, 10))
    inflation = float(guide_covariance_inflation)
    if not math.isfinite(inflation) or inflation <= 0.0:
        raise ValueError("guide covariance inflation must be positive")
    # Proposal construction has its own deterministic stream.  Posterior seed
    # comparisons must vary only SMC/resampling, not silently replace the set
    # of charge-derived proposal basins being compared.
    guide_rng = np.random.default_rng(int(guide_prior_screen_random_seed))
    guide_prior_screen_draws = max(0, int(guide_prior_screen_draws))
    guide_prior_screen_refits = max(0, int(guide_prior_screen_refits))
    guide_range_profile_cycles = max(0, int(guide_range_profile_cycles))
    latent_dimension = 2 * modes_per_plane
    propagated_path = None
    if initial_path_coefficients is not None:
        propagated_path = np.asarray(
            initial_path_coefficients, dtype=np.float64
        ).reshape(latent_dimension)
        if np.any(~np.isfinite(propagated_path)):
            raise ValueError("initial path coefficients must be finite")
        propagated_path = np.ascontiguousarray(propagated_path)
    guide_diagnostics["propagated_path_guide"] = bool(
        propagated_path is not None
    )
    for anchor_energy in anchor_energies:
        anchor = {
            "energy_mev": float(anchor_energy),
            "z_range": 0.0,
            "realized_range_mm": float(
                straggling.realized_range_mm(float(anchor_energy), 0.0)
            ),
        }
        guide_theta = theta.copy()
        guide_theta[5] = anchor["realized_range_mm"]
        guide_model = evaluator.model(
            guide_theta, initial_kinetic_energy_mev=float(anchor_energy)
        )
        if guide_model is None:
            anchor["failure"] = "invalid coherent model"
            guide_diagnostics["anchors"].append(anchor)
            continue
        screened = _screen_prior_path_starts(
            guide_model,
            dimension=latent_dimension,
            draws=guide_prior_screen_draws,
            rng=guide_rng,
        )
        # Preserve the historical straight-path start as one proposal basin,
        # then add the best exact-posterior FE-prior starts.  No event truth,
        # simulation template, fitted width, or energy label enters this set.
        starts: list[tuple[str, int, np.ndarray, float | None]] = [
            ("zero_path", -1, np.zeros(latent_dimension), None)
        ]
        if propagated_path is not None:
            starts.insert(
                0,
                (
                    "coherent_global_path",
                    -2,
                    propagated_path.copy(),
                    None,
                ),
            )
        starts.extend(
            (
                "screened_fe_prior",
                int(row["screen_index"]),
                np.asarray(row["coefficients"], dtype=np.float64),
                float(row["posterior_nll"]),
            )
            for row in screened[:guide_prior_screen_refits]
        )
        anchor["prior_screen_requested_draws"] = guide_prior_screen_draws
        anchor["prior_screen_valid_draws"] = int(len(screened))
        anchor["prior_screen_requested_refits"] = guide_prior_screen_refits
        anchor["prior_screen_best_posterior_nll"] = (
            None if not screened else float(screened[0]["posterior_nll"])
        )
        anchor["basins"] = []
        for start_kind, screen_index, initial_coefficients, screen_nll in starts:
            basin = {
                "start_kind": start_kind,
                "screen_index": int(screen_index),
                "screen_posterior_nll": screen_nll,
            }
            try:
                latent = solve_latent_charge_map(
                    guide_model,
                    initial_coefficients=initial_coefficients,
                    fd_step=float(guide_latent_fd),
                    max_iterations=int(guide_latent_iterations),
                    trust_max_component=1.0,
                )
                profiled_z = 0.0
                profiled_model = guide_model
                range_profile_evaluations = 0
                for _cycle in range(guide_range_profile_cycles):
                    fixed_coefficients = np.asarray(
                        latent.coefficients, dtype=np.float64
                    )

                    def conditional_range_posterior(z_value):
                        nonlocal range_profile_evaluations
                        range_profile_evaluations += 1
                        candidate_range = float(
                            straggling.realized_range_mm(
                                float(anchor_energy), float(z_value)
                            )
                        )
                        if not math.isfinite(candidate_range) or candidate_range <= 0.0:
                            return 1.0e30
                        candidate_theta = theta.copy()
                        candidate_theta[5] = candidate_range
                        candidate_model = evaluator.model(
                            candidate_theta,
                            initial_kinetic_energy_mev=float(anchor_energy),
                        )
                        if candidate_model is None:
                            return 1.0e30
                        try:
                            charge = float(
                                candidate_model.charge_data_nll(fixed_coefficients)
                            )
                        except Exception:
                            return 1.0e30
                        return float(
                            charge
                            + 0.5 * fixed_coefficients @ fixed_coefficients
                            + 0.5 * float(z_value) ** 2
                        )

                    range_result = minimize_scalar(
                        conditional_range_posterior,
                        bounds=(-3.0, 3.0),
                        method="bounded",
                        options={"xatol": 0.02, "maxiter": 32},
                    )
                    if not (
                        bool(range_result.success)
                        and math.isfinite(float(range_result.fun))
                        and float(range_result.fun) < 1.0e29
                    ):
                        break
                    candidate_z = float(range_result.x)
                    candidate_range = float(
                        straggling.realized_range_mm(
                            float(anchor_energy), candidate_z
                        )
                    )
                    candidate_theta = theta.copy()
                    candidate_theta[5] = candidate_range
                    candidate_model = evaluator.model(
                        candidate_theta,
                        initial_kinetic_energy_mev=float(anchor_energy),
                    )
                    if candidate_model is None:
                        break
                    profiled_z = candidate_z
                    profiled_model = candidate_model
                    latent = solve_latent_charge_map(
                        profiled_model,
                        initial_coefficients=fixed_coefficients,
                        fd_step=float(guide_latent_fd),
                        max_iterations=int(guide_latent_iterations),
                        trust_max_component=1.0,
                    )

                # Conditional curvature supplies only a proposal width.  The
                # physical z_R prior and exact optical likelihood remain in the
                # SMC target, and the complete Gaussian proposal density is
                # included in the bridge/MH corrections.
                curvature_step = 0.10
                final_coefficients = np.asarray(
                    latent.coefficients, dtype=np.float64
                )

                def final_conditional_value(z_value):
                    candidate_range = float(
                        straggling.realized_range_mm(
                            float(anchor_energy), float(z_value)
                        )
                    )
                    candidate_theta = theta.copy()
                    candidate_theta[5] = candidate_range
                    candidate_model = evaluator.model(
                        candidate_theta,
                        initial_kinetic_energy_mev=float(anchor_energy),
                    )
                    if candidate_model is None:
                        return math.inf
                    return float(
                        candidate_model.charge_data_nll(final_coefficients)
                        + 0.5 * final_coefficients @ final_coefficients
                        + 0.5 * float(z_value) ** 2
                    )

                center_value = final_conditional_value(profiled_z)
                plus_value = final_conditional_value(
                    profiled_z + curvature_step
                )
                minus_value = final_conditional_value(
                    profiled_z - curvature_step
                )
                z_curvature = float(
                    (plus_value - 2.0 * center_value + minus_value)
                    / (curvature_step * curvature_step)
                )
                if math.isfinite(z_curvature) and z_curvature > 0.0:
                    z_proposal_sd = float(
                        np.clip(1.0 / math.sqrt(z_curvature), 0.15, 1.5)
                    )
                else:
                    z_proposal_sd = 1.0
                covariance = inflation * np.asarray(
                    latent.covariance, dtype=np.float64
                )
                covariance += 1.0e-8 * np.eye(covariance.shape[0])
                guides.append(EnergyPathGuide(
                    kinetic_energy_mev=float(anchor_energy),
                    energy_proposal_sd_mev=float(guide_energy_proposal_sd_mev),
                    path=GaussianPathGuide(latent.coefficients, covariance),
                    z_range_mean=float(profiled_z),
                    z_range_proposal_sd=float(z_proposal_sd),
                ))
                joint_laplace_nll = float(
                    latent.laplace_nll
                    + 0.5 * profiled_z * profiled_z
                    - math.log(z_proposal_sd)
                )
                guide_log_scores.append(
                    -joint_laplace_nll
                    if math.isfinite(joint_laplace_nll)
                    else -float(
                        latent.posterior_nll + 0.5 * profiled_z * profiled_z
                    )
                )
                basin.update({
                    "guide_added": True,
                    "guide_index": int(len(guides) - 1),
                    "latent_converged": bool(latent.converged),
                    "latent_termination_reason": str(latent.termination_reason),
                    "latent_final_gradient_max_abs": float(
                        latent.final_gradient_max_abs
                    ),
                    "latent_iterations": int(len(latent.iterations)),
                    "coefficient_norm": float(
                        np.linalg.norm(latent.coefficients)
                    ),
                    "profiled_z_range": float(profiled_z),
                    "profiled_realized_range_mm": float(
                        straggling.realized_range_mm(
                            float(anchor_energy), profiled_z
                        )
                    ),
                    "z_range_proposal_sd": float(z_proposal_sd),
                    "z_range_conditional_curvature": float(z_curvature),
                    "range_profile_evaluations": int(
                        range_profile_evaluations + 3
                    ),
                    "range_profile_cycles": int(guide_range_profile_cycles),
                    "covariance_inflation": inflation,
                    "charge_nll": float(latent.charge_nll),
                    "posterior_nll": float(latent.posterior_nll),
                    "laplace_nll": float(latent.laplace_nll),
                    "joint_laplace_nll": float(joint_laplace_nll),
                    "logdet_information": float(latent.logdet_information),
                })
                guide_basin_rows.append(basin)
            except Exception as exc:
                # Every guide component is optional and exactly corrected.
                # Falling back to the remaining mixture changes efficiency,
                # never the physical posterior target.
                basin.update({
                    "guide_added": False,
                    "failure": f"{type(exc).__name__}: {exc}",
                })
            anchor["basins"].append(basin)
        if not any(row.get("guide_added", False) for row in anchor["basins"]):
            anchor["failure"] = "no valid guide basin at this energy anchor"
        guide_diagnostics["anchors"].append(anchor)
    defensive = float(guide_laplace_uniform_mixture_probability)
    if not 0.0 <= defensive <= 1.0:
        raise ValueError(
            "guide_laplace_uniform_mixture_probability must lie in [0,1]"
        )
    if guides:
        log_scores = np.asarray(guide_log_scores, dtype=np.float64)
        maximum = float(np.max(log_scores))
        laplace_weights = np.exp(np.clip(log_scores - maximum, -745.0, 0.0))
        laplace_weights /= float(np.sum(laplace_weights))
        mixture_weights = (
            (1.0 - defensive) * laplace_weights
            + defensive / float(len(guides))
        )
        guides = [
            EnergyPathGuide(
                guide.kinetic_energy_mev,
                guide.energy_proposal_sd_mev,
                guide.path,
                mixture_weight=float(weight),
                z_range_mean=float(guide.z_range_mean),
                z_range_proposal_sd=float(guide.z_range_proposal_sd),
            )
            for guide, weight in zip(guides, mixture_weights, strict=True)
        ]
        for basin, laplace_weight, mixture_weight in zip(
            guide_basin_rows,
            laplace_weights,
            mixture_weights,
            strict=True,
        ):
            basin["laplace_normalized_weight"] = float(laplace_weight)
            basin["proposal_mixture_weight"] = float(mixture_weight)
    guide_diagnostics["enabled"] = bool(guides)
    guide_diagnostics["guide_count"] = int(len(guides))
    guide_diagnostics["covariance_inflation"] = inflation
    guide_diagnostics["prior_screen_draws_per_anchor"] = int(
        guide_prior_screen_draws
    )
    guide_diagnostics["prior_screen_refits_per_anchor"] = int(
        guide_prior_screen_refits
    )
    guide_diagnostics["prior_screen_random_seed"] = int(
        guide_prior_screen_random_seed
    )
    guide_diagnostics["prior_screen_distribution"] = (
        "antithetic standard-normal FE prior"
    )
    guide_diagnostics["laplace_uniform_defensive_probability"] = defensive
    guide_diagnostics["range_profile_cycles"] = int(
        guide_range_profile_cycles
    )
    guide_diagnostics["range_profile_support_z"] = [-3.0, 3.0]
    guide_diagnostics["range_profile_support_role"] = (
        "proposal construction only; posterior z_R support remains unbounded"
    )
    guide_diagnostics["proposal_weighting"] = (
        "normalized exp(-local Laplace NLL) with a uniform defensive mixture; "
        "proposal density included exactly in the bridge and MH ratios"
    )

    def realized_range(k0: float, z_value: float) -> float:
        return float(straggling.realized_range_mm(k0, z_value))

    def exact_log_likelihood(
        k0: float, z_value: float, coefficients: np.ndarray
    ) -> float:
        candidate = theta.copy()
        candidate[5] = realized_range(k0, z_value)
        value = evaluator(
            candidate,
            coefficients,
            include_prior=False,
            initial_kinetic_energy_mev=float(k0),
        )
        return -float(value) if math.isfinite(value) else -math.inf

    smc = run_joint_energy_range_smc(
        exact_log_likelihood,
        realized_range,
        n_path_modes=2 * modes_per_plane,
        config=JointEnergyRangeSMCConfig(
            kinetic_energy_bounds_mev=(energy_low, energy_high),
            particles=int(particles),
            target_ess_fraction=float(target_ess_fraction),
            resample_ess_fraction=float(resample_ess_fraction),
            max_temperatures=int(max_temperatures),
            rejuvenation_steps=int(rejuvenation_steps),
            posterior_rejuvenation_steps=int(posterior_rejuvenation_steps),
            energy_random_walk_mev=float(energy_random_walk_mev),
            energy_independence_probability=float(
                energy_independence_probability
            ),
            range_pcn_rho=float(range_pcn_rho),
            path_pcn_rho=float(path_pcn_rho),
            path_guide_rho=float(path_guide_rho),
            path_guide_probability=float(path_guide_probability),
            joint_guide_prior_mixture_probability=float(
                joint_guide_prior_mixture_probability
            ),
            joint_guide_initial_beta=float(joint_guide_initial_beta),
            random_seed=int(random_seed),
        ),
        path_guide=tuple(guides),
    )
    posterior = smc.summary()
    posterior_range = float(posterior["realized_range_mm_mean"])
    posterior_energy = float(posterior["initial_kinetic_energy_mev_mean"])
    posterior_theta = theta.copy()
    posterior_theta[5] = posterior_range
    updated_resolved = evaluator.resolved_track(posterior_theta)
    if updated_resolved is None:
        raise RuntimeError("posterior-mean realised range is outside the selected topology")

    coefficient_mean = np.asarray(
        smc.weights @ smc.coefficients, dtype=np.float64
    )
    centered = smc.coefficients - coefficient_mean[None, :]
    coefficient_covariance = (centered * smc.weights[:, None]).T @ centered
    log_posterior = (
        smc.log_likelihood
        - 0.5 * smc.z_range**2
        - 0.5 * np.sum(smc.coefficients**2, axis=1)
    )
    map_index = int(np.argmax(log_posterior))
    updated_chart = TangentDirectionChart.from_direction(
        np.asarray(updated_resolved.direction, dtype=np.float64)
    )
    updated_values = {
        "x0": float(updated_resolved.start[0]),
        "y0": float(updated_resolved.start[1]),
        "z0": float(updated_resolved.start[2]),
        "dir_u": 0.0,
        "dir_v": 0.0,
        "length": posterior_range,
        "initial_kinetic_energy_mev": posterior_energy,
        "z_range": float(posterior["z_range_mean"]),
        "t0": float(theta[6]),
    }
    diagnostics = {
        "implementation": "continuous_noncentered_energy_range_fe_smc_v1",
        "inference_method": "reference_smc",
        "line_geometry_conditioned": True,
        "starts_at_boundary": bool(starts_at_boundary),
        "exits_detector": bool(initial_resolved.exits_detector),
        "initial_topology": str(initial_resolved.topology),
        "updated_topology": str(updated_resolved.topology),
        "modes_per_plane": modes_per_plane,
        "latent_dimension": 2 * modes_per_plane,
        "grid_points": int(grid_points),
        "kinetic_energy_prior": "uniform_on_documented_broad_support",
        "range_prior": "Geant4_UniversalFluctuation_first_passage_moment",
        "path_prior": "standard_normal_Fermi_Eyges_KL",
        "uses_event_truth": False,
        "uses_empirical_mcs_scale": False,
        "uses_wcsim_range_width": False,
        "uses_discrete_range_grid": False,
        "range_coordinate": "continuous_noncentered_float64_z_R",
        "energy_coordinate": "continuous_float64_K0_mev",
        "path_grid_role": "finite_element_quadrature_only",
        "output_length_quantization_mm": None,
        "charge_only": bool(charge_only),
        "timing_used": bool(not charge_only),
        "optical_model_build_count": int(evaluator.optical_model_build_count),
        "optical_model_build_wall_s": float(
            getattr(evaluator, "optical_model_build_wall_s", math.nan)
        ),
        "coherent_field_evaluation_count": int(
            evaluator.coherent_field_evaluation_count
        ),
        "invalid_model_evaluation_count": int(evaluator.invalid_evaluations),
        "posterior": posterior,
    }
    return CosmicJointEnergyRangeResult(
        initial_values=dict(local_values),
        updated_values=updated_values,
        updated_chart=updated_chart,
        initial_resolved=initial_resolved,
        updated_resolved=updated_resolved,
        smc=smc,
        coefficients_mean=np.ascontiguousarray(coefficient_mean),
        coefficients_covariance=np.ascontiguousarray(coefficient_covariance),
        guide_diagnostics=guide_diagnostics,
        map_sample_index=map_index,
        map_charge_nll=float(-smc.log_likelihood[map_index]),
        wall_s=float(time.perf_counter() - wall0),
        diagnostics=diagnostics,
    )


def run_cosmic_joint_energy_range_update(
    template_emitter,
    *,
    inference_method: str = "laplace_cubature",
    path_model: str | None = None,
    **kwargs,
) -> CosmicJointEnergyRangeResult:
    """Infer the continuous joint physical state with a selectable path model.

    ``laplace_cubature`` is the deterministic production engine.  ``smc``
    retains the exact annealed reference implementation for validation studies.
    Both engines use the same continuous K0, non-centred stopping range,
    full-rank Gaussian FE path, optical likelihood, and physical priors.

    The experimental ``mixed_mcs`` path model dispatches to the reference SMC
    integration with a Gaussian soft Wentzel core and explicit marked-Poisson
    hard scatters.  It is intentionally separate from the Gaussian production
    engine and currently fails closed for detector-exiting curved paths.
    """
    if path_model is None:
        # Keep direct library calls and the batch driver on the same explicit
        # selector without importing configuration at module initialization.
        from .Emitter import emitter_switch_summary_from_env

        selected = str(
            emitter_switch_summary_from_env()["cosmic_mcs_continuation"]
        )
        path_model = (
            "mixed_mcs"
            if selected == "joint_k0_range_mixed_mcs"
            else "gaussian_fe"
        )
    canonical_path = str(path_model).strip().lower().replace("-", "_")
    aliases = {
        "gaussian": "gaussian_fe",
        "joint_k0_range_gaussian_fe": "gaussian_fe",
        "mixed": "mixed_mcs",
        "soft_hard": "mixed_mcs",
        "joint_k0_range_mixed_mcs": "mixed_mcs",
    }
    canonical_path = aliases.get(canonical_path, canonical_path)
    if canonical_path == "mixed_mcs":
        from .cosmic_mixed_mcs import (
            run_cosmic_joint_mixed_energy_range_update,
        )

        return run_cosmic_joint_mixed_energy_range_update(
            template_emitter,
            inference_method="reference_smc",
            **kwargs,
        )
    if canonical_path != "gaussian_fe":
        raise ValueError("joint path_model must be gaussian_fe or mixed_mcs")
    overall_wall0 = time.perf_counter()
    caller_values, caller_chart = reanchor_values(
        kwargs["values"], kwargs["chart"]
    )
    global_preconditioner = None
    if bool(kwargs.get("joint_global_precondition_enabled", False)):
        # The joint K0/range/path posterior is evaluated on a fixed global line.
        # Select that line with the truth-blind coherent charge profile first;
        # otherwise a wrong straight-track basin is outside the conditional
        # path state and no amount of K0/z_R/path optimization can repair it.
        global_preconditioner = run_cosmic_coherent_profile_update(
            template_emitter,
            values=caller_values,
            chart=caller_chart,
            detector=kwargs["detector"],
            range_lookup=kwargs["range_lookup"],
            wcd=kwargs["wcd"],
            pmt_model=kwargs["pmt_model"],
            p_locations=kwargs["p_locations"],
            pmt_normals=kwargs["pmt_normals"],
            obs_pes=kwargs["obs_pes"],
            obs_ts=kwargs["obs_ts"],
            starts_at_boundary=bool(kwargs["starts_at_boundary"]),
            mpmt_types=kwargs.get("mpmt_types"),
            inset_mm=float(kwargs.get("inset_mm", 0.5)),
            range_limits=kwargs.get("range_limits"),
            t0_limits=kwargs.get("t0_limits"),
            modes_per_plane=int(kwargs.get("modes_per_plane", 12)),
            grid_points=int(kwargs.get("grid_points", 41)),
            latent_fd=float(kwargs.get("guide_latent_fd", 0.20)),
            latent_max_iterations=int(
                kwargs.get("joint_global_precondition_latent_iterations", 4)
            ),
            candidate_latent_max_iterations=int(
                kwargs.get(
                    "joint_global_precondition_candidate_latent_iterations",
                    2,
                )
            ),
            track_cycles=int(
                kwargs.get("joint_global_precondition_track_cycles", 2)
            ),
            transverse_step_mm=float(
                kwargs.get("joint_global_precondition_transverse_step_mm", 12.0)
            ),
            longitudinal_step_mm=float(
                kwargs.get("joint_global_precondition_longitudinal_step_mm", 12.0)
            ),
            direction_step=float(
                kwargs.get("joint_global_precondition_direction_step", 0.010)
            ),
            range_step_mm=float(
                kwargs.get("joint_global_precondition_range_step_mm", 30.0)
            ),
            latent_trust_max_component=float(
                kwargs.get("joint_global_precondition_latent_trust", 1.0)
            ),
            track_trust_max_scaled_component=float(
                kwargs.get("joint_global_precondition_track_trust", 1.0)
            ),
            sparse_neighbor_radius_mm=float(
                kwargs.get("sparse_neighbor_radius_mm", 100.0)
            ),
            fixed_initial_kinetic_energy_mev=kwargs.get(
                "joint_global_precondition_fixed_energy_mev"
            ),
            profile_selection_objective=str(
                kwargs.get(
                    "joint_global_precondition_selection_objective",
                    "laplace_surrogate",
                )
            ),
            initial_path_coefficients=kwargs.get("initial_path_coefficients"),
        )
        kwargs["values"] = global_preconditioner.updated_values
        kwargs["chart"] = global_preconditioner.updated_chart
        kwargs["initial_path_coefficients"] = np.asarray(
            global_preconditioner.coefficients_mean, dtype=np.float64
        )

    global_preconditioner_diagnostics = {
        "enabled": bool(global_preconditioner is not None),
        "role": (
            "truth_blind_charge_data_global_line_conditioning_before_joint_target"
            if global_preconditioner is not None
            else "disabled"
        ),
        "line_geometry_marginalized": False,
        "input_start_mm": [
            float(caller_values[name]) for name in ("x0", "y0", "z0")
        ],
        "input_direction": [float(value) for value in caller_chart.anchor],
        "input_range_mm": float(caller_values["length"]),
    }
    if global_preconditioner is not None:
        global_preconditioner_diagnostics.update({
            "updated_start_mm": [
                float(global_preconditioner.updated_values[name])
                for name in ("x0", "y0", "z0")
            ],
            "updated_direction": [
                float(value) for value in global_preconditioner.updated_chart.anchor
            ],
            "updated_range_mm": float(
                global_preconditioner.updated_values["length"]
            ),
            "selection_nll": float(global_preconditioner.selection_nll),
            "posterior_nll": float(global_preconditioner.posterior_nll),
            "accepted_global_steps": int(sum(
                bool(row.accepted) for row in global_preconditioner.iterations
            )),
            "wall_s": float(global_preconditioner.wall_s),
            "selection_objective": str(
                global_preconditioner.diagnostics.get(
                    "global_selection_objective", "unknown"
                )
            ),
        })
    method = str(inference_method).strip().lower().replace("-", "_")
    if method in {"smc", "annealed_smc", "reference_smc"}:
        result = _run_cosmic_joint_energy_range_smc_update(
            template_emitter, **kwargs
        )
        result.wall_s = float(time.perf_counter() - overall_wall0)
        result.diagnostics.update({
            "line_geometry_marginalized": False,
            "line_geometry_source": (
                "coherent_FE_charge_profile_preconditioner"
                if global_preconditioner is not None
                else "caller_supplied_straight_fit"
            ),
            "global_line_preconditioner": global_preconditioner_diagnostics,
        })
        return result
    if method not in {"laplace_cubature", "laplace_gh", "deterministic"}:
        raise ValueError(
            "joint inference_method must be laplace_cubature or reference_smc"
        )

    wall0 = overall_wall0
    values = kwargs["values"]
    chart = kwargs["chart"]
    detector = kwargs["detector"]
    range_lookup = kwargs["range_lookup"]
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
    starts_at_boundary = bool(kwargs["starts_at_boundary"])
    inset_mm = float(kwargs.get("inset_mm", 0.5))
    initial_resolved = resolve_range_clipped_track(
        detector,
        theta[:3],
        local_chart.anchor,
        theta[5],
        starts_at_boundary=starts_at_boundary,
        inset_mm=inset_mm,
        tolerance_mm=1.0e-6,
    )
    if initial_resolved is None:
        raise ValueError("initial cosmic track cannot be resolved")
    modes_per_plane = int(kwargs.get("modes_per_plane", 12))
    grid_points = int(kwargs.get("grid_points", 41))
    if modes_per_plane < 1:
        raise ValueError("modes_per_plane must be positive")
    evaluator = CosmicCoherentEvaluator(
        template_emitter,
        kwargs["wcd"],
        kwargs["pmt_model"],
        kwargs["p_locations"],
        kwargs["pmt_normals"],
        kwargs["obs_pes"],
        kwargs["obs_ts"],
        chart=local_chart,
        detector=detector,
        range_lookup=range_lookup,
        starts_at_boundary=starts_at_boundary,
        expected_exits_detector=bool(initial_resolved.exits_detector),
        mpmt_types=kwargs.get("mpmt_types"),
        inset_mm=inset_mm,
        range_limits=kwargs.get("range_limits"),
        t0_limits=kwargs.get("t0_limits"),
        n_modes=2 * modes_per_plane,
        n_grid=grid_points,
        aperture_radius_mm=float(
            getattr(template_emitter, "primary_endpoint_aperture_radius_mm", 45.0)
        ),
        sparse_receiver=True,
        sparse_neighbor_radius_mm=float(
            kwargs.get("sparse_neighbor_radius_mm", 100.0)
        ),
        charge_only=bool(kwargs.get("charge_only", True)),
    )
    straggling = StoppingRangeStraggling(
        particle=str(getattr(template_emitter, "particle_name", "muon"))
    )
    energy_low, energy_high = map(
        float, kwargs["kinetic_energy_bounds_mev"]
    )
    if energy_low <= straggling.threshold_mev:
        raise ValueError(
            "kinetic-energy support must lie above the Cherenkov threshold"
        )
    seed_energy = float(range_lookup.range_mm_to_energy(theta[5]))
    seed_energy = float(np.clip(seed_energy, energy_low, energy_high))
    seed_z = float(straggling.z_range(seed_energy, theta[5]))
    laplace = run_joint_laplace_cubature(
        evaluator,
        theta,
        straggling,
        kinetic_energy_bounds_mev=(energy_low, energy_high),
        initial_energy_mev=seed_energy,
        initial_z_range=seed_z,
        n_path_modes=2 * modes_per_plane,
        initial_path_coefficients=kwargs.get("initial_path_coefficients"),
        config=JointLaplaceCubatureConfig(
            latent_iterations=3,
        ),
        random_seed=int(kwargs.get("random_seed", 41873)),
        t0_bounds=kwargs.get("t0_limits"),
    )
    posterior = laplace.posterior.summary()
    posterior_range = float(posterior["realized_range_mm_mean"])
    posterior_energy = float(posterior["initial_kinetic_energy_mev_mean"])
    posterior_theta = theta.copy()
    posterior_theta[5] = posterior_range
    updated_resolved = evaluator.resolved_track(posterior_theta)
    if updated_resolved is None:
        raise RuntimeError(
            "posterior-mean realised range is outside the selected topology"
        )
    updated_chart = TangentDirectionChart.from_direction(
        np.asarray(updated_resolved.direction, dtype=np.float64)
    )
    updated_values = {
        "x0": float(updated_resolved.start[0]),
        "y0": float(updated_resolved.start[1]),
        "z0": float(updated_resolved.start[2]),
        "dir_u": 0.0,
        "dir_v": 0.0,
        "length": posterior_range,
        "initial_kinetic_energy_mev": posterior_energy,
        "z_range": float(posterior["z_range_mean"]),
        "t0": float(laplace.posterior_t0_mean_ns),
    }
    diagnostics = {
        **laplace.diagnostics,
        "line_geometry_conditioned": True,
        "line_geometry_marginalized": False,
        "line_geometry_source": (
            "coherent_FE_charge_profile_preconditioner"
            if global_preconditioner is not None
            else "caller_supplied_straight_fit"
        ),
        "global_line_preconditioner": global_preconditioner_diagnostics,
        "starts_at_boundary": starts_at_boundary,
        "exits_detector": bool(initial_resolved.exits_detector),
        "initial_topology": str(initial_resolved.topology),
        "updated_topology": str(updated_resolved.topology),
        "modes_per_plane": modes_per_plane,
        "latent_dimension": 2 * modes_per_plane,
        "grid_points": grid_points,
        "kinetic_energy_prior": "uniform_on_documented_broad_support",
        "range_prior": "Geant4_UniversalFluctuation_first_passage_moment",
        "path_prior": "standard_normal_Fermi_Eyges_KL",
        "range_coordinate": "continuous_noncentered_float64_z_R",
        "energy_coordinate": "continuous_float64_K0_mev",
        "path_grid_role": "finite_element_quadrature_only",
        "output_length_quantization_mm": None,
        "charge_only": bool(kwargs.get("charge_only", True)),
        "timing_used": bool(not kwargs.get("charge_only", True)),
        "optical_model_build_count": int(evaluator.optical_model_build_count),
        "optical_model_build_wall_s": float(
            getattr(evaluator, "optical_model_build_wall_s", math.nan)
        ),
        "coherent_field_evaluation_count": int(
            evaluator.coherent_field_evaluation_count
        ),
        "invalid_model_evaluation_count": int(evaluator.invalid_evaluations),
        "posterior": posterior,
    }
    guide_diagnostics = {
        "enabled": False,
        "target_altering": False,
        "seed_energy_mev": seed_energy,
        "seed_z_range": seed_z,
        "selection_policy": (
            "deterministic local exact-score solve and full-rank "
            "Laplace/Gauss-Hermite cubature"
        ),
        "reference_smc_available": True,
    }
    return CosmicJointEnergyRangeResult(
        initial_values=dict(local_values),
        updated_values=updated_values,
        updated_chart=updated_chart,
        initial_resolved=initial_resolved,
        updated_resolved=updated_resolved,
        smc=laplace.posterior,
        coefficients_mean=np.ascontiguousarray(laplace.coefficients_mean),
        coefficients_covariance=np.ascontiguousarray(
            laplace.coefficients_covariance
        ),
        guide_diagnostics=guide_diagnostics,
        map_sample_index=int(laplace.map_sample_index),
        map_charge_nll=float(laplace.map_charge_nll),
        wall_s=float(time.perf_counter() - wall0),
        diagnostics=diagnostics,
    )


def run_cosmic_coherent_joint_length_update(
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
    initial_path_coefficients: Sequence[float],
    mpmt_types=None,
    inset_mm: float = 0.5,
    range_limits: tuple[float, float] | None = None,
    t0_limits: tuple[float, float] | None = None,
    modes_per_plane: int = 12,
    grid_points: int = 41,
    response_fd_step: float = 0.20,
    latent_max_iterations: int = 2,
    final_latent_max_iterations: int | None = None,
    candidate_latent_max_iterations: int = 1,
    range_step_mm: float = 30.0,
    minimum_range_step_mm: float = 2.0,
    maximum_range_cycles: int = 8,
    transverse_step_mm: float = 12.0,
    minimum_transverse_step_mm: float = 1.0,
    longitudinal_step_mm: float = 12.0,
    minimum_longitudinal_step_mm: float = 1.0,
    direction_step: float = 0.010,
    minimum_direction_step: float = 5.0e-4,
    maximum_global_cycles: int = 4,
    allow_longitudinal: bool | None = None,
    allow_transverse: bool = True,
    allow_direction: bool = True,
    fixed_parameter_names: Sequence[str] = (),
    path_trust_max_component: float = 1.0,
    minimum_line_scale: float = 0.0625,
    gradient_tolerance: float = 2.0e-3,
    sparse_neighbor_radius_mm: float = 100.0,
    fixed_initial_kinetic_energy_mev: float | None = None,
    profile_t0: bool = True,
    t0_profile_coarse_step_ns: float = 0.25,
    t0_profile_refine_levels: int = 2,
    t0_profile_global_points: int = 9,
    t0_profile_seed_half_width_ns: float = 2.0,
    model_cache_size: int = 8,
    analytic_complete_candidate_response: bool = True,
    boundary_interface_model=None,
    boundary_interface_timing_policy: str = "mask_module",
) -> CosmicCoherentJointLengthResult:
    """Fit the coherent-MCS global line, range, path, and time jointly.

    This is deliberately a *joint* refinement, not a timing-only correction.
    The preceding charge solution fixes only the discrete topology and supplies
    a basin/path seed.  Deterministic physical-coordinate polls then update the
    two transverse line coordinates, the observable longitudinal coordinate,
    both direction tangents, and remaining range.  At every candidate the
    complete curved-path charge-plus-first-arrival likelihood profiles ``t0``;
    every accepted global move is followed by a posterior-decreasing FE-path
    update.  Thus timing can correct the global line and range without a
    straight-path timing polish or a second, inconsistent objective.

    The standard-normal FE prior and optional universal range-straggling prior
    are included in every comparison.  Nonprimary production components remain
    conditioned on the same coherent direct path.  No event truth enters this
    calculation.
    """
    wall0 = time.perf_counter()
    modes_per_plane = int(modes_per_plane)
    if modes_per_plane < 1:
        raise ValueError("modes_per_plane must be positive")
    dimension = 2 * modes_per_plane
    initial_coefficients = np.asarray(
        initial_path_coefficients, dtype=np.float64
    ).reshape(dimension)
    if np.any(~np.isfinite(initial_coefficients)):
        raise ValueError("initial coherent path coefficients must be finite")

    range_step_mm = float(range_step_mm)
    minimum_range_step_mm = float(minimum_range_step_mm)
    if not math.isfinite(range_step_mm) or range_step_mm <= 0.0:
        raise ValueError("range_step_mm must be positive and finite")
    if (
        not math.isfinite(minimum_range_step_mm)
        or minimum_range_step_mm <= 0.0
        or minimum_range_step_mm > range_step_mm
    ):
        raise ValueError(
            "minimum_range_step_mm must be positive and no larger than "
            "range_step_mm"
        )
    maximum_range_cycles = int(maximum_range_cycles)
    if maximum_range_cycles < 1:
        raise ValueError("maximum_range_cycles must be positive")
    maximum_global_cycles = int(maximum_global_cycles)
    if maximum_global_cycles < 0:
        raise ValueError("maximum_global_cycles must be nonnegative")
    coordinate_step_pairs = {
        "transverse": (
            float(transverse_step_mm),
            float(minimum_transverse_step_mm),
        ),
        "longitudinal": (
            float(longitudinal_step_mm),
            float(minimum_longitudinal_step_mm),
        ),
        "direction": (
            float(direction_step),
            float(minimum_direction_step),
        ),
    }
    for label, (initial_step, minimum_step) in coordinate_step_pairs.items():
        if not (
            math.isfinite(initial_step)
            and math.isfinite(minimum_step)
            and 0.0 < minimum_step <= initial_step
        ):
            raise ValueError(
                f"{label} steps must be finite, positive, and ordered"
            )
    fixed_names = {
        str(name).strip().lower() for name in fixed_parameter_names
    }
    vertex_is_fixed = bool(fixed_names.intersection({"x0", "y0", "z0"}))
    direction_is_fixed = bool(
        fixed_names.intersection({"cx", "cy", "cz", "dir_u", "dir_v"})
    )
    range_is_fixed = "length" in fixed_names or "full_range" in fixed_names
    if allow_longitudinal is None:
        allow_longitudinal = not bool(starts_at_boundary)
    allow_longitudinal = bool(allow_longitudinal) and not vertex_is_fixed
    allow_transverse = bool(allow_transverse) and not vertex_is_fixed
    allow_direction = bool(allow_direction) and not direction_is_fixed
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

    if range_limits is None:
        range_low = 1.0e-9
        range_high = float(range_lookup.overall_distances_mm[-1])
    else:
        range_low, range_high = map(float, range_limits)
        # The public cosmic optimizer uses a closed numerical lower bound of
        # zero, while a stopping range of exactly zero is outside the emitter's
        # physical domain.  Intersect that numerical box with R > 0 here; no
        # admissible track is removed and callers do not need a second set of
        # almost-identical limits solely for this continuation.
        range_low = max(range_low, 1.0e-9)
    if not (
        math.isfinite(range_low)
        and math.isfinite(range_high)
        and 0.0 < range_low <= range_high
    ):
        raise ValueError("range_limits must be positive, finite, and ordered")
    if not range_low <= float(theta[5]) <= range_high:
        raise ValueError("initial coherent range is outside range_limits")

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
        range_limits=(range_low, range_high),
        t0_limits=t0_limits,
        n_modes=dimension,
        n_grid=int(grid_points),
        aperture_radius_mm=float(
            getattr(
                template_emitter,
                "primary_endpoint_aperture_radius_mm",
                45.0,
            )
        ),
        sparse_receiver=True,
        sparse_neighbor_radius_mm=float(sparse_neighbor_radius_mm),
        fixed_initial_kinetic_energy_mev=fixed_initial_kinetic_energy_mev,
        model_cache_size=int(model_cache_size),
        charge_only=False,
        boundary_interface_model=boundary_interface_model,
        boundary_interface_timing_policy=boundary_interface_timing_policy,
    )

    straggling = None
    fixed_ke0 = None
    range_mean_mm = math.nan
    range_sigma_mm = math.nan
    if fixed_initial_kinetic_energy_mev is not None:
        fixed_ke0 = float(fixed_initial_kinetic_energy_mev)
        straggling = StoppingRangeStraggling(
            particle=str(getattr(template_emitter, "particle_name", "muon"))
        )
        if fixed_ke0 <= straggling.threshold_mev:
            raise ValueError(
                "fixed initial kinetic energy must exceed the Cherenkov threshold"
            )
        range_mean_mm = float(straggling.mean_range_mm(fixed_ke0))
        range_sigma_mm = float(straggling.sigma_mm(fixed_ke0))
        if not math.isfinite(range_sigma_mm) or range_sigma_mm <= 0.0:
            raise ValueError("stopping-range straggling sigma must be positive")

    def range_prior_nll(length_mm: float) -> float:
        if straggling is None:
            return 0.0
        z_range = (float(length_mm) - range_mean_mm) / range_sigma_mm
        return 0.5 * z_range * z_range

    objective_evaluations = 0
    latent_solve_count = 0
    latent_jacobian_evaluations = 0
    latent_jacobian_field_evaluations = 0
    latent_prediction_wall_s = 0.0
    complete_fisher_summary_count = 0
    invalid_candidate_count = 0
    candidate_failures: list[dict] = []
    screened_candidate_count = 0

    def evaluate_state(
        candidate_theta_values: Sequence[float],
        seed_coefficients: np.ndarray,
        *,
        latent_iterations: int,
        seed_t0_ns: float,
        fail_closed: bool = False,
        finalize_fisher_summary: bool = False,
        analytic_complete_response: bool = False,
    ):
        nonlocal objective_evaluations, latent_solve_count
        nonlocal latent_jacobian_evaluations
        nonlocal latent_jacobian_field_evaluations, latent_prediction_wall_s
        nonlocal complete_fisher_summary_count
        nonlocal invalid_candidate_count
        candidate_theta = np.asarray(
            candidate_theta_values, dtype=np.float64
        ).reshape(7).copy()
        candidate_theta[5] = float(
            np.clip(candidate_theta[5], range_low, range_high)
        )
        candidate_theta[6] = float(seed_t0_ns)
        model = evaluator.model(candidate_theta)
        if model is None:
            invalid_candidate_count += 1
            if fail_closed:
                detail = evaluator.last_model_build_error
                suffix = "" if not detail else f": {detail}"
                raise ValueError(
                    "initial coherent charge-time model is invalid" + suffix
                )
            return None
        try:
            latent = _solve_latent_with_central_response(
                model,
                initial_coefficients=np.asarray(
                    seed_coefficients, dtype=np.float64
                ),
                response_fd_step=float(response_fd_step),
                max_iterations=max(0, int(latent_iterations)),
                gradient_tolerance=float(gradient_tolerance),
                trust_max_component=float(path_trust_max_component),
                minimum_line_scale=float(minimum_line_scale),
                t0=float(seed_t0_ns),
                profile_t0=bool(profile_t0),
                t0_bounds=t0_limits,
                t0_profile_coarse_step_ns=float(
                    t0_profile_coarse_step_ns
                ),
                t0_profile_refine_levels=int(t0_profile_refine_levels),
                t0_profile_global_points=int(t0_profile_global_points),
                t0_profile_seed_half_width_ns=float(
                    t0_profile_seed_half_width_ns
                ),
                finalize_fisher_summary=bool(finalize_fisher_summary),
                analytic_complete_response=bool(analytic_complete_response),
            )
        except Exception as exc:
            invalid_candidate_count += 1
            candidate_failures.append({
                "theta": tuple(map(float, candidate_theta)),
                "error": repr(exc),
            })
            if fail_closed:
                raise
            return None
        latent_solve_count += 1
        objective_evaluations += int(latent.objective_evaluations)
        latent_jacobian_evaluations += int(latent.jacobian_evaluations)
        latent_jacobian_field_evaluations += int(
            latent.jacobian_field_evaluations
        )
        latent_prediction_wall_s += float(latent.prediction_wall_s)
        complete_fisher_summary_count += int(
            bool(latent.fisher_summary_complete)
        )
        prior_range = float(range_prior_nll(candidate_theta[5]))
        posterior = float(latent.posterior_nll + prior_range)
        if not math.isfinite(posterior):
            invalid_candidate_count += 1
            return None
        resolved = evaluator.resolved_track(candidate_theta)
        if resolved is None:
            invalid_candidate_count += 1
            return None
        candidate_theta[6] = float(latent.profiled_t0_ns)
        return {
            "theta": candidate_theta,
            "resolved": resolved,
            "latent": latent,
            "data_nll": float(latent.charge_nll),
            "path_prior_nll": 0.5
            * float(latent.coefficients @ latent.coefficients),
            "range_prior_nll": prior_range,
            "posterior_nll": posterior,
        }

    current = evaluate_state(
        theta,
        initial_coefficients,
        latent_iterations=int(latent_max_iterations),
        seed_t0_ns=float(theta[6]),
        fail_closed=True,
        analytic_complete_response=bool(
            analytic_complete_candidate_response
        ),
    )
    initial_profiled_posterior = float(current["posterior_nll"])
    history: list[CoherentJointLengthIteration] = []
    global_history: list[dict] = []
    coordinate_steps = {
        "internal_start_along_track": float(longitudinal_step_mm),
        "line_transverse_1": float(transverse_step_mm),
        "line_transverse_2": float(transverse_step_mm),
        "direction_tangent_1": float(direction_step),
        "direction_tangent_2": float(direction_step),
        "full_range": float(range_step_mm),
    }
    coordinate_minimums = {
        "internal_start_along_track": float(minimum_longitudinal_step_mm),
        "line_transverse_1": float(minimum_transverse_step_mm),
        "line_transverse_2": float(minimum_transverse_step_mm),
        "direction_tangent_1": float(minimum_direction_step),
        "direction_tangent_2": float(minimum_direction_step),
        "full_range": float(minimum_range_step_mm),
    }
    range_cycle = 0

    # This is deterministic block coordinate descent on the *same* posterior
    # throughout.  Screening candidates hold the current non-centred FE vector
    # fixed and profile only t0.  A selected global candidate is then given a
    # posterior-decreasing FE update.  Since re-optimizing a nuisance block can
    # only improve the selected state, no surrogate objective is needed for an
    # acceptance decision.
    total_cycles = max(maximum_global_cycles, maximum_range_cycles)
    for joint_cycle in range(total_cycles):
        direction = local_chart.direction(
            float(current["theta"][3]), float(current["theta"][4])
        )
        if direction is None:
            raise RuntimeError("joint coherent state lost its direction chart")
        specifications: list[tuple[str, np.ndarray, bool]] = []
        if joint_cycle < maximum_global_cycles:
            if allow_longitudinal:
                vector = np.zeros(7, dtype=np.float64)
                vector[:3] = direction
                specifications.append(
                    ("internal_start_along_track", vector, False)
                )
            if allow_transverse:
                for basis, label in (
                    (local_chart.e1, "line_transverse_1"),
                    (local_chart.e2, "line_transverse_2"),
                ):
                    vector = np.zeros(7, dtype=np.float64)
                    vector[:3] = basis
                    specifications.append((label, vector, False))
            if allow_direction:
                for index, label in (
                    (3, "direction_tangent_1"),
                    (4, "direction_tangent_2"),
                ):
                    vector = np.zeros(7, dtype=np.float64)
                    vector[index] = 1.0
                    specifications.append((label, vector, False))
        if (
            joint_cycle < maximum_range_cycles
            and not range_is_fixed
            and range_high > range_low + 1.0e-12
        ):
            vector = np.zeros(7, dtype=np.float64)
            vector[5] = 1.0
            specifications.append(("full_range", vector, True))

        attempted_coordinate = False
        for label, vector, is_range_coordinate in specifications:
            step = float(coordinate_steps[label])
            minimum_step = float(coordinate_minimums[label])
            if step < minimum_step - 1.0e-15:
                continue
            attempted_coordinate = True
            base = current
            base_theta = np.asarray(base["theta"], dtype=np.float64).copy()
            posterior_before = float(base["posterior_nll"])
            candidates: list[dict] = []
            by_sign: dict[float, dict] = {}
            for sign in (-1.0, 1.0):
                offset = float(sign * step)
                candidate_theta = base_theta + offset * vector
                if is_range_coordinate:
                    candidate_theta[5] = float(
                        np.clip(candidate_theta[5], range_low, range_high)
                    )
                    offset = float(candidate_theta[5] - base_theta[5])
                if float(np.linalg.norm(candidate_theta - base_theta)) <= 1.0e-12:
                    continue
                candidate = evaluate_state(
                    candidate_theta,
                    base["latent"].coefficients,
                    latent_iterations=0,
                    seed_t0_ns=float(base_theta[6]),
                    analytic_complete_response=bool(
                        analytic_complete_candidate_response
                    ),
                )
                if candidate is None:
                    continue
                candidate["_coordinate_offset"] = offset
                candidate["_proposal_kind"] = "side"
                candidates.append(candidate)
                by_sign[float(sign)] = candidate
                screened_candidate_count += 1

            # A local quadratic interpolation gives sub-step resolution without
            # imposing a grid on any continuous physical coordinate.  It is
            # merely another exact-posterior candidate, never an acceptance
            # surrogate.
            minus = by_sign.get(-1.0)
            plus = by_sign.get(1.0)
            if minus is not None and plus is not None:
                fminus = float(minus["posterior_nll"])
                fzero = posterior_before
                fplus = float(plus["posterior_nll"])
                curvature = fminus - 2.0 * fzero + fplus
                if math.isfinite(curvature) and curvature > 1.0e-10:
                    scaled_offset = 0.5 * (fminus - fplus) / curvature
                    offset = float(np.clip(scaled_offset, -1.0, 1.0) * step)
                    separated = all(
                        abs(offset - value)
                        > max(1.0e-10, 0.05 * minimum_step)
                        for value in (0.0, -step, step)
                    )
                    if separated:
                        candidate_theta = base_theta + offset * vector
                        if is_range_coordinate:
                            candidate_theta[5] = float(
                                np.clip(
                                    candidate_theta[5], range_low, range_high
                                )
                            )
                            offset = float(
                                candidate_theta[5] - base_theta[5]
                            )
                        quadratic = evaluate_state(
                            candidate_theta,
                            base["latent"].coefficients,
                            latent_iterations=0,
                            seed_t0_ns=float(base_theta[6]),
                            analytic_complete_response=bool(
                                analytic_complete_candidate_response
                            ),
                        )
                        if quadratic is not None:
                            quadratic["_coordinate_offset"] = offset
                            quadratic["_proposal_kind"] = "quadratic"
                            candidates.append(quadratic)
                            screened_candidate_count += 1

            best_screen = min(
                [base, *candidates],
                key=lambda row: float(row["posterior_nll"]),
            )
            selected = best_screen
            selected_offset = float(
                best_screen.get("_coordinate_offset", 0.0)
            )
            accepted = bool(
                best_screen is not base
                and float(best_screen["posterior_nll"])
                < posterior_before - 1.0e-8
            )
            if accepted:
                polished = evaluate_state(
                    best_screen["theta"],
                    best_screen["latent"].coefficients,
                    latent_iterations=int(candidate_latent_max_iterations),
                    seed_t0_ns=float(best_screen["theta"][6]),
                    analytic_complete_response=bool(
                        analytic_complete_candidate_response
                    ),
                )
                if (
                    polished is not None
                    and float(polished["posterior_nll"])
                    <= float(selected["posterior_nll"]) + 1.0e-8
                ):
                    selected = polished
                current = selected
                if abs(selected_offset) < 0.95 * step:
                    coordinate_steps[label] = 0.5 * step
            else:
                coordinate_steps[label] = 0.5 * step

            row = {
                "joint_cycle": int(joint_cycle),
                "coordinate": label,
                "step": step,
                "minimum_step": minimum_step,
                "theta_before": tuple(map(float, base_theta)),
                "candidate_offsets": tuple(
                    float(candidate.get("_coordinate_offset", math.nan))
                    for candidate in candidates
                ),
                "candidate_posteriors": tuple(
                    float(candidate["posterior_nll"])
                    for candidate in candidates
                ),
                "accepted": bool(accepted),
                "accepted_offset": selected_offset if accepted else 0.0,
                "posterior_before": posterior_before,
                "posterior_after": float(current["posterior_nll"]),
                "theta_after": tuple(map(float, current["theta"])),
            }
            global_history.append(row)
            if is_range_coordinate:
                history.append(
                    CoherentJointLengthIteration(
                        cycle=int(range_cycle),
                        step_mm=step,
                        length_before_mm=float(base_theta[5]),
                        posterior_before=posterior_before,
                        candidate_lengths_mm=tuple(
                            float(candidate["theta"][5])
                            for candidate in candidates
                        ),
                        candidate_posteriors=tuple(
                            float(candidate["posterior_nll"])
                            for candidate in candidates
                        ),
                        length_after_mm=float(current["theta"][5]),
                        posterior_after=float(current["posterior_nll"]),
                        accepted=bool(accepted),
                    )
                )
                range_cycle += 1
        if not attempted_coordinate:
            break

    active_line_labels = []
    if allow_longitudinal:
        active_line_labels.append("internal_start_along_track")
    if allow_transverse:
        active_line_labels.extend(("line_transverse_1", "line_transverse_2"))
    if allow_direction:
        active_line_labels.extend(("direction_tangent_1", "direction_tangent_2"))
    line_converged = maximum_global_cycles == 0 or all(
        coordinate_steps[label] < coordinate_minimums[label] - 1.0e-15
        for label in active_line_labels
    )
    range_converged = range_is_fixed or range_high <= range_low + 1.0e-12 or (
        coordinate_steps["full_range"]
        < coordinate_minimums["full_range"] - 1.0e-15
    )
    converged = bool(line_converged and range_converged)

    # Finish with the full latent budget at the selected range.  The exact
    # latent solver uses posterior-decreasing line searches, so this polish
    # cannot invalidate the range selection.
    final_latent_iterations = (
        int(latent_max_iterations)
        if final_latent_max_iterations is None
        else max(0, int(final_latent_max_iterations))
    )
    final_state = evaluate_state(
        current["theta"],
        current["latent"].coefficients,
        latent_iterations=final_latent_iterations,
        seed_t0_ns=float(current["theta"][6]),
        fail_closed=True,
        finalize_fisher_summary=True,
        analytic_complete_response=bool(
            analytic_complete_candidate_response
        ),
    )
    if float(final_state["posterior_nll"]) <= float(
        current["posterior_nll"]
    ) + 1.0e-8:
        current = final_state

    final_theta = np.asarray(current["theta"], dtype=np.float64)
    updated_resolved = current["resolved"]
    updated_chart = TangentDirectionChart.from_direction(
        updated_resolved.direction
    )
    updated_values = {
        "x0": float(updated_resolved.start[0]),
        "y0": float(updated_resolved.start[1]),
        "z0": float(updated_resolved.start[2]),
        "dir_u": 0.0,
        "dir_v": 0.0,
        "length": float(updated_resolved.full_range_mm),
        "t0": float(final_theta[6]),
    }
    if straggling is not None:
        updated_values.update({
            "initial_kinetic_energy_mev": float(fixed_ke0),
            "z_range": float(
                (float(updated_resolved.full_range_mm) - range_mean_mm)
                / range_sigma_mm
            ),
        })

    normalization_mode = str(
        getattr(template_emitter, "charge_normalization_mode", "event_mean")
    ).strip().lower().replace("-", "_")
    final_hardware_model = evaluator.model(final_theta)
    if boundary_interface_model is None:
        final_hardware_metadata = {
            "enabled": False,
            "profile_calls": 0,
            "profile": None,
            "status": "disabled",
        }
    elif final_hardware_model is None:
        final_hardware_metadata = {
            "enabled": True,
            "profile_calls": 0,
            "profile": None,
            "status": "final_model_unavailable",
        }
    else:
        final_hardware_metadata = final_hardware_model.boundary_hardware_metadata(
            current["latent"].coefficients
        )
    diagnostics = {
        "implementation": "coherent_FE_exact_joint_global_charge_time_profile_v2",
        "uses_event_truth": False,
        "timing_used_for_range_selection": True,
        "charge_used_for_range_selection": True,
        "timing_used_for_global_line_selection": bool(
            maximum_global_cycles > 0
            and (allow_longitudinal or allow_transverse or allow_direction)
        ),
        "charge_used_for_global_line_selection": bool(
            maximum_global_cycles > 0
            and (allow_longitudinal or allow_transverse or allow_direction)
        ),
        "charge_normalization_mode": normalization_mode,
        "absolute_light_used": normalization_mode == "global_scale",
        "t0_profiled_at_every_state_candidate": bool(profile_t0),
        "t0_profiled_at_every_range_candidate": bool(profile_t0),
        "global_line_conditioned_on_coherent_charge_solution": False,
        "global_line_seeded_by_coherent_charge_solution": True,
        "global_line_jointly_profiled": True,
        "line_geometry_marginalized": False,
        "topology_conditioned": str(initial_resolved.topology),
        "initial_start_mm": tuple(map(float, initial_resolved.start)),
        "final_start_mm": tuple(map(float, updated_resolved.start)),
        "start_change_mm": tuple(map(
            float, updated_resolved.start - initial_resolved.start
        )),
        "initial_direction": tuple(map(float, initial_resolved.direction)),
        "final_direction": tuple(map(float, updated_resolved.direction)),
        "initial_range_mm": float(theta[5]),
        "final_range_mm": float(updated_resolved.full_range_mm),
        "range_change_mm": float(updated_resolved.full_range_mm - theta[5]),
        "initial_profiled_posterior_nll": initial_profiled_posterior,
        "final_profiled_posterior_nll": float(current["posterior_nll"]),
        "profiled_posterior_improvement_nll": float(
            initial_profiled_posterior - current["posterior_nll"]
        ),
        "range_step_initial_mm": float(range_step_mm),
        "range_step_minimum_mm": float(minimum_range_step_mm),
        "range_cycles_maximum": int(maximum_range_cycles),
        "range_cycles_completed": int(len(history)),
        "global_cycles_maximum": int(maximum_global_cycles),
        "global_coordinate_attempts": int(sum(
            row["coordinate"] != "full_range" for row in global_history
        )),
        "global_coordinate_accepts": int(sum(
            row["coordinate"] != "full_range" and row["accepted"]
            for row in global_history
        )),
        "joint_coordinate_history": global_history,
        "coordinate_initial_steps": {
            "transverse_mm": float(transverse_step_mm),
            "longitudinal_mm": float(longitudinal_step_mm),
            "direction_tangent": float(direction_step),
            "range_mm": float(range_step_mm),
        },
        "coordinate_minimum_steps": {
            "transverse_mm": float(minimum_transverse_step_mm),
            "longitudinal_mm": float(minimum_longitudinal_step_mm),
            "direction_tangent": float(minimum_direction_step),
            "range_mm": float(minimum_range_step_mm),
        },
        "coordinate_final_steps": dict(coordinate_steps),
        "allow_longitudinal": bool(allow_longitudinal),
        "allow_transverse": bool(allow_transverse),
        "allow_direction": bool(allow_direction),
        "fixed_parameter_names": tuple(sorted(fixed_names)),
        "latent_solve_count": int(latent_solve_count),
        "latent_jacobian_evaluations": int(latent_jacobian_evaluations),
        "latent_jacobian_field_evaluations": int(
            latent_jacobian_field_evaluations
        ),
        "latent_prediction_wall_s": float(latent_prediction_wall_s),
        "complete_fisher_summary_count": int(complete_fisher_summary_count),
        "initial_latent_max_iterations": int(latent_max_iterations),
        "final_latent_max_iterations": int(final_latent_iterations),
        "objective_evaluations": int(objective_evaluations),
        "intermediate_fisher_summaries_omitted": True,
        "analytic_complete_candidate_response": bool(
            analytic_complete_candidate_response
        ),
        "screened_candidate_count": int(screened_candidate_count),
        "invalid_candidate_count": int(invalid_candidate_count),
        "candidate_failures": candidate_failures,
        "optical_model_build_count": int(evaluator.optical_model_build_count),
        "optical_model_build_wall_s": float(
            getattr(evaluator, "optical_model_build_wall_s", math.nan)
        ),
        "coherent_field_evaluation_count": int(
            evaluator.coherent_field_evaluation_count
        ),
        "path_prior": "standard_normal_noncentred_Fermi_Eyges",
        "range_prior": (
            "universal_stopping_range_fluctuation"
            if straggling is not None
            else "deterministic_inverse_range_energy_coupling"
        ),
        "nonprimary_timing_components": (
            "delta_source_resolved_curved; molecular_scatter_and_reflection_"
            "conditioned_on_coherent_path"
        ),
        "mpmt_hardware": final_hardware_metadata,
    }
    latent = current["latent"]
    return CosmicCoherentJointLengthResult(
        initial_values=dict(local_values),
        updated_values=updated_values,
        updated_chart=updated_chart,
        coefficients_mean=np.ascontiguousarray(latent.coefficients),
        coefficients_covariance=np.ascontiguousarray(latent.covariance),
        data_nll=float(current["data_nll"]),
        path_prior_nll=float(current["path_prior_nll"]),
        range_prior_nll=float(current["range_prior_nll"]),
        posterior_nll=float(current["posterior_nll"]),
        profiled_t0_ns=float(final_theta[6]),
        iterations=tuple(history),
        converged=bool(converged),
        wall_s=float(time.perf_counter() - wall0),
        diagnostics=diagnostics,
    )


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
    adaptive_track_max_cycles: int | None = None,
    adaptive_track_trigger_gain_nll: float = 5.0,
    adaptive_track_stop_gain_nll: float = 0.5,
    adaptive_track_stop_patience: int = 2,
    transverse_step_mm: float = 12.0,
    longitudinal_step_mm: float = 12.0,
    direction_step: float = 0.010,
    range_step_mm: float = 30.0,
    latent_trust_max_component: float = 1.0,
    track_trust_max_scaled_component: float = 1.0,
    sparse_neighbor_radius_mm: float = 100.0,
    fixed_initial_kinetic_energy_mev: float | None = None,
    profile_selection_objective: str = "laplace_surrogate",
    initial_path_coefficients: Sequence[float] | None = None,
    track_one_sided_half_step: bool = False,
    boundary_interface_model=None,
    boundary_interface_timing_policy: str = "mask_module",
) -> CosmicCoherentResult:
    """Profile a nonlinear coherent FE path and nearby cosmic track coordinates.

    When ``fixed_initial_kinetic_energy_mev`` is supplied, energy sets the
    Cherenkov yield and FE scattering scale while realized stopping range is a
    separate fitted variable with the parameter-free universal-fluctuation
    prior.  This is a controlled monoenergetic validation mode; the default
    retains the historical deterministic inverse-range energy coupling.
    """
    wall0 = time.perf_counter()
    selection_objective = str(profile_selection_objective).strip().lower()
    if selection_objective not in {"posterior", "laplace_surrogate"}:
        raise ValueError(
            "profile_selection_objective must be 'posterior' or "
            "'laplace_surrogate'"
        )
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
        fixed_initial_kinetic_energy_mev=fixed_initial_kinetic_energy_mev,
        boundary_interface_model=boundary_interface_model,
        boundary_interface_timing_policy=boundary_interface_timing_policy,
    )
    straggling = None
    fixed_ke0 = None
    range_mean_mm = math.nan
    range_sigma_mm = math.nan
    if fixed_initial_kinetic_energy_mev is not None:
        fixed_ke0 = float(fixed_initial_kinetic_energy_mev)
        straggling = StoppingRangeStraggling(
            particle=str(getattr(template_emitter, "particle_name", "muon"))
        )
        if fixed_ke0 <= straggling.threshold_mev:
            raise ValueError(
                "fixed initial kinetic energy must exceed the Cherenkov threshold"
            )
        range_mean_mm = float(straggling.mean_range_mm(fixed_ke0))
        range_sigma_mm = float(straggling.sigma_mm(fixed_ke0))
        if not math.isfinite(range_sigma_mm) or range_sigma_mm <= 0.0:
            raise ValueError("stopping-range straggling sigma must be positive")

    def range_prior_nll(candidate_theta: Sequence[float]) -> float:
        if straggling is None:
            return 0.0
        z_range = (float(candidate_theta[5]) - range_mean_mm) / range_sigma_mm
        return 0.5 * z_range * z_range

    def selection_nll(
        candidate_latent,
        candidate_theta: Sequence[float],
    ) -> float:
        core = (
            float(candidate_latent.posterior_nll)
            if selection_objective == "posterior"
            else float(candidate_latent.laplace_nll)
        )
        return core + range_prior_nll(candidate_theta)

    model = evaluator.model(theta)
    if model is None:
        build_error = evaluator.last_model_build_error
        if build_error:
            raise ValueError(
                "initial coherent cosmic model construction failed:\n"
                + str(build_error)
            )
        raise ValueError(
            "initial coherent cosmic state is outside the conditioned "
            "geometry/range/time domain"
        )
    latent = solve_latent_charge_map(
        model,
        initial_coefficients=initial_path_coefficients,
        fd_step=float(latent_fd),
        max_iterations=int(latent_max_iterations),
        trust_max_component=float(latent_trust_max_component),
    )
    history: list[ProfileIteration] = []
    converged = False
    global_descent_fallback_attempts = 0
    global_descent_fallback_accepts = 0
    base_track_cycles = max(0, int(track_cycles))
    maximum_track_cycles = (
        base_track_cycles
        if adaptive_track_max_cycles is None
        else max(base_track_cycles, int(adaptive_track_max_cycles))
    )
    adaptive_trigger_gain = float(adaptive_track_trigger_gain_nll)
    adaptive_stop_gain = float(adaptive_track_stop_gain_nll)
    adaptive_stop_patience = max(1, int(adaptive_track_stop_patience))
    if not math.isfinite(adaptive_trigger_gain) or adaptive_trigger_gain < 0.0:
        raise ValueError(
            "adaptive track trigger gain must be finite and nonnegative"
        )
    if not math.isfinite(adaptive_stop_gain) or adaptive_stop_gain < 0.0:
        raise ValueError(
            "adaptive track stop gain must be finite and nonnegative"
        )
    adaptive_extension_enabled = maximum_track_cycles > base_track_cycles
    adaptive_extension_triggered = False
    adaptive_small_gain_streak = 0
    termination_reason = "maximum_track_cycles"

    for cycle in range(maximum_track_cycles):
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

        prior_gradient = np.zeros(len(vectors), dtype=np.float64)
        prior_information = np.zeros(
            (len(vectors), len(vectors)), dtype=np.float64
        )
        if straggling is not None:
            range_coordinate = len(vectors) - 1
            z_range = (float(theta[5]) - range_mean_mm) / range_sigma_mm
            scaled_sigma = float(range_step_mm) / range_sigma_mm
            prior_gradient[range_coordinate] = z_range * scaled_sigma
            prior_information[range_coordinate, range_coordinate] = (
                scaled_sigma * scaled_sigma
            )

        proposal = profiled_charge_track_step_directions(
            evaluator,
            theta,
            latent,
            coordinate_vectors=np.asarray(vectors),
            coordinate_steps=np.asarray(steps),
            coordinate_labels=labels,
            coordinate_prior_gradient_scaled=prior_gradient,
            coordinate_prior_information_scaled=prior_information,
            trust_max_scaled_component=float(track_trust_max_scaled_component),
            one_sided_half_step=bool(track_one_sided_half_step),
        )
        if float(np.linalg.norm(proposal.delta_theta)) <= 1.0e-7:
            converged = True
            termination_reason = "negligible_proposed_track_step"
            break
        base_theta = theta.copy()
        base_laplace = selection_nll(latent, theta)
        accepted = False
        accepted_scale = 0.0
        best_theta = theta
        best_latent = latent
        line_search_scales = (
            1.0,
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
        )
        accepted_delta_theta = np.asarray(
            proposal.delta_theta, dtype=np.float64
        )
        for scale in line_search_scales:
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
            candidate_laplace = selection_nll(
                candidate_latent, candidate_theta
            )
            if candidate_laplace < base_laplace - 1.0e-8:
                accepted = True
                accepted_scale = float(scale)
                best_theta = candidate_theta
                best_latent = candidate_latent
                break
        if not accepted:
            # The Schur-complement Fisher/Gauss--Newton direction is a useful
            # physical preconditioner, but sharp optical support transitions
            # can make its local curvature model point across an active-set
            # boundary.  Retry with the negative *exact configured charge*
            # gradient in the same dimensionless physical coordinates.  This
            # mirrors the latent solver's recovery path.  The unchanged exact
            # posterior line search below remains authoritative, so this can
            # improve numerical convergence without changing the target.
            global_descent_fallback_attempts += 1
            gradient_scaled = np.asarray(
                proposal.gradient_scaled, dtype=np.float64
            )
            gradient_max_abs = (
                float(np.max(np.abs(gradient_scaled)))
                if gradient_scaled.size else 0.0
            )
            if math.isfinite(gradient_max_abs) and gradient_max_abs > 0.0:
                fallback_scaled = -gradient_scaled / gradient_max_abs
                fallback_trust = max(
                    float(track_trust_max_scaled_component), 0.0
                )
                if fallback_trust > 0.0:
                    fallback_scaled *= min(fallback_trust, 1.0)
                fallback_delta_theta = (
                    fallback_scaled * np.asarray(steps, dtype=np.float64)
                ) @ np.asarray(vectors, dtype=np.float64)
                for scale in line_search_scales:
                    candidate_theta = (
                        base_theta + float(scale) * fallback_delta_theta
                    )
                    candidate_model = evaluator.model(candidate_theta)
                    if candidate_model is None:
                        continue
                    candidate_latent = solve_latent_charge_map(
                        candidate_model,
                        initial_coefficients=latent.coefficients,
                        fd_step=float(latent_fd),
                        max_iterations=int(
                            candidate_latent_max_iterations
                        ),
                        trust_max_component=float(
                            latent_trust_max_component
                        ),
                    )
                    candidate_laplace = selection_nll(
                        candidate_latent, candidate_theta
                    )
                    if candidate_laplace < base_laplace - 1.0e-8:
                        accepted = True
                        accepted_scale = float(scale)
                        accepted_delta_theta = np.asarray(
                            fallback_delta_theta, dtype=np.float64
                        )
                        best_theta = candidate_theta
                        best_latent = candidate_latent
                        global_descent_fallback_accepts += 1
                        break
        history.append(
            ProfileIteration(
                cycle=int(cycle),
                theta_before=tuple(map(float, base_theta)),
                theta_after=tuple(map(float, best_theta)),
                laplace_before=float(base_laplace),
                laplace_after=selection_nll(best_latent, best_theta),
                accepted_scale=float(accepted_scale),
                accepted=bool(accepted),
                proposed_delta=tuple(map(float, accepted_delta_theta)),
            )
        )
        if not accepted:
            converged = True
            termination_reason = "no_accepted_exact_global_step"
            break
        theta = np.asarray(best_theta, dtype=np.float64)
        latent = best_latent
        if float(
            np.linalg.norm(accepted_scale * accepted_delta_theta)
        ) <= 1.0e-3:
            converged = True
            termination_reason = "negligible_accepted_track_step"
            break
        completed_cycles = int(cycle) + 1
        accepted_gain_nll = float(
            base_laplace - selection_nll(best_latent, best_theta)
        )
        if completed_cycles < base_track_cycles:
            continue
        if not adaptive_extension_enabled:
            termination_reason = "fixed_track_cycle_budget"
            continue
        if not adaptive_extension_triggered:
            if accepted_gain_nll > adaptive_trigger_gain:
                adaptive_extension_triggered = True
                adaptive_small_gain_streak = 0
                termination_reason = "adaptive_extension_in_progress"
                continue
            termination_reason = "adaptive_extension_trigger_not_met"
            break
        if accepted_gain_nll <= adaptive_stop_gain:
            adaptive_small_gain_streak += 1
        else:
            adaptive_small_gain_streak = 0
        if adaptive_small_gain_streak >= adaptive_stop_patience:
            converged = True
            termination_reason = "adaptive_exact_gain_plateau"
            break
        termination_reason = "adaptive_extension_in_progress"

    if not converged and len(history) >= maximum_track_cycles:
        termination_reason = (
            "adaptive_maximum_track_cycles"
            if adaptive_extension_enabled and adaptive_extension_triggered
            else "fixed_track_cycle_budget"
        )

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
    final_range_prior_nll = range_prior_nll(theta)
    if straggling is not None:
        updated_values.update({
            "initial_kinetic_energy_mev": float(fixed_ke0),
            "z_range": float(
                (float(theta[5]) - range_mean_mm) / range_sigma_mm
            ),
        })
    final_hardware_model = evaluator.model(theta)
    if boundary_interface_model is None:
        final_hardware_metadata = {
            "enabled": False,
            "profile_calls": 0,
            "profile": None,
            "status": "disabled",
        }
    elif final_hardware_model is None:
        final_hardware_metadata = {
            "enabled": True,
            "profile_calls": 0,
            "profile": None,
            "status": "final_model_unavailable",
        }
    else:
        final_hardware_metadata = final_hardware_model.boundary_hardware_metadata(
            latent.coefficients
        )
    diagnostics = {
        "implementation": "cosmic_clipped_nonlinear_fali_fisher_laplace_v2",
        "starts_at_boundary": bool(starts_at_boundary),
        "exits_detector": bool(initial_resolved.exits_detector),
        "initial_topology": str(initial_resolved.topology),
        "updated_topology": str(updated_resolved.topology),
        "modes_per_plane": int(modes_per_plane),
        "latent_dimension": int(2 * modes_per_plane),
        "grid_points": int(grid_points),
        "optical_model_build_count": int(evaluator.optical_model_build_count),
        "optical_model_build_wall_s": float(
            getattr(evaluator, "optical_model_build_wall_s", math.nan)
        ),
        "coherent_field_evaluation_count": int(evaluator.coherent_field_evaluation_count),
        "invalid_model_evaluation_count": int(evaluator.invalid_evaluations),
        "uses_event_truth": False,
        "uses_empirical_mcs_scale": False,
        "keeps_visible_length_and_full_range_distinct": True,
        "boundary_longitudinal_reference_removed": bool(starts_at_boundary),
        "energy_range_parameterization": (
            "fixed_energy_plus_universal_fluctuation_range"
            if straggling is not None
            else "deterministic_inverse_range_legacy"
        ),
        "fixed_initial_kinetic_energy_mev": (
            None if fixed_ke0 is None else float(fixed_ke0)
        ),
        "range_mean_mm": float(range_mean_mm),
        "range_sigma_mm": float(range_sigma_mm),
        "range_prior_nll": float(final_range_prior_nll),
        "global_selection_objective": str(selection_objective),
        "poisson_shape_information_role": (
            "proposal_preconditioner_only"
            if selection_objective == "posterior"
            else "proposal_preconditioner_and_laplace_surrogate"
        ),
        "global_descent_fallback_attempts": int(
            global_descent_fallback_attempts
        ),
        "global_descent_fallback_accepts": int(
            global_descent_fallback_accepts
        ),
        "track_cycle_controller": {
            "base_cycles": int(base_track_cycles),
            "maximum_cycles": int(maximum_track_cycles),
            "adaptive_enabled": bool(adaptive_extension_enabled),
            "adaptive_trigger_gain_nll": float(adaptive_trigger_gain),
            "adaptive_stop_gain_nll": float(adaptive_stop_gain),
            "adaptive_stop_patience": int(adaptive_stop_patience),
            "adaptive_extension_triggered": bool(
                adaptive_extension_triggered
            ),
            "final_small_gain_streak": int(adaptive_small_gain_streak),
            "cycles_completed": int(len(history)),
            "termination_reason": str(termination_reason),
        },
        "mpmt_hardware": final_hardware_metadata,
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
        posterior_nll=float(latent.posterior_nll + final_range_prior_nll),
        laplace_nll=float(latent.laplace_nll + final_range_prior_nll),
        selection_nll=float(selection_nll(latent, theta)),
        iterations=tuple(history),
        latent_iterations=tuple(latent.iterations),
        converged=bool(converged),
        wall_s=float(time.perf_counter() - wall0),
        diagnostics=diagnostics,
    )
