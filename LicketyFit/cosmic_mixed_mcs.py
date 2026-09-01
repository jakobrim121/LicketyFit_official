"""Experimental cosmic joint K0/range inference with mixed MCS paths.

This module is deliberately a thin integration layer.  The posterior engine,
mixed soft/hard path prior, coherent optical likelihood, and stopping-range
prior remain in their existing focused modules.  It conditions on the line
geometry selected by the cosmic fitter and currently supports only contained
curved paths; exiting curved tracks require a curved-path clipping model and
therefore fail closed instead of being evaluated with the wrong support.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .Emitter import shift_timing_prediction
from .cosmic_track_fit import resolve_range_clipped_track
from .mcs_joint_energy_range import EnergyPathGuide, GaussianPathGuide
from .mcs_joint_laplace import _profile_charge_time_t0
from .mcs_latent_profile import (
    solve_latent_charge_map,
    solve_latent_charge_map_derivative_free,
)
from .mcs_curved_path import MCSPhysicalDomainError
from .mcs_mixed_energy_range import (
    MixedEnergyRangeContext,
    MixedEnergyRangeSMCConfig,
    run_mixed_energy_range_smc,
    run_stratified_mixed_energy_range_smc,
)
from .mcs_mixed_path import MixedMCSLatent, MixedMCSPathPrior
from .stopping_straggling import StoppingRangeStraggling
from .track_parameterization import TangentDirectionChart, reanchor_values


class MixedMCSUnderpoweredConfiguration(ValueError):
    """The requested SMC population cannot support a convergence claim."""


class MixedMCSConvergenceError(RuntimeError):
    """A completed mixed-MCS run failed its within-run diversity checks."""


class _FrozenExplicitPathTimingLikelihood:
    """Profile ``t0`` without repeating the curved optical field.

    The optical trajectory is the expensive part of a mixed-state likelihood
    evaluation.  The additive event time is a global nuisance coordinate, and
    evaluating several candidate offsets changes only the inexpensive PMT
    first-arrival likelihood.  Freezing the explicit-path prediction here
    gives :func:`_profile_charge_time_t0` the same cache semantics as the
    coefficient-based coherent model.
    """

    def __init__(self, model, path, obs_pes, obs_ts):
        prediction = model.prediction_from_path(path)
        self.exp_pes = np.asarray(prediction[0], dtype=np.float64)
        self.timing_pes = np.asarray(prediction[1], dtype=np.float64)
        self.timing = prediction[2]
        self.pmt_model = model.pmt_model
        self.obs_pes = np.asarray(obs_pes, dtype=np.float64)
        self.obs_ts = np.asarray(obs_ts, dtype=np.float64)
        charge_interface = getattr(
            self.pmt_model, "get_neg_log_likelihood_npe", None
        )
        self.charge_nll = (
            None
            if charge_interface is None
            else float(charge_interface(self.exp_pes, self.obs_pes))
        )

    def data_nll(self, _path, *, t0=None):
        offset = float(0.0 if t0 is None else t0)
        timing_interface = getattr(
            self.pmt_model, "get_neg_log_likelihood_t", None
        )
        if self.charge_nll is not None and timing_interface is not None:
            return float(
                self.charge_nll
                + timing_interface(
                    self.exp_pes,
                    self.obs_pes,
                    self.timing,
                    self.obs_ts,
                    timing_pes=self.timing_pes,
                    model_time_shift_ns=offset,
                )
            )
        timing = (
            self.timing
            if offset == 0.0
            else shift_timing_prediction(self.timing, offset)
        )
        return float(
            self.pmt_model.get_neg_log_likelihood_npe_t(
                self.exp_pes,
                self.obs_pes,
                timing,
                self.obs_ts,
                timing_pes=self.timing_pes,
            )
        )

    def data_nll_many_t0(self, _path, t0_values):
        offsets = np.asarray(t0_values, dtype=np.float64).reshape(-1)
        timing_interface = getattr(
            self.pmt_model,
            "get_neg_log_likelihood_t_many_t0",
            None,
        )
        if self.charge_nll is not None and timing_interface is not None:
            return np.asarray(
                timing_interface(
                    self.exp_pes,
                    self.obs_pes,
                    self.timing,
                    self.obs_ts,
                    offsets,
                    timing_pes=self.timing_pes,
                ),
                dtype=np.float64,
            ) + float(self.charge_nll)
        interface = getattr(
            self.pmt_model,
            "get_neg_log_likelihood_npe_t_many_t0",
            None,
        )
        if interface is not None:
            return np.asarray(
                interface(
                    self.exp_pes,
                    self.obs_pes,
                    self.timing,
                    self.obs_ts,
                    offsets,
                    timing_pes=self.timing_pes,
                ),
                dtype=np.float64,
            )
        return np.asarray(
            [self.data_nll(_path, t0=float(value)) for value in offsets],
            dtype=np.float64,
        )


def _profile_explicit_path_t0(
    model,
    path,
    *,
    obs_pes,
    obs_ts,
    seed_t0,
    bounds,
    coarse_step_ns,
    refine_levels,
    global_points,
    seed_half_width_ns,
):
    frozen = _FrozenExplicitPathTimingLikelihood(model, path, obs_pes, obs_ts)
    return _profile_charge_time_t0(
        frozen,
        path,
        seed_t0=float(seed_t0),
        bounds=bounds,
        coarse_step_ns=float(coarse_step_ns),
        refine_levels=int(refine_levels),
        global_points=int(global_points),
        seed_half_width_ns=float(seed_half_width_ns),
    )


def _contained_curved_path(detector, path) -> bool:
    """Validate every node and represented segment in the detector water."""
    try:
        positions = np.asarray(path["position"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return False
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or positions.shape[0] == 0
        or np.any(~np.isfinite(positions))
    ):
        return False
    contains_many = getattr(detector, "contains_many", None)
    if contains_many is not None:
        nodes_valid = bool(contains_many(positions, tolerance_mm=1.0e-6))
    else:
        nodes_valid = bool(all(
            detector.contains(point, tolerance_mm=1.0e-6)
            for point in positions
        ))
    if not nodes_valid:
        return False
    # WCTE active water is not globally convex once mPMT dome/cap exclusions
    # are included.  Its exact segment helper catches a segment crossing such
    # an exclusion even when both stored endpoints are in water.
    segment_contained = getattr(detector, "segment_contained", None)
    if segment_contained is None:
        return True
    for start, stop in zip(positions[:-1], positions[1:], strict=True):
        delta = stop - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-12:
            continue
        if not bool(
            segment_contained(
                start,
                delta / length,
                length,
                tolerance_mm=1.0e-6,
            )
        ):
            return False
    return True


def run_cosmic_joint_mixed_energy_range_update(
    template_emitter,
    **kwargs,
):
    """Run the exact mixed soft/hard SMC posterior on a fitted cosmic line.

    This is an experimental reference engine, not a production fast path.  It
    uses a normalized defensive proposal mixture, while the beta=1 target is
    the uniform K0 prior, non-centred range prior, Gaussian soft Wentzel prior,
    marked-Poisson hard Wentzel prior, and configured charge or charge-plus-time
    likelihood.
    """
    # Imported lazily to avoid a module cycle: the public dispatcher lives in
    # cosmic_coherent_mcs and imports this module only for the mixed selector.
    from .cosmic_coherent_mcs import (
        CosmicCoherentEvaluator,
        CosmicJointEnergyRangeResult,
        run_cosmic_coherent_profile_update,
    )

    wall0 = time.perf_counter()
    values = kwargs["values"]
    chart = kwargs["chart"]
    detector = kwargs["detector"]
    range_lookup = kwargs["range_lookup"]
    caller_values, caller_chart = reanchor_values(values, chart)
    global_preconditioner = None
    if bool(kwargs.get("mixed_global_precondition_enabled", False)):
        # The expensive mixed posterior below is conditional on one global
        # straight-line chart.  A wrong straight-track basin cannot be repaired
        # by soft/hard deflections expressed around that frozen chart.  Use the
        # coherent FE profile as a truth-blind, charge-data preconditioner for
        # the line and range, then evaluate the unchanged mixed target on that
        # selected slice.  This is deliberately labelled conditioning rather
        # than global marginalization in the returned diagnostics.
        global_preconditioner = run_cosmic_coherent_profile_update(
            template_emitter,
            values=caller_values,
            chart=caller_chart,
            detector=detector,
            range_lookup=range_lookup,
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
                kwargs.get("mixed_global_precondition_latent_iterations", 4)
            ),
            candidate_latent_max_iterations=int(
                kwargs.get(
                    "mixed_global_precondition_candidate_latent_iterations",
                    2,
                )
            ),
            track_cycles=int(
                kwargs.get("mixed_global_precondition_track_cycles", 2)
            ),
            transverse_step_mm=float(
                kwargs.get("mixed_global_precondition_transverse_step_mm", 12.0)
            ),
            longitudinal_step_mm=float(
                kwargs.get("mixed_global_precondition_longitudinal_step_mm", 12.0)
            ),
            direction_step=float(
                kwargs.get("mixed_global_precondition_direction_step", 0.010)
            ),
            range_step_mm=float(
                kwargs.get("mixed_global_precondition_range_step_mm", 30.0)
            ),
            latent_trust_max_component=float(
                kwargs.get("mixed_global_precondition_latent_trust", 1.0)
            ),
            track_trust_max_scaled_component=float(
                kwargs.get("mixed_global_precondition_track_trust", 1.0)
            ),
            sparse_neighbor_radius_mm=float(
                kwargs.get("sparse_neighbor_radius_mm", 100.0)
            ),
            fixed_initial_kinetic_energy_mev=kwargs.get(
                "mixed_global_precondition_fixed_energy_mev"
            ),
            profile_selection_objective=str(
                kwargs.get(
                    "mixed_global_precondition_selection_objective",
                    "laplace_surrogate",
                )
            ),
            initial_path_coefficients=kwargs.get("initial_path_coefficients"),
        )
        values = global_preconditioner.updated_values
        chart = global_preconditioner.updated_chart
        kwargs["initial_path_coefficients"] = np.asarray(
            global_preconditioner.coefficients_mean, dtype=np.float64
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
    if bool(initial_resolved.exits_detector):
        raise NotImplementedError(
            "mixed-MCS cosmic inference currently supports only contained "
            "curved paths; an exiting path needs curved-trajectory clipping"
        )

    modes_per_plane = int(kwargs.get("modes_per_plane", 12))
    grid_points = int(kwargs.get("grid_points", 41))
    if modes_per_plane < 1:
        raise ValueError("modes_per_plane must be positive")
    if grid_points < 17:
        raise ValueError("grid_points must be at least 17")
    dimension = 2 * modes_per_plane
    particles = int(kwargs.get("particles", 32))
    posterior_sweeps = int(kwargs.get("posterior_rejuvenation_steps", 12))
    inference_engine = str(
        kwargs.get("mixed_inference_engine", "smc")
    ).strip().lower().replace("-", "_")
    if inference_engine not in {"smc", "importance"}:
        raise ValueError(
            "mixed_inference_engine must be 'smc' or 'importance'"
        )
    allow_underpowered = bool(
        kwargs.get("mixed_allow_underpowered_diagnostic", False)
    )
    minimum_particles = max(16, dimension)
    minimum_posterior_sweeps = 0 if inference_engine == "importance" else 4
    configuration_issues = []
    if particles < minimum_particles:
        configuration_issues.append(
            f"particles={particles} < {minimum_particles} for dimension={dimension}"
        )
    if posterior_sweeps < minimum_posterior_sweeps:
        configuration_issues.append(
            "posterior_rejuvenation_steps="
            f"{posterior_sweeps} < {minimum_posterior_sweeps}"
        )
    if configuration_issues and not allow_underpowered:
        raise MixedMCSUnderpoweredConfiguration(
            "underpowered mixed-MCS reference run: "
            + "; ".join(configuration_issues)
            + ". Increase the SMC controls or set "
            "COSMIC_MIXED_ALLOW_UNDERPOWERED_DIAGNOSTIC=1 only for "
            "non-inferential diagnostics."
        )
    sparse_receiver = bool(kwargs.get("mixed_sparse_receiver", False))
    charge_only = bool(kwargs.get("charge_only", True))
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
        expected_exits_detector=False,
        mpmt_types=kwargs.get("mpmt_types"),
        inset_mm=inset_mm,
        range_limits=kwargs.get("range_limits"),
        t0_limits=kwargs.get("t0_limits"),
        n_modes=2 * modes_per_plane,
        n_grid=grid_points,
        aperture_radius_mm=float(
            getattr(template_emitter, "primary_endpoint_aperture_radius_mm", 45.0)
        ),
        sparse_receiver=sparse_receiver,
        sparse_neighbor_radius_mm=float(
            kwargs.get("sparse_neighbor_radius_mm", 100.0)
        ),
        model_cache_size=int(kwargs.get("mixed_model_cache_size", 16)),
        charge_only=charge_only,
    )
    straggling = StoppingRangeStraggling(
        particle=str(getattr(template_emitter, "particle_name", "muon"))
    )
    requested_t0_treatment = kwargs.get("mixed_t0_treatment")
    if requested_t0_treatment is None:
        if "mixed_profile_t0" in kwargs:
            requested_t0_treatment = (
                "profile" if bool(kwargs["mixed_profile_t0"]) else "fixed"
            )
        else:
            requested_t0_treatment = "fixed" if charge_only else "sample"
    t0_treatment = str(requested_t0_treatment).strip().lower()
    if t0_treatment not in {"fixed", "profile", "sample"}:
        raise ValueError(
            "mixed_t0_treatment must be 'fixed', 'profile', or 'sample'"
        )
    if charge_only:
        t0_treatment = "fixed"
    profile_t0 = bool(t0_treatment == "profile")
    sample_t0 = bool(t0_treatment == "sample")
    t0_bounds = kwargs.get("t0_limits")
    if sample_t0 and t0_bounds is None:
        raise ValueError("sampled mixed event time requires finite t0_limits")
    t0_profile_coarse_step_ns = float(
        kwargs.get("mixed_t0_profile_coarse_step_ns", 0.25)
    )
    t0_profile_refine_levels = int(
        kwargs.get("mixed_t0_profile_refine_levels", 2)
    )
    t0_profile_global_points = int(
        kwargs.get("mixed_t0_profile_global_points", 9)
    )
    t0_profile_seed_half_width_ns = float(
        kwargs.get("mixed_t0_profile_seed_half_width_ns", 2.0)
    )
    if t0_profile_coarse_step_ns <= 0.0:
        raise ValueError("mixed t0 profile coarse step must be positive")
    if t0_profile_refine_levels < 0:
        raise ValueError("mixed t0 profile refine levels must be non-negative")
    if t0_profile_global_points < 1:
        raise ValueError("mixed t0 profile global points must be positive")
    if t0_profile_seed_half_width_ns <= 0.0:
        raise ValueError("mixed t0 profile seed half-width must be positive")
    guide_profile_t0 = bool(
        kwargs.get("mixed_guide_profile_t0", not charge_only)
    ) and not charge_only
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
    initial_path = kwargs.get("initial_path_coefficients")
    initial_path_covariance = kwargs.get("initial_path_covariance")
    best_charge_guide_posterior = [math.inf]

    def fit_soft_path_guide(
        kinetic_energy_mev,
        z_range,
        *,
        initial_coefficients,
        max_iterations,
    ):
        """Build one data-informed proposal at a fixed physical energy/range.

        A charge-shape MAP supplies a stable starting path.  In charge-plus-
        time mode it is then polished against the complete configured timing
        likelihood with the event-time nuisance profiled exactly as in the SMC
        target.  The guide remains only a normalized proposal: its density is
        included in every bridge and Metropolis ratio.  Separate energy guides
        are needed because standardized FE coordinates map to different
        physical bends as the scattering power changes.
        """
        guide_energy = float(kinetic_energy_mev)
        guide_z = float(z_range)
        guide_theta = theta.copy()
        guide_theta[5] = float(
            straggling.realized_range_mm(guide_energy, guide_z)
        )
        guide_model = evaluator.model(
            guide_theta, initial_kinetic_energy_mev=guide_energy
        )
        if guide_model is None:
            raise RuntimeError("invalid optical model at mixed guide anchor")
        guide_prior = MixedMCSPathPrior(
            guide_model.path_emitter,
            modes_per_plane=modes_per_plane,
            grid_points=grid_points,
            transport_grid_points=int(
                kwargs.get("mixed_transport_grid_points", 161)
            ),
        )

        class _SoftOnlyGuideModel:
            n_modes = dimension
            obs_pes = np.asarray(kwargs["obs_pes"], dtype=np.float64)
            pmt_model = guide_model.pmt_model

            def _path(self, coefficients):
                latent = MixedMCSLatent(coefficients, ())
                path = guide_prior.build_path(latent)
                if not _contained_curved_path(detector, path):
                    raise MCSPhysicalDomainError(
                        "soft guide path leaves detector"
                    )
                return path

            def charge_prediction(self, coefficients):
                path = self._path(coefficients)
                if charge_only:
                    return guide_model.charge_prediction_from_path(path)
                return guide_model.prediction_from_path(path)[0]

            def charge_data_nll(self, coefficients):
                prediction = np.maximum(
                    np.asarray(
                        self.charge_prediction(coefficients),
                        dtype=np.float64,
                    ),
                    1.0e-300,
                )
                return float(
                    self.pmt_model.get_neg_log_likelihood_npe(
                        prediction, self.obs_pes
                    )
                )

        class _TimedProfileGuideModel(_SoftOnlyGuideModel):
            def charge_data_nll(self, coefficients):
                path = self._path(coefficients)
                value, _profiled_t0 = _profile_explicit_path_t0(
                    guide_model,
                    path,
                    obs_pes=kwargs["obs_pes"],
                    obs_ts=kwargs["obs_ts"],
                    seed_t0=float(guide_theta[6]),
                    bounds=t0_bounds,
                    coarse_step_ns=t0_profile_coarse_step_ns,
                    refine_levels=t0_profile_refine_levels,
                    global_points=t0_profile_global_points,
                    seed_half_width_ns=t0_profile_seed_half_width_ns,
                )
                return float(value)

        guide_fit = solve_latent_charge_map(
            _SoftOnlyGuideModel(),
            initial_coefficients=np.asarray(
                initial_coefficients, dtype=np.float64
            ).reshape(dimension),
            fd_step=float(kwargs.get("guide_latent_fd", 0.20)),
            max_iterations=int(max_iterations),
            trust_max_component=1.0,
        )
        # The exact derivative-free timing polish intentionally reports no
        # Hessian/covariance. Preserve the finite Fisher proposal covariance
        # from its charge-shape seed instead of silently replacing a polished
        # guide by a broad identity proposal. This covariance is used only for
        # normalized importance proposals, never as a Laplace evidence term.
        proposal_covariance = np.asarray(
            guide_fit.covariance, dtype=np.float64
        ).copy()
        polish_diagnostics = {
            "attempted": False,
            "accepted": False,
        }
        charge_seed_posterior = float(guide_fit.posterior_nll)
        competitive_window_nll = float(
            kwargs.get("mixed_timed_guide_competitive_window_nll", 30.0)
        )
        timed_guide_competitive = bool(
            charge_seed_posterior
            <= best_charge_guide_posterior[0] + competitive_window_nll
        )
        best_charge_guide_posterior[0] = min(
            best_charge_guide_posterior[0], charge_seed_posterior
        )
        polish_all_timed_anchors = bool(
            kwargs.get("mixed_timed_guide_polish_all_anchors", True)
        )
        polish_diagnostics.update({
            "charge_seed_posterior_nll": charge_seed_posterior,
            "timed_guide_competitive": timed_guide_competitive,
            "timed_guide_competitive_window_nll": competitive_window_nll,
            "timed_guide_polish_all_anchors": polish_all_timed_anchors,
        })
        polish_model = (
            _TimedProfileGuideModel()
            if guide_profile_t0
            else _SoftOnlyGuideModel()
        )
        needs_polish = bool(
            guide_profile_t0
            and (polish_all_timed_anchors or timed_guide_competitive)
        ) or (
            not guide_profile_t0
            and (
                not bool(guide_fit.converged)
                or not math.isfinite(float(guide_fit.final_gradient_max_abs))
                or float(guide_fit.final_gradient_max_abs) > 0.10
            )
        )
        if bool(kwargs.get("mixed_guide_derivative_free_polish", True)) and (
            needs_polish
        ):
            polish_diagnostics["attempted"] = True
            try:
                polish_start_nll = float(
                    polish_model.charge_data_nll(guide_fit.coefficients)
                    + 0.5
                    * float(guide_fit.coefficients @ guide_fit.coefficients)
                )
                polished = solve_latent_charge_map_derivative_free(
                    polish_model,
                    initial_coefficients=guide_fit.coefficients,
                    max_evaluations=int(
                        kwargs.get(
                            "mixed_guide_derivative_free_max_evaluations",
                            120 if guide_profile_t0 else 300,
                        )
                    ),
                    initial_trust_radius=float(
                        kwargs.get(
                            "mixed_guide_derivative_free_initial_radius",
                            0.50,
                        )
                    ),
                    final_trust_radius=float(
                        kwargs.get(
                            "mixed_guide_derivative_free_final_radius",
                            0.01,
                        )
                    ),
                    poll_radii=(0.02,),
                    poll_tolerance=1.0e-3,
                )
                polish_diagnostics.update({
                    "optimizer_success": bool(polished.optimizer_success),
                    "optimizer_message": str(polished.optimizer_message),
                    "objective_evaluations": int(
                        polished.objective_evaluations
                    ),
                    "posterior_nll": float(polished.posterior_nll),
                    "improvement_nll": float(
                        polish_start_nll - polished.posterior_nll
                    ),
                    "target": (
                        "profiled_charge_time_posterior"
                        if guide_profile_t0
                        else "charge_posterior"
                    ),
                })
                if float(polished.posterior_nll) <= polish_start_nll:
                    guide_fit = polished
                    polish_diagnostics["accepted"] = True
            except Exception as exc:
                polish_diagnostics["failure"] = (
                    f"{type(exc).__name__}: {exc}"
                )
        guide_fit._mixed_guide_polish_diagnostics = polish_diagnostics
        guide_fit._mixed_guide_proposal_covariance = proposal_covariance
        return guide_fit

    def guide_fit_diagnostics(guide_fit):
        proposal_covariance = np.asarray(
            getattr(
                guide_fit,
                "_mixed_guide_proposal_covariance",
                guide_fit.covariance,
            ),
            dtype=np.float64,
        )
        covariance_eigenvalues = (
            np.linalg.eigvalsh(proposal_covariance)
            if np.all(np.isfinite(proposal_covariance))
            else np.asarray((math.nan,), dtype=np.float64)
        )
        return {
            "succeeded": True,
            "converged": bool(guide_fit.converged),
            "termination_reason": str(guide_fit.termination_reason),
            "iterations": int(len(guide_fit.iterations)),
            "final_gradient_max_abs": float(
                guide_fit.final_gradient_max_abs
            ),
            "posterior_nll": float(guide_fit.posterior_nll),
            "coefficient_norm": float(
                np.linalg.norm(guide_fit.coefficients)
            ),
            "solver_method": str(guide_fit.solver_method),
            "objective_evaluations": int(guide_fit.objective_evaluations),
            "proposal_covariance_source": (
                "charge_seed_fisher_after_exact_timing_polish"
                if bool(getattr(
                    guide_fit,
                    "_mixed_guide_polish_diagnostics",
                    {},
                ).get("accepted", False))
                else "final_guide_fit"
            ),
            "proposal_covariance_minimum_eigenvalue": float(
                np.min(covariance_eigenvalues)
            ),
            "proposal_covariance_maximum_eigenvalue": float(
                np.max(covariance_eigenvalues)
            ),
            "derivative_free_polish": dict(
                getattr(
                    guide_fit,
                    "_mixed_guide_polish_diagnostics",
                    {"attempted": False, "accepted": False},
                )
            ),
        }

    guide_construction = {
        "attempted": False,
        "succeeded": False,
        "method": (
            "charge_seed_then_profiled_charge_time_path_map"
            if guide_profile_t0
            else "charge_shape_path_map"
        ),
    }
    if initial_path is None and bool(
        kwargs.get("mixed_build_soft_path_guide", True)
    ):
        guide_construction["attempted"] = True
        try:
            guide_fit = fit_soft_path_guide(
                seed_energy,
                seed_z,
                initial_coefficients=np.zeros(dimension, dtype=np.float64),
                max_iterations=int(
                    kwargs.get(
                        "mixed_guide_latent_iterations",
                        min(int(kwargs.get("guide_latent_iterations", 80)), 20),
                    )
                ),
            )
            initial_path = np.asarray(
                guide_fit.coefficients, dtype=np.float64
            )
            candidate_covariance = np.asarray(
                getattr(
                    guide_fit,
                    "_mixed_guide_proposal_covariance",
                    guide_fit.covariance,
                ),
                dtype=np.float64,
            )
            initial_path_covariance = (
                candidate_covariance
                if np.all(np.isfinite(candidate_covariance))
                else None
            )
            guide_construction.update(guide_fit_diagnostics(guide_fit))
        except Exception as exc:
            guide_construction["failure"] = f"{type(exc).__name__}: {exc}"
    guide_mean = (
        np.zeros(dimension, dtype=np.float64)
        if initial_path is None
        else np.asarray(initial_path, dtype=np.float64).reshape(dimension)
    )
    if np.any(~np.isfinite(guide_mean)):
        raise ValueError("initial path coefficients must be finite")
    guide_inflation = float(
        kwargs.get(
            "mixed_guide_covariance_inflation",
            min(float(kwargs.get("guide_covariance_inflation", 4.0)), 2.0),
        )
    )
    if not math.isfinite(guide_inflation) or guide_inflation <= 0.0:
        raise ValueError("guide covariance inflation must be positive")
    energy_guide_sd = float(kwargs.get("guide_energy_proposal_sd_mev", 30.0))
    if not math.isfinite(energy_guide_sd) or energy_guide_sd <= 0.0:
        raise ValueError("guide energy proposal width must be positive")
    if initial_path_covariance is None:
        guide_covariance = guide_inflation * np.eye(
            dimension, dtype=np.float64
        )
    else:
        guide_covariance = guide_inflation * np.asarray(
            initial_path_covariance, dtype=np.float64
        ).reshape(dimension, dimension)
        guide_covariance = 0.5 * (
            guide_covariance + guide_covariance.T
        )
        guide_covariance += 1.0e-8 * np.eye(dimension, dtype=np.float64)
    guide_list = [
        EnergyPathGuide(
            kinetic_energy_mev=seed_energy,
            energy_proposal_sd_mev=energy_guide_sd,
            path=GaussianPathGuide(
                guide_mean,
                guide_covariance,
            ),
            mixture_weight=2.0,
            z_range_mean=seed_z,
            z_range_proposal_sd=1.0,
        )
    ]
    broad_anchor_covariance = 4.0 * np.eye(dimension, dtype=np.float64)
    configured_anchors = kwargs.get(
        "mixed_guide_energy_anchors_mev",
        (100.0, 200.0, 300.0, 450.0, 600.0),
    )
    retained_anchors = []
    anchor_guide_construction = []
    build_anchor_guides = bool(
        kwargs.get("mixed_build_energy_anchor_guides", True)
    )
    anchor_iterations = int(
        kwargs.get(
            "mixed_anchor_guide_latent_iterations",
            min(
                int(kwargs.get("mixed_guide_latent_iterations", 20)),
                12,
            ),
        )
    )
    for anchor in configured_anchors:
        anchor_energy = float(np.clip(float(anchor), energy_low, energy_high))
        if any(
            abs(anchor_energy - float(row.kinetic_energy_mev)) < 1.0e-9
            for row in guide_list
        ):
            continue
        retained_anchors.append(anchor_energy)
        anchor_mean = np.zeros(dimension, dtype=np.float64)
        anchor_covariance = broad_anchor_covariance
        construction = {
            "energy_mev": anchor_energy,
            "attempted": bool(build_anchor_guides),
            "succeeded": False,
            "method": (
                "charge_seed_then_profiled_charge_time_path_map"
                if guide_profile_t0
                else "charge_shape_path_map"
            ),
        }
        if build_anchor_guides:
            try:
                anchor_fit = fit_soft_path_guide(
                    anchor_energy,
                    0.0,
                    initial_coefficients=anchor_mean,
                    max_iterations=anchor_iterations,
                )
                anchor_mean = np.asarray(
                    anchor_fit.coefficients, dtype=np.float64
                )
                fitted_covariance = np.asarray(
                    getattr(
                        anchor_fit,
                        "_mixed_guide_proposal_covariance",
                        anchor_fit.covariance,
                    ),
                    dtype=np.float64,
                )
                if np.all(np.isfinite(fitted_covariance)):
                    anchor_covariance = guide_inflation * fitted_covariance
                    anchor_covariance = 0.5 * (
                        anchor_covariance + anchor_covariance.T
                    )
                    anchor_covariance += 1.0e-8 * np.eye(
                        dimension, dtype=np.float64
                    )
                construction.update(guide_fit_diagnostics(anchor_fit))
            except Exception as exc:
                construction["failure"] = f"{type(exc).__name__}: {exc}"
        anchor_guide_construction.append(construction)
        guide_list.append(EnergyPathGuide(
            kinetic_energy_mev=anchor_energy,
            energy_proposal_sd_mev=energy_guide_sd,
            path=GaussianPathGuide(
                anchor_mean,
                anchor_covariance,
            ),
            mixture_weight=1.0,
            z_range_mean=0.0,
            z_range_proposal_sd=1.0,
        ))

    # A single inflated Laplace covariance is a poor finite-particle proposal
    # in the 24-dimensional path space: even when its mean is an excellent
    # charge-plus-time solution, a small multinomial population can put no
    # particle near that basin.  Add normalized local copies at two smaller
    # scales.  These are proposal components only; the complete mixture density
    # is retained in every bridge and MH ratio, so neither the physical path
    # prior nor the likelihood is altered.
    configured_core_scales = kwargs.get(
        "mixed_guide_core_covariance_scales", (0.04, 0.20)
    )
    if np.isscalar(configured_core_scales):
        configured_core_scales = (float(configured_core_scales),)
    guide_core_scales = tuple(float(value) for value in configured_core_scales)
    if any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in guide_core_scales
    ):
        raise ValueError(
            "mixed guide core covariance scales must lie strictly in (0,1)"
        )
    broad_guides = tuple(guide_list)
    for scale in guide_core_scales:
        coordinate_scale = math.sqrt(scale)
        for guide in broad_guides:
            core_covariance = (
                scale * np.asarray(guide.path.covariance, dtype=np.float64)
                + 1.0e-10 * np.eye(dimension, dtype=np.float64)
            )
            guide_list.append(EnergyPathGuide(
                kinetic_energy_mev=float(guide.kinetic_energy_mev),
                energy_proposal_sd_mev=max(
                    2.0,
                    coordinate_scale * float(guide.energy_proposal_sd_mev),
                ),
                path=GaussianPathGuide(
                    np.asarray(guide.path.mean, dtype=np.float64),
                    core_covariance,
                ),
                mixture_weight=float(guide.mixture_weight),
                z_range_mean=float(guide.z_range_mean),
                z_range_proposal_sd=max(
                    0.15,
                    coordinate_scale * float(guide.z_range_proposal_sd),
                ),
            ))
    guides = tuple(guide_list)

    path_validation = {"evaluations": 0, "rejections": 0}

    def state_components(k0: float, z_value: float):
        realized_range = float(straggling.realized_range_mm(k0, z_value))
        candidate = theta.copy()
        candidate[5] = realized_range
        model = evaluator.model(
            candidate, initial_kinetic_energy_mev=float(k0)
        )
        if model is None:
            raise ValueError("invalid optical model for mixed state")
        prior = MixedMCSPathPrior(
            model.path_emitter,
            modes_per_plane=modes_per_plane,
            grid_points=grid_points,
            transport_grid_points=int(
                kwargs.get("mixed_transport_grid_points", 161)
            ),
        )
        return realized_range, candidate, model, prior

    def context_factory(k0: float, z_value: float) -> MixedEnergyRangeContext:
        realized_range, candidate, model, prior = state_components(k0, z_value)

        def validated_path(latent):
            path = prior.build_path(latent)
            path_validation["evaluations"] += 1
            if not _contained_curved_path(detector, path):
                path_validation["rejections"] += 1
                return None
            return path

        def timed_log_likelihood(latent, event_time_ns) -> float:
            path = validated_path(latent)
            if path is None:
                return -math.inf
            return -float(
                model.data_nll_from_path(path, t0=float(event_time_ns))
            )

        def exact_log_likelihood(latent) -> float:
            path = validated_path(latent)
            if path is None:
                return -math.inf
            if charge_only:
                value = model.charge_data_nll_from_path(path)
            elif profile_t0:
                value, _profiled_t0 = _profile_explicit_path_t0(
                    model,
                    path,
                    obs_pes=kwargs["obs_pes"],
                    obs_ts=kwargs["obs_ts"],
                    seed_t0=float(candidate[6]),
                    bounds=t0_bounds,
                    coarse_step_ns=t0_profile_coarse_step_ns,
                    refine_levels=t0_profile_refine_levels,
                    global_points=t0_profile_global_points,
                    seed_half_width_ns=t0_profile_seed_half_width_ns,
                )
            else:
                value = model.data_nll_from_path(path, t0=float(candidate[6]))
            return -float(value)

        return MixedEnergyRangeContext(
            kinetic_energy_mev=float(k0),
            z_range=float(z_value),
            realized_range_mm=realized_range,
            path_prior=prior,
            log_likelihood=exact_log_likelihood,
            log_likelihood_t0=(timed_log_likelihood if sample_t0 else None),
        )

    smc_config = MixedEnergyRangeSMCConfig(
            kinetic_energy_bounds_mev=(energy_low, energy_high),
            particles=particles,
            target_ess_fraction=float(
                kwargs.get("target_ess_fraction", 0.80)
            ),
            resample_ess_fraction=float(
                kwargs.get("resample_ess_fraction", 0.55)
            ),
            max_temperatures=int(kwargs.get("max_temperatures", 64)),
            rejuvenation_steps=int(kwargs.get("rejuvenation_steps", 3)),
            posterior_rejuvenation_steps=posterior_sweeps,
            energy_random_walk_mev=float(
                kwargs.get(
                    "mixed_energy_random_walk_mev",
                    min(float(kwargs.get("energy_random_walk_mev", 30.0)), 8.0),
                )
            ),
            energy_independence_probability=float(
                kwargs.get("energy_independence_probability", 0.15)
            ),
            event_time_bounds_ns=(
                tuple(map(float, t0_bounds)) if sample_t0 else None
            ),
            event_time_seed_ns=float(theta[6]),
            event_time_guide_sd_ns=float(
                kwargs.get("mixed_t0_guide_sd_ns", 1.0)
            ),
            event_time_prior_mixture_probability=float(
                kwargs.get("mixed_t0_prior_mixture_probability", 0.20)
            ),
            event_time_random_walk_ns=float(
                kwargs.get("mixed_t0_random_walk_ns", 0.20)
            ),
            range_pcn_rho=float(kwargs.get("range_pcn_rho", 0.80)),
            soft_pcn_rho=float(
                kwargs.get("mixed_soft_pcn_rho", 0.995)
            ),
            soft_guide_probability=float(
                kwargs.get(
                    "mixed_soft_guide_probability",
                    min(float(kwargs.get("path_guide_probability", 0.50)), 0.20),
                )
            ),
            hard_retention_probability=float(
                kwargs.get("mixed_hard_retention_probability", 0.72)
            ),
            hard_local_position_step_fraction=float(
                kwargs.get("mixed_hard_local_position_step_fraction", 0.08)
            ),
            hard_local_log_angle_step=float(
                kwargs.get("mixed_hard_local_log_angle_step", 0.35)
            ),
            hard_local_azimuth_step_rad=float(
                kwargs.get("mixed_hard_local_azimuth_step_rad", 0.50)
            ),
            guide_prior_mixture_probability=float(
                kwargs.get("joint_guide_prior_mixture_probability", 0.20)
            ),
            hard_empty_guide_probability=float(
                kwargs.get("mixed_hard_empty_guide_probability", 0.70)
            ),
            stratified_guide_initialization=bool(
                kwargs.get("mixed_stratified_guide_initialization", True)
            ),
            direct_importance_sampling=bool(
                inference_engine == "importance"
            ),
            initialization_attempts_per_particle=int(
                kwargs.get("mixed_initialization_attempts_per_particle", 100)
            ),
            random_seed=int(kwargs.get("random_seed", 41873)),
    )
    configured_stratum_edges = kwargs.get(
        "mixed_energy_stratum_edges_mev", ()
    )
    if isinstance(configured_stratum_edges, str):
        configured_stratum_edges = tuple(
            float(value)
            for value in configured_stratum_edges.replace(";", ",").split(",")
            if value.strip()
        )
    else:
        configured_stratum_edges = tuple(
            map(float, configured_stratum_edges or ())
        )
    stratum_replicates = int(
        kwargs.get("mixed_energy_stratum_replicates", 1)
    )
    if stratum_replicates < 1:
        raise ValueError("mixed energy-stratum replicates must be positive")
    if len(configured_stratum_edges) > 2 or stratum_replicates > 1:
        if not configured_stratum_edges:
            configured_stratum_edges = (energy_low, energy_high)
        smc = run_stratified_mixed_energy_range_smc(
            context_factory,
            dimension,
            guides,
            smc_config,
            configured_stratum_edges,
            replicates_per_stratum=stratum_replicates,
        )
    else:
        smc = run_mixed_energy_range_smc(
            context_factory,
            dimension,
            guides,
            smc_config,
        )
    posterior = smc.summary()
    evidence_stability_threshold_log_units = float(
        kwargs.get("mixed_evidence_stability_max_log_range", 1.0)
    )
    replicate_range_stability_threshold_mm = float(
        kwargs.get("mixed_replicate_range_mean_max_spread_mm", 20.0)
    )
    minimum_importance_weight_ess = float(
        kwargs.get("mixed_importance_min_weight_ess", 8.0)
    )
    minimum_importance_weight_ess_fraction = float(
        kwargs.get("mixed_importance_min_weight_ess_fraction", 0.01)
    )
    evidence_stability_issues = []
    stratum_rows = list(posterior.get("energy_strata", []))
    if not bool(posterior.get("energy_stratified", False)):
        evidence_stability_issues.append(
            "broad energy support was not split into disjoint strata"
        )
    if not bool(posterior.get("energy_stratum_replicated", False)):
        evidence_stability_issues.append(
            "energy strata do not have independent evidence replicates"
        )
    for stratum_index in sorted({
        int(row.get("index", -1)) for row in stratum_rows
    }):
        group = [
            row for row in stratum_rows
            if int(row.get("index", -1)) == stratum_index
        ]
        if len(group) < 3:
            evidence_stability_issues.append(
                f"energy stratum {stratum_index} has {len(group)} < 3 replicates"
            )
            continue
        log_range = float(group[0].get(
            "conditional_log_evidence_range", math.inf
        ))
        range_spread = float(group[0].get(
            "replicate_range_mean_range_mm", math.inf
        ))
        if (
            not math.isfinite(log_range)
            or log_range > evidence_stability_threshold_log_units
        ):
            evidence_stability_issues.append(
                f"energy stratum {stratum_index} evidence log-range "
                f"{log_range:.6g} exceeds "
                f"{evidence_stability_threshold_log_units:.6g}"
            )
        if (
            not math.isfinite(range_spread)
            or range_spread > replicate_range_stability_threshold_mm
        ):
            evidence_stability_issues.append(
                f"energy stratum {stratum_index} posterior range-mean spread "
                f"{range_spread:.6g} mm exceeds "
                f"{replicate_range_stability_threshold_mm:.6g} mm"
            )
        if inference_engine == "importance":
            for row in group:
                weight_ess = float(row.get(
                    "nominal_posterior_weight_ess", math.nan
                ))
                weight_ess_fraction = float(row.get(
                    "posterior_weight_ess_fraction", math.nan
                ))
                if (
                    not math.isfinite(weight_ess)
                    or weight_ess < minimum_importance_weight_ess
                ):
                    evidence_stability_issues.append(
                        f"energy stratum {stratum_index} replicate "
                        f"{int(row.get('replicate', -1))} importance-weight "
                        f"ESS {weight_ess:.6g} is below "
                        f"{minimum_importance_weight_ess:.6g}"
                    )
                if (
                    not math.isfinite(weight_ess_fraction)
                    or weight_ess_fraction
                    < minimum_importance_weight_ess_fraction
                ):
                    evidence_stability_issues.append(
                        f"energy stratum {stratum_index} replicate "
                        f"{int(row.get('replicate', -1))} importance-weight "
                        f"ESS fraction {weight_ess_fraction:.6g} is below "
                        f"{minimum_importance_weight_ess_fraction:.6g}"
                    )
    within_event_evidence_stability_pass = bool(
        stratum_rows and not evidence_stability_issues
    )
    # Fail only on categorical collapse that makes a posterior distribution
    # impossible to estimate.  Larger ESS/diversity thresholds would need
    # calibrated ensemble evidence and must not be inferred from one p8 run.
    minimum_unique_soft_states = 2
    minimum_unique_joint_states = 2
    minimum_diversity_failures = []
    if int(posterior["unique_soft_path_states"]) < minimum_unique_soft_states:
        minimum_diversity_failures.append(
            "unique_soft_path_states="
            f"{posterior['unique_soft_path_states']} < {minimum_unique_soft_states}"
        )
    if int(posterior["unique_joint_states"]) < minimum_unique_joint_states:
        minimum_diversity_failures.append(
            "unique_joint_states="
            f"{posterior['unique_joint_states']} < {minimum_unique_joint_states}"
        )
    within_run_minimum_diversity_pass = bool(
        not minimum_diversity_failures
    )
    posterior_range = float(posterior["realized_range_mm_mean"])
    posterior_energy = float(posterior["initial_kinetic_energy_mev_mean"])
    posterior_theta = theta.copy()
    posterior_theta[5] = posterior_range
    updated_resolved = evaluator.resolved_track(posterior_theta)
    if updated_resolved is None or bool(updated_resolved.exits_detector):
        raise RuntimeError(
            "posterior-mean mixed range is outside the contained topology"
        )

    coefficients = smc.coefficients
    coefficient_mean = np.asarray(
        smc.weights @ coefficients, dtype=np.float64
    )
    centered = coefficients - coefficient_mean[None, :]
    coefficient_covariance = (centered * smc.weights[:, None]).T @ centered
    # This score includes K0, z_R, Gaussian soft, and marked-Poisson hard priors.
    map_index = int(np.argmax(smc.posterior_log_density))
    map_event_time_t0 = float(theta[6])
    reported_event_time_t0 = float(theta[6])
    if sample_t0:
        map_event_time_t0 = float(smc.event_time_ns[map_index])
        reported_event_time_t0 = float(posterior["event_time_ns_mean"])
    if profile_t0:
        (
            _map_range,
            map_candidate,
            map_model,
            map_prior,
        ) = state_components(
            float(smc.kinetic_energy_mev[map_index]),
            float(smc.z_range[map_index]),
        )
        map_path = map_prior.build_path(smc.latents[map_index])
        if not _contained_curved_path(detector, map_path):
            raise RuntimeError("maximum-density mixed path leaves detector")
        _map_nll, map_event_time_t0 = _profile_explicit_path_t0(
            map_model,
            map_path,
            obs_pes=kwargs["obs_pes"],
            obs_ts=kwargs["obs_ts"],
            seed_t0=float(map_candidate[6]),
            bounds=t0_bounds,
            coarse_step_ns=t0_profile_coarse_step_ns,
            refine_levels=t0_profile_refine_levels,
            global_points=t0_profile_global_points,
            seed_half_width_ns=t0_profile_seed_half_width_ns,
        )
        reported_event_time_t0 = float(map_event_time_t0)
    likelihood_component_diagnostics = {
        "enabled": False,
        "failure": None,
    }
    if (
        not charge_only
        and bool(kwargs.get("mixed_report_likelihood_components", True))
    ):
        try:
            charge_nll = np.empty(smc.weights.size, dtype=np.float64)
            for sample_index, (k0, z_value, event_t0, latent) in enumerate(zip(
                smc.kinetic_energy_mev,
                smc.z_range,
                smc.event_time_ns,
                smc.latents,
                strict=True,
            )):
                _sample_range, _candidate, sample_model, sample_prior = (
                    state_components(float(k0), float(z_value))
                )
                sample_path = sample_prior.build_path(latent)
                if not _contained_curved_path(detector, sample_path):
                    raise RuntimeError(
                        "recorded posterior sample leaves detector"
                    )
                expected_charge = np.asarray(
                    sample_model.prediction_from_path(sample_path)[0],
                    dtype=np.float64,
                )
                charge_nll[sample_index] = float(
                    sample_model.pmt_model.get_neg_log_likelihood_npe(
                        expected_charge,
                        kwargs["obs_pes"],
                    )
                )
            total_nll = -np.asarray(smc.log_likelihood, dtype=np.float64)
            timing_nll = total_nll - charge_nll
            energy_values = np.asarray(
                smc.kinetic_energy_mev, dtype=np.float64
            )
            energy_mean = float(np.dot(smc.weights, energy_values))
            energy_centered = energy_values - energy_mean
            energy_variance = float(
                np.dot(smc.weights, energy_centered * energy_centered)
            )

            def component_summary(values):
                mean = float(np.dot(smc.weights, values))
                centered = values - mean
                return {
                    "posterior_weighted_mean": mean,
                    "posterior_weighted_sd": float(math.sqrt(max(
                        np.dot(smc.weights, centered * centered), 0.0
                    ))),
                    "energy_linear_slope_nll_per_mev": (
                        float(
                            np.dot(smc.weights, energy_centered * centered)
                            / energy_variance
                        )
                        if energy_variance > 0.0 else math.nan
                    ),
                    "map_value": float(values[map_index]),
                }

            likelihood_component_diagnostics = {
                "enabled": True,
                "failure": None,
                "definition": (
                    "configured compound charge NLL plus conditional "
                    "first-arrival timing NLL; their sum is the sampled data NLL"
                ),
                "charge": component_summary(charge_nll),
                "timing": component_summary(timing_nll),
                "maximum_recomposition_error": float(np.max(np.abs(
                    charge_nll + timing_nll - total_nll
                ))),
            }
        except Exception as exc:
            likelihood_component_diagnostics = {
                "enabled": False,
                "failure": f"{type(exc).__name__}: {exc}",
            }
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
        "t0": float(reported_event_time_t0),
    }
    guide_diagnostics = {
        "enabled": True,
        "target_altering": False,
        "guide_count": int(len(guides)),
        "seed_energy_mev": seed_energy,
        "seed_z_range": seed_z,
        "path_mean_source": (
            "charge_shape_map_with_optional_profiled_timing_polish"
            if guide_construction["succeeded"]
            else "zero"
            if initial_path is None
            else "caller_propagated_path"
        ),
        "construction": guide_construction,
        "covariance_inflation": guide_inflation,
        "broad_guide_count": int(len(broad_guides)),
        "core_covariance_scales": [
            float(value) for value in guide_core_scales
        ],
        "randomized_stratified_initialization": bool(
            smc_config.stratified_guide_initialization
        ),
        "hard_empty_guide_probability": float(
            smc_config.hard_empty_guide_probability
        ),
        "guide_component_covariance_spectrum": [
            {
                "energy_mev": float(guide.kinetic_energy_mev),
                "energy_sd_mev": float(guide.energy_proposal_sd_mev),
                "z_range_sd": float(guide.z_range_proposal_sd),
                "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(
                    guide.path.covariance
                ))),
                "maximum_eigenvalue": float(np.max(np.linalg.eigvalsh(
                    guide.path.covariance
                ))),
            }
            for guide in guides
        ],
        "broad_energy_anchor_mev": [float(value) for value in retained_anchors],
        "energy_anchor_path_guides_enabled": bool(build_anchor_guides),
        "energy_anchor_path_guide_iterations": int(anchor_iterations),
        "energy_anchor_construction": anchor_guide_construction,
        "selection_policy": (
            "independently fitted soft-path guides at separated energy anchors "
            "with narrow, medium, and broad covariance scales plus a "
            "uniform/normal defensive mixture; randomized systematic "
            "component allocation guarantees finite-particle basin coverage, "
            "and every proposal density is included exactly in annealing and "
            "Metropolis ratios"
        ),
    }
    global_preconditioner_diagnostics = {
        "enabled": bool(global_preconditioner is not None),
        "role": (
            "truth_blind_charge_data_global_line_conditioning_before_mixed_target"
            if global_preconditioner is not None
            else "disabled"
        ),
        "line_geometry_marginalized": False,
        "input_start_mm": [
            float(caller_values[name]) for name in ("x0", "y0", "z0")
        ],
        "input_direction": [
            float(value) for value in caller_chart.anchor
        ],
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
    diagnostics = {
        "implementation": "continuous_noncentered_energy_range_mixed_wentzel_smc_v1",
        "experimental": True,
        "production_ready": False,
        "within_run_minimum_diversity_pass": bool(
            within_run_minimum_diversity_pass
        ),
        "convergence_certified": False,
        "configuration_underpowered": bool(configuration_issues),
        "configuration_quality_issues": list(configuration_issues),
        "minimum_diversity_failures": list(minimum_diversity_failures),
        "minimum_particles_for_dimension": int(minimum_particles),
        "minimum_posterior_rejuvenation_sweeps": int(
            minimum_posterior_sweeps
        ),
        "minimum_unique_soft_path_states": int(minimum_unique_soft_states),
        "minimum_unique_joint_states": int(minimum_unique_joint_states),
        "duplicate_state_ess_is_diagnostic_only": True,
        "underpowered_diagnostic_override": bool(allow_underpowered),
        "cross_seed_replication_required": True,
        "recommended_minimum_independent_seeds": 3,
        "within_event_evidence_stability_pass": bool(
            within_event_evidence_stability_pass
        ),
        "evidence_stability_issues": list(evidence_stability_issues),
        "evidence_stability_max_log_range": float(
            evidence_stability_threshold_log_units
        ),
        "replicate_range_mean_max_spread_mm": float(
            replicate_range_stability_threshold_mm
        ),
        "importance_minimum_weight_ess": float(
            minimum_importance_weight_ess
        ),
        "importance_minimum_weight_ess_fraction": float(
            minimum_importance_weight_ess_fraction
        ),
        "cross_seed_acceptance_guidance": (
            "compare posterior K0/range means, tail probabilities, and hard-count "
            "moments across at least three independently seeded runs; a single "
            "run is not an ensemble convergence certificate"
        ),
        "inference_method": (
            "independent_importance_sampling"
            if inference_engine == "importance"
            else "reference_smc"
        ),
        "line_geometry_conditioned": True,
        "line_geometry_marginalized": False,
        "line_geometry_source": (
            "coherent_FE_charge_profile_preconditioner"
            if global_preconditioner is not None
            else "caller_supplied_straight_fit"
        ),
        "global_line_preconditioner": global_preconditioner_diagnostics,
        "starts_at_boundary": starts_at_boundary,
        "exits_detector": False,
        "initial_topology": str(initial_resolved.topology),
        "updated_topology": str(updated_resolved.topology),
        "modes_per_plane": modes_per_plane,
        "latent_dimension": dimension,
        "grid_points": grid_points,
        "kinetic_energy_prior": "uniform_on_documented_broad_support",
        "range_prior": "Geant4_UniversalFluctuation_first_passage_moment",
        "path_prior": "soft_Wentzel_FE_KL_plus_marked_Poisson_hard_Wentzel",
        "map_definition": (
            "maximum sampled density under dK0 dz "
            + ("dt0 " if sample_t0 else "")
            + "du and ordered prod(ds dtheta dphi) hard-mark measure, "
            "including every prior"
        ),
        "map_log_posterior_density": float(
            smc.posterior_log_density[map_index]
        ),
        "uses_event_truth": False,
        "uses_empirical_mcs_scale": False,
        "uses_wcsim_range_width": False,
        "uses_discrete_range_grid": False,
        "range_coordinate": "continuous_noncentered_float64_z_R",
        "energy_coordinate": "continuous_float64_K0_mev",
        "energy_stratified": bool(
            posterior.get("energy_stratified", False)
        ),
        "energy_stratum_replicated": bool(
            posterior.get("energy_stratum_replicated", False)
        ),
        "energy_strata": list(posterior.get("energy_strata", [])),
        "path_grid_role": "finite_element_quadrature_only",
        "output_length_quantization_mm": None,
        "charge_only": bool(charge_only),
        "timing_used": bool(not charge_only),
        "likelihood_components": likelihood_component_diagnostics,
        "t0_treatment": t0_treatment,
        "t0_sampled": bool(sample_t0),
        "t0_profiled": bool(profile_t0),
        "reported_t0_ns": float(reported_event_time_t0),
        "map_event_time_t0_ns": float(map_event_time_t0),
        "t0_profile_seed_ns": float(theta[6]),
        "t0_profile_bounds_ns": (
            None if t0_bounds is None else list(map(float, t0_bounds))
        ),
        "t0_profile_coarse_step_ns": float(t0_profile_coarse_step_ns),
        "t0_profile_refine_levels": int(t0_profile_refine_levels),
        "t0_profile_global_points": int(t0_profile_global_points),
        "t0_profile_seed_half_width_ns": float(
            t0_profile_seed_half_width_ns
        ),
        "receiver_mode": "sparse" if sparse_receiver else "dense",
        "curved_path_domain_policy": (
            "reject_any_node_or_represented_segment_outside_detector"
        ),
        "curved_path_validation_evaluations": int(
            path_validation["evaluations"]
        ),
        "curved_path_validation_rejections": int(
            path_validation["rejections"]
        ),
        "optical_model_build_count": int(evaluator.optical_model_build_count),
        "coherent_field_evaluation_count": int(
            evaluator.coherent_field_evaluation_count
        ),
        "invalid_model_evaluation_count": int(evaluator.invalid_evaluations),
        "optical_model_cache_limit": int(
            kwargs.get("mixed_model_cache_size", 16)
        ),
        "optical_model_cache_evictions": int(evaluator.model_cache_evictions),
        "posterior": posterior,
    }
    if minimum_diversity_failures and not allow_underpowered:
        raise MixedMCSConvergenceError(
            "mixed-MCS posterior suffered categorical state collapse: "
            + "; ".join(minimum_diversity_failures)
            + f"; unique_initial_lineages={posterior['unique_initial_lineages']}. "
            "Do not apply this continuation; increase particles/posterior "
            "rejuvenation and repeat independent random seeds."
        )
    if (
        inference_engine == "importance"
        and evidence_stability_issues
        and not allow_underpowered
    ):
        raise MixedMCSConvergenceError(
            "mixed-MCS independent importance sampling failed its within-event "
            "weight/evidence certificate: "
            + "; ".join(evidence_stability_issues)
            + ". Increase proposal quality/sample count or use the annealed "
            "reference engine; this conditional estimate must not be applied."
        )
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


__all__ = [
    "MixedMCSConvergenceError",
    "MixedMCSUnderpoweredConfiguration",
    "run_cosmic_joint_mixed_energy_range_update",
]
