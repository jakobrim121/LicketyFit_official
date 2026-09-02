"""Fast orientation-independent track-fit primitives.

This module contains the numerical pieces that were previously embedded in the
WCSim batch driver:

* geometry-derived convex vertex constraints;
* a tangent direction chart with no global-axis singularity;
* track-aligned low-call block optimization;
* finite-safe quadratic stencils; and
* staged proxy/exact optimization with chart re-anchoring.

The optical prediction remains entirely in :mod:`LicketyFit.Emitter`; this file
changes only fit coordinates and optimizer navigation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import os
import time
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .track_parameterization import (
    TangentDirectionChart,
    attach_direction_components,
    direction_from_mapping,
    local_to_cartesian_covariance,
    reanchor_values,
)
from .detector_geometry import (
    WCTE_PRISM_APOTHEM_MM,
    WCTE_PRISM_N_SIDES,
    WCTE_PRISM_Y_MAX_MM,
    WCTE_PRISM_Y_MIN_MM,
)


_BATCH_T0_BLOCK_STENCIL = str(
    os.environ.get("LF_BATCH_T0_BLOCK_STENCIL", "1")
).strip().lower() not in {"0", "false", "no", "off"}
_EXACT_CHARGE_NLL_REUSE = str(
    os.environ.get("LF_EXACT_CHARGE_NLL_REUSE", "1")
).strip().lower() not in {"0", "false", "no", "off"}


@lru_cache(maxsize=8)
def _regular_prism_face_normals_xz(n_sides: int) -> tuple[tuple[float, float], ...]:
    """Return detector-static regular-prism face normals.

    Geometry containment is evaluated thousands of times per fit.  The face
    angles are detector constants, so recomputing the same sine and cosine in
    every call adds latency without adding physical information.  Computing
    them once retains the exact scalar inequalities and operation order.
    """
    n = int(n_sides)
    if n <= 0:
        return ()
    return tuple(
        (
            math.cos(2.0 * math.pi * j / float(n)),
            math.sin(2.0 * math.pi * j / float(n)),
        )
        for j in range(n)
    )


_WCTE_PRISM_FACE_NORMALS_XZ = _regular_prism_face_normals_xz(
    WCTE_PRISM_N_SIDES
)


@dataclass(frozen=True)
class ConvexDetectorVolume:
    """Convex detector interior and ray-exit model.

    Two boundary representations are supported:

    ``convex_mpmt_planes``
        A detector-generic intersection of inward-facing mPMT base planes and
        the geometry bounding box.  This is suitable for a convex IWCD layout
        and is deliberately independent of any WCTE dimensions.

    ``wcte_prism``
        The exact 16-sided WCTE inner-water prism used by the optical transport.
        This keeps the track containment and photon boundary mathematically
        consistent instead of approximating WCTE with the mPMT base planes.
    """

    locations: np.ndarray
    inward_axes: np.ndarray
    axis_lo: np.ndarray
    axis_hi: np.ndarray
    margin_mm: float = 0.0
    reference_center: np.ndarray | None = None
    normal_flips: int = 0
    boundary_model: str = "convex_mpmt_planes"
    prism_n_sides: int = 0
    prism_apothem_mm: float = math.nan
    prism_y_min_mm: float = math.nan
    prism_y_max_mm: float = math.nan

    @classmethod
    def from_wcd(
        cls,
        wcd,
        *,
        placement: str = "design",
        margin_mm: float = 0.0,
        fallback_bounds: Sequence[Sequence[float]] | None = None,
        boundary_model: str = "convex_mpmt_planes",
    ) -> "ConvexDetectorVolume":
        model = str(boundary_model).strip().lower().replace("-", "_")
        if model not in {"convex_mpmt_planes", "wcte_prism"}:
            raise ValueError(
                "boundary_model must be 'convex_mpmt_planes' or 'wcte_prism'"
            )
        locations: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        for mpmt in getattr(wcd, "mpmts", []):
            if mpmt is None:
                continue
            try:
                p = mpmt.get_placement(placement, wcd)
                location = np.asarray(p["location"], dtype=np.float64)
                axis = np.asarray(p["direction_z"], dtype=np.float64)
                norm = float(np.linalg.norm(axis))
                if (
                    location.shape == (3,)
                    and axis.shape == (3,)
                    and norm > 0.0
                    and np.all(np.isfinite(location))
                    and np.all(np.isfinite(axis))
                ):
                    locations.append(location)
                    axes.append(axis / norm)
            except Exception:
                continue

        if locations:
            loc = np.ascontiguousarray(np.asarray(locations, dtype=np.float64))
            raw_axes = np.asarray(axes, dtype=np.float64)
            # Geometry files are not fully consistent about whether an mPMT's
            # local +z axis points into or out of the water volume.  Orient every
            # plane normal toward a robust detector centre so the half-space
            # convention used below is deterministic for WCTE and IWCD alike.
            centre = np.median(loc, axis=0)
            toward_centre = centre[None, :] - loc
            flip = np.einsum("ij,ij->i", toward_centre, raw_axes) < 0.0
            raw_axes[flip] *= -1.0
            inward = np.ascontiguousarray(raw_axes, dtype=np.float64)
            normal_flips = int(np.count_nonzero(flip))
            lo = np.min(loc, axis=0)
            hi = np.max(loc, axis=0)
        else:
            loc = np.empty((0, 3), dtype=np.float64)
            inward = np.empty((0, 3), dtype=np.float64)
            if fallback_bounds is None:
                fallback_bounds = ((-2000.0, 2000.0),) * 3
            bounds = np.asarray(fallback_bounds, dtype=np.float64)
            if bounds.shape != (3, 2):
                raise ValueError("fallback_bounds must have shape (3,2)")
            lo = bounds[:, 0]
            hi = bounds[:, 1]
            centre = 0.5 * (lo + hi)
            normal_flips = 0
        if model == "wcte_prism":
            # The prism's Cartesian bounding box is exact because face normals
            # include the +/-x and +/-z directions.
            lo = np.asarray(
                [-WCTE_PRISM_APOTHEM_MM, WCTE_PRISM_Y_MIN_MM,
                 -WCTE_PRISM_APOTHEM_MM],
                dtype=np.float64,
            )
            hi = np.asarray(
                [WCTE_PRISM_APOTHEM_MM, WCTE_PRISM_Y_MAX_MM,
                 WCTE_PRISM_APOTHEM_MM],
                dtype=np.float64,
            )
            centre = 0.5 * (lo + hi)

        return cls(
            loc,
            inward,
            np.ascontiguousarray(lo, dtype=np.float64),
            np.ascontiguousarray(hi, dtype=np.float64),
            float(margin_mm),
            np.ascontiguousarray(centre, dtype=np.float64),
            int(normal_flips),
            model,
            int(WCTE_PRISM_N_SIDES if model == "wcte_prism" else 0),
            float(WCTE_PRISM_APOTHEM_MM if model == "wcte_prism" else math.nan),
            float(WCTE_PRISM_Y_MIN_MM if model == "wcte_prism" else math.nan),
            float(WCTE_PRISM_Y_MAX_MM if model == "wcte_prism" else math.nan),
        )

    def _effective_margin(self, extra_margin_mm: float) -> float:
        return max(float(self.margin_mm) + float(extra_margin_mm), 0.0)

    def contains(
        self,
        point: Sequence[float],
        *,
        tolerance_mm: float = 1.0e-8,
        extra_margin_mm: float = 0.0,
    ) -> bool:
        x = np.asarray(point, dtype=np.float64)
        if x.shape != (3,) or not np.all(np.isfinite(x)):
            return False
        margin = self._effective_margin(extra_margin_mm)
        if np.any(x < self.axis_lo + margin - tolerance_mm) or np.any(
            x > self.axis_hi - margin + tolerance_mm
        ):
            return False
        if self.boundary_model == "wcte_prism":
            limit = float(self.prism_apothem_mm) - margin
            face_normals = (
                _WCTE_PRISM_FACE_NORMALS_XZ
                if int(self.prism_n_sides) == int(WCTE_PRISM_N_SIDES)
                else _regular_prism_face_normals_xz(self.prism_n_sides)
            )
            for nx, nz in face_normals:
                if (
                    nx * float(x[0])
                    + nz * float(x[2])
                    > limit + tolerance_mm
                ):
                    return False
            return True
        if self.locations.size:
            distances = np.einsum("ij,ij->i", x[None, :] - self.locations, self.inward_axes)
            if float(np.min(distances)) < margin - tolerance_mm:
                return False
        return True

    def contains_many(
        self,
        points,
        *,
        tolerance_mm: float = 1.0e-8,
        extra_margin_mm: float = 0.0,
    ) -> bool:
        """Vectorized equivalent of calling :meth:`contains` for every point.

        Coherent paths carry 81--241 nodes and are validated after every exact
        optical evaluation.  Applying the same convex half-space inequalities
        to whole coordinate columns removes Python method/allocation overhead;
        it does not approximate or omit a detector boundary.
        """
        xyz = np.asarray(points, dtype=np.float64)
        if (
            xyz.ndim != 2
            or xyz.shape[1:] != (3,)
            or xyz.shape[0] == 0
            or not np.all(np.isfinite(xyz))
        ):
            return False
        margin = self._effective_margin(extra_margin_mm)
        if np.any(xyz < self.axis_lo[None, :] + margin - tolerance_mm) or np.any(
            xyz > self.axis_hi[None, :] - margin + tolerance_mm
        ):
            return False
        if self.boundary_model == "wcte_prism":
            limit = float(self.prism_apothem_mm) - margin + tolerance_mm
            face_normals = (
                _WCTE_PRISM_FACE_NORMALS_XZ
                if int(self.prism_n_sides) == int(WCTE_PRISM_N_SIDES)
                else _regular_prism_face_normals_xz(self.prism_n_sides)
            )
            x = xyz[:, 0]
            z = xyz[:, 2]
            for nx, nz in face_normals:
                if np.any(nx * x + nz * z > limit):
                    return False
            return True
        if self.locations.size:
            # Preserve the plane-by-plane half-space definition while doing
            # each plane's independent point arithmetic in compiled NumPy.
            threshold = margin - tolerance_mm
            for location, inward in zip(self.locations, self.inward_axes):
                distances = np.sum(
                    (xyz - location[None, :]) * inward[None, :], axis=1
                )
                if np.any(distances < threshold):
                    return False
        return True

    def ray_exit_distance(
        self,
        point: Sequence[float],
        direction: Sequence[float],
        *,
        extra_margin_mm: float = 0.0,
        tolerance_mm: float = 1.0e-8,
    ) -> float:
        """Return the forward distance to the first detector boundary [mm].

        The input point must lie inside the requested shrunken volume.  A NaN
        is returned for invalid inputs; an outward direction from a boundary
        correctly returns zero.
        """
        x = np.asarray(point, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        if x.shape != (3,) or d.shape != (3,):
            return math.nan
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(d))):
            return math.nan
        norm = float(np.linalg.norm(d))
        if norm <= 0.0:
            return math.nan
        d = d / norm
        margin = self._effective_margin(extra_margin_mm)
        if not self.contains(
            x, tolerance_mm=tolerance_mm, extra_margin_mm=extra_margin_mm
        ):
            return math.nan

        candidates: list[float] = []

        # Cartesian box constraints are useful for the generic plane model and
        # exactly reproduce the y caps / x,z extrema for the WCTE prism.
        lo = self.axis_lo + margin
        hi = self.axis_hi - margin
        for k in range(3):
            if d[k] > 1.0e-15:
                candidates.append(float((hi[k] - x[k]) / d[k]))
            elif d[k] < -1.0e-15:
                candidates.append(float((lo[k] - x[k]) / d[k]))

        if self.boundary_model == "wcte_prism":
            limit = float(self.prism_apothem_mm) - margin
            face_normals = (
                _WCTE_PRISM_FACE_NORMALS_XZ
                if int(self.prism_n_sides) == int(WCTE_PRISM_N_SIDES)
                else _regular_prism_face_normals_xz(self.prism_n_sides)
            )
            for nx, nz in face_normals:
                velocity = nx * float(d[0]) + nz * float(d[2])
                if velocity > 1.0e-15:
                    distance = (
                        limit - nx * float(x[0]) - nz * float(x[2])
                    ) / velocity
                    candidates.append(float(distance))
        elif self.locations.size:
            current = np.einsum(
                "ij,ij->i", x[None, :] - self.locations, self.inward_axes
            )
            velocity = self.inward_axes @ d
            outward = velocity < -1.0e-15
            if np.any(outward):
                values = (current[outward] - margin) / (-velocity[outward])
                candidates.extend(float(v) for v in values if np.isfinite(v))

        positive = [v for v in candidates if math.isfinite(v) and v >= -tolerance_mm]
        if not positive:
            return math.inf
        return max(0.0, float(min(positive)))

    def segment_contained(
        self,
        point: Sequence[float],
        direction: Sequence[float],
        length_mm: float,
        *,
        extra_margin_mm: float = 0.0,
        tolerance_mm: float = 1.0e-6,
    ) -> bool:
        length = float(length_mm)
        if not math.isfinite(length) or length < 0.0:
            return False
        exit_distance = self.ray_exit_distance(
            point,
            direction,
            extra_margin_mm=extra_margin_mm,
            tolerance_mm=tolerance_mm,
        )
        return bool(
            math.isfinite(exit_distance)
            and length <= exit_distance + float(tolerance_mm)
        )

    def project_step(self, point: Sequence[float], delta: Sequence[float]) -> np.ndarray:
        """Shorten a vertex step to the first detector boundary.

        This keeps finite-difference stencil points physical without clipping
        each Cartesian component independently, which would rotate the intended
        track-aligned step near a wall.
        """
        x = np.asarray(point, dtype=np.float64)
        dx = np.asarray(delta, dtype=np.float64)
        if x.shape != (3,) or dx.shape != (3,):
            raise ValueError("point and delta must be 3-vectors")
        norm = float(np.linalg.norm(dx))
        if norm <= 0.0:
            return np.ascontiguousarray(x, dtype=np.float64)
        distance = self.ray_exit_distance(x, dx / norm)
        if not math.isfinite(distance):
            raise ValueError("cannot project a step from a point outside the detector")
        alpha = min(1.0, max(0.0, distance / norm))
        if alpha < 1.0:
            alpha *= 1.0 - 1.0e-10
        out = x + alpha * dx
        return np.ascontiguousarray(out, dtype=np.float64)

    def metadata(self) -> dict[str, object]:
        return {
            "boundary_model": str(self.boundary_model),
            "axis_limits_mm": np.column_stack((self.axis_lo, self.axis_hi)).tolist(),
            "inward_plane_count": int(self.locations.shape[0]),
            "margin_mm": float(self.margin_mm),
            "normal_flips": int(self.normal_flips),
            "reference_center_mm": (
                None if self.reference_center is None
                else np.asarray(self.reference_center, dtype=np.float64).tolist()
            ),
            "prism": (
                None
                if self.boundary_model != "wcte_prism"
                else {
                    "n_sides": int(self.prism_n_sides),
                    "apothem_mm": float(self.prism_apothem_mm),
                    "y_min_mm": float(self.prism_y_min_mm),
                    "y_max_mm": float(self.prism_y_max_mm),
                }
            ),
        }


@dataclass
class FitEvaluation:
    fval: float
    values: dict[str, float]


@dataclass
class BlockOptimizerResult:
    values: dict[str, float]
    fval: float
    errors: dict[str, float]
    nfcn: int
    history: list[dict[str, object]]
    chart: TangentDirectionChart
    wall_s: float
    invalid_evaluations: int = 0
    quadratic_skips: int = 0

    def output_values(self) -> dict[str, object]:
        return attach_direction_components(self.values, chart=self.chart)


@dataclass(frozen=True)
class T0ProfileResult:
    values: dict[str, float]
    fval: float
    error_ns: float
    nll_evaluations: int
    optical_evaluations: int
    sampled_points: tuple[tuple[float, float], ...]
    wall_s: float


@dataclass(frozen=True)
class OptimizerSteps:
    longitudinal_mm: float = 60.0
    transverse_mm: float = 60.0
    direction_tangent: float = 0.035
    length_mm: float = 100.0
    full_range_mm: float = 120.0
    t0_ns: float = 0.10


@dataclass(frozen=True)
class OptimizerMinSteps:
    longitudinal_mm: float = 0.5
    transverse_mm: float = 0.5
    direction_tangent: float = 2.0e-4
    length_mm: float = 1.0
    full_range_mm: float = 1.0
    t0_ns: float = 1.0e-3


@dataclass
class TrackObjective:
    """Callable adapter from local chart values to the optical NLL."""

    emitter: object
    wcd: object
    pmt_model: object
    p_locations: np.ndarray
    pmt_normals: np.ndarray
    obs_pes: np.ndarray
    obs_ts: np.ndarray
    chart: TangentDirectionChart
    detector: ConvexDetectorVolume
    objective_mode: str = "charge_only"
    mpmt_types: object | None = None
    track_end_mode: str = "full_length"
    range_lookup: object | None = None
    particle_threshold_mev: float = 0.0
    use_t0_prior: bool = False
    t0_prior_sigma: float | None = None
    t0_limits: tuple[float, float] | None = None
    require_contained_track: bool = True
    containment_tolerance_mm: float = 1.0e-6
    # In public full_length mode the fitted coordinate may represent the full
    # remaining CSDA range while only the in-detector segment is optically
    # visible. The historical contained model is recovered exactly when this
    # switch is false or when the range ends before the boundary.
    clip_full_length_to_boundary: bool = False
    boundary_clip_inset_mm: float = 0.5
    cache: dict[tuple[float, ...], float] = field(default_factory=dict)
    prediction_cache: dict[
        tuple[float, ...], tuple[np.ndarray, object, np.ndarray | None] | None
    ] = field(default_factory=dict)
    charge_nll_cache: dict[tuple[float, ...], float] = field(default_factory=dict)
    # Event-mean nuisance values are geometry dependent just like the optical
    # prediction.  Cache them alongside that prediction so a caller can report
    # the value at the accepted geometry without relying on whichever trial
    # happened to be the Emitter's most recent evaluation.
    event_mean_contamination_cache: dict[tuple[float, ...], float] = field(
        default_factory=dict
    )
    calls: int = 0
    evaluations: int = 0
    optical_evaluations: int = 0
    invalid_evaluations: int = 0

    def _key(self, values: Mapping[str, float]) -> tuple[float, ...]:
        if self.track_end_mode == "absorption":
            names = (
                "x0", "y0", "z0", "dir_u", "dir_v",
                "visible_length", "full_range", "t0",
            )
        else:
            names = ("x0", "y0", "z0", "dir_u", "dir_v", "length", "t0")
        return tuple(round(float(values.get(name, 0.0)), 12) for name in names)

    def _geometry_key(self, values: Mapping[str, float]) -> tuple[float, ...]:
        if self.track_end_mode == "absorption":
            names = (
                "x0", "y0", "z0", "dir_u", "dir_v",
                "visible_length", "full_range",
            )
        else:
            names = ("x0", "y0", "z0", "dir_u", "dir_v", "length")
        return tuple(round(float(values.get(name, 0.0)), 12) for name in names)

    def _prediction_charge_nll(
        self,
        prediction,
        geometry_key: tuple[float, ...] | None,
    ) -> float:
        if (
            _EXACT_CHARGE_NLL_REUSE
            and geometry_key is not None
            and geometry_key in self.charge_nll_cache
        ):
            return float(self.charge_nll_cache[geometry_key])
        value = float(
            self.pmt_model.get_neg_log_likelihood_npe(
                prediction[0], self.obs_pes
            )
        )
        if _EXACT_CHARGE_NLL_REUSE and geometry_key is not None:
            self.charge_nll_cache[geometry_key] = value
        return value

    def _prediction_nll(
        self,
        prediction,
        t0: float,
        *,
        geometry_key: tuple[float, ...] | None = None,
    ) -> float:
        """Evaluate the configured likelihood for an existing optical field."""
        exp_pes_array, exp_ts_zero, timing_pes = prediction
        if self.objective_mode == "charge_only":
            fval = self._prediction_charge_nll(prediction, geometry_key)
        elif self.objective_mode == "timing_only":
            fval = self.pmt_model.get_neg_log_likelihood_t(
                exp_pes_array,
                self.obs_pes,
                exp_ts_zero,
                self.obs_ts,
                timing_pes=timing_pes,
                model_time_shift_ns=t0,
            )
        elif self.objective_mode == "charge_time":
            if (
                _EXACT_CHARGE_NLL_REUSE
                and geometry_key is not None
                and hasattr(exp_ts_zero, "first_arrival_active_indices")
            ):
                charge_nll = self._prediction_charge_nll(
                    prediction, geometry_key
                )
                timing_nll = self.pmt_model.get_neg_log_likelihood_t(
                    exp_pes_array,
                    self.obs_pes,
                    exp_ts_zero,
                    self.obs_ts,
                    timing_pes=timing_pes,
                    model_time_shift_ns=t0,
                )
                fval = float(charge_nll) + float(timing_nll)
            else:
                fval = self.pmt_model.get_neg_log_likelihood_npe_t(
                    exp_pes_array,
                    self.obs_pes,
                    exp_ts_zero,
                    self.obs_ts,
                    timing_pes=timing_pes,
                    model_time_shift_ns=t0,
                )
        else:
            raise ValueError(f"unknown objective_mode={self.objective_mode!r}")
        fval = float(fval)
        if (
            self.use_t0_prior
            and self.objective_mode != "charge_only"
            and self.t0_prior_sigma
        ):
            fval += 0.5 * (t0 / float(self.t0_prior_sigma)) ** 2
        return float(fval)

    def evaluate_t0_many(
        self, values: Mapping[str, float], t0_values
    ) -> np.ndarray:
        """Evaluate exact NLLs for many additive-time values in one kernel.

        Only the evaluation schedule changes: callers provide the same t0
        sample points as the scalar path.  The cached optical prediction is
        immutable in t0, and the PMT model's compiled many-t0 routine evaluates
        the identical conditional first-arrival likelihood for every point.
        """
        shifts = np.ascontiguousarray(t0_values, dtype=np.float64).reshape(-1)
        out = np.full(shifts.size, np.inf, dtype=np.float64)
        if shifts.size == 0:
            return out

        base = {k: float(v) for k, v in values.items()}
        geometry_key = self._geometry_key(base)
        # A scalar evaluation builds and validates this optical prediction.
        # Retain the scalar compatibility path for direct callers that have not
        # yet evaluated the geometry.
        if geometry_key not in self.prediction_cache:
            for i, shift in enumerate(shifts):
                trial = dict(base)
                trial["t0"] = float(shift)
                out[i] = float(self(trial))
            return out
        prediction = self.prediction_cache.get(geometry_key)
        if prediction is None:
            for i, shift in enumerate(shifts):
                trial = dict(base)
                trial["t0"] = float(shift)
                out[i] = float(self(trial))
            return out

        pending_indices: list[int] = []
        pending_values: list[float] = []
        pending_keys: list[tuple[float, ...]] = []
        for i, shift_value in enumerate(shifts):
            self.calls += 1
            shift = float(shift_value)
            trial = dict(base)
            trial["t0"] = shift
            key = self._key(trial)
            cached = self.cache.get(key)
            if cached is not None:
                out[i] = float(cached)
                continue
            self.evaluations += 1
            if not math.isfinite(shift):
                self.invalid_evaluations += 1
                self.cache[key] = np.inf
                continue
            if self.t0_limits is not None:
                lo, hi = (float(self.t0_limits[0]), float(self.t0_limits[1]))
                if shift < lo or shift > hi:
                    self.invalid_evaluations += 1
                    self.cache[key] = np.inf
                    continue
            pending_indices.append(i)
            pending_values.append(shift)
            pending_keys.append(key)

        if not pending_indices:
            return out

        exp_pes_array, exp_ts_zero, timing_pes = prediction
        shift_array = np.ascontiguousarray(pending_values, dtype=np.float64)
        values_include_prior = False
        if (
            self.objective_mode == "charge_time"
            and _EXACT_CHARGE_NLL_REUSE
            and hasattr(self.pmt_model, "get_neg_log_likelihood_t_many_t0")
            and hasattr(exp_ts_zero, "first_arrival_active_indices")
        ):
            charge_nll = self._prediction_charge_nll(
                prediction, geometry_key
            )
            values_array = self.pmt_model.get_neg_log_likelihood_t_many_t0(
                exp_pes_array,
                self.obs_pes,
                exp_ts_zero,
                self.obs_ts,
                shift_array,
                timing_pes=timing_pes,
            )
            values_array = np.asarray(values_array, dtype=np.float64) + float(
                charge_nll
            )
        elif (
            self.objective_mode == "charge_time"
            and hasattr(
                self.pmt_model, "get_neg_log_likelihood_npe_t_many_t0"
            )
        ):
            values_array = self.pmt_model.get_neg_log_likelihood_npe_t_many_t0(
                exp_pes_array,
                self.obs_pes,
                exp_ts_zero,
                self.obs_ts,
                shift_array,
                timing_pes=timing_pes,
            )
        elif (
            self.objective_mode == "timing_only"
            and hasattr(self.pmt_model, "get_neg_log_likelihood_t_many_t0")
        ):
            values_array = self.pmt_model.get_neg_log_likelihood_t_many_t0(
                exp_pes_array,
                self.obs_pes,
                exp_ts_zero,
                self.obs_ts,
                shift_array,
                timing_pes=timing_pes,
            )
        else:
            values_array = np.asarray(
                [
                    self._prediction_nll(
                        prediction,
                        float(v),
                        geometry_key=geometry_key,
                    )
                    for v in shift_array
                ],
                dtype=np.float64,
            )
            values_include_prior = True

        values_array = np.asarray(values_array, dtype=np.float64).reshape(-1)
        if values_array.size != len(pending_indices):
            raise RuntimeError("batched t0 likelihood returned the wrong size")
        if (
            not values_include_prior
            and self.use_t0_prior
            and self.objective_mode != "charge_only"
            and self.t0_prior_sigma
        ):
            values_array = values_array + 0.5 * (
                shift_array / float(self.t0_prior_sigma)
            ) ** 2

        for index, key, value in zip(
            pending_indices, pending_keys, values_array, strict=True
        ):
            fval = float(value)
            if not math.isfinite(fval):
                self.invalid_evaluations += 1
                fval = np.inf
            self.cache[key] = fval
            out[index] = fval
        return out

    def __call__(self, values: Mapping[str, float]) -> float:
        self.calls += 1
        key = self._key(values)
        cached = self.cache.get(key)
        if cached is not None:
            return float(cached)
        self.evaluations += 1

        def invalid() -> float:
            self.invalid_evaluations += 1
            self.cache[key] = np.inf
            return np.inf

        vertex = np.asarray([values["x0"], values["y0"], values["z0"]], dtype=np.float64)
        direction = self.chart.direction(values.get("dir_u", 0.0), values.get("dir_v", 0.0))
        if direction is None or not self.detector.contains(vertex):
            return invalid()
        t0 = float(values.get("t0", 0.0))
        if not math.isfinite(t0):
            return invalid()
        if self.t0_limits is not None:
            lo, hi = (float(self.t0_limits[0]), float(self.t0_limits[1]))
            if t0 < lo or t0 > hi:
                return invalid()

        if self.track_end_mode == "absorption":
            visible = float(values["visible_length"])
            full_range = float(values["full_range"])
            if (
                not math.isfinite(visible)
                or not math.isfinite(full_range)
                or visible < 0.0
                or full_range <= 0.0
                or visible > full_range
                or self.range_lookup is None
            ):
                return invalid()
            max_range = float(self.range_lookup.overall_distances_mm[-1])
            if full_range > max_range:
                return invalid()
            ke0 = float(self.range_lookup.range_mm_to_energy(full_range))
            if not math.isfinite(ke0) or ke0 <= float(self.particle_threshold_mev):
                return invalid()
            if hasattr(self.emitter, "configure_track_end"):
                self.emitter.configure_track_end(
                    "abrupt", fixed_initial_KE=ke0, refresh=False
                )
            else:
                self.emitter.track_end_mode = "abrupt"
                self.emitter.fixed_initial_KE = ke0
            length = visible
        elif self.track_end_mode == "full_length" and self.clip_full_length_to_boundary:
            full_range = float(values["length"])
            if (
                not math.isfinite(full_range)
                or full_range <= 0.0
                or self.range_lookup is None
            ):
                return invalid()
            max_range = float(self.range_lookup.overall_distances_mm[-1])
            if full_range > max_range:
                return invalid()
            ke0 = float(self.range_lookup.range_mm_to_energy(full_range))
            if not math.isfinite(ke0) or ke0 <= float(self.particle_threshold_mev):
                return invalid()
            distance = float(
                self.detector.ray_exit_distance(
                    vertex,
                    direction,
                    tolerance_mm=float(self.containment_tolerance_mm),
                )
            )
            if not math.isfinite(distance) or distance <= 0.0:
                return invalid()
            inset = min(
                max(0.0, float(self.boundary_clip_inset_mm)),
                max(0.0, distance - 1.0e-6),
            )
            visible_to_boundary = float(distance - inset)
            length = float(min(full_range, visible_to_boundary))
            if not math.isfinite(length) or length <= 0.0:
                return invalid()
            if hasattr(self.emitter, "configure_track_end"):
                self.emitter.configure_track_end(
                    "abrupt", fixed_initial_KE=ke0, refresh=False
                )
            else:
                self.emitter.track_end_mode = "abrupt"
                self.emitter.fixed_initial_KE = ke0
        else:
            length = float(values["length"])
            if not math.isfinite(length) or length < 0.0:
                return invalid()
            if self.range_lookup is not None:
                if length > float(self.range_lookup.overall_distances_mm[-1]):
                    return invalid()
            if hasattr(self.emitter, "configure_track_end"):
                self.emitter.configure_track_end(
                    "threshold", fixed_initial_KE=None, refresh=False
                )
            else:
                self.emitter.track_end_mode = "threshold"
                self.emitter.fixed_initial_KE = None

        if self.require_contained_track and not self.detector.segment_contained(
            vertex,
            direction,
            length,
            tolerance_mm=float(self.containment_tolerance_mm),
        ):
            return invalid()

        geometry_key = self._geometry_key(values)
        if geometry_key not in self.prediction_cache:
            self.emitter.start_coord = tuple(float(x) for x in vertex)
            self.emitter.direction = tuple(float(x) for x in direction)
            # All source and photon times are additive in t0.  Evaluate the
            # expensive optical model once at zero and profile/optimize t0 by
            # shifting the returned timing prediction below.
            self.emitter.starting_time = 0.0
            init_ke = self.emitter.refresh_kinematics_from_length(length)
            if hasattr(self.emitter, "visible_length_is_physical"):
                if not self.emitter.visible_length_is_physical():
                    self.prediction_cache[geometry_key] = None
                    return invalid()
            elif getattr(self.emitter, "last_visible_length_exceeds_range", False):
                self.prediction_cache[geometry_key] = None
                return invalid()

            need_times = self.objective_mode != "charge_only"
            emission = self.emitter.get_emission_points(self.p_locations, init_ke)
            exp_pes, exp_ts_zero = self.emitter.get_expected_pes_ts(
                self.wcd,
                emission,
                self.p_locations,
                self.pmt_normals,
                self.mpmt_types,
                self.obs_pes,
                need_times=need_times,
            )
            self.optical_evaluations += 1
            exp_pes_array = np.asarray(exp_pes, dtype=np.float64)
            timing_pes = getattr(self.emitter, "_last_expected_pes_for_timing", None)
            timing_array = (
                None
                if timing_pes is None
                else np.asarray(timing_pes, dtype=np.float64)
            )
            if (
                exp_pes_array.shape != self.obs_pes.shape
                or np.any(~np.isfinite(exp_pes_array))
                or np.any(exp_pes_array < 0.0)
                or (
                    timing_array is not None
                    and (
                        timing_array.shape != self.obs_pes.shape
                        or np.any(~np.isfinite(timing_array))
                        or np.any(timing_array < 0.0)
                    )
                )
            ):
                self.prediction_cache[geometry_key] = None
                return invalid()
            prediction = (
                np.ascontiguousarray(exp_pes_array, dtype=np.float64),
                exp_ts_zero,
                None if timing_array is None else np.ascontiguousarray(
                    timing_array, dtype=np.float64
                ),
            )
            self.prediction_cache[geometry_key] = prediction
            self.event_mean_contamination_cache[geometry_key] = float(
                getattr(
                    self.emitter,
                    "_last_event_mean_contamination_fraction",
                    0.0,
                )
            )
        else:
            prediction = self.prediction_cache[geometry_key]
        if prediction is None:
            return invalid()

        fval = self._prediction_nll(
            prediction, t0, geometry_key=geometry_key
        )
        if not math.isfinite(fval):
            return invalid()
        self.cache[key] = fval
        return fval


@dataclass(frozen=True)
class _Block:
    name: str
    dimensions: tuple[str, ...]


def _free_vertex_basis(
    chart: TangentDirectionChart,
    fixed_params: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Return an orthonormal track-oriented basis in the free Cartesian subspace.

    With no fixed vertex coordinates this is exactly ``(d,e1,e2)``.  When one
    or two Cartesian coordinates are fixed, the old implementation merely
    zeroed components independently, leaving non-orthogonal and sometimes
    duplicate optimizer directions.  Projected Gram--Schmidt preserves the
    physical fixed coordinates while retaining the best available longitudinal
    and transverse directions.
    """
    free = np.asarray(
        [name not in fixed_params for name in ("x0", "y0", "z0")],
        dtype=np.float64,
    )
    names = ("longitudinal", "transverse_1", "transverse_2")
    candidates = (chart.anchor, chart.e1, chart.e2)
    basis: dict[str, np.ndarray] = {}
    accepted: list[np.ndarray] = []
    for name, candidate in zip(names, candidates):
        vector = np.asarray(candidate, dtype=np.float64) * free
        for previous in accepted:
            vector = vector - float(np.dot(vector, previous)) * previous
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-12:
            continue
        vector = np.ascontiguousarray(vector / norm, dtype=np.float64)
        basis[name] = vector
        accepted.append(vector)
    return basis


def _active_blocks(
    *,
    track_end_mode: str,
    objective_mode: str,
    fixed_params: Mapping[str, float],
    allow_longitudinal: bool = True,
    allow_transverse: bool = True,
    allow_direction: bool = True,
    vertex_basis: Mapping[str, np.ndarray] | None = None,
) -> list[_Block]:
    fixed = set(fixed_params)
    blocks: list[_Block] = []
    length_name = "visible_length" if track_end_mode == "absorption" else "length"
    longitudinal_dims: list[str] = []
    vertex_dims = (
        {"longitudinal", "transverse_1", "transverse_2"}
        if vertex_basis is None else set(vertex_basis)
    )
    if allow_longitudinal and "longitudinal" in vertex_dims:
        longitudinal_dims.append("longitudinal")
    if length_name not in fixed:
        longitudinal_dims.append(length_name)
    if longitudinal_dims:
        blocks.append(_Block("longitudinal_length", tuple(longitudinal_dims)))

    transverse_1: list[str] = []
    if allow_transverse and "transverse_1" in vertex_dims:
        transverse_1.append("transverse_1")
    if allow_direction and "direction" not in fixed:
        transverse_1.append("dir_u")
    if transverse_1:
        blocks.append(_Block("transverse_1_direction", tuple(transverse_1)))

    transverse_2: list[str] = []
    if allow_transverse and "transverse_2" in vertex_dims:
        transverse_2.append("transverse_2")
    if allow_direction and "direction" not in fixed:
        transverse_2.append("dir_v")
    if transverse_2:
        blocks.append(_Block("transverse_2_direction", tuple(transverse_2)))

    if track_end_mode == "absorption" and "full_range" not in fixed:
        blocks.append(_Block("full_range", ("full_range",)))
    if objective_mode != "charge_only" and "t0" not in fixed:
        blocks.append(_Block("t0", ("t0",)))
    return blocks


def _step_for_dimension(dimension: str, steps: OptimizerSteps) -> float:
    return {
        "longitudinal": steps.longitudinal_mm,
        "transverse_1": steps.transverse_mm,
        "transverse_2": steps.transverse_mm,
        "dir_u": steps.direction_tangent,
        "dir_v": steps.direction_tangent,
        "length": steps.length_mm,
        "visible_length": steps.length_mm,
        "full_range": steps.full_range_mm,
        "t0": steps.t0_ns,
    }[dimension]


def _min_step_for_dimension(dimension: str, steps: OptimizerMinSteps) -> float:
    return {
        "longitudinal": steps.longitudinal_mm,
        "transverse_1": steps.transverse_mm,
        "transverse_2": steps.transverse_mm,
        "dir_u": steps.direction_tangent,
        "dir_v": steps.direction_tangent,
        "length": steps.length_mm,
        "visible_length": steps.length_mm,
        "full_range": steps.full_range_mm,
        "t0": steps.t0_ns,
    }[dimension]


def _apply_dimension_updates(
    values: Mapping[str, float],
    updates: Mapping[str, float],
    *,
    chart: TangentDirectionChart,
    detector: ConvexDetectorVolume,
    vertex_basis: Mapping[str, np.ndarray],
    fixed_params: Mapping[str, float],
    length_limits: tuple[float, float],
    full_range_limits: tuple[float, float] | None,
    track_end_mode: str,
    project_vertex_steps: bool,
) -> dict[str, float]:
    out = {k: float(v) for k, v in values.items()}
    vertex = np.asarray([out["x0"], out["y0"], out["z0"]], dtype=np.float64)
    vertex_delta = np.zeros(3, dtype=np.float64)
    for dimension in ("longitudinal", "transverse_1", "transverse_2"):
        if dimension in updates and dimension in vertex_basis:
            vertex_delta += float(updates[dimension]) * vertex_basis[dimension]
    if np.any(vertex_delta):
        vertex = (
            detector.project_step(vertex, vertex_delta)
            if project_vertex_steps else vertex + vertex_delta
        )
    for i, name in enumerate(("x0", "y0", "z0")):
        out[name] = float(fixed_params.get(name, vertex[i]))

    for name in ("dir_u", "dir_v", "t0"):
        if name in updates:
            out[name] = float(out.get(name, 0.0) + float(updates[name]))
    if "direction" in fixed_params:
        out["dir_u"] = 0.0
        out["dir_v"] = 0.0

    length_name = "visible_length" if track_end_mode == "absorption" else "length"
    if length_name in updates:
        out[length_name] = float(np.clip(
            out[length_name] + float(updates[length_name]), length_limits[0], length_limits[1]
        ))
    if length_name in fixed_params:
        out[length_name] = float(fixed_params[length_name])

    if track_end_mode == "absorption":
        assert full_range_limits is not None
        if "full_range" in updates:
            out["full_range"] = float(np.clip(
                out["full_range"] + float(updates["full_range"]),
                full_range_limits[0], full_range_limits[1],
            ))
        if "full_range" in fixed_params:
            out["full_range"] = float(fixed_params["full_range"])
        if out["visible_length"] > out["full_range"]:
            # Preserve the proposed visible length and move the range outward if
            # it is free; otherwise shorten the visible part.
            if "full_range" not in fixed_params:
                out["full_range"] = min(full_range_limits[1], out["visible_length"])
            if out["visible_length"] > out["full_range"]:
                out["visible_length"] = out["full_range"]
    return out


def track_aligned_block_optimize(
    objective: TrackObjective,
    start_values: Mapping[str, float],
    *,
    fixed_params: Mapping[str, float] | None = None,
    sweeps: int = 3,
    initial_steps: OptimizerSteps = OptimizerSteps(),
    min_steps: OptimizerMinSteps = OptimizerMinSteps(),
    length_limits: tuple[float, float] = (0.0, 3000.0),
    full_range_limits: tuple[float, float] | None = None,
    min_improvement: float = 1.0e-3,
    full_cross_sweeps: int = 1,
    allow_longitudinal: bool = True,
    allow_transverse: bool = True,
    allow_direction: bool = True,
    project_vertex_steps: bool = False,
) -> BlockOptimizerResult:
    """Minimize an optical objective in track-aligned one/two-dimensional blocks.

    Every accepted point is evaluated with the supplied objective.  A quadratic
    proposal is made only when every required stencil value, gradient and Hessian
    element is finite.  Otherwise the best finite sampled point is retained and
    the trust radius is reduced.  Non-finite matrix multiplication is therefore
    impossible by construction.
    """
    wall0 = time.perf_counter()
    fixed = {} if fixed_params is None else {k: float(v) for k, v in fixed_params.items()}
    values = {k: float(v) for k, v in start_values.items()}
    values.setdefault("dir_u", 0.0)
    values.setdefault("dir_v", 0.0)
    values.setdefault("t0", 0.0)
    for name, value in fixed.items():
        if name in values:
            values[name] = value
    if not objective.detector.contains([values["x0"], values["y0"], values["z0"]]):
        raise ValueError("block optimizer start vertex lies outside detector volume")
    vertex_basis = _free_vertex_basis(objective.chart, fixed)

    nfcn_start = objective.evaluations
    invalid_start = objective.invalid_evaluations
    fval = float(objective(values))
    if not math.isfinite(fval):
        raise RuntimeError(f"non-finite block-optimizer start FCN: {values}")
    cache: dict[tuple[float, ...], tuple[float, dict[str, float]]] = {}
    dimensions = {
        dim
        for block in _active_blocks(
            track_end_mode=objective.track_end_mode,
            objective_mode=objective.objective_mode,
            fixed_params=fixed,
            allow_longitudinal=allow_longitudinal,
            allow_transverse=allow_transverse,
            allow_direction=allow_direction,
            vertex_basis=vertex_basis,
        )
        for dim in block.dimensions
    }
    steps = {dim: _step_for_dimension(dim, initial_steps) for dim in dimensions}
    errors: dict[str, float] = {
        "x0": np.nan, "y0": np.nan, "z0": np.nan,
        "dir_u": np.nan, "dir_v": np.nan,
        "length": np.nan, "visible_length": np.nan,
        "full_range": np.nan, "t0": np.nan,
    }
    history: list[dict[str, object]] = []
    quadratic_skips = 0

    key_names = [
        "x0", "y0", "z0", "dir_u", "dir_v",
        "visible_length" if objective.track_end_mode == "absorption" else "length",
    ]
    if objective.track_end_mode == "absorption":
        key_names.append("full_range")
    key_names.append("t0")
    key_names = tuple(key_names)

    def evaluate_updates(current: Mapping[str, float], updates: Mapping[str, float]):
        trial = _apply_dimension_updates(
            current,
            updates,
            chart=objective.chart,
            detector=objective.detector,
            vertex_basis=vertex_basis,
            fixed_params=fixed,
            length_limits=length_limits,
            full_range_limits=full_range_limits,
            track_end_mode=objective.track_end_mode,
            project_vertex_steps=bool(project_vertex_steps),
        )
        key = tuple(round(float(trial.get(name, 0.0)), 12) for name in key_names)
        cached = cache.get(key)
        if cached is not None:
            return cached
        value = float(objective(trial))
        result = (value, trial)
        cache[key] = result
        return result

    def evaluate_t0_updates_many(
        current: Mapping[str, float], deltas: Sequence[float]
    ) -> list[tuple[float, dict[str, float]]]:
        """Evaluate the unchanged fixed-geometry t0 stencil in one kernel."""
        rows: list[tuple[float, dict[str, float]] | None] = []
        pending_indices: list[int] = []
        pending_trials: list[dict[str, float]] = []
        pending_keys: list[tuple[float, ...]] = []
        for delta in deltas:
            trial = _apply_dimension_updates(
                current,
                {"t0": float(delta)},
                chart=objective.chart,
                detector=objective.detector,
                vertex_basis=vertex_basis,
                fixed_params=fixed,
                length_limits=length_limits,
                full_range_limits=full_range_limits,
                track_end_mode=objective.track_end_mode,
                project_vertex_steps=bool(project_vertex_steps),
            )
            key = tuple(
                round(float(trial.get(name, 0.0)), 12)
                for name in key_names
            )
            cached = cache.get(key)
            if cached is not None:
                rows.append(cached)
                continue
            pending_indices.append(len(rows))
            pending_trials.append(trial)
            pending_keys.append(key)
            rows.append(None)

        if pending_trials:
            values_many = objective.evaluate_t0_many(
                current,
                np.asarray(
                    [trial["t0"] for trial in pending_trials],
                    dtype=np.float64,
                ),
            )
            for row_index, trial, key, value in zip(
                pending_indices,
                pending_trials,
                pending_keys,
                np.asarray(values_many, dtype=np.float64),
                strict=True,
            ):
                result = (float(value), trial)
                cache[key] = result
                rows[row_index] = result

        return [row for row in rows if row is not None]

    for sweep in range(max(1, int(sweeps))):
        sweep_start = fval
        blocks = _active_blocks(
            track_end_mode=objective.track_end_mode,
            objective_mode=objective.objective_mode,
            fixed_params=fixed,
            allow_longitudinal=allow_longitudinal,
            allow_transverse=allow_transverse,
            allow_direction=allow_direction,
            vertex_basis=vertex_basis,
        )
        for block in blocks:
            dims = block.dimensions
            if len(dims) == 1:
                dim = dims[0]
                h = float(steps[dim])
                candidates: list[tuple[float, dict[str, float], float]] = [(fval, dict(values), 0.0)]
                if dim == "t0" and _BATCH_T0_BLOCK_STENCIL:
                    signed_trials = zip(
                        (-1.0, 1.0),
                        evaluate_t0_updates_many(values, (-h, h)),
                        strict=True,
                    )
                else:
                    signed_trials = (
                        (
                            sign,
                            evaluate_updates(values, {dim: sign * h}),
                        )
                        for sign in (-1.0, 1.0)
                    )
                for sign, (fv, trial) in signed_trials:
                    candidates.append((fv, trial, sign * h))
                fm, fp = candidates[1][0], candidates[2][0]
                if math.isfinite(fm) and math.isfinite(fp):
                    curvature = fm - 2.0 * fval + fp
                    if math.isfinite(curvature) and curvature > 1.0e-10:
                        delta = float(np.clip(0.5 * h * (fm - fp) / curvature, -2.0 * h, 2.0 * h))
                        fc, trial = evaluate_updates(values, {dim: delta})
                        candidates.append((fc, trial, delta))
                        errors[dim] = float(h / math.sqrt(max(curvature, 1.0e-30)))
                finite = [candidate for candidate in candidates if math.isfinite(candidate[0])]
                best = min(finite, key=lambda item: item[0]) if finite else candidates[0]
                if best[0] < fval - 1.0e-10:
                    move = abs(float(best[2]))
                    fval = float(best[0])
                    values = dict(best[1])
                    factor = 1.20 if move > 0.7 * h else 0.70
                    steps[dim] = max(_min_step_for_dimension(dim, min_steps), h * factor)
                else:
                    steps[dim] = max(_min_step_for_dimension(dim, min_steps), 0.5 * h)
                continue

            if len(dims) != 2:
                # This implementation intentionally keeps blocks one- or
                # two-dimensional so the exact stencil call count stays small.
                raise RuntimeError(f"unsupported block dimensions: {dims}")
            dim1, dim2 = dims
            h1 = float(steps[dim1])
            h2 = float(steps[dim2])
            points = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
            full_cross = sweep < int(full_cross_sweeps)
            if full_cross:
                points.extend([(-1, 1), (1, -1)])
            samples: dict[tuple[int, int], tuple[float, dict[str, float]]] = {}
            candidates_2d: list[tuple[float, dict[str, float]]] = [(fval, dict(values))]
            for a, b in points:
                fv, trial = evaluate_updates(values, {dim1: a * h1, dim2: b * h2})
                samples[(a, b)] = (fv, trial)
                candidates_2d.append((fv, trial))

            required_keys = list(points)
            required_finite = all(math.isfinite(samples[key][0]) for key in required_keys)
            if required_finite:
                fm1 = samples[(-1, 0)][0]
                fp1 = samples[(1, 0)][0]
                fm2 = samples[(0, -1)][0]
                fp2 = samples[(0, 1)][0]
                gradient = np.asarray(
                    [(fp1 - fm1) / (2.0 * h1), (fp2 - fm2) / (2.0 * h2)],
                    dtype=np.float64,
                )
                h11 = (fp1 - 2.0 * fval + fm1) / (h1 * h1)
                h22 = (fp2 - 2.0 * fval + fm2) / (h2 * h2)
                if full_cross:
                    cross = (
                        samples[(1, 1)][0] - samples[(1, -1)][0]
                        - samples[(-1, 1)][0] + samples[(-1, -1)][0]
                    ) / (4.0 * h1 * h2)
                else:
                    cross_scaled = (
                        0.5 * (samples[(1, 1)][0] + samples[(-1, -1)][0])
                        - fval - 0.5 * (h11 * h1 * h1 + h22 * h2 * h2)
                    )
                    cross = cross_scaled / (h1 * h2)
                hessian = np.asarray([[h11, cross], [cross, h22]], dtype=np.float64)
                if np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian)):
                    scale = np.diag([h1, h2])
                    scaled_hessian = scale @ hessian @ scale
                    scaled_gradient = scale @ gradient
                    if np.all(np.isfinite(scaled_hessian)) and np.all(np.isfinite(scaled_gradient)):
                        try:
                            eig, vec = np.linalg.eigh(0.5 * (scaled_hessian + scaled_hessian.T))
                            if np.all(np.isfinite(eig)) and np.all(np.isfinite(vec)):
                                floor = max(1.0e-3, 0.05 * float(np.max(np.abs(eig))))
                                regularized = np.maximum(eig, floor)
                                delta_scaled = -vec @ ((vec.T @ scaled_gradient) / regularized)
                                delta = scale @ delta_scaled
                                delta = np.clip(delta, [-2.0 * h1, -2.0 * h2], [2.0 * h1, 2.0 * h2])
                                fc, trial = evaluate_updates(
                                    values, {dim1: float(delta[0]), dim2: float(delta[1])}
                                )
                                candidates_2d.append((fc, trial))
                                inverse = scale @ vec @ np.diag(1.0 / regularized) @ vec.T @ scale
                                errors[dim1] = float(math.sqrt(max(inverse[0, 0], 0.0)))
                                errors[dim2] = float(math.sqrt(max(inverse[1, 1], 0.0)))
                            else:
                                quadratic_skips += 1
                        except np.linalg.LinAlgError:
                            quadratic_skips += 1
                    else:
                        quadratic_skips += 1
                else:
                    quadratic_skips += 1
            else:
                quadratic_skips += 1

            finite = [candidate for candidate in candidates_2d if math.isfinite(candidate[0])]
            best = min(finite, key=lambda item: item[0]) if finite else (fval, dict(values))
            old_values = dict(values)
            if best[0] < fval - 1.0e-10:
                fval = float(best[0])
                values = dict(best[1])
                # Infer movement in the local block coordinates from the state.
                if dim1.startswith("dir_"):
                    move1 = abs(values[dim1] - old_values[dim1])
                elif dim1 in {"length", "visible_length", "full_range", "t0"}:
                    move1 = abs(values[dim1] - old_values[dim1])
                else:
                    dv = np.asarray(
                        [values["x0"] - old_values["x0"], values["y0"] - old_values["y0"], values["z0"] - old_values["z0"]]
                    )
                    basis = vertex_basis[dim1]
                    move1 = abs(float(np.dot(dv, basis)))
                if dim2.startswith("dir_"):
                    move2 = abs(values[dim2] - old_values[dim2])
                elif dim2 in {"length", "visible_length", "full_range", "t0"}:
                    move2 = abs(values[dim2] - old_values[dim2])
                else:
                    dv = np.asarray(
                        [values["x0"] - old_values["x0"], values["y0"] - old_values["y0"], values["z0"] - old_values["z0"]]
                    )
                    basis = vertex_basis[dim2]
                    move2 = abs(float(np.dot(dv, basis)))
                for dim, h, move in ((dim1, h1, move1), (dim2, h2, move2)):
                    factor = 1.20 if move > 0.7 * h else 0.70
                    steps[dim] = max(_min_step_for_dimension(dim, min_steps), h * factor)
            else:
                steps[dim1] = max(_min_step_for_dimension(dim1, min_steps), 0.5 * h1)
                steps[dim2] = max(_min_step_for_dimension(dim2, min_steps), 0.5 * h2)

        history.append(
            {
                "sweep": int(sweep),
                "sweep_start_fval": float(sweep_start),
                "sweep_gain": float(sweep_start - fval),
                "fval": float(fval),
                "nfcn": int(objective.evaluations - nfcn_start),
                "invalid_evaluations": int(objective.invalid_evaluations - invalid_start),
                "values": dict(values),
                "steps": dict(steps),
            }
        )
        if sweep >= 1 and sweep_start - fval < float(min_improvement):
            break

    # Convert diagonal track-aligned curvature estimates into Cartesian
    # summaries.  These remain optimizer diagnostics rather than a replacement
    # for HESSE/FE covariance, but unlike the old x-only placeholder they are
    # rotationally meaningful.
    finite_vertex_modes = [
        name for name in ("longitudinal", "transverse_1", "transverse_2")
        if name in vertex_basis and np.isfinite(errors.get(name, np.nan))
    ]
    if finite_vertex_modes:
        basis = np.column_stack([vertex_basis[name] for name in finite_vertex_modes])
        variances = np.asarray([errors[name] ** 2 for name in finite_vertex_modes])
        vertex_covariance = basis @ np.diag(variances) @ basis.T
        for i, name in enumerate(("x0", "y0", "z0")):
            errors[name] = float(math.sqrt(max(vertex_covariance[i, i], 0.0)))
    direction_local_errors = np.asarray(
        [errors.get("dir_u", np.nan), errors.get("dir_v", np.nan)], dtype=np.float64
    )
    if np.all(np.isfinite(direction_local_errors)):
        jacobian = objective.chart.direction_jacobian(
            float(values.get("dir_u", 0.0)), float(values.get("dir_v", 0.0))
        )
        direction_covariance = jacobian @ np.diag(direction_local_errors ** 2) @ jacobian.T
        for i, name in enumerate(("cx", "cy", "cz")):
            errors[name] = float(math.sqrt(max(direction_covariance[i, i], 0.0)))
    return BlockOptimizerResult(
        values=dict(values),
        fval=float(fval),
        errors=errors,
        nfcn=int(objective.evaluations - nfcn_start),
        history=history,
        chart=objective.chart,
        wall_s=float(time.perf_counter() - wall0),
        invalid_evaluations=int(objective.invalid_evaluations - invalid_start),
        quadratic_skips=int(quadratic_skips),
    )


def cobyqa_exact_continuation(
    objective: TrackObjective,
    start_values: Mapping[str, float],
    *,
    fixed_params: Mapping[str, float] | None = None,
    initial_steps: OptimizerSteps = OptimizerSteps(),
    length_limits: tuple[float, float] = (0.0, 3000.0),
    full_range_limits: tuple[float, float] | None = None,
    allow_longitudinal: bool = True,
    allow_transverse: bool = True,
    allow_direction: bool = True,
    project_vertex_steps: bool = False,
    max_evaluations: int = 240,
    initial_trust_radius: float = 1.0,
    final_trust_radius: float = 5.0e-3,
    poll_radii: Sequence[float] = (1.0e-2, 3.0e-3),
    poll_tolerance: float = 1.0e-3,
    max_restarts: int = 2,
    initial_errors: Mapping[str, float] | None = None,
) -> tuple[BlockOptimizerResult, dict[str, object]]:
    """Derivative-free exact continuation in the existing physical fit chart.

    Active coordinates, fixed-parameter handling, detector projection, endpoint
    inequalities, and track topology are exactly those of the block optimizer.
    Standardized COBYQA coordinates merely scale each active physical direction
    by the carried trust radius; they are not an additional physical model.
    """

    wall0 = time.perf_counter()
    fixed = {} if fixed_params is None else {
        str(name): float(value) for name, value in fixed_params.items()
    }
    base = {str(name): float(value) for name, value in start_values.items()}
    base.setdefault("dir_u", 0.0)
    base.setdefault("dir_v", 0.0)
    base.setdefault("t0", 0.0)
    for name, value in fixed.items():
        if name in base:
            base[name] = value
    if not objective.detector.contains([base["x0"], base["y0"], base["z0"]]):
        raise ValueError("COBYQA continuation start vertex lies outside detector volume")
    if objective.track_end_mode == "absorption" and full_range_limits is None:
        raise ValueError("absorption continuation requires full_range_limits")

    vertex_basis = _free_vertex_basis(objective.chart, fixed)
    blocks = _active_blocks(
        track_end_mode=objective.track_end_mode,
        objective_mode=objective.objective_mode,
        fixed_params=fixed,
        allow_longitudinal=bool(allow_longitudinal),
        allow_transverse=bool(allow_transverse),
        allow_direction=bool(allow_direction),
        vertex_basis=vertex_basis,
    )
    dimensions = tuple(dict.fromkeys(
        dimension for block in blocks for dimension in block.dimensions
    ))
    if not dimensions:
        raise ValueError("COBYQA continuation has no active fit dimensions")
    scales = np.asarray(
        [_step_for_dimension(name, initial_steps) for name in dimensions],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("COBYQA continuation scales must be positive and finite")

    nfcn_start = int(objective.evaluations)
    invalid_start = int(objective.invalid_evaluations)
    initial_fval = float(objective(base))
    if not math.isfinite(initial_fval):
        raise RuntimeError("non-finite COBYQA continuation start FCN")
    invalid_penalty = max(1.0e6, abs(initial_fval) + 1.0e6)
    cache: dict[tuple[float, ...], tuple[float, float | None, dict[str, float]]] = {}
    history: list[dict[str, object]] = []

    def values_from_scaled(scaled) -> dict[str, float]:
        x = np.asarray(scaled, dtype=np.float64).reshape(len(dimensions))
        updates = {
            name: float(x[index] * scales[index])
            for index, name in enumerate(dimensions)
        }
        return _apply_dimension_updates(
            base,
            updates,
            chart=objective.chart,
            detector=objective.detector,
            vertex_basis=vertex_basis,
            fixed_params=fixed,
            length_limits=length_limits,
            full_range_limits=full_range_limits,
            track_end_mode=objective.track_end_mode,
            project_vertex_steps=bool(project_vertex_steps),
        )

    def evaluate_scaled(scaled) -> float:
        x = np.asarray(scaled, dtype=np.float64).reshape(len(dimensions))
        key = tuple(np.round(x, 12))
        cached = cache.get(key)
        if cached is not None:
            return float(cached[0])
        candidate = values_from_scaled(x)
        exact = float(objective(candidate))
        physical = math.isfinite(exact)
        value = exact if physical else invalid_penalty * (
            1.0 + 1.0e-6 * float(x @ x)
        )
        cache[key] = (float(value), exact if physical else None, candidate)
        history.append({
            "scaled_coordinates": np.ascontiguousarray(x.copy()),
            "values": dict(candidate),
            "objective_value": float(value),
            "exact_fval": float(exact) if physical else None,
            "physical": bool(physical),
        })
        return float(value)

    def best_physical():
        physical = [entry for entry in history if entry["physical"]]
        if not physical:
            raise RuntimeError("COBYQA continuation found no physical exact point")
        return min(physical, key=lambda entry: float(entry["exact_fval"]))

    def poll(center_scaled):
        center = np.asarray(center_scaled, dtype=np.float64)
        center_value = float(evaluate_scaled(center))
        best_value = center_value
        best_scaled = center.copy()
        rows = []
        maximum_downhill = 0.0
        for radius in tuple(float(value) for value in poll_radii):
            radius_downhill = 0.0
            physical_trials = 0
            for index in range(len(dimensions)):
                for sign in (-1.0, 1.0):
                    trial = center.copy()
                    trial[index] += sign * radius
                    value = float(evaluate_scaled(trial))
                    row = cache[tuple(np.round(trial, 12))]
                    if row[1] is None:
                        continue
                    physical_trials += 1
                    downhill = center_value - float(row[1])
                    radius_downhill = max(radius_downhill, downhill)
                    if float(row[1]) < best_value:
                        best_value = float(row[1])
                        best_scaled = trial.copy()
            maximum_downhill = max(maximum_downhill, radius_downhill)
            rows.append({
                "radius_scaled": radius,
                "max_downhill_nll": float(radius_downhill),
                "physical_trials": int(physical_trials),
            })
        return {
            "base_value": center_value,
            "best_value": best_value,
            "best_scaled": np.ascontiguousarray(best_scaled),
            "max_downhill_nll": float(maximum_downhill),
            "rows": tuple(rows),
        }

    def run(start, initial_radius):
        return minimize(
            evaluate_scaled,
            np.asarray(start, dtype=np.float64),
            method="COBYQA",
            options={
                "maxfev": max(1, int(max_evaluations)),
                "initial_tr_radius": float(initial_radius),
                "final_tr_radius": float(final_trust_radius),
                "scale": False,
                "disp": False,
            },
        )

    attempts = []
    result = run(np.zeros(len(dimensions), dtype=np.float64), initial_trust_radius)
    attempts.append(result)
    best = best_physical()
    best_scaled = np.asarray(best["scaled_coordinates"], dtype=np.float64)
    certificate = poll(best_scaled)
    restart_count = 0
    while (
        float(certificate["max_downhill_nll"]) > float(poll_tolerance)
        and restart_count < max(0, int(max_restarts))
    ):
        restart_start = np.asarray(certificate["best_scaled"], dtype=np.float64)
        result = run(
            restart_start,
            max(10.0 * float(final_trust_radius), 5.0e-2 / (2.0 ** restart_count)),
        )
        attempts.append(result)
        restart_count += 1
        best = best_physical()
        best_scaled = np.asarray(best["scaled_coordinates"], dtype=np.float64)
        certificate = poll(best_scaled)

    best = best_physical()
    final_values = dict(best["values"])
    final_fval = float(best["exact_fval"])
    errors = {
        "x0": np.nan, "y0": np.nan, "z0": np.nan,
        "dir_u": np.nan, "dir_v": np.nan,
        "length": np.nan, "visible_length": np.nan,
        "full_range": np.nan, "t0": np.nan,
    }
    if initial_errors is not None:
        for name, value in initial_errors.items():
            errors[str(name)] = float(value)
    backend_diagnostics = {
        "backend": "cobyqa_exact_physical_chart",
        "active_dimensions": dimensions,
        "scales": tuple(float(value) for value in scales),
        "initial_fval": initial_fval,
        "final_fval": final_fval,
        "gain_nll": float(initial_fval - final_fval),
        "nfev": int(len(cache)),
        "restart_count": int(restart_count),
        "max_restarts": int(max_restarts),
        "max_evaluations_per_attempt": int(max_evaluations),
        "converged": bool(
            float(certificate["max_downhill_nll"]) <= float(poll_tolerance)
        ),
        "max_poll_downhill_nll": float(certificate["max_downhill_nll"]),
        "poll_tolerance_nll": float(poll_tolerance),
        "poll": tuple(certificate["rows"]),
        "optimizer_attempts": tuple({
            "success": bool(attempt.success),
            "message": str(attempt.message),
            "nfev": int(getattr(attempt, "nfev", 0)),
            "nit": int(getattr(attempt, "nit", -1)),
        } for attempt in attempts),
    }
    result_history = [{
        "sweep": 0,
        "sweep_start_fval": initial_fval,
        "sweep_gain": float(initial_fval - final_fval),
        "fval": final_fval,
        "nfcn": int(objective.evaluations - nfcn_start),
        "invalid_evaluations": int(objective.invalid_evaluations - invalid_start),
        "values": dict(final_values),
        "steps": {
            name: float(scales[index]) for index, name in enumerate(dimensions)
        },
        "adaptive_backend": "cobyqa_exact_physical_chart",
        "backend_diagnostics": dict(backend_diagnostics),
    }]
    return BlockOptimizerResult(
        values=final_values,
        fval=final_fval,
        errors=errors,
        nfcn=int(objective.evaluations - nfcn_start),
        history=result_history,
        chart=objective.chart,
        wall_s=float(time.perf_counter() - wall0),
        invalid_evaluations=int(objective.invalid_evaluations - invalid_start),
        quadratic_skips=0,
    ), backend_diagnostics


def profile_t0(
    objective,
    values: Mapping[str, float],
    *,
    limits: tuple[float, float],
    coarse_step_ns: float = 0.25,
    refine_levels: int = 2,
    refine_factor: float = 5.0,
) -> T0ProfileResult:
    """Globally profile the additive event-time offset at negligible optical cost.

    :class:`TrackObjective` caches the expensive optical prediction without
    ``t0``.  Consequently a dense global t0 scan invokes only the compiled PMT
    likelihood after the first point.  This is both faster and substantially
    more reliable than estimating t0 from a nominal mean-arrival array when the
    production objective is a conditional first-photoelectron mixture.
    """
    wall0 = time.perf_counter()
    lo, hi = float(limits[0]), float(limits[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and hi >= lo):
        raise ValueError("t0 profile limits must be finite and ordered")
    step = float(coarse_step_ns)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("coarse_step_ns must be finite and positive")
    factor = float(refine_factor)
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError("refine_factor must be greater than one")

    start_eval = int(getattr(objective, "evaluations", 0))
    start_optical = int(getattr(objective, "optical_evaluations", 0))
    base = {k: float(v) for k, v in values.items()}
    samples: dict[float, float] = {}

    def evaluate(t0: float) -> float:
        t = float(np.clip(t0, lo, hi))
        key = round(t, 12)
        if key in samples:
            return samples[key]
        trial = dict(base)
        trial["t0"] = t
        value = float(objective(trial))
        samples[key] = value
        return value

    def evaluate_many(points) -> None:
        requested: list[float] = []
        requested_keys: list[float] = []
        for point in points:
            t = float(np.clip(float(point), lo, hi))
            key = round(t, 12)
            if key in samples or key in requested_keys:
                continue
            requested.append(t)
            requested_keys.append(key)
        if not requested:
            return
        if hasattr(objective, "evaluate_t0_many"):
            values_array = np.asarray(
                objective.evaluate_t0_many(base, requested), dtype=np.float64
            ).reshape(-1)
            if values_array.size != len(requested):
                raise RuntimeError("batched t0 objective returned the wrong size")
            for key, value in zip(requested_keys, values_array, strict=True):
                samples[key] = float(value)
        else:
            for point in requested:
                evaluate(point)

    if hi == lo:
        evaluate(lo)
    else:
        n = max(1, int(math.ceil((hi - lo) / step)))
        grid = np.linspace(lo, hi, n + 1, dtype=np.float64)
        evaluate_many(grid)

        for _ in range(max(0, int(refine_levels))):
            finite = [(t, f) for t, f in samples.items() if math.isfinite(f)]
            if not finite:
                break
            best_t, _ = min(finite, key=lambda item: item[1])
            new_step = step / factor
            local_lo = max(lo, best_t - step)
            local_hi = min(hi, best_t + step)
            n_local = max(2, int(math.ceil((local_hi - local_lo) / new_step)))
            evaluate_many(np.linspace(local_lo, local_hi, n_local + 1))
            step = new_step

    finite = [(t, f) for t, f in samples.items() if math.isfinite(f)]
    if not finite:
        raise RuntimeError("all t0 profile points were non-finite")
    best_t, best_f = min(finite, key=lambda item: item[1])

    # Curvature diagnostic from the closest available point on each side.
    left = max((item for item in finite if item[0] < best_t), default=None, key=lambda x: x[0])
    right = min((item for item in finite if item[0] > best_t), default=None, key=lambda x: x[0])
    error = math.nan
    if left is not None and right is not None:
        h_left = best_t - left[0]
        h_right = right[0] - best_t
        if h_left > 0.0 and h_right > 0.0:
            # Nonuniform three-point second derivative.
            curvature = 2.0 * (
                left[1] / (h_left * (h_left + h_right))
                - best_f / (h_left * h_right)
                + right[1] / (h_right * (h_left + h_right))
            )
            if math.isfinite(curvature) and curvature > 0.0:
                error = 1.0 / math.sqrt(curvature)

    out = dict(base)
    out["t0"] = float(best_t)
    return T0ProfileResult(
        values=out,
        fval=float(best_f),
        error_ns=float(error),
        nll_evaluations=int(getattr(objective, "evaluations", 0)) - start_eval,
        optical_evaluations=int(getattr(objective, "optical_evaluations", 0)) - start_optical,
        sampled_points=tuple(sorted((float(t), float(f)) for t, f in samples.items())),
        wall_s=float(time.perf_counter() - wall0),
    )


def seed_values_from_mapping(seed: Mapping[str, object], *, track_end_mode: str) -> tuple[dict[str, float], TangentDirectionChart]:
    direction = direction_from_mapping(seed)
    chart = TangentDirectionChart.from_direction(direction)
    values = {
        "x0": float(seed["x0"]),
        "y0": float(seed["y0"]),
        "z0": float(seed["z0"]),
        "dir_u": 0.0,
        "dir_v": 0.0,
        "t0": float(seed.get("t0", 0.0)),
    }
    if track_end_mode == "absorption":
        values["visible_length"] = float(seed["visible_length"])
        values["full_range"] = float(seed["full_range"])
    else:
        values["length"] = float(seed["length"])
    return values, chart


def result_to_seed(result: BlockOptimizerResult, *, track_end_mode: str) -> dict[str, float]:
    values = result.output_values()
    out = {
        "x0": float(values["x0"]),
        "y0": float(values["y0"]),
        "z0": float(values["z0"]),
        "cx": float(values["cx"]),
        "cy": float(values["cy"]),
        "cz": float(values["cz"]),
        "dir_x": float(values["cx"]),
        "dir_y": float(values["cy"]),
        "dir_z": float(values["cz"]),
        "t0": float(values.get("t0", 0.0)),
    }
    if track_end_mode == "absorption":
        out["visible_length"] = float(values["visible_length"])
        out["full_range"] = float(values["full_range"])
    else:
        out["length"] = float(values["length"])
    return out


def reanchor_result(result: BlockOptimizerResult) -> tuple[dict[str, float], TangentDirectionChart]:
    return reanchor_values(result.values, result.chart)

# -----------------------------------------------------------------------------
# Compact global seed proxy library
# -----------------------------------------------------------------------------
import hashlib
import json
import os
from pathlib import Path

from numba import njit

_PROXY_SCHEMA_VERSION = 3
_PROXY_LOG_MIN = -20.0
_PROXY_LOG_MAX = 8.0


@njit(cache=True, fastmath=True)
def _score_quantized_log_shapes_numba(
    codes: np.ndarray,
    hit_indices: np.ndarray,
    hit_charge: np.ndarray,
    log_offset: float,
    log_scale: float,
    log_floor: np.ndarray,
) -> np.ndarray:
    nseed = codes.shape[0]
    nhit = hit_indices.size
    out = np.empty(nseed, dtype=np.float64)
    for i in range(nseed):
        score = 0.0
        for j in range(nhit):
            value = log_offset + log_scale * float(codes[i, hit_indices[j]])
            floor_value = float(log_floor[hit_indices[j]])
            if value < floor_value:
                value = floor_value
            score -= float(hit_charge[j]) * value
        out[i] = score
    return out


def _quantization_range(bits: int) -> tuple[int, int, np.dtype]:
    bits = int(bits)
    if bits == 8:
        return -127, 127, np.dtype(np.int8)
    if bits == 16:
        return -32767, 32767, np.dtype(np.int16)
    raise ValueError("proxy quantization_bits must be 8 or 16")


def _quantize_log_shapes(
    log_shapes: np.ndarray,
    *,
    bits: int = 8,
) -> tuple[np.ndarray, float, float]:
    values = np.clip(np.asarray(log_shapes, dtype=np.float64), _PROXY_LOG_MIN, _PROXY_LOG_MAX)
    code_min, code_max, dtype = _quantization_range(bits)
    offset = 0.5 * (_PROXY_LOG_MIN + _PROXY_LOG_MAX)
    scale = (_PROXY_LOG_MAX - _PROXY_LOG_MIN) / float(code_max - code_min)
    codes = np.rint((values - offset) / scale)
    codes = np.clip(codes, code_min, code_max).astype(dtype)
    return np.ascontiguousarray(codes), float(offset), float(scale)


def _seed_matrix(seeds: Sequence[Mapping[str, object]], *, track_end_mode: str) -> np.ndarray:
    rows = []
    for seed in seeds:
        direction = direction_from_mapping(seed)
        row = [
            float(seed["x0"]), float(seed["y0"]), float(seed["z0"]),
            float(direction[0]), float(direction[1]), float(direction[2]),
        ]
        if track_end_mode == "absorption":
            row.extend([float(seed["visible_length"]), float(seed["full_range"])])
        else:
            row.append(float(seed["length"]))
        rows.append(row)
    return np.ascontiguousarray(np.asarray(rows, dtype=np.float64))


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass
class QuantizedSeedProxyLibrary:
    """Memory-efficient normalized charge-shape library.

    Only quantized log-shapes are retained.  Because every proxy prediction is
    normalized to the event's mean charge, the unhit-PMT Poisson sum is common
    to all seeds to leading order.  Ranking therefore uses the hit-PMT term
    ``-sum(q log shape)``.  The best candidates are always re-evaluated with the
    exact optical objective before fitting.
    """

    codes: np.ndarray
    log_offset: float
    log_scale: float
    seed_matrix: np.ndarray
    group_index: np.ndarray
    group_counts: np.ndarray
    metadata: dict[str, object]
    path: str | None = None

    def score(self, obs_pes: np.ndarray, *, charge_floor_pe: float = 1.0e-4) -> np.ndarray:
        q = np.asarray(obs_pes, dtype=np.float64)
        if q.ndim != 1 or q.size != self.group_index.size:
            raise ValueError(
                f"obs_pes must have length {self.group_index.size}, got {q.shape}"
            )
        valid_pmt = (self.group_index >= 0) & np.isfinite(q)
        grouped_q = np.bincount(
            self.group_index[valid_pmt],
            weights=q[valid_pmt],
            minlength=int(self.group_counts.size),
        ).astype(np.float64, copy=False)
        hit = np.flatnonzero(grouped_q > 0.0).astype(np.int32)
        if hit.size == 0:
            return np.zeros(self.codes.shape[0], dtype=np.float64)
        qhit = np.ascontiguousarray(grouped_q[hit], dtype=np.float64)
        qmean = max(float(np.mean(grouped_q)), 1.0e-12)
        group_floor = np.maximum(
            float(charge_floor_pe) * np.asarray(self.group_counts, dtype=np.float64),
            qmean * math.exp(_PROXY_LOG_MIN),
        )
        log_floor = np.ascontiguousarray(np.log(group_floor / qmean), dtype=np.float64)
        return _score_quantized_log_shapes_numba(
            self.codes,
            np.ascontiguousarray(hit, dtype=np.int32),
            qhit,
            float(self.log_offset),
            float(self.log_scale),
            log_floor,
        )

    @property
    def memory_bytes(self) -> int:
        return int(
            self.codes.nbytes + self.seed_matrix.nbytes
            + self.group_index.nbytes + self.group_counts.nbytes
        )

    @classmethod
    def load(cls, path: str | Path, expected_metadata: Mapping[str, object] | None = None):
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
            if expected_metadata is not None and metadata != dict(expected_metadata):
                raise ValueError("seed proxy metadata does not match current configuration")
            if int(metadata.get("schema_version", -1)) != _PROXY_SCHEMA_VERSION:
                raise ValueError("unsupported seed proxy schema")
            return cls(
                np.ascontiguousarray(payload["log_shape_codes"]),
                float(np.asarray(payload["log_offset"]).item()),
                float(np.asarray(payload["log_scale"]).item()),
                np.ascontiguousarray(payload["seed_matrix"], dtype=np.float64),
                np.ascontiguousarray(payload["group_index"], dtype=np.int32),
                np.ascontiguousarray(payload["group_counts"], dtype=np.int32),
                metadata,
                str(path),
            )

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + f".tmp.{os.getpid()}.npz")
        np.savez_compressed(
            tmp,
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            seed_matrix=self.seed_matrix,
            log_shape_codes=self.codes,
            log_offset=np.asarray(self.log_offset, dtype=np.float64),
            log_scale=np.asarray(self.log_scale, dtype=np.float64),
            group_index=self.group_index,
            group_counts=self.group_counts,
        )
        os.replace(tmp, path)
        self.path = str(path)
        return str(path)


def proxy_library_metadata(
    seeds: Sequence[Mapping[str, object]],
    *,
    track_end_mode: str,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    particle: str,
    group_index: np.ndarray | None = None,
    quantization_bits: int = 8,
    source_files: Iterable[str | Path] = (),
    extra: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    matrix = _seed_matrix(seeds, track_end_mode=track_end_mode)
    npmts = int(np.asarray(p_locations).shape[0])
    if group_index is None:
        groups = np.arange(npmts, dtype=np.int32)
    else:
        raw_groups = np.asarray(group_index, dtype=np.int64)
        if raw_groups.shape != (npmts,):
            raise ValueError("group_index must have one entry per PMT")
        valid = raw_groups >= 0
        unique = np.unique(raw_groups[valid])
        remap = {int(value): i for i, value in enumerate(unique)}
        groups = np.full(npmts, -1, dtype=np.int32)
        for value, code in remap.items():
            groups[raw_groups == value] = int(code)
    n_groups = int(np.max(groups) + 1) if np.any(groups >= 0) else 0
    digest = hashlib.sha256()
    digest.update(matrix.tobytes())
    digest.update(np.ascontiguousarray(p_locations, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(pmt_normals, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(groups, dtype=np.int32).tobytes())
    for source in source_files:
        path = Path(source)
        try:
            digest.update(path.read_bytes())
        except OSError:
            # Optional analytic-fallback tables may be deliberately absent.
            # Hash a stable logical identity rather than the absolute
            # extraction path, otherwise moving a release ZIP changes its
            # proxy filename and forces an unnecessary rebuild.
            logical_name = "/".join(path.parts[-2:])
            digest.update(b"<missing-proxy-source-v1>\0")
            digest.update(logical_name.encode("utf-8"))
    if extra:
        # Configuration fields that alter proxy construction must participate in
        # the cache filename, not merely in the compatibility check. Otherwise
        # two modes repeatedly overwrite the same path and force a rebuild on
        # every alternating job.
        def _json_default(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Path):
                return str(value)
            raise TypeError(f"cannot serialize proxy metadata value {type(value)!r}")

        payload = json.dumps(
            dict(extra), sort_keys=True, separators=(",", ":"),
            default=_json_default,
        )
        digest.update(payload.encode("utf-8"))
    metadata: dict[str, object] = {
        "schema_version": _PROXY_SCHEMA_VERSION,
        "table_kind": "licketyfit_quantized_global_seed_log_shapes",
        "track_end_mode": str(track_end_mode),
        "particle": str(particle),
        "n_seeds": int(matrix.shape[0]),
        "n_pmts": npmts,
        "n_groups": n_groups,
        "group_index_sha256": _hash_array(groups),
        "quantization_bits": int(quantization_bits),
        "seed_matrix_sha256": _hash_array(matrix),
        "pmt_positions_sha256": _hash_array(np.asarray(p_locations, dtype=np.float64)),
        "pmt_normals_sha256": _hash_array(np.asarray(pmt_normals, dtype=np.float64)),
        "log_min": _PROXY_LOG_MIN,
        "log_max": _PROXY_LOG_MAX,
        "digest": digest.hexdigest(),
    }
    if extra:
        metadata.update(dict(extra))
    return metadata, matrix


def build_quantized_proxy_library(
    seeds: Sequence[Mapping[str, object]],
    predict_shape: Callable[[Mapping[str, object]], np.ndarray],
    *,
    metadata: Mapping[str, object],
    seed_matrix: np.ndarray,
    group_index: np.ndarray | None = None,
    quantization_bits: int = 8,
    progress_every: int = 1000,
) -> QuantizedSeedProxyLibrary:
    nseed = len(seeds)
    if nseed < 1:
        raise ValueError("cannot build an empty seed proxy library")
    first_full = np.asarray(predict_shape(seeds[0]), dtype=np.float64)
    if first_full.ndim != 1:
        raise ValueError("predict_shape must return a one-dimensional array")
    npmts = first_full.size
    if group_index is None:
        groups = np.arange(npmts, dtype=np.int32)
    else:
        raw_groups = np.asarray(group_index, dtype=np.int64)
        if raw_groups.shape != (npmts,):
            raise ValueError("group_index must have one entry per predicted PMT")
        valid = raw_groups >= 0
        unique = np.unique(raw_groups[valid])
        remap = {int(value): i for i, value in enumerate(unique)}
        groups = np.full(npmts, -1, dtype=np.int32)
        for value, code in remap.items():
            groups[raw_groups == value] = int(code)
    n_groups = int(np.max(groups) + 1) if np.any(groups >= 0) else 0
    if n_groups < 1:
        raise ValueError("proxy grouping contains no valid groups")
    group_counts = np.bincount(groups[groups >= 0], minlength=n_groups).astype(np.int32)
    _, _, code_dtype = _quantization_range(quantization_bits)
    codes = np.empty((nseed, n_groups), dtype=code_dtype)

    def aggregate(row: np.ndarray) -> np.ndarray:
        shape = np.asarray(row, dtype=np.float64)
        if shape.shape != (npmts,):
            raise ValueError("inconsistent proxy shape length")
        valid = (groups >= 0) & np.isfinite(shape) & (shape > 0.0)
        return np.bincount(
            groups[valid], weights=shape[valid], minlength=n_groups
        ).astype(np.float64, copy=False)

    def encode(row: np.ndarray) -> np.ndarray:
        shape = aggregate(row)
        mean = float(np.mean(shape))
        if not math.isfinite(mean) or mean <= 0.0:
            code_min, _, dtype = _quantization_range(quantization_bits)
            return np.full(n_groups, code_min, dtype=dtype)
        logshape = np.log(np.maximum(shape / mean, math.exp(_PROXY_LOG_MIN)))
        encoded, _, _ = _quantize_log_shapes(logshape, bits=quantization_bits)
        return encoded

    codes[0] = encode(first_full)
    for i in range(1, nseed):
        codes[i] = encode(predict_shape(seeds[i]))
        if progress_every > 0 and (i + 1) % int(progress_every) == 0:
            print(f"  built proxy shapes {i + 1}/{nseed}", flush=True)
    _, offset, scale = _quantize_log_shapes(
        np.asarray([0.0]), bits=quantization_bits
    )
    return QuantizedSeedProxyLibrary(
        np.ascontiguousarray(codes),
        float(offset),
        float(scale),
        np.ascontiguousarray(seed_matrix, dtype=np.float64),
        np.ascontiguousarray(groups, dtype=np.int32),
        np.ascontiguousarray(group_counts, dtype=np.int32),
        dict(metadata),
    )


def select_diverse_seed_indices(
    scores: np.ndarray,
    seeds: Sequence[Mapping[str, object]],
    *,
    n_keep: int = 6,
    raw_top: int = 2,
) -> list[int]:
    """Select low-score candidates spanning distinct directions and vertices."""
    order = np.argsort(np.asarray(scores, dtype=np.float64))
    n_keep = max(1, int(n_keep))
    selected: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> bool:
        index = int(index)
        if index in seen or len(selected) >= n_keep:
            return False
        seen.add(index)
        selected.append(index)
        return True

    for index in order[: max(1, min(int(raw_top), n_keep))]:
        add(int(index))

    direction_target = min(n_keep, len(selected) + max(2, n_keep // 3))
    seen_direction = {
        tuple(np.round(direction_from_mapping(seeds[i]), 3)) for i in selected
    }
    for index in order:
        key = tuple(np.round(direction_from_mapping(seeds[int(index)]), 3))
        if key in seen_direction:
            continue
        if add(int(index)):
            seen_direction.add(key)
        if len(selected) >= direction_target:
            break

    vertex_target = min(n_keep, len(selected) + max(2, n_keep // 3))
    seen_vertex = {
        tuple(round(float(seeds[i][k]), 1) for k in ("x0", "y0", "z0"))
        for i in selected
    }
    for index in order:
        seed = seeds[int(index)]
        key = tuple(round(float(seed[k]), 1) for k in ("x0", "y0", "z0"))
        if key in seen_vertex:
            continue
        if add(int(index)):
            seen_vertex.add(key)
        if len(selected) >= vertex_target:
            break

    length_name = "visible_length" if "visible_length" in seeds[0] else "length"
    seen_length = {round(float(seeds[i][length_name]), 1) for i in selected}
    for index in order:
        key = round(float(seeds[int(index)][length_name]), 1)
        if key in seen_length:
            continue
        if add(int(index)):
            seen_length.add(key)
        if len(selected) >= n_keep:
            break

    for index in order:
        add(int(index))
        if len(selected) >= n_keep:
            break
    return selected


def build_vertex_grid(
    detector: ConvexDetectorVolume,
    *,
    spacing_mm: float = 700.0,
    wall_margin_mm: float = 80.0,
    extra_points: Iterable[Sequence[float]] = (),
) -> list[tuple[float, float, float]]:
    spacing = max(float(spacing_mm), 1.0)
    center = 0.5 * (detector.axis_lo + detector.axis_hi)
    axes: list[list[float]] = []
    for k in range(3):
        lo = float(detector.axis_lo[k])
        hi = float(detector.axis_hi[k])
        c = float(center[k])
        start = int(math.ceil((lo - c) / spacing))
        stop = int(math.floor((hi - c) / spacing))
        vals = [c + j * spacing for j in range(start, stop + 1)]
        # Do not force exact detector-boundary coordinates into every grid.
        # They create zero-exit seeds and, at very coarse spacing, leave a
        # fixed 3x3x3 anchor product that cannot be reduced by the seed budget.
        # The centre-anchored lattice already approaches each wall as the
        # requested spacing is refined.  Keep the conventional coordinate
        # origin only when it lies inside this Cartesian extent.
        if lo <= 0.0 <= hi:
            vals.append(0.0)
        axes.append(sorted({round(float(v), 6) for v in vals if lo <= v <= hi}))
    points: list[tuple[float, float, float]] = []
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                p = np.asarray([x, y, z], dtype=np.float64)
                if not detector.contains(p, extra_margin_mm=float(wall_margin_mm)):
                    continue
                points.append((float(x), float(y), float(z)))
    for point in extra_points:
        p = np.asarray(point, dtype=np.float64)
        if detector.contains(p, extra_margin_mm=float(wall_margin_mm)):
            points.append(tuple(float(x) for x in p))
    if detector.contains(center, extra_margin_mm=float(wall_margin_mm)):
        points.append(tuple(float(x) for x in center))
    unique: list[tuple[float, float, float]] = []
    seen = set()
    for point in points:
        key = tuple(round(float(x), 6) for x in point)
        if key not in seen:
            seen.add(key)
            unique.append(tuple(float(x) for x in point))
    if not unique:
        raise ValueError(
            "The requested detector-global wall margin leaves no valid vertex "
            "seeds. Reduce DETECTOR_GLOBAL_SEED_WALL_MARGIN_MM or provide a "
            "valid extra seed point."
        )
    return unique


def default_length_grid(
    max_length_mm: float,
    *,
    fractions: Sequence[float] = (0.10, 0.20, 0.35, 0.50, 0.70, 0.90),
    minimum_mm: float = 80.0,
) -> list[float]:
    """Truth-independent global visible-length seeds."""
    maximum = max(float(max_length_mm), float(minimum_mm))
    values = [float(np.clip(maximum * float(f), minimum_mm, maximum)) for f in fractions]
    return sorted({round(x, 6) for x in values})


def tangent_offset_directions(
    anchor_direction: Sequence[float],
    tangent_offsets: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return unit directions generated around one arbitrary anchor.

    ``tangent_offsets`` contains ``(u,v)`` coordinates in the normalized
    tangent chart. Near the origin these are angular offsets in radians. This
    helper is used by the WCTE beam-focused seed library and is equally valid
    for any detector orientation.
    """
    chart = TangentDirectionChart.from_direction(anchor_direction)
    directions: list[np.ndarray] = []
    for offset in tangent_offsets:
        if len(offset) != 2:
            raise ValueError("each tangent offset must contain (u,v)")
        direction = chart.direction(float(offset[0]), float(offset[1]))
        if direction is None:
            continue
        if any(
            float(np.dot(direction, previous)) > 1.0 - 1.0e-12
            for previous in directions
        ):
            continue
        directions.append(direction)
    if not directions:
        raise ValueError("tangent offset direction grid is empty")
    return np.ascontiguousarray(np.asarray(directions, dtype=np.float64))


def _append_contained_seed_hypotheses(
    output: list[dict[str, object]],
    detector: ConvexDetectorVolume,
    vertex: Sequence[float],
    direction: Sequence[float],
    lengths_mm: Sequence[float],
    *,
    track_end_mode: str,
    full_ranges_mm: Sequence[float] | None,
    seed_family: str,
) -> None:
    """Append every physically contained length/range hypothesis."""
    point = np.asarray(vertex, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if point.shape != (3,) or d.shape != (3,):
        return
    if not detector.contains(point):
        return
    dnorm = float(np.linalg.norm(d))
    if not math.isfinite(dnorm) or dnorm <= 0.0:
        return
    d = d / dnorm
    exit_distance = detector.ray_exit_distance(point, d)
    if not math.isfinite(exit_distance) or exit_distance <= 0.0:
        return
    for length_value in lengths_mm:
        length = float(length_value)
        if not math.isfinite(length) or length < 0.0:
            continue
        if length > exit_distance + 1.0e-6:
            continue
        base: dict[str, object] = {
            "x0": float(point[0]),
            "y0": float(point[1]),
            "z0": float(point[2]),
            "dir_x": float(d[0]),
            "dir_y": float(d[1]),
            "dir_z": float(d[2]),
            "cx": float(d[0]),
            "cy": float(d[1]),
            "cz": float(d[2]),
            "cz_sign": -1.0 if float(d[2]) < 0.0 else 1.0,
            "t0": 0.0,
            "seed_family": str(seed_family),
        }
        if track_end_mode == "absorption":
            if full_ranges_mm is None:
                raise ValueError("absorption mode requires full_ranges_mm")
            for full_range_value in full_ranges_mm:
                full_range = float(full_range_value)
                if not math.isfinite(full_range) or full_range <= 0.0:
                    continue
                if length <= full_range:
                    seed = dict(base)
                    seed["visible_length"] = length
                    seed["full_range"] = full_range
                    output.append(seed)
        else:
            seed = dict(base)
            seed["length"] = length
            output.append(seed)


def build_focused_seed_grid(
    detector: ConvexDetectorVolume,
    *,
    reference_vertex_mm: Sequence[float],
    anchor_direction: Sequence[float],
    lengths_mm: Sequence[float],
    longitudinal_offsets_mm: Sequence[float] = (0.0,),
    transverse_offsets_mm: Sequence[Sequence[float]] = ((0.0, 0.0),),
    direction_tangent_offsets: Sequence[Sequence[float]] = ((0.0, 0.0),),
    track_end_mode: str = "full_length",
    full_ranges_mm: Sequence[float] | None = None,
    seed_family: str = "focused",
) -> list[dict[str, object]]:
    """Build a compact orientation-independent seed cluster.

    Vertex offsets are expressed in the physical frame of ``anchor_direction``:
    longitudinal along the track and two transverse coordinates in its tangent
    plane. Direction perturbations use the same nonsingular tangent chart. This
    is the generalized replacement for the old WCTE x/y/z/cx/cy beam-pipe grid.
    """
    reference = np.asarray(reference_vertex_mm, dtype=np.float64)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_vertex_mm must be a finite 3-vector")
    chart = TangentDirectionChart.from_direction(anchor_direction)
    directions = tangent_offset_directions(
        anchor_direction, direction_tangent_offsets
    )
    seeds: list[dict[str, object]] = []
    for longitudinal in longitudinal_offsets_mm:
        for transverse in transverse_offsets_mm:
            if len(transverse) != 2:
                raise ValueError("each transverse offset must contain two values")
            vertex = (
                reference
                + float(longitudinal) * chart.anchor
                + float(transverse[0]) * chart.e1
                + float(transverse[1]) * chart.e2
            )
            for direction in directions:
                _append_contained_seed_hypotheses(
                    seeds,
                    detector,
                    vertex,
                    direction,
                    lengths_mm,
                    track_end_mode=track_end_mode,
                    full_ranges_mm=full_ranges_mm,
                    seed_family=seed_family,
                )
    return seeds


def detector_entry_reference(
    detector: ConvexDetectorVolume,
    direction: Sequence[float],
    *,
    inset_mm: float,
) -> np.ndarray | None:
    """Return a point ``inset_mm`` inside the upstream detector boundary."""
    d = np.asarray(direction, dtype=np.float64)
    if d.shape != (3,) or not np.all(np.isfinite(d)):
        return None
    norm = float(np.linalg.norm(d))
    if norm <= 0.0:
        return None
    d = d / norm
    center = (
        np.asarray(detector.reference_center, dtype=np.float64)
        if detector.reference_center is not None
        else 0.5 * (detector.axis_lo + detector.axis_hi)
    )
    upstream_distance = detector.ray_exit_distance(center, -d)
    if not math.isfinite(upstream_distance):
        return None
    point = center - upstream_distance * d + max(float(inset_mm), 0.0) * d
    if not detector.contains(point, tolerance_mm=1.0e-6):
        return None
    return np.ascontiguousarray(point, dtype=np.float64)


def build_entry_guard_seed_grid(
    detector: ConvexDetectorVolume,
    directions: np.ndarray,
    lengths_mm: Sequence[float],
    *,
    insets_mm: Sequence[float] = (200.0, 400.0, 650.0),
    transverse_offsets_mm: Sequence[Sequence[float]] = (
        (0.0, 0.0),
        (250.0, 0.0),
        (-250.0, 0.0),
        (0.0, 250.0),
        (0.0, -250.0),
    ),
    track_end_mode: str = "full_length",
    full_ranges_mm: Sequence[float] | None = None,
    seed_family: str = "orientation_guard",
) -> list[dict[str, object]]:
    """Build a sparse full-sphere entry library for focused detector running."""
    seeds: list[dict[str, object]] = []
    for direction_value in np.asarray(directions, dtype=np.float64):
        chart = TangentDirectionChart.from_direction(direction_value)
        for inset in insets_mm:
            reference = detector_entry_reference(
                detector, chart.anchor, inset_mm=float(inset)
            )
            if reference is None:
                continue
            for transverse in transverse_offsets_mm:
                if len(transverse) != 2:
                    raise ValueError("each transverse offset must contain two values")
                vertex = (
                    reference
                    + float(transverse[0]) * chart.e1
                    + float(transverse[1]) * chart.e2
                )
                _append_contained_seed_hypotheses(
                    seeds,
                    detector,
                    vertex,
                    chart.anchor,
                    lengths_mm,
                    track_end_mode=track_end_mode,
                    full_ranges_mm=full_ranges_mm,
                    seed_family=seed_family,
                )
    return seeds


def filter_seed_grid_for_fixed_parameters(
    seeds: Sequence[Mapping[str, object]],
    detector: ConvexDetectorVolume,
    *,
    track_end_mode: str,
    fixed_params: Mapping[str, float] | None = None,
    fixed_direction: Sequence[float] | None = None,
    full_range_limits: tuple[float, float] | None = None,
    prevalidated: bool = False,
) -> list[dict[str, object]]:
    """Apply fixed values, reject nonphysical rows, and deduplicate.

    ``prevalidated=True`` is an explicit contract for seed builders that have
    already checked every vertex, direction, visible length, full range and
    detector segment.  It avoids repeating the comparatively expensive ray
    intersection for an unchanged bank.  Any fixed value other than ``t0`` (or
    any fixed direction) can change physicality and therefore automatically
    restores the complete validation path.
    """
    fixed = {} if fixed_params is None else dict(fixed_params)
    fixed_d = None
    if fixed_direction is not None:
        fixed_d = np.asarray(fixed_direction, dtype=np.float64)
        norm = float(np.linalg.norm(fixed_d))
        if (
            fixed_d.shape != (3,)
            or not np.all(np.isfinite(fixed_d))
            or norm <= 0.0
        ):
            raise ValueError("fixed_direction must be a finite nonzero 3-vector")
        fixed_d = fixed_d / norm
    must_validate = (
        not bool(prevalidated)
        or fixed_d is not None
        or any(str(name) != "t0" for name in fixed)
    )
    output: list[dict[str, object]] = []
    for original in seeds:
        seed = dict(original)
        for name, value in fixed.items():
            if name != "direction" and name in seed:
                seed[name] = float(value)
        if fixed_d is not None:
            seed.update(
                {
                    "dir_x": float(fixed_d[0]),
                    "dir_y": float(fixed_d[1]),
                    "dir_z": float(fixed_d[2]),
                    "cx": float(fixed_d[0]),
                    "cy": float(fixed_d[1]),
                    "cz": float(fixed_d[2]),
                    "cz_sign": -1.0 if float(fixed_d[2]) < 0.0 else 1.0,
                }
            )
        if not must_validate:
            output.append(seed)
            continue
        try:
            direction = direction_from_mapping(seed)
            vertex = np.asarray(
                [float(seed["x0"]), float(seed["y0"]), float(seed["z0"])],
                dtype=np.float64,
            )
        except Exception:
            continue
        if not detector.contains(vertex):
            continue
        if track_end_mode == "absorption":
            try:
                visible = float(seed["visible_length"])
                full_range = float(seed["full_range"])
            except Exception:
                continue
            if (
                not math.isfinite(visible)
                or visible < 0.0
                or not math.isfinite(full_range)
                or full_range <= 0.0
                or visible > full_range
            ):
                continue
            if full_range_limits is not None and not (
                float(full_range_limits[0])
                <= full_range
                <= float(full_range_limits[1])
            ):
                continue
            length = visible
        else:
            try:
                length = float(seed["length"])
            except Exception:
                continue
            if not math.isfinite(length) or length < 0.0:
                continue
        if not detector.segment_contained(vertex, direction, length):
            continue
        output.append(seed)
    output = deduplicate_seed_grid(output, track_end_mode=track_end_mode)
    if not output:
        raise ValueError(
            "No physical seeds remain after applying fixed parameters. Check "
            "the fixed vertex, direction, length/range, and detector mode."
        )
    return output


def build_global_seed_grid(
    detector: ConvexDetectorVolume,
    directions: np.ndarray,
    lengths_mm: Sequence[float],
    *,
    track_end_mode: str = "full_length",
    full_ranges_mm: Sequence[float] | None = None,
    vertex_spacing_mm: float = 700.0,
    wall_margin_mm: float = 80.0,
    extra_vertices: Iterable[Sequence[float]] = (),
) -> list[dict[str, float]]:
    vertices = build_vertex_grid(
        detector,
        spacing_mm=vertex_spacing_mm,
        wall_margin_mm=wall_margin_mm,
        extra_points=extra_vertices,
    )
    return _build_global_seed_grid_from_vertices(
        detector,
        vertices,
        directions,
        lengths_mm,
        track_end_mode=track_end_mode,
        full_ranges_mm=full_ranges_mm,
    )


def _build_global_seed_grid_from_vertices(
    detector: ConvexDetectorVolume,
    vertices: Sequence[Sequence[float]],
    directions: np.ndarray,
    lengths_mm: Sequence[float],
    *,
    track_end_mode: str,
    full_ranges_mm: Sequence[float] | None,
) -> list[dict[str, float]]:
    """Construct physical seed dictionaries from an already-budgeted vertex set."""
    seeds: list[dict[str, float]] = []
    for x0, y0, z0 in vertices:
        for direction in np.asarray(directions, dtype=np.float64):
            exit_distance = detector.ray_exit_distance(
                (x0, y0, z0), direction
            )
            if not math.isfinite(exit_distance) or exit_distance <= 0.0:
                continue
            for length in lengths_mm:
                if float(length) > exit_distance + 1.0e-6:
                    continue
                base = {
                    "x0": float(x0), "y0": float(y0), "z0": float(z0),
                    "dir_x": float(direction[0]),
                    "dir_y": float(direction[1]),
                    "dir_z": float(direction[2]),
                    "cx": float(direction[0]),
                    "cy": float(direction[1]),
                    "cz": float(direction[2]),
                    "cz_sign": -1.0 if float(direction[2]) < 0.0 else 1.0,
                    "t0": 0.0,
                }
                if track_end_mode == "absorption":
                    if full_ranges_mm is None:
                        raise ValueError("absorption mode requires full_ranges_mm")
                    for full_range in full_ranges_mm:
                        if float(length) <= float(full_range):
                            seed = dict(base)
                            seed["visible_length"] = float(length)
                            seed["full_range"] = float(full_range)
                            seeds.append(seed)
                else:
                    seed = dict(base)
                    seed["length"] = float(length)
                    seeds.append(seed)
    return seeds


def _count_global_seed_grid_from_vertices(
    detector: ConvexDetectorVolume,
    vertices: Sequence[Sequence[float]],
    directions: np.ndarray,
    lengths_mm: Sequence[float],
    *,
    track_end_mode: str,
    full_ranges_mm: Sequence[float] | None,
) -> int:
    """Count contained hypotheses without allocating thousands of dictionaries."""
    directions_array = np.asarray(directions, dtype=np.float64)
    visible_lengths = tuple(float(x) for x in lengths_mm)
    full_ranges = (
        None
        if full_ranges_mm is None
        else tuple(sorted(float(x) for x in full_ranges_mm))
    )
    if track_end_mode == "absorption" and full_ranges is None:
        raise ValueError("absorption mode requires full_ranges_mm")
    count = 0
    for vertex in vertices:
        for direction in directions_array:
            exit_distance = detector.ray_exit_distance(vertex, direction)
            if not math.isfinite(exit_distance) or exit_distance <= 0.0:
                continue
            for length in visible_lengths:
                if length > exit_distance + 1.0e-6:
                    continue
                if track_end_mode == "absorption":
                    assert full_ranges is not None
                    count += sum(1 for full_range in full_ranges if length <= full_range)
                else:
                    count += 1
    return int(count)


def build_budgeted_global_seed_grid(
    detector: ConvexDetectorVolume,
    directions: np.ndarray,
    lengths_mm: Sequence[float],
    *,
    track_end_mode: str = "full_length",
    full_ranges_mm: Sequence[float] | None = None,
    vertex_spacing_mm: float = 700.0,
    wall_margin_mm: float = 80.0,
    extra_vertices: Iterable[Sequence[float]] = (),
    max_total_seeds: int = 20_000,
    max_iterations: int = 16,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Build a detector-global seed grid within a predictable latency budget.

    A full Cartesian product grows approximately with detector volume.  That is
    harmless for WCTE but can produce hundreds of thousands of proxy rows for a
    larger IWCD, increasing both score time and resident memory.  This helper
    keeps the direction and length coverage unchanged and increases only the
    coarse vertex spacing until the number of *physically contained* hypotheses
    fits the requested budget.

    The proxy library remains deterministic: the effective spacing and final
    seed matrix are part of its metadata/digest.  If even one-vertex direction ×
    length coverage exceeds the budget, the function raises with an actionable
    message rather than silently dropping directions.
    """
    requested_spacing = max(float(vertex_spacing_mm), 1.0)
    spacing = requested_spacing
    directions_array = np.ascontiguousarray(directions, dtype=np.float64)
    visible_lengths = tuple(float(x) for x in lengths_mm)
    full_ranges = (
        None
        if full_ranges_mm is None
        else tuple(float(x) for x in full_ranges_mm)
    )
    extra_vertices_tuple = tuple(tuple(float(x) for x in p) for p in extra_vertices)
    budget = int(max_total_seeds)
    if budget <= 0:
        budget = np.iinfo(np.int32).max
    iterations = 0
    final_vertices: list[tuple[float, float, float]] | None = None
    final_count = 0

    iteration_count = 0
    for iteration in range(max(1, int(max_iterations))):
        iteration_count = iteration + 1
        vertices = build_vertex_grid(
            detector,
            spacing_mm=spacing,
            wall_margin_mm=wall_margin_mm,
            extra_points=extra_vertices_tuple,
        )
        count = _count_global_seed_grid_from_vertices(
            detector,
            vertices,
            directions_array,
            visible_lengths,
            track_end_mode=track_end_mode,
            full_ranges_mm=full_ranges,
        )
        final_vertices = vertices
        final_count = count
        if count <= budget:
            break
        # Seed count scales roughly as spacing^-3.  Use that scaling with a
        # conservative minimum increase to converge in a few count-only passes.
        factor = max(1.15, (float(count) / float(budget)) ** (1.0 / 3.0))
        spacing *= factor
    assert final_vertices is not None
    if final_count > budget:
        raise RuntimeError(
            "Could not fit the global seed library into max_total_seeds="
            f"{budget}. Even spacing={spacing:.3f} mm leaves {final_count} "
            "contained hypotheses. Increase DETECTOR_GLOBAL_MAX_SEEDS, reduce "
            "the direction/length grids, or supply a detector-specific proxy."
        )

    seeds = _build_global_seed_grid_from_vertices(
        detector,
        final_vertices,
        directions_array,
        visible_lengths,
        track_end_mode=track_end_mode,
        full_ranges_mm=full_ranges,
    )
    metadata = {
        "requested_vertex_spacing_mm": float(requested_spacing),
        "effective_vertex_spacing_mm": float(spacing),
        "wall_margin_mm": float(wall_margin_mm),
        "vertex_count": int(len(final_vertices)),
        "seed_count": int(len(seeds)),
        "max_total_seeds": int(budget),
        "spacing_iterations": int(iteration_count),
        "direction_count": int(directions_array.shape[0]),
        "visible_length_count": int(len(visible_lengths)),
        "full_range_count": (
            0 if full_ranges is None else int(len(full_ranges))
        ),
    }
    return seeds, metadata


def deduplicate_seed_grid(
    seeds: Sequence[Mapping[str, object]],
    *,
    track_end_mode: str = "full_length",
    decimals: int = 9,
) -> list[dict[str, object]]:
    """Remove duplicate hypotheses after fixed-parameter overrides.

    Applying a fixed vertex coordinate, direction, or length after constructing
    a Cartesian seed product can otherwise leave thousands of identical rows in
    the proxy table.  This function compares the physical direction vector, not
    historical chart labels, so equivalent old/new seed schemas collapse.
    """
    if not seeds:
        return []
    matrix = _seed_matrix(seeds, track_end_mode=track_end_mode)
    rounded = np.round(matrix, int(decimals))
    _, first = np.unique(rounded, axis=0, return_index=True)
    return [dict(seeds[int(i)]) for i in np.sort(first)]

# -----------------------------------------------------------------------------
# Generalized Fermi--Eyges process update and timing prior
# -----------------------------------------------------------------------------

_LOCAL_TRACK_NAMES = ("x0", "y0", "z0", "dir_u", "dir_v", "length")
_ALIGNED_TRACK_NAMES = (
    "vertex_longitudinal", "vertex_transverse_1", "vertex_transverse_2",
    "dir_u", "dir_v", "length",
)


def local_track_vector(values: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(values[name]) for name in _LOCAL_TRACK_NAMES], dtype=np.float64)


def aligned_delta_vector(
    values: Mapping[str, float],
    center: Mapping[str, float],
    chart: TangentDirectionChart,
) -> np.ndarray:
    dv = np.asarray(
        [
            float(values["x0"]) - float(center["x0"]),
            float(values["y0"]) - float(center["y0"]),
            float(values["z0"]) - float(center["z0"]),
        ],
        dtype=np.float64,
    )
    return np.asarray(
        [
            float(np.dot(dv, chart.anchor)),
            float(np.dot(dv, chart.e1)),
            float(np.dot(dv, chart.e2)),
            float(values.get("dir_u", 0.0)) - float(center.get("dir_u", 0.0)),
            float(values.get("dir_v", 0.0)) - float(center.get("dir_v", 0.0)),
            float(values["length"]) - float(center["length"]),
        ],
        dtype=np.float64,
    )


def local_covariance_to_aligned(
    covariance: np.ndarray,
    chart: TangentDirectionChart,
) -> np.ndarray:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (6, 6):
        raise ValueError("local covariance must have shape (6,6)")
    transform = np.zeros((6, 6), dtype=np.float64)
    transform[0, :3] = chart.anchor
    transform[1, :3] = chart.e1
    transform[2, :3] = chart.e2
    transform[3, 3] = 1.0
    transform[4, 4] = 1.0
    transform[5, 5] = 1.0
    out = transform @ cov @ transform.T
    return np.ascontiguousarray(0.5 * (out + out.T), dtype=np.float64)


def _psd_precision(covariance: np.ndarray, *, rcond: float = 1.0e-10) -> np.ndarray:
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be square")
    if cov.size == 0:
        return cov.copy()
    if np.any(~np.isfinite(cov)):
        raise ValueError("covariance contains non-finite entries")
    cov = 0.5 * (cov + cov.T)
    scale = np.sqrt(np.maximum(np.diag(cov), 1.0e-30))
    corr = cov / scale[:, None] / scale[None, :]
    corr = 0.5 * (corr + corr.T)
    eig, vec = np.linalg.eigh(corr)
    cutoff = float(rcond) * max(float(np.max(eig)), 1.0)
    inv = np.zeros_like(eig)
    np.divide(1.0, eig, out=inv, where=(eig > cutoff))
    precision = (vec @ np.diag(inv) @ vec.T) / scale[:, None] / scale[None, :]
    return np.ascontiguousarray(0.5 * (precision + precision.T), dtype=np.float64)


@dataclass
class AlignedPriorObjective:
    """Track objective plus a Gaussian prior in track-aligned coordinates."""

    base: TrackObjective
    center: dict[str, float]
    aligned_indices: tuple[int, ...]
    precision: np.ndarray

    @property
    def chart(self):
        return self.base.chart

    @property
    def detector(self):
        return self.base.detector

    @property
    def track_end_mode(self):
        return self.base.track_end_mode

    @property
    def objective_mode(self):
        return self.base.objective_mode

    @property
    def evaluations(self):
        return self.base.evaluations

    @property
    def calls(self):
        return self.base.calls

    @property
    def optical_evaluations(self):
        return self.base.optical_evaluations

    @property
    def invalid_evaluations(self):
        return self.base.invalid_evaluations

    def evaluate_t0_many(
        self, values: Mapping[str, float], t0_values
    ) -> np.ndarray:
        """Delegate exact t0 batching and add the t0-independent prior once."""
        out = np.asarray(
            self.base.evaluate_t0_many(values, t0_values), dtype=np.float64
        ).copy()
        if not self.aligned_indices:
            return out
        delta = aligned_delta_vector(values, self.center, self.chart)
        selected = delta[list(self.aligned_indices)]
        penalty = 0.5 * float(selected @ self.precision @ selected)
        finite = np.isfinite(out)
        out[finite] += penalty
        return out

    def __call__(self, values: Mapping[str, float]) -> float:
        fval = float(self.base(values))
        if not math.isfinite(fval):
            return np.inf
        if not self.aligned_indices:
            return fval
        delta = aligned_delta_vector(values, self.center, self.chart)
        selected = delta[list(self.aligned_indices)]
        penalty = 0.5 * float(selected @ self.precision @ selected)
        return fval + penalty


def make_aligned_prior_objective(
    base: TrackObjective,
    center: Mapping[str, float],
    local_covariance: np.ndarray,
    aligned_indices: Sequence[int],
) -> AlignedPriorObjective:
    aligned_cov = local_covariance_to_aligned(local_covariance, base.chart)
    requested = tuple(int(i) for i in aligned_indices)
    idx = tuple(
        i for i in requested
        if 0 <= i < aligned_cov.shape[0]
        and np.isfinite(aligned_cov[i, i])
        and aligned_cov[i, i] > 0.0
    )
    sub = aligned_cov[np.ix_(idx, idx)] if idx else np.empty((0, 0))
    if idx and np.any(~np.isfinite(sub)):
        # Drop coordinates whose cross-covariances were not computed (e.g. a
        # user-fixed parameter excluded from the FE finite-difference block).
        keep = [j for j in range(len(idx)) if np.all(np.isfinite(sub[j])) and np.all(np.isfinite(sub[:, j]))]
        idx = tuple(idx[j] for j in keep)
        sub = aligned_cov[np.ix_(idx, idx)] if idx else np.empty((0, 0))
    precision = _psd_precision(sub) if idx else np.empty((0, 0), dtype=np.float64)
    return AlignedPriorObjective(base, {k: float(v) for k, v in center.items()}, idx, precision)


def predict_charge_and_process_jacobian(
    template_emitter,
    *,
    values: Mapping[str, float],
    chart: TangentDirectionChart,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    detector: ConvexDetectorVolume | None = None,
    mpmt_types=None,
    need_process_jacobian: bool = False,
    process_modes_per_plane: int = 4,
    process_grid_points: int = 41,
    need_times: bool = False,
    return_emitter_context: bool = False,
):
    direction = chart.direction(values.get("dir_u", 0.0), values.get("dir_v", 0.0))
    if direction is None:
        return np.empty(0), None, None
    vertex = np.asarray(
        [float(values["x0"]), float(values["y0"]), float(values["z0"])],
        dtype=np.float64,
    )
    length = float(values["length"])
    if detector is not None and not detector.segment_contained(
        vertex, direction, length
    ):
        return np.empty(0), None, None
    emitter = template_emitter.copy()
    emitter.enable_primary_mcs_smearing = False
    emitter.primary_mcs_model = "fermi_eyges_process"
    emitter.primary_mcs_process_modes_per_plane = int(process_modes_per_plane)
    emitter.primary_mcs_process_grid_points = int(process_grid_points)
    emitter.compute_primary_mcs_process_jacobian = bool(need_process_jacobian)
    # This helper represents a contained full-length hypothesis: ``length`` is
    # the remaining CSDA range to Cherenkov threshold.  The production template
    # can be configured in abrupt mode to support boundary-clipped straight
    # fits, but carrying that fixed nominal KE into this contained FE context
    # makes the precomputed emitter's full range disagree with ``length``.
    # Reset the copied emitter explicitly; never mutate the shared template.
    if hasattr(emitter, "configure_track_end"):
        emitter.configure_track_end(
            "threshold", fixed_initial_KE=None, refresh=False
        )
    else:
        emitter.track_end_mode = "threshold"
        emitter.fixed_initial_KE = None
    emitter.start_coord = tuple(float(x) for x in vertex)
    emitter.direction = tuple(float(x) for x in direction)
    emitter.starting_time = 0.0
    init_ke = emitter.refresh_kinematics_from_length(length)
    emission = emitter.get_emission_points(p_locations, init_ke)
    mu, timing = emitter.get_expected_pes_ts(
        wcd, emission, p_locations, pmt_normals, mpmt_types, obs_pes,
        need_times=bool(need_times),
    )
    process_jacobian = getattr(emitter, "_last_mcs_charge_jacobian", None)
    explained = getattr(emitter, "_last_mcs_basis_explained_fraction", None)
    result = (
        np.asarray(mu, dtype=np.float64),
        None if process_jacobian is None else np.asarray(process_jacobian, dtype=np.float64),
        None if explained is None else np.asarray(explained, dtype=np.float64),
    )
    if return_emitter_context:
        return result + (emitter, timing)
    return result


def finite_difference_local_charge_jacobian(
    template_emitter,
    *,
    values: Mapping[str, float],
    chart: TangentDirectionChart,
    detector: ConvexDetectorVolume,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    mpmt_types=None,
    derivative_indices: Sequence[int] | None = None,
    xyz_step_mm: float = 1.0,
    direction_step: float = 2.0e-4,
    length_step_mm: float = 1.0,
    length_limits: tuple[float, float] = (0.0, 3000.0),
) -> tuple[np.ndarray, np.ndarray, int]:
    center = {k: float(v) for k, v in values.items()}
    base, _, _ = predict_charge_and_process_jacobian(
        template_emitter,
        values=center, chart=chart, wcd=wcd,
        p_locations=p_locations, pmt_normals=pmt_normals,
        obs_pes=obs_pes, mpmt_types=mpmt_types,
        detector=detector,
        need_process_jacobian=False,
    )
    if base.size == 0 or not np.all(np.isfinite(base)):
        raise RuntimeError("invalid FE charge point")
    prediction_count = 1
    if derivative_indices is None:
        derivative_indices = tuple(range(6))
    derivative_indices = tuple(int(i) for i in derivative_indices)
    steps = (xyz_step_mm, xyz_step_mm, xyz_step_mm, direction_step, direction_step, length_step_mm)
    jacobian = np.empty((base.size, len(derivative_indices)), dtype=np.float64)

    def predict(trial: Mapping[str, float]) -> np.ndarray | None:
        nonlocal prediction_count
        vertex = [trial["x0"], trial["y0"], trial["z0"]]
        if not detector.contains(vertex):
            return None
        if not (length_limits[0] <= float(trial["length"]) <= length_limits[1]):
            return None
        direction = chart.direction(
            trial.get("dir_u", 0.0), trial.get("dir_v", 0.0)
        )
        if direction is None or not detector.segment_contained(
            vertex, direction, float(trial["length"])
        ):
            return None
        mu, _, _ = predict_charge_and_process_jacobian(
            template_emitter,
            values=trial, chart=chart, wcd=wcd,
            p_locations=p_locations, pmt_normals=pmt_normals,
            obs_pes=obs_pes, mpmt_types=mpmt_types,
            detector=detector,
            need_process_jacobian=False,
        )
        prediction_count += 1
        if mu.size == 0 or not np.all(np.isfinite(mu)):
            return None
        return mu

    for out_col, index in enumerate(derivative_indices):
        name = _LOCAL_TRACK_NAMES[index]
        derivative = None
        for shrink in (1.0, 0.5, 0.25, 0.1):
            h = float(steps[index]) * shrink
            plus = dict(center)
            minus = dict(center)
            plus[name] += h
            minus[name] -= h
            mup = predict(plus)
            mum = predict(minus)
            if mup is not None and mum is not None:
                derivative = (mup - mum) / (2.0 * h)
            elif mup is not None:
                derivative = (mup - base) / h
            elif mum is not None:
                derivative = (base - mum) / h
            if derivative is not None and np.all(np.isfinite(derivative)):
                break
            derivative = None
        if derivative is None:
            raise RuntimeError(f"could not evaluate FE derivative for {name}")
        jacobian[:, out_col] = derivative
    return base, np.ascontiguousarray(jacobian, dtype=np.float64), int(prediction_count)


def apply_local_process_update(
    values: Mapping[str, float],
    raw_delta_block: np.ndarray,
    update_indices: Sequence[int],
    *,
    chart: TangentDirectionChart,
    detector: ConvexDetectorVolume,
    length_limits: tuple[float, float],
    max_tangent_radius: float = 2.0,
) -> tuple[dict[str, float], np.ndarray, float]:
    center = local_track_vector(values)
    full_delta = np.zeros(6, dtype=np.float64)
    for local, index in enumerate(update_indices):
        full_delta[int(index)] = float(raw_delta_block[local])
    alpha = 1.0
    # Length boundary.
    if full_delta[5] > 0.0:
        alpha = min(alpha, max(0.0, (length_limits[1] - center[5]) / full_delta[5]))
    elif full_delta[5] < 0.0:
        alpha = min(alpha, max(0.0, (length_limits[0] - center[5]) / full_delta[5]))
    # Tangent chart radius.  This is a numerical trust boundary, not a physical
    # direction boundary; a re-anchor follows immediately after the update.
    u0, v0 = center[3], center[4]
    du, dv = full_delta[3], full_delta[4]
    if (u0 + alpha * du) ** 2 + (v0 + alpha * dv) ** 2 > max_tangent_radius ** 2:
        A = du * du + dv * dv
        B = 2.0 * (u0 * du + v0 * dv)
        C = u0 * u0 + v0 * v0 - max_tangent_radius ** 2
        if A > 0.0:
            disc = max(B * B - 4.0 * A * C, 0.0)
            roots = [
                x for x in ((-B - math.sqrt(disc)) / (2.0 * A), (-B + math.sqrt(disc)) / (2.0 * A))
                if x >= 0.0
            ]
            if roots:
                alpha = min(alpha, min(roots))
    # Enforce the coupled detector-volume and complete-segment constraint.  A
    # direction or length update can make the downstream endpoint leave the
    # water even when the vertex itself remains inside, so a vertex-only
    # projection is insufficient.  The valid interval is found by bisection
    # along the one physical GEE step direction.
    def physical(scale_value: float) -> bool:
        trial = center + float(scale_value) * full_delta
        direction = chart.direction(float(trial[3]), float(trial[4]))
        return bool(
            direction is not None
            and length_limits[0] <= float(trial[5]) <= length_limits[1]
            and detector.segment_contained(trial[:3], direction, float(trial[5]))
        )

    if not physical(alpha):
        low = 0.0
        high = float(alpha)
        for _ in range(48):
            middle = 0.5 * (low + high)
            if physical(middle):
                low = middle
            else:
                high = middle
        alpha = low
    if alpha < 1.0:
        alpha = max(0.0, alpha * (1.0 - 1.0e-10))
    updated = center + alpha * full_delta
    updated[5] = float(np.clip(updated[5], *length_limits))
    out = {k: float(v) for k, v in values.items()}
    for name, value in zip(_LOCAL_TRACK_NAMES, updated):
        out[name] = float(value)
    return out, full_delta, float(alpha)


def run_generalized_fermi_eyges_update(
    template_emitter,
    *,
    values: Mapping[str, float],
    chart: TangentDirectionChart,
    detector: ConvexDetectorVolume,
    wcd,
    p_locations: np.ndarray,
    pmt_normals: np.ndarray,
    obs_pes: np.ndarray,
    mpmt_types=None,
    update_indices: Sequence[int] = tuple(range(6)),
    process_modes_per_plane: int = 4,
    process_grid_points: int = 41,
    xyz_step_mm: float = 1.0,
    direction_step: float = 2.0e-4,
    length_step_mm: float = 1.0,
    length_limits: tuple[float, float] = (0.0, 3000.0),
    charge_floor_pe: float = 1.0e-4,
) -> dict[str, object]:
    from .mcs_process import fermi_eyges_process_update

    wall0 = time.perf_counter()
    update_indices = tuple(int(i) for i in update_indices)
    mu, track_jacobian, finite_difference_predictions = finite_difference_local_charge_jacobian(
        template_emitter,
        values=values, chart=chart, detector=detector, wcd=wcd,
        p_locations=p_locations, pmt_normals=pmt_normals,
        obs_pes=obs_pes, mpmt_types=mpmt_types,
        derivative_indices=update_indices,
        xyz_step_mm=xyz_step_mm, direction_step=direction_step,
        length_step_mm=length_step_mm, length_limits=length_limits,
    )
    mu_process, process_jacobian, explained = predict_charge_and_process_jacobian(
        template_emitter,
        values=values, chart=chart, wcd=wcd,
        p_locations=p_locations, pmt_normals=pmt_normals,
        obs_pes=obs_pes, mpmt_types=mpmt_types,
        detector=detector,
        need_process_jacobian=True,
        process_modes_per_plane=process_modes_per_plane,
        process_grid_points=process_grid_points,
    )
    if process_jacobian is None:
        raise RuntimeError("Emitter did not return a Fermi--Eyges process Jacobian")
    result = fermi_eyges_process_update(
        mu,
        np.asarray(obs_pes, dtype=np.float64),
        track_jacobian,
        process_jacobian,
        update_indices=tuple(range(len(update_indices))),
        charge_floor_pe=float(charge_floor_pe),
    )
    # Expand selected-coordinate covariance into the six local coordinates.
    naive_small = np.asarray(result["naive_covariance"], dtype=np.float64)
    robust_small = np.asarray(result["robust_covariance"], dtype=np.float64)
    naive = np.full((6, 6), np.nan, dtype=np.float64)
    robust = np.full((6, 6), np.nan, dtype=np.float64)
    for a, ia in enumerate(update_indices):
        for b, ib in enumerate(update_indices):
            naive[ia, ib] = naive_small[a, b]
            robust[ia, ib] = robust_small[a, b]
    updated, raw_delta, physical_scale = apply_local_process_update(
        values,
        np.asarray(result["delta_block"], dtype=np.float64),
        update_indices,
        chart=chart,
        detector=detector,
        length_limits=length_limits,
    )
    updated_before_reanchor = dict(updated)
    updated, updated_chart = reanchor_values(updated, chart)
    # For covariance transformation use the pre-reanchor local point, then map
    # to global direction components.  Also provide an aligned covariance in the
    # final chart for conditional timing.
    robust_global = local_to_cartesian_covariance(
        np.nan_to_num(robust, nan=0.0),
        chart,
        float(updated_before_reanchor.get("dir_u", 0.0)),
        float(updated_before_reanchor.get("dir_v", 0.0)),
    )
    # Pull global covariance into the new chart's six local coordinates.
    map_new = np.zeros((7, 6), dtype=np.float64)
    map_new[:3, :3] = np.eye(3)
    map_new[3:6, 3:5] = updated_chart.direction_jacobian(0.0, 0.0)
    map_new[6, 5] = 1.0
    pinv = np.linalg.pinv(map_new, rcond=1.0e-12)
    robust_new_local = pinv @ robust_global @ pinv.T
    robust_new_local = 0.5 * (robust_new_local + robust_new_local.T)
    return {
        **result,
        "local_parameter_names": list(_LOCAL_TRACK_NAMES),
        "aligned_parameter_names": list(_ALIGNED_TRACK_NAMES),
        "update_indices": update_indices,
        "update_parameter_names": [_LOCAL_TRACK_NAMES[i] for i in update_indices],
        "theta_initial": local_track_vector(values),
        "theta_updated_before_reanchor": local_track_vector(updated_before_reanchor),
        "updated_values": updated,
        "updated_chart": updated_chart,
        "raw_delta_full": raw_delta,
        "physical_step_scale": physical_scale,
        "applied_delta": local_track_vector(updated_before_reanchor) - local_track_vector(values),
        "naive_covariance_local": naive,
        "robust_covariance_local": robust,
        "robust_covariance_global_xyz_dir_length": robust_global,
        "robust_covariance_reanchored_local": robust_new_local,
        "basis_explained_fraction": explained,
        "mean_prediction_max_abs_difference": float(np.max(np.abs(mu_process - mu))),
        "charge_prediction_count": int(finite_difference_predictions + 1),
        "wall_s": float(time.perf_counter() - wall0),
    }
