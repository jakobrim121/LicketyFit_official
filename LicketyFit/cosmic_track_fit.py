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
import math
import os
import time
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from numba import njit

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
    WCTE_MPMT_DOME_OUTER_RADIUS_MM,
    WCTE_MPMT_DOME_CAP_CUT_MM,
    WCTE_MPMT_DOME_CYLINDER_HEIGHT_MM,
)

_FAST_WCTE_GEOMETRY = str(
    os.environ.get("LF_COSMIC_FAST_WCTE_GEOMETRY", "1")
).strip().lower() not in {"0", "false", "no", "off"}


# -----------------------------------------------------------------------------
# Exact compiled WCTE active-water geometry
# -----------------------------------------------------------------------------
# Cosmic fitting evaluates detector containment and first-boundary intersections
# thousands of times per event.  The historical implementation performed the
# 16 prism tests and O(N_mPMT) dome tests through Python/NumPy orchestration on
# every call.  The routines below implement the same inequalities and candidate
# priorities in one Numba kernel.  No search branch or geometric surface is
# removed; the public reference implementation remains available through
# LF_COSMIC_FAST_WCTE_GEOMETRY=0 for bit-level regression checks.

@njit(cache=True)
def _contains_wcte_active_water_numba(
    x,
    axis_lo,
    axis_hi,
    margin,
    prism_n_sides,
    prism_apothem,
    exclusion_centres,
    exclusion_axes,
    exclusion_radius,
    exclusion_cut,
    tolerance,
):
    for k in range(3):
        if x[k] < axis_lo[k] + margin - tolerance:
            return False
        if x[k] > axis_hi[k] - margin + tolerance:
            return False

    limit = prism_apothem - margin
    for j in range(prism_n_sides):
        phi = 2.0 * math.pi * j / prism_n_sides
        if math.cos(phi) * x[0] + math.sin(phi) * x[2] > limit + tolerance:
            return False

    radius2 = exclusion_radius * exclusion_radius
    radial_margin = max(0.0, 2.0 * exclusion_radius * tolerance)
    for i in range(exclusion_centres.shape[0]):
        q0 = x[0] - exclusion_centres[i, 0]
        q1 = x[1] - exclusion_centres[i, 1]
        q2 = x[2] - exclusion_centres[i, 2]
        radial2 = q0 * q0 + q1 * q1 + q2 * q2
        axial = (
            q0 * exclusion_axes[i, 0]
            + q1 * exclusion_axes[i, 1]
            + q2 * exclusion_axes[i, 2]
        )
        if radial2 < radius2 - radial_margin and axial > exclusion_cut + tolerance:
            return False
    return True


@njit(cache=True)
def _ray_exit_hit_wcte_numba(
    x,
    d,
    axis_lo,
    axis_hi,
    margin,
    prism_n_sides,
    prism_apothem,
    exclusion_centres,
    exclusion_axes,
    exclusion_slots,
    exclusion_radius,
    exclusion_cut,
    tolerance,
    check_start,
):
    """Return the same first WCTE active-water surface as ray_exit_hit.

    Return tuple:
      valid, distance, kind_code, piece_code, slot, index, nx, ny, nz

    kind_code: 0=mPMT dome, 1=detector outer
    piece_code: 0=sphere, 1=cap_plane, 2=y_max, 3=y_min, 4=prism_face
    """
    if check_start and not _contains_wcte_active_water_numba(
        x, axis_lo, axis_hi, margin, prism_n_sides, prism_apothem,
        exclusion_centres, exclusion_axes, exclusion_radius, exclusion_cut,
        tolerance,
    ):
        return False, 0.0, -1, -1, -1, -1, 0.0, 0.0, 0.0

    n_excl = exclusion_centres.shape[0]
    max_candidates = 2 + prism_n_sides + 2 * n_excl
    distances = np.full(max_candidates, np.inf, dtype=np.float64)
    priorities = np.full(max_candidates, 1000, dtype=np.int64)
    kinds = np.full(max_candidates, -1, dtype=np.int64)
    pieces = np.full(max_candidates, -1, dtype=np.int64)
    slots = np.full(max_candidates, -1, dtype=np.int64)
    indices = np.full(max_candidates, -1, dtype=np.int64)
    normals = np.zeros((max_candidates, 3), dtype=np.float64)
    count = 0

    def add(value, priority, kind, piece, slot, index, nx, ny, nz):
        nonlocal count
        if not math.isfinite(value) or value < -tolerance:
            return
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if not math.isfinite(norm) or norm <= 0.0:
            return
        distances[count] = max(0.0, value)
        priorities[count] = priority
        kinds[count] = kind
        pieces[count] = piece
        slots[count] = slot
        indices[count] = index
        normals[count, 0] = nx / norm
        normals[count, 1] = ny / norm
        normals[count, 2] = nz / norm
        count += 1

    lo_y = axis_lo[1] + margin
    hi_y = axis_hi[1] - margin
    if d[1] > 1.0e-15:
        add((hi_y - x[1]) / d[1], 20, 1, 2, -1, 1, 0.0, 1.0, 0.0)
    elif d[1] < -1.0e-15:
        add((lo_y - x[1]) / d[1], 20, 1, 3, -1, 1, 0.0, -1.0, 0.0)

    limit = prism_apothem - margin
    for j in range(prism_n_sides):
        phi = 2.0 * math.pi * j / prism_n_sides
        nx = math.cos(phi)
        nz = math.sin(phi)
        velocity = nx * d[0] + nz * d[2]
        if velocity > 1.0e-15:
            value = (limit - (nx * x[0] + nz * x[2])) / velocity
            add(value, 20, 1, 4, -1, j, nx, 0.0, nz)

    radius = exclusion_radius
    radius2 = radius * radius
    disc_tol = max(1.0e-9, 2.0 * radius * tolerance)
    for i in range(n_excl):
        q0 = x[0] - exclusion_centres[i, 0]
        q1 = x[1] - exclusion_centres[i, 1]
        q2 = x[2] - exclusion_centres[i, 2]
        b = q0 * d[0] + q1 * d[1] + q2 * d[2]
        c = q0 * q0 + q1 * q1 + q2 * q2 - radius2
        disc = b * b - c
        if disc >= -disc_tol:
            root = math.sqrt(max(disc, 0.0))
            t = -b - root
            if math.isfinite(t) and t >= -tolerance:
                tt = max(t, 0.0)
                rel0 = q0 + tt * d[0]
                rel1 = q1 + tt * d[1]
                rel2 = q2 + tt * d[2]
                axial = (
                    rel0 * exclusion_axes[i, 0]
                    + rel1 * exclusion_axes[i, 1]
                    + rel2 * exclusion_axes[i, 2]
                )
                deriv = rel0 * d[0] + rel1 * d[1] + rel2 * d[2]
                if axial >= exclusion_cut - tolerance and deriv <= max(1.0e-12, tolerance):
                    relnorm = math.sqrt(rel0 * rel0 + rel1 * rel1 + rel2 * rel2)
                    if relnorm > 0.0:
                        add(
                            t, 0, 0, 0, int(exclusion_slots[i]), i,
                            -rel0 / relnorm, -rel1 / relnorm, -rel2 / relnorm,
                        )

        axial0 = (
            q0 * exclusion_axes[i, 0]
            + q1 * exclusion_axes[i, 1]
            + q2 * exclusion_axes[i, 2]
        )
        axial_v = (
            exclusion_axes[i, 0] * d[0]
            + exclusion_axes[i, 1] * d[1]
            + exclusion_axes[i, 2] * d[2]
        )
        if axial_v > 1.0e-15:
            t = (exclusion_cut - axial0) / axial_v
            if math.isfinite(t) and t >= -tolerance:
                tt = max(t, 0.0)
                rel0 = q0 + tt * d[0]
                rel1 = q1 + tt * d[1]
                rel2 = q2 + tt * d[2]
                radial2 = rel0 * rel0 + rel1 * rel1 + rel2 * rel2
                if radial2 <= radius2 + 2.0 * radius * tolerance:
                    add(
                        t, 1, 0, 1, int(exclusion_slots[i]), i,
                        exclusion_axes[i, 0], exclusion_axes[i, 1], exclusion_axes[i, 2],
                    )

    if count == 0:
        return False, 0.0, -1, -1, -1, -1, 0.0, 0.0, 0.0

    probe = max(1.0e-5, 10.0 * tolerance)
    probe_tol = max(1.0e-10, 0.1 * tolerance)
    # Usually the first chronological candidate is valid.  Exact tangencies and
    # coincident seams are handled by removing an invalid candidate and trying
    # the next, matching the reference loop.
    for _ in range(count):
        best = -1
        best_distance = np.inf
        best_priority = 1000000
        for i in range(count):
            value = distances[i]
            priority = priorities[i]
            if value < best_distance or (value == best_distance and priority < best_priority):
                best = i
                best_distance = value
                best_priority = priority
        if best < 0 or not math.isfinite(best_distance):
            break
        after = np.empty(3, dtype=np.float64)
        after[0] = x[0] + (best_distance + probe) * d[0]
        after[1] = x[1] + (best_distance + probe) * d[1]
        after[2] = x[2] + (best_distance + probe) * d[2]
        if not _contains_wcte_active_water_numba(
            after, axis_lo, axis_hi, margin, prism_n_sides, prism_apothem,
            exclusion_centres, exclusion_axes, exclusion_radius, exclusion_cut,
            probe_tol,
        ):
            return (
                True, best_distance, kinds[best], pieces[best], slots[best],
                indices[best], normals[best, 0], normals[best, 1], normals[best, 2],
            )
        distances[best] = np.inf
    return False, 0.0, -1, -1, -1, -1, 0.0, 0.0, 0.0


@dataclass(frozen=True)
class BoundarySurfaceHit:
    """First intersection of an interior ray with the active-water boundary.

    ``surface_kind`` describes the *interface* crossed by the ray.  For a WCTE
    mPMT dome, ``slot`` is the geometry-package module slot and
    ``surface_piece`` is either ``"sphere"`` or ``"cap_plane"``.  Ordinary
    prism/cap boundaries leave ``slot`` as ``None``.  The normal points out of
    the active-water region, so it is aligned with an exiting ray at a clean
    surface and points into the non-water mPMT exclusion at a dome surface.

    Returning this structure instead of only a distance is what lets cosmic
    mode keep its existing start/stop topology while refining a boundary into
    clean-wall versus mPMT-entry/mPMT-exit subclasses.
    """

    distance_mm: float
    point_mm: np.ndarray
    normal: np.ndarray
    surface_kind: str
    surface_piece: str
    slot: int | None = None
    surface_index: int | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "distance_mm": float(self.distance_mm),
            "point_mm": np.asarray(self.point_mm, dtype=np.float64).tolist(),
            "normal": np.asarray(self.normal, dtype=np.float64).tolist(),
            "surface_kind": str(self.surface_kind),
            "surface_piece": str(self.surface_piece),
            "slot": None if self.slot is None else int(self.slot),
            "surface_index": (
                None if self.surface_index is None else int(self.surface_index)
            ),
        }


@dataclass(frozen=True)
class ConvexDetectorVolume:
    """Detector active-water interior and ray-exit model.

    The outer boundary is convex. Optional boundary-attached cap exclusions
    make the active optical water region locally non-convex while preserving a
    single connected water interval for ordinary detector-crossing lines.

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
    # Optional boundary-attached non-water spherical caps. WCTE mPMT domes
    # are the first use. Each exclusion is the interior of a sphere intersected
    # with (x-centre).axis >= cap_cut.
    exclusion_centres_mm: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    exclusion_axes: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    exclusion_slots: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    exclusion_radius_mm: float = math.nan
    exclusion_cap_cut_mm: float = math.nan

    @classmethod
    def from_wcd(
        cls,
        wcd,
        *,
        placement: str = "design",
        margin_mm: float = 0.0,
        fallback_bounds: Sequence[Sequence[float]] | None = None,
        boundary_model: str = "convex_mpmt_planes",
        include_mpmt_dome_exclusions: bool = False,
        inactive_slots: Iterable[int] | None = None,
    ) -> "ConvexDetectorVolume":
        model = str(boundary_model).strip().lower().replace("-", "_")
        if model not in {"convex_mpmt_planes", "wcte_prism"}:
            raise ValueError(
                "boundary_model must be 'convex_mpmt_planes' or 'wcte_prism'"
            )
        locations: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        slot_ids: list[int] = []
        for slot, mpmt in enumerate(getattr(wcd, "mpmts", [])):
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
                    slot_ids.append(int(slot))
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

        exclusion_centres = np.empty((0, 3), dtype=np.float64)
        exclusion_axes = np.empty((0, 3), dtype=np.float64)
        exclusion_slots = np.empty(0, dtype=np.int32)
        exclusion_radius = math.nan
        exclusion_cut = math.nan
        if bool(include_mpmt_dome_exclusions) and model == "wcte_prism" and loc.size:
            inactive = set() if inactive_slots is None else {int(x) for x in inactive_slots}
            slot_array = np.asarray(slot_ids, dtype=np.int32)
            active = np.asarray([int(x) not in inactive for x in slot_array], dtype=bool)
            active_loc = loc[active]
            active_axes = inward[active]
            # WCSim defines the spherical-cap centre relative to the mPMT base
            # plane. The signed offset is negative because the sphere centre
            # lies behind the cap-cut plane along the inward module axis.
            offset = (
                float(WCTE_MPMT_DOME_CYLINDER_HEIGHT_MM)
                - float(WCTE_MPMT_DOME_CAP_CUT_MM)
            )
            exclusion_centres = np.ascontiguousarray(
                active_loc + offset * active_axes, dtype=np.float64
            )
            exclusion_axes = np.ascontiguousarray(active_axes, dtype=np.float64)
            exclusion_slots = np.ascontiguousarray(slot_array[active], dtype=np.int32)
            exclusion_radius = float(WCTE_MPMT_DOME_OUTER_RADIUS_MM)
            exclusion_cut = float(WCTE_MPMT_DOME_CAP_CUT_MM)

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
            exclusion_centres,
            exclusion_axes,
            exclusion_slots,
            exclusion_radius,
            exclusion_cut,
        )

    def _effective_margin(self, extra_margin_mm: float) -> float:
        return max(float(self.margin_mm) + float(extra_margin_mm), 0.0)

    def _inside_exclusion(
        self, point: np.ndarray, *, tolerance_mm: float = 1.0e-8
    ) -> bool:
        """Return True only when *point* is strictly inside a non-water cap."""
        if self.exclusion_centres_mm.size == 0:
            return False
        radius = float(self.exclusion_radius_mm)
        cut = float(self.exclusion_cap_cut_mm)
        if not (math.isfinite(radius) and radius > 0.0 and math.isfinite(cut)):
            return False
        q = point[None, :] - self.exclusion_centres_mm
        radial2 = np.einsum("ij,ij->i", q, q)
        axial = np.einsum("ij,ij->i", q, self.exclusion_axes)
        radial_margin = max(0.0, 2.0 * radius * float(tolerance_mm))
        return bool(np.any(
            (radial2 < radius * radius - radial_margin)
            & (axial > cut + float(tolerance_mm))
        ))

    def _exclusion_entry_distances(
        self, point: np.ndarray, direction: np.ndarray, *, tolerance_mm: float
    ) -> list[float]:
        """Candidate distances at which a ray first enters a non-water cap."""
        if self.exclusion_centres_mm.size == 0:
            return []
        radius = float(self.exclusion_radius_mm)
        cut = float(self.exclusion_cap_cut_mm)
        if not (math.isfinite(radius) and radius > 0.0 and math.isfinite(cut)):
            return []
        q = point[None, :] - self.exclusion_centres_mm
        b = q @ direction
        c = np.einsum("ij,ij->i", q, q) - radius * radius
        disc = b * b - c
        candidates: list[float] = []
        good = disc >= -max(1.0e-9, 2.0 * radius * float(tolerance_mm))
        if np.any(good):
            root = np.sqrt(np.maximum(disc, 0.0))
            for values in (-b - root, -b + root):
                valid = good & (values >= -float(tolerance_mm)) & np.isfinite(values)
                if not np.any(valid):
                    continue
                hit = point[None, :] + values[:, None] * direction[None, :]
                axial = np.einsum(
                    "ij,ij->i", hit - self.exclusion_centres_mm, self.exclusion_axes
                )
                valid &= axial >= cut - float(tolerance_mm)
                for value in values[valid]:
                    candidates.append(max(0.0, float(value)))

        # Retain the cap-plane disk intersection for generic cap geometries.
        axial0 = np.einsum("ij,ij->i", q, self.exclusion_axes)
        axial_v = self.exclusion_axes @ direction
        entering = axial_v > 1.0e-15
        plane_t = np.divide(
            cut - axial0,
            axial_v,
            out=np.full_like(axial0, np.inf),
            where=entering,
        )
        valid = entering & (plane_t >= -float(tolerance_mm)) & np.isfinite(plane_t)
        if np.any(valid):
            hit = point[None, :] + plane_t[:, None] * direction[None, :]
            radial2 = np.einsum(
                "ij,ij->i", hit - self.exclusion_centres_mm,
                hit - self.exclusion_centres_mm,
            )
            valid &= radial2 <= radius * radius + 2.0 * radius * float(tolerance_mm)
            for value in plane_t[valid]:
                candidates.append(max(0.0, float(value)))
        return candidates

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
        if self.boundary_model == "wcte_prism" and _FAST_WCTE_GEOMETRY:
            # The compiled kernel already performs the axis, prism and mPMT
            # exclusion tests.  Avoid duplicating the axis test and repeatedly
            # wrapping detector-static contiguous arrays in the Python hot path.
            return bool(_contains_wcte_active_water_numba(
                x, self.axis_lo, self.axis_hi, float(margin),
                int(self.prism_n_sides), float(self.prism_apothem_mm),
                self.exclusion_centres_mm, self.exclusion_axes,
                float(self.exclusion_radius_mm),
                float(self.exclusion_cap_cut_mm), float(tolerance_mm),
            ))
        if np.any(x < self.axis_lo + margin - tolerance_mm) or np.any(
            x > self.axis_hi - margin + tolerance_mm
        ):
            return False
        if self.boundary_model == "wcte_prism":
            limit = float(self.prism_apothem_mm) - margin
            for j in range(int(self.prism_n_sides)):
                phi = 2.0 * math.pi * j / float(self.prism_n_sides)
                if (
                    math.cos(phi) * float(x[0])
                    + math.sin(phi) * float(x[2])
                    > limit + tolerance_mm
                ):
                    return False
            if self._inside_exclusion(x, tolerance_mm=tolerance_mm):
                return False
            return True
        if self.locations.size:
            distances = np.einsum("ij,ij->i", x[None, :] - self.locations, self.inward_axes)
            if float(np.min(distances)) < margin - tolerance_mm:
                return False
        if self._inside_exclusion(x, tolerance_mm=tolerance_mm):
            return False
        return True

    def ray_exit_hit(
        self,
        point: Sequence[float],
        direction: Sequence[float],
        *,
        extra_margin_mm: float = 0.0,
        tolerance_mm: float = 1.0e-8,
        _assume_inside: bool = False,
    ) -> BoundarySurfaceHit | None:
        """Return the first forward active-water boundary intersection.

        The input point must lie inside the requested shrunken volume. ``None``
        is returned for invalid inputs or for a mathematically unbounded ray.

        This routine is deliberately linear in the number of detector modules.
        Earlier metadata support generated every possible surface hit and then
        called :meth:`contains` for every candidate.  With O(100) WCTE domes
        that made proxy-cache construction effectively quadratic.  Starting in
        active water lets us classify the *entering* root of each excluded cap
        analytically, so only the chronologically earliest candidate normally
        needs one forward-probe containment check.
        """
        x = np.asarray(point, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        if x.shape != (3,) or d.shape != (3,):
            return None
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(d))):
            return None
        norm = float(np.linalg.norm(d))
        if norm <= 0.0:
            return None
        d = np.ascontiguousarray(d / norm, dtype=np.float64)
        margin = self._effective_margin(extra_margin_mm)
        if self.boundary_model == "wcte_prism" and _FAST_WCTE_GEOMETRY:
            result = _ray_exit_hit_wcte_numba(
                x, d, self.axis_lo, self.axis_hi, float(margin),
                int(self.prism_n_sides), float(self.prism_apothem_mm),
                self.exclusion_centres_mm, self.exclusion_axes,
                self.exclusion_slots,
                float(self.exclusion_radius_mm),
                float(self.exclusion_cap_cut_mm),
                float(tolerance_mm),
                bool(not _assume_inside),
            )
            valid, distance, kind_code, piece_code, slot, index, nx, ny, nz = result
            if not valid:
                return None
            kind = "mpmt_dome" if int(kind_code) == 0 else "detector_outer"
            pieces = ("sphere", "cap_plane", "y_max", "y_min", "prism_face")
            piece = pieces[int(piece_code)]
            return BoundarySurfaceHit(
                distance_mm=float(distance),
                point_mm=np.ascontiguousarray(x + float(distance) * d, dtype=np.float64),
                normal=np.ascontiguousarray([nx, ny, nz], dtype=np.float64),
                surface_kind=kind,
                surface_piece=piece,
                slot=None if int(slot) < 0 else int(slot),
                surface_index=None if int(index) < 0 else int(index),
            )
        if (not _assume_inside) and not self.contains(
            x, tolerance_mm=tolerance_mm, extra_margin_mm=extra_margin_mm
        ):
            return None

        # Candidate tuple:
        # (distance, priority, kind, piece, slot, index, outward-water-normal)
        candidates: list[
            tuple[float, int, str, str, int | None, int | None, np.ndarray]
        ] = []

        def add_candidate(
            value: float,
            *,
            priority: int,
            kind: str,
            piece: str,
            normal: Sequence[float],
            slot: int | None = None,
            index: int | None = None,
        ) -> None:
            value = float(value)
            normal_array = np.asarray(normal, dtype=np.float64)
            normal_norm = float(np.linalg.norm(normal_array))
            if (
                not math.isfinite(value)
                or value < -float(tolerance_mm)
                or normal_array.shape != (3,)
                or not np.all(np.isfinite(normal_array))
                or normal_norm <= 0.0
            ):
                return
            candidates.append((
                max(0.0, value), int(priority), str(kind), str(piece),
                None if slot is None else int(slot),
                None if index is None else int(index),
                np.ascontiguousarray(normal_array / normal_norm, dtype=np.float64),
            ))

        lo = self.axis_lo + margin
        hi = self.axis_hi - margin
        if self.boundary_model == "wcte_prism":
            if d[1] > 1.0e-15:
                add_candidate(
                    (hi[1] - x[1]) / d[1], priority=20,
                    kind="detector_outer", piece="y_max",
                    normal=(0.0, 1.0, 0.0), index=1,
                )
            elif d[1] < -1.0e-15:
                add_candidate(
                    (lo[1] - x[1]) / d[1], priority=20,
                    kind="detector_outer", piece="y_min",
                    normal=(0.0, -1.0, 0.0), index=1,
                )

            limit = float(self.prism_apothem_mm) - margin
            for j in range(int(self.prism_n_sides)):
                phi = 2.0 * math.pi * j / float(self.prism_n_sides)
                normal = np.asarray(
                    [math.cos(phi), 0.0, math.sin(phi)], dtype=np.float64
                )
                velocity = float(np.dot(normal, d))
                if velocity > 1.0e-15:
                    add_candidate(
                        (limit - float(np.dot(normal, x))) / velocity,
                        priority=20, kind="detector_outer", piece="prism_face",
                        normal=normal, index=j,
                    )
        else:
            for k in range(3):
                if d[k] > 1.0e-15:
                    normal = np.zeros(3, dtype=np.float64)
                    normal[k] = 1.0
                    add_candidate(
                        (hi[k] - x[k]) / d[k], priority=30,
                        kind="detector_outer", piece=f"axis_{k}_max",
                        normal=normal, index=k,
                    )
                elif d[k] < -1.0e-15:
                    normal = np.zeros(3, dtype=np.float64)
                    normal[k] = -1.0
                    add_candidate(
                        (lo[k] - x[k]) / d[k], priority=30,
                        kind="detector_outer", piece=f"axis_{k}_min",
                        normal=normal, index=k,
                    )

            if self.locations.size:
                current = np.einsum(
                    "ij,ij->i", x[None, :] - self.locations, self.inward_axes
                )
                velocity = self.inward_axes @ d
                for index in np.flatnonzero(velocity < -1.0e-15):
                    add_candidate(
                        (float(current[index]) - margin)
                        / (-float(velocity[index])),
                        priority=10, kind="detector_outer",
                        piece="convex_mpmt_plane",
                        normal=-self.inward_axes[index], index=int(index),
                    )

        # Entering intersections with boundary-attached excluded spherical caps.
        # Because x starts outside every cap, only the smaller sphere root can
        # enter a sphere.  The larger root exits it and must never be considered
        # an active-water exit.  A separate cap-plane candidate covers rays that
        # start inside the parent sphere but below the cap cut.
        if self.exclusion_centres_mm.size:
            radius = float(self.exclusion_radius_mm)
            cut = float(self.exclusion_cap_cut_mm)
            if math.isfinite(radius) and radius > 0.0 and math.isfinite(cut):
                centres = np.asarray(self.exclusion_centres_mm, dtype=np.float64)
                axes = np.asarray(self.exclusion_axes, dtype=np.float64)
                slots = np.asarray(self.exclusion_slots, dtype=np.int32)
                q = x[None, :] - centres
                b = q @ d
                c = np.einsum("ij,ij->i", q, q) - radius * radius
                disc = b * b - c
                disc_tol = max(1.0e-9, 2.0 * radius * float(tolerance_mm))
                good = disc >= -disc_tol
                root = np.sqrt(np.maximum(disc, 0.0))
                sphere_t = -b - root
                valid = good & np.isfinite(sphere_t) & (
                    sphere_t >= -float(tolerance_mm)
                )
                if np.any(valid):
                    hit = x[None, :] + np.maximum(sphere_t, 0.0)[:, None] * d[None, :]
                    rel = hit - centres
                    axial = np.einsum("ij,ij->i", rel, axes)
                    # Strictly entering roots have rel.d < 0.  Keep tangencies
                    # within tolerance and let the one forward probe reject them.
                    deriv = rel @ d
                    valid &= axial >= cut - float(tolerance_mm)
                    valid &= deriv <= max(1.0e-12, float(tolerance_mm))
                    for index in np.flatnonzero(valid):
                        rel_i = rel[index]
                        rel_norm = float(np.linalg.norm(rel_i))
                        if rel_norm <= 0.0:
                            continue
                        add_candidate(
                            float(sphere_t[index]), priority=0,
                            kind="mpmt_dome", piece="sphere",
                            normal=-rel_i / rel_norm,
                            slot=int(slots[index]), index=int(index),
                        )

                axial0 = np.einsum("ij,ij->i", q, axes)
                axial_v = axes @ d
                plane_t = np.divide(
                    cut - axial0,
                    axial_v,
                    out=np.full_like(axial0, np.inf),
                    where=axial_v > 1.0e-15,
                )
                valid = (
                    (axial_v > 1.0e-15)
                    & np.isfinite(plane_t)
                    & (plane_t >= -float(tolerance_mm))
                )
                if np.any(valid):
                    hit = x[None, :] + np.maximum(plane_t, 0.0)[:, None] * d[None, :]
                    rel = hit - centres
                    radial2 = np.einsum("ij,ij->i", rel, rel)
                    valid &= radial2 <= (
                        radius * radius + 2.0 * radius * float(tolerance_mm)
                    )
                    for index in np.flatnonzero(valid):
                        add_candidate(
                            float(plane_t[index]), priority=1,
                            kind="mpmt_dome", piece="cap_plane",
                            normal=axes[index], slot=int(slots[index]),
                            index=int(index),
                        )

        if not candidates:
            return None

        candidates.sort(key=lambda row: (row[0], row[1]))
        probe = max(1.0e-5, 10.0 * float(tolerance_mm))
        probe_tol = max(1.0e-10, 0.1 * float(tolerance_mm))
        # In ordinary geometry the first candidate is valid.  The loop only
        # matters for exact tangencies or coincident seam surfaces.
        for distance, _, kind, piece, slot, index, normal in candidates:
            after = x + (distance + probe) * d
            if self.contains(
                after,
                tolerance_mm=probe_tol,
                extra_margin_mm=extra_margin_mm,
            ):
                continue
            return BoundarySurfaceHit(
                distance_mm=float(distance),
                point_mm=np.ascontiguousarray(x + distance * d, dtype=np.float64),
                normal=np.ascontiguousarray(normal, dtype=np.float64),
                surface_kind=str(kind),
                surface_piece=str(piece),
                slot=slot,
                surface_index=index,
            )
        return None

    def ray_exit_distance(
        self,
        point: Sequence[float],
        direction: Sequence[float],
        *,
        extra_margin_mm: float = 0.0,
        tolerance_mm: float = 1.0e-8,
    ) -> float:
        """Return the forward distance to the first detector boundary [mm].

        This preserves the historical scalar API.  Invalid inputs return NaN;
        a valid but unbounded ray returns infinity.
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
        if not self.contains(
            x, tolerance_mm=tolerance_mm, extra_margin_mm=extra_margin_mm
        ):
            return math.nan
        hit = self.ray_exit_hit(
            x,
            d / norm,
            extra_margin_mm=extra_margin_mm,
            tolerance_mm=tolerance_mm,
            _assume_inside=True,
        )
        return math.inf if hit is None else float(hit.distance_mm)

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
            "nonwater_cap_exclusions": {
                "count": int(self.exclusion_centres_mm.shape[0]),
                "slots": np.asarray(self.exclusion_slots, dtype=np.int32).tolist(),
                "radius_mm": (
                    float(self.exclusion_radius_mm)
                    if math.isfinite(float(self.exclusion_radius_mm)) else None
                ),
                "cap_cut_mm": (
                    float(self.exclusion_cap_cut_mm)
                    if math.isfinite(float(self.exclusion_cap_cut_mm)) else None
                ),
            },
        }


@dataclass
class FitEvaluation:
    fval: float
    values: dict[str, float]


@dataclass(frozen=True)
class BoundaryClippedTrack:
    """Canonical detector chord for a through-going line hypothesis.

    ``reference`` is any interior point on the fitted line.  Translation of
    that point along the line does not change the physical chord; the returned
    entry/exit points therefore remove the otherwise exact longitudinal/t0
    degeneracy of a through-going track.
    """

    reference: np.ndarray
    entry: np.ndarray
    exit: np.ndarray
    direction: np.ndarray
    visible_length_mm: float
    backward_distance_mm: float
    forward_distance_mm: float
    inset_mm: float
    entry_hit: BoundarySurfaceHit
    exit_hit: BoundarySurfaceHit


def resolve_boundary_clipped_track(
    detector: ConvexDetectorVolume,
    reference: Sequence[float],
    direction: Sequence[float],
    *,
    inset_mm: float = 0.5,
    tolerance_mm: float = 1.0e-6,
) -> BoundaryClippedTrack | None:
    """Clip an infinite oriented line to a convex detector volume.

    The source point is placed a small distance inside the upstream boundary
    and the endpoint the same distance inside the downstream boundary.  The
    inset avoids floating-point rejection at a half-space boundary while being
    negligible compared with detector/PMT scales.  The returned direction keeps
    its supplied orientation, so swapping ``d`` for ``-d`` swaps entry and exit.
    """
    point = np.asarray(reference, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    if point.shape != (3,) or d.shape != (3,):
        return None
    if not (np.all(np.isfinite(point)) and np.all(np.isfinite(d))):
        return None
    norm = float(np.linalg.norm(d))
    if norm <= 0.0:
        return None
    d = np.ascontiguousarray(d / norm, dtype=np.float64)
    # The first ray call performs the same start containment test internally;
    # the forward call can then safely assume the point is in active water.
    backward_hit = detector.ray_exit_hit(
        point, -d, tolerance_mm=tolerance_mm, _assume_inside=False
    )
    forward_hit = detector.ray_exit_hit(
        point, d, tolerance_mm=tolerance_mm, _assume_inside=True
    )
    if backward_hit is None or forward_hit is None:
        return None
    backward = float(backward_hit.distance_mm)
    forward = float(forward_hit.distance_mm)
    if not (math.isfinite(backward) and math.isfinite(forward)):
        return None
    chord = backward + forward
    if chord <= 0.0:
        return None
    inset = max(0.0, float(inset_mm))
    # Leave a positive modeled segment even for a pathological grazing line.
    inset = min(inset, max(0.0, 0.5 * chord - 1.0e-6))
    length = chord - 2.0 * inset
    if not math.isfinite(length) or length <= 0.0:
        return None
    entry = point - (backward - inset) * d
    exit_point = point + (forward - inset) * d
    # The two points are inset from the first boundary reached in each
    # direction.  Every point between them is therefore in the same connected
    # active-water interval; re-testing both endpoints is redundant.
    return BoundaryClippedTrack(
        reference=np.ascontiguousarray(point, dtype=np.float64),
        entry=np.ascontiguousarray(entry, dtype=np.float64),
        exit=np.ascontiguousarray(exit_point, dtype=np.float64),
        direction=d,
        visible_length_mm=float(length),
        backward_distance_mm=float(backward),
        forward_distance_mm=float(forward),
        inset_mm=float(inset),
        entry_hit=backward_hit,
        exit_hit=forward_hit,
    )


@dataclass(frozen=True)
class RangeClippedTrack:
    """Finite-range track clipped to the active detector volume.

    ``full_range_mm`` is the fitted water-equivalent CSDA range remaining to
    Cherenkov threshold at ``start``.  ``visible_length_mm`` is only the part
    of that range that lies in the active detector.  The construction therefore
    never predicts light outside the detector, while the in-detector light yield,
    Cherenkov angle and particle timing still depend on the fitted entry/start
    energy.
    """

    reference: np.ndarray
    start: np.ndarray
    endpoint: np.ndarray
    boundary_exit: np.ndarray
    direction: np.ndarray
    full_range_mm: float
    visible_length_mm: float
    distance_to_boundary_mm: float
    starts_at_boundary: bool
    exits_detector: bool
    topology: str
    inset_mm: float
    backward_distance_mm: float
    start_boundary_hit: BoundarySurfaceHit | None
    exit_boundary_hit: BoundarySurfaceHit


def resolve_range_clipped_track(
    detector: ConvexDetectorVolume,
    reference: Sequence[float],
    direction: Sequence[float],
    full_range_mm: float,
    *,
    starts_at_boundary: bool,
    inset_mm: float = 0.5,
    tolerance_mm: float = 1.0e-6,
) -> RangeClippedTrack | None:
    """Resolve any start/stop versus boundary-crossing topology.

    For an internal-start hypothesis, ``reference`` is the physical light-onset
    point.  For a boundary-entry hypothesis it is any interior point on the
    oriented line; the canonical upstream water entry is derived by clipping the
    infinite line.  In both cases the fitted range is independent of the visible
    detector segment.
    """
    point = np.asarray(reference, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    full_range = float(full_range_mm)
    if point.shape != (3,) or d.shape != (3,):
        return None
    if not (np.all(np.isfinite(point)) and np.all(np.isfinite(d))):
        return None
    if not math.isfinite(full_range) or full_range <= 0.0:
        return None
    norm = float(np.linalg.norm(d))
    if norm <= 0.0:
        return None
    d = np.ascontiguousarray(d / norm, dtype=np.float64)
    inset = max(0.0, float(inset_mm))

    backward_distance = 0.0
    start_boundary_hit: BoundarySurfaceHit | None = None
    if bool(starts_at_boundary):
        chord = resolve_boundary_clipped_track(
            detector, point, d, inset_mm=inset, tolerance_mm=tolerance_mm
        )
        if chord is None:
            return None
        start = chord.entry
        boundary_exit = chord.exit
        distance_to_boundary = float(chord.visible_length_mm)
        backward_distance = float(chord.backward_distance_mm - chord.inset_mm)
        inset = float(chord.inset_mm)
        start_boundary_hit = chord.entry_hit
        exit_boundary_hit = chord.exit_hit
    else:
        start = np.ascontiguousarray(point, dtype=np.float64)
        forward_hit = detector.ray_exit_hit(
            start, d, tolerance_mm=tolerance_mm, _assume_inside=False
        )
        if forward_hit is None:
            return None
        forward = float(forward_hit.distance_mm)
        if not math.isfinite(forward) or forward <= 0.0:
            return None
        # Keep the emitted endpoint infinitesimally inside the convex volume.
        effective_inset = min(inset, max(0.0, forward - 1.0e-6))
        distance_to_boundary = float(forward - effective_inset)
        if distance_to_boundary <= 0.0:
            return None
        boundary_exit = start + distance_to_boundary * d
        inset = float(effective_inset)
        exit_boundary_hit = forward_hit

    # A range that reaches the inset downstream surface is an exiting track.
    # The tolerance prevents unstable topology flips from roundoff at equality.
    exits = bool(full_range >= distance_to_boundary - max(tolerance_mm, 1.0e-9))
    visible = float(distance_to_boundary if exits else full_range)
    endpoint = start + visible * d
    if visible <= 0.0 or not math.isfinite(visible):
        return None
    # ``boundary_exit`` is the first forward boundary and ``endpoint`` lies on
    # the active-water segment before it.  The start was validated by the ray
    # intersection used above.  Additional contains() calls here repeat the
    # same exact geometry without adding a physical constraint.

    if bool(starts_at_boundary):
        topology = (
            "boundary_entry_boundary_exit" if exits
            else "boundary_entry_internal_stop"
        )
    else:
        topology = (
            "internal_start_boundary_exit" if exits
            else "internal_start_internal_stop"
        )
    return RangeClippedTrack(
        reference=np.ascontiguousarray(point, dtype=np.float64),
        start=np.ascontiguousarray(start, dtype=np.float64),
        endpoint=np.ascontiguousarray(endpoint, dtype=np.float64),
        boundary_exit=np.ascontiguousarray(boundary_exit, dtype=np.float64),
        direction=d,
        full_range_mm=float(full_range),
        visible_length_mm=float(visible),
        distance_to_boundary_mm=float(distance_to_boundary),
        starts_at_boundary=bool(starts_at_boundary),
        exits_detector=bool(exits),
        topology=str(topology),
        inset_mm=float(inset),
        backward_distance_mm=float(backward_distance),
        start_boundary_hit=start_boundary_hit,
        exit_boundary_hit=exit_boundary_hit,
    )


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
    boundary_clip_track: bool = False
    boundary_clip_inset_mm: float = 0.5
    range_clip_track: bool = False
    boundary_entry_track: bool = False
    boundary_interface_model: object | None = None
    boundary_interface: str | None = None
    required_boundary_slot: int | None = None
    boundary_interface_timing_policy: str = "augment"
    # Per-objective caches are deliberately bounded.  Cosmic navigation creates
    # several short-lived objectives per event, and an unconstrained prediction
    # cache retains large first-arrival arrays for every trial geometry.  The
    # limits below preserve the useful t0/nearby-point reuse while guaranteeing
    # bounded memory in long supervised jobs.
    max_cache_entries: int = 4096
    max_prediction_cache_entries: int = 128
    max_boundary_diagnostic_entries: int = 64
    cache: dict[tuple[float, ...], float] = field(default_factory=dict)
    prediction_cache: dict[
        tuple[float, ...], tuple[np.ndarray, object, np.ndarray | None] | None
    ] = field(default_factory=dict)
    calls: int = 0
    evaluations: int = 0
    optical_evaluations: int = 0
    invalid_evaluations: int = 0
    boundary_interface_diagnostics: dict[tuple[float, ...], dict[str, object]] = field(
        default_factory=dict
    )
    last_boundary_interface_error: str | None = None

    @staticmethod
    def _bounded_put(mapping: dict, key, value, limit: int) -> None:
        """Insert into an insertion-ordered dict with a strict FIFO bound."""
        if key in mapping:
            mapping.pop(key, None)
        mapping[key] = value
        limit = max(int(limit), 1)
        while len(mapping) > limit:
            try:
                mapping.pop(next(iter(mapping)))
            except StopIteration:
                break

    def _store_nll(self, key: tuple[float, ...], value: float) -> None:
        self._bounded_put(self.cache, key, float(value), self.max_cache_entries)

    def _store_prediction(self, key: tuple[float, ...], value) -> None:
        self._bounded_put(
            self.prediction_cache, key, value, self.max_prediction_cache_entries
        )

    def _store_boundary_diagnostic(
        self, key: tuple[float, ...], value: dict[str, object]
    ) -> None:
        self._bounded_put(
            self.boundary_interface_diagnostics, key, value,
            self.max_boundary_diagnostic_entries,
        )

    def clear_caches(self) -> None:
        """Release all event/geometry arrays retained by this objective."""
        self.cache.clear()
        self.prediction_cache.clear()
        self.boundary_interface_diagnostics.clear()
        self.last_boundary_interface_error = None

    def cache_sizes(self) -> dict[str, int]:
        return {
            "nll": int(len(self.cache)),
            "prediction": int(len(self.prediction_cache)),
            "boundary_diagnostics": int(len(self.boundary_interface_diagnostics)),
        }

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

    def _prediction_nll(self, prediction, t0: float) -> float:
        """Evaluate the configured likelihood for an already-built prediction.

        This small helper is intentionally independent of the mutable Emitter
        state.  It lets t0-only probes reuse a cached optical prediction before
        performing any detector clipping, range conversion, or source setup.
        The returned value is mathematically identical to the historical tail
        of :meth:`__call__`.
        """
        exp_pes_array, exp_ts_zero, timing_pes = prediction
        if self.objective_mode == "charge_only":
            fval = self.pmt_model.get_neg_log_likelihood_npe(
                exp_pes_array, self.obs_pes
            )
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
        """Evaluate exact NLLs for many t0 values using one prediction.

        This is a narrow acceleration for the additive event-time coordinate.
        It does not approximate or alter the likelihood.  The production
        first-arrival timing kernel sorts and compresses source nodes once per
        PMT, then evaluates all requested shifts in one compiled call.  Cache
        and evaluation counters retain their scalar-call interpretation.
        """
        shifts = np.ascontiguousarray(t0_values, dtype=np.float64).reshape(-1)
        out = np.full(shifts.size, np.inf, dtype=np.float64)
        if shifts.size == 0:
            return out

        base = {k: float(v) for k, v in values.items()}
        geometry_key = self._geometry_key(base)
        # A normal scalar evaluation creates and validates the geometry
        # prediction.  profile_t0 always evaluates its base point first, but
        # retain a safe compatibility path for direct callers.
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
                self._store_nll(key, np.inf)
                continue
            if self.t0_limits is not None:
                lo, hi = (float(self.t0_limits[0]), float(self.t0_limits[1]))
                if shift < lo or shift > hi:
                    self.invalid_evaluations += 1
                    self._store_nll(key, np.inf)
                    continue
            pending_indices.append(i)
            pending_values.append(shift)
            pending_keys.append(key)

        if not pending_indices:
            return out

        exp_pes_array, exp_ts_zero, timing_pes = prediction
        shift_array = np.ascontiguousarray(pending_values, dtype=np.float64)
        if (
            self.objective_mode == "charge_time"
            and hasattr(self.pmt_model, "get_neg_log_likelihood_npe_t_many_t0")
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
                [self._prediction_nll(prediction, float(v)) for v in shift_array],
                dtype=np.float64,
            )

        values_array = np.asarray(values_array, dtype=np.float64).reshape(-1)
        if values_array.size != len(pending_indices):
            raise RuntimeError("batched t0 likelihood returned the wrong size")
        if (
            self.use_t0_prior
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
            self._store_nll(key, fval)
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
            self._store_nll(key, np.inf)
            return np.inf

        # t0 does not affect detector clipping, particle kinematics, charge, or
        # source-relative timing.  Check it first, then answer a t0-only probe
        # directly from the geometry prediction cache.  Previously every t0
        # stencil repeated the full active-water intersection and range setup
        # before reaching the same cache entry.
        t0 = float(values.get("t0", 0.0))
        if not math.isfinite(t0):
            return invalid()
        if self.t0_limits is not None:
            lo, hi = (float(self.t0_limits[0]), float(self.t0_limits[1]))
            if t0 < lo or t0 > hi:
                return invalid()

        geometry_key = self._geometry_key(values)
        if geometry_key in self.prediction_cache:
            prediction = self.prediction_cache[geometry_key]
            if prediction is None:
                return invalid()
            fval = self._prediction_nll(prediction, t0)
            if not math.isfinite(fval):
                return invalid()
            self._store_nll(key, fval)
            return float(fval)

        reference_vertex = np.asarray(
            [values["x0"], values["y0"], values["z0"]], dtype=np.float64
        )
        direction = self.chart.direction(
            values.get("dir_u", 0.0), values.get("dir_v", 0.0)
        )
        if direction is None:
            return invalid()
        # The clipping resolvers validate the reference point themselves.  Do
        # not perform the same exact containment test twice for cosmic tracks.
        if (
            not self.range_clip_track
            and not self.boundary_clip_track
            and not self.detector.contains(reference_vertex)
        ):
            return invalid()
        clipped = None
        range_clipped = None
        if self.range_clip_track:
            if self.track_end_mode != "full_length":
                return invalid()
            range_clipped = resolve_range_clipped_track(
                self.detector,
                reference_vertex,
                direction,
                float(values.get("length", math.nan)),
                starts_at_boundary=bool(self.boundary_entry_track),
                inset_mm=float(self.boundary_clip_inset_mm),
                tolerance_mm=float(self.containment_tolerance_mm),
            )
            if range_clipped is None:
                return invalid()
            vertex = range_clipped.start
            direction = range_clipped.direction

            if self.boundary_interface_model is not None:
                if hasattr(self.boundary_interface_model, "validate_track"):
                    try:
                        if not bool(self.boundary_interface_model.validate_track(range_clipped)):
                            return invalid()
                    except Exception as exc:
                        self.last_boundary_interface_error = repr(exc)
                        return invalid()
                else:
                    interface = str(self.boundary_interface or "").strip().lower()
                    if interface == "entry":
                        hit = range_clipped.start_boundary_hit
                        if not range_clipped.starts_at_boundary:
                            return invalid()
                    elif interface == "exit":
                        hit = (
                            range_clipped.exit_boundary_hit
                            if range_clipped.exits_detector else None
                        )
                    else:
                        return invalid()
                    if hit is None or hit.surface_kind != "mpmt_dome":
                        return invalid()
                    if (
                        self.required_boundary_slot is not None
                        and hit.slot != int(self.required_boundary_slot)
                    ):
                        return invalid()
        elif self.boundary_clip_track:
            if self.track_end_mode != "absorption":
                return invalid()
            clipped = resolve_boundary_clipped_track(
                self.detector,
                reference_vertex,
                direction,
                inset_mm=float(self.boundary_clip_inset_mm),
                tolerance_mm=float(self.containment_tolerance_mm),
            )
            if clipped is None:
                return invalid()
            vertex = clipped.entry
            direction = clipped.direction
        else:
            vertex = reference_vertex

        if self.range_clip_track:
            assert range_clipped is not None
            full_range = float(range_clipped.full_range_mm)
            length = float(range_clipped.visible_length_mm)
            if self.range_lookup is None:
                return invalid()
            max_range = float(self.range_lookup.overall_distances_mm[-1])
            if full_range > max_range:
                return invalid()
            ke0 = float(self.range_lookup.range_mm_to_energy(full_range))
            if not math.isfinite(ke0) or ke0 <= float(self.particle_threshold_mev):
                return invalid()
            self.emitter.track_end_mode = "abrupt"
            self.emitter.fixed_initial_KE = ke0
        elif self.track_end_mode == "absorption":
            visible = (
                float(clipped.visible_length_mm)
                if clipped is not None
                else float(values["visible_length"])
            )
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
            self.emitter.track_end_mode = "abrupt"
            self.emitter.fixed_initial_KE = ke0
            length = visible
        else:
            length = float(values["length"])
            if not math.isfinite(length) or length < 0.0:
                return invalid()
            if self.range_lookup is not None:
                if length > float(self.range_lookup.overall_distances_mm[-1]):
                    return invalid()
            self.emitter.track_end_mode = "threshold"
            self.emitter.fixed_initial_KE = None

        # Both clipping resolvers return a segment whose first boundary is the
        # active-water surface and explicitly validate their inset endpoints.
        # Re-running ray_exit_distance here is therefore an exact duplicate.
        if (
            self.require_contained_track
            and not self.range_clip_track
            and not self.boundary_clip_track
            and not self.detector.segment_contained(
                vertex,
                direction,
                length,
                tolerance_mm=float(self.containment_tolerance_mm),
            )
        ):
            return invalid()

        need_times = self.objective_mode != "charge_only"
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
                    self._store_prediction(geometry_key, None)
                    return invalid()
            elif getattr(self.emitter, "last_visible_length_exceeds_range", False):
                self._store_prediction(geometry_key, None)
                return invalid()

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
                self._store_prediction(geometry_key, None)
                return invalid()

            if self.boundary_interface_model is not None:
                assert range_clipped is not None
                interface = str(self.boundary_interface).strip().lower()
                boundary_hit = (
                    range_clipped.start_boundary_hit
                    if interface == "entry"
                    else range_clipped.exit_boundary_hit
                )
                if boundary_hit is None:
                    self._store_prediction(geometry_key, None)
                    return invalid()
                water_path_to_boundary = (
                    0.0
                    if interface == "entry"
                    else float(
                        range_clipped.distance_to_boundary_mm
                        + range_clipped.inset_mm
                    )
                )
                try:
                    if hasattr(self.boundary_interface_model, "predict_track_modes"):
                        local_modes = self.boundary_interface_model.predict_track_modes(
                            track=range_clipped,
                            direction=direction,
                            kinetic_energy_at_water_entry_mev=float(ke0),
                            range_lookup=self.range_lookup,
                            emitter=self.emitter,
                        )
                        local_profile = self.boundary_interface_model.profile_charge(
                            exp_pes_array, self.obs_pes, local_modes
                        )
                        exp_pes_array = np.asarray(
                            local_profile.expected_pes, dtype=np.float64
                        )
                        timing_policy = str(
                            self.boundary_interface_timing_policy or "augment"
                        ).strip().lower()
                        if need_times and timing_policy == "augment":
                            exp_ts_zero = self.boundary_interface_model.augment_timing_prediction(
                                exp_ts_zero,
                                modes=local_modes,
                                fractions=local_profile.fractions,
                                observed_pes=self.obs_pes,
                            )
                        elif timing_policy not in {"augment", "mask_module", "baseline"}:
                            raise ValueError(
                                "boundary_interface_timing_policy must be augment, "
                                "mask_module, or baseline"
                            )
                        # The timing PE marginal must follow the charge redistribution.
                        # Under mask_module, crossed-module timestamps are removed by
                        # the caller while all unaffected PMTs retain ordinary timing.
                        timing_array = np.ascontiguousarray(
                            exp_pes_array, dtype=np.float64
                        )
                        self._store_boundary_diagnostic(geometry_key, {
                            "timing_policy": timing_policy,
                            "fractions": np.asarray(
                                local_profile.fractions, dtype=np.float64
                            ).tolist(),
                            "mode_names": list(local_modes.mode_names),
                            "charge_nll_improvement": float(
                                local_profile.improvement
                            ),
                            "profile_iterations": int(local_profile.iterations),
                            "profile_converged": bool(local_profile.converged),
                            "model": dict(local_modes.diagnostics),
                            "composite": (
                                self.boundary_interface_model.metadata()
                                if hasattr(self.boundary_interface_model, "metadata")
                                else None
                            ),
                        })
                    # Research/compatibility path for the earlier absolute-yield
                    # prototype. Production uses the nested convex profile above.
                    elif hasattr(self.boundary_interface_model, "predict_raw"):
                        if interface == "entry":
                            boundary_ke_mev = float(ke0)
                            boundary_particle_time_ns = 0.0
                        else:
                            remaining_range_mm = max(
                                0.0,
                                float(range_clipped.full_range_mm)
                                - float(range_clipped.visible_length_mm),
                            )
                            boundary_ke_mev = float(
                                self.range_lookup.range_mm_to_energy(
                                    remaining_range_mm
                                )
                            )
                            try:
                                from .Emitter import _wcte_integrated_primary_tof_fast
                                boundary_particle_time_ns = float(
                                    _wcte_integrated_primary_tof_fast(
                                        self.emitter,
                                        float(range_clipped.visible_length_mm),
                                    )
                                )
                            except Exception:
                                beta0 = math.sqrt(max(
                                    0.0,
                                    1.0
                                    - 1.0
                                    / (
                                        1.0
                                        + max(float(ke0), 0.0)
                                        / max(float(self.emitter.particle_mass), 1.0e-12)
                                    ) ** 2,
                                ))
                                boundary_particle_time_ns = (
                                    float(range_clipped.visible_length_mm)
                                    / max(beta0 * 299.792458, 1.0e-12)
                                )
                        hardware = self.boundary_interface_model.predict_raw(
                            boundary_hit=boundary_hit,
                            direction=direction,
                            interface=interface,
                            kinetic_energy_mev=boundary_ke_mev,
                            emitter=self.emitter,
                            boundary_particle_time_ns=boundary_particle_time_ns,
                        )
                        combined = self.boundary_interface_model.combine_raw(
                            emitter=self.emitter,
                            observed_pes=self.obs_pes,
                            hardware=hardware,
                        )
                        baseline_charge_nll = float(
                            self.pmt_model.get_neg_log_likelihood_npe(
                                exp_pes_array, self.obs_pes
                            )
                        )
                        exp_pes_array = np.asarray(
                            combined.expected_pes, dtype=np.float64
                        )
                        timing_array = np.asarray(
                            combined.timing_pes, dtype=np.float64
                        )
                        if need_times:
                            exp_ts_zero = (
                                self.boundary_interface_model.augment_timing_prediction(
                                    exp_ts_zero,
                                    hardware=hardware,
                                    combined_norm=float(combined.norm),
                                )
                            )
                        augmented_charge_nll = float(
                            self.pmt_model.get_neg_log_likelihood_npe(
                                exp_pes_array, self.obs_pes
                            )
                        )
                        self._store_boundary_diagnostic(geometry_key, {
                            "interface": interface,
                            "slot": (
                                None if boundary_hit.slot is None
                                else int(boundary_hit.slot)
                            ),
                            "kinetic_energy_at_interface_mev": float(
                                boundary_ke_mev
                            ),
                            "particle_time_at_interface_ns": float(
                                boundary_particle_time_ns
                            ),
                            "combined_norm": float(combined.norm),
                            "hardware_raw_total": float(
                                np.sum(hardware.raw_charge)
                            ),
                            "hardware_raw_fraction": float(
                                np.sum(hardware.raw_charge)
                                / max(np.sum(combined.raw_charge), 1.0e-30)
                            ),
                            "charge_nll_improvement": float(
                                baseline_charge_nll - augmented_charge_nll
                            ),
                            "model": dict(hardware.diagnostics),
                        })
                    else:
                        # Retained only for old diagnostic prototypes.  The
                        # production driver never enables free local fractions.
                        local_modes = self.boundary_interface_model.predict_modes(
                            boundary_hit=boundary_hit,
                            direction=direction,
                            interface=interface,
                            kinetic_energy_mev=ke0,
                            water_start_mm=range_clipped.start,
                            water_path_to_boundary_mm=water_path_to_boundary,
                        )
                        local_profile = self.boundary_interface_model.profile_charge(
                            exp_pes_array,
                            self.obs_pes,
                            local_modes,
                        )
                        exp_pes_array = np.asarray(
                            local_profile.expected_pes, dtype=np.float64
                        )
                        if need_times:
                            exp_ts_zero = (
                                self.boundary_interface_model.augment_timing_prediction(
                                    exp_ts_zero,
                                    modes=local_modes,
                                    fractions=local_profile.fractions,
                                    observed_pes=self.obs_pes,
                                )
                            )
                        timing_array = np.ascontiguousarray(
                            exp_pes_array, dtype=np.float64
                        )
                        self._store_boundary_diagnostic(geometry_key, {
                            "interface": interface,
                            "slot": (
                                None if boundary_hit.slot is None
                                else int(boundary_hit.slot)
                            ),
                            "fractions": np.asarray(
                                local_profile.fractions, dtype=np.float64
                            ).tolist(),
                            "mode_names": list(local_modes.mode_names),
                            "charge_nll_improvement": float(
                                local_profile.improvement
                            ),
                            "profile_iterations": int(local_profile.iterations),
                            "profile_converged": bool(local_profile.converged),
                            "model": dict(local_modes.diagnostics),
                        })
                except Exception as exc:
                    self.last_boundary_interface_error = repr(exc)
                    self._store_prediction(geometry_key, None)
                    return invalid()
            prediction = (
                np.ascontiguousarray(exp_pes_array, dtype=np.float64),
                exp_ts_zero,
                None if timing_array is None else np.ascontiguousarray(
                    timing_array, dtype=np.float64
                ),
            )
            self._store_prediction(geometry_key, prediction)
        else:
            prediction = self.prediction_cache[geometry_key]
        if prediction is None:
            return invalid()

        fval = self._prediction_nll(prediction, t0)
        if not math.isfinite(fval):
            return invalid()
        self._store_nll(key, fval)
        return float(fval)


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
                for sign in (-1.0, 1.0):
                    fv, trial = evaluate_updates(values, {dim: sign * h})
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



def _reanchor_objective_state(
    objective: TrackObjective,
    values: Mapping[str, float],
    fval: float,
) -> dict[str, float]:
    """Re-anchor ``objective`` at the current physical direction without a new FCN.

    ``TrackObjective`` cache keys are expressed in the local tangent chart.  A
    chart change therefore has to invalidate old keys.  The current exact
    prediction and NLL are physically unchanged, so transfer that single cache
    entry into the new chart instead of paying for another optical evaluation.
    """
    current = {k: float(v) for k, v in values.items()}
    direction = objective.chart.direction(
        current.get("dir_u", 0.0), current.get("dir_v", 0.0)
    )
    if direction is None:
        raise ValueError("cannot re-anchor a non-finite cosmic direction")

    old_geometry_key = objective._geometry_key(current)
    prediction = objective.prediction_cache.get(old_geometry_key)
    new_chart = TangentDirectionChart.from_direction(direction)
    current["dir_u"] = 0.0
    current["dir_v"] = 0.0
    objective.chart = new_chart
    objective.cache.clear()
    objective.prediction_cache.clear()
    objective._store_nll(objective._key(current), float(fval))
    if prediction is not None:
        objective._store_prediction(objective._geometry_key(current), prediction)
    return current


def two_anchor_block_optimize(
    objective: TrackObjective,
    start_values: Mapping[str, float],
    *,
    fixed_params: Mapping[str, float] | None = None,
    sweeps: int = 4,
    initial_steps: OptimizerSteps = OptimizerSteps(),
    min_steps: OptimizerMinSteps = OptimizerMinSteps(),
    length_limits: tuple[float, float] = (0.0, 3000.0),
    min_improvement: float = 0.5,
    full_cross_sweeps: int = 1,
    minimum_sweeps: int = 1,
    min_anchor_separation_mm: float = 20.0,
    preserve_stop_exit_topology: bool = True,
    initial_errors: Mapping[str, float] | None = None,
) -> BlockOptimizerResult:
    """Refine a clipped cosmic track in two physical-anchor coordinates.

    The ordinary local optimizer represents a line by ``(start, direction)``.
    For a long or oblique track those variables are strongly correlated: a
    transverse start displacement can be almost exactly cancelled by a small
    direction rotation at the downstream end.  This routine instead uses the
    physical visible-segment anchors ``A`` and ``B`` and optimizes their paired
    coordinates.  The line direction is derived analytically from ``B-A``.

    For stopping tracks, ``B`` is the stopping endpoint and ``|B-A|`` is the
    fitted range.  For exiting tracks, ``B`` is the geometric downstream water
    intersection and the full CSDA range remains a separate scalar.  Boundary
    entry is always re-derived by exact convex clipping, so no light is inferred
    outside the detector and no WCTE-specific direction convention enters.

    The routine is intended only for ``range_clip_track`` cosmic objectives.
    User-fixed Cartesian vertex, direction, or range coordinates are deliberately
    rejected; the caller should retain the ordinary optimizer for those unusual
    constrained fits.  A fixed ``t0`` is supported.
    """
    if not bool(objective.range_clip_track) or objective.track_end_mode != "full_length":
        raise ValueError("two-anchor optimization requires a cosmic range-clipped full-length objective")

    fixed = {} if fixed_params is None else {k: float(v) for k, v in fixed_params.items()}
    incompatible = {
        "x0", "y0", "z0", "direction", "dir_u", "dir_v", "length",
        "visible_length", "full_range",
    }.intersection(fixed)
    if incompatible:
        raise ValueError(
            "two-anchor optimization cannot preserve fixed geometry/range fields: "
            + ", ".join(sorted(incompatible))
        )

    wall0 = time.perf_counter()
    values = {k: float(v) for k, v in start_values.items()}
    values.setdefault("dir_u", 0.0)
    values.setdefault("dir_v", 0.0)
    values.setdefault("t0", 0.0)
    if "t0" in fixed:
        values["t0"] = float(fixed["t0"])

    nfcn_start = int(objective.evaluations)
    invalid_start = int(objective.invalid_evaluations)
    fval = float(objective(values))
    if not math.isfinite(fval):
        raise RuntimeError(f"non-finite two-anchor start FCN: {values}")

    errors: dict[str, float] = {
        "x0": np.nan, "y0": np.nan, "z0": np.nan,
        "dir_u": np.nan, "dir_v": np.nan,
        "length": np.nan, "visible_length": np.nan,
        "full_range": np.nan, "t0": np.nan,
    }
    if initial_errors is not None:
        for name, value in initial_errors.items():
            try:
                errors[str(name)] = float(value)
            except Exception:
                continue

    # Trust radii are physical displacements of the two visible anchors.  The
    # downstream transverse radius also reflects the angular trust radius, which
    # matters for multi-metre through-going chords.
    current_direction = objective.chart.direction(
        values.get("dir_u", 0.0), values.get("dir_v", 0.0)
    )
    resolved0 = None if current_direction is None else resolve_range_clipped_track(
        objective.detector,
        [values["x0"], values["y0"], values["z0"]],
        current_direction,
        values["length"],
        starts_at_boundary=bool(objective.boundary_entry_track),
        inset_mm=float(objective.boundary_clip_inset_mm),
        tolerance_mm=float(objective.containment_tolerance_mm),
    )
    if resolved0 is None:
        raise RuntimeError("two-anchor start does not define a physical clipped track")
    angular_end_step = float(
        resolved0.visible_length_mm * max(initial_steps.direction_tangent, 0.0)
    )
    steps: dict[str, float] = {
        "start_longitudinal": float(initial_steps.longitudinal_mm),
        "end_longitudinal": float(initial_steps.length_mm),
        "start_transverse_1": float(initial_steps.transverse_mm),
        "end_transverse_1": float(max(initial_steps.transverse_mm, angular_end_step)),
        "start_transverse_2": float(initial_steps.transverse_mm),
        "end_transverse_2": float(max(initial_steps.transverse_mm, angular_end_step)),
        "range": float(max(initial_steps.length_mm, initial_steps.full_range_mm)),
        "t0": float(initial_steps.t0_ns),
    }
    min_step: dict[str, float] = {
        "start_longitudinal": float(min_steps.longitudinal_mm),
        "end_longitudinal": float(min_steps.length_mm),
        "start_transverse_1": float(min_steps.transverse_mm),
        "end_transverse_1": float(min_steps.transverse_mm),
        "start_transverse_2": float(min_steps.transverse_mm),
        "end_transverse_2": float(min_steps.transverse_mm),
        "range": float(max(min_steps.length_mm, min_steps.full_range_mm)),
        "t0": float(min_steps.t0_ns),
    }

    history: list[dict[str, object]] = []
    quadratic_skips = 0
    min_anchor_separation = max(float(min_anchor_separation_mm), 1.0e-6)
    lo_range, hi_range = float(length_limits[0]), float(length_limits[1])

    def resolve_current(current: Mapping[str, float]) -> RangeClippedTrack | None:
        direction = objective.chart.direction(
            current.get("dir_u", 0.0), current.get("dir_v", 0.0)
        )
        if direction is None:
            return None
        return resolve_range_clipped_track(
            objective.detector,
            [current["x0"], current["y0"], current["z0"]],
            direction,
            current["length"],
            starts_at_boundary=bool(objective.boundary_entry_track),
            inset_mm=float(objective.boundary_clip_inset_mm),
            tolerance_mm=float(objective.containment_tolerance_mm),
        )

    def make_trial_from_anchors(
        base: Mapping[str, float],
        start_anchor: np.ndarray,
        end_anchor: np.ndarray,
        *,
        full_range_mm: float,
        expect_exits: bool,
        t0_value: float | None = None,
    ) -> tuple[float, dict[str, float] | None]:
        nonlocal quadratic_skips
        a = np.asarray(start_anchor, dtype=np.float64)
        b = np.asarray(end_anchor, dtype=np.float64)
        chord = b - a
        separation = float(np.linalg.norm(chord))
        if (
            a.shape != (3,) or b.shape != (3,)
            or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))
            or separation < min_anchor_separation
        ):
            return math.inf, None
        direction = np.ascontiguousarray(chord / separation, dtype=np.float64)
        local = objective.chart.coordinates(direction)
        if local is None:
            return math.inf, None

        if objective.boundary_entry_track:
            reference = 0.5 * (a + b)
            if not objective.detector.contains(
                reference, tolerance_mm=float(objective.containment_tolerance_mm)
            ):
                return math.inf, None
            boundary = resolve_boundary_clipped_track(
                objective.detector,
                reference,
                direction,
                inset_mm=float(objective.boundary_clip_inset_mm),
                tolerance_mm=float(objective.containment_tolerance_mm),
            )
            if boundary is None:
                return math.inf, None
            if expect_exits:
                fitted_range = float(full_range_mm)
            else:
                # Preserve the proposed stopping anchor after the canonical
                # upstream entry has been re-derived from the infinite line.
                fitted_range = float(np.dot(b - boundary.entry, direction))
                residual = float(np.linalg.norm(
                    (b - boundary.entry) - fitted_range * direction
                ))
                if residual > 1.0e-5 or fitted_range <= 0.0:
                    return math.inf, None
        else:
            reference = a
            if not objective.detector.contains(
                reference, tolerance_mm=float(objective.containment_tolerance_mm)
            ):
                return math.inf, None
            fitted_range = float(full_range_mm if expect_exits else separation)

        if not math.isfinite(fitted_range) or fitted_range < lo_range or fitted_range > hi_range:
            return math.inf, None
        trial = {k: float(v) for k, v in base.items()}
        trial.update({
            "x0": float(reference[0]),
            "y0": float(reference[1]),
            "z0": float(reference[2]),
            "dir_u": float(local[0]),
            "dir_v": float(local[1]),
            "length": float(fitted_range),
            "t0": float(base.get("t0", 0.0) if t0_value is None else t0_value),
        })
        if "t0" in fixed:
            trial["t0"] = float(fixed["t0"])
        trial_resolved = resolve_current(trial)
        if trial_resolved is None:
            return math.inf, None
        if preserve_stop_exit_topology and bool(trial_resolved.exits_detector) != bool(expect_exits):
            return math.inf, None
        if not expect_exits:
            endpoint_residual = float(np.linalg.norm(trial_resolved.endpoint - b))
            if endpoint_residual > 1.0e-4:
                return math.inf, None
        value = float(objective(trial))
        if not math.isfinite(value):
            return math.inf, None
        return value, trial

    def update_step(name: str, old_h: float, move: float, improved: bool) -> None:
        if improved:
            factor = 1.20 if abs(float(move)) > 0.70 * old_h else 0.70
        else:
            factor = 0.50
        steps[name] = max(min_step[name], old_h * factor)

    def optimize_1d(
        current_values: Mapping[str, float],
        current_fval: float,
        dimension: str,
        evaluator: Callable[[float], tuple[float, dict[str, float] | None]],
    ) -> tuple[float, dict[str, float], float]:
        h = float(steps[dimension])
        candidates: list[tuple[float, dict[str, float], float]] = [
            (float(current_fval), dict(current_values), 0.0)
        ]
        side: dict[int, tuple[float, dict[str, float] | None]] = {}
        for sign in (-1, 1):
            fv, trial = evaluator(float(sign) * h)
            side[sign] = (fv, trial)
            if trial is not None and math.isfinite(fv):
                candidates.append((float(fv), trial, float(sign) * h))
        fm, fp = side[-1][0], side[1][0]
        if math.isfinite(fm) and math.isfinite(fp):
            curvature = float(fm - 2.0 * current_fval + fp)
            if curvature > 1.0e-10 and math.isfinite(curvature):
                delta = float(np.clip(
                    0.5 * h * (fm - fp) / curvature, -2.0 * h, 2.0 * h
                ))
                fc, trial = evaluator(delta)
                if trial is not None and math.isfinite(fc):
                    candidates.append((float(fc), trial, delta))
                if dimension == "t0":
                    errors["t0"] = float(h / math.sqrt(max(curvature, 1.0e-30)))
                elif dimension == "range":
                    errors["length"] = float(h / math.sqrt(max(curvature, 1.0e-30)))
        best = min(candidates, key=lambda item: item[0])
        improved = best[0] < float(current_fval) - 1.0e-10
        update_step(dimension, h, best[2], improved)
        return float(best[0]), dict(best[1]), float(best[2])

    def optimize_2d(
        current_values: Mapping[str, float],
        current_fval: float,
        dimension_1: str,
        dimension_2: str,
        evaluator: Callable[[float, float], tuple[float, dict[str, float] | None]],
        *,
        full_cross: bool,
    ) -> tuple[float, dict[str, float], tuple[float, float]]:
        nonlocal quadratic_skips
        h1 = float(steps[dimension_1])
        h2 = float(steps[dimension_2])
        points = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
        if full_cross:
            points.extend([(-1, 1), (1, -1)])
        samples: dict[tuple[int, int], tuple[float, dict[str, float] | None]] = {}
        candidates: list[tuple[float, dict[str, float], tuple[float, float]]] = [
            (float(current_fval), dict(current_values), (0.0, 0.0))
        ]
        for i, j in points:
            fv, trial = evaluator(float(i) * h1, float(j) * h2)
            samples[(i, j)] = (fv, trial)
            if trial is not None and math.isfinite(fv):
                candidates.append((float(fv), trial, (float(i) * h1, float(j) * h2)))

        if all(math.isfinite(samples[key][0]) for key in points):
            fm1 = samples[(-1, 0)][0]
            fp1 = samples[(1, 0)][0]
            fm2 = samples[(0, -1)][0]
            fp2 = samples[(0, 1)][0]
            gradient = np.asarray(
                [(fp1 - fm1) / (2.0 * h1), (fp2 - fm2) / (2.0 * h2)],
                dtype=np.float64,
            )
            h11 = (fp1 - 2.0 * current_fval + fm1) / (h1 * h1)
            h22 = (fp2 - 2.0 * current_fval + fm2) / (h2 * h2)
            if full_cross:
                cross = (
                    samples[(1, 1)][0] - samples[(1, -1)][0]
                    - samples[(-1, 1)][0] + samples[(-1, -1)][0]
                ) / (4.0 * h1 * h2)
            else:
                cross_scaled = (
                    0.5 * (samples[(1, 1)][0] + samples[(-1, -1)][0])
                    - current_fval - 0.5 * (h11 * h1 * h1 + h22 * h2 * h2)
                )
                cross = cross_scaled / (h1 * h2)
            hessian = np.asarray([[h11, cross], [cross, h22]], dtype=np.float64)
            if np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian)):
                try:
                    scale = np.diag([h1, h2])
                    scaled_hessian = scale @ hessian @ scale
                    scaled_gradient = scale @ gradient
                    eig, vec = np.linalg.eigh(0.5 * (scaled_hessian + scaled_hessian.T))
                    if np.all(np.isfinite(eig)) and np.all(np.isfinite(vec)):
                        floor = max(1.0e-3, 0.05 * float(np.max(np.abs(eig))))
                        delta_scaled = -vec @ ((vec.T @ scaled_gradient) / np.maximum(eig, floor))
                        delta_scaled = np.clip(delta_scaled, -2.0, 2.0)
                        delta = scale @ delta_scaled
                        fq, trial = evaluator(float(delta[0]), float(delta[1]))
                        if trial is not None and math.isfinite(fq):
                            candidates.append((
                                float(fq), trial, (float(delta[0]), float(delta[1]))
                            ))
                    else:
                        quadratic_skips += 1
                except (np.linalg.LinAlgError, FloatingPointError):
                    quadratic_skips += 1
            else:
                quadratic_skips += 1
        else:
            quadratic_skips += 1

        best = min(candidates, key=lambda item: item[0])
        improved = best[0] < float(current_fval) - 1.0e-10
        update_step(dimension_1, h1, best[2][0], improved)
        update_step(dimension_2, h2, best[2][1], improved)
        return float(best[0]), dict(best[1]), tuple(best[2])

    for sweep in range(max(1, int(sweeps))):
        sweep_start = float(fval)
        # Re-anchor before every physical block.  This keeps the anchor axes
        # aligned with the current line even after a large diagonal correction.
        values = _reanchor_objective_state(objective, values, fval)
        resolved = resolve_current(values)
        if resolved is None:
            raise RuntimeError("two-anchor state became non-physical")
        expect_exits = bool(resolved.exits_detector)
        boundary_entry = bool(objective.boundary_entry_track)

        block_specs: list[tuple[str, str]] = []
        if not expect_exits and not boundary_entry:
            block_specs.append(("pair", "longitudinal"))
        elif not expect_exits and boundary_entry:
            block_specs.append(("endpoint", "longitudinal"))
        elif expect_exits and not boundary_entry:
            block_specs.append(("start_range", "longitudinal"))
        else:
            block_specs.append(("range", "range"))
        block_specs.extend((("pair", "transverse_1"), ("pair", "transverse_2")))
        if objective.objective_mode != "charge_only" and "t0" not in fixed:
            block_specs.append(("t0", "t0"))

        for block_kind, axis_name in block_specs:
            values = _reanchor_objective_state(objective, values, fval)
            resolved = resolve_current(values)
            if resolved is None:
                raise RuntimeError("two-anchor block state became non-physical")
            a0 = np.asarray(resolved.start, dtype=np.float64)
            b0 = np.asarray(resolved.endpoint, dtype=np.float64)
            expect_exits = bool(resolved.exits_detector)
            axis = {
                "longitudinal": objective.chart.anchor,
                "transverse_1": objective.chart.e1,
                "transverse_2": objective.chart.e2,
            }.get(axis_name)
            base = dict(values)
            full_range = float(resolved.full_range_mm)
            full_cross = sweep < int(full_cross_sweeps)

            if block_kind == "pair":
                d1 = (
                    "start_longitudinal" if axis_name == "longitudinal"
                    else f"start_{axis_name}"
                )
                d2 = (
                    "end_longitudinal" if axis_name == "longitudinal"
                    else f"end_{axis_name}"
                )

                def pair_eval(da: float, db: float):
                    return make_trial_from_anchors(
                        base, a0 + da * axis, b0 + db * axis,
                        full_range_mm=full_range,
                        expect_exits=expect_exits,
                    )

                fval, values, _ = optimize_2d(
                    values, fval, d1, d2, pair_eval, full_cross=full_cross
                )
            elif block_kind == "endpoint":
                def endpoint_eval(db: float):
                    return make_trial_from_anchors(
                        base, a0, b0 + db * axis,
                        full_range_mm=full_range,
                        expect_exits=False,
                    )

                fval, values, _ = optimize_1d(
                    values, fval, "end_longitudinal", endpoint_eval
                )
            elif block_kind == "start_range":
                def start_range_eval(da: float, dr: float):
                    return make_trial_from_anchors(
                        base, a0 + da * axis, b0,
                        full_range_mm=full_range + dr,
                        expect_exits=True,
                    )

                fval, values, _ = optimize_2d(
                    values, fval,
                    "start_longitudinal", "range",
                    start_range_eval,
                    full_cross=full_cross,
                )
            elif block_kind == "range":
                def range_eval(dr: float):
                    return make_trial_from_anchors(
                        base, a0, b0,
                        full_range_mm=full_range + dr,
                        expect_exits=True,
                    )

                fval, values, _ = optimize_1d(values, fval, "range", range_eval)
            elif block_kind == "t0":
                def t0_eval(dt: float):
                    trial = dict(base)
                    trial["t0"] = float(base.get("t0", 0.0) + dt)
                    value = float(objective(trial))
                    return (value, trial) if math.isfinite(value) else (math.inf, None)

                fval, values, _ = optimize_1d(values, fval, "t0", t0_eval)
            else:
                raise RuntimeError(f"unknown two-anchor block kind {block_kind!r}")

        values = _reanchor_objective_state(objective, values, fval)
        resolved = resolve_current(values)
        if resolved is None:
            raise RuntimeError("two-anchor sweep ended at a non-physical state")
        gain = float(sweep_start - fval)
        history.append({
            "sweep": int(sweep),
            "fval": float(fval),
            "gain_nll": gain,
            "nfcn": int(objective.evaluations - nfcn_start),
            "invalid_evaluations": int(objective.invalid_evaluations - invalid_start),
            "values": dict(values),
            "steps": dict(steps),
            "physical_start": resolved.start.tolist(),
            "physical_endpoint": resolved.endpoint.tolist(),
            "physical_direction": resolved.direction.tolist(),
            "visible_length_mm": float(resolved.visible_length_mm),
            "full_range_mm": float(resolved.full_range_mm),
            "topology": str(resolved.topology),
            "two_anchor": True,
        })
        if sweep + 1 >= max(1, int(minimum_sweeps)) and gain < float(min_improvement):
            break

    values = _reanchor_objective_state(objective, values, fval)
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

def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        return None
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = float(cumulative[-1])
    if not math.isfinite(total) or total <= 0.0:
        return None
    index = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    return float(values[min(index, values.size - 1)])


def _prediction_based_t0_seed(
    objective, values: Mapping[str, float], *, limits: tuple[float, float]
) -> float | None:
    """Estimate the additive timing offset from cached first-arrival nodes.

    The scalar ``TimingPrediction`` array is not always the quantity used by the
    production first-photoelectron likelihood.  When available, use the earliest
    positive direct/deferred node for each active PMT and take a charge-robust
    weighted median of ``t_obs - t_pred``.  This is only a navigation seed; the
    returned t0 is subsequently evaluated with the exact timing likelihood.
    """
    lo, hi = float(limits[0]), float(limits[1])
    try:
        geometry_key = objective._geometry_key(values)
        prediction = objective.prediction_cache.get(geometry_key)
    except Exception:
        prediction = None
    if prediction is None:
        return None
    try:
        _, timing_prediction, _ = prediction
        obs_pes = np.asarray(objective.obs_pes, dtype=np.float64)
        obs_ts = np.asarray(objective.obs_ts, dtype=np.float64)
    except Exception:
        return None

    active_raw = getattr(timing_prediction, "first_arrival_active_indices", None)
    reference_times: np.ndarray | None = None
    active: np.ndarray | None = None
    if active_raw is not None:
        active = np.asarray(active_raw, dtype=np.int64).reshape(-1)
        component_times: list[np.ndarray] = []
        for mu_name, time_name in (
            ("first_arrival_node_mu", "first_arrival_node_t"),
            ("first_arrival_deferred_base_mu", "first_arrival_deferred_base_t"),
        ):
            mu_raw = getattr(timing_prediction, mu_name, None)
            time_raw = getattr(timing_prediction, time_name, None)
            if mu_raw is None or time_raw is None:
                continue
            mu = np.asarray(mu_raw, dtype=np.float64)
            node_t = np.asarray(time_raw, dtype=np.float64)
            if (
                mu.ndim != 2
                or node_t.shape != mu.shape
                or mu.shape[1] != active.size
            ):
                continue
            valid = (mu > 0.0) & np.isfinite(node_t)
            with np.errstate(invalid="ignore"):
                earliest = np.min(np.where(valid, node_t, np.inf), axis=0)
            component_times.append(np.asarray(earliest, dtype=np.float64))

        # Reflection is normally later than direct light, but for PMTs whose only
        # modeled timing support is reflected light it still provides a useful
        # coarse event-time anchor.
        transfer_raw = getattr(
            timing_prediction, "first_arrival_reflection_transfer_active", None
        )
        base_raw = getattr(timing_prediction, "first_arrival_reflection_tbase", None)
        offset_raw = getattr(
            timing_prediction, "first_arrival_reflection_time_offset_active", None
        )
        if transfer_raw is not None and base_raw is not None:
            transfer = np.asarray(transfer_raw, dtype=np.float64)
            base_time = np.asarray(base_raw, dtype=np.float64).reshape(-1)
            if transfer.ndim == 2 and transfer.shape[1] == active.size:
                if offset_raw is None:
                    offset = np.zeros_like(transfer)
                else:
                    offset = np.asarray(offset_raw, dtype=np.float64)
                if offset.shape == transfer.shape and base_time.size == transfer.shape[0]:
                    reflected_t = base_time[:, None] + offset
                    valid = (transfer > 0.0) & np.isfinite(reflected_t)
                    with np.errstate(invalid="ignore"):
                        earliest_reflected = np.min(
                            np.where(valid, reflected_t, np.inf), axis=0
                        )
                    component_times.append(
                        np.asarray(earliest_reflected, dtype=np.float64)
                    )

        if component_times:
            stacked = np.vstack(component_times)
            with np.errstate(invalid="ignore"):
                reference_times = np.min(stacked, axis=0)

    if reference_times is not None and active is not None:
        valid_active = (active >= 0) & (active < obs_pes.size)
        active = active[valid_active]
        reference_times = reference_times[valid_active]
        if active.size:
            q = obs_pes[active]
            t_obs = obs_ts[active]
            mask = (
                (q > 0.0)
                & np.isfinite(q)
                & np.isfinite(t_obs)
                & np.isfinite(reference_times)
                & (np.abs(reference_times) < 1.0e6)
            )
            if np.any(mask):
                residual = t_obs[mask] - reference_times[mask]
                estimate = _weighted_median(
                    residual, np.sqrt(np.maximum(q[mask], 1.0e-12))
                )
                if estimate is not None and math.isfinite(estimate):
                    return float(np.clip(estimate, lo, hi))

    # Generic fallback for simpler timing models used in tests or detector
    # configurations without first-arrival node metadata.
    try:
        expected = np.asarray(timing_prediction, dtype=np.float64)
    except Exception:
        return None
    if expected.shape != obs_ts.shape:
        return None
    mask = (
        (obs_pes > 0.0)
        & np.isfinite(obs_pes)
        & np.isfinite(obs_ts)
        & np.isfinite(expected)
        & (np.abs(expected) < 1.0e6)
    )
    if not np.any(mask):
        return None
    estimate = _weighted_median(
        obs_ts[mask] - expected[mask],
        np.sqrt(np.maximum(obs_pes[mask], 1.0e-12)),
    )
    if estimate is None or not math.isfinite(estimate):
        return None
    return float(np.clip(estimate, lo, hi))


def profile_t0(
    objective,
    values: Mapping[str, float],
    *,
    limits: tuple[float, float],
    coarse_step_ns: float = 0.25,
    refine_levels: int = 2,
    refine_factor: float = 5.0,
    max_global_points: int = 9,
    seed_half_width_ns: float = 2.0,
    use_prediction_seed: bool = True,
) -> T0ProfileResult:
    """Profile the additive event-time offset with one optical prediction.

    A dense scan over a narrow range is cheap, but it scales poorly when an
    external particle can enter the water several nanoseconds after the WCSim
    generator time.  This implementation keeps global protection through a
    capped sparse grid, then places a dense local grid around a robust
    first-arrival-node estimate and recursively refines the best bracket.  Every
    sampled point is still evaluated with the exact production likelihood.
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
    global_points = max(0, int(max_global_points))
    half_width = max(0.0, float(seed_half_width_ns))

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

    base_t0 = float(np.clip(base.get("t0", 0.0), lo, hi))
    evaluate(base_t0)

    timing_seed = (
        _prediction_based_t0_seed(objective, base, limits=(lo, hi))
        if use_prediction_seed and hi > lo
        else None
    )

    if hi > lo:
        nominal_points = int(math.ceil((hi - lo) / step)) + 1
        if global_points > 0:
            if nominal_points <= global_points:
                evaluate_many(np.linspace(lo, hi, nominal_points))
            elif global_points == 1:
                evaluate(0.5 * (lo + hi))
            else:
                evaluate_many(np.linspace(lo, hi, global_points))

        if timing_seed is not None:
            local_lo = max(lo, float(timing_seed) - half_width)
            local_hi = min(hi, float(timing_seed) + half_width)
            if local_hi > local_lo:
                n_local = max(1, int(math.ceil((local_hi - local_lo) / step)))
                evaluate_many(np.linspace(local_lo, local_hi, n_local + 1))
            else:
                evaluate(float(timing_seed))

        # Refine the bracket surrounding the current best point.  This handles
        # both the prediction-seeded local grid and the sparse-global fallback.
        for _ in range(max(0, int(refine_levels))):
            finite = sorted(
                ((float(t), float(f)) for t, f in samples.items() if math.isfinite(f)),
                key=lambda item: item[0],
            )
            if not finite:
                break
            best_index = min(range(len(finite)), key=lambda i: finite[i][1])
            best_t = finite[best_index][0]
            if best_index > 0:
                bracket_lo = finite[best_index - 1][0]
            else:
                gap = finite[1][0] - best_t if len(finite) > 1 else step
                bracket_lo = max(lo, best_t - max(gap, step))
            if best_index + 1 < len(finite):
                bracket_hi = finite[best_index + 1][0]
            else:
                gap = best_t - finite[-2][0] if len(finite) > 1 else step
                bracket_hi = min(hi, best_t + max(gap, step))
            if bracket_hi <= bracket_lo + 1.0e-15:
                break
            n_refine = max(4, int(math.ceil(2.0 * factor)))
            evaluate_many(np.linspace(bracket_lo, bracket_hi, n_refine + 1))

    finite = [(t, f) for t, f in samples.items() if math.isfinite(f)]
    if not finite:
        raise RuntimeError("all t0 profile points were non-finite")
    best_t, best_f = min(finite, key=lambda item: item[1])

    left = max(
        (item for item in finite if item[0] < best_t),
        default=None,
        key=lambda x: x[0],
    )
    right = min(
        (item for item in finite if item[0] > best_t),
        default=None,
        key=lambda x: x[0],
    )
    error = math.nan
    if left is not None and right is not None:
        h_left = best_t - left[0]
        h_right = right[0] - best_t
        if h_left > 0.0 and h_right > 0.0:
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
        optical_evaluations=(
            int(getattr(objective, "optical_evaluations", 0)) - start_optical
        ),
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
        except Exception:
            digest.update(str(path).encode("utf-8"))
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


def broad_length_grid(
    max_length_mm: float,
    *,
    fractions: Sequence[float] = (
        0.02, 0.04, 0.08, 0.16, 0.24, 0.34, 0.48, 0.65, 0.82, 0.97
    ),
    minimum_mm: float = 80.0,
    extra_lengths_mm: Sequence[float] = (),
) -> list[float]:
    """Return a compact, energy-independent visible-length bank.

    The original WCTE drivers scaled every seed length from one assumed kinetic
    energy.  If that assumption was low, no seed existed in the correct range
    basin and the local optimizer could not cross the resulting gap.  This grid
    instead spans the complete detector-supported interval.  The approximately
    logarithmic low-end coverage retains short stopping tracks while the dense
    upper fractions cover long contained and cosmic chords.

    ``extra_lengths_mm`` may add a calibration-specific hint, but the mandatory
    detector-wide coverage is present even when that hint is absent or wrong.
    """
    maximum = float(max_length_mm)
    minimum = max(0.0, float(minimum_mm))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_length_mm must be finite and positive")
    if maximum < minimum:
        minimum = maximum
    raw = [minimum]
    raw.extend(maximum * float(f) for f in fractions)
    raw.extend(float(value) for value in extra_lengths_mm)
    cleaned = sorted({
        round(float(value), 6)
        for value in raw
        if math.isfinite(float(value)) and minimum <= float(value) <= maximum
    })
    if not cleaned:
        raise ValueError("the broad visible-length grid is empty")
    return [float(value) for value in cleaned]


def _full_range_candidates_for_visible(
    visible_length_mm: float,
    *,
    full_ranges_mm: Sequence[float] | None,
    full_range_factors: Sequence[float],
    full_range_max_mm: float | None,
) -> list[float]:
    """Construct a compact set of physically valid absorption-range seeds."""
    visible = float(visible_length_mm)
    maximum = (
        math.inf
        if full_range_max_mm is None
        else float(full_range_max_mm)
    )
    raw: list[float] = []
    if full_ranges_mm is not None:
        raw.extend(float(value) for value in full_ranges_mm)
    raw.extend(visible * float(factor) for factor in full_range_factors)
    cleaned = sorted({
        round(float(value), 6)
        for value in raw
        if (
            math.isfinite(float(value))
            and float(value) >= visible
            and float(value) > 0.0
            and float(value) <= maximum
        )
    })
    return [float(value) for value in cleaned]


def build_boundary_chord_seed_grid(
    detector: ConvexDetectorVolume,
    directions: np.ndarray,
    *,
    insets_mm: Sequence[float] = (0.5, 25.0, 150.0, 350.0),
    transverse_offsets_mm: Sequence[Sequence[float]] = (
        (0.0, 0.0),
        (450.0, 0.0), (-450.0, 0.0),
        (0.0, 450.0), (0.0, -450.0),
        (900.0, 0.0), (-900.0, 0.0),
        (0.0, 900.0), (0.0, -900.0),
    ),
    chord_fractions: Sequence[float] = (0.20, 0.45, 0.70, 0.90, 0.985, 1.0),
    absolute_lengths_mm: Sequence[float] = (),
    minimum_length_mm: float = 20.0,
    maximum_length_mm: float | None = None,
    track_end_mode: str = "full_length",
    full_ranges_mm: Sequence[float] | None = None,
    full_range_factors: Sequence[float] = (1.0, 1.15, 2.0),
    full_range_max_mm: float | None = None,
    seed_family: str = "boundary_chord_guard",
    allow_length_beyond_chord: bool = False,
) -> list[dict[str, object]]:
    """Build full-sphere seeds tied to the local detector chord.

    These hypotheses are specifically important for particles that enter from
    outside the active water.  Their visible length is controlled by the water
    chord, not by the generator energy or by the full CSDA range.  Near-boundary
    insets and near-unit chord fractions therefore provide the basin that an
    energy-scaled absolute-length grid systematically misses.

    In absorption mode the full range is seeded independently through compact
    multiplicative factors and optional absolute anchors.  This represents both
    a stopping track (factor 1) and a through-going high-energy track without a
    large visible-length × energy Cartesian product.
    """
    mode = str(track_end_mode).strip().lower().replace("-", "_")
    if mode not in {"full_length", "absorption"}:
        raise ValueError("track_end_mode must be full_length or absorption")
    maximum_length = (
        math.inf if maximum_length_mm is None else float(maximum_length_mm)
    )
    minimum_length = max(0.0, float(minimum_length_mm))
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
                if not detector.contains(vertex, tolerance_mm=1.0e-6):
                    continue
                exit_distance = detector.ray_exit_distance(vertex, chart.anchor)
                if not math.isfinite(exit_distance) or exit_distance <= 0.0:
                    continue
                local_lengths = list(float(value) for value in absolute_lengths_mm)
                local_lengths.extend(
                    float(exit_distance) * float(fraction)
                    for fraction in chord_fractions
                )
                local_lengths = sorted({
                    round(float(value), 6)
                    for value in local_lengths
                    if (
                        math.isfinite(float(value))
                        and minimum_length <= float(value) <= maximum_length
                        and (
                            bool(allow_length_beyond_chord)
                            or float(value) <= float(exit_distance) + 1.0e-6
                        )
                    )
                })
                for length in local_lengths:
                    base: dict[str, object] = {
                        "x0": float(vertex[0]),
                        "y0": float(vertex[1]),
                        "z0": float(vertex[2]),
                        "dir_x": float(chart.anchor[0]),
                        "dir_y": float(chart.anchor[1]),
                        "dir_z": float(chart.anchor[2]),
                        "cx": float(chart.anchor[0]),
                        "cy": float(chart.anchor[1]),
                        "cz": float(chart.anchor[2]),
                        "cz_sign": -1.0 if float(chart.anchor[2]) < 0.0 else 1.0,
                        "t0": 0.0,
                        "seed_family": str(seed_family),
                        "boundary_exit_distance_mm": float(exit_distance),
                    }
                    if mode == "absorption":
                        ranges = _full_range_candidates_for_visible(
                            float(length),
                            full_ranges_mm=full_ranges_mm,
                            full_range_factors=full_range_factors,
                            full_range_max_mm=full_range_max_mm,
                        )
                        for full_range in ranges:
                            seed = dict(base)
                            seed["visible_length"] = float(length)
                            seed["full_range"] = float(full_range)
                            seeds.append(seed)
                    else:
                        seed = dict(base)
                        seed["length"] = float(length)
                        seeds.append(seed)
    return deduplicate_seed_grid(seeds, track_end_mode=mode)


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
    allow_boundary_clipping: bool = False,
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
        if (
            not bool(allow_boundary_clipping)
            and length > exit_distance + 1.0e-6
        ):
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
    allow_boundary_clipping: bool = False,
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
                    allow_boundary_clipping=bool(allow_boundary_clipping),
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
    allow_boundary_clipping: bool = False,
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
                    allow_boundary_clipping=bool(allow_boundary_clipping),
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
    allow_boundary_clipping: bool = False,
) -> list[dict[str, object]]:
    """Apply fixed values, reject nonphysical rows, and deduplicate."""
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
        if not (
            bool(allow_boundary_clipping) and track_end_mode == "full_length"
        ):
            if not detector.segment_contained(vertex, direction, length):
                continue
        else:
            exit_distance = detector.ray_exit_distance(vertex, direction)
            if not math.isfinite(exit_distance) or exit_distance <= 0.0:
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
    allow_boundary_clipping: bool = False,
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
        allow_boundary_clipping=bool(allow_boundary_clipping),
    )


def _build_global_seed_grid_from_vertices(
    detector: ConvexDetectorVolume,
    vertices: Sequence[Sequence[float]],
    directions: np.ndarray,
    lengths_mm: Sequence[float],
    *,
    track_end_mode: str,
    full_ranges_mm: Sequence[float] | None,
    allow_boundary_clipping: bool = False,
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
                if (
                    not bool(allow_boundary_clipping)
                    and float(length) > exit_distance + 1.0e-6
                ):
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
    allow_boundary_clipping: bool = False,
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
                if (
                    not bool(allow_boundary_clipping)
                    and length > exit_distance + 1.0e-6
                ):
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
    allow_boundary_clipping: bool = False,
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
            allow_boundary_clipping=bool(allow_boundary_clipping),
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
        allow_boundary_clipping=bool(allow_boundary_clipping),
    )
    metadata = {
        "requested_vertex_spacing_mm": float(requested_spacing),
        "effective_vertex_spacing_mm": float(spacing),
        "wall_margin_mm": float(wall_margin_mm),
        "vertex_count": int(len(final_vertices)),
        "seed_count": int(len(seeds)),
        "max_total_seeds": int(budget),
        "allow_boundary_clipping": bool(allow_boundary_clipping),
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
        """Delegate exact t0 batching to the base objective and add the prior.

        The aligned Fermi--Eyges prior contains vertex, direction and length
        coordinates only; it is independent of the additive event time.  The
        prior penalty is therefore constant across the complete t0 grid and
        can be evaluated once without touching the base prediction cache.
        """
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
