"""Detector-geometry classification and optical-model compatibility helpers.

Track coordinates and direct Cherenkov light are detector agnostic.  Some
secondary optical components in the historical fitter are not:

* the validated blacksheet reflection transfer is an exact WCTE 16-sided
  surface model;
* the molecular-scattering transport historically used the same WCTE prism
  for the photon flight-to-boundary calculation; and
* the sparse receiver-moment table is tied to one PMT geometry by hashes.

This module makes those assumptions explicit.  WCTE keeps its validated models.
Other convex WCD/IWCD geometries use mPMT-derived boundary planes for molecular
scattering and never silently apply the WCTE reflection transfer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np


WCTE_DEFAULT_INACTIVE_SLOTS = (27, 32, 45, 74, 77, 79, 85, 91, 99)

# Exact inner-water prism used by the validated WCTE molecular-scattering and
# blacksheet-reflection models.  Keeping one public definition prevents the
# tracking constraints and optical boundary from drifting apart.
WCTE_PRISM_N_SIDES = 16
WCTE_PRISM_APOTHEM_MM = 3075.926 / 2.0
WCTE_PRISM_HEIGHT_MM = 2714.235
WCTE_PRISM_Y_CENTER_MM = 424.763
WCTE_PRISM_Y_MIN_MM = WCTE_PRISM_Y_CENTER_MM - 0.5 * WCTE_PRISM_HEIGHT_MM
WCTE_PRISM_Y_MAX_MM = WCTE_PRISM_Y_CENTER_MM + 0.5 * WCTE_PRISM_HEIGHT_MM

# Physical WCTE mPMT vessel envelope used when converting the nominal inner
# blacksheet prism into the actual connected central-water volume.  These
# constants follow WCSimConstructInSitu/ExSituMultiPMT.cc: a 347 mm outer
# dome, a 235 mm spherical-cap cut, and a 2*77.785 mm vessel cylinder.
WCTE_MPMT_DOME_OUTER_RADIUS_MM = 347.0
WCTE_MPMT_DOME_INNER_RADIUS_MM = 332.0
WCTE_MPMT_DOME_CAP_CUT_MM = 235.0
WCTE_MPMT_DOME_CYLINDER_HEIGHT_MM = 2.0 * 77.785
WCTE_MPMT_VESSEL_OUTER_RADIUS_MM = 254.0
WCTE_MPMT_VESSEL_INNER_RADIUS_MM = 250.93

# Local 3-inch PMT / reflector dimensions in the in-situ WCTE module.  The
# boundary-interaction model consumes these through a module-design object, so
# an IWCD geometry adapter can replace them without changing the fitter logic.
WCTE_PMT_GLASS_RADIUS_MM = 53.0
WCTE_PMT_GLASS_CUT_MM = 34.597
WCTE_PMT_REFLECTOR_MOUTH_RADIUS_MM = 45.0
WCTE_PMT_REFLECTOR_INNER_RADIUS_MM = 36.96
WCTE_PMT_REFLECTOR_HEIGHT_MM = 13.0


@dataclass(frozen=True)
class DetectorGeometrySummary:
    name: str
    kind: str
    n_slots: int
    n_active_mpmts: int
    locations_mm: np.ndarray
    inward_axes: np.ndarray
    axis_lo_mm: np.ndarray
    axis_hi_mm: np.ndarray
    center_mm: np.ndarray
    normal_flips: int
    is_wcte_like: bool
    geometry_sha256: str

    @classmethod
    def from_wcd(cls, wcd, *, placement: str = "design") -> "DetectorGeometrySummary":
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
            except Exception:
                continue
            if (
                location.shape == (3,)
                and axis.shape == (3,)
                and np.all(np.isfinite(location))
                and np.all(np.isfinite(axis))
                and norm > 0.0
            ):
                locations.append(location)
                axes.append(axis / norm)

        if locations:
            loc = np.ascontiguousarray(np.asarray(locations, dtype=np.float64))
            raw_axes = np.asarray(axes, dtype=np.float64)
            center = np.median(loc, axis=0)
            flip = np.einsum("ij,ij->i", center[None, :] - loc, raw_axes) < 0.0
            raw_axes[flip] *= -1.0
            inward = np.ascontiguousarray(raw_axes, dtype=np.float64)
            lo = np.min(loc, axis=0)
            hi = np.max(loc, axis=0)
            n_flips = int(np.count_nonzero(flip))
        else:
            loc = np.empty((0, 3), dtype=np.float64)
            inward = np.empty((0, 3), dtype=np.float64)
            lo = np.full(3, np.nan, dtype=np.float64)
            hi = np.full(3, np.nan, dtype=np.float64)
            center = np.zeros(3, dtype=np.float64)
            n_flips = 0

        name = str(getattr(wcd, "name", ""))
        kind = str(getattr(wcd, "kind", ""))
        n_slots = len(getattr(wcd, "mpmts", []))
        n_active = int(loc.shape[0])
        lower_label = f"{name} {kind}".lower()
        labelled_wcte = "wcte" in lower_label
        extent_wcte = False
        if n_active:
            span = hi - lo
            bounding_center = 0.5 * (lo + hi)
            extent_wcte = bool(
                95 <= n_active <= 110
                and 3250.0 <= span[0] <= 3550.0
                and 2850.0 <= span[1] <= 3250.0
                and 3250.0 <= span[2] <= 3550.0
                and abs(span[0] - span[2]) <= 120.0
                and 300.0 <= bounding_center[1] <= 550.0
            )
        # A name/kind label is not sufficient to authorize the hard-coded
        # WCTE prism and blacksheet transfer. Require the measured geometry to
        # have the WCTE scale/aspect as well, so a mislabeled future detector
        # cannot silently receive WCTE-only optical boundaries.
        is_wcte = bool(extent_wcte and (labelled_wcte or n_slots == 106))

        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(loc, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(inward, dtype=np.float64).tobytes())
        digest.update(name.encode("utf-8"))
        digest.update(kind.encode("utf-8"))
        return cls(
            name=name,
            kind=kind,
            n_slots=int(n_slots),
            n_active_mpmts=n_active,
            locations_mm=loc,
            inward_axes=inward,
            axis_lo_mm=np.ascontiguousarray(lo, dtype=np.float64),
            axis_hi_mm=np.ascontiguousarray(hi, dtype=np.float64),
            center_mm=np.ascontiguousarray(center, dtype=np.float64),
            normal_flips=n_flips,
            is_wcte_like=is_wcte,
            geometry_sha256=digest.hexdigest(),
        )

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "n_slots": self.n_slots,
            "n_active_mpmts": self.n_active_mpmts,
            "axis_limits_mm": np.column_stack((self.axis_lo_mm, self.axis_hi_mm)).tolist(),
            "center_mm": self.center_mm.tolist(),
            "normal_flips": self.normal_flips,
            "is_wcte_like": self.is_wcte_like,
            "geometry_sha256": self.geometry_sha256,
        }


def configure_emitter_for_detector(
    emitter,
    summary: DetectorGeometrySummary,
    *,
    reflection_policy: str = "disable",
    photon_boundary_model: str = "auto",
) -> dict[str, object]:
    """Configure detector-dependent optical switches and return decisions.

    ``reflection_policy`` for a non-WCTE detector:
      ``disable`` (default)
          Turn off the WCTE-only analytic blacksheet transfer and record why.
      ``error``
          Refuse to run when reflection is enabled.

    Molecular scattering selects the exact WCTE prism for WCTE and an inward
    mPMT-plane convex boundary for other detectors.  The latter is a geometry
    approximation whose quality should be validated for the final IWCD layout,
    but it is rotationally and dimensionally general and avoids applying WCTE
    dimensions to IWCD.
    """
    policy = str(reflection_policy).strip().lower().replace("-", "_")
    if policy not in {"disable", "error"}:
        raise ValueError("reflection_policy must be 'disable' or 'error'")
    requested_boundary = str(photon_boundary_model).strip().lower().replace("-", "_")
    if requested_boundary == "auto":
        boundary = "wcte_prism" if summary.is_wcte_like else "convex_mpmt_planes"
    elif requested_boundary in {"wcte_prism", "convex_mpmt_planes"}:
        boundary = requested_boundary
    else:
        raise ValueError(
            "photon_boundary_model must be auto, wcte_prism, or convex_mpmt_planes"
        )
    if boundary == "wcte_prism" and not summary.is_wcte_like:
        raise ValueError("wcte_prism was requested for a detector not identified as WCTE")
    emitter.photon_scatter_boundary_model = boundary

    reflection_was_enabled = bool(getattr(emitter, "enable_blacksheet_reflection", False))
    reflection_disabled_reason = None
    if reflection_was_enabled and not summary.is_wcte_like:
        if policy == "error":
            raise RuntimeError(
                "The analytic blacksheet-reflection model is WCTE-specific. "
                "Disable reflection or provide an IWCD surface-transfer model."
            )
        emitter.enable_blacksheet_reflection = False
        reflection_disabled_reason = "WCTE-only reflection model disabled on non-WCTE geometry"

    return {
        "photon_scatter_boundary_model": boundary,
        "reflection_requested": reflection_was_enabled,
        "reflection_enabled": bool(getattr(emitter, "enable_blacksheet_reflection", False)),
        "reflection_disabled_reason": reflection_disabled_reason,
    }
