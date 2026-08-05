"""Fast analytic charged-particle traversal model for mPMT boundaries.

The ordinary LicketyFit optical model starts at the central active-water
boundary.  A cosmic particle can, however, cross an mPMT assembly before it
enters that water (or after it exits).  Transparent gel, acrylic and PMT glass
then produce a small, highly local Cherenkov signal.  Ignoring that signal lets
an optimizer prefer a geometrically nearby line that just misses the module.

This module supplies a deliberately low-dimensional correction for an explicit
``mPMT-entry`` or ``mPMT-exit`` hypothesis.  It is not a Geant4 replacement and
it does not trace individual photons.  Instead it uses:

* the detector-geometry package for module and PMT placements;
* analytic line intersections with the module envelope and transparent shells;
* exact Cherenkov-cone/finite-angular-aperture overlap for a small deterministic
  source quadrature;
* a separate traversed-PMT (near-field) mode with an analytic orientation gate;
* a convex profile over two non-negative charge fractions; and
* optional deterministic first-arrival timing nodes.

The existing water-only prediction is an exact nested point of the model: both
fractions may be zero.  Production code should nevertheless compare this
augmented hypothesis to the untouched clean-boundary hypothesis with an
explicit complexity/look-elsewhere penalty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .cosmic_track_fit import BoundarySurfaceHit, RangeClippedTrack
from .detector_geometry import (
    WCTE_MPMT_DOME_CAP_CUT_MM,
    WCTE_MPMT_DOME_CYLINDER_HEIGHT_MM,
    WCTE_MPMT_DOME_INNER_RADIUS_MM,
    WCTE_MPMT_DOME_OUTER_RADIUS_MM,
    WCTE_MPMT_VESSEL_OUTER_RADIUS_MM,
    WCTE_PMT_GLASS_RADIUS_MM,
    WCTE_PMT_REFLECTOR_MOUTH_RADIUS_MM,
)

_MUON_MASS_MEV = 105.6583755
_C_MM_PER_NS = 299.792458


@dataclass(frozen=True)
class OpticalMaterial:
    """Effective optical constants for one transparent construction material.

    The phase index sets the Cherenkov angle.  The group index sets propagation
    time.  ``relative_yield`` only combines source points *within* the broad
    spatial mode; the overall mode amplitude is profiled from the event.
    """

    name: str
    phase_index: float
    group_index: float
    relative_yield: float = 1.0


# Configuration-derived effective values.  They are intentionally exposed and
# serialized in metadata so a later WCTE/IWCD geometry/material adapter can
# replace them without changing the likelihood machinery.
DEFAULT_WCTE_MATERIALS: tuple[OpticalMaterial, ...] = (
    OpticalMaterial("silicone_gel", phase_index=1.46, group_index=1.49, relative_yield=1.0),
    OpticalMaterial("acrylic_dome", phase_index=1.49, group_index=1.51, relative_yield=0.80),
)


@dataclass(frozen=True)
class ModuleDesign:
    """Geometry of one mPMT design in a module-local axial coordinate."""

    cylinder_height_mm: float = WCTE_MPMT_DOME_CYLINDER_HEIGHT_MM
    vessel_outer_radius_mm: float = WCTE_MPMT_VESSEL_OUTER_RADIUS_MM
    dome_outer_radius_mm: float = WCTE_MPMT_DOME_OUTER_RADIUS_MM
    dome_inner_radius_mm: float = WCTE_MPMT_DOME_INNER_RADIUS_MM
    dome_cut_mm: float = WCTE_MPMT_DOME_CAP_CUT_MM
    gel_inner_radius_mm: float = 325.603  # matrix outer radius; avoids plastic overlap
    gel_outer_radius_mm: float = WCTE_MPMT_DOME_INNER_RADIUS_MM
    pmt_effective_radius_mm: float = WCTE_PMT_GLASS_RADIUS_MM
    receiver_aperture_radius_mm: float = WCTE_PMT_REFLECTOR_MOUTH_RADIUS_MM
    broad_quadrature_order: int = 6
    max_hardware_fraction: float = 0.35

    @property
    def sphere_centre_offset_mm(self) -> float:
        return float(self.cylinder_height_mm - self.dome_cut_mm)


@dataclass(frozen=True)
class ModuleGeometry:
    """Placed mPMT geometry assembled from the detector geometry package."""

    slot: int
    base_mm: np.ndarray
    inward_axis: np.ndarray
    sphere_centre_mm: np.ndarray
    pmt_indices: np.ndarray
    pmt_positions_mm: np.ndarray
    pmt_normals: np.ndarray
    design: ModuleDesign = field(default_factory=ModuleDesign)
    module_kind: str = "unknown"

    @classmethod
    def from_geometry(
        cls,
        *,
        slot: int,
        wcd: object,
        detector: object,
        pmt_positions_mm: np.ndarray,
        pmt_normals: np.ndarray,
        pmt_slots: np.ndarray,
        placement: str = "design",
        design: ModuleDesign | None = None,
    ) -> "ModuleGeometry":
        slot = int(slot)
        design = ModuleDesign() if design is None else design
        module = getattr(wcd, "mpmts", [])[slot]
        if module is None:
            raise ValueError(f"mPMT slot {slot} is absent from the geometry")
        placed = module.get_placement(placement, wcd)
        base = np.asarray(placed["location"], dtype=np.float64)
        raw_axis = np.asarray(placed["direction_z"], dtype=np.float64)
        norm = float(np.linalg.norm(raw_axis))
        if base.shape != (3,) or raw_axis.shape != (3,) or norm <= 0.0:
            raise ValueError(f"invalid geometry placement for mPMT slot {slot}")
        raw_axis = raw_axis / norm

        # Use the active-water exclusion orientation as the authoritative inward
        # axis whenever available.  This guarantees that the local optical model
        # and cosmic clipping use the same surface convention.
        exclusion_slots = np.asarray(
            getattr(detector, "exclusion_slots", np.empty(0)), dtype=np.int64
        )
        matches = np.flatnonzero(exclusion_slots == slot)
        if matches.size:
            idx = int(matches[0])
            axis = np.asarray(detector.exclusion_axes[idx], dtype=np.float64)
            centre = np.asarray(detector.exclusion_centres_mm[idx], dtype=np.float64)
            if float(np.dot(axis, raw_axis)) < 0.0:
                raw_axis = -raw_axis
            # Derive the base from the exclusion centre so small differences in
            # geometry-package and detector-adapter conventions do not accumulate.
            base = centre - float(design.sphere_centre_offset_mm) * axis
        else:
            axis = raw_axis
            centre = base + float(design.sphere_centre_offset_mm) * axis
        axis = axis / float(np.linalg.norm(axis))

        all_positions = np.asarray(pmt_positions_mm, dtype=np.float64)
        all_normals = np.asarray(pmt_normals, dtype=np.float64)
        all_slots = np.asarray(pmt_slots, dtype=np.int64)
        indices = np.flatnonzero(all_slots == slot).astype(np.int32)
        if indices.size == 0:
            raise ValueError(f"no PMTs are mapped to mPMT slot {slot}")
        return cls(
            slot=slot,
            base_mm=np.ascontiguousarray(base, dtype=np.float64),
            inward_axis=np.ascontiguousarray(axis, dtype=np.float64),
            sphere_centre_mm=np.ascontiguousarray(centre, dtype=np.float64),
            pmt_indices=np.ascontiguousarray(indices, dtype=np.int32),
            pmt_positions_mm=np.ascontiguousarray(all_positions[indices], dtype=np.float64),
            pmt_normals=np.ascontiguousarray(all_normals[indices], dtype=np.float64),
            design=design,
            module_kind=str(getattr(module, "kind", "unknown")),
        )

    def local_axial_mm(self, point_mm: Sequence[float]) -> float:
        return float(np.dot(np.asarray(point_mm, dtype=np.float64) - self.base_mm, self.inward_axis))

    def inside_outer_envelope(self, point_mm: Sequence[float], *, tolerance_mm: float = 1.0e-7) -> bool:
        point = np.asarray(point_mm, dtype=np.float64)
        rel = point - self.base_mm
        z = float(np.dot(rel, self.inward_axis))
        radial = rel - z * self.inward_axis
        r2 = float(np.dot(radial, radial))
        h = float(self.design.cylinder_height_mm)
        rc = float(self.design.vessel_outer_radius_mm)
        if -tolerance_mm <= z <= h + tolerance_mm and r2 <= (rc + tolerance_mm) ** 2:
            return True
        q = point - self.sphere_centre_mm
        return bool(
            z >= h - tolerance_mm
            and float(np.dot(q, q)) <= (float(self.design.dome_outer_radius_mm) + tolerance_mm) ** 2
        )

    def _transparent_material(self, point_mm: Sequence[float]) -> str | None:
        point = np.asarray(point_mm, dtype=np.float64)
        z = self.local_axial_mm(point)
        if z < float(self.design.cylinder_height_mm):
            return None
        radius = float(np.linalg.norm(point - self.sphere_centre_mm))
        eps = 1.0e-7
        if (
            float(self.design.dome_inner_radius_mm) - eps
            <= radius
            <= float(self.design.dome_outer_radius_mm) + eps
        ):
            return "acrylic_dome"
        if (
            float(self.design.gel_inner_radius_mm) - eps
            <= radius
            < float(self.design.gel_outer_radius_mm) - eps
        ):
            return "silicone_gel"
        return None

    def hardware_interval_from_boundary(
        self,
        boundary_point_mm: Sequence[float],
        direction: Sequence[float],
        *,
        interface: str,
        max_search_mm: float = 1500.0,
    ) -> tuple[float, float] | None:
        """Signed line interval occupied by the module next to a water boundary.

        The line is ``x(t)=boundary_point+t*direction``.  Entry hardware lies at
        negative ``t``; exit hardware lies at positive ``t``.  A monotone search
        followed by bisection is more robust at the cylinder/dome union seam than
        selecting among nearly coincident analytic roots.
        """
        point = np.asarray(boundary_point_mm, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(d))
        if point.shape != (3,) or d.shape != (3,) or norm <= 0.0:
            return None
        d = d / norm
        label = str(interface).strip().lower()
        if label not in {"entry", "exit"}:
            raise ValueError("interface must be 'entry' or 'exit'")
        sign = -1.0 if label == "entry" else 1.0
        probe = 1.0e-4
        if not self.inside_outer_envelope(point + sign * probe * d, tolerance_mm=1.0e-5):
            return None
        last_inside = probe
        step = 0.25
        outside = None
        while step <= float(max_search_mm):
            if self.inside_outer_envelope(point + sign * step * d):
                last_inside = step
                step *= 1.6
            else:
                outside = step
                break
        if outside is None:
            return None
        lo, hi = float(last_inside), float(outside)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.inside_outer_envelope(point + sign * mid * d):
                lo = mid
            else:
                hi = mid
        far = sign * 0.5 * (lo + hi)
        return (float(far), 0.0) if far < 0.0 else (0.0, float(far))

    def broad_source_quadrature(
        self,
        boundary_point_mm: Sequence[float],
        direction: Sequence[float],
        *,
        interface: str,
        materials: Sequence[OpticalMaterial] = DEFAULT_WCTE_MATERIALS,
        interval: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[OpticalMaterial, ...], np.ndarray] | None:
        """Return deterministic source points, path weights and material labels.

        ``interval`` may be supplied by a caller that already intersected the
        line with the module envelope, avoiding a duplicate hot-path solve.
        """
        if interval is None:
            interval = self.hardware_interval_from_boundary(
                boundary_point_mm, direction, interface=interface
            )
        if interval is None:
            return None
        t_lo, t_hi = interval
        point = np.asarray(boundary_point_mm, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        d = d / float(np.linalg.norm(d))

        # Split at every radial shell and cap-plane root, then classify each open
        # interval by its midpoint.  This resolves long grazing shell chords with
        # only a few quadrature nodes and no step-size tuning.
        roots: list[float] = [float(t_lo), float(t_hi)]
        rel = point - self.sphere_centre_mm
        b = float(np.dot(rel, d))
        rel2 = float(np.dot(rel, rel))
        for radius in {
            float(self.design.dome_outer_radius_mm),
            float(self.design.dome_inner_radius_mm),
            float(self.design.gel_inner_radius_mm),
            float(self.design.gel_outer_radius_mm),
        }:
            disc = b * b - (rel2 - radius * radius)
            if disc >= 0.0:
                root = math.sqrt(max(disc, 0.0))
                for value in (-b - root, -b + root):
                    if t_lo - 1.0e-8 <= value <= t_hi + 1.0e-8:
                        roots.append(float(value))
        axial0 = self.local_axial_mm(point)
        axial_v = float(np.dot(d, self.inward_axis))
        if abs(axial_v) > 1.0e-15:
            value = (float(self.design.cylinder_height_mm) - axial0) / axial_v
            if t_lo - 1.0e-8 <= value <= t_hi + 1.0e-8:
                roots.append(float(value))
        roots = sorted(roots)
        unique: list[float] = []
        for value in roots:
            if not unique or abs(value - unique[-1]) > 1.0e-7:
                unique.append(value)

        material_by_name = {material.name: material for material in materials}
        order = max(2, int(self.design.broad_quadrature_order))
        nodes, weights = np.polynomial.legendre.leggauss(order)
        source_points: list[np.ndarray] = []
        source_weights: list[float] = []
        source_materials: list[OpticalMaterial] = []
        source_t: list[float] = []
        for left, right in zip(unique[:-1], unique[1:]):
            if right - left <= 1.0e-7:
                continue
            middle = 0.5 * (left + right)
            name = self._transparent_material(point + middle * d)
            material = material_by_name.get(name or "")
            if material is None:
                continue
            half = 0.5 * (right - left)
            centre = 0.5 * (right + left)
            for node, weight in zip(nodes, weights):
                t = centre + half * float(node)
                source_points.append(point + t * d)
                source_weights.append(half * float(weight) * float(material.relative_yield))
                source_materials.append(material)
                source_t.append(float(t))
        if not source_points:
            return None
        return (
            np.ascontiguousarray(np.asarray(source_points), dtype=np.float64),
            np.ascontiguousarray(np.asarray(source_weights), dtype=np.float64),
            tuple(source_materials),
            np.ascontiguousarray(np.asarray(source_t), dtype=np.float64),
        )

    def metadata(self) -> dict[str, object]:
        return {
            "slot": int(self.slot),
            "module_kind": str(self.module_kind),
            "base_mm": self.base_mm.tolist(),
            "inward_axis": self.inward_axis.tolist(),
            "sphere_centre_mm": self.sphere_centre_mm.tolist(),
            "pmt_indices": self.pmt_indices.astype(int).tolist(),
            "design": {
                name: (int(value) if isinstance(value, (int, np.integer)) else float(value))
                for name, value in vars(self.design).items()
            },
        }


def beta_from_kinetic_energy(kinetic_energy_mev: float, mass_mev: float = _MUON_MASS_MEV) -> float:
    kinetic = float(kinetic_energy_mev)
    gamma = 1.0 + max(kinetic, 0.0) / float(mass_mev)
    return float(math.sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma))))


def cherenkov_angle_rad(beta: float, phase_index: float) -> float | None:
    product = float(beta) * float(phase_index)
    if not math.isfinite(product) or product <= 1.0:
        return None
    return float(math.acos(1.0 / product))


def front_facing_cone_fraction(
    direction: Sequence[float], pmt_normal: Sequence[float], theta_rad: float
) -> float:
    """Azimuthal fraction of a Cherenkov cone travelling into a PMT front face.

    PMT normals point out from the photocathode into the optically active region;
    an incident photon therefore requires ``k dot n < 0``.  Integrating that
    inequality around the Cherenkov cone gives this exact closed form.
    """
    d = np.asarray(direction, dtype=np.float64)
    n = np.asarray(pmt_normal, dtype=np.float64)
    d /= float(np.linalg.norm(d))
    n /= float(np.linalg.norm(n))
    u = float(np.clip(np.dot(d, n), -1.0, 1.0))
    transverse = math.sqrt(max(0.0, 1.0 - u * u))
    sine = math.sin(float(theta_rad))
    if transverse <= 1.0e-15 or sine <= 1.0e-15:
        return 1.0 if u * math.cos(float(theta_rad)) < 0.0 else 0.0
    threshold = -u * math.cos(float(theta_rad)) / (sine * transverse)
    if threshold <= -1.0:
        return 0.0
    if threshold >= 1.0:
        return 1.0
    return float(1.0 - math.acos(threshold) / math.pi)


def cone_angular_disk_overlap_fraction(alpha_rad: np.ndarray, theta_rad: float, rho_rad: np.ndarray) -> np.ndarray:
    """Exact azimuthal cone fraction inside a circular angular receiver.

    A source-to-PMT centre at polar angle ``alpha`` is surrounded by an angular
    disk of radius ``rho``.  The result is the fraction of the Cherenkov cone's
    azimuth satisfying the spherical angular-distance inequality.  This avoids
    an empirical Gaussian cone width.
    """
    alpha = np.asarray(alpha_rad, dtype=np.float64)
    rho = np.asarray(rho_rad, dtype=np.float64)
    theta = float(theta_rad)
    denom = math.sin(theta) * np.sin(alpha)
    numerator = np.cos(rho) - math.cos(theta) * np.cos(alpha)
    out = np.zeros(np.broadcast(alpha, rho).shape, dtype=np.float64)
    regular = np.abs(denom) > 1.0e-14
    ratio = np.empty_like(out)
    ratio[regular] = numerator[regular] / denom[regular]
    out[regular & (ratio <= -1.0)] = 1.0
    middle = regular & (ratio > -1.0) & (ratio < 1.0)
    out[middle] = np.arccos(ratio[middle]) / math.pi
    # Degenerate alpha=0/pi: the complete ring is either in or out.
    if np.any(~regular):
        separation = np.abs(alpha[~regular] - theta)
        out[~regular] = (separation <= rho[~regular]).astype(np.float64)
    return np.clip(out, 0.0, 1.0)


def default_pmt_angular_response(cost: np.ndarray) -> np.ndarray:
    """Current production effective PMT angular response (old y0=0.1209 family)."""
    x = np.clip(np.asarray(cost, dtype=np.float64), 0.0, 1.0)
    xn = x ** 3.0777
    return (
        0.1209 + (1.6397 - 0.1209) * (xn / (xn + 0.7942886659271312))
    ) / 1.002379253316015


def finite_disk_solid_angle(distance_mm: np.ndarray, radius_mm: float) -> np.ndarray:
    r = np.maximum(np.asarray(distance_mm, dtype=np.float64), 1.0e-9)
    a = max(float(radius_mm), 0.0)
    if a == 0.0:
        return np.zeros_like(r)
    return 2.0 * math.pi * (1.0 - r / np.sqrt(r * r + a * a))


@dataclass(frozen=True)
class LocalModePrediction:
    """Normalized local charge modes and deterministic source-level nodes.

    ``max_fractions`` supplies geometry-dependent upper bounds for the convex
    charge profile.  A mode whose physical chord vanishes therefore cannot
    claim an arbitrary event charge fraction merely by grazing a dome.
    """

    templates: np.ndarray  # (n_modes, n_pmts), each row sums to one if active
    mode_names: tuple[str, ...]
    node_weights: np.ndarray  # (n_nodes, n_pmts), sums to templates by mode mapping
    node_times_ns: np.ndarray  # (n_nodes, n_pmts)
    node_modes: np.ndarray  # integer mode index for each node
    diagnostics: Mapping[str, object]
    max_fractions: np.ndarray | None = None
    reference_fractions: np.ndarray | None = None


@dataclass(frozen=True)
class ChargeProfileResult:
    fractions: np.ndarray
    expected_pes: np.ndarray
    nll: float
    baseline_nll: float
    improvement: float
    iterations: int
    converged: bool


@dataclass
class MPMTBoundaryModel:
    """Two-mode analytic mPMT boundary-light model."""

    module: ModuleGeometry
    n_detector_pmts: int
    materials: tuple[OpticalMaterial, ...] = DEFAULT_WCTE_MATERIALS
    angular_response: Callable[[np.ndarray], np.ndarray] = default_pmt_angular_response
    max_fraction: float | None = None
    profile_tolerance: float = 1.0e-9
    profile_max_iterations: int = 30

    def __post_init__(self) -> None:
        if self.max_fraction is None:
            self.max_fraction = float(self.module.design.max_hardware_fraction)
        self.max_fraction = float(self.max_fraction)
        if not (0.0 < self.max_fraction < 1.0):
            raise ValueError("max_fraction must lie strictly between zero and one")

    def _core_mode(
        self,
        *,
        boundary_point_mm: np.ndarray,
        direction: np.ndarray,
        interface: str,
        kinetic_energy_mev: float,
        water_start_mm: np.ndarray,
        water_path_to_boundary_mm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        interval = self.module.hardware_interval_from_boundary(
            boundary_point_mm, direction, interface=interface
        )
        local = np.zeros(self.module.pmt_indices.size, dtype=np.float64)
        times = np.full(self.module.pmt_indices.size, np.inf, dtype=np.float64)
        source_points = np.zeros((self.module.pmt_indices.size, 3), dtype=np.float64)
        if interval is None:
            return local, times, source_points, {"active_pmts": 0}
        t_lo, t_hi = interval
        d = direction / float(np.linalg.norm(direction))
        beta = beta_from_kinetic_energy(kinetic_energy_mev)
        theta = cherenkov_angle_rad(beta, self.materials[0].phase_index)
        if theta is None:
            return local, times, source_points, {"active_pmts": 0}
        radius = float(self.module.design.pmt_effective_radius_mm)
        for j, (position, normal) in enumerate(zip(
            self.module.pmt_positions_mm, self.module.pmt_normals
        )):
            rel = position - boundary_point_mm
            t_closest = float(np.dot(rel, d))
            t_use = float(np.clip(t_closest, t_lo, t_hi))
            closest = boundary_point_mm + t_use * d
            distance = float(np.linalg.norm(position - closest))
            if distance >= radius:
                continue
            half = math.sqrt(max(radius * radius - distance * distance, 0.0))
            chord_lo = max(t_lo, t_closest - half)
            chord_hi = min(t_hi, t_closest + half)
            chord = max(0.0, chord_hi - chord_lo)
            if chord <= 0.0:
                continue
            orientation = front_facing_cone_fraction(d, normal, theta)
            weight = chord * orientation
            if weight <= 0.0:
                continue
            t_source = 0.5 * (chord_lo + chord_hi)
            source = boundary_point_mm + t_source * d
            source_points[j] = source
            local[j] = weight
            particle_path = (
                float(np.dot(source - water_start_mm, d))
                if interface == "entry"
                else float(water_path_to_boundary_mm + t_source)
            )
            photon_path = float(np.linalg.norm(position - source))
            times[j] = (
                particle_path / max(beta * _C_MM_PER_NS, 1.0e-12)
                + self.materials[0].group_index * photon_path / _C_MM_PER_NS
            )
        total = float(np.sum(local))
        if total > 0.0:
            local /= total
        return local, times, source_points, {
            "active_pmts": int(np.count_nonzero(local)),
            "raw_weight_sum": total,
            "cherenkov_angle_rad": float(theta),
        }

    def _broad_mode(
        self,
        *,
        boundary_point_mm: np.ndarray,
        direction: np.ndarray,
        interface: str,
        kinetic_energy_mev: float,
        water_start_mm: np.ndarray,
        water_path_to_boundary_mm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        quadrature = self.module.broad_source_quadrature(
            boundary_point_mm,
            direction,
            interface=interface,
            materials=self.materials,
        )
        if quadrature is None:
            return (
                np.zeros(self.module.pmt_indices.size, dtype=np.float64),
                np.empty((0, self.module.pmt_indices.size), dtype=np.float64),
                np.empty((0, self.module.pmt_indices.size), dtype=np.float64),
                {"source_count": 0},
            )
        source_points, source_path_weights, source_materials, source_t = quadrature
        d = direction / float(np.linalg.norm(direction))
        beta = beta_from_kinetic_energy(kinetic_energy_mev)
        n_source = source_points.shape[0]
        n_local = self.module.pmt_indices.size
        node = np.zeros((n_source, n_local), dtype=np.float64)
        node_t = np.full((n_source, n_local), np.inf, dtype=np.float64)
        aperture = float(self.module.design.receiver_aperture_radius_mm)
        for m in range(n_source):
            material = source_materials[m]
            theta = cherenkov_angle_rad(beta, material.phase_index)
            if theta is None:
                continue
            vectors = self.module.pmt_positions_mm - source_points[m][None, :]
            distances = np.linalg.norm(vectors, axis=1)
            good = distances > 1.0e-9
            unit = np.zeros_like(vectors)
            unit[good] = vectors[good] / distances[good, None]
            alpha = np.arccos(np.clip(unit @ d, -1.0, 1.0))
            rho = np.arctan2(aperture, np.maximum(distances, 1.0e-9))
            cone_fraction = cone_angular_disk_overlap_fraction(alpha, theta, rho)
            cost = np.clip(-np.einsum("ij,ij->i", unit, self.module.pmt_normals), 0.0, 1.0)
            response = self.angular_response(cost)
            solid_angle = finite_disk_solid_angle(distances, aperture)
            yield_factor = max(0.0, 1.0 - 1.0 / max((beta * material.phase_index) ** 2, 1.0e-12))
            weight = (
                float(source_path_weights[m])
                * yield_factor
                * cone_fraction
                * response
                * solid_angle
            )
            weight[~good] = 0.0
            node[m] = np.where(np.isfinite(weight) & (weight > 0.0), weight, 0.0)
            particle_path = (
                float(np.dot(source_points[m] - water_start_mm, d))
                if interface == "entry"
                else float(water_path_to_boundary_mm + source_t[m])
            )
            node_t[m, good] = (
                particle_path / max(beta * _C_MM_PER_NS, 1.0e-12)
                + float(material.group_index) * distances[good] / _C_MM_PER_NS
            )
        total = float(np.sum(node))
        if total > 0.0:
            node /= total
        template = np.sum(node, axis=0)
        return template, node, node_t, {
            "source_count": int(n_source),
            "raw_weight_sum": total,
            "materials": sorted({material.name for material in source_materials}),
            "hardware_signed_t_range_mm": [float(np.min(source_t)), float(np.max(source_t))],
        }

    def predict_modes(
        self,
        *,
        boundary_hit: BoundarySurfaceHit,
        direction: Sequence[float],
        interface: str,
        kinetic_energy_mev: float,
        water_start_mm: Sequence[float],
        water_path_to_boundary_mm: float,
    ) -> LocalModePrediction:
        if boundary_hit.surface_kind != "mpmt_dome" or int(boundary_hit.slot or -1) != int(self.module.slot):
            raise ValueError(
                f"boundary hit is not mPMT slot {self.module.slot}: {boundary_hit.metadata()}"
            )
        d = np.asarray(direction, dtype=np.float64)
        d /= float(np.linalg.norm(d))
        boundary = np.asarray(boundary_hit.point_mm, dtype=np.float64)
        water_start = np.asarray(water_start_mm, dtype=np.float64)
        core, core_times, core_sources, core_diag = self._core_mode(
            boundary_point_mm=boundary,
            direction=d,
            interface=interface,
            kinetic_energy_mev=float(kinetic_energy_mev),
            water_start_mm=water_start,
            water_path_to_boundary_mm=float(water_path_to_boundary_mm),
        )
        broad, broad_nodes_local, broad_times_local, broad_diag = self._broad_mode(
            boundary_point_mm=boundary,
            direction=d,
            interface=interface,
            kinetic_energy_mev=float(kinetic_energy_mev),
            water_start_mm=water_start,
            water_path_to_boundary_mm=float(water_path_to_boundary_mm),
        )

        templates = np.zeros((2, self.n_detector_pmts), dtype=np.float64)
        templates[0, self.module.pmt_indices] = core
        templates[1, self.module.pmt_indices] = broad

        # One core node can hold a PMT-dependent source time.  Broad quadrature
        # rows remain source resolved.  Node amplitudes are normalized such that
        # rows belonging to one mode sum to that mode's normalized template.
        core_node = np.zeros((1, self.n_detector_pmts), dtype=np.float64)
        core_node[0, self.module.pmt_indices] = core
        core_node_t = np.full((1, self.n_detector_pmts), np.inf, dtype=np.float64)
        core_node_t[0, self.module.pmt_indices] = core_times
        broad_nodes = np.zeros((broad_nodes_local.shape[0], self.n_detector_pmts), dtype=np.float64)
        broad_node_t = np.full((broad_nodes_local.shape[0], self.n_detector_pmts), np.inf, dtype=np.float64)
        if broad_nodes_local.size:
            broad_nodes[:, self.module.pmt_indices] = broad_nodes_local
            broad_node_t[:, self.module.pmt_indices] = broad_times_local
        node_weights = np.vstack((core_node, broad_nodes))
        node_times = np.vstack((core_node_t, broad_node_t))
        node_modes = np.concatenate((
            np.zeros(1, dtype=np.int8),
            np.ones(broad_nodes.shape[0], dtype=np.int8),
        ))
        return LocalModePrediction(
            templates=np.ascontiguousarray(templates, dtype=np.float64),
            mode_names=("traversed_pmt_core", "transparent_shell_cone"),
            node_weights=np.ascontiguousarray(node_weights, dtype=np.float64),
            node_times_ns=np.ascontiguousarray(node_times, dtype=np.float64),
            node_modes=np.ascontiguousarray(node_modes, dtype=np.int8),
            diagnostics={
                "slot": int(self.module.slot),
                "interface": str(interface),
                "boundary": boundary_hit.metadata(),
                "core": core_diag,
                "broad": broad_diag,
                "core_source_points_mm": core_sources.tolist(),
            },
            max_fractions=np.full(2, float(self.max_fraction), dtype=np.float64),
            reference_fractions=None,
        )

    @staticmethod
    def _poisson_nll(expected: np.ndarray, observed: np.ndarray) -> float:
        lam = np.maximum(np.asarray(expected, dtype=np.float64), 1.0e-12)
        q = np.asarray(observed, dtype=np.float64)
        return float(np.sum(lam - q * np.log(lam)))

    @staticmethod
    def _project_fraction_box_simplex(
        values: np.ndarray,
        upper_bounds: np.ndarray,
        maximum: float,
    ) -> np.ndarray:
        """Euclidean projection onto ``0 <= f <= u, sum(f) <= maximum``.

        The physical finite-cone model supplies a separate candidate-dependent
        upper bound for each local-light component.  A plain non-negative
        simplex projection is insufficient: without the box bounds, a line that
        only clips a tiny transparent-material chord could still assign a large
        event charge fraction to that component.

        If the clipped point violates the total cap, the constrained projection
        has the water-filling form ``clip(values - tau, 0, upper_bounds)``.  The
        scalar ``tau`` is found by monotone bisection; with at most four modes in
        production this is negligible compared with one Emitter evaluation.
        """
        v = np.asarray(values, dtype=np.float64)
        upper = np.asarray(upper_bounds, dtype=np.float64)
        if v.shape != upper.shape:
            raise ValueError("fraction values and upper bounds have different shapes")
        upper = np.where(np.isfinite(upper) & (upper > 0.0), upper, 0.0)
        cap = min(max(float(maximum), 0.0), float(np.sum(upper)))
        if cap <= 0.0 or v.size == 0:
            return np.zeros_like(v)
        clipped = np.clip(v, 0.0, upper)
        if float(np.sum(clipped)) <= cap + 1.0e-15:
            return clipped

        # F(tau) = sum clip(v-tau, 0, upper) is continuous and monotone.
        lo = float(np.min(v - upper))
        hi = float(np.max(v))
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            total = float(np.sum(np.clip(v - mid, 0.0, upper)))
            if total > cap:
                lo = mid
            else:
                hi = mid
        out = np.clip(v - hi, 0.0, upper)
        # Remove sub-ulp excess from the bisection without changing the active
        # set appreciably.
        total = float(np.sum(out))
        if total > cap and total > 0.0:
            out *= cap / total
            out = np.minimum(out, upper)
        return out

    @staticmethod
    def _project_fraction_simplex(values: np.ndarray, maximum: float) -> np.ndarray:
        """Backward-compatible projection used by older tests and utilities."""
        values = np.asarray(values, dtype=np.float64)
        upper = np.full(values.shape, max(float(maximum), 0.0), dtype=np.float64)
        return MPMTBoundaryModel._project_fraction_box_simplex(values, upper, maximum)

    @staticmethod
    def _solve_profile_coordinate(
        base_without_mode: np.ndarray,
        observed: np.ndarray,
        delta: np.ndarray,
        upper: float,
        *,
        max_iterations: int = 6,
    ) -> float:
        """Solve one convex fraction coordinate with a safeguarded Newton step."""
        limit = max(float(upper), 0.0)
        if limit <= 0.0:
            return 0.0

        def derivative(value: float) -> tuple[float, float]:
            expected = np.maximum(base_without_mode + float(value) * delta, 1.0e-12)
            gradient = float(np.sum(delta * (1.0 - observed / expected)))
            curvature = float(np.sum(observed * delta * delta / (expected * expected)))
            return gradient, curvature

        g0, _ = derivative(0.0)
        if g0 >= 0.0:
            return 0.0
        g1, _ = derivative(limit)
        if g1 <= 0.0:
            return limit
        lo, hi = 0.0, limit
        value = 0.5 * limit
        for _ in range(max(1, int(max_iterations))):
            gradient, curvature = derivative(value)
            if gradient > 0.0:
                hi = value
            else:
                lo = value
            if hi - lo <= 1.0e-10 * max(1.0, limit):
                break
            if curvature > 0.0 and math.isfinite(curvature):
                candidate = value - gradient / curvature
            else:
                candidate = math.nan
            if not math.isfinite(candidate) or candidate <= lo or candidate >= hi:
                candidate = 0.5 * (lo + hi)
            value = float(candidate)
        return float(np.clip(value, 0.0, limit))

    @staticmethod
    def _profile_terms(
        baseline: np.ndarray,
        observed: np.ndarray,
        deltas: np.ndarray,
        fractions: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Return convex Poisson objective, gradient and Hessian.

        The total-fraction cap is always below one, so a positive baseline
        remains strictly positive throughout the feasible polytope.  The small
        numerical floor protects only pathological external inputs; it is not
        part of the physical mixture model.
        """
        expected = np.maximum(
            baseline + np.asarray(fractions, dtype=np.float64) @ deltas,
            1.0e-12,
        )
        residual = 1.0 - observed / expected
        gradient = deltas @ residual
        weighted = observed / (expected * expected)
        hessian = (deltas * weighted[None, :]) @ deltas.T
        nll = MPMTBoundaryModel._poisson_nll(expected, observed)
        return float(nll), expected, gradient, hessian

    @staticmethod
    def _fraction_polygon_vertices_2d(
        upper_bounds: np.ndarray,
        maximum: float,
    ) -> np.ndarray:
        """Vertices of ``0<=f<=u, f0+f1<=maximum`` in cyclic order."""
        upper = np.asarray(upper_bounds, dtype=np.float64)
        if upper.shape != (2,):
            raise ValueError("the two-mode polygon requires two upper bounds")
        cap = min(max(float(maximum), 0.0), float(np.sum(upper)))
        candidates = (
            (0.0, 0.0),
            (float(upper[0]), 0.0),
            (0.0, float(upper[1])),
            (float(upper[0]), float(upper[1])),
            (cap, 0.0),
            (0.0, cap),
            (float(upper[0]), cap - float(upper[0])),
            (cap - float(upper[1]), float(upper[1])),
        )
        points: list[np.ndarray] = []
        for x, y in candidates:
            if (
                x >= -1.0e-12
                and y >= -1.0e-12
                and x <= float(upper[0]) + 1.0e-12
                and y <= float(upper[1]) + 1.0e-12
                and x + y <= cap + 1.0e-12
            ):
                point = np.asarray((
                    float(np.clip(x, 0.0, upper[0])),
                    float(np.clip(y, 0.0, upper[1])),
                ), dtype=np.float64)
                if float(np.sum(point)) > cap:
                    point *= cap / max(float(np.sum(point)), 1.0e-300)
                if not any(
                    float(np.linalg.norm(point - old)) <= 1.0e-10
                    for old in points
                ):
                    points.append(point)
        if not points:
            return np.zeros((1, 2), dtype=np.float64)
        centre = np.mean(np.asarray(points), axis=0)
        points.sort(key=lambda point: math.atan2(
            float(point[1] - centre[1]),
            float(point[0] - centre[0]),
        ))
        return np.ascontiguousarray(np.asarray(points), dtype=np.float64)

    def _projected_gradient_profile(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        deltas: np.ndarray,
        upper: np.ndarray,
        start: np.ndarray,
    ) -> tuple[np.ndarray, float, int, bool]:
        """Globally convergent projected-gradient solve on the capped box.

        The local-light problem has at most four variables.  A Hessian-derived
        Lipschitz estimate plus a descent-lemma backtrack gives a deterministic
        convex optimizer without SciPy or an unconstrained nuisance Minuit.
        """
        fractions = self._project_fraction_box_simplex(
            start, upper, float(self.max_fraction)
        )
        nll, _, gradient, hessian = self._profile_terms(
            baseline, observed, deltas, fractions
        )
        best_fractions = fractions.copy()
        best_nll = float(nll)
        # Two-mode fits normally finish in a few steps.  Four-mode joint
        # entry/exit fits need a larger formal cap for worst-case nearly
        # collinear templates; each iteration is only a tiny dense operation.
        max_iterations = max(int(self.profile_max_iterations), 256)
        mapping_tolerance = max(float(self.profile_tolerance), 1.0e-8)
        previous_lipschitz = 0.0
        for iteration in range(1, max_iterations + 1):
            try:
                local_lipschitz = float(np.linalg.eigvalsh(hessian)[-1])
            except np.linalg.LinAlgError:
                local_lipschitz = float(np.max(np.sum(np.abs(hessian), axis=1)))
            if not math.isfinite(local_lipschitz) or local_lipschitz <= 0.0:
                local_lipschitz = 1.0
            lipschitz = max(
                local_lipschitz,
                0.5 * previous_lipschitz,
                1.0e-10,
            )
            accepted = False
            trial = fractions
            trial_nll = nll
            trial_gradient = gradient
            trial_hessian = hessian
            displacement = np.zeros_like(fractions)
            for _ in range(64):
                trial = self._project_fraction_box_simplex(
                    fractions - gradient / lipschitz,
                    upper,
                    float(self.max_fraction),
                )
                displacement = trial - fractions
                trial_nll, _, trial_gradient, trial_hessian = self._profile_terms(
                    baseline, observed, deltas, trial
                )
                descent_bound = (
                    nll
                    + float(np.dot(gradient, displacement))
                    + 0.5 * lipschitz * float(np.dot(displacement, displacement))
                )
                if trial_nll <= descent_bound + 1.0e-12:
                    accepted = True
                    break
                lipschitz *= 2.0
            if not accepted:
                return best_fractions, best_nll, iteration, False

            projected_mapping = lipschitz * displacement
            fractions = trial
            nll = float(trial_nll)
            gradient = trial_gradient
            hessian = trial_hessian
            previous_lipschitz = float(lipschitz)
            if nll < best_nll:
                best_nll = float(nll)
                best_fractions = fractions.copy()
            if float(np.linalg.norm(projected_mapping, ord=np.inf)) <= mapping_tolerance:
                return best_fractions, best_nll, iteration, True
        return best_fractions, best_nll, max_iterations, False

    def _interior_newton_profile_2d(
        self,
        baseline: np.ndarray,
        observed: np.ndarray,
        deltas: np.ndarray,
        upper: np.ndarray,
        start: np.ndarray,
    ) -> tuple[np.ndarray, float, int, bool]:
        """Find a possible two-mode interior optimum with damped Newton steps.

        For a convex objective on a polygon, the global minimizer is either on
        an edge or at an interior stationary point.  Every edge is minimized
        independently by :meth:`profile_charge`; this routine therefore needs
        only search for ``gradient == 0`` in the strict interior.  It avoids the
        hundreds of projected-gradient iterations that were previously paid at
        every FCN even after the exact edge solution was already known.

        The step is clipped analytically to the box and total-fraction planes
        and accepted by an Armijo backtrack.  Failure to find an interior point
        is harmless: the exact edge minimum remains authoritative.
        """
        cap = min(float(self.max_fraction), float(np.sum(upper)))
        if cap <= 0.0:
            zero = np.zeros(2, dtype=np.float64)
            return zero, self._poisson_nll(baseline, observed), 0, False

        # Blend a projected warm start with a guaranteed strict-interior point.
        interior = np.minimum(0.25 * upper, 0.20 * cap)
        if float(np.sum(interior)) >= 0.8 * cap:
            interior *= (0.5 * cap) / max(float(np.sum(interior)), 1.0e-300)
        fractions = self._project_fraction_box_simplex(
            0.75 * np.asarray(start, dtype=np.float64) + 0.25 * interior,
            upper,
            cap,
        )
        margin = 1.0e-12
        fractions = np.minimum(
            np.maximum(fractions, margin),
            np.maximum(upper - margin, margin),
        )
        total = float(np.sum(fractions))
        if total >= cap - margin:
            fractions *= max(cap - 10.0 * margin, 0.5 * cap) / max(
                total, 1.0e-300
            )

        nll, _, gradient, hessian = self._profile_terms(
            baseline, observed, deltas, fractions
        )
        best = fractions.copy()
        best_nll = float(nll)
        tolerance = max(float(self.profile_tolerance), 1.0e-10)
        for iteration in range(1, 33):
            if float(np.linalg.norm(gradient, ord=np.inf)) <= tolerance:
                return best, best_nll, iteration, True
            try:
                eigenvalues = np.linalg.eigvalsh(hessian)
                regularization = max(0.0, 1.0e-12 - float(eigenvalues[0]))
                step = np.linalg.solve(
                    hessian + regularization * np.eye(2), -gradient
                )
            except np.linalg.LinAlgError:
                return best, best_nll, iteration, False
            if not np.all(np.isfinite(step)) or float(np.dot(gradient, step)) >= 0.0:
                return best, best_nll, iteration, False

            alpha_max = 1.0
            for j in range(2):
                if step[j] > 0.0:
                    alpha_max = min(
                        alpha_max,
                        (float(upper[j]) - float(fractions[j])) / float(step[j]),
                    )
                elif step[j] < 0.0:
                    alpha_max = min(
                        alpha_max,
                        -float(fractions[j]) / float(step[j]),
                    )
            step_sum = float(np.sum(step))
            if step_sum > 0.0:
                alpha_max = min(
                    alpha_max,
                    (cap - float(np.sum(fractions))) / step_sum,
                )
            alpha = min(1.0, max(0.0, 0.995 * alpha_max))
            if alpha <= 1.0e-14:
                return best, best_nll, iteration, False

            directional = float(np.dot(gradient, step))
            accepted = False
            for _ in range(32):
                trial = fractions + alpha * step
                if (
                    np.any(trial <= 0.0)
                    or np.any(trial >= upper)
                    or float(np.sum(trial)) >= cap
                ):
                    alpha *= 0.5
                    continue
                trial_nll, _, trial_gradient, trial_hessian = self._profile_terms(
                    baseline, observed, deltas, trial
                )
                if trial_nll <= nll + 1.0e-4 * alpha * directional:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                return best, best_nll, iteration, False
            fractions = trial
            nll = float(trial_nll)
            gradient = trial_gradient
            hessian = trial_hessian
            if nll < best_nll:
                best_nll = nll
                best = fractions.copy()
        return best, best_nll, 32, False

    def profile_charge(
        self,
        baseline_expected_pes: Sequence[float],
        observed_pes: Sequence[float],
        modes: LocalModePrediction,
    ) -> ChargeProfileResult:
        baseline = np.asarray(baseline_expected_pes, dtype=np.float64)
        observed = np.asarray(observed_pes, dtype=np.float64)
        templates = np.asarray(modes.templates, dtype=np.float64)
        if baseline.shape != observed.shape or templates.shape[1:] != baseline.shape:
            raise ValueError("baseline, observation and local templates have incompatible shapes")
        q_total = float(np.sum(observed))
        baseline_nll = self._poisson_nll(baseline, observed)
        if modes.max_fractions is None:
            upper_all = np.full(
                templates.shape[0], float(self.max_fraction), dtype=np.float64
            )
        else:
            upper_all = np.asarray(modes.max_fractions, dtype=np.float64)
            if upper_all.shape != (templates.shape[0],):
                raise ValueError("local-mode maximum fractions have the wrong shape")
            upper_all = np.where(
                np.isfinite(upper_all) & (upper_all > 0.0), upper_all, 0.0
            )
        template_sums = np.sum(templates, axis=1)
        active_modes = np.flatnonzero((template_sums > 0.0) & (upper_all > 0.0))
        if q_total <= 0.0 or active_modes.size == 0:
            return ChargeProfileResult(
                fractions=np.zeros(templates.shape[0], dtype=np.float64),
                expected_pes=np.ascontiguousarray(baseline, dtype=np.float64),
                nll=float(baseline_nll), baseline_nll=float(baseline_nll),
                improvement=0.0, iterations=0, converged=True,
            )
        deltas = q_total * templates[active_modes] - baseline[None, :]
        upper = np.minimum(upper_all[active_modes], float(self.max_fraction))

        # Exact one-dimensional solve.
        if active_modes.size == 1:
            value = self._solve_profile_coordinate(
                baseline,
                observed,
                deltas[0],
                min(float(upper[0]), float(self.max_fraction)),
                max_iterations=48,
            )
            fractions = np.asarray((value,), dtype=np.float64)
            iterations = 1
            converged = True
        else:
            # Coordinate minimization supplies a cheap, bright-PMT-aware start.
            fractions = np.zeros(active_modes.size, dtype=np.float64)
            for _ in range(2):
                for j in range(active_modes.size):
                    others = float(np.sum(fractions) - fractions[j])
                    coordinate_upper = min(
                        float(upper[j]),
                        max(0.0, float(self.max_fraction) - others),
                    )
                    base_without = (
                        baseline + fractions @ deltas - fractions[j] * deltas[j]
                    )
                    fractions[j] = self._solve_profile_coordinate(
                        base_without,
                        observed,
                        deltas[j],
                        coordinate_upper,
                        max_iterations=32,
                    )
            fractions = self._project_fraction_box_simplex(
                fractions, upper, float(self.max_fraction)
            )

            # For the normal two-component module model, solve every polygon
            # edge exactly.  This removes the slow/stalling boundary cases that
            # a projected Newton step can mishandle when one component is at an
            # upper bound or the total cap is active.
            if active_modes.size == 2:
                vertices = self._fraction_polygon_vertices_2d(
                    upper, float(self.max_fraction)
                )
                edge_best = fractions.copy()
                edge_best_nll = self._poisson_nll(
                    np.maximum(baseline + edge_best @ deltas, 1.0e-12),
                    observed,
                )
                if vertices.shape[0] == 1:
                    edge_best = vertices[0].copy()
                    edge_best_nll = self._poisson_nll(
                        np.maximum(baseline + edge_best @ deltas, 1.0e-12),
                        observed,
                    )
                else:
                    for edge_index in range(vertices.shape[0]):
                        left = vertices[edge_index]
                        right = vertices[(edge_index + 1) % vertices.shape[0]]
                        edge_vector = right - left
                        edge_t = self._solve_profile_coordinate(
                            baseline + left @ deltas,
                            observed,
                            edge_vector @ deltas,
                            1.0,
                            max_iterations=48,
                        )
                        candidate = left + float(edge_t) * edge_vector
                        candidate_nll = self._poisson_nll(
                            np.maximum(baseline + candidate @ deltas, 1.0e-12),
                            observed,
                        )
                        if candidate_nll < edge_best_nll:
                            edge_best = candidate
                            edge_best_nll = float(candidate_nll)
                warm_nll = self._poisson_nll(
                    np.maximum(baseline + fractions @ deltas, 1.0e-12),
                    observed,
                )
                if edge_best_nll < warm_nll:
                    fractions = edge_best
                interior_best, interior_nll, iterations, interior_converged = (
                    self._interior_newton_profile_2d(
                        baseline, observed, deltas, upper, fractions
                    )
                )
                if interior_nll < edge_best_nll:
                    fractions = interior_best
                    edge_best_nll = float(interior_nll)
                else:
                    fractions = edge_best
                # The edge solve is exact even when there is no interior
                # stationary point, so the global two-mode result is converged.
                converged = True
            else:
                fractions, _, iterations, converged = self._projected_gradient_profile(
                    baseline, observed, deltas, upper, fractions
                )

        final_expected = np.maximum(baseline + fractions @ deltas, 1.0e-12)
        full_fractions = np.zeros(templates.shape[0], dtype=np.float64)
        full_fractions[active_modes] = fractions
        final_nll = self._poisson_nll(final_expected, observed)
        return ChargeProfileResult(
            fractions=np.ascontiguousarray(full_fractions, dtype=np.float64),
            expected_pes=np.ascontiguousarray(final_expected, dtype=np.float64),
            nll=float(final_nll),
            baseline_nll=float(baseline_nll),
            improvement=float(baseline_nll - final_nll),
            iterations=int(iterations),
            converged=bool(converged),
        )

    def augment_timing_prediction(
        self,
        timing_prediction: object,
        *,
        modes: LocalModePrediction,
        fractions: Sequence[float],
        observed_pes: Sequence[float],
    ) -> object:
        """Append local deterministic nodes to an Emitter TimingPrediction.

        The function uses the public attribute protocol rather than importing
        PMT internals.  If the prediction is not the deferred first-arrival form,
        it is returned unchanged; charge reconstruction remains valid.
        """
        from .Emitter import TimingPrediction

        if not isinstance(timing_prediction, TimingPrediction):
            return timing_prediction
        base_mu = getattr(timing_prediction, "first_arrival_deferred_base_mu", None)
        base_t = getattr(timing_prediction, "first_arrival_deferred_base_t", None)
        active = getattr(timing_prediction, "first_arrival_active_indices", None)
        scale = getattr(timing_prediction, "first_arrival_node_pe_scale", None)
        if base_mu is None or base_t is None or active is None or scale is None:
            return timing_prediction
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            return timing_prediction
        fractions = np.asarray(fractions, dtype=np.float64)
        total_fraction = float(np.sum(fractions))
        if total_fraction <= 0.0:
            return timing_prediction
        active = np.asarray(active, dtype=np.int64)
        base_mu_array = np.asarray(base_mu, dtype=np.float64) * max(0.0, 1.0 - total_fraction)
        base_t_array = np.asarray(base_t, dtype=np.float64)
        q_total = float(np.sum(np.asarray(observed_pes, dtype=np.float64)))

        local_rows: list[np.ndarray] = []
        local_times: list[np.ndarray] = []
        for row, times, mode_index in zip(
            modes.node_weights, modes.node_times_ns, modes.node_modes
        ):
            fraction = float(fractions[int(mode_index)])
            if fraction <= 0.0:
                continue
            expected = q_total * fraction * np.asarray(row, dtype=np.float64)
            local = expected[active] / scale
            if not np.any(local > 0.0):
                continue
            local_rows.append(local)
            local_times.append(np.asarray(times, dtype=np.float64)[active])
        if not local_rows:
            return timing_prediction
        new_base_mu = np.vstack((base_mu_array, np.asarray(local_rows, dtype=np.float64)))
        new_base_t = np.vstack((base_t_array, np.asarray(local_times, dtype=np.float64)))

        reflection_u = getattr(timing_prediction, "first_arrival_reflection_u", None)
        if reflection_u is not None:
            reflection_u = np.asarray(reflection_u, dtype=np.float64) * max(0.0, 1.0 - total_fraction)
        nominal = np.asarray(timing_prediction, dtype=np.float64)
        return TimingPrediction(
            nominal,
            active_indices=active,
            deferred_base_mu=np.ascontiguousarray(new_base_mu, dtype=np.float32),
            deferred_base_t=np.ascontiguousarray(new_base_t, dtype=np.float32),
            reflection_u=None if reflection_u is None else np.ascontiguousarray(reflection_u, dtype=np.float64),
            reflection_tbase=getattr(timing_prediction, "first_arrival_reflection_tbase", None),
            reflection_transfer_active=getattr(timing_prediction, "first_arrival_reflection_transfer_active", None),
            reflection_time_offset_active=getattr(timing_prediction, "first_arrival_reflection_time_offset_active", None),
            reflection_patch_min_time_offset=getattr(timing_prediction, "first_arrival_reflection_patch_min_time_offset", None),
            reflection_patch_max_time_offset=getattr(timing_prediction, "first_arrival_reflection_patch_max_time_offset", None),
            reflection_n_bins=getattr(timing_prediction, "first_arrival_reflection_n_bins", None),
            node_pe_scale=scale,
        )

    def model_selection_penalty(
        self,
        *,
        observed_pes: Sequence[float],
        n_tested_interfaces: int = 1,
        n_active_modes: int = 2,
    ) -> float:
        """Conservative negative-log-likelihood BIC plus discrete-trial penalty."""
        n_obs = max(int(np.asarray(observed_pes).size), 2)
        return float(
            0.5 * max(int(n_active_modes), 0) * math.log(float(n_obs))
            + math.log(float(max(int(n_tested_interfaces), 1)))
        )


def select_boundary_hit(track: RangeClippedTrack, interface: str) -> BoundarySurfaceHit | None:
    label = str(interface).strip().lower()
    if label == "entry":
        return track.start_boundary_hit
    if label == "exit":
        return track.exit_boundary_hit if track.exits_detector else None
    raise ValueError("interface must be 'entry' or 'exit'")


@dataclass(frozen=True)
class MPMTBoundaryInterface:
    """One explicit mPMT boundary subclass in a composite hypothesis.

    ``required_pmt_index`` defines the more specific ``mPMT/WCPMT`` subclass.
    It is a discrete geometry constraint, not a fitted parameter: a candidate
    must continue to traverse that placed PMT's transparent sector throughout
    optimization.  This prevents a bright local-flash hypothesis from escaping
    continuously into the familiar dome-tangent/outer-shell basin.
    """

    interface: str
    model: object
    required_pmt_index: int | None = None

    def __post_init__(self) -> None:
        label = str(self.interface).strip().lower()
        if label not in {"entry", "exit"}:
            raise ValueError("interface must be 'entry' or 'exit'")
        object.__setattr__(self, "interface", label)
        required = self.required_pmt_index
        if required is not None:
            required = int(required)
            local = np.asarray(self.model.module.pmt_indices, dtype=np.int64)
            if required not in set(int(x) for x in local):
                raise ValueError(
                    "required_pmt_index does not belong to the selected module"
                )
            object.__setattr__(self, "required_pmt_index", required)

    @property
    def slot(self) -> int:
        return int(self.model.module.slot)

    @property
    def subclass(self) -> str:
        return "wcpmt" if self.required_pmt_index is not None else "module_shell"


@dataclass
class CompositeMPMTBoundaryModel:
    """Joint profile for one or more explicit entry/exit mPMT interfaces.

    Entry and exit are independent refinements of the ordinary cosmic start/end
    topology.  This object keeps those discrete constraints explicit while
    profiling all local charge modes in one convex problem.  The water-only
    model remains the exact zero-fraction point.
    """

    interfaces: tuple[MPMTBoundaryInterface, ...]
    max_fraction: float = 0.45
    profile_tolerance: float = 1.0e-9
    profile_max_iterations: int = 30

    def __post_init__(self) -> None:
        self.interfaces = tuple(self.interfaces)
        if not self.interfaces:
            raise ValueError("at least one mPMT boundary interface is required")
        labels = [item.interface for item in self.interfaces]
        if len(labels) != len(set(labels)):
            raise ValueError("a composite hypothesis may contain at most one entry and one exit interface")
        keys = [(item.interface, item.slot) for item in self.interfaces]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate mPMT boundary interface")
        if not (0.0 < float(self.max_fraction) < 1.0):
            raise ValueError("max_fraction must lie strictly between zero and one")

    @property
    def module_indices(self) -> np.ndarray:
        return np.unique(np.concatenate([
            item.model.module.pmt_indices for item in self.interfaces
        ])).astype(np.int32, copy=False)

    @property
    def n_profile_parameters(self) -> int:
        return int(sum(
            int(getattr(item.model, "n_profile_parameters", 2))
            for item in self.interfaces
        ))

    def validate_track(self, track: RangeClippedTrack) -> bool:
        for item in self.interfaces:
            hit = select_boundary_hit(track, item.interface)
            if hit is None or hit.surface_kind != "mpmt_dome":
                return False
            if hit.slot is None or int(hit.slot) != item.slot:
                return False
        return True

    def predict_track_modes(
        self,
        *,
        track: RangeClippedTrack,
        direction: Sequence[float],
        kinetic_energy_at_water_entry_mev: float,
        range_lookup: object,
        emitter: object | None = None,
    ) -> LocalModePrediction:
        if not self.validate_track(track):
            raise ValueError("track does not satisfy the explicit mPMT boundary interfaces")
        predictions: list[LocalModePrediction] = []
        names: list[str] = []
        diagnostics: list[dict[str, object]] = []
        for item in self.interfaces:
            hit = select_boundary_hit(track, item.interface)
            assert hit is not None
            if item.interface == "entry":
                kinetic = float(kinetic_energy_at_water_entry_mev)
                water_path = 0.0
            else:
                remaining_range = max(
                    0.0, float(track.full_range_mm) - float(track.visible_length_mm)
                )
                kinetic = float(range_lookup.range_mm_to_energy(remaining_range))
                water_path = float(track.distance_to_boundary_mm + track.inset_mm)
            if hasattr(item.model, "predict_profile_modes"):
                if emitter is None:
                    raise ValueError("profiled finite-cone mPMT model requires the current Emitter")
                prediction = item.model.predict_profile_modes(
                    boundary_hit=hit,
                    direction=direction,
                    interface=item.interface,
                    kinetic_energy_mev=kinetic,
                    emitter=emitter,
                    boundary_particle_time_ns=0.0,
                )
            else:
                prediction = item.model.predict_modes(
                    boundary_hit=hit,
                    direction=direction,
                    interface=item.interface,
                    kinetic_energy_mev=kinetic,
                    water_start_mm=track.start,
                    water_path_to_boundary_mm=water_path,
                )
            if item.required_pmt_index is not None:
                diagnostics_row = dict(prediction.diagnostics)
                physical = diagnostics_row.get("physical")
                traversed: set[int] = set()
                if isinstance(physical, Mapping):
                    for row in physical.get("traversed_pmts", ()) or ():
                        if isinstance(row, Mapping):
                            value = row.get(
                                "detector_index", row.get("local_index", -1)
                            )
                        else:
                            value = row
                        try:
                            value = int(value)
                        except (TypeError, ValueError):
                            continue
                        if value >= 0:
                            traversed.add(value)
                if int(item.required_pmt_index) not in traversed:
                    raise ValueError(
                        "track left the required WCPMT transparent-sector subclass"
                    )
            predictions.append(prediction)
            names.extend([
                f"{item.interface}:slot{item.slot}:{name}"
                for name in prediction.mode_names
            ])
            diagnostics.append({
                "interface": item.interface,
                "slot": item.slot,
                "subclass": item.subclass,
                "required_pmt_index": (
                    None if item.required_pmt_index is None
                    else int(item.required_pmt_index)
                ),
                "kinetic_energy_at_interface_mev": kinetic,
                "prediction": dict(prediction.diagnostics),
            })

        templates = np.concatenate([p.templates for p in predictions], axis=0)
        node_weights: list[np.ndarray] = []
        node_times: list[np.ndarray] = []
        node_modes: list[np.ndarray] = []
        maximums: list[np.ndarray] = []
        references: list[np.ndarray] = []
        have_references = True
        mode_offset = 0
        for prediction in predictions:
            node_weights.append(np.asarray(prediction.node_weights, dtype=np.float64))
            node_times.append(np.asarray(prediction.node_times_ns, dtype=np.float64))
            node_modes.append(np.asarray(prediction.node_modes, dtype=np.int32) + mode_offset)
            n_modes = int(prediction.templates.shape[0])
            if prediction.max_fractions is None:
                maximums.append(np.full(n_modes, float(self.max_fraction), dtype=np.float64))
            else:
                bounds = np.asarray(prediction.max_fractions, dtype=np.float64)
                if bounds.shape != (n_modes,):
                    raise ValueError("member mPMT maximum fractions have the wrong shape")
                maximums.append(bounds)
            if prediction.reference_fractions is None:
                have_references = False
            else:
                refs = np.asarray(prediction.reference_fractions, dtype=np.float64)
                if refs.shape != (n_modes,):
                    raise ValueError("member mPMT reference fractions have the wrong shape")
                references.append(refs)
            mode_offset += n_modes

        n_pmts = int(templates.shape[1])
        if node_weights:
            combined_node_weights = np.concatenate(node_weights, axis=0)
            combined_node_times = np.concatenate(node_times, axis=0)
            combined_node_modes = np.concatenate(node_modes)
        else:
            combined_node_weights = np.empty((0, n_pmts), dtype=np.float64)
            combined_node_times = np.empty((0, n_pmts), dtype=np.float64)
            combined_node_modes = np.empty(0, dtype=np.int32)
        return LocalModePrediction(
            templates=np.ascontiguousarray(templates, dtype=np.float64),
            mode_names=tuple(names),
            node_weights=np.ascontiguousarray(combined_node_weights, dtype=np.float64),
            node_times_ns=np.ascontiguousarray(combined_node_times, dtype=np.float64),
            node_modes=np.ascontiguousarray(combined_node_modes, dtype=np.int32),
            diagnostics={"interfaces": diagnostics},
            max_fractions=np.ascontiguousarray(
                np.concatenate(maximums), dtype=np.float64
            ),
            reference_fractions=(
                np.ascontiguousarray(np.concatenate(references), dtype=np.float64)
                if have_references else None
            ),
        )

    def profile_charge(
        self,
        baseline_expected_pes: Sequence[float],
        observed_pes: Sequence[float],
        modes: LocalModePrediction,
    ) -> ChargeProfileResult:
        # Reuse the validated convex solver with a composite total-fraction cap.
        first = self.interfaces[0].model
        n_detector_pmts = int(getattr(
            first, "n_detector_pmts",
            getattr(getattr(first, "physical", None), "n_detector_pmts", 0),
        ))
        if n_detector_pmts <= 0:
            n_detector_pmts = int(np.asarray(baseline_expected_pes).size)
        proxy = MPMTBoundaryModel(
            module=first.module,
            n_detector_pmts=n_detector_pmts,
            max_fraction=float(self.max_fraction),
            profile_tolerance=float(self.profile_tolerance),
            profile_max_iterations=int(self.profile_max_iterations),
        )
        return proxy.profile_charge(baseline_expected_pes, observed_pes, modes)

    def augment_timing_prediction(self, timing_prediction: object, **kwargs: object) -> object:
        # Any member has the same public TimingPrediction protocol.
        return self.interfaces[0].model.augment_timing_prediction(timing_prediction, **kwargs)

    def model_selection_penalty(
        self,
        *,
        observed_pes: Sequence[float],
        n_tested_interfaces: int = 1,
        n_active_modes: int | None = None,
    ) -> float:
        n_obs = max(int(np.asarray(observed_pes).size), 2)
        k = self.n_profile_parameters if n_active_modes is None else max(int(n_active_modes), 0)
        return float(0.5 * k * math.log(float(n_obs)) + math.log(float(max(int(n_tested_interfaces), 1))))

    def metadata(self) -> dict[str, object]:
        return {
            "interfaces": [
                {"interface": item.interface, "slot": item.slot,
                 "subclass": item.subclass,
                 "required_pmt_index": (
                     None if item.required_pmt_index is None
                     else int(item.required_pmt_index)
                 ),
                 "module_kind": item.model.module.module_kind}
                for item in self.interfaces
            ],
            "max_fraction": float(self.max_fraction),
            "n_profile_parameters": int(self.n_profile_parameters),
            "timing_recommendation": "mask_crossed_modules",
        }
