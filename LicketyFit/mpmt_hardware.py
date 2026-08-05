"""Fast, geometry-derived Cherenkov light from charged-particle mPMT crossings.

This module augments an explicit mPMT-entry or mPMT-exit cosmic hypothesis.
It does *not* trace Geant4 optical photons.  Instead it evaluates a small,
deterministic line quadrature through the transparent parts of the module and
computes the finite-aperture fraction of each Cherenkov cone that reaches the
local PMTs.

The amplitude is expressed in the same unnormalised units as ``Emitter``.  It
is therefore added to the ordinary water prediction *before* the existing
per-event charge conditioning.  There is no freely fitted event-by-event
hardware-light fraction: moving the track away from the transparent material
makes the term vanish geometrically.

The key normalisation identity is

    integral ds f_disk(s) -> a^2 / (2 r sin(theta_C)^2),

for a distant circular receiver of radius ``a``.  This connects the finite-cone
line integral exactly to LicketyFit's primary ``N_geo`` convention and fixes the
raw conversion factor without WCSim event templates or a fitted scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from .cosmic_track_fit import BoundarySurfaceHit
from .mpmt_boundary import (
    ModuleGeometry,
    OpticalMaterial,
    beta_from_kinetic_energy,
    cherenkov_angle_rad,
    cone_angular_disk_overlap_fraction,
    default_pmt_angular_response,
)

_C_MM_PER_NS = 299.792458


@dataclass(frozen=True)
class HardwareMaterial:
    """Effective phase/group optics for a short transparent module segment."""

    name: str
    phase_index: float
    group_index: float


# The effective values lie in the central detected wavelength band of the
# wavelength-dependent WCSim tables.  They are detector-design constants, not
# event-derived calibrations, and are emitted in diagnostics for replacement by
# a future geometry/material adapter.
WCTE_SILGEL = HardwareMaterial("SilGel_WCTE", 1.46, 1.49)
WCTE_ACRYLIC = HardwareMaterial("G4_PLEXIGLASS", 1.49, 1.51)
WCTE_GLASS = HardwareMaterial("GlassWCTE", 1.49, 1.52)


@dataclass(frozen=True)
class WCTEInSituPMTDesign:
    """Local dimensions copied from the WCTE in-situ construction."""

    wc_pmt_outer_radius_mm: float = 325.603
    wc_pmt_inner_radius_mm: float = 263.003
    wc_pmt_opening_half_angle_deg: float = 8.14

    glass_outer_radius_mm: float = 53.0
    glass_thickness_mm: float = 2.0
    glass_cut_mm: float = 34.597
    glass_centre_radius_mm: float = 267.406

    reflector_inner_radius_mm: float = 36.96
    reflector_mouth_radius_mm: float = 45.0
    reflector_height_mm: float = 13.0
    reflector_lower_z_mm: float = 307.403
    reflector_upper_z_mm: float = 320.403
    reflector_thickness_mm: float = 0.8

    source_quadrature_order: int = 10
    cap_azimuth_samples: int = 256

    @property
    def glass_inner_radius_mm(self) -> float:
        return float(self.glass_outer_radius_mm - self.glass_thickness_mm)

    @property
    def glass_rim_radius_mm(self) -> float:
        return float(math.sqrt(max(
            self.glass_outer_radius_mm ** 2 - self.glass_cut_mm ** 2, 0.0
        )))


@dataclass(frozen=True)
class RawHardwarePrediction:
    """Unnormalised charge components and source-resolved timing nodes.

    ``raw_charge_modes`` contains the physically distinct local WCPMT and
    outer-shell contributions.  Keeping them separate lets the production
    likelihood profile uncertain local optical throughput without changing the
    geometry-dependent shapes.
    """

    raw_charge: np.ndarray
    raw_charge_modes: np.ndarray
    mode_names: tuple[str, ...]
    node_mu_raw: np.ndarray
    node_t_ns: np.ndarray
    node_modes: np.ndarray
    diagnostics: Mapping[str, object]

    @property
    def active(self) -> bool:
        return bool(np.any(np.asarray(self.raw_charge) > 0.0))


@dataclass(frozen=True)
class CombinedRawPrediction:
    expected_pes: np.ndarray
    timing_pes: np.ndarray
    norm: float
    raw_charge: np.ndarray
    raw_timing: np.ndarray


def _quadratic_roots(a: float, b: float, c: float) -> tuple[float, ...]:
    eps = 1.0e-14
    if abs(a) <= eps:
        if abs(b) <= eps:
            return ()
        return (-c / b,)
    disc = b * b - 4.0 * a * c
    if disc < -1.0e-12:
        return ()
    disc = max(disc, 0.0)
    root = math.sqrt(disc)
    q = -0.5 * (b + math.copysign(root, b)) if root > 0.0 else -0.5 * b
    if abs(q) <= eps:
        return (-b / (2.0 * a),)
    x1 = q / a
    x2 = c / q
    if abs(x1 - x2) <= 1.0e-12:
        return (x1,)
    return (min(x1, x2), max(x1, x2))


def _sphere_line_roots(
    point: np.ndarray, direction: np.ndarray, centre: np.ndarray, radius: float
) -> tuple[float, ...]:
    rel = point - centre
    b = 2.0 * float(np.dot(rel, direction))
    c = float(np.dot(rel, rel) - radius * radius)
    return _quadratic_roots(1.0, b, c)


def _unique_sorted(values: Sequence[float], *, tolerance: float = 1.0e-8) -> list[float]:
    out: list[float] = []
    for value in sorted(float(x) for x in values if math.isfinite(float(x))):
        if not out or abs(value - out[-1]) > tolerance:
            out.append(value)
    return out


def _segment_segment_distance(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
) -> float:
    """Minimum Euclidean distance between two closed three-dimensional segments."""
    u = np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)
    v = np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
    w = np.asarray(p0, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
    a = float(np.dot(u, u)); b = float(np.dot(u, v)); c = float(np.dot(v, v))
    d = float(np.dot(u, w)); e = float(np.dot(v, w)); eps = 1.0e-14
    denominator = a * c - b * b
    if a <= eps and c <= eps:
        return float(np.linalg.norm(w))
    if a <= eps:
        s = 0.0; t = float(np.clip(e / max(c, eps), 0.0, 1.0))
    elif c <= eps:
        t = 0.0; s = float(np.clip(-d / max(a, eps), 0.0, 1.0))
    else:
        s = float(np.clip((b * e - c * d) / denominator, 0.0, 1.0)) if denominator > eps else 0.0
        t = (b * s + e) / c
        if t < 0.0:
            t = 0.0; s = float(np.clip(-d / a, 0.0, 1.0))
        elif t > 1.0:
            t = 1.0; s = float(np.clip((b - d) / a, 0.0, 1.0))
    return float(np.linalg.norm(w + s * u - t * v))


def _predicate_intervals(
    roots: Sequence[float],
    predicate,
    t_lo: float,
    t_hi: float,
) -> list[tuple[float, float]]:
    clipped = [float(t_lo), float(t_hi)]
    clipped.extend(
        float(x) for x in roots
        if float(t_lo) - 1.0e-9 <= float(x) <= float(t_hi) + 1.0e-9
    )
    bounds = _unique_sorted(clipped)
    intervals: list[tuple[float, float]] = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        if right - left <= 1.0e-8:
            continue
        middle = 0.5 * (left + right)
        if bool(predicate(middle)):
            intervals.append((float(left), float(right)))
    return intervals


def _support_sphere_fit(
    pmt_positions_mm: np.ndarray, pmt_normals: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Fit p_i = c + R n_i and return centre, radius and max residual."""
    p = np.asarray(pmt_positions_mm, dtype=np.float64)
    n = np.asarray(pmt_normals, dtype=np.float64)
    rows = np.zeros((3 * p.shape[0], 4), dtype=np.float64)
    rhs = p.reshape(-1)
    for j in range(p.shape[0]):
        rows[3 * j:3 * j + 3, :3] = np.eye(3)
        rows[3 * j:3 * j + 3, 3] = n[j]
    solution, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    centre = np.asarray(solution[:3], dtype=np.float64)
    radius = float(solution[3])
    reconstructed = centre[None, :] + radius * n
    residual = float(np.max(np.linalg.norm(reconstructed - p, axis=1)))
    return centre, radius, residual


def _stable_transverse_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(direction, dtype=np.float64)
    d = d / float(np.linalg.norm(d))
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(d[0])) < 0.85 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, helper)
    u /= float(np.linalg.norm(u))
    v = np.cross(d, u)
    return u, v


def _cap_capture_fraction_and_path(
    source_mm: np.ndarray,
    track_direction: np.ndarray,
    theta_rad: float,
    sphere_centre_mm: np.ndarray,
    cap_axis: np.ndarray,
    sphere_radius_mm: float,
    cap_cut_mm: float,
    n_phi: int,
) -> tuple[float, float]:
    """Deterministic azimuthal cone fraction reaching a spherical cap.

    The actual glass surface is used, so the external effective PMT angular
    response must *not* be applied again.  This avoids the double suppression
    that caused the first prototype to underpredict the traversed PMT.
    """
    d = np.asarray(track_direction, dtype=np.float64)
    d /= float(np.linalg.norm(d))
    axis = np.asarray(cap_axis, dtype=np.float64)
    axis /= float(np.linalg.norm(axis))
    u, v = _stable_transverse_basis(d)
    count = max(int(n_phi), 32)
    phi = (np.arange(count, dtype=np.float64) + 0.5) * (2.0 * math.pi / count)
    directions = (
        math.cos(theta_rad) * d[None, :]
        + math.sin(theta_rad) * (
            np.cos(phi)[:, None] * u[None, :]
            + np.sin(phi)[:, None] * v[None, :]
        )
    )
    q = np.asarray(source_mm, dtype=np.float64) - np.asarray(sphere_centre_mm, dtype=np.float64)
    qk = directions @ q
    discriminant = qk * qk - (float(np.dot(q, q)) - float(sphere_radius_mm) ** 2)
    good = discriminant >= 0.0
    distance = np.full(count, np.inf, dtype=np.float64)
    if np.any(good):
        square = np.sqrt(np.maximum(discriminant, 0.0))
        near = -qk - square
        far = -qk + square
        selected = np.where(near > 1.0e-8, near, np.where(far > 1.0e-8, far, np.inf))
        distance[good] = selected[good]
    good &= np.isfinite(distance)
    if not np.any(good):
        return 0.0, math.inf
    axial = np.full(count, -np.inf, dtype=np.float64)
    finite_indices = np.flatnonzero(good)
    if finite_indices.size:
        hit_rel = (
            q[None, :]
            + distance[finite_indices, None] * directions[finite_indices]
        )
        axial[finite_indices] = hit_rel @ axis
    good &= axial >= float(cap_cut_mm)
    n_good = int(np.count_nonzero(good))
    if n_good == 0:
        return 0.0, math.inf
    return float(n_good / count), float(np.mean(distance[good]))


def _relative_efficiency(
    emitter: object, pmt_indices: np.ndarray, costs: np.ndarray
) -> np.ndarray:
    codes = getattr(emitter, "_last_mpmt_type_codes", None)
    if codes is None:
        return np.ones(np.asarray(costs).shape, dtype=np.float64)
    try:
        from .Emitter import _interp_rel_mpmt_eff_from_codes
        return np.asarray(_interp_rel_mpmt_eff_from_codes(
            np.asarray(costs, dtype=np.float64),
            np.asarray(codes, dtype=np.int16)[np.asarray(pmt_indices, dtype=np.int64)],
            fill_empty=1.0,
        ), dtype=np.float64)
    except Exception:
        return np.ones(np.asarray(costs).shape, dtype=np.float64)


@dataclass
class PhysicalMPMTBoundaryModel:
    """Absolute raw-yield model for one placed mPMT boundary interface."""

    module: ModuleGeometry
    n_detector_pmts: int
    design: WCTEInSituPMTDesign = field(default_factory=WCTEInSituPMTDesign)
    gel: HardwareMaterial = WCTE_SILGEL
    acrylic: HardwareMaterial = WCTE_ACRYLIC
    glass: HardwareMaterial = WCTE_GLASS
    angular_response: object = default_pmt_angular_response

    def __post_init__(self) -> None:
        self.support_centre_mm, self.support_radius_mm, self.support_fit_max_residual_mm = (
            _support_sphere_fit(self.module.pmt_positions_mm, self.module.pmt_normals)
        )
        if self.support_fit_max_residual_mm > 2.0:
            raise ValueError(
                "PMT positions/normals do not define a common module support sphere; "
                f"max residual={self.support_fit_max_residual_mm:.3f} mm"
            )
        self.glass_centres_mm = np.ascontiguousarray(
            self.support_centre_mm[None, :]
            + float(self.design.glass_centre_radius_mm) * self.module.pmt_normals,
            dtype=np.float64,
        )
        self.reflector_mouth_centres_mm = np.ascontiguousarray(
            self.support_centre_mm[None, :]
            + float(self.design.reflector_upper_z_mm) * self.module.pmt_normals,
            dtype=np.float64,
        )
        order = max(2, int(self.design.source_quadrature_order))
        self._quad_x, self._quad_w = np.polynomial.legendre.leggauss(order)
        half_angle = math.radians(float(self.design.wc_pmt_opening_half_angle_deg))
        z_lo = float(self.design.wc_pmt_inner_radius_mm) * math.cos(half_angle)
        z_hi = float(self.design.wc_pmt_outer_radius_mm)
        self._wcpmt_axis_lo_mm = np.ascontiguousarray(
            self.support_centre_mm[None, :] + z_lo * self.module.pmt_normals,
            dtype=np.float64,
        )
        self._wcpmt_axis_hi_mm = np.ascontiguousarray(
            self.support_centre_mm[None, :] + z_hi * self.module.pmt_normals,
            dtype=np.float64,
        )
        self._wcpmt_capsule_radius_mm = float(
            self.design.wc_pmt_outer_radius_mm * math.sin(half_angle) + 1.0e-6
        )

    def _raw_conversion(self, emitter: object, aperture_radius_mm: float) -> float:
        n_water = float(getattr(emitter, "n"))
        ft_inf = 1.0 - 1.0 / (n_water * n_water)
        if ft_inf <= 0.0:
            return 0.0
        return float(
            2.0
            * float(getattr(emitter, "intensity"))
            * float(emitter.primary_ngeo_normalization())
            / (float(aperture_radius_mm) ** 2 * ft_inf)
        )

    def _wcpmt_intervals(
        self,
        boundary: np.ndarray,
        direction: np.ndarray,
        t_lo: float,
        t_hi: float,
        axis: np.ndarray,
    ) -> list[tuple[float, float]]:
        centre = self.support_centre_mm
        rel = boundary - centre
        roots: list[float] = []
        for radius in (
            float(self.design.wc_pmt_inner_radius_mm),
            float(self.design.wc_pmt_outer_radius_mm),
        ):
            roots.extend(_sphere_line_roots(boundary, direction, centre, radius))
        cos2 = math.cos(math.radians(float(self.design.wc_pmt_opening_half_angle_deg))) ** 2
        rn = float(np.dot(rel, axis))
        dn = float(np.dot(direction, axis))
        rd = float(np.dot(rel, direction))
        rr = float(np.dot(rel, rel))
        a = dn * dn - cos2
        b = 2.0 * (rn * dn - cos2 * rd)
        c = rn * rn - cos2 * rr
        roots.extend(_quadratic_roots(a, b, c))

        def inside(t: float) -> bool:
            q = rel + float(t) * direction
            radius = float(np.linalg.norm(q))
            if not (
                float(self.design.wc_pmt_inner_radius_mm) - 1.0e-7
                <= radius
                <= float(self.design.wc_pmt_outer_radius_mm) + 1.0e-7
            ):
                return False
            return bool(
                float(np.dot(q, axis))
                >= math.cos(math.radians(float(self.design.wc_pmt_opening_half_angle_deg)))
                * radius
                - 1.0e-7
            )

        return _predicate_intervals(roots, inside, t_lo, t_hi)

    def _inside_top_reflector(self, point: np.ndarray, axis: np.ndarray) -> bool:
        rel = np.asarray(point, dtype=np.float64) - self.support_centre_mm
        z = float(np.dot(rel, axis))
        lo = float(self.design.reflector_lower_z_mm)
        hi = float(self.design.reflector_upper_z_mm)
        if z < lo or z > hi:
            return False
        radial_vec = rel - z * axis
        radius = float(np.linalg.norm(radial_vec))
        fraction = (z - lo) / max(hi - lo, 1.0e-12)
        inner = (
            float(self.design.reflector_inner_radius_mm)
            + fraction * (
                float(self.design.reflector_mouth_radius_mm)
                - float(self.design.reflector_inner_radius_mm)
            )
        )
        return bool(inner <= radius <= inner + float(self.design.reflector_thickness_mm))

    def _source_material_in_wcpmt(
        self, point: np.ndarray, pmt_local_index: int
    ) -> HardwareMaterial | None:
        axis = self.module.pmt_normals[pmt_local_index]
        centre = self.glass_centres_mm[pmt_local_index]
        rel = np.asarray(point, dtype=np.float64) - centre
        r = float(np.linalg.norm(rel))
        z = float(np.dot(rel, axis))
        if z >= float(self.design.glass_cut_mm):
            if r < float(self.design.glass_inner_radius_mm) - 1.0e-7:
                return None  # PMT interior air
            if r <= float(self.design.glass_outer_radius_mm) + 1.0e-7:
                return self.glass
        if self._inside_top_reflector(point, axis):
            return None
        return self.gel

    @staticmethod
    def _ft(beta: float, phase_index: float) -> float:
        product = float(beta) * float(phase_index)
        if product <= 1.0:
            return 0.0
        return float(1.0 - 1.0 / (product * product))

    def _source_particle_time(
        self,
        signed_t_mm: float,
        *,
        interface: str,
        beta: float,
        boundary_particle_time_ns: float,
    ) -> float:
        # The short hardware segment is evaluated at the boundary beta.  For an
        # entry the signed coordinate is negative; for an exit it is positive.
        return float(
            boundary_particle_time_ns
            + float(signed_t_mm) / max(float(beta) * _C_MM_PER_NS, 1.0e-12)
        )

    def predict_raw(
        self,
        *,
        boundary_hit: BoundarySurfaceHit,
        direction: Sequence[float],
        interface: str,
        kinetic_energy_mev: float,
        emitter: object,
        boundary_particle_time_ns: float = 0.0,
        include_timing_nodes: bool = True,
    ) -> RawHardwarePrediction:
        label = str(interface).strip().lower()
        if label not in {"entry", "exit"}:
            raise ValueError("interface must be 'entry' or 'exit'")
        if boundary_hit.surface_kind != "mpmt_dome":
            raise ValueError("the physical mPMT model requires an mPMT boundary hit")
        if boundary_hit.slot is None or int(boundary_hit.slot) != int(self.module.slot):
            raise ValueError(
                f"boundary slot {boundary_hit.slot!r} does not match model slot {self.module.slot}"
            )

        boundary = np.asarray(boundary_hit.point_mm, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        d /= float(np.linalg.norm(d))
        beta = beta_from_kinetic_energy(float(kinetic_energy_mev))
        interval = self.module.hardware_interval_from_boundary(
            boundary, d, interface=label
        )
        raw = np.zeros(int(self.n_detector_pmts), dtype=np.float64)
        raw_modes = np.zeros((2, int(self.n_detector_pmts)), dtype=np.float64)
        node_rows: list[np.ndarray] = []
        node_times: list[np.ndarray] = []
        node_modes: list[int] = []
        if interval is None or beta <= 0.0:
            return RawHardwarePrediction(
                raw_charge=raw,
                raw_charge_modes=np.zeros((2, raw.size), dtype=np.float64),
                mode_names=("local_wcpmt", "outer_shell"),
                node_mu_raw=np.empty((0, raw.size), dtype=np.float64),
                node_t_ns=np.empty((0, raw.size), dtype=np.float64),
                node_modes=np.empty(0, dtype=np.int8),
                diagnostics={"slot": int(self.module.slot), "interface": label, "active": False},
            )
        t_lo, t_hi = map(float, interval)

        cap_conversion = self._raw_conversion(
            emitter, float(self.design.glass_rim_radius_mm)
        )
        mouth_conversion = self._raw_conversion(
            emitter, float(self.design.reflector_mouth_radius_mm)
        )
        traversed_pmts: list[dict[str, object]] = []
        material_sums = {self.gel.name: 0.0, self.glass.name: 0.0, self.acrylic.name: 0.0}

        # 1) Light created inside a PMT's WCPMT mother.  Reject PMTs whose
        # complete cone-shell capsule cannot intersect the hardware line.  This
        # is a conservative containment test; exact roots remain authoritative
        # for every survivor.
        segment_start = boundary + float(t_lo) * d
        segment_end = boundary + float(t_hi) * d
        candidate_local_indices = [
            local_index
            for local_index in range(self.module.pmt_indices.size)
            if _segment_segment_distance(
                segment_start,
                segment_end,
                self._wcpmt_axis_lo_mm[local_index],
                self._wcpmt_axis_hi_mm[local_index],
            ) <= self._wcpmt_capsule_radius_mm
        ]
        for local_index in candidate_local_indices:
            detector_index = int(self.module.pmt_indices[local_index])
            axis = self.module.pmt_normals[local_index]
            intervals = self._wcpmt_intervals(
                boundary, d, t_lo, t_hi, axis
            )
            if not intervals:
                continue
            pmt_raw = 0.0
            source_count = 0
            path_length = 0.0
            for left, right in intervals:
                half = 0.5 * (right - left)
                centre_t = 0.5 * (right + left)
                path_length += right - left
                for xq, wq in zip(self._quad_x, self._quad_w):
                    signed_t = centre_t + half * float(xq)
                    path_weight = half * float(wq)
                    source = boundary + signed_t * d
                    material = self._source_material_in_wcpmt(source, local_index)
                    if material is None:
                        continue
                    theta = cherenkov_angle_rad(beta, material.phase_index)
                    ft = self._ft(beta, material.phase_index)
                    if theta is None or ft <= 0.0:
                        continue
                    if include_timing_nodes:
                        # Validation/timing path: intersect the full spherical
                        # photocathode cap and retain its path length.
                        if material.name == self.glass.name:
                            receiver_radius = float(self.design.glass_inner_radius_mm)
                        else:
                            receiver_radius = float(self.design.glass_outer_radius_mm)
                        fraction, photon_path = _cap_capture_fraction_and_path(
                            source,
                            d,
                            float(theta),
                            self.glass_centres_mm[local_index],
                            axis,
                            receiver_radius,
                            float(self.design.glass_cut_mm),
                            int(self.design.cap_azimuth_samples),
                        )
                    else:
                        # Production charge-profile path.  Only the normalized
                        # owning-PMT shape and a geometry-dependent amplitude
                        # envelope are required.  The reflector-mouth disk is a
                        # conservative analytic proxy for the explicit cap: it
                        # preserves the finite-cone and transparent-chord limits
                        # but avoids hundreds of azimuthal ray intersections per
                        # FCN.  The exact cap remains available above whenever
                        # source-resolved timing is requested.
                        receiver = self.reflector_mouth_centres_mm[local_index]
                        vector = receiver - source
                        photon_path = float(np.linalg.norm(vector))
                        if photon_path <= 1.0e-8:
                            fraction = 0.0
                        else:
                            unit = vector / photon_path
                            alpha = math.acos(float(np.clip(np.dot(unit, d), -1.0, 1.0)))
                            rho = math.atan2(
                                float(self.design.reflector_mouth_radius_mm),
                                photon_path,
                            )
                            fraction = float(cone_angular_disk_overlap_fraction(
                                np.asarray([alpha], dtype=np.float64),
                                float(theta),
                                np.asarray([rho], dtype=np.float64),
                            )[0])
                    if fraction <= 0.0 or not math.isfinite(photon_path):
                        continue
                    contribution = cap_conversion * path_weight * ft * fraction
                    if contribution <= 0.0 or not math.isfinite(contribution):
                        continue
                    raw[int(detector_index)] += contribution
                    raw_modes[0, int(detector_index)] += contribution
                    pmt_raw += contribution
                    material_sums[material.name] += contribution
                    if include_timing_nodes:
                        row = np.zeros(raw.size, dtype=np.float64)
                        row[int(detector_index)] = contribution
                        time_row = np.full(raw.size, np.inf, dtype=np.float64)
                        time_row[int(detector_index)] = (
                            self._source_particle_time(
                                signed_t,
                                interface=label,
                                beta=beta,
                                boundary_particle_time_ns=float(boundary_particle_time_ns),
                            )
                            + float(material.group_index) * photon_path / _C_MM_PER_NS
                        )
                        node_rows.append(row)
                        node_times.append(time_row)
                        node_modes.append(0)
                    source_count += 1
            traversed_pmts.append({
                "local_index": int(local_index),
                "detector_index": int(detector_index),
                "sector_path_length_mm": float(path_length),
                "raw_charge": float(pmt_raw),
                "source_nodes": int(source_count),
            })

        # 2) Light created in the outer gel shell and acrylic dome.  These source
        # points are outside the matrix and may illuminate several local PMTs.
        # The reflector mouth is the finite receiver; the existing effective PMT
        # response and relative module efficiency are then applied once.
        broad_materials = (
            OpticalMaterial("silicone_gel", self.gel.phase_index, self.gel.group_index, 1.0),
            OpticalMaterial("acrylic_dome", self.acrylic.phase_index, self.acrylic.group_index, 1.0),
        )
        broad = self.module.broad_source_quadrature(
            boundary,
            d,
            interface=label,
            materials=broad_materials,
            interval=(float(t_lo), float(t_hi)),
        )
        broad_nodes = 0
        if broad is not None and mouth_conversion > 0.0:
            source_points, path_weights, source_materials, signed_t_values = broad
            for source, path_weight, old_material, signed_t in zip(
                source_points, path_weights, source_materials, signed_t_values
            ):
                material = self.gel if old_material.name == "silicone_gel" else self.acrylic
                theta = cherenkov_angle_rad(beta, material.phase_index)
                ft = self._ft(beta, material.phase_index)
                if theta is None or ft <= 0.0:
                    continue
                vectors = self.reflector_mouth_centres_mm - source[None, :]
                distances = np.linalg.norm(vectors, axis=1)
                good = distances > 1.0e-8
                unit = np.zeros_like(vectors)
                unit[good] = vectors[good] / distances[good, None]
                alpha = np.arccos(np.clip(unit @ d, -1.0, 1.0))
                rho = np.arctan2(
                    float(self.design.reflector_mouth_radius_mm),
                    np.maximum(distances, 1.0e-9),
                )
                cone_fraction = cone_angular_disk_overlap_fraction(
                    alpha, float(theta), rho
                )
                cost = np.clip(
                    -np.einsum("ij,ij->i", unit, self.module.pmt_normals),
                    0.0,
                    1.0,
                )
                response = np.asarray(self.angular_response(cost), dtype=np.float64)
                rel_eff = _relative_efficiency(
                    emitter, self.module.pmt_indices, cost
                )
                local = (
                    mouth_conversion
                    * float(path_weight)
                    * ft
                    * cone_fraction
                    * response
                    * rel_eff
                )
                local = np.where(
                    good & np.isfinite(local) & (local > 0.0), local, 0.0
                )
                if not np.any(local > 0.0):
                    continue
                raw[self.module.pmt_indices] += local
                raw_modes[1, self.module.pmt_indices] += local
                material_sums[material.name] += float(np.sum(local))
                if include_timing_nodes:
                    row = np.zeros(raw.size, dtype=np.float64)
                    row[self.module.pmt_indices] = local
                    time_row = np.full(raw.size, np.inf, dtype=np.float64)
                    source_particle_t = self._source_particle_time(
                        float(signed_t),
                        interface=label,
                        beta=beta,
                        boundary_particle_time_ns=float(boundary_particle_time_ns),
                    )
                    time_row[self.module.pmt_indices[good]] = (
                        source_particle_t
                        + float(material.group_index) * distances[good] / _C_MM_PER_NS
                    )
                    node_rows.append(row)
                    node_times.append(time_row)
                    node_modes.append(1)
                broad_nodes += 1

        node_mu = (
            np.ascontiguousarray(np.vstack(node_rows), dtype=np.float64)
            if node_rows else np.empty((0, raw.size), dtype=np.float64)
        )
        node_t = (
            np.ascontiguousarray(np.vstack(node_times), dtype=np.float64)
            if node_times else np.empty((0, raw.size), dtype=np.float64)
        )
        diagnostics = {
            "model": "absolute_raw_finite_cone_v1",
            "slot": int(self.module.slot),
            "interface": label,
            "active": bool(np.any(raw > 0.0)),
            "raw_total": float(np.sum(raw)),
            "module_raw_total": float(np.sum(raw[self.module.pmt_indices])),
            "hardware_signed_t_range_mm": [float(t_lo), float(t_hi)],
            "support_centre_mm": self.support_centre_mm.tolist(),
            "support_radius_mm": float(self.support_radius_mm),
            "support_fit_max_residual_mm": float(self.support_fit_max_residual_mm),
            "beta_at_water_boundary": float(beta),
            "kinetic_energy_at_water_boundary_mev": float(kinetic_energy_mev),
            "cap_conversion_raw_per_mm": float(cap_conversion),
            "mouth_conversion_raw_per_mm": float(mouth_conversion),
            "material_raw_sums": {k: float(v) for k, v in material_sums.items()},
            "mode_raw_sums": {
                "local_wcpmt": float(np.sum(raw_modes[0])),
                "outer_shell": float(np.sum(raw_modes[1])),
            },
            "wcpmt_capsule_candidates": int(len(candidate_local_indices)),
            "traversed_pmts": traversed_pmts,
            "broad_source_nodes": int(broad_nodes),
            "timing_nodes_requested": bool(include_timing_nodes),
            "local_capture_geometry": (
                "spherical_cap" if include_timing_nodes else "reflector_mouth_envelope"
            ),
            "timing_node_count": int(node_mu.shape[0]),
            "materials": {
                self.gel.name: vars(self.gel),
                self.acrylic.name: vars(self.acrylic),
                self.glass.name: vars(self.glass),
            },
            "design": vars(self.design),
        }
        return RawHardwarePrediction(
            raw_charge=np.ascontiguousarray(raw, dtype=np.float64),
            raw_charge_modes=np.ascontiguousarray(raw_modes, dtype=np.float64),
            mode_names=("local_wcpmt", "outer_shell"),
            node_mu_raw=node_mu,
            node_t_ns=node_t,
            node_modes=np.ascontiguousarray(np.asarray(node_modes, dtype=np.int8)),
            diagnostics=diagnostics,
        )

    @staticmethod
    def combine_raw(
        *,
        emitter: object,
        observed_pes: Sequence[float],
        hardware: RawHardwarePrediction,
    ) -> CombinedRawPrediction:
        base_raw = np.asarray(
            getattr(emitter, "_last_expected_pes_raw"), dtype=np.float64
        )
        base_timing_raw = np.asarray(
            getattr(emitter, "_last_expected_pes_timing_raw", base_raw),
            dtype=np.float64,
        )
        hw = np.asarray(hardware.raw_charge, dtype=np.float64)
        if base_raw.shape != hw.shape:
            raise ValueError("hardware and Emitter raw predictions have different shapes")
        raw_charge = base_raw + hw
        raw_timing = base_timing_raw + hw
        observed = np.asarray(observed_pes, dtype=np.float64)
        raw_mean = float(np.mean(raw_charge))
        obs_mean = float(np.mean(observed))
        if raw_mean <= 0.0:
            if obs_mean > 0.0:
                raise ValueError("nonzero observation with zero combined raw prediction")
            norm = 0.0
        else:
            norm = obs_mean / raw_mean
        if not math.isfinite(norm) or norm < 0.0:
            raise ValueError("invalid combined charge normalisation")
        floor = float(getattr(emitter, "charge_floor_pe", 1.0e-4))
        expected = np.maximum(raw_charge * norm, floor)
        timing = raw_timing * norm
        return CombinedRawPrediction(
            expected_pes=np.ascontiguousarray(expected, dtype=np.float64),
            timing_pes=np.ascontiguousarray(timing, dtype=np.float64),
            norm=float(norm),
            raw_charge=np.ascontiguousarray(raw_charge, dtype=np.float64),
            raw_timing=np.ascontiguousarray(raw_timing, dtype=np.float64),
        )

    @staticmethod
    def augment_timing_prediction(
        timing_prediction: object,
        *,
        hardware: RawHardwarePrediction,
        combined_norm: float,
    ) -> object:
        """Append hardware nodes and update the common raw-to-PE scale."""
        from .Emitter import TimingPrediction

        if not isinstance(timing_prediction, TimingPrediction):
            return timing_prediction
        active = getattr(timing_prediction, "first_arrival_active_indices", None)
        if active is None:
            return timing_prediction
        active = np.asarray(active, dtype=np.int64)
        hw_mu = np.asarray(hardware.node_mu_raw, dtype=np.float64)
        hw_t = np.asarray(hardware.node_t_ns, dtype=np.float64)
        if hw_mu.ndim != 2 or hw_t.shape != hw_mu.shape:
            return timing_prediction
        if hw_mu.shape[0] == 0:
            # Even when no local nodes are active, the common node scale must
            # follow the joint charge normalisation.
            hw_mu_active = np.empty((0, active.size), dtype=np.float32)
            hw_t_active = np.empty((0, active.size), dtype=np.float32)
        else:
            hw_mu_active = np.ascontiguousarray(hw_mu[:, active], dtype=np.float32)
            hw_t_active = np.ascontiguousarray(hw_t[:, active], dtype=np.float32)
            keep = np.any(hw_mu_active > 0.0, axis=1)
            hw_mu_active = hw_mu_active[keep]
            hw_t_active = hw_t_active[keep]

        deferred_mu = getattr(timing_prediction, "first_arrival_deferred_base_mu", None)
        deferred_t = getattr(timing_prediction, "first_arrival_deferred_base_t", None)
        if deferred_mu is not None and deferred_t is not None:
            new_mu = np.vstack((
                np.asarray(deferred_mu, dtype=np.float32), hw_mu_active
            ))
            new_t = np.vstack((
                np.asarray(deferred_t, dtype=np.float32), hw_t_active
            ))
            return TimingPrediction(
                np.asarray(timing_prediction, dtype=np.float64),
                active_indices=np.ascontiguousarray(active, dtype=np.int32),
                deferred_base_mu=np.ascontiguousarray(new_mu, dtype=np.float32),
                deferred_base_t=np.ascontiguousarray(new_t, dtype=np.float32),
                reflection_u=getattr(timing_prediction, "first_arrival_reflection_u", None),
                reflection_tbase=getattr(timing_prediction, "first_arrival_reflection_tbase", None),
                reflection_transfer_active=getattr(timing_prediction, "first_arrival_reflection_transfer_active", None),
                reflection_time_offset_active=getattr(timing_prediction, "first_arrival_reflection_time_offset_active", None),
                reflection_patch_min_time_offset=getattr(timing_prediction, "first_arrival_reflection_patch_min_time_offset", None),
                reflection_patch_max_time_offset=getattr(timing_prediction, "first_arrival_reflection_patch_max_time_offset", None),
                reflection_n_bins=getattr(timing_prediction, "first_arrival_reflection_n_bins", None),
                node_pe_scale=float(combined_norm),
            )

        node_mu = getattr(timing_prediction, "first_arrival_node_mu", None)
        node_t = getattr(timing_prediction, "first_arrival_node_t", None)
        if node_mu is None or node_t is None:
            return timing_prediction
        base_mu = np.asarray(node_mu, dtype=np.float32)
        base_t = np.asarray(node_t, dtype=np.float32)
        new_mu = np.vstack((base_mu, hw_mu_active))
        new_t = np.vstack((base_t, hw_t_active))
        # Sort each PMT's nodes chronologically.  The generic node-mu likelihood
        # then computes the conditional first-arrival law from raw amplitudes.
        order = np.argsort(new_t, axis=0)
        new_mu = np.take_along_axis(new_mu, order, axis=0)
        new_t = np.take_along_axis(new_t, order, axis=0)
        return TimingPrediction(
            np.asarray(timing_prediction, dtype=np.float64),
            node_mu=np.ascontiguousarray(new_mu, dtype=np.float32),
            node_t=np.ascontiguousarray(new_t, dtype=np.float32),
            active_indices=np.ascontiguousarray(active, dtype=np.int32),
            node_pe_scale=float(combined_norm),
        )
