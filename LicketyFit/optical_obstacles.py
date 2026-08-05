"""Analytic detector-obstacle visibility helpers.

The main LicketyFit geometry package describes the active water and PMT modules,
but the WCTE WCSim configuration also contains a central deployment-system (CDS)
CAD object.  This module supplies a compact analytic representation of the
optically dominant CDS shaft: a finite opaque annular cylinder around a water
bore.  The representation is detector-configuration input; no event-level truth
enters the likelihood.

The public Python wrappers are useful for tests and diagnostics.  The Numba
functions are used directly by the hot primary and delta-light kernels.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def segment_intersects_annular_y_cylinder_numba(
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    axis_x: float,
    axis_z: float,
    inner_radius: float,
    outer_radius: float,
    y_min: float,
    y_max: float,
) -> bool:
    """Return whether a line segment intersects a finite y-axis annular shell.

    The solid is

        y_min <= y <= y_max,
        inner_radius <= sqrt((x-axis_x)^2 + (z-axis_z)^2) <= outer_radius.

    The method first clips the segment to the y slab.  Over that interval the
    squared radius is a convex quadratic, whose image is a continuous interval
    [rho2_min, rho2_max].  The segment intersects the annulus exactly when that
    interval overlaps [inner_radius^2, outer_radius^2].
    """
    if not (
        math.isfinite(x0) and math.isfinite(y0) and math.isfinite(z0)
        and math.isfinite(x1) and math.isfinite(y1) and math.isfinite(z1)
        and math.isfinite(axis_x) and math.isfinite(axis_z)
        and math.isfinite(inner_radius) and math.isfinite(outer_radius)
        and math.isfinite(y_min) and math.isfinite(y_max)
    ):
        return False
    rin = inner_radius
    rout = outer_radius
    if rin < 0.0:
        rin = 0.0
    if rout <= rin or y_max <= y_min:
        return False

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    # Clip t in [0,1] to the finite y slab.
    t_lo = 0.0
    t_hi = 1.0
    eps = 1.0e-12
    if abs(dy) <= eps:
        if y0 < y_min or y0 > y_max:
            return False
    else:
        ta = (y_min - y0) / dy
        tb = (y_max - y0) / dy
        if ta > tb:
            tmp = ta
            ta = tb
            tb = tmp
        if ta > t_lo:
            t_lo = ta
        if tb < t_hi:
            t_hi = tb
        if t_hi < t_lo:
            return False
    if t_hi < 0.0 or t_lo > 1.0:
        return False
    if t_lo < 0.0:
        t_lo = 0.0
    if t_hi > 1.0:
        t_hi = 1.0
    if t_hi < t_lo:
        return False

    rx0 = x0 - axis_x
    rz0 = z0 - axis_z
    aa = dx * dx + dz * dz
    bb = 2.0 * (rx0 * dx + rz0 * dz)
    cc = rx0 * rx0 + rz0 * rz0

    def q(tt):
        return (aa * tt + bb) * tt + cc

    q_lo = q(t_lo)
    q_hi = q(t_hi)
    q_min = q_lo if q_lo < q_hi else q_hi
    q_max = q_hi if q_hi > q_lo else q_lo
    if aa > eps:
        tv = -bb / (2.0 * aa)
        if tv > t_lo and tv < t_hi:
            qv = q(tv)
            if qv < q_min:
                q_min = qv
    # A convex quadratic reaches its maximum at an interval endpoint.
    rin2 = rin * rin
    rout2 = rout * rout
    tol = 1.0e-8
    return (q_max >= rin2 - tol) and (q_min <= rout2 + tol)


@njit(cache=True, fastmath=True)
def annular_cylinder_aperture_visibility_numba(
    source_x: float,
    source_y: float,
    source_z: float,
    pmt_x: float,
    pmt_y: float,
    pmt_z: float,
    normal_x: float,
    normal_y: float,
    normal_z: float,
    axis_x: float,
    axis_z: float,
    inner_radius: float,
    outer_radius: float,
    y_min: float,
    y_max: float,
    aperture_radius: float,
) -> float:
    """Approximate the unobscured fraction of a circular PMT aperture.

    A seven-point rotationally symmetric disk cubature is used: the centre and
    six equally weighted points on a ring.  The ring radius sqrt(7/12)*a makes
    the equal-weight rule reproduce the exact second radial moment of a uniform
    disk.  The result is deterministic, bounded in [0,1], and gives partial
    visibility near the CDS silhouette rather than a centre-ray step.
    """
    if outer_radius <= inner_radius or y_max <= y_min:
        return 1.0
    if aperture_radius <= 1.0e-12:
        blocked = segment_intersects_annular_y_cylinder_numba(
            source_x, source_y, source_z,
            pmt_x, pmt_y, pmt_z,
            axis_x, axis_z, inner_radius, outer_radius, y_min, y_max,
        )
        return 0.0 if blocked else 1.0

    # Build a stable orthonormal basis in the PMT aperture plane.
    nn = math.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
    if nn <= 1.0e-15:
        nx = 0.0
        ny = 1.0
        nz = 0.0
    else:
        nx = normal_x / nn
        ny = normal_y / nn
        nz = normal_z / nn

    if abs(ny) < 0.9:
        # e1 = normalize(n x yhat) = (-nz, 0, nx)
        e1x = -nz
        e1y = 0.0
        e1z = nx
    else:
        # e1 = normalize(n x xhat) = (0, nz, -ny)
        e1x = 0.0
        e1y = nz
        e1z = -ny
    e1n = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    if e1n <= 1.0e-15:
        e1x = 1.0
        e1y = 0.0
        e1z = 0.0
    else:
        e1x /= e1n
        e1y /= e1n
        e1z /= e1n
    # e2 = n x e1
    e2x = ny * e1z - nz * e1y
    e2y = nz * e1x - nx * e1z
    e2z = nx * e1y - ny * e1x

    visible = 0
    if not segment_intersects_annular_y_cylinder_numba(
        source_x, source_y, source_z,
        pmt_x, pmt_y, pmt_z,
        axis_x, axis_z, inner_radius, outer_radius, y_min, y_max,
    ):
        visible += 1

    rr = aperture_radius * math.sqrt(7.0 / 12.0)
    # Fixed exact trigonometric values at multiples of 60 degrees.
    cvals = (1.0, 0.5, -0.5, -1.0, -0.5, 0.5)
    s60 = 0.86602540378443864676
    svals = (0.0, s60, s60, 0.0, -s60, -s60)
    for k in range(6):
        ox = rr * (cvals[k] * e1x + svals[k] * e2x)
        oy = rr * (cvals[k] * e1y + svals[k] * e2y)
        oz = rr * (cvals[k] * e1z + svals[k] * e2z)
        if not segment_intersects_annular_y_cylinder_numba(
            source_x, source_y, source_z,
            pmt_x + ox, pmt_y + oy, pmt_z + oz,
            axis_x, axis_z, inner_radius, outer_radius, y_min, y_max,
        ):
            visible += 1
    return visible / 7.0


def segment_intersects_annular_y_cylinder(p0, p1, *, axis_x=0.0, axis_z=0.0,
                                           inner_radius, outer_radius, y_min, y_max):
    """Python wrapper around :func:`segment_intersects_annular_y_cylinder_numba`."""
    a = np.asarray(p0, dtype=float)
    b = np.asarray(p1, dtype=float)
    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("p0 and p1 must be three-vectors")
    return bool(segment_intersects_annular_y_cylinder_numba(
        float(a[0]), float(a[1]), float(a[2]),
        float(b[0]), float(b[1]), float(b[2]),
        float(axis_x), float(axis_z), float(inner_radius), float(outer_radius),
        float(y_min), float(y_max),
    ))


def annular_cylinder_aperture_visibility(source, pmt, normal, *, axis_x=0.0,
                                          axis_z=0.0, inner_radius, outer_radius,
                                          y_min, y_max, aperture_radius=45.0):
    """Python wrapper returning the seven-point aperture visibility fraction."""
    s = np.asarray(source, dtype=float)
    p = np.asarray(pmt, dtype=float)
    n = np.asarray(normal, dtype=float)
    if s.shape != (3,) or p.shape != (3,) or n.shape != (3,):
        raise ValueError("source, pmt, and normal must be three-vectors")
    return float(annular_cylinder_aperture_visibility_numba(
        float(s[0]), float(s[1]), float(s[2]),
        float(p[0]), float(p[1]), float(p[2]),
        float(n[0]), float(n[1]), float(n[2]),
        float(axis_x), float(axis_z), float(inner_radius), float(outer_radius),
        float(y_min), float(y_max), float(aperture_radius),
    ))


def build_inner_cylinder_surface(*, axis_x=0.0, axis_z=0.0, radius,
                                 y_min, y_max, n_phi=72, n_y=32):
    """Return midpoint quadrature for the water-facing wall of a cylinder.

    The returned normal points radially from the bore into the surrounding
    solid.  This is the convenient sign for the incident projected-area factor;
    the specular reflection law itself is invariant under normal reversal.

    Parameters are detector-configuration geometry.  No event data enter the
    quadrature.
    """
    radius = float(radius)
    y_min = float(y_min)
    y_max = float(y_max)
    n_phi = int(n_phi)
    n_y = int(n_y)
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError("radius must be positive and finite")
    if not (math.isfinite(y_min) and math.isfinite(y_max) and y_max > y_min):
        raise ValueError("y_max must exceed y_min")
    if n_phi < 8 or n_y < 1:
        raise ValueError("n_phi must be >=8 and n_y must be >=1")

    dphi = 2.0 * math.pi / n_phi
    dy = (y_max - y_min) / n_y
    phi = (np.arange(n_phi, dtype=np.float64) + 0.5) * dphi
    yy = y_min + (np.arange(n_y, dtype=np.float64) + 0.5) * dy
    pp, yg = np.meshgrid(phi, yy, indexing="xy")
    c = np.cos(pp).ravel()
    s = np.sin(pp).ravel()
    y = yg.ravel()
    xyz = np.column_stack((float(axis_x) + radius * c,
                           y,
                           float(axis_z) + radius * s))
    normal = np.column_stack((c, np.zeros_like(c), s))
    area = np.full(xyz.shape[0], radius * dphi * dy, dtype=np.float64)
    return (np.ascontiguousarray(xyz, dtype=np.float64),
            np.ascontiguousarray(normal, dtype=np.float64),
            np.ascontiguousarray(area, dtype=np.float64))


@njit(cache=True, fastmath=True)
def _old_primary_angular_response_numba(cost: float) -> float:
    """Current effective WCTE PMT angular factor for one incidence cosine."""
    if cost <= 0.0 or not math.isfinite(cost):
        return 0.0
    if cost > 1.0:
        cost = 1.0
    xn = cost ** 3.0777000000000001
    return ((0.1209 + (1.6396999999999999 - 0.1209)
             * (xn / (xn + 0.79428866592713121)))
            / 1.002379253316015)


@njit(cache=True, fastmath=True)
def trace_specular_inner_cylinder_to_pmt_disks_numba(
    surface_xyz,
    surface_normal,
    incident_mu,
    incident_tbase,
    source_s,
    track_start,
    track_direction,
    track_length,
    pmt_xyz,
    pmt_normal,
    axis_x,
    axis_z,
    inner_radius,
    outer_radius,
    y_min,
    y_max,
    pmt_aperture_radius,
    reflectivity,
    attenuation_length,
    group_index_over_c,
):
    """Trace one ideal polished-metal reflection from each inner-wall patch.

    ``incident_mu`` is the analytic primary-light mass incident on each surface
    patch.  The centre ray of that small deterministic patch is reflected with
    the exact specular law and assigned to the nearest PMT disk it intersects.
    A second intersection with the CDS annulus rejects rays that would require
    another metal bounce.  The function returns one sparse receiver record per
    surface patch; downstream code may aggregate or time-bin it without tracing
    the geometry again.

    The amplitude multiplier is detector-configuration physics only:
    flat WCSim aluminum reflectivity, water attenuation on the outgoing leg,
    and the existing effective PMT incidence response.  No event-truth scale is
    fitted or read.
    """
    npatch = surface_xyz.shape[0]
    npmts = pmt_xyz.shape[0]
    hit_index = np.full(npatch, -1, dtype=np.int32)
    hit_mu = np.zeros(npatch, dtype=np.float64)
    hit_time = np.full(npatch, np.inf, dtype=np.float64)
    hit_distance = np.zeros(npatch, dtype=np.float64)

    tx = track_direction[0]
    ty = track_direction[1]
    tz = track_direction[2]
    a2 = pmt_aperture_radius * pmt_aperture_radius
    eps_t = 1.0e-7
    # Move the second-leg obstacle test a small physical distance into water so
    # the intentional first contact is not counted as a second intersection.
    launch_eps = 0.05  # mm

    for ip in range(npatch):
        base_mu = incident_mu[ip]
        if not (base_mu > 0.0 and math.isfinite(base_mu)):
            continue

        ss = source_s[ip]
        if ss < 0.0:
            ss = 0.0
        elif ss > track_length:
            ss = track_length
        sx = track_start[0] + ss * tx
        sy = track_start[1] + ss * ty
        sz = track_start[2] + ss * tz

        px = surface_xyz[ip, 0]
        py = surface_xyz[ip, 1]
        pz = surface_xyz[ip, 2]
        kix = px - sx
        kiy = py - sy
        kiz = pz - sz
        kn = math.sqrt(kix*kix + kiy*kiy + kiz*kiz)
        if kn <= 1.0e-12:
            continue
        kix /= kn
        kiy /= kn
        kiz /= kn

        nx = surface_normal[ip, 0]
        ny = surface_normal[ip, 1]
        nz = surface_normal[ip, 2]
        nn = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nn <= 1.0e-12:
            continue
        nx /= nn
        ny /= nn
        nz /= nn
        kin_dot_n = kix*nx + kiy*ny + kiz*nz
        # The patch source calculation already enforces positive projected
        # incidence, but keep the ray tracer fail-closed for standalone use.
        if kin_dot_n <= 0.0:
            continue
        kox = kix - 2.0 * kin_dot_n * nx
        koy = kiy - 2.0 * kin_dot_n * ny
        koz = kiz - 2.0 * kin_dot_n * nz
        kon = math.sqrt(kox*kox + koy*koy + koz*koz)
        if kon <= 1.0e-12:
            continue
        kox /= kon
        koy /= kon
        koz /= kon
        # An inner-wall reflection must initially return into the water bore.
        if kox*nx + koy*ny + koz*nz >= -1.0e-10:
            continue

        best_i = -1
        best_t = 1.0e300
        best_cost = 0.0
        best_qx = 0.0
        best_qy = 0.0
        best_qz = 0.0
        for i in range(npmts):
            pnx = pmt_normal[i, 0]
            pny = pmt_normal[i, 1]
            pnz = pmt_normal[i, 2]
            pnn = math.sqrt(pnx*pnx + pny*pny + pnz*pnz)
            if pnn <= 1.0e-12:
                continue
            pnx /= pnn
            pny /= pnn
            pnz /= pnn
            denom = kox*pnx + koy*pny + koz*pnz
            # PMT normal points from photocathode into water; front incidence
            # therefore has k_out . n_pmt < 0.
            if denom >= -1.0e-10:
                continue
            vx = pmt_xyz[i, 0] - px
            vy = pmt_xyz[i, 1] - py
            vz = pmt_xyz[i, 2] - pz
            tt = (vx*pnx + vy*pny + vz*pnz) / denom
            if tt <= eps_t or tt >= best_t:
                continue
            qx = px + tt*kox
            qy = py + tt*koy
            qz = pz + tt*koz
            rx = qx - pmt_xyz[i, 0]
            ry = qy - pmt_xyz[i, 1]
            rz = qz - pmt_xyz[i, 2]
            # Remove the tiny normal numerical component before the disk test.
            rd = rx*pnx + ry*pny + rz*pnz
            rx -= rd*pnx
            ry -= rd*pny
            rz -= rd*pnz
            if rx*rx + ry*ry + rz*rz > a2:
                continue
            best_i = i
            best_t = tt
            best_cost = -denom
            best_qx = qx
            best_qy = qy
            best_qz = qz

        if best_i < 0:
            continue

        # Reject a ray that hits the cylindrical steel again before the PMT.
        lx = px + launch_eps*kox
        ly = py + launch_eps*koy
        lz = pz + launch_eps*koz
        if segment_intersects_annular_y_cylinder_numba(
            lx, ly, lz,
            best_qx, best_qy, best_qz,
            axis_x, axis_z, inner_radius, outer_radius, y_min, y_max,
        ):
            continue

        ang = _old_primary_angular_response_numba(best_cost)
        if ang <= 0.0:
            continue
        survive = 1.0
        if attenuation_length > 0.0 and math.isfinite(attenuation_length):
            survive = math.exp(-best_t / attenuation_length)
        # ``base_mu`` is the incident patch integral expressed in the same
        # PE-equivalent normalization as a face-on reference receiver.  The
        # direct primary N_geo normalization already contains one effective
        # circular receiver area, so a surface integral must be divided by that
        # reference area before it can be mapped deterministically to one PMT.
        # The blacksheet transfer performs the identical normalization through
        # omega_disk / (pi a^2).  Omitting it makes the specular component larger
        # by O(pi a^2) (~6.4e3 for a=45 mm).
        reference_area = math.pi * a2
        if reference_area <= 0.0:
            continue
        amp = base_mu * reflectivity * ang * survive / reference_area
        if not (amp > 0.0 and math.isfinite(amp)):
            continue
        hit_index[ip] = best_i
        hit_mu[ip] = amp
        hit_distance[ip] = best_t
        hit_time[ip] = incident_tbase[ip] + best_t * group_index_over_c

    return hit_index, hit_mu, hit_time, hit_distance


@njit(cache=True, fastmath=True)
def accumulate_sparse_patch_receivers_numba(
    hit_index,
    hit_mu,
    hit_time,
    n_pmts,
    n_time_bins,
):
    """Aggregate sparse patch receiver records into PMT charge/time nodes."""
    mu = np.zeros(n_pmts, dtype=np.float64)
    tnum = np.zeros(n_pmts, dtype=np.float64)
    for k in range(hit_index.size):
        i = hit_index[k]
        w = hit_mu[k]
        tt = hit_time[k]
        if i >= 0 and i < n_pmts and w > 0.0 and math.isfinite(tt):
            mu[i] += w
            tnum[i] += w * tt

    nbin = n_time_bins
    if nbin <= 0:
        return (mu, tnum,
                np.zeros((0, n_pmts), dtype=np.float32),
                np.zeros((0, n_pmts), dtype=np.float32))

    tmin = 1.0e300
    tmax = -1.0e300
    for k in range(hit_index.size):
        if hit_index[k] >= 0 and hit_mu[k] > 0.0 and math.isfinite(hit_time[k]):
            tt = hit_time[k]
            if tt < tmin:
                tmin = tt
            if tt > tmax:
                tmax = tt
    node_mu = np.zeros((nbin, n_pmts), dtype=np.float32)
    node_tnum = np.zeros((nbin, n_pmts), dtype=np.float64)
    if not (tmax >= tmin and math.isfinite(tmin) and math.isfinite(tmax)):
        node_t = np.full((nbin, n_pmts), np.inf, dtype=np.float32)
        return mu, tnum, node_mu, node_t
    span = tmax - tmin
    if span < 1.0e-9:
        span = 1.0e-9
    for k in range(hit_index.size):
        i = hit_index[k]
        w = hit_mu[k]
        tt = hit_time[k]
        if i < 0 or i >= n_pmts or not (w > 0.0 and math.isfinite(tt)):
            continue
        ib = int((tt - tmin) / span * nbin)
        if ib < 0:
            ib = 0
        elif ib >= nbin:
            ib = nbin - 1
        node_mu[ib, i] += w
        node_tnum[ib, i] += w * tt
    node_t = np.full((nbin, n_pmts), np.inf, dtype=np.float32)
    for b in range(nbin):
        for i in range(n_pmts):
            w = node_mu[b, i]
            if w > 0.0:
                node_t[b, i] = node_tnum[b, i] / w
    return mu, tnum, node_mu, node_t
