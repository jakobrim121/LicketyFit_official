"""Nonlinear coherent Fermi--Eyges trajectory utilities.

Research implementation for event-specific MCS reconstruction.  The path is
parameterized by standardized KL coefficients but preserves physical arc length:

    X'(s) = sqrt(1-|q(s)|^2) d0 + q1(s)e1 + q2(s)e2,

where qa(s)=sum_k u[a,k] phi'_k(s).  This is the second-order completion missing
from the historical first-order optical Jacobian.  It reduces exactly to the
straight reference path at u=0 and uses no simulation-derived parameters.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit, prange

from .mcs_process import build_raw_fe_kl_basis, stable_transverse_basis
from . import Emitter as emod


def _aperture_quadrature(n_radial: int = 3, n_azimuth: int = 8):
    """Area-normalized deterministic quadrature on a unit circular disk."""
    x, w = np.polynomial.legendre.leggauss(max(1, int(n_radial)))
    t = 0.5 * (x + 1.0)              # t = r^2/a^2, uniform under disk area
    wt = 0.5 * w
    phi = 2.0 * np.pi * (np.arange(max(1, int(n_azimuth))) + 0.5) / max(1, int(n_azimuth))
    pts = []
    weights = []
    for tr, wr in zip(t, wt):
        rr = math.sqrt(max(float(tr), 0.0))
        for ph in phi:
            pts.append((rr * math.cos(ph), rr * math.sin(ph)))
            weights.append(float(wr) / len(phi))
    return np.ascontiguousarray(pts, dtype=np.float64), np.ascontiguousarray(weights, dtype=np.float64)


_APERTURE_XY, _APERTURE_W = _aperture_quadrature(3, 8)


def build_arclength_fe_path(emitter, coefficients, *, n_grid: int = 81):
    """Build an arc-length-preserving nonlinear FE path on the KL grid.

    Parameters
    ----------
    emitter
        Configured Emitter at the straight-track hypothesis.
    coefficients
        Length 2*M array, first plane followed by second plane.
    n_grid
        KL/path grid.  Defaults to 81 for nonlinear root stability.
    """
    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coeff.size % 2:
        raise ValueError("coefficients must contain the same number of modes in two transverse planes")
    nm = coeff.size // 2
    sg, phi, slope, curvature, frac = build_raw_fe_kl_basis(emitter, nm, n_grid)
    direction, e1, e2 = stable_transverse_basis(emitter.direction)
    u1, u2 = coeff[:nm], coeff[nm:]
    x1 = phi[:, :nm] @ u1
    x2 = phi[:, :nm] @ u2
    q1 = slope[:, :nm] @ u1
    q2 = slope[:, :nm] @ u2
    qp1 = curvature[:, :nm] @ u1
    qp2 = curvature[:, :nm] @ u2

    qsq = q1 * q1 + q2 * q2
    # Gaussian-core FE paths are overwhelmingly inside this domain at the
    # energies of interest.  A smooth cap prevents nonphysical superluminal
    # transverse slopes in extreme posterior proposals without tuning a scale.
    max_qsq = 1.0 - 1.0e-10
    over = qsq > max_qsq
    if np.any(over):
        fac = np.sqrt(max_qsq / qsq[over])
        q1[over] *= fac
        q2[over] *= fac
        qp1[over] *= fac
        qp2[over] *= fac
        qsq = q1 * q1 + q2 * q2
    parallel_slope = np.sqrt(np.maximum(1.0 - qsq, 1.0e-12))

    # Integrate the longitudinal component so |dX/ds|=1 exactly on the grid.
    zpar = np.zeros_like(sg)
    if sg.size > 1:
        zpar[1:] = np.cumsum(0.5 * (parallel_slope[1:] + parallel_slope[:-1]) * np.diff(sg))
    start = np.asarray(emitter.start_coord, dtype=np.float64)
    position = (
        start[None, :]
        + zpar[:, None] * direction[None, :]
        + x1[:, None] * e1[None, :]
        + x2[:, None] * e2[None, :]
    )
    tangent = (
        parallel_slope[:, None] * direction[None, :]
        + q1[:, None] * e1[None, :]
        + q2[:, None] * e2[None, :]
    )
    # Analytic derivative of the unit tangent.
    parallel_second = -(q1 * qp1 + q2 * qp2) / np.maximum(parallel_slope, 1.0e-12)
    tangent_derivative = (
        parallel_second[:, None] * direction[None, :]
        + qp1[:, None] * e1[None, :]
        + qp2[:, None] * e2[None, :]
    )

    R = float(getattr(emitter, "range_to_threshold_mm", emitter.length))
    energy = np.asarray(emitter.muon_energy_at_s_array(sg, R), dtype=np.float64)
    mass = float(emitter.particle_mass)
    nwater = float(emitter.n)
    gamma = 1.0 + np.maximum(energy, 0.0) / mass
    beta2 = np.maximum(1.0 - 1.0 / np.maximum(gamma * gamma, 1.0), 0.0)
    beta = np.sqrt(beta2)
    cos_ch = np.ones_like(beta)
    above = nwater * beta > 1.0
    cos_ch[above] = 1.0 / (nwater * beta[above])
    ft = np.asarray(emod._cherenkov_weight_from_energy(energy, mass, nwater), dtype=np.float64)
    ft_sat = max(1.0 - 1.0 / (nwater * nwater), 1.0e-30)
    ft = ft / ft_sat
    dedx_E, dedx_S = emod._get_particle_stopping_power_table(emitter.particle_name)
    stopping = np.interp(energy, dedx_E, dedx_S)
    dcos_ds = np.zeros_like(energy)
    valid = above & (beta > 0.0)
    dcos_ds[valid] = stopping[valid] / (
        nwater * mass * beta[valid] ** 3 * gamma[valid] ** 3
    )
    # Existing exact range-table antiderivative; path coordinate s remains the
    # physical traveled distance by construction.
    particle_time = np.asarray(emod._wcte_integrated_primary_tof_fast(emitter, sg), dtype=np.float64)
    return {
        "s": np.ascontiguousarray(sg),
        "position": np.ascontiguousarray(position),
        "tangent": np.ascontiguousarray(tangent),
        "tangent_derivative": np.ascontiguousarray(tangent_derivative),
        "energy": np.ascontiguousarray(energy),
        "cos_cherenkov": np.ascontiguousarray(cos_ch),
        "dcos_ds": np.ascontiguousarray(dcos_ds),
        "frank_tamm": np.ascontiguousarray(ft),
        "particle_time_ns": np.ascontiguousarray(particle_time),
        "parallel_coordinate": np.ascontiguousarray(zpar),
        "basis_explained_fraction": np.ascontiguousarray(frac),
    }


def build_arclength_fe_path_with_derivatives(emitter, coefficients, *, n_grid: int = 81):
    """Build the coherent path and its exact first derivatives in KL space.

    The standardized Fermi--Eyges coefficients enter the transverse displacement
    and slope linearly.  The only nonlinear kinematic completion is the
    arc-length constraint ``sqrt(1-q1**2-q2**2)``.  Differentiating that
    constraint analytically gives the position and unit-tangent derivatives used
    by the compiled FALI charge Jacobian below.

    The extreme slope cap in :func:`build_arclength_fe_path` is a last-resort
    physical-domain guard.  Its non-smooth projection is deliberately not
    differentiated: callers must fall back to finite differences if it is ever
    active.  It is inactive for all validated posterior paths in this release.
    """
    coeff = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coeff.size % 2:
        raise ValueError("coefficients must contain equal transverse-plane blocks")
    nm = coeff.size // 2
    path = build_arclength_fe_path(emitter, coeff, n_grid=n_grid)
    sg, phi, slope, _curvature, _frac = build_raw_fe_kl_basis(emitter, nm, n_grid)
    direction, e1, e2 = stable_transverse_basis(emitter.direction)
    u1 = coeff[:nm]
    u2 = coeff[nm:]
    q1 = slope[:, :nm] @ u1
    q2 = slope[:, :nm] @ u2
    qsq = q1 * q1 + q2 * q2
    if np.any(qsq >= 1.0 - 1.0e-10):
        raise FloatingPointError(
            "analytic coherent-path derivative is undefined at the emergency slope cap"
        )
    parallel = np.sqrt(np.maximum(1.0 - qsq, 1.0e-15))
    nlatent = 2 * nm
    dq1 = np.zeros((sg.size, nlatent), dtype=np.float64)
    dq2 = np.zeros_like(dq1)
    dq1[:, :nm] = slope[:, :nm]
    dq2[:, nm:] = slope[:, :nm]
    dparallel = -(
        q1[:, None] * dq1 + q2[:, None] * dq2
    ) / parallel[:, None]

    dzpar = np.zeros((sg.size, nlatent), dtype=np.float64)
    if sg.size > 1:
        increments = (
            0.5 * (dparallel[1:, :] + dparallel[:-1, :])
            * np.diff(sg)[:, None]
        )
        dzpar[1:, :] = np.cumsum(increments, axis=0)

    dpos = (
        dzpar[:, :, None] * direction[None, None, :]
        + np.concatenate(
            (
                phi[:, :nm, None] * e1[None, None, :],
                phi[:, :nm, None] * e2[None, None, :],
            ),
            axis=1,
        )
    )
    dtan = (
        dparallel[:, :, None] * direction[None, None, :]
        + dq1[:, :, None] * e1[None, None, :]
        + dq2[:, :, None] * e2[None, None, :]
    )
    # Layout (s, xyz, latent) is contiguous for the innermost latent loop in
    # the Numba receiver kernel.
    path["position_derivative"] = np.ascontiguousarray(
        np.transpose(dpos, (0, 2, 1)), dtype=np.float64
    )
    path["tangent_derivative_coeff"] = np.ascontiguousarray(
        np.transpose(dtan, (0, 2, 1)), dtype=np.float64
    )
    return path


@njit(cache=True, inline="always")
def _power_law(c):
    if c < 0.0:
        c = 0.0
    elif c > 1.0:
        c = 1.0
    cn = c ** 3.0777000000000001
    return (0.1209 + (1.6396999999999999 - 0.1209) * (cn / (cn + 0.79428866592713121))) / 1.002379253316015


@njit(cache=True, inline="always")
def _visibility(c, width, centered):
    if width <= 0.0:
        return 1.0 if c > 0.0 else 0.0
    if centered != 0:
        if c <= -width:
            return 0.0
        if c >= width:
            return 1.0
        u = (c + width) / (2.0 * width)
    else:
        if c <= 0.0:
            return 0.0
        if c >= width:
            return 1.0
        u = c / width
    return u * u * (3.0 - 2.0 * u)


@njit(cache=True, inline="always")
def _interp_vec(t, a, b):
    return a + t * (b - a)


@njit(cache=True, fastmath=True)
def _curved_primary_aperture_kernel(
    pmt_pos, pmt_normal,
    sgrid, path_pos, path_tan, path_dt,
    energy, cos_ch, dcos_ds, ft, particle_time,
    aperture_xy, aperture_w, aperture_radius,
    ngeo_norm, ngeo_radius, intensity,
    start_time, group_index_over_c,
    cost_soft, cost_soft_centered,
):
    npmts = pmt_pos.shape[0]
    ns = sgrid.size
    nq = aperture_w.size
    mu = np.zeros(npmts, dtype=np.float64)
    tmean = np.empty(npmts, dtype=np.float64)
    smean = np.empty(npmts, dtype=np.float64)
    for ip in range(npmts):
        nx = pmt_normal[ip, 0]; ny = pmt_normal[ip, 1]; nz = pmt_normal[ip, 2]
        # deterministic aperture basis
        if abs(nz) < 0.9:
            ax = -ny; ay = nx; az = 0.0
        else:
            ax = 0.0; ay = -nz; az = ny
        an = math.sqrt(ax*ax + ay*ay + az*az)
        if an <= 1e-20:
            ax=1.0;ay=0.0;az=0.0;an=1.0
        ax/=an;ay/=an;az/=an
        bx = ny*az - nz*ay
        by = nz*ax - nx*az
        bz = nx*ay - ny*ax
        smu=0.0; st=0.0; ssacc=0.0
        for iq in range(nq):
            px = pmt_pos[ip,0] + aperture_radius*(aperture_xy[iq,0]*ax + aperture_xy[iq,1]*bx)
            py = pmt_pos[ip,1] + aperture_radius*(aperture_xy[iq,0]*ay + aperture_xy[iq,1]*by)
            pz = pmt_pos[ip,2] + aperture_radius*(aperture_xy[iq,0]*az + aperture_xy[iq,1]*bz)
            prev_f = 0.0
            prev_valid = False
            for j in range(ns-1):
                # f at left and right grid points
                dx0=px-path_pos[j,0];dy0=py-path_pos[j,1];dz0=pz-path_pos[j,2]
                r0=math.sqrt(dx0*dx0+dy0*dy0+dz0*dz0)+1e-12
                f0=(path_tan[j,0]*dx0+path_tan[j,1]*dy0+path_tan[j,2]*dz0)/r0-cos_ch[j]
                dx1=px-path_pos[j+1,0];dy1=py-path_pos[j+1,1];dz1=pz-path_pos[j+1,2]
                r1=math.sqrt(dx1*dx1+dy1*dy1+dz1*dz1)+1e-12
                f1=(path_tan[j+1,0]*dx1+path_tan[j+1,1]*dy1+path_tan[j+1,2]*dz1)/r1-cos_ch[j+1]
                if (f0 == 0.0) or (f0*f1 < 0.0) or (f1 == 0.0):
                    den=f0-f1
                    aroot=0.5 if abs(den)<1e-20 else f0/den
                    if aroot<0.0:aroot=0.0
                    elif aroot>1.0:aroot=1.0
                    sx=_interp_vec(aroot,path_pos[j,0],path_pos[j+1,0])
                    sy=_interp_vec(aroot,path_pos[j,1],path_pos[j+1,1])
                    sz=_interp_vec(aroot,path_pos[j,2],path_pos[j+1,2])
                    tx=_interp_vec(aroot,path_tan[j,0],path_tan[j+1,0])
                    ty=_interp_vec(aroot,path_tan[j,1],path_tan[j+1,1])
                    tz=_interp_vec(aroot,path_tan[j,2],path_tan[j+1,2])
                    tn=math.sqrt(tx*tx+ty*ty+tz*tz)+1e-30;tx/=tn;ty/=tn;tz/=tn
                    kx=_interp_vec(aroot,path_dt[j,0],path_dt[j+1,0])
                    ky=_interp_vec(aroot,path_dt[j,1],path_dt[j+1,1])
                    kz=_interp_vec(aroot,path_dt[j,2],path_dt[j+1,2])
                    dx=px-sx;dy=py-sy;dz=pz-sz
                    r=math.sqrt(dx*dx+dy*dy+dz*dz)+0.01
                    rx=dx/r;ry=dy/r;rz=dz/r
                    c=tx*rx+ty*ry+tz*rz
                    dc=_interp_vec(aroot,dcos_ds[j],dcos_ds[j+1])
                    reff=math.sqrt(r*r+ngeo_radius*ngeo_radius)
                    # Generalized cone-map Jacobian.  For a straight path this
                    # is exactly the active analytic Ngeo denominator.
                    fp=kx*rx+ky*ry+kz*rz-(1.0-c*c)/reff-dc
                    denom=reff*reff*abs(fp)
                    if denom>1e-20 and math.isfinite(denom):
                        facing=-(rx*nx+ry*ny+rz*nz)
                        vis=_visibility(facing,cost_soft,cost_soft_centered)
                        if vis>0.0:
                            amp=(intensity*ngeo_norm*_interp_vec(aroot,ft[j],ft[j+1])
                                 *_power_law(facing)*vis/denom*aperture_w[iq])
                            if amp>0.0 and math.isfinite(amp):
                                sr=_interp_vec(aroot,sgrid[j],sgrid[j+1])
                                tt=start_time+_interp_vec(aroot,particle_time[j],particle_time[j+1])+r*group_index_over_c
                                smu+=amp;st+=amp*tt;ssacc+=amp*sr
                # avoid double counting a root exactly on a shared grid point
                if f1 == 0.0:
                    break
        mu[ip]=smu
        if smu>0.0:
            tmean[ip]=st/smu;smean[ip]=ssacc/smu
        else:
            tmean[ip]=np.nan;smean[ip]=np.nan
    return mu,tmean,smean


def curved_primary_field(emitter, pmt_positions, pmt_normals, coefficients, *, n_grid=81,
                         aperture_radius_mm=45.0, aperture_points_xy=None,
                         aperture_weights=None):
    """Evaluate nonlinear direct-primary charge/time for one coherent FE path."""
    path = build_arclength_fe_path(emitter, coefficients, n_grid=n_grid)
    if aperture_points_xy is None:
        aperture_points_xy = _APERTURE_XY
        aperture_weights = _APERTURE_W
    ng = float(getattr(emitter, "direct_group_index", 1.384730463))
    # Active first-arrival direct timing uses the same detected-spectrum group
    # index as the reflection/direct molecular state.  Fall back to the current
    # WCTE constant when the Emitter has no explicit field.
    ng = 1.384730463 if not math.isfinite(ng) else ng
    mu, t, sroot = _curved_primary_aperture_kernel(
        np.ascontiguousarray(pmt_positions, dtype=np.float64),
        np.ascontiguousarray(pmt_normals, dtype=np.float64),
        path["s"], path["position"], path["tangent"], path["tangent_derivative"],
        path["energy"], path["cos_cherenkov"], path["dcos_ds"],
        path["frank_tamm"], path["particle_time_ns"],
        np.ascontiguousarray(aperture_points_xy, dtype=np.float64),
        np.ascontiguousarray(aperture_weights, dtype=np.float64),
        float(aperture_radius_mm), float(emitter.primary_ngeo_normalization()),
        float(emitter.primary_ngeo_pmt_radius_mm), float(emitter.intensity),
        float(emitter.starting_time), float(ng/emitter.c),
        float(getattr(emitter,"primary_cost_soft",0.0)),
        int(1 if getattr(emitter,"primary_cost_soft_centered",False) else 0),
    )
    return mu, t, sroot, path

# Three-point Gauss--Legendre rule on [0,1] for the finite-aperture line integral.
_GL3_X = np.ascontiguousarray([
    0.5 * (1.0 - math.sqrt(3.0 / 5.0)),
    0.5,
    0.5 * (1.0 + math.sqrt(3.0 / 5.0)),
], dtype=np.float64)
_GL3_W = np.ascontiguousarray([5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0], dtype=np.float64)


@njit(cache=True, fastmath=True)
def _curved_primary_finite_disk_line_kernel(
    pmt_pos, pmt_normal,
    sgrid, path_pos, path_tan,
    cos_ch, ft, particle_time,
    aperture_radius,
    ngeo_norm, ngeo_radius, intensity,
    start_time, group_index_over_c,
    cost_soft, cost_soft_centered,
    gl_x, gl_w,
):
    """Finite-aperture Cherenkov line integral for a coherent curved path.

    This evaluates the small-aperture analytic reduction

        <delta(f)>_disk = 2/(pi h) sqrt(1-(f/h)^2), |f|<h,

    where f=t(s).rhat-cos(theta_c) and h=a |P_disk grad_P f|.  It is
    the exact linearized integral over a uniformly filled circular aperture.
    Integrating it over s recovers 1/|df/ds| for an isolated straight-track
    root, but remains finite at cone caustics and turns on finite-aperture
    receivers whose centre ray has no root.
    """
    npmts = pmt_pos.shape[0]
    ns = sgrid.size
    ngl = gl_x.size
    mu = np.zeros(npmts, dtype=np.float64)
    tmean = np.empty(npmts, dtype=np.float64)
    smean = np.empty(npmts, dtype=np.float64)
    inv_pi = 1.0 / math.pi
    for ip in range(npmts):
        nx = pmt_normal[ip, 0]
        ny = pmt_normal[ip, 1]
        nz = pmt_normal[ip, 2]
        smu = 0.0
        st = 0.0
        ssacc = 0.0
        px = pmt_pos[ip, 0]
        py = pmt_pos[ip, 1]
        pz = pmt_pos[ip, 2]
        for j in range(ns - 1):
            ds = sgrid[j + 1] - sgrid[j]
            if ds <= 0.0:
                continue
            for ig in range(ngl):
                aroot = gl_x[ig]
                wq = ds * gl_w[ig]
                sx = _interp_vec(aroot, path_pos[j, 0], path_pos[j + 1, 0])
                sy = _interp_vec(aroot, path_pos[j, 1], path_pos[j + 1, 1])
                sz = _interp_vec(aroot, path_pos[j, 2], path_pos[j + 1, 2])
                tx = _interp_vec(aroot, path_tan[j, 0], path_tan[j + 1, 0])
                ty = _interp_vec(aroot, path_tan[j, 1], path_tan[j + 1, 1])
                tz = _interp_vec(aroot, path_tan[j, 2], path_tan[j + 1, 2])
                tn = math.sqrt(tx * tx + ty * ty + tz * tz) + 1.0e-30
                tx /= tn
                ty /= tn
                tz /= tn
                dx = px - sx
                dy = py - sy
                dz = pz - sz
                r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1.0e-12
                rx = dx / r
                ry = dy / r
                rz = dz / r
                cview = tx * rx + ty * ry + tz * rz
                cc = _interp_vec(aroot, cos_ch[j], cos_ch[j + 1])
                f = cview - cc

                # Position derivative of t.rhat, projected into the PMT disk.
                # grad_P f = (t-cview*rhat)/r.
                gx = (tx - cview * rx) / r
                gy = (ty - cview * ry) / r
                gz = (tz - cview * rz) / r
                gdotn = gx * nx + gy * ny + gz * nz
                gx -= gdotn * nx
                gy -= gdotn * ny
                gz -= gdotn * nz
                gmag = math.sqrt(gx * gx + gy * gy + gz * gz)
                h = aperture_radius * gmag
                if h <= 1.0e-14 or abs(f) >= h:
                    continue
                z = f / h
                disk_delta = 2.0 * inv_pi * math.sqrt(max(1.0 - z * z, 0.0)) / h

                facing = -(rx * nx + ry * ny + rz * nz)
                vis = _visibility(facing, cost_soft, cost_soft_centered)
                if vis <= 0.0:
                    continue
                reff2 = r * r + ngeo_radius * ngeo_radius
                amp = (
                    intensity * ngeo_norm
                    * _interp_vec(aroot, ft[j], ft[j + 1])
                    * _power_law(facing) * vis
                    * disk_delta / reff2 * wq
                )
                if amp <= 0.0 or not math.isfinite(amp):
                    continue
                sr = _interp_vec(aroot, sgrid[j], sgrid[j + 1])
                tt = (
                    start_time
                    + _interp_vec(aroot, particle_time[j], particle_time[j + 1])
                    + r * group_index_over_c
                )
                smu += amp
                st += amp * tt
                ssacc += amp * sr
        mu[ip] = smu
        if smu > 0.0:
            tmean[ip] = st / smu
            smean[ip] = ssacc / smu
        else:
            tmean[ip] = np.nan
            smean[ip] = np.nan
    return mu, tmean, smean


def curved_primary_finite_disk_line_field(
    emitter, pmt_positions, pmt_normals, coefficients, *, n_grid=81,
    aperture_radius_mm=45.0,
):
    """Robust nonlinear direct-primary field using the finite-disk line integral.

    The model is analytic and contains no simulation-derived scale.  Its zero-path
    value can be difference-matched to the accepted production mean, while its
    coherent path response supplies the event-specific MCS correction.
    """
    path = build_arclength_fe_path(emitter, coefficients, n_grid=n_grid)
    ng = float(getattr(emitter, "direct_group_index", 1.384730463))
    ng = 1.384730463 if not math.isfinite(ng) else ng
    mu, t, sroot = _curved_primary_finite_disk_line_kernel(
        np.ascontiguousarray(pmt_positions, dtype=np.float64),
        np.ascontiguousarray(pmt_normals, dtype=np.float64),
        path["s"], path["position"], path["tangent"],
        path["cos_cherenkov"], path["frank_tamm"], path["particle_time_ns"],
        float(aperture_radius_mm), float(emitter.primary_ngeo_normalization()),
        float(emitter.primary_ngeo_pmt_radius_mm), float(emitter.intensity),
        float(emitter.starting_time), float(ng / emitter.c),
        float(getattr(emitter, "primary_cost_soft", 0.0)),
        int(1 if getattr(emitter, "primary_cost_soft_centered", False) else 0),
        _GL3_X, _GL3_W,
    )
    return mu, t, sroot, path

@njit(cache=True, fastmath=True)
def _curved_delta_kernel(
    pmt_pos, pmt_normal,
    src_pos, src_tan, src_lo, src_hi, src_time,
    ds_cm, K_mu, K_grid, u_grid, table,
    use_finite_disk, pmt_radius_mm, ref_r_mm, distance_power,
    analytic_delta_scale, source_k_power, source_k_ref, source_k_floor,
    intensity, node_group_index_over_c, cost_soft, use_seg_gate,
):
    """Refined analytic delta field on a coherent primary trajectory.

    This is the same dS_delta/du and finite-receiver construction as the
    production straight-track accumulator, with only source position, local
    primary tangent and source flight time promoted to per-source arrays.
    """
    n_src = src_pos.shape[0]
    n_pmts = pmt_pos.shape[0]
    out = np.zeros(n_pmts, dtype=np.float64)
    tnum = np.zeros(n_pmts, dtype=np.float64)
    K_min = K_grid[0]; K_max = K_grid[-1]
    inv_dK = 1.0 / (K_grid[1] - K_grid[0])
    inv_du = 1.0 / (u_grid[1] - u_grid[0])
    nK = K_grid.size; nU = u_grid.size
    omega_ref = 1.0 - ref_r_mm / math.sqrt(ref_r_mm * ref_r_mm + pmt_radius_mm * pmt_radius_mm)
    for i in range(n_pmts):
        px=pmt_pos[i,0];py=pmt_pos[i,1];pz=pmt_pos[i,2]
        nx=pmt_normal[i,0];ny=pmt_normal[i,1];nz=pmt_normal[i,2]
        sm=0.0; st=0.0
        for j in range(n_src):
            K=K_mu[j]; ds=ds_cm[j]
            if K<=0.0 or ds<=0.0 or not math.isfinite(K) or not math.isfinite(ds):
                continue
            sw=1.0
            if source_k_power != 0.0 and source_k_ref > 0.0:
                sw=(max(K,source_k_floor)/source_k_ref)**source_k_power
            wsrc=analytic_delta_scale*sw*ds
            if wsrc<=0.0 or not math.isfinite(wsrc): continue
            dx=px-src_pos[j,0];dy=py-src_pos[j,1];dz=pz-src_pos[j,2]
            r2=dx*dx+dy*dy+dz*dz
            if r2<=0.0: continue
            r=math.sqrt(r2)+0.01; invr=1.0/r
            ux=src_tan[j,0];uy=src_tan[j,1];uz=src_tan[j,2]
            u=(dx*ux+dy*uy+dz*uz)*invr
            if not math.isfinite(u): continue
            if u<-1.0: u=-1.0
            elif u>1.0: u=1.0
            cost=-(dx*nx+dy*ny+dz*nz)*invr
            if not math.isfinite(cost): continue
            vis_seg=1.0
            if use_seg_gate != 0:
                alo=-((px-src_lo[j,0])*nx+(py-src_lo[j,1])*ny+(pz-src_lo[j,2])*nz)
                ahi=-((px-src_hi[j,0])*nx+(py-src_hi[j,1])*ny+(pz-src_hi[j,2])*nz)
                if alo<=0.0 and ahi<=0.0: continue
                if alo>0.0 and ahi>0.0: vis_seg=1.0
                elif ahi>0.0: vis_seg=ahi/(ahi-alo)
                else: vis_seg=alo/(alo-ahi)
                if cost<=0.0: cost=0.0
            elif cost<=0.0:
                continue
            if cost_soft>0.0:
                if cost>=cost_soft: vis=1.0
                else:
                    uu=max(cost,0.0)/cost_soft
                    vis=uu*uu*(3.0-2.0*uu)
            else: vis=1.0
            Kc=min(max(K,K_min),K_max)
            iK=int(math.floor((Kc-K_min)*inv_dK));iK=max(0,min(iK,nK-2))
            tK=(Kc-K_grid[iK])/(K_grid[iK+1]-K_grid[iK]+1e-300)
            uc=min(max(u,u_grid[0]),u_grid[-1])
            iu=int(math.floor((uc-u_grid[0])*inv_du));iu=max(0,min(iu,nU-2))
            tu=(uc-u_grid[iu])/(u_grid[iu+1]-u_grid[iu]+1e-300)
            p0=table[iK,iu]+tu*(table[iK,iu+1]-table[iK,iu])
            p1=table[iK+1,iu]+tu*(table[iK+1,iu+1]-table[iK+1,iu])
            kern=p0+tK*(p1-p0)
            if kern<=0.0 or not math.isfinite(kern): continue
            pwr=_power_law(cost)
            if use_finite_disk and pmt_radius_mm>0.0 and omega_ref>0.0:
                shape=1.0-r/math.sqrt(r*r+pmt_radius_mm*pmt_radius_mm)
                optical=max(shape/omega_ref,0.0)*pwr
            else:
                optical=(ref_r_mm/r)**distance_power*pwr
            contrib=wsrc*optical*kern*vis*vis_seg*intensity
            if contrib<=0.0 or not math.isfinite(contrib):continue
            sm+=contrib;st+=contrib*(src_time[j]+r*node_group_index_over_c)
        out[i]=sm;tnum[i]=st
    t=np.empty(n_pmts,dtype=np.float64)
    for i in range(n_pmts):
        t[i]=tnum[i]/out[i] if out[i]>0.0 else np.nan
    return out,t


def _interpolate_path_state(path, s_values):
    """Interpolate/extrapolate coherent position and tangent at source coordinates."""
    s=np.asarray(s_values,dtype=np.float64);sg=path['s'];pos=path['position'];tan=path['tangent']
    outp=np.empty((s.size,3),dtype=np.float64);outt=np.empty((s.size,3),dtype=np.float64)
    for k in range(3):
        outp[:,k]=np.interp(np.minimum(s,sg[-1]),sg,pos[:,k])
        outt[:,k]=np.interp(np.minimum(s,sg[-1]),sg,tan[:,k])
    over=s>sg[-1]
    if np.any(over): outp[over]+= (s[over]-sg[-1])[:,None]*tan[-1][None,:]
    outt/=np.maximum(np.linalg.norm(outt,axis=1)[:,None],1e-30)
    return np.ascontiguousarray(outp),np.ascontiguousarray(outt)


def curved_delta_field(emitter,pmt_positions,pmt_normals,coefficients,*,n_grid=81):
    """Evaluate the refined delta-electron field on one coherent FE path."""
    path=build_arclength_fe_path(emitter,coefficients,n_grid=n_grid)
    s_centers,ds_cm,K_mu,valid=emitter._build_delta_source_grid()
    if not valid:
        z=np.zeros(len(pmt_positions),dtype=np.float64);return z,np.full_like(z,np.nan),path
    s_centers=np.asarray(s_centers,dtype=np.float64);ds_cm=np.asarray(ds_cm,dtype=np.float64)
    pos,tan=_interpolate_path_state(path,s_centers)
    half=5.0*ds_cm
    plo,_=_interpolate_path_state(path,s_centers-half);phi,_=_interpolate_path_state(path,s_centers+half)
    K_grid,u_grid,table=emod.get_refined_analytic_delta_cache(emitter.n,projectile_mass=float(emitter.particle_mass),particle=emitter.particle_name)
    src_time=np.asarray(emod._wcte_integrated_primary_tof_fast(emitter,s_centers),dtype=np.float64)+float(emitter.starting_time)+float(getattr(emitter,'delta_e_time_offset_ns',0.0))
    mu,t=_curved_delta_kernel(
        np.ascontiguousarray(pmt_positions,dtype=np.float64),np.ascontiguousarray(pmt_normals,dtype=np.float64),
        pos,tan,plo,phi,np.ascontiguousarray(src_time),np.ascontiguousarray(ds_cm),np.ascontiguousarray(K_mu,dtype=np.float64),
        np.ascontiguousarray(K_grid,dtype=np.float64),np.ascontiguousarray(u_grid,dtype=np.float64),np.ascontiguousarray(table,dtype=np.float64),
        bool(getattr(emitter,'delta_e_use_finite_disk_solid_angle',True)),float(getattr(emitter,'delta_e_distance_pmt_radius_mm',37.0)),
        float(getattr(emitter,'delta_e_distance_ref_r_mm',1000.0)),float(getattr(emitter,'delta_e_distance_power',2.0)),
        float(getattr(emitter,'analytic_delta_scale',1.0)),float(getattr(emitter,'delta_e_source_k_power',0.0)),
        float(getattr(emitter,'delta_e_source_k_ref_MeV',100.0)),float(getattr(emitter,'delta_e_source_k_floor_MeV',25.0)),
        float(emitter.intensity),float(1.384730463/emitter.c),float(getattr(emitter,'delta_e_cost_soft',0.0)),
        int(1 if (getattr(emitter,'smooth_tables',True) if getattr(emitter,'delta_e_segment_gate',None) is None else bool(getattr(emitter,'delta_e_segment_gate'))) else 0),
    )
    return mu,t,path

# Five-point Gauss--Legendre nodes on [0,1].  Unlike the older fixed whole-segment
# rule, the interval kernel below first solves analytically for the part of each
# path segment whose linearized finite PMT disk intersects the Cherenkov cone.
# The quadrature nodes therefore move with the physical support boundary instead
# of abruptly entering or leaving it as the track changes.
_GL5_RAW_X, _GL5_RAW_W = np.polynomial.legendre.leggauss(5)
_GL5_X = np.ascontiguousarray(0.5 * (_GL5_RAW_X + 1.0), dtype=np.float64)
_GL5_W = np.ascontiguousarray(0.5 * _GL5_RAW_W, dtype=np.float64)


@njit(cache=True, inline="always")
def _linear_positive_interval(v0, v1, lo, hi):
    """Intersect [lo,hi] with the region where linear v(a)>0, a in [0,1]."""
    if hi <= lo:
        return lo, lo
    if v0 > 0.0 and v1 > 0.0:
        return lo, hi
    if v0 <= 0.0 and v1 <= 0.0:
        return lo, lo
    den = v1 - v0
    if abs(den) < 1.0e-30:
        return lo, lo
    cross = -v0 / den
    if cross < 0.0:
        cross = 0.0
    elif cross > 1.0:
        cross = 1.0
    if v0 > 0.0:
        if cross < hi:
            hi = cross
    else:
        if cross > lo:
            lo = cross
    if hi <= lo:
        return lo, lo
    return lo, hi


@njit(cache=True, inline="always")
def _fali_f_h(px, py, pz, nx, ny, nz, sx, sy, sz, tx, ty, tz, cc, aperture_radius):
    tn = math.sqrt(tx * tx + ty * ty + tz * tz) + 1.0e-30
    tx /= tn; ty /= tn; tz /= tn
    dx = px - sx; dy = py - sy; dz = pz - sz
    r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1.0e-12
    rx = dx / r; ry = dy / r; rz = dz / r
    cview = tx * rx + ty * ry + tz * rz
    f = cview - cc
    gx = (tx - cview * rx) / r
    gy = (ty - cview * ry) / r
    gz = (tz - cview * rz) / r
    gdotn = gx * nx + gy * ny + gz * nz
    gx -= gdotn * nx; gy -= gdotn * ny; gz -= gdotn * nz
    h = aperture_radius * math.sqrt(gx * gx + gy * gy + gz * gz)
    return f, h


@njit(cache=True, fastmath=True, parallel=True)
def _curved_primary_finite_disk_interval_kernel(
    pmt_pos, pmt_normal,
    sgrid, path_pos, path_tan,
    cos_ch, ft, particle_time,
    aperture_radius,
    ngeo_norm, ngeo_radius, intensity,
    start_time, group_index_over_c,
    cost_soft, cost_soft_centered,
    gl_x, gl_w,
):
    """Support-tracked finite-aperture line integral for a coherent FE path.

    On each path interval, f=t.rhat-cos(theta_c) and the disk half-width h are
    linearly interpolated.  The active subinterval satisfying -h<f<h is found
    analytically from the two linear inequalities f+h>0 and h-f>0.  Gaussian
    quadrature is then applied only on that moving physical support.  This
    removes the grid-crossing roughness of the original fixed-node FALI kernel
    while retaining the exact filled-disk delta density.
    """
    npmts = pmt_pos.shape[0]
    ns = sgrid.size
    ngl = gl_x.size
    mu = np.zeros(npmts, dtype=np.float64)
    tmean = np.empty(npmts, dtype=np.float64)
    smean = np.empty(npmts, dtype=np.float64)
    inv_pi = 1.0 / math.pi
    for ip in prange(npmts):
        px = pmt_pos[ip, 0]; py = pmt_pos[ip, 1]; pz = pmt_pos[ip, 2]
        nx = pmt_normal[ip, 0]; ny = pmt_normal[ip, 1]; nz = pmt_normal[ip, 2]
        smu = 0.0; st = 0.0; ssacc = 0.0
        for j in range(ns - 1):
            ds = sgrid[j + 1] - sgrid[j]
            if ds <= 0.0:
                continue
            f0, h0 = _fali_f_h(
                px, py, pz, nx, ny, nz,
                path_pos[j, 0], path_pos[j, 1], path_pos[j, 2],
                path_tan[j, 0], path_tan[j, 1], path_tan[j, 2],
                cos_ch[j], aperture_radius,
            )
            f1, h1 = _fali_f_h(
                px, py, pz, nx, ny, nz,
                path_pos[j + 1, 0], path_pos[j + 1, 1], path_pos[j + 1, 2],
                path_tan[j + 1, 0], path_tan[j + 1, 1], path_tan[j + 1, 2],
                cos_ch[j + 1], aperture_radius,
            )
            # Conditions: f+h > 0 and h-f > 0.
            alo = 0.0; ahi = 1.0
            alo, ahi = _linear_positive_interval(f0 + h0, f1 + h1, alo, ahi)
            if ahi <= alo:
                continue
            alo, ahi = _linear_positive_interval(h0 - f0, h1 - f1, alo, ahi)
            if ahi <= alo:
                continue
            width = ahi - alo
            for ig in range(ngl):
                aroot = alo + width * gl_x[ig]
                wq = ds * width * gl_w[ig]
                sx = _interp_vec(aroot, path_pos[j, 0], path_pos[j + 1, 0])
                sy = _interp_vec(aroot, path_pos[j, 1], path_pos[j + 1, 1])
                sz = _interp_vec(aroot, path_pos[j, 2], path_pos[j + 1, 2])
                tx = _interp_vec(aroot, path_tan[j, 0], path_tan[j + 1, 0])
                ty = _interp_vec(aroot, path_tan[j, 1], path_tan[j + 1, 1])
                tz = _interp_vec(aroot, path_tan[j, 2], path_tan[j + 1, 2])
                tn = math.sqrt(tx * tx + ty * ty + tz * tz) + 1.0e-30
                tx /= tn; ty /= tn; tz /= tn
                dx = px - sx; dy = py - sy; dz = pz - sz
                r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1.0e-12
                rx = dx / r; ry = dy / r; rz = dz / r
                cview = tx * rx + ty * ry + tz * rz
                cc = _interp_vec(aroot, cos_ch[j], cos_ch[j + 1])
                f = cview - cc
                gx = (tx - cview * rx) / r
                gy = (ty - cview * ry) / r
                gz = (tz - cview * rz) / r
                gdotn = gx * nx + gy * ny + gz * nz
                gx -= gdotn * nx; gy -= gdotn * ny; gz -= gdotn * nz
                h = aperture_radius * math.sqrt(gx * gx + gy * gy + gz * gz)
                if h <= 1.0e-14:
                    continue
                z = f / h
                # Roundoff at a solved support endpoint can put |z| barely > 1.
                if z <= -1.0 or z >= 1.0:
                    continue
                disk_delta = 2.0 * inv_pi * math.sqrt(max(1.0 - z * z, 0.0)) / h
                facing = -(rx * nx + ry * ny + rz * nz)
                vis = _visibility(facing, cost_soft, cost_soft_centered)
                if vis <= 0.0:
                    continue
                reff2 = r * r + ngeo_radius * ngeo_radius
                amp = (
                    intensity * ngeo_norm
                    * _interp_vec(aroot, ft[j], ft[j + 1])
                    * _power_law(facing) * vis
                    * disk_delta / reff2 * wq
                )
                if amp <= 0.0 or not math.isfinite(amp):
                    continue
                sr = _interp_vec(aroot, sgrid[j], sgrid[j + 1])
                tt = (
                    start_time
                    + _interp_vec(aroot, particle_time[j], particle_time[j + 1])
                    + r * group_index_over_c
                )
                smu += amp; st += amp * tt; ssacc += amp * sr
        mu[ip] = smu
        if smu > 0.0:
            tmean[ip] = st / smu; smean[ip] = ssacc / smu
        else:
            tmean[ip] = np.nan; smean[ip] = np.nan
    return mu, tmean, smean


@njit(cache=True, inline="always")
def _crossing_derivative(v0, v1, dv0, dv1):
    """Derivative of the linear zero crossing ``-v0/(v1-v0)``."""
    den = v1 - v0
    if abs(den) < 1.0e-30:
        return 0.0
    return (-dv0 * den + v0 * (dv1 - dv0)) / (den * den)


@njit(cache=True, inline="always")
def _power_law_derivative(c):
    if c <= 0.0 or c >= 1.0:
        return 0.0
    exponent = 3.0777000000000001
    xhalf = 0.79428866592713121
    cn = c ** exponent
    return (
        (1.6396999999999999 - 0.1209)
        * exponent * (c ** (exponent - 1.0)) * xhalf
        / ((cn + xhalf) * (cn + xhalf))
        / 1.002379253316015
    )


@njit(cache=True, inline="always")
def _visibility_derivative(c, width, centered):
    if width <= 0.0:
        return 0.0
    if centered != 0:
        if c <= -width or c >= width:
            return 0.0
        u = (c + width) / (2.0 * width)
        return 3.0 * u * (1.0 - u) / width
    if c <= 0.0 or c >= width:
        return 0.0
    u = c / width
    return 6.0 * u * (1.0 - u) / width


@njit(cache=True, fastmath=True, parallel=True)
def _curved_primary_finite_disk_interval_charge_jacobian_kernel(
    pmt_pos, pmt_normal,
    sgrid, path_pos, path_tan, path_pos_du, path_tan_du,
    cos_ch, ft,
    aperture_radius,
    ngeo_norm, ngeo_radius, intensity,
    cost_soft, cost_soft_centered,
    gl_x, gl_w,
):
    """FALI charge and exact first derivative in standardized KL space.

    The moving finite-disk support is differentiated as part of the integral.
    Its boundary term vanishes because the filled-disk density is zero at
    ``|f/h|=1``; the remaining dependence enters through the transformed
    quadrature nodes and interval width.  This is the analytic counterpart of
    the validated central finite-difference Jacobian and contains no fitted
    approximation.
    """
    npmts = pmt_pos.shape[0]
    ns = sgrid.size
    ngl = gl_x.size
    nlatent = path_pos_du.shape[2]
    mu = np.zeros(npmts, dtype=np.float64)
    jac = np.zeros((npmts, nlatent), dtype=np.float64)
    inv_pi = 1.0 / math.pi

    for ip in prange(npmts):
        px = pmt_pos[ip, 0]; py = pmt_pos[ip, 1]; pz = pmt_pos[ip, 2]
        nx = pmt_normal[ip, 0]; ny = pmt_normal[ip, 1]; nz = pmt_normal[ip, 2]
        smu = 0.0
        jrow = np.zeros(nlatent, dtype=np.float64)
        df0 = np.empty(nlatent, dtype=np.float64)
        dh0 = np.empty(nlatent, dtype=np.float64)
        df1 = np.empty(nlatent, dtype=np.float64)
        dh1 = np.empty(nlatent, dtype=np.float64)

        for j in range(ns - 1):
            ds = sgrid[j + 1] - sgrid[j]
            if ds <= 0.0:
                continue

            # Endpoint 0 geometry and derivatives.
            sx0 = path_pos[j, 0]; sy0 = path_pos[j, 1]; sz0 = path_pos[j, 2]
            tx0 = path_tan[j, 0]; ty0 = path_tan[j, 1]; tz0 = path_tan[j, 2]
            dx0 = px - sx0; dy0 = py - sy0; dz0 = pz - sz0
            r0 = math.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0) + 1.0e-12
            rx0 = dx0 / r0; ry0 = dy0 / r0; rz0 = dz0 / r0
            cv0 = tx0 * rx0 + ty0 * ry0 + tz0 * rz0
            f0 = cv0 - cos_ch[j]
            gx0 = (tx0 - cv0 * rx0) / r0
            gy0 = (ty0 - cv0 * ry0) / r0
            gz0 = (tz0 - cv0 * rz0) / r0
            gdn0 = gx0 * nx + gy0 * ny + gz0 * nz
            gpx0 = gx0 - gdn0 * nx; gpy0 = gy0 - gdn0 * ny; gpz0 = gz0 - gdn0 * nz
            gm0 = math.sqrt(gpx0 * gpx0 + gpy0 * gpy0 + gpz0 * gpz0)
            h0 = aperture_radius * gm0
            ax0 = tx0 - cv0 * rx0; ay0 = ty0 - cv0 * ry0; az0 = tz0 - cv0 * rz0

            # Endpoint 1 geometry and derivatives.
            sx1 = path_pos[j + 1, 0]; sy1 = path_pos[j + 1, 1]; sz1 = path_pos[j + 1, 2]
            tx1 = path_tan[j + 1, 0]; ty1 = path_tan[j + 1, 1]; tz1 = path_tan[j + 1, 2]
            dx1 = px - sx1; dy1 = py - sy1; dz1 = pz - sz1
            r1 = math.sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1) + 1.0e-12
            rx1 = dx1 / r1; ry1 = dy1 / r1; rz1 = dz1 / r1
            cv1 = tx1 * rx1 + ty1 * ry1 + tz1 * rz1
            f1 = cv1 - cos_ch[j + 1]
            gx1 = (tx1 - cv1 * rx1) / r1
            gy1 = (ty1 - cv1 * ry1) / r1
            gz1 = (tz1 - cv1 * rz1) / r1
            gdn1 = gx1 * nx + gy1 * ny + gz1 * nz
            gpx1 = gx1 - gdn1 * nx; gpy1 = gy1 - gdn1 * ny; gpz1 = gz1 - gdn1 * nz
            gm1 = math.sqrt(gpx1 * gpx1 + gpy1 * gpy1 + gpz1 * gpz1)
            h1 = aperture_radius * gm1
            ax1 = tx1 - cv1 * rx1; ay1 = ty1 - cv1 * ry1; az1 = tz1 - cv1 * rz1

            for k in range(nlatent):
                # Endpoint 0.
                dsx = path_pos_du[j, 0, k]; dsy = path_pos_du[j, 1, k]; dsz = path_pos_du[j, 2, k]
                dtx = path_tan_du[j, 0, k]; dty = path_tan_du[j, 1, k]; dtz = path_tan_du[j, 2, k]
                dr = -(rx0 * dsx + ry0 * dsy + rz0 * dsz)
                drx = (-dsx - rx0 * dr) / r0
                dry = (-dsy - ry0 * dr) / r0
                drz = (-dsz - rz0 * dr) / r0
                dcv = dtx * rx0 + dty * ry0 + dtz * rz0 + tx0 * drx + ty0 * dry + tz0 * drz
                df0[k] = dcv
                dax = dtx - dcv * rx0 - cv0 * drx
                day = dty - dcv * ry0 - cv0 * dry
                daz = dtz - dcv * rz0 - cv0 * drz
                dgx = dax / r0 - ax0 * dr / (r0 * r0)
                dgy = day / r0 - ay0 * dr / (r0 * r0)
                dgz = daz / r0 - az0 * dr / (r0 * r0)
                dgdn = dgx * nx + dgy * ny + dgz * nz
                dgpx = dgx - dgdn * nx; dgpy = dgy - dgdn * ny; dgpz = dgz - dgdn * nz
                if gm0 > 1.0e-20:
                    dh0[k] = aperture_radius * (gpx0 * dgpx + gpy0 * dgpy + gpz0 * dgpz) / gm0
                else:
                    dh0[k] = 0.0

                # Endpoint 1.
                dsx = path_pos_du[j + 1, 0, k]; dsy = path_pos_du[j + 1, 1, k]; dsz = path_pos_du[j + 1, 2, k]
                dtx = path_tan_du[j + 1, 0, k]; dty = path_tan_du[j + 1, 1, k]; dtz = path_tan_du[j + 1, 2, k]
                dr = -(rx1 * dsx + ry1 * dsy + rz1 * dsz)
                drx = (-dsx - rx1 * dr) / r1
                dry = (-dsy - ry1 * dr) / r1
                drz = (-dsz - rz1 * dr) / r1
                dcv = dtx * rx1 + dty * ry1 + dtz * rz1 + tx1 * drx + ty1 * dry + tz1 * drz
                df1[k] = dcv
                dax = dtx - dcv * rx1 - cv1 * drx
                day = dty - dcv * ry1 - cv1 * dry
                daz = dtz - dcv * rz1 - cv1 * drz
                dgx = dax / r1 - ax1 * dr / (r1 * r1)
                dgy = day / r1 - ay1 * dr / (r1 * r1)
                dgz = daz / r1 - az1 * dr / (r1 * r1)
                dgdn = dgx * nx + dgy * ny + dgz * nz
                dgpx = dgx - dgdn * nx; dgpy = dgy - dgdn * ny; dgpz = dgz - dgdn * nz
                if gm1 > 1.0e-20:
                    dh1[k] = aperture_radius * (gpx1 * dgpx + gpy1 * dgpy + gpz1 * dgpz) / gm1
                else:
                    dh1[k] = 0.0

            # Intersect the two linear support inequalities and record which
            # physical crossing defines each moving endpoint.
            alo = 0.0; ahi = 1.0
            alo_code = 0; ahi_code = 0  # 0=fixed, 1=f+h, 2=h-f
            va0 = f0 + h0; va1 = f1 + h1
            if va0 <= 0.0 and va1 <= 0.0:
                continue
            if not (va0 > 0.0 and va1 > 0.0):
                den = va1 - va0
                if abs(den) < 1.0e-30:
                    continue
                cross = -va0 / den
                if cross < 0.0: cross = 0.0
                elif cross > 1.0: cross = 1.0
                if va0 > 0.0:
                    ahi = cross; ahi_code = 1
                else:
                    alo = cross; alo_code = 1
            vb0 = h0 - f0; vb1 = h1 - f1
            if vb0 <= 0.0 and vb1 <= 0.0:
                continue
            if not (vb0 > 0.0 and vb1 > 0.0):
                den = vb1 - vb0
                if abs(den) < 1.0e-30:
                    continue
                cross = -vb0 / den
                if cross < 0.0: cross = 0.0
                elif cross > 1.0: cross = 1.0
                if vb0 > 0.0:
                    if cross < ahi:
                        ahi = cross; ahi_code = 2
                else:
                    if cross > alo:
                        alo = cross; alo_code = 2
            if ahi <= alo:
                continue
            width = ahi - alo

            for ig in range(ngl):
                xq = gl_x[ig]
                aroot = alo + width * xq
                wq = ds * width * gl_w[ig]
                sx = _interp_vec(aroot, sx0, sx1)
                sy = _interp_vec(aroot, sy0, sy1)
                sz = _interp_vec(aroot, sz0, sz1)
                trx = _interp_vec(aroot, tx0, tx1)
                try_ = _interp_vec(aroot, ty0, ty1)
                trz = _interp_vec(aroot, tz0, tz1)
                tn = math.sqrt(trx * trx + try_ * try_ + trz * trz) + 1.0e-30
                tx = trx / tn; ty = try_ / tn; tz = trz / tn
                dx = px - sx; dy = py - sy; dz = pz - sz
                r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1.0e-12
                rx = dx / r; ry = dy / r; rz = dz / r
                cview = tx * rx + ty * ry + tz * rz
                cc = _interp_vec(aroot, cos_ch[j], cos_ch[j + 1])
                f = cview - cc
                gx = (tx - cview * rx) / r
                gy = (ty - cview * ry) / r
                gz = (tz - cview * rz) / r
                gdn = gx * nx + gy * ny + gz * nz
                gpx = gx - gdn * nx; gpy = gy - gdn * ny; gpz = gz - gdn * nz
                gm = math.sqrt(gpx * gpx + gpy * gpy + gpz * gpz)
                h = aperture_radius * gm
                if h <= 1.0e-14:
                    continue
                z = f / h
                if z <= -1.0 or z >= 1.0:
                    continue
                one_minus = max(1.0 - z * z, 1.0e-18)
                disk_delta = 2.0 * inv_pi * math.sqrt(one_minus) / h
                facing = -(rx * nx + ry * ny + rz * nz)
                vis = _visibility(facing, cost_soft, cost_soft_centered)
                if vis <= 0.0:
                    continue
                power = _power_law(facing)
                ftq = _interp_vec(aroot, ft[j], ft[j + 1])
                if ftq <= 0.0 or power <= 0.0 or wq <= 0.0:
                    continue
                reff2 = r * r + ngeo_radius * ngeo_radius
                amp = intensity * ngeo_norm * ftq * power * vis * disk_delta / reff2 * wq
                if amp <= 0.0 or not math.isfinite(amp):
                    continue
                smu += amp

                # Common vectors needed by every latent derivative.
                ax = tx - cview * rx; ay = ty - cview * ry; az = tz - cview * rz
                dpdc = _power_law_derivative(facing)
                dvdc = _visibility_derivative(facing, cost_soft, cost_soft_centered)
                for k in range(nlatent):
                    if alo_code == 1:
                        dalo = _crossing_derivative(
                            va0, va1, df0[k] + dh0[k], df1[k] + dh1[k]
                        )
                    elif alo_code == 2:
                        dalo = _crossing_derivative(
                            vb0, vb1, dh0[k] - df0[k], dh1[k] - df1[k]
                        )
                    else:
                        dalo = 0.0
                    if ahi_code == 1:
                        dahi = _crossing_derivative(
                            va0, va1, df0[k] + dh0[k], df1[k] + dh1[k]
                        )
                    elif ahi_code == 2:
                        dahi = _crossing_derivative(
                            vb0, vb1, dh0[k] - df0[k], dh1[k] - df1[k]
                        )
                    else:
                        dahi = 0.0
                    dwidth = dahi - dalo
                    da = dalo + xq * dwidth
                    dwq = ds * gl_w[ig] * dwidth

                    dsx = _interp_vec(aroot, path_pos_du[j, 0, k], path_pos_du[j + 1, 0, k]) + da * (sx1 - sx0)
                    dsy = _interp_vec(aroot, path_pos_du[j, 1, k], path_pos_du[j + 1, 1, k]) + da * (sy1 - sy0)
                    dsz = _interp_vec(aroot, path_pos_du[j, 2, k], path_pos_du[j + 1, 2, k]) + da * (sz1 - sz0)
                    dtrx = _interp_vec(aroot, path_tan_du[j, 0, k], path_tan_du[j + 1, 0, k]) + da * (tx1 - tx0)
                    dtry = _interp_vec(aroot, path_tan_du[j, 1, k], path_tan_du[j + 1, 1, k]) + da * (ty1 - ty0)
                    dtrz = _interp_vec(aroot, path_tan_du[j, 2, k], path_tan_du[j + 1, 2, k]) + da * (tz1 - tz0)
                    tdot = tx * dtrx + ty * dtry + tz * dtrz
                    dtx = (dtrx - tx * tdot) / tn
                    dty = (dtry - ty * tdot) / tn
                    dtz = (dtrz - tz * tdot) / tn

                    dr = -(rx * dsx + ry * dsy + rz * dsz)
                    drx = (-dsx - rx * dr) / r
                    dry = (-dsy - ry * dr) / r
                    drz = (-dsz - rz * dr) / r
                    dcview = dtx * rx + dty * ry + dtz * rz + tx * drx + ty * dry + tz * drz
                    dcc = da * (cos_ch[j + 1] - cos_ch[j])
                    df = dcview - dcc
                    dax = dtx - dcview * rx - cview * drx
                    day = dty - dcview * ry - cview * dry
                    daz = dtz - dcview * rz - cview * drz
                    dgx = dax / r - ax * dr / (r * r)
                    dgy = day / r - ay * dr / (r * r)
                    dgz = daz / r - az * dr / (r * r)
                    dgdn = dgx * nx + dgy * ny + dgz * nz
                    dgpx = dgx - dgdn * nx; dgpy = dgy - dgdn * ny; dgpz = dgz - dgdn * nz
                    dh = aperture_radius * (gpx * dgpx + gpy * dgpy + gpz * dgpz) / max(gm, 1.0e-30)
                    dzeta = (df * h - f * dh) / (h * h)
                    ddisk_over_disk = -z * dzeta / one_minus - dh / h
                    dfacing = -(drx * nx + dry * ny + drz * nz)
                    dft = da * (ft[j + 1] - ft[j])
                    dlog = (
                        dft / ftq
                        + (dpdc / power) * dfacing
                        + (dvdc / vis) * dfacing
                        + ddisk_over_disk
                        - (2.0 * r * dr) / reff2
                        + dwq / wq
                    )
                    contribution = amp * dlog
                    if math.isfinite(contribution):
                        jrow[k] += contribution

        mu[ip] = smu
        for k in range(nlatent):
            jac[ip, k] = jrow[k]
    return mu, jac


def _fali_quadrature_for_emitter(emitter):
    """Return the configured support-interval quadrature.

    Three nodes reproduce the five-node FALI charge field at sub-per-mille
    normalized L1 accuracy in the validated 300--500 MeV geometries while
    reducing the hot receiver work.  Five nodes remain available as the
    conservative reference.
    """
    nodes = int(getattr(emitter, "primary_mcs_fali_quadrature_nodes", 5))
    if nodes <= 3:
        return _GL3_X, _GL3_W
    return _GL5_X, _GL5_W


def curved_primary_finite_disk_interval_charge_jacobian_field(
    emitter, pmt_positions, pmt_normals, coefficients, *, n_grid=81,
    aperture_radius_mm=45.0,
):
    """Return the smooth FALI charge and its analytic KL Jacobian."""
    path = build_arclength_fe_path_with_derivatives(
        emitter, coefficients, n_grid=n_grid
    )
    mu, jac = _curved_primary_finite_disk_interval_charge_jacobian_kernel(
        np.ascontiguousarray(pmt_positions, dtype=np.float64),
        np.ascontiguousarray(pmt_normals, dtype=np.float64),
        path["s"], path["position"], path["tangent"],
        path["position_derivative"], path["tangent_derivative_coeff"],
        path["cos_cherenkov"], path["frank_tamm"],
        float(aperture_radius_mm), float(emitter.primary_ngeo_normalization()),
        float(emitter.primary_ngeo_pmt_radius_mm), float(emitter.intensity),
        float(getattr(emitter, "primary_cost_soft", 0.0)),
        int(1 if getattr(emitter, "primary_cost_soft_centered", False) else 0),
        *_fali_quadrature_for_emitter(emitter),
    )
    return mu, jac, path


def curved_primary_finite_disk_interval_field(
    emitter, pmt_positions, pmt_normals, coefficients, *, n_grid=81,
    aperture_radius_mm=45.0,
):
    """Smooth support-tracked FALI direct-primary field."""
    path = build_arclength_fe_path(emitter, coefficients, n_grid=n_grid)
    ng = float(getattr(emitter, "direct_group_index", 1.384730463))
    ng = 1.384730463 if not math.isfinite(ng) else ng
    mu, t, sroot = _curved_primary_finite_disk_interval_kernel(
        np.ascontiguousarray(pmt_positions, dtype=np.float64),
        np.ascontiguousarray(pmt_normals, dtype=np.float64),
        path["s"], path["position"], path["tangent"],
        path["cos_cherenkov"], path["frank_tamm"], path["particle_time_ns"],
        float(aperture_radius_mm), float(emitter.primary_ngeo_normalization()),
        float(emitter.primary_ngeo_pmt_radius_mm), float(emitter.intensity),
        float(emitter.starting_time), float(ng / emitter.c),
        float(getattr(emitter, "primary_cost_soft", 0.0)),
        int(1 if getattr(emitter, "primary_cost_soft_centered", False) else 0),
        *_fali_quadrature_for_emitter(emitter),
    )
    return mu, t, sroot, path
