"""Analytic low-rank Fermi--Eyges charge-process model for LicketyFit.

Production module. It builds parameter-free raw Karhunen--Loeve modes of the
projected Fermi--Eyges displacement process, differentiates the sharp primary
charge prediction with respect to those standardized coherent trajectory
coordinates, and performs the validated low-rank generalized-estimating-equation
update. No WCSim-derived scale, template, or fitted MCS width enters the model.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit

from . import Emitter as emod

_KL_CACHE = {}
_KL_CACHE_MAX = 256


def stable_transverse_basis(direction):
    n = np.asarray(direction, dtype=np.float64)
    n /= max(float(np.linalg.norm(n)), 1e-30)
    if n[2] < -0.999999999:
        e1 = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        e2 = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        e1 = np.array([1.0 - n[0] * n[0] * a, b, -n[0]], dtype=np.float64)
        e2 = np.array([b, 1.0 - n[1] * n[1] * a, -n[1]], dtype=np.float64)
    return np.ascontiguousarray(n), np.ascontiguousarray(e1), np.ascontiguousarray(e2)


def build_raw_fe_kl_basis(emitter, n_modes_per_plane=4, n_grid=41):
    """Return raw Fermi--Eyges KL displacement/slope/curvature modes.

    The fitted vertex and direction are the physical initial state.  Therefore
    the raw process, whose displacement and slope both vanish at s=0, is the
    appropriate stochastic trajectory basis; no line/chord component is
    projected out.
    """
    n_modes = max(1, min(int(n_modes_per_plane), max(1, int(n_grid) - 2)))
    n_grid = max(17, min(int(n_grid), 401))
    pname = emod.canonical_particle_name(getattr(emitter, "particle_name", "muon"))
    L = max(float(getattr(emitter, "length", 0.0)), 0.0)
    R = max(float(getattr(emitter, "range_to_threshold_mm", L)), 0.0)
    X0 = max(float(getattr(emitter, "primary_mcs_radiation_length_mm", 360.8)), 1e-30)
    zq = abs(float(getattr(emitter, "primary_mcs_charge_number", 1.0)))
    key = (pname, L, R, float(getattr(emitter, "n", 1.344)), X0, zq, n_modes, n_grid)
    cached = _KL_CACHE.get(key)
    if cached is not None:
        return cached

    if L <= 0.0:
        sg = np.asarray([0.0], dtype=np.float64)
        shapes = np.zeros((1, n_modes), dtype=np.float64)
        slopes = np.zeros_like(shapes)
        curv = np.zeros_like(shapes)
        frac = np.zeros(n_modes, dtype=np.float64)
    else:
        sg = np.linspace(0.0, L, n_grid, dtype=np.float64)
        K = np.asarray(emitter.muon_energy_at_s_array(sg, R), dtype=np.float64)
        mass = float(getattr(emitter, "particle_mass", emod.particle_mass_mev(pname)))
        gamma = 1.0 + K / mass
        beta2 = np.maximum(1.0 - 1.0 / np.maximum(gamma * gamma, 1.0), 1e-15)
        beta = np.sqrt(beta2)
        mom = np.sqrt(np.maximum(K * (K + 2.0 * mass), 1e-30))
        T = (13.6 * zq / np.maximum(beta * mom, 1e-30)) ** 2 / X0

        I0 = np.zeros_like(sg)
        I1 = np.zeros_like(sg)
        I2 = np.zeros_like(sg)
        ds = np.diff(sg)
        I0[1:] = np.cumsum(0.5 * (T[1:] + T[:-1]) * ds)
        I1[1:] = np.cumsum(0.5 * (sg[1:] * T[1:] + sg[:-1] * T[:-1]) * ds)
        I2[1:] = np.cumsum(0.5 * (sg[1:] ** 2 * T[1:] + sg[:-1] ** 2 * T[:-1]) * ds)
        midx = np.minimum.outer(np.arange(sg.size), np.arange(sg.size))
        C = (
            np.outer(sg, sg) * I0[midx]
            - (sg[:, None] + sg[None, :]) * I1[midx]
            + I2[midx]
        )

        ft = np.asarray(
            emod._cherenkov_weight_from_energy(K, mass, float(getattr(emitter, "n", 1.344))),
            dtype=np.float64,
        )
        quad = np.ones_like(sg)
        quad[[0, -1]] = 0.5
        quad *= L / max(sg.size - 1, 1)
        w = ft * quad
        active = np.flatnonzero(w > 1e-14)
        if active.size < 3:
            shapes = np.zeros((sg.size, n_modes), dtype=np.float64)
            slopes = np.zeros_like(shapes)
            curv = np.zeros_like(shapes)
            frac = np.zeros(n_modes, dtype=np.float64)
        else:
            wa = w[active]
            Ca = C[np.ix_(active, active)]
            sw = np.sqrt(wa)
            B = (sw[:, None] * Ca) * sw[None, :]
            evals, evecs = np.linalg.eigh(0.5 * (B + B.T))
            order = np.argsort(evals)[::-1]
            evals = np.maximum(evals[order], 0.0)
            evecs = evecs[:, order]
            total = max(float(np.sum(evals)), 1e-300)
            nm = min(n_modes, int(np.count_nonzero(evals > 1e-18 * evals[0])))
            shapes = np.zeros((sg.size, n_modes), dtype=np.float64)
            for k in range(nm):
                lam = float(evals[k])
                v = evecs[:, k]
                shapes[:, k] = C[:, active] @ (sw * v) / math.sqrt(max(lam, 1e-300))
                imax = int(np.argmax(np.abs(shapes[:, k])))
                if shapes[imax, k] < 0.0:
                    shapes[:, k] *= -1.0
            shapes[0, :] = 0.0
            slopes = np.gradient(shapes, sg, axis=0, edge_order=2)
            slopes[0, :] = 0.0
            curv = np.gradient(slopes, sg, axis=0, edge_order=2)
            frac = np.cumsum(evals[:n_modes]) / total

    cached = tuple(
        np.ascontiguousarray(x, dtype=np.float64)
        for x in (sg, shapes, slopes, curv, frac)
    )
    if len(_KL_CACHE) >= _KL_CACHE_MAX:
        _KL_CACHE.pop(next(iter(_KL_CACHE)))
    _KL_CACHE[key] = cached
    return cached


_FE_CHOLESKY_CACHE = {}
_FE_CHOLESKY_CACHE_MAX = 256



@njit(cache=True, inline="always")
def _interp1(x, grid, vals):
    n = grid.size
    if n == 0:
        return 0.0
    if x <= grid[0]:
        return vals[0]
    if x >= grid[n - 1]:
        return vals[n - 1]
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if grid[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    x0 = grid[lo - 1]
    x1 = grid[lo]
    y0 = vals[lo - 1]
    y1 = vals[lo]
    if x1 <= x0:
        return y0
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


@njit(cache=True, inline="always")
def _dedx_and_slope(E, Eg, Sg):
    n = Eg.size
    if n < 2:
        return (Sg[0] if n else 0.0), 0.0
    if E <= Eg[0]:
        return Sg[0], 0.0
    if E >= Eg[n - 1]:
        return Sg[n - 1], 0.0
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if Eg[mid] < E:
            lo = mid + 1
        else:
            hi = mid
    x0 = Eg[lo - 1]
    x1 = Eg[lo]
    y0 = Sg[lo - 1]
    y1 = Sg[lo]
    if x1 <= x0:
        return y0, 0.0
    slope = (y1 - y0) / (x1 - x0)
    return y0 + (E - x0) * slope, slope


@njit(cache=True, inline="always")
def _disk_cdf(z):
    if z <= -1.0:
        return 0.0
    if z >= 1.0:
        return 1.0
    root = math.sqrt(max(0.0, 1.0 - z * z))
    return 0.5 + (math.asin(z) + z * root) / math.pi


@njit(cache=True, inline="always")
def _disk_pdf(z):
    if z <= -1.0 or z >= 1.0:
        return 0.0
    return 2.0 * math.sqrt(max(0.0, 1.0 - z * z)) / math.pi


@njit(cache=True, inline="always")
def _endpoint_weight_derivative(s, L, h, scope):
    if h <= 1e-12:
        inside = (s >= 0.0) if scope == 0 else ((s <= L) if scope == 1 else (s >= 0.0 and s <= L))
        return (1.0 if inside else 0.0), 0.0
    if scope == 0:
        zlo = -s / h
        w = 1.0 - _disk_cdf(zlo)
        dwds = _disk_pdf(zlo) / h
    elif scope == 1:
        zhi = (L - s) / h
        w = _disk_cdf(zhi)
        dwds = -_disk_pdf(zhi) / h
    else:
        zlo = -s / h
        zhi = (L - s) / h
        w = _disk_cdf(zhi) - _disk_cdf(zlo)
        dwds = (_disk_pdf(zlo) - _disk_pdf(zhi)) / h
    if w < 0.0:
        w = 0.0
    return w, dwds


@njit(cache=True, inline="always")
def _power_and_derivative(x):
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    n = 3.0777000000000001
    b = 0.79428866592713121
    y0 = 0.1209
    dy = 1.6396999999999999 - y0
    norm = 1.002379253316015
    xn = x ** n
    den = xn + b
    y = (y0 + dy * xn / den) / norm
    if x <= 0.0 or x >= 1.0:
        yp = 0.0
    else:
        yp = (dy / norm) * n * (x ** (n - 1.0)) * b / (den * den)
    return y, yp


@njit(cache=True, inline="always")
def _rel_eff_and_derivative(cost, code, table):
    if code < 0 or code >= table.shape[0] or table.shape[1] < 2:
        return 1.0, 0.0
    n = table.shape[1]
    x = cost
    if x <= 0.0:
        return table[code, 0], 0.0
    if x >= 1.0:
        return table[code, n - 1], 0.0
    f = x * (n - 1)
    i0 = int(math.floor(f))
    if i0 > n - 2:
        i0 = n - 2
    t = f - i0
    y0 = table[code, i0]
    y1 = table[code, i0 + 1]
    return y0 + t * (y1 - y0), (y1 - y0) * (n - 1)


@njit(cache=True, inline="always")
def _visibility_and_derivative(cost, width, centered):
    if width <= 0.0:
        return (1.0, 0.0) if cost > 0.0 else (0.0, 0.0)
    if centered != 0:
        if cost <= -width:
            return 0.0, 0.0
        if cost >= width:
            return 1.0, 0.0
        u = (cost + width) / (2.0 * width)
        return u * u * (3.0 - 2.0 * u), 3.0 * u * (1.0 - u) / width
    if cost <= 0.0:
        return 0.0, 0.0
    if cost >= width:
        return 1.0, 0.0
    u = cost / width
    return u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u) / width


@njit(cache=True, inline="always")
def _kinematics(E, mass, nwater):
    gamma = 1.0 + E / mass
    beta2 = 1.0 - 1.0 / max(gamma * gamma, 1e-30)
    if beta2 <= 0.0:
        return gamma, 0.0, 1.0, 0.0
    beta = math.sqrt(beta2)
    cth = 1.0 / (nwater * beta)
    if cth > 1.0:
        cth = 1.0
    elif cth < -1.0:
        cth = -1.0
    sin2 = max(1.0 - cth * cth, 0.0)
    return gamma, beta, cth, sin2


@njit(cache=True, inline="always")
def _f_at_s(s, px, py, pz, sx, sy, sz, dx, dy, dz, range_stop, master_r, master_k, mass, nwater):
    ex = sx + s * dx
    ey = sy + s * dy
    ez = sz + s * dz
    rx = px - ex
    ry = py - ey
    rz = pz - ez
    rr = math.sqrt(max(rx * rx + ry * ry + rz * rz, 1e-30))
    c = (dx * rx + dy * ry + dz * rz) / rr
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    alpha = math.acos(c)
    rem = range_stop - s
    if rem < 0.0:
        rem = 0.0
    E = _interp1(rem, master_r, master_k)
    gamma, beta, cth, sin2 = _kinematics(E, mass, nwater)
    theta = math.acos(cth)
    return alpha - theta


@njit(cache=True, fastmath=True)
def primary_charge_jacobian(
    p_locations, direction_zs, mu_primary, raw_s, scale_in, s_b, E_b,
    start, d0, e1, e2, basis_s, shapes, slopes, curvatures,
    mpmt_codes, rel_table, dedx_E, dedx_S, master_r, master_k,
    range_stop, track_length, nwater, mass, ngeo_a,
    endpoint_mode, endpoint_scope, endpoint_radius, legacy_pmt_radius,
    base_sigma, cost_soft, cost_centered,
):
    npmts = p_locations.shape[0]
    nm = shapes.shape[1]
    npar = 2 * nm
    J = np.zeros((npmts, npar), dtype=np.float64)
    ftinf = 1.0 - 1.0 / (nwater * nwater)
    sa = 0.001
    for i in range(npmts):
        mu = mu_primary[i]
        if mu <= 0.0 or (not math.isfinite(mu)) or scale_in[i] <= 0.0:
            continue

        px, py, pz = p_locations[i, 0], p_locations[i, 1], p_locations[i, 2]
        f_lo = _f_at_s(sa, px, py, pz, start[0], start[1], start[2], d0[0], d0[1], d0[2], range_stop, master_r, master_k, mass, nwater)
        f_hi = _f_at_s(track_length, px, py, pz, start[0], start[1], start[2], d0[0], d0[1], d0[2], range_stop, master_r, master_k, mass, nwater)
        crossing = (f_lo < 0.0) and (f_hi >= 0.0)
        s_center = s_b[i] if scale_in[i] > 0.0 else raw_s[i]

        # Endpoint overlap base state.  h is held fixed in the first analytic
        # implementation; its s-dependence is included and its parameter
        # derivative is a higher-order aperture correction to be validated.
        wend = 1.0
        dwds = 0.0
        s_eff = s_center
        if endpoint_mode > 0:
            yx = px - start[0]
            yy = py - start[1]
            yz = pz - start[2]
            uline = yx * d0[0] + yy * d0[1] + yz * d0[2]
            bx = yx - uline * d0[0]
            by = yy - uline * d0[1]
            bz = yz - uline * d0[2]
            rho = math.sqrt(max(bx * bx + by * by + bz * bz, 0.0))
            Eep = E_b[i]
            gam, beta, cth, sin2 = _kinematics(Eep, mass, nwater)
            if beta > 0.0 and sin2 > 1e-18:
                sth = math.sqrt(sin2)
                cot = cth / sth
                S, Sp = _dedx_and_slope(Eep, dedx_E, dedx_S)
                a = 1.0 / (nwater * mass * beta ** 3 * gam ** 3)
                dc_ds = S * a
                dcot_ds = dc_ds / (sin2 * sth)
            else:
                cot = 0.0
                dcot_ds = 0.0
            if rho > 1e-12:
                denroot = 1.0 + rho * dcot_ds
                if denroot <= 1e-12 or not math.isfinite(denroot):
                    denroot = 1.0
                gx = (d0[0] - cot * bx / rho) / denroot
                gy = (d0[1] - cot * by / rho) / denroot
                gz = (d0[2] - cot * bz / rho) / denroot
            else:
                gx, gy, gz = d0[0], d0[1], d0[2]
            nx, ny, nz = direction_zs[i, 0], direction_zs[i, 1], direction_zs[i, 2]
            n2 = nx * nx + ny * ny + nz * nz
            if n2 > 1e-18:
                gd = (gx * nx + gy * ny + gz * nz) / n2
                gx -= gd * nx
                gy -= gd * ny
                gz -= gd * nz
            h = endpoint_radius * math.sqrt(gx * gx + gy * gy + gz * gz)
            wend, dwds = _endpoint_weight_derivative(s_center, track_length, h, endpoint_scope)
            if wend <= 0.0:
                continue
            if endpoint_mode == 1:
                if s_eff < 0.0:
                    s_eff = 0.0
                elif s_eff > track_length:
                    s_eff = track_length
            # endpoint_mode==2 is diagnostic and not used in production; its
            # mean-coordinate derivative is not implemented in this prototype.
        else:
            if s_eff < legacy_pmt_radius:
                front = (s_eff + legacy_pmt_radius) / (2.0 * legacy_pmt_radius)
                if front <= 0.0:
                    continue
            if s_eff < -legacy_pmt_radius:
                continue

        ex = start[0] + s_eff * d0[0]
        ey = start[1] + s_eff * d0[1]
        ez = start[2] + s_eff * d0[2]
        Rx = px - ex
        Ry = py - ey
        Rz = pz - ez
        rg = math.sqrt(max(Rx * Rx + Ry * Ry + Rz * Rz, 1e-30))
        nxr, nyr, nzr = Rx / rg, Ry / rg, Rz / rg
        r = rg + 0.01
        cost = -(Rx * direction_zs[i, 0] + Ry * direction_zs[i, 1] + Rz * direction_zs[i, 2]) / r
        pwr, dpwr = _power_and_derivative(cost)
        rel, drel = _rel_eff_and_derivative(cost, int(mpmt_codes[i]), rel_table)
        vis, dvis = _visibility_and_derivative(cost, cost_soft, cost_centered)
        if pwr <= 0.0 or rel <= 0.0 or vis <= 0.0:
            continue

        E = E_b[i]
        gam, beta, cth, sin2 = _kinematics(E, mass, nwater)
        if beta <= 0.0 or nwater * beta <= 1.0 or sin2 <= 0.0:
            continue
        S, Sp = _dedx_and_slope(E, dedx_E, dedx_S)
        a = 1.0 / (nwater * mass * beta ** 3 * gam ** 3)
        qdc = S * a
        reff = math.sqrt(r * r + ngeo_a * ngeo_a)
        D = reff * sin2 + reff * reff * qdc
        if D <= 0.0 or not math.isfinite(D):
            continue
        ft = 1.0 - 1.0 / (nwater * nwater * beta * beta)
        if ft <= 0.0:
            continue
        dftdE = 2.0 / (nwater * nwater * mass * beta ** 4 * gam ** 3)
        dsin2dE = dftdE
        dln_a_dE = -3.0 / (mass * beta * beta * gam ** 3) - 3.0 / (mass * gam)
        dqdc_dE = Sp * a + qdc * dln_a_dE

        # Determine the endpoint used by the no-crossing Gaussian tail.
        if not crossing:
            if abs(f_lo) <= abs(f_hi):
                sf = sa
                fmin = f_lo
            else:
                sf = track_length
                fmin = f_hi
        else:
            sf = s_center
            fmin = 0.0

        for plane in range(2):
            eb = e1 if plane == 0 else e2
            for k in range(nm):
                col = plane * nm + k
                phi_root = _interp1(sf, basis_s, shapes[:, k])
                psi_root = _interp1(sf, basis_s, slopes[:, k])
                # f_u at the physical root or selected no-crossing endpoint.
                exf = start[0] + sf * d0[0]
                eyf = start[1] + sf * d0[1]
                ezf = start[2] + sf * d0[2]
                rx = px - exf
                ry = py - eyf
                rz = pz - ezf
                rrf = math.sqrt(max(rx * rx + ry * ry + rz * rz, 1e-30))
                ux, uy, uz = rx / rrf, ry / rrf, rz / rrf
                c = d0[0] * ux + d0[1] * uy + d0[2] * uz
                if c > 1.0:
                    c = 1.0
                elif c < -1.0:
                    c = -1.0
                sina = math.sqrt(max(1.0 - c * c, 1e-18))
                en = eb[0] * ux + eb[1] * uy + eb[2] * uz
                fu = -en * (psi_root + c * phi_root / rrf) / sina
                if crossing:
                    Sroot, _ = _dedx_and_slope(E_b[i], dedx_E, dedx_S)
                    gamr, betar, cthr, sin2r = _kinematics(E_b[i], mass, nwater)
                    sth = math.sqrt(max(sin2r, 1e-18))
                    dc_ds = Sroot / (nwater * mass * betar ** 3 * gamr ** 3)
                    fs = sina / rrf + dc_ds / sth
                    dsdu = -fu / fs if abs(fs) > 1e-12 else 0.0
                    dln_scale = (dftdE / ft) * (-S * dsdu)
                else:
                    dsdu = 0.0
                    dln_scale = -(fmin / max(base_sigma * base_sigma, 1e-30)) * fu if base_sigma > 0.0 else 0.0

                dln_endpoint = (dwds / wend) * dsdu if endpoint_mode > 0 and wend > 0.0 else 0.0
                ds_eff = dsdu if crossing and (s_center > 0.0 and s_center < track_length) else 0.0
                phi = _interp1(s_eff, basis_s, shapes[:, k])
                chi = _interp1(s_eff, basis_s, curvatures[:, k])
                dXx = d0[0] * ds_eff + eb[0] * phi
                dXy = d0[1] * ds_eff + eb[1] * phi
                dXz = d0[2] * ds_eff + eb[2] * phi
                dRx, dRy, dRz = -dXx, -dXy, -dXz
                dr = nxr * dRx + nyr * dRy + nzr * dRz
                qpn = direction_zs[i, 0] * nxr + direction_zs[i, 1] * nyr + direction_zs[i, 2] * nzr
                dcost = -(
                    direction_zs[i, 0] * dRx + direction_zs[i, 1] * dRy + direction_zs[i, 2] * dRz
                    - qpn * dr
                ) / r
                dE = -S * ds_eff
                dreff = (r / reff) * dr
                en_here = eb[0] * nxr + eb[1] * nyr + eb[2] * nzr
                dD = (
                    dreff * sin2
                    + reff * dsin2dE * dE
                    + 2.0 * reff * dreff * qdc
                    + reff * reff * dqdc_dE * dE
                    - reff * reff * chi * en_here
                )
                dln_ngeo = -dD / D
                dln_pwr = (dpwr / pwr) * dcost
                dln_rel = (drel / rel) * dcost
                dln_vis = (dvis / vis) * dcost
                J[i, col] = mu * (dln_scale + dln_endpoint + dln_ngeo + dln_pwr + dln_rel + dln_vis)
    return J


def normalized_charge_jacobian(raw_mu, raw_J, norm, floor_pe=0.0):
    raw_mu = np.asarray(raw_mu, dtype=np.float64)
    raw_J = np.asarray(raw_J, dtype=np.float64)
    total = float(np.sum(raw_mu))
    if total <= 0.0:
        return np.zeros_like(raw_J)
    dsum = np.sum(raw_J, axis=0)
    J = float(norm) * (raw_J - raw_mu[:, None] * (dsum / total)[None, :])
    if floor_pe > 0.0:
        physical = raw_mu * float(norm)
        J[physical <= floor_pe, :] = 0.0
    return np.ascontiguousarray(J, dtype=np.float64)



def woodbury_apply(diag_inv, process_jacobian, matrix):
    """Apply (D + J J^T)^-1 without forming the PMT covariance matrix.

    Parameters
    ----------
    diag_inv : (n_pmt,) array
        Diagonal inverse D^-1. For the Poisson working covariance,
        D_ii = max(mu_i, floor).
    process_jacobian : (n_pmt, n_mode) array
        Derivative of expected charge with respect to standardized FE modes.
    matrix : (n_pmt,) or (n_pmt, n_col) array
        Vector or matrix to which the inverse covariance is applied.
    """
    di = np.asarray(diag_inv, dtype=np.float64)
    J = np.asarray(process_jacobian, dtype=np.float64)
    X = np.asarray(matrix, dtype=np.float64)
    one_dimensional = X.ndim == 1
    if one_dimensional:
        X = X[:, None]
    if J.ndim != 2 or J.shape[0] != di.size or X.shape[0] != di.size:
        raise ValueError("Incompatible dimensions in woodbury_apply")
    if J.shape[1] == 0:
        out = di[:, None] * X
        return out[:, 0] if one_dimensional else out
    weighted_J = di[:, None] * J
    H = np.eye(J.shape[1], dtype=np.float64) + J.T @ weighted_J
    try:
        chol = np.linalg.cholesky(0.5 * (H + H.T))
        rhs = J.T @ (di[:, None] * X)
        tmp = np.linalg.solve(chol, rhs)
        coeff = np.linalg.solve(chol.T, tmp)
    except np.linalg.LinAlgError:
        coeff = np.linalg.pinv(H, rcond=1.0e-12) @ (J.T @ (di[:, None] * X))
    out = di[:, None] * X - weighted_J @ coeff
    return out[:, 0] if one_dimensional else out


def _scaled_psd_pseudoinverse(matrix, rcond=1.0e-10):
    """Pseudoinvert a symmetric positive-semidefinite information matrix.

    Track parameters mix millimetres and dimensionless direction components, so
    the raw information matrix can have a very large condition number for purely
    dimensional reasons.  We therefore normalize by the square root of the
    diagonal information, pseudoinvert the resulting dimensionless correlation
    matrix, and transform back.  This is algebraically identical to the ordinary
    inverse when the matrix is full rank; it only makes the rank decision
    unit-stable when a direction is weakly constrained.
    """
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("matrix must be square")
    if M.size == 0:
        return M.copy(), np.empty(0, dtype=np.float64), np.nan, np.nan

    M = 0.5 * (M + M.T)
    diagonal = np.maximum(np.diag(M), 0.0)
    scale = np.sqrt(diagonal)
    safe = np.where(scale > 1.0e-150, scale, 1.0)
    R = M / safe[:, None] / safe[None, :]
    inactive = scale <= 1.0e-150
    if np.any(inactive):
        R[inactive, :] = 0.0
        R[:, inactive] = 0.0
    R = 0.5 * (R + R.T)

    eigenvalues, eigenvectors = np.linalg.eigh(R)
    largest = max(float(np.max(eigenvalues)), 0.0)
    cutoff = float(rcond) * max(largest, 1.0)
    keep = eigenvalues > cutoff
    inverse_eigenvalues = np.zeros_like(eigenvalues)
    inverse_eigenvalues[keep] = 1.0 / eigenvalues[keep]
    Rinv = eigenvectors @ (inverse_eigenvalues[:, None] * eigenvectors.T)
    pinv = Rinv / safe[:, None] / safe[None, :]
    if np.any(inactive):
        pinv[inactive, :] = 0.0
        pinv[:, inactive] = 0.0
    pinv = 0.5 * (pinv + pinv.T)

    try:
        condition_raw = float(np.linalg.cond(M))
    except Exception:
        condition_raw = np.inf
    if np.any(keep):
        condition_scaled = float(largest / np.min(eigenvalues[keep]))
    else:
        condition_scaled = np.inf
    return pinv, scale, condition_raw, condition_scaled


def fermi_eyges_process_update(
    expected_charge,
    observed_charge,
    track_jacobian,
    process_jacobian,
    *,
    update_indices=None,
    charge_floor_pe=1.0e-4,
    rcond=1.0e-10,
):
    """Compute a one-step FE-GEE update and robust covariance.

    The starting track estimate is assumed to come from the ordinary sharp
    charge likelihood.  The process working covariance is

        V = D + J_u J_u^T,

    where D is the Poisson working covariance and J_u contains derivatives with
    respect to standardized Fermi--Eyges trajectory modes.  ``update_indices``
    selects any subset of the supplied track coordinates; ``None`` updates every
    coordinate.  The full six-coordinate driver therefore updates
    ``(x0, y0, z0, cx, cy, length)`` in one coupled solve while retaining the
    legacy longitudinal-only block as an explicit option.

    No empirical MCS scale or damping factor enters this calculation.
    """
    mu = np.asarray(expected_charge, dtype=np.float64)
    q = np.asarray(observed_charge, dtype=np.float64)
    Jtheta = np.asarray(track_jacobian, dtype=np.float64)
    Ju = np.asarray(process_jacobian, dtype=np.float64)
    if mu.ndim != 1 or q.shape != mu.shape:
        raise ValueError("expected_charge and observed_charge must be matching 1-D arrays")
    if Jtheta.ndim != 2 or Jtheta.shape[0] != mu.size:
        raise ValueError("track_jacobian has incompatible shape")
    if Ju.ndim != 2 or Ju.shape[0] != mu.size:
        raise ValueError("process_jacobian has incompatible shape")

    good = np.isfinite(mu) & np.isfinite(q)
    if not np.all(good):
        mu = mu[good]
        q = q[good]
        Jtheta = Jtheta[good]
        Ju = Ju[good]
    D = np.maximum(mu, float(charge_floor_pe))
    di = 1.0 / D
    residual = q - mu

    Vinv_residual = woodbury_apply(di, Ju, residual)
    Vinv_Jtheta = woodbury_apply(di, Ju, Jtheta)

    if update_indices is None:
        update_indices = tuple(range(Jtheta.shape[1]))
    else:
        update_indices = tuple(int(i) for i in update_indices)
    if len(set(update_indices)) != len(update_indices):
        raise ValueError("update_indices contains duplicates")
    if any(i < 0 or i >= Jtheta.shape[1] for i in update_indices):
        raise IndexError("update_indices is outside the track_jacobian columns")

    Jblock = Jtheta[:, update_indices]
    Vinv_Jblock = Vinv_Jtheta[:, update_indices]
    information_block = Jblock.T @ Vinv_Jblock
    score_block = Jblock.T @ Vinv_residual
    information_inverse, information_scale, condition_raw, condition_scaled = (
        _scaled_psd_pseudoinverse(information_block, rcond=float(rcond))
    )
    delta_block = information_inverse @ score_block

    # Naive and FE-robust sandwich covariance of every supplied sharp-track
    # coordinate.  The same diagonal-information normalization makes this
    # numerically stable for mixed mm/direction units.
    A = Jtheta.T @ (di[:, None] * Jtheta)
    C = Jtheta.T @ (di[:, None] * Ju)
    Ainv, A_scale, A_condition_raw, A_condition_scaled = (
        _scaled_psd_pseudoinverse(A, rcond=float(rcond))
    )
    naive_covariance = 0.5 * (Ainv + Ainv.T)
    robust_covariance = Ainv @ (A + C @ C.T) @ Ainv
    robust_covariance = 0.5 * (robust_covariance + robust_covariance.T)

    H = np.eye(Ju.shape[1], dtype=np.float64) + Ju.T @ (di[:, None] * Ju)
    try:
        process_posterior_covariance = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        process_posterior_covariance = np.linalg.pinv(H, rcond=float(rcond))
    process_posterior_mean = process_posterior_covariance @ (Ju.T @ (di * residual))

    return {
        "update_indices": update_indices,
        "delta_block": np.asarray(delta_block, dtype=np.float64),
        "information_block": np.asarray(information_block, dtype=np.float64),
        "score_block": np.asarray(score_block, dtype=np.float64),
        "information_parameter_scale": np.asarray(information_scale, dtype=np.float64),
        "information_condition_raw": condition_raw,
        "information_condition_scaled": condition_scaled,
        "naive_information_parameter_scale": np.asarray(A_scale, dtype=np.float64),
        "naive_information_condition_raw": A_condition_raw,
        "naive_information_condition_scaled": A_condition_scaled,
        "naive_covariance": np.asarray(naive_covariance, dtype=np.float64),
        "robust_covariance": np.asarray(robust_covariance, dtype=np.float64),
        "process_posterior_mean": np.asarray(process_posterior_mean, dtype=np.float64),
        "process_posterior_covariance": np.asarray(process_posterior_covariance, dtype=np.float64),
        "process_mode_count": int(Ju.shape[1]),
    }

