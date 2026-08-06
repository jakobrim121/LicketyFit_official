import math

import numpy as np
from typing import List, Tuple
from numba import njit

try:
    from .particle_cherenkov_model import (
        find_scale_for_pmts,
        get_cerenkov_angle_table,
        get_energy_distance_tables,
        theta_c_func,
        get_rel_mpmt_eff_tables,
        set_active_particle,
        canonical_particle_name,
        particle_mass_mev,
        cherenkov_threshold_kinetic_mev,
    )
    from .n_model_wrapper import *
except ImportError:
    # Support the historical usage pattern where LicketyFit/ itself is placed
    # directly on sys.path and modules are imported as top-level files.
    from particle_cherenkov_model import (
        find_scale_for_pmts,
        get_cerenkov_angle_table,
        get_energy_distance_tables,
        theta_c_func,
        get_rel_mpmt_eff_tables,
        set_active_particle,
        canonical_particle_name,
        particle_mass_mev,
        cherenkov_threshold_kinetic_mev,
    )
    from n_model_wrapper import *


_TABLE_CACHE = {}

def _get_tables(particle=None):
    """Load and cache lookup tables once per Python process and particle."""
    pname = canonical_particle_name(particle)
    cached = _TABLE_CACHE.get(pname)
    if cached is None:
        c_ang, energy_for_angle = get_cerenkov_angle_table(pname)
        overall_distances, energy_rows, distance_rows = get_energy_distance_tables(pname)
        tri_exsitu, tri_insitu, wut_insitu, wut_exsitu = get_rel_mpmt_eff_tables()
        cached = (
            c_ang, energy_for_angle, overall_distances, energy_rows, distance_rows,
            tri_exsitu, tri_insitu, wut_insitu, wut_exsitu,
        )
        _TABLE_CACHE[pname] = cached
    return cached


# -----------------------------------------------------------------------------
# Hot-loop helper caches
# -----------------------------------------------------------------------------
# These are intentionally module-level.  In a Minuit fit the Emitter may be
# constructed many times, but the detector geometry tables and mPMT response
# tables are fixed.  Caching here avoids repeated string handling, table stacking,
# and scalar normalization work in every FCN call.

_MPMT_TYPE_TO_CODE = {
    "tri_exsitu": 0,
    "tri_insitu": 1,
    "wut_exsitu": 2,
    "wut_insitu": 3,
}

# Relative mPMT efficiency mode.
#   "type": historical 4-curve per-mPMT-type model (tri/wut x in/ex-situ).
#   "slot": per-slot data/MC ratio curves, one row per mPMT slot.
# Default is "type" so all existing behavior (and WCSim, which passes no mPMT
# types) is bit-identical unless a caller explicitly opts into slot mode via
# set_rel_eff_mode("slot", slot_stack=...).
_REL_EFF_MODE = "type"
_REL_EFF_STACK_CACHE = None
_REL_EFF_SLOT_STACK_CACHE = None
_PRIMARY_NGEO_NORM_CACHE = {}
_PARTICLE_STOPPING_POWER_CACHE = {}
_PMT_RADIUS_CACHE = {}
_RANGE_FROM_ENERGY_CACHE = {}


def _get_range_from_energy_arrays(particle):
    """Return cached, ascending (initial_energies, overall_distances) arrays.

    Depends only on the particle's fixed range table, so it is computed once
    per particle.  Used by the absorption-mode FCN hot path.
    """
    pname = canonical_particle_name(particle)
    cached = _RANGE_FROM_ENERGY_CACHE.get(pname)
    if cached is None:
        tables = _get_tables(pname)
        overall_distances = np.asarray(tables[2], dtype=np.float64)
        energy_rows = tables[3]
        initial_energies = np.asarray(
            [float(row[0]) for row in energy_rows], dtype=np.float64
        )
        order = np.argsort(initial_energies)
        initial_energies = np.ascontiguousarray(initial_energies[order])
        overall_distances = np.ascontiguousarray(overall_distances[order])
        cached = (initial_energies, overall_distances)
        _RANGE_FROM_ENERGY_CACHE[pname] = cached
    return cached


def _get_type_rel_eff_stack():
    """
    Return relative mPMT efficiency curves in the code order

        0: tri_exsitu
        1: tri_insitu
        2: wut_exsitu
        3: wut_insitu

    The raw table order returned by get_rel_mpmt_eff_tables() is
    tri_exsitu, tri_insitu, wut_insitu, wut_exsitu, so wut entries are swapped
    here to match the string labels used throughout the Emitter.
    """
    global _REL_EFF_STACK_CACHE
    if _REL_EFF_STACK_CACHE is None:
        tables = _get_tables()
        tri_exsitu = np.asarray(tables[5], dtype=np.float64)
        tri_insitu = np.asarray(tables[6], dtype=np.float64)
        wut_insitu = np.asarray(tables[7], dtype=np.float64)
        wut_exsitu = np.asarray(tables[8], dtype=np.float64)
        _REL_EFF_STACK_CACHE = np.ascontiguousarray(
            np.vstack([tri_exsitu, tri_insitu, wut_exsitu, wut_insitu])
        )
    return _REL_EFF_STACK_CACHE


def _get_rel_eff_stack():
    """
    Return the active relative mPMT efficiency stack for the current mode.

    type mode -> 4 per-mPMT-type rows; slot mode -> per-slot rows.  In both
    cases the per-PMT integer codes passed to the kernels index rows of this
    stack, so the hot kernels are unchanged: only the table contents and the
    meaning of the codes differ between modes.
    """
    if _REL_EFF_MODE == "slot":
        if _REL_EFF_SLOT_STACK_CACHE is None:
            raise RuntimeError(
                "rel-eff mode is 'slot' but no slot stack has been set; "
                "call set_rel_eff_mode('slot', slot_stack=...) first"
            )
        return _REL_EFF_SLOT_STACK_CACHE
    return _get_type_rel_eff_stack()


def set_rel_eff_mode(mode, slot_stack=None):
    """Select the relative mPMT efficiency model used by the hot kernels.

    mode="type" (default): historical 4-curve per-mPMT-type model.
    mode="slot": per-slot data/MC ratio curves.  slot_stack must be a
        (n_slots, n_grid) float64 array, one relative-efficiency curve per mPMT
        slot, sampled on the uniform cos-incidence grid linspace(0, 1, n_grid).
        The per-PMT "codes" handed to the kernels are then slot numbers.

    Slot codes are carried in int8 by the callers, so at most 128 slots
    (0..127) are supported in slot mode.
    """
    global _REL_EFF_MODE, _REL_EFF_SLOT_STACK_CACHE
    mode = str(mode).strip().lower()
    if mode not in ("type", "slot"):
        raise ValueError("rel-eff mode must be 'type' or 'slot' (got %r)" % (mode,))

    if mode == "slot":
        if slot_stack is None:
            raise ValueError("slot mode requires slot_stack=(n_slots, n_grid) array")
        slot_stack = np.ascontiguousarray(np.asarray(slot_stack, dtype=np.float64))
        if slot_stack.ndim != 2 or slot_stack.shape[0] < 1 or slot_stack.shape[1] < 2:
            raise ValueError("slot_stack must have shape (n_slots>=1, n_grid>=2)")
        if slot_stack.shape[0] > 127:
            raise ValueError(
                "slot mode supports at most 128 slots because codes are int8 "
                "(got %d rows)" % (slot_stack.shape[0],)
            )
        _REL_EFF_SLOT_STACK_CACHE = slot_stack

    _REL_EFF_MODE = mode
    return _REL_EFF_MODE


def reset_rel_eff_mode():
    """Reset to the historical per-mPMT-type relative-efficiency model."""
    global _REL_EFF_MODE, _REL_EFF_SLOT_STACK_CACHE
    _REL_EFF_MODE = "type"
    _REL_EFF_SLOT_STACK_CACHE = None
    return _REL_EFF_MODE


def get_rel_eff_mode():
    """Return the active relative mPMT efficiency mode ('type' or 'slot')."""
    return _REL_EFF_MODE


def _encode_mpmt_types(mpmt_types):
    """
    Convert mPMT type strings to small integer codes once.

    Unknown/empty types get code -1 and are treated as fill_empty in the
    interpolation helper.  This replaces repeated string masks in the fit loop.
    """
    arr = np.asarray(mpmt_types)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int8, copy=False)

    codes = np.full(arr.shape, -1, dtype=np.int8)
    for typ, code in _MPMT_TYPE_TO_CODE.items():
        codes[arr == typ] = code
    return codes


def _interp_rel_mpmt_eff_from_codes(cost, mpmt_type_codes, fill_empty=1.0):
    """
    Fast relative mPMT efficiency interpolation on the fixed uniform cost grid.

    This is equivalent to np.interp(cost, linspace(0,1,N), yvals,
    left=yvals[0], right=yvals[-1]) for each mPMT type, but avoids building
    four boolean string masks and avoids np.tile() for the secondary-electron
    source grid.
    """
    cost = np.asarray(cost, dtype=np.float64)
    codes = np.asarray(mpmt_type_codes)

    # Broadcast PMT codes over a source x PMT cost grid without allocating a
    # tiled string array.  For 1D cost this is a no-op.
    if cost.ndim == 2 and codes.ndim == 1:
        codes = np.broadcast_to(codes[None, :], cost.shape)
    else:
        codes = np.broadcast_to(codes, cost.shape)

    table = _get_rel_eff_stack()
    n_rows = table.shape[0]
    n_grid = table.shape[1]

    out = np.full(cost.shape, fill_empty, dtype=np.float64)
    valid = np.isfinite(cost) & (codes >= 0) & (codes < n_rows)
    if not np.any(valid):
        return out

    # np.interp with x-grid linspace(0,1,N) is just linear interpolation in
    # fractional index space.  Clipping reproduces left/right edge behavior.
    x = np.clip(cost[valid], 0.0, 1.0) * (n_grid - 1)
    i0 = np.floor(x).astype(np.int64)
    i0 = np.clip(i0, 0, n_grid - 2)
    t = x - i0

    c = codes[valid].astype(np.int64, copy=False)
    y0 = table[c, i0]
    y1 = table[c, i0 + 1]
    out[valid] = y0 + t * (y1 - y0)
    return out


# -----------------------------------------------------------------------------
# Numba-compiled hot-path helpers
# -----------------------------------------------------------------------------
# These helpers are deliberately standalone rather than methods so that Numba can
# compile the source x PMT loops.  They keep the same algebra as the vectorized
# Python implementation, but avoid allocating large intermediate matrices such as
# dx, dy, dz, r, cost, optical_corr, forward_kernel, and delta_contrib.

@njit(cache=True)
def _power_law_scalar_numba(x):
    if x < 0.0:
        x = 0.0
    y0_fit = 0.1209
    yinf = 1.6397
    x50 = 0.9279
    n_fit = 3.0777
    max_ = 0.967354918872639
    xn = x ** n_fit
    # x50n = x50 ** n_fit is loop-invariant; fold to the exact literal so the
    # ~20k-per-FCN inner-loop calls don't recompute pow(0.9279, 3.0777).
    x50n = 0.7942886659271312
    return (y0_fit + (yinf - y0_fit) * (xn / (xn + x50n))) / max_


@njit(cache=True)
def _power_law_lut_scalar_numba(x, lut):
    """Linear-interpolated power_law via a precomputed LUT on x in [0, 1].

    Replaces the per-(PMT,source) pow(x, 3.0777) in the delta accumulator with a
    clamped table lookup.  Accuracy vs the exact form (4096-pt LUT): max rel err
    ~1.3e-7 -- far below the model's numerical noise.  Caller passes lut.size==0
    to fall back to the exact scalar.
    """
    n = lut.shape[0]
    if n < 2:
        return _power_law_scalar_numba(x)
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    f = x * (n - 1)
    i0 = int(math.floor(f))
    if i0 > n - 2:
        i0 = n - 2
    t = f - i0
    return lut[i0] + t * (lut[i0 + 1] - lut[i0])


def _build_power_law_lut(n=4096):
    """Build the power_law LUT once (module-level cache)."""
    x = np.linspace(0.0, 1.0, n)
    xn = np.power(x, 3.0777)
    x50n = 0.7942886659271312
    y = (0.1209 + (1.6397 - 0.1209) * (xn / (xn + x50n))) / 0.967354918872639
    return np.ascontiguousarray(y, dtype=np.float64)


# Module-level LUT, built once.  Empty array disables the LUT path.
_POWER_LAW_LUT = _build_power_law_lut(4096)
_POWER_LAW_LUT_EMPTY = np.empty(0, dtype=np.float64)


@njit(cache=True)
def _rel_mpmt_eff_scalar_numba(cost, code, table):
    if code < 0 or code >= table.shape[0]:
        return 1.0
    n_grid = table.shape[1]
    if n_grid < 2:
        return 1.0
    x = cost
    if x < 0.0:
        x = 0.0
    elif x > 1.0:
        x = 1.0
    x *= (n_grid - 1)
    i0 = int(math.floor(x))
    if i0 < 0:
        i0 = 0
    elif i0 > n_grid - 2:
        i0 = n_grid - 2
    t = x - i0
    y0 = table[code, i0]
    y1 = table[code, i0 + 1]
    return y0 + t * (y1 - y0)


@njit(cache=True)
def _finite_disk_rel_scalar_numba(r, pmt_radius_mm, ref_r_mm):
    r_safe = r
    if r_safe < 1e-9:
        r_safe = 1e-9
    a = pmt_radius_mm
    R0 = ref_r_mm
    if a <= 0.0:
        return (R0 / r_safe) * (R0 / r_safe)
    omega_shape = 1.0 - r_safe / math.sqrt(r_safe * r_safe + a * a)
    omega_ref = 1.0 - R0 / math.sqrt(R0 * R0 + a * a)
    if (not math.isfinite(omega_ref)) or omega_ref <= 0.0:
        return (R0 / r_safe) * (R0 / r_safe)
    out = omega_shape / omega_ref
    if (not math.isfinite(out)) or out < 0.0:
        return 0.0
    return out


@njit(cache=True)
def _refined_delta_dSdu_scalar_numba(K, u, K_grid, u_grid, table):
    """Scalar bilinear interpolation of dS_delta/du(K,u)."""
    if (not math.isfinite(K)) or (not math.isfinite(u)):
        return 0.0
    if K < K_grid[0] or u < -1.0 or u > 1.0:
        return 0.0

    # Clip high values exactly like the vectorized implementation.
    Kc = K
    if Kc > K_grid[K_grid.size - 1]:
        Kc = K_grid[K_grid.size - 1]
    uc = u
    if uc < u_grid[0]:
        uc = u_grid[0]
    elif uc > u_grid[u_grid.size - 1]:
        uc = u_grid[u_grid.size - 1]

    # Uniform grids in current table builder, so use direct index math.
    dK = K_grid[1] - K_grid[0]
    iK = int(math.floor((Kc - K_grid[0]) / dK))
    if iK < 0:
        iK = 0
    elif iK > K_grid.size - 2:
        iK = K_grid.size - 2
    K0 = K_grid[iK]
    K1 = K_grid[iK + 1]
    tK = (Kc - K0) / (K1 - K0 + 1e-300)
    if tK < 0.0:
        tK = 0.0
    elif tK > 1.0:
        tK = 1.0

    du = u_grid[1] - u_grid[0]
    iu = int(math.floor((uc - u_grid[0]) / du))
    if iu < 0:
        iu = 0
    elif iu > u_grid.size - 2:
        iu = u_grid.size - 2
    u0 = u_grid[iu]
    u1 = u_grid[iu + 1]
    tu = (uc - u0) / (u1 - u0 + 1e-300)
    if tu < 0.0:
        tu = 0.0
    elif tu > 1.0:
        tu = 1.0

    p00 = table[iK, iu]
    p01 = table[iK, iu + 1]
    p10 = table[iK + 1, iu]
    p11 = table[iK + 1, iu + 1]
    p0 = p00 + tu * (p01 - p00)
    p1 = p10 + tu * (p11 - p10)
    out = p0 + tK * (p1 - p0)
    if (not math.isfinite(out)) or out < 0.0:
        return 0.0
    return out


@njit(cache=True)
def _accumulate_refined_delta_numba(
    p_locations,
    direction_zs,
    start_pos,
    track_dir,
    s_centers,
    ds_cm,
    K_mu,
    K_grid,
    u_grid,
    table,
    mpmt_codes,
    rel_eff_table,
    apply_mpmt_eff,
    use_finite_disk,
    pmt_radius_mm,
    ref_r_mm,
    distance_power,
    analytic_delta_scale,
    source_k_power,
    source_k_ref,
    source_k_floor,
    intensity,
    starting_time,
    v,
    n_water,
    c_light,
    delta_e_time_offset_ns,
    return_times,
    pl_lut,
    cost_soft,
    src_lo,
    src_hi,
    use_seg_gate,
):
    """
    Fast PMT-parallel secondary-electron accumulator.

    This keeps the same physics as the original source x PMT loop, but removes
    repeated source-only work from the hot PMT loop:

      * source positions, source times, source weights are precomputed once;
      * K-grid interpolation indices/fractions are precomputed once per source;
      * each Numba thread accumulates one PMT, avoiding write conflicts;
      * impossible contributions are rejected before expensive optical factors.

    The table is still interpreted as dS_delta/du(K_mu, u), where u is the
    photon direction cosine relative to the primary muon direction.
    """
    n_src = s_centers.size
    n_pmts = p_locations.shape[0]

    mu = np.zeros(n_pmts, dtype=np.float64)
    tnum = np.zeros(n_pmts, dtype=np.float64)

    if n_src == 0 or n_pmts == 0:
        if return_times:
            t_empty = np.empty(n_pmts, dtype=np.float64)
            for i in range(n_pmts):
                t_empty[i] = np.nan
            return mu, t_empty
        return mu, tnum

    # ------------------------------------------------------------------
    # Precompute source-only quantities.
    # ------------------------------------------------------------------
    src_x = np.empty(n_src, dtype=np.float64)
    src_y = np.empty(n_src, dtype=np.float64)
    src_z = np.empty(n_src, dtype=np.float64)
    src_t = np.empty(n_src, dtype=np.float64)
    src_w = np.empty(n_src, dtype=np.float64)
    src_iK = np.empty(n_src, dtype=np.int64)
    src_tK = np.empty(n_src, dtype=np.float64)
    src_valid = np.zeros(n_src, dtype=np.uint8)

    K_min = K_grid[0]
    K_max = K_grid[K_grid.size - 1]
    dK = K_grid[1] - K_grid[0]
    inv_dK = 1.0 / dK
    nK = K_grid.size

    for j in range(n_src):
        K = K_mu[j]
        ds = ds_cm[j]

        if (not math.isfinite(K)) or K <= 0.0:
            continue
        if (not math.isfinite(ds)) or ds <= 0.0:
            continue

        if source_k_power == 0.0:
            source_weight = 1.0
        else:
            K_for_weight = K
            if K_for_weight < source_k_floor:
                K_for_weight = source_k_floor

            if source_k_ref <= 0.0:
                source_weight = 1.0
            else:
                source_weight = (K_for_weight / source_k_ref) ** source_k_power

        w_src = analytic_delta_scale * source_weight * ds
        if (not math.isfinite(w_src)) or w_src <= 0.0:
            continue

        s = s_centers[j]
        src_x[j] = start_pos[0] + s * track_dir[0]
        src_y[j] = start_pos[1] + s * track_dir[1]
        src_z[j] = start_pos[2] + s * track_dir[2]
        src_w[j] = w_src

        if return_times:
            src_t[j] = starting_time + s / v + delta_e_time_offset_ns
        else:
            src_t[j] = 0.0

        # The original scalar interpolation returned zero for K below the
        # table minimum.  In this model K_grid[0] is 0, and nonpositive K has
        # already been filtered, so the lower clip only protects roundoff.
        Kc = K
        if Kc < K_min:
            Kc = K_min
        elif Kc > K_max:
            Kc = K_max

        iK = int(math.floor((Kc - K_min) * inv_dK))
        if iK < 0:
            iK = 0
        elif iK > nK - 2:
            iK = nK - 2

        K0 = K_grid[iK]
        K1 = K_grid[iK + 1]
        tK = (Kc - K0) / (K1 - K0 + 1e-300)
        if tK < 0.0:
            tK = 0.0
        elif tK > 1.0:
            tK = 1.0

        src_iK[j] = iK
        src_tK[j] = tK
        src_valid[j] = 1

    # ------------------------------------------------------------------
    # Constants for u-grid interpolation.
    # ------------------------------------------------------------------
    u_min = u_grid[0]
    u_max = u_grid[u_grid.size - 1]
    du = u_grid[1] - u_grid[0]
    inv_du = 1.0 / du
    nU = u_grid.size

    # omega_ref is loop-invariant (depends only on pmt_radius_mm and ref_r_mm),
    # so compute it once instead of inside _finite_disk_rel_scalar_numba on every
    # (PMT, source) pair.  The per-pair sqrt(r^2 + a^2) is left untouched so this
    # is bit-identical to calling the helper.
    fd_a = pmt_radius_mm
    fd_R0 = ref_r_mm
    fd_use_ratio = use_finite_disk and (fd_a > 0.0)
    fd_omega_ref = 1.0 - fd_R0 / math.sqrt(fd_R0 * fd_R0 + fd_a * fd_a)
    fd_omega_ref_ok = fd_use_ratio and math.isfinite(fd_omega_ref) and (fd_omega_ref > 0.0)

    # ------------------------------------------------------------------
    # PMT accumulation.
    # IMPORTANT: this is intentionally not parallelized with prange/OpenMP.
    # Some batch drivers fork worker processes after importing/compiling this
    # module, and GNU OpenMP aborts on fork-after-OpenMP.  Keeping this loop
    # serial preserves multiprocessing compatibility while retaining the
    # source precomputation and fast interpolation optimizations.
    # ------------------------------------------------------------------
    for i in range(n_pmts):
        px = p_locations[i, 0]
        py = p_locations[i, 1]
        pz = p_locations[i, 2]

        nx = direction_zs[i, 0]
        ny = direction_zs[i, 1]
        nz = direction_zs[i, 2]

        mpmt_code = mpmt_codes[i]

        mu_i = 0.0
        tnum_i = 0.0

        for j in range(n_src):
            if src_valid[j] == 0:
                continue

            dx = px - src_x[j]
            dy = py - src_y[j]
            dz = pz - src_z[j]

            r2 = dx * dx + dy * dy + dz * dz
            if r2 <= 0.0:
                continue

            r = math.sqrt(r2) + 0.01
            inv_r = 1.0 / r

            # Direction cosine of photon direction relative to the muon.
            # The refined table now spans the full range -1 <= u <= 1, so
            # backward-hemisphere PMTs receive the (physical) backward knock-on
            # light instead of being dropped.  Values slightly outside [-1, 1]
            # from roundoff are clamped to the nearest edge below.
            u = (dx * track_dir[0] + dy * track_dir[1] + dz * track_dir[2]) * inv_r
            if (not math.isfinite(u)) or u < -1.0:
                continue
            if u > 1.0:
                u = 1.0

            # PMT-facing factor.  This is checked before the optical response;
            # negative values cannot contribute.
            cost = -(dx * nx + dy * ny + dz * nz) * inv_r
            if not math.isfinite(cost):
                continue

            # SMOOTH-NLL: segment-averaged visibility gate.
            # Each secondary source represents a finite track segment, but its
            # light is lumped at the segment center.  With a hard point gate,
            # a PMT whose horizon (cost=0) lies inside the segment turns fully
            # on/off as the fitted parameters slide the center across it -- a
            # step of up to several NLL units (verified: one crossing tripled
            # a tube's delta charge within 0.1 mm of L).  Here the gate weight
            # is instead the fraction of the segment on the visible side,
            # linear from the sign of the (unnormalized) facing product at the
            # segment endpoints: continuous in the parameters (kink, not step),
            # and light-conserving.  use_seg_gate=0 restores the point gate.
            vis_seg = 1.0
            if use_seg_gate != 0:
                a_lo = -((px - src_lo[j, 0]) * nx + (py - src_lo[j, 1]) * ny + (pz - src_lo[j, 2]) * nz)
                a_hi = -((px - src_hi[j, 0]) * nx + (py - src_hi[j, 1]) * ny + (pz - src_hi[j, 2]) * nz)
                if a_lo <= 0.0 and a_hi <= 0.0:
                    continue
                if a_lo > 0.0 and a_hi > 0.0:
                    vis_seg = 1.0
                elif a_hi > 0.0:
                    vis_seg = a_hi / (a_hi - a_lo)
                else:
                    vis_seg = a_lo / (a_lo - a_hi)
                if cost <= 0.0:
                    # partial-visibility segment whose center is just behind the
                    # horizon: evaluate the optical factors at grazing incidence.
                    cost = 0.0
            elif cost <= 0.0:
                continue

            # SMOOTH-NLL: soft visibility ramp at the PMT-facing gate.
            # The hard cost<=0 cutoff combined with power_law(0+)~0.125 makes
            # each (source, PMT) contribution a step function of the track
            # parameters; as sources slide with the fitted length, these steps
            # imprint discontinuities of up to several NLL units on the FCN
            # (verified: a single edge-on crossing tripled a tube's expected
            # delta charge within 0.1 mm of L).  The smoothstep below takes the
            # contribution to zero continuously over cost in [0, cost_soft],
            # mirroring the primary path's cost_soft option.  cost_soft<=0
            # restores the historical hard gate exactly.
            if cost_soft > 0.0:
                if cost >= cost_soft:
                    vis_d = 1.0
                else:
                    uu_d = cost / cost_soft
                    vis_d = uu_d * uu_d * (3.0 - 2.0 * uu_d)
            else:
                vis_d = 1.0

            # ----------------------------------------------------------
            # Fast bilinear interpolation of dS_delta/du(K, u).
            # K interpolation terms are source-only and were precomputed.
            # ----------------------------------------------------------
            uc = u
            if uc < u_min:
                uc = u_min
            elif uc > u_max:
                uc = u_max

            iu = int(math.floor((uc - u_min) * inv_du))
            if iu < 0:
                iu = 0
            elif iu > nU - 2:
                iu = nU - 2

            u0 = u_grid[iu]
            u1 = u_grid[iu + 1]
            tu = (uc - u0) / (u1 - u0 + 1e-300)
            if tu < 0.0:
                tu = 0.0
            elif tu > 1.0:
                tu = 1.0

            iK = src_iK[j]
            tK = src_tK[j]

            p00 = table[iK, iu]
            p01 = table[iK, iu + 1]
            p10 = table[iK + 1, iu]
            p11 = table[iK + 1, iu + 1]

            p0 = p00 + tu * (p01 - p00)
            p1 = p10 + tu * (p11 - p10)
            kernel = p0 + tK * (p1 - p0)

            if (not math.isfinite(kernel)) or kernel <= 0.0:
                continue

            pwr = _power_law_lut_scalar_numba(cost, pl_lut)

            if use_finite_disk:
                # Faithful inline of _finite_disk_rel_scalar_numba with the
                # loop-invariant omega_ref hoisted out (see above).  r already
                # includes the +0.01 used by the helper's r_safe (r > 1e-9 here).
                if fd_omega_ref_ok:
                    fd_shape = 1.0 - r / math.sqrt(r * r + fd_a * fd_a)
                    fd_out = fd_shape / fd_omega_ref
                    if (not math.isfinite(fd_out)) or fd_out < 0.0:
                        fd_out = 0.0
                    optical = fd_out * pwr
                else:
                    optical = (fd_R0 / r) * (fd_R0 / r) * pwr
            else:
                R0 = 1000.0
                optical = (R0 / r) ** distance_power * pwr

            if apply_mpmt_eff:
                optical *= _rel_mpmt_eff_scalar_numba(cost, mpmt_code, rel_eff_table)

            if (not math.isfinite(optical)) or optical <= 0.0:
                continue

            contrib = src_w[j] * optical * kernel * vis_d * vis_seg
            mu_i += contrib

            if return_times:
                t_delta = src_t[j] + r * n_water / c_light
                tnum_i += contrib * t_delta

        mu_i *= intensity
        tnum_i *= intensity
        mu[i] = mu_i
        tnum[i] = tnum_i

    if return_times:
        t = np.empty(n_pmts, dtype=np.float64)
        for i in range(n_pmts):
            if mu[i] > 0.0 and math.isfinite(mu[i]) and math.isfinite(tnum[i]):
                t[i] = tnum[i] / mu[i]
            else:
                t[i] = np.nan
        return mu, t

    return mu, tnum


@njit(cache=True)
def _fused_primary_kernel(
    p_locations,        # n_pmts x 3
    direction_zs,       # n_pmts x 3
    s,                  # n_pmts (raw emission coordinate)
    scale_in,           # n_pmts (from find_scale; 0 outside collapse set)
    s_b,                # n_pmts (collapsed emission coord)
    E_b,                # n_pmts (energy at emission point)
    start_pos,          # 3
    track_dir,          # 3 (already normalized)
    mpmt_codes,         # n_pmts int8
    rel_eff_table,      # 4 x n_grid
    dedx_E_grid,        # stopping-power table energies
    dedx_grid,          # stopping-power table values
    ngeo_norm,          # scalar primary_ngeo normalization
    ngeo_a,             # primary_ngeo_pmt_radius_mm
    pmt_radius,         # front-face radius (collapse pmt_radius)
    n_water,
    mu_mass,
    intensity,
    starting_time,
    v,
    c_light,
    need_times,
    cost_soft,
    cost_soft_centered,
):
    """Single-pass fusion of the primary-light path in get_expected_pes_ts.

    Reproduces, per PMT, the exact algebra of the vectorized implementation:
      s_eff/front-face/valid-s gating, geometry (dx,dy,dz,r,cost), power_law
      angular response, analytic primary_ngeo cone-density falloff (with inline
      dedx interpolation), and uniform-grid rel-mPMT efficiency.  Produces
      mu_primary and (optionally) t_primary with no intermediate full-length
      arrays.

    Numerically equivalent to the original to ~1e-12 (the only differences are
    floating-point summation/order); NOT bit-identical by construction.
    """
    n_pmts = p_locations.shape[0]
    mu_primary = np.zeros(n_pmts, dtype=np.float64)
    if need_times:
        t_primary = np.empty(n_pmts, dtype=np.float64)
    else:
        t_primary = np.empty(0, dtype=np.float64)

    n_grid = rel_eff_table.shape[1]
    nde = dedx_E_grid.shape[0]

    tdx = track_dir[0]
    tdy = track_dir[1]
    tdz = track_dir[2]
    two_pr = 2.0 * pmt_radius

    # power_law constants
    pl_y0 = 0.1209
    pl_yinf = 1.6397
    pl_nfit = 3.0777
    pl_x50n = 0.7942886659271312
    pl_max = 0.967354918872639

    for i in range(n_pmts):
        scale = scale_in[i]
        # s_eff = collapsed coord where scale>0, else raw s
        if scale > 0.0:
            s_eff = s_b[i]
        else:
            s_eff = s[i]

        # front-face attenuation
        if s_eff < pmt_radius:
            scale = scale * (s_eff + pmt_radius) / two_pr

        # valid-s gate
        if s_eff >= -pmt_radius:
            valid_s = True
        else:
            valid_s = False
            scale = 0.0
            s_eff = 0.0

        if need_times:
            t_primary[i] = np.nan  # set below if active; matches t_primary array shape

        # geometry at the effective emission point
        ex = start_pos[0] + s_eff * tdx
        ey = start_pos[1] + s_eff * tdy
        ez = start_pos[2] + s_eff * tdz
        dx = p_locations[i, 0] - ex
        dy = p_locations[i, 1] - ey
        dz = p_locations[i, 2] - ez
        r = math.sqrt(dx * dx + dy * dy + dz * dz) + 0.01

        cost = -(dx * direction_zs[i, 0] + dy * direction_zs[i, 1] + dz * direction_zs[i, 2]) / r
        # Visibility gate.  Hard (cost_soft<=0): contribution is on for cost>0,
        # off otherwise -> a discontinuity at cost=0 (power_law(0+)~0.125).
        # Soft (cost_soft>0): smoothstep ramp over cost in [0, cost_soft] so the
        # PMT turns on/off continuously, removing the per-PMT cliff that roughens
        # the FCN and can trap optimizer seeds.
        if cost_soft > 0.0 and cost_soft_centered != 0:
            # SMOOTH-NLL (opt-in): visibility ramp CENTERED on cost=0, over
            # [-cost_soft, +cost_soft].  The hard gate at the collapsed
            # emission point switches a tube's full primary contribution in
            # one step when its facing cosine crosses zero (verified on a
            # production event: cost(s_b) sign flip within 0.02 mm of L
            # toggled 0 <-> 0.59 PE).  Unlike the one-sided ramp below, this
            # form gives HALF the light at exact grazing -- the finite PMT
            # face is half-illuminated by a source on its horizon -- so tubes
            # at small positive cost are not starved of expected light (the
            # failure mode of the one-sided ramp in the delta path).
            valid_cost = math.isfinite(cost) and (cost > -cost_soft)
            if cost >= cost_soft:
                vis = 1.0
            elif cost <= -cost_soft:
                vis = 0.0
            else:
                u_s = (cost + cost_soft) / (2.0 * cost_soft)
                vis = u_s * u_s * (3.0 - 2.0 * u_s)
        else:
            valid_cost = math.isfinite(cost) and (cost > 0.0)
            if cost_soft > 0.0:
                if cost <= 0.0:
                    vis = 0.0
                elif cost >= cost_soft:
                    vis = 1.0
                else:
                    u_s = cost / cost_soft
                    vis = u_s * u_s * (3.0 - 2.0 * u_s)
            else:
                vis = 1.0
        if not valid_cost:
            scale = 0.0

        # primary time uses s_eff and r regardless of activity (matches vectorized
        # t_primary which is computed for all PMTs); the caller only consumes it
        # where mu>0, but we fill it to preserve identical values where finite.
        if need_times:
            t_primary[i] = starting_time + s_eff / v + r * n_water / c_light

        active = (scale > 0.0) and valid_cost
        if not active:
            continue

        # power_law(cost)
        x = cost
        if x < 0.0:
            x = 0.0
        xn = x ** pl_nfit
        pwr = (pl_y0 + (pl_yinf - pl_y0) * (xn / (xn + pl_x50n))) / pl_max

        # primary_ngeo_falloff_raw(E_b, r) * ngeo_norm
        E = E_b[i]
        gamma = 1.0 + E / mu_mass
        g2 = gamma * gamma
        beta2 = 1.0 - 1.0 / g2
        if beta2 < 0.0:
            beta2 = 0.0
        beta = math.sqrt(beta2)
        corr = 0.0
        if n_water * beta > 1.0:
            cos_tc = 1.0 / (n_water * beta)
            sin2_tc = 1.0 - cos_tc * cos_tc
            # inline dedx interp (np.interp with edge clamp)
            if E <= dedx_E_grid[0]:
                dEdx = dedx_grid[0]
            elif E >= dedx_E_grid[nde - 1]:
                dEdx = dedx_grid[nde - 1]
            else:
                lo = 0
                hi = nde
                while lo < hi:
                    mid = (lo + hi) // 2
                    if dedx_E_grid[mid] < E:
                        lo = mid + 1
                    else:
                        hi = mid
                # lo is first index with grid >= E
                x0 = dedx_E_grid[lo - 1]
                x1 = dedx_E_grid[lo]
                y0 = dedx_grid[lo - 1]
                y1 = dedx_grid[lo]
                dEdx = y0 + (E - x0) / (x1 - x0) * (y1 - y0)
            dc_ds = dEdx / (n_water * mu_mass * beta ** 3 * gamma ** 3)
            r_eff = math.sqrt(r * r + ngeo_a * ngeo_a)
            denom = r_eff * sin2_tc + r_eff * r_eff * dc_ds
            if math.isfinite(denom) and denom > 0.0:
                corr = (1.0 / denom) * ngeo_norm * pwr

        # rel-mPMT efficiency (uniform grid, code-based)
        rel = 1.0
        code = mpmt_codes[i]
        if code >= 0 and code < rel_eff_table.shape[0] and n_grid >= 2:
            xc = cost
            if xc < 0.0:
                xc = 0.0
            elif xc > 1.0:
                xc = 1.0
            xc *= (n_grid - 1)
            i0 = int(math.floor(xc))
            if i0 < 0:
                i0 = 0
            elif i0 > n_grid - 2:
                i0 = n_grid - 2
            t = xc - i0
            rel = rel_eff_table[code, i0] + t * (rel_eff_table[code, i0 + 1] - rel_eff_table[code, i0])

        mu_primary[i] = intensity * corr * scale * rel * vis

    return mu_primary, t_primary


def _get_pmt_radius_cached(wcd):
    """Cache the PMT radius lookup from the WCD object."""
    key = id(wcd)
    val = _PMT_RADIUS_CACHE.get(key)
    if val is None:
        val = float(wcd.mpmts[0].pmts[0].get_properties("design")["size"]) / 2.0
        _PMT_RADIUS_CACHE[key] = val
    return val

def _finite_disk_solid_angle_rel(r_mm, pmt_radius_mm=37.0, ref_r_mm=1000.0):
    """
    Relative face-on solid angle of a circular PMT disk.

    Exact face-on solid angle:

        Omega(r) = 2*pi * (1 - r / sqrt(r^2 + a^2))

    where:
        r = source-to-PMT distance
        a = PMT radius

    This function returns Omega(r) / Omega(ref_r_mm), so the factor is
    dimensionless and equals 1 at the reference distance.

    This replaces the arbitrary (R0/r)^p distance law with the finite-aperture
    point-source collection law.  It does NOT include PMT angular response,
    because that is already handled by pwr_corr.
    """
    r = np.asarray(r_mm, dtype=np.float64)
    r_safe = np.maximum(r, 1e-9)

    a = float(pmt_radius_mm)
    R0 = float(ref_r_mm)

    if a <= 0.0:
        # Far-field point-aperture limit.
        return (R0 / r_safe) ** 2

    omega_shape = 1.0 - r_safe / np.sqrt(r_safe * r_safe + a * a)
    omega_ref = 1.0 - R0 / np.sqrt(R0 * R0 + a * a)

    if (not np.isfinite(omega_ref)) or omega_ref <= 0.0:
        return (R0 / r_safe) ** 2

    out = omega_shape / omega_ref
    out[~np.isfinite(out)] = 0.0
    out[out < 0.0] = 0.0

    return out

def _primary_ngeo_raw_static(E_MeV, r_mm, *, n=1.344, mu_mass=105.658, pmt_radius_mm=37.0):
    """Static version of primary_ngeo_falloff_raw used for cached normalization."""
    E = np.asarray(E_MeV, dtype=np.float64)
    r = np.asarray(r_mm, dtype=np.float64)

    gamma = 1.0 + E / mu_mass
    beta2 = np.clip(1.0 - 1.0 / np.maximum(gamma, 1e-30) ** 2, 0.0, None)
    beta = np.sqrt(beta2)

    above = n * beta > 1.0

    cos_tc = np.zeros_like(E, dtype=np.float64)
    cos_tc[above] = 1.0 / (n * beta[above])

    sin2_tc = np.zeros_like(E, dtype=np.float64)
    sin2_tc[above] = 1.0 - cos_tc[above] ** 2

    dEdx = _interp_muon_dedx_positive(E)
    dc_ds = np.zeros_like(E, dtype=np.float64)
    dc_ds[above] = dEdx[above] / (
        n * mu_mass * beta[above] ** 3 * gamma[above] ** 3
    )

    r_eff = np.sqrt(r * r + pmt_radius_mm * pmt_radius_mm)
    denom = r_eff * sin2_tc + r_eff * r_eff * dc_ds

    out = np.zeros(np.broadcast(E, r).shape, dtype=np.float64)
    good = above & np.isfinite(denom) & (denom > 0.0)
    out[good] = 1.0 / denom[good]
    return out

def _electron_cherenkov_threshold_MeV(n=1.344, m_e=0.51099895):
    """
    Electron kinetic-energy Cherenkov threshold in MeV.

        beta_thr = 1/n
        gamma_thr = 1 / sqrt(1 - beta_thr^2)
        T_thr = m_e (gamma_thr - 1)
    """
    beta_thr = 1.0 / float(n)
    gamma_thr = 1.0 / np.sqrt(1.0 - beta_thr * beta_thr)
    return float(m_e * (gamma_thr - 1.0))


def _electron_range_cm_water_approx(T_MeV):
    """
    Approximate electron CSDA range in water [cm].

    This is the same empirical relation already used in your secondary-electron
    model.  For water, density ~= 1 g/cm^3, so g/cm^2 and cm are numerically
    equivalent to good approximation.

    T_MeV can be scalar or array.
    """
    T = np.asarray(T_MeV, dtype=np.float64)
    T_safe = np.maximum(T, 1e-12)

    R_cm = np.zeros_like(T_safe, dtype=np.float64)

    low = T_safe <= 2.5
    R_cm[low] = 0.412 * T_safe[low] ** (
        1.265 - 0.0954 * np.log(T_safe[low])
    )
    R_cm[~low] = 0.530 * T_safe[~low] - 0.106

    R_cm = np.maximum(R_cm, 0.0)
    return R_cm


def _electron_stopping_power_MeV_per_cm_water_approx(T_MeV):
    """
    Effective electron stopping power in water [MeV/cm].

    Uses the derivative of the same range relation:

        R = R(T)
        dR/dT = cm / MeV
        S(T) = dT/dR = 1 / (dR/dT)

    This is not yet as good as an ESTAR table, but it is already better than
    treating the full electron range as if it emitted at the initial T0.
    """
    T = np.asarray(T_MeV, dtype=np.float64)
    T_safe = np.maximum(T, 1e-8)

    # Relative finite-difference step.  Keep it small but not catastrophically
    # small near threshold.
    dT = np.maximum(1e-4 * T_safe, 1e-6)

    T_lo = np.maximum(T_safe - dT, 1e-8)
    T_hi = T_safe + dT

    R_lo = _electron_range_cm_water_approx(T_lo)
    R_hi = _electron_range_cm_water_approx(T_hi)

    dR_dT = (R_hi - R_lo) / np.maximum(T_hi - T_lo, 1e-30)

    # Avoid division by zero or negative numerical artifacts.
    dR_dT = np.where(np.isfinite(dR_dT) & (dR_dT > 0.0), dR_dT, np.nan)

    S = 1.0 / dR_dT
    S = np.where(np.isfinite(S) & (S > 0.0), S, 1e30)

    return S


def _electron_frank_tamm_factor(T_MeV, n=1.344, m_e=0.51099895):
    """
    Electron Frank--Tamm factor:

        F(T) = 1 - 1 / (n^2 beta(T)^2)

    Returns zero below Cherenkov threshold.
    """
    T = np.asarray(T_MeV, dtype=np.float64)

    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / np.maximum(gamma, 1e-30) ** 2
    beta2 = np.clip(beta2, 0.0, None)

    ft = 1.0 - 1.0 / (float(n) ** 2 * np.maximum(beta2, 1e-30))
    ft = np.where(beta2 * float(n) ** 2 > 1.0, np.maximum(ft, 0.0), 0.0)

    return ft


def _electron_cherenkov_cos_alpha(T_MeV, n=1.344, m_e=0.51099895):
    """
    cos(alpha_e) for an electron of kinetic energy T_MeV.

        cos(alpha_e) = 1 / (n beta_e)

    Values below threshold are returned as nan.
    """
    T = np.asarray(T_MeV, dtype=np.float64)

    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / np.maximum(gamma, 1e-30) ** 2
    beta2 = np.clip(beta2, 0.0, None)
    beta = np.sqrt(beta2)

    above = float(n) * beta > 1.0

    cos_alpha = np.full_like(T, np.nan, dtype=np.float64)
    cos_alpha[above] = 1.0 / (float(n) * beta[above])
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

    return cos_alpha

def _get_particle_stopping_power_table(particle="muon"):
    """
    Build and cache a smooth positive stopping-power table for a primary particle.

    Returns
    -------
    E_grid : ndarray
        Kinetic energies [MeV].
    dEdx_grid : ndarray
        Positive stopping power, -dE/ds [MeV/mm].

    Notes
    -----
    The range table stores total above-threshold range versus initial kinetic
    energy. Differentiating range with respect to kinetic energy gives dR/dE, so

        -dE/ds = 1 / (dR/dE).

    This is the same range-table information used by the collapse solver, just
    rearranged into the derivative needed by the analytic N_geo formula.
    """
    pname = canonical_particle_name(particle)
    cached = _PARTICLE_STOPPING_POWER_CACHE.get(pname)
    if cached is not None:
        return cached

    overall_distances = np.asarray(_get_tables(pname)[2], dtype=np.float64)  # mm
    energy_rows = _get_tables(pname)[3]

    # Initial kinetic energy for each stopping range.
    E0 = np.asarray([float(row[0]) for row in energy_rows], dtype=np.float64)

    order = np.argsort(E0)
    E0 = E0[order]
    ranges = overall_distances[order]

    keep = np.isfinite(E0) & np.isfinite(ranges)
    E0 = E0[keep]
    ranges = ranges[keep]

    unique_E, unique_idx = np.unique(E0, return_index=True)
    E0 = unique_E
    ranges = ranges[unique_idx]

    dR_dE = np.gradient(ranges, E0)  # mm / MeV
    dEdx = 1.0 / np.maximum(dR_dE, 1e-30)  # MeV / mm

    good = np.isfinite(E0) & np.isfinite(dEdx) & (dEdx > 0.0)
    cached = (E0[good], dEdx[good])
    _PARTICLE_STOPPING_POWER_CACHE[pname] = cached
    return cached


def _interp_particle_dedx_positive(E_MeV, particle="muon"):
    """
    Interpolate positive primary-particle stopping power -dE/ds [MeV/mm].
    """
    E_grid, dEdx_grid = _get_particle_stopping_power_table(particle)
    E = np.asarray(E_MeV, dtype=np.float64)
    return np.interp(E, E_grid, dEdx_grid, left=dEdx_grid[0], right=dEdx_grid[-1])


# Backward-compatible names.
def _get_muon_stopping_power_table():
    return _get_particle_stopping_power_table("muon")


def _interp_muon_dedx_positive(E_MeV):
    return _interp_particle_dedx_positive(E_MeV, "muon")


_REFINED_ANALYTIC_DELTA_CACHE = {}

@njit(cache=True)
def _electron_cherenkov_threshold_numba(n, m_e):
    beta_thr = 1.0 / n
    gamma_thr = 1.0 / math.sqrt(1.0 - beta_thr * beta_thr)
    return m_e * (gamma_thr - 1.0)


@njit(cache=True)
def _electron_range_cm_water_approx_scalar_numba(T):
    T_safe = T
    if T_safe < 1e-12:
        T_safe = 1e-12

    if T_safe <= 2.5:
        R = 0.412 * T_safe ** (1.265 - 0.0954 * math.log(T_safe))
    else:
        R = 0.530 * T_safe - 0.106

    if R < 0.0 or not math.isfinite(R):
        return 0.0
    return R


@njit(cache=True)
def _electron_stopping_power_MeV_per_cm_scalar_numba(T):
    """
    Effective electron stopping power in water [MeV/cm].

    This is the scalar compiled equivalent of
    _electron_stopping_power_MeV_per_cm_water_approx().  It uses the derivative
    of the same empirical range relation already used in the model:

        S(T) = dT/dR = 1 / (dR/dT).
    """
    T_safe = T
    if T_safe < 1e-8:
        T_safe = 1e-8

    dT = 1e-4 * T_safe
    if dT < 1e-6:
        dT = 1e-6

    T_lo = T_safe - dT
    if T_lo < 1e-8:
        T_lo = 1e-8

    T_hi = T_safe + dT

    R_lo = _electron_range_cm_water_approx_scalar_numba(T_lo)
    R_hi = _electron_range_cm_water_approx_scalar_numba(T_hi)

    dR_dT = (R_hi - R_lo) / (T_hi - T_lo)

    if (not math.isfinite(dR_dT)) or dR_dT <= 0.0:
        return 1e30

    S = 1.0 / dR_dT

    if (not math.isfinite(S)) or S <= 0.0:
        return 1e30

    return S


@njit(cache=True)
def _electron_frank_tamm_factor_scalar_numba(T, n, m_e):
    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / (gamma * gamma)

    if beta2 <= 0.0:
        return 0.0

    if beta2 * n * n <= 1.0:
        return 0.0

    ft = 1.0 - 1.0 / (n * n * beta2)

    if ft < 0.0 or not math.isfinite(ft):
        return 0.0

    return ft


@njit(cache=True)
def _electron_cherenkov_cos_alpha_scalar_numba(T, n, m_e):
    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / (gamma * gamma)

    if beta2 <= 0.0:
        return np.nan

    beta = math.sqrt(beta2)

    if n * beta <= 1.0:
        return np.nan

    c = 1.0 / (n * beta)

    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0

    return c


@njit(cache=True)
def _electron_recoil_cos_theta_numba(K_mu, T_e0, m_mu, m_e):
    E_mu = K_mu + m_mu
    p_mu2 = E_mu * E_mu - m_mu * m_mu
    if p_mu2 <= 0.0:
        return 1.0

    p_e2 = T_e0 * (T_e0 + 2.0 * m_e)
    if p_e2 <= 0.0:
        return 1.0

    p_mu = math.sqrt(p_mu2)
    p_e = math.sqrt(p_e2)

    c = T_e0 * (E_mu + m_e) / (p_mu * p_e)

    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0

    return c


@njit(cache=True)
def _add_arcsine_kernel_to_row_numba(row, u_centers, du, A, B, weight):
    """
    Add weight * p(u) to row, where

        u = A + B cos(phi),

    and p(u) is the bin-averaged arcsine density.

    This avoids allocating a full kernel array for every T0/T step.
    """
    n_u = u_centers.size

    if weight <= 0.0:
        return

    if (not math.isfinite(A)) or (not math.isfinite(B)) or (not math.isfinite(weight)):
        return

    if B < 0.0:
        B = -B

    u_min_edge = u_centers[0] - 0.5 * du
    u_max_edge = u_centers[n_u - 1] + 0.5 * du

    # Collapsed-cone limit: p(u) = delta(u - A).
    if B <= 1e-12:
        if A < u_min_edge or A > u_max_edge:
            return

        idx = int(math.floor((A - u_min_edge) / du))
        if idx < 0:
            idx = 0
        elif idx >= n_u:
            idx = n_u - 1

        row[idx] += weight / du
        return

    support_lo = A - B
    support_hi = A + B

    if support_hi < u_min_edge or support_lo > u_max_edge:
        return

    j0 = int(math.floor((support_lo - u_min_edge) / du))
    j1 = int(math.floor((support_hi - u_min_edge) / du))

    if j0 < 0:
        j0 = 0
    if j1 >= n_u:
        j1 = n_u - 1

    inv_pi = 1.0 / math.pi

    for j in range(j0, j1 + 1):
        u = u_centers[j]

        lo = (u - 0.5 * du - A) / B
        hi = (u + 0.5 * du - A) / B

        if hi < -1.0 or lo > 1.0:
            continue

        if lo < -1.0:
            lo = -1.0
        elif lo > 1.0:
            lo = 1.0

        if hi < -1.0:
            hi = -1.0
        elif hi > 1.0:
            hi = 1.0

        prob = (math.asin(hi) - math.asin(lo)) * inv_pi

        if prob > 0.0 and math.isfinite(prob):
            row[j] += weight * prob / du


@njit(cache=True)
def _inv_lambda_tr_electron_numba(T, m_e):
    """
    First-principles transport inverse-mean-free-path 1/lambda_tr(T) [1/cm]
    for an electron of kinetic energy T (MeV) in water.

    This is the screened-Rutherford elastic cross section with the relativistic
    (Mott) spin correction, screened by the Thomas-Fermi atomic potential, summed
    over the atoms of H2O.  No empirical normalization or tuning factor is used;
    every quantity is a physical constant or a kinematic function of T.

        1/lambda_tr = N_mol * sum_atoms nat * Z(Z+1) * (r_e^2 / (4 gamma^2 beta^4))
                                      * 8 pi * (L0 - beta^2 L2),

    with screening angle parameter

        eta_screen = 0.25 * chi^2 * (1.13 + 3.76 (alpha Z / beta)^2),
        chi        = hbar c / (p a_TF),   a_TF = 0.885 a0 Z^(-1/3),

    and the transport integrals

        L0 = ln(1 + 1/eta) - 1/(1+eta),
        L2 = 1 + eta - 2 eta ln((1+eta)/eta) - eta^2/(1+eta)   (clipped at 0).

    The l=1 (transport) coefficient returned here sets the mean cosine of the
    accumulated multiple scattering through <cos> = exp(-Integral 1/lambda_tr ds),
    i.e. the small-angle multiple-scattering (diffusion) limit, which is the
    correct regime for the sub-MeV knock-on electrons that carry the light.
    """
    if T <= 0.0:
        return 0.0

    gamma = 1.0 + T / m_e
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    if beta2 <= 0.0:
        return 0.0
    beta = math.sqrt(beta2)
    p = math.sqrt(T * (T + 2.0 * m_e))  # MeV/c

    r_e_cm = 2.8179403262e-13
    N_A = 6.02214076e23
    N_mol = N_A / 18.01528  # water molecules per cm^3 at rho = 1 g/cm^3

    alpha = 1.0 / 137.036
    hbarc = 197.3269804     # MeV fm
    a0_fm = 0.529177e5      # Bohr radius in fm

    inv = 0.0
    for ia in range(2):
        if ia == 0:
            Z = 8.0
            nat = 1.0       # one O per H2O
        else:
            Z = 1.0
            nat = 2.0       # two H per H2O

        a_TF = 0.885 * a0_fm * Z ** (-1.0 / 3.0)
        chi = hbarc / (p * a_TF)
        eta_s = 0.25 * chi * chi * (1.13 + 3.76 * (alpha * Z / beta) ** 2)
        if eta_s <= 0.0:
            continue

        L0 = math.log(1.0 + 1.0 / eta_s) - 1.0 / (1.0 + eta_s)
        L2 = (
            1.0
            + eta_s
            - 2.0 * eta_s * math.log((1.0 + eta_s) / eta_s)
            - eta_s * eta_s / (1.0 + eta_s)
        )
        if L2 < 0.0:
            L2 = 0.0

        sig_tr = (
            nat
            * Z * (Z + 1.0)
            * r_e_cm * r_e_cm / (4.0 * gamma * gamma * beta2 * beta2)
            * 8.0 * math.pi
            * (L0 - beta2 * L2)
        )
        if math.isfinite(sig_tr) and sig_tr > 0.0:
            inv += sig_tr

    out = N_mol * inv
    if (not math.isfinite(out)) or out < 0.0:
        return 0.0
    return out


@njit(cache=True)
def _fill_refined_analytic_delta_table_numba(
    K_grid,
    u_centers,
    table,
    n,
    n_T0,
    n_T_slow,
    projectile_mass,
    n_psi,
):
    """
    Compiled version of the slowing-down secondary-electron table builder.

    Same physics as the slow Python version:

        dS_delta/du(K,u)
        =
        integral dT0 dN/dT0
        integral dT [F_e(T)/S_e(T)] p(u | T0,T),

    but without Python loops or repeated kernel-array allocations.
    """
    m_e = 0.51099895
    m_mu = projectile_mass

    r_e_cm = 2.8179403262e-13
    N_A = 6.02214076e23
    rho_water = 1.0

    n_e = rho_water * N_A * (10.0 / 18.01528)
    ft_sat_mu = 1.0 - 1.0 / (n * n)

    T_thr = _electron_cherenkov_threshold_numba(n, m_e)

    n_K = K_grid.size
    n_u = u_centers.size
    du = u_centers[1] - u_centers[0]

    T0_min = T_thr * 1.0001
    log_T0_min = math.log(T0_min)

    for iK in range(n_K):
        K_mu = K_grid[iK]

        gamma_mu = 1.0 + K_mu / m_mu
        beta2_mu = 1.0 - 1.0 / (gamma_mu * gamma_mu)

        if beta2_mu <= 0.0:
            continue

        T_max = (
            2.0 * m_e * beta2_mu * gamma_mu * gamma_mu
            / (1.0 + 2.0 * gamma_mu * m_e / m_mu + (m_e / m_mu) * (m_e / m_mu))
        )

        if (not math.isfinite(T_max)) or T_max <= T_thr:
            continue

        log_T0_max = math.log(T_max)
        dlog_T0 = (log_T0_max - log_T0_min) / n_T0

        prefactor = n_e * 2.0 * math.pi * r_e_cm * r_e_cm * m_e / beta2_mu

        for iT0 in range(n_T0):
            T0_lo = math.exp(log_T0_min + iT0 * dlog_T0)
            T0_hi = math.exp(log_T0_min + (iT0 + 1) * dlog_T0)

            T0 = math.sqrt(T0_lo * T0_hi)
            dT0 = T0_hi - T0_lo

            if T0 <= T_thr:
                continue

            dN_dx_dT0 = (
                prefactor
                * (1.0 / (T0 * T0))
                * (1.0 - beta2_mu * T0 / T_max)
            )

            if dN_dx_dT0 <= 0.0 or not math.isfinite(dN_dx_dT0):
                continue

            prod_weight = dN_dx_dT0 * dT0

            cos_te = _electron_recoil_cos_theta_numba(K_mu, T0, m_mu, m_e)
            sin2_te = 1.0 - cos_te * cos_te
            if sin2_te < 0.0:
                sin2_te = 0.0
            sin_te = math.sqrt(sin2_te)

            if T0 <= T_thr * 1.0002:
                continue

            log_T_min = math.log(T_thr * 1.0001)
            log_T_max = math.log(T0)
            dlog_T = (log_T_max - log_T_min) / n_T_slow

            # Accumulate the electron multiple-scattering transport optical depth
            #   eta(T) = Integral_T^T0 (1/lambda_tr) dT'/S_e(T')
            # while the electron slows from its birth energy T0 down to each
            # emission energy T.  The mean cosine of the accumulated scattering
            # relative to the birth (recoil) direction is exp(-eta(T)) in the
            # small-angle multiple-scattering (diffusion) limit, which is the
            # correct regime for these sub-MeV knock-on electrons.  We therefore
            # march iT from high T (just born, eta ~ 0) down to low T (most
            # scattered).  Light emitted at energy T is Cherenkov-coned about the
            # scattered electron direction, which is itself distributed about the
            # recoil axis with mean cosine exp(-eta).  There is no tuning factor
            # anywhere in this chain.
            two_pi = 2.0 * math.pi
            inv_n_psi = 1.0 / n_psi
            run_eta = 0.0

            for iT in range(n_T_slow - 1, -1, -1):
                T_lo = math.exp(log_T_min + iT * dlog_T)
                T_hi = math.exp(log_T_min + (iT + 1) * dlog_T)

                T = math.sqrt(T_lo * T_hi)
                dT = T_hi - T_lo

                S = _electron_stopping_power_MeV_per_cm_scalar_numba(T)
                if S <= 0.0 or not math.isfinite(S):
                    continue

                # Path length traversed in this energy bin [cm] and the transport
                # optical depth accrued over it (mid-bin value of the cumulative).
                ds_path = dT / S
                inv_ltr = _inv_lambda_tr_electron_numba(T, m_e)
                eta_mid = run_eta + 0.5 * inv_ltr * ds_path
                run_eta += inv_ltr * ds_path

                ft = _electron_frank_tamm_factor_scalar_numba(T, n, m_e)
                if ft <= 0.0:
                    continue

                # dT/S is path length in cm.
                dY_equiv = ft * dT / S / ft_sat_mu
                if dY_equiv <= 0.0 or not math.isfinite(dY_equiv):
                    continue

                cos_alpha = _electron_cherenkov_cos_alpha_scalar_numba(T, n, m_e)
                if not math.isfinite(cos_alpha):
                    continue

                sin2_alpha = 1.0 - cos_alpha * cos_alpha
                if sin2_alpha < 0.0:
                    sin2_alpha = 0.0
                sin_alpha = math.sqrt(sin2_alpha)

                # Mean cosine of the accumulated multiple scattering at this
                # emission energy (diffusion limit).
                cos_ms = math.exp(-eta_mid)
                if cos_ms > 1.0:
                    cos_ms = 1.0
                elif cos_ms < -1.0:
                    cos_ms = -1.0
                sin2_ms = 1.0 - cos_ms * cos_ms
                if sin2_ms < 0.0:
                    sin2_ms = 0.0
                sin_ms = math.sqrt(sin2_ms)

                weight = prod_weight * dY_equiv
                w_psi = weight * inv_n_psi

                # Compose recoil (cos_te) (x) multiple scattering (cos_ms) by
                # averaging over the scattering azimuth psi, then add the
                # Cherenkov cone (cos_alpha) about the resulting electron
                # direction through the azimuthal arcsine kernel.  This is the
                # full recoil (x) MS (x) cone angular composition; with cos_ms = 1
                # it reduces exactly to the previous recoil (x) cone model.
                for ip in range(n_psi):
                    psi = (ip + 0.5) * two_pi * inv_n_psi
                    cpsi = math.cos(psi)

                    cos_e = cos_te * cos_ms + sin_te * sin_ms * cpsi
                    if cos_e > 1.0:
                        cos_e = 1.0
                    elif cos_e < -1.0:
                        cos_e = -1.0
                    sin2_e = 1.0 - cos_e * cos_e
                    if sin2_e < 0.0:
                        sin2_e = 0.0
                    sin_e = math.sqrt(sin2_e)

                    A = cos_e * cos_alpha
                    B = sin_e * sin_alpha

                    _add_arcsine_kernel_to_row_numba(
                        table[iK],
                        u_centers,
                        du,
                        A,
                        B,
                        w_psi,
                    )

    # Safety cleanup.
    for iK in range(n_K):
        for iu in range(n_u):
            val = table[iK, iu]
            if (not math.isfinite(val)) or val < 0.0:
                table[iK, iu] = 0.0


def _build_refined_analytic_delta_table(
    n=1.344,
    K_min=0.0,
    K_max=1000.0,
    n_K=180,
    n_u=240,
    n_T0=120,
    n_T_slow=60,
    n_T=None,
    projectile_mass=105.6583755,
    n_psi=12,
):
    """
    Fast compiled builder for dS_delta/du(K_mu, u).

    Physics is the same as the slow Python slowing-down version:

        dS_delta/du
        =
        integral dT0 [dN_delta/(ds dT0)]
        integral dT [F_e(T)/S_e(T)] p(u | T0,T).

    The speedup comes from:
      - no Python loop over K/T0/T/u,
      - no repeated kernel-array allocation,
      - direct bin accumulation into table[iK, iu].
    """
    if n_T is not None:
        n_T0 = int(n_T)

    K_grid = np.linspace(K_min, K_max, int(n_K), dtype=np.float64)

    # Full physical range of the photon direction cosine relative to the muon.
    # The knock-on light is no longer forward-only: once first-principles
    # multiple scattering is included, a physical ~14% of the light is emitted
    # into the backward hemisphere (u < 0), so the table must cover [-1, 1].
    u_centers = np.linspace(
        -1.0 + 1.0 / int(n_u),
        1.0 - 1.0 / int(n_u),
        int(n_u),
        dtype=np.float64,
    )

    table = np.zeros((int(n_K), int(n_u)), dtype=np.float64)

    _fill_refined_analytic_delta_table_numba(
        K_grid,
        u_centers,
        table,
        float(n),
        int(n_T0),
        int(n_T_slow),
        float(projectile_mass),
        int(n_psi),
    )

    return K_grid, u_centers, table




def get_refined_analytic_delta_cache(n=1.344, projectile_mass=105.6583755, particle="muon"):
    """
    Return cached refined analytic secondary-electron table.

    The table depends on projectile mass through the delta-ray kinematics, so it
    is cached per (particle, n, mass) rather than globally as a muon-only table.
    """
    global _REFINED_ANALYTIC_DELTA_CACHE

    pname = canonical_particle_name(particle)
    key = (pname, float(n), float(projectile_mass))
    cached = _REFINED_ANALYTIC_DELTA_CACHE.get(key)
    if cached is None:
        cached = _build_refined_analytic_delta_table(
            n=n,
            projectile_mass=float(projectile_mass),
        )
        _REFINED_ANALYTIC_DELTA_CACHE[key] = cached

    return cached


_DELTA_E_CACHE = None

class Emitter:
    """
    Optimized Cherenkov emitter model used by the fitter.

    The public fit-facing API is preserved, but the hot methods avoid:
      - pickle-based copying
      - debug prints in the fit loop
      - repeated temporary allocations when not needed
    """

    def __init__(
        self,
        starting_time,
        start_coord,
        direction,
        beta,
        length,
        intensity,
        particle="muon",
        track_end_mode="threshold",
        fixed_initial_KE=None,
    ):
        if not isinstance(starting_time, (int, float)):
            raise TypeError("starting_time must be a number")
        if not (
            isinstance(start_coord, tuple)
            and len(start_coord) == 3
            and all(isinstance(c, (int, float)) for c in start_coord)
        ):
            raise TypeError("start_coord must be a tuple of three numbers")
        if not (
            isinstance(direction, tuple)
            and len(direction) == 3
            and all(isinstance(c, (int, float)) for c in direction)
        ):
            raise TypeError("direction must be a tuple of three numbers")
        if not isinstance(beta, (int, float)) or not (0 < beta < 1):
            raise ValueError("beta must be a number between 0 and 1")
        if not isinstance(length, (int, float)) or length <= 0:
            raise ValueError("length must be a positive number")
        if not isinstance(intensity, (int, float)) or intensity <= 0:
            raise ValueError("intensity must be a positive number")

        self.starting_time = float(starting_time)
        self.start_coord = tuple(float(c) for c in start_coord)
        self.direction = tuple(float(c) for c in direction)
        self.length = float(length)
        self.intensity = float(intensity)

        self.particle_name = canonical_particle_name(particle)
        set_active_particle(self.particle_name)
        self.particle_mass = particle_mass_mev(self.particle_name)
        self.mu_mass = self.particle_mass  # Backward-compatible alias used by older helper names.

        # ------------------------------------------------------------------
        # Primary-track endpoint model.
        #
        # threshold:
        #     Old behavior.  The fitted parameter ``length`` is interpreted as
        #     the dE/dx-only range to Cherenkov threshold, so it determines the
        #     initial kinetic energy through the range table.
        #
        # abrupt:
        #     The fitted parameter ``length`` is interpreted as the visible
        #     primary-Cherenkov length before an abrupt cutoff/interaction.
        #     The initial kinetic energy is fixed independently by
        #     ``fixed_initial_KE``.  Internally, ``range_to_threshold_mm`` still
        #     comes from the dE/dx table and is used to evaluate K(s), theta_c(s),
        #     beta, and dE/dx along the visible part of the track.
        # ------------------------------------------------------------------
        self.track_end_mode = "threshold"
        self.fixed_initial_KE = None
        self.range_to_threshold_mm = float(length)
        self.last_visible_length_exceeds_range = False

        self.n = 1.344
        self.c = 299.792458  # mm/ns

        self.beta = float(beta)
        self.v = self.beta * self.c
        self.cos_tq = None
        self.cot_tq = None
        self.interp_E_init = None

        # Per-instance caches for quantities that are repeatedly needed in the
        # Minuit FCN hot loop.
        self._energy_main_idx = None
        self._energy_dist_row = None
        self._energy_energy_row = None
        self._last_geometry_cache_key = None
        self._last_mpmt_type_codes = None
        self._delta_src_grid_cache = {}

        # Production fits do not need to copy large diagnostic arrays on every
        # FCN call.  Set this True only for residual/component debugging.
        self.store_expected_component_diagnostics = False

        self.muon_subthreshold_range_mm = 120 # Backward-compatible name: subthreshold tail distance after primary Cherenkov threshold (mm)
        self.enable_delta_e = True
        self.delta_e_scale = 1



        # Number of source bins along the above-threshold, Cherenkov-visible muon path.
        self.n_delta_steps = 4

        # Force the below-threshold tail to be sampled separately.
        # This prevents the 110 mm tail from disappearing when n_delta_steps is small.
        self.delta_e_tail_step_mm = 20.0
        self.delta_e_tail_min_steps = 3

        # ------------------------------------------------------------------
        # Secondary-electron timing model.
        #
        # The observed times in the current batch driver are charge-weighted
        # mean hit times per PMT, so the expected time should also be a
        # PE-weighted mixture of primary-muon light and secondary-electron
        # light.  The secondary-electron emission time is approximated as the
        # time for the muon to reach the secondary source point plus the photon
        # time of flight from that source point to the PMT.  Any explicit
        # electron-propagation delay can be added with delta_e_time_offset_ns.
        # ------------------------------------------------------------------
        self.use_delta_e_timing = False
        self.delta_e_time_offset_ns = 0

        # Secondary electrons are treated as localized light sources.
        # Their geometric collection factor is therefore projected PMT area / r^2,
        # rather than the primary muon cone/line-source-like 1/r factor.
        self.delta_e_point_source_geometry = True

        # ------------------------------------------------------------------
        # Analytic primary-muon falloff replacement for n_from_E_r.
        #
        # This replaces the WCSim-derived empirical falloff surface with
        #
        #   N_geo(E,r) = C / [r_eff sin^2(theta_c)
        #                    + r_eff^2 d cos(theta_c)/ds]
        #
        # where r_eff = sqrt(r^2 + a^2), a ~= 37 mm is the PMT radius, and
        # d cos(theta_c)/ds is computed from the muon stopping power table.
        #
        # This term is only the geometric/cone-density falloff.  The
        # Frank-Tamm yield factor, PMT angular response, and relative mPMT
        # efficiency remain separate, as in the old model.
        # ------------------------------------------------------------------
        self.use_analytic_primary_ngeo = True
        self.primary_ngeo_pmt_radius_mm = 60.0
        self.primary_ngeo_ref_energy_MeV = 304.0
        self.primary_ngeo_ref_r_mm = 1000.0

        # The original reference energy (304 MeV) is safely above the muon
        # Cherenkov threshold, but it is below the proton threshold in water.
        # If the normalization reference is below threshold,
        # primary_ngeo_falloff_raw() is zero and the old code silently used
        # norm = 1.0, suppressing primary proton light by orders of magnitude.
        # Keep the old muon/pion behavior, but automatically move the reference
        # energy above threshold for heavy particles such as protons.
        self.primary_ngeo_auto_ref_above_threshold = True
        self.primary_ngeo_ref_threshold_factor = 2.0
        self.primary_ngeo_ref_threshold_margin_MeV = 25.0

        # Apply relative mPMT efficiency using each secondary source point's
        # actual incidence angle, not the primary-muon emission angle.
        self.delta_e_apply_mpmt_eff_by_source = True


        # ------------------------------------------------------------------
        # Best data-matching secondary-electron option from the analytic tests.
        #
        # When enabled, the secondary-electron angular/yield model uses a
        # physically motivated dS_delta/du(K_mu, u) table built from:
        #   knock-on electron production,
        #   electron range * Frank-Tamm light yield,
        #   recoil angle + electron Cherenkov cone kinematics,
        #   bin-integrated forward-endpoint handling,
        #   modest electron-transport / multiple-scattering broadening.
        #
        # It replaces the old factorized model:
        #   S_delta(K_mu) * external p(u | K_mu).
        # ------------------------------------------------------------------
        self.use_refined_analytic_delta_e = True


        # ------------------------------------------------------------------
        # Secondary-electron distance falloff.
        #
        # The refined secondary-electron table already contains the energy/yield
        # and angular distribution dS_delta/du(K_mu, u).  The remaining geometric
        # distance factor should be the finite-disk solid-angle falloff of the PMT,
        # normalized to a reference distance.
        #
        # Since pwr_corr already represents the angular detection efficiency of
        # the PMT relative to a face-on PMT at the same distance, do NOT add an
        # extra cos(eta) projected-area factor here.
        # ------------------------------------------------------------------
        self.delta_e_use_finite_disk_solid_angle = True
        self.delta_e_distance_ref_r_mm = 1000.0
        self.delta_e_distance_pmt_radius_mm = 60

        # Kept only for backward-compatible fallback when
        self.delta_e_distance_power = 2

        self.delta_e_source_k_power = 0 #-2.5 #-0.5
        self.delta_e_source_k_ref_MeV = 100.0
        self.delta_e_source_k_floor_MeV = 25.0


        # Overall secondary-electron strength for the refined analytic table.
        # After fixing the electron-energy dT integration and the forward-u
        # endpoint handling, the best low+high joint value was about 3.4.
        self.analytic_delta_scale = 1 #2.5
        
        # SMOOTH-NLL CANDIDATE CONFIG (pending 1000-event A/B validation):
        # small expected-charge floor.  Bounds the -obs*ln(mu) term for tubes
        # with observed charge but ~zero expected charge, whose numerical dust
        # otherwise dominates NLL curvature once the gate steps are fixed
        # (production event 1: 1.28 -> 0.062 max|d2| at 1 mm).  This changes
        # the penalty for unexplained observed charge, i.e. likelihood policy:
        # verify fitted-L vs truth in the batch A/B before trusting.
        # Set to 0 for the historical likelihood.
        self.charge_floor_pe = 1e-4
        
        # Diagnostic only:
        # Artificially shift secondary-electron emission points downstream
        # along the muon direction. This tests whether collapsing electron
        # range back to the muon point is causing the central-light deficit.
        self.delta_e_test_forward_shift_mm = 0
        
        # ------------------------------------------------------------------
        # DEBUG / DIAGNOSTIC ONLY:
        # Reweight the refined secondary-electron angular table toward high u,
        # while optionally preserving each K-row's total integrated yield.
        #
        # u_power = 0 -> original table
        # u_power = 1 -> multiply by u
        # u_power = 2 -> multiply by u^2
        # u_power = 4 -> strongly forward-weighted
        # ------------------------------------------------------------------
        self.delta_e_debug_u_power = 0
        self.delta_e_debug_u_min = 0.0
        self.delta_e_debug_preserve_yield = True
        
        # ------------------------------------------------------------------
        # Primary-muon cone softening.
        #
        # The collapse solver compares the PMT direction alpha(s) to the
        # shrinking Cherenkov angle theta_c(s).  If there is no exact crossing,
        # this width gives a soft nonzero contribution based on the closest
        # angular mismatch:
        #
        #   exp[-0.5 * (min|alpha - theta_c| / sigma)^2]
        #
        # Units: radians.  Set to 0.0 to recover hard zero behavior for
        # no-crossing PMTs.
        # ------------------------------------------------------------------
        self.primary_soft_cone_sigma_rad = 0.005

        # SMOOTH-NLL (opt-in, default off): centered visibility ramp at the
        # PRIMARY path's facing gate.  Set primary_cost_soft to the half-width
        # (e.g. 0.02-0.05 ~ PMT face radius / typical distance) and
        # primary_cost_soft_centered=True to ramp over [-w, +w] instead of the
        # one-sided [0, w].  Both False/0.0 -> historical hard gate, bit-exact.
        self.primary_cost_soft_centered = True

        # SMOOTH-NLL: independent override for the delta segment-visibility
        # gate.  None (default) -> follow smooth_tables; True/False -> force.
        # Exists so ablation studies can isolate the segment gate's effect on
        # fitted lengths without touching the other smooth-table machinery.
        self.delta_e_segment_gate = None

        # SMOOTH-NLL CANDIDATE CONFIG: half-width of the primary facing-gate
        # ramp.  The hard cost<=0 gate at the collapsed emission point switches
        # whole tubes on/off as parameters vary (verified: cost(s_b) sign flip
        # within 0.02 mm of L toggled 0 <-> 0.59 PE; softening took event 1
        # from max|d2| ~17 to 1.28).  Centered vs one-sided was empirically
        # indistinguishable on the tested event; centered is kept for the
        # half-illuminated-face rationale at grazing.  Set to 0.0 (and
        # primary_cost_soft_centered=False) for the historical hard gate.
        self.primary_cost_soft = 0.02

        # SMOOTH-NLL: width of the smoothstep visibility ramp at the delta
        # accumulator's PMT-facing gate (see _accumulate_refined_delta_numba).
        # 0.0 restores the historical hard cost<=0 cutoff.
        self.delta_e_cost_soft = 0.0

        # Configure the endpoint mode after all attributes used by the kinematic
        # refresh helpers exist.  For the default threshold mode this is exactly
        # the old behavior.
        self.configure_track_end(track_end_mode, fixed_initial_KE=fixed_initial_KE, refresh=False)

        # Match the original behavior: initialise beta from the length-dependent
        # lookup table rather than trusting the constructor beta argument.
        self.refresh_kinematics_from_length(self.length)

        
        


    def __repr__(self):
        return (
            f"Emitter(starting_time={self.starting_time}, start_coord={self.start_coord}, "
            f"direction={self.direction}, beta={self.beta}, length={self.length}, "
            f"intensity={self.intensity})"
        )

    def copy(self):
        """
        Lightweight copy.

        The original version used pickle for every copy, which is much more
        expensive than needed for this small numeric state.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        return new

    def set_particle(self, particle):
        """Set the primary particle species for this emitter."""
        self.particle_name = canonical_particle_name(particle)
        set_active_particle(self.particle_name)
        self.particle_mass = particle_mass_mev(self.particle_name)
        self.mu_mass = self.particle_mass  # compatibility alias
        self.interp_E_init = None
        self._energy_main_idx = None
        self._energy_dist_row = None
        self._energy_energy_row = None
        self.range_to_threshold_mm = float(self.length)
        self.last_visible_length_exceeds_range = False
        self.refresh_kinematics_from_length(self.length)
        return self

    def calc_constants(self, n):
        self.n = float(n)
        self.cos_tq = 1.0 / (self.beta * self.n)
        self.cos_tq = np.clip(self.cos_tq, -1.0, 1.0)
        sin_tq = np.sqrt(max(1e-15, 1.0 - self.cos_tq**2))
        self.cot_tq = self.cos_tq / sin_tq
        self.c = 299.792458
        self.v = self.beta * self.c

    @staticmethod
    def nearest_main_idx(length_mm, particle=None):
        tables = _get_tables(particle)
        overall = tables[2]
        idx = np.searchsorted(overall, float(length_mm))
        idx = np.clip(idx, 1, len(overall) - 1)
        left = overall[idx - 1]
        right = overall[idx]
        if (float(length_mm) - left) <= (right - float(length_mm)):
            idx -= 1
        return int(idx)

    def configure_track_end(self, mode="threshold", fixed_initial_KE=None, refresh=True):
        """Configure how the primary Cherenkov track terminates.

        Parameters
        ----------
        mode : {"threshold", "abrupt"}
            ``threshold`` keeps the original behavior: fitted ``length`` is the
            dE/dx range to Cherenkov threshold and therefore determines the
            initial kinetic energy.

            ``abrupt`` makes fitted ``length`` the visible primary-Cherenkov
            length only.  The initial kinetic energy must be supplied separately
            through ``fixed_initial_KE``.  This is intended for protons/hadrons
            whose clean Cherenkov ring can end suddenly before the particle has
            slowed to threshold.
        fixed_initial_KE : float or None
            Fixed initial kinetic energy in MeV for abrupt mode.
        refresh : bool
            If True, immediately refresh beta and the cached K(s) lookup row.
        """
        mode = str(mode).strip().lower()
        aliases = {
            "threshold": "threshold",
            "range": "threshold",
            "csda": "threshold",
            "old": "threshold",
            "abrupt": "abrupt",
            "truncated": "abrupt",
            "interaction": "abrupt",
            "absorbed": "abrupt",
            "absorption": "abrupt",
        }
        if mode not in aliases:
            raise ValueError(
                "track_end_mode must be 'threshold' or 'abrupt' "
                f"(got {mode!r})"
            )

        self.track_end_mode = aliases[mode]
        if fixed_initial_KE is None:
            self.fixed_initial_KE = None
        else:
            self.fixed_initial_KE = float(fixed_initial_KE)

        if self.track_end_mode == "abrupt" and self.fixed_initial_KE is None:
            raise ValueError(
                "abrupt track-end mode requires fixed_initial_KE in MeV. "
                "Use threshold mode if fitted length should determine energy."
            )

        self.interp_E_init = None
        self._energy_main_idx = None
        self._energy_dist_row = None
        self._energy_energy_row = None

        if refresh:
            self.refresh_kinematics_from_length(self.length)
        return self

    # More explicit aliases for fit-driver code.
    set_track_end_mode = configure_track_end
    set_primary_truncation_mode = configure_track_end

    def _range_to_threshold_from_energy(self, initial_KE):
        """Interpolate dE/dx-only Cherenkov-visible range [mm] from K0 [MeV].

        The (initial_energy, overall_distance) lookup arrays depend only on the
        particle's fixed range table, so they are sorted once and cached
        per particle.  Previously this rebuilt a ~3000-element list
        comprehension and argsort on every FCN call in absorption mode.
        """
        initial_energies, overall_distances = _get_range_from_energy_arrays(
            self.particle_name
        )

        return float(
            np.interp(
                float(initial_KE),
                initial_energies,
                overall_distances,
                left=overall_distances[0],
                right=overall_distances[-1],
            )
        )

    def _cache_energy_row_for_range(self, range_mm):
        """Cache the K(s) table row corresponding to a dE/dx range."""
        main_idx = self.nearest_main_idx(float(range_mm), particle=self.particle_name)
        tables = _get_tables(self.particle_name)
        self._energy_main_idx = main_idx
        self._energy_dist_row = tables[4][main_idx]
        self._energy_energy_row = tables[3][main_idx]
        return main_idx

    def visible_length_is_physical(self, tol_mm=1e-9):
        """Return False when an abrupt visible length exceeds the dE/dx range."""
        if self.track_end_mode != "abrupt":
            return True
        return float(self.length) <= float(self.range_to_threshold_mm) + float(tol_mm)

    def _get_energy_rows_for_length(self, L_stop_mm):
        """
        Return the range-table row used to map distance along track to muon KE.

        For the common case L_stop_mm == self.length, the row is cached by
        refresh_kinematics_from_length(), avoiding repeated table searches for
        every secondary-electron source calculation.
        """
        cached_range = float(getattr(self, "range_to_threshold_mm", self.length))
        if (
            self._energy_dist_row is not None
            and self._energy_energy_row is not None
            and np.isclose(float(L_stop_mm), cached_range, rtol=0.0, atol=1e-12)
        ):
            return self._energy_dist_row, self._energy_energy_row

        overall_distances, energy_rows, distance_rows = _get_tables(self.particle_name)[2:5]
        main_idx = np.searchsorted(overall_distances, float(L_stop_mm))
        main_idx = np.clip(main_idx, 1, len(overall_distances) - 1)

        left = overall_distances[main_idx - 1]
        right = overall_distances[main_idx]
        if (float(L_stop_mm) - left) <= (right - float(L_stop_mm)):
            main_idx -= 1

        return distance_rows[main_idx], energy_rows[main_idx]

    def muon_energy_at_s(self, s_mm, L_stop_mm):
        """
        Approximate primary-particle kinetic energy at distance s along the physical muon path.

        Uses the same range-table philosophy as the collapse solver.
        """
        if getattr(self, "smooth_tables", True):
            # SMOOTH-NLL: scalar version of the continuous master-curve lookup.
            return float(self.muon_energy_at_s_array(np.asarray([float(s_mm)]), L_stop_mm)[0])
        dist_row, energy_row = self._get_energy_rows_for_length(L_stop_mm)
        idx = np.searchsorted(dist_row, s_mm)
        idx = np.clip(idx, 0, len(dist_row) - 1)
        return energy_row[idx]

    def muon_energy_at_s_array(self, s_mm, L_stop_mm):
        if getattr(self, "smooth_tables", True):
            # SMOOTH-NLL: continuous K(s) from the master KE(range) curve.
            # The row-based lookup swaps the entire K(s) row when the fitted
            # length crosses a table-row boundary (~5 mm) and snaps K to the
            # nearest row point.  With the secondary-electron model enabled
            # this staircase re-enters the FCN through the delta source grid,
            # re-roughening NLL(L) even after the primary-path fixes.
            try:
                from .particle_cherenkov_model import _ensure_tables_loaded as _pcm_tables
            except ImportError:
                from particle_cherenkov_model import _ensure_tables_loaded as _pcm_tables
            _mt = _pcm_tables(self.particle_name)
            rem = np.maximum(float(L_stop_mm) - np.asarray(s_mm, dtype=np.float64), 0.0)
            return np.interp(rem, _mt["master_range"], _mt["master_ke"],
                             left=_mt["master_ke"][0], right=_mt["master_ke"][-1])
        dist_row, energy_row = self._get_energy_rows_for_length(L_stop_mm)
        idx = np.searchsorted(dist_row, s_mm)
        idx = np.clip(idx, 0, len(dist_row) - 1)
        return energy_row[idx]

    def refresh_kinematics_from_energy(self, initial_KE):
        initial_KE = float(initial_KE)
        if self.interp_E_init is not None and initial_KE == self.interp_E_init:
            return self.interp_E_init

        self.interp_E_init = initial_KE
        self.beta = np.sqrt(
            1.0 - (self.mu_mass / (self.interp_E_init + self.mu_mass)) ** 2
        )
        self.calc_constants(self.n)
        return self.interp_E_init

    def refresh_kinematics_from_length(self, length_mm):
        """Refresh beta and K(s) tables from the current fit length.

        In threshold mode this is the historical behavior: ``length_mm`` selects
        the range-table row and therefore the initial kinetic energy.

        In abrupt mode, ``length_mm`` is only the visible/truncated Cherenkov
        length.  The dE/dx range row and beta are instead selected from the fixed
        initial kinetic energy.
        """
        self.length = float(length_mm)

        if getattr(self, "track_end_mode", "threshold") == "abrupt":
            if self.fixed_initial_KE is None:
                raise ValueError(
                    "Emitter is in abrupt track-end mode but fixed_initial_KE is not set."
                )
            self.range_to_threshold_mm = self._range_to_threshold_from_energy(self.fixed_initial_KE)
            self._cache_energy_row_for_range(self.range_to_threshold_mm)
            self.last_visible_length_exceeds_range = not self.visible_length_is_physical()
            return self.refresh_kinematics_from_energy(self.fixed_initial_KE)

        # Default/old behavior: fitted length is the range to Cherenkov threshold.
        self.range_to_threshold_mm = float(self.length)
        self.last_visible_length_exceeds_range = False
        main_idx = self._cache_energy_row_for_range(self.length)
        tables = _get_tables(self.particle_name)
        if getattr(self, "smooth_tables", True):
            # SMOOTH-NLL: initial KE from the continuous master KE(range)
            # curve instead of the nearest table row.  The nearest-row lookup
            # quantizes KE in ~1 MeV steps every ~5 mm of L, which makes
            # cot(theta_c) (and hence every emission point) a staircase
            # function of the fitted length -- one of the two dominant
            # sources of NLL(L) roughness.  Row caching above is retained
            # unchanged for the secondary-electron table paths.
            try:
                from .particle_cherenkov_model import _ensure_tables_loaded as _pcm_tables
            except ImportError:
                from particle_cherenkov_model import _ensure_tables_loaded as _pcm_tables
            _mt = _pcm_tables(self.particle_name)
            ke0 = float(np.interp(self.length, _mt["master_range"], _mt["master_ke"],
                                  left=_mt["master_ke"][0], right=_mt["master_ke"][-1]))
            return self.refresh_kinematics_from_energy(ke0)
        return self.refresh_kinematics_from_energy(tables[3][main_idx][0])

    def set_nominal_track_parameters(self, starting_time, start_coord, direction, length):
        self.starting_time = float(starting_time)
        self.start_coord = tuple(float(c) for c in start_coord)
        self.direction = tuple(float(c) for c in direction)
        self.length = float(length)

    def set_wall_track_parameters(self, starting_time, y_w, phi_w, d_w, w_y, w_phi, length, r, sign_cz=+1):
        """ Set the track parameters of the emitter using "wall" parameters.

        Args:
            starting_time (float): The time that emitter starts emission in nanoseconds.
            y_w (float): y coordinate of the wall intersection point
            phi_w (float): azimuthal angle of the wall intersection point
            d_w (float): distance from start to wall intersection point
            w_y (float): cosine of angle between line direction and y-axis
            w_phi (float): cosine of angle in x-z plane between line direction and tangent to cylinder at wall point
            length (float): The length of the path for the emitter (mm).
            r (float): radius of the cylinder
            sign_cz (int): sign of c_z to choose branch (+1 or -1)
        """
        (x_0, y_0, z_0, c_x, c_y), _ = self.inverse_transform_and_jacobian(y_w, phi_w, d_w, w_y, w_phi, r, sign_cz)
        self.starting_time = float(starting_time)
        self.start_coord = (x_0, y_0, z_0)
        self.direction = (c_x, c_y)
        self.length = float(length)

    def get_wall_parameters_and_jacobian(self, r, sign_cz=+1):
        """
        Forward: (x_0, y_0, z_0, c_x, c_y) -> (y_w, phi_w, d_w, w_y, w_phi), J_f (5x5)
        Cylinder axis is y; wall: x^2 + z^2 = r^2.  phi_w = atan2(x_w, z_w).
        """
        def _safe_sqrt(x):
            return np.sqrt(np.maximum(0.0, x))

        (x_0, y_0, z_0) = self.start_coord
        (c_x, c_y) = self.direction

        # Direction and checks
        beta_xy = c_x ** 2 + c_y ** 2
        c_z = sign_cz * _safe_sqrt(1.0 - beta_xy)
        beta = c_x ** 2 + c_z ** 2  # = 1 - c_y**2 = ||c_perp||^2
        if beta <= 0:
            raise ValueError("Degenerate direction: c_x=c_z=0 (parallel to axis).")

        # Solve (x0 + t cx)^2 + (z0 + t cz)^2 = r^2 for first t>0
        alpha = x_0 * c_x + z_0 * c_z
        rho0_sq = x_0 ** 2 + z_0 ** 2
        disc = alpha ** 2 + beta * (r ** 2 - rho0_sq)
        if disc < 0:
            raise ValueError("No intersection with cylinder (discriminant < 0).")
        d_w = (-alpha + np.sqrt(disc)) / beta

        # Hit point and cylindrical coords
        x_w = x_0 + d_w * c_x
        z_w = z_0 + d_w * c_z
        phi_w = np.arctan2(x_w, z_w)  # φ=0 on +z
        y_w = y_0 + d_w * c_y

        # Cosines
        w_y = c_y
        S, C = np.sin(phi_w), np.cos(phi_w)
        sqrt_beta = np.sqrt(beta)
        w_phi = (c_x * C - c_z * S) / sqrt_beta

        # Jacobian building blocks
        a = c_x * S + c_z * C  # c_perp · n (>=0 for outward hit)

        # ∂d_w/∂(x0,z0,cx,cz) at fixed (cx,cz)
        dd_dx0_ind = -S / a
        dd_dz0_ind = -C / a
        dd_dcx_ind = -d_w * S / a
        dd_dcz_ind = -d_w * C / a

        # c_z depends on (c_x, c_y):  ∂c_z/∂c_x = -c_x/c_z,  ∂c_z/∂c_y = -c_y/c_z
        dcz_dcx = -c_x / (c_z if c_z != 0 else 1e-300)
        dcz_dcy = -c_y / (c_z if c_z != 0 else 1e-300)

        # Chain to (c_x, c_y)
        dd_dx0 = dd_dx0_ind
        dd_dz0 = dd_dz0_ind
        dd_dcx = dd_dcx_ind + dd_dcz_ind * dcz_dcx
        dd_dcy = dd_dcz_ind * dcz_dcy

        # φ partials:  dφ = (-x_w dz_w + z_w dx_w)/r^2  ⇒  at wall: dφ = (C dx - S dz)/r
        dphi_dx0_ind = C / (r * a)
        dphi_dz0_ind = -S / (r * a)
        dphi_dcx_ind = d_w * C / (r * a)
        dphi_dcz_ind = -d_w * S / (r * a)

        dphi_dx0 = dphi_dx0_ind
        dphi_dz0 = dphi_dz0_ind
        dphi_dcx = dphi_dcx_ind + dphi_dcz_ind * dcz_dcx
        dphi_dcy = dphi_dcz_ind * dcz_dcy  # dφ/dc_y via c_z only
        # dφ/dy0 = 0

        # Assemble forward Jacobian J_f
        J = np.zeros((5, 5), dtype=float)

        # (1) y_w = y_0 + d_w c_y
        J[0, 0] = c_y * dd_dx0
        J[0, 1] = 1.0
        J[0, 2] = c_y * dd_dz0
        J[0, 3] = c_y * dd_dcx
        J[0, 4] = d_w + c_y * dd_dcy

        # (2) φ_w
        J[1, 0] = dphi_dx0
        J[1, 1] = 0.0
        J[1, 2] = dphi_dz0
        J[1, 3] = dphi_dcx
        J[1, 4] = dphi_dcy

        # (3) d_w
        J[2, 0] = dd_dx0
        J[2, 1] = 0.0
        J[2, 2] = dd_dz0
        J[2, 3] = dd_dcx
        J[2, 4] = dd_dcy

        # (4) w_y = c_y
        J[3, 0] = 0.0
        J[3, 1] = 0.0
        J[3, 2] = 0.0
        J[3, 3] = 0.0
        J[3, 4] = 1.0

        # (5) w_phi = (c_x C - c_z S)/sqrt_beta
        inv_sqrtb = 1.0 / (sqrt_beta if sqrt_beta != 0 else 1e-300)
        inv_beta = 1.0 / (beta if beta != 0 else 1e-300)

        # φ-coupling factor:  ∂w_phi/∂φ at fixed (cx,cz) equals (-a)/sqrt_beta
        fac = (-a) * inv_sqrtb

        # wrt (x0,y0,z0): only via φ
        J[4, 0] = fac * dphi_dx0
        J[4, 1] = 0.0
        J[4, 2] = fac * dphi_dz0

        # wrt (c_x, c_y) including c_z and φ dependences
        # general: dw = (C dcx - S dcz)/sqrtβ + fac dφ - w_phi/β (c_x dcx + c_z dcz)
        coeff_dcz = (-S) * inv_sqrtb - w_phi * c_z * inv_beta
        coeff_dcx = (C) * inv_sqrtb - w_phi * c_x * inv_beta

        J[4, 3] = coeff_dcx + coeff_dcz * dcz_dcx + fac * dphi_dcx  # ∂/∂c_x
        J[4, 4] = coeff_dcz * dcz_dcy + fac * dphi_dcy  # ∂/∂c_y

        return (y_w, phi_w, d_w, w_y, w_phi), J

    def inverse_transform_and_jacobian(y_w, phi_w, d_w, w_y, w_phi, r, sign_cz=+1):
        """
        Inverse: (y_w, phi_w, d_w, w_y, w_phi) -> (x_0, y_0, z_0, c_x, c_y), J_g (5x5)
        Using t_phi = (cosφ, 0, -sinφ).
        """
        def _safe_sqrt(x):
            return np.sqrt(np.maximum(0.0, x))

        S, C = np.sin(phi_w), np.cos(phi_w)
        s = _safe_sqrt(1.0 - w_phi ** 2)  # = sin(angle to t_phi) in xz-plane
        sb = _safe_sqrt(1.0 - w_y ** 2)  # = ||c_perp||

        # Direction (c_perp = sb*(w_phi t_phi + s n))
        c_y = w_y
        c_x = sb * (w_phi * C + s * S)
        c_z = sb * (-w_phi * S + s * C)

        # Optional: enforce chosen c_z branch sign
        if sign_cz < 0 and c_z > 0: c_z = -c_z
        if sign_cz > 0 and c_z < 0: c_z = -c_z

        # Wall point and start point
        x_w = r * S
        z_w = r * C
        x_0 = x_w - d_w * c_x
        y_0 = y_w - d_w * c_y
        z_0 = z_w - d_w * c_z

        # Inverse Jacobian J_g
        J = np.zeros((5, 5), dtype=float)

        # helpers
        dsb_dwy = -(w_y / (sb if sb != 0 else 1e-300))
        ds_dwp = -(w_phi / (s if s != 0 else 1e-300))

        # Direction partials
        dcx_dphi = sb * (w_phi * (-S) + s * C)
        dcx_dwy = dsb_dwy * (w_phi * C + s * S)
        dcx_dwp = sb * (C + ds_dwp * S)

        dcz_dphi = sb * (-w_phi * C - s * S)
        dcz_dwy = dsb_dwy * (-w_phi * S + s * C)
        dcz_dwp = sb * (-S + ds_dwp * C)

        # Rows for start point: x_0 = r*S - d_w*c_x;  z_0 = r*C - d_w*c_z;  y_0 = y_w - d_w*c_y
        J[0, 0] = 0.0
        J[0, 1] = r * C - d_w * dcx_dphi
        J[0, 2] = -c_x
        J[0, 3] = -d_w * dcx_dwy
        J[0, 4] = -d_w * dcx_dwp

        J[1, 0] = 1.0
        J[1, 1] = 0.0
        J[1, 2] = -c_y
        J[1, 3] = -d_w
        J[1, 4] = 0.0

        J[2, 0] = 0.0
        J[2, 1] = -r * S - d_w * dcz_dphi  # because d(r*C)/dφ = -r*S
        J[2, 2] = -c_z
        J[2, 3] = -d_w * dcz_dwy
        J[2, 4] = -d_w * dcz_dwp

        # Rows for direction (outputs 4,5)
        J[3, 0] = 0.0
        J[3, 1] = dcx_dphi
        J[3, 2] = 0.0
        J[3, 3] = dcx_dwy
        J[3, 4] = dcx_dwp

        J[4, 0] = 0.0
        J[4, 1] = 0.0
        J[4, 2] = 0.0
        J[4, 3] = 1.0
        J[4, 4] = 0.0

        return (x_0, y_0, z_0, c_x, c_y), J

    def get_emission_point(self, pmt_coord, initial_KE):
        """
        Emission point for a single PMT.
        """
        x0, y0, z0 = self.start_coord
        cx, cy, cz = self.direction
        px, py, pz = pmt_coord

        dx = px - x0
        dy = py - y0
        dz = pz - z0

        self.refresh_kinematics_from_energy(initial_KE)

        u = cx * dx + cy * dy + cz * dz
        A = dx**2 + dy**2 + dz**2

        if A <= u**2:
            return u
        return u - self.cot_tq * np.sqrt(A - u**2)

    def get_emission_points(self, p_locations, initial_KE):
        """
        Vectorized Cherenkov emission-point calculation for many PMTs.
        """
        x0, y0, z0 = self.start_coord
        cx, cy, cz = self.direction

        p_locations = np.asarray(p_locations, dtype=np.float64)
        dx = p_locations[:, 0] - x0
        dy = p_locations[:, 1] - y0
        dz = p_locations[:, 2] - z0

        self.refresh_kinematics_from_energy(initial_KE)

        u = cx * dx + cy * dy + cz * dz
        A = dx * dx + dy * dy + dz * dz

        ss = np.empty(p_locations.shape[0], dtype=np.float64)
        valid = A > u * u
        ss[valid] = u[valid] - self.cot_tq * np.sqrt(A[valid] - u[valid] * u[valid])
        ss[~valid] = u[~valid]
        return ss

    def power_law(self, x):
        y0_fit = 0.1209
        yinf = 1.6397
        x50 = 0.9279
        n_fit = 3.0777

        x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
        xn = x**n_fit
        x50n = x50**n_fit

        max_ = 0.967354918872639
        return (y0_fit + (yinf - y0_fit) * (xn / (xn + x50n))) / max_

    def wl_corr(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_safe = np.maximum(x, 1e-12)

        ymin_wl = 0.1399
        ymax_wl = 1.0
        x50_wl = 3.7620
        n_wl = 2.1020

        return ymin_wl + (ymax_wl - ymin_wl) / (1.0 + (x50_wl / x_safe) ** n_wl)


    def interp_by_mpmt_type(
        self,
        cost,
        mpmt_types,
        cost_for_fit,
        tri_exsitu,
        tri_insitu,
        wut_exsitu,
        wut_insitu,
        fill_empty=1.0,   # was np.nan
    ):
        """
        Interpolate relative mPMT efficiency by mPMT type.

        This keeps the public method signature intact, but uses the faster coded
        implementation whenever the cost grid is the standard uniform [0, 1]
        grid used by this model.  It falls back to the original np.interp loop
        only for a non-standard grid.
        """
        cost = np.asarray(cost, dtype=np.float64)
        cost_for_fit = np.asarray(cost_for_fit, dtype=np.float64)

        if (
            cost_for_fit.ndim == 1
            and cost_for_fit.size == len(tri_exsitu)
            and cost_for_fit.size >= 2
            and np.isclose(cost_for_fit[0], 0.0)
            and np.isclose(cost_for_fit[-1], 1.0)
            and np.allclose(np.diff(cost_for_fit), cost_for_fit[1] - cost_for_fit[0])
        ):
            return _interp_rel_mpmt_eff_from_codes(
                cost,
                _encode_mpmt_types(mpmt_types),
                fill_empty=fill_empty,
            )

        # Fallback: original generic implementation.
        mpmt_types = np.asarray(mpmt_types)
        out = np.full(cost.shape, fill_empty, dtype=np.float64)
        y_by_type = {
            "tri_exsitu": np.asarray(tri_exsitu, dtype=np.float64),
            "tri_insitu": np.asarray(tri_insitu, dtype=np.float64),
            "wut_exsitu": np.asarray(wut_exsitu, dtype=np.float64),
            "wut_insitu": np.asarray(wut_insitu, dtype=np.float64),
        }
        for typ, yvals in y_by_type.items():
            mask = mpmt_types == typ
            if np.any(mask):
                out[mask] = np.interp(
                    cost[mask],
                    cost_for_fit,
                    yvals,
                    left=yvals[0],
                    right=yvals[-1],
                )
        return out


    def particle_dedx_positive(self, E_MeV):
        """
        Positive primary-particle stopping power, -dE/ds, in MeV/mm.

        This is derived from the same particle range table used by the collapse
        solver. It is needed for the analytic cone-density falloff:

            d cos(theta_c)/ds = (-dE/ds) / (n m beta^3 gamma^3).
        """
        return _interp_particle_dedx_positive(E_MeV, self.particle_name)

    def muon_dedx_positive(self, E_MeV):
        """Backward-compatible alias for particle_dedx_positive()."""
        return self.particle_dedx_positive(E_MeV)


    def primary_ngeo_falloff_raw(self, E_MeV, r_mm):
        """
        Analytic cone-density geometric falloff for primary muon light.

        This is the analytic replacement for n_from_E_r(E, r).  It excludes:
          - Frank-Tamm / Cherenkov light-yield scale
          - PMT angular response
          - relative mPMT efficiency

        Those factors are applied elsewhere in get_expected_pes_ts.

        Formula
        -------
        N_geo(E,r) = 1 / [ r_eff sin^2(theta_c(E))
                           + r_eff^2 d cos(theta_c)/ds ]

        where

            r_eff = sqrt(r^2 + a^2)

        and

            d cos(theta_c)/ds = (-dE/ds) / (n m beta^3 gamma^3).

        Units are arbitrary up to an overall constant; the public
        primary_ngeo_falloff() applies a fixed reference normalization so that
        the result has approximately the same convention as n_from_E_r.
        """
        E = np.asarray(E_MeV, dtype=np.float64)
        r = np.asarray(r_mm, dtype=np.float64)

        gamma = 1.0 + E / self.mu_mass
        beta2 = np.clip(1.0 - 1.0 / np.maximum(gamma, 1e-30)**2, 0.0, None)
        beta = np.sqrt(beta2)

        above = self.n * beta > 1.0

        cos_tc = np.zeros_like(E, dtype=np.float64)
        cos_tc[above] = 1.0 / (self.n * beta[above])

        sin2_tc = np.zeros_like(E, dtype=np.float64)
        sin2_tc[above] = 1.0 - cos_tc[above]**2

        dEdx = self.muon_dedx_positive(E)
        dc_ds = np.zeros_like(E, dtype=np.float64)
        dc_ds[above] = dEdx[above] / (
            self.n * self.mu_mass * beta[above]**3 * gamma[above]**3
        )

        a = float(self.primary_ngeo_pmt_radius_mm)
        r_eff = np.sqrt(r*r + a*a)

        denom = r_eff * sin2_tc + r_eff*r_eff * dc_ds

        out = np.zeros(np.broadcast(E, r).shape, dtype=np.float64)
        good = above & np.isfinite(denom) & (denom > 0.0)
        out[good] = 1.0 / denom[good]

        return out


    def primary_ngeo_reference_energy(self):
        """Return a particle-safe reference energy for N_geo normalization.

        The historical value primary_ngeo_ref_energy_MeV=304 MeV was tuned for
        muons.  For protons in water, 304 MeV is below Cherenkov threshold, so
        primary_ngeo_falloff_raw(304 MeV, r_ref) is exactly zero.  That made the
        normalization fall back to 1.0 and effectively removed the primary
        proton ring.
        """
        E_ref = float(self.primary_ngeo_ref_energy_MeV)
        if getattr(self, "primary_ngeo_auto_ref_above_threshold", True):
            threshold = float(self.get_cherenkov_threshold_kinetic_energy())
            min_ref = max(
                threshold + float(getattr(self, "primary_ngeo_ref_threshold_margin_MeV", 25.0)),
                threshold * float(getattr(self, "primary_ngeo_ref_threshold_factor", 2.0)),
            )
            if E_ref <= min_ref:
                E_ref = min_ref
        return float(E_ref)


    def primary_ngeo_normalization(self):
        """
        Fixed convention factor for N_geo.

        Cached globally because this scalar normalization is otherwise
        recomputed for every Minuit FCN call even though it only depends on the
        optical constants and chosen reference point.
        """
        E_ref = self.primary_ngeo_reference_energy()
        r_ref = float(self.primary_ngeo_ref_r_mm)
        key = (
            self.particle_name,
            float(self.n),
            float(self.particle_mass),
            float(self.primary_ngeo_pmt_radius_mm),
            E_ref,
            r_ref,
        )

        cached = _PRIMARY_NGEO_NORM_CACHE.get(key)
        if cached is not None:
            return cached

        raw_ref = self.primary_ngeo_falloff_raw(
            np.asarray([E_ref], dtype=np.float64),
            np.asarray([r_ref], dtype=np.float64),
        )[0]

        if not np.isfinite(raw_ref) or raw_ref <= 0.0:
            # This should no longer happen for protons because the reference
            # energy is moved above threshold, but keep a clear diagnostic state
            # instead of silently producing a tiny primary component.
            norm = 1.0
        else:
            norm = float(n_from_E_r(E_ref, r_ref) / raw_ref)

        self._last_primary_ngeo_norm = float(norm)
        self._last_primary_ngeo_ref_energy_MeV = float(E_ref)
        self._last_primary_ngeo_raw_ref = float(raw_ref)

        _PRIMARY_NGEO_NORM_CACHE[key] = norm
        return norm


    def primary_ngeo_falloff(self, E_MeV, r_mm):
        """
        Normalized analytic primary-muon falloff.

        Use this in place of n_from_E_r(E_b, r) for the primary muon term.
        """
        return self.primary_ngeo_normalization() * self.primary_ngeo_falloff_raw(E_MeV, r_mm)


    def get_physical_stop_length_from_cherenkov_length(self):
        if getattr(self, "track_end_mode", "threshold") == "abrupt":
            return float(self.length)
        return self.length + self.muon_subthreshold_range_mm

    def get_cherenkov_threshold_kinetic_energy(self):
        return cherenkov_threshold_kinetic_mev(float(self.particle_mass), n=float(self.n))

    def beta2_from_K(self, K, mass):
        K = np.asarray(K, dtype=np.float64)
        gamma = 1.0 + K / mass
        return np.clip(1.0 - 1.0 / gamma**2, 0.0, 1.0)


    def frank_tamm_factor(self, K, mass):
        beta2 = self.beta2_from_K(K, mass)
        out = 1.0 - 1.0 / (self.n**2 * np.maximum(beta2, 1e-30))
        return np.where(beta2 * self.n**2 > 1.0, np.maximum(out, 0.0), 0.0)


    def electron_cherenkov_threshold(self):
        m_e = 0.51099895
        beta_thr = 1.0 / self.n
        gamma_thr = 1.0 / np.sqrt(1.0 - beta_thr**2)
        return m_e * (gamma_thr - 1.0)


    def electron_range_cm(self, T):
        """
        Approximate electron CSDA range in water.

        Returns range in cm. Since liquid water has rho ~= 1 g/cm^3,
        a mass range in g/cm^2 is numerically equal to a length in cm.

        Uses a Katz-Penfold-style empirical approximation.
        For serious production use, replace this with ESTAR interpolation.
        """
        T = np.asarray(T, dtype=np.float64)
        T_safe = np.maximum(T, 1e-12)

        out = np.zeros_like(T_safe)

        low = T_safe <= 2.5

        # Corrected low-energy form:
        out[low] = 0.412 * T_safe[low] ** (
            1.265 - 0.0954 * np.log(T_safe[low])
        )

        # Higher-energy empirical form.
        out[~low] = 0.530 * T_safe[~low] - 0.106

        return np.maximum(out, 0.0)


    def Tmax_delta_e(self, K_mu):
        m_e = 0.51099895
        m_mu = self.mu_mass

        K_mu = np.asarray(K_mu, dtype=np.float64)
        beta2 = self.beta2_from_K(K_mu, m_mu)
        gamma = 1.0 + K_mu / m_mu

        return (
            2.0 * m_e * beta2 * gamma**2
            / (1.0 + 2.0 * gamma * m_e / m_mu + (m_e / m_mu)**2)
        )

    def delta_e_photon_angle_deg(self, K_mu):
        """
        Returns the average photon angle produced by knock-on electrons at a given muon energy
        Should replace this with the actual shape of angular distribution
        """
        theta_max = 55.41
        A = 41.43
        tau = 32.3

        return theta_max - A*np.exp(-K_mu/tau)

    def load_delta_e_angular_pdf_table(self, path):
        self.delta_e_angular_pdf_path = path
        return self

    def evaluate_refined_analytic_delta_dSdu(self, K_mu, cos_forward):
        """
        Evaluate the refined analytic secondary-electron angular/yield model.

        Returns dS_delta/du(K_mu, u), where u = cos_forward is the photon
        direction cosine relative to the primary muon direction.

        Unlike the old factorized model, this table already includes the
        secondary-electron yield and the angular shape together.  It therefore
        replaces

            S_delta(K_mu) * evaluate_delta_e_angular_pdf(K_mu, u)

        in get_delta_e_expected_pes.
        """
        K_mu = np.asarray(K_mu, dtype=np.float64)
        u = np.asarray(cos_forward, dtype=np.float64)

        K_grid, u_grid, table = get_refined_analytic_delta_cache(self.n, projectile_mass=float(self.particle_mass), particle=self.particle_name)

        valid_K = np.isfinite(K_mu)
        # The table now spans the full physical range -1 <= u <= 1 (the knock-on
        # light has a backward hemisphere once multiple scattering is included).
        valid_u = np.isfinite(u) & (u >= -1.0) & (u <= 1.0)

        K_safe = np.where(valid_K, K_mu, K_grid[0])
        u_safe = np.where(np.isfinite(u), u, u_grid[0])

        K_clip = np.clip(K_safe, K_grid[0], K_grid[-1])
        u_clip = np.clip(u_safe, u_grid[0], u_grid[-1])

        iK = np.searchsorted(K_grid, K_clip, side="right") - 1
        iK = np.clip(iK, 0, len(K_grid) - 2)

        K0 = K_grid[iK]
        K1 = K_grid[iK + 1]
        tK = (K_clip - K0) / (K1 - K0 + 1e-300)
        tK = np.clip(tK, 0.0, 1.0)

        du = u_grid[1] - u_grid[0]
        iu = np.floor((u_clip - u_grid[0]) / du).astype(np.int64)
        iu = np.clip(iu, 0, len(u_grid) - 2)

        tu = (u_clip - u_grid[iu]) / (u_grid[iu + 1] - u_grid[iu] + 1e-300)
        tu = np.clip(tu, 0.0, 1.0)

        src_idx = np.arange(K_mu.size)[:, None]

        row0 = table[iK]
        row1 = table[iK + 1]

        p00 = row0[src_idx, iu]
        p01 = row0[src_idx, iu + 1]
        p10 = row1[src_idx, iu]
        p11 = row1[src_idx, iu + 1]

        p0 = p00 + tu * (p01 - p00)
        p1 = p10 + tu * (p11 - p10)
        out = p0 + tK[:, None] * (p1 - p0)

        out[~valid_u] = 0.0
        out[~valid_K, :] = 0.0
        out[~np.isfinite(out)] = 0.0
        out[out < 0.0] = 0.0

        return out


    def _build_delta_source_grid(self):
        """Build the secondary-electron source grid (s_centers, ds_cm, K_mu).

        Extracted verbatim from get_delta_e_expected_pes so the result can be
        cached by length scalars.  Returns (s_centers, ds_cm, K_mu, any_valid)
        with the valid-source mask already applied; any_valid is False when no
        source bin survives (the caller then returns zeros).
        """
        L_ch = max(float(self.length), 0.0)
        if getattr(self, "track_end_mode", "threshold") == "abrupt":
            L_tail = 0.0
        else:
            L_tail = max(float(self.muon_subthreshold_range_mm), 0.0)
        L_stop_for_energy = float(getattr(self, "range_to_threshold_mm", L_ch))
        n_ch = max(1, int(self.n_delta_steps))

        tail_step_mm = max(float(getattr(self, "delta_e_tail_step_mm", 20.0)), 1e-12)
        tail_min_steps = max(1, int(getattr(self, "delta_e_tail_min_steps", 3)))

        if L_ch > 0.0:
            s_edges_ch = np.linspace(0.0, L_ch, n_ch + 1, dtype=np.float64)
        else:
            s_edges_ch = np.array([0.0], dtype=np.float64)

        if L_tail > 0.0:
            n_tail = max(tail_min_steps, int(np.ceil(L_tail / tail_step_mm)))
            s_edges_tail = L_ch + np.linspace(0.0, L_tail, n_tail + 1, dtype=np.float64)[1:]
            s_edges = np.concatenate([s_edges_ch, s_edges_tail])
        else:
            s_edges = s_edges_ch

        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        ds_cm = np.diff(s_edges) / 10.0

        K_mu = np.zeros_like(s_centers, dtype=np.float64)
        above_threshold = s_centers <= L_ch
        below_threshold = ~above_threshold

        if np.any(above_threshold):
            K_mu[above_threshold] = self.muon_energy_at_s_array(s_centers[above_threshold], L_stop_for_energy)

        if np.any(below_threshold):
            K_thr = self.muon_energy_at_s_array(np.array([L_ch], dtype=np.float64), L_stop_for_energy)[0]
            d_post = s_centers[below_threshold] - L_ch
            frac = np.clip(d_post / max(L_tail, 1e-12), 0.0, 1.0)
            K_mu[below_threshold] = K_thr * (1.0 - frac)

        K_mu = np.maximum(K_mu, 0.0)

        valid_src = (
            np.isfinite(K_mu)
            & (K_mu > 0.0)
            & np.isfinite(ds_cm)
            & (ds_cm > 0.0)
        )

        if not np.any(valid_src):
            return None, None, None, False

        s_centers = np.ascontiguousarray(s_centers[valid_src], dtype=np.float64)
        ds_cm = np.ascontiguousarray(ds_cm[valid_src], dtype=np.float64)
        K_mu = np.ascontiguousarray(K_mu[valid_src], dtype=np.float64)
        return s_centers, ds_cm, K_mu, True

    def get_delta_e_expected_pes(
        self,
        p_locations,
        direction_zs,
        start_pos,
        track_dir,
        mpmt_types=None,
        return_times=False,
    ):
        """
        Fast secondary-electron expected PE model.

        For the refined analytic model this uses a Numba-compiled source x PMT
        accumulator.  The physics and algebra are the same as the previous
        vectorized implementation, but it avoids materializing large temporary
        matrices for dx/dy/dz/r/cost/optical_corr/forward_kernel/delta_contrib.

        """

        p_locations = np.asarray(p_locations, dtype=np.float64)
        direction_zs = np.asarray(direction_zs, dtype=np.float64)
        start_pos = np.asarray(start_pos, dtype=np.float64)
        track_dir = np.asarray(track_dir, dtype=np.float64)
        track_dir = track_dir / np.linalg.norm(track_dir)

        n_pmts = p_locations.shape[0]

        # The secondary-electron *source* grid (s_centers, ds_cm, K_mu) depends
        # only on the track length scalars and particle, not on the vertex or
        # direction.  During a fit (and especially the seed prescan, which sweeps
        # ~15k vertex/direction combinations over only ~160 unique length pairs)
        # this is recomputed far more often than it changes, so cache it.
        # Bit-identical: identical scalar key -> identical arrays.
        L_ch = max(float(self.length), 0.0)
        L_stop_for_energy = float(getattr(self, "range_to_threshold_mm", L_ch))
        _src_key = (
            self.particle_name,
            L_ch,
            L_stop_for_energy,
            # SMOOTH-NLL: K(s) values depend on the table mode, so the cached
            # source grid must be keyed by it (prevents stale cross-mode reuse
            # when smooth_tables is toggled on a live emitter, e.g. in A/B tests).
            bool(getattr(self, "smooth_tables", True)),
            getattr(self, "track_end_mode", "threshold"),
            int(self.n_delta_steps),
            float(getattr(self, "muon_subthreshold_range_mm", 0.0)),
            float(getattr(self, "delta_e_tail_step_mm", 20.0)),
            int(getattr(self, "delta_e_tail_min_steps", 3)),
        )
        _src_cached = self._delta_src_grid_cache.get(_src_key) if getattr(
            self, "_delta_src_grid_cache", None) is not None else None

        if _src_cached is not None:
            s_centers, ds_cm, K_mu, _any_valid = _src_cached
            if not _any_valid:
                zeros = np.zeros(n_pmts, dtype=np.float64)
                if return_times:
                    return zeros, np.full(n_pmts, np.nan, dtype=np.float64)
                return zeros
        else:
            s_centers, ds_cm, K_mu, _any_valid = self._build_delta_source_grid()
            if self._delta_src_grid_cache is None:
                self._delta_src_grid_cache = {}
            if len(self._delta_src_grid_cache) >= 4096:
                self._delta_src_grid_cache.clear()
            self._delta_src_grid_cache[_src_key] = (s_centers, ds_cm, K_mu, _any_valid)
            if not _any_valid:
                zeros = np.zeros(n_pmts, dtype=np.float64)
                if return_times:
                    return zeros, np.full(n_pmts, np.nan, dtype=np.float64)
                return zeros

        # ------------------------------------------------------------
        # DIAGNOSTIC ONLY:
        # Keep the secondary-electron yield and muon-energy assignment
        # fixed, but project the light as if it were emitted downstream
        # from the parent muon point.
        #
        # This tests whether spatially collapsing electron range back to
        # the muon point is responsible for underfilling the ring center.
        # ------------------------------------------------------------
#         delta_shift_mm = float(getattr(self, "delta_e_test_forward_shift_mm", 0.0))
#         s_centers_for_projection = np.ascontiguousarray(
#             s_centers + delta_shift_mm,
#             dtype=np.float64,
#         )
        
        ###
        

        K_grid, u_grid, table = get_refined_analytic_delta_cache(self.n, projectile_mass=float(self.particle_mass), particle=self.particle_name)
       
        K_grid = np.ascontiguousarray(K_grid, dtype=np.float64)
        u_grid = np.ascontiguousarray(u_grid, dtype=np.float64)
        table = np.ascontiguousarray(table, dtype=np.float64)
        
        
        # ------------------------------------------------------------------
        # DEBUG / DIAGNOSTIC ONLY:
        # Modify the refined secondary-electron angular table shape.
        # This tests whether the electron light is too broad in u.
        #
        # Important:
        #   If delta_e_debug_preserve_yield=True, each K row is renormalized so
        #   the total secondary-electron yield S_delta(K) stays approximately fixed.
        #   Therefore this tests angular broadness, not total electron intensity.
        # ------------------------------------------------------------------
        u_power = float(getattr(self, "delta_e_debug_u_power", 0.0))
        u_min = float(getattr(self, "delta_e_debug_u_min", 0.0))
        preserve_yield = bool(getattr(self, "delta_e_debug_preserve_yield", True))

        if (u_power != 0.0) or (u_min > 0.0):
            table_mod = np.array(table, dtype=np.float64, copy=True)

            du = float(u_grid[1] - u_grid[0])
            row_yield_before = np.sum(table_mod, axis=1) * du

            if u_min > 0.0:
                table_mod[:, u_grid < u_min] = 0.0

            if u_power != 0.0:
                weights = np.clip(u_grid, 0.0, 1.0) ** u_power
                table_mod *= weights[None, :]

            if preserve_yield:
                row_yield_after = np.sum(table_mod, axis=1) * du
                renorm = np.divide(
                    row_yield_before,
                    row_yield_after,
                    out=np.ones_like(row_yield_before),
                    where=(row_yield_after > 0.0) & np.isfinite(row_yield_after),
                )
                table_mod *= renorm[:, None]

            table = table_mod
        ###

        if mpmt_types is None:
            mpmt_codes = np.full(n_pmts, -1, dtype=np.int8)
        else:
            mpmt_codes = _encode_mpmt_types(mpmt_types)
            mpmt_codes = np.asarray(mpmt_codes, dtype=np.int8)
            if mpmt_codes.ndim != 1 or mpmt_codes.size != n_pmts:
                mpmt_codes = np.broadcast_to(mpmt_codes, (n_pmts,)).astype(np.int8, copy=False)

        rel_table = _get_rel_eff_stack()
        rel_table = np.ascontiguousarray(rel_table, dtype=np.float64)

        mu_delta, t_delta = _accumulate_refined_delta_numba(
            np.ascontiguousarray(p_locations, dtype=np.float64),
            np.ascontiguousarray(direction_zs, dtype=np.float64),
            np.ascontiguousarray(start_pos, dtype=np.float64),
            np.ascontiguousarray(track_dir, dtype=np.float64),
            s_centers,
            ds_cm,
            K_mu,
            K_grid,
            u_grid,
            table,
            np.ascontiguousarray(mpmt_codes, dtype=np.int8),
            rel_table,
            bool((mpmt_types is not None) and getattr(self, "delta_e_apply_mpmt_eff_by_source", True)),
            bool(getattr(self, "delta_e_use_finite_disk_solid_angle", True)),
            float(getattr(self, "delta_e_distance_pmt_radius_mm", 37.0)),
            float(getattr(self, "delta_e_distance_ref_r_mm", 1000.0)),
            float(getattr(self, "delta_e_distance_power", 2.0)),
            float(getattr(self, "analytic_delta_scale", 1.0)),
            float(getattr(self, "delta_e_source_k_power", 0.0)),
            float(getattr(self, "delta_e_source_k_ref_MeV", 100.0)),
            float(getattr(self, "delta_e_source_k_floor_MeV", 25.0)),
            float(self.intensity),
            float(self.starting_time),
            float(self.v),
            float(self.n),
            float(self.c),
            float(getattr(self, "delta_e_time_offset_ns", 0.0)),
            bool(return_times),
            (_POWER_LAW_LUT if getattr(self, "use_delta_power_law_lut", True)
             else _POWER_LAW_LUT_EMPTY),
            float(getattr(self, "delta_e_cost_soft", 0.0)),
            np.ascontiguousarray(
                start_pos[None, :] + (s_centers - 5.0 * ds_cm)[:, None] * track_dir[None, :],
                dtype=np.float64),
            np.ascontiguousarray(
                start_pos[None, :] + (s_centers + 5.0 * ds_cm)[:, None] * track_dir[None, :],
                dtype=np.float64),
            int(1 if (getattr(self, "smooth_tables", True)
                      if getattr(self, "delta_e_segment_gate", None) is None
                      else bool(getattr(self, "delta_e_segment_gate"))) else 0),
        )

        if return_times:
            return mu_delta, t_delta
        return mu_delta

    def get_expected_pes_ts(
        self,
        wcd,
        s,
        p_locations,
        direction_zs,
        mpmt_types,
        obs_pes,
        need_times=True,
    ):

        """
        Expected PE and first-hit-time model used by the fit.

        The heavy cone-collapse work is delegated to the optimized solver in
        particle_cherenkov_model.py.
        """
        pmt_radius = _get_pmt_radius_cached(wcd) + 20 # Add 20 mm for additional reflector surface area

        p_locations = np.asarray(p_locations, dtype=np.float64)
        direction_zs = np.asarray(direction_zs, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        obs_pes = np.asarray(obs_pes, dtype=np.float64)

        # Convert mPMT type strings to integer codes once per geometry object.
        # If your fit creates a fresh Emitter each FCN call this still saves the
        # source x PMT tiling inside the secondary model; if the Emitter is
        # reused, this also saves the string comparisons.
        geom_key = (id(mpmt_types), np.shape(mpmt_types))
        if geom_key == self._last_geometry_cache_key and self._last_mpmt_type_codes is not None:
            mpmt_codes = self._last_mpmt_type_codes
        else:
            mpmt_codes = _encode_mpmt_types(mpmt_types)
            self._last_geometry_cache_key = geom_key
            self._last_mpmt_type_codes = mpmt_codes

        n_pmts = s.size

        start_pos = np.asarray(self.start_coord, dtype=np.float64)
        track_dir = np.asarray(self.direction, dtype=np.float64)
        track_dir = track_dir / np.linalg.norm(track_dir)

        scale = np.zeros(n_pmts, dtype=np.float64)
        s_b = np.zeros(n_pmts, dtype=np.float64)
        E_b = np.zeros(n_pmts, dtype=np.float64)

        collapse_mask = s > -200.0
        idx = np.flatnonzero(collapse_mask)

        if idx.size:
            scale_sub, s_b_sub, E_b_sub = find_scale_for_pmts(
                pmt_pos=p_locations[idx],
                start_pos=start_pos,
                track_dir=track_dir,
                s_a_mm=0.001,
                s_max_mm=self.length,
                theta_c_func=theta_c_func,
                range_stop_mm=float(getattr(self, "range_to_threshold_mm", self.length)),
                n_scan=150,
                near_cross_tol=float(getattr(self, "primary_soft_cone_sigma_rad", 0.03)),#0.02,
                particle=self.particle_name,
                particle_mass=float(self.particle_mass),
                n_water=float(self.n),
                subgrid_refine=bool(getattr(self, "use_subgrid_refine", True)),
                legacy_grid=not bool(getattr(self, "smooth_tables", True)),
            )

            scale[idx] = scale_sub
            s_b[idx] = s_b_sub
            E_b[idx] = E_b_sub

        use_collapse = scale > 0.0

        if getattr(self, "use_fused_primary", True) and getattr(self, "use_analytic_primary_ngeo", True):
            # Single-pass fused primary path: collapses ~15 full-length numpy
            # operations into one compiled loop with no intermediate arrays.
            # Numerically equivalent to the vectorized path below to ~1e-12.
            Eg_dedx, dg_dedx = _get_particle_stopping_power_table(self.particle_name)
            _ngeo_norm = self.primary_ngeo_normalization()
            _rel_stack = _get_rel_eff_stack()
            mu_primary, t_primary_fused = _fused_primary_kernel(
                np.ascontiguousarray(p_locations, dtype=np.float64),
                np.ascontiguousarray(direction_zs, dtype=np.float64),
                np.ascontiguousarray(s, dtype=np.float64),
                np.ascontiguousarray(scale, dtype=np.float64),
                np.ascontiguousarray(s_b, dtype=np.float64),
                np.ascontiguousarray(E_b, dtype=np.float64),
                np.ascontiguousarray(start_pos, dtype=np.float64),
                np.ascontiguousarray(track_dir, dtype=np.float64),
                np.ascontiguousarray(mpmt_codes, dtype=np.int8),
                np.ascontiguousarray(_rel_stack, dtype=np.float64),
                np.ascontiguousarray(Eg_dedx, dtype=np.float64),
                np.ascontiguousarray(dg_dedx, dtype=np.float64),
                float(_ngeo_norm),
                float(self.primary_ngeo_pmt_radius_mm),
                float(pmt_radius),
                float(self.n),
                float(self.mu_mass),
                float(self.intensity),
                float(self.starting_time),
                float(self.v),
                float(self.c),
                bool(need_times),
                float(getattr(self, "primary_cost_soft", 0.0)),
                int(1 if getattr(self, "primary_cost_soft_centered", False) else 0),
            )
            t_primary = t_primary_fused if bool(need_times) else None
        else:
            s_eff = np.where(use_collapse, s_b, s)

            front_mask = s_eff < pmt_radius
            if np.any(front_mask):
                scale[front_mask] *= (s_eff[front_mask] + pmt_radius) / (2.0 * pmt_radius)

            valid_s = s_eff >= -pmt_radius
            scale *= valid_s
            s_eff = np.where(valid_s, s_eff, 0.0)

            e_pos = start_pos[None, :] + s_eff[:, None] * track_dir[None, :]
            dx = p_locations[:, 0] - e_pos[:, 0]
            dy = p_locations[:, 1] - e_pos[:, 1]
            dz = p_locations[:, 2] - e_pos[:, 2]
            r = np.sqrt(dx * dx + dy * dy + dz * dz) + 0.01

            cost = -(dx * direction_zs[:, 0] + dy * direction_zs[:, 1] + dz * direction_zs[:, 2]) / r
            valid_cost = np.isfinite(cost) & (cost > 0.0)
            scale *= valid_cost

            active = (scale > 0.0) & valid_cost

            pwr_corr = np.zeros(n_pmts, dtype=np.float64)
            if np.any(active):
                pwr_corr[active] = self.power_law(cost[active])

            corr = np.zeros(n_pmts, dtype=np.float64)
            if np.any(active):
                if getattr(self, "use_analytic_primary_ngeo", True):
                    corr[active] = self.primary_ngeo_falloff(E_b[active], r[active]) * pwr_corr[active]
                else:
                    corr[active] = n_from_E_r(E_b[active], r[active]) * pwr_corr[active]

            rel_mpmt_scaling = _interp_rel_mpmt_eff_from_codes(
                cost,
                mpmt_codes,
                fill_empty=1.0,
            )

            mu_primary = self.intensity * corr * scale * rel_mpmt_scaling

            if bool(need_times):
                t_light_primary = r * self.n / self.c
                t_emitter_primary = s_eff / self.v
                t_primary = self.starting_time + t_emitter_primary + t_light_primary
            else:
                t_primary = None

        mu_delta = None
        t_delta = None

        if self.enable_delta_e and self.delta_e_scale != 0.0:
            if bool(need_times) and getattr(self, "use_delta_e_timing", True):
                mu_delta, t_delta = self.get_delta_e_expected_pes(
                    p_locations=p_locations,
                    direction_zs=direction_zs,
                    start_pos=start_pos,
                    track_dir=track_dir,
                    mpmt_types=mpmt_codes,
                    return_times=True,
                )
            else:
                mu_delta = self.get_delta_e_expected_pes(
                    p_locations=p_locations,
                    direction_zs=direction_zs,
                    start_pos=start_pos,
                    track_dir=track_dir,
                    mpmt_types=mpmt_codes,
                    return_times=False,
                )

            mu_delta_scaled = self.delta_e_scale * mu_delta
            mean_pes_raw = mu_primary + mu_delta_scaled
        else:
            mu_delta_scaled = np.zeros_like(mu_primary)
            mean_pes_raw = mu_primary


        obs_mean = float(np.mean(obs_pes))
        raw_mean = float(np.mean(mean_pes_raw))
        norm = obs_mean / raw_mean if raw_mean > 0.0 else 1.0
        mean_pes = mean_pes_raw * norm
        #mean_pes[mean_pes < 1e-3] = 0.0
        # Test a small physical floor
        mean_pes = np.maximum(mean_pes, self.charge_floor_pe)

        # Store component diagnostics only when explicitly requested.
        # The old unconditional path copied several nPMT-length arrays every FCN
        # call, which is useful for debugging but pure overhead in production.
        if getattr(self, "store_expected_component_diagnostics", False):
            sum_primary = float(np.sum(mu_primary))
            sum_delta = float(np.sum(mu_delta_scaled))
            sum_total = float(np.sum(mean_pes_raw))
            self._last_expected_components = {
                "mu_primary_raw": mu_primary.copy(),
                "mu_delta_raw": mu_delta_scaled.copy(),
                "mean_pes_raw": mean_pes_raw.copy(),
                "mean_pes": mean_pes.copy(),
                "norm": float(norm),
                "obs_mean": float(obs_mean),
                "raw_mean": float(raw_mean),
                "sum_primary_raw": sum_primary,
                "sum_delta_raw": sum_delta,
                "primary_fraction_raw": (sum_primary / sum_total) if sum_total > 0.0 else np.nan,
                "primary_ngeo_ref_energy_MeV": float(getattr(self, "_last_primary_ngeo_ref_energy_MeV", np.nan)),
                "primary_ngeo_norm": float(getattr(self, "_last_primary_ngeo_norm", np.nan)),
                "primary_ngeo_raw_ref": float(getattr(self, "_last_primary_ngeo_raw_ref", np.nan)),
            }

        # Expected time model.
        #
        # If secondary-electron timing is enabled, return the PE-weighted mean
        # time of the primary and secondary components:
        #
        #           / (mu_primary + mu_delta)
        #
        # This is the model that corresponds to using charge-weighted mean hit
        # times in the data.  The event-level normalization cancels in the time
        # weights, so the raw component PEs are used here.
        if not bool(need_times):
            # Charge-only caller: keep the API shape, but avoid any time algebra.
            t_hits = np.empty_like(mean_pes)
        elif (
            self.enable_delta_e
            and self.delta_e_scale != 0.0
            and getattr(self, "use_delta_e_timing", True)
            and t_delta is not None
        ):
            t_hits = t_primary.copy()

            valid_delta_time = np.isfinite(t_delta) & (mu_delta_scaled > 0.0)
            denom = mu_primary + mu_delta_scaled
            mix = valid_delta_time & np.isfinite(t_primary) & (denom > 0.0)

            t_hits[mix] = (
                mu_primary[mix] * t_primary[mix]
                + mu_delta_scaled[mix] * t_delta[mix]
            ) / denom[mix]
        else:
            t_hits = t_primary

        return mean_pes, t_hits

    @staticmethod
    def get_pmt_placements(event, wcd, place_info):
        """
        Cache-friendly PMT geometry extraction.

        The fitter usually calls this once per detector configuration, so a
        simple straight loop is enough here.
        """
        p_locations = []
        direction_zs = []
        mpmt_slots = []

        for i_mpmt in range(event.n_mpmt):
            if not event.mpmt_status[i_mpmt]:
                continue

            mpmt = wcd.mpmts[i_mpmt]
            if mpmt is None:
                continue

            for i_pmt in range(event.npmt_per_mpmt):
                if not event.pmt_status[i_mpmt][i_pmt]:
                    continue

                pmt = mpmt.pmts[i_pmt]
                if pmt is None:
                    continue

                placement = pmt.get_placement(place_info, wcd)
                p_locations.append(np.asarray(placement["location"], dtype=np.float64))
                direction_zs.append(np.asarray(placement["direction_z"], dtype=np.float64))
                mpmt_slots.append(i_mpmt)

        return np.asarray(p_locations, dtype=np.float64), np.asarray(direction_zs, dtype=np.float64), np.asarray(mpmt_slots, dtype=np.int64)

    def get_cone_can_intersection_points(self,
            r: float,  # cylinder radius
            ht: float, hb: float,  # top and bottom endcap y (ht > hb)
            n: int,  # number of azimuth samples
            flen: float = 0.  # fractional position along cone axis for apex (0=start, 1=end)
    ) -> List[Tuple[float, float, float]]:
        """
        Return n+1 intersection points (last repeats first) of a right circular cone
        (apex at (x0,y0,z0), axis with direction cosines (cx,cy,cz), half-angle q)
        with the finite cylinder (axis = y, radius r, endcaps at y = hb and y = ht).

        For each azimuth ray on the cone, there is exactly one intersection with the
        cylindrical can (your assumption). If the side intersection is outside the
        y-interval, the intersection is on the corresponding endcap.

        Returns: list of (xi, yi, zi), length n+1 with points[0] == points[-1].
        """
        (x0, y0, z0) = self.start_coord + flen * self.length * np.array(self.direction)
        (cx, cy, cz) = self.direction
        q = np.arccos(self.cos_tq)  # half-angle in radians
        if not (0.0 < q < 0.5 * np.pi):
            raise ValueError("Cone half-angle q must be in (0, pi/2) radians.")
        if ht <= hb:
            raise ValueError("Cylinder top ht must be greater than bottom hb.")
        if r <= 0.0:
            raise ValueError("Cylinder radius r must be positive.")
        if n < 3:
            raise ValueError("Number of azimuth samples n must be at least 3.")

        eps = 1e-12

        # Normalize axis c
        c = np.array([cx, cy, cz], dtype=float)
        c_norm = np.linalg.norm(c)
        if c_norm == 0:
            raise ValueError("Axis direction (cx,cy,cz) must be nonzero.")
        c = c / c_norm

        # Build orthonormal basis {u, v, c} with u,v ⟂ c
        # Choose a helper not nearly parallel to c
        helper = np.array([1.0, 0.0, 0.0]) if abs(c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(c, helper)
        u_norm = np.linalg.norm(u)
        if u_norm < eps:
            helper = np.array([0.0, 0.0, 1.0])
            u = np.cross(c, helper)
            u_norm = np.linalg.norm(u)
            if u_norm < eps:
                raise ValueError("Failed to construct basis perpendicular to axis.")
        u = u / u_norm
        v = np.cross(c, u)  # already unit

        # Precompute constants
        cosq = np.cos(q)
        sinq = np.sin(q)
        apex = np.array([x0, y0, z0], dtype=float)

        # Azimuth samples
        theta = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
        ct = np.cos(theta)
        st = np.sin(theta)

        # Generator directions (unit) for each azimuth
        dirs = (cosq * c)[None, :] + (sinq * ct)[:, None] * u[None, :] + (sinq * st)[:, None] * v[None, :]
        dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]

        # Quadratic for intersection with infinite cylinder x^2 + z^2 = r^2:
        # a t^2 + b t + c0 = 0 for (x,y,z) = apex + t*dir
        a = dx * dx + dz * dz
        b = 2.0 * (x0 * dx + z0 * dz)
        c0 = x0 * x0 + z0 * z0 - r * r

        # Discriminant and roots (vectorized)
        disc = np.maximum(0.0, b * b - 4.0 * a * c0)  # clamp tiny negatives to 0
        sqrt_disc = np.sqrt(disc)
        denom = 2.0 * a

        # Two roots; pick the smallest positive t
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom

        # Mask out non-forward intersections; choose min positive
        t_candidates = np.stack([np.where(t1 > eps, t1, np.inf),
                                 np.where(t2 > eps, t2, np.inf)], axis=0)
        t_side = np.min(t_candidates, axis=0)

        # Side hit position
        x_side = x0 + t_side * dx
        y_side = y0 + t_side * dy
        z_side = z0 + t_side * dz

        # Decide final intersection:
        # - if hb <= y_side <= ht: keep side hit
        # - if y_side < hb: snap to bottom cap at y=hb
        # - if y_side > ht: snap to top cap at y=ht
        y_plane = np.where(y_side < hb, hb, np.where(y_side > ht, ht, np.nan))

        # For plane hits, recompute t from y = y_plane
        # Guard against |dy| ~ 0 by nudging with eps; your uniqueness guarantee
        # implies this division should be safe, but we keep it numerically stable.
        dy_safe = np.where(np.abs(dy) < eps, np.sign(dy) * eps, dy)
        t_cap = (y_plane - y0) / dy_safe  # valid only where y_plane is not nan

        # Compute cap positions
        x_cap = x0 + t_cap * dx
        z_cap = z0 + t_cap * dz

        # Choose between side and cap per-ray
        use_cap = ~np.isnan(y_plane)
        xi = np.where(use_cap, x_cap, x_side)
        yi = np.where(use_cap, y_plane, y_side)
        zi = np.where(use_cap, z_cap, z_side)

        # Stack and append first point to close the loop
        pts = np.stack([xi, yi, zi], axis=1)
        pts_closed = np.vstack([pts, pts[:1]])

        # Convert to list of tuples
        return [tuple(row) for row in pts_closed]

