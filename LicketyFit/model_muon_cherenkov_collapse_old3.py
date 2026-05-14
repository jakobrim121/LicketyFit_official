import os
from functools import lru_cache

import numpy as np
from numba import njit
import pickle


# -----------------------------------------------------------------------------
# Table loading
# -----------------------------------------------------------------------------
# These files live in the user's normal CERN environment. The helper below also
# allows local fallbacks so this module can still be imported outside that setup.
# -----------------------------------------------------------------------------
def _load_first_existing(paths, *, allow_pickle=False):
    for path in paths:
        if path and os.path.exists(path):
            return np.load(path, allow_pickle=allow_pickle)
    raise FileNotFoundError(
        "Could not find any of the required lookup tables. Checked:\n"
        + "\n".join(str(p) for p in paths if p)
    )


_TABLES = None


def _ensure_tables_loaded():
    global _TABLES
    if _TABLES is not None:
        return _TABLES

    base_dir = os.path.dirname(__file__)

    cang_paths = [
        "../tables/mu_cAng_vs_E_n1344.npy",
       # os.path.join(base_dir, "mu_cAng_vs_E_n1344.npy"),
    ]
    evsd_paths = [
        "/eos/experiment/wcte/wcte_tests/mPMT_led_events/LicketyFit_stuff/E_vs_dist_cm.npy",
    ]
    odist_paths = [
        "../tables/overall_distances_cm.npy",
    ]
    rdep_paths = [
        "../tables/n_vs_E_vs_r_v1.npy",
    ]
    sec_e_ang = [
        "../tables/delta_e_angular_pdf_table.npz",
    ]
    rel_mpmt_eff_path = "../tables/rel_mpmt_eff.dict"
    
   

    c_ang_vs_E = _load_first_existing(cang_paths, allow_pickle=False)
    E_vs_dist = _load_first_existing(evsd_paths, allow_pickle=True)
    overall_distances = _load_first_existing(odist_paths, allow_pickle=False) * 10.0
    #rdep = _load_first_existing(rdep_paths, allow_pickle=False)

    # Store the jagged lookup rows exactly once. The fit repeatedly picks a
    # single row from these tables, so loading them repeatedly is wasted work.
    energy_rows = [np.asarray(row[:, 1], dtype=np.float64) for row in E_vs_dist]
    distance_rows = [np.asarray(row[:, 0], dtype=np.float64) * 10.0 for row in E_vs_dist]
    
    with open(rel_mpmt_eff_path, 'rb') as f:
        
        rel_mpmt_eff = pickle.load(f)
        
    tri_exsitu = rel_mpmt_eff['tri_exsitu']
    tri_insitu = rel_mpmt_eff['tri_insitu']
    
    wut_insitu = rel_mpmt_eff['wut_insitu']
    wut_exsitu = rel_mpmt_eff['wut_exsitu']

    _TABLES = {
        "c_ang": np.asarray(c_ang_vs_E[:, 0], dtype=np.float64),
        "energy_for_angle": np.asarray(c_ang_vs_E[:, 1], dtype=np.float64),
        "overall_distances": np.asarray(overall_distances, dtype=np.float64),
        "energy_rows": energy_rows,
        "distance_rows": distance_rows,
        "tri_exsitu": tri_exsitu,
        "tri_insitu": tri_insitu,
        "wut_insitu": wut_insitu,
        "wut_exsitu": wut_exsitu
#         "r_rdep": np.asarray(rdep[:, 0], dtype=np.float64),
#         "E_rdep": np.asarray(rdep[:, 1], dtype=np.float64),
#         "n_rdep": np.asarray(rdep[:, 2], dtype=np.float64),
    }
    return _TABLES


def get_cerenkov_angle_table():
    t = _ensure_tables_loaded()
    return t["c_ang"], t["energy_for_angle"]


def get_energy_distance_tables():
    t = _ensure_tables_loaded()
    return t["overall_distances"], t["energy_rows"], t["distance_rows"]


def get_rel_mpmt_eff_tables():
    t = _ensure_tables_loaded()
    return t["tri_exsitu"], t["tri_insitu"], t["wut_insitu"], t["wut_exsitu"]

# def get_rdep_tables():
#     t = _ensure_tables_loaded()
#     return t["r_rdep"], t["E_rdep"], t["n_rdep"]


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------
def cherenkov_scale_muon_water(T_MeV, n=1.344, hard_saturate_above_MeV=None):
    """
    Dimensionless Cherenkov light-yield scale factor for a muon in water.

    This matches the original Frank-Tamm-based logic, but keeps the code compact
    and vectorized.
    """
    m_mu = 105.6583755  # MeV/c^2

    T = np.asarray(T_MeV, dtype=np.float64)
    gamma = (T + m_mu) / m_mu
    beta2 = 1.0 - 1.0 / gamma**2
    beta2 = np.clip(beta2, 0.0, None)

    ft = 1.0 - 1.0 / (beta2 * n**2)
    ft_inf = 1.0 - 1.0 / (n**2)

    scale = np.zeros_like(T, dtype=np.float64)
    mask = beta2 * n**2 > 1.0
    scale[mask] = ft[mask] / ft_inf
    scale = np.clip(scale, 0.0, 1.0)

    if hard_saturate_above_MeV is not None:
        scale[T >= hard_saturate_above_MeV] = 1.0

    return scale


@njit(cache=True)
def _theta_interp_numba(energy_grid, angle_grid, x):
    out = np.empty_like(x)
    n = energy_grid.size

    for i in range(x.size):
        xi = x[i]
        if xi <= energy_grid[0]:
            out[i] = angle_grid[0]
            continue
        if xi >= energy_grid[n - 1]:
            out[i] = angle_grid[n - 1]
            continue

        idx = np.searchsorted(energy_grid, xi)
        x0 = energy_grid[idx - 1]
        x1 = energy_grid[idx]
        y0 = angle_grid[idx - 1]
        y1 = angle_grid[idx]
        w = (xi - x0) / (x1 - x0)
        out[i] = y0 + w * (y1 - y0)

    return out


def theta_c_func(angles, E, E_k):
    """
    Vectorized Cherenkov-angle interpolation with no debug printing.

    Returning clipped endpoint values is substantially cheaper than raising
    exceptions inside a fit loop.
    """
    angles = np.asarray(angles, dtype=np.float64)
    E = np.asarray(E, dtype=np.float64)
    E_k = np.asarray(E_k, dtype=np.float64)

    flat = E_k.ravel()
    theta = _theta_interp_numba(E, angles, flat)
    return theta.reshape(E_k.shape)


@njit(cache=True)
def _nearest_index_1d(arr, x):
    idx = np.searchsorted(arr, x)
    if idx <= 0:
        return 0
    if idx >= arr.size:
        return arr.size - 1

    left = arr[idx - 1]
    right = arr[idx]
    if (x - left) <= (right - x):
        return idx - 1
    return idx


@njit(cache=True)
def _scale_from_energy_scalar(T_MeV, n=1.344):
    m_mu = 105.6583755
    gamma = (T_MeV + m_mu) / m_mu
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    if beta2 < 0.0:
        beta2 = 0.0

    if beta2 * n * n <= 1.0:
        return 0.0

    ft = 1.0 - 1.0 / (beta2 * n * n)
    ft_inf = 1.0 - 1.0 / (n * n)

    scale = ft / ft_inf
    if scale < 0.0:
        return 0.0
    if scale > 1.0:
        return 1.0
    return scale


@njit(cache=True)
def _find_scale_kernel(
    pmt_pos,
    start_pos,
    track_dir,
    s_grid,
    theta_c_grid,
    dist_row,
    energy_row,
    s_a_mm,
    s_max_mm,
    near_cross_tol,
):
    """
    Hybrid binary-search version.

    It is exactly equivalent to the original linear scan in the no-cross case,
    and only uses binary search when a genuine crossing exists.
    """
    n_pmts = pmt_pos.shape[0]
    n_scan = s_grid.size

    scale = np.zeros(n_pmts, dtype=np.float64)
    s_b = np.full(n_pmts, s_a_mm, dtype=np.float64)
    E_b = np.empty(n_pmts, dtype=np.float64)

    tx = track_dir[0]
    ty = track_dir[1]
    tz = track_dir[2]

    denom = s_max_mm - s_a_mm
    if denom <= 0.0:
        denom = 1.0

    for i in range(n_pmts):
        wx = pmt_pos[i, 0] - start_pos[0]
        wy = pmt_pos[i, 1] - start_pos[1]
        wz = pmt_pos[i, 2] - start_pos[2]

        w2 = wx * wx + wy * wy + wz * wz
        u0 = wx * tx + wy * ty + wz * tz

        # Same geometry reduction as your original optimized kernel
        perp2 = w2 - u0 * u0
        if perp2 < 0.0:
            perp2 = 0.0
        perp = np.sqrt(perp2)

        # Endpoints
        parallel_lo = u0 - s_grid[0]
        f_lo = np.arctan2(perp, parallel_lo) - theta_c_grid[0]

        parallel_hi = u0 - s_grid[n_scan - 1]
        f_hi = np.arctan2(perp, parallel_hi) - theta_c_grid[n_scan - 1]

        abs_lo = abs(f_lo)
        abs_hi = abs(f_hi)

        if abs_lo <= abs_hi:
            min_abs_f = abs_lo
            min_idx = 0
        else:
            min_abs_f = abs_hi
            min_idx = n_scan - 1

        found_crossing = False
        cross_idx = 0

        # Match the original linear logic exactly:
        # a crossing only exists if some later point goes from prev_f < 0 to f >= 0.
        # Therefore f_lo >= 0 is NOT a valid crossing.
        if (n_scan >= 2) and (f_lo < 0.0) and (f_hi >= 0.0):
            lo = 0
            hi = n_scan - 1

            while hi - lo > 1:
                mid = (lo + hi) // 2
                parallel_mid = u0 - s_grid[mid]
                f_mid = np.arctan2(perp, parallel_mid) - theta_c_grid[mid]

                abs_mid = abs(f_mid)
                if abs_mid < min_abs_f:
                    min_abs_f = abs_mid
                    min_idx = mid

                if f_mid >= 0.0:
                    hi = mid
                else:
                    lo = mid

            # hi is the first index with f >= 0, so preserve the original j-1 rule
            cross_idx = hi - 1
            if cross_idx < 0:
                cross_idx = 0
            found_crossing = True
        else:
            # No crossing case: preserve original behavior exactly by scanning all points
            for j in range(1, n_scan - 1):
                parallel = u0 - s_grid[j]
                f = np.arctan2(perp, parallel) - theta_c_grid[j]
                af = abs(f)
                if af < min_abs_f:
                    min_abs_f = af
                    min_idx = j

        if found_crossing:
            sb = s_grid[cross_idx]
            eidx = _nearest_index_1d(dist_row, sb - s_a_mm)
            Eb = energy_row[eidx]
            s_b[i] = sb
            E_b[i] = Eb
            scale[i] = _scale_from_energy_scalar(Eb)
        else:
            sb = s_grid[min_idx]
            frac_along = (sb - s_a_mm) / denom

            if (min_abs_f < near_cross_tol) and (frac_along < 0.3):
                eidx = _nearest_index_1d(dist_row, sb - s_a_mm)
                Eb = energy_row[eidx]
                s_b[i] = sb
                E_b[i] = Eb
                scale[i] = 1.0
            else:
                E_b[i] = energy_row[0]
                scale[i] = 0.0

    return scale, s_b, E_b

def find_scale_for_pmts(
    pmt_pos,
    start_pos,
    track_dir,
    s_a_mm,
    s_max_mm,
    theta_c_func,
    mpmt_bool=False,
    n_scan=150,
    near_cross_tol=0.02,
):
    """
    Fast Cherenkov-cone-collapse solver for many PMTs.

    This is a drop-in replacement for the original function. The public argument
    order and names are preserved exactly, while the internal crossing search is
    accelerated.

    Parameters
    ----------
    pmt_pos : array-like, shape (N, 3)
        PMT positions.
    start_pos : array-like, shape (3,)
        Start position of the currently visible track segment.
    track_dir : array-like, shape (3,)
        Unit direction vector of the track.
    s_a_mm : float
        Start of the scan interval along the track.
    s_max_mm : float
        End of the scan interval along the track.
    theta_c_func : callable
        Retained for API compatibility. The compiled helper below performs the
        interpolation directly.
    mpmt_bool : bool, optional
        Retained for API compatibility/debug compatibility.
    n_scan : int, optional
        Number of scan points.
    near_cross_tol : float, optional
        Tolerance for the no-crossing near-match fallback.

    Returns
    -------
    scale : ndarray, shape (N,)
        Cherenkov light-yield scale factors.
    s_b : ndarray, shape (N,)
        Collapse-adjusted emission points.
    E_b : ndarray, shape (N,)
        Corresponding muon kinetic energies.
    """
    del theta_c_func  # kept only for drop-in API compatibility

    tables = _ensure_tables_loaded()
    overall_distances = tables["overall_distances"]
    energy_rows = tables["energy_rows"]
    distance_rows = tables["distance_rows"]
    c_ang = tables["c_ang"]
    energy_for_angle = tables["energy_for_angle"]

    pmt_pos = np.asarray(pmt_pos, dtype=np.float64)
    start_pos = np.asarray(start_pos, dtype=np.float64)
    track_dir = np.asarray(track_dir, dtype=np.float64)
    track_dir = track_dir / np.linalg.norm(track_dir)

    main_idx = _nearest_index_1d(overall_distances, float(s_max_mm))
    dist_row = distance_rows[main_idx]
    energy_row = energy_rows[main_idx]

    s_grid = np.linspace(float(s_a_mm), float(s_max_mm), int(n_scan), dtype=np.float64)

    # Preserve the original table lookup rule used for theta_c_grid:
    # searchsorted + clipping, without later nearest-neighbour refinement.
    ds_mm = s_grid - float(s_a_mm)
    idx = np.searchsorted(dist_row, ds_mm)
    idx = np.clip(idx, 1, dist_row.size - 1)
    E_grid = energy_row[idx]
    E_grid = np.maximum(E_grid, 52.5)

    theta_c_grid = _theta_interp_numba(
        np.asarray(energy_for_angle, dtype=np.float64),
        np.asarray(c_ang, dtype=np.float64),
        np.asarray(E_grid, dtype=np.float64),
    )

    scale, s_b, E_b = _find_scale_kernel(
        np.asarray(pmt_pos, dtype=np.float64),
        np.asarray(start_pos, dtype=np.float64),
        np.asarray(track_dir, dtype=np.float64),
        s_grid,
        theta_c_grid,
        np.asarray(dist_row, dtype=np.float64),
        np.asarray(energy_row, dtype=np.float64),
        float(s_a_mm),
        float(s_max_mm),
        float(near_cross_tol),
    )

    # The original function exposed mpmt_bool only for debug printing.
    # The fast path intentionally stays silent.
    _ = mpmt_bool

    return scale, s_b, E_b


# Backward-compatible aliases. Older exploratory notebooks sometimes imported
# these names even though the current fitter only uses find_scale_for_pmts.
def find_scale_for_pmts_old2(*args, **kwargs):
    return find_scale_for_pmts(*args, **kwargs)


def find_scale_for_pmts_old(*args, **kwargs):
    out = find_scale_for_pmts(*args, **kwargs)
    return out[0], out[1]