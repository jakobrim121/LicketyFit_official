"""
particle_cherenkov_model.py

Particle-aware Cherenkov-cone collapse helpers.

Use set_active_particle() or pass particle=... into find_scale_for_pmts() to choose between:

    muon, pion, kaon, proton

Range tables are expected to use the shared particle table format.
For non-muon particles the default filenames are:

    E_vs_dist_cm_pion.npy       overall_distances_cm_pion.npy
    E_vs_dist_cm_kaon.npy       overall_distances_cm_kaon.npy
    E_vs_dist_cm_proton.npy     overall_distances_cm_proton.npy

The old legacy filenames E_vs_dist_cm.npy and overall_distances_cm.npy are still accepted as fallbacks for muons.
"""

from __future__ import annotations

import math
import os
import pickle
from functools import lru_cache
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numba import njit


# -----------------------------------------------------------------------------
# Particle definitions
# -----------------------------------------------------------------------------
PARTICLE_ALIASES = {
    "mu": "muon",
    "mu-": "muon",
    "mu+": "muon",
    "muon": "muon",
    "pi": "pion",
    "pi-": "pion",
    "pi+": "pion",
    "pion": "pion",
    "k": "kaon",
    "k-": "kaon",
    "k+": "kaon",
    "kaon": "kaon",
    "p": "proton",
    "p+": "proton",
    "proton": "proton",
}

PARTICLE_MASS_MEV = {
    "muon": 105.6583755,
    "pion": 139.57039,
    "kaon": 493.677,
    "proton": 938.27208816,
}

_ACTIVE_PARTICLE = "muon"
_TABLES_BY_PARTICLE = {}
_REL_MPMT_EFF_CACHE = None
# Exact scan-grid reuse is valuable within one minimization because vertex and
# direction probes often share the same length.  It is not useful to retain
# thousands of event-specific floating-point length hypotheses indefinitely.
# The old 4096-entry clear-all cache accumulated roughly O(100 MB) of arrays and
# produced a severe latency cliff when it was cleared during long single-process
# jobs.  A small exact-key LRU preserves all intra-event reuse without the
# long-run memory/deallocation spike.
_SCAN_GRID_CACHE = OrderedDict()
_SCAN_GRID_CACHE_MAX = max(32, int(os.environ.get("LF_SCAN_GRID_CACHE_MAX", "256")))


def canonical_particle_name(particle: str | None = None) -> str:
    if particle is None:
        return _ACTIVE_PARTICLE
    key = str(particle).strip().lower()
    if key not in PARTICLE_ALIASES:
        raise ValueError(
            f"Unknown particle {particle!r}. Use one of: "
            f"{', '.join(sorted(PARTICLE_ALIASES))}"
        )
    return PARTICLE_ALIASES[key]


def set_active_particle(particle: str) -> str:
    """
    Set the process-wide default particle used by legacy calls.

    The batch driver calls this once before constructing the Emitter template.
    New code should prefer passing particle=... explicitly where available.
    """
    global _ACTIVE_PARTICLE
    _ACTIVE_PARTICLE = canonical_particle_name(particle)
    return _ACTIVE_PARTICLE


def get_active_particle() -> str:
    return _ACTIVE_PARTICLE


def particle_mass_mev(particle: str | None = None) -> float:
    return float(PARTICLE_MASS_MEV[canonical_particle_name(particle)])


def cherenkov_threshold_kinetic_mev(particle_or_mass, n: float = 1.344) -> float:
    if isinstance(particle_or_mass, str) or particle_or_mass is None:
        mass = particle_mass_mev(particle_or_mass)
    else:
        mass = float(particle_or_mass)
    beta_thr = 1.0 / float(n)
    gamma_thr = 1.0 / math.sqrt(1.0 - beta_thr * beta_thr)
    return float(mass * (gamma_thr - 1.0))


# -----------------------------------------------------------------------------
# Table loading
# -----------------------------------------------------------------------------
def _load_first_existing(paths, *, allow_pickle=False):
    for path in paths:
        if path and os.path.exists(path):
            return np.load(path, allow_pickle=allow_pickle)
    raise FileNotFoundError(
        "Could not find any of the required lookup tables. Checked:\n"
        + "\n".join(str(p) for p in paths if p)
    )


def _table_dirs():
    """Return table search directories, preferring this self-contained checkout.

    Expected layout:
        LF_multiParticles/
          LicketyFit/
          scripts/
          tables/

    You can override or prepend table directories with LF_TABLE_DIR, using
    os.pathsep-separated entries if needed.
    """
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent

    dirs = []
    for env_name in ("LF_TABLE_DIR", "LF_MULTIPARTICLES_TABLE_DIR"):
        env = os.environ.get(env_name)
        if env:
            dirs.extend([d for d in env.split(os.pathsep) if d])

    dirs.extend([
        str(project_root / "tables"),
        str(module_dir / "tables"),
        str(Path.cwd() / "tables"),
        str(module_dir),
        str(Path.cwd()),
    ])

    # Preserve old CERN locations as last-resort fallbacks only.
    dirs.extend([
        "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables",
        "/eos/experiment/wcte/wcte_tests/mPMT_led_events/LicketyFit_stuff",
    ])

    unique = []
    seen = set()
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def _candidate_paths(filename):
    return [os.path.join(d, filename) for d in _table_dirs()]


def _load_rel_mpmt_eff_tables():
    global _REL_MPMT_EFF_CACHE
    if _REL_MPMT_EFF_CACHE is not None:
        return _REL_MPMT_EFF_CACHE

    candidates = _candidate_paths("rel_mpmt_eff.dict")
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        # Self-contained WCSim/design fits do not need real-data relative mPMT
        # efficiency corrections.  Use unity curves so imports and particle
        # table loading work even when calibration dictionaries are absent.
        unity = np.ones(200, dtype=np.float64)
        _REL_MPMT_EFF_CACHE = (unity, unity, unity, unity)
        return _REL_MPMT_EFF_CACHE

    with open(path, "rb") as f:
        rel_mpmt_eff = pickle.load(f)

    _REL_MPMT_EFF_CACHE = (
        np.asarray(rel_mpmt_eff["tri_exsitu"], dtype=np.float64),
        np.asarray(rel_mpmt_eff["tri_insitu"], dtype=np.float64),
        np.asarray(rel_mpmt_eff["wut_insitu"], dtype=np.float64),
        np.asarray(rel_mpmt_eff["wut_exsitu"], dtype=np.float64),
    )
    return _REL_MPMT_EFF_CACHE


def _load_energy_distance_arrays(particle: str):
    pname = canonical_particle_name(particle)

    evsd_paths = _candidate_paths(f"E_vs_dist_cm_{pname}.npy")
    overall_paths = _candidate_paths(f"overall_distances_cm_{pname}.npy")

    # Backward-compatible fallback for historical muon table names.
    if pname == "muon":
        evsd_paths += _candidate_paths("E_vs_dist_cm.npy")
        overall_paths += _candidate_paths("overall_distances_cm.npy")

    E_vs_dist = _load_first_existing(evsd_paths, allow_pickle=True)
    overall_distances = _load_first_existing(overall_paths, allow_pickle=False) * 10.0

    energy_rows = [np.asarray(row[:, 1], dtype=np.float64) for row in E_vs_dist]
    distance_rows = [np.asarray(row[:, 0], dtype=np.float64) * 10.0 for row in E_vs_dist]

    return np.asarray(overall_distances, dtype=np.float64), energy_rows, distance_rows


def _analytic_cerenkov_angle_table(particle: str, n: float = 1.344):
    mass = particle_mass_mev(particle)
    eth = cherenkov_threshold_kinetic_mev(mass, n=n)

    # Start slightly below threshold so interpolation safely returns 0 there.
    e_min = max(0.0, eth - 5.0)
    e_max = 5000.0
    energy = np.linspace(e_min, e_max, 20000, dtype=np.float64)

    gamma = 1.0 + energy / mass
    beta2 = 1.0 - 1.0 / np.maximum(gamma, 1.0) ** 2
    beta2 = np.clip(beta2, 0.0, None)
    beta = np.sqrt(beta2)

    angles = np.zeros_like(energy)
    above = n * beta > 1.0
    angles[above] = np.arccos(np.clip(1.0 / (n * beta[above]), -1.0, 1.0))
    return angles, energy


def _load_cerenkov_angle_table(particle: str):
    pname = canonical_particle_name(particle)

    # Prefer the explicit particle-aware muon angle filename, while keeping
    # historical table names as fallbacks. Other particles use the analytic
    # mass-based table.
    if pname == "muon":
        cang_paths = [
            *(_candidate_paths("cherenkov_angle_vs_E_muon_n1344.npy")),
            *(_candidate_paths("mu_cAng_vs_E_n1344.npy")),
            "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/mu_cAng_vs_E_n1344.npy",
        ]
        for path in cang_paths:
            if path and os.path.exists(path):
                c_ang_vs_E = np.load(path, allow_pickle=False)
                angles = np.asarray(c_ang_vs_E[:, 0], dtype=np.float64)
                energy = np.asarray(c_ang_vs_E[:, 1], dtype=np.float64)
                # Original table has nan below threshold. Replace with zero.
                angles = np.where(np.isfinite(angles), angles, 0.0)
                return angles, energy

    return _analytic_cerenkov_angle_table(pname)


def _ensure_tables_loaded(particle: str | None = None):
    pname = canonical_particle_name(particle)
    cached = _TABLES_BY_PARTICLE.get(pname)
    if cached is not None:
        return cached

    c_ang, energy_for_angle = _load_cerenkov_angle_table(pname)
    overall_distances, energy_rows, distance_rows = _load_energy_distance_arrays(pname)
    tri_exsitu, tri_insitu, wut_insitu, wut_exsitu = _load_rel_mpmt_eff_tables()

    cached = {
        "particle": pname,
        "mass_mev": particle_mass_mev(pname),
        "threshold_mev": cherenkov_threshold_kinetic_mev(pname),
        "c_ang": np.asarray(c_ang, dtype=np.float64),
        "energy_for_angle": np.asarray(energy_for_angle, dtype=np.float64),
        "overall_distances": np.asarray(overall_distances, dtype=np.float64),
        "energy_rows": energy_rows,
        "distance_rows": distance_rows,
        "tri_exsitu": tri_exsitu,
        "tri_insitu": tri_insitu,
        "wut_insitu": wut_insitu,
        "wut_exsitu": wut_exsitu,
        # SMOOTH-NLL ADDITION: master KE(range) curve.  Row i of the K(s)
        # tables is an exact slice of this single monotone curve (verified to
        # <=1e-4 MeV), so E at remaining-range r can be evaluated continuously
        # as interp(r, master_range, master_ke) instead of snapping to the
        # nearest row.  This removes the ~1 MeV / ~5 mm staircase in L.
        "master_range": np.asarray(overall_distances, dtype=np.float64),
        "master_ke": np.asarray(
            [np.asarray(energy_rows[i], dtype=np.float64)[0] for i in range(len(overall_distances))],
            dtype=np.float64,
        ),
    }
    _TABLES_BY_PARTICLE[pname] = cached
    return cached


def get_cerenkov_angle_table(particle: str | None = None):
    t = _ensure_tables_loaded(particle)
    return t["c_ang"], t["energy_for_angle"]


def get_energy_distance_tables(particle: str | None = None):
    t = _ensure_tables_loaded(particle)
    return t["overall_distances"], t["energy_rows"], t["distance_rows"]


def get_rel_mpmt_eff_tables():
    return _load_rel_mpmt_eff_tables()


# -----------------------------------------------------------------------------
# Physics helpers
# -----------------------------------------------------------------------------
def cherenkov_scale_particle_water(
    T_MeV,
    particle: str | None = None,
    *,
    mass_mev: float | None = None,
    n: float = 1.344,
    hard_saturate_above_MeV=None,
):
    """
    Dimensionless Frank-Tamm light-yield scale for a singly charged particle.
    """
    if mass_mev is None:
        mass_mev = particle_mass_mev(particle)

    T = np.asarray(T_MeV, dtype=np.float64)
    gamma = 1.0 + T / float(mass_mev)
    beta2 = 1.0 - 1.0 / np.maximum(gamma, 1.0) ** 2
    beta2 = np.clip(beta2, 0.0, None)

    ft = 1.0 - 1.0 / (np.maximum(beta2, 1e-30) * n * n)
    ft_inf = 1.0 - 1.0 / (n * n)

    scale = np.zeros_like(T, dtype=np.float64)
    mask = beta2 * n * n > 1.0
    scale[mask] = ft[mask] / ft_inf
    scale = np.clip(scale, 0.0, 1.0)

    if hard_saturate_above_MeV is not None:
        scale[T >= hard_saturate_above_MeV] = 1.0

    return scale


def cherenkov_scale_primary_water(T_MeV, n=1.344, hard_saturate_above_MeV=None, particle="muon"):
    """Frank-Tamm Cherenkov scale for the selected primary particle in water."""
    return cherenkov_scale_particle_water(
        T_MeV,
        particle,
        n=n,
        hard_saturate_above_MeV=hard_saturate_above_MeV,
    )


# Backward-compatible alias for older notebooks/scripts. Prefer
# cherenkov_scale_primary_water(..., particle=...).
cherenkov_scale_muon_water = cherenkov_scale_primary_water


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
def _interp_energy_at_dist(dist_row, energy_row, x):
    """Linear interpolation of energy_row vs dist_row at distance x.

    dist_row is increasing; energy_row is the (decreasing) energy at that
    distance.  Used by the sub-grid refinement so E_b varies smoothly with the
    emission point instead of snapping to the nearest table row.
    """
    n = dist_row.size
    if n == 0:
        return 0.0
    if x <= dist_row[0]:
        return energy_row[0]
    if x >= dist_row[n - 1]:
        return energy_row[n - 1]
    idx = np.searchsorted(dist_row, x)
    if idx <= 0:
        return energy_row[0]
    if idx >= n:
        return energy_row[n - 1]
    x0 = dist_row[idx - 1]
    x1 = dist_row[idx]
    y0 = energy_row[idx - 1]
    y1 = energy_row[idx]
    dx = x1 - x0
    if dx <= 0.0:
        return y0
    return y0 + (x - x0) / dx * (y1 - y0)


@njit(cache=True)
def _scale_from_energy_scalar(T_MeV, mass_mev, n):
    gamma = 1.0 + T_MeV / mass_mev
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
    cot_grid,
    dist_row,
    energy_row,
    s_a_mm,
    s_max_mm,
    near_cross_tol,
    particle_mass_mev,
    n_water,
    refine,
    master_r,
    master_k,
    range_stop_mm,
    edge_model,
):
    """
    Hybrid binary-search collapse solver.

    It preserves your optimized crossing logic but uses a supplied particle mass
    for the Frank-Tamm scale rather than a hard-coded muon mass.
    """
    n_pmts = pmt_pos.shape[0]
    n_scan = s_grid.size

    scale = np.zeros(n_pmts, dtype=np.float64)
    s_b = np.full(n_pmts, s_a_mm, dtype=np.float64)
    E_b = np.empty(n_pmts, dtype=np.float64)

    tx = track_dir[0]
    ty = track_dir[1]
    tz = track_dir[2]

    for i in range(n_pmts):
        wx = pmt_pos[i, 0] - start_pos[0]
        wy = pmt_pos[i, 1] - start_pos[1]
        wz = pmt_pos[i, 2] - start_pos[2]

        w2 = wx * wx + wy * wy + wz * wz
        u0 = wx * tx + wy * ty + wz * tz

        perp2 = w2 - u0 * u0
        if perp2 < 0.0:
            perp2 = 0.0
        perp = np.sqrt(perp2)

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

        if (n_scan >= 2) and (f_lo < 0.0) and (f_hi >= 0.0):
            # Crossing case.  The root of f(s) = arctan2(perp, u0-s) - theta_c(s)
            # in the forward region is identical to the root of
            #   g(s) = (u0 - s) - perp * cot(theta_c(s)),
            # because arctan2(perp, u0-s) = theta_c  <=>  (u0-s) = perp*cot(theta_c).
            # The sign maps as f >= 0  <=>  g <= 0, so we bisect on g's sign with
            # pure arithmetic (one multiply/subtract per step) instead of an
            # arctan2 per step.  cot_grid is precomputed once per grid.
            # min_abs_f / min_idx are only used by the soft no-crossing branch,
            # so they need no update here.
            lo = 0
            hi = n_scan - 1

            while hi - lo > 1:
                mid = (lo + hi) // 2
                g_mid = (u0 - s_grid[mid]) - perp * cot_grid[mid]
                if g_mid <= 0.0:   # f_mid >= 0.0
                    hi = mid
                else:
                    lo = mid

            cross_idx = hi - 1
            if cross_idx < 0:
                cross_idx = 0
            found_crossing = True

            # Sub-grid linear refinement of the crossing.  Without it, sb snaps
            # to s_grid[cross_idx], so as the fit parameters move continuously the
            # emission point (and the FCN) move in ~grid-sized steps -- a
            # staircase that makes the numerical gradient/Hessian noisy and
            # inflates MIGRAD's EDM (frequent "invalid" with otherwise-good fits).
            # Here we interpolate the true zero of g(s)=(u0-s)-perp*cot(theta_c)
            # between the bracketing grid points lo=cross_idx and hi=cross_idx+1,
            # so sb varies smoothly.  g_lo > 0 >= g_hi by construction.
            if refine != 0 and (hi < n_scan) and (hi > 0):
                lo_i = hi - 1
                g_lo = (u0 - s_grid[lo_i]) - perp * cot_grid[lo_i]
                g_hi = (u0 - s_grid[hi]) - perp * cot_grid[hi]
                denom = g_lo - g_hi
                if denom > 0.0:
                    frac = g_lo / denom
                    if frac < 0.0:
                        frac = 0.0
                    elif frac > 1.0:
                        frac = 1.0
                    sb_refined = s_grid[lo_i] + frac * (s_grid[hi] - s_grid[lo_i])
                else:
                    sb_refined = s_grid[cross_idx]
            else:
                sb_refined = s_grid[cross_idx]
        # No-crossing branch: f(s) = arctan2(perp, u0 - s) - theta_c(s) is
        # monotone non-decreasing in s (the viewing angle grows as s increases
        # while theta_c shrinks as the particle loses energy), so the minimum of
        # |f| over the interior is always attained at an endpoint.  The endpoint
        # comparison above already set min_abs_f/min_idx, so the original
        # interior linear scan over j was redundant and has been removed.
        # Verified bit-identical against the scanning version over ~1500 track
        # configurations x 1805 PMTs (max abs diff 0.0).

        if found_crossing:
            sb = sb_refined
            if master_r.size > 1:
                # SMOOTH-NLL: E from the continuous master KE(range) curve.
                # The row-based lookup swaps the ENTIRE (dist,energy) row when
                # the fitted length crosses a table-row boundary (~5 mm), which
                # shifts every crossing tube's Frank-Tamm scale coherently and
                # imprints cliffs on NLL(L).
                rem = range_stop_mm - (sb - s_a_mm)
                if rem < 0.0:
                    rem = 0.0
                Eb = np.interp(rem, master_r, master_k)
            elif refine != 0:
                Eb = _interp_energy_at_dist(dist_row, energy_row, sb - s_a_mm)
            else:
                eidx = _nearest_index_1d(dist_row, sb - s_a_mm)
                Eb = energy_row[eidx]

            s_b[i] = sb
            E_b[i] = Eb
            amp = _scale_from_energy_scalar(Eb, particle_mass_mev, n_water)
            if edge_model == 1 and near_cross_tol > 0.0:
                # SYMMETRIC (charge-conserving) smeared edge.
                # f(s) is monotone in s, so f_lo / f_hi are the signed margins
                # across the OUTER ring edge (sign change at f_lo=0) and the
                # INNER edge / end-cap hole (sign change at f_hi=0).  A sharp
                # edge convolved with a Gaussian of width sigma is an erf edge:
                #   w = Phi(-f_lo/sigma) * Phi(f_hi/sigma)
                # deep-lit -> 1; outer edge -> first factor 1->1/2->tail;
                # inner hole -> second factor likewise.  Unlike the legacy
                # one-sided tail this DIMS just-inside PMTs by the same charge
                # it adds just-outside (exact for a locally straight edge).
                inv_s = 1.0 / (near_cross_tol * 1.4142135623730951)
                w = 0.25 * (1.0 + math.erf(-f_lo * inv_s)) * (1.0 + math.erf(f_hi * inv_s))
                amp = amp * w
            scale[i] = amp

        else:
            sb = s_grid[min_idx]
            if master_r.size > 1:
                rem = range_stop_mm - (sb - s_a_mm)
                if rem < 0.0:
                    rem = 0.0
                Eb = np.interp(rem, master_r, master_k)
            else:
                eidx = _nearest_index_1d(dist_row, sb - s_a_mm)
                Eb = energy_row[eidx]

            s_b[i] = sb
            E_b[i] = Eb

            sigma_theta = near_cross_tol
            if sigma_theta <= 0.0:
                scale[i] = 0.0
            else:
                ft_scale = _scale_from_energy_scalar(Eb, particle_mass_mev, n_water)
                if edge_model == 1:
                    # same continuous erf-edge weight as the crossing branch;
                    # here f_lo>0 (beyond outer edge) or f_hi<0 (inner hole),
                    # so w reduces to the appropriate complementary-erf tail.
                    inv_s = 1.0 / (sigma_theta * 1.4142135623730951)
                    w = 0.25 * (1.0 + math.erf(-f_lo * inv_s)) * (1.0 + math.erf(f_hi * inv_s))
                    scale[i] = ft_scale * w
                else:
                    soft = math.exp(-0.5 * (min_abs_f / sigma_theta) * (min_abs_f / sigma_theta))
                    scale[i] = ft_scale * soft

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
    edge_model: str = "legacy",
    particle: str | None = None,
    particle_mass: float | None = None,
    n_water: float = 1.344,
    range_stop_mm: float | None = None,
    subgrid_refine: bool = True,
    legacy_grid: bool = False,
):
    """
    Fast Cherenkov-cone-collapse solver for many PMTs.

    The original argument order is preserved.  New optional arguments:
      particle      : "muon", "pion", "kaon", or "proton"
      particle_mass : override mass in MeV/c^2
      n_water       : refractive index
      range_stop_mm : dE/dx range-to-threshold used for K(s).  If omitted,
                      the old behavior uses s_max_mm for both visible length
                      and range-to-threshold.
    """
    del theta_c_func  # kept only for drop-in API compatibility

    pname = canonical_particle_name(particle)
    tables = _ensure_tables_loaded(pname)
    overall_distances = tables["overall_distances"]
    energy_rows = tables["energy_rows"]
    distance_rows = tables["distance_rows"]
    c_ang = tables["c_ang"]
    energy_for_angle = tables["energy_for_angle"]
    if particle_mass is None:
        particle_mass = tables["mass_mev"]

    pmt_pos = np.asarray(pmt_pos, dtype=np.float64)
    start_pos = np.asarray(start_pos, dtype=np.float64)
    track_dir = np.asarray(track_dir, dtype=np.float64)
    track_dir = track_dir / np.linalg.norm(track_dir)

    if range_stop_mm is None:
        range_stop_for_energy = float(s_max_mm)
    else:
        range_stop_for_energy = float(range_stop_mm)

    visible_s_max = min(float(s_max_mm), range_stop_for_energy)
    visible_s_max = max(visible_s_max, float(s_a_mm))

    # The Cherenkov-angle scan grid (dist_row, energy_row, s_grid, E_grid,
    # theta_c_grid) depends only on these scalars, not on the per-call track
    # vertex/direction.  During a Minuit fit the vertex and direction change on
    # every FCN call while the length parameters change far less often, so
    # caching this construction removes a linspace + searchsorted + theta
    # interpolation from the majority of calls.  Bit-identical: same scalar key
    # -> same arrays.
    cache_key = (
        pname,
        float(s_a_mm),
        visible_s_max,
        int(n_scan),
        range_stop_for_energy,
        float(particle_mass),
        float(n_water),
        bool(legacy_grid),
    )
    cached = _SCAN_GRID_CACHE.pop(cache_key, None)
    if cached is not None:
        # Exact-key LRU refresh.  No quantization is introduced, so FCN values
        # and smoothness are unchanged.
        _SCAN_GRID_CACHE[cache_key] = cached
    else:
        main_idx = _nearest_index_1d(overall_distances, range_stop_for_energy)
        dist_row = np.ascontiguousarray(distance_rows[main_idx], dtype=np.float64)
        energy_row = np.ascontiguousarray(energy_rows[main_idx], dtype=np.float64)

        threshold = cherenkov_threshold_kinetic_mev(float(particle_mass), n=float(n_water))

        if legacy_grid:
            # ---- historical behavior (bit-exact): L-dependent grid spacing,
            # nearest-row K(s), nearest-point E lookup.  Kept for A/B checks. ----
            s_grid = np.linspace(float(s_a_mm), visible_s_max, int(n_scan), dtype=np.float64)
            ds_mm = s_grid - float(s_a_mm)
            idx = np.searchsorted(dist_row, ds_mm)
            idx_right = np.clip(idx, 0, dist_row.size - 1)
            idx_left = np.clip(idx - 1, 0, dist_row.size - 1)
            use_left = (ds_mm - dist_row[idx_left]) <= (dist_row[idx_right] - ds_mm)
            idx = np.where(use_left, idx_left, idx_right)
            E_grid = energy_row[idx]
            E_grid = np.maximum(E_grid, threshold)
        else:
            # ---- SMOOTH-NLL path (default). Two changes:
            # (1) Fixed ABSOLUTE grid step instead of n_scan points over [s_a, L]:
            #     with linspace, every grid point moves when L changes, which
            #     imprints a sawtooth of period ~L/n_scan on NLL(L).  With a
            #     fixed step anchored at s_a, existing points stay put as L
            #     varies; only the (low-weight) endpoint region changes.
            #     The step is chosen so the default cost matches the old
            #     n_scan=150 at L~1200 (8 mm) unless the caller's n_scan
            #     implies finer sampling.
            # (2) E along the track from the continuous master KE(range) curve
            #     E(s) = master(range_stop - s) instead of the nearest table
            #     row + nearest-point lookup, removing the ~1 MeV staircase.
            step = min(2.0, max(0.5, visible_s_max / max(int(n_scan), 1)))
            n_pts = int(np.floor((visible_s_max - float(s_a_mm)) / step)) + 1
            s_grid = float(s_a_mm) + step * np.arange(n_pts, dtype=np.float64)
            if s_grid[-1] < visible_s_max - 1e-9:
                s_grid = np.append(s_grid, visible_s_max)
            master_r = tables["master_range"]
            master_k = tables["master_ke"]
            remaining = np.maximum(range_stop_for_energy - (s_grid - float(s_a_mm)), 0.0)
            E_grid = np.interp(remaining, master_r, master_k,
                               left=master_k[0], right=master_k[-1])
            E_grid = np.maximum(E_grid, threshold)

        theta_c_grid = _theta_interp_numba(
            np.asarray(energy_for_angle, dtype=np.float64),
            np.asarray(c_ang, dtype=np.float64),
            np.asarray(E_grid, dtype=np.float64),
        )
        while len(_SCAN_GRID_CACHE) >= _SCAN_GRID_CACHE_MAX:
            _SCAN_GRID_CACHE.popitem(last=False)
        # Precompute cot(theta_c) once per (length-scalar) grid so the per-PMT
        # crossing bisection can test (u0 - s) - perp*cot(theta_c) >= 0 with pure
        # arithmetic instead of an arctan2 every bisection step.  sin(theta_c) is
        # bounded away from 0 over the Cherenkov-active grid; guard the rare
        # near-zero (sub-threshold tail) so cot stays finite.
        _sin_tc = np.sin(theta_c_grid)
        _cos_tc = np.cos(theta_c_grid)
        cot_grid = np.where(_sin_tc > 1e-12, _cos_tc / np.where(_sin_tc > 1e-12, _sin_tc, 1.0), 1e30)
        cached = (dist_row, energy_row, s_grid, theta_c_grid, cot_grid)
        _SCAN_GRID_CACHE[cache_key] = cached

    dist_row, energy_row, s_grid, theta_c_grid, cot_grid = cached

    scale, s_b, E_b = _find_scale_kernel(
        np.asarray(pmt_pos, dtype=np.float64),
        np.asarray(start_pos, dtype=np.float64),
        np.asarray(track_dir, dtype=np.float64),
        s_grid,
        theta_c_grid,
        cot_grid,
        dist_row,
        energy_row,
        float(s_a_mm),
        float(visible_s_max),
        float(near_cross_tol),
        float(particle_mass),
        float(n_water),
        int(1 if subgrid_refine else 0),
        (np.empty(0, dtype=np.float64) if legacy_grid else tables["master_range"]),
        (np.empty(0, dtype=np.float64) if legacy_grid else tables["master_ke"]),
        float(range_stop_for_energy),
        int(1 if str(edge_model).lower() == "erf" else 0),
    )

    _ = mpmt_bool
    return scale, s_b, E_b


# Backward-compatible aliases.
def find_scale_for_pmts_old2(*args, **kwargs):
    return find_scale_for_pmts(*args, **kwargs)


def find_scale_for_pmts_old(*args, **kwargs):
    out = find_scale_for_pmts(*args, **kwargs)
    return out[0], out[1]

