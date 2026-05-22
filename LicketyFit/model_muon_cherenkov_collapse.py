"""
model_muon_cherenkov_collapse.py

Particle-aware Cherenkov-cone collapse helpers.

This file keeps the historical module name so existing imports continue to work,
but it no longer assumes the primary track is a muon.  Use set_active_particle()
or pass particle=... into find_scale_for_pmts() to choose between:

    muon, pion, kaon, proton

Range tables are expected to use the same format as your original muon tables.
For non-muon particles the default filenames are:

    E_vs_dist_cm_pion.npy       overall_distances_cm_pion.npy
    E_vs_dist_cm_kaon.npy       overall_distances_cm_kaon.npy
    E_vs_dist_cm_proton.npy     overall_distances_cm_proton.npy

The old muon filenames E_vs_dist_cm.npy and overall_distances_cm.npy are still
accepted as fallbacks.
"""

from __future__ import annotations

import math
import os
import pickle
from functools import lru_cache
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

    # Backward-compatible fallback for the old muon table names.
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

    # Preserve the original muon angle table if it exists; otherwise all
    # particles can use the analytic mass-based table.
    if pname == "muon":
        cang_paths = [
            "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/mu_cAng_vs_E_n1344.npy",
            *(_candidate_paths("mu_cAng_vs_E_n1344.npy")),
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


def cherenkov_scale_muon_water(T_MeV, n=1.344, hard_saturate_above_MeV=None):
    """Backward-compatible wrapper."""
    return cherenkov_scale_particle_water(
        T_MeV,
        "muon",
        n=n,
        hard_saturate_above_MeV=hard_saturate_above_MeV,
    )


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
    dist_row,
    energy_row,
    s_a_mm,
    s_max_mm,
    near_cross_tol,
    particle_mass_mev,
    n_water,
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

            cross_idx = hi - 1
            if cross_idx < 0:
                cross_idx = 0
            found_crossing = True
        else:
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
            scale[i] = _scale_from_energy_scalar(Eb, particle_mass_mev, n_water)

        else:
            sb = s_grid[min_idx]
            eidx = _nearest_index_1d(dist_row, sb - s_a_mm)
            Eb = energy_row[eidx]

            s_b[i] = sb
            E_b[i] = Eb

            sigma_theta = near_cross_tol
            if sigma_theta <= 0.0:
                scale[i] = 0.0
            else:
                ft_scale = _scale_from_energy_scalar(Eb, particle_mass_mev, n_water)
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
    particle: str | None = None,
    particle_mass: float | None = None,
    n_water: float = 1.344,
    range_stop_mm: float | None = None,
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

    main_idx = _nearest_index_1d(overall_distances, range_stop_for_energy)
    dist_row = distance_rows[main_idx]
    energy_row = energy_rows[main_idx]

    visible_s_max = min(float(s_max_mm), range_stop_for_energy)
    visible_s_max = max(visible_s_max, float(s_a_mm))
    s_grid = np.linspace(float(s_a_mm), visible_s_max, int(n_scan), dtype=np.float64)

    ds_mm = s_grid - float(s_a_mm)
    idx = np.searchsorted(dist_row, ds_mm)
    idx_right = np.clip(idx, 0, dist_row.size - 1)
    idx_left = np.clip(idx - 1, 0, dist_row.size - 1)
    use_left = (ds_mm - dist_row[idx_left]) <= (dist_row[idx_right] - ds_mm)
    idx = np.where(use_left, idx_left, idx_right)
    E_grid = energy_row[idx]

    threshold = cherenkov_threshold_kinetic_mev(float(particle_mass), n=float(n_water))
    E_grid = np.maximum(E_grid, threshold)

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
        float(visible_s_max),
        float(near_cross_tol),
        float(particle_mass),
        float(n_water),
    )

    _ = mpmt_bool
    return scale, s_b, E_b


# Backward-compatible aliases.
def find_scale_for_pmts_old2(*args, **kwargs):
    return find_scale_for_pmts(*args, **kwargs)


def find_scale_for_pmts_old(*args, **kwargs):
    out = find_scale_for_pmts(*args, **kwargs)
    return out[0], out[1]
