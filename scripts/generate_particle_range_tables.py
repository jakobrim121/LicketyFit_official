"""
generate_particle_range_tables.py

Generate water Cherenkov range tables for singly charged particles.

The output format intentionally matches the original muon tables used by the
fitter:

    E_vs_dist_cm_<particle>.npy
        object array; each element is a two-column float array:
            column 0: distance travelled from the track start [cm]
            column 1: kinetic energy at that distance [MeV]

    overall_distances_cm_<particle>.npy
        total distance travelled while above Cherenkov threshold [cm]

The tables are CSDA ionization-loss tables generated from a Bethe-Bloch model
for water. They do not include pion/kaon decay, nuclear interactions, hadronic
scattering, or charge exchange. Use Geant4/NIST-derived tables later if you want
a full production-grade hadron model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PARTICLE_MASS_MEV = {
    "muon": 105.6583755,
    "pion": 139.57039,
    "kaon": 493.677,
    "proton": 938.27208816,
}

ALIASES = {
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


def canonical_particle_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in ALIASES:
        raise ValueError(
            f"Unknown particle {name!r}. Use one of: "
            f"{', '.join(sorted(ALIASES))}"
        )
    return ALIASES[key]


def cherenkov_threshold_kinetic_mev(mass_mev: float, n: float = 1.344) -> float:
    beta_thr = 1.0 / float(n)
    gamma_thr = 1.0 / np.sqrt(1.0 - beta_thr * beta_thr)
    return float(mass_mev * (gamma_thr - 1.0))


def bethe_bloch_dedx_mev_per_cm(
    kinetic_mev,
    mass_mev: float,
    *,
    n: float = 1.344,
    density_g_cm3: float = 1.0,
    z_over_a_water: float = 10.0 / 18.01528,
    mean_excitation_mev: float = 75.0e-6,
    charge_abs: float = 1.0,
):
    """
    Approximate positive stopping power -dE/dx [MeV/cm] in water.

    This is the heavy-particle Bethe-Bloch collision loss without density-effect
    or shell corrections. That is usually good enough to make internally
    consistent fitter lookup tables, but it is not a substitute for PSTAR/ASTAR
    or a full Geant4 material model.
    """
    K_BETHE = 0.307075  # MeV mol^-1 cm^2 / g
    M_E = 0.51099895000  # MeV

    T = np.asarray(kinetic_mev, dtype=np.float64)
    mass = float(mass_mev)

    gamma = 1.0 + T / mass
    beta2 = 1.0 - 1.0 / np.maximum(gamma, 1.0) ** 2
    beta2 = np.clip(beta2, 1e-12, 1.0 - 1e-12)

    mass_ratio = M_E / mass
    tmax = (
        2.0 * M_E * beta2 * gamma * gamma
        / (1.0 + 2.0 * gamma * mass_ratio + mass_ratio * mass_ratio)
    )

    arg = 2.0 * M_E * beta2 * gamma * gamma * tmax / (mean_excitation_mev ** 2)
    log_term = 0.5 * np.log(np.maximum(arg, 1.0 + 1e-12))

    mass_stopping = (
        K_BETHE
        * (charge_abs * charge_abs)
        * z_over_a_water
        * (1.0 / beta2)
        * (log_term - beta2)
    )
    mass_stopping = np.maximum(mass_stopping, 1e-9)

    return density_g_cm3 * mass_stopping


def cumulative_range_above_threshold_cm(
    mass_mev: float,
    *,
    e_max_mev: float,
    n: float = 1.344,
    fine_step_mev: float = 0.05,
):
    """
    Return an energy grid and cumulative CSDA range from threshold to energy.

    range_cm[i] = integral from K_threshold to E_grid[i] dK / S(K)
    """
    eth = cherenkov_threshold_kinetic_mev(mass_mev, n=n)
    e_hi = max(float(e_max_mev), eth + fine_step_mev)
    e_grid = np.arange(eth, e_hi + fine_step_mev, fine_step_mev, dtype=np.float64)
    e_grid[0] = eth

    dedx = bethe_bloch_dedx_mev_per_cm(e_grid, mass_mev, n=n)
    inv = 1.0 / np.maximum(dedx, 1e-30)

    dE = np.diff(e_grid)
    dR = 0.5 * (inv[1:] + inv[:-1]) * dE
    r_grid = np.empty_like(e_grid)
    r_grid[0] = 0.0
    r_grid[1:] = np.cumsum(dR)

    return e_grid, r_grid


def build_particle_tables(
    particle: str,
    *,
    e_max_mev: float = 3000.0,
    initial_energy_step_mev: float = 1.0,
    n_water: float = 1.344,
    row_step_cm: float = 0.2,
    max_points_per_row: int = 1600,
):
    """
    Build the pair (E_vs_dist, overall_distances_cm) for one particle.
    """
    pname = canonical_particle_name(particle)
    mass = PARTICLE_MASS_MEV[pname]
    eth = cherenkov_threshold_kinetic_mev(mass, n=n_water)

    e_fine, r_fine = cumulative_range_above_threshold_cm(
        mass,
        e_max_mev=e_max_mev,
        n=n_water,
    )

    # Match the original table philosophy: one row per initial kinetic energy.
    e0_grid = np.arange(
        np.ceil(eth),
        float(e_max_mev) + 0.5 * initial_energy_step_mev,
        initial_energy_step_mev,
        dtype=np.float64,
    )
    e0_grid = e0_grid[e0_grid > eth]
    if e0_grid.size == 0:
        raise ValueError(f"No initial energies above threshold for {pname}.")

    rows = []
    overall = []

    for e0 in e0_grid:
        r0 = float(np.interp(e0, e_fine, r_fine))
        overall.append(r0)

        if r0 <= 0.0 or not np.isfinite(r0):
            row = np.array([[0.0, float(e0)]], dtype=np.float64)
            rows.append(row)
            continue

        n_points = int(np.ceil(r0 / max(row_step_cm, 1e-12))) + 1
        n_points = max(2, min(int(max_points_per_row), n_points))

        s_cm = np.linspace(0.0, r0, n_points, dtype=np.float64)

        # Distance already travelled from start s corresponds to remaining
        # above-threshold range r0 - s. Since r_fine grows monotonically with K,
        # invert range -> kinetic energy.
        remaining_range = np.maximum(r0 - s_cm, 0.0)
        e_at_s = np.interp(remaining_range, r_fine, e_fine)
        e_at_s[0] = e0

        rows.append(np.column_stack([s_cm, e_at_s]).astype(np.float64))

    return np.asarray(rows, dtype=object), np.asarray(overall, dtype=np.float64)


def write_particle_tables(
    particle: str,
    output_dir: str | Path,
    *,
    e_max_mev: float = 3000.0,
    initial_energy_step_mev: float = 1.0,
    n_water: float = 1.344,
    row_step_cm: float = 0.2,
    max_points_per_row: int = 1600,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pname = canonical_particle_name(particle)
    rows, overall = build_particle_tables(
        pname,
        e_max_mev=e_max_mev,
        initial_energy_step_mev=initial_energy_step_mev,
        n_water=n_water,
        row_step_cm=row_step_cm,
        max_points_per_row=max_points_per_row,
    )

    evsd_path = output_dir / f"E_vs_dist_cm_{pname}.npy"
    overall_path = output_dir / f"overall_distances_cm_{pname}.npy"

    np.save(evsd_path, rows, allow_pickle=True)
    np.save(overall_path, overall)

    return evsd_path, overall_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--particles",
        nargs="+",
        default=["pion", "kaon", "proton"],
        help="Particles to generate: muon pion kaon proton",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--e-max-mev", type=float, default=3000.0)
    parser.add_argument("--initial-energy-step-mev", type=float, default=1.0)
    parser.add_argument("--row-step-cm", type=float, default=0.2)
    parser.add_argument("--max-points-per-row", type=int, default=1600)
    args = parser.parse_args()

    for particle in args.particles:
        evsd, overall = write_particle_tables(
            particle,
            args.output_dir,
            e_max_mev=args.e_max_mev,
            initial_energy_step_mev=args.initial_energy_step_mev,
            row_step_cm=args.row_step_cm,
            max_points_per_row=args.max_points_per_row,
        )
        print(f"{canonical_particle_name(particle)}:")
        print(f"  {evsd}")
        print(f"  {overall}")


if __name__ == "__main__":
    main()
