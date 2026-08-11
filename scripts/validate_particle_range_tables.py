#!/usr/bin/env python3
"""Validate all LicketyFit particle range tables and primary-model loading."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate_hadron_range_tables import (  # noqa: E402
    N_WATER,
    integrate_visible_range_cm,
    pdg_bethe_mass_stopping_power,
)
from particle_range_lookup import (  # noqa: E402
    PARTICLE_MASS_MEV,
    ParticleRangeLookup,
    cherenkov_threshold_kinetic_mev,
)
from LicketyFit.Emitter import Emitter  # noqa: E402
from LicketyFit.particle_cherenkov_model import (  # noqa: E402
    find_scale_for_pmts,
    get_energy_distance_tables,
    particle_subthreshold_range_mm,
)


PARTICLES = ("pion", "kaon", "proton")
EXPECTED_SUBTHRESHOLD_MM = {
    "pion": 158.5,
    "kaon": 560.7,
    "proton": 1048.2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _synthetic_pmts(kinetic_energy_mev, mass_mev, count=12):
    gamma = 1.0 + float(kinetic_energy_mev) / float(mass_mev)
    beta = math.sqrt(1.0 - 1.0 / (gamma * gamma))
    theta = math.acos(1.0 / (N_WATER * beta))
    # Put tubes just inside the initial cone so the monotone collapse equation
    # has a well-defined crossing a short distance down the track.
    angle = 0.98 * theta
    phi = np.linspace(0.0, 2.0 * math.pi, int(count), endpoint=False)
    radius = 1000.0
    return np.column_stack(
        (
            radius * math.sin(angle) * np.cos(phi),
            radius * math.sin(angle) * np.sin(phi),
            np.full(phi.shape, radius * math.cos(angle)),
        )
    )


def validate_metadata():
    metadata_path = PROJECT_ROOT / "tables" / "PARTICLE_RANGE_TABLES.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _check(
        metadata["table_format"] == "licketyfit_compact_range_v1",
        "unexpected range-table format",
    )
    _check(
        float(metadata["water"]["mean_excitation_energy_ev"]) == 78.0,
        "nominal liquid-water mean excitation energy is not 78 eV",
    )
    for filename, info in metadata["files"].items():
        path = PROJECT_ROOT / "tables" / filename
        _check(path.is_file(), f"missing generated table {path}")
        _check(path.stat().st_size == int(info["bytes"]), f"size mismatch for {filename}")
        _check(_sha256(path) == info["sha256"], f"SHA-256 mismatch for {filename}")
    pstar_rel = np.asarray(
        metadata["proton_reference_comparison"][
            "visible_range_relative_difference_nominal78_minus_pstar75"
        ],
        dtype=np.float64,
    )
    _check(
        np.max(np.abs(pstar_rel)) < 0.011,
        "nominal proton range differs from the independent PSTAR-75 comparison by >=1.1%",
    )
    pdg79p7_rel = np.asarray(
        metadata["proton_reference_comparison"][
            "visible_range_relative_difference_nominal78_minus_pdg79p7"
        ],
        dtype=np.float64,
    )
    _check(
        np.max(np.abs(pdg79p7_rel)) < 0.003,
        "nominal proton range differs from the PDG-79.7 systematic by >=0.3%",
    )
    return (
        metadata,
        float(np.max(np.abs(pstar_rel))),
        float(np.max(np.abs(pdg79p7_rel))),
    )


def validate_one_particle(particle):
    threshold = cherenkov_threshold_kinetic_mev(particle, n=N_WATER)
    compact_path = PROJECT_ROOT / "tables" / f"E_vs_dist_cm_{particle}.npy"
    distance_path = PROJECT_ROOT / "tables" / f"overall_distances_cm_{particle}.npy"
    compact = np.load(compact_path, allow_pickle=False)
    ranges_cm = np.load(distance_path, allow_pickle=False)
    _check(compact.ndim == 2 and compact.shape[1] == 2, f"bad compact shape for {particle}")
    _check(compact.shape[0] == ranges_cm.size, f"row-count mismatch for {particle}")
    _check(np.array_equal(compact[:, 0], ranges_cm), f"range columns disagree for {particle}")
    energies = compact[:, 1]
    _check(energies[0] == math.ceil(threshold), f"bad first energy for {particle}")
    _check(energies[-1] == 3000.0, f"bad last energy for {particle}")
    _check(np.all(np.diff(energies) == 1.0), f"energy grid is not 1 MeV for {particle}")
    _check(np.all(np.isfinite(ranges_cm)), f"nonfinite range for {particle}")
    _check(np.all(np.diff(ranges_cm) > 0.0), f"nonmonotone range for {particle}")

    # Numerical integration and derivative/stopping-power consistency.
    rebuilt = integrate_visible_range_cm(particle, energies, quadrature_order=48)
    integration_rel = float(
        np.max(np.abs(rebuilt - ranges_cm) / np.maximum(ranges_cm, 1.0e-12))
    )
    # The Sternheimer density correction has a piecewise boundary; changing
    # Gauss order shifts the one interval containing that boundary at the
    # few-parts-in-1e9 level.
    _check(integration_rel < 1.0e-8, f"quadrature instability for {particle}")
    numerical_stopping = 1.0 / np.gradient(ranges_cm, energies)
    theory_stopping = pdg_bethe_mass_stopping_power(
        energies,
        PARTICLE_MASS_MEV[particle],
        spin_half=particle == "proton",
    )
    checkpoint_mask = np.isin(energies, [500.0, 800.0, 1000.0, 2000.0, 2990.0])
    checkpoint_mask &= energies > threshold
    stopping_rel = float(
        np.max(
            np.abs(
                numerical_stopping[checkpoint_mask] - theory_stopping[checkpoint_mask]
            )
            / theory_stopping[checkpoint_mask]
        )
    )
    _check(stopping_rel < 2.0e-5, f"range derivative mismatch for {particle}")

    # Public lookup conversion and exact round trips at tabulated energies.
    lookup = ParticleRangeLookup(particle)
    _check(lookup.energy_to_range_mm(threshold) == 0.0, f"threshold range not zero for {particle}")
    samples = energies[np.linspace(0, energies.size - 1, 9, dtype=int)]
    roundtrip_error = 0.0
    for energy in samples:
        range_mm = lookup.energy_to_range_mm(float(energy))
        recovered = lookup.range_mm_to_energy(range_mm)
        roundtrip_error = max(roundtrip_error, abs(recovered - energy))
    _check(roundtrip_error < 2.0e-10, f"lookup round-trip failure for {particle}")

    # The compact loader must reconstruct the old trajectory-row contract.
    overall_mm, energy_rows, distance_rows = get_energy_distance_tables(particle)
    for index in (0, len(overall_mm) // 2, len(overall_mm) - 1):
        energy_row = np.asarray(energy_rows[index], dtype=np.float64)
        distance_row = np.asarray(distance_rows[index], dtype=np.float64)
        _check(energy_row[0] == energies[index], f"bad K0 row for {particle}")
        _check(np.isclose(energy_row[-1], threshold), f"bad threshold row for {particle}")
        _check(distance_row[0] == 0.0, f"bad trajectory origin for {particle}")
        _check(
            np.isclose(distance_row[-1], overall_mm[index], rtol=0.0, atol=1.0e-9),
            f"bad trajectory endpoint for {particle}",
        )
        _check(np.all(np.diff(energy_row) < 0.0), f"K(s) not decreasing for {particle}")
        _check(np.all(np.diff(distance_row) > 0.0), f"s row not increasing for {particle}")

    # Emitter initialization must recover the selected kinetic energy and use
    # the particle-aware below-threshold secondary-electron tail.
    smoke_energy = max(800.0, math.ceil(threshold) + 100.0)
    smoke_range = lookup.energy_to_range_mm(smoke_energy)
    emitter = Emitter(
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.9,
        smoke_range,
        1.0,
        particle=particle,
    )
    _check(abs(emitter.interp_E_init - smoke_energy) < 2.0e-10, f"Emitter K0 mismatch for {particle}")
    expected_tail = EXPECTED_SUBTHRESHOLD_MM[particle]
    _check(
        particle_subthreshold_range_mm(particle) == expected_tail
        and emitter.muon_subthreshold_range_mm == expected_tail,
        f"particle-aware subthreshold tail mismatch for {particle}",
    )

    pmts = _synthetic_pmts(smoke_energy, PARTICLE_MASS_MEV[particle])
    model_summary = {}
    for legacy_grid in (False, True):
        scale, emission_s, emission_energy = find_scale_for_pmts(
            pmts,
            np.zeros(3, dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0]),
            0.0,
            smoke_range,
            None,
            particle=particle,
            range_stop_mm=smoke_range,
            legacy_grid=legacy_grid,
        )
        _check(np.all(np.isfinite(scale)), f"nonfinite light scale for {particle}")
        _check(np.max(scale) > 0.0, f"no primary cone crossing for {particle}")
        _check(np.all(np.isfinite(emission_s)), f"nonfinite emission s for {particle}")
        _check(np.all(np.isfinite(emission_energy)), f"nonfinite emission K for {particle}")
        model_summary["legacy" if legacy_grid else "smooth"] = float(np.max(scale))

    return {
        "threshold_mev": float(threshold),
        "rows": int(energies.size),
        "range_at_3gev_cm": float(ranges_cm[-1]),
        "quadrature_max_relative_difference": integration_rel,
        "dedx_checkpoint_max_relative_difference": stopping_rel,
        "lookup_roundtrip_max_mev": roundtrip_error,
        "primary_scale_max": model_summary,
        "subthreshold_tail_mm": expected_tail,
    }


def validate_virtual_muon_against_existing_table():
    legacy = np.load(PROJECT_ROOT / "tables" / "E_vs_dist_cm_muon.npy", allow_pickle=True)
    legacy_ranges = np.load(PROJECT_ROOT / "tables" / "overall_distances_cm_muon.npy")
    legacy_energy = np.asarray([float(row[0, 1]) for row in legacy], dtype=np.float64)
    energies = np.arange(
        math.ceil(cherenkov_threshold_kinetic_mev("muon", n=N_WATER)),
        3001,
        dtype=np.float64,
    )
    nominal_ranges = integrate_visible_range_cm("muon", energies, quadrature_order=48)
    reference_ranges = np.interp(energies, legacy_energy, legacy_ranges)
    mask = energies >= 100.0
    max_relative = float(
        np.max(
            np.abs(nominal_ranges[mask] - reference_ranges[mask])
            / reference_ranges[mask]
        )
    )
    _check(
        max_relative < 0.0035,
        "nominal PDG/ICRU-90 implementation differs from the existing muon range by >=0.35%",
    )
    return max_relative


def main():
    _, pstar_max_relative, pdg79p7_max_relative = validate_metadata()
    report = {particle: validate_one_particle(particle) for particle in PARTICLES}
    muon_max_relative = validate_virtual_muon_against_existing_table()

    print("Particle range-table validation: PASS")
    for particle, values in report.items():
        print(
            f"  {particle:6s} threshold={values['threshold_mev']:.6f} MeV  "
            f"rows={values['rows']:4d}  "
            f"R_visible(3 GeV)={values['range_at_3gev_cm']:.6f} cm"
        )
    print(
        "  nominal implementation vs existing muon table (100-3000 MeV): "
        f"max |delta R/R|={100.0 * muon_max_relative:.4f}%"
    )
    print(
        "  nominal proton vs NIST PSTAR-75 comparison (500-3000 MeV): "
        f"max |delta R/R|={100.0 * pstar_max_relative:.4f}%"
    )
    print(
        "  nominal proton vs PDG I=79.7 eV systematic (500-3000 MeV): "
        f"max |delta R/R|={100.0 * pdg79p7_max_relative:.4f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
