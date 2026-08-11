#!/usr/bin/env python3
"""Generate LicketyFit's charged-pion, charged-kaon, and proton range tables.

The fitted length is the continuous-slowing-down distance in liquid water from
the initial kinetic energy to the particle's Cherenkov threshold at n=1.344.
It is *not* a sampled hadronic interaction length or decay length.

Nominal tables use the PDG Bethe equation, the exact finite-projectile-mass
T_max expression, and the Sternheimer density correction for liquid water.
The water mean excitation energy is I=78 eV, the ICRU Report 90 value used by
Geant4's G4_WATER material and consistent with direct proton-range data.  The
checked-in NIST PSTAR snapshot (I=75 eV) and the PDG Atomic and Nuclear
Properties water value (I=79.7 eV) are retained as independent comparisons.

The output ``E_vs_dist`` files use LicketyFit compact range-table format v1:
each numeric row is ``[above_threshold_range_cm, initial_kinetic_energy_MeV]``.
The runtime lazily reconstructs the historical per-energy K(s) trajectories.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import unescape
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from particle_range_lookup import (  # noqa: E402
    PARTICLE_MASS_MEV,
    cherenkov_threshold_kinetic_mev,
)


TABLE_FORMAT = "licketyfit_compact_range_v1"
N_WATER = 1.344
MAX_KINETIC_ENERGY_MEV = 3000.0
WATER_DENSITY_G_CM3 = 1.0

# PDG Review of Particle Physics, Passage of Particles Through Matter,
# Eqs. 34.4, 34.5, and 34.7 (2024 numbering).
BETHE_K_MEV_CM2_MOL = 0.307075
ELECTRON_MASS_MEV = 0.51099895000
WATER_Z_OVER_A_MOL_G = 0.55509
# ICRU Report 90 recommends 78 eV for liquid water.  This is also the value in
# the current Geant4 G4_WATER definition and agrees with the 78.4 +/- 1.0 eV
# proton-range measurement of Akagi et al.  PDG's Atomic and Nuclear Properties
# page lists 79.7 eV; it is evaluated explicitly as a systematic comparison.
WATER_MEAN_EXCITATION_ENERGY_EV = 78.0
PDG_WATER_MEAN_EXCITATION_ENERGY_EV = 79.7
NIST_PSTAR_WATER_MEAN_EXCITATION_ENERGY_EV = 75.0
WATER_STERNHEIMER = {
    "a": 0.0912,
    "m": 3.4773,
    "x0": 0.2400,
    "x1": 2.8004,
    "Cbar": 3.5017,
    "delta0": 0.0,
}

PDG_PASSAGE_URL = (
    "https://pdg.lbl.gov/2024/reviews/"
    "rpp2024-rev-passage-particles-matter.pdf"
)
PDG_WATER_URL = (
    "https://pdg.lbl.gov/2023/AtomicNuclearProperties/HTML/water_liquid.html"
)
NIST_PSTAR_URL = "https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html"
NIST_PSTAR_CGI_URL = "https://physics.nist.gov/cgi-bin/Star/ap_table.pl"
NIST_WATER_COMPOSITION_URL = (
    "https://physics.nist.gov/cgi-bin/Star/"
    "compos.pl?matno=276&mode=text&refer=ap"
)
ICRU_90_URL = (
    "https://www.icru.org/report/"
    "icru-report-90-key-data-for-ionizing-radiation-dosimetry-"
    "measurement-standards-and-applications/"
)
ICRU_49_URL = (
    "https://www.icru.org/report/"
    "stopping-power-and-ranges-for-protons-and-alpha-particles-report-49/"
)
GEANT4_WATER_SOURCE_URL = (
    "https://github.com/Geant4/geant4/blob/master/source/materials/src/"
    "G4NistMaterialBuilder.cc"
)
AKAGI_WATER_I_DOI_URL = "https://doi.org/10.1016/j.radmeas.2007.10.019"

SUBTHRESHOLD_RANGE_MM = {
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


def sternheimer_density_effect(beta_gamma):
    """PDG/Sternheimer liquid-water density-effect correction delta(beta*gamma)."""
    bg = np.asarray(beta_gamma, dtype=np.float64)
    if np.any(bg <= 0.0):
        raise ValueError("beta_gamma must be positive")

    pars = WATER_STERNHEIMER
    x = np.log10(bg)
    delta = np.zeros_like(x)
    high = x >= pars["x1"]
    middle = (x >= pars["x0"]) & (x < pars["x1"])
    delta[high] = 2.0 * math.log(10.0) * x[high] - pars["Cbar"]
    delta[middle] = (
        2.0 * math.log(10.0) * x[middle]
        - pars["Cbar"]
        + pars["a"] * (pars["x1"] - x[middle]) ** pars["m"]
    )
    # Liquid water is a nonconductor, so delta=0 below x0.
    return delta


def pdg_bethe_mass_stopping_power(
    kinetic_energy_mev,
    mass_mev,
    *,
    spin_half: bool = False,
    mean_excitation_energy_ev: float = WATER_MEAN_EXCITATION_ENERGY_EV,
):
    """Mean electronic stopping power in liquid water [MeV cm^2/g].

    The table domain begins at beta*gamma=1.112, comfortably inside the PDG's
    quoted 0.1 < beta*gamma < 1000 Bethe region. Charged pions and kaons are
    spin zero. The small PDG spin-1/2 term is included when requested.
    """
    kinetic = np.asarray(kinetic_energy_mev, dtype=np.float64)
    mass = float(mass_mev)
    if mass <= 0.0 or np.any(kinetic <= 0.0):
        raise ValueError("mass and kinetic energy must be positive")

    gamma = 1.0 + kinetic / mass
    beta_gamma = np.sqrt(np.maximum(gamma * gamma - 1.0, 0.0))
    beta2 = beta_gamma * beta_gamma / (1.0 + beta_gamma * beta_gamma)
    tmax = (
        2.0 * ELECTRON_MASS_MEV * beta_gamma * beta_gamma
        / (
            1.0
            + 2.0 * gamma * ELECTRON_MASS_MEV / mass
            + (ELECTRON_MASS_MEV / mass) ** 2
        )
    )
    excitation_mev = float(mean_excitation_energy_ev) * 1.0e-6
    delta = sternheimer_density_effect(beta_gamma)
    bracket = (
        0.5
        * np.log(
            2.0
            * ELECTRON_MASS_MEV
            * beta2
            * gamma
            * gamma
            * tmax
            / (excitation_mev * excitation_mev)
        )
        - beta2
        - 0.5 * delta
    )
    if spin_half:
        total_energy = kinetic + mass
        bracket = bracket + 0.25 * (tmax / total_energy) ** 2

    stopping = (
        BETHE_K_MEV_CM2_MOL
        * WATER_Z_OVER_A_MOL_G
        * bracket
        / beta2
    )
    if np.any(~np.isfinite(stopping)) or np.any(stopping <= 0.0):
        raise ValueError("PDG stopping-power evaluation produced invalid values")
    return stopping


def integrate_visible_range_cm(
    particle: str,
    initial_energies_mev,
    *,
    quadrature_order: int = 24,
    mean_excitation_energy_ev: float = WATER_MEAN_EXCITATION_ENERGY_EV,
):
    """Integrate dT/(rho*S) from threshold to each requested energy."""
    particle = str(particle).lower()
    if particle not in {"muon", "pion", "kaon", "proton"}:
        raise ValueError(f"Unsupported particle: {particle}")
    energies = np.asarray(initial_energies_mev, dtype=np.float64)
    if energies.ndim != 1 or energies.size == 0 or np.any(np.diff(energies) <= 0.0):
        raise ValueError("initial energies must be a nonempty, strictly increasing 1-D array")

    mass = float(PARTICLE_MASS_MEV[particle])
    threshold = cherenkov_threshold_kinetic_mev(mass, n=N_WATER)
    if energies[0] <= threshold:
        raise ValueError("all initial energies must be above the Cherenkov threshold")

    edges = np.concatenate(([threshold], energies))
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_order))
    lo = edges[:-1, None]
    hi = edges[1:, None]
    sample_energy = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
    stopping = pdg_bethe_mass_stopping_power(
        sample_energy,
        mass,
        spin_half=particle in {"muon", "proton"},
        mean_excitation_energy_ev=mean_excitation_energy_ev,
    )
    increments = (
        0.5
        * (hi[:, 0] - lo[:, 0])
        * np.sum(weights / stopping, axis=1)
        / WATER_DENSITY_G_CM3
    )
    ranges = np.cumsum(increments, dtype=np.float64)
    if np.any(~np.isfinite(ranges)) or np.any(np.diff(ranges) <= 0.0):
        raise ValueError("integrated range is not finite and strictly increasing")
    return ranges


def _parse_pstar_html(payload: str):
    cells = [
        unescape(value).strip()
        for value in re.findall(
            r"<td\s+align=center>(.*?)</td>", payload, flags=re.IGNORECASE | re.DOTALL
        )
    ]
    if len(cells) % 7 != 0:
        raise RuntimeError(f"Unexpected PSTAR response with {len(cells)} numeric cells")
    rows = np.asarray([float(value) for value in cells], dtype=np.float64).reshape(-1, 7)
    return rows


def fetch_pstar_water_reference(
    first_energy_mev: int = 450,
    last_energy_mev: int = 3000,
    *,
    batch_size: int = 250,
):
    """Download an integer-energy liquid-water PSTAR comparison snapshot."""
    requested = np.arange(
        int(first_energy_mev), int(last_energy_mev) + 1, dtype=np.float64
    )
    batches = []
    for start in range(0, requested.size, int(batch_size)):
        batch = requested[start : start + int(batch_size)]
        fields = {
            "prog": "PSTAR",
            "matno": "276",
            "GraphType": "None",
            "Energies": "\n".join(f"{value:.0f}" for value in batch),
        }
        request = Request(
            NIST_PSTAR_CGI_URL,
            data=urlencode(fields).encode("ascii"),
            headers={"User-Agent": "LicketyFit-range-table-generator/1.0"},
        )
        with urlopen(request, timeout=120) as response:
            payload = response.read().decode("latin1")
        rows = _parse_pstar_html(payload)
        if rows.shape[0] != batch.size or not np.array_equal(rows[:, 0], batch):
            raise RuntimeError(
                "PSTAR response energies did not match the requested integer grid"
            )
        batches.append(rows)
    return np.vstack(batches)


def save_pstar_reference(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(rows, dtype=np.float64)
    np.savez_compressed(
        path,
        kinetic_energy_mev=rows[:, 0],
        electronic_stopping_mev_cm2_g=rows[:, 1],
        nuclear_stopping_mev_cm2_g=rows[:, 2],
        total_stopping_mev_cm2_g=rows[:, 3],
        csda_range_g_cm2=rows[:, 4],
        projected_range_g_cm2=rows[:, 5],
        detour_factor=rows[:, 6],
        source_url=np.asarray(NIST_PSTAR_URL),
        material=np.asarray("Water, Liquid (PSTAR material 276)"),
        water_mean_excitation_energy_ev=np.asarray(
            NIST_PSTAR_WATER_MEAN_EXCITATION_ENERGY_EV, dtype=np.float64
        ),
        retrieved_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
    )


def load_pstar_reference(path: Path):
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "kinetic_energy_mev",
            "total_stopping_mev_cm2_g",
            "csda_range_g_cm2",
        }
        if not required.issubset(payload.files):
            raise ValueError(f"PSTAR reference is missing keys: {sorted(required - set(payload.files))}")
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


def _proton_reference_comparison(reference, proton_energies, proton_ranges_cm):
    ref_energy = np.asarray(reference["kinetic_energy_mev"], dtype=np.float64)
    ref_stopping = np.asarray(
        reference["total_stopping_mev_cm2_g"], dtype=np.float64
    )
    nominal_stopping = pdg_bethe_mass_stopping_power(
        ref_energy,
        PARTICLE_MASS_MEV["proton"],
        spin_half=True,
    )
    pdg79p7_stopping = pdg_bethe_mass_stopping_power(
        ref_energy,
        PARTICLE_MASS_MEV["proton"],
        spin_half=True,
        mean_excitation_energy_ev=PDG_WATER_MEAN_EXCITATION_ENERGY_EV,
    )
    checkpoints = np.asarray([500.0, 1000.0, 2000.0, 3000.0])
    pstar_at_checkpoints = np.interp(checkpoints, ref_energy, ref_stopping)
    nominal_at_checkpoints = np.interp(
        checkpoints, ref_energy, nominal_stopping
    )
    pdg79p7_at_checkpoints = np.interp(
        checkpoints, ref_energy, pdg79p7_stopping
    )

    # Independently integrate the evaluated PSTAR total stopping power from the
    # same Cherenkov threshold. A 1 MeV source grid makes simple log-linear
    # interpolation much more precise than the underlying four-digit values.
    threshold = cherenkov_threshold_kinetic_mev("proton", n=N_WATER)
    pstar_edges = np.concatenate(([threshold], proton_energies))
    nodes, weights = np.polynomial.legendre.leggauss(24)
    lo = pstar_edges[:-1, None]
    hi = pstar_edges[1:, None]
    samples = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
    samples_stopping = np.exp(
        np.interp(samples, ref_energy, np.log(ref_stopping))
    )
    increments = 0.5 * (hi[:, 0] - lo[:, 0]) * np.sum(
        weights / samples_stopping, axis=1
    )
    pstar_visible = np.cumsum(increments)
    nominal_ranges_check = np.interp(
        checkpoints, proton_energies, proton_ranges_cm
    )
    pdg79p7_ranges = integrate_visible_range_cm(
        "proton",
        proton_energies,
        mean_excitation_energy_ev=PDG_WATER_MEAN_EXCITATION_ENERGY_EV,
    )
    pdg79p7_ranges_check = np.interp(
        checkpoints, proton_energies, pdg79p7_ranges
    )
    pstar_ranges_check = np.interp(checkpoints, proton_energies, pstar_visible)

    return {
        "checkpoint_energy_mev": checkpoints.tolist(),
        "nominal78_stopping_mev_cm2_g": nominal_at_checkpoints.tolist(),
        "pdg79p7_stopping_mev_cm2_g": pdg79p7_at_checkpoints.tolist(),
        "pstar75_stopping_mev_cm2_g": pstar_at_checkpoints.tolist(),
        "stopping_relative_difference_nominal78_minus_pstar75": (
            (nominal_at_checkpoints - pstar_at_checkpoints)
            / pstar_at_checkpoints
        ).tolist(),
        "stopping_relative_difference_nominal78_minus_pdg79p7": (
            (nominal_at_checkpoints - pdg79p7_at_checkpoints)
            / pdg79p7_at_checkpoints
        ).tolist(),
        "nominal78_visible_range_cm": nominal_ranges_check.tolist(),
        "pdg79p7_visible_range_cm": pdg79p7_ranges_check.tolist(),
        "pstar75_visible_range_cm": pstar_ranges_check.tolist(),
        "visible_range_relative_difference_nominal78_minus_pstar75": (
            (nominal_ranges_check - pstar_ranges_check) / pstar_ranges_check
        ).tolist(),
        "visible_range_relative_difference_nominal78_minus_pdg79p7": (
            (nominal_ranges_check - pdg79p7_ranges_check)
            / pdg79p7_ranges_check
        ).tolist(),
    }


def generate_tables(output_dir: Path, pstar_reference_path: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_pstar_reference(pstar_reference_path)
    particle_metadata = {}
    output_paths = []

    generated = {}
    for particle in ("pion", "kaon", "proton"):
        mass = float(PARTICLE_MASS_MEV[particle])
        threshold = cherenkov_threshold_kinetic_mev(mass, n=N_WATER)
        energies = np.arange(
            math.ceil(threshold),
            int(MAX_KINETIC_ENERGY_MEV) + 1,
            dtype=np.float64,
        )
        ranges_cm = integrate_visible_range_cm(particle, energies)
        compact = np.column_stack((ranges_cm, energies))

        trajectory_path = output_dir / f"E_vs_dist_cm_{particle}.npy"
        distance_path = output_dir / f"overall_distances_cm_{particle}.npy"
        np.save(trajectory_path, compact, allow_pickle=False)
        np.save(distance_path, ranges_cm, allow_pickle=False)
        output_paths.extend((trajectory_path, distance_path))
        generated[particle] = (energies, ranges_cm)
        particle_metadata[particle] = {
            "mass_mev_c2": mass,
            "spin": 0.0 if particle in {"pion", "kaon"} else 0.5,
            "cherenkov_threshold_kinetic_energy_mev": threshold,
            "first_initial_kinetic_energy_mev": float(energies[0]),
            "last_initial_kinetic_energy_mev": float(energies[-1]),
            "energy_step_mev": 1.0,
            "rows": int(energies.size),
            "maximum_above_threshold_range_cm": float(ranges_cm[-1]),
            "method": (
                "PDG Bethe + finite-mass Tmax + Sternheimer density effect; "
                "ICRU-90/Geant4 liquid-water I=78 eV"
            ),
            "approximate_subthreshold_csda_range_mm": SUBTHRESHOLD_RANGE_MM[particle],
            "subthreshold_range_use": (
                "Secondary-electron tail only; not part of the fitted range coordinate"
            ),
        }

    proton_energies, proton_ranges = generated["proton"]
    metadata = {
        "table_format": TABLE_FORMAT,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "semantics": (
            "Electromagnetic CSDA path length in liquid water from initial kinetic "
            "energy to the n=1.344 Cherenkov threshold. Hadronic interactions, "
            "decays, energy-loss straggling, and multiple scattering are not folded "
            "into this deterministic range."
        ),
        "water": {
            "density_g_cm3": WATER_DENSITY_G_CM3,
            "z_over_a_mol_g": WATER_Z_OVER_A_MOL_G,
            "mean_excitation_energy_ev": WATER_MEAN_EXCITATION_ENERGY_EV,
            "refractive_index_used_by_licketyfit": N_WATER,
            "sternheimer": dict(WATER_STERNHEIMER),
        },
        "sources": {
            "pdg_passage_of_particles_through_matter": PDG_PASSAGE_URL,
            "pdg_liquid_water_properties": PDG_WATER_URL,
            "icru_report_90": ICRU_90_URL,
            "icru_report_49": ICRU_49_URL,
            "geant4_g4_water_material_source": GEANT4_WATER_SOURCE_URL,
            "akagi_water_mean_excitation_energy_measurement": AKAGI_WATER_I_DOI_URL,
            "nist_pstar": NIST_PSTAR_URL,
            "nist_pstar_water_composition": NIST_WATER_COMPOSITION_URL,
        },
        "pstar_reference": {
            "path": str(pstar_reference_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(pstar_reference_path),
            "note": (
                "Independent evaluated proton comparison. PSTAR material 276 uses "
                "the historical I=75 eV convention, versus the nominal modern "
                "ICRU-90/Geant4 value I=78 eV."
            ),
        },
        "particles": particle_metadata,
        "proton_reference_comparison": _proton_reference_comparison(
            reference, proton_energies, proton_ranges
        ),
        "files": {},
    }
    for path in output_paths:
        metadata["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    metadata_path = output_dir / "PARTICLE_RANGE_TABLES.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_paths.append(metadata_path)
    return output_paths, metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tables",
        help="Directory receiving LicketyFit range tables (default: tables/)",
    )
    parser.add_argument(
        "--pstar-reference",
        type=Path,
        default=PROJECT_ROOT / "tables" / "reference" / "pstar_water_liquid_450_3000.npz",
        help="Checked-in NIST PSTAR comparison snapshot",
    )
    parser.add_argument(
        "--refresh-pstar",
        action="store_true",
        help="Refresh the official PSTAR snapshot from NIST before generation",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    reference_path = args.pstar_reference.resolve()
    if args.refresh_pstar:
        rows = fetch_pstar_water_reference()
        save_pstar_reference(reference_path, rows)
        print(f"Wrote {reference_path} ({rows.shape[0]} PSTAR rows)")
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Missing {reference_path}. Run once with --refresh-pstar to obtain "
            "the official comparison snapshot."
        )

    paths, metadata = generate_tables(output_dir, reference_path)
    for path in paths:
        print(f"Wrote {path}")
    for particle, info in metadata["particles"].items():
        print(
            f"{particle:6s}: threshold={info['cherenkov_threshold_kinetic_energy_mev']:.6f} MeV, "
            f"R_visible(3 GeV)={info['maximum_above_threshold_range_cm']:.6f} cm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
