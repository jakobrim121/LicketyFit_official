"""
particle_range_lookup.py

Generic replacement for the old muon_range_lookup.py helper.

It supports muons, charged pions, charged kaons, and protons using the same
E_vs_dist/overall_distances table format as the existing muon lookup.
Distances returned by this helper are in mm.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _default_table_dirs():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dirs = []
    for env_name in ("LF_TABLE_DIR", "LF_MULTIPARTICLES_TABLE_DIR"):
        env = os.environ.get(env_name)
        if env:
            dirs.extend([d for d in env.split(os.pathsep) if d])
    dirs.extend([
        str(project_root / "tables"),
        str(script_dir / "tables"),
        str(Path.cwd() / "tables"),
        str(script_dir),
        str(Path.cwd()),
        # Old CERN locations are last-resort fallbacks only.
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


DEFAULT_TABLE_DIRS = _default_table_dirs()

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


def canonical_particle_name(particle: str) -> str:
    key = str(particle).strip().lower()
    if key not in PARTICLE_ALIASES:
        raise ValueError(
            f"Unknown particle {particle!r}. Use one of: "
            f"{', '.join(sorted(PARTICLE_ALIASES))}"
        )
    return PARTICLE_ALIASES[key]


def cherenkov_threshold_kinetic_mev(particle_or_mass, n: float = 1.344) -> float:
    if isinstance(particle_or_mass, str):
        mass = PARTICLE_MASS_MEV[canonical_particle_name(particle_or_mass)]
    else:
        mass = float(particle_or_mass)
    beta_thr = 1.0 / float(n)
    gamma_thr = 1.0 / np.sqrt(1.0 - beta_thr * beta_thr)
    return float(mass * (gamma_thr - 1.0))


def _first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find any particle range table. Checked:\n"
        + "\n".join(str(p) for p in paths if p)
    )


def _candidate_paths(filename: str, table_dirs=None):
    dirs = DEFAULT_TABLE_DIRS if table_dirs is None else table_dirs
    return [str(Path(d) / filename) for d in dirs]


def default_range_table_paths(particle: str, table_dirs=None):
    """
    Return (E_vs_dist_path, overall_distances_path) for a particle.
    """
    pname = canonical_particle_name(particle)

    evsd_candidates = _candidate_paths(f"E_vs_dist_cm_{pname}.npy", table_dirs)
    overall_candidates = _candidate_paths(f"overall_distances_cm_{pname}.npy", table_dirs)

    # Backward-compatible fallback for the original muon filenames.
    if pname == "muon":
        evsd_candidates.extend(_candidate_paths("E_vs_dist_cm.npy", table_dirs))
        overall_candidates.extend(_candidate_paths("overall_distances_cm.npy", table_dirs))

    return _first_existing(evsd_candidates), _first_existing(overall_candidates)


class ParticleRangeLookup:
    """
    Convert between initial kinetic energy and above-threshold Cherenkov range.

    Parameters
    ----------
    particle : str
        One of muon, pion, kaon, proton, or common aliases such as mu-, pi+, K+.
    e_vs_dist_path, overall_distances_path : str, optional
        Explicit lookup table paths. If omitted, default paths are searched.
    """

    def __init__(
        self,
        particle: str = "muon",
        e_vs_dist_path: str | None = None,
        overall_distances_path: str | None = None,
        table_dirs=None,
    ):
        self.particle = canonical_particle_name(particle)
        self.mass_mev = PARTICLE_MASS_MEV[self.particle]
        self.threshold_mev = cherenkov_threshold_kinetic_mev(self.mass_mev)

        if e_vs_dist_path is None or overall_distances_path is None:
            default_evsd, default_overall = default_range_table_paths(
                self.particle,
                table_dirs=table_dirs,
            )
            if e_vs_dist_path is None:
                e_vs_dist_path = default_evsd
            if overall_distances_path is None:
                overall_distances_path = default_overall

        self.e_vs_dist_path = str(e_vs_dist_path)
        self.overall_distances_path = str(overall_distances_path)

        self.e_vs_dist = np.load(self.e_vs_dist_path, allow_pickle=True)
        self.overall_distances_mm = np.load(self.overall_distances_path) * 10.0

        if (
            self.e_vs_dist.dtype != object
            and self.e_vs_dist.ndim == 2
            and self.e_vs_dist.shape[1] == 2
        ):
            compact_ranges_mm = np.asarray(self.e_vs_dist[:, 0], dtype=np.float64) * 10.0
            self.initial_energies_mev = np.asarray(
                self.e_vs_dist[:, 1], dtype=np.float64
            )
            if compact_ranges_mm.size != self.overall_distances_mm.size or not np.allclose(
                compact_ranges_mm,
                self.overall_distances_mm,
                rtol=0.0,
                atol=1.0e-8,
            ):
                raise ValueError(
                    f"Compact range columns disagree for {self.particle}; "
                    "regenerate both table files together."
                )
        elif self.e_vs_dist.dtype == object and self.e_vs_dist.ndim == 1:
            self.initial_energies_mev = np.asarray(
                [float(np.asarray(traj)[0, 1]) for traj in self.e_vs_dist],
                dtype=np.float64,
            )
        else:
            raise ValueError(
                f"Unrecognized E_vs_dist format for {self.particle}: "
                f"shape={self.e_vs_dist.shape}, dtype={self.e_vs_dist.dtype}."
            )

        if self.initial_energies_mev.size != self.overall_distances_mm.size:
            raise ValueError(
                f"Range table row-count mismatch for {self.particle}: "
                f"{self.initial_energies_mev.size} energies versus "
                f"{self.overall_distances_mm.size} distances."
            )

        order = np.argsort(self.initial_energies_mev)
        self.initial_energies_mev = self.initial_energies_mev[order]
        self.overall_distances_mm = self.overall_distances_mm[order]
        self.e_vs_dist = self.e_vs_dist[order]

        good = (
            np.isfinite(self.initial_energies_mev)
            & np.isfinite(self.overall_distances_mm)
            & (self.overall_distances_mm >= 0.0)
        )
        self.initial_energies_mev = self.initial_energies_mev[good]
        self.overall_distances_mm = self.overall_distances_mm[good]
        self.e_vs_dist = self.e_vs_dist[good]

        if self.initial_energies_mev.size < 2:
            raise ValueError(f"Range table for {self.particle} has too few rows.")

    def energy_to_range_mm(self, kinetic_energy_mev: float) -> float:
        """
        Return the above-threshold Cherenkov-visible range in mm.
        """
        return float(
            np.interp(
                float(kinetic_energy_mev),
                self.initial_energies_mev,
                self.overall_distances_mm,
                left=0.0,
                right=self.overall_distances_mm[-1],
            )
        )

    def range_mm_to_energy(self, travel_distance_mm: float) -> float:
        """
        Return the initial kinetic energy corresponding to an above-threshold
        range in mm.
        """
        return float(
            np.interp(
                float(travel_distance_mm),
                self.overall_distances_mm,
                self.initial_energies_mev,
                left=self.initial_energies_mev[0],
                right=self.initial_energies_mev[-1],
            )
        )


_LOOKUP_CACHE = {}


def get_lookup(particle: str = "muon") -> ParticleRangeLookup:
    pname = canonical_particle_name(particle)
    lookup = _LOOKUP_CACHE.get(pname)
    if lookup is None:
        lookup = ParticleRangeLookup(pname)
        _LOOKUP_CACHE[pname] = lookup
    return lookup


def particle_energy_to_range_mm(particle: str, kinetic_energy_mev: float) -> float:
    return get_lookup(particle).energy_to_range_mm(kinetic_energy_mev)


def particle_range_mm_to_energy(particle: str, travel_distance_mm: float) -> float:
    return get_lookup(particle).range_mm_to_energy(travel_distance_mm)


# Backward-compatible names used by old notebooks/scripts.
class MuonRangeLookup(ParticleRangeLookup):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("particle", "muon")
        super().__init__(*args, **kwargs)


def muon_energy_to_range_mm(kinetic_energy_mev: float) -> float:
    return particle_energy_to_range_mm("muon", kinetic_energy_mev)


def muon_range_mm_to_energy(travel_distance_mm: float) -> float:
    return particle_range_mm_to_energy("muon", travel_distance_mm)


if __name__ == "__main__":
    for particle in ("muon", "pion", "kaon", "proton"):
        lookup = get_lookup(particle)
        e = max(lookup.threshold_mev + 100.0, lookup.initial_energies_mev[0])
        r = lookup.energy_to_range_mm(e)
        print(f"{particle:6s}: {e:8.2f} MeV -> {r:9.2f} mm")
