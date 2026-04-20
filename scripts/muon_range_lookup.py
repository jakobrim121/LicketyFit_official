import numpy as np

# Hard-coded table paths
E_VS_DIST_PATH = "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/E_vs_dist_cm.npy"
OVERALL_DISTANCES_PATH = "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/overall_distances_cm.npy"


class MuonRangeLookup:
    """
    Helper for converting between muon initial kinetic energy and total
    travel distance before the muon falls below Cherenkov threshold.

    Notes
    -----
    - E_vs_dist_cm.npy is assumed to be an object array where each entry is
      one trajectory with two columns:
          column 0 -> distance travelled in cm
          column 1 -> muon kinetic energy in MeV
    - overall_distances_cm.npy is assumed to contain the total trajectory
      distance (in cm) for each corresponding initial-energy trajectory.
    - Distances returned by this helper are in mm.
    """

    def __init__(
        self,
        e_vs_dist_path: str = E_VS_DIST_PATH,
        overall_distances_path: str = OVERALL_DISTANCES_PATH,
    ):
        self.e_vs_dist = np.load(e_vs_dist_path, allow_pickle=True)
        self.overall_distances_mm = np.load(overall_distances_path) * 10.0

        # Initial kinetic energy for each full trajectory
        self.initial_energies_mev = np.array(
            [traj[0, 1] for traj in self.e_vs_dist], dtype=float
        )

        # Sort by initial energy so interpolation is well-defined
        order = np.argsort(self.initial_energies_mev)
        self.initial_energies_mev = self.initial_energies_mev[order]
        self.overall_distances_mm = self.overall_distances_mm[order]
        self.e_vs_dist = self.e_vs_dist[order]

    def energy_to_range_mm(self, kinetic_energy_mev: float) -> float:
        """
        Return the total distance in mm that a muon with the given initial
        kinetic energy travels before falling below Cherenkov threshold.
        """
        return float(
            np.interp(
                kinetic_energy_mev,
                self.initial_energies_mev,
                self.overall_distances_mm,
            )
        )

    def range_mm_to_energy(self, travel_distance_mm: float) -> float:
        """
        Return the initial muon kinetic energy in MeV corresponding to a muon
        whose total distance-to-threshold is the given travel distance in mm.
        """
        return float(
            np.interp(
                travel_distance_mm,
                self.overall_distances_mm,
                self.initial_energies_mev,
            )
        )


# Module-level singleton so the tables are loaded only once on import
_DEFAULT_LOOKUP = MuonRangeLookup()


def muon_energy_to_range_mm(kinetic_energy_mev: float) -> float:
    """
    Take a muon initial kinetic energy in MeV and return the total distance
    travelled in mm before the muon falls below Cherenkov threshold.
    """
    return _DEFAULT_LOOKUP.energy_to_range_mm(kinetic_energy_mev)



def muon_range_mm_to_energy(travel_distance_mm: float) -> float:
    """
    Take a total travel distance in mm and return the corresponding initial
    muon kinetic energy in MeV.
    """
    return _DEFAULT_LOOKUP.range_mm_to_energy(travel_distance_mm)


if __name__ == "__main__":
    # Simple example usage
    test_energy = 300.0
    test_range = muon_energy_to_range_mm(test_energy)
    print(f"Initial energy {test_energy:.1f} MeV -> range {test_range:.2f} mm")
    print(
        f"Range {test_range:.2f} mm -> initial energy "
        f"{muon_range_mm_to_energy(test_range):.2f} MeV"
    )

