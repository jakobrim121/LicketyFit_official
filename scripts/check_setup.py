"""Quick self-contained setup check for LF_multiParticles.

Run from anywhere with:
    python LF_multiParticles/scripts/check_setup.py
"""
from pathlib import Path
import os
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LICKETYFIT_DIR = PROJECT_ROOT / "LicketyFit"
TABLE_DIR = PROJECT_ROOT / "tables"

for p in (str(LICKETYFIT_DIR), str(SCRIPT_DIR), str(PROJECT_ROOT), str(TABLE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["LF_TABLE_DIR"] = str(TABLE_DIR)
os.environ["LF_MULTIPARTICLES_TABLE_DIR"] = str(TABLE_DIR)

from particle_cherenkov_model import (  # noqa: E402
    get_energy_distance_tables,
    get_cerenkov_angle_table,
)
from particle_range_lookup import ParticleRangeLookup  # noqa: E402
from LicketyFit.Emitter import Emitter  # noqa: E402

required = [
    "E_vs_dist_cm_muon.npy",
    "overall_distances_cm_muon.npy",
    "cherenkov_angle_vs_E_muon_n1344.npy",
    "E_vs_dist_cm_pion.npy",
    "overall_distances_cm_pion.npy",
    "E_vs_dist_cm_kaon.npy",
    "overall_distances_cm_kaon.npy",
    "E_vs_dist_cm_proton.npy",
    "overall_distances_cm_proton.npy",
]

print("Project root:", PROJECT_ROOT)
print("Tables dir:  ", TABLE_DIR)
print("\nRequired core tables:")
for name in required:
    path = TABLE_DIR / name
    print(f"  {'OK' if path.exists() else 'MISSING':7s} {name}")

optional = [
    "wcsim_wcte_mapping.txt",
    "rel_mpmt_eff.dict",
    "other_mpmt_info_v2.dict",
    "delta_e_angular_pdf_table.npz",
]
print("\nOptional/external-mode tables:")
for name in optional:
    path = TABLE_DIR / name
    print(f"  {'OK' if path.exists() else 'optional':7s} {name}")

print("\nParticle table summary:")
for particle in ("muon", "pion", "kaon", "proton"):
    overall, energy_rows, distance_rows = get_energy_distance_tables(particle)
    angles, angle_E = get_cerenkov_angle_table(particle)
    lookup = ParticleRangeLookup(particle, table_dirs=[str(TABLE_DIR)])
    print(
        f"  {particle:6s}: rows={len(overall):5d}, "
        f"E0=[{energy_rows[0][0]:.1f}, {energy_rows[-1][0]:.1f}] MeV, "
        f"range_max={overall[-1]:.1f} mm, "
        f"threshold={lookup.threshold_mev:.3f} MeV, "
        f"angle_points={len(angles)}"
    )

print("\nEndpoint-mode smoke check:")
em_threshold = Emitter(0.0, (0, 0, 0), (0, 0, 1), 0.96, 500.0, 18.0, particle="proton")
ke_threshold = em_threshold.refresh_kinematics_from_length(500.0)
print(f"  threshold mode: length=500 mm -> K0={ke_threshold:.3f} MeV")

em_abrupt = Emitter(
    0.0,
    (0, 0, 0),
    (0, 0, 1),
    0.96,
    500.0,
    18.0,
    particle="proton",
    track_end_mode="abrupt",
    fixed_initial_KE=1000.0,
)
ke_abrupt = em_abrupt.refresh_kinematics_from_length(500.0)
print(
    "  abrupt mode:    visible=500 mm, "
    f"fixed K0={ke_abrupt:.3f} MeV, "
    f"range_to_threshold={em_abrupt.range_to_threshold_mm:.1f} mm"
)

print("\nImport check: OK")
