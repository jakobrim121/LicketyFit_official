#!/usr/bin/env python3
"""Offline validation of the analysis_tools BeamSelection controls exposed by LicketyFit.

This does not open a ROOT file. It verifies ACT, ACT+TOF and TOF-only nominal
construction, calibrated light-particle boundaries, zero-TOF behavior, scalar
overrides, custom cuts, and branch resolution.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from wcte_data_loader_adapter import (  # noqa: E402
    WCTESelectionConfig,
    _selection_branch_name,
    _selection_specs,
    _selection_thresholds,
)


class _ScalarLoader:
    def __init__(self, values):
        self.values = dict(values)

    def get_vme_analysis_scalar_results(self):
        return dict(self.values)


FULL = {
    "proton_tof_cut": 12.0,
    "act_eveto_cut": 2.0,
    "act_tagger_cut": 3.0,
    "mu_tag_cut": 4.0,
    "tof_mean_electron": 1.0,
    "tof_std_electron": 0.2,
    "tof_mean_muon": 2.0,
    "tof_std_muon": 0.2,
    "tof_mean_pion": 3.0,
    "tof_std_pion": 0.2,
}

# Production calibration snapshots used to guard the run-by-run calculation.
# The two runs deliberately have different nominal momenta and very different
# peak separations; no boundary from one run is reused for the other.
RUN_CALIBRATIONS = {
    2079: {
        "momentum_mev_c": 430.0,
        "proton_tof_cut": 31.23748796445347,
        "act_eveto_cut": 4.522613065326634,
        "act_tagger_cut": 13.333333333333332,
        "tof_mean_electron": 14.621782113010996,
        "tof_std_electron": 0.19591188323436146,
        "tof_mean_muon": 15.098534425878453,
        "tof_std_muon": 0.1935474943583112,
        "tof_mean_pion": 15.434338555313719,
        "tof_std_pion": 0.18543939931312584,
        "expected_electron_muon_boundary": 14.860639751277143,
        "expected_muon_pion_boundary": 15.265457291289914,
    },
    1775: {
        "momentum_mev_c": 260.0,
        "proton_tof_cut": 54.89340804367443,
        "act_eveto_cut": 2.9145728643216082,
        "act_tagger_cut": 3.6363636363636367,
        "tof_mean_electron": 14.621782113010996,
        "tof_std_electron": 0.19667797366052792,
        "tof_mean_muon": 15.934613206756838,
        "tof_std_muon": 0.20132078417122745,
        "tof_mean_pion": 16.815182201257684,
        "tof_std_pion": 0.23227668815758956,
        "expected_electron_muon_boundary": 15.27124400301956,
        "expected_muon_pion_boundary": 16.351050525118296,
    },
}


def _specs(particle: str, **kwargs):
    cfg = WCTESelectionConfig(run=1, root_file="unused.root", particle=particle, **kwargs)
    thresholds = _selection_thresholds(_ScalarLoader(FULL), cfg, particle)
    return _selection_specs(particle, thresholds, cfg)


def main() -> None:
    assert _specs("muon") == [
        ["vme_act_eveto", "<", 2.0],
        ["vme_act_tagger", ">", 3.0],
        ["vme_tof_corr", "between", [1.5, 2.5]],
    ]
    assert _specs("pion") == [
        ["vme_act_eveto", "<", 2.0],
        ["vme_act_tagger", "<", 3.0],
        ["vme_tof_corr", "between", [2.5, 12.0]],
    ]
    assert _specs("electron") == [
        ["vme_act_eveto", ">", 2.0],
        ["vme_tof_corr", "<", 1.5],
    ]
    assert _specs("proton") == [
        ["vme_tof_corr", "between", [12.0, 22.0]],
    ]
    assert _specs("muon", require_muon_tagger=True)[-1] == [
        "vme_mu_tag_total", ">", 4.0
    ]
    assert _specs(
        "electron", extra_selection_cuts=(("vme_act0_l_charge", ">", 10.0),)
    )[-1] == ["vme_act0_l_charge", ">", 10.0]
    assert _specs("muon", light_particle_pid_mode="act") == [
        ["vme_act_eveto", "<", 2.0],
        ["vme_act_tagger", ">", 3.0],
        ["vme_tof_corr", "<", 12.0],
    ]
    assert _specs("muon", light_particle_pid_mode="tof") == [
        ["vme_tof_corr", "between", [1.5, 2.5]],
    ]
    assert _specs(
        "muon",
        electron_muon_tof_boundary_override_ns=1.6,
        muon_pion_tof_boundary_override_ns=2.4,
    )[-1] == ["vme_tof_corr", "between", [1.6, 2.4]]

    explicit = dict(FULL)
    explicit.update(muon_tof_cut=1.7, pion_tof_cut=2.3)
    explicit_cfg = WCTESelectionConfig(
        run=1, root_file="unused.root", particle="muon"
    )
    explicit_thresholds = _selection_thresholds(
        _ScalarLoader(explicit), explicit_cfg, "muon"
    )
    assert explicit_thresholds["electron_muon_tof_boundary_ns"] == 1.7
    assert explicit_thresholds["muon_pion_tof_boundary_ns"] == 2.3

    resolved_by_run = {}
    for run, calibration in RUN_CALIBRATIONS.items():
        scalar_values = {
            key: value
            for key, value in calibration.items()
            if key not in {
                "momentum_mev_c",
                "expected_electron_muon_boundary",
                "expected_muon_pion_boundary",
            }
        }
        cfg = WCTESelectionConfig(
            run=run, root_file="unused.root", particle="muon"
        )
        thresholds = _selection_thresholds(
            _ScalarLoader(scalar_values), cfg, "muon"
        )
        electron_muon = thresholds["electron_muon_tof_boundary_ns"]
        muon_pion = thresholds["muon_pion_tof_boundary_ns"]
        assert math.isclose(
            electron_muon,
            calibration["expected_electron_muon_boundary"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        assert math.isclose(
            muon_pion,
            calibration["expected_muon_pion_boundary"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        resolved_by_run[run] = (electron_muon, muon_pion)
    assert resolved_by_run[2079] != resolved_by_run[1775]

    zero_tof_cfg = WCTESelectionConfig(
        run=1,
        root_file="unused.root",
        particle="muon",
        light_particle_pid_mode="act",
        tof_cut_mode="auto",
    )
    zero_tof = dict(FULL)
    zero_tof["proton_tof_cut"] = 0.0
    thresholds = _selection_thresholds(_ScalarLoader(zero_tof), zero_tof_cfg, "muon")
    assert thresholds["proton_tof_cut_ns"] is None
    assert all(spec[0] != "vme_tof_corr" for spec in _selection_specs(
        "muon", thresholds, zero_tof_cfg
    ))

    minimal_proton_cfg = WCTESelectionConfig(
        run=1, root_file="unused.root", particle="proton"
    )
    thresholds = _selection_thresholds(
        _ScalarLoader({"proton_tof_cut": 12.0}), minimal_proton_cfg, "proton"
    )
    assert _selection_specs("proton", thresholds, minimal_proton_cfg) == [
        ["vme_tof_corr", "between", [12.0, 22.0]]
    ]

    custom_cfg = WCTESelectionConfig(
        run=1,
        root_file="unused.root",
        particle="kaon",
        selection_mode="custom",
        extra_selection_cuts=(
            ("vme_tof_corr", "between", [5.0, 8.0]),
            ("T5_particle_nr", "==", 1),
        ),
    )
    custom_thresholds = _selection_thresholds(_ScalarLoader({}), custom_cfg, "kaon")
    custom = _selection_specs("kaon", custom_thresholds, custom_cfg)
    assert [_selection_branch_name(spec[0]) for spec in custom] == [
        "vme_tof_corr", "T5_particle_nr"
    ]

    try:
        _specs("muon", light_particle_pid_mode="act_tof", **{
            "electron_muon_tof_boundary_override_ns": 3.0,
            "muon_pion_tof_boundary_override_ns": 2.0,
        })
    except RuntimeError as exc:
        assert "not ordered" in str(exc)
    else:
        raise AssertionError("unordered light-particle TOF boundaries were accepted")

    try:
        _specs("pion", light_particle_pid_mode="tof", tof_cut_mode="disable")
    except RuntimeError as exc:
        assert "positive proton_tof_cut" in str(exc)
    else:
        raise AssertionError("uncapped TOF-only pion selection was accepted")

    print("WCTE selection-control validation passed")
    print("  light-particle PID modes: act, act_tof, tof")
    print("  nominal particles: muon, pion, electron, proton")
    print("  custom selection: arbitrary BeamSelection triplets, including kaon labels")
    print("  boundaries: explicit scalar/override or equal-PDF run calibration")
    print("  run regression: r2079 at 430 MeV/c and r1775 at 260 MeV/c")
    print("  zero proton-TOF auto mode: legacy fast cut omitted; pion/proton stay capped")


if __name__ == "__main__":
    main()
