#!/usr/bin/env python3
"""Offline validation of the analysis_tools BeamSelection controls exposed by LicketyFit.

This does not open a ROOT file.  It verifies nominal/custom cut construction,
zero-TOF behavior, scalar overrides, branch resolution, and the separate
selection-versus-fit particle contract used by ``batch_fit_driver.py``.
"""

from __future__ import annotations

from pathlib import Path
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
}


def _specs(particle: str, **kwargs):
    cfg = WCTESelectionConfig(run=1, root_file="unused.root", particle=particle, **kwargs)
    thresholds = _selection_thresholds(_ScalarLoader(FULL), cfg, particle)
    return _selection_specs(particle, thresholds, cfg)


def main() -> None:
    assert _specs("muon") == [
        ["vme_act_eveto", "<", 2.0],
        ["vme_act_tagger", ">", 3.0],
        ["vme_tof_corr", "<", 12.0],
    ]
    assert _specs("pion") == [
        ["vme_act_eveto", "<", 2.0],
        ["vme_act_tagger", "<", 3.0],
        ["vme_tof_corr", "<", 12.0],
    ]
    assert _specs("electron") == [
        ["vme_act_eveto", ">", 2.0],
        ["vme_tof_corr", "<", 12.0],
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

    zero_tof_cfg = WCTESelectionConfig(
        run=1, root_file="unused.root", particle="muon", tof_cut_mode="auto"
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

    print("WCTE selection-control validation passed")
    print("  nominal particles: muon, pion, electron, proton")
    print("  custom selection: arbitrary BeamSelection triplets, including kaon labels")
    print("  zero-TOF auto mode: fast-particle TOF cut omitted; proton requires override/custom")


if __name__ == "__main__":
    main()
