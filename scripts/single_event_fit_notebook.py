#!/usr/bin/env python3
"""
Notebook-friendly utilities for fitting and inspecting a single Cherenkov event.

This module is designed to be imported from a Jupyter notebook. It reuses the
same backend logic as batch_fit_driver.py, but exposes it as Python functions
and a small helper class so you can inspect intermediate arrays interactively.

Typical usage in a notebook
---------------------------

from single_event_fit_notebook import SingleEventInspector

inspector = SingleEventInspector(
    input_file="/eos/user/j/jrimmer/sim_work_dir/WCSim/sim_data/mu-/1kmu-_300MeV_x0y0zn1350mm_noScat.npz"
)

prepared = inspector.prepare_event(0)
result = inspector.fit_prepared_event(prepared, seed=123)
summary = inspector.summarize_result(prepared, result)
summary

# Inspect arrays
prepared["obs_pes"]
prepared["obs_ts"]

# Evaluate the FCN at an arbitrary parameter point
inspector.evaluate_prepared_event(prepared, x0=0, y0=0, z0=-1350, cx=0, cy=0, length=1172, t0=0)

# Build the Minuit object directly for manual experimentation
m = inspector.make_minuit_for_prepared_event(prepared)
m.migrad(ncall=6000)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import sys

import numpy as np

# -----------------------------------------------------------------------------
# Path bootstrapping
# -----------------------------------------------------------------------------
# These defaults mirror your current environment. If you already have the right
# paths on sys.path in your notebook, these insertions are harmless.
GEOMETRY_PATH = "/eos/user/j/jrimmer/Geometry"
LICKETYFIT_PATH = "/eos/user/j/jrimmer/SWAN_projects/beam/LicketFit2/LicketyFit"

for p in [
    LICKETYFIT_PATH,
    "../../LicketyFit",
    GEOMETRY_PATH,
    "../",
    "../../",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from Geometry.Device import Device
from read_sim_data import read_sim_data
from LicketyFit.Emitter import Emitter
from LicketyFit.PMT import PMT
import batch_fit_driver as bfd


@dataclass
class PreparedEvent:
    """
    Container holding all useful objects/arrays for one prepared event.

    This is intentionally lightweight and notebook-friendly, so you can inspect
    the fields directly.
    """
    event_index: int
    event_object: Any
    sim_data: Dict[str, np.ndarray]
    obs_pes: np.ndarray
    obs_ts: np.ndarray
    p_locations: np.ndarray
    direction_zs: np.ndarray
    pmt_ids: np.ndarray
    ring_keep_mask: np.ndarray
    n_hits_before_cut: int
    n_hits_after_cut: int


class SingleEventInspector:
    """
    Notebook-oriented wrapper around the single-event fit workflow.

    It initializes the same globals used by batch_fit_driver.py, prepares one
    event at a time, and exposes helpers to fit, evaluate, and inspect results.
    """

    def __init__(
        self,
        input_file: str = "/eos/user/j/jrimmer/sim_work_dir/WCSim/sim_data/mu-/1kmu-_300MeV_x0y0zn1350mm_noScat.npz",
        mapping_file: str = "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/wcsim_wcte_mapping.txt",
        geo_file: str = GEOMETRY_PATH + "/examples/wcte_bldg157.geo",
        cut_time: float = 17.0,
        use_ring_mask: bool = True,
        pe_scale: float = 1.0,
    ):
        self.input_file = input_file
        self.mapping_file = mapping_file
        self.geo_file = geo_file
        self.cut_time = float(cut_time)
        self.use_ring_mask = bool(use_ring_mask)
        self.pe_scale = float(pe_scale)

        self.data_raw: Optional[Dict[str, Any]] = None
        self._initialized = False

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def initialize(self) -> None:
        """
        Initialize the backend globals the same way the batch driver does.

        This needs to happen once before fitting/evaluating events.
        """
        if self._initialized:
            return

        # Lookup tables used by the FCN.
        bfd.OVERALL_DISTANCES, bfd.INIT_ENERGY_TABLE, _ = bfd.get_energy_distance_tables()

        # WCSim -> WCTE PMT mapping.
        wcte_mapping = np.loadtxt(self.mapping_file)
        bfd.SIM_WCTE_MAPPING = {}
        for i in range(len(wcte_mapping)):
            bfd.SIM_WCTE_MAPPING[int(wcte_mapping[i][0])] = int(
                wcte_mapping[i][1] * 100 + wcte_mapping[i][2] - 1
            )

        # Geometry.
        hall = Device.open_file(self.geo_file)
        bfd.WCD = hall.wcds[0]

        # Backend model objects.
        emitter_model = Emitter(
            0.0,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            0.96,
            500.0,
            18.0,
        )
        bfd.PMT_MODEL = PMT(1.0, 0.3, 1.0, 40.0, 0.2, 0.0)
        bfd.EMITTER_TEMPLATE = emitter_model.copy()
        bfd.CORR_POS = None

        # Reset event-dependent globals so they are filled from the first event
        # you prepare in this notebook session.
        bfd.P_LOCATIONS = None
        bfd.DIRECTION_ZS = None
        bfd.RING_KEEP_MASK = None
        bfd.OBS_PES_ALL = None
        bfd.OBS_TS_ALL = None

        self._initialized = True

    def load_data(self) -> Dict[str, Any]:
        """
        Load the input .npz event file once and cache it.
        """
        if self.data_raw is None:
            self.data_raw = read_sim_data(self.input_file)
        return self.data_raw

    # -------------------------------------------------------------------------
    # Event preparation
    # -------------------------------------------------------------------------
    def prepare_event(self, event_index: int) -> PreparedEvent:
        """
        Prepare one event for fitting/evaluation.

        This applies the prompt time cut, builds the Event object, builds the
        observable arrays, and applies the same ring mask used in the batch
        driver (unless use_ring_mask=False was chosen).
        """
        self.initialize()
        data_raw = self.load_data()

        n_events = len(data_raw["digi_hit_time"])
        if event_index < 0 or event_index >= n_events:
            raise IndexError(f"event_index={event_index} is outside [0, {n_events - 1}]")

        hit_times = np.asarray(data_raw["digi_hit_time"][event_index], dtype=np.float64)
        hit_pmts = np.asarray(data_raw["digi_hit_pmt"][event_index], dtype=int)
        hit_charges = np.asarray(data_raw["digi_hit_charge"][event_index], dtype=np.float64)

        keep = (hit_times > 0.0) & (hit_times < self.cut_time)
        sim_data = {
            "digi_hit_pmt": hit_pmts[keep],
            "digi_hit_time": hit_times[keep],
            "digi_hit_charge": hit_charges[keep],
        }

        ev, pmt_ids = bfd.sim_to_event(sim_data, n_mpmt_total=106, pe_scale=self.pe_scale)

        # PMT geometry and ordering must be built from the same Event object so
        # they match the observable-array ordering exactly.
        p_locations, direction_zs = bfd.EMITTER_TEMPLATE.get_pmt_placements(ev, bfd.WCD, "design")
        p_locations = np.asarray(p_locations, dtype=np.float64)
        direction_zs = np.asarray(direction_zs, dtype=np.float64)

        obs_pes, obs_ts = bfd.build_observables_from_event(ev, pe_scale=self.pe_scale)

        if self.use_ring_mask:
            ring_keep_mask = np.isin(pmt_ids // 100, bfd.ALL_RING)
            obs_pes = obs_pes.copy()
            obs_pes[~ring_keep_mask] = 0.0
        else:
            ring_keep_mask = np.ones_like(obs_pes, dtype=bool)

        # Update backend globals so subsequent FCN calls use this event's PMT geometry.
        bfd.P_LOCATIONS = p_locations
        bfd.DIRECTION_ZS = direction_zs
        bfd.RING_KEEP_MASK = ring_keep_mask

        return PreparedEvent(
            event_index=int(event_index),
            event_object=ev,
            sim_data=sim_data,
            obs_pes=obs_pes,
            obs_ts=obs_ts,
            p_locations=p_locations,
            direction_zs=direction_zs,
            pmt_ids=pmt_ids,
            ring_keep_mask=ring_keep_mask,
            n_hits_before_cut=len(hit_times),
            n_hits_after_cut=len(sim_data["digi_hit_time"]),
        )

    # -------------------------------------------------------------------------
    # Core FCN / expected-value helpers
    # -------------------------------------------------------------------------
    def evaluate_prepared_event(
        self,
        prepared: PreparedEvent,
        x0: float,
        y0: float,
        z0: float,
        cx: float,
        cy: float,
        length: float,
        t0: float,
    ) -> float:
        """
        Evaluate the event negative log-likelihood at a chosen parameter point.
        """
        self._activate_prepared_event(prepared)
        emitter = bfd.EMITTER_TEMPLATE.copy()
        return bfd.evaluate_neg_log_likelihood(
            prepared.obs_pes,
            prepared.obs_ts,
            emitter,
            x0,
            y0,
            z0,
            cx,
            cy,
            length,
            t0,
        )

    def expected_for_prepared_event(
        self,
        prepared: PreparedEvent,
        x0: float,
        y0: float,
        z0: float,
        cx: float,
        cy: float,
        length: float,
        t0: float,
    ) -> Dict[str, np.ndarray]:
        """
        Compute the model-predicted emission distances, PE, and times for one event.

        This is useful when you want to inspect the model response without
        running Minuit.
        """
        self._activate_prepared_event(prepared)

        cz2 = 1.0 - cx * cx - cy * cy
        if cz2 < 0.0:
            raise ValueError("cx and cy give an unphysical direction: 1 - cx^2 - cy^2 < 0")

        cz = np.sqrt(cz2)
        emitter = bfd.EMITTER_TEMPLATE.copy()
        emitter.start_coord = (x0, y0, z0)
        emitter.starting_time = t0
        emitter.direction = (cx, cy, cz)
        emitter.length = length

        main_idx = bfd.get_main_idx_from_length(length)
        init_ke = bfd.INIT_ENERGY_TABLE[main_idx][0]

        s = emitter.get_emission_points(prepared.p_locations, init_ke)
        exp_pes, exp_ts = emitter.get_expected_pes_ts(
            bfd.WCD,
            s,
            prepared.p_locations,
            prepared.direction_zs,
            bfd.CORR_POS,
            prepared.obs_pes,
        )

        return {
            "s": np.asarray(s, dtype=np.float64),
            "exp_pes": np.asarray(exp_pes, dtype=np.float64),
            "exp_ts": np.asarray(exp_ts, dtype=np.float64),
        }

    def truth_fcn_for_prepared_event(self, prepared: PreparedEvent) -> float:
        """
        Evaluate the FCN at the truth point stored in batch_fit_driver.TRUE_PARAMS.
        """
        tp = bfd.TRUE_PARAMS
        return self.evaluate_prepared_event(
            prepared,
            x0=tp["x0"],
            y0=tp["y0"],
            z0=tp["z0"],
            cx=tp["cx"],
            cy=tp["cy"],
            length=tp["length"],
            t0=tp["t0"],
        )

    # -------------------------------------------------------------------------
    # Minuit helpers
    # -------------------------------------------------------------------------
    def make_minuit_for_prepared_event(
        self,
        prepared: PreparedEvent,
        start_params: Optional[Dict[str, float]] = None,
    ):
        """
        Build the Minuit object for one prepared event.

        This is useful when you want to run simplex/migrad manually from a notebook.
        """
        self._activate_prepared_event(prepared)
        if start_params is None:
            start_params = dict(bfd.INIT_PARAMS)
        return bfd.make_minuit_for_event(prepared.obs_pes, prepared.obs_ts, start_params)

    def fit_prepared_event(
        self,
        prepared: PreparedEvent,
        fcn_threshold: float = 1300.0,
        max_attempts: int = 4,
        ncall: int = 6000,
        seed: Optional[int] = None,
        keep_minuit: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit one prepared event using the same retry logic as the batch driver.

        Returns a dictionary containing the best fit, the starting point used on
        each attempt, and optionally the final Minuit object for inspection.
        """
        self._activate_prepared_event(prepared)

        rng = np.random.default_rng(seed)
        best_result: Optional[Dict[str, Any]] = None
        best_fval = np.inf
        attempt_results = []

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                start_params = dict(bfd.INIT_PARAMS)
            else:
                start_params = bfd.randomize_vertex_only(bfd.INIT_PARAMS, attempt, rng)

            m = bfd.make_minuit_for_event(prepared.obs_pes, prepared.obs_ts, start_params)

            if attempt > 1:
                m.simplex(ncall=ncall)
            m.migrad(ncall=ncall)

            current_fval = float(m.fval) if np.isfinite(m.fval) else np.inf
            attempt_result = {
                "attempt": attempt,
                "start_params": start_params,
                "values": m.values.to_dict(),
                "errors": m.errors.to_dict(),
                "fval": current_fval,
                "valid": bool(m.valid),
                "minuit": m if keep_minuit else None,
            }
            attempt_results.append(attempt_result)

            if current_fval < best_fval:
                best_fval = current_fval
                best_result = {
                    "values": m.values.to_dict(),
                    "errors": m.errors.to_dict(),
                    "fval": current_fval,
                    "valid": bool(m.valid),
                    "attempts": attempt,
                    "start_params": start_params,
                    "minuit": m if keep_minuit else None,
                    "attempt_history": attempt_results,
                }

            if m.valid and np.isfinite(current_fval) and current_fval <= fcn_threshold:
                break

        return best_result

    def summarize_result(self, prepared: PreparedEvent, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a compact summary dictionary that is convenient to display in a notebook.
        """
        summary = {
            "event_index": prepared.event_index,
            "n_hits_before_cut": prepared.n_hits_before_cut,
            "n_hits_after_cut": prepared.n_hits_after_cut,
            "n_pmts_with_pe": int(np.sum(prepared.obs_pes > 0.0)),
            "n_pmts_with_time": int(np.sum(np.isfinite(prepared.obs_ts))),
            "ring_mask_enabled": self.use_ring_mask,
            "valid": result["valid"],
            "attempts": result["attempts"],
            "best_fcn": result["fval"],
            "truth_fcn": self.truth_fcn_for_prepared_event(prepared),
        }

        for key in ["x0", "y0", "z0", "cx", "cy", "length", "t0"]:
            summary[key] = result["values"][key]
            summary[f"{key}_err"] = result["errors"][key]

        return summary

    # -------------------------------------------------------------------------
    # Internal helper
    # -------------------------------------------------------------------------
    def _activate_prepared_event(self, prepared: PreparedEvent) -> None:
        """
        Copy the prepared event's PMT geometry into the backend globals.

        The likelihood backend expects these globals to point at the current
        event geometry, so any time you switch events in a notebook, this keeps
        the backend aligned with the chosen event.
        """
        self.initialize()
        bfd.P_LOCATIONS = prepared.p_locations
        bfd.DIRECTION_ZS = prepared.direction_zs
        bfd.RING_KEEP_MASK = prepared.ring_keep_mask


def prepare_and_fit_event(
    event_index: int,
    input_file: str = "/eos/user/j/jrimmer/sim_work_dir/WCSim/sim_data/mu-/1kmu-_300MeV_x0y0zn1350mm_noScat.npz",
    cut_time: float = 17.0,
    use_ring_mask: bool = True,
    seed: Optional[int] = None,
    fcn_threshold: float = 1300.0,
    max_attempts: int = 4,
    ncall: int = 6000,
) -> Dict[str, Any]:
    """
    Convenience function for notebooks when you do not want to manage the class.

    Returns a dictionary with the inspector, the prepared event, the fit result,
    and a compact summary.
    """
    inspector = SingleEventInspector(
        input_file=input_file,
        cut_time=cut_time,
        use_ring_mask=use_ring_mask,
    )
    prepared = inspector.prepare_event(event_index)
    result = inspector.fit_prepared_event(
        prepared,
        fcn_threshold=fcn_threshold,
        max_attempts=max_attempts,
        ncall=ncall,
        seed=seed,
        keep_minuit=True,
    )
    summary = inspector.summarize_result(prepared, result)
    return {
        "inspector": inspector,
        "prepared": prepared,
        "result": result,
        "summary": summary,
    }
