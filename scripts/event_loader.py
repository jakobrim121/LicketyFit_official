"""
event_loader.py

General selected-event loader for WCTE production ROOT files.

The historical function name get_mu_events is kept as a compatibility alias, but
new code should call get_selected_events(...).  The selection is configurable so
it can be used for muon-like, pion-like, kaon-like, proton-like, or custom beam
samples as long as the ROOT file contains the needed trigger/TOF quantities.
"""

from __future__ import annotations

import pickle

import awkward as ak
import numpy as np
import pandas as pd
import uproot


# ============================================================
# Defaults
# ============================================================
DEFAULT_PEAK_WINDOW = 100.0       # ns; require event peak within +/- this window
DEFAULT_PEAK_BIN_WIDTH = 50.0     # ns; histogram bin width used to find event peak
DEFAULT_PEAK_TIME_MIN = 0.0       # ns
DEFAULT_PEAK_TIME_MAX = 10000.0   # ns
DEFAULT_SAMPLE_FRACTION = 0.1    # fraction of requested entries used for peak estimate
DEFAULT_MAX_SAMPLE_EVENTS = 100    # selected events used for median/reference peak
DEFAULT_TOF_WINDOW = 0.2          # ns around selected TOF mean

# T5 timing cut defaults. This cut is based on:
#     hit_pmt_calibrated_times - first T5_hit_time
DEFAULT_T5_PEAK_WINDOW = 200.0       # ns; keep events within +/- this window
DEFAULT_T5_PEAK_TIME_MIN = -2000.0   # ns; histogram range for PMT - first T5 time
DEFAULT_T5_PEAK_TIME_MAX = 4000.0    # ns; histogram range for PMT - first T5 time

PARTICLE_TO_T5_NR = {
    # Keep muon as the historical default.  Override from the driver if your
    # production files use a different code for pion/kaon/proton selections.
    "muon": 1,
    "mu": 1,
    "mu-": 1,
    "mu+": 1,
    "pion": 1,
    "pi": 1,
    "pi+": 1,
    "pi-": 1,
    "kaon": 1,
    "k": 1,
    "k+": 1,
    "k-": 1,
    "proton": 1,
    "p": 1,
    "p+": 1,
}


def _canonical_particle_name(particle: str | None) -> str:
    if particle is None:
        return "primary"
    key = str(particle).strip().lower()
    aliases = {
        "mu": "muon", "mu-": "muon", "mu+": "muon", "muon": "muon",
        "pi": "pion", "pi-": "pion", "pi+": "pion", "pion": "pion",
        "k": "kaon", "k-": "kaon", "k+": "kaon", "kaon": "kaon",
        "p": "proton", "p+": "proton", "proton": "proton",
    }
    return aliases.get(key, key)


def _default_tof_scalar_field(particle: str | None) -> str:
    pname = _canonical_particle_name(particle)
    if pname == "muon":
        return "tof_mean_muon"
    return f"tof_mean_{pname}"


def _default_momentum_scalar_field(particle: str | None) -> str:
    pname = _canonical_particle_name(particle)
    if pname == "muon":
        return "momentum_after_beam_window_mean_muon"
    return f"momentum_after_beam_window_mean_{pname}"


# ============================================================
# Selection mask
# ============================================================
def _make_beam_event_mask(
    arr,
    eveto_cut,
    tagger_cut,
    tof_primary,
    *,
    tof_window=DEFAULT_TOF_WINDOW,
    t5_particle_nr=1,
    particle = "muon",
):
    """Return the standard beam/quality/TOF event mask."""
    if particle == "muon":
        return (
            (arr["vme_act_eveto"] < eveto_cut)
            & (arr["vme_act_tagger"] > tagger_cut)
            & (arr["T5_HasMultipleScintillatorsHit"] == False)
            & (arr["T5_HasOutOfTimeWindow"] == False)
            & (arr["vme_evt_quality_bitmask"] == 0)
            & (arr["T5_HasValidHit"] == True)
            & (arr["T5_particle_nr"] == int(t5_particle_nr))
            & (arr["window_data_quality_mask"] == 0)
            & (arr["vme_digi_issues_bitmask"] == 0)
            & (arr["T5_HasInTimeWindow"] == True)
            & (arr["vme_tof_corr"] > float(tof_primary) - float(tof_window))
            & (arr["vme_tof_corr"] < float(tof_primary) + float(tof_window))
        )
    #elif particle=="proton":
    else:
        
        return (
            (arr["vme_act_eveto"] < eveto_cut)
            & (arr["vme_act_tagger"] < tagger_cut)
            & (arr["T5_HasMultipleScintillatorsHit"] == False)
            & (arr["T5_HasOutOfTimeWindow"] == False)
            & (arr["vme_evt_quality_bitmask"] == 0)
            & (arr["T5_HasValidHit"] == True)
            & (arr["T5_particle_nr"] == int(t5_particle_nr))
            & (arr["window_data_quality_mask"] == 0)
            & (arr["vme_digi_issues_bitmask"] == 0)
            & (arr["T5_HasInTimeWindow"] == True)
            & (arr["vme_tof_corr"] > float(tof_primary) - float(tof_window))
            & (arr["vme_tof_corr"] < float(tof_primary) + float(tof_window))
        )


# ============================================================
# Timing peak helpers
# ============================================================
def _as_1d_float_array(x):
    """
    Convert scalar/list/awkward/list-of-lists into a flat finite float array.
    """
    try:
        arr = ak.to_numpy(ak.flatten(x, axis=None))
    except Exception:
        arr = np.asarray(x)

    arr = np.asarray(arr, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _event_peak_time(
    times,
    bin_width=DEFAULT_PEAK_BIN_WIDTH,
    time_min=DEFAULT_PEAK_TIME_MIN,
    time_max=DEFAULT_PEAK_TIME_MAX,
):
    """Return the center of the most-populated timing histogram bin."""
    times = _as_1d_float_array(times)
    if len(times) == 0:
        return np.nan

    times = times[(times >= time_min) & (times < time_max)]
    if len(times) == 0:
        return np.nan

    bins = np.arange(time_min, time_max + bin_width, bin_width)
    counts, edges = np.histogram(times, bins=bins)
    if len(counts) == 0 or np.max(counts) == 0:
        return np.nan

    max_bin = int(np.argmax(counts))
    return float(0.5 * (edges[max_bin] + edges[max_bin + 1]))


def _make_peak_time_mask(
    arr,
    median_peak_time,
    peak_window=DEFAULT_PEAK_WINDOW,
    peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
    peak_time_min=DEFAULT_PEAK_TIME_MIN,
    peak_time_max=DEFAULT_PEAK_TIME_MAX,
):
    """Original PMT calibrated-time peak cut."""
    times_list = ak.to_list(arr["hit_pmt_calibrated_times"])

    keep = []
    for times in times_list:
        peak = _event_peak_time(
            times,
            bin_width=peak_bin_width,
            time_min=peak_time_min,
            time_max=peak_time_max,
        )
        keep.append(bool(np.isfinite(peak) and abs(peak - median_peak_time) <= peak_window))

    return np.asarray(keep, dtype=bool)


def _event_pmt_minus_first_t5_times(pmt_times, t5_times):
    """
    For one event, compute:

        hit_pmt_calibrated_times - first T5_hit_time

    If the event has N PMT hits and M T5 hits, this returns N values,
    not N*M pairwise combinations.
    """
    pmt_times = _as_1d_float_array(pmt_times)
    t5_times = _as_1d_float_array(t5_times)

    if pmt_times.size == 0 or t5_times.size == 0:
        return np.asarray([], dtype=np.float64)

    first_t5_time = float(t5_times[0])
    return pmt_times - first_t5_time


def _make_t5_peak_time_mask(
    arr,
    reference_t5_delta_peak_time,
    peak_window=DEFAULT_T5_PEAK_WINDOW,
    peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
    peak_time_min=DEFAULT_T5_PEAK_TIME_MIN,
    peak_time_max=DEFAULT_T5_PEAK_TIME_MAX,
):
    """
    Keep events whose event-level peak of:

        hit_pmt_calibrated_times - first T5_hit_time

    is within +/- peak_window of reference_t5_delta_peak_time.
    """
    pmt_times_list = ak.to_list(arr["hit_pmt_calibrated_times"])
    t5_times_list = ak.to_list(arr["T5_hit_time"])

    keep = []
    for pmt_times, t5_times in zip(pmt_times_list, t5_times_list):
        deltas = _event_pmt_minus_first_t5_times(pmt_times, t5_times)

        peak = _event_peak_time(
            deltas,
            bin_width=peak_bin_width,
            time_min=peak_time_min,
            time_max=peak_time_max,
        )

        keep.append(
            bool(
                np.isfinite(peak)
                and np.isfinite(reference_t5_delta_peak_time)
                and abs(peak - reference_t5_delta_peak_time) <= peak_window
            )
        )

    return np.asarray(keep, dtype=bool)


# ============================================================
# ROOT iteration / timing calibration helpers
# ============================================================
def _iter_chunks(fname, tree_name, branches, n_entries_to_process, step_size):
    """
    Iterate over chunks of the ROOT tree, guaranteed to stop at
    n_entries_to_process total entries regardless of uproot's entry_stop
    behaviour.  Each yielded chunk is an awkward array of exactly
    min(step_size, remaining) entries.
    """
    entries_yielded = 0
    for arr in uproot.iterate(
        f"{fname}:{tree_name}",
        branches,
        library="ak",
        step_size=int(step_size),
    ):
        if entries_yielded >= n_entries_to_process:
            break

        remaining = n_entries_to_process - entries_yielded
        if len(arr) > remaining:
            arr = arr[:remaining]

        yield arr
        entries_yielded += len(arr)

        if entries_yielded >= n_entries_to_process:
            break


def _estimate_median_peak_time(
    fname,
    tree_name,
    branches,
    eveto_cut,
    tagger_cut,
    tof_primary,
    *,
    n=None,
    step_size=10000,
    sample_fraction=DEFAULT_SAMPLE_FRACTION,
    max_sample_events=DEFAULT_MAX_SAMPLE_EVENTS,
    peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
    peak_time_min=DEFAULT_PEAK_TIME_MIN,
    peak_time_max=DEFAULT_PEAK_TIME_MAX,
    tof_window=DEFAULT_TOF_WINDOW,
    t5_particle_nr=1,
    verbose=True,
):
    with uproot.open(fname) as f:
        tree = f[tree_name]
        total_entries = tree.num_entries

    n_to_process = total_entries if n is None else min(int(n), total_entries)
    if n_to_process <= 0:
        raise ValueError("n_to_process is zero. Check n and the input ROOT file.")

    n_entries_to_scan = int(np.ceil(sample_fraction * n_to_process))
    n_entries_to_scan = max(1, min(n_entries_to_scan, n_to_process))

    peak_times = []
    for arr in _iter_chunks(fname, tree_name, branches, n_entries_to_scan, step_size):
        sample_mask = _make_beam_event_mask(
            arr,
            eveto_cut,
            tagger_cut,
            tof_primary,
            tof_window=tof_window,
            t5_particle_nr=t5_particle_nr,
        )
        arr_sample = arr[sample_mask]
        times_list = ak.to_list(arr_sample["hit_pmt_calibrated_times"])

        for times in times_list:
            peak = _event_peak_time(
                times,
                bin_width=peak_bin_width,
                time_min=peak_time_min,
                time_max=peak_time_max,
            )
            if np.isfinite(peak):
                peak_times.append(float(peak))
            if len(peak_times) >= max_sample_events:
                break
        if len(peak_times) >= max_sample_events:
            break

    if len(peak_times) == 0:
        raise RuntimeError(
            "Could not estimate median peak time. No valid selected events were "
            "found in the initial sample. Try increasing sample_fraction or "
            "supplying tof_primary explicitly."
        )

    median_peak_time = float(np.median(peak_times))
    if verbose:
        print("")
        print("Peak-time calibration")
        print("---------------------")
        print(f"ROOT entries requested:        {n_to_process}")
        print(f"ROOT entries scanned:          {n_entries_to_scan}")
        print(f"Selected events used:          {len(peak_times)}")
        print(f"Estimated median peak time:    {median_peak_time:.2f} ns")
        print("")
    return median_peak_time


def _estimate_t5_delta_peak_time(
    fname,
    tree_name,
    branches,
    eveto_cut,
    tagger_cut,
    tof_primary,
    *,
    n=None,
    step_size=10000,
    sample_fraction=DEFAULT_SAMPLE_FRACTION,
    max_sample_events=DEFAULT_MAX_SAMPLE_EVENTS,
    peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
    peak_time_min=DEFAULT_T5_PEAK_TIME_MIN,
    peak_time_max=DEFAULT_T5_PEAK_TIME_MAX,
    tof_window=DEFAULT_TOF_WINDOW,
    t5_particle_nr=1,
    verbose=True,
    particle = "muon",
):
    """
    Estimate the global reference peak of:

        hit_pmt_calibrated_times - first T5_hit_time

    using a selected-event sample.
    """
    with uproot.open(fname) as f:
        tree = f[tree_name]
        total_entries = tree.num_entries

    n_to_process = total_entries if n is None else min(int(n), total_entries)
    if n_to_process <= 0:
        raise ValueError("n_to_process is zero. Check n and the input ROOT file.")

    n_entries_to_scan = int(np.ceil(sample_fraction * n_to_process))
    n_entries_to_scan = max(1, min(n_entries_to_scan, n_to_process))

    all_deltas = []
    n_events_used = 0

    for arr in _iter_chunks(fname, tree_name, branches, n_entries_to_scan, step_size):
        sample_mask = _make_beam_event_mask(
            arr,
            eveto_cut,
            tagger_cut,
            tof_primary,
            tof_window=tof_window,
            t5_particle_nr=t5_particle_nr,
            particle=particle,
        )
        arr_sample = arr[sample_mask]

        pmt_times_list = ak.to_list(arr_sample["hit_pmt_calibrated_times"])
        t5_times_list = ak.to_list(arr_sample["T5_hit_time"])

        for pmt_times, t5_times in zip(pmt_times_list, t5_times_list):
            deltas = _event_pmt_minus_first_t5_times(pmt_times, t5_times)
            if deltas.size == 0:
                continue

            all_deltas.append(deltas)
            n_events_used += 1

            if n_events_used >= max_sample_events:
                break

        if n_events_used >= max_sample_events:
            break

    if len(all_deltas) == 0:
        raise RuntimeError(
            "Could not estimate T5 delta peak time. No valid selected events with "
            "both PMT hit times and T5_hit_time were found in the initial sample. "
            "Try increasing sample_fraction or checking that T5_hit_time exists."
        )

    all_deltas = np.concatenate(all_deltas)
    reference_peak = _event_peak_time(
        all_deltas,
        bin_width=peak_bin_width,
        time_min=peak_time_min,
        time_max=peak_time_max,
    )

    if not np.isfinite(reference_peak):
        raise RuntimeError(
            "Could not estimate T5 delta peak time. The PMT-minus-first-T5 "
            "distribution had no valid histogram peak in the configured range."
        )

    if verbose:
        print("")
        print("T5 delta-time calibration")
        print("-------------------------")
        print(f"ROOT entries requested:        {n_to_process}")
        print(f"ROOT entries scanned:          {n_entries_to_scan}")
        print(f"Selected events used:          {n_events_used}")
        print(f"Estimated PMT - first T5 peak: {reference_peak:.2f} ns")
        print("")

    return float(reference_peak)


# ============================================================
# Event conversion
# ============================================================
def _mask_to_numpy(mask):
    if mask is None:
        return None
    try:
        return ak.to_numpy(mask).astype(bool)
    except Exception:
        return np.asarray(mask, dtype=bool)


def _arr_to_events(arr, event_mask=None, entry_start=0):
    event_numbers_all = np.arange(entry_start, entry_start + len(arr), dtype=np.int64)

    if event_mask is None:
        arr_sel = arr
        event_numbers = event_numbers_all
    else:
        mask_np = _mask_to_numpy(event_mask)
        if len(mask_np) != len(arr):
            raise ValueError(f"Mask length {len(mask_np)} does not match array length {len(arr)}")
        arr_sel = arr[mask_np]
        event_numbers = event_numbers_all[mask_np]

    slots = ak.to_list(arr_sel["hit_mpmt_slot_ids"])
    pos = ak.to_list(arr_sel["hit_pmt_position_ids"])
    charge = ak.to_list(arr_sel["hit_pmt_charges"])
    time = ak.to_list(arr_sel["hit_pmt_calibrated_times"])

    events = []
    for ev_num, s, p, q, t in zip(event_numbers, slots, pos, charge, time):
        pmt_ids = np.asarray(s, dtype=np.int64) * 100 + np.asarray(p, dtype=np.int64)
        charges = np.asarray(q, dtype=np.float64)
        times = np.asarray(t, dtype=np.float64)
        ev_nums = np.full(len(pmt_ids), ev_num, dtype=np.int64)
        events.append(np.column_stack((pmt_ids, charges, times, ev_nums)))
    return events


def _read_scalar_results(fname, *, fields=None, entry_stop=1):
    """
    Read only the scalar-result fields that are actually needed.
    """
    with uproot.open(fname) as f:
        tree = f["vme_analysis_scalar_results"]
        if fields is None:
            fields_to_read = list(tree.keys())
        else:
            available = set(tree.keys())
            fields_to_read = []
            for field in fields:
                if field and field in available and field not in fields_to_read:
                    fields_to_read.append(field)

        if not fields_to_read:
            raise KeyError(
                "None of the requested scalar fields were found in "
                "vme_analysis_scalar_results. Requested fields: "
                f"{fields!r}"
            )

        arr_s = tree.arrays(fields_to_read, library="ak", entry_stop=entry_stop)

    data_s = {}
    for key in arr_s.fields:
        try:
            if isinstance(arr_s[key].type.content, ak.types.NumpyType) or "var *" in str(arr_s[key].type):
                data_s[key] = ak.to_list(arr_s[key])
            else:
                data_s[key] = np.asarray(arr_s[key])
        except Exception:
            data_s[key] = ak.to_list(arr_s[key])
    return pd.DataFrame(data_s)


def _field_or_value(df_scalar, field_name, explicit_value, description):
    if explicit_value is not None:
        return float(explicit_value)
    if field_name and field_name in df_scalar.columns:
        return float(df_scalar[field_name].iloc[0])
    raise KeyError(
        f"Could not determine {description}. Field {field_name!r} was not found "
        "and no explicit value was supplied."
    )


# ============================================================
# Public loader
# ============================================================
def get_selected_events(
    run,
    n,
    *,
    max_selected_events=None,
    particle="muon",
    root_file=None,
    step_size=1000,
    out_pkl=None,

    # Apply PMT-time minus first-T5-time peak cut only when requested.
    use_t5_hit_time=True,

    use_peak_time_cut=False,
    sample_fraction=DEFAULT_SAMPLE_FRACTION,
    max_sample_events=DEFAULT_MAX_SAMPLE_EVENTS,

    # Original PMT calibrated-time peak cut settings.
    peak_window=DEFAULT_PEAK_WINDOW,
    peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
    peak_time_min=DEFAULT_PEAK_TIME_MIN,
    peak_time_max=DEFAULT_PEAK_TIME_MAX,

    # T5 delta-time peak cut settings.
    t5_peak_window=DEFAULT_T5_PEAK_WINDOW,
    t5_peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
    t5_peak_time_min=DEFAULT_T5_PEAK_TIME_MIN,
    t5_peak_time_max=DEFAULT_T5_PEAK_TIME_MAX,

    eveto_cut=None,
    tagger_cut=None,
    tof_primary=None,
    tof_window=DEFAULT_TOF_WINDOW,
    t5_particle_nr=None,
    eveto_scalar_field="act_eveto_cut",
    tagger_scalar_field="act_tagger_cut",
    tof_scalar_field=None,
    momentum_scalar_field=None,
    verbose=True,
):
    """
    Load selected WCTE events from a production ROOT file.

    Parameters
    ----------
    n:
        Maximum number of WCTEReadoutWindows ROOT entries to inspect.
        This is an entry/read limit, not a selected-event return limit.
    max_selected_events:
        Optional hard cap on the number of selected events returned/written.
        Leave as None to return every selected event found in the first n ROOT
        entries.
    use_peak_time_cut:
        If True, apply the original PMT calibrated-time peak cut.
    use_t5_hit_time:
        If True, apply a timing cut based on the event-level peak of:

            hit_pmt_calibrated_times - first T5_hit_time

        Events are kept only if their PMT-minus-first-T5 peak is within
        +/- t5_peak_window of the sampled global PMT-minus-first-T5 peak.

    Returns a list of arrays.  Each array has columns:
        0: WCTE PMT ID = slot*100 + pmt_position
        1: charge
        2: calibrated time [ns]
        3: ROOT entry index

    The selection defaults reproduce the historical muon selection.  For other
    beam particles, set particle plus any needed TOF/T5 overrides in the driver:
        tof_primary=..., tof_scalar_field=..., t5_particle_nr=...
    """
    particle_label = _canonical_particle_name(particle)
    if root_file is None:
        fname = (
            f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/"
            f"production_v1_0/{run}/WCTE_merged_production_R{run}.root"
        )
    else:
        fname = str(root_file)

    tree_name = "WCTEReadoutWindows"
    branches = [
        "vme_act0_l_charge", "vme_act0_r_charge",
        "vme_act1_l_charge", "vme_act1_r_charge",
        "vme_act2_l_charge", "vme_act2_r_charge",
        "vme_act3_l_charge", "vme_act3_r_charge",
        "vme_act4_l_charge", "vme_act4_r_charge",
        "T5_HasMultipleScintillatorsHit",
        "T5_HasOutOfTimeWindow",
        "vme_evt_quality_bitmask",
        "T5_HasValidHit",
        "T5_particle_nr",
        "window_data_quality_mask",
        "vme_digi_issues_bitmask",
        "T5_HasInTimeWindow",
        "T5_hit_is_in_bounds",
        "hit_mpmt_slot_ids",
        "hit_pmt_position_ids",
        "hit_pmt_charges",
        "hit_pmt_calibrated_times",
        "vme_tof_t0t1",
        "vme_tof_corr",
        "vme_act_eveto",
        "vme_act_tagger",
    ]

    if use_t5_hit_time:
        branches.append("T5_hit_time")

    tof_scalar_field = tof_scalar_field or _default_tof_scalar_field(particle_label)
    momentum_scalar_field = momentum_scalar_field or _default_momentum_scalar_field(particle_label)
    if t5_particle_nr is None:
        t5_particle_nr = PARTICLE_TO_T5_NR.get(str(particle).strip().lower(), 1)

    with uproot.open(fname) as f:
        total_entries = int(f[tree_name].num_entries)

    if n is None:
        n_entries_to_process = total_entries
    else:
        n_entries_to_process = min(int(n), total_entries)

    print("PROCESSING ENTRIES:", n_entries_to_process)

    if n_entries_to_process < 0:
        raise ValueError("n must be None or a non-negative integer.")

    if max_selected_events is not None:
        max_selected_events = int(max_selected_events)
        if max_selected_events < 0:
            raise ValueError("max_selected_events must be None or a non-negative integer.")

    scalar_fields = [
        eveto_scalar_field,
        tagger_scalar_field,
        tof_scalar_field,
        momentum_scalar_field,
    ]
    df_scalar = _read_scalar_results(fname, fields=scalar_fields, entry_stop=1)
    eveto_cut_value = _field_or_value(df_scalar, eveto_scalar_field, eveto_cut, "eveto cut")
    tagger_cut_value = _field_or_value(df_scalar, tagger_scalar_field, tagger_cut, "tagger cut")
    tof_primary_value = _field_or_value(df_scalar, tof_scalar_field, tof_primary, "TOF mean")

    p_after_window = np.nan
    if momentum_scalar_field in df_scalar.columns:
        try:
            p_after_window = float(df_scalar[momentum_scalar_field].iloc[0])
        except Exception:
            p_after_window = np.nan

    if verbose:
        print("Selected-event loader")
        print("---------------------")
        print(f"Particle label:              {particle_label}")
        print(f"ROOT file:                   {fname}")
        print(f"ROOT entries in file:         {total_entries}")
        print(f"ROOT entries requested:       {n}")
        print(f"ROOT entries to process:      {n_entries_to_process}")
        if max_selected_events is not None:
            print(f"Max selected events:          {max_selected_events}")
        print(f"T5 particle nr:              {int(t5_particle_nr)}")
        print(f"TOF mean [ns]:               {tof_primary_value:.4f}")
        print(f"TOF window [ns]:             +/- {tof_window}")
        print(f"Use PMT peak-time cut:        {bool(use_peak_time_cut)}")
        print(f"Use T5 delta-time cut:        {bool(use_t5_hit_time)}")
        if use_t5_hit_time:
            print(f"T5 delta window [ns]:         +/- {t5_peak_window}")
        if np.isfinite(p_after_window):
            print(f"Beam momentum after window:  {p_after_window}")
        print("")

    median_peak_time = None
    if use_peak_time_cut:
        median_peak_time = _estimate_median_peak_time(
            fname=fname,
            tree_name=tree_name,
            branches=branches,
            eveto_cut=eveto_cut_value,
            tagger_cut=tagger_cut_value,
            tof_primary=tof_primary_value,
            n=n_entries_to_process,
            step_size=step_size,
            sample_fraction=sample_fraction,
            max_sample_events=max_sample_events,
            peak_bin_width=peak_bin_width,
            peak_time_min=peak_time_min,
            peak_time_max=peak_time_max,
            tof_window=tof_window,
            t5_particle_nr=t5_particle_nr,
            verbose=verbose,
        )

    reference_t5_delta_peak_time = None
    if use_t5_hit_time:
        reference_t5_delta_peak_time = _estimate_t5_delta_peak_time(
            fname=fname,
            tree_name=tree_name,
            branches=branches,
            eveto_cut=eveto_cut_value,
            tagger_cut=tagger_cut_value,
            tof_primary=tof_primary_value,
            n=n_entries_to_process,
            step_size=step_size,
            sample_fraction=sample_fraction,
            max_sample_events=max_sample_events,
            peak_bin_width=t5_peak_bin_width,
            peak_time_min=t5_peak_time_min,
            peak_time_max=t5_peak_time_max,
            tof_window=tof_window,
            t5_particle_nr=t5_particle_nr,
            verbose=verbose,
            particle=particle,
        )

    if n_entries_to_process == 0 or max_selected_events == 0:
        if verbose:
            print("Final selection")
            print("---------------")
            print("Selected events:               0")
            print("")
        return [] if out_pkl is None else None

    events = [] if out_pkl is None else None
    fout = open(out_pkl, "wb") if out_pkl is not None else None
    n_before_timing_cuts = 0
    n_after_timing_cuts = 0
    n_selected_written = 0
    entry_start = 0

    try:
        for arr in _iter_chunks(fname, tree_name, branches, n_entries_to_process, step_size):
            this_entry_start = entry_start
            entry_start += len(arr)

            event_mask = _make_beam_event_mask(
                arr,
                eveto_cut_value,
                tagger_cut_value,
                tof_primary_value,
                tof_window=tof_window,
                t5_particle_nr=t5_particle_nr,
                particle=particle,
            )
            event_mask_np = _mask_to_numpy(event_mask)
            n_before_timing_cuts += int(np.sum(event_mask_np))

            if np.sum(event_mask_np) == 0:
                continue

            final_mask = event_mask_np.copy()

            if use_peak_time_cut:
                peak_mask_np = _make_peak_time_mask(
                    arr,
                    median_peak_time=median_peak_time,
                    peak_window=peak_window,
                    peak_bin_width=peak_bin_width,
                    peak_time_min=peak_time_min,
                    peak_time_max=peak_time_max,
                )
                final_mask = final_mask & peak_mask_np

            if use_t5_hit_time:
                t5_peak_mask_np = _make_t5_peak_time_mask(
                    arr,
                    reference_t5_delta_peak_time=reference_t5_delta_peak_time,
                    peak_window=t5_peak_window,
                    peak_bin_width=t5_peak_bin_width,
                    peak_time_min=t5_peak_time_min,
                    peak_time_max=t5_peak_time_max,
                )
                final_mask = final_mask & t5_peak_mask_np

            n_after_timing_cuts += int(np.sum(final_mask))
            if np.sum(final_mask) == 0:
                continue

            chunk_events = _arr_to_events(arr, event_mask=final_mask, entry_start=this_entry_start)

            if max_selected_events is not None:
                remaining = max_selected_events - n_selected_written
                if remaining <= 0:
                    break
                chunk_events = chunk_events[:remaining]

            if out_pkl is None:
                events.extend(chunk_events)
            else:
                for event in chunk_events:
                    pickle.dump(event, fout)

            n_selected_written += len(chunk_events)
            if max_selected_events is not None and n_selected_written >= max_selected_events:
                break
    finally:
        if fout is not None:
            fout.close()

    if verbose:
        print("Final selection")
        print("---------------")
        if use_peak_time_cut or use_t5_hit_time:
            print(f"Selected before timing cuts:   {n_before_timing_cuts}")
            print(f"Selected after timing cuts:    {n_after_timing_cuts}")
        else:
            print(f"Selected events:               {n_after_timing_cuts}")
        if np.isfinite(p_after_window):
            print(f"Beam momentum after window:    {p_after_window}")
        print("")

    return events


# Particle-generic alias used by the batch drivers.
def get_particle_events(run, n=None, *, particle="muon", **kwargs):
    return get_selected_events(run, n, particle=particle, **kwargs)


# Backward-compatible alias for older notebooks/scripts. Prefer get_particle_events()
# or get_selected_events().  Passing particle="proton"/"pion"/etc. makes this
# generic despite the historical name.
def get_mu_events(run, n=None, **kwargs):
    kwargs.setdefault("particle", "muon")
    return get_selected_events(run, n, **kwargs)











# """
# event_loader.py

# General selected-event loader for WCTE production ROOT files.

# The historical function name get_mu_events is kept as a compatibility alias, but
# new code should call get_selected_events(...).  The selection is configurable so
# it can be used for muon-like, pion-like, kaon-like, proton-like, or custom beam
# samples as long as the ROOT file contains the needed trigger/TOF quantities.
# """

# from __future__ import annotations

# import pickle
# from pathlib import Path

# import awkward as ak
# import numpy as np
# import pandas as pd
# import uproot


# # ============================================================
# # Defaults
# # ============================================================
# DEFAULT_PEAK_WINDOW = 100.0       # ns; require event peak within +/- this window
# DEFAULT_PEAK_BIN_WIDTH = 50.0     # ns; histogram bin width used to find event peak
# DEFAULT_PEAK_TIME_MIN = 0.0       # ns
# DEFAULT_PEAK_TIME_MAX = 10000.0   # ns
# DEFAULT_SAMPLE_FRACTION = 0.05    # fraction of requested entries used for peak estimate
# DEFAULT_MAX_SAMPLE_EVENTS = 50    # selected events used for median peak
# DEFAULT_TOF_WINDOW = 0.2          # ns around selected TOF mean

# PARTICLE_TO_T5_NR = {
#     # Keep muon as the historical default.  Override from the driver if your
#     # production files use a different code for pion/kaon/proton selections.
#     "muon": 1,
#     "mu": 1,
#     "mu-": 1,
#     "mu+": 1,
#     "pion": 1,
#     "pi": 1,
#     "pi+": 1,
#     "pi-": 1,
#     "kaon": 1,
#     "k": 1,
#     "k+": 1,
#     "k-": 1,
#     "proton": 1,
#     "p": 1,
#     "p+": 1,
# }


# def _canonical_particle_name(particle: str | None) -> str:
#     if particle is None:
#         return "primary"
#     key = str(particle).strip().lower()
#     aliases = {
#         "mu": "muon", "mu-": "muon", "mu+": "muon", "muon": "muon",
#         "pi": "pion", "pi-": "pion", "pi+": "pion", "pion": "pion",
#         "k": "kaon", "k-": "kaon", "k+": "kaon", "kaon": "kaon",
#         "p": "proton", "p+": "proton", "proton": "proton",
#     }
#     return aliases.get(key, key)


# def _default_tof_scalar_field(particle: str | None) -> str:
#     pname = _canonical_particle_name(particle)
#     if pname == "muon":
#         return "tof_mean_muon"
#     return f"tof_mean_{pname}"


# def _default_momentum_scalar_field(particle: str | None) -> str:
#     pname = _canonical_particle_name(particle)
#     if pname == "muon":
#         return "momentum_after_beam_window_mean_muon"
#     return f"momentum_after_beam_window_mean_{pname}"


# # ============================================================
# # Selection mask
# # ============================================================
# def _make_beam_event_mask(
#     arr,
#     eveto_cut,
#     tagger_cut,
#     tof_primary,
#     *,
#     tof_window=DEFAULT_TOF_WINDOW,
#     t5_particle_nr=1,
# ):
#     """Return the standard beam/quality/TOF event mask."""
#     return (
#         (arr["vme_act_eveto"] < eveto_cut)
#         & (arr["vme_act_tagger"] > tagger_cut)
#         & (arr["T5_HasMultipleScintillatorsHit"] == False)
#         & (arr["T5_HasOutOfTimeWindow"] == False)
#         & (arr["vme_evt_quality_bitmask"] == 0)
#         & (arr["T5_HasValidHit"] == True)
#         & (arr["T5_particle_nr"] == int(t5_particle_nr))
#         & (arr["window_data_quality_mask"] == 0)
#         & (arr["vme_digi_issues_bitmask"] == 0)
#         & (arr["T5_HasInTimeWindow"] == True)
#         & (arr["vme_tof_corr"] > float(tof_primary) - float(tof_window))
#         & (arr["vme_tof_corr"] < float(tof_primary) + float(tof_window))
#     )


# # ============================================================
# # Timing peak helpers
# # ============================================================
# def _event_peak_time(
#     times,
#     bin_width=DEFAULT_PEAK_BIN_WIDTH,
#     time_min=DEFAULT_PEAK_TIME_MIN,
#     time_max=DEFAULT_PEAK_TIME_MAX,
# ):
#     times = np.asarray(times, dtype=float)
#     times = times[np.isfinite(times)]
#     if len(times) == 0:
#         return np.nan

#     times = times[(times >= time_min) & (times < time_max)]
#     if len(times) == 0:
#         return np.nan

#     bins = np.arange(time_min, time_max + bin_width, bin_width)
#     counts, edges = np.histogram(times, bins=bins)
#     if len(counts) == 0 or np.max(counts) == 0:
#         return np.nan

#     max_bin = int(np.argmax(counts))
#     return float(0.5 * (edges[max_bin] + edges[max_bin + 1]))


# def _make_peak_time_mask(
#     arr,
#     median_peak_time,
#     peak_window=DEFAULT_PEAK_WINDOW,
#     peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
#     peak_time_min=DEFAULT_PEAK_TIME_MIN,
#     peak_time_max=DEFAULT_PEAK_TIME_MAX,
# ):
#     times_list = ak.to_list(arr["hit_pmt_calibrated_times"])
#     keep = []
#     for times in times_list:
#         peak = _event_peak_time(
#             times,
#             bin_width=peak_bin_width,
#             time_min=peak_time_min,
#             time_max=peak_time_max,
#         )
#         keep.append(bool(np.isfinite(peak) and abs(peak - median_peak_time) <= peak_window))
#     return np.asarray(keep, dtype=bool)

# def _make_t5_peak_time_mask(
#     arr,
#     peak_window=DEFAULT_PEAK_WINDOW,
#     peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
#     peak_time_min=DEFAULT_PEAK_TIME_MIN,
#     peak_time_max=DEFAULT_PEAK_TIME_MAX,
# ):
#     times_list = ak.to_list(arr["hit_pmt_calibrated_times"])
#     t5_hit_times = ak.to_list(arr["'T5_hit_time'"])
#     keep = []
#     for times in times_list:
#         peak = _event_peak_time(
#             times,
#             bin_width=peak_bin_width,
#             time_min=peak_time_min,
#             time_max=peak_time_max,
#         )
#         keep.append(bool(np.isfinite(peak) and abs(peak - median_peak_time) <= peak_window))
#     return np.asarray(keep, dtype=bool)


# def _iter_chunks(fname, tree_name, branches, n_entries_to_process, step_size):
#     """
#     Iterate over chunks of the ROOT tree, guaranteed to stop at
#     n_entries_to_process total entries regardless of uproot's entry_stop
#     behaviour.  Each yielded chunk is an awkward array of exactly
#     min(step_size, remaining) entries.
#     """
#     entries_yielded = 0
#     for arr in uproot.iterate(
#         f"{fname}:{tree_name}",
#         branches,
#         library="ak",
#         step_size=int(step_size),
#     ):
#         if entries_yielded >= n_entries_to_process:
#             break

#         remaining = n_entries_to_process - entries_yielded
#         if len(arr) > remaining:
#             arr = arr[:remaining]

#         yield arr
#         entries_yielded += len(arr)

#         if entries_yielded >= n_entries_to_process:
#             break


# def _estimate_median_peak_time(
#     fname,
#     tree_name,
#     branches,
#     eveto_cut,
#     tagger_cut,
#     tof_primary,
#     *,
#     n=None,
#     step_size=10000,
#     sample_fraction=DEFAULT_SAMPLE_FRACTION,
#     max_sample_events=DEFAULT_MAX_SAMPLE_EVENTS,
#     peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
#     peak_time_min=DEFAULT_PEAK_TIME_MIN,
#     peak_time_max=DEFAULT_PEAK_TIME_MAX,
#     tof_window=DEFAULT_TOF_WINDOW,
#     t5_particle_nr=1,
#     verbose=True,
# ):
#     with uproot.open(fname) as f:
#         tree = f[tree_name]
#         total_entries = tree.num_entries

#     n_to_process = total_entries if n is None else min(int(n), total_entries)
#     if n_to_process <= 0:
#         raise ValueError("n_to_process is zero. Check n and the input ROOT file.")

#     n_entries_to_scan = int(np.ceil(sample_fraction * n_to_process))
#     n_entries_to_scan = max(1, min(n_entries_to_scan, n_to_process))

#     peak_times = []
#     for arr in _iter_chunks(fname, tree_name, branches, n_entries_to_scan, step_size):
#         sample_mask = _make_beam_event_mask(
#             arr,
#             eveto_cut,
#             tagger_cut,
#             tof_primary,
#             tof_window=tof_window,
#             t5_particle_nr=t5_particle_nr,
#         )
#         arr_sample = arr[sample_mask]
#         times_list = ak.to_list(arr_sample["hit_pmt_calibrated_times"])

#         for times in times_list:
#             peak = _event_peak_time(
#                 times,
#                 bin_width=peak_bin_width,
#                 time_min=peak_time_min,
#                 time_max=peak_time_max,
#             )
#             if np.isfinite(peak):
#                 peak_times.append(float(peak))
#             if len(peak_times) >= max_sample_events:
#                 break
#         if len(peak_times) >= max_sample_events:
#             break

#     if len(peak_times) == 0:
#         raise RuntimeError(
#             "Could not estimate median peak time. No valid selected events were "
#             "found in the initial sample. Try increasing sample_fraction or "
#             "supplying tof_primary explicitly."
#         )

#     median_peak_time = float(np.median(peak_times))
#     if verbose:
#         print("")
#         print("Peak-time calibration")
#         print("---------------------")
#         print(f"ROOT entries requested:        {n_to_process}")
#         print(f"ROOT entries scanned:          {n_entries_to_scan}")
#         print(f"Selected events used:          {len(peak_times)}")
#         print(f"Estimated median peak time:    {median_peak_time:.2f} ns")
#         print("")
#     return median_peak_time


# # ============================================================
# # Event conversion
# # ============================================================
# def _mask_to_numpy(mask):
#     if mask is None:
#         return None
#     try:
#         return ak.to_numpy(mask).astype(bool)
#     except Exception:
#         return np.asarray(mask, dtype=bool)


# def _arr_to_events(arr, event_mask=None, entry_start=0):
#     event_numbers_all = np.arange(entry_start, entry_start + len(arr), dtype=np.int64)

#     if event_mask is None:
#         arr_sel = arr
#         event_numbers = event_numbers_all
#     else:
#         mask_np = _mask_to_numpy(event_mask)
#         if len(mask_np) != len(arr):
#             raise ValueError(f"Mask length {len(mask_np)} does not match array length {len(arr)}")
#         arr_sel = arr[mask_np]
#         event_numbers = event_numbers_all[mask_np]

#     slots = ak.to_list(arr_sel["hit_mpmt_slot_ids"])
#     pos = ak.to_list(arr_sel["hit_pmt_position_ids"])
#     charge = ak.to_list(arr_sel["hit_pmt_charges"])
#     time = ak.to_list(arr_sel["hit_pmt_calibrated_times"])

#     events = []
#     for ev_num, s, p, q, t in zip(event_numbers, slots, pos, charge, time):
#         pmt_ids = np.asarray(s, dtype=np.int64) * 100 + np.asarray(p, dtype=np.int64)
#         charges = np.asarray(q, dtype=np.float64)
#         times = np.asarray(t, dtype=np.float64)
#         ev_nums = np.full(len(pmt_ids), ev_num, dtype=np.int64)
#         events.append(np.column_stack((pmt_ids, charges, times, ev_nums)))
#     return events


# def _read_scalar_results(fname, *, fields=None, entry_stop=1):
#     """
#     Read only the scalar-result fields that are actually needed.
#     """
#     with uproot.open(fname) as f:
#         tree = f["vme_analysis_scalar_results"]
#         if fields is None:
#             fields_to_read = list(tree.keys())
#         else:
#             available = set(tree.keys())
#             fields_to_read = []
#             for field in fields:
#                 if field and field in available and field not in fields_to_read:
#                     fields_to_read.append(field)

#         if not fields_to_read:
#             raise KeyError(
#                 "None of the requested scalar fields were found in "
#                 "vme_analysis_scalar_results. Requested fields: "
#                 f"{fields!r}"
#             )

#         arr_s = tree.arrays(fields_to_read, library="ak", entry_stop=entry_stop)

#     data_s = {}
#     for key in arr_s.fields:
#         try:
#             if isinstance(arr_s[key].type.content, ak.types.NumpyType) or "var *" in str(arr_s[key].type):
#                 data_s[key] = ak.to_list(arr_s[key])
#             else:
#                 data_s[key] = np.asarray(arr_s[key])
#         except Exception:
#             data_s[key] = ak.to_list(arr_s[key])
#     return pd.DataFrame(data_s)


# def _field_or_value(df_scalar, field_name, explicit_value, description):
#     if explicit_value is not None:
#         return float(explicit_value)
#     if field_name and field_name in df_scalar.columns:
#         return float(df_scalar[field_name].iloc[0])
#     raise KeyError(
#         f"Could not determine {description}. Field {field_name!r} was not found "
#         "and no explicit value was supplied."
#     )


# # ============================================================
# # Public loader
# # ============================================================
# def get_selected_events(
#     run,
#     n,
#     *,
#     max_selected_events=None,
#     particle="muon",
#     root_file=None,
#     step_size=1000,
#     out_pkl=None,
#     use_t5_hit_time = True,
#     use_peak_time_cut=False,
#     sample_fraction=DEFAULT_SAMPLE_FRACTION,
#     max_sample_events=DEFAULT_MAX_SAMPLE_EVENTS,
#     peak_window=DEFAULT_PEAK_WINDOW,
#     peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
#     peak_time_min=DEFAULT_PEAK_TIME_MIN,
#     peak_time_max=DEFAULT_PEAK_TIME_MAX,
#     eveto_cut=None,
#     tagger_cut=None,
#     tof_primary=None,
#     tof_window=DEFAULT_TOF_WINDOW,
#     t5_particle_nr=None,
#     eveto_scalar_field="act_eveto_cut",
#     tagger_scalar_field="act_tagger_cut",
#     tof_scalar_field=None,
#     momentum_scalar_field=None,
#     verbose=True,
# ):
#     """
#     Load selected WCTE events from a production ROOT file.

#     Parameters
#     ----------
#     n:
#         Maximum number of WCTEReadoutWindows ROOT entries to inspect.
#         This is an entry/read limit, not a selected-event return limit.
#     max_selected_events:
#         Optional hard cap on the number of selected events returned/written.
#         Leave as None to return every selected event found in the first n ROOT
#         entries.

#     Returns a list of arrays.  Each array has columns:
#         0: WCTE PMT ID = slot*100 + pmt_position
#         1: charge
#         2: calibrated time [ns]
#         3: ROOT entry index

#     The selection defaults reproduce the historical muon selection.  For other
#     beam particles, set particle plus any needed TOF/T5 overrides in the driver:
#         tof_primary=..., tof_scalar_field=..., t5_particle_nr=...
#     """
#     particle_label = _canonical_particle_name(particle)
#     if root_file is None:
#         fname = (
#             f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/"
#             f"production_v1_0/{run}/WCTE_merged_production_R{run}.root"
#         )
#     else:
#         fname = str(root_file)

#     tree_name = "WCTEReadoutWindows"
#     branches = [
#         "vme_act0_l_charge", "vme_act0_r_charge",
#         "vme_act1_l_charge", "vme_act1_r_charge",
#         "vme_act2_l_charge", "vme_act2_r_charge",
#         "vme_act3_l_charge", "vme_act3_r_charge",
#         "vme_act4_l_charge", "vme_act4_r_charge",
#         "T5_HasMultipleScintillatorsHit",
#         "T5_HasOutOfTimeWindow",
#         "vme_evt_quality_bitmask",
#         "T5_HasValidHit",
#         "T5_particle_nr",
#         "window_data_quality_mask",
#         "vme_digi_issues_bitmask",
#         "T5_HasInTimeWindow",
#         "T5_hit_is_in_bounds",
#         "hit_mpmt_slot_ids",
#         "hit_pmt_position_ids",
#         "hit_pmt_charges",
#         "hit_pmt_calibrated_times",
#         "vme_tof_t0t1",
#         "vme_tof_corr",
#         "vme_act_eveto",
#         "vme_act_tagger",
#     ]

#     tof_scalar_field = tof_scalar_field or _default_tof_scalar_field(particle_label)
#     momentum_scalar_field = momentum_scalar_field or _default_momentum_scalar_field(particle_label)
#     if t5_particle_nr is None:
#         t5_particle_nr = PARTICLE_TO_T5_NR.get(str(particle).strip().lower(), 1)

#     with uproot.open(fname) as f:
#         total_entries = int(f[tree_name].num_entries)

#     if n is None:
#         n_entries_to_process = total_entries
#     else:
#         n_entries_to_process = min(int(n), total_entries)

#     print("PROCESSING ENTRIES:", n_entries_to_process)

#     if n_entries_to_process < 0:
#         raise ValueError("n must be None or a non-negative integer.")

#     if max_selected_events is not None:
#         max_selected_events = int(max_selected_events)
#         if max_selected_events < 0:
#             raise ValueError("max_selected_events must be None or a non-negative integer.")

#     scalar_fields = [
#         eveto_scalar_field,
#         tagger_scalar_field,
#         tof_scalar_field,
#         momentum_scalar_field,
#     ]
#     df_scalar = _read_scalar_results(fname, fields=scalar_fields, entry_stop=1)
#     eveto_cut_value = _field_or_value(df_scalar, eveto_scalar_field, eveto_cut, "eveto cut")
#     tagger_cut_value = _field_or_value(df_scalar, tagger_scalar_field, tagger_cut, "tagger cut")
#     tof_primary_value = _field_or_value(df_scalar, tof_scalar_field, tof_primary, "TOF mean")

#     p_after_window = np.nan
#     if momentum_scalar_field in df_scalar.columns:
#         try:
#             p_after_window = float(df_scalar[momentum_scalar_field].iloc[0])
#         except Exception:
#             p_after_window = np.nan

#     if verbose:
#         print("Selected-event loader")
#         print("---------------------")
#         print(f"Particle label:              {particle_label}")
#         print(f"ROOT file:                   {fname}")
#         print(f"ROOT entries in file:         {total_entries}")
#         print(f"ROOT entries requested:       {n}")
#         print(f"ROOT entries to process:      {n_entries_to_process}")
#         if max_selected_events is not None:
#             print(f"Max selected events:          {max_selected_events}")
#         print(f"T5 particle nr:              {int(t5_particle_nr)}")
#         print(f"TOF mean [ns]:               {tof_primary_value:.4f}")
#         print(f"TOF window [ns]:             +/- {tof_window}")
#         if np.isfinite(p_after_window):
#             print(f"Beam momentum after window:  {p_after_window}")
#         print("")

#     median_peak_time = None
#     if use_peak_time_cut:
#         median_peak_time = _estimate_median_peak_time(
#             fname=fname,
#             tree_name=tree_name,
#             branches=branches,
#             eveto_cut=eveto_cut_value,
#             tagger_cut=tagger_cut_value,
#             tof_primary=tof_primary_value,
#             n=n_entries_to_process,
#             step_size=step_size,
#             sample_fraction=sample_fraction,
#             max_sample_events=max_sample_events,
#             peak_bin_width=peak_bin_width,
#             peak_time_min=peak_time_min,
#             peak_time_max=peak_time_max,
#             tof_window=tof_window,
#             t5_particle_nr=t5_particle_nr,
#             verbose=verbose,
#         )

#     if n_entries_to_process == 0 or max_selected_events == 0:
#         if verbose:
#             print("Final selection")
#             print("---------------")
#             print("Selected events:               0")
#             print("")
#         return [] if out_pkl is None else None

#     events = [] if out_pkl is None else None
#     fout = open(out_pkl, "wb") if out_pkl is not None else None
#     n_before_peak_cut = 0
#     n_after_peak_cut = 0
#     n_selected_written = 0
#     entry_start = 0

#     try:
#         for arr in _iter_chunks(fname, tree_name, branches, n_entries_to_process, step_size):
#             this_entry_start = entry_start
#             entry_start += len(arr)

#             event_mask = _make_beam_event_mask(
#                 arr,
#                 eveto_cut_value,
#                 tagger_cut_value,
#                 tof_primary_value,
#                 tof_window=tof_window,
#                 t5_particle_nr=t5_particle_nr,
#             )
#             event_mask_np = _mask_to_numpy(event_mask)
#             n_before_peak_cut += int(np.sum(event_mask_np))

#             if np.sum(event_mask_np) == 0:
#                 continue

#             if use_peak_time_cut:
#                 peak_mask_np = _make_peak_time_mask(
#                     arr,
#                     median_peak_time=median_peak_time,
#                     peak_window=peak_window,
#                     peak_bin_width=peak_bin_width,
#                     peak_time_min=peak_time_min,
#                     peak_time_max=peak_time_max,
#                 )
#                 final_mask = event_mask_np & peak_mask_np
#             else:
#                 final_mask = event_mask_np

#             n_after_peak_cut += int(np.sum(final_mask))
#             if np.sum(final_mask) == 0:
#                 continue

#             chunk_events = _arr_to_events(arr, event_mask=final_mask, entry_start=this_entry_start)

#             if max_selected_events is not None:
#                 remaining = max_selected_events - n_selected_written
#                 if remaining <= 0:
#                     break
#                 chunk_events = chunk_events[:remaining]

#             if out_pkl is None:
#                 events.extend(chunk_events)
#             else:
#                 for event in chunk_events:
#                     pickle.dump(event, fout)

#             n_selected_written += len(chunk_events)
#             if max_selected_events is not None and n_selected_written >= max_selected_events:
#                 break
#     finally:
#         if fout is not None:
#             fout.close()

#     if verbose:
#         print("Final selection")
#         print("---------------")
#         if use_peak_time_cut:
#             print(f"Selected before peak-time cut: {n_before_peak_cut}")
#             print(f"Selected after peak-time cut:  {n_after_peak_cut}")
#         else:
#             print(f"Selected events:               {n_after_peak_cut}")
#         if np.isfinite(p_after_window):
#             print(f"Beam momentum after window:    {p_after_window}")
#         print("")

#     return events


# # Particle-generic alias used by the batch drivers.
# def get_particle_events(run, n=None, *, particle="muon", **kwargs):
#     return get_selected_events(run, n, particle=particle, **kwargs)


# # Backward-compatible alias for older notebooks/scripts. Prefer get_particle_events()
# # or get_selected_events().  Passing particle="proton"/"pion"/etc. makes this
# # generic despite the historical name.
# def get_mu_events(run, n=None, **kwargs):
#     kwargs.setdefault("particle", "muon")
#     return get_selected_events(run, n, **kwargs)
