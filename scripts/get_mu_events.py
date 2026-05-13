import uproot
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

#sys.path.insert(0, "../LicketyFit2/event_display")
sys.path.insert(0, "../tables")


# from plot_event import *


# ============================================================
# USER OPTIONS / DEFAULTS
# ============================================================

DEFAULT_PEAK_WINDOW = 100.0       # ns; require event peak within +/- this window
DEFAULT_PEAK_BIN_WIDTH = 50.0    # ns; histogram bin width used to find event peak
DEFAULT_PEAK_TIME_MIN = 0.0      # ns
DEFAULT_PEAK_TIME_MAX = 10000.0  # ns
DEFAULT_SAMPLE_FRACTION = 0.05   # use up to 5% of requested entries to estimate peak time
DEFAULT_MAX_SAMPLE_EVENTS = 50   # maximum number of selected events used for median peak


# ============================================================
# ORIGINAL MUON MASK
# ============================================================

def _make_mu_mask(arr,eveto_cut,tagger_cut,tof_muon):
    first_in_bounds = ak.fill_none(
        ak.firsts(arr["T5_hit_is_in_bounds"], axis=1),
        0
    )

    return (
        (
            arr["vme_act_eveto"] < eveto_cut
        )
        &
        (
            arr["vme_act_tagger"] > tagger_cut
        )
        & (arr["T5_HasMultipleScintillatorsHit"] == False)
        & (arr["T5_HasOutOfTimeWindow"] == False)
        & (arr["vme_evt_quality_bitmask"] == 0)
        & (arr["T5_HasValidHit"] == True)
        & (arr["T5_particle_nr"] == 1)
        & (arr["window_data_quality_mask"] == 0)
        & (arr["vme_digi_issues_bitmask"] == 0)
        & (arr["T5_HasInTimeWindow"] == True)
        # & (first_in_bounds == 1)
        & (arr["vme_tof_corr"] > tof_muon-0.2)
        & (arr["vme_tof_corr"] < tof_muon+0.2)
    )


# ============================================================
# TIMING PEAK HELPERS
# ============================================================

def _event_peak_time(
    times,
    bin_width=DEFAULT_PEAK_BIN_WIDTH,
    time_min=DEFAULT_PEAK_TIME_MIN,
    time_max=DEFAULT_PEAK_TIME_MAX,
):
    """
    Find the main timing peak for one event.

    This returns the center of the hit-time histogram bin with the largest
    number of hits.
    """
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]

    if len(times) == 0:
        return np.nan

    times = times[(times >= time_min) & (times < time_max)]

    if len(times) == 0:
        return np.nan

    bins = np.arange(time_min, time_max + bin_width, bin_width)
    counts, edges = np.histogram(times, bins=bins)

    if len(counts) == 0 or np.max(counts) == 0:
        return np.nan

    max_bin = np.argmax(counts)
    peak_time = 0.5 * (edges[max_bin] + edges[max_bin + 1])

    return peak_time


def _make_peak_time_mask(
    arr,
    median_peak_time,
    peak_window=DEFAULT_PEAK_WINDOW,
    peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
    peak_time_min=DEFAULT_PEAK_TIME_MIN,
    peak_time_max=DEFAULT_PEAK_TIME_MAX,
):
    """
    For each event in arr, find the event's main hit-time peak.

    Keep the event only if:

        abs(event_peak_time - median_peak_time) <= peak_window
    """
    times_list = ak.to_list(arr["hit_pmt_calibrated_times"])

    keep = []

    for times in times_list:
        peak = _event_peak_time(
            times,
            bin_width=peak_bin_width,
            time_min=peak_time_min,
            time_max=peak_time_max,
        )

        if not np.isfinite(peak):
            keep.append(False)
        else:
            keep.append(abs(peak - median_peak_time) <= peak_window)

    return np.asarray(keep, dtype=bool)


def _estimate_median_peak_time(
    fname,
    tree_name,
    branches,
    eveto_cut,
    tagger_cut,
    tof_muon,
    n=None,
    step_size=10000,
    sample_fraction=DEFAULT_SAMPLE_FRACTION,
    max_sample_events=DEFAULT_MAX_SAMPLE_EVENTS,
    peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
    peak_time_min=DEFAULT_PEAK_TIME_MIN,
    peak_time_max=DEFAULT_PEAK_TIME_MAX,
    make_mask_func=None,
    verbose=True,
):
    """
    Estimate the typical good-event peak time from a small initial sample.

    The function reads at most sample_fraction of the requested ROOT entries,
    but only uses at most max_sample_events selected events to calculate the
    median peak time.

    If make_mask_func is provided, the sample is first filtered using that mask.
    For get_mu_events, this means the median peak is estimated from events
    passing your original muon selection.
    """
    with uproot.open(fname) as f:
        tree = f[tree_name]
        total_entries = tree.num_entries

    if n is None:
        n_to_process = total_entries
    else:
        n_to_process = min(int(n), total_entries)

    if n_to_process <= 0:
        raise ValueError("n_to_process is zero. Check n and the input file.")

    # Number of ROOT entries to scan for estimating the timing peak.
    # Use a small fraction of the data
    n_entries_to_scan = int(np.ceil(sample_fraction * n_to_process))
    n_entries_to_scan = max(1, n_entries_to_scan)
    n_entries_to_scan = min(n_entries_to_scan, n_to_process)

    peak_times = []

    for arr in uproot.iterate(
        f"{fname}:{tree_name}",
        branches,
        library="ak",
        entry_stop=n_entries_to_scan,
        step_size=min(step_size, n_entries_to_scan),
    ):
        if make_mask_func is not None:
            sample_mask = make_mask_func(arr,eveto_cut,tagger_cut,tof_muon)
            arr_sample = arr[sample_mask]
        else:
            arr_sample = arr

        times_list = ak.to_list(arr_sample["hit_pmt_calibrated_times"])

        for times in times_list:
            peak = _event_peak_time(
                times,
                bin_width=peak_bin_width,
                time_min=peak_time_min,
                time_max=peak_time_max,
            )

            if np.isfinite(peak):
                peak_times.append(peak)

            if len(peak_times) >= max_sample_events:
                break

        if len(peak_times) >= max_sample_events:
            break

    if len(peak_times) == 0:
        raise RuntimeError(
            "Could not estimate median peak time. "
            "No valid selected events were found in the initial sample. "
            "Try increasing sample_fraction."
        )

    median_peak_time = float(np.median(peak_times))

    if verbose:
        print("")
        print("Peak-time calibration")
        print("---------------------")
        print(f"ROOT entries requested:        {n_to_process}")
        print(f"ROOT entries scanned:          {n_entries_to_scan}")
        print(f"Selected events used for timing scan:          {len(peak_times)}")
        print(f"Estimated median peak time:    {median_peak_time:.2f} ns")
        print("")

    return median_peak_time


# ============================================================
# EVENT CONVERSION
# ============================================================

def _mask_to_numpy(mask):
    """
    Convert either an awkward mask or numpy/list mask to a numpy bool array.
    """
    if mask is None:
        return None

    try:
        return ak.to_numpy(mask).astype(bool)
    except Exception:
        return np.asarray(mask, dtype=bool)


def _arr_to_events(arr, mu_mask=None, entry_start=0):
    """
    Convert awkward array entries into a list of per-event numpy arrays.

    Each returned event has columns:

        column 0: PMT ID = hit_mpmt_slot_id * 100 + hit_pmt_position_id
        column 1: hit charge
        column 2: hit calibrated time
        column 3: original event index in the ROOT array

    Parameters
    ----------
    arr : awkward.Array
        The chunk of data read from the ROOT tree.

    mu_mask : awkward.Array, numpy array, list, or None
        Boolean mask selecting events. If None, all events are used.

    entry_start : int
        Global index of the first event in this chunk.
        This matters for chunked reading because each chunk starts locally at 0.
    """
    event_numbers_all = np.arange(
        entry_start,
        entry_start + len(arr),
        dtype=np.int64
    )

    if mu_mask is None:
        arr_sel = arr
        event_numbers = event_numbers_all
    else:
        mask_np = _mask_to_numpy(mu_mask)

        if len(mask_np) != len(arr):
            raise ValueError(
                f"Mask length {len(mask_np)} does not match array length {len(arr)}"
            )

        arr_sel = arr[mask_np]
        event_numbers = event_numbers_all[mask_np]

    slots = ak.to_list(arr_sel["hit_mpmt_slot_ids"])
    pos = ak.to_list(arr_sel["hit_pmt_position_ids"])
    charge = ak.to_list(arr_sel["hit_pmt_charges"])
    time = ak.to_list(arr_sel["hit_pmt_calibrated_times"])

    events = []

    for ev_num, s, p, q, t in zip(event_numbers, slots, pos, charge, time):
        pmt_ids = np.asarray(s) * 100 + np.asarray(p)
        charges = np.asarray(q)
        times = np.asarray(t)

        ev_nums = np.full(len(pmt_ids), ev_num, dtype=np.int64)

        event = np.column_stack((
            pmt_ids,
            charges,
            times,
            ev_nums,
        ))

        events.append(event)

    return events


# ============================================================
# READ FULL PRODUCTION FILE AND APPLY MUON + PEAK-TIME CUTS
# ============================================================

def get_mu_events(
    run,
    n=None,
    step_size=50000,
    out_pkl=None,
    use_peak_time_cut=True,
    sample_fraction=DEFAULT_SAMPLE_FRACTION,
    max_sample_events=DEFAULT_MAX_SAMPLE_EVENTS,
    peak_window=DEFAULT_PEAK_WINDOW,
    peak_bin_width=DEFAULT_PEAK_BIN_WIDTH,
    peak_time_min=DEFAULT_PEAK_TIME_MIN,
    peak_time_max=DEFAULT_PEAK_TIME_MAX,
    verbose=True,
):
    fname = (
        f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/"
        f"production_v1_0/{run}/WCTE_merged_production_R{run}.root"
    )

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

    median_peak_time = None
    
    with uproot.open(fname) as f:
        tree = f[tree_name]

        t_s = f['vme_analysis_scalar_results']

        arr_s = t_s.arrays(library="ak", entry_stop=n)
        
    data_s = {} 

    for key in arr_s.fields:

        # If branch is jagged (variable length array per event)
        if isinstance(arr_s[key].type.content, ak.types.NumpyType) or "var *" in str(arr_s[key].type):
            data_s[key] = ak.to_list(arr_s[key])   # preserve one list per event

        else:
            data_s[key] = np.array(arr_s[key])     # scalar branch

    df_scalar = pd.DataFrame(data_s)        
    
    eveto_cut = float(df_scalar['act_eveto_cut'].iloc[0])
    tagger_cut = float(df_scalar['act_tagger_cut'].iloc[0])
    tof_muon = float(df_scalar['tof_mean_muon'].iloc[0])
    p_after_window_muon = float(df_scalar['momentum_after_beam_window_mean_muon'].iloc[0])
    

    if use_peak_time_cut:
        median_peak_time = _estimate_median_peak_time(
            fname=fname,
            tree_name=tree_name,
            branches=branches,
            eveto_cut=eveto_cut,
            tagger_cut=tagger_cut,
            tof_muon=tof_muon,
            n=n,
            step_size=step_size,
            sample_fraction=sample_fraction,
            max_sample_events=max_sample_events,
            peak_bin_width=peak_bin_width,
            peak_time_min=peak_time_min,
            peak_time_max=peak_time_max,
            make_mask_func=_make_mu_mask,
            verbose=verbose,
        )

    # --------------------------------------------------
    # SMALL READ MODE
    # --------------------------------------------------
    if n is not None and n <= step_size:
        with uproot.open(fname) as f:
            tree = f[tree_name]
           

            arr = tree.arrays(
                branches,
                library="ak",
                entry_start=0,
                entry_stop=n,
            )
            
            
           
            
        mu_mask = _make_mu_mask(arr,eveto_cut,tagger_cut,tof_muon)
        mu_mask_np = _mask_to_numpy(mu_mask)

        if use_peak_time_cut:
            peak_mask_np = _make_peak_time_mask(
                arr,
                median_peak_time=median_peak_time,
                peak_window=peak_window,
                peak_bin_width=peak_bin_width,
                peak_time_min=peak_time_min,
                peak_time_max=peak_time_max,
            )

            final_mask = mu_mask_np & peak_mask_np

            if verbose:
                print("Final selection")
                print("---------------")
                print(f"Muon-like before peak-time cut: {np.sum(mu_mask_np)}")
                print(f"Muon-like after peak-time cut:  {np.sum(final_mask)}")
                print("")
        else:
            final_mask = mu_mask_np

        events = _arr_to_events(
            arr,
            mu_mask=final_mask,
            entry_start=0,
        )

        if out_pkl is not None:
            with open(out_pkl, "wb") as fout:
                for event in events:
                    pickle.dump(event, fout)

        return events if out_pkl is None else None

    # --------------------------------------------------
    # LARGE CHUNKED MODE
    # --------------------------------------------------
    events = [] if out_pkl is None else None
    fout = open(out_pkl, "wb") if out_pkl is not None else None

    entry_start = 0
    n_mu_before_peak_cut = 0
    n_mu_after_peak_cut = 0
    
    
   


    try:
        for arr in uproot.iterate(
            f"{fname}:{tree_name}",
            branches,
            library="ak",
            entry_stop=n,
            step_size=step_size,
        ):
            this_entry_start = entry_start
            entry_start += len(arr)

            mu_mask = _make_mu_mask(arr,eveto_cut,tagger_cut,tof_muon)
            mu_mask_np = _mask_to_numpy(mu_mask)

            if np.sum(mu_mask_np) == 0:
                continue

            if use_peak_time_cut:
                peak_mask_np = _make_peak_time_mask(
                    arr,
                    median_peak_time=median_peak_time,
                    peak_window=peak_window,
                    peak_bin_width=peak_bin_width,
                    peak_time_min=peak_time_min,
                    peak_time_max=peak_time_max,
                )

                final_mask = mu_mask_np & peak_mask_np
            else:
                final_mask = mu_mask_np

            n_mu_before_peak_cut += int(np.sum(mu_mask_np))
            n_mu_after_peak_cut += int(np.sum(final_mask))

            if np.sum(final_mask) == 0:
                continue

            chunk_events = _arr_to_events(
                arr,
                mu_mask=final_mask,
                entry_start=this_entry_start,
            )

            if out_pkl is None:
                events.extend(chunk_events)
            else:
                for event in chunk_events:
                    pickle.dump(event, fout)

    finally:
        if fout is not None:
            fout.close()

    if verbose:
        print("Final selection")
        print("---------------")
        if use_peak_time_cut:
            print(f"Muon-like before peak-time cut: {n_mu_before_peak_cut}")
            print(f"Muon-like after peak-time cut:  {n_mu_after_peak_cut}")
        else:
            print(f"Muon-like events selected:      {n_mu_after_peak_cut}")
        print('Beam momentum after window:', float(p_after_window_muon))
        print("")
        
       

    return events

