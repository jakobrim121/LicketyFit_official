"""
lf_single_event.py

Importable single-event front-end for the LicketyFit/Minuit particle-track fitter.

Typical notebook usage
----------------------
from lf_single_event import LF_single_event

events = get_mu_events(RUN, N_EVENTS)
event = events[12]

result = LF_single_event(
    event,
    fit_type="charge_only",      # "charge_only", "time_only", "timing_only", "charge_time", "both"
    max_attempts=4,
    ncall=70000,
    verbose=True,
)

result["values"]
"""

import sys
import pickle
from pathlib import Path

import numpy as np

# Directory containing this .py file
HERE = Path(__file__).resolve().parent

# Path to the directory containing the module you want to import
OTHER_MODULE_DIR = (HERE / "../LicketyFit").resolve()


if str(OTHER_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(OTHER_MODULE_DIR))
    
OTHER_MODULE_DIR = (HERE / "../tables").resolve()

if str(OTHER_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(OTHER_MODULE_DIR))

OTHER_MODULE_DIR = (HERE / "../scripts").resolve()

if str(OTHER_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(OTHER_MODULE_DIR))
    
OTHER_MODULE_DIR = (HERE / "../").resolve()

if str(OTHER_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(OTHER_MODULE_DIR))



# =============================================================================
# Default paths
# =============================================================================
# Edit these if your SWAN/lxplus paths differ.
DEFAULT_GEOMETRY_PATH = "/eos/user/j/jrimmer/Geometry"

DEFAULT_EXTRA_SYS_PATHS = [
    # The original batch script used this spelling.
    "../LicketyFit",
    
    "../scripts",
    
    "../tables",

    DEFAULT_GEOMETRY_PATH,
    "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/",
    
]

DEFAULT_GEOMETRY_FILE = (
    "/eos/user/j/jrimmer/Geometry/examples/wcte_bldg157.geo"
)

DEFAULT_MPMT_INFO_PATH = (
    "/eos/experiment/wcte/wcte_tests/mPMT_led_events/dictionaries/"
    "other_mpmt_info_v2.dict"
)

DEFAULT_DELTA_E_ANGULAR_PDF_PATH = (
    "../tables/"
    "delta_e_angular_pdf_table.npz"
)


# =============================================================================
# Default detector and fit settings
# =============================================================================
OUTER_RING = np.array([
    0, 7, 19, 34, 50, 66, 82, 83, 105, 94, 95, 71, 72,
    56, 40, 24, 11, 3, 18
], dtype=int)

INNER_RING = np.array([
    1, 8, 35, 51, 67, 84, 69, 70, 55, 39, 23, 10, 2,
    20, 36, 52, 68, 53, 54, 38, 22, 21, 37, 9
], dtype=int)

OUTSIDE_RING = np.array([
    12, 13, 4, 5, 6, 17, 33, 49, 65, 81, 82, 104,
    93, 86, 87, 72, 57, 41, 25
], dtype=int)

ALL_SLOTS = np.arange(0, 106, dtype=int)

DEFAULT_INACTIVE_SLOTS = [
    27, 32, 45, 74, 77, 79, 85, 91, 99, 9, 67
]

DEFAULT_TRUE_PARAMS = {
    "x0": 0.0,
    "y0": 0.0,
    "z0": -1330.0,
    "cx": 0.0,
    "cy": 0.0,
    "length": 2178.0,
    "t0": 0.0,
}

DEFAULT_INIT_PARAM_SETS = [
    {
        "x0": 0.0,
        "y0": 0.0,
        "z0": -1000.0,
        "cx": 0.0,
        "cy": 0.0,
        "length": 500.0,
        "t0": 0.0,
    },
    {
        "x0": 0.0,
        "y0": 0.0,
        "z0": -1300.0,
        "cx": 0.0,
        "cy": 0.0,
        "length": 500.0,
        "t0": 0.0,
    },
    {
        "x0": 0.0,
        "y0": 0.0,
        "z0": -1500,
        "cx": 0.0,
        "cy": 0.0,
        "length": 700.0,
        "t0": 0.0,
    },
    {
        "x0": 0.0,
        "y0": 0.0,
        "z0": -1000.0,
        "cx": 0.0,
        "cy": 0.0,
        "length": 800.0,
        "t0": 0.0,
    },
]

DEFAULT_LIMITS = {
    "x0": (-2000.0, 2000.0),
    "y0": (-2000.0, 2000.0),
    "z0": (-2000.0, 2000.0),
    "cx": (-0.5, 0.5),
    "cy": (-0.5, 0.5),
    "length": (0.0, 4000.0),
    "t0": (-8.0, 8.0),
}

DEFAULT_ERRORS = {
    "x0": 30.0,
    "y0": 30.0,
    "z0": 30.0,
    "cx": 0.01,
    "cy": 0.01,
    "length": 50.0,
    "t0": 0.1,
}


# Global cache. This avoids re-opening geometry and re-loading tables every time
# LF_single_event(...) is called from the notebook.
_LF_CONTEXT = None


# =============================================================================
# Import and setup helpers
# =============================================================================
def _add_sys_paths(extra_sys_paths=None):
    paths = list(DEFAULT_EXTRA_SYS_PATHS)

    if extra_sys_paths is not None:
        if isinstance(extra_sys_paths, (str, Path)):
            paths.append(str(extra_sys_paths))
        else:
            paths.extend([str(p) for p in extra_sys_paths])

    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


def _import_external_dependencies(extra_sys_paths=None):
    _add_sys_paths(extra_sys_paths)

    try:
        from iminuit import Minuit
        from Geometry.Device import Device

        # Import the actual classes, not the modules.
        from LicketyFit.Event import Event
        from LicketyFit.PMT import PMT
        from LicketyFit.Emitter import Emitter

        from model_muon_cherenkov_collapse import get_energy_distance_tables

    except Exception as exc:
        raise ImportError(
            "Could not import one or more LicketyFit dependencies. "
            "Check DEFAULT_EXTRA_SYS_PATHS near the top of lf_single_event.py, "
            "or pass extra_sys_paths=[...] to LF_setup_context(...)."
        ) from exc

    return {
        "Minuit": Minuit,
        "Device": Device,
        "Event": Event,
        "PMT": PMT,
        "Emitter": Emitter,
        "get_energy_distance_tables": get_energy_distance_tables,
    }

def LF_clear_context():
    """
    Clear the cached detector/model context.

    Use this if you edit paths, geometry, or model tables and want the next
    LF_single_event(...) call to reload everything.
    """
    global _LF_CONTEXT
    _LF_CONTEXT = None


def LF_setup_context(
    *,
    force_reload=False,
    extra_sys_paths=None,
    geometry_file=DEFAULT_GEOMETRY_FILE,
    mpmt_info_path=DEFAULT_MPMT_INFO_PATH,
    delta_e_angular_pdf_path=DEFAULT_DELTA_E_ANGULAR_PDF_PATH,
    n_mpmt_total=106,
    verbose=True,
):
    """
    Initialize and cache geometry, PMT model, emitter template, mPMT info, and tables.

    You usually do not need to call this explicitly. LF_single_event(...) calls it
    on the first use. Calling it manually can be useful if you want to confirm
    paths before fitting.
    """
    global _LF_CONTEXT

    if (_LF_CONTEXT is not None) and (not force_reload):
        return _LF_CONTEXT

    deps = _import_external_dependencies(extra_sys_paths=extra_sys_paths)

    if verbose:
        print("Loading energy-distance tables...")
    overall_distances, init_energy_table, distance_rows = (
        deps["get_energy_distance_tables"]()
    )

    if verbose:
        print(f"Opening geometry: {geometry_file}")
    hall = deps["Device"].open_file(geometry_file)
    WCD = hall.wcds[0]

    if verbose:
        print(f"Loading mPMT info: {mpmt_info_path}")
    with open(mpmt_info_path, "rb") as f:
        mpmt_info = pickle.load(f)

    if verbose:
        print("Constructing Emitter and PMT models...")
    emitter_model = deps["Emitter"](
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.96,
        500.0,
        18.0,
    )

    emitter_model.load_delta_e_angular_pdf_table(delta_e_angular_pdf_path)

    pmt_model = deps["PMT"](1.0, 0.3, 1.0, 40.0, 0.2, 0.0)

    _LF_CONTEXT = {
        **deps,
        "overall_distances": overall_distances,
        "init_energy_table": init_energy_table,
        "distance_rows": distance_rows,
        "WCD": WCD,
        "mpmt_info": mpmt_info,
        "emitter_template": emitter_model.copy(),
        "pmt_model": pmt_model,
        "n_mpmt_total": int(n_mpmt_total),

        # Built lazily from the first Event object. This is cached because the
        # active PMT ordering is fixed for these settings.
        "p_locations": None,
        "direction_zs": None,
        "mpmt_slots": None,
        "ring_keep_mask_cache": {},
    }

    if verbose:
        print("LF context ready.")

    return _LF_CONTEXT


# =============================================================================
# Fit-mode and ring helpers
# =============================================================================
def _parse_fit_type(fit_type):
    """
    Convert a user string to likelihood booleans.

    Accepted names:
        "charge_only", "charge", "q"
        "time_only", "timing_only", "time", "timing", "t"
        "charge_time", "charge+time", "both", "q+t"
    """
    key = str(fit_type).strip().lower().replace("-", "_").replace(" ", "_")

    charge_only = {"charge_only", "charge", "q", "pe", "pes"}
    time_only = {"time_only", "timing_only", "time", "timing", "t"}
    charge_time = {
        "charge_time",
        "charge+time",
        "charge_and_time",
        "charge_timing",
        "both",
        "q+t",
        "pe+t",
        "pes+t",
    }

    if key in charge_only:
        return True, False, "charge_only"
    if key in time_only:
        return False, True, "timing_only"
    if key in charge_time:
        return True, True, "charge_time"

    raise ValueError(
        f"Unknown fit_type={fit_type!r}. Use one of: "
        "'charge_only', 'time_only'/'timing_only', or 'charge_time'/'both'."
    )


def _resolve_ring_keep_slots(ring_keep_slots):
    """
    Convert ring_keep_slots into an array of mPMT slots.

    Useful string presets:
        "all"           -> all 0..105 slots
        "outer"         -> OUTER_RING
        "inner"         -> INNER_RING
        "outer_inner"   -> OUTER_RING + INNER_RING
        "outside"       -> OUTSIDE_RING
        "original_all"  -> OUTER_RING + INNER_RING + OUTSIDE_RING
    """
    if ring_keep_slots is None:
        return ALL_SLOTS.copy()

    if isinstance(ring_keep_slots, str):
        key = ring_keep_slots.strip().lower().replace("-", "_").replace(" ", "_")

        if key in {"all", "all_slots", "none"}:
            return ALL_SLOTS.copy()
        if key == "outer":
            return OUTER_RING.copy()
        if key == "inner":
            return INNER_RING.copy()
        if key in {"outer_inner", "inner_outer", "ring", "rings"}:
            return np.unique(np.concatenate([OUTER_RING, INNER_RING])).astype(int)
        if key == "outside":
            return OUTSIDE_RING.copy()
        if key in {"original_all", "original_ring_set"}:
            return np.unique(
                np.concatenate([OUTER_RING, INNER_RING, OUTSIDE_RING])
            ).astype(int)

        raise ValueError(
            f"Unknown ring_keep_slots preset {ring_keep_slots!r}. "
            "Use 'all', 'outer', 'inner', 'outer_inner', 'outside', "
            "'original_all', or an array/list of slot ids."
        )

    return np.asarray(ring_keep_slots, dtype=int)


def _validate_seed_sets(init_param_sets):
    param_names = ("x0", "y0", "z0", "cx", "cy", "length", "t0")

    if init_param_sets is None:
        init_param_sets = DEFAULT_INIT_PARAM_SETS

    if not (1 <= len(init_param_sets) <= 10):
        raise ValueError("init_param_sets must contain between 1 and 10 seeds.")

    seeds = []
    for i, seed in enumerate(init_param_sets):
        missing = [k for k in param_names if k not in seed]
        if missing:
            raise ValueError(f"Seed {i} is missing keys: {missing}")

        seeds.append({k: float(seed[k]) for k in param_names})

    return seeds


# =============================================================================
# Event and observable helpers
# =============================================================================
def apply_peak_time_window(
    event,
    *,
    use_peak_time_window=True,
    time_hist_min=0.0,
    time_hist_max=4000.0,
    time_hist_bin_width=1.0,
    n_bins_before_peak=20,
    n_bins_after_peak=5,
):
    """
    Apply the same peak-centered timing window used in the batch driver.

    Parameters
    ----------
    event : array-like
        Single event with columns [pmt_id, charge, time].
    """
    event = np.asarray(event)

    if (not use_peak_time_window) or len(event) == 0:
        return event

    bins = np.arange(
        float(time_hist_min),
        float(time_hist_max) + float(time_hist_bin_width),
        float(time_hist_bin_width),
    )

    hist, edges = np.histogram(event[:, 2], bins=bins)

    if len(hist) == 0 or np.max(hist) <= 0:
        return event

    max_idx = int(np.argmax(hist))

    low_idx = max(0, max_idx - int(n_bins_before_peak))
    high_idx = min(len(edges) - 1, max_idx + int(n_bins_after_peak))

    min_time = edges[low_idx]
    cut_time = edges[high_idx]

    time_mask = (event[:, 2] > min_time) & (event[:, 2] < cut_time)
    return event[time_mask]


def sim_to_event(
    sim_data,
    ctx,
    *,
    inactive_slots=None,
    pe_scale=143.0,
    shift_times=True,
    n_earliest_for_t0=10,
):
    """
    Convert one event array into the LicketyFit Event class.

    Expected sim_data columns:
        sim_data[:, 0] = WCTE-style PMT id, slot * 100 + pmt_position_id
        sim_data[:, 1] = charge
        sim_data[:, 2] = time

    pe_scale is accepted for API symmetry but is not used here directly.
    """
    del pe_scale

    sim_data = np.asarray(sim_data)
    inactive_slots = DEFAULT_INACTIVE_SLOTS if inactive_slots is None else inactive_slots

    Event = ctx["Event"]
    WCD = ctx["WCD"]
    n_mpmt_total = int(ctx["n_mpmt_total"])

    slots = []
    pmt_pos_ids = []
    charges = []
    times = []
    vw = 223.0598645833333  # mm/ns

    for i in range(len(sim_data[:, 0])):
        sim_pmt = sim_data[i][0]
        wcte_pmt = sim_pmt

        slots.append(int(wcte_pmt / 100))
        pmt_pos_ids.append(int(wcte_pmt % 100))
        charges.append(float(sim_data[i][1]))
        times.append(float(sim_data[i][2]))

    ev = Event(0, 0, n_mpmt_total)
    ev.set_mpmt_status(list(range(n_mpmt_total)), True)

    wcte_pmt_ids = []

    for i_mpmt in range(n_mpmt_total):
        if i_mpmt in inactive_slots:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), False)
        else:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), True)

            for i_pmt in range(19):
                wcte_pmt_ids.append(i_mpmt * 100 + i_pmt)

    for s, p, q, t in zip(slots, pmt_pos_ids, charges, times):
        ev.hit_times[s][p].append(t)
        ev.hit_charges[s][p].append(q)

    if shift_times:
        bp_loc = np.array([0.0, 0.0, -1350.0])
        early_hits = []

        for i_mpmt in range(ev.n_mpmt):
            for i_pmt in range(ev.npmt_per_mpmt):
                if len(ev.hit_times[i_mpmt][i_pmt]) == 0:
                    continue

                pmt_loc = WCD.mpmts[i_mpmt].pmts[i_pmt].get_placement("est")[
                    "location"
                ]
                r = np.linalg.norm(pmt_loc - bp_loc)

                for t in ev.hit_times[i_mpmt][i_pmt]:
                    t0_est = float(t) - r / vw
                    early_hits.append(
                        {
                            "time": float(t),
                            "t0_est": t0_est,
                            "r": r,
                            "i_mpmt": i_mpmt,
                            "i_pmt": i_pmt,
                        }
                    )

        if len(early_hits) > 0:
            early_hits = sorted(early_hits, key=lambda x: x["time"])

            n_use = min(int(n_earliest_for_t0), len(early_hits))
            earliest_hits = early_hits[:n_use]

            time_offset = np.median([hit["t0_est"] for hit in earliest_hits])

            for i_mpmt in range(ev.n_mpmt):
                for i_pmt in range(ev.npmt_per_mpmt):
                    ev.hit_times[i_mpmt][i_pmt] = [
                        t - time_offset for t in ev.hit_times[i_mpmt][i_pmt]
                    ]

            ev.global_time_offset = float(time_offset)

    return ev, np.asarray(wcte_pmt_ids, dtype=int)


def build_observables_from_event(ev, *, pe_scale=143.0):
    """
    Build per-PMT observed PE and charge-weighted hit-time arrays.

    Missing times are stored as np.nan.
    """
    obs_pes = []
    obs_ts = []

    for i_mpmt in range(ev.n_mpmt):
        if not ev.mpmt_status[i_mpmt]:
            continue

        for i_pmt in range(ev.npmt_per_mpmt):
            if not ev.pmt_status[i_mpmt][i_pmt]:
                continue

            charges = ev.hit_charges[i_mpmt][i_pmt]
            times = ev.hit_times[i_mpmt][i_pmt]

            q = np.asarray(charges, dtype=np.float64)
            t = np.asarray(times, dtype=np.float64)

            if len(charges) == 0:
                obs_pes.append(0.0)
                obs_ts.append(np.nan)
            else:
                obs_pes.append(float(np.sum(q)) / float(pe_scale))
                obs_ts.append(float(np.sum(q * t) / np.sum(q)))

    return np.asarray(obs_pes, dtype=np.float64), np.asarray(obs_ts, dtype=np.float64)


def get_mpmt_slot_type(mpmt_slots, ctx):
    """
    Return mPMT type labels expected by Emitter.get_expected_pes_ts.
    """
    mpmt_info = ctx["mpmt_info"]

    slot_type = []

    for slot in mpmt_slots:
        slot = int(slot)

        try:
            if mpmt_info[slot]["mpmt_site"] == "TRI":
                if mpmt_info[slot]["mpmt_type"] == "In-situ":
                    slot_type.append("tri_insitu")
                else:
                    slot_type.append("tri_exsitu")
            else:
                if mpmt_info[slot]["mpmt_type"] == "In-situ":
                    slot_type.append("wut_insitu")
                else:
                    slot_type.append("wut_exsitu")
        except Exception:
            slot_type.append("empty")

    return slot_type


def _get_or_build_geometry_cache(ctx, ev, pmt_ids, ring_keep_slots):
    """
    Build PMT placement arrays once and cache them in ctx.
    """
    if ctx["p_locations"] is None or ctx["direction_zs"] is None:
        p_locations, direction_zs, mpmt_slots = (
            ctx["emitter_template"].get_pmt_placements(ev, ctx["WCD"], "est")
        )

        ctx["p_locations"] = p_locations
        ctx["direction_zs"] = direction_zs
        ctx["mpmt_slots"] = mpmt_slots

    ring_keep_slots = np.asarray(ring_keep_slots, dtype=int)
    cache_key = tuple(sorted(int(x) for x in ring_keep_slots))

    if cache_key not in ctx["ring_keep_mask_cache"]:
        mpmt_ids = np.asarray(pmt_ids, dtype=int) // 100
        ctx["ring_keep_mask_cache"][cache_key] = np.isin(mpmt_ids, ring_keep_slots)

    return (
        ctx["p_locations"],
        ctx["direction_zs"],
        ctx["mpmt_slots"],
        ctx["ring_keep_mask_cache"][cache_key],
    )


def apply_ring_mask_to_observables(obs_pes, obs_ts, ring_keep_mask, *, mode="both"):
    """
    Apply an mPMT ring mask to observed PEs and/or observed times.
    """
    if mode not in {"none", "pes", "ts", "both"}:
        raise ValueError("ring_mask_mode must be one of: 'none', 'pes', 'ts', 'both'.")

    obs_pes = obs_pes.copy()
    obs_ts = obs_ts.copy()

    if mode in {"pes", "both"}:
        obs_pes[~ring_keep_mask] = 0.0

    if mode in {"ts", "both"}:
        obs_ts[~ring_keep_mask] = np.nan

    return obs_pes, obs_ts


def prepare_single_event_observables(
    event,
    ctx,
    *,
    pe_scale=143.0,
    inactive_slots=None,
    shift_times=True,
    n_earliest_for_t0=10,
    use_peak_time_window=True,
    time_hist_min=0.0,
    time_hist_max=4000.0,
    time_hist_bin_width=1.0,
    n_bins_before_peak=20,
    n_bins_after_peak=5,
    ring_mask_mode="both",
    ring_keep_slots=None,
):
    """
    Convert a raw single-event array into the arrays used by the likelihood.
    """
    ring_keep_slots = _resolve_ring_keep_slots(ring_keep_slots)

    event_cut = apply_peak_time_window(
        event,
        use_peak_time_window=use_peak_time_window,
        time_hist_min=time_hist_min,
        time_hist_max=time_hist_max,
        time_hist_bin_width=time_hist_bin_width,
        n_bins_before_peak=n_bins_before_peak,
        n_bins_after_peak=n_bins_after_peak,
    )

    ev, pmt_ids = sim_to_event(
        event_cut,
        ctx,
        inactive_slots=inactive_slots,
        pe_scale=pe_scale,
        shift_times=shift_times,
        n_earliest_for_t0=n_earliest_for_t0,
    )

    p_locations, direction_zs, mpmt_slots, ring_keep_mask = (
        _get_or_build_geometry_cache(ctx, ev, pmt_ids, ring_keep_slots)
    )

    obs_pes, obs_ts = build_observables_from_event(ev, pe_scale=pe_scale)

    obs_pes, obs_ts = apply_ring_mask_to_observables(
        obs_pes,
        obs_ts,
        ring_keep_mask,
        mode=ring_mask_mode,
    )

    mpmt_types = get_mpmt_slot_type(mpmt_slots, ctx)

    return {
        "obs_pes": obs_pes,
        "obs_ts": obs_ts,
        "mpmt_types": mpmt_types,
        "ev": ev,
        "event_cut": event_cut,
        "p_locations": p_locations,
        "direction_zs": direction_zs,
        "mpmt_slots": mpmt_slots,
        "ring_keep_mask": ring_keep_mask,
    }


# =============================================================================
# Likelihood helpers
# =============================================================================
def get_t0_prior_sigma(obs_pes, obs_ts):
    n_timed = np.count_nonzero(np.isfinite(obs_ts))
    total_pe = np.sum(obs_pes)

    if (n_timed < 250) or (total_pe < 300):
        return 0.1
    elif (n_timed < 275) or (total_pe < 350):
        return 0.2
    elif (n_timed < 300) or (total_pe < 400):
        return 0.3
    elif (n_timed < 325) or (total_pe < 450):
        return 0.4
    elif (n_timed < 350) or (total_pe < 500):
        return 0.5
    elif (n_timed < 375) or (total_pe < 550):
        return 0.6
    elif (n_timed < 400) or (total_pe < 600):
        return 0.7
    elif (n_timed < 425) or (total_pe < 650):
        return 0.8
    elif (n_timed < 450) or (total_pe < 700):
        return 1.0
    elif (n_timed < 475) or (total_pe < 750):
        return 1.2
    elif (n_timed < 500) or (total_pe < 800):
        return 1.4
    elif (n_timed < 525) or (total_pe < 850):
        return 1.6
    elif (n_timed < 550) or (total_pe < 900):
        return 1.8
    else:
        return 2.0


def get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts, ctx):
    """
    Timing-only negative log likelihood.

    This matches the timing contribution used in the original batch driver.
    """
    pmt_model = ctx["pmt_model"]

    exp_pes = np.asarray(exp_pes, dtype=np.float64)
    obs_pes = np.asarray(obs_pes, dtype=np.float64)
    exp_ts = np.asarray(exp_ts, dtype=np.float64)
    obs_ts = np.asarray(obs_ts, dtype=np.float64)

    mask = (
        (exp_pes > 0.0)
        & (obs_pes > 0.0)
        & np.isfinite(exp_ts)
        & np.isfinite(obs_ts)
    )

    if not np.any(mask):
        return 1e30

    sigma_t = pmt_model.single_pe_time_std / np.sqrt(obs_pes[mask])
    dt = (obs_ts[mask] - exp_ts[mask]) / sigma_t

    return float(0.5 * np.sum(dt * dt))


def evaluate_pmt_nll(
    exp_pes,
    obs_pes,
    exp_ts,
    obs_ts,
    ctx,
    *,
    use_charge_likelihood=True,
    use_timing_likelihood=False,
):
    pmt_model = ctx["pmt_model"]

    if use_charge_likelihood and use_timing_likelihood:
        return pmt_model.get_neg_log_likelihood_npe_t(
            exp_pes,
            obs_pes,
            exp_ts,
            obs_ts,
        )

    if use_charge_likelihood:
        return pmt_model.get_neg_log_likelihood_npe(exp_pes, obs_pes)

    return get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts, ctx)


def evaluate_neg_log_likelihood(
    obs_pes,
    obs_ts,
    emitter,
    mpmt_types,
    ctx,
    *,
    x0,
    y0,
    z0,
    cx,
    cy,
    length,
    t0,
    use_charge_likelihood=True,
    use_timing_likelihood=False,
    use_t0_prior=False,
):
    """
    Evaluate the selected negative log likelihood for one parameter point.
    """
    cz2 = 1.0 - cx * cx - cy * cy
    if cz2 < 0.0:
        return 1e30

    cz = np.sqrt(cz2)

    emitter.start_coord = (x0, y0, z0)
    emitter.starting_time = t0
    emitter.direction = (cx, cy, cz)

    init_ke = emitter.refresh_kinematics_from_length(length)

    s = emitter.get_emission_points(ctx["p_locations"], init_ke)

    exp_pes, exp_ts = emitter.get_expected_pes_ts(
        ctx["WCD"],
        s,
        ctx["p_locations"],
        ctx["direction_zs"],
        mpmt_types,
        obs_pes,
    )

    nll = evaluate_pmt_nll(
        exp_pes,
        obs_pes,
        exp_ts,
        obs_ts,
        ctx,
        use_charge_likelihood=use_charge_likelihood,
        use_timing_likelihood=use_timing_likelihood,
    )

    if not np.isfinite(nll):
        return 1e30

    if use_timing_likelihood and use_t0_prior:
        sigma_t0 = get_t0_prior_sigma(obs_pes, obs_ts)
        nll += abs(0.5 * (t0 / sigma_t0) ** 2)

    return float(nll)


def select_best_initial_seed(
    obs_pes,
    obs_ts,
    mpmt_types,
    ctx,
    *,
    init_param_sets=None,
    use_charge_likelihood=True,
    use_timing_likelihood=False,
    use_t0_prior=False,
):
    """
    Evaluate all user-provided seed points and sort by raw NLL.
    """
    init_param_sets = _validate_seed_sets(init_param_sets)

    seed_scan = []

    for i, seed in enumerate(init_param_sets):
        emitter = ctx["emitter_template"].copy()

        fval = evaluate_neg_log_likelihood(
            obs_pes,
            obs_ts,
            emitter,
            mpmt_types,
            ctx,
            x0=seed["x0"],
            y0=seed["y0"],
            z0=seed["z0"],
            cx=seed["cx"],
            cy=seed["cy"],
            length=seed["length"],
            t0=seed["t0"],
            use_charge_likelihood=use_charge_likelihood,
            use_timing_likelihood=use_timing_likelihood,
            use_t0_prior=use_t0_prior,
        )

        if not np.isfinite(fval):
            fval = np.inf

        seed_scan.append(
            {
                "seed_index": i,
                "fval": float(fval),
                "params": dict(seed),
            }
        )

    return sorted(seed_scan, key=lambda x: x["fval"])


def make_minuit_for_event(
    obs_pes,
    obs_ts,
    start_params,
    mpmt_types,
    ctx,
    *,
    use_charge_likelihood=True,
    use_timing_likelihood=False,
    use_t0_prior=False,
    limits=None,
    errors=None,
    m_strategy=1,
):
    """
    Build a Minuit object for one event.
    """
    Minuit = ctx["Minuit"]
    emitter = ctx["emitter_template"].copy()

    limits = DEFAULT_LIMITS if limits is None else limits
    errors = DEFAULT_ERRORS if errors is None else errors

    def nll(x0, y0, z0, cx, cy, length, t0):
        return evaluate_neg_log_likelihood(
            obs_pes,
            obs_ts,
            emitter,
            mpmt_types,
            ctx,
            x0=x0,
            y0=y0,
            z0=z0,
            cx=cx,
            cy=cy,
            length=length,
            t0=t0,
            use_charge_likelihood=use_charge_likelihood,
            use_timing_likelihood=use_timing_likelihood,
            use_t0_prior=use_t0_prior,
        )

    m = Minuit(nll, **start_params)

    for name, lim in limits.items():
        m.limits[name] = lim

    for name, err in errors.items():
        m.errors[name] = err

    # Charge-only likelihood does not depend on t0. Fix it to avoid a flat
    # Minuit direction.
    if not use_timing_likelihood:
        m.fixed["t0"] = True

    m.errordef = Minuit.LIKELIHOOD
    m.strategy = int(m_strategy)

    return m


def compute_true_fcn(
    obs_pes,
    obs_ts,
    mpmt_types,
    ctx,
    *,
    true_params=None,
    use_charge_likelihood=True,
    use_timing_likelihood=False,
    use_t0_prior=False,
):
    """
    Evaluate the FCN at a supplied truth point.
    """
    true_params = DEFAULT_TRUE_PARAMS if true_params is None else true_params

    emitter = ctx["emitter_template"].copy()

    return evaluate_neg_log_likelihood(
        obs_pes,
        obs_ts,
        emitter,
        mpmt_types,
        ctx,
        x0=float(true_params["x0"]),
        y0=float(true_params["y0"]),
        z0=float(true_params["z0"]),
        cx=float(true_params["cx"]),
        cy=float(true_params["cy"]),
        length=float(true_params["length"]),
        t0=float(true_params["t0"]),
        use_charge_likelihood=use_charge_likelihood,
        use_timing_likelihood=use_timing_likelihood,
        use_t0_prior=use_t0_prior,
    )


# =============================================================================
# Main public function
# =============================================================================
def LF_single_event(
    event,
    fit_type="charge_only",
    *,
    # Optional context/path control.
    ctx=None,
    force_reload_context=False,
    extra_sys_paths=None,
    geometry_file=DEFAULT_GEOMETRY_FILE,
    mpmt_info_path=DEFAULT_MPMT_INFO_PATH,
    delta_e_angular_pdf_path=DEFAULT_DELTA_E_ANGULAR_PDF_PATH,

    # Event preprocessing.
    pe_scale=143.0,
    inactive_slots=None,
    shift_times=True,
    n_earliest_for_t0=10,
    use_peak_time_window=True,
    time_hist_min=0.0,
    time_hist_max=4000.0,
    time_hist_bin_width=1.0,
    n_bins_before_peak=20,
    n_bins_after_peak=5,

    # Ring selection.
    ring_mask_mode="both",
    ring_keep_slots=None,

    # Likelihood options.
    use_t0_prior=False,

    # Seed and retry settings.
    init_param_sets=None,
    max_attempts=4,
    fcn_retry_threshold=1100.0,
    length_retry_threshold=2700.0,
    z_seed_eps=20.0,
    length_seed_eps=40.0,
    t_min=-8.0,

    # Minuit settings.
    ncall=70000,
    ncall_simplex=None,
    ncall_migrad=None,
    run_simplex=True,
    run_migrad=True,
    m_strategy=1,
    limits=None,
    errors=None,

    # Optional truth FCN.
    true_params=None,
    compute_truth=True,

    # Return/print options.
    return_minuit=False,
    verbose=True,
):
    """
    Fit exactly one event from get_mu_events(RUN, N_EVENTS).

    Parameters
    ----------
    event : array-like
        A single event, e.g.

            events = get_mu_events(RUN, N_EVENTS)
            event = events[event_index]

        The expected columns are [pmt_id, charge, time].

    fit_type : str
        One of:
            "charge_only"
            "time_only" / "timing_only"
            "charge_time" / "both"

    Returns
    -------
    result : dict
        Important keys:
            result["values"]       -> fitted parameter values
            result["errors"]       -> Minuit errors
            result["fval"]         -> best fit FCN
            result["true_fcn"]     -> FCN at true_params, if compute_truth=True
            result["seed_scan"]    -> initial seeds sorted by raw NLL
            result["obs_pes"]      -> observed PE array used in the fit
            result["obs_ts"]       -> observed time array used in the fit
            result["event_cut"]    -> event after peak time-window cut
    """
    use_charge_likelihood, use_timing_likelihood, likelihood_mode = _parse_fit_type(
        fit_type
    )

    init_param_sets = _validate_seed_sets(init_param_sets)

    if ctx is None:
        ctx = LF_setup_context(
            force_reload=force_reload_context,
            extra_sys_paths=extra_sys_paths,
            geometry_file=geometry_file,
            mpmt_info_path=mpmt_info_path,
            delta_e_angular_pdf_path=delta_e_angular_pdf_path,
            verbose=verbose,
        )

    if ncall_simplex is None:
        ncall_simplex = ncall
    if ncall_migrad is None:
        ncall_migrad = ncall

    prepared = prepare_single_event_observables(
        event,
        ctx,
        pe_scale=pe_scale,
        inactive_slots=inactive_slots,
        shift_times=shift_times,
        n_earliest_for_t0=n_earliest_for_t0,
        use_peak_time_window=use_peak_time_window,
        time_hist_min=time_hist_min,
        time_hist_max=time_hist_max,
        time_hist_bin_width=time_hist_bin_width,
        n_bins_before_peak=n_bins_before_peak,
        n_bins_after_peak=n_bins_after_peak,
        ring_mask_mode=ring_mask_mode,
        ring_keep_slots=ring_keep_slots,
    )

    obs_pes = prepared["obs_pes"]
    obs_ts = prepared["obs_ts"]
    mpmt_types = prepared["mpmt_types"]

    seed_scan_sorted = select_best_initial_seed(
        obs_pes,
        obs_ts,
        mpmt_types,
        ctx,
        init_param_sets=init_param_sets,
        use_charge_likelihood=use_charge_likelihood,
        use_timing_likelihood=use_timing_likelihood,
        use_t0_prior=use_t0_prior,
    )

    best_result = None
    best_rank = (999, np.inf)

    n_seed_attempts = min(int(max_attempts), len(seed_scan_sorted))

    if verbose:
        print(f"LF_single_event fit_type: {likelihood_mode}")
        print(f"Number of hits before time cut: {len(event)}")
        print(f"Number of hits after time cut:  {len(prepared['event_cut'])}")
        print(f"Trying {n_seed_attempts} seed(s).")

    for attempt in range(1, n_seed_attempts + 1):
        chosen_seed_info = seed_scan_sorted[attempt - 1]
        start_params = dict(chosen_seed_info["params"])

        chosen_seed_idx = int(chosen_seed_info["seed_index"])
        chosen_seed_fcn = float(chosen_seed_info["fval"])

        if verbose:
            print(
                f"\nAttempt {attempt}/{n_seed_attempts}: "
                f"seed {chosen_seed_idx}, seed FCN = {chosen_seed_fcn:.3f}"
            )

        m = make_minuit_for_event(
            obs_pes,
            obs_ts,
            start_params,
            mpmt_types,
            ctx,
            use_charge_likelihood=use_charge_likelihood,
            use_timing_likelihood=use_timing_likelihood,
            use_t0_prior=use_t0_prior,
            limits=limits,
            errors=errors,
            m_strategy=m_strategy,
        )

        if run_simplex:
            m.simplex(ncall=int(ncall_simplex))

        if run_migrad:
            m.migrad(ncall=int(ncall_migrad))

        current_fval = float(m.fval) if np.isfinite(m.fval) else np.inf
        current_values = m.values.to_dict()

        fitted_z0 = float(current_values["z0"])
        fitted_length = float(current_values["length"])

        length_too_large = fitted_length > float(length_retry_threshold)

        z_near_seed = abs(fitted_z0 - float(start_params["z0"])) <= float(z_seed_eps)

        length_near_seed = (
            abs(fitted_length - float(start_params["length"]))
            <= float(length_seed_eps)
        )

        seed_stuck = z_near_seed and length_near_seed

        below_t_min = (
            use_timing_likelihood
            and (float(current_values["t0"]) < float(t_min))
        )

        result = {
            "values": current_values,
            "errors": m.errors.to_dict(),
            "fval": current_fval,
            "valid": bool(m.valid),
            "accurate": bool(m.accurate),
            "edm": float(m.fmin.edm) if m.fmin is not None else np.nan,
            "nfcn": int(m.nfcn),
            "attempts": attempt,
            "fit_type": likelihood_mode,
            "use_charge_likelihood": bool(use_charge_likelihood),
            "use_timing_likelihood": bool(use_timing_likelihood),
            "use_t0_prior": bool(use_t0_prior),
            "length_too_large": bool(length_too_large),
            "seed_stuck": bool(seed_stuck),
            "z_near_seed": bool(z_near_seed),
            "length_near_seed": bool(length_near_seed),
            "below_t_min": bool(below_t_min),
            "chosen_seed_index": chosen_seed_idx,
            "chosen_seed_fcn": chosen_seed_fcn,
            "chosen_seed_params": dict(start_params),
            "seed_scan": seed_scan_sorted,
            "obs_pes": obs_pes,
            "obs_ts": obs_ts,
            "mpmt_types": mpmt_types,
            "event_cut": prepared["event_cut"],
            "event_obj": prepared["ev"],
            "ring_keep_mask": prepared["ring_keep_mask"],
        }

        if return_minuit:
            result["minuit"] = m

        if (
            np.isfinite(current_fval)
            and current_fval <= float(fcn_retry_threshold)
            and (not length_too_large)
            and (not seed_stuck)
            and (not below_t_min)
        ):
            current_rank = (0, current_fval)
        elif (
            np.isfinite(current_fval)
            and (not length_too_large)
            and (not seed_stuck)
            and (not below_t_min)
        ):
            current_rank = (1, current_fval)
        elif (
            np.isfinite(current_fval)
            and (not length_too_large)
            and (not seed_stuck)
        ):
            current_rank = (2, current_fval)
        elif np.isfinite(current_fval) and (not seed_stuck):
            current_rank = (3, current_fval)
        else:
            current_rank = (4, current_fval)

        if current_rank < best_rank:
            best_rank = current_rank
            best_result = result

        if verbose:
            vals = current_values
            print(
                f"  fval={current_fval:.3f}"
                f" nfcn={result['nfcn']}"
            )
            print(
                f"  x={vals['x0']:.1f}, y={vals['y0']:.1f}, z={vals['z0']:.1f}, "
                f"cx={vals['cx']:.5f}, cy={vals['cy']:.5f}, "
                f"length={vals['length']:.1f}, t0={vals['t0']:.3f}"
            )
#             print(
#                 f"  retry flags: length_too_large={length_too_large}, "
#                 f"seed_stuck={seed_stuck}, below_t_min={below_t_min}"
#             )

        if current_rank[0] == 0:
            break

    if best_result is None:
        raise RuntimeError("No fit attempt produced a result.")

    if compute_truth:
        best_result["true_fcn"] = compute_true_fcn(
            obs_pes,
            obs_ts,
            mpmt_types,
            ctx,
            true_params=true_params,
            use_charge_likelihood=use_charge_likelihood,
            use_timing_likelihood=use_timing_likelihood,
            use_t0_prior=use_t0_prior,
        )
    else:
        best_result["true_fcn"] = None

    if verbose:
        vals = best_result["values"]
        print("\nBest result:")
        print(
            f"  fval={best_result['fval']:.3f}, "
            f"true_fcn={best_result['true_fcn']}, "
            f"attempts={best_result['attempts']}, "
            f"seed={best_result['chosen_seed_index']}"
        )
        print(
            f"  x={vals['x0']:.2f}, y={vals['y0']:.2f}, z={vals['z0']:.2f}, "
            f"cx={vals['cx']:.5f}, cy={vals['cy']:.5f}, "
            f"length={vals['length']:.2f}, t0={vals['t0']:.4f}"
        )

    return best_result
