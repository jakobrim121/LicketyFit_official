"""
Optimized batch front-end for the Minuit Cherenkov fit.

This version allows the user to define 3-5 initial parameter sets.
For each event, the code evaluates the negative log likelihood at each
seed, ranks them, and then uses those ranked seeds deterministically on
successive retry attempts. No random seed perturbation is used.
"""

import os
import sys
import pickle
import multiprocessing as mp

import numpy as np
from iminuit import Minuit

geometry_path = "/eos/user/j/jrimmer/Geometry/"


sys.path.insert(0, "../LicketyFit")
sys.path.insert(0, geometry_path)
sys.path.insert(0, "../")


from Geometry.Device import Device
from LicketyFit.Event import Event
from LicketyFit.PMT import PMT
from LicketyFit.Emitter import Emitter
from read_sim_data import read_sim_data
from model_muon_cherenkov_collapse import get_energy_distance_tables


CUT_TIME = 10
ENERGY_TRUE = 300
TOT_EVENTS = 1000
N_EVENTS_PER_BATCH = 100
NPROC = 16
M_STRAT = 1  # use 1 for faster processing, use 2 for slower processing with better overall performance

# Retry if BOTH z and length stay too close to the seed used on that attempt
Z_SEED_EPS = 20.0        # mm
LENGTH_SEED_EPS = 40.0   # mm
T_MIN = 0

FCN_RETRY_THRESHOLD = 1000.0
LENGTH_RETRY_THRESHOLD = 2700.0
MAX_FIT_ATTEMPTS = 4
NCALL_MIGRAD = 70000

INPUT_FILE = (
    f"/eos/user/j/jrimmer/sim_work_dir/WCSim/sim_data/mu-/"
    f"1kmu-_{ENERGY_TRUE}MeV_x0y424z0.npz"
)
OUTPUT_FILE = (
    f"/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/work_dir/estimates/with_n_fit/"
    f"estimates_{ENERGY_TRUE}MeV_batches_x0y424z0_ringOnly_nfit_NotimePen_newPhotocathode_multiseed.dict"
)

# Ring-mask mode:
#   "none" -> do not mask anything
#   "pes"  -> mask only observed PEs outside ALL_RING
#   "ts"   -> mask only observed times outside ALL_RING
#   "both" -> mask both observed PEs and observed times outside ALL_RING
#
# In this driver, missing times are represented by np.nan, so masking obs_ts means
# setting them to np.nan for PMTs outside the selected ring set.
RING_MASK_MODE = "both"


# =============================================================================
# DETECTOR CONFIGURATION
# =============================================================================
INACTIVE_SLOTS = [27, 32, 45, 74, 77, 79, 85, 91, 99]

OUTER_RING = np.array([0, 7, 19, 34, 50, 66, 82, 83, 105, 94, 95, 71, 72, 56, 40, 24, 11, 3, 18])
INNER_RING = np.array([1, 8, 35, 51, 67, 84, 69, 70, 55, 39, 23, 10, 2, 20, 36, 52, 68, 53, 54, 38, 22, 21, 37, 9])
ALL_RING = np.concatenate([OUTER_RING, INNER_RING])
# ALL_RING = np.array(np.arange(0, 106))

TRUE_PARAMS = {
    "x0": 0.0,
    "y0": 0.0,
    "z0": -1330.0,
    "cx": 0.0,
    "cy": 0.0,
    "length": 2178.0,
    "t0": 0.0,
}

# =============================================================================
# USER-DEFINED INITIAL SEED SETS
# Choose 3-5 candidate starting points here.
# The code will evaluate the NLL for each one and use them in ranked order.
# =============================================================================
INIT_PARAM_SETS = [
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
        "z0": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        "length": 500.0,
        "t0": 0.0,
    },
    {
        "x0": 0.0,
        "y0": 0.0,
        "z0": 1000.0,
        "cx": 0.0,
        "cy": 0.0,
        "length": 500.0,
        "t0": 0.0,
    },
    {
        "x0": 0.0,
        "y0": 1000.0,
        "z0": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        "length": 500.0,
        "t0": 0.0,
    },
]

PARAM_NAMES = ("x0", "y0", "z0", "cx", "cy", "length", "t0")

if not (3 <= len(INIT_PARAM_SETS) <= 5):
    raise ValueError("INIT_PARAM_SETS must contain between 3 and 5 parameter dictionaries.")

for i, seed in enumerate(INIT_PARAM_SETS):
    missing = [k for k in PARAM_NAMES if k not in seed]
    if missing:
        raise ValueError(f"Seed {i} is missing keys: {missing}")


# =============================================================================
# GLOBALS SHARED BY WORKERS
# =============================================================================
SIM_WCTE_MAPPING = None
OVERALL_DISTANCES = None
INIT_ENERGY_TABLE = None

WCD = None
PMT_MODEL = None
EMITTER_TEMPLATE = None
P_LOCATIONS = None
DIRECTION_ZS = None
RING_KEEP_MASK = None
CORR_POS = None

OBS_PES_ALL = None
OBS_TS_ALL = None


# =============================================================================
# EVENT / OBSERVABLE HELPERS
# =============================================================================
def sim_to_event(sim_data, n_mpmt_total=106, pe_scale=1.0):
    """
    Convert one simulated event into the LicketyFit Event class.
    """
    slots = []
    pmt_pos_ids = []
    charges = []
    times = []

    # Convert WCSim PMT numbering into WCTE PMT numbering.
    for i in range(len(sim_data["digi_hit_pmt"])):
        wcte_pmt = SIM_WCTE_MAPPING[int(sim_data["digi_hit_pmt"][i]) + 1]
        slots.append(int(wcte_pmt // 100))
        pmt_pos_ids.append(int(wcte_pmt % 100))
        charges.append(float(sim_data["digi_hit_charge"][i]))
        times.append(float(sim_data["digi_hit_time"][i]))

    ev = Event(0, 0, n_mpmt_total)
    ev.set_mpmt_status(list(range(n_mpmt_total)), True)

    # Build the PMT activity pattern once for this event.
    wcte_pmt_ids = []
    for i_mpmt in range(n_mpmt_total):
        if i_mpmt in INACTIVE_SLOTS:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), False)
        else:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), True)
            for i_pmt in range(19):
                wcte_pmt_ids.append(i_mpmt * 100 + i_pmt)

    # Fill the per-PMT hit times and charges.
    for s, p, q, t in zip(slots, pmt_pos_ids, charges, times):
        ev.hit_times[s][p].append(t)
        ev.hit_charges[s][p].append(q)

    return ev, np.asarray(wcte_pmt_ids, dtype=int)


def build_observables_from_event(ev, pe_scale=1.0):
    """
    Build per-PMT observed PE and first-hit-time arrays.

    Missing times are stored as np.nan so they can be masked numerically later.
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

            if len(charges) == 0:
                obs_pes.append(0.0)
                obs_ts.append(np.nan)
            else:
                obs_pes.append(float(np.sum(charges)) / pe_scale)
                obs_ts.append(float(np.min(times)))

    return np.asarray(obs_pes, dtype=np.float64), np.asarray(obs_ts, dtype=np.float64)


def apply_ring_mask_to_observables(obs_pes, obs_ts, ring_keep_mask, mode="pes"):
    """
    Apply the ALL_RING mask to the observed PEs, times, or both.
    """
    obs_pes = obs_pes.copy()
    obs_ts = obs_ts.copy()

    if mode not in {"none", "pes", "ts", "both"}:
        raise ValueError("RING_MASK_MODE must be one of: 'none', 'pes', 'ts', 'both'")

    if mode in {"pes", "both"}:
        obs_pes[~ring_keep_mask] = 0.0

    if mode in {"ts", "both"}:
        obs_ts[~ring_keep_mask] = np.nan

    return obs_pes, obs_ts


def get_main_idx_from_length(length_mm):
    """
    Map a fitted track length to the nearest pre-tabulated distance row.
    """
    idx = np.searchsorted(OVERALL_DISTANCES, float(length_mm))
    idx = np.clip(idx, 1, len(OVERALL_DISTANCES) - 1)

    left = OVERALL_DISTANCES[idx - 1]
    right = OVERALL_DISTANCES[idx]
    if (float(length_mm) - left) <= (right - float(length_mm)):
        idx -= 1

    return int(idx)


def get_t0_prior_sigma(obs_pes, obs_ts):
    n_timed = np.count_nonzero(obs_ts)
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


# =============================================================================
# LIKELIHOOD EVALUATION
# =============================================================================
def evaluate_neg_log_likelihood(obs_pes, obs_ts, emitter, x0, y0, z0, cx, cy, length, t0):
    """
    Evaluate the negative log-likelihood for one event and one parameter point.
    """
    cz2 = 1.0 - cx * cx - cy * cy
    if cz2 < 0.0:
        return 1e30

    cz = np.sqrt(cz2)
    emitter.start_coord = (x0, y0, z0)
    emitter.starting_time = t0
    emitter.direction = (cx, cy, cz)
    emitter.length = length

    main_idx = get_main_idx_from_length(length)
    init_ke = INIT_ENERGY_TABLE[main_idx][0]

    s = emitter.get_emission_points(P_LOCATIONS, init_ke)
    exp_pes, exp_ts = emitter.get_expected_pes_ts(
        WCD,
        s,
        P_LOCATIONS,
        DIRECTION_ZS,
        CORR_POS,
        obs_pes,
    )

    nll = PMT_MODEL.get_neg_log_likelihood_npe_t(exp_pes, obs_pes, exp_ts, obs_ts)
    if not np.isfinite(nll):
        return 1e30

    # Optional soft preference for times close to 0
    sigma_t0 = get_t0_prior_sigma(obs_pes, obs_ts)
    t0_penalty = abs(0.5 * (t0 / sigma_t0) ** 2)

    # return float(nll + t0_penalty)
    return float(nll)


def select_best_initial_seed(obs_pes, obs_ts, init_param_sets):
    """
    Evaluate the raw negative log-likelihood at each user-provided seed
    and return the best one, along with all seeds ranked from best to worst.

    Returns
    -------
    best_seed : dict
        The parameter dictionary with the lowest finite NLL.
    best_seed_idx : int
        Index of the best seed in init_param_sets.
    best_seed_fval : float
        NLL value at that seed.
    seed_scan_sorted : list of dict
        All tested seeds sorted by increasing FCN.
    """
    seed_scan = []

    for i, seed in enumerate(init_param_sets):
        emitter = EMITTER_TEMPLATE.copy()

        fval = evaluate_neg_log_likelihood(
            obs_pes,
            obs_ts,
            emitter,
            seed["x0"],
            seed["y0"],
            seed["z0"],
            seed["cx"],
            seed["cy"],
            seed["length"],
            seed["t0"],
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

    seed_scan_sorted = sorted(seed_scan, key=lambda x: x["fval"])

    best_seed = dict(seed_scan_sorted[0]["params"])
    best_seed_idx = int(seed_scan_sorted[0]["seed_index"])
    best_seed_fval = float(seed_scan_sorted[0]["fval"])

    return best_seed, best_seed_idx, best_seed_fval, seed_scan_sorted


def compute_true_fcn_for_event(event_index):
    """
    Evaluate the FCN at the known truth parameters for one event.
    """
    emitter = EMITTER_TEMPLATE.copy()
    return evaluate_neg_log_likelihood(
        OBS_PES_ALL[event_index],
        OBS_TS_ALL[event_index],
        emitter,
        TRUE_PARAMS["x0"],
        TRUE_PARAMS["y0"],
        TRUE_PARAMS["z0"],
        TRUE_PARAMS["cx"],
        TRUE_PARAMS["cy"],
        TRUE_PARAMS["length"],
        TRUE_PARAMS["t0"],
    )


# =============================================================================
# MINUIT HELPERS
# =============================================================================
def make_minuit_for_event(obs_pes, obs_ts, start_params):
    """
    Build a Minuit object for one event.
    """
    emitter = EMITTER_TEMPLATE.copy()

    def nll(x0, y0, z0, cx, cy, length, t0):
        return evaluate_neg_log_likelihood(
            obs_pes,
            obs_ts,
            emitter,
            x0,
            y0,
            z0,
            cx,
            cy,
            length,
            t0,
        )

    m = Minuit(nll, **start_params)

    m.limits["x0"] = (-2000, 2000)
    m.limits["y0"] = (-2000, 2000)
    m.limits["z0"] = (-2000, 2000)
    m.limits["cx"] = (-0.5, 0.5)
    m.limits["cy"] = (-0.5, 0.5)
    m.limits["length"] = (0, 3000)
    m.limits["t0"] = (-2, 2)

    m.errors["x0"] = 30.0
    m.errors["y0"] = 30.0
    m.errors["z0"] = 30.0
    m.errors["cx"] = 0.01
    m.errors["cy"] = 0.01
    m.errors["length"] = 50.0
    m.errors["t0"] = 0.1

    m.errordef = Minuit.LIKELIHOOD
    m.strategy = M_STRAT

    return m


def fit_one_event_by_index(args):
    """
    Worker function for multiprocessing.

    Flow:
      1) Evaluate the NLL at each user-provided initial seed.
      2) Rank the seeds from best to worst.
      3) Run Minuit from the best seed first.
      4) If retry is needed, use the next-best seed instead of randomizing.
    """
    event_index, init_param_sets, fcn_threshold, max_attempts, ncall = args

    obs_pes = OBS_PES_ALL[event_index]
    obs_ts = OBS_TS_ALL[event_index]

    # -------------------------------------------------------------------------
    # Pre-scan user seed set and rank all initial points for this event
    # -------------------------------------------------------------------------
    best_seed, best_seed_idx, best_seed_fval, seed_scan_sorted = select_best_initial_seed(
        obs_pes,
        obs_ts,
        init_param_sets,
    )

    best_result = None
    best_rank = (999, np.inf)

    # Only try as many attempts as there are available seeds
    n_seed_attempts = min(max_attempts, len(seed_scan_sorted))

    for attempt in range(1, n_seed_attempts + 1):
        chosen_seed_info = seed_scan_sorted[attempt - 1]
        start_params = dict(chosen_seed_info["params"])
        chosen_seed_idx = int(chosen_seed_info["seed_index"])
        chosen_seed_fcn = float(chosen_seed_info["fval"])

        m = make_minuit_for_event(obs_pes, obs_ts, start_params)

        if attempt > 1:
            m.simplex(ncall=ncall)

        m.migrad(ncall=ncall)

        current_fval = float(m.fval) if np.isfinite(m.fval) else np.inf
        current_values = m.values.to_dict()

        fitted_z0 = float(current_values["z0"])
        fitted_length = float(current_values["length"])

        length_too_large = fitted_length > LENGTH_RETRY_THRESHOLD

        # Compare to the seed used on THIS attempt
        z_near_seed = abs(fitted_z0 - float(start_params["z0"])) <= Z_SEED_EPS
        length_near_seed = abs(fitted_length - float(start_params["length"])) <= LENGTH_SEED_EPS
        seed_stuck = z_near_seed and length_near_seed

        below_t_min = current_values["t0"] < T_MIN

        result = {
            "values": current_values,
            "errors": m.errors.to_dict(),
            "fval": current_fval,
            "valid": bool(m.valid),
            "attempts": attempt,
            "length_too_large": bool(length_too_large),
            "seed_stuck": bool(seed_stuck),
            "z_near_seed": bool(z_near_seed),
            "length_near_seed": bool(length_near_seed),
            "below_t_min": bool(below_t_min),
            "chosen_seed_index": chosen_seed_idx,
            "chosen_seed_fcn": chosen_seed_fcn,
            "chosen_seed_params": dict(start_params),
            "seed_scan": seed_scan_sorted,
        }

        if (
            result["valid"]
            and np.isfinite(current_fval)
            and current_fval <= fcn_threshold
            and (not length_too_large)
            and (not seed_stuck)
            and (not below_t_min)
        ):
            current_rank = (0, current_fval)
        elif (
            result["valid"]
            and np.isfinite(current_fval)
            and (not length_too_large)
            and (not seed_stuck)
            and (not below_t_min)
        ):
            current_rank = (1, current_fval)
        elif np.isfinite(current_fval) and (not length_too_large) and (not seed_stuck) and (not below_t_min):
            current_rank = (2, current_fval)
        elif result["valid"] and np.isfinite(current_fval) and (not seed_stuck) and (not below_t_min):
            current_rank = (3, current_fval)
        elif np.isfinite(current_fval) and (not seed_stuck):
            current_rank = (4, current_fval)
        else:
            current_rank = (5, current_fval)

        if current_rank < best_rank:
            best_rank = current_rank
            best_result = result

        # Stop retrying only if all conditions are satisfied
        if (
            m.valid
            and np.isfinite(current_fval)
            and current_fval <= fcn_threshold
            and (not length_too_large)
            and (not seed_stuck)
            and (not below_t_min)
        ):
            break

    return best_result


def run_batch(event_indices, init_param_sets, nproc, fcn_threshold, max_attempts, ncall):
    """
    Run one batch of events in parallel.
    """
    args = [(idx, init_param_sets, fcn_threshold, max_attempts, ncall) for idx in event_indices]

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    with ctx.Pool(processes=nproc) as pool:
        return pool.map(fit_one_event_by_index, args)


# =============================================================================
# MAIN DRIVER
# =============================================================================
def main():
    global SIM_WCTE_MAPPING, OVERALL_DISTANCES, INIT_ENERGY_TABLE
    global WCD, PMT_MODEL, EMITTER_TEMPLATE, P_LOCATIONS, DIRECTION_ZS, RING_KEEP_MASK, CORR_POS
    global OBS_PES_ALL, OBS_TS_ALL

    data_raw = read_sim_data(INPUT_FILE)

    # Load the pre-tabulated muon energy-vs-distance information used by the model.
    OVERALL_DISTANCES, INIT_ENERGY_TABLE, _distance_rows = get_energy_distance_tables()

    wcte_mapping = np.loadtxt(
        "../tables/wcsim_wcte_mapping.txt"
    )

    # Build WCSim -> WCTE PMT mapping once.
    SIM_WCTE_MAPPING = {}
    for i in range(len(wcte_mapping)):
        SIM_WCTE_MAPPING[int(wcte_mapping[i][0])] = int(
            wcte_mapping[i][1] * 100 + wcte_mapping[i][2] - 1
        )

    hall = Device.open_file(geometry_path + "examples/wcte_bldg157.geo")
    WCD = hall.wcds[0]

    emitter_model = Emitter(
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.96,
        500.0,
        18.0,
    )
    PMT_MODEL = PMT(1.0, 0.3, 1.0, 40.0, 0.2, 0.0)
    EMITTER_TEMPLATE = emitter_model.copy()
    CORR_POS = None

    print("Building event observables...")

    obs_pes_all = []
    obs_ts_all = []

    for evt_num in range(TOT_EVENTS):
        hit_times = np.asarray(data_raw["digi_hit_time"][evt_num], dtype=np.float64)
        hit_pmts = np.asarray(data_raw["digi_hit_pmt"][evt_num], dtype=int)
        hit_charges = np.asarray(data_raw["digi_hit_charge"][evt_num], dtype=np.float64)

        # Keep only prompt hits, as in the original front-end script.
        keep = (hit_times > 0.0) & (hit_times < CUT_TIME)

        sim_data = {
            "digi_hit_pmt": hit_pmts[keep],
            "digi_hit_time": hit_times[keep],
            "digi_hit_charge": hit_charges[keep],
        }

        ev, pmt_ids = sim_to_event(sim_data, n_mpmt_total=106, pe_scale=1.0)

        if P_LOCATIONS is None or DIRECTION_ZS is None:
            # The PMT ordering is fixed, so the geometry and ring mask only need
            # to be built once.
            P_LOCATIONS, DIRECTION_ZS = EMITTER_TEMPLATE.get_pmt_placements(ev, WCD, "design")
            mpmt_ids = pmt_ids // 100
            RING_KEEP_MASK = np.isin(mpmt_ids, ALL_RING)

        obs_pes, obs_ts = build_observables_from_event(ev, pe_scale=1.0)
        obs_pes, obs_ts = apply_ring_mask_to_observables(
            obs_pes,
            obs_ts,
            RING_KEEP_MASK,
            mode=RING_MASK_MODE,
        )

        obs_pes_all.append(obs_pes)
        obs_ts_all.append(obs_ts)

    OBS_PES_ALL = obs_pes_all
    OBS_TS_ALL = obs_ts_all

    print("Computing truth FCNs...")
    true_fcn_all = [compute_true_fcn_for_event(i) for i in range(TOT_EVENTS)]

    est_dict = {
        "minimum_found": [],
        "x": [],
        "y": [],
        "z": [],
        "length": [],
        "t": [],
        "est_fcn": [],
        "true_fcn": [],
        "cx": [],
        "cy": [],
        "n_attempts": [],
        "chosen_seed_idx": [],
        "chosen_seed_fcn": [],
        "chosen_seed_params": [],
        "seed_scan": [],
    }

    n_batches = TOT_EVENTS // N_EVENTS_PER_BATCH

    for batch_idx in range(n_batches):
        batch_start = batch_idx * N_EVENTS_PER_BATCH
        batch_end = batch_start + N_EVENTS_PER_BATCH
        event_indices = list(range(batch_start, batch_end))

        print(f"Starting event number {batch_start}")

        results = run_batch(
            event_indices=event_indices,
            init_param_sets=INIT_PARAM_SETS,
            nproc=NPROC,
            fcn_threshold=FCN_RETRY_THRESHOLD,
            max_attempts=MAX_FIT_ATTEMPTS,
            ncall=NCALL_MIGRAD,
        )

        for local_i, result in enumerate(results):
            event_index = event_indices[local_i]

            est_dict["minimum_found"].append(int(result["valid"]))
            est_dict["x"].append(result["values"]["x0"])
            est_dict["y"].append(result["values"]["y0"])
            est_dict["z"].append(result["values"]["z0"])
            est_dict["length"].append(result["values"]["length"])
            est_dict["t"].append(result["values"]["t0"])
            est_dict["cx"].append(result["values"]["cx"])
            est_dict["cy"].append(result["values"]["cy"])
            est_dict["est_fcn"].append(result["fval"])
            est_dict["true_fcn"].append(true_fcn_all[event_index])
            est_dict["n_attempts"].append(result["attempts"])
            est_dict["chosen_seed_idx"].append(result["chosen_seed_index"])
            est_dict["chosen_seed_fcn"].append(result["chosen_seed_fcn"])
            est_dict["chosen_seed_params"].append(result["chosen_seed_params"])
            est_dict["seed_scan"].append(result["seed_scan"])

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(est_dict, f)

    print("Done.")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()