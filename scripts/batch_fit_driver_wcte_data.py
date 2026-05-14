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

geometry_path = "/eos/user/j/jrimmer/Geometry"

sys.path.insert(0, "../LicketyFit")
sys.path.insert(0, "../scripts")
sys.path.insert(0, "../tab;es")
sys.path.insert(0, geometry_path)
sys.path.insert(0, "../")
sys.path.insert(0, "../../")

from Geometry.Device import Device
from LicketyFit.Event import Event
from LicketyFit.PMT import PMT
from LicketyFit.Emitter import Emitter
from read_sim_data import read_sim_data
from model_muon_cherenkov_collapse import get_energy_distance_tables
from get_mu_events import *



N_EVENTS_PER_BATCH = 100
NPROC = 16
M_STRAT = 1  

# Retry if BOTH z and length stay too close to the seed used on that attempt
Z_SEED_EPS = 20.0        # mm
LENGTH_SEED_EPS = 40.0   # mm
T_MIN = -8

FCN_RETRY_THRESHOLD = 1100
LENGTH_RETRY_THRESHOLD = 2700.0
MAX_FIT_ATTEMPTS = 4
NCALL_MIGRAD = 70000

RUN = 1580
BEAM_P = 280

N_EVENTS = 10000

# =============================================================================
# LIKELIHOOD TERM TOGGLES
# =============================================================================
# Choose which pieces of the likelihood to include:
#
#   charge only: USE_CHARGE_LIKELIHOOD = True,  USE_TIMING_LIKELIHOOD = False
#   timing only: USE_CHARGE_LIKELIHOOD = False, USE_TIMING_LIKELIHOOD = True
#   charge+time: USE_CHARGE_LIKELIHOOD = True, USE_TIMING_LIKELIHOOD = True
#
# Notes:
#   - If timing is disabled, t0 is fixed in Minuit because charge-only fits do
#     not constrain the event start time.
#   - If timing is enabled, t0 is left free.
#   - USE_T0_PRIOR is optional and is only applied when timing is enabled.
USE_CHARGE_LIKELIHOOD = True
USE_TIMING_LIKELIHOOD = True
USE_T0_PRIOR = False

if (not USE_CHARGE_LIKELIHOOD) and (not USE_TIMING_LIKELIHOOD):
    raise ValueError(
        "At least one likelihood term must be enabled: "
        "USE_CHARGE_LIKELIHOOD or USE_TIMING_LIKELIHOOD."
    )

if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
    LIKELIHOOD_MODE = "charge_time"
elif USE_CHARGE_LIKELIHOOD:
    LIKELIHOOD_MODE = "charge_only"
else:
    LIKELIHOOD_MODE = "timing_only"





OUTPUT_FILE = (
    f"/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/work_dir/wcte_estimates/"
    f"estimates_run{RUN}_{BEAM_P}p_mu-_mpmtEff_{LIKELIHOOD_MODE}.dict"
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

# Get the config file to mask out PMTs that are not in operation

fname = '/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/'+str(RUN)+'/WCTE_merged_production_R'+str(RUN)+'.root'
with uproot.open(fname) as f:
    
    
    t_c = f['Configuration']
    arr_config = t_c.arrays(library="ak")

GOOD_WCTE_PMTS = np.asarray(arr_config["good_wcte_pmts"][0], dtype=int)
INACTIVE_SLOTS = [27, 32, 45, 74, 77, 79, 85, 91, 99,9, 67]


GOOD_WCTE_PMTS_SET = set(np.asarray(GOOD_WCTE_PMTS, dtype=int).tolist())
INACTIVE_SLOTS_SET = set(int(s) for s in INACTIVE_SLOTS)

# =============================================================================
# DETECTOR CONFIGURATION
# =============================================================================


OUTER_RING = np.array([0, 7, 19, 34, 50, 66, 82, 83, 105, 94, 95, 71, 72, 56, 40, 24, 11, 3, 18])
INNER_RING = np.array([1, 8, 35, 51, 67, 84, 69, 70, 55, 39, 23, 10, 2, 20, 36, 52, 68, 53, 54, 38, 22, 21, 37, 9])
OUTSIDE_RING = np.array([12,13,4,5,6,17,33,49,65,81,82,104,93,86,87,72,57,41,25])

ALL_RING = np.concatenate([OUTER_RING, INNER_RING])
ALL_RING = np.concatenate([ALL_RING, OUTSIDE_RING])
ALL_RING = np.array(np.arange(0, 106))

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
        "t0": 0,
    },
    {
        "x0": 0.0,
        "y0": 0.0,
        "z0": -1300,
        "cx": 0.0,
        "cy": 0.0,
        "length": 500.0,
        "t0": 0,
    },
    {
        "x0": 0.0,
        "y0": 0.0,
        "z0": -1500,
        "cx": 0.0,
        "cy": 0.0,
        "length": 700.0,
        "t0": 0,
    },
    {
        "x0": 0.0,
        "y0": 0,
        "z0": -1000,
        "cx": 0.0,
        "cy": 0.0,
        "length": 1000.0,
        "t0": 0,
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

# Get the mPMT-by-mPMT dictionary info
with open('../tables/other_mpmt_info_v2.dict', 'rb') as f:
    mpmt_info = pickle.load(f)
    
# Get mPMT slot by slot efficiency correction
rel_mpmt_eff_path = "../tables/rel_mpmt_eff.dict"
with open(rel_mpmt_eff_path, 'rb') as f:
        
    rel_mpmt_eff = pickle.load(f)
        
tri_exsitu = rel_mpmt_eff['tri_exsitu']
tri_insitu = rel_mpmt_eff['tri_insitu']

wut_insitu = rel_mpmt_eff['wut_insitu']
wut_exsitu = rel_mpmt_eff['wut_exsitu']


# Write a helper function that returns the PMT-by-PMT correction that is needed (based on mPMT type)

def get_mpmt_slot_type(mpmt_slots):
    
    slot_type = []
    
    for slot in mpmt_slots:
        slot = int(slot)
        
        try:
        
            if mpmt_info[slot]['mpmt_site'] == 'TRI':
                if mpmt_info[slot]['mpmt_type'] == 'In-situ':
                    slot_type.append('tri_insitu')
                else:
                    slot_type.append('tri_exsitu')

            else:
                if mpmt_info[slot]['mpmt_type'] == 'In-situ':
                    slot_type.append('wut_insitu')
                else:
                    slot_type.append('wut_exsitu')
        except:
            slot_type.append('empty')
                
    return slot_type

# =============================================================================
# EVENT / OBSERVABLE HELPERS
# =============================================================================

def sim_to_event(
    sim_data,
    WCD,
    n_mpmt_total=106,
    pe_scale=143,
    shift_times=True,
    n_earliest_for_t0=10,
):
    """
    Convert one data event into the LicketyFit Event class.

    Critical design:
      - PMT statuses are fixed by the detector/run configuration.
      - PMT statuses are NOT determined by which PMTs happened to have hits
        in this individual event.
      - Therefore obs_pes, obs_ts, P_LOCATIONS, DIRECTION_ZS, MPMT_SLOTS,
        and RING_KEEP_MASK all have consistent lengths/orderings.
    """
    vw = 223.0598645833333  # mm/ns

    ev = Event(0, 0, n_mpmt_total)

    # Start with everything off.
    ev.set_mpmt_status(list(range(n_mpmt_total)), False)

    active_wcte_pmt_ids = []

    # ------------------------------------------------------------------
    # Fixed detector/run PMT status pattern.
    # This is independent of event hits.
    # ------------------------------------------------------------------
    for slot in range(n_mpmt_total):

        if slot in INACTIVE_SLOTS_SET:
            continue

        slot_has_good_pmt = False

        for pmt_pos_id in range(ev.npmt_per_mpmt):
            wcte_pmt = int(slot * 100 + pmt_pos_id)

            if wcte_pmt in GOOD_WCTE_PMTS_SET:
                ev.set_pmt_status(slot, [pmt_pos_id], True)
                slot_has_good_pmt = True
                active_wcte_pmt_ids.append(wcte_pmt)

        if slot_has_good_pmt:
            ev.set_mpmt_status([slot], True)

    # ------------------------------------------------------------------
    # Fill hits only if they belong to configured-good PMTs.
    # ------------------------------------------------------------------
    for i in range(len(sim_data[:, 0])):

        wcte_pmt = int(sim_data[i, 0])

        slot = int(wcte_pmt // 100)
        pmt_pos_id = int(wcte_pmt % 100)

        if slot < 0 or slot >= ev.n_mpmt:
            continue

        if pmt_pos_id < 0 or pmt_pos_id >= ev.npmt_per_mpmt:
            continue

        if not ev.mpmt_status[slot]:
            continue

        if not ev.pmt_status[slot][pmt_pos_id]:
            continue

        ev.hit_charges[slot][pmt_pos_id].append(float(sim_data[i, 1]))
        ev.hit_times[slot][pmt_pos_id].append(float(sim_data[i, 2]))

    # ------------------------------------------------------------------
    # TIME SHIFTING
    # ------------------------------------------------------------------
    if shift_times:

        bp_loc = np.array([0.0, 0.0, -1350.0])
        early_hits = []

        for i_mpmt in range(ev.n_mpmt):

            if not ev.mpmt_status[i_mpmt]:
                continue

            for i_pmt in range(ev.npmt_per_mpmt):

                if not ev.pmt_status[i_mpmt][i_pmt]:
                    continue

                if len(ev.hit_times[i_mpmt][i_pmt]) == 0:
                    continue

                pmt_loc = WCD.mpmts[i_mpmt].pmts[i_pmt].get_placement("est")["location"]
                r = np.linalg.norm(pmt_loc - bp_loc)

                for t in ev.hit_times[i_mpmt][i_pmt]:
                    t0_est = float(t) - r / vw

                    early_hits.append({
                        "time": float(t),
                        "t0_est": t0_est,
                        "r": r,
                        "i_mpmt": i_mpmt,
                        "i_pmt": i_pmt,
                    })

        if len(early_hits) > 0:

            early_hits = sorted(early_hits, key=lambda x: x["time"])

            n_use = min(n_earliest_for_t0, len(early_hits))
            earliest_hits = early_hits[:n_use]

            time_offset = np.median([hit["t0_est"] for hit in earliest_hits])

            for i_mpmt in range(ev.n_mpmt):
                for i_pmt in range(ev.npmt_per_mpmt):
                    ev.hit_times[i_mpmt][i_pmt] = [
                        t - time_offset for t in ev.hit_times[i_mpmt][i_pmt]
                    ]

            ev.global_time_offset = time_offset

    return ev, np.asarray(active_wcte_pmt_ids, dtype=int)



def build_observables_from_event(ev, pe_scale=143):
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
            
            q = np.asarray(charges, dtype=np.float64)
            t = np.asarray(times, dtype=np.float64)

            if len(charges) == 0:
                obs_pes.append(0.0)
                obs_ts.append(np.nan)
            else:
                #obs_pes.append(float(np.sum(charges)) / pe_scale)
                #obs_ts.append(float(np.min(times)))
              

                obs_pes.append(float(np.sum(q)) / pe_scale)
                obs_ts.append(float(np.sum(q * t) / np.sum(q)))

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
    # Count only PMTs with actual finite hit times. np.count_nonzero(obs_ts)
    # counts np.nan as nonzero, so use np.isfinite instead.
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


# =============================================================================
# LIKELIHOOD EVALUATION
# =============================================================================
def get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts):
    """
    Timing-only negative log likelihood.

    This matches the timing part of PMT.get_neg_log_likelihood_npe_t:

        0.5 * ((t_obs - t_exp) / (sigma_t / sqrt(obs_pe)))**2

    evaluated only for PMTs with:
      - expected PE > 0,
      - observed PE > 0,
      - finite observed time,
      - finite expected time.

    PMTs with no observed hit time do not contribute to the timing likelihood.
    """
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

    sigma_t = PMT_MODEL.single_pe_time_std / np.sqrt(obs_pes[mask])
    dt = (obs_ts[mask] - exp_ts[mask]) / sigma_t

    return float(0.5 * np.sum(dt * dt))


def evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts):
    """
    Evaluate the selected likelihood terms according to the top-of-file toggles.
    """
    if USE_CHARGE_LIKELIHOOD and USE_TIMING_LIKELIHOOD:
        # Fast compiled charge+time path already implemented in PMT.
        return PMT_MODEL.get_neg_log_likelihood_npe_t(
            exp_pes,
            obs_pes,
            exp_ts,
            obs_ts,
        )

    if USE_CHARGE_LIKELIHOOD:
        return PMT_MODEL.get_neg_log_likelihood_npe(exp_pes, obs_pes)

    # Timing-only.
    return get_timing_only_nll(exp_pes, obs_pes, exp_ts, obs_ts)


def evaluate_neg_log_likelihood(obs_pes, obs_ts, emitter, mpmt_types, x0, y0, z0, cx, cy, length, t0):
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
    
#     emitter.length = length

#     main_idx = get_main_idx_from_length(length)
#     init_ke = INIT_ENERGY_TABLE[main_idx][0]

    init_ke = emitter.refresh_kinematics_from_length(length)

    s = emitter.get_emission_points(P_LOCATIONS, init_ke)
    exp_pes, exp_ts = emitter.get_expected_pes_ts(
        WCD,
        s,
        P_LOCATIONS,
        DIRECTION_ZS,
        mpmt_types,
        obs_pes,
    )

    nll = evaluate_pmt_nll(exp_pes, obs_pes, exp_ts, obs_ts)
    if not np.isfinite(nll):
        return 1e30

    # Optional soft preference for t0 close to 0.  Only apply this when timing
    # is enabled; in charge-only mode t0 is fixed and has no charge meaning.
    if USE_TIMING_LIKELIHOOD and USE_T0_PRIOR:
        sigma_t0 = get_t0_prior_sigma(obs_pes, obs_ts)
        t0_penalty = abs(0.5 * (t0 / sigma_t0) ** 2)
        nll += t0_penalty

    return float(nll)


def select_best_initial_seed(obs_pes, obs_ts, init_param_sets, mpmt_types):
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
            mpmt_types,
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
    mpmt_types = get_mpmt_slot_type(MPMT_SLOTS_ALL[event_index])
    emitter = EMITTER_TEMPLATE.copy()
    return evaluate_neg_log_likelihood(
        OBS_PES_ALL[event_index],
        OBS_TS_ALL[event_index],
        emitter,
        mpmt_types,
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
def make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types):
    """
    Build a Minuit object for one event.
    """
    emitter = EMITTER_TEMPLATE.copy()

    def nll(x0, y0, z0, cx, cy, length, t0):
        return evaluate_neg_log_likelihood(
            obs_pes,
            obs_ts,
            emitter,
            mpmt_types,
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
    m.limits["length"] = (0, 4000)
    m.limits["t0"] = (-8, 8)

    # Charge-only likelihood does not depend on t0, so leave it fixed to avoid
    # a flat Minuit direction.  Timing-only and charge+time fits keep t0 free.
    if not USE_TIMING_LIKELIHOOD:
        m.fixed["t0"] = True

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
    mpmt_slots = MPMT_SLOTS_ALL[event_index]
    
    mpmt_types = get_mpmt_slot_type(mpmt_slots)

    # -------------------------------------------------------------------------
    # Pre-scan user seed set and rank all initial points for this event
    # -------------------------------------------------------------------------
    best_seed, best_seed_idx, best_seed_fval, seed_scan_sorted = select_best_initial_seed(
        obs_pes,
        obs_ts,
        init_param_sets,
        mpmt_types
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

        m = make_minuit_for_event(obs_pes, obs_ts, start_params, mpmt_types)

        #if attempt > 1:
        #    m.simplex(ncall=ncall)
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

        # Only use the t0 quality cut when timing participates in the fit.
        below_t_min = USE_TIMING_LIKELIHOOD and (current_values["t0"] < T_MIN)

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
            #result["valid"]
            np.isfinite(current_fval)
            and current_fval <= fcn_threshold
            and (not length_too_large)
            and (not seed_stuck)
            and (not below_t_min)
        ):
            current_rank = (0, current_fval)
        elif (
            #result["valid"]
            np.isfinite(current_fval)
            and (not length_too_large)
            and (not seed_stuck)
            and (not below_t_min)
        ):
            current_rank = (1, current_fval)
        elif np.isfinite(current_fval) and (not length_too_large) and (not seed_stuck) and (not below_t_min):
            current_rank = (2, current_fval)
        elif np.isfinite(current_fval) and (not seed_stuck) and (not below_t_min):
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
            #m.valid
            np.isfinite(current_fval)
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
    global OBS_PES_ALL, OBS_TS_ALL, MPMT_SLOTS_ALL

    print("Likelihood mode:", LIKELIHOOD_MODE)
    

    #data_raw = read_sim_data(INPUT_FILE)

    # Load the pre-tabulated muon energy-vs-distance information used by the model.
    OVERALL_DISTANCES, INIT_ENERGY_TABLE, _distance_rows = get_energy_distance_tables()

    wcte_mapping = np.loadtxt(
        "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/wcsim_wcte_mapping.txt"
    )

    # Build WCSim -> WCTE PMT mapping once.
#     SIM_WCTE_MAPPING = {}
#     for i in range(len(wcte_mapping)):
#         SIM_WCTE_MAPPING[int(wcte_mapping[i][0])] = int(
#             wcte_mapping[i][1] * 100 + wcte_mapping[i][2] - 1
#         )

    hall = Device.open_file(geometry_path + "/examples/wcte_bldg157.geo")
    WCD = hall.wcds[0]

    emitter_model = Emitter(
        0.0,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.96,
        500.0,
        18.0,
    )
    
#     DELTA_E_ANGULAR_PDF_PATH = (
#         "/eos/user/j/jrimmer/SWAN_projects/beam/LicketyFit2/tables/"
#         "delta_e_angular_pdf_table.npz"
#     )

#     emitter_model.load_delta_e_angular_pdf_table(DELTA_E_ANGULAR_PDF_PATH)
    
    #self.load_delta_e_angular_pdf_table("delta_e_angular_pdf_table.npz")
    PMT_MODEL = PMT(1.0, 0.3, 1.0, 40.0, 0.2, 0.0)
    EMITTER_TEMPLATE = emitter_model.copy()
    CORR_POS = None

    print("Building event observables...")

    obs_pes_all = []
    obs_ts_all = []
    mpmt_slots_all = []
    
    events = get_mu_events(RUN,N_EVENTS)
    
    TOT_EVENTS = len(events)
    print('Total Events to Fit:', TOT_EVENTS)

    
    for i in range(len(events)):
        
        event = events[i]
        time_hist = np.histogram(event[:,2],bins = np.arange(0,4000))
        max_idx = np.argmax(time_hist[0])
        MIN_TIME = time_hist[1][max_idx-20]
        CUT_TIME = time_hist[1][max_idx+5]

        time_mask = (event[:,2] > MIN_TIME) & (event[:,2] < CUT_TIME)

        event = event[time_mask]

        ev, pmt_ids = sim_to_event(event, WCD, n_mpmt_total=106, pe_scale=143)
        
        if P_LOCATIONS is None or DIRECTION_ZS is None:
            P_LOCATIONS, DIRECTION_ZS, MPMT_SLOTS = EMITTER_TEMPLATE.get_pmt_placements(ev, WCD, "est")

            MPMT_SLOTS = np.asarray(MPMT_SLOTS, dtype=int)
            RING_KEEP_MASK = np.isin(MPMT_SLOTS, ALL_RING)

#         if P_LOCATIONS is None or DIRECTION_ZS is None:
#             # The PMT ordering is fixed, so the geometry and ring mask only need
#             # to be built once.
#             P_LOCATIONS, DIRECTION_ZS, MPMT_SLOTS = EMITTER_TEMPLATE.get_pmt_placements(ev, WCD, "est")
#             mpmt_ids = pmt_ids // 100
#             RING_KEEP_MASK = np.isin(MPMT_SLOTS, ALL_RING)
            
            

        obs_pes, obs_ts = build_observables_from_event(ev, pe_scale=143)
        
        obs_pes, obs_ts = apply_ring_mask_to_observables(
            obs_pes,
            obs_ts,
            RING_KEEP_MASK,
            mode=RING_MASK_MODE,
        )

        obs_pes_all.append(obs_pes)
        obs_ts_all.append(obs_ts)
        mpmt_slots_all.append(MPMT_SLOTS)
        

    OBS_PES_ALL = obs_pes_all
    OBS_TS_ALL = obs_ts_all
    MPMT_SLOTS_ALL = mpmt_slots_all

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

    N_EVENTS_PER_BATCH = 100
    
    if TOT_EVENTS<=N_EVENTS_PER_BATCH:
        N_EVENTS_PER_BATCH = TOT_EVENTS
        
        
    n_batches = TOT_EVENTS // N_EVENTS_PER_BATCH
    
        
    print('n_batches',n_batches)

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