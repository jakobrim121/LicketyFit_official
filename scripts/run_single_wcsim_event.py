# This file gives an output for a single WCSim event
geo_path = '/eos/user/j/jrimmer/Geometry'
import sys
sys.path.insert(0, geo_path)
sys.path.insert(0, "../")
sys.path.insert(0, "../LicketyFit")
from Geometry.Device import Device

from LicketyFit.Event import *
from LicketyFit.PMT import *
from LicketyFit.MarkovChain import *
from LicketyFit.Emitter import *
from minuit_fit import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import uproot, json, awkward as ak
from read_sim_data import *


wcte_mapping = np.loadtxt('../tables/wcsim_wcte_mapping.txt')

# wcsim uses positions 1-19, so have to subtract 1 in the mapping...
sim_wcte_mapping = {}
for i in range(len(wcte_mapping)):
    sim_wcte_mapping[int(wcte_mapping[i][0])] = int(wcte_mapping[i][1]*100 + wcte_mapping[i][2] - 1)

    
E_vs_dist = np.load('../tables/E_vs_dist.npy', allow_pickle=True) # need to convert this from cm to mm later
overall_distances = np.load('../tables/overall_distances.npy')*10 # convert to mm
init_energy = np.array([a[:, 1] for a in E_vs_dist], dtype=object)
overall_energies = np.array([a[0] for a in init_energy])
dist_travelled = np.array([a[:, 0] for a in E_vs_dist], dtype=object)*10 #change to mm    


hall = Device.open_file(geo_path+'/examples/wcte_bldg157.geo')
wcte = hall.wcds[0]
n_mpmt_geom = len(wcte.mpmts)
    
inactive_slots = [27,32,45,74,77,79,85,91,99]
def sim_to_Event(sim_data, n_mpmt_total=None, pe_scale=1.0, shift_times=False):
    """
    Convert one raw JSON event into Dean's Event class.

    Parameters
    ----------
    raw : dict
        One decoded JSON entry, containing:
            hit_mpmt_slot_ids
            hit_pmt_position_ids
            hit_pmt_charges
            hit_pmt_times
            run_id
            event_number
    n_mpmt_total : int or None
        If None, inferred from max slot ID + 1.  
        If geometry index space is full (0–105), set this to 106.
    pe_scale : float
        ADC counts per 1 PE (for later use)
    shift_times : bool
        Whether to subtract earliest hit time to make event times relative.

    Returns
    -------
    Event object
    """

    # ---------
    # Determine total number of mPMTs
    # ---------
    slots = []
    pmt_pos_ids = []
    charges = []
    times = []
    for i in range(len(sim_data['digi_hit_pmt'])):
        
        sim_pmt = sim_data['digi_hit_pmt'][i]
        wcte_pmt = sim_wcte_mapping[sim_data['digi_hit_pmt'][i]+1]
        slots.append(int(wcte_pmt/100))
        pmt_pos_ids.append(wcte_pmt%100)
        charges.append(sim_data['digi_hit_charge'][i])
        times.append(sim_data['digi_hit_time'][i])
        
    #slots = raw["hit_mpmt_slot_ids"]
    if n_mpmt_total is None:
        n_mpmt = int(np.max(slots)) + 1
    else:
        n_mpmt = n_mpmt_total

    # Create event
    ev = Event(0, 0, n_mpmt)

    # Activate all PMTs
    wcte_pmt_ids = []
    ev.set_mpmt_status(list(range(n_mpmt)), True)
    for i_mpmt in range(n_mpmt):
        if i_mpmt in inactive_slots:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), False)
        else:
            ev.set_pmt_status(i_mpmt, list(range(ev.npmt_per_mpmt)), True)
            for j in range(19):
                wcte_pmt_ids.append(i_mpmt*100+j)

    # Fill hits
    for s, p, q, t in zip(slots,
                          pmt_pos_ids,
                          charges,
                          times):
        ev.hit_times[s][p].append(float(t))
        ev.hit_charges[s][p].append(float(q))

    # -------------
    # TIME SHIFTING (this is new)
    # -------------
    if shift_times:
        min_time = float('inf')
        for i_mpmt in range(ev.n_mpmt):
            for i_pmt in range(ev.npmt_per_mpmt):
                if ev.hit_times[i_mpmt][i_pmt]:
                    tmin = min(ev.hit_times[i_mpmt][i_pmt])
                    if tmin < min_time:
                        min_time = tmin

        # Shift all hit times so earliest hit = 0 ns
#         if min_time < float('inf'):
#             for i_mpmt in range(ev.n_mpmt):
#                 for i_pmt in range(ev.npmt_per_mpmt):
#                     ev.hit_times[i_mpmt][i_pmt] = [
#                         t - min_time for t in ev.hit_times[i_mpmt][i_pmt]
#                     ]

        # Store original offset for bookkeeping if needed
        #ev.global_time_offset = min_time

    return ev, wcte_pmt_ids



def build_observables_from_event(ev, pe_scale=1.0):
    """
    Build obs_pes and obs_ts from a real Event.

    obs_pes: float npe per PMT (from total charge / pe_scale)
    obs_ts:  first hit time per PMT (or None if no hit)

    Returns:
        obs_pes, obs_ts, pmt_indices
        - obs_pes: np.array of len N_pmts_used
        - obs_ts:  np.array of len N_pmts_used (dtype=object so None allowed)
        - pmt_indices: list of (i_mpmt, i_pmt) for each entry
    """
    obs_pes = []
    obs_ts = []
    pmt_indices = []

    for i_mpmt in range(ev.n_mpmt):
        if not ev.mpmt_status[i_mpmt]:
            continue
        for i_pmt in range(ev.npmt_per_mpmt):
            if not ev.pmt_status[i_mpmt][i_pmt]:
                continue

            charges = ev.hit_charges[i_mpmt][i_pmt]
            times   = ev.hit_times[i_mpmt][i_pmt]

            if len(charges) == 0:
                # no hit in this PMT
                obs_pes.append(0.0)
                obs_ts.append(None)
                pmt_indices.append((i_mpmt, i_pmt))
                continue

            # crude PE estimate: total charge / single-PE scale
            total_q = np.sum(charges)
            npe = total_q / pe_scale

            # take earliest hit as "time" of this PMT
            t_first = float(np.min(times))

            obs_pes.append(float(npe))
            obs_ts.append(t_first)
            pmt_indices.append((i_mpmt, i_pmt))

    # use dtype=object so that None is allowed (if you ever keep Nones)
    return np.array(obs_pes, dtype=float), np.array(obs_ts, dtype=object), pmt_indices


single_pe_amp_mean = 60.
single_pe_amp_std = 20.
single_pe_time_std = 1.
separation_time = 40.
amp_threshold = 20.
noise_rate = 0.
pmt = PMT(single_pe_amp_mean, single_pe_amp_std, single_pe_time_std, separation_time, amp_threshold, noise_rate)

pmt_model = PMT(single_pe_amp_mean,
                single_pe_amp_std,
                single_pe_time_std,
                separation_time,
                amp_threshold,
                noise_rate)

length_guess = 1165
starting_time_guess = 0
start_coord_guess   = (0.0, 0, -1530)   # mm
direction_guess     = (0.0, 0.0, 1.0)   # along +z
beta_guess          = 0.96
length_guess        = length_guess          # mm
intensity_guess     = 10.0               # PE at 1 m, normal incidence

emitter_model = Emitter(starting_time_guess,
                        start_coord_guess,
                        direction_guess,
                        beta_guess,
                        length_guess,
                        intensity_guess)

emitter_copy = emitter_model.copy()




def run_minuit(file_path, evt_num, cut_time = 17):
    
    data_raw = read_sim_data(file_path)
    
    data = {'digi_hit_pmt':[],'digi_hit_time':[],'digi_hit_charge':[]}
    for i in range(len(data_raw['digi_hit_time'][evt_num])):
        if 0<data_raw['digi_hit_time'][evt_num][i]<cut_time:
            data['digi_hit_time'].append(data_raw['digi_hit_time'][evt_num][i])
            data['digi_hit_pmt'].append(data_raw['digi_hit_pmt'][evt_num][i])
            data['digi_hit_charge'].append(data_raw['digi_hit_charge'][evt_num][i])
            
            
            
    ev, pmt_ids = sim_to_Event(data, n_mpmt_total=None, pe_scale=1.0, shift_times=False)
    
    p_locations, direction_zs = emitter_copy.get_pmt_placements(ev, wcte, 'design')
    
    obs_pes, obs_ts, pmt_indices = build_observables_from_event(ev, pe_scale=1.0)
    
    outer_ring = np.array([0,7,19,34,50,66,82,83,105,94,95,71,72,56,40,24,11,3,18])
    inner_ring = np.array([1,8,35,51,67,84,69,70,55,39,23,10,2,20,36,52,68,53,54,38,22,21,37,9])
    close_to_ring = np.array([85,86,93,104,81,65,49,33,18,6,5,4,12,25,41,57])
    all_ring = np.concatenate([outer_ring,inner_ring])

    mPMT_id = []
    PMT_pos = []

    for i in range(len(pmt_ids)):
        mPMT_id.append(int(pmt_ids[i]/100))
        PMT_pos.append(int(pmt_ids[i]%100))

    obs_pes_new = []

    for i in range(len(obs_pes)):
        if mPMT_id[i] in all_ring or mPMT_id[i] in close_to_ring:
        #if mPMT_id[i] not in close_to_ring:
            obs_pes_new.append(obs_pes[i])
        else:
            obs_pes_new.append(0)

    obs_pes = np.array(obs_pes_new)
    
    
    corr_pos = None
    def get_neg_log_likelihood_npe_t(x0, y0, z0, cx, cy, length, t0):
        # build direction unit vector
        cz = np.sqrt(1.0 - cx**2 - cy**2)
        direction = (cx, cy, cz)
        start_coord = (x0, y0, z0)

        #emitter_copy.intensity = intensity
        # update emitter model
        emitter_copy.start_coord   = start_coord
        emitter_copy.starting_time = t0
        emitter_copy.direction     = direction
        emitter_copy.length        = length

        main_idx = np.searchsorted(overall_distances, length)
        main_idx = np.clip(main_idx, 1, len(overall_distances) - 1)
        left = overall_distances[main_idx - 1]
        right = overall_distances[main_idx]
        main_idx -= (length_guess - left) <= (right - length)


        init_KE = init_energy[main_idx][0]

        # expected number of PE and times at each PMT
        ss = emitter_copy.get_emission_points(p_locations,init_KE)

        exp_pes, exp_ts = emitter_copy.get_expected_pes_ts(wcte, ss, p_locations, direction_zs,corr_pos,obs_pes)

        # compare to observed data using the PMT-likelihood model
        neg_ll = pmt_model.get_neg_log_likelihood_npe_t(exp_pes, obs_pes, exp_ts, obs_ts)


        return float(neg_ll)

    
    

    x0_init = 0
    y0_init = 0
    z0_init = -500
    cx_init = 0.0
    cy_init = 0.0
    length_init = 500
    t0_init = 0

    m = Minuit(
            get_neg_log_likelihood_npe_t,
            x0=x0_init,
            y0=y0_init,
            z0=z0_init,
            cx=cx_init,
            cy=cy_init,
            length=length_init,
            t0=t0_init
        )

    m.limits["x0"] = (-2000, 2000)
    m.limits["y0"] = (-2000, 2000)
    m.limits["z0"] = (-2000, 2000)
    m.limits["cx"] = (-0.5, 0.5)
    m.limits["cy"] = (-0.5, 0.5)
    m.limits["length"] = (0, 3000)   
    m.limits["t0"] = (-10, 10)
    #m.limits["intensity"] = (0,30)

    m.errors["x0"] = 20.0
    m.errors["y0"] = 20.0
    m.errors["z0"] = 20.0
    m.errors["cx"] = 0.01
    m.errors["cy"] = 0.01
    m.errors["length"] = 50.0
    m.errors["t0"] = 0.5

    m.errordef = Minuit.LIKELIHOOD
    m.strategy = 1  # more careful

    #m.simplex()
    m.migrad(ncall=5000)
    
    while (m.fval >3000):
        print('First attempted fit failed. Trying to fit again...')
        x0_init = 0
        y0_init = 0
        z0_init = np.random.randint(low=-1500,high=500)
        cx_init = 0.0
        cy_init = 0.0
        length_init = 500
        t0_init = 0

        m = Minuit(
                get_neg_log_likelihood_npe_t,
                x0=x0_init,
                y0=y0_init,
                z0=z0_init,
                cx=cx_init,
                cy=cy_init,
                length=length_init,
                t0=t0_init
            )

        m.limits["x0"] = (-2000, 2000)
        m.limits["y0"] = (-2000, 2000)
        m.limits["z0"] = (-2000, 2000)
        m.limits["cx"] = (-0.5, 0.5)
        m.limits["cy"] = (-0.5, 0.5)
        m.limits["length"] = (0, 3000)   
        m.limits["t0"] = (-10, 10)
        #m.limits["intensity"] = (0,30)

        m.errors["x0"] = 20.0
        m.errors["y0"] = 20.0
        m.errors["z0"] = 20.0
        m.errors["cx"] = 0.01
        m.errors["cy"] = 0.01
        m.errors["length"] = 50.0
        m.errors["t0"] = 0.5
        
        m.errordef = Minuit.LIKELIHOOD
        m.strategy = 1  # more careful

        #m.simplex()
        m.migrad(ncall=5000)
        

    return {
        "values": m.values.to_dict(),
        "errors": m.errors.to_dict(),
        "fval": m.fval,
        "valid": m.valid
    }