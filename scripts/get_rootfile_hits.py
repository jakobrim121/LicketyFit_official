
import uproot
import awkward as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys, site
sys.path.append(site.getusersitepackages())
#sys.path.append("/eos/user/j/jrimmer/.local/lib/python3.X/site-packages")
#import k3d
sys.path.insert(0, "../../LicketyFit")
sys.path.insert(0, "./")
from plot_event import *

sys.path.insert(0, "/eos/user/j/jrimmer/Geometry")

from Geometry.Device import Device

hall = Device.open_file('/eos/user/j/jrimmer/Geometry/examples/wcte_bldg157.geo')
wcte = hall.wcds[0]

# wcsim uses positions 1-19, so have to subtract 1 in the mapping...
sim_wcte_mapping = {}
for i in range(len(wcte_mapping)):
    sim_wcte_mapping[int(wcte_mapping[i][0])-1] = int(wcte_mapping[i][1]*100 + wcte_mapping[i][2] - 1)
    

# url = '/eos/user/j/jrimmer/sim_work_dir/WCSim/1kmu-_300MeV_x0y0zn1350_noScat.root'
# url = '/eos/user/j/jrimmer/sim_work_dir/WCSim/1kmu-_300MeV_x0y424zn1350_noScat.root'
# fname = url
tree_name1 = "all_photons"
tree_name2 = "pe_tree"
n = 10000000   # number of entries to load

# Get the mapping between wcsim and wcte PMT positions
wcte_mapping = np.loadtxt('../tables/wcsim_wcte_mapping.txt')


def power_law(x):
    # x must be >= 0; enforce numerical safety
    y0_fit   = 0.1209
    #yinf = 41.3407 # plateau
    yinf = 1.6397 # make it less steep at large cost
    x50  = 0.9279    #(half-saturation)
    n_fit    = 3.0777    #  (steepness)

    x = np.clip(x, 0.0, None)
    xn = x**n_fit
    x50n = x50**n_fit

    max_ = 0.967354918872639
    norm = (y0_fit + (yinf - y0_fit) * (xn / (xn + x50n)))/max_

    return norm
        

def get_pe_tree_hits(fname,max_z_cm = 100):

    with uproot.open(fname) as f:
        t = f[tree_name1]
        arr = t.arrays(library="ak", entry_stop=n)

        t_pe = f[tree_name2]
        arr_pe = t_pe.arrays(library="ak", entry_stop=n)


    df_pe = ak.to_dataframe(arr_pe).reset_index(drop=True)

    df = ak.to_dataframe(arr).reset_index(drop=True)



    prop_info = "design"
    z_cut_cm = max_z_cm
    angle_cut_deg = 5.0

    # ------------------------------------------------------------
    # Inputs assumed to already exist:
    #   df_all            -> pandas DataFrame for all_photons
    #   df_pe             -> pandas DataFrame for pe_tree
    #   wcte              -> geometry object
    #   sim_wcte_mapping  -> dict with:
    #                        all_photons PMT = sim_wcte_mapping[pe_tree tubeid + 1]
    # ------------------------------------------------------------

    # -----------------------------
    # 1) Keep only needed columns
    # -----------------------------
    all_cols = [
        "evt", "trk",
        "ex_cm", "ey_cm", "ez_cm",
        "edir_x", "edir_y", "edir_z",
    ]

    pe_cols = [
        "event", "tubeid", "trackid", "time_ns", "lambda_nm"
    ]

    df_all_small = df[all_cols].copy()
    df_pe_small  = df_pe[pe_cols].copy()

    # Make sure merge keys are integer type
    df_all_small["evt"] = pd.to_numeric(df_all_small["evt"], errors="coerce").astype("Int64")
    df_all_small["trk"] = pd.to_numeric(df_all_small["trk"], errors="coerce").astype("Int64")

    df_pe_small["event"]   = pd.to_numeric(df_pe_small["event"], errors="coerce").astype("Int64")
    df_pe_small["trackid"] = pd.to_numeric(df_pe_small["trackid"], errors="coerce").astype("Int64")
    df_pe_small["tubeid"]  = pd.to_numeric(df_pe_small["tubeid"], errors="coerce").astype("Int64")

    # Drop bad rows if any
    df_all_small = df_all_small.dropna(subset=["evt", "trk"])
    df_pe_small  = df_pe_small.dropna(subset=["event", "trackid", "tubeid"])

    # Convert back to normal ints after cleaning
    df_all_small["evt"] = df_all_small["evt"].astype(np.int32)
    df_all_small["trk"] = df_all_small["trk"].astype(np.int32)

    df_pe_small["event"]   = df_pe_small["event"].astype(np.int32)
    df_pe_small["trackid"] = df_pe_small["trackid"].astype(np.int32)
    df_pe_small["tubeid"]  = df_pe_small["tubeid"].astype(np.int32)

    # ---------------------------------------------------------
    # 2) Map pe_tree tubeid -> all_photons/WCTE PMT numbering
    #    using: all_photons PMT = sim_wcte_mapping[tubeid + 1]
    # ---------------------------------------------------------
    tube_keys = (df_pe_small["tubeid"].to_numpy(np.int32) - 1)

    # Fast safe mapping: missing keys become -1
    mapped_pmts = np.array([sim_wcte_mapping.get(int(k), -1) for k in tube_keys], dtype=np.int32)
    df_pe_small["pmt"] = mapped_pmts

    # Keep only rows with a valid PMT mapping
    df_pe_small = df_pe_small[df_pe_small["pmt"] > -1].copy()

    # ----------------------------------------------------------------
    # 3) Match each PE to its corresponding all_photons row by event+trk
    # ----------------------------------------------------------------
    matched = df_pe_small.merge(
        df_all_small,
        left_on=["event", "trackid"],
        right_on=["evt", "trk"],
        how="inner",
        validate="many_to_one",   # one all_photons row per photon track
    )

    # If you are not certain the match is unique in df_all, remove validate=...
    # and optionally inspect duplicates there.

    # ----------------------------------------
    # 4) Apply the same emission-location cut
    # ----------------------------------------
    base_mask = matched["ez_cm"].to_numpy(dtype=float) < z_cut_cm

    matched = matched.loc[base_mask].copy()

    # ----------------------------------------------------
    # 5) Build emission positions/directions from all_photons
    #    and PMT locations from pe_tree-mapped PMT IDs
    # ----------------------------------------------------
    emit_locs = np.column_stack((
        matched["ex_cm"].to_numpy(dtype=float),
        matched["ey_cm"].to_numpy(dtype=float)+42.47625,
        matched["ez_cm"].to_numpy(dtype=float),
    )) * 10.0   # cm -> mm

    emit_dir = np.column_stack((
        matched["edir_x"].to_numpy(dtype=float),
        matched["edir_y"].to_numpy(dtype=float),
        matched["edir_z"].to_numpy(dtype=float),
    ))

    pmts_sel = matched["pmt"].to_numpy(dtype=np.int32)

    # Only query geometry once per unique PMT
    unique_pmts, inv = np.unique(pmts_sel, return_inverse=True)

    unique_pmt_locs = np.array([
        wcte.mpmts[p // 100].pmts[p % 100].get_placement(prop_info)["location"]
        for p in unique_pmts
    ], dtype=float)

    pmt_locs = unique_pmt_locs[inv]

    # -------------------------
    # 6) Apply the same angle cut
    # -------------------------
    to_pmt   = pmt_locs - emit_locs
    r        = np.linalg.norm(to_pmt, axis=1)
    dir_norm = np.linalg.norm(emit_dir, axis=1)

    good = (r > 0) & (dir_norm > 0)

    cosang = np.full(r.shape, -2.0, dtype=float)
    cosang[good] = np.einsum("ij,ij->i", to_pmt[good], emit_dir[good]) / (r[good] * dir_norm[good])

    ang_mask = cosang > np.cos(np.deg2rad(angle_cut_deg))

    matched_pass = matched.loc[ang_mask].copy()

    # ------------------------------------------------------
    # 7) Final PE charge per PMT
    #    Each surviving pe_tree row is already a true PE, so
    #    just count rows per PMT.
    # ------------------------------------------------------
    pmts_pass = matched_pass["pmt"].to_numpy(dtype=np.int32)

    pmts_charges = np.array(
        np.unique(pmts_pass, return_counts=True)
    ).T.astype(np.int32)
    
    return pmts_charges



def get_all_photons_hits(fname, apply_qe_cut = False, apply_angle_corr = False, max_z_cm=100):
    
    with uproot.open(fname) as f:
        t = f[tree_name1]
        arr = t.arrays(library="ak", entry_stop=n)


    df = ak.to_dataframe(arr).reset_index(drop=True)
    
    prop_info = "design"
    z_cut_cm = max_z_cm
    angle_cut_deg = 5.0
    rng_seed = 12345          # set to None for non-reproducible QE sampling
    #apply_qe_cut = True       # <- set to False to skip QE

    rng = np.random.default_rng(rng_seed)

    df_small = df

    # Pull columns once
    ez      = df_small["ez_cm"].to_numpy(dtype=float)
    hit_pmt = df_small["hit_pmt"].to_numpy(dtype=np.int32)
    wl      = df_small["lambda_nm"].to_numpy(dtype=float)

    ex      = df_small["ex_cm"].to_numpy(dtype=float)
    ey      = df_small["ey_cm"].to_numpy(dtype=float) +42.47625

    edir_x  = df_small["edir_x"].to_numpy(dtype=float)
    edir_y  = df_small["edir_y"].to_numpy(dtype=float)
    edir_z  = df_small["edir_z"].to_numpy(dtype=float)

    # Basic cuts:
    #   - photon hits a PMT
    #   - emitted below z_cut_cm
    base_mask = (ez < z_cut_cm) & (hit_pmt > -1)

    pmts_sel = hit_pmt[base_mask]
    wl_sel   = wl[base_mask]

    emit_locs = np.column_stack((
        ex[base_mask],
        ey[base_mask],
        ez[base_mask]
    )) * 10.0   # cm -> mm

    emit_dir = np.column_stack((
        edir_x[base_mask],
        edir_y[base_mask],
        edir_z[base_mask]
    ))

    # PMT locations: only look up each unique PMT once
    unique_pmts_geom, inv_geom = np.unique(pmts_sel, return_inverse=True)

    unique_pmt_locs = np.array([
        wcte.mpmts[p // 100].pmts[p % 100].get_placement(prop_info)["location"]
        for p in unique_pmts_geom
    ], dtype=float)
    
    unique_pmt_dirs = np.array([
        wcte.mpmts[p // 100].pmts[p % 100].get_placement(prop_info)["direction_z"]
        for p in unique_pmts_geom
    ], dtype=float)

    pmt_locs = unique_pmt_locs[inv_geom]
    # Expand back to one row per photon

    pmt_dirs = unique_pmt_dirs[inv_geom]


    # Vectorized angle cut: angle < angle_cut_deg
    to_pmt   = pmt_locs - emit_locs
    r        = np.linalg.norm(to_pmt, axis=1)
    dir_norm = np.linalg.norm(emit_dir, axis=1)
    

    good = (r > 0) & (dir_norm > 0)
    
    cost = np.full(r.shape, -2.0, dtype=float)

    good_cost = r > 0
    cost[good_cost] = -np.einsum("ij,ij->i", pmt_dirs[good_cost], to_pmt[good_cost]) / r[good_cost]

    cosang = np.full(r.shape, -2.0, dtype=float)
    cosang[good] = np.einsum("ij,ij->i", to_pmt[good], emit_dir[good]) / (r[good] * dir_norm[good])

    ang_mask = cosang > np.cos(np.deg2rad(angle_cut_deg))

    # Surviving hits after angle cut
    pmts_pass = pmts_sel[ang_mask]
    wl_pass   = wl_sel[ang_mask]
    cost_good = cost[ang_mask]
    
    

    # Optionally apply QE
    if apply_qe_cut:
        # qe_int[:,0] = wavelength (nm)
        # qe_int[:,1] = QE
        # outside the qe_int wavelength range, QE is set to 0
        qe_int = np.load('../tables/qe.npy')
        qe_wl  = qe_int[:, 0].astype(float)
        qe_val = qe_int[:, 1].astype(float)

        qe_prob = np.interp(wl_pass, qe_wl, qe_val, left=0.0, right=0.0)

        # Accept/reject each hit according to QE
        qe_accept_mask = rng.random(wl_pass.size) < qe_prob
        #qe_accept_mask = rng.random(wl_pass.size) < qe_prob/(1-0.25)
        pmts_final = pmts_pass[qe_accept_mask]
        cost_final = cost_good[qe_accept_mask]
    else:
        pmts_final = pmts_pass
        cost_final = cost_good
   

    # Final array:
    # column 0 = PMT number
    # column 1 = number of photons surviving all requested cuts
    pmts_charges = np.array(np.unique(pmts_final, return_counts=True)).T.astype(np.int32)
    
    if apply_angle_corr:
        #corrs = power_law(cost_final)
        unique_pmts, inv = np.unique(pmts_final, return_inverse=True)

        # Sum of cost values for each PMT
        cost_sum = np.bincount(inv, weights=cost_final)

        # Number of hits for each PMT
        hit_count = np.bincount(inv)

        # Average cost for each PMT
        avg_cost = cost_sum / hit_count

        corrs = power_law(avg_cost)
        #corrs = avg_cost

        # Multiply the PMT charges by the average cost
        weighted_charges = np.array([pmts_charges[:,0],pmts_charges[:, 1].astype(float) * corrs]).T
        
        return weighted_charges
        
    else:
        
        return pmts_charges