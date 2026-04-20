import numpy as np

def align_event_times_by_peak(ev, bin_width=1.0):
    """
    Shift hit times in an Event so that the main physics peak is at ~0 ns.

    This is robust against early dark hits, because it uses the peak of the
    global time distribution (like your professor's get_combined_event).

    Modifies ev.hit_times in place and adds ev.global_time_offset.
    """
    all_times = []
    for i_mpmt in range(ev.n_mpmt):
        for i_pmt in range(ev.npmt_per_mpmt):
            all_times.extend(ev.hit_times[i_mpmt][i_pmt])

    if not all_times:
        ev.global_time_offset = None
        return

    all_times = np.array(all_times, dtype=float)
    t_min = np.min(all_times)
    t_max = np.max(all_times)

    # 1 ns bins by default
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    counts, edges = np.histogram(all_times, bins=bins)

    peak_time = edges[np.argmax(counts)]

    # shift so that the physics peak is at ~0 ns
    for i_mpmt in range(ev.n_mpmt):
        for i_pmt in range(ev.npmt_per_mpmt):
            ev.hit_times[i_mpmt][i_pmt] = [
                t - peak_time for t in ev.hit_times[i_mpmt][i_pmt]
            ]

    ev.global_time_offset = peak_time

    
    
def build_observables_from_event(ev, wcte, pe_scale=145.0):
    """
    Build obs_pes and obs_ts arrays from a real Event, in the same PMT order
    as used by Emitter.get_pmt_placements(simulated_event, wcte, 'design').

    - obs_pes: approximate PE per PMT (total charge / pe_scale)
    - obs_ts:  first hit time per PMT, or None if no hits

    Returns:
        obs_pes (np.ndarray, float)
        obs_ts (np.ndarray, dtype=object)
    """
    obs_pes = []
    obs_ts = []

    n_mpmt = ev.n_mpmt
    npmt_per_mpmt = ev.npmt_per_mpmt

    for i_mpmt in range(n_mpmt):
        if not ev.mpmt_status[i_mpmt]:
            continue
        # skip geometry holes
        if wcte.mpmts[i_mpmt] is None:
            continue

        for i_pmt in range(npmt_per_mpmt):
            if not ev.pmt_status[i_mpmt][i_pmt]:
                continue
            if wcte.mpmts[i_mpmt].pmts[i_pmt] is None:
                continue

            charges = ev.hit_charges[i_mpmt][i_pmt]
            times   = ev.hit_times[i_mpmt][i_pmt]

            if len(charges) == 0:
                obs_pes.append(0.0)
                obs_ts.append(None)
            else:
                total_q = float(np.sum(charges))
                npe = total_q / pe_scale
                t_first = float(np.min(times))

                obs_pes.append(npe)
                obs_ts.append(t_first)

    return np.array(obs_pes, dtype=float), np.array(obs_ts, dtype=object)


from iminuit import Minuit
import numpy as np

def fit_track_with_minuit(
    ev,
    wcte,
    pmt_model,
    beta=0.999,
    intensity=5.0,
    pe_scale=145.0,
    align_times=False,
    verbose=True,
):
    """
    Minuit-based track fit with:
     - free vertex (x0,y0,z0)
     - free direction (cx,cy)
     - free track length
     - free t0
    """

    # ---------------------------------------------------------
    # (1) Build observables
    # ---------------------------------------------------------
    obs_pes, obs_ts = build_observables_from_event(ev, wcte, pe_scale=pe_scale)
    mask_hit = obs_pes > 0

    if verbose:
        print(f"Using {np.sum(mask_hit)} PMTs with hits")

    # ---------------------------------------------------------
    # (2) Build Emitter model
    # ---------------------------------------------------------
    from LicketyFit.Emitter import Emitter

    emitter_model = Emitter(
        starting_time=0.0,
        start_coord=(0, 0, 0),
        direction=(0, 0, 1),
        beta=beta,
        length=400.0,
        intensity=intensity,
    )
    emitter_copy = emitter_model.copy()

    # PMT geometry
    p_locations, direction_zs = emitter_copy.get_pmt_placements(ev, wcte, "design")
    p_locations = np.array(p_locations, float)
    direction_zs = np.array(direction_zs, float)

    # ---------------------------------------------------------
    # (3) Initial guesses
    # ---------------------------------------------------------
    if np.any(mask_hit):
        center = np.average(p_locations[mask_hit], axis=0, weights=obs_pes[mask_hit])
        x0_init, y0_init, z0_init = center
    else:
        x0_init = y0_init = z0_init = 0.0

    # time guess
    t_valid = np.array([t for t in obs_ts if t is not None], float)
    t0_init = float(np.median(t_valid)) if len(t_valid) else 0.0

    # direction
    cx_init, cy_init = 0.0, 0.0

    # length estimate: highest-z PMT minus lowest-z PMT
    length_init = np.max(p_locations[:,2]) - np.min(p_locations[:,2])

    if verbose:
        print("Initial guesses:")
        print(f"  x0,y0,z0 = {x0_init:.1f}, {y0_init:.1f}, {z0_init:.1f}")
        print(f"  cx,cy    = {cx_init:.4f}, {cy_init:.4f}")
        print(f"  length   = {length_init:.1f}")
        print(f"  t0       = {t0_init:.2f}")

    # ---------------------------------------------------------
    # (4) Likelihood function
    # ---------------------------------------------------------
    def negloglik(x0, y0, z0, cx, cy, length, t0):

        # enforce normalization
        cz2 = 1. - cx**2 - cy**2
        if cz2 <= 0:
            return 1e30
        cz = np.sqrt(cz2)

        emitter_copy.start_coord = (x0, y0, z0)
        emitter_copy.direction   = (cx, cy, cz)
        emitter_copy.length      = length
        emitter_copy.starting_time = t0

        ss = emitter_copy.get_emission_points(p_locations)
        exp_pes, exp_ts = emitter_copy.get_expected_pes_ts(
            wcte, ss, p_locations, direction_zs
        )

        return pmt_model.get_neg_log_likelihood_npe_t(
            exp_pes, obs_pes, exp_ts, obs_ts
        )

    # ---------------------------------------------------------
    # (5) Minuit setup
    # ---------------------------------------------------------
    m = Minuit(
        negloglik,
        x0=x0_init,
        y0=y0_init,
        z0=z0_init,
        cx=cx_init,
        cy=cy_init,
        length=length_init,
        t0=t0_init,
    )

    m.errordef = Minuit.LIKELIHOOD
    m.strategy = 2

    # parameter limits
    m.limits["x0"] = (-2000, 2000)
    m.limits["y0"] = (-2000, 2000)
    m.limits["z0"] = (-2000, 2000)
    m.limits["cx"] = (-0.5, 0.5)
    m.limits["cy"] = (-0.5, 0.5)
    m.limits["length"] = (100, 3000)   # now FREE
    m.limits["t0"] = (-30, 60)

    # step sizes
    m.errors["x0"] = 20
    m.errors["y0"] = 20
    m.errors["z0"] = 20
    m.errors["cx"] = 0.01
    m.errors["cy"] = 0.01
    m.errors["length"] = 200
    m.errors["t0"] = 2

    # ---------------------------------------------------------
    # (6) Staged minimization
    # ---------------------------------------------------------
    if verbose:
        print("\nStage 1: vertex-only")

    m.fixed["cx"] = True
    m.fixed["cy"] = True
    m.fixed["length"] = True
    m.simplex()
    m.migrad()

    if verbose:
        print("\nStage 2: +direction")

    m.fixed["cx"] = False
    m.fixed["cy"] = False
    m.simplex()
    m.migrad()

    if verbose:
        print("\nStage 3: +length (FULL FREE FIT)")

    m.fixed["length"] = False
    m.simplex()
    m.migrad()

    # Hesse for errors
    m.hesse()

    if verbose:
        print("\nMinuit valid:", m.valid)
        print(m.fmin)

    return {
        "values": m.values,
        "errors": m.errors,
        "covariance": m.covariance,
        "minuit": m,
    }
