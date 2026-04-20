import numpy as np
from iminuit import Minuit


def align_event_times_by_peak(ev, bin_width=1.0):
    """
    Shift hit times in an Event so that the global hit-time peak sits near 0 ns.
    """
    all_times = []
    for i_mpmt in range(ev.n_mpmt):
        for i_pmt in range(ev.npmt_per_mpmt):
            all_times.extend(ev.hit_times[i_mpmt][i_pmt])

    if not all_times:
        ev.global_time_offset = None
        return

    all_times = np.asarray(all_times, dtype=np.float64)
    t_min = np.min(all_times)
    t_max = np.max(all_times)

    bins = np.arange(t_min, t_max + bin_width, bin_width)
    counts, edges = np.histogram(all_times, bins=bins)
    peak_time = edges[np.argmax(counts)]

    for i_mpmt in range(ev.n_mpmt):
        for i_pmt in range(ev.npmt_per_mpmt):
            ev.hit_times[i_mpmt][i_pmt] = [
                t - peak_time for t in ev.hit_times[i_mpmt][i_pmt]
            ]

    ev.global_time_offset = peak_time


def build_observables_from_event(ev, wcte, pe_scale=145.0):
    """
    Build the observed PE and first-hit-time arrays in the same PMT ordering used
    by Emitter.get_pmt_placements(...).

    The time array uses NaN for PMTs with no hit so the downstream likelihood can
    stay purely numeric.
    """
    obs_pes = []
    obs_ts = []

    for i_mpmt in range(ev.n_mpmt):
        if not ev.mpmt_status[i_mpmt]:
            continue
        if wcte.mpmts[i_mpmt] is None:
            continue

        for i_pmt in range(ev.npmt_per_mpmt):
            if not ev.pmt_status[i_mpmt][i_pmt]:
                continue
            if wcte.mpmts[i_mpmt].pmts[i_pmt] is None:
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


def fit_track_with_minuit(
    ev,
    wcte,
    pmt_model,
    beta=0.999,
    intensity=5.0,
    pe_scale=145.0,
    align_times=False,
    verbose=True,
    use_simplex=False,
    run_hesse=False,
    strategy=1,
    ncall=6000,
):
    """
    Fast Minuit-based track fit.

    Compared with the original helper, this version:
      - keeps all arrays numeric
      - avoids repeated object conversions
      - defaults to MIGRAD-only (SIMPLEX and HESSE are optional because they are
        usually much more expensive than the core fit)
    """
    if align_times:
        align_event_times_by_peak(ev)

    obs_pes, obs_ts = build_observables_from_event(ev, wcte, pe_scale=pe_scale)
    mask_hit = obs_pes > 0.0

    if verbose:
        print(f"Using {np.sum(mask_hit)} PMTs with hits")

    from Emitter import Emitter

    emitter = Emitter(
        starting_time=0.0,
        start_coord=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        beta=beta,
        length=400.0,
        intensity=intensity,
    )

    p_locations, direction_zs = emitter.get_pmt_placements(ev, wcte, "design")
    p_locations = np.asarray(p_locations, dtype=np.float64)
    direction_zs = np.asarray(direction_zs, dtype=np.float64)

    if np.any(mask_hit):
        center = np.average(p_locations[mask_hit], axis=0, weights=obs_pes[mask_hit])
        x0_init, y0_init, z0_init = center
    else:
        x0_init = y0_init = z0_init = 0.0

    t_valid = obs_ts[np.isfinite(obs_ts)]
    t0_init = float(np.median(t_valid)) if t_valid.size else 0.0

    cx_init = 0.0
    cy_init = 0.0
    length_init = float(np.max(p_locations[:, 2]) - np.min(p_locations[:, 2]))

    if verbose:
        print("Initial guesses:")
        print(f"  x0,y0,z0 = {x0_init:.1f}, {y0_init:.1f}, {z0_init:.1f}")
        print(f"  cx,cy    = {cx_init:.4f}, {cy_init:.4f}")
        print(f"  length   = {length_init:.1f}")
        print(f"  t0       = {t0_init:.2f}")

    def negloglik(x0, y0, z0, cx, cy, length, t0):
        cz2 = 1.0 - cx * cx - cy * cy
        if cz2 <= 0.0:
            return 1e30

        cz = np.sqrt(cz2)
        emitter.start_coord = (x0, y0, z0)
        emitter.direction = (cx, cy, cz)
        emitter.length = length
        emitter.starting_time = t0

        init_ke = emitter.refresh_kinematics_from_length(length)
        s = emitter.get_emission_points(p_locations, init_ke)
        exp_pes, exp_ts = emitter.get_expected_pes_ts(
            wcte,
            s,
            p_locations,
            direction_zs,
            corr_pos=None,
            obs_pes=obs_pes,
        )

        return pmt_model.get_neg_log_likelihood_npe_t(exp_pes, obs_pes, exp_ts, obs_ts)

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
    m.strategy = strategy

    m.limits["x0"] = (-2000, 2000)
    m.limits["y0"] = (-2000, 2000)
    m.limits["z0"] = (-2000, 2000)
    m.limits["cx"] = (-0.5, 0.5)
    m.limits["cy"] = (-0.5, 0.5)
    m.limits["length"] = (100, 3000)
    m.limits["t0"] = (-30, 60)

    m.errors["x0"] = 20.0
    m.errors["y0"] = 20.0
    m.errors["z0"] = 20.0
    m.errors["cx"] = 0.01
    m.errors["cy"] = 0.01
    m.errors["length"] = 200.0
    m.errors["t0"] = 2.0

    if use_simplex:
        m.simplex(ncall=ncall)
    m.migrad(ncall=ncall)

    if run_hesse:
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
