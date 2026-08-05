import math
import os

import numpy as np
from numba import njit

@njit(cache=True)
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@njit(cache=True)
def _timing_weight_scalar(obs, mu_time, policy_code):
    """Return the effective PE count used as timing weight.

    policy_code:
      0 = legacy/current observed-PE weight, independent of mu_time
      1 = observed-PE weight after a model-light gate
      2 = predicted/model PE weight
      3 = min(observed PE, predicted/model PE) weight  [new default]
      4 = harmonic observed/model PE weight
    """
    if policy_code == 0 or policy_code == 1:
        return obs
    if policy_code == 2:
        return mu_time
    if policy_code == 3:
        return obs if obs < mu_time else mu_time
    if policy_code == 4:
        return (obs * mu_time) / (obs + mu_time + 1e-300)
    return obs if obs < mu_time else mu_time


@njit(cache=True)
def _poisson_time_nll(exp_pes, obs_pes, exp_ts, obs_ts, single_pe_time_std):
    """Legacy charge+time likelihood.

    Kept for explicit backwards-compatibility tests only.  The production default
    is _poisson_time_nll_split(), which separates the floored charge expectation
    from the unfloored physical timing expectation.
    """
    n = exp_pes.size
    nll = 0.0
    log_noise = math.log(1e-4)

    for i in range(n):
        lam = exp_pes[i]
        obs = obs_pes[i]

        # Charge term
        if lam > 0.0:
            nll += lam - obs * math.log(lam)
        elif obs > 0.0:
            nll -= obs * log_noise

        # Timing term
        t_obs = obs_ts[i]
        if (lam > 0.0) and (obs > 0.0) and np.isfinite(t_obs) and np.isfinite(exp_ts[i]):
            sigma_t = single_pe_time_std / math.sqrt(obs)
            dt = (t_obs - exp_ts[i]) / sigma_t
            nll += 0.5 * dt * dt

    return nll


@njit(cache=True)
def _poisson_time_nll_split(
    exp_pes_charge,
    obs_pes,
    exp_ts,
    obs_ts,
    exp_pes_timing,
    single_pe_time_std,
    timing_policy_code,
    timing_mu_min_pe,
    timing_sigma_sys_ns,
    timing_include_lognorm,
):
    """Charge+time NLL with split charge and timing PE expectations.

    exp_pes_charge is the charge-likelihood expectation and may include a small
    floor.  exp_pes_timing is the unfloored physical expected PE and controls
    whether/how strongly a PMT time is used.

    Default policy_code=3 uses w_time=min(obs_pe, exp_pes_timing).  This keeps
    the timing term strong only where both data and model support light, and it
    prevents the charge floor from giving timing leverage to model-unlit PMTs.
    """
    n = exp_pes_charge.size
    nll = 0.0
    log_noise = math.log(1e-4)
    sig2 = single_pe_time_std * single_pe_time_std

    for i in range(n):
        lam = exp_pes_charge[i]
        obs = obs_pes[i]

        # Charge term: use the floored/protected charge expectation.
        if lam > 0.0:
            nll += lam - obs * math.log(lam)
        elif obs > 0.0:
            nll -= obs * log_noise

        # Timing term: use the unfloored physical expectation for eligibility
        # and weighting.
        if obs <= 0.0:
            continue
        t_obs = obs_ts[i]
        t_exp = exp_ts[i]
        if (not np.isfinite(t_obs)) or (not np.isfinite(t_exp)):
            continue

        if timing_policy_code == 0:
            # Exact legacy/current timing behavior: any positive charge-likelihood
            # expectation makes an observed PMT time-able.
            if lam <= 0.0:
                continue
            w_time = obs
        else:
            mu_time = exp_pes_timing[i]
            if (not np.isfinite(mu_time)) or mu_time <= timing_mu_min_pe:
                continue
            w_time = _timing_weight_scalar(obs, mu_time, timing_policy_code)
            if (not np.isfinite(w_time)) or w_time <= 0.0:
                continue

        dt = t_obs - t_exp
        # If w_time is model-dependent, it is an inverse-variance parameter.
        # The proper Gaussian NLL includes the normalization term 0.5*log(var).
        # Omitting this term rewards hypotheses that artificially reduce w_time,
        # which caused the charge+time short-length bias in the noE/noScat tests.
        var = sig2 / w_time + timing_sigma_sys_ns * timing_sigma_sys_ns
        if (not np.isfinite(var)) or var <= 0.0:
            continue
        nll += 0.5 * dt * dt / var
        if timing_include_lognorm != 0 and timing_policy_code != 0:
            nll += 0.5 * math.log(var)

    return nll


@njit(cache=True)
def _time_nll_split_only(
    exp_pes_charge,
    obs_pes,
    exp_ts,
    obs_ts,
    exp_pes_timing,
    single_pe_time_std,
    timing_policy_code,
    timing_mu_min_pe,
    timing_sigma_sys_ns,
    timing_include_lognorm,
):
    """Timing-only NLL using the same split timing policy as npe+t."""
    n = obs_pes.size
    nll = 0.0
    sig2 = single_pe_time_std * single_pe_time_std
    any_used = False

    for i in range(n):
        obs = obs_pes[i]
        if obs <= 0.0:
            continue
        t_obs = obs_ts[i]
        t_exp = exp_ts[i]
        if (not np.isfinite(t_obs)) or (not np.isfinite(t_exp)):
            continue

        if timing_policy_code == 0:
            if exp_pes_charge[i] <= 0.0:
                continue
            w_time = obs
        else:
            mu_time = exp_pes_timing[i]
            if (not np.isfinite(mu_time)) or mu_time <= timing_mu_min_pe:
                continue
            w_time = _timing_weight_scalar(obs, mu_time, timing_policy_code)
            if (not np.isfinite(w_time)) or w_time <= 0.0:
                continue

        dt = t_obs - t_exp
        # If w_time is model-dependent, it is an inverse-variance parameter.
        # The proper Gaussian NLL includes the normalization term 0.5*log(var).
        # Omitting this term rewards hypotheses that artificially reduce w_time,
        # which caused the charge+time short-length bias in the noE/noScat tests.
        var = sig2 / w_time + timing_sigma_sys_ns * timing_sigma_sys_ns
        if (not np.isfinite(var)) or var <= 0.0:
            continue
        nll += 0.5 * dt * dt / var
        if timing_include_lognorm != 0 and timing_policy_code != 0:
            nll += 0.5 * math.log(var)
        any_used = True

    if not any_used:
        return 1e30
    return nll

@njit(cache=True)
def _poisson_nll(exp_pes, obs_pes):
    """Tight charge-only Poisson NLL kernel with no temporary masks."""
    n = exp_pes.size
    nll = 0.0
    log_noise = math.log(1e-4)
    for i in range(n):
        lam = exp_pes[i]
        obs = obs_pes[i]
        if lam > 0.0:
            nll += lam - obs * math.log(lam)
        elif obs > 0.0:
            nll -= obs * log_noise
    return nll


_WCSIM_WCTE_TTS_Q = np.array([0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0], dtype=np.float64)
_WCSIM_WCTE_TTS_RESOL = np.array([1.1654,0.61088,0.4186,0.32532,0.26484,0.23084,0.20969,0.19297,0.17716,0.17046,0.15455,0.1427,0.13699,0.13229], dtype=np.float64)


_FIRST_ARRIVAL_OBS_CACHE = {}

def _wcsim_wcte_first_digit_sigma_ns(q, output_efficiency=0.985):
    q = np.asarray(q, dtype=np.float64)
    qi = np.clip(q / float(output_efficiency), 0.5, 4.0)
    resol = np.interp(qi, _WCSIM_WCTE_TTS_Q, _WCSIM_WCTE_TTS_RESOL)
    return np.sqrt(1.5**2 + resol**2) / 2.355


@njit(cache=True)
def _wcsim_wcte_first_digit_sigma_scalar_ns(q, output_efficiency):
    qi = q / output_efficiency
    if qi < 0.5:
        qi = 0.5
    elif qi > 4.0:
        qi = 4.0
    n = _WCSIM_WCTE_TTS_Q.size
    if qi <= _WCSIM_WCTE_TTS_Q[0]:
        resol = _WCSIM_WCTE_TTS_RESOL[0]
    elif qi >= _WCSIM_WCTE_TTS_Q[n - 1]:
        resol = _WCSIM_WCTE_TTS_RESOL[n - 1]
    else:
        k = np.searchsorted(_WCSIM_WCTE_TTS_Q, qi)
        x0 = _WCSIM_WCTE_TTS_Q[k - 1]
        x1 = _WCSIM_WCTE_TTS_Q[k]
        y0 = _WCSIM_WCTE_TTS_RESOL[k - 1]
        y1 = _WCSIM_WCTE_TTS_RESOL[k]
        resol = y0 + (qi - x0) * (y1 - y0) / (x1 - x0)
    return math.sqrt(1.5 * 1.5 + resol * resol) / 2.355


@njit(cache=True, inline='always')
def _normal_interval_probability_stable(zlo, zhi):
    """Stable standard-normal probability on [zlo, zhi].

    A prompt-window likelihood is conditioned on a truncated Gaussian.  Both
    the observed-time density and the window acceptance may lie many sigma in
    the same tail, so replacing an exterior interval by exactly zero changes
    their finite ratio into an artificial 1e-300 likelihood floor.
    """
    if (not math.isfinite(zlo)) or (not math.isfinite(zhi)) or zhi <= zlo:
        return 0.0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    if zhi <= 0.0:
        return 0.5 * (math.erfc(-zhi * inv_sqrt2) - math.erfc(-zlo * inv_sqrt2))
    if zlo >= 0.0:
        return 0.5 * (math.erfc(zlo * inv_sqrt2) - math.erfc(zhi * inv_sqrt2))
    return 0.5 * (math.erf(zhi * inv_sqrt2) - math.erf(zlo * inv_sqrt2))


@njit(cache=True)
def _first_arrival_nodes_nll_numba(
    node_mu,
    node_t,
    active,
    obs_pes,
    obs_ts,
    prompt_lo,
    prompt_hi,
    output_efficiency,
):
    """Fast conditional first-photoelectron NLL for ordered source nodes.

    The node arrays are ordered from earliest to latest in each PMT column.
    For an observed effective count n, a node with survival probabilities S_j
    and S_{j+1} is first with probability S_j**n - S_{j+1}**n.  That discrete
    first-arrival law is convolved with the configured WCSim WCTE first-digit
    transit-time spread and conditioned on the selected prompt window.
    """
    n_nodes = node_mu.shape[0]
    n_cols = node_mu.shape[1]
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    use_window = math.isfinite(prompt_lo) and math.isfinite(prompt_hi) and prompt_hi > prompt_lo
    nll = 0.0
    n_used = 0

    for i in range(n_cols):
        ipmt = int(active[i])
        if ipmt < 0 or ipmt >= obs_pes.size:
            continue
        q = obs_pes[ipmt]
        tobs = obs_ts[ipmt]
        if q <= 0.0 or (not math.isfinite(q)) or (not math.isfinite(tobs)):
            continue

        total = 0.0
        for j in range(n_nodes):
            m = float(node_mu[j, i])
            if math.isfinite(m) and m > 0.0:
                total += m
        if total <= 0.0 or (not math.isfinite(total)):
            continue

        neff = q / output_efficiency
        if neff < 1.0e-6:
            neff = 1.0e-6
        sigma = _wcsim_wcte_first_digit_sigma_scalar_ns(q, output_efficiency)
        if sigma <= 0.0 or (not math.isfinite(sigma)):
            continue

        remaining = 1.0
        remaining_power = 1.0
        mix = 0.0
        acceptance = 0.0
        sum_w = 0.0
        inv_sigma = 1.0 / sigma

        for j in range(n_nodes):
            m = float(node_mu[j, i])
            if (not math.isfinite(m)) or m <= 0.0:
                continue
            p = m / total
            next_remaining = remaining - p
            if next_remaining < 0.0:
                next_remaining = 0.0
            next_power = next_remaining ** neff
            w = remaining_power - next_power
            remaining = next_remaining
            remaining_power = next_power
            if w <= 0.0 or (not math.isfinite(w)):
                continue
            tau = float(node_t[j, i])
            if not math.isfinite(tau):
                continue
            z = (tobs - tau) * inv_sigma
            pdf = math.exp(-0.5 * z * z) * inv_sigma * inv_sqrt_2pi
            mix += w * pdf
            if use_window:
                zhi_std = (prompt_hi - tau) * inv_sigma
                zlo_std = (prompt_lo - tau) * inv_sigma
                a = _normal_interval_probability_stable(zlo_std, zhi_std)
                if a > 0.0 and math.isfinite(a):
                    acceptance += w * a
            sum_w += w

        if use_window:
            if acceptance <= 0.0 or mix <= 0.0:
                nll += -math.log(1.0e-300)
            else:
                nll += -math.log(max(mix / acceptance, 1.0e-300))
        else:
            if sum_w <= 0.0 or mix <= 0.0:
                nll += -math.log(1.0e-300)
            else:
                nll += -math.log(max(mix / sum_w, 1.0e-300))
        n_used += 1

    if n_used == 0:
        return 1.0e30
    return nll


@njit(cache=True)
def _first_arrival_weighted_nll_numba(
    node_weight, node_t, active, obs_pes, obs_ts, prompt_lo, prompt_hi,
    output_efficiency,
):
    """Conditional first-photoelectron NLL from precomputed exact weights."""
    n_nodes = node_weight.shape[0]
    n_cols = node_weight.shape[1]
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    use_window = (
        math.isfinite(prompt_lo) and math.isfinite(prompt_hi)
        and prompt_hi > prompt_lo
    )
    nll = 0.0
    n_used = 0
    for i in range(n_cols):
        ipmt = int(active[i])
        if ipmt < 0 or ipmt >= obs_pes.size:
            continue
        q = obs_pes[ipmt]
        tobs = obs_ts[ipmt]
        if q <= 0.0 or (not math.isfinite(q)) or (not math.isfinite(tobs)):
            continue
        sigma = _wcsim_wcte_first_digit_sigma_scalar_ns(q, output_efficiency)
        if sigma <= 0.0 or (not math.isfinite(sigma)):
            continue
        inv_sigma = 1.0 / sigma
        mix = 0.0
        acceptance = 0.0
        sum_w = 0.0
        for j in range(n_nodes):
            w = float(node_weight[j, i])
            if (not math.isfinite(w)) or w <= 0.0:
                continue
            tau = float(node_t[j, i])
            if not math.isfinite(tau):
                continue
            z = (tobs - tau) * inv_sigma
            pdf = math.exp(-0.5 * z * z) * inv_sigma * inv_sqrt_2pi
            mix += w * pdf
            if use_window:
                zhi_std = (prompt_hi - tau) * inv_sigma
                zlo_std = (prompt_lo - tau) * inv_sigma
                a = _normal_interval_probability_stable(zlo_std, zhi_std)
                if a > 0.0 and math.isfinite(a):
                    acceptance += w * a
            sum_w += w
        if use_window:
            if acceptance <= 0.0 or mix <= 0.0:
                nll += -math.log(1.0e-300)
            else:
                nll += -math.log(max(mix / acceptance, 1.0e-300))
        else:
            if sum_w <= 0.0 or mix <= 0.0:
                nll += -math.log(1.0e-300)
            else:
                nll += -math.log(max(mix / sum_w, 1.0e-300))
        n_used += 1
    if n_used == 0:
        return 1.0e30
    return nll



@njit(cache=True, fastmath=True)
def _first_arrival_weighted_prepared_nll_numba(
    node_weight, node_t, q_active, t_active, inv_sigma_active,
    prompt_lo, prompt_hi,
):
    n_nodes, n_cols = node_weight.shape
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    use_window = math.isfinite(prompt_lo) and math.isfinite(prompt_hi) and prompt_hi > prompt_lo
    nll = 0.0
    n_used = 0
    for i in range(n_cols):
        q = float(q_active[i]); tobs = float(t_active[i]); inv_sigma = float(inv_sigma_active[i])
        if q <= 0.0 or (not math.isfinite(tobs)) or inv_sigma <= 0.0:
            continue
        mix = 0.0; acceptance = 0.0; sum_w = 0.0
        for j in range(n_nodes):
            w = float(node_weight[j, i])
            if w <= 0.0 or (not math.isfinite(w)):
                continue
            tau = float(node_t[j, i])
            if not math.isfinite(tau):
                continue
            z = (tobs - tau) * inv_sigma
            gpdf = _first_arrival_exp_lut(z)
            if gpdf > 0.0:
                mix += w * gpdf * inv_sigma * inv_sqrt_2pi
            if use_window:
                zhi_std = (prompt_hi - tau) * inv_sigma
                zlo_std = (prompt_lo - tau) * inv_sigma
                a = _normal_interval_probability_stable(zlo_std, zhi_std)
                if a > 0.0 and math.isfinite(a):
                    acceptance += w*a
            sum_w += w
        if use_window:
            if acceptance <= 0.0 or mix <= 0.0: nll += -math.log(1e-300)
            else: nll += -math.log(max(mix/acceptance,1e-300))
        else:
            if sum_w <= 0.0 or mix <= 0.0: nll += -math.log(1e-300)
            else: nll += -math.log(max(mix/sum_w,1e-300))
        n_used += 1
    if n_used == 0: return 1e30
    return nll



@njit(cache=True, fastmath=True)
def _first_arrival_deferred_reflection_nll_numba(
    base_mu, base_t, ref_u, ref_tbase, transfer_active, time_offset_active,
    patch_min_offset, patch_max_offset, n_bins,
    q_active, t_active, inv_sigma_active,
    output_efficiency, prompt_lo, prompt_hi, node_pe_scale, reflection_occupancy_mix, direct_support_scale_pe,
):
    """24-bin reflected first-arrival likelihood with an optional leverage guard.

    The historical conditional source mixture normalized direct and reflected
    node amplitudes separately inside every observed PMT.  A PMT with no
    resolved direct/delta/scattered prediction could therefore receive full
    timing leverage from an arbitrarily tiny reflected expectation.  This
    kernel retains the validated parameter-dependent 192-patch/24-bin optical
    field.  When the opt-in guard is enabled, a weak reflection-only prediction
    must pay its absolute Poisson occupancy probability before constraining the
    track.  The production default remains the verified legacy likelihood.

    For resolved non-reflected support the result tends smoothly to the
    original conditional first-arrival likelihood.  For sub-floor support and
    vanishing reflected PE it tends to a maximum-entropy nuisance density over
    the already selected prompt interval.  No event-truth template, reflected
    fraction, time offset, or detector-position correction enters the model.
    """
    nb, nc = base_mu.shape
    npatch = ref_u.size
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0*math.pi)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    use_window = math.isfinite(prompt_lo) and math.isfinite(prompt_hi) and prompt_hi>prompt_lo
    tmin=1.0e300; tmax=-1.0e300
    for p in range(npatch):
        if float(ref_u[p]) <= 0.0: continue
        lo=float(ref_tbase[p])+float(patch_min_offset[p])
        hi=float(ref_tbase[p])+float(patch_max_offset[p])
        if lo<tmin:tmin=lo
        if hi>tmax:tmax=hi
    if tmax<tmin:
        return 1.0e30
    span=tmax-tmin
    if span<1.0e-12:span=1.0e-12
    inv_span_bins=float(n_bins)/span
    # Reused serial workspaces: no node matrices leave this kernel.
    sbm=np.empty(nb,dtype=np.float32);sbt=np.empty(nb,dtype=np.float32)
    rmu=np.empty(n_bins,dtype=np.float64);rtn=np.empty(n_bins,dtype=np.float64)
    nll=0.0;n_used=0
    for i in range(nc):
        nvalid=0
        for j in range(nb):
            m=float(base_mu[j,i]);tt=float(base_t[j,i])
            if m<=0.0 or (not math.isfinite(m)) or (not math.isfinite(tt)):continue
            k=nvalid
            while k>0 and tt<float(sbt[k-1]):
                sbt[k]=sbt[k-1];sbm[k]=sbm[k-1];k-=1
            sbt[k]=tt;sbm[k]=m;nvalid+=1
        for b in range(n_bins):rmu[b]=0.0;rtn[b]=0.0
        ref_total=0.0
        for p in range(npatch):
            m=float(ref_u[p])*float(transfer_active[i,p])
            if m<=0.0:continue
            tt=float(ref_tbase[p])+float(time_offset_active[i,p])
            b=int((tt-tmin)*inv_span_bins)
            if b<0:b=0
            elif b>=n_bins:b=n_bins-1
            rmu[b]+=m;rtn[b]+=m*tt;ref_total+=m
        q=float(q_active[i]);tobs=float(t_active[i]);inv_sigma=float(inv_sigma_active[i])
        if q<=0.0 or (not math.isfinite(tobs)) or inv_sigma<=0.0:continue
        total=ref_total
        for j in range(nvalid):total+=float(sbm[j])
        if total<=0.0 or (not math.isfinite(total)):
            nll+=-math.log(1.0e-300);n_used+=1;continue
        neff=q/output_efficiency if output_efficiency>0.0 else q
        if neff<1.0e-6:neff=1.0e-6
        # Only the absolute non-reflected support is needed by the optional
        # leverage guard.  Do not evaluate a second timing mixture here: with
        # the guard disabled this must remain the validated legacy hot path.
        base_total=0.0
        if reflection_occupancy_mix:
            for j in range(nvalid):base_total+=float(sbm[j])
        remaining=1.0;remaining_power=1.0
        mix=0.0;acceptance=0.0;sum_w=0.0
        ib=0;ir=0
        while ib<nvalid or ir<n_bins:
            while ir<n_bins and rmu[ir]<=0.0:ir+=1
            if ib>=nvalid and ir>=n_bins:break
            take_base=False
            if ir>=n_bins:take_base=True
            elif ib<nvalid:
                rt=rtn[ir]/rmu[ir]
                if float(sbt[ib])<=rt:take_base=True
            if take_base:
                mnode=float(sbm[ib]);tau=float(sbt[ib]);ib+=1
            else:
                mnode=rmu[ir];tau=rtn[ir]/mnode;ir+=1
            pnode=mnode/total;next_remaining=remaining-pnode
            if next_remaining<0.0:next_remaining=0.0
            next_power=next_remaining**neff
            w=remaining_power-next_power
            remaining=next_remaining;remaining_power=next_power
            if w<=0.0 or (not math.isfinite(w)):
                # The likelihood itself is floored at 1e-300, so only a truly
                # sub-floor remaining first-arrival probability can be skipped.
                if remaining_power <= 1.0e-300:
                    break
                continue
            z=(tobs-tau)*inv_sigma
            gpdf=_first_arrival_exp_lut(z)
            if gpdf>0.0:
                mix+=w*gpdf*inv_sigma*inv_sqrt_2pi
            if use_window:
                zhi=(prompt_hi-tau)*inv_sigma;zlo=(prompt_lo-tau)*inv_sigma
                a=_normal_interval_probability_stable(zlo,zhi)
                if a>0.0 and math.isfinite(a):acceptance+=w*a
            sum_w+=w
        if use_window:
            full_density=0.0 if (acceptance<=0.0 or mix<=0.0) else mix/acceptance
            if reflection_occupancy_mix:
                mu_base=max(float(node_pe_scale)*base_total,0.0)
                mu_ref=max(float(node_pe_scale)*ref_total,0.0)
                # Probability that the absolute reflected expectation
                # realizes at least one detected PE.  Unlike a normalized source
                # fraction, this vanishes continuously with reflection amplitude.
                p_ref=-math.expm1(-mu_ref)
                # Smoothly retain the validated conditional likelihood wherever
                # the non-reflected prediction is resolved above the same 1e-4
                # PE numerical floor used by the charge likelihood.
                support_scale=max(float(direct_support_scale_pe),1.0e-12)
                direct_gate=mu_base/(mu_base+support_scale)
                trust=direct_gate+(1.0-direct_gate)*p_ref
                unresolved_density=1.0/max(prompt_hi-prompt_lo,1.0e-12)
                density=trust*full_density+(1.0-trust)*unresolved_density
            else:
                density=full_density
            nll+=-math.log(max(density,1.0e-300))
        else:
            full_density=0.0 if (sum_w<=0.0 or mix<=0.0) else mix/sum_w
            if reflection_occupancy_mix:
                mu_base=max(float(node_pe_scale)*base_total,0.0)
                mu_ref=max(float(node_pe_scale)*ref_total,0.0)
                # Probability that the absolute reflected expectation
                # realizes at least one detected PE.  Unlike a normalized source
                # fraction, this vanishes continuously with reflection amplitude.
                p_ref=-math.expm1(-mu_ref)
                # Smoothly retain the validated conditional likelihood wherever
                # the non-reflected prediction is resolved above the same 1e-4
                # PE numerical floor used by the charge likelihood.
                support_scale=max(float(direct_support_scale_pe),1.0e-12)
                direct_gate=mu_base/(mu_base+support_scale)
                trust=direct_gate+(1.0-direct_gate)*p_ref
                unresolved_density=1.0/max(prompt_hi-prompt_lo,1.0e-12)
                density=trust*full_density+(1.0-trust)*unresolved_density
            else:
                density=full_density
            nll+=-math.log(max(density,1.0e-300))
        n_used+=1
    return nll if n_used>0 else 1.0e30



@njit(cache=True, fastmath=True)
def _first_arrival_deferred_reflection_t0_grid_nll_numba(
    base_mu, base_t, ref_u, ref_tbase, transfer_active, time_offset_active,
    patch_min_offset, patch_max_offset, n_bins,
    q_active, t_active, inv_sigma_active, t0_values,
    output_efficiency, prompt_lo, prompt_hi, node_pe_scale,
    reflection_occupancy_mix, direct_support_scale_pe,
):
    """Exact production first-arrival NLL for many additive time shifts.

    Reflection compression, base-node sorting, and discrete first-source
    weights depend on geometry and observed charge, but not on the additive
    event time.  The scalar production kernel historically rebuilt those
    quantities for every point in a t0 profile.  This kernel performs the same
    operations once per PMT and then evaluates every requested shift in the
    same PMT order and with the same likelihood algebra.
    """
    nb, nc = base_mu.shape
    nt = t0_values.size
    out = np.zeros(nt, dtype=np.float64)
    if nt == 0:
        return out
    npatch = ref_u.size
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    use_window = (
        math.isfinite(prompt_lo) and math.isfinite(prompt_hi)
        and prompt_hi > prompt_lo
    )
    tmin = 1.0e300
    tmax = -1.0e300
    for p in range(npatch):
        if float(ref_u[p]) <= 0.0:
            continue
        lo = float(ref_tbase[p]) + float(patch_min_offset[p])
        hi = float(ref_tbase[p]) + float(patch_max_offset[p])
        if lo < tmin:
            tmin = lo
        if hi > tmax:
            tmax = hi
    if tmax < tmin:
        for it in range(nt):
            out[it] = 1.0e30
        return out
    span = tmax - tmin
    if span < 1.0e-12:
        span = 1.0e-12
    inv_span_bins = float(n_bins) / span

    max_nodes = nb + n_bins
    sbm = np.empty(nb, dtype=np.float32)
    sbt = np.empty(nb, dtype=np.float32)
    rmu = np.empty(n_bins, dtype=np.float64)
    rtn = np.empty(n_bins, dtype=np.float64)
    node_t = np.empty(max_nodes, dtype=np.float64)
    node_w = np.empty(max_nodes, dtype=np.float64)
    n_used = 0

    for i in range(nc):
        q = float(q_active[i])
        tobs = float(t_active[i])
        inv_sigma = float(inv_sigma_active[i])
        if q <= 0.0 or (not math.isfinite(tobs)) or inv_sigma <= 0.0:
            continue

        nvalid = 0
        for j in range(nb):
            m = float(base_mu[j, i])
            tt = float(base_t[j, i])
            if m <= 0.0 or (not math.isfinite(m)) or (not math.isfinite(tt)):
                continue
            k = nvalid
            while k > 0 and tt < float(sbt[k - 1]):
                sbt[k] = sbt[k - 1]
                sbm[k] = sbm[k - 1]
                k -= 1
            sbt[k] = tt
            sbm[k] = m
            nvalid += 1

        for b in range(n_bins):
            rmu[b] = 0.0
            rtn[b] = 0.0
        ref_total = 0.0
        for p in range(npatch):
            m = float(ref_u[p]) * float(transfer_active[i, p])
            if m <= 0.0:
                continue
            tt = float(ref_tbase[p]) + float(time_offset_active[i, p])
            b = int((tt - tmin) * inv_span_bins)
            if b < 0:
                b = 0
            elif b >= n_bins:
                b = n_bins - 1
            rmu[b] += m
            rtn[b] += m * tt
            ref_total += m

        total = ref_total
        for j in range(nvalid):
            total += float(sbm[j])
        if total <= 0.0 or (not math.isfinite(total)):
            penalty = -math.log(1.0e-300)
            for it in range(nt):
                out[it] += penalty
            n_used += 1
            continue

        neff = q / output_efficiency if output_efficiency > 0.0 else q
        if neff < 1.0e-6:
            neff = 1.0e-6
        base_total = 0.0
        if reflection_occupancy_mix:
            for j in range(nvalid):
                base_total += float(sbm[j])

        remaining = 1.0
        remaining_power = 1.0
        ib = 0
        ir = 0
        nn = 0
        while ib < nvalid or ir < n_bins:
            while ir < n_bins and rmu[ir] <= 0.0:
                ir += 1
            if ib >= nvalid and ir >= n_bins:
                break
            take_base = False
            if ir >= n_bins:
                take_base = True
            elif ib < nvalid:
                rt = rtn[ir] / rmu[ir]
                if float(sbt[ib]) <= rt:
                    take_base = True
            if take_base:
                mnode = float(sbm[ib])
                tau = float(sbt[ib])
                ib += 1
            else:
                mnode = rmu[ir]
                tau = rtn[ir] / mnode
                ir += 1
            pnode = mnode / total
            next_remaining = remaining - pnode
            if next_remaining < 0.0:
                next_remaining = 0.0
            next_power = next_remaining ** neff
            w = remaining_power - next_power
            remaining = next_remaining
            remaining_power = next_power
            if w <= 0.0 or (not math.isfinite(w)):
                if remaining_power <= 1.0e-300:
                    break
                continue
            node_t[nn] = tau
            node_w[nn] = w
            nn += 1

        trust = 1.0
        unresolved_density = 0.0
        if reflection_occupancy_mix:
            mu_base = max(float(node_pe_scale) * base_total, 0.0)
            mu_ref = max(float(node_pe_scale) * ref_total, 0.0)
            p_ref = -math.expm1(-mu_ref)
            support_scale = max(float(direct_support_scale_pe), 1.0e-12)
            direct_gate = mu_base / (mu_base + support_scale)
            trust = direct_gate + (1.0 - direct_gate) * p_ref
            unresolved_density = 1.0 / max(prompt_hi - prompt_lo, 1.0e-12)

        for it in range(nt):
            shift = float(t0_values[it])
            shifted_obs = tobs - shift
            shifted_lo = prompt_lo - shift
            shifted_hi = prompt_hi - shift
            mix = 0.0
            acceptance = 0.0
            sum_w = 0.0
            for j in range(nn):
                w = node_w[j]
                tau = node_t[j]
                z = (shifted_obs - tau) * inv_sigma
                gpdf = _first_arrival_exp_lut(z)
                if gpdf > 0.0:
                    mix += w * gpdf * inv_sigma * inv_sqrt_2pi
                if use_window:
                    zhi = (shifted_hi - tau) * inv_sigma
                    zlo = (shifted_lo - tau) * inv_sigma
                    a = _normal_interval_probability_stable(zlo, zhi)
                    if a > 0.0 and math.isfinite(a):
                        acceptance += w * a
                sum_w += w
            if use_window:
                full_density = (
                    0.0 if (acceptance <= 0.0 or mix <= 0.0)
                    else mix / acceptance
                )
            else:
                full_density = (
                    0.0 if (sum_w <= 0.0 or mix <= 0.0)
                    else mix / sum_w
                )
            density = (
                trust * full_density + (1.0 - trust) * unresolved_density
                if reflection_occupancy_mix else full_density
            )
            out[it] += -math.log(max(density, 1.0e-300))
        n_used += 1

    if n_used == 0:
        for it in range(nt):
            out[it] = 1.0e30
    return out

def _prepare_first_arrival_observations(obs_pes, obs_ts, active, output_efficiency):
    """Prepare event observations without an unsafe identity cache.

    The previous speed prototype keyed this data by ``id(array)``.  That is not
    safe across an event loop because Python may recycle object IDs, and it is
    not safe when a caller reuses a writable array buffer.  Re-indexing the
    active PMTs and evaluating the small TTS interpolation costs only a few
    microseconds per FCN, so production correctness takes priority here.
    """
    q_all = np.asarray(obs_pes, dtype=np.float64)
    t_all = np.asarray(obs_ts, dtype=np.float64)
    active = np.ascontiguousarray(active, dtype=np.int32)
    q = np.ascontiguousarray(q_all[active], dtype=np.float64)
    tt = np.ascontiguousarray(t_all[active], dtype=np.float64)
    sig = _wcsim_wcte_first_digit_sigma_ns(q, output_efficiency)
    inv = np.divide(
        1.0, sig, out=np.zeros_like(sig), where=np.isfinite(sig) & (sig > 0.0)
    )
    return q, tt, np.ascontiguousarray(inv, dtype=np.float64)

def _first_arrival_prediction_nll(
    prediction, obs_pes, obs_ts, *, prompt_lo=0.0, prompt_hi=17.0,
    output_efficiency=0.985, reflection_occupancy_mix=False,
    direct_support_scale_pe=1.0e-4, model_time_shift_ns=0.0,
):
    """Conditional first-photoelectron NLL from embedded deterministic nodes.

    ``model_time_shift_ns`` applies the additive track/event time without
    materializing shifted copies of every source-resolved arrival array.
    Translating the model by ``dt`` is exactly equivalent to evaluating the
    zero-time prediction against ``t_obs-dt`` and the translated prompt window.
    """
    model_shift = float(model_time_shift_ns)
    if not np.isfinite(model_shift):
        return 1.0e30
    prompt_lo = float(prompt_lo) - model_shift
    prompt_hi = float(prompt_hi) - model_shift
    active = np.ascontiguousarray(prediction.first_arrival_active_indices, dtype=np.int64)
    _dbm = getattr(prediction, "first_arrival_deferred_base_mu", None)
    _dbt = getattr(prediction, "first_arrival_deferred_base_t", None)
    _ru = getattr(prediction, "first_arrival_reflection_u", None)
    _rtb = getattr(prediction, "first_arrival_reflection_tbase", None)
    _rtr = getattr(prediction, "first_arrival_reflection_transfer_active", None)
    _rto = getattr(prediction, "first_arrival_reflection_time_offset_active", None)
    _node_pe_scale = getattr(prediction, "first_arrival_node_pe_scale", 1.0)
    if _node_pe_scale is None or not np.isfinite(float(_node_pe_scale)) or float(_node_pe_scale) < 0.0:
        return 1.0e30
    if (_dbm is not None and _dbt is not None and _ru is not None
            and _rtb is not None and _rtr is not None and _rto is not None):
        q_all=np.asarray(obs_pes,dtype=np.float64);t_all=np.asarray(obs_ts,dtype=np.float64) - model_shift
        q_active,t_active,inv_sigma=_prepare_first_arrival_observations(
            q_all,t_all,active,float(output_efficiency))
        return float(_first_arrival_deferred_reflection_nll_numba(
            np.ascontiguousarray(_dbm,dtype=np.float32),
            np.ascontiguousarray(_dbt,dtype=np.float32),
            np.ascontiguousarray(_ru,dtype=np.float64),
            np.ascontiguousarray(_rtb,dtype=np.float64),
            np.ascontiguousarray(_rtr,dtype=np.float32),
            np.ascontiguousarray(_rto,dtype=np.float32),
            np.ascontiguousarray(getattr(prediction,"first_arrival_reflection_patch_min_time_offset"),dtype=np.float32),
            np.ascontiguousarray(getattr(prediction,"first_arrival_reflection_patch_max_time_offset"),dtype=np.float32),
            int(getattr(prediction,"first_arrival_reflection_n_bins")),
            q_active,t_active,inv_sigma,float(output_efficiency),
            float(prompt_lo),float(prompt_hi),float(_node_pe_scale),bool(reflection_occupancy_mix),float(direct_support_scale_pe),
        ))
    tau = np.ascontiguousarray(prediction.first_arrival_node_t)
    q_all = np.ascontiguousarray(obs_pes, dtype=np.float64)
    t_all = np.ascontiguousarray(np.asarray(obs_ts, dtype=np.float64) - model_shift)
    weights = getattr(prediction, "first_arrival_node_weight", None)
    weight_eff = getattr(
        prediction, "first_arrival_weight_output_efficiency", None
    )
    if (
        weights is not None and weight_eff is not None
        and abs(float(weight_eff) - float(output_efficiency)) <= 1.0e-12
    ):
        weights = np.ascontiguousarray(weights)
        if weights.ndim != 2 or tau.shape != weights.shape or active.size != weights.shape[1]:
            return 1e30
        q_active, t_active, inv_sigma = _prepare_first_arrival_observations(
            q_all, t_all, active, float(output_efficiency)
        )
        return float(_first_arrival_weighted_prepared_nll_numba(
            weights, tau, q_active, t_active, inv_sigma,
            float(prompt_lo), float(prompt_hi),
        ))

    mu = np.ascontiguousarray(prediction.first_arrival_node_mu)
    if mu.ndim != 2 or tau.shape != mu.shape or active.size != mu.shape[1]:
        return 1e30
    return float(_first_arrival_nodes_nll_numba(
        mu, tau, active, q_all, t_all,
        float(prompt_lo), float(prompt_hi), float(output_efficiency),
    ))



def _first_arrival_prediction_nll_many(
    prediction, obs_pes, obs_ts, model_time_shifts_ns, *,
    prompt_lo=0.0, prompt_hi=17.0, output_efficiency=0.985,
    reflection_occupancy_mix=False, direct_support_scale_pe=1.0e-4,
):
    """Vectorized exact first-arrival NLL for additive t0 hypotheses.

    The deferred-reflection production representation has a dedicated compiled
    grid kernel.  Less common timing representations retain exact scalar
    evaluation as a compatibility fallback.
    """
    shifts = np.ascontiguousarray(model_time_shifts_ns, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(shifts)):
        return np.full(shifts.size, 1.0e30, dtype=np.float64)
    if shifts.size == 0:
        return np.empty(0, dtype=np.float64)

    active = np.ascontiguousarray(
        prediction.first_arrival_active_indices, dtype=np.int32
    )
    dbm = getattr(prediction, "first_arrival_deferred_base_mu", None)
    dbt = getattr(prediction, "first_arrival_deferred_base_t", None)
    ru = getattr(prediction, "first_arrival_reflection_u", None)
    rtb = getattr(prediction, "first_arrival_reflection_tbase", None)
    rtr = getattr(prediction, "first_arrival_reflection_transfer_active", None)
    rto = getattr(prediction, "first_arrival_reflection_time_offset_active", None)
    node_pe_scale = getattr(prediction, "first_arrival_node_pe_scale", 1.0)
    if (
        node_pe_scale is None or not np.isfinite(float(node_pe_scale))
        or float(node_pe_scale) < 0.0
    ):
        return np.full(shifts.size, 1.0e30, dtype=np.float64)

    if (
        dbm is not None and dbt is not None and ru is not None
        and rtb is not None and rtr is not None and rto is not None
    ):
        q_active, t_active, inv_sigma = _prepare_first_arrival_observations(
            np.asarray(obs_pes, dtype=np.float64),
            np.asarray(obs_ts, dtype=np.float64),
            active,
            float(output_efficiency),
        )
        return np.asarray(
            _first_arrival_deferred_reflection_t0_grid_nll_numba(
                np.ascontiguousarray(dbm, dtype=np.float32),
                np.ascontiguousarray(dbt, dtype=np.float32),
                np.ascontiguousarray(ru, dtype=np.float64),
                np.ascontiguousarray(rtb, dtype=np.float64),
                np.ascontiguousarray(rtr, dtype=np.float32),
                np.ascontiguousarray(rto, dtype=np.float32),
                np.ascontiguousarray(
                    getattr(
                        prediction,
                        "first_arrival_reflection_patch_min_time_offset",
                    ),
                    dtype=np.float32,
                ),
                np.ascontiguousarray(
                    getattr(
                        prediction,
                        "first_arrival_reflection_patch_max_time_offset",
                    ),
                    dtype=np.float32,
                ),
                int(getattr(prediction, "first_arrival_reflection_n_bins")),
                q_active,
                t_active,
                inv_sigma,
                shifts,
                float(output_efficiency),
                float(prompt_lo),
                float(prompt_hi),
                float(node_pe_scale),
                bool(reflection_occupancy_mix),
                float(direct_support_scale_pe),
            ),
            dtype=np.float64,
        )

    # Compatibility path for first-arrival predictions without deferred
    # reflection.  It is deliberately scalar because those configurations are
    # not the current cosmic production hot path.
    return np.asarray([
        _first_arrival_prediction_nll(
            prediction,
            obs_pes,
            obs_ts,
            prompt_lo=float(prompt_lo),
            prompt_hi=float(prompt_hi),
            output_efficiency=float(output_efficiency),
            reflection_occupancy_mix=bool(reflection_occupancy_mix),
            direct_support_scale_pe=float(direct_support_scale_pe),
            model_time_shift_ns=float(shift),
        )
        for shift in shifts
    ], dtype=np.float64)

def _has_first_arrival_prediction(exp_ts):
    return bool(getattr(exp_ts, "first_arrival_model", False))


class PMT:
    """
    PMT response model used by the fitter.

    The hot likelihood path is now handled by a compiled helper.  The public API
    is unchanged, so existing fit scripts can keep calling the same methods.
    """

    def __init__(
        self,
        single_pe_amp_mean,
        single_pe_amp_std,
        single_pe_time_std,
        separation_time,
        amp_threshold,
        noise_rate,
    ):
        if not isinstance(single_pe_amp_mean, (int, float)) or single_pe_amp_mean <= 0:
            raise ValueError("single_pe_amp_mean must be a positive number")
        if not isinstance(single_pe_amp_std, (int, float)) or single_pe_amp_std <= 0:
            raise ValueError("single_pe_amp_std must be a positive number")
        if not isinstance(single_pe_time_std, (int, float)) or single_pe_time_std <= 0:
            raise ValueError("single_pe_time_std must be a positive number")
        if not isinstance(separation_time, (int, float)) or separation_time <= 0:
            raise ValueError("separation_time must be a positive number")
        if not isinstance(amp_threshold, (int, float)) or amp_threshold < 0:
            raise ValueError("amp_threshold must be a non-negative number")
        if not isinstance(noise_rate, (int, float)) or noise_rate < 0:
            raise ValueError("noise_rate must be a non-negative number")

        self.single_pe_amp_mean = float(single_pe_amp_mean)
        self.single_pe_amp_std = float(single_pe_amp_std)
        self.single_pe_time_std = float(single_pe_time_std)
        self.separation_time = float(separation_time)
        self.amp_threshold = float(amp_threshold)
        self.noise_rate = float(noise_rate)

        # Probability that a single PE falls below threshold.
        z = (self.amp_threshold - self.single_pe_amp_mean) / self.single_pe_amp_std
        self.prob01 = _norm_cdf(z)

        # Precompute the small-PE charge response exactly once.
        self.charge_response = self.precalculate_charge_response()

        # Timing likelihood policy.  The default is the split charge/time policy
        # that performed best in the 300 MeV noE/noScat diagnostic and is also
        # physically conservative: a PMT cannot contribute more timing information
        # than either the data or the physical model expectation supports.
        #
        # Environment overrides:
        #   PMT_TIMING_POLICY=current_obs          # exact old behavior
        #   PMT_TIMING_POLICY=obs_gated            # obs weight, gated by model PE
        #   PMT_TIMING_POLICY=predicted            # model PE weight
        #   PMT_TIMING_POLICY=min_obs_model        # default
        #   PMT_TIMING_POLICY=harmonic             # obs*model/(obs+model)
        #   PMT_TIMING_MU_MIN_PE=0.1               # optional model-light gate
        self.timing_likelihood_policy = os.environ.get(
            "PMT_TIMING_POLICY",
            os.environ.get("TIMING_LIKELIHOOD_POLICY", "min_obs_model"),
        ).strip().lower()
        self.timing_mu_min_pe = float(os.environ.get("PMT_TIMING_MU_MIN_PE", "0.0"))
        self.timing_sigma_sys_ns = float(os.environ.get("PMT_TIMING_SIGMA_SYS_NS", "0.0"))
        # Required for model-dependent timing weights (min_obs_model, harmonic,
        # predicted).  Set to 0 only for exact A/B comparison with the previous
        # split-timing implementation.
        self.timing_include_lognorm = str(os.environ.get(
            "PMT_TIMING_INCLUDE_LOGNORM", "1"
        )).strip().lower() not in {"0", "false", "no", "off"}
        self.first_arrival_prompt_min_ns = float(os.environ.get("PMT_FIRST_ARRIVAL_PROMPT_MIN_NS", "0.0"))
        self.first_arrival_prompt_max_ns = float(os.environ.get("PMT_FIRST_ARRIVAL_PROMPT_MAX_NS", "17.0"))
        self.first_arrival_output_efficiency = float(os.environ.get("PMT_FIRST_ARRIVAL_OUTPUT_EFFICIENCY", "0.985"))
        self.first_arrival_reflection_occupancy_mix = str(os.environ.get("PMT_FIRST_ARRIVAL_REFLECTION_OCCUPANCY_MIX", "1")).strip().lower() not in {"0", "false", "off", "no"}
        self.first_arrival_direct_support_scale_pe = float(os.environ.get("PMT_FIRST_ARRIVAL_DIRECT_SUPPORT_SCALE_PE", "1e-4"))

    def __repr__(self):
        return (
            f"PMT(single_pe_amp_mean={self.single_pe_amp_mean}, "
            f"single_pe_amp_std={self.single_pe_amp_std}, "
            f"single_pe_time_std={self.single_pe_time_std}, "
            f"separation_time={self.separation_time}, "
            f"amp_threshold={self.amp_threshold}, "
            f"noise_rate={self.noise_rate})"
        )

    def precalculate_charge_response(self):
        """
        Precompute the small-PE charge response grid used by q+t fits.

        This method runs only during PMT construction, so clarity matters more
        than micro-optimizing every line here.
        """
        n_pes_max = 8
        n_bins = 50
        threshold_ope = self.amp_threshold / self.single_pe_amp_mean

        response = np.zeros((n_pes_max, n_bins), dtype=np.float64)

        for npe in range(1, n_pes_max + 1):
            mean = float(npe)
            std = self.single_pe_amp_std * np.sqrt(npe) / self.single_pe_amp_mean

            for ope10 in range(n_bins):
                ope_low = ope10 / 10.0
                ope_high = ope_low + 0.1

                if ope_high <= threshold_ope:
                    continue

                z_low = (ope_low - mean) / std
                z_high = (ope_high - mean) / std
                prob = _norm_cdf(z_high) - _norm_cdf(z_low)
                response[npe - 1, ope10] = prob

        return response

    def add_noise(self, simulated_event):
        """
        Add dark-noise hits to a simulated event.

        This is not part of the fitter hot path, so the implementation stays
        close to the original for readability.
        """
        min_time = float("inf")
        max_time = float("-inf")

        for i_mpmt in range(simulated_event.n_mpmt):
            if not simulated_event.mpmt_status[i_mpmt]:
                continue
            for i_pmt in range(simulated_event.npmt_per_mpmt):
                if not simulated_event.pmt_status[i_mpmt][i_pmt]:
                    continue
                for exp_hit_time in simulated_event.expected_hit_times[i_mpmt][i_pmt]:
                    if exp_hit_time < min_time:
                        min_time = exp_hit_time
                    if exp_hit_time > max_time:
                        max_time = exp_hit_time

        min_time -= 10.0
        max_time += 10.0
        expected_n_noise_hits = self.noise_rate * (max_time - min_time)

        for i_mpmt in range(simulated_event.n_mpmt):
            if not simulated_event.mpmt_status[i_mpmt]:
                continue
            for i_pmt in range(simulated_event.npmt_per_mpmt):
                if not simulated_event.pmt_status[i_mpmt][i_pmt]:
                    continue
                n_noise_hits = np.random.poisson(expected_n_noise_hits)
                for _ in range(n_noise_hits):
                    noise_time = np.random.uniform(min_time, max_time)
                    simulated_event.noise_hit_times[i_mpmt][i_pmt].append(noise_time)
                    simulated_event.noise_hit_pe[i_mpmt][i_pmt].append(1)

    def apply_response(self, simulated_event):
        """
        Apply the electronics response to a simulated event.

        This method is still mostly simulation-side code, so it is kept close to
        the original implementation.
        """
        for i_mpmt in range(simulated_event.n_mpmt):
            if not simulated_event.mpmt_status[i_mpmt]:
                continue
            for i_pmt in range(simulated_event.npmt_per_mpmt):
                if not simulated_event.pmt_status[i_mpmt][i_pmt]:
                    continue

                all_hit_times = (
                    simulated_event.expected_hit_times[i_mpmt][i_pmt]
                    + simulated_event.noise_hit_times[i_mpmt][i_pmt]
                )
                all_hit_pe = (
                    simulated_event.true_hit_pe[i_mpmt][i_pmt]
                    + simulated_event.noise_hit_pe[i_mpmt][i_pmt]
                )

                if len(all_hit_times) == 0:
                    continue

                if len(all_hit_times) > 1:
                    sorted_indices = np.argsort(all_hit_times)
                    sorted_hit_times = [all_hit_times[i] for i in sorted_indices]
                    sorted_hit_pe = [all_hit_pe[i] for i in sorted_indices]

                    merged_hit_times = []
                    merged_hit_pe = []

                    current_hit_time = sorted_hit_times[0]
                    current_hit_pe = sorted_hit_pe[0]

                    for j in range(1, len(sorted_hit_times)):
                        if sorted_hit_times[j] - current_hit_time < self.separation_time:
                            total_pe = current_hit_pe + sorted_hit_pe[j]
                            current_hit_time = (
                                current_hit_time * current_hit_pe
                                + sorted_hit_times[j] * sorted_hit_pe[j]
                            ) / total_pe
                            current_hit_pe = total_pe
                        else:
                            merged_hit_times.append(current_hit_time)
                            merged_hit_pe.append(current_hit_pe)
                            current_hit_time = sorted_hit_times[j]
                            current_hit_pe = sorted_hit_pe[j]

                    merged_hit_times.append(current_hit_time)
                    merged_hit_pe.append(current_hit_pe)
                else:
                    merged_hit_times = all_hit_times
                    merged_hit_pe = all_hit_pe

                for k in range(len(merged_hit_times)):
                    true_pe = merged_hit_pe[k]
                    if true_pe <= 0:
                        continue

                    amp = np.random.normal(
                        true_pe * self.single_pe_amp_mean,
                        self.single_pe_amp_std * np.sqrt(true_pe),
                    )
                    if amp > self.amp_threshold:
                        time = np.random.normal(
                            merged_hit_times[k],
                            self.single_pe_time_std / np.sqrt(true_pe),
                        )
                        simulated_event.hit_charges[i_mpmt][i_pmt].append(amp)
                        simulated_event.hit_times[i_mpmt][i_pmt].append(time)

    def _timing_policy_code(self):
        policy = str(getattr(self, "timing_likelihood_policy", "min_obs_model")).strip().lower()
        aliases = {
            "legacy": 0,
            "current": 0,
            "current_obs": 0,
            "current_floor_obsweight_allobstimed": 0,
            "obs": 0,

            "obs_gated": 1,
            "observed_gated": 1,
            "obs_weight_gated": 1,
            "floor_obsweight_rawgt0p10": 1,
            "floor_obsweight_rawgt0p20": 1,

            "pred": 2,
            "predicted": 2,
            "model": 2,
            "model_weight": 2,
            "floor_predweight_rawgt0": 2,

            "min": 3,
            "min_obs_model": 3,
            "min_obs_raw": 3,
            "minobsraw": 3,
            "floor_minobsrawweight_rawgt0": 3,

            "harmonic": 4,
            "harmonic_weight": 4,
            "floor_harmonicweight_rawgt0": 4,
        }
        if policy not in aliases:
            raise ValueError(
                "Unknown PMT_TIMING_POLICY=%r. Use one of: current_obs, "
                "obs_gated, predicted, min_obs_model, harmonic." % policy
            )
        return int(aliases[policy])

    def _prepare_timing_pes(self, exp_pes, timing_pes):
        if timing_pes is None:
            # Backward-compatible fallback for callers not yet updated to pass
            # Emitter._last_expected_pes_for_timing.  Updated fit drivers pass the
            # unfloored physical expectation explicitly.
            timing_pes = exp_pes
        return np.asarray(timing_pes, dtype=np.float64)

    @staticmethod
    def _valid_pe_arrays(exp_pes, obs_pes):
        exp = np.asarray(exp_pes, dtype=np.float64)
        obs = np.asarray(obs_pes, dtype=np.float64)
        valid = (
            exp.ndim == 1
            and obs.shape == exp.shape
            and np.all(np.isfinite(exp))
            and np.all(np.isfinite(obs))
            and np.all(exp >= 0.0)
            and np.all(obs >= 0.0)
        )
        return exp, obs, bool(valid)

    @staticmethod
    def _valid_observed_times(obs_ts, shape):
        times = np.asarray(obs_ts, dtype=np.float64)
        # NaN is the intentional no-timestamp sentinel; infinities are never a
        # valid observation and usually indicate an upstream arithmetic error.
        valid = times.shape == shape and not np.any(np.isinf(times))
        return times, bool(valid)

    def _model_time_eligibility_mask(
        self, exp_pes, obs_pes, obs_ts, timing_pes
    ):
        """Return PMTs for which a finite model timestamp is required.

        NaN model times are a legitimate sentinel on model-unlit PMTs. They
        must not silently remove a timing pull on a PMT where both the data and
        the configured timing-weight policy support light.
        """
        policy_code = self._timing_policy_code()
        observed_time = (obs_pes > 0.0) & np.isfinite(obs_ts)
        if policy_code == 0:
            return observed_time & (exp_pes > 0.0)
        return observed_time & (
            timing_pes > float(getattr(self, "timing_mu_min_pe", 0.0))
        )

    def _valid_model_times(
        self, exp_ts, exp_pes, obs_pes, obs_ts, timing_pes
    ):
        times = np.asarray(exp_ts, dtype=np.float64)
        if times.shape != exp_pes.shape or np.any(np.isinf(times)):
            return times, False
        eligible = self._model_time_eligibility_mask(
            exp_pes, obs_pes, obs_ts, timing_pes
        )
        return times, bool(np.all(np.isfinite(times[eligible])))

    def get_neg_log_likelihood_npe(self, exp_pes, obs_pes):
        exp_pes, obs_pes, valid = self._valid_pe_arrays(exp_pes, obs_pes)
        if not valid:
            return 1.0e30
        return float(_poisson_nll(exp_pes, obs_pes))

    def get_neg_log_likelihood_npe_t(self, exp_pes, obs_pes, exp_ts, obs_ts, timing_pes=None, model_time_shift_ns=0.0):
        """Charge+time NLL with split charge and timing expectations by default.

        Parameters
        ----------
        exp_pes : array
            Charge-likelihood expectation.  This may be floored.
        timing_pes : array, optional
            Unfloored physical expected PE used for timing eligibility/weighting.
            Updated fit drivers pass Emitter._last_expected_pes_for_timing here.

        Default policy: PMT_TIMING_POLICY=min_obs_model, i.e.
            w_time = min(obs_pes, timing_pes).
        Set PMT_TIMING_POLICY=current_obs to recover the old timing behavior.
        """
        timing_prediction = exp_ts
        exp_pes, obs_pes, valid = self._valid_pe_arrays(exp_pes, obs_pes)
        obs_ts, valid_times = self._valid_observed_times(obs_ts, exp_pes.shape)
        if not (valid and valid_times):
            return 1.0e30
        if _has_first_arrival_prediction(timing_prediction):
            return float(
                _poisson_nll(exp_pes, obs_pes)
                + _first_arrival_prediction_nll(
                    timing_prediction, obs_pes, obs_ts,
                    prompt_lo=float(self.first_arrival_prompt_min_ns),
                    prompt_hi=float(self.first_arrival_prompt_max_ns),
                    output_efficiency=float(self.first_arrival_output_efficiency),
                    reflection_occupancy_mix=bool(self.first_arrival_reflection_occupancy_mix),
                    direct_support_scale_pe=float(self.first_arrival_direct_support_scale_pe),
                    model_time_shift_ns=float(model_time_shift_ns),
                )
            )
        model_time_shift_ns = float(model_time_shift_ns)
        if not np.isfinite(model_time_shift_ns):
            return 1.0e30
        if model_time_shift_ns != 0.0:
            obs_ts = np.asarray(obs_ts, dtype=np.float64) - model_time_shift_ns
        timing_pes = self._prepare_timing_pes(exp_pes, timing_pes)
        if (
            timing_pes.shape != exp_pes.shape
            or np.any(~np.isfinite(timing_pes))
            or np.any(timing_pes < 0.0)
        ):
            return 1.0e30
        exp_ts, valid_model_times = self._valid_model_times(
            exp_ts, exp_pes, obs_pes, obs_ts, timing_pes
        )
        if not valid_model_times:
            return 1.0e30

        return float(
            _poisson_time_nll_split(
                exp_pes,
                obs_pes,
                exp_ts,
                obs_ts,
                timing_pes,
                float(self.single_pe_time_std),
                self._timing_policy_code(),
                float(getattr(self, "timing_mu_min_pe", 0.0)),
                float(getattr(self, "timing_sigma_sys_ns", 0.0)),
                int(1 if getattr(self, "timing_include_lognorm", True) else 0),
            )
        )

    def get_neg_log_likelihood_npe_t_many_t0(
        self, exp_pes, obs_pes, exp_ts, obs_ts, model_time_shifts_ns,
        timing_pes=None,
    ):
        """Exact charge-plus-time NLL for a vector of additive t0 values."""
        shifts = np.ascontiguousarray(model_time_shifts_ns, dtype=np.float64).reshape(-1)
        timing_prediction = exp_ts
        exp_pes, obs_pes, valid = self._valid_pe_arrays(exp_pes, obs_pes)
        obs_ts, valid_times = self._valid_observed_times(obs_ts, exp_pes.shape)
        if not (valid and valid_times) or np.any(~np.isfinite(shifts)):
            return np.full(shifts.size, 1.0e30, dtype=np.float64)
        if _has_first_arrival_prediction(timing_prediction):
            charge = float(_poisson_nll(exp_pes, obs_pes))
            timing = _first_arrival_prediction_nll_many(
                timing_prediction,
                obs_pes,
                obs_ts,
                shifts,
                prompt_lo=float(self.first_arrival_prompt_min_ns),
                prompt_hi=float(self.first_arrival_prompt_max_ns),
                output_efficiency=float(self.first_arrival_output_efficiency),
                reflection_occupancy_mix=bool(
                    self.first_arrival_reflection_occupancy_mix
                ),
                direct_support_scale_pe=float(
                    self.first_arrival_direct_support_scale_pe
                ),
            )
            return np.asarray(timing, dtype=np.float64) + charge
        return np.asarray([
            self.get_neg_log_likelihood_npe_t(
                exp_pes, obs_pes, exp_ts, obs_ts,
                timing_pes=timing_pes,
                model_time_shift_ns=float(shift),
            )
            for shift in shifts
        ], dtype=np.float64)

    def get_neg_log_likelihood_t_many_t0(
        self, exp_pes, obs_pes, exp_ts, obs_ts, model_time_shifts_ns,
        timing_pes=None,
    ):
        """Exact timing-only NLL for a vector of additive t0 values."""
        shifts = np.ascontiguousarray(model_time_shifts_ns, dtype=np.float64).reshape(-1)
        timing_prediction = exp_ts
        exp_pes, obs_pes, valid = self._valid_pe_arrays(exp_pes, obs_pes)
        obs_ts, valid_times = self._valid_observed_times(obs_ts, exp_pes.shape)
        if not (valid and valid_times) or np.any(~np.isfinite(shifts)):
            return np.full(shifts.size, 1.0e30, dtype=np.float64)
        if _has_first_arrival_prediction(timing_prediction):
            return _first_arrival_prediction_nll_many(
                timing_prediction,
                obs_pes,
                obs_ts,
                shifts,
                prompt_lo=float(self.first_arrival_prompt_min_ns),
                prompt_hi=float(self.first_arrival_prompt_max_ns),
                output_efficiency=float(self.first_arrival_output_efficiency),
                reflection_occupancy_mix=bool(
                    self.first_arrival_reflection_occupancy_mix
                ),
                direct_support_scale_pe=float(
                    self.first_arrival_direct_support_scale_pe
                ),
            )
        return np.asarray([
            self.get_neg_log_likelihood_t(
                exp_pes, obs_pes, exp_ts, obs_ts,
                timing_pes=timing_pes,
                model_time_shift_ns=float(shift),
            )
            for shift in shifts
        ], dtype=np.float64)

    def get_neg_log_likelihood_t(self, exp_pes, obs_pes, exp_ts, obs_ts, timing_pes=None, model_time_shift_ns=0.0):
        """Timing-only NLL using the same split timing policy as npe+t."""
        timing_prediction = exp_ts
        exp_pes, obs_pes, valid = self._valid_pe_arrays(exp_pes, obs_pes)
        obs_ts, valid_times = self._valid_observed_times(obs_ts, exp_pes.shape)
        if not (valid and valid_times):
            return 1.0e30
        if _has_first_arrival_prediction(timing_prediction):
            return _first_arrival_prediction_nll(
                timing_prediction, obs_pes, obs_ts,
                prompt_lo=float(self.first_arrival_prompt_min_ns),
                prompt_hi=float(self.first_arrival_prompt_max_ns),
                output_efficiency=float(self.first_arrival_output_efficiency),
                reflection_occupancy_mix=bool(self.first_arrival_reflection_occupancy_mix),
                direct_support_scale_pe=float(self.first_arrival_direct_support_scale_pe),
                model_time_shift_ns=float(model_time_shift_ns),
            )
        model_time_shift_ns = float(model_time_shift_ns)
        if not np.isfinite(model_time_shift_ns):
            return 1.0e30
        if model_time_shift_ns != 0.0:
            obs_ts = np.asarray(obs_ts, dtype=np.float64) - model_time_shift_ns
        timing_pes = self._prepare_timing_pes(exp_pes, timing_pes)
        if (
            timing_pes.shape != exp_pes.shape
            or np.any(~np.isfinite(timing_pes))
            or np.any(timing_pes < 0.0)
        ):
            return 1.0e30
        exp_ts, valid_model_times = self._valid_model_times(
            exp_ts, exp_pes, obs_pes, obs_ts, timing_pes
        )
        if not valid_model_times:
            return 1.0e30
        return float(
            _time_nll_split_only(
                exp_pes,
                obs_pes,
                exp_ts,
                obs_ts,
                timing_pes,
                float(self.single_pe_time_std),
                self._timing_policy_code(),
                float(getattr(self, "timing_mu_min_pe", 0.0)),
                float(getattr(self, "timing_sigma_sys_ns", 0.0)),
                int(1 if getattr(self, "timing_include_lognorm", True) else 0),
            )
        )

    def get_neg_log_likelihood_q_t(self, exp_pes, obs_qs, exp_ts, obs_ts, timing_pes=None):
        """
        Charge+time likelihood.  This path is typically not the bottleneck for
        your current fits, so it remains close to the original implementation.
        """
        timing_prediction = exp_ts
        exp_pes = np.asarray(exp_pes, dtype=np.float64)
        if exp_pes.ndim != 1 or np.any(~np.isfinite(exp_pes)) or np.any(exp_pes < 0.0):
            return 1.0e30
        exp_ts = np.asarray(exp_ts, dtype=np.float64)
        obs_ts, valid_times = self._valid_observed_times(obs_ts, exp_pes.shape)
        if not valid_times or exp_ts.shape != exp_pes.shape or np.any(np.isinf(exp_ts)):
            return 1.0e30

        obs_qs = np.asarray(obs_qs)
        obs_pes = np.zeros(len(obs_qs), dtype=np.float64)

        valid_q = np.isfinite(obs_qs.astype(np.float64, copy=False))
        obs_pes[valid_q] = obs_qs[valid_q].astype(np.float64) / self.single_pe_amp_mean

        mask0 = (exp_pes > 0.0) & (obs_pes == 0.0)
        no_signal_prob = np.exp(-exp_pes[mask0]) + self.prob01 * exp_pes[mask0] * np.exp(-exp_pes[mask0])
        no_signal_nll = -np.log(no_signal_prob + 1e-10)

        mask = (exp_pes > 0.0) & (obs_pes <= 5.0)
        n_pes = np.arange(1, 9, dtype=np.float64)
        exp_pe = exp_pes[mask]
        prob_ns = (
            np.exp(-exp_pe[:, None])
            * exp_pe[:, None] ** n_pes[None, :]
            / np.array([math.factorial(int(n)) for n in n_pes], dtype=np.float64)[None, :]
        )
        observed_ope10 = np.clip((obs_pes[mask] * 10.0).astype(int), 0, self.charge_response.shape[1] - 1)
        pdfs_sums = np.sum(prob_ns * self.charge_response[:, observed_ope10].T, axis=1)
        signal_nllq = -np.log(pdfs_sums + 1e-10)

        mask1 = (exp_pes > 0.0) & (obs_pes > 5.0)
        sigmas = np.sqrt(exp_pes[mask1] + self.single_pe_amp_std**2 * exp_pes[mask1])
        n_sigma = np.abs(obs_pes[mask1] - exp_pes[mask1]) / sigmas
        high_sigma = n_sigma >= 4.0
        low_sigma = ~high_sigma
        high_sigma_nllq = -obs_pes[mask1][high_sigma] * np.log(1e-4)
        low_sigma_nllq = 0.5 * (obs_pes[mask1][low_sigma] - exp_pes[mask1][low_sigma]) ** 2 / sigmas[low_sigma] ** 2

        mask2 = (exp_pes <= 0.0) & (obs_pes > 0.0)
        background_nll = -obs_pes[mask2] * np.log(1e-4)

        if _has_first_arrival_prediction(timing_prediction):
            first_arrival_nll = _first_arrival_prediction_nll(
                timing_prediction, obs_pes, obs_ts,
                prompt_lo=float(self.first_arrival_prompt_min_ns),
                prompt_hi=float(self.first_arrival_prompt_max_ns),
                output_efficiency=float(self.first_arrival_output_efficiency),
                reflection_occupancy_mix=bool(self.first_arrival_reflection_occupancy_mix),
                direct_support_scale_pe=float(self.first_arrival_direct_support_scale_pe),
            )
            return float(
                np.sum(no_signal_nll) + np.sum(signal_nllq)
                + np.sum(high_sigma_nllq) + np.sum(low_sigma_nllq)
                + np.sum(background_nll) + first_arrival_nll
            )

        timing_pes = self._prepare_timing_pes(exp_pes, timing_pes)
        if (
            timing_pes.shape != exp_pes.shape
            or np.any(~np.isfinite(timing_pes))
            or np.any(timing_pes < 0.0)
        ):
            return 1.0e30
        exp_ts, valid_model_times = self._valid_model_times(
            exp_ts, exp_pes, obs_pes, obs_ts, timing_pes
        )
        if not valid_model_times:
            return 1.0e30
        policy_code = self._timing_policy_code()
        if policy_code == 0:
            mask_t = (exp_pes > 0.0) & (obs_pes > 0.0) & np.isfinite(obs_ts) & np.isfinite(exp_ts)
            timing_weight = obs_pes
        else:
            mask_t = (timing_pes > float(getattr(self, "timing_mu_min_pe", 0.0))) & (obs_pes > 0.0) & np.isfinite(obs_ts) & np.isfinite(exp_ts)
            if policy_code == 2:
                timing_weight = timing_pes
            elif policy_code == 3:
                timing_weight = np.minimum(obs_pes, timing_pes)
            elif policy_code == 4:
                timing_weight = obs_pes * timing_pes / (obs_pes + timing_pes + 1e-300)
            else:
                timing_weight = obs_pes
        high_sigma_nllt = np.array([], dtype=np.float64)
        low_sigma_nllt = np.array([], dtype=np.float64)
        if np.any(mask_t):
            sigma_t = np.sqrt(
                self.single_pe_time_std ** 2 / np.maximum(timing_weight[mask_t], 1e-300)
                + float(getattr(self, "timing_sigma_sys_ns", 0.0)) ** 2
            )
            n_sigma_t = np.abs(obs_ts[mask_t] - exp_ts[mask_t]) / sigma_t
            high_sigma_t = n_sigma_t >= 4.0
            low_sigma_t = ~high_sigma_t
            _tw_mask = timing_weight[mask_t]
            high_sigma_nllt = -_tw_mask[high_sigma_t] * np.log(1e-4)
            low_sigma_nllt = 0.5 * (obs_ts[mask_t][low_sigma_t] - exp_ts[mask_t][low_sigma_t]) ** 2 / sigma_t[low_sigma_t] ** 2
            if getattr(self, "timing_include_lognorm", True) and policy_code != 0:
                low_sigma_nllt = low_sigma_nllt + 0.5 * np.log(np.maximum(sigma_t[low_sigma_t] ** 2, 1e-300))

        return float(
            np.sum(no_signal_nll)
            + np.sum(signal_nllq)
            + np.sum(high_sigma_nllq)
            + np.sum(low_sigma_nllq)
            + np.sum(background_nll)
            + np.sum(high_sigma_nllt)
            + np.sum(low_sigma_nllt)
        )







# import math

# import numpy as np
# from numba import njit


# High-resolution deterministic Gaussian lookup tables for the hot first-PE
# likelihood. Linear interpolation on dz=8/32768 has sub-1e-8 absolute error
# for both exp(-z^2/2) and Phi(z), far below the optical-model precision.
_FIRST_ARRIVAL_LUT_N = 32769
_FIRST_ARRIVAL_LUT_ZMAX = 8.0
_FIRST_ARRIVAL_EXP_LUT = np.exp(
    -0.5 * np.linspace(0.0, _FIRST_ARRIVAL_LUT_ZMAX, _FIRST_ARRIVAL_LUT_N) ** 2
).astype(np.float64)
_FIRST_ARRIVAL_CDF_LUT = np.asarray([
    0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))
    for z in np.linspace(-_FIRST_ARRIVAL_LUT_ZMAX, _FIRST_ARRIVAL_LUT_ZMAX, _FIRST_ARRIVAL_LUT_N)
], dtype=np.float64)


@njit(cache=True, inline='always')
def _first_arrival_exp_lut(z):
    az = abs(z)
    if az >= 8.0:
        # Tiny Gaussian tails can still dominate a likelihood mixture when all
        # earlier source nodes are even farther from the observed timestamp.
        # Dropping them at 8 sigma created artificial 1e-300 floors for a small
        # subset of PMTs.  Evaluate the rare tail directly until IEEE-754
        # underflow; the common |z|<8 path remains a fast LUT lookup.
        if az >= 38.5:
            return 0.0
        return math.exp(-0.5 * az * az)
    x = az * ((_FIRST_ARRIVAL_LUT_N - 1) / 8.0)
    i = int(x)
    f = x - i
    return _FIRST_ARRIVAL_EXP_LUT[i] + f * (_FIRST_ARRIVAL_EXP_LUT[i + 1] - _FIRST_ARRIVAL_EXP_LUT[i])


@njit(cache=True, inline='always')
def _first_arrival_cdf_lut(z):
    if z <= -8.0:
        return 0.0
    if z >= 8.0:
        return 1.0
    x = (z + 8.0) * ((_FIRST_ARRIVAL_LUT_N - 1) / 16.0)
    i = int(x)
    f = x - i
    return _FIRST_ARRIVAL_CDF_LUT[i] + f * (_FIRST_ARRIVAL_CDF_LUT[i + 1] - _FIRST_ARRIVAL_CDF_LUT[i])


# @njit(cache=True)
# def _norm_cdf(x):
#     return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# @njit(cache=True)
# def _poisson_time_nll(exp_pes, obs_pes, exp_ts, obs_ts, single_pe_time_std):
#     """
#     Tight likelihood kernel for the npe+t fit.

#     Compared with the original implementation, this avoids repeated boolean-mask
#     allocations and works directly on contiguous float arrays.
#     """
#     n = exp_pes.size
#     nll = 0.0
#     log_noise = math.log(1e-4)

#     for i in range(n):
#         lam = exp_pes[i]
#         obs = obs_pes[i]

#         # Charge term
#         if lam > 0.0:
#             nll += lam - obs * math.log(lam)
#         elif obs > 0.0:
#             nll -= obs * log_noise

#         # Timing term
#         t_obs = obs_ts[i]
#         if (lam > 0.0) and (obs > 0.0) and np.isfinite(t_obs):
#             sigma_t = single_pe_time_std / math.sqrt(obs)
#             dt = (t_obs - exp_ts[i]) / sigma_t
#             nll += 0.5 * dt * dt

#     return nll


# class PMT:
#     """
#     PMT response model used by the fitter.

#     The hot likelihood path is now handled by a compiled helper.  The public API
#     is unchanged, so existing fit scripts can keep calling the same methods.
#     """

#     def __init__(
#         self,
#         single_pe_amp_mean,
#         single_pe_amp_std,
#         single_pe_time_std,
#         separation_time,
#         amp_threshold,
#         noise_rate,
#     ):
#         if not isinstance(single_pe_amp_mean, (int, float)) or single_pe_amp_mean <= 0:
#             raise ValueError("single_pe_amp_mean must be a positive number")
#         if not isinstance(single_pe_amp_std, (int, float)) or single_pe_amp_std <= 0:
#             raise ValueError("single_pe_amp_std must be a positive number")
#         if not isinstance(single_pe_time_std, (int, float)) or single_pe_time_std <= 0:
#             raise ValueError("single_pe_time_std must be a positive number")
#         if not isinstance(separation_time, (int, float)) or separation_time <= 0:
#             raise ValueError("separation_time must be a positive number")
#         if not isinstance(amp_threshold, (int, float)) or amp_threshold < 0:
#             raise ValueError("amp_threshold must be a non-negative number")
#         if not isinstance(noise_rate, (int, float)) or noise_rate < 0:
#             raise ValueError("noise_rate must be a non-negative number")

#         self.single_pe_amp_mean = float(single_pe_amp_mean)
#         self.single_pe_amp_std = float(single_pe_amp_std)
#         self.single_pe_time_std = float(single_pe_time_std)
#         self.separation_time = float(separation_time)
#         self.amp_threshold = float(amp_threshold)
#         self.noise_rate = float(noise_rate)

#         # Probability that a single PE falls below threshold.
#         z = (self.amp_threshold - self.single_pe_amp_mean) / self.single_pe_amp_std
#         self.prob01 = _norm_cdf(z)

#         # Precompute the small-PE charge response exactly once.
#         self.charge_response = self.precalculate_charge_response()

#     def __repr__(self):
#         return (
#             f"PMT(single_pe_amp_mean={self.single_pe_amp_mean}, "
#             f"single_pe_amp_std={self.single_pe_amp_std}, "
#             f"single_pe_time_std={self.single_pe_time_std}, "
#             f"separation_time={self.separation_time}, "
#             f"amp_threshold={self.amp_threshold}, "
#             f"noise_rate={self.noise_rate})"
#         )

#     def precalculate_charge_response(self):
#         """
#         Precompute the small-PE charge response grid used by q+t fits.

#         This method runs only during PMT construction, so clarity matters more
#         than micro-optimizing every line here.
#         """
#         n_pes_max = 8
#         n_bins = 50
#         threshold_ope = self.amp_threshold / self.single_pe_amp_mean

#         response = np.zeros((n_pes_max, n_bins), dtype=np.float64)

#         for npe in range(1, n_pes_max + 1):
#             mean = float(npe)
#             std = self.single_pe_amp_std * np.sqrt(npe) / self.single_pe_amp_mean

#             for ope10 in range(n_bins):
#                 ope_low = ope10 / 10.0
#                 ope_high = ope_low + 0.1

#                 if ope_high <= threshold_ope:
#                     continue

#                 z_low = (ope_low - mean) / std
#                 z_high = (ope_high - mean) / std
#                 prob = _norm_cdf(z_high) - _norm_cdf(z_low)
#                 response[npe - 1, ope10] = prob

#         return response

#     def add_noise(self, simulated_event):
#         """
#         Add dark-noise hits to a simulated event.

#         This is not part of the fitter hot path, so the implementation stays
#         close to the original for readability.
#         """
#         min_time = float("inf")
#         max_time = float("-inf")

#         for i_mpmt in range(simulated_event.n_mpmt):
#             if not simulated_event.mpmt_status[i_mpmt]:
#                 continue
#             for i_pmt in range(simulated_event.npmt_per_mpmt):
#                 if not simulated_event.pmt_status[i_mpmt][i_pmt]:
#                     continue
#                 for exp_hit_time in simulated_event.expected_hit_times[i_mpmt][i_pmt]:
#                     if exp_hit_time < min_time:
#                         min_time = exp_hit_time
#                     if exp_hit_time > max_time:
#                         max_time = exp_hit_time

#         min_time -= 10.0
#         max_time += 10.0
#         expected_n_noise_hits = self.noise_rate * (max_time - min_time)

#         for i_mpmt in range(simulated_event.n_mpmt):
#             if not simulated_event.mpmt_status[i_mpmt]:
#                 continue
#             for i_pmt in range(simulated_event.npmt_per_mpmt):
#                 if not simulated_event.pmt_status[i_mpmt][i_pmt]:
#                     continue
#                 n_noise_hits = np.random.poisson(expected_n_noise_hits)
#                 for _ in range(n_noise_hits):
#                     noise_time = np.random.uniform(min_time, max_time)
#                     simulated_event.noise_hit_times[i_mpmt][i_pmt].append(noise_time)
#                     simulated_event.noise_hit_pe[i_mpmt][i_pmt].append(1)

#     def apply_response(self, simulated_event):
#         """
#         Apply the electronics response to a simulated event.

#         This method is still mostly simulation-side code, so it is kept close to
#         the original implementation.
#         """
#         for i_mpmt in range(simulated_event.n_mpmt):
#             if not simulated_event.mpmt_status[i_mpmt]:
#                 continue
#             for i_pmt in range(simulated_event.npmt_per_mpmt):
#                 if not simulated_event.pmt_status[i_mpmt][i_pmt]:
#                     continue

#                 all_hit_times = (
#                     simulated_event.expected_hit_times[i_mpmt][i_pmt]
#                     + simulated_event.noise_hit_times[i_mpmt][i_pmt]
#                 )
#                 all_hit_pe = (
#                     simulated_event.true_hit_pe[i_mpmt][i_pmt]
#                     + simulated_event.noise_hit_pe[i_mpmt][i_pmt]
#                 )

#                 if len(all_hit_times) == 0:
#                     continue

#                 if len(all_hit_times) > 1:
#                     sorted_indices = np.argsort(all_hit_times)
#                     sorted_hit_times = [all_hit_times[i] for i in sorted_indices]
#                     sorted_hit_pe = [all_hit_pe[i] for i in sorted_indices]

#                     merged_hit_times = []
#                     merged_hit_pe = []

#                     current_hit_time = sorted_hit_times[0]
#                     current_hit_pe = sorted_hit_pe[0]

#                     for j in range(1, len(sorted_hit_times)):
#                         if sorted_hit_times[j] - current_hit_time < self.separation_time:
#                             total_pe = current_hit_pe + sorted_hit_pe[j]
#                             current_hit_time = (
#                                 current_hit_time * current_hit_pe
#                                 + sorted_hit_times[j] * sorted_hit_pe[j]
#                             ) / total_pe
#                             current_hit_pe = total_pe
#                         else:
#                             merged_hit_times.append(current_hit_time)
#                             merged_hit_pe.append(current_hit_pe)
#                             current_hit_time = sorted_hit_times[j]
#                             current_hit_pe = sorted_hit_pe[j]

#                     merged_hit_times.append(current_hit_time)
#                     merged_hit_pe.append(current_hit_pe)
#                 else:
#                     merged_hit_times = all_hit_times
#                     merged_hit_pe = all_hit_pe

#                 for k in range(len(merged_hit_times)):
#                     true_pe = merged_hit_pe[k]
#                     if true_pe <= 0:
#                         continue

#                     amp = np.random.normal(
#                         true_pe * self.single_pe_amp_mean,
#                         self.single_pe_amp_std * np.sqrt(true_pe),
#                     )
#                     if amp > self.amp_threshold:
#                         time = np.random.normal(
#                             merged_hit_times[k],
#                             self.single_pe_time_std / np.sqrt(true_pe),
#                         )
#                         simulated_event.hit_charges[i_mpmt][i_pmt].append(amp)
#                         simulated_event.hit_times[i_mpmt][i_pmt].append(time)

#     def get_neg_log_likelihood_npe(self, exp_pes, obs_pes):
#         exp_pes = np.asarray(exp_pes, dtype=np.float64)
#         obs_pes = np.asarray(obs_pes, dtype=np.float64)

#         mask = exp_pes > 0.0
#         signal_nll = exp_pes[mask] - obs_pes[mask] * np.log(exp_pes[mask])

#         mask2 = (exp_pes <= 0.0) & (obs_pes > 0.0)
#         background_nll = -obs_pes[mask2] * np.log(1e-4)

#         return float(np.sum(signal_nll) + np.sum(background_nll))

#     def get_neg_log_likelihood_npe_t(self, exp_pes, obs_pes, exp_ts, obs_ts):
#         exp_pes = np.asarray(exp_pes, dtype=np.float64)
#         obs_pes = np.asarray(obs_pes, dtype=np.float64)
#         exp_ts = np.asarray(exp_ts, dtype=np.float64)
#         obs_ts = np.asarray(obs_ts, dtype=np.float64)

#         return float(
#             _poisson_time_nll(
#                 exp_pes,
#                 obs_pes,
#                 exp_ts,
#                 obs_ts,
#                 float(self.single_pe_time_std),
#             )
#         )

#     def get_neg_log_likelihood_q_t(self, exp_pes, obs_qs, exp_ts, obs_ts):
#         """
#         Charge+time likelihood.  This path is typically not the bottleneck for
#         your current fits, so it remains close to the original implementation.
#         """
#         exp_pes = np.asarray(exp_pes, dtype=np.float64)
#         exp_ts = np.asarray(exp_ts, dtype=np.float64)
#         obs_ts = np.asarray(obs_ts, dtype=np.float64)

#         obs_qs = np.asarray(obs_qs)
#         obs_pes = np.zeros(len(obs_qs), dtype=np.float64)

#         valid_q = np.isfinite(obs_qs.astype(np.float64, copy=False))
#         obs_pes[valid_q] = obs_qs[valid_q].astype(np.float64) / self.single_pe_amp_mean

#         mask0 = (exp_pes > 0.0) & (obs_pes == 0.0)
#         no_signal_prob = np.exp(-exp_pes[mask0]) + self.prob01 * exp_pes[mask0] * np.exp(-exp_pes[mask0])
#         no_signal_nll = -np.log(no_signal_prob + 1e-10)

#         mask = (exp_pes > 0.0) & (obs_pes <= 5.0)
#         n_pes = np.arange(1, 9, dtype=np.float64)
#         exp_pe = exp_pes[mask]
#         prob_ns = (
#             np.exp(-exp_pe[:, None])
#             * exp_pe[:, None] ** n_pes[None, :]
#             / np.array([math.factorial(int(n)) for n in n_pes], dtype=np.float64)[None, :]
#         )
#         observed_ope10 = np.clip((obs_pes[mask] * 10.0).astype(int), 0, self.charge_response.shape[1] - 1)
#         pdfs_sums = np.sum(prob_ns * self.charge_response[:, observed_ope10].T, axis=1)
#         signal_nllq = -np.log(pdfs_sums + 1e-10)

#         mask1 = (exp_pes > 0.0) & (obs_pes > 5.0)
#         sigmas = np.sqrt(exp_pes[mask1] + self.single_pe_amp_std**2 * exp_pes[mask1])
#         n_sigma = np.abs(obs_pes[mask1] - exp_pes[mask1]) / sigmas
#         high_sigma = n_sigma >= 4.0
#         low_sigma = ~high_sigma
#         high_sigma_nllq = -obs_pes[mask1][high_sigma] * np.log(1e-4)
#         low_sigma_nllq = 0.5 * (obs_pes[mask1][low_sigma] - exp_pes[mask1][low_sigma]) ** 2 / sigmas[low_sigma] ** 2

#         mask2 = (exp_pes <= 0.0) & (obs_pes > 0.0)
#         background_nll = -obs_pes[mask2] * np.log(1e-4)

#         mask_t = (exp_pes > 0.0) & (obs_pes > 0.0) & np.isfinite(obs_ts)
#         high_sigma_nllt = np.array([], dtype=np.float64)
#         low_sigma_nllt = np.array([], dtype=np.float64)
#         if np.any(mask_t):
#             sigma_t = self.single_pe_time_std / np.sqrt(obs_pes[mask_t])
#             n_sigma_t = np.abs(obs_ts[mask_t] - exp_ts[mask_t]) / sigma_t
#             high_sigma_t = n_sigma_t >= 4.0
#             low_sigma_t = ~high_sigma_t
#             high_sigma_nllt = -obs_pes[mask_t][high_sigma_t] * np.log(1e-4)
#             low_sigma_nllt = 0.5 * (obs_ts[mask_t][low_sigma_t] - exp_ts[mask_t][low_sigma_t]) ** 2 / sigma_t[low_sigma_t] ** 2

#         return float(
#             np.sum(no_signal_nll)
#             + np.sum(signal_nllq)
#             + np.sum(high_sigma_nllq)
#             + np.sum(low_sigma_nllq)
#             + np.sum(background_nll)
#             + np.sum(high_sigma_nllt)
#             + np.sum(low_sigma_nllt)
#         )

