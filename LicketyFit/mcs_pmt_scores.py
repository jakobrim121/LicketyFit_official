"""Per-PMT likelihood contributions and score matrices for coherent MCS research.

The production charge-plus-time objective is a sum over PMTs.  This module
exposes the same decomposition without changing the likelihood itself.  It is
used to build positive-semidefinite outer-product/Fisher blocks for a local
Fermi--Eyges latent-process update around the accepted joint fit.

No detector truth or WCSim-derived quantity enters this calculation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numba import njit

from .Emitter import shift_timing_prediction
from .PMT import (
    _first_arrival_exp_lut,
    _normal_interval_probability_stable,
    _prepare_first_arrival_observations,
    _has_first_arrival_prediction,
)


@njit(cache=True)
def _poisson_nll_vector_numba(exp_pes, obs_pes):
    n = exp_pes.size
    out = np.zeros(n, dtype=np.float64)
    log_noise = math.log(1.0e-4)
    for i in range(n):
        lam = float(exp_pes[i])
        obs = float(obs_pes[i])
        if lam > 0.0:
            out[i] = lam - obs * math.log(lam)
        elif obs > 0.0:
            out[i] = -obs * log_noise
    return out


@njit(cache=True, fastmath=True)
def _first_arrival_deferred_reflection_nll_vector_numba(
    base_mu, base_t, ref_u, ref_tbase, transfer_active, time_offset_active,
    patch_min_offset, patch_max_offset, n_bins,
    q_active, t_active, inv_sigma_active,
    output_efficiency, prompt_lo, prompt_hi, node_pe_scale,
    reflection_occupancy_mix, direct_support_scale_pe,
):
    """Return one conditional first-arrival NLL contribution per active PMT.

    This is algebraically identical to the production scalar kernel in PMT.py;
    only the final reduction is omitted.  The output ordering is the ordering of
    ``prediction.first_arrival_active_indices``.
    """
    nb, nc = base_mu.shape
    npatch = ref_u.size
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    use_window = (
        math.isfinite(prompt_lo) and math.isfinite(prompt_hi)
        and prompt_hi > prompt_lo
    )

    out = np.zeros(nc, dtype=np.float64)
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
        for i in range(nc):
            out[i] = 1.0e30 / max(nc, 1)
        return out

    span = tmax - tmin
    if span < 1.0e-12:
        span = 1.0e-12
    inv_span_bins = float(n_bins) / span

    sbm = np.empty(nb, dtype=np.float32)
    sbt = np.empty(nb, dtype=np.float32)
    rmu = np.empty(n_bins, dtype=np.float64)
    rtn = np.empty(n_bins, dtype=np.float64)

    for i in range(nc):
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

        q = float(q_active[i])
        tobs = float(t_active[i])
        inv_sigma = float(inv_sigma_active[i])
        if q <= 0.0 or (not math.isfinite(tobs)) or inv_sigma <= 0.0:
            continue

        total = ref_total
        for j in range(nvalid):
            total += float(sbm[j])
        if total <= 0.0 or (not math.isfinite(total)):
            out[i] = -math.log(1.0e-300)
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
        mix = 0.0
        acceptance = 0.0
        sum_w = 0.0
        ib = 0
        ir = 0
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

            z = (tobs - tau) * inv_sigma
            gpdf = _first_arrival_exp_lut(z)
            if gpdf > 0.0:
                mix += w * gpdf * inv_sigma * inv_sqrt_2pi
            if use_window:
                zhi = (prompt_hi - tau) * inv_sigma
                zlo = (prompt_lo - tau) * inv_sigma
                a = _normal_interval_probability_stable(zlo, zhi)
                if a > 0.0 and math.isfinite(a):
                    acceptance += w * a
            sum_w += w

        if use_window:
            full_density = 0.0 if (acceptance <= 0.0 or mix <= 0.0) else mix / acceptance
        else:
            full_density = 0.0 if (sum_w <= 0.0 or mix <= 0.0) else mix / sum_w

        if reflection_occupancy_mix:
            mu_base = max(float(node_pe_scale) * base_total, 0.0)
            mu_ref = max(float(node_pe_scale) * ref_total, 0.0)
            p_ref = -math.expm1(-mu_ref)
            support_scale = max(float(direct_support_scale_pe), 1.0e-12)
            direct_gate = mu_base / (mu_base + support_scale)
            trust = direct_gate + (1.0 - direct_gate) * p_ref
            # In the non-window branch, the production kernel uses the same
            # configured prompt span as its maximum-entropy nuisance density.
            unresolved_density = 1.0 / max(prompt_hi - prompt_lo, 1.0e-12)
            density = trust * full_density + (1.0 - trust) * unresolved_density
        else:
            density = full_density
        out[i] = -math.log(max(density, 1.0e-300))

    return out


def first_arrival_nll_vector(
    prediction,
    obs_pes,
    obs_ts,
    *,
    prompt_lo: float,
    prompt_hi: float,
    output_efficiency: float,
    reflection_occupancy_mix: bool,
    direct_support_scale_pe: float,
) -> np.ndarray:
    """Return a detector-length vector of timing NLL contributions."""
    q_all = np.asarray(obs_pes, dtype=np.float64)
    t_all = np.asarray(obs_ts, dtype=np.float64)
    result = np.zeros(q_all.size, dtype=np.float64)
    if not _has_first_arrival_prediction(prediction):
        raise NotImplementedError("research OPG path currently requires first-arrival timing")

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
    if node_pe_scale is None or not np.isfinite(float(node_pe_scale)) or float(node_pe_scale) < 0.0:
        raise ValueError("invalid first-arrival node PE scale")
    if not (
        dbm is not None and dbt is not None and ru is not None and rtb is not None
        and rtr is not None and rto is not None
    ):
        raise NotImplementedError("research OPG path currently requires deferred reflection timing")

    q_active, t_active, inv_sigma = _prepare_first_arrival_observations(
        q_all, t_all, active, float(output_efficiency)
    )
    active_values = _first_arrival_deferred_reflection_nll_vector_numba(
        np.ascontiguousarray(dbm, dtype=np.float32),
        np.ascontiguousarray(dbt, dtype=np.float32),
        np.ascontiguousarray(ru, dtype=np.float64),
        np.ascontiguousarray(rtb, dtype=np.float64),
        np.ascontiguousarray(rtr, dtype=np.float32),
        np.ascontiguousarray(rto, dtype=np.float32),
        np.ascontiguousarray(
            getattr(prediction, "first_arrival_reflection_patch_min_time_offset"),
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            getattr(prediction, "first_arrival_reflection_patch_max_time_offset"),
            dtype=np.float32,
        ),
        int(getattr(prediction, "first_arrival_reflection_n_bins")),
        q_active,
        t_active,
        inv_sigma,
        float(output_efficiency),
        float(prompt_lo),
        float(prompt_hi),
        float(node_pe_scale),
        bool(reflection_occupancy_mix),
        float(direct_support_scale_pe),
    )
    result[active] = active_values
    return result


def pmt_nll_vector(
    pmt_model,
    exp_pes,
    obs_pes,
    timing_prediction,
    obs_ts,
    *,
    t0: float = 0.0,
) -> np.ndarray:
    """Exact production charge+first-arrival contribution for every PMT."""
    exp = np.asarray(exp_pes, dtype=np.float64)
    obs = np.asarray(obs_pes, dtype=np.float64)
    if exp.ndim != 1 or obs.shape != exp.shape:
        raise ValueError("charge arrays have incompatible shapes")
    if np.any(~np.isfinite(exp)) or np.any(exp < 0.0) or np.any(~np.isfinite(obs)) or np.any(obs < 0.0):
        raise ValueError("invalid charge arrays")
    timing = timing_prediction
    if float(t0) != 0.0:
        timing = shift_timing_prediction(timing, float(t0))
    charge = _poisson_nll_vector_numba(exp, obs)
    timing_vec = first_arrival_nll_vector(
        timing,
        obs,
        obs_ts,
        prompt_lo=float(pmt_model.first_arrival_prompt_min_ns),
        prompt_hi=float(pmt_model.first_arrival_prompt_max_ns),
        output_efficiency=float(pmt_model.first_arrival_output_efficiency),
        reflection_occupancy_mix=bool(pmt_model.first_arrival_reflection_occupancy_mix),
        direct_support_scale_pe=float(pmt_model.first_arrival_direct_support_scale_pe),
    )
    return charge + timing_vec


def coherent_pmt_nll_vector(model, coefficients, *, t0: float | None = None) -> np.ndarray:
    """Per-PMT vector for :class:`FixedTrackCoherentMCSObjective`."""
    exp_pes, _timing_pes, timing, *_ = model.prediction(coefficients)
    dt = model.t0 if t0 is None else float(t0)
    return pmt_nll_vector(
        model.pmt_model,
        exp_pes,
        model.obs_pes,
        timing,
        model.obs_ts,
        t0=dt,
    )




def _finite_difference_weights(offsets: Sequence[float]) -> np.ndarray:
    """Interpolation weights for ``df/dx`` at zero.

    ``offsets`` are expressed in units of the configured physical finite-
    difference step.  Two points give a first-order secant; three points give
    the derivative of the unique quadratic interpolant.  The common symmetric
    stencil is returned exactly so established interior results remain
    bit-for-bit unchanged.
    """
    x = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if x.size not in (2, 3):
        raise ValueError("finite-difference stencil must contain two or three points")
    if np.any(~np.isfinite(x)) or np.unique(x).size != x.size:
        raise ValueError("finite-difference offsets must be finite and distinct")
    if x.size == 3 and tuple(float(v) for v in x) == (-1.0, 0.0, 1.0):
        return np.asarray([-0.5, 0.0, 0.5], dtype=np.float64)
    # Sum_i w_i x_i^k = d(x^k)/dx|_0 for k=0,...,n-1.
    vandermonde = np.vstack([x ** k for k in range(x.size)])
    rhs = np.zeros(x.size, dtype=np.float64)
    rhs[1] = 1.0
    return np.linalg.solve(vandermonde, rhs)


def _adaptive_parameter_stencil(
    evaluate_offset,
    base_value,
    *,
    min_fraction: float = 2.0 ** -10,
):
    """Build a boundary-safe local derivative stencil.

    The accepted track can be perfectly physical while a fixed symmetric
    ``theta +/- h`` proposal crosses a detector wall, a direction-chart limit,
    or a length bound.  This helper first keeps the historical full central
    stencil when it is valid.  Otherwise it searches inward independently on
    both sides, uses an asymmetric quadratic stencil when both sides exist,
    and falls back to a second-order one-sided stencil at a physical boundary.
    If a coordinate has no axis-aligned feasible direction, its derivative is
    declared blocked rather than aborting the event.

    ``evaluate_offset`` receives a signed offset in units of the nominal step
    and returns either the evaluated object or ``None`` for an invalid point.
    """
    minimum = max(float(min_fraction), np.finfo(np.float64).eps)
    cache = {0.0: base_value}

    def evaluate(offset: float):
        key = float(offset)
        if key not in cache:
            cache[key] = evaluate_offset(key)
        return cache[key]

    plus = evaluate(1.0)
    minus = evaluate(-1.0)
    if plus is not None and minus is not None:
        offsets = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
        return {
            "offsets": offsets,
            "samples": (minus, base_value, plus),
            "weights": _finite_difference_weights(offsets),
            "scheme": "central",
            "minus_fraction": 1.0,
            "plus_fraction": 1.0,
        }

    def largest_valid(sign: float, initial):
        if initial is not None:
            return 1.0, initial
        fraction = 0.5
        while fraction >= minimum:
            value = evaluate(sign * fraction)
            if value is not None:
                return float(fraction), value
            fraction *= 0.5
        return 0.0, None

    plus_fraction, plus = largest_valid(+1.0, plus)
    minus_fraction, minus = largest_valid(-1.0, minus)

    if plus is not None and minus is not None:
        offsets = np.asarray(
            [-minus_fraction, 0.0, plus_fraction], dtype=np.float64
        )
        return {
            "offsets": offsets,
            "samples": (minus, base_value, plus),
            "weights": _finite_difference_weights(offsets),
            "scheme": "asymmetric_central",
            "minus_fraction": float(minus_fraction),
            "plus_fraction": float(plus_fraction),
        }

    if plus is not None:
        near_fraction = 0.5 * plus_fraction
        near = evaluate(near_fraction) if near_fraction >= minimum else None
        if near is not None:
            offsets = np.asarray(
                [0.0, near_fraction, plus_fraction], dtype=np.float64
            )
            samples = (base_value, near, plus)
            scheme = "forward_quadratic"
        else:
            offsets = np.asarray([0.0, plus_fraction], dtype=np.float64)
            samples = (base_value, plus)
            scheme = "forward_linear"
        return {
            "offsets": offsets,
            "samples": samples,
            "weights": _finite_difference_weights(offsets),
            "scheme": scheme,
            "minus_fraction": 0.0,
            "plus_fraction": float(plus_fraction),
        }

    if minus is not None:
        near_fraction = 0.5 * minus_fraction
        near = evaluate(-near_fraction) if near_fraction >= minimum else None
        if near is not None:
            offsets = np.asarray(
                [-minus_fraction, -near_fraction, 0.0], dtype=np.float64
            )
            samples = (minus, near, base_value)
            scheme = "backward_quadratic"
        else:
            offsets = np.asarray([-minus_fraction, 0.0], dtype=np.float64)
            samples = (minus, base_value)
            scheme = "backward_linear"
        return {
            "offsets": offsets,
            "samples": samples,
            "weights": _finite_difference_weights(offsets),
            "scheme": scheme,
            "minus_fraction": float(minus_fraction),
            "plus_fraction": 0.0,
        }

    return {
        "offsets": np.asarray([0.0], dtype=np.float64),
        "samples": (base_value,),
        "weights": None,
        "scheme": "blocked",
        "minus_fraction": 0.0,
        "plus_fraction": 0.0,
    }


def _apply_adaptive_stencil(stencil, value_index: int | None = None):
    """Apply one adaptive stencil to scalar or array-valued samples."""
    weights = stencil["weights"]
    samples = stencil["samples"]
    if weights is None:
        base = samples[0] if value_index is None else samples[0][value_index]
        return np.zeros_like(np.asarray(base, dtype=np.float64))
    values = [
        sample if value_index is None else sample[value_index]
        for sample in samples
    ]
    out = np.zeros_like(np.asarray(values[0], dtype=np.float64))
    for weight, value in zip(weights, values):
        out = out + float(weight) * np.asarray(value, dtype=np.float64)
    return out


@dataclass
class OPGBlocks:
    theta_scores: np.ndarray
    latent_scores: np.ndarray
    gradient_theta: np.ndarray
    gradient_latent: np.ndarray
    information_theta: np.ndarray
    information_cross: np.ndarray
    information_latent_data: np.ndarray
    information_latent_posterior: np.ndarray
    base_contributions: np.ndarray
    theta_fd: np.ndarray
    latent_fd: np.ndarray
    theta_fd_minus_fraction: np.ndarray
    theta_fd_plus_fraction: np.ndarray
    theta_fd_scheme: tuple[str, ...]


def finite_difference_opg_blocks(
    evaluator,
    theta0: Sequence[float],
    *,
    theta_fd: Sequence[float],
    latent_fd: float | Sequence[float] = 0.25,
) -> OPGBlocks:
    """Build full charge+time OPG/Fisher blocks at ``(theta0,u=0)``.

    Track derivatives are expressed with respect to one configured finite-
    difference step, while latent derivatives are with respect to physical
    standardized KL coefficients.  The prior contributes identity only to the
    latent information block.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(7)
    tfd = np.asarray(theta_fd, dtype=np.float64).reshape(7)
    if np.any(~np.isfinite(tfd)) or np.any(tfd <= 0.0):
        raise ValueError("theta finite-difference steps must be positive")
    if np.isscalar(latent_fd):
        ufd = np.full(8, float(latent_fd), dtype=np.float64)
    else:
        ufd = np.asarray(latent_fd, dtype=np.float64).reshape(8)
    if np.any(~np.isfinite(ufd)) or np.any(ufd <= 0.0):
        raise ValueError("latent finite-difference steps must be positive")

    u0 = np.zeros(8, dtype=np.float64)
    base_model = evaluator.model(theta)
    if base_model is None:
        raise ValueError("invalid reference track")
    base = coherent_pmt_nll_vector(base_model, u0, t0=float(theta[6]))
    n_pmt = base.size
    Gt = np.empty((n_pmt, 7), dtype=np.float64)
    Gu = np.empty((n_pmt, 8), dtype=np.float64)

    theta_minus_fraction = np.zeros(7, dtype=np.float64)
    theta_plus_fraction = np.zeros(7, dtype=np.float64)
    theta_schemes: list[str] = []
    for j in range(7):
        def evaluate(offset):
            proposal = theta.copy()
            proposal[j] += float(offset) * tfd[j]
            model = evaluator.model(proposal)
            if model is None:
                return None
            try:
                return coherent_pmt_nll_vector(
                    model, u0, t0=float(proposal[6])
                )
            except Exception:
                return None

        stencil = _adaptive_parameter_stencil(evaluate, base)
        Gt[:, j] = _apply_adaptive_stencil(stencil)
        theta_minus_fraction[j] = float(stencil["minus_fraction"])
        theta_plus_fraction[j] = float(stencil["plus_fraction"])
        theta_schemes.append(str(stencil["scheme"]))

    for k in range(8):
        up = u0.copy()
        um = u0.copy()
        up[k] += ufd[k]
        um[k] -= ufd[k]
        lp = coherent_pmt_nll_vector(base_model, up, t0=float(theta[6]))
        lm = coherent_pmt_nll_vector(base_model, um, t0=float(theta[6]))
        Gu[:, k] = (lp - lm) / (2.0 * ufd[k])

    gt = np.sum(Gt, axis=0)
    gu = np.sum(Gu, axis=0)
    Itt = Gt.T @ Gt
    Itu = Gt.T @ Gu
    Iuu = Gu.T @ Gu
    return OPGBlocks(
        theta_scores=Gt,
        latent_scores=Gu,
        gradient_theta=gt,
        gradient_latent=gu,
        information_theta=0.5 * (Itt + Itt.T),
        information_cross=Itu,
        information_latent_data=0.5 * (Iuu + Iuu.T),
        information_latent_posterior=0.5 * (Iuu + Iuu.T) + np.eye(8),
        base_contributions=base,
        theta_fd=tfd,
        latent_fd=ufd,
        theta_fd_minus_fraction=theta_minus_fraction,
        theta_fd_plus_fraction=theta_plus_fraction,
        theta_fd_scheme=tuple(theta_schemes),
    )

@njit(cache=True, fastmath=True)
def _first_arrival_deferred_reflection_log_density_grid_numba(
    base_mu, base_t, ref_u, ref_tbase, transfer_active, time_offset_active,
    patch_min_offset, patch_max_offset, n_bins,
    q_active, inv_sigma_active, eval_times,
    output_efficiency, prompt_lo, prompt_hi, node_pe_scale,
    reflection_occupancy_mix, direct_support_scale_pe,
):
    """Conditional timing log density on a common quadrature grid.

    Nodes and first-photoelectron order-statistic weights are built once per
    PMT, then reused for all requested times.  This is the exact density whose
    value at the observed timestamp enters the production likelihood.
    """
    nb, nc = base_mu.shape
    nt = eval_times.size
    npatch = ref_u.size
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    use_window = (
        math.isfinite(prompt_lo) and math.isfinite(prompt_hi)
        and prompt_hi > prompt_lo
    )
    out = np.full((nc, nt), math.log(1.0e-300), dtype=np.float64)

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
        return out
    span = tmax - tmin
    if span < 1.0e-12:
        span = 1.0e-12
    inv_span_bins = float(n_bins) / span

    # The maximum ordered-node count is base rows plus compressed reflection bins.
    maxnodes = nb + n_bins
    node_mu = np.empty(maxnodes, dtype=np.float64)
    node_t = np.empty(maxnodes, dtype=np.float64)
    node_w = np.empty(maxnodes, dtype=np.float64)
    sbm = np.empty(nb, dtype=np.float32)
    sbt = np.empty(nb, dtype=np.float32)
    rmu = np.empty(n_bins, dtype=np.float64)
    rtn = np.empty(n_bins, dtype=np.float64)

    for i in range(nc):
        q = float(q_active[i])
        inv_sigma = float(inv_sigma_active[i])
        if q <= 0.0 or inv_sigma <= 0.0 or (not math.isfinite(inv_sigma)):
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
            continue

        neff = q / output_efficiency if output_efficiency > 0.0 else q
        if neff < 1.0e-6:
            neff = 1.0e-6
        base_total = 0.0
        if reflection_occupancy_mix:
            for j in range(nvalid):
                base_total += float(sbm[j])

        # Merge base and reflection nodes in chronological order and compute
        # exact discrete first-source weights.
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
            node_mu[nn] = mnode
            node_t[nn] = tau
            node_w[nn] = w
            nn += 1

        if nn == 0:
            continue
        acceptance = 0.0
        sum_w = 0.0
        for j in range(nn):
            w = node_w[j]
            tau = node_t[j]
            if use_window:
                zhi = (prompt_hi - tau) * inv_sigma
                zlo = (prompt_lo - tau) * inv_sigma
                a = _normal_interval_probability_stable(zlo, zhi)
                if a > 0.0 and math.isfinite(a):
                    acceptance += w * a
            sum_w += w
        denom = acceptance if use_window else sum_w
        if denom <= 0.0 or (not math.isfinite(denom)):
            continue

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
            tobs = float(eval_times[it])
            mix = 0.0
            for j in range(nn):
                z = (tobs - node_t[j]) * inv_sigma
                gpdf = _first_arrival_exp_lut(z)
                if gpdf > 0.0:
                    mix += node_w[j] * gpdf * inv_sigma * inv_sqrt_2pi
            full_density = mix / denom if mix > 0.0 else 0.0
            density = (
                trust * full_density + (1.0 - trust) * unresolved_density
                if reflection_occupancy_mix else full_density
            )
            out[i, it] = math.log(max(density, 1.0e-300))
    return out


def first_arrival_log_density_grid(
    prediction,
    obs_pes,
    eval_times,
    *,
    pmt_model,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(active_indices, log p_i(t_q))`` for the production timing law."""
    if not _has_first_arrival_prediction(prediction):
        raise NotImplementedError("expected-Fisher path requires first-arrival timing")
    active = np.ascontiguousarray(
        prediction.first_arrival_active_indices, dtype=np.int32
    )
    dbm = getattr(prediction, "first_arrival_deferred_base_mu", None)
    dbt = getattr(prediction, "first_arrival_deferred_base_t", None)
    ru = getattr(prediction, "first_arrival_reflection_u", None)
    rtb = getattr(prediction, "first_arrival_reflection_tbase", None)
    rtr = getattr(prediction, "first_arrival_reflection_transfer_active", None)
    rto = getattr(prediction, "first_arrival_reflection_time_offset_active", None)
    if not (
        dbm is not None and dbt is not None and ru is not None and rtb is not None
        and rtr is not None and rto is not None
    ):
        raise NotImplementedError("expected-Fisher path currently requires deferred reflection timing")
    q_all = np.asarray(obs_pes, dtype=np.float64)
    q_active, _unused_t, inv_sigma = _prepare_first_arrival_observations(
        q_all, np.zeros_like(q_all), active,
        float(pmt_model.first_arrival_output_efficiency),
    )
    node_pe_scale = getattr(prediction, "first_arrival_node_pe_scale", 1.0)
    if node_pe_scale is None or not np.isfinite(float(node_pe_scale)):
        raise ValueError("invalid first-arrival node PE scale")
    logp = _first_arrival_deferred_reflection_log_density_grid_numba(
        np.ascontiguousarray(dbm, dtype=np.float32),
        np.ascontiguousarray(dbt, dtype=np.float32),
        np.ascontiguousarray(ru, dtype=np.float64),
        np.ascontiguousarray(rtb, dtype=np.float64),
        np.ascontiguousarray(rtr, dtype=np.float32),
        np.ascontiguousarray(rto, dtype=np.float32),
        np.ascontiguousarray(
            getattr(prediction, "first_arrival_reflection_patch_min_time_offset"),
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            getattr(prediction, "first_arrival_reflection_patch_max_time_offset"),
            dtype=np.float32,
        ),
        int(getattr(prediction, "first_arrival_reflection_n_bins")),
        q_active,
        inv_sigma,
        np.ascontiguousarray(eval_times, dtype=np.float64),
        float(pmt_model.first_arrival_output_efficiency),
        float(pmt_model.first_arrival_prompt_min_ns),
        float(pmt_model.first_arrival_prompt_max_ns),
        float(node_pe_scale),
        bool(pmt_model.first_arrival_reflection_occupancy_mix),
        float(pmt_model.first_arrival_direct_support_scale_pe),
    )
    return active, logp



@njit(cache=True, fastmath=True)
def _first_arrival_direct_variant_log_density_grid_numba(
    fixed_base_mu, fixed_base_t, direct_mu_variants, direct_t_variants,
    node_pe_scale_variants,
    ref_u, ref_tbase, transfer_active, time_offset_active,
    patch_min_offset, patch_max_offset, n_bins,
    q_active, inv_sigma_active, eval_times, time_shift_ns,
    output_efficiency, prompt_lo, prompt_hi,
    reflection_occupancy_mix, direct_support_scale_pe,
):
    """Exact timing log densities for variants that change only direct light.

    The coherent latent finite differences leave delta, molecular, and reflected
    source nodes unchanged.  This kernel bins reflection and sorts all fixed
    base nodes once per PMT, then inserts each variant's direct node.  It is
    algebraically identical to evaluating the generic production-density kernel
    separately for every latent +/- proposal.
    """
    nb_fixed, nc = fixed_base_mu.shape
    nv = direct_mu_variants.shape[0]
    nt = eval_times.size
    npatch = ref_u.size
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    use_window = (
        math.isfinite(prompt_lo) and math.isfinite(prompt_hi)
        and prompt_hi > prompt_lo
    )
    log_floor = math.log(1.0e-300)
    out = np.full((nv, nc, nt), log_floor, dtype=np.float64)

    tmin = 1.0e300
    tmax = -1.0e300
    for p in range(npatch):
        if float(ref_u[p]) <= 0.0:
            continue
        lo = float(ref_tbase[p]) + time_shift_ns + float(patch_min_offset[p])
        hi = float(ref_tbase[p]) + time_shift_ns + float(patch_max_offset[p])
        if lo < tmin:
            tmin = lo
        if hi > tmax:
            tmax = hi
    if tmax < tmin:
        return out
    span = max(tmax - tmin, 1.0e-12)
    inv_span_bins = float(n_bins) / span

    # Reused work arrays.  Stable insertion sorting preserves the production
    # tie ordering of the original base-node rows.
    fixed_mu = np.empty(nb_fixed, dtype=np.float64)
    fixed_t = np.empty(nb_fixed, dtype=np.float64)
    base_mu = np.empty(nb_fixed + 1, dtype=np.float64)
    base_t = np.empty(nb_fixed + 1, dtype=np.float64)
    rmu = np.empty(n_bins, dtype=np.float64)
    rtn = np.empty(n_bins, dtype=np.float64)
    node_t = np.empty(nb_fixed + 1 + n_bins, dtype=np.float64)
    node_w = np.empty(nb_fixed + 1 + n_bins, dtype=np.float64)

    for i in range(nc):
        q = float(q_active[i])
        inv_sigma = float(inv_sigma_active[i])
        if q <= 0.0 or inv_sigma <= 0.0 or (not math.isfinite(inv_sigma)):
            continue

        nfixed = 0
        fixed_total = 0.0
        for j in range(nb_fixed):
            m = float(fixed_base_mu[j, i])
            tt = float(fixed_base_t[j, i])
            if m <= 0.0 or (not math.isfinite(m)) or (not math.isfinite(tt)):
                continue
            tt += time_shift_ns
            k = nfixed
            while k > 0 and tt < fixed_t[k - 1]:
                fixed_t[k] = fixed_t[k - 1]
                fixed_mu[k] = fixed_mu[k - 1]
                k -= 1
            fixed_t[k] = tt
            fixed_mu[k] = m
            nfixed += 1
            fixed_total += m

        for b in range(n_bins):
            rmu[b] = 0.0
            rtn[b] = 0.0
        ref_total = 0.0
        for pp in range(npatch):
            m = float(ref_u[pp]) * float(transfer_active[i, pp])
            if m <= 0.0:
                continue
            tt = float(ref_tbase[pp]) + time_shift_ns + float(time_offset_active[i, pp])
            b = int((tt - tmin) * inv_span_bins)
            if b < 0:
                b = 0
            elif b >= n_bins:
                b = n_bins - 1
            rmu[b] += m
            rtn[b] += m * tt
            ref_total += m

        neff = q / output_efficiency if output_efficiency > 0.0 else q
        if neff < 1.0e-6:
            neff = 1.0e-6

        for v in range(nv):
            dm = float(direct_mu_variants[v, i])
            dt = float(direct_t_variants[v, i])
            direct_valid = dm > 0.0 and math.isfinite(dm) and math.isfinite(dt)
            if direct_valid:
                dt += time_shift_ns

            # Stable insertion of original row zero: it precedes fixed rows at
            # equal time, matching insertion-sort order in the generic kernel.
            nbase = nfixed
            if direct_valid:
                pos = 0
                while pos < nfixed and fixed_t[pos] < dt:
                    pos += 1
                for j in range(pos):
                    base_mu[j] = fixed_mu[j]
                    base_t[j] = fixed_t[j]
                base_mu[pos] = dm
                base_t[pos] = dt
                for j in range(pos, nfixed):
                    base_mu[j + 1] = fixed_mu[j]
                    base_t[j + 1] = fixed_t[j]
                nbase = nfixed + 1
            else:
                for j in range(nfixed):
                    base_mu[j] = fixed_mu[j]
                    base_t[j] = fixed_t[j]

            total = fixed_total + ref_total + (dm if direct_valid else 0.0)
            if total <= 0.0 or (not math.isfinite(total)):
                continue

            remaining = 1.0
            remaining_power = 1.0
            ib = 0
            ir = 0
            nn = 0
            while ib < nbase or ir < n_bins:
                while ir < n_bins and rmu[ir] <= 0.0:
                    ir += 1
                if ib >= nbase and ir >= n_bins:
                    break
                take_base = False
                if ir >= n_bins:
                    take_base = True
                elif ib < nbase:
                    rt = rtn[ir] / rmu[ir]
                    if base_t[ib] <= rt:
                        take_base = True
                if take_base:
                    mnode = base_mu[ib]
                    tau = base_t[ib]
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

            if nn == 0:
                continue
            acceptance = 0.0
            sum_w = 0.0
            for j in range(nn):
                w = node_w[j]
                tau = node_t[j]
                if use_window:
                    zhi = (prompt_hi - tau) * inv_sigma
                    zlo = (prompt_lo - tau) * inv_sigma
                    a = _normal_interval_probability_stable(zlo, zhi)
                    if a > 0.0 and math.isfinite(a):
                        acceptance += w * a
                sum_w += w
            denom = acceptance if use_window else sum_w
            if denom <= 0.0 or (not math.isfinite(denom)):
                continue

            trust = 1.0
            unresolved_density = 0.0
            if reflection_occupancy_mix:
                scale = max(float(node_pe_scale_variants[v]), 0.0)
                mu_base = scale * (fixed_total + (dm if direct_valid else 0.0))
                mu_ref = scale * ref_total
                p_ref = -math.expm1(-mu_ref)
                support_scale = max(float(direct_support_scale_pe), 1.0e-12)
                direct_gate = mu_base / (mu_base + support_scale)
                trust = direct_gate + (1.0 - direct_gate) * p_ref
                unresolved_density = 1.0 / max(prompt_hi - prompt_lo, 1.0e-12)

            for it in range(nt):
                tobs = float(eval_times[it])
                mix = 0.0
                for j in range(nn):
                    z = (tobs - node_t[j]) * inv_sigma
                    gpdf = _first_arrival_exp_lut(z)
                    if gpdf > 0.0:
                        mix += node_w[j] * gpdf * inv_sigma * inv_sqrt_2pi
                full_density = mix / denom if mix > 0.0 else 0.0
                density = (
                    trust * full_density + (1.0 - trust) * unresolved_density
                    if reflection_occupancy_mix else full_density
                )
                out[v, i, it] = math.log(max(density, 1.0e-300))
    return out


def first_arrival_direct_variant_log_density_grid(
    base_prediction,
    variant_predictions,
    obs_pes,
    eval_times,
    *,
    pmt_model,
    time_shift_ns: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch exact timing densities for coherent direct-light variants."""
    variants = tuple(variant_predictions)
    if not variants:
        raise ValueError("at least one direct-light timing variant is required")
    active = np.ascontiguousarray(
        base_prediction.first_arrival_active_indices, dtype=np.int32
    )
    base_mu = np.asarray(
        base_prediction.first_arrival_deferred_base_mu, dtype=np.float32
    )
    base_t = np.asarray(
        base_prediction.first_arrival_deferred_base_t, dtype=np.float32
    )
    if base_mu.shape[0] < 1:
        raise RuntimeError("deferred timing prediction has no direct row")
    nv = len(variants)
    direct_mu = np.empty((nv, active.size), dtype=np.float32)
    direct_t = np.empty((nv, active.size), dtype=np.float32)
    node_scale = np.empty(nv, dtype=np.float64)
    for i, pred in enumerate(variants):
        if not np.array_equal(
            np.asarray(pred.first_arrival_active_indices, dtype=np.int32), active
        ):
            raise RuntimeError("latent timing-active PMT support changed")
        pmu = np.asarray(pred.first_arrival_deferred_base_mu, dtype=np.float32)
        pt = np.asarray(pred.first_arrival_deferred_base_t, dtype=np.float32)
        if pmu.shape != base_mu.shape or pt.shape != base_t.shape:
            raise RuntimeError("latent timing-node shape changed")
        direct_mu[i] = pmu[0]
        direct_t[i] = pt[0]
        node_scale[i] = float(pred.first_arrival_node_pe_scale)

    q_all = np.asarray(obs_pes, dtype=np.float64)
    q_active, _unused_t, inv_sigma = _prepare_first_arrival_observations(
        q_all,
        np.zeros_like(q_all),
        active,
        float(pmt_model.first_arrival_output_efficiency),
    )
    logp = _first_arrival_direct_variant_log_density_grid_numba(
        np.ascontiguousarray(base_mu[1:], dtype=np.float32),
        np.ascontiguousarray(base_t[1:], dtype=np.float32),
        np.ascontiguousarray(direct_mu, dtype=np.float32),
        np.ascontiguousarray(direct_t, dtype=np.float32),
        np.ascontiguousarray(node_scale, dtype=np.float64),
        np.ascontiguousarray(base_prediction.first_arrival_reflection_u, dtype=np.float64),
        np.ascontiguousarray(base_prediction.first_arrival_reflection_tbase, dtype=np.float64),
        np.ascontiguousarray(
            base_prediction.first_arrival_reflection_transfer_active, dtype=np.float32
        ),
        np.ascontiguousarray(
            base_prediction.first_arrival_reflection_time_offset_active, dtype=np.float32
        ),
        np.ascontiguousarray(
            base_prediction.first_arrival_reflection_patch_min_time_offset,
            dtype=np.float32,
        ),
        np.ascontiguousarray(
            base_prediction.first_arrival_reflection_patch_max_time_offset,
            dtype=np.float32,
        ),
        int(base_prediction.first_arrival_reflection_n_bins),
        q_active,
        inv_sigma,
        np.ascontiguousarray(eval_times, dtype=np.float64),
        float(time_shift_ns),
        float(pmt_model.first_arrival_output_efficiency),
        float(pmt_model.first_arrival_prompt_min_ns),
        float(pmt_model.first_arrival_prompt_max_ns),
        bool(pmt_model.first_arrival_reflection_occupancy_mix),
        float(pmt_model.first_arrival_direct_support_scale_pe),
    )
    return active, logp

@dataclass
class ExpectedFisherBlocks:
    information_theta_charge: np.ndarray
    information_cross_charge: np.ndarray
    information_latent_charge: np.ndarray
    information_theta_timing: np.ndarray
    information_cross_timing: np.ndarray
    information_latent_timing: np.ndarray
    information_theta: np.ndarray
    information_cross: np.ndarray
    information_latent_data: np.ndarray
    information_latent_posterior: np.ndarray
    charge_jacobian_theta: np.ndarray
    charge_jacobian_latent: np.ndarray
    timing_score_theta: np.ndarray
    timing_score_latent: np.ndarray
    timing_weights: np.ndarray
    timing_active_indices: np.ndarray
    timing_quadrature_times: np.ndarray
    timing_normalization: np.ndarray
    theta_fd: np.ndarray
    latent_fd: np.ndarray
    theta_fd_minus_fraction: np.ndarray
    theta_fd_plus_fraction: np.ndarray
    theta_fd_scheme: tuple[str, ...]


def finite_difference_expected_fisher_blocks(
    evaluator,
    theta0: Sequence[float],
    *,
    theta_fd: Sequence[float],
    latent_fd: float | Sequence[float] = 0.25,
    timing_quadrature_nodes: int = 32,
) -> ExpectedFisherBlocks:
    """Expected charge+first-arrival Fisher blocks at ``(theta0,u=0)``.

    The charge block is the exact conditional Poisson Fisher information.  The
    timing block numerically integrates the exact production conditional
    first-photoelectron density over the configured prompt interval.  This is
    positive semidefinite by construction and avoids using WCSim truth or an
    observed, potentially indefinite Hessian.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(7)
    tfd = np.asarray(theta_fd, dtype=np.float64).reshape(7)
    if np.isscalar(latent_fd):
        ufd = np.full(8, float(latent_fd), dtype=np.float64)
    else:
        ufd = np.asarray(latent_fd, dtype=np.float64).reshape(8)
    if np.any(tfd <= 0.0) or np.any(ufd <= 0.0):
        raise ValueError("finite-difference steps must be positive")

    nq = max(int(timing_quadrature_nodes), 8)
    lo = float(evaluator.pmt_model.first_arrival_prompt_min_ns)
    hi = float(evaluator.pmt_model.first_arrival_prompt_max_ns)
    xq, wq = np.polynomial.legendre.leggauss(nq)
    tq = 0.5 * (hi + lo) + 0.5 * (hi - lo) * xq
    wt = 0.5 * (hi - lo) * wq

    u0 = np.zeros(8, dtype=np.float64)
    m0 = evaluator.model(theta)
    if m0 is None:
        raise ValueError("invalid reference track")
    exp0, _tp0, pred0, *_ = m0.prediction(u0)
    pred0s = shift_timing_prediction(pred0, float(theta[6])) if float(theta[6]) != 0.0 else pred0
    active0, logp0 = first_arrival_log_density_grid(
        pred0s, m0.obs_pes, tq, pmt_model=evaluator.pmt_model
    )
    p0 = np.exp(np.maximum(logp0, -745.0))
    norm = p0 @ wt
    # Quadrature error is diagnostic; normalize only at the integration level,
    # not inside the likelihood or its derivatives.
    pweight = p0 * wt[None, :]

    Jtq = np.empty((exp0.size, 7), dtype=np.float64)
    Juq = np.empty((exp0.size, 8), dtype=np.float64)
    St = np.empty((active0.size, nq, 7), dtype=np.float64)
    Su = np.empty((active0.size, nq, 8), dtype=np.float64)

    theta_minus_fraction = np.zeros(7, dtype=np.float64)
    theta_plus_fraction = np.zeros(7, dtype=np.float64)
    theta_schemes: list[str] = []
    base_sample = (exp0, logp0)
    for j in range(7):
        if j == 6:
            # t0 is an exact translation of every timing node.  Preserve the
            # historical full central stencil without rebuilding optics.
            tp = theta.copy(); tm = theta.copy()
            tp[j] += tfd[j]; tm[j] -= tfd[j]
            pp = shift_timing_prediction(pred0, float(tp[6]))
            pm = shift_timing_prediction(pred0, float(tm[6]))
            ap, lp = first_arrival_log_density_grid(
                pp, evaluator.obs_pes, tq, pmt_model=evaluator.pmt_model
            )
            am, lm = first_arrival_log_density_grid(
                pm, evaluator.obs_pes, tq, pmt_model=evaluator.pmt_model
            )
            if not (np.array_equal(ap, active0) and np.array_equal(am, active0)):
                raise RuntimeError(
                    "timing-active PMT support changed across t0 finite difference"
                )
            Jtq[:, j] = 0.0
            St[:, :, j] = 0.5 * (lp - lm)
            theta_minus_fraction[j] = 1.0
            theta_plus_fraction[j] = 1.0
            theta_schemes.append("central_analytic_t0")
            continue

        def evaluate(offset):
            proposal = theta.copy()
            proposal[j] += float(offset) * tfd[j]
            straight = evaluator.straight_prediction(
                proposal, raise_on_model_error=True
            )
            if straight is None:
                return None
            expected, _timing_expected, prediction = straight
            if float(proposal[6]) != 0.0:
                prediction = shift_timing_prediction(
                    prediction, float(proposal[6])
                )
            try:
                active, log_density = first_arrival_log_density_grid(
                    prediction,
                    evaluator.obs_pes,
                    tq,
                    pmt_model=evaluator.pmt_model,
                )
            except Exception:
                return None
            if not np.array_equal(active, active0):
                return None
            return np.asarray(expected, dtype=np.float64), log_density

        stencil = _adaptive_parameter_stencil(evaluate, base_sample)
        Jtq[:, j] = _apply_adaptive_stencil(stencil, 0)
        St[:, :, j] = _apply_adaptive_stencil(stencil, 1)
        theta_minus_fraction[j] = float(stencil["minus_fraction"])
        theta_plus_fraction[j] = float(stencil["plus_fraction"])
        theta_schemes.append(str(stencil["scheme"]))

    latent_timing_predictions = []
    for k in range(8):
        up = u0.copy(); um = u0.copy()
        up[k] += ufd[k]; um[k] -= ufd[k]
        ep, _x, pp, *_ = m0.prediction(up)
        em, _x, pm, *_ = m0.prediction(um)
        Juq[:, k] = (ep - em) / (2.0 * ufd[k])
        latent_timing_predictions.extend((pp, pm))

    au, latent_logp = first_arrival_direct_variant_log_density_grid(
        pred0,
        latent_timing_predictions,
        m0.obs_pes,
        tq,
        pmt_model=evaluator.pmt_model,
        time_shift_ns=float(theta[6]),
    )
    if not np.array_equal(au, active0):
        raise RuntimeError("timing-active PMT support changed across latent finite difference")
    for k in range(8):
        Su[:, :, k] = (
            latent_logp[2 * k] - latent_logp[2 * k + 1]
        ) / (2.0 * ufd[k])

    muw = np.maximum(np.asarray(exp0, dtype=np.float64), 1.0e-12)
    Itt_q = Jtq.T @ (Jtq / muw[:, None])
    Itu_q = Jtq.T @ (Juq / muw[:, None])
    Iuu_q = Juq.T @ (Juq / muw[:, None])

    sw = np.sqrt(np.maximum(pweight, 0.0))
    Stw = (St * sw[:, :, None]).reshape(-1, 7)
    Suw = (Su * sw[:, :, None]).reshape(-1, 8)
    Itt_t = Stw.T @ Stw
    Itu_t = Stw.T @ Suw
    Iuu_t = Suw.T @ Suw

    Itt = Itt_q + Itt_t
    Itu = Itu_q + Itu_t
    Iuu = Iuu_q + Iuu_t
    return ExpectedFisherBlocks(
        information_theta_charge=0.5 * (Itt_q + Itt_q.T),
        information_cross_charge=Itu_q,
        information_latent_charge=0.5 * (Iuu_q + Iuu_q.T),
        information_theta_timing=0.5 * (Itt_t + Itt_t.T),
        information_cross_timing=Itu_t,
        information_latent_timing=0.5 * (Iuu_t + Iuu_t.T),
        information_theta=0.5 * (Itt + Itt.T),
        information_cross=Itu,
        information_latent_data=0.5 * (Iuu + Iuu.T),
        information_latent_posterior=0.5 * (Iuu + Iuu.T) + np.eye(8),
        charge_jacobian_theta=Jtq,
        charge_jacobian_latent=Juq,
        timing_score_theta=St,
        timing_score_latent=Su,
        timing_weights=pweight,
        timing_active_indices=active0,
        timing_quadrature_times=tq,
        timing_normalization=norm,
        theta_fd=tfd,
        latent_fd=ufd,
        theta_fd_minus_fraction=theta_minus_fraction,
        theta_fd_plus_fraction=theta_plus_fraction,
        theta_fd_scheme=tuple(theta_schemes),
    )
