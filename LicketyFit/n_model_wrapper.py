"""
n_model_wrapper.py

Simple vectorized wrapper for evaluating n(E, r) for broadcastable arrays E and r.

Usage
-----
from n_model_wrapper import n_from_E_r

n = n_from_E_r(E, r)

where E and r can be scalars, lists, or numpy arrays. The function returns
a numpy array with the broadcasted shape of E and r.
"""

import numpy as np


# Best / active parameters from the accepted phenomenological form
PARAMS = np.array([
    0.0,    # n_low
    320.0,  # n_high
    55.0,   # E50_n
    8.0,    # q_n

    500.0,  # rpeak_lowE
    520.0,  # rpeak_highE
    120.0,  # E50_rpeak
    3.0,    # q_rpeak

    1.9,    # alpha_low
    1.0,    # alpha_high
    120.0,  # E50_alpha
    4.0,    # q_alpha

    0.85,   # b_low
    0.65,   # b_high
    100.0,  # E50_b
    4.0,    # q_b

    200.0,  # E_plateau
    8.0,    # q_plateau

    0.22    # blend_w
], dtype=float)


def hill(E, E50, q):
    """Hill-style transition function."""
    E = np.asarray(E, dtype=float)
    E = np.clip(E, 1e-9, None)
    return E**q / (E**q + E50**q)


def blend_weight(r, r_peak, w):
    """
    Smooth blend in log(r) around r_peak.
    Returns ~1 for r << r_peak and ~0 for r >> r_peak.
    """
    lr = np.log(np.maximum(r, 1e-9))
    lp = np.log(np.maximum(r_peak, 1e-9))
    return 1.0 / (1.0 + np.exp((lr - lp) / w))


def n_model(E, r, p=PARAMS):
    """
    Vectorized phenomenological model for n(E, r).

    Parameters
    ----------
    E : array-like
        Energy values.
    r : array-like
        Distance values.
    p : array-like, optional
        Model parameter vector. Defaults to PARAMS.

    Returns
    -------
    ndarray
        n values with the broadcasted shape of E and r.
    """
    E = np.asarray(E, dtype=float)
    r = np.asarray(r, dtype=float)
    E, r = np.broadcast_arrays(E, r)

    (
        n_low, n_high, E50_n, q_n,
        rpeak_lowE, rpeak_highE, E50_rpeak, q_rpeak,
        alpha_low, alpha_high, E50_alpha, q_alpha,
        b_low, b_high, E50_b, q_b,
        E_plateau, q_plateau,
        blend_w
    ) = p

    # Maximum available light vs energy
    f_n = hill(E, E50_n, q_n)
    n_max = n_low + (n_high - n_low) * f_n

    # Location of the peak / turnover in r as a function of energy
    f_rp = hill(E, E50_rpeak, q_rpeak)
    r_peak = rpeak_lowE + (rpeak_highE - rpeak_lowE) * f_rp

    # Large-r falloff exponent
    f_a = hill(E, E50_alpha, q_alpha)
    alpha = alpha_low + (alpha_high - alpha_low) * f_a

    # Base small-r suppression exponent
    f_b = hill(E, E50_b, q_b)
    b = b_low + (b_high - b_low) * f_b

    # Collapse switch:
    # ~1 below E_plateau, ~0 above E_plateau
    plateau_switch = 1.0 - hill(E, E_plateau, q_plateau)

    # Effective exponent on the left branch
    # Above E_plateau, this tends toward 0, giving a plateau at small r
    b_eff = b * plateau_switch

    # Small-r branch: rises toward the peak
    n_left = n_max * (np.maximum(r, 1e-9) / np.maximum(r_peak, 1e-9))**b_eff

    # Large-r branch: falls away from the peak
    n_right = n_max * (np.maximum(r_peak, 1e-9) / np.maximum(r, 1e-9))**alpha

    # Smooth blend around r_peak
    w = blend_weight(r, r_peak, blend_w)

    return w * n_left + (1.0 - w) * n_right


def n_from_E_r(E, r):
    """
    Convenience wrapper: return n for each broadcasted pair of E and r.
    """
    return n_model(E, r, PARAMS)


if __name__ == "__main__":
    # Tiny example
    E = np.array([50, 150, 250, 350])
    r = np.array([300, 500, 700, 900])
    print(n_from_E_r(E, r))
