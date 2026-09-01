"""WCSim R14374-WCTE single-photoelectron charge response.

The WCSim samples used by LicketyFit do not have a Gaussian one-PE response.
``WCSimWCPMT::rn1pe`` samples the tabulated cumulative ``qpe0`` distribution
for ``PMT3inchR14374_WCTE`` and the SK-I digitizer then applies its stochastic
low-charge threshold, one 0.03-PE electronics smear per integrated digit, and
the 0.985 charge factor.  This module evaluates that same detector response
without Monte Carlo noise.

For the occupancies that control the likelihood, sums of the tabulated qpe
law are convolved on a grid whose cells exactly subdivide WCSim's native
1/22.83-PE bins.  Above ``exact_n_max`` a fourth-order Edgeworth density uses
the exact first four cumulants.  At that transition the standardized skewness
is already small; the approximation is used only for high-occupancy channels
and keeps event response preparation fast.

The CDF below is a direct transcription of ``PMT3inchR14374_WCTE::Getqpe`` in
WCSim_changes commit ``bc5ca65893ee10dc42259ec541690ec09b15facb``.  Entries
after index 192 are identically one and are omitted.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import ndtr


WCSIM_QPE_BIN_DENOMINATOR = 22.83
WCSIM_QPE_INDEX_OFFSET = 50
WCSIM_DIGITIZER_NOISE_SIGMA_PE = 0.03
WCSIM_DIGITIZER_CHARGE_FACTOR = 0.985
WCSIM_QPE_SOURCE_COMMIT = "bc5ca65893ee10dc42259ec541690ec09b15facb"

# CDF indices 0..192, inclusive.  Index 192 is the first value equal to one.
WCSIM_R14374_WCTE_QPE_CDF = np.asarray(
    (
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
        0.000381, 0.002359, 0.006956, 0.014050, 0.021470,
        0.029045, 0.036750, 0.044913, 0.053878, 0.063897,
        0.075367, 0.089078, 0.105693, 0.126747, 0.152918,
        0.184803, 0.223584, 0.268127, 0.317283, 0.367630,
        0.417750, 0.466039, 0.512878, 0.557372, 0.598186,
        0.636186, 0.671044, 0.701999, 0.730782, 0.757566,
        0.782031, 0.804206, 0.824292, 0.842793, 0.859734,
        0.875207, 0.889501, 0.902172, 0.913244, 0.923582,
        0.932835, 0.940867, 0.947906, 0.954192, 0.959824,
        0.964409, 0.968364, 0.971983, 0.975084, 0.977725,
        0.979961, 0.982130, 0.984062, 0.985578, 0.986994,
        0.988236, 0.989201, 0.990078, 0.990896, 0.991671,
        0.992372, 0.992994, 0.993539, 0.994000, 0.994403,
        0.994807, 0.995191, 0.995541, 0.995831, 0.996070,
        0.996275, 0.996491, 0.996711, 0.996929, 0.997135,
        0.997340, 0.997542, 0.997677, 0.997806, 0.997960,
        0.998096, 0.998204, 0.998276, 0.998337, 0.998383,
        0.998410, 0.998456, 0.998514, 0.998578, 0.998642,
        0.998703, 0.998759, 0.998797, 0.998825, 0.998852,
        0.998885, 0.998926, 0.998979, 0.999029, 0.999073,
        0.999111, 0.999162, 0.999223, 0.999272, 0.999309,
        0.999343, 0.999407, 0.999451, 0.999469, 0.999534,
        0.999593, 0.999619, 0.999633, 0.999653, 0.999688,
        0.999705, 0.999714, 0.999724, 0.999733, 0.999739,
        0.999739, 0.999757, 0.999782, 0.999800, 0.999824,
        0.999852, 0.999879, 0.999901, 0.999915, 0.999916,
        0.999916, 0.999916, 0.999916, 0.999916, 0.999918,
        0.999926, 0.999940, 0.999957, 0.999968, 0.999977,
        0.999986, 0.999995, 1.000000,
    ),
    dtype=np.float64,
)


def _raw_moments() -> tuple[float, float, float, float]:
    """Return mean, variance, third cumulant and fourth cumulant."""
    cdf = WCSIM_R14374_WCTE_QPE_CDF
    probability = np.diff(np.concatenate(([0.0], cdf)))
    indices = np.arange(cdf.size, dtype=np.float64)
    low = (indices - WCSIM_QPE_INDEX_OFFSET) / WCSIM_QPE_BIN_DENOMINATOR
    high = low + 1.0 / WCSIM_QPE_BIN_DENOMINATOR
    raw = []
    for order in range(1, 5):
        moment = (high ** (order + 1) - low ** (order + 1)) / (
            (order + 1) * (high - low)
        )
        raw.append(float(probability @ moment))
    mean = raw[0]
    variance = raw[1] - mean * mean
    central3 = raw[2] - 3.0 * mean * raw[1] + 2.0 * mean**3
    central4 = (
        raw[3]
        - 4.0 * mean * raw[2]
        + 6.0 * mean * mean * raw[1]
        - 3.0 * mean**4
    )
    cumulant4 = central4 - 3.0 * variance * variance
    return mean, variance, central3, cumulant4


QPE_MEAN, QPE_VARIANCE, QPE_CUMULANT3, QPE_CUMULANT4 = _raw_moments()
QPE_STD = math.sqrt(QPE_VARIANCE)
QPE_SKEWNESS = QPE_CUMULANT3 / QPE_VARIANCE**1.5
QPE_EXCESS_KURTOSIS = QPE_CUMULANT4 / QPE_VARIANCE**2


def ski_threshold_acceptance(raw_charge_pe: np.ndarray | float) -> np.ndarray:
    """Return the SK-I digit-acceptance probability used by WCSim."""
    raw = np.asarray(raw_charge_pe, dtype=np.float64)
    x = raw + 0.1
    polynomial = -0.06374 + x * (
        3.748
        + x
        * (
            -63.23
            + x
            * (
                452.0
                + x
                * (
                    -1449.0
                    + x
                    * (
                        2513.0
                        + x
                        * (
                            -2529.0
                            + x
                            * (
                                1472.0
                                + x * (-452.2 + x * (51.34 + x * 2.370))
                            )
                        )
                    )
                )
            )
        )
    )
    # WCSim uses min(1, polynomial).  Its qpe support is non-negative and the
    # polynomial is non-negative there; clip roundoff defensively at zero.
    return np.where(x < 1.1, np.clip(polynomial, 0.0, 1.0), 1.0)


def _single_qpe_mass_grid(subbins: int) -> tuple[np.ndarray, float]:
    subbins = int(subbins)
    if subbins < 2:
        raise ValueError("subbins must be at least two")
    cdf = WCSIM_R14374_WCTE_QPE_CDF
    probabilities = np.diff(np.concatenate(([0.0], cdf)))
    first = WCSIM_QPE_INDEX_OFFSET
    if np.any(probabilities[:first] != 0.0):
        raise RuntimeError("bundled WCSim qpe CDF has unexpected negative support")
    active = probabilities[first:]
    mass = np.repeat(active / float(subbins), subbins)
    mass /= float(np.sum(mass))
    dx = 1.0 / (WCSIM_QPE_BIN_DENOMINATOR * float(subbins))
    return np.ascontiguousarray(mass), dx


@lru_cache(maxsize=8)
def _exact_response_grids(
    exact_n_max: int, subbins: int
) -> tuple[tuple[tuple[float, float, np.ndarray], ...], float]:
    """Build response grids as ``(zero_mass, x0, positive_density)``."""
    exact_n_max = int(exact_n_max)
    if exact_n_max < 1:
        raise ValueError("exact_n_max must be positive")
    single, dx = _single_qpe_mass_grid(subbins)
    noise_sigma = WCSIM_DIGITIZER_NOISE_SIGMA_PE
    pad = int(math.ceil(8.0 * noise_sigma / dx))
    offsets = np.arange(-pad, pad + 1, dtype=np.float64) * dx
    gaussian_density = np.exp(-0.5 * (offsets / noise_sigma) ** 2) / (
        math.sqrt(2.0 * math.pi) * noise_sigma
    )
    gaussian_density /= float(np.sum(gaussian_density) * dx)

    current = single
    output: list[tuple[float, float, np.ndarray]] = []
    for npe in range(1, exact_n_max + 1):
        raw_x0 = 0.5 * float(npe) * dx
        raw_x = raw_x0 + np.arange(current.size, dtype=np.float64) * dx
        acceptance = ski_threshold_acceptance(raw_x)
        accepted_mass = current * acceptance
        rejected_mass = float(np.sum(current * (1.0 - acceptance)))
        negative_after_smear = float(
            np.sum(accepted_mass * ndtr(-raw_x / noise_sigma))
        )
        zero_mass = min(max(rejected_mass + negative_after_smear, 0.0), 1.0)
        density = fftconvolve(accepted_mass, gaussian_density, mode="full")
        density = np.maximum(np.asarray(density, dtype=np.float64), 0.0)
        smeared_x0 = raw_x0 - float(pad) * dx
        output.append((zero_mass, smeared_x0, np.ascontiguousarray(density)))
        if npe != exact_n_max:
            current = fftconvolve(current, single, mode="full")
            current = np.maximum(np.asarray(current, dtype=np.float64), 0.0)
            current /= float(np.sum(current))
    return tuple(output), dx


def _edgeworth_density(raw_charge: np.ndarray, npe: int) -> np.ndarray:
    """Fourth-order density for high-occupancy summed qpe charge."""
    npe = int(npe)
    mean = float(npe) * QPE_MEAN
    variance = (
        float(npe) * QPE_VARIANCE + WCSIM_DIGITIZER_NOISE_SIGMA_PE**2
    )
    sigma = math.sqrt(variance)
    z = (raw_charge - mean) / sigma
    gaussian = np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sigma)
    standardized3 = float(npe) * QPE_CUMULANT3 / variance**1.5
    standardized4 = float(npe) * QPE_CUMULANT4 / variance**2
    h3 = z**3 - 3.0 * z
    h4 = z**4 - 6.0 * z**2 + 3.0
    h6 = z**6 - 15.0 * z**4 + 45.0 * z**2 - 15.0
    correction = (
        1.0
        + standardized3 * h3 / 6.0
        + standardized4 * h4 / 24.0
        + standardized3**2 * h6 / 72.0
    )
    return np.maximum(gaussian * correction, 1.0e-300)


def _edgeworth_density_many(
    raw_charge: np.ndarray, npe: np.ndarray
) -> np.ndarray:
    """Vectorized high-occupancy densities for many latent PE counts.

    The old setup loop rebuilt the same charge powers for every count from 25
    through 256.  Broadcasting the count and charge axes performs exactly the
    same fourth-order Edgeworth arithmetic in NumPy while eliminating hundreds
    of Python dispatches during each event's response-table construction.
    """
    charge = np.asarray(raw_charge, dtype=np.float64)[None, :]
    counts = np.asarray(npe, dtype=np.float64)[:, None]
    mean = counts * QPE_MEAN
    variance = counts * QPE_VARIANCE + WCSIM_DIGITIZER_NOISE_SIGMA_PE**2
    sigma = np.sqrt(variance)
    z = (charge - mean) / sigma
    gaussian = np.exp(-0.5 * z * z) / (
        math.sqrt(2.0 * math.pi) * sigma
    )
    standardized3 = counts * QPE_CUMULANT3 / variance**1.5
    standardized4 = counts * QPE_CUMULANT4 / variance**2
    z2 = z * z
    z3 = z2 * z
    z4 = z2 * z2
    z6 = z3 * z3
    h3 = z3 - 3.0 * z
    h4 = z4 - 6.0 * z2 + 3.0
    h6 = z6 - 15.0 * z4 + 45.0 * z2 - 15.0
    correction = (
        1.0
        + standardized3 * h3 / 6.0
        + standardized4 * h4 / 24.0
        + standardized3**2 * h6 / 72.0
    )
    return np.maximum(gaussian * correction, 1.0e-300)


def precompute_wcsim_compound_response(
    observed_charge: np.ndarray,
    *,
    n_cap: int,
    exact_n_max: int = 24,
    subbins: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute ``p(observed digit | N true PE)`` for one event.

    A zero array entry means no positive prompt digit.  It includes both the
    stochastic SK-I threshold rejection and the negligible accepted-but-negative
    electronics tail.  A positive entry is an analog charge density.
    """
    observed = np.asarray(observed_charge, dtype=np.float64)
    if observed.ndim != 1 or np.any(~np.isfinite(observed)) or np.any(observed < 0.0):
        raise ValueError("observed_charge must be a finite nonnegative vector")
    n_cap = int(n_cap)
    exact_n_max = min(int(exact_n_max), n_cap)
    if n_cap < 1 or exact_n_max < 1:
        raise ValueError("n_cap and exact_n_max must be positive")

    grids, dx = _exact_response_grids(exact_n_max, int(subbins))
    response = np.zeros((observed.size, n_cap + 1), dtype=np.float64)
    zero = observed <= 0.0
    positive = ~zero
    response[zero, 0] = 1.0
    raw_observed = observed[positive] / WCSIM_DIGITIZER_CHARGE_FACTOR

    for npe, (zero_mass, x0, density) in enumerate(grids, start=1):
        if np.any(zero):
            response[zero, npe] = zero_mass
        if np.any(positive):
            coordinate = (raw_observed - x0) / dx
            lower = np.floor(coordinate).astype(np.int64)
            fraction = coordinate - lower
            valid = (lower >= 0) & (lower + 1 < density.size)
            values = np.zeros(raw_observed.size, dtype=np.float64)
            if np.any(valid):
                indices = lower[valid]
                values[valid] = (
                    (1.0 - fraction[valid]) * density[indices]
                    + fraction[valid] * density[indices + 1]
                ) / WCSIM_DIGITIZER_CHARGE_FACTOR
            response[positive, npe] = values

    if np.any(positive) and exact_n_max < n_cap:
        high_counts = np.arange(
            exact_n_max + 1, n_cap + 1, dtype=np.float64
        )
        response[positive, exact_n_max + 1:] = (
            _edgeworth_density_many(raw_observed, high_counts).T
            / WCSIM_DIGITIZER_CHARGE_FACTOR
        )

    # The likelihood's Poisson truncation needs to reach the response support
    # implied by charge, even when a trial lambda is initially very small.
    effective_count = np.maximum(
        observed / (WCSIM_DIGITIZER_CHARGE_FACTOR * QPE_MEAN), 0.0
    )
    count_spread = (QPE_STD / QPE_MEAN) * np.sqrt(effective_count + 1.0)
    n_from_charge = np.ceil(effective_count + 12.0 * count_spread + 12.0).astype(
        np.int64
    )
    n_from_charge = np.minimum(n_from_charge, n_cap)
    return np.ascontiguousarray(response), np.ascontiguousarray(n_from_charge)


def response_metadata() -> dict[str, float | int | str]:
    """Return immutable physics metadata for calibration manifests."""
    return {
        "model": "wcsim_r14374_ski",
        "implementation_version": 1,
        "wcsim_source_commit": WCSIM_QPE_SOURCE_COMMIT,
        "qpe_mean": QPE_MEAN,
        "qpe_std": QPE_STD,
        "qpe_skewness": QPE_SKEWNESS,
        "qpe_excess_kurtosis": QPE_EXCESS_KURTOSIS,
        "digitizer_noise_sigma_pe": WCSIM_DIGITIZER_NOISE_SIGMA_PE,
        "digitizer_charge_factor": WCSIM_DIGITIZER_CHARGE_FACTOR,
        "qpe_cdf_entries": int(WCSIM_R14374_WCTE_QPE_CDF.size),
    }


__all__ = [
    "QPE_EXCESS_KURTOSIS",
    "QPE_MEAN",
    "QPE_SKEWNESS",
    "QPE_STD",
    "WCSIM_DIGITIZER_CHARGE_FACTOR",
    "WCSIM_DIGITIZER_NOISE_SIGMA_PE",
    "WCSIM_R14374_WCTE_QPE_CDF",
    "precompute_wcsim_compound_response",
    "response_metadata",
    "ski_threshold_acceptance",
]
