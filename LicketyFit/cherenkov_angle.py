import numpy as np

def cherenkov_angle_deg(n, mass_mev, kinetic_energy_mev):
    """
    Return the Cherenkov angle in degrees.

    Parameters
    ----------
    n : float or array-like
        Index of refraction of the medium.
    mass_mev : float
        Particle rest mass in MeV/c^2.
    kinetic_energy_mev : float or array-like
        Particle kinetic energy in MeV.

    Returns
    -------
    theta_deg : float or np.ndarray
        Cherenkov angle in degrees.

    Notes
    -----
    Uses:
        gamma = 1 + KE / m
        beta = sqrt(1 - 1/gamma^2)
        cos(theta) = 1 / (n * beta)

    If n*beta < 1, no Cherenkov light is produced, and np.nan is returned.
    """
    n = np.asarray(n, dtype=float)
    kinetic_energy_mev = np.asarray(kinetic_energy_mev, dtype=float)

    gamma = 1.0 + kinetic_energy_mev / mass_mev
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    nbeta = n * beta

    theta_deg = np.full(np.shape(nbeta), np.nan, dtype=float)
    mask = nbeta >= 1.0

    theta_deg[mask] = np.degrees(np.arccos(1.0 / nbeta[mask]))

    if theta_deg.shape == ():
        return float(theta_deg)

    return theta_deg


import numpy as np

def n_water(lambda_nm):
    """
    Refractive index of distilled water vs wavelength using the
    Daimon & Masumura (2007) 4-term Sellmeier fit at 21.5 C.

    Parameters
    ----------
    lambda_nm : float or array-like
        Wavelength in nm.

    Returns
    -------
    n : float or np.ndarray
        Refractive index of water.

    Notes
    -----
    Valid approximately over 182–1129 nm.
    Wavelength is converted to micrometres for the formula.
    """
    lam_um = np.asarray(lambda_nm, dtype=float) / 1000.0
    lam2 = lam_um**2

    n2 = (
        1.0
        + (5.689093832e-1 * lam2) / (lam2 - 5.110301794e-3)
        + (1.719708856e-1 * lam2) / (lam2 - 1.825180155e-2)
        + (2.062501582e-2 * lam2) / (lam2 - 2.624158904e-2)
        + (1.123965424e-1 * lam2) / (lam2 - 1.067505178e1)
    )

    n = np.sqrt(n2)

    if np.ndim(n) == 0:
        return float(n)
    return n