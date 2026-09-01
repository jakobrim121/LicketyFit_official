"""Parameter-free stopping-range straggling for charged particles in water.

The mean range remains the checked-in continuous-slowing-down table.  The
variance is propagated from the second moment of Geant4's default
``G4UniversalFluctuation`` thin-absorber model.  No event-level WCSim range
distribution or fitted scale enters this module.

For stopping power ``S(K)=-dK/dx`` and local loss-variance rate ``q(K)``, the
linear first-passage approximation is

    Var[R(K0)] = integral(q(K) / S(K)^3 dK).

``G4UniversalFluctuation`` partitions the continuous mean loss between an
effective excitation and a 1/T^2 ionisation spectrum.  In the many-collision
limit their second moment per unit length is

    q = S * c * ((1-r) I fw + r (Tup-E0)/log(Tup/E0)),

with the source constants r=0.56, fw=4, E0=10 eV.  ``c`` is Geant4's small-cut
width correction and ``Tup`` is the smaller of the electron production-cut
energy and the kinematic maximum transfer.  The WCTE 100 mm electron range cut
is above the muon transfer maximum in the 100--500 MeV region, but the cut is
kept explicit for transfer to other configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from .particle_cherenkov_model import (
    _ensure_tables_loaded,
    canonical_particle_name,
    cherenkov_threshold_kinetic_mev,
    particle_mass_mev,
)


ELECTRON_MASS_MEV = 0.51099895
DEFAULT_MEAN_EXCITATION_ENERGY_MEV = 78.0e-6
DEFAULT_MIN_IONISATION_ENERGY_MEV = 10.0e-6
DEFAULT_IONISATION_FRACTION = 0.56
DEFAULT_EXCITATION_WIDTH_FACTOR = 4.0


def maximum_electron_transfer_mev(
    kinetic_energy_mev,
    *,
    particle_mass_mev_value: float,
) -> np.ndarray:
    """Maximum kinetic-energy transfer to an electron by a heavy particle."""

    kinetic = np.asarray(kinetic_energy_mev, dtype=np.float64)
    mass = float(particle_mass_mev_value)
    gamma = 1.0 + kinetic / mass
    beta_gamma_sq = np.maximum(gamma * gamma - 1.0, 0.0)
    mass_ratio = ELECTRON_MASS_MEV / mass
    denominator = 1.0 + 2.0 * gamma * mass_ratio + mass_ratio * mass_ratio
    return 2.0 * ELECTRON_MASS_MEV * beta_gamma_sq / denominator


@dataclass(frozen=True)
class UniversalFluctuationConfig:
    """Published/material inputs to the continuous-loss moment."""

    mean_excitation_energy_mev: float = DEFAULT_MEAN_EXCITATION_ENERGY_MEV
    minimum_ionisation_energy_mev: float = DEFAULT_MIN_IONISATION_ENERGY_MEV
    ionisation_fraction: float = DEFAULT_IONISATION_FRACTION
    excitation_width_factor: float = DEFAULT_EXCITATION_WIDTH_FACTOR
    # ``inf`` is correct whenever the configured range cut converts to an
    # electron energy above Tmax.  A finite independently calculated cut may be
    # supplied for another detector/process configuration.
    electron_production_cut_mev: float = math.inf

    def validate(self) -> None:
        if self.mean_excitation_energy_mev <= 0.0:
            raise ValueError("mean excitation energy must be positive")
        if self.minimum_ionisation_energy_mev <= 0.0:
            raise ValueError("minimum ionisation energy must be positive")
        if not 0.0 < self.ionisation_fraction < 1.0:
            raise ValueError("ionisation fraction must lie in (0,1)")
        if self.excitation_width_factor <= 0.0:
            raise ValueError("excitation width factor must be positive")
        if self.electron_production_cut_mev <= 0.0:
            raise ValueError("electron production cut must be positive")


def universal_loss_variance_rate(
    kinetic_energy_mev,
    stopping_power_mev_per_mm,
    *,
    particle_mass_mev_value: float,
    config: UniversalFluctuationConfig = UniversalFluctuationConfig(),
) -> np.ndarray:
    """Return ``q(K)=d Var[dE]/dx`` in MeV^2/mm."""

    config.validate()
    kinetic = np.asarray(kinetic_energy_mev, dtype=np.float64)
    stopping = np.asarray(stopping_power_mev_per_mm, dtype=np.float64)
    tmax = maximum_electron_transfer_mev(
        kinetic, particle_mass_mev_value=particle_mass_mev_value
    )
    upper = np.minimum(tmax, float(config.electron_production_cut_mev))
    e0 = float(config.minimum_ionisation_energy_mev)
    upper = np.maximum(upper, np.nextafter(e0, math.inf))

    # This is the scaling applied around SampleGlandz in the Geant4 source.
    width_correction = np.minimum(1.0 + 0.5e-3 / upper, 1.5)
    r = float(config.ionisation_fraction)
    excitation_second_moment = (
        (1.0 - r)
        * float(config.mean_excitation_energy_mev)
        * float(config.excitation_width_factor)
    )
    ionisation_second_moment = r * (upper - e0) / np.log(upper / e0)
    return stopping * width_correction * (
        excitation_second_moment + ionisation_second_moment
    )


@lru_cache(maxsize=16)
def _range_moment_table(
    particle: str,
    config: UniversalFluctuationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pname = canonical_particle_name(particle)
    config.validate()
    tables = _ensure_tables_loaded(pname)
    threshold = float(cherenkov_threshold_kinetic_mev(pname))
    energy = np.asarray(tables["master_ke"], dtype=np.float64)
    mean_range = np.asarray(tables["master_range"], dtype=np.float64)
    keep = np.isfinite(energy) & np.isfinite(mean_range) & (energy > threshold)
    energy = np.concatenate(([threshold], energy[keep]))
    mean_range = np.concatenate(([0.0], mean_range[keep]))

    order = np.argsort(energy)
    energy = energy[order]
    mean_range = mean_range[order]
    energy, unique_index = np.unique(energy, return_index=True)
    mean_range = mean_range[unique_index]
    d_range_d_energy = np.gradient(mean_range, energy)
    stopping = 1.0 / np.maximum(d_range_d_energy, 1.0e-30)
    q = universal_loss_variance_rate(
        energy,
        stopping,
        particle_mass_mev_value=float(particle_mass_mev(pname)),
        config=config,
    )
    integrand = q / np.maximum(stopping, 1.0e-30) ** 3
    variance = np.empty_like(energy)
    variance[0] = 0.0
    variance[1:] = np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(energy)
    )
    return (
        np.ascontiguousarray(energy),
        np.ascontiguousarray(mean_range),
        np.ascontiguousarray(np.maximum(variance, 0.0)),
    )


@dataclass(frozen=True)
class StoppingRangeStraggling:
    """Continuous ``L | K0`` moment model tied to the existing range table."""

    particle: str = "muon"
    fluctuation: UniversalFluctuationConfig = UniversalFluctuationConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "particle", canonical_particle_name(self.particle))
        self.fluctuation.validate()

    @property
    def threshold_mev(self) -> float:
        return float(cherenkov_threshold_kinetic_mev(self.particle))

    def _table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _range_moment_table(self.particle, self.fluctuation)

    def mean_range_mm(self, kinetic_energy_mev):
        energy, mean_range, _ = self._table()
        value = np.asarray(kinetic_energy_mev, dtype=np.float64)
        out = np.interp(value, energy, mean_range, left=0.0, right=mean_range[-1])
        return float(out) if out.ndim == 0 else out

    def variance_mm2(self, kinetic_energy_mev):
        energy, _, variance = self._table()
        value = np.asarray(kinetic_energy_mev, dtype=np.float64)
        out = np.interp(value, energy, variance, left=0.0, right=variance[-1])
        return float(out) if out.ndim == 0 else out

    def sigma_mm(self, kinetic_energy_mev):
        variance = np.asarray(self.variance_mm2(kinetic_energy_mev))
        out = np.sqrt(np.maximum(variance, 0.0))
        return float(out) if out.ndim == 0 else out

    def realized_range_mm(self, kinetic_energy_mev, z_range):
        out = np.asarray(self.mean_range_mm(kinetic_energy_mev)) + np.asarray(
            self.sigma_mm(kinetic_energy_mev)
        ) * np.asarray(z_range, dtype=np.float64)
        return float(out) if out.ndim == 0 else out

    def z_range(self, kinetic_energy_mev, realized_range_mm):
        sigma, delta = np.broadcast_arrays(
            np.asarray(self.sigma_mm(kinetic_energy_mev), dtype=np.float64),
            (
                np.asarray(realized_range_mm, dtype=np.float64)
                - np.asarray(
                    self.mean_range_mm(kinetic_energy_mev), dtype=np.float64
                )
            ),
        )
        out = np.full(sigma.shape, np.nan, dtype=np.float64)
        positive_width = sigma > 0.0
        np.divide(delta, sigma, out=out, where=positive_width)

        # At the Cherenkov threshold both the mean residual range and its width
        # are exactly zero. This is a point mass, not a narrow Gaussian: use the
        # canonical latent coordinate z_R=0 on support and an infinite signed
        # standardized displacement off support. No empirical sigma floor is
        # introduced into the physical range moment.
        zero_width = sigma == 0.0
        out = np.where(zero_width & (delta == 0.0), 0.0, out)
        out = np.where(
            zero_width & (delta != 0.0),
            np.copysign(np.inf, delta),
            out,
        )
        return float(out) if out.ndim == 0 else out

    def logpdf_range(self, realized_range_mm, kinetic_energy_mev):
        sigma, delta = np.broadcast_arrays(
            np.asarray(self.sigma_mm(kinetic_energy_mev), dtype=np.float64),
            (
                np.asarray(realized_range_mm, dtype=np.float64)
                - np.asarray(
                    self.mean_range_mm(kinetic_energy_mev), dtype=np.float64
                )
            ),
        )
        out = np.full(sigma.shape, np.nan, dtype=np.float64)
        positive_width = sigma > 0.0
        z = np.zeros(sigma.shape, dtype=np.float64)
        np.divide(delta, sigma, out=z, where=positive_width)
        out[positive_width] = (
            -0.5 * z[positive_width] * z[positive_width]
            - np.log(sigma[positive_width])
            - 0.5 * math.log(2.0 * math.pi)
        )
        zero_width = sigma == 0.0
        out[zero_width & (delta == 0.0)] = 0.0
        out[zero_width & (delta != 0.0)] = -np.inf
        return float(out) if out.ndim == 0 else out


__all__ = [
    "StoppingRangeStraggling",
    "UniversalFluctuationConfig",
    "maximum_electron_transfer_mev",
    "universal_loss_variance_rate",
]
