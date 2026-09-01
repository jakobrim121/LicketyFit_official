"""Parameter-free soft-plus-hard multiple scattering in liquid water.

The model is a resolution-invariant split of the screened Wentzel elastic
cross section at a declared polar separation angle.  Collisions below the
separation angle contribute their exact projected second moment to a Gaussian
Fermi--Eyges core.  Collisions above it form a marked Poisson process with the
same differential cross section.  The split therefore never subtracts a hard
moment from the empirical 13.6 MeV Rossi/Highland scale.

The cross section follows the Geant4 Physics Reference Manual, Single
Scattering, Eqs. 59--60.  The nuclear form factor is

    F^2(q) = [1 + (q R_N)^2 / 12]^-2.

Natural-water stoichiometry, liquid-water density, CODATA constants, and
measured proton/O-16 rms charge radii are the only material inputs.  There is
no WCSim-derived normalization or scattering multiplier.

All angular densities are axisymmetric about the incident tangent.  A hard
mark is stored as polar angle and azimuth and should be applied as an exact
three-dimensional tangent rotation; the small-angle two-vector helper exists
only for analytic and synthetic validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np


FINE_STRUCTURE = 1.0 / 137.035999084
HBARC_MEV_FM = 197.3269804
E2_MEV_FM = FINE_STRUCTURE * HBARC_MEV_FM
BOHR_RADIUS_FM = 0.529177210903e5
AVOGADRO = 6.02214076e23
ELECTRON_MASS_MEV = 0.51099895000
MUON_MASS_MEV = 105.6583755
FM2_TO_MM2 = 1.0e-24


@lru_cache(maxsize=8)
def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return np.ascontiguousarray(nodes), np.ascontiguousarray(weights)


@dataclass(frozen=True)
class WentzelAtom:
    atomic_number: int
    multiplicity: int
    rms_charge_radius_fm: float

    def validate(self) -> None:
        if int(self.atomic_number) < 1:
            raise ValueError("atomic number must be positive")
        if int(self.multiplicity) < 1:
            raise ValueError("atomic multiplicity must be positive")
        if not math.isfinite(float(self.rms_charge_radius_fm)) or float(
            self.rms_charge_radius_fm
        ) <= 0.0:
            raise ValueError("nuclear charge radius must be positive and finite")


@dataclass(frozen=True)
class HardScatterMark:
    theta_rad: float
    azimuth_rad: float
    atomic_number: int
    target: str

    @property
    def transverse_kick(self) -> np.ndarray:
        return np.asarray(
            (
                self.theta_rad * math.cos(self.azimuth_rad),
                self.theta_rad * math.sin(self.azimuth_rad),
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class WaterWentzelScattering:
    """Screened-Wentzel mixed scattering law for liquid water."""

    separation_angle_rad: float = 0.020
    density_g_cm3: float = 0.997
    molar_mass_g_mol: float = 18.01528
    projectile_mass_mev: float = MUON_MASS_MEV
    # Proton and O-16 rms charge radii.  At the 20 mrad separation used here,
    # the core moment is essentially insensitive to their last quoted digits;
    # they matter in the far nuclear tail.
    atoms: tuple[WentzelAtom, ...] = (
        WentzelAtom(1, 2, 0.8409),
        WentzelAtom(8, 1, 2.6991),
    )
    quadrature_order: int = 96

    def __post_init__(self) -> None:
        if not 0.0 < float(self.separation_angle_rad) < math.pi:
            raise ValueError("separation angle must lie in (0, pi)")
        if not math.isfinite(float(self.density_g_cm3)) or float(
            self.density_g_cm3
        ) <= 0.0:
            raise ValueError("water density must be positive and finite")
        if not math.isfinite(float(self.molar_mass_g_mol)) or float(
            self.molar_mass_g_mol
        ) <= 0.0:
            raise ValueError("molar mass must be positive and finite")
        if not math.isfinite(float(self.projectile_mass_mev)) or float(
            self.projectile_mass_mev
        ) <= ELECTRON_MASS_MEV:
            raise ValueError("projectile mass must exceed the electron mass")
        if int(self.quadrature_order) < 24:
            raise ValueError("quadrature order must be at least 24")
        if not self.atoms:
            raise ValueError("material must contain at least one atom")
        for atom in self.atoms:
            atom.validate()

    @property
    def molecule_density_per_mm3(self) -> float:
        return float(
            self.density_g_cm3 / self.molar_mass_g_mol * AVOGADRO / 1000.0
        )

    @property
    def electron_max_scattering_angle_rad(self) -> float:
        """Maximum lab deflection of a heavy projectile by a free electron."""
        return float(math.asin(ELECTRON_MASS_MEV / self.projectile_mass_mev))

    def _kinematics(self, kinetic_energy_mev):
        energy = np.asarray(kinetic_energy_mev, dtype=np.float64)
        if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
            raise ValueError("kinetic energy must be positive and finite")
        mass = float(self.projectile_mass_mev)
        total = energy + mass
        momentum = np.sqrt(np.maximum(energy * (energy + 2.0 * mass), 1e-30))
        beta = momentum / total
        return energy, momentum, beta

    @staticmethod
    def _scalar_or_array(value, template):
        out = np.asarray(value, dtype=np.float64)
        return float(out) if np.asarray(template).ndim == 0 else out

    def screening_parameter(self, kinetic_energy_mev, atomic_number: int):
        """Moliere--Bethe screening parameter A from the Wentzel law."""
        source = kinetic_energy_mev
        _, momentum, beta = self._kinematics(source)
        z = float(atomic_number)
        radius = 0.885 * BOHR_RADIUS_FM * z ** (-1.0 / 3.0)
        value = (
            (HBARC_MEV_FM / (2.0 * momentum * radius)) ** 2
            * (1.13 + 3.76 * (FINE_STRUCTURE * z / beta) ** 2)
        )
        return self._scalar_or_array(value, source)

    def _integrate_component(
        self,
        kinetic_energy_mev,
        atom: WentzelAtom,
        *,
        target: str,
        theta_low: float,
        theta_high: float,
        projected_moment: bool,
    ):
        """Integrate one atom/target channel in log(2A+1-cos(theta))."""
        source = kinetic_energy_mev
        _, momentum, beta = self._kinematics(source)
        scalar = np.asarray(source).ndim == 0
        momentum = np.atleast_1d(momentum)
        beta = np.atleast_1d(beta)
        z = float(atom.atomic_number)
        screening = np.atleast_1d(self.screening_parameter(source, atom.atomic_number))
        u_low = max(0.0, 1.0 - math.cos(float(theta_low)))
        u_high = min(2.0, 1.0 - math.cos(float(theta_high)))
        if u_high <= u_low:
            zero = np.zeros_like(momentum)
            return float(zero[0]) if scalar else zero

        nodes, weights = _gauss_legendre(int(self.quadrature_order))
        log_low = np.log(2.0 * screening + u_low)
        log_high = np.log(2.0 * screening + u_high)
        log_denominator = (
            0.5 * (log_high - log_low)[None, :] * nodes[:, None]
            + 0.5 * (log_high + log_low)[None, :]
        )
        denominator = np.exp(log_denominator)
        u = np.maximum(denominator - 2.0 * screening[None, :], 0.0)
        jacobian = 0.5 * (log_high - log_low)[None, :] * denominator

        prefactor = (E2_MEV_FM / np.maximum(momentum * beta, 1e-30)) ** 2
        if target == "nucleus":
            q_fm_inv = momentum[None, :] * np.sqrt(2.0 * u) / HBARC_MEV_FM
            form_factor = 1.0 / (
                1.0
                + (q_fm_inv * float(atom.rms_charge_radius_fm)) ** 2 / 12.0
            ) ** 2
            target_factor = z * z * form_factor
        elif target == "electron":
            target_factor = np.full_like(u, z)
        else:
            raise ValueError("target must be 'nucleus' or 'electron'")

        integrand = (
            2.0
            * math.pi
            * prefactor[None, :]
            * target_factor
            / denominator**2
        )
        if projected_moment:
            theta = np.arccos(np.clip(1.0 - u, -1.0, 1.0))
            integrand = integrand * (0.5 * theta * theta)
        integral_fm2 = np.sum(weights[:, None] * jacobian * integrand, axis=0)
        macroscopic = (
            self.molecule_density_per_mm3
            * int(atom.multiplicity)
            * FM2_TO_MM2
            * integral_fm2
        )
        return float(macroscopic[0]) if scalar else macroscopic

    def soft_projected_power_rad2_per_mm(self, kinetic_energy_mev):
        """Per-plane Gaussian core power from collisions below the split."""
        source = kinetic_energy_mev
        total = np.zeros_like(np.atleast_1d(np.asarray(source, dtype=float)))
        electron_high = min(
            float(self.separation_angle_rad),
            float(self.electron_max_scattering_angle_rad),
        )
        for atom in self.atoms:
            total += np.atleast_1d(
                self._integrate_component(
                    source,
                    atom,
                    target="nucleus",
                    theta_low=0.0,
                    theta_high=float(self.separation_angle_rad),
                    projected_moment=True,
                )
            )
            total += np.atleast_1d(
                self._integrate_component(
                    source,
                    atom,
                    target="electron",
                    theta_low=0.0,
                    theta_high=electron_high,
                    projected_moment=True,
                )
            )
        return self._scalar_or_array(total[0] if np.asarray(source).ndim == 0 else total, source)

    @lru_cache(maxsize=512)
    def _hard_component_rates(self, kinetic_energy_mev: float):
        energy = float(kinetic_energy_mev)
        rows: list[tuple[WentzelAtom, str, float, float]] = []
        separation = float(self.separation_angle_rad)
        for atom in self.atoms:
            nuclear_rate = float(
                self._integrate_component(
                    energy,
                    atom,
                    target="nucleus",
                    theta_low=separation,
                    theta_high=math.pi,
                    projected_moment=False,
                )
            )
            if nuclear_rate > 0.0:
                rows.append((atom, "nucleus", nuclear_rate, math.pi))
            if separation < self.electron_max_scattering_angle_rad:
                electron_rate = float(
                    self._integrate_component(
                        energy,
                        atom,
                        target="electron",
                        theta_low=separation,
                        theta_high=self.electron_max_scattering_angle_rad,
                        projected_moment=False,
                    )
                )
                if electron_rate > 0.0:
                    rows.append(
                        (
                            atom,
                            "electron",
                            electron_rate,
                            self.electron_max_scattering_angle_rad,
                        )
                    )
        return tuple(rows)

    def hard_scatter_rate_per_mm(self, kinetic_energy_mev):
        """Macroscopic marked-Poisson rate above the separation angle."""
        source = np.asarray(kinetic_energy_mev, dtype=float)
        if source.ndim == 0:
            return float(sum(row[2] for row in self._hard_component_rates(float(source))))
        flat = source.reshape(-1)
        out = np.asarray(
            [sum(row[2] for row in self._hard_component_rates(float(k))) for k in flat],
            dtype=np.float64,
        ).reshape(source.shape)
        return out

    def hard_projected_power_rad2_per_mm(self, kinetic_energy_mev):
        """Per-plane second-moment rate of the explicit hard marks."""
        source = kinetic_energy_mev
        total = np.zeros_like(np.atleast_1d(np.asarray(source, dtype=float)))
        separation = float(self.separation_angle_rad)
        for atom in self.atoms:
            total += np.atleast_1d(
                self._integrate_component(
                    source,
                    atom,
                    target="nucleus",
                    theta_low=separation,
                    theta_high=math.pi,
                    projected_moment=True,
                )
            )
            if separation < self.electron_max_scattering_angle_rad:
                total += np.atleast_1d(
                    self._integrate_component(
                        source,
                        atom,
                        target="electron",
                        theta_low=separation,
                        theta_high=self.electron_max_scattering_angle_rad,
                        projected_moment=True,
                    )
                )
        return self._scalar_or_array(total[0] if np.asarray(source).ndim == 0 else total, source)

    def rossi_projected_power_rad2_per_mm(self, kinetic_energy_mev):
        source = kinetic_energy_mev
        _, momentum, beta = self._kinematics(source)
        value = (13.6 / np.maximum(beta * momentum, 1e-30)) ** 2 / 360.8
        return self._scalar_or_array(value, source)

    def _sample_component_theta(
        self,
        kinetic_energy_mev: float,
        atom: WentzelAtom,
        target: str,
        theta_high: float,
        rng: np.random.Generator,
    ) -> float:
        """Exact rejection sample from the screened component cross section."""
        _, momentum_array, _ = self._kinematics(float(kinetic_energy_mev))
        momentum = float(momentum_array)
        a = 2.0 * float(
            self.screening_parameter(float(kinetic_energy_mev), atom.atomic_number)
        )
        u_low = 1.0 - math.cos(float(self.separation_angle_rad))
        u_high = 1.0 - math.cos(float(theta_high))
        inv_low = 1.0 / (a + u_low)
        inv_high = 1.0 / (a + u_high)
        for _ in range(10000):
            inverse = inv_low - float(rng.random()) * (inv_low - inv_high)
            u = max(1.0 / inverse - a, 0.0)
            if target == "nucleus":
                q_fm_inv = momentum * math.sqrt(2.0 * u) / HBARC_MEV_FM
                accept = 1.0 / (
                    1.0
                    + (q_fm_inv * float(atom.rms_charge_radius_fm)) ** 2 / 12.0
                ) ** 2
                if float(rng.random()) > accept:
                    continue
            return float(math.acos(max(-1.0, min(1.0, 1.0 - u))))
        raise RuntimeError("hard-scatter rejection sampler failed to accept")

    def sample_hard_mark(
        self, kinetic_energy_mev: float, rng: np.random.Generator
    ) -> HardScatterMark:
        components = self._hard_component_rates(float(kinetic_energy_mev))
        rates = np.asarray([row[2] for row in components], dtype=np.float64)
        total = float(np.sum(rates))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("hard-scatter rate is not positive")
        choice = int(np.searchsorted(np.cumsum(rates), float(rng.random()) * total))
        atom, target, _rate, theta_high = components[min(choice, len(components) - 1)]
        theta = self._sample_component_theta(
            float(kinetic_energy_mev), atom, target, theta_high, rng
        )
        return HardScatterMark(
            theta_rad=theta,
            azimuth_rad=float(2.0 * math.pi * rng.random()),
            atomic_number=int(atom.atomic_number),
            target=str(target),
        )

    def hard_mark_log_intensity_per_mm_rad2(
        self,
        kinetic_energy_mev: float,
        mark: HardScatterMark,
    ) -> float:
        """Log marked intensity density with respect to ``dtheta dphi ds``.

        Integrating this density over azimuth, polar angle above the declared
        split, and path length gives the hard marked-Poisson mean.  Exposing the
        density permits exact prior ratios when continuous range moves
        transport existing marks instead of redrawing their observable
        curvature.
        """
        energy = float(kinetic_energy_mev)
        theta = float(mark.theta_rad)
        if (
            not math.isfinite(theta)
            or theta < float(self.separation_angle_rad)
            or theta > math.pi
        ):
            return -math.inf
        atom = next(
            (
                candidate
                for candidate in self.atoms
                if int(candidate.atomic_number) == int(mark.atomic_number)
            ),
            None,
        )
        if atom is None:
            return -math.inf
        target = str(mark.target)
        if target == "electron" and theta > self.electron_max_scattering_angle_rad:
            return -math.inf
        if target not in {"nucleus", "electron"}:
            return -math.inf
        _, momentum_array, beta_array = self._kinematics(energy)
        momentum = float(momentum_array)
        beta = float(beta_array)
        screening = float(self.screening_parameter(energy, atom.atomic_number))
        u = 1.0 - math.cos(theta)
        denominator = 2.0 * screening + u
        z = float(atom.atomic_number)
        if target == "nucleus":
            q_fm_inv = momentum * math.sqrt(max(2.0 * u, 0.0)) / HBARC_MEV_FM
            factor = z * z / (
                1.0
                + (q_fm_inv * float(atom.rms_charge_radius_fm)) ** 2 / 12.0
            ) ** 2
        else:
            factor = z
        density = (
            self.molecule_density_per_mm3
            * int(atom.multiplicity)
            * FM2_TO_MM2
            * (E2_MEV_FM / max(momentum * beta, 1.0e-30)) ** 2
            * factor
            * math.sin(theta)
            / denominator**2
        )
        if not math.isfinite(density) or density <= 0.0:
            return -math.inf
        return float(math.log(density))

    def sample_constant_energy_increment(
        self,
        kinetic_energy_mev: float,
        thickness_mm: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, tuple[HardScatterMark, ...]]:
        """Sample a small-angle two-vector increment for prior validation."""
        thickness = float(thickness_mm)
        if not math.isfinite(thickness) or thickness < 0.0:
            raise ValueError("thickness must be non-negative and finite")
        soft_power, hard_rate = self._constant_energy_sampling_parameters(
            float(kinetic_energy_mev)
        )
        soft_variance = soft_power * thickness
        increment = rng.normal(scale=math.sqrt(max(soft_variance, 0.0)), size=2)
        count = int(
            rng.poisson(
                hard_rate * thickness
            )
        )
        marks = tuple(
            self.sample_hard_mark(float(kinetic_energy_mev), rng)
            for _ in range(count)
        )
        for mark in marks:
            increment += mark.transverse_kick
        return np.ascontiguousarray(increment), marks

    @lru_cache(maxsize=512)
    def _constant_energy_sampling_parameters(
        self, kinetic_energy_mev: float
    ) -> tuple[float, float]:
        return (
            float(self.soft_projected_power_rad2_per_mm(kinetic_energy_mev)),
            float(self.hard_scatter_rate_per_mm(kinetic_energy_mev)),
        )

    def hard_mark_characteristic_j0(
        self, kinetic_energy_mev: float, radial_frequency: Sequence[float] | float
    ):
        """Return E[J0(k theta)] for one hard mark by deterministic quadrature."""
        from scipy.special import j0

        frequency = np.asarray(radial_frequency, dtype=np.float64)
        components = self._hard_component_rates(float(kinetic_energy_mev))
        total_rate = float(sum(row[2] for row in components))
        if total_rate <= 0.0:
            return np.ones_like(frequency)
        numerator = np.zeros_like(frequency, dtype=np.float64)
        for atom, target, _rate, theta_high in components:
            # Integrate the same component density with J0(k theta).  A direct
            # log-denominator quadrature mirrors _integrate_component.
            _, momentum_array, beta_array = self._kinematics(float(kinetic_energy_mev))
            momentum = float(momentum_array)
            beta = float(beta_array)
            z = float(atom.atomic_number)
            screening = float(
                self.screening_parameter(float(kinetic_energy_mev), atom.atomic_number)
            )
            u_low = 1.0 - math.cos(float(self.separation_angle_rad))
            u_high = 1.0 - math.cos(float(theta_high))
            nodes, weights = _gauss_legendre(int(self.quadrature_order))
            log_low = math.log(2.0 * screening + u_low)
            log_high = math.log(2.0 * screening + u_high)
            log_denominator = (
                0.5 * (log_high - log_low) * nodes
                + 0.5 * (log_high + log_low)
            )
            denominator = np.exp(log_denominator)
            u = np.maximum(denominator - 2.0 * screening, 0.0)
            theta = np.arccos(np.clip(1.0 - u, -1.0, 1.0))
            jacobian = 0.5 * (log_high - log_low) * denominator
            if target == "nucleus":
                q_fm_inv = momentum * np.sqrt(2.0 * u) / HBARC_MEV_FM
                factor = z * z / (
                    1.0
                    + (q_fm_inv * float(atom.rms_charge_radius_fm)) ** 2 / 12.0
                ) ** 2
            else:
                factor = np.full_like(u, z)
            density = (
                2.0
                * math.pi
                * (E2_MEV_FM / max(momentum * beta, 1e-30)) ** 2
                * factor
                / denominator**2
            )
            macro_weight = (
                self.molecule_density_per_mm3
                * int(atom.multiplicity)
                * FM2_TO_MM2
            )
            numerator += macro_weight * np.sum(
                weights[:, None]
                * (jacobian * density)[:, None]
                * j0(theta[:, None] * frequency.reshape(1, -1)),
                axis=0,
            ).reshape(frequency.shape)
        value = numerator / total_rate
        return float(value) if frequency.ndim == 0 else value

    def projected_increment_characteristic(
        self,
        kinetic_energy_mev: float,
        thickness_mm: float,
        radial_frequency: Sequence[float] | float,
    ):
        """Characteristic function of the compound 2-D angular increment."""
        frequency = np.asarray(radial_frequency, dtype=np.float64)
        thickness = float(thickness_mm)
        soft = float(self.soft_projected_power_rad2_per_mm(kinetic_energy_mev))
        rate = float(self.hard_scatter_rate_per_mm(kinetic_energy_mev))
        mark_cf = np.asarray(
            self.hard_mark_characteristic_j0(kinetic_energy_mev, frequency)
        )
        value = np.exp(
            -0.5 * soft * thickness * frequency**2
            + rate * thickness * (mark_cf - 1.0)
        )
        return float(value) if frequency.ndim == 0 else value


__all__ = [
    "HardScatterMark",
    "WaterWentzelScattering",
    "WentzelAtom",
]
