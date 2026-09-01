"""Coherent trajectories for the screened-Wentzel mixed MCS process.

The path prior is the direct trajectory counterpart of
:mod:`LicketyFit.mcs_mixed_scattering`:

* collisions below the declared separation angle form a Gaussian
  Fermi--Eyges process with the integrated soft Wentzel projected moment;
* collisions above it are explicit marks of a non-homogeneous Poisson process
  along the mean energy-loss coordinate;
* every hard mark is applied as an exact three-dimensional tangent rotation.

The optical grid is not a scattering binning convention.  Hard marks retain
their continuous arc-length coordinates and are inserted as zero-length
left/right path nodes, so changing the smooth path grid cannot move or smear a
scatter.  No WCSim-derived normalization or tuned angular scale enters here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numba import njit

from . import Emitter as emod
from .mcs_mixed_scattering import HardScatterMark, WaterWentzelScattering
from .mcs_process import configured_transverse_basis


@dataclass(frozen=True)
class PositionedHardScatter:
    """One hard Wentzel mark at a continuous path arc length."""

    s_mm: float
    theta_rad: float
    azimuth_rad: float
    atomic_number: int
    target: str

    @classmethod
    def from_mark(cls, s_mm: float, mark: HardScatterMark):
        return cls(
            s_mm=float(s_mm),
            theta_rad=float(mark.theta_rad),
            azimuth_rad=float(mark.azimuth_rad),
            atomic_number=int(mark.atomic_number),
            target=str(mark.target),
        )

    @property
    def mark(self) -> HardScatterMark:
        return HardScatterMark(
            theta_rad=float(self.theta_rad),
            azimuth_rad=float(self.azimuth_rad),
            atomic_number=int(self.atomic_number),
            target=str(self.target),
        )


@dataclass(frozen=True)
class MixedMCSLatent:
    """Non-centred soft coordinates and continuous hard-scatter marks."""

    soft_coefficients: np.ndarray
    hard_scatters: tuple[PositionedHardScatter, ...] = ()

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.soft_coefficients, dtype=np.float64).reshape(-1)
        if coefficients.size == 0 or coefficients.size % 2:
            raise ValueError("soft coefficients require equal non-empty plane blocks")
        if np.any(~np.isfinite(coefficients)):
            raise ValueError("soft coefficients must be finite")
        scatters = tuple(sorted(self.hard_scatters, key=lambda mark: mark.s_mm))
        for mark in scatters:
            values = (mark.s_mm, mark.theta_rad, mark.azimuth_rad)
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError("hard-scatter coordinates must be finite")
            if float(mark.theta_rad) <= 0.0:
                raise ValueError("hard-scatter angle must be positive")
        object.__setattr__(self, "soft_coefficients", np.ascontiguousarray(coefficients))
        object.__setattr__(self, "hard_scatters", scatters)


def _weighted_fe_kl_basis(
    emitter,
    scattering: WaterWentzelScattering,
    n_modes_per_plane: int,
    n_grid: int,
):
    """Build raw FE displacement/slope modes from exact soft Wentzel power."""
    n_grid = max(17, min(int(n_grid), 401))
    n_modes = max(1, min(int(n_modes_per_plane), n_grid - 2))
    length = max(float(getattr(emitter, "length", 0.0)), 0.0)
    if length <= 0.0:
        grid = np.asarray((0.0,), dtype=np.float64)
        zeros = np.zeros((1, n_modes), dtype=np.float64)
        return grid, zeros, zeros.copy(), np.zeros(n_modes, dtype=np.float64)

    grid = np.linspace(0.0, length, n_grid, dtype=np.float64)
    mean_range = float(getattr(emitter, "range_to_threshold_mm", length))
    energy = np.asarray(
        emitter.muon_energy_at_s_array(grid, mean_range), dtype=np.float64
    )
    power = np.asarray(
        scattering.soft_projected_power_rad2_per_mm(energy), dtype=np.float64
    )
    if np.any(~np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("soft Wentzel scattering power is not physical")

    ds = np.diff(grid)
    integral0 = np.zeros_like(grid)
    integral1 = np.zeros_like(grid)
    integral2 = np.zeros_like(grid)
    integral0[1:] = np.cumsum(0.5 * (power[1:] + power[:-1]) * ds)
    integral1[1:] = np.cumsum(
        0.5 * (grid[1:] * power[1:] + grid[:-1] * power[:-1]) * ds
    )
    integral2[1:] = np.cumsum(
        0.5
        * (grid[1:] ** 2 * power[1:] + grid[:-1] ** 2 * power[:-1])
        * ds
    )
    minimum = np.minimum.outer(np.arange(grid.size), np.arange(grid.size))
    covariance = (
        np.outer(grid, grid) * integral0[minimum]
        - (grid[:, None] + grid[None, :]) * integral1[minimum]
        + integral2[minimum]
    )

    mass = float(getattr(emitter, "particle_mass", scattering.projectile_mass_mev))
    refractive_index = float(getattr(emitter, "n", 1.344))
    frank_tamm = np.asarray(
        emod._cherenkov_weight_from_energy(energy, mass, refractive_index),
        dtype=np.float64,
    )
    quadrature = np.ones_like(grid)
    quadrature[[0, -1]] = 0.5
    quadrature *= length / max(grid.size - 1, 1)
    weight = frank_tamm * quadrature
    active = np.flatnonzero(weight > 1.0e-14)
    shapes = np.zeros((grid.size, n_modes), dtype=np.float64)
    explained = np.zeros(n_modes, dtype=np.float64)
    if active.size >= 3:
        active_weight = weight[active]
        active_covariance = covariance[np.ix_(active, active)]
        square_root_weight = np.sqrt(active_weight)
        weighted = (
            square_root_weight[:, None]
            * active_covariance
            * square_root_weight[None, :]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (weighted + weighted.T))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        total = max(float(np.sum(eigenvalues)), 1.0e-300)
        retained = min(
            n_modes,
            int(np.count_nonzero(eigenvalues > 1.0e-18 * eigenvalues[0])),
        )
        for index in range(retained):
            vector = eigenvectors[:, index]
            shapes[:, index] = (
                covariance[:, active] @ (square_root_weight * vector)
                / math.sqrt(max(float(eigenvalues[index]), 1.0e-300))
            )
            maximum = int(np.argmax(np.abs(shapes[:, index])))
            if shapes[maximum, index] < 0.0:
                shapes[:, index] *= -1.0
        explained[:] = np.cumsum(eigenvalues[:n_modes]) / total
    shapes[0, :] = 0.0
    slopes = np.gradient(shapes, grid, axis=0, edge_order=2)
    slopes[0, :] = 0.0
    return tuple(
        np.ascontiguousarray(value, dtype=np.float64)
        for value in (grid, shapes, slopes, explained)
    )


def _rotate_frame(tangent, first, second, theta_first: float, theta_second: float):
    """Rotate a local orthonormal frame toward one transverse angular mark."""
    angle = math.hypot(float(theta_first), float(theta_second))
    if angle <= 0.0:
        return tangent, first, second
    transverse = (
        float(theta_first) * first + float(theta_second) * second
    ) / angle
    axis = np.cross(tangent, transverse)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-15:
        return tangent, first, second
    axis /= axis_norm
    cosine = math.cos(angle)
    sine = math.sin(angle)

    def rotate(vector):
        return (
            cosine * vector
            + sine * np.cross(axis, vector)
            + (1.0 - cosine) * float(axis @ vector) * axis
        )

    new_tangent = rotate(tangent)
    new_first = rotate(first)
    new_second = rotate(second)
    return (
        np.ascontiguousarray(new_tangent / np.linalg.norm(new_tangent)),
        np.ascontiguousarray(new_first / np.linalg.norm(new_first)),
        np.ascontiguousarray(new_second / np.linalg.norm(new_second)),
    )


@njit(cache=True)
def _rotate_frame_components(tangent, first, second, theta_first, theta_second):
    """Numba equivalent of :func:`_rotate_frame` for the path hot loop."""
    angle = math.hypot(theta_first, theta_second)
    if angle <= 0.0:
        return tangent, first, second
    transverse = (theta_first * first + theta_second * second) / angle
    axis = np.empty(3, dtype=np.float64)
    axis[0] = tangent[1] * transverse[2] - tangent[2] * transverse[1]
    axis[1] = tangent[2] * transverse[0] - tangent[0] * transverse[2]
    axis[2] = tangent[0] * transverse[1] - tangent[1] * transverse[0]
    axis_norm = math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
    if axis_norm <= 1.0e-15:
        return tangent, first, second
    axis /= axis_norm
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine

    rotated = np.empty((3, 3), dtype=np.float64)
    vectors = (tangent, first, second)
    for vector_index in range(3):
        vector = vectors[vector_index]
        cross0 = axis[1] * vector[2] - axis[2] * vector[1]
        cross1 = axis[2] * vector[0] - axis[0] * vector[2]
        cross2 = axis[0] * vector[1] - axis[1] * vector[0]
        dot = axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]
        rotated[vector_index, 0] = (
            cosine * vector[0] + sine * cross0 + one_minus_cosine * dot * axis[0]
        )
        rotated[vector_index, 1] = (
            cosine * vector[1] + sine * cross1 + one_minus_cosine * dot * axis[1]
        )
        rotated[vector_index, 2] = (
            cosine * vector[2] + sine * cross2 + one_minus_cosine * dot * axis[2]
        )
        norm = math.sqrt(
            rotated[vector_index, 0] ** 2
            + rotated[vector_index, 1] ** 2
            + rotated[vector_index, 2] ** 2
        )
        rotated[vector_index] /= norm
    return rotated[0].copy(), rotated[1].copy(), rotated[2].copy()


@njit(cache=True)
def _propagate_mixed_path_frames(
    grid,
    q_first,
    q_second,
    mark_indptr,
    mark_theta_first,
    mark_theta_second,
    start,
    direction,
    basis_first,
    basis_second,
):
    """Propagate the exact frame rotations without Python per-node overhead."""
    output_size = grid.size + mark_theta_first.size
    output_s = np.empty(output_size, dtype=np.float64)
    output_position = np.empty((output_size, 3), dtype=np.float64)
    output_tangent = np.empty((output_size, 3), dtype=np.float64)
    tangent = direction.copy()
    local_first = basis_first.copy()
    local_second = basis_second.copy()
    position = start.copy()
    output_index = 0

    output_s[output_index] = grid[0]
    output_position[output_index] = position
    output_tangent[output_index] = tangent
    output_index += 1
    for mark_index in range(mark_indptr[0], mark_indptr[1]):
        tangent, local_first, local_second = _rotate_frame_components(
            tangent,
            local_first,
            local_second,
            mark_theta_first[mark_index],
            mark_theta_second[mark_index],
        )
        output_s[output_index] = grid[0]
        output_position[output_index] = position
        output_tangent[output_index] = tangent
        output_index += 1

    for grid_index in range(1, grid.size):
        old_tangent = tangent.copy()
        tangent, local_first, local_second = _rotate_frame_components(
            tangent,
            local_first,
            local_second,
            q_first[grid_index] - q_first[grid_index - 1],
            q_second[grid_index] - q_second[grid_index - 1],
        )
        distance = grid[grid_index] - grid[grid_index - 1]
        position += 0.5 * (old_tangent + tangent) * distance
        output_s[output_index] = grid[grid_index]
        output_position[output_index] = position
        output_tangent[output_index] = tangent
        output_index += 1
        for mark_index in range(
            mark_indptr[grid_index], mark_indptr[grid_index + 1]
        ):
            tangent, local_first, local_second = _rotate_frame_components(
                tangent,
                local_first,
                local_second,
                mark_theta_first[mark_index],
                mark_theta_second[mark_index],
            )
            output_s[output_index] = grid[grid_index]
            output_position[output_index] = position
            output_tangent[output_index] = tangent
            output_index += 1
    return output_s, output_position, output_tangent


class MixedMCSPathPrior:
    """Fixed-emitter mixed-MCS path prior and prior-reversible proposals."""

    def __init__(
        self,
        emitter,
        *,
        modes_per_plane: int = 12,
        grid_points: int = 41,
        transport_grid_points: int = 161,
        scattering: WaterWentzelScattering | None = None,
    ):
        self.emitter = emitter
        self.scattering = WaterWentzelScattering() if scattering is None else scattering
        self.modes_per_plane = int(modes_per_plane)
        self.grid_points = int(grid_points)
        self.transport_grid_points = max(41, int(transport_grid_points))
        if self.modes_per_plane < 1:
            raise ValueError("modes_per_plane must be positive")
        if self.grid_points < 17:
            raise ValueError("grid_points must be at least 17")
        particle = emod.canonical_particle_name(
            getattr(emitter, "particle_name", "muon")
        )
        if particle not in {"muon", "mu+", "mu-"} and not particle.startswith("mu"):
            raise ValueError("the current mixed scattering material law is for muons")
        self._soft_grid, self._soft_shapes, self._soft_slopes, self._soft_fraction = (
            _weighted_fe_kl_basis(
                emitter,
                self.scattering,
                self.modes_per_plane,
                self.grid_points,
            )
        )
        self.length_mm = float(getattr(emitter, "length", 0.0))
        if not math.isfinite(self.length_mm) or self.length_mm <= 0.0:
            raise ValueError("mixed path requires a positive visible length")
        self.mean_range_mm = float(
            getattr(emitter, "range_to_threshold_mm", self.length_mm)
        )
        self._rate_grid = np.linspace(
            0.0, self.length_mm, self.transport_grid_points, dtype=np.float64
        )
        rate_energy = np.asarray(
            emitter.muon_energy_at_s_array(self._rate_grid, self.mean_range_mm),
            dtype=np.float64,
        )
        self._rate = np.asarray(
            self.scattering.hard_scatter_rate_per_mm(rate_energy), dtype=np.float64
        )
        ds = np.diff(self._rate_grid)
        interval = 0.5 * (self._rate[1:] + self._rate[:-1]) * ds
        self._cumulative_rate = np.concatenate(
            (np.asarray((0.0,), dtype=np.float64), np.cumsum(interval))
        )
        self.expected_hard_scatter_count = float(self._cumulative_rate[-1])
        if (
            not math.isfinite(self.expected_hard_scatter_count)
            or self.expected_hard_scatter_count < 0.0
        ):
            raise ValueError("integrated hard-scatter rate is not physical")

    @property
    def dimension(self) -> int:
        return 2 * self.modes_per_plane

    @property
    def soft_explained_fraction(self) -> np.ndarray:
        return self._soft_fraction.copy()

    def _sample_hard_position(self, rng: np.random.Generator) -> float:
        total = self.expected_hard_scatter_count
        if total <= 0.0:
            raise ValueError("hard-scatter intensity is zero")
        target = float(rng.random()) * total
        interval_index = int(
            np.clip(
                np.searchsorted(self._cumulative_rate, target, side="right") - 1,
                0,
                self._rate_grid.size - 2,
            )
        )
        local_area = target - float(self._cumulative_rate[interval_index])
        x0 = float(self._rate_grid[interval_index])
        width = float(self._rate_grid[interval_index + 1] - x0)
        rate0 = float(self._rate[interval_index])
        rate1 = float(self._rate[interval_index + 1])
        slope = (rate1 - rate0) / width
        if abs(slope) <= 1.0e-14 * max(rate0, rate1, 1.0):
            offset = local_area / max(rate0, 1.0e-300)
        else:
            discriminant = max(rate0 * rate0 + 2.0 * slope * local_area, 0.0)
            offset = 2.0 * local_area / max(
                rate0 + math.sqrt(discriminant), 1.0e-300
            )
        return float(np.clip(x0 + offset, x0, x0 + width))

    def sample_hard_scatters(
        self,
        rng: np.random.Generator,
        *,
        intensity_fraction: float = 1.0,
    ) -> tuple[PositionedHardScatter, ...]:
        fraction = float(intensity_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("intensity_fraction must lie in [0,1]")
        count = int(rng.poisson(fraction * self.expected_hard_scatter_count))
        scatters = []
        for _ in range(count):
            position = self._sample_hard_position(rng)
            energy = float(
                self.emitter.muon_energy_at_s_array(
                    np.asarray((position,), dtype=np.float64), self.mean_range_mm
                )[0]
            )
            mark = self.scattering.sample_hard_mark(energy, rng)
            scatters.append(PositionedHardScatter.from_mark(position, mark))
        return tuple(sorted(scatters, key=lambda item: item.s_mm))

    def sample(self, rng: np.random.Generator) -> MixedMCSLatent:
        return MixedMCSLatent(
            rng.normal(size=self.dimension),
            self.sample_hard_scatters(rng),
        )

    def pcn_soft(
        self,
        latent: MixedMCSLatent,
        rho: float,
        rng: np.random.Generator,
    ) -> MixedMCSLatent:
        correlation = float(rho)
        if not 0.0 <= correlation < 1.0:
            raise ValueError("rho must lie in [0,1)")
        proposed = (
            correlation * latent.soft_coefficients
            + math.sqrt(1.0 - correlation * correlation)
            * rng.normal(size=self.dimension)
        )
        return MixedMCSLatent(proposed, latent.hard_scatters)

    def refresh_hard_scatters(
        self,
        latent: MixedMCSLatent,
        retention_probability: float,
        rng: np.random.Generator,
    ) -> MixedMCSLatent:
        """Poisson thinning/superposition kernel reversible under the prior."""
        retention = float(retention_probability)
        if not 0.0 <= retention < 1.0:
            raise ValueError("retention_probability must lie in [0,1)")
        retained = tuple(
            mark for mark in latent.hard_scatters if rng.random() < retention
        )
        added = self.sample_hard_scatters(
            rng, intensity_fraction=1.0 - retention
        )
        return MixedMCSLatent(
            latent.soft_coefficients,
            tuple(sorted(retained + added, key=lambda item: item.s_mm)),
        )

    @staticmethod
    def _reflect_coordinate(value: float, low: float, high: float) -> float:
        """Reflect a random-walk coordinate at two finite boundaries."""
        width = float(high - low)
        if not width > 0.0:
            return float(low)
        coordinate = (float(value) - float(low)) % (2.0 * width)
        return float(
            low + (coordinate if coordinate <= width else 2.0 * width - coordinate)
        )

    def perturb_hard_scatter(
        self,
        latent: MixedMCSLatent,
        rng: np.random.Generator,
        *,
        position_step_fraction: float = 0.08,
        log_angle_step: float = 0.35,
        azimuth_step_rad: float = 0.50,
    ) -> tuple[MixedMCSLatent, float]:
        """Locally move one hard mark and return ``(proposal, log qrev/qfwd)``.

        Thinning/superposition is excellent for changing the Poisson count but
        is a very poor posterior kernel for locating one observed hard bend: a
        useful mark must otherwise be redrawn with the right arc coordinate,
        polar angle, and azimuth simultaneously.  This complementary kernel
        keeps the mark channel/count fixed and performs symmetric reflected
        walks in fractional position and log polar angle, plus a wrapped
        azimuth walk.  The sole proposal correction is the Jacobian from
        log-angle back to the marked-process ``dtheta`` reference measure.
        """
        if not latent.hard_scatters:
            return latent, 0.0
        position_scale = float(position_step_fraction)
        angle_scale = float(log_angle_step)
        azimuth_scale = float(azimuth_step_rad)
        if not math.isfinite(position_scale) or position_scale <= 0.0:
            raise ValueError("hard position step fraction must be positive")
        if not math.isfinite(angle_scale) or angle_scale <= 0.0:
            raise ValueError("hard log-angle step must be positive")
        if not math.isfinite(azimuth_scale) or azimuth_scale <= 0.0:
            raise ValueError("hard azimuth step must be positive")

        index = int(rng.integers(len(latent.hard_scatters)))
        current = latent.hard_scatters[index]
        position = self._reflect_coordinate(
            float(current.s_mm)
            + position_scale * self.length_mm * float(rng.normal()),
            0.0,
            self.length_mm,
        )
        theta_low = float(self.scattering.separation_angle_rad)
        theta_high = (
            float(self.scattering.electron_max_scattering_angle_rad)
            if str(current.target) == "electron"
            else math.pi
        )
        log_theta = self._reflect_coordinate(
            math.log(float(current.theta_rad))
            + angle_scale * float(rng.normal()),
            math.log(theta_low),
            math.log(theta_high),
        )
        theta = float(math.exp(log_theta))
        azimuth = float(
            (
                float(current.azimuth_rad)
                + azimuth_scale * float(rng.normal())
            )
            % (2.0 * math.pi)
        )
        moved = PositionedHardScatter(
            s_mm=position,
            theta_rad=theta,
            azimuth_rad=azimuth,
            atomic_number=int(current.atomic_number),
            target=str(current.target),
        )
        marks = list(latent.hard_scatters)
        marks[index] = moved
        proposal = MixedMCSLatent(
            latent.soft_coefficients,
            tuple(marks),
        )
        log_reverse_over_forward = math.log(theta / float(current.theta_rad))
        return proposal, float(log_reverse_over_forward)

    def transport_latent_to(
        self,
        target: "MixedMCSPathPrior",
        latent: MixedMCSLatent,
    ) -> MixedMCSLatent:
        """Transport a non-centred path to a new continuous visible range.

        Standardized soft coordinates are retained.  Hard positions keep their
        fractional arc length, while their physical angle, azimuth, and target
        channel remain unchanged.  The companion prior/Jacobian method supplies
        the exact Metropolis factor for this deterministic map.
        """
        if target.dimension != self.dimension:
            raise ValueError("mixed path dimensions differ across range transport")
        scale = float(target.length_mm / self.length_mm)
        transported = tuple(
            PositionedHardScatter(
                s_mm=float(mark.s_mm) * scale,
                theta_rad=float(mark.theta_rad),
                azimuth_rad=float(mark.azimuth_rad),
                atomic_number=int(mark.atomic_number),
                target=str(mark.target),
            )
            for mark in latent.hard_scatters
        )
        return MixedMCSLatent(latent.soft_coefficients, transported)

    def hard_transport_log_prior_jacobian(
        self,
        target: "MixedMCSPathPrior",
        latent: MixedMCSLatent,
    ) -> float:
        """Return ``log p_target-log p_source+log|ds'/ds|`` for hard marks."""
        transported = self.transport_latent_to(target, latent)
        source_log = -float(self.expected_hard_scatter_count)
        target_log = -float(target.expected_hard_scatter_count)
        if latent.hard_scatters:
            source_s = np.asarray(
                [mark.s_mm for mark in latent.hard_scatters], dtype=np.float64
            )
            target_s = np.asarray(
                [mark.s_mm for mark in transported.hard_scatters], dtype=np.float64
            )
            source_energy = np.asarray(
                self.emitter.muon_energy_at_s_array(source_s, self.mean_range_mm),
                dtype=np.float64,
            )
            target_energy = np.asarray(
                target.emitter.muon_energy_at_s_array(
                    target_s, target.mean_range_mm
                ),
                dtype=np.float64,
            )
            for old_energy, new_energy, old_mark, new_mark in zip(
                source_energy,
                target_energy,
                latent.hard_scatters,
                transported.hard_scatters,
                strict=True,
            ):
                source_log += self.scattering.hard_mark_log_intensity_per_mm_rad2(
                    float(old_energy), old_mark.mark
                )
                target_log += target.scattering.hard_mark_log_intensity_per_mm_rad2(
                    float(new_energy), new_mark.mark
                )
        if not (math.isfinite(source_log) and math.isfinite(target_log)):
            return -math.inf
        log_jacobian = len(latent.hard_scatters) * math.log(
            float(target.length_mm / self.length_mm)
        )
        return float(target_log - source_log + log_jacobian)

    def hard_log_prior_density(self, latent: MixedMCSLatent) -> float:
        """Return the explicit hard-mark Poisson log density.

        The reference measure is the ordered point-process measure
        ``prod(ds dtheta dphi)``.  The Janossy density is therefore
        ``exp(-Lambda) prod(lambda(mark))`` with no extra factorial.  Sampling
        and prior-reversible transitions do not need this value; reporting a
        full posterior-density MAP does.
        """
        if np.asarray(latent.soft_coefficients).shape != (self.dimension,):
            raise ValueError("latent soft dimension does not match this prior")
        value = -float(self.expected_hard_scatter_count)
        for mark in latent.hard_scatters:
            if not 0.0 <= float(mark.s_mm) <= self.length_mm:
                return -math.inf
            energy = float(
                self.emitter.muon_energy_at_s_array(
                    np.asarray((mark.s_mm,), dtype=np.float64),
                    self.mean_range_mm,
                )[0]
            )
            mark_log_density = self.scattering.hard_mark_log_intensity_per_mm_rad2(
                energy, mark.mark
            )
            if not math.isfinite(mark_log_density):
                return -math.inf
            value += float(mark_log_density)
        return float(value)

    def build_path(self, latent: MixedMCSLatent):
        coefficients = np.asarray(latent.soft_coefficients, dtype=np.float64)
        if coefficients.shape != (self.dimension,):
            raise ValueError("latent soft dimension does not match this prior")
        for mark in latent.hard_scatters:
            if not 0.0 <= float(mark.s_mm) <= self.length_mm:
                raise ValueError("hard scatter lies outside the visible path")
            if float(mark.theta_rad) < self.scattering.separation_angle_rad:
                raise ValueError("hard scatter lies below the declared separation angle")

        first_coeff = coefficients[: self.modes_per_plane]
        second_coeff = coefficients[self.modes_per_plane :]
        soft_first = self._soft_slopes @ first_coeff
        soft_second = self._soft_slopes @ second_coeff

        mark_positions = np.asarray(
            [mark.s_mm for mark in latent.hard_scatters], dtype=np.float64
        )
        unique_positions = np.unique(
            np.concatenate((self._soft_grid, mark_positions))
            if mark_positions.size
            else self._soft_grid
        )
        q_first = np.interp(unique_positions, self._soft_grid, soft_first)
        q_second = np.interp(unique_positions, self._soft_grid, soft_second)
        direction, basis_first, basis_second = configured_transverse_basis(self.emitter)
        mark_grid_index = np.searchsorted(unique_positions, mark_positions)
        mark_count = np.bincount(
            mark_grid_index, minlength=unique_positions.size
        ).astype(np.int64, copy=False)
        mark_indptr = np.empty(unique_positions.size + 1, dtype=np.int64)
        mark_indptr[0] = 0
        np.cumsum(mark_count, out=mark_indptr[1:])
        mark_theta_first = np.asarray(
            [mark.theta_rad * math.cos(mark.azimuth_rad) for mark in latent.hard_scatters],
            dtype=np.float64,
        )
        mark_theta_second = np.asarray(
            [mark.theta_rad * math.sin(mark.azimuth_rad) for mark in latent.hard_scatters],
            dtype=np.float64,
        )
        path_s, path_position, path_tangent = _propagate_mixed_path_frames(
            np.ascontiguousarray(unique_positions),
            np.ascontiguousarray(q_first),
            np.ascontiguousarray(q_second),
            np.ascontiguousarray(mark_indptr),
            np.ascontiguousarray(mark_theta_first),
            np.ascontiguousarray(mark_theta_second),
            np.ascontiguousarray(self.emitter.start_coord, dtype=np.float64),
            np.ascontiguousarray(direction, dtype=np.float64),
            np.ascontiguousarray(basis_first, dtype=np.float64),
            np.ascontiguousarray(basis_second, dtype=np.float64),
        )
        energy = np.asarray(
            self.emitter.muon_energy_at_s_array(path_s, self.mean_range_mm),
            dtype=np.float64,
        )
        mass = float(self.emitter.particle_mass)
        gamma = 1.0 + np.maximum(energy, 0.0) / mass
        beta2 = np.maximum(1.0 - 1.0 / np.maximum(gamma * gamma, 1.0), 0.0)
        beta = np.sqrt(beta2)
        refractive_index = float(self.emitter.n)
        cos_cherenkov = np.ones_like(beta)
        above = refractive_index * beta > 1.0
        cos_cherenkov[above] = 1.0 / (refractive_index * beta[above])
        frank_tamm = np.asarray(
            emod._cherenkov_weight_from_energy(energy, mass, refractive_index),
            dtype=np.float64,
        )
        saturated = max(1.0 - 1.0 / (refractive_index**2), 1.0e-30)
        frank_tamm /= saturated
        particle_time = np.asarray(
            emod._wcte_integrated_primary_tof_fast(self.emitter, path_s),
            dtype=np.float64,
        )
        parallel = (path_position - path_position[0]) @ direction
        return {
            "s": np.ascontiguousarray(path_s),
            "position": np.ascontiguousarray(path_position),
            "tangent": np.ascontiguousarray(path_tangent),
            "energy": np.ascontiguousarray(energy),
            "beta": np.ascontiguousarray(beta),
            "cos_cherenkov": np.ascontiguousarray(cos_cherenkov),
            "frank_tamm": np.ascontiguousarray(frank_tamm),
            "particle_time_ns": np.ascontiguousarray(particle_time),
            "parallel_coordinate": np.ascontiguousarray(parallel),
            "basis_explained_fraction": self._soft_fraction,
            "hard_scatter_count": len(latent.hard_scatters),
        }


__all__ = [
    "MixedMCSLatent",
    "MixedMCSPathPrior",
    "PositionedHardScatter",
]
