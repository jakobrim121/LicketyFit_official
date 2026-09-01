"""Fast, detector-safe point-multilateration seeds for cosmic tracks.

The supplied TimeCal multilaterator fits an instantaneous point source with an
unknown emission time.  A finite Cherenkov track is not a point source, so this
module deliberately treats that fit as a *navigation measurement*, never as the
final reconstruction or as a covariance statement about the track.

Two physically useful directions are extracted from the misspecified point fit:

* the least-constrained spatial direction after profiling the unknown emission
  time from the robust Gauss--Newton information; and
* the normalized mean photon direction.  For an azimuthally sampled Cherenkov
  cone, ``mean(k_gamma) = cos(theta_C) * u_track``, which also resolves the sign
  ambiguity of the information eigenvector.

Every proposed line is intersected with the fitter's authoritative active-water
geometry.  A point estimate outside active water is never coordinate-clipped;
it is moved along the detector-centre ray to the first exact boundary and pulled
inside by a tiny relative inset.  Failure of any validation step returns no
seeds so the production driver can fall back to its established cosmic bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from .cosmic_navigation import (
    CausalTimingLineCandidate,
    CausalTimingSeedCandidate,
    robust_causal_timing_score,
    select_range_stratified_seed_indices,
)
from .cosmic_track_fit import (
    ConvexDetectorVolume,
    resolve_boundary_clipped_track,
)
from .track_parameterization import direction_from_mapping, normalize_direction


_LIGHT_SPEED_MM_PER_NS = 299.792458


@dataclass(frozen=True)
class PointMultilaterationResult:
    """Validated navigation result from the robust point-source fit."""

    point_mm: np.ndarray
    emission_time_ns: float
    weak_axis: np.ndarray
    mean_direction: np.ndarray
    safe_reference_mm: np.ndarray
    spatial_information_eigenvalues: np.ndarray
    spatial_information_condition: float
    photon_resultant_length: float
    chi2: float
    n_dof: int
    n_hits: int
    nfev: int
    wall_s: float
    point_inside_active_water: bool

    def metadata(self) -> dict[str, object]:
        return {
            "point_mm": np.asarray(self.point_mm, dtype=np.float64).tolist(),
            "emission_time_ns": float(self.emission_time_ns),
            "weak_axis": np.asarray(self.weak_axis, dtype=np.float64).tolist(),
            "mean_direction": np.asarray(
                self.mean_direction, dtype=np.float64
            ).tolist(),
            "safe_reference_mm": np.asarray(
                self.safe_reference_mm, dtype=np.float64
            ).tolist(),
            "spatial_information_eigenvalues": np.asarray(
                self.spatial_information_eigenvalues, dtype=np.float64
            ).tolist(),
            "spatial_information_condition": float(
                self.spatial_information_condition
            ),
            "photon_resultant_length": float(self.photon_resultant_length),
            "chi2": float(self.chi2),
            "n_dof": int(self.n_dof),
            "chi2_per_dof": (
                float(self.chi2 / self.n_dof) if self.n_dof > 0 else None
            ),
            "n_hits": int(self.n_hits),
            "nfev": int(self.nfev),
            "wall_s": float(self.wall_s),
            "point_inside_active_water": bool(
                self.point_inside_active_water
            ),
        }


@dataclass(frozen=True)
class MultilaterationSeedBankCandidate:
    """One immutable bank seed compatible with a multilaterated track line."""

    seed_index: int
    track_start_hypothesis: str
    line_rank: int
    angular_separation_deg: float
    transverse_line_distance_mm: float
    geometric_score: float

    def metadata(self) -> dict[str, object]:
        return {
            "seed_index": int(self.seed_index),
            "track_start_hypothesis": str(self.track_start_hypothesis),
            "line_rank": int(self.line_rank),
            "angular_separation_deg": float(self.angular_separation_deg),
            "transverse_line_distance_mm": float(
                self.transverse_line_distance_mm
            ),
            "geometric_score": float(self.geometric_score),
        }


def select_guided_seed_bank_indices(
    candidates: Sequence[MultilaterationSeedBankCandidate],
    scores: np.ndarray,
    seeds: Sequence[Mapping[str, object]],
    strata: Sequence[int],
    *,
    start_hypothesis: str,
    range_representatives: int = 2,
    proxy_geometry_representatives: int = 2,
    maximum_proxy_geometry_gap_nll: float = 125.0,
    maximum_quantization_tie_representatives: int = 4,
    proxy_quantization_uncertainty_nll: float = 0.0,
) -> list[int]:
    """Select a compact, range-safe bank subset without false proxy tie breaks.

    The cached proxy stores quantized log charge shapes.  For observed charges
    ``q_g`` and log-code spacing ``delta``, the difference of two proxy scores
    has a worst-case quantization uncertainty ``delta * sum(q_g)``.  Seeds
    inside that uncertainty cannot be ordered by the proxy table itself.  A
    few geometrically distinct members of that unresolved set are therefore
    retained for the later exact-likelihood screen, in addition to the normal
    range-stratified representatives.

    This rule uses neither event truth nor an energy label.  The tie candidates
    receive only the driver's existing single-point exact screen; it does not
    imply that every tied seed receives a full optimizer continuation.
    """

    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    stratum_array = np.asarray(strata).reshape(-1)
    if score_array.size != len(seeds) or stratum_array.size != len(seeds):
        raise ValueError("scores, seeds, and strata must be aligned")
    uncertainty = float(proxy_quantization_uncertainty_nll)
    if not math.isfinite(uncertainty) or uncertainty < 0.0:
        raise ValueError(
            "proxy_quantization_uncertainty_nll must be finite and nonnegative"
        )
    geometry_gap = float(maximum_proxy_geometry_gap_nll)
    if not math.isfinite(geometry_gap) or geometry_gap < 0.0:
        raise ValueError(
            "maximum_proxy_geometry_gap_nll must be finite and nonnegative"
        )

    hypothesis = str(start_hypothesis)
    eligible: list[int] = []
    seen_indices: set[int] = set()
    for row in candidates:
        if str(row.track_start_hypothesis) != hypothesis:
            continue
        index = int(row.seed_index)
        if index < 0 or index >= len(seeds):
            raise ValueError("guided seed-bank candidate index is out of range")
        if index not in seen_indices:
            seen_indices.add(index)
            eligible.append(index)
    if not eligible:
        return []

    selected = select_range_stratified_seed_indices(
        score_array,
        seeds,
        eligible,
        stratum_array,
        n_keep=min(max(1, int(range_representatives)), len(eligible)),
        raw_top=1,
        geometry_representatives_per_stratum=1,
    )
    selected_set = set(int(index) for index in selected)

    finite = [
        int(index) for index in eligible
        if math.isfinite(float(score_array[int(index)]))
    ]
    if not finite:
        return [int(index) for index in selected]
    best_score = min(float(score_array[index]) for index in finite)
    score_order = sorted(
        finite,
        key=lambda index: (float(score_array[index]), int(index)),
    )
    unresolved = [
        index for index in score_order
        if float(score_array[index]) <= best_score + uncertainty + 1.0e-12
    ]

    def geometry_key(index: int) -> tuple[object, ...]:
        seed = seeds[int(index)]
        direction = direction_from_mapping(seed)
        return (
            hypothesis,
            round(float(seed["x0"]), 3),
            round(float(seed["y0"]), 3),
            round(float(seed["z0"]), 3),
            round(float(direction[0]), 6),
            round(float(direction[1]), 6),
            round(float(direction[2]), 6),
        )

    # A point-source fit can be transversely biased by the finite emitting
    # segment.  Preserve one additional proxy-ranked bank geometry before
    # relying on the exact screen; repeated range anchors at the same vertex
    # and direction do not consume this geometry budget.
    maximum_geometries = max(1, int(proxy_geometry_representatives))
    proxy_geometries: set[tuple[object, ...]] = set()
    for index in score_order:
        if float(score_array[index]) > best_score + geometry_gap + 1.0e-12:
            break
        key = geometry_key(index)
        if key in proxy_geometries:
            continue
        proxy_geometries.add(key)
        if index not in selected_set:
            selected_set.add(index)
            selected.append(index)
        if len(proxy_geometries) >= maximum_geometries:
            break

    maximum_ties = max(1, int(maximum_quantization_tie_representatives))
    retained_geometries: set[tuple[object, ...]] = set()
    retained_ties = 0
    for index in unresolved:
        key = geometry_key(index)
        if key in retained_geometries:
            continue
        retained_geometries.add(key)
        retained_ties += 1
        if index not in selected_set:
            selected_set.add(index)
            selected.append(index)
        if retained_ties >= maximum_ties:
            break
    return [int(index) for index in selected]


def rank_multilateration_seed_bank(
    seed_matrix: np.ndarray,
    seed_start_hypotheses: Sequence[str],
    point_fit: PointMultilaterationResult,
    *,
    maximum_angle_deg: float = 20.0,
    maximum_transverse_distance_mm: float = 450.0,
    maximum_per_hypothesis: int = 128,
) -> list[MultilaterationSeedBankCandidate]:
    """Return a detector-scale geometric neighbourhood of the inferred line.

    The first seven columns of ``seed_matrix`` are ``(x,y,z,cx,cy,cz,range)``.
    Range is intentionally absent from this geometric gate: after direction and
    transverse line compatibility are established, the event charge proxy must
    compare multiple energy/range strata without an energy prior.
    """

    matrix = np.asarray(seed_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 6:
        raise ValueError("seed_matrix must have at least six columns")
    hypotheses = np.asarray(
        [str(value) for value in seed_start_hypotheses], dtype=object
    )
    if hypotheses.size != matrix.shape[0]:
        raise ValueError("seed hypotheses are not aligned with seed_matrix")
    maximum_angle = float(maximum_angle_deg)
    maximum_distance = float(maximum_transverse_distance_mm)
    if (
        not math.isfinite(maximum_angle)
        or maximum_angle <= 0.0
        or maximum_angle > 180.0
    ):
        raise ValueError("maximum_angle_deg must lie in (0,180]")
    if not math.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise ValueError(
            "maximum_transverse_distance_mm must be finite and positive"
        )

    references = matrix[:, :3]
    directions = matrix[:, 3:6]
    direction_norm = np.linalg.norm(directions, axis=1)
    finite = (
        np.all(np.isfinite(references), axis=1)
        & np.all(np.isfinite(directions), axis=1)
        & np.isfinite(direction_norm)
        & (direction_norm > 1.0e-12)
    )
    unit_directions = np.zeros_like(directions)
    unit_directions[finite] = (
        directions[finite] / direction_norm[finite, None]
    )
    lines = _unique_directions((
        point_fit.mean_direction,
        point_fit.weak_axis,
    ))
    if not lines:
        raise ValueError("point fit supplies no valid track direction")
    reference = np.asarray(point_fit.safe_reference_mm, dtype=np.float64)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("point fit safe reference is invalid")

    n_seed = int(matrix.shape[0])
    best_score = np.full(n_seed, np.inf, dtype=np.float64)
    best_angle = np.full(n_seed, np.inf, dtype=np.float64)
    best_distance = np.full(n_seed, np.inf, dtype=np.float64)
    best_line = np.full(n_seed, -1, dtype=np.int32)
    angle_scale = math.radians(maximum_angle) / 2.0
    distance_scale = maximum_distance / 2.0
    displacement = references - reference[None, :]
    for line_rank, line in enumerate(lines):
        cosine = np.clip(unit_directions @ line, -1.0, 1.0)
        angle = np.arccos(cosine)
        longitudinal = displacement @ line
        transverse = displacement - longitudinal[:, None] * line[None, :]
        distance = np.linalg.norm(transverse, axis=1)
        score = (
            (angle / max(angle_scale, 1.0e-12)) ** 2
            + (distance / max(distance_scale, 1.0e-12)) ** 2
        )
        valid = (
            finite
            & (angle <= math.radians(maximum_angle))
            & (distance <= maximum_distance)
            & np.isfinite(score)
        )
        improve = valid & (score < best_score)
        best_score[improve] = score[improve]
        best_angle[improve] = angle[improve]
        best_distance[improve] = distance[improve]
        best_line[improve] = int(line_rank)

    retained: list[MultilaterationSeedBankCandidate] = []
    limit = max(1, int(maximum_per_hypothesis))
    for hypothesis in ("internal_start", "boundary_entry"):
        indices = np.flatnonzero(
            (hypotheses == hypothesis) & np.isfinite(best_score)
        )
        if not indices.size:
            continue
        order = indices[np.argsort(best_score[indices], kind="mergesort")]
        for raw_index in order[:limit]:
            index = int(raw_index)
            retained.append(MultilaterationSeedBankCandidate(
                seed_index=index,
                track_start_hypothesis=hypothesis,
                line_rank=int(best_line[index]),
                angular_separation_deg=float(math.degrees(best_angle[index])),
                transverse_line_distance_mm=float(best_distance[index]),
                geometric_score=float(best_score[index]),
            ))
    retained.sort(key=lambda row: (
        str(row.track_start_hypothesis),
        float(row.geometric_score),
        int(row.seed_index),
    ))
    return retained


def _validated_event_arrays(
    pmt_positions_mm: np.ndarray,
    observed_charge_pe: np.ndarray,
    observed_time_ns: np.ndarray,
    timing_sigma_ns: float | Sequence[float] | np.ndarray,
    *,
    minimum_hits: int,
    maximum_hits: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(pmt_positions_mm, dtype=np.float64)
    charge = np.asarray(observed_charge_pe, dtype=np.float64).reshape(-1)
    times = np.asarray(observed_time_ns, dtype=np.float64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("pmt_positions_mm must have shape (N,3)")
    if positions.shape[0] != charge.size or times.size != charge.size:
        raise ValueError("PMT position, charge, and time arrays are not aligned")

    sigma_raw = np.asarray(timing_sigma_ns, dtype=np.float64)
    if sigma_raw.ndim == 0:
        sigma = np.full(charge.size, float(sigma_raw), dtype=np.float64)
    else:
        sigma = sigma_raw.reshape(-1)
        if sigma.size != charge.size:
            raise ValueError("timing_sigma_ns is not event-aligned")
    good = (
        np.isfinite(charge)
        & (charge > 0.0)
        & np.isfinite(times)
        & np.isfinite(sigma)
        & (sigma > 0.0)
        & np.all(np.isfinite(positions), axis=1)
    )
    indices = np.flatnonzero(good)
    if indices.size < max(4, int(minimum_hits)):
        raise ValueError("too few finite timed PMTs for multilateration")

    if maximum_hits is not None and int(maximum_hits) > 0:
        limit = max(int(minimum_hits), int(maximum_hits))
        if indices.size > limit:
            # Retain an equal-status mixture of early and geometrically bright
            # channels without duplicating a PMT or treating charge as repeated
            # independent timestamps.
            n_early = max(1, limit // 2)
            early = indices[
                np.argsort(times[indices], kind="mergesort")[:n_early]
            ]
            bright = indices[
                np.argsort(-charge[indices], kind="mergesort")[: limit - n_early]
            ]
            selected: list[int] = []
            seen: set[int] = set()
            for raw in np.concatenate((early, bright, indices)):
                index = int(raw)
                if index not in seen:
                    seen.add(index)
                    selected.append(index)
                if len(selected) >= limit:
                    break
            indices = np.asarray(selected, dtype=np.int64)

    return (
        np.ascontiguousarray(positions[indices], dtype=np.float64),
        np.ascontiguousarray(charge[indices], dtype=np.float64),
        np.ascontiguousarray(times[indices], dtype=np.float64),
        np.ascontiguousarray(sigma[indices], dtype=np.float64),
    )


def project_point_to_active_water(
    detector: ConvexDetectorVolume,
    point_mm: Sequence[float],
    *,
    relative_inset: float = 1.0e-3,
) -> np.ndarray | None:
    """Return an exact-geometry-safe reference without coordinate clipping."""

    point = np.asarray(point_mm, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        return None
    if detector.contains(point, tolerance_mm=1.0e-5):
        return np.ascontiguousarray(point, dtype=np.float64)
    centre = np.asarray(detector.reference_center, dtype=np.float64)
    if centre.shape != (3,) or not detector.contains(
        centre, tolerance_mm=1.0e-5
    ):
        return None
    displacement = point - centre
    if float(np.linalg.norm(displacement)) <= 0.0:
        return np.ascontiguousarray(centre, dtype=np.float64)
    try:
        boundary = detector.project_step(centre, displacement)
    except Exception:
        return None
    inset = min(max(float(relative_inset), 1.0e-9), 0.1)
    reference = centre + (1.0 - inset) * (boundary - centre)
    if not detector.contains(reference, tolerance_mm=1.0e-5):
        return None
    return np.ascontiguousarray(reference, dtype=np.float64)


def fit_point_multilateration(
    detector: ConvexDetectorVolume,
    pmt_positions_mm: np.ndarray,
    observed_charge_pe: np.ndarray,
    observed_time_ns: np.ndarray,
    *,
    timing_sigma_ns: float | Sequence[float] | np.ndarray = 1.0,
    group_refractive_index: float = 1.373,
    minimum_hits: int = 8,
    maximum_hits: int | None = None,
    huber_scale_sigma: float = 1.0,
    maximum_nfev: int = 64,
    maximum_information_condition: float = 1.0e6,
) -> PointMultilaterationResult:
    """Fit ``(x,y,z,t)`` with the supplied point-source timing equations.

    The bounds are a detector-scaled safety box, not a replacement for exact
    active-water containment.  Exact containment/projection is applied only
    after the robust unconstrained navigation fit so outside estimates remain
    visible in diagnostics.
    """

    positions, charge, times, sigma = _validated_event_arrays(
        pmt_positions_mm,
        observed_charge_pe,
        observed_time_ns,
        timing_sigma_ns,
        minimum_hits=int(minimum_hits),
        maximum_hits=maximum_hits,
    )
    n_group = float(group_refractive_index)
    if not math.isfinite(n_group) or n_group <= 1.0:
        raise ValueError("group_refractive_index must be finite and greater than one")
    velocity = _LIGHT_SPEED_MM_PER_NS / n_group
    centre = np.asarray(detector.reference_center, dtype=np.float64)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError("detector reference centre is unavailable")
    axis_lo = np.asarray(detector.axis_lo, dtype=np.float64)
    axis_hi = np.asarray(detector.axis_hi, dtype=np.float64)
    span = axis_hi - axis_lo
    diagonal = float(np.linalg.norm(span))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("detector bounds are invalid")

    distance0 = np.linalg.norm(positions - centre[None, :], axis=1)
    emission0 = float(np.median(times - distance0 / velocity))
    spatial_padding = diagonal
    time_padding = max(100.0, 2.0 * diagonal / velocity)
    lower = np.concatenate((axis_lo - spatial_padding, [emission0 - time_padding]))
    upper = np.concatenate((axis_hi + spatial_padding, [emission0 + time_padding]))
    initial = np.concatenate((centre, [emission0]))

    def residual(parameters: np.ndarray) -> np.ndarray:
        point = parameters[:3]
        emission_time = float(parameters[3])
        displacement = positions - point[None, :]
        distance = np.linalg.norm(displacement, axis=1)
        return (times - emission_time - distance / velocity) / sigma

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        point = parameters[:3]
        displacement = positions - point[None, :]
        distance = np.linalg.norm(displacement, axis=1)
        if np.any(distance <= 1.0e-9):
            # Returning finite derivatives is required by least_squares; a PMT
            # centre cannot be a physical light-source point in active water.
            distance = np.maximum(distance, 1.0e-9)
        spatial = displacement / distance[:, None] / velocity / sigma[:, None]
        time_column = -1.0 / sigma
        return np.ascontiguousarray(
            np.column_stack((spatial, time_column)), dtype=np.float64
        )

    wall0 = time.perf_counter()
    result = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=(lower, upper),
        method="trf",
        loss="huber",
        f_scale=max(float(huber_scale_sigma), 1.0e-6),
        max_nfev=max(8, int(maximum_nfev)),
    )
    wall_s = float(time.perf_counter() - wall0)
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            "point multilateration did not converge: " + str(result.message)
        )

    robust_jacobian = np.asarray(result.jac, dtype=np.float64)
    if robust_jacobian.shape != (positions.shape[0], 4):
        raise RuntimeError("point multilateration returned an invalid Jacobian")
    information = robust_jacobian.T @ robust_jacobian
    time_information = float(information[3, 3])
    if not math.isfinite(time_information) or time_information <= 0.0:
        raise RuntimeError("emission-time information is singular")
    profiled_spatial = (
        information[:3, :3]
        - np.outer(information[:3, 3], information[3, :3]) / time_information
    )
    eigenvalues, eigenvectors = np.linalg.eigh(profiled_spatial)
    if (
        not np.all(np.isfinite(eigenvalues))
        or float(eigenvalues[0]) <= 0.0
    ):
        raise RuntimeError("profiled spatial information is not positive definite")
    condition = float(eigenvalues[-1] / eigenvalues[0])
    if not math.isfinite(condition) or condition > float(
        maximum_information_condition
    ):
        raise RuntimeError("profiled spatial information is ill-conditioned")

    point = np.asarray(result.x[:3], dtype=np.float64)
    photon_vectors = positions - point[None, :]
    photon_norm = np.linalg.norm(photon_vectors, axis=1)
    if np.any(photon_norm <= 1.0e-9):
        raise RuntimeError("multilateration point coincides with a PMT")
    photon_directions = photon_vectors / photon_norm[:, None]
    photon_mean = np.mean(photon_directions, axis=0)
    resultant = float(np.linalg.norm(photon_mean))
    if not math.isfinite(resultant) or resultant <= 1.0e-8:
        raise RuntimeError("mean photon direction is undefined")
    mean_direction = normalize_direction(photon_mean)
    weak_axis = normalize_direction(eigenvectors[:, 0])
    if float(np.dot(weak_axis, mean_direction)) < 0.0:
        weak_axis = -weak_axis

    safe_reference = project_point_to_active_water(detector, point)
    if safe_reference is None:
        raise RuntimeError("multilateration point cannot be projected into active water")
    pulls = residual(result.x)
    chi2 = float(np.dot(pulls, pulls))
    return PointMultilaterationResult(
        point_mm=np.ascontiguousarray(point, dtype=np.float64),
        emission_time_ns=float(result.x[3]),
        weak_axis=np.ascontiguousarray(weak_axis, dtype=np.float64),
        mean_direction=np.ascontiguousarray(mean_direction, dtype=np.float64),
        safe_reference_mm=np.ascontiguousarray(safe_reference, dtype=np.float64),
        spatial_information_eigenvalues=np.ascontiguousarray(
            eigenvalues, dtype=np.float64
        ),
        spatial_information_condition=condition,
        photon_resultant_length=resultant,
        chi2=chi2,
        n_dof=int(positions.shape[0] - 4),
        n_hits=int(positions.shape[0]),
        nfev=int(result.nfev),
        wall_s=wall_s,
        point_inside_active_water=bool(detector.contains(point)),
    )


def _unique_directions(
    directions: Sequence[Sequence[float]],
    *,
    minimum_separation_deg: float = 1.0,
) -> list[np.ndarray]:
    selected: list[np.ndarray] = []
    cosine_limit = math.cos(math.radians(max(float(minimum_separation_deg), 0.0)))
    for raw in directions:
        direction = normalize_direction(raw)
        if any(float(np.dot(direction, old)) >= cosine_limit for old in selected):
            continue
        selected.append(direction)
    return selected


def build_multilateration_timing_seed_guard(
    detector: ConvexDetectorVolume,
    pmt_positions_mm: np.ndarray,
    observed_charge_pe: np.ndarray,
    observed_time_ns: np.ndarray,
    *,
    timing_sigma_ns: float | Sequence[float] | np.ndarray = 1.0,
    group_refractive_index: float = 1.373,
    phase_refractive_index: float = 1.344,
    maximum_full_range_mm: float,
    boundary_inset_mm: float = 0.5,
    minimum_hits: int = 8,
    maximum_hits: int | None = None,
    maximum_nfev: int = 64,
    include_reverse_directions: bool = False,
    reverse_if_resultant_below: float = 0.15,
    maximum_per_line_hypothesis: int = 4,
    maximum_pool_seeds: int = 16,
    minimum_segment_mm: float = 120.0,
) -> tuple[
    list[CausalTimingSeedCandidate],
    list[CausalTimingLineCandidate],
    PointMultilaterationResult,
]:
    """Build a compact, energy-blind finite-track pool from point timing.

    Segment timing chooses only a small set of longitudinal/range realizations.
    The production driver's direct-primary charge proxy and exact likelihood
    remain responsible for choosing among them.
    """

    point_fit = fit_point_multilateration(
        detector,
        pmt_positions_mm,
        observed_charge_pe,
        observed_time_ns,
        timing_sigma_ns=timing_sigma_ns,
        group_refractive_index=float(group_refractive_index),
        minimum_hits=int(minimum_hits),
        maximum_hits=maximum_hits,
        maximum_nfev=int(maximum_nfev),
    )
    include_reverse = bool(include_reverse_directions) or (
        float(point_fit.photon_resultant_length)
        < float(reverse_if_resultant_below)
    )
    raw_directions: list[np.ndarray] = [
        point_fit.mean_direction,
        point_fit.weak_axis,
    ]
    if include_reverse:
        raw_directions.extend(
            [-point_fit.mean_direction, -point_fit.weak_axis]
        )
    directions = _unique_directions(raw_directions)

    lines: list[CausalTimingLineCandidate] = []
    for direction in directions:
        chord = resolve_boundary_clipped_track(
            detector,
            point_fit.safe_reference_mm,
            direction,
            inset_mm=float(boundary_inset_mm),
        )
        if chord is None or chord.visible_length_mm <= float(minimum_segment_mm):
            continue
        score = robust_causal_timing_score(
            pmt_positions_mm,
            observed_charge_pe,
            observed_time_ns,
            chord.entry,
            direction,
            chord.visible_length_mm,
            refractive_index=float(phase_refractive_index),
        )
        if score is None:
            continue
        lines.append(CausalTimingLineCandidate(
            score=float(score.score),
            t0_ns=float(score.t0_ns),
            reference_mm=np.ascontiguousarray(
                point_fit.safe_reference_mm, dtype=np.float64
            ),
            direction=np.ascontiguousarray(direction, dtype=np.float64),
            entry_mm=np.ascontiguousarray(chord.entry, dtype=np.float64),
            exit_mm=np.ascontiguousarray(chord.exit, dtype=np.float64),
            chord_length_mm=float(chord.visible_length_mm),
            n_hits=int(score.n_hits),
            n_inliers=int(score.n_inliers),
        ))
    lines.sort(key=lambda line: (float(line.score), float(line.t0_ns)))
    if not lines:
        raise RuntimeError("multilateration directions do not define a timed detector chord")

    # Fractions are detector-scale rather than energy-scale.  The two close
    # upstream anchors are important for particles created just inside a water
    # boundary, while the progressively wider spacing retains coverage for
    # genuinely internal starts without assuming a beam direction or energy.
    start_fractions = (0.0, 0.02, 0.06, 0.14, 0.28, 0.45, 0.65, 0.82)
    remaining_end_fractions = (0.25, 0.45, 0.70, 1.0)
    rows: list[CausalTimingSeedCandidate] = []
    maximum_range = float(maximum_full_range_mm)
    for line_rank, line in enumerate(lines):
        chord_length = float(line.chord_length_mm)
        local_rows: list[CausalTimingSeedCandidate] = []
        for start_fraction in start_fractions:
            for remaining_fraction in remaining_end_fractions:
                end_fraction = float(
                    start_fraction
                    + remaining_fraction * (1.0 - start_fraction)
                )
                segment_length = (end_fraction - start_fraction) * chord_length
                if segment_length < float(minimum_segment_mm):
                    continue
                segment_start = (
                    line.entry_mm
                    + start_fraction * chord_length * line.direction
                )
                score = robust_causal_timing_score(
                    pmt_positions_mm,
                    observed_charge_pe,
                    observed_time_ns,
                    segment_start,
                    line.direction,
                    segment_length,
                    refractive_index=float(phase_refractive_index),
                )
                if score is None:
                    continue
                starts_at_boundary = bool(start_fraction <= 1.0e-9)
                exits_detector = bool(end_fraction >= 1.0 - 1.0e-9)
                hypothesis = (
                    "boundary_entry" if starts_at_boundary else "internal_start"
                )
                topology = (
                    "boundary_entry_boundary_exit"
                    if starts_at_boundary and exits_detector
                    else "boundary_entry_internal_stop"
                    if starts_at_boundary
                    else "internal_start_boundary_exit"
                    if exits_detector
                    else "internal_start_internal_stop"
                )
                full_range = (
                    min(
                        maximum_range,
                        max(segment_length + 250.0, 1.28 * segment_length),
                    )
                    if exits_detector
                    else segment_length
                )
                if not math.isfinite(full_range) or full_range <= 0.0:
                    continue
                reference = (
                    0.5 * (line.entry_mm + line.exit_mm)
                    if starts_at_boundary
                    else segment_start
                )
                seed = {
                    "x0": float(reference[0]),
                    "y0": float(reference[1]),
                    "z0": float(reference[2]),
                    "dir_x": float(line.direction[0]),
                    "dir_y": float(line.direction[1]),
                    "dir_z": float(line.direction[2]),
                    "cx": float(line.direction[0]),
                    "cy": float(line.direction[1]),
                    "cz": float(line.direction[2]),
                    "length": float(full_range),
                    "t0": float(score.t0_ns),
                    "track_start_hypothesis": str(hypothesis),
                    "seed_family": "multilateration_timing",
                    "multilateration_timing_score": float(score.score),
                    "multilateration_line_score": float(line.score),
                    "multilateration_line_rank": int(line_rank),
                    "multilateration_start_fraction": float(start_fraction),
                    "multilateration_end_fraction": float(end_fraction),
                    "multilateration_topology": str(topology),
                }
                local_rows.append(CausalTimingSeedCandidate(
                    score=float(score.score),
                    line_score=float(line.score),
                    t0_ns=float(score.t0_ns),
                    start_fraction=float(start_fraction),
                    end_fraction=float(end_fraction),
                    track_start_hypothesis=str(hypothesis),
                    topology=str(topology),
                    line_rank=int(line_rank),
                    seed=seed,
                ))
        for hypothesis in ("internal_start", "boundary_entry"):
            candidates = sorted(
                (
                    row
                    for row in local_rows
                    if row.track_start_hypothesis == hypothesis
                ),
                key=lambda row: (
                    float(row.score),
                    float(row.line_score),
                    float(row.start_fraction),
                    float(row.end_fraction),
                ),
            )
            rows.extend(candidates[: max(1, int(maximum_per_line_hypothesis))])

    rows.sort(key=lambda row: (
        float(row.score),
        float(row.line_score),
        int(row.line_rank),
        str(row.track_start_hypothesis),
    ))
    return rows[: max(1, int(maximum_pool_seeds))], lines, point_fit


__all__ = [
    "MultilaterationSeedBankCandidate",
    "PointMultilaterationResult",
    "build_multilateration_timing_seed_guard",
    "fit_point_multilateration",
    "project_point_to_active_water",
    "rank_multilateration_seed_bank",
]
