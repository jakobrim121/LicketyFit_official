"""Production helpers for explicit mPMT boundary-interface hypotheses.

This module contains only geometry- and observation-driven navigation.  It does
not use WCSim truth.  A normal cosmic fit is run first; nearby mPMT entry/exit
subclasses are then nominated from the fitted line, module charge residuals and
the placed detector geometry.  Each subclass remains a separate discrete model,
so an optimizer cannot escape the local-light term by moving infinitesimally to
the clean side of a dome.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .cosmic_track_fit import ConvexDetectorVolume, RangeClippedTrack
from .mpmt_boundary import ModuleGeometry
from .track_parameterization import TangentDirectionChart, normalize_direction


@dataclass(frozen=True)
class MPMTBoundaryFitConfig:
    enabled: bool = True
    max_slots_per_interface: int = 2
    proximity_margin_mm: float = 140.0
    residual_proximity_margin_mm: float = 260.0
    minimum_positive_deviance: float = 8.0
    minimum_positive_residual_pe: float = 4.0
    maximum_seed_pmts_per_slot: int = 2
    seed_transverse_offsets_mm: tuple[float, ...] = (0.0, 22.5)
    timing_policy: str = "mask_module"
    max_total_hardware_fraction: float = 0.45
    minimum_penalized_gain_nll: float = 0.0


@dataclass(frozen=True)
class ModuleResidualScore:
    slot: int
    observed_charge: float
    expected_charge: float
    positive_residual_pe: float
    positive_deviance_nll: float
    maximum_pmt_positive_deviance_nll: float
    brightest_pmt_index: int
    brightest_residual_pmt_index: int
    earliest_time_ns: float

    def metadata(self) -> dict[str, object]:
        return {
            "slot": int(self.slot),
            "observed_charge": float(self.observed_charge),
            "expected_charge": float(self.expected_charge),
            "positive_residual_pe": float(self.positive_residual_pe),
            "positive_deviance_nll": float(self.positive_deviance_nll),
            "maximum_pmt_positive_deviance_nll": float(
                self.maximum_pmt_positive_deviance_nll
            ),
            "brightest_pmt_index": int(self.brightest_pmt_index),
            "brightest_residual_pmt_index": int(
                self.brightest_residual_pmt_index
            ),
            "earliest_time_ns": float(self.earliest_time_ns),
        }


@dataclass(frozen=True)
class BoundarySlotNomination:
    interface: str
    slot: int
    direct_boundary_hit: bool
    boundary_surface_clearance_mm: float
    residual: ModuleResidualScore
    ranking_score: float
    reasons: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        return {
            "interface": str(self.interface),
            "slot": int(self.slot),
            "direct_boundary_hit": bool(self.direct_boundary_hit),
            "boundary_surface_clearance_mm": float(
                self.boundary_surface_clearance_mm
            ),
            "ranking_score": float(self.ranking_score),
            "reasons": list(self.reasons),
            "residual": self.residual.metadata(),
        }


def module_residual_scores(
    observed_pes: Sequence[float],
    expected_pes: Sequence[float],
    observed_times_ns: Sequence[float],
    pmt_slots: Sequence[int],
) -> dict[int, ModuleResidualScore]:
    """Aggregate one-sided Poisson underprediction evidence by module."""
    q = np.asarray(observed_pes, dtype=np.float64)
    lam = np.maximum(np.asarray(expected_pes, dtype=np.float64), 1.0e-12)
    ts = np.asarray(observed_times_ns, dtype=np.float64)
    slots = np.asarray(pmt_slots, dtype=np.int64)
    if not (q.shape == lam.shape == ts.shape == slots.shape):
        raise ValueError("PMT arrays must have identical shape")
    positive = q > lam
    dev = np.zeros_like(q)
    dev[positive] = (
        q[positive] * np.log(np.maximum(q[positive], 1.0e-300) / lam[positive])
        - (q[positive] - lam[positive])
    )
    residual = np.maximum(q - lam, 0.0)
    out: dict[int, ModuleResidualScore] = {}
    for slot in np.unique(slots):
        idx = np.flatnonzero(slots == slot)
        if idx.size == 0:
            continue
        local_q = q[idx]
        local_residual = residual[idx]
        local_dev = dev[idx]
        finite_t = ts[idx][np.isfinite(ts[idx])]
        out[int(slot)] = ModuleResidualScore(
            slot=int(slot),
            observed_charge=float(np.sum(local_q)),
            expected_charge=float(np.sum(lam[idx])),
            positive_residual_pe=float(np.sum(local_residual)),
            positive_deviance_nll=float(np.sum(local_dev)),
            maximum_pmt_positive_deviance_nll=float(np.max(local_dev, initial=0.0)),
            brightest_pmt_index=int(idx[int(np.argmax(local_q))]),
            brightest_residual_pmt_index=int(idx[int(np.argmax(local_residual))]),
            earliest_time_ns=(
                float(np.min(finite_t)) if finite_t.size else math.inf
            ),
        )
    return out


def _boundary_point_for_interface(
    track: RangeClippedTrack, interface: str
) -> np.ndarray | None:
    label = str(interface).strip().lower()
    if label == "entry":
        if not track.starts_at_boundary:
            return None
        return np.asarray(track.start, dtype=np.float64)
    if label == "exit":
        if not track.exits_detector:
            return None
        return np.asarray(track.boundary_exit, dtype=np.float64)
    raise ValueError("interface must be entry or exit")


def exclusion_surface_clearance_mm(
    detector: ConvexDetectorVolume,
    point_mm: Sequence[float],
    slot: int,
) -> float:
    """Unsigned distance from a point to the finite dome-exclusion surface.

    Near the spherical cap this is the radial sphere clearance.  Near the cap
    plane it is the Euclidean distance to the finite disk.  The minimum of both
    pieces is continuous at the seam and is sufficient for hypothesis
    nomination; exact intersections remain authoritative in the objective.
    """
    slots = np.asarray(detector.exclusion_slots, dtype=np.int64)
    match = np.flatnonzero(slots == int(slot))
    if match.size == 0:
        return math.inf
    i = int(match[0])
    centre = np.asarray(detector.exclusion_centres_mm[i], dtype=np.float64)
    axis = normalize_direction(detector.exclusion_axes[i])
    point = np.asarray(point_mm, dtype=np.float64)
    q = point - centre
    radius = float(detector.exclusion_radius_mm)
    cut = float(detector.exclusion_cap_cut_mm)
    axial = float(np.dot(q, axis))
    radial_vec = q - axial * axis
    radial = float(np.linalg.norm(radial_vec))

    sphere_clearance = math.inf
    norm_q = float(np.linalg.norm(q))
    if axial >= cut:
        sphere_clearance = abs(norm_q - radius)

    disk_radius = math.sqrt(max(radius * radius - cut * cut, 0.0))
    plane_gap = abs(axial - cut)
    radial_gap = max(0.0, radial - disk_radius)
    plane_clearance = math.hypot(plane_gap, radial_gap)
    return float(min(sphere_clearance, plane_clearance))


def nominate_boundary_slots(
    *,
    track: RangeClippedTrack,
    detector: ConvexDetectorVolume,
    residual_scores: Mapping[int, ModuleResidualScore],
    interface: str,
    config: MPMTBoundaryFitConfig,
) -> list[BoundarySlotNomination]:
    """Nominate a small set of explicit mPMT interfaces for one boundary."""
    point = _boundary_point_for_interface(track, interface)
    if point is None or detector.exclusion_slots.size == 0:
        return []
    label = str(interface).strip().lower()
    direct_hit = (
        track.start_boundary_hit if label == "entry" else track.exit_boundary_hit
    )
    direct_slot = (
        int(direct_hit.slot)
        if direct_hit is not None
        and direct_hit.surface_kind == "mpmt_dome"
        and direct_hit.slot is not None
        else None
    )
    rows: list[BoundarySlotNomination] = []
    for raw_slot in np.asarray(detector.exclusion_slots, dtype=np.int64):
        slot = int(raw_slot)
        score = residual_scores.get(slot)
        if score is None:
            continue
        clearance = exclusion_surface_clearance_mm(detector, point, slot)
        is_direct = direct_slot == slot
        reasons: list[str] = []
        if is_direct:
            reasons.append("direct_boundary_hit")
        if clearance <= float(config.proximity_margin_mm):
            reasons.append("near_fitted_boundary")
        residual_ok = (
            score.positive_deviance_nll >= float(config.minimum_positive_deviance)
            and score.positive_residual_pe >= float(config.minimum_positive_residual_pe)
        )
        if (
            residual_ok
            and clearance <= float(config.residual_proximity_margin_mm)
        ):
            reasons.append("local_charge_underprediction")
        if not reasons:
            continue
        # Residual evidence ranks physical alternatives; proximity is a gentle
        # tie-breaker, not a likelihood reward.
        ranking = (
            score.positive_deviance_nll
            + 0.15 * score.maximum_pmt_positive_deviance_nll
            + (1.0e6 if is_direct else 0.0)
            - 0.01 * max(clearance, 0.0)
        )
        rows.append(BoundarySlotNomination(
            interface=label,
            slot=slot,
            direct_boundary_hit=is_direct,
            boundary_surface_clearance_mm=float(clearance),
            residual=score,
            ranking_score=float(ranking),
            reasons=tuple(reasons),
        ))
    rows.sort(key=lambda row: (-row.ranking_score, row.slot))
    return rows[: max(0, int(config.max_slots_per_interface))]



def evaluate_local_wcpmt_selection(
    *,
    baseline_region: Mapping[str, float],
    candidate_region: Mapping[str, float],
    required_pmt_indices: Sequence[int],
    n_active_modes: int,
    n_tested_hypotheses: int,
    n_module_pmts: int,
    minimum_penalized_gain_nll: float,
    minimum_balanced_gain_per_pe: float,
    fallback_eligible: bool,
    expected_ratio_bounds: tuple[float, float] = (0.25, 4.0),
    additional_penalty_nll: float = 0.0,
) -> dict[str, object]:
    """Evaluate the guarded conditional WCPMT fallback statistic.

    The complete-event charge/BIC comparison remains authoritative whenever it
    accepts a candidate.  This statistic is only a fallback for a discrete,
    geometry-constrained WCPMT-entry hypothesis when the unmodelled straight-track
    residual outside that module would otherwise veto a strongly supported local
    crossing.

    The local likelihood is the module-conditional multinomial shape.  Complexity
    is charged using a BIC term for the active local fractions plus a full
    hypothesis/PMT look-elsewhere factor.  A per-PE guard subtracts the average
    shape degradation outside the module, preventing a small bright region from
    excusing an arbitrarily bad global line.
    """
    module_shape_gain = float(
        baseline_region["module_shape_nll"]
        - candidate_region["module_shape_nll"]
    )
    outside_degradation = float(max(
        candidate_region["outside_shape_nll"]
        - baseline_region["outside_shape_nll"],
        0.0,
    ))
    module_q = max(float(baseline_region["module_observed_pe"]), 1.0)
    outside_q = max(float(baseline_region["outside_observed_pe"]), 1.0)
    balanced = float(
        module_shape_gain / module_q - outside_degradation / outside_q
    )
    trials = max(
        1, int(n_tested_hypotheses) * max(int(n_module_pmts), 1)
    )
    penalty = float(
        0.5 * max(int(n_active_modes), 0) * math.log(float(max(n_module_pmts, 2)))
        + math.log(float(trials))
        + float(additional_penalty_nll)
    )
    penalized_gain = float(module_shape_gain - penalty)
    expected_ratio = float(
        candidate_region["module_expected_pe"] / module_q
    )
    ratio_lo, ratio_hi = (float(expected_ratio_bounds[0]), float(expected_ratio_bounds[1]))
    passes = bool(
        fallback_eligible
        and required_pmt_indices
        and penalized_gain > float(minimum_penalized_gain_nll)
        and balanced > float(minimum_balanced_gain_per_pe)
        and ratio_lo <= expected_ratio <= ratio_hi
    )
    return {
        "required_pmt_indices": [int(x) for x in required_pmt_indices],
        "module_shape_gain_nll": module_shape_gain,
        "outside_shape_degradation_nll": outside_degradation,
        "balanced_gain_per_pe": balanced,
        "look_elsewhere_trials": int(trials),
        "additional_penalty_nll": float(additional_penalty_nll),
        "penalty_nll": penalty,
        "penalized_gain_nll": penalized_gain,
        "module_expected_to_observed_ratio": expected_ratio,
        "fallback_eligible": bool(fallback_eligible),
        "passes": passes,
    }


def coarsened_wcpmt_multinomial_nll(
    observed_pes: Sequence[float],
    expected_pes: Sequence[float],
    module_target_groups: Sequence[tuple[Sequence[int], int]],
) -> float:
    """Charge-conditioned NLL with unresolved module optics coarse-grained.

    PMTs outside every crossed module remain separate multinomial bins.  Inside
    each explicit WCPMT-crossing module, the traversed PMT is one bin and the
    remaining PMTs are aggregated into a second bin.  This is an exact
    likelihood for the coarsened observations; it does not reweight residuals
    or insert an empirical hardware template.

    The grouping is useful when the geometry and Cherenkov yield identify the
    traversed PMT reliably but detailed gel/glass/reflector transfer among the
    other 18 PMTs is intentionally not simulated.
    """
    q = np.asarray(observed_pes, dtype=np.float64)
    lam = np.maximum(np.asarray(expected_pes, dtype=np.float64), 1.0e-300)
    if q.shape != lam.shape or q.ndim != 1:
        raise ValueError("observed and expected PMT arrays must be matching vectors")

    grouped_mask = np.zeros(q.size, dtype=bool)
    grouped_q: list[float] = []
    grouped_lam: list[float] = []
    for raw_indices, raw_target in module_target_groups:
        indices = np.unique(np.asarray(raw_indices, dtype=np.int64))
        target = int(raw_target)
        if indices.size < 2:
            raise ValueError("a WCPMT module group must contain at least two PMTs")
        if np.any(indices < 0) or np.any(indices >= q.size):
            raise ValueError("module PMT index is outside the detector vector")
        if target not in set(int(value) for value in indices):
            raise ValueError("the traversed PMT must belong to its module group")
        if np.any(grouped_mask[indices]):
            raise ValueError("coarsened mPMT module groups must be disjoint")
        grouped_mask[indices] = True
        rest = indices[indices != target]
        grouped_q.extend((float(q[target]), float(np.sum(q[rest]))))
        grouped_lam.extend((float(lam[target]), float(np.sum(lam[rest]))))

    outside = np.flatnonzero(~grouped_mask)
    if outside.size:
        grouped_q.extend(np.asarray(q[outside], dtype=np.float64).tolist())
        grouped_lam.extend(np.asarray(lam[outside], dtype=np.float64).tolist())
    q_grouped = np.asarray(grouped_q, dtype=np.float64)
    lam_grouped = np.maximum(np.asarray(grouped_lam, dtype=np.float64), 1.0e-300)
    if q_grouped.size == 0:
        return 0.0
    probability = lam_grouped / max(float(np.sum(lam_grouped)), 1.0e-300)
    return float(-np.sum(q_grouped * np.log(np.maximum(probability, 1.0e-300))))


def evaluate_coarsened_wcpmt_selection(
    *,
    observed_pes: Sequence[float],
    baseline_expected_pes: Sequence[float],
    candidate_expected_pes: Sequence[float],
    module_target_groups: Sequence[tuple[Sequence[int], int]],
    penalty_nll: float,
    minimum_penalized_gain_nll: float = 0.0,
    fallback_eligible: bool = True,
    expected_ratio_bounds: tuple[float, float] = (0.25, 4.0),
) -> dict[str, object]:
    """Evaluate a global, physically coarse-grained WCPMT model comparison.

    Baseline and candidate are compared on *the same* grouped data.  The
    existing BIC/discrete-hypothesis penalty is then subtracted once.  This is a
    global comparison because every PMT outside the unresolved crossed modules
    remains an individual likelihood bin.
    """
    q = np.asarray(observed_pes, dtype=np.float64)
    baseline = np.asarray(baseline_expected_pes, dtype=np.float64)
    candidate = np.asarray(candidate_expected_pes, dtype=np.float64)
    baseline_nll = coarsened_wcpmt_multinomial_nll(
        q, baseline, module_target_groups
    )
    candidate_nll = coarsened_wcpmt_multinomial_nll(
        q, candidate, module_target_groups
    )
    raw_gain = float(baseline_nll - candidate_nll)
    penalized_gain = float(raw_gain - float(penalty_nll))

    ratio_lo, ratio_hi = map(float, expected_ratio_bounds)
    ratios: list[float] = []
    ratio_passes = True
    for raw_indices, _ in module_target_groups:
        indices = np.unique(np.asarray(raw_indices, dtype=np.int64))
        observed_module = max(float(np.sum(q[indices])), 1.0e-12)
        ratio = float(np.sum(candidate[indices]) / observed_module)
        ratios.append(ratio)
        ratio_passes = bool(ratio_passes and ratio_lo <= ratio <= ratio_hi)
    passes = bool(
        fallback_eligible
        and len(module_target_groups) > 0
        and ratio_passes
        and penalized_gain > float(minimum_penalized_gain_nll)
    )
    return {
        "baseline_coarsened_nll": float(baseline_nll),
        "candidate_coarsened_nll": float(candidate_nll),
        "raw_gain_nll": float(raw_gain),
        "penalty_nll": float(penalty_nll),
        "penalized_gain_nll": float(penalized_gain),
        "module_expected_to_observed_ratios": ratios,
        "fallback_eligible": bool(fallback_eligible),
        "passes": bool(passes),
    }

def masked_module_times(
    observed_times_ns: Sequence[float],
    pmt_slots: Sequence[int],
    slots_to_mask: Sequence[int],
) -> np.ndarray:
    out = np.asarray(observed_times_ns, dtype=np.float64).copy()
    slots = np.asarray(pmt_slots, dtype=np.int64)
    wanted = np.asarray(tuple(int(x) for x in slots_to_mask), dtype=np.int64)
    if wanted.size:
        out[np.isin(slots, wanted)] = np.nan
    return np.ascontiguousarray(out, dtype=np.float64)


def _orthonormal_transverse(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = normalize_direction(direction)
    axis = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(axis, d))) > 0.85:
        axis = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    e1 = np.cross(d, axis)
    e1 /= float(np.linalg.norm(e1))
    e2 = np.cross(d, e1)
    e2 /= float(np.linalg.norm(e2))
    return e1, e2


def _shift_line_through_target(
    reference_mm: np.ndarray, direction: np.ndarray, target_mm: np.ndarray
) -> np.ndarray:
    t = float(np.dot(target_mm - reference_mm, direction))
    closest = reference_mm + t * direction
    return reference_mm + (target_mm - closest)


def _find_interior_reference_on_line(
    detector: ConvexDetectorVolume,
    line_point_mm: np.ndarray,
    direction: np.ndarray,
    *,
    interface: str,
) -> np.ndarray | None:
    """Find an active-water point on a candidate line near the module."""
    d = normalize_direction(direction)
    sign = 1.0 if str(interface).lower() == "entry" else -1.0
    # Search far enough to pass the complete WCTE mPMT vessel but remain local.
    distances = np.concatenate((
        np.linspace(0.0, 500.0, 101),
        np.linspace(520.0, 1400.0, 45),
    ))
    for distance in distances:
        candidate = line_point_mm + sign * float(distance) * d
        if detector.contains(candidate, tolerance_mm=1.0e-6):
            return np.ascontiguousarray(candidate, dtype=np.float64)
    return None


def build_mpmt_seed_values(
    *,
    baseline_values: Mapping[str, float],
    baseline_chart: TangentDirectionChart,
    baseline_track: RangeClippedTrack,
    module: ModuleGeometry,
    residual: ModuleResidualScore,
    interface: str,
    detector: ConvexDetectorVolume,
    config: MPMTBoundaryFitConfig,
) -> list[tuple[dict[str, float], TangentDirectionChart, dict[str, object]]]:
    """Construct a small truth-blind seed fan that crosses the nominated module."""
    d = baseline_chart.direction(
        float(baseline_values.get("dir_u", 0.0)),
        float(baseline_values.get("dir_v", 0.0)),
    )
    if d is None:
        return []
    d = normalize_direction(d)
    reference = np.asarray((
        baseline_values["x0"], baseline_values["y0"], baseline_values["z0"]
    ), dtype=np.float64)
    local_indices = module.pmt_indices
    # Prefer the largest positive residual, then the brightest observed PMT.
    global_targets = [
        int(residual.brightest_residual_pmt_index),
        int(residual.brightest_pmt_index),
    ]
    unique_targets: list[int] = []
    for index in global_targets:
        if index in set(int(x) for x in local_indices) and index not in unique_targets:
            unique_targets.append(index)
    unique_targets = unique_targets[: max(1, int(config.maximum_seed_pmts_per_slot))]
    if not unique_targets:
        unique_targets = [int(local_indices[0])]

    e1, e2 = _orthonormal_transverse(d)
    candidates: list[tuple[np.ndarray, str, int | None]] = []
    # Minimal shift from the baseline boundary point to the finite dome surface.
    boundary_point = _boundary_point_for_interface(baseline_track, interface)
    if boundary_point is not None:
        q = boundary_point - module.sphere_centre_mm
        norm_q = float(np.linalg.norm(q))
        if norm_q > 0.0:
            dome_target = (
                module.sphere_centre_mm
                + float(module.design.dome_outer_radius_mm) * q / norm_q
            )
            candidates.append((dome_target, "nearest_dome_surface", None))

    all_positions = np.asarray(module.pmt_positions_mm, dtype=np.float64)
    for global_index in unique_targets:
        local = int(np.flatnonzero(local_indices == global_index)[0])
        target = all_positions[local]
        candidates.append((target, "pmt_centre", global_index))
        for offset in config.seed_transverse_offsets_mm:
            offset = float(offset)
            if offset <= 0.0:
                continue
            candidates.append((target + offset * e1, "pmt_offset_e1", global_index))
            candidates.append((target - offset * e1, "pmt_offset_minus_e1", global_index))
            candidates.append((target + offset * e2, "pmt_offset_e2", global_index))
            candidates.append((target - offset * e2, "pmt_offset_minus_e2", global_index))

    out: list[tuple[dict[str, float], TangentDirectionChart, dict[str, object]]] = []
    seen: set[tuple[float, ...]] = set()
    label = str(interface).strip().lower()
    for target, family, pmt_index in candidates:
        values = dict(baseline_values)
        if label == "exit":
            # Exit-local hypotheses should pivot around the already reconstructed
            # start, not translate the complete track.  Aim the forward line at
            # the nominated local target; the driver subsequently keeps x0/y0/z0
            # fixed while optimizing direction and range.
            target_direction = normalize_direction(
                np.asarray(target, dtype=np.float64) - reference
            )
            chart = TangentDirectionChart.from_direction(target_direction)
            values.update({
                "x0": float(reference[0]),
                "y0": float(reference[1]),
                "z0": float(reference[2]),
                "dir_u": 0.0,
                "dir_v": 0.0,
            })
            key = tuple(np.round(np.concatenate((reference, target_direction)), 6))
            seed_reference = reference
        else:
            chart = TangentDirectionChart.from_direction(d)
            shifted_reference = _shift_line_through_target(reference, d, target)
            if not detector.contains(shifted_reference, tolerance_mm=1.0e-6):
                alternative = _find_interior_reference_on_line(
                    detector, target, d, interface=interface
                )
                if alternative is None:
                    continue
                shifted_reference = alternative
            key = tuple(np.round(shifted_reference, 5))
            values.update({
                "x0": float(shifted_reference[0]),
                "y0": float(shifted_reference[1]),
                "z0": float(shifted_reference[2]),
                "dir_u": 0.0,
                "dir_v": 0.0,
            })
            seed_reference = shifted_reference
        if key in seen:
            continue
        seen.add(key)
        out.append((values, chart, {
            "seed_family": str(family),
            "target_pmt_index": None if pmt_index is None else int(pmt_index),
            "target_point_mm": np.asarray(target, dtype=np.float64).tolist(),
            "reference_mm": np.asarray(seed_reference, dtype=np.float64).tolist(),
            "exit_start_anchor_fixed": bool(label == "exit"),
        }))
    return out
