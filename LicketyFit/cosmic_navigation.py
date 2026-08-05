"""Navigation helpers used only by the geometry-general cosmic fit mode.

The ordinary ``full_length`` and ``absorption`` engines deliberately do not
import this module.  Its purpose is to prevent the broad cosmic proxy from
silently collapsing onto a single saturated-energy shape.  A Cherenkov pattern
changes only weakly once a muon is sufficiently relativistic, so the raw proxy
ranking can otherwise retain several nearly identical high-range candidates and
remove the stopping-range basin before the exact likelihood is evaluated.

The helpers below keep the selection energy-independent.  They stratify a seed
by the dimensionless ratio

    fitted CSDA range / distance from modeled start to the downstream boundary.

That ratio is geometry-derived and directly separates short stopping tracks,
long stopping tracks, near-boundary/exiting tracks, and saturated high-range
tracks.  No expected or true particle energy enters the construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np

from .cosmic_track_fit import ConvexDetectorVolume, resolve_range_clipped_track
from .track_parameterization import (
    TangentDirectionChart,
    direction_from_mapping,
    fibonacci_sphere_directions,
    normalize_direction,
    stable_tangent_basis,
)


DEFAULT_RANGE_RATIO_EDGES = (0.55, 1.0, 1.75)


@dataclass(frozen=True)
class StopExitRangeAnchors:
    """Deterministic line-fixed range anchors on both sides of a boundary.

    ``stop_mm`` lies strictly below the downstream water distance and
    ``exit_mm`` lies strictly above it.  Keeping the topology boundary out of
    both sets avoids numerical classification flips at equality while still
    allowing the exact likelihood to compare the two physical branches.
    """

    stop_mm: tuple[float, ...]
    exit_mm: tuple[float, ...]
    boundary_mm: float
    stop_upper_mm: float
    exit_lower_mm: float
    epsilon_mm: float


def _finite_unique_ranges(values: Sequence[float], lo: float, hi: float) -> list[float]:
    """Return sorted finite ranges in ``[lo, hi]`` with stable de-duplication."""

    out: list[float] = []
    seen: set[float] = set()
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or value < float(lo) or value > float(hi):
            continue
        key = round(value, 8)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    out.sort()
    return out


def _select_log_diverse_ranges(
    pool: Sequence[float],
    mandatory: Sequence[float],
    *,
    n_keep: int,
) -> tuple[float, ...]:
    """Keep mandatory values then maximize logarithmic distance greedily."""

    n_keep = max(1, int(n_keep))
    candidates = [float(x) for x in pool if math.isfinite(float(x)) and float(x) > 0.0]
    chosen: list[float] = []
    for raw in mandatory:
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            continue
        if all(abs(value - old) > 1.0e-7 for old in chosen):
            chosen.append(value)
        if len(chosen) >= n_keep:
            return tuple(sorted(chosen[:n_keep]))
    remaining = [
        value for value in candidates
        if all(abs(value - old) > 1.0e-7 for old in chosen)
    ]
    while remaining and len(chosen) < n_keep:
        if not chosen:
            value = remaining[len(remaining) // 2]
        else:
            value = max(
                remaining,
                key=lambda x: min(abs(math.log(x / old)) for old in chosen),
            )
        chosen.append(float(value))
        remaining.remove(value)
    return tuple(sorted(chosen))


def build_stop_exit_range_anchors(
    distance_to_boundary_mm: float,
    current_range_mm: float,
    minimum_range_mm: float,
    maximum_range_mm: float,
    *,
    candidate_ranges_mm: Sequence[float] = (),
    energy_anchor_ranges_mm: Sequence[float] = (),
    seed_ranges_mm: Sequence[float] = (),
    stop_fractions: Sequence[float] = (0.25, 0.45, 0.65, 0.82, 0.93, 0.985),
    exit_factors: Sequence[float] = (1.005, 1.05, 1.15, 1.35, 1.75, 2.50, 4.00),
    boundary_epsilon_mm: float = 2.0,
    maximum_per_branch: int = 6,
) -> StopExitRangeAnchors:
    """Build a symmetric, energy-blind stop/exit bracket for one fitted line.

    The boundary distance is supplied by analytic detector clipping.  Range
    table values may be included as broad anchors, but no nominal event energy
    or truth is used.  Both branches are retained even when the current fit is
    on only one side of the topology boundary.
    """

    boundary = float(distance_to_boundary_mm)
    current = float(current_range_mm)
    lo = max(1.0e-6, float(minimum_range_mm))
    hi = float(maximum_range_mm)
    eps = max(1.0e-6, float(boundary_epsilon_mm))
    if not (math.isfinite(boundary) and math.isfinite(current) and math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("range bracket inputs must be finite")
    if boundary <= lo or hi <= lo:
        raise ValueError("invalid physical range interval")

    # Keep a small absolute gap around equality.  The production resolver has a
    # topology tolerance, so sampling exactly at the wall would make the branch
    # label depend on roundoff rather than physics.
    stop_hi = min(hi, boundary - eps)
    exit_lo = max(lo, boundary + eps)

    common = list(candidate_ranges_mm) + list(energy_anchor_ranges_mm) + list(seed_ranges_mm)
    stop_raw: list[float] = [lo, stop_hi]
    stop_raw.extend(boundary * float(f) for f in stop_fractions)
    stop_raw.extend(value for value in common if float(value) < boundary)
    if current < boundary:
        stop_raw.append(current)

    exit_raw: list[float] = [exit_lo, hi]
    exit_raw.extend(boundary * float(f) for f in exit_factors)
    exit_raw.extend(value for value in common if float(value) >= boundary)
    if current >= boundary:
        exit_raw.append(current)

    stop_pool = (
        _finite_unique_ranges(stop_raw, lo, stop_hi)
        if stop_hi >= lo else []
    )
    exit_pool = (
        _finite_unique_ranges(exit_raw, exit_lo, hi)
        if exit_lo <= hi else []
    )

    stop_mandatory = [stop_hi]
    if current < boundary:
        stop_mandatory.insert(0, min(max(current, lo), stop_hi))
    if stop_pool:
        stop_mandatory.append(stop_pool[0])

    exit_mandatory = [exit_lo]
    if current >= boundary:
        exit_mandatory.insert(0, min(max(current, exit_lo), hi))
    if exit_pool:
        exit_mandatory.append(exit_pool[-1])

    return StopExitRangeAnchors(
        stop_mm=_select_log_diverse_ranges(
            stop_pool, stop_mandatory, n_keep=maximum_per_branch
        ) if stop_pool else (),
        exit_mm=_select_log_diverse_ranges(
            exit_pool, exit_mandatory, n_keep=maximum_per_branch
        ) if exit_pool else (),
        boundary_mm=boundary,
        stop_upper_mm=float(stop_hi),
        exit_lower_mm=float(exit_lo),
        epsilon_mm=eps,
    )


def safeguarded_log_parabolic_range_candidate(
    rows: Sequence[Mapping[str, object]],
    lower_mm: float,
    upper_mm: float,
    *,
    range_key: str = "full_range_mm",
    objective_key: str = "fval",
    minimum_relative_separation: float = 2.0e-3,
) -> float | None:
    """Suggest one exact refinement point from line-fixed range samples.

    A positive quadratic in ``log(range)`` is used only when the best point is
    bracketed.  At an edge, the geometric midpoint to the nearest sample is
    returned.  Every suggestion is bounded and kept away from existing points;
    the caller still evaluates the unchanged exact objective.
    """

    finite: list[tuple[float, float]] = []
    for row in rows:
        try:
            x = float(row[range_key])
            y = float(row[objective_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and x > 0.0 and math.isfinite(y):
            finite.append((x, y))
    finite.sort()
    if len(finite) < 2:
        return None
    best_i = min(range(len(finite)), key=lambda i: finite[i][1])
    candidate: float | None = None
    if 0 < best_i < len(finite) - 1:
        triple = finite[best_i - 1: best_i + 2]
        xx = np.log(np.asarray([p[0] for p in triple], dtype=np.float64))
        yy = np.asarray([p[1] for p in triple], dtype=np.float64)
        matrix = np.column_stack((xx * xx, xx, np.ones(3, dtype=np.float64)))
        try:
            a, b, _ = np.linalg.solve(matrix, yy)
        except np.linalg.LinAlgError:
            a = b = math.nan
        if math.isfinite(a) and math.isfinite(b) and a > 0.0:
            vertex = -b / (2.0 * a)
            if xx[0] < vertex < xx[-1]:
                candidate = float(math.exp(vertex))
    if candidate is None:
        best_x = finite[best_i][0]
        neighbor_i = 1 if best_i == 0 else len(finite) - 2 if best_i == len(finite) - 1 else None
        if neighbor_i is None:
            return None
        candidate = math.sqrt(best_x * finite[neighbor_i][0])

    candidate = min(max(float(candidate), float(lower_mm)), float(upper_mm))
    if not math.isfinite(candidate) or candidate <= 0.0:
        return None
    if any(
        abs(math.log(candidate / x)) < float(minimum_relative_separation)
        for x, _ in finite
    ):
        return None
    return candidate


@dataclass(frozen=True)
class TangentDirectionFanCandidate:
    direction: np.ndarray
    tangent_u: float
    tangent_v: float
    angular_offset_rad: float
    azimuth_rad: float


def build_tangent_direction_fan(
    direction: Sequence[float],
    *,
    angular_radii_deg: Sequence[float] = (1.0, 2.5, 5.0),
    azimuth_count: int = 8,
) -> tuple[TangentDirectionFanCandidate, ...]:
    """Return a deterministic local full-azimuth direction fan.

    The fan is expressed in the same normalized tangent chart used by the exact
    optimizer.  Angular radii are exact because the chart radius is
    ``tan(delta_angle)``.  This fills the angular gaps between compact global
    proxy directions without assuming a preferred detector axis.
    """

    chart = TangentDirectionChart.from_direction(direction)
    nphi = max(4, int(azimuth_count))
    out: list[TangentDirectionFanCandidate] = []
    for radius_deg in angular_radii_deg:
        radius = math.radians(float(radius_deg))
        if not math.isfinite(radius) or radius <= 0.0 or radius >= 0.5 * math.pi:
            continue
        tangent_radius = math.tan(radius)
        for i in range(nphi):
            phi = 2.0 * math.pi * float(i) / float(nphi)
            u = tangent_radius * math.cos(phi)
            v = tangent_radius * math.sin(phi)
            candidate_direction = chart.direction(u, v)
            if candidate_direction is None:
                continue
            out.append(TangentDirectionFanCandidate(
                direction=np.ascontiguousarray(candidate_direction, dtype=np.float64),
                tangent_u=float(u),
                tangent_v=float(v),
                angular_offset_rad=float(radius),
                azimuth_rad=float(phi),
            ))
    return tuple(out)


def pivot_reference_for_direction_fan(
    visible_start_mm: Sequence[float],
    visible_endpoint_mm: Sequence[float],
    new_direction: Sequence[float],
    *,
    start_hypothesis: str,
) -> np.ndarray:
    """Rotate a fitted visible segment about its midpoint.

    For an internal start, the returned reference remains the physical start of
    a segment with the same midpoint and visible length.  For a boundary-entry
    hypothesis, any point on the line is equivalent because detector clipping
    derives the physical entry; the midpoint is the numerically best-conditioned
    reference.
    """

    start = np.asarray(visible_start_mm, dtype=np.float64)
    endpoint = np.asarray(visible_endpoint_mm, dtype=np.float64)
    d = normalize_direction(new_direction)
    if start.shape != (3,) or endpoint.shape != (3,):
        raise ValueError("visible endpoints must be three-vectors")
    midpoint = 0.5 * (start + endpoint)
    if str(start_hypothesis) == "boundary_entry":
        return np.ascontiguousarray(midpoint, dtype=np.float64)
    if str(start_hypothesis) != "internal_start":
        raise ValueError(f"unknown start_hypothesis={start_hypothesis!r}")
    visible = float(np.linalg.norm(endpoint - start))
    return np.ascontiguousarray(midpoint - 0.5 * visible * d, dtype=np.float64)


@dataclass(frozen=True)
class CosmicSeedRangeClassification:
    """Immutable range-to-boundary classification for a fixed seed library."""

    strata: np.ndarray
    ratios: np.ndarray
    edges: tuple[float, ...]
    counts: dict[int, int]
    invalid_count: int

    def metadata(self) -> dict[str, object]:
        labels = range_stratum_labels(self.edges)
        return {
            "ratio_edges": [float(x) for x in self.edges],
            "stratum_labels": {str(k): str(v) for k, v in labels.items()},
            "stratum_counts": {
                str(int(k)): int(v) for k, v in sorted(self.counts.items())
            },
            "invalid_count": int(self.invalid_count),
        }


def validate_range_ratio_edges(
    edges: Sequence[float] = DEFAULT_RANGE_RATIO_EDGES,
) -> tuple[float, ...]:
    """Return finite, positive, strictly increasing ratio edges."""

    out = tuple(float(x) for x in edges)
    if not out:
        raise ValueError("At least one cosmic range-ratio edge is required")
    if not all(math.isfinite(x) and x > 0.0 for x in out):
        raise ValueError("Cosmic range-ratio edges must be finite and positive")
    if any(b <= a for a, b in zip(out[:-1], out[1:])):
        raise ValueError("Cosmic range-ratio edges must be strictly increasing")
    return out


def range_stratum_labels(
    edges: Sequence[float] = DEFAULT_RANGE_RATIO_EDGES,
) -> dict[int, str]:
    """Human-readable labels for the default physical range regions."""

    values = validate_range_ratio_edges(edges)
    labels: dict[int, str] = {}
    lower = 0.0
    for i, upper in enumerate(values):
        if i == 0:
            labels[i] = f"short_stop_ratio_le_{upper:g}"
        elif upper <= 1.0 + 1.0e-12:
            labels[i] = f"long_stop_ratio_{lower:g}_to_{upper:g}"
        else:
            labels[i] = f"transition_or_exit_ratio_{lower:g}_to_{upper:g}"
        lower = upper
    labels[len(values)] = f"high_or_saturated_ratio_gt_{values[-1]:g}"
    return labels


def range_stratum_from_ratio(
    ratio: float,
    edges: Sequence[float] = DEFAULT_RANGE_RATIO_EDGES,
) -> int:
    """Map a positive range-to-boundary ratio to a compact integer stratum."""

    values = validate_range_ratio_edges(edges)
    value = float(ratio)
    if not math.isfinite(value) or value <= 0.0:
        return -1
    return int(np.searchsorted(np.asarray(values), value, side="right"))


def classify_cosmic_seed_ranges(
    seeds: Sequence[Mapping[str, object]],
    detector: ConvexDetectorVolume,
    *,
    start_hypothesis: Callable[[Mapping[str, object]], str],
    inset_mm: float = 0.5,
    edges: Sequence[float] = DEFAULT_RANGE_RATIO_EDGES,
    range_key: str = "length",
) -> CosmicSeedRangeClassification:
    """Classify a fixed cosmic seed library once during driver initialization.

    ``range_key`` is ``length`` in the cosmic engine because that internal
    coordinate is the full CSDA range to Cherenkov threshold.  The visible
    segment is derived later by detector clipping.
    """

    ratio_edges = validate_range_ratio_edges(edges)
    nseed = len(seeds)
    strata = np.full(nseed, -1, dtype=np.int8)
    ratios = np.full(nseed, np.nan, dtype=np.float64)

    for i, seed in enumerate(seeds):
        try:
            reference = np.asarray(
                [float(seed["x0"]), float(seed["y0"]), float(seed["z0"])],
                dtype=np.float64,
            )
            direction = direction_from_mapping(seed)
            full_range = float(seed[range_key])
            hypothesis = str(start_hypothesis(seed))
            resolved = resolve_range_clipped_track(
                detector,
                reference,
                direction,
                full_range,
                starts_at_boundary=(hypothesis == "boundary_entry"),
                inset_mm=float(inset_mm),
            )
            if resolved is None or resolved.distance_to_boundary_mm <= 0.0:
                continue
            ratio = float(full_range / resolved.distance_to_boundary_mm)
            stratum = range_stratum_from_ratio(ratio, ratio_edges)
            if stratum < 0:
                continue
            ratios[i] = ratio
            strata[i] = np.int8(stratum)
        except Exception:
            # An invalid seed will receive a zero/non-finite proxy prediction and
            # cannot become a useful candidate.  Keep classification best-effort
            # rather than making setup brittle over a diagnostic seed.
            continue

    counts = {
        int(value): int(np.count_nonzero(strata == value))
        for value in sorted(int(x) for x in np.unique(strata) if int(x) >= 0)
    }
    return CosmicSeedRangeClassification(
        strata=np.ascontiguousarray(strata, dtype=np.int8),
        ratios=np.ascontiguousarray(ratios, dtype=np.float64),
        edges=ratio_edges,
        counts=counts,
        invalid_count=int(np.count_nonzero(strata < 0)),
    )


def _ordered_indices(scores: np.ndarray, indices: Sequence[int]) -> list[int]:
    candidates = np.asarray([int(i) for i in indices], dtype=np.int64)
    if candidates.size == 0:
        return []
    local_scores = np.asarray(scores, dtype=np.float64)[candidates]
    finite_key = np.where(np.isfinite(local_scores), local_scores, np.inf)
    # mergesort makes ties deterministic with respect to the seed-library order.
    order = np.argsort(finite_key, kind="mergesort")
    return [int(candidates[int(j)]) for j in order]


def select_range_stratified_seed_indices(
    scores: np.ndarray,
    seeds: Sequence[Mapping[str, object]],
    indices: Sequence[int],
    strata: Sequence[int],
    *,
    n_keep: int,
    raw_top: int,
    geometry_representatives_per_stratum: int = 2,
) -> list[int]:
    """Select proxy candidates with mandatory range-basin coverage.

    Selection order is:

    1. the best proxy candidate in every available valid range stratum;
    2. additional *geometrically distinct* representatives in each stratum;
    3. the raw proxy-top candidates;
    4. distinct directions;
    5. distinct vertices;
    6. distinct absolute ranges;
    7. remaining candidates by proxy score.

    The first two steps are the essential cosmic safeguard.  A saturated long
    track can mimic a shorter track after shifting its start upstream, so one
    global representative per range stratum is insufficient: the correct short
    track may be the second-best geometry in that stratum even though it becomes
    the best basin after exact charge/timing closure.  Representatives are
    selected round-robin by stratum so one candidate per available stratum is
    always retained before any second representative is added.
    """

    n_keep = max(1, int(n_keep))
    raw_top = max(0, int(raw_top))
    geometry_representatives_per_stratum = max(
        1, int(geometry_representatives_per_stratum)
    )
    if len(strata) != len(seeds):
        raise ValueError("strata must have one entry per seed")

    order = _ordered_indices(scores, indices)
    if not order:
        return []

    selected: list[int] = []
    seen: set[int] = set()

    def add(index: int) -> bool:
        idx = int(index)
        if idx in seen or len(selected) >= n_keep:
            return False
        seen.add(idx)
        selected.append(idx)
        return True

    # Keep several geometry hypotheses inside each range region.  The seed
    # library intentionally repeats a vertex/direction geometry over many range
    # anchors, so geometry is identified without the range coordinate.  Round
    # one guarantees every available stratum; later rounds add the next-best
    # distinct geometry in each stratum while budget remains.
    def geometry_key(index: int) -> tuple[object, ...]:
        seed = seeds[int(index)]
        direction = direction_from_mapping(seed)
        return (
            str(seed.get("track_start_hypothesis", "")),
            round(float(seed["x0"]), 3),
            round(float(seed["y0"]), 3),
            round(float(seed["z0"]), 3),
            round(float(direction[0]), 6),
            round(float(direction[1]), 6),
            round(float(direction[2]), 6),
        )

    representatives: dict[int, list[int]] = {}
    geometry_seen: dict[int, set[tuple[object, ...]]] = {}
    for index in order:
        stratum = int(strata[int(index)])
        if stratum < 0:
            continue
        rows = representatives.setdefault(stratum, [])
        if len(rows) >= geometry_representatives_per_stratum:
            continue
        key = geometry_key(index)
        seen_geometry = geometry_seen.setdefault(stratum, set())
        if key in seen_geometry:
            continue
        seen_geometry.add(key)
        rows.append(int(index))

    available_strata = sorted(representatives)
    for representative_round in range(geometry_representatives_per_stratum):
        round_rows = [
            representatives[stratum][representative_round]
            for stratum in available_strata
            if len(representatives[stratum]) > representative_round
        ]
        for index in sorted(
            round_rows,
            key=lambda i: (
                float(scores[int(i)])
                if np.isfinite(scores[int(i)]) else math.inf,
                int(i),
            ),
        ):
            add(index)

    for index in order[:raw_top]:
        add(index)

    direction_target = min(n_keep, len(selected) + max(2, n_keep // 3))
    seen_direction = {
        tuple(np.round(direction_from_mapping(seeds[i]), 3)) for i in selected
    }
    for index in order:
        key = tuple(np.round(direction_from_mapping(seeds[index]), 3))
        if key in seen_direction:
            continue
        if add(index):
            seen_direction.add(key)
        if len(selected) >= direction_target:
            break

    vertex_target = min(n_keep, len(selected) + max(2, n_keep // 3))
    seen_vertex = {
        tuple(round(float(seeds[i][key]), 1) for key in ("x0", "y0", "z0"))
        for i in selected
    }
    for index in order:
        seed = seeds[index]
        key = tuple(round(float(seed[name]), 1) for name in ("x0", "y0", "z0"))
        if key in seen_vertex:
            continue
        if add(index):
            seen_vertex.add(key)
        if len(selected) >= vertex_target:
            break

    range_name = "visible_length" if "visible_length" in seeds[order[0]] else "length"
    seen_range = {round(float(seeds[i][range_name]), 1) for i in selected}
    for index in order:
        key = round(float(seeds[index][range_name]), 1)
        if key in seen_range:
            continue
        if add(index):
            seen_range.add(key)
        if len(selected) >= n_keep:
            break

    for index in order:
        add(index)
        if len(selected) >= n_keep:
            break
    return selected


def select_range_stratified_positions(
    scores: Sequence[float],
    strata: Sequence[int],
    *,
    n_keep: int,
    raw_top: int = 2,
) -> list[int]:
    """Select positions from an already compact candidate list.

    This is used after exact charge closure.  It guarantees one candidate from
    each retained range stratum before filling the remaining joint-likelihood
    budget with the lowest exact charge NLL rows.
    """

    score_array = np.asarray(scores, dtype=np.float64)
    stratum_array = np.asarray(strata, dtype=np.int16)
    if score_array.ndim != 1 or stratum_array.shape != score_array.shape:
        raise ValueError("scores and strata must be one-dimensional and aligned")
    n_keep = max(1, min(int(n_keep), int(score_array.size)))
    raw_top = max(0, int(raw_top))
    order = list(np.argsort(
        np.where(np.isfinite(score_array), score_array, np.inf),
        kind="mergesort",
    ).astype(int))

    selected: list[int] = []
    seen: set[int] = set()

    def add(position: int) -> None:
        pos = int(position)
        if pos not in seen and len(selected) < n_keep:
            seen.add(pos)
            selected.append(pos)

    representatives: dict[int, int] = {}
    for position in order:
        stratum = int(stratum_array[position])
        if stratum >= 0 and stratum not in representatives:
            representatives[stratum] = int(position)
    for position in sorted(
        representatives.values(),
        key=lambda p: (
            float(score_array[p]) if np.isfinite(score_array[p]) else math.inf,
            int(p),
        ),
    ):
        add(position)

    for position in order[:raw_top]:
        add(position)
    for position in order:
        add(position)
        if len(selected) >= n_keep:
            break
    return selected

def select_cosmic_tournament_positions(
    exact_start_scores: Sequence[float],
    hypotheses: Sequence[str],
    baseline_flags: Sequence[bool],
    range_guard_flags: Sequence[bool],
    *,
    max_probes: int = 4,
    alternate_start_gate_nll: float = 125.0,
) -> list[int]:
    """Choose a compact, symmetric set of cosmic basins for exact probing.

    The best exact-start row from each available start hypothesis is mandatory.
    A second row per hypothesis is then retained when it represents the other
    navigation origin (baseline versus range guard), or otherwise the next best
    distinct row.  Alternate rows must be reasonably close to the globally best
    exact start.  The rule is deterministic and does not use an expected energy.
    """

    scores = np.asarray(exact_start_scores, dtype=np.float64)
    hypotheses = [str(x) for x in hypotheses]
    baseline = np.asarray(baseline_flags, dtype=bool)
    guards = np.asarray(range_guard_flags, dtype=bool)
    n = int(scores.size)
    if not (len(hypotheses) == baseline.size == guards.size == n):
        raise ValueError("cosmic tournament inputs must have equal length")
    max_probes = max(1, int(max_probes))
    finite = [i for i in range(n) if math.isfinite(float(scores[i]))]
    if not finite:
        return []

    unique_hypotheses: list[str] = []
    for i in sorted(finite, key=lambda j: (float(scores[j]), int(j))):
        if hypotheses[i] not in unique_hypotheses:
            unique_hypotheses.append(hypotheses[i])

    primary: dict[str, int] = {}
    for hypothesis in unique_hypotheses:
        rows = [i for i in finite if hypotheses[i] == hypothesis]
        primary[hypothesis] = min(rows, key=lambda j: (float(scores[j]), int(j)))

    selected = sorted(
        primary.values(), key=lambda j: (float(scores[j]), int(j))
    )[:max_probes]
    seen = set(selected)
    if len(selected) >= max_probes:
        return selected

    global_best = min(float(scores[i]) for i in finite)
    gate = max(0.0, float(alternate_start_gate_nll))
    alternates: list[int] = []
    for hypothesis in unique_hypotheses:
        p = primary[hypothesis]
        rows = [
            i for i in finite
            if hypotheses[i] == hypothesis
            and i != p
            and float(scores[i]) <= global_best + gate
        ]
        if not rows:
            continue
        p_baseline = bool(baseline[p])
        p_guard = bool(guards[p])

        def alternate_key(i: int) -> tuple[object, ...]:
            # Prefer a row from the navigation origin not represented by the
            # primary.  If the primary belongs to both sets, ordinary NLL order
            # is the least-assumptive tie-breaker.
            opposite = (
                (p_baseline and bool(guards[i]) and not bool(baseline[i]))
                or (p_guard and bool(baseline[i]) and not bool(guards[i]))
            )
            distinct_origin = (
                bool(baseline[i]) != p_baseline
                or bool(guards[i]) != p_guard
            )
            return (
                0 if opposite else 1 if distinct_origin else 2,
                float(scores[i]),
                int(i),
            )

        alternates.append(min(rows, key=alternate_key))

    global_primary_hypothesis = hypotheses[min(
        primary.values(), key=lambda j: (float(scores[j]), int(j))
    )]
    for i in sorted(
        alternates,
        key=lambda j: (
            0 if hypotheses[j] == global_primary_hypothesis else 1,
            float(scores[j]),
            int(j),
        ),
    ):
        if i not in seen:
            seen.add(i)
            selected.append(i)
        if len(selected) >= max_probes:
            return selected

    # Do not fill spare budget with a third nearly redundant row from one
    # hypothesis.  The tournament is deliberately capped at one primary and one
    # navigation-origin alternate per start hypothesis; broader exact-start
    # coverage is retained by the strict post-tournament escape.
    return selected


def select_cosmic_continuation_positions(
    probe_scores: Sequence[float],
    exact_start_scores: Sequence[float],
    *,
    max_continuations: int = 2,
    probe_gate_nll: float = 18.0,
    strong_descent_nll: float = 40.0,
) -> list[int]:
    """Choose which one-sweep cosmic probes receive scheduled continuation.

    The lowest probe score always continues.  A second basin may continue when
    it is still close in exact NLL or when the first sweep made a large descent,
    which is the characteristic signature of a distant but viable local basin.
    """

    probe = np.asarray(probe_scores, dtype=np.float64)
    starts = np.asarray(exact_start_scores, dtype=np.float64)
    if probe.shape != starts.shape or probe.ndim != 1:
        raise ValueError("probe and exact-start scores must be aligned vectors")
    finite = [i for i in range(probe.size) if math.isfinite(float(probe[i]))]
    if not finite:
        return []
    limit = max(1, int(max_continuations))
    best = min(finite, key=lambda i: (float(probe[i]), int(i)))
    selected = [best]
    if limit == 1:
        return selected

    best_score = float(probe[best])
    gate = max(0.0, float(probe_gate_nll))
    strong = max(0.0, float(strong_descent_nll))
    eligible = []
    for i in finite:
        if i == best:
            continue
        gap = float(probe[i]) - best_score
        descent = float(starts[i]) - float(probe[i])
        within_gate = gap <= gate
        strong_descent = descent >= strong
        if not (within_gate or strong_descent):
            continue
        eligible.append((
            0 if within_gate and strong_descent else 1 if within_gate else 2,
            gap if within_gate else math.inf,
            -descent,
            float(probe[i]),
            int(i),
        ))
    eligible.sort()
    for row in eligible:
        selected.append(int(row[-1]))
        if len(selected) >= limit:
            break
    return selected



# =============================================================================
# Analytic causal-timing navigation
# =============================================================================

_LIGHT_SPEED_MM_PER_NS = 299.792458


@dataclass(frozen=True)
class CausalTimingScore:
    """Robust timing score for a finite oriented detector segment."""

    score: float
    t0_ns: float
    n_hits: int
    n_inliers: int
    segment_start_mm: np.ndarray
    direction: np.ndarray
    segment_length_mm: float


@dataclass(frozen=True)
class CausalTimingLineCandidate:
    """A detector chord found by the geometry-only causal timing search."""

    score: float
    t0_ns: float
    reference_mm: np.ndarray
    direction: np.ndarray
    entry_mm: np.ndarray
    exit_mm: np.ndarray
    chord_length_mm: float
    n_hits: int
    n_inliers: int


@dataclass(frozen=True)
class CausalTimingSeedCandidate:
    """Event-specific cosmic seed derived from a causal timing line."""

    score: float
    line_score: float
    t0_ns: float
    start_fraction: float
    end_fraction: float
    track_start_hypothesis: str
    topology: str
    line_rank: int
    seed: dict[str, object]

    def metadata(self) -> dict[str, object]:
        return {
            "score": float(self.score),
            "line_score": float(self.line_score),
            "t0_ns": float(self.t0_ns),
            "start_fraction": float(self.start_fraction),
            "end_fraction": float(self.end_fraction),
            "track_start_hypothesis": str(self.track_start_hypothesis),
            "topology": str(self.topology),
            "line_rank": int(self.line_rank),
            "seed": dict(self.seed),
        }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return math.nan
    x = values[good]
    w = weights[good]
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w)
    return float(x[int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))])


def _weighted_median_prevalidated(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted median for finite positive arrays validated once per search."""
    order = np.argsort(values, kind="mergesort")
    x = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    if cumulative.size == 0 or not math.isfinite(float(cumulative[-1])):
        return math.nan
    return float(
        x[int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))]
    )


def _causal_direct_arrival_times_prevalidated(
    positions: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    *,
    refractive_index: float,
    beta: float,
    light_speed_mm_per_ns: float = _LIGHT_SPEED_MM_PER_NS,
) -> np.ndarray:
    """Allocation-light causal time evaluation for prevalidated arrays."""
    n = float(refractive_index)
    b = float(beta)
    c = float(light_speed_mm_per_ns)
    q2 = n * n * b * b - 1.0
    displacement = positions - start[None, :]
    longitudinal = displacement @ direction
    radial2 = np.maximum(
        np.einsum("ij,ij->i", displacement, displacement)
        - longitudinal * longitudinal,
        0.0,
    )
    radial = np.sqrt(radial2)
    source_s = np.clip(longitudinal - radial / math.sqrt(q2), 0.0, length)
    photon_vector = displacement - source_s[:, None] * direction[None, :]
    photon_distance = np.sqrt(
        np.einsum("ij,ij->i", photon_vector, photon_vector)
    )
    return np.ascontiguousarray(
        source_s / (b * c) + n * photon_distance / c,
        dtype=np.float64,
    )


def _prepare_causal_score_arrays(
    positions: np.ndarray,
    charge: np.ndarray,
    times: np.ndarray,
    *,
    base_sigma_ns: float,
    charge_sigma_ns: float,
    max_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pp = np.ascontiguousarray(positions, dtype=np.float64)
    qq = np.ascontiguousarray(charge, dtype=np.float64)
    tt = np.ascontiguousarray(times, dtype=np.float64)
    if (
        pp.ndim != 2
        or pp.shape[1] != 3
        or qq.shape != (pp.shape[0],)
        or tt.shape != qq.shape
        or np.any(~np.isfinite(pp))
        or np.any(~np.isfinite(qq))
        or np.any(qq <= 0.0)
        or np.any(~np.isfinite(tt))
    ):
        raise ValueError("prepared causal timing arrays must be finite and aligned")
    weights = np.ascontiguousarray(
        np.clip(np.sqrt(np.maximum(qq, 1.0e-12)), 0.35, float(max_weight)),
        dtype=np.float64,
    )
    sigma = np.ascontiguousarray(
        float(base_sigma_ns)
        + float(charge_sigma_ns) / np.sqrt(np.maximum(qq, 1.0e-6)),
        dtype=np.float64,
    )
    return pp, qq, tt, weights, sigma


def _robust_causal_timing_score_preselected(
    positions: np.ndarray,
    charge: np.ndarray,
    times: np.ndarray,
    weights: np.ndarray,
    sigma: np.ndarray,
    segment_start_mm: Sequence[float],
    direction: Sequence[float],
    segment_length_mm: float,
    *,
    refractive_index: float = 1.344,
    beta: float = 1.0,
    huber_transition_sigma: float = 2.5,
    trim_fraction: float = 0.22,
    minimum_hits: int = 8,
) -> CausalTimingScore | None:
    """Exact robust score after one-time hit validation and weighting."""
    n_hits = int(times.size)
    if n_hits < int(minimum_hits):
        return None
    start = np.asarray(segment_start_mm, dtype=np.float64)
    if start.shape != (3,) or np.any(~np.isfinite(start)):
        return None
    try:
        d = normalize_direction(direction)
    except ValueError:
        return None
    length = float(segment_length_mm)
    n = float(refractive_index)
    b = float(beta)
    if (
        not math.isfinite(length)
        or length <= 0.0
        or not math.isfinite(n)
        or not math.isfinite(b)
        or n <= 0.0
        or b <= 0.0
        or n * n * b * b <= 1.0
    ):
        return None
    predicted = _causal_direct_arrival_times_prevalidated(
        positions,
        start,
        d,
        length,
        refractive_index=n,
        beta=b,
    )
    residual = times - predicted
    t0 = _weighted_median_prevalidated(residual, weights)
    if not math.isfinite(t0):
        return None
    standardized = (residual - t0) / sigma
    abs_z = np.abs(standardized)
    transition = max(float(huber_transition_sigma), 1.0e-6)
    loss = np.where(
        abs_z <= transition,
        0.5 * standardized * standardized,
        transition * (abs_z - 0.5 * transition),
    )
    n_keep = max(
        int(minimum_hits),
        int(math.ceil((1.0 - float(trim_fraction)) * n_hits)),
    )
    n_keep = min(max(1, n_keep), n_hits)
    order = np.argsort(loss, kind="mergesort")[:n_keep]
    denominator = float(np.sum(weights[order]))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    score = float(np.sum(weights[order] * loss[order]) / denominator)
    if not math.isfinite(score):
        return None
    return CausalTimingScore(
        score=score,
        t0_ns=float(t0),
        n_hits=n_hits,
        n_inliers=int(n_keep),
        segment_start_mm=np.ascontiguousarray(start, dtype=np.float64),
        direction=np.ascontiguousarray(d, dtype=np.float64),
        segment_length_mm=length,
    )


def _angle_degrees(a: Sequence[float], b: Sequence[float]) -> float:
    aa = normalize_direction(a)
    bb = normalize_direction(b)
    return float(math.degrees(math.acos(float(np.clip(np.dot(aa, bb), -1.0, 1.0)))))


def causal_direct_arrival_times(
    pmt_positions_mm: np.ndarray,
    segment_start_mm: Sequence[float],
    direction: Sequence[float],
    segment_length_mm: float,
    *,
    refractive_index: float = 1.344,
    beta: float = 1.0,
    light_speed_mm_per_ns: float = _LIGHT_SPEED_MM_PER_NS,
) -> np.ndarray:
    """Earliest direct-photon time from a finite straight track.

    The returned time excludes the unknown additive event time.  It is the
    analytic minimum of particle flight time plus photon flight time, with the
    unconstrained Cherenkov source coordinate clipped to the finite segment.
    No detector-specific direction convention or assumed particle energy is
    used; ``beta=1`` is deliberately used only by the coarse navigator.
    """

    positions = np.asarray(pmt_positions_mm, dtype=np.float64)
    start = np.asarray(segment_start_mm, dtype=np.float64)
    d = normalize_direction(direction)
    length = float(segment_length_mm)
    n = float(refractive_index)
    b = float(beta)
    c = float(light_speed_mm_per_ns)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("pmt_positions_mm must have shape (N,3)")
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError("segment_start_mm must be a finite 3-vector")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("segment_length_mm must be positive and finite")
    if not (math.isfinite(n) and math.isfinite(b) and n > 0.0 and b > 0.0):
        raise ValueError("refractive_index and beta must be positive and finite")
    q2 = n * n * b * b - 1.0
    if q2 <= 0.0:
        raise ValueError("track is below the Cherenkov threshold")

    displacement = positions - start[None, :]
    longitudinal = displacement @ d
    radial2 = np.maximum(
        np.einsum("ij,ij->i", displacement, displacement)
        - longitudinal * longitudinal,
        0.0,
    )
    radial = np.sqrt(radial2)
    source_s = np.clip(longitudinal - radial / math.sqrt(q2), 0.0, length)
    photon_vector = displacement - source_s[:, None] * d[None, :]
    photon_distance = np.sqrt(np.einsum("ij,ij->i", photon_vector, photon_vector))
    return np.ascontiguousarray(
        source_s / (b * c) + n * photon_distance / c,
        dtype=np.float64,
    )


def robust_causal_timing_score(
    pmt_positions_mm: np.ndarray,
    observed_charge_pe: np.ndarray,
    observed_time_ns: np.ndarray,
    segment_start_mm: Sequence[float],
    direction: Sequence[float],
    segment_length_mm: float,
    *,
    refractive_index: float = 1.344,
    beta: float = 1.0,
    huber_transition_sigma: float = 2.5,
    trim_fraction: float = 0.22,
    base_sigma_ns: float = 0.45,
    charge_sigma_ns: float = 1.20,
    max_weight: float = 4.0,
    minimum_hits: int = 8,
) -> CausalTimingScore | None:
    """Score a finite segment after analytically profiling the unknown ``t0``.

    The score is a charge-weighted, trimmed Huber loss.  It is intentionally a
    robust navigation statistic rather than a replacement for the exact PMT
    first-arrival likelihood.  A minority of reflected/scattered photons cannot
    dominate it, while the oriented particle-flight term distinguishes ``d``
    from ``-d``.
    """

    positions = np.asarray(pmt_positions_mm, dtype=np.float64)
    charge = np.asarray(observed_charge_pe, dtype=np.float64)
    times = np.asarray(observed_time_ns, dtype=np.float64)
    if positions.shape != (charge.size, 3) or times.shape != charge.shape:
        raise ValueError("PMT positions, charge, and time arrays are not aligned")
    good = (
        np.isfinite(charge) & (charge > 0.0)
        & np.isfinite(times)
        & np.all(np.isfinite(positions), axis=1)
    )
    if int(np.count_nonzero(good)) < int(minimum_hits):
        return None
    pp = positions[good]
    qq = charge[good]
    tt = times[good]
    try:
        predicted = causal_direct_arrival_times(
            pp,
            segment_start_mm,
            direction,
            segment_length_mm,
            refractive_index=float(refractive_index),
            beta=float(beta),
        )
    except (ValueError, FloatingPointError):
        return None
    residual = tt - predicted
    weights = np.clip(np.sqrt(np.maximum(qq, 1.0e-12)), 0.35, float(max_weight))
    t0 = _weighted_median(residual, weights)
    if not math.isfinite(t0):
        return None
    sigma = float(base_sigma_ns) + float(charge_sigma_ns) / np.sqrt(
        np.maximum(qq, 1.0e-6)
    )
    standardized = (residual - t0) / sigma
    abs_z = np.abs(standardized)
    transition = max(float(huber_transition_sigma), 1.0e-6)
    loss = np.where(
        abs_z <= transition,
        0.5 * standardized * standardized,
        transition * (abs_z - 0.5 * transition),
    )
    n = int(loss.size)
    n_keep = max(int(minimum_hits), int(math.ceil((1.0 - float(trim_fraction)) * n)))
    n_keep = min(max(1, n_keep), n)
    order = np.argsort(loss, kind="mergesort")[:n_keep]
    denominator = float(np.sum(weights[order]))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    score = float(np.sum(weights[order] * loss[order]) / denominator)
    if not math.isfinite(score):
        return None
    return CausalTimingScore(
        score=score,
        t0_ns=float(t0),
        n_hits=n,
        n_inliers=int(n_keep),
        segment_start_mm=np.ascontiguousarray(segment_start_mm, dtype=np.float64),
        direction=np.ascontiguousarray(normalize_direction(direction), dtype=np.float64),
        segment_length_mm=float(segment_length_mm),
    )


def _select_navigation_hits(
    pmt_positions_mm: np.ndarray,
    observed_charge_pe: np.ndarray,
    observed_time_ns: np.ndarray,
    *,
    maximum_hits: int,
    earliest_fraction: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(pmt_positions_mm, dtype=np.float64)
    charge = np.asarray(observed_charge_pe, dtype=np.float64)
    times = np.asarray(observed_time_ns, dtype=np.float64)
    good = np.flatnonzero(
        np.isfinite(charge) & (charge > 0.0)
        & np.isfinite(times)
        & np.all(np.isfinite(positions), axis=1)
    )
    if good.size <= max(1, int(maximum_hits)):
        return positions[good], charge[good], times[good]
    limit = max(8, int(maximum_hits))
    n_early = min(good.size, max(8, int(round(float(earliest_fraction) * limit))))
    n_charge = max(0, limit - n_early)
    early = good[np.argsort(times[good], kind="mergesort")[:n_early]]
    bright = good[np.argsort(-charge[good], kind="mergesort")[:n_charge]]
    selected = []
    seen: set[int] = set()
    for index in np.concatenate((early, bright, good)):
        i = int(index)
        if i not in seen:
            seen.add(i)
            selected.append(i)
        if len(selected) >= limit:
            break
    selected_array = np.asarray(selected, dtype=np.int64)
    return (
        np.ascontiguousarray(positions[selected_array], dtype=np.float64),
        np.ascontiguousarray(charge[selected_array], dtype=np.float64),
        np.ascontiguousarray(times[selected_array], dtype=np.float64),
    )


def _project_reference_inside(
    detector: ConvexDetectorVolume,
    centre: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray | None:
    try:
        point = detector.project_step(centre, displacement)
    except Exception:
        return None
    point = np.asarray(point, dtype=np.float64)
    # ``project_step`` may land numerically on the surface.  Pull a projected
    # point very slightly toward the robust detector centre before clipping.
    point = centre + 0.999 * (point - centre)
    return point if detector.contains(point, tolerance_mm=1.0e-5) else None


def find_causal_timing_lines(
    detector: ConvexDetectorVolume,
    pmt_positions_mm: np.ndarray,
    observed_charge_pe: np.ndarray,
    observed_time_ns: np.ndarray,
    *,
    refractive_index: float = 1.344,
    direction_count: int = 72,
    retained_lines: int = 10,
    refinement_rounds: int = 2,
    minimum_direction_separation_deg: float = 8.0,
    maximum_hits: int = 360,
    boundary_inset_mm: float = 0.5,
) -> list[CausalTimingLineCandidate]:
    """Find a small, full-sphere set of oriented detector lines analytically."""

    pp, qq, tt = _select_navigation_hits(
        pmt_positions_mm,
        observed_charge_pe,
        observed_time_ns,
        maximum_hits=max(8, int(maximum_hits)),
    )
    if pp.shape[0] < 8:
        return []
    pp, qq, tt, score_weights, score_sigma = _prepare_causal_score_arrays(
        pp,
        qq,
        tt,
        base_sigma_ns=0.45,
        charge_sigma_ns=1.20,
        max_weight=4.0,
    )
    centre = np.asarray(detector.reference_center, dtype=np.float64)
    if centre.shape != (3,) or not detector.contains(centre):
        centre = 0.5 * (np.asarray(detector.axis_lo) + np.asarray(detector.axis_hi))
    charge_weight = np.clip(np.sqrt(np.maximum(qq, 1.0e-12)), 0.2, 4.0)
    charge_centroid = np.sum(pp * charge_weight[:, None], axis=0) / np.sum(charge_weight)
    n_early = max(8, min(32, max(8, pp.shape[0] // 4)))
    early_indices = np.argsort(tt, kind="mergesort")[:n_early]
    early_weight = charge_weight[early_indices]
    early_centroid = (
        np.sum(pp[early_indices] * early_weight[:, None], axis=0)
        / np.sum(early_weight)
    )
    detector_scale = float(np.min(np.asarray(detector.axis_hi) - np.asarray(detector.axis_lo)))
    transverse_scale = max(50.0, 0.22 * detector_scale)

    def score_line(direction: np.ndarray, reference: np.ndarray):
        clipped = resolve_range_clipped_track(
            detector,
            reference,
            direction,
            max(1.0, 10.0 * detector_scale),
            starts_at_boundary=True,
            inset_mm=float(boundary_inset_mm),
        )
        if clipped is None:
            return None
        scored = _robust_causal_timing_score_preselected(
            pp,
            qq,
            tt,
            score_weights,
            score_sigma,
            clipped.start,
            clipped.direction,
            clipped.visible_length_mm,
            refractive_index=float(refractive_index),
        )
        if scored is None:
            return None
        return CausalTimingLineCandidate(
            score=float(scored.score),
            t0_ns=float(scored.t0_ns),
            reference_mm=np.ascontiguousarray(reference, dtype=np.float64),
            direction=np.ascontiguousarray(clipped.direction, dtype=np.float64),
            entry_mm=np.ascontiguousarray(clipped.start, dtype=np.float64),
            exit_mm=np.ascontiguousarray(clipped.boundary_exit, dtype=np.float64),
            chord_length_mm=float(clipped.visible_length_mm),
            n_hits=int(scored.n_hits),
            n_inliers=int(scored.n_inliers),
        )

    raw: list[CausalTimingLineCandidate] = []
    for direction in fibonacci_sphere_directions(max(12, int(direction_count))):
        d, e1, e2 = stable_tangent_basis(direction)
        charge_offset = charge_centroid - centre
        charge_offset -= float(np.dot(charge_offset, d)) * d
        early_offset = early_centroid - centre
        early_offset -= float(np.dot(early_offset, d)) * d
        displacements = [
            np.zeros(3, dtype=np.float64),
            0.45 * charge_offset,
            0.90 * charge_offset,
            0.45 * early_offset,
            0.90 * early_offset,
            transverse_scale * e1,
            -transverse_scale * e1,
            transverse_scale * e2,
            -transverse_scale * e2,
        ]
        best = None
        for displacement in displacements:
            reference = _project_reference_inside(detector, centre, displacement)
            if reference is None:
                continue
            candidate = score_line(d, reference)
            if candidate is not None and (
                best is None or candidate.score < best.score
            ):
                best = candidate
        if best is not None:
            raw.append(best)
    raw.sort(key=lambda row: (float(row.score), float(row.t0_ns)))

    diverse: list[CausalTimingLineCandidate] = []
    for row in raw:
        if all(
            _angle_degrees(row.direction, kept.direction)
            >= float(minimum_direction_separation_deg)
            for kept in diverse
        ):
            diverse.append(row)
        if len(diverse) >= max(1, int(retained_lines)):
            break

    refined: list[CausalTimingLineCandidate] = []
    for initial in diverse:
        current = initial
        line_step = 0.55 * transverse_scale
        direction_step = 0.10
        for _ in range(max(0, int(refinement_rounds))):
            d, e1, e2 = stable_tangent_basis(current.direction)
            line_options = [current]
            for a in (-1.0, 0.0, 1.0):
                for b in (-1.0, 0.0, 1.0):
                    if a == 0.0 and b == 0.0:
                        continue
                    reference = _project_reference_inside(
                        detector,
                        current.reference_mm,
                        line_step * (a * e1 + b * e2),
                    )
                    if reference is None:
                        continue
                    candidate = score_line(d, reference)
                    if candidate is not None:
                        line_options.append(candidate)
            current = min(line_options, key=lambda row: float(row.score))

            d, e1, e2 = stable_tangent_basis(current.direction)
            direction_options = [current]
            for a in (-1.0, 0.0, 1.0):
                for b in (-1.0, 0.0, 1.0):
                    if a == 0.0 and b == 0.0:
                        continue
                    trial_direction = normalize_direction(
                        d + direction_step * (a * e1 + b * e2)
                    )
                    candidate = score_line(trial_direction, current.reference_mm)
                    if candidate is not None:
                        direction_options.append(candidate)
            current = min(direction_options, key=lambda row: float(row.score))
            line_step *= 0.45
            direction_step *= 0.45
        refined.append(current)
    refined.sort(key=lambda row: (float(row.score), float(row.t0_ns)))
    return refined



def refine_local_causal_timing_lines(
    detector: ConvexDetectorVolume,
    pmt_positions_mm: np.ndarray,
    observed_charge_pe: np.ndarray,
    observed_time_ns: np.ndarray,
    reference_mm: Sequence[float],
    direction: Sequence[float],
    *,
    starts_at_boundary: bool,
    refractive_index: float = 1.344,
    maximum_hits: int = 120,
    earliest_fraction: float = 0.72,
    refinement_rounds: int = 3,
    initial_transverse_step_mm: float = 90.0,
    initial_direction_step_deg: float = 3.0,
    retained_lines: int = 6,
    minimum_line_separation_mm: float = 12.0,
    minimum_direction_separation_deg: float = 0.35,
    boundary_inset_mm: float = 0.5,
) -> list[CausalTimingLineCandidate]:
    """Refine an already selected cosmic line with early first-arrival causality.

    This is a deliberately *local* navigation guard.  The global charge/timing
    tournament remains authoritative; this routine only proposes nearby oriented
    detector lines for later evaluation with the unchanged exact likelihood.

    The score uses a hit subset dominated by the earliest timestamps so that late
    coherent multiple scattering cannot completely redefine the inferred entry
    tangent.  Reflections and scattered photons are handled robustly by the
    trimmed Huber statistic.  No truth position, nominal energy, or detector axis
    enters the construction.
    """

    pp, qq, tt = _select_navigation_hits(
        pmt_positions_mm,
        observed_charge_pe,
        observed_time_ns,
        maximum_hits=max(8, int(maximum_hits)),
        earliest_fraction=float(earliest_fraction),
    )
    if pp.shape[0] < 8:
        return []
    pp, qq, tt, score_weights, score_sigma = _prepare_causal_score_arrays(
        pp,
        qq,
        tt,
        base_sigma_ns=0.40,
        charge_sigma_ns=1.00,
        max_weight=4.0,
    )

    reference0 = np.asarray(reference_mm, dtype=np.float64)
    if reference0.shape != (3,) or not np.all(np.isfinite(reference0)):
        raise ValueError("reference_mm must be a finite three-vector")
    direction0 = normalize_direction(direction)

    def score_line(trial_direction: np.ndarray, trial_reference: np.ndarray):
        clipped = resolve_range_clipped_track(
            detector,
            trial_reference,
            trial_direction,
            max(1.0, 10.0 * float(np.max(np.asarray(detector.axis_hi) - np.asarray(detector.axis_lo)))),
            starts_at_boundary=bool(starts_at_boundary),
            inset_mm=float(boundary_inset_mm),
        )
        if clipped is None:
            return None
        scored = _robust_causal_timing_score_preselected(
            pp,
            qq,
            tt,
            score_weights,
            score_sigma,
            clipped.start,
            clipped.direction,
            clipped.visible_length_mm,
            refractive_index=float(refractive_index),
            # The local guard intentionally gives the earliest subset more
            # leverage than the broad global navigator while retaining robust
            # trimming for indirect-light outliers.
            trim_fraction=0.18,
            minimum_hits=8,
        )
        if scored is None:
            return None
        # Use the visible midpoint as the canonical numerical reference.  For a
        # boundary-entry hypothesis this removes the line's unobservable
        # longitudinal gauge; for an internal start it remains a stable point
        # from which the caller can reconstruct the physical start.
        midpoint = 0.5 * (clipped.start + clipped.endpoint)
        return CausalTimingLineCandidate(
            score=float(scored.score),
            t0_ns=float(scored.t0_ns),
            reference_mm=np.ascontiguousarray(midpoint, dtype=np.float64),
            direction=np.ascontiguousarray(clipped.direction, dtype=np.float64),
            entry_mm=np.ascontiguousarray(clipped.start, dtype=np.float64),
            exit_mm=np.ascontiguousarray(clipped.boundary_exit, dtype=np.float64),
            chord_length_mm=float(clipped.visible_length_mm),
            n_hits=int(scored.n_hits),
            n_inliers=int(scored.n_inliers),
        )

    initial = score_line(direction0, reference0)
    if initial is None:
        return []

    current = initial
    all_rows: list[CausalTimingLineCandidate] = [initial]
    transverse_step = max(1.0, float(initial_transverse_step_mm))
    direction_step = max(0.05, float(initial_direction_step_deg))

    for _ in range(max(0, int(refinement_rounds))):
        d, e1, e2 = stable_tangent_basis(current.direction)
        line_rows = [current]
        for a, b in (
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
        ):
            trial_ref = _project_reference_inside(
                detector,
                current.reference_mm,
                transverse_step * (a * e1 + b * e2),
            )
            if trial_ref is None:
                continue
            row = score_line(d, trial_ref)
            if row is not None:
                line_rows.append(row)
                all_rows.append(row)
        current = min(line_rows, key=lambda row: float(row.score))

        d, e1, e2 = stable_tangent_basis(current.direction)
        direction_rows = [current]
        tangent_radius = math.tan(math.radians(direction_step))
        for i in range(8):
            phi = 2.0 * math.pi * float(i) / 8.0
            trial_direction = normalize_direction(
                d + tangent_radius * (math.cos(phi) * e1 + math.sin(phi) * e2)
            )
            row = score_line(trial_direction, current.reference_mm)
            if row is not None:
                direction_rows.append(row)
                all_rows.append(row)
        current = min(direction_rows, key=lambda row: float(row.score))
        transverse_step *= 0.45
        direction_step *= 0.45

    # Stable de-duplication and diversity.  Keep the original line even when it
    # is not among the lowest scores so downstream diagnostics can measure the
    # causal gain relative to the accepted fit.
    all_rows.sort(key=lambda row: (float(row.score), float(row.t0_ns)))
    selected: list[CausalTimingLineCandidate] = []
    for row in all_rows:
        duplicate = False
        for kept in selected:
            angle = _angle_degrees(row.direction, kept.direction)
            delta = row.entry_mm - kept.entry_mm
            delta -= float(np.dot(delta, kept.direction)) * kept.direction
            transverse = float(np.linalg.norm(delta))
            if (
                angle < float(minimum_direction_separation_deg)
                and transverse < float(minimum_line_separation_mm)
            ):
                duplicate = True
                break
        if not duplicate:
            selected.append(row)
        if len(selected) >= max(1, int(retained_lines)):
            break

    if all(
        _angle_degrees(initial.direction, row.direction) >= 1.0e-8
        or float(np.linalg.norm(initial.entry_mm - row.entry_mm)) >= 1.0e-6
        for row in selected
    ):
        selected.append(initial)
    selected.sort(key=lambda row: (float(row.score), float(row.t0_ns)))
    return selected[: max(1, int(retained_lines))]

def build_causal_timing_seed_guard(
    detector: ConvexDetectorVolume,
    pmt_positions_mm: np.ndarray,
    observed_charge_pe: np.ndarray,
    observed_time_ns: np.ndarray,
    *,
    refractive_index: float = 1.344,
    maximum_full_range_mm: float,
    boundary_inset_mm: float = 0.5,
    direction_count: int = 72,
    retained_lines: int = 8,
    refinement_rounds: int = 2,
    maximum_hits: int = 360,
    maximum_seeds: int = 8,
    maximum_per_topology: int = 2,
    minimum_segment_mm: float = 120.0,
) -> tuple[list[CausalTimingSeedCandidate], list[CausalTimingLineCandidate]]:
    """Construct compact event-specific seeds for all four cosmic topologies.

    A full-sphere causal line search is followed by a small finite-segment scan.
    Segment endpoints, not an assumed particle energy, determine whether a seed
    begins/ends at a detector boundary.  The exact optical likelihood remains
    responsible for final basin selection and energy estimation.
    """

    lines = find_causal_timing_lines(
        detector,
        pmt_positions_mm,
        observed_charge_pe,
        observed_time_ns,
        refractive_index=float(refractive_index),
        direction_count=int(direction_count),
        retained_lines=int(retained_lines),
        refinement_rounds=int(refinement_rounds),
        maximum_hits=int(maximum_hits),
        boundary_inset_mm=float(boundary_inset_mm),
    )
    if not lines:
        return [], []

    pp, qq, tt = _select_navigation_hits(
        pmt_positions_mm,
        observed_charge_pe,
        observed_time_ns,
        maximum_hits=max(8, int(maximum_hits)),
    )
    pp, qq, tt, score_weights, score_sigma = _prepare_causal_score_arrays(
        pp,
        qq,
        tt,
        base_sigma_ns=0.45,
        charge_sigma_ns=1.20,
        max_weight=4.0,
    )
    # First-arrival timing locates the oriented line but only weakly constrains
    # a late optical start or stopping endpoint.  Sample the longitudinal
    # coordinates in detector-chord fractions, not in energy or WCTE-specific
    # millimetres.  The direct-primary charge prefilter in the driver selects
    # the event-supported segment.  This grid covers near-entry, central, and
    # near-wall starts while retaining both short stopping and exiting tracks.
    start_fractions = (0.0, 0.08, 0.20, 0.35, 0.50, 0.65, 0.80)
    remaining_end_fractions = (0.25, 0.45, 0.70, 1.0)
    rows: list[CausalTimingSeedCandidate] = []
    maximum_range = float(maximum_full_range_mm)

    for line_rank, line in enumerate(lines):
        chord = float(line.chord_length_mm)
        if not math.isfinite(chord) or chord <= float(minimum_segment_mm):
            continue
        for start_fraction in start_fractions:
            for remaining_fraction in remaining_end_fractions:
                end_fraction = float(
                    start_fraction
                    + remaining_fraction * (1.0 - start_fraction)
                )
                if end_fraction <= start_fraction:
                    continue
                segment_length = (end_fraction - start_fraction) * chord
                if segment_length < float(minimum_segment_mm):
                    continue
                segment_start = line.entry_mm + start_fraction * chord * line.direction
                scored = _robust_causal_timing_score_preselected(
                    pp,
                    qq,
                    tt,
                    score_weights,
                    score_sigma,
                    segment_start,
                    line.direction,
                    segment_length,
                    refractive_index=float(refractive_index),
                )
                if scored is None:
                    continue
                starts_at_boundary = bool(start_fraction <= 1.0e-9)
                exits_detector = bool(end_fraction >= 1.0 - 1.0e-9)
                hypothesis = "boundary_entry" if starts_at_boundary else "internal_start"
                topology = (
                    "boundary_entry_boundary_exit"
                    if starts_at_boundary and exits_detector else
                    "boundary_entry_internal_stop"
                    if starts_at_boundary else
                    "internal_start_boundary_exit"
                    if exits_detector else
                    "internal_start_internal_stop"
                )
                if exits_detector:
                    full_range = min(
                        maximum_range,
                        max(segment_length + 250.0, 1.28 * segment_length),
                    )
                else:
                    full_range = segment_length
                if not math.isfinite(full_range) or full_range <= 0.0:
                    continue
                if starts_at_boundary:
                    reference = 0.5 * (line.entry_mm + line.exit_mm)
                else:
                    reference = segment_start
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
                    "t0": float(scored.t0_ns),
                    "track_start_hypothesis": str(hypothesis),
                    "seed_family": "causal_timing_guard",
                    "causal_timing_score": float(scored.score),
                    "causal_line_score": float(line.score),
                    "causal_line_rank": int(line_rank),
                    "causal_start_fraction": float(start_fraction),
                    "causal_end_fraction": float(end_fraction),
                    "causal_topology": str(topology),
                }
                rows.append(CausalTimingSeedCandidate(
                    score=float(scored.score),
                    line_score=float(line.score),
                    t0_ns=float(scored.t0_ns),
                    start_fraction=float(start_fraction),
                    end_fraction=float(end_fraction),
                    track_start_hypothesis=str(hypothesis),
                    topology=str(topology),
                    line_rank=int(line_rank),
                    seed=seed,
                ))

    # The causal guard's principal job is to protect an oriented line basin;
    # stopping versus exiting is continuous in the fitted remaining range, while
    # internal-start versus boundary-entry is the only discrete choice.  Pair
    # the strongest timing lines with both start hypotheses.  This is more
    # robust than spending the budget on four endpoint labels and prevents a
    # good direction from being deleted merely because its first finite-segment
    # timing score preferred the wrong end fraction.
    by_line_hypothesis: dict[
        tuple[int, str], list[CausalTimingSeedCandidate]
    ] = {}
    for row in rows:
        by_line_hypothesis.setdefault(
            (int(row.line_rank), str(row.track_start_hypothesis)), []
        ).append(row)

    selected: list[CausalTimingSeedCandidate] = []
    seen_signature: set[tuple[object, ...]] = set()

    def add(row: CausalTimingSeedCandidate) -> None:
        signature = (
            str(row.track_start_hypothesis),
            int(row.line_rank),
            round(float(row.start_fraction), 4),
            round(float(row.end_fraction), 4),
        )
        if signature not in seen_signature and len(selected) < int(maximum_seeds):
            seen_signature.add(signature)
            selected.append(row)

    pair_count = min(
        len(lines),
        max(1, int(maximum_seeds) // 2),
    )
    for line_rank in range(pair_count):
        for hypothesis in ("internal_start", "boundary_entry"):
            candidates = sorted(
                by_line_hypothesis.get((int(line_rank), hypothesis), []),
                key=lambda row: (
                    float(row.score),
                    float(row.line_score),
                    str(row.topology),
                ),
            )
            if candidates:
                add(candidates[0])

    # If an unusually short chord did not support both hypotheses for a top
    # line, restore the missing discrete hypothesis from the globally best row.
    represented_hypotheses = {
        str(row.track_start_hypothesis) for row in selected
    }
    for hypothesis in ("internal_start", "boundary_entry"):
        if hypothesis in represented_hypotheses:
            continue
        candidates = sorted(
            (row for row in rows if row.track_start_hypothesis == hypothesis),
            key=lambda row: (
                float(row.score),
                float(row.line_score),
                int(row.line_rank),
            ),
        )
        if candidates:
            add(candidates[0])

    # Fill any remaining odd budget or missing-line slots by the robust timing
    # statistic. ``maximum_per_topology`` is retained in the API for backward
    # compatibility but the physical quota is now the two start hypotheses.
    for row in sorted(
        rows,
        key=lambda item: (
            float(item.score),
            float(item.line_score),
            int(item.line_rank),
        ),
    ):
        add(row)
        if len(selected) >= int(maximum_seeds):
            break
    selected.sort(
        key=lambda item: (
            float(item.score),
            float(item.line_score),
            int(item.line_rank),
        )
    )
    return selected, lines
