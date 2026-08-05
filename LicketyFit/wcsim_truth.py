"""Fast optional WCSim per-step truth diagnostics for fitted events.

The production fit never consumes these values.  When explicitly requested by
``scripts/batch_fit_driver.py``, this module reads a separate WCSim
``AllSecondaries`` per-step TTree and records where the simulated primary first
enters the same *active-water* volume used by the cosmic geometry.

The reader is designed for multi-gigabyte ROOT files:

* ``uproot`` is imported lazily, so ordinary fits pay no ROOT startup cost;
* only the requested event entry ranges are read;
* monotonic event ranges are located from TTree basket boundaries and cached in a persistent sidecar, with a streaming event-branch fallback;
* the main pass reads only numeric branches needed to identify the crossing;
* optional material/volume strings and stored direction branches are disabled
  by default and, when requested, fetched only for selected crossing rows;
* all per-step geometry work is vectorized or stops at the first crossing for a
  track, rather than looping over every later step inside the detector.

A material label alone is deliberately insufficient.  In WCTE a particle can
enter water, traverse an active mPMT vessel/dome assembly, and only later enter
the connected central active-water region.  Entry is therefore defined by the
provided detector-volume predicate (the WCTE prism minus active non-water dome
caps in the current release), with pre/post step values interpolated to the
geometry crossing.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


DEFAULT_TREE_NAME = "AllSecondaries"
DEFAULT_UPROOT_STEP_SIZE = "64 MB"
DEFAULT_EVENT_INDEX_STEP_SIZE = "16 MB"
DEFAULT_UPROOT_IO_WORKERS = 4
DEFAULT_USE_EVENT_INDEX_CACHE = True
DEFAULT_INCLUDE_OPTIONAL_DETAILS = False
_EVENT_INDEX_FORMAT_VERSION = 3

# Canonical branch names used internally. Matching is case-insensitive and
# ignores punctuation, but the resolved names are recorded in metadata.
_REQUIRED_BRANCHES = (
    "evt",
    "parent",
    "trk",
    "x_cm",
    "y_cm",
    "z_cm",
    "t_ns",
    "ke_MeV",
    "post_x_cm",
    "post_y_cm",
    "post_z_cm",
    "post_t_ns",
    "post_ke_MeV",
)
_MAIN_OPTIONAL_BRANCHES = (
    "pdg",
    "particle",
    "step",
)
_SPARSE_NUMERIC_BRANCHES = (
    "dir_x",
    "dir_y",
    "dir_z",
    "post_dir_x",
    "post_dir_y",
    "post_dir_z",
)
_SPARSE_TEXT_BRANCHES = (
    "material",
    "post_material",
    "volume",
    "post_volume",
    "step_process",
)
_OPTIONAL_BRANCHES = (
    *_MAIN_OPTIONAL_BRANCHES,
    *_SPARSE_NUMERIC_BRANCHES,
    *_SPARSE_TEXT_BRANCHES,
    "creator",
)

_FLOAT_OUTPUT_FIELDS = {
    "truth_active_water_entry_x_mm": "x_mm",
    "truth_active_water_entry_y_mm": "y_mm",
    "truth_active_water_entry_z_mm": "z_mm",
    "truth_active_water_entry_t_ns": "t_ns",
    "truth_active_water_entry_ke_mev": "ke_mev",
    "truth_active_water_entry_dir_x": "dir_x",
    "truth_active_water_entry_dir_y": "dir_y",
    "truth_active_water_entry_dir_z": "dir_z",
    "truth_active_water_entry_segment_fraction": "segment_fraction",
    "truth_primary_start_x_mm": "primary_start_x_mm",
    "truth_primary_start_y_mm": "primary_start_y_mm",
    "truth_primary_start_z_mm": "primary_start_z_mm",
    "truth_primary_start_t_ns": "primary_start_t_ns",
    "truth_primary_start_ke_mev": "primary_start_ke_mev",
}
_INT_OUTPUT_FIELDS = {
    "truth_active_water_entry_root_event_id": "root_event_id",
    "truth_active_water_entry_track_id": "track_id",
    "truth_active_water_entry_pdg": "pdg",
    "truth_active_water_entry_step": "step",
    "truth_primary_candidate_count": "primary_candidate_count",
}
_STRING_OUTPUT_FIELDS = {
    "truth_active_water_entry_status": "status",
    "truth_primary_selection_status": "primary_selection_status",
    "truth_active_water_entry_method": "entry_method",
    "truth_active_water_entry_pre_material": "pre_material",
    "truth_active_water_entry_post_material": "post_material",
    "truth_active_water_entry_pre_volume": "pre_volume",
    "truth_active_water_entry_post_volume": "post_volume",
    "truth_active_water_entry_step_process": "step_process",
}

_PARTICLE_ABS_PDG = {
    "electron": 11,
    "e": 11,
    "muon": 13,
    "mu": 13,
    "pion": 211,
    "pi": 211,
    "kaon": 321,
    "k": 321,
    "proton": 2212,
    "p": 2212,
}


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _particle_abs_pdg(particle: str) -> int | None:
    key = str(particle).strip().lower().replace("+", "").replace("-", "")
    key = key.replace("_", "").replace(" ", "")
    if key in {"mu", "muon", "muplus", "muminus"}:
        return 13
    if key in {"e", "electron", "eplus", "eminus", "positron"}:
        return 11
    if key in {"pi", "pion", "piplus", "piminus"}:
        return 211
    if key in {"k", "kaon", "kplus", "kminus"}:
        return 321
    if key in {"p", "proton", "antiproton"}:
        return 2212
    return _PARTICLE_ABS_PDG.get(key)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(value).decode("latin-1", errors="replace")
    return str(value)


def _as_1d(values: Any, *, dtype=None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _finite_int(value: Any, default: int = -1) -> int:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return int(default)
    if not math.isfinite(out):
        return int(default)
    return int(round(out))


def _normalise_direction(vector: Sequence[float]) -> tuple[float, float, float]:
    direction = np.asarray(vector, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        return (math.nan, math.nan, math.nan)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        return (math.nan, math.nan, math.nan)
    direction /= norm
    return (float(direction[0]), float(direction[1]), float(direction[2]))


def _contains_outer_many(
    detector,
    points_mm: np.ndarray,
    *,
    tolerance_mm: float = 1.0e-7,
) -> np.ndarray:
    """Vectorized detector predicate excluding local dome-cap intrusions."""
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_mm must have shape (N,3)")
    n_points = int(points.shape[0])
    if n_points == 0:
        return np.empty(0, dtype=bool)

    required = ("axis_lo", "axis_hi", "boundary_model")
    if not all(hasattr(detector, name) for name in required):
        # No generic outer-only API exists. Fall back to the complete predicate.
        return np.asarray(
            [bool(detector.contains(point, tolerance_mm=tolerance_mm)) for point in points],
            dtype=bool,
        )

    valid = np.all(np.isfinite(points), axis=1)
    axis_lo = np.asarray(detector.axis_lo, dtype=np.float64)
    axis_hi = np.asarray(detector.axis_hi, dtype=np.float64)
    margin = max(float(getattr(detector, "margin_mm", 0.0)), 0.0)
    valid &= np.all(points >= axis_lo[None, :] + margin - tolerance_mm, axis=1)
    valid &= np.all(points <= axis_hi[None, :] - margin + tolerance_mm, axis=1)

    model = str(getattr(detector, "boundary_model", "convex_mpmt_planes"))
    if model == "wcte_prism":
        n_sides = int(getattr(detector, "prism_n_sides", 0))
        limit = float(getattr(detector, "prism_apothem_mm", math.nan)) - margin
        if n_sides <= 0 or not math.isfinite(limit):
            return np.asarray(
                [bool(detector.contains(point, tolerance_mm=tolerance_mm)) for point in points],
                dtype=bool,
            )
        phi = 2.0 * math.pi * np.arange(n_sides, dtype=np.float64) / float(n_sides)
        normals_xz = np.column_stack((np.cos(phi), np.sin(phi)))
        xz = points[:, (0, 2)]
        valid &= np.all(xz @ normals_xz.T <= limit + tolerance_mm, axis=1)
    else:
        locations = np.asarray(
            getattr(detector, "locations", np.empty((0, 3))), dtype=np.float64
        )
        axes = np.asarray(
            getattr(detector, "inward_axes", np.empty((0, 3))), dtype=np.float64
        )
        if locations.size and axes.shape == locations.shape:
            for start in range(0, n_points, 8192):
                stop = min(start + 8192, n_points)
                block = points[start:stop]
                distances = np.einsum(
                    "bpi,pi->bp", block[:, None, :] - locations[None, :, :], axes
                )
                valid[start:stop] &= np.min(distances, axis=1) >= margin - tolerance_mm
    return valid


def _has_exclusions(detector) -> bool:
    centres = np.asarray(
        getattr(detector, "exclusion_centres_mm", np.empty((0, 3))), dtype=np.float64
    )
    axes = np.asarray(
        getattr(detector, "exclusion_axes", np.empty((0, 3))), dtype=np.float64
    )
    radius = float(getattr(detector, "exclusion_radius_mm", math.nan))
    return bool(
        centres.ndim == 2
        and centres.shape[1:] == (3,)
        and centres.size
        and axes.shape == centres.shape
        and math.isfinite(radius)
        and radius > 0.0
    )


def _contains_many(
    detector,
    points_mm: np.ndarray,
    *,
    tolerance_mm: float = 1.0e-7,
) -> np.ndarray:
    """Vectorized equivalent of ``ConvexDetectorVolume.contains``.

    The expensive dome-cap exclusion test is applied only to points that already
    pass the outer detector predicate. This matters for long cosmic tracks with
    many steps outside the detector.
    """
    points = np.asarray(points_mm, dtype=np.float64)
    valid = _contains_outer_many(detector, points, tolerance_mm=tolerance_mm)
    if not _has_exclusions(detector) or not np.any(valid):
        return valid

    centres = np.asarray(detector.exclusion_centres_mm, dtype=np.float64)
    axes = np.asarray(detector.exclusion_axes, dtype=np.float64)
    radius = float(detector.exclusion_radius_mm)
    cap_cut = float(getattr(detector, "exclusion_cap_cut_mm", math.nan))
    candidate_indices = np.flatnonzero(valid)
    radial_margin = max(0.0, 2.0 * radius * tolerance_mm)
    for start in range(0, candidate_indices.size, 4096):
        indices = candidate_indices[start : start + 4096]
        q = points[indices, None, :] - centres[None, :, :]
        radial2 = np.einsum("bpi,bpi->bp", q, q)
        axial = np.einsum("bpi,pi->bp", q, axes)
        excluded = np.any(
            (radial2 < radius * radius - radial_margin)
            & (axial > cap_cut + tolerance_mm),
            axis=1,
        )
        valid[indices] &= ~excluded
    return valid


def _first_inside_fraction(
    detector,
    pre_mm: np.ndarray,
    post_mm: np.ndarray,
    *,
    pre_inside: bool = False,
    post_inside: bool = False,
    tolerance_mm: float = 1.0e-7,
) -> float | None:
    """Return the first segment fraction inside the active volume."""
    pre = np.asarray(pre_mm, dtype=np.float64)
    post = np.asarray(post_mm, dtype=np.float64)
    if pre.shape != (3,) or post.shape != (3,):
        return None
    if not (np.all(np.isfinite(pre)) and np.all(np.isfinite(post))):
        return None
    if pre_inside or bool(detector.contains(pre, tolerance_mm=tolerance_mm)):
        return 0.0
    delta = post - pre
    segment_length = float(np.linalg.norm(delta))
    if segment_length <= 0.0:
        return None

    if post_inside or bool(detector.contains(post, tolerance_mm=tolerance_mm)):
        low = 0.0
        high = 1.0
    else:
        # Defensive rare case: a long step enters and exits while both stored
        # endpoints are outside. Locate the first outside-to-inside bracket.
        fractions = np.linspace(0.0, 1.0, 65, dtype=np.float64)
        sample = pre[None, :] + fractions[:, None] * delta[None, :]
        inside = _contains_many(detector, sample, tolerance_mm=tolerance_mm)
        indices = np.flatnonzero(inside)
        if indices.size == 0:
            return None
        first = int(indices[0])
        if first == 0:
            return 0.0
        low = float(fractions[first - 1])
        high = float(fractions[first])

    # Stop once the spatial bracket is much smaller than any detector-model
    # precision used by the fitter. The hard iteration cap is a safety bound.
    for _ in range(50):
        if (high - low) * segment_length <= 1.0e-7:
            break
        mid = 0.5 * (low + high)
        point = pre + mid * delta
        if bool(detector.contains(point, tolerance_mm=tolerance_mm)):
            high = mid
        else:
            low = mid
    return float(high)


def infer_primary_track_ids_from_npz(
    raw: Mapping[str, Any],
    source_event_ids: Sequence[int],
    *,
    fit_particle: str,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Infer unambiguous primary track IDs from the fitted NPZ when available."""
    required = ("track_id", "track_pid", "track_parent")
    if not all(name in raw for name in required):
        return {}, {
            "available": False,
            "reason": "NPZ lacks one or more of track_id, track_pid, track_parent",
            "n_unambiguous": 0,
            "n_ambiguous": 0,
        }
    target_abs_pdg = _particle_abs_pdg(fit_particle)
    mapping: dict[int, int] = {}
    ambiguous = 0
    invalid = 0
    for source_event_id in source_event_ids:
        try:
            ids = _as_1d(raw["track_id"][int(source_event_id)])
            pids = _as_1d(raw["track_pid"][int(source_event_id)])
            parents = _as_1d(raw["track_parent"][int(source_event_id)])
        except Exception:
            invalid += 1
            continue
        n = min(len(ids), len(pids), len(parents))
        if n <= 0:
            invalid += 1
            continue
        ids = ids[:n]
        pids = pids[:n]
        parents = parents[:n]
        finite = np.isfinite(ids.astype(np.float64, copy=False))
        finite &= np.isfinite(pids.astype(np.float64, copy=False))
        finite &= np.isfinite(parents.astype(np.float64, copy=False))
        mask = finite & (parents.astype(np.int64, copy=False) == 0)
        if target_abs_pdg is not None:
            mask &= np.abs(pids.astype(np.int64, copy=False)) == int(target_abs_pdg)
        candidates = sorted({int(round(float(value))) for value in ids[mask]})
        if len(candidates) == 1:
            mapping[int(source_event_id)] = int(candidates[0])
        elif len(candidates) > 1:
            ambiguous += 1
        else:
            invalid += 1
    return mapping, {
        "available": True,
        "n_unambiguous": int(len(mapping)),
        "n_ambiguous": int(ambiguous),
        "n_missing_or_invalid": int(invalid),
        "particle_abs_pdg": target_abs_pdg,
    }


def _empty_record(source_event_id: int, root_event_id: int, status: str) -> dict[str, Any]:
    return {
        "found": False,
        "source_event_index": int(source_event_id),
        "root_event_id": int(root_event_id),
        "status": str(status),
        "primary_selection_status": "",
        "primary_candidate_count": 0,
        "track_id": -1,
        "pdg": 0,
        "step": -1,
        "segment_fraction": math.nan,
        "entry_method": "",
        "x_mm": math.nan,
        "y_mm": math.nan,
        "z_mm": math.nan,
        "t_ns": math.nan,
        "ke_mev": math.nan,
        "dir_x": math.nan,
        "dir_y": math.nan,
        "dir_z": math.nan,
        "pre_material": "",
        "post_material": "",
        "pre_volume": "",
        "post_volume": "",
        "step_process": "",
        "primary_start_x_mm": math.nan,
        "primary_start_y_mm": math.nan,
        "primary_start_z_mm": math.nan,
        "primary_start_t_ns": math.nan,
        "primary_start_ke_mev": math.nan,
    }


def _row_context(chunk: Mapping[str, np.ndarray], index: int, name: str) -> str:
    values = chunk.get(name)
    if values is None or index >= len(values):
        return ""
    return _text(values[index])


def _row_value(chunk: Mapping[str, np.ndarray], index: int, name: str) -> float:
    values = chunk.get(name)
    if values is None or index >= len(values):
        return math.nan
    return _finite_float(values[index])


def _entry_candidate(
    chunk: Mapping[str, np.ndarray],
    index: int,
    detector,
    *,
    pre_inside: bool,
    post_inside: bool,
    coordinate_offset_mm: np.ndarray,
    global_entry_index: int | None = None,
) -> dict[str, Any] | None:
    pre = np.asarray(
        [chunk["x_cm"][index], chunk["y_cm"][index], chunk["z_cm"][index]],
        dtype=np.float64,
    ) * 10.0 + coordinate_offset_mm
    post = np.asarray(
        [
            chunk["post_x_cm"][index],
            chunk["post_y_cm"][index],
            chunk["post_z_cm"][index],
        ],
        dtype=np.float64,
    ) * 10.0 + coordinate_offset_mm
    if pre_inside:
        fraction = 0.0
    else:
        fraction = _first_inside_fraction(
            detector,
            pre,
            post,
            pre_inside=pre_inside,
            post_inside=post_inside,
        )
    if fraction is None:
        return None
    fraction = float(np.clip(fraction, 0.0, 1.0))
    point = pre + fraction * (post - pre)

    t0 = _row_value(chunk, index, "t_ns")
    t1 = _row_value(chunk, index, "post_t_ns")
    ke0 = _row_value(chunk, index, "ke_MeV")
    ke1 = _row_value(chunk, index, "post_ke_MeV")

    def interpolate_scalar(pre_value: float, post_value: float) -> float:
        if math.isfinite(pre_value) and math.isfinite(post_value):
            return float(pre_value + fraction * (post_value - pre_value))
        if math.isfinite(pre_value):
            return float(pre_value)
        if math.isfinite(post_value):
            return float(post_value)
        return math.nan

    entry_t = interpolate_scalar(t0, t1)
    entry_ke = interpolate_scalar(ke0, ke1)

    pre_direction = np.asarray(
        [
            _row_value(chunk, index, "dir_x"),
            _row_value(chunk, index, "dir_y"),
            _row_value(chunk, index, "dir_z"),
        ],
        dtype=np.float64,
    )
    post_direction = np.asarray(
        [
            _row_value(chunk, index, "post_dir_x"),
            _row_value(chunk, index, "post_dir_y"),
            _row_value(chunk, index, "post_dir_z"),
        ],
        dtype=np.float64,
    )
    if np.all(np.isfinite(pre_direction)) and np.all(np.isfinite(post_direction)):
        direction = (1.0 - fraction) * pre_direction + fraction * post_direction
    elif np.all(np.isfinite(post_direction)):
        direction = post_direction
    elif np.all(np.isfinite(pre_direction)):
        direction = pre_direction
    else:
        # The post-minus-pre Geant4 step tangent is already present in the
        # required numeric pass, so the default fast mode can report a local
        # entry direction without reading six additional direction branches.
        direction = post - pre
    dir_x, dir_y, dir_z = _normalise_direction(direction)

    step_values = chunk.get("step")
    step = _finite_int(step_values[index], -1) if step_values is not None else -1
    candidate = {
        "found": True,
        "status": "ok_boundary_crossing" if fraction > 1.0e-10 else "ok_first_inside_step",
        "entry_method": (
            "active_geometry_segment_bisection"
            if fraction > 1.0e-10
            else "active_geometry_first_inside_pre_step"
        ),
        "step": int(step),
        "segment_fraction": float(fraction),
        "x_mm": float(point[0]),
        "y_mm": float(point[1]),
        "z_mm": float(point[2]),
        "t_ns": float(entry_t),
        "ke_mev": float(entry_ke),
        "dir_x": dir_x,
        "dir_y": dir_y,
        "dir_z": dir_z,
        "pre_material": _row_context(chunk, index, "material"),
        "post_material": _row_context(chunk, index, "post_material"),
        "pre_volume": _row_context(chunk, index, "volume"),
        "post_volume": _row_context(chunk, index, "post_volume"),
        "step_process": _row_context(chunk, index, "step_process"),
        "_candidate_order": (
            float(entry_t) if math.isfinite(entry_t) else math.inf,
            int(step),
            float(fraction),
            int(global_entry_index if global_entry_index is not None else index),
        ),
        "_global_entry_index": (
            None if global_entry_index is None else int(global_entry_index)
        ),
    }
    return candidate


def _track_rank(state: Mapping[str, Any]) -> tuple[float, float, float, int]:
    """Deterministic fallback rank for multiple parent-zero particle tracks."""
    entry = state.get("entry")
    entry_time = _finite_float(entry.get("t_ns")) if isinstance(entry, Mapping) else math.inf
    if not math.isfinite(entry_time):
        entry_time = math.inf
    earliest = _finite_float(state.get("earliest_t_ns"), math.inf)
    if not math.isfinite(earliest):
        earliest = math.inf
    max_ke = _finite_float(state.get("max_ke_mev"), -math.inf)
    energy_rank = -max_ke if math.isfinite(max_ke) else math.inf
    return (entry_time, earliest, energy_rank, -int(state.get("row_count", 0)))


def _particle_mask_from_names(values: np.ndarray, target_abs_pdg: int | None, fit_particle: str) -> np.ndarray:
    if target_abs_pdg is None:
        return np.ones(len(values), dtype=bool)
    wanted = str(fit_particle).lower()
    output = np.empty(len(values), dtype=bool)
    for index, value in enumerate(values):
        text = _text(value).strip().lower()
        if target_abs_pdg == 13:
            output[index] = "mu" in text
        elif target_abs_pdg == 11:
            output[index] = text.startswith("e") or "electron" in text or "positron" in text
        else:
            output[index] = wanted in text
    return output


def _event_mask(evt: np.ndarray, target_root_ids: np.ndarray) -> np.ndarray:
    if target_root_ids.size == 0:
        return np.zeros(evt.size, dtype=bool)
    lo = int(target_root_ids[0])
    hi = int(target_root_ids[-1])
    if target_root_ids.size == hi - lo + 1:
        return (evt >= lo) & (evt <= hi)
    return np.isin(evt, target_root_ids, assume_unique=False)


def _ordered_group_candidate(
    *,
    chunk: Mapping[str, np.ndarray],
    row_indices: np.ndarray,
    selected_positions: np.ndarray,
    pre_points: np.ndarray,
    post_points: np.ndarray,
    pre_outer: np.ndarray,
    post_outer: np.ndarray,
    segment_may_cross_bounds: np.ndarray,
    detector,
    coordinate_offset: np.ndarray,
    chunk_entry_start: int,
    metrics: dict[str, int],
) -> dict[str, Any] | None:
    """Find the first active crossing for one contiguous track segment.

    Full dome-exclusion geometry is evaluated in chronological blocks and stops
    as soon as a crossing is found. Within each block the first endpoint that is
    active is selected vectorially, so later excluded/inside steps do not create
    a Python loop proportional to track length.
    """
    if row_indices.size == 0:
        return None
    times = _as_1d(chunk["t_ns"], dtype=np.float64)[row_indices]
    step_values = chunk.get("step")
    if step_values is None:
        steps = np.arange(row_indices.size, dtype=np.int64)
    else:
        steps = _as_1d(step_values, dtype=np.int64)[row_indices]
    order_times = np.where(np.isfinite(times), times, math.inf)
    order = np.lexsort((steps, order_times))

    broad = (
        pre_outer[selected_positions]
        | post_outer[selected_positions]
        | segment_may_cross_bounds[selected_positions]
    )
    ordered_positions = order[broad[order]]
    if ordered_positions.size == 0:
        return None

    has_exclusions = _has_exclusions(detector)
    block_size = 1024 if has_exclusions else max(1, int(ordered_positions.size))
    for block_start in range(0, ordered_positions.size, block_size):
        block_pos = ordered_positions[block_start : block_start + block_size]
        global_selected_pos = selected_positions[block_pos]
        if has_exclusions:
            block_pre_inside = _contains_many(detector, pre_points[global_selected_pos])
            block_post_inside = _contains_many(detector, post_points[global_selected_pos])
            metrics["full_geometry_points_tested"] += int(2 * block_pos.size)
        else:
            block_pre_inside = pre_outer[global_selected_pos]
            block_post_inside = post_outer[global_selected_pos]

        direct = block_pre_inside | block_post_inside
        direct_locations = np.flatnonzero(direct)
        if direct_locations.size:
            local_in_block = int(direct_locations[0])
            group_pos = int(block_pos[local_in_block])
            row_index = int(row_indices[group_pos])
            metrics["entry_candidate_rows_tested"] += 1
            return _entry_candidate(
                chunk,
                row_index,
                detector,
                pre_inside=bool(block_pre_inside[local_in_block]),
                post_inside=bool(block_post_inside[local_in_block]),
                coordinate_offset_mm=coordinate_offset,
                global_entry_index=int(chunk_entry_start + row_index),
            )

        # Defensive support for a genuinely long step that enters and exits the
        # active volume while both endpoints are outside. Geant4 normally ends a
        # step at every geometry boundary, so short both-outside steps cannot be
        # the entry and are skipped without a 65-point scan.
        deltas = post_points[global_selected_pos] - pre_points[global_selected_pos]
        lengths = np.linalg.norm(deltas, axis=1)
        rare = (
            segment_may_cross_bounds[global_selected_pos]
            & np.isfinite(lengths)
            & (lengths >= 1.0)
        )
        for local_in_block in np.flatnonzero(rare):
            group_pos = int(block_pos[int(local_in_block)])
            row_index = int(row_indices[group_pos])
            metrics["entry_candidate_rows_tested"] += 1
            candidate = _entry_candidate(
                chunk,
                row_index,
                detector,
                pre_inside=False,
                post_inside=False,
                coordinate_offset_mm=coordinate_offset,
                global_entry_index=int(chunk_entry_start + row_index),
            )
            if candidate is not None:
                return candidate
    return None


def extract_active_water_entries_from_chunks(
    chunks: Iterable[Mapping[str, Any] | tuple[int, Mapping[str, Any]]],
    *,
    detector,
    source_event_ids: Sequence[int],
    fit_particle: str,
    event_id_offset: int = 0,
    expected_track_ids: Mapping[int, int] | None = None,
    configured_track_id: int | None = None,
    coordinate_offset_mm: Sequence[float] = (0.0, 0.0, 0.0),
    keep_private_entry_indices: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Extract first active-water entries from already materialized chunks.

    Each element may be a branch mapping or ``(global_entry_start, mapping)``.
    The latter form lets the ROOT wrapper sparsely fetch optional details for the
    selected crossing rows after the numeric pass.
    """
    wall0 = time.perf_counter()
    source_ids = [int(value) for value in source_event_ids]
    root_to_source = {
        int(source + int(event_id_offset)): int(source) for source in source_ids
    }
    target_root_ids = np.asarray(sorted(root_to_source), dtype=np.int64)
    target_abs_pdg = _particle_abs_pdg(fit_particle)
    expected_track_ids = {
        int(key): int(value) for key, value in dict(expected_track_ids or {}).items()
    }
    coordinate_offset = np.asarray(coordinate_offset_mm, dtype=np.float64)
    if coordinate_offset.shape != (3,) or not np.all(np.isfinite(coordinate_offset)):
        raise ValueError("coordinate_offset_mm must contain three finite values")

    tracks: dict[tuple[int, int], dict[str, Any]] = {}
    tracks_by_source: dict[int, set[int]] = defaultdict(set)
    seen_events: set[int] = set()
    rows_scanned = 0
    target_rows = 0
    primary_rows = 0
    running_entry_start = 0
    metrics = {
        "full_geometry_points_tested": 0,
        "entry_candidate_rows_tested": 0,
        "track_segments_processed": 0,
    }

    for raw_item in chunks:
        if isinstance(raw_item, tuple) and len(raw_item) == 2:
            chunk_entry_start = int(raw_item[0])
            raw_chunk = raw_item[1]
        else:
            chunk_entry_start = int(running_entry_start)
            raw_chunk = raw_item
        if not isinstance(raw_chunk, Mapping) or "evt" not in raw_chunk:
            raise ValueError("Each AllSecondaries chunk must be a mapping containing evt")
        chunk = {name: _as_1d(values) for name, values in raw_chunk.items()}
        lengths = {len(values) for values in chunk.values()}
        if len(lengths) != 1:
            raise ValueError("All branch arrays in an AllSecondaries chunk must have equal length")
        n_rows = int(next(iter(lengths), 0))
        running_entry_start = chunk_entry_start + n_rows
        rows_scanned += n_rows
        if n_rows == 0:
            continue

        evt = _as_1d(chunk["evt"], dtype=np.int64)
        event_mask = _event_mask(evt, target_root_ids)
        if not np.any(event_mask):
            continue
        target_rows += int(np.count_nonzero(event_mask))
        seen_events.update(int(value) for value in np.unique(evt[event_mask]))

        parent = _as_1d(chunk["parent"], dtype=np.int64)
        trk = _as_1d(chunk["trk"], dtype=np.int64)
        mask = event_mask & (parent == 0)
        if "pdg" in chunk and target_abs_pdg is not None:
            pdg_values = _as_1d(chunk["pdg"], dtype=np.int64)
            mask &= np.abs(pdg_values) == int(target_abs_pdg)
        elif "particle" in chunk and target_abs_pdg is not None:
            mask &= _particle_mask_from_names(chunk["particle"], target_abs_pdg, fit_particle)
        if configured_track_id is not None:
            mask &= trk == int(configured_track_id)
        selected = np.flatnonzero(mask)
        primary_rows += int(selected.size)
        if selected.size == 0:
            continue

        pre_points = np.column_stack(
            (chunk["x_cm"][selected], chunk["y_cm"][selected], chunk["z_cm"][selected])
        ).astype(np.float64, copy=False) * 10.0 + coordinate_offset[None, :]
        post_points = np.column_stack(
            (
                chunk["post_x_cm"][selected],
                chunk["post_y_cm"][selected],
                chunk["post_z_cm"][selected],
            )
        ).astype(np.float64, copy=False) * 10.0 + coordinate_offset[None, :]
        pre_outer = _contains_outer_many(detector, pre_points)
        post_outer = _contains_outer_many(detector, post_points)
        axis_lo = np.asarray(getattr(detector, "axis_lo", [-math.inf] * 3), dtype=np.float64)
        axis_hi = np.asarray(getattr(detector, "axis_hi", [math.inf] * 3), dtype=np.float64)
        segment_lo = np.minimum(pre_points, post_points)
        segment_hi = np.maximum(pre_points, post_points)
        segment_may_cross_bounds = np.all(segment_hi >= axis_lo[None, :], axis=1)
        segment_may_cross_bounds &= np.all(segment_lo <= axis_hi[None, :], axis=1)

        selected_evt = evt[selected]
        selected_trk = trk[selected]
        group_change = np.empty(selected.size, dtype=bool)
        group_change[0] = True
        if selected.size > 1:
            group_change[1:] = (
                (selected_evt[1:] != selected_evt[:-1])
                | (selected_trk[1:] != selected_trk[:-1])
            )
        group_starts = np.flatnonzero(group_change)
        group_stops = np.append(group_starts[1:], selected.size)

        t_values = _as_1d(chunk["t_ns"], dtype=np.float64)
        ke_values = _as_1d(chunk["ke_MeV"], dtype=np.float64)
        step_values = (
            _as_1d(chunk["step"], dtype=np.int64)
            if "step" in chunk
            else np.full(n_rows, -1, dtype=np.int64)
        )
        pdg_values = (
            _as_1d(chunk["pdg"], dtype=np.int64)
            if "pdg" in chunk
            else np.zeros(n_rows, dtype=np.int64)
        )

        for group_start, group_stop in zip(group_starts, group_stops):
            metrics["track_segments_processed"] += 1
            positions = np.arange(group_start, group_stop, dtype=np.int64)
            rows = selected[positions]
            root_event_id = int(selected_evt[group_start])
            source_event_id = int(root_to_source[root_event_id])
            track_id = int(selected_trk[group_start])
            key = (source_event_id, track_id)
            state = tracks.get(key)
            if state is None:
                state = {
                    "source_event_index": source_event_id,
                    "root_event_id": root_event_id,
                    "track_id": track_id,
                    "pdg": _finite_int(pdg_values[rows[0]], 0),
                    "row_count": 0,
                    "earliest_t_ns": math.inf,
                    "max_ke_mev": -math.inf,
                    "first_row_order": (math.inf, math.inf, math.inf),
                    "first_row": None,
                    "entry": None,
                }
                tracks[key] = state
                tracks_by_source[source_event_id].add(track_id)

            state["row_count"] = int(state["row_count"]) + int(rows.size)
            group_t = t_values[rows]
            finite_t = group_t[np.isfinite(group_t)]
            if finite_t.size:
                state["earliest_t_ns"] = min(
                    float(state["earliest_t_ns"]), float(np.min(finite_t))
                )
            group_ke = ke_values[rows]
            finite_ke = group_ke[np.isfinite(group_ke)]
            if finite_ke.size:
                state["max_ke_mev"] = max(
                    float(state["max_ke_mev"]), float(np.max(finite_ke))
                )

            group_steps = step_values[rows]
            order_t = np.where(np.isfinite(group_t), group_t, math.inf)
            first_local = int(np.lexsort((rows, group_steps, order_t))[0])
            first_row_index = int(rows[first_local])
            first_order = (
                float(order_t[first_local]),
                int(group_steps[first_local]),
                int(chunk_entry_start + first_row_index),
            )
            if first_order < tuple(state["first_row_order"]):
                start_direction = _normalise_direction(
                    [
                        _row_value(chunk, first_row_index, "dir_x"),
                        _row_value(chunk, first_row_index, "dir_y"),
                        _row_value(chunk, first_row_index, "dir_z"),
                    ]
                )
                state["first_row_order"] = first_order
                state["first_row"] = {
                    "step": int(group_steps[first_local]),
                    "t_ns": float(group_t[first_local]),
                    "ke_mev": float(group_ke[first_local]),
                    "x_mm": 10.0 * _row_value(chunk, first_row_index, "x_cm")
                    + float(coordinate_offset[0]),
                    "y_mm": 10.0 * _row_value(chunk, first_row_index, "y_cm")
                    + float(coordinate_offset[1]),
                    "z_mm": 10.0 * _row_value(chunk, first_row_index, "z_cm")
                    + float(coordinate_offset[2]),
                    "dir_x": start_direction[0],
                    "dir_y": start_direction[1],
                    "dir_z": start_direction[2],
                }

            existing = state.get("entry")
            earliest_group_t = float(np.min(order_t)) if order_t.size else math.inf
            if isinstance(existing, Mapping):
                existing_t = _finite_float(existing.get("t_ns"), math.inf)
                if earliest_group_t >= existing_t:
                    continue

            candidate = _ordered_group_candidate(
                chunk=chunk,
                row_indices=rows,
                selected_positions=positions,
                pre_points=pre_points,
                post_points=post_points,
                pre_outer=pre_outer,
                post_outer=post_outer,
                segment_may_cross_bounds=segment_may_cross_bounds,
                detector=detector,
                coordinate_offset=coordinate_offset,
                chunk_entry_start=chunk_entry_start,
                metrics=metrics,
            )
            if candidate is not None and (
                existing is None
                or tuple(candidate["_candidate_order"]) < tuple(existing["_candidate_order"])
            ):
                state["entry"] = candidate

    records: dict[int, dict[str, Any]] = {}
    selection_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for source_event_id in source_ids:
        root_event_id = int(source_event_id + int(event_id_offset))
        record = _empty_record(source_event_id, root_event_id, "missing_root_event")
        if root_event_id not in seen_events:
            records[source_event_id] = record
            status_counts[record["status"]] += 1
            continue

        track_ids = sorted(tracks_by_source.get(source_event_id, set()))
        states = [tracks[(source_event_id, track_id)] for track_id in track_ids]
        record["primary_candidate_count"] = int(len(states))
        if not states:
            record["status"] = "no_matching_primary_track"
            records[source_event_id] = record
            status_counts[record["status"]] += 1
            continue

        expected = None
        expected_source = ""
        if configured_track_id is not None:
            expected = int(configured_track_id)
            expected_source = "configured_track_id"
        elif source_event_id in expected_track_ids:
            expected = int(expected_track_ids[source_event_id])
            expected_source = "npz_primary_track_id"

        selected_state = None
        if expected is not None:
            selected_state = next(
                (state for state in states if int(state["track_id"]) == expected),
                None,
            )
        if selected_state is not None:
            selection_status = expected_source
        else:
            states_with_entry = [state for state in states if state.get("entry") is not None]
            ranked = states_with_entry if states_with_entry else states
            selected_state = sorted(
                ranked,
                key=lambda state: (_track_rank(state), int(state["track_id"])),
            )[0]
            if expected is not None:
                selection_status = "expected_track_missing_auto_fallback"
            elif len(states) == 1:
                selection_status = "auto_unique_primary"
            else:
                selection_status = "auto_ranked_multiple_primaries"
        selection_counts[selection_status] += 1

        record["primary_selection_status"] = selection_status
        record["track_id"] = int(selected_state["track_id"])
        record["pdg"] = int(selected_state.get("pdg", 0))
        first_row = selected_state.get("first_row")
        if isinstance(first_row, Mapping):
            record.update(
                {
                    "primary_start_x_mm": _finite_float(first_row.get("x_mm")),
                    "primary_start_y_mm": _finite_float(first_row.get("y_mm")),
                    "primary_start_z_mm": _finite_float(first_row.get("z_mm")),
                    "primary_start_t_ns": _finite_float(first_row.get("t_ns")),
                    "primary_start_ke_mev": _finite_float(first_row.get("ke_mev")),
                }
            )

        entry = selected_state.get("entry")
        if not isinstance(entry, Mapping):
            record["status"] = "primary_never_enters_active_water"
            records[source_event_id] = record
            status_counts[record["status"]] += 1
            continue
        for key, value in entry.items():
            if keep_private_entry_indices or not str(key).startswith("_"):
                record[key] = value
        if (
            float(record.get("segment_fraction", math.nan)) <= 1.0e-10
            and isinstance(first_row, Mapping)
            and int(record.get("step", -1)) == int(first_row.get("step", -2))
            and abs(_finite_float(record.get("t_ns")) - _finite_float(first_row.get("t_ns")))
            < 1.0e-8
        ):
            record["status"] = "ok_started_inside_active_water"
            record["entry_method"] = "active_geometry_primary_start_inside"
        records[source_event_id] = record
        status_counts[str(record["status"])] += 1

    metadata = {
        "rows_scanned": int(rows_scanned),
        "rows_in_requested_events": int(target_rows),
        "matching_primary_rows": int(primary_rows),
        "n_requested_events": int(len(source_ids)),
        "n_root_events_seen": int(len(seen_events)),
        "n_entries_found": int(sum(bool(record.get("found")) for record in records.values())),
        "status_counts": dict(sorted(status_counts.items())),
        "primary_selection_counts": dict(sorted(selection_counts.items())),
        "particle": str(fit_particle),
        "particle_abs_pdg": target_abs_pdg,
        "event_id_offset": int(event_id_offset),
        "configured_track_id": None if configured_track_id is None else int(configured_track_id),
        "event_alignment": "AllSecondaries.evt = source_event_index + event_id_offset",
        "entry_definition": "first crossing into detector active-water predicate",
        "position_input_units": "WCSim cm",
        "position_output_units": "LicketyFit detector mm",
        "coordinate_offset_mm": coordinate_offset.tolist(),
        "coordinate_transform": "x_detector = 10*x_wcsim_cm + coordinate_offset_mm",
        "energy_units": "MeV kinetic energy",
        "time_units": "ns",
        "interpolation": "linear in pre/post step position, time, KE, and direction",
        "full_geometry_points_tested": int(metrics["full_geometry_points_tested"]),
        "entry_candidate_rows_tested": int(metrics["entry_candidate_rows_tested"]),
        "track_segments_processed": int(metrics["track_segments_processed"]),
        "processing_wall_s": float(time.perf_counter() - wall0),
    }
    return records, metadata


def _import_uproot():
    try:
        import uproot  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "WCSIM_USE_TRUTH_ROOT=1 requires the optional runtime dependency "
            "'uproot'. Install the packaged requirements.txt in the same Python "
            "environment used to run batch_fit_driver.py."
        ) from error
    return uproot


def _resolve_tree(root_file, requested_name: str):
    requested = str(requested_name or DEFAULT_TREE_NAME).strip()
    try:
        tree = root_file[requested]
        return tree, requested
    except Exception:
        pass

    classnames = {}
    try:
        classnames = dict(root_file.classnames(recursive=True))
    except Exception:
        try:
            classnames = {key: "" for key in root_file.keys(recursive=True)}
        except Exception:
            classnames = {}
    requested_norm = _normalize_name(Path(requested).name.split(";")[0])
    exact: list[str] = []
    fuzzy: list[str] = []
    for raw_key, classname in classnames.items():
        key = str(raw_key)
        base = Path(key.split(";")[0]).name
        normalized = _normalize_name(base)
        class_text = str(classname)
        if class_text and not (class_text.startswith("TTree") or "RNTuple" in class_text):
            continue
        if normalized == requested_norm:
            exact.append(key)
        elif "allsecondar" in normalized:
            fuzzy.append(key)
    candidates = exact if exact else fuzzy
    if len(candidates) != 1:
        available = ", ".join(sorted(classnames)[:30])
        raise KeyError(
            f"Could not resolve one TTree matching {requested!r}. "
            f"Candidates={candidates!r}; available keys include: {available}"
        )
    key = candidates[0]
    return root_file[key], key


def _resolve_branches(tree) -> tuple[dict[str, str], list[str], list[str]]:
    try:
        available = [str(name).split(";")[0] for name in tree.keys(recursive=False)]
    except TypeError:
        available = [str(name).split(";")[0] for name in tree.keys()]
    by_norm: dict[str, list[str]] = defaultdict(list)
    for name in available:
        by_norm[_normalize_name(Path(name).name)].append(name)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical in _REQUIRED_BRANCHES + _OPTIONAL_BRANCHES:
        matches = by_norm.get(_normalize_name(canonical), [])
        if len(matches) == 1:
            resolved[canonical] = matches[0]
        elif canonical in _REQUIRED_BRANCHES:
            missing.append(canonical)
    if "pdg" not in resolved and "particle" not in resolved:
        missing.append("pdg or particle")
    if missing:
        raise KeyError(
            "AllSecondaries is missing required per-step branches: " + ", ".join(missing)
        )
    optional_missing = [name for name in _OPTIONAL_BRANCHES if name not in resolved]
    return resolved, missing, optional_missing


@contextmanager
def _uproot_executors(n_workers: int):
    workers = max(1, int(n_workers))
    if workers <= 1:
        yield None, None
        return
    decompression = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lfit-root-decomp")
    interpretation = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lfit-root-interpret")
    try:
        yield decompression, interpretation
    finally:
        decompression.shutdown(wait=True, cancel_futures=False)
        interpretation.shutdown(wait=True, cancel_futures=False)


def _open_uproot(uproot, path: Path, *, n_workers: int):
    kwargs = {
        "array_cache": None,
        "num_workers": max(1, int(n_workers)),
        "use_threads": bool(int(n_workers) > 1),
    }
    try:
        return uproot.open(path, **kwargs)
    except TypeError:
        try:
            return uproot.open(path, array_cache=None)
        except TypeError:
            return uproot.open(path)


def _mapping_from_uproot(raw_chunk: Any) -> Mapping[str, Any]:
    if isinstance(raw_chunk, np.ndarray) and raw_chunk.dtype.names:
        return {name: raw_chunk[name] for name in raw_chunk.dtype.names}
    if not isinstance(raw_chunk, Mapping):
        raise TypeError("uproot did not return a branch mapping for AllSecondaries")
    return raw_chunk


def _tree_iterate(
    tree,
    *,
    expressions: Sequence[str],
    entry_start: int,
    entry_stop: int,
    step_size: str | int,
    decompression_executor=None,
    interpretation_executor=None,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    kwargs = {
        "expressions": list(expressions),
        "entry_start": int(entry_start),
        "entry_stop": int(entry_stop),
        "step_size": step_size,
        "library": "np",
        "how": dict,
        "decompression_executor": decompression_executor,
        "interpretation_executor": interpretation_executor,
    }
    try:
        iterator = tree.iterate(**kwargs)
    except TypeError:
        kwargs.pop("decompression_executor", None)
        kwargs.pop("interpretation_executor", None)
        try:
            iterator = tree.iterate(**kwargs)
        except TypeError:
            # Minimal test doubles and older uproot releases may not expose all
            # keyword arguments. Keep the same semantics where possible.
            kwargs.pop("entry_start", None)
            kwargs.pop("entry_stop", None)
            iterator = tree.iterate(**kwargs)
    current = int(entry_start)
    for raw_chunk in iterator:
        mapping = _mapping_from_uproot(raw_chunk)
        first = next(iter(mapping.values()), np.empty(0))
        n_rows = int(len(_as_1d(first)))
        yield current, mapping
        current += n_rows
        if current >= int(entry_stop):
            break


def _branch_array(
    branch,
    *,
    entry_start: int,
    entry_stop: int,
    decompression_executor=None,
    interpretation_executor=None,
) -> np.ndarray:
    kwargs = {
        "entry_start": int(entry_start),
        "entry_stop": int(entry_stop),
        "library": "np",
        "array_cache": None,
        "decompression_executor": decompression_executor,
        "interpretation_executor": interpretation_executor,
    }
    try:
        return _as_1d(branch.array(**kwargs))
    except TypeError:
        kwargs.pop("array_cache", None)
        kwargs.pop("decompression_executor", None)
        kwargs.pop("interpretation_executor", None)
        return _as_1d(branch.array(**kwargs))


def _cache_directory(explicit: str | Path | None) -> Path:
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser()
    env = os.environ.get("WCSIM_TRUTH_INDEX_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME", "").strip()
    if base:
        return Path(base).expanduser() / "licketyfit" / "wcsim_truth"
    try:
        return Path.home() / ".cache" / "licketyfit" / "wcsim_truth"
    except Exception:
        return Path(tempfile.gettempdir()) / "licketyfit-wcsim-truth"


def _event_index_signature(
    *,
    root_path: Path,
    stat,
    tree_resolved: str,
    event_branch: str,
    num_entries: int,
) -> dict[str, Any]:
    return {
        "format_version": _EVENT_INDEX_FORMAT_VERSION,
        "root_path": str(root_path),
        "root_size_bytes": int(stat.st_size),
        "root_mtime_ns": int(stat.st_mtime_ns),
        "tree": str(tree_resolved),
        "event_branch": str(event_branch),
        "num_entries": int(num_entries),
    }


def _event_index_cache_path(cache_dir: Path, signature: Mapping[str, Any]) -> Path:
    token = hashlib.sha256(
        (str(signature["root_path"]) + "\0" + str(signature["tree"]) + "\0" + str(signature["event_branch"])).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return cache_dir / f"allsecondaries_evt_index_{token}.npz"


def _empty_event_index(*, lookup_mode: str = "stream_scan") -> dict[str, Any]:
    return {
        "event_ids": np.empty(0, dtype=np.int64),
        "entry_starts": np.empty(0, dtype=np.int64),
        "entry_stops": np.empty(0, dtype=np.int64),
        "scan_entry_stop": 0,
        "complete": False,
        "monotonic": True,
        "lookup_mode": str(lookup_mode),
        "missing_event_ids": np.empty(0, dtype=np.int64),
    }


def _load_event_index(path: Path, signature: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            stored_signature = json.loads(str(data["signature_json"].item()))
            if stored_signature != dict(signature):
                return None
            result = {
                "event_ids": _as_1d(data["event_ids"], dtype=np.int64),
                "entry_starts": _as_1d(data["entry_starts"], dtype=np.int64),
                "entry_stops": _as_1d(data["entry_stops"], dtype=np.int64),
                "scan_entry_stop": int(data["scan_entry_stop"].item()),
                "complete": bool(int(data["complete"].item())),
                "monotonic": bool(int(data["monotonic"].item())),
                "lookup_mode": (
                    str(data["lookup_mode"].item())
                    if "lookup_mode" in data.files
                    else "stream_scan"
                ),
                "missing_event_ids": (
                    _as_1d(data["missing_event_ids"], dtype=np.int64)
                    if "missing_event_ids" in data.files
                    else np.empty(0, dtype=np.int64)
                ),
            }
    except Exception:
        return None
    n = len(result["event_ids"])
    if not (len(result["entry_starts"]) == n == len(result["entry_stops"])):
        return None
    if n and (
        np.any(result["entry_starts"] < 0)
        or np.any(result["entry_stops"] <= result["entry_starts"])
        or np.any(result["event_ids"][1:] <= result["event_ids"][:-1])
    ):
        return None
    return result


def _save_event_index_atomic(path: Path, signature: Mapping[str, Any], index: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with open(temporary, "wb") as stream:
            np.savez(
                stream,
                signature_json=np.asarray(json.dumps(dict(signature), sort_keys=True)),
                event_ids=np.asarray(index["event_ids"], dtype=np.int64),
                entry_starts=np.asarray(index["entry_starts"], dtype=np.int64),
                entry_stops=np.asarray(index["entry_stops"], dtype=np.int64),
                scan_entry_stop=np.asarray(int(index["scan_entry_stop"]), dtype=np.int64),
                complete=np.asarray(int(bool(index["complete"])), dtype=np.int8),
                monotonic=np.asarray(int(bool(index["monotonic"])), dtype=np.int8),
                lookup_mode=np.asarray(str(index.get("lookup_mode", "stream_scan"))),
                missing_event_ids=np.asarray(
                    index.get("missing_event_ids", []), dtype=np.int64
                ),
            )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass


@contextmanager
def _event_index_lock(path: Path, *, timeout_s: float = 300.0):
    lock_path = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    acquired = False
    while not acquired:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                stream.write(f"pid={os.getpid()} time_ns={time.time_ns()}\n")
            acquired = True
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > max(600.0, timeout_s * 2.0):
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for WCSim truth event-index lock: {lock_path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def _extend_monotonic_event_index(
    tree,
    *,
    event_branch: str,
    index: Mapping[str, Any],
    target_max_event: int,
    num_entries: int,
    step_size: str | int,
    decompression_executor=None,
    interpretation_executor=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    event_ids = list(np.asarray(index["event_ids"], dtype=np.int64))
    entry_starts = list(np.asarray(index["entry_starts"], dtype=np.int64))
    entry_stops = list(np.asarray(index["entry_stops"], dtype=np.int64))
    scan_start = int(index.get("scan_entry_stop", 0))
    complete = bool(index.get("complete", False))
    monotonic = bool(index.get("monotonic", True))
    before_entries = scan_start
    before_events = len(event_ids)
    if complete or not monotonic:
        return dict(index), {
            "entries_scanned": 0,
            "events_added": 0,
            "stopped_after_target": False,
        }

    current_event: int | None = None
    current_start = scan_start
    previous_event = int(event_ids[-1]) if event_ids else None
    stopped_after_target = False
    reached_eof = True

    for chunk_start, mapping in _tree_iterate(
        tree,
        expressions=[event_branch],
        entry_start=scan_start,
        entry_stop=num_entries,
        step_size=step_size,
        decompression_executor=decompression_executor,
        interpretation_executor=interpretation_executor,
    ):
        values = _as_1d(mapping[event_branch], dtype=np.int64)
        if values.size == 0:
            continue
        reached_eof = int(chunk_start + values.size) >= int(num_entries)
        run_starts = np.flatnonzero(np.r_[True, values[1:] != values[:-1]])
        run_stops = np.append(run_starts[1:], values.size)
        for local_start, local_stop in zip(run_starts, run_stops):
            event_id = int(values[local_start])
            global_start = int(chunk_start + local_start)
            if current_event is None:
                current_event = event_id
                current_start = global_start
                if previous_event is not None and event_id < previous_event:
                    monotonic = False
                    break
                if event_id > int(target_max_event):
                    stopped_after_target = True
                    reached_eof = False
                    current_event = None
                    current_start = global_start
                    break
                continue
            if event_id == current_event:
                continue
            if event_id < current_event:
                monotonic = False
                break
            event_ids.append(int(current_event))
            entry_starts.append(int(current_start))
            entry_stops.append(int(global_start))
            previous_event = int(current_event)
            current_event = event_id
            current_start = global_start
            if event_id > int(target_max_event):
                stopped_after_target = True
                reached_eof = False
                current_event = None
                break
        if not monotonic or stopped_after_target:
            break
        scan_start = int(chunk_start + values.size)

    if monotonic and not stopped_after_target:
        if current_event is not None and reached_eof:
            event_ids.append(int(current_event))
            entry_starts.append(int(current_start))
            entry_stops.append(int(num_entries))
            scan_start = int(num_entries)
            complete = True
        elif current_event is not None:
            # The iterator ended unexpectedly before EOF. Do not persist an
            # incomplete final event range; resume at its start next time.
            scan_start = int(current_start)
    elif stopped_after_target:
        scan_start = int(current_start)

    result = {
        "event_ids": np.asarray(event_ids, dtype=np.int64),
        "entry_starts": np.asarray(entry_starts, dtype=np.int64),
        "entry_stops": np.asarray(entry_stops, dtype=np.int64),
        "scan_entry_stop": int(scan_start),
        "complete": bool(complete),
        "monotonic": bool(monotonic),
        "lookup_mode": "stream_scan",
        "missing_event_ids": np.empty(0, dtype=np.int64),
    }
    return result, {
        "entries_scanned": int(max(0, int(scan_start) - before_entries)),
        "events_added": int(len(event_ids) - before_events),
        "stopped_after_target": bool(stopped_after_target),
    }


def _coalesce_ranges(ranges: Sequence[tuple[int, int]], *, max_gap: int = 0) -> list[tuple[int, int]]:
    cleaned = sorted((int(a), int(b)) for a, b in ranges if int(b) > int(a))
    if not cleaned:
        return []
    output = [cleaned[0]]
    for start, stop in cleaned[1:]:
        previous_start, previous_stop = output[-1]
        if start <= previous_stop + int(max_gap):
            output[-1] = (previous_start, max(previous_stop, stop))
        else:
            output.append((start, stop))
    return output



def _contiguous_event_runs(event_ids: np.ndarray) -> list[np.ndarray]:
    values = np.unique(np.asarray(event_ids, dtype=np.int64))
    if values.size == 0:
        return []
    split = np.flatnonzero(values[1:] != values[:-1] + 1) + 1
    return [block for block in np.split(values, split) if block.size]


def _merge_sparse_event_ranges(
    index: Mapping[str, Any],
    ranges_by_event: Mapping[int, tuple[int, int]],
    *,
    attempted_event_ids: Sequence[int] = (),
) -> dict[str, Any]:
    merged = {
        int(event): (int(start), int(stop))
        for event, start, stop in zip(
            np.asarray(index.get("event_ids", []), dtype=np.int64),
            np.asarray(index.get("entry_starts", []), dtype=np.int64),
            np.asarray(index.get("entry_stops", []), dtype=np.int64),
        )
    }
    merged.update(
        {
            int(event): (int(bounds[0]), int(bounds[1]))
            for event, bounds in ranges_by_event.items()
            if int(bounds[1]) > int(bounds[0])
        }
    )
    events = np.asarray(sorted(merged), dtype=np.int64)
    known_missing = set(
        int(value)
        for value in np.asarray(index.get("missing_event_ids", []), dtype=np.int64)
    )
    attempted = {int(value) for value in attempted_event_ids}
    known_missing.update(attempted.difference(merged))
    known_missing.difference_update(merged)
    return {
        "event_ids": events,
        "entry_starts": np.asarray([merged[int(event)][0] for event in events], dtype=np.int64),
        "entry_stops": np.asarray([merged[int(event)][1] for event in events], dtype=np.int64),
        "scan_entry_stop": 0,
        "complete": False,
        "monotonic": True,
        "lookup_mode": "sparse_basket",
        "missing_event_ids": np.asarray(sorted(known_missing), dtype=np.int64),
    }


def _locate_monotonic_event_ranges_by_basket(
    branch,
    *,
    target_root_ids: np.ndarray,
    num_entries: int,
    decompression_executor=None,
    interpretation_executor=None,
) -> tuple[dict[int, tuple[int, int]], dict[str, Any]]:
    """Locate event entry ranges without scanning the preceding tree.

    WCSim writes ``AllSecondaries`` in event order.  Uproot exposes the entry
    boundaries of each compressed TBranch basket, so a binary search over
    basket edge event IDs can jump to late requested events.  Only baskets that
    bracket requested contiguous event runs are then decoded.  Any local
    monotonicity violation raises and the caller falls back to the conservative
    streaming event-branch scan.
    """
    targets = np.unique(np.asarray(target_root_ids, dtype=np.int64))
    if targets.size == 0:
        return {}, {
            "baskets_available": 0,
            "baskets_probed": 0,
            "entry_ranges_decoded": 0,
            "entries_decoded": 0,
            "events_found": 0,
        }
    try:
        offsets = np.asarray(branch.entry_offsets, dtype=np.int64)
    except Exception as error:
        raise RuntimeError("event branch does not expose basket entry offsets") from error
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or int(offsets[0]) != 0
        or int(offsets[-1]) != int(num_entries)
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        raise RuntimeError("event branch basket entry offsets are unavailable or invalid")

    n_baskets = int(offsets.size - 1)
    edge_cache: dict[int, tuple[int, int]] = {}
    baskets_probed: set[int] = set()

    def basket_edges(basket: int) -> tuple[int, int]:
        basket = int(basket)
        cached = edge_cache.get(basket)
        if cached is not None:
            return cached
        start = int(offsets[basket])
        stop = int(offsets[basket + 1])
        first = _branch_array(
            branch,
            entry_start=start,
            entry_stop=start + 1,
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
        )
        if stop - start == 1:
            last = first
        else:
            last = _branch_array(
                branch,
                entry_start=stop - 1,
                entry_stop=stop,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
            )
        if first.size != 1 or last.size != 1:
            raise RuntimeError("could not read event IDs at a basket boundary")
        lo = int(first[0])
        hi = int(last[0])
        if hi < lo:
            raise RuntimeError("event IDs decrease within a TBranch basket")
        edge_cache[basket] = (lo, hi)
        baskets_probed.add(basket)
        return lo, hi

    def first_basket_with_last_at_least(event_id: int) -> int:
        low = 0
        high = n_baskets
        while low < high:
            mid = (low + high) // 2
            _, last = basket_edges(mid)
            if last >= int(event_id):
                high = mid
            else:
                low = mid + 1
        return int(low)

    def first_basket_with_first_greater_than(event_id: int) -> int:
        low = 0
        high = n_baskets
        while low < high:
            mid = (low + high) // 2
            first, _ = basket_edges(mid)
            if first > int(event_id):
                high = mid
            else:
                low = mid + 1
        return int(low)

    found: dict[int, tuple[int, int]] = {}
    entries_decoded = 0
    ranges_decoded = 0
    for run in _contiguous_event_runs(targets):
        run_lo = int(run[0])
        run_hi = int(run[-1])
        first_basket = first_basket_with_last_at_least(run_lo)
        after_last_basket = first_basket_with_first_greater_than(run_hi)
        last_basket = after_last_basket - 1
        if first_basket >= n_baskets or last_basket < first_basket:
            continue
        entry_start = int(offsets[first_basket])
        entry_stop = int(offsets[last_basket + 1])
        values = _branch_array(
            branch,
            entry_start=entry_start,
            entry_stop=entry_stop,
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
        ).astype(np.int64, copy=False)
        if values.size != entry_stop - entry_start:
            raise RuntimeError("event basket lookup returned an incomplete entry range")
        if values.size and np.any(values[1:] < values[:-1]):
            raise RuntimeError("event IDs are not monotonic in a requested basket range")
        entries_decoded += int(values.size)
        ranges_decoded += 1
        lefts = np.searchsorted(values, run, side="left")
        rights = np.searchsorted(values, run, side="right")
        for event_id, left, right in zip(run, lefts, rights):
            if int(right) > int(left):
                found[int(event_id)] = (
                    int(entry_start + int(left)),
                    int(entry_start + int(right)),
                )

    return found, {
        "baskets_available": int(n_baskets),
        "baskets_probed": int(len(baskets_probed)),
        "entry_ranges_decoded": int(ranges_decoded),
        "entries_decoded": int(entries_decoded),
        "events_found": int(len(found)),
    }


def _ranges_from_index(
    index: Mapping[str, Any], target_root_ids: np.ndarray
) -> tuple[list[tuple[int, int]], dict[int, tuple[int, int]]]:
    event_ids = np.asarray(index["event_ids"], dtype=np.int64)
    starts = np.asarray(index["entry_starts"], dtype=np.int64)
    stops = np.asarray(index["entry_stops"], dtype=np.int64)
    by_event: dict[int, tuple[int, int]] = {}
    if event_ids.size:
        locations = np.searchsorted(event_ids, target_root_ids)
        for target, location in zip(target_root_ids, locations):
            if int(location) < event_ids.size and int(event_ids[int(location)]) == int(target):
                by_event[int(target)] = (int(starts[int(location)]), int(stops[int(location)]))
    return _coalesce_ranges(list(by_event.values())), by_event


def _scan_nonmonotonic_target_ranges(
    tree,
    *,
    event_branch: str,
    target_root_ids: np.ndarray,
    num_entries: int,
    step_size: str | int,
    decompression_executor=None,
    interpretation_executor=None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    ranges: list[tuple[int, int]] = []
    rows = 0
    previous_last: int | None = None
    monotonic = True
    for chunk_start, mapping in _tree_iterate(
        tree,
        expressions=[event_branch],
        entry_start=0,
        entry_stop=num_entries,
        step_size=step_size,
        decompression_executor=decompression_executor,
        interpretation_executor=interpretation_executor,
    ):
        values = _as_1d(mapping[event_branch], dtype=np.int64)
        rows += int(values.size)
        if values.size:
            if previous_last is not None and int(values[0]) < previous_last:
                monotonic = False
            if np.any(values[1:] < values[:-1]):
                monotonic = False
            previous_last = int(values[-1])
        mask = _event_mask(values, target_root_ids)
        positions = np.flatnonzero(mask)
        if positions.size:
            changes = np.flatnonzero(np.r_[True, positions[1:] != positions[:-1] + 1])
            stops = np.append(changes[1:], positions.size)
            for begin, end in zip(changes, stops):
                first = int(positions[begin])
                last = int(positions[end - 1]) + 1
                ranges.append((int(chunk_start + first), int(chunk_start + last)))
    return _coalesce_ranges(ranges), {
        "entries_scanned": int(rows),
        "event_branch_monotonic": bool(monotonic),
    }


def _resolve_requested_entry_ranges(
    tree,
    *,
    root_path: Path,
    stat,
    tree_resolved: str,
    event_branch: str,
    target_root_ids: np.ndarray,
    index_target_max_event: int | None,
    num_entries: int,
    step_size: str | int,
    use_cache: bool,
    cache_dir: str | Path | None,
    decompression_executor=None,
    interpretation_executor=None,
) -> tuple[list[tuple[int, int]], dict[int, tuple[int, int]], dict[str, Any], Mapping[str, Any] | None]:
    wall0 = time.perf_counter()
    signature = _event_index_signature(
        root_path=root_path,
        stat=stat,
        tree_resolved=tree_resolved,
        event_branch=event_branch,
        num_entries=num_entries,
    )
    cache_path: Path | None = None
    cache_hit = False
    cache_extended = False
    cache_error = None
    basket_lookup_error = None
    index = _empty_event_index()
    scan_info: dict[str, Any] = {"entries_scanned": 0, "events_added": 0}
    basket_info: dict[str, Any] = {
        "baskets_available": 0,
        "baskets_probed": 0,
        "entry_ranges_decoded": 0,
        "entries_decoded": 0,
        "events_found": 0,
    }

    targets = np.unique(np.asarray(target_root_ids, dtype=np.int64))
    if targets.size == 0:
        return [], {}, {"event_index_wall_s": 0.0}, index
    target_max_for_index = max(
        int(targets[-1]),
        int(index_target_max_event)
        if index_target_max_event is not None
        else int(targets[-1]),
    )

    # Cosmic supervisors can request that the first child populate the compact
    # sidecar through the last child event.  For normal WCSim event numbering,
    # this remains a bounded contiguous integer vector and lets later children
    # perform no event-branch I/O at all.  Avoid pathological allocations if a
    # caller supplies an enormous or sparse numerical event-ID range.
    lookup_targets = targets
    prefetch_applied = False
    if (
        target_max_for_index > int(targets[-1])
        and targets.size == int(targets[-1] - targets[0] + 1)
        and int(target_max_for_index - targets[0] + 1) <= 1_000_000
    ):
        lookup_targets = np.arange(
            int(targets[0]), int(target_max_for_index) + 1, dtype=np.int64
        )
        prefetch_applied = True

    def extend_or_build(current: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        nonlocal basket_lookup_error, basket_info, scan_info
        mode = str(current.get("lookup_mode", "stream_scan"))
        empty = len(np.asarray(current.get("event_ids", []))) == 0
        empty &= int(current.get("scan_entry_stop", 0)) == 0
        if mode == "sparse_basket" or empty:
            known = np.unique(
                np.concatenate(
                    (
                        np.asarray(current.get("event_ids", []), dtype=np.int64),
                        np.asarray(current.get("missing_event_ids", []), dtype=np.int64),
                    )
                )
            )
            missing = np.setdiff1d(lookup_targets, known, assume_unique=True)
            if missing.size == 0:
                return dict(current), False
            try:
                event_branch_object = tree[event_branch]
                found, basket_info = _locate_monotonic_event_ranges_by_basket(
                    event_branch_object,
                    target_root_ids=missing,
                    num_entries=num_entries,
                    decompression_executor=decompression_executor,
                    interpretation_executor=interpretation_executor,
                )
                base = current if mode == "sparse_basket" else _empty_event_index(
                    lookup_mode="sparse_basket"
                )
                return (
                    _merge_sparse_event_ranges(
                        base,
                        found,
                        attempted_event_ids=missing,
                    ),
                    True,
                )
            except Exception as error:
                basket_lookup_error = f"{type(error).__name__}: {error}"

        # Compatibility and correctness fallback for ROOT layouts that do not
        # expose usable basket boundaries or whose event IDs are not monotonic.
        # A sparse basket cache cannot seed a streaming prefix index, so restart
        # from entry zero in that rare case.
        stream_base = (
            current
            if str(current.get("lookup_mode", "stream_scan")) == "stream_scan"
            else _empty_event_index(lookup_mode="stream_scan")
        )
        result, scan_info = _extend_monotonic_event_index(
            tree,
            event_branch=event_branch,
            index=stream_base,
            target_max_event=int(target_max_for_index),
            num_entries=num_entries,
            step_size=step_size,
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
        )
        changed = bool(
            scan_info.get("entries_scanned", 0)
            or scan_info.get("events_added", 0)
        )
        return result, changed

    if use_cache:
        try:
            directory = _cache_directory(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            cache_path = _event_index_cache_path(directory, signature)
            with _event_index_lock(cache_path):
                loaded = _load_event_index(cache_path, signature)
                if loaded is not None:
                    index = loaded
                    cache_hit = True
                index, cache_extended = extend_or_build(index)
                if bool(index.get("monotonic", True)):
                    _save_event_index_atomic(cache_path, signature, index)
        except Exception as error:
            cache_error = f"{type(error).__name__}: {error}"
            cache_path = None
            cache_hit = False
            cache_extended = False
            index = _empty_event_index()

    if not use_cache or cache_path is None:
        index, cache_extended = extend_or_build(index)

    if bool(index.get("monotonic", True)):
        ranges, by_event = _ranges_from_index(index, targets)
        event_branch_monotonic = True
        index_for_synthesis: Mapping[str, Any] | None = index
    else:
        ranges, fallback = _scan_nonmonotonic_target_ranges(
            tree,
            event_branch=event_branch,
            target_root_ids=targets,
            num_entries=num_entries,
            step_size=step_size,
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
        )
        by_event = {}
        event_branch_monotonic = bool(fallback.get("event_branch_monotonic", False))
        scan_info = fallback
        index_for_synthesis = None

    lookup_mode = str(index.get("lookup_mode", "stream_scan"))
    metadata = {
        "event_index_cache_enabled": bool(use_cache),
        "event_index_cache_path": None if cache_path is None else str(cache_path),
        "event_index_cache_hit": bool(cache_hit),
        "event_index_cache_extended": bool(cache_extended),
        "event_index_cache_error": cache_error,
        "event_index_lookup_mode": lookup_mode,
        "event_index_basket_lookup_error": basket_lookup_error,
        "event_index_entries_scanned_this_call": int(scan_info.get("entries_scanned", 0)),
        "event_index_entries_decoded_this_call": int(basket_info.get("entries_decoded", 0)),
        "event_index_baskets_available": int(basket_info.get("baskets_available", 0)),
        "event_index_baskets_probed_this_call": int(basket_info.get("baskets_probed", 0)),
        "event_index_basket_ranges_decoded_this_call": int(
            basket_info.get("entry_ranges_decoded", 0)
        ),
        "event_index_events_added_this_call": int(
            basket_info.get("events_found", 0)
            if lookup_mode == "sparse_basket"
            else scan_info.get("events_added", 0)
        ),
        "event_index_events_stored": int(len(index.get("event_ids", []))),
        "event_index_missing_events_stored": int(
            len(index.get("missing_event_ids", []))
        ),
        "event_index_scan_entry_stop": int(index.get("scan_entry_stop", 0)),
        "event_index_complete": bool(index.get("complete", False)),
        "event_branch_monotonic": bool(event_branch_monotonic),
        "event_index_wall_s": float(time.perf_counter() - wall0),
        "event_index_target_max_event": int(target_max_for_index),
        "event_index_prefetched_beyond_output": bool(prefetch_applied),
        "requested_entry_ranges": [[int(a), int(b)] for a, b in ranges],
        "requested_root_events_found_in_index": int(len(by_event)),
    }
    return ranges, by_event, metadata, index_for_synthesis

def _synthesise_event_ids(
    entry_start: int,
    entry_stop: int,
    index: Mapping[str, Any],
) -> np.ndarray:
    output = np.empty(max(0, int(entry_stop) - int(entry_start)), dtype=np.int64)
    if output.size == 0:
        return output
    event_ids = np.asarray(index["event_ids"], dtype=np.int64)
    starts = np.asarray(index["entry_starts"], dtype=np.int64)
    stops = np.asarray(index["entry_stops"], dtype=np.int64)
    first = int(np.searchsorted(stops, int(entry_start), side="right"))
    cursor = 0
    for location in range(first, event_ids.size):
        start = max(int(entry_start), int(starts[location]))
        stop = min(int(entry_stop), int(stops[location]))
        if stop <= start:
            if int(starts[location]) >= int(entry_stop):
                break
            continue
        out_start = start - int(entry_start)
        out_stop = stop - int(entry_start)
        output[out_start:out_stop] = int(event_ids[location])
        cursor = max(cursor, out_stop)
        if stop >= int(entry_stop):
            break
    if cursor < output.size:
        raise RuntimeError("Event-index ranges did not cover a requested numeric chunk")
    return output


def _read_sparse_branch_rows(
    branch,
    row_indices: np.ndarray,
    *,
    decompression_executor=None,
    interpretation_executor=None,
) -> tuple[dict[int, Any], dict[str, int]]:
    rows = np.unique(np.asarray(row_indices, dtype=np.int64))
    output: dict[int, Any] = {}
    if rows.size == 0:
        return output, {"ranges_read": 0, "entries_decoded": 0}

    ranges: list[tuple[int, int, np.ndarray]] = []
    try:
        offsets = np.asarray(branch.entry_offsets, dtype=np.int64)
    except Exception:
        offsets = np.empty(0, dtype=np.int64)
    if offsets.size >= 2 and np.all(offsets[1:] >= offsets[:-1]):
        baskets = np.searchsorted(offsets, rows, side="right") - 1
        baskets = np.clip(baskets, 0, offsets.size - 2)
        for basket in np.unique(baskets):
            selected = rows[baskets == basket]
            ranges.append((int(offsets[basket]), int(offsets[basket + 1]), selected))
    else:
        # Older/fake branch objects: group nearby rows into bounded spans.
        group_start = 0
        for location in range(1, rows.size + 1):
            split = (
                location == rows.size
                or int(rows[location] - rows[location - 1]) > 4096
                or int(rows[location - 1] - rows[group_start]) >= 65536
            )
            if split:
                selected = rows[group_start:location]
                ranges.append((int(selected[0]), int(selected[-1]) + 1, selected))
                group_start = location

    decoded = 0
    for start, stop, selected in ranges:
        array = _branch_array(
            branch,
            entry_start=start,
            entry_stop=stop,
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
        )
        decoded += int(array.size)
        for row in selected:
            local = int(row) - start
            if 0 <= local < array.size:
                output[int(row)] = array[local]
    return output, {"ranges_read": int(len(ranges)), "entries_decoded": int(decoded)}


def _enrich_sparse_entry_details(
    tree,
    *,
    branch_map: Mapping[str, str],
    records: Mapping[int, dict[str, Any]],
    decompression_executor=None,
    interpretation_executor=None,
) -> dict[str, Any]:
    rows = np.asarray(
        sorted(
            {
                int(record["_global_entry_index"])
                for record in records.values()
                if record.get("found")
                and record.get("_global_entry_index") is not None
            }
        ),
        dtype=np.int64,
    )
    if rows.size == 0:
        return {
            "sparse_detail_rows": 0,
            "sparse_detail_branches": [],
            "sparse_detail_ranges_read": 0,
            "sparse_detail_entries_decoded": 0,
            "sparse_detail_wall_s": 0.0,
        }
    wall0 = time.perf_counter()
    detail_names = [
        name
        for name in (*_SPARSE_NUMERIC_BRANCHES, *_SPARSE_TEXT_BRANCHES)
        if name in branch_map
    ]
    values_by_name: dict[str, dict[int, Any]] = {}
    ranges_read = 0
    entries_decoded = 0
    for canonical in detail_names:
        branch = tree[branch_map[canonical]]
        values, metrics = _read_sparse_branch_rows(
            branch,
            rows,
            decompression_executor=decompression_executor,
            interpretation_executor=interpretation_executor,
        )
        values_by_name[canonical] = values
        ranges_read += int(metrics["ranges_read"])
        entries_decoded += int(metrics["entries_decoded"])

    for record in records.values():
        row = record.get("_global_entry_index")
        if row is None or not record.get("found"):
            continue
        row = int(row)
        fraction = float(record.get("segment_fraction", 0.0))
        pre_direction = np.asarray(
            [
                _finite_float(values_by_name.get("dir_x", {}).get(row)),
                _finite_float(values_by_name.get("dir_y", {}).get(row)),
                _finite_float(values_by_name.get("dir_z", {}).get(row)),
            ],
            dtype=np.float64,
        )
        post_direction = np.asarray(
            [
                _finite_float(values_by_name.get("post_dir_x", {}).get(row)),
                _finite_float(values_by_name.get("post_dir_y", {}).get(row)),
                _finite_float(values_by_name.get("post_dir_z", {}).get(row)),
            ],
            dtype=np.float64,
        )
        if np.all(np.isfinite(pre_direction)) and np.all(np.isfinite(post_direction)):
            direction = (1.0 - fraction) * pre_direction + fraction * post_direction
        elif np.all(np.isfinite(post_direction)):
            direction = post_direction
        elif np.all(np.isfinite(pre_direction)):
            direction = pre_direction
        else:
            # Preserve the required-branch post-minus-pre step tangent when
            # stored direction branches are absent from this ROOT schema.
            direction = np.asarray(
                [record.get("dir_x"), record.get("dir_y"), record.get("dir_z")],
                dtype=np.float64,
            )
        record["dir_x"], record["dir_y"], record["dir_z"] = _normalise_direction(direction)
        for canonical, record_name in (
            ("material", "pre_material"),
            ("post_material", "post_material"),
            ("volume", "pre_volume"),
            ("post_volume", "post_volume"),
            ("step_process", "step_process"),
        ):
            if canonical in values_by_name:
                record[record_name] = _text(values_by_name[canonical].get(row, ""))

    return {
        "sparse_detail_rows": int(rows.size),
        "sparse_detail_branches": detail_names,
        "sparse_detail_ranges_read": int(ranges_read),
        "sparse_detail_entries_decoded": int(entries_decoded),
        "sparse_detail_wall_s": float(time.perf_counter() - wall0),
    }


def read_active_water_entry_truth(
    root_path: str | Path,
    *,
    detector,
    source_event_ids: Sequence[int],
    fit_particle: str,
    tree_name: str = DEFAULT_TREE_NAME,
    event_id_offset: int = 0,
    expected_track_ids: Mapping[int, int] | None = None,
    configured_track_id: int | None = None,
    coordinate_offset_mm: Sequence[float] = (0.0, 0.0, 0.0),
    uproot_step_size: str | int = DEFAULT_UPROOT_STEP_SIZE,
    event_index_step_size: str | int = DEFAULT_EVENT_INDEX_STEP_SIZE,
    uproot_io_workers: int = DEFAULT_UPROOT_IO_WORKERS,
    use_event_index_cache: bool = DEFAULT_USE_EVENT_INDEX_CACHE,
    event_index_cache_dir: str | Path | None = None,
    event_index_prefetch_max_root_event: int | None = None,
    include_optional_details: bool = DEFAULT_INCLUDE_OPTIONAL_DETAILS,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Read a WCSim AllSecondaries tree and return source-indexed records."""
    path = Path(root_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WCSim truth ROOT file does not exist: {path}")
    source_ids = [int(value) for value in source_event_ids]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_event_ids must be unique")
    wall0 = time.perf_counter()
    stat = path.stat()
    uproot = _import_uproot()

    with _uproot_executors(uproot_io_workers) as (decompression_executor, interpretation_executor):
        with _open_uproot(uproot, path, n_workers=uproot_io_workers) as root_file:
            tree, resolved_tree = _resolve_tree(root_file, tree_name)
            branch_map, _, optional_missing = _resolve_branches(tree)
            try:
                num_entries = int(tree.num_entries)
            except Exception:
                num_entries = -1
            if num_entries < 0:
                raise RuntimeError("Could not determine AllSecondaries tree.num_entries")

            target_root_ids = np.asarray(
                sorted(int(source) + int(event_id_offset) for source in source_ids),
                dtype=np.int64,
            )
            entry_ranges, event_ranges, index_metadata, index_for_synthesis = _resolve_requested_entry_ranges(
                tree,
                root_path=path,
                stat=stat,
                tree_resolved=str(resolved_tree),
                event_branch=branch_map["evt"],
                target_root_ids=target_root_ids,
                index_target_max_event=event_index_prefetch_max_root_event,
                num_entries=num_entries,
                step_size=event_index_step_size,
                use_cache=bool(use_event_index_cache),
                cache_dir=event_index_cache_dir,
                decompression_executor=decompression_executor,
                interpretation_executor=interpretation_executor,
            )

            main_canonical = [name for name in _REQUIRED_BRANCHES if name != "evt"]
            if "pdg" in branch_map:
                main_canonical.append("pdg")
            else:
                main_canonical.append("particle")
            if "step" in branch_map:
                main_canonical.append("step")
            main_actual = list(dict.fromkeys(branch_map[name] for name in main_canonical))
            if index_for_synthesis is None:
                main_canonical.insert(0, "evt")
                main_actual.insert(0, branch_map["evt"])

            numeric_read_wall0 = time.perf_counter()

            def canonical_chunks():
                for range_start, range_stop in entry_ranges:
                    for chunk_start, raw_chunk in _tree_iterate(
                        tree,
                        expressions=main_actual,
                        entry_start=range_start,
                        entry_stop=range_stop,
                        step_size=uproot_step_size,
                        decompression_executor=decompression_executor,
                        interpretation_executor=interpretation_executor,
                    ):
                        raw_mapping = _mapping_from_uproot(raw_chunk)
                        canonical = {
                            name: raw_mapping[branch_map[name]]
                            for name in main_canonical
                            if name != "evt" and branch_map[name] in raw_mapping
                        }
                        first = next(iter(canonical.values()), np.empty(0))
                        n_rows = int(len(_as_1d(first)))
                        if index_for_synthesis is not None:
                            canonical["evt"] = _synthesise_event_ids(
                                chunk_start,
                                chunk_start + n_rows,
                                index_for_synthesis,
                            )
                        else:
                            canonical["evt"] = raw_mapping[branch_map["evt"]]
                        yield int(chunk_start), canonical

            records, metadata = extract_active_water_entries_from_chunks(
                canonical_chunks(),
                detector=detector,
                source_event_ids=source_ids,
                fit_particle=fit_particle,
                event_id_offset=event_id_offset,
                expected_track_ids=expected_track_ids,
                configured_track_id=configured_track_id,
                coordinate_offset_mm=coordinate_offset_mm,
                keep_private_entry_indices=True,
            )
            numeric_read_wall_s = float(time.perf_counter() - numeric_read_wall0)

            if include_optional_details:
                sparse_metadata = _enrich_sparse_entry_details(
                    tree,
                    branch_map=branch_map,
                    records=records,
                    decompression_executor=decompression_executor,
                    interpretation_executor=interpretation_executor,
                )
            else:
                sparse_metadata = {
                    "sparse_detail_rows": 0,
                    "sparse_detail_branches": [],
                    "sparse_detail_ranges_read": 0,
                    "sparse_detail_entries_decoded": 0,
                    "sparse_detail_wall_s": 0.0,
                }

    for record in records.values():
        for key in list(record):
            if str(key).startswith("_"):
                record.pop(key, None)

    metadata.update(index_metadata)
    metadata.update(sparse_metadata)
    metadata.update(
        {
            "enabled": True,
            "root_file": str(path),
            "root_file_size_bytes": int(stat.st_size),
            "root_file_mtime_ns": int(stat.st_mtime_ns),
            "tree_requested": str(tree_name),
            "tree_resolved": str(resolved_tree),
            "tree_num_entries": int(num_entries),
            "tree_entry_start_read": (
                min((start for start, _ in entry_ranges), default=None)
            ),
            "tree_entry_stop_read": (
                max((stop for _, stop in entry_ranges), default=None)
            ),
            "tree_entry_ranges_read": [[int(a), int(b)] for a, b in entry_ranges],
            "resolved_event_entry_ranges": {
                str(event): [int(bounds[0]), int(bounds[1])]
                for event, bounds in sorted(event_ranges.items())
            },
            "resolved_branches": dict(sorted(branch_map.items())),
            "main_numeric_branches_read": list(main_canonical),
            "event_branch_synthesised_from_index": bool(index_for_synthesis is not None),
            "optional_branches_missing": list(optional_missing),
            "include_optional_details": bool(include_optional_details),
            "uproot_step_size": uproot_step_size,
            "event_index_step_size": event_index_step_size,
            "uproot_io_workers": int(max(1, uproot_io_workers)),
            "numeric_read_and_processing_wall_s": float(numeric_read_wall_s),
            "read_wall_s": float(time.perf_counter() - wall0),
            "geometry": (
                detector.metadata()
                if hasattr(detector, "metadata")
                else {"type": type(detector).__name__}
            ),
        }
    )
    return records, metadata


def truth_output_records(
    source_event_ids: Sequence[int],
    records: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return complete event-aligned truth records using only pickle-safe values.

    The compact production dictionary historically exposed truth through flat
    ``truth_*`` columns.  A dedicated record list is also useful because users
    naturally expect one self-contained truth object per fitted event, even when
    the much larger optimizer ``results`` payload is disabled.
    """
    rows: list[dict[str, Any]] = []
    for value in source_event_ids:
        source = int(value)
        record = dict(
            records.get(source, _empty_record(source, source, "missing_record"))
        )
        # Preserve the requested source ordering even if an externally supplied
        # record omitted or carried a NumPy scalar for its identifier.
        record["source_event_index"] = int(source)
        rows.append(record)
    return rows


def truth_output_columns(
    source_event_ids: Sequence[int],
    records: Mapping[int, Mapping[str, Any]],
) -> dict[str, list[Any]]:
    """Convert source-indexed truth records into event-aligned output columns."""
    ids = [int(value) for value in source_event_ids]
    rows = truth_output_records(ids, records)
    output: dict[str, list[Any]] = {
        "truth_active_water_entry_found": [bool(row.get("found", False)) for row in rows],
        "truth_active_water_entry_source_event_index": ids,
    }
    for output_name, record_name in _FLOAT_OUTPUT_FIELDS.items():
        output[output_name] = [_finite_float(row.get(record_name)) for row in rows]
    for output_name, record_name in _INT_OUTPUT_FIELDS.items():
        output[output_name] = [_finite_int(row.get(record_name)) for row in rows]
    for output_name, record_name in _STRING_OUTPUT_FIELDS.items():
        output[output_name] = [_text(row.get(record_name, "")) for row in rows]
    return output


def attach_truth_to_output(
    output: dict[str, Any],
    *,
    source_event_ids: Sequence[int],
    records: Mapping[int, Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach truth columns/metadata without changing any fitted estimate."""
    ids = [int(value) for value in source_event_ids]
    selected_rows = truth_output_records(ids, records)
    # Always provide an explicit one-record-per-event container.  This is
    # intentionally independent of SAVE_DETAILED_EVENT_RESULTS; the latter
    # controls optimizer traces, not whether requested truth is accessible.
    output["wcsim_truth"] = selected_rows
    output.update(truth_output_columns(ids, records))
    selected_status = Counter(str(row.get("status", "")) for row in selected_rows)
    truth_metadata = dict(metadata)
    truth_metadata.update(
        {
            "n_output_events": int(len(ids)),
            "n_entries_found_in_output": int(
                sum(bool(row.get("found", False)) for row in selected_rows)
            ),
            "output_status_counts": dict(sorted(selected_status.items())),
            "event_record_key": "wcsim_truth",
            "event_record_count": int(len(selected_rows)),
            "flat_event_columns_attached": True,
        }
    )
    output.setdefault("metadata", {})["wcsim_truth_root"] = truth_metadata
    detailed = output.get("results")
    if isinstance(detailed, list):
        for ordinal, result in enumerate(detailed):
            if not isinstance(result, dict):
                continue
            source = ids[ordinal] if ordinal < len(ids) else int(
                result.get("source_event_index", result.get("event_index", ordinal))
            )
            record = (
                dict(selected_rows[ordinal])
                if ordinal < len(selected_rows)
                else dict(records.get(
                    source, _empty_record(source, source, "missing_record")
                ))
            )
            # Retain the historical nested name and add the clearer general
            # alias for detailed-output consumers.
            result["truth_active_water_entry"] = dict(record)
            result["wcsim_truth"] = dict(record)
    return output
