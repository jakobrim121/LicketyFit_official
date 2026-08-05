"""Orientation-independent local track parameterization utilities.

The old fitter represented directions with two global direction cosines and a
``cz_sign`` branch.  That chart is singular on the complete ``cz == 0``
equator.  Tracks parallel to the x or y axes therefore sat on an excluded or
ill-conditioned boundary.

This module instead uses a local tangent chart anchored on an arbitrary unit
seed direction ``d0``.  The two fitted coordinates ``u`` and ``v`` define

    d(u, v) = normalize(d0 + u e1 + v e2),

where ``(d0, e1, e2)`` is an orthonormal frame.  The chart is regular at every
anchor direction, needs only a square root, and covers the open hemisphere
centred on the anchor.  A modest global seed set covers the sphere; the local
fit is then always performed near ``u=v=0``.

The functions are deliberately NumPy-only and allocation-light.  Their cost is
negligible compared with one optical likelihood evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

_EPS = 1.0e-15


def normalize_direction(direction: Sequence[float]) -> np.ndarray:
    """Return a finite, contiguous unit 3-vector."""
    out = np.asarray(direction, dtype=np.float64)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"direction must be a finite 3-vector, got {direction!r}")
    norm = float(np.linalg.norm(out))
    if norm <= _EPS:
        raise ValueError("direction has zero norm")
    return np.ascontiguousarray(out / norm, dtype=np.float64)


def stable_tangent_basis(direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic right-handed orthonormal frame ``(d, e1, e2)``.

    The Cartesian axis least aligned with ``d`` is used as the reference.  This
    keeps the cross product well conditioned for every direction.  Any basis
    discontinuity occurs only when choosing a *new anchor*; an anchor and its
    basis remain fixed throughout one optimization stage.
    """
    d = normalize_direction(direction)
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(d)))] = 1.0
    e1 = np.cross(d, reference)
    e1 /= max(float(np.linalg.norm(e1)), _EPS)
    e2 = np.cross(d, e1)
    e2 /= max(float(np.linalg.norm(e2)), _EPS)
    return (
        np.ascontiguousarray(d),
        np.ascontiguousarray(e1),
        np.ascontiguousarray(e2),
    )


@dataclass(frozen=True)
class TangentDirectionChart:
    """A fast local chart for directions on the unit sphere."""

    anchor: np.ndarray
    e1: np.ndarray
    e2: np.ndarray

    @classmethod
    def from_direction(cls, direction: Sequence[float]) -> "TangentDirectionChart":
        d, e1, e2 = stable_tangent_basis(direction)
        return cls(d, e1, e2)

    def direction(self, u: float, v: float) -> np.ndarray | None:
        """Map finite tangent coordinates to a unit direction.

        The normalized-linear (gnomonic) retraction has no finite coordinate
        boundary.  Its angular displacement from the anchor is
        ``atan(hypot(u,v))`` and approaches, but never reaches, 90 degrees.
        """
        u = float(u)
        v = float(v)
        if not (math.isfinite(u) and math.isfinite(v)):
            return None
        inv_norm = 1.0 / math.sqrt(1.0 + u * u + v * v)
        out = (self.anchor + u * self.e1 + v * self.e2) * inv_norm
        return np.ascontiguousarray(out, dtype=np.float64)

    def coordinates(self, direction: Sequence[float]) -> tuple[float, float] | None:
        """Return ``(u,v)`` for a direction in this chart's open hemisphere."""
        d = normalize_direction(direction)
        forward = float(np.dot(d, self.anchor))
        if forward <= _EPS:
            return None
        return float(np.dot(d, self.e1) / forward), float(np.dot(d, self.e2) / forward)

    def angular_offset_rad(self, u: float, v: float) -> float:
        return math.atan(math.hypot(float(u), float(v)))

    def direction_jacobian(self, u: float, v: float) -> np.ndarray:
        """Return ``d(direction)/d(u,v)`` as a ``(3,2)`` matrix."""
        u = float(u)
        v = float(v)
        norm = math.sqrt(1.0 + u * u + v * v)
        d = (self.anchor + u * self.e1 + v * self.e2) / norm
        j_u = self.e1 / norm - d * (u / (norm * norm))
        j_v = self.e2 / norm - d * (v / (norm * norm))
        return np.ascontiguousarray(np.column_stack((j_u, j_v)), dtype=np.float64)

    def as_metadata(self) -> dict[str, list[float]]:
        return {
            "anchor": self.anchor.tolist(),
            "e1": self.e1.tolist(),
            "e2": self.e2.tolist(),
        }


def chart_from_metadata(metadata: Mapping[str, Sequence[float]]) -> TangentDirectionChart:
    """Rebuild a chart and validate a stored frame."""
    anchor = normalize_direction(metadata["anchor"])
    e1 = normalize_direction(metadata["e1"])
    e2 = normalize_direction(metadata["e2"])
    frame = np.column_stack((anchor, e1, e2))
    if not np.allclose(frame.T @ frame, np.eye(3), rtol=0.0, atol=1.0e-10):
        raise ValueError("stored direction chart is not orthonormal")
    if float(np.linalg.det(frame)) < 0.0:
        raise ValueError("stored direction chart is not right handed")
    return TangentDirectionChart(anchor, e1, e2)


def direction_from_mapping(mapping: Mapping[str, object]) -> np.ndarray:
    """Read a direction from new or historical result/seed dictionaries.

    Preferred schemas, in order:
      * ``direction`` iterable;
      * ``dir_x``, ``dir_y``, ``dir_z``;
      * ``cx``, ``cy``, ``cz``;
      * historical ``cx``, ``cy``, ``cz_sign``.
    """
    if "direction" in mapping:
        return normalize_direction(mapping["direction"])  # type: ignore[arg-type]
    if all(k in mapping for k in ("dir_x", "dir_y", "dir_z")):
        return normalize_direction([mapping["dir_x"], mapping["dir_y"], mapping["dir_z"]])
    if all(k in mapping for k in ("cx", "cy", "cz")):
        return normalize_direction([mapping["cx"], mapping["cy"], mapping["cz"]])
    if all(k in mapping for k in ("cx", "cy")):
        cx = float(mapping["cx"])
        cy = float(mapping["cy"])
        residual = 1.0 - cx * cx - cy * cy
        if residual < -1.0e-12:
            raise ValueError("historical cx,cy lie outside the unit disk")
        residual = max(residual, 0.0)
        sign_value = mapping.get("cz_sign", 1.0)
        sign = -1.0 if float(sign_value) < 0.0 else 1.0
        return normalize_direction([cx, cy, sign * math.sqrt(residual)])
    raise KeyError("mapping does not contain a recognized direction representation")


def attach_direction_components(
    values: Mapping[str, object],
    *,
    chart: TangentDirectionChart | None = None,
    u_name: str = "dir_u",
    v_name: str = "dir_v",
) -> dict[str, object]:
    """Return a copy carrying normalized ``cx,cy,cz`` and chart metadata."""
    out: dict[str, object] = dict(values)
    if chart is not None:
        direction = chart.direction(float(out.get(u_name, 0.0)), float(out.get(v_name, 0.0)))
        if direction is None:
            direction = np.full(3, np.nan, dtype=np.float64)
        out["direction_chart"] = chart.as_metadata()
        out["direction_chart_u"] = float(out.get(u_name, 0.0))
        out["direction_chart_v"] = float(out.get(v_name, 0.0))
    else:
        direction = direction_from_mapping(out)
    out["cx"] = float(direction[0])
    out["cy"] = float(direction[1])
    out["cz"] = float(direction[2])
    # Retain this as a read-only compatibility alias.  It no longer selects a
    # fit branch and is not sufficient to reconstruct the fitted chart.
    out["cz_sign"] = -1.0 if float(direction[2]) < 0.0 else 1.0
    return out


def reanchor_values(
    values: Mapping[str, object],
    old_chart: TangentDirectionChart,
    *,
    u_name: str = "dir_u",
    v_name: str = "dir_v",
) -> tuple[dict[str, object], TangentDirectionChart]:
    """Make the current direction the regular origin of a new local chart."""
    direction = old_chart.direction(float(values.get(u_name, 0.0)), float(values.get(v_name, 0.0)))
    if direction is None:
        raise ValueError("cannot re-anchor a non-finite direction")
    new_chart = TangentDirectionChart.from_direction(direction)
    out = dict(values)
    out[u_name] = 0.0
    out[v_name] = 0.0
    return out, new_chart


def local_to_cartesian_covariance(
    local_covariance: np.ndarray,
    chart: TangentDirectionChart,
    u: float,
    v: float,
) -> np.ndarray:
    """Transform covariance from ``(x,y,z,u,v,L)`` to ``(x,y,z,cx,cy,cz,L)``."""
    cov = np.asarray(local_covariance, dtype=np.float64)
    if cov.shape != (6, 6):
        raise ValueError("local covariance must have shape (6,6)")
    transform = np.zeros((7, 6), dtype=np.float64)
    transform[:3, :3] = np.eye(3)
    transform[3:6, 3:5] = chart.direction_jacobian(u, v)
    transform[6, 5] = 1.0
    out = transform @ cov @ transform.T
    return np.ascontiguousarray(0.5 * (out + out.T), dtype=np.float64)


def fibonacci_sphere_directions(
    n_fibonacci: int = 26,
    *,
    include_cardinal_axes: bool = True,
    include_equatorial_diagonals: bool = True,
) -> np.ndarray:
    """Return a deterministic, approximately uniform full-sphere seed set.

    Cardinal axes are inserted explicitly so exactly axial tracks are never
    absent from the library.  Equatorial diagonals reduce the largest gap in the
    plane where the old z charts were singular.
    """
    n = max(1, int(n_fibonacci))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    directions: list[np.ndarray] = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        radius = math.sqrt(max(1.0 - z * z, 0.0))
        phi = i * golden_angle
        directions.append(np.asarray([radius * math.cos(phi), radius * math.sin(phi), z]))
    if include_cardinal_axes:
        directions.extend(
            np.asarray(x, dtype=np.float64)
            for x in (
                (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
            )
        )
    if include_equatorial_diagonals:
        root = 1.0 / math.sqrt(2.0)
        directions.extend(
            np.asarray(x, dtype=np.float64)
            for x in (
                (root, root, 0.0), (root, -root, 0.0),
                (-root, root, 0.0), (-root, -root, 0.0),
            )
        )

    unique: list[np.ndarray] = []
    for item in directions:
        d = normalize_direction(item)
        if any(float(np.dot(d, previous)) > 1.0 - 1.0e-12 for previous in unique):
            continue
        unique.append(d)
    return np.ascontiguousarray(np.asarray(unique, dtype=np.float64))


def angular_separation_rad(a: Sequence[float], b: Sequence[float]) -> float:
    aa = normalize_direction(a)
    bb = normalize_direction(b)
    return math.acos(float(np.clip(np.dot(aa, bb), -1.0, 1.0)))


def nearest_direction_index(direction: Sequence[float], candidates: np.ndarray) -> int:
    d = normalize_direction(direction)
    c = np.asarray(candidates, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError("candidates must have shape (N,3)")
    return int(np.argmax(c @ d))
