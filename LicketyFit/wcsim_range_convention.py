"""Explicit WCSim reporting conversion for the visible-range convention.

The reconstruction's internal range coordinate follows its frozen CSDA table.
WCSim's primary endpoint and above-Cherenkov-threshold arc use a slightly
different convention whose offset grows with energy.  This module applies a
separately calibrated conversion only to reported WCSim lengths.  It never
changes the likelihood, optimizer, MCS path, or WCTE-data output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "licketyfit-wcsim-visible-range-convention-v1"


def canonical_calibration_id(manifest: Mapping[str, Any]) -> str:
    core = dict(manifest)
    core.pop("calibration_id", None)
    payload = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class WCSimVisibleRangeConvention:
    calibration_id: str
    slope_mm_per_mev: float
    intercept_mm: float
    kinetic_energy_domain_mev: tuple[float, float]
    manifest_path: str

    def correction_mm(self, kinetic_energy_mev, *, allow_extrapolation=False):
        energy = np.asarray(kinetic_energy_mev, dtype=np.float64)
        if np.any(~np.isfinite(energy)):
            raise ValueError("range-equivalent kinetic energy must be finite")
        low, high = self.kinetic_energy_domain_mev
        if not allow_extrapolation and np.any((energy < low) | (energy > high)):
            raise ValueError(
                "WCSim visible-range calibration is valid only on "
                f"[{low:g}, {high:g}] MeV"
            )
        correction = self.slope_mm_per_mev * energy + self.intercept_mm
        if np.any(correction < 0.0):
            raise ValueError("WCSim visible-range correction became negative")
        return float(correction) if correction.ndim == 0 else correction

    def corrected_visible_length_mm(
        self,
        raw_visible_length_mm,
        range_equivalent_kinetic_energy_mev,
        *,
        allow_extrapolation=False,
    ):
        raw = np.asarray(raw_visible_length_mm, dtype=np.float64)
        correction = np.asarray(
            self.correction_mm(
                range_equivalent_kinetic_energy_mev,
                allow_extrapolation=allow_extrapolation,
            ),
            dtype=np.float64,
        )
        raw, correction = np.broadcast_arrays(raw, correction)
        corrected = raw - correction
        if np.any(~np.isfinite(corrected)) or np.any(corrected < 0.0):
            raise ValueError("calibrated WCSim visible length is invalid")
        return float(corrected) if corrected.ndim == 0 else corrected

    def provenance(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "scope": "WCSim reporting only; likelihood and WCTE data unchanged",
            "calibration_id": self.calibration_id,
            "manifest_path": self.manifest_path,
            "kinetic_energy_domain_mev": list(self.kinetic_energy_domain_mev),
            "correction_model": {
                "kind": "linear_in_range_equivalent_kinetic_energy",
                "slope_mm_per_mev": self.slope_mm_per_mev,
                "intercept_mm": self.intercept_mm,
            },
        }


def load_wcsim_visible_range_convention(
    path: str | Path,
) -> WCSimVisibleRangeConvention:
    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read WCSim visible-range manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported WCSim visible-range manifest schema")
    if manifest.get("allowed_scope") != "wcsim_reporting_only":
        raise ValueError("WCSim range calibration has an invalid application scope")
    if manifest.get("changes_likelihood") is not False:
        raise ValueError("WCSim range calibration must not change the likelihood")
    if manifest.get("applies_to_wcte_data") is not False:
        raise ValueError("WCSim range calibration must not apply to WCTE data")
    if manifest.get("particle") != "muon":
        raise ValueError("WCSim range calibration is not a muon calibration")
    calibration_id = manifest.get("calibration_id")
    if (
        not isinstance(calibration_id, str)
        or calibration_id != canonical_calibration_id(manifest)
    ):
        raise ValueError("WCSim range calibration ID does not match its contents")
    model = manifest.get("correction_model")
    if not isinstance(model, Mapping) or model.get("kind") != "linear_in_range_equivalent_kinetic_energy":
        raise ValueError("unsupported WCSim range correction model")
    slope = float(model.get("slope_mm_per_mev", math.nan))
    intercept = float(model.get("intercept_mm", math.nan))
    domain = manifest.get("kinetic_energy_domain_mev")
    if not (
        math.isfinite(slope)
        and slope > 0.0
        and math.isfinite(intercept)
        and isinstance(domain, list)
        and len(domain) == 2
    ):
        raise ValueError("invalid WCSim range correction coefficients/domain")
    low, high = map(float, domain)
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        raise ValueError("invalid WCSim range calibration energy domain")
    return WCSimVisibleRangeConvention(
        calibration_id=calibration_id,
        slope_mm_per_mev=slope,
        intercept_mm=intercept,
        kinetic_energy_domain_mev=(low, high),
        manifest_path=str(manifest_path),
    )


__all__ = [
    "SCHEMA",
    "WCSimVisibleRangeConvention",
    "canonical_calibration_id",
    "load_wcsim_visible_range_convention",
]
