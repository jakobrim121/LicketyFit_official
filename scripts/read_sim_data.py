"""Read only the WCSim NPZ arrays needed by the caller.

The historical loader eagerly decompressed sixteen arrays, including large
photon-truth payloads, even though production fitting uses only digitized PMT
IDs, charges, and times. ``fields=None`` preserves the old public helper
contract; the production driver requests its three fit arrays explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np


FIT_FIELDS = (
    "digi_hit_pmt",
    "digi_hit_charge",
    "digi_hit_time",
)

TRUTH_TRACK_ID_FIELDS = (
    "track_id",
    "track_pid",
    "track_parent",
)

LEGACY_FIELDS = (
    "true_hit_pmt",
    "true_hit_time",
    "digi_hit_pmt",
    "digi_hit_time",
    "track_start_position",
    "track_stop_position",
    "track_id",
    "track_pid",
    "track_start_time",
    "track_parent",
    "digi_hit_charge",
    "position",
    "direction",
    "energy",
    "track_energy",
    "track_boundary_kes",
)


def read_sim_data(
    file_path: str | Path,
    *,
    fields: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load selected arrays from a WCSim NPZ and close the archive promptly."""
    requested = tuple(LEGACY_FIELDS if fields is None else fields)
    if not requested:
        raise ValueError("at least one WCSim NPZ field must be requested")
    if len(set(requested)) != len(requested):
        raise ValueError("WCSim NPZ field requests must not contain duplicates")

    path = Path(file_path).expanduser()
    with np.load(path, allow_pickle=True) as archive:
        missing = [name for name in requested if name not in archive.files]
        if missing:
            raise KeyError(
                f"WCSim NPZ {path} is missing required arrays: {missing}"
            )
        return {name: archive[name] for name in requested}
