"""Validated user-event file support for real-WCTE LicketyFit fits.

The public batch driver accepts either production-selected ROOT events or an
already-selected user file.  This module keeps the file contract in one place
so the standard and cosmic engines cannot drift apart.

Supported event containers are ``.npy``, ``.npz``, ``.pkl`` and ``.pickle``.
An event is an ``N x 3``, ``N x 4`` or ``N x 5`` numeric table:

``[global_pmt_id, charge_adc, calibrated_time_ns]``
    No embedded identity; the selected-list index is used.

``[..., legacy_event_identity]``
    The one legacy identity is exposed as both event number and ROOT-entry
    identity.  Output metadata marks this alias explicitly.

``[..., root_entry_index, production_event_number]``
    Recommended format preserving the two identities independently.

Object-array NPY files and pickle files use Python pickle internally and must
therefore come from a trusted source.  Flat numeric NPY/NPZ files avoid that
requirement and are preferred for data exchange.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np


_KNOWN_EVENT_KEYS = ("events", "data", "arr_0")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_column(values: np.ndarray, *, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} contains non-finite values")
    rounded = np.rint(values)
    if not np.all(values == rounded):
        bad = values[values != rounded][:5].tolist()
        raise ValueError(f"{label} must be integer-valued; examples: {bad}")
    return rounded.astype(np.int64)


def _constant_identity(values: np.ndarray, *, label: str) -> None:
    identities = _integer_column(values, label=label)
    if identities.size and np.any(identities != identities[0]):
        unique = np.unique(identities)
        preview = unique[:8].tolist()
        raise ValueError(
            f"{label} must be constant within one event; found {preview}"
        )


def coerce_event_array(
    event: Any,
    *,
    event_label: str = "event",
    strict: bool = True,
) -> np.ndarray:
    """Return one normalized event and enforce the public hit-table contract."""
    array = np.asarray(event)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(
            f"{event_label} must be a 2D array with at least three columns: "
            "[WCTE PMT ID, charge, calibrated time]."
        )
    n_columns = 5 if array.shape[1] >= 5 else (4 if array.shape[1] >= 4 else 3)
    try:
        result = np.asarray(array[:, :n_columns], dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"{event_label} is not a numeric hit table") from exc

    if not strict or result.shape[0] == 0:
        return result

    pmt_ids = _integer_column(result[:, 0], label=f"{event_label} PMT-ID column")
    if np.any(pmt_ids < 0):
        raise ValueError(f"{event_label} contains negative PMT IDs")
    positions = np.mod(pmt_ids, 100)
    if np.any(positions > 18):
        bad = pmt_ids[positions > 18][:5].tolist()
        raise ValueError(
            f"{event_label} contains IDs that are not WCTE slot*100+position "
            f"with position 0..18; examples: {bad}"
        )

    charges = result[:, 1]
    times = result[:, 2]
    if not np.all(np.isfinite(charges)):
        raise ValueError(f"{event_label} charge column contains non-finite values")
    if np.any(charges < 0.0):
        bad = charges[charges < 0.0][:5].tolist()
        raise ValueError(f"{event_label} contains negative charges; examples: {bad}")
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{event_label} calibrated-time column contains non-finite values")

    if n_columns >= 4:
        _constant_identity(
            result[:, 3], label=f"{event_label} ROOT/legacy identity column"
        )
    if n_columns >= 5:
        _constant_identity(
            result[:, 4], label=f"{event_label} production event-number column"
        )
    return result


def _choose_mapping_payload(
    payload: Mapping[str, Any], *, user_event_key: str | None
) -> tuple[Any, str]:
    if user_event_key is not None:
        if user_event_key not in payload:
            raise KeyError(
                f"USER_EVENT_KEY={user_event_key!r} is absent; available keys "
                f"are {list(payload.keys())}"
            )
        return payload[user_event_key], str(user_event_key)
    for key in _KNOWN_EVENT_KEYS:
        if key in payload:
            return payload[key], key
    if len(payload) == 1:
        key = next(iter(payload))
        return payload[key], str(key)
    raise KeyError(
        "Could not choose an event array from the dictionary payload; "
        f"available keys are {list(payload.keys())}. Set USER_EVENT_KEY."
    )


def _choose_npz_payload(
    payload: np.lib.npyio.NpzFile, *, user_event_key: str | None
) -> tuple[Any, str]:
    files = list(payload.files)
    if user_event_key is not None:
        key = str(user_event_key)
        if key not in files:
            raise KeyError(
                f"USER_EVENT_KEY={key!r} is absent; available keys are {files}"
            )
        return payload[key], key
    for key in _KNOWN_EVENT_KEYS:
        if key in files:
            return payload[key], key
    if len(files) == 1:
        return payload[files[0]], files[0]
    raise KeyError(
        "Could not choose an event array from the npz payload; "
        f"available keys are {files}. Set USER_EVENT_KEY."
    )


def events_from_loaded_object(
    obj: Any,
    *,
    user_event_key: str | None = None,
    strict: bool = True,
) -> tuple[list[np.ndarray], str | None, bool]:
    """Normalize one loaded container.

    Returns ``(events, selected_key, pickle_backed_payload)``.
    """
    selected_key: str | None = None
    if isinstance(obj, Mapping):
        obj, selected_key = _choose_mapping_payload(
            obj, user_event_key=user_event_key
        )
    elif isinstance(obj, np.lib.npyio.NpzFile):
        obj, selected_key = _choose_npz_payload(
            obj, user_event_key=user_event_key
        )

    pickle_backed = bool(getattr(obj, "dtype", None) == object)
    if isinstance(obj, (list, tuple)):
        pickle_backed = True
        return (
            [
                coerce_event_array(
                    event, event_label=f"event[{index}]", strict=strict
                )
                for index, event in enumerate(obj)
            ],
            selected_key,
            pickle_backed,
        )

    array = np.asarray(
        obj, dtype=object if getattr(obj, "dtype", None) == object else None
    )
    if array.dtype == object and array.ndim == 1:
        return (
            [
                coerce_event_array(
                    event, event_label=f"event[{index}]", strict=strict
                )
                for index, event in enumerate(array)
            ],
            selected_key,
            True,
        )
    if array.ndim == 3:
        return (
            [
                coerce_event_array(
                    array[index], event_label=f"event[{index}]", strict=strict
                )
                for index in range(array.shape[0])
            ],
            selected_key,
            pickle_backed,
        )
    if array.ndim == 2:
        try:
            numeric = np.asarray(array, dtype=np.float64)
        except Exception as exc:
            raise ValueError("The user hit table is not numeric") from exc
        if numeric.shape[1] >= 4:
            event_keys = _integer_column(
                numeric[:, 3], label="user event grouping-identity column"
            )
            ordered_ids: list[int] = []
            seen: set[int] = set()
            for value in event_keys:
                key = int(value)
                if key not in seen:
                    seen.add(key)
                    ordered_ids.append(key)
            return (
                [
                    coerce_event_array(
                        numeric[event_keys == event_id],
                        event_label=f"event_identity={event_id}",
                        strict=strict,
                    )
                    for event_id in ordered_ids
                ],
                selected_key,
                pickle_backed,
            )
        return (
            [coerce_event_array(numeric, event_label="single_event", strict=strict)],
            selected_key,
            pickle_backed,
        )
    raise ValueError(
        "Unsupported USER_EVENT_FILE payload. Expected a list/object array of "
        "event arrays, a 3D event array, or a 2D hit table."
    )


def identity_schema_metadata(events: list[np.ndarray]) -> dict[str, Any]:
    widths = sorted({int(event.shape[1]) for event in events})
    if widths == [5]:
        schema = "five_column_distinct_root_entry_and_event_number"
        event_column = 4
        root_column = 3
        alias = False
    elif widths == [4]:
        schema = "legacy_four_column_aliased_identity"
        event_column = 3
        root_column = 3
        alias = True
    elif widths == [3]:
        schema = "three_column_selected_list_index_fallback"
        event_column = None
        root_column = None
        alias = True
    else:
        schema = "mixed_supported_event_schemas"
        event_column = None
        root_column = None
        alias = None
    return {
        "schema": schema,
        "column_counts": widths,
        "source_event_id_column": event_column,
        "source_root_entry_index_column": root_column,
        "legacy_identity_aliased": alias,
        "identity_note": (
            "In four-column legacy events, the only supplied identity is "
            "reported as both source_event_id and source_root_entry_index."
            if schema == "legacy_four_column_aliased_identity"
            else None
        ),
    }


def load_user_event_file(
    path: str | Path,
    *,
    user_event_key: str | None = None,
    max_events: int | None = None,
    strict: bool = True,
    trusted_internal: bool = False,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Load and validate a user-selected WCTE event file."""
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file():
        raise ValueError(f"USER_EVENT_FILE is not a regular file: {source}")
    suffix = source.suffix.lower()
    explicit_pickle_container = suffix in {".pkl", ".pickle"}
    if suffix == ".npz":
        loaded: Any = np.load(source, allow_pickle=True)
    elif suffix == ".npy":
        loaded = np.load(source, allow_pickle=True)
    elif explicit_pickle_container:
        with source.open("rb") as stream:
            loaded = pickle.load(stream)
    else:
        raise ValueError(
            f"Unsupported USER_EVENT_FILE suffix {suffix!r}; use npy, npz, "
            "pkl, or pickle"
        )

    try:
        events, selected_key, object_pickle = events_from_loaded_object(
            loaded, user_event_key=user_event_key, strict=strict
        )
    finally:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()

    if max_events is not None:
        limit = int(max_events)
        if limit < 0:
            raise ValueError("MAX_EVENTS_TO_FIT must be non-negative or unset")
        events = events[:limit]
    pickle_backed = bool(explicit_pickle_container or object_pickle)
    if pickle_backed and not trusted_internal:
        print(
            "WARNING: USER_EVENT_FILE uses Python pickle/object-array storage; "
            "load it only when the file is trusted. Prefer a flat numeric NPY "
            "or NPZ table for untrusted exchange.",
            flush=True,
        )

    metadata: dict[str, Any] = {
        "adapter": "LicketyFit.scripts.wcte_user_event_file",
        "path": str(source),
        "suffix": suffix,
        "selected_key": selected_key,
        "strict_validation": bool(strict),
        "pickle_backed": pickle_backed,
        "trusted_internal_handoff": bool(trusted_internal),
        "file_size_bytes": int(source.stat().st_size),
        "sha256": None if trusted_internal else _sha256_file(source),
        "events_returned": int(len(events)),
        "total_hit_rows": int(sum(event.shape[0] for event in events)),
        "hit_table_columns": [
            "global_wcte_pmt_id",
            "hit_pmt_charges_adc",
            "hit_pmt_calibrated_times_ns",
            "optional_root_or_legacy_identity",
            "optional_production_event_number",
        ],
        "identity": identity_schema_metadata(events),
    }
    return events, metadata


def source_event_id(event: np.ndarray, fallback_index: int) -> int:
    if event.shape[0] > 0:
        column = 4 if event.shape[1] >= 5 else (3 if event.shape[1] >= 4 else None)
        if column is not None:
            return int(event[0, column])
    return int(fallback_index)


def source_root_entry_index(event: np.ndarray, fallback_index: int) -> int:
    if event.shape[0] > 0 and event.shape[1] >= 4:
        return int(event[0, 3])
    return int(fallback_index)


__all__ = [
    "coerce_event_array",
    "events_from_loaded_object",
    "identity_schema_metadata",
    "load_user_event_file",
    "source_event_id",
    "source_root_entry_index",
]
