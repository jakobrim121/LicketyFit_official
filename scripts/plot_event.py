"""Portable WCTE/WCSim event-display helper for the example notebook.

The plotting assets are resolved relative to the repository, not the notebook's
working directory.  ``EventRecord`` and ``FitResult`` objects from
``single_event_fit.py`` are accepted directly; the historical array/path calling
style remains available for simple scripts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVENT_DISPLAY_DIR = PROJECT_ROOT / "event_display"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "tables" / "wcsim_wcte_mapping.txt"


@lru_cache(maxsize=1)
def _event_display():
    if str(EVENT_DISPLAY_DIR) not in sys.path:
        sys.path.insert(0, str(EVENT_DISPLAY_DIR))
    from EventDisplay import EventDisplay

    display = EventDisplay()
    display.load_mPMT_positions("mPMT_2D_projection_angles.csv")
    return display


@lru_cache(maxsize=8)
def load_wcsim_to_wcte_mapping(mapping_path: str | Path = DEFAULT_MAPPING_PATH) -> dict[int, int]:
    """Return ``WCSim tube number -> 100*slot + zero-based PMT position``."""
    path = Path(mapping_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    table = np.atleast_2d(np.loadtxt(path))
    return {
        int(row[0]): int(row[1]) * 100 + int(row[2]) - 1
        for row in table
    }


def _mapped_wcsim_ids(
    raw_ids: np.ndarray,
    *,
    mapping_path: str | Path,
    pmt_id_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    mapping = load_wcsim_to_wcte_mapping(mapping_path)
    mapped: list[int] = []
    kept: list[int] = []
    for index, raw in enumerate(np.asarray(raw_ids, dtype=np.int64)):
        value = mapping.get(int(raw) + int(pmt_id_offset))
        if value is not None:
            mapped.append(int(value))
            kept.append(index)
    return np.asarray(mapped, dtype=np.int64), np.asarray(kept, dtype=np.int64)


def _from_path(path: str | Path, event_index: int | None):
    source = Path(path).expanduser()
    loaded = np.load(source, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile) and {
        "digi_hit_pmt", "digi_hit_charge", "digi_hit_time"
    }.issubset(loaded.files):
        if event_index is None:
            raise ValueError("event_index is required for a WCSim NPZ")
        return (
            np.asarray(loaded["digi_hit_pmt"][event_index], dtype=np.int64),
            np.asarray(loaded["digi_hit_charge"][event_index], dtype=np.float64),
            np.asarray(loaded["digi_hit_time"][event_index], dtype=np.float64),
            "wcsim",
            {},
        )
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if len(loaded.files) != 1:
            raise ValueError(
                "A non-WCSim NPZ must contain exactly one event-array key"
            )
        loaded = loaded[loaded.files[0]]
    array = np.asarray(loaded)
    if array.ndim == 3:
        if event_index is None:
            raise ValueError("event_index is required for a multi-event array")
        array = np.asarray(array[int(event_index)], dtype=np.float64)
    elif array.ndim == 1 and array.dtype == object:
        if event_index is None:
            raise ValueError("event_index is required for an object event array")
        array = np.asarray(array[int(event_index)], dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("Array event input must have columns [PMT id, value, optional time]")
    times = (
        np.asarray(array[:, 2], dtype=np.float64)
        if array.shape[1] >= 3
        else np.full(array.shape[0], np.nan, dtype=np.float64)
    )
    return (
        np.asarray(array[:, 0], dtype=np.int64),
        np.asarray(array[:, 1], dtype=np.float64),
        times,
        "wcte",
        {},
    )


def _extract(event: Any, event_index: int | None, quantity: str):
    key = str(quantity).strip().lower()
    if hasattr(event, "observed_pe") and hasattr(event, "expected_pe"):
        choices = {
            "observed_pe": np.asarray(event.observed_pe, dtype=np.float64),
            "expected_pe": np.asarray(event.expected_pe, dtype=np.float64),
            "observed_time": np.asarray(event.observed_time_ns, dtype=np.float64),
            "expected_time": np.asarray(event.expected_time_ns, dtype=np.float64),
        }
        aliases = {
            "charge": "observed_pe", "pe": "observed_pe",
            "time": "observed_time", "observed_time_ns": "observed_time",
            "expected_time_ns": "expected_time",
        }
        key = aliases.get(key, key)
        if key not in choices:
            raise ValueError(f"Unknown FitResult quantity {quantity!r}; choose {sorted(choices)}")
        return (
            np.asarray(event.pmt_ids, dtype=np.int64),
            choices[key],
            "wcte",
            {},
            key,
        )

    if hasattr(event, "pmt_ids") and hasattr(event, "charges") and hasattr(event, "times_ns"):
        values = (
            np.asarray(event.times_ns, dtype=np.float64)
            if key in {"time", "observed_time", "observed_time_ns"}
            else np.asarray(event.charges, dtype=np.float64)
        )
        return (
            np.asarray(event.pmt_ids, dtype=np.int64),
            values,
            str(getattr(event, "pmt_id_mode", "wcte")),
            dict(getattr(event, "metadata", {}) or {}),
            "time" if key.startswith("time") or "time" in key else "charge",
        )

    if isinstance(event, (str, Path)):
        pmt_ids, charges, times, mode, metadata = _from_path(event, event_index)
        values = times if key in {"time", "observed_time", "observed_time_ns"} else charges
        return pmt_ids, values, mode, metadata, ("time" if values is times else "charge")

    if isinstance(event, Mapping) and {
        "digi_hit_pmt", "digi_hit_charge", "digi_hit_time"
    }.issubset(event):
        if event_index is None:
            raise ValueError("event_index is required for a WCSim mapping")
        values = (
            np.asarray(event["digi_hit_time"][event_index], dtype=np.float64)
            if key in {"time", "observed_time", "observed_time_ns"}
            else np.asarray(event["digi_hit_charge"][event_index], dtype=np.float64)
        )
        return (
            np.asarray(event["digi_hit_pmt"][event_index], dtype=np.int64),
            values,
            "wcsim",
            {},
            "time" if "time" in key else "charge",
        )

    array = np.asarray(event)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("Event arrays must have columns [WCTE PMT id, value]")
    return (
        np.asarray(array[:, 0], dtype=np.int64),
        np.asarray(array[:, 1], dtype=np.float64),
        "wcte",
        {},
        key,
    )


def plot_event(
    event: Any,
    event_index: int | None = None,
    vmax: float | None = None,
    vmin: float | None = None,
    log_scale: bool = False,
    LF_data: bool | None = None,
    *,
    quantity: str = "charge",
    pmt_id_mode: str | None = None,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    pmt_id_offset: int = 1,
    style: str | None = "dark_background",
    color_label: str | None = None,
    mask_slots: tuple[int, ...] | list[int] = (),
):
    """Plot one raw, loaded, observed, or expected event on the WCTE display.

    Parameters
    ----------
    event
        ``EventRecord``/``FitResult``, an LF-style array, a WCSim NPZ path, or
        a loaded WCSim mapping.
    event_index
        Required only for a multi-event WCSim NPZ/mapping.  The old positional
        ``evt_num`` use is therefore still accepted.
    quantity
        ``charge`` for a loaded event, or one of ``observed_pe``,
        ``expected_pe``, ``observed_time`` and ``expected_time`` for a fit result.
    """
    pmt_ids, values, inferred_mode, metadata, resolved_quantity = _extract(
        event, event_index, quantity
    )
    mode = str(pmt_id_mode or inferred_mode).strip().lower()
    if LF_data is not None:
        mode = "wcte" if bool(LF_data) else "wcsim"

    if mode in {"mapping", "wcsim", "raw"}:
        selected_mapping = metadata.get("mapping_path", mapping_path)
        selected_offset = int(metadata.get("pmt_id_offset", pmt_id_offset))
        pmt_ids, keep = _mapped_wcsim_ids(
            pmt_ids,
            mapping_path=selected_mapping,
            pmt_id_offset=selected_offset,
        )
        values = values[keep]
    elif mode not in {"wcte", "mapped"}:
        raise ValueError("pmt_id_mode must be 'wcte' or 'wcsim'")

    finite = np.isfinite(values)
    pmt_ids = pmt_ids[finite]
    values = np.asarray(values[finite], dtype=np.float64)
    slots = pmt_ids // 100
    positions = pmt_ids % 100
    valid_ids = (slots >= 0) & (slots < 106) & (positions >= 0) & (positions < 19)
    slots, positions, values = slots[valid_ids], positions[valid_ids], values[valid_ids]
    if values.size == 0:
        raise ValueError("No finite, displayable PMT values remain")

    display = _event_display()
    display.mask_mPMTs([int(value) for value in mask_slots] or None)
    sum_data = "time" not in str(resolved_quantity)
    display_values = display.process_data(slots, positions, values, sum_data=sum_data)

    finite_display = display_values[np.isfinite(display_values)]
    positive = finite_display[finite_display > 0.0]
    if vmax is None:
        vmax = float(np.nanmax(finite_display)) if finite_display.size else 1.0
    if vmin is None:
        if log_scale:
            vmin = float(np.nanmin(positive)) if positive.size else 1.0e-3
        else:
            vmin = float(np.nanmin(finite_display)) if finite_display.size else 0.0
    if float(vmax) <= float(vmin):
        vmax = float(vmin) + max(1.0e-12, abs(float(vmin)) * 1.0e-6)

    if color_label is None:
        color_label = {
            "charge": "charge",
            "observed_pe": "observed PE",
            "expected_pe": "expected PE",
            "observed_time": "observed time [ns]",
            "expected_time": "expected time [ns]",
        }.get(str(resolved_quantity), str(resolved_quantity))
    display.plotEventDisplay(
        display_values,
        vmax=float(vmax),
        vmin=float(vmin),
        log_scale=bool(log_scale),
        color_label=color_label,
        style=style,
    )
    return plt.gcf(), plt.gca()


__all__ = ["plot_event", "load_wcsim_to_wcte_mapping"]
