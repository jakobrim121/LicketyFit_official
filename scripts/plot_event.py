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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVENT_DISPLAY_DIR = PROJECT_ROOT / "event_display"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "tables" / "wcsim_wcte_mapping.txt"


def _render_one_figure(figure) -> None:
    """Render this figure and nothing else that happens to be open.

    ``plt.show()`` under the Jupyter inline backend renders *every* open figure,
    not just the newest one.  Any figure another cell left open is therefore
    re-rendered by this call, so one ``plot_event`` call appears to emit a pile
    of unrelated plots -- including plain 1-D ones from elsewhere in the
    notebook.  Displaying this figure directly renders exactly one plot, and
    closing it afterwards stops the backend's end-of-cell flush from drawing it
    a second time.

    Outside the inline backend the conventional ``plt.show()`` is used: for a
    GUI backend, showing every open window is the expected behaviour, and under
    Agg it is a no-op that leaves the figure for the caller to save.
    """
    if "inline" in str(matplotlib.get_backend()).lower():
        try:
            from IPython.display import display as ipython_display
        except ImportError:
            ipython_display = None
        if ipython_display is not None:
            ipython_display(figure)
            plt.close(figure)
            return
    plt.show()


def _draw_event_display(
    display,
    display_values,
    *,
    vmax: float,
    vmin: float,
    log_scale: bool,
    color_label: str | None,
    style: str | None,
    show: bool,
):
    """Draw with ``EventDisplay`` and return the single figure the call created.

    ``EventDisplay.plotEventDisplay`` ends with ``plt.show()`` and applies its
    style with ``plt.style.use``.  Both are reasonable in a script and both need
    handling when the drawing is wrapped by a function that returns handles.

    Under the inline backend ``plt.show()`` renders every open figure and then
    *closes* it, so by the time ``plotEventDisplay`` returns there is no figure
    left, and a following ``plt.gcf()`` does not retrieve the event display --
    it silently creates a new, empty figure, which the backend then renders as a
    blank panel.  Wrapping ``plt.show`` takes the reference at the only moment
    the figure is still alive, and moves the decision about what to render here,
    where it can be limited to this call's own figure.

    ``plt.style.use`` mutates the global rcParams, so a dark-background event
    display would otherwise restyle every later plot in the notebook.  The
    figure is created and rendered inside ``rc_context``; the global defaults
    are restored on exit.
    """
    existing = set(plt.get_fignums())
    captured: dict[str, Any] = {}
    original_show = plt.show

    def capturing_show(*args, **kwargs):
        created = [num for num in plt.get_fignums() if num not in existing]
        if created:
            captured["figure"] = plt.figure(created[-1])
        return None

    with plt.rc_context():
        plt.show = capturing_show
        try:
            display.plotEventDisplay(
                display_values,
                vmax=float(vmax),
                vmin=float(vmin),
                log_scale=bool(log_scale),
                color_label=color_label,
                style=style,
            )
            created = [num for num in plt.get_fignums() if num not in existing]
            figure = captured.get("figure")
            if figure is None:
                if not created:
                    raise RuntimeError("The event display did not create a figure")
                figure = plt.figure(created[-1])
            # One call, one plot: discard anything else this call happened to
            # open, so it cannot surface later as a stray panel.
            for num in created:
                if num != figure.number:
                    plt.close(num)
            if show:
                _render_one_figure(figure)
        finally:
            plt.show = original_show
    return figure


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
        # A PreparedEvent already holds post-cut, PMT-aligned observables in PE
        # and the event-relative time reference, so label it as such rather than
        # as raw loader charge.
        prepared = hasattr(event, "prompt_min_ns")
        if prepared and key in {"expected_pe", "expected_time", "expected_time_ns"}:
            raise ValueError(
                f"A PreparedEvent has no model prediction, so {quantity!r} is not "
                "available; fit the event and plot the FitResult instead"
            )
        values = (
            np.asarray(event.times_ns, dtype=np.float64)
            if key in {"time", "observed_time", "observed_time_ns"}
            else np.asarray(event.charges, dtype=np.float64)
        )
        is_time = values is not None and key in {
            "time", "observed_time", "observed_time_ns"
        }
        if prepared:
            resolved = "observed_time" if is_time else "observed_pe"
        else:
            resolved = "time" if is_time else "charge"
        return (
            np.asarray(event.pmt_ids, dtype=np.int64),
            values,
            str(getattr(event, "pmt_id_mode", "wcte")),
            dict(getattr(event, "metadata", {}) or {}),
            resolved,
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
    show: bool = True,
):
    """Plot one raw, prepared, observed, or expected event on the WCTE display.

    Parameters
    ----------
    event
        ``EventRecord`` (raw loader output over the whole readout window),
        ``PreparedEvent`` (post-cut observables as the likelihood sees them),
        ``FitResult``, an LF-style array, a WCSim NPZ path, or a loaded WCSim
        mapping.  Prefer ``PreparedEvent`` for a display that matches the fit:
        an ``EventRecord`` has had no prompt-time or active-PMT cut applied and
        will look far busier than the data the fit used.
    event_index
        Required only for a multi-event WCSim NPZ/mapping.  The old positional
        ``evt_num`` use is therefore still accepted.
    quantity
        ``charge`` for a loaded event, or one of ``observed_pe``,
        ``expected_pe``, ``observed_time`` and ``expected_time``.  A
        ``PreparedEvent`` accepts the observed pair; only a ``FitResult`` has
        the expected pair.
    show
        Render the plot before returning, which is what makes it appear in a
        notebook cell.  Exactly one plot is produced: only the figure this call
        created is rendered, so figures other cells left open are not swept in.
        Pass ``False`` to keep the figure open and unrendered so it can be
        annotated or saved first, then render it yourself with
        ``figure.savefig(...)`` or, in a notebook,
        ``IPython.display.display(figure)`` -- note that a bare ``plt.show()``
        would also re-render every other figure that happens to be open.

    Returns
    -------
    tuple
        The ``(figure, axes)`` this call created; exactly one figure is created
        per call.  In a notebook the returned tuple is displayed as a repr like
        ``(<Figure ...>, <Axes: >)`` if the call is the last expression in the
        cell -- end that line with a semicolon to suppress it, as with
        ``plt.plot``.  With the default ``show=True`` the figure has already
        been rendered and, under the inline backend, closed; later changes to it
        will not appear.
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
    figure = _draw_event_display(
        display,
        display_values,
        vmax=vmax,
        vmin=vmin,
        log_scale=log_scale,
        color_label=color_label,
        style=style,
        show=bool(show),
    )
    axes = figure.axes[0] if figure.axes else None
    return figure, axes


__all__ = ["plot_event", "load_wcsim_to_wcte_mapping"]
