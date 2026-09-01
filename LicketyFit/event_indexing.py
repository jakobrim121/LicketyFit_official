"""Validated event-window indexing shared by LicketyFit batch adapters.

Batch fits use dense zero-based indices internally, while their inputs, truth
records, and output provenance retain source indices.  Keeping the window
calculation here prevents those two coordinate systems from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


class EventWindowError(ValueError):
    """Raised when a requested source-event window cannot be selected."""


def _nonnegative_integer(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise EventWindowError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class EventWindow:
    """Half-open source-event interval with dense local/source translation."""

    available_count: int
    start: int
    stop: int
    requested_count: int | None

    @property
    def count(self) -> int:
        return int(self.stop - self.start)

    @property
    def source_indices(self) -> range:
        return range(self.start, self.stop)

    def source_index(self, local_index: int) -> int:
        local = _nonnegative_integer(local_index, name="local_index")
        if local >= self.count:
            raise EventWindowError(
                f"local_index={local} is outside the selected window of "
                f"{self.count} events"
            )
        return int(self.start + local)


def resolve_event_window(
    available_count: int,
    start_index: int,
    requested_count: int | None,
    *,
    source_label: str = "input",
) -> EventWindow:
    """Return the requested half-open source interval, validating all bounds.

    ``requested_count=None`` means every event from ``start_index`` onward.
    A count larger than the remaining input is clipped at the input boundary.
    Empty inputs, starts at or beyond the boundary, and non-positive requested
    counts are rejected explicitly rather than silently fitting another event.
    """

    available = _nonnegative_integer(available_count, name="available_count")
    start = _nonnegative_integer(start_index, name="start_index")
    if requested_count is None:
        count = None
    else:
        count = _nonnegative_integer(requested_count, name="requested_count")
        if count == 0:
            raise EventWindowError("requested_count must be positive or None")

    label = str(source_label).strip() or "input"
    if available == 0:
        raise EventWindowError(f"No events are available in {label}")
    if start >= available:
        raise EventWindowError(
            f"start_index={start} is outside {label}, which contains "
            f"{available} events"
        )

    stop = available if count is None else min(available, start + count)
    return EventWindow(
        available_count=available,
        start=start,
        stop=int(stop),
        requested_count=count,
    )
