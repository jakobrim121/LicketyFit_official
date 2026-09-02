"""Small, dependency-free console helpers for public LicketyFit runs.

This module deliberately imports no numerical packages.  The launchers and the
lightweight multiprocessing supervisor can therefore format status messages
without initializing NumPy, BLAS, Numba, or any fitter state.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import re
import subprocess
import sys


DIVIDER = "-" * 60
DEFAULT_PROGRESS_INTERVAL = 50
_PROGRESS_LINE = re.compile(r"^Completed [0-9]+(?:/[0-9]+)? fits\.$")
_IMPORTANT_LINE = re.compile(
    r"(?:warning|error|failed|exception|traceback|fallback)", re.IGNORECASE
)


def print_welcome(seeding_mode: str, interaction_mode: str) -> None:
    """Print the public two-axis reconstruction banner."""
    print(DIVIDER)
    print("Welcome to LicketyFit!")
    print()
    print("(Where the fits are fast and the physics is questionable)")
    print()
    print(f"Mode: {seeding_mode}, {interaction_mode}")
    print(DIVIDER)


def print_details(items: Iterable[tuple[str, object]]) -> None:
    """Print a compact aligned run summary."""
    rows = [(str(label), str(value)) for label, value in items]
    if not rows:
        return
    width = max(len(label) for label, _ in rows)
    print()
    for label, value in rows:
        print(f"{label:<{width}} : {value}")


def print_preparation_notice() -> None:
    """Describe the only routine pre-fit pause shown by the simple console."""
    print()
    print(
        "One-time warm-up underway; fits will begin automatically when ready.",
        flush=True,
    )


def progress_milestones(
    previous_completed: int,
    completed: int,
    *,
    total: int | None,
    interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> tuple[int, ...]:
    """Return newly crossed positive progress milestones.

    ``completed`` may jump by more than one when a multiprocessing supervisor
    observes an atomic checkpoint.  Returning every crossed boundary keeps the
    user-visible contract independent of worker count and checkpoint cadence.
    """
    step = int(interval)
    if step <= 0:
        return ()
    before = max(0, int(previous_completed))
    after = max(before, int(completed))
    upper = after if total is None else min(after, max(0, int(total)))
    first = ((before // step) + 1) * step
    if first > upper:
        return ()
    return tuple(range(first, upper + 1, step))


def print_progress(
    previous_completed: int,
    completed: int,
    *,
    total: int | None,
    interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> None:
    """Print one timeless status line for each newly crossed milestone."""
    for milestone in progress_milestones(
        previous_completed,
        completed,
        total=total,
        interval=interval,
    ):
        suffix = "" if total is None else f"/{int(total)}"
        print(f"Completed {milestone}{suffix} fits.", flush=True)


def print_goodbye(output_file: object) -> None:
    """Print the successful-run footer owned by the public launcher."""
    print()
    print(DIVIDER, flush=True)
    print("LicketyFit finished successfully.")
    print(f"Output: {output_file}")
    print("Thanks for using LicketyFit. Goodbye!")
    print(DIVIDER)


def run_with_simple_console(
    command: list[str],
    *,
    environment: dict[str, str],
) -> int:
    """Run the numerical driver while exposing only progress and diagnostics."""
    child_environment = dict(environment)
    child_environment.setdefault("PYTHONUNBUFFERED", "1")
    recent_output: deque[str] = deque(maxlen=200)
    process = subprocess.Popen(
        command,
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            recent_output.append(line)
            stripped = line.strip()
            if _PROGRESS_LINE.fullmatch(stripped):
                print(stripped, flush=True)
            elif _IMPORTANT_LINE.search(stripped):
                print(stripped, file=sys.stderr, flush=True)
        return_code = int(process.wait())
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()

    if return_code != 0:
        print(
            f"LicketyFit driver exited with status {return_code}. Recent output:",
            file=sys.stderr,
        )
        for line in recent_output:
            print(line, file=sys.stderr)
    return return_code
