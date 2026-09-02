#!/usr/bin/env python3
"""Populate LicketyFit's source-addressed runtime caches, then exit.

Run this command once in a disposable process after changing LicketyFit source
or the Python ABI.  It deliberately discards the fitted bootstrap events.  The
subsequent production launcher starts in a fresh process and loads the compiled
Numba/native kernels used by the release fingerprint gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = PROJECT_ROOT / "benchmarks" / "benchmark_single_event.py"


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _command(
    args: argparse.Namespace, seeding_mode: str, interaction_mode: str
) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK),
        args.source,
        "--input",
        str(args.input),
        "--seeding-mode",
        seeding_mode,
        "--interaction-mode",
        interaction_mode,
        "--events",
        str(args.event),
        "--warmup-event",
        str(args.warmup_event),
    ]
    if args.source == "wcte":
        command.extend(
            [
                "--good-pmt-root",
                str(args.good_pmt_root),
                "--run",
                str(args.run),
                "--root-entries",
                str(args.root_entries),
                "--beam-momentum",
                str(args.beam_momentum),
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("wcsim", "wcte"))
    parser.add_argument("--input", type=_existing_file, required=True)
    parser.add_argument(
        "--seeding-mode",
        choices=("general", "beam", "both"),
        default="both",
        help="seeding path(s) whose runtime kernels should be compiled",
    )
    parser.add_argument(
        "--interaction-mode",
        choices=("full_length", "absorption", "both"),
        default="full_length",
        help="endpoint path(s) whose runtime kernels should be compiled",
    )
    parser.add_argument(
        "--warmup-event",
        type=int,
        default=0,
        help="selected/source event used by the benchmark's first full fit",
    )
    parser.add_argument(
        "--event",
        type=int,
        default=1,
        help="second event fitted in the disposable bootstrap process",
    )
    parser.add_argument("--good-pmt-root", type=_existing_file)
    parser.add_argument("--run", type=int, default=1775)
    parser.add_argument("--root-entries", type=int, default=2000)
    parser.add_argument("--beam-momentum", type=float, default=260.0)
    args = parser.parse_args()

    if args.warmup_event < 0 or args.event < 0:
        parser.error("--warmup-event and --event must be nonnegative")
    if args.source == "wcte" and args.good_pmt_root is None:
        parser.error("--good-pmt-root is required for WCTE selection")

    seedings = (
        ("general", "beam")
        if args.seeding_mode == "both" else (args.seeding_mode,)
    )
    interactions = (
        ("full_length", "absorption")
        if args.interaction_mode == "both" else (args.interaction_mode,)
    )
    for seeding_mode in seedings:
        for interaction_mode in interactions:
            print(
                f"Bootstrapping {args.source} {seeding_mode}+"
                f"{interaction_mode} runtime kernels...",
                flush=True,
            )
            subprocess.run(
                _command(args, seeding_mode, interaction_mode),
                cwd=PROJECT_ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
    print(
        "Runtime caches are populated. Start the production fitter in a new "
        "process.",
        flush=True,
    )


if __name__ == "__main__":
    main()
