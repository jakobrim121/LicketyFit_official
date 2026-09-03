#!/usr/bin/env python3
"""Build LicketyFit's generated caches once, outside the notebook.

A first fit on a cold cache spends most of its wall time not fitting.  It
compiles the Numba kernels, compiles the small native scattering receiver,
mirrors the large range and receiver tables, builds the refined delta table, and
builds the seed proxy library for the requested configuration.  All of that is
cached, so it is paid once -- but a tutorial audience should not be the ones
paying it while they watch.

Run this once, in a terminal, with the same configuration the notebook uses.
It fits one or two throwaway events and discards the results.  The point is the
artifacts left behind, not the fit.

    python3 scripts/warm_up_cache.py wcsim --input /path/to/events.npz
    python3 scripts/warm_up_cache.py wcte --run 2079 --set particle_selection_label=pion

The configuration must match the notebook's, because the proxy library is keyed
by a digest over the physics settings, particle, seeding and endpoint modes.
Anything the launcher accepts can be set with repeated ``--set name=value``, so
an exact match is always expressible:

    python3 scripts/warm_up_cache.py wcsim --input events.npz \\
        --set seeding_mode=beam --set fit_particle=muon

Two caches behave differently and it is worth knowing which is which:

* The proxy library is written to ``tables/`` and is keyed only by content.  It
  is shareable -- build it once and ship it with the tutorial material, and no
  attendee rebuilds it.
* The runtime cache holding the compiled Numba kernels is private by design
  (mode 0700, ownership checked), so it cannot be shared between accounts.
  Every user runs this script once on their own account.

This must run as its own process.  The first execution of a freshly compiled
kernel can be both slow and numerically different from a cache-loaded one, which
is why the production driver also warms up in a disposable process.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _parse_setting(text: str) -> tuple[str, object]:
    """Parse ``name=value``, interpreting the value as a Python literal."""
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"--set expects name=value, got {text!r}"
        )
    name, _, raw = text.partition("=")
    name = name.strip()
    raw = raw.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"--set has an empty name: {text!r}")
    try:
        value: object = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        # Bare words are the common case: fit_particle=muon, seeding_mode=beam.
        value = raw
    return name, value


def _table_snapshot() -> dict[str, int]:
    tables = PROJECT_ROOT / "tables"
    if not tables.is_dir():
        return {}
    return {
        path.name: path.stat().st_size
        for path in tables.iterdir()
        if path.is_file()
    }


def _report_line(label: str, value: object) -> None:
    print(f"  {label:34} {value}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", choices=("wcsim", "wcte"))
    parser.add_argument(
        "--input",
        help="WCSim NPZ path; required for the wcsim source",
    )
    parser.add_argument(
        "--run", type=int, default=None,
        help="WCTE run number; required for the wcte source",
    )
    parser.add_argument(
        "--event", type=int, default=0,
        help="index of the event to fit (default: 0)",
    )
    parser.add_argument(
        "--events", type=int, default=2,
        help=(
            "how many events to fit. The default of 2 also measures a warm "
            "per-event time, which is what the notebook will feel afterwards. "
            "Use 1 to build the caches slightly faster."
        ),
    )
    parser.add_argument(
        "--publish", metavar="DIR",
        help=(
            "after warming, copy the compiled kernels and proxy libraries into "
            "DIR as a bundle other accounts can prime from. Run this once; "
            "attendees then copy instead of compiling."
        ),
    )
    parser.add_argument(
        "--set", dest="settings", action="append", default=[],
        metavar="NAME=VALUE",
        help=(
            "any launcher setting, repeatable; must match the notebook for the "
            "proxy library to be reused"
        ),
    )
    args = parser.parse_args()

    if args.source == "wcsim" and not args.input:
        parser.error("--input is required for the wcsim source")
    if args.source == "wcte" and args.run is None:
        parser.error("--run is required for the wcte source")
    if args.events < 1:
        parser.error("--events must be at least 1")
    if args.event < 0:
        parser.error("--event must be nonnegative")

    try:
        from single_event_fit import (
            SingleEventFitter,
            WCSimConfig,
            WCTEConfig,
            publish_cache_bundle,
            runtime_cache_report,
        )
    except ImportError as error:
        print(f"Could not import the notebook API: {error}", file=sys.stderr)
        print(
            "Run this from a checkout with its dependencies installed "
            "(python3 -m pip install -r requirements.txt).",
            file=sys.stderr,
        )
        return 2

    overrides: dict[str, object] = {}
    for item in args.settings:
        name, value = _parse_setting(item)
        overrides[name] = value

    if args.source == "wcsim":
        source_path = Path(args.input).expanduser()
        if not source_path.is_file():
            parser.error(f"--input is not a file: {source_path}")
        overrides.setdefault("input_file", str(source_path))
        config = WCSimConfig(**overrides)
    else:
        overrides.setdefault("run", int(args.run))
        config = WCTEConfig(**overrides)

    print("LicketyFit cache warm-up")
    print()
    print("Configuration (differences from the launcher defaults):")
    changed = config.changed_options()
    if changed:
        for name, value in changed.items():
            _report_line(name, value)
    else:
        _report_line("(none)", "using launcher defaults throughout")
    print()
    resolved = config.reconstruction()
    _report_line("engine", resolved.internal_engine_mode)
    _report_line("seeding / interaction", f"{resolved.seeding_mode} / {resolved.interaction_mode}")
    _report_line("likelihood", resolved.likelihood_mode)
    print()

    before = runtime_cache_report()
    print("Cache state before:")
    _report_line("runtime cache root", before.get("runtime_cache_root"))
    _report_line("persistent", before.get("runtime_cache_persistent"))
    _report_line("compiled numba entries", before.get("compiled_numba_entries"))
    _report_line("native receiver built", before.get("native_receiver_built"))
    _report_line("proxy libraries present", before.get("proxy_library_count"))
    if not before.get("runtime_cache_persistent", True):
        print()
        print(
            "  WARNING: the runtime cache is not in persistent storage, so this "
            "work\n           will be repeated on the next session. Set "
            "LF_RUNTIME_CACHE_DIR to a\n           directory that survives, then "
            "re-run this script."
        )
    print()

    tables_before = _table_snapshot()

    fitter = SingleEventFitter(config)

    print("Loading events...", flush=True)
    start = time.perf_counter()
    events = fitter.load_events(max_events=args.event + args.events)
    load_wall = time.perf_counter() - start
    if len(events) <= args.event:
        print(
            f"Only {len(events)} event(s) available; --event {args.event} is out "
            "of range.",
            file=sys.stderr,
        )
        return 1
    _report_line("loaded events", f"{len(events)} in {load_wall:.1f} s")
    print()

    print(
        "Initializing geometry, tables, seeds and the proxy library...\n"
        "  (this is the slow part; on a cold cache it can take a few minutes)",
        flush=True,
    )
    start = time.perf_counter()
    fitter.initialize()
    setup_wall = time.perf_counter() - start
    _report_line("setup", f"{setup_wall:.1f} s")
    print()

    available = min(args.events, len(events) - args.event)
    fit_walls: list[float] = []
    for offset in range(available):
        index = args.event + offset
        label = "first (cold kernels)" if offset == 0 else f"event {index}"
        print(f"Fitting {label}...", flush=True)
        start = time.perf_counter()
        result = fitter.fit(events[index])
        wall = time.perf_counter() - start
        fit_walls.append(wall)
        _report_line(
            f"{label} wall",
            f"{wall:.1f} s (accepted={result.fit_accepted}, fval={result.fval:.6g})",
        )
    print()

    after = runtime_cache_report()
    tables_after = _table_snapshot()
    created = sorted(set(tables_after) - set(tables_before))

    print("Cache state after:")
    _report_line("compiled numba entries", after.get("compiled_numba_entries"))
    _report_line("native receiver built", after.get("native_receiver_built"))
    _report_line("proxy libraries present", after.get("proxy_library_count"))
    if created:
        for name in created:
            _report_line("new file in tables/", f"{name} ({tables_after[name]/1e6:.1f} MB)")
    else:
        _report_line("new files in tables/", "none (everything was already built)")
    print()

    if args.publish:
        print("Publishing a cache bundle...")
        published = publish_cache_bundle(args.publish)
        for item in published["copied"]:
            _report_line("copied", item)
        for item in published["skipped"]:
            _report_line("skipped", item)
        _report_line("bundle", f"{published['bundle']} ({published['bundle_mb']} MB)")
        print()

    print("Summary:")
    _report_line("setup", f"{setup_wall:.1f} s")
    if fit_walls:
        _report_line("first fit", f"{fit_walls[0]:.1f} s")
    if len(fit_walls) > 1:
        warm = sum(fit_walls[1:]) / len(fit_walls[1:])
        _report_line("warm fit (mean of the rest)", f"{warm:.1f} s")
        _report_line(
            "what the notebook will feel",
            f"~{setup_wall:.0f} s setup once, then ~{warm:.0f} s per event",
        )
    else:
        _report_line(
            "warm per-event time",
            "not measured; re-run with --events 2 to measure it",
        )
    print()
    if args.publish:
        print(
            "Attendees can now skip the compile entirely. Point the notebook's\n"
            "TUTORIAL_CACHE_BUNDLE at the published directory, or call\n"
            "prime_cache_bundle(<dir>) directly. The bundle only has to be\n"
            "readable by them, not writable."
        )
    elif created:
        print(
            "The new tables/ file(s) above are shareable as they stand. To share\n"
            "the compiled kernels too, re-run with --publish DIR."
        )
    print("Done. Start the notebook in a fresh process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
