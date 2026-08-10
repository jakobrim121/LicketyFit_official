#!/usr/bin/env python3
"""Contract tests for WCTE user events and independent good-PMT masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import tempfile

import numpy as np

from wcte_data_loader_adapter import (
    authoritative_active_wcte_pmts,
    load_good_wcte_pmts_file,
    resolve_good_wcte_pmts,
)
from wcte_user_event_file import load_user_event_file


def _expect_error(callable_, text: str) -> None:
    try:
        callable_()
    except Exception as exc:
        if text.lower() not in str(exc).lower():
            raise AssertionError(
                f"Expected error containing {text!r}, got {exc!r}"
            ) from exc
        return
    raise AssertionError(f"Expected an error containing {text!r}")


def _write_fake_dq_root(path: Path, ids: list[int]) -> None:
    import awkward as ak
    import uproot

    slots = [value // 100 for value in ids]
    positions = [value % 100 for value in ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as root:
        root["Configuration"] = {
            "good_wcte_pmts_slot": ak.Array([slots]),
            "good_wcte_pmts_position": ak.Array([positions]),
        }


def run_contract_tests(supplied_event_file: Path | None = None) -> None:
    # Slot 27 was historically disabled for WCSim.  It must remain active when
    # an authoritative real-WCTE user/DQ list includes it.
    historical_wcsim_inactive_id = 2700
    expected = {0, 1, 100, 118, 204, historical_wcsim_inactive_id}
    with tempfile.TemporaryDirectory(prefix="licketyfit-user-file-test-") as raw:
        tmp = Path(raw)

        global_ids = tmp / "good.npy"
        np.save(global_ids, np.asarray(sorted(expected), dtype=np.int64))
        good, metadata = load_good_wcte_pmts_file(global_ids)
        assert good == expected
        assert metadata["source_resolved"] == "user_file"
        assert metadata["file_layout"] == "global_pmt_id"

        geometry_ids = expected | {300, 301}
        active = authoritative_active_wcte_pmts(expected, geometry_ids)
        assert active == expected
        assert historical_wcsim_inactive_id in active
        assert 300 not in active and 301 not in active
        _expect_error(
            lambda: authoritative_active_wcte_pmts(expected | {9900}, geometry_ids),
            "absent from the loaded detector geometry",
        )
        _expect_error(
            lambda: authoritative_active_wcte_pmts(set(), geometry_ids),
            "is empty",
        )

        pairs = np.asarray(
            [[value // 100, value % 100] for value in sorted(expected)],
            dtype=np.int64,
        )
        pair_file = tmp / "good_pairs.npz"
        np.savez(pair_file, good_pmt_ids=pairs)
        good, metadata = load_good_wcte_pmts_file(pair_file)
        assert good == expected
        assert metadata["file_layout"] == "slot_position"

        text_file = tmp / "good.txt"
        text_file.write_text(
            "# global WCTE PMT IDs\n" + "\n".join(map(str, sorted(expected))) + "\n"
        )
        assert load_good_wcte_pmts_file(text_file)[0] == expected

        csv_file = tmp / "good.csv"
        csv_file.write_text(
            "slot,position\n"
            + "\n".join(f"{value // 100},{value % 100}" for value in sorted(expected))
            + "\n"
        )
        assert load_good_wcte_pmts_file(csv_file)[0] == expected

        json_file = tmp / "good.json"
        json_file.write_text(json.dumps({"good_pmt_ids": sorted(expected)}))
        assert load_good_wcte_pmts_file(json_file)[0] == expected

        bad_mask = tmp / "bad.npy"
        np.save(bad_mask, np.asarray([0.5, 100.0]))
        _expect_error(
            lambda: load_good_wcte_pmts_file(bad_mask), "integer-valued"
        )

        event = np.asarray(
            [[0, 150.0, 2000.0, 7], [100, 160.0, 2001.0, 7]],
            dtype=np.float64,
        )
        event_file = tmp / "events.npy"
        np.save(event_file, event)
        events, event_metadata = load_user_event_file(event_file, strict=True)
        assert len(events) == 1
        assert event_metadata["identity"]["legacy_identity_aliased"] is True

        second_event = event.copy()
        second_event[:, 3] = 8
        object_file = tmp / "events_object.npy"
        object_payload = np.empty(2, dtype=object)
        object_payload[:] = [event, second_event]
        np.save(object_file, object_payload)
        assert len(load_user_event_file(object_file, strict=True)[0]) == 2

        three_dimensional = tmp / "events_3d.npz"
        np.savez(three_dimensional, events=np.stack((event, second_event)))
        assert len(load_user_event_file(three_dimensional, strict=True)[0]) == 2

        pickle_file = tmp / "events.pkl"
        with pickle_file.open("wb") as stream:
            pickle.dump({"events": [event, second_event]}, stream, protocol=4)
        assert len(load_user_event_file(pickle_file, strict=True)[0]) == 2

        five_column = np.column_stack((
            np.vstack((event[:, :3], second_event[:, :3])),
            np.repeat([900, 901], event.shape[0]),
            np.repeat([1900, 1901], event.shape[0]),
        ))
        five_column_file = tmp / "events_5col.npy"
        np.save(five_column_file, five_column)
        five_events, five_metadata = load_user_event_file(
            five_column_file, strict=True
        )
        assert len(five_events) == 2
        assert five_metadata["identity"]["legacy_identity_aliased"] is False

        ambiguous_file = tmp / "ambiguous.npz"
        np.savez(ambiguous_file, first=event, second=second_event)
        _expect_error(
            lambda: load_user_event_file(ambiguous_file, strict=True),
            "Set USER_EVENT_KEY",
        )
        assert len(load_user_event_file(
            ambiguous_file, user_event_key="first", strict=True
        )[0]) == 1

        invalid_cases = {
            "PMT-ID": np.asarray([[0.5, 150.0, 2000.0, 7]], dtype=np.float64),
            "negative charges": np.asarray([[0, -1.0, 2000.0, 7]], dtype=np.float64),
            "constant within one event": np.asarray(
                [[0, 150.0, 2000.0, 7], [100, 160.0, 2001.0, 8]],
                dtype=np.float64,
            ),
        }
        for expected_text, payload in invalid_cases.items():
            path = tmp / (expected_text.replace(" ", "_") + ".npy")
            # A one-dimensional object array preserves the two-row payload as one
            # event so mixed-identity validation is exercised rather than flat-table
            # grouping semantics.
            wrapper = np.empty(1, dtype=object)
            wrapper[0] = payload
            np.save(path, wrapper)
            _expect_error(
                lambda path=path: load_user_event_file(path, strict=True),
                expected_text,
            )

        run = 9999
        dq_root = (
            tmp / "eos" / "dq_flags" / str(run)
            / f"WCTE_dq_flags_R{run}.root"
        )
        _write_fake_dq_root(dq_root, sorted(expected))
        project_root = Path(__file__).resolve().parent.parent
        good, metadata = resolve_good_wcte_pmts(
            source="run",
            run=run,
            root_search_bases=(tmp / "eos",),
            project_root=project_root,
        )
        assert good == expected
        assert metadata["source_resolved"] == "run_root"
        assert metadata["root_file"] == str(dq_root)
        assert metadata["root_loader"] == "direct_uproot_configuration"

        _expect_error(
            lambda: resolve_good_wcte_pmts(
                source="run",
                run=run,
                root_search_bases=(tmp / "eos",),
                analysis_tools_path=tmp / "external_analysis_tools",
                project_root=project_root,
            ),
            "External analysis_tools paths are no longer supported",
        )

        good, metadata = resolve_good_wcte_pmts(
            source="auto",
            run=run,
            good_pmt_file=global_ids,
            good_pmt_root_file=tmp / "does_not_exist.root",
        )
        assert good == expected
        assert metadata["source_resolved"] == "user_file"

        _expect_error(
            lambda: resolve_good_wcte_pmts(source="geometry", run=run),
            "authoritative user or run/DQ",
        )

    if supplied_event_file is not None:
        events, metadata = load_user_event_file(
            supplied_event_file, strict=True
        )
        if not events:
            raise AssertionError("The supplied event file contains no events")
        print(
            "Supplied file:", len(events), "events,",
            sum(event.shape[0] for event in events), "hit rows,",
            metadata["identity"]["schema"],
        )
    print(
        "PASS: WCTE user-event and authoritative good-PMT contracts "
        "(including historical WCSim-slot independence)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "event_file",
        nargs="?",
        type=Path,
        help="optional real user-event NPY/NPZ/PKL file to validate",
    )
    args = parser.parse_args()
    run_contract_tests(args.event_file)


if __name__ == "__main__":
    main()
