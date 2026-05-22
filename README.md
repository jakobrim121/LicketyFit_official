# LF_multiParticles

Self-contained particle-aware LicketyFit checkout for WCTE/WCSim fits.

## Main driver files

- `scripts/batch_fit_driver.py` — WCTE/real-data-style selected-event driver.
- `scripts/batch_fit_driver_wcsim_multiparticle.py` — WCSim `.npz` driver.
- `scripts/check_setup.py` — verifies imports and required local tables.

Both batch drivers support two fit modes via `FIT_MODE` / `TRACK_END_MODE`:

```text
FIT_MODE=full_length
    7-parameter original mode:
    x0, y0, z0, cx, cy, length, t0

    length is the full visible track/range to Cherenkov threshold.
    The initial KE is inferred from length using the particle range table.

FIT_MODE=absorption
    8-parameter abrupt-endpoint mode:
    x0, y0, z0, cx, cy, visible_length, full_range, t0

    visible_length is the observed primary-Cherenkov segment before an abrupt cutoff.
    full_range is the dE/dx-only range to Cherenkov threshold that maps to the inferred initial KE.
```

`absorption` is the default. Use `full_length` for the original muon-style behavior.

## Important user-facing configuration

Edit the top of the relevant driver or set environment variables:

```bash
export FIT_PARTICLE=proton          # muon, pion, kaon, proton and common aliases
export FIT_MODE=absorption           # absorption or full_length
export N_EVENTS=10000               # real-data driver
export TOT_EVENTS=1000              # WCSim driver
export NPROC=16
export USE_CHARGE_LIKELIHOOD=1
export USE_TIMING_LIKELIHOOD=0
export SAVE_SEED_SCAN=0             # keep off for production
export SAVE_TOP_N_SEEDS=0           # set e.g. 5 only for debugging
```

For the real-data driver, `event_loader.py` replaces the old `get_mu_events.py`.  The compatibility alias `get_mu_events()` still exists, but new code should use `get_selected_events()`.

## Renamed particle-generic files

- `model_muon_cherenkov_collapse.py` -> `LicketyFit/particle_cherenkov_model.py`
- `muon_range_lookup.py` -> `scripts/particle_range_lookup.py`
- `get_mu_events.py` -> `scripts/event_loader.py`

Imports in the drivers and emitter have been updated accordingly.

## Real-data driver: using your own selected events

The real-data driver can either run its internal ROOT/event-loader selection or fit events you already selected yourself.

Internal selection (default):

```bash
export EVENT_SOURCE=selection
export FIT_PARTICLE=proton
python scripts/batch_fit_driver.py
```

The internal selection reads the particle-specific scalar TOF mean, for example `tof_mean_proton`, `tof_mean_pion`, or `tof_mean_muon`, and applies the default `±0.2 ns` TOF window unless `SELECTION_TOF_WINDOW_NS` is overridden.

User-provided events:

```bash
export EVENT_SOURCE=file
export USER_EVENT_FILE=/path/to/events.npy   # also supports .npz, .pkl, .pickle
export USER_EVENT_KEY=events                 # optional for npz/dict payloads
export USER_EVENT_APPLY_PEAK_WINDOW=1         # default; set 0 if already time-windowed
python scripts/batch_fit_driver.py
```

Each event should be an array with columns `[pmt_id, charge, time]`. A single concatenated 2D array with `[pmt_id, charge, time, event_number]` is also accepted and will be grouped by `event_number`.
