# LicketyFit: analytic water-Cherenkov particle-track fitter

LicketyFit is an analytic likelihood fitter for reconstructing charged-particle tracks in water-Cherenkov detectors.  The current codebase supports both WCTE/real-data-style event inputs and WCSim simulation inputs, and can fit tracks using charge-only, timing-only, or joint charge+timing likelihoods.

The fitter is built around an `Emitter` model that predicts the expected photoelectrons and mean hit times at each PMT for a hypothesized particle track.  Batch drivers then use `iminuit.Minuit` to minimize the negative log likelihood over the track parameters.

## Main capabilities

- Fits single primary-particle track hypotheses for `muon`, `pion`, `kaon`, and `proton`-style particles, depending on the support provided by `particle_cherenkov_model.py` and the available range tables.
- Supports two track-end parameterizations:
  - `full_length`: the original 7-parameter fit, where the fitted length is the full Cherenkov-visible range to threshold.
  - `absorption`: an 8-parameter fit, where the visible track length can end abruptly before the full range implied by the initial kinetic energy.
- Supports three likelihood modes:
  - `charge_time`: use both charge and timing information.
  - `charge_only`: use only PMT charge/PE information.
  - `timing_only`: use only timing information.
- Supports fixed-parameter fits through the driver configuration, including fixing `ke0_mev` in absorption mode.
- Uses seed-grid scanning and retry logic for robust batch fitting.
- Can run either on WCTE production ROOT selections or on user-provided already-selected event arrays.
- Can run on WCSim-derived files through `read_sim_data(INPUT_FILE)`.

## Repository layout

A typical layout for the files in this snapshot is:

```text
repo/
├── LicketyFit/
│   ├── Emitter.py
│   ├── Event.py
│   └── PMT.py
├── scripts/
│   ├── batch_fit_driver_wcte.py
│   ├── batch_fit_driver_wcsim.py
│   ├── event_loader.py
│   ├── get_mu_events.py
│   ├── generate_particle_range_tables.py
│   └── particle_range_lookup.py
├── tables/
│   ├── E_vs_dist_cm_muon.npy
│   ├── overall_distances_cm_muon.npy
│   ├── E_vs_dist_cm_pion.npy
│   ├── overall_distances_cm_pion.npy
│   ├── E_vs_dist_cm_kaon.npy
│   ├── overall_distances_cm_kaon.npy
│   ├── E_vs_dist_cm_proton.npy
│   ├── overall_distances_cm_proton.npy
│   ├── other_mpmt_info_v2.dict
│   ├── rel_mpmt_eff.dict
│   ├── delta_e_angular_pdf_table.npz          # optional
│   └── wcsim_wcte_mapping.txt                # optional, for WCSim PMT-ID mapping
└── outputs/
```

The batch drivers infer paths relative to their own location.  In the driver scripts, `SCRIPT_DIR = Path(__file__).resolve().parent`, `PROJECT_ROOT = SCRIPT_DIR.parent`, `LICKETYFIT_DIR = PROJECT_ROOT / "LicketyFit"`, and the default table directory is `PROJECT_ROOT / "tables"`.

## Python dependencies

The uploaded code imports the following Python packages:

```bash
python -m pip install numpy pandas uproot awkward iminuit numba matplotlib
```

You also need the local/external modules used by the fitter:

- `Geometry.Device`, usually from the WCTE geometry package.
- `particle_cherenkov_model.py`, which supplies particle aliases, Cherenkov-angle tables, energy-distance tables, relative mPMT efficiency tables, and active-particle selection.
- `read_sim_data.py`, required by `batch_fit_driver_wcsim.py`.

The geometry file defaults to:

```text
/eos/user/j/jrimmer/Geometry/examples/wcte_bldg157.geo
```

Override this with `GEOMETRY_PATH` or `WCTE_GEOMETRY_FILE`.

## Lookup/range tables

The fitter expects particle range tables in the common format:

```text
E_vs_dist_cm_<particle>.npy
overall_distances_cm_<particle>.npy
```

where `<particle>` is one of:

```text
muon, pion, kaon, proton
```

`generate_particle_range_tables.py` can generate Bethe-Bloch/CSDA-style water range tables:

```bash
python scripts/generate_particle_range_tables.py \
  --particles muon pion kaon proton \
  --output-dir tables
```

The generated files are:

- `E_vs_dist_cm_<particle>.npy`: object array whose rows contain two-column arrays `[distance_cm, kinetic_energy_MeV]`.
- `overall_distances_cm_<particle>.npy`: total above-threshold range for each initial kinetic energy, in cm.

`particle_range_lookup.py` loads these tables and returns distances in mm.  It searches, in order, directories from `LF_TABLE_DIR`, `LF_MULTIPARTICLES_TABLE_DIR`, local `tables/` folders, and a few historical CERN fallback paths.

## Running the WCTE/real-data-style batch driver

Use:

```bash
python3 scripts/batch_fit_driver_wcte.py
```

or override settings through environment variables:

```bash
RUN=2079 \
BEAM_P=1500 \
N_EVENTS=1000 \
FIT_PARTICLE=muon \
FIT_MODE=full_length \
LIKELIHOOD_MODE=charge_time \
python scripts/batch_fit_driver_wcte.py
```

Nearly all user-facing settings are collected at the top of `batch_fit_driver_wcte.py` in the `USER CONFIGURATION` block.  You can either edit values directly in that block or set the corresponding environment variables.

### WCTE driver: run, input, and output settings

| Setting | Default | Meaning |
|---|---:|---|
| `RUN` | `1589` | WCTE run number. Used to build the default production ROOT path. |
| `BEAM_P` | `1500` | Beam momentum label used in metadata/output naming. |
| `N_EVENTS` | `10000` | Number of ROOT entries to inspect, or maximum events to load from a custom file. For `EVENT_SOURCE="selection"`, this is not the same thing as the number of selected events returned. |
| `N_EVENTS_PER_BATCH` | `100` | Number of events sent to each batch loop before writing results. |
| `EVENT_SOURCE` | `selection` | `selection` uses `event_loader.get_selected_events`; `file` loads an already-selected user event file. Aliases such as `user`, `custom`, and `provided` are normalized to `file`. |
| `USER_EVENT_FILE` | empty | Required when `EVENT_SOURCE=file`. Path to `.npy`, `.npz`, `.pkl`, or `.pickle` file. |
| `USER_EVENT_KEY` | `None` | Optional key to select from a `.npz` or dictionary payload. |
| `USER_EVENT_APPLY_PEAK_WINDOW` | `True` | If `True`, custom events are still trimmed to the local timing peak before fitting. |
| `LF_OUTPUT_FILE` | empty | Override output file path. If empty, the driver writes under `outputs/`. |

### WCTE driver: internal event selection mode

With `EVENT_SOURCE=selection`, the driver calls:

```python
get_selected_events(
    RUN,
    N_EVENTS,
    particle=PARTICLE_SELECTION_LABEL,
    root_file=CONFIG_ROOT_FILE,
    use_peak_time_cut=USE_PEAK_TIME_CUT,
    peak_window=PEAK_WINDOW_NS,
    peak_bin_width=PEAK_BIN_WIDTH_NS,
    tof_primary=SELECTION_TOF_NS,
    tof_window=SELECTION_TOF_WINDOW_NS,
    tof_scalar_field=SELECTION_TOF_FIELD,
    momentum_scalar_field=SELECTION_MOMENTUM_FIELD,
    t5_particle_nr=SELECTION_T5_PARTICLE_NR,
)
```

The default ROOT file is:

```text
/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/<RUN>/WCTE_merged_production_R<RUN>.root
```

Override it with `CONFIG_ROOT_FILE`.

The loader expects a `WCTEReadoutWindows` tree with branches including:

```text
hit_mpmt_slot_ids
hit_pmt_position_ids
hit_pmt_charges
hit_pmt_calibrated_times
vme_act_eveto
vme_act_tagger
vme_tof_corr
T5_HasMultipleScintillatorsHit
T5_HasOutOfTimeWindow
T5_HasValidHit
T5_particle_nr
T5_HasInTimeWindow
T5_hit_time                  # only required when use_t5_hit_time=True
window_data_quality_mask
vme_digi_issues_bitmask
vme_evt_quality_bitmask
```

It also reads scalar values from the `vme_analysis_scalar_results` tree.  By default, the scalar fields are selected from the particle label:

```text
act_eveto_cut
act_tagger_cut
tof_mean_<particle>
momentum_after_beam_window_mean_<particle>
```

For muons, the default field names are `tof_mean_muon` and `momentum_after_beam_window_mean_muon`.  Override these with `SELECTION_TOF_FIELD`, `SELECTION_MOMENTUM_FIELD`, or provide the value directly using `SELECTION_TOF_NS`.

The event loader returns a list of arrays.  Each event array has columns:

```text
column 0: WCTE PMT ID = mPMT slot * 100 + PMT position
column 1: charge
column 2: calibrated hit time [ns]
column 3: ROOT entry index
```

In the WCTE driver, charges are converted to PE by dividing by `pe_scale=143`.

### WCTE driver: passing your own data

Set:

```bash
EVENT_SOURCE=file USER_EVENT_FILE=/path/to/events.npz python scripts/batch_fit_driver_wcte.py
```

Supported file extensions are:

```text
.npy, .npz, .pkl, .pickle
```

The payload must represent already-selected events.  The accepted forms are:

1. A Python `list` or `tuple` of event arrays.
2. A one-dimensional object array where each element is an event array.
3. A three-dimensional numeric array with shape:

   ```text
   (n_events, n_hits, n_columns)
   ```

4. A two-dimensional numeric array with columns:

   ```text
   [pmt_id, charge, time]
   ```

   This is interpreted as a single event.

5. A two-dimensional numeric array with at least four columns:

   ```text
   [pmt_id, charge, time, event_number]
   ```

   This is interpreted as a concatenated event table and grouped by `event_number`.

Each per-event array must have at least three columns:

```text
column 0: WCTE PMT ID = mPMT slot * 100 + PMT position
column 1: charge, in the same raw charge scale expected by the WCTE driver
column 2: hit time [ns]
column 3: optional event number or bookkeeping column
```

The fitter only consumes columns `0:3`.  If a per-event array has a fourth column, it is kept during loading but is not used by the likelihood.

For `.npz` or dictionary payloads, the driver chooses the data in this order:

1. `USER_EVENT_KEY`, if supplied.
2. Key `events`, if present.
3. Key `data`, if present.
4. Key `arr_0`, if present.
5. For `.npz` only, if there is exactly one array in the file, that array is used.

Example: save variable-length selected events as an object-array `.npz`:

```python
import numpy as np

# events[i] is an array with columns [pmt_id, charge, time]
events = [
    np.column_stack([pmt_ids_0, charges_0, times_0]),
    np.column_stack([pmt_ids_1, charges_1, times_1]),
]

np.savez("my_selected_events.npz", events=np.asarray(events, dtype=object))
```

Example: save a concatenated table:

```python
import numpy as np

# rows are [pmt_id, charge, time, event_number]
all_hits = np.asarray([
    [3700, 143.0, 1023.5, 0],
    [3701,  96.0, 1024.0, 0],
    [3800, 120.0,  998.2, 1],
])

np.save("my_selected_events.npy", all_hits)
```

Run it with:

```bash
EVENT_SOURCE=file \
USER_EVENT_FILE=my_selected_events.npz \
USER_EVENT_KEY=events \
FIT_PARTICLE=proton \
FIT_MODE=absorption \
LIKELIHOOD_MODE=charge_time \
python scripts/batch_fit_driver_wcte.py
```

### WCTE driver: physics and fit settings

| Setting | Values | Meaning |
|---|---|---|
| `FIT_PARTICLE` | `muon`, `pion`, `kaon`, `proton`, aliases supported by `particle_cherenkov_model.py` | Particle hypothesis used by the likelihood and range table. |
| `FIT_MODE` / `TRACK_END_MODE` | `full_length`, `absorption` | Selects 7- or 8-parameter track model. |
| `LIKELIHOOD_MODE` / `FIT_TYPE` | `charge_time`, `charge_only`, `timing_only` | Selects likelihood terms. Legacy `USE_CHARGE_LIKELIHOOD` and `USE_TIMING_LIKELIHOOD` still work. |
| `USE_T0_PRIOR` | boolean | Adds a configurable prior on `t0`. |

In `full_length` mode the fit parameters are:

```text
x0, y0, z0, cx, cy, length, t0
```

In `absorption` mode the fit parameters are:

```text
x0, y0, z0, cx, cy, visible_length, full_range, t0
```

`visible_length` is the actually visible primary-Cherenkov track length.  `full_range` is the dE/dx range to Cherenkov threshold that corresponds to the initial kinetic energy.  The driver converts `full_range` to `ke0` through `ParticleRangeLookup`.

### WCTE driver: fixed parameters

Set any parameter to a number to hold it fixed.  Leave it as `None`, blank, `free`, `false`, `none`, or `null` to let it float.

Environment variables:

```text
FIX_X0
FIX_Y0
FIX_Z0
FIX_CX
FIX_CY
FIX_LENGTH              # full_length mode only
FIX_VISIBLE_LENGTH      # absorption mode only
FIX_FULL_RANGE          # absorption mode only
FIX_KE0_MEV or FIXED_KE0_MEV
FIX_T0
```

In absorption mode, `FIX_KE0_MEV`/`FIXED_KE0_MEV` is converted internally to `full_range` using the same particle range table used by the fitter.  Do not fix both `full_range` and `ke0_mev` unless they intentionally describe the same range.

Example: fit a proton sample with fixed initial kinetic energy and fixed beam start:

```bash
EVENT_SOURCE=file \
USER_EVENT_FILE=my_proton_events.npz \
FIT_PARTICLE=proton \
FIT_MODE=absorption \
FIX_Z0=-1350 \
FIXED_KE0_MEV=1000 \
python scripts/batch_fit_driver_wcte.py
```

### WCTE driver: retry and seed-grid settings

| Setting | Default | Meaning |
|---|---:|---|
| `NPROC` | `16` | Multiprocessing worker count. |
| `M_STRAT` | `1` | Minuit strategy. |
| `FCN_RETRY_THRESHOLD` | `1100.0` | Retry if the current best FCN is non-finite or above this threshold. This logic does not use `m.valid` as a failure criterion. |
| `MAX_FIT_ATTEMPTS` | `4` | Maximum retry attempts. |
| `NCALL_MIGRAD` | `70000` | Main Minuit call limit. |
| `NCALL_SIMPLEX` | `NCALL_MIGRAD` | Simplex call limit, when used. |
| `ENABLE_STAGE1_SEED_GRID` | `True` | Use the seed grid to choose the starting point. |
| `ENABLE_STAGE2_MIGRAD_FIRST` | `False` | Optional alternate first-stage minimization. |
| `ENABLE_STAGE3_ADAPTIVE_RESCUE` | `False` | Optional rescue stage. |
| `ENABLE_STAGE4_LENGTH_PROFILE` | `False` | Optional length-profile rescue. |

Seed-grid environment variables:

```text
FAST_SEED_X0
FAST_SEED_Y0
FAST_SEED_Z0
FAST_SEED_VISIBLE_LENGTHS
FAST_SEED_KE0_MEV
FAST_SEED_FULL_RANGES_MM
FAST_SEED_FULL_CARTESIAN
```

List-valued environment variables can be comma- or semicolon-separated:

```bash
FAST_SEED_Z0="-1500,-1400,-1350,-1300" \
FAST_SEED_VISIBLE_LENGTHS="300,500,700,900,1100" \
python scripts/batch_fit_driver_wcte.py
```

### WCTE driver: masks, calibration, and paths

| Setting | Default | Meaning |
|---|---|---|
| `RING_MASK_MODE` | `both` | `none`, `pes`, `ts`, or `both`. Controls whether PMTs outside the kept ring are masked in charge, timing, or both. |
| `PARTICLE_SELECTION_LABEL` | `FIT_PARTICLE` | Particle label passed to `event_loader`. |
| `SELECTION_TOF_NS` | `None` | Explicit TOF mean. If `None`, read from scalar tree. |
| `SELECTION_TOF_WINDOW_NS` | `0.2` | TOF selection half-window. |
| `SELECTION_TOF_FIELD` | particle-dependent | Override scalar field used for TOF. |
| `SELECTION_MOMENTUM_FIELD` | particle-dependent | Override scalar field used for beam momentum diagnostic. |
| `SELECTION_T5_PARTICLE_NR` | `1` | T5 particle-number selection. |
| `USE_PEAK_TIME_CUT` | `False` | Apply original calibrated-hit-time peak cut. |
| `PEAK_WINDOW_NS` | `100.0` | Peak-time window. |
| `PEAK_BIN_WIDTH_NS` | `50.0` | Peak-time histogram bin width. |
| `INACTIVE_SLOTS` | driver default list | mPMT slots excluded from event construction. |
| `GEOMETRY_PATH` | `/eos/user/j/jrimmer/Geometry` | Geometry package path added to `sys.path`. |
| `WCTE_GEOMETRY_FILE` | `<GEOMETRY_PATH>/examples/wcte_bldg157.geo` | Geometry file loaded by `Device.open_file`. |
| `TABLE_DIR` / `LF_TABLE_DIR` | `<project root>/tables` | Table directory. |
| `OTHER_MPMT_INFO_PATH` | `<table dir>/other_mpmt_info_v2.dict` | mPMT type/calibration dictionary. |
| `REL_MPMT_EFF_PATH` | `<table dir>/rel_mpmt_eff.dict` | Relative mPMT efficiency dictionary. |

### WCTE driver: debug output and truth diagnostics

Use these sparingly because some can make output files very large:

```text
SAVE_ATTEMPT_RESULTS
SAVE_SEED_SCAN
SAVE_TOP_N_SEEDS
```

Truth diagnostic variables:

```text
TRUE_X0, TRUE_Y0, TRUE_Z0
TRUE_CX, TRUE_CY
TRUE_T0
TRUE_LENGTH_MM
TRUE_VISIBLE_LENGTH_MM
TRUE_FULL_RANGE_MM
TRUE_INITIAL_KE_MEV
```

If truth values are not supplied, `true_fcn` is stored as `NaN`.

## Running the WCSim batch driver

Use:

```bash
python scripts/batch_fit_driver_wcsim.py
```

or:

```bash
WCSIM_INPUT_FILE=/path/to/events.npz \
ENERGY_TRUE=1000 \
FIT_PARTICLE=proton \
FIT_MODE=absorption \
LIKELIHOOD_MODE=charge_time \
TOT_EVENTS=200 \
python scripts/batch_fit_driver_wcsim.py
```

`batch_fit_driver_wcsim.py` is parallel in structure to the WCTE driver, but with WCSim-specific input handling:

- The input is read with `read_sim_data(INPUT_FILE)`.
- WCSim PMT IDs can be mapped to WCTE PMT IDs using `tables/wcsim_wcte_mapping.txt`.
- WCSim charges use `pe_scale=1.0`.
- PMT placements use the `design` geometry.
- Real-data mPMT efficiency correction is not applied; `mpmt_types` remains `None`.

### WCSim driver: input, output, and sample labels

| Setting | Default | Meaning |
|---|---:|---|
| `ENERGY_TRUE` | `1000.0` | Nominal kinetic energy of the simulated sample. Used for filenames, truth diagnostics, and default fixed `ke0_mev` in absorption mode. |
| `TOT_EVENTS` | `1000` | Maximum number of WCSim events to fit from the input file. |
| `N_EVENTS_PER_BATCH` | `100` | Events per batch. |
| `WCSIM_PARTICLE_LABEL` | `FIT_PARTICLE_STR` or `p+` | File/directory label for the WCSim sample. This is separate from the fit hypothesis. |
| `WCSIM_PARTICLE_DIR` | `WCSIM_PARTICLE_LABEL` | Directory label used if constructing a default path. |
| `WCSIM_DATA_PATH` | CERN EOS default | Base data area used only if constructing your own default path. |
| `DEFAULT_WCSIM_INPUT_FILE` | sample `.npz` path | Default input file when `WCSIM_INPUT_FILE` is not set. |
| `WCSIM_INPUT_FILE` | empty | Explicit input file override. |
| `LF_OUTPUT_FILE` | empty | Explicit output file override. |

The driver sets:

```python
INPUT_FILE = WCSIM_INPUT_FILE or DEFAULT_WCSIM_INPUT_FILE
```

and then calls:

```python
data_raw = read_sim_data(INPUT_FILE)
```

### WCSim input file format

The exact on-disk format is controlled by `read_sim_data.py`, which is not included in this uploaded snapshot.  From the driver side, `read_sim_data(INPUT_FILE)` must return an object/dictionary with at least these keys:

```text
data_raw["digi_hit_time"]
data_raw["digi_hit_pmt"]
data_raw["digi_hit_charge"]
```

Each key must index by event number.  For event `i`:

```python
hit_times   = np.asarray(data_raw["digi_hit_time"][i], dtype=np.float64)
hit_pmts    = np.asarray(data_raw["digi_hit_pmt"][i], dtype=int)
hit_charges = np.asarray(data_raw["digi_hit_charge"][i], dtype=np.float64)
```

These arrays must have the same length for each event.  The driver applies a simple event-level timing window before constructing the `Event` object.

The PMT ID interpretation is controlled by `WCSIM_PMT_ID_MODE`:

```text
mapping: require a mapping text file
wcte:    assume digi_hit_pmt already stores WCTE IDs of the form slot*100+pmt_position
auto:    use the mapping file if present; otherwise fall back to WCTE-ID mode
```

When using a mapping file, the default path is:

```text
<table dir>/wcsim_wcte_mapping.txt
```

Override it with `WCSIM_WCTE_MAPPING_PATH`.  The current mapping loader expects rows where:

```text
column 0: raw WCSim PMT ID, after applying WCSIM_PMT_ID_OFFSET
column 1: WCTE mPMT slot
column 2: WCTE PMT position, one-based
```

The code converts each mapping row into:

```python
wcte_pmt_id = slot * 100 + (pmt_position_one_based - 1)
```

If `WCSIM_PMT_ID_MODE=wcte`, then `digi_hit_pmt` should already use:

```text
WCTE PMT ID = mPMT slot * 100 + PMT position
```

WCSim charges are treated as PE-like values because the WCSim driver uses `pe_scale=1.0`.

### WCSim driver: physics, fixed parameters, and fit controls

The WCSim driver uses the same core settings as the WCTE driver:

```text
FIT_PARTICLE
FIT_MODE / TRACK_END_MODE
LIKELIHOOD_MODE / FIT_TYPE
USE_T0_PRIOR
NPROC
M_STRAT
FCN_RETRY_THRESHOLD
MAX_FIT_ATTEMPTS
NCALL_MIGRAD
NCALL_SIMPLEX
ENABLE_STAGE1_SEED_GRID
ENABLE_STAGE2_MIGRAD_FIRST
ENABLE_STAGE3_ADAPTIVE_RESCUE
ENABLE_STAGE4_LENGTH_PROFILE
FAST_SEED_X0
FAST_SEED_Y0
FAST_SEED_Z0
FAST_SEED_VISIBLE_LENGTHS
FAST_SEED_KE0_MEV
FAST_SEED_FULL_RANGES_MM
FAST_SEED_FULL_CARTESIAN
RING_MASK_MODE
INACTIVE_SLOTS
GEOMETRY_PATH
WCTE_GEOMETRY_FILE
TABLE_DIR / LF_TABLE_DIR
SAVE_ATTEMPT_RESULTS
SAVE_SEED_SCAN
SAVE_TOP_N_SEEDS
```

The main fixed-parameter difference is the default behavior.  In `batch_fit_driver_wcsim.py`, the default is:

```text
FIX_Z0 = -1350.0
FIXED_KE0_MEV = ENERGY_TRUE       # absorption mode only
```

In `full_length` mode, `ke0_mev` defaults to `None` because the fitted length determines the initial kinetic energy.

You can override all fixed parameters with:

```text
FIX_X0
FIX_Y0
FIX_Z0
FIX_CX
FIX_CY
FIX_LENGTH
FIX_VISIBLE_LENGTH
FIX_FULL_RANGE
FIX_KE0_MEV or FIXED_KE0_MEV
FIX_T0
```

### WCSim driver: optional truth diagnostics

By default, the WCSim driver uses `ENERGY_TRUE` for `TRUE_INITIAL_KE_MEV`.  You can override truth diagnostics with:

```text
TRUE_X0, TRUE_Y0, TRUE_Z0
TRUE_CX, TRUE_CY
TRUE_T0
TRUE_LENGTH_MM
TRUE_VISIBLE_LENGTH_MM
TRUE_FULL_RANGE_MM
TRUE_INITIAL_KE_MEV
```

For absorption-mode truth, provide both a visible length and either `TRUE_FULL_RANGE_MM` or `TRUE_INITIAL_KE_MEV`.  For full-length-mode truth, provide `TRUE_LENGTH_MM` or `TRUE_INITIAL_KE_MEV`.

## Output file format

Both batch drivers write a pickle file containing a Python dictionary.  The default extension is `.dict`, but the payload is loaded with `pickle`:

```python
import pickle

with open("outputs/estimates_...dict", "rb") as f:
    est = pickle.load(f)
```

The output dictionary contains metadata plus one entry per fitted event.  Common keys include:

```text
metadata
minimum_found
x, y, z
cx, cy
t
visible_length
full_range
ke0
length                 # legacy alias for visible_length
est_fcn
true_fcn
n_attempts
chosen_seed_idx
chosen_seed_fcn
chosen_seed_params
adaptive_rescue_used
fcn_retry_used
length_profile_rescue_considered
length_profile_rescue_used
edm
```

If enabled, it can also contain:

```text
seed_scan
attempt_results
```

## Notes on the internal model

### `Emitter.py`

`Emitter` is the core Cherenkov light model.  It stores the particle species, track start, direction, length, time, and intensity, and predicts expected PE and timing at each PMT.  Important model features in this snapshot include:

- Particle-dependent range/energy lookup.
- `threshold`/`full_length` endpoint mode.
- `abrupt`/`absorption` endpoint mode for tracks whose visible light ends before the full dE/dx range.
- Analytic primary geometric falloff `N_geo`.
- Refined analytic secondary-electron light model based on a table interpreted as `dS_delta/du(K_mu, u)`.
- Optional secondary-electron timing contribution.
- Caches for range tables, relative mPMT efficiency curves, PMT-radius corrections, and hot-loop quantities.

### `PMT.py`

`PMT` implements the PMT likelihood response.  It validates PMT model parameters, precalculates small-PE charge response tables, and uses numba-compiled likelihood kernels for charge-only and charge+time fits.

### `Event.py`

`Event` stores detector hit information in nested mPMT/PMT lists:

```text
hit_times[mpmt][pmt]
hit_charges[mpmt][pmt]
mpmt_status[mpmt]
pmt_status[mpmt][pmt]
```

It also supports pickle and JSON save/load helpers.  `SimulatedEvent` extends `Event` for storing expected/true PE information.

### `event_loader.py`

`event_loader.py` reads WCTE production ROOT files, applies beam/quality/TOF/T5 timing selections, and returns already-selected event arrays with columns:

```text
[pmt_id, charge, calibrated_time_ns, root_entry_index]
```

The historical `get_mu_events` name remains as a compatibility alias, but new code should call `get_selected_events` or `get_particle_events`.

### `n_model_wrapper.py`

`n_model_wrapper.py` contains the older/vectorized phenomenological `n_from_E_r(E, r)` model.  The current `Emitter` can use the analytic primary `N_geo` replacement, but this wrapper is still available for compatibility and comparison.

## Common examples

### Fit WCTE selected events from a production run

```bash
RUN=1589 \
BEAM_P=1500 \
N_EVENTS=10000 \
FIT_PARTICLE=proton \
FIT_MODE=absorption \
LIKELIHOOD_MODE=charge_time \
python scripts/batch_fit_driver_wcte.py
```

### Fit a custom WCTE-like event file

```bash
EVENT_SOURCE=file \
USER_EVENT_FILE=/path/to/selected_events.npz \
USER_EVENT_KEY=events \
USER_EVENT_APPLY_PEAK_WINDOW=true \
FIT_PARTICLE=proton \
FIT_MODE=absorption \
LIKELIHOOD_MODE=charge_time \
python scripts/batch_fit_driver_wcte.py
```

### Fit WCSim events

```bash
WCSIM_INPUT_FILE=/path/to/wcsim_events.npz \
ENERGY_TRUE=1000 \
TOT_EVENTS=218 \
FIT_PARTICLE=proton \
WCSIM_PARTICLE_LABEL=p+ \
FIT_MODE=absorption \
LIKELIHOOD_MODE=charge_time \
python scripts/batch_fit_driver_wcsim.py
```

### Charge-only full-length muon fit

```bash
FIT_PARTICLE=muon \
FIT_MODE=full_length \
LIKELIHOOD_MODE=charge_only \
python scripts/batch_fit_driver_wcte.py
```

### Save more diagnostics

```bash
SAVE_TOP_N_SEEDS=10 \
SAVE_ATTEMPT_RESULTS=true \
python scripts/batch_fit_driver_wcte.py
```

Be careful with `SAVE_SEED_SCAN=true`; full seed scans can become very large.

## Troubleshooting

### `event_loader.py was not found`

Make sure `event_loader.py` is in the same `scripts/` directory as `batch_fit_driver_wcte.py`, or add its directory to `PYTHONPATH`.  Alternatively, bypass the ROOT event loader with:

```bash
EVENT_SOURCE=file USER_EVENT_FILE=/path/to/events.npz python scripts/batch_fit_driver_wcte.py
```

### `USER_EVENT_FILE` errors

Check that:

- The suffix is one of `.npy`, `.npz`, `.pkl`, `.pickle`.
- Each event array is two-dimensional.
- Each event array has at least three columns: `[pmt_id, charge, time]`.
- If using `.npz`, either use key `events`, key `data`, key `arr_0`, a single-array file, or set `USER_EVENT_KEY`.

### WCSim PMT IDs look wrong

Check `WCSIM_PMT_ID_MODE`:

- Use `wcte` if `digi_hit_pmt` already stores `slot*100 + pmt_position`.
- Use `mapping` if you need `tables/wcsim_wcte_mapping.txt`.
- Use `auto` to try the mapping file first and fall back to WCTE-ID mode.

Also check `WCSIM_PMT_ID_OFFSET`.  The driver subtracts this offset before looking up WCSim IDs in the mapping dictionary.

### Fixed `ke0_mev` is rejected

In absorption mode, fixed `ke0_mev` must be above the Cherenkov threshold for the selected particle and must be convertible to a valid `full_range` by the particle range table.

### Too much output data

Leave these off for production runs unless you are debugging fits:

```text
SAVE_ATTEMPT_RESULTS=false
SAVE_SEED_SCAN=false
SAVE_TOP_N_SEEDS=0
```

## Development notes

- The batch drivers intentionally do not treat `m.valid == False` by itself as a failed fit.  Retry decisions are based on non-finite FCN, FCN threshold, parameter-bound/stuck diagnostics, and optional rescue stages.
- `N_EVENTS` in the WCTE selection path is a ROOT-entry inspection limit, not a guaranteed number of selected events.
- WCTE real-data-style charges are converted to PE with `pe_scale=143`; WCSim charges are used with `pe_scale=1.0`.
- The WCTE driver uses estimated PMT placements (`place_info="est"`), while the WCSim driver uses design placements (`place_info="design"`).
- The `outputs/` directory is created automatically by the drivers.
