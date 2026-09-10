# LicketyFit

LicketyFit reconstructs a **single charged-particle track** from PMT charge and
hit-time data in a water Cherenkov detector. It accepts digitized WCSim NPZ files
and real WCTE data, with `muon`, `pion`, `kaon`, and `proton` fit hypotheses.
It provides batch launchers and an interactive interface that uses the same
production fitter for one event at a time.

The current launcher release identifier is
`2026-09-04-v1.45.2-absorption-endpoint-fix`. This README describes the current
source, including the updated notebook interface. For an exact source revision,
record `git rev-parse HEAD` and `git submodule status` with your analysis.

## Installation

Clone the repository and initialize its two direct runtime submodules:

```bash
git clone https://github.com/jakobrim121/LicketyFit_official.git
cd LicketyFit_official
git submodule update --init analysis_tools Geometry

python3 -m pip install -r requirements.txt
```

For an existing checkout, the `git submodule update` command above fills empty
submodule directories and checks out the revisions recorded by the parent
repository. Run it again after updating the parent checkout. Recursive
initialization is unnecessary for LicketyFit: the adapter loads the top-level
`analysis_tools` and `Geometry` sources directly, without their nested
submodules.

| Component | Location | Purpose |
|---|---|---|
| Detector geometry | `Geometry/` | Geometry classes and the default `Geometry/examples/wcte_bldg157.geo` |
| Collaboration input | `analysis_tools/` | WCTE `DataLoader`, `BeamSelection`, and run-derived channel masks |
| Numerical dependencies | `requirements.txt` | NumPy, Numba, SciPy, Uproot, Awkward, Matplotlib, pandas, and iminuit |
| Thread-pool helper | `runtime_deps/` | Bundled `threadpoolctl` used by the launchers |

Use a Python environment compatible with the numerical dependencies; the
source uses Python 3.10+ syntax. The production runtime includes Linux-specific
process/cache support. The native photon-scattering backend builds a C++17
shared library with OpenMP, normally using `g++` (`CXX` can select the compiler).
The repository does not pin a complete Python environment.

For the notebooks and interactive 3D displays, also install:

```bash
python3 -m pip install jupyterlab plotly
jupyter lab examples/lf_tutorial.ipynb
```

The tutorial contains CERN EOS data paths. Change those to accessible data
files before running its cells. ROOT input is read with Uproot; the tutorial
does not require a PyROOT import.

## Choose the reconstruction

**Seeding and interaction are two independent settings.** Set both in the
launcher or notebook configuration:

```python
SEEDING_MODE = "general"          # "general" or "beam"
INTERACTION_MODE = "full_length"  # "full_length" or "absorption"
LIKELIHOOD_MODE = "charge_time"   # "charge_time", "charge_only", "timing_only"
FIT_PARTICLE = "muon"             # "muon", "pion", "kaon", "proton"
```

| Seeding | Interaction | Reconstruction |
|---|---|---|
| `beam` | `full_length` | Compact beam-pipe seed bank; one remaining range-to-threshold coordinate |
| `general` | `full_length` | Detector-wide navigation; finite-range tracks clipped to active water, including internal starts and boundary entry/exit |
| `beam` | `absorption` | Compact beam-pipe seed bank; separate visible endpoint and full range |
| `general` | `absorption` | Broad seed coverage; separate visible endpoint and full range |

`beam` selects a search strategy; it does not fix the vertex or direction.
`BEAM_USE_GLOBAL_SEED_GUARD=True` restores broader seed guards for an intentional
off-axis study. Exact physical constraints belong in `FIXED_PARAMETERS`.

The old single `FIT_MODE` interface is retired from the public launchers and
notebook configuration. Its compatibility mapping is:

| Old value | `SEEDING_MODE` | `INTERACTION_MODE` |
|---|---|---|
| `full_length` or `beam` | `beam` | `full_length` |
| `cosmic` or `general` | `general` | `full_length` |
| `absorption` | `general` | `absorption` |

Names beginning with `COSMIC_` and modules named `cosmic_*.py` remain in the
implementation. In the public interface, that engine is selected by
`general` + `full_length`. General seeding with absorption uses the absorption
engine; it does not activate the general/full-length MCS continuation.

### Parameters and units

The nominal track has seven physical degrees of freedom in `full_length` and
eight in `absorption`: three position coordinates, two direction coordinates,
one or two longitudinal coordinates, and event time. Fixed parameters reduce
the number of free coordinates; MCS models can introduce additional latent
process coordinates.

| Quantity | Meaning | Unit |
|---|---|---|
| `x0`, `y0`, `z0` | Fitted start/onset coordinates in the detector geometry frame | mm |
| `cx`, `cy`, `cz` | Cartesian components of the unit track direction | Dimensionless |
| `length` | Range-to-threshold coordinate in `full_length`; visible support in `absorption` | mm |
| `visible_length` | Visible track support reported by the engine, including geometry clipping where applicable | mm |
| `full_range` | Range coordinate that determines the initial energy under the selected range model | mm |
| `t0` | Event time in the fitter's event-relative reference | ns |
| `initial_kinetic_energy_mev` | Initial energy when explicitly returned by a joint energy/range model | MeV |

Directions are optimized in a local tangent chart. `dir_u` and `dir_v` are
internal chart coordinates, not additional physical direction components.
The compact notebook table shows `cx` and `cy`; the full `cz` value remains in
`result.estimates`, including its sign for arbitrary track orientations.

For an entering general/full-length track, the reported onset follows the
active-water boundary convention. A track that exits the detector may have a
remaining range greater than its visible in-water length. Inspect its topology
and range-constraint diagnostics before interpreting the range as a contained
stopping measurement.

### Range conventions

The deterministic tables describe electromagnetic CSDA distance **down to the
Cherenkov threshold**, not distance to rest or hadronic interaction length.
The pion, kaon, and proton tables extend from their first integer energy above
threshold through 3000 MeV. Definitions, inputs, hashes, and reference
comparisons are recorded in [tables/PARTICLE_RANGE_TABLES.json](tables/PARTICLE_RANGE_TABLES.json).

Use the supplied lookup functions for conversions:

```python
from scripts.particle_range_lookup import (
    particle_energy_to_range_mm,
    particle_range_mm_to_energy,
)

range_mm = particle_energy_to_range_mm("muon", 300.0)
kinetic_energy_mev = particle_range_mm_to_energy("muon", range_mm)
```

`APPLY_WCSIM_VISIBLE_RANGE_CONVENTION` optionally adds a calibrated,
reporting-only WCSim muon range conversion. It defaults to `False`, has a
restricted calibration domain, and does not alter the fitted objective or
apply to real WCTE data.

## Batch fitting

Edit the configuration block in the appropriate launcher. Both launchers
translate those settings into the shared driver's environment, using common
validation rules. Their private `_run_*_impl.py` modules handle validation and
execution; normal users need not edit `batch_fit_driver.py`.

### WCSim

In [scripts/run_wcsim.py](scripts/run_wcsim.py), set at least:

```python
INPUT_FILE = "/path/to/muon_sample.npz"
FIT_PARTICLE = "muon"
ENERGY_LABEL_MEV = 300.0
SEEDING_MODE = "general"
INTERACTION_MODE = "full_length"
LIKELIHOOD_MODE = "charge_time"
N_EVENTS = 100
EVENT_START_INDEX = 0
NPROC = 1
OUTPUT_FILE = "outputs/my_wcsim_fits.dict"
```

`ENERGY_LABEL_MEV` describes the sample and supplies beam-seed range guidance;
it does not fix the reconstructed energy in a general-mode fit. Match this
label and the particle hypothesis to your intended study. The checked-in
input path and event counts are examples, not automatic dataset discovery.

```bash
python3 scripts/run_wcsim.py --show-config
python3 scripts/run_wcsim.py --check
python3 scripts/run_wcsim.py
```

The fit reads `digi_hit_pmt`, `digi_hit_charge`, and `digi_hit_time` from the NPZ.
The normal WCTE mapping is [tables/wcsim_wcte_mapping.txt](tables/wcsim_wcte_mapping.txt).
Leave `PMT_ID_MODE`, `PMT_ID_OFFSET`, and `WCSIM_WCTE_MAPPING_FILE` at their
standard values unless the input uses a different ID convention.

`EVENT_START_INDEX` is a zero-based NPZ event index. `N_EVENTS` counts events
starting there. `INACTIVE_SLOTS=None` uses the WCSim detector-mode default;
this is separate from real-data channel masking.

### Real WCTE

In [scripts/run_wcte.py](scripts/run_wcte.py), configure the run, selection,
fit hypothesis, and event limits:

```python
RUN = 2079
COLLABORATION_ROOT_FILE = ""  # Or an explicit merged-production ROOT file.
EVENT_SOURCE = "selection"
GOOD_PMT_SOURCE = "run"
PARTICLE_SELECTION_LABEL = "muon"
LIGHT_PARTICLE_PID_MODE = "act_tof"
FIT_PARTICLE = "muon"
SEEDING_MODE = "general"
INTERACTION_MODE = "full_length"
N_ROOT_ENTRIES = 5000
MAX_EVENTS_TO_FIT = 100
EVENT_START_INDEX = 0
NPROC = 1
OUTPUT_FILE = "outputs/my_wcte_fits.dict"
```

```bash
python3 scripts/run_wcte.py --show-config
python3 scripts/run_wcte.py --check
python3 scripts/run_wcte.py
```

`N_ROOT_ENTRIES` limits the raw readout windows scanned. `EVENT_START_INDEX`
skips entries in the **selected event list**, and `MAX_EVENTS_TO_FIT` limits
how many selected events are fitted after that skip. These indices are distinct
from the original ROOT entry and production event number, which are retained
as provenance. `--check` validates configuration and explicit paths; it does
not perform the ROOT selection or guarantee that every run-specific branch or
calibration will be available at runtime.

### WCTE beam selection

`PARTICLE_SELECTION_LABEL` controls the beam population selected by
`analysis_tools`; `FIT_PARTICLE` controls the reconstruction hypothesis.
Nominal selection supports electrons, muons, pions, and protons. LicketyFit
itself has no electron, positron, or gamma shower hypothesis. Kaon selection
requires `SELECTION_MODE="custom"` and explicit cuts.

| Setting | Behavior |
|---|---|
| `LIGHT_PARTICLE_PID_MODE="act_tof"` | ACT identity plus every usable run-calibrated electron/muon/pion TOF boundary; records a per-boundary ACT fallback when a TOF boundary is unavailable |
| `LIGHT_PARTICLE_PID_MODE="tof"` | TOF-only light-particle identification; required missing boundaries are an error |
| `LIGHT_PARTICLE_PID_MODE="act"` | Legacy ACT identification, with the separately configured proton/fast-particle TOF policy |
| Proton nominal selection | TOF interval starting at the run's proton cut, with width `PROTON_TOF_WINDOW_NS` |
| `TOF_CUT_MODE` | `auto`, `require`, or `disable` for the proton/fast-particle boundary; separate from light-particle PID |
| `SELECTION_MODE="nominal"` | Nominal population cuts plus `EXTRA_SELECTION_CUTS` |
| `SELECTION_MODE="custom"` | `EXTRA_SELECTION_CUTS` defines the beam selection; data-quality controls remain separate |

The launchers also expose mPMT, VME, and T5 quality switches; ACT veto/tagger
switches; a muon-tagger requirement; and explicit ACT/TOF boundary overrides.
Cuts have the form `(branch, operator, value)`, for example:

```python
EXTRA_SELECTION_CUTS = (("T5_particle_nr", "==", 1),)
```

Light-particle TOF boundaries are derived from that run's calibration or an
explicit override. Inspect `fitter.selection_summary()` in a notebook to see
the cuts and any fallback actually applied.

### Input events and active PMTs are independent

| `EVENT_SOURCE` | Event input |
|---|---|
| `selection` | Collaboration ROOT through `DataLoader` and `BeamSelection` |
| `file` | Already-selected `USER_EVENT_FILE`, with optional `USER_EVENT_KEY` |

| `GOOD_PMT_SOURCE` | Active-channel input |
|---|---|
| `run` | `Configuration/good_wcte_pmts` from a run/DQ/merged ROOT |
| `file` | `GOOD_PMT_FILE`, with optional `GOOD_PMT_FILE_KEY` |
| `auto` | User PMT file when supplied; otherwise run-mask discovery |

These choices can be combined freely. `GOOD_PMT_ROOT_FILE` can point to a
separate mask ROOT without replacing the event source. Every listed geometry
PMT is active; unlisted PMTs are disabled. WCSim's inactive-slot list is not
intersected with the real-data mask. Missing geometry IDs are errors; an active
PMT lacking surveyed placement can use its design placement, recorded in
metadata.

User events may be NPY, NPZ, PKL, or PICKLE containers. Each event is an
`N_hit × 3`, `N_hit × 4`, or `N_hit × 5` table:

```text
[global_pmt_id, charge_adc, calibrated_time_ns, root_entry, event_number]
```

The identity columns are optional. Global WCTE IDs are `100*slot + position`.
Good-PMT files accept NPY, NPZ, TXT, CSV, or JSON ID lists or slot/position pairs.
Use trusted files when loading pickle or object-array containers.

## Prompt preparation and detector response

Use `SingleEventFitter.prepare_event()` to obtain exactly the prepared
observables used by the fit. The production prompt selection is not a universal
fixed 0–17 ns cut.

| Source | Default prompt selection |
|---|---|
| WCSim | Find the most populated 1 ns time bin in the early search window (normally 0–100 ns); keep hits strictly between 0 and five bins after the peak bin's left edge, capped at the search boundary |
| WCTE | Find the most populated 1 ns bin in the raw-time search range (normally 0–4000 ns); keep hits strictly from 20 ns before to 5 ns after that bin's left edge, clipped to the search range |

For WCSim, set both `PROMPT_TIME_MIN_NS` and `PROMPT_TIME_MAX_NS` to request a
fixed inclusive window. For WCTE, use `PROMPT_WINDOW_MODE="fixed"` with both
bounds, or `"none"` to retain finite times. User-event files normally receive
the same WCTE prompt cut; `USER_EVENT_APPLY_PROMPT_WINDOW` controls that behavior.
The collaboration loader's broader timing preselection is an additional stage.

WCTE then applies the active-channel mask and subtracts an event time offset.
The default `TIME_REFERENCE_MODE="beam_corrected_peak"` estimates that offset
from the peak of PMT times corrected for photon travel from the reference
point. Prepared-window bounds and `t0` are expressed in this shifted reference.
The removed offset is retained in metadata.

For each active PMT, retained charges are summed and the observed time is the
**charge-weighted mean of the retained hit times**. An active PMT with no
positive prompt charge has zero observed PE and a NaN time. The first-arrival
model is a timing likelihood; the input preparation does not replace this
aggregation with an earliest-hit selection.

Real-data charge is converted from ADC to PE using `CHARGE_ADC_PER_PE` (default
143). `RELATIVE_EFFICIENCY_MODE` chooses `slot`, `type`, or `none` for the
model's mPMT efficiency treatment; the default is `slot`. `GEOMETRY_PLACEMENT`
selects surveyed `est` or nominal `design` geometry. Efficiency response is
applied in the prediction, rather than by manually rescaling the notebook's
observed-PE arrays.

## Physics and likelihood

The forward model predicts the light pattern from the selected particle,
range, trajectory, PMT geometry, and optical response. The main stages are
seed ranking, exact track optimization, optional mode-specific MCS continuation,
and final observed/predicted PMT diagnostics. The current track optimizer uses
track-aligned block/quadratic updates and adaptive continuations; a historical
`minuit_valid` output column does not imply that this path ran Minuit.

Both launchers expose the same process switches:

```python
ENABLE_DELTA_ELECTRONS = True
ENABLE_MCS = True
ENABLE_REFLECTION = True
ENABLE_PHOTON_SCATTERING = True
```

Direct primary Cherenkov light is combined with the enabled delta-electron,
photon-scattering, and reflection models. MCS is routed to the implementation
appropriate to the selected reconstruction. The general engine also contains
explicit hypotheses for charged tracks crossing mPMT hardware.

`charge_only` uses the PMT charge likelihood; `timing_only` uses timing; and
`charge_time` combines them. First-arrival timing uses source-resolved optical
arrival information and prompt conditioning. The notebook's scalar
`expected_time_ns` is a diagnostic summary of that prediction; the likelihood
uses more information than a squared residual against that scalar alone.

### Charge shape versus absolute light yield

| Setting | Charge treatment |
|---|---|
| `USE_ABSOLUTE_LIGHT_YIELD=False` | Profile out the event's overall light normalization; fit its charge distribution |
| `USE_ABSOLUTE_LIGHT_YIELD=True` | Retain calibrated total light and select `compound_spe_calibrated`, including the detector's censored single-PE response |

The checked-in WCSim launcher defaults to absolute light **on**; WCTE defaults
to **off**. With absolute light off, `CHARGE_LIKELIHOOD="poisson_pe"` is the
normal default and `"compound_spe_profile"` is available for the censored
compound response.

| Data source | `ABSOLUTE_LIGHT_YIELD_SOURCE` | Calibration |
|---|---|---|
| WCSim | `wcsim_calibration` | Packaged `tables/wcsim_muon_absolute_light_direct_shape_v3.json` by default |
| WCSim | `mathematical` | Packaged ground-up estimate including the audited WCSim QE factors |
| WCTE | `measured` | User-supplied WCTE `GLOBAL_CHARGE_CALIBRATION_MANIFEST` |
| WCTE | `mathematical` | Packaged WCTE ground-up estimate with the WCTE QE convention |

Manifests bind the scale to the particle hypothesis, detector response, and
model context. The packaged absolute-light manifests are for muons. For pion,
kaon, or proton fits, use shape-only operation or supply a compatible
particle-specific manifest; the runtime rejects a particle mismatch. A WCSim
manifest is not a real-WCTE calibration. The WCTE mathematical estimate is an
engineering prediction, not an in-situ measurement. When changing physics or
calibration settings, keep the chosen manifest compatible with them.

Enabled reflection participates in timing on supported geometry. Its inclusion
in absolute charge is specified by the manifest; the normal shape-only policy
excludes analytic reflected charge. Thus `ENABLE_REFLECTION=True` does not by
itself mean reflected charge is added to the fitted charge pattern.

### MCS routing and compatibility

| Reconstruction | Default MCS route when `ENABLE_MCS=True` |
|---|---|
| `general` + `full_length` | `COSMIC_MCS_CONTINUATION="coherent_fermi_eyges"` |
| `beam` + `full_length` | `PRIMARY_MCS_MODEL="coherent_fisher"`, with `COHERENT_MCS_IMPLEMENTATION="auto"` resolving to `physics_reference` |
| Either seeding + `absorption` | `PRIMARY_MCS_MODEL="coherent_fisher"`, with `auto` resolving to `standard_fe_process` |

The general/full-length route also offers `off`, `linear_fermi_eyges`,
`joint_k0_range_gaussian_fe`, and experimental `joint_k0_range_mixed_mcs`.
The mixed model requires the reference SMC inference engine and has additional
physical-support and convergence requirements. Other routes expose
`fermi_eyges_process`, `legacy`, and explicit coherent implementations subject
to the shared compatibility checks.

For common alternative likelihoods:

- **General + full-length + charge-only:** set
  `COSMIC_MULTILATERATION_SEED_MODE="off"`; timing-derived navigation is not
  allowed for this combination.
- **Timing-only:** `ENABLE_MCS=False` is the straightforward compatible setting.
  General/full-length can instead set `COSMIC_MCS_CONTINUATION="off"`; the
  charge-based MCS continuations cannot be used in a timing-only fit.
- **Absorption with coherent primary MCS:** use `auto` or
  `standard_fe_process`; other coherent endpoint implementations are rejected.

Inspect `config.reconstruction()` to see the resolved selectors before
initialization. The rules are implemented in
[LicketyFit/run_configuration.py](LicketyFit/run_configuration.py).

### Fixed physical parameters

Use the same dictionary in either launcher or the notebook configuration:

```python
FIXED_PARAMETERS = {
    "x0_mm": 0.0,
    "y0_mm": 0.0,
    "z0_mm": -1000.0,
    "kinetic_energy_mev": 300.0,
}
```

This example is valid for absorption or general/full-length reconstruction;
the coordinates are illustrative and must be in your detector frame.

| Reconstruction | Accepted longitudinal constraints |
|---|---|
| `beam` + `full_length` | `length_mm` |
| `general` + `full_length` | One of `length_mm`, `full_range_mm`, or `kinetic_energy_mev` |
| Either seeding + `absorption` | `visible_length_mm`, and one of `full_range_mm` or `kinetic_energy_mev` |

Every combination also accepts `x0_mm`, `y0_mm`, `z0_mm`, `t0_ns`, and a complete
`direction=(dx, dy, dz)` vector, normalized by the configuration layer. Do not
supply both range and energy aliases for the same coordinate. Leave the
dictionary empty to fit all available coordinates.

## Interactive event fitting

[examples/lf_tutorial.ipynb](examples/lf_tutorial.ipynb) walks through WCSim
truth displays, WCSim fits, and WCTE selection/fits.
[examples/example.ipynb](examples/example.ipynb) provides additional configuration,
diagnostics, and cache examples. The interface lives in
[scripts/single_event_fit.py](scripts/single_event_fit.py).

From a notebook in the repository root or `examples/`:

```python
from pathlib import Path
import sys

repo = next(
    p for p in (Path.cwd(), *Path.cwd().parents)
    if (p / "scripts" / "single_event_fit.py").is_file()
)
sys.path.insert(0, str(repo / "scripts"))

from single_event_fit import WCSimConfig, WCTEConfig, SingleEventFitter, summarize_fit
from plot_event import plot_event

config = WCSimConfig(
    input_file="/path/to/muon_sample.npz",
    fit_particle="muon",
    seeding_mode="general",
    interaction_mode="full_length",
    likelihood_mode="charge_time",
    use_absolute_light_yield=False,
    n_events=10,
    event_start_index=0,
)
fitter = SingleEventFitter(config)
events = fitter.load_events()
fitter.initialize()  # Reused by subsequent fits with this fitter.

event = events[0]
prepared = fitter.prepare_event(event)
plot_event(prepared, quantity="observed_pe")

result = fitter.fit(event)
statistics = summarize_fit(result)
plot_event(result, quantity="expected_pe");
```

For real data, replace the configuration with, for example:

```python
config = WCTEConfig(
    run=2079,
    event_source="selection",
    good_pmt_source="run",
    particle_selection_label="muon",
    light_particle_pid_mode="act_tof",
    fit_particle="muon",
    seeding_mode="general",
    interaction_mode="full_length",
    likelihood_mode="charge_time",
    n_root_entries=1000,
    max_events_to_fit=10,
)
```

Then create a new `SingleEventFitter(config)` and repeat the loading and fitting
steps. Configuration option names are case-insensitive and mirror the public
launcher. Useful inspection methods include:

```python
WCTEConfig.search_options("tof")
WCSimConfig.available_options()
config.changed_options()
config.reconstruction()
config.validate(check_paths=True)
```

The fitter copies its configuration when constructed. Create a new fitter to
apply configuration changes; reuse the existing fitter when only changing the
event. Notebook execution fits one event in the current process and does not
write batch checkpoints. It can still use internal numerical threads.

### Summary output

`summarize_fit(result)` displays a physical-parameter table with estimates and
reported errors, plus **only these five fit diagnostics**, in this order:

| Field | Definition |
|---|---|
| `minimum_valid` | The production fit was accepted and its returned objective is finite |
| `fval` | The objective value returned by the selected fitting route |
| `fit_wall_s` | The driver's recorded per-event fit time in seconds |
| `n_active_pmts` | Number of PMTs in the active observation arrays, including unhit PMTs |
| `n_observed_pmts` | Number of active PMTs with positive summed prompt charge |

The parameter table includes the available `x0`, `y0`, `z0`, `cx`, `cy`,
`length`, `visible_length`, `full_range`, `initial_kinetic_energy_mev`, and `t0`
entries. It omits internal direction-chart coordinates and topology metadata.
`FitResult.summary()` returns the same five diagnostics; `summarize_fit()`
returns them as a pandas Series.

`minimum_valid` is an acceptance/finite-objective flag, not proof of a global
minimum or a goodness-of-fit probability. `fval` is not a reduced chi-squared.
`fit_wall_s` excludes the separate initialization and the notebook helper's
subsequent diagnostic-prediction work; a cold first fit may include compilation.
An unavailable reported parameter uncertainty appears as NaN.

The complete data remain accessible:

```python
result.estimates
result.errors
result.fit_statistics
result.raw_result
result.pmt_table()
```

`result.pmt_table()` aligns PMT IDs, geometry, observed and expected PE, observed
and expected times, and timing-use flags. `EventRecord.hit_table()` gives raw
loaded hits; `PreparedEvent.hit_table()` gives PMTs with positive prepared
charge. Plot the prepared event when you want to see what entered the fit.

### WCSim truth and displays

Set `use_truth_root=True` with a matching `truth_root_file` for optional
`AllSecondaries` diagnostics. `wcsim_npz_primary_truth()` reads primary truth
from an NPZ, and `truth_residuals()` supports post-fit comparisons. Truth
information does not enter the fit's event selection, seeds, constraints, or
likelihood unless you explicitly choose to pass a physical constraint yourself.

`scripts/display_3D_event.py` uses `AllSecondaries` and `AllSecondaryPhotons`
dataframes for interactive track/photon displays. These truth-display time
cuts are separately configured; they do not set the fitter's prompt window.
If an inline Plotly figure does not appear, the tutorial includes an HTML
export using `fig.write_html(..., include_plotlyjs=True)`.

## Geometry support

WCTE is the default detector. WCSim runs expose `USE_WCTE_GEOMETRY` and
`USE_IWCD_GEOMETRY`; select exactly one. IWCD requires an explicit compatible
`GEOMETRY_FILE`. An alternative serialized geometry still uses the top-level
Geometry submodule's Python classes.

The direct-light and convex-track machinery is geometry-aware, but the
blacksheet reflection transfer is WCTE-specific. For non-WCTE geometry it is
disabled by the default detector policy; molecular scattering uses an
mPMT-plane approximation to the convex boundary. This is not a claim of full
IWCD optical validation. The notebook's supplied 2D event display is also a
WCTE layout.

## Startup, performance, and failure handling

Initialization prepares geometry, tables, optical transfers, and seed/proxy
libraries. First use also compiles numerical kernels. These caches depend on
the source, configuration, Python environment, and—in the native backend—CPU.
Do not interpret a cold first-event time as steady-state fit latency.

To warm the same configuration outside Jupyter:

```bash
python3 scripts/warm_up_cache.py wcsim --input /path/to/muon_sample.npz \
    --set fit_particle=muon --set seeding_mode=general \
    --set interaction_mode=full_length --set use_absolute_light_yield=False
```

There is an analogous `wcte --run 2079` interface. Repeat `--set NAME=VALUE` for
settings that differ from the launcher. `runtime_cache_report()` inspects the
cache; the example notebook also covers cache bundles. Batch startup has its
own cache/bootstrap handling. WCSim exposes `AUTO_BOOTSTRAP_RUNTIME_CACHE`,
`RUNTIME_CACHE_DIR`, and optional node-local staging controls.

`NPROC` counts concurrent **event workers**, not total CPU threads. For WCSim,
`MAX_INTERNAL_THREADS_PER_WORKER` defaults to 4 when using one worker, with
thread counts restricted for multiple workers. The notebook sets one event
worker with a four-thread Numba budget. For CPU-constrained benchmarking,
record the actual thread budget or process affinity as well as `NPROC`.

| Setting | Effect |
|---|---|
| `N_EVENTS_PER_BATCH` | Batch/checkpoint grouping |
| `SAVE_AFTER_EACH_BATCH` | Save completed progress during the run |
| `SAVE_DETAILED_EVENT_RESULTS` | Include the large per-event diagnostic payloads |
| `CONTINUE_AFTER_EVENT_FAILURE` | Batch mode records a rejected row and failure details, then continues |
| `RETAIN_STRAIGHT_ON_MCS_FAILURE` | Retain an accepted straight fit if its optional MCS stage fails, recording the fallback |
| `EVENT_RESULT_STALL_TIMEOUT_SECONDS` | Bound a multiprocessing interval with no returned result |

The single-event API executes the fitting path directly; exceptions can
propagate to the notebook. Consult `result.fit_statistics` or batch MCS status
columns when distinguishing a complete MCS result from a retained straight fit.
For a Linux stall investigation, run `python3 scripts/lf_diagnose_stall.py` in
another terminal on the same node.

## Batch output

The `.dict` output is a pickled Python dictionary, with event-aligned arrays
and a `metadata` record:

```python
import pickle
import numpy as np

with open("outputs/my_wcsim_fits.dict", "rb") as stream:
    output = pickle.load(stream)

accepted = np.asarray(output["fit_accepted"], dtype=bool)
x_mm = np.asarray(output["x"])
fcn = np.asarray(output["est_fcn"])
```

Common fields include `x`, `y`, `z`, `cx`, `cy`, `cz`, `length`,
`visible_length`, `full_range`, `t`, `est_fcn`, `event_fit_wall_s`,
`fit_accepted`, `mcs_status`, and `mcs_applied`. Additional fields depend on
source and reconstruction route. Batch names differ from notebook names:
for example `x` versus `x0`, `t` versus `t0`, and `est_fcn` versus `fval`.

Metadata records reconstruction choices, input/event provenance, geometry,
calibration, prompt preparation, runtime settings, and launcher identity/hash.
Keep it with the estimates. Check acceptance, failure records, MCS status, and
range-constraint information before selecting events for an analysis.

## Source map

| Location | Responsibility |
|---|---|
| `scripts/run_wcte.py`, `scripts/run_wcsim.py` | Public batch configurations |
| `scripts/_run_wcte_impl.py`, `scripts/_run_wcsim_impl.py`, `scripts/launcher_loader.py` | Launcher validation, environment construction, and execution |
| `scripts/batch_fit_driver.py` | Shared dispatch and embedded WCTE/WCSim standard/general engines |
| `scripts/single_event_fit.py` | Notebook configuration, event preparation, fitting, and result tables |
| `scripts/wcte_data_loader_adapter.py`, `scripts/wcte_user_event_file.py` | Collaboration selection, channel masks, and user-event containers |
| `scripts/read_sim_data.py` | Selective WCSim NPZ loading |
| `LicketyFit/Emitter.py`, `LicketyFit/PMT.py` | Optical predictions and charge/timing likelihoods |
| `LicketyFit/fast_track_fit.py`, `LicketyFit/cosmic_track_fit.py`, `LicketyFit/adaptive_exact.py` | Geometry constraints, track optimization, and continuation |
| `LicketyFit/track_parameterization.py`, `LicketyFit/cosmic_navigation.py`, `LicketyFit/multilateration_seeding.py` | Direction charts and general-track navigation |
| `LicketyFit/mcs_*.py`, `LicketyFit/cosmic_*mcs*.py` | Fermi–Eyges paths, response derivatives, profiling, and alternative energy/range inference |
| `LicketyFit/mpmt_*.py`, `LicketyFit/optical_obstacles.py` | mPMT crossing hypotheses and detector-obstacle optics |
| `LicketyFit/photon_scattering_*.py`, `LicketyFit/photon_scattering_native.cpp` | Molecular scattering and its native execution backend |
| `LicketyFit/absolute_light_calibration.py`, `LicketyFit/ground_up_light_calibration.py`, `LicketyFit/wcsim_charge_response.py` | Calibration contracts and detector charge response |
| `LicketyFit/detector_geometry.py`, `LicketyFit/particle_cherenkov_model.py`, `LicketyFit/Event.py` | Detector/particle conventions and event containers |
| `LicketyFit/wcsim_truth.py`, `LicketyFit/wcsim_range_convention.py` | Optional truth diagnostics and reporting conversion |
| `LicketyFit/run_configuration.py`, `LicketyFit/fixed_parameters.py`, `LicketyFit/mcs_configuration.py` | Shared configuration contracts |
| `LicketyFit/runtime_cache.py`, `scripts/runtime_bootstrap.py`, `scripts/warm_up_cache.py` | Generated cache and warm-up management |
| `scripts/plot_event.py`, `scripts/display_3D_event.py`, `event_display/` | Event visualization |
| `scripts/particle_range_lookup.py`, `scripts/generate_hadron_range_tables.py`, `tables/` | Range conversion and packaged physics/response data |
| `examples/` | Tutorial and expanded notebook examples |

The source tree includes research/alternative continuations and compatibility
helpers as well as the default production route. Their presence alone does
not make them active; the resolved configuration determines the execution path.

## Included validation utilities

These commands exercise the packaged contracts without a full event-fit
campaign:

```bash
python3 scripts/validate_wcte_selection_controls.py
python3 scripts/validate_wcte_user_file_controls.py
python3 scripts/validate_particle_range_tables.py
```

The user-file validator optionally accepts an event-file path. The range-table
validator checks the packaged tables and primary-model loading. For a new
dataset or configuration, also inspect prepared events and fit diagnostics
before launching a large batch.
