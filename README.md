# LicketyFit

LicketyFit is an analytic single-track Cherenkov fitter for WCTE/IWCD detector
geometries. It supports digitized WCSim samples and calibrated real WCTE data,
with charge-only, timing-only, and joint charge-time likelihoods.

The current public driver release is:

```text
2026-08-10-three-mode-nonmcs-universal-v1.23-portable-runtime-submodules
```

## Start here

Users should not edit `scripts/batch_fit_driver.py`. Choose one launcher, edit
its clearly marked configuration section, and run it:

```bash
python3 scripts/run_wcte.py
```

or:

```bash
python3 scripts/run_wcsim.py
```

The launchers execute `batch_fit_driver.py` directly. They do not duplicate the
fit implementation. Launcher values override matching variables left in the
shell environment, which makes a run reproducible from the edited file alone.

Before running a fit, the launcher can display or check its configuration:

```bash
python3 scripts/run_wcte.py --show-config
python3 scripts/run_wcte.py --check

python3 scripts/run_wcsim.py --show-config
python3 scripts/run_wcsim.py --check
```

`--check` validates choices and explicit local file paths. It cannot guarantee
that a run ROOT will later be visible through EOS discovery or that a batch node
will retain access to an external service.

## Installation

Clone the repository with its two direct runtime submodules, or initialize them
in an existing clone:

```bash
git clone <your-LicketyFit-repository-URL>
cd LicketyFit_official
git submodule update --init analysis_tools Geometry
```

Do not use `--recursive` for normal setup. LicketyFit does not use the nested
submodules declared by `analysis_tools`, and those repositories may require
separate credentials. Then create or activate a Python environment and install
the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Geometry classes/files are loaded only from the pinned top-level `Geometry`
submodule. WCTE collaboration loading, beam selection, and run-mask discovery
are loaded only from the pinned top-level `analysis_tools` submodule. Personal
checkout paths and globally installed copies are intentionally not used. See
`SUBMODULE_SETUP.md` for the pinned revisions and verification commands.

## Interactive single-event notebook

`examples/fit_single_event_tutorial.ipynb` is the recommended starting point
for event-by-event studies. Start Jupyter from the repository root or
`examples/`, then open the notebook. If the environment does not already supply
Jupyter, install it separately (for example, `python3 -m pip install jupyterlab`).

The notebook-facing API is `scripts/single_event_fit.py`. It exposes
`WCTEConfig` and `WCSimConfig`, each of which accepts every public option in the
matching run launcher. `SingleEventFitter` calls the selected embedded engine in
`batch_fit_driver.py`; it does not maintain a simplified second fitter. It uses
the production loaders, active-PMT policy, prompt preparation, seed bank,
likelihood, optimizer, and final prediction path.

The tutorial demonstrates how to:

- load a small collection with the production WCTE or WCSim loader;
- select and plot one event with `scripts/plot_event.py`;
- fit that event while reusing initialized fitter state for later events;
- inspect the accepted/finite status, FCN, objective evaluations, fit attempts,
  optimizer sweeps, selected seed, topology, and wall time; and
- compare PMT-aligned post-cut observed PE/time with expected PE and the model's
  first-arrival time summary at the final track estimate.

The current optimizer is the track-aligned block/quadratic optimizer, not
Minuit. The batch field `minuit_valid` is retained only as a historical output
compatibility alias, so the notebook reports the optimizer's actual diagnostics.

## Fit modes

Both launchers expose the same three public modes:

| `FIT_MODE` | Meaning |
|---|---|
| `full_length` | Seven-parameter internal-start fit. The longitudinal coordinate is the remaining stopping range; visible light is clipped at the detector boundary when necessary. |
| `absorption` | Eight-parameter abrupt-endpoint fit with separate visible length and full range/initial kinetic energy. |
| `cosmic` | Geometry-clipped fit supporting inside/outside start and stop combinations. |

Supported fit hypotheses are `muon`, `pion`, `kaon`, and `proton`. The WCTE
selection adapter can select electron events, but LicketyFit does not currently
have an electron track hypothesis; an electron-selected sample must be tested
under an explicitly chosen supported hypothesis.

## Particle range tables

The package now includes working liquid-water range tables for all four fit
hypotheses. The pion, kaon, and proton tables span every integer initial kinetic
energy above the species' Cherenkov threshold through 3 GeV. They use the PDG
Bethe stopping-power treatment with finite-mass and density-effect corrections
and the modern ICRU-90/Geant4 liquid-water mean excitation energy of 78 eV.

These are electromagnetic CSDA distances from the initial energy down to the
particle's Cherenkov threshold. They are not hadronic interaction/decay survival
lengths and are not full ranges to rest. See
[`docs/PARTICLE_RANGE_TABLES.md`](docs/PARTICLE_RANGE_TABLES.md) for the exact
definition, checkpoint values, comparison with PDG water constants and NIST
PSTAR, limitations, reproduction instructions, and source links.

## Real WCTE data

Edit `scripts/run_wcte.py`. Event provenance and active-PMT provenance are
independent:

| `EVENT_SOURCE` | `GOOD_PMT_SOURCE` | Events | Authoritative active PMTs |
|---|---|---|---|
| `selection` | `run` | DataLoader/BeamSelection events from the collaboration ROOT for `RUN` | `Configuration/good_wcte_pmts` for `RUN` |
| `selection` | `file` | Collaboration-selected events | User `GOOD_PMT_FILE` |
| `file` | `run` | Already-selected `USER_EVENT_FILE` events | Run DQ/merged ROOT mask |
| `file` | `file` | Already-selected `USER_EVENT_FILE` events | User `GOOD_PMT_FILE` |

`GOOD_PMT_SOURCE="auto"` uses a nonblank `GOOD_PMT_FILE`; otherwise it performs
run-root discovery.

### Authoritative PMT rule

For real WCTE data, the resolved user or run/DQ list is the complete active set:

- every listed geometry PMT is enabled;
- every unlisted PMT is disabled;
- the historical WCSim inactive-slot list is never applied;
- a listed PMT absent from detector geometry is a configuration error; and
- an active PMT without a surveyed `est` placement uses its design coordinates,
  with the fallback recorded in output metadata.

User good-PMT files may be NPY, NPZ, TXT, CSV, or JSON and may contain global
`100*slot + position` IDs or slot/position pairs.

Already-selected event files may be NPY, NPZ, PKL, or PICKLE. The recommended
per-hit rows are:

```text
PMT ID, calibrated charge ADC, calibrated time ns, ROOT entry, event number
```

Legacy three- and four-column layouts remain supported. Object-array NPY and
pickle files must come from a trusted source because loading them invokes Python
pickle machinery.

## WCSim data

Edit `scripts/run_wcsim.py`. The most important settings are `INPUT_FILE`,
`N_EVENTS`, `FIT_MODE`, `FIT_PARTICLE`, `LIKELIHOOD_MODE`, and the detector
geometry choice.

Optional `AllSecondaries` ROOT input supplies per-event truth diagnostics only.
It does not seed, constrain, or otherwise enter the likelihood. The ROOT must
correspond to the same WCSim sample as the digitized NPZ.

The historical inactive-slot list is retained for WCSim. Set
`INACTIVE_SLOTS=None` to use the established detector-mode default. This policy
is deliberately separate from real-WCTE channel masking.

## Physics switches

The process-model switches remain centralized near the top of
`LicketyFit/Emitter.py`, including delta electrons, primary-particle multiple
scattering, molecular scattering, and blacksheet reflection. The two launchers
configure datasets and fits; they do not silently alter those physics switches.

## Output and reproducibility

Output is a Python dictionary containing event fit arrays plus a `metadata`
record. When a launcher is used, metadata includes:

- the launcher kind (`wcte` or `wcsim`);
- the absolute launcher path;
- the launcher SHA-256 at execution time; and
- the public and embedded engine release identifiers.

WCTE output also records event/mask provenance, the complete active-PMT ID list,
geometry-placement fallbacks, calibration settings, and whether collaboration
selection stages were actually applied.

## Validation commands

The package includes contract checks for real-data selection and user files:

```bash
python3 scripts/validate_wcte_selection_controls.py
python3 scripts/validate_wcte_user_file_controls.py /path/to/events.npy
python3 scripts/validate_particle_range_tables.py
```

Release-level validation and integrity information are supplied in the outer
archive as `VALIDATION_2026-08-11.txt`, `RELEASE_MANIFEST.txt`, and
`SHA256SUMS.txt`.

## Repository layout

```text
LicketyFit/                 likelihood, detector, optical, and optimizer modules
scripts/run_wcte.py         real-WCTE user configuration and launcher
scripts/run_wcsim.py        WCSim user configuration and launcher
scripts/batch_fit_driver.py unified implementation used by both launchers
scripts/single_event_fit.py notebook API for production single-event fits
scripts/plot_event.py       portable raw/observed/expected event display helper
examples/fit_single_event_tutorial.ipynb interactive WCTE and WCSim tutorial
scripts/                    input adapters and validation utilities
tables/                     range, response, receiver, mapping, and proxy tables
docs/                       physics definitions, sources, and supporting material
```

## Important qualification boundary

The frozen run-2079 investigation predates the authoritative-PMT correction. Its
DQ list contained 1,579 PMTs, while the old fit incorrectly used 1,560 after a
WCSim-only slot intersection. Those numerical real-data fits must be repeated
with the corrected mask before being treated as final. See
`../WCTE_ACTIVE_PMT_POLICY_ERRATUM_2026-08-10.md` in the release archive.

This repository snapshot does not include a software license. Add the
appropriate collaboration-approved license before public redistribution.
