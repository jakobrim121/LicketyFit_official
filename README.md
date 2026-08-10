# LicketyFit

LicketyFit is an analytic single-track Cherenkov fitter for WCTE/IWCD detector
geometries. It supports digitized WCSim samples and calibrated real WCTE data,
with charge-only, timing-only, and joint charge-time likelihoods.

The current public driver release is:

```text
2026-08-10-three-mode-nonmcs-universal-v1.21-separate-run-configs
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

Create or activate a Python environment, then install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The release archive includes its detector geometry repository beside this
directory. The WCTE collaboration-selection workflow additionally needs the
collaboration `analysis_tools` checkout or installed package. Set
`ANALYSIS_TOOLS_PATH` near the end of `scripts/run_wcte.py` when it is not in the
standard CERN location.

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
```

Release-level validation and integrity information are supplied in the outer
archive as `VALIDATION_2026-08-10.txt`, `RELEASE_MANIFEST.txt`, and
`SHA256SUMS.txt`.

## Repository layout

```text
LicketyFit/                 likelihood, detector, optical, and optimizer modules
scripts/run_wcte.py         real-WCTE user configuration and launcher
scripts/run_wcsim.py        WCSim user configuration and launcher
scripts/batch_fit_driver.py unified implementation used by both launchers
scripts/                    input adapters and validation utilities
tables/                     range, response, receiver, mapping, and proxy tables
docs/                       run-2079 investigation sources and supporting material
```

## Important qualification boundary

The frozen run-2079 investigation predates the authoritative-PMT correction. Its
DQ list contained 1,579 PMTs, while the old fit incorrectly used 1,560 after a
WCSim-only slot intersection. Those numerical real-data fits must be repeated
with the corrected mask before being treated as final. See
`../WCTE_ACTIVE_PMT_POLICY_ERRATUM_2026-08-10.md` in the release archive.

This repository snapshot does not include a software license. Add the
appropriate collaboration-approved license before public redistribution.
