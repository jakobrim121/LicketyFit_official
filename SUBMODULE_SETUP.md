# LicketyFit runtime submodules

LicketyFit uses two direct, pinned source repositories:

| Path | Repository | Used for |
|---|---|---|
| `analysis_tools` | `https://github.com/WCTE/analysis_tools.git` | WCTE collaboration ROOT loading, beam selection, and run-derived good-PMT masks |
| `Geometry` | `https://github.com/WCTE/Geometry.git` | Geometry Python classes and the default WCTE `wcte_bldg157.geo` |

The fitter deliberately does not use a globally installed copy, a personal
checkout, or `analysis_tools/extern/Geometry`.

## Add both submodules to the parent repository

Run these commands from the `LicketyFit_official` Git repository root. If
`analysis_tools` has already been added, omit its `git submodule add` command.

```bash
git submodule add https://github.com/WCTE/analysis_tools.git analysis_tools
git submodule add https://github.com/WCTE/Geometry.git Geometry

git -C analysis_tools checkout 236608ec295908b6436f8b1fd584bdc8651af9a3
git -C Geometry checkout ed2164f6ac9e72c2d03b5fcfc3a6e059b69b5df0

git add .gitmodules analysis_tools Geometry
git commit -m "Add pinned WCTE runtime submodules"
```

Those are the exact upstream revisions used for the v1.23 release validation.
The parent commit records the two gitlinks, so collaborators receive the same
source versions rather than whatever happens to be upstream later.

## Initialize an existing clone

```bash
git submodule update --init analysis_tools Geometry
```

For a new checkout, prefer:

```bash
git clone <your-LicketyFit-repository-URL>
cd LicketyFit_official
git submodule update --init analysis_tools Geometry
```

Do not use `--recursive` for normal LicketyFit setup. The `analysis_tools`
repository declares nested `T5_analysis`, `TimeCal`, and Geometry submodules,
but the LicketyFit adapter imports only its DataLoader and BeamSelection source
files. Recursive initialization would fetch code LicketyFit does not use and
may require GitHub SSH credentials.

## Verify the checkout

```bash
git submodule status analysis_tools Geometry
test -f analysis_tools/analysis_tools/data_loader.py
test -f analysis_tools/analysis_tools/beam_selection.py
test -f Geometry/Geometry/Device.py
test -f Geometry/examples/wcte_bldg157.geo
```

Then run a launcher check after setting its required input file:

```bash
python3 scripts/run_wcte.py --check
# or
python3 scripts/run_wcsim.py --check
```

## Why the release ZIP does not contain the submodule directories

A ZIP stores ordinary files and directories; it cannot represent Git's gitlink
object. This source archive therefore leaves `analysis_tools/` and `Geometry/`
unvendored. Extract it into the parent checkout, preserve or create the two
gitlinks with the commands above, and initialize them before running the fitter.

An explicit `GEOMETRY_FILE` may select a different serialized detector, but the
Python classes that open it still come from the pinned top-level `Geometry`
submodule. External `GEOMETRY_PATH`, `ANALYSIS_TOOLS_PATH`, and
`WCTE_ANALYSIS_TOOLS_PATH` source overrides are rejected.
