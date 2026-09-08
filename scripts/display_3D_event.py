"""
wcte_event_display_legend_pmt_linked.py

Reusable interactive 3D WCTE event display for WCSim particle tracks and
Cherenkov-photon PE information.

Expected inputs
---------------
Pass existing pandas DataFrames or dict-of-arrays:

    df_sec
        Particle/secondary track-step table. Must contain the same columns as
        all_particles_pi+_500MeV.csv.

    df_sec_ph
        Cherenkov photon table. Must contain the same columns as
        all_photons_pi+_500MeV.parquet.

Typical notebook usage
----------------------

    from wcte_event_display import WCTEEventDisplay

    display3d = WCTEEventDisplay(
        df_sec,
        df_sec_ph,
        geometry_parent="/path/to/folder/containing/Geometry/or/geometry_py_files",
    )

    fig, summary, chosen, pmt_pe_df = display3d.plot_event(
        event_id=0,
        time_cut_ns=20.0,
        min_creation_ke_MeV=100.0,
        apply_min_track_length_cut=False,  # show all track lengths, including very short tracks
        show_electrons=True,
        show_mpmt_slot_numbers=True,
        pmt_display_mode="all",     # "all" or "hit_only"
    )

    display(summary.head(30))
    display(pmt_pe_df.head(30))
    fig.show()

One-shot usage
--------------

    from wcte_event_display import plot_wcsim_event

    fig, summary, chosen, pmt_pe_df = plot_wcsim_event(
        df_sec,
        df_sec_ph,
        event_id=0,
        geometry_parent=".",
    )

Notes
-----
Coordinates in df_sec are assumed to be WCSim coordinates in cm.
The WCTE geometry package uses mm. The conversion applied here is:

    x_geom_mm = 10*x_wcsim_cm
    y_geom_mm = 10*y_wcsim_cm + y_wcsim_origin_in_geom_mm
    z_geom_mm = 10*z_wcsim_cm

The default offset is 424.763 mm, i.e. WCSim y=0 is 42.4763 cm above
the geometry-package y=0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Set, Tuple, Union
import shutil
import sys
import html
import textwrap
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


Y_WCSIM_ORIGIN_IN_GEOM_MM_DEFAULT = 424.763

DEFAULT_INTERACTION_PROCESSES = {
    "Decay",
    "RadioactiveDecay",
    "pi+Inelastic",
    "pi-Inelastic",
    "neutronInelastic",
    "protonInelastic",
    "dInelastic",
    "tInelastic",
    "hadElastic",
    "nCapture",
    "conv",
    "annihil",
}

DEFAULT_EM_EXTRA_PROCESSES = {
    "compt",
    "phot",
    "eBrem",
    "eIoni",
    "muIoni",
    "hIoni",
    "ionIoni",
    "CoulombScat",
}

NEUTRINO_PDGS = {12, -12, 14, -14, 16, -16}
ELECTRON_PDGS = {11, -11}

# By default, always keep pion-decay muons in the display even if they
# are below the normal kinetic-energy/track-length thresholds.
# PDG convention: mu- = 13, mu+ = -13.
DEFAULT_DECAY_DAUGHTER_PDGS = {-13, 13}

# Processes that can appear as the step-limiting process but should not usually
# be interpreted as the physical process that killed/stopped the particle track.
OPTICAL_BOOKKEEPING_PROCESSES = {"Scintillation", "Cerenkov", "OpAbsorption", "OpRayleigh", "OpBoundary"}

# A loose threshold used to decide whether a charged particle has effectively
# reached rest in the saved WCSim step table.
DEFAULT_PHYSICAL_REST_KE_THRESHOLD_MEV = 1.0

# PDGs for particles whose stopped positive state normally decays rather than
# being a stable ranged-out particle. We still prefer explicit daughter evidence.
STOPPED_POSITIVE_DECAY_PDGS = {211, 321, 13}      # pi+, K+, mu-
STOPPED_NEGATIVE_CAPTURE_PDGS = {-211, -321}      # pi-, K- in material are commonly captured
STABLE_RANGEOUT_PDGS = {2212, -2212, 1000010020, 1000010030, 1000020040}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def ensure_dataframe(obj: Union[pd.DataFrame, Dict[str, Any]], name: str = "dataframe") -> pd.DataFrame:
    """Accept a pandas DataFrame or dict-of-arrays and return a copy as DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    raise TypeError(f"{name} must be a pandas DataFrame or dict-of-arrays, got {type(obj)}")


def add_geometry_package_to_path(geometry_parent: Union[str, Path]) -> Path:
    """
    Make `from Geometry.WCD import WCD` work.

    Supports either:
      1. a parent folder containing a Geometry/ package,
      2. a folder directly containing Device.py, WCD.py, etc.,
      3. a folder containing uploaded files like Device(12).py, WCD(12).py, etc.
    """
    geometry_parent = Path(geometry_parent).expanduser().resolve()

    if (geometry_parent / "Geometry").is_dir():
        path_str = str(geometry_parent)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        return geometry_parent / "Geometry"

    module_names = [
        "Device",
        "PMT",
        "LED",
        "MPMT",
        "CAMERA",
        "TARGET",
        "SM",
        "WCD",
        "HALL",
    ]

    # Prefer to build the temporary import package next to the geometry files,
    # but fall back to a system temp directory if geometry_parent is read-only.
    build_parent = geometry_parent / "_Geometry_import"
    pkg = build_parent / "Geometry"
    try:
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").touch()
    except PermissionError:
        build_parent = Path(tempfile.mkdtemp(prefix="wcte_geometry_import_"))
        pkg = build_parent / "Geometry"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").touch()

    missing = []
    for name in module_names:
        exact = geometry_parent / f"{name}.py"
        if exact.exists():
            source = exact
        else:
            candidates = sorted(geometry_parent.glob(f"{name}*.py"))
            if not candidates:
                missing.append(name)
                continue
            source = candidates[0]
        shutil.copy2(source, pkg / f"{name}.py")

    if missing:
        raise FileNotFoundError(
            f"Missing geometry modules: {missing}. "
            "Point geometry_parent at the folder containing the WCTE geometry .py files."
        )

    build_path_str = str(build_parent)
    if build_path_str not in sys.path:
        sys.path.insert(0, build_path_str)
    return pkg


def _import_wcd_class(geometry_parent: Union[str, Path]):
    add_geometry_package_to_path(geometry_parent)
    from Geometry.WCD import WCD  # pylint: disable=import-outside-toplevel

    return WCD


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def prepare_secondary_df(
    df_sec: Union[pd.DataFrame, Dict[str, Any]],
    y_wcsim_origin_in_geom_mm: float = Y_WCSIM_ORIGIN_IN_GEOM_MM_DEFAULT,
) -> pd.DataFrame:
    """
    Prepare the track-step DataFrame.

    Adds x_mm/y_mm/z_mm and post_x_mm/post_y_mm/post_z_mm columns using the
    WCSim cm -> geometry mm conversion.
    """
    out = ensure_dataframe(df_sec, "df_sec")

    needed = [
        "evt",
        "trk",
        "parent",
        "pdg",
        "step",
        "x_cm",
        "y_cm",
        "z_cm",
        "post_x_cm",
        "post_y_cm",
        "post_z_cm",
        "t_ns",
        "post_t_ns",
        "particle",
        "creator",
        "step_process",
        "volume",
        "post_volume",
        "material",
        "post_material",
        "ke_MeV",
        "post_ke_MeV",
        "edep_MeV",
        "step_length_cm",
        "track_length_cm",
        "charge",
    ]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"df_sec is missing required columns: {missing}")

    for prefix in ["", "post_"]:
        out[f"{prefix}x_mm"] = 10.0 * out[f"{prefix}x_cm"].astype(float)
        out[f"{prefix}y_mm"] = (
            10.0 * out[f"{prefix}y_cm"].astype(float) + float(y_wcsim_origin_in_geom_mm)
        )
        out[f"{prefix}z_mm"] = 10.0 * out[f"{prefix}z_cm"].astype(float)

    for col in [
        "particle",
        "creator",
        "step_process",
        "volume",
        "post_volume",
        "material",
        "post_material",
    ]:
        out[col] = out[col].astype("category")

    return out


def prepare_photon_df(df_sec_ph: Optional[Union[pd.DataFrame, Dict[str, Any]]]) -> Optional[pd.DataFrame]:
    """
    Prepare the photon/PE DataFrame.

    Returns None if df_sec_ph is None, allowing track-only displays.
    """
    if df_sec_ph is None:
        return None

    out = ensure_dataframe(df_sec_ph, "df_sec_ph")

    needed = [
        "evt",
        "trk",  # photon track id
        "parent",  # original particle track id
        "parent_pdg",
        "parent_ke_MeV",
        "made_pe",
        "pe_pmt",  # PMT ID = 100*mPMT slot + PMT position ID
        "pe_time_ns",
    ]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"df_sec_ph is missing required columns: {missing}")

    for col in ["creator", "parent_creator", "end_process", "hit_volume", "end_boundary"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    return out


# -----------------------------------------------------------------------------
# Geometry cache
# -----------------------------------------------------------------------------

def build_wcte_geometry_cache(wcd: Any, place_info: str = "design") -> Dict[str, Any]:
    """
    Build mPMT/PMT geometry cache.

    PMT IDs are constructed as:

        PMT ID = 100*mPMT slot + PMT position ID
    """
    mpmt_polys = []
    mpmt_centers = []
    mpmt_hover = []
    pmt_centers = []
    pmt_hover = []
    pmt_ids = []
    pmt_id_to_loc = {}
    pmt_id_to_info = {}

    for slot, mpmt in enumerate(wcd.mpmts):
        mplace = mpmt.get_placement(place_info, device_for_coordinate_system=wcd)
        mloc = np.asarray(mplace["location"], dtype=float)

        mpmt_centers.append(mloc)
        mpmt_hover.append(
            f"mPMT slot {slot}<br>"
            f"kind={mpmt.kind}<br>"
            f"x={mloc[0]:.1f} mm<br>"
            f"y={mloc[1]:.1f} mm<br>"
            f"z={mloc[2]:.1f} mm"
        )

        try:
            poly = np.asarray(
                mpmt.get_xy_points(place_info, device_for_coordinate_system=wcd),
                dtype=float,
            )
            if len(poly) > 0:
                poly = np.vstack([poly, poly[0]])
                mpmt_polys.append((slot, mpmt.kind, poly))
        except Exception:
            pass

        for pmt in getattr(mpmt, "pmts", []):
            pplace = pmt.get_placement(place_info, device_for_coordinate_system=wcd)
            ploc = np.asarray(pplace["location"], dtype=float)

            pmt_pos_id = int(pmt.name)
            pmt_id = 100 * slot + pmt_pos_id

            pmt_ids.append(pmt_id)
            pmt_centers.append(ploc)
            pmt_hover.append(
                f"PMT {pmt_id}<br>"
                f"slot={slot}, pos={pmt_pos_id}<br>"
                f"x={ploc[0]:.1f} mm<br>"
                f"y={ploc[1]:.1f} mm<br>"
                f"z={ploc[2]:.1f} mm"
            )

            pmt_id_to_loc[pmt_id] = ploc
            pmt_id_to_info[pmt_id] = {
                "slot": slot,
                "pmt_pos_id": pmt_pos_id,
                "kind": mpmt.kind,
                "x": float(ploc[0]),
                "y": float(ploc[1]),
                "z": float(ploc[2]),
            }

    mpmt_centers = np.asarray(mpmt_centers, dtype=float)
    pmt_centers = np.asarray(pmt_centers, dtype=float)
    pmt_ids = np.asarray(pmt_ids, dtype=int)

    radius_xz = float(np.nanmax(np.sqrt(mpmt_centers[:, 0] ** 2 + mpmt_centers[:, 2] ** 2))) + 300.0
    y_min = float(np.nanmin(mpmt_centers[:, 1])) - 300.0
    y_max = float(np.nanmax(mpmt_centers[:, 1])) + 300.0

    return {
        "mpmt_polys": mpmt_polys,
        "mpmt_centers": mpmt_centers,
        "mpmt_hover": mpmt_hover,
        "pmt_centers": pmt_centers,
        "pmt_hover": pmt_hover,
        "pmt_ids": pmt_ids,
        "pmt_id_to_loc": pmt_id_to_loc,
        "pmt_id_to_info": pmt_id_to_info,
        "radius_xz": radius_xz,
        "y_min": y_min,
        "y_max": y_max,
    }


# -----------------------------------------------------------------------------
# Track utilities
# -----------------------------------------------------------------------------

def apply_event_display_cuts(
    ev: pd.DataFrame,
    time_cut_ns: Optional[float] = 20.0,
    time_cut_column: str = "t_ns",
    show_electrons: bool = True,
    hide_neutrinos: bool = True,
) -> pd.DataFrame:
    """Apply row-level cuts for the track-step display."""
    out = ev.copy()

    if time_cut_ns is not None:
        out = out[out[time_cut_column].astype(float) < float(time_cut_ns)]

    if not show_electrons:
        out = out[~out["pdg"].astype(int).isin(ELECTRON_PDGS)]

    if hide_neutrinos:
        out = out[~out["pdg"].astype(int).isin(NEUTRINO_PDGS)]

    return out


def _is_decay_daughter_summary(
    summary: pd.DataFrame,
    decay_daughter_pdgs: Optional[Iterable[int]] = DEFAULT_DECAY_DAUGHTER_PDGS,
) -> pd.Series:
    """Return a boolean mask for tracks that are decay daughters to be force-kept."""
    if summary.empty or decay_daughter_pdgs is None:
        return pd.Series(False, index=summary.index)

    pdg_set = set(int(p) for p in decay_daughter_pdgs)
    creator_is_decay = summary["creator"].astype(str).str.contains("Decay", case=False, na=False)
    pdg_is_requested = summary["pdg"].astype(int).isin(pdg_set)
    return creator_is_decay & pdg_is_requested


def _add_forced_decay_daughter_rows(
    ev_cut: pd.DataFrame,
    ev_all: pd.DataFrame,
    show_electrons: bool = True,
    hide_neutrinos: bool = True,
    decay_daughter_pdgs: Optional[Iterable[int]] = DEFAULT_DECAY_DAUGHTER_PDGS,
) -> pd.DataFrame:
    """
    Add full rows for selected decay daughters even if they failed the timing cut.

    This is useful for stopped-pion decay muons, which are often late and only
    about millimetres long. It does not override explicit electron/neutrino
    display choices.
    """
    if decay_daughter_pdgs is None or ev_all.empty:
        return ev_cut

    pdg_set = set(int(p) for p in decay_daughter_pdgs)
    decay_mask = (
        ev_all["creator"].astype(str).str.contains("Decay", case=False, na=False)
        & ev_all["pdg"].astype(int).isin(pdg_set)
    )

    decay_track_ids = set(ev_all.loc[decay_mask, "trk"].astype(int).unique())
    if not decay_track_ids:
        return ev_cut

    extra = ev_all[ev_all["trk"].astype(int).isin(decay_track_ids)].copy()

    if not show_electrons:
        extra = extra[~extra["pdg"].astype(int).isin(ELECTRON_PDGS)]

    if hide_neutrinos:
        extra = extra[~extra["pdg"].astype(int).isin(NEUTRINO_PDGS)]

    if extra.empty:
        return ev_cut

    out = pd.concat([ev_cut, extra], axis=0)
    out = out.loc[~out.index.duplicated(keep="first")]
    return out.sort_values(["trk", "step"])


def get_track_creation_info(ev_all: pd.DataFrame) -> pd.DataFrame:
    """Per-track information from the first step in the full event."""
    first_rows = ev_all.sort_values(["trk", "step"]).groupby("trk", observed=True).first().reset_index()

    creation_info = first_rows[["trk", "ke_MeV", "t_ns", "x_mm", "y_mm", "z_mm"]].copy()
    creation_info = creation_info.rename(
        columns={
            "ke_MeV": "creation_ke_MeV",
            "t_ns": "creation_t_ns",
            "x_mm": "creation_x_mm",
            "y_mm": "creation_y_mm",
            "z_mm": "creation_z_mm",
        }
    )
    return creation_info


def _particle_label_from_pdg(pdg: int) -> str:
    """Small PDG label helper for hover text."""
    labels = {
        22: "gamma",
        11: "e-",
        -11: "e+",
        13: "mu-",
        -13: "mu+",
        111: "pi0",
        211: "pi+",
        -211: "pi-",
        321: "K+",
        -321: "K-",
        130: "K0L",
        310: "K0S",
        2212: "proton",
        -2212: "anti-proton",
        2112: "neutron",
        -2112: "anti-neutron",
    }
    return labels.get(int(pdg), str(int(pdg)))


def _classify_raw_geant4_process(raw_process: Any, pdg: int, terminal_ke_MeV: float) -> Tuple[str, str, bool]:
    """
    Map an explicit Geant4 process name onto a physical-ish label.

    Returns
    -------
    physical_process, detail, is_inferred
    """
    raw = "" if pd.isna(raw_process) else str(raw_process)
    raw_lower = raw.lower()
    pdg = int(pdg)

    # Direct physical terminal processes.
    if raw == "Decay" or raw_lower.endswith("decay"):
        return "Decay", f"raw Geant4 terminal process was {raw}", False

    if raw == "RadioactiveDecay":
        return "Radioactive decay", f"raw Geant4 terminal process was {raw}", False

    if "inelastic" in raw_lower:
        return "Hadronic inelastic interaction", f"raw Geant4 terminal process was {raw}", False

    if raw == "nCapture" or "capture" in raw_lower:
        return "Capture", f"raw Geant4 terminal process was {raw}", False

    if raw == "annihil" or "annihil" in raw_lower:
        return "Annihilation", f"raw Geant4 terminal process was {raw}", False

    if raw == "conv":
        return "Gamma conversion", "raw Geant4 terminal process was conv", False

    if raw == "phot":
        return "Photoelectric absorption", "raw Geant4 terminal process was phot", False

    if raw == "compt":
        return "Compton scatter / final gamma scatter", "raw Geant4 terminal process was compt", False

    # Elastic and EM processes are real processes, but if they appear as the
    # final row of the saved table they often mean "last saved step was limited
    # by this process", not necessarily that the particle was physically killed.
    if raw == "hadElastic":
        return "Hadronic elastic scatter / last saved step", "raw Geant4 terminal process was hadElastic", True

    if raw in {"eIoni", "muIoni", "hIoni", "ionIoni"}:
        if np.isfinite(terminal_ke_MeV) and terminal_ke_MeV <= DEFAULT_PHYSICAL_REST_KE_THRESHOLD_MEV:
            return "Stopped / ranged out by ionization", f"raw final step was {raw}; final KE is near zero", True
        return "Ionization-limited last saved step", f"raw Geant4 terminal process was {raw}", True

    if raw in {"msc", "CoulombScat"}:
        return "Scattering-limited last saved step", f"raw Geant4 terminal process was {raw}", True

    if raw == "Transportation":
        return "Geometry boundary / transportation", "raw final step was Transportation", True

    if raw in OPTICAL_BOOKKEEPING_PROCESSES:
        return "Optical-bookkeeping final step", f"raw final step was {raw}", True

    if raw == "" or raw_lower == "none" or raw_lower == "nan":
        return "Unknown terminal process", "no raw terminal process was available", True

    return raw, f"unmapped raw Geant4 terminal process was {raw}", True


def _infer_physical_termination_for_track(
    first_daughters: pd.DataFrame,
    terminal_row: pd.Series,
    physical_rest_ke_threshold_MeV: float = DEFAULT_PHYSICAL_REST_KE_THRESHOLD_MEV,
) -> Dict[str, Any]:
    """
    Infer the physical termination process for one track.

    Important:
        A Geant4 daughter/child track is not automatically a terminal-process
        product. For example, a primary pi+ may have many e- children with
        creator == hIoni. Those are delta rays from ionization, not decay
        daughters. Therefore, when the terminal process is classified as Decay,
        the detail string only lists children with creator containing Decay.
        The same is true for inelastic/capture/annihilation labels.
    """
    pdg = int(terminal_row["pdg"])
    raw = "" if pd.isna(terminal_row["step_process"]) else str(terminal_row["step_process"])
    terminal_ke = float(terminal_row["post_ke_MeV"]) if pd.notna(terminal_row["post_ke_MeV"]) else np.nan

    if first_daughters is None:
        first_daughters = pd.DataFrame()

    def _daughter_summary_for(mask: pd.Series, label: str) -> str:
        """
        Summarize only the daughter rows selected by mask.
        """
        if first_daughters.empty:
            return f"no explicit {label} daughter tracks found"

        selected = first_daughters.loc[mask].copy()
        if selected.empty:
            return f"no explicit {label} daughter tracks found"

        pieces = []
        for d in selected.itertuples(index=False):
            particle_name = getattr(d, "particle", _particle_label_from_pdg(int(d.pdg)))
            pieces.append(
                f"trk {int(d.trk)}: {particle_name} "
                f"(PDG {int(d.pdg)}), creator={d.creator}"
            )

        return f"explicit {label} daughter/product track(s): " + "; ".join(pieces)

    if not first_daughters.empty:
        creator_str = first_daughters["creator"].astype(str)

        decay_mask = creator_str.str.contains("Decay", case=False, na=False)
        inelastic_mask = creator_str.str.contains("Inelastic", case=False, na=False)
        capture_mask = creator_str.str.contains("Capture", case=False, na=False)
        annihil_mask = creator_str.str.contains("annihil", case=False, na=False)

        if decay_mask.any():
            return {
                "physical_terminal_process": "Decay",
                "physical_terminal_detail": _daughter_summary_for(decay_mask, "decay"),
                "terminal_is_inferred": raw != "Decay",
                "terminal_evidence": "daughter_creator_decay",
            }

        if inelastic_mask.any():
            return {
                "physical_terminal_process": "Hadronic inelastic interaction",
                "physical_terminal_detail": _daughter_summary_for(inelastic_mask, "inelastic"),
                "terminal_is_inferred": "inelastic" not in raw.lower(),
                "terminal_evidence": "daughter_creator_inelastic",
            }

        if capture_mask.any():
            return {
                "physical_terminal_process": "Capture",
                "physical_terminal_detail": _daughter_summary_for(capture_mask, "capture"),
                "terminal_is_inferred": "capture" not in raw.lower(),
                "terminal_evidence": "daughter_creator_capture",
            }

        if annihil_mask.any():
            return {
                "physical_terminal_process": "Annihilation",
                "physical_terminal_detail": _daughter_summary_for(annihil_mask, "annihilation"),
                "terminal_is_inferred": "annihil" not in raw.lower(),
                "terminal_evidence": "daughter_creator_annihilation",
            }

    raw_label, raw_detail, raw_inferred = _classify_raw_geant4_process(raw, pdg, terminal_ke)

    clear_raw = {
        "Decay",
        "Radioactive decay",
        "Hadronic inelastic interaction",
        "Capture",
        "Annihilation",
        "Gamma conversion",
        "Photoelectric absorption",
    }

    if raw_label in clear_raw:
        return {
            "physical_terminal_process": raw_label,
            "physical_terminal_detail": raw_detail,
            "terminal_is_inferred": raw_inferred,
            "terminal_evidence": "raw_terminal_process",
        }

    stopped = np.isfinite(terminal_ke) and terminal_ke <= float(physical_rest_ke_threshold_MeV)
    raw_is_bookkeeping = raw in OPTICAL_BOOKKEEPING_PROCESSES or raw in {
        "eIoni", "muIoni", "hIoni", "ionIoni", "msc", "CoulombScat", "Transportation"
    }

    if stopped:
        if pdg == 211:
            return {
                "physical_terminal_process": "Decay at rest",
                "physical_terminal_detail": (
                    f"inferred for stopped pi+ with no explicit Decay daughter and no inelastic daughter; "
                    f"raw final Geant4 step was {raw}; terminal KE={terminal_ke:.3g} MeV"
                ),
                "terminal_is_inferred": True,
                "terminal_evidence": "stopped_pi_plus_no_explicit_terminal_daughter",
            }

        if pdg in STOPPED_NEGATIVE_CAPTURE_PDGS:
            return {
                "physical_terminal_process": "Capture / absorption at rest",
                "physical_terminal_detail": (
                    f"inferred for stopped negative meson in material; raw final Geant4 step was {raw}; "
                    f"terminal KE={terminal_ke:.3g} MeV"
                ),
                "terminal_is_inferred": True,
                "terminal_evidence": "stopped_negative_meson",
            }

        if pdg in STABLE_RANGEOUT_PDGS or abs(pdg) in {2212}:
            return {
                "physical_terminal_process": "Stopped / ranged out",
                "physical_terminal_detail": (
                    f"inferred stable charged-particle range-out; raw final Geant4 step was {raw}; "
                    f"terminal KE={terminal_ke:.3g} MeV"
                ),
                "terminal_is_inferred": True,
                "terminal_evidence": "stopped_stable_particle",
            }

        if pdg == -11:
            return {
                "physical_terminal_process": "Annihilation at rest",
                "physical_terminal_detail": (
                    f"inferred stopped positron; raw final Geant4 step was {raw}; "
                    f"terminal KE={terminal_ke:.3g} MeV"
                ),
                "terminal_is_inferred": True,
                "terminal_evidence": "stopped_positron",
            }

        if abs(pdg) == 11:
            return {
                "physical_terminal_process": "Stopped / ranged out by EM energy loss",
                "physical_terminal_detail": (
                    f"inferred stopped electron; raw final Geant4 step was {raw}; "
                    f"terminal KE={terminal_ke:.3g} MeV"
                ),
                "terminal_is_inferred": True,
                "terminal_evidence": "stopped_electron",
            }

        if pdg in STOPPED_POSITIVE_DECAY_PDGS:
            return {
                "physical_terminal_process": "Stopped; decay expected if tracked long enough",
                "physical_terminal_detail": (
                    f"inferred stopped unstable positive particle; raw final Geant4 step was {raw}; "
                    f"terminal KE={terminal_ke:.3g} MeV; no explicit Decay daughter found"
                ),
                "terminal_is_inferred": True,
                "terminal_evidence": "stopped_unstable_particle_no_explicit_decay_daughter",
            }

        return {
            "physical_terminal_process": "Stopped / ranged out",
            "physical_terminal_detail": (
                f"inferred near-zero final KE; raw final Geant4 step was {raw}; terminal KE={terminal_ke:.3g} MeV"
            ),
            "terminal_is_inferred": True,
            "terminal_evidence": "stopped_unknown_particle",
        }

    if raw_is_bookkeeping:
        return {
            "physical_terminal_process": "Last saved step only",
            "physical_terminal_detail": (
                f"raw final Geant4 step was {raw}, but terminal KE={terminal_ke:.3g} MeV is not near zero "
                "and no explicit Decay/Inelastic/Capture/Annihilation daughter evidence was found"
            ),
            "terminal_is_inferred": True,
            "terminal_evidence": "ambiguous_bookkeeping_process",
        }

    return {
        "physical_terminal_process": raw_label,
        "physical_terminal_detail": raw_detail,
        "terminal_is_inferred": raw_inferred,
        "terminal_evidence": "raw_or_unmapped_process",
    }

def get_track_terminal_info(
    ev_all: pd.DataFrame,
    physical_rest_ke_threshold_MeV: float = DEFAULT_PHYSICAL_REST_KE_THRESHOLD_MEV,
) -> pd.DataFrame:
    """
    Per-track terminal information from the final recorded step in the full event,
    plus an inferred physical termination process.

    `terminal_process` is the raw final step_process from the TTree.
    `physical_terminal_process` is the label intended for physics interpretation
    and plotting.
    """
    ev_sorted = ev_all.sort_values(["trk", "step"])
    last_rows = ev_sorted.groupby("trk", observed=True).last().reset_index()

    # First row of each track, then group those first rows by parent. This gives
    # daughter evidence without scanning the whole event once per track.
    first_rows = ev_sorted.groupby("trk", observed=True).first().reset_index()
    first_daughters_by_parent = {
        int(parent): group.copy()
        for parent, group in first_rows.groupby("parent", observed=True)
    }

    cols = [
        "trk",
        "pdg",
        "particle",
        "step_process",
        "post_t_ns",
        "post_ke_MeV",
        "post_x_mm",
        "post_y_mm",
        "post_z_mm",
        "post_volume",
        "post_material",
        "step",
    ]
    terminal_info = last_rows[cols].copy()

    inferred_rows = []
    for row in terminal_info.itertuples(index=False):
        track_id = int(row.trk)
        terminal_row = terminal_info.loc[terminal_info["trk"].astype(int) == track_id].iloc[0]
        first_daughters = first_daughters_by_parent.get(track_id, pd.DataFrame())
        inferred_rows.append(
            {
                "trk": track_id,
                **_infer_physical_termination_for_track(
                    first_daughters,
                    terminal_row,
                    physical_rest_ke_threshold_MeV=physical_rest_ke_threshold_MeV,
                ),
            }
        )

    inferred = pd.DataFrame(inferred_rows)

    terminal_info = terminal_info.rename(
        columns={
            "step_process": "terminal_process",
            "post_t_ns": "terminal_t_ns",
            "post_ke_MeV": "terminal_ke_MeV",
            "post_x_mm": "terminal_x_mm",
            "post_y_mm": "terminal_y_mm",
            "post_z_mm": "terminal_z_mm",
            "post_volume": "terminal_volume",
            "post_material": "terminal_material",
            "step": "terminal_step",
        }
    )

    terminal_info = terminal_info.drop(columns=["pdg", "particle"])
    terminal_info = terminal_info.merge(inferred, on="trk", how="left")
    return terminal_info

def event_track_summary(
    tracks_df: pd.DataFrame,
    event_id: int,
    time_cut_ns: Optional[float] = 20.0,
    time_cut_column: str = "t_ns",
    show_electrons: bool = True,
    hide_neutrinos: bool = True,
    always_show_decay_daughters: bool = False,
    decay_daughter_pdgs: Optional[Iterable[int]] = DEFAULT_DECAY_DAUGHTER_PDGS,
    decay_daughters_ignore_time_cut: bool = True,
    physical_rest_ke_threshold_MeV: float = DEFAULT_PHYSICAL_REST_KE_THRESHOLD_MEV,
) -> pd.DataFrame:
    """Summarize tracks after row-level display cuts; retain original creation KE."""
    ev_all = tracks_df.loc[tracks_df["evt"] == event_id].sort_values(["trk", "step"])
    if ev_all.empty:
        raise ValueError(f"No track rows found for evt={event_id}")

    creation_info = get_track_creation_info(ev_all)
    terminal_info = get_track_terminal_info(
        ev_all,
        physical_rest_ke_threshold_MeV=physical_rest_ke_threshold_MeV,
    )

    ev = apply_event_display_cuts(
        ev_all,
        time_cut_ns=time_cut_ns,
        time_cut_column=time_cut_column,
        show_electrons=show_electrons,
        hide_neutrinos=hide_neutrinos,
    )

    if always_show_decay_daughters and decay_daughters_ignore_time_cut:
        ev = _add_forced_decay_daughter_rows(
            ev,
            ev_all,
            show_electrons=show_electrons,
            hide_neutrinos=hide_neutrinos,
            decay_daughter_pdgs=decay_daughter_pdgs,
        )

    empty_cols = [
        "trk",
        "parent",
        "pdg",
        "particle",
        "creator",
        "charge",
        "creation_ke_MeV",
        "creation_t_ns",
        "creation_x_mm",
        "creation_y_mm",
        "creation_z_mm",
        "terminal_process",
        "terminal_t_ns",
        "terminal_ke_MeV",
        "terminal_x_mm",
        "terminal_y_mm",
        "terminal_z_mm",
        "terminal_volume",
        "terminal_material",
        "terminal_step",
        "physical_terminal_process",
        "physical_terminal_detail",
        "terminal_is_inferred",
        "terminal_evidence",
        "display_t0_ns",
        "display_t1_ns",
        "display_ke0_MeV",
        "display_ke1_MeV",
        "displayed_length_mm",
        "full_track_length_mm",
        "n_steps",
        "first_volume",
        "last_volume",
        "last_process",
        "is_primary",
    ]
    if ev.empty:
        return pd.DataFrame(columns=empty_cols)

    summary = ev.groupby("trk", observed=True).agg(
        parent=("parent", "first"),
        pdg=("pdg", "first"),
        particle=("particle", "first"),
        creator=("creator", "first"),
        charge=("charge", "first"),
        display_t0_ns=("t_ns", "first"),
        display_t1_ns=("post_t_ns", "last"),
        display_ke0_MeV=("ke_MeV", "first"),
        display_ke1_MeV=("post_ke_MeV", "last"),
        displayed_length_mm=("step_length_cm", lambda s: 10.0 * float(np.nansum(s))),
        full_track_length_mm=("track_length_cm", lambda s: 10.0 * float(np.nanmax(s))),
        n_steps=("step", "size"),
        first_volume=("volume", "first"),
        last_volume=("post_volume", "last"),
        last_process=("step_process", "last"),
    ).reset_index()

    summary = summary.merge(creation_info, on="trk", how="left")
    summary = summary.merge(terminal_info, on="trk", how="left")
    summary["is_primary"] = summary["parent"].astype(int) == 0
    summary["is_forced_decay_daughter"] = _is_decay_daughter_summary(summary, decay_daughter_pdgs)

    return summary.sort_values(
        ["is_primary", "is_forced_decay_daughter", "creation_ke_MeV", "displayed_length_mm"],
        ascending=[False, False, False, False],
    )


def select_tracks(
    summary: pd.DataFrame,
    min_track_length_mm: Optional[float] = 0.0,
    apply_min_track_length_cut: bool = True,
    min_creation_ke_MeV: Optional[float] = 100.0,
    max_tracks: Optional[int] = 200,
    charged_only: bool = False,
    include_pdgs: Optional[Iterable[int]] = None,
    exclude_pdgs: Optional[Iterable[int]] = None,
    always_show_decay_daughters: bool = False,
    decay_daughter_pdgs: Optional[Iterable[int]] = DEFAULT_DECAY_DAUGHTER_PDGS,
) -> pd.DataFrame:
    """
    Track-level selection for what gets drawn.

    If apply_min_track_length_cut=False, the displayed-length cut is disabled
    for all particles. This is useful for seeing short low-range daughters
    or all processes associated with particles that pass the creation-KE cut.

    You can also disable the length cut by passing min_track_length_mm=None.

    If always_show_decay_daughters=True, selected decay daughters, by default
    mu+/mu-, bypass the normal creation-KE and displayed-track-length cuts.
    This prevents low-energy stopped-pion decay muons from being silently
    removed by cuts intended for Cherenkov-bright tracks.

    Explicit include_pdgs/exclude_pdgs are still respected.
    """
    out = summary.copy()
    if len(out) == 0:
        return out

    if include_pdgs is not None:
        out = out[out["pdg"].astype(int).isin(set(include_pdgs))]

    if exclude_pdgs is not None:
        out = out[~out["pdg"].astype(int).isin(set(exclude_pdgs))]

    if len(out) == 0:
        return out

    force_keep = _is_decay_daughter_summary(out, decay_daughter_pdgs) if always_show_decay_daughters else pd.Series(False, index=out.index)

    if apply_min_track_length_cut and min_track_length_mm is not None:
        normal_keep = out["displayed_length_mm"].astype(float) >= float(min_track_length_mm)
    else:
        normal_keep = pd.Series(True, index=out.index)

    if min_creation_ke_MeV is not None:
        normal_keep &= out["creation_ke_MeV"].astype(float) >= float(min_creation_ke_MeV)

    if charged_only:
        charged = out["charge"].astype(float) != 0.0
        normal_keep &= charged
        force_keep &= charged

    keep_mask = normal_keep | force_keep
    forced_mask = force_keep & ~normal_keep

    out = out[keep_mask].copy()
    if len(out) == 0:
        return out

    out["is_decay_daughter"] = _is_decay_daughter_summary(out, decay_daughter_pdgs)
    out["is_forced_decay_daughter"] = forced_mask.reindex(out.index).fillna(False).astype(bool)
    out = out.sort_values(
        ["is_primary", "is_forced_decay_daughter", "creation_ke_MeV", "displayed_length_mm"],
        ascending=[False, False, False, False],
    )

    if max_tracks is not None:
        out = out.head(int(max_tracks))

    return out

def _track_points_mm(g: pd.DataFrame) -> np.ndarray:
    """Build displayed polyline from surviving step rows."""
    g = g.sort_values("step")
    pre = g[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
    last_post = g[["post_x_mm", "post_y_mm", "post_z_mm"]].iloc[-1].to_numpy(dtype=float)
    return np.vstack([pre, last_post])


def _make_track_hover(row: Any, time_cut_ns: Optional[float], min_creation_ke_MeV: Optional[float]) -> str:
    """
    Compact hover label for a displayed particle track.

    The terminal-detail string can be long because it may list many daughters.
    It is therefore wrapped/truncated with _wrap_hover_text().
    """
    time_cut_text = "none" if time_cut_ns is None else f"t_ns < {time_cut_ns:g} ns"
    ke_cut_text = "none" if min_creation_ke_MeV is None else f"creation KE ≥ {min_creation_ke_MeV:g} MeV"

    physical_process = _short_hover_value(getattr(row, "physical_terminal_process", "unknown"), 80)
    raw_process = _short_hover_value(getattr(row, "terminal_process", "unknown"), 80)
    detail = _wrap_hover_text(getattr(row, "physical_terminal_detail", ""), width=62, max_chars=520)

    return (
        f"<b>track {int(row.trk)}: {_hover_safe_text(row.particle)}</b><br>"
        f"PDG={int(row.pdg)}, parent={int(row.parent)}, creator={_short_hover_value(row.creator, 50)}<br>"
        f"charge={row.charge:g}<br>"
        f"creation KE={row.creation_ke_MeV:.3g} MeV, creation t={row.creation_t_ns:.3g} ns<br>"
        f"displayed KE={row.display_ke0_MeV:.3g} → {row.display_ke1_MeV:.3g} MeV<br>"
        f"displayed length={row.displayed_length_mm:.1f} mm; steps={int(row.n_steps)}<br>"
        f"displayed t={row.display_t0_ns:.3g} → {row.display_t1_ns:.3g} ns<br>"
        f"time cut: {time_cut_text}; KE cut: {ke_cut_text}<br>"
        f"<b>physical termination: {physical_process}</b><br>"
        f"raw final Geant4 process: {raw_process}<br>"
        f"terminal KE={row.terminal_ke_MeV:.3g} MeV, terminal t={row.terminal_t_ns:.3g} ns<br>"
        f"terminal volume={_short_hover_value(row.terminal_volume, 70)}<br>"
        f"termination detail:<br>{detail}"
    )


# -----------------------------------------------------------------------------
# PMT / PE utilities
# -----------------------------------------------------------------------------

def get_event_pe_summary(
    photons_df: Optional[pd.DataFrame],
    event_id: int,
    shown_track_ids: Iterable[int],
    geom: Dict[str, Any],
    pe_time_cut_ns: Union[str, float, None] = "same_as_track",
    track_time_cut_ns: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build a per-event summary of PE contributions to each PMT.

    Only photons whose parent track is in shown_track_ids are retained.

    pe_time_cut_ns:
      - "same_as_track" -> use track_time_cut_ns
      - float           -> require pe_time_ns < that value
      - None            -> no PE time cut
    """
    empty = pd.DataFrame(columns=["pe_pmt", "origin_trk", "n_pe"])
    if photons_df is None:
        return empty

    if pe_time_cut_ns == "same_as_track":
        effective_pe_time_cut_ns = track_time_cut_ns
    else:
        effective_pe_time_cut_ns = pe_time_cut_ns

    ph = photons_df.loc[photons_df["evt"] == event_id].copy()
    if ph.empty:
        return empty

    ph = ph[(ph["made_pe"].astype(int) != 0) & (ph["pe_pmt"].astype(int) >= 0)].copy()

    if effective_pe_time_cut_ns is not None:
        ph = ph[ph["pe_time_ns"].astype(float) < float(effective_pe_time_cut_ns)]

    shown_track_ids = set(int(t) for t in shown_track_ids)
    ph = ph[ph["parent"].astype(int).isin(shown_track_ids)].copy()

    if ph.empty:
        return empty

    valid_pmts = set(int(pid) for pid in geom["pmt_id_to_loc"].keys())
    ph = ph[ph["pe_pmt"].astype(int).isin(valid_pmts)].copy()

    if ph.empty:
        return empty

    return (
        ph.groupby(["pe_pmt", "parent"], observed=True)
        .size()
        .reset_index(name="n_pe")
        .rename(columns={"parent": "origin_trk"})
    )


def build_pmt_display_dataframe(
    pe_summary: pd.DataFrame,
    chosen_summary: pd.DataFrame,
    geom: Dict[str, Any],
) -> pd.DataFrame:
    """Attach track labels and PMT coordinates to a PE summary."""
    empty_cols = [
        "pe_pmt",
        "origin_trk",
        "n_pe",
        "particle",
        "pdg",
        "creation_ke_MeV",
        "x_mm",
        "y_mm",
        "z_mm",
    ]
    if pe_summary.empty:
        return pd.DataFrame(columns=empty_cols)

    chosen_small = chosen_summary[["trk", "particle", "pdg", "creation_ke_MeV"]].copy()
    chosen_small = chosen_small.rename(columns={"trk": "origin_trk"})

    out = pe_summary.merge(chosen_small, on="origin_trk", how="left")
    out["x_mm"] = out["pe_pmt"].map(lambda pid: geom["pmt_id_to_loc"][int(pid)][0])
    out["y_mm"] = out["pe_pmt"].map(lambda pid: geom["pmt_id_to_loc"][int(pid)][1])
    out["z_mm"] = out["pe_pmt"].map(lambda pid: geom["pmt_id_to_loc"][int(pid)][2])
    return out


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def add_detector_context(
    fig: go.Figure,
    geom: Dict[str, Any],
    show_shell: bool = True,
    show_mpmt_outlines: bool = True,
    show_mpmt_centers: bool = False,
    show_mpmt_slot_numbers: bool = False,
) -> None:
    """Add detector shell, mPMT outlines, and optional slot labels."""
    if show_shell:
        radius = geom["radius_xz"]
        y0 = geom["y_min"]
        y1 = geom["y_max"]
        theta = np.linspace(0, 2 * np.pi, 121)

        for y in [y0, y1]:
            fig.add_trace(
                go.Scatter3d(
                    x=radius * np.cos(theta),
                    y=np.full_like(theta, y),
                    z=radius * np.sin(theta),
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.35)", width=2),
                    name="detector shell",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        for t in np.linspace(0, 2 * np.pi, 17)[:-1]:
            fig.add_trace(
                go.Scatter3d(
                    x=[radius * np.cos(t), radius * np.cos(t)],
                    y=[y0, y1],
                    z=[radius * np.sin(t), radius * np.sin(t)],
                    mode="lines",
                    line=dict(color="rgba(120,120,120,0.18)", width=1),
                    name="detector shell",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if show_mpmt_outlines:
        for slot, kind, poly in geom["mpmt_polys"]:
            fig.add_trace(
                go.Scatter3d(
                    x=poly[:, 0],
                    y=poly[:, 1],
                    z=poly[:, 2],
                    mode="lines",
                    line=dict(color="rgba(80,80,80,0.40)", width=2),
                    name="mPMT outlines",
                    hovertemplate=f"mPMT slot {slot}<br>kind={kind}<extra></extra>",
                    showlegend=False,
                )
            )

    if (show_mpmt_centers or show_mpmt_slot_numbers) and len(geom["mpmt_centers"]):
        m = geom["mpmt_centers"]
        mode = "markers+text" if show_mpmt_slot_numbers else "markers"
        text = [str(i) for i in range(len(m))] if show_mpmt_slot_numbers else None

        fig.add_trace(
            go.Scatter3d(
                x=m[:, 0],
                y=m[:, 1],
                z=m[:, 2],
                mode=mode,
                marker=dict(size=5, color="rgba(0,0,0,0.70)"),
                text=text,
                textposition="middle center",
                textfont=dict(size=10),
                hovertext=geom["mpmt_hover"],
                hovertemplate="%{hovertext}<extra></extra>",
                name="mPMT slots",
            )
        )


def add_pmt_pe_markers(
    fig: go.Figure,
    pmt_df: pd.DataFrame,
    geom: Dict[str, Any],
    track_color_map: Dict[int, str],
    pmt_display_mode: str = "all",
    max_pmt_rings: Optional[int] = None,
    unhit_pmt_size: int = 4,
    outer_hit_size: int = 15,
    hit_size_step: int = 4,
    link_pmt_visibility_to_tracks: bool = True,
) -> None:
    """
    Draw PMTs as markers.

    Important legend behavior
    -------------------------
    The PMT display is split into two layers:

      1. A grey base layer.
         - If pmt_display_mode == "all", this contains every detector PMT.
         - If pmt_display_mode == "hit_only", this contains PMTs that received
           at least one PE from any currently shown particle track.

      2. Coloured PE overlays, split by origin track and ring index.
         These traces have the same legendgroup as their origin particle track:
             legendgroup = f"trk{origin_trk}"

    Therefore, when you double-click a particle track in the legend, Plotly
    isolates that track's legendgroup. The coloured PMT overlays from the other
    tracks disappear, while the grey base layer remains. Visually, this makes
    PMTs not hit by the selected particle look like non-hit PMTs.
    """
    if pmt_display_mode not in {"all", "hit_only"}:
        raise ValueError("pmt_display_mode must be 'all' or 'hit_only'")

    hit_pmts = set(int(pid) for pid in pmt_df["pe_pmt"].unique()) if len(pmt_df) else set()

    # ------------------------------------------------------------------
    # Grey base layer
    # ------------------------------------------------------------------
    if pmt_display_mode == "all":
        base_ids = np.asarray(geom["pmt_ids"], dtype=int)
        base_pts = np.asarray(geom["pmt_centers"], dtype=float)
        base_hover_status = "grey PMT base layer"
    else:
        # In hit_only mode, the grey base contains only PMTs that were hit by
        # at least one shown particle. When a track is isolated, hit PMTs from
        # other particles remain visible but grey.
        base_ids = np.asarray(sorted(hit_pmts), dtype=int)
        if len(base_ids):
            base_pts = np.asarray([geom["pmt_id_to_loc"][int(pid)] for pid in base_ids], dtype=float)
        else:
            base_pts = np.empty((0, 3), dtype=float)
        base_hover_status = "grey PMT base layer: PE from some shown particle"

    if len(base_ids):
        hover = []
        for pid in base_ids:
            info = geom["pmt_id_to_info"][int(pid)]
            hover.append(
                f"PMT {int(pid)}<br>"
                f"slot={info['slot']}, pos={info['pmt_pos_id']}<br>"
                f"{base_hover_status}"
            )

        # This trace intentionally has no legendgroup and showlegend=False.
        # Plotly's double-click legend isolation leaves it visible, so it acts
        # as the non-hit PMT colour underneath the track-specific coloured
        # overlays.
        fig.add_trace(
            go.Scatter3d(
                x=base_pts[:, 0],
                y=base_pts[:, 1],
                z=base_pts[:, 2],
                mode="markers",
                marker=dict(size=unhit_pmt_size, color="rgba(150,150,150,0.55)"),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                name="PMT grey base",
                showlegend=False,
            )
        )

    if pmt_df.empty:
        return

    # ------------------------------------------------------------------
    # Coloured overlays grouped by origin track
    # ------------------------------------------------------------------
    grouped = pmt_df.groupby("pe_pmt", observed=True)

    # ring_traces[(origin_trk, ring_index)] = trace data
    ring_traces: Dict[Tuple[int, int], Dict[str, list]] = {}

    for pe_pmt, full_g in grouped:
        full_g = full_g.sort_values(["n_pe", "origin_trk"], ascending=[False, True]).copy()
        g = full_g.copy()

        if max_pmt_rings is not None:
            g = g.head(int(max_pmt_rings)).copy()

        x = float(full_g["x_mm"].iloc[0])
        y = float(full_g["y_mm"].iloc[0])
        z = float(full_g["z_mm"].iloc[0])

        info = geom["pmt_id_to_info"][int(pe_pmt)]

        contributor_lines = []
        for _, row in full_g.iterrows():
            contributor_lines.append(
                f"track {int(row.origin_trk)}: {_short_hover_value(row.particle, 30)}, "
                f"KE₀={row.creation_ke_MeV:.1f} MeV, "
                f"PEs={int(row.n_pe)}"
            )

        if len(contributor_lines) > 10:
            contributor_lines = contributor_lines[:10] + [f"... {len(full_g) - 10} more contributor(s)"]

        hover_text = (
            f"PMT {int(pe_pmt)}<br>"
            f"slot={info['slot']}, pos={info['pmt_pos_id']}<br>"
            f"contributors={len(full_g)}<br>"
            + "<br>".join(contributor_lines)
        )

        for ring_index, (_, row) in enumerate(g.iterrows()):
            origin_trk = int(row.origin_trk)
            size = max(outer_hit_size - ring_index * hit_size_step, 3)
            color = track_color_map.get(origin_trk, "black")

            key = (origin_trk, ring_index)
            if key not in ring_traces:
                ring_traces[key] = {
                    "x": [],
                    "y": [],
                    "z": [],
                    "size": [],
                    "color": [],
                    "text": [],
                    "origin_trk": origin_trk,
                    "ring_index": ring_index,
                }

            ring_traces[key]["x"].append(x)
            ring_traces[key]["y"].append(y)
            ring_traces[key]["z"].append(z)
            ring_traces[key]["size"].append(size)
            ring_traces[key]["color"].append(color)
            ring_traces[key]["text"].append(hover_text)

    # Add coloured PMT overlays grouped with their parent particle-track legendgroup.
    # This is the key part that makes legend isolation affect PMT colouring.
    for key in sorted(ring_traces.keys(), key=lambda k: (k[1], k[0])):
        d = ring_traces[key]
        origin_trk = int(d["origin_trk"])
        ring_index = int(d["ring_index"])

        fig.add_trace(
            go.Scatter3d(
                x=d["x"],
                y=d["y"],
                z=d["z"],
                mode="markers",
                marker=dict(size=d["size"], color=d["color"], opacity=0.98),
                text=d["text"],
                hovertemplate="%{text}<extra></extra>",
                name=f"PMT PE overlay trk {origin_trk} ring {ring_index}",
                legendgroup=f"trk{origin_trk}" if link_pmt_visibility_to_tracks else None,
                showlegend=False,
            )
        )



def _format_terminal_process_name(value: Any) -> str:
    """Return a readable terminal-process name."""
    if pd.isna(value):
        return "unknown"
    value = str(value)
    return value if value else "unknown"


def _hover_safe_text(value: Any) -> str:
    """
    Convert arbitrary text to safe, compact HTML for Plotly hover labels.

    The important fix is that long semicolon-separated details are wrapped into
    multiple <br> lines instead of becoming one huge horizontal hover box.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    s = str(value)
    s = s.replace("\n", " ")
    s = " ".join(s.split())

    return html.escape(s, quote=False)


def _wrap_hover_text(
    value: Any,
    width: int = 64,
    max_chars: int = 700,
    max_semicolon_items: int = 8,
) -> str:
    """
    Return short HTML text suitable for a Plotly hover label.

    - splits long semicolon-separated lists, such as daughter summaries
    - wraps each line to a reasonable width
    - truncates very long details
    """
    s = _hover_safe_text(value)

    if not s:
        return ""

    was_truncated = False
    if len(s) > int(max_chars):
        s = s[: int(max_chars)].rstrip()
        was_truncated = True

    # First split semicolon-separated lists, since those are what made the
    # previous hover boxes enormous.
    parts = [p.strip() for p in s.split(";") if p.strip()]

    if len(parts) > int(max_semicolon_items):
        hidden = len(parts) - int(max_semicolon_items)
        parts = parts[: int(max_semicolon_items)]
        parts.append(f"... {hidden} more item(s)")
        was_truncated = True

    wrapped_lines = []
    for part in parts:
        for line in textwrap.wrap(
            part,
            width=int(width),
            break_long_words=False,
            break_on_hyphens=False,
        ):
            wrapped_lines.append(line)

    if was_truncated and (not wrapped_lines or "truncated" not in wrapped_lines[-1].lower()):
        wrapped_lines.append("... truncated")

    return "<br>".join(wrapped_lines)


def _short_hover_value(value: Any, max_chars: int = 80) -> str:
    """Compact single-line value for hover text."""
    s = _hover_safe_text(value)
    if len(s) > max_chars:
        return s[: max_chars - 3].rstrip() + "..."
    return s


def _format_physical_terminal_process_name(row: Any) -> str:
    """Return the inferred physical termination label for a selected-track row."""
    value = getattr(row, "physical_terminal_process", None)
    return _format_terminal_process_name(value)


def _terminal_process_color_map(process_names: Iterable[str]) -> Dict[str, str]:
    """Deterministic colour map for terminal-process marker traces."""
    palette = (
        px.colors.qualitative.Bold
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Set2
        + px.colors.qualitative.Plotly
        + px.colors.qualitative.Safe
    )
    names = sorted({_format_terminal_process_name(p) for p in process_names})
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}


def add_terminal_process_markers(
    fig: go.Figure,
    chosen: pd.DataFrame,
    track_color_map: Optional[Dict[int, str]] = None,
    show_terminal_process_labels: bool = True,
    terminal_marker_size: int = 10,
    terminal_marker_symbol: str = "diamond",
    show_terminal_process_legend_item: bool = True,
) -> None:
    """
    Add one terminal-process marker for each selected track.

    Marker colour is keyed by inferred physical terminal process, not by track.
    Hover text is intentionally compact/wrapped so it does not cover the plot.
    """
    if chosen.empty:
        return

    required = [
        "terminal_x_mm",
        "terminal_y_mm",
        "terminal_z_mm",
        "terminal_process",
        "terminal_t_ns",
        "terminal_ke_MeV",
        "physical_terminal_process",
    ]
    missing = [c for c in required if c not in chosen.columns]
    if missing:
        raise ValueError(f"chosen summary is missing terminal-process columns: {missing}")

    rows_by_process: Dict[str, Dict[str, list]] = {}
    process_color_map = _terminal_process_color_map(chosen["physical_terminal_process"].astype(str).tolist())

    for row in chosen.itertuples(index=False):
        if not np.isfinite(float(row.terminal_x_mm)):
            continue

        trk = int(row.trk)
        proc = _format_physical_terminal_process_name(row)
        raw_proc = _format_terminal_process_name(row.terminal_process)
        color = process_color_map[proc]

        if proc not in rows_by_process:
            rows_by_process[proc] = {
                "x": [],
                "y": [],
                "z": [],
                "text": [],
                "label": [],
                "color": color,
            }

        rows_by_process[proc]["x"].append(float(row.terminal_x_mm))
        rows_by_process[proc]["y"].append(float(row.terminal_y_mm))
        rows_by_process[proc]["z"].append(float(row.terminal_z_mm))
        rows_by_process[proc]["label"].append(proc if show_terminal_process_labels else "")

        detail = _wrap_hover_text(
            getattr(row, "physical_terminal_detail", ""),
            width=62,
            max_chars=520,
            max_semicolon_items=6,
        )

        rows_by_process[proc]["text"].append(
            f"<b>termination: {_hover_safe_text(proc)}</b><br>"
            f"raw G4 final step: {_short_hover_value(raw_proc, 70)}<br>"
            f"track={trk}: {_hover_safe_text(row.particle)} (PDG {int(row.pdg)})<br>"
            f"creation KE={row.creation_ke_MeV:.3g} MeV<br>"
            f"terminal KE={row.terminal_ke_MeV:.3g} MeV, t={row.terminal_t_ns:.3g} ns<br>"
            f"terminal step={int(row.terminal_step)}<br>"
            f"terminal volume={_short_hover_value(row.terminal_volume, 70)}<br>"
            f"evidence={_short_hover_value(row.terminal_evidence, 60)}, "
            f"inferred={bool(row.terminal_is_inferred)}<br>"
            f"termination detail:<br>{detail}<br>"
            f"x,y,z=({row.terminal_x_mm:.1f}, {row.terminal_y_mm:.1f}, {row.terminal_z_mm:.1f}) mm"
        )

    mode = "markers+text" if show_terminal_process_labels else "markers"

    for proc in sorted(rows_by_process.keys()):
        d = rows_by_process[proc]
        fig.add_trace(
            go.Scatter3d(
                x=d["x"],
                y=d["y"],
                z=d["z"],
                mode=mode,
                marker=dict(
                    size=terminal_marker_size,
                    color=d["color"],
                    symbol=terminal_marker_symbol,
                    opacity=1.0,
                    line=dict(color="black", width=2),
                ),
                text=d["label"],
                textposition="top center",
                textfont=dict(size=12),
                hovertext=d["text"],
                hovertemplate="%{hovertext}<extra></extra>",
                name=f"termination: {proc}",
                legendgroup=f"termination:{proc}",
                showlegend=show_terminal_process_legend_item,
            )
        )


# -----------------------------------------------------------------------------
# Main display class
# -----------------------------------------------------------------------------

@dataclass
class WCTEEventDisplay:
    """Reusable event display object. Build once, plot many events."""

    df_sec: Union[pd.DataFrame, Dict[str, Any]]
    df_sec_ph: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None
    geometry_parent: Union[str, Path] = "."
    place_info: str = "design"
    y_wcsim_origin_in_geom_mm: float = Y_WCSIM_ORIGIN_IN_GEOM_MM_DEFAULT

    def __post_init__(self) -> None:
        self.tracks = prepare_secondary_df(self.df_sec, self.y_wcsim_origin_in_geom_mm)
        self.photons = prepare_photon_df(self.df_sec_ph)

        WCD = _import_wcd_class(self.geometry_parent)
        self.wcd = WCD("wcte")
        self.geom_cache = build_wcte_geometry_cache(self.wcd, place_info=self.place_info)

    def plot_event(
        self,
        event_id: int,
        # Track display cuts
        time_cut_ns: Optional[float] = 20.0,
        time_cut_column: str = "t_ns",
        show_electrons: bool = True,
        hide_neutrinos: bool = True,
        min_track_length_mm: Optional[float] = 2.0,
        apply_min_track_length_cut: bool = True,
        min_creation_ke_MeV: Optional[float] = 100.0,
        max_tracks: Optional[int] = 200,
        always_show_decay_daughters: bool = False,
        decay_daughter_pdgs: Optional[Iterable[int]] = DEFAULT_DECAY_DAUGHTER_PDGS,
        decay_daughters_ignore_time_cut: bool = True,
        physical_rest_ke_threshold_MeV: float = DEFAULT_PHYSICAL_REST_KE_THRESHOLD_MEV,
        charged_only: bool = False,
        include_pdgs: Optional[Iterable[int]] = None,
        exclude_pdgs: Optional[Iterable[int]] = None,
        # PMT / PE display
        show_pmt_pe: bool = True,
        pmt_display_mode: str = "all",  # "all" or "hit_only"
        pe_time_cut_ns: Union[str, float, None] = "same_as_track",
        max_pmt_rings: Optional[int] = None,
        unhit_pmt_size: int = 4,
        outer_hit_size: int = 15,
        hit_size_step: int = 4,
        link_pmt_visibility_to_tracks: bool = True,
        # Detector display
        show_detector: bool = True,
        show_shell: bool = True,
        show_mpmt_outlines: bool = True,
        show_mpmt_centers: bool = False,
        show_mpmt_slot_numbers: bool = False,
        # Track display
        show_track_starts: bool = True,
        show_track_ends: bool = False,
        # Process / interaction display
        show_interactions: bool = True,
        show_em_interactions: bool = False,
        interaction_processes: Optional[Iterable[str]] = None,
        max_interaction_markers: int = 600,
        # Terminal-process display
        show_terminal_process_markers: bool = True,
        show_terminal_process_labels: bool = True,
        terminal_process_in_track_legend: bool = True,
        terminal_marker_size: int = 10,
        terminal_marker_symbol: str = "diamond",
        show_terminal_process_legend_item: bool = True,
        # Figure size
        width: int = 1100,
        height: int = 850,
    ) -> Tuple[go.Figure, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Plot one WCSim event.

        Returns
        -------
        fig, summary, chosen, pmt_pe_df
        """
        tracks_df = self.tracks
        photons_df = self.photons
        geom_cache = self.geom_cache

        ev_all = tracks_df.loc[tracks_df["evt"] == event_id].sort_values(["trk", "step"])
        if ev_all.empty:
            raise ValueError(f"No track rows found for evt={event_id}")

        ev = apply_event_display_cuts(
            ev_all,
            time_cut_ns=time_cut_ns,
            time_cut_column=time_cut_column,
            show_electrons=show_electrons,
            hide_neutrinos=hide_neutrinos,
        )

        if always_show_decay_daughters and decay_daughters_ignore_time_cut:
            ev = _add_forced_decay_daughter_rows(
                ev,
                ev_all,
                show_electrons=show_electrons,
                hide_neutrinos=hide_neutrinos,
                decay_daughter_pdgs=decay_daughter_pdgs,
            )

        summary = event_track_summary(
            tracks_df,
            event_id,
            time_cut_ns=time_cut_ns,
            time_cut_column=time_cut_column,
            show_electrons=show_electrons,
            hide_neutrinos=hide_neutrinos,
            always_show_decay_daughters=always_show_decay_daughters,
            decay_daughter_pdgs=decay_daughter_pdgs,
            decay_daughters_ignore_time_cut=decay_daughters_ignore_time_cut,
            physical_rest_ke_threshold_MeV=physical_rest_ke_threshold_MeV,
        )

        chosen = select_tracks(
            summary,
            min_track_length_mm=min_track_length_mm,
            apply_min_track_length_cut=apply_min_track_length_cut,
            min_creation_ke_MeV=min_creation_ke_MeV,
            max_tracks=max_tracks,
            charged_only=charged_only,
            include_pdgs=include_pdgs,
            exclude_pdgs=exclude_pdgs,
            always_show_decay_daughters=always_show_decay_daughters,
            decay_daughter_pdgs=decay_daughter_pdgs,
        )

        chosen_trks = set(chosen["trk"].astype(int).tolist())
        fig = go.Figure()

        if show_detector:
            add_detector_context(
                fig,
                geom_cache,
                show_shell=show_shell,
                show_mpmt_outlines=show_mpmt_outlines,
                show_mpmt_centers=show_mpmt_centers,
                show_mpmt_slot_numbers=show_mpmt_slot_numbers,
            )

        palette = (
            px.colors.qualitative.Alphabet
            + px.colors.qualitative.Dark24
            + px.colors.qualitative.Light24
            + px.colors.qualitative.Safe
        )

        track_color_map = {}
        for i, row in enumerate(chosen.itertuples(index=False)):
            track_color_map[int(row.trk)] = palette[i % len(palette)]

        # Tracks
        for row in chosen.itertuples(index=False):
            g = ev.loc[ev["trk"] == row.trk]
            if g.empty:
                continue

            pts = _track_points_mm(g)
            color = track_color_map[int(row.trk)]

            is_primary = int(row.parent) == 0
            is_charged = float(row.charge) != 0.0
            is_forced_decay = bool(getattr(row, "is_forced_decay_daughter", False))
            line_width = 8 if is_primary else (7 if is_forced_decay else (5 if is_charged else 3))

            hover = _make_track_hover(row, time_cut_ns, min_creation_ke_MeV)

            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    line=dict(color=color, width=line_width),
                    name=(
                        f"{int(row.trk)}: {row.particle}, KE₀={row.creation_ke_MeV:.1f} MeV"
                        + (f" → {_format_physical_terminal_process_name(row)}" if terminal_process_in_track_legend else "")
                        + (" [forced decay daughter]" if is_forced_decay else "")
                    ),
                    legendgroup=f"trk{int(row.trk)}",
                    hovertemplate=hover + "<extra></extra>",
                )
            )

            if show_track_starts:
                fig.add_trace(
                    go.Scatter3d(
                        x=[pts[0, 0]],
                        y=[pts[0, 1]],
                        z=[pts[0, 2]],
                        mode="markers",
                        marker=dict(size=10 if is_forced_decay else (8 if is_primary else 5), color=color, symbol="circle"),
                        name=f"start {int(row.trk)}",
                        legendgroup=f"trk{int(row.trk)}",
                        showlegend=False,
                        hovertemplate="START<br>" + hover + "<extra></extra>",
                    )
                )

            if show_track_ends:
                fig.add_trace(
                    go.Scatter3d(
                        x=[pts[-1, 0]],
                        y=[pts[-1, 1]],
                        z=[pts[-1, 2]],
                        mode="markers",
                        marker=dict(size=5, color=color, symbol="x"),
                        name=f"end {int(row.trk)}",
                        legendgroup=f"trk{int(row.trk)}",
                        showlegend=False,
                        hovertemplate="END<br>" + hover + "<extra></extra>",
                    )
                )

        # Terminal-process markers from the full uncut track history.
        # This lets, for example, a selected pi+ show a Decay endpoint even when
        # the low-energy daughter mu+ is hidden by the creation-KE cut.
        if show_terminal_process_markers:
            add_terminal_process_markers(
                fig,
                chosen,
                track_color_map=track_color_map,
                show_terminal_process_labels=show_terminal_process_labels,
                terminal_marker_size=terminal_marker_size,
                terminal_marker_symbol=terminal_marker_symbol,
                show_terminal_process_legend_item=show_terminal_process_legend_item,
            )

        # PMTs coloured by PE-origin track
        pmt_pe_df = pd.DataFrame()
        if show_pmt_pe:
            pe_summary = get_event_pe_summary(
                photons_df,
                event_id,
                shown_track_ids=chosen_trks,
                geom=geom_cache,
                pe_time_cut_ns=pe_time_cut_ns,
                track_time_cut_ns=time_cut_ns,
            )
            pmt_pe_df = build_pmt_display_dataframe(pe_summary, chosen, geom_cache)
            add_pmt_pe_markers(
                fig,
                pmt_pe_df,
                geom=geom_cache,
                track_color_map=track_color_map,
                pmt_display_mode=pmt_display_mode,
                max_pmt_rings=max_pmt_rings,
                unhit_pmt_size=unhit_pmt_size,
                outer_hit_size=outer_hit_size,
                hit_size_step=hit_size_step,
                link_pmt_visibility_to_tracks=link_pmt_visibility_to_tracks,
            )

        # Interaction markers
        if show_interactions and not ev.empty:
            processes = set(DEFAULT_INTERACTION_PROCESSES if interaction_processes is None else interaction_processes)
            if show_em_interactions:
                processes |= DEFAULT_EM_EXTRA_PROCESSES

            inter = ev[
                ev["trk"].astype(int).isin(chosen_trks)
                & ev["step_process"].astype(str).isin(processes)
            ].copy()

            if len(inter) > max_interaction_markers:
                inter = inter.sort_values(["t_ns", "trk", "step"]).head(max_interaction_markers)

            proc_palette = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

            for j, (proc, g) in enumerate(inter.groupby(inter["step_process"].astype(str), observed=True)):
                text = [
                    f"process={r.step_process}<br>"
                    f"track={int(r.trk)}; particle={r.particle} (PDG {int(r.pdg)})<br>"
                    f"parent={int(r.parent)}; creator={r.creator}<br>"
                    f"t={r.post_t_ns:.3g} ns<br>"
                    f"KE: {r.ke_MeV:.3g} → {r.post_ke_MeV:.3g} MeV<br>"
                    f"edep={r.edep_MeV:.3g} MeV<br>"
                    f"volume: {r.volume} → {r.post_volume}<br>"
                    f"material: {r.material} → {r.post_material}<br>"
                    f"x,y,z=({r.post_x_mm:.1f}, {r.post_y_mm:.1f}, {r.post_z_mm:.1f}) mm"
                    for r in g.itertuples(index=False)
                ]

                fig.add_trace(
                    go.Scatter3d(
                        x=g["post_x_mm"],
                        y=g["post_y_mm"],
                        z=g["post_z_mm"],
                        mode="markers",
                        marker=dict(size=7, color=proc_palette[j % len(proc_palette)], symbol="diamond"),
                        text=text,
                        hovertemplate="%{text}<extra></extra>",
                        name=f"process: {proc}",
                    )
                )

        # Layout
        time_cut_text = "no track timing cut" if time_cut_ns is None else f"track t_ns < {time_cut_ns:g} ns"
        if pe_time_cut_ns == "same_as_track":
            pe_cut_text = "no PE timing cut" if time_cut_ns is None else f"PE time cut follows track cut ({time_cut_ns:g} ns)"
        elif pe_time_cut_ns is None:
            pe_cut_text = "no PE timing cut"
        else:
            pe_cut_text = f"PE time < {float(pe_time_cut_ns):g} ns"

        electron_text = "electrons shown" if show_electrons else "electrons hidden"
        ke_cut_text = "no creation KE cut" if min_creation_ke_MeV is None else f"creation KE ≥ {min_creation_ke_MeV:g} MeV"
        if apply_min_track_length_cut and min_track_length_mm is not None:
            length_cut_text = f"displayed length ≥ {min_track_length_mm:g} mm"
        else:
            length_cut_text = "no track-length cut"
        pmt_mode_text = "all PMTs shown" if pmt_display_mode == "all" else "only PE PMTs shown"
        if always_show_decay_daughters:
            if decay_daughters_ignore_time_cut:
                decay_text = "low-KE decay daughters force-kept, including outside timing cut"
            else:
                decay_text = "low-KE decay daughters force-kept within timing cut"
        else:
            decay_text = "low-KE daughters hidden unless they pass cuts"
        n_hit_pmts = pmt_pe_df["pe_pmt"].nunique() if len(pmt_pe_df) else 0

        title = (
            f"WCSim event {event_id}: {len(chosen)} plotted tracks, "
            f"{n_hit_pmts} PMTs with PE from shown tracks<br>"
            f"{time_cut_text}; {pe_cut_text}; {ke_cut_text}; {length_cut_text}; "
            f"{electron_text}; {pmt_mode_text}; {decay_text}; "
            f"terminal process markers={'on' if show_terminal_process_markers else 'off'}"
        )

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            scene=dict(
                xaxis_title="x_geom [mm]",
                yaxis_title="y_geom [mm] vertical",
                zaxis_title="z_geom [mm] beam",
                aspectmode="data",
                camera=dict(eye=dict(x=1.55, y=1.25, z=1.35)),
            ),
            legend=dict(
                itemsizing="constant",
                groupclick="togglegroup",
            ),
            hoverlabel=dict(
                align="left",
                font_size=12,
                namelength=-1,
            ),
            margin=dict(l=0, r=0, b=0, t=90),
        )

        return fig, summary, chosen, pmt_pe_df

    def make_browser(
        self,
        default_event: Optional[int] = None,
        time_cut_ns: Optional[float] = 20.0,
        min_creation_ke_MeV: Optional[float] = 100.0,
        show_electrons: bool = True,
        show_mpmt_slot_numbers: bool = True,
        pmt_display_mode: str = "all",
        pe_time_cut_ns: Union[str, float, None] = "same_as_track",
        min_track_length_mm: Optional[float] = 2.0,
        apply_min_track_length_cut: bool = True,
        max_tracks: Optional[int] = 200,
        always_show_decay_daughters: bool = False,
        decay_daughters_ignore_time_cut: bool = True,
    ) -> None:
        """Create a simple Jupyter widget browser. Requires ipywidgets."""
        import ipywidgets as widgets  # pylint: disable=import-outside-toplevel
        from IPython.display import clear_output, display  # pylint: disable=import-outside-toplevel

        event_ids = sorted(self.tracks["evt"].unique().tolist())
        if default_event is None:
            default_event = event_ids[0]

        evt_dropdown = widgets.Dropdown(options=event_ids, value=default_event, description="evt")
        use_track_time_cut_box = widgets.Checkbox(value=time_cut_ns is not None, description="use track t cut")
        track_time_cut_box = widgets.FloatText(value=time_cut_ns if time_cut_ns is not None else 20.0, description="track t ns")
        use_pe_time_cut_box = widgets.Checkbox(value=(pe_time_cut_ns is not None), description="use PE t cut")
        pe_time_same_box = widgets.Checkbox(value=(pe_time_cut_ns == "same_as_track"), description="PE t = track t")
        pe_time_cut_box = widgets.FloatText(value=20.0, description="PE t ns")
        use_ke_cut_box = widgets.Checkbox(value=min_creation_ke_MeV is not None, description="use KE cut")
        min_ke_box = widgets.FloatText(value=min_creation_ke_MeV if min_creation_ke_MeV is not None else 100.0, description="min KE MeV")
        electron_box = widgets.Checkbox(value=show_electrons, description="show e±")
        force_decay_box = widgets.Checkbox(value=always_show_decay_daughters, description="force low-KE daughters")
        decay_ignore_time_box = widgets.Checkbox(value=decay_daughters_ignore_time_cut, description="forced ignores t")
        slot_box = widgets.Checkbox(value=show_mpmt_slot_numbers, description="slot nums")
        pmt_mode_box = widgets.Dropdown(options=["all", "hit_only"], value=pmt_display_mode, description="PMTs")
        min_len_box = widgets.FloatText(
            value=min_track_length_mm if min_track_length_mm is not None else 2.0,
            description="min len mm",
        )
        use_len_cut_box = widgets.Checkbox(value=apply_min_track_length_cut, description="use len cut")
        max_tracks_box = widgets.IntText(value=max_tracks if max_tracks is not None else 999999, description="max tracks")
        out = widgets.Output()

        def redraw(_change=None):
            with out:
                clear_output(wait=True)
                this_track_time_cut = track_time_cut_box.value if use_track_time_cut_box.value else None

                if not use_pe_time_cut_box.value:
                    this_pe_time_cut = None
                elif pe_time_same_box.value:
                    this_pe_time_cut = "same_as_track"
                else:
                    this_pe_time_cut = pe_time_cut_box.value

                this_min_ke = min_ke_box.value if use_ke_cut_box.value else None
                this_max_tracks = max_tracks_box.value
                if this_max_tracks <= 0 or this_max_tracks >= 999999:
                    this_max_tracks = None

                this_apply_len_cut = bool(use_len_cut_box.value)
                this_min_track_length = min_len_box.value if this_apply_len_cut else None

                fig, summary, _chosen, pmt_pe_df = self.plot_event(
                    evt_dropdown.value,
                    time_cut_ns=this_track_time_cut,
                    pe_time_cut_ns=this_pe_time_cut,
                    min_creation_ke_MeV=this_min_ke,
                    show_electrons=electron_box.value,
                    show_mpmt_slot_numbers=slot_box.value,
                    pmt_display_mode=pmt_mode_box.value,
                    min_track_length_mm=this_min_track_length,
                    apply_min_track_length_cut=this_apply_len_cut,
                    max_tracks=this_max_tracks,
                    always_show_decay_daughters=force_decay_box.value,
                    decay_daughters_ignore_time_cut=decay_ignore_time_box.value,
                )
                display(summary.head(30))
                if len(pmt_pe_df):
                    display(pmt_pe_df.head(30))
                fig.show()

        controls_1 = widgets.HBox([
            evt_dropdown,
            use_track_time_cut_box,
            track_time_cut_box,
            use_pe_time_cut_box,
            pe_time_same_box,
            pe_time_cut_box,
        ])
        controls_2 = widgets.HBox([use_ke_cut_box, min_ke_box, electron_box, force_decay_box, decay_ignore_time_box, slot_box, pmt_mode_box])
        controls_3 = widgets.HBox([use_len_cut_box, min_len_box, max_tracks_box])

        for widget in [
            evt_dropdown,
            use_track_time_cut_box,
            track_time_cut_box,
            use_pe_time_cut_box,
            pe_time_same_box,
            pe_time_cut_box,
            use_ke_cut_box,
            min_ke_box,
            electron_box,
            force_decay_box,
            decay_ignore_time_box,
            slot_box,
            pmt_mode_box,
            min_len_box,
            use_len_cut_box,
            max_tracks_box,
        ]:
            widget.observe(redraw, names="value")

        display(controls_1, controls_2, controls_3, out)
        redraw()


# -----------------------------------------------------------------------------
# One-shot convenience function
# -----------------------------------------------------------------------------

def plot_wcsim_event(
    df_sec: Union[pd.DataFrame, Dict[str, Any]],
    df_sec_ph: Optional[Union[pd.DataFrame, Dict[str, Any]]],
    event_id: int,
    geometry_parent: Union[str, Path] = ".",
    place_info: str = "design",
    y_wcsim_origin_in_geom_mm: float = Y_WCSIM_ORIGIN_IN_GEOM_MM_DEFAULT,
    **plot_kwargs: Any,
) -> Tuple[go.Figure, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-shot convenience wrapper.

    For plotting many events, prefer WCTEEventDisplay(...) and then
    display3d.plot_event(...), because that only builds the geometry cache once.
    """
    display3d = WCTEEventDisplay(
        df_sec=df_sec,
        df_sec_ph=df_sec_ph,
        geometry_parent=geometry_parent,
        place_info=place_info,
        y_wcsim_origin_in_geom_mm=y_wcsim_origin_in_geom_mm,
    )
    return display3d.plot_event(event_id=event_id, **plot_kwargs)


__all__ = [
    "WCTEEventDisplay",
    "plot_wcsim_event",
    "prepare_secondary_df",
    "prepare_photon_df",
    "build_wcte_geometry_cache",
    "Y_WCSIM_ORIGIN_IN_GEOM_MM_DEFAULT",
    "DEFAULT_DECAY_DAUGHTER_PDGS",
]
