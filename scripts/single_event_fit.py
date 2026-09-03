#!/usr/bin/env python3
"""Notebook-facing, single-event interface to the production LicketyFit driver.

This module deliberately does not implement a second fitter.  It configures and
imports the selected embedded engine from :mod:`batch_fit_driver`, uses that
engine's WCTE/WCSim loader and event-preparation functions, and calls its exact
``fit_one_event_by_index`` path for one event at a time.

The configuration classes mirror every public setting in ``run_wcte.py`` and
``run_wcsim.py``. Keyword names are case-insensitive, so ``seeding_mode`` /
``SEEDING_MODE`` and ``interaction_mode`` / ``INTERACTION_MODE`` are accepted.
The notebook layer forces serial execution and
suppresses batch checkpoint/output behavior because one interactive fit runs in
the notebook process.

Reconstruction mode is two independent public axes, not one ``FIT_MODE``:

===================  ==================  ===========================
``seeding_mode``     ``interaction_mode``  retired ``FIT_MODE`` name
===================  ==================  ===========================
``beam``             ``full_length``     ``full_length`` / ``beam``
``general``          ``full_length``     ``cosmic``
``general``          ``absorption``      ``absorption``
``beam``             ``absorption``      (had no legacy name)
===================  ==================  ===========================

``LauncherConfig.reconstruction()`` reports how a chosen pair routes through the
shared rules in :mod:`LicketyFit.run_configuration`, including which MCS
selector is actually live.  Passing the retired ``fit_mode`` keyword raises a
message naming the replacement pair instead of a generic unknown-option list.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field
import importlib.util
import io
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any
import uuid

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DRIVER_PATH = SCRIPT_DIR / "batch_fit_driver.py"

_LAUNCHER_CACHE: dict[str, ModuleType] = {}

# Settings that older notebooks and launch scripts still pass.  A generic
# "unknown option" message followed by a hundred valid names is not useful here,
# so name the replacement directly.
_LEGACY_FIT_MODE_REPLACEMENT = {
    "beam": "seeding_mode='beam', interaction_mode='full_length'",
    "full_length": "seeding_mode='beam', interaction_mode='full_length'",
    "cosmic": "seeding_mode='general', interaction_mode='full_length'",
    "general": "seeding_mode='general', interaction_mode='full_length'",
    "absorption": "seeding_mode='general', interaction_mode='absorption'",
}
_RETIRED_OPTION_GUIDANCE = {
    "FIT_MODE": (
        "Seed coverage and endpoint physics are now independent axes. Use "
        "seeding_mode='beam' or 'general' together with "
        "interaction_mode='full_length' or 'absorption'. The retired names map "
        "as: "
        + "; ".join(
            f"{legacy!r} -> {replacement}"
            for legacy, replacement in _LEGACY_FIT_MODE_REPLACEMENT.items()
        )
        + ". Read the resolved routing back with config.reconstruction()."
    ),
    "SEED_MODE": (
        "The general-mode navigator setting is now "
        "cosmic_multilateration_seed_mode ('off', 'hybrid', 'guided', "
        "'additive', or 'primary')."
    ),
}


def _canonical_internal_fit_mode(
    engine: ModuleType,
    requested: object,
    interaction_mode: object | None = None,
) -> str:
    """Resolve two public axes (or one legacy label) to an engine name."""
    if interaction_mode is not None:
        seeding = str(requested).strip().lower().replace("-", "_")
        interaction = str(interaction_mode).strip().lower().replace("-", "_")
        if seeding == "general" and interaction == "full_length":
            return "cosmic"
        return interaction
    normalized = str(requested).strip().lower().replace("-", "_")
    public = getattr(engine, "_LEGACY_FIT_MODE_ALIASES", {}).get(
        normalized, normalized
    )
    return getattr(engine, "_INTERNAL_FIT_MODE_BY_PUBLIC", {}).get(public, public)


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _launcher(kind: str) -> ModuleType:
    key = str(kind).strip().lower()
    if key not in {"wcte", "wcsim"}:
        raise ValueError("kind must be 'wcte' or 'wcsim'")
    cached = _LAUNCHER_CACHE.get(key)
    if cached is not None:
        return cached
    module = _load_module_from_path(
        f"_licketyfit_single_event_launcher_{key}",
        SCRIPT_DIR / f"run_{key}.py",
    )
    _LAUNCHER_CACHE[key] = module
    return module


def _launcher_field_names(kind: str) -> tuple[str, ...]:
    module = _launcher(kind)
    return tuple(
        name
        for name in module.__dict__
        if name.isupper() and not name.startswith("_")
    )


class LauncherConfig:
    """Mutable configuration backed by one public run launcher.

    Every uppercase setting in the launcher's user-editable configuration block
    is exposed as a lower- or uppercase attribute.  Unknown settings fail fast;
    advanced driver-only environment controls belong in ``extra_driver_env``.
    """

    kind = ""

    def __init__(self, **overrides: Any):
        if self.kind not in {"wcte", "wcsim"}:
            raise TypeError("LauncherConfig must be instantiated through WCTEConfig or WCSimConfig")
        defaults = self.default_values()
        object.__setattr__(self, "_values", defaults)
        self.update(**overrides)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return _launcher_field_names(cls.kind)

    @classmethod
    def default_values(cls) -> dict[str, Any]:
        module = _launcher(cls.kind)
        return {
            name: deepcopy(getattr(module, name))
            for name in cls.field_names()
        }

    @classmethod
    def available_options(cls) -> tuple[str, ...]:
        """Return every public option accepted from the corresponding launcher."""
        return tuple(name.lower() for name in cls.field_names())

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """Return launcher defaults keyed by notebook-style lowercase names."""
        return {name.lower(): value for name, value in cls.default_values().items()}

    @classmethod
    def search_options(cls, substring: str) -> dict[str, Any]:
        """Return the launcher defaults whose option name contains ``substring``.

        ``available_options()`` returns roughly a hundred names.  This narrows
        that list when looking for, for example, every ``pid`` or ``tof``
        setting without opening ``run_wcte.py``.
        """
        needle = str(substring).strip().lower()
        return {
            name: value
            for name, value in cls.defaults().items()
            if needle in name
        }

    def reconstruction(self):
        """Return how this configuration routes through the shared model rules.

        The result is the :class:`LicketyFit.run_configuration.\
ReconstructionConfiguration` produced by the same function both launchers call.
        Reading it before a fit shows which selectors are actually live: in
        ``general`` + ``full_length`` the primary-MCS and coherent-implementation
        choices are inert and ``cosmic_mcs_continuation`` is what applies, while
        every other pair is the reverse.  It also raises the same errors the
        launcher would, so an incompatible combination fails here instead of
        after geometry and table setup.
        """
        _launcher(self.kind)  # Loaded for its sys.path side effect.
        from LicketyFit.run_configuration import (
            resolve_reconstruction_configuration,
        )

        return resolve_reconstruction_configuration(
            likelihood_mode=self.likelihood_mode,
            enable_mcs=bool(self.enable_mcs),
            seed_mode=self.cosmic_multilateration_seed_mode,
            primary_mcs_model=self.primary_mcs_model,
            coherent_mcs_implementation=self.coherent_mcs_implementation,
            cosmic_mcs_continuation=self.cosmic_mcs_continuation,
            cosmic_joint_inference_method=self.cosmic_joint_inference_method,
            seeding_mode=self.seeding_mode,
            interaction_mode=self.interaction_mode,
        )

    def describe(self) -> dict[str, Any]:
        """Return the headline settings plus the resolved reconstruction routing."""
        resolved = self.reconstruction()
        summary: dict[str, Any] = {
            "source": self.kind,
            "fit_particle": self.fit_particle,
            "seeding_mode": resolved.seeding_mode,
            "interaction_mode": resolved.interaction_mode,
            "public_mode_label": resolved.public_mode_label,
            "internal_engine_mode": resolved.internal_engine_mode,
            "likelihood_mode": resolved.likelihood_mode,
            "enable_delta_electrons": bool(self.enable_delta_electrons),
            "enable_mcs": bool(self.enable_mcs),
            "enable_reflection": bool(self.enable_reflection),
            "enable_photon_scattering": bool(self.enable_photon_scattering),
            "primary_mcs_model": resolved.primary_mcs_model,
            "coherent_mcs_implementation": resolved.coherent_mcs_implementation,
            "effective_cosmic_mcs_continuation": (
                resolved.effective_cosmic_mcs_continuation
            ),
            "navigation_mode": resolved.navigation_mode,
            "use_absolute_light_yield": bool(self.use_absolute_light_yield),
            "absolute_light_yield_source": self.absolute_light_yield_source,
            "charge_likelihood": self.charge_likelihood,
            "fixed_parameters": dict(self.fixed_parameters),
        }
        if self.kind == "wcte":
            summary.update({
                "run": int(self.run),
                "event_source": self.event_source,
                "good_pmt_source": self.good_pmt_source,
                "particle_selection_label": self.particle_selection_label,
                "selection_mode": self.selection_mode,
                "light_particle_pid_mode": self.light_particle_pid_mode,
                "relative_efficiency_mode": self.relative_efficiency_mode,
                "geometry_placement": self.geometry_placement,
                "prompt_window_mode": self.prompt_window_mode,
                "time_reference_mode": self.time_reference_mode,
            })
        else:
            summary.update({
                "input_file": self.input_file,
                "energy_label_mev": float(self.energy_label_mev),
                "use_wcte_geometry": bool(self.use_wcte_geometry),
                "use_iwcd_geometry": bool(self.use_iwcd_geometry),
                "use_truth_root": bool(self.use_truth_root),
                "apply_wcsim_visible_range_convention": bool(
                    self.apply_wcsim_visible_range_convention
                ),
            })
        return summary

    def _canonical_name(self, name: str) -> str:
        canonical = str(name).strip().upper()
        if canonical not in self._values:
            guidance = _RETIRED_OPTION_GUIDANCE.get(canonical)
            if guidance is not None:
                raise TypeError(
                    f"{canonical} is no longer a {self.kind.upper()} "
                    f"configuration option. {guidance}"
                )
            choices = ", ".join(self.available_options())
            raise TypeError(
                f"Unknown {self.kind.upper()} configuration option {name!r}. "
                f"Available options are: {choices}"
            )
        return canonical

    def update(self, **overrides: Any) -> "LauncherConfig":
        for name, value in overrides.items():
            self._values[self._canonical_name(name)] = deepcopy(value)
        return self

    def copy(self, **overrides: Any) -> "LauncherConfig":
        clone = type(self)(**self.as_dict(lowercase=True))
        return clone.update(**overrides)

    def as_dict(self, *, lowercase: bool = False) -> dict[str, Any]:
        return {
            (name.lower() if lowercase else name): deepcopy(value)
            for name, value in self._values.items()
        }

    def changed_options(self) -> dict[str, Any]:
        defaults = self.default_values()
        return {
            name.lower(): deepcopy(value)
            for name, value in self._values.items()
            if value != defaults[name]
        }

    def validate(self, *, check_paths: bool = True) -> None:
        """Run the same validation function used by the public launcher."""
        module = _launcher(self.kind)
        originals = {name: getattr(module, name) for name in self.field_names()}
        try:
            for name, value in self._values.items():
                setattr(module, name, deepcopy(value))
            module._validate(check_paths=bool(check_paths))
        finally:
            for name, value in originals.items():
                setattr(module, name, value)

    def driver_environment(self) -> dict[str, str]:
        """Return the production-driver environment encoded by the launcher."""
        module = _launcher(self.kind)
        originals = {name: getattr(module, name) for name in self.field_names()}
        try:
            for name, value in self._values.items():
                setattr(module, name, deepcopy(value))
            return module.build_environment(base=dict(os.environ))
        finally:
            for name, value in originals.items():
                setattr(module, name, value)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        canonical = str(name).upper()
        if canonical in values:
            return values[canonical]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or "_values" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        self._values[self._canonical_name(name)] = deepcopy(value)

    def __repr__(self) -> str:
        changed = self.changed_options()
        payload = ", ".join(f"{key}={value!r}" for key, value in changed.items())
        return f"{type(self).__name__}({payload})"


class WCTEConfig(LauncherConfig):
    """All settings from :mod:`run_wcte`, usable from a notebook."""

    kind = "wcte"


class WCSimConfig(LauncherConfig):
    """All settings from :mod:`run_wcsim`, usable from a notebook."""

    kind = "wcsim"


@dataclass(frozen=True)
class EventRecord:
    """One loaded event before LicketyFit's prompt/channel preparation."""

    source: str
    source_index: int
    pmt_ids: np.ndarray
    charges: np.ndarray
    times_ns: np.ndarray
    pmt_id_mode: str
    raw: Any = field(repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def hit_table(self) -> np.ndarray:
        """Return columns ``[PMT id, charge, time_ns]``."""
        return np.column_stack((self.pmt_ids, self.charges, self.times_ns))

    @property
    def n_hits(self) -> int:
        return int(self.pmt_ids.size)

    @property
    def total_charge(self) -> float:
        return float(np.sum(self.charges))


class EventCollection(Sequence[EventRecord]):
    """Selected/loaded events plus provenance from the production loader."""

    def __init__(self, events: Sequence[EventRecord], *, metadata: Mapping[str, Any] | None = None):
        self._events = tuple(events)
        self.metadata = {} if metadata is None else dict(metadata)

    def __getitem__(self, index):
        return self._events[index]

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[EventRecord]:
        return iter(self._events)

    def __repr__(self) -> str:
        source = self._events[0].source if self._events else self.metadata.get("source", "unknown")
        return f"EventCollection(source={source!r}, n_events={len(self)})"


@dataclass
class FitResult:
    """Compact notebook result plus full production diagnostics."""

    event: EventRecord
    estimates: dict[str, Any]
    errors: dict[str, float]
    fit_statistics: dict[str, Any]
    pmt_ids: np.ndarray
    pmt_slots: np.ndarray
    pmt_positions: np.ndarray
    pmt_coordinates_mm: np.ndarray
    observed_pe: np.ndarray
    expected_pe: np.ndarray
    observed_time_ns: np.ndarray
    expected_time_ns: np.ndarray
    timing_likelihood_mask: np.ndarray
    raw_result: dict[str, Any] = field(repr=False)
    loader_metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    prediction_metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    truth: Mapping[str, Any] | None = field(default=None, repr=False)
    truth_metadata: Mapping[str, Any] | None = field(default=None, repr=False)

    @property
    def fit_accepted(self) -> bool:
        return bool(self.fit_statistics.get("fit_accepted", False))

    @property
    def fval(self) -> float:
        return float(self.fit_statistics.get("fval", math.nan))

    @property
    def timing_comparison_mask(self) -> np.ndarray:
        """PMTs with both an observed time and a scalar model-time summary."""
        return np.isfinite(self.observed_time_ns) & np.isfinite(self.expected_time_ns)

    def summary(self) -> dict[str, Any]:
        """Return the intentionally small interactive fit summary."""
        return dict(self.fit_statistics)

    def parameter_table(self):
        """Return a pandas table of estimates and reported uncertainties."""
        import pandas as pd

        preferred = (
            "x0", "y0", "z0", "cx", "cy", "cz", "length",
            "visible_length", "full_range", "t0", "track_topology",
        )
        names = [name for name in preferred if name in self.estimates]
        names.extend(name for name in self.estimates if name not in names and name in self.errors)
        rows = []
        for name in names:
            value = self.estimates[name]
            if isinstance(value, (str, bool)) or np.isscalar(value):
                rows.append({
                    "parameter": name,
                    "estimate": value,
                    "error": self.errors.get(name, math.nan),
                })
        return pd.DataFrame(rows).set_index("parameter")

    def pmt_table(self, *, hit_only: bool = False):
        """Return PMT-aligned observed and predicted PE/time diagnostics."""
        import pandas as pd

        table = pd.DataFrame({
            "pmt_id": self.pmt_ids,
            "slot": self.pmt_slots,
            "pmt_position": self.pmt_positions,
            "pmt_x_mm": self.pmt_coordinates_mm[:, 0],
            "pmt_y_mm": self.pmt_coordinates_mm[:, 1],
            "pmt_z_mm": self.pmt_coordinates_mm[:, 2],
            "observed_pe": self.observed_pe,
            "expected_pe": self.expected_pe,
            "observed_time_ns": self.observed_time_ns,
            "expected_time_ns": self.expected_time_ns,
            "timing_used_in_fit": self.timing_likelihood_mask,
            "time_diagnostic_available": self.timing_comparison_mask,
        })
        if hit_only:
            table = table.loc[
                (table["observed_pe"] > 0.0)
                | (table["expected_pe"] > 0.0)
                | np.isfinite(table["observed_time_ns"])
            ]
        return table.reset_index(drop=True)

    def __repr__(self) -> str:
        return (
            f"FitResult(source={self.event.source!r}, source_index={self.event.source_index}, "
            f"accepted={self.fit_accepted}, fval={self.fval:.6g})"
        )


@contextmanager
def _temporary_environment(environment: Mapping[str, str]):
    original = dict(os.environ)
    os.environ.clear()
    os.environ.update({str(key): str(value) for key, value in environment.items()})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


@contextmanager
def _quiet(enabled: bool):
    if not enabled:
        yield None
        return
    stream = io.StringIO()
    with redirect_stdout(stream):
        yield stream


class SingleEventFitter:
    """Load and fit individual WCTE or WCSim events with the production engine."""

    def __init__(self, config: WCTEConfig | WCSimConfig, *, verbose: bool = False):
        if not isinstance(config, (WCTEConfig, WCSimConfig)):
            raise TypeError("config must be WCTEConfig or WCSimConfig")
        self.config = config.copy()
        self.verbose = bool(verbose)
        self._engine: ModuleType | None = None
        self._driver_environment: dict[str, str] | None = None
        self._initialized = False
        self._events: EventCollection | None = None
        self._wcsim_raw: Mapping[str, Any] | None = None
        self._loader_metadata: dict[str, Any] = {}
        self._active_pmt_ids_cache: np.ndarray | None = None

    @property
    def source(self) -> str:
        return self.config.kind

    @property
    def events(self) -> EventCollection:
        if self._events is None:
            raise RuntimeError("No events are loaded. Call load_events() first.")
        return self._events

    def _notebook_environment(self) -> dict[str, str]:
        env = self.config.driver_environment()
        # Interactive single-event policy. These change execution mechanics only;
        # the loader, detector, physics, seed bank and fit objective are unchanged.
        env.update({
            "NPROC": "1",
            "N_EVENTS_PER_BATCH": "1",
            # A notebook can initialize WCTE and WCSim engines in one kernel.
            # Numba forbids changing its thread-pool size after first launch,
            # while the two batch launchers deliberately carry different CPU
            # budgets.  One stable four-thread latency budget avoids cross-engine
            # pool-size changes while matching the production NPROC=1 MCS policy.
            "NUMBA_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "1",
            "WARM_FIT_KERNELS": "0",
            "SAVE_AFTER_EACH_BATCH": "0",
            "SAVE_DETAILED_EVENT_RESULTS": "1",
            "PRINT_EVENT_RESULTS": "0",
            "PRINT_BATCH_PROGRESS": "0",
            "PRINT_CHECKPOINT_MESSAGES": "0",
            "VERBOSE_SETUP": "1" if self.verbose else "0",
            "LF_COSMIC_SUPERVISED_CHILD": "1",
            "LF_COSMIC_CHILD_QUIET": "0" if self.verbose else "1",
            "LF_COSMIC_HARD_EXIT": "0",
            "LF_SINGLE_EVENT_API": "1",
        })
        return env

    def _load_engine(self) -> ModuleType:
        if self._engine is not None:
            return self._engine
        self.config.validate(check_paths=True)
        for path in (PROJECT_ROOT, SCRIPT_DIR):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        environment = self._notebook_environment()
        self._driver_environment = environment
        module_name = f"_licketyfit_single_event_engine_{self.source}_{uuid.uuid4().hex}"
        with _temporary_environment(environment), _quiet(not self.verbose):
            engine = _load_module_from_path(module_name, DRIVER_PATH)
        if getattr(engine, "_UNIFIED_DATA_SOURCE", None) != self.source:
            raise RuntimeError("The embedded driver selected the wrong data source")
        expected_mode = _canonical_internal_fit_mode(
            engine,
            self.config.seeding_mode,
            self.config.interaction_mode,
        )
        if getattr(engine, "_UNIFIED_FIT_MODE", None) != expected_mode:
            raise RuntimeError("The embedded driver selected the wrong fit mode")
        self._engine = engine
        return engine

    def _runtime_environment(self):
        if self._driver_environment is None:
            raise RuntimeError("The production engine environment is not initialized")
        return _temporary_environment(self._driver_environment)

    def initialize(self) -> "SingleEventFitter":
        """Build geometry, PMT state, range tables, seeds and proxy library once."""
        if self._initialized:
            return self
        engine = self._load_engine()
        start = time.perf_counter()
        with self._runtime_environment(), _quiet(not self.verbose):
            engine.initialize()
        self._initialized = True
        self._loader_metadata.setdefault("single_event_setup_wall_s", time.perf_counter() - start)
        return self

    def load_events(
        self,
        *,
        max_events: int | None = None,
        start_index: int | None = None,
    ) -> EventCollection:
        """Load events through the same input functions as the production driver.

        For WCTE, ``max_events`` means selected events after collaboration cuts.
        ``n_root_entries`` in :class:`WCTEConfig` still controls the maximum raw
        entries inspected. For WCSim it means consecutive NPZ events.
        """
        engine = self._load_engine()
        start = int(self.config.event_start_index if start_index is None else start_index)
        if start < 0:
            raise ValueError("start_index must be nonnegative")

        if max_events is None:
            configured = (
                self.config.max_events_to_fit
                if self.source == "wcte"
                else self.config.n_events
            )
            count = None if configured is None else int(configured)
        else:
            count = int(max_events)
        if count is not None and count < 1:
            raise ValueError("max_events must be positive or None")

        records: list[EventRecord] = []
        if self.source == "wcsim":
            fields = list(getattr(
                engine,
                "FIT_FIELDS",
                ("digi_hit_pmt", "digi_hit_charge", "digi_hit_time"),
            ))
            if bool(self.config.use_truth_root):
                fields.extend(getattr(
                    engine,
                    "TRUTH_TRACK_ID_FIELDS",
                    ("track_id", "track_pid", "track_parent"),
                ))
            with self._runtime_environment():
                raw = engine.read_sim_data(
                    str(engine.INPUT_FILE), fields=fields
                )
            self._wcsim_raw = raw
            available = int(len(raw["digi_hit_time"]))
            stop = available if count is None else min(available, start + count)
            configured_id_mode = str(self.config.pmt_id_mode).strip().lower()
            event_id_mode = "wcte" if configured_id_mode == "wcte" else "wcsim"
            configured_mapping = str(self.config.wcsim_wcte_mapping_file).strip()
            mapping_path = (
                configured_mapping
                if configured_mapping
                else str(PROJECT_ROOT / "tables" / "wcsim_wcte_mapping.txt")
            )
            for source_index in range(min(start, available), stop):
                records.append(EventRecord(
                    source="wcsim",
                    source_index=int(source_index),
                    pmt_ids=np.asarray(raw["digi_hit_pmt"][source_index], dtype=np.int64),
                    charges=np.asarray(raw["digi_hit_charge"][source_index], dtype=np.float64),
                    times_ns=np.asarray(raw["digi_hit_time"][source_index], dtype=np.float64),
                    pmt_id_mode=event_id_mode,
                    raw=raw,
                    metadata={
                        "input_file": str(engine.INPUT_FILE),
                        "configured_pmt_id_mode": configured_id_mode,
                        "pmt_id_offset": int(self.config.pmt_id_offset),
                        "mapping_path": mapping_path,
                    },
                ))
            metadata = {
                "source": "wcsim",
                "input_file": str(engine.INPUT_FILE),
                "available_events": available,
                "loaded_interval": (min(start, available), stop),
            }
        else:
            load_limit = None if count is None else start + count
            with self._runtime_environment(), _quiet(not self.verbose):
                if "max_events_override" in engine.load_input_events.__code__.co_varnames:
                    raw_events = engine.load_input_events(max_events_override=load_limit)
                else:
                    previous = engine.MAX_EVENTS_TO_FIT
                    try:
                        engine.MAX_EVENTS_TO_FIT = load_limit
                        raw_events = engine.load_input_events()
                    finally:
                        engine.MAX_EVENTS_TO_FIT = previous
            stop = len(raw_events) if count is None else min(len(raw_events), start + count)
            for source_index in range(min(start, len(raw_events)), stop):
                array = np.asarray(raw_events[source_index], dtype=np.float64)
                if array.ndim != 2 or array.shape[1] < 3:
                    raise ValueError(
                        f"WCTE loader event {source_index} is not an N_hit x >=3 table"
                    )
                metadata_row: dict[str, Any] = {}
                if array.shape[1] >= 4 and array.shape[0]:
                    metadata_row["source_root_entry_index"] = int(array[0, 3])
                if array.shape[1] >= 5 and array.shape[0]:
                    metadata_row["source_event_id"] = int(array[0, 4])
                records.append(EventRecord(
                    source="wcte",
                    source_index=int(source_index),
                    pmt_ids=np.asarray(array[:, 0], dtype=np.int64),
                    charges=np.asarray(array[:, 1], dtype=np.float64),
                    times_ns=np.asarray(array[:, 2], dtype=np.float64),
                    pmt_id_mode="wcte",
                    raw=array,
                    metadata=metadata_row,
                ))
            metadata = dict(getattr(engine, "WCTE_DATA_LOADER_METADATA", {}) or {})
            metadata.update({
                "source": "wcte",
                "event_source": str(engine.EVENT_SOURCE),
                "loaded_selected_events_before_slice": int(len(raw_events)),
                "loaded_interval": (min(start, len(raw_events)), stop),
            })

        self._loader_metadata = metadata
        self._events = EventCollection(records, metadata=metadata)
        return self._events

    def selection_summary(self) -> dict[str, Any]:
        """Return the beam PID and cuts that produced the loaded WCTE events.

        ``LIGHT_PARTICLE_PID_MODE`` decides a policy, not a fixed cut: the
        electron/muon and muon/pion TOF boundaries are resolved per ROOT input.
        This reports the requested mode, the mode actually applied to this
        sample, the resolved boundaries and where each came from, whether the
        per-boundary ACT fallback was used, and the cut list handed to
        ``BeamSelection``.  A boundary of ``None`` was not usable for this run
        and is not a value to copy into another run's override.
        """
        if self.source != "wcte":
            raise RuntimeError(
                "selection_summary() describes WCTE beam selection; WCSim NPZ "
                "input has no BeamSelection stage"
            )
        metadata = dict(self._loader_metadata)
        if not metadata:
            raise RuntimeError("No events are loaded. Call load_events() first.")
        thresholds = dict(metadata.get("selection_thresholds", {}) or {})
        summary: dict[str, Any] = {
            "run": int(self.config.run),
            "event_source": str(metadata.get("event_source", self.config.event_source)),
            "particle_selection_label": str(metadata.get("particle", "")),
            "selection_mode": str(metadata.get("selection_mode", "")),
            "light_particle_pid_mode": metadata.get("light_particle_pid_mode"),
            # None here means the sample is not a nominal light-particle
            # population, so the PID mode did not apply to it at all.
            "light_particle_pid_mode_applied": metadata.get(
                "light_particle_pid_mode_applied"
            ),
            "act_fallback_used": bool(
                thresholds.get("light_particle_tof_fallback_used", False)
            ),
            "unavailable_tof_boundaries": list(
                thresholds.get("light_particle_tof_unavailable_boundaries", ()) or ()
            ),
            "selection_cut_specs": list(metadata.get("selection_cut_specs", []) or []),
            "events_returned": metadata.get("events_returned"),
            "selected_before_hit_time_cut": metadata.get(
                "selected_before_hit_time_cut"
            ),
        }
        for name in (
            "electron_muon_tof_boundary_ns",
            "muon_pion_tof_boundary_ns",
            "proton_tof_cut_ns",
            "act_eveto_cut_pe",
            "act_tagger_cut_pe",
            "mu_tag_cut",
        ):
            summary[name] = thresholds.get(name)
            source_name = name.replace("_ns", "").replace("_pe", "") + "_source"
            if source_name in thresholds:
                summary[source_name] = thresholds.get(source_name)
        for name in (
            "electron_muon_tof_boundary_error",
            "muon_pion_tof_boundary_error",
        ):
            if thresholds.get(name):
                summary[name] = thresholds.get(name)
        return summary

    def _resolve_event(self, event: int | EventRecord) -> EventRecord:
        if isinstance(event, EventRecord):
            if event.source != self.source:
                raise ValueError(
                    f"Cannot fit a {event.source} event with a {self.source} fitter"
                )
            return event
        if isinstance(event, (int, np.integer)):
            return self.events[int(event)]
        raise TypeError("event must be an EventRecord or an index into loaded events")

    @staticmethod
    def _set_prepared_arrays(engine: ModuleType, prepared: tuple, record: EventRecord) -> None:
        engine.OBS_PES_ALL = [prepared[0]]
        engine.OBS_TS_ALL = [prepared[1]]
        engine.OBS_PROMPT_MIN_ALL = np.asarray([prepared[2]], dtype=np.float64)
        engine.OBS_PROMPT_MAX_ALL = np.asarray([prepared[3]], dtype=np.float64)
        if record.source == "wcte":
            assignments = {
                "OBS_SOURCE_EVENT_ID_ALL": np.asarray([prepared[4]], dtype=np.int64),
                "OBS_SOURCE_ROOT_ENTRY_INDEX_ALL": np.asarray([prepared[5]], dtype=np.int64),
                "OBS_SOURCE_INPUT_INDEX_ALL": np.asarray([record.source_index], dtype=np.int64),
                "OBS_TIME_OFFSET_ALL": np.asarray([prepared[6]], dtype=np.float64),
                "OBS_RAW_HIT_COUNT_ALL": np.asarray([prepared[7]], dtype=np.int64),
                "OBS_PROMPT_HIT_COUNT_ALL": np.asarray([prepared[8]], dtype=np.int64),
            }
            for name, value in assignments.items():
                if hasattr(engine, name):
                    setattr(engine, name, value)

    @staticmethod
    def _chart_and_local_values(engine: ModuleType, result: Mapping[str, Any]):
        values = dict(result["values"])
        metadata = values.get("direction_chart")
        if isinstance(metadata, Mapping):
            chart = engine.TangentDirectionChart(
                np.asarray(metadata["anchor"], dtype=np.float64),
                np.asarray(metadata["e1"], dtype=np.float64),
                np.asarray(metadata["e2"], dtype=np.float64),
            )
        else:
            chart = engine.TangentDirectionChart.from_direction(
                (values["cx"], values["cy"], values["cz"])
            )
        local = dict(values)
        local["dir_u"] = float(values.get("direction_chart_u", values.get("dir_u", 0.0)))
        local["dir_v"] = float(values.get("direction_chart_v", values.get("dir_v", 0.0)))
        for axis in ("x0", "y0", "z0"):
            local[axis] = float(values.get(f"line_reference_{axis}", values[axis]))
        if bool(getattr(engine, "AUTO_CLIPPED_TRACK", False)):
            local["length"] = float(values.get(
                "fit_full_range_parameter_mm", values.get("full_range", values["length"])
            ))
        return chart, local

    @staticmethod
    def _active_pmt_ids(engine: ModuleType) -> np.ndarray:
        count = int(len(engine.P_LOCATIONS))
        # Real-WCTE placements are built in sorted authoritative-ID order.
        # PMT_ID_TO_POSITION is deliberately a geometry lookup (ID -> xyz), not
        # an array-index lookup, despite its historical name.
        active_wcte_ids = getattr(engine, "ACTIVE_WCTE_PMT_IDS", None)
        if active_wcte_ids is not None:
            ids = np.asarray(sorted(int(value) for value in active_wcte_ids), dtype=np.int64)
            if ids.size != count:
                raise RuntimeError("Authoritative WCTE PMT IDs do not match the fitter ordering")
            return ids

        empty = engine.event_from_wcsim([], [], [])
        ids: list[int] = []
        for slot in range(int(empty.n_mpmt)):
            if not bool(empty.mpmt_status[slot]) or engine.WCD.mpmts[slot] is None:
                continue
            for position in range(int(empty.npmt_per_mpmt)):
                if (
                    bool(empty.pmt_status[slot][position])
                    and position < len(engine.WCD.mpmts[slot].pmts)
                    and engine.WCD.mpmts[slot].pmts[position] is not None
                ):
                    ids.append(100 * slot + position)
        array = np.asarray(ids, dtype=np.int64)
        if array.size != count:
            raise RuntimeError("WCSim active-PMT IDs do not match the fitter ordering")
        return array

    @staticmethod
    def _first_arrival_node_mean(
        node_mu: np.ndarray,
        node_time: np.ndarray,
        observed_pe: float,
        output_efficiency: float,
    ) -> float:
        """Return the production discrete first-source distribution's mean time."""
        mu = np.asarray(node_mu, dtype=np.float64)
        times = np.asarray(node_time, dtype=np.float64)
        valid = np.isfinite(mu) & (mu > 0.0) & np.isfinite(times)
        if not np.any(valid):
            return math.nan
        mu = mu[valid]
        times = times[valid]
        order = np.argsort(times, kind="stable")
        mu = mu[order]
        times = times[order]
        total = float(np.sum(mu))
        if not math.isfinite(total) or total <= 0.0:
            return math.nan
        neff = max(
            float(observed_pe) / max(float(output_efficiency), 1.0e-12),
            1.0e-6,
        )
        remaining = 1.0
        remaining_power = 1.0
        weighted_time = 0.0
        weight_sum = 0.0
        for amplitude, node_t in zip(mu, times):
            next_remaining = max(0.0, remaining - float(amplitude) / total)
            next_power = next_remaining ** neff
            weight = remaining_power - next_power
            remaining = next_remaining
            remaining_power = next_power
            if math.isfinite(weight) and weight > 0.0:
                weighted_time += weight * float(node_t)
                weight_sum += weight
        return weighted_time / weight_sum if weight_sum > 0.0 else math.nan

    @staticmethod
    def _expected_time_summary(
        prediction: Any,
        observed_pe: np.ndarray,
        t0: float,
        pmt_model: Any,
    ) -> tuple[np.ndarray, str]:
        """Make a PMT-aligned point summary of the production timing model.

        First-arrival timing is a source-resolved probability distribution, so
        the nominal ndarray carried by ``TimingPrediction`` is intentionally
        uninitialized.  For that model we report the mean of the exact discrete
        first-source distribution before zero-mean transit-time smearing.  For
        the legacy mean-time model, its conventional PMT prediction is returned.
        """
        nominal = np.asarray(prediction, dtype=np.float64)
        output = np.full(np.asarray(observed_pe).shape, np.nan, dtype=np.float64)
        if not bool(getattr(prediction, "first_arrival_model", False)):
            if nominal.shape != output.shape:
                raise RuntimeError("Conventional expected-time prediction is not PMT aligned")
            finite = np.isfinite(nominal)
            output[finite] = nominal[finite] + float(t0)
            return output, "production conventional mean arrival time"

        active = np.asarray(
            getattr(prediction, "first_arrival_active_indices", ()), dtype=np.int64
        )
        efficiency = float(getattr(
            pmt_model, "first_arrival_output_efficiency", 0.985
        ))
        node_time = getattr(prediction, "first_arrival_node_t", None)
        node_weight = getattr(prediction, "first_arrival_node_weight", None)
        node_mu = getattr(prediction, "first_arrival_node_mu", None)

        deferred_mu = getattr(prediction, "first_arrival_deferred_base_mu", None)
        deferred_time = getattr(prediction, "first_arrival_deferred_base_t", None)
        reflection_u = getattr(prediction, "first_arrival_reflection_u", None)
        reflection_tbase = getattr(prediction, "first_arrival_reflection_tbase", None)
        reflection_transfer = getattr(
            prediction, "first_arrival_reflection_transfer_active", None
        )
        reflection_offset = getattr(
            prediction, "first_arrival_reflection_time_offset_active", None
        )

        if all(value is not None for value in (
            deferred_mu, deferred_time, reflection_u, reflection_tbase,
            reflection_transfer, reflection_offset,
        )):
            base_mu = np.asarray(deferred_mu, dtype=np.float64)
            base_time = np.asarray(deferred_time, dtype=np.float64)
            ref_u = np.asarray(reflection_u, dtype=np.float64)
            ref_tbase = np.asarray(reflection_tbase, dtype=np.float64)
            transfer = np.asarray(reflection_transfer, dtype=np.float64)
            offsets = np.asarray(reflection_offset, dtype=np.float64)
            patch_min = np.asarray(getattr(
                prediction, "first_arrival_reflection_patch_min_time_offset"
            ), dtype=np.float64)
            patch_max = np.asarray(getattr(
                prediction, "first_arrival_reflection_patch_max_time_offset"
            ), dtype=np.float64)
            n_bins = int(getattr(prediction, "first_arrival_reflection_n_bins"))
            valid_patches = np.isfinite(ref_u) & (ref_u > 0.0)
            if np.any(valid_patches) and n_bins > 0:
                t_min = float(np.min(ref_tbase[valid_patches] + patch_min[valid_patches]))
                t_max = float(np.max(ref_tbase[valid_patches] + patch_max[valid_patches]))
                span = max(t_max - t_min, 1.0e-12)
                for column, pmt_index in enumerate(active):
                    if not (0 <= int(pmt_index) < output.size):
                        continue
                    amplitudes = list(base_mu[:, column])
                    times = list(base_time[:, column])
                    reflected_mu = ref_u * transfer[column]
                    reflected_t = ref_tbase + offsets[column]
                    valid_reflection = (
                        np.isfinite(reflected_mu)
                        & (reflected_mu > 0.0)
                        & np.isfinite(reflected_t)
                    )
                    safe_reflected_t = np.where(valid_reflection, reflected_t, t_min)
                    bin_index = np.clip(
                        ((safe_reflected_t - t_min) * n_bins / span).astype(np.int64),
                        0,
                        n_bins - 1,
                    )
                    bin_mu = np.bincount(
                        bin_index,
                        weights=np.where(valid_reflection, reflected_mu, 0.0),
                        minlength=n_bins,
                    )
                    bin_mt = np.bincount(
                        bin_index,
                        weights=np.where(
                            valid_reflection,
                            reflected_mu * reflected_t,
                            0.0,
                        ),
                        minlength=n_bins,
                    )
                    populated = bin_mu > 0.0
                    amplitudes.extend(bin_mu[populated])
                    times.extend(bin_mt[populated] / bin_mu[populated])
                    output[int(pmt_index)] = SingleEventFitter._first_arrival_node_mean(
                        np.asarray(amplitudes),
                        np.asarray(times),
                        float(observed_pe[int(pmt_index)]),
                        efficiency,
                    )
        elif node_time is not None and active.size:
            times = np.asarray(node_time, dtype=np.float64)
            if node_weight is not None:
                weights = np.asarray(node_weight, dtype=np.float64)
                if weights.shape != times.shape:
                    raise RuntimeError("First-arrival weight/time arrays are not aligned")
                for column, pmt_index in enumerate(active):
                    valid = (
                        np.isfinite(weights[:, column])
                        & (weights[:, column] > 0.0)
                        & np.isfinite(times[:, column])
                    )
                    if np.any(valid) and 0 <= int(pmt_index) < output.size:
                        output[int(pmt_index)] = float(np.average(
                            times[valid, column], weights=weights[valid, column]
                        ))
            elif node_mu is not None:
                amplitudes = np.asarray(node_mu, dtype=np.float64)
                if amplitudes.shape != times.shape:
                    raise RuntimeError("First-arrival amplitude/time arrays are not aligned")
                for column, pmt_index in enumerate(active):
                    if 0 <= int(pmt_index) < output.size:
                        output[int(pmt_index)] = SingleEventFitter._first_arrival_node_mean(
                            amplitudes[:, column],
                            times[:, column],
                            float(observed_pe[int(pmt_index)]),
                            efficiency,
                        )

        finite = np.isfinite(output)
        output[finite] += float(t0)
        return (
            output,
            "production first-arrival source-distribution mean before transit-time smearing",
        )

    @staticmethod
    def _prediction_from_objective(
        engine: ModuleType,
        raw_result: Mapping[str, Any],
        obs_pes: np.ndarray,
        obs_ts: np.ndarray,
        captured_boundary_winner: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        chart, local = SingleEventFitter._chart_and_local_values(engine, raw_result)
        objective_obs_ts = np.asarray(obs_ts, dtype=np.float64)
        objective_kwargs: dict[str, Any] = {"proxy": False}
        prediction_metadata: dict[str, Any] = {
            "prediction_likelihood_mode": "charge_time",
            "uses_final_track_parameters": True,
            "includes_selected_mpmt_boundary_model": False,
        }
        timing_mask = np.isfinite(objective_obs_ts)

        if bool(getattr(engine, "AUTO_CLIPPED_TRACK", False)) or bool(
            getattr(engine, "BOUNDARY_CLIPPED_TRACK", False)
        ):
            hypothesis = str(raw_result.get(
                "chosen_track_start_hypothesis",
                raw_result.get("values", {}).get("track_start_hypothesis", "internal_start"),
            ))
            objective_kwargs["start_hypothesis"] = hypothesis
            prediction_metadata["track_start_hypothesis"] = hypothesis

        if captured_boundary_winner is not None:
            winner_result = captured_boundary_winner.get("result")
            if winner_result is not None:
                chart = winner_result.chart
                local = dict(winner_result.values)
            objective_kwargs["boundary_interface_model"] = captured_boundary_winner["model"]
            objective_kwargs["boundary_interface_timing_policy"] = "baseline"
            objective_kwargs["start_hypothesis"] = str(
                captured_boundary_winner["start_hypothesis"]
            )
            prediction_metadata["includes_selected_mpmt_boundary_model"] = True
            timing_info = captured_boundary_winner.get("timing_metadata") or {}
            masked_slots = [int(value) for value in timing_info.get("masked_slots", [])]
            if bool(timing_info.get("accepted", False)) and masked_slots:
                timing_mask &= ~np.isin(
                    np.asarray(engine.PMT_SLOTS, dtype=np.int64), masked_slots
                )
                objective_obs_ts = objective_obs_ts.copy()
                objective_obs_ts[~timing_mask] = np.nan
                prediction_metadata["timing_masked_slots"] = masked_slots

        objective = engine.make_objective(
            obs_pes,
            objective_obs_ts,
            chart,
            "charge_time",
            **objective_kwargs,
        )
        fval = float(objective(local))
        if not math.isfinite(fval):
            raise RuntimeError("The final fitted point did not produce a finite diagnostic prediction")
        key = objective._geometry_key(local)
        prediction = objective.prediction_cache.get(key)
        if prediction is None:
            raise RuntimeError("The production objective did not retain its final prediction")
        expected_pe, expected_time_zero, _timing_pe = prediction
        expected_pe = np.asarray(expected_pe, dtype=np.float64)
        if expected_time_zero is None:
            expected_time = np.full(expected_pe.shape, np.nan, dtype=np.float64)
            expected_time_definition = "timing prediction unavailable"
        else:
            expected_time, expected_time_definition = (
                SingleEventFitter._expected_time_summary(
                    expected_time_zero,
                    obs_pes,
                    float(local.get("t0", 0.0)),
                    engine.PMT_MODEL,
                )
            )
        prediction_metadata["expected_time_definition"] = expected_time_definition
        prediction_metadata["diagnostic_charge_time_fval"] = fval
        prediction_metadata["fitted_fval"] = float(raw_result.get("fval", math.nan))
        if str(getattr(engine, "LIKELIHOOD_MODE", "charge_only")) == "charge_only":
            timing_mask[:] = False
        return expected_pe, expected_time, timing_mask, prediction_metadata

    @staticmethod
    def _fit_statistics(raw_result: Mapping[str, Any], obs_pe: np.ndarray, exp_pe: np.ndarray) -> dict[str, Any]:
        arbitration = raw_result.get("basin_arbitration")
        basins = [] if not isinstance(arbitration, Mapping) else arbitration.get("optimized_basins", [])
        failures = raw_result.get("candidate_failures", [])
        proxy_results = raw_result.get("proxy_results", [])
        history_records = 0
        for row in proxy_results if isinstance(proxy_results, Sequence) else ():
            if isinstance(row, Mapping):
                history = row.get("history", [])
                history_records += len(history) if isinstance(history, Sequence) else 0
        for name in ("charge_stage", "timing_stage"):
            stage = raw_result.get(name)
            if isinstance(stage, Mapping) and isinstance(stage.get("history"), Sequence):
                history_records += len(stage["history"])
        values = raw_result.get("values", {})
        fitted_fval = float(raw_result.get("fval", math.nan))
        fit_accepted = bool(raw_result.get("fit_accepted", False))
        final_stage = raw_result.get("timing_stage") or raw_result.get("charge_stage")
        final_history = (
            final_stage.get("history", []) if isinstance(final_stage, Mapping) else []
        )
        return {
            "minimum_valid": bool(fit_accepted and math.isfinite(fitted_fval)),
            "fit_accepted": fit_accepted,
            "objective_finite": bool(math.isfinite(fitted_fval)),
            "fval": fitted_fval,
            # "not_requested" means MCS was switched off or routed to a stage
            # this mode does not use; "not_applied" means it was requested and
            # no stage produced a result. Neither is a failure by itself.
            "mcs_status": str(raw_result.get("mcs_status", "not_requested")),
            "mcs_applied": bool(raw_result.get("mcs_applied", False)),
            "straight_fit_fval": float(
                raw_result.get("straight_fit_fval", math.nan)
            ),
            "optimizer": str(raw_result.get("optimizer", "unknown")),
            "total_objective_evaluations": int(raw_result.get("total_nfcn", 0)),
            "fit_attempts": int(len(basins) if isinstance(basins, Sequence) else 0),
            "optimized_basins": int(len(basins) if isinstance(basins, Sequence) else 0),
            "seed_candidates_retained": int(
                len(proxy_results) if isinstance(proxy_results, Sequence) else 0
            ),
            "candidate_failures": int(len(failures) if isinstance(failures, Sequence) else 0),
            "optimizer_history_records": int(history_records),
            "final_optimizer_sweeps": int(
                len(final_history) if isinstance(final_history, Sequence) else 0
            ),
            "invalid_objective_evaluations": int(raw_result.get("invalid_evaluation_count", 0)),
            "fit_wall_s": float(raw_result.get("event_fit_wall_s", math.nan)),
            "chosen_seed_index": int(raw_result.get("chosen_seed_index", -1)),
            "chosen_seed_family": str(
                raw_result.get("chosen_seed", {}).get("seed_family", "unknown")
            ),
            "track_topology": values.get("track_topology", "not_applicable"),
            "observed_post_cut_pe": float(np.sum(obs_pe)),
            "expected_pe_at_estimate": float(np.sum(exp_pe)),
            "n_active_pmts": int(obs_pe.size),
            "n_observed_pmts": int(np.count_nonzero(obs_pe > 0.0)),
            "n_timed_pmts": 0,
        }

    def _load_truth_for_event(self, engine: ModuleType, record: EventRecord):
        if self.source != "wcsim" or not bool(self.config.use_truth_root):
            return None, None
        if self._wcsim_raw is None:
            raise RuntimeError("WCSim truth requires events loaded from the configured NPZ")
        records, metadata = engine._load_optional_wcsim_truth(
            raw=self._wcsim_raw,
            source_event_ids=[int(record.source_index)],
            fit_particle=engine.FIT_PARTICLE,
            fit_detector=engine.DETECTOR,
            wcd=engine.WCD,
            detector_summary=engine.DETECTOR_SUMMARY,
            inactive_slots=engine.INACTIVE_SLOTS,
            volume_class=engine.ConvexDetectorVolume,
            placement="design",
        )
        return records.get(int(record.source_index)), metadata

    def fit(self, event: int | EventRecord) -> FitResult:
        """Fit one loaded event and evaluate PMT predictions at the estimate."""
        record = self._resolve_event(event)
        self.initialize()
        assert self._engine is not None
        engine = self._engine

        with self._runtime_environment():
            # Particle tables live in shared imported modules. Reassert this
            # engine's hypothesis so alternating two notebook fitters remains
            # deterministic even when they use different particles.
            engine.set_active_particle(engine.FIT_PARTICLE)
            prepared = engine.prepare_event_observables(record.raw, record.source_index)
        if prepared is None:
            raise RuntimeError(
                "The selected event has no finite active PE after the production "
                "channel, prompt-window, and time-reference preparation."
            )
        self._set_prepared_arrays(engine, prepared, record)

        captured: dict[str, Any] = {}
        original_challenge = getattr(engine, "_run_mpmt_boundary_challenge", None)
        if callable(original_challenge):
            def capture_challenge(*args, **kwargs):
                winner, summary = original_challenge(*args, **kwargs)
                captured["winner"] = winner
                return winner, summary

            engine._run_mpmt_boundary_challenge = capture_challenge
        try:
            with self._runtime_environment(), _quiet(not self.verbose):
                raw_result = engine.fit_one_event_by_index(0)
        finally:
            if callable(original_challenge):
                engine._run_mpmt_boundary_challenge = original_challenge

        raw_result["source_event_index"] = int(record.source_index)
        obs_pe = np.asarray(prepared[0], dtype=np.float64)
        obs_ts = np.asarray(prepared[1], dtype=np.float64)
        with self._runtime_environment():
            expected_pe, expected_ts, timing_mask, prediction_metadata = (
                self._prediction_from_objective(
                    engine,
                    raw_result,
                    obs_pe,
                    obs_ts,
                    captured.get("winner"),
                )
            )
        if self._active_pmt_ids_cache is None:
            cached_pmt_ids = self._active_pmt_ids(engine)
            cached_pmt_ids.setflags(write=False)
            self._active_pmt_ids_cache = cached_pmt_ids
        # FitResult historically exposed a writable diagnostic array. Keep the
        # immutable resolution cached internally without sharing that storage.
        pmt_ids = np.array(self._active_pmt_ids_cache, copy=True)
        pmt_slots = np.asarray(engine.PMT_SLOTS, dtype=np.int64)
        pmt_positions = pmt_ids % 100
        pmt_coordinates = np.asarray(engine.P_LOCATIONS, dtype=np.float64)
        if not (
            pmt_ids.shape == obs_pe.shape == expected_pe.shape == obs_ts.shape == expected_ts.shape
        ) or pmt_coordinates.shape != (pmt_ids.size, 3):
            raise RuntimeError("PMT diagnostic arrays are not aligned")

        statistics = self._fit_statistics(raw_result, obs_pe, expected_pe)
        statistics["n_timed_pmts"] = int(np.count_nonzero(timing_mask))
        statistics["n_time_diagnostics"] = int(np.count_nonzero(
            np.isfinite(obs_ts) & np.isfinite(expected_ts)
        ))
        with self._runtime_environment():
            truth, truth_metadata = self._load_truth_for_event(engine, record)
        return FitResult(
            event=record,
            estimates=dict(raw_result.get("values", {})),
            errors={
                str(name): float(value)
                for name, value in dict(raw_result.get("errors", {})).items()
                if np.isscalar(value)
            },
            fit_statistics=statistics,
            pmt_ids=pmt_ids,
            pmt_slots=pmt_slots,
            pmt_positions=pmt_positions,
            pmt_coordinates_mm=pmt_coordinates,
            observed_pe=obs_pe,
            expected_pe=expected_pe,
            observed_time_ns=obs_ts,
            expected_time_ns=expected_ts,
            timing_likelihood_mask=np.asarray(timing_mask, dtype=bool),
            raw_result=raw_result,
            loader_metadata=dict(self._loader_metadata),
            prediction_metadata=prediction_metadata,
            truth=truth,
            truth_metadata=truth_metadata,
        )


_PDG_TO_FIT_PARTICLE = {11: "electron", 13: "muon", 211: "pion", 321: "kaon", 2212: "proton"}


def _ensure_repository_on_path() -> None:
    """Make the packaged modules importable without first loading an engine."""
    for path in (PROJECT_ROOT, SCRIPT_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def wcsim_npz_primary_truth(
    npz_path: str | Path,
    event_index: int,
    *,
    detector: str = "wcte",
) -> dict[str, Any]:
    """Return one WCSim NPZ event's primary-particle truth in fitter coordinates.

    This reads only the per-event primary arrays a digitized NPZ already
    carries (``pid``, ``position``, ``direction``, ``energy``).  It is a
    convenience for validating a fit against the sample that produced it; it is
    not the ``AllSecondaries`` truth path, which needs the ROOT file and
    ``use_truth_root=True``.  Nothing here is available to, or used by, the
    likelihood.

    NPZ positions are centimetres in the WCSim frame.  The fitter works in
    millimetres in the geometry frame, and for WCTE the two differ by the prism
    y-centre, so the transform applied is the driver's own
    ``x_detector = 10 * x_wcsim_cm + offset_mm``.

    ``csda_range_mm`` is the packaged range table evaluated at the truth kinetic
    energy: the electromagnetic CSDA distance down to the Cherenkov threshold,
    which is the quantity a ``full_length`` fit estimates.  WCSim muons are
    known to travel slightly less far above threshold than this table predicts
    (roughly 8 mm at 200 MeV rising to 27 mm at 400 MeV), so a small negative
    length residual against this number is expected rather than a fit bias.
    ``run_wcsim.py``'s ``APPLY_WCSIM_VISIBLE_RANGE_CONVENTION`` applies that
    measured offset as a reporting-only correction.
    """
    import numpy as _np

    _ensure_repository_on_path()
    path = Path(npz_path).expanduser()
    with _np.load(path, allow_pickle=True) as raw:
        missing = [
            name for name in ("pid", "position", "direction", "energy")
            if name not in raw.files
        ]
        if missing:
            raise KeyError(
                f"{path.name} has no primary-truth arrays {missing}; it may be a "
                "digitized-only file"
            )
        index = int(event_index)
        available = int(len(raw["pid"]))
        if not 0 <= index < available:
            raise IndexError(
                f"event_index {index} is outside the {available} events in {path.name}"
            )
        pdg = int(raw["pid"][index])
        position_cm = _np.asarray(raw["position"][index], dtype=_np.float64)
        direction = _np.asarray(raw["direction"][index], dtype=_np.float64)
        total_energy_mev = float(raw["energy"][index])

    kind = str(detector).strip().lower()
    if kind == "wcte":
        from LicketyFit.detector_geometry import WCTE_PRISM_Y_CENTER_MM

        offset_mm = _np.array([0.0, float(WCTE_PRISM_Y_CENTER_MM), 0.0])
        offset_source = "wcte_prism_y_center"
    else:
        offset_mm = _np.zeros(3, dtype=_np.float64)
        offset_source = "identity_non_wcte"

    norm = float(_np.linalg.norm(direction))
    unit_direction = direction / norm if norm > 0.0 else direction
    particle = _PDG_TO_FIT_PARTICLE.get(abs(pdg))

    from particle_range_lookup import (
        PARTICLE_MASS_MEV,
        cherenkov_threshold_kinetic_mev,
        particle_energy_to_range_mm,
    )

    truth: dict[str, Any] = {
        "source_event_index": index,
        "pdg_code": pdg,
        "particle": particle,
        "vertex_mm": (10.0 * position_cm + offset_mm),
        "direction": unit_direction,
        "total_energy_mev": total_energy_mev,
        "coordinate_offset_mm": offset_mm,
        "coordinate_offset_source": offset_source,
        "coordinate_transform": "x_detector = 10*x_wcsim_cm + coordinate_offset_mm",
        "kinetic_energy_mev": math.nan,
        "cherenkov_threshold_kinetic_mev": math.nan,
        "csda_range_mm": math.nan,
    }
    if particle is None or particle not in PARTICLE_MASS_MEV:
        return truth
    kinetic = total_energy_mev - float(PARTICLE_MASS_MEV[particle])
    truth["kinetic_energy_mev"] = kinetic
    try:
        truth["cherenkov_threshold_kinetic_mev"] = float(
            cherenkov_threshold_kinetic_mev(particle)
        )
        truth["csda_range_mm"] = float(
            particle_energy_to_range_mm(particle, kinetic)
        )
    except (KeyError, ValueError, FileNotFoundError, NotImplementedError):
        # No packaged range table for this hypothesis; the geometry truth above
        # is still valid.
        pass
    return truth


def truth_residuals(result: FitResult, truth: Mapping[str, Any]):
    """Return fitted-minus-truth residuals for one event as a pandas table.

    ``truth`` is a mapping like :func:`wcsim_npz_primary_truth` returns.  The
    length row compares the fitted longitudinal coordinate with the truth CSDA
    range, so read it with the WCSim range-convention caveat in that function's
    documentation.  One event's residuals are not a bias measurement; that needs
    the mean and the width over an ensemble.
    """
    import pandas as pd

    if not isinstance(result, FitResult):
        raise TypeError("result must be a FitResult")
    estimates = dict(result.estimates)
    vertex = np.asarray(truth["vertex_mm"], dtype=np.float64)
    direction = np.asarray(truth["direction"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for name, true_value in (
        ("x0", vertex[0]), ("y0", vertex[1]), ("z0", vertex[2]),
        ("cx", direction[0]), ("cy", direction[1]), ("cz", direction[2]),
    ):
        if name in estimates:
            fitted = float(estimates[name])
            rows.append({
                "parameter": name,
                "fitted": fitted,
                "truth": float(true_value),
                "residual": fitted - float(true_value),
            })
    length_name = next(
        (name for name in ("length", "full_range") if name in estimates), None
    )
    csda = float(truth.get("csda_range_mm", math.nan))
    if length_name is not None and math.isfinite(csda):
        fitted_length = float(estimates[length_name])
        rows.append({
            "parameter": f"{length_name} vs CSDA range",
            "fitted": fitted_length,
            "truth": csda,
            "residual": fitted_length - csda,
        })
    fitted_dir = np.asarray(
        [estimates.get(name, math.nan) for name in ("cx", "cy", "cz")],
        dtype=np.float64,
    )
    if np.all(np.isfinite(fitted_dir)) and np.linalg.norm(direction) > 0.0:
        cosine = float(np.clip(
            np.dot(fitted_dir, direction)
            / (np.linalg.norm(fitted_dir) * np.linalg.norm(direction)),
            -1.0, 1.0,
        ))
        rows.append({
            "parameter": "opening angle [deg]",
            "fitted": math.nan,
            "truth": 0.0,
            "residual": math.degrees(math.acos(cosine)),
        })
    return pd.DataFrame(rows).set_index("parameter")


def summarize_fit(result: FitResult):
    """Display a compact statistics table and return it as a pandas Series."""
    if not isinstance(result, FitResult):
        raise TypeError("result must be a FitResult")
    import pandas as pd

    summary = pd.Series(result.summary(), name="value")
    try:
        from IPython.display import display
        display(summary.to_frame())
        display(result.parameter_table())
    except ImportError:
        print(summary.to_string())
        print(result.parameter_table().to_string())
    return summary


__all__ = [
    "WCTEConfig",
    "WCSimConfig",
    "EventRecord",
    "EventCollection",
    "FitResult",
    "SingleEventFitter",
    "summarize_fit",
    "truth_residuals",
    "wcsim_npz_primary_truth",
]
