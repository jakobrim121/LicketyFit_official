"""Geometry-profiled mPMT boundary light for production cosmic fits.

A candidate charged-particle line is intersected with the placed module and the
finite-cone model supplies two detector-local shapes:

``local_wcpmt``
    Light made inside a WCPMT gel/glass sector.  Matrix and reflector walls
    localise this population overwhelmingly to the traversed PMT.

``outer_shell``
    Light made in the outer silicone-gel shell and acrylic dome.  This can
    illuminate several PMTs through their finite openings.

The spatial shapes and their relative timing nodes are geometry derived.  Their
absolute detected throughput is not claimed to be known at Geant4 precision:
the ordinary LicketyFit Emitter itself is charge conditioned and uses an
effective optical normalization.  We therefore profile one non-negative charge
fraction per *active physical component*.

Crucially, each fraction has a candidate-dependent upper bound derived from the
finite-cone raw prediction relative to the ordinary water raw prediction.  A
track that only grazes transparent material has a vanishing bound and cannot
claim an arbitrary local flash.  The water-only model remains the exact
zero-fraction point.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .cosmic_track_fit import BoundarySurfaceHit
from .mpmt_boundary import LocalModePrediction, ModuleGeometry
from .mpmt_hardware import PhysicalMPMTBoundaryModel


@dataclass
class GeometryProfiledMPMTModel:
    """Profile physically separated local-light components for one module."""

    physical: PhysicalMPMTBoundaryModel
    amplitude_cap_multiplier: float = 4.0
    maximum_total_fraction: float = 0.35
    minimum_active_raw_fraction: float = 1.0e-8
    include_timing_nodes: bool = False

    @property
    def module(self) -> ModuleGeometry:
        return self.physical.module

    @property
    def n_profile_parameters(self) -> int:
        # The model exposes at most two component amplitudes.  Inactive rows are
        # removed automatically by the convex profiler and do not contribute to
        # the event-level complexity penalty.
        return 2

    def predict_profile_modes(
        self,
        *,
        boundary_hit: BoundarySurfaceHit,
        direction: Sequence[float],
        interface: str,
        kinetic_energy_mev: float,
        emitter: object,
        boundary_particle_time_ns: float = 0.0,
    ) -> LocalModePrediction:
        raw = self.physical.predict_raw(
            boundary_hit=boundary_hit,
            direction=direction,
            interface=interface,
            kinetic_energy_mev=float(kinetic_energy_mev),
            emitter=emitter,
            boundary_particle_time_ns=float(boundary_particle_time_ns),
            include_timing_nodes=bool(self.include_timing_nodes),
        )
        components = np.asarray(raw.raw_charge_modes, dtype=np.float64)
        if components.ndim != 2 or components.shape[1] != int(self.physical.n_detector_pmts):
            components = np.asarray(raw.raw_charge, dtype=np.float64)[None, :]
            mode_names = ("finite_cone_hardware",)
        else:
            mode_names = tuple(str(name) for name in raw.mode_names)

        component_totals = np.sum(components, axis=1)
        templates = np.zeros_like(components)
        active = np.isfinite(component_totals) & (
            component_totals > float(self.minimum_active_raw_fraction)
        )
        if np.any(active):
            templates[active] = components[active] / component_totals[active, None]

        base_raw = np.asarray(
            getattr(emitter, "_last_expected_pes_raw", np.empty(0)),
            dtype=np.float64,
        )
        base_total = float(np.sum(base_raw[np.isfinite(base_raw) & (base_raw > 0.0)]))
        hw_total = float(np.sum(component_totals[active]))
        denominator = max(base_total + hw_total, 1.0e-300)
        reference = np.where(active, np.maximum(component_totals, 0.0) / denominator, 0.0)
        multiplier = max(float(self.amplitude_cap_multiplier), 1.0)
        max_fractions = np.minimum(
            multiplier * reference,
            float(self.maximum_total_fraction),
        )
        max_fractions[~active] = 0.0

        node_mu = np.asarray(raw.node_mu_raw, dtype=np.float64)
        node_t = np.asarray(raw.node_t_ns, dtype=np.float64)
        node_modes = np.asarray(raw.node_modes, dtype=np.int32)
        if (
            node_mu.ndim != 2
            or node_t.shape != node_mu.shape
            or node_modes.shape != (node_mu.shape[0],)
        ):
            node_mu = np.empty((0, components.shape[1]), dtype=np.float64)
            node_t = np.empty((0, components.shape[1]), dtype=np.float64)
            node_modes = np.empty(0, dtype=np.int32)
        else:
            # Normalize source rows independently within each physical mode so
            # their sum reproduces that mode's unit charge template.
            for mode_index, total in enumerate(component_totals):
                rows = node_modes == int(mode_index)
                if np.any(rows):
                    if math.isfinite(float(total)) and float(total) > 0.0:
                        node_mu[rows] /= float(total)
                    else:
                        node_mu[rows] = 0.0

        return LocalModePrediction(
            templates=np.ascontiguousarray(templates, dtype=np.float64),
            mode_names=mode_names,
            node_weights=np.ascontiguousarray(node_mu, dtype=np.float64),
            node_times_ns=np.ascontiguousarray(node_t, dtype=np.float64),
            node_modes=np.ascontiguousarray(node_modes, dtype=np.int32),
            diagnostics={
                "model": "geometry_profiled_finite_cone_components_v2",
                "absolute_amplitude_used": False,
                "amplitude_cap_multiplier": float(multiplier),
                "include_timing_nodes": bool(self.include_timing_nodes),
                "base_raw_total": float(base_total),
                "component_raw_totals": component_totals.astype(float).tolist(),
                "reference_fractions": reference.astype(float).tolist(),
                "max_fractions": max_fractions.astype(float).tolist(),
                "physical": dict(raw.diagnostics),
            },
            max_fractions=np.ascontiguousarray(max_fractions, dtype=np.float64),
            reference_fractions=np.ascontiguousarray(reference, dtype=np.float64),
        )
