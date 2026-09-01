"""Fail-closed configuration contracts for coherent MCS implementations.

The public ``coherent_fisher`` emitter model is an umbrella for three distinct
continuations.  Their latent ranks and numerical grids are not interchangeable,
so configuration is resolved before any multiprocessing worker pool is created.
"""
from __future__ import annotations


PHYSICS_REFERENCE = "physics_reference"
FAST12_PROFILE = "fast12_profile"
LEGACY_FISHER = "legacy_fisher"
COHERENT_IMPLEMENTATIONS = (
    PHYSICS_REFERENCE,
    FAST12_PROFILE,
    LEGACY_FISHER,
)

_MODES_PER_PLANE = {
    PHYSICS_REFERENCE: 12,
    FAST12_PROFILE: 12,
    LEGACY_FISHER: 4,
}


def normalize_coherent_implementation(implementation: str) -> str:
    """Return a canonical implementation name or reject it explicitly."""
    name = str(implementation).strip().lower().replace("-", "_")
    if name not in COHERENT_IMPLEMENTATIONS:
        choices = ", ".join(COHERENT_IMPLEMENTATIONS)
        raise ValueError(
            f"unknown coherent MCS implementation {implementation!r}; "
            f"expected one of: {choices}"
        )
    return name


def default_coherent_modes_per_plane(implementation: str) -> int:
    """Return the implementation's physics-defined transverse KL rank."""
    return int(_MODES_PER_PLANE[normalize_coherent_implementation(implementation)])


def validate_coherent_configuration(
    implementation: str,
    modes_per_plane: int,
    process_grid_points: int,
    coherent_grid_points: int,
) -> str:
    """Validate rank/grid coupling and return the canonical implementation.

    This is intentionally called in the batch parent before detector setup and
    before a worker pool exists.  A bad configuration therefore produces one
    actionable error instead of an initializer failure followed by worker
    respawning.
    """
    name = normalize_coherent_implementation(implementation)
    modes = int(modes_per_plane)
    expected_modes = default_coherent_modes_per_plane(name)
    if modes != expected_modes:
        total = 2 * expected_modes
        raise ValueError(
            f"MCS_COHERENT_IMPLEMENTATION={name!r} requires exactly "
            f"{expected_modes} modes per transverse plane ({total} latent "
            f"coordinates), but MCS_PROCESS_MODES_PER_PLANE={modes}. "
            "Remove the incompatible override or set the required value."
        )

    process_grid = int(process_grid_points)
    optical_grid = int(coherent_grid_points)
    if process_grid <= 0 or optical_grid <= 0:
        raise ValueError("coherent MCS grid sizes must be positive")

    if name in {PHYSICS_REFERENCE, FAST12_PROFILE} and process_grid != optical_grid:
        raise ValueError(
            f"MCS_COHERENT_IMPLEMENTATION={name!r} requires identical FE and "
            "optical path grids"
        )
    if name == PHYSICS_REFERENCE and process_grid < 241:
        raise ValueError(
            "physics_reference requires a common FE/optical grid of at least "
            "241 points; coarser grids do not meet the validated optical-field "
            "convergence criterion"
        )
    return name


def coherent_warmup_action(implementation: str) -> str:
    """Return the exhaustive warm-up action for a coherent implementation.

    The physics reference is deliberately deferred until an accepted real-event
    straight track exists.  Running its complete exact MAP continuation on the
    proxy warm-up seed would duplicate the scientific fit in every worker.
    """
    name = normalize_coherent_implementation(implementation)
    if name == PHYSICS_REFERENCE:
        return "defer_physics_reference"
    if name == FAST12_PROFILE:
        return FAST12_PROFILE
    if name == LEGACY_FISHER:
        return LEGACY_FISHER
    raise AssertionError(f"unreachable coherent implementation {name!r}")
