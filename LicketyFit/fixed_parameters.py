"""Validated user-facing fixed-parameter configuration for run launchers."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real


# Include the historical aliases so a launcher with no configured constraints
# cannot accidentally inherit a fixed parameter from the calling shell.
FIXED_PARAMETER_ENV_NAMES = (
    "FIX_X0",
    "FIX_Y0",
    "FIX_Z0",
    "FIX_DIR_X",
    "FIX_DIR_Y",
    "FIX_DIR_Z",
    "FIX_T0",
    "FIX_LENGTH",
    "FIX_VISIBLE_LENGTH",
    "FIX_FULL_RANGE",
    "FIXED_KE0_MEV",
    "FIX_KE0_MEV",
    "FIX_CX",
    "FIX_CY",
)

_SCALAR_ENV_BY_KEY = {
    "x0_mm": "FIX_X0",
    "y0_mm": "FIX_Y0",
    "z0_mm": "FIX_Z0",
    "t0_ns": "FIX_T0",
    "length_mm": "FIX_LENGTH",
    "visible_length_mm": "FIX_VISIBLE_LENGTH",
    "full_range_mm": "FIX_FULL_RANGE",
    "kinetic_energy_mev": "FIXED_KE0_MEV",
}
_KNOWN_KEYS = frozenset((*_SCALAR_ENV_BY_KEY, "direction"))
_COMMON_KEYS = frozenset(("x0_mm", "y0_mm", "z0_mm", "direction", "t0_ns"))
_FULL_LENGTH_GENERAL_KEYS = _COMMON_KEYS | {
    "length_mm", "full_range_mm", "kinetic_energy_mev"
}
_FULL_LENGTH_BEAM_KEYS = _COMMON_KEYS | {"length_mm"}
_ABSORPTION_KEYS = _COMMON_KEYS | {
    "visible_length_mm", "full_range_mm", "kinetic_energy_mev"
}
_LEGACY_MODE_AXES = {
    "beam": ("beam", "full_length"),
    "full_length": ("beam", "full_length"),
    "general": ("general", "full_length"),
    "cosmic": ("general", "full_length"),
    "absorption": ("general", "absorption"),
}


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"FIXED_PARAMETERS[{name!r}] must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"FIXED_PARAMETERS[{name!r}] must be finite")
    return number


def _unit_direction(value: object) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError(
            "FIXED_PARAMETERS['direction'] must contain exactly three numbers"
        )
    try:
        components = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            "FIXED_PARAMETERS['direction'] must contain exactly three numbers"
        ) from exc
    if len(components) != 3:
        raise ValueError(
            "FIXED_PARAMETERS['direction'] must contain exactly three numbers"
        )
    direction = tuple(
        _finite_number(f"direction[{index}]", component)
        for index, component in enumerate(components)
    )
    norm = math.sqrt(sum(component * component for component in direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("FIXED_PARAMETERS['direction'] must be nonzero")
    return tuple(component / norm for component in direction)  # type: ignore[return-value]


def resolve_fixed_parameter_environment(
    fixed_parameters: Mapping[str, object],
    *,
    extra_driver_env: Mapping[object, object],
    seeding_mode: str | None = None,
    interaction_mode: str | None = None,
    fit_mode: str | None = None,
) -> dict[str, float | None]:
    """Return deterministic driver variables for physical fixed parameters.

    ``kinetic_energy_mev`` is an alternate specification of the full CSDA
    range.  In general/full-length fits, ``length_mm`` names that same
    coordinate, so exactly one of those aliases may be supplied.
    """
    if not isinstance(fixed_parameters, Mapping):
        raise TypeError("FIXED_PARAMETERS must be a mapping")
    if not isinstance(extra_driver_env, Mapping):
        raise TypeError("EXTRA_DRIVER_ENV must be a mapping")

    normalized = {str(name): value for name, value in fixed_parameters.items()}
    if len(normalized) != len(fixed_parameters):
        raise ValueError("FIXED_PARAMETERS contains duplicate string-equivalent keys")
    unknown = sorted(set(normalized) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown FIXED_PARAMETERS key(s) {unknown}; valid keys are "
            f"{sorted(_KNOWN_KEYS)}"
        )

    supplied_axes = seeding_mode is not None or interaction_mode is not None
    if supplied_axes:
        if seeding_mode is None or interaction_mode is None:
            raise ValueError(
                "SEEDING_MODE and INTERACTION_MODE must be supplied together"
            )
        if fit_mode is not None:
            raise ValueError(
                "Do not combine legacy FIT_MODE with SEEDING_MODE/INTERACTION_MODE"
            )
        seeding = str(seeding_mode).strip().lower().replace("-", "_")
        interaction = str(interaction_mode).strip().lower().replace("-", "_")
        if seeding not in {"beam", "general"}:
            raise ValueError("SEEDING_MODE must be beam or general")
        if interaction not in {"full_length", "absorption"}:
            raise ValueError("INTERACTION_MODE must be full_length or absorption")
    else:
        legacy = str(fit_mode).strip().lower().replace("-", "_")
        try:
            seeding, interaction = _LEGACY_MODE_AXES[legacy]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported legacy FIT_MODE={fit_mode!r} for fixed parameters"
            ) from exc

    if interaction == "absorption":
        allowed_keys = _ABSORPTION_KEYS
    elif seeding == "general":
        allowed_keys = _FULL_LENGTH_GENERAL_KEYS
    else:
        allowed_keys = _FULL_LENGTH_BEAM_KEYS
    invalid = sorted(set(normalized) - allowed_keys)
    if invalid:
        raise ValueError(
            f"FIXED_PARAMETERS key(s) {invalid} are not physical coordinates "
            f"of SEEDING_MODE={seeding!r}, INTERACTION_MODE={interaction!r}; "
            f"valid keys are {sorted(allowed_keys)}"
        )
    if "full_range_mm" in normalized and "kinetic_energy_mev" in normalized:
        raise ValueError(
            "Choose either full_range_mm or kinetic_energy_mev; they specify "
            "the same physical degree of freedom"
        )
    if (
        interaction == "full_length"
        and seeding == "general"
        and "length_mm" in normalized
        and (
            "full_range_mm" in normalized
            or "kinetic_energy_mev" in normalized
        )
    ):
        raise ValueError(
            "Choose one of length_mm, full_range_mm, or kinetic_energy_mev; "
            "they specify the same full-length coordinate in general seeding"
        )

    fixed_extra_names = sorted(
        set(map(str, extra_driver_env)).intersection(FIXED_PARAMETER_ENV_NAMES)
    )
    if normalized and fixed_extra_names:
        raise ValueError(
            "Configure physical constraints in FIXED_PARAMETERS only; remove "
            f"{fixed_extra_names} from EXTRA_DRIVER_ENV"
        )

    environment: dict[str, float | None] = {
        name: None for name in FIXED_PARAMETER_ENV_NAMES
    }
    for key, env_name in _SCALAR_ENV_BY_KEY.items():
        if key not in normalized:
            continue
        number = _finite_number(key, normalized[key])
        if key in {
            "length_mm",
            "visible_length_mm",
            "full_range_mm",
            "kinetic_energy_mev",
        } and number <= 0.0:
            raise ValueError(f"FIXED_PARAMETERS[{key!r}] must be positive")
        environment[env_name] = number

    if "direction" in normalized:
        direction = _unit_direction(normalized["direction"])
        environment["FIX_DIR_X"] = direction[0]
        environment["FIX_DIR_Y"] = direction[1]
        environment["FIX_DIR_Z"] = direction[2]

    visible = environment["FIX_VISIBLE_LENGTH"]
    full_range = environment["FIX_FULL_RANGE"]
    if visible is not None and full_range is not None and visible > full_range:
        raise ValueError("visible_length_mm cannot exceed full_range_mm")
    return environment
