"""Private implementation loaded by ``run_wcte.py``.

Keep user-editable choices in the public launcher; this module owns validation, environment translation, and process orchestration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys


_DRIVER = Path(__file__).resolve().with_name("batch_fit_driver.py")
# CPython never caches bytecode for the script named on the command line, so
# launching the 44k-line driver directly recompiles it on every process start
# (~0.3 s, and once per bootstrap/production/chunk process).  This stub runs
# the same module through the import system, which maintains
# scripts/__pycache__.  __file__, sys.argv and sys.path[0] are unchanged.
_DRIVER_ENTRY = _DRIVER.with_name("lf_driver_main.py")
if not _DRIVER_ENTRY.is_file():
    _DRIVER_ENTRY = _DRIVER
_PROJECT_ROOT = _DRIVER.parent.parent
_ANALYSIS_TOOLS_SUBMODULE = _PROJECT_ROOT / "analysis_tools"
_GEOMETRY_SUBMODULE = _PROJECT_ROOT / "Geometry"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_RUNTIME_DEPS = _PROJECT_ROOT / "runtime_deps"
if _RUNTIME_DEPS.is_dir() and str(_RUNTIME_DEPS) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_DEPS))
_CHARGE_LIKELIHOODS = {
    "poisson_pe", "compound_spe_profile", "compound_spe_profile_reference",
    "compound_spe_calibrated",
}
_REMOVED_MODEL_ENV_SWITCHES = {
    "MCS_RECONSTRUCTION_MODE",
    "EMITTER_ENABLE_PRIMARY_MCS",
    "ENABLE_PRIMARY_MCS",
    "COSMIC_FERMI_EYGES_ENABLED",
    "COSMIC_COHERENT_PROFILE_ENABLED",
    "COSMIC_JOINT_ENERGY_RANGE_ENABLED",
}

from LicketyFit.absolute_light_calibration import resolve_manifest_calibration
from LicketyFit.fixed_parameters import resolve_fixed_parameter_environment
from LicketyFit.run_configuration import (
    FIT_PARTICLES as _FIT_PARTICLES,
    LIKELIHOOD_MODES as _LIKELIHOODS,
    resolve_reconstruction_configuration,
)
from LicketyFit.run_console import (
    print_details,
    print_goodbye,
    print_preparation_notice,
    print_welcome,
    run_with_simple_console,
)


_PUBLIC_DRIVER_RELEASE = "2026-09-04-v1.45.2-absorption-endpoint-fix"
_DEDICATED_RECONSTRUCTION_ENV = {
    "FIT_MODE",
    "LF_SEEDING_MODE",
    "LF_INTERACTION_MODE",
    "LF_PUBLIC_FIT_MODE",
    "EMITTER_PRIMARY_MCS_MODEL",
    "MCS_COHERENT_IMPLEMENTATION",
    "EMITTER_COSMIC_MCS_CONTINUATION",
    "EMITTER_COSMIC_JOINT_INFERENCE_METHOD",
}


def _runtime_dependency_available(module_name: str) -> bool:
    """Return whether a required execution helper imports successfully."""
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError):
        return False
    if module_name == "threadpoolctl":
        return callable(getattr(module, "threadpool_limits", None))
    return True


def _validate_runtime_dependencies() -> None:
    """Fail in the launcher, before data preparation or workers are created."""
    if _runtime_dependency_available("threadpoolctl"):
        return
    requirements = _PROJECT_ROOT / "requirements.txt"
    raise RuntimeError(
        "The validated small-KL accelerator requires threadpoolctl, but it "
        "could not be imported. From the LicketyFit repository root run:\n  "
        f"{sys.executable} -m pip install -r {requirements}\n"
        "Then start this launcher again."
    )


def _resolved_reconstruction_configuration():
    return resolve_reconstruction_configuration(
        seeding_mode=SEEDING_MODE,
        interaction_mode=INTERACTION_MODE,
        likelihood_mode=LIKELIHOOD_MODE,
        enable_mcs=ENABLE_MCS,
        seed_mode=COSMIC_MULTILATERATION_SEED_MODE,
        primary_mcs_model=PRIMARY_MCS_MODEL,
        coherent_mcs_implementation=COHERENT_MCS_IMPLEMENTATION,
        cosmic_mcs_continuation=COSMIC_MCS_CONTINUATION,
        cosmic_joint_inference_method=COSMIC_JOINT_INFERENCE_METHOD,
    )


def _resolved_fixed_parameter_items() -> list[tuple[str, float | None]]:
    return list(
        resolve_fixed_parameter_environment(
            FIXED_PARAMETERS,
            seeding_mode=SEEDING_MODE,
            interaction_mode=INTERACTION_MODE,
            extra_driver_env=EXTRA_DRIVER_ENV,
        ).items()
    )


def _resolved_charge_configuration() -> tuple[
    str,
    str,
    float | None,
    str,
    str | None,
    str | None,
    dict | None,
]:
    """Resolve and bind the WCTE shape-versus-absolute charge contract."""
    shape_mode = str(CHARGE_LIKELIHOOD).strip().lower().replace("-", "_")
    if shape_mode not in _CHARGE_LIKELIHOODS:
        raise ValueError(
            f"CHARGE_LIKELIHOOD must be one of {sorted(_CHARGE_LIKELIHOODS)}"
    )
    if bool(USE_ABSOLUTE_LIGHT_YIELD):
        source = str(ABSOLUTE_LIGHT_YIELD_SOURCE).strip().lower().replace(
            "-", "_"
        )
        if source not in {"measured", "mathematical"}:
            raise ValueError(
                "ABSOLUTE_LIGHT_YIELD_SOURCE must be 'measured' or "
                "'mathematical'"
            )
        manifest_value = str(
            MATHEMATICAL_CHARGE_CALIBRATION_MANIFEST
            if source == "mathematical"
            else GLOBAL_CHARGE_CALIBRATION_MANIFEST
        ).strip()
        if not manifest_value:
            raise ValueError(
                "USE_ABSOLUTE_LIGHT_YIELD=True requires the selected "
                "WCTE GLOBAL_CHARGE_CALIBRATION_MANIFEST or the packaged "
                "mathematical manifest"
            )
        scale, calibration_id, path, manifest_sha256, manifest = (
            resolve_manifest_calibration(
                manifest_value,
                expected_detector="wcte",
                expected_particle=FIT_PARTICLE,
                project_root=_PROJECT_ROOT,
                manual_scale=GLOBAL_CHARGE_SCALE,
                manual_calibration_id=GLOBAL_CHARGE_CALIBRATION_ID,
            )
        )
        basis = str(manifest.get("calibration_basis", "")).strip().lower()
        if source == "mathematical":
            if basis != "ground_up_mathematical":
                raise ValueError(
                    "the mathematical WCTE option requires a "
                    "ground_up_mathematical manifest"
                )
            ground_up = manifest.get("ground_up_light_model", {})
            if ground_up.get("include_wcsim_qe_boosts") is not False:
                raise ValueError(
                    "the WCTE mathematical manifest must use the unboosted "
                    "R14374 QE convention"
                )
        elif basis == "ground_up_mathematical":
            raise ValueError(
                "the measured option cannot select a mathematical manifest"
            )
        context = manifest.get("wcte_calibration_context")
        if not isinstance(context, dict):
            raise ValueError(
                "the WCTE absolute-light manifest lacks "
                "wcte_calibration_context"
            )
        relative_context = context.get("relative_efficiency")
        if not isinstance(relative_context, dict):
            raise ValueError(
                "the WCTE absolute-light manifest lacks its mPMT "
                "relative-efficiency context"
            )
        active_rel_eff = str(RELATIVE_EFFICIENCY_MODE).strip().lower().replace(
            "-", "_"
        )
        if source == "mathematical":
            if context.get("runtime_active_mask_policy") != "per_sensor":
                raise ValueError(
                    "the mathematical WCTE manifest must declare a per-sensor "
                    "runtime active-mask policy"
                )
            allowed = relative_context.get("allowed_runtime_modes", [])
            if active_rel_eff not in set(map(str, allowed)):
                raise ValueError(
                    "the active WCTE relative-efficiency mode is not allowed "
                    "by the mathematical manifest"
                )
        else:
            calibrated_rel_eff = str(relative_context.get("mode", "")).strip().lower().replace(
                "-", "_"
            )
            if calibrated_rel_eff != active_rel_eff:
                raise ValueError(
                    "The measured absolute-light calibration was made with "
                    f"RELATIVE_EFFICIENCY_MODE={calibrated_rel_eff!r}, but the "
                    f"active mode is {active_rel_eff!r}"
                )
            if not math.isclose(
                float(context.get("charge_adc_per_pe", math.nan)),
                float(CHARGE_ADC_PER_PE),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "The measured absolute-light calibration CHARGE_ADC_PER_PE "
                    "does not match the active WCTE conversion"
                )
            if str(context.get("geometry_placement", "")) != str(GEOMETRY_PLACEMENT):
                raise ValueError(
                    "The measured absolute-light calibration geometry placement "
                    "does not match the active WCTE geometry"
                )
        response = manifest.get("pmt_charge_response", {})
        if response.get("model") != str(SPE_RESPONSE_MODEL):
            raise ValueError(
                "The WCTE absolute-light calibration PMT charge response does "
                f"not match SPE_RESPONSE_MODEL={SPE_RESPONSE_MODEL!r}"
            )
        reflection_in_charge = manifest.get("model_contract", {}).get(
            "analytic_reflection_in_charge"
        )
        if not isinstance(reflection_in_charge, bool):
            raise ValueError(
                "the WCTE absolute-light manifest must declare boolean "
                "model_contract.analytic_reflection_in_charge"
            )
        reflection_charge_policy = str(
            manifest.get("model_contract", {}).get(
                "analytic_reflection_charge_policy", "unconditional"
            )
        ).strip().lower().replace("-", "_")
        if reflection_charge_policy not in {
            "unconditional",
            "prompt_group_gated",
        }:
            raise ValueError(
                "the WCTE absolute-light reflection charge policy must be "
                "unconditional or prompt_group_gated"
            )
        return (
            "compound_spe_calibrated",
            "global_scale",
            scale,
            calibration_id,
            path,
            manifest_sha256,
            manifest,
        )
    if shape_mode == "compound_spe_calibrated":
        raise ValueError(
            "Set USE_ABSOLUTE_LIGHT_YIELD=True to use compound_spe_calibrated"
        )
    return shape_mode, "event_mean", None, "", None, None, None


def _validate_coherent_numerics() -> None:
    if COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY not in {
        "straight_conditioned", "free"
    }:
        raise ValueError(
            "COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY must be "
            "'straight_conditioned' or 'free'"
        )
    if int(COHERENT_TRACK_CYCLES) < 1:
        raise ValueError("COHERENT_TRACK_CYCLES must be positive")
    if int(COHERENT_ADAPTIVE_MAX_CYCLES) < int(COHERENT_TRACK_CYCLES):
        raise ValueError(
            "COHERENT_ADAPTIVE_MAX_CYCLES must be at least "
            "COHERENT_TRACK_CYCLES"
        )
    if int(COHERENT_ADAPTIVE_STOP_PATIENCE) < 1:
        raise ValueError("COHERENT_ADAPTIVE_STOP_PATIENCE must be positive")
    if int(COHERENT_BASIN_CANDIDATES) < 1:
        raise ValueError("COHERENT_BASIN_CANDIDATES must be positive")
    if int(COHERENT_JOINT_TIMING_RANGE_CYCLES) < 1:
        raise ValueError(
            "COHERENT_JOINT_TIMING_RANGE_CYCLES must be positive"
        )
    if int(COHERENT_JOINT_TIMING_GLOBAL_CYCLES) < 1:
        raise ValueError(
            "COHERENT_JOINT_TIMING_GLOBAL_CYCLES must be positive"
        )
    if int(COHERENT_JOINT_TIMING_LATENT_ITERATIONS) < 0:
        raise ValueError(
            "COHERENT_JOINT_TIMING_LATENT_ITERATIONS must be nonnegative"
        )
    if int(COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS) < 0:
        raise ValueError(
            "COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS must be "
            "nonnegative"
        )
    thresholds = {
        "COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL": (
            COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL
        ),
        "COHERENT_ADAPTIVE_STOP_GAIN_NLL": COHERENT_ADAPTIVE_STOP_GAIN_NLL,
        "COHERENT_BASIN_GATE_NLL": COHERENT_BASIN_GATE_NLL,
        "COHERENT_JOINT_TIMING_RANGE_STEP_MM": (
            COHERENT_JOINT_TIMING_RANGE_STEP_MM
        ),
        "COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM": (
            COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM
        ),
    }
    for name, value in thresholds.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if float(COHERENT_JOINT_TIMING_RANGE_STEP_MM) <= 0.0:
        raise ValueError(
            "COHERENT_JOINT_TIMING_RANGE_STEP_MM must be positive"
        )
    if not (
        0.0 < float(COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM)
        <= float(COHERENT_JOINT_TIMING_RANGE_STEP_MM)
    ):
        raise ValueError(
            "COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM must be positive and "
            "no larger than COHERENT_JOINT_TIMING_RANGE_STEP_MM"
        )


def _require_file(value: str, label: str) -> None:
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise ValueError(f"{label} does not exist or is not a file: {path}")


def _require_analysis_tools_submodule() -> None:
    required = (
        _ANALYSIS_TOOLS_SUBMODULE / "analysis_tools" / "data_loader.py",
        _ANALYSIS_TOOLS_SUBMODULE / "analysis_tools" / "beam_selection.py",
    )
    if not all(path.is_file() for path in required):
        raise ValueError(
            "The analysis_tools submodule is missing or uninitialized at "
            f"{_ANALYSIS_TOOLS_SUBMODULE}. From LicketyFit_official run: "
            "git submodule update --init analysis_tools"
        )


def _require_geometry_submodule() -> None:
    required = (
        _GEOMETRY_SUBMODULE / "Geometry" / "Device.py",
        _GEOMETRY_SUBMODULE / "Geometry" / "WCD.py",
        _GEOMETRY_SUBMODULE / "examples" / "wcte_bldg157.geo",
    )
    if not all(path.is_file() for path in required):
        raise ValueError(
            "The Geometry submodule is missing or uninitialized at "
            f"{_GEOMETRY_SUBMODULE}. From LicketyFit_official run: "
            "git submodule update --init Geometry"
        )


def _validate(*, check_paths: bool) -> None:
    if FIT_PARTICLE not in _FIT_PARTICLES:
        raise ValueError(f"FIT_PARTICLE must be one of {sorted(_FIT_PARTICLES)}")
    if LIKELIHOOD_MODE not in _LIKELIHOODS:
        raise ValueError(f"LIKELIHOOD_MODE must be one of {sorted(_LIKELIHOODS)}")
    _resolved_fixed_parameter_items()
    _resolved_reconstruction_configuration()
    _resolved_charge_configuration()
    if (
        bool(USE_ABSOLUTE_LIGHT_YIELD)
        and "EMITTER_REFLECTION_IN_CHARGE" in EXTRA_DRIVER_ENV
    ):
        raise ValueError(
            "EMITTER_REFLECTION_IN_CHARGE is fixed by the selected WCTE "
            "absolute-light calibration manifest; remove the override from "
            "EXTRA_DRIVER_ENV"
        )
    if (
        bool(USE_ABSOLUTE_LIGHT_YIELD)
        and "EMITTER_REFLECTION_CHARGE_POLICY" in EXTRA_DRIVER_ENV
    ):
        raise ValueError(
            "EMITTER_REFLECTION_CHARGE_POLICY is fixed by the selected WCTE "
            "absolute-light calibration manifest; remove the override from "
            "EXTRA_DRIVER_ENV"
        )
    _validate_coherent_numerics()
    if EVENT_SOURCE not in {"selection", "file"}:
        raise ValueError("EVENT_SOURCE must be 'selection' or 'file'")
    if GOOD_PMT_SOURCE not in {"auto", "file", "run"}:
        raise ValueError("GOOD_PMT_SOURCE must be 'auto', 'file', or 'run'")
    if int(RUN) < 0 or int(NPROC) < 1 or int(N_ROOT_ENTRIES) < 1:
        raise ValueError("RUN must be nonnegative; NPROC and N_ROOT_ENTRIES must be positive")
    result_stall_timeout = float(EVENT_RESULT_STALL_TIMEOUT_SECONDS)
    if not math.isfinite(result_stall_timeout) or result_stall_timeout <= 0.0:
        raise ValueError(
            "EVENT_RESULT_STALL_TIMEOUT_SECONDS must be finite and positive"
        )
    if MAX_EVENTS_TO_FIT is not None and int(MAX_EVENTS_TO_FIT) < 1:
        raise ValueError("MAX_EVENTS_TO_FIT must be positive or None")
    if int(EVENT_START_INDEX) < 0:
        raise ValueError("EVENT_START_INDEX must be nonnegative")
    if SELECTION_MODE not in {"nominal", "custom"}:
        raise ValueError("SELECTION_MODE must be 'nominal' or 'custom'")
    if LIGHT_PARTICLE_PID_MODE not in {"act", "act_tof", "tof"}:
        raise ValueError(
            "LIGHT_PARTICLE_PID_MODE must be 'act', 'act_tof', or 'tof'"
        )
    if TOF_CUT_MODE not in {"auto", "require", "disable"}:
        raise ValueError("TOF_CUT_MODE must be 'auto', 'require', or 'disable'")
    if GEOMETRY_PLACEMENT not in {"est", "design"}:
        raise ValueError("GEOMETRY_PLACEMENT must be 'est' or 'design'")
    if RELATIVE_EFFICIENCY_MODE not in {"slot", "type", "none"}:
        raise ValueError("RELATIVE_EFFICIENCY_MODE must be 'slot', 'type', or 'none'")
    if PROMPT_WINDOW_MODE not in {"peak_relative", "fixed", "none"}:
        raise ValueError("PROMPT_WINDOW_MODE must be 'peak_relative', 'fixed', or 'none'")
    if PROMPT_WINDOW_MODE == "fixed" and (
        PROMPT_TIME_MIN_NS is None or PROMPT_TIME_MAX_NS is None
    ):
        raise ValueError("Fixed prompt mode requires PROMPT_TIME_MIN_NS and PROMPT_TIME_MAX_NS")
    if (
        PROMPT_TIME_MIN_NS is not None and PROMPT_TIME_MAX_NS is not None
        and float(PROMPT_TIME_MAX_NS) <= float(PROMPT_TIME_MIN_NS)
    ):
        raise ValueError("PROMPT_TIME_MAX_NS must exceed PROMPT_TIME_MIN_NS")
    if TIME_REFERENCE_MODE not in {
        "beam_earliest", "beam_corrected_peak", "beam_corrected_local_median",
        "beam_all_median", "none",
    }:
        raise ValueError("Unsupported TIME_REFERENCE_MODE")
    if float(CHARGE_ADC_PER_PE) <= 0.0:
        raise ValueError("CHARGE_ADC_PER_PE must be positive")
    if float(TIME_REFERENCE_BIN_WIDTH_NS) <= 0.0 or float(TIME_REFERENCE_LOCAL_HALF_WIDTH_NS) < 0.0:
        raise ValueError("Time-reference bin width must be positive and local half-width nonnegative")
    if GOOD_PMT_SOURCE == "file" and not str(GOOD_PMT_FILE).strip():
        raise ValueError("GOOD_PMT_SOURCE='file' requires GOOD_PMT_FILE")
    if EVENT_SOURCE == "selection" and SELECTION_MODE == "custom" and not EXTRA_SELECTION_CUTS:
        raise ValueError("Custom selection mode requires EXTRA_SELECTION_CUTS")
    if EVENT_SOURCE == "selection" and PARTICLE_SELECTION_LABEL == "kaon" and SELECTION_MODE != "custom":
        raise ValueError("Kaon selection requires SELECTION_MODE='custom'")
    for cut in EXTRA_SELECTION_CUTS:
        if not isinstance(cut, (tuple, list)) or len(cut) != 3:
            raise ValueError(f"Malformed selection cut: {cut!r}")
    json.dumps([list(cut) for cut in EXTRA_SELECTION_CUTS])
    forbidden_path_overrides = {
        name for name in ("WCTE_ANALYSIS_TOOLS_PATH", "ANALYSIS_TOOLS_PATH")
        if name in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV[name]).strip()
    }
    if forbidden_path_overrides:
        raise ValueError(
            "analysis_tools path overrides are no longer supported; remove "
            + ", ".join(sorted(forbidden_path_overrides))
            + " from EXTRA_DRIVER_ENV and use the repository submodule"
        )
    if "GEOMETRY_PATH" in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV["GEOMETRY_PATH"]).strip():
        raise ValueError(
            "GEOMETRY_PATH overrides are no longer supported; remove "
            "GEOMETRY_PATH from EXTRA_DRIVER_ENV and use the Geometry submodule"
        )
    forbidden_model_overrides = _REMOVED_MODEL_ENV_SWITCHES.intersection(
        map(str, EXTRA_DRIVER_ENV)
    )
    if forbidden_model_overrides:
        raise ValueError(
            "Removed reconstruction-model environment switches are not "
            "supported; use ENABLE_MCS, PRIMARY_MCS_MODEL, "
            "COHERENT_MCS_IMPLEMENTATION, and COSMIC_MCS_CONTINUATION at the "
            "top of this launcher, and remove "
            + ", ".join(sorted(forbidden_model_overrides))
            + " from EXTRA_DRIVER_ENV"
        )
    duplicate_reconstruction_overrides = (
        _DEDICATED_RECONSTRUCTION_ENV.intersection(map(str, EXTRA_DRIVER_ENV))
    )
    if duplicate_reconstruction_overrides:
        raise ValueError(
            "Use the dedicated reconstruction settings at the top of this file "
            "instead of duplicating them in EXTRA_DRIVER_ENV: "
            + ", ".join(sorted(duplicate_reconstruction_overrides))
        )
    if check_paths:
        _require_geometry_submodule()
        if EVENT_SOURCE == "file":
            _require_file(USER_EVENT_FILE, "USER_EVENT_FILE")
        if EVENT_SOURCE == "selection" and str(COLLABORATION_ROOT_FILE).strip():
            _require_file(COLLABORATION_ROOT_FILE, "COLLABORATION_ROOT_FILE")
        use_user_mask = GOOD_PMT_SOURCE == "file" or (
            GOOD_PMT_SOURCE == "auto" and bool(str(GOOD_PMT_FILE).strip())
        )
        if EVENT_SOURCE == "selection" or not use_user_mask:
            _require_analysis_tools_submodule()
        if use_user_mask:
            _require_file(GOOD_PMT_FILE, "GOOD_PMT_FILE")
        if not use_user_mask and str(GOOD_PMT_ROOT_FILE).strip():
            _require_file(GOOD_PMT_ROOT_FILE, "GOOD_PMT_ROOT_FILE")
        if str(GEOMETRY_FILE).strip():
            _require_file(GEOMETRY_FILE, "GEOMETRY_FILE")
        _require_file(str(_DRIVER), "batch_fit_driver.py")


def _encode(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value)
    return str(value)


def _configuration_items() -> list[tuple[str, object]]:
    _validate_coherent_numerics()
    reconstruction = _resolved_reconstruction_configuration()
    duplicate_reconstruction_overrides = (
        _DEDICATED_RECONSTRUCTION_ENV.intersection(map(str, EXTRA_DRIVER_ENV))
    )
    if duplicate_reconstruction_overrides:
        raise ValueError(
            "Use the dedicated reconstruction settings at the top of this file "
            "instead of duplicating them in EXTRA_DRIVER_ENV: "
            + ", ".join(sorted(duplicate_reconstruction_overrides))
        )
    forbidden_path_overrides = {
        name for name in ("WCTE_ANALYSIS_TOOLS_PATH", "ANALYSIS_TOOLS_PATH")
        if name in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV[name]).strip()
    }
    if forbidden_path_overrides:
        raise ValueError(
            "analysis_tools path overrides are no longer supported; remove "
            + ", ".join(sorted(forbidden_path_overrides))
            + " from EXTRA_DRIVER_ENV"
        )
    if "GEOMETRY_PATH" in EXTRA_DRIVER_ENV and str(EXTRA_DRIVER_ENV["GEOMETRY_PATH"]).strip():
        raise ValueError(
            "GEOMETRY_PATH overrides are no longer supported; remove "
            "GEOMETRY_PATH from EXTRA_DRIVER_ENV"
        )
    root_file = str(COLLABORATION_ROOT_FILE).strip() or None
    geometry_file = str(GEOMETRY_FILE).strip() or None
    output_file = str(OUTPUT_FILE).strip() or None
    (
        charge_likelihood,
        normalization_mode,
        global_scale,
        calibration_id,
        calibration_manifest,
        calibration_manifest_sha256,
        calibration_payload,
    ) = _resolved_charge_configuration()
    calibration_context = (
        calibration_payload.get("wcte_calibration_context", {})
        if calibration_payload is not None
        else {}
    )
    mathematical_calibration = bool(
        calibration_payload is not None
        and calibration_payload.get("calibration_basis")
        == "ground_up_mathematical"
    )
    calibration_rel_eff = (
        {"mode": str(RELATIVE_EFFICIENCY_MODE)}
        if mathematical_calibration
        else calibration_context.get("relative_efficiency", {})
    )
    calibration_adc_per_pe = (
        float(CHARGE_ADC_PER_PE)
        if mathematical_calibration
        else calibration_context.get("charge_adc_per_pe")
    )
    calibration_geometry_placement = (
        str(GEOMETRY_PLACEMENT)
        if mathematical_calibration
        else calibration_context.get("geometry_placement")
    )
    calibrated_reflection_in_charge = (
        calibration_payload.get("model_contract", {}).get(
            "analytic_reflection_in_charge"
        )
        if calibration_payload is not None
        else None
    )
    calibrated_reflection_charge_policy = (
        str(
            calibration_payload.get("model_contract", {}).get(
                "analytic_reflection_charge_policy", "unconditional"
            )
        ).strip().lower().replace("-", "_")
        if calibration_payload is not None
        else None
    )
    config_file = Path(__file__).resolve()
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    items = [
        ("LF_DATA_SOURCE", "wcte"), ("DATA_SOURCE", "wcte"),
        ("FIT_MODE", None), ("FIT_PARTICLE", FIT_PARTICLE),
        ("LIKELIHOOD_MODE", LIKELIHOOD_MODE), ("NPROC", NPROC),
        ("EMITTER_ENABLE_DELTA_E", ENABLE_DELTA_ELECTRONS),
        ("EMITTER_ENABLE_MCS", ENABLE_MCS),
        ("EMITTER_ENABLE_REFLECTION", ENABLE_REFLECTION),
        ("EMITTER_ENABLE_RAYLEIGH", ENABLE_PHOTON_SCATTERING),
        (
            "COSMIC_MULTILATERATION_SEED_MODE",
            reconstruction.navigation_mode,
        ),
        (
            "WCTE_INCLUDE_ORIENTATION_GUARD",
            True if reconstruction.seeding_mode == "general"
            else BEAM_USE_GLOBAL_SEED_GUARD,
        ),
        (
            "WCTE_INCLUDE_DETECTOR_GLOBAL",
            True if reconstruction.seeding_mode == "general"
            else BEAM_USE_GLOBAL_SEED_GUARD,
        ),
        *reconstruction.environment().items(),
        ("PMT_CHARGE_LIKELIHOOD", charge_likelihood),
        ("PMT_SPE_RESPONSE_MODEL", SPE_RESPONSE_MODEL),
        ("PMT_COMPOUND_PROFILE_MAX_ITER", COMPOUND_PROFILE_MAX_ITER),
        ("PMT_COMPOUND_PROFILE_TOL", COMPOUND_PROFILE_TOL),
        ("PMT_COMPOUND_PROFILE_N_CAP", COMPOUND_PROFILE_N_CAP),
        ("WCTE_USE_ABSOLUTE_LIGHT_YIELD", USE_ABSOLUTE_LIGHT_YIELD),
        (
            "WCTE_ABSOLUTE_LIGHT_YIELD_SOURCE",
            ABSOLUTE_LIGHT_YIELD_SOURCE if USE_ABSOLUTE_LIGHT_YIELD else "shape_only",
        ),
        ("WCTE_GLOBAL_CHARGE_CALIBRATION_ID", calibration_id),
        ("WCTE_GLOBAL_CHARGE_CALIBRATION_MANIFEST", calibration_manifest),
        (
            "WCTE_GLOBAL_CHARGE_CALIBRATION_MANIFEST_SHA256",
            calibration_manifest_sha256,
        ),
        (
            "WCTE_GLOBAL_CHARGE_CALIBRATION_REL_EFF_MODE",
            calibration_rel_eff.get("mode"),
        ),
        (
            "WCTE_GLOBAL_CHARGE_CALIBRATION_ADC_PER_PE",
            calibration_adc_per_pe,
        ),
        (
            "WCTE_GLOBAL_CHARGE_CALIBRATION_GEOMETRY_PLACEMENT",
            calibration_geometry_placement,
        ),
        (
            "EMITTER_REFLECTION_IN_CHARGE",
            calibrated_reflection_in_charge,
        ),
        (
            "EMITTER_REFLECTION_CHARGE_POLICY",
            calibrated_reflection_charge_policy,
        ),
        ("LF_OUTPUT_FILE", output_file),
        ("EVENT_SOURCE", EVENT_SOURCE), ("RUN", RUN),
        ("CONFIG_ROOT_FILE", root_file), ("USER_EVENT_FILE", USER_EVENT_FILE),
        ("USER_EVENT_KEY", USER_EVENT_KEY), ("N_ROOT_ENTRIES", N_ROOT_ENTRIES),
        ("N_EVENTS", N_ROOT_ENTRIES), ("MAX_EVENTS_TO_FIT", MAX_EVENTS_TO_FIT),
        ("TOT_EVENTS", MAX_EVENTS_TO_FIT), ("LF_EVENT_START_INDEX", EVENT_START_INDEX),
        ("WCTE_GOOD_PMT_SOURCE", GOOD_PMT_SOURCE),
        ("WCTE_GOOD_PMT_FILE", GOOD_PMT_FILE),
        ("WCTE_GOOD_PMT_FILE_KEY", GOOD_PMT_FILE_KEY),
        ("WCTE_GOOD_PMT_ROOT_FILE", GOOD_PMT_ROOT_FILE),
        ("WCTE_GOOD_PMT_ROOT_SEARCH_BASES", os.pathsep.join(map(str, GOOD_PMT_ROOT_SEARCH_BASES))),
        ("PARTICLE_SELECTION_LABEL", PARTICLE_SELECTION_LABEL),
        ("WCTE_SELECTION_MODE", SELECTION_MODE),
        ("WCTE_EXTRA_SELECTION_CUTS_JSON", json.dumps([list(cut) for cut in EXTRA_SELECTION_CUTS])),
        ("WCTE_APPLY_MPMT_DATA_QUALITY_CUTS", APPLY_MPMT_DATA_QUALITY_CUTS),
        ("WCTE_APPLY_VME_EVENT_QUALITY_CUTS", APPLY_VME_EVENT_QUALITY_CUTS),
        ("WCTE_APPLY_T5_EVENT_QUALITY_CUTS", APPLY_T5_EVENT_QUALITY_CUTS),
        ("WCTE_USE_ACT_EVETO_CUT", USE_ACT_EVETO_CUT),
        ("WCTE_USE_ACT_TAGGER_CUT", USE_ACT_TAGGER_CUT),
        ("WCTE_LIGHT_PARTICLE_PID_MODE", LIGHT_PARTICLE_PID_MODE),
        ("WCTE_TOF_CUT_MODE", TOF_CUT_MODE),
        ("WCTE_PROTON_TOF_WINDOW_NS", PROTON_TOF_WINDOW_NS),
        ("WCTE_REQUIRE_MUON_TAGGER", REQUIRE_MUON_TAGGER),
        ("WCTE_ACT_EVETO_CUT_OVERRIDE_PE", ACT_EVETO_CUT_OVERRIDE_PE),
        ("WCTE_ACT_TAGGER_CUT_OVERRIDE_PE", ACT_TAGGER_CUT_OVERRIDE_PE),
        ("WCTE_PROTON_TOF_CUT_OVERRIDE_NS", PROTON_TOF_CUT_OVERRIDE_NS),
        ("WCTE_MUON_TAG_CUT_OVERRIDE", MUON_TAG_CUT_OVERRIDE),
        (
            "WCTE_ELECTRON_MUON_TOF_BOUNDARY_OVERRIDE_NS",
            ELECTRON_MUON_TOF_BOUNDARY_OVERRIDE_NS,
        ),
        (
            "WCTE_MUON_PION_TOF_BOUNDARY_OVERRIDE_NS",
            MUON_PION_TOF_BOUNDARY_OVERRIDE_NS,
        ),
        ("BEAM_P", BEAM_MOMENTUM_MEV_C),
        ("WCTE_EXPECTED_KE_MEV", EXPECTED_KINETIC_ENERGY_MEV),
        ("WCTE_USE_EXPECTED_ENERGY_HINT", USE_EXPECTED_ENERGY_SEED_HINT),
        ("REL_EFF_MODE", RELATIVE_EFFICIENCY_MODE),
        ("WCTE_PLACEMENT_KEY", GEOMETRY_PLACEMENT),
        ("CHARGE_ADC_PER_PE", CHARGE_ADC_PER_PE),
        ("WCTE_CHARGE_NORMALIZATION_MODE", normalization_mode),
        ("WCTE_GLOBAL_CHARGE_SCALE", global_scale),
        ("COSMIC_COHERENT_TRACK_CYCLES", COHERENT_TRACK_CYCLES),
        (
            "COSMIC_COHERENT_ADAPTIVE_MAX_CYCLES",
            COHERENT_ADAPTIVE_MAX_CYCLES,
        ),
        (
            "COSMIC_COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL",
            COHERENT_ADAPTIVE_TRIGGER_GAIN_NLL,
        ),
        (
            "COSMIC_COHERENT_ADAPTIVE_STOP_GAIN_NLL",
            COHERENT_ADAPTIVE_STOP_GAIN_NLL,
        ),
        (
            "COSMIC_COHERENT_ADAPTIVE_STOP_PATIENCE",
            COHERENT_ADAPTIVE_STOP_PATIENCE,
        ),
        ("COSMIC_COHERENT_BASIN_CANDIDATES", COHERENT_BASIN_CANDIDATES),
        ("COSMIC_COHERENT_BASIN_GATE_NLL", COHERENT_BASIN_GATE_NLL),
        (
            "COSMIC_COHERENT_JOINT_TIMING_RANGE_STEP_MM",
            COHERENT_JOINT_TIMING_RANGE_STEP_MM,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM",
            COHERENT_JOINT_TIMING_MIN_RANGE_STEP_MM,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_RANGE_CYCLES",
            COHERENT_JOINT_TIMING_RANGE_CYCLES,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_GLOBAL_CYCLES",
            COHERENT_JOINT_TIMING_GLOBAL_CYCLES,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY",
            COHERENT_JOINT_TIMING_GLOBAL_GEOMETRY_POLICY,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_LATENT_ITERATIONS",
            COHERENT_JOINT_TIMING_LATENT_ITERATIONS,
        ),
        (
            "COSMIC_COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS",
            COHERENT_JOINT_TIMING_CANDIDATE_LATENT_ITERATIONS,
        ),
        ("WCTE_PROMPT_WINDOW_MODE", PROMPT_WINDOW_MODE),
        ("WCTE_PROMPT_TIME_MIN_NS", PROMPT_TIME_MIN_NS),
        ("WCTE_PROMPT_TIME_MAX_NS", PROMPT_TIME_MAX_NS),
        ("USER_EVENT_APPLY_PROMPT_WINDOW", USER_EVENT_APPLY_PROMPT_WINDOW),
        ("WCTE_STRICT_USER_EVENT_VALIDATION", STRICT_USER_EVENT_VALIDATION),
        ("WCTE_TIME_REFERENCE_MODE", TIME_REFERENCE_MODE),
        ("WCTE_TIME_REFERENCE_BIN_WIDTH_NS", TIME_REFERENCE_BIN_WIDTH_NS),
        ("WCTE_TIME_REFERENCE_LOCAL_HALF_WIDTH_NS", TIME_REFERENCE_LOCAL_HALF_WIDTH_NS),
        ("WCD_GEOMETRY_FILE", geometry_file), ("WCTE_GEOMETRY_FILE", geometry_file),
        ("GEOMETRY_PATH", None),
        ("LF_GEOMETRY_POLICY", "required_repository_submodule"),
        ("WCTE_ANALYSIS_TOOLS_PATH", None),
        ("ANALYSIS_TOOLS_PATH", None),
        ("LF_ANALYSIS_TOOLS_POLICY", "required_repository_submodule"),
        ("SELECTION_STEP_SIZE", SELECTION_STEP_SIZE),
        ("N_EVENTS_PER_BATCH", N_EVENTS_PER_BATCH),
        ("WARM_FIT_KERNELS", WARM_FIT_KERNELS),
        (
            "LF_EVENT_RESULT_STALL_TIMEOUT_SECONDS",
            EVENT_RESULT_STALL_TIMEOUT_SECONDS,
        ),
        ("SAVE_AFTER_EACH_BATCH", SAVE_AFTER_EACH_BATCH),
        ("SAVE_DETAILED_EVENT_RESULTS", SAVE_DETAILED_EVENT_RESULTS),
        ("MCS_RECORD_EVENT_FAILURES", CONTINUE_AFTER_EVENT_FAILURE),
        ("MCS_RETAIN_STRAIGHT_ON_FAILURE", RETAIN_STRAIGHT_ON_MCS_FAILURE),
        ("PRINT_EVENT_RESULTS", PRINT_EVENT_RESULTS),
        ("PRINT_BATCH_PROGRESS", PRINT_BATCH_PROGRESS),
        ("PRINT_CHECKPOINT_MESSAGES", PRINT_CHECKPOINT_MESSAGES),
        ("VERBOSE_SETUP", VERBOSE_SETUP),
        ("WCTE_PRINT_SELECTION_DESCRIPTION", PRINT_SELECTION_DESCRIPTION),
        ("WCTE_PRINT_CHERENKOV_THRESHOLDS", PRINT_CHERENKOV_THRESHOLDS),
        ("WCSIM_USE_TRUTH_ROOT", False),
        ("ALLOW_MISSING_GOOD_PMTS", False),
        ("LF_RUN_CONFIG_KIND", "wcte"),
        ("LF_RUN_CONFIG_FILE", str(config_file)),
        ("LF_RUN_CONFIG_SHA256", config_hash),
        ("LF_PUBLIC_DRIVER_RELEASE", _PUBLIC_DRIVER_RELEASE),
    ]
    if ENABLE_EXACT_EXECUTION_ACCELERATORS:
        items.extend(
            (str(name), value)
            for name, value in EXACT_EXECUTION_ENV.items()
        )
    if ENABLE_VALIDATED_COSMIC_LATENCY_PROFILE:
        items.extend(
            (str(name), value)
            for name, value in VALIDATED_COSMIC_LATENCY_ENV.items()
        )
    items.extend(_resolved_fixed_parameter_items())
    items.extend((str(name), value) for name, value in EXTRA_DRIVER_ENV.items())
    return items


def build_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return the exact environment handed to batch_fit_driver.py."""
    env = dict(os.environ if base is None else base)
    # This calibration converts the fitter's CSDA coordinate to a WCSim-only
    # reporting convention.  Never allow an inherited shell setting to enter
    # a real-data WCTE run.
    env.pop("WCSIM_VISIBLE_RANGE_CONVENTION_MANIFEST", None)
    for internal in (
        "LF_COSMIC_SUPERVISED_CHILD", "LF_COSMIC_CHILD_QUIET",
        "LF_WCTE_INTERNAL_PREPARED_EVENT_FILE", "LF_WCTE_PREPARE_EVENTS_ONLY",
        "LF_WCTE_PREPARED_EVENT_FILE", "LF_EVENT_COUNT",
    ):
        env.pop(internal, None)
    for name in _REMOVED_MODEL_ENV_SWITCHES:
        env.pop(name, None)
    for name, value in _configuration_items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = _encode(value)
    if _RUNTIME_DEPS.is_dir():
        inherited = env.get("PYTHONPATH", "").strip()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_RUNTIME_DEPS)] + ([inherited] if inherited else [])
        )
    return env


def _print_configuration() -> None:
    print(f"Launcher: {Path(__file__).resolve()}")
    print(f"Driver:   {_DRIVER}")
    print(f"analysis_tools submodule: {_ANALYSIS_TOOLS_SUBMODULE}")
    print(f"Geometry submodule:       {_GEOMETRY_SUBMODULE}")
    for name, value in _configuration_items():
        shown = "<unset>" if value is None else _encode(value)
        print(f"{name}={shown}")


def _resolved_output_file() -> str:
    """Return the exact explicit/default output path used by the driver."""
    requested = str(OUTPUT_FILE).strip()
    if requested:
        return str(Path(requested).expanduser())
    reconstruction = _resolved_reconstruction_configuration()
    return str(
        _PROJECT_ROOT
        / "outputs"
        / (
            f"estimates_run{int(RUN)}_{float(BEAM_MOMENTUM_MEV_C):g}p_"
            f"{FIT_PARTICLE}_{reconstruction.public_mode_label}_"
            f"{LIKELIHOOD_MODE}_relEff-{RELATIVE_EFFICIENCY_MODE}.dict"
        )
    )


def _event_count_summary() -> str:
    if MAX_EVENTS_TO_FIT is None:
        limit = "all selected"
    else:
        limit = f"up to {int(MAX_EVENTS_TO_FIT)} selected"
    return (
        f"{limit} from index {int(EVENT_START_INDEX)} "
        f"(scan limit {int(N_ROOT_ENTRIES)})"
    )


def _print_run_summary(output_file: str) -> None:
    print_welcome(SEEDING_MODE, INTERACTION_MODE)
    details = [
        ("Run", int(RUN)),
        ("Particle", FIT_PARTICLE),
    ]
    if str(PARTICLE_SELECTION_LABEL) != str(FIT_PARTICLE):
        details.append(("Selected sample", PARTICLE_SELECTION_LABEL))
    if (
        EVENT_SOURCE == "selection"
        and SELECTION_MODE == "nominal"
        and str(PARTICLE_SELECTION_LABEL).strip().lower()
        in {
            "electron", "e", "e-", "e+",
            "muon", "mu", "mu-", "mu+",
            "pion", "pi", "pi-", "pi+",
        }
    ):
        details.append(("Beam PID", LIGHT_PARTICLE_PID_MODE))
    details.extend(
        (
            ("Likelihood", LIKELIHOOD_MODE),
            ("Events", _event_count_summary()),
            ("Workers", int(NPROC)),
            ("Output", output_file),
        )
    )
    print_details(details)
    print_preparation_notice()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure and run LicketyFit on real WCTE data."
    )
    parser.add_argument(
        "--show-config", action="store_true",
        help="print the driver environment generated by this file and exit",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="validate choices and required local paths without running a fit",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="show the legacy detailed setup, timing, and checkpoint diagnostics",
    )
    args = parser.parse_args()
    try:
        _validate_runtime_dependencies()
        _validate(check_paths=bool(args.check or not args.show_config))
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.show_config:
        _print_configuration()
        return
    if args.check:
        print("WCTE run configuration is valid.")
        return
    environment = build_environment()
    output_file = _resolved_output_file()
    environment["LF_OUTPUT_FILE"] = output_file
    environment["LF_SIMPLE_CONSOLE"] = "0" if args.verbose else "1"
    environment["LF_PROGRESS_INTERVAL"] = "50"
    if args.verbose:
        os.execve(sys.executable, [sys.executable, str(_DRIVER_ENTRY)], environment)
    _print_run_summary(output_file)
    return_code = run_with_simple_console(
        [sys.executable, str(_DRIVER_ENTRY)],
        environment=environment,
    )
    if return_code != 0:
        raise SystemExit(return_code)
    print_goodbye(output_file)


if __name__ == "__main__":
    main()
