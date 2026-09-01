"""Immutable absolute-light calibration manifests and runtime checks.

Absolute charge is a detector-response calibration, not an eventwise nuisance
parameter.  This module binds a frozen scale to the forward-model sources that
produced it.  Real-WCTE manifests additionally bind the scale to the active
channel mask, ADC-to-PE conversion, geometry placement, and the *contents* of
the selected mPMT relative-efficiency tables.

The WCSim and WCTE scales are intentionally detector-specific.  Their emission
physics can be shared, but a WCSim optical-response constant is never accepted
as a real-data WCTE calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CALIBRATION_SCHEMA = "licketyfit-absolute-light-calibration-v2"
SUPPORTED_DETECTORS = {"wcsim", "wcte"}


def _resolve_manifest_path(
    path: str | Path, *, project_root: str | Path | None
) -> Path:
    """Resolve packaged relative manifests from the repository root.

    The public launchers are commonly executed from ``scripts/``.  A relative
    manifest in their configuration is a package resource, so resolving it
    from the process working directory would incorrectly search
    ``scripts/tables``.  Explicit absolute paths retain their normal meaning.
    """

    manifest_path = Path(path).expanduser()
    if not manifest_path.is_absolute() and project_root is not None:
        manifest_path = Path(project_root).expanduser().resolve() / manifest_path
    return manifest_path.resolve()


def file_sha256(path: str | Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of one file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(int(block_bytes)):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: Sequence[int] | np.ndarray) -> str:
    """Hash an active-channel vector using the production driver's convention."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    if array.ndim != 1:
        raise ValueError("active PMT IDs must be a one-dimensional vector")
    return hashlib.sha256(array.tobytes()).hexdigest()


def canonical_calibration_id(manifest: Mapping[str, Any]) -> str:
    """Return the content-addressed identifier for a manifest."""
    core = dict(manifest)
    core.pop("calibration_id", None)
    encoded = json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _nonempty_text(value: Any, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"absolute-light calibration {label} is empty")
    return text


def _positive_scale(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("absolute-light calibration scale must be numeric")
    try:
        scale = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("absolute-light calibration scale must be numeric") from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("absolute-light calibration scale must be finite and positive")
    return scale


def load_calibration_manifest(
    path: str | Path,
    *,
    expected_detector: str | None = None,
    expected_particle: str | None = None,
    project_root: str | Path | None = None,
    verify_model_files: bool = True,
) -> dict[str, Any]:
    """Load and validate an immutable absolute-light calibration manifest."""
    manifest_path = _resolve_manifest_path(path, project_root=project_root)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read absolute-light calibration manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("absolute-light calibration manifest must be a JSON object")
    if manifest.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(
            f"absolute-light calibration manifest must use schema {CALIBRATION_SCHEMA!r}"
        )
    detector = _nonempty_text(manifest.get("detector", ""), label="detector").lower()
    if detector not in SUPPORTED_DETECTORS:
        raise ValueError(f"unsupported absolute-light detector {detector!r}")
    if expected_detector is not None and detector != str(expected_detector).lower():
        raise ValueError(
            f"cannot use a {detector} absolute-light calibration in "
            f"{str(expected_detector).lower()} mode"
        )
    particle = _nonempty_text(manifest.get("particle", ""), label="particle").lower()
    if expected_particle is not None and particle != str(expected_particle).lower():
        raise ValueError(
            f"absolute-light calibration particle {particle!r} does not match "
            f"the active hypothesis {str(expected_particle).lower()!r}"
        )
    _positive_scale(manifest.get("global_charge_scale"))
    if manifest.get("charge_likelihood") != "compound_spe_calibrated":
        raise ValueError(
            "absolute-light calibration requires charge_likelihood="
            "'compound_spe_calibrated'"
        )
    if manifest.get("charge_normalization") != "global_scale":
        raise ValueError(
            "absolute-light calibration requires charge_normalization='global_scale'"
        )
    response = manifest.get("pmt_charge_response")
    if not isinstance(response, dict):
        raise ValueError(
            "absolute-light calibration requires pmt_charge_response provenance"
        )
    _nonempty_text(response.get("model", ""), label="PMT charge-response model")
    basis = str(manifest.get("calibration_basis", "")).strip().lower()
    if basis == "ground_up_mathematical":
        ground_up = manifest.get("ground_up_light_model")
        if not isinstance(ground_up, dict):
            raise ValueError(
                "mathematical absolute-light calibration requires "
                "ground_up_light_model provenance"
            )
        if ground_up.get("calibration_data_used") is not False:
            raise ValueError(
                "mathematical absolute-light calibration must declare that "
                "no calibration data were used"
            )
        for field in (
            "all_charged_detected_pe",
            "prompt_detected_pe",
            "licketyfit_scale_one_pe",
            "relative_systematic_uncertainty",
        ):
            _positive_scale(ground_up.get(field))
        include_boosts = ground_up.get("include_wcsim_qe_boosts")
        if not isinstance(include_boosts, bool):
            raise ValueError(
                "mathematical absolute-light calibration must declare its "
                "WCSim QE-boost policy"
            )
    recorded_id = _nonempty_text(
        manifest.get("calibration_id", ""), label="calibration_id"
    )
    computed_id = canonical_calibration_id(manifest)
    if recorded_id != computed_id:
        raise ValueError(
            "absolute-light calibration content does not match calibration_id "
            f"({recorded_id!r} != {computed_id!r})"
        )

    hashes = manifest.get("model_files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("absolute-light calibration requires model_files_sha256")
    provenance_hashes = manifest.get(
        "calibration_provenance_files_sha256", {}
    )
    if not isinstance(provenance_hashes, dict):
        raise ValueError(
            "absolute-light calibration provenance hashes must be a mapping"
        )
    for relative, digest in sorted(provenance_hashes.items()):
        relative_path = Path(
            _nonempty_text(relative, label="calibration provenance path")
        )
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                "unsafe absolute-light calibration provenance path "
                f"{str(relative_path)!r}"
            )
        digest_text = str(digest).strip().lower()
        if len(digest_text) != 64 or any(
            character not in "0123456789abcdef" for character in digest_text
        ):
            raise ValueError(
                "absolute-light calibration provenance contains an invalid "
                f"SHA-256 for {relative_path}"
            )
    if verify_model_files:
        if project_root is None:
            raise ValueError("project_root is required when verifying model files")
        root = Path(project_root).expanduser().resolve()
        for relative, expected_hash in sorted(hashes.items()):
            relative_path = Path(_nonempty_text(relative, label="model file path"))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"unsafe absolute-light model file path {str(relative_path)!r}"
                )
            model_path = root / relative_path
            if not model_path.is_file():
                raise ValueError(
                    f"absolute-light model file is missing: {model_path}"
                )
            actual_hash = file_sha256(model_path)
            if actual_hash != str(expected_hash):
                raise ValueError(
                    "absolute-light calibration is stale for model file "
                    f"{relative_path}: {actual_hash} != {expected_hash}"
                )

        # ``calibration_provenance_files_sha256`` records how the immutable
        # scale was produced.  Those offline analysis programs are deliberately
        # not opened here: reconstruction depends only on the runtime model
        # files above and must work in a minimal deployment without analysis/.

    return manifest


def validate_pmt_charge_response_context(
    manifest: Mapping[str, Any],
    *,
    model: str,
    single_pe_amp_mean: float,
    single_pe_amp_std: float,
    amp_threshold_pe: float,
) -> None:
    """Bind a frozen light scale to the runtime digit-charge response."""
    response = manifest.get("pmt_charge_response")
    if not isinstance(response, dict):
        raise ValueError(
            "absolute-light calibration lacks pmt_charge_response provenance"
        )
    expected_model = _nonempty_text(
        response.get("model", ""), label="PMT charge-response model"
    ).lower().replace("-", "_")
    active_model = str(model).strip().lower().replace("-", "_")
    if expected_model != active_model:
        raise ValueError(
            "absolute-light calibration PMT charge-response model does not "
            f"match runtime: {expected_model!r} != {active_model!r}"
        )
    if active_model == "gaussian_censored":
        checks = (
            ("single_pe_amp_mean", single_pe_amp_mean),
            ("single_pe_amp_std", single_pe_amp_std),
            ("amp_threshold_pe", amp_threshold_pe),
        )
        for field, active in checks:
            expected = _positive_scale(response.get(field))
            if not math.isclose(
                expected, float(active), rel_tol=0.0, abs_tol=1.0e-12
            ):
                raise ValueError(
                    "absolute-light calibration PMT response does not match "
                    f"runtime {field}"
                )
    elif active_model == "wcsim_r14374_ski":
        from LicketyFit.wcsim_charge_response import response_metadata

        authoritative = response_metadata()
        for field in (
            "implementation_version",
            "wcsim_source_commit",
            "qpe_cdf_entries",
            "qpe_mean",
            "qpe_std",
            "qpe_skewness",
            "qpe_excess_kurtosis",
            "digitizer_noise_sigma_pe",
            "digitizer_charge_factor",
        ):
            expected = response.get(field)
            active = authoritative[field]
            if isinstance(active, float):
                matches = isinstance(expected, (int, float)) and math.isclose(
                    float(expected), active, rel_tol=0.0, abs_tol=1.0e-14
                )
            else:
                matches = expected == active
            if not matches:
                raise ValueError(
                    "absolute-light calibration WCSim PMT response is stale "
                    f"for {field}: {expected!r} != {active!r}"
                )
    else:
        raise ValueError(
            f"unsupported absolute-light PMT charge-response model {active_model!r}"
        )


def resolve_manifest_calibration(
    manifest_path: str | Path,
    *,
    expected_detector: str,
    expected_particle: str,
    project_root: str | Path,
    manual_scale: float | None = None,
    manual_calibration_id: str = "",
    verify_model_files: bool = True,
) -> tuple[float, str, str, str, dict[str, Any]]:
    """Resolve a manifest and reject conflicting duplicated scalar settings."""
    path = _resolve_manifest_path(manifest_path, project_root=project_root)
    manifest = load_calibration_manifest(
        path,
        expected_detector=expected_detector,
        expected_particle=expected_particle,
        project_root=project_root,
        verify_model_files=verify_model_files,
    )
    scale = _positive_scale(manifest["global_charge_scale"])
    calibration_id = str(manifest["calibration_id"])
    if manual_scale is not None and not math.isclose(
        float(manual_scale), scale, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise ValueError(
            "GLOBAL_CHARGE_SCALE conflicts with the selected calibration manifest"
        )
    manual_id = str(manual_calibration_id).strip()
    if manual_id and manual_id != calibration_id:
        raise ValueError(
            "GLOBAL_CHARGE_CALIBRATION_ID conflicts with the selected calibration manifest"
        )
    return scale, calibration_id, str(path), file_sha256(path), manifest


def validate_wcte_runtime_context(
    manifest: Mapping[str, Any],
    *,
    relative_efficiency_metadata: Mapping[str, Any],
    adc_per_pe: float,
    geometry_placement: str,
    geometry_sha256: str | None,
    active_pmt_ids: Sequence[int] | np.ndarray,
) -> None:
    """Bind a WCTE scale to the realized mPMT/geometry/channel response."""
    if str(manifest.get("detector", "")).strip().lower() != "wcte":
        raise ValueError("WCTE runtime validation requires a WCTE calibration")
    context = manifest.get("wcte_calibration_context")
    if not isinstance(context, dict):
        raise ValueError(
            "WCTE absolute-light calibration lacks wcte_calibration_context"
        )
    if str(manifest.get("calibration_basis", "")).strip().lower() == (
        "ground_up_mathematical"
    ):
        # A ground-up scale is a nominal per-sensor optical normalization. The
        # realized good-PMT mask, surveyed/design coordinates, ADC-to-PE
        # conversion, and relative-efficiency vector are applied explicitly by
        # the WCTE runtime; freezing any one run's mask into this manifest would
        # incorrectly turn a physical estimate into a run-specific calibration.
        if context.get("runtime_active_mask_policy") != "per_sensor":
            raise ValueError(
                "mathematical WCTE absolute light requires a per-sensor "
                "runtime active-mask policy"
            )
        active = np.asarray(active_pmt_ids)
        if (
            active.ndim != 1
            or active.size == 0
            or active.dtype.kind not in "iu"
            or np.unique(active).size != active.size
        ):
            raise ValueError(
                "mathematical WCTE absolute light requires a nonempty unique "
                "integer active-PMT vector"
            )
        _positive_scale(adc_per_pe)
        _nonempty_text(geometry_placement, label="runtime geometry placement")
        _nonempty_text(geometry_sha256 or "", label="runtime geometry hash")
        expected_eff = context.get("relative_efficiency")
        if not isinstance(expected_eff, dict):
            raise ValueError(
                "mathematical WCTE absolute light lacks its runtime "
                "relative-efficiency policy"
            )
        allowed_modes = expected_eff.get("allowed_runtime_modes")
        if not isinstance(allowed_modes, list) or not allowed_modes:
            raise ValueError(
                "mathematical WCTE absolute light requires allowed runtime "
                "relative-efficiency modes"
            )
        active_mode = str(relative_efficiency_metadata.get("mode", "")).strip()
        if active_mode not in {str(value) for value in allowed_modes}:
            raise ValueError(
                "mathematical WCTE absolute-light relative-efficiency mode "
                f"{active_mode!r} is not allowed"
            )
        return
    expected_adc = _positive_scale(context.get("charge_adc_per_pe"))
    if not math.isclose(
        expected_adc, float(adc_per_pe), rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "WCTE absolute-light calibration ADC-per-PE conversion does not "
            "match the active conversion"
        )
    if str(context.get("geometry_placement", "")) != str(geometry_placement):
        raise ValueError(
            "WCTE absolute-light calibration geometry placement does not match"
        )
    if str(context.get("geometry_sha256", "")) != str(geometry_sha256 or ""):
        raise ValueError("WCTE absolute-light calibration geometry hash does not match")

    expected_active_hash = _nonempty_text(
        context.get("active_pmt_ids_sha256", ""), label="active PMT hash"
    )
    if array_sha256(active_pmt_ids) != expected_active_hash:
        raise ValueError("WCTE absolute-light calibration active-PMT mask does not match")

    expected_eff = context.get("relative_efficiency")
    if not isinstance(expected_eff, dict):
        raise ValueError(
            "WCTE absolute-light calibration lacks relative-efficiency provenance"
        )
    fields = (
        "mode",
        "slot_table_sha256",
        "slot_stack_sha256",
        "type_info_sha256",
        "type_eff_sha256",
    )
    for field in fields:
        expected = expected_eff.get(field)
        active = relative_efficiency_metadata.get(field)
        if expected != active:
            raise ValueError(
                "WCTE absolute-light calibration relative-efficiency context "
                f"does not match for {field}: {expected!r} != {active!r}"
            )


__all__ = [
    "CALIBRATION_SCHEMA",
    "array_sha256",
    "canonical_calibration_id",
    "file_sha256",
    "load_calibration_manifest",
    "resolve_manifest_calibration",
    "validate_pmt_charge_response_context",
    "validate_wcte_runtime_context",
]
