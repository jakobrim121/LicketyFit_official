"""Build/load the exact native molecular-scattering receiver.

The C++ kernel is an execution backend for the existing analytic receiver. It
uses the same quadrature nodes and equations and is checked against the Numba
reference before being enabled by the production driver.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_SOURCE = _MODULE_DIR / "photon_scattering_native.cpp"
_LIB = None
_FUNC = None
_BUILD_ERROR = None
_BUILD_FLAGS = ("-O3", "-ffast-math", "-march=native", "-fPIC", "-shared", "-fopenmp", "-std=c++17")


def _cache_dir() -> Path:
    raw = os.environ.get("LF_NATIVE_CACHE_DIR", "").strip()
    if raw:
        out = Path(raw).expanduser()
    else:
        numba = os.environ.get("NUMBA_CACHE_DIR", "").strip()
        out = Path(numba).expanduser() / "licketyfit_native" if numba else _MODULE_DIR / ".native_cache"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _compiler() -> str:
    return os.environ.get("CXX", "").strip() or shutil.which("g++") or "g++"


def _cpu_tag() -> str:
    """Return a short CPU-specific tag for the -march=native binary cache."""
    model = platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                model += "_" + line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    return hashlib.sha256(model.encode("utf-8")).hexdigest()[:10]


def _library_path() -> Path:
    digest = hashlib.sha256()
    digest.update(_SOURCE.read_bytes())
    digest.update(" ".join(_BUILD_FLAGS).encode("utf-8"))
    source_hash = digest.hexdigest()[:16]
    abi = (
        f"py{sys.version_info.major}{sys.version_info.minor}_"
        f"{sysconfig.get_platform()}_cpu{_cpu_tag()}"
    )
    return _cache_dir() / f"photon_scattering_native_{source_hash}_{abi}.so"


def ensure_native_receiver_built(*, required: bool = False) -> Path | None:
    """Compile the small shared library once, with an inter-process file lock."""
    global _BUILD_ERROR
    if not _SOURCE.is_file():
        _BUILD_ERROR = FileNotFoundError(f"Native receiver source not found: {_SOURCE}")
        if required:
            raise _BUILD_ERROR
        return None
    target = _library_path()
    if target.is_file():
        return target
    lock_path = target.with_suffix(target.suffix + ".lock")
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target.is_file():
            return target
        tmp = target.with_suffix(f".tmp.{os.getpid()}.so")
        command = [
            _compiler(), *_BUILD_FLAGS, str(_SOURCE), "-o", str(tmp),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.replace(tmp, target)
            _BUILD_ERROR = None
            return target
        except Exception as exc:
            _BUILD_ERROR = exc
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if required:
                raise RuntimeError(
                    "Failed to compile the native photon-scattering receiver. "
                    f"Command: {' '.join(command)}"
                ) from exc
            return None


def _load(required: bool = False):
    global _LIB, _FUNC
    if _FUNC is not None:
        return _FUNC
    path = ensure_native_receiver_built(required=required)
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(str(path))
        func = lib.lf_scatter_fused_selected
        p_double = ctypes.POINTER(ctypes.c_double)
        p_i8 = ctypes.POINTER(ctypes.c_int8)
        func.argtypes = [
            ctypes.c_int, ctypes.c_int,
            p_double, p_double, p_double, p_double,
            p_double, p_double, p_i8, p_double, p_double, p_double,
            ctypes.c_double, ctypes.c_double,
            ctypes.c_int, ctypes.c_double, ctypes.c_double,
            p_double, ctypes.c_int, p_double, ctypes.c_int, ctypes.c_double,
            p_double, p_double, p_double, p_double, p_double, ctypes.c_int,
        ]
        func.restype = ctypes.c_int
        _LIB = lib
        _FUNC = func
        return func
    except Exception as exc:
        if required:
            raise RuntimeError(f"Could not load native receiver library: {path}") from exc
        return None


def native_receiver_available() -> bool:
    return _load(required=False) is not None


def accumulate_fused_selected_native(
    p_selected: np.ndarray,
    n_selected: np.ndarray,
    node_pos: np.ndarray,
    node_pol: np.ndarray,
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    node_channel: np.ndarray,
    node_b: np.ndarray,
    node_ng: np.ndarray,
    node_base_time: np.ndarray,
    aperture_radius: float,
    facing_width: float,
    n_bins: int,
    t_min: float,
    t_max: float,
    response_lut: np.ndarray,
    attenuation_lut: np.ndarray,
    attenuation_xmax: float,
    *,
    n_threads: int = 1,
    required: bool = False,
):
    func = _load(required=required)
    if func is None:
        return None
    p = np.ascontiguousarray(p_selected, dtype=np.float64)
    n = np.ascontiguousarray(n_selected, dtype=np.float64)
    position = np.ascontiguousarray(node_pos, dtype=np.float64)
    polarization = np.ascontiguousarray(node_pol, dtype=np.float64)
    pa = np.ascontiguousarray(phase_a, dtype=np.float64)
    pb = np.ascontiguousarray(phase_b, dtype=np.float64)
    channel = np.ascontiguousarray(node_channel, dtype=np.int8)
    bout = np.ascontiguousarray(node_b, dtype=np.float64)
    ng = np.ascontiguousarray(node_ng, dtype=np.float64)
    bt = np.ascontiguousarray(node_base_time, dtype=np.float64)
    response = np.ascontiguousarray(response_lut, dtype=np.float64)
    attenuation = np.ascontiguousarray(attenuation_lut, dtype=np.float64)

    nsel = int(p.shape[0])
    nnode = int(position.shape[0])
    nbin = max(0, int(n_bins))
    charge = np.zeros(nsel, dtype=np.float64)
    rayleigh = np.zeros(nsel, dtype=np.float64)
    raman = np.zeros(nsel, dtype=np.float64)
    node_mu = np.zeros((nbin, nsel), dtype=np.float64)
    node_mt = np.zeros((nbin, nsel), dtype=np.float64)

    p_double = ctypes.POINTER(ctypes.c_double)
    p_i8 = ctypes.POINTER(ctypes.c_int8)
    rc = func(
        nsel, nnode,
        p.ctypes.data_as(p_double), n.ctypes.data_as(p_double),
        position.ctypes.data_as(p_double), polarization.ctypes.data_as(p_double),
        pa.ctypes.data_as(p_double), pb.ctypes.data_as(p_double),
        channel.ctypes.data_as(p_i8), bout.ctypes.data_as(p_double),
        ng.ctypes.data_as(p_double), bt.ctypes.data_as(p_double),
        float(aperture_radius), float(facing_width),
        nbin, float(t_min), float(t_max),
        response.ctypes.data_as(p_double), int(response.size),
        attenuation.ctypes.data_as(p_double), int(attenuation.size),
        float(attenuation_xmax),
        charge.ctypes.data_as(p_double), rayleigh.ctypes.data_as(p_double),
        raman.ctypes.data_as(p_double), node_mu.ctypes.data_as(p_double),
        node_mt.ctypes.data_as(p_double), max(1, int(n_threads)),
    )
    if rc != 0:
        if required:
            raise RuntimeError(f"Native scattering receiver returned error code {rc}")
        return None
    return charge, rayleigh, raman, node_mu, node_mt
