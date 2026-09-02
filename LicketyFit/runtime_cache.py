"""Portable, source-guarded locations for generated runtime artifacts.

The cache must survive a change of batch node.  A node-local ``/tmp`` cache is
useful only as a last-resort fallback: on lxplus it makes every new job pay the
full Numba/native compilation cost again.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import stat
import sys
from typing import Iterable, Mapping


RUNTIME_IDENTITY_FILES = (
    "scripts/batch_fit_driver.py",
)


def runtime_identity_files(project_root: str | Path) -> tuple[str, ...]:
    """Return every source file capable of changing a compiled fit path.

    Earlier releases hashed only six files.  Changes in modules such as
    ``mcs_curved_path.py`` or ``cosmic_navigation.py`` could therefore leave a
    bootstrap-ready marker in a cache whose relevant Numba entries had become
    stale.  Numba would repair those entries lazily inside event zero.  Hashing
    the small source tree prevents that false-ready state.
    """
    root = Path(project_root).resolve()
    relative = set(RUNTIME_IDENTITY_FILES)
    module_root = root / "LicketyFit"
    for pattern in ("*.py", "*.cpp"):
        for path in module_root.glob(pattern):
            if path.is_file():
                relative.add(path.relative_to(root).as_posix())
    return tuple(sorted(relative))


def runtime_cache_content_tag(
    project_root: str | Path,
    *,
    source_files: Iterable[str] | None = None,
    python_identity: str | None = None,
) -> str:
    """Return a path-independent tag for compiled/runtime physics artifacts.

    The absolute extraction path is deliberately excluded.  Every source that
    owns a cached binary or optical transfer participates, along with Python's
    ABI cache tag.  Missing files are represented explicitly, so an incomplete
    installation can never alias a complete release.
    """
    root = Path(project_root).resolve()
    if python_identity is None:
        dependency_versions = []
        for distribution in ("numba", "llvmlite", "numpy"):
            try:
                resolved = importlib_metadata.version(distribution)
            except importlib_metadata.PackageNotFoundError:
                resolved = "missing"
            dependency_versions.append(f"{distribution}={resolved}")
        python_identity = (
            f"{sys.implementation.name}:{sys.version_info.major}."
            f"{sys.version_info.minor}:{getattr(sys.implementation, 'cache_tag', '')}:"
            + ":".join(dependency_versions)
        )
    if source_files is None:
        source_files = runtime_identity_files(root)
    digest = hashlib.sha256(str(python_identity).encode("utf-8"))
    for relative in source_files:
        normalized = Path(relative).as_posix()
        digest.update(normalized.encode("utf-8"))
        try:
            digest.update((root / normalized).read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class RuntimeCacheLocation:
    """Resolved cache root plus enough provenance for user-facing warnings."""

    root: Path
    persistent: bool
    source: str


def _runtime_suffix(project_root: str | Path) -> str:
    uid = int(os.getuid()) if hasattr(os, "getuid") else 0
    tag = runtime_cache_content_tag(project_root)
    return (
        f"licketyfit-{uid}-{tag}-"
        f"py{sys.version_info.major}{sys.version_info.minor}"
    )


def resolve_runtime_cache_location(
    project_root: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    create: bool = True,
) -> RuntimeCacheLocation:
    """Resolve a persistent cache, falling back to node-local storage safely.

    Resolution order is an explicit ``LF_RUNTIME_CACHE_DIR``, XDG cache home,
    the ordinary per-user cache directory, and finally ``TMPDIR``/``/tmp``.
    The content/Python suffix keeps independent releases and ABIs isolated.
    """
    env = os.environ if environ is None else environ
    explicit = str(env.get("LF_RUNTIME_CACHE_DIR", "")).strip()
    xdg = str(env.get("XDG_CACHE_HOME", "")).strip()
    candidates: list[tuple[Path, bool, str]] = []
    if explicit:
        candidates.append((Path(explicit).expanduser(), True, "explicit"))
    elif xdg:
        candidates.append((Path(xdg).expanduser() / "licketyfit", True, "xdg"))
    else:
        try:
            home = str(env.get("HOME", "")).strip()
            home_path = Path(home).expanduser() if home else Path.home()
            candidates.append((home_path / ".cache" / "licketyfit", True, "home"))
        except Exception:
            pass

    temporary = str(env.get("TMPDIR", "")).strip() or "/tmp"
    uid = int(os.getuid()) if hasattr(os, "getuid") else int(os.getpid())
    candidates.append((
        Path(temporary).expanduser() / f"licketyfit-{uid}",
        False,
        "temporary-private",
    ))
    suffix = _runtime_suffix(project_root)
    errors: list[str] = []
    for base, persistent, source in candidates:
        root = base / suffix
        if not create:
            return RuntimeCacheLocation(root=root, persistent=persistent, source=source)
        try:
            # Native shared objects are executed directly from this tree.
            # Every candidate, including explicit/XDG/home paths, must therefore
            # be owned by this account, be a real directory rather than a
            # symlink, and exclude group/other access. This also makes
            # XDG_CACHE_HOME=/tmp fail closed instead of trusting a pre-created
            # cross-user path.
            base.mkdir(mode=0o700, parents=True, exist_ok=True)
            base_info = base.lstat()
            if (
                stat.S_ISLNK(base_info.st_mode)
                or (hasattr(os, "getuid") and int(base_info.st_uid) != uid)
            ):
                raise OSError(f"unsafe runtime cache owner or symlink: {base}")
            if stat.S_IMODE(base_info.st_mode) != 0o700:
                base.chmod(0o700)
                if stat.S_IMODE(base.lstat().st_mode) != 0o700:
                    raise OSError(f"runtime cache is not private mode 0700: {base}")
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            root_info = root.lstat()
            if (
                stat.S_ISLNK(root_info.st_mode)
                or (hasattr(os, "getuid") and int(root_info.st_uid) != uid)
            ):
                raise OSError(f"unsafe runtime root owner or symlink: {root}")
            if stat.S_IMODE(root_info.st_mode) != 0o700:
                root.chmod(0o700)
                if stat.S_IMODE(root.lstat().st_mode) != 0o700:
                    raise OSError(f"runtime root is not private mode 0700: {root}")
            probe = root / ".write_probe"
            probe.touch(mode=0o600, exist_ok=True)
            probe.unlink(missing_ok=True)
            return RuntimeCacheLocation(
                root=root,
                persistent=persistent,
                source=source,
            )
        except OSError as exc:
            errors.append(f"{root}: {exc}")
    raise OSError("No writable LicketyFit runtime cache: " + "; ".join(errors))


# ---------------------------------------------------------------------------
# Portable Numba on-disk cache
# ---------------------------------------------------------------------------
#
# Numba's stock ``UserProvidedCacheLocator`` places compiled artifacts under
# ``NUMBA_CACHE_DIR/<parentdir>_<sha1(absolute source directory)>`` and marks
# them fresh with the source file's ``(mtime, size)``.  Both choices defeat
# reuse on a batch farm: every job that extracts the package into a fresh
# scratch directory gets a new absolute path (new subdirectory, cache miss)
# and often new mtimes (stamp mismatch, cache miss), and so pays the full
# ~1.5 minute compilation before its first event.
#
# The locator below keys the subdirectory on the source path RELATIVE to the
# package root and stamps freshness with the SHA-256 of the source bytes.
# Compiled code is therefore reused wherever the same source content is run,
# and never reused for edited source.  Numba still records the host CPU name
# and feature set inside each index entry, so a cache directory shared by
# heterogeneous nodes simply accumulates one entry per CPU family.
#
# Only files under the package root are handled; everything else falls
# through to Numba's own locators unchanged.

_PORTABLE_LOCATOR_INSTALLED = False
_SOURCE_DIGEST_CACHE: dict = {}


def _source_content_digest(path: str) -> str:
    try:
        st = __import__("os").stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        key = (path, None, None)
    cached = _SOURCE_DIGEST_CACHE.get(key)
    if cached is None:
        with open(path, "rb") as handle:
            cached = hashlib.sha256(handle.read()).hexdigest()
        _SOURCE_DIGEST_CACHE[key] = cached
    return cached


def install_portable_numba_cache_locator(
    project_root: str | Path, cache_root: str | Path | None = None
) -> bool:
    """Make Numba's disk cache reusable across extraction paths and mtimes.

    Must run before any ``@njit`` function in the package is first compiled
    or loaded.  ``cache_root`` is where portable artifacts live; when omitted
    they go under ``NUMBA_CACHE_DIR/portable``.  Because every entry is
    stamped with its own source file's content hash, the root does NOT need
    to include the package content tag: editing one module only invalidates
    that module's entries.  Returns True when the locator is active.  Set
    ``LF_PORTABLE_NUMBA_CACHE=0`` to keep Numba's stock behaviour.
    """
    global _PORTABLE_LOCATOR_INSTALLED
    import os

    if _PORTABLE_LOCATOR_INSTALLED:
        return True
    if str(os.environ.get("LF_PORTABLE_NUMBA_CACHE", "1")).strip().lower() in {
        "0", "false", "no", "off"
    }:
        return False
    try:
        from numba.core import caching as _caching
        from numba.core import config as _numba_config
    except Exception:
        return False

    root = str(Path(project_root).resolve())
    if cache_root is not None:
        portable_root = str(Path(cache_root).expanduser())
    else:
        portable_root = os.path.join(str(_numba_config.CACHE_DIR or ""), "portable")

    class PortableProjectCacheLocator(_caching.UserProvidedCacheLocator):
        """Path-independent, content-stamped locator for package sources."""

        def __init__(self, py_func, py_file):  # noqa: D401 - numba API
            self._py_file = py_file
            self._lineno = py_func.__code__.co_firstlineno
            relative = os.path.relpath(
                os.path.dirname(os.path.abspath(py_file)), root
            ).replace(os.sep, "__")
            self._cache_path = os.path.join(portable_root, relative)

        def get_source_stamp(self):
            return "sha256:" + _source_content_digest(self._py_file)

        @classmethod
        def from_function(cls, py_func, py_file):
            if not portable_root:
                return None
            try:
                absolute = os.path.abspath(py_file)
            except Exception:
                return None
            if not absolute.startswith(root + os.sep):
                return None
            if not os.path.exists(absolute):
                return None
            self = cls(py_func, absolute)
            try:
                self.ensure_cache_path()
            except OSError:
                return None
            return self

    impl = getattr(_caching, "CacheImpl", None) or getattr(_caching, "_CacheImpl", None)
    if impl is None:
        return False
    classes = impl._locator_classes
    if not any(c.__name__ == "PortableProjectCacheLocator" for c in classes):
        classes.insert(0, PortableProjectCacheLocator)
    _PORTABLE_LOCATOR_INSTALLED = True
    return True


# ---------------------------------------------------------------------------
# Process-wide shared loader for large read-only lookup tables
# ---------------------------------------------------------------------------
#
# ``tables/E_vs_dist_cm_muon.npy`` is a 67 MB pickled object array of 2948
# trajectories.  Two independent consumers (the cone-collapse tables in
# ``particle_cherenkov_model`` and ``scripts/particle_range_lookup``) each
# ``np.load`` it at startup, and unpickling 2948 separate arrays costs
# 0.2-0.6 s each time.  Neither consumer mutates the rows.
#
# Two layers remove that cost:
#   1. an in-process memo so the second consumer reuses the first load;
#   2. a derived, memory-mapped flat cache (one contiguous (N, 2) block plus
#      row offsets) written once per source-file content hash into the
#      runtime cache directory.  A cache hit costs about a millisecond and
#      pages in only the rows a fit actually touches.
# The object array returned on a cache hit has identical shape, dtype and
# row contents to the pickled original (verified element-for-element when the
# cache is written); each row is a read-only view into the mapped block.

_SHARED_TABLE_CACHE: dict = {}
_FLAT_TABLE_FORMAT = "lf_flat_trajectory_table_v1"


def _runtime_cache_root() -> Path | None:
    import os
    raw = os.environ.get("LF_RESOLVED_RUNTIME_CACHE_DIR", "").strip()
    return Path(raw).expanduser() if raw else None


def _digest_memo_path(root: Path) -> Path:
    return root / "tables" / "source_digests.json"


def _source_digest_memoized(real: str, st) -> str:
    """SHA-256 of a source table, memoized on (path, size, mtime) on disk.

    Hashing the 67 MB table costs ~0.2-0.5 s.  The memo uses the same
    freshness stamp as NumPy/Numba/make: a file with unchanged size and
    mtime is assumed to have unchanged content.  The derived cache itself
    always records the full digest of the source it was built from.
    """
    import json
    import os
    key = f"{real}|{st.st_size}|{st.st_mtime_ns}"
    root = _runtime_cache_root()
    memo_path = _digest_memo_path(root) if root is not None else None
    memo = {}
    if memo_path is not None and memo_path.is_file():
        try:
            memo = json.loads(memo_path.read_text())
        except Exception:
            memo = {}
        digest = memo.get(key)
        if isinstance(digest, str) and len(digest) == 64:
            return digest
    hasher = hashlib.sha256()
    with open(real, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if memo_path is not None:
        try:
            memo_path.parent.mkdir(parents=True, exist_ok=True)
            memo[key] = digest
            tmp = memo_path.with_suffix(".json.tmp%d" % os.getpid())
            tmp.write_text(json.dumps(memo, indent=0, sort_keys=True))
            os.replace(tmp, memo_path)
        except OSError:
            pass
    return digest


def _flat_cache_paths(root: Path, digest: str, stem: str):
    base = root / "tables" / f"{stem}_{digest[:16]}_{_FLAT_TABLE_FORMAT}"
    return base.with_suffix(".rows.npy"), base.with_suffix(".offsets.npy"), base.with_suffix(".json")


def _load_flat_object_table(real: str, st, allow_pickle: bool):
    """Return the object array of trajectory rows, via the flat cache."""
    import json
    import os
    import numpy as np

    root = _runtime_cache_root()
    if root is None or str(os.environ.get("LF_FLAT_TABLE_CACHE", "1")).strip().lower() in {
        "0", "false", "no", "off"
    }:
        return np.load(real, allow_pickle=allow_pickle)
    digest = _source_digest_memoized(real, st)
    stem = Path(real).stem
    rows_path, offsets_path, meta_path = _flat_cache_paths(root, digest, stem)

    if rows_path.is_file() and offsets_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            if (
                meta.get("format") == _FLAT_TABLE_FORMAT
                and meta.get("source_sha256") == digest
                and int(meta.get("source_size", -1)) == int(st.st_size)
            ):
                rows = np.load(rows_path, mmap_mode="r")
                offsets = np.load(offsets_path)
                n = offsets.size - 1
                if rows.ndim == 2 and rows.shape[1] == 2 and n == int(meta.get("n_rows", -1)):
                    obj = np.empty(n, dtype=object)
                    for i in range(n):
                        obj[i] = rows[offsets[i]:offsets[i + 1]]
                    return obj
        except Exception:
            pass  # fall through and rebuild

    array = np.load(real, allow_pickle=allow_pickle)
    if not (
        isinstance(array, np.ndarray) and array.dtype == object and array.ndim == 1
        and all(isinstance(r, np.ndarray) and r.ndim == 2 and r.shape[1] == 2 for r in array)
    ):
        return array  # not the legacy trajectory format; no derived cache
    try:
        lengths = np.asarray([r.shape[0] for r in array], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        rows = np.ascontiguousarray(np.concatenate(list(array), axis=0), dtype=np.float64)
        # Verify the flat block reproduces every source row exactly.
        for i, r in enumerate(array):
            if not np.array_equal(rows[offsets[i]:offsets[i + 1]], np.asarray(r, dtype=np.float64)):
                return array
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        tmp_rows = rows_path.with_name(rows_path.name + f".tmp{pid}")
        tmp_off = offsets_path.with_name(offsets_path.name + f".tmp{pid}")
        tmp_meta = meta_path.with_name(meta_path.name + f".tmp{pid}")
        # Write through open handles: np.save(path) would append ".npy".
        with open(tmp_rows, "wb") as handle:
            np.save(handle, rows, allow_pickle=False)
        with open(tmp_off, "wb") as handle:
            np.save(handle, offsets, allow_pickle=False)
        tmp_meta.write_text(json.dumps({
            "format": _FLAT_TABLE_FORMAT,
            "source_path": real,
            "source_sha256": digest,
            "source_size": int(st.st_size),
            "n_rows": int(array.size),
            "n_points": int(rows.shape[0]),
        }, indent=1))
        os.replace(tmp_rows, rows_path)
        os.replace(tmp_off, offsets_path)
        os.replace(tmp_meta, meta_path)
    except OSError:
        pass
    return array


def load_shared_table(path: str | Path, *, allow_pickle: bool = False):
    """Return a process-shared, read-only NumPy array for ``path``."""
    import os
    import numpy as np

    real = os.path.realpath(str(path))
    try:
        st = os.stat(real)
        key = (real, st.st_mtime_ns, st.st_size, bool(allow_pickle))
    except OSError:
        return np.load(real, allow_pickle=allow_pickle)
    array = _SHARED_TABLE_CACHE.get(key)
    if array is None:
        if allow_pickle:
            array = _load_flat_object_table(real, st, allow_pickle)
        else:
            array = np.load(real, allow_pickle=False)
        if isinstance(array, np.ndarray):
            if array.dtype == object:
                for row in array.ravel():
                    if isinstance(row, np.ndarray) and row.flags.writeable:
                        row.setflags(write=False)
            elif array.flags.writeable:
                array.setflags(write=False)
        _SHARED_TABLE_CACHE[key] = array
    return array


# ---------------------------------------------------------------------------
# Derived PMT placement cache
# ---------------------------------------------------------------------------
#
# Extracting PMT locations and normals from the serialized geometry builds two
# SciPy Rotation objects per PMT (~0.6 s for the ~1900-PMT WCTE geometry) on
# every process start.  The result depends only on the geometry file, the
# placement key ("design"/"est") and the active-PMT status masks, so it is
# cached as plain arrays keyed on the SHA-256 of those inputs.

_PLACEMENT_CACHE_FORMAT = "lf_pmt_placements_v1"


def cached_pmt_placements(
    compute,
    *,
    geometry_file: str | Path,
    place_info: str,
    mpmt_status,
    pmt_status,
):
    """Return ``(locations, normals, slots)``, computing via ``compute()``
    only when no cached copy matches the geometry file content, placement
    key and status masks.  Set ``LF_PMT_PLACEMENT_CACHE=0`` to bypass."""
    import json
    import os
    import numpy as np

    root = _runtime_cache_root()
    disabled = str(os.environ.get("LF_PMT_PLACEMENT_CACHE", "1")).strip().lower() in {
        "0", "false", "no", "off"
    }
    geo = os.path.realpath(str(geometry_file))
    if root is None or disabled or not os.path.isfile(geo):
        return compute()
    try:
        st = os.stat(geo)
        geo_digest = _source_digest_memoized(geo, st)
        mask = hashlib.sha256()
        mask.update(np.asarray(mpmt_status, dtype=np.uint8).tobytes())
        mask.update(np.asarray(pmt_status, dtype=np.uint8).tobytes())
        key = hashlib.sha256(
            f"{_PLACEMENT_CACHE_FORMAT}|{geo_digest}|{place_info}|{mask.hexdigest()}".encode()
        ).hexdigest()[:24]
    except Exception:
        return compute()
    base = root / "geometry" / f"pmt_placements_{key}"
    data_path = base.with_suffix(".npz")
    meta_path = base.with_suffix(".json")
    if data_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("format") == _PLACEMENT_CACHE_FORMAT and meta.get("key") == key:
                with np.load(data_path, allow_pickle=False) as z:
                    loc = np.ascontiguousarray(z["locations"], dtype=np.float64)
                    nrm = np.ascontiguousarray(z["normals"], dtype=np.float64)
                    slots = np.ascontiguousarray(z["slots"], dtype=np.int64)
                if (
                    loc.ndim == 2 and loc.shape[1] == 3 and nrm.shape == loc.shape
                    and slots.shape == (loc.shape[0],)
                    and int(meta.get("n_pmts", -1)) == loc.shape[0]
                ):
                    return loc, nrm, slots
        except Exception:
            pass
    loc, nrm, slots = compute()
    try:
        loc = np.ascontiguousarray(loc, dtype=np.float64)
        nrm = np.ascontiguousarray(nrm, dtype=np.float64)
        slots = np.ascontiguousarray(slots, dtype=np.int64)
        base.parent.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        tmp_data = base.with_name(base.name + f".tmp{pid}.npz")
        tmp_meta = base.with_name(base.name + f".tmp{pid}.json")
        with open(tmp_data, "wb") as handle:
            np.savez(handle, locations=loc, normals=nrm, slots=slots)
        tmp_meta.write_text(json.dumps({
            "format": _PLACEMENT_CACHE_FORMAT, "key": key,
            "geometry_file": geo, "geometry_sha256": geo_digest,
            "place_info": str(place_info), "n_pmts": int(loc.shape[0]),
        }, indent=1))
        os.replace(tmp_data, data_path)
        os.replace(tmp_meta, meta_path)
    except OSError:
        pass
    return loc, nrm, slots


# ---------------------------------------------------------------------------
# Uncompressed mirrors of compressed .npz tables
# ---------------------------------------------------------------------------
#
# The receiver-moment (19 MB) and spectral-moment (7 -> 18 MB) tables ship as
# zlib-compressed .npz archives and are inflated on every process start.  An
# uncompressed mirror in the runtime cache directory, keyed on the source's
# content hash and verified array-for-array when written, loads several
# times faster.  ``open_npz_table`` is a drop-in for ``np.load(path)`` and
# returns an ``NpzFile`` usable as a context manager.

_NPZ_MIRROR_FORMAT = "lf_uncompressed_npz_v1"


def open_npz_table(path: str | Path, *, allow_pickle: bool = False):
    """Open ``path`` (an .npz) via an uncompressed mirror when available."""
    import json
    import os
    import zipfile
    import numpy as np

    real = os.path.realpath(str(path))
    root = _runtime_cache_root()
    disabled = str(os.environ.get("LF_NPZ_MIRROR_CACHE", "1")).strip().lower() in {
        "0", "false", "no", "off"
    }
    if root is None or disabled or not os.path.isfile(real):
        return np.load(real, allow_pickle=allow_pickle)
    try:
        with zipfile.ZipFile(real) as zf:
            compressed = any(i.compress_type != zipfile.ZIP_STORED for i in zf.infolist())
    except Exception:
        compressed = False
    if not compressed:
        return np.load(real, allow_pickle=allow_pickle)
    try:
        st = os.stat(real)
        digest = _source_digest_memoized(real, st)
    except OSError:
        return np.load(real, allow_pickle=allow_pickle)
    stem = Path(real).stem
    mirror = root / "tables" / f"{stem}_{digest[:16]}_{_NPZ_MIRROR_FORMAT}.npz"
    meta_path = mirror.with_suffix(".json")
    if mirror.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            if (
                meta.get("format") == _NPZ_MIRROR_FORMAT
                and meta.get("source_sha256") == digest
                and int(meta.get("source_size", -1)) == int(st.st_size)
            ):
                return np.load(mirror, allow_pickle=allow_pickle)
        except Exception:
            pass
    source = np.load(real, allow_pickle=allow_pickle)
    try:
        # np.array keeps 0-d scalars 0-d (ascontiguousarray would promote
        # them to shape (1,)).
        arrays = {k: np.array(source[k], copy=True) for k in source.files}
        mirror.parent.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        tmp = mirror.with_name(mirror.name + f".tmp{pid}")
        with open(tmp, "wb") as handle:
            np.savez(handle, **arrays)
        # Verify against a fresh read of the SOURCE (shape, dtype and values).
        with np.load(tmp, allow_pickle=allow_pickle) as check, \
                np.load(real, allow_pickle=allow_pickle) as fresh:
            ok = set(check.files) == set(fresh.files) and all(
                check[k].shape == fresh[k].shape
                and check[k].dtype == fresh[k].dtype
                and np.array_equal(check[k], fresh[k])
                for k in fresh.files
            )
        if ok:
            meta_path.with_name(meta_path.name + f".tmp{pid}").write_text(json.dumps({
                "format": _NPZ_MIRROR_FORMAT, "source_path": real,
                "source_sha256": digest, "source_size": int(st.st_size),
                "arrays": sorted(arrays),
            }, indent=1))
            os.replace(tmp, mirror)
            os.replace(meta_path.with_name(meta_path.name + f".tmp{pid}"), meta_path)
        else:
            os.unlink(tmp)
    except OSError:
        pass
    return source
