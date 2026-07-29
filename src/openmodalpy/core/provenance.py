"""Runtime provenance attached to every result file by :func:`write_results`.

The block records *what produced the file* — package versions, FFT backend,
thread count, config hash, seed, git SHA, UTC timestamp. Keys use the
``prov_`` prefix so they never collide with analysis attributes. Nothing here
may raise or omit a key: unresolved values are ``"unavailable"`` or ``"none"``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Cached once: git is best-effort and must not run on every write after the first.
_GIT_SHA: str | None = None

# Cap recursion when normalising attrs so a cyclic mapping cannot blow the stack.
_HASH_MAX_DEPTH = 64


def _package_version(name: str) -> str:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return "unavailable"
    except Exception:
        return "unavailable"


def _openmodalpy_version() -> str:
    """Version of the running package, not stale install-time metadata."""
    try:
        import openmodalpy

        version = getattr(openmodalpy, "__version__", None)
        if isinstance(version, str) and version:
            return version
    except Exception:
        pass
    return _package_version("openmodalpy")


def _fft_backend() -> str:
    try:
        from fftkit import DEFAULT_BACKEND

        return str(DEFAULT_BACKEND)
    except Exception:
        return "unavailable"


def _blas_threads() -> int:
    """Observed BLAS/OpenMP thread count (record only; never set).

    Returns ``0`` when the count cannot be determined (not a plausible ``1``).
    """
    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
        counts = [int(p["num_threads"]) for p in pools if p.get("num_threads") is not None]
        if counts:
            return max(counts)
    except Exception:
        pass
    return 0


def _git_sha() -> str:
    """Best-effort ``git rev-parse HEAD`` for the openmodalpy package checkout.

    Resolve only from the package directory (``Path(__file__)``), never from
    the process working directory — an analysis run inside another repo must
    not stamp that repo's HEAD onto the result.
    """
    global _GIT_SHA
    if _GIT_SHA is not None:
        return _GIT_SHA

    try:
        start = Path(__file__).resolve().parent
    except Exception:
        _GIT_SHA = "unavailable"
        return _GIT_SHA

    try:
        d = start
    except Exception:
        _GIT_SHA = "unavailable"
        return _GIT_SHA

    for _ in range(16):
        git_dir = d / ".git"
        if git_dir.exists():
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=d,
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                    check=False,
                )
                sha = (result.stdout or "").strip()
                if result.returncode == 0 and sha:
                    _GIT_SHA = sha
                    return _GIT_SHA
            except Exception:
                pass
            _GIT_SHA = "unavailable"
            return _GIT_SHA
        parent = d.parent
        if parent == d:
            break
        d = parent

    _GIT_SHA = "unavailable"
    return _GIT_SHA


def _seed_from_attrs(attrs: Mapping[str, Any]) -> str:
    for key in ("data_seed", "seed", "random_seed"):
        if key in attrs and attrs[key] is not None and attrs[key] != "":
            return str(attrs[key])
    return "none"


def _normalise_for_hash(value: Any, depth: int = 0) -> Any:
    """Turn analysis attrs into JSON-stable Python values.

    Depth-limited so cyclic containers never raise ``RecursionError``.
    Unstringable objects fall back to a type name rather than propagating.
    """
    if depth > _HASH_MAX_DEPTH:
        return "<max-depth>"
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        try:
            keys = sorted(value, key=str)
        except Exception:
            return "<unsortable-mapping>"
        out: dict[str, Any] = {}
        for k in keys:
            try:
                out[str(k)] = _normalise_for_hash(value[k], depth + 1)
            except Exception:
                out[str(k)] = "<unavailable>"
        return out
    if isinstance(value, (list, tuple)):
        items = []
        for v in value:
            try:
                items.append(_normalise_for_hash(v, depth + 1))
            except Exception:
                items.append("<unavailable>")
        return items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def config_sha256(attrs: Mapping[str, Any] | None) -> str:
    """SHA-256 of the caller's attrs, excluding every ``prov_`` key.

    Same config → same hash; the write timestamp must not feed the digest.
    Any failure yields ``"unavailable"`` so a write never aborts on the hash.
    """
    try:
        items = sorted((attrs or {}).items(), key=lambda kv: str(kv[0]))
        payload = {str(k): _normalise_for_hash(v) for k, v in items if not str(k).startswith("prov_")}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    except Exception:
        return "unavailable"


def safe_attrs_for_hdf5(attrs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy of *attrs* with values that h5py can store as attributes.

    Provenance hashing already tolerates cycles and unstringable objects; this
    path is only for the final ``handle.attrs.update`` so a pathological caller
    attr cannot abort a write that has already computed provenance. Scalars and
    non-object arrays pass through; only values h5py would reject are replaced.

    ``None`` is deliberately NOT rescued: h5py has always rejected it, so a
    ``None`` attribute is a bug in whatever assembled the metadata. Writing
    ``"none"`` instead would hide it, and this module's job is to make a result
    file more honest, not less.
    """
    out: dict[str, Any] = {}
    for key, value in dict(attrs or {}).items():
        name = str(key)
        if isinstance(value, (str, int, float, bool)):
            out[name] = value
            continue
        if value is None:
            out[name] = value
            continue
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            out[name] = value.item()
            continue
        if isinstance(value, bytes):
            out[name] = value.decode("utf-8", errors="replace")
            continue
        if isinstance(value, np.ndarray) and value.dtype != object:
            out[name] = value
            continue
        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
                if arr.dtype != object:
                    out[name] = arr
                    continue
            except Exception:
                pass
        try:
            out[name] = str(value)
        except Exception:
            out[name] = f"<{type(value).__name__}>"
    return out


def collect_provenance(attrs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the twelve-field provenance block for one result write.

    All string fields are non-empty. ``prov_blas_threads`` is an ``int``.
    Unresolved values are ``"unavailable"`` or ``"none"`` — never omitted.
    """
    attrs = dict(attrs or {})
    return {
        "prov_openmodalpy_version": _openmodalpy_version(),
        "prov_python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "prov_numpy_version": _package_version("numpy"),
        "prov_scipy_version": _package_version("scipy"),
        "prov_h5py_version": _package_version("h5py"),
        "prov_fftkit_version": _package_version("fftkit"),
        "prov_fft_backend": _fft_backend(),
        "prov_blas_threads": _blas_threads(),
        "prov_config_sha256": config_sha256(attrs),
        "prov_created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prov_git_sha": _git_sha(),
        "prov_seed": _seed_from_attrs(attrs),
    }
