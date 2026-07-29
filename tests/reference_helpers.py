"""Shared helpers for analytic reference fixtures (regen + comparison tests).

Canonicalization of the DMD spectrum and POD energy normalisation live here
so the committed JSON and the comparison test cannot drift apart.
"""

from __future__ import annotations

import tempfile
import warnings
from typing import Any

import numpy as np

from openmodalpy import DMDAnalyzer, PODAnalyzer
from openmodalpy.example_data import generate_example_dataset

# Taylor–Green rank-1 fields emit this; ignore only that message.
_DMD_EFFECTIVE_RANK_WARNING = r"DMD effective rank .* is below the requested"


def analyzer_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Hand a generator payload to an analyzer under the loader contract."""
    return {
        "q": payload["q"],
        "x": payload["x"],
        "y": payload["y"],
        "z": payload["z"],
        "dt": payload["dt"],
        "Nx": payload["Nx"],
        "Ny": payload["Ny"],
        "Nz": payload["Nz"],
        "Ns": payload["Ns"],
        "metadata": payload.get("metadata", {}),
    }


def make_loader(payload: dict[str, Any]):
    data = analyzer_data(payload)
    return lambda _path: data


def _mags_agree(a: float, b: float, rtol: float) -> bool:
    """Relative consecutive-magnitude test (no fixed-digit rounding)."""
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return True
    return abs(a - b) <= rtol * scale


def canonicalize_dmd_eigenvalues(eigvals: np.ndarray, rtol: float) -> np.ndarray:
    """Stable DMD spectrum order for reference fixtures.

    Magnitudes stay descending (same primary key as the analyzer). Eigenvalues
    whose consecutive ``|λ|`` agree within ``rtol`` form a group; within each
    group the order is phase ascending. That removes LAPACK conjugate-pair
    emission order from the recorded spectrum without changing analyzer code.
    """
    eigvals = np.asarray(eigvals, dtype=np.complex128).reshape(-1)
    if eigvals.size == 0:
        return eigvals.copy()

    mag = np.abs(eigvals)
    # Primary order: |λ| descending (stable for unequal magnitudes).
    order = np.argsort(mag)[::-1]
    sorted_eigs = eigvals[order]
    sorted_mag = mag[order]

    out = np.empty_like(sorted_eigs)
    i = 0
    n = sorted_eigs.size
    while i < n:
        j = i + 1
        while j < n and _mags_agree(sorted_mag[j - 1], sorted_mag[j], rtol):
            j += 1
        group = sorted_eigs[i:j]
        # Phase ascending within a tied-|λ| group (np.angle in (-π, π]).
        group = group[np.argsort(np.angle(group))]
        out[i:j] = group
        i = j
    return out


def pod_fractions_over_pretruncation_total(
    eigenvalues: np.ndarray,
    energy_captured_fraction: float,
) -> np.ndarray:
    """Normalise kept POD eigenvalues by the pre-truncation energy total.

    ``pod.eigenvalues`` is already truncated to ``n_modes_save``. Recover the
    full sum via ``total = sum(kept) / energy_captured_fraction`` (set before
    truncation on the analyzer). Fractions then sum to the captured fraction,
    not to 1.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    kept = float(np.sum(lam))
    ecf = float(energy_captured_fraction)
    if ecf > 0.0:
        total = kept / ecf
    else:
        # Guard: no captured energy — fall back to the kept sum only.
        total = kept
    if total <= 0.0:
        raise RuntimeError(
            f"POD energy total non-positive (kept={kept}, energy_captured_fraction={ecf})"
        )
    return lam / total


def compute_reference_spectra(
    generator: str,
    params: dict[str, Any],
    n_modes: int,
    rtol: float,
) -> dict[str, Any]:
    """Run POD + DMD; return arrays ready for fixture write or comparison."""
    payload = generate_example_dataset(generator, params)
    loader = make_loader(payload)

    with tempfile.TemporaryDirectory(prefix=f"ref_{generator}_") as tmp:
        pod = PODAnalyzer(
            file_path=f"ref_{generator}",
            data_loader=loader,
            spatial_weight_type="uniform",
            n_modes_save=n_modes,
            results_dir=tmp,
            figures_dir=tmp,
            use_parallel=False,
        )
        pod.load_and_preprocess()
        pod.perform_pod()
        lam = np.asarray(pod.eigenvalues, dtype=np.float64)
        ecf = float(pod.energy_captured_fraction)
        pod_frac = pod_fractions_over_pretruncation_total(lam, ecf)

        dmd = DMDAnalyzer(
            file_path=f"ref_{generator}_dmd",
            data_loader=loader,
            spatial_weight_type="uniform",
            n_modes_save=n_modes,
            results_dir=tmp,
            figures_dir=tmp,
            rank=n_modes,
            use_parallel=False,
        )
        dmd.load_and_preprocess()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=_DMD_EFFECTIVE_RANK_WARNING,
                category=RuntimeWarning,
            )
            dmd.perform_dmd()
        eig = canonicalize_dmd_eigenvalues(np.asarray(dmd.eigenvalues), rtol=rtol)
        dmd_abs = np.abs(eig).astype(np.float64)
        dmd_phase = np.angle(eig).astype(np.float64)

    return {
        "pod_energy_fractions": pod_frac,
        "dmd_abs_lambda": dmd_abs,
        "dmd_phase": dmd_phase,
        "energy_captured_fraction": ecf,
        "payload": payload,
    }
