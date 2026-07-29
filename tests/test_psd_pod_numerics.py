"""Numerical checks for the PSD-POD complex path (``_solve_eigh_complex``).

The CLI suite reaches this path but does not assert its arithmetic: dropping the
metric from the time coefficients, or dropping the ``1/sqrt(lambda*N)`` mode
normalization, still leaves that suite green. These tests compare against an
independent oracle transcribed from the pre-refactor formula.
"""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy.core.decomposition import SpatialMetric, weighted_second_order


def reference_psd_pod(ensemble: np.ndarray, weights: np.ndarray, n_modes_save: int):
    """Verbatim pre-refactor formula (commands.py::_run_psd_pod at 3102d9a)."""
    n_realizations = ensemble.shape[0]
    ensemble_weighted = ensemble * np.sqrt(weights)[np.newaxis, :]
    kernel = (ensemble_weighted @ ensemble_weighted.conj().T) / n_realizations
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    order = np.argsort(eigenvalues.real)[::-1]
    keep = min(n_modes_save, len(order))
    eigenvalues = np.real_if_close(eigenvalues[order][:keep])
    eigenvectors = eigenvectors[:, order][:, :keep]
    safe_eigs = np.maximum(np.real(eigenvalues), 1e-16)
    ensemble_conj = ensemble.conj()
    modes = (ensemble_conj.T @ eigenvectors) / np.sqrt(safe_eigs * n_realizations)
    time_coefficients = ensemble_conj @ (weights[:, np.newaxis] * modes)
    return modes, eigenvalues, time_coefficients


def _phase_invariant_overlap(modes: np.ndarray, ref_modes: np.ndarray) -> float:
    """Min |<u, u_ref>| / (|u| |u_ref|) over modes; complex vectors up to phase."""
    num = np.abs(np.sum(np.conj(modes) * ref_modes, axis=0))
    den = np.linalg.norm(modes, axis=0) * np.linalg.norm(ref_modes, axis=0)
    # Degenerate cases must FAIL, not score a perfect overlap: an empty basis or
    # a zero-norm mode is a broken solve, and defaulting them to 1.0 would make
    # this helper unable to detect exactly the failures it exists to catch.
    assert num.size > 0, "solver returned an empty mode set"
    assert np.all(den > 0), "a mode has zero norm; overlap is undefined"
    return float(np.min(num / den))


def _run_solver(ensemble: np.ndarray, weights: np.ndarray, n_keep: int):
    return weighted_second_order(
        ensemble,
        SpatialMetric(weights),
        method="eigh",
        drop_nonpositive=False,
        n_keep=n_keep,
    )


def _assert_matches_reference(ensemble: np.ndarray, weights: np.ndarray, n_keep: int) -> None:
    ref_modes, ref_eigs, ref_coeffs = reference_psd_pod(ensemble, weights, n_keep)
    modes, eigs, coeffs = _run_solver(ensemble, weights, n_keep)

    # Shapes first: an empty or wrongly-truncated result must not reach the
    # value comparisons, several of which are vacuous on empty arrays.
    n_samples, n_space = ensemble.shape
    assert modes.shape == (n_space, n_keep)
    assert eigs.shape == (n_keep,)
    assert coeffs.shape == (n_samples, n_keep)

    np.testing.assert_allclose(np.real(eigs), np.real(ref_eigs), rtol=1e-10, atol=1e-12)
    assert _phase_invariant_overlap(modes, ref_modes) >= 1.0 - 1e-8
    np.testing.assert_allclose(np.abs(coeffs), np.abs(ref_coeffs), rtol=1e-8, atol=1e-10)


def _fourier_ensemble(seed: int = 4242) -> tuple[np.ndarray, np.ndarray]:
    """10 complex Fourier realizations over 6 spatial points."""
    rng = np.random.default_rng(seed)
    ensemble = rng.standard_normal((10, 6)) + 1j * rng.standard_normal((10, 6))
    weights = np.array([0.5, 1.0, 2.0, 0.25, 3.0, 1.5])
    return ensemble, weights


def test_psd_pod_positive_nonuniform_metric():
    ensemble, weights = _fourier_ensemble()
    assert np.all(weights > 0)
    assert len(np.unique(weights)) == weights.size
    _assert_matches_reference(ensemble, weights, n_keep=4)


def test_psd_pod_isolated_zero_weight_station():
    ensemble, weights = _fourier_ensemble()
    weights = weights.copy()
    weights[2] = 0.0
    assert weights[2] == 0.0
    assert np.count_nonzero(weights == 0.0) == 1
    # Shared solver floors sqrt(W) at 1e-12; bare sqrt(0) is exact zero. On this
    # ensemble the two paths still agree within the tolerances below.
    _assert_matches_reference(ensemble, weights, n_keep=4)


def test_psd_pod_negative_weight_station_is_silently_accepted():
    """A negative weight used to abort the solve; now it is quietly floored.

    This pins a known regression rather than endorsing it. A negative entry in
    the spatial metric is not a small numerical nuisance: it means the inner
    product is not an inner product, so every "energy" downstream is
    meaningless. The pre-refactor code let NaN reach the kernel and
    ``np.linalg.eigh`` raised; flooring sqrt(W) at 1e-12 replaced that loud
    failure with a plausible-looking answer.

    A separate open issue tracks making a zero-measure or negative metric raise.
    When that lands, this test should flip to asserting the exception.
    """
    ensemble, weights = _fourier_ensemble()
    weights = weights.copy()
    weights[1] = -0.5
    assert weights[1] < 0.0

    # errstate because sqrt of a negative number is precisely the failure being
    # pinned, and this suite runs with numpy warnings escalated to errors.
    with np.errstate(invalid="ignore"), pytest.raises(np.linalg.LinAlgError):
        reference_psd_pod(ensemble, weights, 4)

    modes, eigs, coeffs = _run_solver(ensemble, weights, n_keep=4)
    assert np.isfinite(modes).all()
    assert np.isfinite(eigs).all()
    assert np.isfinite(coeffs).all()
    # Finite is not enough on its own — all-zeros would satisfy it too.
    assert np.linalg.norm(modes) > 0.0
    assert np.linalg.norm(coeffs) > 0.0
    assert np.any(np.abs(eigs) > 0.0)
