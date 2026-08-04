"""Numerical checks for the PSD-POD complex path (``_solve_eigh_complex``).

The CLI suite reaches this path but does not assert its arithmetic: dropping the
metric from the time coefficients, or dropping the ``1/sqrt(lambda*N)`` mode
normalization, still leaves that suite green.

NOTE: ``reference_psd_pod`` is the pre-refactor formula (commands.py at
3102d9a), kept as a refactoring guard against the shared solver. It mirrors
``_solve_eigh_complex`` and is a characterization test, NOT an independent
physics oracle — it only proves the shared path stays consistent with that
historical expression. Correctness of the PSD-POD construction itself is not
claimed here.
"""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy.core.decomposition import SpatialMetric, weighted_second_order


def reference_psd_pod(ensemble: np.ndarray, weights: np.ndarray, n_modes_save: int):
    """Verbatim pre-refactor formula (commands.py::_run_psd_pod at 3102d9a).

    Characterization / refactoring guard only — not an independent oracle.
    """
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


def test_manufactured_rank_one_ensemble_matches_closed_form():
    """Rank-1 ensemble a_n * phi has a closed-form leading eigenvalue and mode.

    Build ensemble[n] = a_n * phi for a known W-normalised real phi. Under the
    weighted temporal-kernel convention the kernel is rank-1 with leading
    eigenvalue (sum_n |a_n|^2) * ||phi||_W^2 / N; the leading mode recovers
    phi up to a global phase; residual modes sit at ~0 energy. Independent of
    the pre-refactor twin — a wrong sqrt(W) seam moves the eigenvalue off the
    closed form.
    """
    rng = np.random.default_rng(11)
    n_realizations = 8
    weights = np.array([0.5, 1.0, 2.0, 0.25, 3.0])
    phi_raw = rng.standard_normal(weights.size)
    phi_norm = np.sqrt(np.sum(phi_raw**2 * weights))
    phi = phi_raw / phi_norm
    # ||phi||_W^2 == 1 by construction
    assert np.isclose(np.sum(phi**2 * weights), 1.0)

    amplitudes = rng.standard_normal(n_realizations) + 1j * rng.standard_normal(n_realizations)
    ensemble = amplitudes[:, np.newaxis] * phi[np.newaxis, :]

    modes, eigenvalues, _coeffs = _run_solver(ensemble, weights, n_keep=n_realizations)

    expected_leading = float(np.sum(np.abs(amplitudes) ** 2) * 1.0 / n_realizations)
    # Relative Gram-rank filter keeps only the single significant eigenvalue;
    # residual directions sit at the noise floor and are not returned as modes.
    assert eigenvalues.size == 1
    assert modes.shape[1] == 1
    np.testing.assert_allclose(np.real(eigenvalues[0]), expected_leading, rtol=1e-10, atol=1e-12)

    # Leading mode recovers phi up to a global phase (real phi → sign only).
    mode0 = modes[:, 0]
    overlap = np.abs(np.vdot(mode0 * weights, phi)) / (
        np.sqrt(np.vdot(mode0 * weights, mode0).real) * np.sqrt(np.sum(phi**2 * weights))
    )
    assert float(overlap) >= 1.0 - 1e-8


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
    # Zero measure: the cell contributes nothing (exact sqrt(0) = 0). On this
    # ensemble the shared path still agrees with the reference within tol.
    _assert_matches_reference(ensemble, weights, n_keep=4)


def test_psd_pod_negative_weight_station_raises():
    """A negative weight is not a valid inner-product metric — raise ValueError.

    An isolated zero among positive weights is still allowed (it contributes
    nothing); a negative entry means the metric is not an inner product, so
    the solver refuses it before taking ``sqrt(W)``.
    """
    ensemble, weights = _fourier_ensemble()
    weights = weights.copy()
    weights[1] = -0.5
    assert weights[1] < 0.0

    with pytest.raises(ValueError, match="negative weight"):
        _run_solver(ensemble, weights, n_keep=4)
