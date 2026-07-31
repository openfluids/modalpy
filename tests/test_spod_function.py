import numpy as np
import pytest

from openmodalpy.core.base import PARALLEL_AVAILABLE, spod_function
from openmodalpy.core.parallel import spod_single_frequency_optimized


def test_spod_function_simple():
    qhat = np.array([[1.0], [0.0]], dtype=complex)
    w = np.ones((2, 1))
    phi, lam, psi = spod_function(qhat, nblocks=1, dst=1.0, w=w, return_psi=True)
    assert phi.shape == (2, 1)
    assert psi.shape == (1, 1)
    assert np.allclose(lam, 1.0)
    assert np.allclose(phi[:, 0], [1.0, 0.0])


def test_spod_function_per_component_weights():
    qhat = np.array([[1.0], [2.0]], dtype=complex)
    w = np.zeros((1, 1, 2))
    w[0, 0, 0] = 1.0
    w[0, 0, 1] = 2.0
    phi, lam, psi = spod_function(qhat, nblocks=1, dst=1.0, w=w, return_psi=True)
    assert phi.shape == (2, 1)
    assert psi.shape == (1, 1)
    assert np.allclose(lam, 9.0)
    assert np.allclose(phi[:, 0], [1 / 3, 2 / 3])


@pytest.mark.parametrize("use_parallel", [False, True])
def test_spod_function_rejects_invalid_metric(use_parallel):
    """Negative weights and a zero-measure metric raise; isolated zeros stay allowed.

    Both the serial and optimized routes must refuse the same invalid metrics.
    An isolated zero among positive weights is still accepted: SPOD applies the
    weights as they are, so that cell contributes nothing to the CSD. The 1e-12
    floor is the POD seam's, not this one's.
    """
    if use_parallel and not PARALLEL_AVAILABLE:
        pytest.skip("optimized SPOD route unavailable")

    rng = np.random.default_rng(11)
    n_space, nblocks = 6, 4
    qhat = rng.standard_normal((n_space, nblocks)) + 1j * rng.standard_normal((n_space, nblocks))
    w_neg = np.array([1.0, -0.5, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
    w_zero = np.zeros((n_space, 1))
    w_iso = np.array([1.0, 0.0, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
    w_ok = np.array([0.5, 1.0, 2.0, 0.25, 3.0, 1.5]).reshape(-1, 1)

    with pytest.raises(ValueError, match="negative weight"):
        spod_function(qhat, nblocks, 0.1, w_neg, use_parallel=use_parallel)
    with pytest.raises(ValueError, match="zero total measure"):
        spod_function(qhat, nblocks, 0.1, w_zero, use_parallel=use_parallel)

    phi_iso, lam_iso = spod_function(qhat, nblocks, 0.1, w_iso, use_parallel=use_parallel)
    phi_ok, lam_ok = spod_function(qhat, nblocks, 0.1, w_ok, use_parallel=use_parallel)
    assert phi_iso.shape[0] == n_space
    assert phi_ok.shape[0] == n_space
    assert np.all(np.isfinite(lam_iso))
    assert np.all(np.isfinite(lam_ok))


@pytest.mark.parametrize("use_parallel", [False, True])
def test_spod_function_rejects_nonfinite_metric(use_parallel):
    """NaN and inf weights raise the same way on both SPOD routes."""
    if use_parallel and not PARALLEL_AVAILABLE:
        pytest.skip("optimized SPOD route unavailable")

    rng = np.random.default_rng(11)
    n_space, nblocks = 6, 4
    qhat = rng.standard_normal((n_space, nblocks)) + 1j * rng.standard_normal((n_space, nblocks))
    w_nan = np.array([1.0, np.nan, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1)
    w_inf = np.array([1.0, np.inf, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1)

    with pytest.raises(ValueError, match="non-finite"):
        spod_function(qhat, nblocks, 0.1, w_nan, use_parallel=use_parallel)
    with pytest.raises(ValueError, match="non-finite"):
        spod_function(qhat, nblocks, 0.1, w_inf, use_parallel=use_parallel)


def test_spod_single_frequency_optimized_rejects_invalid_metric():
    """Direct call refuses negative, zero-measure, and non-finite metrics.

    Guards the path that already validates through the shared body so a future
    edit cannot drop that check without this test failing first.
    """
    if not PARALLEL_AVAILABLE:
        pytest.skip("optimized SPOD route unavailable")

    rng = np.random.default_rng(11)
    n_space, nblocks = 6, 4
    qhat = rng.standard_normal((n_space, nblocks)) + 1j * rng.standard_normal((n_space, nblocks))

    with pytest.raises(ValueError, match="negative weight"):
        spod_single_frequency_optimized(qhat, np.array([1.0, -0.5, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1), nblocks, 0.1)
    with pytest.raises(ValueError, match="zero total measure"):
        spod_single_frequency_optimized(qhat, np.zeros((n_space, 1)), nblocks, 0.1)
    with pytest.raises(ValueError, match="non-finite"):
        spod_single_frequency_optimized(qhat, np.array([1.0, np.nan, 2.0, 1.0, 1.0, 1.0]).reshape(-1, 1), nblocks, 0.1)
