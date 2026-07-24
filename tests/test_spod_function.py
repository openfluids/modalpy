import numpy as np
from modalpy.core.base import spod_function


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

