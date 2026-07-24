import numpy as np
import pytest
import sys

from modalpy.fft import fft_backends


def test_backend_registration():
    backends = fft_backends.get_fft_backend_names()
    assert 'numpy' in backends
    assert 'scipy' in backends
    assert 'accelerate' in backends
    assert 'mkl' in backends


def test_accelerate_unavailable():
    func = fft_backends.get_fft_func('accelerate')
    if sys.platform != 'darwin':
        with pytest.raises(NotImplementedError):
            func(np.ones(4))


def test_mkl_unavailable():
    try:
        import mkl_fft  # noqa: F401
    except Exception:
        func = fft_backends.get_fft_func('mkl')
        with pytest.raises(Exception):
            func(np.ones(4))
