import numpy as np
import pytest

from openmodalpy.core.base import blocksfft

WINDOWS = ("hamming", "hann", "blackman", "bartlett", "sine")


def test_blocksfft_constant_signal():
    q = np.ones((4, 2))
    result = blocksfft(q, nfft=4, nblocks=1, novlap=0, n_threads=2)
    assert result.shape == (4 // 2 + 1, 2, 1)
    assert np.allclose(result, 0)


@pytest.mark.parametrize("window_type", WINDOWS)
def test_blocksfft_serial_parallel_window_identity(window_type):
    """Serial and parallel paths must agree bit-for-bit for every window."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((64, 3))
    ser = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type=window_type, use_parallel=False)
    par = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type=window_type, use_parallel=True)
    assert np.allclose(ser, par, rtol=0, atol=1e-12), (
        f"serial/parallel disagree for window_type={window_type!r}: max|diff|={np.max(np.abs(ser - par)):.3e}"
    )


def test_blocksfft_hann_blackman_differ_from_hamming():
    """hann/blackman must not silently collapse onto the hamming path."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((64, 3))
    hamming = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type="hamming")
    for w in ("hann", "blackman", "sine"):
        other = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type=w)
        assert not np.allclose(other, hamming), f"{w} collapsed onto hamming"


def test_blocksfft_unsupported_window_raises():
    """Unsupported window names must raise, not silently substitute."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((64, 3))
    with pytest.raises(ValueError) as exc_info:
        blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type="not_a_window")
    # The offending name must appear, so an unrelated ValueError cannot satisfy this.
    assert "not_a_window" in str(exc_info.value)
