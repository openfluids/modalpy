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


def test_normvar_divides_by_variance_two_scales():
    """normvar divides by variance: normalised block has var 1/v, at two scales.

    Provenance: spod_matlab opts.normvar / PySPOD normalize_data. Dividing by
    the standard deviation would yield unit variance and scale invariance;
    both scales must assert var -> 1/v so a "fix" to std would break this.
    """
    rng = np.random.default_rng(0)
    nfft, npts = 32, 3
    base = rng.standard_normal((nfft, npts))
    base = base - base.mean(axis=0)

    for scale in (1.0, 7.0):
        q = base * scale
        v = np.var(q, axis=0, ddof=1)
        # What the arithmetic does (same as base.py / parallel.py).
        normalised = q / v
        got = np.var(normalised, axis=0, ddof=1)
        expected = 1.0 / v
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

        # Full path: scaling input by c scales the FFT by 1/c under normvar.
        out = blocksfft(
            q,
            nfft=nfft,
            nblocks=1,
            novlap=0,
            normvar=True,
            use_parallel=False,
            window_type="boxcar",
        )
        out_base = blocksfft(
            base,
            nfft=nfft,
            nblocks=1,
            novlap=0,
            normvar=True,
            use_parallel=False,
            window_type="boxcar",
        )
        np.testing.assert_allclose(out, out_base / scale, rtol=0, atol=1e-12)


def test_normvar_serial_parallel_agree():
    """Serial and parallel blocksfft paths must agree with normvar=True.

    Both paths implement the same divide-by-variance step; neither was
    exercised with the flag on before this bead (two-paths-one-maintained).
    """
    rng = np.random.default_rng(42)
    q = rng.standard_normal((64, 4))
    kwargs = dict(nfft=16, nblocks=3, novlap=4, normvar=True, blockwise_mean=True)
    ser = blocksfft(q, use_parallel=False, **kwargs)
    par = blocksfft(q, use_parallel=True, **kwargs)
    np.testing.assert_allclose(
        ser,
        par,
        rtol=0,
        atol=1e-12,
        err_msg=f"serial/parallel disagree with normvar=True: max|diff|={np.max(np.abs(ser - par)):.3e}",
    )


def test_normvar_zero_variance_channel_isfinite():
    """Constant (zero-variance) channel is clamped, not Inf/NaN, under normvar.

    After mean subtraction a constant channel has variance 0; the 4*eps clamp
    sets the divisor to 1 so the channel matches the no-normvar path.
    """
    rng = np.random.default_rng(1)
    nfft = 32
    varying = rng.standard_normal(nfft)
    constant = np.full(nfft, 3.0)
    q = np.column_stack([varying, constant])

    with_nv = blocksfft(q, nfft=nfft, nblocks=1, novlap=0, normvar=True, use_parallel=False)
    without = blocksfft(q, nfft=nfft, nblocks=1, novlap=0, normvar=False, use_parallel=False)

    assert np.all(np.isfinite(with_nv)), "normvar produced non-finite values on a constant channel"
    # Constant channel: clamp divisor to 1 → identical to no-normvar path.
    np.testing.assert_allclose(with_nv[:, 1, :], without[:, 1, :], rtol=0, atol=1e-12)
    # Varying channel must actually change under normvar (else the flag is a no-op).
    assert not np.allclose(with_nv[:, 0, :], without[:, 0, :])
