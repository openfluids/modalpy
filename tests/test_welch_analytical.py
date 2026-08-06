"""Analytical checks for the single windowed-block FFT convention.

These tests pin the scaling and window against independent references
(Parseval on a bin-centred tone, amplitude recovery, scipy.signal.welch).
They do not compare the two public entry points to each other.
"""

import numpy as np
from scipy.signal import get_window, welch

from openmodalpy.core.base import blocksfft


def _onesided_mean_square(q_hat):
    """Mean over blocks of |q_hat|^2, inner bins doubled (one-sided energy)."""
    ms = np.mean(np.abs(q_hat) ** 2, axis=-1)
    # ms shape: (nfreq, nmesh) — double positive-frequency bins, not DC/Nyquist.
    if ms.shape[0] > 2:
        ms = ms.copy()
        ms[1:-1] *= 2.0
    return ms


def test_parseval_bin_centred_tone_power_norm():
    """Power-norm one-sided mean-square spectrum sums to signal variance.

    fs=1000 Hz, nfft=256, f0=125 Hz (bin 32), hann, amplitude A=2 so var=A^2/2=2.
    """
    fs = 1000.0
    nfft = 256
    f0 = 125.0
    amp = 2.0
    # Integer number of periods in nfft so the tone sits on a bin with no leakage.
    t = np.arange(nfft) / fs
    x = amp * np.cos(2.0 * np.pi * f0 * t)
    q = x[:, np.newaxis]

    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        window_type="hann",
        window_norm="power",
        blockwise_mean=False,
    )

    peak_bin = int(round(f0 * nfft / fs))
    assert peak_bin == 32
    assert np.argmax(np.abs(q_hat[:, 0, 0])) == peak_bin

    ms = _onesided_mean_square(q_hat)
    total = float(np.sum(ms[:, 0]))
    variance = float(np.var(x))  # population var of pure tone = A^2/2
    # Bin-centred pure tone under power window: energy conserved to ~1e-15.
    np.testing.assert_allclose(total, variance, rtol=0, atol=1e-14)


def test_amplitude_norm_recovers_half_tone_amplitude():
    """With window_norm='amplitude', peak |q_hat| is half the cosine amplitude."""
    fs = 1000.0
    nfft = 256
    f0 = 125.0
    amp = 2.0
    t = np.arange(nfft) / fs
    x = amp * np.cos(2.0 * np.pi * f0 * t)
    q = x[:, np.newaxis]

    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        window_type="hann",
        window_norm="amplitude",
        blockwise_mean=False,
    )

    peak_bin = 32
    peak_mag = float(np.abs(q_hat[peak_bin, 0, 0]))
    np.testing.assert_allclose(peak_mag, amp / 2.0, rtol=0, atol=1e-12)


def test_scipy_welch_broadband_mean_square_matches_psd_times_df():
    """One-sided mean-square spectrum equals scipy.signal.welch density * df.

    Independent of our own FFT path: scipy builds the density, and we check that
    mean |q_hat|^2 (inner bins doubled) equals PSD * (fs / nfft).

    The agreement is exact, not approximate. Both sides reduce to
    2 |FFT(w*x)|^2 / (nfft * sum(w^2)) once scipy's 1/(fs * sum(w^2)) density
    scaling is multiplied by df = fs/nfft, so any real difference in window,
    hop, or scaling shows up immediately. Measured worst deviation on this
    case, DC included, is 1.1e-15; the assertion leaves five orders of slack
    for platform FFT noise and no more.
    """
    rng = np.random.default_rng(0)
    fs = 1000.0
    nfft = 256
    novlap = nfft // 2
    # Long enough for several Welch blocks under 50% overlap.
    n_samples = nfft + 7 * (nfft - novlap)
    # Zero-mean so our global-mean subtraction matches scipy detrend=False.
    x = rng.standard_normal(n_samples)
    x = x - x.mean()
    q = x[:, np.newaxis]
    nblocks = 1 + (n_samples - nfft) // (nfft - novlap)

    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=nblocks,
        novlap=novlap,
        window_type="hann",
        window_norm="power",
        blockwise_mean=False,
    )
    ms = _onesided_mean_square(q_hat)[:, 0]

    # scipy welch: density (power / Hz); floor partition, same hop and nfft.
    f_scipy, psd = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nfft,
        noverlap=novlap,
        detrend=False,
        return_onesided=True,
        scaling="density",
    )
    df = fs / nfft
    expected = psd * df

    # Same frequency grid.
    assert f_scipy.shape == ms.shape
    # Every bin, DC and Nyquist included. A loose bound here would let a
    # percent-level scaling error through, which is the error this test exists
    # to catch.
    np.testing.assert_allclose(ms, expected, rtol=1e-10, atol=0.0)


def test_power_norm_matches_manual_formula_on_boxcar():
    """q_hat = FFT(w*x) / (nfft * sqrt(mean(w^2))) with w=boxcar is FFT/nfft."""
    rng = np.random.default_rng(1)
    nfft = 64
    x = rng.standard_normal(nfft)
    x = x - x.mean()  # path always subtracts the long-time mean
    q = x[:, np.newaxis]
    q_hat = blocksfft(
        q,
        nfft=nfft,
        nblocks=1,
        novlap=0,
        window_type="boxcar",
        window_norm="power",
        blockwise_mean=False,
    )
    # boxcar periodic window is all ones; power scale is 1.
    w = get_window("boxcar", nfft, fftbins=True)
    assert np.allclose(w, 1.0)
    manual = np.fft.fft(x)[: nfft // 2 + 1] / nfft
    np.testing.assert_allclose(q_hat[:, 0, 0], manual, rtol=0, atol=1e-14)
