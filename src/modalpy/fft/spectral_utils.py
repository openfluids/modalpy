import numpy as np
from scipy import signal

try:
    from scipy.fft import rfft, rfftfreq  # SciPy >= 1.4.0
except ImportError:
    from scipy.fftpack import rfft, rfftfreq  # SciPy < 1.4.0


def periodogram_rfft(x, fs):
    """Compute PSD using periodogram with real FFT."""
    freqs, psd = signal.periodogram(x, fs, scaling='density')
    return freqs, psd


def blackman_tukey_rfft(x, fs):
    """Compute PSD using the Blackman-Tukey method."""
    N = len(x)
    autocorr = signal.correlate(x, x, mode='full') / N
    autocorr = autocorr[N - 1:]
    X = rfft(autocorr)
    freqs = rfftfreq(N, 1 / fs)
    psd = np.abs(X) / fs
    return freqs, psd


def welch_method(x, fs, nperseg=None):
    """Compute PSD using Welch's method with segment averaging.

    Args:
        x: Input signal
        fs: Sampling frequency
        nperseg: Segment length for averaging. Default None uses scipy's default
                 (256 or len(x) if shorter). Use larger values for finer
                 frequency resolution, smaller for more averaging/smoother PSD.

    Returns:
        freqs: Frequency array
        psd: Power spectral density estimate
    """
    # scipy default is 256, which gives good averaging for most signals
    freqs, psd = signal.welch(x, fs, nperseg=nperseg, scaling='density')
    return freqs, psd


def find_peaks(st, psd, threshold=0.01):
    """Return peak locations and values above a fraction of the maximum."""
    peak_indices = signal.find_peaks(psd, height=max(psd) * threshold)[0]
    return st[peak_indices], psd[peak_indices]


def calculate_error(detected_peaks, true_peaks):
    """Return mean absolute error between detected and true peaks."""
    errors = []
    for true_peak in true_peaks:
        if len(detected_peaks) > 0:
            error = min(abs(detected_peak - true_peak) for detected_peak in detected_peaks)
            errors.append(error)
        else:
            errors.append(true_peak)
    return np.mean(errors)
