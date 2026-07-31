#!/usr/bin/env python3
"""
Simple Parallel Utilities for Modal Decomposition Analysis

This module provides optimized implementations using vectorized NumPy and
high-performance BLAS routines. OpenMP and Numba are no longer required; the
functions run on any standard Python installation.

Author: Modal Decomposition Team
"""

import logging
import multiprocessing

import numpy as np
from scipy.signal import get_window
from threadpoolctl import threadpool_info

from openmodalpy.core.welch import _validate_welch_blocks

logger = logging.getLogger(__name__)

# OpenMP support was removed. All routines rely on NumPy vectorization and the
# underlying BLAS implementation.
OPENMP_AVAILABLE = False
PARALLEL_AVAILABLE = True


def calculate_polar_weights_optimized(x, y):
    """
    Calculate integration weights for 2D cylindrical grid.

    This function uses a fully vectorized NumPy implementation that works on any
    platform without special dependencies.

    Parameters:
    -----------
    x : np.ndarray
        Axial coordinates
    y : np.ndarray
        Radial coordinates

    Returns:
    --------
    np.ndarray
        Integration weights, shape (Nx * Ny, 1)
    """
    return _calculate_weights_numpy(x, y)


def _calculate_weights_numpy(x, y):
    """Vectorized NumPy implementation of polar weights."""
    Nx, Ny = len(x), len(y)

    # Calculate y-direction (r-direction) integration weights (Wy) - vectorized
    Wy = np.zeros(Ny)

    if Ny > 1:
        # First point (centerline)
        y_mid_right = (y[0] + y[1]) / 2
        Wy[0] = np.pi * y_mid_right**2

        # Middle points - vectorized
        if Ny > 2:
            y_mid_left = (y[:-2] + y[1:-1]) / 2
            y_mid_right = (y[1:-1] + y[2:]) / 2
            Wy[1:-1] = np.pi * (y_mid_right**2 - y_mid_left**2)

        # Last point
        y_mid_left = (y[-2] + y[-1]) / 2
        Wy[-1] = np.pi * (y[-1] ** 2 - y_mid_left**2)
    else:
        Wy[0] = np.pi * y[0] ** 2

    # Calculate x-direction integration weights (Wx) - vectorized
    Wx = np.zeros(Nx)

    if Nx > 1:
        # First point
        Wx[0] = (x[1] - x[0]) / 2

        # Middle points - vectorized
        if Nx > 2:
            Wx[1:-1] = (x[2:] - x[:-2]) / 2

        # Last point
        Wx[-1] = (x[-1] - x[-2]) / 2
    else:
        Wx[0] = 1.0

    # Combine weights using outer product (much faster than loops)
    W = np.outer(Wx, Wy).flatten()

    return W.reshape(-1, 1)


# Placeholder function maintained for backward compatibility. It simply calls
# the NumPy implementation as OpenMP acceleration has been removed.
def _calculate_weights_openmp(x, y):
    return _calculate_weights_numpy(x, y)


def blocksfft_optimized(
    q, nfft, nblocks, novlap, blockwise_mean=False, normvar=False, window_norm="power", window_type="hamming"
):
    """
    Optimized blocked FFT computation.

    This function uses the best available linear algebra backend (BLAS/LAPACK)
    and optimized memory access patterns for better performance.

    Parameters:
    -----------
    q : np.ndarray
        Input data [time, space]
    nfft : int
        Number of FFT points
    nblocks : int
        Number of blocks
    novlap : int
        Number of overlapping points between blocks
    blockwise_mean : bool
        Subtract blockwise mean if True
    normvar : bool
        If True, divide each block pointwise in space by its variance
        (unbiased, ``ddof=1``), matching ``spod_matlab`` (``opts.normvar``)
        and PySPOD (``normalize_data``). This does **not** produce unit
        variance and is therefore scale-dependent: scaling the input by
        ``c`` scales the normalized block by ``1/c``. Values below
        ``4*eps`` are clamped to 1. Implementation option, not a step in
        Towne, Schmidt & Colonius (2018). Defaults to False.
    window_norm : str
        Window normalization type ('amplitude' or 'power')
    window_type : str
        Window type. Use 'sine' for the custom sine window or any name
        recognized by ``scipy.signal.get_window`` (periodic / fftbins=True).

    Returns:
    --------
    np.ndarray
        FFT coefficients [freq, space, block]

    Notes
    -----
    Block starts are ``iblk * (nfft - novlap)`` with no end-of-record clamp.
    Callers must pass an ``nblocks`` that fits; oversize requests raise
    ``ValueError`` rather than re-using trailing samples.
    """
    _validate_welch_blocks(q.shape[0], nfft, nblocks, novlap)

    # Import FFT backend
    from fftkit import get_fft_func

    # Select window function — PERIODIC convention (get_window fftbins=True)
    # so both FFT paths agree with Welch spectral estimation practice.
    if window_type == "sine":
        window = np.sin(np.pi * (np.arange(nfft) + 0.5) / nfft)
    else:
        window = get_window(window_type, nfft, fftbins=True)

    # Normalize window
    if window_norm == "amplitude":
        cw = 1.0 / window.mean()
    else:  # 'power' normalization (default)
        cw = 1.0 / np.sqrt(np.mean(window**2))

    nmesh = q.shape[1]  # Number of spatial points (Nx * Ny)
    n_freq_out = nfft // 2 + 1  # Number of frequency bins for one-sided spectrum
    q_hat = np.zeros((n_freq_out, nmesh, nblocks), dtype=complex)
    q_mean = np.mean(q, axis=0)  # Temporal mean (long-time mean)
    window_broadcast = window[:, np.newaxis]  # Reshape window for broadcasting

    # Process each block with optimized memory access
    fft_func = get_fft_func()
    hop = nfft - novlap

    for iblk in range(nblocks):
        ts = iblk * hop  # no end-clamp: validated above that every block fits
        tf = np.arange(ts, ts + nfft)  # Time indices for the block
        block = q[tf, :]

        # Subtract mean
        if blockwise_mean:
            block_mean = np.mean(block, axis=0)
        else:
            block_mean = q_mean
        block_centered = block - block_mean

        # Normalize variance if requested
        if normvar:
            block_var = np.var(block_centered, axis=0, ddof=1)
            block_var[block_var < 4 * np.finfo(float).eps] = 1.0  # Avoid division by zero
            block_centered = block_centered / block_var

        # Apply window and FFT
        windowed_block = block_centered * window_broadcast

        # Compute full FFT (uses optimized BLAS/LAPACK routines)
        full_fft_result = fft_func(windowed_block, axis=0)

        # Store only the one-sided spectrum (first n_freq_out points)
        q_hat[:, :, iblk] = (cw / nfft) * full_fft_result[:n_freq_out, :]

    return q_hat


def spod_single_frequency_optimized(qhat, w, nblocks, dst, num_modes=None, return_psi=False):
    """Single-frequency SPOD via the shared eigenproblem body.

    Thin wrapper around ``decomposition.spod_single_frequency``. Threading /
    BLAS setup for this path (if any) stays here; the algorithm does not.
    """
    # Late import avoids a parallel → decomposition → base cycle at module load.
    from openmodalpy.core.decomposition import spod_single_frequency

    return spod_single_frequency(
        qhat,
        nblocks,
        dst,
        w,
        num_modes=num_modes,
        return_psi=return_psi,
    )


def get_optimization_info():
    """Return information about available optimizations."""
    info = {"parallel_available": PARALLEL_AVAILABLE, "cpu_count": multiprocessing.cpu_count(), "numpy_blas": "Unknown"}

    # Try to detect BLAS implementation
    try:
        from contextlib import redirect_stdout
        from io import StringIO

        import numpy as np

        # Capture config output instead of printing it
        with redirect_stdout(StringIO()) as config_output:
            np.__config__.show()
        config_info = config_output.getvalue()

        if "mkl" in str(config_info).lower():
            info["numpy_blas"] = "Intel MKL"
        elif "openblas" in str(config_info).lower():
            info["numpy_blas"] = "OpenBLAS"
        elif "atlas" in str(config_info).lower():
            info["numpy_blas"] = "ATLAS"
        else:
            info["numpy_blas"] = "Standard"
    except Exception:
        info["numpy_blas"] = "Unknown"

    return info


def get_threadpool_summary():
    """Return a short description of active thread pools."""
    try:
        pools = threadpool_info()
        return (
            ", ".join(f"{p.get('prefix', '')}{p.get('internal_api')}={p.get('num_threads')}" for p in pools) or "none"
        )
    except Exception:
        return "unavailable"


def print_optimization_status():
    """Log current optimization status."""
    info = get_optimization_info()

    logger.info("Optimization Status:")
    logger.info("Parallel Available: %s", info["parallel_available"])
    logger.info("CPU Cores: %s", info["cpu_count"])
    logger.info("NumPy BLAS: %s", info["numpy_blas"])
    logger.info("Thread pools: %s", get_threadpool_summary())

    if info["parallel_available"]:
        logger.info("High performance mode (vectorized)")
    else:
        logger.info("Standard performance mode")
