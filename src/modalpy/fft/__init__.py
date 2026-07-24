"""
FFT module for ModalPy with multi-backend FFT utilities.

Available backends: scipy (default), numpy, mkl, cupy, torch, tensorflow

Quick usage:
    from modalpy.fft import get_fft_func, get_available_backends
    backends = get_available_backends()
    fft_func = get_fft_func('mkl')  # or 'scipy', 'cupy', etc.
    spectrum = fft_func(signal)

For GPU batch processing:
    from modalpy.fft import GPUBatchFFT
    processor = GPUBatchFFT()
    result = processor.fft_batch(batch_signals, axis=1)
"""

from .fft_backends import (
    get_fft_func,
    get_fft_backend_names,
    get_available_backends,
    get_optimal_backend,
    benchmark_backends,
    gpu_available,
    mkl_available,
    scipy_fft,
    numpy_fft,
    mkl_fft,
    cupy_fft,
    register_mkl_scipy_backend,
)

from .gpu_utils import (
    GPUBatchFFT,
    GPUConfig,
    should_use_gpu,
    get_gpu_info,
    gpu_fft,
    gpu_rfft,
    benchmark_cpu_vs_gpu,
)

from .spectral_utils import (
    periodogram_rfft,
    blackman_tukey_rfft,
    welch_method,
    find_peaks,
    calculate_error,
)

from .complex_signal import generate_complex_signal

__all__ = [
    # Backend selection
    'get_fft_func',
    'get_fft_backend_names',
    'get_available_backends',
    'get_optimal_backend',
    'benchmark_backends',
    'gpu_available',
    'mkl_available',
    'register_mkl_scipy_backend',
    # FFT functions
    'scipy_fft',
    'numpy_fft',
    'mkl_fft',
    'cupy_fft',
    # GPU utilities
    'GPUBatchFFT',
    'GPUConfig',
    'should_use_gpu',
    'get_gpu_info',
    'gpu_fft',
    'gpu_rfft',
    'benchmark_cpu_vs_gpu',
    # Spectral utilities
    'periodogram_rfft',
    'blackman_tukey_rfft',
    'welch_method',
    'find_peaks',
    'calculate_error',
    # Signal generation
    'generate_complex_signal',
]
