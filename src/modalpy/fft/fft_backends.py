"""
Shared FFT backend selection and wrapper utilities for modal decomposition and benchmarking.

=============================================================================
INSTALLATION GUIDE FOR HIGH-PERFORMANCE FFT BACKENDS
=============================================================================

This module supports multiple FFT backends with different performance characteristics.
Below are installation instructions for each backend.

1. SCIPY (Default) - No extra installation needed
   - Included with scipy
   - Good general-purpose performance
   - pip install scipy

2. NUMPY - No extra installation needed
   - Included with numpy
   - Fallback option

3. INTEL MKL (Recommended for CPU) - 2-20x faster than scipy for large FFTs
   Installation:
       # Load Intel oneAPI environment first (if available)
       source /opt/intel/oneapi/setvars.sh  # or use alias like 'int25'

       # Install via uv (preferred)
       uv pip install mkl_fft mkl-service \\
           --index-url https://software.repos.intel.com/python/pypi \\
           --extra-index-url https://pypi.org/simple

       # Or via pip
       pip install mkl_fft mkl-service \\
           --index-url https://software.repos.intel.com/python/pypi \\
           --extra-index-url https://pypi.org/simple

   Verify:
       python -c "import mkl_fft; print('MKL FFT OK')"

   Performance (typical speedups vs scipy):
       1K points:   ~2x faster
       64K points:  ~5-6x faster
       256K points: ~5x faster

4. CUPY (GPU - NVIDIA CUDA) - Up to 100x faster for batch operations
   Prerequisites:
       # NVIDIA driver must be installed (check with: nvidia-smi)
       # CUDA toolkit must be installed:
       sudo apt install nvidia-cuda-toolkit  # Ubuntu/Debian
       # or download from: https://developer.nvidia.com/cuda-downloads

   Installation:
       # For CUDA 12.x (check version with: nvcc --version)
       uv pip install cupy-cuda12x

       # For CUDA 11.x
       uv pip install cupy-cuda11x

   Verify:
       python -c "import cupy as cp; x = cp.array([1,2,3]); print(f'CuPy OK, device: {cp.cuda.Device(0).id}')"

   When to use GPU:
       - Batch processing (64+ FFTs at once): GPU is much faster
       - Large single FFTs (>64K points): GPU wins
       - Small single FFTs (<4K): CPU is often faster due to transfer overhead

5. PYTORCH (torch) - Good for ML pipelines
   Installation:
       pip install torch  # CPU only
       pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4

6. TENSORFLOW - High overhead, use for TF pipelines only
   Installation:
       pip install tensorflow

7. PYFFTW - FFTW wrapper (optional)
   Installation:
       pip install pyfftw

8. ACCELERATE - macOS only (Apple Silicon optimized)
   - Uses Apple's Accelerate framework via ctypes
   - Only works on macOS

=============================================================================
ENVIRONMENT VARIABLES
=============================================================================

PYMODAL_FFT_BACKEND: Override default FFT backend
    export PYMODAL_FFT_BACKEND=mkl  # Use MKL by default

OMP_NUM_THREADS: Control OpenMP thread count for MKL
    export OMP_NUM_THREADS=4

MKL_NUM_THREADS: Control MKL-specific thread count
    export MKL_NUM_THREADS=4

=============================================================================
QUICK BENCHMARK
=============================================================================

To compare backends on your system:

    from modalpy.fft.fft_backends import benchmark_backends, get_available_backends
    print(f"Available: {get_available_backends()}")
    results = benchmark_backends(size=65536, iterations=50)
    for name, time_ms in sorted(results.items(), key=lambda x: x[1] if isinstance(x[1], float) else 999):
        print(f"  {name}: {time_ms:.3f} ms" if isinstance(time_ms, float) else f"  {name}: {time_ms}")

=============================================================================
"""

import sys

from modalpy.core.config import FFT_BACKEND


def accelerate_fft(x, axis=0):
    """FFT using Apple's Accelerate framework via PyObjC or ctypes."""
    import ctypes
    import ctypes.util

    import numpy as np

    if sys.platform != 'darwin':
        raise NotImplementedError('Accelerate FFT is only available on macOS.')

    lib_path = ctypes.util.find_library('Accelerate')
    if lib_path is None:
        raise RuntimeError('Accelerate framework not found.')
    accel = ctypes.cdll.LoadLibrary(lib_path)

    class DSPDoubleSplitComplex(ctypes.Structure):
        _fields_ = [
            ('realp', ctypes.POINTER(ctypes.c_double)),
            ('imagp', ctypes.POINTER(ctypes.c_double)),
        ]

    x_arr = np.asarray(x, dtype=np.complex128)
    n = x_arr.shape[axis]
    log2n = int(np.log2(n))
    if 2**log2n != n:
        raise ValueError('vDSP FFT requires power-of-two length.')

    real = np.ascontiguousarray(np.real(x_arr), dtype=np.float64)
    imag = np.ascontiguousarray(np.imag(x_arr), dtype=np.float64)
    split = DSPDoubleSplitComplex(real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    accel.vDSP_create_fftsetupD.restype = ctypes.c_void_p
    setup = accel.vDSP_create_fftsetupD(ctypes.c_uint(log2n), 2)
    if not setup:
        raise RuntimeError('Failed to create FFT setup.')
    accel.vDSP_fft_zipD(ctypes.c_void_p(setup), ctypes.byref(split), 1, ctypes.c_uint(log2n), 1)
    accel.vDSP_destroy_fftsetupD(ctypes.c_void_p(setup))
    return real + 1j * imag


def scipy_fft(x, axis=0):
    try:
        from scipy.fft import fft
    except ImportError:
        from scipy.fftpack import fft
    return fft(x, axis=axis)


def numpy_fft(x, axis=0):
    from numpy.fft import fft

    return fft(x, axis=axis)


def tensorflow_fft(x, axis=0):
    import tensorflow as tf

    x_tf = tf.convert_to_tensor(x)
    x_tf_complex = tf.cast(x_tf, tf.complex64)
    return tf.signal.fft(x_tf_complex).numpy()


def torch_fft(x, axis=0):
    import torch

    x_torch = torch.from_numpy(x)
    x_torch_complex = x_torch.type(torch.complex64)
    return torch.fft.fft(x_torch_complex, dim=axis).numpy()


def _pyfftw_fft_impl(x, axis=0):
    import numpy as np
    import pyfftw

    # Match dtype: if float64 or complex128, use complex128; else use complex64
    if np.issubdtype(x.dtype, np.floating):
        dtype = np.complex128 if x.dtype == np.float64 else np.complex64
        x = x.astype(dtype)
    elif x.dtype == np.complex128 or x.dtype == np.complex64:
        dtype = x.dtype
    else:
        dtype = np.complex64
        x = x.astype(dtype)
    a = pyfftw.empty_aligned(x.shape, dtype=dtype)
    a[:] = x
    fft_object = pyfftw.builders.fft(a, axis=axis)
    return fft_object()


def mkl_fft(x, axis=0):
    """FFT via Intel MKL (2-20x faster for large arrays).

    Requires mkl_fft package. Install with:
        uv pip install mkl_fft --index-url https://software.repos.intel.com/python/pypi
    """
    try:
        from mkl_fft import fft as mkl_fft_func
    except ImportError as e:
        raise ImportError(
            "mkl_fft not installed. Install with:\n"
            "uv pip install mkl_fft --index-url https://software.repos.intel.com/python/pypi "
            "--extra-index-url https://pypi.org/simple"
        ) from e
    return mkl_fft_func(x, axis=axis)


def register_mkl_scipy_backend():
    """Set MKL as the global scipy.fft backend for all scipy.fft calls.

    After calling this, all scipy.fft operations will use MKL automatically.
    """
    try:
        import mkl_fft.interfaces.scipy_fft
        from scipy.fft import set_global_backend
        set_global_backend(mkl_fft.interfaces.scipy_fft)
        return True
    except ImportError:
        return False


# =============================================================================
# GPU Backends (CuPy / PyTorch CUDA)
# =============================================================================

def cupy_fft(x, axis=0):
    """FFT using NVIDIA GPU via CuPy (auto-transfer mode).

    Accepts numpy array, transfers to GPU, computes FFT, returns numpy array.
    For batch operations or pipelines, use cupy_fft_gpu_resident() instead.

    Requires: pip install cupy-cuda12x (for CUDA 12.x)
    """
    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError(
            "CuPy not installed. Install with:\n"
            "uv pip install cupy-cuda12x"
        ) from e

    # Transfer to GPU if numpy array
    if not isinstance(x, cp.ndarray):
        x_gpu = cp.asarray(x)
    else:
        x_gpu = x

    # Compute FFT on GPU
    result_gpu = cp.fft.fft(x_gpu, axis=axis)

    # Transfer back to CPU as numpy
    return cp.asnumpy(result_gpu)


def cupy_fft_gpu_resident(x_gpu, axis=0):
    """FFT for data already on GPU (no transfer overhead).

    Args:
        x_gpu: CuPy array (must already be on GPU)
        axis: Axis along which to compute FFT

    Returns:
        CuPy array (stays on GPU)
    """
    import cupy as cp
    if not isinstance(x_gpu, cp.ndarray):
        raise TypeError("Input must be a CuPy array. Use cupy_fft() for numpy arrays.")
    return cp.fft.fft(x_gpu, axis=axis)


def cupy_rfft(x, axis=0):
    """Real FFT using NVIDIA GPU via CuPy (more efficient for real data)."""
    try:
        import cupy as cp
    except ImportError as e:
        raise ImportError("CuPy not installed. Install with: uv pip install cupy-cuda12x") from e

    if not isinstance(x, cp.ndarray):
        x_gpu = cp.asarray(x)
    else:
        x_gpu = x

    result_gpu = cp.fft.rfft(x_gpu, axis=axis)
    return cp.asnumpy(result_gpu)


def torch_cuda_fft(x, axis=0):
    """FFT using PyTorch on CUDA GPU.

    Useful when integrating with PyTorch deep learning pipelines.
    """
    import torch

    if not isinstance(x, torch.Tensor):
        x_torch = torch.from_numpy(x).to('cuda')
    else:
        x_torch = x.to('cuda') if not x.is_cuda else x

    result = torch.fft.fft(x_torch, dim=axis)
    return result.cpu().numpy()


# =============================================================================
# Backend Registry
# =============================================================================

FFT_BACKENDS = {
    # CPU backends
    'scipy': scipy_fft,
    'numpy': numpy_fft,
    'mkl': mkl_fft,
    'accelerate': accelerate_fft,
    # GPU backends
    'cupy': cupy_fft,
    'torch': torch_fft,
    'torch_cuda': torch_cuda_fft,
    'tensorflow': tensorflow_fft,
}

# Optionally enable PyFFTW if available
try:
    import pyfftw  # noqa: F401
    FFT_BACKENDS["pyfftw"] = _pyfftw_fft_impl
except ImportError:
    pass


# =============================================================================
# Backend Selection Utilities
# =============================================================================

def get_fft_func(backend=None):
    """Get FFT function for specified backend."""
    backend = backend or FFT_BACKEND
    if backend not in FFT_BACKENDS:
        raise ValueError(f"Unknown FFT_BACKEND: {backend}. Available: {list(FFT_BACKENDS.keys())}")
    return FFT_BACKENDS[backend]


def get_fft_backend_names():
    """Return list of all registered backend names."""
    return list(FFT_BACKENDS.keys())


def get_available_backends():
    """Return list of backends that are actually importable/working."""
    import numpy as np
    available = []
    test_signal = np.array([1, 2, 3, 4], dtype=np.complex128)

    for name in FFT_BACKENDS.keys():
        try:
            func = FFT_BACKENDS[name]
            result = func(test_signal)
            if result is not None and len(result) == len(test_signal):
                available.append(name)
        except Exception:
            pass
    return available


def gpu_available():
    """Check if CUDA GPU is available for CuPy."""
    try:
        import cupy as cp
        cp.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


def mkl_available():
    """Check if Intel MKL FFT is available."""
    try:
        import mkl_fft  # noqa: F401
        return True
    except ImportError:
        return False


def get_optimal_backend(array_size, batch_size=1, prefer_gpu=True, gpu_resident=False):
    """Select optimal backend based on workload characteristics.

    Args:
        array_size: Total number of elements in FFT input
        batch_size: Number of FFTs to compute (for batching decisions)
        prefer_gpu: Whether to prefer GPU when beneficial
        gpu_resident: If True, assume data stays on GPU (no transfer overhead)
                      This dramatically changes GPU vs CPU tradeoffs.

    Returns:
        str: Name of recommended backend ('scipy', 'mkl', 'cupy', etc.)

    Note:
        With data transfer included (gpu_resident=False), MKL typically wins
        due to PCIe overhead (~0.5ms per transfer). GPU only wins when:
        - Data stays on GPU (gpu_resident=True), OR
        - Very large single FFTs (256K+), OR
        - Multiple FFT operations in a pipeline
    """
    # GPU-resident mode: GPU wins for batches
    if gpu_resident and prefer_gpu and gpu_available():
        if batch_size >= 16 or array_size >= 16384:
            return 'cupy'

    # With transfer: GPU only wins for very large FFTs
    if prefer_gpu and gpu_available() and not gpu_resident:
        if array_size >= 262144:  # 256K+ single FFTs
            return 'cupy'

    # MKL is beneficial for most CPU FFTs (any size >= 1K)
    if mkl_available() and array_size >= 1024:
        return 'mkl'

    # Default to scipy
    return 'scipy'


def benchmark_backends(size=8192, iterations=100):
    """Quick benchmark of available backends.

    Returns dict of {backend_name: time_per_fft_ms}
    """
    import time

    import numpy as np

    results = {}
    test_signal = np.random.randn(size) + 1j * np.random.randn(size)

    for name in get_available_backends():
        try:
            func = FFT_BACKENDS[name]
            # Warmup
            func(test_signal)

            start = time.perf_counter()
            for _ in range(iterations):
                func(test_signal)
            elapsed = time.perf_counter() - start

            results[name] = (elapsed / iterations) * 1000  # ms per FFT
        except Exception as e:
            results[name] = f"Error: {e}"

    return results
