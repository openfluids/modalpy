"""
GPU-accelerated batch FFT processing utilities for ModalPy.

=============================================================================
SETUP INSTRUCTIONS
=============================================================================

STEP 1: Verify NVIDIA Driver
    nvidia-smi
    # Should show your GPU (e.g., RTX 4060) and driver version

STEP 2: Install CUDA Toolkit (if not present)
    # Ubuntu/Debian:
    sudo apt install nvidia-cuda-toolkit

    # Verify:
    nvcc --version

STEP 3: Install CuPy
    # For CUDA 12.x:
    uv pip install cupy-cuda12x

    # For CUDA 11.x:
    uv pip install cupy-cuda11x

STEP 4: Verify Installation
    python -c "
    import cupy as cp
    x = cp.array([1,2,3,4])
    y = cp.fft.fft(x)
    print(f'GPU FFT OK: {cp.asnumpy(y)[:2]}')
    "

=============================================================================
WHEN TO USE GPU vs CPU
=============================================================================

Use GPU (CuPy) when:
    - Processing batches of 16+ FFTs simultaneously (GPU-resident mode)
    - Single FFTs larger than 256K points (with data transfer)
    - Data will stay on GPU for multiple operations (pipeline)

Use CPU (MKL/scipy) when:
    - Single small FFTs (<4K points)
    - Data needs immediate transfer back to CPU
    - Memory-constrained environments

Approximate speedups (RTX 4060) - GPU-RESIDENT mode (no transfer):
    Batch 256x16K:  ~12x faster than MKL
    Batch 256x128K: ~75x faster than MKL
    Batch 256x256K: ~100x faster than MKL

With data transfer (H2D + FFT + D2H):
    MKL typically wins due to PCIe transfer overhead (~0.5ms per transfer)
    GPU only wins for very large single FFTs (256K+) or when data stays on GPU

=============================================================================
USAGE
=============================================================================

    from modalpy.fft.gpu_utils import GPUBatchFFT, should_use_gpu

    # Check if GPU is beneficial for your workload
    if should_use_gpu(array_size=65536, batch_size=64):
        processor = GPUBatchFFT()
        result = processor.fft_batch(data)
    else:
        result = scipy.fft.fft(data)

=============================================================================
MEMORY MANAGEMENT (RTX 4060 = 8GB VRAM)
=============================================================================

This module limits GPU memory usage to 5GB by default, leaving 3GB for:
    - NVIDIA driver/context
    - cuFFT plan cache
    - OS overhead

To adjust memory limit:
    processor = GPUBatchFFT(memory_limit=6e9)  # 6GB

To clear memory:
    processor.clear_cache()

=============================================================================
"""

import numpy as np

# =============================================================================
# Configuration for RTX 4060 (8GB VRAM)
# =============================================================================

class GPUConfig:
    """Configuration for GPU FFT processing, tuned for RTX 4060.

    IMPORTANT: These thresholds assume GPU-RESIDENT mode (data stays on GPU).
    With data transfer (H2D + FFT + D2H), MKL typically wins due to PCIe overhead.

    Benchmark results (Dec 2025):
        - GPU-resident 4Kx64:   235x faster than MKL
        - GPU-resident 128Kx256: 75x faster than MKL
        - WITH transfer: MKL wins in most cases
    """

    # Memory limits
    VRAM_TOTAL = 8e9  # 8GB total VRAM
    VRAM_LIMIT = 5e9  # 5GB usable (leave 3GB for OS/driver/context)

    # Performance thresholds for GPU-RESIDENT mode (no transfer)
    # These are when GPU beats MKL assuming data is already on GPU
    BATCH_BREAKEVEN_RESIDENT = 16  # Batch size where GPU-resident wins
    SIZE_BREAKEVEN_RESIDENT = 4096  # FFT size where GPU-resident wins

    # Performance thresholds WITH data transfer
    # GPU only wins for very large FFTs due to PCIe overhead
    SIZE_BREAKEVEN_WITH_TRANSFER = 262144  # 256K+ for GPU to win with transfer

    # Legacy aliases (for backwards compatibility)
    BATCH_BREAKEVEN = BATCH_BREAKEVEN_RESIDENT
    SIZE_BREAKEVEN = SIZE_BREAKEVEN_RESIDENT
    SIZE_GPU_OPTIMAL = 65536  # Size where GPU clearly wins in resident mode

    # cuFFT plan cache (smaller = less memory, but more recompilation)
    PLAN_CACHE_SIZE = 64  # Default is 1024, reduced for 8GB VRAM


# =============================================================================
# GPU Availability Check
# =============================================================================

def gpu_available():
    """Check if CUDA GPU is available and working."""
    try:
        import cupy as cp
        # Try to access device
        device = cp.cuda.Device(0)
        _ = device.compute_capability
        return True
    except Exception:
        return False


def get_gpu_info():
    """Get GPU information if available."""
    try:
        import cupy as cp
        device = cp.cuda.Device(0)
        free, total = device.mem_info
        return {
            'available': True,
            'device_id': device.id,
            'compute_capability': device.compute_capability,
            'memory_total_gb': total / 1e9,
            'memory_free_gb': free / 1e9,
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e),
        }


def should_use_gpu(array_size, batch_size=1, prefer_gpu=True, gpu_resident=False):
    """Decide if GPU is beneficial for given workload.

    Args:
        array_size: Total number of elements in single FFT input
        batch_size: Number of FFTs to compute
        prefer_gpu: Whether to prefer GPU when beneficial
        gpu_resident: If True, data stays on GPU (no transfer overhead).
                      This is key - GPU wins with gpu_resident=True,
                      but MKL often wins when transfers are included.

    Returns:
        bool: True if GPU is recommended, False for CPU

    Note:
        With data transfer, MKL typically wins due to PCIe overhead.
        Set gpu_resident=True for pipelines where data stays on GPU.
    """
    if not prefer_gpu or not gpu_available():
        return False

    # GPU-resident mode: GPU wins for most batches
    if gpu_resident:
        if (batch_size >= GPUConfig.BATCH_BREAKEVEN_RESIDENT or
                array_size >= GPUConfig.SIZE_GPU_OPTIMAL):
            return True

    # With transfer: GPU only wins for very large single FFTs
    if array_size >= GPUConfig.SIZE_BREAKEVEN_WITH_TRANSFER:
        return True

    return False


# =============================================================================
# GPU Batch FFT Processor
# =============================================================================

class GPUBatchFFT:
    """Efficient batch FFT processor using CuPy.

    This class handles:
    - Efficient H2D/D2H memory transfers
    - cuFFT plan caching
    - Memory management for limited VRAM

    Example:
        processor = GPUBatchFFT()

        # Process batch of signals
        signals = np.random.randn(128, 4096)  # 128 signals, 4096 points each
        spectra = processor.fft_batch(signals, axis=1)

        # Keep data on GPU for pipeline operations
        gpu_data = processor.to_gpu(signals)
        gpu_spectra = processor.fft_gpu_resident(gpu_data, axis=1)
        # ... more GPU operations ...
        result = processor.to_cpu(gpu_spectra)
    """

    def __init__(self, memory_limit=None, plan_cache_size=None):
        """Initialize GPU processor.

        Args:
            memory_limit: Max GPU memory to use (bytes). Default: 5GB
            plan_cache_size: cuFFT plan cache size. Default: 64
        """
        self.memory_limit = memory_limit or GPUConfig.VRAM_LIMIT
        self.plan_cache_size = plan_cache_size or GPUConfig.PLAN_CACHE_SIZE

        self._init_gpu()

    def _init_gpu(self):
        """Initialize CuPy and configure settings."""
        try:
            import cupy as cp
            self.cp = cp

            # Configure plan cache (API varies by CuPy version)
            try:
                plan_cache = cp.fft.config.get_plan_cache()
                # Try newer API first
                if hasattr(plan_cache, 'set_size'):
                    plan_cache.set_size(self.plan_cache_size)
                elif hasattr(plan_cache, 'max_size'):
                    plan_cache.max_size = self.plan_cache_size
                # If neither exists, skip plan cache configuration
            except (AttributeError, TypeError):
                pass  # Plan cache config not available in this CuPy version

            # Configure memory pool - store as instance attribute for later access
            self._mempool = cp.cuda.MemoryPool()
            self._mempool.set_limit(size=int(self.memory_limit))
            cp.cuda.set_allocator(self._mempool.malloc)

            self._available = True
        except ImportError:
            self.cp = None
            self._available = False

    @property
    def available(self):
        """Check if GPU is available."""
        return self._available

    def to_gpu(self, data):
        """Transfer numpy array to GPU.

        Args:
            data: numpy array

        Returns:
            CuPy array on GPU
        """
        if not self._available:
            raise RuntimeError("GPU not available")
        return self.cp.asarray(data)

    def to_cpu(self, data):
        """Transfer GPU array back to CPU.

        Args:
            data: CuPy array

        Returns:
            numpy array
        """
        if not self._available:
            raise RuntimeError("GPU not available")
        return self.cp.asnumpy(data)

    def fft_batch(self, data, axis=-1):
        """Compute FFT on batch of signals (auto-transfer).

        Args:
            data: numpy array [batch, signal_length] or similar
            axis: Axis along which to compute FFT

        Returns:
            numpy array with FFT results
        """
        if not self._available:
            # Fallback to scipy
            from scipy.fft import fft
            return fft(data, axis=axis)

        gpu_data = self.to_gpu(data)
        gpu_result = self.cp.fft.fft(gpu_data, axis=axis)
        return self.to_cpu(gpu_result)

    def rfft_batch(self, data, axis=-1):
        """Compute real FFT on batch of real signals (more efficient).

        Args:
            data: numpy array of real values
            axis: Axis along which to compute FFT

        Returns:
            numpy array with one-sided FFT results
        """
        if not self._available:
            from scipy.fft import rfft
            return rfft(data, axis=axis)

        gpu_data = self.to_gpu(data)
        gpu_result = self.cp.fft.rfft(gpu_data, axis=axis)
        return self.to_cpu(gpu_result)

    def fft_gpu_resident(self, gpu_data, axis=-1):
        """Compute FFT on data already on GPU (no transfer).

        Args:
            gpu_data: CuPy array (must already be on GPU)
            axis: Axis along which to compute FFT

        Returns:
            CuPy array (stays on GPU)
        """
        if not self._available:
            raise RuntimeError("GPU not available")
        return self.cp.fft.fft(gpu_data, axis=axis)

    def ifft_gpu_resident(self, gpu_data, axis=-1):
        """Compute inverse FFT on data already on GPU.

        Args:
            gpu_data: CuPy array (must already be on GPU)
            axis: Axis along which to compute IFFT

        Returns:
            CuPy array (stays on GPU)
        """
        if not self._available:
            raise RuntimeError("GPU not available")
        return self.cp.fft.ifft(gpu_data, axis=axis)

    def memory_info(self):
        """Get current GPU memory usage."""
        if not self._available:
            return {'available': False}

        return {
            'used_bytes': self._mempool.used_bytes(),
            'total_bytes': self._mempool.total_bytes(),
            'limit_bytes': int(self.memory_limit),
        }

    def clear_cache(self):
        """Clear cuFFT plan cache and free unused memory."""
        if not self._available:
            return

        # Clear plan cache (API varies by CuPy version)
        try:
            plan_cache = self.cp.fft.config.get_plan_cache()
            plan_cache.clear()
        except (AttributeError, TypeError):
            pass  # Plan cache API not available in this CuPy version

        # Free unused memory from our pool
        self._mempool.free_all_blocks()


# =============================================================================
# Convenience Functions
# =============================================================================

def gpu_fft(data, axis=-1):
    """One-shot GPU FFT with auto-transfer.

    Args:
        data: numpy array
        axis: Axis along which to compute FFT

    Returns:
        numpy array with FFT result
    """
    processor = GPUBatchFFT()
    return processor.fft_batch(data, axis=axis)


def gpu_rfft(data, axis=-1):
    """One-shot GPU real FFT with auto-transfer.

    Args:
        data: numpy array of real values
        axis: Axis along which to compute FFT

    Returns:
        numpy array with one-sided FFT result
    """
    processor = GPUBatchFFT()
    return processor.rfft_batch(data, axis=axis)


# =============================================================================
# Benchmarking Utility
# =============================================================================

def benchmark_cpu_vs_gpu(sizes=None, batch_sizes=None, iterations=10):
    """Compare CPU vs GPU FFT performance.

    Args:
        sizes: List of FFT sizes to test. Default: [1024, 4096, 16384, 65536]
        batch_sizes: List of batch sizes. Default: [1, 16, 64, 128]
        iterations: Number of iterations per test

    Returns:
        dict with timing results
    """
    import time

    from scipy.fft import fft as scipy_fft

    if sizes is None:
        sizes = [1024, 4096, 16384, 65536]
    if batch_sizes is None:
        batch_sizes = [1, 16, 64, 128]

    results = {}
    processor = GPUBatchFFT()

    for size in sizes:
        for batch in batch_sizes:
            key = f'{size}x{batch}'
            data = np.random.randn(batch, size) + 1j * np.random.randn(batch, size)

            # CPU timing
            start = time.perf_counter()
            for _ in range(iterations):
                scipy_fft(data, axis=1)
            cpu_time = (time.perf_counter() - start) / iterations * 1000

            # GPU timing (if available)
            if processor.available:
                # Warmup
                try:
                    processor.fft_batch(data, axis=1)
                except Exception:
                    results[key] = {'cpu_ms': cpu_time, 'gpu_ms': 'N/A', 'speedup': 'N/A'}
                    continue

                start = time.perf_counter()
                for _ in range(iterations):
                    processor.fft_batch(data, axis=1)
                gpu_time = (time.perf_counter() - start) / iterations * 1000

                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                results[key] = {
                    'cpu_ms': round(cpu_time, 3),
                    'gpu_ms': round(gpu_time, 3),
                    'speedup': round(speedup, 2),
                }
            else:
                results[key] = {
                    'cpu_ms': round(cpu_time, 3),
                    'gpu_ms': 'N/A',
                    'speedup': 'N/A',
                }

    return results


if __name__ == '__main__':
    print('=== GPU FFT Utilities ===\n')

    # Check GPU
    print('GPU Info:')
    info = get_gpu_info()
    for k, v in info.items():
        print(f'  {k}: {v}')

    print('\n=== Quick Benchmark ===')
    if gpu_available():
        results = benchmark_cpu_vs_gpu(
            sizes=[4096, 16384, 65536],
            batch_sizes=[1, 64],
            iterations=10
        )
        print(f'{"Config":>15s} {"CPU (ms)":>10s} {"GPU (ms)":>10s} {"Speedup":>10s}')
        print('-' * 50)
        for key, vals in results.items():
            print(f'{key:>15s} {vals["cpu_ms"]:>10} {vals["gpu_ms"]:>10} {vals["speedup"]:>10}')
    else:
        print('GPU not available for benchmarking')
