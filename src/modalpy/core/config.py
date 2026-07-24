"""
Central configuration for modal decomposition analysis.

NOTE: ALL imports are available here and this is imported in utils.py
so we only need to import utils in other files.
"""

import json
import os

os.environ["OS_ACTIVITY_MODE"] = "disable"  # suppress macOS IMKClient logs
"""
Configuration and shared imports for modal decomposition tools.
"""

# Default directories - organized by case, then method
# Structure: results/{case}/{method}/, figures/{case}/{method}/
RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"
CACHE_DIR = "./cache"

# Legacy analyzer-specific directories (for backwards compatibility)
# New code should use results/{case}/{method}/ pattern
RESULTS_DIR_SPOD = "./results"
RESULTS_DIR_POD = "./results"
RESULTS_DIR_BSMD = "./results"
RESULTS_DIR_DMD = "./results"
RESULTS_DIR_STPOD = "./results"

FIGURES_DIR_SPOD = "./figures"
FIGURES_DIR_POD = "./figures"
FIGURES_DIR_BSMD = "./figures"
FIGURES_DIR_DMD = "./figures"
FIGURES_DIR_STPOD = "./figures"

# Data directory structure
DATA_DIR = "./data"
DATA_DIR_CAVITY = "./data/cavity"
DATA_DIR_JET = "./data/jet"
DATA_DIR_DNAMIX = "./data/dnamix"

# No benchmark dataset is bundled with the public package. Direct module
# execution should always receive an explicit input path.
DEFAULT_DATA_FILE = None


def require_existing_data_path(data_file: str | None) -> str:
    """Validate an explicit input path for direct module execution."""
    if not data_file:
        raise ValueError(
            "No default input dataset is bundled with modalpy. "
            "Provide --data PATH or use a packaged generator-backed example."
        )
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Input data path does not exist: {data_file}")
    return data_file

# Figure saving options
FIG_DPI = 500
FIG_FORMAT = "png"  # or "pdf"

# FFT backend selection. Options include:
#  - 'scipy', 'numpy'
#  - 'tensorflow', 'torch'
#  - 'mkl' for Intel MKL via :mod:`mkl_fft`
#  - 'accelerate' for macOS vDSP/Accelerate
#  - 'cv2' (OpenCV)
# The name must match the keys defined in :mod:`fft.fft_backends`.
#
# Auto-detection priority: PYMODAL_FFT_BACKEND env var > MKL > scipy
def _detect_fft_backend():
    """Auto-detect best available FFT backend."""
    # 1. Check environment variable override
    env_backend = os.environ.get("PYMODAL_FFT_BACKEND")
    if env_backend:
        return env_backend.lower()

    # 2. Try MKL (2-10x faster than scipy on Intel CPUs)
    try:
        import mkl_fft
        return "mkl"
    except ImportError:
        pass

    # 3. Default to scipy (always available)
    return "scipy"

FFT_BACKEND = _detect_fft_backend()

# Matplotlib/LaTeX options
USE_LATEX = False  # Set True to enable LaTeX rendering
FONT_FAMILY = "serif"
FONT_SIZE = 12
CMAP_SEQ = "viridis"  # Sequential colormap for general use
CMAP_DIV = "RdBu_r"  # Diverging colormap for signed data

# Default window type for FFT
WINDOW_TYPE = "hamming"
WINDOW_NORM = "power"

# Other global options can be added here as needed


def load_config(config_path):
    """Load a JSON or YAML configuration file and override defaults.

    Parameters
    ----------
    config_path : str
        Path to the configuration file. Supported formats are JSON
        and YAML (requires ``PyYAML``).
    """

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found")

    _, ext = os.path.splitext(config_path)
    ext = ext.lower()

    with open(config_path, "r") as f:
        if ext in {".yml", ".yaml"}:
            try:
                import yaml
            except Exception as exc:  # pragma: no cover - import error path
                raise ImportError("PyYAML must be installed to read YAML configuration files") from exc
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a dictionary")

    # Update any matching globals using upper-case keys
    for key, value in config.items():
        key_upper = key.upper()
        if key_upper in globals():
            globals()[key_upper] = value
