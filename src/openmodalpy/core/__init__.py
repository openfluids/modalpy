"""Core utilities for OpenModalPy."""

from openmodalpy.core.base import BaseAnalyzer
from openmodalpy.core.config import (
    FFT_BACKEND,
    FIG_DPI,
    FIGURES_DIR_BSMD,
    FIGURES_DIR_DMD,
    FIGURES_DIR_POD,
    FIGURES_DIR_SPOD,
    RESULTS_DIR_BSMD,
    RESULTS_DIR_DMD,
    RESULTS_DIR_POD,
    RESULTS_DIR_SPOD,
)
from openmodalpy.core.provenance import collect_provenance
from openmodalpy.core.results import AnalysisResults, read_results, write_results

# decomposition is imported as a submodule (openmodalpy.core.decomposition)
# so analyzers can `from openmodalpy.core import decomposition`.

__all__ = [
    "BaseAnalyzer",
    "AnalysisResults",
    "read_results",
    "write_results",
    "collect_provenance",
    "FFT_BACKEND",
    "FIG_DPI",
    "RESULTS_DIR_POD",
    "RESULTS_DIR_DMD",
    "RESULTS_DIR_SPOD",
    "RESULTS_DIR_BSMD",
    "FIGURES_DIR_POD",
    "FIGURES_DIR_DMD",
    "FIGURES_DIR_SPOD",
    "FIGURES_DIR_BSMD",
]
