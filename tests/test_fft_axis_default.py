"""Executable guard: openmodalpy never hits fftkit's AxisDefaultWarning.

fftkit warns (``AxisDefaultWarning``) when FFT entry points omit ``axis`` and
rely on the library default. openmodalpy's FFT call sites pass ``axis=0``
explicitly; this test is the source of truth for that claim (replacing the
stale pyproject comment that once said "verified: 94 passed").
"""

from __future__ import annotations

import warnings

import numpy as np
from fftkit import AxisDefaultWarning

from openmodalpy import SPODAnalyzer


def test_full_spod_fft_pipeline_raises_no_axis_default_warning(tmp_path):
    """Run a real analyzer FFT path and assert AxisDefaultWarning never fires."""
    rng = np.random.default_rng(0)
    Ns, Nx, Ny = 64, 8, 4
    q = rng.standard_normal((Ns, Nx * Ny))
    data = {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "dt": 0.1,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }
    analyzer = SPODAnalyzer(
        file_path="axis_default_guard",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        nfft=16,
        overlap=0.5,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )

    with warnings.catch_warnings(record=True) as caught:
        # Promote only this category so other intentional notices elsewhere
        # cannot mask a miss here; filterwarnings=error already covers the suite.
        warnings.simplefilter("always", AxisDefaultWarning)
        analyzer.load_and_preprocess()
        analyzer.compute_fft_blocks()
        analyzer.perform_spod()

    axis_defaults = [w for w in caught if issubclass(w.category, AxisDefaultWarning)]
    assert axis_defaults == [], (
        f"AxisDefaultWarning fired {len(axis_defaults)} time(s) on the SPOD FFT "
        f"path: {[str(w.message) for w in axis_defaults]}"
    )
    assert analyzer.qhat is not None
    assert analyzer.qhat.size > 0
