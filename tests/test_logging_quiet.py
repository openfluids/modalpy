"""Silence checks for the logging seam (phase 1: core/ + perform_spod guard).

Under default logging configuration the library must not write to stdout.
CLI installs its own handler; library import paths do not.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from openmodalpy import SPODAnalyzer
from openmodalpy.core.base import BaseAnalyzer, print_summary
from openmodalpy.core.io import MATDataLoader
from openmodalpy.core.parallel import print_optimization_status


def _synthetic_data(Ns: int = 32, Nspace: int = 4, dt: float = 1.0) -> dict:
    np.random.seed(0)
    Nx = int(np.sqrt(Nspace))
    Ny = Nspace // Nx
    return {
        "q": np.random.randn(Ns, Nspace),
        "x": np.linspace(0.0, 1.0, Nx),
        "y": np.linspace(0.0, 1.0, Ny),
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }


def _make_base(tmp_path, data: dict) -> BaseAnalyzer:
    """BaseAnalyzer only — no analyzer-module prints (those are phase 2)."""
    return BaseAnalyzer(
        file_path="dummy.h5",
        nfft=8,
        overlap=0.0,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_parallel=False,
    )


def test_load_and_preprocess_is_quiet_on_stdout(tmp_path, capsys):
    """BaseAnalyzer.load_and_preprocess logs shapes; stdout stays empty."""
    analyzer = _make_base(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_load_and_preprocess_still_says_something(tmp_path, caplog):
    """Silence must come from routing, not from deleting the diagnostics.

    The stdout checks above would also pass if every print AND every logger call
    were simply removed. This pins the other half: the same call still reports
    what it loaded, on the module logger, at INFO.
    """
    analyzer = _make_base(tmp_path, _synthetic_data())
    with caplog.at_level(logging.INFO, logger="openmodalpy.core.base"):
        analyzer.load_and_preprocess()

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info, "load_and_preprocess emitted no INFO records at all"
    assert any("Data loaded" in r.getMessage() for r in info), [r.getMessage() for r in info]


def test_compute_fft_blocks_is_quiet_on_stdout(tmp_path, capsys):
    """BaseAnalyzer.compute_fft_blocks logs timings; stdout stays empty."""
    analyzer = _make_base(tmp_path, _synthetic_data())
    analyzer.load_and_preprocess()
    capsys.readouterr()
    analyzer.compute_fft_blocks()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_print_summary_and_optimization_status_are_quiet(capsys):
    """core helpers that used to print now go through the module logger."""
    print_summary("POD", "/tmp/results", "/tmp/figures")
    print_optimization_status()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_mat_loader_is_quiet_on_stdout(tmp_path, capsys):
    """MATDataLoader.load logs progress; stdout stays empty under default logging."""
    import h5py

    file_path = tmp_path / "quiet.mat"
    q = np.arange(12, dtype=float).reshape(6, 2)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("u", data=q)
        f.create_dataset("x", data=np.array([0.0, 0.5, 1.0]))
        f.create_dataset("y", data=np.array([0.0, 1.0]))
        f.create_dataset("dt", data=np.array([[0.25]]))

    data = MATDataLoader().load(str(file_path))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert data["Ns"] == 2


def test_perform_spod_raises_when_qhat_not_computed(tmp_path):
    """Calling perform_spod before FFT blocks exist raises RuntimeError.

    Previously this path printed one line and returned None, so the caller
    continued as if an analysis had run.
    """
    data = _synthetic_data()
    analyzer = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=8,
        overlap=0.0,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    assert analyzer.qhat is None or analyzer.qhat.size == 0

    with pytest.raises(RuntimeError, match=r"qhat not computed|compute_fft_blocks|run\(compute_fft"):
        analyzer.perform_spod()
