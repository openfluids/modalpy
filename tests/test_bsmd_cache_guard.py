"""BSMD FFT-cache OSError guards cover reads only.

A failed cache WRITE must propagate and must not be printed as a load failure.
A failed cache READ must still recompute and name the file.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import openmodalpy.bsmd as bsmd_mod
from openmodalpy import BSMDAnalyzer, SPODAnalyzer

NFFT = 8
NS = 32
NSPACE = 4


def _data() -> dict:
    q = np.random.default_rng(0).standard_normal((NS, NSPACE))
    return {
        "q": q,
        "x": np.linspace(0.0, 1.0, NSPACE),
        "y": np.linspace(0.0, 1.0, 1),
        "dt": 1.0,
        "Nx": NSPACE,
        "Ny": 1,
        "Ns": NS,
    }


def _spod(results_dir) -> SPODAnalyzer:
    an = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=NFFT,
        overlap=0.0,
        results_dir=str(results_dir),
        figures_dir=str(results_dir),
        data_loader=lambda _: _data(),
        spatial_weight_type="uniform",
    )
    an.load_and_preprocess()
    an.compute_fft_blocks()
    return an


def _bsmd(results_dir) -> BSMDAnalyzer:
    return BSMDAnalyzer(
        file_path="dummy.h5",
        nfft=NFFT,
        overlap=0.0,
        results_dir=str(results_dir),
        figures_dir=str(results_dir),
        data_loader=lambda _: _data(),
        spatial_weight_type="uniform",
        use_static_triads=True,
        static_triads=[(0, 0, 0)],
        use_parallel=False,
    )


def test_write_failure_is_not_called_a_load_failure(tmp_path, monkeypatch, capsys):
    """A full disk while SAVING the BSMD cache must not print a load failure."""
    spod_dir = tmp_path / "spod"
    bsmd_dir = tmp_path / "bsmd"
    bsmd_dir.mkdir(parents=True, exist_ok=True)
    _spod(spod_dir)
    monkeypatch.setattr(bsmd_mod, "RESULTS_DIR_SPOD", str(spod_dir))

    real_file = h5py.File

    def fake_file(path, mode="r", *args, **kwargs):
        if str(path).startswith(str(bsmd_dir)) and mode != "r":
            raise OSError("No space left on device")
        return real_file(path, mode, *args, **kwargs)

    monkeypatch.setattr(bsmd_mod.h5py, "File", fake_file)

    analyzer = _bsmd(bsmd_dir)
    analyzer.load_and_preprocess()
    with pytest.raises(OSError, match="No space left on device"):
        analyzer.compute_fft_blocks()
    out = capsys.readouterr().out
    assert "Failed to load cached FFT blocks" not in out, (
        "a cache WRITE failure was reported as a cache LOAD failure:\n" + out
    )


def test_read_failure_still_recomputes_and_names_the_file(tmp_path, capsys):
    """A corrupt BSMD cache must still recompute, with the file named."""
    bsmd_dir = tmp_path / "bsmd"
    bsmd_dir.mkdir(parents=True, exist_ok=True)
    analyzer = _bsmd(bsmd_dir)
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    cache_path = analyzer._qhat_cache_path
    assert cache_path is not None

    with open(cache_path, "wb") as fh:
        fh.write(b"not an hdf5 file at all")

    fresh = _bsmd(bsmd_dir)
    fresh.load_and_preprocess()
    fresh.compute_fft_blocks()
    out = capsys.readouterr().out
    # The FAILURE line itself must name the file — a later "Saved FFT blocks to
    # cache at ..." line would otherwise satisfy a whole-output search.
    failure_lines = [line for line in out.splitlines() if "Failed to load cached FFT blocks" in line]
    assert len(failure_lines) == 1, out
    assert str(cache_path) in failure_lines[0]
    assert fresh.qhat_cached is False
    assert fresh.qhat.shape[0] == NFFT // 2 + 1
