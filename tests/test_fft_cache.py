"""Regression tests for the FFT block cache stamp/verify mechanism.

These tests target `BaseAnalyzer._qhat_cache_stamp`/`_write_qhat_stamp`/
`_verify_qhat_stamp` (src/openmodalpy/core/base.py) as exercised through
`SPODAnalyzer.compute_fft_blocks` (src/openmodalpy/spod.py). Before the fix,
the cache key/validation ignored window_type and the content of `q`, so a
cache hit could silently serve blocks computed under different parameters or
from a different dataset. Each test below documents (in its docstring) the
failure mode it would have hit against the pre-fix code.
"""

import h5py
import numpy as np

from openmodalpy import SPODAnalyzer
from openmodalpy.core.base import blocksfft


def _make_data(q, dt=1.0):
    Ns, Nspace = q.shape
    Nx = Nspace
    Ny = 1
    return {
        "q": q,
        "x": np.linspace(0, 1, Nx),
        "y": np.linspace(0, 1, Ny),
        "dt": dt,
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }


def _make_spod(tmp_path, q, *, nfft=8, overlap=0.0, window_type="hamming", file_path="dummy.h5"):
    data = _make_data(q)
    analyzer = SPODAnalyzer(
        file_path=file_path,
        nfft=nfft,
        overlap=overlap,
        window_type=window_type,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    return analyzer


def test_window_type_change_produces_different_qhat(tmp_path):
    """Same data/nfft/overlap/Ns, only window_type changes hamming -> hann.

    Pre-fix: the cache key/validation carried no window_type, so the second
    run silently returned the first run's hamming-windowed blocks and this
    assertion failed (qhat was bit-identical instead of different).
    """
    np.random.seed(0)
    q = np.random.randn(32, 4)

    analyzer_hamming = _make_spod(tmp_path, q, window_type="hamming")
    analyzer_hann = _make_spod(tmp_path, q, window_type="hann")

    assert not analyzer_hamming.qhat_cached  # cold cache for the first run
    assert not np.allclose(analyzer_hamming.qhat, analyzer_hann.qhat)


def test_different_q_arrays_do_not_share_cache(tmp_path):
    """Two different `q` arrays sharing (data_root, nfft, overlap, Ns) must
    not serve each other's FFT blocks.

    Pre-fix: the cache path is built solely from (data_root, nfft, overlap,
    Ns, analysis_type) with no content check, so the second analyzer (over a
    completely different `q`) reused the first analyzer's cached blocks.
    """
    np.random.seed(1)
    q1 = np.random.randn(32, 4)
    q2 = np.random.randn(32, 4) * 5.0 + 3.0  # clearly distinct content

    analyzer1 = _make_spod(tmp_path, q1, file_path="dummy.h5")
    assert not analyzer1.qhat_cached

    analyzer2 = _make_spod(tmp_path, q2, file_path="dummy.h5")

    # Freshly computed reference for q2, independent of any cache.
    novlap = int(0.0 * 8)
    nblocks = int(np.ceil((q2.shape[0] - novlap) / (8 - novlap)))
    reference = blocksfft(
        q2, 8, nblocks, novlap,
        blockwise_mean=False, normvar=False,
        window_norm="power", window_type="hamming",
        use_parallel=False,
    )

    np.testing.assert_allclose(analyzer2.qhat, reference)
    assert not np.allclose(analyzer2.qhat, analyzer1.qhat)


def test_stamp_mismatch_recomputes_without_raising(tmp_path, capsys):
    """A cache file whose stamped parameters disagree with the current
    analyzer must be rejected (recomputed), never raise, and never silently
    serve the mismatched blocks.

    Pre-fix: there was no stamp at all, so this scenario could not even be
    expressed — any existing file with the right shape was accepted.
    """
    np.random.seed(2)
    q = np.random.randn(32, 4)

    _make_spod(tmp_path, q, window_type="hamming", file_path="dummy.h5")
    fname = "dummy_Nfft8_ovlap0.0_32snapshots_spod.hdf5"
    cache_file = tmp_path / fname
    assert cache_file.exists()

    # Corrupt the stamp in place to simulate a disagreeing/legacy cache file.
    with h5py.File(cache_file, "a") as f:
        f.attrs["_fftcache_window_type"] = "hann"

    analyzer2 = _make_spod(tmp_path, q, window_type="hamming", file_path="dummy.h5")
    captured = capsys.readouterr()

    assert not analyzer2.qhat_cached
    assert "FFT cache stamp mismatch" in captured.out

    # The recomputed result must be correct (hamming), not the corrupted stamp's hann.
    novlap = 0
    nblocks = int(np.ceil((q.shape[0] - novlap) / (8 - novlap)))
    reference = blocksfft(
        q, 8, nblocks, novlap,
        blockwise_mean=False, normvar=False,
        window_norm="power", window_type="hamming",
        use_parallel=False,
    )
    np.testing.assert_allclose(analyzer2.qhat, reference)


def test_matching_cache_hit_is_still_used(tmp_path, capsys):
    """A legitimate, fully-matching cache hit must still be served from disk.

    This is the mandatory counterpart to the tests above: a fix that merely
    disables caching (e.g. always recompute) would pass all of them while
    destroying the feature.
    """
    np.random.seed(3)
    q = np.random.randn(32, 4)

    analyzer1 = _make_spod(tmp_path, q, window_type="hamming", file_path="dummy.h5")
    assert not analyzer1.qhat_cached

    analyzer2 = _make_spod(tmp_path, q, window_type="hamming", file_path="dummy.h5")
    captured = capsys.readouterr()

    assert analyzer2.qhat_cached
    assert "Loaded cached FFT blocks" in captured.out
    np.testing.assert_array_equal(analyzer2.qhat, analyzer1.qhat)
