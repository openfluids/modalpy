import h5py
import numpy as np
import pytest

from openmodalpy import BSMDAnalyzer


def _make_analyzer(tmp_path, triads, nfft=4, Ns=10, Nspace=4, use_static=True):
    """Helper to build a BSMDAnalyzer with synthetic data."""
    Nx = int(np.sqrt(Nspace))
    Ny = Nspace // Nx
    data = {
        'q': np.random.randn(Ns, Nspace),
        'x': np.linspace(0, 1, Nx),
        'y': np.linspace(0, 1, Ny),
        'dt': 1.0,
        'Nx': Nx,
        'Ny': Ny,
        'Ns': Ns,
    }
    analyzer = BSMDAnalyzer(
        file_path='dummy.h5',
        nfft=nfft,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type='uniform',
        use_static_triads=use_static,
        static_triads=triads,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    return analyzer


def test_static_bsmd_core_small(tmp_path):
    """Basic smoke test: single zero-frequency triad produces results."""
    analyzer = _make_analyzer(tmp_path, triads=[(0, 0, 0)])
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)
    assert analyzer.modes1.shape[0] == 1
    assert analyzer.modes1.shape[1] == 4


def test_negative_frequency_conjugate_symmetry(tmp_path):
    """Negative frequency bin indices are served via conjugate symmetry."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    # qhat has shape (nfft//2+1, Nspace, Nblocks) = (5, 4, Nblocks)
    assert analyzer.qhat.shape[0] == 5  # bins 0..4

    # Directly check the helper: qhat[-1] should equal conj(qhat[1])
    q_pos = analyzer._get_qhat_for_index(1)
    q_neg = analyzer._get_qhat_for_index(-1)
    np.testing.assert_array_equal(q_neg, np.conj(q_pos))

    # Run BSMD — should not crash with negative indices
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)
    assert not np.isnan(analyzer.eigenvalues[0])


def test_out_of_range_index_raises(tmp_path):
    """Triads with |p| > nfft//2 are unanalysable and raise ValueError."""
    analyzer = _make_analyzer(tmp_path, triads=[(99, -99, 0)], nfft=4, Ns=10)
    # nfft=4 → rfft bins 0..2; |p| = 99 exceeds nfft//2 = 2.
    with pytest.raises(ValueError, match=r"p=99"):
        analyzer._perform_static_bsmd_core()


def test_nyquist_index_is_accepted(tmp_path):
    """|p| == nfft//2 is the last real rfft bin and must be analysed, not rejected."""
    analyzer = _make_analyzer(tmp_path, triads=[(4, -4, 0)], nfft=8, Ns=32)
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)


def test_one_bin_past_nyquist_raises(tmp_path):
    """|p| == nfft//2 + 1 is the first unanalysable bin; the bound is exclusive there."""
    analyzer = _make_analyzer(tmp_path, triads=[(5, -5, 0)], nfft=8, Ns=32)
    with pytest.raises(ValueError, match=r"p=5"):
        analyzer._perform_static_bsmd_core()


def test_dynamic_triad_selection_raises(tmp_path):
    """Dynamic triad selection is unimplemented and must say so, not return empty arrays."""
    analyzer = _make_analyzer(tmp_path, triads=[], nfft=8, Ns=32, use_static=False)
    with pytest.raises(NotImplementedError):
        analyzer.perform_bsmd()


def test_energy_map_keeps_triads_beyond_the_default_range(tmp_path):
    """A triad inside the rfft range but outside |p| <= 8 must still reach the energy map.

    The map used to be a fixed 17x17 grid centred on |p| = 8, so a valid triad at
    p = 12 (nfft = 32 gives 16 usable bins) was analysed and then silently dropped.
    """
    analyzer = _make_analyzer(tmp_path, triads=[(12, -12, 0)], nfft=32, Ns=96)
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (1,)
    grid = analyzer.energy_map
    assert grid.shape == (25, 25), f"grid must span |p| = 12, got {grid.shape}"
    assert np.count_nonzero(np.isfinite(grid)) == 1, "the analysed triad is missing from the map"
    assert np.isfinite(grid[12 + 12, -12 + 12])


def test_triadic_constraint_violation_skipped(tmp_path):
    """Triads that violate p1+p2=p3 are skipped with NaN eigenvalue."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, 1, 1)], nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    assert np.isnan(np.abs(analyzer.eigenvalues[0]))


def test_multiple_triads_with_negatives(tmp_path):
    """Multiple triads including negative bins all produce finite results."""
    triads = [(1, -1, 0), (2, -2, 0), (1, 1, 2), (0, 0, 0)]
    analyzer = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    assert analyzer.eigenvalues.shape == (len(triads),)
    assert analyzer.modes1.shape == (len(triads), 4)
    assert analyzer.modes2.shape == (len(triads), 4)
    # All valid triads should produce finite eigenvalues
    for idx, (p1, p2, p3) in enumerate(triads):
        assert not np.isnan(analyzer.eigenvalues[idx]), f"Triad {(p1,p2,p3)} produced NaN"


def test_bispectral_correlation_uses_all_three_frequencies(tmp_path):
    """Verify that the bispectral correlation C involves Q1, Q2, AND Q3.

    Construct a case where Q3 is zeroed out.  If the algorithm correctly
    uses Q3 as B in C = A^H W B, all eigenvalues should be zero.
    """
    triads = [(1, 1, 2)]
    analyzer = _make_analyzer(tmp_path, triads=triads, nfft=8, Ns=24)
    # Zero out qhat at bin 2 (= p3) → B = 0 → C = 0 → eigenvalue = 0
    analyzer.qhat[2, :, :] = 0.0
    analyzer._perform_static_bsmd_core()
    assert np.abs(analyzer.eigenvalues[0]) == pytest.approx(0.0, abs=1e-12)


def test_disk_backed_qhat_matches_ram(tmp_path):
    """Disk-backed mode (max_qhat_gb=0) produces identical results to RAM mode."""
    triads = [(1, -1, 0), (2, -2, 0), (1, 1, 2), (0, 0, 0)]
    np.random.seed(42)

    # RAM mode (default)
    ram = _make_analyzer(tmp_path / "ram", triads=triads, nfft=8, Ns=24)
    ram._perform_static_bsmd_core()

    # Disk-backed mode: max_qhat_gb=0 forces offload on any qhat
    np.random.seed(42)
    Nspace = 4
    Nx, Ny = 2, 2
    data = {
        'q': np.random.randn(24, Nspace),
        'x': np.linspace(0, 1, Nx),
        'y': np.linspace(0, 1, Ny),
        'dt': 1.0, 'Nx': Nx, 'Ny': Ny, 'Ns': 24,
    }
    disk_dir = tmp_path / "disk"
    disk_dir.mkdir()
    disk = BSMDAnalyzer(
        file_path='dummy.h5', nfft=8, overlap=0.0,
        results_dir=disk_dir, figures_dir=disk_dir,
        data_loader=lambda _: data,
        spatial_weight_type='uniform',
        use_static_triads=True, static_triads=triads,
        max_qhat_gb=0,  # force disk-backed
    )
    disk.load_and_preprocess()
    disk.compute_fft_blocks()
    assert disk._qhat_on_disk, "Expected disk-backed mode with max_qhat_gb=0"

    disk._perform_static_bsmd_core()

    np.testing.assert_allclose(np.abs(disk.eigenvalues), np.abs(ram.eigenvalues), rtol=1e-12)
    np.testing.assert_allclose(np.abs(disk.modes1), np.abs(ram.modes1), rtol=1e-12)
    np.testing.assert_allclose(np.abs(disk.modes2), np.abs(ram.modes2), rtol=1e-12)
    disk.close()


def test_compute_single_triad_matches_dominant_eigenpair_shortcut(tmp_path):
    """The current BSMD core returns the dominant eigenpair of C.

    NOTE: this test re-derives the formula inline and therefore mirrors the
    implementation in src/openmodalpy/bsmd.py::_compute_single_triad -- it is
    a characterization test, NOT an independent oracle. It only proves the
    method is internally self-consistent with its own formula; it says nothing
    about whether that formula is physically correct. The independent oracle
    for correctness (manufactured phase-locked/control triads) lives in
    tests/test_bsmd_manufactured.py.
    """
    analyzer = _make_analyzer(tmp_path, triads=[(1, 2, 3)], nfft=8, Ns=24, Nspace=2)
    analyzer.W = np.ones((2, 1), dtype=complex)

    q1 = np.ones((2, 2), dtype=complex)
    q2 = np.eye(2, dtype=complex)
    q3 = np.array([[0.0, 4.0], [0.0, 2.0]], dtype=complex)
    mapping = {1: q1, 2: q2, 3: q3}
    analyzer._get_qhat_for_index = lambda idx: mapping[idx]

    eigval, mode1, mode2 = analyzer._compute_single_triad(1, 2, 3)

    prod = q1 * q2
    c_matrix = (np.conj(q3).T @ (analyzer.W * prod)) / q1.shape[1]
    eigvals, eigvecs = np.linalg.eig(c_matrix)
    dom = np.argmax(np.abs(eigvals))
    coeffs = eigvecs[:, dom]

    np.testing.assert_allclose(eigval, eigvals[dom], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mode1, q3 @ coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mode2, prod @ coeffs, rtol=1e-12, atol=1e-12)


def test_save_results_records_bsmd_approximation_contract(tmp_path):
    analyzer = _make_analyzer(tmp_path, triads=[(0, 0, 0)], nfft=4, Ns=10)
    analyzer._perform_static_bsmd_core()
    analyzer.save_results("bsmd_contract.hdf5")

    with h5py.File(tmp_path / "bsmd_contract.hdf5", "r") as handle:
        assert handle.attrs["bsmd_solver"] == "dominant_eigenpair_approximation"
        assert handle.attrs["bsmd_target_objective"] == "numerical_radius"
        assert handle.attrs["lift_kind"] == "triadic_spectral_product"
        assert bool(handle.attrs["uses_shared_triadic_coefficients"])
        assert handle.attrs["bispectrum_conjugation"] == "sum_frequency_conjugated"


def test_load_results_roundtrip_accepts_current_stamp(tmp_path):
    """A file written by the current build reloads without complaint."""
    analyzer = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    analyzer.save_results("bsmd_roundtrip.hdf5")

    reloaded = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    reloaded.load_results("bsmd_roundtrip.hdf5")
    np.testing.assert_allclose(reloaded.eigenvalues, analyzer.eigenvalues, rtol=1e-12)


def test_load_results_rejects_prefix_unconjugated_file(tmp_path):
    """Results written before the sum-frequency conjugation fix must not load.

    Such files carry eigenvalues and modes computed from E[X(f1)X(f2)X(f1+f2)]
    instead of the bispectrum E[X(f1)X(f2)X*(f1+f2)]; reloading them silently
    would hand the caller invalid numbers.
    """
    analyzer = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    analyzer._perform_static_bsmd_core()
    analyzer.save_results("bsmd_stale.hdf5")

    # Simulate a file produced by a pre-fix build: the stamp did not exist.
    with h5py.File(tmp_path / "bsmd_stale.hdf5", "a") as handle:
        del handle.attrs["bispectrum_conjugation"]

    reloaded = _make_analyzer(tmp_path, triads=[(1, -1, 0)], nfft=8, Ns=24)
    with pytest.raises(ValueError, match="sum-frequency term was not conjugated"):
        reloaded.load_results("bsmd_stale.hdf5")
