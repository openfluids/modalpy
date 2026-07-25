import h5py
import matplotlib
import numpy as np
import pytest

from openmodalpy import DMDAnalyzer
from openmodalpy.dmd import _delay_embed


def _exact_dmd_eigenvalues(q: np.ndarray, n_modes_save: int) -> np.ndarray:
    x = q[:-1, :].T
    y = q[1:, :].T

    u, s, vh = np.linalg.svd(x, full_matrices=False)
    r = min(n_modes_save, len(s))
    u_r = u[:, :r]
    s_r = np.diag(s[:r])
    v_r = vh.conj().T[:, :r]

    atilde = u_r.conj().T @ y @ v_r @ np.linalg.inv(s_r)
    eigvals, _ = np.linalg.eig(atilde)
    idx = np.argsort(np.abs(eigvals))[::-1]
    return eigvals[idx][:n_modes_save]


def test_perform_dmd_simple():
    data = {
        "q": np.array([[1, 2], [2, 4], [4, 8]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()
    assert analyzer.modes.shape == (2, 2)
    assert analyzer.time_coefficients.shape == (3, 2)
    assert np.isclose(analyzer.eigenvalues[0], 2.0, atol=1e-6)


def test_plot_eigenspectra_stem_compat(monkeypatch, tmp_path):
    data = {
        'q': np.random.randn(8, 4),
        'x': np.linspace(0, 1, 2),
        'y': np.linspace(0, 1, 2),
        'dt': 1.0,
        'Nx': 2,
        'Ny': 2,
        'Ns': 8,
    }
    analyzer = DMDAnalyzer(
        file_path='dummy.h5',
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type='uniform',
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()

    calls = []

    def stem_no_use(self, x, y, linefmt=None, markerfmt=None, basefmt=None):
        calls.append('no_use')
        return None

    monkeypatch.setattr(matplotlib.axes.Axes, 'stem', stem_no_use)
    analyzer.plot_eigenspectra()
    assert 'no_use' in calls

    calls.clear()

    def stem_use(self, x, y, linefmt=None, markerfmt=None, basefmt=None, use_line_collection=True):
        calls.append('use_line_collection' if use_line_collection else 'use')
        return None

    monkeypatch.setattr(matplotlib.axes.Axes, 'stem', stem_use)
    analyzer.plot_eigenspectra()
    assert 'use_line_collection' in calls

    expected = tmp_path / 'dummy_dmd_eigenspectra.png'
    assert expected.exists()


def test_dmd_uses_raw_shifted_snapshots_without_weighting():
    data = {
        "q": np.array(
            [
                [10.0, 1.0],
                [11.0, 2.0],
                [13.0, 4.0],
                [16.0, 8.0],
                [20.0, 16.0],
            ],
            dtype=float,
        ),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 5,
    }

    analyzer_a = DMDAnalyzer(
        file_path="dummy_a",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer_a.load_and_preprocess()
    analyzer_a.W = np.array([1.0, 50.0])
    analyzer_a.perform_dmd()

    analyzer_b = DMDAnalyzer(
        file_path="dummy_b",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer_b.load_and_preprocess()
    analyzer_b.W = np.array([50.0, 1.0])
    analyzer_b.perform_dmd()

    expected_raw = _exact_dmd_eigenvalues(data["q"], n_modes_save=2)
    expected_centered = _exact_dmd_eigenvalues(
        data["q"] - np.mean(data["q"], axis=0, keepdims=True),
        n_modes_save=2,
    )

    np.testing.assert_allclose(analyzer_a.eigenvalues, expected_raw, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(analyzer_b.eigenvalues, expected_raw, rtol=1e-10, atol=1e-10)
    assert not np.allclose(
        np.sort_complex(analyzer_a.eigenvalues),
        np.sort_complex(expected_centered),
        rtol=1e-8,
        atol=1e-8,
    )


def test_dmd_save_results_records_current_contract(tmp_path):
    data = {
        "q": np.array([[1, 2], [2, 4], [4, 8]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()
    analyzer.save_results("dmd_contract.hdf5")

    with h5py.File(tmp_path / "dmd_contract.hdf5", "r") as handle:
        assert handle.attrs["dmd_variant"] == "exact_dmd"
        assert handle.attrs["paired_data_contract"] == "raw_shifted_snapshots"
        assert not bool(handle.attrs["uses_mean_subtraction"])
        assert not bool(handle.attrs["uses_spatial_metric_in_regression"])
        assert handle.attrs["mode_ranking"] == "abs_lambda_desc"


# ---------------------------------------------------------------------------
# Helper: generate snapshots from a linear system x_{k+1} = A x_k
# ---------------------------------------------------------------------------


def _make_linear_snapshots(A, x0, n_steps):
    """Return q array of shape (n_steps+1, n_spatial)."""
    snapshots = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        x = A @ x
        snapshots.append(x.copy())
    return np.array(snapshots)


def _make_analyzer(q, n_modes_save=None):
    """Shorthand to build a DMDAnalyzer from a snapshot array."""
    n_spatial = q.shape[1]
    if n_modes_save is None:
        n_modes_save = n_spatial
    data = {
        "q": q,
        "x": np.arange(n_spatial, dtype=float),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": n_spatial,
        "Ny": 1,
        "Ns": q.shape[0],
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=n_modes_save,
    )
    analyzer.load_and_preprocess()
    return analyzer


# ---------------------------------------------------------------------------
# Delay embedding
# ---------------------------------------------------------------------------


def test_delay_embed_shape():
    """_delay_embed produces the correct Hankel matrix dimensions."""
    X = np.random.randn(3, 10)
    d = 4
    Xd = _delay_embed(X, d)
    assert Xd.shape == (3 * d, 10 - d + 1)


def test_delay_embed_d1_identity():
    """With d=1, _delay_embed returns the input unchanged."""
    X = np.random.randn(5, 8)
    Xd = _delay_embed(X, 1)
    np.testing.assert_array_equal(Xd, X)


def test_delay_embed_values():
    """Verify the stacking order of _delay_embed."""
    # 2 spatial points, 5 time steps
    X = np.arange(10).reshape(2, 5).astype(float)
    Xd = _delay_embed(X, 3)
    # Row block 0: X[:, 0:3], block 1: X[:, 1:4], block 2: X[:, 2:5]
    expected = np.vstack([X[:, 0:3], X[:, 1:4], X[:, 2:5]])
    np.testing.assert_array_equal(Xd, expected)


# ---------------------------------------------------------------------------
# Omega (continuous-time eigenvalues)
# ---------------------------------------------------------------------------


def test_omega_returned():
    """perform_dmd populates self.omega = log(eigvals) / dt."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 30)
    analyzer = _make_analyzer(q)
    analyzer.perform_dmd()

    assert analyzer.omega.size == analyzer.eigenvalues.size
    dt = analyzer.data.get("dt", 1.0)
    expected_omega = np.log(analyzer.eigenvalues.astype(complex)) / dt
    np.testing.assert_allclose(analyzer.omega, expected_omega)


# ---------------------------------------------------------------------------
# Default backward compatibility
# ---------------------------------------------------------------------------


def test_default_args_match_original():
    """Default perform_dmd() gives identical eigenvalues to the reference helper."""
    data_q = np.array(
        [[10.0, 1.0], [11.0, 2.0], [13.0, 4.0], [16.0, 8.0], [20.0, 16.0]],
        dtype=float,
    )
    expected = _exact_dmd_eigenvalues(data_q, n_modes_save=2)

    analyzer = _make_analyzer(data_q, n_modes_save=2)
    analyzer.perform_dmd()  # defaults: method="ls", delays=1
    np.testing.assert_allclose(analyzer.eigenvalues, expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# TLS-DMD
# ---------------------------------------------------------------------------


def test_tls_dmd_clean_data():
    """On noise-free data, TLS eigenvalues recover the true system eigenvalues."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    true_eigvals = np.sort(np.linalg.eigvals(A))
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 50)

    analyzer = _make_analyzer(q)
    analyzer.perform_dmd(method="tls")

    recovered = np.sort(analyzer.eigenvalues)
    np.testing.assert_allclose(recovered, true_eigvals, atol=1e-8)


def test_tls_noise_robustness():
    """TLS-DMD eigenvalues should be at least as close to ground truth as LS under noise."""
    rng = np.random.default_rng(42)
    A = np.array([[0.95, 0.05], [-0.05, 0.90]])
    true_eigvals = np.sort(np.linalg.eigvals(A))
    q_clean = _make_linear_snapshots(A, np.array([1.0, 0.5]), 80)

    noise_level = 0.05 * np.std(q_clean)
    q_noisy = q_clean + rng.normal(0, noise_level, q_clean.shape)

    # LS
    analyzer_ls = _make_analyzer(q_noisy)
    analyzer_ls.perform_dmd(method="ls")
    err_ls = np.linalg.norm(np.sort(analyzer_ls.eigenvalues) - true_eigvals)

    # TLS
    analyzer_tls = _make_analyzer(q_noisy)
    analyzer_tls.perform_dmd(method="tls")
    err_tls = np.linalg.norm(np.sort(analyzer_tls.eigenvalues) - true_eigvals)

    # TLS should not be worse (allow small tolerance for edge cases)
    assert err_tls <= err_ls * 1.1, (
        f"TLS error {err_tls:.6e} exceeded LS error {err_ls:.6e} by more than 10%"
    )


# ---------------------------------------------------------------------------
# Delay-embedded DMD
# ---------------------------------------------------------------------------


def test_dmd_with_delays():
    """DMD with delay embedding runs and returns valid eigenvalues."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 40)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.perform_dmd(delays=3)

    assert analyzer.eigenvalues.size == 4
    assert analyzer.modes.shape[0] == 2 * 3  # n_spatial * delays
    assert analyzer.omega.size == 4
    # All eigenvalues should be finite
    assert np.all(np.isfinite(analyzer.eigenvalues))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_method_raises():
    A = np.eye(2) * 0.9
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 10)
    analyzer = _make_analyzer(q)
    with pytest.raises(ValueError, match="Unknown method"):
        analyzer.perform_dmd(method="bogus")


def test_invalid_delays_raises():
    A = np.eye(2) * 0.9
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 10)
    analyzer = _make_analyzer(q)
    with pytest.raises(ValueError, match="delays must be >= 1"):
        analyzer.perform_dmd(delays=0)


# ---------------------------------------------------------------------------
# Metadata reflects variant
# ---------------------------------------------------------------------------


def test_metadata_tls_delays(tmp_path):
    """Metadata should reflect TLS + delay embedding settings."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 30)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.perform_dmd(method="tls", delays=3)

    meta = analyzer._get_algorithm_metadata()
    assert meta["dmd_variant"] == "delay_embedded_tls_dmd"
    assert meta["dmd_method"] == "tls"
    assert meta["dmd_delays"] == 3
    assert meta["lift_kind"] == "delay_embedding"


def test_load_results_restores_variant_metadata(tmp_path):
    """Saved DMD variant metadata should survive a load/save round-trip."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 30)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.results_dir = tmp_path
    analyzer.perform_dmd(method="tls", delays=3)
    analyzer.save_results("dmd_variant_roundtrip.hdf5")

    reloaded = _make_analyzer(q, n_modes_save=4)
    reloaded.results_dir = tmp_path
    reloaded.load_results("dmd_variant_roundtrip.hdf5")

    meta = reloaded._get_algorithm_metadata()
    assert meta["dmd_variant"] == "delay_embedded_tls_dmd"
    assert meta["dmd_method"] == "tls"
    assert meta["dmd_delays"] == 3


# ---------------------------------------------------------------------------
# HODMD / TLS-HODMD named variant metadata
# ---------------------------------------------------------------------------


def test_hodmd_named_variant_metadata():
    """perform_dmd with named_variant='hodmd' sets the correct metadata."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 40)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.perform_dmd(method="ls", delays=3, named_variant="hodmd")

    assert analyzer._dmd_named_variant == "hodmd"
    meta = analyzer._get_algorithm_metadata()
    assert meta["dmd_variant"] == "hodmd"
    assert meta["dmd_named_variant"] == "hodmd"
    assert meta["dmd_method"] == "ls"
    assert meta["dmd_delays"] == 3
    assert meta["lift_kind"] == "delay_embedding"


def test_tls_hodmd_named_variant_metadata():
    """perform_dmd with named_variant='tls_hodmd' sets the correct metadata."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 40)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.perform_dmd(method="tls", delays=3, named_variant="tls_hodmd")

    assert analyzer._dmd_named_variant == "tls_hodmd"
    meta = analyzer._get_algorithm_metadata()
    assert meta["dmd_variant"] == "tls_hodmd"
    assert meta["dmd_named_variant"] == "tls_hodmd"
    assert meta["dmd_method"] == "tls"
    assert meta["dmd_delays"] == 3
    assert meta["lift_kind"] == "delay_embedding"


def test_hodmd_save_load_roundtrip(tmp_path):
    """HODMD named variant survives a save/load round-trip."""
    A = np.array([[0.9, 0.1], [-0.1, 0.8]])
    q = _make_linear_snapshots(A, np.array([1.0, 0.5]), 40)

    analyzer = _make_analyzer(q, n_modes_save=4)
    analyzer.results_dir = tmp_path
    analyzer.perform_dmd(method="ls", delays=3, named_variant="hodmd")
    analyzer.save_results("hodmd_roundtrip.hdf5")

    reloaded = _make_analyzer(q, n_modes_save=4)
    reloaded.results_dir = tmp_path
    reloaded.load_results("hodmd_roundtrip.hdf5")

    assert reloaded._dmd_named_variant == "hodmd"
    meta = reloaded._get_algorithm_metadata()
    assert meta["dmd_variant"] == "hodmd"
    assert meta["dmd_method"] == "ls"
    assert meta["dmd_delays"] == 3


def test_hodmd_plot_modes_uses_2d_slice(monkeypatch, tmp_path):
    """Delay-embedded DMD modes should be visualized as 2D maps, not 1D lines."""
    nx, ny = 4, 3
    n_space = nx * ny
    q = np.random.randn(40, n_space)
    data = {
        "q": q,
        "x": np.arange(nx, dtype=float),
        "y": np.arange(ny, dtype=float),
        "dt": 1.0,
        "Nx": nx,
        "Ny": ny,
        "Ns": q.shape[0],
        "metadata": {"var_name": "u"},
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd(method="ls", delays=2, named_variant="hodmd")

    line_calls = {"count": 0}
    orig_plot = matplotlib.axes.Axes.plot

    def plot_wrapper(self, *args, **kwargs):
        line_calls["count"] += 1
        return orig_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", plot_wrapper)
    analyzer.plot_modes(plot_n_modes=2, modes_per_fig=2)

    assert line_calls["count"] == 0
    assert (tmp_path / "dummy_dmd_modes_1_to_2_u.png").exists()
