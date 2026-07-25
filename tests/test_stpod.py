"""
Unit tests for STPODAnalyzer.
"""

import h5py
import numpy as np
import pytest

from openmodalpy import STPODAnalyzer


class TestSTPODBasic:
    """Basic functionality tests for ST-POD."""

    def test_perform_stpod_simple(self):
        """Basic ST-POD execution on synthetic data."""
        np.random.seed(42)
        Ns, Nx, Ny = 50, 10, 10
        Nspace = Nx * Ny

        data = {
            "q": np.random.randn(Ns, Nspace),
            "x": np.linspace(0, 1, Nx),
            "y": np.linspace(0, 1, Ny),
            "dt": 0.1,
            "Nx": Nx,
            "Ny": Ny,
            "Ns": Ns,
        }

        embedding_dim = 5
        n_modes = 10
        analyzer = STPODAnalyzer(
            file_path="test_stpod",
            embedding_dim=embedding_dim,
            n_modes_save=n_modes,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # Check output shapes
        m = Ns - embedding_dim + 1
        assert analyzer.modes.shape == (embedding_dim * Nspace, n_modes)
        assert analyzer.time_coefficients.shape == (m, n_modes)
        assert analyzer.eigenvalues.shape == (n_modes,)

    def test_hankel_matrix_shape(self):
        """Verify Hankel matrix construction."""
        Ns, Nspace = 20, 15
        embedding_dim = 5

        data = {
            "q": np.arange(Ns * Nspace).reshape(Ns, Nspace).astype(float),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_hankel",
            embedding_dim=embedding_dim,
            n_modes_save=5,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()

        # Build Hankel manually to test
        data_centered = data["q"] - np.mean(data["q"], axis=0)
        H = analyzer._build_hankel_matrix(data_centered)

        m = Ns - embedding_dim + 1
        assert H.shape == (embedding_dim * Nspace, m)

    def test_extract_spatial_mode(self):
        """Test extraction of spatial modes from space-time modes."""
        np.random.seed(123)
        Ns, Nspace = 30, 20
        embedding_dim = 4

        data = {
            "q": np.random.randn(Ns, Nspace),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_extract",
            embedding_dim=embedding_dim,
            n_modes_save=5,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        # Extract at different delays
        for delay in range(embedding_dim):
            spatial_mode = analyzer.extract_spatial_mode(0, delay)
            assert spatial_mode.shape == (Nspace,)

    def test_get_mode_as_movie(self):
        """Test getting mode as temporal sequence."""
        np.random.seed(456)
        Ns, Nspace = 40, 25
        embedding_dim = 6

        data = {
            "q": np.random.randn(Ns, Nspace),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_movie",
            embedding_dim=embedding_dim,
            n_modes_save=5,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        movie = analyzer.get_mode_as_movie(0)
        assert movie.shape == (embedding_dim, Nspace)

    def test_eigenvalues_match_sigma_squared_over_hankel_columns(self):
        """ST-POD eigenvalues should use the same per-realization scaling as POD."""
        np.random.seed(7)
        Ns, Nspace = 12, 3
        embedding_dim = 4
        n_modes = 3

        data = {
            "q": np.random.randn(Ns, Nspace),
            "x": np.arange(Nspace),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": Nspace,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_stpod_norm",
            embedding_dim=embedding_dim,
            n_modes_save=n_modes,
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()

        data_centered = data["q"] - np.mean(data["q"], axis=0)
        hankel = analyzer._build_hankel_matrix(data_centered)
        sigma = np.linalg.svd(hankel, full_matrices=False, compute_uv=False)[:n_modes]
        n_hankel_cols = Ns - embedding_dim + 1

        np.testing.assert_allclose(
            analyzer.eigenvalues,
            sigma**2 / n_hankel_cols,
            rtol=1e-10,
            atol=1e-10,
        )
        assert not np.allclose(
            analyzer.eigenvalues,
            sigma**2,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_save_results_records_delay_embedded_contract(self, tmp_path):
        np.random.seed(9)
        Ns, Nspace = 10, 3
        analyzer = STPODAnalyzer(
            file_path="test_stpod_contract",
            embedding_dim=3,
            n_modes_save=2,
            results_dir=tmp_path,
            figures_dir=tmp_path,
            data_loader=lambda _: {
                "q": np.random.randn(Ns, Nspace),
                "x": np.arange(Nspace),
                "y": np.array([0.0]),
                "dt": 0.1,
                "Nx": Nspace,
                "Ny": 1,
                "Ns": Ns,
            },
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()
        analyzer.perform_stpod()
        analyzer.save_results("stpod_contract.hdf5")

        with h5py.File(tmp_path / "stpod_contract.hdf5", "r") as handle:
            assert handle.attrs["stpod_variant"] == "delay_embedded_pod"
            assert handle.attrs["lift_kind"] == "delay_embedding"
            assert handle.attrs["eigenvalue_normalization"] == "sigma_squared_over_n_hankel_cols"
            assert not bool(handle.attrs["is_full_spacetime_pod"])


class TestSTPODValidation:
    """Validation tests for ST-POD parameters."""

    def test_embedding_dim_too_small_raises(self):
        """embedding_dim < 2 should raise ValueError."""
        data = {
            "q": np.random.randn(20, 10),
            "x": np.arange(10),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": 10,
            "Ny": 1,
            "Ns": 20,
        }

        analyzer = STPODAnalyzer(
            file_path="test_small_d",
            embedding_dim=1,  # Invalid
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()

        with pytest.raises(ValueError, match="embedding_dim must be >= 2"):
            analyzer.perform_stpod()

    def test_embedding_dim_too_large_raises(self):
        """embedding_dim >= Ns should raise ValueError."""
        Ns = 10
        data = {
            "q": np.random.randn(Ns, 5),
            "x": np.arange(5),
            "y": np.array([0.0]),
            "dt": 0.1,
            "Nx": 5,
            "Ny": 1,
            "Ns": Ns,
        }

        analyzer = STPODAnalyzer(
            file_path="test_large_d",
            embedding_dim=Ns,  # Invalid: equal to Ns
            data_loader=lambda _: data,
            spatial_weight_type="uniform",
        )
        analyzer.load_and_preprocess()

        with pytest.raises(ValueError, match="must be < number of snapshots"):
            analyzer.perform_stpod()

    def test_no_data_raises(self):
        """Calling perform_stpod without data should raise."""
        analyzer = STPODAnalyzer(
            file_path="nonexistent",
            embedding_dim=5,
        )
        # Don't call load_and_preprocess

        with pytest.raises(ValueError, match="Data not loaded"):
            analyzer.perform_stpod()
