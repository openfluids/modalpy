import h5py
import numpy as np
from scipy import signal

from openmodalpy import PODAnalyzer
from openmodalpy.core.base import subset_volume_focus_3d


def test_perform_pod_simple():
    data = {
        "q": np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    assert analyzer.modes.shape == (2, 2)
    assert analyzer.time_coefficients.shape == (3, 2)
    assert np.isclose(analyzer.eigenvalues[0], 5.333333333333333, atol=1e-6)


def test_plot_time_coefficients_strouhal(monkeypatch, tmp_path):
    data = {
        "q": np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,

    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    labels = []
    x_data = []

    def xlabel_mock(text):
        labels.append(text)

    def semilogy_mock(x, y, **kwargs):
        x_data.append(np.array(x))
        return None

    monkeypatch.setattr("matplotlib.pyplot.xlabel", xlabel_mock)
    monkeypatch.setattr("matplotlib.pyplot.semilogy", semilogy_mock)

    analyzer.plot_time_coefficients(n_coeffs_to_plot=1, L=2.0, U=4.0)

    assert "Strouhal Number (St)" in labels
    coeff = analyzer.time_coefficients[:3, 0]
    freqs, _ = signal.periodogram(coeff, analyzer.fs, scaling="density")
    expected = freqs * 2.0 / 4.0
    assert np.allclose(x_data[0], expected)


def test_spatial_kernel_time_coefficients_use_weighted_inner_product():
    data = {
        "q": np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 1.0],
                [4.0, 3.0],
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
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.W = np.array([1.0, 9.0])
    analyzer.perform_pod()

    data_mean_removed = data["q"] - np.mean(data["q"], axis=0, keepdims=True)
    expected = (data_mean_removed * analyzer.W) @ analyzer.modes
    np.testing.assert_allclose(
        analyzer.time_coefficients,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_pod_save_results_records_second_order_contract(tmp_path):
    data = {
        "q": np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    analyzer.save_results("pod_contract.hdf5")

    with h5py.File(tmp_path / "pod_contract.hdf5", "r") as handle:
        assert handle.attrs["lift_kind"] == "identity_centered_snapshots"
        assert bool(handle.attrs["uses_mean_subtraction"])
        assert bool(handle.attrs["uses_spatial_metric_in_second_order_operator"])
        assert handle.attrs["eigenvalue_normalization"] == "snapshot_average"


def test_run_analysis_uses_3d_slice_plots_for_volumetric_data(monkeypatch, tmp_path):
    data = {
        "q": np.array(
            [
                np.arange(8, dtype=float),
                np.arange(8, dtype=float) + 1.0,
                np.arange(8, dtype=float) + 2.0,
                np.arange(8, dtype=float) + 3.0,
            ]
        ),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0, 1.0]),
        "z": np.array([0.0, 1.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 2,
        "Nz": 2,
        "Ns": 4,
    }
    analyzer = PODAnalyzer(
        file_path="3d",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )

    slice_calls = []
    iso_calls = []
    monkeypatch.setattr(PODAnalyzer, "plot_modes_3d_slices", lambda self, plot_n_modes=4: slice_calls.append(plot_n_modes))
    monkeypatch.setattr(PODAnalyzer, "plot_modes_3d_isometric", lambda self, plot_n_modes=4: iso_calls.append(plot_n_modes))
    monkeypatch.setattr(PODAnalyzer, "plot_eigenvalues", lambda self: None)
    monkeypatch.setattr(PODAnalyzer, "plot_time_coefficients", lambda self, **kwargs: None)
    monkeypatch.setattr(PODAnalyzer, "plot_cumulative_energy", lambda self: None)
    monkeypatch.setattr(PODAnalyzer, "plot_reconstruction_error", lambda self: None)
    monkeypatch.setattr(PODAnalyzer, "plot_reconstruction_comparison", lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("2D reconstruction comparison should not be used for 3D data")))
    monkeypatch.setattr(PODAnalyzer, "plot_modes_pair_detailed", lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("2D mode plotting should not be used for 3D data")))
    monkeypatch.setattr(PODAnalyzer, "plot_modes_grid", lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("2D grid plotting should not be used for 3D data")))
    monkeypatch.setattr(PODAnalyzer, "plot_mode_pair_phase", lambda self: (_ for _ in ()).throw(AssertionError("2D pair-phase plotting should not be used for 3D data")))

    analyzer.run_analysis(plot_n_modes_spatial=2, plot_n_coeffs_time=1)

    assert slice_calls == [2]
    assert iso_calls == [2]


def test_subset_volume_focus_3d_respects_volume_xlim():
    field = np.arange(5 * 3 * 2, dtype=float).reshape(5, 3, 2)
    data = {
        "metadata": {
            "plot_style": {
                "volume": {
                    "xlim": [0.0, 1.0],
                }
            }
        }
    }
    x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0])

    focused, x_focus, y_focus, z_focus = subset_volume_focus_3d(field, x, y, z, data)

    assert focused.shape == (3, 3, 2)
    np.testing.assert_array_equal(x_focus, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(y_focus, y)
    np.testing.assert_array_equal(z_focus, z)
