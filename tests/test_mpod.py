from __future__ import annotations

import h5py
import numpy as np

from modalpy import MPODAnalyzer, PODAnalyzer


def _make_uniform_data(q: np.ndarray, dt: float = 1.0) -> dict:
    n_space = q.shape[1]
    return {
        "q": q,
        "x": np.arange(n_space, dtype=float),
        "y": np.array([0.0]),
        "dt": dt,
        "Nx": n_space,
        "Ny": 1,
        "Ns": q.shape[0],
    }


def _normalized(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def test_single_band_mpod_matches_pod():
    data = _make_uniform_data(
        np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [2.0, 2.0],
                [3.0, 1.0],
                [4.0, 3.0],
            ],
            dtype=float,
        )
    )

    pod = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    pod.load_and_preprocess()
    pod.perform_pod()

    mpod = MPODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 0.5],
    )
    mpod.load_and_preprocess()
    mpod.perform_mpod()

    np.testing.assert_allclose(mpod.eigenvalues, pod.eigenvalues, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.abs(mpod.modes), np.abs(pod.modes), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.abs(mpod.time_coefficients), np.abs(pod.time_coefficients), rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(mpod.mode_band_indices, np.zeros(2, dtype=int))


def test_mpod_separates_modes_by_frequency_band():
    dt = 0.05
    ns = 200
    t = np.arange(ns) * dt
    phi_low = _normalized(np.array([1.0, 0.0, 0.0, 1.0]))
    phi_high = _normalized(np.array([0.0, 1.0, 1.0, 0.0]))
    q = (
        np.sin(2 * np.pi * 1.0 * t)[:, None] * phi_low[None, :]
        + 0.7 * np.sin(2 * np.pi * 4.0 * t)[:, None] * phi_high[None, :]
    )
    data = _make_uniform_data(q, dt=dt)

    analyzer = MPODAnalyzer(
        file_path="toy_signal",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 2.0, 5.0],
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()

    assert set(analyzer.mode_band_indices.tolist()) == {0, 1}

    low_mode = analyzer.modes[:, np.where(analyzer.mode_band_indices == 0)[0][0]]
    high_mode = analyzer.modes[:, np.where(analyzer.mode_band_indices == 1)[0][0]]
    assert abs(np.dot(_normalized(low_mode), phi_low)) > 0.95
    assert abs(np.dot(_normalized(high_mode), phi_high)) > 0.95


def test_mpod_save_results_records_band_metadata(tmp_path):
    data = _make_uniform_data(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        dt=0.1,
    )

    analyzer = MPODAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
        band_edges=[0.0, 2.0, 5.0],
        filter_kind="rectangular",
    )
    analyzer.load_and_preprocess()
    analyzer.perform_mpod()
    analyzer.save_results("mpod_contract.hdf5")

    with h5py.File(tmp_path / "mpod_contract.hdf5", "r") as handle:
        assert handle.attrs["lift_kind"] == "multiscale_filtered_snapshots"
        assert handle.attrs["filter_kind"] == "rectangular"
        np.testing.assert_allclose(handle.attrs["band_edges_hz"], np.array([0.0, 2.0, 5.0]))
        np.testing.assert_array_equal(handle.attrs["mode_band_indices"], analyzer.mode_band_indices)
