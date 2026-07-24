import h5py
import numpy as np

from modalpy.core.io import MATDataLoader


def test_mat_loader_handles_flattened_space_by_time_matrix(tmp_path):
    file_path = tmp_path / "flattened.mat"
    q = np.arange(12, dtype=float).reshape(6, 2)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("u", data=q)
        f.create_dataset("x", data=np.array([0.0, 0.5, 1.0]))
        f.create_dataset("y", data=np.array([0.0, 1.0]))
        f.create_dataset("dt", data=np.array([[0.25]]))

    data = MATDataLoader().load(str(file_path))

    assert data["q"].shape == (2, 6)
    assert np.array_equal(data["q"], q.T)
    assert np.array_equal(data["x"], np.array([0.0, 0.5, 1.0]))
    assert np.array_equal(data["y"], np.array([0.0, 1.0]))
    assert data["Nx"] == 3
    assert data["Ny"] == 2
    assert data["Nz"] == 1
    assert data["Ns"] == 2
    assert np.isclose(data["dt"], 0.25)
