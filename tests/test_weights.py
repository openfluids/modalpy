import numpy as np
from modalpy.core.base import calculate_uniform_weights, calculate_polar_weights


def test_uniform_weights_1d_vs_2d():
    x = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, 2.0, 3)
    x2d = np.repeat(x[:, None], len(y), axis=1)
    y2d = np.repeat(y[None, :], len(x), axis=0)
    w_1d = calculate_uniform_weights(x, y)
    w_2d = calculate_uniform_weights(x2d, y2d)
    assert w_1d.shape == (len(x) * len(y), 1)
    assert np.array_equal(w_1d, w_2d)


def test_polar_weights_1d_vs_2d():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    x2d = np.repeat(x[:, None], len(y), axis=1)
    y2d = np.repeat(y[None, :], len(x), axis=0)
    w_1d = calculate_polar_weights(x, y, use_parallel=False)
    w_2d = calculate_polar_weights(x2d, y2d, use_parallel=False)
    assert w_1d.shape == (len(x) * len(y), 1)
    assert np.allclose(w_1d, w_2d)


def test_weights_with_npz_grid_fixture(tmp_path):
    x1d = np.array([0.0, 0.5, 1.5, 3.0])
    y1d = np.array([0.0, 0.25, 0.75, 1.5])
    x2d = np.repeat(x1d[:, None], len(y1d), axis=1)
    y2d = np.repeat(y1d[None, :], len(x1d), axis=0)

    fixture_path = tmp_path / "grid_fixture.npz"
    np.savez(fixture_path, x=x2d, y=y2d)

    npz = np.load(fixture_path)
    x2d = npz["x"]
    y2d = npz["y"]
    x1d = x2d[:, 0]
    y1d = y2d[0, :]
    w_uniform_1d = calculate_uniform_weights(x1d, y1d)
    w_uniform_2d = calculate_uniform_weights(x2d, y2d)
    assert np.allclose(w_uniform_1d, w_uniform_2d)
    w_polar_1d = calculate_polar_weights(x1d, y1d, use_parallel=False)
    w_polar_2d = calculate_polar_weights(x2d, y2d, use_parallel=False)
    assert np.allclose(w_polar_1d, w_polar_2d)


def test_uniform_weights_3d_length():
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0])
    w = calculate_uniform_weights(x, y, z)
    assert w.shape == (len(x) * len(y) * len(z), 1)
