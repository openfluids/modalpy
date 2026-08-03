import warnings

import numpy as np
import pytest

from openmodalpy.core.base import (
    _coerce_spatial_weights,
    calculate_polar_weights,
    calculate_uniform_weights,
    require_spatial_metric,
)
from openmodalpy.core.decomposition import SpatialMetric, _as_weight_vector


def test_square_weight_matrix_yields_diagonal():
    """A diagonal spatial metric stored as a full matrix keeps its diagonal."""
    diag = np.array([0.5, 1.0, 2.0, 0.25])
    W = np.diag(diag)
    col = _coerce_spatial_weights(W, 4).reshape(-1, 1)
    vec = _as_weight_vector(W, 4)
    assert col.shape == (4, 1)
    assert vec.shape == (4,)
    np.testing.assert_allclose(col.ravel(), diag)
    np.testing.assert_allclose(vec, diag)


def test_complex_weights_are_rejected_not_truncated():
    """A complex metric must fail loudly rather than lose its imaginary part."""
    W = np.array([1.0 + 0j, 2.0 + 1j, 3.0 + 0j])
    for entry in (
        lambda: require_spatial_metric(W),
        lambda: _coerce_spatial_weights(W, 3),
        lambda: _as_weight_vector(W, 3),
        lambda: _as_weight_vector(SpatialMetric(W), 3),
        lambda: SpatialMetric(W),
    ):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="complex"):
                entry()
        assert not any(w.category is np.exceptions.ComplexWarning for w in rec)


def test_three_d_weights_with_equal_space_and_components():
    """Per-component 3-D weights must not fall through into a second diagonal.

    When the stacked (n, k) matrix is square (n == k), the 3-D branch alone
    is the correct route: flatten the stacked diagonals, do not re-diag.
    """
    w3 = np.stack([np.diag([1.0, 2.0]), np.diag([3.0, 4.0])], axis=2)
    got = _coerce_spatial_weights(w3, 4)
    np.testing.assert_allclose(got, [1.0, 3.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="n_space=2"):
        _coerce_spatial_weights(w3, 2)


def test_three_d_per_component_route_through_the_seam():
    """3-D per-component weights reach the seam as stacked diagonals.

    Pins ``_as_weight_vector`` (not only ``_coerce_spatial_weights``) so a
    future narrowing of the seam cannot drop the 3-D route silently.
    """
    # shape (3, 3, 2): two component diagonals -> n_space=6
    w3 = np.stack(
        [np.diag([1.0, 2.0, 3.0]), np.diag([4.0, 5.0, 6.0])],
        axis=2,
    )
    got = _as_weight_vector(w3, 6)
    np.testing.assert_allclose(got, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])


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
