"""Tests for the lift / metric / weighted-second-order seam."""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy.core.decomposition import (
    BandFilteredLift,
    DelayEmbeddingLift,
    IdentityLift,
    SpatialMetric,
    weighted_second_order,
)


def test_identity_lift_kind_and_apply():
    lift = IdentityLift()
    assert lift.kind == "identity_centered_snapshots"
    q = np.arange(12.0).reshape(4, 3)
    out = lift.apply(q)
    np.testing.assert_array_equal(out, q)


def test_delay_embedding_lift_shape_and_kind():
    lift = DelayEmbeddingLift(embedding_dim=3)
    assert lift.kind == "delay_embedding"
    # snapshots: t=0..4, space=2
    data = np.arange(10.0).reshape(5, 2)
    lifted = lift.apply(data)
    # m = 5 - 3 + 1 = 3 rows, d*Nspace = 6 cols
    assert lifted.shape == (3, 6)
    # first sample: stack of rows 0,1,2
    np.testing.assert_array_equal(lifted[0], np.concatenate([data[0], data[1], data[2]]))
    np.testing.assert_array_equal(lifted[2], np.concatenate([data[2], data[3], data[4]]))


def test_delay_embedding_rejects_small_series():
    lift = DelayEmbeddingLift(embedding_dim=5)
    with pytest.raises(ValueError, match="embedding_dim"):
        lift.apply(np.zeros((4, 2)))


def test_band_filtered_lift_kind_and_passband():
    lift = BandFilteredLift()
    assert lift.kind == "multiscale_filtered_snapshots"
    # A pure tone at 1 Hz, dt=0.1 → Nyquist 5 Hz
    t = np.arange(64) * 0.1
    signal = np.sin(2 * np.pi * 1.0 * t)[:, None]
    band = BandFilteredLift(f_low=0.5, f_high=1.5, dt=0.1, is_last=True)
    filtered = band.apply(signal)
    # Energy should remain concentrated near the tone
    assert filtered.shape == signal.shape
    assert np.linalg.norm(filtered) > 0.5 * np.linalg.norm(signal)

    out_of_band = BandFilteredLift(f_low=3.0, f_high=4.0, dt=0.1, is_last=True)
    quiet = out_of_band.apply(signal)
    assert np.linalg.norm(quiet) < 0.1 * np.linalg.norm(signal)


def test_spatial_metric_tile():
    m = SpatialMetric(np.array([1.0, 2.0, 3.0]))
    tiled = m.tile(2)
    np.testing.assert_array_equal(tiled, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


def test_weighted_second_order_eigh_matches_manual_unweighted():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((20, 5))
    q = q - q.mean(axis=0)
    w = np.ones(5)
    modes, eigs, coeffs = weighted_second_order(q, w, method="eigh")
    # Spatial kernel path (Ns > Nspace): eigenvalues of (X^T X)/Ns
    gram = (q.T @ q) / q.shape[0]
    ref_eigs = np.sort(np.linalg.eigvalsh(gram))[::-1]
    np.testing.assert_allclose(eigs, ref_eigs, rtol=1e-10, atol=1e-12)
    assert modes.shape == (5, 5)
    assert coeffs.shape == (20, 5)


def test_weighted_second_order_drop_nonpositive_and_n_keep():
    rng = np.random.default_rng(1)
    # Rank-2 signal in 6D space
    a = rng.standard_normal((30, 2))
    b = rng.standard_normal((2, 6))
    q = a @ b
    q = q - q.mean(axis=0)
    modes, eigs, coeffs = weighted_second_order(
        q,
        np.ones(6),
        method="eigh",
        drop_nonpositive=True,
        n_keep=2,
    )
    assert eigs.size == 2
    assert modes.shape == (6, 2)
    assert coeffs.shape == (30, 2)
    assert np.all(eigs > 1e-12)


def test_weighted_second_order_svd_route():
    rng = np.random.default_rng(2)
    # samples × features (as DelayEmbeddingLift returns)
    data = rng.standard_normal((12, 8))
    modes, eigs, coeffs = weighted_second_order(
        data,
        np.ones(8),
        method="svd",
        n_keep=3,
    )
    assert modes.shape == (8, 3)
    assert eigs.shape == (3,)
    assert coeffs.shape == (12, 3)
    # eigenvalues = sigma^2 / n_samples
    sigma = np.linalg.svd(data, full_matrices=False, compute_uv=False)[:3]
    np.testing.assert_allclose(eigs, sigma**2 / 12, rtol=1e-10, atol=1e-12)
