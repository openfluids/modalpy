"""SVD route relative cutoff: honest rank on null space, keeps weak modes.

The eigh path already drops eigenvalues below ``n_kernel * eps * lambda_max``.
The SVD path applies the same relative scale in the singular-value domain
(``sigma > n_kernel * eps * sigma_max``), so rank-deficient data returns the
honest mode count without deleting modes the SVD route exists to recover.
"""

from __future__ import annotations

import numpy as np

from openmodalpy.core.decomposition import weighted_second_order


def test_both_routes_return_exact_rank_on_rank3_data():
    """Exactly rank-3 snapshots: both routes keep exactly 3 modes."""
    rng = np.random.default_rng(0)
    n_s, n_x = 40, 400
    base = rng.standard_normal((n_s, 3)) @ rng.standard_normal((3, n_x))
    assert np.linalg.matrix_rank(base) == 3
    w = np.ones(n_x)
    for method in ("eigh", "svd"):
        modes, eigs, _ = weighted_second_order(base, w, method=method)
        assert modes.shape[1] == 3, f"{method} kept {modes.shape[1]}, expected 3"
        assert eigs.shape == (3,)


def test_svd_route_keeps_planted_mode_at_singular_ratio_1e_10():
    """A genuine mode at singular-value ratio 1e-10 survives the SVD floor.

    That ratio sits at eigenvalue ratio 1e-20 — below the eigh-style floor
    ``n_kernel * eps`` (~1e-14) but above the SVD floor
    ``(n_kernel * eps)**2`` (~1e-28). Recovering it is why the floor must
    live in the singular-value domain.
    """
    rng = np.random.default_rng(0)
    n_s, n_x = 40, 400
    base = rng.standard_normal((n_s, 3)) @ rng.standard_normal((3, n_x))
    weak_dir = rng.standard_normal(n_x)
    weak_time = rng.standard_normal(n_s)
    outer = np.outer(weak_time, weak_dir)
    ratio = 1e-10
    q = base + ratio * np.linalg.norm(base) / np.linalg.norm(outer) * outer
    target = weak_dir / np.linalg.norm(weak_dir)
    modes, _eigs, _ = weighted_second_order(q, np.ones(n_x), method="svd")
    assert modes.shape[1] >= 4, f"expected at least 4 modes, got {modes.shape[1]}"
    corr = max(
        (abs(float(modes[:, k] @ target)) / float(np.linalg.norm(modes[:, k])) for k in range(modes.shape[1])),
        default=0.0,
    )
    assert corr > 0.9, f"planted mode lost (best corr {corr:.6f})"
