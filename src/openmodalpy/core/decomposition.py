"""Lift / metric / weighted second-order operator seam.

POD, mPOD and ST-POD are the same weighted second-order decomposition applied
to different lifts of the raw snapshot data. This module names those pieces:

- a **lift** (``.kind``, ``.apply``) that maps centered snapshots into the
  space where the second-order problem is posed;
- a **spatial metric** that defines the inner product, with ``.tile(d)`` for
  delay-embedded (lifted) spaces;
- **``weighted_second_order``**, the single solver both the eigh and SVD
  operator routes go through;
- **``spod_single_frequency``**, the single-frequency SPOD eigenproblem.

Every eigh route drops eigenvalues at or below the relative cutoff
``n_kernel * eps * lambda_max`` (numerical rank of the correlation matrix).
``n_keep`` still truncates the leading end after that filter.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np
import scipy.linalg

from openmodalpy.core.base import (
    _coerce_spatial_weights,
    canonicalize_modes,
    compute_reduced_svd,
    require_spatial_metric,
)
from openmodalpy.core.threads import apply_blas_limit


@runtime_checkable
class Lift(Protocol):
    """A named transformation of centered snapshots into an analysis space."""

    kind: str

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Return the lifted matrix (samples × lifted features)."""
        ...


class IdentityLift:
    """Centered snapshots as-is — the POD lift."""

    kind = "identity_centered_snapshots"

    def apply(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data)


class DelayEmbeddingLift:
    """Block-Hankel (delay) lift — the ST-POD lift.

    Each output row is a stack of ``embedding_dim`` consecutive snapshots, so
    the returned matrix has shape ``(m, d * Nspace)`` with
    ``m = Ns - d + 1``.
    """

    kind = "delay_embedding"

    def __init__(self, embedding_dim: int):
        if embedding_dim < 2:
            raise ValueError(f"embedding_dim must be >= 2, got {embedding_dim}")
        self.embedding_dim = int(embedding_dim)

    def apply(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"DelayEmbeddingLift expects 2D data, got shape {data.shape}")
        n_snapshots, n_space = data.shape
        d = self.embedding_dim
        if d >= n_snapshots:
            raise ValueError(f"embedding_dim ({d}) must be < number of snapshots ({n_snapshots})")
        m = n_snapshots - d + 1
        # col_idx[k, j] = j + k → snapshot row for delay block k, column j
        col_idx = np.arange(m)[np.newaxis, :] + np.arange(d)[:, np.newaxis]
        # (d, m, Nspace) → (m, d, Nspace) → (m, d*Nspace)
        stacked = data[col_idx].transpose(1, 0, 2).reshape(m, d * n_space)
        return np.ascontiguousarray(stacked)


class BandFilteredLift:
    """Temporal band-pass lift — the mPOD lift for one frequency band.

    When constructed without band edges the object still carries the paper's
    ``kind`` string for metadata; call ``apply`` only after setting the band
    (via constructor args) so the FFT mask is well-defined.
    """

    kind = "multiscale_filtered_snapshots"

    def __init__(
        self,
        f_low: float | None = None,
        f_high: float | None = None,
        dt: float | None = None,
        *,
        is_last: bool = False,
    ):
        self.f_low = f_low
        self.f_high = f_high
        self.dt = dt
        self.is_last = bool(is_last)

    def mask(self, n_snapshots: int) -> np.ndarray:
        """Boolean rfft-bin mask for this band.

        Exposed so a caller can detect an empty band without duplicating the
        half-open/closed edge convention that ``apply`` uses.
        """
        if self.f_low is None or self.f_high is None or self.dt is None:
            raise ValueError("BandFilteredLift requires f_low, f_high and dt (pass them to the constructor).")
        freq = np.fft.rfftfreq(int(n_snapshots), d=float(self.dt))
        if self.is_last:
            return (freq >= self.f_low) & (freq <= self.f_high)
        return (freq >= self.f_low) & (freq < self.f_high)

    def apply(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        n_snapshots = data.shape[0]
        mask = self.mask(n_snapshots)
        qhat = np.fft.rfft(data, axis=0)
        qhat_band = np.zeros_like(qhat)
        qhat_band[mask, :] = qhat[mask, :]
        return np.real(np.fft.irfft(qhat_band, n=n_snapshots, axis=0))


class SpatialMetric:
    """Diagonal spatial inner-product weights (the POD/SPOD metric W)."""

    def __init__(self, weights: np.ndarray):
        # Validate on the raw input before any real cast — complex would
        # otherwise truncate under ComplexWarning and store only the real part.
        require_spatial_metric(weights)
        self.weights = np.asarray(weights, dtype=float).reshape(-1)

    def tile(self, d: int) -> np.ndarray:
        """Repeat the metric ``d`` times for a delay-embedded space (I_d ⊗ W)."""
        if d < 1:
            raise ValueError(f"tile count must be >= 1, got {d}")
        return np.tile(self.weights, int(d))


def _as_weight_vector(metric: SpatialMetric | np.ndarray, n_space: int) -> np.ndarray:
    if isinstance(metric, SpatialMetric):
        metric = metric.weights
    return _coerce_spatial_weights(metric, n_space)


def _significant_eigenvalue_mask(
    eigenvalues: np.ndarray,
    n_kernel: int,
) -> np.ndarray:
    """Keep eigenvalues above ``n_kernel * eps * lambda_max``.

    This is the numerical rank of the correlation (Gram) matrix that was
    factored, not of the snapshot data. ``n_kernel`` is the dimension of that
    matrix (``n_samples`` on the temporal branch, ``n_space`` on the spatial
    branch). ``eps`` is machine epsilon of the working real dtype.
    """
    real = np.asarray(np.real(eigenvalues))
    if real.size == 0:
        return np.zeros(0, dtype=bool)
    lam_max = float(np.max(real))
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        return np.zeros(real.shape, dtype=bool)
    work = real.dtype if real.dtype.kind == "f" else np.dtype(float)
    eps = float(np.finfo(work).eps)
    cutoff = float(n_kernel) * eps * lam_max
    return real > cutoff


def weighted_second_order(
    data: np.ndarray,
    metric: SpatialMetric | np.ndarray,
    *,
    method: Literal["eigh", "svd"] = "eigh",
    drop_nonpositive: bool = True,
    n_keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the weighted second-order problem on lifted data.

    Parameters
    ----------
    data
        Lifted snapshot matrix, shape ``(n_samples, n_space)``. Samples run
        along axis 0 (time, Fourier realizations, or Hankel columns).
    metric
        Spatial (or already-tiled lifted) weights as a ``SpatialMetric`` or
        a 1D array of length ``n_space``.
    method
        ``"eigh"`` — covariance / Gram kernel eigenproblem (POD, mPOD,
        PSD-POD). ``"svd"`` — weighted SVD of the data matrix (ST-POD); use
        this rather than squaring a Hankel matrix.
    drop_nonpositive
        Retained for call-site compatibility. The eigh routes always drop
        eigenvalues at or below the relative cutoff
        ``n_kernel * eps * lambda_max``; this flag no longer selects an
        absolute floor or a keep-all policy.
    n_keep
        If set, keep only the leading ``n_keep`` modes inside the solver
        (mPOD / ST-POD / PSD-POD). POD passes None and truncates after.

    Returns
    -------
    modes, eigenvalues, time_coefficients
        ``modes`` has shape ``(n_space, r)``, ``eigenvalues`` shape ``(r,)``,
        ``time_coefficients`` shape ``(n_samples, r)``. Rank-deficient input
        returns fewer than ``n_keep`` / ``n_modes_save`` modes — the honest
        count of eigenvalues above the relative cutoff.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"weighted_second_order expects 2D data, got shape {data.shape}")
    with apply_blas_limit():
        if method == "svd":
            return _solve_svd(data, metric, n_keep=n_keep)
        if method == "eigh":
            return _solve_eigh(
                data,
                metric,
                n_keep=n_keep,
            )
    raise ValueError(f"method must be 'eigh' or 'svd', got {method!r}")


def _solve_eigh(
    data: np.ndarray,
    metric: SpatialMetric | np.ndarray,
    *,
    n_keep: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_space = data.shape
    weights = _as_weight_vector(metric, n_space)
    sqrt_weights = np.sqrt(np.maximum(weights, 1e-12))
    data_weighted = data * sqrt_weights

    # Complex ensembles (PSD-POD Fourier realizations) use the Hermitian
    # temporal kernel and the reconstruction that path has always used.
    if np.iscomplexobj(data):
        return _solve_eigh_complex(
            data,
            data_weighted,
            weights,
            n_samples,
            n_keep=n_keep,
        )

    use_temporal = n_samples < n_space
    if use_temporal:
        n_kernel = n_samples
        kernel = np.dot(data_weighted, data_weighted.T) / n_samples
        eigenvalues, vectors = scipy.linalg.eigh(kernel)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        vectors = vectors[:, order]
        # Recompute each eigenvalue as the Rayleigh quotient v.K.v. The
        # quotient is stationary at an eigenvector, so a first-order error in
        # the vector costs only a second-order error in the value: it is a
        # more accurate eigenvalue than the one LAPACK returns, which is what
        # makes the rank test below trustworthy for a value near the cutoff.
        # On well-conditioned data it agrees with LAPACK to the last bit; it can
        # differ by about 1e-15 relative when the kernel is poorly conditioned.
        # The spatial branch below and the complex path do the same thing.
        eigenvalues = np.sum(vectors * (kernel @ vectors), axis=0)

        keep = _significant_eigenvalue_mask(eigenvalues, n_kernel)
        eigenvalues = eigenvalues[keep]
        vectors = vectors[:, keep]
        if eigenvalues.size == 0:
            return (
                np.empty((n_space, 0)),
                np.empty((0,)),
                np.empty((n_samples, 0)),
            )

        if n_keep is not None:
            take = min(int(n_keep), eigenvalues.size)
            eigenvalues = eigenvalues[:take]
            vectors = vectors[:, :take]

        # After the relative filter every eigenvalue is strictly positive.
        safe = eigenvalues * n_samples
        normalization = 1.0 / np.sqrt(safe)
        weighted_modes = np.dot(data_weighted.T, vectors) * normalization
        modes = weighted_modes / sqrt_weights[:, np.newaxis]
        coeffs = vectors * np.sqrt(safe)
    else:
        n_kernel = n_space
        kernel = np.dot(data_weighted.T, data_weighted) / n_samples
        eigenvalues, weighted_modes = scipy.linalg.eigh(kernel)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        weighted_modes = weighted_modes[:, order]
        eigenvalues = np.sum(weighted_modes * (kernel @ weighted_modes), axis=0)

        keep = _significant_eigenvalue_mask(eigenvalues, n_kernel)
        eigenvalues = eigenvalues[keep]
        weighted_modes = weighted_modes[:, keep]
        if eigenvalues.size == 0:
            return (
                np.empty((n_space, 0)),
                np.empty((0,)),
                np.empty((n_samples, 0)),
            )

        if n_keep is not None:
            take = min(int(n_keep), eigenvalues.size)
            eigenvalues = eigenvalues[:take]
            weighted_modes = weighted_modes[:, :take]

        modes = weighted_modes / sqrt_weights[:, np.newaxis]
        coeffs = np.dot(data_weighted, weighted_modes)

    modes, coeffs = canonicalize_modes(np.real(modes), np.real(coeffs))
    return modes, np.real(eigenvalues), coeffs


def _solve_eigh_complex(
    data: np.ndarray,
    data_weighted: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    *,
    n_keep: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hermitian temporal-kernel path used by PSD-POD Fourier ensembles."""
    n_space = data.shape[1]
    n_kernel = n_samples
    kernel = (data_weighted @ data_weighted.conj().T) / n_samples
    # eigh on a Hermitian Gram matrix — same contract as the real POD path.
    eigenvalues, eigenvectors = scipy.linalg.eigh(kernel)
    order = np.argsort(eigenvalues.real)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvalues = np.sum(
        np.conj(eigenvectors) * (kernel @ eigenvectors), axis=0
    ).real

    keep = _significant_eigenvalue_mask(eigenvalues, n_kernel)
    eigenvalues = eigenvalues[keep]
    eigenvectors = eigenvectors[:, keep]
    if eigenvalues.size == 0:
        return (
            np.empty((n_space, 0), dtype=data.dtype),
            np.empty((0,)),
            np.empty((n_samples, 0), dtype=data.dtype),
        )

    if n_keep is not None:
        take = min(int(n_keep), eigenvalues.size)
        eigenvalues = eigenvalues[:take]
        eigenvectors = eigenvectors[:, :take]

    eigenvalues = np.real_if_close(eigenvalues)
    safe_eigs = np.real(eigenvalues)
    modes = (data.conj().T @ eigenvectors) / np.sqrt(safe_eigs * n_samples)
    coeffs = data.conj() @ (weights[:, np.newaxis] * modes)
    modes, coeffs = canonicalize_modes(modes, coeffs)
    return modes, np.asarray(eigenvalues), coeffs


def _solve_svd(
    data: np.ndarray,
    metric: SpatialMetric | np.ndarray,
    *,
    n_keep: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted SVD route (ST-POD). ``data`` is samples × lifted features."""
    n_samples, n_space = data.shape
    weights = _as_weight_vector(metric, n_space)
    sqrt_weights = np.sqrt(np.maximum(weights, 1e-12))
    data_weighted = data * sqrt_weights

    n_min = min(data_weighted.shape)
    if n_keep is None:
        k = max(n_min - 1, 0)
    else:
        k = min(int(n_keep), max(n_min - 1, 0))
    if k < 1:
        return (
            np.empty((n_space, 0)),
            np.empty((0,)),
            np.empty((n_samples, 0)),
        )

    # SVD of the feature × sample matrix so left singular vectors are modes.
    u_full, sigma_full, vt_full = compute_reduced_svd(data_weighted.T, k)
    sigma = sigma_full[:k]
    u = u_full[:, :k]
    vt = vt_full[:k, :]

    eigenvalues = (sigma**2) / n_samples
    modes = u / sqrt_weights[:, np.newaxis]
    coeffs = (vt * sigma[:, np.newaxis]).T
    modes, coeffs = canonicalize_modes(np.real(modes), np.real(coeffs))
    return modes, np.real(eigenvalues), coeffs


def spod_single_frequency(
    qhat: np.ndarray,
    nblocks: int,
    dst: float,
    w: np.ndarray,
    *,
    num_modes: int | None = None,
    return_psi: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SPOD eigenproblem at one frequency (shared serial / parallel body).

    ``qhat`` is ``(n_space, n_blocks)``; ``dst`` is the spectral weight in
    ``1/sqrt(nblocks * dst)``. Optional ``num_modes`` truncates after sorting;
    ``return_psi`` also returns the block-space eigenvectors.
    """
    with apply_blas_limit():
        x = qhat / np.sqrt(nblocks * dst)
        w_col = _coerce_spatial_weights(w, qhat.shape[0]).reshape(-1, 1)
        xprime_w = np.conj(x).T * w_col.T  # X_f^H * W
        m = xprime_w @ x
        lambda_tilde, psi = np.linalg.eigh(m)
        idx = lambda_tilde.argsort()[::-1]
        lambda_tilde = lambda_tilde[idx]
        psi = psi[:, idx]
        if num_modes is not None:
            keep = min(int(num_modes), len(lambda_tilde))
            lambda_tilde = lambda_tilde[:keep]
            psi = psi[:, :keep]
        inv_sqrt_lambda = np.zeros_like(lambda_tilde)
        # n_kernel is the block dimension of the CSD matrix that was factored.
        mask = _significant_eigenvalue_mask(lambda_tilde, nblocks)
        inv_sqrt_lambda[mask] = 1.0 / np.sqrt(lambda_tilde[mask])
        phi = x @ (psi * inv_sqrt_lambda[np.newaxis, :])
        if return_psi:
            return phi, np.abs(lambda_tilde), psi
        return phi, np.abs(lambda_tilde)
