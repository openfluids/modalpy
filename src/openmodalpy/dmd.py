#!/usr/bin/env python3
"""
Dynamic Mode Decomposition (exact DMD) implementation.

The current analyzer implements standard exact DMD on raw shifted snapshots
with Euclidean least squares. It does not currently apply the spatial metric
``W`` or mean subtraction inside ``perform_dmd()``.
"""

# Standard library imports
import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import inspect
import warnings
from typing import Optional

import matplotlib.pyplot as plt

# Suppress contour warnings when no levels can be plotted
warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
import numpy as np  # noqa: E402

from openmodalpy.core.base import (  # noqa: E402
    BaseAnalyzer,
    add_inset_colorbar,
    compute_reduced_svd,
    format_mode_title,
    get_fig_aspect_ratio,
    make_result_filename,
    plot_isometric_slices_3d,
    plot_orthogonal_slices_3d,
    print_summary,
    reshape_mode_to_volume,
    resolve_volume_layout,
    style_spatial_axes,
)
from openmodalpy.core.config import (  # noqa: E402
    CMAP_DIV,
    CMAP_SEQ,
    FIG_DPI,
    FIGURES_DIR_DMD,
    RESULTS_DIR_DMD,
    require_existing_data_path,
)

# Try to import DNamiDataLoader for npz support
try:
    from openmodalpy.core.io import DNamiDataLoader
except ImportError:
    DNamiDataLoader = None


def _delay_embed(X, d):
    """Build delay-embedded (Hankel) matrix from snapshot matrix X.

    Parameters
    ----------
    X : ndarray, shape (n, m)
        Snapshot matrix with *n* spatial points and *m* time steps.
    d : int
        Number of delays (stack depth).  ``d=1`` returns *X* unchanged.

    Returns
    -------
    ndarray, shape (n*d, m-d+1)
    """
    n, m = X.shape
    cols = m - d + 1
    out = np.empty((n * d, cols), dtype=X.dtype)
    for i in range(d):
        out[i * n : (i + 1) * n, :] = X[:, i : i + cols]
    return out


def _dmd_pinv_rcond(shape, dtype) -> float:
    """Return ``max(shape) * finfo(dtype).eps`` (numpy.linalg.pinv default).

    Used as the relative singular-value floor: keep ``s_j > rcond * s[0]``.
    Not a public constructor parameter.
    """
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.complexfloating):
        dt = np.dtype(dt.type(0).real.dtype)
    elif not np.issubdtype(dt, np.floating):
        dt = np.dtype(np.float64)
    return max(shape) * np.finfo(dt).eps


# Relative SVD / pinv cutoff policy: numpy.linalg.pinv rcond convention.
# rcond = max(M, N) * finfo(dtype).eps; rank keeps s_j > rcond * s[0].
# Shape/dtype-dependent — computed per call via _dmd_pinv_rcond. Not a user knob.
DMD_PINV_RCOND = _dmd_pinv_rcond


def svht_lambda(beta):
    """Gavish–Donoho (2014) optimal hard-threshold coefficient (unknown noise).

    Parameters
    ----------
    beta : float
        Aspect ratio ``min(shape) / max(shape)`` of the snapshot-pair matrix,
        in ``(0, 1]``.

    Returns
    -------
    float
        ``lambda(beta)`` such that the hard threshold is
        ``tau = lambda(beta) * median(singular values)``.

    Notes
    -----
    ``lambda(1) = 4/sqrt(3) ≈ 2.309401`` is the square-matrix value, not a
    universal constant. As ``beta -> 0``, ``lambda -> sqrt(2)``.
    """
    beta = float(beta)
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"svht_lambda requires 0 < beta <= 1, got {beta!r}")
    return float(np.sqrt(2.0 * (beta + 1.0) + 8.0 * beta / ((beta + 1.0) + np.sqrt(beta * beta + 14.0 * beta + 1.0))))


class DMDAnalyzer(BaseAnalyzer):
    """Exact Dynamic Mode Decomposition analyzer.

    The current implementation is intentionally narrow: raw snapshot pairs are
    regressed in Euclidean norm and the resulting modes are sorted by
    ``|lambda|``.
    """

    def __init__(
        self,
        file_path,
        results_dir=RESULTS_DIR_DMD,
        figures_dir=FIGURES_DIR_DMD,
        data_loader=None,
        spatial_weight_type="auto",
        n_modes_save=10,
        rank=None,
        energy_fraction=0.999,
        use_parallel=True,
    ):
        super().__init__(
            file_path=file_path,
            nfft=1,
            overlap=0.0,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            use_parallel=use_parallel,
        )
        self.n_modes_save = n_modes_save
        self.rank = rank
        self.energy_fraction = float(energy_fraction)
        self.modes = np.array([])
        self.eigenvalues = np.array([])
        self.time_coefficients = np.array([])
        self.analysis_type = "dmd"
        self.temporal_mean = np.array([])
        # Store modal amplitudes (|b|) after perform_dmd()
        self.amplitudes = np.array([])
        # Continuous-time eigenvalues: omega = log(lambda) / dt
        self.omega = np.array([])
        # Numerical rank used by the last perform_dmd() (after criterion + rcond)
        self.effective_rank = 0
        # Algorithm settings (written by perform_dmd, read by metadata)
        self._dmd_method = "ls"
        self._dmd_delays = 1
        self._dmd_named_variant = "dmd"

    def _svd_request_rank(self, shape) -> int:
        """How many singular triplets to request from ``compute_reduced_svd``.

        Explicit integer ``rank`` and the deprecated default (``None`` →
        ``n_modes_save``) can use a truncated path; spectrum-based criteria
        need the full thin SVD of ``X1``.
        """
        max_r = min(shape)
        rank = self.rank
        if rank is None:
            # Deprecated default: same request as pre-decoupling code.
            return min(int(self.n_modes_save), max_r)
        if isinstance(rank, (int, np.integer)):
            if int(rank) < 1:
                raise ValueError(f"rank must be >= 1 when given as int, got {rank!r}")
            return min(int(rank), max_r)
        if rank in ("svht", "energy"):
            return max_r
        raise ValueError(f"Unknown rank {rank!r}; use None, a positive int, 'svht', or 'energy'.")

    def _resolve_rank(self, s, shape, rcond):
        """Map singular values + ``self.rank`` to ``(effective_r, r_requested)``.

        Every path floors by the relative threshold ``s_j > rcond * s[0]``.
        ``r_requested`` is the criterion's target before that floor (used for
        the under-rank warning).

        ``rank=None`` (deprecated) still resolves to
        ``min(n_modes_save, min(shape))`` then the rcond floor — bit-for-bit the
        pre-decoupling default. Explicit int / ``"svht"`` / ``"energy"`` never
        consult ``n_modes_save``.
        """
        max_r = min(shape)
        if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0.0:
            return 0, max_r

        r_numeric = int(np.sum(s > (rcond * s[0])))
        rank = self.rank

        if rank is None:
            # Deprecated default: reproduce today's coupling exactly.
            r_requested = min(int(self.n_modes_save), max_r)
            r = min(r_requested, r_numeric)
            return r, r_requested

        if isinstance(rank, (int, np.integer)):
            r_requested = min(int(rank), max_r)
            r = min(r_requested, r_numeric)
            return r, r_requested

        if rank == "svht":
            beta = min(shape) / max(shape)
            tau = svht_lambda(beta) * float(np.median(s))
            r_svht = int(np.sum(s > tau))
            r = min(r_svht, r_numeric)
            return r, max(r_svht, 1) if r_svht > 0 else max_r

        if rank == "energy":
            frac = self.energy_fraction
            if not (0.0 < frac <= 1.0):
                raise ValueError(f"energy_fraction must be in (0, 1], got {frac!r}")
            energy = np.cumsum(s.astype(np.float64) ** 2)
            total = float(energy[-1])
            if total <= 0.0 or not np.isfinite(total):
                return 0, max_r
            # Smallest r with cumulative s^2 fraction >= energy_fraction.
            r_energy = int(np.searchsorted(energy / total, frac, side="left") + 1)
            r_energy = min(max(r_energy, 1), max_r, s.size)
            r = min(r_energy, r_numeric)
            return r, r_energy

        raise ValueError(f"Unknown rank {rank!r}; use None, a positive int, 'svht', or 'energy'.")

    def perform_dmd(self, method="ls", delays=1, named_variant=None):
        """Compute DMD on raw shifted snapshots.

        Parameters
        ----------
        method : ``"ls"`` | ``"tls"``
            ``"ls"``  — standard exact DMD (least-squares).
            ``"tls"`` — total least-squares DMD.
        delays : int, default 1
            Number of delay embeddings.  ``delays=1`` is standard DMD;
            ``delays>1`` builds a Hankel matrix before forming snapshot pairs.

        Notes
        -----
        - Uses ``q[:-1]`` and ``q[1:]`` directly as the paired data.
        - Does not subtract the temporal mean.
        - Does not use the spatial metric ``self.W`` in the regression.
        - Sorts modes by descending ``|lambda|``.
        - Truncation rank is controlled by ``self.rank``. Default ``rank=None``
          still uses ``min(n_modes_save, min(X1.shape))`` then the rcond floor
          (deprecated; pass an explicit int, ``"svht"``, or ``"energy"``).
          With an explicit ``rank``, ``n_modes_save`` only bounds how many
          modes are kept after sorting.
        """
        if method not in ("ls", "tls"):
            raise ValueError(f"Unknown method '{method}'; use 'ls' or 'tls'.")
        if delays < 1:
            raise ValueError("delays must be >= 1.")

        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        q = self.data["q"]
        n_snapshots = q.shape[0]
        X = q.T  # (n_spatial, n_time)

        # Delay embedding (Hankel lift)
        if delays > 1:
            if delays > n_snapshots - 2:
                raise ValueError(
                    f"delays={delays} too large for n_snapshots={n_snapshots}; "
                    "need at least 2 snapshot pairs after embedding "
                    f"(max delays = {n_snapshots - 2})."
                )
            if delays >= n_snapshots // 2:
                warnings.warn(
                    f"delays={delays} is large relative to n_snapshots={n_snapshots}; "
                    "the effective snapshot count will be small.",
                    stacklevel=2,
                )
            X = _delay_embed(X, delays)

        X1 = X[:, :-1]
        X2 = X[:, 1:]

        # Default rank=None still couples to n_modes_save (deprecated). Explicit
        # int / "svht" / "energy" select the operator without consulting it.
        if self.rank is None:
            warnings.warn(
                "the DMD truncation rank currently defaults to n_modes_save, which "
                "couples a plotting parameter to the numerics; pass rank explicitly "
                "(an int, 'svht', or 'energy'). This default will change in a future "
                "release.",
                DeprecationWarning,
                stacklevel=2,
            )
        r_svd = self._svd_request_rank(X1.shape)
        u, s, vh = compute_reduced_svd(X1, r_svd)
        rcond = DMD_PINV_RCOND(X1.shape, s.dtype if s.size else X1.dtype)
        r, r_requested = self._resolve_rank(s, X1.shape, rcond)
        # Order matters. The relative test always keeps s[0] (s[0] > rcond * s[0]
        # reduces to 1 > rcond), so r reaches 0 only when the spectrum itself is
        # unusable: empty, non-finite, or all zero. Bumping r back to 1 there would
        # divide by that unusable s[0] and surface as an opaque LinAlgError out of
        # np.linalg.eig, so the degenerate case has to return before the bump.
        if r < r_requested:
            # RuntimeWarning, not UserWarning: this reports a numerical property of
            # the data, not a misuse of the API. The caller asking for more modes
            # than the data supports is normal and expected -- the analytic cases
            # are rank-1 by construction -- so it belongs in the same category numpy
            # uses for conditioning and precision notices.
            warnings.warn(
                f"DMD effective rank {r} is below the requested {r_requested} "
                f"(relative singular-value threshold rcond={rcond:.3e}).",
                RuntimeWarning,
                stacklevel=2,
            )
        self.effective_rank = int(r)
        if r == 0:
            self.eigenvalues = np.array([])
            self.omega = np.array([])
            self.modes = np.array([])
            self.time_coefficients = np.array([])
            self.amplitudes = np.array([])
            self._dmd_method = method
            self._dmd_delays = delays
            self._dmd_named_variant = named_variant or "dmd"
            return

        u_r = u[:, :r]
        s_r = s[:r]
        v_r = vh[:r].conj().T

        # Reduced operator
        if method == "tls":
            Z = np.vstack([X1, X2])
            Uz, _, _ = compute_reduced_svd(Z, r)
            Uz = Uz[:, :r]
            n1 = X1.shape[0]
            U11 = Uz[:n1, :]
            U21 = Uz[n1:, :]
            # Project into the reduced basis so atilde is (r, r)
            u_r_H_U11 = u_r.conj().T @ U11
            atilde = (u_r.conj().T @ U21) @ np.linalg.pinv(u_r_H_U11, rcond=rcond)
        else:
            atilde = (u_r.conj().T @ X2 @ v_r) / s_r

        eigvals, w = np.linalg.eig(atilde)
        # Exact DMD mode recovery.  For TLS this is an approximation: the
        # eigenvalues benefit from the TLS operator, while the spatial modes
        # are projected through the LS basis.  This is standard practice;
        # see Hemati et al. (2017) for a discussion of TLS-DMD variants.
        modes = X2 @ (v_r / s_r) @ w

        # Continuous-time eigenvalues (guard against log(0))
        dt = self._require_dt()
        safe_eigvals = np.where(np.abs(eigvals) > 0, eigvals, np.finfo(float).tiny)
        omega = np.log(safe_eigvals.astype(complex)) / dt

        # Amplitudes and time dynamics (use original snapshot count)
        b = np.linalg.pinv(modes, rcond=rcond) @ X[:, 0]
        t = np.arange(n_snapshots)
        time_dynamics = (b[:, None] * eigvals[:, None] ** t).T

        idx = np.argsort(np.abs(eigvals))[::-1]
        n_keep = min(self.n_modes_save, r)
        self.eigenvalues = eigvals[idx][:n_keep]
        self.omega = omega[idx][:n_keep]
        self.modes = modes[:, idx][:, :n_keep]
        self.time_coefficients = time_dynamics[:, idx][:, :n_keep]
        self.amplitudes = np.abs(b[idx][:n_keep])
        self._dmd_method = method
        self._dmd_delays = delays
        self._dmd_named_variant = named_variant or "dmd"

    def _get_algorithm_metadata(self) -> dict:
        """Describe the DMD contract currently implemented."""
        method = self._dmd_method
        delays = self._dmd_delays
        named_variant = self._dmd_named_variant
        variant = "tls_dmd" if method == "tls" else "exact_dmd"
        if named_variant == "hodmd":
            variant = "hodmd"
        elif named_variant == "tls_hodmd":
            variant = "tls_hodmd"
        elif delays > 1:
            variant = f"delay_embedded_{variant}"
        return {
            "lift_kind": "delay_embedding" if delays > 1 else "identity_paired_snapshots",
            "paired_data_contract": "raw_shifted_snapshots",
            "uses_mean_subtraction": False,
            "uses_spatial_metric_in_regression": False,
            "regression_norm": "frobenius",
            "mode_ranking": "abs_lambda_desc",
            "dmd_variant": variant,
            "dmd_named_variant": named_variant,
            "dmd_method": method,
            "dmd_delays": delays,
        }

    def save_results(self, filename=None):
        """Save DMD results to an HDF5 file."""
        if not filename:
            filename = make_result_filename(
                self.data_root,
                self.nfft,
                self.overlap,
                self.data.get("Ns", 0),
                self.analysis_type,
            )
        path = os.path.join(self.results_dir, filename)
        with h5py.File(path, "w") as f:
            f.attrs.update(self._get_metadata())
            f.create_dataset("eigenvalues", data=self.eigenvalues, compression="gzip")
            f.create_dataset("modes", data=self.modes, compression="gzip")
            f.create_dataset("time_coefficients", data=self.time_coefficients, compression="gzip")
            f.create_dataset("amplitudes", data=self.amplitudes, compression="gzip")
            if self.omega.size > 0:
                f.create_dataset("omega", data=self.omega, compression="gzip")
            f.create_dataset("x", data=self.data["x"], compression="gzip")
            f.create_dataset("y", data=self.data["y"], compression="gzip")
            if "z" in self.data and self.data["z"] is not None:
                f.create_dataset("z", data=self.data["z"], compression="gzip")
        print(f"DMD results saved to {path}")

    def load_results(self, filename=None):
        """Load DMD results from an HDF5 file."""
        if not filename:
            filename = make_result_filename(
                self.data_root,
                self.nfft,
                self.overlap,
                self.data.get("Ns", 0),
                self.analysis_type,
            )
        path = os.path.join(self.results_dir, filename)

        if not os.path.exists(path):
            # Try to auto-detect a results file for this variable and analysis type
            import glob

            pattern = os.path.join(self.results_dir, f"*_{self.analysis_type}.hdf5")
            matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if matches:
                path = matches[0]
                print(f"[Auto-detect] Using available results file: {path}")
            else:
                print(
                    f"[ERROR] No results file found for plotting in {self.results_dir} matching '*_{self.analysis_type}.hdf5'. Run with --compute first."
                )
                return  # Or: raise FileNotFoundError("No DMD results file found for plotting.")

        with h5py.File(path, "r") as f:

            def _decode_attr(name, default=None):
                if name not in f.attrs:
                    return default
                value = f.attrs[name]
                if isinstance(value, bytes):
                    return value.decode("utf-8")
                return value

            self.eigenvalues = f["eigenvalues"][:]
            self.modes = f["modes"][:]
            self.time_coefficients = f["time_coefficients"][:]
            # Load amplitudes if available (backward compatibility)
            if "amplitudes" in f:
                self.amplitudes = f["amplitudes"][:]
            else:
                self.amplitudes = np.abs(self.eigenvalues)
            # Load continuous-time eigenvalues if available
            if "omega" in f:
                self.omega = f["omega"][:]
            self._dmd_method = str(_decode_attr("dmd_method", "ls"))
            self._dmd_delays = int(_decode_attr("dmd_delays", 1))
            self._dmd_named_variant = str(_decode_attr("dmd_named_variant", "dmd"))
            # Load spatial coordinates if they exist
            if "x" in f:
                self.data["x"] = f["x"][:]
            if "y" in f:
                self.data["y"] = f["y"][:]
            if "z" in f:
                self.data["z"] = f["z"][:]
            if "dt" in f.attrs:
                self.data["dt"] = f.attrs["dt"]
            if "Ns" in f.attrs:
                self.data["Ns"] = int(f.attrs["Ns"])
            if "Nx" in f.attrs:
                self.data["Nx"] = int(f.attrs["Nx"])
            if "Ny" in f.attrs:
                self.data["Ny"] = int(f.attrs["Ny"])
            if "Nz" in f.attrs:
                self.data["Nz"] = int(f.attrs["Nz"])
        print(f"DMD results loaded from {path}")

    def _mode_freq(self, eigvals):
        """Return mode frequencies in Hz, or ``None`` when ``dt`` is unusable.

        Computes ``angle(eigvals) / (2π · dt)`` when ``self.data["dt"]`` is a
        positive finite scalar. Returns ``None`` (never raises) when ``dt`` is
        missing, ``None``, zero, or non-finite — suitable for optional title
        annotations. Physics-bearing plots must use :meth:`_require_dt` instead.
        """
        dt = self.data.get("dt") if self.data else None
        try:
            dt_f = float(dt)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(dt_f) or dt_f <= 0.0:
            return None
        return np.angle(eigvals) / (2 * np.pi * dt_f)

    def plot_eigenvalues(self):
        """Plot DMD eigenvalues in the complex plane."""
        if self.eigenvalues.size == 0:
            print("No eigenvalues to plot.")
            return
        plt.figure(figsize=(6, 6))
        plt.plot(self.eigenvalues.real, self.eigenvalues.imag, "bo")
        circle = plt.Circle((0, 0), 1.0, color="green", fill=False, linestyle="--")
        ax = plt.gca()
        ax.add_artist(circle)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
        ax.set_aspect("equal")
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.title("DMD Eigenvalues (Complex Plane)")
        fname = os.path.join(self.figures_dir, f"{self.data_root}_dmd_eigenvalues.png")
        plt.savefig(fname, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {fname}")

    def plot_eigenspectra(self):
        """Create composite spectra figure: eigenvalues circle, amplitude vs frequency and growth rate."""
        if self.eigenvalues.size == 0 or self.amplitudes.size == 0:
            print("No eigenvalue data to plot. Run perform_dmd() first.")
            return
        dt = self._require_dt()
        eigvals = self.eigenvalues
        amps = self.amplitudes
        amps_norm = amps / np.max(amps)
        freq = np.angle(eigvals) / (2 * np.pi * dt)
        growth = np.log(np.abs(eigvals)) / dt

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax_complex = fig.add_subplot(gs[0, :])
        ax_freq = fig.add_subplot(gs[1, 0])
        ax_growth = fig.add_subplot(gs[1, 1])

        # Complex eigenvalue plot
        ax_complex.plot(eigvals.real, eigvals.imag, "o", mfc="none", mec="brown")
        # Annotate every eigenvalue with its mode number; mark mean explicitly
        for k, lam in enumerate(eigvals):
            label = f"{k + 1}"
            if np.isclose(lam, 1 + 0j, atol=1e-3):
                label += " (mean)"
            ax_complex.text(lam.real, lam.imag, f" {label}", fontsize=7, color="black")
        idx_mean = int(np.argmin(np.abs(eigvals - 1)))
        ax_complex.text(eigvals.real[idx_mean], eigvals.imag[idx_mean], "  mean", color="red", fontsize=8, va="center")
        # Annotate first few oscillatory modes with frequency
        for k in range(min(4, len(eigvals))):
            if k == idx_mean:
                continue
            ax_complex.text(eigvals.real[k], eigvals.imag[k], f"  f={freq[k]:.2f}", fontsize=7, color="black")
        unit_circle = plt.Circle((0, 0), 1.0, color="brown", fill=False, linewidth=1.0)
        ax_complex.add_patch(unit_circle)
        ax_complex.axhline(0.0, color="k", linestyle="--", linewidth=0.5)
        ax_complex.axvline(0.0, color="k", linestyle="--", linewidth=0.5)
        ax_complex.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax_complex.set_ylabel(r"$\mathrm{Im}(\lambda)$")
        ax_complex.set_aspect("equal")
        ax_complex.set_title("DMD eigenvalues")

        # Amplitude vs frequency
        if "use_line_collection" in inspect.signature(ax_freq.stem).parameters:
            ax_freq.stem(
                freq,
                amps_norm,
                linefmt="brown",
                markerfmt="ro",
                basefmt=" ",
                use_line_collection=True,
            )
        else:
            ax_freq.stem(freq, amps_norm, linefmt="brown", markerfmt="ro", basefmt=" ")
        for k, (x, y) in enumerate(zip(freq, amps_norm)):
            ax_freq.text(x, y, f" {k + 1}", fontsize=6, rotation=45, va="bottom")
        ax_freq.set_xlabel("frequency")
        ax_freq.set_ylabel("normalized amplitude")
        ax_freq.set_yscale("log")
        ax_freq.set_title("Amplitude vs frequency")

        # Amplitude vs growth rate
        if "use_line_collection" in inspect.signature(ax_growth.stem).parameters:
            ax_growth.stem(
                growth,
                amps_norm,
                linefmt="brown",
                markerfmt="ro",
                basefmt=" ",
                use_line_collection=True,
            )
        else:
            ax_growth.stem(growth, amps_norm, linefmt="brown", markerfmt="ro", basefmt=" ")
        for k, (x, y) in enumerate(zip(growth, amps_norm)):
            ax_growth.text(x, y, f" {k + 1}", fontsize=6, rotation=45, va="bottom")
        ax_growth.set_xlabel("growth rate")
        ax_growth.set_yscale("log")
        ax_growth.set_title("Amplitude vs growth rate")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        fname_spec = os.path.join(self.figures_dir, f"{self.data_root}_dmd_eigenspectra.png")
        fig.savefig(fname_spec, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {fname_spec}")

    def plot_modes_detailed(
        self,
        plot_n_modes: int = 8,
        zero_phase_ref: bool = False,
        unwrap_phase: bool = False,
        ref_method: str = "max",
        show_cylinder: bool = False,
    ):
        """Plot real, imaginary, magnitude, and phase of several modes in a 4-row grid.

        Args:
            plot_n_modes: Number of modes to plot
            zero_phase_ref: If True, reference phase to 0
            unwrap_phase: If True, unwrap phase
            ref_method: Method for phase reference ('max' or 'mean')
            show_cylinder: If True, add cylinder mask at origin with radius 0.5
        """
        if self.modes.size == 0:
            print("No modes to plot. Run perform_dmd() first.")
            return
        n_modes = min(plot_n_modes, self.modes.shape[1])
        if n_modes == 0:
            print("No modes available to plot.")
            return

        nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        if self.modes.shape[0] != nx * ny or nx <= 1 or ny <= 1:
            print("Detailed mode plotting supports 2D data only.")
            return

        x_coords = self.data.get("x", np.arange(nx))
        y_coords = self.data.get("y", np.arange(ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")
        # Optional frequency annotation; modes still plot without a usable dt
        freq = self._mode_freq(self.eigenvalues[:n_modes])

        fig, axes = plt.subplots(4, n_modes, figsize=(3 * n_modes * fig_aspect, 12), squeeze=False)
        row_labels = ["real", "imaginary", "magnitude", "phase"]
        cmaps = [CMAP_DIV, CMAP_DIV, CMAP_SEQ, "twilight"]

        if x_coords.ndim == 1 and y_coords.ndim == 1:
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
        else:
            x_mesh, y_mesh = x_coords, y_coords
        # Optionally create cylinder mask
        if show_cylinder:
            distance = np.sqrt(x_mesh**2 + y_mesh**2)
            cylinder_mask = distance <= 0.5
        else:
            cylinder_mask = None

        for m in range(n_modes):
            vec = self.modes[:, m]
            # Phase processing with optional reference and unwrapping
            phase_arr = np.angle(vec)
            if zero_phase_ref:
                if ref_method == "max":
                    phase0 = phase_arr[np.argmax(np.abs(vec))]
                else:  # 'mean'
                    phase0 = np.mean(phase_arr)
                phase_arr = (phase_arr - phase0 + np.pi) % (2 * np.pi) - np.pi
            if unwrap_phase:
                phase_arr = np.unwrap(phase_arr)
            comps = [vec.real, vec.imag, np.abs(vec), phase_arr]
            for r, comp in enumerate(comps):
                ax = axes[r, m]
                comp2d = comp.reshape((nx, ny))
                if cylinder_mask is not None:
                    comp_plot = np.ma.array(comp2d, mask=cylinder_mask)
                else:
                    comp_plot = comp2d
                if r == 2:
                    vmin, vmax = 0.0, np.nanmax(comp_plot)
                elif r == 3:
                    vmin, vmax = -np.pi, np.pi
                else:
                    vmin, vmax = np.nanmin(comp_plot), np.nanmax(comp_plot)
                # Ensure levels are valid and strictly increasing
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    continue  # skip if invalid
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-12  # tiny range to allow contouring
                levels = np.linspace(vmin, vmax, 21)
                cf = ax.contourf(x_mesh, y_mesh, comp_plot, levels=levels, cmap=cmaps[r], extend="both")
                # Add line contours only if range is significant
                if vmax - vmin > 1e-12:
                    ax.contour(x_mesh, y_mesh, comp_plot, levels=levels[::4], colors="k", linewidths=0.4, alpha=0.4)

                # Add individual small colorbar inside the data area (upper right)
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                cax = inset_axes(ax, width="15%", height="6%", loc="upper right", borderpad=3)
                cb = fig.colorbar(cf, cax=cax, orientation="horizontal", format="%.2f")
                cb.ax.tick_params(labelsize=8, pad=1, colors="black")
                cb.ax.xaxis.set_ticks_position("top")
                cb.ax.xaxis.set_label_position("top")
                # Set custom ticks: min, 0, max (except for magnitude which starts at 0)
                if r == 2:  # magnitude
                    cb.set_ticks([0, vmax / 2, vmax])
                    cb.set_ticklabels(["0", f"{vmax / 2:.2f}", f"{vmax:.2f}"])
                elif r == 3:  # phase
                    cb.set_ticks([-np.pi, 0, np.pi])
                    cb.set_ticklabels(["-π", "0", "π"])
                else:  # real and imaginary
                    cb.set_ticks([vmin, 0, vmax])
                    cb.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])
                # Make colorbar background semi-transparent
                cax.patch.set_facecolor("black")
                cax.patch.set_alpha(0.7)

                # Optionally add cylinder overlay
                if show_cylinder:
                    cylinder = plt.Circle((0, 0), 0.5, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                    ax.add_patch(cylinder)
                # Phase zero-line overlay
                if r == 3 and vmax - vmin > 1e-12:
                    ax.contour(x_mesh, y_mesh, comp_plot, levels=[0.0], colors="white", linewidths=0.6)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if m == 0:
                    ax.set_ylabel(row_labels[r])
                if r == 0:
                    # Column header annotations
                    if m == 0:
                        header = "1 (mean)"
                    elif freq is not None:
                        header = f"{m + 1} (f={freq[m]:.2f})"
                    else:
                        header = f"{m + 1}"
                    ax.set_title(header)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        fname_modes = os.path.join(self.figures_dir, f"{self.data_root}_dmd_modes_detailed_{n_modes}_{var_name}.png")
        fig.savefig(fname_modes, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {fname_modes}")

    def plot_cumulative_energy(self):
        """Plot the cumulative energy captured by DMD modes (using |eigval|^2 as proxy)."""
        if self.eigenvalues.size == 0:
            print("No eigenvalues to plot. Run perform_dmd() first.")
            return
        # Use squared modulus as 'energy' proxy
        eigvals_abs2 = np.abs(self.eigenvalues) ** 2
        cumulative_energy = np.cumsum(eigvals_abs2) / np.sum(eigvals_abs2) * 100
        mode_indices = np.arange(1, len(self.eigenvalues) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, cumulative_energy, "o-", linewidth=2, markersize=6)
        plt.xlabel("Number of Modes")
        plt.ylabel("Cumulative |Eigval|^2 (%)")
        plt.title("Cumulative Energy of DMD Modes (|eigval|^2)")
        plt.grid(True, which="both", ls="--")
        plt.ylim(0, 105)
        fname = os.path.join(self.figures_dir, f"{self.data_root}_dmd_cumulative_energy.png")
        plt.savefig(fname, dpi=FIG_DPI * 0.8)
        plt.close()
        print(f"Saving figure {fname}")

    def plot_modes(self, plot_n_modes: Optional[int] = 10, modes_per_fig: int = 1, show_cylinder: bool = False):
        """Plot the spatial DMD modes (1D/2D, like POD).

        Args:
            plot_n_modes: Number of modes to plot
            modes_per_fig: Number of modes per figure
            show_cylinder: If True, add cylinder mask at origin with radius 0.5
        """
        if self.modes.size == 0:
            print("No modes to plot. Run perform_dmd() first.")
            return
        n_modes = self.modes.shape[1]
        if plot_n_modes is not None:
            n_modes = min(plot_n_modes, n_modes, self.n_modes_save)
        if n_modes == 0:
            print("No modes available to plot.")
            return
        if resolve_volume_layout(self.data, self.modes.shape[0]) is not None:
            self.plot_modes_3d_slices(plot_n_modes=n_modes)
            return
        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        mode_size = self.modes.shape[0]
        physical_nspace = Nx * Ny
        lifted_delays = 1
        is_2d = False
        if Nx > 1 and Ny > 1:
            if mode_size == physical_nspace:
                is_2d = True
            elif physical_nspace > 0 and mode_size % physical_nspace == 0:
                lifted_delays = mode_size // physical_nspace
                is_2d = True
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")
        # Compute mode frequencies (Hz) for annotation purposes

        for start in range(0, n_modes, modes_per_fig):
            end = min(start + modes_per_fig, n_modes)
            ncols = end - start
            if is_2d:
                fig, axes = plt.subplots(
                    1,
                    ncols,
                    figsize=(4 * ncols * fig_aspect, 4),
                    squeeze=False,
                )
            else:
                fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3), squeeze=False)
            axes = axes.ravel()
            for j, i in enumerate(range(start, end)):
                ax = axes[j]
                mode = self.modes[:, i].real
                if is_2d:
                    if lifted_delays > 1:
                        mode_2d = mode.reshape((lifted_delays, Nx, Ny))[0]
                    else:
                        mode_2d = mode.reshape((Nx, Ny))
                    # Get meshgrid for plotting
                    if x_coords.ndim == 1 and y_coords.ndim == 1:
                        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
                    else:
                        x_mesh, y_mesh = x_coords, y_coords
                    # Optionally apply cylinder mask (always mask NaNs)
                    nan_mask = np.isnan(mode_2d)
                    if show_cylinder:
                        distance = np.sqrt(x_mesh**2 + y_mesh**2)
                        cylinder_mask = distance <= 0.5
                        combined_mask = nan_mask | cylinder_mask
                    else:
                        combined_mask = nan_mask
                    mode_plot = np.ma.array(mode_2d, mask=combined_mask)
                    mode_flat = mode_2d[~combined_mask]
                    # Guard against empty array (e.g., all points masked)
                    if mode_flat.size == 0:
                        print(f"  Warning: Mode {i} has no valid data points, skipping plot.")
                        continue
                    # Compute levels with robust limits
                    mode_clean = mode_flat[np.isfinite(mode_flat)]
                    vmin, vmax = np.percentile(mode_clean, [2, 98]) if len(mode_clean) > 0 else (0, 1)
                    levels = np.linspace(vmin, vmax, 21)
                    # Plot filled contour
                    cf = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=CMAP_DIV, extend="both")
                    # Contour lines
                    _ = ax.contour(x_mesh, y_mesh, mode_plot, levels=levels[::4], colors="k", linewidths=0.5, alpha=0.5)
                    # Optionally add cylinder overlay
                    if show_cylinder:
                        cylinder = plt.Circle(
                            (0, 0), 0.5, fill=True, linewidth=0.5, zorder=3, facecolor="lightgray", edgecolor="black"
                        )
                        ax.add_patch(cylinder)
                    style_spatial_axes(ax, self.data, x_coords=x_coords, y_coords=y_coords, equal_default=True)
                    add_inset_colorbar(
                        fig,
                        ax,
                        cf,
                        self.data,
                        ticks=[vmin, 0, vmax],
                        ticklabels=[f"{vmin:.2f}", "0", f"{vmax:.2f}"],
                    )
                else:
                    ax.plot(mode)
                    ax.set_xlabel("Spatial index")
                    ax.set_ylabel("Amplitude")
                delay_suffix = " (delay 0)" if is_2d and lifted_delays > 1 else ""
                ax.set_title(format_mode_title(self.data, i, default=f"DMD Mode {i + 1}{delay_suffix} [{var_name}]"))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
            fname = os.path.join(
                self.figures_dir,
                f"{self.data_root}_dmd_modes_{start + 1}_to_{end}_{var_name}.png",
            )
            fig.savefig(fname, dpi=FIG_DPI)
            plt.close(fig)
            print(f"Saving figure {fname}")

    def plot_modes_3d_slices(self, plot_n_modes: Optional[int] = 4, delay_idx: int = 0) -> None:
        """Plot orthogonal 3D slices for leading DMD/HODMD modes."""
        if self.modes.size == 0:
            print("No modes to plot. Run perform_dmd() first.")
            return
        layout = resolve_volume_layout(self.data, self.modes.shape[0])
        if layout is None:
            print("plot_modes_3d_slices requires volumetric data.")
            return
        _nx, _ny, _nz, lifted_delays = layout
        if delay_idx >= lifted_delays:
            raise ValueError(f"delay_idx={delay_idx} exceeds available lifted delays ({lifted_delays}).")
        n_modes = min(self.modes.shape[1], self.n_modes_save)
        if plot_n_modes is not None:
            n_modes = min(n_modes, plot_n_modes)
        x_coords = self.data.get("x")
        y_coords = self.data.get("y")
        z_coords = self.data.get("z")
        freq = self._mode_freq(self.eigenvalues[:n_modes])
        for mode_idx in range(n_modes):
            mode_3d = reshape_mode_to_volume(self.modes[:, mode_idx].real, self.data, block_index=delay_idx)
            delay_suffix = f" | delay={delay_idx}" if lifted_delays > 1 else ""
            if freq is not None:
                title = f"DMD Mode {mode_idx + 1} | f={freq[mode_idx]:.3g}{delay_suffix}"
            else:
                title = f"DMD Mode {mode_idx + 1}{delay_suffix}"
            output_path = os.path.join(self.figures_dir, f"{self.data_root}_dmd_mode_{mode_idx + 1}_slices.png")
            plot_orthogonal_slices_3d(
                mode_3d,
                x_coords,
                y_coords,
                z_coords,
                output_path=output_path,
                title_prefix=title,
                data=self.data,
                scalar_name="dmd_mode",
            )

    def plot_modes_3d_isometric(self, plot_n_modes: Optional[int] = 4, delay_idx: int = 0) -> None:
        """Plot 3D isosurfaces for leading DMD/HODMD modes."""
        if self.modes.size == 0:
            print("No modes to plot. Run perform_dmd() first.")
            return
        layout = resolve_volume_layout(self.data, self.modes.shape[0])
        if layout is None:
            print("plot_modes_3d_isometric requires volumetric data.")
            return
        _nx, _ny, _nz, lifted_delays = layout
        if delay_idx >= lifted_delays:
            raise ValueError(f"delay_idx={delay_idx} exceeds available lifted delays ({lifted_delays}).")
        n_modes = min(self.modes.shape[1], self.n_modes_save)
        if plot_n_modes is not None:
            n_modes = min(n_modes, plot_n_modes)
        x_coords = self.data.get("x")
        y_coords = self.data.get("y")
        z_coords = self.data.get("z")
        freq = self._mode_freq(self.eigenvalues[:n_modes])
        for mode_idx in range(n_modes):
            mode_3d = reshape_mode_to_volume(self.modes[:, mode_idx].real, self.data, block_index=delay_idx)
            delay_suffix = f" | delay={delay_idx}" if lifted_delays > 1 else ""
            if freq is not None:
                title = f"DMD Mode {mode_idx + 1} | f={freq[mode_idx]:.3g}{delay_suffix}"
            else:
                title = f"DMD Mode {mode_idx + 1}{delay_suffix}"
            output_path = os.path.join(self.figures_dir, f"{self.data_root}_dmd_mode_{mode_idx + 1}_isometric.png")
            plot_isometric_slices_3d(
                mode_3d,
                x_coords,
                y_coords,
                z_coords,
                output_path=output_path,
                title_prefix=title,
                data=self.data,
                scalar_name="dmd_mode",
            )

    def plot_time_coefficients(self, n_coeffs_to_plot=2):
        """Plot DMD temporal coefficients."""
        if self.time_coefficients.size == 0:
            print("No time coefficients to plot. Run perform_dmd() first.")
            return
        n_coeffs_to_plot = min(n_coeffs_to_plot, self.time_coefficients.shape[1], self.n_modes_save)
        if n_coeffs_to_plot == 0:
            print("No coefficients available to plot.")
            return
        Ns_total = self.time_coefficients.shape[0]
        t, xlabel = self._time_axis(Ns_total)
        fig = plt.figure(figsize=(10, 3 * n_coeffs_to_plot))
        for i in range(n_coeffs_to_plot):
            plt.subplot(n_coeffs_to_plot, 1, i + 1)
            coeff = np.asarray(self.time_coefficients[:Ns_total, i].real, dtype=float)
            finite = np.isfinite(coeff)
            if not np.any(finite):
                plt.text(0.5, 0.5, "No finite coefficients", ha="center", va="center", transform=plt.gca().transAxes)
                plt.axis("off")
                continue
            t_plot = t[finite]
            coeff_plot = coeff[finite]
            amp_scale = float(np.max(np.abs(coeff_plot)))
            ylabel = f"Amplitude Mode {i + 1}"
            if np.isfinite(amp_scale) and amp_scale > 1e50:
                coeff_plot = coeff_plot / amp_scale
                ylabel += " (normalized)"
            plt.plot(t_plot, coeff_plot, linewidth=1.0)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"Temporal Coefficient for DMD Mode {i + 1}")
            plt.grid(True, linestyle=":")
            plt.xlim(t_plot.min(), t_plot.max())
            y_min = float(np.min(coeff_plot))
            y_max = float(np.max(coeff_plot))
            if np.isclose(y_min, y_max):
                margin = 1.0 if np.isclose(y_min, 0.0) else 0.05 * max(abs(y_min), 1.0)
                plt.ylim(y_min - margin, y_max + margin)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_dmd_time_coeffs.png")
        fig.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {plot_filename}")

    def plot_reconstruction_error(self):
        """Plot the data reconstruction error using an increasing number of DMD modes."""
        if self.modes.size == 0 or self.time_coefficients.size == 0 or "q" not in self.data:
            print("Data, modes, or time coefficients not available. Run perform_dmd() first.")
            return
        data_matrix = self.data["q"]
        # Guard: delay-embedded modes live in a lifted space incompatible with q
        if self.modes.shape[0] != data_matrix.shape[1]:
            print(
                "Reconstruction error plot is not available for delay-embedded DMD "
                f"(mode dimension {self.modes.shape[0]} != spatial dimension {data_matrix.shape[1]})."
            )
            return
        # DMD reconstruction: sum_k a_k(t) * phi_k
        n_modes_check = self.modes.shape[1]
        reconstruction_errors = []
        for k in range(1, n_modes_check + 1):
            reconstructed_data_k_modes = self.time_coefficients[:, :k] @ self.modes[:, :k].T
            error = np.linalg.norm(data_matrix - reconstructed_data_k_modes, "fro") / np.linalg.norm(data_matrix, "fro")
            reconstruction_errors.append(error * 100)
        mode_indices = np.arange(1, n_modes_check + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, reconstruction_errors, "s-", linewidth=2, markersize=6)
        plt.xlabel("Number of Modes Used for Reconstruction")
        plt.ylabel("Reconstruction Error (%)")
        plt.title("Data Reconstruction Error vs. Number of DMD Modes")
        plt.grid(True, which="both", ls="--")
        plt.yscale("log")
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_dmd_reconstruction_error.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {plot_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMD analysis")
    parser.add_argument("--config", help="Path to JSON/YAML configuration file", default=None)
    parser.add_argument("--data", help="Path to input data file", default=None)
    parser.add_argument("--prep", action="store_true", help="Load data and prepare for DMD")
    parser.add_argument("--compute", action="store_true", help="Perform DMD and save results")
    parser.add_argument("--plot", action="store_true", help="Generate default plots")
    args = parser.parse_args()

    if args.config:
        from openmodalpy.core.config import load_config

        load_config(args.config)

    try:
        data_file = require_existing_data_path(args.data)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    n_modes_to_save_main = 8
    n_modes_to_plot_spatial_main = 8
    n_coeffs_to_plot_time_main = 8

    # Support batch field analysis for npz files
    if DNamiDataLoader is not None and data_file.endswith(".npz"):
        loader = DNamiDataLoader()
        available_fields = loader.get_available_fields(data_file)
        print(f"Available fields in {data_file}: {available_fields}")
        for field in available_fields:
            print(f"\n===== Running DMD for variable: {field} =====")
            results_dir = os.path.join(RESULTS_DIR_DMD, field)
            figures_dir = os.path.join(FIGURES_DIR_DMD, field)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)
            analyzer = DMDAnalyzer(
                file_path=data_file,
                results_dir=results_dir,
                figures_dir=figures_dir,
                data_loader=lambda fp: loader.load(fp, field=field),
                n_modes_save=n_modes_to_save_main,
                spatial_weight_type="uniform",
            )
            analyzer.analysis_type = f"dmd_{field}"

            if args.compute or args.prep:
                data = loader.load(data_file, field=field)
                analyzer.data = data
                if args.compute:
                    analyzer.perform_dmd()
                    analyzer.save_results()
                    analyzer.plot_eigenspectra()
                    analyzer.plot_modes_detailed(plot_n_modes=n_modes_to_plot_spatial_main)
                    analyzer.plot_time_coefficients(n_coeffs_to_plot=n_coeffs_to_plot_time_main)
                    analyzer.plot_cumulative_energy()
                    analyzer.plot_reconstruction_error()
                elif args.prep:
                    analyzer.load_and_preprocess()
                    # Optionally save preprocessed data if needed
            elif args.plot:
                analyzer.load_results()
                analyzer.plot_eigenspectra()
                analyzer.plot_modes_detailed(plot_n_modes=n_modes_to_plot_spatial_main)
                analyzer.plot_time_coefficients(n_coeffs_to_plot=n_coeffs_to_plot_time_main)
                analyzer.plot_cumulative_energy()
                analyzer.plot_reconstruction_error()
            print_summary("DMD", analyzer.results_dir, analyzer.figures_dir)
    else:
        # Fallback for legacy .mat/.h5 files
        from openmodalpy.core.base import load_mat_data

        loader = load_mat_data
        analyzer = DMDAnalyzer(
            file_path=data_file,
            results_dir=RESULTS_DIR_DMD,
            figures_dir=FIGURES_DIR_DMD,
            data_loader=loader,
            spatial_weight_type="uniform",
            n_modes_save=n_modes_to_save_main,
        )
        run_all = not (args.prep or args.compute or args.plot)
        if run_all or args.prep:
            analyzer.load_and_preprocess()
        if run_all or args.compute:
            if analyzer.data == {}:
                analyzer.load_and_preprocess()
            analyzer.perform_dmd()
            analyzer.save_results()
        if run_all or args.plot:
            if analyzer.eigenvalues.size == 0:
                print("No DMD results to plot. Run with --compute first.")
            else:
                analyzer.plot_eigenspectra()
                analyzer.plot_modes_detailed(plot_n_modes=n_modes_to_plot_spatial_main)
                analyzer.plot_time_coefficients(n_coeffs_to_plot=n_coeffs_to_plot_time_main)
                analyzer.plot_cumulative_energy()
                analyzer.plot_reconstruction_error()
        if run_all:
            print_summary("DMD", analyzer.results_dir, analyzer.figures_dir)
