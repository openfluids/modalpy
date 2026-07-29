#!/usr/bin/env python3
"""
Spatio-Temporal Proper Orthogonal Decomposition (ST-POD)

The current analyzer is delay-embedded POD: it constructs a block Hankel matrix
from time-delayed snapshots and performs a weighted SVD in that lifted space.
It should not be read as a direct discretization of the full two-time
space-time covariance operator.

Mathematical basis:
    H = [q₁    q₂    ...  qₘ   ]     d = embedding dimension
        [q₂    q₃    ...  qₘ₊₁ ]     m = Ns - d + 1 (columns)
        [⋮     ⋮     ⋱    ⋮    ]
        [qₐ   qₐ₊₁   ...  qₘ₊ₐ₋₁]

    SVD(H) = U Σ Vᴴ  →  U columns = space-time modes (d stacked spatial fields)

Author: R. Frantz

Reference:
    - Sieber, M., Paschereit, C. O., & Oberleithner, K. (2016).
      "Spectral proper orthogonal decomposition." JFM, 792, 798-828.
"""

import argparse
import os
import time
from typing import Optional

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fftkit import find_peaks, periodogram_rfft

import openmodalpy.core.decomposition as decomposition
from openmodalpy.core.base import (
    BaseAnalyzer,
    get_fig_aspect_ratio,
    plot_isometric_slices_3d,
    plot_orthogonal_slices_3d,
    print_summary,
    reshape_mode_to_volume,
    resolve_volume_layout,
)
from openmodalpy.core.config import (
    CMAP_DIV,
    FIG_DPI,
    FIGURES_DIR_STPOD,
    RESULTS_DIR_STPOD,
    require_existing_data_path,
)


class STPODAnalyzer(BaseAnalyzer):
    """Spatio-Temporal POD analyzer using time-delay embedding.

    ST-POD constructs a block Hankel matrix from the data snapshots and performs
    SVD to extract space-time modes that capture both spatial structure and
    temporal evolution.

    Key Attributes:
        embedding_dim (int): Number of time delays (d). The Hankel matrix has
            d*Nspace rows.
        modes (np.ndarray): Space-time modes. Shape: (d * Nspace, n_modes_save).
            Each mode consists of d stacked spatial fields.
        eigenvalues (np.ndarray): Squared singular values from the weighted
            Hankel matrix divided by the number of Hankel columns. Shape:
            (n_modes_save,).
        time_coefficients (np.ndarray): Temporal coefficients from Vᵀ.
            Shape: (m, n_modes_save) where m = Ns - d + 1.
        temporal_mean (np.ndarray): Mean snapshot. Shape: (Nspace,).

    Example:
        >>> analyzer = STPODAnalyzer("data.npz", embedding_dim=20, n_modes_save=10)
        >>> analyzer.run_analysis()
    """

    def __init__(
        self,
        file_path: str,
        embedding_dim: int = 10,
        n_modes_save: int = 10,
        results_dir: str = RESULTS_DIR_STPOD,
        figures_dir: str = FIGURES_DIR_STPOD,
        data_loader=None,
        spatial_weight_type: str = "auto",
        use_parallel: bool = True,
    ):
        """Initialize the STPODAnalyzer.

        Args:
            file_path: Path to the data file.
            embedding_dim: Time delay embedding dimension (d). Must be >= 2.
            n_modes_save: Number of modes to compute and save.
            results_dir: Directory to save results.
            figures_dir: Directory to save figures.
            data_loader: Custom function to load data.
            spatial_weight_type: Type of spatial weights ('auto', 'uniform', 'polar').
            use_parallel: Whether to use parallel computation where available.
        """
        super().__init__(
            file_path=file_path,
            nfft=1,  # Not used by ST-POD
            overlap=0,  # Not used by ST-POD
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            use_parallel=use_parallel,
        )

        self.embedding_dim = embedding_dim
        self.n_modes_save = n_modes_save
        self.modes = np.array([])
        self.eigenvalues = np.array([])
        self.time_coefficients = np.array([])
        self.temporal_mean = np.array([])

        self.analysis_type = "stpod"

    def _build_hankel_matrix(self, data_centered: np.ndarray) -> np.ndarray:
        """Build the block Hankel matrix from centered data.

        Returns shape ``(d * Nspace, m)`` with ``m = Ns - d + 1`` — the
        historical layout used by reconstruction checks. The delay lift
        itself produces samples × features; this method transposes for the
        column-oriented Hankel contract.
        """
        lift = decomposition.DelayEmbeddingLift(self.embedding_dim)
        return np.ascontiguousarray(lift.apply(data_centered).T)

    def _get_weight_vector(self, num_space_points: int) -> np.ndarray:
        """Extract weight vector from self.W, handling various shapes."""
        if self.spatial_weight_type == "uniform":
            return np.ones(num_space_points, dtype=np.float64)

        if self.W.ndim == 2:
            if self.W.shape[0] == self.W.shape[1]:
                return np.diag(self.W)
            elif self.W.shape[1] == 1:
                return self.W.ravel()
            else:
                raise ValueError(f"Unexpected weight shape: {self.W.shape}")
        return self.W

    def perform_stpod(self) -> None:
        """Perform ST-POD analysis on the loaded data.

        The algorithm:
        1. Validate embedding dimension.
        2. Subtract temporal mean.
        3. Apply the delay lift → (m, d*Nspace), m = Ns - d + 1.
        4. Tile the spatial metric over the d delay blocks (I_d ⊗ W).
        5. Solve via ``weighted_second_order(..., method="svd")``, which
           weights, takes the SVD, unweights the modes, and returns
           eigenvalues = sigma²[:k] / m with coefficients Vt scaled by sigma.

        The SVD route is deliberate: forming the Gram matrix would square the
        condition number of an already ill-conditioned Hankel matrix.
        """
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        data_matrix = self.data["q"]  # Shape (Ns, Nspace)
        Ns, Nspace = data_matrix.shape

        # Validate parameters
        if self.embedding_dim < 2:
            raise ValueError(f"embedding_dim must be >= 2, got {self.embedding_dim}")
        if self.embedding_dim >= Ns:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be < number of snapshots ({Ns})")

        m = Ns - self.embedding_dim + 1  # Number of Hankel columns
        print(
            f"Performing ST-POD: d={self.embedding_dim}, m={m} columns, "
            f"Hankel shape=({self.embedding_dim * Nspace}, {m})"
        )
        start_time = time.time()

        # 1. Subtract temporal mean
        self.temporal_mean = np.mean(data_matrix, axis=0, dtype=np.float64)
        data_centered = data_matrix - self.temporal_mean

        # 2. Delay lift → samples × lifted features (m, d*Nspace)
        self._lift = decomposition.DelayEmbeddingLift(self.embedding_dim)
        lifted = self._lift.apply(data_centered)

        # 3. Metric in the lifted space: I_d ⊗ W
        weight_vector = self._get_weight_vector(Nspace)
        base_metric = decomposition.SpatialMetric(weight_vector)
        lifted_metric = decomposition.SpatialMetric(base_metric.tile(self.embedding_dim))

        # 4. Weighted SVD in the lifted space (do not square the Hankel matrix)
        n_min = min(lifted.shape)
        k = min(self.n_modes_save, n_min - 1)
        if k < self.n_modes_save:
            print(f"Warning: Only {k} modes available, requested {self.n_modes_save}")
            self.n_modes_save = k

        self.modes, self.eigenvalues, self.time_coefficients = decomposition.weighted_second_order(
            lifted,
            lifted_metric,
            method="svd",
            n_keep=k,
        )

        end_time = time.time()
        print(f"ST-POD completed in {end_time - start_time:.2f} seconds.")
        print(f"Computed {self.n_modes_save} ST-POD modes.")

    def _get_algorithm_metadata(self) -> dict:
        """Describe the current delay-embedded POD contract."""
        lift = getattr(self, "_lift", None) or decomposition.DelayEmbeddingLift(max(self.embedding_dim, 2))
        return {
            "lift_kind": lift.kind,
            "stpod_variant": "delay_embedded_pod",
            "uses_mean_subtraction": True,
            "uses_spatial_metric_in_lifted_space": True,
            "eigenvalue_normalization": "sigma_squared_over_n_hankel_cols",
            "is_full_spacetime_pod": False,
        }

    def extract_spatial_mode(self, mode_idx: int, delay_idx: int = 0) -> np.ndarray:
        """Extract a single spatial field from a space-time mode.

        Args:
            mode_idx: Mode index (0-based).
            delay_idx: Which delay to extract (0 to d-1).

        Returns:
            Spatial mode field, shape (Nspace,).
        """
        if self.modes.size == 0:
            raise ValueError("No modes available. Run perform_stpod() first.")
        if delay_idx < 0 or delay_idx >= self.embedding_dim:
            raise ValueError(f"delay_idx must be in [0, {self.embedding_dim - 1}]")

        Nspace = self.modes.shape[0] // self.embedding_dim
        start = delay_idx * Nspace
        end = (delay_idx + 1) * Nspace
        return self.modes[start:end, mode_idx]

    def get_mode_as_movie(self, mode_idx: int) -> np.ndarray:
        """Get a mode as a sequence of spatial fields (for animation).

        Args:
            mode_idx: Mode index (0-based).

        Returns:
            Array of shape (d, Nspace) representing temporal evolution.
        """
        if self.modes.size == 0:
            raise ValueError("No modes available. Run perform_stpod() first.")

        Nspace = self.modes.shape[0] // self.embedding_dim
        mode_frames = np.zeros((self.embedding_dim, Nspace))
        for k in range(self.embedding_dim):
            mode_frames[k, :] = self.extract_spatial_mode(mode_idx, k)
        return mode_frames

    def save_results(self, filename: Optional[str] = None) -> None:
        """Save ST-POD results to HDF5 file."""
        if not filename:
            filename = (
                f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_d{self.embedding_dim}_{self.analysis_type}.hdf5"
            )

        save_path = os.path.join(self.results_dir, filename)
        print(f"Saving ST-POD results to {save_path}")

        with h5py.File(save_path, "w") as f:
            f.attrs.update(self._get_metadata())
            f.attrs["embedding_dim"] = self.embedding_dim
            f.attrs["n_modes_saved"] = self.n_modes_save
            f.attrs["n_snapshots"] = self.data.get("Ns", 0)
            f.attrs["Nspace"] = self.modes.shape[0] // self.embedding_dim

            if "x" in self.data:
                f.create_dataset("x", data=self.data["x"], compression="gzip")
            if "y" in self.data:
                f.create_dataset("y", data=self.data["y"], compression="gzip")
            if "z" in self.data and self.data["z"] is not None:
                f.create_dataset("z", data=self.data["z"], compression="gzip")
            if self.W.size > 0:
                f.create_dataset("W", data=self.W, compression="gzip")
            if self.temporal_mean.size > 0:
                f.create_dataset("temporal_mean", data=self.temporal_mean, compression="gzip")

            f.create_dataset("modes", data=self.modes, compression="gzip")
            f.create_dataset("eigenvalues", data=self.eigenvalues, compression="gzip")
            f.create_dataset("time_coefficients", data=self.time_coefficients, compression="gzip")

        print("ST-POD results saved.")

    def load_results(self, filename: Optional[str] = None) -> None:
        """Load ST-POD results from HDF5 file."""
        if not filename:
            filename = (
                f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_d{self.embedding_dim}_{self.analysis_type}.hdf5"
            )

        load_path = os.path.join(self.results_dir, filename)
        print(f"Loading ST-POD results from {load_path}")

        if not os.path.isfile(load_path):
            import glob

            pattern = os.path.join(self.results_dir, f"*_{self.analysis_type}.hdf5")
            matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
            if matches:
                load_path = matches[0]
                print(f"[Auto-detect] Using: {load_path}")
            else:
                print(f"[ERROR] No results file found in {self.results_dir}")
                return

        with h5py.File(load_path, "r") as f:
            if "x" in f:
                self.data["x"] = f["x"][:]
            if "y" in f:
                self.data["y"] = f["y"][:]
            if "z" in f:
                self.data["z"] = f["z"][:]
            if "W" in f:
                self.W = f["W"][:]
            if "temporal_mean" in f:
                self.temporal_mean = f["temporal_mean"][:]

            self.modes = f["modes"][:]
            self.eigenvalues = f["eigenvalues"][:]
            self.time_coefficients = f["time_coefficients"][:]

            if "embedding_dim" in f.attrs:
                self.embedding_dim = f.attrs["embedding_dim"]
            if "dt" in f.attrs:
                self.data["dt"] = f.attrs["dt"]
            if "n_snapshots" in f.attrs:
                self.data["Ns"] = f.attrs["n_snapshots"]
            if "Nspace" in f.attrs:
                self.data["Nspace"] = f.attrs["Nspace"]
            if "Nx" in f.attrs:
                self.data["Nx"] = int(f.attrs["Nx"])
            if "Ny" in f.attrs:
                self.data["Ny"] = int(f.attrs["Ny"])
            if "Nz" in f.attrs:
                self.data["Nz"] = int(f.attrs["Nz"])

        print("ST-POD results loaded.")

    def plot_eigenvalues(self) -> None:
        """Plot the ST-POD eigenvalue spectrum."""
        if self.eigenvalues.size == 0:
            print("No eigenvalues to plot. Run perform_stpod() first.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            mode_indices = np.arange(1, len(self.eigenvalues) + 1)
            normalized = self.eigenvalues / np.sum(self.eigenvalues) * 100

            ax.plot(mode_indices, normalized, "o-", linewidth=2, markersize=6)

            n_annotate = min(5, len(mode_indices))
            for idx in range(n_annotate):
                ax.text(mode_indices[idx], normalized[idx], f" {idx + 1}", fontsize=7, va="bottom")

            ax.set_yscale("log")
            ax.set_xlabel("Mode Number")
            ax.set_ylabel("Normalized Eigenvalue (%)")
            ax.set_title(f"ST-POD Eigenvalue Spectrum (d={self.embedding_dim})")
            ax.grid(True, which="both", ls="--")

            plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_eigenvalues.png")
            plt.savefig(plot_filename, dpi=FIG_DPI, bbox_inches="tight")
            print(f"Saving figure {plot_filename}")
        finally:
            plt.close(fig)

    def plot_modes(
        self,
        plot_n_modes: int = 4,
        delay_idx: int = 0,
        show_cylinder: bool = False,
    ) -> None:
        """Plot spatial modes at a specific delay index.

        Args:
            plot_n_modes: Number of modes to plot.
            delay_idx: Which time delay to show (0 to d-1).
            show_cylinder: If True, mask cylinder at origin.
        """
        if self.modes.size == 0:
            print("No modes to plot. Run perform_stpod() first.")
            return

        Nspace = self.modes.shape[0] // self.embedding_dim
        if resolve_volume_layout(self.data, Nspace) is not None:
            self.plot_modes_3d_slices(plot_n_modes=plot_n_modes, delay_idx=delay_idx)
            return
        Nx = self.data.get("Nx", int(np.sqrt(Nspace)))
        Ny = self.data.get("Ny", int(np.sqrt(Nspace)))
        is_2d = (Nspace == Nx * Ny) and (Nx > 1 and Ny > 1)

        if not is_2d:
            print("plot_modes currently supports 2-D fields only.")
            return

        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)

        n_modes = min(plot_n_modes, self.n_modes_save)
        ncols = min(n_modes, 2)
        nrows = int(np.ceil(n_modes / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols * fig_aspect, 4 * nrows),
            squeeze=False,
            constrained_layout=True,
        )

        if x_coords.ndim == 1 and y_coords.ndim == 1:
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
        else:
            x_mesh, y_mesh = x_coords, y_coords

        total_energy = np.sum(self.eigenvalues)

        for k in range(n_modes):
            row, col = divmod(k, ncols)
            ax = axes[row, col]

            mode_spatial = self.extract_spatial_mode(k, delay_idx)
            mode_2d = mode_spatial.reshape((Nx, Ny))

            if show_cylinder:
                dist = np.sqrt(x_mesh**2 + y_mesh**2)
                mask = dist <= 0.5
                mode_plot = np.ma.array(mode_2d, mask=mask)
            else:
                mode_plot = mode_2d

            from openmodalpy.core.base import get_robust_clim

            vmin, vmax = get_robust_clim(mode_plot, method="percentile")
            levels = np.linspace(vmin, vmax, 21)

            cf = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=CMAP_DIV, extend="both")

            if show_cylinder:
                cyl = plt.Circle((0, 0), 0.5, fill=True, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                ax.add_patch(cyl)

            ax.set_aspect("equal", "box")
            ax.set_xlim(np.min(x_coords), np.max(x_coords))
            ax.set_ylim(np.min(y_coords), np.max(y_coords))
            ax.set_xlabel(r"$x/D$")
            ax.set_ylabel(r"$y/D$")
            ax.grid(True, linestyle="--", alpha=0.3)

            energy_pct = 100.0 * self.eigenvalues[k] / total_energy
            cum_pct = 100.0 * np.sum(self.eigenvalues[: k + 1]) / total_energy
            ax.set_title(f"Mode {k + 1} (τ={delay_idx})\nE={energy_pct:.2f}% Cum={cum_pct:.2f}%", fontsize=9)

            fig.colorbar(cf, ax=ax, shrink=0.8)

        # Hide empty subplots
        for idx in range(n_modes, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].axis("off")

        fig.suptitle(f"ST-POD Modes (d={self.embedding_dim}, delay={delay_idx})", fontsize=12)
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_modes_delay{delay_idx}.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {plot_filename}")

    def plot_modes_3d_slices(self, plot_n_modes: int = 4, delay_idx: int = 0) -> None:
        """Plot orthogonal 3D slices for ST-POD modes at one delay index."""
        if self.modes.size == 0:
            print("No modes to plot. Run perform_stpod() first.")
            return
        nspace = self.modes.shape[0] // self.embedding_dim
        layout = resolve_volume_layout(self.data, nspace)
        if layout is None:
            print("plot_modes_3d_slices requires volumetric data.")
            return
        if not 0 <= delay_idx < self.embedding_dim:
            raise ValueError(f"delay_idx={delay_idx} outside [0, {self.embedding_dim - 1}]")
        x_coords = self.data.get("x")
        y_coords = self.data.get("y")
        z_coords = self.data.get("z")
        n_modes = min(plot_n_modes, self.n_modes_save)
        total_energy = np.sum(self.eigenvalues) if self.eigenvalues.size else None
        for mode_idx in range(n_modes):
            mode_3d = reshape_mode_to_volume(self.extract_spatial_mode(mode_idx, delay_idx), self.data)
            if total_energy:
                energy_pct = 100.0 * self.eigenvalues[mode_idx] / total_energy
                title = f"ST-POD Mode {mode_idx + 1} | delay={delay_idx} | E={energy_pct:.2f}%"
            else:
                title = f"ST-POD Mode {mode_idx + 1} | delay={delay_idx}"
            output_path = os.path.join(
                self.figures_dir, f"{self.data_root}_stpod_mode_{mode_idx + 1}_delay{delay_idx}_slices.png"
            )
            plot_orthogonal_slices_3d(
                mode_3d,
                x_coords,
                y_coords,
                z_coords,
                output_path=output_path,
                title_prefix=title,
                data=self.data,
                scalar_name="stpod_mode",
            )

    def plot_modes_3d_isometric(self, plot_n_modes: int = 4, delay_idx: int = 0) -> None:
        """Plot 3D isosurfaces for ST-POD modes at one delay index."""
        if self.modes.size == 0:
            print("No modes to plot. Run perform_stpod() first.")
            return
        nspace = self.modes.shape[0] // self.embedding_dim
        layout = resolve_volume_layout(self.data, nspace)
        if layout is None:
            print("plot_modes_3d_isometric requires volumetric data.")
            return
        if not 0 <= delay_idx < self.embedding_dim:
            raise ValueError(f"delay_idx={delay_idx} outside [0, {self.embedding_dim - 1}]")
        x_coords = self.data.get("x")
        y_coords = self.data.get("y")
        z_coords = self.data.get("z")
        n_modes = min(plot_n_modes, self.n_modes_save)
        total_energy = np.sum(self.eigenvalues) if self.eigenvalues.size else None
        for mode_idx in range(n_modes):
            mode_3d = reshape_mode_to_volume(self.extract_spatial_mode(mode_idx, delay_idx), self.data)
            if total_energy:
                energy_pct = 100.0 * self.eigenvalues[mode_idx] / total_energy
                title = f"ST-POD Mode {mode_idx + 1} | delay={delay_idx} | E={energy_pct:.2f}%"
            else:
                title = f"ST-POD Mode {mode_idx + 1} | delay={delay_idx}"
            output_path = os.path.join(
                self.figures_dir, f"{self.data_root}_stpod_mode_{mode_idx + 1}_delay{delay_idx}_isometric.png"
            )
            plot_isometric_slices_3d(
                mode_3d,
                x_coords,
                y_coords,
                z_coords,
                output_path=output_path,
                title_prefix=title,
                data=self.data,
                scalar_name="stpod_mode",
            )

    def plot_spacetime_mode(
        self,
        mode_idx: int = 0,
        n_delays_show: Optional[int] = None,
        show_cylinder: bool = False,
    ) -> None:
        """Plot a space-time mode showing its temporal evolution.

        Args:
            mode_idx: Which mode to visualize.
            n_delays_show: Number of delay frames to show. Default: min(d, 6).
            show_cylinder: If True, mask cylinder at origin.
        """
        if self.modes.size == 0:
            print("No modes to plot. Run perform_stpod() first.")
            return

        if n_delays_show is None:
            n_delays_show = min(self.embedding_dim, 6)

        Nspace = self.modes.shape[0] // self.embedding_dim
        Nx = self.data.get("Nx", int(np.sqrt(Nspace)))
        Ny = self.data.get("Ny", int(np.sqrt(Nspace)))
        is_2d = (Nspace == Nx * Ny) and (Nx > 1 and Ny > 1)

        if not is_2d:
            print("plot_spacetime_mode currently supports 2-D fields only.")
            return

        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)

        # Select delays to show (evenly spaced)
        if n_delays_show < self.embedding_dim:
            delay_indices = np.linspace(0, self.embedding_dim - 1, n_delays_show, dtype=int)
        else:
            delay_indices = np.arange(self.embedding_dim)

        ncols = min(len(delay_indices), 3)
        nrows = int(np.ceil(len(delay_indices) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols * fig_aspect, 4 * nrows),
            squeeze=False,
            constrained_layout=True,
        )

        if x_coords.ndim == 1 and y_coords.ndim == 1:
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
        else:
            x_mesh, y_mesh = x_coords, y_coords

        # Get global vmax for consistent colorscale
        mode_frames = self.get_mode_as_movie(mode_idx)
        global_vmax = np.max(np.abs(mode_frames))
        levels = np.linspace(-global_vmax, global_vmax, 21)

        for i, delay_idx in enumerate(delay_indices):
            row, col = divmod(i, ncols)
            ax = axes[row, col]

            mode_spatial = self.extract_spatial_mode(mode_idx, delay_idx)
            mode_2d = mode_spatial.reshape((Nx, Ny))

            if show_cylinder:
                dist = np.sqrt(x_mesh**2 + y_mesh**2)
                mask = dist <= 0.5
                mode_plot = np.ma.array(mode_2d, mask=mask)
            else:
                mode_plot = mode_2d

            _ = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=CMAP_DIV, extend="both")

            if show_cylinder:
                cyl = plt.Circle((0, 0), 0.5, fill=True, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                ax.add_patch(cyl)

            ax.set_aspect("equal", "box")
            ax.set_xlim(np.min(x_coords), np.max(x_coords))
            ax.set_ylim(np.min(y_coords), np.max(y_coords))
            ax.set_xlabel(r"$x/D$")
            ax.set_ylabel(r"$y/D$")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_title(f"τ = {delay_idx}", fontsize=9)

        # Hide empty subplots
        for idx in range(len(delay_indices), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].axis("off")

        energy_pct = 100.0 * self.eigenvalues[mode_idx] / np.sum(self.eigenvalues)
        fig.suptitle(f"ST-POD Mode {mode_idx + 1} Evolution (E={energy_pct:.2f}%)", fontsize=12)

        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_spacetime_mode{mode_idx + 1}.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {plot_filename}")

    def plot_time_coefficients(
        self,
        n_coeffs: int = 2,
        n_snapshots_plot: Optional[int] = None,
        L: float = 1.0,
        U: float = 1.0,
    ) -> None:
        """Plot temporal coefficients and their spectra.

        Args:
            n_coeffs: Number of coefficients to plot.
            n_snapshots_plot: Number of time points to show.
            L: Characteristic length for Strouhal number.
            U: Characteristic velocity for Strouhal number.
        """
        if self.time_coefficients.size == 0:
            print("No time coefficients to plot. Run perform_stpod() first.")
            return

        n_coeffs = min(n_coeffs, self.time_coefficients.shape[1])
        m = self.time_coefficients.shape[0]  # Number of Hankel columns

        if n_snapshots_plot is None or n_snapshots_plot > m:
            n_snapshots_plot = m

        time_vector, xlabel = self._time_axis(n_snapshots_plot)
        fs = self._require_fs()

        fig, axes = plt.subplots(n_coeffs, 2, figsize=(12, 3 * n_coeffs))
        if n_coeffs == 1:
            axes = axes.reshape(1, 2)

        for i in range(n_coeffs):
            coeff = self.time_coefficients[:n_snapshots_plot, i]

            # Time series
            axes[i, 0].plot(time_vector, coeff, ls="-", lw=0.8, marker="o", markersize=1)
            axes[i, 0].set_xlabel(xlabel)
            axes[i, 0].set_ylabel(f"a_{i + 1}(t)")
            axes[i, 0].set_title(f"ST-POD Coefficient {i + 1}")
            axes[i, 0].grid(True, linestyle=":")
            axes[i, 0].set_xlim(time_vector.min(), time_vector.max())

            # Periodogram
            freqs, psd = periodogram_rfft(coeff, fs)
            peak_freqs, peak_psd = find_peaks(freqs, psd)

            if L is not None and U is not None:
                freqs_st = freqs * L / U
                peak_freqs_st = peak_freqs * L / U if peak_freqs.size > 0 else peak_freqs
            else:
                freqs_st = freqs
                peak_freqs_st = peak_freqs

            axes[i, 1].semilogy(freqs_st, psd)
            if peak_freqs_st.size > 0:
                axes[i, 1].plot(peak_freqs_st, peak_psd, "o", markersize=4)
                for pf, pv in zip(peak_freqs_st[:3], peak_psd[:3]):
                    axes[i, 1].text(pf, pv, f" {pf:.2f}", fontsize=8, ha="left", va="bottom")

            axes[i, 1].set_xscale("log")
            # Fit y-limits: show 6 decades below peak
            psd_pos = psd[psd > 0]
            if len(psd_pos) > 0:
                ymax = psd_pos.max() * 3
                axes[i, 1].set_ylim(ymax * 1e-6, ymax)
            axes[i, 1].set_xlabel("Strouhal Number (St)")
            axes[i, 1].set_ylabel("PSD")
            axes[i, 1].set_title(f"Periodogram Mode {i + 1}")
            axes[i, 1].grid(True, linestyle=":")

        plt.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_time_coeffs.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {plot_filename}")

    def plot_cumulative_energy(self) -> None:
        """Plot cumulative energy captured by ST-POD modes."""
        if self.eigenvalues.size == 0:
            print("No eigenvalues to plot. Run perform_stpod() first.")
            return

        cumulative = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues) * 100
        mode_indices = np.arange(1, len(self.eigenvalues) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(mode_indices, cumulative, "o-", linewidth=2, markersize=6)

        for idx, (x, y) in enumerate(zip(mode_indices, cumulative)):
            ax.text(x, y, f" {idx + 1}", fontsize=7, va="bottom")

        ax.set_xlabel("Number of Modes")
        ax.set_ylabel("Cumulative Energy (%)")
        ax.set_title(f"Cumulative Energy of ST-POD Modes (d={self.embedding_dim})")
        ax.grid(True, which="both", ls="--")
        ax.set_ylim(0, 105)

        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_stpod_cumulative_energy.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {plot_filename}")

    def check_mode_orthogonality(self, tolerance: float = 1e-9) -> bool:
        """Check orthonormality of modes with respect to extended weights.

        The weighted inner product uses W extended to d copies for the
        d*Nspace-dimensional mode vectors.

        Args:
            tolerance: Maximum allowed deviation from identity.

        Returns:
            True if modes are orthonormal within tolerance.
        """
        if self.modes.size == 0 or self.W.size == 0:
            print("Modes or weights not available.")
            return False

        print("\nChecking ST-POD mode orthonormality...")
        Nspace = self.modes.shape[0] // self.embedding_dim
        n_modes = self.modes.shape[1]

        weight_vector = self._get_weight_vector(Nspace)
        W_extended = np.tile(weight_vector, self.embedding_dim)

        # Element-wise multiply instead of forming dense (d*Nspace)² diagonal matrix
        gram = self.modes.T @ (W_extended[:, np.newaxis] * self.modes)

        diag_dev = np.max(np.abs(np.diag(gram) - 1.0))
        off_diag_mask = ~np.eye(n_modes, dtype=bool)
        off_diag_max = np.max(np.abs(gram[off_diag_mask])) if n_modes > 1 else 0.0

        is_orthonormal = (diag_dev < tolerance) and (off_diag_max < tolerance)

        print(f"  Max diagonal deviation from 1: {diag_dev:.2e}")
        print(f"  Max off-diagonal value: {off_diag_max:.2e}")
        print(f"  Orthonormal: {'Yes' if is_orthonormal else 'No'}")

        return is_orthonormal

    def run_analysis(
        self,
        plot_n_modes: int = 4,
        plot_n_coeffs: int = 4,
        check_orthogonality: bool = False,
    ) -> None:
        """Main entry point for ST-POD analysis.

        Args:
            plot_n_modes: Number of modes to plot.
            plot_n_coeffs: Number of time coefficients to plot.
            check_orthogonality: Whether to verify mode orthonormality.
        """
        print(f"Starting ST-POD analysis for {os.path.basename(self.file_path)}")
        start_time = time.time()

        # Load data
        super().run(compute_fft=False)

        # Perform ST-POD
        self.perform_stpod()

        # Save results
        self.save_results()

        # Plotting
        self.plot_eigenvalues()
        if int(self.data.get("Nz", 1)) > 1:
            self.plot_modes_3d_slices(plot_n_modes=plot_n_modes, delay_idx=0)
            self.plot_modes_3d_isometric(plot_n_modes=plot_n_modes, delay_idx=0)
        else:
            self.plot_modes(plot_n_modes=plot_n_modes, delay_idx=0)
            self.plot_spacetime_mode(mode_idx=0)
            if self.n_modes_save > 1:
                self.plot_spacetime_mode(mode_idx=1)
        self.plot_time_coefficients(n_coeffs=plot_n_coeffs)
        self.plot_cumulative_energy()

        if check_orthogonality:
            self.check_mode_orthogonality()

        end_time = time.time()
        print(f"\nST-POD analysis completed in {end_time - start_time:.2f} seconds.")
        print_summary("ST-POD", self.results_dir, self.figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ST-POD analysis")
    parser.add_argument("--data", type=str, default=None, help="Path to data file")
    parser.add_argument("--embedding-dim", type=int, default=10, help="Time delay embedding dimension")
    parser.add_argument("--n-modes", type=int, default=10, help="Number of modes to save")
    parser.add_argument("--compute", action="store_true", help="Compute analysis")
    parser.add_argument("--plot", action="store_true", help="Generate plots only")
    args = parser.parse_args()

    if not any([args.compute, args.plot]):
        args.compute = True

    try:
        data_file = require_existing_data_path(args.data)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    analyzer = STPODAnalyzer(
        file_path=data_file,
        embedding_dim=args.embedding_dim,
        n_modes_save=args.n_modes,
        spatial_weight_type="uniform",
    )

    if args.compute:
        analyzer.run_analysis()
    elif args.plot:
        analyzer.load_results()
        analyzer.plot_eigenvalues()
        analyzer.plot_modes()
        analyzer.plot_spacetime_mode()
        analyzer.plot_time_coefficients()
        analyzer.plot_cumulative_energy()
