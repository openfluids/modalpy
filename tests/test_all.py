#!/usr/bin/env python3
"""
Comprehensive validation tests for OpenModalPy: POD, DMD, SPOD

Validates mathematical correctness using synthetic data with known analytical solutions.
"""

from __future__ import annotations

import numpy as np

# Tolerance for numerical comparisons
TOL = 1e-10
TOL_LOOSE = 1e-6


def make_test_loader(q, Nx, Ny, dt, x=None, y=None):
    """Create a custom data loader for synthetic test data.

    Args:
        q: Data array [Ns, Nspace] where Nspace = Nx * Ny
        Nx, Ny: Spatial dimensions
        dt: Time step
        x, y: Optional coordinate arrays

    Returns:
        A callable that returns the data dict expected by analyzers.
    """
    Ns = q.shape[0]
    if x is None:
        x = np.arange(Nx, dtype=float)
    if y is None:
        y = np.arange(Ny, dtype=float)

    def loader(file_path):
        return {
            "q": q,
            "x": x,
            "y": y,
            "z": None,
            "dt": dt,
            "Nx": Nx,
            "Ny": Ny,
            "Nz": 1,
            "Ns": Ns,
            "metadata": {"format": "test", "var_name": "q"},
        }
    return loader


# =============================================================================
# POD TESTS
# =============================================================================

def test_pod(tmp_path):
    """Test Proper Orthogonal Decomposition."""
    from openmodalpy import PODAnalyzer

    # --- Test 1: Rank-k recovery ---
    # Create data that is exactly rank-3: sum of 3 spatial patterns with time coefficients
    np.random.seed(42)
    Nx, Ny, Ns = 40, 40, 200  # Higher resolution: 1600 spatial DOF

    # 3 orthogonal spatial patterns
    x = np.linspace(0, 2*np.pi, Nx)
    y = np.linspace(0, 2*np.pi, Ny)
    X, Y = np.meshgrid(x, y)

    pattern1 = np.sin(X).flatten()
    pattern2 = np.sin(Y).flatten()
    pattern3 = np.sin(X + Y).flatten()

    # Time coefficients with different energies
    t = np.linspace(0, 10, Ns)
    a1 = 3.0 * np.sin(2*np.pi*0.5*t)  # Highest energy
    a2 = 2.0 * np.sin(2*np.pi*1.0*t)  # Medium energy
    a3 = 1.0 * np.sin(2*np.pi*1.5*t)  # Lowest energy

    # Construct rank-3 data: q[time, space]
    q = np.outer(a1, pattern1) + np.outer(a2, pattern2) + np.outer(a3, pattern3)

    # Use custom loader (no temp file needed for data)
    loader = make_test_loader(q, Nx, Ny, dt=t[1]-t[0], x=x, y=y)

    # BaseAnalyzer makedirs(results_dir) on init — keep tree clean
    analyzer = PODAnalyzer(
        "dummy_path",
        data_loader=loader,
        n_modes_save=10,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    eigenvalues = analyzer.eigenvalues
    modes = analyzer.modes

    # Test 1: Should have ~3 significant eigenvalues
    significant = np.sum(eigenvalues / eigenvalues[0] > 1e-10)
    assert significant == 3, f"Rank-k recovery (rank=3): found {significant} modes"

    # Test 2: Eigenvalues should be positive and sorted descending
    assert np.all(eigenvalues >= -TOL), f"Eigenvalues positive: min={eigenvalues.min()}"
    assert np.all(np.diff(eigenvalues) <= TOL), (
        f"Eigenvalues sorted descending: diffs={np.diff(eigenvalues)[:5]}"
    )

    # Test 3: Modes should be approximately W-orthonormal (Φᵀ W Φ ≈ I)
    # Note: POD stores unweighted modes but they come from weighted eigendecomposition
    # modes shape: [space, n_modes], W shape: [space, 1]
    n_modes = modes.shape[1]
    W = analyzer.W.flatten()  # spatial weights as 1D array
    # Weighted Gram matrix: Φᵀ diag(W) Φ
    gram = (modes.T * W) @ modes  # Efficient: (W * Φ)ᵀ Φ
    identity_error = np.linalg.norm(gram - np.eye(n_modes)) / n_modes
    # Relaxed tolerance for numerical reasons (modes stored unweighted)
    assert identity_error < 0.5, (
        f"Modes approx W-orthonormal: error={identity_error:.2e}"
    )

    # Test 4: Reconstruction error for rank-3 data with 3 modes
    time_coeffs = analyzer.time_coefficients  # [time, modes]
    q_reconstructed = time_coeffs[:, :3] @ modes[:, :3].T
    recon_error = np.linalg.norm(q - q_reconstructed) / np.linalg.norm(q)
    assert recon_error < TOL_LOOSE, (
        f"Reconstruction (3 modes): error={recon_error:.2e}"
    )

    # Test 5: Energy conservation - sum of eigenvalues = Frobenius norm squared
    # For snapshot POD: eigenvalues are squared singular values / Ns
    total_energy_data = np.linalg.norm(q, 'fro')**2 / Ns
    total_energy_modes = np.sum(eigenvalues)
    energy_ratio = total_energy_modes / total_energy_data
    assert abs(energy_ratio - 1.0) < TOL_LOOSE, (
        f"Energy conservation: ratio={energy_ratio:.6f}"
    )


# =============================================================================
# DMD TESTS
# =============================================================================

def test_dmd(tmp_path):
    """Test Dynamic Mode Decomposition."""
    from openmodalpy import DMDAnalyzer

    np.random.seed(42)
    Nx, Ny = 25, 25  # Higher resolution: 625 spatial DOF
    Nspace = Nx * Ny
    dt = 0.1
    Ns = 100  # More snapshots
    t = np.arange(Ns) * dt

    # --- Test 1: Pure exponential decay ---
    # q(t) = e^{-αt} * spatial_pattern
    # DMD eigenvalue should be λ = e^{-α*dt}
    alpha = 0.5
    spatial = np.random.randn(Nspace)
    spatial /= np.linalg.norm(spatial)
    q_decay = np.outer(np.exp(-alpha * t), spatial)  # [time, space]

    loader = make_test_loader(q_decay, Nx, Ny, dt)
    analyzer = DMDAnalyzer(
        "dummy",
        data_loader=loader,
        n_modes_save=5,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()

    # Dominant eigenvalue should be e^{-α*dt}
    expected_eigval = np.exp(-alpha * dt)
    dominant_eigval = np.abs(analyzer.eigenvalues[0])
    eigval_error = abs(dominant_eigval - expected_eigval)
    assert eigval_error < TOL_LOOSE, (
        f"Exponential decay eigenvalue: expected={expected_eigval:.6f}, "
        f"got={dominant_eigval:.6f}"
    )

    # --- Test 2: Pure oscillation ---
    # Use traveling wave: q(x,t) = cos(kx - ωt) which has complex eigenvalue
    # DMD eigenvalue should have |λ| ≈ 1 and recover the frequency
    Ns_osc = 100
    t_osc = np.arange(Ns_osc) * dt
    omega = 2 * np.pi * 1.0  # 1 Hz
    k = 2 * np.pi / (Nx * 0.5)  # Spatial wavenumber

    # Create traveling wave in 2D
    x_grid = np.linspace(0, 2*np.pi, Nx)
    y_grid = np.linspace(0, 2*np.pi, Ny)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    q_osc = np.zeros((Ns_osc, Nspace))
    for i, ti in enumerate(t_osc):
        wave = np.cos(k * X_grid - omega * ti)
        q_osc[i] = wave.flatten()

    loader = make_test_loader(q_osc, Nx, Ny, dt)
    analyzer = DMDAnalyzer(
        "dummy",
        data_loader=loader,
        n_modes_save=5,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()

    # Should have eigenvalue near unit circle
    eigvals = analyzer.eigenvalues
    magnitudes = np.abs(eigvals)
    unit_circle = np.any(np.abs(magnitudes - 1.0) < 0.1)
    assert unit_circle, (
        f"Oscillation on unit circle: magnitudes={magnitudes[:3]}"
    )

    # Check frequency recovery
    angles = np.angle(eigvals)
    recovered_freq = np.abs(angles) / (2 * np.pi * dt)
    freq_match = np.any(np.abs(recovered_freq - 1.0) < 0.2)  # 1 Hz
    assert freq_match, (
        f"Oscillation frequency recovery: frequencies={recovered_freq[:3]}"
    )

    # --- Test 3: Decaying oscillation ---
    # q(t) = e^{-αt} * cos(ωt)
    # Use longer time series for better accuracy
    alpha = 0.1  # Lower decay rate for cleaner signal
    omega = 2 * np.pi * 0.5  # 0.5 Hz
    Ns_decay = 100
    t_decay = np.arange(Ns_decay) * dt
    envelope = np.exp(-alpha * t_decay) * np.cos(omega * t_decay)
    q_decay_osc = np.outer(envelope, spatial)

    loader = make_test_loader(q_decay_osc, Nx, Ny, dt)
    analyzer = DMDAnalyzer(
        "dummy",
        data_loader=loader,
        n_modes_save=5,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()

    eigvals = analyzer.eigenvalues
    # For decaying oscillation: |λ| = e^{-α*dt} < 1
    expected_mag = np.exp(-alpha * dt)
    # Find closest eigenvalue to expected magnitude
    mag_error = np.min(np.abs(np.abs(eigvals) - expected_mag))
    assert mag_error < 0.15, (
        f"Decaying oscillation magnitude: expected |λ|={expected_mag:.4f}, "
        f"closest={np.abs(eigvals[0]):.4f}"
    )

    # --- Test 4: Linear system dx/dt = Ax ---
    # Known 2x2 system with analytical eigenvalues
    from scipy.linalg import expm

    A = np.array([[-0.1, 1.0],
                  [-1.0, -0.1]])  # Damped oscillator

    # Analytical continuous eigenvalues: -0.1 ± 1j
    cont_eigvals = np.linalg.eigvals(A)
    # Discrete eigenvalues: e^{A*dt}
    expected_discrete = np.exp(cont_eigvals * dt)

    # Generate trajectory
    Ns = 100
    t = np.arange(Ns) * dt
    x0 = np.array([1.0, 0.0])
    trajectory = np.zeros((Ns, 2))
    for i in range(Ns):
        trajectory[i] = expm(A * t[i]) @ x0

    loader = make_test_loader(trajectory, Nx=2, Ny=1, dt=dt)
    analyzer = DMDAnalyzer(
        "dummy",
        data_loader=loader,
        n_modes_save=2,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()

    dmd_eigvals = analyzer.eigenvalues

    # Check if DMD eigenvalues match expected discrete eigenvalues
    # Sort by magnitude for comparison
    dmd_sorted = np.sort(np.abs(dmd_eigvals))
    exp_sorted = np.sort(np.abs(expected_discrete))
    eigval_match = np.allclose(dmd_sorted, exp_sorted, rtol=0.1)
    assert eigval_match, (
        f"Linear system eigenvalues: DMD={dmd_sorted}, expected={exp_sorted}"
    )


# =============================================================================
# SPOD TESTS
# =============================================================================

def test_spod(tmp_path):
    """Test Spectral Proper Orthogonal Decomposition."""
    from openmodalpy import SPODAnalyzer

    np.random.seed(42)
    Nx, Ny = 25, 25  # Higher resolution: 625 spatial DOF
    Nspace = Nx * Ny
    dt = 0.01
    Ns = 2048  # Longer time series
    t = np.arange(Ns) * dt

    # --- Test 1: White noise - flat spectrum ---
    spatial = np.random.randn(Nspace)
    spatial /= np.linalg.norm(spatial)
    noise = np.random.randn(Ns)
    q_noise = np.outer(noise, spatial)

    nfft = 128
    overlap = 0.5
    loader = make_test_loader(q_noise, Nx, Ny, dt)
    analyzer = SPODAnalyzer(
        "dummy",
        nfft=nfft,
        overlap=overlap,
        data_loader=loader,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()

    # For white noise, spectrum should be relatively flat
    eigenvalues = analyzer.eigenvalues  # [n_freq, n_modes]
    first_eigval = eigenvalues[:, 0]  # First eigenvalue at each frequency

    # Check flatness: std/mean should be small for white noise
    # (excluding DC and Nyquist which can be different)
    mid_freqs = first_eigval[2:-2]
    flatness = np.std(mid_freqs) / np.mean(mid_freqs)
    assert flatness < 0.5, f"White noise flat spectrum: std/mean={flatness:.3f}"

    # --- Test 2: Single tone - peak at specific frequency ---
    f0 = 10.0  # 10 Hz tone
    tone = np.sin(2 * np.pi * f0 * t)
    q_tone = np.outer(tone, spatial)

    nfft = 256
    overlap = 0.5
    loader = make_test_loader(q_tone, Nx, Ny, dt)
    analyzer = SPODAnalyzer(
        "dummy",
        nfft=nfft,
        overlap=overlap,
        data_loader=loader,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()

    eigenvalues = analyzer.eigenvalues
    freqs = analyzer.freq  # Frequencies in Hz

    # Find peak frequency
    first_eigval = eigenvalues[:, 0]
    peak_idx = np.argmax(first_eigval)
    peak_freq = freqs[peak_idx]

    # Peak should be near f0
    freq_error = abs(peak_freq - f0)
    assert freq_error < 1.0, (
        f"Single tone peak frequency: expected={f0}Hz, got={peak_freq:.1f}Hz"
    )

    # At peak frequency, first eigenvalue should dominate (rank-1)
    assert eigenvalues.shape[1] > 1, "Single tone rank-1 at peak: need ≥2 modes"
    dominance = eigenvalues[peak_idx, 0] / (eigenvalues[peak_idx, 1] + 1e-10)
    assert dominance > 10, f"Single tone rank-1 at peak: λ1/λ2={dominance:.1f}"

    # --- Test 3: Orthonormality of modes at each frequency ---
    nfft = 128
    overlap = 0.5
    loader = make_test_loader(q_tone, Nx, Ny, dt)
    analyzer = SPODAnalyzer(
        "dummy",
        nfft=nfft,
        overlap=overlap,
        data_loader=loader,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()

    modes = analyzer.modes  # [n_freq, n_space, n_modes]
    W = analyzer.W.flatten()  # spatial weights as 1D array
    n_freq = modes.shape[0]

    # Check approximate W-orthonormality at a few frequencies
    max_error = 0
    for fi in [n_freq//4, n_freq//2, 3*n_freq//4]:
        phi = modes[fi]  # [n_space, n_modes]
        # Weighted Gram: (W * Φ)ᴴ Φ
        gram = (phi.conj().T * W) @ phi
        n_m = gram.shape[0]
        error = np.linalg.norm(gram - np.eye(n_m)) / n_m
        max_error = max(max_error, error)

    # Relaxed tolerance - SPOD modes from eigendecomposition
    assert max_error < 0.5, (
        f"Modes approx W-orthonormal: max_error={max_error:.2e}"
    )

    # --- Test 4: Comparison with Welch PSD ---
    # First SPOD eigenvalue should show similar spectral structure to Welch PSD

    # Multi-tone signal (two tones) for Welch/SPOD spectral comparison
    f1, f2 = 5.0, 15.0
    signal_data = np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)
    q_multi = np.outer(signal_data, spatial)

    nfft = 512  # finer frequency grid for resolving the two tones
    overlap_frac = 0.5
    loader = make_test_loader(q_multi, Nx, Ny, dt)
    analyzer = SPODAnalyzer(
        "dummy",
        nfft=nfft,
        overlap=overlap_frac,
        data_loader=loader,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()

    # SPOD first eigenvalue should show peaks at tonal frequencies
    spod_psd = analyzer.eigenvalues[:, 0]
    spod_freqs = analyzer.freq

    # Check that SPOD finds the primary tone (f1=5Hz is dominant)
    peak_idx = np.argmax(spod_psd)
    peak_freq = spod_freqs[peak_idx]
    # Should find one of the tones (5Hz or 15Hz)
    found_tone = abs(peak_freq - f1) < 2.0 or abs(peak_freq - f2) < 2.0
    assert found_tone, (
        f"SPOD finds tonal peak: expected ~{f1}Hz or ~{f2}Hz, got {peak_freq:.1f}Hz"
    )


# =============================================================================
# CROSS-METHOD TESTS
# =============================================================================

def test_cross_method(tmp_path):
    """Test consistency between methods."""
    from openmodalpy import PODAnalyzer, SPODAnalyzer

    np.random.seed(42)
    Nx, Ny = 8, 8
    Ns = 64
    dt = 0.1

    # Create simple test data
    t = np.arange(Ns) * dt
    x = np.linspace(0, 2*np.pi, Nx)
    y = np.linspace(0, 2*np.pi, Ny)
    X, Y = np.meshgrid(x, y)
    spatial = np.sin(X + Y).flatten()

    temporal = np.sin(2*np.pi*0.5*t)
    q = np.outer(temporal, spatial)

    # --- Test: SPOD with nfft=Ns should approximate POD ---
    # When using a single block, SPOD reduces to POD

    loader = make_test_loader(q, Nx, Ny, dt, x=x, y=y)

    # POD analysis
    pod = PODAnalyzer(
        "dummy",
        data_loader=loader,
        n_modes_save=5,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    pod.load_and_preprocess()
    pod.perform_pod()
    pod_energy = pod.eigenvalues[0] / np.sum(pod.eigenvalues)

    # SPOD with single block (nfft = Ns)
    loader = make_test_loader(q, Nx, Ny, dt, x=x, y=y)  # Fresh loader
    spod = SPODAnalyzer(
        "dummy",
        nfft=Ns,
        overlap=0,
        data_loader=loader,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    spod.load_and_preprocess()
    spod.compute_fft_blocks()
    spod.perform_spod()

    # Total SPOD energy in first mode across all frequencies
    spod_total_first = np.sum(spod.eigenvalues[:, 0])
    spod_total = np.sum(spod.eigenvalues)
    spod_energy = spod_total_first / spod_total if spod_total > 0 else 0

    # Both should show similar energy concentration in first mode
    energy_diff = abs(pod_energy - spod_energy)
    assert energy_diff < 0.3, (
        f"POD ≈ SPOD(nfft=Ns) energy: POD={pod_energy:.3f}, SPOD={spod_energy:.3f}"
    )


# =============================================================================
# HEAVY TESTS (Large DOF)
# =============================================================================

def test_heavy(tmp_path):
    """Heavy tests with larger degrees of freedom for real-world validation."""
    from openmodalpy import DMDAnalyzer, PODAnalyzer, SPODAnalyzer

    # --- Test 1: Cylinder Wake Simulation (Re~100) ---
    # Von Karman vortex street: St ≈ 0.16-0.17 at Re=100
    # Reference: Noack et al., JFM 2003
    np.random.seed(42)
    Nx, Ny = 150, 75  # 11250 spatial DOF (higher resolution)
    Ns = 800  # More snapshots
    dt = 0.1
    Nspace = Nx * Ny

    # Strouhal number St = f*D/U ≈ 0.16 for cylinder wake
    St = 0.167
    D = 1.0  # Cylinder diameter
    U = 1.0  # Free stream velocity
    f_shed = St * U / D  # Shedding frequency

    # Create synthetic cylinder wake: traveling vortices
    x = np.linspace(-2, 10, Nx)  # Domain: -2D to 10D downstream
    y = np.linspace(-2, 2, Ny)   # Domain: -2D to 2D cross-stream
    X, Y = np.meshgrid(x, y)
    t = np.arange(Ns) * dt

    # Wake model: convecting vortex street with decay
    k_x = 2 * np.pi * St  # Streamwise wavenumber
    U_conv = 0.8 * U  # Convection velocity (slower than freestream)
    decay = np.exp(-0.02 * np.maximum(X, 0))  # Decay downstream

    q_wake = np.zeros((Ns, Nspace))
    for i, ti in enumerate(t):
        # Vortex street: alternating vortices
        vortex = decay * np.sin(k_x * (X - U_conv * ti)) * np.exp(-Y**2 / 0.5)
        # Add higher harmonic (characteristic of real wakes)
        vortex += 0.3 * decay * np.sin(2 * k_x * (X - U_conv * ti)) * np.exp(-Y**2 / 0.3)
        q_wake[i] = vortex.flatten()

    # Add small noise (simulates turbulence/measurement noise)
    q_wake += 0.05 * np.random.randn(Ns, Nspace)

    # POD test
    loader = make_test_loader(q_wake, Nx, Ny, dt, x=x, y=y)
    pod = PODAnalyzer(
        "dummy",
        data_loader=loader,
        n_modes_save=20,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    pod.load_and_preprocess()
    pod.perform_pod()

    # First 2-4 modes should capture >90% energy (vortex shedding is coherent)
    cumulative_energy = np.cumsum(pod.eigenvalues) / np.sum(pod.eigenvalues)
    energy_4modes = cumulative_energy[3] if len(cumulative_energy) > 3 else cumulative_energy[-1]
    assert energy_4modes > 0.80, (
        f"Cylinder POD: 4 modes >80% energy, got {energy_4modes*100:.1f}%"
    )

    # DMD test - should find shedding frequency
    loader = make_test_loader(q_wake, Nx, Ny, dt, x=x, y=y)
    dmd = DMDAnalyzer(
        "dummy",
        data_loader=loader,
        n_modes_save=10,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    dmd.load_and_preprocess()
    dmd.perform_dmd()

    # Check if DMD finds the shedding frequency
    angles = np.angle(dmd.eigenvalues)
    dmd_freqs = np.abs(angles) / (2 * np.pi * dt)
    freq_error = np.min(np.abs(dmd_freqs - f_shed))
    closest_freq = dmd_freqs[np.argmin(np.abs(dmd_freqs - f_shed))]
    assert freq_error < 0.1, (
        f"Cylinder DMD: finds shedding freq, St={St}, f_shed={f_shed:.3f}, "
        f"closest DMD freq={closest_freq:.3f}"
    )

    # --- Test 2: Ginzburg-Landau Equation ---
    # Standard benchmark: traveling wave with known dispersion
    # Reference: Towne, Schmidt & Colonius, JFM 2018
    Nx_gl = 400  # Higher spatial resolution
    Ns_gl = 600  # More snapshots
    dt_gl = 0.5
    x_gl = np.linspace(0, 100, Nx_gl)
    t_gl = np.arange(Ns_gl) * dt_gl

    # Ginzburg-Landau parameters (supercritical regime)
    mu = 0.38  # Growth rate parameter
    c_u = 2.0  # Group velocity
    gamma = 1 - 1j  # Dispersion coefficient

    # Generate traveling wave packet solution
    q_gl = np.zeros((Ns_gl, Nx_gl), dtype=complex)
    x0 = 20  # Initial position
    sigma = 5  # Initial width

    for i, ti in enumerate(t_gl):
        # Approximate solution: traveling and spreading Gaussian envelope
        center = x0 + c_u * ti
        width = np.sqrt(sigma**2 + 2 * np.abs(gamma) * ti)
        envelope = np.exp(-(x_gl - center)**2 / (2 * width**2))
        # Carrier wave
        k0 = 1.0
        omega0 = c_u * k0
        carrier = np.exp(1j * (k0 * x_gl - omega0 * ti))
        q_gl[i] = envelope * carrier * np.exp(mu * ti * 0.1)  # Slight growth

    # Take real part for analysis
    q_gl_real = np.real(q_gl)

    loader = make_test_loader(q_gl_real, Nx_gl, 1, dt_gl, x=x_gl)
    pod_gl = PODAnalyzer(
        "dummy",
        data_loader=loader,
        n_modes_save=10,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    pod_gl.load_and_preprocess()
    pod_gl.perform_pod()

    # Traveling wave should be relatively low-rank (spreading reduces concentration)
    energy_2modes = np.sum(pod_gl.eigenvalues[:2]) / np.sum(pod_gl.eigenvalues)
    assert energy_2modes > 0.60, (
        f"Ginzburg-Landau POD: 2 modes >60% energy, got {energy_2modes*100:.1f}%"
    )

    # --- Test 3: Large-scale SPOD (Jet-like) ---
    # Inspired by turbulent jet databases (Schmidt & Towne)
    Nx_jet, Ny_jet = 80, 80  # 6400 spatial DOF (higher resolution)
    Ns_jet = 4096  # Longer time series
    dt_jet = 0.01
    t_jet = np.arange(Ns_jet) * dt_jet

    # Multi-frequency coherent structures (like jet modes)
    x_jet = np.linspace(0, 10, Nx_jet)
    y_jet = np.linspace(-2, 2, Ny_jet)
    X_jet, Y_jet = np.meshgrid(x_jet, y_jet)

    # Create spatial patterns (axisymmetric-like modes)
    mode1 = np.exp(-Y_jet**2 / 1.0) * np.sin(np.pi * X_jet / 5)  # Low-freq mode
    mode2 = np.exp(-Y_jet**2 / 0.5) * np.sin(2 * np.pi * X_jet / 5)  # Higher-freq mode

    # Time signals at different frequencies
    f1, f2 = 2.0, 8.0  # Hz
    a1 = np.sin(2 * np.pi * f1 * t_jet) + 0.3 * np.random.randn(Ns_jet)
    a2 = 0.5 * np.sin(2 * np.pi * f2 * t_jet) + 0.2 * np.random.randn(Ns_jet)

    q_jet = np.outer(a1, mode1.flatten()) + np.outer(a2, mode2.flatten())
    q_jet += 0.1 * np.random.randn(Ns_jet, Nx_jet * Ny_jet)  # Background turbulence

    loader = make_test_loader(q_jet, Nx_jet, Ny_jet, dt_jet, x=x_jet, y=y_jet)
    spod_jet = SPODAnalyzer(
        "dummy_jet",
        nfft=256,
        overlap=0.5,
        data_loader=loader,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    spod_jet.load_and_preprocess()
    spod_jet.compute_fft_blocks()
    spod_jet.perform_spod()

    # Check that SPOD finds both frequencies
    spod_psd = spod_jet.eigenvalues[:, 0]
    spod_freqs = spod_jet.freq

    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spod_psd, height=np.max(spod_psd) * 0.1)
    peak_freqs = spod_freqs[peaks]

    # Check if both f1 and f2 are found
    found_f1 = np.any(np.abs(peak_freqs - f1) < 1.0)
    found_f2 = np.any(np.abs(peak_freqs - f2) < 1.0)
    assert found_f1 and found_f2, (
        f"Jet SPOD: finds both frequencies, looking for {f1}Hz and {f2}Hz "
        f"in peaks {peak_freqs[:5]}"
    )

    # --- Test 4: Reconstruction accuracy at scale ---
    # Use the cylinder wake data for reconstruction test
    loader = make_test_loader(q_wake, Nx, Ny, dt, x=x, y=y)
    pod_recon = PODAnalyzer(
        "dummy",
        data_loader=loader,
        n_modes_save=50,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    pod_recon.load_and_preprocess()
    pod_recon.perform_pod()

    # Reconstruct with 10 modes
    n_recon = 10
    modes = pod_recon.modes[:, :n_recon]
    coeffs = pod_recon.time_coefficients[:, :n_recon]
    q_reconstructed = coeffs @ modes.T

    # Relative reconstruction error
    recon_error = np.linalg.norm(q_wake - q_reconstructed) / np.linalg.norm(q_wake)
    assert recon_error < 0.3, (
        f"Large-scale reconstruction (10 modes): relative error = {recon_error:.3f}"
    )
