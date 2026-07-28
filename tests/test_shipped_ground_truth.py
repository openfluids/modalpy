"""Assert shipped generators against their own analytic metadata.

Each generator in ``openmodalpy.example_data`` carries the closed-form answer in
its metadata dict. These tests hand that dict to analyzers and compare the
recovered quantity to the metadata value — never a copied literal. Tolerances
are derived from the record or spectral resolution so the bound moves with the
discretization.
"""

from __future__ import annotations

import numpy as np
import pytest

from openmodalpy import DMDAnalyzer, PODAnalyzer, SPODAnalyzer
from openmodalpy.example_data import generate_example_dataset


def _analyzer_data(payload: dict) -> dict:
    """Hand the generator payload to an analyzer (same keys as ``test_all`` loaders)."""
    return {
        "q": payload["q"],
        "x": payload["x"],
        "y": payload["y"],
        "z": payload["z"],
        "dt": payload["dt"],
        "Nx": payload["Nx"],
        "Ny": payload["Ny"],
        "Nz": payload["Nz"],
        "Ns": payload["Ns"],
        "metadata": payload.get("metadata", {}),
    }


def _loader(payload: dict):
    data = _analyzer_data(payload)
    return lambda _path: data


# ---------------------------------------------------------------------------
# Taylor-Green → DMD eigenvalue magnitude (exact rank-1 exponential)
# ---------------------------------------------------------------------------


def test_taylor_green_dmd_eigenvalue_matches_metadata(tmp_path):
    """DMD recovers |λ| = exp(-2 ν dt) stored as metadata ``dmd_eigenvalue``.

    WHY exact at machine precision: the field is one spatial pattern times a pure
    exponential, and exact DMD does not subtract the temporal mean, so the
    companion eigenvalue is the analytic discrete multiplier.
    """
    # Goal table size: 24×24, Nt=60 — far smaller than generator defaults.
    payload = generate_example_dataset("taylor_green", {"Nx": 24, "Ny": 24, "Nt": 60})
    expected = payload["metadata"]["dmd_eigenvalue"]

    analyzer = DMDAnalyzer(
        file_path="shipped_tg",
        data_loader=_loader(payload),
        spatial_weight_type="uniform",
        n_modes_save=5,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    with pytest.warns(DeprecationWarning, match="n_modes_save"):
        with pytest.warns(RuntimeWarning, match="effective rank"):
            analyzer.perform_dmd()

    recovered = float(np.abs(analyzer.eigenvalues[0]))
    # Rank-1 pure exponential → machine-precision agreement (measured abs err 0.0).
    rtol = 1e-10
    rel_err = abs(recovered - expected) / abs(expected)
    assert rel_err <= rtol, (
        f"Taylor-Green DMD |λ0|: recovered={recovered!r}, "
        f"metadata dmd_eigenvalue={expected!r}, rel_err={rel_err:.3e} (rtol={rtol})"
    )


# ---------------------------------------------------------------------------
# Cylinder wake → DMD frequency vs metadata f_shed
# ---------------------------------------------------------------------------


def test_cylinder_wake_dmd_frequency_matches_metadata(tmp_path):
    """DMD recovers shedding frequency from metadata ``f_shed``.

    Frequency from discrete eigenvalue: f = |arg(λ)| / (2π dt). Unlike a Welch
    periodogram, DMD is not bin-limited — it solves an eigenproblem, so it resolves
    far below the Rayleigh frequency 1/(Ns·dt) ~ 0.017 Hz. Measured error is 3e-5 Hz,
    i.e. 0.002 of a bin. Bounding at one full bin would pass a 10% frequency error
    (verified by mutation), so the bound is a tenth of a bin: still 55x the measured
    error, and tight enough to catch a 1% regression.
    """
    payload = generate_example_dataset("cylinder_wake", {"Nx": 40, "Ny": 24, "Nt": 400})
    expected = payload["metadata"]["f_shed"]
    dt = float(payload["dt"])
    Ns = int(payload["Ns"])
    # Record resolution: Rayleigh frequency of the full snapshot train.
    freq_res = 1.0 / (Ns * dt)
    # DMD is not bin-limited (see docstring): a tenth of a bin.
    bound = freq_res / 10.0

    analyzer = DMDAnalyzer(
        file_path="shipped_cw_dmd",
        data_loader=_loader(payload),
        spatial_weight_type="uniform",
        n_modes_save=10,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    with pytest.warns(DeprecationWarning, match="n_modes_save"):
        analyzer.perform_dmd()

    # Eigenvalues are ranked |λ| desc; take the first mode with a finite frequency
    # (pure-real mean/decay modes have arg≈0). Do not pick "closest to expected".
    recovered = None
    for lam in analyzer.eigenvalues:
        f = float(np.abs(np.angle(lam)) / (2.0 * np.pi * dt))
        if f > freq_res * 0.25:
            recovered = f
            break
    assert recovered is not None, (
        f"Cylinder DMD: no oscillatory eigenvalues; "
        f"raw |arg|/2πdt="
        f"{(np.abs(np.angle(analyzer.eigenvalues)) / (2.0 * np.pi * dt))[:6]!r}"
    )
    err = abs(recovered - expected)
    assert err <= bound, (
        f"Cylinder DMD f: recovered={recovered:.6g} Hz, "
        f"metadata f_shed={expected:.6g} Hz, err={err:.3e} Hz, "
        f"bound 1/(10*Ns*dt)={bound:.6g} Hz"
    )


# ---------------------------------------------------------------------------
# Cylinder wake → SPOD leading-mode spectral peak vs metadata f_shed
# ---------------------------------------------------------------------------


def test_cylinder_wake_spod_peak_matches_metadata(tmp_path):
    """SPOD peak of the leading eigenvalue lands in the FFT bin nearest ``f_shed``.

    SPOD frequency axis is ``analyzer.freq`` (rfftfreq), so the recovered value can
    only ever be a bin centre. The correct claim is therefore "the peak lands in the
    bin NEAREST f_shed", which is a half-bin bound, fs/(2·nfft). A full-bin bound
    would also accept the adjacent bin: at nfft=256 the neighbours sit 0.0108 Hz and
    0.0152 Hz from f_shed and both are under fs/nfft = 0.0260 Hz, so rolling the
    spectrum by one bin passed unnoticed (verified by mutation). Half a bin admits
    only the true neighbour, with 17% margin.
    """
    payload = generate_example_dataset("cylinder_wake", {"Nx": 40, "Ny": 24, "Nt": 400})
    expected = payload["metadata"]["f_shed"]
    dt = float(payload["dt"])
    nfft = 256
    overlap = 0.5
    fs = 1.0 / dt
    # Welch frequency resolution: one rfft bin width; the peak must land in the
    # NEAREST bin, which is half of it (see docstring).
    bin_width = fs / nfft
    bound = bin_width / 2.0

    analyzer = SPODAnalyzer(
        file_path="shipped_cw_spod",
        data_loader=_loader(payload),
        spatial_weight_type="uniform",
        nfft=nfft,
        overlap=overlap,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()

    # Leading SPOD mode energy vs frequency; skip DC bin for the peak search.
    lead = analyzer.eigenvalues[:, 0]
    peak_idx = 1 + int(np.argmax(lead[1:]))
    recovered = float(analyzer.freq[peak_idx])
    err = abs(recovered - expected)
    assert err <= bound, (
        f"Cylinder SPOD peak: recovered={recovered:.6g} Hz, "
        f"metadata f_shed={expected:.6g} Hz, err={err:.3e} Hz, "
        f"bound fs/(2*nfft)={bound:.6g} Hz"
    )


# ---------------------------------------------------------------------------
# Double gyre → POD a_1 spectrum peak vs metadata expected_freq
# ---------------------------------------------------------------------------


def test_double_gyre_pod_a1_peak_matches_metadata(tmp_path):
    """POD leading time coefficient peaks at metadata ``expected_freq``.

    Double gyre defaults only cover ~2 cycles (t_max=20); use t_max=100 so the
    record holds ~10 periods. Peak from rfft of a_1; bound is half the record
    resolution 1/(2·Ns·dt) — measured err 2.5e-4 Hz vs 1/(2T) ~ 5e-3 Hz.
    """
    payload = generate_example_dataset(
        "double_gyre",
        {"Nx": 32, "Ny": 16, "Nt": 400, "t_max": 100.0},
    )
    expected = payload["metadata"]["expected_freq"]
    dt = float(payload["dt"])
    Ns = int(payload["Ns"])
    # Half-bin of the full-record Rayleigh frequency (Nyquist spacing of rfft is 1/T).
    half_bin = 1.0 / (2.0 * Ns * dt)

    analyzer = PODAnalyzer(
        file_path="shipped_dg_pod",
        data_loader=_loader(payload),
        spatial_weight_type="uniform",
        n_modes_save=5,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    a1 = np.asarray(analyzer.time_coefficients[:, 0], dtype=float)
    a1 = a1 - a1.mean()
    # rfftfreq spacing = 1/(len(a1)*dt) = 1/(Ns*dt); peak search skips DC.
    spectrum = np.abs(np.fft.rfft(a1))
    freqs = np.fft.rfftfreq(len(a1), d=dt)
    peak_idx = 1 + int(np.argmax(spectrum[1:]))
    recovered = float(freqs[peak_idx])
    err = abs(recovered - expected)
    assert err <= half_bin, (
        f"Double-gyre POD a1 peak: recovered={recovered:.6g} Hz, "
        f"metadata expected_freq={expected:.6g} Hz, err={err:.3e} Hz, "
        f"bound 1/(2*Ns*dt)={half_bin:.6g} Hz"
    )


# ---------------------------------------------------------------------------
# Cross-analyzer: cylinder DMD frequency vs SPOD peak (one Welch bin)
# ---------------------------------------------------------------------------


def test_cylinder_wake_dmd_spod_frequency_cross_agreement(tmp_path):
    """DMD and SPOD must agree on the cylinder shedding frequency within one bin.

    WHY this adds information: the two tests above compare each analyzer to the same
    metadata value, so both would still pass if the generator's own physics drifted
    away from what it advertises — the checks would move together with it. This one
    compares the analyzers to EACH OTHER, using no metadata as the reference, so it
    holds even then and pins that the two independent estimators agree. Bound is the
    coarser of the two resolutions, SPOD's fs/nfft, since that is the floor on how
    closely a bin-limited method can track an eigenproblem-based one.
    """
    payload = generate_example_dataset("cylinder_wake", {"Nx": 40, "Ny": 24, "Nt": 400})
    expected = payload["metadata"]["f_shed"]
    dt = float(payload["dt"])
    Ns = int(payload["Ns"])
    nfft = 256
    overlap = 0.5
    fs = 1.0 / dt
    bin_width = fs / nfft
    freq_res = 1.0 / (Ns * dt)

    dmd = DMDAnalyzer(
        file_path="shipped_cw_cross_dmd",
        data_loader=_loader(payload),
        spatial_weight_type="uniform",
        n_modes_save=10,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    dmd.load_and_preprocess()
    with pytest.warns(DeprecationWarning, match="n_modes_save"):
        dmd.perform_dmd()
    f_dmd = None
    for lam in dmd.eigenvalues:
        f = float(np.abs(np.angle(lam)) / (2.0 * np.pi * dt))
        if f > freq_res * 0.25:
            f_dmd = f
            break
    assert f_dmd is not None, (
        f"Cross-agree DMD: no oscillatory modes; "
        f"raw |arg|/2πdt="
        f"{(np.abs(np.angle(dmd.eigenvalues)) / (2.0 * np.pi * dt))[:6]!r}"
    )

    spod = SPODAnalyzer(
        file_path="shipped_cw_cross_spod",
        data_loader=_loader(payload),
        spatial_weight_type="uniform",
        nfft=nfft,
        overlap=overlap,
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    spod.load_and_preprocess()
    spod.compute_fft_blocks()
    spod.perform_spod()
    lead = spod.eigenvalues[:, 0]
    peak_idx = 1 + int(np.argmax(lead[1:]))
    f_spod = float(spod.freq[peak_idx])

    # Each method still lands near the shipped analytic frequency.
    # Same tenth-of-a-bin DMD bound as the dedicated test above.
    assert abs(f_dmd - expected) <= freq_res / 10.0, (
        f"Cross-agree DMD vs f_shed: f_dmd={f_dmd:.6g}, f_shed={expected:.6g}, "
        f"err={abs(f_dmd - expected):.3e}, bound 1/(10*Ns*dt)={freq_res / 10.0:.6g}"
    )
    # Same nearest-bin rule as the dedicated SPOD test above.
    assert abs(f_spod - expected) <= bin_width / 2.0, (
        f"Cross-agree SPOD vs f_shed: f_spod={f_spod:.6g}, f_shed={expected:.6g}, "
        f"err={abs(f_spod - expected):.3e}, bound fs/(2*nfft)={bin_width / 2.0:.6g}"
    )
    # Cross-analyzer: one Welch bin (SPOD is the coarser resolver).
    cross_err = abs(f_dmd - f_spod)
    assert cross_err <= bin_width, (
        f"Cross-agree DMD vs SPOD: f_dmd={f_dmd:.6g}, f_spod={f_spod:.6g}, "
        f"err={cross_err:.3e} Hz, bound fs/nfft={bin_width:.6g} Hz"
    )
