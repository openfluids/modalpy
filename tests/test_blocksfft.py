import numpy as np
import pytest

from openmodalpy.core.base import PARALLEL_AVAILABLE, blocksfft, spod_function
from openmodalpy.core.parallel import blocksfft_optimized

WINDOWS = ("hamming", "hann", "blackman", "bartlett", "sine")


# Floor partitioning matching scipy.signal.welch / the fixed production formula.
def _nblocks_floor(Ns, nfft, overlap):
    novlap = int(overlap * nfft)
    return (Ns - novlap) // (nfft - novlap), novlap


def _block_starts(nblocks, nfft, novlap):
    hop = nfft - novlap
    return [iblk * hop for iblk in range(nblocks)]


def test_blocksfft_constant_signal():
    q = np.ones((4, 2))
    result = blocksfft(q, nfft=4, nblocks=1, novlap=0)
    assert result.shape == (4 // 2 + 1, 2, 1)
    assert np.allclose(result, 0)


@pytest.mark.parametrize("window_type", WINDOWS)
def test_blocksfft_serial_parallel_window_identity(window_type):
    """Serial and parallel paths must agree bit-for-bit for every window."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((64, 3))
    ser = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type=window_type, use_parallel=False)
    par = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type=window_type, use_parallel=True)
    assert np.allclose(ser, par, rtol=0, atol=1e-12), (
        f"serial/parallel disagree for window_type={window_type!r}: max|diff|={np.max(np.abs(ser - par)):.3e}"
    )


def test_blocksfft_hann_blackman_differ_from_hamming():
    """hann/blackman must not silently collapse onto the hamming path."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((64, 3))
    hamming = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type="hamming")
    for w in ("hann", "blackman", "sine"):
        other = blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type=w)
        assert not np.allclose(other, hamming), f"{w} collapsed onto hamming"


def test_blocksfft_unsupported_window_raises():
    """Unsupported window names must raise, not silently substitute."""
    rng = np.random.default_rng(0)
    q = rng.standard_normal((64, 3))
    with pytest.raises(ValueError) as exc_info:
        blocksfft(q, nfft=16, nblocks=3, novlap=0, window_type="not_a_window")
    # The offending name must appear, so an unrelated ValueError cannot satisfy this.
    assert "not_a_window" in str(exc_info.value)


def test_normvar_divides_by_variance_two_scales():
    """normvar divides by variance: normalised block has var 1/v, at two scales.

    Provenance: spod_matlab opts.normvar / PySPOD normalize_data. Dividing by
    the standard deviation would yield unit variance and scale invariance;
    both scales must assert var -> 1/v so a "fix" to std would break this.
    """
    rng = np.random.default_rng(0)
    nfft, npts = 32, 3
    base = rng.standard_normal((nfft, npts))
    base = base - base.mean(axis=0)

    for scale in (1.0, 7.0):
        q = base * scale
        v = np.var(q, axis=0, ddof=1)
        # What the arithmetic does (same as base.py / parallel.py).
        normalised = q / v
        got = np.var(normalised, axis=0, ddof=1)
        expected = 1.0 / v
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

        # Full path: scaling input by c scales the FFT by 1/c under normvar.
        out = blocksfft(
            q,
            nfft=nfft,
            nblocks=1,
            novlap=0,
            normvar=True,
            use_parallel=False,
            window_type="boxcar",
        )
        out_base = blocksfft(
            base,
            nfft=nfft,
            nblocks=1,
            novlap=0,
            normvar=True,
            use_parallel=False,
            window_type="boxcar",
        )
        np.testing.assert_allclose(out, out_base / scale, rtol=0, atol=1e-12)


def test_normvar_serial_parallel_agree():
    """Serial and parallel blocksfft paths must agree with normvar=True.

    Both paths implement the same divide-by-variance step; neither was
    exercised with the flag on before this bead (two-paths-one-maintained).
    """
    rng = np.random.default_rng(42)
    q = rng.standard_normal((64, 4))
    kwargs = dict(nfft=16, nblocks=3, novlap=4, normvar=True, blockwise_mean=True)
    ser = blocksfft(q, use_parallel=False, **kwargs)
    par = blocksfft(q, use_parallel=True, **kwargs)
    np.testing.assert_allclose(
        ser,
        par,
        rtol=0,
        atol=1e-12,
        err_msg=f"serial/parallel disagree with normvar=True: max|diff|={np.max(np.abs(ser - par)):.3e}",
    )


# ---------------------------------------------------------------------------
# Serial/parallel identity across the real parameter surface
# ---------------------------------------------------------------------------
# Both paths own separate window_norm / mean / normvar / placement code. The
# three point tests above pin windows@novlap=0, one normvar case, and boxcar
# placement; this sweep covers the full cross product, including amplitude
# normalization (no prior identity coverage).


@pytest.mark.parametrize("window_type", ["hamming", "hann"])
@pytest.mark.parametrize("window_norm", ["power", "amplitude"], ids=["power", "amplitude"])
@pytest.mark.parametrize("blockwise_mean", [False, True], ids=["bwmean0", "bwmean1"])
@pytest.mark.parametrize("normvar", [False, True], ids=["nv0", "nv1"])
@pytest.mark.parametrize(
    "novlap,nfft,Ns,nblocks",
    [
        # hop = nfft; Ns is a whole multiple of hop
        (0, 16, 64, 3),
        # hop = 12; Ns=70 is not a multiple of hop (remainder dropped)
        (4, 16, 70, 5),
    ],
    ids=["ovl0", "ovl4-uneven"],
)
def test_blocksfft_serial_parallel_param_surface(
    window_type, window_norm, blockwise_mean, normvar, novlap, nfft, Ns, nblocks
):
    """Serial == parallel across window_norm × blockwise_mean × normvar × novlap.

    At least one case uses an uneven record (id token ``uneven``): Ns is not a
    whole multiple of the block hop, so placement cannot hide behind a tidy
    partition. The two paths agree to atol=1e-12 with rtol=0; flipping any one
    of these options moves the output by ~1e-1, so the margin is real.
    """
    # Without this, a failed `openmodalpy.core.parallel` import would send both
    # calls down the serial body (base.py sets PARALLEL_AVAILABLE=False on
    # ImportError) and every case below would pass while comparing serial to
    # itself. A broken parallel stack should be loud, not silently vacuous.
    assert PARALLEL_AVAILABLE, "optimized path unavailable: this test would compare serial to serial"
    rng = np.random.default_rng(0)
    q = rng.standard_normal((Ns, 3))
    kwargs = dict(
        nfft=nfft,
        nblocks=nblocks,
        novlap=novlap,
        window_type=window_type,
        window_norm=window_norm,
        blockwise_mean=blockwise_mean,
        normvar=normvar,
    )
    ser = blocksfft(q, use_parallel=False, **kwargs)
    par = blocksfft(q, use_parallel=True, **kwargs)
    max_diff = float(np.max(np.abs(ser - par)))
    assert np.allclose(ser, par, rtol=0, atol=1e-12), (
        f"serial/parallel disagree: window={window_type!r} window_norm={window_norm!r} "
        f"blockwise_mean={blockwise_mean} normvar={normvar} novlap={novlap} Ns={Ns}: "
        f"max|diff|={max_diff:.3e}"
    )


def test_normvar_zero_variance_channel_isfinite():
    """Constant (zero-variance) channel is clamped, not Inf/NaN, under normvar.

    After mean subtraction a constant channel has variance 0; the 4*eps clamp
    sets the divisor to 1 so the channel matches the no-normvar path.
    """
    rng = np.random.default_rng(1)
    nfft = 32
    varying = rng.standard_normal(nfft)
    constant = np.full(nfft, 3.0)
    q = np.column_stack([varying, constant])

    with_nv = blocksfft(q, nfft=nfft, nblocks=1, novlap=0, normvar=True, use_parallel=False)
    without = blocksfft(q, nfft=nfft, nblocks=1, novlap=0, normvar=False, use_parallel=False)

    assert np.all(np.isfinite(with_nv)), "normvar produced non-finite values on a constant channel"
    # Constant channel: clamp divisor to 1 → identical to no-normvar path.
    np.testing.assert_allclose(with_nv[:, 1, :], without[:, 1, :], rtol=0, atol=1e-12)
    # Varying channel must actually change under normvar (else the flag is a no-op).
    assert not np.allclose(with_nv[:, 0, :], without[:, 0, :])


# ---------------------------------------------------------------------------
# Welch block partitioning (floor, drop remainder — no end-clamp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Ns,nfft,overlap,label",
    [
        (100, 128, 0.5, "Ns < nfft"),
        (64, 128, 0.5, "Ns <= novlap"),
    ],
)
@pytest.mark.parametrize(
    "fn",
    [lambda *a, **k: blocksfft(*a, use_parallel=False, **k), blocksfft_optimized],
    ids=["serial", "parallel"],
)
def test_short_record_raises_naming_Ns_and_nfft(Ns, nfft, overlap, label, fn):
    """Short records must raise ValueError naming both Ns and nfft.

    Pre-fix: ceil/floor of a short record could yield nblocks=0 (silent empty)
    or a negative start (numpy wrap → garbage). Assert on the exception message,
    not on an output shape.
    """
    nb, novlap = _nblocks_floor(Ns, nfft, overlap)
    q = np.zeros((Ns, 4))
    with pytest.raises(ValueError) as ei:
        fn(q, nfft, max(nb, 1), novlap)
    msg = str(ei.value)
    assert str(Ns) in msg and str(nfft) in msg, f"{label}: message {msg!r} must name Ns and nfft"


@pytest.mark.parametrize("Ns,nfft,overlap", [(500, 128, 0.5), (300, 128, 0.5)])
@pytest.mark.parametrize(
    "fn",
    [lambda *a, **k: blocksfft(*a, use_parallel=False, **k), blocksfft_optimized],
    ids=["serial", "parallel"],
)
def test_oversize_nblocks_raises_not_clamp(Ns, nfft, overlap, fn):
    """Old ceil nblocks that no longer fit must raise, not silently clamp."""
    nb_floor, novlap = _nblocks_floor(Ns, nfft, overlap)
    nb_ceil = int(np.ceil((Ns - novlap) / (nfft - novlap)))
    if nb_ceil == nb_floor:
        pytest.skip("ceil and floor agree; clamp path not exercised")
    q = np.zeros((Ns, 4))
    with pytest.raises(ValueError) as ei:
        fn(q, nfft, nb_ceil, novlap)
    msg = str(ei.value)
    assert str(Ns) in msg or str(nb_ceil) in msg, msg


@pytest.mark.parametrize(
    "Ns,nfft,overlap",
    [
        (500, 128, 0.5),  # shipped cylinder_wake shape: 6 blocks, remainder dropped
        (300, 128, 0.5),  # does not divide evenly
        (500, 100, 0.5),
        (256, 64, 0.25),
    ],
)
def test_block_starts_strict_hop_and_fit(Ns, nfft, overlap):
    """Block starts must be strictly increasing with constant hop, last fits.

    Asserted on the indices themselves (not merely on nblocks), via an impulse
    basis: column j is an impulse at time j; energy in block k recovers coverage.
    boxcar is required so edge impulses are not window-attenuated below threshold.
    """
    nb, novlap = _nblocks_floor(Ns, nfft, overlap)
    hop = nfft - novlap
    expected_starts = _block_starts(nb, nfft, novlap)
    assert expected_starts[-1] + nfft <= Ns

    def recovered_starts(fn):
        qhat = fn(np.eye(Ns), nfft, nb, novlap, window_type="boxcar")
        energy = np.sum(np.abs(qhat) ** 2, axis=0)  # [time, block]
        starts = []
        for k in range(nb):
            idx = np.flatnonzero(energy[:, k] > 0.5 * energy[:, k].max())
            assert idx.size > 0
            starts.append(int(idx[0]))
        return starts

    for fn, name in (
        (lambda *a, **k: blocksfft(*a, use_parallel=False, **k), "serial"),
        (blocksfft_optimized, "parallel"),
    ):
        starts = recovered_starts(fn)
        assert starts == expected_starts, f"{name}: starts {starts} != {expected_starts}"
        assert all(s2 - s1 == hop for s1, s2 in zip(starts, starts[1:])), f"{name}: non-constant hop"
        assert starts[-1] + nfft <= Ns


def test_serial_parallel_identical_placement_uneven_records():
    """Serial and parallel paths place blocks identically, including uneven Ns."""
    for Ns, nfft, overlap in ((500, 128, 0.5), (300, 128, 0.5), (256, 64, 0.25)):
        nb, novlap = _nblocks_floor(Ns, nfft, overlap)
        rng = np.random.default_rng(Ns)
        q = rng.standard_normal((Ns, 3))
        ser = blocksfft(q, nfft, nb, novlap, use_parallel=False, window_type="boxcar")
        par = blocksfft_optimized(q, nfft, nb, novlap, window_type="boxcar")
        np.testing.assert_allclose(ser, par, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "fn",
    [lambda *a, **k: blocksfft(*a, use_parallel=False, **k), blocksfft_optimized],
    ids=["serial", "parallel"],
)
@pytest.mark.parametrize(
    "novlap",
    [64, 65],  # == nfft and > nfft; both yield hop <= 0
    ids=["novlap_eq_nfft", "novlap_gt_nfft"],
)
def test_novlap_ge_nfft_raises_not_repeat_block0(fn, novlap):
    """hop <= 0 must raise: otherwise every block starts at 0 (identical members)."""
    q = np.zeros((300, 4))
    with pytest.raises(ValueError, match=r"hop|novlap|nfft"):
        fn(q, 64, 3, novlap)


def test_apply_snapshot_limit_uses_floor_nblocks():
    """commands._apply_snapshot_limit must set floor nblocks and stay runnable.

    Slice-2 regression: ceil after max_snapshots truncation requested more
    blocks than fit (Ns=400, nfft=128, overlap=0.5 → floor 5, ceil 6 needs 448).
    Pinning nblocks alone is not enough — blocksfft must accept the result.
    """
    import types

    from openmodalpy.commands import _apply_snapshot_limit

    nfft, novlap, limit = 128, 64, 400
    expect = (limit - novlap) // (nfft - novlap)
    assert expect == 5
    # Old ceil would have been 6 and needed (6-1)*64+128 = 448 > 400.
    assert int(np.ceil((limit - novlap) / (nfft - novlap))) == 6

    an = types.SimpleNamespace(
        data={"q": np.zeros((500, 4)), "Ns": 500},
        novlap=novlap,
        nfft=nfft,
        nblocks=6,
    )
    spec = types.SimpleNamespace(params={"max_snapshots": limit})
    _apply_snapshot_limit(an, spec)

    assert an.data["Ns"] == limit
    assert an.data["q"].shape[0] == limit
    assert an.nblocks == expect

    # Must actually be usable: the original break was a ValueError here.
    out = blocksfft(an.data["q"], nfft, an.nblocks, novlap, use_parallel=False)
    assert out.shape[2] == expect


def test_load_and_preprocess_nblocks_uses_floor(tmp_path):
    """BaseAnalyzer.load_and_preprocess must use floor nblocks, not ceil.

    Slice-1 tests call blocksfft with a precomputed nblocks, so the analyzer's
    own formula was untested. Ns=400, nfft=128, overlap=0.5 → floor 5, ceil 6.
    """
    from openmodalpy import SPODAnalyzer

    Ns, nfft, overlap = 400, 128, 0.5
    novlap = int(overlap * nfft)
    expect_floor = (Ns - novlap) // (nfft - novlap)
    expect_ceil = int(np.ceil((Ns - novlap) / (nfft - novlap)))
    assert expect_floor == 5 and expect_ceil == 6

    q = np.zeros((Ns, 4))
    data = {
        "q": q,
        "x": np.linspace(0, 1, 4),
        "y": np.linspace(0, 1, 1),
        "dt": 1.0,
        "Nx": 4,
        "Ny": 1,
        "Ns": Ns,
    }
    analyzer = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=nfft,
        overlap=overlap,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    assert analyzer.nblocks == expect_floor


def test_load_and_preprocess_short_record_raises(tmp_path):
    """Truncated-to-shorter-than-nfft record must raise a clear ValueError."""
    from openmodalpy import SPODAnalyzer

    Ns, nfft, overlap = 100, 128, 0.5
    q = np.zeros((Ns, 4))
    data = {
        "q": q,
        "x": np.linspace(0, 1, 4),
        "y": np.linspace(0, 1, 1),
        "dt": 1.0,
        "Nx": 4,
        "Ny": 1,
        "Ns": Ns,
    }
    analyzer = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=nfft,
        overlap=overlap,
        results_dir=str(tmp_path),
        figures_dir=str(tmp_path),
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    with pytest.raises(ValueError) as ei:
        analyzer.load_and_preprocess()
    msg = str(ei.value)
    assert str(Ns) in msg and str(nfft) in msg


def test_white_noise_spod_eigenvalue_matches_analytic():
    """Unit-variance white noise: mean mid-band SPOD eigenvalue ≈ 1.

    Derivation (single spatial point, W=1, boxcar, dt=1 so fs=1, dst=df=1/nfft):
    blocksfft stores q_hat = FFT(x)/nfft. For unit-variance white noise,
    E[|X[k]/nfft|^2] = 1/nfft at each one-sided bin (numpy unnormalized FFT).
    spod_function forms λ = mean_b |q_hat_b|^2 / dst = mean |q_hat|^2 * nfft,
    so E[λ(f)] = 1 at every interior frequency.

    This pins the FFT → SPOD normalization chain (window, 1/nfft, dst), not
    block placement. On white noise a clamped/reused block does not bias
    E[λ]; it only changes the variance of the mean. Partitioning is covered
    by the impulse-probe and oversize-nblocks tests above.

    Tolerance: an ensemble of nblocks approximately independent periodogram
    members has relative standard error ~ 1/sqrt(nblocks) for the mean over
    frequencies of comparable width. We average over ~nfft/2 mid-band bins and
    require |mean(λ) - 1| < 4/sqrt(nblocks). Tighter than ~1/sqrt(nblocks) is
    flaky under finite-sample noise.
    """
    rng = np.random.default_rng(0)
    Ns, nfft, overlap = 4096, 128, 0.5
    nb, novlap = _nblocks_floor(Ns, nfft, overlap)
    assert nb >= 16
    q = rng.standard_normal((Ns, 1))
    qhat = blocksfft(q, nfft, nb, novlap, window_type="boxcar", use_parallel=False)
    dst = 1.0 / nfft
    w = np.ones((1, 1))
    lams = []
    for ifreq in range(1, qhat.shape[0] - 1):  # skip DC and Nyquist
        _, lam = spod_function(qhat[ifreq], nblocks=nb, dst=dst, w=w, use_parallel=False)
        lams.append(lam[0])
    mean_lam = float(np.mean(lams))
    tol = 4.0 / np.sqrt(nb)
    assert abs(mean_lam - 1.0) < tol, (
        f"mean mid-band SPOD λ={mean_lam:.4f} vs analytic 1.0 (tol={tol:.4f} = 4/sqrt(nblocks={nb}))"
    )
