# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- POD, mPOD, ST-POD and PSD-POD now share one lift / metric / second-order
  seam in `core/decomposition.py` (`IdentityLift`, `DelayEmbeddingLift`,
  `BandFilteredLift`, `SpatialMetric`, `weighted_second_order`). Results are
  unchanged; each caller keeps its own truncation policy via
  `drop_nonpositive` / `n_keep`.
- PSD-POD now floors its spatial weights at `1e-12` before taking the square
  root, matching POD, mPOD and ST-POD. Results are unaffected for strictly
  positive weights. Known regression at the edges: a zero-weight point
  previously contributed exactly nothing, and a **negative** weight previously
  aborted the solve (`LinAlgError` out of `eigh`) — both are now accepted
  silently. A negative entry in the metric means the inner product is not an
  inner product, so this should raise; making a zero-measure or negative metric
  fail loudly is tracked separately and applies to all four analyzers, not only
  PSD-POD.

### Fixed
- PSD-POD result metadata now records `uses_mean_subtraction=True`, matching
  `blocksfft` (which always removes a mean — global by default, per-block when
  `blockwise_mean` is set). The previous write stored `False`.
- SPOD `load_and_preprocess` docstring no longer claims parent mean subtraction
  or a `self.data_matrix` attribute that is never assigned.
- DOC.md: `bsmd.py` filename, POD branch condition `Ns < Nspace` (no false
  `<<` margin), dropped a stale hardcoded test count, and notes that DMD neither
  centers nor applies the spatial metric.

### Changed
- Welch block partitioning now matches `scipy.signal.welch`: `nblocks` is
  computed with floor arithmetic and the remainder is dropped, rather than
  ceil plus a clamped final block that re-uses samples. Records that do not
  divide evenly therefore change block count (the shipped cylinder_wake
  example with `Ns=500`, `nfft=128`, `overlap=0.5` goes from 7 blocks to 6),
  so SPOD / PSD-POD / BSMD numbers on those records move. Short records that
  cannot form one full block, and callers that request more blocks than fit,
  now raise `ValueError` instead of returning empty or wrapped indices.
  The same floor helper (`welch_nblocks`) is used by
  `BaseAnalyzer.load_and_preprocess` and by `commands._apply_snapshot_limit`
  after `max_snapshots` truncation — the snapshot-limit path previously
  recomputed `nblocks` with ceil and could request more blocks than fit
  (e.g. Ns=400, nfft=128, overlap=0.5 → floor 5 vs ceil 6). `novlap >= nfft`
  (hop ≤ 0) is rejected in both FFT paths instead of repeating block 0.

### Fixed
- BSMD static-triad validation bounds by both `nfft//2` and the loaded `qhat`
  length, and no longer swallows out-of-range bin reads into a silent NaN
  eigenvalue. The two bounds coincide for a freshly computed transform; when
  they diverge, the triad is now rejected with a `ValueError` naming the real
  bound instead of returning `NaN` with no diagnostic.
- POD energy-captured report no longer always prints 100%: the fraction is
  truncated eigenvalue sum over the pre-truncation total, stored as
  `energy_captured_fraction` on the analyzer and in result metadata.
- ARPACK-path SVD (`compute_reduced_svd` with `min_dim >= 256`) is bit-reproducible
  via a deterministic local start vector. Of the synthetic generators, only the
  cylinder wake accepts a `seed` and records it into result metadata as
  `data_seed`; the JetLES-like dummy generator accepts a `seed` for its noise RNG
  but does not surface it; `double_gyre` and `taylor_green` are deterministic and
  take no seed. Tests reseed NumPy from `OMPY_TEST_RNG_JITTER` so collection
  order cannot leak unseeded draws.

### Changed
- DMD operator rank is now a separate constructor/`CaseSpec` parameter `rank`.
  With an **explicit** `rank`, `n_modes_save` only bounds how many modes are kept
  after sorting and no longer moves eigenvalues. The **default is unchanged**:
  `rank=None` still resolves to `min(n_modes_save, min(X1.shape))` then the
  relative singular-value floor (`s_j > rcond * s[0]`) — bit-for-bit today's
  behaviour — and emits a `DeprecationWarning` naming `rank`. Pass `rank`
  explicitly to silence the warning and pin the numerics. Opt-in criteria:
  positive `int`, `"svht"` (Gavish–Donoho optimal hard threshold, unknown-noise
  form), and `"energy"` (cumulative `s²` fraction, default
  `energy_fraction=0.999`). No published number moves under the default path.
  Full numerical rank was **rejected** as the new default: on the shipped
  cylinder wake the singular spectrum decays smoothly and never reaches the
  machine floor, so untruncated DMD surfaces spurious `|λ| > 1` modes that
  outrank the physical shedding frequency.

### Fixed
- DMD no longer amplifies noise into modes when the snapshot pair is ill-conditioned.
  The reduced operator and the mode recovery both divide by the singular values of the
  first snapshot matrix, and the number kept was whatever you asked for rather than
  whatever the data supports. A rank-deficient sequence alone is harmless — the small
  singular values cancel — but as soon as the second snapshot matrix carries content the
  first one cannot represent, which is what a transient, an arriving structure or a
  truncated record produces, the division has nothing to cancel against. On a rank-3
  sequence with a perturbation applied to the final snapshot this returned eigenvalues of
  magnitude 6.7e9 and modes of magnitude 1.9e9, all finite, so nothing raised. Singular
  values are now kept only above a threshold relative to the largest one, following the
  `numpy.linalg.pinv` convention, which makes the cut invariant to the overall scale of
  the data; both `pinv` calls pass that same conditioning explicitly instead of relying
  on a default. Well-conditioned data is unaffected.
- DMD reports the rank it actually used as `effective_rank`, and warns when that is below
  the number of modes requested. Asking for more modes than the data supports is normal,
  so this is a `RuntimeWarning` about the data rather than an error.
- An all-zero or otherwise degenerate field returns empty results with that warning
  instead of failing inside the eigensolver with `LinAlgError: Array must not contain
  infs or NaNs`.

### Added
- The three built-in synthetic generators are now checked against the closed-form
  answers they already carry. `example_data.py` has always returned the double gyre's
  forcing frequency, the Taylor-Green decay eigenvalue and the cylinder wake's Strouhal
  number alongside the data, but nothing read them, and two of the three generators were
  not exercised by any test. Each is now run through the analyzer that should recover its
  quantity, comparing against the value read from the generator's own metadata rather
  than a constant copied into the test, so the check follows the generator if its physics
  changes. Tolerances are computed from the discretization: machine precision for
  Taylor-Green, where the field is rank-1 in space times a pure exponential and DMD
  recovers the multiplier exactly; a tenth of the Rayleigh frequency for DMD, which is
  not bin-limited; half an FFT bin for SPOD, which is. A cross-analyzer check pins that
  DMD and SPOD agree on the shedding frequency without reference to the metadata.

### Changed
- Four simplifications the code made silently are now written down where a user would
  look for them. `spatial_weight_type="uniform"` returns ones rather than cell volumes,
  so reported POD/SPOD energy is a sum over mesh points, not a domain integral, and its
  numerical value changes when the grid is refined. mPOD decomposes each band
  independently and then concatenates and re-sorts the modes with no joint
  orthonormalization, so the pooled mode matrix is not a W-orthonormal basis even though
  each band's modes are; measured on a three-band case, cross-band inner products reach
  0.5 while within-band ones sit at 1e-16. SPOD's `dst` is a Strouhal step,
  `St[1] - St[0] = df·L/U`, not the frequency resolution `fs/nfft`, so the characteristic
  length and velocity rescale the reported eigenvalues; the two coincide only at the
  default `L = U = 1`, which is why the shipped generators never revealed it. The default
  BSMD triad table covers frequency-bin indices up to `|p| = 8`, which at the default
  `nfft=128` is the bottom 12.5% of the spectrum. A docstring in `core/base.py` that
  claimed the opposite about `dst` has been corrected. None of the underlying arithmetic
  changed; only what the project says about it.
- The bispectral energy map no longer silently discards triads. Its grid was a fixed
  17×17 centred on `|p| = 8`, so a triad outside that window was computed and then
  dropped from the map without a word — with `nfft=32`, where 16 bins are available, a
  triad at `p=12` vanished. The half-width is now derived from the triads actually
  analysed, and the plot extent follows it. The default triad list still produces the
  same 17×17 grid with the same values.
- BSMD now rejects input it cannot analyse instead of returning something plausible.
  A triad component outside the available rfft bins (`|p| > nfft//2`) raises `ValueError`
  naming the offending index and the bin count, where it previously produced a NaN
  eigenvalue; the last real bin, `|p| = nfft//2`, is still accepted. Dynamic triad
  selection (`use_static_triads=False`) raises `NotImplementedError` instead of printing
  a notice and returning empty arrays. Note the consequence for small transforms: the
  default triad table reaches `|p| = 8`, so `nfft < 16` combined with the default triads
  now raises rather than filling the high-index rows with NaN.
- The validation suite now enforces its claims. `tests/test_all.py` describes itself as
  validating mathematical correctness against known analytical solutions, but every one
  of its 22 checks reported through a helper that printed a tick and appended to a list;
  the pass/fail decision lived in a `main()` reachable only by running the file as a
  script, while CI runs pytest. Under pytest the five tests passed unconditionally, and
  each was wrapped in a bare `except Exception`, so a crash inside POD, DMD or SPOD was
  still reported as a pass. All 22 checks are now plain assertions with the measured
  value in the failure message, at their original tolerances, and analyzer output is
  routed to pytest's `tmp_path` instead of a `./results` directory in the working tree.
  The conversion was verified by mutation, not by the suite going green: perturbing POD
  eigenvalues by 5%, DMD eigenvalues by 2%, or shifting the SPOD spectrum by four
  frequency bins each turns the suite red.

### Fixed
- A corrupt or truncated FFT-block cache no longer aborts the analysis. Interrupted
  runs, full disks, and killed jobs can leave a half-written HDF5 cache that still
  exists on disk; opening it for append used to raise and stop SPOD or BSMD even though
  the blocks are re-derivable from the raw data. Write mode is now chosen by whether the
  file is actually readable as HDF5, not by whether it exists, so an unreadable cache is
  overwritten after a recompute. The same recovery applies when BSMD tries to reuse a
  SPOD cache that turns out to be truncated: it prints a reason and recomputes rather
  than raising. Reading a saved results file keeps the opposite policy and still raises,
  since results are not re-derivable from the raw data the way FFT blocks are.
- The sampling rate `fs` fails with a diagnosis instead of an accident. `fs` starts at
  `0.0` until a dataset is loaded, and on paths that never load one — reopening saved
  results, for instance — that zero used to reach the frequency code, where a periodogram
  rejected it with a message naming nothing and an `rfftfreq` axis raised
  `ZeroDivisionError`. Both now raise a single `ValueError` naming the data source and
  saying what to supply, matching the message the timestep already used. Frequency axes
  are unchanged whenever the sampling rate is valid.

## [0.3.0] - 2026-07-27

### Breaking
- **The BSMD cross-bispectral matrix now conjugates the sum-frequency term.**
  Earlier releases formed `E[X(f1) X(f2) X(f1+f2)]`, a plain third-order moment.
  The bispectrum takes the conjugate of the sum-frequency component, so the matrix
  is now built as `B = Q_{k+l}^H W (Q_k ∘ Q_l) / N_blk`, where `∘` is the
  elementwise product, with the modes read from the right eigenvector.

  **BSMD results produced by 0.1.0 or 0.2.0 are invalid and must be recomputed.**
  This is not a question of precision. Without the conjugate the phases of the
  three components never cancel, so the eigenvalue reduces to a random walk of
  size `1/sqrt(N_blk)` instead of converging to the bispectral amplitude.
  Eigenvalues, modes, energy maps and every figure derived from them are affected.
  One case is unaffected: for the `(k, -k, 0)` triads the sum frequency falls on
  the DC bin, which is real for a real field, so conjugating it changes nothing and
  those results were correct all along.

  `load_results` refuses an HDF5 file written before the fix, so an old results
  file raises with the reason instead of feeding stale eigenvalues into a new
  analysis in silence.
- **Window convention for blocked FFT / SPOD:** both serial and parallel
  `blocksfft` paths now use the PERIODIC window from
  `scipy.signal.get_window(..., fftbins=True)`. The optimized path previously
  ignored `window_type` other than `"sine"` (falling back to a SYMMETRIC
  `np.hamming`) and silently substituted Hamming for `hann`/`blackman`/etc.
  Default SPOD spectra therefore change for existing users; serial and parallel
  results now agree bit-for-bit for all supported window names.
- An unrecognised `window_type` now raises instead of quietly falling back to
  Hamming. Names the optimized path used to accept by accident — `hanning`, any
  capitalised spelling, and outright typos — are rejected, since only the exact
  names `scipy.signal.get_window` knows (plus `sine`) are valid.
- **Loaders no longer invent a timestep, and neither do the physical quantities that
  consume one.** A `.mat` file with no `dt` used to be given `dt = 1.0` by the loader
  before validation ever saw it, so the check added below passed on a fabricated value.
  The `.mat` loader now leaves `dt` unset, and `_infer_dt_from_times` returns nothing
  rather than `1.0` when the time vector has fewer than two samples or is constant.
  Every physics-bearing consumer — the DMD continuous-time eigenvalues
  `ω = log(λ)/dt`, the mPOD Nyquist frequency that resolves band edges, and the SPOD
  sampling rate restored on reload — now goes through one validated accessor and raises
  when the timestep is missing, zero, negative or non-finite. Reloading results from an
  HDF5 file that carries no `dt` attribute raises instead of reporting growth rates
  computed at a unit timestep. The mPOD case is the one worth re-checking in existing
  work: a wrong `dt` there moved the band edges, so different modes landed in different
  bands rather than merely being mislabelled.
- **Plots stop labelling an axis in seconds when no timestep is known.** The
  time-coefficient plots for POD, ST-POD and DMD used to build their abscissa as
  `arange(n) * 1.0` and label it `Time` regardless. They now fall back to sample
  indices labelled `Sample index`, and use an explicit time vector or a real `dt`
  when one exists. `DMDAnalyzer.plot_eigenspectra` raises without a usable `dt`,
  since frequency and growth rate are that figure's two axes. The DMD mode-field
  plots keep drawing and simply omit the `f=...` fragment from their titles — the
  mode shapes never depended on the timestep.
- A missing or unusable timestep now raises `ValueError` instead of defaulting to
  `0.1` and continuing. This covers `dt` zero, negative or non-finite, as well as
  absent, `None`, or non-scalar — all of which previously either fabricated a
  timestep or leaked a `KeyError`/`TypeError`. Frequency axes, Strouhal numbers,
  and DMD growth rates require a real positive `dt` from the data or loader.

### Changed
- Relicense from MIT to Apache-2.0, effective from 0.3.0 onward. The 0.1.0 and
  0.2.0 releases remain under MIT.

### Fixed
- The blocked-FFT cache is validated against the parameters that produced it.
  Each cache file now records the window type, window normalization, overlap,
  `nfft`, the block preprocessing flags and a digest of the input array, and a
  cached result is reused only when all of them match. The cache was previously
  keyed on the result filename alone, so changing only `window_type` reused blocks
  computed under the old window, and two datasets sharing a data root, `nfft`,
  overlap and snapshot count could serve each other's blocks. Cache files written
  by earlier versions carry no stamp and are recomputed once.

## [0.2.0] - 2026-07-25

### Changed
- **Breaking:** the import name is now `openmodalpy`, matching the PyPI
  distribution name. Update `import modalpy` / `from modalpy import ...` to
  `import openmodalpy` / `from openmodalpy import ...`.
- **Breaking:** the console script is renamed from `modalpy` to `openmodalpy`
  (e.g. `openmodalpy run --config analysis.jsonc`).

## [0.1.0] - 2026-07-24

First public release, distributed on PyPI as `openmodalpy`.

### Added
- Modal decomposition analyzers: POD, mPOD, PSD-POD, SPOD, ST-POD, DMD (LS/TLS),
  HODMD (LS/TLS) and BSMD.
- Configuration-driven workflow: a single JSONC file runs several methods over one
  dataset, via `openmodalpy run --config`.
- Command line interface: `analyze`, `run`, `methods`, `examples`, `results`.
- Data loaders for `.mat` and `.npz` inputs, plus support for user-supplied loaders.
- Bundled self-contained example configs backed by analytic generators
  (`double_gyre`, `cylinder_wake`, `taylor_green`, `run_benchmarks`).

### Changed
- FFT backend dispatch moved out of this package into
  [`fftkit`](https://github.com/openfluids/fftkit), now a required dependency.
  `openmodalpy.core.config.FFT_BACKEND` re-exports the backend fftkit resolves, so the
  reported backend always matches the one actually used.
- The `mkl` and `gpu` extras now defer to `fftkit[mkl]` and `fftkit[gpu]`.

### Removed
- The bundled `openmodalpy.fft` subpackage. Import FFT helpers from `fftkit` instead:
  `get_fft_func`, `periodogram_rfft`, `find_peaks` and related functions.

[Unreleased]: https://github.com/openfluids/openmodalpy/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/openfluids/openmodalpy/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/openfluids/openmodalpy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/openfluids/openmodalpy/releases/tag/v0.1.0
