# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
