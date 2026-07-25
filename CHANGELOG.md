# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking
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

### Changed
- Relicense from MIT to Apache-2.0, effective from 0.3.0 onward. The 0.1.0 and
  0.2.0 releases remain under MIT.

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

[Unreleased]: https://github.com/openfluids/openmodalpy/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/openfluids/openmodalpy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/openfluids/openmodalpy/releases/tag/v0.1.0
