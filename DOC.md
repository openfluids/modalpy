# OpenModalPy — Technical Reference

This document is the single reference for humans and LLMs working with the
OpenModalPy codebase. It covers architecture, every supported method, the data
contract, the configuration system, the CLI, and extension paths.

---

## Architecture

```
src/openmodalpy/
├── __init__.py          # public exports: PODAnalyzer, MPODAnalyzer, DMDAnalyzer,
│                        #   SPODAnalyzer, BSMDAnalyzer, STPODAnalyzer
├── core/
│   ├── base.py          # BaseAnalyzer, compute_reduced_svd, blocksfft,
│   │                    #   spod_function, weight calculation, plot helpers
│   ├── io.py            # MATDataLoader, DNamiDataLoader, _slice_block_in_time
│   ├── config.py        # FFT_BACKEND, FIG_DPI, directory defaults
│   └── parallel.py      # thread-pool FFT + SPOD acceleration
├── pod.py               # PODAnalyzer (variance-optimal, identity lift)
├── mpod.py              # MPODAnalyzer (band-filtered POD)
├── spod.py              # SPODAnalyzer (frequency-by-frequency POD)
├── dmd.py               # DMDAnalyzer (LS/TLS, delay embedding, HODMD)
├── bmsd.py              # BSMDAnalyzer (triadic bispectral decomposition)
├── stpod.py             # STPODAnalyzer (delay-embedded POD via Hankel lift)
├── commands.py          # dispatch core: analyze_from_spec, _run_pod_like,
│                        #   _run_dmd, _run_spod, _run_bsmd, _run_psd_pod,
│                        #   PSD-POD implementation, example discovery
├── cli.py               # argparse frontend: analyze, run, methods, examples, results
├── config_io.py         # load_jsonc, resolve_path, strip_jsonc_comments
├── specs.py             # DataSourceSpec, CaseSpec, AnalyzeSpec, RunOutcome, etc.
├── example_data.py      # built-in generators: double_gyre, taylor_green, cylinder_wake
└── examples/            # packaged .jsonc configs shipped in the wheel
```

FFT backend dispatch lives in the external [`fftkit`](https://github.com/openfluids/fftkit)
package: `get_fft_func()` selects among scipy/numpy/mkl/cupy/accelerate, and
`core.config.FFT_BACKEND` re-exports the backend fftkit resolved. Override it with the
`FFTKIT_BACKEND` environment variable.

### Analyzer lifecycle

Every analyzer follows the same sequence:

1. **Construct** — pass `file_path`, loader, weight type, method params
2. **`load_and_preprocess()`** — load data → compute spatial weights → set derived params
3. **Method-specific computation** — `perform_pod()`, `perform_dmd()`, `perform_spod()`, etc.
4. **`save_results()`** — write HDF5 with modes, eigenvalues, metadata
5. **Plot** — `plot_eigenvalues()`, `plot_modes()`, etc.

The `commands.py` dispatch core (`analyze_from_spec`) automates steps 1–5 from
a single `AnalyzeSpec` dataclass, which is built from a JSONC config file.

### Design principle

All analyzers share:
- One **data contract** (snapshot matrix Q, coordinates, dt, spatial weights W)
- One **metric layer** (W defines the inner product for orthogonality and energy)
- One **lift** concept (the method-specific transformation of raw snapshots)

They differ only in the **operator problem** solved on the lifted data:
- Variance-optimal (eigendecomposition of weighted covariance/kernel)
- Evolution-fit (SVD-based regression on paired snapshots)
- Triadic interaction (cross-bispectral coupling optimization)

**Limitation — uniform W is not a domain integral.** With
`spatial_weight_type="uniform"`, W is the all-ones vector, not cell volumes or
grid spacing. Reported "energy" is therefore a **sum over mesh points**, not a
domain integral, and is **mesh-resolution dependent**: refining the grid changes
the numerical value even when the continuum field is unchanged. Comparing
energies across resolutions requires care (or explicit cell-volume weights).

---

## Data Contract

Every analyzer expects a Python dict with these keys:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `q` | `ndarray (Ns, Nspace)` | yes | snapshot matrix (time × flattened spatial) |
| `x` | `ndarray` | yes | x-coordinates (1D or 2D mesh) |
| `y` | `ndarray` | yes | y-coordinates (1D or 2D mesh) |
| `z` | `ndarray` or `None` | no | z-coordinates for 3D data |
| `dt` | `float` | yes | time step between snapshots |
| `Nx` | `int` | yes | grid points in x |
| `Ny` | `int` | yes | grid points in y |
| `Nz` | `int` | no | grid points in z (default 1) |
| `Ns` | `int` | yes | number of snapshots |
| `t` | `ndarray` | no | time vector |
| `metadata` | `dict` | no | format info, var_name, plot_style, etc. |

### Supported input formats

- **MATLAB `.mat`** — auto-detected via `MATDataLoader`
- **NumPy `.npz`** — consolidated or split layouts via `DNamiDataLoader`
- **Custom loader** — any callable `(file_path: str) -> dict`

### Spatial weights

| Type | When to use |
|------|------------|
| `"uniform"` | Cartesian grids, single-component data |
| `"polar"` | Cylindrical grids (jet nozzle coordinates) |
| `"auto"` | Auto-detect from data/filename |

`"uniform"` currently returns ones (`calculate_uniform_weights`); it does **not**
apply Δx·Δy cell volumes. Polar weights do use radial spacing. Until cell-volume
weights exist for Cartesian grids, treat POD/SPOD eigenvalues under uniform W as
mesh-dependent sums, not resolution-invariant energies.

---

## Supported Methods

### 1. POD — Proper Orthogonal Decomposition

**Class:** `PODAnalyzer` · **Lift:** identity on centered snapshots · **Operator:** covariance kernel eigenproblem

```python
from openmodalpy import PODAnalyzer

pod = PODAnalyzer(file_path="data.mat", n_modes_save=10)
pod.run_analysis()
# pod.modes          — (Nspace, n_modes)
# pod.eigenvalues    — (n_modes,)
# pod.time_coefficients — (Ns, n_modes)
```

**Key facts:**
- Modes are W-orthogonal, ranked by captured weighted variance
- Uses method of snapshots (kernel eigenproblem) when Ns << Nspace
- Mean is subtracted before decomposition

### 2. mPOD — Multiscale POD

**Class:** `MPODAnalyzer` · **Lift:** temporal band filtering · **Operator:** POD per band

```python
from openmodalpy import MPODAnalyzer

mpod = MPODAnalyzer(
    file_path="data.mat", n_modes_save=10,
    band_edges=[0.0, 0.15, 0.35, 1.0],  # normalized Nyquist
    band_scale="normalized_nyquist",
)
mpod.run_analysis()
```

**Key facts:**
- Partitions temporal frequency axis into non-overlapping bands
- Current implementation uses rectangular (brick-wall) filters
- Modes live in the same space as POD modes but are scale-separated
- Reference: Mendez et al. (2019), JFM 870

**Limitation — POD-per-band pool, not Mendez MRA.** Each band is POD'd
independently; band modes are concatenated and re-sorted by eigenvalue with no
joint W-orthonormalization. Modes from different bands are not orthonormal
across bands (`Φᵀ W Φ ≠ I` for the pooled set): cross-band inner products are
generally nonzero. Within a single band, POD orthonormality still holds. This is
a simplification relative to the Mendez et al. multiresolution construction; do
not treat the full mode matrix as a W-orthonormal basis.

### 3. PSD-POD — Power-Spectral-Density POD

**Implementation:** in `commands.py` (no separate analyzer class)

**Lift:** pooled blockwise Fourier realizations across all frequencies and blocks

**Key facts:**
- Uses same Welch-block preprocessing as SPOD
- Solves one global eigenproblem instead of per-frequency
- Captures broadband coherent structures
- Triggered via `method="psd-pod"` in config

### 4. SPOD — Spectral POD

**Class:** `SPODAnalyzer` · **Lift:** blockwise Fourier transform · **Operator:** per-frequency covariance eigenproblem

```python
from openmodalpy import SPODAnalyzer

spod = SPODAnalyzer(file_path="data.mat", nfft=256, overlap=0.5)
spod.run_analysis()
# spod.eigenvalues — (n_freq, n_blocks)
# spod.modes       — (n_freq, Nspace, n_blocks)
# spod.St          — Strouhal number array
# spod.freq        — frequency array (Hz)
```

**Key facts:**
- Frequency-by-frequency POD of Welch-block spectral ensemble
- Block length controls bias-variance tradeoff of spectral estimator
- Caches FFT blocks in HDF5 for reuse
- Strouhal normalization via `characteristic_length` and `characteristic_velocity` params
- Reference: Towne, Schmidt & Colonius (2018), JFM 847

**Limitation — `dst` is a Strouhal step, not a frequency step.** After the
block FFT, SPOD normalizes by `sqrt(nblocks * dst)` where
`dst = St[1] - St[0] = df · L / U` (not `df = fs / nfft`). Reported eigenvalues
therefore **scale with U/L**. With the default L = U = 1 (and with the shipped
generators), this coincides with a pure frequency-step weight; any other
characteristic scales silently rescale the energy axis.

### 5. ST-POD — Delay-Embedded Space-Time POD

**Class:** `STPODAnalyzer` · **Lift:** delay/Hankel stacking · **Operator:** POD in delay space

```python
from openmodalpy import STPODAnalyzer

stpod = STPODAnalyzer(file_path="data.mat", embedding_dim=10, n_modes_save=10)
stpod.run_analysis()
```

**Key facts:**
- Constructs block-Hankel matrix from centered snapshots
- Metric in lifted space: I_d ⊗ W
- Modes live in C^{d·Nx}; visualization extracts delay-index-0 block
- Uses `compute_reduced_svd` (ARPACK for large matrices)
- Requires uniform dt

### 6. DMD — Dynamic Mode Decomposition

**Class:** `DMDAnalyzer` · **Lift:** identity (shifted pairs) · **Operator:** LS or TLS regression

```python
from openmodalpy import DMDAnalyzer

dmd = DMDAnalyzer(file_path="data.mat", n_modes_save=10, rank=10)
dmd.load_and_preprocess()
dmd.perform_dmd(method="ls", delays=1)        # standard DMD
dmd.perform_dmd(method="tls", delays=1)       # TLS-DMD
dmd.perform_dmd(method="ls", delays=4,
                named_variant="hodmd")          # HODMD
dmd.perform_dmd(method="tls", delays=4,
                named_variant="tls_hodmd")      # TLS-HODMD
dmd.save_results()
```

**Rank vs saved modes:**
- `rank` — SVD truncation of the DMD operator (the reduced system size).
- `n_modes_save` — how many modes are kept for save/plot after sorting by `|λ|`.
  With an **explicit** `rank`, changing `n_modes_save` alone must not change
  eigenvalues. The default still couples them (deprecated).

| `rank` | Criterion |
|--------|-----------|
| `None` (default, **deprecated**) | Same as today: `min(n_modes_save, min(X1.shape))`, then the relative floor `s_j > rcond * s[0]` (`rcond = max(shape) * eps`, as in `numpy.linalg.pinv`). Emits a `DeprecationWarning`; pass `rank` explicitly. |
| `int` | Explicit rank, still floored by `rcond` (never exceeds what the data supports). |
| `"svht"` | Gavish & Donoho (2014) optimal hard threshold, unknown-noise variant: `τ = λ(β) · median(s)` with `β = min(shape)/max(shape)`. |
| `"energy"` | Smallest `r` with cumulative `s²` fraction ≥ `energy_fraction` (default `0.999`). |

**Why full numerical rank is not the default:** On the shipped cylinder wake
(`Nx=40, Ny=24, Nt=400`, so `X1` is 960×399) the singular spectrum decays
**smoothly** to `σ_min/σ_1 ≈ 4.5×10⁻⁴` and never approaches the machine floor
(`rcond ≈ 2×10⁻¹³`). Keeping every direction above that floor (rank 399)
produces **spurious modes with `|λ| > 1`** (growth outside the unit circle)
that sort *first* by the amplitude ranking — recovered dominant frequency
**≈ 3.11 Hz** against true shedding **≈ 0.167 Hz**. Truncating at
`n_modes_save=10`, `"svht"`, or `"energy"` all recover the physical shedding
mode with `|λ| = 1`. A stability library that manufactures instabilities
cannot default to untruncated DMD. Full rank remains available only by
passing an explicit large `int`.

**Why the default is not SVHT either:** SVHT assumes a low-rank signal plus
**i.i.d. Gaussian noise of constant variance**, and its median-based noise
estimate requires the **true rank below n/2** — otherwise the median singular
value is signal, not noise, and the criterion collapses the rank. Neither
assumption holds for a typical deterministic fluid simulation with a smoothly
decaying spectrum. SVHT is also computed from `X1` alone, while the DMD
operator error depends on content of `X2` outside `range(X1)`. So `"svht"`
ships selectable and documented, not as the default. No automatic criterion is
free of trade-offs; report `effective_rank` and the criterion you used.

**Key facts:**
- Eigenvalues encode frequency (angle) and growth/decay (modulus)
- LS regression assumes noise only in Z+; TLS allows errors on both sides
- Implementation uses broadcasting (`/ s_r`) instead of `np.diag(1/s_r)`
- `named_variant` parameter sets metadata; avoids monkey-patching
- Explicit large `rank` forces the dense SVD path when `min(X1.shape) ≥ 256`
- Reference: Schmid (2010), JFM 656; Tu et al. (2014), JCD 1; Hemati et al. (2017), TCFD 31; Gavish & Donoho (2014), IEEE TIT

### 7. HODMD — Higher-Order DMD

Same class as DMD (`DMDAnalyzer`), with `delays >= 2`.

**Key facts:**
- Delay-embeds snapshots into Hankel vectors before forming pairs
- Captures dynamics that appear nonlinear in original coordinates
- Requires uniform dt (non-uniform sampling corrupts the Hankel lift)
- Both LS-HODMD and TLS-HODMD supported
- Reference: Le Clainche & Vega (2017), SIAM J. Appl. Dyn. Syst. 16

### 8. BSMD — Bispectral Mode Decomposition

**Class:** `BSMDAnalyzer` · **Lift:** Hadamard product of Fourier pairs · **Operator:** cross-bispectral eigenproblem

```python
from openmodalpy import BSMDAnalyzer

bsmd = BSMDAnalyzer(file_path="data.mat", nfft=256, overlap=0.5)
bsmd.run_analysis()
# bsmd.energy_map   — bispectral energy over (f1, f2)
# bsmd.triads       — identified triadic frequency pairs
# bsmd.eigenvalues  — coupling strength per triad
```

**Key facts:**
- Identifies nonlinear triadic interactions (f1 + f2 = f3)
- Uses dominant eigenpair as practical approximation to numerical-radius problem
- Inspired by [Schmidt's MATLAB BMD](https://github.com/olivertschmidt/bmd)
- Reference: Schmidt (2020), Nonlinear Dynamics 102

**Limitation — default triad list is `ALL_TRIADS` with |p| ≤ 8.** The shipped
static triad table only covers frequency-bin indices with absolute value at most
8. At the default `nfft=128` that is the bottom 12.5% of the rfft spectrum;
higher-frequency triads are not analysed unless you pass a custom
`static_triads` list. Triads with any component `|p| > nfft // 2` (outside the
rfft bin range) raise `ValueError`. Dynamic triad selection
(`use_static_triads=False`) is not implemented and raises `NotImplementedError`.

---

## Configuration System

### JSONC config structure

```jsonc
{
  // Suite metadata
  "name": "My analysis suite",
  "description": "Optional description",

  // Case definition (shared across all runs)
  "case": {
    "name": "case_name",           // used for output directory names
    "case_type": "experimental",   // or "analytical"
    "data": {
      "kind": "file",              // "file", "generator", or "dnami"
      "path": "../data/file.mat"   // relative to this config file
    },
    "spatial_weight_type": "uniform",  // "uniform", "polar", or "auto"
    "n_modes_save": 10,
    "rank": null,                  // DMD only: null (deprecated default) | int | "svht" | "energy"
    "nfft": 128,                   // FFT block size (SPOD/BSMD/PSD-POD)
    "overlap": 0.5,                // block overlap fraction
    "embedding_dim": 10,           // delay depth (ST-POD/HODMD)
    "generate_plots": true,
    "results_root": "../results/case_name",
    "figures_root": "../figures/case_name"
  },

  // Runs: each gets its own subdirectory under results/figures root
  "runs": [
    { "id": "pod",      "method": "pod" },
    { "id": "mpod",     "method": "mpod",
      "params": { "band_edges": [0, 0.15, 0.35, 1.0],
                  "band_scale": "normalized_nyquist" } },
    { "id": "dmd_ls",   "method": "dmd",
      "params": { "method": "ls", "delays": 1 } },
    { "id": "dmd_tls",  "method": "dmd",
      "params": { "method": "tls", "delays": 1 } },
    { "id": "hodmd",    "method": "hodmd",
      "params": { "delays": 4 } },
    { "id": "tls_hodmd","method": "tls-hodmd",
      "params": { "delays": 4 } },
    { "id": "spod",     "method": "spod" },
    { "id": "psd_pod",  "method": "psd-pod" },
    { "id": "bsmd",     "method": "bsmd" },
    { "id": "stpod",    "method": "stpod" }
  ]
}
```

### Data source kinds

| Kind | Required fields | Description |
|------|----------------|-------------|
| `"file"` | `path` | Load from `.mat` or `.npz` file |
| `"generator"` | `name`, `params` | Generate at runtime (`double_gyre`, `taylor_green`, `cylinder_wake`) |
| `"dnami"` | `path`, `schema` | Schema-driven dNami NPZ loader (consolidated or split layout) |

### Method aliases

| CLI/config name | Internal ID |
|----------------|-------------|
| `psd-pod` | `psd_pod` |
| `tls-hodmd` | `tls_hodmd` |

All other method names work as-is: `pod`, `mpod`, `dmd`, `hodmd`, `spod`, `bsmd`, `stpod`.

---

## CLI Reference

```
openmodalpy analyze <method> --config <path.jsonc> [options]
openmodalpy run --config <path.jsonc> [--dry-run]
openmodalpy methods list
openmodalpy methods show <name>
openmodalpy examples list
openmodalpy examples show <name>
openmodalpy examples run <name> [--dry-run]
openmodalpy results inspect <path>
```

### analyze options

| Flag | Description |
|------|-------------|
| `--config` | Path to JSONC case config (required) |
| `--run-id` | Custom output subdirectory name |
| `--dry-run` | Preview without executing |
| `--no-plots` | Disable figure generation |
| `--n-modes` | Override n_modes_save |
| `--nfft` | Override FFT block size |
| `--overlap` | Override overlap fraction |
| `--embedding-dim` | Override delay depth |
| `--method ls\|tls` | DMD regression model |
| `--delays` | DMD delay embedding depth |
| `--band-edges` | mPOD band edges (comma-separated) |
| `--band-scale` | mPOD band scale (`hz` or `normalized_nyquist`) |
| `--results-dir` | Override results root |
| `--figures-dir` | Override figures root |
| `--weight-type` | Override spatial weight type |

---

## Implementation Notes

### SVD strategy

`compute_reduced_svd(X, rank)` in `core/base.py`:
- If `rank < min(X.shape)` and `min(X.shape) >= 256`: uses `scipy.sparse.linalg.svds` (ARPACK, truncated)
- Otherwise: uses `np.linalg.svd(X, full_matrices=False)` (dense LAPACK)

Shared by DMD, HODMD, and ST-POD.

### DMD broadcasting

The exact DMD operator `A_tilde = U_r* Z+ V_r Sigma_r^{-1}` is computed as:
```python
atilde = (u_r.conj().T @ X2 @ v_r) / s_r  # broadcasting, not np.diag
modes = X2 @ (v_r / s_r) @ w              # same trick for mode recovery
```

### SPOD FFT caching

SPOD caches blockwise FFT results in the HDF5 output file under the
`FFTBlocks` dataset. Subsequent SPOD runs on the same data with the same
`nfft`/`overlap` skip the FFT computation.

### Delay embedding

`_delay_embed(X, d)` in `dmd.py` builds the Hankel matrix with pre-allocated
output (avoids `np.vstack` of d temporary arrays):
```python
out = np.empty((n * d, cols), dtype=X.dtype)
for i in range(d):
    out[i * n : (i + 1) * n, :] = X[:, i : i + cols]
```

---

## Built-in Generators

| Name | Function | Parameters | Ground truth |
|------|----------|-----------|-------------|
| `double_gyre` | `generate_double_gyre()` | Nx, Ny, Nt, A, epsilon, period, t_max | Known period T, frequency f₀=1/T |
| `taylor_green` | `generate_taylor_green()` | Nx, Ny, Nt, nu, U0, L | Exact decay: λ = e^{-2νΔt}, rank-1 |
| `cylinder_wake` | `generate_cylinder_wake()` | Nx, Ny, Nt, Re, D, U_inf, seed | Known St = 0.212(1 - 21.2/Re) |

---

## Output Format

All analyzers write HDF5 files with:
- **Datasets:** `eigenvalues`, `modes`, `time_coefficients`, coordinates
- **Attributes:** `analysis_type`, `nfft`, `overlap`, `dt`, `Ns`, `Nx`, `Ny`,
  `spatial_weight_type`, method-specific metadata

DMD additionally stores: `amplitudes`, `omega` (continuous-time eigenvalues),
`dmd_variant`, `dmd_method`, `dmd_delays`, `dmd_named_variant`.

SPOD additionally stores: `Eigenvalues`, `Modes`, `Freq`, `St`, `FFTBlocks`.

BSMD additionally stores: `energy_map`, `triads`, `Modes1`, `Modes2`.

---

## Testing

101 tests across 11 test files. Key test categories:

| File | What it tests |
|------|--------------|
| `test_pod.py` | POD eigenvalues, mode shapes, energy convergence |
| `test_mpod.py` | Band filtering, frequency separation |
| `test_dmd.py` | Exact DMD, TLS, delays, HODMD metadata, roundtrip |
| `test_stpod.py` | Delay embedding, Hankel shape, validation |
| `test_spod_plot.py` | SPOD plotting paths |
| `test_bsmd_core.py` | BSMD triad detection, energy map |
| `test_cli_commands.py` | CLI dispatch, config parsing, dry-run |
| `test_dnami_loader.py` | NPZ loading, schema handling |
| `test_weights.py` | Polar and uniform weight computation |

Run all: `uv run pytest tests/ -q`

---

## Dependencies

**Runtime:** `numpy`, `scipy`, `matplotlib`, `h5py`, `tqdm`

**Dev:** `pytest`, `ruff`

**No external solvers.** All linear algebra uses NumPy/SciPy. Future versions
may optionally use SLEPc (`slepc4py`) for distributed eigensolvers.

---

## Extension Paths

To add a new decomposition:

1. **Variance-optimal method** — write a new lift (realization generator), then
   reuse the existing kernel eigenproblem in `core/base.py`.
2. **Evolution-fit method** — write a new lift for paired data, then reuse the
   SVD-based regression in `dmd.py`.
3. **Interaction method** — write a new lift for higher-order objects, then
   implement the coupling optimization.

In all cases:
- Subclass `BaseAnalyzer`
- Register in `METHOD_REGISTRY` in `commands.py`
- Add a `_run_*` function and a dispatch entry in `analyze_from_spec`
- Add a JSONC example config
- Add tests
