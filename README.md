# OpenModalPy

Modal decomposition of spatiotemporal data in Python.
Pure NumPy/SciPy — no external solver dependencies.

**Methods:** POD · mPOD · PSD-POD · SPOD · ST-POD · DMD (LS/TLS) · HODMD (LS/TLS) · BSMD

## Installation

```bash
uv add openmodalpy
```

The installed package is imported as `openmodalpy`.

Or as a standalone CLI:

```bash
uv tool install openmodalpy
```

For modern 3D slice/isosurface plotting:

```bash
uv add "openmodalpy[viz3d]"
```

## FFT Backend

FFT dispatch is handled by [`fftkit`](https://github.com/openfluids/fftkit), installed
automatically. It probes the available backends and picks the fastest one, falling back
to SciPy when nothing else is present — so no configuration is needed.

To pin a backend explicitly:

```bash
export FFTKIT_BACKEND=mkl      # or scipy, numpy, cupy, accelerate
```

```python
from openmodalpy.core.config import FFT_BACKEND
print(FFT_BACKEND)   # the backend actually in use
```

Accelerator support comes from the corresponding extra:

```bash
uv add "openmodalpy[mkl]"   # Intel MKL
uv add "openmodalpy[gpu]"   # CuPy / PyTorch
```

> The legacy `PYMODAL_FFT_BACKEND` variable is still honoured as a fallback, but
> `FFTKIT_BACKEND` is the supported name.

## Quick Start

```python
from openmodalpy import PODAnalyzer, DMDAnalyzer, SPODAnalyzer

pod = PODAnalyzer(file_path="data.mat", n_modes_save=10)
pod.run_analysis()

dmd = DMDAnalyzer(file_path="data.mat", n_modes_save=10)
dmd.load_and_preprocess()
dmd.perform_dmd(method="ls")

spod = SPODAnalyzer(file_path="data.mat", nfft=256, overlap=0.5)
spod.run_analysis()
```

## Configuration-Driven Workflow

A single JSONC file runs multiple methods on one dataset:

```jsonc
{
  "case": {
    "name": "my_case",
    "data": { "kind": "file", "path": "data.mat" },
    "n_modes_save": 10, "nfft": 128, "overlap": 0.5
  },
  "runs": [
    { "id": "pod",  "method": "pod" },
    { "id": "spod", "method": "spod" },
    { "id": "dmd",  "method": "dmd", "params": { "method": "ls" } },
    { "id": "hodmd","method": "hodmd", "params": { "delays": 4 } },
    { "id": "bsmd", "method": "bsmd" }
  ]
}
```

```bash
openmodalpy run --config analysis.jsonc
```

## CLI

```bash
openmodalpy analyze pod --config case.jsonc         # one method
openmodalpy run --config suite.jsonc                # full suite
openmodalpy run --config suite.jsonc --dry-run      # preview
openmodalpy methods list                            # supported methods
openmodalpy examples list                           # bundled examples
openmodalpy results inspect output.hdf5             # inspect result
```

## Bundled Example Configs

The public package ships only self-contained generator-backed example configs:

- `double_gyre.jsonc`
- `cylinder_wake.jsonc`
- `taylor_green.jsonc`
- `run_benchmarks.jsonc`

These configs do not depend on repository-local benchmark datasets.

## Supported Methods

| Method | Class | What it extracts |
|--------|-------|-----------------|
| POD | variance-optimal | energy-ranked spatial modes |
| mPOD | variance-optimal | scale-separated modes |
| PSD-POD | variance-optimal | broadband spectral modes |
| SPOD | variance-optimal | frequency-local modes |
| ST-POD | variance-optimal | space-time structures (delay embedding) |
| DMD (LS) | evolution-fit | modes with frequency and growth rate |
| DMD (TLS) | evolution-fit | de-biased DMD for noisy data |
| HODMD | evolution-fit | delay-embedded DMD |
| TLS-HODMD | evolution-fit | de-biased delay-embedded DMD |
| BSMD | triadic interaction | nonlinear triad structures |

The BSMD implementation follows
[Schmidt (2020)](https://doi.org/10.1007/s11071-020-06037-z) and was
inspired by the reference
[MATLAB implementation](https://github.com/olivertschmidt/bmd).

## Data Format

OpenModalPy auto-detects `.mat` and `.npz` files:

```python
{
    "q": np.ndarray,   # (Ns, Nspace) — snapshots × spatial points
    "dt": float,       # time step
    "Nx": int,         # grid points in x
    "Ny": int,         # grid points in y
    "x": np.ndarray,   # x-coordinates
    "y": np.ndarray,   # y-coordinates
}
```

Custom loaders:

```python
def my_loader(path):
    return {"q": data, "dt": 0.01, "Nx": 100, "Ny": 50, "x": x, "y": y}

pod = PODAnalyzer(file_path="ignored", data_loader=my_loader)
```

## References

| Method | Key reference |
|--------|--------------|
| POD | Lumley (1967); Sirovich (1987) |
| mPOD | [Mendez et al. (2019)](https://doi.org/10.1017/jfm.2019.212) |
| SPOD | [Towne, Schmidt & Colonius (2018)](https://doi.org/10.1017/jfm.2018.283) |
| DMD | [Schmid (2010)](https://doi.org/10.1017/S0022112010001217); [Tu et al. (2014)](https://doi.org/10.3934/jcd.2014.1.391) |
| TLS-DMD | [Hemati et al. (2017)](https://doi.org/10.1007/s00162-017-0432-2) |
| HODMD | [Le Clainche & Vega (2017)](https://doi.org/10.1137/15M1054924) |
| BSMD | [Schmidt (2020)](https://doi.org/10.1007/s11071-020-06037-z) |

## License

This project is licensed under Apache-2.0.
Originally developed by Ricardo A S Frantz. See `LICENSE` and `NOTICE` for license terms and attribution notices.
