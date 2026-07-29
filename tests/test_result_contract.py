"""Result contract: every producer writes the same dataset names.

One reader (:func:`read_results`) loads any result file — including the old
capitalised layout — into :class:`AnalysisResults`. The gate for
openmodalpy-unify-result-contract-vig greps this file for every producer name.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from openmodalpy import (
    BSMDAnalyzer,
    DMDAnalyzer,
    MPODAnalyzer,
    PODAnalyzer,
    SPODAnalyzer,
    STPODAnalyzer,
    read_results,
)
from openmodalpy.commands import analyze_from_config


def _toy_field(ns: int = 16, nspace: int = 8) -> dict:
    rng = np.random.default_rng(0)
    nx = int(np.sqrt(nspace))
    ny = max(1, nspace // nx)
    nspace = nx * ny
    t = np.linspace(0.0, 2.0 * np.pi, ns, endpoint=False)
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    q = 1.0 + np.outer(np.sin(t), np.sin(2.0 * np.pi * np.tile(x, ny)))
    q = q + 0.3 * rng.standard_normal(q.shape)
    return {
        "q": np.ascontiguousarray(q, dtype=float),
        "x": x,
        "y": y,
        "dt": 0.1,
        "Nx": nx,
        "Ny": ny,
        "Ns": ns,
    }


def _assert_canonical_keys(path: Path, required: set[str]) -> None:
    with h5py.File(path, "r") as handle:
        keys = set(handle.keys())
    for name in required:
        assert name in keys, f"{path}: missing canonical dataset '{name}' (have {sorted(keys)})"
    capitalised = {
        "Modes",
        "Eigenvalues",
        "TimeCoefficients",
        "Freq",
        "St",
        "Modes1",
        "Modes2",
        "Weights",
    }
    assert not (keys & capitalised), f"{path}: still writing capitalised names {keys & capitalised}"


def test_result_contract_all_producers(tmp_path: Path) -> None:
    """Every producer writes lowercase result keys; read_results loads them all.

    Producers covered (gate greps these names): PODAnalyzer, MPODAnalyzer,
    DMDAnalyzer, SPODAnalyzer, BSMDAnalyzer, STPODAnalyzer, psd_pod.
    """
    field = _toy_field()
    common = dict(
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: field,
        spatial_weight_type="uniform",
    )

    # PODAnalyzer
    pod = PODAnalyzer(file_path="pod_contract", n_modes_save=3, **common)
    pod.load_and_preprocess()
    pod.perform_pod()
    pod.save_results("pod.hdf5")
    pod_path = tmp_path / "pod.hdf5"
    _assert_canonical_keys(pod_path, {"modes", "eigenvalues", "time_coefficients"})
    pod_res = read_results(pod_path)
    assert pod_res.modes is not None and pod_res.modes.shape[1] == 3

    # MPODAnalyzer (inherits POD save_results)
    mpod = MPODAnalyzer(file_path="mpod_contract", n_modes_save=2, band_edges=[0.0, 2.0, 5.0], **common)
    mpod.load_and_preprocess()
    mpod.perform_mpod()
    mpod.save_results("mpod.hdf5")
    mpod_path = tmp_path / "mpod.hdf5"
    _assert_canonical_keys(mpod_path, {"modes", "eigenvalues", "time_coefficients"})
    assert read_results(mpod_path).eigenvalues is not None

    # DMDAnalyzer
    dmd = DMDAnalyzer(file_path="dmd_contract", n_modes_save=2, **common, rank=2)
    dmd.load_and_preprocess()
    dmd.perform_dmd()
    dmd.save_results("dmd.hdf5")
    dmd_path = tmp_path / "dmd.hdf5"
    _assert_canonical_keys(dmd_path, {"modes", "eigenvalues", "time_coefficients", "amplitudes"})
    assert read_results(dmd_path).amplitudes is not None

    # SPODAnalyzer
    spod = SPODAnalyzer(file_path="spod_contract", nfft=8, overlap=0.0, **common)
    spod.load_and_preprocess()
    spod.compute_fft_blocks()
    spod.perform_spod()
    spod.save_results("spod.hdf5")
    spod_path = tmp_path / "spod.hdf5"
    _assert_canonical_keys(spod_path, {"modes", "eigenvalues", "freq", "st"})
    spod_res = read_results(spod_path)
    assert spod_res.freq is not None and spod_res.st is not None
    assert spod_res.W is not None  # SPOD used to write Weights

    # BSMDAnalyzer
    bsmd = BSMDAnalyzer(
        file_path="bsmd_contract",
        nfft=8,
        overlap=0.0,
        use_static_triads=True,
        static_triads=[(0, 0, 0)],
        use_parallel=False,
        **common,
    )
    bsmd.load_and_preprocess()
    bsmd.compute_fft_blocks()
    bsmd._perform_static_bsmd_core()
    bsmd.save_results("bsmd.hdf5")
    bsmd_path = tmp_path / "bsmd.hdf5"
    _assert_canonical_keys(bsmd_path, {"modes1", "modes2", "triads", "eigenvalues"})
    bsmd_res = read_results(bsmd_path)
    assert bsmd_res.modes1 is not None and bsmd_res.modes2 is not None

    # STPODAnalyzer
    stpod = STPODAnalyzer(file_path="stpod_contract", embedding_dim=2, n_modes_save=2, **common)
    stpod.load_and_preprocess()
    stpod.perform_stpod()
    stpod.save_results("stpod.hdf5")
    stpod_path = tmp_path / "stpod.hdf5"
    _assert_canonical_keys(stpod_path, {"modes", "eigenvalues", "time_coefficients"})
    assert read_results(stpod_path).modes is not None

    # psd_pod (commands path)
    config_path = tmp_path / "psd_pod.jsonc"
    config_path.write_text(
        json.dumps(
            {
                "name": "psd contract",
                "description": "result contract coverage for psd_pod",
                "case": {
                    "name": "toy",
                    "case_type": "analytical",
                    "data": {
                        "kind": "generator",
                        "name": "double_gyre",
                        "params": {"Nx": 6, "Ny": 4, "Nt": 16},
                    },
                    "spatial_weight_type": "uniform",
                    "n_modes_save": 2,
                    "nfft": 8,
                    "overlap": 0.0,
                    "generate_plots": False,
                    "results_root": str(tmp_path / "psd_results"),
                    "figures_root": str(tmp_path / "psd_figures"),
                },
                "runs": [{"id": "psd", "method": "psd-pod"}],
            },
            indent=2,
        )
    )
    outcome = analyze_from_config(config_path, method="psd-pod")
    assert outcome.results_path is not None
    psd_path = Path(outcome.results_path)
    _assert_canonical_keys(psd_path, {"modes", "eigenvalues", "time_coefficients", "freq", "st"})
    assert read_results(psd_path).attrs.get("analysis_type") == "psd_pod"


def test_read_results_legacy_layout_emits_deprecation(tmp_path: Path) -> None:
    """A file written with the old capitalised SPOD keys still loads."""
    legacy = tmp_path / "legacy_spod.hdf5"
    rng = np.random.default_rng(1)
    nfr, nsp, nmd = 5, 6, 2
    with h5py.File(legacy, "w") as handle:
        handle.create_dataset("Modes", data=rng.standard_normal((nfr, nsp, nmd)))
        handle.create_dataset("Eigenvalues", data=rng.standard_normal((nfr, nmd)))
        handle.create_dataset("TimeCoefficients", data=rng.standard_normal((nfr, nmd, 3)))
        handle.create_dataset("Freq", data=np.linspace(0, 1, nfr))
        handle.create_dataset("St", data=np.linspace(0, 2, nfr))
        handle.create_dataset("Weights", data=np.ones(nsp))
        handle.attrs["analysis_type"] = "spod"

    with pytest.warns(DeprecationWarning, match="legacy name"):
        res = read_results(legacy)

    assert res.modes is not None and res.modes.shape == (nfr, nsp, nmd)
    assert res.eigenvalues is not None and res.eigenvalues.shape == (nfr, nmd)
    assert res.time_coefficients is not None
    assert res.freq is not None and res.st is not None
    assert res.W is not None and res.W.shape == (nsp,)


def test_spod_load_results_rejects_a_file_without_modes(tmp_path: Path) -> None:
    """A file that is not a SPOD result must fail loudly, not load as empty arrays."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    not_spod = results_dir / "not_spod.hdf5"
    with h5py.File(not_spod, "w") as handle:
        handle.create_dataset("x", data=np.linspace(0, 1, 4))
        handle.attrs["analysis_type"] = "spod"

    analyzer = SPODAnalyzer.__new__(SPODAnalyzer)
    analyzer.results_dir = str(results_dir)

    with pytest.raises(KeyError, match="not a SPOD result file"):
        analyzer.load_results("not_spod.hdf5")
