"""POD solver route: selectable SVD vs eigh, default unchanged, config plumbing.

Pins that ``perform_pod(solver=...)`` reaches ``weighted_second_order`` and that
``params: {solver: "svd"}`` reaches the analyzer through ``analyze_from_spec``.
These tests must fail against HEAD's hardcoded ``method="eigh"`` call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openmodalpy import PODAnalyzer, analyze_from_spec
from openmodalpy.commands import METHOD_REGISTRY, get_method_spec
from openmodalpy.specs import AnalyzeSpec, CaseSpec, DataSourceSpec


def _clean_rank3(n_s: int = 40, n_x: int = 200):
    t = np.linspace(0.0, 1.0, n_s)
    x = np.linspace(0.0, 1.0, n_x)
    q = (
        np.sin(2 * np.pi * t)[:, None] * np.sin(np.pi * x)[None, :]
        + 0.5 * np.cos(4 * np.pi * t)[:, None] * np.sin(2 * np.pi * x)[None, :]
        + 0.2 * np.sin(6 * np.pi * t)[:, None] * np.sin(3 * np.pi * x)[None, :]
    )
    return {
        "q": q,
        "x": x,
        "y": np.array([0.0]),
        "dt": float(t[1] - t[0]),
        "Nx": n_x,
        "Ny": 1,
        "Ns": n_s,
    }


def _weak_mode_fixture(n_s: int = 40, n_x: int = 200, ratio: float = 1e-9):
    rng = np.random.default_rng(1)
    u = np.linalg.qr(rng.standard_normal((n_s, 4)))[0]
    v = np.linalg.qr(rng.standard_normal((n_x, 4)))[0]
    sig = np.array([1.0, 0.5, 0.25, ratio])
    q = (u[:, :4] * sig) @ v[:, :4].T
    return {
        "q": q,
        "x": np.linspace(0.0, 1.0, n_x),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": n_x,
        "Ny": 1,
        "Ns": n_s,
    }, v[:, 3]


def _make_pod(data: dict, *, n_modes_save: int = 3) -> PODAnalyzer:
    return PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: dict(data, q=np.asarray(data["q"]).copy()),
        spatial_weight_type="uniform",
        n_modes_save=n_modes_save,
    )


def _best_corr(modes: np.ndarray, target: np.ndarray) -> float:
    if modes.shape[1] == 0:
        return 0.0
    c = np.abs(target @ modes) / (np.linalg.norm(target) * np.linalg.norm(modes, axis=0))
    return float(np.max(c))


def test_perform_pod_accepts_solver_keyword():
    """``solver=`` is a keyword on perform_pod (TypeError on HEAD)."""
    data = _clean_rank3()
    a = _make_pod(data)
    a.load_and_preprocess()
    a.perform_pod(solver="eigh")
    assert a.modes.shape[1] >= 1


def test_svd_and_eigh_agree_on_well_conditioned_data():
    data = _clean_rank3()
    a_e = _make_pod(data, n_modes_save=3)
    a_e.load_and_preprocess()
    a_e.perform_pod(solver="eigh")

    a_s = _make_pod(data, n_modes_save=3)
    a_s.load_and_preprocess()
    a_s.perform_pod(solver="svd")

    assert a_e.modes.shape[1] >= 3 and a_s.modes.shape[1] >= 3
    for i in range(3):
        corr = abs(float(np.dot(a_e.modes[:, i], a_s.modes[:, i]))) / (
            np.linalg.norm(a_e.modes[:, i]) * np.linalg.norm(a_s.modes[:, i])
        )
        assert corr > 0.999, f"mode {i} correlation {corr}"
    rel = np.abs(a_e.eigenvalues[:3] - a_s.eigenvalues[:3]) / max(float(a_e.eigenvalues[0]), 1e-300)
    assert float(np.max(rel)) <= 1e-8


def test_svd_recovers_weak_mode_eigh_does_not():
    data, planted = _weak_mode_fixture()
    a_s = _make_pod(data, n_modes_save=8)
    a_s.load_and_preprocess()
    a_s.perform_pod(solver="svd")

    a_e = _make_pod(data, n_modes_save=8)
    a_e.load_and_preprocess()
    a_e.perform_pod(solver="eigh")

    c_svd = _best_corr(a_s.modes, planted)
    c_eigh = _best_corr(a_e.modes, planted)
    assert c_svd >= 0.9, f"svd correlation {c_svd}"
    assert c_eigh <= 0.5, f"eigh correlation {c_eigh} (fixture must show the gap)"


def test_default_perform_pod_is_eigh():
    """Bare perform_pod() takes the eigh route specifically, not merely "a" route.

    Deliberately uses the weak-mode fixture rather than clean data: on
    well-conditioned input the two routes agree to ~1e-16 (the test above pins
    exactly that), so clean data cannot tell them apart and would pass even if
    the default silently became "svd". Here the routes give different answers,
    so the assertions below have something to bite on.
    """
    data, planted = _weak_mode_fixture()

    bare = _make_pod(data, n_modes_save=8)
    bare.load_and_preprocess()
    bare.perform_pod()

    explicit = _make_pod(data, n_modes_save=8)
    explicit.load_and_preprocess()
    explicit.perform_pod(solver="eigh")

    # Bit-for-bit the explicit eigh route: same code path, same input.
    assert bare.modes.shape == explicit.modes.shape
    assert np.allclose(bare.eigenvalues, explicit.eigenvalues, rtol=0, atol=0)
    assert np.allclose(bare.modes, explicit.modes, rtol=0, atol=0)

    # ...and demonstrably NOT the svd route, which finds the planted mode here.
    assert _best_corr(bare.modes, planted) <= 0.5


def test_unknown_solver_raises_value_error():
    data = {
        "q": np.random.default_rng(2).standard_normal((10, 20)),
        "x": np.linspace(0.0, 1.0, 20),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 20,
        "Ny": 1,
        "Ns": 10,
    }
    a = _make_pod(data)
    a.load_and_preprocess()
    with pytest.raises(ValueError, match="solver"):
        a.perform_pod(solver="banana")


def test_parameter_help_documents_solver():
    info = get_method_spec("pod")
    assert "solver" in info.parameter_help
    text = info.parameter_help["solver"].lower()
    assert "eigh" in text and "svd" in text
    # Registry entry is the same object the CLI help walks.
    assert "solver" in METHOD_REGISTRY["pod"].parameter_help


def test_analyze_from_spec_passes_solver_to_perform_pod(tmp_path: Path, monkeypatch):
    """``params: {solver: "svd"}`` reaches perform_pod through analyze_from_spec."""
    captured: list[dict] = []
    orig = PODAnalyzer.perform_pod

    def wrap(self, *args, **kwargs):
        captured.append(dict(kwargs))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(PODAnalyzer, "perform_pod", wrap)
    # Skip plots and result I/O that needs a real figure backend setup.
    monkeypatch.setattr(PODAnalyzer, "save_results", lambda self, *a, **k: None)
    monkeypatch.setattr(PODAnalyzer, "plot_eigenvalues", lambda self, *a, **k: None)
    monkeypatch.setattr(PODAnalyzer, "plot_modes", lambda self, *a, **k: None)
    monkeypatch.setattr(PODAnalyzer, "plot_time_coefficients", lambda self, *a, **k: None)
    monkeypatch.setattr(PODAnalyzer, "plot_cumulative_energy", lambda self, *a, **k: None)

    case = CaseSpec(
        name="solver_probe",
        description="probe",
        case_type="analytical",
        data=DataSourceSpec(
            kind="generator",
            name="double_gyre",
            params={"Nx": 8, "Ny": 4, "Nt": 12},
        ),
        n_modes_save=2,
        generate_plots=False,
        use_parallel=False,
        results_root=tmp_path / "results",
        figures_root=tmp_path / "figures",
    )
    spec = AnalyzeSpec(
        run_id="pod_svd",
        method="pod",
        case=case,
        params={"solver": "svd"},
    )
    outcome = analyze_from_spec(spec)
    assert outcome.success and outcome.executed
    assert captured, "perform_pod was never called"
    assert captured[0].get("solver") == "svd"
