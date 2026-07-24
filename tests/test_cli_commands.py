from __future__ import annotations

import json
from pathlib import Path

import h5py

from modalpy import analyze_from_config
from modalpy.cli import main
from modalpy.commands import _maybe_plot_volumetric_modes, discover_examples, get_method_spec, inspect_results, run_from_config


def _write_jsonc(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def test_method_registry_exposes_mpod_psd_pod_and_hodmd() -> None:
    mpod = get_method_spec("mpod")
    psd_pod = get_method_spec("psd-pod")
    hodmd = get_method_spec("hodmd")
    tls_hodmd = get_method_spec("tls-hodmd")

    assert mpod.method_id == "mpod"
    assert mpod.cli_name == "mpod"
    assert "second-order" in mpod.description
    assert psd_pod.method_id == "psd_pod"
    assert psd_pod.cli_name == "psd-pod"
    assert "Fourier realizations" in psd_pod.description
    assert hodmd.method_id == "hodmd"
    assert "Hankel" in hodmd.description
    assert "delays" in hodmd.parameter_help
    assert tls_hodmd.method_id == "tls_hodmd"


def test_analyze_from_config_routes_hodmd_aliases(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "case.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy case",
            "description": "Toy analytical case",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 8, "Ny": 4, "Nt": 12},
                },
                "embedding_dim": 5,
                "spatial_weight_type": "uniform",
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
            },
            "runs": [],
        },
    )

    captured: list[dict[str, object]] = []

    class FakeDMDAnalyzer:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        def load_and_preprocess(self):
            return None

        def perform_dmd(self, *, method: str, delays: int, named_variant: str | None = None):
            captured.append({"method": method, "delays": delays, "named_variant": named_variant, "results_dir": self._kwargs["results_dir"]})

        def save_results(self):
            Path(self._kwargs["results_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(self._kwargs["results_dir"]) / "fake.hdf5").write_text("fake")

        def plot_eigenvalues(self):
            raise AssertionError("plots should be disabled in this test")

        def plot_modes(self, **kwargs):
            raise AssertionError("plots should be disabled in this test")

        def plot_time_coefficients(self, **kwargs):
            raise AssertionError("plots should be disabled in this test")

        def plot_cumulative_energy(self):
            raise AssertionError("plots should be disabled in this test")

    monkeypatch.setattr("modalpy.commands.DMDAnalyzer", FakeDMDAnalyzer)

    outcome_hodmd = analyze_from_config(config_path, method="hodmd", overrides={"generate_plots": False})
    outcome_tls = analyze_from_config(config_path, method="tls-hodmd", overrides={"generate_plots": False})

    assert captured[0]["method"] == "ls"
    assert captured[0]["delays"] == 5
    assert captured[0]["named_variant"] == "hodmd"
    assert captured[1]["method"] == "tls"
    assert captured[1]["delays"] == 5
    assert captured[1]["named_variant"] == "tls_hodmd"
    assert outcome_hodmd.method == "hodmd"
    assert outcome_tls.method == "tls_hodmd"


def test_analyze_from_config_forwards_dmd_variant_options(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "case.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy case",
            "description": "Toy analytical case",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 8, "Ny": 4, "Nt": 12},
                },
                "spatial_weight_type": "uniform",
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
            },
            "runs": [],
        },
    )

    captured: dict[str, object] = {}

    class FakeDMDAnalyzer:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def load_and_preprocess(self):
            captured["loaded"] = True

        def perform_dmd(self, *, method: str, delays: int, named_variant: str | None = None):
            captured["perform"] = {"method": method, "delays": delays}

        def save_results(self):
            results_dir = Path(captured["init"]["results_dir"])
            results_dir.mkdir(parents=True, exist_ok=True)
            (results_dir / "fake.hdf5").write_text("fake")

        def plot_eigenvalues(self):
            raise AssertionError("plots should be disabled in this test")

        def plot_modes(self, **kwargs):
            raise AssertionError("plots should be disabled in this test")

        def plot_time_coefficients(self, **kwargs):
            raise AssertionError("plots should be disabled in this test")

        def plot_cumulative_energy(self):
            raise AssertionError("plots should be disabled in this test")

    monkeypatch.setattr("modalpy.commands.DMDAnalyzer", FakeDMDAnalyzer)

    outcome = analyze_from_config(
        config_path,
        method="dmd",
        overrides={
            "method": "tls",
            "delays": 4,
            "generate_plots": False,
            "results_root": str(tmp_path / "custom_results"),
            "figures_root": str(tmp_path / "custom_figures"),
        },
    )

    init_kwargs = captured["init"]
    assert captured["loaded"] is True
    assert captured["perform"] == {"method": "tls", "delays": 4}
    assert init_kwargs["file_path"] == "toy_case"
    assert callable(init_kwargs["data_loader"])
    assert init_kwargs["results_dir"].endswith("custom_results/dmd_cli")
    assert init_kwargs["figures_dir"].endswith("custom_figures/dmd_cli")
    assert outcome.method == "dmd"
    assert outcome.executed is True


def test_run_from_config_executes_runs_schema(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "suite.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy suite",
            "description": "Toy suite",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {"kind": "generator", "name": "double_gyre", "params": {"Nx": 8, "Ny": 4, "Nt": 12}},
            },
            "runs": [
                {"id": "pod", "method": "pod"},
                {"id": "mpod", "method": "mpod"},
                {"id": "psd", "method": "psd-pod"},
                {"id": "hodmd", "method": "hodmd"},
                {"id": "tls", "method": "dmd", "params": {"method": "tls", "delays": 3}},
            ],
        },
    )

    seen: list[tuple[str, str, dict[str, object]]] = []

    def fake_analyze(spec, *, dry_run: bool = False):
        seen.append((spec.run_id, spec.method, dict(spec.params)))
        return object()

    monkeypatch.setattr("modalpy.commands.analyze_from_spec", fake_analyze)

    run_from_config(config_path)

    assert seen == [
        ("pod", "pod", {}),
        ("mpod", "mpod", {}),
        ("psd", "psd_pod", {}),
        ("hodmd", "hodmd", {}),
        ("tls", "dmd", {"method": "tls", "delays": 3}),
    ]


def test_run_from_config_executes_nested_config_suite(tmp_path: Path, monkeypatch) -> None:
    case_a = tmp_path / "a.jsonc"
    case_b = tmp_path / "b.jsonc"
    suite = tmp_path / "suite.jsonc"

    payload = {
        "name": "Toy case",
        "description": "Toy case",
        "case": {
            "name": "toy_case",
            "case_type": "analytical",
            "data": {"kind": "generator", "name": "double_gyre", "params": {"Nx": 8, "Ny": 4, "Nt": 12}},
        },
        "runs": [{"id": "pod", "method": "pod"}],
    }
    _write_jsonc(case_a, payload)
    _write_jsonc(case_b, payload | {"name": "Toy case B"})
    _write_jsonc(
        suite,
        {
            "name": "Nested suite",
            "description": "Nested suite",
            "configs": [str(case_a), str(case_b)],
        },
    )

    seen: list[str] = []

    def fake_analyze(spec, *, dry_run: bool = False):
        seen.append(spec.config_path.name)
        return object()

    monkeypatch.setattr("modalpy.commands.analyze_from_spec", fake_analyze)

    run_from_config(suite)

    assert seen == ["a.jsonc", "b.jsonc"]


def test_discover_examples_lists_repo_configs() -> None:
    example_names = {info.name for info in discover_examples()}

    assert "run_benchmarks" in example_names
    assert "cavity" in example_names
    assert "cylinder" in example_names
    assert "jet" in example_names


def test_discover_examples_falls_back_to_packaged_resources(monkeypatch) -> None:
    monkeypatch.setattr("modalpy.commands.examples_root", lambda: Path("/definitely/missing/examples"))

    example_names = {info.name for info in discover_examples()}

    assert "run_benchmarks" in example_names
    assert "double_gyre" in example_names


def test_command_core_uses_volumetric_plot_hooks_when_available() -> None:
    calls = []

    class FakeAnalyzer:
        data = {"Nz": 2}

        def plot_modes_3d_slices(self, **kwargs):
            calls.append(("slices", kwargs))

        def plot_modes_3d_isometric(self, **kwargs):
            calls.append(("iso", kwargs))

    analyzer = FakeAnalyzer()
    used = _maybe_plot_volumetric_modes(analyzer, plot_n_modes=2, slices_kwargs={"freqs_to_plot": [1]})

    assert used is True
    assert calls == [
        ("slices", {"plot_n_modes": 2, "freqs_to_plot": [1]}),
        ("iso", {"plot_n_modes": 2}),
    ]


def test_inspect_results_reads_hdf5_metadata(tmp_path: Path) -> None:
    result_path = tmp_path / "toy.hdf5"
    with h5py.File(result_path, "w") as handle:
        handle.create_dataset("Modes", data=[[1.0, 2.0]])
        handle.attrs["analysis_type"] = "pod"
        handle.attrs["Ns"] = 5

    summary = inspect_results(result_path)

    assert summary["type"] == "hdf5"
    assert summary["datasets"]["Modes"]["shape"] == [1, 2]
    assert summary["attrs"]["analysis_type"] == "pod"
    assert summary["attrs"]["Ns"] == 5


def test_cli_analyze_subcommand_routes_overrides(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "case.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "Toy case",
            "description": "Toy case",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {"kind": "generator", "name": "double_gyre", "params": {"Nx": 8, "Ny": 4, "Nt": 12}},
            },
            "runs": [{"id": "pod", "method": "pod"}],
        },
    )

    captured: dict[str, object] = {}

    def fake_analyze_from_config(config_path, *, method, run_id=None, overrides=None, dry_run=False):
        captured["config_path"] = Path(config_path)
        captured["method"] = method
        captured["run_id"] = run_id
        captured["overrides"] = dict(overrides or {})
        captured["dry_run"] = dry_run
        return object()

    monkeypatch.setattr("modalpy.cli.analyze_from_config", fake_analyze_from_config)

    exit_code = main(
        [
            "analyze",
            "dmd",
            "--config",
            str(config_path),
            "--method",
            "tls",
            "--delays",
            "4",
            "--no-plots",
            "--run-id",
            "custom_run",
        ]
    )

    assert exit_code == 0
    assert captured["config_path"] == config_path.resolve()
    assert captured["method"] == "dmd"
    assert captured["run_id"] == "custom_run"
    assert captured["dry_run"] is False
    assert captured["overrides"]["method"] == "tls"
    assert captured["overrides"]["delays"] == 4
    assert captured["overrides"]["generate_plots"] is False
    assert capsys.readouterr().out == ""


def test_run_from_config_executes_real_psd_pod(tmp_path: Path) -> None:
    config_path = tmp_path / "psd_pod_case.jsonc"
    _write_jsonc(
        config_path,
        {
            "name": "PSD-POD toy case",
            "description": "Real PSD-POD command-core execution",
            "case": {
                "name": "toy_case",
                "case_type": "analytical",
                "data": {
                    "kind": "generator",
                    "name": "double_gyre",
                    "params": {"Nx": 8, "Ny": 4, "Nt": 20},
                },
                "spatial_weight_type": "uniform",
                "n_modes_save": 4,
                "nfft": 8,
                "overlap": 0.5,
                "embedding_dim": 4,
                "generate_plots": False,
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
            },
            "runs": [{"id": "psd", "method": "psd-pod"}],
        },
    )

    outcomes = run_from_config(config_path)

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.method == "psd_pod"
    assert outcome.executed is True
    assert outcome.results_path is not None
    assert outcome.results_path.is_file()

    summary = inspect_results(outcome.results_path)
    assert summary["attrs"]["analysis_type"] == "psd_pod"
    assert "Eigenvalues" in summary["datasets"]
    assert "Modes" in summary["datasets"]
    assert "TimeCoefficients" in summary["datasets"]
