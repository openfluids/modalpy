"""Typed specs and command metadata for the OpenModalPy CLI/API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class DataSourceSpec:
    """Describe where a case gets its snapshots."""

    kind: Literal["file", "generator", "dnami"]
    path: Path | None = None
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseSpec:
    """Shared case-level settings used by one or more modal runs."""

    name: str
    description: str
    case_type: str
    data: DataSourceSpec
    spatial_weight_type: str = "uniform"
    n_modes_save: int = 10
    # DMD truncation rank: positive int, "svht", or "energy" (required for DMD;
    # None reaches DMDAnalyzer and raises). n_modes_save only bounds saved/plotted
    # output and never sets the operator rank.
    rank: int | str | None = None
    nfft: int = 128
    overlap: float = 0.5
    embedding_dim: int = 10
    use_parallel: bool = True
    generate_plots: bool = True
    results_root: Path | None = None
    figures_root: Path | None = None


@dataclass(frozen=True)
class AnalyzeSpec:
    """Describe one concrete analysis run."""

    run_id: str
    method: str
    case: CaseSpec
    params: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None


@dataclass(frozen=True)
class RunCollectionSpec:
    """Describe either a direct set of analyses or a suite of nested configs."""

    name: str
    description: str
    config_path: Path
    analyses: list[AnalyzeSpec] = field(default_factory=list)
    nested_configs: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class MethodInfo:
    """Metadata exposed through ``openmodalpy methods``."""

    method_id: str
    cli_name: str
    display_name: str
    description: str
    parameter_help: dict[str, str] = field(default_factory=dict)
    implementation_scope: str = "analyzer"


@dataclass(frozen=True)
class ExampleInfo:
    """Discovered example config metadata."""

    name: str
    config_path: Path
    kind: str
    title: str
    description: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOutcome:
    """Execution record returned by the command core."""

    run_id: str
    method: str
    case_name: str
    results_dir: Path
    figures_dir: Path
    results_path: Path | None
    success: bool
    executed: bool
    message: str = ""
