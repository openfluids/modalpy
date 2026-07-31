"""The shipped example configs must still run after an API change.

`openmodalpy examples run <name>` is the project's own smoke test. When DMD
stopped defaulting its truncation rank, the packaged configs under
``src/openmodalpy/examples/`` were updated and the repo-level ``examples/``
copies were not. Nothing caught it, because the two trees are never compared:
an installed wheel sees only the packaged copy and works, while a checkout
prefers the repo copy and fails at the first DMD run.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_EXAMPLES = ROOT / "examples"
PACKAGED_EXAMPLES = ROOT / "src" / "openmodalpy" / "examples"


def _load(path: Path) -> dict:
    """Read a .jsonc, dropping whole-line // comments."""
    return json.loads(re.sub(r"^\s*//.*$", "", path.read_text(encoding="utf-8"), flags=re.M))


def _needs_rank(doc: dict) -> bool:
    return any("dmd" in str(r.get("method", "")).lower() for r in doc.get("runs", []))


def _resolved_rank(doc: dict):
    """Rank the runner would use: per-run params win, else the case default."""
    case_rank = (doc.get("case") or {}).get("rank")
    for run in doc.get("runs", []):
        if "dmd" in str(run.get("method", "")).lower():
            if "rank" not in (run.get("params") or {}) and case_rank is None:
                return None
    return case_rank


def _example_files(root: Path) -> list[Path]:
    return sorted(root.glob("*.jsonc"))


@pytest.mark.parametrize("path", _example_files(PACKAGED_EXAMPLES), ids=lambda p: p.stem)
def test_packaged_example_declares_a_dmd_rank(path: Path) -> None:
    doc = _load(path)
    if not _needs_rank(doc):
        pytest.skip("no DMD-family run in this example")
    assert _resolved_rank(doc) is not None, (
        f"{path.name}: runs DMD but declares no rank; DMDAnalyzer refuses to guess one, so `examples run` fails"
    )


@pytest.mark.parametrize("path", _example_files(REPO_EXAMPLES), ids=lambda p: p.stem)
def test_repo_example_declares_a_dmd_rank(path: Path) -> None:
    doc = _load(path)
    if not _needs_rank(doc):
        pytest.skip("no DMD-family run in this example")
    assert _resolved_rank(doc) is not None, (
        f"{path.name}: runs DMD but declares no rank; DMDAnalyzer refuses to guess one, so `examples run` fails"
    )


@pytest.mark.parametrize("path", _example_files(PACKAGED_EXAMPLES), ids=lambda p: p.stem)
def test_repo_and_packaged_examples_agree_on_rank(path: Path) -> None:
    """The two trees may differ in output paths, but not in what they compute."""
    repo = REPO_EXAMPLES / path.name
    if not repo.exists():
        pytest.skip("packaged-only example")
    assert _resolved_rank(_load(repo)) == _resolved_rank(_load(path)), (
        f"{path.name}: repo and packaged copies disagree on the DMD rank, so a "
        "checkout and an installed wheel would compute different operators"
    )


# Reference fixtures record the generation contract (grid plus any argument the
# generator defaults, such as cylinder_wake's seed) and the DMD rank the
# spectrum was computed at.
FIX_DIR = ROOT / "tests" / "fixtures" / "reference"


@pytest.mark.parametrize("fix_path", sorted(FIX_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_packaged_config_rank_and_params_match_reference_fixture(fix_path: Path) -> None:
    """Packaged config rank and generator params must match each reference fixture.

    For every generator that has a fixture, the resolved DMD rank equals the
    fixture's ``n_modes``, and the packaged ``data.params`` (plus any fixture
    extras such as seed) equal the fixture's ``generator_params``. A rank
    drift in the shipped config would otherwise ship a different operator
    than the one the reference pins.
    """
    fixture = json.loads(fix_path.read_text(encoding="utf-8"))
    name = fixture["generator"]
    assert name == fix_path.stem, f"{fix_path.name}: generator field {name!r} != stem"

    config_path = PACKAGED_EXAMPLES / f"{name}.jsonc"
    assert config_path.is_file(), f"missing packaged config for {name}: {config_path}"
    doc = _load(config_path)

    rank = _resolved_rank(doc)
    assert rank == fixture["n_modes"], (
        f"{name}: packaged config rank {rank} != fixture n_modes {fixture['n_modes']}; "
        "the shipped example would compute a different DMD operator than the reference"
    )

    case = doc.get("case") or {}
    params = dict(case.get("data", {}).get("params", {}))

    # A fixture may record an argument the packaged config leaves unset, because
    # the generator supplies it (cylinder_wake's seed). Fill those from the
    # generator's own signature rather than hardcoding the value here: if a
    # default ever changes, the shipped example starts producing different data,
    # and this test should be what catches it.
    from openmodalpy.example_data import GENERATORS

    signature = inspect.signature(GENERATORS[name]).parameters
    expected = dict(params)
    for key in fixture["generator_params"]:
        if key not in expected:
            assert key in signature, f"{name}: fixture records {key!r}, which the generator does not accept"
            expected[key] = signature[key].default

    assert expected == fixture["generator_params"], (
        f"{name}: packaged generator params {expected} != fixture generator_params {fixture['generator_params']}"
    )
