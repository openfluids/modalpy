"""The shipped example configs must still run after an API change.

`openmodalpy examples run <name>` is the project's own smoke test. When DMD
stopped defaulting its truncation rank, the packaged configs under
``src/openmodalpy/examples/`` were updated and the repo-level ``examples/``
copies were not. Nothing caught it, because the two trees are never compared:
an installed wheel sees only the packaged copy and works, while a checkout
prefers the repo copy and fails at the first DMD run.
"""

from __future__ import annotations

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
