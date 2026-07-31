"""Guards for the public typed surface: py.typed, exported return types, CLI help."""

from __future__ import annotations

import inspect
import typing
from pathlib import Path

import openmodalpy as om
from openmodalpy import cli
from openmodalpy.commands import METHOD_REGISTRY


def test_py_typed_marker_exists() -> None:
    """Installed packages need the PEP 561 marker next to the package root."""
    package_root = Path(om.__file__).resolve().parent
    assert (package_root / "py.typed").is_file()
    # Also assert the source-tree marker when running editable, so a bare
    # delete of src/openmodalpy/py.typed fails the suite (gate mutation).
    src_marker = Path(__file__).resolve().parents[1] / "src" / "openmodalpy" / "py.typed"
    assert src_marker.is_file()


def test_all_public_return_types_are_exported() -> None:
    """Walk every name in ``__all__``; classes returned by public functions must be public."""
    required = ("MethodInfo", "ExampleInfo", "RunCollectionSpec")
    for name in required:
        assert name in om.__all__, f"{name} missing from __all__"
        assert hasattr(om, name), f"{name} not importable from openmodalpy"

    public_classes = {getattr(om, name) for name in om.__all__ if inspect.isclass(getattr(om, name, None))}
    unexported: list[str] = []
    for name in om.__all__:
        obj = getattr(om, name)
        if not inspect.isfunction(obj):
            continue
        hints = typing.get_type_hints(obj)
        ret = hints.get("return")
        parts = getattr(ret, "__args__", ()) or (ret,)
        for part in parts:
            if (
                inspect.isclass(part)
                and getattr(part, "__module__", "").startswith("openmodalpy")
                and part not in public_classes
            ):
                unexported.append(f"{name} -> {part.__name__}")
    assert not unexported, f"public functions returning unexported types: {unexported}"


def test_cli_analyze_help_matches_method_registry(monkeypatch) -> None:
    """Top-level parser help must list every METHOD_REGISTRY cli_name."""
    # Argparse wraps help text to the terminal width, and it will break a long line
    # mid-token ("tls-\nhodmd"), which no amount of whitespace collapsing puts back
    # together. Pin a wide terminal so the check does not depend on the window the
    # suite happens to run in.
    monkeypatch.setenv("COLUMNS", "400")
    help_text = " ".join(cli.build_parser().format_help().split())
    registry_names = {info.cli_name for info in METHOD_REGISTRY.values()}
    assert registry_names, "METHOD_REGISTRY is empty"
    missing = sorted(name for name in registry_names if name not in help_text)
    assert not missing, f"CLI help missing registry method names: {missing}"
