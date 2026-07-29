"""Recompute analytic generator spectra and compare to committed fixtures.

Each fixture under ``tests/fixtures/reference/`` records POD energy fractions
and DMD |λ|/phase for a fixed small grid, with ``rtol``/``atol`` stored in the
file. Tolerances are never hard-coded in this test.

Shared spectrum definition: ``tests/reference_helpers.py`` (also used by regen).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.reference_helpers import (
    canonicalize_dmd_eigenvalues,
    compute_reference_spectra,
)

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "reference"


def _fixture_paths() -> list[Path]:
    paths = sorted(FIX_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No reference fixtures in {FIX_DIR}")
    return paths


def _phase_close(got: np.ndarray, expected: np.ndarray, rtol: float, atol: float) -> None:
    """Compare phases on the circle (wrap difference into [-π, π])."""
    assert got.shape == expected.shape, f"phase shape {got.shape} != {expected.shape}"
    delta = np.angle(np.exp(1j * (got - expected)))
    # Bound |δ| like allclose on the wrapped residual against zero.
    limit = atol + rtol * np.maximum(np.abs(expected), 1.0)
    bad = np.abs(delta) > limit
    if np.any(bad):
        idx = int(np.argmax(np.abs(delta)))
        raise AssertionError(
            f"phase mismatch at index {idx}: got={got[idx]!r}, expected={expected[idx]!r}, "
            f"|wrap(δ)|={abs(delta[idx]):.3e}, limit={limit[idx]:.3e} (rtol={rtol}, atol={atol})"
        )


def test_canonical_order_invariant_under_conjugate_tie_permutation():
    """Hand-built tied |λ| pair: recorded order must not depend on emission order.

    Bare ``argsort(|λ|)[::-1]`` (what the analyzer uses) preserves LAPACK's
    conjugate order, so the two emissions below differ by a phase swap. The
    reference-layer canonicalize step must make them agree.
    """
    rtol = 1e-6
    # Same physics, opposite conjugate emission order (goal demo).
    a = np.array([1.0 + 0.0j, 0.7 + 0.7j, 0.7 - 0.7j])
    b = np.array([1.0 + 0.0j, 0.7 - 0.7j, 0.7 + 0.7j])

    bare_a = a[np.argsort(np.abs(a))[::-1]]
    bare_b = b[np.argsort(np.abs(b))[::-1]]
    # Precondition: bare magnitude sort is NOT order-invariant (would be RED
    # if the fixture recorded that order without canonicalization).
    assert not np.allclose(np.angle(bare_a), np.angle(bare_b), atol=1e-12), (
        "precondition failed: bare argsort[::-1] unexpectedly matched for tied |λ|"
    )

    ca = canonicalize_dmd_eigenvalues(a, rtol=rtol)
    cb = canonicalize_dmd_eigenvalues(b, rtol=rtol)
    np.testing.assert_allclose(np.abs(ca), np.abs(cb), rtol=rtol, atol=1e-12)
    np.testing.assert_allclose(np.angle(ca), np.angle(cb), rtol=rtol, atol=1e-12)
    # Magnitudes descending; within each tied group, phase ascending.
    mags = np.abs(ca)
    assert np.all(mags[:-1] >= mags[1:] - 1e-15)
    # The conjugate pair at indices 1,2 must be phase-sorted ascending.
    assert np.angle(ca[1]) <= np.angle(ca[2])


@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_reference_fixture_matches_recompute(path: Path):
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)

    rtol = float(doc["rtol"])
    atol = float(doc["atol"])
    n_modes = int(doc["n_modes"])
    name = doc["generator"]
    assert name == path.stem, f"{path.name}: generator field {name!r} != stem"

    spectra = compute_reference_spectra(
        name, doc["generator_params"], n_modes=n_modes, rtol=rtol
    )

    pod_exp = np.asarray(doc["pod_energy_fractions"], dtype=np.float64)
    abs_exp = np.asarray(doc["dmd_abs_lambda"], dtype=np.float64)
    phase_exp = np.asarray(doc["dmd_phase"], dtype=np.float64)
    ecf_exp = float(doc["energy_captured_fraction"])

    np.testing.assert_allclose(
        spectra["energy_captured_fraction"],
        ecf_exp,
        rtol=rtol,
        atol=atol,
        err_msg=f"{path.name}: energy_captured_fraction",
    )
    np.testing.assert_allclose(
        spectra["pod_energy_fractions"],
        pod_exp,
        rtol=rtol,
        atol=atol,
        err_msg=f"{path.name}: POD energy fractions",
    )
    # Kept fractions sum to the captured fraction, not to 1.
    np.testing.assert_allclose(
        float(np.sum(spectra["pod_energy_fractions"])),
        spectra["energy_captured_fraction"],
        rtol=rtol,
        atol=atol,
        err_msg=f"{path.name}: sum(pod fractions) vs energy_captured_fraction",
    )
    np.testing.assert_allclose(
        spectra["dmd_abs_lambda"],
        abs_exp,
        rtol=rtol,
        atol=atol,
        err_msg=f"{path.name}: DMD |λ|",
    )
    _phase_close(spectra["dmd_phase"], phase_exp, rtol=rtol, atol=atol)


def test_taylor_green_dmd_abs_lambda_matches_closed_form_dmd_eigenvalue():
    """Taylor–Green |λ| is anchored to the generator's closed-form metadata.

    Independent of the golden file: metadata["dmd_eigenvalue"] = exp(-2 ν Δt).
    """
    path = FIX_DIR / "taylor_green.json"
    with path.open(encoding="utf-8") as fh:
        doc = json.load(fh)

    rtol = float(doc["rtol"])
    atol = float(doc["atol"])
    spectra = compute_reference_spectra(
        "taylor_green",
        doc["generator_params"],
        n_modes=int(doc["n_modes"]),
        rtol=rtol,
    )
    closed = float(spectra["payload"]["metadata"]["dmd_eigenvalue"])
    got = float(spectra["dmd_abs_lambda"][0])
    fixture_val = float(doc["dmd_abs_lambda"][0])

    np.testing.assert_allclose(
        got,
        closed,
        rtol=rtol,
        atol=atol,
        err_msg="recomputed dmd_abs_lambda[0] vs metadata dmd_eigenvalue",
    )
    np.testing.assert_allclose(
        fixture_val,
        closed,
        rtol=rtol,
        atol=atol,
        err_msg="fixture dmd_abs_lambda[0] vs metadata dmd_eigenvalue",
    )
