"""Shared pytest fixtures for openmodalpy tests."""

from __future__ import annotations

import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reseed_numpy_rng():
    """Reseed NumPy's global RNG before every test from OMPY_TEST_RNG_JITTER.

    Default jitter is 0 (deterministic). Setting the env var to a different int
    changes the global stream so tests that assert on unseeded data fail under
    one jitter and not another — the discriminator for accidental unseeded draws.
    """
    jitter = int(os.environ.get("OMPY_TEST_RNG_JITTER", "0"))
    np.random.seed(jitter)
    yield


def _analytic_rank2_field(Ns: int, Nspace: int) -> dict:
    """Deterministic rank-2 travelling-wave field for POD/ST-POD tests.

    Mean is non-zero so mean-subtraction is exercised. Spatial points exceed or
    trail snapshots depending on the (Ns, Nspace) pair the caller chooses.
    """
    t = np.linspace(0.0, 2.0 * np.pi, Ns, endpoint=False)
    x = np.linspace(0.0, 1.0, Nspace)
    q = (
        1.0
        + np.outer(np.sin(t), np.sin(2.0 * np.pi * x))
        + 0.4 * np.outer(np.cos(3.0 * t), np.cos(2.0 * np.pi * x))
    )
    return {
        "q": np.ascontiguousarray(q, dtype=float),
        "x": x,
        "y": np.array([0.0]),
        "dt": 0.1,
        "Nx": Nspace,
        "Ny": 1,
        "Ns": Ns,
    }


@pytest.fixture
def small_pod_field():
    """POD-able analytic field with Ns > Nspace (spatial-kernel default)."""
    return _analytic_rank2_field(Ns=16, Nspace=10)


@pytest.fixture
def small_stpod_field():
    """ST-POD-able analytic field long enough for modest delay embedding."""
    return _analytic_rank2_field(Ns=40, Nspace=12)
