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
