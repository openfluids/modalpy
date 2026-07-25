"""Timestep validation in BaseAnalyzer.load_and_preprocess."""

import numpy as np
import pytest

from openmodalpy import PODAnalyzer

_MISSING = object()


def _make_data(dt):
    Ns, Nx, Ny = 12, 4, 2
    q = np.random.default_rng(0).standard_normal((Ns, Nx * Ny))
    data = {
        "q": q,
        "x": np.arange(Nx, dtype=float),
        "y": np.arange(Ny, dtype=float),
        "Nx": Nx,
        "Ny": Ny,
        "Ns": Ns,
    }
    if dt is not _MISSING:
        data["dt"] = dt
    return data


def _load(dt, file_path="case_snapshot.h5"):
    analyzer = PODAnalyzer(
        file_path=file_path,
        data_loader=lambda _: _make_data(dt),
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    return analyzer


@pytest.mark.parametrize(
    "dt",
    [
        0.0,
        -0.5,
        float("nan"),
        float("inf"),
        float("-inf"),
        _MISSING,
        None,
        np.array([0.1, 0.2]),
    ],
    ids=["zero", "negative", "nan", "inf", "neg_inf", "missing", "none", "array"],
)
def test_invalid_timestep_raises_value_error_naming_source(dt):
    """Every way of not having a usable dt fails the same way, with the same guidance.

    The assertions pin the *content* the caller needs -- which dataset, which
    value, and that they are expected to supply one -- not the exact prose, so
    the message can be reworded without a spurious failure.
    """
    source = "my_dataset.npz"
    with pytest.raises(ValueError, match=r"timestep") as exc_info:
        _load(dt, file_path=source)
    msg = str(exc_info.value)
    assert source in msg, "the message must name the data source"
    assert "dt" in msg, "the message must name the offending parameter"
    assert "provide" in msg.lower(), "the message must tell the caller to supply a dt"


@pytest.mark.parametrize("dt", [0.25, 2.0])
def test_valid_timestep_sets_fs(dt):
    """Two distinct dt, so the bead cannot be satisfied by always raising."""
    analyzer = _load(dt)
    assert np.isclose(analyzer.fs, 1.0 / dt)
    # Validation must read dt, never write it back -- a coerced or defaulted
    # value in data["dt"] would make downstream consumers (dmd omega, SPOD
    # Strouhal) disagree with what the loader actually returned.
    assert analyzer.data["dt"] == dt
