"""run_analysis must call each analyzer's own decomposition method.

A subclass that inherits run_analysis without overriding the decomposition
hook (the original mPOD path) must fail this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openmodalpy import (
    BSMDAnalyzer,
    DMDAnalyzer,
    MPODAnalyzer,
    PODAnalyzer,
    PSDPODAnalyzer,
    SPODAnalyzer,
    STPODAnalyzer,
)
from openmodalpy.example_data import generate_example_dataset
from tests.reference_helpers import make_loader


def _payload():
    return generate_example_dataset("double_gyre", {"Nx": 24, "Ny": 12, "Nt": 40})


def _common(tmp_path: Path, payload) -> dict:
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    results.mkdir()
    figures.mkdir()
    return dict(
        file_path="dispatch_probe",
        data_loader=make_loader(payload),
        spatial_weight_type="uniform",
        results_dir=str(results),
        figures_dir=str(figures),
        use_parallel=False,
    )


def _silence_plots(analyzer) -> None:
    """Skip figure work; this test only checks decomposition dispatch."""
    for name in dir(type(analyzer)):
        if name.startswith("plot"):
            setattr(analyzer, name, lambda *a, **k: None)


def _wrap_on_mro(cls, name: str, called: list, monkeypatch) -> None:
    for base in cls.__mro__:
        if name in vars(base):
            orig = vars(base)[name]

            def wrap(self, *a, _n=name, _o=orig, **k):
                called.append(_n)
                return _o(self, *a, **k)

            monkeypatch.setattr(base, name, wrap)
            return
    raise AttributeError(f"{cls.__name__} has no {name} on its MRO")


@pytest.mark.parametrize(
    "analyzer_cls, expected, forbid, build, run_kwargs",
    [
        (
            PODAnalyzer,
            "perform_pod",
            (),
            lambda c: PODAnalyzer(n_modes_save=4, **c),
            {"plot_n_modes_spatial": 0, "plot_n_coeffs_time": 0},
        ),
        (
            MPODAnalyzer,
            "perform_mpod",
            ("perform_pod",),
            lambda c: MPODAnalyzer(
                n_modes_save=4,
                band_edges=[0.0, 0.15, 0.35, 1.0],
                band_scale="normalized_nyquist",
                **c,
            ),
            {"plot_n_modes_spatial": 0, "plot_n_coeffs_time": 0},
        ),
        (
            DMDAnalyzer,
            "perform_dmd",
            (),
            lambda c: DMDAnalyzer(n_modes_save=4, rank=4, **c),
            {},
        ),
        (
            SPODAnalyzer,
            "perform_spod",
            (),
            lambda c: SPODAnalyzer(nfft=8, overlap=0.0, **c),
            {},
        ),
        (
            BSMDAnalyzer,
            "perform_bsmd",
            (),
            lambda c: BSMDAnalyzer(
                nfft=8,
                overlap=0.0,
                use_static_triads=True,
                static_triads=[(0, 0, 0)],
                **c,
            ),
            {},
        ),
        (
            STPODAnalyzer,
            "perform_stpod",
            (),
            lambda c: STPODAnalyzer(embedding_dim=2, n_modes_save=2, **c),
            {"plot_n_modes": 0, "plot_n_coeffs": 0},
        ),
        (
            PSDPODAnalyzer,
            "perform_psd_pod",
            (),
            lambda c: PSDPODAnalyzer(nfft=8, overlap=0.5, n_modes_save=3, **c),
            {},
        ),
    ],
    ids=["POD", "mPOD", "DMD", "SPOD", "BSMD", "ST-POD", "PSD-POD"],
)
def test_run_analysis_dispatches_own_decomposition(
    analyzer_cls,
    expected,
    forbid,
    build,
    run_kwargs,
    tmp_path,
    monkeypatch,
):
    """Each analyzer's run_analysis must invoke its own perform_* once."""
    assert "run_analysis" in vars(analyzer_cls) or any("run_analysis" in vars(b) for b in analyzer_cls.__mro__[1:]), (
        f"{analyzer_cls.__name__} has no run_analysis"
    )

    payload = _payload()
    common = _common(tmp_path, payload)
    called: list[str] = []

    _wrap_on_mro(analyzer_cls, expected, called, monkeypatch)
    for name in forbid:
        _wrap_on_mro(analyzer_cls, name, called, monkeypatch)

    analyzer = build(common)
    _silence_plots(analyzer)

    # No try/except here on purpose. A bare ``except TypeError: run_analysis()``
    # retry lets this test pass on a run_analysis that called perform_* and then
    # crashed — the decomposition name is already in ``called`` by then.
    analyzer.run_analysis(**run_kwargs)

    assert expected in called, f"{analyzer_cls.__name__}: expected {expected} in {called}"
    for name in forbid:
        assert name not in called, f"{analyzer_cls.__name__}: {name} must not run; got {called}"
    assert called.count(expected) == 1, f"{analyzer_cls.__name__}: {expected} ran {called.count(expected)} times"


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def test_every_pod_subclass_overrides_the_decomposition_hook():
    """The parametrized test above only covers classes someone listed by hand.

    This one needs no list: it walks every PODAnalyzer subclass that exists and
    requires its own ``_perform_decomposition``. Without it a subclass inherits
    POD's, which is exactly how mPOD came to run plain POD and save it as mPOD.
    """
    missing = [c.__name__ for c in _all_subclasses(PODAnalyzer) if "_perform_decomposition" not in vars(c)]
    assert not missing, (
        f"{missing} inherit PODAnalyzer._perform_decomposition, so their run_analysis "
        f"would silently run perform_pod. Override it (see MPODAnalyzer)."
    )
