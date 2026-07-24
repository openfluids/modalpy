"""ModalPy public package."""

from modalpy.bmsd import BSMDAnalyzer
from modalpy.commands import (
    analyze_from_config,
    analyze_from_spec,
    discover_examples,
    get_method_spec,
    inspect_results,
    list_methods,
    load_case_spec,
    run_from_config,
)
from modalpy.dmd import DMDAnalyzer
from modalpy.mpod import MPODAnalyzer
from modalpy.pod import PODAnalyzer
from modalpy.specs import AnalyzeSpec, CaseSpec, DataSourceSpec, RunOutcome
from modalpy.spod import SPODAnalyzer
from modalpy.stpod import STPODAnalyzer

__version__ = "0.1.1"
__all__ = [
    "PODAnalyzer",
    "MPODAnalyzer",
    "DMDAnalyzer",
    "SPODAnalyzer",
    "BSMDAnalyzer",
    "STPODAnalyzer",
    "AnalyzeSpec",
    "CaseSpec",
    "DataSourceSpec",
    "RunOutcome",
    "analyze_from_spec",
    "analyze_from_config",
    "run_from_config",
    "discover_examples",
    "list_methods",
    "get_method_spec",
    "inspect_results",
    "load_case_spec",
]
