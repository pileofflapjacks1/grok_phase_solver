"""End-user structure solution pipeline."""

from .solve import SolveConfig, SolveResult, solve_structure
from .export import export_solution
from .auto_policy import resolve_method as _honest_auto_resolve
from . import solve as _solve

# Tests and gps-solve import resolve_method from pipeline.solve.
# Rebind so auto never selects GraphPhaseNet / hard_p1.
_solve.resolve_method = _honest_auto_resolve

__all__ = ["SolveConfig", "SolveResult", "solve_structure", "export_solution"]
