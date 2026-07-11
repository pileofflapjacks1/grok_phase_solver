"""Evaluation metrics for phases and maps."""

from .phase_error import mean_phase_error, phase_error_histogram
from .map_cc import map_correlation, map_correlation_origin_invariant, fourier_shell_correlation
from .rfactor import r_factor, r_free
from .success import SuccessThresholds, SuccessReport, evaluate_success, peak_recovery_score
from .failure_taxonomy import (
    diagnose_structure,
    classify_failure,
    information_metrics,
    summarize_taxonomy,
    TaxonomyResult,
)

__all__ = [
    "mean_phase_error",
    "phase_error_histogram",
    "map_correlation",
    "map_correlation_origin_invariant",
    "fourier_shell_correlation",
    "r_factor",
    "r_free",
    "SuccessThresholds",
    "SuccessReport",
    "evaluate_success",
    "peak_recovery_score",
    "diagnose_structure",
    "classify_failure",
    "information_metrics",
    "summarize_taxonomy",
    "TaxonomyResult",
]
