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
    inversion_rate,
    TaxonomyResult,
)
from .seed_quality import (
    predict_seed_quality,
    extract_seed_features,
    oracle_seed_metrics,
    label_class_from_oracle,
    SeedQualityReport,
)
from .strong_seed import strong_seed_metrics, select_strong_indices
from .uncertainty import (
    multistart_phase_uncertainty,
    circular_mean_resultant,
    bootstrap_free_fom_spread,
    density_uncertainty,
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
    "inversion_rate",
    "TaxonomyResult",
    "predict_seed_quality",
    "extract_seed_features",
    "oracle_seed_metrics",
    "label_class_from_oracle",
    "SeedQualityReport",
    "strong_seed_metrics",
    "select_strong_indices",
    "multistart_phase_uncertainty",
    "circular_mean_resultant",
    "bootstrap_free_fom_spread",
    "density_uncertainty",
]
