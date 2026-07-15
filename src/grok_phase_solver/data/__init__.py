"""Data acquisition and synthetic generation."""

from .cod import download_cod_cif, download_cod_hkl, COD_SAMPLE_IDS
from .synthetic import generate_random_organic, simulate_diffraction
from .experimental_phasing import (
    simulate_mir,
    simulate_mad,
    simulate_mr,
    mir_phase_indication,
    hybrid_feature_stack_mir,
)
from .wilson import wilson_plot, domain_gap_wilson, domain_gap_report
from .wilson_match import close_wilson_gap, load_reference_template, WilsonMatchConfig
from .synthetic_v2 import (
    generate_fragment_structure,
    write_training_shard,
    apply_partial_occupancy,
    add_heavy_atom,
    make_centrosymmetric_copy,
)

__all__ = [
    "download_cod_cif",
    "download_cod_hkl",
    "COD_SAMPLE_IDS",
    "generate_random_organic",
    "simulate_diffraction",
    "simulate_mir",
    "simulate_mad",
    "simulate_mr",
    "mir_phase_indication",
    "hybrid_feature_stack_mir",
    "wilson_plot",
    "domain_gap_wilson",
    "domain_gap_report",
    "close_wilson_gap",
    "load_reference_template",
    "WilsonMatchConfig",
    "generate_fragment_structure",
    "write_training_shard",
    "apply_partial_occupancy",
    "add_heavy_atom",
    "make_centrosymmetric_copy",
]
