"""First-principles crystallographic physics."""

from .form_factors import atomic_form_factor, form_factor_table
from .structure_factors import compute_structure_factors, structure_factors_from_density
from .density import density_from_structure_factors, grid_coordinates
from .reciprocal import generate_hkl, d_spacing, resolution_shells
from .patterson import patterson_from_amplitudes, find_patterson_peaks, autocorrelation_density
from .symmetry import (
    parse_space_group,
    expand_fractional_coords,
    apply_centro_phase_constraint,
    filter_systematic_absences,
    space_group_diagnostics,
    is_centrosymmetric,
)
from .device import resolve_device, list_devices, get_device_info

__all__ = [
    "atomic_form_factor",
    "form_factor_table",
    "compute_structure_factors",
    "structure_factors_from_density",
    "density_from_structure_factors",
    "grid_coordinates",
    "generate_hkl",
    "d_spacing",
    "resolution_shells",
    "patterson_from_amplitudes",
    "find_patterson_peaks",
    "autocorrelation_density",
    "parse_space_group",
    "expand_fractional_coords",
    "apply_centro_phase_constraint",
    "filter_systematic_absences",
    "space_group_diagnostics",
    "is_centrosymmetric",
    "resolve_device",
    "list_devices",
    "get_device_info",
]
