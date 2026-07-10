"""Phase retrieval solvers: physics baselines and ML hooks."""

from .charge_flipping import charge_flipping_solve
from .hio import hio_solve
from .baseline import run_physics_baseline, BaselineResult
from .direct_methods import direct_methods_solve, normalize_E, build_triplets, cochran_alpha
from .patterson import patterson_solve
from .difference_patterson import locate_heavy_atom_vectors, difference_patterson_map
from .mir_blow_crick import combine_mir_phases, single_isomorphous_replacement
from .density_modification import density_modification_cycle, solvent_flatten
from .hybrid import hybrid_phase_retrieval, blend_phases
from .phase_recycle import phase_recycle, fourier_modulus_projection
from .iterative_retrieval import raar_solve, difference_map_solve, er_solve
from .conditional_hybrid import conditional_polish, phai_conditional_solve
from .free_fom import free_fom

__all__ = [
    "charge_flipping_solve",
    "hio_solve",
    "run_physics_baseline",
    "BaselineResult",
    "direct_methods_solve",
    "normalize_E",
    "build_triplets",
    "cochran_alpha",
    "patterson_solve",
    "locate_heavy_atom_vectors",
    "difference_patterson_map",
    "combine_mir_phases",
    "single_isomorphous_replacement",
    "density_modification_cycle",
    "solvent_flatten",
    "hybrid_phase_retrieval",
    "blend_phases",
    "phase_recycle",
    "fourier_modulus_projection",
    "raar_solve",
    "difference_map_solve",
    "er_solve",
    "conditional_polish",
    "phai_conditional_solve",
    "free_fom",
]
