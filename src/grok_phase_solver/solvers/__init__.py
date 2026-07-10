"""Phase retrieval solvers: physics baselines and ML hooks."""

from .charge_flipping import charge_flipping_solve
from .hio import hio_solve
from .baseline import run_physics_baseline, BaselineResult
from .direct_methods import direct_methods_solve, normalize_E, build_triplets
from .patterson import patterson_solve
from .difference_patterson import locate_heavy_atom_vectors, difference_patterson_map
from .mir_blow_crick import combine_mir_phases, single_isomorphous_replacement
from .density_modification import density_modification_cycle, solvent_flatten
from .hybrid import hybrid_phase_retrieval, blend_phases

__all__ = [
    "charge_flipping_solve",
    "hio_solve",
    "run_physics_baseline",
    "BaselineResult",
    "direct_methods_solve",
    "normalize_E",
    "build_triplets",
    "patterson_solve",
    "locate_heavy_atom_vectors",
    "difference_patterson_map",
    "combine_mir_phases",
    "single_isomorphous_replacement",
    "density_modification_cycle",
    "solvent_flatten",
    "hybrid_phase_retrieval",
    "blend_phases",
]
