"""Phase retrieval solvers: physics baselines and ML hooks."""

from .charge_flipping import charge_flipping_solve
from .hio import hio_solve
from .baseline import run_physics_baseline, BaselineResult
from .direct_methods import (
    direct_methods_solve,
    normalize_E,
    build_triplets,
    cochran_alpha,
    dm_ai_hybrid_refine,
    tangent_formula_iteration,
)
from .patterson import patterson_solve
from .difference_patterson import locate_heavy_atom_vectors, difference_patterson_map
from .mir_blow_crick import combine_mir_phases, single_isomorphous_replacement
from .density_modification import density_modification_cycle, solvent_flatten
from .hybrid import hybrid_phase_retrieval, blend_phases
from .phase_recycle import phase_recycle, fourier_modulus_projection
from .iterative_retrieval import (
    raar_solve,
    difference_map_solve,
    er_solve,
    retune_difference_map,
)
from .conditional_hybrid import conditional_polish, phai_conditional_solve
from .free_fom import (
    free_fom,
    should_accept_polish,
    positivity_residual,
    phase_displacement,
    rank_phase_sets,
    DEFAULT_WEIGHTS,
)
from .ensemble import ensemble_solve, ensemble_cf_raar
from .recycle_net import recycle_net_solve, train_recycle_net_hard, load_recycle_net
from .ai_phaseed import (
    ai_phaseed_solve,
    phai_phaseed_solve,
    select_seed_indices,
    phase_extend,
)

__all__ = [
    "charge_flipping_solve",
    "hio_solve",
    "run_physics_baseline",
    "BaselineResult",
    "direct_methods_solve",
    "normalize_E",
    "build_triplets",
    "cochran_alpha",
    "dm_ai_hybrid_refine",
    "tangent_formula_iteration",
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
    "retune_difference_map",
    "conditional_polish",
    "phai_conditional_solve",
    "free_fom",
    "should_accept_polish",
    "positivity_residual",
    "phase_displacement",
    "rank_phase_sets",
    "DEFAULT_WEIGHTS",
    "ensemble_solve",
    "ensemble_cf_raar",
    "recycle_net_solve",
    "train_recycle_net_hard",
    "load_recycle_net",
    "ai_phaseed_solve",
    "phai_phaseed_solve",
    "select_seed_indices",
    "phase_extend",
]
