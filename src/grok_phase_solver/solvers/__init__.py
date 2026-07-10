"""Phase retrieval solvers: physics baselines and ML hooks."""

from .charge_flipping import charge_flipping_solve
from .hio import hio_solve
from .baseline import run_physics_baseline, BaselineResult
from .direct_methods import direct_methods_solve, normalize_E, build_triplets
from .patterson import patterson_solve

__all__ = [
    "charge_flipping_solve",
    "hio_solve",
    "run_physics_baseline",
    "BaselineResult",
    "direct_methods_solve",
    "normalize_E",
    "build_triplets",
    "patterson_solve",
]
