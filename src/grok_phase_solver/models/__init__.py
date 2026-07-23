"""Neural models for phase prediction (Phase 2+)."""

from .phai_interface import PhAIConfig, PhAIInterface, describe_phai_architecture
from .losses import (
    circular_phase_loss,
    positivity_loss,
    fourier_modulus_loss,
    triplet_fom_loss,
    combined_phase_loss,
)
from .phase_mlp import PhaseMLP, reflection_features, train_phase_mlp_on_structure
from .representations import voxelize_amplitudes, patterson_voxel, reflection_graph
from .diffusion_phase import (
    diffusion_hybrid_solve,
    reverse_diffusion_phases,
    conditional_diffusion_complete,
    diffusion_phase_available,
)
# hard_p1_prior imported lazily (avoids cycle: models → data → solvers → models)

__all__ = [
    "PhAIConfig",
    "PhAIInterface",
    "describe_phai_architecture",
    "circular_phase_loss",
    "positivity_loss",
    "fourier_modulus_loss",
    "triplet_fom_loss",
    "combined_phase_loss",
    "PhaseMLP",
    "reflection_features",
    "train_phase_mlp_on_structure",
    "voxelize_amplitudes",
    "patterson_voxel",
    "reflection_graph",
    "diffusion_hybrid_solve",
    "reverse_diffusion_phases",
    "conditional_diffusion_complete",
    "diffusion_phase_available",
]
