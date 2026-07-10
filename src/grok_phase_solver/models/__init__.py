"""Neural models for phase prediction (Phase 2+)."""

from .phai_interface import PhAIConfig, PhAIInterface, describe_phai_architecture
from .losses import (
    circular_phase_loss,
    positivity_loss,
    fourier_modulus_loss,
    triplet_fom_loss,
    combined_phase_loss,
)

__all__ = [
    "PhAIConfig",
    "PhAIInterface",
    "describe_phai_architecture",
    "circular_phase_loss",
    "positivity_loss",
    "fourier_modulus_loss",
    "triplet_fom_loss",
    "combined_phase_loss",
]
