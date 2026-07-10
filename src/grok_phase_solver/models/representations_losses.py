"""
Optional PyTorch losses — same mathematics as ``models.losses`` (NumPy).

Only imported when torch is installed. Circular phase identity:

    1 − cos(Δφ) = ½ || (cos φ̂, sin φ̂) − (cos φ, sin φ) ||²
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def require_torch():
    if torch is None:
        raise ImportError("PyTorch not installed. pip install torch  OR use models.losses (NumPy).")


def circular_phase_loss_torch(
    phase_pred: "torch.Tensor",
    phase_true: "torch.Tensor",
    weights: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    require_torch()
    val = 1.0 - torch.cos(phase_pred - phase_true)
    if weights is None:
        return val.mean()
    w = weights / (weights.sum() + 1e-16)
    return (w * val).sum()


def positivity_loss_torch(rho: "torch.Tensor") -> "torch.Tensor":
    require_torch()
    return torch.mean(torch.clamp(rho, max=0.0) ** 2)


def fourier_modulus_loss_torch(
    F_pred: "torch.Tensor",
    amplitudes: "torch.Tensor",
    weights: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    require_torch()
    err = torch.abs(F_pred) - amplitudes
    if weights is None:
        return torch.mean(err**2)
    w = weights / (weights.sum() + 1e-16)
    return (w * err**2).sum()


def laplacian_sharpness_loss_torch(rho: "torch.Tensor") -> "torch.Tensor":
    """
    Encourage atomic peak sharpness: maximize mean |∇²ρ| ≡ minimize −|∇²ρ|.

    Discrete Laplacian via finite differences (periodic).
    """
    require_torch()
    # 3D assumed
    if rho.ndim != 3:
        rho = rho.reshape(rho.shape[0], rho.shape[1], -1)
    lap = (
        torch.roll(rho, 1, 0)
        + torch.roll(rho, -1, 0)
        + torch.roll(rho, 1, 1)
        + torch.roll(rho, -1, 1)
        + torch.roll(rho, 1, 2)
        + torch.roll(rho, -1, 2)
        - 6 * rho
    )
    # want sharp peaks → large |lap| at atoms → minimize negative mean abs
    return -torch.mean(torch.abs(lap))
