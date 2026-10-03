"""Noise schedule for the coordinate denoiser.

σ is log-uniform on [SIGMA_MIN, SIGMA_MAX] during training. Sampling walks
σ linearly from SIGMA_MAX down to SIGMA_MIN. Coordinates stay fractional:
x_t = (x_0 + σ ε) mod 1.
"""

from __future__ import annotations

import numpy as np

SIGMA_MIN = 0.02
SIGMA_MAX = 0.45
SAMPLE_STEPS = 8


def sigma_schedule(n_steps: int) -> np.ndarray:
    """Descending σ values, shape ``(n_steps,)``, inclusive of both endpoints."""
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1")
    if int(n_steps) == 1:
        return np.array([SIGMA_MIN], dtype=np.float64)
    return np.linspace(SIGMA_MAX, SIGMA_MIN, int(n_steps), dtype=np.float64)
