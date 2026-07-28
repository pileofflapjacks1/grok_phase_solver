"""
Research-only SE(3)-aware score network skeleton (v0.7).

**Not production.** Documents a future equivariant atomic / density denoiser
conditioned on |F(hkl)| and symmetry (PXRDnet / XRDSol conceptual lineage).

Current status
--------------
- No trained weights; no default pipeline routing.
- Prefer ``diffusion_phase`` Langevin + optional ``PhaseScoreNet`` for experiments.
- Physics fallback: classical ensemble / AI-PhaSeed / partial_phaseed.

Honest non-claim: this stub does not implement full SE(3) equivariance.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def se3_diffusion_available() -> bool:
    """Always False until a real checkpoint + architecture ships."""
    return False


def se3_score_step_stub(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases: np.ndarray,
    *,
    t: float = 0.5,
) -> Tuple[np.ndarray, Dict]:
    """
    Placeholder: returns phases unchanged with research metadata.
    """
    return np.asarray(phases, dtype=np.float64).copy(), {
        "algorithm": "diffusion_se3_stub",
        "status": "research_only",
        "trained": False,
        "t": float(t),
        "note": (
            "SE(3) equivariant diffusion not implemented. "
            "Use diffusion_hybrid / PhaseScoreNet or classical methods."
        ),
    }
