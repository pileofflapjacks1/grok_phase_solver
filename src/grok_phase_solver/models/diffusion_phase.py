"""
Experimental: conditional diffusion stub for density / phase completion.

**Research / not production.** Marked experimental for v0.4+ roadmap.
Does not ship trained weights. Intended future direction: complete weak
reflections or density given AI-PhaSeed seeds (complement to EDM extension).

Honest status
-------------
- No trained model; API sketch only.
- Not claimed to solve hard ab initio phasing.
- Prefer classical AI-PhaSeed + free-FOM polish for real solves.

See ``docs/math/ai_phaseed.md`` and TODO.md.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def diffusion_phase_available() -> bool:
    """True only if a future trained checkpoint is present (never yet)."""
    return False


def conditional_diffusion_complete(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    seed_phases: np.ndarray,
    *,
    n_steps: int = 20,
    seed: int = 0,
    d_min: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Placeholder: would denoise / complete phases conditioned on seed + |F|.

    Currently returns the seed vector unchanged with an experimental flag.
    """
    ph = np.asarray(seed_phases, dtype=np.float64).copy()
    return ph, {
        "algorithm": "diffusion_phase",
        "status": "experimental_stub",
        "trained": False,
        "n_steps": int(n_steps),
        "note": (
            "No diffusion weights shipped. Use ai_phaseed_solve / partial_phaseed "
            "for production paths."
        ),
    }
