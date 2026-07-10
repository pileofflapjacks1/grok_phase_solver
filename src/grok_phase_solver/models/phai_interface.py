"""
PhAI integration interface (Phase 1 stub + architecture notes).

Reference
---------
Larsen, A. S., Rekis, T. & Madsen, A. Ø. (2024).
"PhAI: A deep-learning approach to solve the crystallographic phase problem."
*Science* 385, 522–528. doi:10.1126/science.adn2777

Official code/data archive (U. Copenhagen ERDA)::

    https://erda.ku.dk/archives/e3be15d017d8c4fe81402da833e26894/published-archive.html

PhAI trains a CNN+MLP network with phase recycling on millions of synthetic
structures (primarily P2₁/c small-molecule) and solves phases at ~2 Å using
far fewer reflections than classical direct methods.

This module provides:
- Architecture documentation for reimplementation
- A loader hook for official weights when placed in ``third_party/phai/``
- A NumPy-compatible API so solvers can call NN predictions as phase seeds

Full training/inference requires PyTorch (optional dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


PHAI_ERDA_URL = (
    "https://erda.ku.dk/archives/e3be15d017d8c4fe81402da833e26894/published-archive.html"
)
PHAI_PAPER_DOI = "10.1126/science.adn2777"


def describe_phai_architecture() -> str:
    """Return a human-readable summary of the PhAI method (from the paper)."""
    return """
PhAI architecture summary (Larsen et al., Science 2024)
=======================================================

Problem
-------
Input:  observed amplitudes |F(hkl)| (often incomplete / low resolution ~2 Å)
Output: phases φ(hkl), or equivalently a phased density map

Network design (high level)
---------------------------
1. Reflection features: amplitudes (normalized), Miller indices, resolution
   features, and optionally space-group encodings.
2. Convolutional / dense stages learn local patterns in reciprocal space
   related to atomicity and Sayre-type relationships.
3. Phase recycling: predicted phases are recombined with observed |F|,
   mapped to real space (or intermediate features), and fed back for
   iterative refinement — analogous to density modification loops.
4. Training: millions of synthetic structures with known ground-truth phases;
   loss on phase error (circular) and/or density map correlation.

Training distribution
---------------------
- Small-molecule organic crystals
- Emphasized space group P2₁/c (common for organics)
- Variable resolution and completeness to match difficult experimental cases

Why it works (physics view)
---------------------------
Atomicity + positivity strongly constrain the admissible phase set.
A network trained on valid structures learns an implicit prior p(φ | |F|)
approximating Bayesian phase retrieval / maximum-entropy solutions.
Phase recycling injects the Fourier consistency constraint |F_pred|=|F_obs|.

Integration plan in grok_phase_solver
-------------------------------------
Phase 1: document + download hook + seed-phase API (this module)
Phase 2: reimplement / wrap official weights; physics-informed losses
Phase 3: hybrid CF/HIO refinement seeded by network phases
"""


@dataclass
class PhAIConfig:
    """Configuration for PhAI-compatible inference."""

    weights_path: Optional[str] = None
    device: str = "cpu"
    n_recycle: int = 3
    d_min: float = 2.0
    space_group: str = "P21/c"
    third_party_dir: str = "third_party/phai"
    meta: Dict[str, Any] = field(default_factory=dict)


class PhAIInterface:
    """
    Interface for PhAI-style phase prediction.

    Phase 1: if weights are absent, ``predict_phases`` raises a clear error
    and ``seed_phases`` falls back to random or charge-flipping warm-start.
    """

    def __init__(self, config: Optional[PhAIConfig] = None):
        self.config = config or PhAIConfig()
        self._model = None
        self._loaded = False

    @property
    def available(self) -> bool:
        """True if torch is installed and weights file exists."""
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
        wp = self.config.weights_path
        if wp is None:
            # Search default location
            candidates = list(Path(self.config.third_party_dir).glob("**/*.{pt,pth,ckpt}"))
            return len(candidates) > 0
        return Path(wp).exists()

    def load(self) -> None:
        """Load official or reimplemented weights (Phase 2)."""
        if not self.available:
            raise FileNotFoundError(
                "PhAI weights not found. Download from:\n"
                f"  {PHAI_ERDA_URL}\n"
                f"and place checkpoints under {self.config.third_party_dir}/\n"
                "Also install torch: pip install 'grok-phase-solver[ml]'"
            )
        # Placeholder for actual load
        raise NotImplementedError(
            "Official PhAI weight loader will be wired in Phase 2 once "
            f"checkpoints are present under {self.config.third_party_dir}/. "
            f"See {PHAI_ERDA_URL}"
        )

    def predict_phases(
        self,
        hkl: np.ndarray,
        amplitudes: np.ndarray,
        cell: np.ndarray,
    ) -> np.ndarray:
        """
        Predict phases (radians) from amplitudes.

        Requires loaded model. Use :meth:`seed_phases` for a safe fallback.
        """
        if not self._loaded:
            self.load()
        raise NotImplementedError("PhAI forward pass pending weight integration")

    def seed_phases(
        self,
        hkl: np.ndarray,
        amplitudes: np.ndarray,
        cell: np.ndarray,
        method: str = "auto",
        seed: int = 0,
    ) -> np.ndarray:
        """
        Provide initial phases for hybrid solvers.

        method:
          - ``auto``: use PhAI if available else random
          - ``random``: uniform phases
          - ``phai``: require network
        """
        if method == "phai" or (method == "auto" and self.available):
            return self.predict_phases(hkl, amplitudes, cell)
        rng = np.random.default_rng(seed)
        return rng.uniform(-np.pi, np.pi, size=len(amplitudes))

    def status(self) -> Dict[str, Any]:
        return {
            "paper_doi": PHAI_PAPER_DOI,
            "erda_url": PHAI_ERDA_URL,
            "weights_available": self.available,
            "config": self.config,
            "architecture_notes": "call describe_phai_architecture()",
        }
