"""
Input representations for phase retrieval ML.

1. **Reflection features** (vector per hkl) — see phase_mlp.reflection_features
2. **Voxel density** from |F| with zero phases (Patterson-related / random-phase maps)
3. **Graph** — reflections as nodes, edges if h+k+l=0 triplets (direct-methods graph)

All transforms are deterministic and documented so networks cannot hide physics bugs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.density import density_from_structure_factors
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.solvers.direct_methods import build_triplets, normalize_E


def voxelize_amplitudes(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases: Optional[np.ndarray] = None,
    d_min: Optional[float] = None,
    sampling: float = 2.5,
) -> np.ndarray:
    """
    Build a real density map from amplitudes + phases (default phases=0 →
    not Patterson; Patterson uses |F|². For |F| with φ=0 this is the
    'synthetic origin-phased' map — useful as a baseline feature, not a
    solution).
    """
    if phases is None:
        phases = np.zeros(len(amplitudes))
    F = np.asarray(amplitudes) * np.exp(1j * np.asarray(phases))
    return density_from_structure_factors(hkl, F, cell, d_min=d_min, sampling=sampling)


def patterson_voxel(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    d_min: Optional[float] = None,
) -> np.ndarray:
    from grok_phase_solver.physics.patterson import patterson_from_amplitudes

    return patterson_from_amplitudes(hkl, amplitudes, cell, d_min=d_min, remove_origin=True)


def reflection_graph(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    e_min: float = 1.0,
    max_reflections: int = 100,
) -> Dict:
    """
    Build a triplet graph on strong reflections.

    Returns:
      node_idx: indices into original reflection list
      node_features: (M, 4) = [E, s_norm, |h|, amp_norm]
      edges: (T, 3) indices into node list for (h, k, h+k)
      edge_weight: (T,) |E_h E_k E_{h+k}|
    """
    E = normalize_E(hkl, amplitudes, cell)
    strong_idx, E_s, triplets = build_triplets(
        hkl, E, e_min=e_min, max_reflections=max_reflections
    )
    hkl_s = hkl[strong_idx]
    d = d_spacing(hkl_s, cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-6))
    amp_s = amplitudes[strong_idx]
    node_features = np.column_stack(
        [
            E_s,
            s / (s.max() + 1e-16),
            np.linalg.norm(hkl_s, axis=1),
            amp_s / (amp_s.std() + 1e-16),
        ]
    )
    edges = np.array([[t.i_h, t.i_k, t.i_hpk] for t in triplets], dtype=np.int32)
    weights = np.array([t.weight for t in triplets], dtype=np.float64)
    return {
        "node_idx": strong_idx,
        "node_features": node_features,
        "edges": edges,
        "edge_weight": weights,
        "hkl_strong": hkl_s,
        "E": E_s,
    }
