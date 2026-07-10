"""
PhAI inference runner using official public weights + vendored architecture.

See models/phai_network.py for architecture attribution (Larsen et al., Science 2024).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "third_party" / "phai" / "weights"

MODEL_ARGS = {
    "max_index": 10,
    "filters": 96,
    "kernel_size": 3,
    "cnn_depth": 6,
    "dim": 1024,
    "dim_exp": 2048,
    "dim_token_exp": 512,
    "mlp_depth": 8,
    "reflections": 1205,
}


def find_model_path() -> Optional[Path]:
    candidates = [
        WEIGHTS / "PhAI_model.pth",
        ROOT / "third_party" / "phai" / "PhAI_model.pth",
    ]
    if WEIGHTS.exists():
        candidates.extend(sorted(WEIGHTS.glob("*.pth")))
    for c in candidates:
        if c.is_file() and c.stat().st_size > 1_000_000:
            return c
    return None


def phai_available() -> bool:
    try:
        import torch  # noqa: F401
        import einops  # noqa: F401
    except ImportError:
        return False
    return find_model_path() is not None


def _build_hkl_array(max_index: int = 10) -> np.ndarray:
    hkl = []
    for h in range(-max_index, max_index + 1):
        for k in range(0, max_index + 1):
            for l in range(0, max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if math.sqrt(h * h + k * k + l * l) <= max_index:
                    hkl.append([h, k, l])
    return np.array(hkl, dtype=np.int32)


def amplitudes_to_phai_grid(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    max_index: int = 10,
):
    import torch

    grid = torch.zeros(1, 21, 11, 11)
    amp_lookup = {}
    for (h, k, l), a in zip(np.asarray(hkl, dtype=int), amplitudes):
        amp_lookup[(int(h), int(k), int(l))] = float(a)
        amp_lookup.setdefault((-int(h), -int(k), -int(l)), float(a))

    hkl_array = _build_hkl_array(max_index)
    ordered = []
    for h, k, l in hkl_array:
        a = amp_lookup.get((int(h), int(k), int(l)), 0.0)
        ordered.append(a)
        hh, kk, ll = int(h), int(k), int(l)
        if -max_index <= hh <= max_index and 0 <= kk <= max_index and 0 <= ll <= max_index:
            grid[0, hh + max_index, kk, ll] = a

    # Scale non-zero amplitudes to unit mean (common NN input hygiene;
    # official notebook uses raw F — rescale may still help transfer)
    nz = grid[grid > 0]
    if nz.numel() > 0:
        grid = grid / (nz.mean() + 1e-8)

    order_index = {(int(h), int(k), int(l)): i for i, (h, k, l) in enumerate(hkl_array)}
    idx_map = []
    for h, k, l in np.asarray(hkl, dtype=int):
        key = (int(h), int(k), int(l))
        if key in order_index:
            idx_map.append(order_index[key])
        elif (-key[0], -key[1], -key[2]) in order_index:
            idx_map.append(order_index[(-key[0], -key[1], -key[2])])
        else:
            idx_map.append(-1)
    return grid, np.array(ordered), idx_map, hkl_array


class PhAIRunner:
    """Load and run official PhAI with n recycle cycles."""

    def __init__(self, weights_path: Optional[Path] = None, device: str = "cpu"):
        import torch
        from grok_phase_solver.models.phai_network import (
            PhAINeuralNetwork,
            randomize_output,
            phases_from_logits,
        )

        self.torch = torch
        self.randomize_output = randomize_output
        self.phases_from_logits = phases_from_logits
        self.device = device
        path = Path(weights_path) if weights_path else find_model_path()
        if path is None:
            raise FileNotFoundError(
                f"PhAI_model.pth not found under {WEIGHTS}. See third_party/phai/README.md"
            )
        self.model = PhAINeuralNetwork(**MODEL_ARGS)
        state = torch.load(str(path), map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(device)
        self.hkl_array = _build_hkl_array(MODEL_ARGS["max_index"])
        self.weights_path = path

    def predict(
        self,
        hkl: np.ndarray,
        amplitudes: np.ndarray,
        n_cycles: int = 5,
        random_init: bool = True,
        seed: int = 0,
    ) -> Tuple[np.ndarray, Dict]:
        torch = self.torch
        max_index = MODEL_ARGS["max_index"]
        amp_grid, ordered_amp, idx_map, hkl_array = amplitudes_to_phai_grid(
            hkl, amplitudes, max_index=max_index
        )
        amp_grid = amp_grid.to(self.device)
        self.hkl_array = hkl_array

        if random_init:
            # seed python random for reproducibility of randomize_output
            import random as pyrandom

            pyrandom.seed(seed)
            init_phases = self.randomize_output(torch.zeros(1, 21, 11, 11))
        else:
            init_phases = torch.zeros(1, 21, 11, 11)
        init_phases = init_phases.to(self.device)

        with torch.no_grad():
            output = None
            for i in range(n_cycles):
                if i > 0:
                    for j, (h, k, l) in enumerate(self.hkl_array):
                        init_phases[0, int(h) + 10, int(k), int(l)] = output[0, j]
                raw = self.model(amp_grid, init_phases)
                output = self.phases_from_logits(raw)

        # Avoid torch↔numpy bridge (breaks when torch built against NumPy 1.x)
        out_list = output.detach().cpu().tolist()
        if isinstance(out_list[0], list):
            ordered_phase_deg = out_list[0]
        else:
            ordered_phase_deg = out_list

        phases = np.zeros(len(hkl))
        for i, j in enumerate(idx_map):
            if j >= 0 and j < len(ordered_phase_deg):
                phases[i] = float(ordered_phase_deg[j]) * (np.pi / 180.0)
            else:
                phases[i] = 0.0

        info = {
            "weights": str(self.weights_path),
            "n_cycles": n_cycles,
            "n_mapped": int(sum(1 for x in idx_map if x >= 0)),
            "n_total": len(hkl),
            "max_index": max_index,
            "space_group_note": "Public PhAI model: P21/c-oriented, max_index=10 grid",
        }
        return phases, info
