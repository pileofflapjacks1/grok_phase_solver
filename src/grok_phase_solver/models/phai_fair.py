"""
Fair PhAI input preparation matching the public notebook + crystallography_module.

Critical details from authors' ``merge_reflections``:
  1. reindex_monoclinic → hemisphere (k≥0, l≥0) with P2₁/c-related sign flips
  2. average duplicates
  3. **scale |F| by 1/max(|F|)**

Public notebook then packs into (1, 21, 11, 11) with index [h+10, k, l].
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np


def reindex_monoclinic(H: np.ndarray) -> np.ndarray:
    """Match PhAI crystallography_module.reindex_monoclinic."""
    H = np.asarray(H, dtype=int)
    H_new = []
    symm_eq = [(-1, 1, -1), (1, -1, 1), (-1, -1, -1)]
    for h in H:
        h = (int(h[0]), int(h[1]), int(h[2]))
        if h[1] < 0 or h[2] < 0:
            placed = False
            for eq in symm_eq:
                h_new = (h[0] * eq[0], h[1] * eq[1], h[2] * eq[2])
                if h_new[1] >= 0 and h_new[2] >= 0:
                    H_new.append(h_new)
                    placed = True
                    break
            if not placed:
                H_new.append(h)
        else:
            if h[2] == 0 and h[0] < 0:  # locus layer
                H_new.append((-h[0], h[1], h[2]))
            else:
                H_new.append(h)
    return np.array(H_new, dtype=np.int32)


def merge_reflections_phai(H: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match PhAI merge_reflections: reindex, average, divide by max.
    """
    H = np.asarray(H, dtype=int)
    F = np.asarray(F, dtype=np.float64)
    H_reind = reindex_monoclinic(H)
    sort_array = np.lexsort((H_reind[:, 2], H_reind[:, 1], H_reind[:, 0]))
    H_reind = H_reind[sort_array]
    F = F[sort_array]

    H_final = []
    F_final = []
    group = [F[0]]
    H_curr = H_reind[0]
    for i in range(len(H_reind)):
        if (H_reind[i] == H_curr).all():
            group.append(F[i])
        else:
            H_final.append(H_curr.copy())
            F_final.append(sum(group) / len(group))
            H_curr = H_reind[i]
            group = [F[i]]
    H_final.append(H_curr.copy())
    F_final.append(sum(group) / len(group))
    H_final = np.array(H_final, dtype=np.int32)
    F_final = np.array(F_final, dtype=np.float64)
    max_f = float(np.max(F_final)) if len(F_final) else 1.0
    if max_f > 0:
        F_final = F_final / max_f
    return H_final, F_final


def build_phai_hkl_array(max_index: int = 10) -> np.ndarray:
    hkl = []
    for h in range(-max_index, max_index + 1):
        for k in range(0, max_index + 1):
            for l in range(0, max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if math.sqrt(h * h + k * k + l * l) <= max_index:
                    hkl.append([h, k, l])
    return np.array(hkl, dtype=np.int32)


def pack_phai_amplitudes_fair(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    max_index: int = 10,
) -> Tuple["object", np.ndarray, List[int], np.ndarray, Dict]:
    """
    Notebook-faithful packing after PhAI merge_reflections.
    """
    import torch

    H_m, F_m = merge_reflections_phai(hkl, amplitudes)
    lookup = {(int(h), int(k), int(l)): float(a) for (h, k, l), a in zip(H_m, F_m)}

    grid = torch.zeros(1, 21, 11, 11)
    hkl_array = build_phai_hkl_array(max_index)
    ordered = []
    for h, k, l in hkl_array:
        a = lookup.get((int(h), int(k), int(l)), 0.0)
        ordered.append(a)
        hh, kk, ll = int(h), int(k), int(l)
        if 0 <= kk <= max_index and 0 <= ll <= max_index and abs(hh) <= max_index:
            # notebook: amplitudes[0][h+10][k][l] = Fabs
            if hh + max_index < 21:
                grid[0, hh + max_index, kk, ll] = a

    order_index = {(int(h), int(k), int(l)): i for i, (h, k, l) in enumerate(hkl_array)}
    # Map original input indices via reindexed key
    H_re_orig = reindex_monoclinic(hkl)
    idx_map: List[int] = []
    for row in H_re_orig:
        key = (int(row[0]), int(row[1]), int(row[2]))
        idx_map.append(order_index.get(key, -1))

    meta = {
        "max_index": max_index,
        "n_merged": len(H_m),
        "n_grid_nonzero": int((grid > 0).sum().item()),
        "n_ordered": len(hkl_array),
        "normalize": "max_after_merge (PhAI official)",
        "frac_input_mapped": float(sum(1 for x in idx_map if x >= 0) / max(len(idx_map), 1)),
        "protocol": "reindex_monoclinic + merge_avg + /max + notebook grid",
    }
    return grid, np.array(ordered, dtype=np.float64), idx_map, hkl_array, meta


def run_phai_fair(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    n_cycles: int = 5,
    random_init: bool = True,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[np.ndarray, Dict]:
    """Run PhAI with notebook-faithful packing; phases in radians for input hkl."""
    import random as pyrandom

    import torch

    from grok_phase_solver.models.phai_network import (
        PhAINeuralNetwork,
        phases_from_logits,
        randomize_output,
    )
    from grok_phase_solver.models.phai_runner import MODEL_ARGS, find_model_path

    path = find_model_path()
    if path is None:
        raise FileNotFoundError("PhAI_model.pth not found")

    grid, ordered, idx_map, hkl_array, meta = pack_phai_amplitudes_fair(hkl, amplitudes)
    grid = grid.to(device)

    model = PhAINeuralNetwork(**MODEL_ARGS)
    state = torch.load(str(path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    pyrandom.seed(seed)
    if random_init:
        init_phases = randomize_output(torch.zeros(1, 21, 11, 11))
    else:
        init_phases = torch.zeros(1, 21, 11, 11)
    init_phases = init_phases.to(device)

    with torch.no_grad():
        output = None
        for i in range(n_cycles):
            if i > 0:
                for j, (h, k, l) in enumerate(hkl_array):
                    init_phases[0, int(h) + 10, int(k), int(l)] = output[0, j]
            raw = model(grid, init_phases)
            output = phases_from_logits(raw)

    out_list = output.detach().cpu().tolist()
    ordered_deg = out_list[0] if isinstance(out_list[0], list) else out_list
    phases = np.zeros(len(hkl))
    for i, j in enumerate(idx_map):
        if j >= 0:
            phases[i] = float(ordered_deg[j]) * (np.pi / 180.0)

    meta = dict(meta)
    meta.update(
        {
            "n_cycles": n_cycles,
            "weights": str(path),
            "n_mapped": int(sum(1 for x in idx_map if x >= 0)),
            "n_total": len(hkl),
            "random_init": random_init,
            "seed": seed,
        }
    )
    return phases, meta
