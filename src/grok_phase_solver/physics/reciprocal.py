"""Reciprocal-space geometry: Miller indices, d-spacings, resolution shells."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def metric_tensor(cell: np.ndarray) -> np.ndarray:
    """Direct-space metric tensor G (3x3) from cell (a,b,c,α,β,γ)."""
    a, b, c, al, be, ga = cell
    al, be, ga = np.deg2rad([al, be, ga])
    g = np.array(
        [
            [a * a, a * b * np.cos(ga), a * c * np.cos(be)],
            [a * b * np.cos(ga), b * b, b * c * np.cos(al)],
            [a * c * np.cos(be), b * c * np.cos(al), c * c],
        ],
        dtype=np.float64,
    )
    return g


def reciprocal_metric(cell: np.ndarray) -> np.ndarray:
    """Reciprocal metric tensor G* = G^{-1}."""
    return np.linalg.inv(metric_tensor(cell))


def d_spacing(hkl: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    d-spacings in Å for Miller indices.

    1/d² = h^T G* h
    """
    hkl = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    gstar = reciprocal_metric(cell)
    # s2 = diag(hkl @ gstar @ hkl.T)
    tmp = hkl @ gstar
    s2 = np.sum(tmp * hkl, axis=1)
    s2 = np.maximum(s2, 1e-16)
    return 1.0 / np.sqrt(s2)


def generate_hkl(
    cell: np.ndarray,
    d_min: float,
    expand_friedel: bool = True,
    include_000: bool = False,
) -> np.ndarray:
    """
    Generate all Miller indices with d ≥ d_min (resolution limit d_min in Å).

    Parameters
    ----------
    cell : (6,) unit cell
    d_min : high-resolution limit (Å)
    expand_friedel : if True, include both h and -h
    """
    a, b, c = cell[:3]
    # Conservative index bounds from d_min ≈ a/h_max
    h_max = int(np.ceil(a / d_min)) + 1
    k_max = int(np.ceil(b / d_min)) + 1
    l_max = int(np.ceil(c / d_min)) + 1

    hs = np.arange(-h_max, h_max + 1)
    ks = np.arange(-k_max, k_max + 1)
    ls = np.arange(-l_max, l_max + 1)
    H, K, L = np.meshgrid(hs, ks, ls, indexing="ij")
    hkl = np.column_stack([H.ravel(), K.ravel(), L.ravel()])

    if not include_000:
        hkl = hkl[~((hkl[:, 0] == 0) & (hkl[:, 1] == 0) & (hkl[:, 2] == 0))]

    d = d_spacing(hkl, cell)
    hkl = hkl[d >= d_min - 1e-8]

    if not expand_friedel:
        # Keep hemisphere: h>0 or (h==0 and k>0) or (h==k==0 and l>0)
        mask = (
            (hkl[:, 0] > 0)
            | ((hkl[:, 0] == 0) & (hkl[:, 1] > 0))
            | ((hkl[:, 0] == 0) & (hkl[:, 1] == 0) & (hkl[:, 2] > 0))
        )
        hkl = hkl[mask]

    return hkl.astype(np.int32)


def resolution_shells(
    d: np.ndarray,
    n_shells: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition reflections into resolution shells by equal 1/d³ volume bins.

    Returns shell_id per reflection, d_edges, shell centers.
    """
    d = np.asarray(d, dtype=np.float64)
    invd3 = 1.0 / np.maximum(d, 1e-6) ** 3
    edges = np.quantile(invd3, np.linspace(0, 1, n_shells + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    shell_id = np.digitize(invd3, edges) - 1
    shell_id = np.clip(shell_id, 0, n_shells - 1)
    # d edges from invd3 edges
    d_edges = (1.0 / np.maximum(edges, 1e-16)) ** (1.0 / 3.0)
    centers = 0.5 * (d_edges[:-1] + d_edges[1:])
    return shell_id, d_edges, centers


def completeness(
    hkl_obs: np.ndarray,
    cell: np.ndarray,
    d_min: float,
    friedel_unique: bool = True,
) -> float:
    """Fraction of unique reflections observed at resolution d_min."""
    all_hkl = generate_hkl(cell, d_min, expand_friedel=not friedel_unique)
    if friedel_unique:
        # Map to unique hemisphere keys
        def keyset(hkl):
            keys = set()
            for h, k, l in hkl:
                if h < 0 or (h == 0 and k < 0) or (h == 0 and k == 0 and l < 0):
                    h, k, l = -h, -k, -l
                keys.add((int(h), int(k), int(l)))
            return keys

        return len(keyset(hkl_obs) & keyset(all_hkl)) / max(len(keyset(all_hkl)), 1)
    obs = set(map(tuple, hkl_obs.tolist()))
    all_set = set(map(tuple, all_hkl.tolist()))
    return len(obs & all_set) / max(len(all_set), 1)
