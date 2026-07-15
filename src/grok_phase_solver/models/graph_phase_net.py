"""
Triplet-graph phase network (NumPy).

Stronger prior than per-reflection PhaseMLP: reflections are nodes; Cochran
triplets define edges so message passing can encode φ_h + φ_k ≈ φ_{h+k}
structure without claiming a general phase-problem solution.

Architecture
------------
1. Node features x_i ∈ R^{d_in} (E, resolution, |h|, amp, …)
2. h^{(0)} = ReLU(x W_in + b_in)
3. For L layers: aggregate neighbor states (weighted by triplet κ/|EEE|),
   h ← ReLU(h W_self + agg W_msg + b)
4. Output (cos φ, sin φ) = h W_out + b_out

Training uses origin/enantiomorph-invariant targets (same idea as hard_p1_prior).
Inference: predict strong-reflection phases → free-FOM origin search →
AI-PhaSeed for full map.

Honest scope: synthetic hard-region seed prior, not a general experimental solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from grok_phase_solver.models.representations import reflection_graph
from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.solvers.direct_methods import normalize_E


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def build_undirected_adj(
    n_nodes: int,
    edges: np.ndarray,
    edge_weight: np.ndarray,
) -> Tuple[List[List[int]], List[List[float]]]:
    """Adjacency lists from triplet edges (connect all three pairs)."""
    nbrs: List[List[int]] = [[] for _ in range(n_nodes)]
    wts: List[List[float]] = [[] for _ in range(n_nodes)]
    if len(edges) == 0:
        return nbrs, wts
    for e, w in zip(edges, edge_weight):
        i, j, k = int(e[0]), int(e[1]), int(e[2])
        w = float(max(w, 1e-8))
        for a, b in ((i, j), (i, k), (j, k)):
            if a == b:
                continue
            nbrs[a].append(b)
            wts[a].append(w)
            nbrs[b].append(a)
            wts[b].append(w)
    return nbrs, wts


def node_features_from_graph(graph: Dict, hkl: np.ndarray, amp: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Richer node features: [E, s_norm, s², |h|_norm, amp_norm, h/hmax, k/kmax, l/lmax]
    """
    idx = graph["node_idx"]
    hkl_s = np.asarray(hkl[idx], dtype=np.float64)
    amp_s = np.asarray(amp[idx], dtype=np.float64)
    E = graph["E"]
    d = d_spacing(hkl_s, cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-6))
    s_n = s / (s.max() + 1e-16)
    hmax = np.maximum(np.abs(hkl_s).max(axis=0), 1.0)
    hn = np.linalg.norm(hkl_s, axis=1)
    hn = hn / (hn.max() + 1e-16)
    amp_n = amp_s / (amp_s.std() + 1e-16)
    return np.column_stack(
        [
            E,
            s_n,
            s_n ** 2,
            hn,
            amp_n,
            hkl_s[:, 0] / hmax[0],
            hkl_s[:, 1] / hmax[1],
            hkl_s[:, 2] / hmax[2],
        ]
    ).astype(np.float64)


@dataclass
class GraphPhaseNet:
    """2-layer message-passing net → (cos φ, sin φ) per strong reflection."""

    d_in: int = 8
    hidden: int = 64
    n_layers: int = 2
    seed: int = 0

    W_in: np.ndarray = field(init=False)
    b_in: np.ndarray = field(init=False)
    W_self: List[np.ndarray] = field(init=False)
    W_msg: List[np.ndarray] = field(init=False)
    b_h: List[np.ndarray] = field(init=False)
    W_out: np.ndarray = field(init=False)
    b_out: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        h = self.hidden
        self.W_in = rng.normal(0, np.sqrt(2 / self.d_in), (self.d_in, h))
        self.b_in = np.zeros(h)
        self.W_self = []
        self.W_msg = []
        self.b_h = []
        for _ in range(self.n_layers):
            self.W_self.append(rng.normal(0, np.sqrt(2 / h), (h, h)))
            self.W_msg.append(rng.normal(0, np.sqrt(2 / h), (h, h)) * 0.5)
            self.b_h.append(np.zeros(h))
        self.W_out = rng.normal(0, np.sqrt(2 / h), (h, 2)) * 0.1
        self.b_out = np.zeros(2)

    def forward(
        self,
        X: np.ndarray,
        nbrs: List[List[int]],
        wts: List[List[float]],
    ) -> Tuple[np.ndarray, dict]:
        """Return (N, 2) cos/sin logits and cache for backward."""
        z0 = X @ self.W_in + self.b_in
        h = _relu(z0)
        cache = {"X": X, "z0": z0, "h0": h, "hs": [h], "zs": [], "aggs": []}

        for ell in range(self.n_layers):
            n = h.shape[0]
            agg = np.zeros_like(h)
            for i in range(n):
                if not nbrs[i]:
                    continue
                ww = np.asarray(wts[i], dtype=np.float64)
                ww = ww / (ww.sum() + 1e-16)
                idx = nbrs[i]
                agg[i] = (ww[:, None] * h[idx]).sum(axis=0)
            z = h @ self.W_self[ell] + agg @ self.W_msg[ell] + self.b_h[ell]
            cache["zs"].append(z)
            cache["aggs"].append(agg)
            h = _relu(z)
            cache["hs"].append(h)

        out = h @ self.W_out + self.b_out
        cache["out"] = out
        cache["h_final"] = h
        return out, cache

    def predict_cos_sin(
        self,
        X: np.ndarray,
        nbrs: List[List[int]],
        wts: List[List[float]],
    ) -> np.ndarray:
        out, _ = self.forward(X, nbrs, wts)
        return out

    def predict_phases(
        self,
        X: np.ndarray,
        nbrs: List[List[int]],
        wts: List[List[float]],
    ) -> np.ndarray:
        z = self.predict_cos_sin(X, nbrs, wts)
        return np.arctan2(z[:, 1], z[:, 0])

    def loss_and_backward(
        self,
        X: np.ndarray,
        nbrs: List[List[int]],
        wts: List[List[float]],
        phase_true: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, dict]:
        out, cache = self.forward(X, nbrs, wts)
        ut = np.column_stack([np.cos(phase_true), np.sin(phase_true)])
        if weights is None:
            w = np.ones(len(X))
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = w / (w.mean() + 1e-16)
        diff = out - ut
        N = len(X)
        loss = 0.5 * np.mean(w * np.sum(diff ** 2, axis=1))
        nrm = np.linalg.norm(out, axis=1)
        loss = loss + 0.05 * np.mean((nrm - 1.0) ** 2)

        dout = (w[:, None] * diff) / N
        scale = 0.05 * 2.0 * (nrm - 1.0) / (N * (nrm + 1e-16))
        dout = dout + scale[:, None] * out

        h = cache["h_final"]
        dW_out = h.T @ dout
        db_out = dout.sum(axis=0)
        dh = dout @ self.W_out.T

        grads = {
            "W_out": dW_out,
            "b_out": db_out,
            "W_self": [],
            "W_msg": [],
            "b_h": [],
        }

        # backprop layers reverse
        for ell in range(self.n_layers - 1, -1, -1):
            z = cache["zs"][ell]
            h_prev = cache["hs"][ell]
            agg = cache["aggs"][ell]
            dz = dh * _relu_grad(z)
            dW_self = h_prev.T @ dz
            dW_msg = agg.T @ dz
            db = dz.sum(axis=0)
            grads["W_self"].insert(0, dW_self)
            grads["W_msg"].insert(0, dW_msg)
            grads["b_h"].insert(0, db)

            dh_prev = dz @ self.W_self[ell].T
            dagg = dz @ self.W_msg[ell].T
            # distribute dagg to neighbors (symmetric undirected — approximate)
            n = h_prev.shape[0]
            dh_from_agg = np.zeros_like(h_prev)
            for i in range(n):
                if not nbrs[i]:
                    continue
                ww = np.asarray(wts[i], dtype=np.float64)
                ww = ww / (ww.sum() + 1e-16)
                for j, wj in zip(nbrs[i], ww):
                    dh_from_agg[j] += wj * dagg[i]
            dh = dh_prev + dh_from_agg

        # input layer
        z0 = cache["z0"]
        dz0 = dh * _relu_grad(z0)
        dW_in = cache["X"].T @ dz0
        db_in = dz0.sum(axis=0)
        grads["W_in"] = dW_in
        grads["b_in"] = db_in
        return float(loss), grads

    def step(self, grads: dict, lr: float = 1e-3) -> None:
        self.W_in -= lr * grads["W_in"]
        self.b_in -= lr * grads["b_in"]
        for ell in range(self.n_layers):
            self.W_self[ell] -= lr * grads["W_self"][ell]
            self.W_msg[ell] -= lr * grads["W_msg"][ell]
            self.b_h[ell] -= lr * grads["b_h"][ell]
        self.W_out -= lr * grads["W_out"]
        self.b_out -= lr * grads["b_out"]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "d_in": self.d_in,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "seed": self.seed,
            "W_in": self.W_in,
            "b_in": self.b_in,
            "W_out": self.W_out,
            "b_out": self.b_out,
        }
        for ell in range(self.n_layers):
            payload[f"W_self_{ell}"] = self.W_self[ell]
            payload[f"W_msg_{ell}"] = self.W_msg[ell]
            payload[f"b_h_{ell}"] = self.b_h[ell]
        if hasattr(self, "_feat_mu"):
            payload["feat_mu"] = self._feat_mu
            payload["feat_sig"] = self._feat_sig
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "GraphPhaseNet":
        z = np.load(path, allow_pickle=True)
        m = cls(
            d_in=int(z["d_in"]),
            hidden=int(z["hidden"]),
            n_layers=int(z["n_layers"]),
            seed=int(z["seed"]),
        )
        m.W_in, m.b_in = z["W_in"], z["b_in"]
        m.W_out, m.b_out = z["W_out"], z["b_out"]
        m.W_self, m.W_msg, m.b_h = [], [], []
        for ell in range(m.n_layers):
            m.W_self.append(z[f"W_self_{ell}"])
            m.W_msg.append(z[f"W_msg_{ell}"])
            m.b_h.append(z[f"b_h_{ell}"])
        if "feat_mu" in z.files:
            m._feat_mu = z["feat_mu"]  # type: ignore[attr-defined]
            m._feat_sig = z["feat_sig"]  # type: ignore[attr-defined]
        return m


def prepare_graph_batch(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    max_reflections: int = 120,
    e_min: float = 0.9,
) -> Dict:
    """Build graph + features for one structure."""
    graph = reflection_graph(
        hkl, amplitudes, cell, e_min=e_min, max_reflections=max_reflections
    )
    X = node_features_from_graph(graph, hkl, amplitudes, cell)
    n = X.shape[0]
    nbrs, wts = build_undirected_adj(n, graph["edges"], graph["edge_weight"])
    idx = graph["node_idx"]
    return {
        "X": X,
        "nbrs": nbrs,
        "wts": wts,
        "node_idx": idx,
        "phases_strong": None,  # filled by caller
        "amp_strong": amplitudes[idx],
        "hkl_strong": hkl[idx],
        "n_edges": len(graph["edges"]),
    }
