"""
Triplet-graph phase network (NumPy, vectorized).

Stronger prior than per-reflection PhaseMLP: reflections are nodes; Cochran
triplets define edges so message passing can encode φ_h + φ_k ≈ φ_{h+k}
structure without claiming a general phase-problem solution.

Architecture
------------
1. Node features x_i ∈ R^{d_in} (E, resolution, |h|, amp, …)
2. h^{(0)} = ReLU(x W_in + b_in)
3. For L layers: agg = Â h  (row-normalized weighted adjacency from triplets),
   h ← ReLU(h W_self + agg W_msg + b)
4. Output (cos φ, sin φ) = h W_out + b_out

Loss = OI MSE on (cos, sin) + unit-norm penalty + optional triplet-consistency
auxiliary (origin-invariant Cochran invariant matching).

Honest scope: synthetic hard-region seed prior, not a general experimental solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from grok_phase_solver.models.representations import reflection_graph
from grok_phase_solver.physics.reciprocal import d_spacing


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


def build_normalized_adj(
    n_nodes: int,
    edges: np.ndarray,
    edge_weight: np.ndarray,
) -> np.ndarray:
    """
    Dense row-normalized weighted adjacency (N×N).

    Triplet (i,j,k) adds undirected edges (i,j), (i,k), (j,k) with weight κ/|EEE|.
    """
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if n_nodes == 0 or len(edges) == 0:
        return A
    for e, w in zip(np.asarray(edges), np.asarray(edge_weight, dtype=np.float64)):
        i, j, k = int(e[0]), int(e[1]), int(e[2])
        w = float(max(w, 1e-8))
        for a, b in ((i, j), (i, k), (j, k)):
            if a == b or a < 0 or b < 0 or a >= n_nodes or b >= n_nodes:
                continue
            A[a, b] += w
            A[b, a] += w
    rs = A.sum(axis=1, keepdims=True)
    A = A / np.maximum(rs, 1e-16)
    return A


def node_features_from_graph(graph: Dict, hkl: np.ndarray, amp: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Node features (d_in=10):
    [E, s_norm, s², |h|_norm, amp_norm, h/hmax, k/kmax, l/lmax, deg_norm, E²_norm]
    """
    idx = graph["node_idx"]
    hkl_s = np.asarray(hkl[idx], dtype=np.float64)
    amp_s = np.asarray(amp[idx], dtype=np.float64)
    E = np.asarray(graph["E"], dtype=np.float64)
    d = d_spacing(hkl_s, cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-6))
    s_n = s / (s.max() + 1e-16)
    hmax = np.maximum(np.abs(hkl_s).max(axis=0), 1.0)
    hn = np.linalg.norm(hkl_s, axis=1)
    hn = hn / (hn.max() + 1e-16)
    amp_n = amp_s / (amp_s.std() + 1e-16)
    n = len(E)
    deg = np.zeros(n, dtype=np.float64)
    edges = graph.get("edges")
    if edges is not None and len(edges) > 0:
        for e in np.asarray(edges):
            i, j, k = int(e[0]), int(e[1]), int(e[2])
            for a, b in ((i, j), (i, k), (j, k)):
                if a == b or a < 0 or b < 0 or a >= n or b >= n:
                    continue
                deg[a] += 1.0
                deg[b] += 1.0
    deg_n = deg / (deg.max() + 1e-16)
    e2 = E ** 2
    e2_n = e2 / (e2.max() + 1e-16)
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
            deg_n,
            e2_n,
        ]
    ).astype(np.float64)


def triplet_cos_invariant(c: np.ndarray, s: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    cos(φ_i + φ_j − φ_k) from cos/sin arrays for each triplet edge (i,j,k).

    Re(z_i z_j conj(z_k)).
    """
    if len(edges) == 0:
        return np.zeros(0, dtype=np.float64)
    i = edges[:, 0].astype(np.int64)
    j = edges[:, 1].astype(np.int64)
    k = edges[:, 2].astype(np.int64)
    return (
        c[i] * c[j] * c[k]
        - s[i] * s[j] * c[k]
        + c[i] * s[j] * s[k]
        + s[i] * c[j] * s[k]
    )


def triplet_loss_and_grad(
    out: np.ndarray,
    edges: np.ndarray,
    edge_weight: np.ndarray,
    phase_true: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """
    MSE on Cochran cos-invariants; returns (loss, dout).

    If phase_true given, match predicted invariant to true invariant
    (origin-invariant supervision). Else push cos → +1 (unsupervised).
    """
    n = out.shape[0]
    dout = np.zeros_like(out)
    if n == 0 or len(edges) == 0:
        return 0.0, dout

    c = out[:, 0]
    s = out[:, 1]
    edges = np.asarray(edges, dtype=np.int64)
    w = np.asarray(edge_weight, dtype=np.float64)
    w = w / (w.mean() + 1e-16)

    cos_p = triplet_cos_invariant(c, s, edges)
    if phase_true is not None:
        ct = np.cos(phase_true)
        st = np.sin(phase_true)
        cos_t = triplet_cos_invariant(ct, st, edges)
    else:
        cos_t = np.ones_like(cos_p)

    diff = cos_p - cos_t
    loss = 0.5 * float(np.mean(w * diff ** 2))
    # dL/d cos_p
    dcos = (w * diff) / max(len(edges), 1)

    i = edges[:, 0]
    j = edges[:, 1]
    k = edges[:, 2]
    # cos = ci cj ck - si sj ck + ci sj sk + si cj sk
    # d/dci = cj ck + sj sk
    # d/dsi = -sj ck + cj sk
    # d/dcj = ci ck + si sk
    # d/dsj = -si ck + ci sk
    # d/dck = ci cj - si sj
    # d/dsk = ci sj + si cj
    np.add.at(dout[:, 0], i, dcos * (c[j] * c[k] + s[j] * s[k]))
    np.add.at(dout[:, 1], i, dcos * (-s[j] * c[k] + c[j] * s[k]))
    np.add.at(dout[:, 0], j, dcos * (c[i] * c[k] + s[i] * s[k]))
    np.add.at(dout[:, 1], j, dcos * (-s[i] * c[k] + c[i] * s[k]))
    np.add.at(dout[:, 0], k, dcos * (c[i] * c[j] - s[i] * s[j]))
    np.add.at(dout[:, 1], k, dcos * (c[i] * s[j] + s[i] * c[j]))
    return loss, dout


@dataclass
class GraphPhaseNet:
    """Message-passing net → (cos φ, sin φ) per strong reflection."""

    d_in: int = 10
    hidden: int = 64
    n_layers: int = 2
    seed: int = 0
    residual: bool = True

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
        self._adam_t = 0
        self._adam_m: Optional[dict] = None
        self._adam_v: Optional[dict] = None

    def _resolve_adj(
        self,
        X: np.ndarray,
        nbrs: Optional[List[List[int]]] = None,
        wts: Optional[List[List[float]]] = None,
        adj: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = X.shape[0]
        if adj is not None:
            return np.asarray(adj, dtype=np.float64)
        if nbrs is None:
            return np.zeros((n, n), dtype=np.float64)
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            if not nbrs[i]:
                continue
            ww = np.asarray(wts[i] if wts is not None else [1.0] * len(nbrs[i]), dtype=np.float64)
            ww = ww / (ww.sum() + 1e-16)
            for j, wj in zip(nbrs[i], ww):
                A[i, int(j)] += float(wj)
        return A

    def _match_features(self, X: np.ndarray) -> np.ndarray:
        """Pad/truncate node features to ``d_in`` (v3 d_in=8 → v4 d_in=10)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            return X
        d = X.shape[1]
        if d == self.d_in:
            return X
        if d > self.d_in:
            return X[:, : self.d_in]
        pad = np.zeros((X.shape[0], self.d_in - d), dtype=np.float64)
        return np.concatenate([X, pad], axis=1)

    def forward(
        self,
        X: np.ndarray,
        nbrs: Optional[List[List[int]]] = None,
        wts: Optional[List[List[float]]] = None,
        adj: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Return (N, 2) cos/sin logits and cache for backward."""
        X = self._match_features(X)
        A = self._resolve_adj(X, nbrs, wts, adj)
        z0 = X @ self.W_in + self.b_in
        h = _relu(z0)
        cache = {
            "X": X,
            "z0": z0,
            "h0": h,
            "hs": [h],
            "zs": [],
            "aggs": [],
            "A": A,
            "residual": self.residual,
        }

        for ell in range(self.n_layers):
            agg = A @ h
            z = h @ self.W_self[ell] + agg @ self.W_msg[ell] + self.b_h[ell]
            cache["zs"].append(z)
            cache["aggs"].append(agg)
            h_act = _relu(z)
            if self.residual:
                h = h + h_act
            else:
                h = h_act
            cache["hs"].append(h)

        out = h @ self.W_out + self.b_out
        cache["out"] = out
        cache["h_final"] = h
        return out, cache

    def predict_cos_sin(
        self,
        X: np.ndarray,
        nbrs: Optional[List[List[int]]] = None,
        wts: Optional[List[List[float]]] = None,
        adj: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out, _ = self.forward(X, nbrs, wts, adj=adj)
        return out

    def predict_phases(
        self,
        X: np.ndarray,
        nbrs: Optional[List[List[int]]] = None,
        wts: Optional[List[List[float]]] = None,
        adj: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        z = self.predict_cos_sin(X, nbrs, wts, adj=adj)
        return np.arctan2(z[:, 1], z[:, 0])

    def loss_and_backward(
        self,
        X: np.ndarray,
        nbrs: Optional[List[List[int]]] = None,
        wts: Optional[List[List[float]]] = None,
        phase_true: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        adj: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        edge_weight: Optional[np.ndarray] = None,
        triplet_weight: float = 0.0,
    ) -> Tuple[float, dict]:
        out, cache = self.forward(X, nbrs, wts, adj=adj)
        if phase_true is None:
            raise ValueError("phase_true required for supervised loss")
        ut = np.column_stack([np.cos(phase_true), np.sin(phase_true)])
        if weights is None:
            w = np.ones(len(X))
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = w / (w.mean() + 1e-16)
        diff = out - ut
        N = max(len(X), 1)
        loss = 0.5 * np.mean(w * np.sum(diff ** 2, axis=1))
        nrm = np.linalg.norm(out, axis=1)
        loss = loss + 0.05 * np.mean((nrm - 1.0) ** 2)

        dout = (w[:, None] * diff) / N
        scale = 0.05 * 2.0 * (nrm - 1.0) / (N * (nrm + 1e-16))
        dout = dout + scale[:, None] * out

        if (
            triplet_weight > 0
            and edges is not None
            and len(edges) > 0
            and edge_weight is not None
        ):
            t_loss, t_dout = triplet_loss_and_grad(
                out, edges, edge_weight, phase_true=phase_true
            )
            loss = loss + triplet_weight * t_loss
            dout = dout + triplet_weight * t_dout

        h = cache["h_final"]
        A = cache["A"]
        residual = bool(cache.get("residual", self.residual))
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

        for ell in range(self.n_layers - 1, -1, -1):
            z = cache["zs"][ell]
            h_prev = cache["hs"][ell]
            agg = cache["aggs"][ell]
            # residual: h = h_prev + relu(z)  →  dh_act = dh, dh_prev gets skip
            dz = dh * _relu_grad(z)
            dW_self = h_prev.T @ dz
            dW_msg = agg.T @ dz
            db = dz.sum(axis=0)
            grads["W_self"].insert(0, dW_self)
            grads["W_msg"].insert(0, dW_msg)
            grads["b_h"].insert(0, db)

            dh_from_lin = dz @ self.W_self[ell].T
            dagg = dz @ self.W_msg[ell].T
            # agg = A @ h_prev  →  dh_prev += A.T @ dagg
            dh_prev = dh_from_lin + A.T @ dagg
            if residual:
                dh = dh + dh_prev  # skip connection + path through layer
            else:
                dh = dh_prev

        z0 = cache["z0"]
        dz0 = dh * _relu_grad(z0)
        dW_in = cache["X"].T @ dz0
        db_in = dz0.sum(axis=0)
        grads["W_in"] = dW_in
        grads["b_in"] = db_in
        return float(loss), grads

    def _init_adam(self) -> None:
        self._adam_m = {
            "W_in": np.zeros_like(self.W_in),
            "b_in": np.zeros_like(self.b_in),
            "W_out": np.zeros_like(self.W_out),
            "b_out": np.zeros_like(self.b_out),
            "W_self": [np.zeros_like(w) for w in self.W_self],
            "W_msg": [np.zeros_like(w) for w in self.W_msg],
            "b_h": [np.zeros_like(b) for b in self.b_h],
        }
        self._adam_v = {
            "W_in": np.zeros_like(self.W_in),
            "b_in": np.zeros_like(self.b_in),
            "W_out": np.zeros_like(self.W_out),
            "b_out": np.zeros_like(self.b_out),
            "W_self": [np.zeros_like(w) for w in self.W_self],
            "W_msg": [np.zeros_like(w) for w in self.W_msg],
            "b_h": [np.zeros_like(b) for b in self.b_h],
        }
        self._adam_t = 0

    def step(
        self,
        grads: dict,
        lr: float = 1e-3,
        clip: float = 5.0,
        *,
        optimizer: str = "adam",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        def _clip(g: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(g)
            if n > clip and n > 0:
                return g * (clip / n)
            return g

        g_in = _clip(grads["W_in"])
        g_bin = _clip(grads["b_in"])
        g_out = _clip(grads["W_out"])
        g_bout = _clip(grads["b_out"])
        g_self = [_clip(g) for g in grads["W_self"]]
        g_msg = [_clip(g) for g in grads["W_msg"]]
        g_bh = [_clip(g) for g in grads["b_h"]]

        if optimizer == "sgd":
            self.W_in -= lr * g_in
            self.b_in -= lr * g_bin
            for ell in range(self.n_layers):
                self.W_self[ell] -= lr * g_self[ell]
                self.W_msg[ell] -= lr * g_msg[ell]
                self.b_h[ell] -= lr * g_bh[ell]
            self.W_out -= lr * g_out
            self.b_out -= lr * g_bout
            return

        # Adam (default)
        if self._adam_m is None or self._adam_v is None:
            self._init_adam()
        assert self._adam_m is not None and self._adam_v is not None
        self._adam_t += 1
        t = self._adam_t
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t

        def _adam_update(param: np.ndarray, g: np.ndarray, m: np.ndarray, v: np.ndarray) -> None:
            m[:] = beta1 * m + (1.0 - beta1) * g
            v[:] = beta2 * v + (1.0 - beta2) * (g * g)
            mhat = m / bc1
            vhat = v / bc2
            param -= lr * mhat / (np.sqrt(vhat) + eps)

        _adam_update(self.W_in, g_in, self._adam_m["W_in"], self._adam_v["W_in"])
        _adam_update(self.b_in, g_bin, self._adam_m["b_in"], self._adam_v["b_in"])
        for ell in range(self.n_layers):
            _adam_update(
                self.W_self[ell], g_self[ell],
                self._adam_m["W_self"][ell], self._adam_v["W_self"][ell],
            )
            _adam_update(
                self.W_msg[ell], g_msg[ell],
                self._adam_m["W_msg"][ell], self._adam_v["W_msg"][ell],
            )
            _adam_update(
                self.b_h[ell], g_bh[ell],
                self._adam_m["b_h"][ell], self._adam_v["b_h"][ell],
            )
        _adam_update(self.W_out, g_out, self._adam_m["W_out"], self._adam_v["W_out"])
        _adam_update(self.b_out, g_bout, self._adam_m["b_out"], self._adam_v["b_out"])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "d_in": self.d_in,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "seed": self.seed,
            "residual": np.array(int(self.residual)),
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
        if hasattr(self, "_meta_extra"):
            for k, v in self._meta_extra.items():  # type: ignore[attr-defined]
                payload[k] = v
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "GraphPhaseNet":
        z = np.load(path, allow_pickle=True)
        residual = True
        if "residual" in z.files:
            residual = bool(int(np.asarray(z["residual"]).reshape(-1)[0]))
        m = cls(
            d_in=int(z["d_in"]),
            hidden=int(z["hidden"]),
            n_layers=int(z["n_layers"]),
            seed=int(z["seed"]),
            residual=residual,
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
    """Build graph + features + dense adj for one structure."""
    graph = reflection_graph(
        hkl, amplitudes, cell, e_min=e_min, max_reflections=max_reflections
    )
    X = node_features_from_graph(graph, hkl, amplitudes, cell)
    n = X.shape[0]
    edges = graph["edges"]
    ewt = graph["edge_weight"]
    nbrs, wts = build_undirected_adj(n, edges, ewt)
    adj = build_normalized_adj(n, edges, ewt)
    idx = graph["node_idx"]
    return {
        "X": X,
        "nbrs": nbrs,
        "wts": wts,
        "adj": adj,
        "edges": edges,
        "edge_weight": ewt,
        "node_idx": idx,
        "phases_strong": None,
        "amp_strong": amplitudes[idx],
        "hkl_strong": hkl[idx],
        "n_edges": len(edges),
    }
