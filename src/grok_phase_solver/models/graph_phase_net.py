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


def node_features_from_graph(
    graph: Dict,
    hkl: np.ndarray,
    amp: np.ndarray,
    cell: np.ndarray,
    *,
    feature_version: int = 5,
) -> np.ndarray:
    """
    Node features for GraphPhaseNet.

    **v4 (d_in=10):**
    ``[E, s_norm, s², |h|_norm, amp_norm, h/hmax, k/kmax, l/lmax, deg_norm, E²_norm]``

    **v5 / v5.1 (d_in=14)** — Melgalvis/Rekis GraPhAI-inspired diffraction-graph:
    v4 + ``[shell_rank, log1p(E·deg), local_E_mean, |F|/⟨|F|⟩_shell]``

    - shell_rank: resolution-shell percentile of |E| (Wilson-aware ranking)
    - log1p(E·deg): couples strong reflections with triplet connectivity
    - local_E_mean: mean |E| of graph neighbors (message-ready structural cue)
    - shell-normalized |F|: amplitude vs local Wilson shell mean

    v5.1 (same d_in): κ-gated edges + optional higher-κ edge emphasis in
    ``prepare_graph_batch`` (GraPhAI-style physics edges without expanding d_in).

    **v6 (d_in=18)** — GraPhAI / HA-aware extensions (v0.8):
    v5 + ``[ha_E_tail, low_res_w, E·low_res, κ_centrality]``

    - ha_E_tail: soft strong-|E| tail (heavy-atom sensitive outliers)
    - low_res_w: (1 − s_norm) low-resolution weight (HA dominate low-s)
    - E·low_res: couples strong |E| with low-s (GraPhAI HA panels)
    - κ_centrality: incident triplet-κ mass / max (physics edge centrality)

    **v7 (d_in=22)** — GraPhAI-aligned multipath cues (v0.9):
    v6 + ``[hop2_local_E, edge_E_geom, wilson_E_shell, centro_HA_cue]``

    - hop2_local_E: 2-hop mean |E| (deeper graph context)
    - edge_E_geom: mean √(E_i E_j) over incident triplet pairs
    - wilson_E_shell: E / shell-mean E (Wilson residual)
    - centro_HA_cue: ha_E_tail · low_res · shell_rank (Z≥19 / centro path)
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
    base = np.column_stack(
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
    if int(feature_version) < 5:
        return base

    # --- v5 extras ---
    # Resolution-shell rank of |E| (0–1 within equal-count shells)
    order = np.argsort(s)
    shell_rank = np.zeros(n, dtype=np.float64)
    n_shells = max(4, min(12, n // 8 + 1))
    edges_s = np.linspace(0, n, n_shells + 1, dtype=int)
    for si in range(n_shells):
        sl = order[edges_s[si] : edges_s[si + 1]]
        if len(sl) == 0:
            continue
        # rank within shell
        r = np.argsort(np.argsort(E[sl])).astype(np.float64)
        shell_rank[sl] = r / max(len(sl) - 1, 1)

    # Neighbor-mean |E| via adjacency lists
    local_E = np.zeros(n, dtype=np.float64)
    if edges is not None and len(edges) > 0:
        nbr_sum = np.zeros(n, dtype=np.float64)
        nbr_cnt = np.zeros(n, dtype=np.float64)
        for e in np.asarray(edges):
            i, j, k = int(e[0]), int(e[1]), int(e[2])
            for a, b in ((i, j), (i, k), (j, k)):
                if a == b or a < 0 or b < 0 or a >= n or b >= n:
                    continue
                nbr_sum[a] += E[b]
                nbr_cnt[a] += 1.0
                nbr_sum[b] += E[a]
                nbr_cnt[b] += 1.0
        local_E = nbr_sum / np.maximum(nbr_cnt, 1.0)
    else:
        local_E = E.copy()
    local_E_n = local_E / (local_E.max() + 1e-16)

    # |F| / shell mean |F|
    shell_amp = np.zeros(n, dtype=np.float64)
    for si in range(n_shells):
        sl = order[edges_s[si] : edges_s[si + 1]]
        if len(sl) == 0:
            continue
        m = float(np.mean(amp_s[sl]) + 1e-16)
        shell_amp[sl] = amp_s[sl] / m

    e_deg = np.log1p(np.maximum(E, 0.0) * np.maximum(deg, 0.0))
    e_deg = e_deg / (e_deg.max() + 1e-16)

    v5 = np.column_stack(
        [
            base,
            shell_rank,
            e_deg,
            local_E_n,
            shell_amp,
        ]
    ).astype(np.float64)
    if int(feature_version) < 6:
        return v5

    # --- v6 extras (GraPhAI HA / centrosymmetric-friendly cues) ---
    # Soft strong-|E| tail: reflections that often carry HA information
    ha_E_tail = np.maximum(E - 1.8, 0.0)
    ha_E_tail = ha_E_tail / (ha_E_tail.max() + 1e-16)

    low_res_w = 1.0 - s_n  # larger at low resolution
    e_low = E * low_res_w
    e_low = e_low / (e_low.max() + 1e-16)

    # Incident κ (edge_weight) mass as centrality in the triplet graph
    kappa_c = np.zeros(n, dtype=np.float64)
    ewt = graph.get("edge_weight")
    if edges is not None and ewt is not None and len(edges) > 0:
        ew = np.asarray(ewt, dtype=np.float64)
        for e, w in zip(np.asarray(edges), ew):
            i, j, k = int(e[0]), int(e[1]), int(e[2])
            ww = float(max(w, 0.0))
            for a in (i, j, k):
                if 0 <= a < n:
                    kappa_c[a] += ww
    kappa_c = kappa_c / (kappa_c.max() + 1e-16)

    v6 = np.column_stack(
        [
            v5,
            ha_E_tail,
            low_res_w,
            e_low,
            kappa_c,
        ]
    ).astype(np.float64)
    if int(feature_version) < 7:
        return v6

    # --- v7 extras (GraPhAI multipath / Wilson residual) ---
    # 2-hop local |E| via one more adjacency multiply on 1-hop local_E
    hop2 = local_E.copy()
    if edges is not None and len(edges) > 0:
        nbr_sum2 = np.zeros(n, dtype=np.float64)
        nbr_cnt2 = np.zeros(n, dtype=np.float64)
        for e in np.asarray(edges):
            i, j, k = int(e[0]), int(e[1]), int(e[2])
            for a, b in ((i, j), (i, k), (j, k)):
                if a == b or a < 0 or b < 0 or a >= n or b >= n:
                    continue
                nbr_sum2[a] += local_E[b]
                nbr_cnt2[a] += 1.0
                nbr_sum2[b] += local_E[a]
                nbr_cnt2[b] += 1.0
        hop2 = nbr_sum2 / np.maximum(nbr_cnt2, 1.0)
    hop2_n = hop2 / (hop2.max() + 1e-16)

    # Geometric mean of |E| on incident undirected edges
    edge_geom = np.zeros(n, dtype=np.float64)
    edge_cnt = np.zeros(n, dtype=np.float64)
    if edges is not None and len(edges) > 0:
        for e in np.asarray(edges):
            i, j, k = int(e[0]), int(e[1]), int(e[2])
            for a, b in ((i, j), (i, k), (j, k)):
                if a == b or a < 0 or b < 0 or a >= n or b >= n:
                    continue
                g = float(np.sqrt(max(E[a], 0.0) * max(E[b], 0.0)))
                edge_geom[a] += g
                edge_geom[b] += g
                edge_cnt[a] += 1.0
                edge_cnt[b] += 1.0
        edge_geom = edge_geom / np.maximum(edge_cnt, 1.0)
    edge_geom = edge_geom / (edge_geom.max() + 1e-16)

    # Wilson residual: E / shell mean E
    shell_E_mean = np.ones(n, dtype=np.float64)
    for si in range(n_shells):
        sl = order[edges_s[si] : edges_s[si + 1]]
        if len(sl) == 0:
            continue
        shell_E_mean[sl] = float(np.mean(E[sl]) + 1e-16)
    wilson_E = E / shell_E_mean
    wilson_E = wilson_E / (wilson_E.max() + 1e-16)

    centro_ha = ha_E_tail * low_res_w * shell_rank
    centro_ha = centro_ha / (centro_ha.max() + 1e-16)

    return np.column_stack(
        [
            v6,
            hop2_n,
            edge_geom,
            wilson_E,
            centro_ha,
        ]
    ).astype(np.float64)


def phase_bin_cross_entropy(
    out: np.ndarray,
    phase_true: np.ndarray,
    *,
    n_bins: int = 4,
    mode: str = "bins",
) -> Tuple[float, np.ndarray]:
    """
    Carrozzini-style discretized phase classification loss on (cos, sin) heads.

    Soft-assigns predicted phases to bins via circular proximity; CE vs true bin.
    Returns (loss, dout) with dout shape matching ``out`` (N, 2).

    ``mode``: ``bins`` (n_bins on circle) or ``centro`` (2 bins: 0 / π).
    """
    out = np.asarray(out, dtype=np.float64)
    ph_t = np.asarray(phase_true, dtype=np.float64)
    n = out.shape[0]
    dout = np.zeros_like(out)
    if n == 0:
        return 0.0, dout

    # predicted angle from (cos, sin) logits (treat as unnormalized direction)
    c, s = out[:, 0], out[:, 1]
    ph_p = np.arctan2(s, c + 1e-16)

    if mode == "centro":
        n_bins = 2
        # bins: 0 and π represented by cos sign
        # true label: 0 if cos(true)>=0 else 1
        y = (np.cos(ph_t) < 0).astype(int)
        # soft pred: p1 = sigmoid(-cos_pred) ≈ P(π)
        logits = -c  # large positive → prefer π
        # binary CE
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        yf = y.astype(np.float64)
        loss = float(-np.mean(yf * np.log(p1) + (1 - yf) * np.log(1 - p1)))
        # dL/dlogit = (p1 - y) / n
        dlog = (p1 - yf) / n
        # logit = -c → dc = -dlog
        dout[:, 0] = -dlog
        return loss, dout

    # multi-bin on circle
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    ph_tw = (ph_t + np.pi) % (2 * np.pi) - np.pi
    y = np.digitize(ph_tw, edges[1:-1])
    y = np.clip(y, 0, n_bins - 1)

    # soft assignment: logits_k = cos(ph_p - mid_k)
    # L = -mean log softmax_y
    # d/dph_p via chain rule, then d cos/sin
    loss_acc = 0.0
    for i in range(n):
        logits = np.cos(ph_p[i] - mids) * 3.0  # temperature
        logits = logits - logits.max()
        ex = np.exp(logits)
        sm = ex / (ex.sum() + 1e-16)
        yi = int(y[i])
        loss_acc += -float(np.log(sm[yi] + 1e-16))
        # dL/dlogit_k = sm_k - 1_{k=y}
        dlog = sm.copy()
        dlog[yi] -= 1.0
        # dlogit_k / dph = -3 sin(ph - mid_k)
        dph = float(np.sum(dlog * (-3.0 * np.sin(ph_p[i] - mids))))
        # ph = atan2(s,c); dph/dc = -s/(c²+s²), dph/ds = c/(c²+s²)
        den = c[i] ** 2 + s[i] ** 2 + 1e-16
        dout[i, 0] = dph * (-s[i] / den)
        dout[i, 1] = dph * (c[i] / den)
    loss = loss_acc / n
    dout = dout / n
    return loss, dout


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
        bin_weight: float = 0.0,
        n_phase_bins: int = 4,
        bin_mode: str = "bins",
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

        # Carrozzini-style discretized phase classification (v0.9)
        if bin_weight > 0 and phase_true is not None:
            b_loss, b_dout = phase_bin_cross_entropy(
                out, phase_true, n_bins=int(n_phase_bins), mode=bin_mode
            )
            loss = loss + float(bin_weight) * b_loss
            dout = dout + float(bin_weight) * b_dout

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
    feature_version: int = 5,
    kappa_power: float = 1.25,
    self_loop: float = 0.05,
) -> Dict:
    """Build graph + features + dense adj for one structure."""
    graph = reflection_graph(
        hkl, amplitudes, cell, e_min=e_min, max_reflections=max_reflections
    )
    X = node_features_from_graph(
        graph, hkl, amplitudes, cell, feature_version=feature_version
    )
    n = X.shape[0]
    edges = graph["edges"]
    ewt = graph["edge_weight"]
    nbrs, wts = build_undirected_adj(n, edges, ewt)
    adj = build_normalized_adj(n, edges, ewt)
    # Soft κ-gated reweight: boost high-κ edges (GraPhAI / Melgalvis physics edges)
    # v5.1 / v6: power-law emphasis on strongest triplets
    # v7: κ × √(E_i E_j E_k) multipath emphasis (GraPhAI-style reliability)
    kpow = float(kappa_power)
    if int(feature_version) >= 7:
        kpow = max(kpow, 1.45)
    elif int(feature_version) >= 6 and kpow <= 1.25:
        kpow = 1.35
    sl = float(self_loop)
    if int(feature_version) >= 7:
        sl = max(sl, 0.10)
    elif int(feature_version) >= 6:
        sl = max(sl, 0.08)
    if len(edges) > 0 and ewt is not None and len(ewt) == len(edges):
        w = np.asarray(ewt, dtype=np.float64)
        if int(feature_version) >= 7 and graph.get("E") is not None:
            E_n = np.asarray(graph["E"], dtype=np.float64)
            boost = np.ones(len(edges), dtype=np.float64)
            for ti, e in enumerate(np.asarray(edges)):
                i, j, k = int(e[0]), int(e[1]), int(e[2])
                if 0 <= i < n and 0 <= j < n and 0 <= k < n:
                    boost[ti] = float(
                        (max(E_n[i], 0.0) * max(E_n[j], 0.0) * max(E_n[k], 0.0) + 1e-16)
                        ** (1.0 / 6.0)
                    )
            w = w * (0.5 + 0.5 * boost / (boost.mean() + 1e-16))
        med = float(np.median(w) + 1e-16)
        w = np.power(np.clip(w / med, 0.15, 6.0), kpow)
        adj = build_normalized_adj(n, edges, w)
        # Add weak self-loops for numerical stability of residual MP
        adj = adj + sl * np.eye(n, dtype=np.float64)
        rs = adj.sum(axis=1, keepdims=True)
        adj = adj / np.maximum(rs, 1e-16)
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
        "feature_version": int(feature_version),
        "d_in": int(X.shape[1]),
    }
