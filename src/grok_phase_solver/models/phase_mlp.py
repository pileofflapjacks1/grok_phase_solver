"""
Phase-prediction MLP (NumPy).

Predicts (cos φ, sin φ) targets with MSE. At inference, φ = atan2(sin, cos).

Loss identity:
  ½ ||(ĉ, ŝ) − (cos φ, sin φ)||² = 1 − cos(Δφ)   when (ĉ,ŝ) is unit length.
We train with MSE to (cos, sin) without hard projection (more stable grads);
optional L2 pull toward unit circle.

Honest scope: supervised synthetic learning / hybrid seeds — not a claimed
general experimental solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.reciprocal import d_spacing


def reflection_features(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    hkl = np.asarray(hkl, dtype=np.float64)
    amp = np.asarray(amplitudes, dtype=np.float64)
    d = d_spacing(hkl, cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-6))
    amp_n = amp / (np.std(amp) + 1e-16)
    hmax = np.maximum(np.abs(hkl).max(axis=0), 1.0)
    return np.column_stack(
        [
            amp_n,
            s / (s.max() + 1e-16),
            (s / (s.max() + 1e-16)) ** 2,
            hkl[:, 0] / hmax[0],
            hkl[:, 1] / hmax[1],
            hkl[:, 2] / hmax[2],
        ]
    ).astype(np.float64)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


@dataclass
class PhaseMLP:
    d_in: int = 6
    hidden: int = 64
    seed: int = 0
    W1: np.ndarray = field(init=False)
    b1: np.ndarray = field(init=False)
    W2: np.ndarray = field(init=False)
    b2: np.ndarray = field(init=False)
    W3: np.ndarray = field(init=False)
    b3: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, np.sqrt(2 / self.d_in), (self.d_in, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / self.hidden), (self.hidden, self.hidden))
        self.b2 = np.zeros(self.hidden)
        self.W3 = rng.normal(0, np.sqrt(2 / self.hidden), (self.hidden, 2)) * 0.1
        self.b3 = np.zeros(2)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        z1 = X @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return z3, {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}

    def predict_phases(self, X: np.ndarray) -> np.ndarray:
        z, _ = self.forward(X)
        return np.arctan2(z[:, 1], z[:, 0])

    def loss_and_backward(
        self,
        X: np.ndarray,
        phase_true: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[float, dict]:
        z, cache = self.forward(X)
        ut = np.column_stack([np.cos(phase_true), np.sin(phase_true)])
        if weights is None:
            w = np.ones(len(X))
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = w / (w.mean() + 1e-16)
        diff = z - ut
        # L = 0.5 mean_i w_i ||diff_i||^2
        loss = 0.5 * np.mean(w * np.sum(diff**2, axis=1))
        # unit-circle regularizer
        nrm = np.linalg.norm(z, axis=1)
        loss = loss + 0.05 * np.mean((nrm - 1.0) ** 2)
        # dL/dz
        N = len(X)
        dz = (w[:, None] * diff) / N
        # reg grad: d/dz 0.05 mean (||z||-1)^2 = 0.05 * 2 (||z||-1) * z/||z|| / N
        scale = 0.05 * 2.0 * (nrm - 1.0) / (N * (nrm + 1e-16))
        dz = dz + scale[:, None] * z

        a2 = cache["a2"]
        dW3 = a2.T @ dz
        db3 = dz.sum(axis=0)
        da2 = dz @ self.W3.T
        dz2 = da2 * _relu_grad(cache["z2"])
        dW2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * _relu_grad(cache["z1"])
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        grads = {
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3,
        }
        return float(loss), grads

    def step(self, grads: dict, lr: float = 1e-3) -> None:
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(self, name, getattr(self, name) - lr * grads[name])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            d_in=self.d_in, hidden=self.hidden, seed=self.seed,
        )

    @classmethod
    def load(cls, path: Path) -> "PhaseMLP":
        z = np.load(path)
        m = cls(d_in=int(z["d_in"]), hidden=int(z["hidden"]), seed=int(z["seed"]))
        m.W1, m.b1 = z["W1"], z["b1"]
        m.W2, m.b2 = z["W2"], z["b2"]
        m.W3, m.b3 = z["W3"], z["b3"]
        return m


def train_phase_mlp_on_structure(
    model: PhaseMLP,
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    cell: np.ndarray,
    n_epochs: int = 200,
    lr: float = 3e-3,
    batch_frac: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> List[float]:
    rng = np.random.default_rng(seed)
    X = reflection_features(hkl, amplitudes, cell)
    # standardize features
    mu, sig = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sig
    y = np.asarray(phases, dtype=np.float64)
    w = np.asarray(amplitudes, dtype=np.float64)
    losses: List[float] = []
    n = len(X)
    for ep in range(n_epochs):
        idx = np.arange(n)
        if batch_frac < 1.0:
            m = max(8, int(batch_frac * n))
            idx = rng.choice(n, size=m, replace=False)
        loss, grads = model.loss_and_backward(X[idx], y[idx], weights=w[idx])
        model.step(grads, lr=lr)
        losses.append(loss)
        if verbose and (ep % 50 == 0 or ep == n_epochs - 1):
            print(f"  epoch {ep:4d}  loss={loss:.4f}")
    # store norm stats on model for predict
    model._feat_mu = mu  # type: ignore[attr-defined]
    model._feat_sig = sig  # type: ignore[attr-defined]
    return losses
