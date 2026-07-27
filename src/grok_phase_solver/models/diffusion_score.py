"""
Trainable score network for phase-circle diffusion (v0.6).

Lightweight NumPy MLP that predicts a phase-score / denoising residual
conditioned on |F|/|E|, resolution, and noisy (cos φ, sin φ).

Architecture (physics-informed, not SE(3) atomic coordinates)
-------------------------------------------------------------
Input per reflection: [E, s_norm, amp_norm, cos_noisy, sin_noisy, t_emb]
Hidden MLP → (Δcos, Δsin) score used to step the reverse process.

Inspired by score-based generative models and diffraction-diffusion
concepts (PXRDnet-style consistency) but adapted to single-crystal |F|→φ.
No claim of equivariant atomic denoising parity.

Training: supervise score toward (cos_true − cos_noisy, sin_true − sin_noisy)
at random noise levels (denoising score matching).

Checkpoint: small ``.npz`` (default models/diffusion_score.npz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from grok_phase_solver.physics.reciprocal import d_spacing
from grok_phase_solver.solvers.direct_methods import normalize_E

PathLike = str | Path


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def reflection_score_features(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases_noisy: np.ndarray,
    t: float,
) -> np.ndarray:
    """(N, 6) features: E, s_n, amp_n, cos, sin, t."""
    hkl = np.asarray(hkl, dtype=float)
    amp = np.asarray(amplitudes, dtype=np.float64)
    ph = np.asarray(phases_noisy, dtype=np.float64)
    E = normalize_E(np.asarray(hkl, dtype=int), amp, cell)
    d = d_spacing(np.asarray(hkl, dtype=int), cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-6))
    s_n = s / (s.max() + 1e-16)
    amp_n = amp / (amp.std() + 1e-16)
    t_col = np.full(len(amp), float(t), dtype=np.float64)
    return np.column_stack(
        [E, s_n, amp_n, np.cos(ph), np.sin(ph), t_col]
    ).astype(np.float64)


@dataclass
class PhaseScoreNet:
    """Small MLP score network: features → (dcos, dsin)."""

    d_in: int = 6
    hidden: int = 64
    n_layers: int = 2
    seed: int = 0

    W: List[np.ndarray] = field(init=False)
    b: List[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        dims = [self.d_in] + [self.hidden] * self.n_layers + [2]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            self.W.append(
                rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
            )
            self.b.append(np.zeros(dims[i + 1]))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        X = np.asarray(X, dtype=np.float64)
        acts = [X]
        h = X
        zs = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            zs.append(z)
            if i < len(self.W) - 1:
                h = _relu(z)
            else:
                h = z
            acts.append(h)
        return h, {"acts": acts, "zs": zs}

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        out, _ = self.forward(X)
        return out

    def loss_and_grad(
        self, X: np.ndarray, target: np.ndarray
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray]]:
        out, cache = self.forward(X)
        target = np.asarray(target, dtype=np.float64)
        diff = out - target
        N = max(len(X), 1)
        loss = 0.5 * float(np.mean(np.sum(diff**2, axis=1)))
        dout = diff / N
        dW: List[np.ndarray] = []
        db: List[np.ndarray] = []
        dh = dout
        for i in range(len(self.W) - 1, -1, -1):
            h_in = cache["acts"][i]
            dW.insert(0, h_in.T @ dh)
            db.insert(0, dh.sum(axis=0))
            d_prev = dh @ self.W[i].T
            if i > 0:
                d_prev = d_prev * _relu_grad(cache["zs"][i - 1])
            dh = d_prev
        return loss, dW, db

    def sgd_step(
        self, X: np.ndarray, target: np.ndarray, lr: float = 1e-3
    ) -> float:
        loss, dW, db = self.loss_and_grad(X, target)
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]
        return loss

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "d_in": self.d_in,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "seed": self.seed,
            "version": "diffusion_score_v1",
        }
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            payload[f"W_{i}"] = W
            payload[f"b_{i}"] = b
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: PathLike) -> "PhaseScoreNet":
        z = np.load(path, allow_pickle=True)
        m = cls(
            d_in=int(z["d_in"]),
            hidden=int(z["hidden"]),
            n_layers=int(z["n_layers"]),
            seed=int(z["seed"]),
        )
        m.W, m.b = [], []
        i = 0
        while f"W_{i}" in z.files:
            m.W.append(z[f"W_{i}"])
            m.b.append(z[f"b_{i}"])
            i += 1
        return m


def default_score_path() -> Path:
    return Path(__file__).resolve().parents[3] / "models" / "diffusion_score.npz"


def score_weights_available(path: Optional[PathLike] = None) -> bool:
    p = Path(path) if path else default_score_path()
    # also check data/processed
    if p.is_file():
        return True
    alt = Path(__file__).resolve().parents[3] / "data" / "processed" / "diffusion_score.npz"
    return alt.is_file()


def load_score_net(path: Optional[PathLike] = None) -> Optional[PhaseScoreNet]:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.append(default_score_path())
    candidates.append(
        Path(__file__).resolve().parents[3] / "data" / "processed" / "diffusion_score.npz"
    )
    for p in candidates:
        if p.is_file():
            try:
                return PhaseScoreNet.load(p)
            except Exception:
                continue
    return None


def train_score_on_structures(
    n_structures: int = 80,
    epochs_per: int = 15,
    hidden: int = 64,
    seed: int = 0,
    max_refl: int = 120,
    sigma_max: float = 1.0,
    verbose: bool = True,
) -> Tuple[PhaseScoreNet, Dict]:
    """
    Train PhaseScoreNet on synthetic organics (denoising score matching).
    """
    from grok_phase_solver.data.synthetic import generate_random_organic
    from grok_phase_solver.solvers.baseline import structure_to_fcalc

    rng = np.random.default_rng(seed)
    net = PhaseScoreNet(hidden=hidden, n_layers=2, seed=seed)
    losses: List[float] = []

    for i in range(n_structures):
        st = generate_random_organic(
            n_atoms=int(rng.integers(8, 16)), seed=int(rng.integers(0, 2**31))
        )
        data = structure_to_fcalc(st, d_min=float(rng.uniform(1.2, 1.8)))
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        # subsample strong
        E = normalize_E(hkl, amp, st.cell)
        order = np.argsort(-E)[: min(max_refl, len(E))]
        hkl_s, amp_s, ph_s = hkl[order], amp[order], ph_t[order]

        for _ in range(epochs_per):
            t = float(rng.uniform(0.05, 1.0))
            sigma = sigma_max * t
            noise = rng.normal(0, sigma, size=len(ph_s))
            ph_n = ph_s + noise
            X = reflection_score_features(hkl_s, amp_s, st.cell, ph_n, t)
            # target: clean − noisy in cos/sin space
            target = np.column_stack(
                [np.cos(ph_s) - np.cos(ph_n), np.sin(ph_s) - np.sin(ph_n)]
            )
            loss = net.sgd_step(X, target, lr=2e-3 * (0.5 + 0.5 * (1 - t)))
            losses.append(loss)
        if verbose and (i % 20 == 0 or i == n_structures - 1):
            print(f"  score train {i+1}/{n_structures} loss≈{np.mean(losses[-epochs_per:]):.4f}")

    meta = {
        "n_structures": n_structures,
        "epochs_per": epochs_per,
        "hidden": hidden,
        "final_loss": float(np.mean(losses[-50:])) if losses else None,
        "algorithm": "phase_score_dsm",
    }
    return net, meta


def apply_score_step(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases: np.ndarray,
    net: PhaseScoreNet,
    t: float,
    step_size: float = 0.35,
) -> np.ndarray:
    """One learned denoising step: φ ← angle( e^{iφ} + α · score )."""
    X = reflection_score_features(hkl, amplitudes, cell, phases, t)
    score = net.predict_score(X)
    z = np.exp(1j * phases) + step_size * (score[:, 0] + 1j * score[:, 1])
    return np.angle(z)
