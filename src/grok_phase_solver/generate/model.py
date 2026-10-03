"""Small coordinate denoiser. Torch is imported only when this module is imported.

Training objective (also in ``docs/math/generative_structure.md``):

    x_t = (x_0 + σ ε) mod 1
    L   = mean || wrap(x̂_0 - x_0) ||²

The network predicts a fractional delta. The last linear layer starts at zero,
so an untrained step predicts no coordinate change. There is no |F| term.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from grok_phase_solver.generate.geometry import element_index
from grok_phase_solver.generate.noise import SAMPLE_STEPS, SIGMA_MAX, SIGMA_MIN, sigma_schedule


def _torch():
    import torch

    return torch


def _as_tensor(values, dtype):
    """Build a CPU tensor from a list. Avoids the torch/NumPy bridge."""
    torch = _torch()
    data = np.asarray(values, dtype=np.float64).tolist()
    return torch.tensor(data, dtype=dtype)


def build_denoiser(hidden: int = 32):
    """Construct a CPU ``CoordDenoiser`` module. Last layer weights are zero."""
    torch = _torch()
    n_elem = 12

    class _Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = int(hidden)
            self.embed = torch.nn.Embedding(n_elem, self.hidden)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.hidden + 1 + 6 + 3, self.hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hidden, self.hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(self.hidden, 3),
            )
            torch.nn.init.zeros_(self.mlp[-1].weight)
            torch.nn.init.zeros_(self.mlp[-1].bias)

        def forward(self, fracs, elem_idx, sigma, cell):
            n = fracs.shape[0]
            emb = self.embed(elem_idx)
            delta = fracs[:, None, :] - fracs[None, :, :]
            delta = delta - torch.round(delta)
            dist2 = (delta * delta).sum(-1)
            eye = torch.eye(n, device=fracs.device, dtype=fracs.dtype)
            weight = torch.exp(-dist2 / (2.0 * 0.12**2)) * (1.0 - eye)
            weight = weight / (weight.sum(-1, keepdim=True) + 1e-6)
            neigh = (weight[..., None] * delta).sum(1)
            sig = sigma.reshape(1, 1).expand(n, 1)
            cell_b = cell.reshape(1, 6).expand(n, 6).clone()
            cell_b[:, :3] = cell_b[:, :3] / 20.0
            cell_b[:, 3:] = cell_b[:, 3:] / 180.0
            feat = torch.cat([emb, sig, cell_b, neigh], dim=-1)
            shift = self.mlp(feat)
            return torch.remainder(fracs + shift, 1.0)

    return _Net()


def denoiser_loss(model, fracs, elem_idx, cell):
    """One-structure minimum-image MSE. σ is log-uniform on [SIGMA_MIN, SIGMA_MAX]."""
    torch = _torch()
    log_sigma = torch.empty((), dtype=fracs.dtype).uniform_(
        math.log(SIGMA_MIN), math.log(SIGMA_MAX)
    )
    sigma = log_sigma.exp()
    eps = torch.randn_like(fracs)
    x_t = torch.remainder(fracs + sigma * eps, 1.0)
    pred = model(x_t, elem_idx, sigma.detach(), cell)
    err = pred - fracs
    err = err - torch.round(err)
    return (err * err).mean()


def train_denoiser(
    examples: Sequence[Mapping],
    *,
    steps: int = 40,
    hidden: int = 32,
    seed: int = 0,
    lr: float = 1e-3,
):
    """Fit on known fractional coordinates. Returns an eval-mode module.

    ``examples`` items need ``fracs`` (N, 3), ``elements``, and ``cell`` (6,).
    Batch size is 1 because N varies. Seed is ``torch.manual_seed(seed)``.
    """
    torch = _torch()
    torch.manual_seed(int(seed))
    usable = [ex for ex in examples if len(ex["elements"]) > 0]
    if not usable:
        raise ValueError("train_denoiser needs at least one non-empty structure")
    model = build_denoiser(hidden=hidden)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    last = 0.0
    n_steps = int(steps)
    for step in range(max(0, n_steps)):
        ex = usable[step % len(usable)]
        fracs = _as_tensor(ex["fracs"], torch.float32)
        elem = torch.tensor(
            [element_index(e) for e in ex["elements"]], dtype=torch.long
        )
        cell = _as_tensor(ex["cell"], torch.float32)
        loss = denoiser_loss(model, fracs, elem, cell)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    model.eval()
    model.final_loss = last  # type: ignore[attr-defined]
    model.train_steps = n_steps  # type: ignore[attr-defined]
    model.train_seed = int(seed)  # type: ignore[attr-defined]
    return model


def sample_denoiser(
    model,
    elements: Sequence[str],
    cell: np.ndarray,
    *,
    n: int,
    seed: int,
    n_steps: int = SAMPLE_STEPS,
) -> np.ndarray:
    """Draw ``n`` fractional coordinate sets, shape ``(n, N, 3)``.

    Starts from uniform coordinates and steps σ from high to low. The last step
    is the predicted clean coordinate. ``torch.manual_seed(seed)`` fixes the draws.
    """
    torch = _torch()
    torch.manual_seed(int(seed))
    model.eval()
    idx = torch.tensor([element_index(e) for e in elements], dtype=torch.long)
    cell_t = _as_tensor(cell, torch.float32)
    sigmas = _as_tensor(sigma_schedule(n_steps), torch.float32)
    n_atoms = int(idx.shape[0])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(int(n)):
            x = torch.rand((n_atoms, 3))
            for i, sigma in enumerate(sigmas):
                x0 = model(x, idx, sigma, cell_t)
                if i == len(sigmas) - 1:
                    x = x0
                else:
                    err = x - x0
                    err = err - torch.round(err)
                    eps = err / sigma.clamp_min(1e-4)
                    x = torch.remainder(x0 + sigmas[i + 1] * eps, 1.0)
            out.append(np.asarray(x.detach().cpu().tolist(), dtype=np.float64))
    if not out:
        return np.zeros((0, n_atoms, 3), dtype=np.float64)
    return np.stack(out, axis=0)


def denoiser_state(model) -> dict:
    """CPU state dict plus the hidden width, for ``train_generate.py``."""
    return {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "hidden": int(model.hidden),
        "steps": int(getattr(model, "train_steps", 0)),
        "seed": int(getattr(model, "train_seed", 0)),
        "final_loss": float(getattr(model, "final_loss", float("nan"))),
    }


def load_denoiser(state: Mapping):
    """Rebuild a denoiser from ``denoiser_state``."""
    model = build_denoiser(hidden=int(state["hidden"]))
    model.load_state_dict(state["state_dict"])
    model.eval()
    model.train_steps = int(state.get("steps", 0))  # type: ignore[attr-defined]
    model.train_seed = int(state.get("seed", 0))  # type: ignore[attr-defined]
    model.final_loss = float(state.get("final_loss", float("nan")))  # type: ignore[attr-defined]
    return model
