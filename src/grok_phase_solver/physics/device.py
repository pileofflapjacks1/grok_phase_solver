"""
Optional device / FFT backend selection (CPU NumPy default; Torch GPU/MPS).

Physics fidelity: same mathematical transforms; only the backend changes.
Graceful fallback when torch / CUDA / MPS is unavailable.

Usage
-----
    from grok_phase_solver.physics.device import resolve_device, ifftn, fftn

    dev = resolve_device("auto")  # "cpu" | "cuda" | "mps"
    rho_c = ifftn(F_grid, device=dev)
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

import numpy as np

DeviceName = Literal["cpu", "cuda", "mps", "auto"]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def list_devices() -> dict:
    """Report available compute backends."""
    out: dict = {"numpy": True, "torch": False, "cuda": False, "mps": False}
    try:
        import torch

        out["torch"] = True
        out["cuda"] = bool(torch.cuda.is_available())
        out["mps"] = bool(
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
    except Exception:
        pass
    return out


def resolve_device(device: Optional[str] = None) -> str:
    """
    Resolve ``auto`` / user request to a concrete backend name.

    Defaults to ``cpu`` (NumPy). ``auto`` prefers cuda → mps → cpu.
    """
    req = (device or "cpu").lower().strip()
    if req in ("", "numpy"):
        return "cpu"
    if req == "cpu":
        return "cpu"
    avail = list_devices()
    if req == "auto":
        if avail.get("cuda"):
            return "cuda"
        if avail.get("mps"):
            return "mps"
        return "cpu"
    if req == "cuda":
        return "cuda" if avail.get("cuda") else "cpu"
    if req == "mps":
        return "mps" if avail.get("mps") else "cpu"
    # unknown → cpu
    return "cpu"


def _to_torch(arr: np.ndarray, device: str):
    import torch

    t = torch.as_tensor(np.asarray(arr))
    if np.iscomplexobj(arr):
        t = t.to(torch.complex128 if arr.dtype == np.complex128 else torch.complex64)
    else:
        t = t.to(torch.float64 if arr.dtype == np.float64 else torch.float32)
    return t.to(device)


def ifftn(x: np.ndarray, device: str = "cpu") -> np.ndarray:
    """n-D inverse FFT; returns NumPy array."""
    if device == "cpu" or not torch_available():
        return np.fft.ifftn(x)
    try:
        import torch

        t = _to_torch(x, device)
        y = torch.fft.ifftn(t)
        return y.detach().cpu().numpy()
    except Exception:
        return np.fft.ifftn(x)


def fftn(x: np.ndarray, device: str = "cpu") -> np.ndarray:
    """n-D forward FFT; returns NumPy array."""
    if device == "cpu" or not torch_available():
        return np.fft.fftn(x)
    try:
        import torch

        t = _to_torch(x, device)
        y = torch.fft.fftn(t)
        return y.detach().cpu().numpy()
    except Exception:
        return np.fft.fftn(x)


def get_device_info(device: Optional[str] = None) -> dict:
    """Diagnostics for report.md / CLI."""
    resolved = resolve_device(device)
    return {
        "requested": device or "cpu",
        "resolved": resolved,
        "available": list_devices(),
        "torch": torch_available(),
    }
