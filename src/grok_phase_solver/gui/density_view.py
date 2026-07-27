"""
Lightweight density visualization helpers for the Streamlit GUI.

Uses matplotlib (always available) for central slices; optional plotly
3D isosurface / volume when installed (``pip install plotly``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def central_slices(density: np.ndarray) -> Dict[str, np.ndarray]:
    """Return xy, xz, yz mid-plane slices."""
    rho = np.asarray(density, dtype=np.float64)
    nx, ny, nz = rho.shape
    return {
        "xy": rho[:, :, nz // 2],
        "xz": rho[:, ny // 2, :],
        "yz": rho[nx // 2, :, :],
    }


def save_slice_figure(
    density: np.ndarray,
    path: PathLike,
    *,
    title: str = "Density mid-planes",
) -> Path:
    """Write a 1×3 matplotlib figure of mid-plane slices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    slices = central_slices(density)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, (name, sl) in zip(axes, slices.items()):
        im = ax.imshow(sl.T, origin="lower", cmap="magma")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plotly_volume_html(
    density: np.ndarray,
    path: PathLike,
    *,
    isomin_frac: float = 0.35,
    title: str = "Density (interactive)",
    max_dim: int = 48,
) -> Optional[Path]:
    """
    Write a plotly volume HTML viewer if plotly is installed.

    Downsamples large grids for browser performance.
    Returns path or None if plotly missing.
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rho = np.asarray(density, dtype=np.float64)
    # downsample
    step = tuple(max(1, s // max_dim) for s in rho.shape)
    rho_s = rho[:: step[0], :: step[1], :: step[2]]
    lo = float(np.percentile(rho_s, 50))
    hi = float(np.percentile(rho_s, 99))
    isomin = lo + isomin_frac * (hi - lo)
    X, Y, Z = np.mgrid[
        0 : rho_s.shape[0],
        0 : rho_s.shape[1],
        0 : rho_s.shape[2],
    ]
    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=rho_s.flatten(),
            isomin=isomin,
            isomax=hi,
            opacity=0.12,
            surface_count=8,
            colorscale="Magma",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def density_view_bundle(
    density: np.ndarray,
    out_dir: PathLike,
    *,
    prefix: str = "density",
) -> Dict[str, Any]:
    """Write slice PNG + optional plotly HTML; return paths dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = save_slice_figure(density, out / f"{prefix}_slices.png")
    html = plotly_volume_html(density, out / f"{prefix}_volume.html")
    return {
        "slices_png": str(png),
        "volume_html": str(html) if html else None,
        "plotly_available": html is not None,
        "shape": list(np.asarray(density).shape),
    }
