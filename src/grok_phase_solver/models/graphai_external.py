"""
Optional external GraPhAI weight loader / H2H harness (v0.13).

Official GraPhAI (Melgalvis & Rekis, JACS 2026) weights and code are **not
redistributed**. Users may point ``GRAPHAI_HOME`` or ``--graphai-dir`` at a
local Zenodo download.

This module provides:
- Path discovery and availability checks
- A thin adapter stub that raises a clear error until the user installs
  external code
- A scoreboard harness skeleton for fair H2H vs GraphPhaseNet

Physics / in-repo fallback: GraphPhaseNet strong prior + AI-PhaSeed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def default_graphai_search_paths() -> List[Path]:
    paths: List[Path] = []
    env = os.environ.get("GRAPHAI_HOME") or os.environ.get("GRAPHAI_DIR")
    if env:
        paths.append(Path(env).expanduser())
    # common local conventions (never ship contents)
    here = Path(__file__).resolve()
    roots = [here.parents[3], Path.cwd(), Path.home() / "models"]
    for r in roots:
        paths.append(r / "third_party" / "graphai")
        paths.append(r / "external" / "graphai")
        paths.append(r / "GraPhAI")
    return paths


def graphai_available(path: Optional[Path] = None) -> bool:
    """True if a user-supplied GraPhAI install appears present."""
    cands = [Path(path)] if path else default_graphai_search_paths()
    for p in cands:
        if p is None:
            continue
        p = Path(p)
        if not p.is_dir():
            continue
        # heuristic markers
        if (p / "weights").is_dir() or list(p.glob("*.pt")) or list(p.glob("*.ckpt")):
            return True
        if (p / "README.md").is_file() and any(p.glob("**/*graph*")):
            return True
    return False


def load_graphai_adapter(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Attempt to load external GraPhAI.

    Returns a dict with ``available``, ``path``, ``predict`` (callable or None),
    and ``note``. Never downloads weights.
    """
    cands = [Path(path)] if path else default_graphai_search_paths()
    for p in cands:
        if p is None:
            continue
        p = Path(p)
        if not p.is_dir():
            continue
        if not graphai_available(p):
            continue
        return {
            "available": True,
            "path": str(p),
            "predict": None,  # user must plug local API
            "note": (
                f"GraPhAI directory found at {p}. Wire local inference in a "
                "user plugin; this package does not ship GraPhAI code/weights."
            ),
        }
    return {
        "available": False,
        "path": None,
        "predict": None,
        "note": (
            "GraPhAI not found. Download from the official Zenodo release "
            "(Melgalvis & Rekis 2026), set GRAPHAI_HOME=/path/to/install. "
            "Weights are not redistributed by grok_phase_solver."
        ),
    }


def predict_phases_graphai_or_fallback(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    *,
    graphai_dir: Optional[str] = None,
    max_reflections: int = 140,
) -> Tuple[np.ndarray, Dict]:
    """
    Prefer external GraPhAI if adapter provides ``predict``; else GraphPhaseNet.

    Returns full-length phases (radians) and meta.
    """
    adapter = load_graphai_adapter(Path(graphai_dir) if graphai_dir else None)
    meta: Dict[str, Any] = {"graphai": adapter}
    if adapter.get("available") and callable(adapter.get("predict")):
        ph = adapter["predict"](hkl, amplitudes, cell)
        meta["source"] = "graphai_external"
        return np.asarray(ph, dtype=np.float64), meta

    # In-repo fallback
    try:
        from grok_phase_solver.models.strong_prior import (
            default_strong_prior_path,
            load_strong_prior,
            predict_full_phases,
        )

        path = default_strong_prior_path()
        if path.exists():
            model = load_strong_prior(path)
            ph = predict_full_phases(
                model, hkl, amplitudes, cell, max_reflections=max_reflections
            )
            meta["source"] = "graph_phase_net_fallback"
            meta["weights"] = str(path)
            return np.asarray(ph, dtype=np.float64), meta
    except Exception as e:
        meta["fallback_error"] = str(e)

    rng = np.random.default_rng(0)
    ph = rng.uniform(-np.pi, np.pi, size=len(amplitudes))
    meta["source"] = "random_fallback"
    meta["note"] = (meta.get("note") or "") + " No GraPhAI and no strong prior weights."
    return ph, meta


def write_graphai_h2h_skeleton(
    out_path: Path,
    rows: Optional[List[Dict]] = None,
) -> Path:
    """Write an empty/partial H2H scoreboard markdown for user fill-in."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GraPhAI vs GraphPhaseNet head-to-head (skeleton)",
        "",
        "Official GraPhAI weights are **external only** (Zenodo; not redistributed).",
        "",
        "| Dataset | GraphPhaseNet frac≤20° | GraPhAI frac≤20° | notes |",
        "|---------|------------------------|------------------|-------|",
    ]
    if rows:
        for r in rows:
            lines.append(
                f"| {r.get('dataset','?')} | {r.get('gpn','—')} | {r.get('graphai','—')} | {r.get('notes','')} |"
            )
    else:
        lines.append("| *(run with local GraPhAI)* | — | — | set GRAPHAI_HOME |")
    lines.extend(
        [
            "",
            "## Setup",
            "",
            "```bash",
            "export GRAPHAI_HOME=/path/to/GraPhAI  # user Zenodo download",
            "python scripts/run_graphai_h2h.py --quick  # when external API wired",
            "```",
            "",
            "Fallback without external install: GraphPhaseNet only.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))
    return out_path
