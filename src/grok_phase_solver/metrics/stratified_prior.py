"""
Stratified hold-out reporting for graph priors (v0.10).

Breaks seed-quality metrics by heaviest element (Z), organic vs HA-bearing,
and centrosymmetric vs P1 — aligned with GraPhAI reporting style without
claiming published GraPhAI numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "CL": 17, "Cl": 17, "BR": 35, "Br": 35, "I": 53,
    "FE": 26, "Fe": 26, "ZN": 30, "Zn": 30,
}


def max_Z_from_elements(elements: Sequence[str]) -> int:
    z = 1
    for e in elements:
        z = max(z, int(_Z.get(e, _Z.get(str(e).upper(), 6))))
    return z


def is_ha_bearing(elements: Sequence[str], z_min: int = 17) -> bool:
    return max_Z_from_elements(elements) >= z_min


def stratify_holdout_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate hold-out rows that include optional keys:
    max_Z, ha_bearing, space_group, frac_within_20, seedOK, strong_mpe_oi.
    """
    def _agg(sub: List[Dict]) -> Dict[str, Any]:
        if not sub:
            return {"n": 0}
        frac = [float(r["frac_within_20"]) for r in sub if "frac_within_20" in r]
        mpe = [float(r["strong_mpe_oi"]) for r in sub if r.get("strong_mpe_oi") is not None]
        sok = [1.0 if r.get("seedOK") else 0.0 for r in sub]
        return {
            "n": len(sub),
            "mean_frac_within_20": float(np.mean(frac)) if frac else float("nan"),
            "mean_strong_mpe_oi": float(np.mean(mpe)) if mpe else float("nan"),
            "seedOK_rate": float(np.mean(sok)) if sok else float("nan"),
        }

    ha = [r for r in rows if r.get("ha_bearing")]
    org = [r for r in rows if not r.get("ha_bearing")]
    centro = [
        r for r in rows
        if "P-1" in str(r.get("space_group", "")).upper().replace(" ", "")
        or r.get("centrosymmetric")
    ]
    p1 = [r for r in rows if r not in centro]
    z19 = [r for r in rows if int(r.get("max_Z", 0)) >= 19]
    z_lo = [r for r in rows if int(r.get("max_Z", 0)) < 19]

    return {
        "all": _agg(rows),
        "ha_bearing_Zge17": _agg(ha),
        "organic_light": _agg(org),
        "max_Z_ge19": _agg(z19),
        "max_Z_lt19": _agg(z_lo),
        "centrosymmetric": _agg(centro),
        "non_centrosymmetric_panel": _agg(p1),
        "note": (
            "Stratified synthetic hold-out (GraPhAI-style reporting). "
            "Not a claim of published GraPhAI COD/Z≥19 success rates."
        ),
    }


def format_stratified_md(strat: Dict[str, Any]) -> str:
    lines = [
        "## Stratified seed quality (v0.10)",
        "",
        "| Cohort | n | frac≤20° | seedOK | strong MPE |",
        "|--------|---|----------|--------|------------|",
    ]
    for key in (
        "all",
        "ha_bearing_Zge17",
        "organic_light",
        "max_Z_ge19",
        "max_Z_lt19",
        "centrosymmetric",
        "non_centrosymmetric_panel",
    ):
        s = strat.get(key) or {}
        if not s.get("n"):
            lines.append(f"| `{key}` | 0 | — | — | — |")
            continue
        frac = s.get("mean_frac_within_20", float("nan"))
        sok = s.get("seedOK_rate", float("nan"))
        mpe = s.get("mean_strong_mpe_oi", float("nan"))
        lines.append(
            f"| `{key}` | {s['n']} | **{100*frac:.1f}%** | {100*sok:.1f}% | {mpe:.1f}° |"
        )
    lines.append("")
    lines.append(strat.get("note", ""))
    lines.append("")
    return "\n".join(lines)
