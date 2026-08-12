"""
Truth-free next-action chooser for gps-solve reports.

Uses unit-cell volume band + whether a seed was already applied + free-FOM
outlook. Numbers cited are the local COD Vol-band panel (C25), not a 1505-COD
replication. Free FOM is a ranking score, not proof of a correct structure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from grok_phase_solver.solvers.projectors import unit_cell_volume

CellLike = Union[Sequence[float], np.ndarray]

_SEED_METHODS = frozenset(
    {
        "partial_phaseed",
        "fragment_phaseed",
        "ha_phaseed",
        "diffusion_phaseed",
    }
)


def classify_vol_band(volume: float) -> str:
    """Return ``vol_lt_1000`` / ``vol_1000_3500`` / ``vol_gt_3500``."""
    v = float(volume)
    if v < 1000.0:
        return "vol_lt_1000"
    if v <= 3500.0:
        return "vol_1000_3500"
    return "vol_gt_3500"


def vol_band_label(band: str) -> str:
    return {
        "vol_lt_1000": "Vol < 1000 Å³ (small-cell)",
        "vol_1000_3500": "Vol 1000–3500 Å³ (hybrid-friendly)",
        "vol_gt_3500": "Vol > 3500 Å³ (large / hard)",
    }.get(band, "Vol unknown")


def _volume_from_cell(cell: Optional[CellLike]) -> Optional[float]:
    if cell is None:
        return None
    arr = np.asarray(cell, dtype=np.float64).ravel()
    if arr.size < 6 or not np.all(np.isfinite(arr[:6])):
        return None
    try:
        v = float(unit_cell_volume(arr[:6]))
    except Exception:
        return None
    if not np.isfinite(v) or v <= 0:
        return None
    return v


def _fom(diagnostics: Mapping[str, Any]) -> Optional[float]:
    raw = diagnostics.get("free_fom_composite")
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _already_seeded(method: str, diagnostics: Mapping[str, Any]) -> bool:
    m = (method or "").lower()
    if m in _SEED_METHODS:
        return True
    if diagnostics.get("seed_kind") or diagnostics.get("seed_source"):
        return True
    sq = diagnostics.get("seed_quality")
    if isinstance(sq, dict) and (
        sq.get("frac_strong_seeded") is not None or sq.get("n_seed")
    ):
        return True
    return False


def _map_outlook(fom: Optional[float], n_peaks: int) -> str:
    if fom is not None and fom >= 0.70 and n_peaks >= 8:
        return "looks_healthy"
    if fom is not None and fom >= 0.55:
        return "inspect"
    return "likely_unsolved"


def recommend_next_action(
    *,
    cell: Optional[CellLike],
    d_min: Optional[float],
    method: str,
    n_reflections: int = 0,
    n_peaks: int = 0,
    diagnostics: Optional[Mapping[str, Any]] = None,
    space_group: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Structured next action for report.md / GUI / solve_summary.json.

    Does not use ground-truth phases.
    """
    d = dict(diagnostics or {})
    vol = _volume_from_cell(cell)
    band = classify_vol_band(vol) if vol is not None else "unknown"
    fom = _fom(d)
    outlook = _map_outlook(fom, int(n_peaks or 0))
    seeded = _already_seeded(method, d)
    sq = d.get("seed_quality") if isinstance(d.get("seed_quality"), dict) else {}
    size_ok = sq.get("size_meets_bar")
    seed_class = sq.get("predicted_class")
    dmin = float(d_min) if d_min is not None else None

    rec: Dict[str, Any] = {
        "vol": None if vol is None else round(vol, 1),
        "vol_band": band,
        "vol_band_label": vol_band_label(band),
        "map_outlook": outlook,
        "already_seeded": seeded,
        "free_fom_composite": fom,
        "n_peaks": int(n_peaks or 0),
        "n_reflections": int(n_reflections or 0),
        "method": method,
        "d_min": dmin,
        "space_group": space_group,
        "seed_size_meets_bar": size_ok,
        "seed_predicted_class": seed_class,
        "evidence": "COD Vol-band panel (6 local structures; C25) + partial-φ 30%/20° bar",
    }

    if outlook == "looks_healthy":
        rec.update(
            {
                "primary_id": "refine_shelxl",
                "primary": (
                    "Map outlook looks healthy (truth-free). Inspect density_slice.png "
                    "and assign elements in trial.res, then refine with SHELXL / Olex2."
                ),
                "why": (
                    "Free-FOM composite is a ranking score, not proof. "
                    "Chemical sense + SHELXL R1 decide."
                ),
                "commands": [
                    "cp trial.res work.ins && cp your.hkl work.hkl && ShelX/shelxl work",
                ],
                "alternatives": [
                    "If chemistry looks wrong, treat as unsolved and add a fragment / HA seed.",
                ],
            }
        )
        return rec

    if seeded and size_ok is False:
        rec.update(
            {
                "primary_id": "enlarge_seed",
                "primary": (
                    "Seed coverage is below the ~30% strong-|E| size bar. "
                    "Enlarge it: more known φ, a heavier fragment, or HA sites."
                ),
                "why": (
                    "Oracle curves: extension works when ≥~30% of strong |E| phases "
                    "are correct within ~20°. Size is necessary but not sufficient."
                ),
                "commands": [
                    "gps-make-seed --hkl your.hkl --ins your.ins --from-res model.res -o seed.csv",
                    "gps-solve --hkl your.hkl --ins your.ins --method partial_phaseed "
                    "--phase-seed-csv seed.csv --out ./out_partial",
                ],
                "alternatives": [
                    "Predicted-model CIF: gps-solve … --predicted-model model.cif --method partial_phaseed",
                    "HA pair: --native-hkl … --derivative-hkl … --method ha_phaseed",
                ],
            }
        )
        return rec

    if seeded:
        rec.update(
            {
                "primary_id": "better_seed",
                "primary": (
                    "A seed was applied but the map still looks weak. "
                    "Size may be OK — try a *different* source (SHELXS fragment, "
                    "predicted-model CIF, or HA), not more polish on the same seed."
                ),
                "why": (
                    "Hard-region bottleneck is seed quality, not free-FOM inversion. "
                    "A coherent half-model beats more CF iterations on the Vol-band panel."
                ),
                "commands": [
                    "gps-solve --hkl your.hkl --ins your.ins --method partial_phaseed "
                    "--phase-seed-res fragment.res --out ./out_frag",
                    "gps-solve --hkl your.hkl --ins your.ins --method partial_phaseed "
                    "--predicted-model model.cif --out ./out_pred",
                ],
                "alternatives": [
                    "Recycle this run's peaks: --seed-peaks-csv peaks.csv",
                    "External classical: --method shelxs (academic binary in ShelX/)",
                ],
            }
        )
        return rec

    if band == "vol_gt_3500":
        rec.update(
            {
                "primary_id": "large_fragment_or_ha",
                "primary": (
                    "Large cell (Vol > 3500 Å³): a half-model is often not enough. "
                    "Bring a larger fragment, HA/MAD pair, or ~30% known strong φ."
                ),
                "why": (
                    "Local COD Vol-band (2017775): fragment_half mean mapCC ~0.49 vs "
                    "oracle partial_30 ~0.66 vs auto ~0.07. Not a 1505-COD panel."
                ),
                "commands": [
                    "gps-solve --hkl your.hkl --ins your.ins --method partial_phaseed "
                    "--phase-seed-res large_fragment.res --out ./out_frag",
                    "gps-solve --hkl your.hkl --ins your.ins --method ha_phaseed "
                    "--native-hkl native.hkl --derivative-hkl deriv.hkl --out ./out_ha",
                ],
                "alternatives": [
                    "If you only have a small fragment, expect a weak map; enlarge the model.",
                    "External: --method shelxs+shelxe",
                ],
            }
        )
        return rec

    if (
        band == "vol_lt_1000"
        and dmin is not None
        and dmin <= 1.15
        and (method or "").lower() not in ("ensemble",)
        and outlook == "likely_unsolved"
    ):
        rec.update(
            {
                "primary_id": "try_ensemble",
                "primary": (
                    "Small cell at good resolution: retry `--method ensemble` "
                    "(auto's easy-path pick) before hunting for seeds."
                ),
                "why": (
                    "On easy synthetic panels, multistart ensemble is the strongest "
                    "in-repo ab initio path. If ensemble also fails, use a fragment seed."
                ),
                "commands": [
                    "gps-solve --hkl your.hkl --ins your.ins --method ensemble "
                    "--n-starts 5 --out ./out_ens",
                ],
                "alternatives": [
                    "Fragment: --method partial_phaseed --phase-seed-res model.res",
                    "Peaks recycle: --seed-peaks-csv peaks.csv",
                ],
            }
        )
        return rec

    # Mid-band default, and small-cell after ensemble already tried.
    rec.update(
        {
            "primary_id": "fragment_or_predicted",
            "primary": (
                "Bring a fragment or predicted-model seed and re-run "
                "`partial_phaseed`. Pure ab initio is the weak path here."
            ),
            "why": (
                "Local COD Vol 1000–3500 Å³ panel: fragment_half mean mapCC ~0.71, "
                "oracle partial_30 ~0.70, auto ~0.27 (Fobs+Fcalc pooled; C25). "
                "Small-cell fragment_half is similarly strong (~0.74)."
            ),
            "commands": [
                "gps-solve --hkl your.hkl --ins your.ins --method partial_phaseed "
                "--phase-seed-res fragment.res --out ./out_frag",
                "gps-solve --hkl your.hkl --ins your.ins --method partial_phaseed "
                "--predicted-model model.cif --out ./out_pred",
            ],
            "alternatives": [
                "Known φ CSV: --phase-seed-csv known.csv (aim ≥~30% of strong |E|)",
                "Build seed only: gps-make-seed --hkl your.hkl --ins your.ins "
                "--from-res model.res -o seed.csv",
                "HA pair: --native-hkl … --derivative-hkl … --method ha_phaseed",
            ],
        }
    )
    return rec


def format_next_action_md(rec: Mapping[str, Any]) -> str:
    """Markdown section for report.md."""
    vol = rec.get("vol")
    vol_s = "—" if vol is None else f"{vol:.0f} Å³"
    fom = rec.get("free_fom_composite")
    fom_s = "—" if fom is None else f"{float(fom):.3f}"
    lines: List[str] = [
        "## Next action",
        "",
        f"- **Volume band:** {rec.get('vol_band_label')} (V = {vol_s})",
        f"- **Map outlook:** `{rec.get('map_outlook')}` "
        f"(free-FOM {fom_s}; {rec.get('n_peaks')} peaks) — ranking only",
        f"- **Do this:** {rec.get('primary')}",
        f"- **Why:** {rec.get('why')}",
        "",
        "```bash",
    ]
    for cmd in rec.get("commands") or []:
        lines.append(str(cmd))
    lines.extend(["```", ""])
    alts = rec.get("alternatives") or []
    if alts:
        lines.append("Other options:")
        for a in alts:
            lines.append(f"- {a}")
        lines.append("")
    lines.append(
        "Evidence: local COD Vol-band panel (C25), not Carrozzini’s 1505-structure set."
    )
    return "\n".join(lines)


def next_action_banner(rec: Mapping[str, Any]) -> str:
    """One-line banner for report header / GUI."""
    return f"Next action ({rec.get('vol_band_label')}): {rec.get('primary')}"
