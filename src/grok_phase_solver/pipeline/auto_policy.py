"""Honest auto method policy for gps-solve.

GraphPhaseNet (strong_prior_phaseed) and hard_p1_phaseed are never
selected by auto; they remain explicit --method values for research.
"""

from __future__ import annotations

from typing import Optional, Tuple


def _sg_norm(space_group: Optional[str]) -> str:
    return (space_group or "").replace(" ", "").upper()


def _is_p21c_like(sg: str) -> bool:
    s = _sg_norm(sg)
    return any(x in s for x in ("P21/C", "P121/C1", "P21/C1", "P121/C"))


def _phai_ok() -> bool:
    try:
        from grok_phase_solver.models.phai_runner import phai_available

        return bool(phai_available())
    except Exception:
        return False


def resolve_method(
    method: str,
    space_group: Optional[str],
    data_dmin: float,
    n_refl: int,
) -> Tuple[str, str]:
    """Resolve auto to a concrete method. GraphPhaseNet is never auto-picked."""
    from grok_phase_solver.pipeline.solve import KNOWN_METHODS

    m = method.lower().strip()
    if m != "auto":
        if m not in KNOWN_METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {KNOWN_METHODS}")
        if m == "fragment_phaseed":
            return "partial_phaseed", "user-selected (fragment/atomic model seed)"
        if m == "ha_phaseed":
            return "partial_phaseed", "user-selected (HA / difference Patterson seed)"
        return m, "user-selected"

    sg = space_group or ""
    phai = _phai_ok()

    if data_dmin <= 1.15 and n_refl >= 80:
        return "ensemble", "auto: easy/high-res \u2192 multistart ensemble (CF+RAAR free-FOM)"

    if phai and _is_p21c_like(sg):
        return "phai_phaseed", "auto: P21/c-like + PhAI \u2192 AI-PhaSeed"
    if phai and data_dmin <= 1.25:
        return "phai+cf_cond", "auto: PhAI + free-FOM-gated CF"

    if data_dmin <= 1.4 and n_refl >= 100:
        return "ensemble", "auto: mid-res \u2192 ensemble"

    return (
        "charge_flipping",
        "auto: last-resort charge flipping (hard ab initio not claimed, "
        "~0% strict success). Prefer --method partial_phaseed with a seed "
        "(--phase-seed-csv / --phase-seed-res / --predicted-model / "
        "native+derivative HKL / --patterson-ha).",
    )
