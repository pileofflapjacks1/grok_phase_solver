"""
Second-pass hard retry: recycle this run's peaks as a fragment seed.

Matches the GUI "Retry with peaks as seed (partial_phaseed)" button.
Cheap path when auto/ensemble/CF is weak and the user has no external
fragment/HA yet. Not a claim that peaks-as-C atoms meet the 30%/20° bar.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Tuple

from grok_phase_solver.pipeline.solve import SolveConfig, SolveResult, solve_structure

_AB_INITIO = frozenset(
    {
        "charge_flipping",
        "ensemble",
        "raar",
        "hio",
        "dual_space",
        "direct_methods",
        "recycle",
        "strong_prior_phaseed",
        "hard_p1_phaseed",
        "phai_phaseed",
        "phai+cf",
        "phai+cf_cond",
        "phai+recycle",
        "phai",
    }
)
_ALREADY_SEEDED = frozenset(
    {
        "partial_phaseed",
        "fragment_phaseed",
        "ha_phaseed",
        "diffusion_phaseed",
    }
)


def _fom(result: SolveResult) -> Optional[float]:
    raw = (result.diagnostics or {}).get("free_fom_composite")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def should_retry_with_peaks(
    result: SolveResult,
    *,
    fom_threshold: float = 0.60,
    min_peaks: int = 3,
) -> Tuple[bool, str]:
    """Truth-free gate matching the GUI retry button (FOM + method family)."""
    method = (result.method or "").lower()
    n_peaks = len(result.peaks or [])
    if method in _ALREADY_SEEDED or (result.diagnostics or {}).get("seed_kind"):
        return False, "already on a seeded path — enlarge/change the seed instead"
    if n_peaks < min_peaks:
        return False, f"too few peaks ({n_peaks} < {min_peaks}) to recycle as a fragment"
    fom = _fom(result)
    if fom is not None and fom >= 0.70:
        return False, f"free-FOM {fom:.3f} looks healthy — inspect trial.res / density.map first"
    weak = fom is None or fom < fom_threshold
    ab_initio = method in _AB_INITIO
    if weak or (ab_initio and (fom is None or fom < 0.70)):
        why = (
            f"weak map (free-FOM {fom if fom is None else f'{fom:.3f}'}, "
            f"method `{method}`, {n_peaks} peaks) → recycle peaks as partial_phaseed"
        )
        return True, why
    return False, "map not clearly weak; pass --seed-peaks-csv peaks.csv to force"


def retry_config(cfg: SolveConfig, peaks_csv: Path) -> SolveConfig:
    """Copy config for a peaks-seeded partial_phaseed pass."""
    return replace(
        cfg,
        method="partial_phaseed",
        phase_seed_csv=None,
        phase_seed_res=None,
        seed_atoms_csv=None,
        seed_peaks_csv=str(peaks_csv),
        predicted_model_cif=None,
        native_hkl=None,
        derivative_hkl=None,
        patterson_ha=False,
    )


def append_retry_note(
    first_report: Path,
    retry_dir: Path,
    retry_result: SolveResult,
    reason: str,
) -> None:
    """Point the first-pass report at retry_peaks/."""
    fom = _fom(retry_result)
    fom_s = "—" if fom is None else f"{fom:.3f}"
    note = (
        "\n## Retry with peaks (second pass)\n\n"
        f"- **Trigger:** {reason}\n"
        f"- **Method:** `partial_phaseed` with this run's `peaks.csv` as light-atom Fcalc seed\n"
        f"- **Retry free-FOM:** {fom_s} (ranking only)\n"
        f"- **Output:** `{retry_dir.name}/` (`report.md`, `density.map`, `trial.res`)\n"
        "\n"
        "Peaks-as-carbon is a cheap fallback when you have no fragment/HA yet. "
        "A real SHELXS/predicted-model fragment is stronger (Vol-band C25).\n"
    )
    try:
        text = first_report.read_text()
        if "## Retry with peaks" not in text:
            first_report.write_text(text.rstrip() + "\n" + note)
    except OSError:
        pass


def run_peaks_retry(
    *,
    hkl_path: str,
    first_result: SolveResult,
    first_out: Path,
    cfg: SolveConfig,
    ins_path: Optional[str] = None,
    cell: Optional[str] = None,
    space_group: Optional[str] = None,
    wavelength: Optional[float] = None,
) -> Tuple[Optional[SolveResult], Path, str]:
    """
    If the first pass is weak, solve again with peaks.csv → partial_phaseed.

    Writes ``first_out / retry_peaks /``. Returns (result_or_None, retry_dir, reason).
    """
    from grok_phase_solver.pipeline.export import export_solution

    ok, reason = should_retry_with_peaks(first_result)
    retry_dir = Path(first_out) / "retry_peaks"
    if not ok:
        return None, retry_dir, reason
    peaks = Path(first_out) / "peaks.csv"
    if not peaks.is_file():
        return None, retry_dir, "peaks.csv missing after first export"
    cfg2 = retry_config(cfg, peaks)
    result2 = solve_structure(
        hkl_path=hkl_path,
        ins_path=ins_path,
        cell=cell,
        space_group=space_group,
        wavelength=wavelength,
        config=cfg2,
    )
    result2.diagnostics["retry_of"] = {
        "first_method": first_result.method,
        "first_free_fom": _fom(first_result),
        "reason": reason,
        "seed_peaks_csv": str(peaks),
    }
    export_solution(result2, retry_dir)
    append_retry_note(Path(first_out) / "report.md", retry_dir, result2, reason)
    return result2, retry_dir, reason
