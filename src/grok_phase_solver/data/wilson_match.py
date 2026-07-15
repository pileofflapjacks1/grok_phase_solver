"""
Close the Wilson / amplitude domain gap between synthetic and experimental |F|.

Hard synthetic Fcalc often has a steeper or shallower Wilson slope, different
intensity tails, and no measurement noise — so training priors on raw Fcalc
can fail to transfer. This module:

1. Matches overall Wilson slope (extra isotropic B)
2. Matches resolution-shell mean intensities
3. Matches amplitude quantiles (histogram matching)
4. Adds calibrated noise + optional incompleteness
5. Reports before/after domain_gap_score

Physics: multiplying |F| by exp(−ΔB · s²) is equivalent to an overall
Debye–Waller adjustment; phases are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from grok_phase_solver.data.wilson import (
    amplitude_moments,
    domain_gap_report,
    domain_gap_wilson,
    wilson_plot,
)
from grok_phase_solver.physics.reciprocal import d_spacing

PathLike = Union[str, Path]


def s2_from_hkl(hkl: np.ndarray, cell: np.ndarray) -> np.ndarray:
    d = d_spacing(hkl, cell)
    s = 1.0 / (2.0 * np.maximum(d, 1e-8))
    return s * s


def apply_overall_b(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    B_extra: float,
) -> np.ndarray:
    """
    |F|' = |F| · exp(−B_extra · s²).

    B_extra > 0 damps high-angle (more thermal-like).
    """
    amp = np.asarray(amplitudes, dtype=np.float64)
    s2 = s2_from_hkl(hkl, cell)
    return amp * np.exp(-float(B_extra) * s2)


def match_wilson_slope(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    target_slope: float,
    *,
    n_shells: int = 10,
    max_abs_B: float = 80.0,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply overall B so relative Wilson slope ≈ target_slope.

    For relative Wilson, ln⟨I⟩ = a + b s². Scaling I by exp(δ s²) shifts b by δ.
    We set δ = target_slope − current_slope, then |F| *= exp(δ/2 · s²).

    Note: apply_overall_b uses exp(−B s²) on |F| ⇒ I *= exp(−2 B s²),
    so slope shift = −2 B ⇒ B = −δ/2.
    """
    w = wilson_plot(hkl, amplitudes, cell, n_shells=n_shells)
    b_cur = float(w["slope"])
    delta = float(target_slope) - b_cur
    # I' = I * exp(delta * s²)  ⇒  amp' = amp * exp(0.5 * delta * s²)
    # Also amp' = amp * exp(−B s²) with B = −0.5 * delta
    B_extra = -0.5 * delta
    B_extra = float(np.clip(B_extra, -max_abs_B, max_abs_B))
    amp2 = apply_overall_b(hkl, amplitudes, cell, B_extra)
    w2 = wilson_plot(hkl, amp2, cell, n_shells=n_shells)
    return amp2, {
        "slope_before": b_cur,
        "slope_target": float(target_slope),
        "slope_after": float(w2["slope"]),
        "B_extra": B_extra,
        "B_before": float(w["B_overall"]),
        "B_after": float(w2["B_overall"]),
    }


def match_wilson_to_reference(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    ref_hkl: np.ndarray,
    ref_amp: np.ndarray,
    ref_cell: np.ndarray,
    *,
    n_shells: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """Match Wilson slope of amplitudes to a reference dataset."""
    wr = wilson_plot(ref_hkl, ref_amp, ref_cell, n_shells=n_shells)
    return match_wilson_slope(
        hkl, amplitudes, cell, target_slope=float(wr["slope"]), n_shells=n_shells
    )


def match_shell_means(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    ref_hkl: np.ndarray,
    ref_amp: np.ndarray,
    ref_cell: np.ndarray,
    *,
    n_shells: int = 12,
    relative: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Scale |F| in resolution shells toward the reference intensity *profile*.

    If ``relative=True`` (default), match ⟨I⟩_shell / ⟨I⟩_global so absolute
    scale and overall Wilson slope are not destroyed by cell-volume mismatch.
    """
    amp = np.asarray(amplitudes, dtype=np.float64).copy()
    s2 = s2_from_hkl(hkl, cell)
    s2_r = s2_from_hkl(ref_hkl, ref_cell)
    I_r = np.asarray(ref_amp, dtype=np.float64) ** 2

    order = np.argsort(s2)
    edges = np.linspace(0, len(amp), n_shells + 1, dtype=int)
    order_r = np.argsort(s2_r)
    edges_r = np.linspace(0, len(ref_amp), n_shells + 1, dtype=int)
    ref_s2c, ref_meanI = [], []
    for k in range(n_shells):
        idx = order_r[edges_r[k] : edges_r[k + 1]]
        if len(idx) < 3:
            continue
        ref_s2c.append(float(np.mean(s2_r[idx])))
        ref_meanI.append(float(np.mean(I_r[idx])))
    if len(ref_s2c) < 2:
        return amp, {"n_shells_scaled": 0, "scales": [], "relative": relative}

    ref_s2c = np.array(ref_s2c)
    ref_meanI = np.array(ref_meanI)
    o = np.argsort(ref_s2c)
    ref_s2c, ref_meanI = ref_s2c[o], ref_meanI[o]
    ref_global = float(np.mean(I_r)) + 1e-16
    src_global = float(np.mean(amp ** 2)) + 1e-16

    scales = []
    out = amp.copy()
    for k in range(n_shells):
        idx = order[edges[k] : edges[k + 1]]
        if len(idx) < 3:
            continue
        s2c = float(np.mean(s2[idx]))
        mean_I = float(np.mean(amp[idx] ** 2))
        target_I_abs = float(np.interp(s2c, ref_s2c, ref_meanI))
        if relative:
            # target relative profile on source overall scale
            target_I = src_global * (target_I_abs / ref_global)
        else:
            target_I = target_I_abs
        if mean_I < 1e-30 or target_I < 1e-30:
            sc = 1.0
        else:
            sc = np.sqrt(target_I / mean_I)
        # mild clamp — relative profile should not need huge scales
        sc = float(np.clip(sc, 0.5, 2.0) if relative else np.clip(sc, 0.05, 20.0))
        out[idx] *= sc
        scales.append({"shell": k, "s2": s2c, "scale": sc})
    return out, {"n_shells_scaled": len(scales), "scales": scales, "relative": relative}


def match_amplitude_quantiles(
    amplitudes: np.ndarray,
    ref_amplitudes: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """
    Histogram-match |F| magnitudes to reference distribution (rank mapping).

    Preserves rank order. Prefer ``match_quantiles_per_shell`` so Wilson slope
    is not destroyed by mixing resolution shells.
    """
    amp = np.asarray(amplitudes, dtype=np.float64)
    ref = np.asarray(ref_amplitudes, dtype=np.float64)
    ref = ref[np.isfinite(ref) & (ref > 0)]
    if len(amp) < 5 or len(ref) < 5:
        return amp.copy(), {"applied": False}

    order = np.argsort(amp)
    ranks = np.empty(len(amp), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(amp))
    ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
    ref_sorted = np.sort(ref)
    q = ranks * (len(ref_sorted) - 1)
    lo = np.floor(q).astype(int)
    hi = np.minimum(lo + 1, len(ref_sorted) - 1)
    t = q - lo
    matched = (1 - t) * ref_sorted[lo] + t * ref_sorted[hi]
    # keep shell/source mean (critical for Wilson)
    matched = matched * (amp.mean() / (matched.mean() + 1e-16))
    return matched, {
        "applied": True,
        "mean_before": float(amp.mean()),
        "mean_after": float(matched.mean()),
        "std_before": float(amp.std()),
        "std_after": float(matched.std()),
    }


def match_quantiles_per_shell(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    ref_hkl: np.ndarray,
    ref_amp: np.ndarray,
    ref_cell: np.ndarray,
    *,
    n_shells: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """
    Within each resolution shell, quantile-match |F| to the reference shell.

    Avoids global histogram match, which mixes high- and low-angle |F| and
    destroys the Wilson slope.
    """
    amp = np.asarray(amplitudes, dtype=np.float64).copy()
    ref_amp = np.asarray(ref_amp, dtype=np.float64)
    s2 = s2_from_hkl(hkl, cell)
    s2_r = s2_from_hkl(ref_hkl, ref_cell)
    order = np.argsort(s2)
    edges = np.linspace(0, len(amp), n_shells + 1, dtype=int)
    order_r = np.argsort(s2_r)
    edges_r = np.linspace(0, len(ref_amp), n_shells + 1, dtype=int)

    # ref shell list by s2 center for nearest-shell lookup
    ref_shells = []
    for k in range(n_shells):
        idx = order_r[edges_r[k] : edges_r[k + 1]]
        if len(idx) < 5:
            continue
        ref_shells.append((float(np.mean(s2_r[idx])), ref_amp[idx]))
    if not ref_shells:
        return amp, {"n_shells": 0}

    out = amp.copy()
    n_done = 0
    for k in range(n_shells):
        idx = order[edges[k] : edges[k + 1]]
        if len(idx) < 5:
            continue
        s2c = float(np.mean(s2[idx]))
        # nearest ref shell
        j = int(np.argmin([abs(s2c - rs[0]) for rs in ref_shells]))
        matched, info = match_amplitude_quantiles(amp[idx], ref_shells[j][1])
        if info.get("applied"):
            out[idx] = matched
            n_done += 1
    return out, {"n_shells": n_done}


def add_measurement_noise(
    amplitudes: np.ndarray,
    noise_level: float = 0.05,
    seed: int = 0,
    floor: float = 0.0,
) -> np.ndarray:
    """Relative Gaussian noise on |F|; clip to ≥ floor."""
    rng = np.random.default_rng(seed)
    amp = np.asarray(amplitudes, dtype=np.float64)
    if noise_level <= 0:
        return amp.copy()
    out = amp * (1.0 + noise_level * rng.standard_normal(len(amp)))
    return np.maximum(out, floor)


def filter_resolution_overlap(
    pack: Dict,
    ref: Dict,
    *,
    margin: float = 0.02,
) -> Dict:
    """
    Keep only reflections in the overlapping d-range with a reference set.

    Helps fair Wilson comparison when hard synth is truncated at 1.7 Å
    vs experiment spanning a wider shell set.
    """
    d = d_spacing(pack["hkl"], pack["cell"])
    d_r = d_spacing(ref["hkl"], ref["cell"])
    d_lo = max(float(d.min()), float(d_r.min())) - margin
    d_hi = min(float(d.max()), float(d_r.max())) + margin
    # keep d in [d_lo_eff, d_hi_eff] where high-angle = small d
    d_min_keep = max(float(d.min()), float(d_r.min()))
    d_max_keep = min(float(d.max()), float(d_r.max()))
    m = (d >= d_min_keep - 1e-9) & (d <= d_max_keep + 1e-9)
    if m.sum() < 20:
        return pack
    out = dict(pack)
    out["hkl"] = pack["hkl"][m]
    out["amplitudes"] = np.asarray(pack["amplitudes"])[m]
    if "phases" in pack and pack["phases"] is not None:
        out["phases"] = np.asarray(pack["phases"])[m]
    out["n_refl"] = int(m.sum())
    out["d_overlap"] = (float(d_min_keep), float(d_max_keep))
    return out


@dataclass
class WilsonMatchConfig:
    """Controls for close_wilson_gap.

    Recommended default: slope + relative shells + per-shell quantiles,
    with slope re-locked after each step. Global quantile match is unsafe.
    """

    match_slope: bool = True
    match_shells: bool = True
    match_quantiles: bool = True  # per-shell quantiles
    noise_level: float = 0.02
    resolution_overlap: bool = False
    n_shells: int = 12
    seed: int = 0
    relock_slope: bool = True  # re-apply Wilson slope after other steps


def close_wilson_gap(
    synth: Dict,
    ref: Dict,
    config: Optional[WilsonMatchConfig] = None,
) -> Tuple[Dict, Dict]:
    """
    Transform synthetic amplitudes toward reference |F| statistics.

    Parameters
    ----------
    synth, ref : dicts with hkl, amplitudes, cell (phases optional, unchanged)

    Returns
    -------
    matched_pack : copy of synth with matched amplitudes
    report : before/after gaps and step diagnostics
    """
    cfg = config or WilsonMatchConfig()
    hkl = np.asarray(synth["hkl"])
    amp = np.asarray(synth["amplitudes"], dtype=np.float64).copy()
    cell = np.asarray(synth["cell"], dtype=np.float64)
    rhkl = np.asarray(ref["hkl"])
    ramp = np.asarray(ref["amplitudes"], dtype=np.float64)
    rcell = np.asarray(ref["cell"], dtype=np.float64)

    steps: List[Dict] = []
    pack0 = {
        "hkl": hkl,
        "amplitudes": amp,
        "cell": cell,
        "name": synth.get("name", "synth"),
    }
    gap0 = domain_gap_report(
        pack0, ref, label_a="synth_raw", label_b=ref.get("name", "ref")
    )

    def _relock(label: str) -> None:
        nonlocal amp
        if not (cfg.relock_slope and cfg.match_slope):
            return
        amp, info = match_wilson_to_reference(
            hkl, amp, cell, rhkl, ramp, rcell, n_shells=cfg.n_shells
        )
        steps.append({"step": f"relock_slope_after_{label}", **info})

    if cfg.match_slope:
        amp, info = match_wilson_to_reference(
            hkl, amp, cell, rhkl, ramp, rcell, n_shells=cfg.n_shells
        )
        steps.append({"step": "wilson_slope", **info})

    if cfg.match_shells:
        amp, info = match_shell_means(
            hkl, amp, cell, rhkl, ramp, rcell, n_shells=cfg.n_shells, relative=True
        )
        steps.append(
            {
                "step": "shell_means_relative",
                "n_shells_scaled": info["n_shells_scaled"],
            }
        )
        _relock("shells")

    if cfg.match_quantiles:
        amp, info = match_quantiles_per_shell(
            hkl, amp, cell, rhkl, ramp, rcell, n_shells=max(6, cfg.n_shells // 1)
        )
        steps.append({"step": "quantiles_per_shell", **info})
        _relock("quantiles")

    if cfg.noise_level > 0:
        amp = add_measurement_noise(amp, noise_level=cfg.noise_level, seed=cfg.seed)
        steps.append({"step": "noise", "noise_level": cfg.noise_level})
        # do not relock after noise — noise is intended perturbation

    matched = dict(synth)
    matched["amplitudes"] = amp
    matched["hkl"] = hkl
    matched["cell"] = cell
    matched["wilson_matched"] = True
    if "phases" in synth:
        matched["phases"] = synth["phases"]

    if cfg.resolution_overlap:
        matched = filter_resolution_overlap(matched, ref)
        steps.append({"step": "resolution_overlap", "n_refl": matched.get("n_refl")})

    gap1 = domain_gap_report(
        matched, ref, label_a="synth_matched", label_b=ref.get("name", "ref")
    )
    report = {
        "gap_before": gap0["domain_gap_score"],
        "gap_after": gap1["domain_gap_score"],
        "gap_reduction": float(gap0["domain_gap_score"] - gap1["domain_gap_score"]),
        "gap_reduction_frac": float(
            (gap0["domain_gap_score"] - gap1["domain_gap_score"])
            / max(gap0["domain_gap_score"], 1e-9)
        ),
        "wilson_before": gap0["wilson"],
        "wilson_after": gap1["wilson"],
        "moments_before": gap0["moments_a"],
        "moments_after": gap1["moments_a"],
        "steps": steps,
        "config": {
            "match_slope": cfg.match_slope,
            "match_shells": cfg.match_shells,
            "match_quantiles": cfg.match_quantiles,
            "noise_level": cfg.noise_level,
            "resolution_overlap": cfg.resolution_overlap,
            "relock_slope": cfg.relock_slope,
        },
    }
    return matched, report


def match_batch_to_reference(
    packs: Sequence[Dict],
    ref: Dict,
    config: Optional[WilsonMatchConfig] = None,
) -> Tuple[List[Dict], Dict]:
    """Apply close_wilson_gap to a list of synthetic packs; aggregate stats."""
    cfg = config or WilsonMatchConfig()
    out = []
    gaps_b, gaps_a = [], []
    for i, p in enumerate(packs):
        cfg_i = WilsonMatchConfig(
            match_slope=cfg.match_slope,
            match_shells=cfg.match_shells,
            match_quantiles=cfg.match_quantiles,
            noise_level=cfg.noise_level,
            resolution_overlap=cfg.resolution_overlap,
            n_shells=cfg.n_shells,
            seed=cfg.seed + i,
        )
        m, rep = close_wilson_gap(p, ref, cfg_i)
        out.append(m)
        gaps_b.append(rep["gap_before"])
        gaps_a.append(rep["gap_after"])
    return out, {
        "n": len(out),
        "mean_gap_before": float(np.mean(gaps_b)) if gaps_b else None,
        "mean_gap_after": float(np.mean(gaps_a)) if gaps_a else None,
        "mean_reduction": float(np.mean(np.array(gaps_b) - np.array(gaps_a))) if gaps_b else None,
        "mean_reduction_frac": float(
            np.mean((np.array(gaps_b) - np.array(gaps_a)) / np.maximum(gaps_b, 1e-9))
        )
        if gaps_b
        else None,
    }


def load_reference_template(path: Optional[PathLike] = None) -> Optional[Dict]:
    """
    Load a cached experimental |F| template for matching.

    Default path: data/processed/wilson_ref_template.npz
    """
    if path is None:
        path = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "processed"
            / "wilson_ref_template.npz"
        )
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return {
        "hkl": z["hkl"],
        "amplitudes": z["amplitudes"],
        "cell": z["cell"],
        "name": str(z["name"]) if "name" in z.files else path.stem,
    }


def save_reference_template(pack: Dict, path: Optional[PathLike] = None) -> Path:
    if path is None:
        path = (
            Path(__file__).resolve().parents[3]
            / "data"
            / "processed"
            / "wilson_ref_template.npz"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        hkl=np.asarray(pack["hkl"]),
        amplitudes=np.asarray(pack["amplitudes"], dtype=np.float64),
        cell=np.asarray(pack["cell"], dtype=np.float64),
        name=str(pack.get("name", "template")),
    )
    return path


def apply_wilson_match_if_template(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    phases: Optional[np.ndarray] = None,
    template: Optional[Dict] = None,
    config: Optional[WilsonMatchConfig] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience for training loops: match amplitudes if a template is available.

    Returns (amplitudes_out, meta). Phases unchanged (caller keeps them).
    """
    if template is None:
        template = load_reference_template()
    if template is None:
        return np.asarray(amplitudes, dtype=np.float64), {"matched": False, "reason": "no_template"}
    synth = {
        "hkl": hkl,
        "amplitudes": amplitudes,
        "cell": cell,
        "phases": phases,
        "name": "train_sample",
    }
    matched, report = close_wilson_gap(synth, template, config=config)
    meta = {"matched": True, **{k: report[k] for k in ("gap_before", "gap_after", "gap_reduction")}}
    return matched["amplitudes"], meta
