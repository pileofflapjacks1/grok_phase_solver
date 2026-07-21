"""
Seed-quality prediction for AI-PhaSeed (Carrozzini et al. 2025 alignment).

The 2025 AI-PhaSeed paper classifies seeds with k-means + random forest on
features including MPE_seed, CORR_seed, max W, N_asym, unit-cell volume, and
seed fraction. Class 1 seeds achieved >90% efficiency on their COD panel.

This module provides:

1. **Feature extraction** aligned with that toolkit (truth-free when needed).
2. **Heuristic Class 0/1 predictor** (no extra deps) with success probability.
3. **Optional sklearn RF** if ``scikit-learn`` is installed and a small model
   has been fit/persisted (or coefficients provided).

Honest limits
-------------
- Without ground-truth phases we **estimate** MPE/CORR from free-FOM proxies
  and Wilson structure of the seed set — not the paper's oracle metrics.
- Class labels here are **operational heuristics**, not a claim of the
  published RF trained on 1505 COD structures.
- Low predicted quality should trigger warnings / partial-φ fallbacks, not
  automatic claims of failure.

References
----------
- Carrozzini et al. (2025). J. Appl. Cryst. 58, 1859–1869.
  DOI: 10.1107/S1600576725008271
- Larsen et al. (2024). Science 385, 522–528 (PhAI foundation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from grok_phase_solver.solvers.direct_methods import normalize_E
from grok_phase_solver.solvers.projectors import unit_cell_volume


# Typical organic non-H atom volume (Å³) for N_asym heuristics
_VOL_PER_ATOM_ASU = 18.0  # rough P2₁/c organic packing


@dataclass
class SeedQualityReport:
    """Structured seed-quality prediction (truth-free or oracle-augmented)."""

    predicted_class: int  # 0 = low success, 1 = high success (paper-like)
    success_probability: float  # ∈ [0, 1]
    predicted_mpe_deg: float  # estimated seed MPE (°)
    predicted_corr: float  # estimated seed phase correlation ∈ [-1, 1]
    features: Dict[str, float] = field(default_factory=dict)
    warning: Optional[str] = None
    recommend_fallback: bool = False
    method: str = "heuristic"  # heuristic | sklearn_rf | coefficients
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def estimate_n_asym(
    cell: np.ndarray,
    n_atoms_user: Optional[int] = None,
    z: float = 4.0,
    vol_per_atom: float = _VOL_PER_ATOM_ASU,
) -> float:
    """
    Approximate number of non-H atoms in the asymmetric unit.

    Prefer ``n_atoms_user`` when known. Else Vol / (Z · V_atom) with Z≈4 for
    common P2₁/c organics (Carrozzini panel focus).
    """
    if n_atoms_user is not None and n_atoms_user > 0:
        return float(n_atoms_user)
    vol = float(unit_cell_volume(np.asarray(cell, dtype=np.float64)))
    return max(vol / (max(z, 1.0) * vol_per_atom), 1.0)


def extract_seed_features(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    seed_phases: np.ndarray,
    seed_idx: Optional[np.ndarray] = None,
    *,
    seed_fraction: Optional[float] = None,
    n_asym: Optional[float] = None,
    n_atoms_user: Optional[int] = None,
    d_min: Optional[float] = None,
    free_fom_seed: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Feature vector inspired by Carrozzini 2025 statistical toolkit.

    Features
    --------
    - max_W : max |E| on the seed set (paper: max W)
    - mean_E_seed, median_E_seed
    - N_asym : estimated atoms in ASU
    - Vol : unit-cell volume (Å³)
    - seed_fraction : |S| / N_refl
    - n_seed, n_refl
    - free_fom_composite, R_pos (if free_fom provided or computed lightly)
    - d_min (if given)
    - density_asym : N_asym / Vol
    """
    hkl = np.asarray(hkl, dtype=int)
    amp = np.asarray(amplitudes, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    ph = np.asarray(seed_phases, dtype=np.float64)
    n = len(amp)
    if seed_idx is None:
        # top 25% by |E| if not provided
        E_all = normalize_E(hkl, amp, cell)
        n_s = max(int(0.25 * n), 10)
        n_s = min(n_s, n)
        seed_idx = np.argsort(-E_all)[:n_s]
    seed_idx = np.asarray(seed_idx, dtype=int)
    n_seed = len(seed_idx)
    frac = float(n_seed / max(n, 1))
    if seed_fraction is not None:
        frac = float(seed_fraction)

    E = normalize_E(hkl, amp, cell)
    E_s = E[seed_idx]
    max_W = float(np.max(E_s)) if n_seed else 0.0
    mean_E = float(np.mean(E_s)) if n_seed else 0.0
    med_E = float(np.median(E_s)) if n_seed else 0.0

    vol = float(unit_cell_volume(cell))
    n_as = float(n_asym) if n_asym is not None else estimate_n_asym(
        cell, n_atoms_user=n_atoms_user
    )

    feats: Dict[str, float] = {
        "max_W": max_W,
        "mean_E_seed": mean_E,
        "median_E_seed": med_E,
        "N_asym": n_as,
        "Vol": vol,
        "seed_fraction": frac,
        "n_seed": float(n_seed),
        "n_refl": float(n),
        "density_asym": n_as / max(vol, 1.0),
    }
    if d_min is not None:
        feats["d_min"] = float(d_min)

    # Free-FOM proxy on full seed phase vector (truth-free quality signal)
    fom = free_fom_seed
    if fom is None:
        try:
            from grok_phase_solver.solvers.free_fom import free_fom

            fom = free_fom(hkl, amp, ph, cell)
        except Exception:
            fom = None
    if fom is not None:
        feats["free_fom_composite"] = float(fom.get("composite", 0.0))
        feats["R_pos"] = float(fom.get("R_pos", 1.0))
        feats["excess_kurtosis"] = float(fom.get("excess_kurtosis", 0.0))
    else:
        feats["free_fom_composite"] = 0.0
        feats["R_pos"] = 1.0
        feats["excess_kurtosis"] = 0.0

    return feats


def _heuristic_success_probability(feats: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    Map features → P(success) without sklearn.

    Tuned for organic small-molecule / P2₁/c-like regimes highlighted in
    Carrozzini 2025 (Vol ~1000–3500 Å³, strong |E| seeds, usable free FOM).
    Not a reimplementation of their published RF.
    """
    notes: List[str] = []
    p = 0.15  # base (hard ab initio is low)

    vol = feats.get("Vol", 0.0)
    # Sweet volume band for hybrid EDM protocols in the paper
    if 800.0 <= vol <= 4000.0:
        p += 0.18
        notes.append("Vol in hybrid-friendly band (~800–4000 Å³)")
    elif vol > 8000.0:
        p -= 0.08
        notes.append("Large volume: harder for pure AI-PhaSeed")
    elif 0 < vol < 500.0:
        p += 0.05
        notes.append("Small cell: often easier for classical methods")

    max_W = feats.get("max_W", 0.0)
    if max_W >= 2.5:
        p += 0.12
        notes.append(f"Strong max |E| (max_W={max_W:.2f})")
    elif max_W >= 1.8:
        p += 0.06
    else:
        p -= 0.05
        notes.append("Weak max |E| on seed set")

    frac = feats.get("seed_fraction", 0.0)
    # Paper uses modest seed fractions; too small or huge both hurt
    if 0.10 <= frac <= 0.40:
        p += 0.10
    elif frac < 0.05:
        p -= 0.08
        notes.append("Very small seed fraction")
    elif frac > 0.55:
        p += 0.02  # dense AI prior can still help

    fom_c = feats.get("free_fom_composite", 0.0)
    if fom_c >= 0.55:
        p += 0.22
        notes.append(f"Good free-FOM composite on seed ({fom_c:.3f})")
    elif fom_c >= 0.40:
        p += 0.12
    elif fom_c >= 0.30:
        p += 0.04
    else:
        p -= 0.06
        notes.append("Low free-FOM on seed phases (prior may be weak)")

    r_pos = feats.get("R_pos", 1.0)
    if r_pos < 0.35:
        p += 0.08
    elif r_pos > 0.55:
        p -= 0.05

    d_min = feats.get("d_min")
    if d_min is not None:
        if d_min <= 1.0:
            p += 0.08
            notes.append("High resolution (d_min ≤ 1.0 Å)")
        elif d_min <= 1.2:
            p += 0.04
        elif d_min >= 1.6:
            p -= 0.06
            notes.append("Lower resolution: prefer EDM/DM hybrid")

    n_as = feats.get("N_asym", 20.0)
    if n_as <= 40:
        p += 0.05
    elif n_as >= 120:
        p -= 0.08
        notes.append("Large N_asym: harder seed-only path")

    p = float(np.clip(p, 0.02, 0.95))
    return p, notes


def _estimate_mpe_corr(feats: Dict[str, float], p_success: float) -> Tuple[float, float]:
    """
    Map free-FOM / p_success → rough MPE (°) and phase correlation estimates.

    Calibrated loosely: random phases MPE ~90°, CORR~0; good seeds MPE ≲30°,
    CORR ≳0.5. Not a substitute for oracle MPE_seed / CORR_seed.
    """
    fom_c = feats.get("free_fom_composite", 0.0)
    # blend free FOM and success probability
    quality = 0.55 * fom_c + 0.45 * p_success
    # MPE: ~90° at quality 0 → ~15° at quality 1
    mpe = 90.0 - 75.0 * float(np.clip(quality, 0.0, 1.0))
    # CORR ≈ cos(MPE in rad) rough for circular stats
    corr = float(np.cos(np.deg2rad(mpe)))
    # also pull corr toward free FOM
    corr = 0.6 * corr + 0.4 * (2.0 * fom_c - 1.0)
    corr = float(np.clip(corr, -1.0, 1.0))
    return float(mpe), corr


def _try_sklearn_predict(feats: Dict[str, float], model_path: Optional[Path]) -> Optional[Tuple[int, float, str]]:
    """Optional RF path. Returns (class, proba, method) or None."""
    try:
        import joblib  # type: ignore
    except Exception:
        joblib = None  # type: ignore
    try:
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
    except Exception:
        return None

    path = model_path
    if path is None:
        # default location if user trained one
        cand = Path(__file__).resolve().parents[3] / "models" / "seed_quality_rf.joblib"
        if not cand.is_file():
            cand = Path(__file__).resolve().parents[2] / "models" / "seed_quality_rf.joblib"
        path = cand if cand.is_file() else None
    if path is None or not Path(path).is_file():
        return None
    if joblib is None:
        return None
    try:
        bundle = joblib.load(path)
        clf = bundle["model"] if isinstance(bundle, dict) else bundle
        feature_names = (
            bundle.get("feature_names")
            if isinstance(bundle, dict)
            else [
                "max_W",
                "N_asym",
                "Vol",
                "seed_fraction",
                "free_fom_composite",
                "mean_E_seed",
            ]
        )
        x = np.array([[feats.get(k, 0.0) for k in feature_names]], dtype=np.float64)
        if hasattr(clf, "predict_proba"):
            proba = float(clf.predict_proba(x)[0, 1])
        else:
            proba = float(clf.predict(x)[0])
        cls = 1 if proba >= 0.5 else 0
        return cls, proba, "sklearn_rf"
    except Exception:
        return None


def predict_seed_quality(
    hkl: np.ndarray,
    amplitudes: np.ndarray,
    cell: np.ndarray,
    seed_phases: np.ndarray,
    seed_idx: Optional[np.ndarray] = None,
    *,
    seed_fraction: Optional[float] = None,
    n_asym: Optional[float] = None,
    n_atoms_user: Optional[int] = None,
    d_min: Optional[float] = None,
    free_fom_seed: Optional[Dict] = None,
    class1_threshold: float = 0.45,
    model_path: Optional[Union[str, Path]] = None,
    use_sklearn: bool = True,
) -> Dict[str, Any]:
    """
    Predict AI-PhaSeed seed quality (Class 0/1 style).

    Parameters
    ----------
    hkl, amplitudes, cell : reflection geometry
    seed_phases : full-length AI (or partial) phase vector (radians)
    seed_idx : indices of the strong seed set (optional; auto top-|E|)
    seed_fraction : actual |S|/N if known
    n_asym / n_atoms_user : ASU atom count if known
    d_min : high-resolution limit (Å)
    free_fom_seed : precomputed free_fom dict (optional)
    class1_threshold : P(success) cutoff for Class 1 (default 0.45)
    model_path : optional sklearn joblib path
    use_sklearn : try RF model if available

    Returns
    -------
    dict with keys:
      predicted_class, success_probability, predicted_mpe_deg, predicted_corr,
      features, warning, recommend_fallback, method, notes
    """
    feats = extract_seed_features(
        hkl,
        amplitudes,
        cell,
        seed_phases,
        seed_idx=seed_idx,
        seed_fraction=seed_fraction,
        n_asym=n_asym,
        n_atoms_user=n_atoms_user,
        d_min=d_min,
        free_fom_seed=free_fom_seed,
    )

    method = "heuristic"
    sk = None
    if use_sklearn:
        sk = _try_sklearn_predict(
            feats, Path(model_path) if model_path else None
        )

    if sk is not None:
        cls, p_succ, method = sk
        notes = ["sklearn RF model used"]
    else:
        p_succ, notes = _heuristic_success_probability(feats)
        cls = 1 if p_succ >= class1_threshold else 0

    mpe, corr = _estimate_mpe_corr(feats, p_succ)

    warning = None
    recommend_fallback = False
    if cls == 0 or p_succ < 0.30:
        recommend_fallback = True
        warning = (
            "Predicted seed quality is low (Class 0). "
            "Consider partial-φ / fragment seed, larger seed set, "
            "or classical ensemble; AI-PhaSeed alone may not solve."
        )
        notes.append("recommend_fallback=True")
    elif cls == 1:
        notes.append("Class 1: hybrid extension has higher chance of success")

    report = SeedQualityReport(
        predicted_class=int(cls),
        success_probability=float(p_succ),
        predicted_mpe_deg=float(mpe),
        predicted_corr=float(corr),
        features=feats,
        warning=warning,
        recommend_fallback=recommend_fallback,
        method=method,
        notes=notes,
    )
    return report.to_dict()


def oracle_seed_metrics(
    seed_phases: np.ndarray,
    true_phases: np.ndarray,
    seed_idx: np.ndarray,
    amplitudes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Oracle MPE_seed / CORR_seed on the seed set (for benchmarks & RF labels).

    CORR_seed = Re ⟨exp i(φ_pred − φ_true)⟩ weighted by |F| if given.
    """
    from grok_phase_solver.metrics.phase_error import mean_phase_error, wrap_phase

    si = np.asarray(seed_idx, dtype=int)
    pred = np.asarray(seed_phases, dtype=np.float64)[si]
    true = np.asarray(true_phases, dtype=np.float64)[si]
    mpe = float(mean_phase_error(pred, true))
    dphi = wrap_phase(pred - true)
    if amplitudes is not None:
        w = np.asarray(amplitudes, dtype=np.float64)[si]
        w = w / (np.sum(w) + 1e-16)
        corr = float(np.sum(w * np.cos(dphi)))
    else:
        corr = float(np.mean(np.cos(dphi)))
    return {
        "MPE_seed_deg": mpe,
        "CORR_seed": corr,
        "n_seed": float(len(si)),
    }


def label_class_from_oracle(
    mpe_seed_deg: float,
    corr_seed: float,
    mpe_class1_max: float = 40.0,
    corr_class1_min: float = 0.40,
) -> int:
    """
    Binary Class label from oracle metrics (training proxy for paper Class 1).

    Paper Class 1 is defined by high *solution efficiency*; we use seed accuracy
    as a supervised proxy when building synthetic RF training sets.
    """
    if mpe_seed_deg <= mpe_class1_max and corr_seed >= corr_class1_min:
        return 1
    return 0
