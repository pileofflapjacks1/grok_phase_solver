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


# Carrozzini-aligned default feature order for sklearn RF (v0.8)
DEFAULT_RF_FEATURE_NAMES: List[str] = [
    "max_W",
    "N_asym",
    "Vol",
    "seed_fraction",
    "free_fom_composite",
    "mean_E_seed",
    "median_E_seed",
    "R_pos",
    "density_asym",
    "d_min",
    "n_seed",
    "excess_kurtosis",
]


def default_seed_quality_rf_paths() -> List[Path]:
    """Search paths for a persisted RF classifier (first hit wins)."""
    here = Path(__file__).resolve()
    roots = [
        here.parents[3],  # repo root when installed as src layout
        here.parents[2],
        Path.cwd(),
    ]
    names = [
        Path("data") / "processed" / "seed_quality_rf.npz",
        Path("data") / "processed" / "seed_quality_rf.joblib",
        Path("models") / "seed_quality_rf.joblib",
        Path("models") / "seed_quality_rf.npz",
        Path("seed_quality_rf.npz"),
        Path("seed_quality_rf.joblib"),
    ]
    out: List[Path] = []
    for r in roots:
        for n in names:
            out.append(r / n)
    return out


def save_seed_quality_rf(
    clf: Any,
    path: Union[str, Path],
    *,
    feature_names: Optional[Sequence[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Persist classifier bundle.

    Prefers joblib for sklearn models; pure-NumPy logistic is saved as ``.npz``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(feature_names or DEFAULT_RF_FEATURE_NAMES)
    meta = meta or {}

    if isinstance(clf, _LogisticSeedClassifier) or path.suffix == ".npz":
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        np.savez_compressed(
            path,
            kind=np.array("numpy_logistic"),
            w=np.asarray(clf.w if hasattr(clf, "w") else clf["w"]),
            b=np.array(float(clf.b if hasattr(clf, "b") else clf["b"])),
            mu=np.asarray(clf.mu if hasattr(clf, "mu") else clf["mu"]),
            sig=np.asarray(clf.sig if hasattr(clf, "sig") else clf["sig"]),
            feature_names=np.array(names, dtype=object),
            meta_json=np.array(str(meta)),
            version=np.array("0.8.0"),
        )
        return path

    try:
        import joblib  # type: ignore

        bundle = {
            "model": clf,
            "feature_names": names,
            "meta": meta,
            "version": "0.8.0",
        }
        joblib.dump(bundle, path)
        return path
    except Exception:
        # last resort: if clf has logistic attrs, write npz
        if hasattr(clf, "w") and hasattr(clf, "mu"):
            return save_seed_quality_rf(
                clf, path.with_suffix(".npz"), feature_names=names, meta=meta
            )
        raise


def load_seed_quality_rf(path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """Load RF / logistic bundle or None if unavailable."""
    candidates: List[Path] = []
    if path is not None:
        p = Path(path)
        candidates.append(p)
        if p.suffix == ".joblib":
            candidates.append(p.with_suffix(".npz"))
    else:
        candidates.extend(default_seed_quality_rf_paths())
        # also npz siblings
        extra = []
        for c in list(candidates):
            if c.suffix == ".joblib":
                extra.append(c.with_suffix(".npz"))
        candidates.extend(extra)

    for p in candidates:
        if not p.is_file():
            continue
        if p.suffix == ".npz":
            try:
                z = np.load(p, allow_pickle=True)
                clf = _LogisticSeedClassifier(z["w"], float(z["b"]), z["mu"], z["sig"])
                names = [str(x) for x in z["feature_names"].tolist()]
                return {
                    "model": clf,
                    "feature_names": names,
                    "meta": {},
                    "version": "0.8.0",
                    "backend": "numpy_logistic",
                    "_path": str(p),
                }
            except Exception:
                continue
        try:
            import joblib  # type: ignore

            bundle = joblib.load(p)
            if not isinstance(bundle, dict):
                bundle = {"model": bundle, "feature_names": DEFAULT_RF_FEATURE_NAMES}
            bundle["_path"] = str(p)
            return bundle
        except Exception:
            continue
    return None


class _LogisticSeedClassifier:
    """Pure-NumPy logistic classifier (sklearn-free fallback)."""

    def __init__(self, w: np.ndarray, b: float, mu: np.ndarray, sig: np.ndarray):
        self.w = np.asarray(w, dtype=np.float64)
        self.b = float(b)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sig = np.asarray(sig, dtype=np.float64)
        # approximate importance for reporting
        self.feature_importances_ = np.abs(self.w) / (np.abs(self.w).sum() + 1e-16)

    def _z(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mu) / np.maximum(self.sig, 1e-8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self._z(X) @ self.w + self.b
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        p1 = np.asarray(p1, dtype=np.float64).reshape(-1)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _train_logistic_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
    n_iter: int = 400,
    lr: float = 0.15,
) -> _LogisticSeedClassifier:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-8
    Z = (X - mu) / sig
    w = rng.normal(0, 0.01, size=X.shape[1])
    b = 0.0
    # class-balanced weights
    n1 = max(float(y.sum()), 1.0)
    n0 = max(float(len(y) - y.sum()), 1.0)
    sw = np.where(y > 0.5, 0.5 * len(y) / n1, 0.5 * len(y) / n0)
    for _ in range(n_iter):
        z = Z @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        err = (p - y) * sw
        w -= lr * (Z.T @ err) / len(y)
        b -= lr * float(err.mean())
    return _LogisticSeedClassifier(w, b, mu, sig)


def train_seed_quality_rf_from_matrix(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: Optional[Sequence[str]] = None,
    n_estimators: int = 80,
    max_depth: int = 6,
    seed: int = 0,
    class_weight: str = "balanced",
    prefer_sklearn: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Fit a Class 0/1 seed-quality classifier.

    Prefers sklearn RandomForest when importable; otherwise pure-NumPy logistic
    regression (always available). Returns (clf, meta).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if len(X) < 20:
        raise ValueError("need ≥20 labeled seeds to train RF")
    names = list(feature_names or DEFAULT_RF_FEATURE_NAMES[: X.shape[1]])

    # hold-out split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_te = max(5, int(0.25 * len(X)))
    te, tr = idx[:n_te], idx[n_te:]
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

    backend = "numpy_logistic"
    clf: Any = None
    if prefer_sklearn:
        try:
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                random_state=int(seed),
                class_weight=class_weight,
                n_jobs=1,
            )
            clf.fit(X_tr, y_tr)
            backend = "sklearn_rf"
        except Exception:
            clf = None

    if clf is None:
        clf = _train_logistic_numpy(X_tr, y_tr, seed=seed)
        backend = "numpy_logistic"

    if hasattr(clf, "predict_proba"):
        proba = np.asarray(clf.predict_proba(X_te)[:, 1], dtype=np.float64)
    else:
        proba = np.asarray(clf.predict(X_te), dtype=np.float64)
    pred = (proba >= 0.5).astype(int)
    acc = float(np.mean(pred == y_te))

    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        imp = np.zeros(len(names))
    meta: Dict[str, Any] = {
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "accuracy": acc,
        "backend": backend,
        "feature_names": names,
        "feature_importance": {names[i]: float(imp[i]) for i in range(len(names))},
        "class_balance_train": {
            "class0": int(np.sum(y_tr == 0)),
            "class1": int(np.sum(y_tr == 1)),
        },
        "note": (
            "Synthetic/oracle-labeled Class 0/1 model aligned with Carrozzini "
            "feature list. Not the published 1505-COD RF; heuristic fallback always available."
        ),
    }
    # rough AUC
    try:
        if len(np.unique(y_te)) > 1:
            order = np.argsort(-proba)
            y_ord = y_te[order]
            tps = np.cumsum(y_ord == 1)
            fps = np.cumsum(y_ord == 0)
            tps = tps / max(tps[-1], 1)
            fps = fps / max(fps[-1], 1)
            meta["roc_auc"] = float(np.trapz(tps, fps))
    except Exception:
        pass
    return clf, meta


def _try_sklearn_predict(feats: Dict[str, float], model_path: Optional[Path]) -> Optional[Tuple[int, float, str]]:
    """Optional trained Class 0/1 model path. Returns (class, proba, method) or None."""
    bundle = load_seed_quality_rf(model_path)
    if bundle is None:
        return None
    try:
        clf = bundle["model"]
        feature_names = bundle.get("feature_names") or DEFAULT_RF_FEATURE_NAMES
        x = np.array([[float(feats.get(k, 0.0)) for k in feature_names]], dtype=np.float64)
        if hasattr(clf, "predict_proba"):
            proba = float(clf.predict_proba(x)[0, 1])
        else:
            proba = float(clf.predict(x)[0])
        cls = 1 if proba >= 0.5 else 0
        backend = bundle.get("backend")
        if backend is None:
            backend = "numpy_logistic" if isinstance(clf, _LogisticSeedClassifier) else "sklearn_rf"
        method = "sklearn_rf" if backend == "sklearn_rf" else "numpy_logistic"
        return cls, proba, method
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
