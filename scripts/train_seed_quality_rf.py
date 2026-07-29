#!/usr/bin/env python3
"""
Train a Carrozzini-style Class 0/1 seed-quality RandomForest (v0.8).

Labels are **oracle** on synthetic hard cells: Class 1 if strong-|E| seed
fraction within 20° of truth is ≥ bar (default 0.28), else Class 0.
Features are truth-free (max_W, Vol, N_asym, seed_fraction, free-FOM proxies).

Writes:
  data/processed/seed_quality_rf.joblib
  data/processed/seed_quality_rf.{json,md}

Requires: scikit-learn, joblib (optional extras: seed-quality / all).

Honest: not the published RF on 1505 COD structures; heuristic always remains.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grok_phase_solver.data.synthetic import generate_random_organic
from grok_phase_solver.metrics.seed_quality import (
    DEFAULT_RF_FEATURE_NAMES,
    extract_seed_features,
    oracle_seed_metrics,
    save_seed_quality_rf,
    train_seed_quality_rf_from_matrix,
)
from grok_phase_solver.metrics.strong_seed import full_and_strong_metrics
from grok_phase_solver.solvers.ai_phaseed import select_seed_indices
from grok_phase_solver.solvers.baseline import structure_to_fcalc


def build_dataset(n: int = 200, seed: int = 0, bar: float = 0.28):
    rng = np.random.default_rng(seed)
    X_rows = []
    y_rows = []
    for i in range(n):
        n_atoms = int(rng.integers(10, 20))
        d_min = float(rng.choice([1.2, 1.5, 1.7, 2.0]))
        st = generate_random_organic(
            n_atoms=n_atoms, seed=int(rng.integers(0, 2**31 - 1)), space_group="P1"
        )
        data = structure_to_fcalc(st, d_min=d_min)
        hkl, amp, ph_t = data["hkl"], data["amplitudes"], data["phases"]
        cell = st.cell
        # Seed = top 25% |E| with *true* phases → high quality control
        # Plus a degraded channel: true + noise to create Class 0/1 mix
        idx = select_seed_indices(hkl, amp, cell, n_seed=max(15, int(0.25 * len(amp))), by="E")
        noise_deg = float(rng.choice([5.0, 15.0, 25.0, 40.0, 60.0, 80.0]))
        seed_ph = ph_t.copy()
        seed_ph[idx] = ph_t[idx] + np.deg2rad(noise_deg) * rng.normal(size=len(idx))
        # random elsewhere
        mask_all = np.ones(len(amp), dtype=bool)
        mask_all[idx] = False
        seed_ph[mask_all] = rng.uniform(-np.pi, np.pi, size=int(mask_all.sum()))

        sm = full_and_strong_metrics(
            seed_ph, ph_t, hkl, amp, cell, fraction=0.30, within_deg=20.0
        )
        # Label Class 1 if strong seed quality clears operational bar
        y = 1 if float(sm["frac_within_deg"]) >= bar else 0

        feats = extract_seed_features(
            hkl, amp, cell, seed_ph, seed_idx=idx, d_min=d_min
        )
        # ensure d_min present
        feats["d_min"] = float(d_min)
        row = [float(feats.get(k, 0.0)) for k in DEFAULT_RF_FEATURE_NAMES]
        X_rows.append(row)
        y_rows.append(y)
        if (i + 1) % 40 == 0:
            print(f"  built {i+1}/{n}  class1_so_far={np.mean(y_rows):.0%}", flush=True)

    return np.asarray(X_rows, dtype=np.float64), np.asarray(y_rows, dtype=int)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Train seed-quality RF (Carrozzini-style)")
    p.add_argument("--n", type=int, default=240)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bar", type=float, default=0.28, help="frac≤20° Class 1 threshold")
    p.add_argument(
        "--out",
        type=str,
        default="data/processed/seed_quality_rf.npz",
        help="Output path (.npz pure-NumPy logistic, or .joblib if sklearn works)",
    )
    args = p.parse_args()

    print(f"=== Seed quality RF train N={args.n} bar={args.bar} ===", flush=True)
    X, y = build_dataset(n=args.n, seed=args.seed, bar=args.bar)
    print(f"labels class1={y.mean():.1%} n={len(y)}", flush=True)
    clf, meta = train_seed_quality_rf_from_matrix(
        X, y, feature_names=DEFAULT_RF_FEATURE_NAMES, seed=args.seed
    )
    meta["bar_frac_within_20"] = float(args.bar)
    meta["n_total"] = int(len(y))
    meta["class1_rate"] = float(y.mean())

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    save_seed_quality_rf(clf, out, feature_names=DEFAULT_RF_FEATURE_NAMES, meta=meta)

    js = out.with_suffix(".json")
    js.write_text(json.dumps(meta, indent=2, default=str))
    md = out.with_suffix(".md")
    imp = meta.get("feature_importance") or {}
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:8]
    lines = [
        "# Seed quality RF (v0.8)",
        "",
        "Carrozzini-aligned Class 0/1 features; **synthetic oracle labels**.",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| n_total | {meta['n_total']} |",
        f"| test accuracy | **{100*meta.get('accuracy', 0):.1f}%** |",
        f"| ROC-AUC | {meta.get('roc_auc', float('nan'))} |",
        f"| Class 1 rate (train set) | {100*meta['class1_rate']:.1f}% |",
        f"| bar frac≤20° | {args.bar} |",
        "",
        "## Top feature importance",
        "",
    ]
    for k, v in top:
        lines.append(f"- `{k}`: {v:.3f}")
    lines.extend(
        [
            "",
            "## Honest limits",
            "",
            "- Not the published RF trained on 1505 COD structures.",
            "- Heuristic predictor remains the default fallback without sklearn/joblib.",
            f"- Bundle: `{out.name}`",
            "",
        ]
    )
    md.write_text("\n".join(lines))
    print(json.dumps({k: meta[k] for k in ("accuracy", "roc_auc", "n_train", "n_test") if k in meta}, indent=2))
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {md}", flush=True)


if __name__ == "__main__":
    main()
