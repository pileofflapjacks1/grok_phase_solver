"""
End-to-end crystallography workflow helpers: solve → optional SHELXE → SHELXL hints.

Does not reimplement SHELXL; documents how to finish with the local ShelX/ suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def shelxl_refinement_instructions(
    out_dir: Path,
    *,
    shelxl_path: Optional[str] = None,
    hkl_name: str = "job.hkl",
) -> str:
    """Markdown snippet for report.md: refine trial.res with SHELXL."""
    out_dir = Path(out_dir)
    bin_hint = shelxl_path or "ShelX/shelxl  # or shelxl on PATH"
    return f"""### Refine with SHELXL (after gps-solve)

gps-solve writes **`trial.res`** (peak list). Refinement is done externally:

```bash
# 1) Copy experimental intensities next to a working name
cp /path/to/your.hkl {out_dir / hkl_name}

# 2) Edit trial.res: assign real elements to Q peaks, fix composition (SFAC/UNIT)

# 3) Run SHELXL (academic binary; not redistributed)
{bin_hint} trial   # reads trial.ins/res + trial.hkl — rename as needed
```

Typical rename pattern:

```bash
cp {out_dir}/trial.res ./work.ins
cp /path/to/experiment.hkl ./work.hkl
ShelX/shelxl work
```

Open `work.res` / CIF in Olex2. gps-solve does **not** replace SHELXL R-factor refinement.
"""


def workflow_decision_tree_md() -> str:
    return """## Which method should I use?

```text
                    ┌─────────────────────┐
                    │  Have partial info?  │
                    │  φ / fragment / HA   │
                    └──────────┬──────────┘
                         yes   │   no
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     partial_phaseed                    Resolution good?
     seed source:                         (d ≲ 1.1–1.2 Å)
       --phase-seed-csv                  yes╱        ╲no
       --phase-seed-res                   ▼          ▼
       --seed-peaks-csv               ensemble     hard path:
       --native + --derivative        (auto)     strong_prior /
       gps-make-seed …                           CF / shelxs
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                           Inspect free FOM,
                           seed quality section,
                           density_slice, peaks
                                  │
                    ┌─────────────┴─────────────┐
                    │ map ugly / unsolved?       │
                    │ enlarge seed or SHELXE     │
                    └─────────────┬─────────────┘
                                  ▼
                           trial.res → SHELXL
```

| Situation | Command |
|-----------|---------|
| Default | `gps-solve --hkl … --ins … --method auto` |
| Easy / high-res | `auto` → **ensemble** |
| Hard, pure ab initio | `auto` → prior/CF; expect struggle |
| Hard + known φ | `--method partial_phaseed --phase-seed-csv known.csv` |
| Hard + SHELXS fragment | `--phase-seed-res model.res` (method partial or auto) |
| Hard + density peaks | `--seed-peaks-csv peaks.csv` |
| Hard + isomorphous HA | `--native-hkl … --derivative-hkl … --method ha_phaseed` |
| Build seed only | `gps-make-seed --hkl … --from-res model.res -o seed.csv` |
| External classical solve | `--method shelxs` or `shelxs+shelxe` |
| After any solve | Open `trial.res` → **SHELXL** / Olex2 |
"""
