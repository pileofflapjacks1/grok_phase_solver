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
                    │  Have partial phases │
                    │  (HA/MAD/MR/SHELXS)? │
                    └──────────┬──────────┘
                         yes   │   no
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     partial_phaseed                    Resolution good?
     --phase-seed-csv                     (d ≲ 1.1–1.2 Å)
              │                          yes╱        ╲no
              │                           ▼          ▼
              │                       ensemble     hard path:
              │                       (auto)     strong_prior /
              │                                  CF / shelxs
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                           Inspect free FOM,
                           density_slice, peaks
                                  │
                    ┌─────────────┴─────────────┐
                    │ map ugly / unsolved?       │
                    │ add partial φ or SHELXE    │
                    └─────────────┬─────────────┘
                                  ▼
                           trial.res → SHELXL
```

| Situation | Command |
|-----------|---------|
| Default | `gps-solve --hkl … --ins … --method auto` |
| Easy / high-res | `auto` → **ensemble** |
| Hard, pure ab initio | `auto` → prior/CF; expect struggle |
| Hard, have HA/partial φ | `--method partial_phaseed --phase-seed-csv known.csv` |
| External classical solve | `--method shelxs` or `shelxs+shelxe` |
| After any solve | Open `trial.res` → **SHELXL** / Olex2 |
"""
