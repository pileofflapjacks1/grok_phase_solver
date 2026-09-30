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
    """Markdown snippet for report.md: Olex2 hand-build, then SHELXL."""
    out_dir = Path(out_dir)
    bin_hint = shelxl_path or "ShelX/shelxl  # or shelxl on PATH"
    return f"""### Refine with SHELXL (after gps-solve)

gps-solve writes **`trial.res`** as Q peaks for **Olex2 hand-build** (not a
SHELXL start). Open `olex2_handbuild.md` in this folder. Refinement is external
and comes **after** element assignment:

```bash
# 1) File → Open trial.res in Olex2; View → Work → Info for Z / Z′
# 2) Assign C/N/O from chemistry; delete junk Q peaks
# 3) Only then copy a built molecule and refine (academic SHELXL; not redistributed)
cp built.res ./work.ins
cp /path/to/experiment.hkl ./work.hkl
{bin_hint} work
```

gps-solve does **not** replace SHELXL R-factor refinement.
"""


def workflow_decision_tree_md() -> str:
    return """## Which method should I use?

```text
                    ┌─────────────────────┐
                    │  Have partial info?  │
                    │  φ / fragment / HA   │
                    └──────────┬──────────┘
                         yes   │   no
              ┌───────────────┴───────────────┐
              ▼                                 ▼
     partial_phaseed                    Resolution good?
     seed source:                         (d ≲ 1.1–1.2 Å)
       --phase-seed-csv                  yes╱        ╲no
       --phase-seed-res                   ▼          ▼
       --seed-peaks-csv               ensemble     hard path:
       --native + --derivative        (auto)     CF last-resort;
       gps-make-seed …                           prefer partial_phaseed
              │                                       │
              └──────────────────┬───────────────────┘
                                  ▼
                           Inspect free FOM,
                           seed quality section,
                           density_slice, peaks
                                  │
                    ┌────────────┴─────────────┐
                    │ map ugly / unsolved?       │
                    │ enlarge seed or SHELXE     │
                    └────────────┬─────────────┘
                                  ▼
                           trial.res → Olex2 hand-build → SHELXL
```

| Situation | Command |
|-----------|---------|
| Default | `gps-solve --hkl … --ins … --method auto` |
| Easy / high-res | `auto` → **ensemble** |
| Hard, pure ab initio | `auto` → CF last-resort (~0% strict); prefer `partial_phaseed` + seed |
| Weak auto, no fragment | `--retry-with-peaks` → `retry_peaks/` |
| Hard + known φ | `--method partial_phaseed --phase-seed-csv known.csv` |
| Hard + SHELXS fragment | `--phase-seed-res model.res` (method partial or auto) |
| Hard + density peaks | `--seed-peaks-csv peaks.csv` |
| Hard + isomorphous HA | `--native-hkl … --derivative-hkl … --method ha_phaseed` |
| Build seed only | `gps-make-seed --hkl … --from-res model.res -o seed.csv` |
| External classical solve | `--method shelxs` or `shelxs+shelxe` |
| After any solve | Read `report.md` **Next action** → `trial.res` → **Olex2 hand-build** → SHELXL |
"""
