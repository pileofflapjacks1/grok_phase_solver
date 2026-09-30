"""Olex2 hand-build checklist written next to trial.res.

``trial.res`` is a Q-peak list for peak picking. It is not a SHELXL starting
model and not typed element labels unless an explicit research flag is on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

PathLike = Union[str, Path]


def olex2_handbuild_markdown(
    *,
    space_group: Optional[str] = None,
    n_peaks: Optional[int] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> str:
    sg = space_group or "unknown"
    nq = "—" if n_peaks is None else str(int(n_peaks))
    d = dict(diagnostics or {})
    fom = d.get("free_fom_composite")
    try:
        fom_s = f"{float(fom):.3f}" if fom is not None else "—"
    except (TypeError, ValueError):
        fom_s = "—"
    return f"""# Olex2 hand-build from gps-solve Q peaks

`trial.res` is a **Q-peak list** for Olex2 peak picking / hand-build.
It is **not** a SHELXL starting model. `SFAC C` and `UNIT 1` are placeholders.

- Space-group hint: `{sg}`
- Q peaks written: {nq} (budgeted unique-ASU list)
- Free FOM (rank only): {fom_s}

## Open

1. In Olex2: **File → Open** `trial.res` (this `--out` folder).
2. Header should read `TITL gps-solve hand-build peaks (not a SHELXL start)`.
3. Confirm `SFAC C`, `Q1…Qn`, and LATT/SYMM matching the space group
   (identity omitted — SHELX convention).

## Where Z, Z′, polymeric live

Olex2 does not show a big “polymeric” banner. Open **View → Work → Info**.
That panel lists formula, **Z**, **Z′**, and whether Olex2 thinks the packing
is polymeric. `UNIT 1` in this file is a dummy — do not treat it as chemistry.

## Build

1. Peak-pick / assign C, N, O, … from residual maps and the known formula.
2. Delete junk Q peaks. Do **not** run SHELXL on the raw Q list.
3. `--retry-with-peaks` is **peaks-as-carbon, not a fragment.**
4. When a chemically plausible molecule is built, write a real `.ins` and
   refine with SHELXL:

```bash
# Only after element assignment / a real molecule:
cp built.res work.ins
cp your.hkl work.hkl
ShelX/shelxl work
```

Free FOM ranks trials. Chemical sense + SHELXL R1 decide.
"""


def write_olex2_handbuild_md(
    out_dir: PathLike,
    *,
    space_group: Optional[str] = None,
    n_peaks: Optional[int] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> Path:
    path = Path(out_dir) / "olex2_handbuild.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        olex2_handbuild_markdown(
            space_group=space_group,
            n_peaks=n_peaks,
            diagnostics=diagnostics,
        )
    )
    return path
