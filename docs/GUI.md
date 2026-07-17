# Scientist GUI (Streamlit)

Thin browser front end for the same pipeline as `gps-solve`.

## Install & launch

```bash
python -m pip install -e ".[gui]"
gps-gui
# or: python -m grok_phase_solver.gui
# or: python scripts/run_gui.py
```

Opens **http://localhost:8501** by default.

## What you can do

1. **Upload** experimental `.hkl` and optional `.ins` (or type cell + space group).
2. **Choose method** (`auto`, `ensemble`, `partial_phaseed`, PhAI, SHELXS, …).
3. **Hard-path seeds** — phase CSV, fragment `.res`, or `peaks.csv`.
4. **Packaged demos** — easy ensemble; hard + 30% oracle φ; hard + fragment `.res`.
5. **Inspect** free FOM, density slice, peaks, full `report.md` (including seed quality).
6. **Download** `trial.res`, CSVs, or a zip of the export folder → Olex2 / SHELXL.

## Architecture

| Layer | Module |
|-------|--------|
| UI | `src/grok_phase_solver/gui/app.py` (Streamlit) |
| Job runner | `gui/backend.py` → `pipeline.solve` + `pipeline.export` |
| Launch | `gui/launch.py` → `streamlit run …` |

No separate phasing engine: the GUI only stages files and calls the library.

## Limits

- Local single-user tool (not multi-tenant server hardening).
- Does **not** replace SHELXL / Olex2 refinement.
- Optional heavy methods (PhAI, SHELXS) need weights/binaries as for the CLI.
- For automation and CI, prefer `gps-solve` on the command line.
