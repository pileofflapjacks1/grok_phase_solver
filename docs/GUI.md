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

1. **Scenario wizard** — Easy / hard / known φ / fragment / HA / SHELXS / advanced  
2. **Upload** experimental `.hkl` / `.mtz` and optional `.ins`  
3. **Cell paste** — comma list or full `CELL 0.71 a b c α β γ` line  
4. **Hard-path seeds** — phase CSV, fragment `.res`, or `peaks.csv`  
5. **Packaged demos** — easy ensemble; hard + 30% φ; hard + fragment  
6. **Inspect** free FOM, density, peaks, report, **quality hints**  
7. **Retry with peaks as seed** if the first map looks poor  
8. **SHELXL handoff** — copy-paste shell snippet + download `trial.res`  
9. **Downloads** — individual files or zip; last work dir remembered in the sidebar  

## Architecture

| Layer | Module |
|-------|--------|
| UI | `src/grok_phase_solver/gui/app.py` (Streamlit) |
| Job runner | `gui/backend.py` → `pipeline.solve` + `pipeline.export` |
| Launch | `gui/launch.py` → `streamlit run …` |

No separate phasing engine: the GUI only stages files and calls the library.

### Backend helpers (testable)

- `parse_cell_string` — CELL line / free-form cell  
- `resolve_wizard` — scenario → method defaults  
- `format_user_error` / `map_quality_hints` — scientist-facing messages  
- `shelxl_handoff_snippet` — post-solve refinement recipe  

## Limits

- Local single-user tool (not multi-tenant server hardening).  
- Does **not** replace SHELXL / Olex2 refinement.  
- Optional heavy methods (PhAI, SHELXS) need weights/binaries as for the CLI.  
- Streamlit does not support mid-job **cancel** of native NumPy solvers cleanly; stop the browser tab / Ctrl+C the process if needed.  
- For automation and CI, prefer `gps-solve` on the command line.  
