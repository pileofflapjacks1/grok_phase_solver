# v0.2.1 — grok-phase-solver

Open physics/AI phasing assistant for X-ray crystallography.

## Highlights

- **Scientist CLI** `gps-solve` — HKL/INS → density, peaks, free FOM, **`trial.res`**
- **GUI** `gps-gui` (Streamlit) — scenario wizard, seed uploads, peaks-as-seed retry, SHELXL handoff
- **Partial-φ hard path** — CSV / `.res` / peaks / HA seeds; `gps-make-seed`
- **Honest hard-region science** — partial-φ seed bar (~30%/20°); GraphPhaseNet scale still ~21% ≤20°
- **Experimental COD scoreboard** — 2016452 Fobs (PhAI strict-solves in pipeline); 2100301; 2017775
- **Paper pack** — methods draft + PDF + figures

## Install

```bash
python -m pip install grok-phase-solver
python -m pip install "grok-phase-solver[gui]"   # optional browser UI
gps-solve --help
gps-gui
```

From source:

```bash
git clone https://github.com/pileofflapjacks1/grok_phase_solver.git
cd grok_phase_solver
python -m pip install -e ".[gui]"
```

## Links

| | |
|--|--|
| **PyPI** | https://pypi.org/project/grok-phase-solver/0.2.1/ |
| **Paper PDF** | [docs/paper/arxiv_draft.pdf](https://github.com/pileofflapjacks1/grok_phase_solver/blob/main/docs/paper/arxiv_draft.pdf) |
| **Paper hub** | [docs/paper/README.md](https://github.com/pileofflapjacks1/grok_phase_solver/blob/main/docs/paper/README.md) |
| **For reviewers** | [docs/FOR_REVIEWERS.md](https://github.com/pileofflapjacks1/grok_phase_solver/blob/main/docs/FOR_REVIEWERS.md) |
| **User guide** | [docs/USER_GUIDE.md](https://github.com/pileofflapjacks1/grok_phase_solver/blob/main/docs/USER_GUIDE.md) |
| **Changelog** | [CHANGELOG.md](https://github.com/pileofflapjacks1/grok_phase_solver/blob/main/CHANGELOG.md) |

## Scope (honest)

Best for **small molecules** at good resolution (ensemble path). Hard ab initio remains seed-limited. **Not** a general protein ab initio solver or SHELXL replacement. SHELX binaries and PhAI weights are user-supplied (not redistributed).

## Authors

**Grok (xAI)** and **Joe**.
