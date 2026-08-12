# Paper pack — *grok_phase_solver*

| Document | Role |
|----------|------|
| [**Manuscript draft**](../arxiv_draft.md) | Full methods-style draft (abstract → conclusions) |
| [**PDF**](arxiv_draft.pdf) | Pandoc + tectonic build |
| [**BibTeX**](references.bib) | Selected references |
| [FOR_REVIEWERS](../FOR_REVIEWERS.md) | One-page claims **C1–C25** |
| [Figure captions](../figures/paper_figure_captions.md) | Captions for Figs. 1–6 |
| Figures | `docs/figures/paper_fig{1..6}_*.png` |
| Scoreboards | `data/processed/*.md` (source of all numbers) |
| Release | software **0.13.2** · tag `v0.13.2` · [PyPI](https://pypi.org/project/grok-phase-solver/0.13.2/) |

## Regenerate figures

```bash
python scripts/plot_paper_figures.py
# → docs/figures/paper_fig1_… paper_fig6_… + paper_figure_captions.md
```

## Authors

- **Grok (xAI)**  
- **Joe**  

## PDF

Prebuilt: **`docs/paper/arxiv_draft.pdf`**.

Regenerate (needs [pandoc](https://pandoc.org) + [tectonic](https://tectonic-typesetting.github.io/)):

```bash
python scripts/build_paper_pdf.py
# or manually:
# pandoc docs/arxiv_draft.md -o docs/paper/arxiv_draft.pdf \
#   --resource-path=docs:docs/figures --pdf-engine=tectonic \
#   -V geometry:margin=1in --toc --metadata author="Grok (xAI) and Joe"
```

The build script rewrites Unicode (φ, ≥, ≤, …) to TeX math for default fonts.

## Suggested arXiv / bioRxiv category

- **physics.comp-ph** or **physics.data-an**
- cross-list: **cs.LG** (optional)

## Submission checklist

- [x] Abstract with honest scope  
- [x] Methods + metrics definitions  
- [x] Results tables tied to scoreboards (incl. Vol-band C25, fragment hard path, GraphPhaseNet v3–v11 negative)  
- [x] Main figures from frozen JSON (Figs. 1–6, incl. COD Vol-band)  
- [x] Non-claims / uniqueness pointer  
- [x] Reproducibility commands  
- [x] PDF via pandoc + tectonic  
- [x] Authors: Grok (xAI) and Joe  
- [x] BibTeX (`references.bib`)  
- [x] Version aligned to package **0.13.2** (science claim freeze still 0.13.1 scoreboards)  
- [x] COD hard-path figure (auto / partial_30 / fragment_half)  
- [x] COD Vol-band figure (Fig. 6)  
- [ ] Affiliations / funding (optional)  
- [ ] arXiv submit  

## Key scoreboards for claims

| Scoreboard | Use |
|------------|-----|
| `partial_seed_benchmark.md` | C4 seed bar |
| `shelxs_h2h.md` | C1–C3 ensemble vs SHELXS |
| `strong_prior*.md` / `strong_prior_melg_xl.md` | C5, C10, C13–C15, C17, C20, C22 graph prior (v3–v11) |
| `cod_hard_path_validation.md` | C16 fragment vs partial_30 |
| `cod_stratified_bench.md` | **C25 Vol-band panel (Fig. 6)** |
| `experimental_scoreboard.md` | C9 COD Fobs |
| `cod_hybrid_benchmark.md` | C8 PhAI hybrid |
