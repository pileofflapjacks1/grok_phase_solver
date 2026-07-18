# Paper pack — *grok_phase_solver*

| Document | Role |
|----------|------|
| [**Manuscript draft**](../arxiv_draft.md) | Full methods-style draft (abstract → conclusions) |
| [**PDF**](arxiv_draft.pdf) | Pandoc + tectonic build |
| [**BibTeX**](references.bib) | Selected references |
| [FOR_REVIEWERS](../FOR_REVIEWERS.md) | One-page claims C1–C9 |
| [Figure captions](../figures/paper_figure_captions.md) | Captions for Figs. 1–4 |
| Figures | `docs/figures/paper_fig{1..4}_*.png` |
| Scoreboards | `data/processed/*.md` (source of all numbers) |
| Release | package **0.2.1** on [PyPI](https://pypi.org/project/grok-phase-solver/) |

## Regenerate figures

```bash
python scripts/plot_paper_figures.py
# → docs/figures/paper_fig1_… paper_fig4_… + paper_figure_captions.md
```

## Authors (current draft)

- **Grok (xAI)** — primary research and software contributor  
- Additional human co-authors TBD  

## PDF

Prebuilt: **`docs/paper/arxiv_draft.pdf`** (~287 KB, TOC + figures).

Regenerate (needs [pandoc](https://pandoc.org) + [tectonic](https://tectonic-typesetting.github.io/)):

```bash
python scripts/build_paper_pdf.py
# or manually:
# pandoc docs/arxiv_draft.md -o docs/paper/arxiv_draft.pdf \
#   --resource-path=docs:docs/figures --pdf-engine=tectonic \
#   -V geometry:margin=1in --toc --metadata author="Grok (xAI)"
```

The build script rewrites Unicode (φ, ≥, ≤, …) to TeX math for default fonts.

## Suggested arXiv / bioRxiv category

- **physics.comp-ph** or **physics.data-an**
- cross-list: **cs.LG** (optional)

## Submission checklist

- [x] Abstract with honest scope  
- [x] Methods + metrics definitions  
- [x] Results tables tied to scoreboards  
- [x] Four main figures from frozen JSON  
- [x] Non-claims / uniqueness pointer  
- [x] Reproducibility commands  
- [x] PDF via pandoc + tectonic  
- [x] Author: Grok (xAI); further co-authors TBD  
- [ ] Full bibliography (BibTeX)  
- [ ] arXiv submit  
