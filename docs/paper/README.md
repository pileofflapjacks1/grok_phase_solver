# Paper pack — *grok_phase_solver*

| Document | Role |
|----------|------|
| [**Manuscript draft**](../arxiv_draft.md) | Full methods-style draft (abstract → conclusions) |
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
- [ ] Author list / affiliations  
- [ ] Full bibliography (BibTeX)  
- [ ] Optional: convert MD → PDF via pandoc  

```bash
# Optional local PDF (requires pandoc + latex)
pandoc docs/arxiv_draft.md -o docs/paper/arxiv_draft.pdf \
  --resource-path=docs:docs/figures \
  -V geometry:margin=1in
```
