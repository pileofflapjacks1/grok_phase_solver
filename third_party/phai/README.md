# PhAI third-party integration

## Official source

**Paper:** Larsen, A. S., Rekis, T. & Madsen, A. Ø. (2024). *PhAI: A deep-learning approach to solve the crystallographic phase problem.* Science **385**, 522–528.  
DOI: [10.1126/science.adn2777](https://doi.org/10.1126/science.adn2777)

**Code & data archive (ERDA, University of Copenhagen):**  
https://erda.ku.dk/archives/e3be15d017d8c4fe81402da833e26894/published-archive.html

The archive includes training code, analysis scripts, training data, and model parameters.

## How to integrate

1. Download the ERDA archive (requires browser / their download UI).
2. Extract model checkpoints and example scripts into this directory, e.g.:

```text
third_party/phai/
  README.md          # this file
  weights/           # place *.pt / *.ckpt here
  upstream/          # optional: vendored training/inference scripts
```

3. Install ML extras:

```bash
pip install 'grok-phase-solver[ml]'
```

4. Point `PhAIConfig.weights_path` or `third_party_dir` at the checkpoints.

## License note

Respect the license and citation requirements of the PhAI authors when
redistributing weights or code. This repository does **not** vendor their
binaries by default; we provide an interface and physics baselines that
can be seeded by PhAI predictions once weights are present.

## Citation

```bibtex
@article{Larsen2024PhAI,
  author  = {Larsen, Anders S. and Rekis, Toms and Madsen, Anders {\O}.},
  title   = {PhAI: A deep-learning approach to solve the crystallographic phase problem},
  journal = {Science},
  volume  = {385},
  pages   = {522--528},
  year    = {2024},
  doi     = {10.1126/science.adn2777}
}
```
