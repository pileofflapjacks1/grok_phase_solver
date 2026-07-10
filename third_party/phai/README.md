# PhAI third-party integration

## Official sources

**Paper:** Larsen, A. S., Rekis, T. & Madsen, A. Ø. (2024). *PhAI: A deep-learning approach to solve the crystallographic phase problem.* Science **385**, 522–528. doi:[10.1126/science.adn2777](https://doi.org/10.1126/science.adn2777)

**Public test notebook / code:** https://github.com/AndersOMadsen/PhAI  

**Full archive (training data, ~9.7 GB):**  
https://erda.ku.dk/archives/e3be15d017d8c4fe81402da833e26894/published-archive.html  
(`PhAI.zip`)

## Quick install (public weights)

From the authors’ Colab notebook (Google Drive):

```bash
cd third_party/phai/weights
# Requires: pip install gdown torch einops
gdown 1_eleZ6dBvdKQQeZwxeOJ82g5lPVzmb2M   # one of the notebook artifacts
gdown 14lqkA_Frfy8WpoYyJ-v2sfKkhfPTlNFO
gdown 10U-JUhNQKvoYCRPAv5k-iC2D5vdq6MxM
gdown 1Str3GWahzB1QZtpU2obBj-KSbH9JCV8P
# Rename the .pth file to PhAI_model.pth if needed
ls -la
```

Public Python helpers are vendored under `upstream/` (from `PhAI_files_public`).

## Use in grok_phase_solver

```python
from grok_phase_solver.models.phai_runner import phai_available, PhAIRunner

if phai_available():
    runner = PhAIRunner(device="cpu")
    phases, info = runner.predict(hkl, amplitudes, n_cycles=5)
```

Or run the full scoreboard:

```bash
python scripts/run_scoreboard.py
```

## Limitations (do not ignore)

- Public model is **oriented to P2₁/c** and a fixed reciprocal grid (`max_index=10`).
- Not a general macromolecular phaser.
- Always reimpose observed `|F|` and consider hybrid CF/recycle polish after PhAI.
- Respect the authors’ license and cite the Science paper.

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
