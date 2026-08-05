# Release notes — v0.13.0

**Theme:** GraphPhaseNet v11 + CrystalX trial.res typing + XDXD-style generative coords.

## Highlights

1. **GraphPhaseNet v11** (`d_in=34`) — intensity moments, centro/HA cues, stronger multipath.  
   `python scripts/run_strong_prior_v11.py --quick --melgalvis-preset large`

2. **CrystalX-inspired typing** — peak height/geometry → element + H; richer `trial.res`.

3. **XDXD-inspired research method** — `--method xdxd_structure` (multi-start CF→coords; not auto).

4. **GraPhAI external skeleton** — `GRAPHAI_HOME` discovery; no weight redistribution.

5. **Honesty** — Strict success unchanged; 30% seed bar not claimed without scoreboard YES.

## Install

```bash
python -m pip install -e ".[dev,gui]"
pytest -q
```
