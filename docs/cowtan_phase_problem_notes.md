# Notes on Cowtan (2001): Phase Problem in X-ray Crystallography

**Source:** Kevin Cowtan, University of York — *Phase Problem in X-ray Crystallography, and Its Solution*, Encyclopedia of Life Sciences, Macmillan / Nature Publishing Group, 2001.  
**User PDF:** `phaseproblem.pdf` (reviewed in full; not redistributed here for copyright).

## Article map

1. **Phase problem** — Only amplitudes measured; phases required for \(\rho\) via Fourier transform (Bragg 1915). Diffraction spots = structure factors on reciprocal lattice; Friedel mates \(|F(\mathbf{h})|=|F(-\mathbf{h})|\) without anomalous scattering.
2. **Patterson methods** — Map from \(|F|^2\), phases zero → interatomic vectors. Effective for small molecules / heavy atoms; \(N(N-1)\) vectors → unusable beyond ~20–50 light atoms (Patterson 1934).
3. **Direct methods** — High-angle data + positivity/atomicity → phase relations. **Three-phase invariant** for strong reflections with \(\mathbf{h}+\mathbf{k}+(-\mathbf{h}-\mathbf{k})=0\): phases sum near 0 (Cochran 1952). Multi-solution random starts; up to ~2000 atoms historically; proteins usually lack atomic resolution.
4. **MIR** — Known heavy-atom change; locate HA by Patterson/direct methods; compare \(|F_P|\) vs \(|F_{PH}|\). Constructive → \(\varphi_P\approx\varphi_H\); destructive → \(\varphi_P\approx\varphi_H+\pi\); intermediate → twofold ambiguity → need multiple derivatives (Green, Ingram, Perutz 1954).
5. **MAD** — Tune wavelength near absorption edge; vary \(f',f''\) of anomalous atoms (often SeMet). Bijvoet pairs unequal; single crystal can phase in theory; high precision required (Hendrickson & Ogata 1997).
6. **Molecular replacement** — Rotate/translate known model; match \(|F_{\mathrm{calc}}|\) to \(|F_{\mathrm{obs}}|\); use model phases (Rossmann 1972).
7. **Phase improvement** — Solvent flattening (Wang), NCS averaging, iterative density modification.

## Implementation mapping in grok_phase_solver

| Cowtan topic | Module |
|--------------|--------|
| Patterson | `physics/patterson.py`, `solvers/patterson.py` |
| Direct methods / triplets | `solvers/direct_methods.py` |
| MIR / MAD / MR simulation | `data/experimental_phasing.py` |
| Charge flipping / HIO | `solvers/charge_flipping.py`, `hio.py` |
| Phase improvement (partial) | positivity / solvent hooks in HIO; Phase 3 envelope |
| Hybrid AI tests | `docs/hybrid_ai_tests.md` |

## Citation

```bibtex
@incollection{Cowtan2001Phase,
  author    = {Cowtan, Kevin},
  title     = {Phase Problem in X-ray Crystallography, and Its Solution},
  booktitle = {Encyclopedia of Life Sciences},
  publisher = {Macmillan Publishers Ltd / Nature Publishing Group},
  year      = {2001}
}
```
