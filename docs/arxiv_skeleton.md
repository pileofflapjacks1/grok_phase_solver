# arXiv preprint skeleton

**Title:** Toward an Open, Physics-Informed Framework for the Crystallographic Phase Problem

**Authors:** [TBD]

**Abstract (draft):**  
We present *grok_phase_solver*, an open-source Python framework that unifies classical solutions of the X-ray crystallography phase problem—Patterson methods, direct methods (triplet invariants), charge flipping, hybrid input–output, isomorphous replacement, and density modification—with modular interfaces for machine-learning phase prediction. All algorithms are derived from the Fourier relationship between electron density and structure factors and are evaluated with origin-invariant map correlation and phase-error metrics. We document successes on small synthetic and COD structures at atomic resolution, systematic degradation at low resolution, and the limits of current neural baselines trained on synthetic data. We do not claim a general solution for macromolecular phasing; rather, we provide reproducible baselines and hybrid testbeds for future learning methods conditioned on multi-channel experimental data (MIR/MAD/MR).

## 1. Introduction
- Phase problem statement; Bragg, Patterson, Cochran, Cowtan review
- Direct methods vs experimental phasing vs AI (PhAI 2024)
- Contributions: open library, correct metrics, hybrid benchmarks

## 2. Mathematical background
- Structure factors, Parseval, Friedel
- Patterson autocorrelation
- Triplet invariants and Cochran distribution
- Blow–Crick lack-of-closure
- Non-convex modulus constraint; trivial associates

## 3. Methods
### 3.1 Classical solvers (with equations)
### 3.2 Density modification
### 3.3 Synthetic data generation
### 3.4 Neural baseline (architecture, loss \(1-\cos\Delta\varphi\))
### 3.5 Hybrid loops (seed + polish)

## 4. Experiments
- Synthetic P1 atomic resolution
- COD 2100301 resolution series
- MIR simulation FOM vs map CC
- Domain-gap Wilson plots
- Failure cases

## 5. Discussion
- What is solved / not solved
- Relation to PhAI and SHELX
- Roadmap

## 6. Conclusions

## References
- Bragg 1915; Patterson 1934; Cochran 1952; Blow & Crick 1959  
- Cowtan ELS 2001; Wang 1985; Oszlányi & Sütő; Fienup HIO  
- Larsen et al. Science 2024 (PhAI); Rossmann MR; Hendrickson MAD  

## Supplementary
- Notebooks with proofs; benchmark JSON; code DOI
