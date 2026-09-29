# Temporal Regularized Learning paper

This repository contains the TeReL manuscript and its figures. The paper
constructs a causal surrogate of VICReg's invariance--variance--covariance
decomposition: temporal adjacency makes invariance a slow-feature objective,
detached running moments replace batch statistics, and a lagged same-layer
operator supplies the covariance signal. A detached target defines a signed
postsynaptic neuron state, whose outer product with presynaptic activity gives
the feedforward update. The same-layer matrix also supplies one explicit state
correction and receives an anti-Hebbian state--state contribution.

The reported class-chunked MNIST encoder reaches 96.39 ± 0.19% held-out
linear-probe accuracy with plain SGD and batch size one. TeReL-Offline, the
spatially and temporally relaxed reference, reaches 98.39 ± 0.04%. Continued
unlabeled updates with a validation-selected step preserve the representation
at 96.43 ± 0.15% using the same fitted probe.

Build the paper twice to resolve references:

```bash
SOURCE_DATE_EPOCH=1786317315 FORCE_SOURCE_DATE=1 tectonic main.tex
SOURCE_DATE_EPOCH=1786317315 FORCE_SOURCE_DATE=1 tectonic main.tex
```

Regenerate the native-LaTeX method figure with:

```bash
SOURCE_DATE_EPOCH=1786317315 FORCE_SOURCE_DATE=1 \
  tectonic figures/terel-method-overview.tex --outdir figures
```

The remaining figures have self-contained scripts in `figures/` and read only
the final analysis JSON or the archived MNIST visualization data:

```bash
python figures/performance-mechanisms.py \
  --summary /path/to/strengthening2-final-analysis-mnist.json \
  --output figures/performance-mechanisms
python figures/representation-geometry.py \
  --mnist figures/mnist-visual-evidence.npz \
  --output figures/representation-geometry
python figures/neuron-state-behavior.py \
  --mnist figures/mnist-visual-evidence.npz \
  --output figures/neuron-state-behavior
```

The manuscript contains only the final method, exact evaluation protocol, raw
reported values, and scientifically relevant controls. Execution commits and
checksums are kept in the code artifact's `ARTIFACT_README.md`.
