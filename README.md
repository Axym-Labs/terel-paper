# Temporal Regularized Learning: Deep Slow-Feature Learning Local in Space and Time

This repository contains the anonymous manuscript in the project's
self-contained `axym-publication.sty` template.

The paper centers TeReL-S, a samplewise rule derived from a regularized
soft-SFA target. Each neuron tracks its preactivation residual to that target.
Without lateral coupling, its outer product with presynaptic activity is the
exact feedforward gradient. Learned inhibition settles the state used for the
feedforward update and learns lateral connections from pairs of the same neuron
states. The method retains fixed detached state, sends no error across learned
layers, and stores two dense same-layer matrices.

Under label-ordered MNIST supervision, TeReL-S reaches 95.50 ± 0.19% final
accuracy after two data presentations. Residual-state inhibition improves its
matched samplewise validation reference by 3.01 points with interval
[2.71, 3.31]. TeReL-batched reaches 97.30 ± 0.07%; backpropagation and Local
SupCon reach 98.34 ± 0.08% and 96.98 ± 0.10% under its batched reference
protocol. Objective ablations and a direct-covariance control retain their
roles as mechanism evidence. PAMAP2 remains an inconclusive natural-order
stress test.

Build the PDF with:

```bash
SOURCE_DATE_EPOCH=1785715200 tectonic main.tex
```

Generated scientific inputs are kept separate from the prose:

- `generated_residual_results.tex` and `generated_residual_appendix.tex`
- `generated_primary_results.tex` and `generated_primary_appendix.tex`
- `generated_mechanism_results.tex`
- `generated_local_supcon_results.tex` and
  `generated_local_supcon_appendix.tex`
- `generated_normalization_control.tex`

Regenerate the figures with the frozen analysis artifacts:

```bash
python figures/terel-method-overview.py

python figures/mnist-performance-comparison.py \
  /path/to/residual-confirmatory \
  /path/to/residual-validation-ledger.json \
  /path/to/batched-primary-analysis.json \
  /path/to/local-supcon-analysis.json \
  /path/to/normalization-control-analysis.json

python figures/residual-state-behavior.py \
  /path/to/residual-confirmatory \
  /path/to/residual-state-diagnostic.npz
```

The anonymous supplement contains the pinned environment, source records,
frozen plans, selection ledger, raw results, generation commands, and detailed
provenance in `ARTIFACT_README.md`.
