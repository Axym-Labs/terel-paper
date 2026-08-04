# Temporal Regularized Learning: A Neuron-Local Rule for Deep Slow-Feature Learning

Anonymous-review source for the TeReL manuscript.

The manuscript uses the self-contained `axym-publication.sty` template; no
venue style file is required for the anonymous build.

TeReL realizes Slow Feature Analysis as a soft objective for deep neuron-local
learning. Canonical TeReL retains short temporal gradients within a chunk;
TeReL-S detaches the temporal reference and permits bounded-state samplewise
execution. Under label-constructed MNIST order, canonical TeReL attains 97.30%
test accuracy across five seeds, compared with 98.34% for
data-presentation-matched backpropagation, 96.98% for Local SupCon, and 95.35%
for a BatchNorm-calibrated random encoder. Mechanism interventions separate the
roles of temporal coherence, variance expansion, and decorrelation. A
direct-covariance control measures the accuracy of the lagged lateral signal,
and PAMAP2 provides a secondary natural-order stress test.

Build the PDF with:

```bash
SOURCE_DATE_EPOCH=1785715200 tectonic main.tex
```

The fixed epoch removes build-time metadata variation. Generated scientific
inputs have descriptive names:

- `generated_primary_results.tex` and `generated_primary_appendix.tex`
- `generated_mechanism_results.tex`
- `generated_local_supcon_results.tex` and
  `generated_local_supcon_appendix.tex`
- `generated_normalization_control.tex`

Regenerate the figures with:

```bash
python figures/mnist-performance-comparison.py \
  /path/to/primary-analysis.json \
  /path/to/local-supcon-analysis.json \
  /path/to/normalization-control-analysis.json
python figures/terel-method-overview.py
```

The anonymous supplement contains the pinned environment, source records,
selection ledger, portable generation commands, and detailed provenance in
`ARTIFACT_README.md`.
