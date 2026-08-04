# Temporal Regularized Learning paper

Anonymous-review source for *Temporal Regularized Learning: Deep Local
Soft-SFA with Constructed Temporal Supervision*.

The paper presents two method roles: canonical greedy, layer-local TeReL and
temporally local, bounded-state TeReL-S. On label-ordered MNIST, canonical
TeReL reaches 97.30% test accuracy across five seeds, versus 98.34% for
data-presentation-matched backpropagation, 96.98% for matched Local SupCon, and
95.35% for a BatchNorm-calibrated random encoder. Mechanism controls isolate
the temporal, variance, and decorrelation contributions; the direct-covariance
audit quantifies the lagged lateral signal. PAMAP2 remains a secondary
natural-order stress test.

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

The anonymous supplement contains the pinned environment, exact source
records, selection ledger, portable generation commands, and
`ARTIFACT_README.md` with detailed provenance.
