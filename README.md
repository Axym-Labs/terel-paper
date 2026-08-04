# Temporal Regularized Learning paper

Anonymous-review source for *Temporal Regularized Learning: Deep Local
Soft-SFA with Constructed Temporal Supervision*.

This revision restores the intended TeReL training protocol and separates two
method roles: canonical greedy, layer-local TeReL and temporally local,
bounded-state TeReL-S. In the frozen five-seed confirmation, canonical TeReL
reaches 97.30% MNIST test accuracy versus 98.34% for data-presentation-matched
backpropagation and 95.13% for matched random features. A tuned samplewise
TeReL-S run reaches 95.14% validation accuracy with fixed state and no retained
temporal graph. PAMAP2 remains a secondary natural-order stress test.
A frozen one-factor validation audit additionally shows that temporal
coherence supplies the aligned signal, variance expansion prevents scale
collapse, and decorrelation prevents redundancy collapse.

Build the PDF with:

```bash
SOURCE_DATE_EPOCH=1785715200 tectonic main.tex
```

The fixed epoch makes the committed PDF byte-reproducible by removing build-time
metadata variation.

When present, `generated_results_v2.tex`,
`generated_mechanism_results_v2.tex`, and
`generated_appendix_results_v2.tex` are included automatically, together with
the two `generated_review_patch_*_v3.tex` files. The anonymous supplement
contains their generation commands, frozen records, and provenance.

Regenerate the corrected performance figure from the frozen analysis JSON with:

```bash
python figures/corrected-performance-v2.py \
  /path/to/confirmatory-analysis-v2.json \
  /path/to/review-patch-confirmatory-analysis-v3.json
python figures/terel-revision-overview.py
```

The committed PDF and generated tables correspond to the frozen 25-run v2
confirmatory matrix, the separately frozen 12-run validation mechanism audit,
and the bounded review patch described in the paper. Raw seeds, paired
intervals, configuration hashes, and resource accounting are included in the
paper, appendix, and anonymized supplement.
