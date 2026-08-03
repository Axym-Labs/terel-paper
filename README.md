# Temporal Regularized Learning paper

Source for *Temporal Regularized Learning: Competitive Deep Local Soft-SFA*.

This revision restores the intended TeReL training protocol and separates two
method roles: canonical greedy, layer-local TeReL and temporally local,
bounded-state TeReL-S. In the frozen five-seed confirmation, canonical TeReL
reaches 97.30% MNIST test accuracy versus 98.34% for compute-matched
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
`generated_appendix_results_v2.tex` are included automatically. The generation
command and provenance requirements are documented in the companion
[code repository](https://github.com/Axym-Labs/TeReL).

Regenerate the corrected performance figure from the frozen analysis JSON with:

```bash
python figures/corrected-performance-v2.py /path/to/confirmatory-analysis-v2.json
python figures/terel-revision-overview.py
```

The committed PDF and generated tables correspond to the frozen 25-run v2
confirmatory matrix and the separately frozen 12-run validation mechanism
audit. Raw seeds, paired intervals, configuration hashes, and resource
accounting are included in the paper and appendix.

- [Project page](https://axym.org/work/terel)
- [Code](https://github.com/Axym-Labs/TeReL)
