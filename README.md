# Temporal Regularized Learning paper

Source for *Temporal Regularized Learning: Deep Local Soft-SFA from Temporal
Streams*.

This revision replaces every result affected by the rejected implementation.
It derives the corrected detached and undetached objectives, states the full
quadratic lateral-state cost, treats label-ordered MNIST as label-assisted, and
uses naturally ordered PAMAP2 streams for the self-supervised temporal-order
test. Confirmatory tables are generated only from the manifest-locked,
five-seed result records; no historical number is used as a fallback.

Build the PDF with:

```bash
tectonic main.tex
```

When present, `generated_results.tex` and
`generated_appendix_results.tex` are included automatically. The generation
command and provenance requirements are documented in the companion
[code repository](https://github.com/Axym-Labs/TeReL).

- [Project page](https://axym.org/work/terel)
- [Code](https://github.com/Axym-Labs/TeReL)
