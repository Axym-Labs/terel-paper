# Temporal Regularized Learning paper

The manuscript centers TeReL, a samplewise slow-feature rule local in space and
time. A regularized activation target defines a preactivation neuron state.
The feedforward gradient is the outer product of that postsynaptic state and
presynaptic activity; pairs of the same states drive a learned anti-Hebbian
lateral rule. TeReL-Offline is the less constrained minibatch reference.

The canonical MNIST result uses plain SGD, batch size one, two data
presentations, and one lateral matrix-vector pass. It reaches 95.84 ± 0.07%
accuracy. The lateral pass improves its matched no-inhibition reference by
1.42 points on validation, with 95% Student-t interval [1.32, 1.52].
TeReL-Offline reaches 97.30 ± 0.07%; backpropagation and Local SupCon reach
98.34 ± 0.08% and 96.98 ± 0.10% under the corresponding batched protocol.
Development controls on three sensor streams do not justify extending this
controlled result to natural temporal order.

Build the paper with:

```bash
SOURCE_DATE_EPOCH=1786317315 FORCE_SOURCE_DATE=1 tectonic main.tex
```

Regenerate the neuron-state figure with:

```bash
python figures/neuron-state-dynamics.py figures/neuron-state-dynamics-data.npz
```

The exact sensor-development metrics and shared configuration reported in the
appendix are recorded in `figures/natural-order-development.json`.

The source uses the Axym publication template. Figures are generated from the
frozen result records; their scripts are in `figures/`. Detailed configuration,
raw-run, and mechanism tables are placed in the appendix so that the main text
keeps the scientific argument visible.

The manuscript reports the final method and evaluation protocol. Immutable
execution identifiers, checksums, and portable analysis commands belong in the
artifact README rather than in the paper.
