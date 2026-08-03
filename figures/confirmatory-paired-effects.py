"""Render the paired-effects figure from a confirmatory-analysis JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PRIMARY = "#3F21B6"
CONTROL = "#BDBDBD"
PAIR = "#727272"


def _values(method: dict[str, object]) -> np.ndarray:
    by_seed = method["by_seed"]
    assert isinstance(by_seed, dict)
    return np.asarray([by_seed[key] for key in sorted(by_seed)], dtype=float)


def _panel(axis, left, right, labels, ylabel, panel, effect):
    for left_value, right_value in zip(left, right, strict=True):
        axis.plot(
            [0, 1], [left_value, right_value], color=PAIR, linewidth=0.75,
            alpha=0.72, zorder=1
        )
    axis.scatter(
        np.zeros_like(left), left, marker="o", s=24, color=PRIMARY,
        edgecolor="white", linewidth=0.45, zorder=3
    )
    axis.scatter(
        np.ones_like(right), right, marker="s", s=23, color=CONTROL,
        edgecolor="#555555", linewidth=0.45, zorder=3
    )
    for position, sample in enumerate((left, right)):
        mean = float(np.mean(sample))
        axis.plot(
            [position - 0.16, position + 0.16], [mean, mean],
            color="#111111", linewidth=1.4, zorder=4
        )
    axis.set_xlim(-0.38, 1.38)
    axis.set_xticks([0, 1], labels)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.55, zorder=0)
    axis.text(
        0.01, 0.98, panel, transform=axis.transAxes, va="top",
        fontweight="bold", fontsize=9
    )
    axis.text(
        0.99, 0.03, effect, transform=axis.transAxes, ha="right", va="bottom",
        fontsize=7, color="#333333"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    arguments = parser.parse_args(argv)
    analysis = json.loads(arguments.analysis.read_text())
    mnist = analysis["datasets"]["mnist"]["methods"]
    pamap2 = analysis["datasets"]["pamap2"]["methods"]

    plt.style.use(HERE / "paper.mplstyle")
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.48))
    _panel(
        axes[0], 100 * _values(mnist["terel-local"]),
        100 * _values(mnist["random"]), ("TeReL", "Random"),
        "Test accuracy (%)", "a", r"paired mean: $-0.42$ pp"
    )
    _panel(
        axes[1], _values(pamap2["terel-ordered"]),
        _values(pamap2["terel-shuffled"]), ("Chronological", "Shuffled"),
        "Test macro-F1", "b", r"paired mean: $-0.0047$"
    )
    axes[0].set_ylim(87.5, 90.0)
    axes[0].set_yticks([87.5, 88.0, 88.5, 89.0, 89.5, 90.0])
    axes[1].set_ylim(0.22, 0.43)
    axes[1].set_yticks([0.25, 0.30, 0.35, 0.40])
    figure.subplots_adjust(
        left=0.085, right=0.99, top=0.965, bottom=0.18, wspace=0.31
    )
    output = HERE / "confirmatory-paired-effects"
    figure.savefig(
        output.with_suffix(".pdf"), transparent=True,
        metadata={"CreationDate": None, "ModDate": None}
    )
    figure.savefig(output.with_suffix(".png"), dpi=300, transparent=True)
    plt.close(figure)


if __name__ == "__main__":
    main()
