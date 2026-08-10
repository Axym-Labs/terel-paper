"""Render traces and distributions for four predeclared neuron states."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#3F21B6", "#7656CE", "#B34782", "#D06A43")
GRID = "#D1D1D7"


def render(data_path: Path, output_stem: Path) -> None:
    values = np.load(data_path)
    states = np.asarray(values["trace_values"], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError("trace_values must have shape [time, 4]")
    if not np.isfinite(states).all():
        raise ValueError("state trace contains non-finite values")

    plt.style.use(Path(__file__).with_name("paper.mplstyle"))
    figure = plt.figure(figsize=(7.05, 2.65))
    outer = figure.add_gridspec(1, 2, width_ratios=(1.9, 1.0), wspace=0.30)
    traces = outer[0].subgridspec(4, 1, hspace=0.08)
    time = np.arange(len(states))
    limit = 1.06 * np.abs(states).max()
    labels = ("A", "B", "C", "D")
    for index, (label, color) in enumerate(zip(labels, COLORS, strict=True)):
        axis = figure.add_subplot(traces[index])
        axis.axhline(0.0, color=GRID, linewidth=0.55, zorder=0)
        axis.plot(time, states[:, index], color=color, linewidth=1.05)
        axis.set_ylim(-limit, limit)
        axis.set_yticks([0.0])
        axis.set_ylabel(label, rotation=0, labelpad=8, weight="bold", color=color)
        axis.spines[["top", "right", "left"]].set_visible(False)
        if index != 3:
            axis.set_xticks([])
            axis.spines["bottom"].set_visible(False)
        else:
            axis.set_xlabel("Consecutive observation")
    figure.text(0.018, 0.52, "Neuron state", rotation=90, va="center")
    figure.text(0.045, 0.94, "a", weight="bold")

    distribution = figure.add_subplot(outer[1])
    positions = np.arange(4, 0, -1)
    violins = distribution.violinplot(
        [states[:, index] for index in range(4)],
        positions=positions,
        orientation="horizontal",
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=100,
    )
    for body, color in zip(violins["bodies"], COLORS, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.54)
    for position, state, color in zip(positions, states.T, COLORS, strict=True):
        low, median, high = np.quantile(state, (0.25, 0.5, 0.75))
        distribution.plot([low, high], [position, position], color=color, linewidth=2.2)
        distribution.scatter(median, position, color=color, s=13, zorder=3)
    distribution.axvline(0.0, color=GRID, linewidth=0.7, zorder=0)
    distribution.set_yticks(positions, labels, weight="bold")
    distribution.set_xlabel("Neuron state")
    distribution.spines[["top", "right", "left"]].set_visible(False)
    figure.text(0.69, 0.94, "b", weight="bold")

    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("neuron-state-dynamics"),
    )
    arguments = parser.parse_args()
    render(arguments.data, arguments.output)


if __name__ == "__main__":
    main()
