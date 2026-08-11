"""Render exact TeReL neuron-state trajectories and empirical distributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

AXYM = "#3F21B6"
COLORS = (AXYM, "#0072B2", "#D55E00", "#009E73")
INK = "#111827"
MUTED = "#4B5563"
RULE = "#CBD0D8"


def _load(path: Path):
    values = np.load(path)
    return values["state_values"], values["neuron_indices"]


def _style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["P052", "Palatino", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _trace_panel(ax, values, indices, window):
    shown = values[:window]
    steps = np.arange(shown.shape[0])
    for column, (index, color) in enumerate(zip(indices, COLORS, strict=True)):
        ax.plot(
            steps,
            shown[:, column],
            color=color,
            linewidth=0.85,
            alpha=0.92,
            label=f"Neuron {int(index)}",
        )
    ax.axhline(0.0, color=RULE, linewidth=0.7, zorder=0)
    ax.set_xlim(0, shown.shape[0] - 1)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Neuron state")
    ax.margins(y=0.08)
    ax.legend(
        loc="upper right",
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.8,
        borderaxespad=0.2,
    )


def _distribution_panel(ax, values, indices):
    positions = np.arange(len(indices))
    violin = ax.violinplot(
        [values[:, column] for column in range(values.shape[1])],
        positions=positions,
        orientation="horizontal",
        widths=0.72,
        showextrema=False,
        showmedians=True,
        bw_method="scott",
    )
    for body, color in zip(violin["bodies"], COLORS, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.33)
        body.set_linewidth(0.75)
    violin["cmedians"].set_color(INK)
    violin["cmedians"].set_linewidth(0.8)
    ax.axvline(0.0, color=RULE, linewidth=0.7, zorder=0)
    ax.set_yticks(positions, [f"Neuron {int(index)}" for index in indices])
    ax.set_xlabel("Neuron state")
    ax.tick_params(axis="y", length=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=512)
    args = parser.parse_args()

    _style()
    values, indices = _load(args.mnist)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.15),
        gridspec_kw={"width_ratios": (1.65, 1.0), "wspace": 0.31},
    )
    _trace_panel(axes[0], values, indices, args.window)
    _distribution_panel(axes[1], values, indices)

    for label, ax in zip("ab", axes, strict=True):
        ax.text(
            -0.12,
            1.03,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=INK,
            fontweight="bold",
            fontsize=9,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(
        args.output.with_suffix(".png"),
        bbox_inches="tight",
        transparent=True,
        dpi=320,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
