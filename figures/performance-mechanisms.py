"""Render downstream quality and the most informative matched effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

AXYM = "#3F21B6"
AXYM_LIGHT = "#8C7AD3"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
INK = "#111827"
MUTED = "#4B5563"
RULE = "#D8DCE2"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["P052", "Palatino", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _point_rows(ax, rows, *, xlabel, xlim, reference=None) -> None:
    y = np.arange(len(rows))[::-1]
    for position, (label, summary, color, marker) in zip(y, rows, strict=True):
        mean = 100.0 * summary["mean"]
        sd = 100.0 * summary["sample_sd"]
        ax.errorbar(
            mean,
            position,
            xerr=sd,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.1,
            capsize=2.0,
            zorder=3,
        )
    if reference is not None:
        ax.axvline(100.0 * reference, color=AXYM, linewidth=0.8, linestyle=":")
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=RULE, linewidth=0.55)
    ax.set_axisbelow(True)


def _panel_label(ax, label, title) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=INK,
        fontweight="bold",
        fontsize=9,
    )
    ax.text(
        0.5,
        1.04,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=INK,
        fontweight="bold",
    )


def _difference_rows(ax, rows, *, xlabel, xlim) -> None:
    y = np.arange(len(rows))[::-1]
    for position, (label, mean, low, high, color, marker) in zip(
        y, rows, strict=True
    ):
        ax.errorbar(
            100.0 * mean,
            position,
            xerr=np.array([[100.0 * (mean - low)], [100.0 * (high - mean)]]),
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.1,
            capsize=2.0,
            zorder=3,
        )
    ax.axvline(0.0, color=AXYM, linewidth=0.8, linestyle=":")
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=RULE, linewidth=0.55)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    _style()
    summary = json.loads(arguments.summary.read_text())
    mnist = summary["mnist"]
    methods = mnist["methods"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.05),
        gridspec_kw={"width_ratios": (1.15, 1.0), "wspace": 0.48},
    )

    primary_rows = (
        ("Supervised offline", methods["supervised-offline"]["accuracy"], ORANGE, "D"),
        ("TeReL-Offline", methods["terel-offline"]["accuracy"], AXYM_LIGHT, "s"),
        ("Layer-local SupCon", methods["layer-local-supcon"]["accuracy"], BLUE, "^"),
        ("TeReL", methods["terel"]["accuracy"], AXYM, "o"),
        ("Random encoder", methods["random"]["accuracy"], MUTED, "o"),
        ("Linear input", methods["supervised-linear"]["accuracy"], ORANGE, "x"),
        ("Batch SFA", methods["batch-sfa"]["accuracy"], GREEN, "s"),
        ("Incremental SFA", methods["incremental-sfa"]["accuracy"], GREEN, "^"),
    )
    _point_rows(
        axes[0],
        primary_rows,
        xlabel="MNIST accuracy (%)",
        xlim=(86.0, 100.0),
    )

    online = mnist["online_continuation"]["accuracy_difference"]
    shuffled = mnist["contrasts"]["terel-minus-shuffled-order"]
    no_temporal = mnist["contrasts"]["terel-minus-no-temporal"]
    mechanism_rows = (
        (
            "Continued updates",
            online["mean"],
            online["student_t_ci95_low"],
            online["student_t_ci95_high"],
            AXYM,
            "o",
        ),
        (
            "Shuffled order",
            -shuffled["mean_difference"],
            -shuffled["student_t_ci95_high"],
            -shuffled["student_t_ci95_low"],
            BLUE,
            "s",
        ),
        (
            "No temporal term",
            -no_temporal["mean_difference"],
            -no_temporal["student_t_ci95_high"],
            -no_temporal["student_t_ci95_low"],
            GREEN,
            "^",
        ),
    )
    _difference_rows(
        axes[1],
        mechanism_rows,
        xlabel="Accuracy change from TeReL (points)",
        xlim=(-2.25, 0.35),
    )

    titles = ["Representation quality", "Temporal signal and continuation"]
    for label, title, ax in zip("ab", titles, axes, strict=True):
        _panel_label(ax, label, title)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(arguments.output.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(
        arguments.output.with_suffix(".png"),
        bbox_inches="tight",
        transparent=True,
        dpi=320,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
