"""Render the corrected TeReL figure from confirmatory-analysis-v2.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PRIMARY = "#3F21B6"
SECONDARY = "#8C7AD3"
NEUTRAL = "#747474"
DARK = "#252525"


def render(analysis_path: Path, output_stem: Path) -> None:
    data = json.loads(analysis_path.read_text())
    plt.style.use(Path(__file__).with_name("paper.mplstyle"))
    methods = [
        ("random-all", "Random", NEUTRAL),
        ("terel-s-all", "TeReL-S", SECONDARY),
        ("terel-last", "TeReL\nlast", SECONDARY),
        ("terel-all", "TeReL\nall", PRIMARY),
        ("bp-all", "BP", DARK),
    ]
    contrasts = [
        ("terel-minus-random", r"TeReL $-$ random", PRIMARY),
        ("terel-s-minus-random", r"TeReL-S $-$ random", SECONDARY),
        ("terel-minus-bp", r"TeReL $-$ BP", PRIMARY),
        ("terel-last-minus-all", r"last $-$ all", SECONDARY),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(6.85, 2.65),
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.42},
    )

    ax = axes[0]
    for x, (key, label, color) in enumerate(methods):
        values = 100 * np.asarray(data["methods"][key]["raw"], dtype=float)
        ax.scatter(
            x + np.linspace(-0.10, 0.10, len(values)), values, s=17,
            color=color, alpha=0.82, edgecolor="white", linewidth=0.35, zorder=3,
        )
        mean = float(values.mean())
        ax.plot([x - 0.20, x + 0.20], [mean, mean], color=color, lw=2.0, zorder=4)
        ax.text(x, mean + 0.16, f"{mean:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(methods)), [item[1] for item in methods])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(94.1, 98.85)
    ax.set_title("a  Corrected five-seed performance", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.axvline(0, color="#B4B4B4", lw=0.9, zorder=1)
    for y, (key, label, color) in enumerate(contrasts):
        record = data["contrasts"][key]
        raw = 100 * np.asarray(record["raw_differences"], dtype=float)
        mean = 100 * float(record["mean_difference"])
        low = 100 * float(record["ci95_low"])
        high = 100 * float(record["ci95_high"])
        ax.scatter(raw, y + np.linspace(-0.09, 0.09, len(raw)), s=14,
                   color=color, alpha=0.68, zorder=3)
        ax.plot([low, high], [y, y], color=color, lw=2.2,
                solid_capstyle="round", zorder=4)
        ax.scatter([mean], [y], s=31, color="white", edgecolor=color,
                   linewidth=1.4, zorder=5)
    ax.set_yticks(range(len(contrasts)), [item[1] for item in contrasts])
    ax.invert_yaxis()
    ax.set_xlabel("Paired accuracy difference (points)")
    ax.set_xlim(-1.45, 2.95)
    ax.set_title("b  Matched-seed effects", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=300,
                bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("corrected-performance-v2"))
    args = parser.parse_args()
    render(args.analysis, args.output)
