"""Render the reader-first MNIST performance comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


PRIMARY = "#3F21B6"
SECONDARY = "#8C7AD3"
NEUTRAL = "#747474"
LIGHT = "#C8C8CE"


def _t_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean - half, mean + half


def _final_values(output_root: Path) -> np.ndarray:
    paths = sorted((output_root / "mnist" / "terel-s-residual").glob("seed-*.json"))
    values = [json.loads(path.read_text())["metrics"]["accuracy"] for path in paths]
    if len(values) != 5:
        raise ValueError(f"Expected five frozen final records, found {len(values)}")
    return np.asarray(values, dtype=float)


def render(
    residual_output: Path,
    validation_ledger_path: Path,
    primary_analysis_path: Path,
    comparator_analysis_path: Path,
    normalization_analysis_path: Path,
    output_stem: Path,
) -> None:
    residual = _final_values(residual_output)
    validation = json.loads(validation_ledger_path.read_text())
    primary = json.loads(primary_analysis_path.read_text())
    comparator = json.loads(comparator_analysis_path.read_text())
    normalization = json.loads(normalization_analysis_path.read_text())
    plt.style.use(Path(__file__).with_name("paper.mplstyle"))

    methods = [
        ("TeReL-S", residual, PRIMARY),
        ("Random\n+BN", np.asarray(normalization["random_bn_calibrated"]["raw"]), NEUTRAL),
        ("Local\nSupCon", np.asarray(comparator["local_supcon"]["raw"]), NEUTRAL),
        ("TeReL-\nbatched", np.asarray(primary["methods"]["terel-all"]["raw"]), SECONDARY),
        ("BP", np.asarray(primary["methods"]["bp-all"]["raw"]), NEUTRAL),
    ]

    residual_difference = 100 * np.asarray(
        validation["paired_accuracy"]["residual_minus_reference"]["values"], dtype=float
    )
    contrasts = [
        ("Residual state $-$\nmatched reference", residual_difference, PRIMARY),
        (
            "TeReL-batched $-$\nRandom+BN",
            100 * np.asarray(normalization["terel_minus_random_bn"]["raw_differences"]),
            SECONDARY,
        ),
        (
            "TeReL-batched $-$\nLocal SupCon",
            100 * np.asarray(comparator["terel_minus_local_supcon"]["raw_differences"]),
            SECONDARY,
        ),
        (
            "TeReL-batched $-$ BP",
            100 * np.asarray(primary["contrasts"]["terel-minus-bp"]["raw_differences"]),
            SECONDARY,
        ),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.05, 2.8),
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.52},
    )

    ax = axes[0]
    for x, (label, fractions, color) in enumerate(methods):
        values = 100 * fractions
        offsets = np.linspace(-0.09, 0.09, len(values))
        ax.scatter(x + offsets, values, s=18, color=color, alpha=0.76,
                   edgecolor="white", linewidth=0.35, zorder=3)
        mean = float(values.mean())
        ax.plot([x - 0.19, x + 0.19], [mean, mean], color=color, lw=2.0, zorder=4)
        ax.text(x, float(values.max()) + 0.07, f"{mean:.2f}", ha="center", va="bottom",
                fontsize=7.0, color=color, weight="bold")
    ax.set_xticks(range(len(methods)), [item[0] for item in methods])
    ax.tick_params(axis="x", labelsize=7.1)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(94.8, 98.65)
    ax.text(-0.14, 1.03, "a", transform=ax.transAxes, fontweight="bold")
    ax.text(0.0, 1.03, "Final performance", transform=ax.transAxes,
            fontsize=8.2, weight="bold")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.axvline(0, color=LIGHT, lw=0.9, zorder=1)
    for y, (label, raw, color) in enumerate(contrasts):
        mean, low, high = _t_interval(np.asarray(raw, dtype=float))
        ax.scatter(raw, y + np.linspace(-0.075, 0.075, len(raw)), s=14,
                   color=color, alpha=0.62, zorder=3)
        ax.plot([low, high], [y, y], color=color, lw=2.1,
                solid_capstyle="round", zorder=4)
        ax.scatter([mean], [y], s=30, color="white", edgecolor=color,
                   linewidth=1.4, zorder=5)
        ax.text(high + 0.10, y, f"{mean:+.2f}", ha="left", va="center",
                fontsize=6.9, color=color, weight="bold")
    ax.set_yticks(range(len(contrasts)), [item[0] for item in contrasts])
    ax.invert_yaxis()
    ax.set_xlabel("Paired accuracy difference (points)")
    ax.set_xlim(-1.35, 3.75)
    ax.text(-0.14, 1.03, "b", transform=ax.transAxes, fontweight="bold")
    ax.text(0.0, 1.03, "Matched effects", transform=ax.transAxes,
            fontsize=8.2, weight="bold")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=7.0)

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight",
                transparent=True, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output_stem.with_suffix(".png"), dpi=300,
                bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("residual_output", type=Path)
    parser.add_argument("validation_ledger", type=Path)
    parser.add_argument("primary_analysis", type=Path)
    parser.add_argument("comparator_analysis", type=Path)
    parser.add_argument("normalization_analysis", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("mnist-performance-comparison"),
    )
    args = parser.parse_args()
    render(
        args.residual_output,
        args.validation_ledger,
        args.primary_analysis,
        args.comparator_analysis,
        args.normalization_analysis,
        args.output,
    )
