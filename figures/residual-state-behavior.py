"""Render the matched inhibition effect and the resulting neuron-state change."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PRIMARY = "#3F21B6"
SECONDARY = "#8C7AD3"
NEUTRAL = "#77777F"
LIGHT = "#D1D1D7"


def _records(path: Path) -> list[dict]:
    records = [json.loads(item.read_text()) for item in sorted(path.glob("seed-*.json"))]
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def render(inhibited_path: Path, reference_path: Path, final_path: Path, output_stem: Path) -> None:
    inhibited = _records(inhibited_path)
    reference = _records(reference_path)
    final = _records(final_path)
    if [r["seed"] for r in inhibited] != [r["seed"] for r in reference]:
        raise ValueError("The validation comparison must use matched seeds")

    plt.style.use(Path(__file__).with_name("paper.mplstyle"))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45), gridspec_kw={"wspace": 0.38})

    # Matched validation effect: the scientific reason for the lateral pass.
    ax = axes[0]
    without = 100 * np.asarray([r["metrics"]["accuracy"] for r in reference])
    with_inhibition = 100 * np.asarray([r["metrics"]["accuracy"] for r in inhibited])
    for left, right in zip(without, with_inhibition, strict=True):
        ax.plot([0, 1], [left, right], color=SECONDARY, linewidth=1.15, alpha=0.68)
    ax.scatter(np.zeros_like(without), without, color=NEUTRAL, s=23, zorder=3)
    ax.scatter(np.ones_like(with_inhibition), with_inhibition, color=PRIMARY, s=25, zorder=3)
    ax.plot([-0.12, 0.12], [without.mean()] * 2, color=NEUTRAL, linewidth=2.3)
    ax.plot([0.88, 1.12], [with_inhibition.mean()] * 2, color=PRIMARY, linewidth=2.3)
    ax.text(0.5, 95.35, f"+{(with_inhibition - without).mean():.2f} points",
            ha="center", color=PRIMARY, fontsize=8.0, weight="bold")
    ax.set_xticks([0, 1], ["No inhibition", "One lateral pass"])
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(94.1, 96.25)
    ax.grid(axis="y", color=LIGHT, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.13, 1.03, "a", transform=ax.transAxes, fontweight="bold")

    # Population state before and after the single pass in final runs.
    ax = axes[1]
    positions = ((0.0, 0.75), (1.75, 2.5))
    for layer, (left_x, right_x) in enumerate(positions):
        before = np.asarray([
            r["encoder_training"]["base_residual_state_rms_mean"][layer] for r in final
        ])
        after = np.asarray([
            r["encoder_training"]["residual_state_rms_mean"][layer] for r in final
        ])
        for left, right in zip(before, after, strict=True):
            ax.plot([left_x, right_x], [left, right], color=SECONDARY,
                    linewidth=0.9, alpha=0.48)
        ax.scatter(np.full_like(before, left_x), before, color=NEUTRAL, s=17, zorder=3)
        ax.scatter(np.full_like(after, right_x), after, color=PRIMARY, s=19, zorder=3)
        ax.text((left_x + right_x) / 2, max(before.max(), after.max()) + 0.016,
                f"{before.mean():.3f} → {after.mean():.3f}", ha="center",
                color=PRIMARY, fontsize=7.3, weight="bold")
    ax.set_xticks([0, 0.75, 1.75, 2.5], ["before", "after", "before", "after"])
    ax.text(0.375, -0.20, "layer 1", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=7.1, weight="bold")
    ax.text(2.125, -0.20, "layer 2", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=7.1, weight="bold")
    ax.set_ylabel("Neuron-state RMS")
    ax.set_ylim(0.45, 0.71)
    ax.grid(axis="y", color=LIGHT, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.13, 1.03, "b", transform=ax.transAxes, fontweight="bold")

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight",
                transparent=True, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output_stem.with_suffix(".png"), dpi=300,
                bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inhibited", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("final", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("residual-state-behavior"))
    args = parser.parse_args()
    render(args.inhibited, args.reference, args.final, args.output)
