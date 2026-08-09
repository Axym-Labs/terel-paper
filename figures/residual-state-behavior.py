"""Render residual-state dynamics and matched spectral geometry."""

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
DIGIT_COLORS = plt.get_cmap("tab10").colors


def _final_records(root: Path) -> list[dict]:
    paths = sorted((root / "mnist" / "terel-s-residual").glob("seed-*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    if len(records) != 5:
        raise ValueError(f"Expected five final records, found {len(records)}")
    return records


def _spectral_panel(ax, coordinates, labels, *, title, purity):
    labels = labels.astype(int)
    for digit in range(10):
        mask = labels == digit
        ax.scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=3.5,
            color=DIGIT_COLORS[digit], alpha=0.34, linewidth=0, rasterized=True,
        )
        center = np.median(coordinates[mask], axis=0)
        ax.text(center[0], center[1], str(digit), ha="center", va="center",
                fontsize=7.2, weight="bold", color=DIGIT_COLORS[digit],
                bbox={"boxstyle": "circle,pad=0.10", "facecolor": "white",
                      "edgecolor": "none", "alpha": 0.78})
    ax.set_title(f"{title}\nneighbor purity {purity:.2f}", fontsize=7.6, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render(final_root: Path, diagnostic_path: Path, output_stem: Path) -> None:
    records = _final_records(final_root)
    diagnostic = np.load(diagnostic_path)
    plt.style.use(Path(__file__).with_name("paper.mplstyle"))

    fig = plt.figure(figsize=(7.05, 4.45))
    grid = fig.add_gridspec(2, 4, height_ratios=(0.88, 1.12), hspace=0.48, wspace=0.55)
    trace_ax = fig.add_subplot(grid[0, :2])
    rms_ax = fig.add_subplot(grid[0, 2:])
    reference_ax = fig.add_subplot(grid[1, :2])
    residual_ax = fig.add_subplot(grid[1, 2:])

    # A fixed, transparently selected neuron: largest base-state variance.
    time = diagnostic["trace_time"]
    base = diagnostic["trace_base"]
    settled = diagnostic["trace_settled"]
    trace_ax.axhline(0, color=LIGHT, linewidth=0.8)
    trace_ax.plot(time, base, color=NEUTRAL, linewidth=1.25, linestyle="--",
                  marker="o", markersize=2.6, label="before inhibition")
    trace_ax.plot(time, settled, color=PRIMARY, linewidth=1.7,
                  marker="o", markersize=2.8, label="settled state")
    trace_ax.set_xlabel("Position in a class-homogeneous chunk")
    trace_ax.set_ylabel("Neuron state")
    trace_ax.set_xticks(time)
    trace_ax.tick_params(axis="x", labelsize=6.4)
    trace_ax.legend(frameon=False, loc="upper right", handlelength=2.2)
    trace_ax.text(-0.12, 1.07, "a", transform=trace_ax.transAxes, fontweight="bold")
    trace_ax.set_title(
        f"Layer {int(diagnostic['trace_layer']) + 1}, neuron "
        f"{int(diagnostic['trace_neuron'])}",
        loc="left", fontsize=7.7, pad=2,
    )

    # Population RMS before/after settling; each line is one final run.
    positions = [(0, 1), (2.4, 3.4)]
    for layer, (x0, x1) in enumerate(positions):
        before = np.asarray([r["encoder_training"]["base_residual_state_rms_mean"][layer]
                             for r in records])
        after = np.asarray([r["encoder_training"]["residual_state_rms_mean"][layer]
                            for r in records])
        for left, right in zip(before, after, strict=True):
            rms_ax.plot([x0, x1], [left, right], color=SECONDARY, alpha=0.42,
                        linewidth=0.9, zorder=1)
        rms_ax.scatter(np.full_like(before, x0), before, color=NEUTRAL, s=15, zorder=2)
        rms_ax.scatter(np.full_like(after, x1), after, color=PRIMARY, s=17, zorder=2)
        rms_ax.plot([x0 - 0.15, x0 + 0.15], [before.mean()] * 2,
                    color=NEUTRAL, linewidth=2.0)
        rms_ax.plot([x1 - 0.15, x1 + 0.15], [after.mean()] * 2,
                    color=PRIMARY, linewidth=2.0)
        rms_ax.text((x0 + x1) / 2, max(before.max(), after.max()) + 0.025,
                    f"{before.mean():.3f} $\\rightarrow$ {after.mean():.3f}",
                    ha="center", fontsize=7.0, weight="bold", color=PRIMARY)
    rms_ax.set_xticks([0, 1, 2.4, 3.4],
                      ["before", "settled", "before", "settled"])
    rms_ax.text(0.5, -0.23, "layer 1", transform=rms_ax.get_xaxis_transform(),
                ha="center", fontsize=7.1, weight="bold")
    rms_ax.text(2.9, -0.23, "layer 2", transform=rms_ax.get_xaxis_transform(),
                ha="center", fontsize=7.1, weight="bold")
    rms_ax.set_ylabel("Population state RMS")
    rms_ax.set_ylim(0.28, 0.62)
    rms_ax.text(-0.12, 1.07, "b", transform=rms_ax.transAxes, fontweight="bold")

    labels = diagnostic["spectral_labels"]
    _spectral_panel(
        reference_ax,
        diagnostic["spectral_reference"],
        labels,
        title="Matched samplewise reference",
        purity=float(diagnostic["purity_reference"]),
    )
    reference_ax.text(-0.06, 1.03, "c", transform=reference_ax.transAxes,
                      fontweight="bold")
    _spectral_panel(
        residual_ax,
        diagnostic["spectral_residual"],
        labels,
        title="TeReL-S residual state",
        purity=float(diagnostic["purity_residual"]),
    )
    residual_ax.text(-0.06, 1.03, "d", transform=residual_ax.transAxes,
                     fontweight="bold")

    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight",
                transparent=True, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output_stem.with_suffix(".png"), dpi=300,
                bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("final_output", type=Path)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("residual-state-behavior"),
    )
    args = parser.parse_args()
    render(args.final_output, args.diagnostic, args.output)
