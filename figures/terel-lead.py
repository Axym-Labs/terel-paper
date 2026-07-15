"""Render the TeReL lead figure from the retained spectral plot and MNIST results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPECTRAL_PLOT = ROOT / "v12_standard_setup_terel_baseline/spectral_representations.png"

AXYM_PRIMARY = "#3F21B6"
AXYM_SECONDARY = "#8C7AD3"
COMPARISON_GRAY = "#BDBDBD"


def main() -> None:
    plt.style.use(HERE / "paper.mplstyle")
    spectral = plt.imread(SPECTRAL_PLOT)
    if spectral.shape[-1] == 3:
        spectral = np.dstack((spectral, np.ones(spectral.shape[:2])))
    near_white = np.all(spectral[:, :, :3] > 0.995, axis=2)
    spectral[near_white, 3] = 0.0

    fig = plt.figure(figsize=(7.15, 3.0))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.7, 1.0), wspace=0.14)

    projection_axis = fig.add_subplot(grid[0, 0])
    projection_axis.imshow(spectral)
    projection_axis.set_axis_off()
    projection_axis.text(
        0.015, 0.985, "a", transform=projection_axis.transAxes,
        va="top", fontweight="bold", fontsize=9
    )

    performance_axis = fig.add_subplot(grid[0, 1])
    labels_bar = ["TeReL", "TeReL-S", "BP", "FF"]
    values = [97.99, 97.63, 98.12, 98.64]
    colors = [AXYM_PRIMARY, AXYM_SECONDARY, COMPARISON_GRAY, COMPARISON_GRAY]
    bars = performance_axis.bar(
        np.arange(len(values)), values, width=0.68, color=colors,
        edgecolor="white", linewidth=0.6, zorder=3
    )
    performance_axis.set_ylim(92.0, 100.0)
    performance_axis.set_yticks([92, 94, 96, 98, 100])
    performance_axis.set_ylabel("Validation accuracy (%)")
    performance_axis.set_xticks(np.arange(len(values)), labels_bar)
    performance_axis.grid(axis="y", color="#E5E7EB", linewidth=0.55, zorder=0)
    performance_axis.text(
        0.01, 0.98, "b", transform=performance_axis.transAxes,
        va="top", fontweight="bold", fontsize=9
    )
    for index, (bar, value) in enumerate(zip(bars, values)):
        performance_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.16,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold" if index == 0 else "normal",
        )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.15)
    fig.savefig(
        HERE / "terel-lead.pdf",
        transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(HERE / "terel-lead.png", dpi=300, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
