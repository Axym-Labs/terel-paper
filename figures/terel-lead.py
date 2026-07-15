"""Render the TeReL lead figure from retained representation and MNIST results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "analysis_outputs/v12_standard_setup_terel_baseline"

AXYM_PRIMARY = "#3F21B6"
AXYM_SECONDARY = "#8C7AD3"
COMPARISON_GRAY = "#BDBDBD"


def cosine_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def laplacian_eigenmap(values: np.ndarray, neighbors: int = 12) -> np.ndarray:
    values = cosine_normalize(values.astype(np.float64))
    similarities = values @ values.T
    np.fill_diagonal(similarities, -np.inf)
    neighbors = min(neighbors, len(values) - 1)
    indices = np.argpartition(-similarities, kth=neighbors, axis=1)[:, :neighbors]

    weights = np.zeros_like(similarities)
    rows = np.arange(len(values))[:, None]
    weights[rows, indices] = np.maximum(similarities[rows, indices], 0.0)
    weights = np.maximum(weights, weights.T)

    degrees = weights.sum(axis=1)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-8))
    laplacian = np.eye(len(values)) - (
        inverse_sqrt[:, None] * weights * inverse_sqrt[None, :]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)
    return eigenvectors[:, order[1:3]]


def main() -> None:
    plt.style.use(HERE / "paper.mplstyle")
    representations = np.load(RESULTS / "reps_val.npy")
    labels = np.load(RESULTS / "labels_val.npy")
    indices = np.random.default_rng(7).choice(
        len(representations), size=min(1600, len(representations)), replace=False
    )
    projection = laplacian_eigenmap(representations[indices])

    fig = plt.figure(figsize=(7.15, 2.65))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.28)

    projection_axis = fig.add_subplot(grid[0, 0])
    projection_axis.scatter(
        projection[:, 0],
        projection[:, 1],
        c=labels[indices],
        cmap="tab10",
        s=5,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    projection_axis.set_xlabel("Laplacian eigenmap dimension 1")
    projection_axis.set_ylabel("Dimension 2")
    projection_axis.set_xticks([])
    projection_axis.set_yticks([])
    projection_axis.text(
        0.01, 0.98, "a", transform=projection_axis.transAxes,
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
    performance_axis.set_ylim(97.3, 99.0)
    performance_axis.set_yticks([97.5, 98.0, 98.5, 99.0])
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
            value + 0.035,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold" if index == 0 else "normal",
        )

    fig.subplots_adjust(left=0.055, right=0.995, top=0.98, bottom=0.16)
    fig.savefig(
        HERE / "terel-lead.pdf",
        transparent=True,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(HERE / "terel-lead.png", dpi=300, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
