"""Render Laplacian projections with a full-space neighborhood audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

AXYM = "#3F21B6"
INK = "#111827"
MUTED = "#4B5563"
RULE = "#D8DCE2"
MNIST_COLORS = tuple(plt.get_cmap("tab10").colors)


def _style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["P052", "Palatino", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.7,
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


def _load(path: Path):
    values = np.load(path)
    x = normalize(values["geometry_representations"].astype(np.float64))
    y = values["geometry_labels"].astype(np.int64)
    return x, y


def _embedding(x):
    return SpectralEmbedding(
        n_components=2,
        affinity="nearest_neighbors",
        n_neighbors=20,
        eigen_solver="arpack",
        random_state=1701,
        n_jobs=-1,
    ).fit_transform(x)


def _purity(x, y, ks):
    neighbors = NearestNeighbors(
        n_neighbors=max(ks) + 1,
        metric="cosine",
        n_jobs=-1,
    ).fit(x)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    values = [np.mean(y[indices[:, :k]] == y[:, None]) for k in ks]
    proportions = np.unique(y, return_counts=True)[1] / len(y)
    chance = float(np.sum(proportions**2))
    return np.asarray(values), chance


def _projection_panel(ax, embedding, labels, colors):
    classes = np.unique(labels)
    for color, class_id in zip(colors, classes, strict=True):
        selected = labels == class_id
        ax.scatter(
            embedding[selected, 0],
            embedding[selected, 1],
            s=3.0,
            alpha=0.62,
            color=color,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xlabel("Laplacian coordinate 1")
    ax.set_ylabel("Laplacian coordinate 2")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    _style()
    mnist_x, mnist_y = _load(arguments.mnist)
    mnist_embedding = _embedding(mnist_x)
    ks = (1, 2, 5, 10, 20, 50)
    mnist_purity, mnist_chance = _purity(mnist_x, mnist_y, ks)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.65),
        gridspec_kw={"width_ratios": (1.0, 1.15), "wspace": 0.31},
    )
    _projection_panel(axes[0], mnist_embedding, mnist_y, MNIST_COLORS)
    neighborhood_axis = axes[1]

    neighborhood_axis.plot(
        ks,
        mnist_purity,
        color=AXYM,
        marker="o",
        markersize=3.5,
        linewidth=1.25,
    )
    neighborhood_axis.axhline(
        mnist_chance, color=AXYM, linewidth=0.7, linestyle=":"
    )
    neighborhood_axis.set_xscale("log")
    neighborhood_axis.set_xticks(ks, [str(k) for k in ks])
    neighborhood_axis.set_ylim(0.0, 1.02)
    neighborhood_axis.set_xlabel("Neighbors, k")
    neighborhood_axis.set_ylabel("Same-class neighbor fraction")
    neighborhood_axis.grid(axis="y", color=RULE, linewidth=0.55)

    titles = ["MNIST", "Full-space neighborhoods"]
    for label, title, ax in zip(
        "ab",
        titles,
        axes,
        strict=True,
    ):
        ax.text(
            -0.10,
            1.05,
            label,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            color=INK,
            fontweight="bold",
            fontsize=9,
        )
        ax.text(
            0.5,
            1.05,
            title,
            transform=ax.transAxes,
            va="bottom",
            ha="center",
            color=INK,
            fontweight="bold",
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(arguments.output.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(
        arguments.output.with_suffix(".png"),
        bbox_inches="tight",
        transparent=True,
        dpi=320,
    )
    plt.close(fig)
    arguments.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "neighbors": list(ks),
                "mnist": {
                    "observations": len(mnist_y),
                    "same_class_fraction": mnist_purity.tolist(),
                    "class_frequency_chance": mnist_chance,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
