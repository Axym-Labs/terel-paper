"""Render the TeReL locality schematic used as Figure 1."""

from datetime import datetime, timezone
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


BLUE = "#315DA8"
BLUE_LIGHT = "#EAF0FA"
INK = "#20242D"
MUTED = "#5C6370"
ORANGE = "#C65D16"
ORANGE_LIGHT = "#FFF0E5"
TEAL = "#177E70"
TEAL_LIGHT = "#E5F5F2"
RED = "#B23A48"
RED_LIGHT = "#FBEAEC"
GREY = "#F3F4F6"
MID_GREY = "#A8ADB7"


def rounded_box(
    ax,
    xy,
    width,
    height,
    text,
    *,
    face="white",
    edge=INK,
    size=9.2,
    weight="normal",
    text_color=INK,
    radius=0.018,
    linewidth=1.15,
):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        linewidth=linewidth,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=text_color,
        weight=weight,
        linespacing=1.25,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=INK,
    width=1.25,
    style="-|>",
    connection="arc3",
    mutation_scale=10,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=width,
        color=color,
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


fig, axes = plt.subplots(
    2,
    1,
    figsize=(7.05, 4.75),
    gridspec_kw={"height_ratios": [1.55, 0.85]},
)
fig.patch.set_facecolor("white")

# Panel A: the information path for one feedforward weight update.
ax = axes[0]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(
    0.0,
    0.98,
    "A  Information available to one feedforward weight update",
    fontsize=12.5,
    weight="bold",
    color=INK,
    va="top",
)
ax.text(
    0.0,
    0.905,
    "Signals may arrive from the same layer, but no error arrives from a later layer.",
    fontsize=9.1,
    color=MUTED,
    va="top",
)

# Feedforward path and explicit depth boundary.
rounded_box(
    ax,
    (0.015, 0.57),
    0.17,
    0.13,
    "presynaptic\nactivity\n" + r"$z^{\ell-1}_{t,i}$",
    face="white",
    edge=BLUE,
    size=9.0,
    weight="bold",
)
rounded_box(
    ax,
    (0.27, 0.555),
    0.20,
    0.18,
    "postsynaptic\nneuron " + r"$j$" + "\n" + r"$z^\ell_{t,j}$",
    face=BLUE_LIGHT,
    edge=BLUE,
    size=9.0,
    weight="bold",
)
rounded_box(
    ax,
    (0.82, 0.57),
    0.17,
    0.13,
    "later layer\n" + r"$\ell+1$",
    face=GREY,
    edge=MID_GREY,
    size=9.0,
    text_color=MUTED,
)
arrow(ax, (0.185, 0.635), (0.27, 0.635), color=BLUE)
ax.text(0.227, 0.67, r"$W^\ell_{ji}$", ha="center", va="bottom", fontsize=9.0, color=BLUE, weight="bold")
arrow(ax, (0.47, 0.635), (0.715, 0.635), color=BLUE)
arrow(ax, (0.735, 0.635), (0.82, 0.635), color=MID_GREY)
ax.plot([0.725, 0.725], [0.54, 0.73], color=RED, linewidth=3.0, solid_capstyle="round")
ax.text(
    0.725,
    0.755,
    "no downstream error",
    ha="center",
    va="bottom",
    color=RED,
    fontsize=8.5,
    weight="bold",
)

# The three local contributions remain visually separate and converge on e_tj.
signal_y = 0.185
signal_h = 0.14
rounded_box(
    ax,
    (0.015, signal_y),
    0.285,
    signal_h,
    "temporal coherence\n" + r"$z^\ell_{t,j}-p^\ell_{t,j}$",
    face=ORANGE_LIGHT,
    edge=ORANGE,
    size=9.0,
    weight="bold",
)
rounded_box(
    ax,
    (0.355, signal_y),
    0.29,
    signal_h,
    "variance expansion\n" + r"$[\gamma-v^\ell_j]_+(z^\ell_{t,j}-m^\ell_j)$",
    face=TEAL_LIGHT,
    edge=TEAL,
    size=8.7,
    weight="bold",
)
rounded_box(
    ax,
    (0.70, signal_y),
    0.285,
    signal_h,
    "lateral decorrelation\n" + r"$\sum_{k\ne j}A^\ell_{jk}q^\ell_{t,k}$",
    face=RED_LIGHT,
    edge=RED,
    size=8.8,
    weight="bold",
)
rounded_box(
    ax,
    (0.42, 0.40),
    0.26,
    0.105,
    "postsynaptic factor  " + r"$e^\ell_{t,j}$",
    face="white",
    edge=INK,
    size=9.1,
    weight="bold",
)
arrow(ax, (0.1575, signal_y + signal_h), (0.47, 0.40), color=ORANGE)
arrow(ax, (0.50, signal_y + signal_h), (0.55, 0.40), color=TEAL)
arrow(ax, (0.8425, signal_y + signal_h), (0.63, 0.40), color=RED)
arrow(
    ax,
    (0.42, 0.4525),
    (0.37, 0.555),
    color=BLUE,
    style="-|>",
    mutation_scale=9,
)

ax.text(
    0.50,
    0.055,
    r"Identity normalization:  "
    r"$\Delta W^\ell_{ji}\propto-\sum_t e^\ell_{t,j}\,\phi'(a^\ell_{t,j})\,z^{\ell-1}_{t,i}$",
    ha="center",
    va="center",
    fontsize=9.5,
    color=INK,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": MID_GREY},
)

# Panel B: spatial locality is shared; the temporal reference differs.
ax = axes[1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(
    0.0,
    0.97,
    "B  The temporal distinction between the variants",
    fontsize=12.5,
    weight="bold",
    color=INK,
    va="top",
)

# Column guides.
ax.text(0.18, 0.79, "temporal reference", fontsize=8.4, color=MUTED, ha="center", weight="bold")
ax.text(0.51, 0.79, "gradient through time", fontsize=8.4, color=MUTED, ha="center", weight="bold")
ax.text(0.84, 0.79, "execution consequence", fontsize=8.4, color=MUTED, ha="center", weight="bold")

row_specs = (
    (
        0.49,
        "canonical TeReL",
        "$p_t=z_{t-1}$ in the chunk",
        "retained across\nadjacent samples",
        "short chunk graph\nmust be retained",
        ORANGE,
        ORANGE_LIGHT,
    ),
    (
        0.16,
        "TeReL-S",
        "$p_t=\\mathrm{sg}(z_{t-1})$",
        "stopped at the\nstored reference",
        "fixed temporal state;\nno time graph",
        TEAL,
        TEAL_LIGHT,
    ),
)
for y, label, reference, gradient, consequence, edge, face in row_specs:
    rounded_box(ax, (0.005, y), 0.16, 0.19, label, face=face, edge=edge, size=8.9, weight="bold")
    rounded_box(ax, (0.195, y), 0.25, 0.19, reference, face="white", edge=edge, size=8.8)
    rounded_box(ax, (0.475, y), 0.26, 0.19, gradient, face="white", edge=edge, size=8.7)
    rounded_box(ax, (0.765, y), 0.225, 0.19, consequence, face=face, edge=edge, size=8.5, weight="bold")

plt.subplots_adjust(left=0.025, right=0.985, top=0.985, bottom=0.035, hspace=0.10)

source_directory = Path(__file__).resolve().parent
fixed_pdf_time = datetime(2026, 8, 3, tzinfo=timezone.utc)
fig.savefig(
    source_directory / "terel-method-overview.pdf",
    bbox_inches="tight",
    metadata={"CreationDate": fixed_pdf_time, "ModDate": fixed_pdf_time},
)
fig.savefig(
    source_directory / "terel-method-overview.png",
    dpi=240,
    bbox_inches="tight",
)
plt.close(fig)
