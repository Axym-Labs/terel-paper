"""Render the reader-first TeReL-S mechanism schematic used as Figure 1."""

from datetime import datetime, timezone
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


logging.getLogger("fontTools.ttLib.tables._h_e_a_d").setLevel(logging.ERROR)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PURPLE = "#3F21B6"
PURPLE_LIGHT = "#EEEAFB"
ORANGE = "#B95A16"
ORANGE_LIGHT = "#FFF0E4"
TEAL = "#14786B"
TEAL_LIGHT = "#E5F4F1"
INK = "#202127"
MUTED = "#62636B"
MID = "#AAAAB2"
PALE = "#F5F5F7"


def box(ax, x, y, w, h, text, *, edge=PURPLE, face="white", size=9.0,
        weight="normal", color=INK, radius=0.018, linewidth=1.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, weight=weight, color=color, linespacing=1.18)
    return patch


def arrow(ax, start, end, *, color=INK, width=1.25, connection="arc3",
          style="-|>", scale=10):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=scale, linewidth=width,
        color=color, connectionstyle=connection, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


fig = plt.figure(figsize=(7.05, 3.55), facecolor="white")
ax = fig.add_axes((0.02, 0.04, 0.96, 0.94))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Panel labels and terse headings do the orienting work; the caption carries prose.
ax.text(0.00, 0.98, "a", fontsize=11, weight="bold", va="top", color=INK)
ax.text(0.035, 0.98, "Target $\\rightarrow$ settled neuron state",
        fontsize=11, weight="bold", va="top", color=INK)
ax.text(0.61, 0.98, "b", fontsize=11, weight="bold", va="top", color=INK)
ax.text(0.645, 0.98, "One neuron state $\\rightarrow$ two updates",
        fontsize=10.7, weight="bold", va="top", color=INK)

# Panel a: objective components -> target -> base state -> settled state.
terms = [
    (0.03, 0.72, "slow", r"$z_t-p_t$", ORANGE, ORANGE_LIGHT),
    (0.03, 0.53, "noncollapsed", r"$-[\gamma-v]_+(z_t-m)$", TEAL, TEAL_LIGHT),
    (0.03, 0.34, "decorrelated", r"$A(z_t-m)$", PURPLE, PURPLE_LIGHT),
]
for x, y, name, equation, edge, face in terms:
    box(ax, x, y, 0.185, 0.13, name + "\n" + equation,
        edge=edge, face=face, size=8.2, weight="bold")
    arrow(ax, (x + 0.185, y + 0.065), (0.282, 0.595), color=edge, width=1.15)

box(ax, 0.275, 0.49, 0.14, 0.205,
    "target\n" + r"$\hat z_t=\mathrm{sg}(z_t-r_t)$",
    edge=PURPLE, face="white", size=9.1, weight="bold", linewidth=1.45)
arrow(ax, (0.415, 0.625), (0.445, 0.625), color=PURPLE, width=1.45)
box(ax, 0.450, 0.555, 0.125, 0.14,
    "base state\n" + r"$b_t=J_\phi^\top r_t$",
    edge=PURPLE, face="white", size=8.5, weight="bold", linewidth=1.4)
box(ax, 0.450, 0.34, 0.125, 0.14,
    "settled state\n" + r"$s_t\approx(I+\kappa M)^{-1}b_t$",
    edge=PURPLE, face=PURPLE_LIGHT, size=7.6, weight="bold", linewidth=1.55)
arrow(ax, (0.512, 0.555), (0.512, 0.48), color=PURPLE, width=1.35)
ax.text(0.535, 0.515, "inhibit", ha="left", va="center", fontsize=7.4,
        color=PURPLE, weight="bold")
ax.text(0.345, 0.445, r"$r_t=z_t-\hat z_t$", ha="center", fontsize=8.7,
        color=MUTED)

# Panel b: the same postsynaptic state meets pre-synaptic or lateral state.
cx, cy = 0.76, 0.60
ax.add_patch(Circle((cx, cy), 0.080, facecolor=PURPLE_LIGHT,
                    edgecolor=PURPLE, linewidth=1.6))
ax.text(cx, cy + 0.015, "neuron $j$", ha="center", va="center",
        fontsize=9.3, weight="bold", color=INK)
ax.text(cx, cy - 0.030, r"state $s_{t,j}$", ha="center", va="center",
        fontsize=9.0, weight="bold", color=PURPLE)

ax.add_patch(Circle((0.635, 0.60), 0.042, facecolor="white",
                    edgecolor=INK, linewidth=1.1))
ax.text(0.635, 0.60, r"$x_{t,i}$", ha="center", va="center", fontsize=8.7)
arrow(ax, (0.677, 0.60), (0.680, 0.60), color=INK, width=1.25)
ax.plot([0.677, 0.681], [0.60, 0.60], color=INK, linewidth=1.25)
ax.text(0.675, 0.675, r"$W_{ji}$", ha="center", fontsize=8.2, color=MUTED)

ax.add_patch(Circle((0.905, 0.60), 0.050, facecolor="white",
                    edgecolor=TEAL, linewidth=1.2))
ax.text(0.905, 0.60, r"$s_{t,k}$", ha="center", va="center",
        fontsize=8.7, color=TEAL, weight="bold")
arrow(ax, (0.855, 0.60), (0.840, 0.60), color=TEAL, width=1.35)
ax.text(0.858, 0.647, r"$L^{\rm lat}_{jk}<0$", ha="center", fontsize=7.0,
        color=TEAL)

box(ax, 0.615, 0.30, 0.19, 0.12,
    "feedforward\n" + r"$\Delta W_{ji}\propto-s_{t,j}x_{t,i}$",
    edge=PURPLE, face="white", size=8.5, weight="bold")
box(ax, 0.815, 0.30, 0.17, 0.12,
    "anti-Hebbian\n" + r"$\Delta L^{\rm lat}_{jk}=-\eta_Ms_{t,j}s_{t,k}$",
    edge=TEAL, face=TEAL_LIGHT, size=7.1, weight="bold")
arrow(ax, (0.735, 0.52), (0.71, 0.42), color=PURPLE, width=1.2)
arrow(ax, (0.805, 0.53), (0.885, 0.42), color=TEAL, width=1.2)

# Lower strip: locality in time and depth, without a second explanatory diagram.
ax.plot([0.015, 0.985], [0.235, 0.235], color="#D8D8DE", linewidth=0.8)
ax.text(0.00, 0.19, "c", fontsize=11, weight="bold", va="top", color=INK)
ax.text(0.035, 0.19, "Local in space and time", fontsize=10.5,
        weight="bold", va="top", color=INK)

timeline_y = 0.08
for x, label in [(0.34, r"$t-1$"), (0.46, r"$t$"), (0.58, r"$t+1$")]:
    ax.add_patch(Circle((x, timeline_y), 0.023, facecolor=PALE,
                        edgecolor=MID, linewidth=1.0))
    ax.text(x, timeline_y - 0.055, label, ha="center", fontsize=8.0,
            color=MUTED)
arrow(ax, (0.363, timeline_y), (0.437, timeline_y), color=MID, width=1.0)
arrow(ax, (0.483, timeline_y), (0.557, timeline_y), color=MID, width=1.0)
ax.plot([0.40, 0.40], [0.035, 0.145], color=PURPLE, linewidth=2.4,
        solid_capstyle="round")
ax.text(0.40, 0.16, "detach", ha="center", fontsize=7.8,
        color=PURPLE, weight="bold")
ax.text(0.76, 0.105, "fixed detached state; no temporal graph",
        ha="center", fontsize=8.2, color=INK)
ax.text(0.76, 0.050, "no error crosses a layer boundary",
        ha="center", fontsize=8.2, color=INK)

source_directory = Path(__file__).resolve().parent
fixed_pdf_time = datetime(2026, 8, 9, tzinfo=timezone.utc)
fig.savefig(
    source_directory / "terel-method-overview.pdf",
    bbox_inches="tight",
    metadata={"CreationDate": fixed_pdf_time, "ModDate": fixed_pdf_time},
)
fig.savefig(
    source_directory / "terel-method-overview.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)
