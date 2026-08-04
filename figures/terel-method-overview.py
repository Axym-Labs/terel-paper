"""Render the TeReL method and evidence overview used as the lead figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


PRIMARY = "#3F21B6"
PURPLE_LIGHT = "#EEEAFE"
INK = "#20212B"
MUTED = "#626578"
ORANGE = "#D66B19"
ORANGE_LIGHT = "#FFF0E5"
GREEN = "#167A61"
GREEN_LIGHT = "#E4F5F0"
RED = "#B63A4A"
GREY = "#F3F4F7"


def box(ax, xy, width, height, text, *, face=GREY, edge=INK, size=9, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.15,
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
        color=INK,
        weight=weight,
    )
    return patch


def arrow(ax, start, end, *, color=INK, width=1.15, style="-|>", connection="arc3"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
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
    figsize=(7.05, 5.2),
    gridspec_kw={"height_ratios": [1.05, 0.95]},
)
fig.patch.set_facecolor("white")

# Panel A: exact mechanism.
ax = axes[0]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(0.0, 0.98, "A  Local credit assignment", fontsize=13, weight="bold", color=INK, va="top")
ax.text(
    0.0,
    0.91,
    "Every loss updates one nonlinear layer; graph cuts are explicit.",
    fontsize=9.2,
    color=MUTED,
    va="top",
)

box(ax, (0.02, 0.68), 0.13, 0.11, "$x_t$", face="white", edge=PRIMARY, size=11, weight="bold")
box(ax, (0.23, 0.65), 0.22, 0.17, "Layer 1\n$z_t^1=\\phi(W^1x_t+c^1)$", face=PURPLE_LIGHT, edge=PRIMARY, size=9.2, weight="bold")
box(ax, (0.60, 0.65), 0.22, 0.17, "Layer 2\n$z_t^2=\\phi(W^2\\,\\mathrm{sg}(z_t^1)+c^2)$", face=PURPLE_LIGHT, edge=PRIMARY, size=9.0, weight="bold")
box(ax, (0.89, 0.68), 0.09, 0.11, "$z_t^2$", face="white", edge=PRIMARY, size=10.5, weight="bold")
arrow(ax, (0.15, 0.735), (0.23, 0.735), color=PRIMARY)
arrow(ax, (0.45, 0.735), (0.60, 0.735), color=PRIMARY)
arrow(ax, (0.82, 0.735), (0.89, 0.735), color=PRIMARY)
ax.plot([0.525, 0.525], [0.67, 0.80], color=RED, linewidth=2.3)
ax.text(0.525, 0.835, "stop-gradient", ha="center", va="bottom", color=RED, fontsize=8.5, weight="bold")

box(
    ax,
    (0.16, 0.28),
    0.36,
    0.22,
    "same-layer objective\n$L_S$: live in chunk (canonical)\n$L_V$: detached state $(m,v)$\n$L_C$: detached signal $Aq_t$",
    face="white",
    edge=PRIMARY,
    size=8.7,
)
box(
    ax,
    (0.60, 0.28),
    0.36,
    0.22,
    "same-layer objective\n$L_S$: live in chunk (canonical)\n$L_V$: detached state $(m,v)$\n$L_C$: detached signal $Aq_t$",
    face="white",
    edge=PRIMARY,
    size=8.7,
)
arrow(ax, (0.34, 0.65), (0.34, 0.50), color=PRIMARY)
arrow(ax, (0.71, 0.65), (0.78, 0.50), color=PRIMARY)

box(ax, (0.20, 0.08), 0.28, 0.10, "detached state update\nafter optimizer step", face=GREY, edge=MUTED, size=8.5)
box(ax, (0.64, 0.08), 0.28, 0.10, "detached state update\nafter optimizer step", face=GREY, edge=MUTED, size=8.5)
arrow(ax, (0.34, 0.28), (0.34, 0.18), color=MUTED, style="-|>")
arrow(ax, (0.78, 0.28), (0.78, 0.18), color=MUTED, style="-|>")
ax.text(
    0.02,
    0.01,
    "TeReL-S detaches $p_t$: $D^2+4D+1$ state elements; no temporal graph is retained.",
    fontsize=8.5,
    color=MUTED,
    va="bottom",
)

# Panel B: evidence roles.
ax = axes[1]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(0.0, 0.98, "B  Two locality regimes and retained accuracy", fontsize=13, weight="bold", color=INK, va="top")
ax.text(
    0.0,
    0.88,
    "Matched readouts compare methods; the samplewise variant tests bounded-state execution.",
    fontsize=9.2,
    color=MUTED,
    va="top",
)

box(ax, (0.01, 0.58), 0.18, 0.17, "class chunks\n60 data\npasses", face=ORANGE_LIGHT, edge=ORANGE, size=8.8, weight="bold")
box(ax, (0.28, 0.58), 0.21, 0.17, "canonical TeReL\nlayerwise training\nlive chunk history", face=PURPLE_LIGHT, edge=PRIMARY, size=8.8, weight="bold")
box(ax, (0.58, 0.58), 0.18, 0.17, "matched all-layer\nlinear probe", face="white", edge=ORANGE, size=8.8)
box(ax, (0.83, 0.60), 0.15, 0.13, "97.30%\n5 seeds", face=GREY, edge=RED, size=8.8, weight="bold")
arrow(ax, (0.19, 0.665), (0.28, 0.665), color=ORANGE)
arrow(ax, (0.49, 0.665), (0.58, 0.665), color=PRIMARY)
arrow(ax, (0.76, 0.665), (0.83, 0.665), color=RED)

box(ax, (0.01, 0.24), 0.18, 0.17, "one sample\nper update\n2 passes", face=GREEN_LIGHT, edge=GREEN, size=8.8, weight="bold")
box(ax, (0.28, 0.24), 0.21, 0.17, "TeReL-S\ndetached time\nrunning norm", face=PURPLE_LIGHT, edge=PRIMARY, size=8.8, weight="bold")
box(ax, (0.58, 0.24), 0.18, 0.17, "fixed state\nno time graph", face="white", edge=GREEN, size=8.8)
box(ax, (0.83, 0.26), 0.15, 0.13, "95.14%\n3 val. seeds", face=GREY, edge=RED, size=8.8, weight="bold")
arrow(ax, (0.19, 0.325), (0.28, 0.325), color=GREEN)
arrow(ax, (0.49, 0.325), (0.58, 0.325), color=PRIMARY)
arrow(ax, (0.76, 0.325), (0.83, 0.325), color=RED)

plt.subplots_adjust(left=0.025, right=0.985, top=0.985, bottom=0.035, hspace=0.16)

source_directory = Path(__file__).resolve().parent
fig.savefig(
    source_directory / "terel-method-overview.pdf",
    bbox_inches="tight",
    metadata={"CreationDate": None, "ModDate": None},
)
fig.savefig(
    source_directory / "terel-method-overview.png",
    dpi=220,
    bbox_inches="tight",
)
plt.close(fig)
