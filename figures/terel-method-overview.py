"""Render the central TeReL-S mechanism schematic used as Figure 1.

The composition deliberately keeps algebra and graphical marks separate: text
never sits inside a container, arrows occupy dedicated gutters, and the three
stages follow one left-to-right reading order.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


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
ORANGE = "#B85C16"
TEAL = "#0F766E"
BLUE = "#2B67A0"
INK = "#111827"
MUTED = "#59616E"
HAIRLINE = "#D3D6DC"
LIGHT_PURPLE = "#DCD5F7"


def arrow(ax, x0, y0, x1, y1, *, color=PURPLE, linewidth=1.25, scale=9):
    """Draw one short, unobstructed transition arrow."""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def stage_header(ax, x, panel, title, subtitle):
    ax.text(x, 0.955, panel, fontsize=10.8, weight="bold", va="top", color=INK)
    ax.text(
        x + 0.027,
        0.955,
        title,
        fontsize=10.2,
        weight="bold",
        va="top",
        color=INK,
    )
    ax.text(
        x + 0.027,
        0.888,
        subtitle,
        fontsize=7.3,
        va="top",
        color=MUTED,
    )


def objective_term(ax, y, color, name, expression):
    """Use a color key beside the algebra, never behind it."""
    ax.plot([0.035, 0.035], [y - 0.030, y + 0.030], color=color, linewidth=2.4,
            solid_capstyle="round")
    ax.text(0.049, y + 0.021, name, fontsize=6.5, weight="bold", color=color,
            va="center")
    ax.text(0.049, y - 0.020, expression, fontsize=8.8, color=INK, va="center")


fig = plt.figure(figsize=(7.10, 3.12), facecolor="white")
ax = fig.add_axes((0.018, 0.035, 0.964, 0.94))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Three stages, one reading direction. Whitespace—not containers—separates them.
stage_header(ax, 0.005, "a", "Form a local target", "Three soft-SFA forces")
stage_header(ax, 0.365, "b", "Define the neuron state", "Residual, then same-layer settling")
stage_header(ax, 0.700, "c", "Update local synapses", "The settled state supplies both rules")

# a. The target is displayed in dependency order. The thin colored strokes are
# keys, not backgrounds, so none of the mathematical text collides with shapes.
objective_term(ax, 0.760, ORANGE, "SLOW", r"$\omega_t\,(z_t-p_t)$")
objective_term(
    ax,
    0.650,
    TEAL,
    "NONCOLLAPSE",
    r"$-\frac{\lambda_V}{\lambda_S}\,g\odot(z_t-m)$",
)
objective_term(
    ax,
    0.540,
    BLUE,
    "DECORRELATE",
    r"$+\frac{\lambda_C}{2\lambda_S}\,A(z_t-m)$",
)
ax.plot([0.048, 0.286], [0.460, 0.460], color=HAIRLINE, linewidth=0.9)
ax.text(0.048, 0.405,
        r"$r_t=r_t^{\mathrm{slow}}+r_t^{\mathrm{var}}+r_t^{\mathrm{cov}}$",
        fontsize=8.7, color=INK, va="center")
ax.text(0.048, 0.325, r"$\hat z_t=\mathrm{sg}(z_t-r_t)$", fontsize=10.2,
        color=PURPLE, weight="bold", va="center")

# Dedicated gutters carry stage transitions; arrows never traverse text.
arrow(ax, 0.315, 0.575, 0.350, 0.575)

# b. First map the activation residual to the neuron's preactivation. Then let
# only neurons in the same layer settle that state through M.
ax.text(0.392, 0.745, "activation residual", fontsize=6.8, weight="bold",
        color=MUTED, va="center")
ax.text(0.392, 0.690, r"$z_t-\hat z_t$", fontsize=10.2, color=INK, va="center")
arrow(ax, 0.435, 0.640, 0.435, 0.575, color=MUTED, linewidth=1.0, scale=8)
ax.text(0.451, 0.606, r"$J_{\phi,t}^{\mathsf{T}}$", fontsize=7.8, color=MUTED,
        va="center")
ax.text(0.392, 0.525, "base neuron state", fontsize=6.8, weight="bold",
        color=MUTED, va="center")
ax.text(0.392, 0.470, r"$b_t=J_{\phi,t}^{\mathsf{T}}(z_t-\hat z_t)$",
        fontsize=9.6, color=INK, va="center")

# A separate, quiet settling motif: b enters from the left, same-layer coupling
# acts along the center line, and s leaves on the right.
ax.text(0.392, 0.352, r"$b_t$", fontsize=9.8, color=INK, va="center")
ax.plot([0.425, 0.565], [0.352, 0.352], color=LIGHT_PURPLE, linewidth=4.4,
        solid_capstyle="round")
arrow(ax, 0.425, 0.352, 0.565, 0.352, linewidth=1.15, scale=8)
ax.text(0.495, 0.391, "same-layer inhibition", fontsize=6.8, color=PURPLE,
        ha="center", va="center")
ax.text(0.579, 0.352, r"$s_t$", fontsize=10.0, color=PURPLE, weight="bold",
        va="center")
ax.text(0.392, 0.275, r"$(I+\kappa M)s_t\approx b_t$", fontsize=9.1,
        color=INK, va="center")

arrow(ax, 0.655, 0.575, 0.690, 0.575)

# c. A typographic fork makes the shared-state claim visible without placing
# equations inside boxes or running connectors through their labels.
ax.text(0.725, 0.715, r"$s_{t,j}$", fontsize=12.0, weight="bold", color=PURPLE,
        ha="center", va="center")
ax.plot([0.749, 0.773], [0.715, 0.715], color=PURPLE, linewidth=1.25)
ax.plot([0.773, 0.773], [0.535, 0.715], color=PURPLE, linewidth=1.25)
ax.plot([0.773, 0.795], [0.655, 0.655], color=PURPLE, linewidth=1.25)
ax.plot([0.773, 0.795], [0.535, 0.535], color=PURPLE, linewidth=1.25)

ax.text(0.805, 0.690, "FEEDFORWARD GRADIENT", fontsize=6.5, weight="bold",
        color=PURPLE, va="center")
ax.text(0.805, 0.640, r"$\nabla_{W_{ji}}\widetilde L_a\propto s_{t,j}x_{t,i}$",
        fontsize=9.1, color=INK, va="center")
ax.text(0.805, 0.570, "LATERAL CHANGE", fontsize=6.5, weight="bold",
        color=TEAL, va="center")
ax.text(0.805, 0.520,
        r"$\Delta L^{\mathrm{lat}}_{jk}\propto-s_{t,j}s_{t,k}$",
        fontsize=8.8, color=INK, va="center")
ax.text(0.725, 0.353, r"$x_{t,i}$", fontsize=9.5, color=INK, va="center")
ax.plot([0.767, 0.935], [0.352, 0.352], color=HAIRLINE, linewidth=1.0)
arrow(ax, 0.767, 0.352, 0.935, 0.352, color=MUTED, linewidth=0.9, scale=7)
ax.text(0.851, 0.397, r"$W_{ji}$", fontsize=7.7, color=MUTED, ha="center",
        va="center")
ax.text(0.955, 0.352, r"$s_{t,j}$", fontsize=9.5, color=PURPLE, ha="right",
        va="center")

# The locality claim is a boundary statement, not another flowchart. A single
# rule cleanly separates it from the mechanism above.
ax.plot([0.005, 0.995], [0.205, 0.205], color=HAIRLINE, linewidth=0.9)
ax.text(0.020, 0.153, "LOCAL IN TIME", fontsize=6.7, weight="bold",
        color=PURPLE, va="center")
ax.text(0.020, 0.103, r"one detached predecessor $p_t$; no graph through time",
        fontsize=7.3, color=INK, va="center")
ax.plot([0.500, 0.500], [0.080, 0.175], color=HAIRLINE, linewidth=0.8)
ax.text(0.525, 0.153, "LOCAL IN SPACE", fontsize=6.7, weight="bold",
        color=PURPLE, va="center")
ax.text(0.525, 0.103, "updates read endpoint states; no error crosses layers",
        fontsize=7.3, color=INK, va="center")

source_directory = Path(__file__).resolve().parent
fixed_pdf_time = datetime(2026, 8, 9, tzinfo=timezone.utc)
fig.savefig(
    source_directory / "terel-method-overview.pdf",
    bbox_inches="tight",
    pad_inches=0.01,
    metadata={"CreationDate": fixed_pdf_time, "ModDate": fixed_pdf_time},
)
fig.savefig(
    source_directory / "terel-method-overview.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
    facecolor="white",
)
plt.close(fig)
