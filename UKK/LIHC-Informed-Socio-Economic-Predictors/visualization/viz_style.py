"""
Shared chart style for the validation-findings visualization suite.

A validated categorical palette (fixed hue order -- never cycled;
adjacent pairs clear the colorblind-safety gates), a single-hue sequential
ramp for magnitude, and a blue<->red diverging pair for polarity (odds
ratios above/below 1). See dataviz skill's references/palette.md for the
full validation basis; only the hex values are reproduced here.
"""

from pathlib import Path

import matplotlib.pyplot as plt

# Fixed categorical order -- identity, never re-cycled or reassigned per chart.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# This project's three recurring "methods" get a fixed slot each, everywhere.
METHOD_COLORS = {
    "Traditional LIHC": CATEGORICAL[0],
    "HQRTM": CATEGORICAL[1],
    "Restriction flag": CATEGORICAL[2],
}

SEQUENTIAL_BLUE = {
    100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef",
    300: "#6da7ec", 350: "#5598e7", 400: "#3987e5", 450: "#2a78d6",
    500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281",
    700: "#0d366b",
}

DIVERGING_POSITIVE = "#2a78d6"  # blue: odds ratio > 1
DIVERGING_NEGATIVE = "#e34948"  # red: odds ratio < 1
DIVERGING_NEUTRAL = "#f0efec"   # gray midpoint

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT_FAMILY = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def apply_chart_style(fig: plt.Figure, axes) -> None:
    """Apply consistent chrome: surface color, recessive grid, muted ticks."""
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)
    plt.rcParams["font.family"] = FONT_FAMILY

    for ax in axes:
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED_INK, labelsize=10)
        ax.xaxis.label.set_color(SECONDARY_INK)
        ax.yaxis.label.set_color(SECONDARY_INK)
        ax.title.set_color(PRIMARY_INK)
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path.with_suffix('.png')} (+ .pdf)")
