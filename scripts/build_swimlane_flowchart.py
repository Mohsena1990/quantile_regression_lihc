"""Swimlane flowchart for the paper: replaces scripts/build_process_flowchart.py's
single generic box-and-arrow diagram with a cross-functional swimlane view -- one
lane per pipeline subsystem, boxes advance left-to-right in process order, and
every box carries only a short code (R0-R3 / C0-C1 / A0-A2), never a full
method name -- the code->name mapping lives in outputs/tables/table_symbol_legend.csv
and the figure's own footer, not repeated in every box.

See also scripts/build_data_flow_diagram.py for the companion DFD (data stores +
flows between processes/external entities), a different, complementary view of
the same pipeline: this figure shows *who does what, in what order*; the DFD
shows *what data moves where*.

Output:
  outputs/figures/swimlane_flowchart.png/.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.figures import save_figure

SURFACE, GRID, INK, INK_MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#898781"
LANE_FILL = ["#f5f7fb", "#fcfcfb"]  # alternating banding, barely-there
PROCESS = ("#dce8f5", "#2a78d6")
DECISION = ("#fdf0d5", "#c98a12")
OUTPUT = ("#e3f3e1", "#0ca30c")
BLOCKED = ("#eeece4", "#898781")

LANES = ["DATA", "MEASURE", "LABEL", "SCREEN", "SELECT", "A0-A2 / B", "REPORT"]
LANE_H = 1.5
LEFT_MARGIN = 1.9
TOTAL_W = 12.6


def lane_y(name: str) -> float:
    i = LANES.index(name)
    return (len(LANES) - i) * LANE_H - LANE_H / 2


def box(ax, cx, lane_name, text, role, *, w=1.3, h=1.0, fontsize=8.6):
    cy = lane_y(lane_name)
    fill, border = role
    patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                            linewidth=1.2, edgecolor=border, facecolor=fill, zorder=3)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=4, linespacing=1.2,
             fontweight="bold")
    return (cx, cy, w, h)


def arrow(ax, a, b, *, color=INK_MUTED):
    (ax_, ay, aw, ah) = a
    (bx, by, bw, bh) = b
    if abs(bx - ax_) >= abs(by - ay):
        start = (ax_ + aw / 2 * (1 if bx > ax_ else -1), ay)
        end = (bx - bw / 2 * (1 if bx > ax_ else -1), by)
    else:
        start = (ax_, ay + ah / 2 * (1 if by > ay else -1))
        end = (bx, by - bh / 2 * (1 if by > ay else -1))
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.15,
                                  color=color, zorder=2))


def main() -> None:
    total_h = len(LANES) * LANE_H + 0.6
    fig, ax = plt.subplots(figsize=(TOTAL_W, total_h))
    ax.set_xlim(0, TOTAL_W)
    ax.set_ylim(0, total_h)
    ax.set_axis_off()

    for i, name in enumerate(LANES):
        y0 = total_h - 0.6 - (i + 1) * LANE_H
        ax.add_patch(Rectangle((0, y0), TOTAL_W, LANE_H, facecolor=LANE_FILL[i % 2], edgecolor="none", zorder=0))
        ax.axhline(y0, color=GRID, linewidth=0.8, zorder=1)
        ax.text(0.12, y0 + LANE_H / 2, name, ha="left", va="center", fontsize=9, color=INK_MUTED,
                fontweight="bold")
    ax.axhline(total_h - 0.6, color=GRID, linewidth=0.8, zorder=1)

    # x positions, left to right in process order (kept off the lane-label margin).
    x = dict(data=3.1, cohort=4.7, r0=6.0, r1_7=7.6, label=8.9, screen=8.9,
             champion=10.2, freeze=11.6, a0_4=10.2, bfam=11.6, report=11.6)

    b_data = box(ax, x["data"], "DATA", "ENABLE +\nREF + ERA5", PROCESS, w=1.7)
    b_cohort = box(ax, x["cohort"], "DATA", "COHORT\n+ SPLIT", PROCESS, w=1.4)
    arrow(ax, b_data, b_cohort)

    b_r0 = box(ax, x["r0"], "MEASURE", "R0", PROCESS, w=0.9, h=0.75)
    b_r1_7 = box(ax, x["r1_7"], "MEASURE", "R1-R3\n(A0-A2)", PROCESS, w=1.4)
    arrow(ax, b_cohort, b_r0)
    arrow(ax, b_cohort, b_r1_7)

    b_label = box(ax, x["label"], "LABEL", "4-CLASS", DECISION, w=1.4)
    arrow(ax, b_r0, b_label)
    arrow(ax, b_r1_7, b_label)

    b_screen = box(ax, x["screen"], "SCREEN", "C0-C1\n× imb.", PROCESS, w=1.4)
    arrow(ax, b_label, b_screen)

    b_champion = box(ax, x["champion"], "SELECT", "CHAMPION", DECISION, w=1.4)
    b_freeze = box(ax, x["freeze"], "SELECT", "FREEZE\n(locked)", BLOCKED, w=1.4)
    arrow(ax, b_screen, b_champion)
    arrow(ax, b_champion, b_freeze)

    b_a0_4 = box(ax, x["a0_4"], "A0-A2 / B", "A0-A2", PROCESS, w=1.4)
    b_bfam = box(ax, x["bfam"], "A0-A2 / B", "B-FAM", PROCESS, w=1.4)
    arrow(ax, b_champion, b_a0_4)
    arrow(ax, b_a0_4, b_bfam)

    b_report = box(ax, x["report"], "REPORT", "FIGS +\nTABLES", OUTPUT, w=1.4)
    arrow(ax, b_freeze, b_report)
    arrow(ax, b_bfam, b_report)

    ax.set_title("LIHC pipeline — swimlane flowchart (codes: see table_symbol_legend.csv)",
                 fontsize=12, pad=12, loc="left", x=0.01)
    fig.tight_layout()
    save_figure(fig, ROOT / "outputs/figures/swimlane_flowchart")
    plt.close(fig)
    print("Wrote swimlane flowchart -> outputs/figures/swimlane_flowchart.png/.pdf")


if __name__ == "__main__":
    main()
