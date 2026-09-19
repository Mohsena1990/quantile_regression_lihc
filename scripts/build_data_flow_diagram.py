"""Data-flow diagram (DFD) for the paper: complements
scripts/build_swimlane_flowchart.py with the classic DFD view -- external entities
(square corners), numbered processes (rounded boxes), and data stores (open-ended
bars), connected by labeled data flows. The swimlane figure shows *who does what,
in what order*; this one shows *what data moves where* -- external data sources on
the left, the four on-disk data stores every stage reads/writes on the right, and
each process (1.0-7.0) as the transformation between them. Same matplotlib
patch-based style as the other flowchart scripts (no new diagramming dependency).

Output:
  outputs/figures/data_flow_diagram.png/.pdf
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
ENTITY = ("#f4ede2", "#a6763c")   # external entity -- square corners, warm neutral
PROCESS = ("#dce8f5", "#2a78d6")  # numbered process -- rounded box, matches other figures
STORE = ("#eef7ee", "#3d8f3d")    # data store -- open-ended bar, green family


def entity(ax, cx, cy, w, h, text, *, fontsize=8.3):
    fill, border = ENTITY
    patch = Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=1.3, edgecolor=border,
                       facecolor=fill, zorder=2)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=3, linespacing=1.3)
    return (cx, cy, w, h)


def process(ax, cx, cy, w, h, number, text, *, fontsize=8.3):
    fill, border = PROCESS
    patch = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.09",
                            linewidth=1.3, edgecolor=border, facecolor=fill, zorder=2)
    ax.add_patch(patch)
    ax.text(cx, cy + h * 0.24, number, ha="center", va="center", fontsize=7.3, color=border,
            fontweight="bold", zorder=3)
    ax.text(cx, cy - h * 0.14, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=3,
            linespacing=1.25)
    return (cx, cy, w, h)


def store(ax, cx, cy, w, h, label, text, *, fontsize=8.0):
    fill, border = STORE
    # Open-ended bar: top/bottom rules only, no left/right border -- the standard DFD data-store glyph.
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=0, facecolor=fill, zorder=1))
    ax.plot([cx - w / 2, cx + w / 2], [cy + h / 2, cy + h / 2], color=border, linewidth=1.4, zorder=2)
    ax.plot([cx - w / 2, cx + w / 2], [cy - h / 2, cy - h / 2], color=border, linewidth=1.4, zorder=2)
    ax.text(cx - w / 2 + 0.12, cy, label, ha="left", va="center", fontsize=7.2, color=border,
            fontweight="bold", zorder=3)
    ax.text(cx + 0.18, cy, text, ha="center", va="center", fontsize=fontsize, color=INK, zorder=3, linespacing=1.2)
    return (cx, cy, w, h)


def flow(ax, a, b, label="", *, color=INK_MUTED, connectionstyle="arc3,rad=0.0", label_frac=0.5,
         a_dy=0.0, b_dy=0.0):
    (ax_, ay, aw, ah) = a
    (bx, by, bw, bh) = b
    ay += a_dy
    by += b_dy
    if abs(bx - ax_) >= abs(by - ay):
        start = (ax_ + aw / 2 * (1 if bx > ax_ else -1), ay)
        end = (bx - bw / 2 * (1 if bx > ax_ else -1), by)
    else:
        start = (ax_, ay + ah / 2 * (1 if by > ay else -1))
        end = (bx, by - bh / 2 * (1 if by > ay else -1))
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                                  color=color, connectionstyle=connectionstyle, zorder=1, shrinkA=0, shrinkB=0))
    if label:
        mx = start[0] + (end[0] - start[0]) * label_frac
        my = start[1] + (end[1] - start[1]) * label_frac
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.3, color=INK_MUTED, style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", facecolor=SURFACE, edgecolor="none"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(15.6, 15.5))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 15.5)
    ax.set_axis_off()

    stores_x = 13.4

    # External entities (left column, top to bottom).
    e_enable = entity(ax, 2.0, 14.0, 3.4, 0.85, "ENABLE.EU\nhousehold survey")
    e_ref = entity(ax, 2.0, 12.5, 3.4, 0.85, "Eurostat / EU-SILC /\nEC oil bulletin / NUTS")
    e_era5 = entity(ax, 2.0, 11.0, 3.4, 0.85, "ERA5-Land\n(Copernicus CDS)")

    # Data stores (right column) -- D1 is tall since almost every process touches it;
    # each interacting process gets its own fixed vertical anchor (b_dy below) so the
    # six lines into D1 land at six distinct points instead of one illegible knot.
    s_pred = store(ax, stores_x, 12.15, 3.4, 2.6, "D1", "artifacts/predictions/\n*.parquet")
    s_manifest = store(ax, stores_x, 9.55, 3.4, 0.85, "D2", "artifacts/manifests/\n*.json")
    s_tables = store(ax, stores_x, 6.9, 3.4, 1.9, "D3", "outputs/tables/\n*.csv")
    s_figures = store(ax, stores_x, 4.55, 3.4, 0.85, "D4", "outputs/figures/\n*.png, *.pdf")

    # Processes, numbered in pipeline order, center column.
    p1 = process(ax, 7.4, 13.7, 3.6, 1.0, "1.0", "Build retained cohort +\nimmutable train/val/test split")
    flow(ax, e_enable, p1, "household records")
    flow(ax, e_ref, p1, "currency rates, prices")
    flow(ax, e_era5, p1, "HDD/CDD, frost days")
    flow(ax, p1, s_pred, "cohort frame", b_dy=1.2, label_frac=0.42)

    p2 = process(ax, 7.4, 12.1, 3.6, 1.0, "2.0", "Fit R0-R3 regression\nmeasurements (M0, M2/M5/M7)")
    flow(ax, s_pred, p2, "cohort frame", a_dy=0.9, label_frac=0.58)
    flow(ax, p2, s_pred, "predicted_cost", b_dy=0.6, label_frac=0.42)

    p3 = process(ax, 7.4, 10.5, 3.6, 1.0, "3.0", "Label four-class risk\n(HighCost x LowIncome)")
    flow(ax, s_pred, p3, "predicted_cost", a_dy=0.3, label_frac=0.58)
    flow(ax, p3, s_pred, "risk_class_code", b_dy=0.0, label_frac=0.42)

    p4 = process(ax, 7.4, 8.9, 3.6, 1.0, "4.0", "Screen C0-C1 classifiers")
    flow(ax, s_pred, p4, "risk_class_code", a_dy=-0.3, label_frac=0.58)
    flow(ax, p4, s_pred, "truth, prediction,\nprob_0..3", b_dy=-0.6, label_frac=0.42)
    flow(ax, p4, s_tables, "screening_*.csv", b_dy=0.85, label_frac=0.42)

    p5 = process(ax, 7.4, 7.25, 3.6, 1.0, "5.0", "Select + freeze\nvalidation champion")
    flow(ax, s_tables, p5, "screening metrics", a_dy=0.51, label_frac=0.58)
    flow(ax, p5, s_manifest, "final_selection_frozen.json", label_frac=0.22)
    flow(ax, p5, s_tables, "table_family_champions.csv", b_dy=0.17, label_frac=0.42)

    p6 = process(ax, 7.4, 5.6, 3.6, 1.0, "6.0", "Per-country A0-A2 feature\nselection + B-family scoring")
    flow(ax, s_manifest, p6, "champion choice", label_frac=0.75)
    flow(ax, p6, s_pred, "percountry/*.parquet,\nb{family}.parquet", b_dy=-0.9, connectionstyle="arc3,rad=0.1",
         label_frac=0.22)
    flow(ax, p6, s_tables, "table_percountry_*.csv", b_dy=-0.17, label_frac=0.42)

    p7 = process(ax, 7.4, 3.3, 3.6, 1.0, "7.0", "Build reports:\nconfusion matrices, maps,\nevaluation summary")
    flow(ax, s_pred, p7, "predictions", a_dy=-1.2, connectionstyle="arc3,rad=-0.18", label_frac=0.62)
    flow(ax, s_tables, p7, "metrics tables", a_dy=-0.51, label_frac=0.58)
    flow(ax, p7, s_figures, "figures", label_frac=0.5)
    flow(ax, p7, s_tables, "summary tables", b_dy=-0.85, connectionstyle="arc3,rad=0.18", label_frac=0.35)

    ax.set_title("LIHC measurement + screening pipeline -- data-flow diagram\n"
                 "(external entities: square corners; processes: numbered rounded boxes; "
                 "data stores D1-D4: open-ended bars)", fontsize=11.5, pad=14)
    fig.tight_layout()
    save_figure(fig, ROOT / "outputs/figures/data_flow_diagram")
    plt.close(fig)
    print("Wrote data-flow diagram -> outputs/figures/data_flow_diagram.png/.pdf")


if __name__ == "__main__":
    main()
