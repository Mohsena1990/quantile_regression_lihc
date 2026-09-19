"""Shared publication-figure saving: every figure is written as high-resolution PNG and vector PDF.

Per project convention, all figures live under outputs/figures/ (see AUDIT_AND_REPAIR_REPORT.md).
"""
from __future__ import annotations
from pathlib import Path

PNG_DPI = 600  # high-resolution raster (Elsevier/Nature line-art/combination-figure standard; PDF is vector regardless)

def save_figure(fig, path_stem: Path | str, *, dpi: int = PNG_DPI, pad_inches: float = 0.1) -> tuple[Path, Path]:
    """Save `fig` (a matplotlib Figure, or an Axes -- its .figure is used) as PNG and PDF.

    `path_stem` should have no extension, e.g. ROOT/'outputs/figures/climate_hdd_annual_2017'.
    `pad_inches` widens the `bbox_inches="tight"` crop -- matplotlib's tight-bbox
    calculation can under-measure radially-placed text (e.g. polar-axes tick labels
    at the 0-degree position), clipping the last glyph or two; the default here is
    generous enough to absorb that without visibly affecting normal rectangular
    figures. Returns (png_path, pdf_path).
    """
    figure = fig if hasattr(fig, "savefig") else fig.figure
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=pad_inches)  # PDF is vector; dpi only affects any rasterized elements
    return png_path, pdf_path
