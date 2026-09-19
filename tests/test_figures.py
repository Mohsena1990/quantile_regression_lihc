"""Regression tests: every figure must be saved as both high-resolution PNG and vector PDF."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.reporting.figures import PNG_DPI, save_figure


def test_save_figure_writes_png_and_pdf(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png_path, pdf_path = save_figure(fig, tmp_path / "example")
    plt.close(fig)
    assert png_path.exists() and png_path.suffix == ".png"
    assert pdf_path.exists() and pdf_path.suffix == ".pdf"
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_save_figure_accepts_an_axes_directly(tmp_path):
    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [1, 2])
    png_path, pdf_path = save_figure(ax, tmp_path / "bar")
    plt.close(fig)
    assert png_path.exists() and pdf_path.exists()


def test_png_resolution_is_high_quality():
    assert PNG_DPI >= 300  # publication-quality floor; project default is 600
