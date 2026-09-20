"""Small map helpers shared by the climate figures (fuse_era5_nuts1.py)."""
from __future__ import annotations
import geopandas as gpd
import matplotlib.patheffects as pe


def country_label_points(geo: gpd.GeoDataFrame) -> dict[str, tuple[float, float]]:
    """One label point per country (ISO2 code), placed inside the country's own dissolved geometry via
    representative_point() -- safe for concave/multi-part shapes (e.g. Italy with Sicily and Sardinia) where a
    bounding-box or convex-hull centroid can fall outside the landmass or over the sea."""
    dissolved = geo.dissolve(by="CNTR_CODE")
    points = dissolved.geometry.representative_point()
    return {code: (pt.x, pt.y) for code, pt in points.items()}


def label_countries(ax, label_points: dict[str, tuple[float, float]]) -> None:
    for code, (x, y) in label_points.items():
        ax.annotate(code, (x, y), ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#0b0b0b", path_effects=[pe.withStroke(linewidth=2.2, foreground="#fcfcfb")])
