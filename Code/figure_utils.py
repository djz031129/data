from __future__ import annotations

from pathlib import Path
import pickle
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Wedge,
)
from matplotlib.colors import Normalize, LinearSegmentedColormap


# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------


def si_dir_from_script(script_file: str | Path) -> Path:
    """Return the Supporting_Information directory for scripts inside Code/*/."""
    return Path(script_file).resolve().parents[2]


def output_dir(si_dir: Path, kind: str) -> Path:
    out = si_dir / "Code" / "output" / kind
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------------------
# Common constants
# --------------------------------------------------------------------------------------

MODEL_ORDER = ["ET", "RF", "GB", "XGB", "LGBM", "CAT"]
TOP_MODELS = {
    "hardness": ["LGBM", "RF", "XGB"],
    "conductivity": ["LGBM", "XGB", "GB"],
}
TARGET_LABEL = {
    "hardness": "Hardness",
    "conductivity": "Electrical conductivity",
}
UNIT_LABEL = {
    "hardness": "HV",
    "conductivity": "%IACS",
}
SCATTER_COLOR = {
    "hardness": "#1f88d1",
    "conductivity": "#20a84a",
}
BAR_COLOR = {
    "hardness": "#2f65d9",
    "conductivity": "#2ca25f",
}

SHAP_CMAP = LinearSegmentedColormap.from_list("shap_blue_magenta", ["#1f5fff", "#ff0051"])

FEATURE_LABELS = {
    "num__CR1": "Cold reduction (%)",
    "num__Cu": "Cu (wt.%)",
    "num__time1 (h)": "Aging time (h)",
    "num__homogenization time (h)": "Homogenization time (h)",
    "num__Temp1 (℃)": "Aging temperature (°C)",
    "num__source_reliability": "Source reliability",
    "num__Si": "Si (wt.%)",
    "num__solution time (h)": "Solution time (h)",
    "cat__process classification_two-step aging": "Two-step aging",
    "num__solution temperature (℃)": "Solution temperature (°C)",
    "num__Ti": "Ti (wt.%)",
    "num__Zr": "Zr (wt.%)",
    "num__Ni": "Ni (wt.%)",
    "num__Cr": "Cr (wt.%)",
}

FIG1_CORR_LABELS = [
    "Cr", "Zr", "Mg", "Ti", "Aging\nTemp.", "Aging\nTime", "Cold\nRed.",
    "Solution\nTemp.", "Hardness\n(HV)", "Cond.\n(%IACS)",
]

# This is the correlation matrix displayed in Fig. 1(c). It is stored as numeric
# values in code so the figure can be drawn directly instead of copied from an image.
FIG1_CORR = np.array([
    [ 1.00, -0.12, -0.05, -0.07,  0.36,  0.05, -0.14,  0.39,  0.26, -0.15],
    [-0.12,  1.00, -0.16, -0.28,  0.08,  0.04, -0.19,  0.59,  0.45, -0.24],
    [-0.05, -0.16,  1.00, -0.05,  0.13, -0.11, -0.11,  0.09,  0.06,  0.05],
    [-0.07, -0.28, -0.05,  1.00, -0.12, -0.18,  0.90, -0.70, -0.05, -0.17],
    [ 0.36,  0.08,  0.13, -0.12,  1.00,  0.26, -0.26,  0.34,  0.09,  0.25],
    [ 0.05,  0.04, -0.11, -0.18,  0.26,  1.00, -0.45,  0.37,  0.25,  0.24],
    [-0.14, -0.19, -0.11,  0.90, -0.26, -0.45,  1.00, -0.88, -0.45, -0.24],
    [ 0.39,  0.59,  0.09, -0.70,  0.34,  0.37, -0.88,  1.00, -0.02,  0.43],
    [ 0.26,  0.45,  0.06, -0.05,  0.09,  0.25, -0.45, -0.02,  1.00, -0.13],
    [-0.15, -0.24,  0.05, -0.17,  0.25,  0.24, -0.24,  0.43, -0.13,  1.00],
])


# --------------------------------------------------------------------------------------
# Data readers
# --------------------------------------------------------------------------------------


def tables_dir(si_dir: Path) -> Path:
    return si_dir / "Tables_and_Results"


def data_dir(si_dir: Path) -> Path:
    return si_dir / "Data"


def read_metrics(si_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tables_dir(si_dir) / "metrics_summary_normalized.csv")


def read_predictions(si_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(tables_dir(si_dir) / "predictions.csv")
    df["split_norm"] = df["split"].astype(str).str.lower().replace({"validation": "val"})
    return df


def read_processed_data(si_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = data_dir(si_dir) / "processed"
    hardness = pd.read_csv(base / "cucrx_hardness_processed.csv")
    conductivity = pd.read_csv(base / "cucrx_conductivity_processed.csv")
    return hardness, conductivity


def read_shap_importance(si_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tables_dir(si_dir) / "shap_feature_importance_top10.csv")


def read_screening_map(si_dir: Path) -> pd.DataFrame:
    return pd.read_csv(tables_dir(si_dir) / "screening_map_predictions.csv")


# --------------------------------------------------------------------------------------
# Visual helpers
# --------------------------------------------------------------------------------------


def _panel_border(ax, label: str | None = None, edgecolor: str = "#0b4fa4") -> None:
    ax.set_axis_off()
    rect = FancyBboxPatch(
        (0.005, 0.01), 0.99, 0.98,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        linewidth=1.4, edgecolor=edgecolor, facecolor="white", transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(rect)
    if label:
        ax.text(0.018, 0.96, label, transform=ax.transAxes, va="top", ha="left",
                fontsize=18, fontweight="bold")


def _round_box(ax, x: float, y: float, w: float, h: float, number: int, title: str,
               edge: str, face: str = "#ffffff") -> None:
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                         linewidth=1.25, edgecolor=edge, facecolor=face, transform=ax.transAxes)
    ax.add_patch(box)
    cx, cy = x + 0.035, y + h - 0.055
    ax.scatter([cx], [cy], s=620, marker="o", color=edge, edgecolors=edge,
               transform=ax.transAxes, zorder=5, clip_on=False)
    ax.text(cx, cy, str(number), ha="center", va="center",
            color="white", fontsize=13, fontweight="bold", transform=ax.transAxes, zorder=6)
    title_font = 9.8 if "\n" in title else 10.5
    ax.text(x + 0.066, y + h - 0.045, title, ha="left", va="center",
            fontsize=title_font, fontweight="bold", linespacing=1.08, transform=ax.transAxes)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#0b4fa4") -> None:
    arr = FancyArrowPatch(start, end, arrowstyle="simple", mutation_scale=18,
                          color=color, lw=0, transform=ax.transAxes)
    ax.add_patch(arr)


def _tiny_bar_chart(ax, x: float, y: float, w: float, h: float, color: str) -> None:
    vals = [0.35, 0.55, 0.78]
    for i, v in enumerate(vals):
        ax.add_patch(Rectangle((x + i * w / 4, y), w / 6, h * v, transform=ax.transAxes,
                               facecolor=color, edgecolor=color, alpha=0.85))


def _draw_document_icon(ax, x: float, y: float, s: float) -> None:
    ax.add_patch(Rectangle((x, y), s * 0.70, s, transform=ax.transAxes,
                           facecolor="#f7fbff", edgecolor="#294472", lw=1))
    ax.add_patch(Polygon([[x + s * .52, y + s], [x + s * .70, y + s * .82], [x + s * .52, y + s * .82]],
                         closed=True, transform=ax.transAxes, facecolor="#dbe8ff", edgecolor="#294472", lw=0.8))
    for k in range(5):
        ax.plot([x + s * .12, x + s * .58], [y + s * (.78 - .13 * k)] * 2,
                transform=ax.transAxes, color="#6e8fc7", lw=1)


def _draw_database_icon(ax, x: float, y: float, s: float) -> None:
    ax.add_patch(Rectangle((x, y + s * .15), s * .70, s * .68, transform=ax.transAxes,
                           facecolor="#dfe7ef", edgecolor="#26374f", lw=1))
    for yy in [0.15, 0.38, 0.61, 0.83]:
        ax.add_patch(Wedge((x + s * .35, y + s * yy), s * .35, 0, 180, transform=ax.transAxes,
                           facecolor="#f5f7fa", edgecolor="#26374f", lw=1))
    ax.text(x + s * .35, y + s * .38, "Cu", transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="#f37021", ec="#983b0a", lw=0.8))


def _draw_sparkles(ax, cx: float, cy: float, s: float) -> None:
    for dx, dy, scale in [(0, 0, 1.0), (-0.055, 0.045, 0.42), (0.05, -0.045, 0.55)]:
        x, y = cx + dx, cy + dy
        pts = [[x, y + s * scale], [x + s * .20 * scale, y + s * .20 * scale],
               [x + s * scale, y], [x + s * .20 * scale, y - s * .20 * scale],
               [x, y - s * scale], [x - s * .20 * scale, y - s * .20 * scale],
               [x - s * scale, y], [x - s * .20 * scale, y + s * .20 * scale]]
        ax.add_patch(Polygon(pts, closed=True, transform=ax.transAxes,
                             facecolor="#2c6bb1", edgecolor="#0d2c54", lw=0.8))


def _draw_pie(ax, cx: float, cy: float, r: float) -> None:
    # Draw in an inset axis with equal aspect so it remains circular on the wide workflow panel.
    fig = ax.figure
    fig.canvas.draw_idle()
    bbox = ax.get_position()
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    axes_w = bbox.width * fig_w
    axes_h = bbox.height * fig_h
    width = 2.0 * r * (axes_h / axes_w)
    height = 2.0 * r
    pie_ax = ax.inset_axes([cx - width / 2, cy - height / 2, width, height])
    pie_ax.set_aspect("equal")
    pie_ax.axis("off")
    sizes = [72, 18, 10]
    colors = ["#2678b8", "#66a65c", "#f07c18"]
    wedges, _ = pie_ax.pie(sizes, startangle=70, colors=colors,
                           wedgeprops={"edgecolor": "white", "linewidth": 1.0})
    pie_ax.add_patch(Circle((0, 0), 0.32, facecolor="white", edgecolor="white", zorder=3))
    for size, wedge in zip(sizes, wedges):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        pie_ax.text(0.56 * np.cos(ang), 0.56 * np.sin(ang), f"{size}%",
                    color="white", fontsize=9, fontweight="bold", ha="center", va="center")


def _draw_tree(ax, x: float, y: float, w: float, h: float) -> None:
    nodes = [
        (0.5, 0.92), (0.32, 0.72), (0.68, 0.72), (0.18, 0.52), (0.42, 0.52), (0.58, 0.52), (0.82, 0.52),
        (0.12, 0.32), (0.26, 0.32), (0.38, 0.32), (0.49, 0.32), (0.61, 0.32), (0.74, 0.32), (0.88, 0.32),
    ]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (3, 7), (3, 8), (4, 9), (4, 10), (5, 11), (6, 12), (6, 13)]
    for i, j in edges:
        ax.plot([x + nodes[i][0] * w, x + nodes[j][0] * w], [y + nodes[i][1] * h, y + nodes[j][1] * h],
                transform=ax.transAxes, color="#222222", lw=0.8)
    for idx, (nx, ny) in enumerate(nodes):
        fc = "#367fb5" if idx in [0, 1, 2, 3, 5] else "#e6eef7"
        ax.add_patch(Circle((x + nx * w, y + ny * h), 0.008, transform=ax.transAxes,
                            facecolor=fc, edgecolor="#222", lw=0.6))
    ax.add_patch(Circle((x + w * .5, y + h * .6), w * .44, transform=ax.transAxes,
                        facecolor="none", edgecolor="#3f8fe5", lw=1.0, linestyle=(0, (4, 3))))


def _draw_mini_shap(ax, x: float, y: float, w: float, h: float) -> None:
    labels = ["Cu content", "Cold deformation", "Aging temp.", "Aging time", "Zr content", "..."]
    vals = [0.70, 0.55, 0.48, 0.38, 0.22, 0.05]
    for i, (lab, val) in enumerate(zip(labels, vals)):
        yy = y + h * (0.80 - i * 0.14)
        ax.text(x, yy, lab, transform=ax.transAxes, ha="right", va="center", fontsize=5.3)
        ax.add_patch(Rectangle((x + 0.01, yy - h * 0.035), w * val, h * 0.055,
                               transform=ax.transAxes, facecolor="#4f8ac7", edgecolor="white"))
    ax.text(x + w * .44, y + h * .94, "SHAP summary (example)", transform=ax.transAxes,
            fontsize=5.5, fontweight="bold", ha="center")
    ax.text(x + 0.0, y - h * .05, "Low impact", transform=ax.transAxes, fontsize=5.2)
    ax.text(x + w * .75, y - h * .05, "High impact", transform=ax.transAxes, fontsize=5.2)
    ax.add_patch(FancyArrowPatch((x + .01, y - h * .03), (x + w * .92, y - h * .03), transform=ax.transAxes,
                                 arrowstyle="simple", mutation_scale=10, color="#ff6a21"))


def _draw_mini_screening(ax, x: float, y: float, w: float, h: float) -> None:
    # Small 2-D screening schematic drawn with patches, not an imported image.
    quads = [
        (0, 0, "#2f72b7"), (0.5, 0, "#c7d8e8"),
        (0, 0.5, "#dceaf6"), (0.5, 0.5, "#f26b21"),
    ]
    for qx, qy, color in quads:
        ax.add_patch(Rectangle((x + qx * w, y + qy * h), w * 0.5, h * 0.5,
                               transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.95))
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="none", edgecolor="#1f4b7a", lw=0.8))
    ax.plot([x, x + w], [y + h * .5, y + h * .5], color="#1f4b7a", lw=0.7, transform=ax.transAxes)
    ax.plot([x + w * .5, x + w * .5], [y, y + h], color="#1f4b7a", lw=0.7, transform=ax.transAxes)
    ax.text(x + w * .82, y + h * .78, "★", transform=ax.transAxes, fontsize=14, color="white",
            ha="center", va="center")
    ax.text(x + w * .5, y + h + 0.03, "Screening map", transform=ax.transAxes,
            fontsize=5.8, fontweight="bold", ha="center")
    ax.text(x - 0.018, y + h * .95, "High", transform=ax.transAxes, fontsize=5.2, ha="right", va="center")
    ax.text(x - 0.018, y + h * .05, "Low", transform=ax.transAxes, fontsize=5.2, ha="right", va="center")
    ax.text(x, y - 0.03, "Low", transform=ax.transAxes, fontsize=5.2, ha="left")
    ax.text(x + w, y - 0.03, "High", transform=ax.transAxes, fontsize=5.2, ha="right")
    ax.text(x - 0.032, y + h / 2, "Hardness", rotation=90, transform=ax.transAxes, fontsize=5.2, ha="center")
    ax.text(x + w / 2, y - 0.055, "Conductivity", transform=ax.transAxes, fontsize=5.2, ha="center")


# --------------------------------------------------------------------------------------
# Fig. 1 functions
# --------------------------------------------------------------------------------------


def draw_fig1_workflow(ax) -> None:
    _panel_border(ax, "(a)")
    orange = "#f05a24"
    blue = "#0b4fa4"

    top_y, box_h = 0.59, 0.32
    bot_y, bot_h = 0.06, 0.42
    top_xs = [0.075, 0.29, 0.52, 0.755]
    bot_xs = [0.045, 0.27, 0.49, 0.71]
    top_ws = [0.16, 0.18, 0.18, 0.19]
    bot_ws = [0.17, 0.18, 0.18, 0.25]

    titles_top = ["Data collection", "Cleaning &\npreprocessing", "Feature extraction", "Data split strategy"]
    titles_bot = ["5-fold CV /\ngrid search", "Model training", "Evaluation", "SHAP & screening"]

    for i, (x, w, title) in enumerate(zip(top_xs, top_ws, titles_top), 1):
        _round_box(ax, x, top_y, w, box_h, i, title, orange)
    for i, (x, w, title) in enumerate(zip(bot_xs, bot_ws, titles_bot), 5):
        _round_box(ax, x, bot_y, w, bot_h, i, title, blue)

    # Arrows along workflow and down from stage 4 to stage 5.
    for a, b in [((0.25, 0.75), (0.275, 0.75)), ((0.485, 0.75), (0.51, 0.75)), ((0.705, 0.75), (0.743, 0.75)),
                 ((0.218, 0.29), (0.255, 0.29)), ((0.455, 0.29), (0.482, 0.29)), ((0.675, 0.29), (0.705, 0.29))]:
        _arrow(ax, a, b, blue)
    ax.plot([0.835, 0.835, 0.115, 0.115], [0.59, 0.52, 0.52, 0.49], color=blue, lw=3, transform=ax.transAxes)
    _arrow(ax, (0.115, 0.51), (0.115, 0.485), blue)

    # Stage 1: document/database icons.
    _draw_document_icon(ax, 0.098, 0.71, 0.065)
    _draw_database_icon(ax, 0.174, 0.705, 0.07)
    ax.text(0.157, 0.66, "Cu–Cr–X alloys", transform=ax.transAxes, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(0.157, 0.60, "From peer-reviewed literature\nand curated databases", transform=ax.transAxes,
            ha="center", va="center", fontsize=7.8)

    # Stage 2: cleaning symbol and list.
    _draw_sparkles(ax, 0.335, 0.74, 0.022)
    for yy in [0.77, 0.71, 0.65, 0.59]:
        ax.text(0.365, yy, "✓", color=orange, transform=ax.transAxes, fontsize=12, fontweight="bold")
    for text, yy in zip(["Duplicates removed", "Unit harmonization", "Missing value\nimputation", "Category encoding"], [0.775, 0.715, 0.655, 0.595]):
        ax.text(0.383, yy, text, transform=ax.transAxes, fontsize=7.3, va="center")
    for yy in [0.775, 0.685, 0.62]:
        ax.add_patch(Circle((0.304, yy), 0.0045, transform=ax.transAxes, facecolor="#5d94d1", edgecolor="#5d94d1"))

    # Stage 3: feature extraction.
    ax.text(0.615, 0.80, "Composition features", transform=ax.transAxes, fontsize=7.5, ha="center", fontweight="bold")
    elements = [("Cu", "#d9541e"), ("Cr", "#2f6fb1"), ("Zr", "#57905b"), ("Mg", "#7854a1"), ("Ti", "#666666")]
    for i, (el, color) in enumerate(elements):
        ax.text(0.55 + i * 0.035, 0.755, el, transform=ax.transAxes, ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="#555", lw=0.8))
    ax.plot([0.535, 0.695], [0.83, 0.83], transform=ax.transAxes, color="#f3c09d", lw=1)
    ax.plot([0.535, 0.695], [0.72, 0.72], transform=ax.transAxes, color="#f3c09d", lw=1)
    ax.text(0.615, 0.69, "Process variables", transform=ax.transAxes, fontsize=7.5, ha="center", fontweight="bold")
    # Simplified process icons.
    ax.add_patch(Polygon([[0.55, 0.625], [0.595, 0.625], [0.585, 0.64], [0.56, 0.64]], closed=True,
                         transform=ax.transAxes, facecolor="#c0c7d0", edgecolor="#333"))
    ax.add_patch(Circle((0.568, 0.665), 0.01, transform=ax.transAxes, facecolor="none", edgecolor="#333", lw=1))
    ax.arrow(0.57, 0.658, 0.025, -0.025, transform=ax.transAxes, head_width=0.006, color="#333", lw=1)
    ax.plot([0.63, 0.63], [0.625, 0.675], transform=ax.transAxes, color="#333", lw=1)
    ax.add_patch(Circle((0.63, 0.623), 0.008, transform=ax.transAxes, facecolor="#ff7f2a", edgecolor="#333", lw=0.8))
    ax.add_patch(Circle((0.688, 0.64), 0.018, transform=ax.transAxes, facecolor="none", edgecolor="#163c80", lw=1.6))
    ax.plot([0.688, 0.688], [0.64, 0.662], color="#163c80", transform=ax.transAxes, lw=1.2)
    ax.plot([0.688, 0.702], [0.64, 0.64], color="#163c80", transform=ax.transAxes, lw=1.2)
    ax.text(0.565, 0.585, "Cold\ndeformation", transform=ax.transAxes, fontsize=7, ha="center")
    ax.text(0.63, 0.585, "Aging\ntemperature", transform=ax.transAxes, fontsize=7, ha="center")
    ax.text(0.688, 0.585, "Aging\ntime", transform=ax.transAxes, fontsize=7, ha="center")

    # Stage 4: pie and legend.
    _draw_pie(ax, 0.81, 0.73, 0.095)
    for lab, col, yy in [("10%\nValidation", "#f07c18", 0.78), ("72%\nTraining", "#2678b8", 0.705), ("18%\nTesting", "#66a65c", 0.63)]:
        ax.add_patch(Circle((0.885, yy), 0.008, transform=ax.transAxes, facecolor=col, edgecolor=col))
        ax.text(0.897, yy, lab, transform=ax.transAxes, fontsize=8.5, va="center")
    ax.text(0.81, 0.59, "Stratified by target\ndistributions", transform=ax.transAxes, ha="center", fontsize=7.2)

    # Stage 5: CV/grid search.
    ax.text(0.13, 0.365, "Model tuning with 5-fold CV\nand grid search", transform=ax.transAxes, fontsize=7.5, ha="center")
    fold_x0 = 0.065
    for i in range(5):
        x = fold_x0 + i * 0.032
        ax.add_patch(FancyBboxPatch((x, 0.235), 0.022, 0.105, boxstyle="round,pad=0.002,rounding_size=0.004",
                                    transform=ax.transAxes, facecolor="#f4f7fa", edgecolor="#8c99a8", lw=0.8))
        ax.text(x + 0.011, 0.325, str(i + 1), transform=ax.transAxes, fontsize=6, ha="center", fontweight="bold")
        for j in range(4):
            col = blue if j == i % 4 else "#c9c9c9"
            ax.add_patch(Circle((x + 0.007 + (j % 2) * 0.009, 0.255 + (j // 2) * 0.027), 0.0045,
                                transform=ax.transAxes, facecolor=col, edgecolor=col))
    ax.add_patch(Circle((0.08, 0.195), 0.0055, transform=ax.transAxes, facecolor=blue, edgecolor=blue))
    ax.text(0.09, 0.195, "Validation fold", transform=ax.transAxes, fontsize=6.5, va="center")
    ax.add_patch(Circle((0.165, 0.195), 0.0055, transform=ax.transAxes, facecolor="#c9c9c9", edgecolor="#c9c9c9"))
    ax.text(0.175, 0.195, "Training folds", transform=ax.transAxes, fontsize=6.5, va="center")
    ax.add_patch(FancyBboxPatch((0.065, 0.095), 0.15, 0.07, boxstyle="round,pad=0.01,rounding_size=0.01",
                                transform=ax.transAxes, facecolor="#f6fbff", edgecolor="#2c71d0", lw=0.8))
    _tiny_bar_chart(ax, 0.082, 0.112, 0.04, 0.04, "#3c86d9")
    ax.text(0.16, 0.128, "Compare mean R$^2$\nacross folds", transform=ax.transAxes, fontsize=7, ha="center", va="center")

    # Stage 6: model training.
    ax.text(0.36, 0.405, "Six tree-ensemble regressors", transform=ax.transAxes, fontsize=7.5, ha="center")
    _draw_tree(ax, 0.29, 0.23, 0.14, 0.15)
    ax.text(0.36, 0.265, "...", transform=ax.transAxes, fontsize=10, ha="center")
    for i, model in enumerate(["ET", "RF", "GB", "XGB", "LGBM", "CAT"]):
        row, col = divmod(i, 3)
        ax.text(0.31 + col * 0.05, 0.18 - row * 0.052, model, transform=ax.transAxes, fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.28", fc="#ffffff", ec="#9a9a9a", lw=0.8))
    ax.text(0.36, 0.075, "Trained for each target\nseparately", transform=ax.transAxes, fontsize=7.5, ha="center")

    # Stage 7: parity plot.
    ax.text(0.565, 0.405, "R$^2$, RMSE, NRMSE, NMAE", transform=ax.transAxes, fontsize=7.5, ha="center")
    ax.plot([0.515, 0.615], [0.12, 0.37], transform=ax.transAxes, color="black", ls="--", lw=0.9)
    ax.plot([0.515, 0.62], [0.12, 0.12], transform=ax.transAxes, color="black", lw=0.8)
    ax.plot([0.515, 0.515], [0.12, 0.37], transform=ax.transAxes, color="black", lw=0.8)
    ax.annotate("", xy=(0.625, 0.12), xytext=(0.62, 0.12), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", lw=0.8))
    ax.annotate("", xy=(0.515, 0.375), xytext=(0.515, 0.37), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="-|>", lw=0.8))
    rng = np.random.default_rng(2)
    xs = np.linspace(0.535, 0.585, 6)
    ys = xs - 0.40 + rng.normal(0, 0.01, 6)
    ax.scatter(xs, ys, transform=ax.transAxes, s=26, c="#1f88d1", edgecolor="#00345f", zorder=3)
    xs2 = np.linspace(0.53, 0.60, 6)
    ys2 = xs2 - 0.43 + rng.normal(0, 0.008, 6)
    ax.scatter(xs2, ys2, transform=ax.transAxes, s=26, marker="D", facecolors="white", edgecolors="red", zorder=3)
    ax.text(0.54, 0.08, "Measured", transform=ax.transAxes, fontsize=7)
    ax.text(0.495, 0.22, "Predicted", rotation=90, transform=ax.transAxes, fontsize=7)
    ax.text(0.61, 0.21, "Hardness", transform=ax.transAxes, fontsize=7, va="center")
    ax.text(0.61, 0.165, "Conductivity", transform=ax.transAxes, fontsize=7, va="center")
    ax.scatter([0.595], [0.21], transform=ax.transAxes, s=22, c="#1f88d1", edgecolor="#00345f")
    ax.scatter([0.595], [0.165], transform=ax.transAxes, s=22, marker="D", facecolors="white", edgecolors="red")

    # Stage 8: SHAP and screening.
    ax.text(0.765, 0.405, "Feature effects\nIdentification of candidate\nregion with high hardness\nand high conductivity",
            transform=ax.transAxes, fontsize=5.9, ha="left", va="top")
    _draw_mini_shap(ax, 0.765, 0.125, 0.118, 0.165)
    ax.plot([0.895, 0.895], [0.13, 0.35], transform=ax.transAxes, color="#5d94d1", lw=1, linestyle="--")
    _draw_mini_screening(ax, 0.915, 0.155, 0.070, 0.195)


def draw_fig1_histograms(ax, si_dir: Path) -> None:
    _panel_border(ax, "(b)")
    ax.text(0.09, 0.94, "Target property distributions", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top")
    hardness, conductivity = read_processed_data(si_dir)
    hv = hardness["y_data"].dropna().to_numpy(float)
    ec = conductivity["y_data"].dropna().to_numpy(float)

    ax_h = ax.inset_axes([0.095, 0.28, 0.37, 0.54])
    ax_c = ax.inset_axes([0.56, 0.28, 0.37, 0.54])
    ax_h.hist(hv, bins=28, color="#ffd2d2", edgecolor="red", linewidth=1.0)
    ax_h.set_title("Hardness (HV$_{0.5}$)", color="red", fontsize=11, fontweight="bold")
    ax_h.set_xlabel("Hardness (HV$_{0.5}$)", color="red", fontsize=10)
    ax_h.set_ylabel("Count", color="red", fontsize=10)
    ax_h.tick_params(axis="both", labelsize=8)
    ax_h.tick_params(axis="y", colors="red")
    ax_h.grid(axis="y", linestyle="--", alpha=0.35)
    for sp in ax_h.spines.values():
        sp.set_linewidth(0.8)

    ax_c.hist(ec, bins=28, color="#d7e6ff", edgecolor="#0b55ff", linewidth=1.0)
    ax_c.set_title("Electrical conductivity (%IACS)", color="#0055ff", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Electrical conductivity (%IACS)", color="#0055ff", fontsize=10)
    ax_c.set_ylabel("Count", color="#0055ff", fontsize=10)
    ax_c.tick_params(axis="both", labelsize=8)
    ax_c.tick_params(axis="y", colors="#0055ff")
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)
    for sp in ax_c.spines.values():
        sp.set_linewidth(0.8)

    ax.text(0.5, 0.09, "Separated histograms show the wide property coverage of hardness and electrical conductivity.",
            transform=ax.transAxes, fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#808080", lw=0.8))


def draw_fig1_correlation(ax) -> None:
    _panel_border(ax, "(c)")
    ax.text(0.12, 0.94, "Pearson correlation matrix", transform=ax.transAxes,
            fontsize=12.4, fontweight="bold", va="top", color="#111111")
    # Wider, flatter heatmap block to match the uploaded panel proportion.
    hm_ax = ax.inset_axes([0.080, 0.195, 0.770, 0.615])
    im = hm_ax.imshow(FIG1_CORR, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    hm_ax.set_xticks(range(len(FIG1_CORR_LABELS)))
    hm_ax.set_yticks(range(len(FIG1_CORR_LABELS)))
    hm_ax.set_xticklabels(FIG1_CORR_LABELS, fontsize=6.3, linespacing=0.92, fontweight="bold", color="#111111")
    hm_ax.set_yticklabels(FIG1_CORR_LABELS, fontsize=6.3, linespacing=0.92, fontweight="bold", color="#111111")
    hm_ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0, pad=2)
    for i in range(FIG1_CORR.shape[0]):
        for j in range(FIG1_CORR.shape[1]):
            val = FIG1_CORR[i, j]
            color = "white" if abs(val) >= 0.55 else "#111111"
            hm_ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.7,
                       color=color, fontweight="bold")
    for edge in hm_ax.spines.values():
        edge.set_visible(False)
    hm_ax.set_xticks(np.arange(-.5, len(FIG1_CORR_LABELS), 1), minor=True)
    hm_ax.set_yticks(np.arange(-.5, len(FIG1_CORR_LABELS), 1), minor=True)
    hm_ax.grid(which="minor", color="white", linestyle="-", linewidth=1.25)
    hm_ax.tick_params(which="minor", bottom=False, left=False)
    cax = ax.inset_axes([0.875, 0.285, 0.026, 0.45])
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7.0, colors="#111111")
    ax.text(0.888, 0.75, "Pearson r", transform=ax.transAxes, fontsize=7.2, ha="center", va="bottom", color="#111111", fontweight="bold")
    ax.text(0.5, 0.09, "Positive (red) indicates positive correlation; negative (blue) indicates negative correlation.",
            transform=ax.transAxes, fontsize=8.8, ha="center", color="#111111")


def build_fig1(si_dir: Path, out_path: Path) -> Path:
    fig = plt.figure(figsize=(14.48, 10.86), dpi=100)
    draw_fig1_workflow(fig.add_axes([0.02, 0.40, 0.96, 0.57]))
    draw_fig1_histograms(fig.add_axes([0.02, 0.03, 0.46, 0.35]), si_dir)
    draw_fig1_correlation(fig.add_axes([0.50, 0.03, 0.48, 0.35]))
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Fig. 2 and Fig. 3 functions
# --------------------------------------------------------------------------------------


def _metric_values(metrics: pd.DataFrame, target: str, model: str, split: str = "test") -> dict[str, float]:
    row = metrics[(metrics["target"] == target) & (metrics["model"] == model)].iloc[0]
    return {
        "r2": float(row[f"{split}_r2_percent"]),
        "nrmse": float(row[f"{split}_nrmse_percent"]),
        "nmae": float(row[f"{split}_nmae_percent"]),
    }


def _annotate_bars(ax, bars, pad=1.4, fontsize=8) -> None:
    y_top = ax.get_ylim()[1]
    for bar in bars:
        h = float(bar.get_height())
        y = min(h + pad, y_top - 1.8)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold", clip_on=True)


def draw_metric_bars(ax, metrics: pd.DataFrame, target: str, models: Sequence[str], split: str,
                     title: str, xlabel: str = "Different models", title_loc: str = "center") -> None:
    values = [_metric_values(metrics, target, model, split=split) for model in models]
    r2 = [v["r2"] for v in values]
    nrmse = [v["nrmse"] for v in values]
    nmae = [v["nmae"] for v in values]
    x = np.arange(len(models))
    width = 0.24
    if split == "test":
        colors = ("#2a8be8", "#ff8c1a", "#2ca02c")
    else:
        colors = ("#f7941d", "#ef3b3b", "#8a2be2")
    b1 = ax.bar(x - width, r2, width, label="R$^2$", color=colors[0], edgecolor="#222222", linewidth=0.5)
    b2 = ax.bar(x, nrmse, width, label="NRMSE", color=colors[1], edgecolor="#222222", linewidth=0.5)
    b3 = ax.bar(x + width, nmae, width, label="NMAE", color=colors[2], edgecolor="#222222", linewidth=0.5)
    title_size = 14 if split == "test" else (10.5 if "Electrical-conductivity validation metrics" in title else 12)
    ax.set_title(title, fontsize=title_size, fontweight="bold", loc=title_loc)
    ax.set_ylabel("Value (%)", fontsize=12 if split == "test" else 9, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11 if split == "test" else 8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 108)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    if split == "test":
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.985), borderaxespad=0.0)
    else:
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.985), borderaxespad=0.0, handlelength=1.2)
    _annotate_bars(ax, b1, pad=1.1, fontsize=8 if split == "test" else 7)
    _annotate_bars(ax, b2, pad=0.42, fontsize=8 if split == "test" else 7)
    _annotate_bars(ax, b3, pad=0.42, fontsize=8 if split == "test" else 7)


def build_fig2(si_dir: Path, out_path: Path) -> Path:
    metrics = read_metrics(si_dir)
    # Fixed canvas size matches the manuscript Fig. 2 image (1536 x 1024 px).
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    axes = [
        fig.add_axes([0.055, 0.14, 0.440, 0.755]),
        fig.add_axes([0.540, 0.14, 0.443, 0.755]),
    ]
    draw_metric_bars(axes[0], metrics, "hardness", MODEL_ORDER, "test", "(a) Hardness prediction (HV)")
    draw_metric_bars(axes[1], metrics, "conductivity", MODEL_ORDER, "test", "(b) Electrical conductivity prediction (% IACS)")
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    return out_path


def draw_validation_scatter(ax, predictions: pd.DataFrame, metrics: pd.DataFrame, target: str, model: str,
                            title: str) -> None:
    df = predictions[(predictions["target"] == target) & (predictions["model"] == model) & (predictions["split_norm"] == "val")]
    row = metrics[(metrics["target"] == target) & (metrics["model"] == model)].iloc[0]
    r2 = float(row["val_r2"])
    actual = df["actual"].to_numpy(float)
    predicted = df["predicted"].to_numpy(float)
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    if target == "hardness":
        lo = min(50, lo - 6)
        hi = max(350, hi + 6)
    else:
        lo = min(0, lo - 3)
        hi = max(100, hi + 3)
    ax.scatter(actual, predicted, s=16, color=SCATTER_COLOR[target], edgecolor="#145a8d",
               linewidth=0.35, alpha=0.88, label=f"{TARGET_LABEL[target]} Prediction (R$^2$={r2:.3f})")
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.1, label="Perfect Prediction")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Actual {TARGET_LABEL[target]} ({UNIT_LABEL[target]})", fontsize=9)
    ax.set_ylabel(f"Predicted {TARGET_LABEL[target]} ({UNIT_LABEL[target]})", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=7, loc="upper left")


def build_fig3(si_dir: Path, out_path: Path) -> Path:
    metrics = read_metrics(si_dir)
    preds = read_predictions(si_dir)
    # Fixed canvas size matches the manuscript Fig. 3 image (1672 x 941 px).
    fig = plt.figure(figsize=(16.72, 9.41), dpi=100)
    gs = gridspec.GridSpec(2, 4, figure=fig, left=0.040, right=0.975, top=0.965, bottom=0.080,
                           wspace=0.24, hspace=0.31)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(2)])
    for ax, model, label in zip(axes[0, :3], TOP_MODELS["hardness"], ["(a) LGBM", "(b) RF", "(c) XGB"]):
        draw_validation_scatter(ax, preds, metrics, "hardness", model, label)
    draw_metric_bars(axes[0, 3], metrics, "hardness", TOP_MODELS["hardness"], "val", "(d) Hardness validation metrics",
                     xlabel="Top three models", title_loc="left")
    for ax, model, label in zip(axes[1, :3], TOP_MODELS["conductivity"], ["(e) LGBM", "(f) XGB", "(g) GB"]):
        draw_validation_scatter(ax, preds, metrics, "conductivity", model, label)
    draw_metric_bars(axes[1, 3], metrics, "conductivity", TOP_MODELS["conductivity"], "val",
                     "(h) Electrical-conductivity validation metrics", xlabel="Top three models", title_loc="left")
    fig.savefig(out_path, dpi=100, facecolor="white")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Fig. 4 functions
# --------------------------------------------------------------------------------------


def _feature_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    label = re.sub(r"^(num__|cat__)", "", feature)
    label = label.replace("_", " ")
    return label


def _top_features_for_fig4(si_dir: Path, target: str, n: int = 9) -> pd.DataFrame:
    imp = read_shap_importance(si_dir)
    return imp[imp["target"] == target].head(n).copy()


def draw_shap_importance_bar(ax, si_dir: Path, target: str, title: str, panel: str | None = None) -> None:
    top = _top_features_for_fig4(si_dir, target, 9)
    labels = [_feature_label(f) for f in top["feature"]]
    vals = top["mean_abs_shap"].to_numpy(float)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=BAR_COLOR[target], edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    xmax = 14 if target == "hardness" else 16.5
    ax.set_xlim(0, xmax)
    if panel:
        ax.text(-0.38, 1.02, panel, transform=ax.transAxes, fontsize=20, fontweight="bold")


def _load_modeling_state(si_dir: Path) -> dict:
    with open(tables_dir(si_dir) / "modeling_state.pkl", "rb") as f:
        return pickle.load(f)


def _compute_shap_values(si_dir: Path, target: str):
    import shap  # imported lazily so Fig. 1-Fig. 3 can run without SHAP startup cost
    state = _load_modeling_state(si_dir)[target]
    model = state["models"]["LGBM"]["estimator"]
    X = state["X_val"]
    feature_names = list(state["feature_names"])
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X))
    return X, values, feature_names


def draw_shap_summary(ax, si_dir: Path, target: str, title: str, panel: str | None = None) -> None:
    X, shap_values, feature_names = _compute_shap_values(si_dir, target)
    top = _top_features_for_fig4(si_dir, target, 9)
    selected = [f for f in top["feature"] if f in feature_names]
    rng = np.random.default_rng(123)
    for pos, feature in enumerate(selected[::-1]):
        idx = feature_names.index(feature)
        xvals = shap_values[:, idx]
        cvals = X[:, idx].astype(float)
        # Robust color normalization improves readability while preserving feature order/sign.
        if np.nanmax(cvals) > np.nanmin(cvals):
            lo, hi = np.nanpercentile(cvals, [2, 98])
            norm = Normalize(vmin=lo, vmax=hi)
        else:
            norm = Normalize(vmin=0, vmax=1)
        jitter = rng.normal(0, 0.055, len(xvals))
        ax.scatter(xvals, np.full_like(xvals, pos, dtype=float) + jitter, c=cvals, cmap=SHAP_CMAP,
                   norm=norm, s=13, alpha=0.95, edgecolor="none")
    ax.axvline(0, color="#1d4ed8", lw=1.3)
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels([_feature_label(f) for f in selected[::-1]], fontsize=9.2)
    ax.set_xlabel(f"SHAP value ({UNIT_LABEL[target]})", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    if target == "hardness":
        ax.set_xlim(-30, 50)
    else:
        ax.set_xlim(-42, 18)
    if panel:
        ax.text(-0.38, 1.02, panel, transform=ax.transAxes, fontsize=20, fontweight="bold")
    # Feature-value colorbar made from the same colormap used for the points.
    sm = plt.cm.ScalarMappable(cmap=SHAP_CMAP, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.06)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", rotation=90, fontsize=11)
    cbar.outline.set_visible(False)


def draw_screening_map(ax, si_dir: Path, panel: str | None = None) -> None:
    screening = read_screening_map(si_dir)
    x = screening["predicted_hardness"].to_numpy(float)
    y = screening["predicted_conductivity"].to_numpy(float)
    score = screening["PHCI"].to_numpy(float)
    sc = ax.scatter(x, y, c=score, cmap="plasma", s=8, alpha=0.95, edgecolor="none")
    ax.set_title("Hardness–conductivity screening map", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted hardness (HV)", fontsize=12)
    ax.set_ylabel("Predicted conductivity (% IACS)", fontsize=12)
    ax.set_xlim(72, 278)
    ax.set_ylim(0, 120)
    ax.grid(True, linestyle="--", alpha=0.35)
    rect = Rectangle((180, 68), 88, 27, fill=False, edgecolor="black", linewidth=1.5, linestyle=(0, (4, 3)))
    ax.add_patch(rect)
    ax.text(224, 80, "High-hardness /\nhigh-conductivity candidates", ha="center", va="center",
            fontsize=12, fontweight="bold")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("Candidate screening score", fontsize=11)
    cbar.set_ticks([score.min(), score.max()])
    cbar.set_ticklabels(["Low", "High"])
    if panel:
        ax.text(-0.10, 0.96, panel, transform=ax.transAxes, fontsize=20, fontweight="bold")


def build_fig4(si_dir: Path, out_path: Path) -> Path:
    fig = plt.figure(figsize=(15.36, 10.24), dpi=100)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.00, 1.18, 1.24], hspace=0.58, wspace=0.50)
    fig.subplots_adjust(left=0.13, right=0.94, top=0.95, bottom=0.08)
    draw_shap_importance_bar(fig.add_subplot(gs[0, 0]), si_dir, "hardness",
                             "Hardness model: mean |SHAP value| (top features)", "(a)")
    draw_shap_importance_bar(fig.add_subplot(gs[0, 1]), si_dir, "conductivity",
                             "Electrical-conductivity model: mean |SHAP value| (top features)", "(b)")
    draw_shap_summary(fig.add_subplot(gs[1, 0]), si_dir, "hardness", "Hardness model: SHAP summary", "(c)")
    draw_shap_summary(fig.add_subplot(gs[1, 1]), si_dir, "conductivity", "Electrical-conductivity model: SHAP summary", "(d)")
    draw_screening_map(fig.add_subplot(gs[2, :]), si_dir, "(e)")
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path


