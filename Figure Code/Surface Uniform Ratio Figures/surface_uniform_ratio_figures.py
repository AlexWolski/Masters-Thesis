import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation
from PIL import Image


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SEEDS = [7, 8, 9]
CSV_NAME_FORMAT = "{} Seed {} final_results.csv"

SERIES = [
    ("L1", "#2a78d6", "X", ":"),
    ("LOG", "#eb6834", "o", "--"),
]

METRICS = [
    ("test_iou", "Surface Uniform Ratio IoU.png", "IoU"),
    ("test_chamfer", "Surface Uniform Ratio Chamfer.png", "CD"),
    ("test_emd", "Surface Uniform Ratio EMD.png", "EMD"),
]

FIG_WIDTH = 1.92
FIG_HEIGHT = 1.77
DPI = 600

BACKGROUND_COLOR = "#ffffff"
INK_COLOR = "#000000"

LINE_WIDTH = 1.2
TREND_LINE_WIDTH = 1
TREND_ALPHA = 0.8
MARKER_SIZE = 3.5
AXIS_LINE_WIDTH = 0.6
TICK_FONT_SIZE = 6
AXIS_FONT_SIZE = 6.5
LEGEND_FONT_SIZE = 6
LEGEND_OFFSET = 0.41


def read_series(csv_path, column):
    with open(csv_path) as f:
        rows = sorted(csv.DictReader(f), key=lambda row: float(row["surface_uniform_ratio"]))
    ratios = [float(row["surface_uniform_ratio"]) for row in rows]
    values = [float(row[column]) for row in rows]
    return ratios, values


def read_mean_series(name, column):
    seed_values = []
    for seed in SEEDS:
        ratios, values = read_series(os.path.join(_THIS_DIR, CSV_NAME_FORMAT.format(name, seed)), column)
        seed_values.append(values)
    return ratios, np.mean(seed_values, axis=0)


def plot_metric(column, output_name, y_label):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    fig.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)

    for name, color, marker, trend_style in SERIES:
        ratios, values = read_mean_series(name, column)
        ax.plot(ratios, values, color=color, linewidth=LINE_WIDTH, marker=marker, markersize=MARKER_SIZE, markeredgewidth=0, label=name, zorder=3)

        slope, intercept = np.polyfit(ratios, values, 1)
        trend_x = np.array([ratios[0], ratios[-1]])
        ax.plot(trend_x, slope * trend_x + intercept, color=color, linewidth=TREND_LINE_WIDTH, linestyle=trend_style, alpha=TREND_ALPHA, label="{} trend".format(name), zorder=2)

    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([i / 5 for i in range(6)])
    ax.set_xlabel("Surface-Uniform Ratio", color=INK_COLOR, fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel(y_label, color=INK_COLOR, fontsize=AXIS_FONT_SIZE)

    ax.tick_params(color=INK_COLOR, labelcolor=INK_COLOR, labelsize=TICK_FONT_SIZE, width=AXIS_LINE_WIDTH)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_COLOR)
        ax.spines[side].set_linewidth(AXIS_LINE_WIDTH)

    legend_transform = ax.transAxes + ScaledTranslation(0, -LEGEND_OFFSET, fig.dpi_scale_trans)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0), bbox_transform=legend_transform, ncol=2, fontsize=LEGEND_FONT_SIZE, labelcolor=INK_COLOR, handlelength=2.2, columnspacing=1, borderaxespad=0)

    fig.tight_layout(pad=0.2)
    fig.tight_layout(pad=0.2)
    out_path = os.path.join(_THIS_DIR, output_name)
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).save(out_path, dpi=(DPI, DPI))
    plt.close(fig)
    print("Wrote: {}".format(out_path))


def main():
    for column, output_name, y_label in METRICS:
        plot_metric(column, output_name, y_label)


if __name__ == "__main__":
    main()
