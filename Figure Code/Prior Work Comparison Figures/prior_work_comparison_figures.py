import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SEEN_RESULTS = [
    ("CSG-CRN", 0.534, 26.27, 0.6687),
    ("SuperDec", 0.622, 31.84, 0.5561),
    ("CAPRI-Net", 0.724, 31.40, 0.5993),
    ("BSP-Net", 0.761, 32.77, 0.5913),
    ("UCSG-Net", 2.177, 46.55, 0.5217),
]

UNSEEN_RESULTS = [
    ("CSG-CRN", 1.393, 42.03, 0.3435),
    ("SuperDec", 1.211, 42.23, 0.3317),
    ("CAPRI-Net", 2.687, 54.89, 0.3036),
    ("BSP-Net", 3.703, 73.88, 0.2407),
    ("UCSG-Net", 5.101, 73.49, 0.2350),
]

CHARTS = [
    (SEEN_RESULTS, "Prior Work Seen Comparison.png", (2.5, 0.5), (50, 10), (1.0, 0.2)),
    (UNSEEN_RESULTS, "Prior Work Unseen Comparison.png", (6, 1), (90, 15), (0.6, 0.1)),
]

CD_COLOR = "#e34948"
EMD_COLOR = "#008300"
IOU_COLOR = "#2a78d6"

EMD_HATCH_COLOR = "#8cc78c"
IOU_HATCH_COLOR = "#9fc2ed"

CD_HATCH = None
EMD_HATCH = "////"
IOU_HATCH = "---"

FIG_WIDTH = 5.8
FIG_HEIGHT = 2.6
DPI = 600

BACKGROUND_COLOR = "#ffffff"
INK_COLOR = "#000000"
BAND_COLOR = "#ffeb99"

BAR_WIDTH = 0.26
BAR_GAP = 0.02
BAND_PADDING = 0.05
HATCH_LINE_WIDTH = 0.8
AXIS_LINE_WIDTH = 0.6
OUTER_AXIS_GAP = 8
TICK_FONT_SIZE = 7
AXIS_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8


def style_axis(ax, side, color, label, limit, step):
    ax.set_ylim(0, limit)
    ax.set_yticks(np.arange(0, limit + step / 2, step))
    ax.set_ylabel(label, color=color, fontsize=AXIS_FONT_SIZE, fontweight="bold")
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color(color)
    ax.spines[side].set_linewidth(AXIS_LINE_WIDTH)
    ax.tick_params(axis="y", colors=color, labelsize=TICK_FONT_SIZE, width=AXIS_LINE_WIDTH)


def place_outer_axis(fig, inner_ax, outer_ax):
    renderer = fig.canvas.get_renderer()
    label_left = inner_ax.yaxis.label.get_window_extent(renderer).x0
    axes_left = inner_ax.get_window_extent(renderer).x0
    offset = (axes_left - label_left) * 72 / fig.dpi + OUTER_AXIS_GAP
    outer_ax.spines["left"].set_position(("outward", offset))


def highlight_span(ax, left, right):
    ax.axvspan(left, right, color=BAND_COLOR, linewidth=0, zorder=0)


def plot_comparison(results, output_name, cd_range, emd_range, iou_range):
    positions = np.arange(len(results), dtype=float)
    cd_values = [result[1] for result in results]
    emd_values = [result[2] for result in results]
    iou_values = [result[3] for result in results]

    fig, emd_ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, layout="constrained")
    cd_ax = emd_ax.twinx()
    iou_ax = emd_ax.twinx()
    fig.set_facecolor(BACKGROUND_COLOR)
    emd_ax.set_facecolor(BACKGROUND_COLOR)

    for ax in (emd_ax, cd_ax, iou_ax):
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)

    cd_ax.yaxis.set_label_position("left")
    cd_ax.yaxis.set_ticks_position("left")

    cd_bars = cd_ax.bar(positions - BAR_WIDTH, cd_values, BAR_WIDTH - BAR_GAP, color=CD_COLOR, hatch=CD_HATCH, linewidth=0, label="CD")
    emd_bars = emd_ax.bar(positions, emd_values, BAR_WIDTH - BAR_GAP, color=EMD_COLOR, hatch=EMD_HATCH, edgecolor=EMD_HATCH_COLOR, linewidth=0, label="EMD")
    iou_bars = iou_ax.bar(positions + BAR_WIDTH, iou_values, BAR_WIDTH - BAR_GAP, color=IOU_COLOR, hatch=IOU_HATCH, edgecolor=IOU_HATCH_COLOR, linewidth=0, label="IoU")

    style_axis(cd_ax, "left", CD_COLOR, "CD", *cd_range)
    style_axis(emd_ax, "left", EMD_COLOR, "EMD", *emd_range)
    style_axis(iou_ax, "right", IOU_COLOR, "IoU", *iou_range)

    emd_ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    emd_ax.set_xticks(positions)
    emd_ax.set_xticklabels([result[0] for result in results])
    emd_ax.tick_params(axis="x", colors=INK_COLOR, labelsize=TICK_FONT_SIZE, width=AXIS_LINE_WIDTH)
    emd_ax.spines["bottom"].set_color(INK_COLOR)
    emd_ax.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)

    best_cd = np.argmin(cd_values)
    best_emd = np.argmin(emd_values)
    best_iou = np.argmax(iou_values)
    highlight_span(emd_ax, positions[best_cd] - 1.5 * BAR_WIDTH - BAND_PADDING, positions[best_cd] - 0.5 * BAR_WIDTH)
    highlight_span(emd_ax, positions[best_emd] - 0.5 * BAR_WIDTH, positions[best_emd] + 0.5 * BAR_WIDTH)
    highlight_span(emd_ax, positions[best_iou] + 0.5 * BAR_WIDTH, positions[best_iou] + 1.5 * BAR_WIDTH + BAND_PADDING)
    for best in {best_cd, best_emd, best_iou}:
        emd_ax.get_xticklabels()[best].set_fontweight("bold")

    fig.legend(handles=[cd_bars, emd_bars, iou_bars], loc="outside upper center", ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE, labelcolor=INK_COLOR, handlelength=2.2, handleheight=1.1, columnspacing=1.6)

    fig.canvas.draw()
    place_outer_axis(fig, emd_ax, cd_ax)

    out_path = os.path.join(_THIS_DIR, output_name)
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).save(out_path, dpi=(DPI, DPI))
    plt.close(fig)
    print("Wrote: {}".format(out_path))


def main():
    plt.rcParams["hatch.linewidth"] = HATCH_LINE_WIDTH
    for results, output_name, cd_range, emd_range, iou_range in CHARTS:
        plot_comparison(results, output_name, cd_range, emd_range, iou_range)


if __name__ == "__main__":
    main()
