import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory
from PIL import Image


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_NAME = "Cascade Configuration Figure.png"

RESULTS = [
    ("Single pass", 16, 1, 73.68, 574.8, 0.5315),
    ("Separate", 8, 2, 43.36, 641.6, 0.5103),
    ("Separate", 4, 4, 62.80, 632.9, 0.5282),
    ("Separate", 2, 8, 109.83, 880.3, 0.5055),
    ("Separate", 1, 16, 140.88, 875.6, 0.4277),
    ("Shared", 8, 2, 14.36, 489.3, 0.6054),
    ("Shared", 4, 4, 34.91, 522.1, 0.5586),
    ("Shared", 2, 8, 184.61, 1225.9, 0.3963),
    ("Shared", 1, 16, 222.42, 924.2, 0.4334),
]

CD_COLOR = "#e34948"
EMD_COLOR = "#008300"
IOU_COLOR = "#2a78d6"

EMD_HATCH_COLOR = "#8cc78c"
IOU_HATCH_COLOR = "#9fc2ed"

CD_HATCH = None
EMD_HATCH = "////"
IOU_HATCH = "---"

CD_LIMIT = 250
CD_STEP = 50
EMD_LIMIT = 1250
EMD_STEP = 250
IOU_LIMIT = 1.0
IOU_STEP = 0.2

FIG_WIDTH = 5.8
FIG_HEIGHT = 2.6
DPI = 600

BACKGROUND_COLOR = "#ffffff"
INK_COLOR = "#000000"
BAND_COLOR = "#ffeb99"

GROUP_GAP = 0.6
BAR_WIDTH = 0.26
BAR_GAP = 0.02
BAND_HALF_WIDTH = 0.5
HATCH_LINE_WIDTH = 0.8
AXIS_LINE_WIDTH = 0.6
EMD_AXIS_OFFSET = 34
GROUP_LABEL_OFFSET = -16
TICK_FONT_SIZE = 7
AXIS_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8


def compute_positions():
    positions = []
    x = 0.0
    for i, result in enumerate(RESULTS):
        if i > 0 and result[0] != RESULTS[i - 1][0]:
            x += GROUP_GAP
        positions.append(x)
        x += 1.0
    return np.array(positions)


def style_axis(ax, side, color, label, limit, step):
    ax.set_ylim(0, limit)
    ax.set_yticks(np.arange(0, limit + step / 2, step))
    ax.set_ylabel(label, color=color, fontsize=AXIS_FONT_SIZE, fontweight="bold")
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color(color)
    ax.spines[side].set_linewidth(AXIS_LINE_WIDTH)
    ax.tick_params(axis="y", colors=color, labelsize=TICK_FONT_SIZE, width=AXIS_LINE_WIDTH)


def draw_group_labels(ax, positions):
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    groups = []
    for (mode, *_), x in zip(RESULTS, positions):
        if groups and groups[-1][0] == mode:
            groups[-1][2] = x
        else:
            groups.append([mode, x, x])
    for mode, start, end in groups:
        ax.annotate(mode, xy=((start + end) / 2, 0), xycoords=transform, xytext=(0, GROUP_LABEL_OFFSET), textcoords="offset points", ha="center", va="top", color=INK_COLOR, fontsize=AXIS_FONT_SIZE)


def main():
    plt.rcParams["hatch.linewidth"] = HATCH_LINE_WIDTH

    positions = compute_positions()
    cd_values = [result[3] for result in RESULTS]
    emd_values = [result[4] for result in RESULTS]
    iou_values = [result[5] for result in RESULTS]

    fig, cd_ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, layout="constrained")
    emd_ax = cd_ax.twinx()
    iou_ax = cd_ax.twinx()
    fig.set_facecolor(BACKGROUND_COLOR)
    cd_ax.set_facecolor(BACKGROUND_COLOR)

    for ax in (cd_ax, emd_ax, iou_ax):
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)

    emd_ax.spines["left"].set_position(("outward", EMD_AXIS_OFFSET))
    emd_ax.yaxis.set_label_position("left")
    emd_ax.yaxis.set_ticks_position("left")

    cd_bars = cd_ax.bar(positions - BAR_WIDTH, cd_values, BAR_WIDTH - BAR_GAP, color=CD_COLOR, hatch=CD_HATCH, linewidth=0, label="CD")
    emd_bars = emd_ax.bar(positions, emd_values, BAR_WIDTH - BAR_GAP, color=EMD_COLOR, hatch=EMD_HATCH, edgecolor=EMD_HATCH_COLOR, linewidth=0, label="EMD")
    iou_bars = iou_ax.bar(positions + BAR_WIDTH, iou_values, BAR_WIDTH - BAR_GAP, color=IOU_COLOR, hatch=IOU_HATCH, edgecolor=IOU_HATCH_COLOR, linewidth=0, label="IoU")

    best_index = np.argmax(iou_values)
    cd_ax.axvspan(positions[best_index] - BAND_HALF_WIDTH, positions[best_index] + BAND_HALF_WIDTH, color=BAND_COLOR, linewidth=0, zorder=0)

    style_axis(cd_ax, "left", CD_COLOR, "CD", CD_LIMIT, CD_STEP)
    style_axis(emd_ax, "left", EMD_COLOR, "EMD", EMD_LIMIT, EMD_STEP)
    style_axis(iou_ax, "right", IOU_COLOR, "IoU", IOU_LIMIT, IOU_STEP)

    cd_ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
    cd_ax.set_xticks(positions)
    cd_ax.set_xticklabels(["{} × {}".format(prims, cascades) for _, prims, cascades, *_ in RESULTS])
    cd_ax.get_xticklabels()[best_index].set_fontweight("bold")
    cd_ax.tick_params(axis="x", colors=INK_COLOR, labelsize=TICK_FONT_SIZE, width=AXIS_LINE_WIDTH)
    cd_ax.spines["bottom"].set_color(INK_COLOR)
    cd_ax.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)
    draw_group_labels(cd_ax, positions)

    fig.legend(handles=[cd_bars, emd_bars, iou_bars], loc="outside upper center", ncol=3, frameon=False, fontsize=LEGEND_FONT_SIZE, labelcolor=INK_COLOR, handlelength=2.2, handleheight=1.1, columnspacing=1.6)

    out_path = os.path.join(_THIS_DIR, OUTPUT_NAME)
    fig.canvas.draw()
    Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).save(out_path, dpi=(DPI, DPI))
    plt.close(fig)
    print("Wrote: {}".format(out_path))


if __name__ == "__main__":
    main()
