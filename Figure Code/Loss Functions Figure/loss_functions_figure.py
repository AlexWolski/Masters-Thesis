import os

import matplotlib.pyplot as plt
import numpy as np


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_NAME = "Loss Functions Figure.png"

# Loss constants, matching the defaults in losses/reconstruction_loss.py of the CSG-CRN repository.
LOG_X_OFFSET = 0.01
LOG_Y_OFFSET = 1
LOG_MULTIPLE = 5
SIG_SIGMOID_EXPONENT = -50

MAX_ERROR = 1.0
MAX_LOSS = 1.0
NUM_POINTS = 2001

FIG_SIZE = 5
DPI = 200

SURFACE_COLOR = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#000000"

LINE_WIDTH = 2
GRID_WIDTH = 0.5
LABEL_FONT_SIZE = 10
TICK_FONT_SIZE = 9

LABEL_X = 0.5
LABEL_Y_OFFSET = 0.025


def l1_loss(error):
    return error


def mse_loss(error):
    return error ** 2


def log_loss(error):
    return np.log(error + LOG_X_OFFSET) / LOG_MULTIPLE + LOG_Y_OFFSET


def sig_loss(error):
    sigmoid = 1 / (1 + np.exp(SIG_SIGMOID_EXPONENT * error))
    return ((2 * sigmoid - 1) + error) / 2


LOSSES = [
    ("L1", l1_loss, "#2a78d6"),
    ("LOG", log_loss, "#1baf7a"),
    ("MSE", mse_loss, "#eb6834"),
    ("SIG", sig_loss, "#eda100"),
]


def main():
    error = np.linspace(0, MAX_ERROR, NUM_POINTS)

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE), dpi=DPI, layout="constrained")
    fig.set_facecolor(SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)

    for name, loss_func, color in LOSSES:
        ax.plot(error, loss_func(error), color=color, linewidth=LINE_WIDTH, solid_joinstyle="round", solid_capstyle="round", label=name)
        ax.text(LABEL_X, loss_func(LABEL_X) + LABEL_Y_OFFSET, name, ha="center", va="bottom", color=INK_PRIMARY, fontsize=LABEL_FONT_SIZE, fontweight="bold")

    ax.set_xlim(0, MAX_ERROR)
    ax.set_ylim(0, MAX_LOSS)
    ax.set_xlabel("Error", color=INK_PRIMARY, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Loss", color=INK_PRIMARY, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(color=AXIS_COLOR, labelcolor=INK_SECONDARY, labelsize=TICK_FONT_SIZE)

    ax.grid(color=GRID_COLOR, linewidth=GRID_WIDTH)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)

    ax.legend(loc="upper left", frameon=False, fontsize=LABEL_FONT_SIZE, labelcolor=INK_PRIMARY)

    out_path = os.path.join(_THIS_DIR, OUTPUT_NAME)
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE_COLOR)
    print("Wrote: {}".format(out_path))


if __name__ == "__main__":
    main()
