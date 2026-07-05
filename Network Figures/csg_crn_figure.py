import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTNN_REL = os.path.join(".", "PlotNeuralNet")
sys.path.append(os.path.join(_THIS_DIR, PLOTNN_REL))

from pycore.tikzeng import to_head, to_cor, to_begin, to_end, to_generate


# ---------------------------------------------------------------------------
# Network hyper-parameters depicted in the figure
# ---------------------------------------------------------------------------
NUM_INPUT_POINTS = 2048
NUM_FEATURES = 6
ENCODER_CONV = [64, 128, 1024]
GLOBAL_FEAT = 1024
DECODER_LAYERS = [1024, 512, 128, 64]


# ---------------------------------------------------------------------------
# Extra colours + black edges (added after the stock to_cor() palette)
# ---------------------------------------------------------------------------
def to_colors():
    return r"""
\def\TNetColor{rgb:orange,6;yellow,4}
\def\MaxPoolColor{rgb:orange,8;red,3}
\def\DecoderColor{rgb:blue,5;white,5}
\def\ActivationColor{rgb:red,5;blue,5}
\def\edgecolor{black}
"""


# ---------------------------------------------------------------------------
# Box / label / connection helpers (in the spirit of pycore.tikzeng)
# ---------------------------------------------------------------------------
def to_box(name, fill, to, offset, width=2, height=3, depth=3,
           caption=" ", xlabel="", ylabel="", zlabel=""):
    cap = caption if caption == " " else "{" + caption + "}"
    lines = [
        "        name=" + name + ",",
        "        caption=" + cap + ",",
    ]
    if xlabel != "":
        lines.append("        xlabel={{" + str(xlabel) + ", }},")
    if ylabel != "":
        lines.append("        ylabel=" + str(ylabel) + ",")
    if zlabel != "":
        lines.append("        zlabel=" + str(zlabel) + ",")
    lines += [
        "        fill=" + fill + ",",
        "        height=" + str(height) + ",",
        "        width=" + str(width) + ",",
        "        depth=" + str(depth),
    ]
    return ("\n\\pic[shift={" + offset + "}] at " + to + "\n"
            "    {Box={\n" + "\n".join(lines) + "\n        }\n    };\n")


def text_node(at, text, options="anchor=west, font=\\small\\bfseries"):
    return "\n\\node[" + options + "] at " + at + " {" + text + "};\n"


# Straight black arrow flowing downward (south of one box -> north of the next).
def to_arrow(of, to):
    return ("\n\\draw[connection, draw=black, opacity=1, -{Stealth[length=3.5mm]}] ("
            + of + "-south) -- (" + to + "-north);\n")


# Right-angle black arrow: drop, run along a shared horizontal bus, then drop in.
def to_arrow_angle(of, to, drop=1.0):
    return ("\n\\draw[connection, draw=black, opacity=1, -{Stealth[length=3.5mm]}] ("
            + of + "-south) -- ++(0,-" + str(drop) + ") -| (" + to + "-north);\n")


# Half the on-page width of a box. PlotNeuralNet's Box pic origin is its -west
# corner and it extends +x by width*scale (Box.sty default scale = 0.2). Placing
# a box at (prev-anchor) shifted left by its own half-width therefore centres it
# on the previous box's vertical axis, keeping the whole column centre-aligned.
BOX_SCALE = 0.2


def half_w(width):
    return round(width * BOX_SCALE / 2.0, 3)


# Place a box centred under `prev` (or on the page axis if prev is None), `gap`
# units below it.
def centered_box(name, fill, prev, width, height, depth, gap,
                 caption=" ", xlabel="", ylabel="", zlabel=""):
    if prev is None:
        to, offset = "(0,0,0)", "(-{},0,0)".format(half_w(width))
    else:
        to = "({}-anchor)".format(prev)
        offset = "(-{},-{},0)".format(half_w(width), gap)
    return to_box(name, fill, to, offset, width=width, height=height,
                  depth=depth, caption=caption, xlabel=xlabel, ylabel=ylabel,
                  zlabel=zlabel)


# ---------------------------------------------------------------------------
# Encoder column (top -> down), centre-aligned on the page's vertical axis.
# The input / feature T-Nets sit inline in the column (orange boxes) with equal
# input/output sizes: each learns a k x k transform applied to its features.
# ---------------------------------------------------------------------------
ENC_GAP = 1.5   # centre-to-centre vertical gap between encoder boxes

# Each encoder layer is drawn explicitly, top -> down. Box heights are sized by
# their OUTPUT channel count and labelled with the full output tensor shape.


def out_shape(channels):
    return "{} x {}".format(NUM_INPUT_POINTS, channels)


encoder = [
    centered_box("input", r"\ConvColor", None, 10, 1, 1, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] input-west)", str(NUM_FEATURES), options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] input-east)", "Input Point Samples  (" + out_shape(NUM_FEATURES) + ": x,y,z + SDFs)"),

    centered_box("stnin", r"\TNetColor", "input", 10, 1, 3, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] stnin-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] stnin-east)", "Input Transform Network  (" + out_shape(64) + ")"),
    to_arrow("input", "stnin"),

    centered_box("conv1", r"\ConvColor", "stnin", 10, 1, 3, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] conv1-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv1-east)", "1D Convolution  (" + out_shape(64) + ")"),
    to_arrow("stnin", "conv1"),

    centered_box("stnft", r"\TNetColor", "conv1", 10, 1, 3, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] stnft-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] stnft-east)", "Feature Transform Network  (" + out_shape(64) + ")"),
    to_arrow("conv1", "stnft"),

    centered_box("conv2", r"\ConvColor", "stnft", 10, 1, 3, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] conv2-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv2-east)", "1D Convolution  (" + out_shape(64) + ")"),
    to_arrow("stnft", "conv2"),

    centered_box("conv3", r"\ConvColor", "conv2", 10, 1, 6, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] conv3-west)", "128", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv3-east)", "1D Convolution  (" + out_shape(128) + ")"),
    to_arrow("conv2", "conv3"),

    centered_box("conv4", r"\ConvColor", "conv3", 10, 1, 9, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] conv4-west)", "1024", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv4-east)", "1D Convolution  (" + out_shape(1024) + ")"),
    to_arrow("conv3", "conv4"),

    centered_box("maxpool", r"\MaxPoolColor", "conv4", 10, 1, 10, ENC_GAP),
    text_node("([xshift=10pt] maxpool-east)", "Max Pool (symmetric)"),
    to_arrow("conv4", "maxpool"),

    centered_box("feat", r"\ConvColor", "maxpool", 10, 1, 10, ENC_GAP),
    text_node("([xshift=-4pt] feat-west)", str(GLOBAL_FEAT), options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] feat-east)", "Global Feature  (1024)"),
    to_arrow("maxpool", "feat"),
]


# ---------------------------------------------------------------------------
# Multi-pronged decoder: seven heads fanning out below the global feature.
# ---------------------------------------------------------------------------
HEADS = [
    ("shape",        3, "Shape",       "Softmax"),
    ("operation",    2, "Operation",   "Softmax"),
    ("translation",  3, "Translation", "Tanh"),
    ("rotation",     4, "Rotation",    "Linear"),
    ("scale",        3, "Scale",       "Sigmoid"),
    ("blending",     1, "Blending",    "Sigmoid"),
    ("roundness",    1, "Roundness",   "Sigmoid"),
]

SPACING = 4.5            # horizontal gap between heads
TOP_DROP = 2.25          # vertical gap from the global feature to each head
DEC_GAP = 1.2            # centre-to-centre vertical gap between decoder boxes


def decoder_prong(index, prefix, out_size, head_label, activation):
    dx = round((index - (len(HEADS) - 1) / 2.0) * SPACING, 2)
    elems = []

    # Top hidden layer (1024), branched off the global feature and centred on
    # x = dx (relative to the encoder axis at feat's centre).
    h0 = prefix + "_h0"
    elems.append(to_box(h0, r"\DecoderColor", "(feat-anchor)",
                        "({},-{},0)".format(round(dx - half_w(12), 3), TOP_DROP),
                        width=12, height=3, depth=3, ylabel=1024))
    elems.append(to_arrow_angle("feat", h0))
    # elems.append(text_node("([yshift=14pt] {}-north)".format(h0), head_label, options="anchor=south, font=\\small\\bfseries"))

    # Remaining hidden layers, stacked straight down and centred on the prong.
    h1 = prefix + "_h1"
    elems.append(centered_box(h1, r"\DecoderColor", h0, 9, 3, 3, DEC_GAP, ylabel=512))
    elems.append(to_arrow(h0, h1))

    h2 = prefix + "_h2"
    elems.append(centered_box(h2, r"\DecoderColor", h1, 6, 3, 3, DEC_GAP, ylabel=128))
    elems.append(to_arrow(h1, h2))

    h3 = prefix + "_h3"
    elems.append(centered_box(h3, r"\DecoderColor", h2, 3, 3, 3, DEC_GAP, ylabel=64))
    elems.append(to_arrow(h2, h3))

    # Activation output head (purple).
    out = prefix + "_out"
    elems.append(centered_box(out, r"\ActivationColor", h3, 3, 3, 3, DEC_GAP, caption=head_label, ylabel=out_size))
    elems.append(text_node("([xshift=10pt] {}-east)".format(out), activation))
    elems.append(to_arrow(h3, out))
    return elems


# ---------------------------------------------------------------------------
# Assemble the full architecture
# ---------------------------------------------------------------------------
arch = [to_head(PLOTNN_REL), to_cor(), to_colors(), to_begin()]
arch += encoder

# Section titles.
arch += [
    text_node("([xshift=10pt, yshift=18pt] input-east)", "PointNet Encoder",
              options="anchor=west, font=\\large\\bfseries"),
]

for i, (prefix, out_size, label, activation) in enumerate(HEADS):
    arch += decoder_prong(i, prefix, out_size, label, activation)


arch += to_end()

def main():
    namefile = os.path.splitext(os.path.basename(__file__))[0]
    out_path = os.path.join(_THIS_DIR, namefile + ".tex")
    to_generate(arch, out_path)
    print("\nWrote: {}".format(out_path))


if __name__ == "__main__":
    main()
