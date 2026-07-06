import os
import sys

# Import PlotNeuralNet dependency
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTNN_REL = os.path.join(".", "PlotNeuralNet")
sys.path.append(os.path.join(_THIS_DIR, PLOTNN_REL))

from pycore.tikzeng import to_head, to_cor, to_begin, to_end, to_generate


# Network layer sizes
NUM_INPUT_POINTS = 2048
NUM_FEATURES = 6
ENCODER_CONV = [64, 128, 1024]
GLOBAL_FEAT = 1024
DECODER_LAYERS = [1024, 512, 128, 64]

# Extra colors
def to_colors():
    return r"""
\def\TNetColor{rgb:orange,6;yellow,4}
\def\MaxPoolColor{rgb:orange,8;red,3}
\def\DecoderColor{rgb:blue,5;white,5}
\def\ActivationColor{rgb:red,5;blue,5}
\def\GreyColor{rgb:black,3;white,5}
\def\edgecolor{black}
"""

# Half the on-page width of a box. PlotNeuralNet's Box pic origin is its -west
# corner and it extends +x by width*scale (Box.sty default scale = 0.2). Placing
# a box at (prev-anchor) shifted left by its own half-width therefore centres it
# on the previous box's vertical axis, keeping the whole column centre-aligned.
BOX_SCALE = 0.2
# centre-to-centre vertical gap between encoder boxes
ENC_GAP = 1.5
# horizontal gap between the three in-line pool layers
POOL_REGRESSOR_HORIZONTAL_GAP = 3.0

# Decoder spacing
REGRESSOR_VERTICAL_GAP = 1.2
REGRESSOR_HORIZONTAL_GAP = 4.0
PRIM_DECODER_TO_HEAD_GAP = 2.25


# ---------------
# Element helpers
# ---------------
def to_box(name, fill, to, offset, width=2, height=3, depth=3,
           caption=" ", xlabel="", ylabel="", zlabel=""):
    cap = caption if caption == " " else "{" + caption + "}"
    lines = [
        "name=" + name + ",",
        "caption=" + cap + ",",
    ]
    if xlabel != "":
        lines.append("xlabel={{" + str(xlabel) + ", }},")
    if ylabel != "":
        lines.append("ylabel=" + str(ylabel) + ",")
    if zlabel != "":
        lines.append("zlabel=" + str(zlabel) + ",")

    lines += [
        "fill=" + fill + ",",
        "height=" + str(height) + ",",
        "width=" + str(width) + ",",
        "depth=" + str(depth),
    ]
    return ("\n\\pic[shift={" + offset + "}] at " + to + "\n"
            "    {Box={\n" + "\n".join(lines) + "\n        }\n    };\n")


def text_node(at, text, options="anchor=west, font=\\small\\bfseries"):
    return "\n\\node[" + options + "] at " + at + " {" + text + "};\n"


def to_arrow(of, to):
    return ("\n\\draw[connection, draw=black, opacity=1, -{Stealth[length=3.5mm]}] (" + of + "-south) -- (" + to + "-north);\n")


def to_arrow_angle(of, to, drop=1.0):
    return ("\n\\draw[connection, draw=black, opacity=1, -{Stealth[length=3.5mm]}] (" + of + "-south) -- ++(0,-" + str(drop) + ") -| (" + to + "-north);\n")


def to_arrow_straight_down(of, to):
    return ("\n\\draw[connection, draw=black, opacity=1, -{Stealth[length=3.5mm]}] (" + of + "-south) -- (" + of + "-south |- " + to + "-north);\n")


def sloped_east_label(name, text, options="align=center, font=\\small\\bfseries"):
    shift = "([xshift=16pt, yshift=-9pt] "
    return ("\n\\path " + shift + name + "-neareast) edge [draw=none, \"" + text + "\", sloped, midway, " + options + "] " + shift + name + "-fareast);\n")


def half_w(width):
    return round(width * BOX_SCALE / 2.0, 3)


def centered_box(name, fill, prev, width, height, depth, gap,
                 caption=" ", xlabel="", ylabel="", zlabel=""):
    if prev is None:
        to, offset = "(0,0,0)", "(-{},0,0)".format(half_w(width))
    else:
        to = "({}-anchor)".format(prev)
        offset = "(-{},-{},0)".format(half_w(width), gap)
    return to_box(name, fill, to, offset, width=width, height=height, depth=depth, caption=caption, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)



def out_shape(channels):
    return "{} x {}".format(NUM_INPUT_POINTS, channels)


# ---------------
# Encoder network
# ---------------
encoder = [
    centered_box("input", r"\GreyColor", None, 10, 1, 1, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] input-west)", str(NUM_FEATURES), options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] input-east)", r"Input Point Samples\\(3D coordinate + 3 distances)", options="anchor=west, align=left, font=\\small\\bfseries"),

    centered_box("stnin", r"\TNetColor", "input", 10, 1, 3, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] stnin-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] stnin-east)", "Input Transform Network"),
    to_arrow("input", "stnin"),

    centered_box("conv1", r"\ConvColor", "stnin", 10, 1, 3, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] conv1-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv1-east)", "1D Convolution"),
    to_arrow("stnin", "conv1"),

    centered_box("stnft", r"\TNetColor", "conv1", 10, 1, 3, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] stnft-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] stnft-east)", "Feature Transform Network"),
    to_arrow("conv1", "stnft"),

    centered_box("conv2", r"\ConvColor", "stnft", 10, 1, 3, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] conv2-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv2-east)", "1D Convolution"),
    to_arrow("stnft", "conv2"),

    centered_box("conv3", r"\ConvColor", "conv2", 10, 1, 6, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] conv3-west)", "128", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv3-east)", "1D Convolution"),
    to_arrow("conv2", "conv3"),

    centered_box("conv4", r"\ConvColor", "conv3", 10, 1, 9, ENC_GAP, xlabel=NUM_INPUT_POINTS),
    text_node("([xshift=-4pt] conv4-west)", "1024", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv4-east)", "1D Convolution"),
    to_arrow("conv3", "conv4"),

    # Center pool layer, directly below conv4.
    centered_box("maxpool", r"\MaxPoolColor", "conv4", 10, 1, 10, ENC_GAP),
    to_arrow("conv4", "maxpool"),

    # Left and right pool layers, in-line horizontally with the center pool.
    to_box("maxpool_l", r"\MaxPoolColor", "(maxpool-anchor)", "(-{},0,0)".format(round(POOL_REGRESSOR_HORIZONTAL_GAP + half_w(10), 3), ), width=10, height=1, depth=10),
    to_box("maxpool_r", r"\MaxPoolColor", "(maxpool-anchor)", "({},0,0)".format(round(POOL_REGRESSOR_HORIZONTAL_GAP - half_w(10), 3), ), width=10, height=1, depth=10),

    # conv4 fans out to all three pool layers via right-angle connectors.
    to_arrow_angle("conv4", "maxpool_l", drop=ENC_GAP / 2.0),
    to_arrow_angle("conv4", "maxpool_r", drop=ENC_GAP / 2.0),

    # "Pool" label on the east face of each pool box, parallel to its side.
    sloped_east_label("maxpool_l", "Max"),
    sloped_east_label("maxpool", "Mean"),
    sloped_east_label("maxpool_r", "Avg-TopK"),
    text_node("([xshift=40pt] maxpool_r-east)", "Pooling Layer"),

    centered_box("feat", r"\GreyColor", "maxpool", 40, 1, 10, ENC_GAP),
    text_node("([xshift=-10pt, yshift=-20pt] feat-south)", "3072", options="anchor=east, font=\\small"),
    text_node("([xshift=40pt] feat-east)", "Global Feature"),
    to_arrow("maxpool", "feat"),
    # Side pool layers drop straight down into the widened global feature box.
    to_arrow_straight_down("maxpool_l", "feat"),
    to_arrow_straight_down("maxpool_r", "feat"),
]


# -----------------------
# Shared decoder layers
# -----------------------
shared_decoder = [
    centered_box("dec1024", r"\DecoderColor", "feat", 12, 3, 3, REGRESSOR_VERTICAL_GAP, ylabel=1024),
    text_node("([xshift=10pt] dec1024-east)", "Fully-Connected MLP"),
    to_arrow("feat", "dec1024"),

    centered_box("dec512", r"\DecoderColor", "dec1024", 9, 3, 3, REGRESSOR_VERTICAL_GAP, ylabel=512),
    text_node("([xshift=20pt] dec512-east)", "Fully-Connected MLP"),
    to_arrow("dec1024", "dec512"),
]


# ---------------
# Decoder network
# ---------------
HEADS = [
    ("shape",        3, "Shape",       "Softmax"),
    ("operation",    2, "Operation",   "Softmax"),
    ("translation",  3, "Translation", "Tanh"),
    ("rotation",     4, "Rotation",    "Linear"),
    ("scale",        3, "Scale",       "Sigmoid"),
    ("blending",     1, "Blending",    "Sigmoid"),
    ("roundness",    1, "Roundness",   "Sigmoid"),
]


def decoder_prong(index, prefix, out_size, head_label, activation):
    dx = round((index - (len(HEADS) - 1) / 2.0) * REGRESSOR_HORIZONTAL_GAP, 2)
    elems = []

    # Top hidden layer (128), branched off the last shared decoder layer and
    # centred on x = dx (relative to the encoder axis at dec512's centre).
    h0 = prefix + "_h0"
    elems.append(to_box(h0, r"\DecoderColor", "(dec512-anchor)", "({},-{},0)".format(round(dx - half_w(6), 3), PRIM_DECODER_TO_HEAD_GAP), width=6, height=3, depth=3, ylabel=128))
    elems.append(to_arrow_angle("dec512", h0))

    # Remaining hidden layer, stacked straight down and centred on the prong.
    h1 = prefix + "_h1"
    elems.append(centered_box(h1, r"\DecoderColor", h0, 3, 3, 3, REGRESSOR_VERTICAL_GAP, ylabel=64))
    elems.append(to_arrow(h0, h1))

    # Activation output head (purple).
    out = prefix + "_out"
    elems.append(centered_box(out, r"\ActivationColor", h1, 3, 3, 3, REGRESSOR_VERTICAL_GAP, caption=head_label, ylabel=out_size))
    elems.append(text_node("([xshift=10pt] {}-east)".format(out), activation))
    elems.append(to_arrow(h1, out))
    return elems


# -----------
# Main Output
# -----------
arch = [to_head(PLOTNN_REL), to_cor(), to_colors(), to_begin()]
arch += encoder
arch += shared_decoder

for i, (prefix, out_size, label, activation) in enumerate(HEADS):
    arch += decoder_prong(i, prefix, out_size, label, activation)


arch += to_end()

def main():
    namefile = os.path.splitext(os.path.basename(__file__))[0]
    out_path = os.path.join(_THIS_DIR, namefile + ".tex")

    # Generate tex file
    saved_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        to_generate(arch, out_path)
    finally:
        sys.stdout.close()
        sys.stdout = saved_stdout

    print("Wrote: {}".format(out_path))


if __name__ == "__main__":
    main()
