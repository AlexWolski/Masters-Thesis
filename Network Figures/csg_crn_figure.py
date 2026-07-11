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

# Extra colors and tikz libraries
def to_colors():
    return r"""
\usetikzlibrary{backgrounds}
\def\TNetColor{rgb:orange,6;yellow,4}
\def\MaxPoolColor{rgb:orange,8;red,3}
\def\DecoderColor{rgb:blue,5;white,5}
\def\ActivationColor{rgb:red,5;blue,5}
\def\GreyColor{rgb:black,3;white,5}
\def\EncoderBackdropColor{""" + ENCODER_BACKDROP_COLOR + r"""}
\def\EncoderBackdropOutlineColor{""" + ENCODER_BACKDROP_OUTLINE_COLOR + r"""}
\def\DecoderBackdropColor{""" + DECODER_BACKDROP_COLOR + r"""}
\def\DecoderBackdropOutlineColor{""" + DECODER_BACKDROP_OUTLINE_COLOR + r"""}
\def\edgecolor{black}
"""

# Half the on-page width of a box, used to keep boxes centered.
BOX_SCALE = 0.2
# centre-to-centre vertical gap between encoder boxes
ENC_GAP = 1.5
# horizontal gap between the three in-line pool layers
POOL_REGRESSOR_HORIZONTAL_GAP = 3.0
# extra vertical padding before the pooling layer and after the global features
POOL_PAD = 1.0
FEAT_PAD = 4.5

# height between the Global Pool layer and the regressor head placeholders.
PRIM_DECODER_HEAD_DROP = 3.75

# Decoder spacing
REGRESSOR_VERTICAL_GAP = 1.2
REGRESSOR_HORIZONTAL_GAP = 3.5
PRIM_DECODER_TO_HEAD_GAP = 2.25

ENCODER_BACKDROP_CENTER_X = 0.0
ENCODER_BACKDROP_CENTER_Y = -7.5
ENCODER_BACKDROP_WIDTH = 18
ENCODER_BACKDROP_HEIGHT = 13.5
ENCODER_BACKDROP_OPACITY = 0.2
ENCODER_BACKDROP_CORNER_RADIUS = 8
ENCODER_BACKDROP_COLOR = "rgb:red,2;white,8"
ENCODER_BACKDROP_OUTLINE_COLOR = "rgb:red,4;white,4"
ENCODER_BACKDROP_OUTLINE_WIDTH = 0.4
ENCODER_BACKDROP_OUTLINE_OPACITY = 0.6

DECODER_BACKDROP_CENTER_X = 0.0
DECODER_BACKDROP_CENTER_Y = -22.7
DECODER_BACKDROP_WIDTH = 26
DECODER_BACKDROP_HEIGHT = 9.3
DECODER_BACKDROP_OPACITY = 0.2
DECODER_BACKDROP_CORNER_RADIUS = 8
DECODER_BACKDROP_COLOR = "rgb:blue,2;cyan,1;white,8"
DECODER_BACKDROP_OUTLINE_COLOR = "rgb:blue,4;cyan,2;white,4"
DECODER_BACKDROP_OUTLINE_WIDTH = 0.4
DECODER_BACKDROP_OUTLINE_OPACITY = 0.6


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


def to_arrow_depth_step(start, drop1=1.0, dz=15, drop2=1.0, dx=0, end=None):
    # Right-angle connector that steps through depth: drop in y, travel in z
    # (negative = away from the viewer), then drop in y again. `start` is a full
    # TikZ coordinate string (e.g. "([xshift=5pt] feat-south)"); dz/dx are in pt.
    # If `end` is given, the arrow tip is named "<end>-anchor" so a centered_box
    # can be placed there via prev=<end>.
    tag = " coordinate (" + end + "-anchor)" if end else ""
    return ("\n\\draw[connection, draw=black, opacity=1, -{Stealth[length=3.5mm]}] "
            + start
            + " -- ++(0,-" + str(drop1) + ",0) "
            + "-- ++(" + str(dx) + "pt,0," + str(dz) + "pt) "
            + "-- ++(0,-" + str(drop2) + ",0)" + tag + ";\n")


def sloped_east_label(name, text, options="align=center, font=\\small\\bfseries"):
    shift = "([xshift=16pt, yshift=-9pt] "
    return ("\n\\path " + shift + name + "-neareast) edge [draw=none, \"" + text + "\", sloped, midway, " + options + "] " + shift + name + "-fareast);\n")


def half_w(width):
    return round(width * BOX_SCALE / 2.0, 3)


def background_box(center_x, center_y, width, height,
                   fill=r"\EncoderBackdropColor",
                   opacity=ENCODER_BACKDROP_OPACITY,
                   corner_radius=ENCODER_BACKDROP_CORNER_RADIUS,
                   outline_color=r"\EncoderBackdropOutlineColor",
                   outline_width=ENCODER_BACKDROP_OUTLINE_WIDTH,
                   outline_opacity=ENCODER_BACKDROP_OUTLINE_OPACITY):
    left = round(center_x - width / 2.0, 3)
    right = round(center_x + width / 2.0, 3)
    top = round(center_y + height / 2.0, 3)
    bottom = round(center_y - height / 2.0, 3)

    # Fill and outline are drawn as two separate strokes so each keeps its own
    # opacity (a single \draw would share one opacity for line and fill).
    return (
        "\n\\begin{scope}[on background layer]\n"
        "\\draw[fill=" + fill + ", opacity=" + str(opacity)
        + ", draw=none, rounded corners=" + str(corner_radius) + "pt]\n"
        "    (" + str(left) + ", " + str(top) + ")\n"
        "    rectangle\n"
        "    (" + str(right) + ", " + str(bottom) + ");\n"
        "\\draw[draw=" + outline_color + ", line width=" + str(outline_width)
        + "pt, opacity=" + str(outline_opacity)
        + ", rounded corners=" + str(corner_radius) + "pt]\n"
        "    (" + str(left) + ", " + str(top) + ")\n"
        "    rectangle\n"
        "    (" + str(right) + ", " + str(bottom) + ");\n"
        "\\end{scope}\n"
    )


def backdrop_label(center_x, center_y, width, height, text, inset=0.35):
    left = round(center_x - width / 2.0 + inset, 3)
    top = round(center_y + height / 2.0 - inset, 3)
    return text_node(
        "(" + str(left) + ", " + str(top) + ")",
        text,
        options="anchor=north west, font=\\Large\\bfseries",
    )


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
    centered_box("input", r"\GreyColor", None, 10, 1, 1, ENC_GAP),
    text_node("([xshift=25pt, yshift=-2pt] input-nearsouthwest)", str(NUM_INPUT_POINTS), options="anchor=north east, font=\\small"),
    text_node("([xshift=-4pt] input-west)", str(NUM_FEATURES), options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] input-east)", r"Input Point Samples\\(3D coordinate + 3 distances)", options="anchor=west, align=left, font=\\small\\bfseries"),

    centered_box("stnin", r"\TNetColor", "input", 10, 2, 3, ENC_GAP),
    text_node("([xshift=25pt, yshift=-2pt] stnin-nearsouthwest)", str(NUM_INPUT_POINTS), options="anchor=north east, font=\\small"),
    text_node("([xshift=-4pt] stnin-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] stnin-east)", "Input Transform Network"),
    to_arrow("input", "stnin"),

    centered_box("conv1", r"\ConvColor", "stnin", 10, 2, 3, ENC_GAP),
    text_node("([xshift=25pt, yshift=-2pt] conv1-nearsouthwest)", str(NUM_INPUT_POINTS), options="anchor=north east, font=\\small"),
    text_node("([xshift=-4pt] conv1-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv1-east)", "conv1"),
    to_arrow("stnin", "conv1"),

    centered_box("stnft", r"\TNetColor", "conv1", 10, 2, 3, ENC_GAP),
    text_node("([xshift=25pt, yshift=-2pt] stnft-nearsouthwest)", str(NUM_INPUT_POINTS), options="anchor=north east, font=\\small"),
    text_node("([xshift=-4pt] stnft-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] stnft-east)", "Feature Transform Network"),
    to_arrow("conv1", "stnft"),

    centered_box("conv2", r"\ConvColor", "stnft", 10, 2, 3, ENC_GAP),
    text_node("([xshift=25pt, yshift=-2pt] conv2-nearsouthwest)", str(NUM_INPUT_POINTS), options="anchor=north east, font=\\small"),
    text_node("([xshift=-4pt] conv2-west)", "64", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv2-east)", "conv2"),
    to_arrow("stnft", "conv2"),

    centered_box("conv3", r"\ConvColor", "conv2", 10, 2, 6, ENC_GAP),
    text_node("([xshift=25pt, yshift=-2pt] conv3-nearsouthwest)", str(NUM_INPUT_POINTS), options="anchor=north east, font=\\small"),
    text_node("([xshift=-10pt] conv3-west)", "128", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv3-east)", "conv3"),
    to_arrow("conv2", "conv3"),

    centered_box("conv4", r"\ConvColor", "conv3", 10, 2, 9, ENC_GAP),
    text_node("([xshift=25pt, yshift=-2pt] conv4-nearsouthwest)", str(NUM_INPUT_POINTS), options="anchor=north east, font=\\small"),
    text_node("([xshift=-10pt] conv4-west)", "1024", options="anchor=east, font=\\small"),
    text_node("([xshift=10pt] conv4-east)", "conv4"),
    to_arrow("conv3", "conv4"),

    # Center pool layer, directly below conv4.
    centered_box("maxpool", r"\MaxPoolColor", "conv4", 10, 2, 10, ENC_GAP + POOL_PAD),
    to_arrow("conv4", "maxpool"),

    # Left and right pool layers, in-line horizontally with the center pool.
    to_box("maxpool_l", r"\MaxPoolColor", "(maxpool-anchor)", "(-{},0,0)".format(round(POOL_REGRESSOR_HORIZONTAL_GAP + half_w(10), 3), ), width=10, height=2, depth=10),
    to_box("maxpool_r", r"\MaxPoolColor", "(maxpool-anchor)", "({},0,0)".format(round(POOL_REGRESSOR_HORIZONTAL_GAP - half_w(10), 3), ), width=10, height=2, depth=10),

    # conv4 fans out to all three pool layers via right-angle connectors.
    to_arrow_angle("conv4", "maxpool_l", drop=(ENC_GAP + POOL_PAD) / 2.0),
    to_arrow_angle("conv4", "maxpool_r", drop=(ENC_GAP + POOL_PAD) / 2.0),

    # "Pool" label on the east face of each pool box, parallel to its side.
    sloped_east_label("maxpool_l", "Max"),
    sloped_east_label("maxpool", "Mean"),
    sloped_east_label("maxpool_r", "Avg-TopK"),
    text_node("([xshift=40pt] maxpool_r-east)", "Pooling Layer"),

    centered_box("feat", r"\GreyColor", "maxpool", 40, 2, 10, ENC_GAP),
    text_node("([xshift=-10pt, yshift=-20pt] feat-south)", "3072", options="anchor=east, font=\\small"),
    text_node("([xshift=40pt] feat-east)", "Global Features"),
    to_arrow("maxpool", "feat"),
    # Side pool layers drop straight down into the widened global feature box.
    to_arrow_straight_down("maxpool_l", "feat"),
    to_arrow_straight_down("maxpool_r", "feat"),
    # Two arrows exiting the centre of the global feature pool: down, away from
    # the viewer (-z), then down again. A 3x3x3 decoder-coloured cube sits at
    # the tip of each arrow.
    to_arrow_depth_step("(feat-south)", drop1=PRIM_DECODER_HEAD_DROP, dz=-3, drop2=1.0, end="cubeA"),
    to_arrow_depth_step("(feat-south)", drop1=PRIM_DECODER_HEAD_DROP, dz=-6, drop2=1.0, end="cubeB"),
    centered_box("cubeA_box", r"\DecoderColor", "cubeA", 3, 3, 3, half_w(3)),
    text_node("([xshift=10pt] cubeA_box-east)", "Primitive Decoder 2"),
    centered_box("cubeB_box", r"\DecoderColor", "cubeB", 3, 3, 3, half_w(3)),
    text_node("([xshift=10pt] cubeB_box-east)", r"Primitive Decoder \textit{n}"),

    # Encdoer backdrop
    background_box(ENCODER_BACKDROP_CENTER_X, ENCODER_BACKDROP_CENTER_Y, ENCODER_BACKDROP_WIDTH, ENCODER_BACKDROP_HEIGHT),
    backdrop_label(ENCODER_BACKDROP_CENTER_X, ENCODER_BACKDROP_CENTER_Y, ENCODER_BACKDROP_WIDTH, ENCODER_BACKDROP_HEIGHT, "Encoder Network"),
]


# -----------------------
# Shared decoder layers
# -----------------------
shared_decoder = [
    centered_box("dec1024", r"\DecoderColor", "feat", 12, 2, 2, REGRESSOR_VERTICAL_GAP + FEAT_PAD, ylabel=1024),
    text_node("([xshift=10pt] dec1024-east)", "mlp1"),
    to_arrow("feat", "dec1024"),

    centered_box("dec512", r"\DecoderColor", "dec1024", 9, 2, 2, REGRESSOR_VERTICAL_GAP, ylabel=512),
    text_node("([xshift=20pt] dec512-east)", "mlp2"),
    to_arrow("dec1024", "dec512"),

    centered_box("prim", r"\GreyColor", "dec512", 9, 2, 2, REGRESSOR_VERTICAL_GAP, ylabel=512),
    text_node("([xshift=20pt] prim-east)", "Primitive Features"),
    to_arrow("dec512", "prim"),
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

    # Top hidden layer (128), branched off the primitive features layer and
    # centred on x = dx (relative to the encoder axis at prim's centre).
    h0 = prefix + "_h0"
    elems.append(to_box(h0, r"\DecoderColor", "(prim-anchor)", "({},-{},0)".format(round(dx - half_w(6), 3), PRIM_DECODER_TO_HEAD_GAP), width=6, height=2, depth=2, ylabel=128))
    elems.append(to_arrow_angle("prim", h0))

    # Remaining hidden layer, stacked straight down and centred on the prong.
    h1 = prefix + "_h1"
    elems.append(centered_box(h1, r"\DecoderColor", h0, 2, 2, 2, REGRESSOR_VERTICAL_GAP, ylabel=64))
    elems.append(to_arrow(h0, h1))

    # Activation output head (purple).
    out = prefix + "_out"
    elems.append(centered_box(out, r"\ActivationColor", h1, 2, 2, 2, REGRESSOR_VERTICAL_GAP, caption=head_label, ylabel=out_size))
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

# Low-opacity backdrop grouping the decoder network, below the encoder backdrop.
arch += [background_box(
    DECODER_BACKDROP_CENTER_X, DECODER_BACKDROP_CENTER_Y,
    DECODER_BACKDROP_WIDTH, DECODER_BACKDROP_HEIGHT,
    fill=r"\DecoderBackdropColor",
    opacity=DECODER_BACKDROP_OPACITY,
    corner_radius=DECODER_BACKDROP_CORNER_RADIUS,
    outline_color=r"\DecoderBackdropOutlineColor",
    outline_width=DECODER_BACKDROP_OUTLINE_WIDTH,
    outline_opacity=DECODER_BACKDROP_OUTLINE_OPACITY,
)]
arch += [backdrop_label(
    DECODER_BACKDROP_CENTER_X, DECODER_BACKDROP_CENTER_Y,
    DECODER_BACKDROP_WIDTH, DECODER_BACKDROP_HEIGHT, "Primitive Decoder 1",
)]

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
