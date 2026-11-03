import os
import sys

# Import PlotNeuralNet dependency from the parent figures folder
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTNN_REL = os.path.join("..", "PlotNeuralNet")
sys.path.append(os.path.join(_THIS_DIR, PLOTNN_REL))

from pycore.tikzeng import to_head, to_cor, to_begin, to_end, to_generate


# Data sizes (matching the network architecture figure)
NUM_INPUT_POINTS = 2048
NUM_INPUT_FEATURES = 6
NUM_TARGET_FEATURES = 4

# Extra colors and tikz libraries. Network colors match the architecture figure.
def to_colors():
    return r"""
\usetikzlibrary{backgrounds}
\usetikzlibrary{decorations.pathreplacing}
\usetikzlibrary{fit}
\usetikzlibrary{shapes.misc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\def\TNetColor{rgb:orange,6;yellow,4}
\def\MaxPoolColor{rgb:orange,8;red,3}
\def\DecoderColor{rgb:blue,5;white,5}
\def\ActivationColor{rgb:red,5;blue,5}
\def\GreyColor{rgb:black,3;white,5}
\def\TransformColor{""" + TRANSFORM_COLOR + r"""}
\def\OutputBackdropColor{""" + OUTPUT_BACKDROP_COLOR + r"""}
\def\OutputBackdropOutlineColor{""" + OUTPUT_BACKDROP_OUTLINE_COLOR + r"""}
\def\edgecolor{black}
"""

# Half the on-page width of a box, used to keep boxes centered.
BOX_SCALE = 0.2

# New color for the non-network preprocessing layer
TRANSFORM_COLOR = "rgb:green,3;blue,1;white,6"

# Main column, centered on x = 0, from the input CSG at the top to the output CSG at the bottom
TOP_Y = 0.0
TRANSFORM_Y = -2.7
INPUT_Y = TRANSFORM_Y - 2.6
NETWORK_Y = INPUT_Y - 2.6
NEW_PRIMS_Y = NETWORK_Y - 2.6
OUTPUT_CSG_Y = NEW_PRIMS_Y - 2.2

# Simplified network boxes
NET_BOX_WIDTH = 12
NET_BOX_HEIGHT = 4
NET_BOX_DEPTH = 10

# Primitive cubes of the CSG model
PRIM_SIZE = 3
NUM_DRAWN_PRIMS = 3

# Target input samples (left) and input CSG (right) are offset to either side of the transform layer
TOP_SIDE_OFFSET = 4.5
TRANSFORM_INPUT_SHIFT = 0.6

# Recursion path, right of the network
RECURSION_X = 7.8

# Output CSG model: gap between groups of primitives, holding the + signs
OUTPUT_GROUP_GAP = 0.8

# Backdrop around the output CSG primitives, styled like the decoder backdrop of the architecture figure
OUTPUT_BACKDROP_COLOR = "rgb:blue,2;cyan,1;white,8"
OUTPUT_BACKDROP_OUTLINE_COLOR = "rgb:blue,4;cyan,2;white,4"
OUTPUT_BACKDROP_OPACITY = 0.2
OUTPUT_BACKDROP_OUTLINE_OPACITY = 0.6
OUTPUT_BACKDROP_OUTLINE_WIDTH = 0.4
OUTPUT_BACKDROP_CORNER_RADIUS = 8
OUTPUT_BACKDROP_PADDING = 8
# Dots of the fan-out lines to earlier cascades: twice TikZ's "dotted" dot length (one ultra thick line width)
CASCADE_FAN_DASH_PATTERN = "on 3.2pt off 2pt"

# Left edge of the right-hand label column, clear of the network boxes and their depth
LABEL_X = 1.8
# Extra space between a label title and its description
DESCRIPTION_GAP = "4pt"

# No-gradient cross mark on the recursion arrow
NO_GRAD_MARK_SIZE = 9
NO_GRAD_MARK_WIDTH = 1.6
NO_GRAD_MARK_COLOR = "red"

ARROW = r"\draw[connection, draw=black, opacity=1, -{Stealth[length=3.5mm]}]"


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


def half_w(width):
    return round(width * BOX_SCALE / 2.0, 3)


# Place a box with its center at (center_x, center_y).
def box_at(name, fill, center_x, center_y, width, height, depth, **kwargs):
    offset = "({},{},0)".format(round(center_x - half_w(width), 3), center_y)
    return to_box(name, fill, "(0,0,0)", offset, width=width, height=height, depth=depth, **kwargs)


def to_arrow(of, to):
    return "\n" + ARROW + " (" + of + "-south) -- (" + to + "-north);\n"


# Point sample tensor drawn as a thin slab, labelled like the architecture figure input.
def sample_slab(name, center_x, center_y, num_features):
    return [
        box_at(name, r"\GreyColor", center_x, center_y, 10, 1, 1),
        text_node("([xshift=25pt, yshift=-2pt] " + name + "-nearsouthwest)", str(NUM_INPUT_POINTS), options="anchor=north east, font=\\small"),
        text_node("([xshift=-4pt] " + name + "-west)", str(num_features), options="anchor=east, font=\\small"),
    ]


# Label in the right-hand column, vertically centered on y by default.
# Bold title lines are followed by optional regular description lines, separated by DESCRIPTION_GAP.
def right_label(y, title_lines, description_lines=(), options="anchor=west"):
    text = r"\\".join(r"\textbf{" + line + "}" for line in title_lines)
    if description_lines:
        text += r"\\[" + DESCRIPTION_GAP + "]" + r"\\".join(description_lines)
    return text_node("(" + str(LABEL_X) + ", " + str(y) + ")", text, options=options + ", align=left, font=\\small")


def brace(start, end, text, options="mirror", label_options="below=8pt"):
    return ("\n\\draw[decorate, decoration={brace, amplitude=6pt, " + options + "}, thick] "
            + start + " -- " + end
            + " node[midway, " + label_options + ", align=center, font=\\small\\bfseries] {" + text + "};\n")


prim_widths = "{" + ",".join([str(PRIM_SIZE)] * NUM_DRAWN_PRIMS) + "}"
prims_half_w = half_w(PRIM_SIZE) * NUM_DRAWN_PRIMS


# Group of primitive cubes with its left edge at left_x.
def prim_group(name, fill, left_x, y):
    return to_box(name, fill, "(0,0,0)", "({},{},0)".format(round(left_x, 3), y), width=prim_widths, height=PRIM_SIZE, depth=PRIM_SIZE)


# Grey input CSG primitives with n purple new primitives to their right, centered on (center_x, center_y).
def csg_box(prefix, center_x, center_y):
    return [
        prim_group(prefix + "_prev", r"\GreyColor", center_x - 2 * prims_half_w, center_y),
        prim_group(prefix + "_new", r"\ActivationColor", center_x, center_y),
    ]


# ------------------------------------------------------------
# Top row: target input samples and input CSG model (input + n new)
# ------------------------------------------------------------
inputs = [
    *csg_box("csg", TOP_SIDE_OFFSET, TOP_Y),
    brace("([xshift=2pt, yshift=-6pt] csg_prev-nearsouthwest)", "([xshift=-2pt, yshift=-6pt] csg_new-nearsoutheast)", r"Primitives from cascades 0..\textit{n}$-$1"),

    # Target input samples supply the query points and target distances.
    *sample_slab("target", -TOP_SIDE_OFFSET, TOP_Y, NUM_TARGET_FEATURES),
    text_node("([yshift=-14pt] target-south)", r"Target Input Samples\\(3D coordinate + distance)", options="anchor=north, align=center, font=\\small\\bfseries"),
]


# ----------------------
# Network forward pass
# ----------------------
network = [
    # Transform layer turns the input CSG into the network input point samples.
    box_at("transform", r"\TransformColor", 0, TRANSFORM_Y, NET_BOX_WIDTH, NET_BOX_HEIGHT, NET_BOX_DEPTH),
    "\n" + ARROW + " (target-east) -| ([xshift=-" + str(TRANSFORM_INPUT_SHIFT) + "cm] transform-north);\n",
    "\n" + ARROW + " ([xshift=-4pt] csg_prev-west) -| ([xshift=" + str(TRANSFORM_INPUT_SHIFT) + "cm] transform-north);\n",
    right_label(TRANSFORM_Y, ["Preprocessing Layer"], ["Sample CSG at the query points", "and compute the missing,", "filled, and excess volumes"]),

    *sample_slab("input", 0, INPUT_Y, NUM_INPUT_FEATURES),
    right_label(INPUT_Y, ["Input Point Samples", "(3D coordinate + 3 distances)"]),
    to_arrow("transform", "input"),

    # Encoder and primitive decoders, simplified to a single box.
    box_at("network", r"\DecoderColor", 0, NETWORK_Y, NET_BOX_WIDTH, NET_BOX_HEIGHT, NET_BOX_DEPTH),
    to_arrow("input", "network"),
    right_label(NETWORK_Y, ["CSG-CRN Primitive", "Prediction Network"], ["Network parameters shared", "for all cascades"]),

    # Network output is n new primitives.
    prim_group("new_prims", r"\ActivationColor", -prims_half_w, NEW_PRIMS_Y),
    to_arrow("network", "new_prims"),
    # Label sits above the recursion arrow leaving the new primitives.
    right_label(NEW_PRIMS_Y, [r"\textit{n} New Primitives"], options="anchor=south west, yshift=4pt"),
]


# ----------------------------------------------
# Recursion: new primitives -> input CSG model
# ----------------------------------------------
# Midpoint of the bottom recursion segment, between the new primitives and the right side of the loop
recursion_mid = "({}, {})".format(round((prims_half_w + RECURSION_X) / 2.0, 3), NEW_PRIMS_Y)

recursion = [
    # New primitives are appended to the input CSG for the next cascade.
    "\n" + ARROW + " (new_prims-east) -| (" + str(RECURSION_X) + ", " + str(TOP_Y) + ") -- (csg_new-east);\n",

    # The fed-back CSG is generated without gradients.
    "\n\\node[cross out, draw=" + NO_GRAD_MARK_COLOR + ", line width=" + str(NO_GRAD_MARK_WIDTH) + "pt, minimum size=" + str(NO_GRAD_MARK_SIZE) + "pt, inner sep=0pt] at " + recursion_mid + " {};\n",
    text_node(recursion_mid, r"no-grad", options="anchor=north, yshift=-6pt, font=\\small\\bfseries"),
]


# -------------------------------------------------------------------
# Output CSG model: primitives from each earlier cascade + n new ones
# -------------------------------------------------------------------
# Left edges of the cascade groups, stepping left from the latest cascade aligned under the network output.
group_width = 2 * prims_half_w
cascade_n_left = -prims_half_w
cascade_2_left = cascade_n_left - OUTPUT_GROUP_GAP - group_width
cascade_1_left = cascade_2_left - OUTPUT_GROUP_GAP - group_width
# Drop from the new primitives before the dotted lines fan out to the earlier cascades
cascade_fan_drop = round((NEW_PRIMS_Y - OUTPUT_CSG_Y - 2 * half_w(PRIM_SIZE)) / 2.0, 3)


# Group of cascade primitives with a label node (name + "_label") below it.
def cascade_group(name, fill, left_x, label):
    return [
        prim_group(name, fill, left_x, OUTPUT_CSG_Y),
        "\n\\node[anchor=north, font=\\small\\bfseries] (" + name + "_label) at ([yshift=-8pt] " + name + "-south) {" + label + "};\n",
    ]


# Center of the gap left of a cascade group.
def gap_x(right_group_left):
    return round(right_group_left - OUTPUT_GROUP_GAP / 2.0, 3)


# Dotted arrow showing an earlier cascade's primitives also came from the network.
def cascade_fan_arrow(name):
    return ("\n\\draw[connection, draw=black, opacity=1, dash pattern=" + CASCADE_FAN_DASH_PATTERN + ", -{Stealth[length=3.5mm]}] (new_prims-south) -- ++(0,-"
            + str(cascade_fan_drop) + ") -| (" + name + "-north);\n")


output_csg = [
    *cascade_group("cascade_1", r"\GreyColor", cascade_1_left, "Cascade 1"),
    *cascade_group("cascade_2", r"\GreyColor", cascade_2_left, "Cascade 2"),

    # Latest cascade's new primitives are aligned under the network's new primitives.
    *cascade_group("cascade_n", r"\ActivationColor", cascade_n_left, r"Cascade \textit{n}"),

    # The output CSG is the union of every cascade's primitives.
    text_node("(" + str(gap_x(cascade_2_left)) + ", " + str(OUTPUT_CSG_Y) + ")", r"+", options="anchor=center, font=\\small\\bfseries"),
    text_node("(" + str(gap_x(cascade_n_left)) + ", " + str(OUTPUT_CSG_Y) + ")", r"+", options="anchor=center, font=\\small\\bfseries"),
    text_node("(cascade_2_label.base -| " + str(gap_x(cascade_n_left)) + ",0)", r"$\cdots$", options="anchor=base, font=\\small\\bfseries"),

    # Backdrop around every cascade's primitives and their labels.
    "\n\\begin{scope}[on background layer]\n"
    "\\node[fit=(cascade_1-nearsouthwest)(cascade_1-farnorthwest)(cascade_n-farnortheast)(cascade_1_label)(cascade_n_label), "
    "inner sep=" + str(OUTPUT_BACKDROP_PADDING) + "pt, rounded corners=" + str(OUTPUT_BACKDROP_CORNER_RADIUS) + "pt, "
    "fill=\\OutputBackdropColor, fill opacity=" + str(OUTPUT_BACKDROP_OPACITY) + ", "
    "draw=\\OutputBackdropOutlineColor, draw opacity=" + str(OUTPUT_BACKDROP_OUTLINE_OPACITY) + ", "
    "line width=" + str(OUTPUT_BACKDROP_OUTLINE_WIDTH) + "pt] {};\n"
    "\\end{scope}\n",

    cascade_fan_arrow("cascade_1"),
    cascade_fan_arrow("cascade_2"),
    to_arrow("new_prims", "cascade_n"),
    right_label(OUTPUT_CSG_Y, ["Output CSG Model"]),
]


# -----------
# Main Output
# -----------
arch = [to_head(PLOTNN_REL), to_cor(), to_colors(), to_begin()]
arch += inputs
arch += network
arch += recursion
arch += output_csg
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
