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
\usetikzlibrary{fit}
\usetikzlibrary{shapes.misc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\def\DecoderColor{rgb:blue,5;white,5}
\def\ActivationColor{rgb:red,5;blue,5}
\def\GreyColor{rgb:black,3;white,5}
\def\TransformColor{""" + TRANSFORM_COLOR + r"""}
\def\CascadeBackdropColor{""" + CASCADE_BACKDROP_COLOR + r"""}
\def\CascadeBackdropOutlineColor{""" + CASCADE_BACKDROP_OUTLINE_COLOR + r"""}
\def\edgecolor{black}
"""

# Half the on-page width of a box, used to keep boxes centered.
BOX_SCALE = 0.2

# New color for the non-network preprocessing layer
TRANSFORM_COLOR = "rgb:green,3;blue,1;white,6"

# Number of cascades drawn before the chain is continued with an ellipsis
NUM_DRAWN_CASCADES = 2

# Vertical offsets within a cascade, from its preprocessing layer
INPUT_DY = -2.6
NETWORK_DY = INPUT_DY - 2.6
NEW_PRIMS_DY = NETWORK_DY - 2.6
# Distance between the preprocessing layers of consecutive cascades
CASCADE_SPACING = 11.7

# Simplified network boxes
NET_BOX_WIDTH = 12
NET_BOX_HEIGHT = 4
NET_BOX_DEPTH = 10

# Primitive cubes of the CSG model
PRIM_SIZE = 3
NUM_DRAWN_PRIMS = 3

# Target input samples shared by every cascade, above and left of the cascades.
# The line to each preprocessing layer drops from TARGET_BUS_SHIFT right of the slab center, crosses over
# TARGET_ENTRY_RISE above the layer, and turns down into it TARGET_ENTRY_SHIFT left of its center.
TARGET_X = -4.5
TARGET_Y = 3.1
TARGET_BUS_SHIFT = 0.6
TARGET_ENTRY_RISE = 0.6
TARGET_ENTRY_SHIFT = 0.6

# Backdrop around each cascade
CASCADE_BACKDROP_COLOR = "rgb:black,1;white,4"
CASCADE_BACKDROP_OUTLINE_COLOR = "rgb:black,2;white,3"
CASCADE_BACKDROP_OPACITY = 0.15
CASCADE_BACKDROP_OUTLINE_OPACITY = 0.6
CASCADE_BACKDROP_OUTLINE_WIDTH = 0.4
CASCADE_BACKDROP_CORNER_RADIUS = 8
CASCADE_BACKDROP_LEFT = -3.0
CASCADE_BACKDROP_RIGHT = 7.0
CASCADE_BACKDROP_TOP_PAD = 2.1
CASCADE_BACKDROP_BOTTOM_PAD = 0.8
CASCADE_TITLE_INSET = 0.35

# Length of the final link below the last drawn cascade's backdrop, ending at the ellipsis
CONTINUATION_DROP = 1.2

# Left edge of the right-hand label column, clear of the network boxes and their depth
LABEL_X = 1.8
# Extra space between a label title and its description
DESCRIPTION_GAP = "4pt"

# No-gradient cross mark on the links between cascades
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
    offset = "({},{},0)".format(round(center_x - half_w(width), 3), round(center_y, 3))
    return to_box(name, fill, "(0,0,0)", offset, width=width, height=height, depth=depth, **kwargs)


def to_arrow(of, to):
    return "\n" + ARROW + " (" + of + "-south) -- (" + to + "-north);\n"


def point(x, y):
    return "({}, {})".format(round(x, 3), round(y, 3))


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
    return text_node(point(LABEL_X, y), text, options=options + ", align=left, font=\\small")


prim_widths = "{" + ",".join([str(PRIM_SIZE)] * NUM_DRAWN_PRIMS) + "}"
prims_half_w = half_w(PRIM_SIZE) * NUM_DRAWN_PRIMS


# Group of primitive cubes centered on (center_x, y).
def prim_group(name, fill, center_x, y):
    return to_box(name, fill, "(0,0,0)", "({},{},0)".format(round(center_x - prims_half_w, 3), round(y, 3)), width=prim_widths, height=PRIM_SIZE, depth=PRIM_SIZE)


# Red cross marking a detached (no gradient) vertical link, labelled to its right.
def no_grad_mark(at):
    return [
        "\n\\node[cross out, draw=" + NO_GRAD_MARK_COLOR + ", line width=" + str(NO_GRAD_MARK_WIDTH) + "pt, minimum size=" + str(NO_GRAD_MARK_SIZE) + "pt, inner sep=0pt] at " + at + " {};\n",
        text_node(at, r"no-grad", options="anchor=west, xshift=8pt, font=\\small\\bfseries"),
    ]


# Light backdrop spanning one cascade, titled in its top-left corner.
def cascade_backdrop(name, top_y, bottom_y, title):
    return [
        "\n\\begin{scope}[on background layer]\n"
        "\\node[fit={" + point(CASCADE_BACKDROP_LEFT, top_y) + " " + point(CASCADE_BACKDROP_RIGHT, bottom_y) + "}, "
        "inner sep=0pt, rounded corners=" + str(CASCADE_BACKDROP_CORNER_RADIUS) + "pt, "
        "fill=\\CascadeBackdropColor, fill opacity=" + str(CASCADE_BACKDROP_OPACITY) + ", "
        "draw=\\CascadeBackdropOutlineColor, draw opacity=" + str(CASCADE_BACKDROP_OUTLINE_OPACITY) + ", "
        "line width=" + str(CASCADE_BACKDROP_OUTLINE_WIDTH) + "pt] (" + name + ") {};\n"
        "\\end{scope}\n",
        text_node("([xshift=" + str(CASCADE_TITLE_INSET) + "cm, yshift=-" + str(CASCADE_TITLE_INSET) + "cm] " + name + ".north west)", title, options="anchor=north west, font=\\small\\bfseries"),
    ]


# ------------------------------------------------------------------
# One cascade: its own CSG-CRN network predicting n new primitives.
# Cascade 0 starts from the target alone; later cascades also take the
# primitives predicted by the previous cascade, arriving from above.
# ------------------------------------------------------------------
def cascade(k, transform_y):
    p = "c{}_".format(k)
    input_y = transform_y + INPUT_DY
    network_y = transform_y + NETWORK_DY
    new_prims_y = transform_y + NEW_PRIMS_DY

    return [
        # Preprocessing layer turns the target (and previous primitives) into the network input point samples.
        box_at(p + "transform", r"\TransformColor", 0, transform_y, NET_BOX_WIDTH, NET_BOX_HEIGHT, NET_BOX_DEPTH),
        right_label(transform_y, ["Preprocessing Layer"], ["Sample CSG at the query points", "and compute the missing,", "filled, and excess volumes"]),

        *sample_slab(p + "input", 0, input_y, NUM_INPUT_FEATURES),
        right_label(input_y, ["Input Point Samples", "(3D coordinate + 3 distances)"]),
        to_arrow(p + "transform", p + "input"),

        # Encoder and primitive decoders, simplified to a single box with this cascade's own parameters.
        box_at(p + "network", r"\DecoderColor", 0, network_y, NET_BOX_WIDTH, NET_BOX_HEIGHT, NET_BOX_DEPTH),
        to_arrow(p + "input", p + "network"),
        right_label(network_y, ["CSG-CRN Primitive", "Prediction Network"], [r"Network parameters $\theta_{" + str(k) + "}$", "used only by cascade " + str(k)]),

        # Network output is n new primitives.
        prim_group(p + "new_prims", r"\ActivationColor", 0, new_prims_y),
        to_arrow(p + "network", p + "new_prims"),
        right_label(new_prims_y, [r"\textit{n} New Primitives"]),

        # Shared target input samples come in from the left and turn down into the preprocessing layer.
        "\n" + ARROW + " ([xshift=" + str(TARGET_BUS_SHIFT) + "cm] target-south) |- "
        "([xshift=-" + str(TARGET_ENTRY_SHIFT) + "cm, yshift=" + str(TARGET_ENTRY_RISE) + "cm] " + p + "transform-north) -- "
        "([xshift=-" + str(TARGET_ENTRY_SHIFT) + "cm] " + p + "transform-north);\n",

        *cascade_backdrop(p + "backdrop", transform_y + CASCADE_BACKDROP_TOP_PAD, new_prims_y - CASCADE_BACKDROP_BOTTOM_PAD, "Cascade " + str(k)),
    ]


# Link carrying a cascade's new primitives straight down, either into the next cascade's
# preprocessing layer or, after the last drawn cascade, to an ellipsis showing the chain continues.
def cascade_link(k, transform_y, last):
    p = "c{}_".format(k)
    new_prims_y = transform_y + NEW_PRIMS_DY
    next_transform_y = transform_y - CASCADE_SPACING
    # Middle of the gap between this cascade's backdrop and the next one's
    backdrop_bottom = new_prims_y - CASCADE_BACKDROP_BOTTOM_PAD
    mid = point(0, (backdrop_bottom + next_transform_y + CASCADE_BACKDROP_TOP_PAD) / 2.0)

    if last:
        # End a little below the backdrop, past the no-grad mark.
        end_y = backdrop_bottom - CONTINUATION_DROP
        elements = [
            "\n" + ARROW + " (" + p + "new_prims-south) -- " + point(0, end_y) + ";\n",
            text_node(point(0, end_y), r"$\vdots$", options="anchor=north, yshift=-2pt, font=\\Large"),
        ]
    else:
        elements = [to_arrow(p + "new_prims", "c{}_transform".format(k + 1))]

    # Earlier cascades run in inference mode, so their primitives are passed on without gradients.
    return elements + no_grad_mark(mid)


# Target input samples shared by every cascade
target = [
    *sample_slab("target", TARGET_X, TARGET_Y, NUM_TARGET_FEATURES),
    text_node("([yshift=6pt] target-north)", r"Target Input Samples\\(3D coordinate + distance)", options="anchor=south, align=center, font=\\small\\bfseries"),
]


# -----------
# Main Output
# -----------
arch = [to_head(PLOTNN_REL), to_cor(), to_colors(), to_begin()]
arch += target

for k in range(NUM_DRAWN_CASCADES):
    arch += cascade(k, -k * CASCADE_SPACING)

for k in range(NUM_DRAWN_CASCADES):
    arch += cascade_link(k, -k * CASCADE_SPACING, last=(k == NUM_DRAWN_CASCADES - 1))

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
