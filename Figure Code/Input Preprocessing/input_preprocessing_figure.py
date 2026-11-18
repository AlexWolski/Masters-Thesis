import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pyrender
import trimesh
from PIL import Image
from skimage.measure import marching_cubes

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTNN_REL = os.path.join("..", "PlotNeuralNet")
sys.path.append(os.path.join(_THIS_DIR, PLOTNN_REL))

from pycore.tikzeng import to_head, to_begin, to_end, to_generate


CUBE_HALF_SIZE = 0.6
CYLINDER_RADIUS = 0.57
CYLINDER_CENTER = np.array([CUBE_HALF_SIZE, 0.0, 0.0])
QUERY_POINT = np.array([-1.8, 0.0, 1.6])

ELEVATION = 45.0
AZIMUTH = 30.0
Y_FOV = np.pi / 6.0
UP = np.array([0.0, 1.0, 0.0])
IMAGE_SIZE = 2400
OUTPUT_IMAGE_SIZE = 1200
FRAME_MARGIN = 1.04
GRID_SPACING = 0.01
NEAREST_POINT_GRID = 81
NEAREST_POINT_REFINEMENTS = 4

BLUE = [0.1, 0.2, 0.9, 1.0]
GREEN = [0.1, 0.65, 0.15, 1.0]
RED = [0.9, 0.1, 0.1, 1.0]

CM_PER_PIXEL = 0.00165
CROP_PADDING = 6
PANEL_GAP = 0.5
ROW_GAP = 0.5
TOP_SHAPE_GAP = 2.2
QUERY_LABEL_MARGIN = 0.3
LABEL_GAP = "3pt"
QUERY_POINT_RADIUS = "1.5pt"
DISTANCE_LINE_STYLE = "densely dotted, line width=1.1pt"


def cube_sdf(points):
    offset = np.abs(points) - CUBE_HALF_SIZE
    return np.linalg.norm(np.maximum(offset, 0.0), axis=-1) + np.minimum(offset.max(axis=-1), 0.0)


def cylinder_sdf(points):
    radial = np.linalg.norm((points - CYLINDER_CENTER)[..., [0, 2]], axis=-1) - CYLINDER_RADIUS
    height = np.abs(points[..., 1] - CYLINDER_CENTER[1]) - CUBE_HALF_SIZE
    offset = np.stack([radial, height], axis=-1)
    return np.linalg.norm(np.maximum(offset, 0.0), axis=-1) + np.minimum(offset.max(axis=-1), 0.0)


def missing_sdf(points):
    return np.maximum(cube_sdf(points), -cylinder_sdf(points))


def filled_sdf(points):
    return np.maximum(cube_sdf(points), cylinder_sdf(points))


def excess_sdf(points):
    return np.maximum(-cube_sdf(points), cylinder_sdf(points))


CYLINDER_EXTENT = np.array([CYLINDER_RADIUS, CUBE_HALF_SIZE, CYLINDER_RADIUS])
SCENE_LOW = np.minimum(-CUBE_HALF_SIZE, CYLINDER_CENTER - CYLINDER_EXTENT) - 2.5 * GRID_SPACING
SCENE_HIGH = np.maximum(CUBE_HALF_SIZE, CYLINDER_CENTER + CYLINDER_EXTENT) + 2.5 * GRID_SPACING


def volume_mesh(sdf):
    axes = [np.arange(SCENE_LOW[axis], SCENE_HIGH[axis], GRID_SPACING) for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    values = sdf(grid.reshape(-1, 3)).reshape(grid.shape[:3])
    vertices, faces, _, _ = marching_cubes(values, level=0.0, spacing=(GRID_SPACING,) * 3)
    mesh = trimesh.Trimesh(vertices + SCENE_LOW, faces)
    if mesh.volume < 0.0:
        mesh.invert()
    return mesh


def camera_direction():
    elevation = np.radians(ELEVATION)
    azimuth = np.radians(AZIMUTH)
    return np.array([np.sin(azimuth) * np.cos(elevation), np.sin(elevation), np.cos(azimuth) * np.cos(elevation)])


def camera_basis(direction):
    forward = -direction
    right = np.cross(forward, UP)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return right, true_up, forward


def camera_pose(eye, direction):
    right, true_up, forward = camera_basis(direction)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def camera_eye():
    direction = camera_direction()
    right, true_up, forward = camera_basis(direction)
    corners = np.array([[x, y, z] for x in (SCENE_LOW[0], SCENE_HIGH[0]) for y in (SCENE_LOW[1], SCENE_HIGH[1]) for z in (SCENE_LOW[2], SCENE_HIGH[2])])
    centre = 0.5 * (SCENE_LOW + SCENE_HIGH)
    offset = np.maximum(np.abs((corners - centre) @ right), np.abs((corners - centre) @ true_up))
    distance = (offset / np.tan(Y_FOV / 2.0) - (corners - centre) @ forward).max() * FRAME_MARGIN
    return centre + direction * distance


def project(points, eye):
    right, true_up, forward = camera_basis(camera_direction())
    relative = points - eye
    scale = np.tan(Y_FOV / 2.0) * (relative @ forward)
    return np.stack([(relative @ right / scale + 1.0) * 0.5 * OUTPUT_IMAGE_SIZE, (1.0 - relative @ true_up / scale) * 0.5 * OUTPUT_IMAGE_SIZE], axis=-1)


def render(mesh, colour, eye):
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[1.0, 1.0, 1.0, 1.0])
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=colour, metallicFactor=0.0, roughnessFactor=0.7)
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False))
    pose = camera_pose(eye, camera_direction())
    scene.add(pyrender.PerspectiveCamera(yfov=Y_FOV), pose=pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0), pose=pose)
    renderer = pyrender.OffscreenRenderer(IMAGE_SIZE, IMAGE_SIZE)
    colour_image, _ = renderer.render(scene)
    renderer.delete()
    return Image.fromarray(colour_image).convert("RGB").resize((OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), Image.LANCZOS)


def content_box(image):
    rows, columns = np.nonzero((np.asarray(image) < 255).any(axis=-1))
    return np.array([columns.min(), rows.min(), columns.max() + 1, rows.max() + 1])


def padded_box(box):
    return np.array([box[0] - CROP_PADDING, box[1] - CROP_PADDING, box[2] + CROP_PADDING, box[3] + CROP_PADDING])


def nearest_point(sdf, point):
    center = 0.5 * (SCENE_LOW + SCENE_HIGH)
    half_size = 0.5 * (SCENE_HIGH - SCENE_LOW)
    for _ in range(NEAREST_POINT_REFINEMENTS):
        axes = [np.linspace(center[axis] - half_size[axis], center[axis] + half_size[axis], NEAREST_POINT_GRID) for axis in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = grid[sdf(grid) <= 0.0]
        center = inside[np.argmin(np.linalg.norm(inside - point, axis=-1))]
        half_size = half_size * 4.0 / (NEAREST_POINT_GRID - 1)
    return center


def coordinate(origin, pixel):
    return "({:.4f}, {:.4f})".format(origin[0] + pixel[0] * CM_PER_PIXEL, origin[1] - pixel[1] * CM_PER_PIXEL)


def image_node(origin, image, box, file_name):
    image.crop(tuple(box)).save(os.path.join(_THIS_DIR, file_name))
    return ("\n\\node[anchor=north west, inner sep=0pt] at " + coordinate(origin, box[:2])
        + " {\\includegraphics[width=" + "{:.4f}".format((box[2] - box[0]) * CM_PER_PIXEL) + "cm]{" + file_name + "}};\n")


def shape_label(origin, content, bottom, text):
    return ("\n\\node[anchor=north, font=\\footnotesize\\bfseries, inner sep=0pt, yshift=-" + LABEL_GAP + "] at "
        + coordinate(origin, (0.5 * (content[0] + content[2]), bottom)) + " {" + text + "};\n")


def distance_line(origin, query, surface, label):
    return ("\n\\draw[" + DISTANCE_LINE_STYLE + "] " + coordinate(origin, query) + " -- node[pos=0.45, above, font=\\small] {$" + label + "$} "
        + coordinate(origin, surface) + ";\n"
        + "\\fill " + coordinate(origin, query) + " circle (" + QUERY_POINT_RADIUS + ") node[left, font=\\small] {$p$};\n")


def to_preamble():
    return r"""
\usepackage[T1]{fontenc}
\usepackage{lmodern}
"""


def main():
    eye = camera_eye()
    query = project(QUERY_POINT[None], eye)[0]

    bottom_panels = [
        ("missing.png", missing_sdf, BLUE, "Missing", "d_{missing}"),
        ("filled.png", filled_sdf, GREEN, "Filled", "d_{filled}"),
        ("excess.png", excess_sdf, RED, "Excess", "d_{excess}"),
    ]
    images = [render(volume_mesh(sdf), colour, eye) for _, sdf, colour, _, _ in bottom_panels]
    contents = [content_box(image) for image in images]
    crop = padded_box(np.concatenate([np.min(contents, axis=0)[:2], np.max(contents, axis=0)[2:]]))
    panel_left = query[0] - QUERY_LABEL_MARGIN / CM_PER_PIXEL
    panel_width = (crop[2] - panel_left) * CM_PER_PIXEL
    row_width = 3 * panel_width + 2 * PANEL_GAP

    bottom = []
    for index, ((file_name, sdf, _, title, distance_label), image, content) in enumerate(zip(bottom_panels, images, contents)):
        origin = (-0.5 * row_width + index * (panel_width + PANEL_GAP) - panel_left * CM_PER_PIXEL, crop[1] * CM_PER_PIXEL)
        surface = project(nearest_point(sdf, QUERY_POINT)[None], eye)[0]
        bottom += [
            image_node(origin, image, crop, file_name),
            distance_line(origin, query, surface, distance_label),
            shape_label(origin, content, crop[3], title),
        ]

    target_image = render(volume_mesh(cube_sdf), BLUE, eye)
    reconstruction_image = render(volume_mesh(cylinder_sdf), BLUE, eye)
    target_content = content_box(target_image)
    reconstruction_content = content_box(reconstruction_image)
    target_crop = padded_box(target_content)
    reconstruction_crop = padded_box(reconstruction_content)
    top_crop_bottom = max(target_crop[3], reconstruction_crop[3])
    top_y = ROW_GAP + top_crop_bottom * CM_PER_PIXEL
    target_width = (target_crop[2] - target_crop[0]) * CM_PER_PIXEL
    reconstruction_width = (reconstruction_crop[2] - reconstruction_crop[0]) * CM_PER_PIXEL
    top_width = target_width + TOP_SHAPE_GAP + reconstruction_width
    target_origin = (-0.5 * top_width - target_crop[0] * CM_PER_PIXEL, top_y)
    reconstruction_origin = (-0.5 * top_width + target_width + TOP_SHAPE_GAP - reconstruction_crop[0] * CM_PER_PIXEL, top_y)
    top = [
        image_node(target_origin, target_image, target_crop, "target.png"),
        shape_label(target_origin, target_content, top_crop_bottom, "Target Shape"),
        image_node(reconstruction_origin, reconstruction_image, reconstruction_crop, "reconstruction.png"),
        shape_label(reconstruction_origin, reconstruction_content, top_crop_bottom, "Initial Reconstruction"),
    ]

    arch = [to_head(PLOTNN_REL), to_preamble(), to_begin()]
    arch += top
    arch += bottom
    arch += to_end()

    namefile = os.path.splitext(os.path.basename(__file__))[0]
    out_path = os.path.join(_THIS_DIR, namefile + ".tex")

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
