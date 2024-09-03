""" Source:
https://github.com/hwanhuh/Radiance-Fields-from-VGGSfM-Mast3r/blob/main/colmap_from_mast3r.py
"""

import collections
import numpy as np
import math
import shutil
import trimesh
from pathlib import Path
from plyfile import PlyData, PlyElement
from typing import List, NamedTuple

from device import to_numpy


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# Ensure save directories exist
def init_filestructure(save_path):
    save_path.mkdir(exist_ok=True, parents=True)
    images_path = save_path / "images"
    masks_path = save_path / "masks"
    sparse_path = save_path / "sparse/0"
    images_path.mkdir(exist_ok=True, parents=True)
    # masks_path.mkdir(exist_ok=True, parents=True)
    sparse_path.mkdir(exist_ok=True, parents=True)
    return save_path, images_path, masks_path, sparse_path


# Save images
# def save_images(imgs, images_path, img_files):
#     for i, (image, name) in enumerate(zip(imgs, img_files)):
#         imgname = Path(name).stem
#         image_save_path = images_path / f"{imgname}.png"
#         rgb_image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(str(image_save_path), rgb_image)
def save_images(images_save_dir: Path, img_files: List[Path]):
    for name in img_files:
        imgname = Path(name).stem
        image_save_path = images_save_dir / f"{imgname}.png"
        # Copy image file
        shutil.copy(name, image_save_path)


# Save camera information
def save_cameras(focals, principal_points, sparse_path, imgs_shape):
    cameras_file = sparse_path / "cameras.txt"
    with open(cameras_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            f.write(
                f"{i} PINHOLE {imgs_shape[2]} {imgs_shape[1]} {focal[0]} {focal[1]} {pp[0]} {pp[1]}\n"
            )


# Save image transformations
def save_images_txt(world2cam, img_files: List[Path], sparse_path: Path):
    images_file = sparse_path / "images.txt"
    with open(images_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        CAMERA_ID = 0
        for i in range(world2cam.shape[0]):
            name = img_files[i].stem
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            f.write(
                f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {CAMERA_ID} {name}.png\n\n"
            )


# Save point cloud with normals
def save_pointcloud_with_normals(
    sparse_path, pts3d, colors=None, imgs=None, masks=None
):
    if colors is not None:
        colors = to_numpy(colors)
        vertices = to_numpy(pts3d)
    elif imgs is not None:
        pc = get_point_cloud(imgs, pts3d, masks)
        colors = pc.colors
        vertices = pc.vertices
    else:
        raise ValueError("Either colors or images must be provided.")
    default_normal = [0, 1, 0]
    normals = np.tile(default_normal, (vertices.shape[0], 1))
    save_path = sparse_path / "points3D.ply"
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
end_header
""".format(
        len(vertices)
    )
    with open(save_path, "w") as f:
        f.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            f.write(
                f"{vertex[0]} {vertex[1]} {vertex[2]} {int(color[0])} {int(color[1])} {int(color[2])} {normal[0]} {normal[1]} {normal[2]}\n"
            )
    return colors, vertices


# Generate point cloud
def get_point_cloud(imgs, pts3d, mask=None):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    if mask is not None:
        mask = to_numpy(mask)
        pts3d = [p[m] for p, m in zip(pts3d, mask.reshape(mask.shape[0], -1))]
        imgs = [p[m] for p, m in zip(imgs, mask)]
    pts = np.concatenate(pts3d)
    col = np.concatenate(imgs)
    # pts = pts.reshape(-1, 3)[::3]
    # col = col.reshape(-1, 3)[::3]
    pts = pts.reshape(-1, 3)
    col = col.reshape(-1, 3)
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals
    return pct


# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)


import struct


Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")
