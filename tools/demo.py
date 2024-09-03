from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
import pypose as pp
import rerun as rr
from mini_dpvo.api.inference import inference_dpvo
from mini_dpvo.config import cfg as base_cfg

import colmap_utils
from trajectory_io import TrajectoryIO


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--network-path", type=str, default="checkpoints/dpvo.pth")
    parser.add_argument("--imagedir", type=str)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--buffer", type=int, default=2048)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--outputdir", default=".")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini_dpvo")

    base_cfg.merge_from_file(args.config)
    base_cfg.BUFFER_SIZE = args.buffer

    print("Running with config...")
    print(base_cfg)

    dpvo_pred, total_time, intrinsics, points_filtered, colors_filtered = inference_dpvo(
        cfg=base_cfg,
        network_path=args.network_path,
        imagedir=args.imagedir,
        calib=args.calib,
        stride=args.stride,
        skip=args.skip,
    )
    rr.script_teardown(args)

    ##############################
    # Save results to COLMAP format
    ##############################

    # Save trajectory to file
    save_path, images_save_dir, _, sparse_save_dir = \
        colmap_utils.init_filestructure(Path(args.outputdir))
    traj_file = save_path / "traj.txt"
    TrajectoryIO.write_tum_trajectory(
        filename=traj_file,
        timestamps=dpvo_pred.tstamps,
        poses=dpvo_pred.final_poses,
    )
    print(f"Saved trajectory to {traj_file}")

    num_tracked_poses = dpvo_pred.final_poses.shape[0]

    # Save images
    img_exts = ["*.png", "*.jpeg", "*.jpg"]
    images_list = sorted(
        chain.from_iterable(
            Path(args.imagedir).glob(e) for e in img_exts
        )
    )[args.skip::args.stride][:num_tracked_poses]
    colmap_utils.save_images(images_save_dir, images_list)

    # Save cameras
    world2cam = pp.SE3(dpvo_pred.final_w2c).matrix().detach().cpu().numpy()
    focals = intrinsics[0:2][None,]
    principal_points = intrinsics[2:4][None,]
    img0 = cv2.imread(str(images_list[0]))
    imgs_shape = (len(images_list), img0.shape[0], img0.shape[1])
    colmap_utils.save_cameras(
        focals, principal_points, sparse_save_dir, imgs_shape=imgs_shape
    )
    assert world2cam.shape[0] == len(images_list)
    colmap_utils.save_images_txt(world2cam, images_list, sparse_save_dir)

    # Save pointcloud
    colors, vertices = colmap_utils.save_pointcloud_with_normals(
        sparse_save_dir,
        points_filtered,
        colors=colors_filtered,
        imgs=None,
    )
    points3d = {}
    for i in range(len(points_filtered)):
        points3d[i] = colmap_utils.Point3D(
            id=i,
            xyz=vertices[i],
            rgb=colors[i][:3],
            error=np.array([0]),
            image_ids=np.array([0]),
            point2D_idxs=np.array([0]),
        )
    pcl_bin_save_path = sparse_save_dir / "points3D.bin"
    colmap_utils.write_points3D_binary(points3d, pcl_bin_save_path)
