from argparse import ArgumentParser
from pathlib import Path
import rerun as rr
from mini_dpvo.api.inference import inference_dpvo
from mini_dpvo.config import cfg as base_cfg
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
    parser.add_argument("--log", default="traj.txt")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini_dpvo")

    base_cfg.merge_from_file(args.config)
    base_cfg.BUFFER_SIZE = args.buffer

    print("Running with config...")
    print(base_cfg)

    dpvo_pred, total_time = inference_dpvo(
        cfg=base_cfg,
        network_path=args.network_path,
        imagedir=args.imagedir,
        calib=args.calib,
        stride=args.stride,
        skip=args.skip,
    )
    rr.script_teardown(args)

    output_file = Path(args.log)

    TrajectoryIO.write_tum_trajectory(
        filename=output_file,
        timestamps=dpvo_pred.tstamps,
        poses=dpvo_pred.final_poses,
    )

    print(f"Saved trajectory to {output_file}")

    # TODO: save point cloud to file
    # Ref: https://github.com/hwanhuh/Radiance-Fields-from-VGGSfM-Mast3r/blob/main/colmap_from_mast3r.py
    final_points = dpvo_pred.final_points
    final_colors = dpvo_pred.final_colors
