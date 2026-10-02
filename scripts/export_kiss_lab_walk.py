#!/usr/bin/env python3
"""
Export posed Replica-style lab_walk_kiss from KISS-ICP TUM + D455 RGB-D.

Builds CG-ready poses from KISS TUM + D455 RGB-D:
  1) Interpolate T_world_livox from kiss_icp_tum.txt to each D455 timestamp
  2) T_world_cam = T_world_livox @ T_livox_d455_ros_optical
     (same translation as extrinsics JSON; rotation = standard ROS body -> optical,
      matching fix_lab_walk_poses — not the full TF rotation matrix, which tilts
      the camera ~47 deg and breaks floor-aligned fusion)
  3) Optional --viz-optical-fix remaps axes only for comparing to KISS body X-Y
     in poses_gravity.mp4 (not for ConceptGraphs).

Input:
  - kiss TUM (lidar poses)
  - extrinsics_livox_d455.json
  - lab_walk_rgbd_raw (upright RGB-D + timestamps)

Output:
  - lab_walk_kiss/{results,traj.txt,cam_params.json,export_summary.json}
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Standard ROS body (X fwd, Z up) -> OpenCV optical (Z fwd, Y down, X right)
R_ROS_TO_OPTICAL = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# Viz-only: body horizontal -> optical plan (do not use for CG export)
R_VIZ_BODY_TO_OPTICAL = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, poses = [], []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        ts.append(float(p[0]))
        t = np.array(list(map(float, p[1:4])))
        q = np.array(list(map(float, p[4:8])))
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q).as_matrix()
        T[:3, 3] = t
        poses.append(T)
    return np.array(ts, dtype=np.float64), np.stack(poses)


def interp_pose(ts: np.ndarray, poses: np.ndarray, t: float) -> np.ndarray:
    if t <= ts[0]:
        return poses[0].copy()
    if t >= ts[-1]:
        return poses[-1].copy()
    i = int(np.searchsorted(ts, t) - 1)
    t0, t1 = ts[i], ts[i + 1]
    a = float((t - t0) / (t1 - t0))
    T = np.eye(4)
    T[:3, 3] = (1.0 - a) * poses[i, :3, 3] + a * poses[i + 1, :3, 3]
    slerp = Slerp([0.0, 1.0], R.from_matrix([poses[i, :3, :3], poses[i + 1, :3, :3]]))
    T[:3, :3] = slerp([a]).as_matrix()[0]
    return T


def make_T_livox_d455(T_ld: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_ROS_TO_OPTICAL
    T[:3, 3] = T_ld[:3, 3]
    return T


def apply_viz_optical_fix(T: np.ndarray) -> np.ndarray:
    Rm = np.eye(4)
    Rm[:3, :3] = R_VIZ_BODY_TO_OPTICAL
    return Rm @ T


def write_traj(path: Path, poses: list[np.ndarray]) -> None:
    with open(path, "w") as f:
        for M in poses:
            f.write(" ".join(f"{v:.8f}" for v in M.reshape(-1)) + "\n")


def write_kiss_body_topdown(path: Path, kiss: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = kiss[:, :2, 3]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xy[:, 0], xy[:, 1], lw=1, color="deepskyblue")
    ax.scatter([xy[0, 0]], [xy[0, 1]], c="green", s=40, label="start", zorder=5)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], c="red", s=40, label="end", zorder=5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("KISS body X (m)")
    ax.set_ylabel("KISS body Y (m)")
    ax.set_title(f"KISS-ICP body top-down ({xy[:, 0].ptp():.2f} x {xy[:, 1].ptp():.2f} m)")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tum",
        type=Path,
        default=Path("/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/kiss_icp_tum.txt"),
    )
    ap.add_argument(
        "--extrinsics",
        type=Path,
        default=Path("/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/extrinsics_livox_d455.json"),
    )
    ap.add_argument(
        "--rgbd-src",
        type=Path,
        default=Path("/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_rgbd_raw"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_kiss"),
    )
    ap.add_argument(
        "--sync-ms",
        type=float,
        default=50.0,
        help="Max |color_stamp - depth_stamp| to keep a pair (already filtered in raw)",
    )
    ap.add_argument(
        "--viz-optical-fix",
        action="store_true",
        help="Viz only: remap axes to approximate KISS body X-Y in poses_gravity.mp4",
    )
    ap.add_argument(
        "--use-tf-rotation",
        action="store_true",
        help="Use full TF rotation from extrinsics JSON (not recommended for CG)",
    )
    ap.add_argument(
        "--link-images",
        action="store_true",
        help="Symlink results/ from rgbd-src instead of copying",
    )
    args = ap.parse_args()

    ext = json.loads(args.extrinsics.read_text())
    T_ld = np.array(ext["T_livox_d455_optical"]["matrix_row_major_4x4"], dtype=np.float64)
    T_cam = T_ld if args.use_tf_rotation else make_T_livox_d455(T_ld)

    ts, kiss = load_tum(args.tum)
    ts_path = args.rgbd_src / "timestamps.txt"
    color_ts = []
    for line in ts_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        color_ts.append(float(parts[1]) / 1e9)
    color_ts = np.array(color_ts)

    out = args.out
    results = out / "results"
    if out.exists():
        shutil.rmtree(out)
    results.mkdir(parents=True)

    src_results = args.rgbd_src / "results"
    n = min(len(color_ts), len(list(src_results.glob("frame*.jpg"))))
    kept = 0
    skipped = 0
    poses: list[np.ndarray] = []

    for i in range(n):
        t = color_ts[i]
        if t < ts[0] or t > ts[-1]:
            skipped += 1
            continue
        T_wl = interp_pose(ts, kiss, t)
        T_wc = T_wl @ T_cam
        if args.viz_optical_fix:
            T_wc = apply_viz_optical_fix(T_wc)
        poses.append(T_wc)

        frame = f"frame{i:06d}.jpg"
        depth = f"depth{i:06d}.png"
        if args.link_images:
            (results / frame).symlink_to((src_results / frame).resolve())
            (results / depth).symlink_to((src_results / depth).resolve())
        else:
            shutil.copy2(src_results / frame, results / frame)
            shutil.copy2(src_results / depth, results / depth)
        kept += 1

    write_traj(out / "traj.txt", poses)
    shutil.copy2(args.rgbd_src / "cam_params.json", out / "cam_params.json")

    xyz = np.stack([p[:3, 3] for p in poses])
    height = -xyz[:, 1]
    summary = {
        "frames": kept,
        "skipped": skipped,
        "tum_poses": int(len(kiss)),
        "extrinsic_mode": "tf_rotation" if args.use_tf_rotation else "ros_to_optical",
        "viz_optical_fix": args.viz_optical_fix,
        "T_livox_d455_tf": T_ld.tolist(),
        "T_livox_d455_used": T_cam.tolist(),
        "R_ros_to_optical": R_ROS_TO_OPTICAL.tolist(),
        "z_min": float(xyz[:, 2].min()),
        "z_max": float(xyz[:, 2].max()),
        "height_min": float(height.min()),
        "height_max": float(height.max()),
        "height_range": float(height.max() - height.min()),
        "plan_span_x": float(xyz[:, 0].ptp()),
        "plan_span_forward_z": float(xyz[:, 2].ptp()),
        "out": str(out),
    }
    (out / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_kiss_body_topdown(out / "kiss_body_topdown.png", kiss)
    print(json.dumps(summary, indent=2), flush=True)
    print("EXPORT_OK", flush=True)


if __name__ == "__main__":
    main()
