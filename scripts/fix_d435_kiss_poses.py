#!/usr/bin/env python3
"""
Fix lab_walk_d435_kiss/traj.txt camera poses using KISS-ICP TUM + standard ROS->optical rotation.

Mirrors export_kiss_lab_walk.py logic:
  T_world_cam = T_world_livox @ T_ros_to_optical
where translation comes from extrinsics_livox_d455.json, and rotation is R_ROS_TO_OPTICAL.
"""
from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

R_ROS_TO_OPTICAL = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# Physical camera tilt relative to level forward viewing:
# Pitch = -48.34 deg (downward at floor), Roll = -2.17 deg, Yaw = 4.28 deg
R_PITCH_DOWN = R.from_euler("xyz", [-48.34, -2.17, 4.28], degrees=True).as_matrix()
R_D435_OPTICAL = R_ROS_TO_OPTICAL @ R_PITCH_DOWN

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

def write_traj(path: Path, poses: list[np.ndarray]) -> None:
    with open(path, "w") as f:
        for M in poses:
            f.write(" ".join(f"{v:.8f}" for v in M.reshape(-1)) + "\n")

def main():
    lab_root = Path("/lustre/nvwulf/home/admanoharan/cg_data/lab_walk")
    tum_path = lab_root / "kiss_icp_tum.txt"
    scene_dir = lab_root / "lab_walk_d435_kiss"
    ts_file = scene_dir / "frame_timestamps.txt"
    traj_file = scene_dir / "traj.txt"
    backup_file = scene_dir / "traj_raw_tf_backup.txt"

    if not backup_file.exists() and traj_file.exists():
        shutil.copy2(traj_file, backup_file)
        print(f"Backed up original traj to {backup_file}")

    ts, kiss = load_tum(tum_path)

    # D435 translation in livox frame from extrinsics
    t_d435_livox = np.array([-0.0649453226119896, 0.03994424444752205, 0.12282736831155017], dtype=np.float64)
    T_cam = np.eye(4, dtype=np.float64)
    T_cam[:3, :3] = R_D435_OPTICAL
    T_cam[:3, 3] = t_d435_livox

    lines = ts_file.read_text().splitlines()
    frame_ts = []
    for l in lines:
        if l.startswith("#") or not l.strip():
            continue
        parts = l.split()
        sec = float(parts[3])
        nsec = float(parts[4])
        frame_ts.append(sec + nsec * 1e-9)

    poses = []
    for t in frame_ts:
        T_wl = interp_pose(ts, kiss, t)
        T_wc = T_wl @ T_cam
        poses.append(T_wc)

    write_traj(traj_file, poses)
    print(f"Wrote {len(poses)} corrected poses to {traj_file}")

    xyz = np.stack([p[:3, 3] for p in poses])
    print(f"Trajectory span: X={xyz[:, 0].ptp():.2f}m, Y={xyz[:, 1].ptp():.2f}m, Z={xyz[:, 2].ptp():.2f}m")
    print(f"Forward optical Z ray at frame 0: {np.round(poses[0][:3, 2], 3)}")

if __name__ == "__main__":
    main()
