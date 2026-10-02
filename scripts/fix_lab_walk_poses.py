#!/usr/bin/env python3
"""
Fix lab_walk Replica-style export for ConceptGraphs.

Applies the same correction validated on lab_walk_clean:
  1) Strip exported T_livox_d455 from traj (Livox/FAST-LIO is ~Z-up).
  2) Apply standard ROS-body -> camera-optical rotation (+ export translation).
  3) Rotate RGB+depth 180 and update cx,cy so detectors see upright frames.

Usage:
  python fix_lab_walk_poses.py \
    --src /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_full_raw \
    --dst /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_full
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


# Exported (bad) extrinsics from the clean-segment export_summary
T_LIVOX_D455_EXPORTED = np.array(
    [
        [0.00681448027137242, -0.6879391836756057, -0.7257363449782386, 0.005510776300598444],
        [0.9999473395382352, -0.0008812022970267641, 0.010224560279927557, 0.017960979529907065],
        [-0.007673396186646015, -0.7257678024315033, 0.6878969515448584, 0.2530805708040512],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Standard ROS body (X fwd, Z up) -> OpenCV optical (Z fwd, Y down, X right)
R_ROS_TO_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def load_traj(path: Path) -> np.ndarray:
    mats = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mats.append(np.fromstring(line, sep=" ").reshape(4, 4))
    return np.stack(mats)


def write_traj(path: Path, traj: np.ndarray) -> None:
    with open(path, "w") as f:
        for M in traj:
            f.write(" ".join(f"{x:.8f}" for x in M.reshape(-1)) + "\n")


def make_T_livox_d455() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_ROS_TO_OPTICAL
    T[:3, 3] = T_LIVOX_D455_EXPORTED[:3, 3]
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Raw Replica-like scene dir")
    ap.add_argument("--dst", type=Path, required=True, help="Output fixed scene dir")
    ap.add_argument(
        "--assume-exported-extrinsics",
        action="store_true",
        default=True,
        help="Strip T_LIVOX_D455_EXPORTED then apply ROS->optical (default).",
    )
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    assert (src / "traj.txt").is_file(), f"missing {src}/traj.txt"
    assert (src / "results").is_dir(), f"missing {src}/results"
    cam_path = src / "cam_params.json"
    assert cam_path.is_file(), f"missing {cam_path}"

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "results").mkdir(parents=True)

    traj_raw = load_traj(src / "traj.txt")
    livox = traj_raw @ np.linalg.inv(T_LIVOX_D455_EXPORTED)
    traj = livox @ make_T_livox_d455()
    write_traj(dst / "traj.txt", traj)

    cam = json.loads(cam_path.read_text())
    W, H = int(cam["width"]), int(cam["height"])
    cam2 = dict(cam)
    cam2["cx"] = (W - 1) - float(cam["cx"])
    cam2["cy"] = (H - 1) - float(cam["cy"])
    (dst / "cam_params.json").write_text(json.dumps(cam2, indent=2))

    n = len(traj_raw)
    for i in range(n):
        rgb = cv2.imread(str(src / "results" / f"frame{i:06d}.jpg"))
        dep = cv2.imread(str(src / "results" / f"depth{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or dep is None:
            raise FileNotFoundError(f"missing frame/depth at index {i}")
        cv2.imwrite(
            str(dst / "results" / f"frame{i:06d}.jpg"),
            cv2.rotate(rgb, cv2.ROTATE_180),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        cv2.imwrite(str(dst / "results" / f"depth{i:06d}.png"), cv2.rotate(dep, cv2.ROTATE_180))
        if i % 100 == 0:
            print(f"rotated {i}/{n}", flush=True)

    ups = np.array([M[:3, :3] @ np.array([0.0, -1.0, 0.0]) for M in traj])
    fwds = np.array([M[:3, :3] @ np.array([0.0, 0.0, 1.0]) for M in traj])
    up_ang = float(np.rad2deg(np.arccos(np.clip(ups[:, 2], -1, 1))).mean())
    fwd_el = float(np.rad2deg(np.arcsin(np.clip(fwds[:, 2], -1, 1))).mean())

    summary = {
        "source": str(src),
        "frames": n,
        "fix": "strip exported T_livox_d455; apply ROS->optical; rotate RGB/depth 180; update cx,cy",
        "up_vs_worldZ_deg_mean": up_ang,
        "fwd_elevation_deg_mean": fwd_el,
        "cam_params": cam2,
    }
    (dst / "export_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"OK -> {dst}")


if __name__ == "__main__":
    main()
