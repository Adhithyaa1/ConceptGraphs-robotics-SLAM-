#!/usr/bin/env python3
"""
Frame-by-frame pose viewer for a Replica-style scene (traj.txt + results/).

Shows:
  - camera frustum / axes at the current pose
  - trajectory path so far
  - optional current-frame RGB-D cloud (and optionally accumulate)

Controls (Open3D window must be focused):
  Right arrow / D / space  — next frame
  Left arrow  / A          — previous frame
  R                        — toggle auto-play
  C                        — toggle show current-frame cloud
  G                        — toggle accumulate clouds (heavy)
  Q / Esc                  — quit

Example (OOD Desktop, interactive):
  python cg_jobs/view_poses_realtime.py \\
    --scene /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_full_rgbd \\
    --stride 5 --play

Example (no OOD / no DISPLAY — write MP4):
  python cg_jobs/view_poses_realtime.py \\
    --scene /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_full_rgbd \\
    --stride 5 --video /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_full_rgbd/poses.mp4

Video layout (--video): top-down floor plan (X right, Y forward) +
height (−Y_opt) vs frame. Real climb only shows on the height panel.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def load_traj(path: Path) -> np.ndarray:
    mats = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mats.append(np.fromstring(line, sep=" ").reshape(4, 4))
    return np.stack(mats)


def make_intrinsic(cam: dict) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(
        int(cam["width"]),
        int(cam["height"]),
        float(cam["fx"]),
        float(cam["fy"]),
        float(cam["cx"]),
        float(cam["cy"]),
    )


def frustum_lines(T_wc: np.ndarray, size: float = 0.15) -> o3d.geometry.LineSet:
    """Simple camera pyramid in world frame (OpenCV optical: Z forward)."""
    # camera-frame corners of image plane at z=size
    w, h = size * 0.8, size * 0.6
    pts_c = np.array(
        [
            [0, 0, 0],
            [-w, -h, size],
            [w, -h, size],
            [w, h, size],
            [-w, h, size],
        ],
        dtype=np.float64,
    )
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    pts_w = (R @ pts_c.T).T + t
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [[1, 0, 0] for _ in lines]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_w),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def path_lineset(poses: np.ndarray, upto: int) -> o3d.geometry.LineSet:
    pts = poses[: upto + 1, :3, 3]
    if len(pts) < 2:
        pts = np.vstack([pts, pts])
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector([[0, 0.7, 1]] * len(lines))
    return ls


def frame_cloud(
    color_path: Path,
    depth_path: Path,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    T_wc: np.ndarray,
    depth_scale: float,
    depth_max: float,
    voxel: float,
) -> o3d.geometry.PointCloud:
    color = o3d.io.read_image(str(color_path))
    depth = o3d.io.read_image(str(depth_path))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=depth_scale,
        depth_trunc=depth_max,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd.transform(T_wc)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd


def frustum_segments(T_wc: np.ndarray, size: float = 0.2) -> np.ndarray:
    """Return (N, 2, 3) line segments for a camera frustum (optical world)."""
    w, h = size * 0.8, size * 0.6
    pts_c = np.array(
        [
            [0, 0, 0],
            [-w, -h, size],
            [w, -h, size],
            [w, h, size],
            [-w, h, size],
        ],
        dtype=np.float64,
    )
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    pts_w = (R @ pts_c.T).T + t
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    return np.stack([pts_w[list(e)] for e in edges], axis=0)


def optical_to_plot(xyz: np.ndarray) -> np.ndarray:
    """OpenCV optical (X right, Y down, Z forward) -> gravity-style plot.

    Plot axes: X = right, Y = forward (+Z_opt), Z = up (-Y_opt).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim == 1:
        return np.array([xyz[0], xyz[2], -xyz[1]], dtype=np.float64)
    return np.column_stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]])


def export_video(
    poses: np.ndarray,
    idxs: list[int],
    out_path: Path,
    fps: float,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Headless MP4 of camera path (no DISPLAY needed).

    Two panels so climb vs forward is unambiguous (avoids matplotlib 3D
    depth-axis looking like "up"):
      left  — top-down plan: X right vs Y forward (+Z_opt)
      right — height (−Y_opt) vs frame index
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xyz_opt = poses[np.array(idxs), :3, 3]
    xyz_all = optical_to_plot(xyz_opt)  # cols: right, forward, up
    heights = xyz_all[:, 2]
    frames = np.array(idxs, dtype=np.float64)

    # Fixed limits
    plan_mins = xyz_all[:, :2].min(axis=0)
    plan_maxs = xyz_all[:, :2].max(axis=0)
    plan_span = np.maximum(plan_maxs - plan_mins, 0.5)
    plan_center = 0.5 * (plan_mins + plan_maxs)
    plan_half = 0.55 * plan_span.max()
    plan_lims = np.stack([plan_center - plan_half, plan_center + plan_half], axis=1)

    h_pad = max(0.05, 0.1 * max(float(np.ptp(heights)), 0.01))
    h_lo, h_hi = float(heights.min()) - h_pad, float(heights.max()) + h_pad
    # Keep a shared scale so tiny height change looks tiny (not auto-zoomed)
    if (h_hi - h_lo) < 0.5:
        mid = 0.5 * (h_lo + h_hi)
        h_lo, h_hi = mid - 0.25, mid + 0.25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_path}")

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    # Left plan (wider) + right height strip
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.28)
    ax_plan = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[0, 1])

    for k, i in enumerate(idxs):
        ax_plan.clear()
        ax_h.clear()

        t_opt = poses[i, :3, 3]
        t = optical_to_plot(t_opt)
        height_up = float(t[2])

        fig.suptitle(
            f"frame {i:06d}  ({k+1}/{len(idxs)})  "
            f"t_opt=[{t_opt[0]:.2f},{t_opt[1]:.2f},{t_opt[2]:.2f}]  "
            f"height(−Y)={height_up:.3f}m",
            fontsize=12,
        )

        # --- plan view (top-down) ---
        ax_plan.set_xlim(*plan_lims[0])
        ax_plan.set_ylim(*plan_lims[1])
        ax_plan.set_aspect("equal", adjustable="box")
        ax_plan.set_xlabel("X right (m)")
        ax_plan.set_ylabel("Y forward = +Z_opt (m)")
        ax_plan.set_title("Top-down (floor plan)")
        ax_plan.grid(True, alpha=0.3)
        ax_plan.plot(xyz_all[:, 0], xyz_all[:, 1], color="0.75", lw=1)
        so_far = xyz_all[: k + 1]
        ax_plan.plot(so_far[:, 0], so_far[:, 1], color="deepskyblue", lw=2)
        ax_plan.scatter([t[0]], [t[1]], c="red", s=40, zorder=5)
        # forward arrow in plan
        fwd_opt = poses[i, :3, :3] @ np.array([0.0, 0.0, 0.25])
        fwd = optical_to_plot(fwd_opt)
        ax_plan.annotate(
            "",
            xy=(t[0] + fwd[0], t[1] + fwd[1]),
            xytext=(t[0], t[1]),
            arrowprops=dict(arrowstyle="->", color="lime", lw=2),
        )

        # --- height vs frame ---
        ax_h.set_xlim(frames.min() - 0.5, frames.max() + 0.5)
        ax_h.set_ylim(h_lo, h_hi)
        ax_h.set_xlabel("frame index")
        ax_h.set_ylabel("height = −Y_opt (m)")
        ax_h.set_title("Height (climb here = real climb)")
        ax_h.grid(True, alpha=0.3)
        ax_h.axhline(0.0, color="0.6", lw=1, ls="--")
        ax_h.plot(frames, heights, color="0.75", lw=1)
        ax_h.plot(frames[: k + 1], heights[: k + 1], color="deepskyblue", lw=2)
        ax_h.scatter([frames[k]], [height_up], c="red", s=40, zorder=5)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height))
        writer.write(bgr)
        if k % 20 == 0 or k + 1 == len(idxs):
            print(f"video frame {k+1}/{len(idxs)}", flush=True)

    writer.release()
    plt.close(fig)
    print(f"OK video -> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scene",
        type=Path,
        default=Path("/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_full_rgbd"),
    )
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--depth-max", type=float, default=3.0)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--play", action="store_true", help="Start in auto-play mode")
    ap.add_argument("--fps", type=float, default=5.0, help="Auto-play / video frame rate")
    ap.add_argument("--no-cloud", action="store_true", help="Start with clouds off (path+frustum only)")
    ap.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Write headless MP4 (no DISPLAY/OOD needed) and exit",
    )
    args = ap.parse_args()

    scene = args.scene
    cam = json.loads((scene / "cam_params.json").read_text())
    depth_scale = float(cam.get("png_depth_scale", cam.get("depth_scale", 1000.0)))
    intrinsic = make_intrinsic(cam)
    poses = load_traj(scene / "traj.txt")

    color_paths = sorted((scene / "results").glob("frame*.jpg"))
    depth_paths = sorted((scene / "results").glob("depth*.png"))
    n = min(len(poses), len(color_paths), len(depth_paths))
    idxs = list(range(0, n, max(1, args.stride)))
    if args.max_frames is not None:
        idxs = idxs[: args.max_frames]

    print(f"scene={scene} viewing {len(idxs)} frames (stride={args.stride})", flush=True)

    if args.video is not None:
        export_video(poses, idxs, args.video, fps=args.fps)
        return

    print("Controls: →/D/space next | ←/A prev | R play | C cloud | G accumulate | Q quit", flush=True)

    state = {
        "k": 0,
        "play": bool(args.play),
        "show_cloud": not bool(args.no_cloud),
        "accumulate": False,
        "last_tick": 0.0,
    }
    accumulated = o3d.geometry.PointCloud()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="pose frame viewer", width=1280, height=800)
    # placeholders so we can update geometry by clearing/re-adding
    geoms = {
        "path": path_lineset(poses, idxs[0]),
        "frustum": frustum_lines(poses[idxs[0]]),
        "axes": o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25),
        "cloud": o3d.geometry.PointCloud(),
        "accum": o3d.geometry.PointCloud(),
    }
    geoms["axes"].transform(poses[idxs[0]])

    for g in geoms.values():
        vis.add_geometry(g)

    def refresh():
        i = idxs[state["k"]]
        T = poses[i]
        # path
        new_path = path_lineset(poses, i)
        geoms["path"].points = new_path.points
        geoms["path"].lines = new_path.lines
        geoms["path"].colors = new_path.colors
        vis.update_geometry(geoms["path"])
        # frustum
        new_f = frustum_lines(T)
        geoms["frustum"].points = new_f.points
        geoms["frustum"].lines = new_f.lines
        geoms["frustum"].colors = new_f.colors
        vis.update_geometry(geoms["frustum"])
        # axes: reset via identity then apply — easiest recreate
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        axes.transform(T)
        geoms["axes"].vertices = axes.vertices
        geoms["axes"].triangles = axes.triangles
        geoms["axes"].vertex_colors = axes.vertex_colors
        vis.update_geometry(geoms["axes"])

        if state["show_cloud"]:
            pcd = frame_cloud(
                color_paths[i],
                depth_paths[i],
                intrinsic,
                T,
                depth_scale,
                args.depth_max,
                args.voxel,
            )
            geoms["cloud"].points = pcd.points
            geoms["cloud"].colors = pcd.colors
            if state["accumulate"]:
                accumulated.points = o3d.utility.Vector3dVector(
                    np.vstack([np.asarray(accumulated.points), np.asarray(pcd.points)])
                    if len(accumulated.points)
                    else np.asarray(pcd.points)
                )
                if len(accumulated.colors) or len(pcd.colors):
                    accumulated.colors = o3d.utility.Vector3dVector(
                        np.vstack([np.asarray(accumulated.colors), np.asarray(pcd.colors)])
                        if len(accumulated.colors)
                        else np.asarray(pcd.colors)
                    )
                if args.voxel > 0 and len(accumulated.points) > 0:
                    tmp = accumulated.voxel_down_sample(args.voxel)
                    accumulated.points = tmp.points
                    accumulated.colors = tmp.colors
                geoms["accum"].points = accumulated.points
                geoms["accum"].colors = accumulated.colors
            else:
                geoms["accum"].points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                geoms["accum"].colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        else:
            geoms["cloud"].points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            geoms["cloud"].colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            geoms["accum"].points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            geoms["accum"].colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        vis.update_geometry(geoms["cloud"])
        vis.update_geometry(geoms["accum"])
        t = T[:3, 3]
        print(
            f"\r[{state['k']+1}/{len(idxs)}] frame={i:06d}  "
            f"t=[{t[0]:.2f},{t[1]:.2f},{t[2]:.2f}]  "
            f"play={state['play']} cloud={state['show_cloud']} accum={state['accumulate']}   ",
            end="",
            flush=True,
        )

    def step(delta: int):
        state["k"] = int(np.clip(state["k"] + delta, 0, len(idxs) - 1))
        refresh()
        return False

    vis.register_key_callback(ord(" "), lambda vis: step(1))
    vis.register_key_callback(ord("D"), lambda vis: step(1))
    vis.register_key_callback(ord("A"), lambda vis: step(-1))
    # GLFW_KEY_RIGHT=262, LEFT=263
    vis.register_key_callback(262, lambda vis: step(1))
    vis.register_key_callback(263, lambda vis: step(-1))

    def toggle_play(vis):
        state["play"] = not state["play"]
        print(f"\nplay={state['play']}", flush=True)
        return False

    def toggle_cloud(vis):
        state["show_cloud"] = not state["show_cloud"]
        refresh()
        return False

    def toggle_accum(vis):
        state["accumulate"] = not state["accumulate"]
        if not state["accumulate"]:
            accumulated.clear()
        refresh()
        return False

    def quit_vis(vis):
        vis.close()
        return False

    vis.register_key_callback(ord("R"), toggle_play)
    vis.register_key_callback(ord("C"), toggle_cloud)
    vis.register_key_callback(ord("G"), toggle_accum)
    vis.register_key_callback(ord("Q"), quit_vis)
    vis.register_key_callback(256, quit_vis)  # Esc

    refresh()
    # Keep window alive; drive autoplay in the loop
    while True:
        ok = vis.poll_events()
        vis.update_renderer()
        if not ok:
            break
        if state["play"]:
            now = time.time()
            if now - state["last_tick"] >= 1.0 / max(args.fps, 0.1):
                state["last_tick"] = now
                if state["k"] >= len(idxs) - 1:
                    state["play"] = False
                    print("\nreached end", flush=True)
                else:
                    step(1)
        time.sleep(0.01)

    vis.destroy_window()
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
