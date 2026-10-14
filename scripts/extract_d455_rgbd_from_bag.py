#!/usr/bin/env python3
"""
Stage 1: extract synced D455 RGB-D (+ camera_info) from a ROS 2 bag
into a Replica-like folder for ConceptGraphs / Open3D odometry.

Topics used:
  /D455/color/image_raw
  /D455/aligned_depth_to_color/image_raw
  /D455/aligned_depth_to_color/camera_info

Output (default):
  <out>/
    results/frameXXXXXX.jpg
    results/depthXXXXXX.png
    cam_params.json
    timestamps.txt
    extract_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

COLOR_TOPIC = "/D455/color/image_raw"
DEPTH_TOPIC = "/D455/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/D455/aligned_depth_to_color/camera_info"

PNG_DEPTH_SCALE = 1000.0  # RealSense depth in mm → meters


def image_msg_to_numpy(msg) -> np.ndarray:
    """Convert sensor_msgs/Image to HxW[xC] numpy array."""
    h, w = int(msg.height), int(msg.width)
    enc = str(msg.encoding).lower()
    raw = bytes(msg.data)

    if enc in ("rgb8", "bgr8"):
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if enc in ("rgba8", "bgra8"):
        img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        if enc == "rgba8":
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc in ("mono8", "8uc1"):
        return np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
    if enc in ("16uc1", "mono16"):
        return np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
    if enc == "32fc1":
        depth_m = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        return np.clip(depth_m * PNG_DEPTH_SCALE, 0, 65535).astype(np.uint16)
    raise ValueError(f"unsupported image encoding: {msg.encoding!r}")


def camera_info_to_params(msg) -> dict:
    k = list(msg.k)
    return {
        "fx": float(k[0]),
        "fy": float(k[4]),
        "cx": float(k[2]),
        "cy": float(k[5]),
        "width": int(msg.width),
        "height": int(msg.height),
        "depth_scale": PNG_DEPTH_SCALE,
        "png_depth_scale": PNG_DEPTH_SCALE,
    }


def header_stamp_ns(msg, fallback: int) -> int:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        s = msg.header.stamp
        return int(s.sec) * 1_000_000_000 + int(s.nanosec)
    return int(fallback)


def nearest_color_idx(color_stamps: np.ndarray, t: int, max_dt_ns: int) -> int | None:
    i = int(np.searchsorted(color_stamps, t))
    best, best_dt = None, max_dt_ns + 1
    for j in (i - 1, i):
        if 0 <= j < len(color_stamps):
            dt = abs(int(color_stamps[j]) - t)
            if dt < best_dt:
                best, best_dt = j, dt
    if best is None or best_dt > max_dt_ns:
        return None
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bag",
        type=Path,
        default=Path("/lustre/nvwulf/home/admanoharan/lab_walk_7526_wd455_semantics"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_rgbd_raw"),
    )
    ap.add_argument(
        "--max-dt-ms",
        type=float,
        default=30.0,
        help="Max |color_stamp - depth_stamp| to accept a pair.",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth synced pair (after sync).",
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional seconds from first depth message to keep.",
    )
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    bag: Path = args.bag
    out: Path = args.out
    max_dt_ns = int(args.max_dt_ms * 1e6)
    stride = max(1, int(args.stride))

    if not bag.exists():
        raise FileNotFoundError(bag)

    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.LATEST)
    color_msgs: list[tuple[int, object]] = []
    depth_msgs: list[tuple[int, object]] = []
    cam_params: dict | None = None

    topics = {COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC}
    print(f"Reading bag: {bag}", flush=True)
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        missing = topics - {c.topic for c in conns}
        if missing:
            raise RuntimeError(f"bag missing topics: {sorted(missing)}")

        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            t = header_stamp_ns(msg, timestamp)

            if connection.topic == COLOR_TOPIC:
                color_msgs.append((t, msg))
            elif connection.topic == DEPTH_TOPIC:
                depth_msgs.append((t, msg))
            elif connection.topic == INFO_TOPIC and cam_params is None:
                cam_params = camera_info_to_params(msg)
                print(f"cam_params from {INFO_TOPIC}: {cam_params}", flush=True)

    if cam_params is None:
        raise RuntimeError(f"no messages on {INFO_TOPIC}")
    if not color_msgs or not depth_msgs:
        raise RuntimeError(
            f"empty streams: color={len(color_msgs)} depth={len(depth_msgs)}"
        )

    color_msgs.sort(key=lambda x: x[0])
    depth_msgs.sort(key=lambda x: x[0])
    color_stamps = np.array([t for t, _ in color_msgs], dtype=np.int64)

    t0 = depth_msgs[0][0]
    max_duration_ns = (
        None if args.max_duration is None else int(float(args.max_duration) * 1e9)
    )

    pairs: list[tuple[int, object, int, object]] = []
    used_color: set[int] = set()
    skipped_sync = 0
    skipped_duration = 0
    for depth_t, depth_msg in depth_msgs:
        if max_duration_ns is not None and (depth_t - t0) > max_duration_ns:
            skipped_duration += 1
            continue
        ci = nearest_color_idx(color_stamps, depth_t, max_dt_ns)
        if ci is None or ci in used_color:
            skipped_sync += 1
            continue
        used_color.add(ci)
        color_t, color_msg = color_msgs[ci]
        pairs.append((color_t, color_msg, depth_t, depth_msg))

    pairs = pairs[::stride]
    print(
        f"synced pairs: {len(pairs)} "
        f"(raw depth={len(depth_msgs)} color={len(color_msgs)}; "
        f"skip_sync={skipped_sync} skip_duration={skipped_duration} stride={stride})",
        flush=True,
    )

    ts_lines: list[str] = []
    for i, (color_t, color_msg, depth_t, depth_msg) in enumerate(pairs):
        color = image_msg_to_numpy(color_msg)
        depth = image_msg_to_numpy(depth_msg)
        if color.ndim != 3 or depth.ndim != 2:
            raise RuntimeError(
                f"bad shapes at {i}: color={getattr(color, 'shape', None)} "
                f"depth={getattr(depth, 'shape', None)}"
            )
        if color.shape[:2] != depth.shape[:2]:
            raise RuntimeError(
                f"resolution mismatch at {i}: color={color.shape} depth={depth.shape}"
            )

        ok_c = cv2.imwrite(
            str(results / f"frame{i:06d}.jpg"),
            color,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
        )
        ok_d = cv2.imwrite(str(results / f"depth{i:06d}.png"), depth)
        if not ok_c or not ok_d:
            raise RuntimeError(f"failed writing frame {i}")

        ts_lines.append(f"{i} {color_t} {depth_t} {abs(color_t - depth_t)}")
        if i % 50 == 0 or i + 1 == len(pairs):
            print(f"wrote {i + 1}/{len(pairs)}", flush=True)

    (out / "cam_params.json").write_text(json.dumps(cam_params, indent=2) + "\n")
    (out / "timestamps.txt").write_text(
        "# index color_stamp_ns depth_stamp_ns abs_dt_ns\n" + "\n".join(ts_lines) + "\n"
    )

    summary = {
        "bag": str(bag),
        "out": str(out),
        "topics": {
            "color": COLOR_TOPIC,
            "depth": DEPTH_TOPIC,
            "camera_info": INFO_TOPIC,
        },
        "n_color_msgs": len(color_msgs),
        "n_depth_msgs": len(depth_msgs),
        "n_pairs_written": len(pairs),
        "skipped_sync": skipped_sync,
        "skipped_duration": skipped_duration,
        "stride": stride,
        "max_dt_ms": args.max_dt_ms,
        "max_duration": args.max_duration,
        "cam_params": cam_params,
    }
    (out / "extract_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"OK -> {out}", flush=True)


if __name__ == "__main__":
    main()
