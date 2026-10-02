#!/usr/bin/env python3
"""Headless export of a ConceptGraphs pcd_*.pkl.gz map (no Open3D GUI).

Writes:
  - combined_objects.ply
  - map_topdown.png
  - objects_summary.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from conceptgraph.slam.slam_classes import MapObjectList


def load_map(pkl_path: Path) -> tuple[MapObjectList, dict]:
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    objects = MapObjectList()
    objects.load_serializable(data["objects"])
    return objects, data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--result-path",
        type=Path,
        required=True,
        help="Path to pcd_*.pkl.gz",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as pkl)",
    )
    args = ap.parse_args()

    pkl_path = Path(args.result_path).resolve()
    out_dir = args.out_dir or pkl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    objects, raw = load_map(pkl_path)
    merged = o3d.geometry.PointCloud()
    summaries = []

    for i, obj in enumerate(objects):
        pcd = obj["pcd"]
        if len(pcd.points) == 0:
            continue
        merged += pcd
        center = np.asarray(pcd.points).mean(axis=0)
        summaries.append(
            {
                "index": i,
                "object_tag": obj.get("object_tag"),
                "class_name": obj.get("class_name"),
                "num_detections": int(obj.get("num_detections", 0)),
                "center_xyz": center.tolist(),
            }
        )

    ply_path = out_dir / "combined_objects.ply"
    o3d.io.write_point_cloud(str(ply_path), merged)
    print(f"wrote {ply_path} ({len(merged.points)} points)")

    centers = np.array([s["center_xyz"] for s in summaries], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(centers[:, 0], centers[:, 1], s=20, c=np.arange(len(centers)), cmap="turbo")
    for s in summaries[:20]:
        ax.annotate(
            str(s.get("object_tag") or s["index"]),
            (s["center_xyz"][0], s["center_xyz"][1]),
            fontsize=7,
            alpha=0.8,
        )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_title(
        f"CG object centers top-down ({centers[:, 0].ptp():.1f} x {centers[:, 1].ptp():.1f} m, n={len(summaries)})"
    )
    png_path = out_dir / "map_topdown.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")

    summary_path = out_dir / "objects_summary.json"
    summary = {
        "source_pkl": str(pkl_path),
        "num_objects": len(summaries),
        "span_xyz": centers.ptp(axis=0).tolist() if len(centers) else [0, 0, 0],
        "objects": summaries,
        "class_names_count": len(raw.get("class_names", [])),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")
    print("STATIC_EXPORT_OK")


if __name__ == "__main__":
    main()
