#!/usr/bin/env python3
"""Copy fx/fy/cx/cy and image size from scene cam_params.json into a lab_walk dataconfig YAML."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cam-params",
        type=Path,
        required=True,
        help="Scene cam_params.json (from RGB-D extract or kiss export)",
    )
    ap.add_argument(
        "--dataconfig",
        type=Path,
        required=True,
        help="conceptgraph/dataset/dataconfigs/lab_walk/lab_walk_kiss_*.yaml",
    )
    args = ap.parse_args()

    cp = json.loads(args.cam_params.read_text())
    text = args.dataconfig.read_text()
    repl = {
        "image_height": int(cp.get("height", cp.get("image_height", 480))),
        "image_width": int(cp.get("width", cp.get("image_width", 640))),
        "fx": float(cp["fx"]),
        "fy": float(cp["fy"]),
        "cx": float(cp["cx"]),
        "cy": float(cp["cy"]),
    }
    for key, val in repl.items():
        if isinstance(val, float):
            text = re.sub(
                rf"^(\s*{key}:\s*).*$",
                rf"\g<1>{val}",
                text,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            text = re.sub(
                rf"^(\s*{key}:\s*).*$",
                rf"\g<1>{val}",
                text,
                count=1,
                flags=re.MULTILINE,
            )
    args.dataconfig.write_text(text)
    print(f"Updated {args.dataconfig} from {args.cam_params}")
    print(json.dumps(repl, indent=2))


if __name__ == "__main__":
    main()
