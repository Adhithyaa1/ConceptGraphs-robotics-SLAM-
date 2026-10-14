# ConceptGraphs-robotics-SLAM-
This is an implementation of [concept graph](https://github.com/concept-graphs/concept-graphs) on a custom robotics dataset to create 3D maps that can be queried for object locations.

# lab_walk_conceptgraphs

Portable snapshot of the **KISS-ICP + Multi-Camera (D455 + D435) → ConceptGraphs** 3D semantic mapping pipeline for the `lab_walk` dataset.

Includes **both complete 3D semantic map objects** (`pcd_*.pkl.gz`, `combined_objects.ply`, top-down maps, and metadata), odometry trajectories, calibrations, cluster launch scripts, and Hydra configs.

---

## 1. Directory Layout

```
lab_walk_conceptgraphs/
├── README.md                           # This guide
├── .gitignore                          # Pre-configured to track maps while ignoring raw frames
│
├── docs/                               # Complete pipeline documentation
│   ├── AGENT_HANDOFF.md                # Full technical handoff & data-flow reference
│   ├── PIPELINE.md                     # Operational checklist
│   └── LAB_WALK_D435_MAP.md            # D435 camera calibration & mapping specifics
│
├── odom/                               # Canonical LiDAR trajectory & sensor mounts
│   ├── kiss_icp_tum.txt                # KISS-ICP LiDAR trajectory (1035 poses, 129.6m loop)
│   ├── extrinsics_livox_d455.json      # Mount extrinsics (Livox LiDAR -> D455/D435)
│   └── README.md                       # Provenance notes
│
├── scripts/                            # Execution tools & utilities
│   ├── sbatch/
│   │   ├── map_lab_walk_kiss.sbatch       # D455 GPU mapping job (SLURM)
│   │   └── map_lab_walk_kiss_d435.sbatch  # D435 GPU mapping job (SLURM)
│   ├── fix_d435_kiss_poses.py             # D435 pitch tilt correction (-48.34 deg)
│   ├── export_kiss_lab_walk.py            # Interpolates KISS-ICP to camera timestamps
│   ├── export_cg_map_static.py            # Headless PLY, PNG, and JSON map exporter
│   ├── sync_dataconfig_cam_params.py      # Syncs camera K matrix to Hydra dataconfigs
│   ├── extract_d455_rgbd_from_bag.py      # ROS 2 bag extractor for D455
│   ├── extract_d435_rgbd_from_bag.py      # ROS 2 bag extractor for D435
│   ├── view_poses_realtime.py             # Pose verification video renderer
│   └── ood_tutorial_steps.sh              # Open OnDemand interactive VNC helpers
│
├── conceptgraph/                       # Configs & code patches for ConceptGraphs
│   ├── hydra_configs/                  # Mapping configurations
│   │   ├── base_mapping.yaml              # Includes obj_min_distinct_frames & class filters
│   │   ├── classes.yaml                   # ScanNet-200 class definitions
│   │   ├── lab_walk_kiss.yaml             # D455 scene parameters
│   │   ├── lab_walk_mapping_kiss.yaml     # D455 mapper config
│   │   ├── lab_walk_kiss_d435.yaml        # D435 scene parameters
│   │   └── lab_walk_mapping_kiss_d435.yaml# D435 mapper config with temporal filtering
│   ├── dataset/dataconfigs/lab_walk/   # Camera intrinsics (K matrices)
│   │   ├── lab_walk_kiss.yaml             # D455 intrinsics
│   │   └── lab_walk_kiss_d435.yaml        # D435 intrinsics
│   └── code_patches/                   # Git diff patch of changes to ConceptGraphs
│       └── conceptgraphs_repo_changes.patch
│
└── data/                               # Posed scenes and 3D MAP OBJECTS
    ├── lab_walk_kiss/                  # D455 forward-facing camera
    │   ├── traj.txt                    # Posed camera trajectory
    │   ├── timestamps.txt              # Frame timestamps
    │   ├── cam_params.json             # Intrinsics
    │   ├── export_summary.json         # Spatial bounding box summary
    │   ├── kiss_body_topdown.png       # Odometry top-down path
    │   └── exps/r_mapping_lab_walk_kiss_stride10/
    │       ├── pcd_r_mapping_lab_walk_kiss_stride10.pkl.gz  # 3D Semantic Map (51 objects)
    │       ├── combined_objects.ply                         # 3D Point Cloud PLY
    │       ├── map_topdown.png                              # Metric top-down floor plan
    │       ├── objects_summary.json                         # Centroids, labels, detection counts
    │       └── obj_json_r_mapping_lab_walk_kiss_stride10.json# 3D bounding box dimensions
    │
    └── lab_walk_d435_kiss/             # D435 downcast context camera
        ├── traj.txt                    # Tilt-corrected trajectory (flat floor Z ≈ -1.20m)
        ├── frame_timestamps.txt        # Frame timestamps
        ├── cam_params.json             # Intrinsics
        ├── export_summary.json         # Spatial bounding box summary
        └── exps/r_mapping_lab_walk_d435_kiss_stride10/
            ├── pcd_r_mapping_lab_walk_d435_kiss_stride10.pkl.gz  # 3D Semantic Map (25 objects)
            ├── combined_objects.ply                              # 3D Point Cloud PLY
            ├── map_topdown.png                                   # Metric top-down floor plan
            ├── objects_summary.json                              # Centroids, labels, frame counts
            └── obj_json_r_mapping_lab_walk_d435_kiss_stride10.json# 3D bounding box dimensions
```

---

## 2. Inspecting Map Objects Locally

Both 3D maps exist directly in this repository:

1. **Viewing 3D Point Clouds**:
   - Open `data/lab_walk_kiss/exps/r_mapping_lab_walk_kiss_stride10/combined_objects.ply` or `data/lab_walk_d435_kiss/exps/r_mapping_lab_walk_d435_kiss_stride10/combined_objects.ply` in **CloudCompare**, **MeshLab**, or **Open3D**.
2. **Viewing Top-Down Floor Plans**:
   - Open `map_topdown.png` in either experiment folder to see the metric $(X, Y)$ layout and annotated object classes.
3. **Inspecting Objects & Embeddings in Python**:
   ```python
   import gzip, pickle

   pkl_path = "data/lab_walk_d435_kiss/exps/r_mapping_lab_walk_d435_kiss_stride10/pcd_r_mapping_lab_walk_d435_kiss_stride10.pkl.gz"
   with gzip.open(pkl_path, "rb") as f:
       data = pickle.load(f)

   for obj in data["objects"]:
       print(obj["class_name"], obj["bbox"]["center"], len(obj["pcd_np"]), obj["clip_ft"].shape)
   ```

---

## 3. Uploading to GitHub

Once downloaded to your local machine:

```bash
cd lab_walk_conceptgraphs
git init -b main
git add .
git commit -m "feat: lab_walk conceptgraphs pipeline with D455 and D435 3D maps"
git remote add origin git@github.com:<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

*(Total repo size is ~55 MB, well within GitHub's 100 MB per-file and per-push constraints).*

---

## 4. Re-Running on a Cluster

1. Copy the Hydra YAML configs into your ConceptGraphs clone:
   ```bash
   cp conceptgraph/hydra_configs/*.yaml $CG_ROOT/conceptgraph/hydra_configs/
   cp conceptgraph/dataset/dataconfigs/lab_walk/*.yaml $CG_ROOT/conceptgraph/dataset/dataconfigs/lab_walk/
   ```
2. Apply the code patch if using standard ConceptGraphs:
   ```bash
   cd $CG_ROOT && git apply /path/to/conceptgraph/code_patches/conceptgraphs_repo_changes.patch
   ```
3. Submit mapping jobs via SLURM:
   ```bash
   sbatch scripts/sbatch/map_lab_walk_kiss.sbatch
   sbatch scripts/sbatch/map_lab_walk_kiss_d435.sbatch
   ```
