# Agent Handoff: lab_walk → KISS-ICP → ConceptGraphs Pipeline

**Status (2026-10):** End-to-end pipeline **completed and validated** on NVWulf HPC.
- **D455 (Upright horizontal camera):** Mapped to 51 clean 3D objects with OpenCLIP text query enabled.
- **D435 (Downcast context camera):** Pitch-tilt corrected (-48.34°), floor flat at $Z \approx -1.20\text{ m}$ ($\sigma = 1.3\text{ cm}$), multi-frame filtered (`distinct_frames >= 2`) and noise-pruned to 25 objects.
- **Coordinate Frame:** Both cameras share the identical metric global LiDAR world frame from KISS-ICP on Livox (`kiss_icp_tum.txt`). Ready for multi-camera fusion.

---

## 1. Pipeline Overview & Data Flow

```
[ROS 2 Bag (Local Machine)]
   │
   ├──> KISS-ICP on /livox/lidar ──> kiss_icp_tum.txt ──(rsync)──> [Lustre Cluster]
   ├──> extract_d455_rgbd_from_bag.py ──> lab_walk_rgbd_raw/ (975 frames)
   └──> extract_d435_rgbd_from_bag.py ──> lab_walk_rgbd_raw_d435/ (1035 frames)
                                                │
[NVWulf HPC Cluster]                            │
   ┌────────────────────────────────────────────┘
   ▼
[Step 1: Posed Scene Export]
   ├── D455: export_kiss_lab_walk.py ──> lab_walk_kiss/ (traj.txt + results/ symlinks)
   └── D435: fix_d435_kiss_poses.py  ──> lab_walk_d435_kiss/ (traj.txt + results/ symlinks)
         (Applies pitch-down transform: Euler_xyz(-48.34°, -2.17°, +4.28°))
   ▼
[Step 2: 3D Open-Vocabulary Mapping (GPU)]
   ├── YOLOv8-World (yolov8l-world.pt) ──> 2D Bounding Boxes on ScanNet-200 classes
   ├── MobileSAM (mobile_sam.pt)        ──> Segment Masks
   ├── OpenCLIP (ViT-H-14)              ──> Visual & Text Embeddings
   ├── Backprojection (Depth + K + T)   ──> 3D Point Clouds
   ├── Spatial-Semantic Association     ──> Overlap + Cosine Sim matching
   ├── DBSCAN & Voxel Downsampling      ──> Denoising & clustering
   └── Temporal Multi-Frame Filtering   ──> distinct_frames >= 2 & class exclusion
   ▼
[Step 3: Outputs & Artifacts]
   ├── pcd_*.pkl.gz (MapObjectList: points, bboxes, clip_ft, class_id, image_idx)
   ├── s_detections_*/ (Cached 2D detections, masks, and features)
   ├── obj_json_*.json / edge_json_*.json
   └── export_cg_map_static.py ──> combined_objects.ply, map_topdown.png, objects_summary.json
   ▼
[Step 4: Inspection & Querying]
   ├── OOD VNC Desktop (DISPLAY=:1) ──> visualize_cfslam_results.py (Open3D GUI)
   └── Headless / Scripting         ──> static export PLY/PNG
```

---

## 2. Directory Layout & Critical File Paths

### Cluster Root & Storage
| Resource | Absolute Path |
|---|---|
| User Home | `/lustre/nvwulf/home/admanoharan/` |
| ConceptGraphs Code | `/lustre/nvwulf/home/admanoharan/concept-graphs/` |
| Job Scripts & Utilities | `/lustre/nvwulf/home/admanoharan/cg_jobs/` |
| Data Root | `/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/` |
| Conda Environment | `/lustre/nvwulf/home/admanoharan/.conda/envs/conceptgraph` |
| Slurm Binaries | `/cm/shared/apps/slurm/current/bin/` (`sbatch`, `squeue`) |

### Models & Weights
| Model | File / Cache Location |
|---|---|
| YOLOv8-World | `/lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/yolov8l-world.pt` |
| MobileSAM | `/lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/mobile_sam.pt` |
| OpenCLIP ViT-H-14 | Cached in `/lustre/nvwulf/scratch/admanoharan/cg_cache/huggingface` (symlinked via `cg_data/.cache/huggingface`) |
| CLIP ViT-B-32 (Fallback) | `/lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/weights/clip/ViT-B-32.pt` |
| ScanNet-200 Vocabulary | `/lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/scannet200_classes.txt` |

### Raw Data & Extrinsics (`cg_data/lab_walk/`)
| File / Directory | Description |
|---|---|
| `kiss_icp_tum.txt` | Canonical Livox LiDAR KISS-ICP trajectory (1035 timestamps, 129.6m loop) |
| `extrinsics_livox_d455.json` | Extrinsic transform matrices between Livox body and camera mounts |
| `lab_walk_rgbd_raw/` | D455 RGB-D pairs (975 frames: `color/`, `depth/`, `timestamps.txt`) |
| `lab_walk_rgbd_raw_d435/` | D435 RGB-D pairs (1035 frames: `cam_params.json`, `color/`, `depth/`) |

### Posed Scenes & Map Experiments
| Scene | Path | Key Files |
|---|---|---|
| **D455 Scene** | `cg_data/lab_walk/lab_walk_kiss/` | `traj.txt`, `results/` symlinks |
| **D455 Map** | `cg_data/lab_walk/lab_walk_kiss/exps/r_mapping_lab_walk_kiss_stride10/` | `pcd_*.pkl.gz` (51 objects), `combined_objects.ply`, `map_topdown.png` |
| **D435 Scene** | `cg_data/lab_walk/lab_walk_d435_kiss/` | `traj.txt` (tilt-corrected), `cam_params.json` |
| **D435 Map** | `cg_data/lab_walk/lab_walk_d435_kiss/exps/r_mapping_lab_walk_d435_kiss_stride10/` | `pcd_*.pkl.gz` (25 objects), `combined_objects.ply`, `map_topdown.png` |

---

## 3. Step-by-Step Execution Guide

### Step 0: Environment Activation
On the NVWulf login/compute node:
```bash
module load miniconda/3
module load cuda12.8/toolkit/12.8.1 2>/dev/null || true
module load gcc/13.4.0 2>/dev/null || true
source /lustre/nvwulf/software/miniconda3/etc/profile.d/conda.sh
conda activate /lustre/nvwulf/home/admanoharan/.conda/envs/conceptgraph
export PATH="/cm/shared/apps/slurm/current/bin:$PATH"
```

### Step 1: Camera Pose Setup & Geometry Corrections
- **D455 Camera:**
  $$T_{\text{world\_cam}} = T_{\text{world\_livox}} \times T_{\text{ros\_to\_optical}}$$
  Run: `python /lustre/nvwulf/home/admanoharan/cg_jobs/export_kiss_lab_walk.py --link-images`

- **D435 Camera (Downward Tilt Fix):**
  The physical D435 is pitched downward at ~48°. Without correction, objects are distorted and floor standard deviation is ~50 cm.
  The corrected rotation is:
  $$R_{\text{tilt}} = \text{Euler}_{\text{xyz}}(-48.34^\circ, -2.17^\circ, +4.28^\circ)$$
  $$R_{\text{D435\_OPTICAL}} = R_{\text{ROS\_TO\_OPTICAL}} \times R_{\text{tilt}}$$
  Run: `python /lustre/nvwulf/home/admanoharan/cg_jobs/fix_d435_kiss_poses.py`
  Result: Floor standard deviation drops to **1.3 cm** at $Z \approx -1.20\text{ m}$, perfectly leveled with the D455 world frame.

### Step 2: Hydra Configuration
Configurations live in `/lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/hydra_configs/`:
- D455: `lab_walk_mapping_kiss.yaml` + dataset config `lab_walk_kiss.yaml`
- D435: `lab_walk_mapping_kiss_d435.yaml` + dataset config `lab_walk_kiss_d435.yaml`

Key filter parameters configured in `lab_walk_mapping_kiss_d435.yaml`:
```yaml
downsample_voxel_size: 0.01
obj_pcd_max_points: 5000
dbscan_remove_noise: true
dbscan_eps: 0.1
dbscan_min_points: 10
obj_min_detections: 2
obj_min_distinct_frames: 2
excluded_classes:
  - toilet
  - luggage
  - bathroom vanity
```

### Step 3: Running ConceptGraphs Mapping
Submit via SLURM:
```bash
# D455 Mapping
sbatch /lustre/nvwulf/home/admanoharan/cg_jobs/map_lab_walk_kiss.sbatch

# D435 Mapping
sbatch /lustre/nvwulf/home/admanoharan/cg_jobs/map_lab_walk_kiss_d435.sbatch
```
- Runtime: ~6 minutes on Blackwell/B40 GPU (or < 1 minute if detection cache `s_detections_*` exists).

### Step 4: Headless Map Export & Sanity Check
Run:
```bash
python /lustre/nvwulf/home/admanoharan/cg_jobs/export_cg_map_static.py \
  --result-path /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_d435_kiss/exps/r_mapping_lab_walk_d435_kiss_stride10/pcd_r_mapping_lab_walk_d435_kiss_stride10.pkl.gz
```
Generates:
1. `combined_objects.ply`: Open3D/CloudCompare point cloud.
2. `map_topdown.png`: Top-down metric floor plan labeled by object class and index.
3. `objects_summary.json`: Detailed JSON list of objects, 3D centroids, detection counts, and distinct frame counts.

### Step 5: Interactive Open3D Visualizer
From Open OnDemand Interactive Desktop (VNC session on DISPLAY `:1`):
```bash
export DISPLAY=:1
python /lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py \
  --result_path /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_kiss/exps/r_mapping_lab_walk_kiss_stride10/pcd_r_mapping_lab_walk_kiss_stride10.pkl.gz
```
**Keyboard Controls:**
- `R`: RGB true colors
- `C`: Object class semantic colors
- `I`: Instance colors
- `F`: Query / highlight objects by class
- `B`: Toggle background points (walls/floors)

---

## 4. Key Lessons & Failure Modes (DO NOT REPEAT)

1. **Do not use FAST-LIO / RGB-D SLAM poses:**
   - Both diverged or collapsed on this walk trajectory. KISS-ICP on Livox LiDAR is the only validated trajectory.
2. **Do not use raw camera TF rotation for D435 without tilt correction:**
   - D435 is physically tilted downward by 48.34°. Assuming horizontal forward viewing rotates the map into the floor. Always maintain $R_{\text{tilt}}$ in `fix_d435_kiss_poses.py`.
3. **Do not rely solely on `obj_min_detections`:**
   - In single camera frames, SAM generates multiple overlapping sub-masks for one detection, causing spurious objects (e.g. toilet, luggage, phantom fridge) to get `num_detections = 2`.
   - Always enforce `obj_min_distinct_frames >= 2` (`len(set(image_idx)) >= 2`).
4. **Cluster has NO ROS 2:**
   - ROS bag extractions must be executed locally on a workstation with ROS 2 installed, and extracted image folders rsync'd to Lustre.
5. **VNC Display on OOD:**
   - The Open OnDemand VNC server runs on `:1`. Never export `DISPLAY=:0`.

---

## 5. Next Planned Milestone: Multi-Camera Fusion

Both D455 and D435 now share the exact same KISS-ICP world coordinate frame, ground level ($Z \approx -1.20\text{ m}$), and lab bounds:
- **D455:** Upper-body context, monitor displays, tall shelving, desks, mini fridge.
- **D435:** Lower-body context, carts (aligned to within 9–17 cm), chair bases/casters, floor continuity.
- **Next Task:** Merge `lab_walk_kiss` and `lab_walk_d435_kiss` into a single consolidated 3D scene representation.
