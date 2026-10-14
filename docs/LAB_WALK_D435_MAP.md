# D435-only ConceptGraphs map (same world frame as D455/KISS, separate scene)

Independent of the D455 map: own RGB-D extract, export, and mapping run.

## Prerequisites

- `kiss_icp_tum.txt` on Lustre (same trajectory as D455 map)
- `extrinsics_livox_d455.json` (uses `T_livox_d455_optical` + `T_d455_d435` for D435)
- ROS 2 bag locally (cluster has no ROS): run **extract** on a machine with the bag, then rsync `lab_walk_rgbd_raw_d435/` to Lustre

## 1. Extract D435 RGB-D (local, with bag)

```bash
conda activate conceptgraph   # needs rosbags, opencv
python /lustre/nvwulf/home/admanoharan/cg_jobs/extract_d435_rgbd_from_bag.py \
  --bag /path/to/lab_walk_7526_wd455_semantics \
  --out /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_rgbd_raw_d435
```

If extract fails on missing topics, check `ros2 bag info` for `/D435/aligned_depth_to_color/...`.

## 2. Sync Hydra intrinsics (cluster)

ConceptGraphs reads fx/fy/cx/cy from the dataconfig YAML, not `cam_params.json`:

```bash
python /lustre/nvwulf/home/admanoharan/cg_jobs/sync_dataconfig_cam_params.py \
  --cam-params /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_rgbd_raw_d435/cam_params.json \
  --dataconfig /lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/dataset/dataconfigs/lab_walk/lab_walk_kiss_d435.yaml
```

## 3. Export posed scene `lab_walk_kiss_d435` (cluster)

```bash
python /lustre/nvwulf/home/admanoharan/cg_jobs/export_kiss_lab_walk.py \
  --camera d435 \
  --link-images
```

Outputs: `cg_data/lab_walk/lab_walk_d435_kiss/{results,traj.txt,cam_params.json,...}` (posed scene; skip export if you rsync this folder from local).

Sanity: `export_summary.json` plan spans should be similar to D455 kiss export (~4.8×11 m footprint).

## 4. ConceptGraphs mapping (GPU)

```bash
sbatch /lustre/nvwulf/home/admanoharan/cg_jobs/map_lab_walk_kiss_d435.sbatch
```

Artifacts:

- `lab_walk_d435_kiss/exps/r_mapping_lab_walk_d435_kiss_stride10/pcd_*.pkl.gz`
- `obj_json_*.json`
- Detections cache: `lab_walk_kiss_d435/exps/s_detections_lab_walk_kiss_d435_stride10/`

## 5. Visualize

Same Open3D script as D455, different pkl path:

```bash
python concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py \
  --no_clip \
  --result_path /lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_kiss_d435/exps/r_mapping_lab_walk_kiss_d435_stride10/pcd_r_mapping_lab_walk_kiss_d435_stride10.pkl.gz
```

Or headless: point `export_cg_map_static.py --result-path` at that pkl.
