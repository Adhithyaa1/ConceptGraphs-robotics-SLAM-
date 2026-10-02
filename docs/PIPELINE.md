# lab_walk → ConceptGraphs pipeline

## Inputs

- ROS bag `lab_walk_7526_wd455_semantics` (local): Livox lidar + D455 RGB-D
- KISS-ICP on `/livox/lidar` → `odom/kiss_icp_tum.txt`
- Extrinsics → `odom/extrinsics_livox_d455.json`

## Cluster steps

1. **Extract RGB-D** — `scripts/extract_d455_rgbd_from_bag.py` → `lab_walk_rgbd_raw/` (not in zip; on Lustre)
2. **Export posed scene** — `scripts/export_kiss_lab_walk.py --link-images` → `lab_walk_kiss/` (`traj.txt`, symlinks to frames)
3. **Copy Hydra configs** from `conceptgraph/` into your `concept-graphs/conceptgraph/` tree
4. **Map** — `scripts/sbatch/map_lab_walk_kiss.sbatch` → `data/lab_walk_kiss/exps/r_mapping_lab_walk_kiss_stride10/`

Export uses `T_world_cam = T_world_livox @ T_ros_to_optical` (translation from JSON, standard ROS→optical rotation).

## Outputs in this bundle

- **Object map:** `pcd_r_mapping_lab_walk_kiss_stride10.pkl.gz` (51 objects + `clip_ft`)
- **Metadata:** `obj_json_*.json`, `config_params*.json`, `objects_summary.json`
- **Viz:** `combined_objects.ply`, `map_topdown.png`, `kiss_body_topdown.png`

## Inspect (OOD Interactive Desktop)

```bash
export DISPLAY=:1   # not :0
bash scripts/ood_tutorial_steps.sh view-kiss-clip   # F = text query
bash scripts/ood_tutorial_steps.sh static-kiss
```

Mapping config: stride 10, `make_edges: false`.
