# ConceptGraphs-robotics-SLAM-
This is an implementation of [concept graph](https://github.com/concept-graphs/concept-graphs) on a custom robotics dataset to create 3D maps that can be queried for object locations.
# lab_walk_conceptgraphs

Portable snapshot of the **KISS-ICP + D455 → ConceptGraphs** pipeline for the `lab_walk` dataset.

## Layout

```
lab_walk_conceptgraphs/
├── README.md
├── .gitignore
├── docs/PIPELINE.md
├── scripts/              # cluster jobs + export + OOD viz helpers
├── conceptgraph/         # copy into concept-graphs/conceptgraph/
├── odom/                 # KISS TUM + Livox→D455 extrinsics
└── data/lab_walk_kiss/   # traj, export metadata, final CG exps (no JPEG/PNG frames)
```

## Cluster paths (when unpacked on NVWulf)

| Item | Typical path |
|------|----------------|
| ConceptGraphs repo | `/lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph` |
| Conda env | `~/.conda/envs/conceptgraph` |
| RGB-D frames (not in zip) | `cg_data/lab_walk/lab_walk_rgbd_raw/` or `lab_walk_kiss/results/` |

## Install configs into ConceptGraphs

```bash
CG=/path/to/concept-graphs/conceptgraph
cp conceptgraph/dataset/dataconfigs/lab_walk/lab_walk_kiss.yaml "$CG/dataset/dataconfigs/lab_walk/"
cp conceptgraph/hydra_configs/lab_walk_kiss.yaml conceptgraph/hydra_configs/lab_walk_mapping_kiss.yaml "$CG/hydra_configs/"
```

## Re-run mapping (cluster)

1. Restore full scene on Lustre: `lab_walk_kiss/` with `results/` + `traj.txt` (re-run `export_kiss_lab_walk.py` if needed).
2. `sbatch scripts/sbatch/map_lab_walk_kiss.sbatch`

## View map locally

- Open `data/.../combined_objects.ply` in MeshLab / CloudCompare
- Or use ConceptGraphs `visualize_cfslam_results.py` with the included `pcd_*.pkl.gz` on a machine with GPU + display

Created from NVWulf export bundle.
