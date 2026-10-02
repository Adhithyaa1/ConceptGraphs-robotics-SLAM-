#!/usr/bin/env bash
# ConceptGraphs tutorial remaining steps — run on Open OnDemand Interactive Desktop only.
# Video: https://www.youtube.com/watch?v=56jEFyrqqpo
set -euo pipefail

module load miniconda/3
module load cuda12.8/toolkit/12.8.1 2>/dev/null || true
module load gcc/13.4.0 2>/dev/null || true
source /lustre/nvwulf/software/miniconda3/etc/profile.d/conda.sh
CG_ENV=/lustre/nvwulf/home/admanoharan/.conda/envs/conceptgraph
conda activate "$CG_ENV"
export PATH="$CG_ENV/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.8/toolkit/12.8.1}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=/lustre/nvwulf/home/admanoharan/cg_data/.cache/huggingface
export TORCH_HOME=/lustre/nvwulf/home/admanoharan/cg_data/.cache/torch
export YOLO_CONFIG_DIR=/lustre/nvwulf/home/admanoharan/cg_data/.cache/ultralytics

setup_ood_gui_env() {
  # Do NOT hardcode DISPLAY=:0 — OOD VNC uses :1, :2, ...
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "ERROR: DISPLAY is empty."
    echo "Open a terminal from the OOD desktop menu (xfce4-terminal), not SSH/login."
    return 1
  fi
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-${USER}}"
  mkdir -p "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR"
  export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-/usr/lib64/dri}"
}

require_gui() {
  setup_ood_gui_env || exit 1
  echo "DISPLAY=$DISPLAY"
  echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
  echo "hostname=$(hostname)"
  if ! xdpyinfo >/dev/null 2>&1; then
    echo "ERROR: xdpyinfo failed — this shell is not connected to the OOD desktop X server."
    echo "Fix: launch Terminal from inside the Interactive Desktop window, then rerun."
    exit 1
  fi
}

echo "python=$(command -v python)"
echo "CUDA_HOME=$CUDA_HOME"
python -c "import torch; import pytorch3d; print('torch cuda', torch.cuda.is_available()); print('pytorch3d ok')" || {
  echo "ERROR: pytorch3d/CUDA check failed. Do not run mapping until this passes."
  exit 1
}

cd /lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph

OOD_PKL=/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_usable_v2/exps/r_mapping_lab_walk_usable_v2_ood_rerun/pcd_r_mapping_lab_walk_usable_v2_ood_rerun.pkl.gz
FALLBACK_PKL=/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_usable_v2/exps/r_mapping_lab_walk_usable_v2_tutorial_nr/pcd_r_mapping_lab_walk_usable_v2_tutorial_nr.pkl.gz
KISS_PKL=/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_kiss/exps/r_mapping_lab_walk_kiss_stride10/pcd_r_mapping_lab_walk_kiss_stride10.pkl.gz
LATEST=/lustre/nvwulf/home/admanoharan/concept-graphs/latest_pcd_save

pick_pkl() {
  if [[ -f "$OOD_PKL" ]]; then
    echo "$OOD_PKL"
  elif [[ -f "$LATEST" ]] && [[ -s "$LATEST" ]]; then
    echo "NOTE: ood_rerun pkl missing; using latest_pcd_save -> $(readlink -f "$LATEST")" >&2
    echo "$LATEST"
  elif [[ -f "$FALLBACK_PKL" ]]; then
    echo "NOTE: ood_rerun pkl missing; using tutorial_nr map." >&2
    echo "$FALLBACK_PKL"
  else
    echo "ERROR: no saved map found." >&2
    return 1
  fi
}

case "${1:-help}" in
  check-display)
    require_gui
    echo "GUI env OK (xdpyinfo passed)."
    ;;
  static-kiss)
    echo "=== Headless map export (no Open3D window) ==="
    python /lustre/nvwulf/home/admanoharan/cg_jobs/export_cg_map_static.py \
      --result-path "$KISS_PKL"
    echo "Open: $(dirname "$KISS_PKL")/map_topdown.png"
    ;;
  view-kiss|view-kiss-noclip)
    require_gui
    PKL="$KISS_PKL"
    echo "=== Open3D kiss map (no CLIP text query) ==="
    EXTRA=(--no_clip)
    echo "result_path=$PKL"
    python /lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py \
      "${EXTRA[@]}" --result_path "$PKL"
    ;;
  view-kiss-clip)
    require_gui
    PKL="$KISS_PKL"
    echo "=== Open3D kiss map WITH CLIP (press F, then type e.g. desk) ==="
    echo "result_path=$PKL"
    python /lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py \
      --result_path "$PKL"
    ;;
  rerun)
    echo "=== Step 1: live Rerun mapping (reuses detections, ~few minutes) ==="
    python slam/rerun_realtime_mapping.py --config-name=lab_walk_mapping_ood_rerun
    echo "Done. pkl should be at: $OOD_PKL"
    ;;
  view-noclip|view-clip|view-latest)
    require_gui
    PKL="$(pick_pkl)"
    if [[ "$1" == view-clip ]]; then
      echo "=== Open3D WITH CLIP (press f then type a query) ==="
      EXTRA=()
    else
      echo "=== Open3D without CLIP (press r for RGB) ==="
      EXTRA=(--no_clip)
    fi
    echo "result_path=$PKL"
    python /lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph/scripts/visualize_cfslam_results.py \
      "${EXTRA[@]}" --result_path "$PKL"
    ;;
  edges)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "Set OPENAI_API_KEY first, e.g.: export OPENAI_API_KEY=sk-..."
      exit 1
    fi
    echo "=== Step 3: remap with make_edges=true (OpenAI) ==="
    python slam/rerun_realtime_mapping.py --config-name=lab_walk_mapping_ood_rerun \
      make_edges=true exp_suffix=r_mapping_lab_walk_usable_v2_ood_edges
    ;;
  help|*)
    cat <<'EOF'
Usage (OOD Desktop terminal only — launch from desktop menu, NOT SSH):
  bash /lustre/nvwulf/home/admanoharan/cg_jobs/ood_tutorial_steps.sh check-display
  bash /lustre/nvwulf/home/admanoharan/cg_jobs/ood_tutorial_steps.sh static-kiss
  bash /lustre/nvwulf/home/admanoharan/cg_jobs/ood_tutorial_steps.sh view-kiss
  bash /lustre/nvwulf/home/admanoharan/cg_jobs/ood_tutorial_steps.sh view-kiss-clip
  bash /lustre/nvwulf/home/admanoharan/cg_jobs/ood_tutorial_steps.sh rerun
  bash /lustre/nvwulf/home/admanoharan/cg_jobs/ood_tutorial_steps.sh view-noclip
  bash /lustre/nvwulf/home/admanoharan/cg_jobs/ood_tutorial_steps.sh view-clip
  bash /lustre/nvwulf/home/admanoharan/cg_jobs/ood_tutorial_steps.sh view-latest

Step 1 must finish (look for "Saved point cloud to ...") before step 2 uses ood_rerun pkl.
If step 1 failed, view-noclip falls back to the last good map (tutorial_nr).

Open3D keys: r=RGB  b=background  c=class  i=instance  f=text query (CLIP only)
EOF
    ;;
esac
