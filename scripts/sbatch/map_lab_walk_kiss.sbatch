#!/bin/bash
#SBATCH --job-name=cg_map_kiss
#SBATCH --partition=b40x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/nvwulf/home/admanoharan/cg_jobs/map_lab_walk_kiss_%j.out
#SBATCH --error=/lustre/nvwulf/home/admanoharan/cg_jobs/map_lab_walk_kiss_%j.err
#
# ConceptGraphs mapping on lab_walk_kiss (975 frames, KISS-ICP poses, stride 10).
#
set -euo pipefail

module load miniconda/3
module load cuda12.8/toolkit/12.8.1 2>/dev/null || true
module load gcc/13.4.0 2>/dev/null || true
source /lustre/nvwulf/software/miniconda3/etc/profile.d/conda.sh

CG_ENV=/lustre/nvwulf/home/admanoharan/.conda/envs/conceptgraph
conda activate "$CG_ENV"
export PATH="$CG_ENV/bin:$PATH"
hash -r
if [[ "$(command -v python)" != "$CG_ENV/bin/python" ]]; then
  echo "ERROR: expected python at $CG_ENV/bin/python, got $(command -v python)" >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda12.8/toolkit/12.8.1}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export HF_HOME=/lustre/nvwulf/home/admanoharan/cg_data/.cache/huggingface
export TORCH_HOME=/lustre/nvwulf/home/admanoharan/cg_data/.cache/torch
export YOLO_CONFIG_DIR=/lustre/nvwulf/home/admanoharan/cg_data/.cache/ultralytics
mkdir -p "$HF_HOME" "$TORCH_HOME" "$YOLO_CONFIG_DIR"

pip install -q 'numpy<2' >/dev/null

SCENE=/lustre/nvwulf/home/admanoharan/cg_data/lab_walk/lab_walk_kiss
test -f "$SCENE/traj.txt"
N=$(ls "$SCENE/results"/frame*.jpg | wc -l)
echo "scene frames: $N"
test "$N" -eq "$(wc -l < "$SCENE/traj.txt")"

cd /lustre/nvwulf/home/admanoharan/concept-graphs/conceptgraph

echo "=== lab_walk_kiss mapping start ==="
date
hostname
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python slam/rerun_realtime_mapping.py --config-name=lab_walk_mapping_kiss

echo "=== lab_walk_kiss mapping done ==="
date
OUT_DIR="$SCENE/exps"
ls -lah "$OUT_DIR" 2>/dev/null || true
find "$OUT_DIR" -name 'pcd_*.pkl.gz' 2>/dev/null | head -20 || true
find "$OUT_DIR" -name 'obj_json_*.json' 2>/dev/null | head -20 || true
echo "MAP_KISS_OK"
