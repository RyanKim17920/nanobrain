#!/usr/bin/env bash
# Full collprobe suite: embeddings (GPU) + input-stats baseline + curves, for N checkpoints on one selection.
#   sbatch --account=idle --qos=low -J nb-collprobe scripts/collprobe_suite.sbatch DIAG_WORKTREE TAG K PER_COLL MIN_COUNT NAME=RUN_DIR=CODE_ROOT [...]
# e.g. K=100 x 40:  ... $PWD k100 100 40 40 fb-s7338=/path/fino3d/output/fb-s7338=/path/fino3d r12t-s7338=...
# Outputs in DIAG_WORKTREE/output/collprobe: <NAME>-<TAG>{,_emb,_curves}.json/npz, <TAG>_inputstats.json, inputstats_<TAG>.npz
#SBATCH --partition=n
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --open-mode=append
#SBATCH --output=/admin/home/ryan.kim/nanobrain/slurms/%j.out
set -euo pipefail
W=$1; TAG=$2; K=$3; PER=$4; MINC=$5; shift 5
PY=/admin/home/ryan.kim/nanobrain/worktrees/wave1-base/.venv/bin/python
cd "$W"
export OMP_NUM_THREADS=4
SEL="--n-collections $K --per-collection $PER --min-count $MINC"
first=1
for spec in "$@"; do
  IFS== read -r NAME RUN CODE <<<"$spec"
  PYTHONPATH= $PY scripts/collprobe.py --ckpt "$RUN/checkpoint-last.pth" --config "$RUN/config.yaml" \
      --code-root "$CODE" --name "$NAME-$TAG" $SEL --workers 12
  SIG=$($PY -c "import json;print(json.load(open('output/collprobe/$NAME-$TAG.json'))['volume_sig'])")
  PYTHONPATH="$CODE/src" $PY scripts/collprobe_curves.py --feat "output/collprobe/$NAME-$TAG""_emb.npz" \
      --volumes "output/collprobe/volumes_$SIG.tsv" --out "output/collprobe/$NAME-$TAG""_curves.json"
  if [[ $first == 1 ]]; then
    PYTHONPATH="$CODE/src" $PY scripts/collprobe_input.py --config "$RUN/config.yaml" --name "$TAG" $SEL --workers 14
    first=0
  fi
done
