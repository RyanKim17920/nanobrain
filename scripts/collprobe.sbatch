#!/usr/bin/env bash
# sbatch --account=idle --qos=low scripts/collprobe.sbatch CKPT CONFIG CODE_ROOT [CKPT CONFIG CODE_ROOT ...]
#   collection-identity linear probe (scripts/collprobe.py) for each (ckpt, config, code root) triple
#SBATCH --job-name=nb-collprobe
#SBATCH --partition=n
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=/admin/home/ryan.kim/nanobrain/slurms/%j.out
set -euo pipefail
HERE=/admin/home/ryan.kim/nanobrain/worktrees/diag-collprobe
cd "$HERE"
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri OMP_NUM_THREADS=8
PY=/admin/home/ryan.kim/nanobrain/worktrees/wave1-base/.venv/bin/python
echo "host $(hostname) job ${SLURM_JOB_ID:-} sha $(git rev-parse --short HEAD)"
while [[ $# -ge 3 ]]; do
    # PYTHONPATH empty: collprobe.py puts CODE_ROOT/src first, so each run loads its own nanobrain
    PYTHONPATH= $PY scripts/collprobe.py --ckpt "$1" --config "$2" --code-root "$3"
    shift 3
done
