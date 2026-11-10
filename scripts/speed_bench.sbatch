#!/usr/bin/env bash
#   sbatch speed_bench.sbatch VARIANTS_FILE   (lines: name|ENV=v ENV2=v|override override ...)
#SBATCH --job-name=nb-ms-speed-bench
#SBATCH --gpus=1
#SBATCH --time=1:30:00
#SBATCH --open-mode=append
#SBATCH --output=/admin/home/ryan.kim/nanobrain/worktrees/ms-speed/slurm/%j.out
set -uo pipefail
WT=/admin/home/ryan.kim/nanobrain/worktrees/ms-speed
cd "$WT"
set -a; source /admin/home/ryan.kim/nanobrain/base/.env; set +a
export OMP_NUM_THREADS=2
export PYTHONPATH="$WT/src${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
PY=${NB_PYTHON:-/admin/home/ryan.kim/nanobrain/worktrees/wave1-base/.venv/bin/python}
JOBTMP=$(mktemp -d "${TMPDIR:-/tmp}/nb-ms-speed-${SLURM_JOB_ID:-$$}.XXXXXX")
export TORCHINDUCTOR_CACHE_DIR="$JOBTMP/inductor" TRITON_CACHE_DIR="$JOBTMP/triton"
echo "host $(hostname) job ${SLURM_JOB_ID:-} sha $(git rev-parse --short HEAD) cpus $(nproc)"
nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv
BASE=/data/ryan.kim/nanobrain/outputs/ms-speed/bench/${SLURM_JOB_ID:-x}
while IFS='|' read -r NAME ENVS OVR; do
    [[ -z "$NAME" || "$NAME" == \#* ]] && continue
    echo "=== variant $NAME env [$ENVS] overrides [$OVR] $(date -u +%T)"
    mkdir -p "$BASE/$NAME"
    nvidia-smi dmon -s u -d 1 > "$BASE/$NAME/dmon.txt" 2>&1 &
    DM=$!
    # fresh compile caches per variant (startup is measured cold, like a loop job)
    rm -rf "$JOBTMP/inductor" "$JOBTMP/triton"
    env $ENVS "$PY" scripts/speed_bench.py --config src/nanobrain/config/r12_nat_lvl_mf.yaml \
        --overrides output_dir="$BASE" name="$NAME" wandb=false seed=${BENCH_SEED:-2} $OVR 2>&1 \
        | grep -v '^\s*$' | tail -n 40
    kill $DM 2>/dev/null
    "$PY" - "$BASE/$NAME/dmon.txt" <<'PY'
import sys
v=[int(l.split()[1]) for l in open(sys.argv[1]) if l.strip() and l.split()[0].isdigit() and l.split()[1].isdigit()]
v=v[len(v)//3:]
print("gpu util (sm%%) mean over last 2/3 of variant: %.1f (n=%d)" % (sum(v)/max(len(v),1), len(v)))
PY
done < "$1"
rm -rf "$JOBTMP"
