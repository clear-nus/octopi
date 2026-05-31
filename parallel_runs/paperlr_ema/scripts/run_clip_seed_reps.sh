#!/bin/bash
# 3-seed reproducibility check on two CLIP configs that diverged in the rotation/ranking sweep:
#   A: rot=0,  w=0.5, m=0.3  (best by val_mean)
#   B: rot=20, w=0.5, m=1.0  (best by test_combined, val/test divergence flagged)
#
# For each seed in {0, 1, 2}:
#   - regenerate data/ via process_dataset.py + generate_qa.py with that seed
#   - run config A
#   - run config B
#
# 6 training runs total. Object splits are fixed in constants.py — seed only varies
# stochastic ordering/training. This tests whether the sweep's val/test divergence
# survives across training seeds, not whether a different val split would change ranking.
#
# Restores configs/train_clip_config.yaml and the data dir's prior state on exit.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

CONFIG="configs/train_clip_config.yaml"
BACKUP="${CONFIG}.reps_bak"
DATASET_PATH="dataset"
DATA_DIR="data"
SUMMARY="clip_seed_reps_summary.txt"

cp "$CONFIG" "$BACKUP"
trap 'echo "[reps] restoring $CONFIG from $BACKUP"; cp "$BACKUP" "$CONFIG"; rm -f "$BACKUP"' EXIT

: > "$SUMMARY"
echo "config,seed,val_mean,test_combined" > "${SUMMARY}.csv"

update() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

latest_log_field() {
  local exp_id="$1" field="$2" dir
  dir=$(find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1)
  if [ -z "$dir" ] || [ ! -f "$dir/log.txt" ]; then
    echo "ERROR: no log for ${exp_id}" >&2
    return 1
  fi
  python src/utils/parse_clip_log.py "$dir/log.txt" --field "$field"
}

run_one() {
  local cfg_tag="$1" seed="$2"
  export EXP_ID="clip_reps_${cfg_tag}_s${seed}"
  echo "================================================================"
  echo "[reps] $cfg_tag seed=$seed :: $(date)"
  grep -E '^rotation_degrees|^ranking_loss_weight|^ranking_margin|^seed' "$CONFIG"
  echo "================================================================"
  python src/train_clip.py
  local v t
  v=$(latest_log_field "$EXP_ID" val_mean)
  t=$(latest_log_field "$EXP_ID" test_combined)
  echo "[reps] $cfg_tag seed=$seed :: val_mean=$v test_combined=$t" | tee -a "$SUMMARY"
  echo "${cfg_tag},${seed},${v},${t}" >> "${SUMMARY}.csv"
}

for SEED in 0 1 2; do
  echo "[reps] -------- seed=$SEED data regen --------"
  rm -rf "$DATA_DIR"
  python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$SEED"
  python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed "$SEED"
  update --key data_dir --value "$DATA_DIR" --key seed --value "$SEED"

  # Config A: rot=0, w=0.5, m=0.3
  update --key rotation_degrees --value 0 \
         --key ranking_loss_weight --value 0.5 \
         --key ranking_margin --value 0.3
  run_one "A_rot0_w0.5_m0.3" "$SEED"

  # Config B: rot=20, w=0.5, m=1.0
  update --key rotation_degrees --value 20 \
         --key ranking_loss_weight --value 0.5 \
         --key ranking_margin --value 1.0
  run_one "B_rot20_w0.5_m1.0" "$SEED"
done

echo "================================================================"
echo "[reps] done. raw summary:"
cat "$SUMMARY"
echo "----------------------------------------------------------------"
echo "[reps] aggregate (mean ± std):"
python - <<'PY'
import csv, statistics
rows = list(csv.DictReader(open("clip_seed_reps_summary.txt.csv")))
for cfg in sorted({r["config"] for r in rows}):
    vs = [float(r["val_mean"]) for r in rows if r["config"] == cfg]
    ts = [float(r["test_combined"]) for r in rows if r["config"] == cfg]
    vm, vs_ = statistics.mean(vs), statistics.stdev(vs) if len(vs) > 1 else 0.0
    tm, ts_ = statistics.mean(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0
    print(f"  {cfg}: val_mean={vm:.3f}±{vs_:.3f}  test_combined={tm:.3f}±{ts_:.3f}  (n={len(vs)})")
PY
