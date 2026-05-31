#!/bin/bash
# Resume the interrupted paper-LR CE sequence:
#   1) finish paper_lr_pure_ce from fold 2 using existing fold 0/1 logs
#   2) run paper_lr_smooth01 from fold 0
# The separate queue_after_ce_smoothing_then_paper_cbs_shared.sh waiter will
# start paper_lr_cbs_shared once both copied summaries exist.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

K="${K:-5}"
FOLD_SEED="${FOLD_SEED:-0}"
CONFIG="configs/train_clip_config.yaml"
CONSTANTS="src/utils/constants.py"
CONFIG_BAK="${CONFIG}.resume_paper_lr_bak"
CONSTANTS_BAK="${CONSTANTS}.resume_paper_lr_bak"
DATASET_PATH="dataset"
DATA_ROOT="data"

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

latest_exp_dir() {
  local exp_id="$1"
  find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1
}

latest_log_field() {
  local exp_id="$1" field="$2" dir
  dir=$(latest_exp_dir "$exp_id")
  if [ -z "$dir" ] || [ ! -f "$dir/log.txt" ]; then
    echo "ERROR: no log for ${exp_id}" >&2
    return 1
  fi
  python src/utils/parse_clip_log.py "$dir/log.txt" --field "$field"
}

append_existing_fold() {
  local tag="$1" fold="$2" summary="$3"
  local exp_id="clip_kfold_${FOLD_SEED}${tag}_f${fold}"
  local v tm tc
  v=$(latest_log_field "$exp_id" val_mean)
  tm=$(latest_log_field "$exp_id" test_mean)
  tc=$(latest_log_field "$exp_id" test_combined)
  echo "[resume] existing fold $fold :: val_mean=$v test_mean=$tm test_combined=$tc" | tee -a "$summary"
  echo "${fold},${v},${tm},${tc}" >> "${summary}.csv"
}

configure_paper_lr_ce() {
  local smoothing="$1"
  set_cfg --key data_dir --value data \
          --key class_balanced_loss --value false \
          --key class_balance_mode --value scaled \
          --key decoupled_heads --value false \
          --key decoupled_head_dim --value 128 \
          --key swa --value false \
          --key swa_start_epoch --value 10 \
          --key ema_decay --value 0.0 \
          --key flip_p --value 0.5 \
          --key max_frames --value 5 \
          --key num_epochs --value 30 \
          --key lr --value 0.001 \
          --key classifier_lr --value 0.001 \
          --key label_smoothing --value "$smoothing" \
          --key weight_decay --value 0.0 \
          --key ranking_loss_weight --value 0.0 \
          --key ranking_margin --value 0.3 \
          --key batch_size --value 8 \
          --key gradient_accumulation_steps --value 4
}

restore_defaults() {
  echo "[resume] restoring plain baseline config"
  cp "$CONFIG_BAK" "$CONFIG"
  cp "$CONSTANTS_BAK" "$CONSTANTS"
  rm -f "$CONFIG_BAK" "$CONSTANTS_BAK"
  set_cfg --key data_dir --value data \
          --key class_balanced_loss --value false \
          --key class_balance_mode --value scaled \
          --key decoupled_heads --value false \
          --key decoupled_head_dim --value 128 \
          --key swa --value false \
          --key swa_start_epoch --value 10 \
          --key ema_decay --value 0.0 \
          --key flip_p --value 0.5 \
          --key max_frames --value 8 \
          --key num_epochs --value 15 \
          --key lr --value 0.0003 \
          --key classifier_lr --value 0.0003 \
          --key label_smoothing --value 0.1 \
          --key weight_decay --value 0.05 \
          --key ranking_loss_weight --value 0.5 \
          --key ranking_margin --value 0.3 \
          --key batch_size --value 8 \
          --key gradient_accumulation_steps --value 4
}

run_fold() {
  local tag="$1" fold="$2" summary="$3"
  local data_dir="${DATA_ROOT}/kfold_${FOLD_SEED}${tag}"
  local exp_id="clip_kfold_${FOLD_SEED}${tag}_f${fold}"

  echo "================================================================"
  echo "[resume] ${tag} fold ${fold} / $((K-1)) :: $(date)"
  echo "================================================================"

  cp "$CONSTANTS_BAK" "$CONSTANTS"
  python scripts/_patch_constants.py "scripts/kfold/fold_${fold}.json"

  rm -rf "$data_dir"
  mkdir -p "$DATA_ROOT"
  python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" \
      --output_path "$data_dir" --seed 0
  python src/utils/generate_qa.py --data_path "$data_dir" --seed 0
  set_cfg --key data_dir --value "$data_dir" --key seed --value 0

  export EXP_ID="$exp_id"
  python src/train_clip.py

  local v tm tc exp_dir
  v=$(latest_log_field "$exp_id" val_mean)
  tm=$(latest_log_field "$exp_id" test_mean)
  tc=$(latest_log_field "$exp_id" test_combined)
  echo "[resume] fold $fold :: val_mean=$v test_mean=$tm test_combined=$tc" | tee -a "$summary"
  echo "${fold},${v},${tm},${tc}" >> "${summary}.csv"

  exp_dir=$(latest_exp_dir "$exp_id")
  if [ -n "$exp_dir" ] && [ -f "$exp_dir/vificlip.pt" ]; then
    rm -f "$exp_dir/vificlip.pt"
    echo "[resume] deleted $exp_dir/vificlip.pt"
  fi
}

aggregate_summary() {
  local csv="$1"
  python - "$csv" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
vs = [float(r["val_mean"]) for r in rows]
tms = [float(r["test_mean"]) for r in rows]
ts = [float(r["test_combined"]) for r in rows]
print(f"  val_mean      = {statistics.mean(vs):.3f} ± {statistics.stdev(vs):.3f}  (n={len(vs)})")
print(f"  test_mean     = {statistics.mean(tms):.3f} ± {statistics.stdev(tms):.3f}  (n={len(tms)})")
print(f"  test_combined = {statistics.mean(ts):.3f} ± {statistics.stdev(ts):.3f}  (n={len(ts)})")
PY
}

cp "$CONFIG" "$CONFIG_BAK"
cp "$CONSTANTS" "$CONSTANTS_BAK"
trap restore_defaults EXIT

python scripts/_build_kfold.py --k "$K" --seed "$FOLD_SEED"

echo "================= resume paper_lr_pure_ce ================="
configure_paper_lr_ce 0.0
: > clip_kfold_summary.txt
echo "fold,val_mean,test_mean,test_combined" > clip_kfold_summary.txt.csv
append_existing_fold paper_lr_pure_ce 0 clip_kfold_summary.txt
append_existing_fold paper_lr_pure_ce 1 clip_kfold_summary.txt
for ((i=2; i<K; i++)); do
  run_fold paper_lr_pure_ce "$i" clip_kfold_summary.txt
done
echo "[resume] paper_lr_pure_ce aggregate:"
aggregate_summary clip_kfold_summary.txt.csv
cp clip_kfold_summary.txt clip_kfold_summary_paper_lr_pure_ce.txt
cp clip_kfold_summary.txt.csv clip_kfold_summary_paper_lr_pure_ce.txt.csv

echo "================= run paper_lr_smooth01 ================="
configure_paper_lr_ce 0.1
: > clip_kfold_summary.txt
echo "fold,val_mean,test_mean,test_combined" > clip_kfold_summary.txt.csv
for ((i=0; i<K; i++)); do
  run_fold paper_lr_smooth01 "$i" clip_kfold_summary.txt
done
echo "[resume] paper_lr_smooth01 aggregate:"
aggregate_summary clip_kfold_summary.txt.csv
cp clip_kfold_summary.txt clip_kfold_summary_paper_lr_smooth01.txt
cp clip_kfold_summary.txt.csv clip_kfold_summary_paper_lr_smooth01.txt.csv

echo "[resume] done at $(date)"
