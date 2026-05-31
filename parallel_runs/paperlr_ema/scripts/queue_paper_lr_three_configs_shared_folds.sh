#!/bin/bash
# Run the paper-close CLIP ablation sequence on one shared k-fold split:
#   1) paper_lr_pure_ce      : CE only, label_smoothing=0.0
#   2) paper_lr_smooth01     : CE only, label_smoothing=0.1
#   3) paper_lr_cbs_shared   : scaled class-balanced CE, shared heads
#
# This intentionally builds scripts/kfold/fold_*.json once and reuses those
# folds for all three configs, so the selector comparison is not confounded by
# regenerated fold assignments. Each fold's vificlip.pt is deleted after metrics
# are parsed.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

K="${K:-5}"
FOLD_SEED="${FOLD_SEED:-0}"
CONFIG="configs/train_clip_config.yaml"
CONSTANTS="src/utils/constants.py"
CONFIG_BAK="${CONFIG}.paper_lr_three_bak"
CONSTANTS_BAK="${CONSTANTS}.paper_lr_three_bak"
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

restore_defaults() {
  echo "[queue] restoring baseline config/constants"
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

configure_common_paper_lr() {
  set_cfg --key data_dir --value data \
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
          --key weight_decay --value 0.0 \
          --key ranking_loss_weight --value 0.0 \
          --key ranking_margin --value 0.3 \
          --key batch_size --value 8 \
          --key gradient_accumulation_steps --value 4
}

configure_variant() {
  local tag="$1"
  configure_common_paper_lr
  case "$tag" in
    paper_lr_pure_ce)
      set_cfg --key class_balanced_loss --value false \
              --key label_smoothing --value 0.0
      ;;
    paper_lr_smooth01)
      set_cfg --key class_balanced_loss --value false \
              --key label_smoothing --value 0.1
      ;;
    paper_lr_cbs_shared)
      set_cfg --key class_balanced_loss --value true \
              --key class_balance_mode --value scaled \
              --key label_smoothing --value 0.0
      ;;
    *)
      echo "unknown variant: $tag" >&2
      exit 1
      ;;
  esac
}

run_variant() {
  local tag="$1"
  local summary="clip_kfold_summary_${tag}.txt"
  local data_dir="${DATA_ROOT}/kfold_${FOLD_SEED}${tag}"

  echo "================= ${tag} ================="
  configure_variant "$tag"
  : > clip_kfold_summary.txt
  echo "fold,val_mean,test_mean,test_combined" > clip_kfold_summary.txt.csv
  : > "$summary"
  echo "fold,val_mean,test_mean,test_combined" > "${summary}.csv"

  for ((i=0; i<K; i++)); do
    echo "================================================================"
    echo "[queue] ${tag} fold ${i} / $((K-1)) :: $(date)"
    echo "================================================================"

    cp "$CONSTANTS_BAK" "$CONSTANTS"
    python scripts/_patch_constants.py "scripts/kfold/fold_${i}.json"

    rm -rf "$data_dir"
    mkdir -p "$DATA_ROOT"
    python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" \
        --output_path "$data_dir" --seed 0
    python src/utils/generate_qa.py --data_path "$data_dir" --seed 0
    set_cfg --key data_dir --value "$data_dir" --key seed --value 0

    export EXP_ID="clip_kfold_${FOLD_SEED}${tag}_f${i}"
    python src/train_clip.py

    local v tm tc exp_dir
    v=$(latest_log_field "$EXP_ID" val_mean)
    tm=$(latest_log_field "$EXP_ID" test_mean)
    tc=$(latest_log_field "$EXP_ID" test_combined)
    echo "[queue] fold $i :: val_mean=$v test_mean=$tm test_combined=$tc" | tee -a clip_kfold_summary.txt "$summary"
    echo "${i},${v},${tm},${tc}" >> clip_kfold_summary.txt.csv
    echo "${i},${v},${tm},${tc}" >> "${summary}.csv"

    exp_dir=$(latest_exp_dir "$EXP_ID")
    if [ -n "$exp_dir" ] && [ -f "$exp_dir/vificlip.pt" ]; then
      rm -f "$exp_dir/vificlip.pt"
      echo "[queue] deleted $exp_dir/vificlip.pt"
    fi
  done

  echo "[queue] ${tag} aggregate:"
  python - "${summary}.csv" <<'PY'
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

run_variant paper_lr_pure_ce
run_variant paper_lr_smooth01
run_variant paper_lr_cbs_shared

echo "[queue] done at $(date)"
