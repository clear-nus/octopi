#!/bin/bash
# Compare the two decoupled scaled-CBS endpoint configs and ablate the four
# settings that differ between them on one fixed k-fold split.
#
# Endpoints:
#   base_paperish: 30 epochs, 5 frames, no ranking, no weight decay
#   target_newer : 15 epochs, 8 frames, ranking=0.5, weight_decay=0.05
#
# One-factor toggles from base_paperish:
#   plus_frames8, plus_epochs15, plus_rank05, plus_wd005
# Extra paper-closeness check:
#   base5_nosmooth: base_paperish with label_smoothing=0.0
#   base5_nosmooth_no_cbs: base5_nosmooth with standard CE
#
# Use VARIANTS="base_paperish plus_frames8" to run a subset.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

K="${K:-5}"
FOLD_SEED="${FOLD_SEED:-0}"
VARIANTS="${VARIANTS:-base_paperish plus_frames8 plus_epochs15 plus_rank05 plus_wd005 target_newer}"

CONFIG="configs/train_clip_config.yaml"
CONSTANTS="src/utils/constants.py"
CONFIG_BAK="${CONFIG}.endpoint_ablation_bak"
CONSTANTS_BAK="${CONSTANTS}.endpoint_ablation_bak"
KFOLD_DIR="scripts/kfold"
KFOLD_BAK="scripts/kfold.endpoint_ablation_bak"
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

restore_originals() {
  echo "[queue] restoring original config/constants/kfold files"
  cp "$CONFIG_BAK" "$CONFIG"
  cp "$CONSTANTS_BAK" "$CONSTANTS"
  rm -f "$CONFIG_BAK" "$CONSTANTS_BAK"
  rm -rf "$KFOLD_DIR"
  cp -a "$KFOLD_BAK" "$KFOLD_DIR"
  rm -rf "$KFOLD_BAK"
}

configure_common_base() {
  set_cfg --key data_dir --value data \
          --key class_balanced_loss --value true \
          --key class_balance_mode --value scaled \
          --key class_balance_strength --value 1.0 \
          --key class_balance_properties --value "[hardness,roughness,texture]" \
          --key decoupled_heads --value true \
          --key decoupled_head_dim --value 128 \
          --key swa --value false \
          --key swa_start_epoch --value 10 \
          --key ema_decay --value 0.0 \
          --key flip_p --value 0.5 \
          --key lr --value 0.0003 \
          --key classifier_lr --value 0.0003 \
          --key label_smoothing --value 0.1 \
          --key ranking_margin --value 0.3 \
          --key batch_size --value 8 \
          --key gradient_accumulation_steps --value 4
}

configure_variant() {
  local tag="$1"
  configure_common_base
  case "$tag" in
    base_paperish)
      set_cfg --key num_epochs --value 30 \
              --key max_frames --value 5 \
              --key ranking_loss_weight --value 0.0 \
              --key weight_decay --value 0.0
      ;;
    base5_nosmooth)
      set_cfg --key num_epochs --value 30 \
              --key max_frames --value 5 \
              --key ranking_loss_weight --value 0.0 \
              --key weight_decay --value 0.0 \
              --key label_smoothing --value 0.0
      ;;
    base5_nosmooth_no_cbs)
      set_cfg --key num_epochs --value 30 \
              --key max_frames --value 5 \
              --key ranking_loss_weight --value 0.0 \
              --key weight_decay --value 0.0 \
              --key label_smoothing --value 0.0 \
              --key class_balanced_loss --value false
      ;;
    plus_frames8)
      set_cfg --key num_epochs --value 30 \
              --key max_frames --value 8 \
              --key ranking_loss_weight --value 0.0 \
              --key weight_decay --value 0.0
      ;;
    plus_epochs15)
      set_cfg --key num_epochs --value 15 \
              --key max_frames --value 5 \
              --key ranking_loss_weight --value 0.0 \
              --key weight_decay --value 0.0
      ;;
    plus_rank05)
      set_cfg --key num_epochs --value 30 \
              --key max_frames --value 5 \
              --key ranking_loss_weight --value 0.5 \
              --key weight_decay --value 0.0
      ;;
    plus_wd005)
      set_cfg --key num_epochs --value 30 \
              --key max_frames --value 5 \
              --key ranking_loss_weight --value 0.0 \
              --key weight_decay --value 0.05
      ;;
    target_newer)
      set_cfg --key num_epochs --value 15 \
              --key max_frames --value 8 \
              --key ranking_loss_weight --value 0.5 \
              --key weight_decay --value 0.05
      ;;
    *)
      echo "unknown variant: $tag" >&2
      exit 1
      ;;
  esac
}

aggregate_summary() {
  local csv_path="$1"
  python - "$csv_path" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for key in ["val_mean", "test_mean", "test_combined"]:
    vals = [float(r[key]) for r in rows]
    print(f"  {key:13s} = {statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}  (n={len(vals)})")
PY
}

run_variant() {
  local tag="$1"
  local summary="clip_kfold_summary_${tag}.txt"
  local csv="${summary}.csv"

  echo "================= ${tag} ================="
  configure_variant "$tag"
  : > "$summary"
  echo "fold,val_mean,test_mean,test_combined" > "$csv"

  for ((i=0; i<K; i++)); do
    echo "================================================================"
    echo "[queue] ${tag} fold ${i} / $((K-1)) :: $(date)"
    echo "================================================================"

    cp "$CONSTANTS_BAK" "$CONSTANTS"
    python scripts/_patch_constants.py "${KFOLD_DIR}/fold_${i}.json"

    local data_dir="${DATA_ROOT}/kfold_${FOLD_SEED}${tag}_f${i}"
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
    echo "[queue] fold $i :: val_mean=$v test_mean=$tm test_combined=$tc" | tee -a "$summary"
    echo "${i},${v},${tm},${tc}" >> "$csv"

    exp_dir=$(latest_exp_dir "$EXP_ID")
    if [ -n "$exp_dir" ] && [ -f "$exp_dir/vificlip.pt" ]; then
      rm -f "$exp_dir/vificlip.pt"
      echo "[queue] deleted $exp_dir/vificlip.pt"
    fi
  done

  echo "[queue] ${tag} aggregate:"
  aggregate_summary "$csv"
}

cp "$CONFIG" "$CONFIG_BAK"
cp "$CONSTANTS" "$CONSTANTS_BAK"
rm -rf "$KFOLD_BAK"
cp -a "$KFOLD_DIR" "$KFOLD_BAK"
trap restore_originals EXIT

python scripts/_build_kfold.py --k "$K" --seed "$FOLD_SEED" --out_dir "$KFOLD_DIR"

for tag in $VARIANTS; do
  run_variant "$tag"
done

echo "[queue] done at $(date)"
