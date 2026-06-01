#!/bin/bash
# LR probe for the paper-close CLIP setup with EMA and no label smoothing.
# Runs 3 canonical seeds for lr=5e-4 and lr=1e-4 on cuda:7.

set -euo pipefail

source /data/samson/miniconda3/etc/profile.d/conda.sh
conda activate octopi

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/train_clip_config.yaml"
CONFIG_BAK="${CONFIG}.lr_probe_ema_seed_reps_bak"
DATASET_PATH="dataset"
DATA_DIR="data"
SUMMARY="clip_lr_probe_ema_seed_reps.txt"
CSV="${SUMMARY}.csv"

cp "$CONFIG" "$CONFIG_BAK"
trap 'echo "[lr-probe] restoring config"; cp "$CONFIG_BAK" "$CONFIG"; rm -f "$CONFIG_BAK"' EXIT

set_cfg() {
  python src/utils/update_config.py --config_path "$CONFIG" "$@"
}

latest_exp_dir() {
  local exp_id="$1"
  find exps -maxdepth 1 -type d -name "*_${exp_id}" | sort | tail -1
}

parse_val_mean_field() {
  local exp_id="$1" field="$2" dir
  dir=$(latest_exp_dir "$exp_id")
  if [ -z "$dir" ] || [ ! -f "$dir/log.txt" ]; then
    echo "ERROR: no log for ${exp_id}" >&2
    return 1
  fi
  python src/utils/parse_clip_log.py "$dir/log.txt" --field "$field"
}

parse_val_combined_json() {
  local exp_id="$1" dir
  dir=$(latest_exp_dir "$exp_id")
  if [ -z "$dir" ] || [ ! -f "$dir/log.txt" ]; then
    echo "ERROR: no log for ${exp_id}" >&2
    return 1
  fi
  python - "$dir/log.txt" <<'PY'
import json, re, sys
from pathlib import Path

pat_epoch = re.compile(r"TRAIN epoch: (\d+) /")
pat_val = re.compile(r"VAL accuracies \[hardness, roughness, texture, combined\]: (.*)")
pat_test = re.compile(r"TEST accuracies \[hardness, roughness, texture, combined\]: (.*)")

entries = []
epoch = None
val = None
for line in Path(sys.argv[1]).read_text().splitlines():
    m = pat_epoch.search(line)
    if m:
        epoch = int(m.group(1))
        val = None
    m = pat_val.search(line)
    if m:
        val = [float(x.strip()) for x in m.group(1).split(",")]
    m = pat_test.search(line)
    if m and val is not None:
        test = [float(x.strip()) for x in m.group(1).split(",")]
        entries.append({
            "epoch": epoch,
            "val_mean_combsel": sum(val[:3]) / 3,
            "val_combined_combsel": val[3],
            "test_mean_combsel": sum(test[:3]) / 3,
            "test_combined_combsel": test[3],
        })

if not entries:
    raise SystemExit("no complete epoch entries")

best = max(e["val_combined_combsel"] for e in entries)
selected = next(e for e in entries if e["val_combined_combsel"] == best)
print(json.dumps(selected))
PY
}

configure_variant() {
  local lr="$1"
  set_cfg --key cuda --value 7 \
          --key data_dir --value "$DATA_DIR" \
          --key num_epochs --value 30 \
          --key max_frames --value 5 \
          --key ranking_loss_weight --value 0.0 \
          --key weight_decay --value 0.0 \
          --key label_smoothing --value 0.0 \
          --key lr --value "$lr" \
          --key classifier_lr --value "$lr" \
          --key class_balanced_loss --value true \
          --key class_balance_mode --value scaled \
          --key class_balance_strength --value 1.0 \
          --key class_balance_properties --value "[hardness,roughness,texture]" \
          --key decoupled_heads --value true \
          --key decoupled_head_dim --value 128 \
          --key ema_decay --value 0.98 \
          --key swa --value false \
          --key flip_p --value 0.5 \
          --key rotation_degrees --value 0 \
          --key color_jitter --value 0.0 \
          --key gaussian_blur --value false
}

record_result() {
  local tag="$1" lr="$2" seed="$3" exp_id="$4"
  local vm tm tc comb_json
  vm=$(parse_val_mean_field "$exp_id" val_mean)
  tm=$(parse_val_mean_field "$exp_id" test_mean)
  tc=$(parse_val_mean_field "$exp_id" test_combined)
  comb_json=$(parse_val_combined_json "$exp_id")
  python - "$tag" "$lr" "$seed" "$exp_id" "$vm" "$tm" "$tc" "$comb_json" "$CSV" <<'PY'
import csv, json, sys
tag, lr, seed, exp_id, vm, tm, tc, comb_json, csv_path = sys.argv[1:]
comb = json.loads(comb_json)
row = {
    "tag": tag,
    "lr": lr,
    "seed": seed,
    "exp_id": exp_id,
    "val_mean_select_val_mean": vm,
    "val_mean_select_test_mean": tm,
    "val_mean_select_test_combined": tc,
    "val_combined_select_epoch": comb["epoch"],
    "val_combined_select_val_mean": comb["val_mean_combsel"],
    "val_combined_select_val_combined": comb["val_combined_combsel"],
    "val_combined_select_test_mean": comb["test_mean_combsel"],
    "val_combined_select_test_combined": comb["test_combined_combsel"],
}
with open(csv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writerow(row)
print(
    f"[lr-probe] {tag} seed={seed} :: "
    f"valmean_sel test_mean={float(tm):.3f} test_combined={float(tc):.3f}; "
    f"valcombined_sel epoch={comb['epoch']} test_mean={comb['test_mean_combsel']:.3f} "
    f"test_combined={comb['test_combined_combsel']:.3f}"
)
PY
}

run_one() {
  local tag="$1" lr="$2" seed="$3"
  local exp_id="clip_${tag}_seed_${seed}"

  echo "================================================================"
  echo "[lr-probe] ${tag} lr=${lr} seed=${seed} :: $(date)"
  echo "================================================================"

  rm -rf "$DATA_DIR"
  python src/utils/process_dataset.py --dataset_path "$DATASET_PATH" --output_path "$DATA_DIR" --seed "$seed"
  python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed "$seed"

  configure_variant "$lr"
  set_cfg --key seed --value "$seed" --key data_dir --value "$DATA_DIR"

  export EXP_ID="$exp_id"
  python src/train_clip.py
  record_result "$tag" "$lr" "$seed" "$exp_id"
}

aggregate_summary() {
  python - "$CSV" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for tag in sorted({r["tag"] for r in rows}):
    rs = [r for r in rows if r["tag"] == tag]
    print(f"[lr-probe] aggregate {tag} n={len(rs)}")
    for key in [
        "val_mean_select_test_mean",
        "val_mean_select_test_combined",
        "val_combined_select_test_mean",
        "val_combined_select_test_combined",
    ]:
        vals = [float(r[key]) for r in rs]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {key:35s} = {statistics.mean(vals):.3f} +/- {sd:.3f}")
PY
}

: > "$SUMMARY"
python - "$CSV" <<'PY'
import csv, sys
fields = [
    "tag", "lr", "seed", "exp_id",
    "val_mean_select_val_mean",
    "val_mean_select_test_mean",
    "val_mean_select_test_combined",
    "val_combined_select_epoch",
    "val_combined_select_val_mean",
    "val_combined_select_val_combined",
    "val_combined_select_test_mean",
    "val_combined_select_test_combined",
]
with open(sys.argv[1], "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fields).writeheader()
PY

for seed in 0 1 2; do
  run_one "lr5e-4_ema098_nosmooth" "0.0005" "$seed"
done

aggregate_summary | tee -a "$SUMMARY"

for seed in 0 1 2; do
  run_one "lr1e-4_ema098_nosmooth" "0.0001" "$seed"
done

aggregate_summary | tee -a "$SUMMARY"
echo "[lr-probe] done at $(date)"
