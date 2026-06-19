# New-box setup: data processing + encoder training (for the Claude agent)

Operational companion to `HANDOFF.md` (which has the science/findings). This file is just "how to stand the project up and reproduce the data + encoder."

## 0. What must be transferred
| item | size | how |
|---|---|---|
| repo code | ~35 MB | `git clone` the `fix` branch (fast via GitHub; use a FRESH token, the old one was exposed) |
| **`dataset/`** (raw GelSight `.mov`) | **29 MB** | **must copy — NOT regenerable.** 476 `.mov` files, the only true source data. THIS IS THE ONLY REQUIRED LARGE COPY. |
| chosen encoder `encoder.pt` | 1.2 GB | **do NOT copy if upload is slow — retrain it locally in §3** (full recipe, ~equivalent quality). Copy only if you need this exact encoder byte-for-byte. |

> You do not need to transfer `encoder.pt`. §3 retrains it from `dataset/`. A retrained encoder is equivalent quality (`test_mean ~0.72`) but not byte-identical (cross-machine nondeterminism) — irrelevant for the gold-weights eval (it uses the gold model's own encoder), minor variance only if continuing our LLM runs.
| gold weights (original Octopi) | — | upload separately; this is the reason for the move |

The 67 GB `exps/` does NOT need to move. `data/` and the `/tmp/*` processed dirs are regenerable (§2).

## 1. Environment (CRITICAL gotcha)
CLIP/LLM training MUST run in the `octopi` conda env. The base env's torch is built against a CUDA newer than the driver and dies with "NVIDIA driver too old".
```bash
conda create -n octopi python=3.10 -y && conda activate octopi   # or recreate from the old box's env export
pip install -r requirements.txt
# also need (not all pinned in requirements): torch (CUDA build matching the driver), transformers, peft==0.2.0, accelerate, natsort, safetensors, opencv-python
```
Always run from the repo root (scripts import `utils.*` directly). Set `EXP_ID=<name>` to skip the interactive experiment-name prompt.

**CPU oversubscription:** torch defaults OMP threads to all cores; with several runs this oversubscribes and slows everything ~3x. Export per run:
```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
```
(The queue scripts already set these.)

## 2. Data processing (raw .mov -> frames -> samples -> QA)
Deterministic; seed 0. Pick a DATA_DIR (the scripts default to `/tmp/...`, but anything works):
```bash
DATA_DIR=/tmp/octopi_data_seed0          # or ./data
conda activate octopi && cd <repo>

# 2a. extract salient frames + build train/val/test sample splits (writes frame folders + *_samples.json)
python src/utils/process_dataset.py --dataset_path dataset --output_path "$DATA_DIR" --seed 0

# 2b. generate the QA (writes train_qa.json / val_qa.json / test_qa.json / val_opd_qa.json / test_opd_qa.json)
python src/utils/generate_qa.py --data_path "$DATA_DIR" --seed 0
```
After this, `$DATA_DIR` holds: `physiclear_<obj>_<n>/` frame folders, `{train,val,test}_samples.json` (for CLIP), and `{train,val,test}_qa.json` + `{val,test}_opd_qa.json` (for the LLM).
- Splits are fixed in `src/utils/constants.py` (canonical 60 train / 7 val / 7 test — verify `len(TRAIN/VAL/TEST_OBJECTS)==60/7/7`; do NOT use a k-fold-patched constants.py).
- Reproducibility: `process_dataset.py` sorts raw listings; frame filenames are zero-padded original-sequence indices (the per-frame sinusoidal PE keys off these raw indices — keep them).

## 3. CLIP encoder (the chosen f5-sinusoid encoder)
The locked config (see `AGENTS.md` "Locked Final Paper-Close Config" + the sinusoid note): 5 frames, 15 epochs, lr/classifier_lr 3e-4, EMA 0.98, scaled class-balanced CE, decoupled heads, `prompt_depth_vision=24`, flip 0.5, no ranking/wd/smoothing, **per-frame sinusoidal PE (live in `src/utils/model.py`)**, checkpoint selection by `val_combined`.

Easiest — use the sweep script (handles config + selection + encoder save):
```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
FRAMES=5 CUDA=0 SEEDS="0" NUM_EPOCHS=15 CLIP_DATA="$DATA_DIR" \
  bash scripts/queue_clip_sinusoid_sweep.sh
```
- Writes the run to `exps/<ts>_train_clip_clip_sinusoid_f5_s0_e15_vc/`; the encoder is `encoder.pt` (an averaged/EMA state dict saved via `average_state_dict_files`).
- Pick the seed with the best `val_combined` if you sweep several. The reference encoder this session was seed 0, `val_combined=0.417`.
- Verify: `python src/utils/parse_clip_log.py exps/<run>/log.txt --selector val_combined --field test_mean` (expect ~0.72; `test_combined` ~0.45). Reference 5-seed: `test_mean=0.723`, `test_combined=0.453`.
- The classifier head is discarded downstream; only `encoder.pt` feeds the LLM.

## 4. LLM Stage 1 -> Stage 2 (sanity that the pipeline runs)
Point at the encoder + the same DATA_DIR (LLM_DATA needs the `*_qa.json`). Faithful-schedule run (recommended — includes the T_max fix + Stage-1-constant):
```bash
FRAMES=5 CUDA=0 \
  ENCODER_PATH=exps/<clip_run>/encoder.pt \
  LLM_DATA="$DATA_DIR" \
  bash scripts/queue_faithsched_stage12.sh
```
Other entry points: `queue_sinusoid_stage12.sh` (T_max fix only), `eval_stage1_binding.sh` (test-only eval of a Stage-1 checkpoint, single pass). LLM runs are bf16 single-GPU (PEFT 0.2.0 + multi-GPU device_map is incompatible). `gpu_config_7b_gpu<N>.json` sets the per-GPU memory cap.

## 5. Gold-weights eval (the point of the move)
Load the original released weights and run our eval to confirm ~0.59 POM under our harness (establishes the target). The original output file is checked in at `src/old/opd_pc_pss_pom_7b.json` for reference. Use the test-only pattern from `scripts/eval_stage1_binding.sh` (`train=False, test=True`), pointing `llm_path`/`tokenizer_path`/`projection_path`/`encoder_path` at the gold artifacts. Watch the tokenizer vocab size matches the gold embeddings before loading.

## 6. Reading results (don't get fooled)
- POM `*_partial_results.txt` is written DURING eval and swings wildly under ~100 items — only trust `canonical_slot` at n≳120 / on the final `*_preds.json`.
- POM `canonical_slot` chance = 0.333; "binding" = clearly above (original ~0.59). ~0.33-0.39 = "emits valid candidates, guesses" = NOT binding.
- To check whether a run actually has the T_max schedule fix: compute `‖ΔW‖/‖W‖` on q/k from the saved adapter — ~1.5% = no fix, ~2.8% = fix active.
- Score any preds file by replaying `LLMEvaluator.evaluate(...)` from `src/evaluate_llm.py` over the rows.
