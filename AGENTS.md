# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the training and evaluation entry points: `train_clip.py`, `train_llm.py`, `test_two_step.py`, `evaluate_llm.py`, and `interact.py`. Shared dataset, model, and config helpers live in `src/utils/`. Configs are in `configs/`; reproducibility scripts are in `scripts/`. Raw GelSight `.mov` files live in `dataset/`, processed frame folders and QA JSON files are generated into `data/`, figures stay in `assets/`, and run outputs are written to `exps/<timestamp>_<exp_id>/`.

## Build, Training, and Evaluation Commands
Install dependencies with `pip install -r requirements.txt`.

Prepare data from raw videos:
```bash
python src/utils/process_dataset.py --dataset_path dataset --output_path data --seed 0
python src/utils/generate_qa.py --data_path data --seed 0
```
Run from the repo root; scripts import `utils.*` directly. Set `EXP_ID=my_run` to skip the interactive experiment prompt.

Main training flow:
```bash
python src/train_clip.py
python src/train_llm.py
bash scripts/run_llm_training.sh
```
`run_llm_training.sh` regenerates `data/`, runs Stage 1 tactile-language alignment (`use_lora: False`, currently `max_train_steps: 3200`), then launches the LoRA grid search.

Evaluation commands:
```bash
python src/test_two_step.py
python src/evaluate_llm.py --test_preds_path exps/<exp>/test_best_preds.json
python src/interact.py
```

## Architecture Notes
`src/utils/model.py` holds the core modules: `CLIPTactileEncoder`, `ViFiCLIP`, and `MultimodalLLMForCausalLM`. Tactile frames are encoded by CLIP, projected into Vicuna embedding space, and wrapped with `<tact_start>` / `<tact_end>` tokens. Train/val/test object splits are hardcoded in `src/utils/constants.py`.

## Coding Style & Testing Guidelines
Use 4-space indentation, snake_case names, and keep changes local to `src/utils/` when extending shared logic. No formatter or linter is configured, so match surrounding style. There is no dedicated `tests/` directory; validation is script-driven. When changing training behavior, record the exact command, config file, and resulting metrics. Treat `exps/` artifacts as reproducibility evidence, not source.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects (`Update`, `Add some fixes`). Keep that tone, but be more specific, for example `Tune Stage 1 validation cadence`. PRs should state the goal, changed configs, GPU/runtime impact, and before/after validation or evaluation results.

## Training Setup (bf16 Single-GPU)
All LLM training runs on a single GPU (`cuda:6`, 35 GiB) using `torch_dtype=torch.bfloat16`. This avoids PEFT 0.2.0 + multi-GPU device_map incompatibility. The encoder and projector remain float32; `input_embeds` and `question_embeds` are cast to the LLM dtype before any forward/generate call. LoRA params are cast to bf16 immediately after `get_peft_model` and after any `PeftModel.from_pretrained` reload.

## Changes from Original Octopi Paper

These are deviations from the original paper's methodology. Each is marked with whether it is currently active.

### CLIP Encoder

| Change | Status | Files |
|--------|--------|-------|
| Patch token mean pooling (`hidden_states[-2][:, 1:].mean()`) instead of CLS token | **Reverted** — CLS outperforms patch mean empirically (0.553 vs 0.421 peak test combined) | `src/utils/model.py` |
| Multi-layer feature fusion over layers `[-2, -6, -12]` instead of a single layer | **Reverted to single layer** (`fusion_layers: [-2]`) — multi-layer did not improve over CLS single-layer baseline | `src/utils/model.py`, `src/train_clip.py` |
| Pairwise ranking loss added on top of CE loss (`ranking_loss_weight: 0.5`) | **Active** | `src/train_clip.py` |
| Val save criterion reverted to mean per-property accuracy (combined was too noisy on ~69 val samples) | **Active** | `src/train_clip.py` |
| `num_epochs` reduced to 15 (paper: 30) — consistent overfitting observed after epoch ~13 | **Active** | `configs/train_clip_config.yaml` |
| `max_frames` kept at 8 (paper setting); earlier regression to 5 was unintentional | **Active** | `configs/train_clip_config.yaml` |
| Data augmentation config keys added (rotation, ColorJitter, GaussianBlur) | **Disabled** (all set to 0/false) — hurt GelSight color-encoded force features | `src/utils/dataset.py`, configs |
| `unfreeze_last_n_layers` config key added | **Disabled** (set to 0) | `src/train_clip.py` |

### LLM Training (Stages 1–3)

| Change | Status | Files |
|--------|--------|-------|
| Projector MLP has a `LayerNorm` between the two linear layers | **Active** | `src/utils/model.py` |
| Per-group gradient clipping instead of a single combined `clip_grad_norm_` | **Active** | `src/train_llm.py` |
| Val loader `shuffle=False` for reproducible checkpoint selection | **Active** | `src/train_llm.py` |
| Stage 1 `val_freq` 400 → 200 steps (16 checkpoints over 3200 steps) | **Active** | `scripts/run_llm_training.sh` |
| Stage 2 LoRA LR grid top value 2e-4 → 1e-4 (2e-4 caused catastrophic forgetting) | **Active** | `scripts/run_llm_training.sh` |
| Stage 2 warmup 20 → 50 steps | **Active** | `scripts/run_llm_training.sh` |
| Stage 2 LoRA rank grid [128] → [32, 64, 128] | **Active** | `scripts/run_llm_training.sh` |
| Stage 3: conclusion-only loss masking for reasoning tasks (masks description tokens; OPD tasks unmasked) | **Implemented, untested** | `src/train_llm.py`, `src/utils/model.py` |
| Stage 3: load from Stage 2 LoRA checkpoint (`adapter_model.bin` detection) | **Implemented, untested** | `src/train_llm.py` |

## Planned Improvements

### ✅ CLIP — Better Data Augmentation (IMPLEMENTED, DISABLED)
Added to `src/utils/dataset.py` (`CLIPPropertyUniqueDataset` and `TactileLLMDataset`):
- **Random rotation ±20°** — GelSight contact orientation varies with sensor placement
- **ColorJitter** (brightness=0.1, contrast=0.1, saturation=0.05, hue=0.0) — lighting variation; hue=0 since RGB encodes force direction
- **GaussianBlur** (kernel_size=5, σ=0.5–1.5, p=0.2) — simulates pressure/focus variation
Config keys `rotation_degrees`, `color_jitter`, `gaussian_blur` added to both training configs. Val/test datasets keep defaults (no augmentation).
All three are currently set to 0/false in configs — ColorJitter corrupts GelSight's RGB-encoded force direction and hurt combined accuracy from ~0.55 to ~0.21.

### ✅ CLIP — Unfreeze Last 2 Vision Layers (IMPLEMENTED, DISABLED)
Config key `unfreeze_last_n_layers` added to `train_clip_config.yaml`. Currently set to 0 (disabled).

### ✅ Stage 1 — Better Monitoring (IMPLEMENTED)
Changed `val_freq` from 400 → 200 in `run_llm_training.sh` Stage 1 block. Gives 16 checkpoints over 3200 steps.

### ✅ Stage 2 — LR Grid Refinement (IMPLEMENTED)
Replaced top LR `0.0002` with `0.0001` in `LORA_LRS` array. `2e-4` caused catastrophic regression of Stage 1 alignment.

### ✅ Stage 2 — Warmup Fix (IMPLEMENTED)
Increased warmup from 20 → 50 steps. Previous setting gave only ~1.25 effective warmup optimizer steps before full LR.

### ✅ Stage 2 — LoRA Rank Grid (IMPLEMENTED)
Added `r=32` and `r=64` to `RANKS` array. Full grid: `[32, 64, 128]`.

### ✅ Projector — LayerNorm (IMPLEMENTED)
Added `LayerNorm` between projection MLP layers in `model.py`. Stabilizes gradient flow from tactile features into LLM.

### ✅ Per-Group Gradient Clipping (IMPLEMENTED)
Replaced combined `clip_grad_norm_` with per-group clipping in `train_llm.py`. Prevents LoRA gradients from suppressing projection learning signal. Added gradient norm logging to tqdm.

### ✅ Validation Determinism (IMPLEMENTED)
Fixed val_loader `shuffle=True` → `shuffle=False` in `train_llm.py` for reproducible checkpoint selection.

### ❌ CLIP — Patch Token Mean Pooling (REVERTED)
Tried `hidden_states[-2][:, 1:].mean(dim=1)` instead of the CLS token. Peak test combined dropped from 0.553 → 0.421. CLS is trained by CLIP's contrastive objective to be the aggregated discriminative summary; unweighted patch mean loses this. ViFiCLIP reverted to `hidden_states[-2][:, 0]`.

### ❌ CLIP — Multi-Layer Feature Fusion (REVERTED)
Tried averaging CLS/patch means from layers `[-2, -6, -12]`. Did not improve over single-layer baseline. Reverted to `fusion_layers: [-2]`.

### ✅ CLIP — Pairwise Ranking Loss (IMPLEMENTED)
`train_clip.py` adds a margin ranking loss on top of CE. For each pair (i, j) where label_i > label_j, penalises if predicted expected-rank score_i <= score_j + margin. Weight controlled by `ranking_loss_weight` (currently 0.5). Does not replace CE loss.

### ✅ CLIP — Val Combined Save Criterion (IMPLEMENTED)
Checkpoint saved when val combined accuracy (all 3 properties simultaneously correct) improves. Prevents gaming by a single-property spike.

### TTA
`tta_passes` is set to `1` (disabled). TTA adds 3× test-time compute for marginal gains on this dataset. Re-enable only for final paper numbers.

## CLIP Encoder Investigation (2026-05): Evaluation Methodology & Class Imbalance

### Environment gotcha
CLIP/LLM training must run in the `octopi` conda env (`/data/samson/miniconda3/envs/octopi`). The base env's torch is built against a CUDA version newer than the driver and dies with "NVIDIA driver too old". All sweep/k-fold scripts `conda activate octopi` at the top.

### Evaluation methodology (important — read before trusting any CLIP metric)
- **The fixed 7-object val split is noisy and slightly easy.** K-fold CV val_mean (0.66) runs ~7pp below the locked-split val_mean (0.73) for the same config. Hyperparameter deltas ≤0.05 on the locked val are inside the noise floor.
- **`test_combined` (all-3-properties-correct) is a poor primary metric.** The test set is only 7 objects / ~38 samples, so combined moves in ~±0.14 steps and is a compounded statistic (≈ product of three ~0.6 per-property accuracies). Across k-folds, combined was *negatively* correlated with val_mean (r≈−0.77); per-property test tracked val within ~0.04. **Judge/report on mean per-property test accuracy; treat combined as secondary** (kept for paper comparability — confirm what the paper reports).
- **Selection vs reporting:** select configs on **val** (k-fold val_mean), never test. Report final numbers on the held-out test set *once*, as per-property mean ± std over ~3 seeds on the canonical split. Keep k-fold (non-canonical val splits) as ablation evidence, separate from the headline test number.

### Class-imbalance finding (root cause of weak hardness/roughness)
Per-property confusion (pooled over 5 k-fold checkpoints, `src/test_property_confusion.py`) showed **minority-class collapse**: the classifier defaults to training-frequent classes. Train imbalance ratios (max/min class count, train-only — no leakage): hardness 3.0, texture 2.4, roughness 1.3.
- Hardness: hard-class (2) recall and the rare soft class were the failures.
- Texture: rare class (2) recall collapsed to 0.24.
- Roughness: already balanced; nothing to fix.
- The lone soft (hardness=0) **test** object is `toilet_brush_bristles` (stiff bristles); train softs are all squishy (cotton/pillow/sponge). It stays at recall 0.00 even with weighting — an OOD/borderline label, not an imbalance failure.

### CLIP — Class-Balanced Loss (IMPLEMENTED, gated; default OFF)
`compute_class_weights` in `train_clip.py`, gated by config key `class_balanced_loss` (default `false`). Weights are applied per-property inside `ordinal_loss` (now accepts `weight=`). **Continuous imbalance-scaled** design: interpolate uniform→inverse-frequency by `alpha = 1 − 1/ratio` per property, so balanced properties (roughness) stay ~uniform automatically and skewed ones (hardness/texture) get gentler-than-full weighting. No threshold to tune; train-only. (Earlier full inverse-frequency variant regressed roughness — partly shared-trunk interference, since `CLIPClassifier.fc` is shared across the three heads.)

K-fold results (FOLD_SEED=0, config A = rot=0, ranking_w=0.5, ranking_m=0.3), pooled per-property test accuracy:

| variant | val_mean | test hardness | test roughness | test texture | test mean | test_combined |
|---------|----------|---------------|----------------|--------------|-----------|---------------|
| baseline (no weighting) | 0.663±0.063 | 0.632 | 0.611 | 0.679 | 0.641 | 0.279±0.131 |
| full inverse-freq (`cb`) | 0.668±0.074 | 0.779 | 0.568 | 0.689 | 0.679 | 0.363±0.101 |
| continuous scaled (`cbs`) | **run in progress (2026-05-28)** | — | — | — | — | — |

Full weighting: +0.15 hardness (hard-class recall 0.59→0.87), texture rare-class recall 0.24→0.64, but −0.04 roughness. Scaled variant aims to keep hardness/texture gains without the roughness regression — pending.

NOTE: working config currently has `class_balanced_loss: true` from the in-progress run. Flip back to `false` once the scaled result is settled, per the paper-change policy.

### Sweeps run (single-seed, low confidence — all within noise)
Rotation ∈ {0,15,20} and ranking (weight,margin) grids: no config beat baseline on val outside noise; rot=0 nominally won val_mean. `rotation_degrees` aug did not help. The apparent "rot=20 wins test" was a combined-metric / data-shuffle artifact, not reproducible.

### Tooling added (all CLIP-eval helpers)
- `scripts/run_clip_sweep.sh` — two-phase rotation then ranking sweep; backs up/restores config via trap.
- `scripts/run_clip_seed_reps.sh` — 3-seed × 2-config reproducibility check.
- `scripts/run_clip_kfold.sh` — K-fold CV (`K`, `FOLD_SEED`, `EXP_TAG` env vars). Backs up and restores both `configs/train_clip_config.yaml` and `src/utils/constants.py` via EXIT trap. Reports val_mean / test_mean / test_combined mean±std. **Do not edit this script while a run is in progress** (bash byte-offset corruption).
- `scripts/_build_kfold.py` — stratified (by hardness) fold builder over `TRAIN_OBJECTS ∪ VAL_OBJECTS`; writes `scripts/kfold/fold_<i>.json`. Never touches `TEST_OBJECTS`.
- `scripts/_patch_constants.py` — rewrites only the `TRAIN_OBJECTS`/`VAL_OBJECTS` list literals in `constants.py` from a fold JSON.
- `src/utils/parse_clip_log.py` — extracts best-val-epoch metrics from a run `log.txt` (incl. `val_mean`, `test_mean`, `test_combined`).
- `src/test_property_confusion.py` — pooled 3×3 per-property confusion matrices across checkpoints: `python src/test_property_confusion.py "exps/*_clip_kfold_0cb_f*"`.
