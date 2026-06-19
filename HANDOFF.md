# Octopi binding investigation — handoff (2026-06-19)

## Goal
Reproduce the original Octopi LLM's **POM (property-object-match = video→object binding)**, which the released model reaches at ~0.59 slot accuracy (~8σ over chance). Our reproduction sits at **chance (~0.33)**. This session diagnosed why and shipped fixes. CLIP encoder + perception (OPD) are solid; **binding is the open wall.**

## Standing constraints (do not violate)
- **Do NOT change AdamW code/behavior** (weight_decay stays default 0.01 — confirmed this MATCHES the original code; both deviate from the paper's "no weight decay").
- Always check if a change is in the **original Octopi paper** before implementing; if not, mark as deviation/ablation. When ablations are within noise, prefer the **more paper-faithful** option (e.g. 5 frames over 8).
- Current work is **v1 WITHOUT SupCon** — don't propose contrastive fixes.
- CLIP/LLM training MUST run in the **`octopi` conda env** (`/data/samson/miniconda3/envs/octopi`); base env's torch is too new for the driver.
- `prompt_depth_vision=24` reproduces the original (our fork honors depth; upstream ignores it).
- Never select checkpoints by test metrics. POM partial `*_results.txt` swings wildly on <~100 items — only trust at n≳120.

## Biggest finding: our LLM cosine scheduler over-decayed vs the original
Two independent ways our `src/train_llm.py` starved gradient updates vs `/tmp/orig_train_llm.py`:
1. **Cosine `T_max` bug (FIXED).** Original sets `T_max = len(train_loader)/grad_accum` (full epoch, e.g. 625) but breaks early at `max_train_steps` (187 steps) → only traverses ~30% of the cosine → LR stays near peak (~80%). Ours used `T_max = max_train_steps/grad_accum` (187) → full decay to ~0. **Original delivered ~1.85× our total LoRA update.** Fix: `num_training_steps = int(len(train_loader)/grad_accum)`.
2. **Projector LR: original CONSTANT, ours decayed (gated fix).** Original uses 3 separate optimizers and schedules only the LLM/LoRA one; projector AdamW (2e-5) + encoder SGD are unscheduled = constant. Ours used one grouped optimizer + one scheduler → decayed projector too. Gated: `constant_projector_lr`.

Paper does **not** specify any LR schedule → cosine is a code-only detail, and the original's is misconfigured into ~constant, so **"constant LR" is the honest paper-consistent description**.

## Code changes this session (all gated flags default OFF unless noted)
- `src/train_llm.py`:
  - **T_max fix** (active, not gated).
  - `lr_schedule: cosine|constant` — `constant` = `get_constant_schedule_with_warmup` (no decay, all groups). For Stage-1-constant design.
  - `constant_projector_lr: true` — projector/encoder in separate unscheduled (constant) optimizer.
  - `distinct_delimiters: true` — per-slot `<tact_start_a/b/c>` tokens (NON-generalizing past 3 slots; didn't help).
  - `smart_delimiter_init: true` — seed `<tact_*>` from real word embeddings (`<tact_start_a>` ← mean("start","a")).
  - **Removed all deprecated aux losses** (conclusion-only/weight/POM-conclusion/OPD-consistency + consistency heads + `conclusion_start`/`property_labels` batch fields + config keys). Permutation augmentations KEPT.
  - **Double-pass eval fix**: `else`→`elif not configs["train"]` so `train=True,val=False` runs test once.
- `src/utils/model.py`: distinct-delimiter prompt rewrite in forward; removed consistency heads + loss blocks; per-frame sinusoidal PE already re-added (raw frame indices).
- `src/utils/dataset.py`: dropped `conclusion_start`/`property_labels` from `__getitem__`.
- `configs/train_llm_config.yaml`: removed dead aux-loss keys.

## Recommended design (user, 2026-06-19), built but UNLAUNCHED
- **Stage 1 → `lr_schedule=constant`** (projector+embeddings are from-scratch learners, base frozen, no moving-target → sustain LR).
- **Stage 2 → cosine (T_max-fixed)** = slight decay to ~80% for LoRA+embeddings+projector (moving-target risk → gentle coupled decay).
- Run it with `scripts/queue_faithsched_stage12.sh` (natural mean-init, NO distinct/smart-init — isolates the schedule fix). Built, not yet launched.
- Open ablation: `train_all_token_embeddings=False` (only `<tact_*>` rows) to protect base word-knowledge POM leans on — but probe shows old tokens barely drift, so likely minor; deviates from original (which trains full matrix).

## Results (POM canonical_slot; chance 0.333; binding would be ~0.5–0.6)
| run | what it tests | POM | n | PC | PSS | notes |
|---|---|---|---|---|---|---|
| Stage-1 alignment (no LoRA) | perception only | 0.214 | 176 | 0.14 | 0.34 | sub-chance = ~33% off-candidate gens; mappable-only ≈chance |
| baseline sinusoid f5 | mean-init, old sched | 0.272 | 176 | ~0.34 | ~0.37 | pre-session faithful run |
| distinct-delim | per-slot tokens, old sched | 0.309 | 137 | ~0.34 | ~0.39 | killed mid-eval; ≈chance |
| **smart-init** | better init, old sched | **0.387** | 173 | 0.38 | 0.45 | **best POM; +1.5σ over chance, +0.115 over baseline** |
| tmaxfix | T_max fix only, mean-init | ~0.333 | 24* | 0.47 | 0.29 | *PARTIAL; trending to chance (0.54→0.41→0.36→0.33 as n grew) |

**Interpretation:** Every mechanical link verified correct (encoder separates objects; OPD reads properties; format holds; q/k LoRA trains — `lora_B` non-zero, ‖ΔW‖/‖W‖≈1.5%; Stage 2 loads Stage 1 correctly; `<tact_start>`=single distinct token inserted right before the visual block). So chance binding is a **genuine learning/generalization failure, not a bug.** POM ≈chance means "emits valid candidates, assigns ~randomly" — crossing 0.21→0.39 is fixing candidate-adherence, NOT binding. **Init (smart-init) helped POM more than the update budget (tmaxfix).** PC/PSS (reasoning) rose with both; POM (held-out object binding) is the separate hard problem.

**ΔW litmus:** to check if any run actually got the T_max fix, compute `‖ΔW‖/‖W‖` on q/k from the saved adapter — ~1.5% = no fix, ~2.8% = fix active. (smart-init measured 1.2% = it did NOT have the fix; clean smart-init-alone result.)

## What's running (this server) as of handoff
- **GPU 4** `octopi_tmaxfix_f5` — tmaxfix Stage 2 eval (~130/537). Log `queue_llm_tmaxfix_f5_gpu4.log`. T_max fix ONLY.
- **GPU 7** `octopi_smartinit_se_f5` — smart-init eval (~526/537, basically done). Log `queue_llm_smartinit_se_f5_gpu7.log`.
- Wait for both to exit before moving.

## Open / next steps
1. **Launch `queue_faithsched_stage12.sh`** (T_max + Stage-1-constant + constant projector) — the fuller faithful-schedule test, not yet run.
2. **Gold-weights comparison** (the reason for the server move): load the original released Octopi weights and run our eval to confirm ~0.59 POM under our harness — establishes the target and validates the eval.
3. If faithsched POM ≈ chance too → binding is a generalization ceiling, not under-training; lever moves to the per-video signal (>1 token/video) or what the data can teach. Drop the addressing hacks (distinct delimiters don't generalize past 3 slots anyway).
4. Tracked TODO: normalize per-frame sinusoid indices to press-onset (0-based) — low priority.

## Resume on new server
1. `git clone` the repo (push the `fix` branch first; use a fresh token/credential helper — the old token was exposed in the remote URL).
2. Recreate the `octopi` conda env; `pip install -r requirements.txt`.
3. Regenerate data: `python src/utils/process_dataset.py --dataset_path dataset --output_path data --seed 0` then `python src/utils/generate_qa.py --data_path data --seed 0`. (Or copy `data/`, 205 MB.)
4. Copy the chosen encoder (1.2 GB) `exps/2026_06_18_23_31_22_train_clip_clip_sinusoid_f5_s0_e15_vc/encoder.pt`, or retrain via `scripts/queue_clip_sinusoid_sweep.sh` (FRAMES=5). The 67 GB `exps/` does NOT need to move.
5. Upload the gold weights; eval with a test-only `train_llm.py` run (see `scripts/eval_stage1_binding.sh` for the test-only pattern).

## Key paths
- Chosen encoder: `exps/2026_06_18_23_31_22_train_clip_clip_sinusoid_f5_s0_e15_vc/encoder.pt`
- LLM QA data: `/tmp/octopi_llm_frames8_encoder_full_stage12_data/` (regenerable)
- Original reference code (for diffs): `/tmp/orig_train_llm.py`, `/tmp/octopi_old/octopi-main/`
- Memory: `/home/samson/.claude/projects/-data-samson-octopi/memory/` (project_lr_schedule_findings.md, project_binding_experiments.md, etc.)
