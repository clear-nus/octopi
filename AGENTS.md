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
