# Repository Guidelines

## Project Structure & Module Organization
`main.py` orchestrates training and evaluation, while `dataset.py` handles device selection and dataset wiring. Model components reside in `models/` and reusable blocks in `layers/`. Training logic and experiment configs live in `trainers/`, with utilities in `utils/`. Raw SMD text files remain in `ServerMachineDataset/`; processed artifacts, checkpoints, metrics, and logs are routed to `results/`, `checkpoints/`, and `logs/` by run ID. Shell presets for repeatable experiments sit in `scripts/`.

## Build, Test, and Development Commands
Create the recommended environment and install dependencies before running anything:
```bash
conda create -n omnanomaly python=3.10 -y
conda activate omnanomaly
pip install -r requirements.txt
```
Train the default Transformer with `python main.py --model Transformer --dataset machine-1-1 --normalize`; override `--d_model`, `--d_ff`, and `--num_layers` using the presets in `scripts/transformer.sh` when experimenting. Use `--mode test --pretrained_run <run_id>` to evaluate a finished run, and `--pretrained_epoch` when you need a specific checkpoint. The shell templates in `scripts/` mirror proven hyperparameter sets; copy and adapt them rather than editing in-place.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, limit modules to clear responsibilities, and keep functions under 80–100 lines. Preserve existing type hints and dataclass-style config patterns; add docstrings when behaviour is non-obvious. Use snake_case for Python symbols and CLI flags, and kebab-case for shell script names. Before sending a PR, ensure files stay formatted by your editor (no trailing spaces) and `python -m compileall .` passes when feasible.

## Testing Guidelines
There is no unit-test suite yet; validation is performed through end-to-end evaluation. Always finish a training run with `python main.py --mode test --pretrained_run <run_id>` and review the emitted JSON under `logs/<machine_id>/<model>/<run_id>_metrics.json`. Track the deployment-ready `precision`/`recall`/`f1` (quantile + segment adjusted), and capture `point_f1` and `best_f1` deltas alongside the summary stored in `results/`. When reporting diagnostics, attach the relevant run ID and any anomaly ratio tweaks.

## Commit & Pull Request Guidelines
Git history favours concise, descriptive commit subjects (often in Simplified Chinese, e.g., “修复配置序列化与 Transformer 损失函数”). Keep commits focused and reference affected modules when helpful. Open feature and fix branches from `dev`; reserve `main` for validated releases. A pull request should: summarise motivation and outcomes, list key commands used, link related issues, and paste the latest evaluation metrics or screenshots of result dashboards. Tag reviewers familiar with the touched model family and note any follow-up debt.
