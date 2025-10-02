# Multivariate Anomaly Pipeline

简体中文 | [English](#english-version)

## 项目简介

本仓库基于 KDD 2019 论文 **OmniAnomaly** 的思想做迁移，并扩展为可插拔的通用异常检测训练管线，当前以 Transformer 架构作为默认主力；内置模型：
- Transformer 重构模型（`TransformerModel`，默认）
- 概率重构模型（`OmniAnomalyModel`）

主要特性：
- TensorFlow 版本全面迁移至最新 PyTorch
- 统一的入口 `main.py` 与 `dataset.py`
- 训练过程实时可视化（tqdm）以及量化阈值、区间修正和最佳 F1 的多视角评估
- 按运行编号归档的 `checkpoints/`、`results/` 与 `logs/`
- CUDA、MPS、CPU 多硬件兼容

## 环境准备

推荐使用 Conda（Python ≥ 3.10）：

```bash
conda create -n omnanomaly python=3.10 -y
conda activate omnanomaly
pip install -r requirements.txt
```

## 数据准备

仅保留 SMD 数据集，原始文本位于 `ServerMachineDataset/`。运行时将自动转换到 `data/processed/`：

```bash
python main.py --dataset machine-1-1 --normalize --max_epoch 1
```

首次执行会自动生成 `.pkl` 缓存文件，无需手动调用其他脚本。

## 训练示例

### Transformer 模型（推荐）

```bash
python main.py \
  --model Transformer \
  --dataset machine-1-1 \
  --window_length 100 \
  --d_model 192 \
  --d_ff 320 \
  --n_heads 4 \
  --num_layers 4 \
  --dropout 0.2 \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --max_epoch 25 \
  --anomaly_ratio 0.4 \
  --normalize
```

### 概率重构模型（OmniAnomalyModel）

```bash
python main.py \
  --model OmniAnomaly \
  --dataset machine-1-1 \
  --window_length 100 \
  --d_model 128 \
  --d_ff 256 \
  --latent_dim 64 \
  --batch_size 256 \
  --learning_rate 1e-3 \
  --max_epoch 20 \
  --beta_kl 1.0 \
  --anomaly_ratio 0.5 \
  --normalize
```

训练日志将显示在终端，并同时写入 `logs/<machine_id>/<模型名称>/<run_id>_metrics.json`。模型与指标存放于：
- `checkpoints/<模型名称>.pt`
- `results/<machine_id>/<模型名称>/<run_id>/summary.json`


## 测试/评估

使用已经训练完成的 `run_id`：

```bash
python main.py --mode test --pretrained_run 20250101-120000 --dataset machine-1-1
```

可额外指定 `--pretrained_epoch 10` 加载特定 epoch。指标说明：
- `precision` / `recall` / `f1`：基于量化阈值并经过区间修正后的最终指标，适合作为部署基准。
- `point_precision` / `point_recall` / `point_f1`：未做区间修正的逐点指标，可快速判断阈值是否偏离。
- `best_*`：结合标注进行网格搜索得到的上界，仅用于诊断模型潜力或对齐文献结果。


## 快速脚本

`scripts/` 目录为每个模型提供独立脚本，例如：
- `transformer.sh`（携带推荐超参，可复制调整后运行）
- `omni_anomaly.sh`


## Git 工作流

- 默认存在 `dev` 与 `main` 两个分支，日常开发在 `dev`
- GPU 服务器拉取 `dev` 分支进行训练与评估
- 确认稳定后再合并至 `main`

## English Version

### Overview

This repository refactors **OmniAnomaly** for the SMD dataset using PyTorch, now defaulting to a Transformer reconstruction backbone, featuring:
- Full migration from TensorFlow to PyTorch
- Unified entry via `main.py` and `dataset.py`
- Real-time training visualization with tqdm and layered evaluation (quantile, window-adjusted, best-by-search)
- Run-based archives in `checkpoints/`, `results/`, and `logs/`
- Multi-backend support: CUDA, MPS, CPU

### Requirements

Create a Conda environment (Python ≥ 3.10):

```bash
conda create -n omnanomaly python=3.10 -y
conda activate omnanomaly
pip install -r requirements.txt
```

### Data

Only SMD is supported; raw `.txt` files in `ServerMachineDataset/` are automatically converted into `data/processed/` on-demand when running `main.py` for the first time.

### Training

```
python main.py \
  --model Transformer \
  --dataset machine-1-1 \
  --window_length 100 \
  --d_model 192 \
  --d_ff 320 \
  --n_heads 4 \
  --num_layers 4 \
  --dropout 0.2 \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --max_epoch 25 \
  --anomaly_ratio 0.4 \
  --normalize
```

Results are saved under `results/<machine_id>/<run_id>/summary.json`, checkpoints under `checkpoints/<model_name>.pt`.

### Evaluation

```
python main.py --mode test --pretrained_run 20250101-120000 --dataset machine-1-1
```

Use `--pretrained_epoch` to load a specific epoch. Metrics now report:

- `precision` / `recall` / `f1`: quantile-threshold predictions with segment adjustment (deployment ready).
- `point_precision` / `point_recall` / `point_f1`: raw per-point detection without adjustment for quick diagnosis.
- `best_*`: upper-bound metrics via grid search for diagnostics only.

### Scripts

`scripts/` contains quick-launch templates:
- `transformer.sh` (recommended defaults)
- `omni_anomaly.sh`

### Workflow

- Maintain `dev` as the active development branch, `main` as release branch
- The GPU server pulls `dev` for training and reports back
- Merge into `main` only after stability verification

