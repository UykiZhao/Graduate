# Multivariate Anomaly Pipeline

简体中文 | [English](#english-version)

## 项目简介

本仓库基于 KDD 2019 论文 **OmniAnomaly** 的思想做迁移，并扩展为可插拔的通用异常检测训练管线，目前内置模型：
- 概率重构模型（`OmniAnomalyModel`）
- Transformer 重构模型（`TransformerModel`）

主要特性：
- TensorFlow 版本全面迁移至最新 PyTorch
- 统一的入口 `main.py` 与 `dataset.py`
- 训练过程实时可视化（tqdm）以及 Precision / Recall / F1 评估
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

### Transformer 模型

```bash
python main.py \
  --model Transformer \
  --dataset machine-1-1 \
  --window_length 100 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --num_layers 3 \
  --dropout 0.1 \
  --batch_size 256 \
  --learning_rate 1e-3 \
  --max_epoch 20 \
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

可额外指定 `--pretrained_epoch 10` 加载特定 epoch。

## 快速脚本

`scripts/` 目录为每个模型提供独立脚本，例如：
- `omni_anomaly_smd.sh`
- `transformer_smd.sh`

## Git 工作流

- 默认存在 `dev` 与 `main` 两个分支，日常开发在 `dev`
- GPU 服务器拉取 `dev` 分支进行训练与评估
- 确认稳定后再合并至 `main`

## English Version

### Overview

This repository refactors **OmniAnomaly** for the SMD dataset using PyTorch, featuring:
- Full migration from TensorFlow to PyTorch
- Unified entry via `main.py` and `dataset.py`
- Real-time training visualization with tqdm and full metrics reporting (Precision/Recall/F1)
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

Results are saved under `results/<machine_id>/<run_id>/summary.json`, checkpoints under `checkpoints/<machine_id>/<run_id>/`.

### Evaluation

```
python main.py --mode test --pretrained_run 20250101-120000 --dataset machine-1-1
```

Use `--pretrained_epoch` to load a specific epoch.

### Scripts

`scripts/` contains quick-launch templates for experimentation.

### Workflow

- Maintain `dev` as the active development branch, `main` as release branch
- The GPU server pulls `dev` for training and reports back
- Merge into `main` only after stability verification


