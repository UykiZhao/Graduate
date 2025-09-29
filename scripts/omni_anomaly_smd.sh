#!/bin/bash

# 快速运行示例脚本

export CUDA_VISIBLE_DEVICES=0

python main.py \
  --model OmniAnomaly \
  --dataset machine-1-1 \
  --window_length 100 \
  --d_model 128 \
  --d_ff 256 \
  --latent_dim 64 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --max_epoch 10 \
  --beta_kl 1.0 \
  --anomaly_ratio 0.5 \
  --normalize \
  --stride 1

python main.py \
  --model Transformer \
  --dataset machine-1-1 \
  --window_length 100 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --num_layers 2 \
  --dropout 0.1 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --max_epoch 10 \
  --beta_kl 0.0 \
  --anomaly_ratio 0.5 \
  --normalize \
  --stride 1


