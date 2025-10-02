#!/bin/bash

# Transformer 模型运行示例

export CUDA_VISIBLE_DEVICES=0

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
  --normalize \
  --stride 1
