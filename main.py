# -*- coding: utf-8 -*-
"""统一训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from dataset import get_device
from trainers import PipelineTrainer, ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多模型异常检测统一入口")

    parser.add_argument("--model", type=str, default="Transformer", help="模型名称")
    parser.add_argument("--dataset", type=str, default="machine-1-1", help="SMD 子集")
    parser.add_argument("--window_length", type=int, default=100)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--beta_kl", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--anomaly_ratio", type=float, default=0.2)
    parser.add_argument("--threshold_alpha", type=float, default=0.05)
    parser.add_argument("--threshold_k", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--pretrained_run", type=str, default=None)
    parser.add_argument("--pretrained_epoch", type=int, default=None)
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--extra_params", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--logs_dir", type=str, default="logs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent

    device = get_device(args.device)
    extra_params = json.loads(args.extra_params) if args.extra_params else {}

    config = ExperimentConfig(
        machine_id=args.dataset,
        window_length=args.window_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epoch=args.max_epoch,
        beta_kl=args.beta_kl,
        device=str(device),
        anomaly_ratio=args.anomaly_ratio,
        threshold_alpha=args.threshold_alpha,
        threshold_k=args.threshold_k,
        stride=args.stride,
        normalize=args.normalize,
        result_root=project_root / args.results_dir,
        checkpoint_root=project_root / args.checkpoints_dir,
        log_root=project_root / args.logs_dir,
        mode=args.mode,
        run_id=args.run_id,
        pretrained_run=args.pretrained_run,
        pretrained_epoch=args.pretrained_epoch,
        notes=args.notes,
        extra_params=extra_params,
        model_name=args.model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    trainer = PipelineTrainer(config)
    if args.mode == "train":
        metrics = trainer.train()
    else:
        metrics = trainer.evaluate_only()

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
