# -*- coding: utf-8 -*-
"""OmniAnomaly 训练器。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import SMDDataset, SMDWindowConfig, load_smd_labels
from models import OmniAnomalyModel, TransformerModel
from utils import search_best_f1, adjust_predictions


@dataclass
class TrainingConfig:
    """训练配置。"""

    machine_id: str
    window_length: int
    d_model: int
    d_ff: int
    latent_dim: int
    batch_size: int
    learning_rate: float
    max_epoch: int
    beta_kl: float
    device: str
    anomaly_ratio: float
    result_root: Path
    checkpoint_root: Path
    log_root: Path
    mode: str = "train"
    run_id: Optional[str] = None
    pretrained_run: Optional[str] = None
    pretrained_epoch: Optional[int] = None
    normalize: bool = True
    stride: int = 1
    notes: Optional[str] = None
    extra_params: Dict[str, float] = field(default_factory=dict)
    model_name: str = "OmniAnomaly"
    n_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


class OmniAnomalyTrainer:
    """统一训练流程。"""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        dataset_cfg = SMDWindowConfig(
            machine_id=config.machine_id,
            window_length=config.window_length,
            stride=config.stride,
            normalize=config.normalize,
        )

        self.train_dataset = SMDDataset(dataset_cfg, subset="train")
        self.test_dataset = SMDDataset(dataset_cfg, subset="test")
        self.test_labels = load_smd_labels(config.machine_id)

        self.model = self._build_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
    def _build_model(self) -> torch.nn.Module:
        if self.config.model_name == "Transformer":
            return TransformerModel(
                input_c=self.train_dataset.feature_dim,
                window_length=self.config.window_length,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                d_ff=self.config.d_ff,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
            ).to(self.device)

        return OmniAnomalyModel(
            input_c=self.train_dataset.feature_dim,
            window_length=self.config.window_length,
            d_model=self.config.d_model,
            latent_dim=self.config.latent_dim,
            d_ff=self.config.d_ff,
        ).to(self.device)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.machine_checkpoint_root = config.checkpoint_root / config.machine_id
        self.machine_result_root = config.result_root / config.machine_id
        self.machine_log_root = config.log_root / config.machine_id

        self.machine_checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.machine_result_root.mkdir(parents=True, exist_ok=True)
        self.machine_log_root.mkdir(parents=True, exist_ok=True)

        if config.mode == "train":
            self.run_id = config.run_id or time.strftime("%Y%m%d-%H%M%S")
        else:
            if not config.pretrained_run:
                raise ValueError("测试模式必须提供 pretrained_run")
            self.run_id = config.pretrained_run

        self.run_checkpoint_dir = self.machine_checkpoint_root / self.run_id
        self.run_result_dir = self.machine_result_root / self.run_id

        if config.mode == "train":
            self.run_checkpoint_dir.mkdir(parents=True, exist_ok=False)
            self.run_result_dir.mkdir(parents=True, exist_ok=False)
        else:
            if not self.run_checkpoint_dir.exists():
                raise FileNotFoundError(f"未找到指定 run_id 的检查点目录: {self.run_checkpoint_dir}")
            self.run_result_dir.mkdir(parents=True, exist_ok=True)

        self.epoch_history: List[Dict[str, float]] = []

    def train(self) -> Dict[str, float]:
        summary = {
            "train_losses": [],
            "kl_losses": [],
            "recon_losses": [],
            "hyper_params": self._collect_hyper_params(),
        }

        for epoch in range(1, self.config.max_epoch + 1):
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_recon = 0.0
            self.model.train()

            progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.max_epoch}")
            for batch_x, _ in progress:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                losses = self.model.loss_function(batch_x, outputs, beta=self.config.beta_kl)

                loss = losses["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_kl += losses["kl_loss"].item()
                epoch_recon += losses["recon_loss"].item()

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    recon=f"{losses['recon_loss'].item():.4f}",
                    kl=f"{losses['kl_loss'].item():.4f}",
                )

            num_batches = len(self.train_loader)
            avg_loss = epoch_loss / num_batches
            avg_kl = epoch_kl / num_batches
            avg_recon = epoch_recon / num_batches

            summary["train_losses"].append(avg_loss)
            summary["kl_losses"].append(avg_kl)
            summary["recon_losses"].append(avg_recon)

            self._save_checkpoint(
                epoch,
                {
                    "loss": avg_loss,
                    "kl_loss": avg_kl,
                    "recon_loss": avg_recon,
                },
            )

        metrics = self.evaluate()
        summary.update(metrics)
        summary["epoch_history"] = self.epoch_history

        with (self.run_result_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._save_config_files()

        return metrics

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        recon_errors = []

        with torch.no_grad():
            for batch_x, _ in tqdm(self.test_loader, desc="Testing"):
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                recon = outputs["recon"]
                error = torch.mean((recon - batch_x[:, -1, :]) ** 2, dim=-1)
                recon_errors.extend(error.cpu().numpy())

        recon_errors = np.array(recon_errors)
        test_labels = self.test_labels[-len(recon_errors) :]

        best_f1, best_precision, best_recall, threshold = search_best_f1(
            recon_errors, test_labels, num_steps=2000
        )
        preds = (recon_errors >= threshold).astype(int)
        adjusted_preds = adjust_predictions(test_labels, preds)

        tp = np.sum((adjusted_preds == 1) & (test_labels == 1))
        fp = np.sum((adjusted_preds == 1) & (test_labels == 0))
        fn = np.sum((adjusted_preds == 0) & (test_labels == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
            "best_f1": float(best_f1),
            "best_precision": float(best_precision),
            "best_recall": float(best_recall),
        }

        result_path = self.machine_log_root / f"{self.run_id}_metrics.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        pred_path = self.run_result_dir / "predictions.npy"
        np.save(pred_path, adjusted_preds)
        score_path = self.run_result_dir / "scores.npy"
        np.save(score_path, recon_errors)

        return metrics

    def evaluate_only(self) -> Dict[str, float]:
        self._load_checkpoint()
        return self.evaluate()

    def _save_checkpoint(self, epoch: int, losses: Dict[str, float]) -> None:
        checkpoint_path = self.run_checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        self._append_epoch_log(epoch, checkpoint_path, losses)

    def _load_checkpoint(self) -> None:
        if self.config.pretrained_epoch is not None:
            ckpt_path = self.run_checkpoint_dir / f"epoch_{self.config.pretrained_epoch}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"未找到指定 epoch 检查点: {ckpt_path}")
        else:
            checkpoints = sorted(self.run_checkpoint_dir.glob("epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("该 run 目录下没有检查点文件")
            ckpt_path = checkpoints[-1]
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

    def _append_epoch_log(
        self, epoch: int, checkpoint_path: Path, losses: Dict[str, float]
    ) -> None:
        last_entry = self.epoch_history[-1] if self.epoch_history else None
        if last_entry and last_entry.get("epoch") == epoch:
            return
        entry = {
            "epoch": epoch,
            "checkpoint": str(checkpoint_path.relative_to(self.config.checkpoint_root.parent)),
            "loss": losses.get("loss"),
            "kl_loss": losses.get("kl_loss"),
            "recon_loss": losses.get("recon_loss"),
        }
        self.epoch_history.append(entry)

    def _collect_hyper_params(self) -> Dict[str, float]:
        params = {
            "window_length": self.config.window_length,
            "d_model": self.config.d_model,
            "d_ff": self.config.d_ff,
            "latent_dim": self.config.latent_dim,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "beta_kl": self.config.beta_kl,
            "stride": self.config.stride,
            "normalize": self.config.normalize,
            "anomaly_ratio": self.config.anomaly_ratio,
        }
        params.update(self.config.extra_params)
        return params

    def _save_config_files(self) -> None:
        config_dict = asdict(self.config)
        config_dict["device"] = str(self.device)
        config_dict["run_id"] = self.run_id
        config_dict["checkpoint_root"] = str(self.config.checkpoint_root)
        config_dict["result_root"] = str(self.config.result_root)
        config_dict["log_root"] = str(self.config.log_root)

        with (self.run_result_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)


