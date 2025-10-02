# -*- coding: utf-8 -*-
"""统一训练流程。"""

from __future__ import annotations

import copy
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
class ExperimentConfig:
    """训练与评估配置。"""

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


class PipelineTrainer:
    """多模型统一训练流程。"""

    def __init__(self, config: ExperimentConfig) -> None:
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            verbose=False,
        )

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

        self.result_root = config.result_root / config.machine_id / config.model_name
        self.log_root = config.log_root / config.machine_id / config.model_name
        self.checkpoint_root = config.checkpoint_root

        self.result_root.mkdir(parents=True, exist_ok=True)
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

        if config.mode == "train":
            self.run_id = config.run_id or time.strftime("%Y%m%d-%H%M%S")
        else:
            if not config.pretrained_run:
                raise ValueError("测试模式必须提供 pretrained_run")
            self.run_id = config.pretrained_run

        self.run_result_dir = self.result_root / self.run_id
        self.run_result_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_path = self.checkpoint_root / f"{self.config.model_name}.pt"
        self.epoch_history: List[Dict[str, float]] = []
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_metrics: Optional[Dict[str, float]] = None
        self.best_epoch: int = 0

    def train(self) -> Dict[str, float]:
        summary = {
            "hyper_params": self._collect_hyper_params(),
            "epoch_metrics": [],
        }

        best_f1 = -float("inf")

        for epoch in range(1, self.config.max_epoch + 1):
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_recon = 0.0
            self.model.train()

            progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.max_epoch}")
            for batch_x, _ in progress:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                losses = self._compute_losses(batch_x, outputs)

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

            metrics = self.evaluate(save_outputs=False)
            metrics.update({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_kl": avg_kl,
                "train_recon": avg_recon,
            })
            summary["epoch_metrics"].append(metrics)

            if self.scheduler is not None:
                self.scheduler.step(metrics["f1"])

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.best_metrics = metrics
                self.best_epoch = epoch

        if self.best_state is None:
            self.best_state = copy.deepcopy(self.model.state_dict())
            final_metrics = self.evaluate(save_outputs=False)
            self.best_metrics = final_metrics
        else:
            self.model.load_state_dict(self.best_state)
            final_metrics = self.evaluate(save_outputs=True)

        self._save_best_checkpoint()

        summary.update({
            "best_epoch": self.best_epoch,
            "best_metrics": self.best_metrics,
            "final_metrics": final_metrics,
            "checkpoint_path": str(self.checkpoint_path),
        })

        with (self.run_result_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._save_config_file()

        return final_metrics

    def evaluate(self, save_outputs: bool = True) -> Dict[str, float]:
        self.model.eval()
        recon_errors = []

        with torch.no_grad():
            for batch_x, _ in tqdm(self.test_loader, desc="Testing", disable=not save_outputs):
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                if hasattr(self.model, "compute_anomaly_score"):
                    error = self.model.compute_anomaly_score(batch_x, outputs)
                else:
                    recon = outputs["recon"]
                    error = torch.mean((recon - batch_x[:, -1, :]) ** 2, dim=-1)
                recon_errors.extend(error.detach().cpu().numpy())

        recon_errors = np.array(recon_errors)
        test_labels = self.test_labels[-len(recon_errors) :]

        ratio = float(np.clip(self.config.anomaly_ratio, 1e-4, 0.99))
        quantile_threshold = float(np.quantile(recon_errors, 1 - ratio))
        quantile_threshold = max(quantile_threshold, float(np.min(recon_errors)))

        quantile_preds = (recon_errors >= quantile_threshold).astype(int)
        point_precision, point_recall, point_f1 = self._score_predictions(test_labels, quantile_preds)

        adjusted_preds = adjust_predictions(
            test_labels,
            quantile_preds,
            extend=max(1, self.config.window_length // 4),
        )
        adj_precision, adj_recall, adj_f1 = self._score_predictions(test_labels, adjusted_preds)

        best_f1, best_precision, best_recall, best_threshold = search_best_f1(
            recon_errors,
            test_labels,
            num_steps=2000,
        )

        metrics = {
            "threshold": quantile_threshold,
            "precision": float(adj_precision),
            "recall": float(adj_recall),
            "f1": float(adj_f1),
            "point_precision": float(point_precision),
            "point_recall": float(point_recall),
            "point_f1": float(point_f1),
            "best_threshold": float(best_threshold),
            "best_precision": float(best_precision),
            "best_recall": float(best_recall),
            "best_f1": float(best_f1),
        }

        if save_outputs:
            log_path = self.log_root / f"{self.run_id}_metrics.json"
            with log_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            np.save(self.run_result_dir / "predictions.npy", adjusted_preds)
            np.save(self.run_result_dir / "scores.npy", recon_errors)

        return metrics

    def evaluate_only(self) -> Dict[str, float]:
        if self.config.pretrained_epoch is not None:
            raise ValueError("当前仅支持按模型名称加载 checkpoint")

        checkpoint = self.checkpoint_root / f"{self.config.model_name}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"未找到模型权重: {checkpoint}")

        state_dict = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state_dict)
        return self.evaluate(save_outputs=True)

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

    def _compute_losses(self, batch_x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if hasattr(self.model, "loss_function"):
            return self.model.loss_function(batch_x, outputs, beta=self.config.beta_kl)

        recon = outputs["recon"]
        recon_loss = torch.mean(torch.sum((recon - batch_x[:, -1, :]) ** 2, dim=-1))
        return {
            "loss": recon_loss,
            "recon_loss": recon_loss,
            "kl_loss": torch.tensor(0.0, device=batch_x.device),
        }

    @staticmethod
    def _score_predictions(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float, float]:
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return float(precision), float(recall), float(f1)

    def _save_best_checkpoint(self) -> None:
        torch.save(self.model.state_dict(), self.checkpoint_path)

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
            "model_name": self.config.model_name,
            "n_heads": self.config.n_heads,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
        }
        params.update(self.config.extra_params)
        return params

    def _save_config_file(self) -> None:
        config_dict = asdict(self.config)
        config_dict.update({
            "device": str(self.device),
            "run_id": self.run_id,
            "checkpoint_path": str(self.checkpoint_path),
            "best_epoch": self.best_epoch,
        })
        for key, value in list(config_dict.items()):
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with (self.run_result_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
