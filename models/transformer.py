# -*- coding: utf-8 -*-
"""Transformer 异常检测模型实现。"""

from __future__ import annotations

import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


class TransformerModel(nn.Module):
    """多层 Transformer 编码器用于序列重构。"""

    def __init__(
        self,
        input_c: int,
        window_length: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_c = input_c
        self.window_length = window_length

        self.embedding = DataEmbedding(input_c, d_model, dropout=dropout)
        self.pre_norm = nn.LayerNorm(d_model)

        attn_layers = [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(
                        mask_flag=False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=False,
                    ),
                    d_model,
                    n_heads,
                ),
                d_model,
                d_ff,
                dropout=dropout,
                activation="gelu",
            )
            for _ in range(num_layers)
        ]

        self.encoder = Encoder(attn_layers, norm_layer=nn.LayerNorm(d_model))

        self.reconstructor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, input_c),
        )

        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.smoothness_weight = 1e-2

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.embedding(x)
        normed = self.pre_norm(emb)
        enc_out, _ = self.encoder(normed)
        recon_delta = self.reconstructor(enc_out)
        recon = recon_delta + x  # residual reconstruction to ease optimisation
        return {
            "recon": recon,
            "encoded": enc_out,
        }

    def loss_function(self, x: torch.Tensor, outputs: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        recon = outputs["recon"]
        recon_loss = self.loss_fn(recon, x).mean()

        encoded = outputs["encoded"]
        temporal_diff = encoded[:, 1:, :] - encoded[:, :-1, :]
        smooth_loss = temporal_diff.pow(2).mean()

        total_loss = recon_loss + self.smoothness_weight * smooth_loss
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "smooth_loss": smooth_loss,
            "kl_loss": torch.zeros(1, device=x.device),
        }

    def compute_anomaly_score(self, x: torch.Tensor, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        recon = outputs["recon"]
        encoded = outputs["encoded"]

        recon_error = torch.mean((recon - x) ** 2, dim=-1)
        latent_energy = torch.mean(encoded.pow(2), dim=-1)
        score = 0.7 * recon_error + 0.3 * latent_energy
        return score[:, -1]
