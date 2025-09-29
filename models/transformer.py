# -*- coding: utf-8 -*-
"""Transformer 对比模型。"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """基于 Transformer 的时序重构模型。"""

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

        self.input_proj = nn.Linear(input_c, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, input_c),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.input_proj(x)
        encoded = self.encoder(embedding)
        last_hidden = encoded[:, -1, :]
        recon = self.decoder(last_hidden)

        return {
            "recon": recon,
            "encoded": encoded,
        }

    def loss_function(self, x: torch.Tensor, outputs: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        recon = outputs["recon"]
        recon_loss = torch.mean(torch.sum((recon - x[:, -1, :]) ** 2, dim=-1))
        zero = torch.tensor(0.0, device=x.device)
        return {"loss": recon_loss, "recon_loss": recon_loss, "kl_loss": zero}

