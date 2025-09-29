# -*- coding: utf-8 -*-
"""PyTorch 版 OmniAnomaly 模型。"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from layers import GRUEncoder, GaussianDiagonal, ReparameterizedNormal


class OmniAnomalyModel(nn.Module):
    """OmniAnomaly 模型核心实现。"""

    def __init__(
        self,
        input_c: int,
        window_length: int,
        d_model: int,
        latent_dim: int,
        d_ff: int,
        posterior_flow: bool = False,
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.input_c = input_c
        self.latent_dim = latent_dim
        self.posterior_flow = posterior_flow

        self.encoder = GRUEncoder(input_dim=input_c, hidden_dim=d_model)
        self.fc_hidden = nn.Linear(d_model, d_ff)
        self.fc_hidden_act = nn.ReLU()
        self.prior = GaussianDiagonal(d_ff, latent_dim)
        self.posterior = GaussianDiagonal(d_model, latent_dim)
        self.reparameterize = ReparameterizedNormal()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, input_c),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, window, _ = x.shape
        encoder_out = self.encoder(x)
        hidden = encoder_out[:, -1, :]
        hidden = self.fc_hidden_act(self.fc_hidden(hidden))

        prior_mean, prior_logvar = self.prior(hidden)
        posterior_mean, posterior_logvar = self.posterior(encoder_out[:, -1, :])
        z = self.reparameterize(posterior_mean, posterior_logvar)
        recon = self.decoder(z)

        return {
            "recon": recon,
            "posterior_mean": posterior_mean,
            "posterior_logvar": posterior_logvar,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
        }

    @staticmethod
    def kl_divergence(mean_q: torch.Tensor, logvar_q: torch.Tensor, mean_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
        """计算 KL 散度。"""

        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl = 0.5 * (
            logvar_p - logvar_q
            + (var_q + (mean_q - mean_p).pow(2)) / var_p
            - 1
        )
        return kl.sum(dim=-1)

    def loss_function(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], beta: float = 1.0) -> Dict[str, torch.Tensor]:
        recon = outputs["recon"]
        recon_loss = torch.mean(torch.sum((recon - x[:, -1, :]) ** 2, dim=-1))
        kl = torch.mean(
            self.kl_divergence(
                outputs["posterior_mean"],
                outputs["posterior_logvar"],
                outputs["prior_mean"],
                outputs["prior_logvar"],
            )
        )
        total_loss = recon_loss + beta * kl
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl}


