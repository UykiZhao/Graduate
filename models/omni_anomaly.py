# -*- coding: utf-8 -*-
"""PyTorch 版 OmniAnomaly 模型。"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GRUEncoder, GaussianDiagonal, ReparameterizedNormal


class OmniAnomalyModel(nn.Module):
    """基于随机潜变量的重构模型。"""

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
            nn.Linear(d_ff, window_length * input_c),
        )

        self.input_noise_std = 0.05

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, _, _ = x.shape

        if self.training and self.input_noise_std > 0:
            noise = torch.randn_like(x) * self.input_noise_std
            x_noisy = x + noise
        else:
            x_noisy = x

        encoder_out = self.encoder(x_noisy)
        hidden = encoder_out[:, -1, :]
        hidden = self.fc_hidden_act(self.fc_hidden(hidden))

        prior_mean, prior_logvar = self.prior(hidden)
        posterior_mean, posterior_logvar = self.posterior(encoder_out[:, -1, :])
        z = self.reparameterize(posterior_mean, posterior_logvar)
        recon = self.decoder(z)
        recon = recon.view(batch, self.window_length, self.input_c)

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
        recon_loss = F.mse_loss(recon, x, reduction="mean")
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

    def compute_anomaly_score(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        recon = outputs["recon"]
        error = F.mse_loss(recon[:, -1, :], x[:, -1, :], reduction="none")
        return error.mean(dim=-1)

