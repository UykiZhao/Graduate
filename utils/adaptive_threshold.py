# -*- coding: utf-8 -*-
"""Adaptive thresholding utilities for anomaly scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EMAAdaptiveThreshold:
    """Exponential moving average adaptive threshold.

    Maintains running mean and squared mean statistics using an EMA update.
    When evaluating a new score, anomalies are clipped to the current
    threshold before updating the baseline so that extreme values do not
    immediately skew the estimate.
    """

    alpha: float = 0.05
    k_sigma: float = 3.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.mean: float | None = None
        self.second_moment: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.second_moment is not None

    def fit(self, scores: np.ndarray) -> None:
        if scores.size == 0:
            raise ValueError("scores must be non-empty to fit threshold")
        baseline = np.asarray(scores, dtype=np.float64)
        self.mean = float(baseline.mean())
        self.second_moment = float((baseline ** 2).mean())

    def compute(self, scores: np.ndarray, reset: bool = True) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Threshold must be fitted before compute.")

        mean = float(self.mean)
        second = float(self.second_moment)
        thresholds = np.zeros_like(scores, dtype=np.float64)

        for idx, score in enumerate(scores):
            var = max(second - mean ** 2, self.eps)
            std = var ** 0.5
            threshold = mean + self.k_sigma * std
            thresholds[idx] = threshold

            clipped = score if score < threshold else threshold
            mean = (1 - self.alpha) * mean + self.alpha * clipped
            second = (1 - self.alpha) * second + self.alpha * (clipped ** 2)

        if not reset:
            self.mean = mean
            self.second_moment = second

        return thresholds

