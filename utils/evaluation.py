# -*- coding: utf-8 -*-
"""评估与阈值工具。"""

from __future__ import annotations

import numpy as np


def search_best_f1(scores: np.ndarray, labels: np.ndarray, min_score: float | None = None, max_score: float | None = None, num_steps: int = 1000) -> tuple[float, float, float, float]:
    """通过网格搜索找到最佳 F1 阈值。"""

    if min_score is None:
        min_score = float(np.min(scores))
    if max_score is None:
        max_score = float(np.max(scores))

    thresholds = np.linspace(min_score, max_score, num_steps)
    best_f1 = 0.0
    best_threshold = thresholds[0]
    best_precision = 0.0
    best_recall = 0.0

    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    return best_f1, best_precision, best_recall, best_threshold


def adjust_predictions(labels: np.ndarray, preds: np.ndarray, extend: int = 50) -> np.ndarray:
    """对连续异常进行扩展修正。"""

    adjusted = preds.copy()
    anomaly_regions = np.where(np.diff(np.concatenate(([0], labels, [0]))))[0].reshape(-1, 2)

    for start, end in anomaly_regions:
        start = max(0, start - extend)
        end = min(len(labels), end + extend)
        if np.any(preds[start:end] == 1):
            adjusted[start:end] = 1

    return adjusted


