# -*- coding: utf-8 -*-
"""SMD 数据集加载与预处理模块。"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 原始 SMD 数据目录
SMD_ROOT = PROJECT_ROOT / "ServerMachineDataset"

# 处理后的缓存目录
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def _load_txt(file_path: Path) -> np.ndarray:
    """读取单个 txt 文件并返回 float32 数组。"""

    data = np.loadtxt(str(file_path), delimiter=",", dtype=np.float32)
    return data


def _ensure_processed_dirs() -> None:
    """确保缓存目录存在。"""

    (PROCESSED_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (PROCESSED_ROOT / "test").mkdir(parents=True, exist_ok=True)
    (PROCESSED_ROOT / "test_label").mkdir(parents=True, exist_ok=True)
    (PROCESSED_ROOT / "metadata").mkdir(parents=True, exist_ok=True)


def convert_smd_txt_to_pkl(force: bool = False) -> None:
    """将 SMD 原始 txt 数据转换为 pkl 文件。

    Args:
        force: 若为 True，将强制重新生成所有 pkl 文件。
    """

    _ensure_processed_dirs()

    meta: Dict[str, Dict[str, str]] = {}

    for subset in ["train", "test", "test_label"]:
        subset_dir = SMD_ROOT / subset
        target_dir = PROCESSED_ROOT / subset
        for txt_path in sorted(subset_dir.glob("*.txt")):
            machine_id = txt_path.stem
            target_file = target_dir / f"{machine_id}.pkl"

            if target_file.exists() and not force:
                continue

            array = _load_txt(txt_path)
            with target_file.open("wb") as f:
                pickle.dump(array, f)

            meta.setdefault(machine_id, {})[subset] = str(target_file.relative_to(PROJECT_ROOT))

    with (PROCESSED_ROOT / "metadata" / "smd_files.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_pickle(path: Path) -> np.ndarray:
    """加载 pkl 文件。"""

    with path.open("rb") as f:
        data = pickle.load(f)
    return data


def sliding_window(data: np.ndarray, window_size: int) -> np.ndarray:
    """根据窗口长度生成滑动窗口。"""

    if window_size <= 0:
        raise ValueError("window_size 必须为正整数")

    if data.shape[0] < window_size:
        raise ValueError("数据长度小于窗口长度，无法生成窗口")

    windows = []
    for i in range(data.shape[0] - window_size + 1):
        windows.append(data[i : i + window_size])
    return np.stack(windows, axis=0)


@dataclass
class SMDWindowConfig:
    """SMD 数据窗口配置。"""

    machine_id: str
    window_length: int
    stride: int = 1
    normalize: bool = True


class SMDDataset(Dataset):
    """用于 OmniAnomaly 训练和推理的 SMD 数据集。"""

    def __init__(
        self,
        config: SMDWindowConfig,
        subset: str,
        cache_dir: Optional[Path] = None,
    ) -> None:
        if subset not in {"train", "test", "test_label"}:
            raise ValueError("subset 仅支持 'train'、'test'、'test_label'")

        self.config = config
        self.subset = subset
        self.cache_dir = cache_dir or PROCESSED_ROOT

        convert_smd_txt_to_pkl(force=False)

        data_path = self.cache_dir / subset / f"{config.machine_id}.pkl"
        if not data_path.exists():
            raise FileNotFoundError(f"未找到 {data_path}")

        raw_data = load_pickle(data_path)
        self.raw_data = raw_data.astype(np.float32)
        self.feature_dim = self.raw_data.shape[1]

        if subset == "test_label":
            self.data = torch.from_numpy(raw_data.astype(np.int64))
            self.windows = None
            self.normalized = None
        else:
            if config.normalize:
                mean = self.raw_data.mean(axis=0, keepdims=True)
                std = self.raw_data.std(axis=0, keepdims=True) + 1e-6
                normalized = (self.raw_data - mean) / std
            else:
                normalized = self.raw_data

            windowed = sliding_window(normalized, config.window_length)
            if config.stride > 1:
                windowed = windowed[:: config.stride]

            self.normalized = torch.from_numpy(normalized.astype(np.float32))
            self.windows = torch.from_numpy(windowed.astype(np.float32))
            self.data = self.windows

    def __len__(self) -> int:
        if self.subset == "test_label":
            return self.data.shape[0]
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.subset == "test_label":
            raise RuntimeError("test_label 数据集不支持通过 Dataset 迭代")

        window = self.windows[idx]
        dummy_label = torch.zeros(1, dtype=torch.float32)
        return window, dummy_label


def load_smd_labels(machine_id: str) -> np.ndarray:
    """加载测试标签。"""

    label_path = PROCESSED_ROOT / "test_label" / f"{machine_id}.pkl"
    if not label_path.exists():
        convert_smd_txt_to_pkl(force=False)
    labels = load_pickle(label_path).astype(np.int64)
    return labels


def get_device(prefer_device: Optional[str] = None) -> torch.device:
    """按照 CUDA > MPS > CPU 的优先级获取设备。"""

    if prefer_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def summarize_run(
    machine_id: str,
    hyper_params: Dict[str, float],
    metrics: Dict[str, float],
    output_dir: Path,
) -> None:
    """保存训练运行摘要。"""

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "machine_id": machine_id,
        "hyper_params": hyper_params,
        "metrics": metrics,
    }

    summary_path = output_dir / "summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


