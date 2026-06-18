# -*- coding: utf-8 -*-
"""
ResNet50 マルチラベル分類モデル（Weighted BCE版）

resnet50.py と同一のアーキテクチャ。
学習データからクラスごとの pos_weight を計算し、
BCEWithLogitsLoss の不均衡補正に使用する。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from classification.data.labels import ALL_LABEL_SETS, NUM_CLASSES
from classification.models.resnet50 import ResNet50MultiLabel


def compute_pos_weight(csv_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """
    学習CSVからカテゴリごとの pos_weight を計算する。

    pos_weight[c] = (N - pos_count[c]) / pos_count[c]

    極端な値を避けるため [0.1, 100] にクリップする。

    Returns:
        dict[category, Tensor(shape: num_classes)]
    """
    df = pd.read_csv(csv_path)
    N = len(df)
    pos_weights: dict[str, torch.Tensor] = {}

    for cat, labels in ALL_LABEL_SETS.items():
        label2idx = {lbl: i for i, lbl in enumerate(labels)}
        counts = np.zeros(len(labels), dtype=np.float32)

        for val in df[cat].fillna(""):
            for token in str(val).split("|"):
                token = token.strip()
                if token in label2idx:
                    counts[label2idx[token]] += 1.0

        counts = np.clip(counts, 1.0, None)
        weights = (N - counts) / counts
        weights = np.clip(weights, 0.1, 100.0)
        pos_weights[cat] = torch.tensor(weights, dtype=torch.float32, device=device)

    return pos_weights


def build_resnet50_weighted(
    mode: str = "linear_probe",
    dropout: float = 0.3,
) -> ResNet50MultiLabel:
    """モデルを構築して返すファクトリ関数（Weighted BCE用）。"""
    model = ResNet50MultiLabel(mode=mode, dropout=dropout)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[ResNet50 {mode} weighted] 学習可能パラメータ: {n_trainable:,} / {n_total:,}")
    return model
