# -*- coding: utf-8 -*-
"""
ViT-B/16 マルチラベル分類モデル（Weighted BCE版）

vit.py と同一のアーキテクチャ。
学習データからクラスごとの pos_weight を計算し、
BCEWithLogitsLoss の不均衡補正に使用する。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from classification.data.label_definitions import ALL_LABEL_SETS, NUM_CLASSES
from classification.models.vit import ViTMultiLabel
from classification.models.resnet50_weighted import compute_pos_weight  # 共通関数


def build_vit_weighted(
    mode: str = "linear_probe",
    dropout: float = 0.3,
) -> ViTMultiLabel:
    """モデルを構築して返すファクトリ関数（Weighted BCE用）。"""
    model = ViTMultiLabel(mode=mode, dropout=dropout)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[ViT-B/16 {mode} weighted] 学習可能パラメータ: {n_trainable:,} / {n_total:,}")
    return model
