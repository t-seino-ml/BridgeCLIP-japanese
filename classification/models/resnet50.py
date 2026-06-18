# -*- coding: utf-8 -*-
"""
ResNet50 マルチラベル分類モデル

2モード:
  - linear_probe (ゼロショット相当):
      ImageNet事前学習済み重みを凍結し、分類ヘッドのみ学習する。
      バックボーン自体はこのタスクのデータでは一切学習しない。

  - finetune (ファインチューニング):
      バックボーン全体を含め、モデル全体をこのタスクで学習する。

両モード共通:
  - 4つの分類ヘッド（健全度・対策区分・損傷種類・損傷部位）
  - 各ヘッドは BCEWithLogitsLoss（マルチラベル対応）
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from classification.data.labels import NUM_CLASSES


class ResNet50MultiLabel(nn.Module):
    """
    4ヘッドのマルチラベル分類モデル（ResNet50バックボーン）。

    Args:
        mode: "linear_probe" → バックボーン凍結
              "finetune"     → バックボーン含め全体学習
        dropout: 分類ヘッドのドロップアウト率
    """

    def __init__(self, mode: str = "linear_probe", dropout: float = 0.3):
        super().__init__()
        assert mode in ("linear_probe", "finetune"), \
            f"mode は 'linear_probe' か 'finetune' を指定してください。got: {mode}"
        self.mode = mode

        # ── バックボーン ──
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feature_dim = backbone.fc.in_features          # 2048
        backbone.fc = nn.Identity()                    # 最終FCを除去
        self.backbone = backbone

        # ── linear_probe モードでは全バックボーンを凍結 ──
        if mode == "linear_probe":
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── 4分類ヘッド ──
        self.heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_cls),
            )
            for cat, num_cls in NUM_CLASSES.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)          # (B, 2048)
        return {cat: head(features) for cat, head in self.heads.items()}

    def unfreeze_backbone(self) -> None:
        """linear_probe 後に全体ファインチューニングへ移行する際に呼ぶ。"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def build_resnet50(mode: str = "linear_probe", dropout: float = 0.3) -> ResNet50MultiLabel:
    """モデルを構築して返すファクトリ関数。"""
    model = ResNet50MultiLabel(mode=mode, dropout=dropout)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[ResNet50 {mode}] 学習可能パラメータ: {n_trainable:,} / {n_total:,}")
    return model
