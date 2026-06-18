# -*- coding: utf-8 -*-
"""
ViT-B/16 マルチラベル分類モデル

2モード:
  - linear_probe (ゼロショット相当):
      ImageNet-21k + ImageNet-1k 事前学習済み重みを凍結し、分類ヘッドのみ学習。

  - finetune (ファインチューニング):
      バックボーン全体を含め、モデル全体をこのタスクで学習。

torchvision の ViT_B_16 を使用（pretrained=True）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights

from classification.data.labels import NUM_CLASSES


class ViTMultiLabel(nn.Module):
    """
    4ヘッドのマルチラベル分類モデル（ViT-B/16バックボーン）。

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

        # ── バックボーン（ViT-B/16） ──
        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        feature_dim = vit.heads.head.in_features    # 768
        vit.heads = nn.Identity()                   # 最終分類ヘッドを除去
        self.backbone = vit

        # ── linear_probe モードでは全バックボーンを凍結 ──
        if mode == "linear_probe":
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── 4分類ヘッド ──
        self.heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_cls),
            )
            for cat, num_cls in NUM_CLASSES.items()
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)          # (B, 768)
        return {cat: head(features) for cat, head in self.heads.items()}

    def unfreeze_backbone(self) -> None:
        """linear_probe 後に全体ファインチューニングへ移行する際に呼ぶ。"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]


def build_vit(mode: str = "linear_probe", dropout: float = 0.3) -> ViTMultiLabel:
    """モデルを構築して返すファクトリ関数。"""
    model = ViTMultiLabel(mode=mode, dropout=dropout)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[ViT-B/16 {mode}] 学習可能パラメータ: {n_trainable:,} / {n_total:,}")
    return model
