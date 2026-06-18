# -*- coding: utf-8 -*-
"""
CLIP image encoder + 4 分類ヘッド

Backbone:
  - "clip_base":  事前学習済み CLIP (LAION2B 等)
  - "clip_ft":    橋梁ドメインで対照学習した CLIP ckpt を読み込み

Mode:
  - "linear_probe":   backbone 凍結、分類ヘッドのみ学習
  - "finetune":       backbone も含めて全体を学習

ヘッドは ResNet50MultiLabel と同じ 4 つ:
  - kenzenudo, taisaku (単一ラベル, CE で学習)
  - damage_type, damage_loc (マルチラベル, BCE で学習)

train.py から `--model clip_base` / `--model clip_ft` で呼び出せるよう、
build_*() ファクトリを提供する。
"""

from __future__ import annotations

import json
from pathlib import Path

import open_clip
import torch
import torch.nn as nn

from classification.data.label_definitions import NUM_CLASSES


def _find_best_ckpt(ckpt_dir: str) -> str:
    """results.jsonl から val_loss 最小の実在 ckpt を返す。"""
    p = Path(ckpt_dir)
    rf = p / "results.jsonl"
    if not rf.exists():
        raise FileNotFoundError(f"results.jsonl が見つかりません: {rf}")
    entries: list[tuple[float, int]] = []
    with open(rf) as f:
        for line in f:
            d = json.loads(line)
            entries.append((d.get("clip_val_loss", float("inf")), d.get("epoch", -1)))
    entries.sort()
    for loss, epoch in entries:
        cand = p / f"epoch_{epoch}.pt"
        if cand.exists():
            print(f"[clip_classifier] CLIP ckpt 採用: epoch {epoch} (val_loss={loss:.4f})")
            return str(cand)
    raise FileNotFoundError(f"実在する ckpt が無い: {ckpt_dir}")


class CLIPMultiLabel(nn.Module):
    """
    CLIP image encoder + 4 ヘッド分類器。

    Args:
        mode: "linear_probe" or "finetune"
        model_name: open_clip モデル名（既定 ViT-B-32）
        pretrained: open_clip 事前学習重み名
        ckpt_path: CLIP-FT ckpt のパス（指定時はその重みを上書きロード）
        ckpt_dir:  CLIP-FT ckpt ディレクトリ（results.jsonl から best を採用）
        dropout: ヘッドのドロップアウト
    """

    def __init__(
        self,
        mode: str = "linear_probe",
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        ckpt_path: str | None = None,
        ckpt_dir: str | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert mode in ("linear_probe", "finetune")
        self.mode = mode

        if ckpt_dir is not None:
            ckpt_path = _find_best_ckpt(ckpt_dir)

        # ── open_clip backbone ──
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        if ckpt_path is not None:
            print(f"[clip_classifier] FT ckpt 読み込み: {ckpt_path}")
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            clip_model.load_state_dict(sd, strict=False)

        # 画像エンコーダだけ使う（テキストエンコーダは捨てる）
        self.visual = clip_model.visual
        # CLIP visual の出力次元
        feature_dim = clip_model.visual.output_dim  # ViT-B-32 なら 512

        # ── 凍結設定 ──
        if mode == "linear_probe":
            for p in self.visual.parameters():
                p.requires_grad = False

        # ── 4 分類ヘッド ──
        self.heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_cls),
            )
            for cat, num_cls in NUM_CLASSES.items()
        })

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # CLIP image encoder は eval mode でも仕様上問題ないが、
        # linear_probe では gradient を切る
        if self.mode == "linear_probe":
            with torch.no_grad():
                feats = self.visual(x)
        else:
            feats = self.visual(x)
        # 数値安定のため float32 にキャスト（CLIP は AMP で fp16/bf16 だと head が不安定になりがち）
        feats = feats.float()
        return {cat: head(feats) for cat, head in self.heads.items()}

    def unfreeze_backbone(self) -> None:
        for p in self.visual.parameters():
            p.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_clip_base(mode: str = "linear_probe", dropout: float = 0.3) -> CLIPMultiLabel:
    """事前学習 CLIP（laion2b）を使う分類器。"""
    model = CLIPMultiLabel(
        mode=mode,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        ckpt_path=None,
        dropout=dropout,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[CLIP-base {mode}] 学習可能パラメータ: {n_trainable:,} / {n_total:,}")
    return model


def build_clip_ft(
    mode: str = "linear_probe",
    ckpt_path: str | None = None,
    ckpt_dir: str | None = None,
    dropout: float = 0.3,
) -> CLIPMultiLabel:
    """橋梁ドメインで対照学習した CLIP-FT を使う分類器。"""
    if ckpt_path is None and ckpt_dir is None:
        # 既定: bridgeclip_vitb32_unified の best
        ckpt_dir = "logs_classification/bridgeclip_vitb32_unified/checkpoints"
    model = CLIPMultiLabel(
        mode=mode,
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        ckpt_path=ckpt_path,
        ckpt_dir=ckpt_dir,
        dropout=dropout,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"[CLIP-FT {mode}] 学習可能パラメータ: {n_trainable:,} / {n_total:,}")
    return model


def build_clip_ft_weighted(
    mode: str = "linear_probe",
    ckpt_path: str | None = None,
    ckpt_dir: str | None = None,
    dropout: float = 0.3,
) -> CLIPMultiLabel:
    """CLIP-FT を使う分類器（weighted BCE 用、構造は同じ）。"""
    return build_clip_ft(mode=mode, ckpt_path=ckpt_path, ckpt_dir=ckpt_dir, dropout=dropout)
