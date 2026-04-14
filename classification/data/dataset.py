# -*- coding: utf-8 -*-
"""
マルチラベル分類用 PyTorch Dataset

ラベル付きCSV（extract_labels.py の出力）を読み込み、
画像と4カテゴリのマルチホットベクトルを返す。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from classification.data.label_definitions import (
    KENZENUDO_LABELS,
    TAISAKU_LABELS,
    DAMAGE_TYPE_LABELS,
    DAMAGE_LOC_LABELS,
    ALL_LABEL_SETS,
)


class BridgeInspectionDataset(Dataset):
    """
    橋梁点検マルチラベル分類データセット。

    各サンプルは以下を返す:
        image_tensor : torch.Tensor  shape (C, H, W)
        labels       : dict[str, torch.Tensor]  各カテゴリのマルチホットベクトル

    Args:
        csv_path   : extract_labels.py が出力したラベル付きCSVのパス
        transform  : torchvision の前処理 (None の場合は PIL Image をそのまま返す)
        filter_valid: True の場合、4カテゴリすべて抽出に成功した行のみ使用
        categories : 使用するカテゴリのリスト（デフォルトは全4カテゴリ）
    """

    CATEGORIES = ["kenzenudo", "taisaku", "damage_type", "damage_loc"]

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        filter_valid: bool = False,
        categories: Optional[list[str]] = None,
    ):
        self.transform = transform
        self.categories = categories or self.CATEGORIES

        df = pd.read_csv(csv_path)
        # 必要カラムの確認
        for col in ["image"] + self.CATEGORIES:
            assert col in df.columns, f"カラム '{col}' がCSVに存在しません"

        if filter_valid:
            valid_mask = df[[f"{c}_valid" for c in self.CATEGORIES]].all(axis=1)
            df = df[valid_mask].reset_index(drop=True)
            print(f"有効行のみ使用: {len(df)} 行")

        self.df = df

        # ラベル → インデックス マッピング
        self.label2idx: dict[str, dict[str, int]] = {
            cat: {lbl: i for i, lbl in enumerate(labels)}
            for cat, labels in ALL_LABEL_SETS.items()
        }
        self.num_classes = {cat: len(labels) for cat, labels in ALL_LABEL_SETS.items()}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, dict]:
        row = self.df.iloc[idx]

        # ── 画像読み込み ──
        img_path = str(row["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # 壊れた画像は黒画像で代替（学習中はclean_csv_by_image_decodeで除去済みのはず）
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform is not None:
            image = self.transform(image)

        # ── ラベルをマルチホットベクトルに変換 ──
        label_dict: dict[str, torch.Tensor] = {}
        for cat in self.categories:
            labels = ALL_LABEL_SETS[cat]
            vec = torch.zeros(len(labels), dtype=torch.float32)
            raw_val = str(row.get(cat, "") or "")
            for token in raw_val.split("|"):
                token = token.strip()
                if token in self.label2idx[cat]:
                    vec[self.label2idx[cat][token]] = 1.0
            label_dict[cat] = vec

        return image, label_dict

    def get_label_names(self, cat: str, vec: torch.Tensor) -> list[str]:
        """マルチホットベクトルをラベル名リストに変換する（デバッグ用）。"""
        labels = ALL_LABEL_SETS[cat]
        return [labels[i] for i, v in enumerate(vec) if v > 0.5]


def build_label_matrix(csv_path: str, category: str) -> np.ndarray:
    """
    指定カテゴリのラベル行列を返す（shape: [N, num_classes]）。
    評価スクリプトでの一括処理に使用。
    """
    df = pd.read_csv(csv_path)
    labels = ALL_LABEL_SETS[category]
    label2idx = {lbl: i for i, lbl in enumerate(labels)}
    matrix = np.zeros((len(df), len(labels)), dtype=np.float32)
    for i, row in df.iterrows():
        for token in str(row.get(category, "") or "").split("|"):
            token = token.strip()
            if token in label2idx:
                matrix[i, label2idx[token]] = 1.0
    return matrix
