# -*- coding: utf-8 -*-
"""
学習済み best_model.pt を再ロードして、multi-label カテゴリ (damage_type / damage_loc)
も top-1 で予測する版を書き出す。

公平比較用: weighted variant が「閾値 0.5 でたくさん正例化 → recall 過多」になっていたため、
全てのモデルを 1 サンプル 1 クラスに揃えてから macro-F1 を比較する。

Usage:
    uv run python -m classification.predict_top1 \
        --val_csv  classification/results/unified_val_user.csv \
        --image_root /path/to/images \
        --models resnet50:linear_probe:resnet50_linear_probe/best_model.pt \
                 resnet50:finetune:resnet50_finetune/best_model.pt \
                 resnet50_weighted:finetune:resnet50_weighted_finetune/best_model.pt \
                 vit:linear_probe:vit_linear_probe/best_model.pt \
                 vit:finetune:vit_finetune/best_model.pt \
                 vit_weighted:finetune:vit_weighted_finetune/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification.data.dataset import BridgeInspectionDataset
from classification.data.label_definitions import ALL_LABEL_SETS
from classification.models.resnet50 import build_resnet50
from classification.models.resnet50_weighted import build_resnet50_weighted
from classification.models.vit import build_vit
from classification.models.vit_weighted import build_vit_weighted
from classification.models.clip_classifier import build_clip_base, build_clip_ft, build_clip_ft_weighted
from classification.train import VAL_TRANSFORM, VIT_TRANSFORM, SINGLE_LABEL_CATS, _build_clip_transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CLIP_CKPT_DIR = "logs_classification/bridgeclip_vitb32_unified/checkpoints"


def build_model(model_name: str, mode: str):
    if model_name == "resnet50":
        return build_resnet50(mode=mode), VAL_TRANSFORM
    if model_name == "resnet50_weighted":
        return build_resnet50_weighted(mode=mode), VAL_TRANSFORM
    if model_name == "vit":
        return build_vit(mode=mode), VIT_TRANSFORM
    if model_name == "vit_weighted":
        return build_vit_weighted(mode=mode), VIT_TRANSFORM
    if model_name == "clip_base":
        m = build_clip_base(mode=mode)
        _, vt = _build_clip_transforms(m)
        return m, vt
    if model_name == "clip_ft":
        m = build_clip_ft(mode=mode, ckpt_dir=DEFAULT_CLIP_CKPT_DIR)
        _, vt = _build_clip_transforms(m)
        return m, vt
    if model_name == "clip_ft_weighted":
        m = build_clip_ft_weighted(mode=mode, ckpt_dir=DEFAULT_CLIP_CKPT_DIR)
        _, vt = _build_clip_transforms(m)
        return m, vt
    raise ValueError(model_name)


@torch.no_grad()
def predict_top1(
    model_name: str,
    mode: str,
    ckpt_path: str,
    val_csv: str,
    image_root: str | None,
    out_csv: str,
    batch_size: int = 128,
    num_workers: int = 6,
) -> None:
    model, val_tf = build_model(model_name, mode)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(DEVICE).eval()

    ds = BridgeInspectionDataset(
        val_csv, transform=val_tf, image_root=image_root, strict_paths=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    src = pd.read_csv(val_csv)
    all_logits: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}
    for imgs, _ in tqdm(loader, desc=f"top1 {Path(out_csv).stem}"):
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        for cat in ALL_LABEL_SETS:
            all_logits[cat].append(out[cat].cpu())
    merged = {cat: torch.cat(all_logits[cat]).numpy() for cat in ALL_LABEL_SETS}

    rows = []
    for i in range(len(ds)):
        row: dict = {"image": src.iloc[i]["image"]}
        for cat, labels in ALL_LABEL_SETS.items():
            row[f"pred_{cat}"] = labels[int(merged[cat][i].argmax())]
        rows.append(row)
    out_df = src.merge(pd.DataFrame(rows), on="image", how="left")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"saved: {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv",    required=True)
    p.add_argument("--image_root", default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers",type=int, default=6)
    p.add_argument("--models", nargs="+", required=True,
                   help="model_name:mode:ckpt_relpath  (ckpt は classification/results/ 直下からの相対)")
    p.add_argument("--out_dir",  default="classification/results",
                   help="<out_dir>/<basename(ckpt_dir)>_top1_preds.csv に保存")
    args = p.parse_args()

    for spec in args.models:
        model_name, mode, ckpt_rel = spec.split(":")
        ckpt = Path(args.out_dir) / ckpt_rel
        # ckpt_dir = parent of best_model.pt; その名前を使う
        out_name = Path(ckpt).parent.name + "_top1_preds.csv"
        out_csv  = str(Path(args.out_dir) / out_name)
        print(f"\n=== {model_name} / {mode} / {ckpt} → {out_csv}")
        predict_top1(
            model_name=model_name,
            mode=mode,
            ckpt_path=str(ckpt),
            val_csv=args.val_csv,
            image_root=args.image_root,
            out_csv=out_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
