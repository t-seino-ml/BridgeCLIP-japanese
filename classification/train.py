# -*- coding: utf-8 -*-
"""
ResNet50 / ViT マルチラベル分類 学習スクリプト

使い方:
    # ResNet50 linear_probe（ゼロショット相当）
    python -m classification.train \
        --model resnet50 --mode linear_probe \
        --train_csv classification/results/labeled_train.csv \
        --val_csv   classification/results/labeled_val.csv \
        --out_dir   classification/results/resnet50_linear_probe \
        --epochs 20 --lr 1e-3 --batch_size 64

    # ResNet50 finetune（全体学習）
    python -m classification.train \
        --model resnet50 --mode finetune \
        --train_csv classification/results/labeled_train.csv \
        --val_csv   classification/results/labeled_val.csv \
        --out_dir   classification/results/resnet50_finetune \
        --epochs 30 --lr 1e-4 --batch_size 32

    # ViT linear_probe
    python -m classification.train \
        --model vit --mode linear_probe \
        --train_csv classification/results/labeled_train.csv \
        --val_csv   classification/results/labeled_val.csv \
        --out_dir   classification/results/vit_linear_probe \
        --epochs 20 --lr 1e-3 --batch_size 64

    # ViT finetune
    python -m classification.train \
        --model vit --mode finetune \
        --train_csv classification/results/labeled_train.csv \
        --val_csv   classification/results/labeled_val.csv \
        --out_dir   classification/results/vit_finetune \
        --epochs 30 --lr 5e-5 --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from classification.data.dataset import BridgeInspectionDataset
from classification.data.label_definitions import ALL_LABEL_SETS
from classification.models.resnet50 import build_resnet50
from classification.models.vit import build_vit
from classification.models.resnet50_weighted import build_resnet50_weighted, compute_pos_weight
from classification.models.vit_weighted import build_vit_weighted

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────── 画像前処理 ────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# ViT は 224x224 を前提にするが、より大きいサイズの方が精度が上がることもある
VIT_TRANSFORM = transforms.Compose([
    transforms.Resize(248),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


def compute_metrics(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    バッチのメトリクスを計算する。
    各カテゴリの macro-F1 と全体平均を返す。
    """
    from sklearn.metrics import f1_score

    metrics: dict[str, float] = {}
    f1_scores = []

    for cat in ALL_LABEL_SETS:
        pred = (torch.sigmoid(logits[cat]) > 0.5).cpu().numpy()
        true = labels[cat].cpu().numpy()
        f1 = f1_score(true, pred, average="macro", zero_division=0)
        metrics[f"f1_{cat}"] = f1
        f1_scores.append(f1)

    metrics["f1_mean"] = float(np.mean(f1_scores))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pos_weight: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    model.train()
    if pos_weight is not None:
        criterion = {cat: nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
                     for cat, pw in pos_weight.items()}
    else:
        criterion = {cat: nn.BCEWithLogitsLoss() for cat in ALL_LABEL_SETS}
    total_loss = 0.0
    all_logits: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}
    all_labels: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}

    for images, label_dict in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        label_dict = {k: v.to(device) for k, v in label_dict.items()}

        optimizer.zero_grad()
        logits = model(images)

        # 4カテゴリの損失を合計
        loss = sum(criterion[cat](logits[cat], label_dict[cat]) for cat in ALL_LABEL_SETS)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        for cat in ALL_LABEL_SETS:
            all_logits[cat].append(logits[cat].detach().cpu())
            all_labels[cat].append(label_dict[cat].detach().cpu())

    # エポック全体のメトリクスを計算
    merged_logits = {cat: torch.cat(all_logits[cat]) for cat in ALL_LABEL_SETS}
    merged_labels = {cat: torch.cat(all_labels[cat]) for cat in ALL_LABEL_SETS}
    metrics = compute_metrics(merged_logits, merged_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    model.eval()
    criterion = {cat: nn.BCEWithLogitsLoss() for cat in ALL_LABEL_SETS}
    total_loss = 0.0
    all_logits: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}
    all_labels: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}

    for images, label_dict in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        label_dict_dev = {k: v.to(device) for k, v in label_dict.items()}
        logits = model(images)
        loss = sum(criterion[cat](logits[cat], label_dict_dev[cat]) for cat in ALL_LABEL_SETS)
        total_loss += loss.item() * images.size(0)

        for cat in ALL_LABEL_SETS:
            all_logits[cat].append(logits[cat].cpu())
            all_labels[cat].append(label_dict[cat].cpu())

    merged_logits = {cat: torch.cat(all_logits[cat]) for cat in ALL_LABEL_SETS}
    merged_labels = {cat: torch.cat(all_labels[cat]) for cat in ALL_LABEL_SETS}
    metrics = compute_metrics(merged_logits, merged_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train(
    model_name: str,
    mode: str,
    train_csv: str,
    val_csv: str,
    out_dir: str,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    weight_decay: float,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── モデル構築 ──
    if model_name == "resnet50":
        model = build_resnet50(mode=mode)
        train_tf = TRAIN_TRANSFORM
        val_tf   = VAL_TRANSFORM
    elif model_name == "resnet50_weighted":
        model = build_resnet50_weighted(mode=mode)
        train_tf = TRAIN_TRANSFORM
        val_tf   = VAL_TRANSFORM
    elif model_name == "vit":
        model = build_vit(mode=mode)
        train_tf = TRAIN_TRANSFORM
        val_tf   = VIT_TRANSFORM
    elif model_name == "vit_weighted":
        model = build_vit_weighted(mode=mode)
        train_tf = TRAIN_TRANSFORM
        val_tf   = VIT_TRANSFORM
    else:
        raise ValueError(f"model は 'resnet50' / 'resnet50_weighted' / 'vit' / 'vit_weighted' を指定してください: {model_name}")

    model = model.to(DEVICE)

    # ── pos_weight の計算（weighted モデルのみ）──
    pos_weight = None
    if model_name in ("resnet50_weighted", "vit_weighted"):
        pos_weight = compute_pos_weight(train_csv, device=DEVICE)
        print(f"pos_weight を計算しました（学習CSV: {train_csv}）")

    # ── データセット ──
    train_ds = BridgeInspectionDataset(train_csv, transform=train_tf, filter_valid=True)
    val_ds   = BridgeInspectionDataset(val_csv,   transform=val_tf,   filter_valid=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"学習: {len(train_ds)} 件 / 検証: {len(val_ds)} 件")

    # ── オプティマイザ ──
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── 学習ループ ──
    history = []
    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, DEVICE, pos_weight)
        val_metrics   = evaluate(model, val_loader, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_f1":   round(train_metrics["f1_mean"], 4),
            "val_loss":   round(val_metrics["loss"], 4),
            "val_f1":     round(val_metrics["f1_mean"], 4),
            **{f"val_{k}": round(v, 4) for k, v in val_metrics.items() if k.startswith("f1_")},
            "elapsed_sec": round(elapsed, 1),
        }
        history.append(log)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={log['train_loss']:.4f}  train_f1={log['train_f1']:.4f}  "
            f"val_loss={log['val_loss']:.4f}  val_f1={log['val_f1']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # ベストモデルを保存
        if val_metrics["f1_mean"] > best_val_f1:
            best_val_f1 = val_metrics["f1_mean"]
            torch.save(model.state_dict(), out_path / "best_model.pt")
            print(f"  -> ベストモデル更新 (val_f1={best_val_f1:.4f})")

    # 最終モデルを保存
    torch.save(model.state_dict(), out_path / "last_model.pt")

    # 学習履歴を保存
    with open(out_path / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"\n学習完了。出力先: {out_path}")
    print(f"最終 val_f1: {val_metrics['f1_mean']:.4f}  (ベスト: {best_val_f1:.4f})")


@torch.no_grad()
def predict_and_save(
    model_name: str,
    mode: str,
    ckpt_path: str,
    val_csv: str,
    out_csv: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> None:
    """
    学習済みチェックポイントで検証セットを推論し、予測CSVを保存する。
    evaluate.py の compare_models() でそのまま読み込める形式で出力する。
    """
    import pandas as pd

    if model_name == "resnet50":
        model = build_resnet50(mode=mode)
        val_tf = VAL_TRANSFORM
    elif model_name == "resnet50_weighted":
        model = build_resnet50_weighted(mode=mode)
        val_tf = VAL_TRANSFORM
    elif model_name == "vit":
        model = build_vit(mode=mode)
        val_tf = VIT_TRANSFORM
    elif model_name == "vit_weighted":
        model = build_vit_weighted(mode=mode)
        val_tf = VIT_TRANSFORM
    else:
        raise ValueError(f"未対応のモデル: {model_name}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    print(f"チェックポイントを読み込みました: {ckpt_path}")

    val_ds = BridgeInspectionDataset(val_csv, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    source_df = pd.read_csv(val_csv)
    pred_rows: list[dict] = []

    for images, _ in tqdm(val_loader, desc="推論"):
        images = images.to(DEVICE)
        logits = model(images)
        for cat in ALL_LABEL_SETS:
            probs = torch.sigmoid(logits[cat]).cpu().numpy()
            labels = ALL_LABEL_SETS[cat]
            if cat == "kenzenudo" or cat == "taisaku":
                # 単一ラベル
                pass
            # バッチ分を一時保存
        # ── バッチ単位でなく全件まとめて処理する方がシンプルなので以下で実装 ──
        break

    # 全件を一気に推論する実装
    all_logits: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}
    for images, _ in tqdm(DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers),
                          desc="推論"):
        images = images.to(DEVICE)
        logits = model(images)
        for cat in ALL_LABEL_SETS:
            all_logits[cat].append(torch.sigmoid(logits[cat]).cpu())

    merged = {cat: torch.cat(all_logits[cat]).numpy() for cat in ALL_LABEL_SETS}
    label_sets = ALL_LABEL_SETS

    rows = []
    for i in range(len(val_ds)):
        row: dict = {"image": source_df.iloc[i]["image"]}
        for cat, labels in label_sets.items():
            probs = merged[cat][i]
            if cat in ("kenzenudo", "taisaku"):
                best = labels[int(np.argmax(probs))]
                row[f"pred_{cat}"] = best
            else:
                selected = [labels[j] for j, p in enumerate(probs) if p >= 0.5]
                if not selected:
                    selected = [labels[int(np.argmax(probs))]]
                row[f"pred_{cat}"] = "|".join(selected)
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    result_df = source_df.merge(pred_df, on="image", how="left")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"予測CSV保存: {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="ResNet50/ViT マルチラベル分類学習")
    parser.add_argument("--model",       choices=["resnet50", "resnet50_weighted", "vit", "vit_weighted"], required=True)
    parser.add_argument("--mode",        choices=["linear_probe", "finetune"], required=True)
    parser.add_argument("--train_csv",   default=None, help="学習CSVパス（--predict 時は不要）")
    parser.add_argument("--val_csv",     required=True)
    parser.add_argument("--out_dir",     default=None, help="学習出力先（--predict 時は不要）")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    # 予測モード
    parser.add_argument("--predict",  action="store_true", help="推論のみ実行（学習スキップ）")
    parser.add_argument("--ckpt",     default=None,  help="--predict 時のチェックポイントパス")
    parser.add_argument("--out_csv",  default=None,  help="--predict 時の予測CSV出力先")
    args = parser.parse_args()

    if args.predict:
        assert args.ckpt,    "--predict には --ckpt が必要です"
        assert args.out_csv, "--predict には --out_csv が必要です"
        predict_and_save(
            model_name=args.model,
            mode=args.mode,
            ckpt_path=args.ckpt,
            val_csv=args.val_csv,
            out_csv=args.out_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        assert args.train_csv, "学習には --train_csv が必要です"
        assert args.out_dir,   "学習には --out_dir が必要です"
        train(
            model_name=args.model,
            mode=args.mode,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            out_dir=args.out_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            weight_decay=args.weight_decay,
        )


if __name__ == "__main__":
    main()
