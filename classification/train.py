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
from classification.models.clip_classifier import build_clip_base, build_clip_ft, build_clip_ft_weighted

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────── 画像前処理 ────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

# ResNet50 用（ImageNet 標準）
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

# ViT-B/16 用（torchvision の ViT_B_16_Weights.IMAGENET1K_V1 推奨設定: Resize(256)+CC(224)、ImageNet 統計）
# 学習側は augmentation を載せる
VIT_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

VIT_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# ── 単一ラベル系（CE）/ マルチラベル系（BCE）の分類 ──
SINGLE_LABEL_CATS = ("kenzenudo", "taisaku")
MULTI_LABEL_CATS  = ("damage_type", "damage_loc")


def compute_metrics(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    バッチのメトリクスを計算する。
    単一ラベル系 (kenzenudo/taisaku) は softmax + argmax、
    マルチラベル系 (damage_type/damage_loc) は sigmoid > 0.5 で判定。
    各カテゴリの macro-F1 と全体平均を返す。
    """
    from sklearn.metrics import f1_score

    metrics: dict[str, float] = {}
    f1_scores = []

    for cat in ALL_LABEL_SETS:
        true_mat = labels[cat].cpu().numpy()  # (B, K) マルチホット
        if cat in SINGLE_LABEL_CATS:
            # 単一ラベル: argmax を予測クラスとし、マルチホット相当に展開
            pred_idx = logits[cat].argmax(dim=1).cpu().numpy()  # (B,)
            pred = np.zeros_like(true_mat)
            pred[np.arange(len(pred_idx)), pred_idx] = 1.0
        else:
            pred = (torch.sigmoid(logits[cat]) > 0.5).cpu().numpy()
        f1 = f1_score(true_mat, pred, average="macro", zero_division=0)
        metrics[f"f1_{cat}"] = f1
        f1_scores.append(f1)

    metrics["f1_mean"] = float(np.mean(f1_scores))
    return metrics


def _build_criteria(
    device: str,
    pos_weight: dict[str, torch.Tensor] | None,
) -> dict[str, nn.Module]:
    """単一ラベル系は CE、マルチラベル系は BCE を返す。pos_weight はマルチラベル側のみ適用。"""
    criteria: dict[str, nn.Module] = {}
    for cat in ALL_LABEL_SETS:
        if cat in SINGLE_LABEL_CATS:
            criteria[cat] = nn.CrossEntropyLoss()
        else:
            pw = pos_weight.get(cat) if pos_weight else None
            criteria[cat] = (
                nn.BCEWithLogitsLoss(pos_weight=pw.to(device)) if pw is not None
                else nn.BCEWithLogitsLoss()
            )
    return criteria


def _compute_loss(
    criteria: dict[str, nn.Module],
    logits: dict[str, torch.Tensor],
    label_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """4カテゴリの損失を合計（単一ラベル系は argmax をクラスindex化して CE）。"""
    total = 0.0
    for cat in ALL_LABEL_SETS:
        if cat in SINGLE_LABEL_CATS:
            target_idx = label_dict[cat].argmax(dim=1)
            total = total + criteria[cat](logits[cat], target_idx)
        else:
            total = total + criteria[cat](logits[cat], label_dict[cat])
    return total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    pos_weight: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    model.train()
    criterion = _build_criteria(device, pos_weight)
    total_loss = 0.0
    all_logits: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}
    all_labels: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}

    for images, label_dict in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        label_dict = {k: v.to(device) for k, v in label_dict.items()}

        optimizer.zero_grad()
        logits = model(images)

        loss = _compute_loss(criterion, logits, label_dict)
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
    criterion = _build_criteria(device, pos_weight=None)
    total_loss = 0.0
    all_logits: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}
    all_labels: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}

    for images, label_dict in tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device)
        label_dict_dev = {k: v.to(device) for k, v in label_dict.items()}
        logits = model(images)
        loss = _compute_loss(criterion, logits, label_dict_dev)
        total_loss += loss.item() * images.size(0)

        for cat in ALL_LABEL_SETS:
            all_logits[cat].append(logits[cat].cpu())
            all_labels[cat].append(label_dict[cat].cpu())

    merged_logits = {cat: torch.cat(all_logits[cat]) for cat in ALL_LABEL_SETS}
    merged_labels = {cat: torch.cat(all_labels[cat]) for cat in ALL_LABEL_SETS}
    metrics = compute_metrics(merged_logits, merged_labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def _build_clip_transforms(model):
    """open_clip の visual.image_size に合わせた transforms を返す。"""
    import torchvision.transforms as T
    img_size = getattr(model.visual, "image_size", 224)
    if isinstance(img_size, (tuple, list)):
        img_size = img_size[0]
    # open_clip の標準正規化
    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD  = (0.26862954, 0.26130258, 0.27577711)
    train_tf = T.Compose([
        T.Resize(int(img_size * 1.143), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    val_tf = T.Compose([
        T.Resize(int(img_size * 1.143), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


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
    image_root: str | None = None,
    pos_weight_clip: float = 10.0,
    clip_ckpt_dir: str | None = None,
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
        train_tf = VIT_TRAIN_TRANSFORM
        val_tf   = VIT_TRANSFORM
    elif model_name == "vit_weighted":
        model = build_vit_weighted(mode=mode)
        train_tf = VIT_TRAIN_TRANSFORM
        val_tf   = VIT_TRANSFORM
    elif model_name == "clip_base":
        model = build_clip_base(mode=mode)
        train_tf, val_tf = _build_clip_transforms(model)
    elif model_name == "clip_ft":
        model = build_clip_ft(mode=mode, ckpt_dir=clip_ckpt_dir)
        train_tf, val_tf = _build_clip_transforms(model)
    elif model_name == "clip_ft_weighted":
        model = build_clip_ft_weighted(mode=mode, ckpt_dir=clip_ckpt_dir)
        train_tf, val_tf = _build_clip_transforms(model)
    else:
        raise ValueError(f"未対応の model: {model_name}")

    model = model.to(DEVICE)

    # ── pos_weight の計算（weighted モデルのみ・上限クリップ）──
    pos_weight = None
    if model_name in ("resnet50_weighted", "vit_weighted", "clip_ft_weighted"):
        raw_pw = compute_pos_weight(train_csv, device=DEVICE)
        # マルチラベル系のみ pos_weight を使う（単一ラベル系は CE なので無視される）
        pos_weight = {}
        for cat in MULTI_LABEL_CATS:
            if cat in raw_pw:
                pw = raw_pw[cat].clamp(max=pos_weight_clip)
                pos_weight[cat] = pw
                print(f"  pos_weight[{cat}] (clipped<= {pos_weight_clip}): "
                      f"min={pw.min().item():.2f} max={pw.max().item():.2f} mean={pw.mean().item():.2f}")
        print(f"pos_weight を計算しました（学習CSV: {train_csv}）")

    # ── データセット ──
    train_ds = BridgeInspectionDataset(
        train_csv, transform=train_tf, filter_valid=True,
        image_root=image_root, strict_paths=True,
    )
    val_ds   = BridgeInspectionDataset(
        val_csv,   transform=val_tf,   filter_valid=True,
        image_root=image_root, strict_paths=True,
    )

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
    image_root: str | None = None,
    clip_ckpt_dir: str | None = None,
) -> None:
    """
    学習済みチェックポイントで検証セットを推論し、予測CSVを保存する。
    evaluate.py の compare_models() でそのまま読み込める形式で出力する。
    単一ラベル系 (kenzenudo/taisaku) は softmax + argmax、
    マルチラベル系 (damage_type/damage_loc) は sigmoid + 閾値0.5（無検出なら argmax 補完）。
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
    elif model_name == "clip_base":
        model = build_clip_base(mode=mode)
        _, val_tf = _build_clip_transforms(model)
    elif model_name == "clip_ft":
        model = build_clip_ft(mode=mode, ckpt_dir=clip_ckpt_dir)
        _, val_tf = _build_clip_transforms(model)
    elif model_name == "clip_ft_weighted":
        model = build_clip_ft_weighted(mode=mode, ckpt_dir=clip_ckpt_dir)
        _, val_tf = _build_clip_transforms(model)
    else:
        raise ValueError(f"未対応のモデル: {model_name}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    print(f"チェックポイントを読み込みました: {ckpt_path}")

    val_ds = BridgeInspectionDataset(
        val_csv, transform=val_tf,
        image_root=image_root, strict_paths=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)

    source_df = pd.read_csv(val_csv)

    # 全件まとめて推論
    all_logits: dict[str, list] = {cat: [] for cat in ALL_LABEL_SETS}
    for images, _ in tqdm(val_loader, desc="推論"):
        images = images.to(DEVICE)
        logits = model(images)
        for cat in ALL_LABEL_SETS:
            all_logits[cat].append(logits[cat].cpu())

    merged_logits = {cat: torch.cat(all_logits[cat]).numpy() for cat in ALL_LABEL_SETS}

    rows = []
    for i in range(len(val_ds)):
        row: dict = {"image": source_df.iloc[i]["image"]}
        for cat, labels in ALL_LABEL_SETS.items():
            lg = merged_logits[cat][i]
            if cat in SINGLE_LABEL_CATS:
                # softmax 不要・logit の argmax で十分
                row[f"pred_{cat}"] = labels[int(np.argmax(lg))]
            else:
                probs = 1.0 / (1.0 + np.exp(-lg))
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
    parser = argparse.ArgumentParser(description="ResNet50/ViT/CLIP マルチラベル分類学習")
    parser.add_argument("--model",       choices=[
        "resnet50", "resnet50_weighted", "vit", "vit_weighted",
        "clip_base", "clip_ft", "clip_ft_weighted",
    ], required=True)
    parser.add_argument("--mode",        choices=["linear_probe", "finetune"], required=True)
    parser.add_argument("--train_csv",   default=None, help="学習CSVパス（--predict 時は不要）")
    parser.add_argument("--val_csv",     required=True)
    parser.add_argument("--out_dir",     default=None, help="学習出力先（--predict 時は不要）")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--image_root",  default=None, help="CSV の image 列のディレクトリ部分を全てこの値に差し替える（ホスト間のパス差を吸収）")
    parser.add_argument("--pos_weight_clip", type=float, default=10.0, help="weighted 系の pos_weight 上限クリップ値")
    parser.add_argument("--clip_ckpt_dir", default=None, help="clip_ft / clip_ft_weighted 時に best CLIP ckpt を含む dir（results.jsonl 必須）")
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
            image_root=args.image_root,
            clip_ckpt_dir=args.clip_ckpt_dir,
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
            image_root=args.image_root,
            pos_weight_clip=args.pos_weight_clip,
            clip_ckpt_dir=args.clip_ckpt_dir,
        )


if __name__ == "__main__":
    main()
