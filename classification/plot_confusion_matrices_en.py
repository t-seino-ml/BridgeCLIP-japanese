# -*- coding: utf-8 -*-
"""
Save confusion matrices (English labels) for all methods.

Reads prediction CSVs in `classification/results_en/` and writes confusion
matrix images for the four categories.

Usage:
    python -m classification.plot_confusion_matrices_en \
        --out_dir classification/results_en/confusion_matrices
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from classification.data.label_definitions_en import (
    KENZENUDO_LABELS,
    TAISAKU_LABELS,
    DAMAGE_TYPE_LABELS,
    DAMAGE_LOC_LABELS,
)

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# English prediction CSVs expected in classification/results_en/
MODELS = [
    {"name": "CLIP fine-tuned kNN\n(epoch 5, k=10)", "csv": "classification/results_en/clip_ft_knn_en_preds.csv",        "key": "clip_ft_knn"},
    {"name": "CLIP base kNN\n(k=10)",                 "csv": "classification/results_en/clip_base_knn_en_preds.csv",      "key": "clip_base_knn"},
    {"name": "CLIP zero-shot\n(ensemble templates)",  "csv": "classification/results_en/clip_zeroshot_en_preds.csv",      "key": "clip_zeroshot"},
    {"name": "ResNet50\nfinetune",                    "csv": "classification/results_en/resnet50_finetune_en_preds.csv",  "key": "resnet50_finetune"},
    {"name": "ResNet50\nlinear probe",                "csv": "classification/results_en/resnet50_linear_probe_en_preds.csv", "key": "resnet50_linear_probe"},
    {"name": "ResNet50\nweighted finetune",           "csv": "classification/results_en/resnet50_weighted_finetune_en_preds.csv", "key": "resnet50_weighted_finetune"},
    {"name": "ViT\nfinetune",                         "csv": "classification/results_en/vit_finetune_en_preds.csv",       "key": "vit_finetune"},
    {"name": "ViT\nlinear probe",                     "csv": "classification/results_en/vit_linear_probe_en_preds.csv",   "key": "vit_linear_probe"},
    {"name": "ViT\nweighted finetune",                "csv": "classification/results_en/vit_weighted_finetune_en_preds.csv", "key": "vit_weighted_finetune"},
    {"name": "GPT-4o\n(zero-shot)",                   "csv": "classification/results_en/gpt4o_en_preds.csv",              "key": "gpt4o"},
    {"name": "Qwen3-VL-8B\n(zero-shot)",              "csv": "classification/results_en/qwen3vl_en_preds.csv",            "key": "qwen3vl"},
    {"name": "InternVL3-8B\n(zero-shot)",             "csv": "classification/results_en/internvl_en_preds.csv",           "key": "internvl"},
    {"name": "Llama3.2-Vision\n(zero-shot)",          "csv": "classification/results_en/llama_en_preds.csv",              "key": "llama"},
]

KENZENUDO_ORDER   = KENZENUDO_LABELS
TAISAKU_ORDER     = TAISAKU_LABELS
DAMAGE_TYPE_ORDER = DAMAGE_TYPE_LABELS
DAMAGE_LOC_ORDER  = DAMAGE_LOC_LABELS

UNDEFINED = "(undefined)"


def load_multilabel(csv_path: str, cat: str, valid_labels: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    df = pd.read_csv(csv_path)
    valid_col = f"{cat}_valid"
    if valid_col in df.columns:
        df = df[df[valid_col].astype(str).str.lower() == "true"]
    df = df.dropna(subset=[cat, f"pred_{cat}"])
    valid_set = set(valid_labels)

    def parse(s):
        return [t.strip() for t in str(s).split("|") if t.strip() in valid_set]

    y_true = df[cat].apply(parse).tolist()
    y_pred = df[f"pred_{cat}"].apply(parse).tolist()
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t and p]
    if not pairs:
        return [], []
    y_true, y_pred = zip(*pairs)
    return list(y_true), list(y_pred)


def build_multilabel_cooccurrence(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    labels: list[str],
) -> np.ndarray:
    label2idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for trues, preds in zip(y_true, y_pred):
        for t in trues:
            if t not in label2idx:
                continue
            ti = label2idx[t]
            for p in preds:
                if p in label2idx:
                    cm[ti, label2idx[p]] += 1
    return cm


def load_and_filter(csv_path: str, cat: str, valid_labels: list[str]) -> tuple[list, list]:
    df = pd.read_csv(csv_path)
    valid_col = f"{cat}_valid"
    if valid_col in df.columns:
        df = df[df[valid_col].astype(str).str.lower() == "true"]
    df = df.dropna(subset=[cat, f"pred_{cat}"])
    df[cat]           = df[cat].astype(str).str.strip()
    df[f"pred_{cat}"] = df[f"pred_{cat}"].astype(str).str.strip()
    df = df[df[cat].isin(valid_labels)]
    valid_set = set(valid_labels)
    df[f"pred_{cat}"] = df[f"pred_{cat}"].apply(
        lambda x: x if x in valid_set else UNDEFINED
    )
    return df[cat].tolist(), df[f"pred_{cat}"].tolist()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path,
    accuracy: float,
    figsize: tuple = (6, 5),
    cmap: str = "Blues",
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground truth", fontsize=11)
    ax.set_title(f"{title}\n(accuracy={accuracy:.4f})", fontsize=11, pad=8)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(cm[i, j])
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=8, color=color,
                    fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def run(out_dir: str):
    out_path = Path(out_dir)

    for model in MODELS:
        csv_path = model["csv"]
        if not Path(csv_path).exists():
            print(f"[skip] {csv_path} not found")
            continue

        model_key  = model["key"]
        model_name = model["name"]
        print(f"\n{model_name.replace(chr(10), ' ')} ...")

        # Soundness rating (single-label)
        try:
            y_true, y_pred = load_and_filter(csv_path, "kenzenudo", KENZENUDO_ORDER)
            present = set(y_true + y_pred)
            labels = [l for l in KENZENUDO_ORDER if l in present]
            if UNDEFINED in present:
                labels.append(UNDEFINED)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            acc = np.diag(cm).sum() / cm.sum() if cm.sum() > 0 else 0.0
            plot_confusion_matrix(
                cm, labels,
                title=f"Soundness rating — {model_name.replace(chr(10), ' ')}",
                out_path=out_path / "kenzenudo" / f"{model_key}.png",
                accuracy=acc,
                figsize=(5, 4),
            )
        except Exception as e:
            print(f"  [soundness] error: {e}")

        # Measure classification (single-label)
        try:
            y_true, y_pred = load_and_filter(csv_path, "taisaku", TAISAKU_ORDER)
            present = set(y_true + y_pred)
            labels = [l for l in TAISAKU_ORDER if l in present]
            if UNDEFINED in present:
                labels.append(UNDEFINED)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            acc = np.diag(cm).sum() / cm.sum() if cm.sum() > 0 else 0.0
            fw = max(7, len(labels) * 0.9)
            plot_confusion_matrix(
                cm, labels,
                title=f"Measure classification — {model_name.replace(chr(10), ' ')}",
                out_path=out_path / "taisaku" / f"{model_key}.png",
                accuracy=acc,
                figsize=(fw, fw * 0.85),
            )
        except Exception as e:
            print(f"  [measure] error: {e}")

        # Damage type (multi-label co-occurrence)
        try:
            y_true, y_pred = load_multilabel(csv_path, "damage_type", DAMAGE_TYPE_ORDER)
            if y_true:
                present = set(l for ls in y_true + y_pred for l in ls)
                labels = [l for l in DAMAGE_TYPE_ORDER if l in present]
                cm = build_multilabel_cooccurrence(y_true, y_pred, labels)
                row_sum = cm.sum(axis=1).clip(min=1)
                acc = np.mean(np.diag(cm) / row_sum)
                fw = max(10, len(labels) * 0.95)
                plot_confusion_matrix(
                    cm, labels,
                    title=f"Damage type (co-occurrence) — {model_name.replace(chr(10), ' ')}",
                    out_path=out_path / "damage_type" / f"{model_key}.png",
                    accuracy=acc,
                    figsize=(fw, fw * 0.9),
                    cmap="YlOrRd",
                )
        except Exception as e:
            print(f"  [damage_type] error: {e}")

        # Damage location (multi-label co-occurrence)
        try:
            y_true, y_pred = load_multilabel(csv_path, "damage_loc", DAMAGE_LOC_ORDER)
            if y_true:
                present = set(l for ls in y_true + y_pred for l in ls)
                labels = [l for l in DAMAGE_LOC_ORDER if l in present]
                cm = build_multilabel_cooccurrence(y_true, y_pred, labels)
                row_sum = cm.sum(axis=1).clip(min=1)
                acc = np.mean(np.diag(cm) / row_sum)
                fw = max(12, len(labels) * 0.95)
                plot_confusion_matrix(
                    cm, labels,
                    title=f"Damage location (co-occurrence) — {model_name.replace(chr(10), ' ')}",
                    out_path=out_path / "damage_loc" / f"{model_key}.png",
                    accuracy=acc,
                    figsize=(fw, fw * 0.9),
                    cmap="YlOrRd",
                )
        except Exception as e:
            print(f"  [damage_loc] error: {e}")

    print(f"\ndone: {out_path}")
    print(f"  {out_path}/kenzenudo/   ← soundness rating CMs")
    print(f"  {out_path}/taisaku/     ← measure classification CMs")
    print(f"  {out_path}/damage_type/ ← damage type co-occurrence matrices")
    print(f"  {out_path}/damage_loc/  ← damage location co-occurrence matrices")


def main():
    parser = argparse.ArgumentParser(description="Save confusion matrix images (English)")
    parser.add_argument("--out_dir", default="classification/results_en/confusion_matrices")
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
