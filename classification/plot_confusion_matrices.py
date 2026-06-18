# -*- coding: utf-8 -*-
"""
健全度判定・対策区分の混同行列を画像として保存するスクリプト。

使い方:
    python -m classification.plot_confusion_matrices \
        --out_dir classification/results/confusion_matrices
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np

# 日本語フォントを登録・設定
_JP_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_JP_FONT_PATH)
_jp_prop = fm.FontProperties(fname=_JP_FONT_PATH)
matplotlib.rcParams["font.family"] = _jp_prop.get_name()
matplotlib.rcParams["axes.unicode_minus"] = False
import pandas as pd
from sklearn.metrics import confusion_matrix

from classification.data.label_definitions import KENZENUDO_LABELS, TAISAKU_LABELS

# ── モデル定義（pred CSV パス、表示名）──
MODELS = [
    {
        "name": "CLIP fine-tuned kNN\n(epoch 5, k=10)",
        "csv":  "classification/results/clip_finetuned_knn_preds.csv",
        "key":  "clip_finetuned_knn",
    },
    {
        "name": "CLIP fine-tuned\n+ linear classifier",
        "csv":  "classification/results/clip_ft_linear_probe_preds.csv",
        "key":  "clip_ft_linear_probe",
    },
    {
        "name": "CLIP base kNN\n(k=10)",
        "csv":  "classification/results/clip_base_knn_preds.csv",
        "key":  "clip_base_knn",
    },
    {
        "name": "ResNet50\nfinetune",
        "csv":  "classification/results/resnet50_finetune_preds.csv",
        "key":  "resnet50_finetune",
    },
    {
        "name": "ResNet50\nweighted finetune",
        "csv":  "classification/results/resnet50_weighted_finetune_preds.csv",
        "key":  "resnet50_weighted_finetune",
    },
    {
        "name": "ViT\nfinetune",
        "csv":  "classification/results/vit_finetune_preds.csv",
        "key":  "vit_finetune",
    },
    {
        "name": "ViT\nweighted finetune",
        "csv":  "classification/results/vit_weighted_finetune_preds.csv",
        "key":  "vit_weighted_finetune",
    },
    {
        "name": "GPT-4o\n(zero-shot)",
        "csv":  "classification/results/LLM/gpt4o_preds.csv",
        "key":  "gpt4o",
    },
    {
        "name": "Qwen3-VL-8B\n(zero-shot)",
        "csv":  "classification/results/LLM/qwen3vl_preds.csv",
        "key":  "qwen3vl",
    },
    {
        "name": "InternVL3-8B\n(zero-shot)",
        "csv":  "classification/results/LLM/internvl_preds.csv",
        "key":  "internvl",
    },
    {
        "name": "Llama3.2-Vision\n(zero-shot)",
        "csv":  "classification/results/LLM/llama_preds.csv",
        "key":  "llama",
    },
]

KENZENUDO_ORDER   = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ"]
TAISAKU_ORDER     = ["A", "B", "C1", "C2", "E1", "E2", "M", "S1", "S2"]
DAMAGE_TYPE_ORDER = [
    "ひびわれ", "剥離・鉄筋露出", "腐食", "うき", "漏水・遊離石灰",
    "変形・欠損", "土砂詰まり", "ゆるみ・脱落", "防食機能の劣化",
    "支承の機能障害", "破断", "き裂", "舗装の異常", "洗掘", "その他",
]
DAMAGE_LOC_ORDER = [
    "主桁", "横桁", "床版", "竪壁", "翼壁", "胸壁", "橋脚", "支承",
    "伸縮装置", "舗装", "排水施設", "高欄", "防護柵", "地覆", "添架物",
    "頂版", "底版", "側壁", "落橋防止システム", "その他",
]


def load_multilabel(csv_path: str, cat: str, valid_labels: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """マルチラベルCSVを読み込み、正規ラベル外を除外してリストのリストを返す。"""
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
    # どちらか一方が空の行は除外
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
    """
    正解ラベルごとに予測ラベルの出現回数を数える共起行列を作成する。
    行: 正解ラベル, 列: 予測ラベル
    各サンプルについて、正解ラベル1つにつき予測ラベルをそれぞれカウント。
    """
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
    """CSVを読み込み、valid フィルタおよび正規ラベル外の予測を除外して true/pred リストを返す。

    予測が正規ラベル外（例：Llama の空文字 "JSON出力失敗" 等）の行は **除外** する。
    結果として全モデルで confusion matrix の列数が同じ（valid_labels の長さ）になる。
    除外件数は呼び出し元で必要なら別途確認すること。
    """
    df = pd.read_csv(csv_path)
    valid_col = f"{cat}_valid"
    if valid_col in df.columns:
        df = df[df[valid_col].astype(str).str.lower() == "true"]
    df = df.dropna(subset=[cat, f"pred_{cat}"])
    df[cat]           = df[cat].astype(str).str.strip()
    df[f"pred_{cat}"] = df[f"pred_{cat}"].astype(str).str.strip()
    # 正解ラベルが正規ラベル内の行のみ（ground truth の表記ゆれ対策）
    valid_set = set(valid_labels)
    df = df[df[cat].isin(valid_set)]
    # 予測ラベルが正規ラベル外の行は除外（"(未定義)" を作らない）
    df = df[df[f"pred_{cat}"].isin(valid_set)]
    y_true = df[cat].tolist()
    y_pred = df[f"pred_{cat}"].tolist()
    return y_true, y_pred


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
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("予測ラベル", fontsize=11)
    ax.set_ylabel("正解ラベル", fontsize=11)
    ax.set_title(f"{title}\n(accuracy={accuracy:.4f})", fontsize=11, pad=8)

    # セル内に数値を描画
    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def run(out_dir: str):
    out_path = Path(out_dir)

    for model in MODELS:
        csv_path = model["csv"]
        if not Path(csv_path).exists():
            print(f"[スキップ] {csv_path} が見つかりません")
            continue

        model_key  = model["key"]
        model_name = model["name"]
        print(f"\n{model_name.replace(chr(10), ' ')} ...")

        # ── 健全度判定 ──
        try:
            y_true, y_pred = load_and_filter(csv_path, "kenzenudo", KENZENUDO_ORDER)
            # 全モデル共通：正規 4 ラベルを必ず全て表示（未予測クラスは 0 行/列で表示される）
            labels = list(KENZENUDO_ORDER)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            acc = np.diag(cm).sum() / cm.sum() if cm.sum() > 0 else 0.0
            plot_confusion_matrix(
                cm, labels,
                title=f"健全度判定 — {model_name.replace(chr(10), ' ')}",
                out_path=out_path / "kenzenudo" / f"{model_key}.png",
                accuracy=acc,
                figsize=(5, 4),
            )
        except Exception as e:
            print(f"  [健全度] エラー: {e}")

        # ── 対策区分 ──
        try:
            y_true, y_pred = load_and_filter(csv_path, "taisaku", TAISAKU_ORDER)
            # 全モデル共通：正規 9 ラベルを必ず全て表示
            labels = list(TAISAKU_ORDER)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            acc = np.diag(cm).sum() / cm.sum() if cm.sum() > 0 else 0.0
            # 対策区分はラベル数が多いので横長に
            fw = max(7, len(labels) * 0.9)
            plot_confusion_matrix(
                cm, labels,
                title=f"対策区分 — {model_name.replace(chr(10), ' ')}",
                out_path=out_path / "taisaku" / f"{model_key}.png",
                accuracy=acc,
                figsize=(fw, fw * 0.85),
            )
        except Exception as e:
            print(f"  [対策区分] エラー: {e}")

        # ── 損傷種類（マルチラベル共起行列）──
        try:
            y_true, y_pred = load_multilabel(csv_path, "damage_type", DAMAGE_TYPE_ORDER)
            if y_true:
                # 全モデル共通：正規 15 ラベルを必ず全て表示
                labels = list(DAMAGE_TYPE_ORDER)
                cm = build_multilabel_cooccurrence(y_true, y_pred, labels)
                # 対角成分の割合を accuracy 代わりに表示（各正解クラスのヒット率の平均）
                row_sum = cm.sum(axis=1).clip(min=1)
                acc = np.mean(np.diag(cm) / row_sum)
                fw = max(10, len(labels) * 0.85)
                plot_confusion_matrix(
                    cm, labels,
                    title=f"損傷種類（共起行列）— {model_name.replace(chr(10), ' ')}",
                    out_path=out_path / "damage_type" / f"{model_key}.png",
                    accuracy=acc,
                    figsize=(fw, fw * 0.9),
                    cmap="Blues",
                )
        except Exception as e:
            print(f"  [損傷種類] エラー: {e}")

        # ── 損傷部位（マルチラベル共起行列）──
        try:
            y_true, y_pred = load_multilabel(csv_path, "damage_loc", DAMAGE_LOC_ORDER)
            if y_true:
                # 全モデル共通：正規 20 ラベルを必ず全て表示
                labels = list(DAMAGE_LOC_ORDER)
                cm = build_multilabel_cooccurrence(y_true, y_pred, labels)
                row_sum = cm.sum(axis=1).clip(min=1)
                acc = np.mean(np.diag(cm) / row_sum)
                fw = max(12, len(labels) * 0.85)
                plot_confusion_matrix(
                    cm, labels,
                    title=f"損傷部位（共起行列）— {model_name.replace(chr(10), ' ')}",
                    out_path=out_path / "damage_loc" / f"{model_key}.png",
                    accuracy=acc,
                    figsize=(fw, fw * 0.9),
                    cmap="Blues",
                )
        except Exception as e:
            print(f"  [損傷部位] エラー: {e}")

    print(f"\n完了: {out_path}")
    print(f"  {out_path}/kenzenudo/   ← 健全度判定の混同行列")
    print(f"  {out_path}/taisaku/     ← 対策区分の混同行列")
    print(f"  {out_path}/damage_type/ ← 損傷種類の共起行列（マルチラベル）")
    print(f"  {out_path}/damage_loc/  ← 損傷部位の共起行列（マルチラベル）")


def main():
    parser = argparse.ArgumentParser(description="混同行列の画像保存")
    parser.add_argument("--out_dir", default="classification/results/confusion_matrices")
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
