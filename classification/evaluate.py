# -*- coding: utf-8 -*-
"""
分類結果の評価スクリプト

モデルが出力した予測CSV（pred_kenzenudo, pred_taisaku, etc.）と
正解ラベル（kenzenudo, taisaku, etc.）を比較し、メトリクスを計算する。

対応メトリクス:
  - Accuracy（健全度・対策区分の単一ラベル向け）
  - Macro-F1
  - Micro-F1
  - Weighted-F1
  - mAP（mean Average Precision）

使い方:
    # 単一モデルの評価
    python -m classification.evaluate \
        --pred classification/results/clip_zeroshot_preds.csv \
        --out  classification/results/clip_zeroshot_metrics.json

    # 全モデルの一括比較
    python -m classification.evaluate \
        --compare_dir classification/results \
        --out         classification/results/comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
)

from classification.data.label_definitions import ALL_LABEL_SETS

# ────────────────────────────── ユーティリティ ───────────────────────────────────

def _parse_labels(series: pd.Series, label_list: list[str]) -> np.ndarray:
    """
    "|" 区切りのラベル文字列をマルチホット行列 (N, C) に変換する。
    """
    label2idx = {lbl: i for i, lbl in enumerate(label_list)}
    matrix = np.zeros((len(series), len(label_list)), dtype=np.float32)
    for i, val in enumerate(series):
        for token in str(val or "").split("|"):
            token = token.strip()
            if token in label2idx:
                matrix[i, label2idx[token]] = 1.0
    return matrix


def evaluate_model(pred_csv: str) -> dict:
    """
    1モデルの予測CSVを読み込み、全カテゴリのメトリクスを計算して返す。

    入力CSVの期待カラム:
        kenzenudo, taisaku, damage_type, damage_loc   (正解)
        pred_kenzenudo, pred_taisaku, pred_damage_type, pred_damage_loc (予測)
    """
    df = pd.read_csv(pred_csv)
    results: dict[str, dict] = {}

    for cat, label_list in ALL_LABEL_SETS.items():
        gt_col   = cat
        pred_col = f"pred_{cat}"

        if gt_col not in df.columns or pred_col not in df.columns:
            print(f"  [スキップ] カラムが見つかりません: {gt_col} / {pred_col}")
            continue

        # 正解・予測をマルチホット行列に変換
        y_true = _parse_labels(df[gt_col],   label_list)
        y_pred = _parse_labels(df[pred_col], label_list)

        # 有効行のみ（正解が1件以上ある行）
        valid_mask = y_true.sum(axis=1) > 0
        if valid_mask.sum() == 0:
            print(f"  [スキップ] 有効な正解ラベルがありません: {cat}")
            continue

        y_true_v = y_true[valid_mask]
        y_pred_v = y_pred[valid_mask]

        # ── メトリクス計算 ──
        cat_metrics: dict[str, float] = {}

        # Accuracy（完全一致）
        exact_match = (y_true_v == y_pred_v).all(axis=1).mean()
        cat_metrics["exact_match_accuracy"] = float(exact_match)

        # F1（各閾値 0.5 のマルチホット）
        cat_metrics["macro_f1"]    = float(f1_score(y_true_v, y_pred_v, average="macro",    zero_division=0))
        cat_metrics["micro_f1"]    = float(f1_score(y_true_v, y_pred_v, average="micro",    zero_division=0))
        cat_metrics["weighted_f1"] = float(f1_score(y_true_v, y_pred_v, average="weighted", zero_division=0))

        # mAP（有効クラスのみ）
        valid_classes = y_true_v.sum(axis=0) > 0
        if valid_classes.sum() > 0:
            try:
                mAP = average_precision_score(
                    y_true_v[:, valid_classes],
                    y_pred_v[:, valid_classes],
                    average="macro",
                )
                cat_metrics["mAP"] = float(mAP)
            except Exception:
                cat_metrics["mAP"] = None

        # クラス別F1
        per_class_f1 = f1_score(y_true_v, y_pred_v, average=None, zero_division=0)
        cat_metrics["per_class_f1"] = {
            label_list[i]: round(float(per_class_f1[i]), 4)
            for i in range(len(label_list))
        }

        # サンプル数
        cat_metrics["n_samples"] = int(valid_mask.sum())

        results[cat] = cat_metrics

    # 全カテゴリの平均 macro-F1
    macro_f1_values = [v["macro_f1"] for v in results.values() if "macro_f1" in v]
    results["overall"] = {
        "mean_macro_f1": float(np.mean(macro_f1_values)) if macro_f1_values else 0.0,
    }

    return results


def compare_models(pred_dir: str) -> dict:
    """
    指定ディレクトリ内の *_preds.csv ファイルをすべて評価して比較する。
    """
    pred_files = sorted(Path(pred_dir).glob("*_preds.csv"))
    if not pred_files:
        print(f"*_preds.csv が見つかりません: {pred_dir}")
        return {}

    comparison: dict[str, dict] = {}
    for f in pred_files:
        model_name = f.stem.replace("_preds", "")
        print(f"\n評価中: {model_name}")
        try:
            metrics = evaluate_model(str(f))
            comparison[model_name] = metrics
            print(f"  全体 macro-F1: {metrics.get('overall', {}).get('mean_macro_f1', 0):.4f}")
        except Exception as e:
            print(f"  エラー: {e}")
            comparison[model_name] = {"error": str(e)}

    return comparison


def print_summary_table(comparison: dict) -> None:
    """比較結果を表形式で標準出力する。"""
    print("\n" + "=" * 80)
    print("モデル比較サマリー（macro-F1）")
    print("=" * 80)

    categories = list(ALL_LABEL_SETS.keys())
    header = f"{'モデル':30s}" + "".join(f"  {cat:12s}" for cat in categories) + "  平均"
    print(header)
    print("-" * len(header))

    for model_name, metrics in comparison.items():
        if "error" in metrics:
            print(f"  {model_name:28s}  ERROR: {metrics['error']}")
            continue
        row = f"{model_name:30s}"
        f1_vals = []
        for cat in categories:
            f1 = metrics.get(cat, {}).get("macro_f1", 0.0)
            row += f"  {f1:.4f}      "
            f1_vals.append(f1)
        mean_f1 = np.mean(f1_vals) if f1_vals else 0.0
        row += f"  {mean_f1:.4f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="分類結果の評価")
    parser.add_argument("--pred",        default=None, help="単一予測CSVのパス")
    parser.add_argument("--out",         default=None, help="メトリクスの出力JSONパス")
    parser.add_argument("--compare_dir", default=None, help="複数モデルを比較するディレクトリ")
    args = parser.parse_args()

    if args.compare_dir:
        results = compare_models(args.compare_dir)
        print_summary_table(results)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n結果保存: {args.out}")

    elif args.pred:
        results = evaluate_model(args.pred)

        print("\n── 評価結果 ──")
        for cat, metrics in results.items():
            if cat == "overall":
                continue
            print(f"\n{cat}:")
            for k, v in metrics.items():
                if k != "per_class_f1":
                    print(f"  {k}: {v}")
            print("  クラス別F1:")
            for cls, f1 in (metrics.get("per_class_f1") or {}).items():
                print(f"    {cls}: {f1:.4f}")

        print(f"\n全体 mean_macro_f1: {results.get('overall', {}).get('mean_macro_f1', 0):.4f}")

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"結果保存: {args.out}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
