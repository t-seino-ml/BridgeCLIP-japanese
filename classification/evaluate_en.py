# -*- coding: utf-8 -*-
"""
Evaluation script for English-labeled classification predictions.

Parallel to evaluate.py. Uses ALL_LABEL_SETS from label_definitions_en,
so canonical labels are the English ones (I/II/III/IV, main girder, etc.).

Usage:
    # single model
    python -m classification.evaluate_en \
        --pred classification/results/gpt4o_en_preds.csv \
        --out  classification/results/gpt4o_en_metrics.json

    # compare all *_preds.csv in a dir
    python -m classification.evaluate_en \
        --compare_dir classification/results \
        --out         classification/results/comparison_en.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from classification.data.label_definitions_en import ALL_LABEL_SETS


def _parse_labels(series: pd.Series, label_list: list[str]) -> np.ndarray:
    label2idx = {lbl: i for i, lbl in enumerate(label_list)}
    matrix = np.zeros((len(series), len(label_list)), dtype=np.float32)
    for i, val in enumerate(series):
        for token in str(val or "").split("|"):
            token = token.strip()
            if token in label2idx:
                matrix[i, label2idx[token]] = 1.0
    return matrix


def evaluate_model(pred_csv: str) -> dict:
    df = pd.read_csv(pred_csv)
    results: dict[str, dict] = {}

    for cat, label_list in ALL_LABEL_SETS.items():
        gt_col   = cat
        pred_col = f"pred_{cat}"
        if gt_col not in df.columns or pred_col not in df.columns:
            print(f"  [skip] column not found: {gt_col} / {pred_col}")
            continue

        y_true = _parse_labels(df[gt_col],   label_list)
        y_pred = _parse_labels(df[pred_col], label_list)

        valid_mask = y_true.sum(axis=1) > 0
        if valid_mask.sum() == 0:
            print(f"  [skip] no valid GT labels for: {cat}")
            continue

        y_true_v = y_true[valid_mask]
        y_pred_v = y_pred[valid_mask]

        cat_metrics: dict = {}
        exact_match = (y_true_v == y_pred_v).all(axis=1).mean()
        cat_metrics["exact_match_accuracy"] = float(exact_match)
        cat_metrics["macro_f1"]    = float(f1_score(y_true_v, y_pred_v, average="macro",    zero_division=0))
        cat_metrics["micro_f1"]    = float(f1_score(y_true_v, y_pred_v, average="micro",    zero_division=0))
        cat_metrics["weighted_f1"] = float(f1_score(y_true_v, y_pred_v, average="weighted", zero_division=0))

        cat_metrics["macro_precision"]    = float(precision_score(y_true_v, y_pred_v, average="macro",    zero_division=0))
        cat_metrics["macro_recall"]       = float(recall_score(   y_true_v, y_pred_v, average="macro",    zero_division=0))
        cat_metrics["weighted_precision"] = float(precision_score(y_true_v, y_pred_v, average="weighted", zero_division=0))
        cat_metrics["weighted_recall"]    = float(recall_score(   y_true_v, y_pred_v, average="weighted", zero_division=0))

        if cat in ("kenzenudo", "taisaku"):
            y_true_idx = y_true_v.argmax(axis=1)
            y_pred_active = y_pred_v.sum(axis=1) > 0
            y_pred_idx = np.where(y_pred_active, y_pred_v.argmax(axis=1), -1)
            try:
                cat_metrics["balanced_accuracy"] = float(
                    balanced_accuracy_score(y_true_idx, y_pred_idx)
                )
            except Exception:
                cat_metrics["balanced_accuracy"] = None

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

        per_class_p  = precision_score(y_true_v, y_pred_v, average=None, zero_division=0)
        per_class_r  = recall_score(   y_true_v, y_pred_v, average=None, zero_division=0)
        per_class_f1 = f1_score(       y_true_v, y_pred_v, average=None, zero_division=0)
        per_class_support = y_true_v.sum(axis=0).astype(int)
        cat_metrics["per_class"] = {
            label_list[i]: {
                "precision": round(float(per_class_p[i]),  4),
                "recall":    round(float(per_class_r[i]),  4),
                "f1":        round(float(per_class_f1[i]), 4),
                "support":   int(per_class_support[i]),
            }
            for i in range(len(label_list))
        }
        cat_metrics["per_class_f1"] = {
            label_list[i]: round(float(per_class_f1[i]), 4)
            for i in range(len(label_list))
        }
        cat_metrics["n_samples"] = int(valid_mask.sum())
        results[cat] = cat_metrics

    macro_f1_values = [v["macro_f1"] for v in results.values() if "macro_f1" in v]
    results["overall"] = {
        "mean_macro_f1": float(np.mean(macro_f1_values)) if macro_f1_values else 0.0,
    }
    return results


def compare_models(pred_dir: str) -> dict:
    pred_files = sorted(Path(pred_dir).glob("*_preds.csv"))
    if not pred_files:
        print(f"no *_preds.csv found in: {pred_dir}")
        return {}

    comparison: dict[str, dict] = {}
    for f in pred_files:
        model_name = f.stem.replace("_preds", "")
        print(f"\nevaluating: {model_name}")
        try:
            metrics = evaluate_model(str(f))
            comparison[model_name] = metrics
            print(f"  mean macro-F1: {metrics.get('overall', {}).get('mean_macro_f1', 0):.4f}")
        except Exception as e:
            print(f"  error: {e}")
            comparison[model_name] = {"error": str(e)}
    return comparison


def print_summary_table(comparison: dict) -> None:
    print("\n" + "=" * 80)
    print("Model comparison (macro-F1, English labels)")
    print("=" * 80)

    categories = list(ALL_LABEL_SETS.keys())
    header = f"{'model':30s}" + "".join(f"  {cat:12s}" for cat in categories) + "  mean"
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
    parser = argparse.ArgumentParser(description="Evaluate classification predictions (English)")
    parser.add_argument("--pred",        default=None, help="single prediction CSV path")
    parser.add_argument("--out",         default=None, help="output metrics JSON path")
    parser.add_argument("--compare_dir", default=None, help="compare multiple models in this dir")
    args = parser.parse_args()

    if args.compare_dir:
        results = compare_models(args.compare_dir)
        print_summary_table(results)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nsaved: {args.out}")

    elif args.pred:
        results = evaluate_model(args.pred)
        print("\n── evaluation ──")
        for cat, metrics in results.items():
            if cat == "overall":
                continue
            print(f"\n{cat}:")
            for k, v in metrics.items():
                if k != "per_class_f1":
                    print(f"  {k}: {v}")
            print("  per-class F1:")
            for cls, f1 in (metrics.get("per_class_f1") or {}).items():
                print(f"    {cls}: {f1:.4f}")

        print(f"\noverall mean_macro_f1: {results.get('overall', {}).get('mean_macro_f1', 0):.4f}")
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"saved: {args.out}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
