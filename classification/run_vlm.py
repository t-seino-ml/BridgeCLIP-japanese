# -*- coding: utf-8 -*-
"""
VLMゼロショット分類 一括実行スクリプト

Base CLIP / GPT-4o / Llama / InternVL3.5 / Qwen3-VL を順番に実行し、
evaluation.py で比較評価する。

使い方:
    # すべてのVLMを実行
    python -m classification.run_vlm \
        --csv   classification/results/labeled_val.csv \
        --out   classification/results \
        --n     200 \
        --models clip gpt4o llama internvl qwen3vl

    # 特定のモデルのみ
    python -m classification.run_vlm \
        --csv   classification/results/labeled_val.csv \
        --out   classification/results \
        --n     50 \
        --models gpt4o

注意:
    - GPT-4o  : OPENAI_API_KEY 環境変数が必要
    - Llama   : ollama serve が起動している必要がある
    - InternVL / Qwen3-VL : GPU が必要（モデルは初回実行時にダウンロード）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_clip(csv: str, out_dir: str, batch_size: int = 64, threshold: float = 0.3) -> None:
    from classification.models.clip_zeroshot import run_on_csv
    out_path = str(Path(out_dir) / "clip_zeroshot_preds.csv")
    print("\n" + "=" * 60)
    print("Base CLIP ゼロショット分類")
    print("=" * 60)
    run_on_csv(
        csv_path=csv, out_path=out_path,
        batch_size=batch_size, threshold=threshold,
    )


def run_gpt4o(csv: str, out_dir: str, n: int | None, sleep: float = 0.5) -> None:
    from classification.models.gpt4o import run_on_csv
    out_path = str(Path(out_dir) / "gpt4o_preds.csv")
    print("\n" + "=" * 60)
    print("GPT-4o ゼロショット分類")
    print("=" * 60)
    run_on_csv(csv_path=csv, out_path=out_path, n=n, sleep_interval=sleep)


def run_llama(csv: str, out_dir: str, n: int | None, model: str = "llama3.2-vision") -> None:
    from classification.models.llama import run_on_csv
    model_tag = model.replace("/", "_").replace(":", "_")
    out_path = str(Path(out_dir) / f"llama_{model_tag}_preds.csv")
    print("\n" + "=" * 60)
    print(f"Llama ({model}) ゼロショット分類")
    print("=" * 60)
    run_on_csv(csv_path=csv, out_path=out_path, n=n, model=model)


def run_internvl(
    csv: str, out_dir: str, n: int | None,
    model: str = "OpenGVLab/InternVL3-8B",
) -> None:
    from classification.models.internvl import run_on_csv
    model_tag = model.replace("/", "_")
    out_path = str(Path(out_dir) / f"internvl_{model_tag}_preds.csv")
    print("\n" + "=" * 60)
    print(f"InternVL ({model}) ゼロショット分類")
    print("=" * 60)
    run_on_csv(csv_path=csv, out_path=out_path, n=n, model_name=model)


def run_qwen3vl(
    csv: str, out_dir: str, n: int | None,
    model: str = "Qwen/Qwen3-VL-7B-Instruct",
) -> None:
    from classification.models.qwen3vl import run_on_csv
    model_tag = model.replace("/", "_")
    out_path = str(Path(out_dir) / f"qwen3vl_{model_tag}_preds.csv")
    print("\n" + "=" * 60)
    print(f"Qwen3-VL ({model}) ゼロショット分類")
    print("=" * 60)
    run_on_csv(csv_path=csv, out_path=out_path, n=n, model_name=model)


def run_evaluate(out_dir: str) -> None:
    from classification.evaluate import compare_models, print_summary_table
    import json

    print("\n" + "=" * 60)
    print("全モデル比較評価")
    print("=" * 60)
    comparison = compare_models(out_dir)
    print_summary_table(comparison)

    out_path = Path(out_dir) / "comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n比較結果保存: {out_path}")


AVAILABLE_MODELS = ["clip", "gpt4o", "llama", "internvl", "qwen3vl"]


def main():
    parser = argparse.ArgumentParser(description="VLMゼロショット分類 一括実行")
    parser.add_argument(
        "--csv",    required=True,
        help="ラベル付きCSVのパス（extract_labels.py の出力）",
    )
    parser.add_argument(
        "--out",    required=True,
        help="予測結果CSVの保存先ディレクトリ",
    )
    parser.add_argument(
        "--n",      type=int, default=None,
        help="処理件数の上限（None で全件）",
    )
    parser.add_argument(
        "--models", nargs="+", default=AVAILABLE_MODELS,
        choices=AVAILABLE_MODELS,
        help=f"実行するモデル（デフォルト: 全て）",
    )
    parser.add_argument("--batch_size",   type=int,   default=64,    help="CLIP のバッチサイズ")
    parser.add_argument("--clip_threshold", type=float, default=0.3, help="CLIP のマルチラベル閾値")
    parser.add_argument("--llama_model",  default="llama3.2-vision", help="Ollama モデル名")
    parser.add_argument("--internvl_model", default="OpenGVLab/InternVL3-8B")
    parser.add_argument("--qwen3vl_model",  default="Qwen/Qwen3-VL-7B-Instruct")
    parser.add_argument("--no_eval",      action="store_true",       help="評価をスキップ")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    for model in args.models:
        try:
            if model == "clip":
                run_clip(args.csv, args.out, args.batch_size, args.clip_threshold)
            elif model == "gpt4o":
                run_gpt4o(args.csv, args.out, args.n)
            elif model == "llama":
                run_llama(args.csv, args.out, args.n, args.llama_model)
            elif model == "internvl":
                run_internvl(args.csv, args.out, args.n, args.internvl_model)
            elif model == "qwen3vl":
                run_qwen3vl(args.csv, args.out, args.n, args.qwen3vl_model)
        except Exception as e:
            print(f"\n[エラー] {model}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  {model} をスキップして続行します。")

    if not args.no_eval:
        run_evaluate(args.out)


if __name__ == "__main__":
    main()
