# -*- coding: utf-8 -*-
"""
Qwen3-VL ゼロショット分類（HuggingFace Transformers経由）

事前準備:
    pip install transformers accelerate qwen-vl-utils

    モデルは初回実行時に自動ダウンロードされる。
    GPU VRAM 要件:
      Qwen3-VL-3B  : ~8GB
      Qwen3-VL-7B  : ~16GB
      Qwen3-VL-72B : ~160GB（4-bit量子化推奨）

デフォルトモデル:
    Qwen/Qwen3-VL-7B-Instruct

使い方:
    python -m classification.models.qwen3vl \
        --csv  classification/results/labeled_val.csv \
        --out  classification/results/qwen3vl_preds.csv \
        --n    50 \
        --model Qwen/Qwen3-VL-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from classification.data.label_definitions import ALL_LABEL_SETS
from classification.prompts import SYSTEM_PROMPT, USER_PROMPT

try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL  = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 256
MAX_IMAGE_SIZE = 512   # 画像の長辺をこのピクセルに制限（VRAM節約）
RETRY_LIMIT    = 2


def _parse_response(content: str) -> dict[str, list[str]]:
    """応答テキストからJSONをパースする。"""
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()

    # <think>...</think> ブロックを除去（Qwen3系はthinkingモードあり）
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return {cat: [] for cat in ALL_LABEL_SETS}

    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {cat: [] for cat in ALL_LABEL_SETS}

    def _to_list(val) -> list[str]:
        if isinstance(val, list):
            return [str(v).strip() for v in val]
        if isinstance(val, str):
            return [val.strip()]
        return []

    return {
        "kenzenudo":   _to_list(obj.get("健全度判定", [])),
        "taisaku":     _to_list(obj.get("対策区分",   [])),
        "damage_type": _to_list(obj.get("損傷種類",   [])),
        "damage_loc":  _to_list(obj.get("損傷部位",   [])),
    }


class Qwen3VLClassifier:
    """
    Qwen3-VL を使ったゼロショット分類器。

    Args:
        model_name: HuggingFace のモデル名
        device:     "cuda" / "cpu" / "auto"
        use_flash_attn: Flash Attention 2 を使用するか
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_new_tokens: int = MAX_NEW_TOKENS,
        use_flash_attn: bool = True,
    ):
        assert _TRANSFORMERS_AVAILABLE, \
            "transformers ライブラリが必要です: pip install transformers accelerate"

        import torch
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"[Qwen3-VL] モデルを読み込み中: {model_name}")

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",
        }
        if device == "auto":
            load_kwargs["device_map"] = "auto"
        elif device != "cpu":
            load_kwargs["device_map"] = device

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, **load_kwargs
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"[Qwen3-VL] ロード完了")

    def predict(self, image_path: str) -> dict[str, list[str]]:
        """1枚の画像を分類する。"""
        import torch

        img = Image.open(image_path).convert("RGB")
        # 解像度を制限してビジュアルトークン数を抑える
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.LANCZOS)

        # Qwen-VL のチャット形式
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text":  USER_PROMPT},
                ],
            },
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input],
            images=[img],
            return_tensors="pt",
            padding=True,
        )

        if self.device == "auto":
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        elif self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        for attempt in range(RETRY_LIMIT):
            try:
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                    )
                # 入力トークン分を除去
                input_len = inputs["input_ids"].shape[1]
                output_ids = output_ids[:, input_len:]
                response = self.processor.decode(
                    output_ids[0], skip_special_tokens=True
                )
                return _parse_response(response)
            except Exception as e:
                print(f"  [Qwen3-VL] エラー (attempt {attempt+1}/{RETRY_LIMIT}): {e}")
                if attempt < RETRY_LIMIT - 1:
                    time.sleep(3)

        return {cat: [] for cat in ALL_LABEL_SETS}


def run_on_csv(
    csv_path: str,
    out_path: str,
    n: int | None = None,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
) -> None:
    df = pd.read_csv(csv_path)
    if n is not None:
        df = df.head(n)

    classifier = Qwen3VLClassifier(model_name=model_name, device=device)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Qwen3-VL 推論"):
        pred = classifier.predict(str(row["image"]))
        rows.append({
            "image":            row["image"],
            "pred_kenzenudo":   "|".join(pred["kenzenudo"]),
            "pred_taisaku":     "|".join(pred["taisaku"]),
            "pred_damage_type": "|".join(pred["damage_type"]),
            "pred_damage_loc":  "|".join(pred["damage_loc"]),
        })

    pred_df = pd.DataFrame(rows)
    result_df = df.merge(pred_df, on="image", how="left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"結果保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL ゼロショット分類")
    parser.add_argument("--csv",    required=True)
    parser.add_argument("--out",    required=True)
    parser.add_argument("--n",      type=int, default=None)
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_on_csv(
        csv_path=args.csv, out_path=args.out,
        n=args.n, model_name=args.model, device=args.device,
    )


if __name__ == "__main__":
    main()
