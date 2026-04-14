# -*- coding: utf-8 -*-
"""
InternVL3.5 ゼロショット分類（HuggingFace Transformers経由）

事前準備:
    pip install transformers accelerate einops timm

    モデルは初回実行時に自動ダウンロードされる。
    GPU VRAM 要件:
      InternVL3-2B  : ~6GB
      InternVL3-8B  : ~20GB
      InternVL3-14B : ~35GB

デフォルトモデル:
    OpenGVLab/InternVL3-8B

使い方:
    python -m classification.models.internvl \
        --csv  classification/results/labeled_val.csv \
        --out  classification/results/internvl_preds.csv \
        --n    50 \
        --model OpenGVLab/InternVL3-8B
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from classification.data.label_definitions import ALL_LABEL_SETS
from classification.prompts import SYSTEM_PROMPT, USER_PROMPT

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL  = "OpenGVLab/InternVL3-8B"
MAX_NEW_TOKENS = 512
RETRY_LIMIT    = 2


def _parse_response(content: str) -> dict[str, list[str]]:
    """応答テキストからJSONをパースする。"""
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()

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


class InternVLClassifier:
    """
    InternVL3.5 を使ったゼロショット分類器。

    Args:
        model_name: HuggingFace のモデル名
        device:     "cuda" / "cpu" / "auto"
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        assert _TRANSFORMERS_AVAILABLE, \
            "transformers ライブラリが必要です: pip install transformers accelerate"

        import torch
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"[InternVL] モデルを読み込み中: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        # device_map はメタテンソルを使うため InternVL の __init__ で .item() が失敗する。
        # CPU にロードしてから GPU に移動する。
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        ).eval()

        if device == "auto":
            target = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target = device
        if target != "cpu":
            self.model = self.model.to(target)
        self._target_device = target
        print(f"[InternVL] ロード完了 (device={target})")

    def predict(self, image_path: str) -> dict[str, list[str]]:
        """1枚の画像を分類する。"""
        import torch
        from torchvision import transforms

        # InternVL 用の画像前処理
        img = Image.open(image_path).convert("RGB")
        img = img.resize((448, 448), Image.LANCZOS)

        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        pixel_values = tf(img).unsqueeze(0)

        pixel_values = pixel_values.to(torch.bfloat16).to(self._target_device)

        # プロンプト構築（InternVL の <image> トークン形式）
        # chat() は system_prompt を受け付けないため、質問文に埋め込む
        prompt = f"<image>\n{SYSTEM_PROMPT}\n\n{USER_PROMPT}"

        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        }

        for attempt in range(RETRY_LIMIT):
            try:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    generation_config,
                    history=None,
                    return_history=False,
                )
                return _parse_response(response)
            except Exception as e:
                print(f"  [InternVL] エラー (attempt {attempt+1}/{RETRY_LIMIT}): {e}")
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

    classifier = InternVLClassifier(model_name=model_name, device=device)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="InternVL 推論"):
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
    parser = argparse.ArgumentParser(description="InternVL3.5 ゼロショット分類")
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
