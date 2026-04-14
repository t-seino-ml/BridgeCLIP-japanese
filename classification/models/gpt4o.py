# -*- coding: utf-8 -*-
"""
GPT-4o ゼロショット分類（OpenAI API経由）

事前準備:
    pip install openai
    export OPENAI_API_KEY="sk-..."

使い方:
    python -m classification.models.gpt4o \
        --csv  classification/results/labeled_val.csv \
        --out  classification/results/gpt4o_preds.csv \
        --n    50          # 最初の50件のみ実行（コスト制限用）
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from classification.data.label_definitions import ALL_LABEL_SETS
from classification.prompts import SYSTEM_PROMPT, USER_PROMPT

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

MODEL_NAME = "gpt-4o"
MAX_TOKENS = 512
RETRY_LIMIT = 3
RETRY_WAIT  = 5  # seconds


def _image_to_base64(image_path: str, max_size: int = 1024) -> str:
    """画像をBase64エンコードして返す（長辺をmax_sizeにリサイズ）。"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _parse_response(content: str) -> dict[str, list[str]]:
    """
    GPT-4o の応答テキストからJSONを抽出してパースする。
    パース失敗時はすべて空リストを返す。
    """
    # コードブロック（```json ... ```）を除去
    import re
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()

    # JSONを抽出
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


class GPT4oClassifier:
    """
    GPT-4o を使ったゼロショット分類器。

    Args:
        api_key:    OpenAI APIキー（省略時は環境変数 OPENAI_API_KEY を使用）
        model:      使用するモデル（デフォルト: gpt-4o）
        max_tokens: 応答の最大トークン数
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL_NAME,
        max_tokens: int = MAX_TOKENS,
    ):
        assert _OPENAI_AVAILABLE, "openai ライブラリが必要です: pip install openai"
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def predict(self, image_path: str) -> dict[str, list[str]]:
        """
        1枚の画像パスを受け取り、4カテゴリの予測ラベルリストを返す。
        """
        b64 = _image_to_base64(image_path)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type":  "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

        for attempt in range(RETRY_LIMIT):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                content = response.choices[0].message.content or ""
                return _parse_response(content)
            except Exception as e:
                print(f"  [GPT-4o] エラー (attempt {attempt+1}/{RETRY_LIMIT}): {e}")
                if attempt < RETRY_LIMIT - 1:
                    time.sleep(RETRY_WAIT * (attempt + 1))

        return {cat: [] for cat in ALL_LABEL_SETS}


def run_on_csv(
    csv_path: str,
    out_path: str,
    n: int | None = None,
    api_key: str | None = None,
    sleep_interval: float = 0.5,
) -> None:
    """
    ラベル付きCSVの画像に対してGPT-4oで分類し、結果をCSVに保存する。

    Args:
        n: 処理件数の上限（None で全件）
        sleep_interval: API制限回避のためのリクエスト間隔（秒）
    """
    df = pd.read_csv(csv_path)
    if n is not None:
        df = df.head(n)

    classifier = GPT4oClassifier(api_key=api_key)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="GPT-4o 推論"):
        pred = classifier.predict(str(row["image"]))
        rows.append({
            "image":            row["image"],
            "pred_kenzenudo":   "|".join(pred["kenzenudo"]),
            "pred_taisaku":     "|".join(pred["taisaku"]),
            "pred_damage_type": "|".join(pred["damage_type"]),
            "pred_damage_loc":  "|".join(pred["damage_loc"]),
        })
        if sleep_interval > 0:
            time.sleep(sleep_interval)

    pred_df = pd.DataFrame(rows)
    result_df = df.merge(pred_df, on="image", how="left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"結果保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GPT-4o ゼロショット分類")
    parser.add_argument("--csv",   required=True, help="ラベル付きCSVのパス")
    parser.add_argument("--out",   required=True, help="出力CSVパス")
    parser.add_argument("--n",     type=int, default=None, help="処理件数の上限")
    parser.add_argument("--sleep", type=float, default=0.5, help="リクエスト間隔(秒)")
    args = parser.parse_args()

    run_on_csv(csv_path=args.csv, out_path=args.out, n=args.n, sleep_interval=args.sleep)


if __name__ == "__main__":
    main()
