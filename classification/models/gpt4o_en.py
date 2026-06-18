# -*- coding: utf-8 -*-
"""
GPT-4o zero-shot classification (English prompts).

Parallel to gpt4o.py but prompts the model in English and parses English JSON keys.
Emitted labels are normalized to canonical English forms via normalize_label.

Setup:
    export OPENAI_API_KEY="sk-..."

Usage:
    python -m classification.models.gpt4o_en \
        --csv  classification/results/labeled_val_en.csv \
        --out  classification/results/gpt4o_en_preds.csv \
        --n    50
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

from classification.data.label_definitions_en import ALL_LABEL_SETS, normalize_label
from classification.prompts_en import SYSTEM_PROMPT, USER_PROMPT, JSON_KEYS

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

MODEL_NAME = "gpt-4o"
MAX_TOKENS = 512
RETRY_LIMIT = 3
RETRY_WAIT = 5


def _image_to_base64(image_path: str, max_size: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _to_list(val) -> list[str]:
    if isinstance(val, list):
        return [str(v).strip() for v in val]
    if isinstance(val, str):
        return [val.strip()]
    return []


def _parse_response(content: str) -> dict[str, list[str]]:
    import re
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

    out: dict[str, list[str]] = {}
    for cat, key in JSON_KEYS.items():
        raws = _to_list(obj.get(key, []))
        canonical: list[str] = []
        for r in raws:
            norm = normalize_label(cat, r)
            if norm is not None and norm not in canonical:
                canonical.append(norm)
        out[cat] = canonical
    return out


class GPT4oClassifier:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL_NAME,
        max_tokens: int = MAX_TOKENS,
    ):
        assert _OPENAI_AVAILABLE, "openai library required: pip install openai"
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def predict(self, image_path: str) -> dict[str, list[str]]:
        b64 = _image_to_base64(image_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
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
                print(f"  [GPT-4o-EN] error (attempt {attempt+1}/{RETRY_LIMIT}): {e}")
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
    df = pd.read_csv(csv_path)
    if n is not None:
        df = df.head(n)

    classifier = GPT4oClassifier(api_key=api_key)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="GPT-4o-EN inference"):
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
    print(f"saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GPT-4o zero-shot classification (English)")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()
    run_on_csv(csv_path=args.csv, out_path=args.out, n=args.n, sleep_interval=args.sleep)


if __name__ == "__main__":
    main()
