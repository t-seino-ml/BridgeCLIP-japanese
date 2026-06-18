# -*- coding: utf-8 -*-
"""
Llama vision zero-shot classification via Ollama (English prompts).

Parallel to llama.py. Uses prompts_en / label_definitions_en.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from classification.data.label_definitions_en import ALL_LABEL_SETS, normalize_label
from classification.prompts_en import SYSTEM_PROMPT, USER_PROMPT, JSON_KEYS

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

DEFAULT_MODEL  = "llama3.2-vision"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MAX_TOKENS     = 512
RETRY_LIMIT    = 3
RETRY_WAIT     = 5


def _image_to_base64(image_path: str, max_size: int = 1024) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
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


class LlamaClassifier:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_url: str = OLLAMA_API_URL,
        max_tokens: int = MAX_TOKENS,
    ):
        assert _REQUESTS_AVAILABLE, "requests library required: pip install requests"
        self.model = model
        self.api_url = api_url
        self.max_tokens = max_tokens
        self._check_connection()

    def _check_connection(self) -> None:
        import requests
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in m for m in models):
                print(f"[warn] model '{self.model}' not found.")
                print(f"  available: {models}")
                print(f"  run: ollama pull {self.model}")
        except Exception as e:
            print(f"[warn] cannot connect to Ollama server: {e}")
            print("  start it with: ollama serve")

    def predict(self, image_path: str) -> dict[str, list[str]]:
        import requests
        b64 = _image_to_base64(image_path)
        payload = {
            "model":  self.model,
            "stream": False,
            "options": {"num_predict": self.max_tokens, "temperature": 0.0},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role":    "user",
                    "content": USER_PROMPT,
                    "images":  [b64],
                },
            ],
        }
        for attempt in range(RETRY_LIMIT):
            try:
                resp = requests.post(self.api_url, json=payload, timeout=120)
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                return _parse_response(content)
            except Exception as e:
                print(f"  [Llama-EN] error (attempt {attempt+1}/{RETRY_LIMIT}): {e}")
                if attempt < RETRY_LIMIT - 1:
                    time.sleep(RETRY_WAIT)
        return {cat: [] for cat in ALL_LABEL_SETS}


def run_on_csv(
    csv_path: str,
    out_path: str,
    n: int | None = None,
    model: str = DEFAULT_MODEL,
    sleep_interval: float = 0.1,
) -> None:
    df = pd.read_csv(csv_path)
    if n is not None:
        df = df.head(n)

    classifier = LlamaClassifier(model=model)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Llama-EN inference"):
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
    parser = argparse.ArgumentParser(description="Llama zero-shot classification (English)")
    parser.add_argument("--csv",   required=True)
    parser.add_argument("--out",   required=True)
    parser.add_argument("--n",     type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()
    run_on_csv(csv_path=args.csv, out_path=args.out,
               n=args.n, model=args.model, sleep_interval=args.sleep)


if __name__ == "__main__":
    main()
