# -*- coding: utf-8 -*-
"""
InternVL zero-shot classification (English prompts).

Parallel to internvl.py. Uses prompts_en / label_definitions_en.
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

from classification.data.label_definitions_en import ALL_LABEL_SETS, normalize_label
from classification.prompts_en import SYSTEM_PROMPT, USER_PROMPT, JSON_KEYS

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL  = "OpenGVLab/InternVL3-8B"
MAX_NEW_TOKENS = 512
RETRY_LIMIT    = 2


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


class InternVLClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        assert _TRANSFORMERS_AVAILABLE, \
            "transformers library required: pip install transformers accelerate"
        import torch
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"[InternVL-EN] loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        # Force CPU as the default device during __init__ so torch.linspace()
        # in InternViT's constructor does NOT create meta tensors (newer HF
        # model code triggers this; `low_cpu_mem_usage=False` alone no longer
        # prevents it on recent transformers). Use `torch_dtype` for
        # compatibility with transformers 4.4x (renamed to `dtype` in 4.52+).
        with torch.device("cpu"):
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            ).eval()

        if device == "auto":
            target = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target = device
        if target != "cpu":
            self.model = self.model.to(target)
        self._target_device = target
        print(f"[InternVL-EN] loaded (device={target})")

    def predict(self, image_path: str) -> dict[str, list[str]]:
        import torch
        from torchvision import transforms

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

        # InternVL.chat() does not accept a system prompt — embed it inline.
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
                print(f"  [InternVL-EN] error (attempt {attempt+1}/{RETRY_LIMIT}): {e}")
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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="InternVL-EN inference"):
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
    print(f"saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="InternVL zero-shot classification (English)")
    parser.add_argument("--csv",    required=True)
    parser.add_argument("--out",    required=True)
    parser.add_argument("--n",      type=int, default=None)
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_on_csv(csv_path=args.csv, out_path=args.out, n=args.n,
               model_name=args.model, device=args.device)


if __name__ == "__main__":
    main()
