# -*- coding: utf-8 -*-
"""
Qwen3-VL zero-shot classification (English prompts).

Parallel to qwen3vl.py. Uses prompts_en / label_definitions_en and normalizes
emitted labels to canonical English forms.
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

from classification.data.label_definitions_en import ALL_LABEL_SETS, normalize_label
from classification.prompts_en import SYSTEM_PROMPT, USER_PROMPT, JSON_KEYS

try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL  = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 256
MAX_IMAGE_SIZE = 512
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
    # Strip <think>...</think> (Qwen thinking mode).
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

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


class Qwen3VLClassifier:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_new_tokens: int = MAX_NEW_TOKENS,
        use_flash_attn: bool = True,
    ):
        assert _TRANSFORMERS_AVAILABLE, \
            "transformers library required: pip install transformers accelerate"
        import torch
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"[Qwen3-VL-EN] loading model: {model_name}")
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
        print(f"[Qwen3-VL-EN] loaded")

    def predict(self, image_path: str) -> dict[str, list[str]]:
        import torch
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.LANCZOS)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": USER_PROMPT},
                ],
            },
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text_input], images=[img], return_tensors="pt", padding=True,
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
                input_len = inputs["input_ids"].shape[1]
                output_ids = output_ids[:, input_len:]
                response = self.processor.decode(output_ids[0], skip_special_tokens=True)
                return _parse_response(response)
            except Exception as e:
                print(f"  [Qwen3-VL-EN] error (attempt {attempt+1}/{RETRY_LIMIT}): {e}")
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
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Qwen3-VL-EN inference"):
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
    parser = argparse.ArgumentParser(description="Qwen3-VL zero-shot classification (English)")
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
