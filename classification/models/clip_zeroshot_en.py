# -*- coding: utf-8 -*-
"""
Base CLIP zero-shot classification (English text templates).

Parallel to clip_zeroshot.py. Uses prompts_en.CLIP_TEMPLATES and
label_definitions_en.ALL_LABEL_SETS.

Usage:
    python -m classification.models.clip_zeroshot_en \
        --csv classification/results/labeled_val_en.csv \
        --out classification/results/clip_zeroshot_en_preds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from classification.data.label_definitions_en import ALL_LABEL_SETS
from classification.prompts_en import CLIP_TEMPLATES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPZeroShotClassifier:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        threshold: float = 0.3,
        device: str = DEVICE,
    ):
        self.threshold = threshold
        self.device = device

        print(f"[CLIP-EN] loading model: {model_name} ({pretrained})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        self._text_embeds: dict[str, torch.Tensor] = {}
        self._build_text_embeddings()

    @torch.no_grad()
    def _build_text_embeddings(self) -> None:
        for cat, label_list in ALL_LABEL_SETS.items():
            templates = CLIP_TEMPLATES.get(cat, {})
            class_embeds = []
            for label in label_list:
                texts = templates.get(label, [f"a photograph of a bridge — {label}"])
                tokens = self.tokenizer(texts).to(self.device)
                embeds = self.model.encode_text(tokens)
                embeds = F.normalize(embeds, dim=-1)
                class_embed = embeds.mean(dim=0)
                class_embed = F.normalize(class_embed, dim=-1)
                class_embeds.append(class_embed)
            self._text_embeds[cat] = torch.stack(class_embeds, dim=0)
        print("[CLIP-EN] text embeddings pre-computed")

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> dict[str, list[str]]:
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        img_embed = self.model.encode_image(img_tensor)
        img_embed = F.normalize(img_embed, dim=-1)

        predictions: dict[str, list[str]] = {}
        for cat, label_list in ALL_LABEL_SETS.items():
            text_embeds = self._text_embeds[cat]
            sims = (img_embed @ text_embeds.T).squeeze(0)
            probs = sims.softmax(dim=-1).cpu().numpy()

            if cat in ("kenzenudo", "taisaku"):
                best_idx = int(np.argmax(probs))
                predictions[cat] = [label_list[best_idx]]
            else:
                selected = [label_list[i] for i, p in enumerate(probs) if p >= self.threshold]
                if not selected:
                    selected = [label_list[int(np.argmax(probs))]]
                predictions[cat] = selected
        return predictions

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, list[str]]]:
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        img_embeds = self.model.encode_image(tensors)
        img_embeds = F.normalize(img_embeds, dim=-1)

        batch_predictions: list[dict[str, list[str]]] = [
            {cat: [] for cat in ALL_LABEL_SETS} for _ in images
        ]
        for cat, label_list in ALL_LABEL_SETS.items():
            text_embeds = self._text_embeds[cat]
            sims = img_embeds @ text_embeds.T
            probs = sims.softmax(dim=-1).cpu().numpy()
            for b_idx, prob_row in enumerate(probs):
                if cat in ("kenzenudo", "taisaku"):
                    best_idx = int(np.argmax(prob_row))
                    batch_predictions[b_idx][cat] = [label_list[best_idx]]
                else:
                    selected = [label_list[i] for i, p in enumerate(prob_row)
                                if p >= self.threshold]
                    if not selected:
                        selected = [label_list[int(np.argmax(prob_row))]]
                    batch_predictions[b_idx][cat] = selected
        return batch_predictions


def run_on_csv(
    csv_path: str,
    out_path: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    batch_size: int = 64,
    threshold: float = 0.3,
) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path)
    classifier = CLIPZeroShotClassifier(
        model_name=model_name, pretrained=pretrained, threshold=threshold,
    )

    all_preds: list[dict] = []
    images: list[Image.Image] = []
    img_paths: list[str] = []

    def _flush(paths, imgs):
        preds = classifier.predict_batch(imgs)
        for path, pred in zip(paths, preds):
            all_preds.append({
                "image": path,
                "pred_kenzenudo":   "|".join(pred["kenzenudo"]),
                "pred_taisaku":     "|".join(pred["taisaku"]),
                "pred_damage_type": "|".join(pred["damage_type"]),
                "pred_damage_loc":  "|".join(pred["damage_loc"]),
            })

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CLIP-EN zero-shot inference"):
        try:
            img = Image.open(str(row["image"])).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        images.append(img)
        img_paths.append(str(row["image"]))
        if len(images) >= batch_size:
            _flush(img_paths, images)
            images, img_paths = [], []
    if images:
        _flush(img_paths, images)

    pred_df = pd.DataFrame(all_preds)
    result_df = df.merge(pred_df, on="image", how="left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="CLIP zero-shot classification (English)")
    parser.add_argument("--csv",        required=True)
    parser.add_argument("--out",        required=True)
    parser.add_argument("--model",      default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold",  type=float, default=0.3)
    args = parser.parse_args()
    run_on_csv(
        csv_path=args.csv, out_path=args.out,
        model_name=args.model, pretrained=args.pretrained,
        batch_size=args.batch_size, threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
