#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate baseline (pretrained) OpenCLIP on a CSV val set for retrieval metrics.

Input CSV format:
  columns: image,text
  image: absolute path to image file
  text : caption string (Japanese OK)

Outputs:
  - JSON with retrieval metrics
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import open_clip


class CsvImageTextDataset(Dataset):
    def __init__(self, csv_path: Path, preprocess, img_key: str = "image", text_key: str = "text"):
        self.csv_path = Path(csv_path)
        df = pd.read_csv(self.csv_path)
        if img_key not in df.columns or text_key not in df.columns:
            raise KeyError(f"CSV must have columns '{img_key}' and '{text_key}'. got={list(df.columns)}")

        # drop rows with missing
        df = df[[img_key, text_key]].dropna()
        self.images = df[img_key].astype(str).tolist()
        self.texts = df[text_key].astype(str).tolist()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        text = self.texts[idx]

        # PIL load
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        img = self.preprocess(im)
        return img, text, img_path


def collate_fn(batch):
    # images: torch tensors, texts: strings
    images = torch.stack([b[0] for b in batch], dim=0)
    texts = [b[1] for b in batch]
    paths = [b[2] for b in batch]
    return images, texts, paths


@torch.no_grad()
def encode_all(model, tokenizer, dataloader, device, amp: bool):
    model.eval()
    all_img = []
    all_txt = []
    all_paths = []
    all_texts = []

    for images, texts, paths in dataloader:
        images = images.to(device, non_blocking=True)

        # Tokenize on CPU then move
        tok = tokenizer(texts)
        if isinstance(tok, dict):
            # some tokenizers return dict
            tok = {k: v.to(device, non_blocking=True) for k, v in tok.items()}
        else:
            tok = tok.to(device, non_blocking=True)

        ctx = torch.cuda.amp.autocast(enabled=amp) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        with ctx:
            img_f = model.encode_image(images)
            txt_f = model.encode_text(tok)

        img_f = F.normalize(img_f, dim=-1)
        txt_f = F.normalize(txt_f, dim=-1)

        all_img.append(img_f.cpu())
        all_txt.append(txt_f.cpu())
        all_paths.extend(paths)
        all_texts.extend(texts)

    img_feats = torch.cat(all_img, dim=0)  # [N, D]
    txt_feats = torch.cat(all_txt, dim=0)  # [N, D]
    return img_feats, txt_feats, all_paths, all_texts


def ranks_from_similarity(sim: np.ndarray):
    """
    sim: [N, N] similarity matrix (query x candidates).
    Ground-truth for query i is candidate i.
    Returns:
      ranks: [N] 1-based rank of GT
    """
    # descending sort indices
    order = np.argsort(-sim, axis=1)  # [N, N]
    gt = np.arange(sim.shape[0])[:, None]
    # position where order == gt
    pos = (order == gt).argmax(axis=1)
    ranks = pos + 1  # 1-based
    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def mean_rank(ranks: np.ndarray) -> float:
    return float(np.mean(ranks))


def median_rank(ranks: np.ndarray) -> float:
    return float(np.median(ranks))


@torch.no_grad()
def clip_val_loss(img_feats: torch.Tensor, txt_feats: torch.Tensor, logit_scale: float):
    """
    Compute symmetric InfoNCE loss on the full val set (may be heavy for huge N).
    For your val (≈2.7k) it's fine.
    """
    device = torch.device("cpu")
    img = img_feats.to(device)
    txt = txt_feats.to(device)

    logits_per_image = logit_scale * (img @ txt.T)  # [N, N]
    logits_per_text = logits_per_image.T

    labels = torch.arange(img.shape[0], device=device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return float(((loss_i + loss_t) * 0.5).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True, help="val csv path (columns: image,text)")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", help="use autocast on CUDA")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")

    # load pretrained baseline
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    model = model.to(device)
    model.eval()

    ds = CsvImageTextDataset(Path(args.val_csv), preprocess, img_key="image", text_key="text")
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    img_feats, txt_feats, paths, texts = encode_all(model, tokenizer, dl, device=device, amp=amp)

    # similarity matrix
    # [N, D] @ [D, N] -> [N, N]
    sim_i2t = (img_feats.numpy() @ txt_feats.numpy().T).astype(np.float32)
    sim_t2i = sim_i2t.T

    ranks_i2t = ranks_from_similarity(sim_i2t)
    ranks_t2i = ranks_from_similarity(sim_t2i)

    # logit_scale (OpenCLIP stores as parameter; exp() typically)
    try:
        logit_scale = float(model.logit_scale.exp().detach().cpu().item())
    except Exception:
        logit_scale = 100.0

    val_loss = clip_val_loss(img_feats, txt_feats, logit_scale=logit_scale)

    metrics = {
        "model": args.model,
        "pretrained": args.pretrained,
        "num_samples": int(len(ds)),
        "image_to_text_mean_rank": mean_rank(ranks_i2t),
        "image_to_text_median_rank": median_rank(ranks_i2t),
        "image_to_text_R@1": recall_at_k(ranks_i2t, 1),
        "image_to_text_R@5": recall_at_k(ranks_i2t, 5),
        "image_to_text_R@10": recall_at_k(ranks_i2t, 10),
        "text_to_image_mean_rank": mean_rank(ranks_t2i),
        "text_to_image_median_rank": median_rank(ranks_t2i),
        "text_to_image_R@1": recall_at_k(ranks_t2i, 1),
        "text_to_image_R@5": recall_at_k(ranks_t2i, 5),
        "text_to_image_R@10": recall_at_k(ranks_t2i, 10),
        "clip_val_loss": val_loss,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()