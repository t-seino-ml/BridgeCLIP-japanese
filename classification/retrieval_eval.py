# -*- coding: utf-8 -*-
"""
CLIP 検索評価スクリプト（Image-to-Text / Text-to-Image / Image-to-Image）

使い方:
    # ベースCLIP
    python -m classification.retrieval_eval \
        --val_csv classification/results/unified_val_user.csv \
        --out     classification/results/clip_base_retrieval.json

    # ファインチューニング済み（ckpt直接指定）
    python -m classification.retrieval_eval \
        --val_csv  classification/results/unified_val_user.csv \
        --ckpt     logs_classification/bridgeclip_vitb32_unified/checkpoints/epoch_5.pt \
        --out      classification/results/clip_finetuned_retrieval.json

    # ファインチューニング済み（ckpt_dirでベスト自動選択）
    python -m classification.retrieval_eval \
        --val_csv  classification/results/unified_val_user.csv \
        --ckpt_dir logs_classification/bridgeclip_vitb32_unified/checkpoints \
        --out      classification/results/clip_finetuned_retrieval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.preprocess(img), idx


def find_best_checkpoint(ckpt_dir: str) -> str:
    ckpt_dir = Path(ckpt_dir)
    results_file = ckpt_dir / "results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(f"results.jsonl が見つかりません: {results_file}")
    best_epoch, best_loss = -1, float("inf")
    with open(results_file) as f:
        for line in f:
            d = json.loads(line)
            loss = d.get("clip_val_loss", float("inf"))
            epoch = d.get("epoch", -1)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
    ckpt_path = ckpt_dir / f"epoch_{best_epoch}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"チェックポイントが見つかりません: {ckpt_path}")
    print(f"[CLIP] ベストエポック: epoch {best_epoch} (val_loss={best_loss:.4f})")
    return str(ckpt_path)


@torch.no_grad()
def extract_image_features(model, preprocess, paths, batch_size=128, num_workers=4):
    ds = ImagePathDataset(paths, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    feats = np.zeros((len(paths), model.visual.output_dim), dtype=np.float32)
    for imgs, indices in tqdm(dl, desc="画像特徴抽出"):
        imgs = imgs.to(DEVICE)
        f = F.normalize(model.encode_image(imgs), dim=-1).cpu().numpy()
        for feat, idx in zip(f, indices.numpy()):
            feats[idx] = feat
    return feats


@torch.no_grad()
def extract_text_features(model, tokenizer, texts, batch_size=512):
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="テキスト特徴抽出"):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch).to(DEVICE)
        f = F.normalize(model.encode_text(tokens), dim=-1).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def recall_at_k(sim_matrix: np.ndarray, ks=(1, 5, 10), exclude_self=False) -> dict:
    """
    sim_matrix: (N_query, N_gallery)
    正解は対角成分（クエリiの正解はギャラリーi）
    """
    N = sim_matrix.shape[0]
    results = {}
    for k in ks:
        if exclude_self:
            # 自身を除いてランク付け
            hits = 0
            for i in range(N):
                row = sim_matrix[i].copy()
                row[i] = -np.inf
                topk = np.argsort(row)[::-1][:k]
                if i in topk:
                    hits += 1
            results[f"R@{k}"] = hits / N
        else:
            topk_idx = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :k]
            hits = sum(1 for i in range(N) if i in topk_idx[i])
            results[f"R@{k}"] = hits / N
    return results


def run_retrieval_eval(
    val_csv: str,
    out_path: str,
    ckpt_path: str | None = None,
    ckpt_dir: str | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    batch_size: int = 128,
    num_workers: int = 4,
) -> None:
    if ckpt_dir is not None:
        ckpt_path = find_best_checkpoint(ckpt_dir)

    print(f"[CLIP] モデルを読み込み中: {model_name}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    if ckpt_path is not None:
        print(f"[CLIP] チェックポイントを読み込み中: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print("[CLIP] 読み込み完了")

    model = model.to(DEVICE).eval()

    df = pd.read_csv(val_csv)
    paths = df["image"].tolist()
    texts = df["text"].fillna("").tolist()
    N = len(df)
    print(f"評価サンプル数: {N}")

    img_feats  = extract_image_features(model, preprocess, paths,
                                        batch_size=batch_size, num_workers=num_workers)
    text_feats = extract_text_features(model, tokenizer, texts, batch_size=512)

    # 類似度行列 (N, N)
    sim_i2t = img_feats  @ text_feats.T   # image → text
    sim_t2i = text_feats @ img_feats.T    # text  → image
    sim_i2i = img_feats  @ img_feats.T    # image → image

    ks = (1, 5, 10)
    i2t = recall_at_k(sim_i2t, ks=ks, exclude_self=False)
    t2i = recall_at_k(sim_t2i, ks=ks, exclude_self=False)
    i2i = recall_at_k(sim_i2i, ks=ks, exclude_self=True)   # 自身を除外

    result = {
        "ckpt": ckpt_path or "base",
        "val_csv": val_csv,
        "N": N,
        "i2t": i2t,
        "t2i": t2i,
        "i2i": i2i,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"結果保存: {out_path}")

    print("\n=== Image-to-Text ===")
    for k_str, v in i2t.items(): print(f"  {k_str}: {v:.4f}")
    print("=== Text-to-Image ===")
    for k_str, v in t2i.items(): print(f"  {k_str}: {v:.4f}")
    print("=== Image-to-Image（自身除外） ===")
    for k_str, v in i2i.items(): print(f"  {k_str}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CLIP 検索評価")
    parser.add_argument("--val_csv",     required=True)
    parser.add_argument("--out",         required=True)
    parser.add_argument("--ckpt",        default=None)
    parser.add_argument("--ckpt_dir",    default=None, help="ベストエポック自動選択")
    parser.add_argument("--model",       default="ViT-B-32")
    parser.add_argument("--pretrained",  default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run_retrieval_eval(
        val_csv=args.val_csv,
        out_path=args.out,
        ckpt_path=args.ckpt,
        ckpt_dir=args.ckpt_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
