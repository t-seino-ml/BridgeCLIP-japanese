# -*- coding: utf-8 -*-
"""
定性評価：CLIP 検索結果の可視化

I2T（画像→テキスト）、T2I（テキスト→画像）、I2I（画像→画像）の
上位k件を画像付きで保存する。

使い方:
    # ベースCLIP
    python -m classification.qualitative_retrieval \
        --val_csv   classification/results/unified_val_user.csv \
        --out_dir   classification/results/qualitative/clip_base \
        --n_queries 5

    # ファインチューニング済みCLIP（epoch 5）
    python -m classification.qualitative_retrieval \
        --val_csv   classification/results/unified_val_user.csv \
        --ckpt_dir  logs_classification/bridgeclip_vitb32_unified/checkpoints \
        --out_dir   classification/results/qualitative/clip_finetuned \
        --n_queries 5
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── レイアウト定数 ──
IMG_W        = 200   # 各セルの画像幅
IMG_H        = 160   # 各セルの画像高さ
TEXT_H       = 90    # テキスト領域の高さ
SCORE_H      = 22    # スコア表示行の高さ
HEADER_H     = 28    # ヘッダラベル行の高さ
CELL_W       = IMG_W
CELL_H       = HEADER_H + IMG_H + SCORE_H + TEXT_H
PAD          = 8
BG_COLOR     = (255, 255, 255)
HEADER_COLOR = (220, 230, 255)
QUERY_COLOR  = (255, 230, 200)
SCORE_COLOR  = (230, 255, 230)
BORDER_COLOR = (160, 160, 160)
TEXT_COLOR   = (30, 30, 30)
SCORE_FG     = (0, 120, 0)


def _load_font(size: int):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


FONT_SM  = _load_font(11)
FONT_MD  = _load_font(13)
FONT_LG  = _load_font(15)


def _wrap_text(text: str, font, max_width: int) -> list[str]:
    """フォント幅を考慮してテキストを折り返す。"""
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    lines, current = [], ""
    for ch in text:
        test = current + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        w = bbox[2] - bbox[0]
        if w > max_width:
            if current:
                lines.append(current)
            current = ch
        else:
            current = test
    if current:
        lines.append(current)
    return lines


def _draw_cell(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    x: int,
    y: int,
    header: str,
    img_path: str | None,
    score: float | None,
    caption: str,
    bg_color: tuple,
):
    """1セル（ヘッダ＋画像＋スコア＋テキスト）を描画する。"""
    w, h = CELL_W, CELL_H
    # 背景
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=bg_color, outline=BORDER_COLOR)

    # ヘッダラベル
    hdr_bg = HEADER_COLOR if "k" in header else QUERY_COLOR
    draw.rectangle([x, y, x + w - 1, y + HEADER_H - 1], fill=hdr_bg)
    draw.text((x + PAD, y + 4), header, font=FONT_LG, fill=TEXT_COLOR)

    # 画像
    img_y = y + HEADER_H
    if img_path and Path(img_path).exists():
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
            canvas.paste(img, (x, img_y))
        except Exception:
            draw.rectangle([x, img_y, x + IMG_W - 1, img_y + IMG_H - 1],
                           fill=(200, 200, 200))
            draw.text((x + 4, img_y + IMG_H // 2), "読込エラー", font=FONT_SM, fill=(100,))
    else:
        draw.rectangle([x, img_y, x + IMG_W - 1, img_y + IMG_H - 1], fill=(220, 220, 220))
        draw.text((x + 4, img_y + IMG_H // 2), "画像なし", font=FONT_SM, fill=(100,))

    # スコア
    score_y = img_y + IMG_H
    if score is not None:
        score_str = f"cos={score:.4f}"
        draw.rectangle([x, score_y, x + w - 1, score_y + SCORE_H - 1], fill=SCORE_COLOR)
        draw.text((x + PAD, score_y + 3), score_str, font=FONT_MD, fill=SCORE_FG)

    # テキスト
    text_y = score_y + SCORE_H
    lines = _wrap_text(caption, FONT_SM, w - PAD * 2)
    ty = text_y + 4
    for line in lines[:5]:
        draw.text((x + PAD, ty), line, font=FONT_SM, fill=TEXT_COLOR)
        ty += 16
        if ty + 16 > y + h:
            break


def render_i2t(
    query_idx: int,
    topk_idx: list[int],
    topk_scores: list[float],
    df: pd.DataFrame,
    out_path: Path,
):
    """Image-to-Text: クエリ画像 → 上位k件テキスト（テキスト側の画像を表示）"""
    k = len(topk_idx)
    total_w = CELL_W * (k + 1) + PAD * (k + 2)
    canvas = Image.new("RGB", (total_w, CELL_H + PAD * 2), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    row = df.iloc[query_idx]
    # クエリセル
    _draw_cell(draw, canvas, PAD, PAD,
               "QUERY", row["image"], None, row.get("text", ""), QUERY_COLOR)

    # 結果セル
    for rank, (idx, score) in enumerate(zip(topk_idx, topk_scores)):
        r = df.iloc[idx]
        x = PAD + CELL_W * (rank + 1) + PAD * (rank + 1)
        label = f"k{rank+1}: {score:.3f}"
        _draw_cell(draw, canvas, x, PAD,
                   label, r["image"], score, r.get("text", ""), BG_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_t2i(
    query_idx: int,
    topk_idx: list[int],
    topk_scores: list[float],
    df: pd.DataFrame,
    out_path: Path,
):
    """Text-to-Image: クエリテキスト（＋対応画像）→ 上位k件画像"""
    k = len(topk_idx)
    total_w = CELL_W * (k + 1) + PAD * (k + 2)
    canvas = Image.new("RGB", (total_w, CELL_H + PAD * 2), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    row = df.iloc[query_idx]
    # クエリセル（対応画像＋テキスト）
    _draw_cell(draw, canvas, PAD, PAD,
               "QUERY (text)", row["image"], None, row.get("text", ""), QUERY_COLOR)

    # 結果セル
    for rank, (idx, score) in enumerate(zip(topk_idx, topk_scores)):
        r = df.iloc[idx]
        x = PAD + CELL_W * (rank + 1) + PAD * (rank + 1)
        label = f"k{rank+1}: {score:.3f}"
        _draw_cell(draw, canvas, x, PAD,
                   label, r["image"], score, r.get("text", ""), BG_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_i2i(
    query_idx: int,
    topk_idx: list[int],
    topk_scores: list[float],
    df: pd.DataFrame,
    out_path: Path,
):
    """Image-to-Image: クエリ画像 → 上位k件画像"""
    k = len(topk_idx)
    total_w = CELL_W * (k + 1) + PAD * (k + 2)
    canvas = Image.new("RGB", (total_w, CELL_H + PAD * 2), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    row = df.iloc[query_idx]
    _draw_cell(draw, canvas, PAD, PAD,
               "QUERY", row["image"], None, row.get("text", ""), QUERY_COLOR)

    for rank, (idx, score) in enumerate(zip(topk_idx, topk_scores)):
        r = df.iloc[idx]
        x = PAD + CELL_W * (rank + 1) + PAD * (rank + 1)
        label = f"k{rank+1}: {score:.3f}"
        _draw_cell(draw, canvas, x, PAD,
                   label, r["image"], score, r.get("text", ""), BG_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ── 特徴抽出 ──

class ImagePathDataset(Dataset):
    def __init__(self, paths, preprocess):
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


@torch.no_grad()
def extract_image_features(model, preprocess, paths, batch_size=128):
    ds = ImagePathDataset(paths, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
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
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch).to(DEVICE)
        f = F.normalize(model.encode_text(tokens), dim=-1).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def find_best_checkpoint(ckpt_dir: str) -> str:
    ckpt_dir = Path(ckpt_dir)
    results_file = ckpt_dir / "results.jsonl"
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
    print(f"[CLIP] ベストエポック: epoch {best_epoch} (val_loss={best_loss:.4f})")
    return str(ckpt_path)


def run_qualitative(
    val_csv: str,
    out_dir: str,
    ckpt_path: str | None = None,
    ckpt_dir: str | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    n_queries: int = 5,
    topk: int = 5,
    seed: int = 42,
    hits_only: bool = False,
):
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

    model = model.to(DEVICE).eval()

    df = pd.read_csv(val_csv)
    paths = df["image"].tolist()
    texts = df["text"].fillna("").tolist()
    N = len(df)
    print(f"データ件数: {N}")

    img_feats  = extract_image_features(model, preprocess, paths)
    text_feats = extract_text_features(model, tokenizer, texts)

    # 類似度行列
    img_t  = torch.from_numpy(img_feats).to(DEVICE)
    txt_t  = torch.from_numpy(text_feats).to(DEVICE)

    sim_i2t = (img_t  @ txt_t.T).cpu().numpy()   # (N, N)
    sim_t2i = (txt_t  @ img_t.T).cpu().numpy()   # (N, N)
    sim_i2i = (img_t  @ img_t.T).cpu().numpy()   # (N, N)

    # クエリインデックスをサンプリング
    rng = np.random.default_rng(seed)
    if hits_only:
        # T2I で正解（i番目のクエリの正解はi番目の画像）がR@topk内に入っているものを選ぶ
        hit_indices = []
        for i in range(N):
            row_scores = sim_t2i[i].copy()
            topk_idx_check = np.argsort(row_scores)[::-1][:topk]
            if i in topk_idx_check:
                hit_indices.append(i)
        print(f"[hits_only] T2I R@{topk} hit件数: {len(hit_indices)} / {N}")
        if len(hit_indices) < n_queries:
            print(f"  ※ hit件数 ({len(hit_indices)}) < n_queries ({n_queries})、全件使用")
            query_indices = hit_indices
        else:
            query_indices = rng.choice(hit_indices, size=n_queries, replace=False).tolist()
    else:
        query_indices = rng.choice(N, size=n_queries, replace=False).tolist()

    out_path = Path(out_dir)

    # ── I2T ──
    print("I2T 可視化...")
    for qi in query_indices:
        row_scores = sim_i2t[qi].copy()
        sorted_idx = np.argsort(row_scores)[::-1][:topk]
        sorted_scores = row_scores[sorted_idx].tolist()
        render_i2t(qi, sorted_idx.tolist(), sorted_scores, df,
                   out_path / "I2T" / f"query_{qi:04d}.png")

    # ── T2I ──
    print("T2I 可視化...")
    for qi in query_indices:
        row_scores = sim_t2i[qi].copy()
        sorted_idx = np.argsort(row_scores)[::-1][:topk]
        sorted_scores = row_scores[sorted_idx].tolist()
        render_t2i(qi, sorted_idx.tolist(), sorted_scores, df,
                   out_path / "T2I" / f"query_{qi:04d}.png")

    # ── I2I（自身除外）──
    print("I2I 可視化...")
    for qi in query_indices:
        row_scores = sim_i2i[qi].copy()
        row_scores[qi] = -np.inf  # 自身除外
        sorted_idx = np.argsort(row_scores)[::-1][:topk]
        sorted_scores = row_scores[sorted_idx].tolist()
        render_i2i(qi, sorted_idx.tolist(), sorted_scores, df,
                   out_path / "I2I" / f"query_{qi:04d}.png")

    print(f"\n保存完了: {out_path}")
    print(f"  {out_path}/I2T/  ({n_queries}件)")
    print(f"  {out_path}/T2I/  ({n_queries}件)")
    print(f"  {out_path}/I2I/  ({n_queries}件)")


def main():
    parser = argparse.ArgumentParser(description="CLIP 検索定性評価")
    parser.add_argument("--val_csv",     required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--ckpt",        default=None)
    parser.add_argument("--ckpt_dir",    default=None)
    parser.add_argument("--model",       default="ViT-B-32")
    parser.add_argument("--pretrained",  default="laion2b_s34b_b79k")
    parser.add_argument("--n_queries",   type=int, default=5)
    parser.add_argument("--topk",        type=int, default=5)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--hits_only",   action="store_true",
                        help="T2I R@topk 内に正解が含まれるクエリのみ使用")
    args = parser.parse_args()

    run_qualitative(
        val_csv=args.val_csv,
        out_dir=args.out_dir,
        ckpt_path=args.ckpt,
        ckpt_dir=args.ckpt_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        n_queries=args.n_queries,
        topk=args.topk,
        seed=args.seed,
        hits_only=args.hits_only,
    )


if __name__ == "__main__":
    main()
