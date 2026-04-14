# -*- coding: utf-8 -*-
"""
定性評価：ベースCLIP vs ファインチューニング済みCLIP の比較可視化

同一クエリに対して両モデルの検索結果を上下に並べて出力する。

使い方:
    python -m classification.qualitative_comparison \
        --val_csv   classification/results/unified_val_user.csv \
        --ckpt_dir  logs_classification/bridgeclip_vitb32_unified/checkpoints \
        --out_dir   classification/results/qualitative/comparison \
        --n_queries 5 \
        --topk 5 \
        --hits_only
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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── レイアウト定数 ──
IMG_W        = 190
IMG_H        = 150
TEXT_H       = 80
SCORE_H      = 20
HEADER_H     = 26
LABEL_W      = 120   # 左端のモデルラベル列幅
CELL_W       = IMG_W
CELL_H       = HEADER_H + IMG_H + SCORE_H + TEXT_H
PAD          = 6
BG_COLOR     = (255, 255, 255)
QUERY_BG     = (255, 235, 205)
BASE_BG      = (235, 235, 255)
FT_BG        = (220, 255, 220)
HEADER_BG    = (200, 210, 240)
BORDER_COLOR = (150, 150, 150)
TEXT_COLOR   = (20, 20, 20)
SCORE_FG     = (0, 110, 0)
HIT_COLOR    = (255, 80, 80)   # 正解ヒット時の枠色


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


FONT_SM = _load_font(11)
FONT_MD = _load_font(13)
FONT_LG = _load_font(14)
FONT_XL = _load_font(16)


def _wrap_text(text: str, font, max_width: int) -> list[str]:
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    lines, current = [], ""
    for ch in text:
        test = current + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width:
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
    x: int, y: int,
    header: str,
    img_path: str | None,
    score: float | None,
    caption: str,
    bg_color: tuple,
    is_hit: bool = False,
):
    w, h = CELL_W, CELL_H
    border = HIT_COLOR if is_hit else BORDER_COLOR
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=bg_color, outline=border, width=2 if is_hit else 1)

    # ヘッダ
    draw.rectangle([x, y, x + w - 1, y + HEADER_H - 1], fill=HEADER_BG)
    draw.text((x + PAD, y + 4), header, font=FONT_MD, fill=TEXT_COLOR)

    # 画像
    img_y = y + HEADER_H
    if img_path and Path(img_path).exists():
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_W, IMG_H), Image.LANCZOS)
            canvas.paste(img, (x, img_y))
        except Exception:
            draw.rectangle([x, img_y, x + IMG_W - 1, img_y + IMG_H - 1], fill=(200, 200, 200))
    else:
        draw.rectangle([x, img_y, x + IMG_W - 1, img_y + IMG_H - 1], fill=(220, 220, 220))
        draw.text((x + 4, img_y + IMG_H // 2 - 8), "画像なし", font=FONT_SM, fill=(100,))

    # スコア
    score_y = img_y + IMG_H
    if score is not None:
        draw.rectangle([x, score_y, x + w - 1, score_y + SCORE_H - 1], fill=(230, 255, 230))
        draw.text((x + PAD, score_y + 3), f"cos={score:.4f}", font=FONT_MD, fill=SCORE_FG)

    # テキスト
    text_y = score_y + SCORE_H
    lines = _wrap_text(caption, FONT_SM, w - PAD * 2)
    ty = text_y + 3
    for line in lines[:4]:
        draw.text((x + PAD, ty), line, font=FONT_SM, fill=TEXT_COLOR)
        ty += 15
        if ty + 15 > y + h:
            break


def _draw_row_label(draw, canvas, x, y, label: str, bg_color: tuple):
    """左端のモデル名ラベルを描画する。"""
    draw.rectangle([x, y, x + LABEL_W - 1, y + CELL_H - 1], fill=bg_color, outline=BORDER_COLOR)
    # 縦中央に文字を配置
    lines = _wrap_text(label, FONT_LG, LABEL_W - PAD * 2)
    total_h = len(lines) * 18
    ty = y + (CELL_H - total_h) // 2
    for line in lines:
        draw.text((x + PAD, ty), line, font=FONT_LG, fill=TEXT_COLOR)
        ty += 18


def render_comparison(
    query_idx: int,
    task: str,          # "I2T" / "T2I" / "I2I"
    base_topk_idx: list[int],
    base_topk_scores: list[float],
    ft_topk_idx: list[int],
    ft_topk_scores: list[float],
    df: pd.DataFrame,
    out_path: Path,
):
    """
    1枚の比較画像を生成する。
    レイアウト:
        列: [ラベル列] [クエリ] [k1] [k2] ... [k_topk]
        行: [ベースCLIP] [ファインチューニング済み]
    """
    k = len(base_topk_idx)
    n_cols = k + 1   # クエリ + topk
    total_w = LABEL_W + CELL_W * n_cols + PAD * (n_cols + 2)
    total_h = CELL_H * 2 + PAD * 3 + 30  # 2行 + タイトル行

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # タイトル
    title = f"{task}  query_idx={query_idx}"
    draw.text((PAD, PAD), title, font=FONT_XL, fill=TEXT_COLOR)

    row = df.iloc[query_idx]
    query_img  = row["image"]
    query_text = row.get("text", "")

    for row_i, (topk_idx, topk_scores, label, row_bg) in enumerate([
        (base_topk_idx,  base_topk_scores,  "Base\nCLIP",          BASE_BG),
        (ft_topk_idx,    ft_topk_scores,    "Fine-\ntuned\nCLIP",  FT_BG),
    ]):
        y = 30 + PAD + row_i * (CELL_H + PAD)

        # 行ラベル
        _draw_row_label(draw, canvas, PAD, y, label, row_bg)

        # クエリセル
        qx = PAD + LABEL_W + PAD
        if task == "T2I":
            # T2I: テキストがクエリ → 対応画像も表示
            _draw_cell(draw, canvas, qx, y, "QUERY", query_img, None, query_text, QUERY_BG)
        elif task in ("I2T", "I2I"):
            _draw_cell(draw, canvas, qx, y, "QUERY", query_img, None, query_text, QUERY_BG)

        # 結果セル
        for rank, (idx, score) in enumerate(zip(topk_idx, topk_scores)):
            r = df.iloc[idx]
            cx = qx + CELL_W * (rank + 1) + PAD * (rank + 1)
            is_hit = (idx == query_idx)  # 正解がヒットしているか
            hdr = f"k{rank+1}: {score:.3f}"
            if task == "I2T":
                # 画像→テキスト: 結果はテキスト側（対応画像を表示）
                _draw_cell(draw, canvas, cx, y, hdr, r["image"], score, r.get("text", ""), row_bg, is_hit)
            else:
                # T2I / I2I: 結果は画像
                _draw_cell(draw, canvas, cx, y, hdr, r["image"], score, r.get("text", ""), row_bg, is_hit)

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
        tokens = tokenizer(texts[i:i + batch_size]).to(DEVICE)
        f = F.normalize(model.encode_text(tokens), dim=-1).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def find_best_checkpoint(ckpt_dir: str) -> str:
    ckpt_dir = Path(ckpt_dir)
    best_epoch, best_loss = -1, float("inf")
    with open(ckpt_dir / "results.jsonl") as f:
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


def load_model(model_name, pretrained, ckpt_path=None):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=DEVICE
    )
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print(f"  チェックポイント読み込み完了: {ckpt_path}")
    return model.to(DEVICE).eval(), preprocess


def compute_sims(img_feats, text_feats):
    it = torch.from_numpy(img_feats).to(DEVICE)
    tt = torch.from_numpy(text_feats).to(DEVICE)
    sim_i2t = (it @ tt.T).cpu().numpy()
    sim_t2i = (tt @ it.T).cpu().numpy()
    sim_i2i = (it @ it.T).cpu().numpy()
    return sim_i2t, sim_t2i, sim_i2i


def run_comparison(
    val_csv: str,
    out_dir: str,
    ckpt_dir: str | None = None,
    ckpt_path: str | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    n_queries: int = 5,
    topk: int = 5,
    seed: int = 42,
    hits_only: bool = False,
):
    if ckpt_dir is not None:
        ckpt_path = find_best_checkpoint(ckpt_dir)

    df = pd.read_csv(val_csv)
    paths = df["image"].tolist()
    texts = df["text"].fillna("").tolist()
    N = len(df)
    print(f"データ件数: {N}")

    # ── ベースCLIP ──
    print("\n[1/2] ベースCLIPを読み込み中...")
    tokenizer = open_clip.get_tokenizer(model_name)
    base_model, base_prep = load_model(model_name, pretrained, ckpt_path=None)
    base_img_feats  = extract_image_features(base_model, base_prep, paths)
    base_txt_feats  = extract_text_features(base_model, tokenizer, texts)
    base_i2t, base_t2i, base_i2i = compute_sims(base_img_feats, base_txt_feats)
    del base_model

    # ── ファインチューニング済みCLIP ──
    print("\n[2/2] ファインチューニング済みCLIPを読み込み中...")
    ft_model, ft_prep = load_model(model_name, pretrained, ckpt_path=ckpt_path)
    ft_img_feats  = extract_image_features(ft_model, ft_prep, paths)
    ft_txt_feats  = extract_text_features(ft_model, tokenizer, texts)
    ft_i2t, ft_t2i, ft_i2i = compute_sims(ft_img_feats, ft_txt_feats)
    del ft_model

    # ── クエリ選択 ──
    rng = np.random.default_rng(seed)
    if hits_only:
        # ファインチューニング済みの T2I で R@topk にヒットするクエリ
        hit_indices = [
            i for i in range(N)
            if i in np.argsort(ft_t2i[i])[::-1][:topk]
        ]
        print(f"[hits_only] Fine-tuned T2I R@{topk} hit: {len(hit_indices)} / {N} 件")
        pool = hit_indices if len(hit_indices) >= n_queries else list(range(N))
        query_indices = rng.choice(pool, size=min(n_queries, len(pool)), replace=False).tolist()
    else:
        query_indices = rng.choice(N, size=n_queries, replace=False).tolist()

    print(f"クエリインデックス: {query_indices}")

    out_path = Path(out_dir)

    # ── 描画 ──
    for task, base_sim, ft_sim in [
        ("I2T", base_i2t, ft_i2t),
        ("T2I", base_t2i, ft_t2i),
        ("I2I", base_i2i, ft_i2i),
    ]:
        print(f"\n{task} 可視化中...")
        for qi in query_indices:
            # ベースCLIP
            base_row = base_sim[qi].copy()
            if task == "I2I":
                base_row[qi] = -np.inf
            b_idx   = np.argsort(base_row)[::-1][:topk].tolist()
            b_scores = base_row[np.array(b_idx)].tolist()

            # ファインチューニング済み
            ft_row = ft_sim[qi].copy()
            if task == "I2I":
                ft_row[qi] = -np.inf
            f_idx   = np.argsort(ft_row)[::-1][:topk].tolist()
            f_scores = ft_row[np.array(f_idx)].tolist()

            render_comparison(
                query_idx=qi,
                task=task,
                base_topk_idx=b_idx,
                base_topk_scores=b_scores,
                ft_topk_idx=f_idx,
                ft_topk_scores=f_scores,
                df=df,
                out_path=out_path / task / f"query_{qi:04d}.png",
            )

    print(f"\n保存完了: {out_path}")
    for task in ["I2T", "T2I", "I2I"]:
        print(f"  {out_path}/{task}/  ({len(query_indices)}件)")


def main():
    parser = argparse.ArgumentParser(description="CLIP ベース vs ファインチューニング比較可視化")
    parser.add_argument("--val_csv",    required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--ckpt",       default=None,  help="ファインチューニング済みcheckpoint")
    parser.add_argument("--ckpt_dir",   default=None,  help="checkpoints/ディレクトリ（自動選択）")
    parser.add_argument("--model",      default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--n_queries",  type=int, default=5)
    parser.add_argument("--topk",       type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--hits_only",  action="store_true",
                        help="Fine-tuned T2I R@topk にヒットするクエリのみ使用")
    args = parser.parse_args()

    run_comparison(
        val_csv=args.val_csv,
        out_dir=args.out_dir,
        ckpt_dir=args.ckpt_dir,
        ckpt_path=args.ckpt,
        model_name=args.model,
        pretrained=args.pretrained,
        n_queries=args.n_queries,
        topk=args.topk,
        seed=args.seed,
        hits_only=args.hits_only,
    )


if __name__ == "__main__":
    main()
