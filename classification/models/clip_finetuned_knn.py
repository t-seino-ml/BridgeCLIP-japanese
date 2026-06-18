# -*- coding: utf-8 -*-
"""
ファインチューニング済みCLIP による k近傍多数決分類

学習画像の特徴ベクトルをデータベースとして構築し、
検証画像の特徴ベクトルに対してk近傍探索を行い、
近傍画像のラベルから多数決でクラスを決定する。

使い方:
    # ファインチューニング済みCLIP
    python -m classification.models.clip_finetuned_knn \
        --train_csv classification/results/unified_train_local.csv \
        --val_csv   classification/results/unified_val_local.csv \
        --ckpt      logs_classification/bridgeclip_vitb32_unified/checkpoints/epoch_10.pt \
        --out       classification/results/clip_finetuned_knn_preds.csv \
        --k 10

    # ベースCLIP（チェックポイントなし）
    python -m classification.models.clip_finetuned_knn \
        --train_csv classification/results/unified_train_local.csv \
        --val_csv   classification/results/unified_val_local.csv \
        --out       classification/results/clip_base_knn_preds.csv \
        --k 10
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

from classification.data.labels import ALL_LABEL_SETS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ────────────────────────────── Dataset ──────────────────────────────────────

class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        return self.preprocess(img), idx


# ────────────────────────────── 特徴抽出 ─────────────────────────────────────

@torch.no_grad()
def extract_features(
    model,
    preprocess,
    paths: list[str],
    batch_size: int = 128,
    num_workers: int = 4,
    desc: str = "特徴抽出",
) -> np.ndarray:
    """画像パスリストの特徴ベクトルを抽出して返す（shape: [N, D]）。"""
    ds = ImagePathDataset(paths, preprocess)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    all_feats = np.zeros((len(paths), model.visual.output_dim), dtype=np.float32)

    for imgs, indices in tqdm(dl, desc=desc):
        imgs = imgs.to(DEVICE)
        feats = model.encode_image(imgs)
        feats = F.normalize(feats, dim=-1).cpu().numpy()
        for feat, idx in zip(feats, indices.numpy()):
            all_feats[idx] = feat

    return all_feats


# ────────────────────────────── k近傍多数決 ──────────────────────────────────

def majority_vote(labels: list[str], k: int) -> str:
    """
    k件のラベル文字列（"|"区切りマルチラベル）から多数決で予測ラベルを決定する。
    単一ラベルカテゴリ: 最多出現ラベルを1つ返す。
    マルチラベルカテゴリ: 過半数以上出現したラベルをすべて返す（最低1つ）。
    """
    from collections import Counter
    counts: Counter = Counter()
    for label_str in labels:
        for token in str(label_str or "").split("|"):
            token = token.strip()
            if token:
                counts[token] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def majority_vote_multi(labels: list[str], k: int) -> str:
    """マルチラベル用多数決: k件中過半数以上出現したラベルを返す（最低1つ）。"""
    from collections import Counter
    counts: Counter = Counter()
    for label_str in labels:
        for token in str(label_str or "").split("|"):
            token = token.strip()
            if token:
                counts[token] += 1
    if not counts:
        return ""
    threshold = k / 2
    selected = [lbl for lbl, cnt in counts.items() if cnt >= threshold]
    if not selected:
        selected = [counts.most_common(1)[0][0]]
    return "|".join(selected)


# ────────────────────────────── ベストチェックポイント選択 ───────────────────

def find_best_checkpoint(ckpt_dir: str) -> str:
    """
    checkpoints/ ディレクトリ内の results.jsonl を読み、
    clip_val_loss が最小のエポックのチェックポイントパスを返す。
    """
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


# ────────────────────────────── メイン処理 ───────────────────────────────────

def run_knn(
    train_csv: str,
    val_csv: str,
    out_path: str,
    ckpt_path: str | None = None,
    ckpt_dir: str | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    k: int = 10,
    batch_size: int = 128,
    num_workers: int = 4,
) -> None:
    # ── ckpt_dir からベストチェックポイントを自動選択 ──
    if ckpt_dir is not None:
        ckpt_path = find_best_checkpoint(ckpt_dir)

    # ── モデル読み込み ──
    print(f"[CLIP] モデルを読み込み中: {model_name}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=DEVICE
    )

    if ckpt_path is not None:
        print(f"[CLIP] チェックポイントを読み込み中: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        # open_clip_train は "module." プレフィックスが付く場合がある
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        print("[CLIP] チェックポイント読み込み完了")

    model = model.to(DEVICE).eval()

    # ── データ読み込み ──
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    # filter_valid=True 相当（4カテゴリすべて有効な行のみ学習DBに使用）
    valid_cols = [f"{c}_valid" for c in ALL_LABEL_SETS if f"{c}_valid" in train_df.columns]
    if valid_cols:
        mask = train_df[valid_cols].all(axis=1)
        train_df = train_df[mask].reset_index(drop=True)
        print(f"学習DB: {len(train_df)} 件（有効行のみ）")
    else:
        print(f"学習DB: {len(train_df)} 件")

    print(f"検証: {len(val_df)} 件")

    # ── 特徴抽出 ──
    train_feats = extract_features(
        model, preprocess, train_df["image"].tolist(),
        batch_size=batch_size, num_workers=num_workers, desc="学習DB 特徴抽出",
    )
    val_feats = extract_features(
        model, preprocess, val_df["image"].tolist(),
        batch_size=batch_size, num_workers=num_workers, desc="検証 特徴抽出",
    )

    # ── k近傍探索（コサイン類似度） ──
    print(f"k近傍探索中（k={k}）...")
    train_feats_t = torch.from_numpy(train_feats).to(DEVICE)  # (N_train, D)
    val_feats_t   = torch.from_numpy(val_feats).to(DEVICE)    # (N_val, D)

    # バッチ単位でコサイン類似度を計算
    chunk = 256
    topk_indices = []
    for i in range(0, len(val_feats_t), chunk):
        sims = val_feats_t[i:i+chunk] @ train_feats_t.T  # (chunk, N_train)
        _, idx = sims.topk(k, dim=-1)                    # (chunk, k)
        topk_indices.append(idx.cpu().numpy())
    topk_indices = np.concatenate(topk_indices, axis=0)  # (N_val, k)

    # ── 多数決ラベル決定 ──
    rows = []
    for i, row in enumerate(val_df.itertuples()):
        nn_idx = topk_indices[i]  # shape: (k,)
        nn_rows = train_df.iloc[nn_idx]

        pred_row = {"image": row.image}
        for cat in ALL_LABEL_SETS:
            if cat not in train_df.columns:
                pred_row[f"pred_{cat}"] = ""
                continue
            nn_labels = nn_rows[cat].fillna("").tolist()
            if cat in ("kenzenudo", "taisaku"):
                pred_row[f"pred_{cat}"] = majority_vote(nn_labels, k)
            else:
                pred_row[f"pred_{cat}"] = majority_vote_multi(nn_labels, k)
        rows.append(pred_row)

    pred_df = pd.DataFrame(rows)
    result_df = val_df.merge(pred_df, on="image", how="left")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"結果保存: {out_path}")


# ────────────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLIP k近傍多数決分類")
    parser.add_argument("--train_csv",   required=True, help="学習CSVパス（k近傍DBとして使用）")
    parser.add_argument("--val_csv",     required=True, help="検証CSVパス")
    parser.add_argument("--out",         required=True, help="予測結果の出力CSVパス")
    parser.add_argument("--ckpt",        default=None,  help="ファインチューニング済みチェックポイント（省略時はベースCLIP）")
    parser.add_argument("--ckpt_dir",    default=None,  help="checkpoints/ ディレクトリ。指定するとresults.jsonlからベストエポックを自動選択")
    parser.add_argument("--model",       default="ViT-B-32")
    parser.add_argument("--pretrained",  default="laion2b_s34b_b79k")
    parser.add_argument("--k",           type=int, default=10, help="近傍数")
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    run_knn(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        out_path=args.out,
        ckpt_path=args.ckpt,
        ckpt_dir=args.ckpt_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        k=args.k,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
