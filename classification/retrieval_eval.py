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
    """
    results.jsonl を val_loss 昇順に並べ、**実在する最初の ckpt** を返す。
    （元の「最良エポックの ckpt」が削除されている場合は次点へフォールバック）
    """
    ckpt_dir = Path(ckpt_dir)
    results_file = ckpt_dir / "results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(f"results.jsonl が見つかりません: {results_file}")
    entries: list[tuple[float, int]] = []
    with open(results_file) as f:
        for line in f:
            d = json.loads(line)
            loss = d.get("clip_val_loss", float("inf"))
            epoch = d.get("epoch", -1)
            entries.append((loss, epoch))
    entries.sort()  # val_loss 昇順
    for loss, epoch in entries:
        candidate = ckpt_dir / f"epoch_{epoch}.pt"
        if candidate.exists():
            best_loss_str = "(best by val_loss)" if (loss, epoch) == entries[0] else \
                            f"(fallback: best ckpt epoch_{entries[0][1]}.pt missing)"
            print(f"[CLIP] 採用: epoch {epoch} (val_loss={loss:.4f}) {best_loss_str}")
            return str(candidate)
    raise FileNotFoundError(
        f"results.jsonl にある epoch のうち実在する ckpt が一つもありません: {ckpt_dir}"
    )


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


def _topk_indices(sim_matrix: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    """各行（クエリ）について類似度上位 k の gallery 列インデックスを返す。"""
    sim = sim_matrix.copy()
    if exclude_self:
        N = sim.shape[0]
        sim[np.arange(N), np.arange(N)] = -np.inf
    # argpartition でも可だが安定化のため argsort
    order = np.argsort(sim, axis=1)[:, ::-1]
    return order[:, :k]


def _full_ranking(sim_matrix: np.ndarray, exclude_self: bool) -> np.ndarray:
    """各クエリの全 gallery を類似度降順に並べたインデックス（N_query, N_gallery）。"""
    sim = sim_matrix.copy()
    if exclude_self:
        N = sim.shape[0]
        sim[np.arange(N), np.arange(N)] = -np.inf
    return np.argsort(sim, axis=1)[:, ::-1]


def attribute_match_at_k(
    sim_matrix: np.ndarray,
    query_attrs: dict[str, list[str]],
    gallery_attrs: dict[str, list[str]],
    ks=(1, 5, 10),
    exclude_self: bool = False,
) -> dict:
    """
    属性一致率@k: クエリ i の top-k gallery のうち、当該属性が一致する件数の割合。
    マルチラベル属性（"|"区切り）は集合の交差で一致判定する。
    """
    N = sim_matrix.shape[0]
    out: dict[str, dict[str, float]] = {}

    def to_set(v: str) -> set:
        return {t.strip() for t in str(v or "").split("|") if t.strip()}

    for attr, q_labels in query_attrs.items():
        g_labels = gallery_attrs[attr]
        q_sets = [to_set(v) for v in q_labels]
        g_sets = [to_set(v) for v in g_labels]
        out[attr] = {}
        for k in ks:
            topk = _topk_indices(sim_matrix, k, exclude_self)
            per_query = []
            for i in range(N):
                if not q_sets[i]:
                    continue
                hits = sum(1 for j in topk[i] if q_sets[i] & g_sets[j])
                per_query.append(hits / k)
            out[attr][f"AttrMatch@{k}"] = float(np.mean(per_query)) if per_query else 0.0
    return out


def attribute_map_ndcg(
    sim_matrix: np.ndarray,
    query_attrs: dict[str, list[str]],
    gallery_attrs: dict[str, list[str]],
    exclude_self: bool = False,
    ndcg_k: int = 10,
) -> dict:
    """
    属性ベースの mAP（AveP の平均）と NDCG@k。
    正解 = クエリと当該属性で1つでもラベルが一致する gallery アイテム。
    """
    N = sim_matrix.shape[0]
    ranking = _full_ranking(sim_matrix, exclude_self)
    n_gallery = ranking.shape[1]

    def to_set(v: str) -> set:
        return {t.strip() for t in str(v or "").split("|") if t.strip()}

    out: dict[str, dict[str, float]] = {}
    for attr, q_labels in query_attrs.items():
        g_labels = gallery_attrs[attr]
        q_sets = [to_set(v) for v in q_labels]
        g_sets = [to_set(v) for v in g_labels]

        # 正解マスク（i行 j列 = クエリiにとってgallery jが正解か）を行ごとに作る
        ap_list, ndcg_list = [], []
        ideal_dcg_cache: dict[int, float] = {}
        for i in range(N):
            if not q_sets[i]:
                continue
            order = ranking[i]
            rels = np.array([1 if (q_sets[i] & g_sets[j]) else 0 for j in order], dtype=np.float32)
            total_rel = int(rels.sum())
            if total_rel == 0:
                ap_list.append(0.0)
                ndcg_list.append(0.0)
                continue

            # Average Precision
            cum_hits = np.cumsum(rels)
            precisions = cum_hits / (np.arange(n_gallery) + 1)
            ap = float((precisions * rels).sum() / total_rel)
            ap_list.append(ap)

            # NDCG@k
            k = min(ndcg_k, n_gallery)
            gains = rels[:k]
            discounts = 1.0 / np.log2(np.arange(2, k + 2))
            dcg = float((gains * discounts).sum())
            if total_rel not in ideal_dcg_cache:
                ideal_rels = np.ones(min(total_rel, k), dtype=np.float32)
                ideal_disc = 1.0 / np.log2(np.arange(2, ideal_rels.size + 2))
                ideal_dcg_cache[total_rel] = float((ideal_rels * ideal_disc).sum())
            idcg = ideal_dcg_cache[total_rel]
            ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

        out[attr] = {
            "mAP": float(np.mean(ap_list)) if ap_list else 0.0,
            f"NDCG@{ndcg_k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
            "n_query": int(len(ap_list)),
        }
    return out


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
    i2i = recall_at_k(sim_i2i, ks=ks, exclude_self=True)   # 自身を除外（ペア一致のみ）

    # ── 属性ベース指標 ──
    attr_cols = ["kenzenudo", "taisaku", "damage_type", "damage_loc"]
    attr_cols = [c for c in attr_cols if c in df.columns]
    attrs_all = {c: df[c].fillna("").astype(str).tolist() for c in attr_cols}

    attr_i2i_match = attribute_match_at_k(sim_i2i, attrs_all, attrs_all, ks=ks, exclude_self=True)
    attr_i2i_rank  = attribute_map_ndcg(  sim_i2i, attrs_all, attrs_all, exclude_self=True, ndcg_k=10)
    attr_t2i_match = attribute_match_at_k(sim_t2i, attrs_all, attrs_all, ks=ks, exclude_self=False)
    attr_t2i_rank  = attribute_map_ndcg(  sim_t2i, attrs_all, attrs_all, exclude_self=False, ndcg_k=10)

    result = {
        "ckpt": ckpt_path or "base",
        "val_csv": val_csv,
        "N": N,
        "i2t": i2t,
        "t2i": t2i,
        "i2i_pair": i2i,
        "i2i_attribute_match": attr_i2i_match,
        "i2i_attribute_rank":  attr_i2i_rank,
        "t2i_attribute_match": attr_t2i_match,
        "t2i_attribute_rank":  attr_t2i_rank,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"結果保存: {out_path}")

    print("\n=== Image-to-Text ===")
    for k_str, v in i2t.items(): print(f"  {k_str}: {v:.4f}")
    print("=== Text-to-Image ===")
    for k_str, v in t2i.items(): print(f"  {k_str}: {v:.4f}")
    print("=== Image-to-Image (pair, exclude self) ===")
    for k_str, v in i2i.items(): print(f"  {k_str}: {v:.4f}")

    print("\n=== I2I 属性一致率@k（自身除外） ===")
    for attr, mvals in attr_i2i_match.items():
        row = "  ".join(f"{k}={v:.4f}" for k, v in mvals.items())
        print(f"  [{attr:12s}] {row}")
    print("=== I2I 属性ベース mAP / NDCG@10 ===")
    for attr, mvals in attr_i2i_rank.items():
        print(f"  [{attr:12s}] mAP={mvals['mAP']:.4f}  NDCG@10={mvals['NDCG@10']:.4f}  n={mvals['n_query']}")
    print("\n=== T2I 属性一致率@k ===")
    for attr, mvals in attr_t2i_match.items():
        row = "  ".join(f"{k}={v:.4f}" for k, v in mvals.items())
        print(f"  [{attr:12s}] {row}")
    print("=== T2I 属性ベース mAP / NDCG@10 ===")
    for attr, mvals in attr_t2i_rank.items():
        print(f"  [{attr:12s}] mAP={mvals['mAP']:.4f}  NDCG@10={mvals['NDCG@10']:.4f}  n={mvals['n_query']}")


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
