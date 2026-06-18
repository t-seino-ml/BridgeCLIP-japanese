# -*- coding: utf-8 -*-
"""
Base CLIP ゼロショット分類

open_clip の ViT-B-32 (LAION2B 事前学習済み、ファインチューニングなし) を使い、
画像埋め込みとクラステキスト埋め込みのコサイン類似度で分類する。

各クラスに複数のテキストテンプレートを用意し、平均埋め込みで比較する
（ensemble-of-prompts 方式、OpenAI CLIPと同様）。

使い方:
    python -m classification.models.clip_zeroshot \
        --csv classification/results/labeled_val.csv \
        --out classification/results/clip_zeroshot_preds.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from classification.data.label_definitions import ALL_LABEL_SETS
from classification.prompts import CLIP_TEMPLATES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _find_best_checkpoint(ckpt_dir: str) -> str:
    """results.jsonl を val_loss 昇順に並べ、実在する最初の ckpt を返す。"""
    p = Path(ckpt_dir)
    results_file = p / "results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(f"results.jsonl が見つかりません: {results_file}")
    entries: list[tuple[float, int]] = []
    with open(results_file) as f:
        for line in f:
            d = json.loads(line)
            loss = d.get("clip_val_loss", float("inf"))
            epoch = d.get("epoch", -1)
            entries.append((loss, epoch))
    entries.sort()
    for loss, epoch in entries:
        candidate = p / f"epoch_{epoch}.pt"
        if candidate.exists():
            print(f"[CLIP] 採用: epoch {epoch} (val_loss={loss:.4f})")
            return str(candidate)
    raise FileNotFoundError(f"実在する ckpt が一つもありません: {ckpt_dir}")


class CLIPZeroShotClassifier:
    """
    CLIP ゼロショット分類器。

    Args:
        model_name:  open_clip のモデル名（デフォルト: ViT-B-32）
        pretrained:  open_clip の事前学習重み名（デフォルト: laion2b_s34b_b79k）
        threshold:   マルチラベル出力の閾値（0〜1）
        device:      "cuda" / "cpu"
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        threshold: float = 0.3,
        device: str = DEVICE,
        ckpt_path: str | None = None,
        ckpt_dir: str | None = None,
    ):
        self.threshold = threshold
        self.device = device

        # ckpt_dir 指定時は results.jsonl から best (val_loss 最小・実在する ckpt) を選択
        if ckpt_dir is not None:
            ckpt_path = _find_best_checkpoint(ckpt_dir)

        print(f"[CLIP] モデルを読み込み中: {model_name} ({pretrained})")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        if ckpt_path is not None:
            print(f"[CLIP] FT チェックポイントを読み込み中: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)
            print("[CLIP] FT チェックポイント読み込み完了")

        self.model.eval()

        # クラステキスト埋め込みを事前計算
        self._text_embeds: dict[str, torch.Tensor] = {}
        self._build_text_embeddings()

    @torch.no_grad()
    def _build_text_embeddings(self) -> None:
        """全カテゴリのクラステキスト埋め込みを事前計算する。"""
        for cat, label_list in ALL_LABEL_SETS.items():
            templates = CLIP_TEMPLATES.get(cat, {})
            class_embeds = []
            for label in label_list:
                texts = templates.get(label, [f"道路橋の{label}"])
                tokens = self.tokenizer(texts).to(self.device)
                embeds = self.model.encode_text(tokens)          # (T, D)
                embeds = F.normalize(embeds, dim=-1)
                class_embed = embeds.mean(dim=0)                 # テンプレート平均
                class_embed = F.normalize(class_embed, dim=-1)
                class_embeds.append(class_embed)
            # shape: (num_classes, D)
            self._text_embeds[cat] = torch.stack(class_embeds, dim=0)
        print("[CLIP] テキスト埋め込みの事前計算完了")

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> dict[str, list[str]]:
        """
        1枚の画像を受け取り、4カテゴリの予測ラベルリストを返す。

        Returns:
            {
              "kenzenudo":   ["Ⅱ"],
              "taisaku":     ["C1"],
              "damage_type": ["ひびわれ"],
              "damage_loc":  ["主桁", "床版"],
            }
        """
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        img_embed = self.model.encode_image(img_tensor)          # (1, D)
        img_embed = F.normalize(img_embed, dim=-1)

        predictions: dict[str, list[str]] = {}
        for cat, label_list in ALL_LABEL_SETS.items():
            text_embeds = self._text_embeds[cat]                 # (C, D)
            sims = (img_embed @ text_embeds.T).squeeze(0)        # (C,)
            probs = sims.softmax(dim=-1).cpu().numpy()

            if cat in ("kenzenudo", "taisaku"):
                # 単一ラベル: 最大スコアを選択
                best_idx = int(np.argmax(probs))
                predictions[cat] = [label_list[best_idx]]
            else:
                # マルチラベル: 閾値を超えたものすべて選択（最低1つ）
                selected = [label_list[i] for i, p in enumerate(probs) if p >= self.threshold]
                if not selected:
                    selected = [label_list[int(np.argmax(probs))]]
                predictions[cat] = selected

        return predictions

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, list[str]]]:
        """複数枚の画像をバッチ処理して予測する。"""
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        img_embeds = self.model.encode_image(tensors)            # (B, D)
        img_embeds = F.normalize(img_embeds, dim=-1)

        batch_predictions: list[dict[str, list[str]]] = [
            {cat: [] for cat in ALL_LABEL_SETS} for _ in images
        ]
        for cat, label_list in ALL_LABEL_SETS.items():
            text_embeds = self._text_embeds[cat]                 # (C, D)
            sims = img_embeds @ text_embeds.T                    # (B, C)
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
    ckpt_path: str | None = None,
    ckpt_dir: str | None = None,
    image_root: str | None = None,
) -> None:
    """
    ラベル付きCSVの全画像に対してゼロショット分類を実行し、結果をCSVに保存する。
    `ckpt_path` / `ckpt_dir` を指定すると CLIP-FT ckpt を読み込んで推論する。
    `image_root` を指定すると CSV の image 列のディレクトリ部分を差し替える。
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if image_root is not None:
        df = df.copy()
        df["image"] = df["image"].map(lambda p: str(Path(image_root) / Path(p).name))

    classifier = CLIPZeroShotClassifier(
        model_name=model_name, pretrained=pretrained,
        threshold=threshold,
        ckpt_path=ckpt_path, ckpt_dir=ckpt_dir,
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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CLIP ゼロショット推論"):
        try:
            img = Image.open(str(row["image"])).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(
                f"画像が読み込めません: {row['image']} ({e}) — `--image_root` か CSV パスを確認してください。"
            ) from e
        images.append(img)
        img_paths.append(str(row["image"]))

        if len(images) >= batch_size:
            _flush(img_paths, images)
            images, img_paths = [], []

    if images:
        _flush(img_paths, images)

    pred_df = pd.DataFrame(all_preds)
    # 正解ラベルも結合
    result_df = df.merge(pred_df, on="image", how="left")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"結果保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="CLIP ゼロショット分類（FT ckpt 対応）")
    parser.add_argument("--csv",        required=True, help="ラベル付きCSVのパス")
    parser.add_argument("--out",        required=True, help="予測結果の出力CSVパス")
    parser.add_argument("--model",      default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold",  type=float, default=0.3)
    parser.add_argument("--ckpt",       default=None, help="CLIP-FT ckpt パス（省略時はベースCLIP）")
    parser.add_argument("--ckpt_dir",   default=None, help="ckpt ディレクトリ。results.jsonl から best を自動採用")
    parser.add_argument("--image_root", default=None, help="CSV の image 列のディレクトリ部分を差し替え")
    args = parser.parse_args()

    run_on_csv(
        csv_path=args.csv,
        out_path=args.out,
        model_name=args.model,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        threshold=args.threshold,
        ckpt_path=args.ckpt,
        ckpt_dir=args.ckpt_dir,
        image_root=args.image_root,
    )


if __name__ == "__main__":
    main()
