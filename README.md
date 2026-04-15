# BridgeCLIP: Fine-tuned CLIP for Bridge Inspection Image Classification and Retrieval

A fine-tuned [OpenCLIP ViT-B/32](https://github.com/mlfoundations/open_clip) model for bridge inspection image classification and image-text retrieval, trained on the Japanese national road facility inspection database (xROAD).

## Overview

- **Proposed method**: CLIP fine-tuning + k-NN classification (k=10, cosine similarity)
- **Baselines**: ResNet50, ViT-B/16 (supervised), GPT-4o, Qwen3-VL, InternVL3, Llama3.2-Vision (zero-shot)
- **Tasks**: 4-category multi-label classification (Soundness / Countermeasure / Damage Type / Damage Location), Image-Text retrieval (I2T / T2I / I2I)

## Performance

### Classification (Exact Match Accuracy, k=10 k-NN)

| Category | Accuracy |
|---|---|
| Soundness | 0.7361 |
| Countermeasure | 0.6186 |
| Damage Type | 0.6209 |
| Damage Location | 0.5740 |
| **Mean** | **0.6374** |

### Retrieval (Recall@k)

| Task | R@1 | R@5 | R@10 |
|---|---|---|---|
| Image-to-Text | 0.0362 | 0.1437 | 0.2266 |
| Text-to-Image | 0.0493 | 0.1620 | 0.2437 |

## Model

Fine-tuned checkpoint is available on Hugging Face Hub:

> [**Seino404/bridge-inspection-clip**](https://huggingface.co/Seino404/bridge-inspection-clip)

## Dataset

Image-text pairs from the Japanese national road facility inspection database (xROAD).

| Split | Samples |
|---|---|
| Train | 130,930 |
| Test | 2,679 |
| k-NN DB (all 4 categories valid) | 90,987 |

## Citation

Coming soon (paper in preparation).

## License

MIT License

---

# 日本語版 / Japanese

# BridgeCLIP: 橋梁点検画像の分類・検索のためのCLIPファインチューニング

橋梁点検画像と変状所見テキストのペアデータを用いて [OpenCLIP (ViT-B/32)](https://github.com/mlfoundations/open_clip) をコントラスティブ学習でファインチューニングし、k近傍分類および画像-テキスト検索を行う手法の実装です。

## 概要

- **提案手法**: CLIP ファインチューニング + k近傍分類（k=10, コサイン類似度）
- **比較手法**: ResNet50, ViT-B/16（教師あり学習）、GPT-4o, Qwen3-VL, InternVL3, Llama3.2-Vision（ゼロショット）
- **評価タスク**: 4カテゴリ多ラベル分類（健全度判定・対策区分・損傷種類・損傷部位）、画像-テキスト検索（I2T/T2I/I2I）

## ディレクトリ構成

```
.
├── classification/
│   ├── data/
│   │   ├── label_definitions.py   # ラベル定義（4カテゴリ）
│   │   ├── extract_labels.py      # CSVからのラベル抽出
│   │   └── dataset.py             # PyTorch Dataset
│   ├── models/
│   │   ├── clip_finetuned_knn.py  # 提案手法: CLIP FT + kNN
│   │   ├── clip_zeroshot.py       # CLIP ゼロショット分類
│   │   ├── resnet50.py            # ResNet50 (linear_probe / finetune)
│   │   ├── resnet50_weighted.py   # ResNet50 (weighted BCE)
│   │   ├── vit.py                 # ViT-B/16 (linear_probe / finetune)
│   │   ├── vit_weighted.py        # ViT-B/16 (weighted BCE)
│   │   ├── gpt4o.py               # GPT-4o ゼロショット
│   │   ├── qwen3vl.py             # Qwen3-VL ゼロショット
│   │   ├── internvl.py            # InternVL3 ゼロショット
│   │   └── llama.py               # Llama3.2-Vision ゼロショット
│   ├── train.py                   # ResNet50/ViT 学習スクリプト
│   ├── evaluate.py                # 評価（Accuracy, F1, mAP）
│   ├── prompts.py                 # VLM共通プロンプト定義
│   ├── retrieval_eval.py          # 検索評価（Recall@k）
│   ├── qualitative_retrieval.py   # 検索の定性評価可視化
│   ├── qualitative_comparison.py  # Base vs FT 比較可視化
│   └── plot_confusion_matrices.py # 混同行列の画像生成
├── eval_retrieval.py              # CLIP検索評価（ルートレベル）
├── eval_retrieval_baseline.py     # ベースCLIP検索評価
├── ckpt_loader.py                 # チェックポイント読み込みユーティリティ
├── pick_best_ckpt.py              # 最良エポック自動選択
├── Pretreatment/
│   ├── image_check.py             # データセット画像存在チェック
│   └── build_two_csvs_from_texts_complete.py  # CSV前処理
└── pyproject.toml                 # 依存関係定義
```

## セットアップ

```bash
# Python 3.12+
pip install uv
uv sync
```

## 使い方

### 1. CLIPファインチューニング（open_clip_train を使用）

```bash
open_clip_train \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k \
    --train-data path/to/train.csv \
    --val-data path/to/val.csv \
    --csv-img-key image \
    --csv-caption-key text \
    --batch-size 128 \
    --lr 1e-4 \
    --warmup 1000
```

### 2. 提案手法（CLIP FT + kNN 分類）

```bash
python -m classification.models.clip_finetuned_knn \
    --train_csv  path/to/labeled_train.csv \
    --val_csv    path/to/labeled_val.csv \
    --ckpt_dir   path/to/checkpoints/ \
    --out        classification/results/clip_finetuned_knn_preds.csv
```

### 3. 比較手法

#### ResNet50 / ViT 学習

```bash
# ResNet50 finetune
python -m classification.train \
    --model resnet50 --mode finetune \
    --train_csv path/to/labeled_train.csv \
    --val_csv   path/to/labeled_val.csv \
    --out_dir   results/resnet50_finetune \
    --lr 1e-4 --batch_size 32

# ViT finetune
python -m classification.train \
    --model vit --mode finetune \
    --train_csv path/to/labeled_train.csv \
    --val_csv   path/to/labeled_val.csv \
    --out_dir   results/vit_finetune \
    --lr 5e-5 --batch_size 32
```

#### VLM ゼロショット分類

```bash
# GPT-4o
python -m classification.models.gpt4o \
    --csv path/to/val.csv --out results/gpt4o_preds.csv

# Qwen3-VL-8B
python -m classification.models.qwen3vl \
    --csv path/to/val.csv --out results/qwen3vl_preds.csv \
    --model Qwen/Qwen3-VL-8B --device cuda

# InternVL3-8B
python -m classification.models.internvl \
    --csv path/to/val.csv --out results/internvl_preds.csv \
    --model OpenGVLab/InternVL3-8B --device cuda
```

### 4. 評価

```bash
python -m classification.evaluate \
    --pred results/clip_finetuned_knn_preds.csv \
    --out  results/clip_finetuned_knn_metrics.json
```

### 5. 検索評価

```bash
python -m classification.retrieval_eval \
    --val_csv path/to/val.csv \
    --ckpt_dir path/to/checkpoints/ \
    --out results/retrieval_metrics.json
```

## モデル

ファインチューニング済みチェックポイントは Hugging Face Hub で公開しています:

> [**Seino404/bridge-inspection-clip**](https://huggingface.co/Seino404/bridge-inspection-clip)

## データセット

全国道路施設点検データベース（xROAD）の橋梁点検画像・所見テキストを使用。

| | 件数 |
|---|---|
| 学習データ | 130,930 |
| テストデータ | 2,679 |
| kNN DB（4カテゴリ有効） | 90,987 |

## 引用

論文準備中のため、今後公開予定です。

## ライセンス

MIT License
