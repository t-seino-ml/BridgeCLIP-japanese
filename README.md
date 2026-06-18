# BridgeCLIP — 橋梁点検画像の分類と検索のための CLIP ファインチューニング

[OpenCLIP ViT-B/32](https://github.com/mlfoundations/open_clip) を全国道路施設点検データベース（xROAD）の橋梁点検画像と所見テキストのペアでファインチューニングし、**画像分類（4 カテゴリ）と画像⇔テキスト検索**を 1 つのモデルで実現する研究コード。

## 提案手法

CLIP image encoder と text encoder をコントラスティブ損失で橋梁ドメインにファインチューニングした上で、

- **画像分類**：訓練画像との k近傍多数決（k = 10、コサイン類似度）でラベル付与
- **画像⇔テキスト検索**：image / text 埋め込みの内積で検索
- **属性ベース検索**：「クエリと同じカテゴリラベルを持つギャラリーアイテムを hit とする」評価フレーム（AttrMatch@k、属性ベース mAP、NDCG@10）

を行う。

## タスク

| カテゴリ | クラス数 | タイプ |
|---|---|---|
| 健全度判定 | 4 (Ⅰ/Ⅱ/Ⅲ/Ⅳ) | 単一ラベル |
| 対策区分 | 9 (A/B/C1/C2/E1/E2/M/S1/S2) | 単一ラベル |
| 損傷種類 | 15 | マルチラベル |
| 損傷部位 | 20 | マルチラベル |

## 主な結果（検証データ 2,679 件）

### 分類性能（macro-F1、val=2,679）

| モデル | 健全度 | 対策 | 損傷種類 | 損傷部位 | **平均** |
|---|---|---|---|---|---|
| ResNet50 weighted finetune | **0.6106** | 0.3293 | **0.5379** | 0.4640 | **0.4855** |
| ViT weighted finetune | 0.5502 | **0.3593** | 0.5310 | 0.4368 | 0.4693 |
| ResNet50 finetune | 0.5515 | 0.3416 | 0.5005 | 0.3793 | 0.4432 |
| ViT finetune | 0.5519 | 0.3502 | 0.4777 | 0.3592 | 0.4347 |
| **提案: CLIP fine-tuned + kNN** | 0.4170 | 0.2415 | 0.5115 | **0.4904** | **0.4151** |
| 提案: CLIP fine-tuned + 線形分類器 | 0.3879 | 0.2273 | 0.4939 | 0.5021 | 0.4028 |
| 提案: CLIP fine-tuned + 教師あり + weighted BCE | 0.4617 | 0.3082 | 0.5363 | 0.4721 | 0.4446 |
| GPT-4o (zero-shot) | 0.1550 | 0.1593 | 0.2967 | 0.2783 | 0.2223 |
| Qwen3-VL-8B (zero-shot) | 0.1102 | 0.1394 | 0.2195 | 0.1610 | 0.1575 |
| InternVL3-8B (zero-shot) | 0.0668 | 0.0235 | 0.2043 | 0.1291 | 0.1059 |
| Llama3.2-Vision (zero-shot) | 0.0042 | 0.0005 | 0.0004 | 0.0011 | 0.0015 |

提案手法は単一の埋め込み空間で分類と検索を同時にこなすことが特徴。**損傷部位の macro-F1 / macro-Precision で全モデル中首位**を維持しながら、教師あり強ベースラインと比較しても遜色ない水準を達成。

### 特徴空間の比較（同じ supervised 線形プローブヘッド、backbone のみ違う）

| Backbone | 平均 macro-F1 |
|---|---|
| ResNet50 (ImageNet) | 0.3278 |
| ViT-B/16 (ImageNet) | 0.3496 |
| CLIP-base (LAION-2B) | 0.3304 |
| **CLIP-FT（本提案）** | **0.4028 (+21.9% 相対改善)** |

橋梁ドメインでの対照学習は、ImageNet / LAION-2B の事前学習を **21.9% 相対** で上回ることを実証。

### 検索性能（Recall@k、val=2,679）

| 方向 | Base CLIP | **CLIP fine-tuned (epoch 5)** |
|---|---|---|
| Image→Text R@1 | 0.0004 | **0.0370** |
| Image→Text R@10 | 0.0037 | **0.2251** |
| Text→Image R@1 | 0.0011 | **0.0493** |
| Text→Image R@10 | 0.0045 | **0.2437** |

### 属性ベース検索（Text → Image, AttrMatch@1）

クエリテキストと検索画像の間で属性ラベルが 1 つでも一致すれば hit。

| カテゴリ | Base CLIP | **CLIP fine-tuned** | 倍率 |
|---|---|---|---|
| 健全度判定 | 0.3502 | **0.7043** | ×2.01 |
| 対策区分 | 0.3101 | **0.6281** | ×2.03 |
| 損傷種類 | 0.1248 | **0.6667** | **×5.34** |
| 損傷部位 | 0.1252 | **0.6397** | **×5.11** |

損傷種類・部位の T2I 検索で **5 倍超の改善**を確認。これは ResNet50/ViT のような教師あり分類器には原理的に不可能な能力。

## モデルチェックポイント

ファインチューニング済みチェックポイントは Hugging Face Hub で公開：

- [**Seino404/bridge-inspection-clip**](https://huggingface.co/Seino404/bridge-inspection-clip)

## データ

全国道路施設点検データベース（xROAD）から抽出した橋梁点検画像と所見テキストのペアデータ。

| Split | サンプル数 |
|---|---|
| Train | 130,930 |
| Validation | 2,679 |
| kNN データベース（4 カテゴリ全て有効な行のみ） | 90,987 |

データセットライセンスの都合上、画像実体は同梱しません。CSV を再構築するためのスクリプトは `Pretreatment/` を参照。

## 環境構築

```bash
# uv（推奨）
uv sync

# pip
pip install -e .
```

GPU は **1 台以上あれば動作**します。`CUDA_VISIBLE_DEVICES` 環境変数で使用 GPU を指定。

## クイックスタート

### 1. CLIP のファインチューニング

```bash
# HuggingFace から学習済み ckpt をダウンロード推奨（時間節約）
# もしくは自分でファインチューニング:
CUDA_VISIBLE_DEVICES=0 bash classification/train_clip.sh
```

### 2. 提案手法（CLIP-FT + kNN）で分類

```bash
CUDA_VISIBLE_DEVICES=0 python -m classification.models.clip_finetuned_knn \
  --train_csv classification/results/unified_train_user.csv \
  --val_csv   classification/results/unified_val_user.csv \
  --ckpt_dir  logs_classification/bridgeclip_vitb32_unified/checkpoints \
  --out       classification/results/clip_finetuned_knn_preds.csv \
  --k 10
```

### 3. 評価

```bash
python -m classification.evaluate \
  --pred classification/results/clip_finetuned_knn_preds.csv \
  --out  classification/results/clip_finetuned_knn_metrics.json
```

### 4. ベースライン（ResNet50 / ViT / GPT-4o 等）の再実行

```bash
# 教師あり baseline 6 構成を順次学習
IMAGE_ROOT=/path/to/images \
CUDA_VISIBLE_DEVICES=0 \
bash classification/run_all_classifiers.sh
```

### 5. 検索評価（標準 R@k + 属性ベース）

```bash
CUDA_VISIBLE_DEVICES=0 python -m classification.retrieval_eval \
  --val_csv  classification/results/unified_val_user.csv \
  --ckpt_dir logs_classification/bridgeclip_vitb32_unified/checkpoints \
  --out      classification/results/clip_finetuned_retrieval.json
```

出力 JSON には I2T / T2I の R@k に加えて、**AttrMatch@k / 属性ベース mAP / NDCG@10** が含まれます。

### 6. 混同行列の可視化（Blues カラーマップ）

```bash
python -m classification.plot_confusion_matrices \
  --out_dir classification/results/confusion_matrices
```

## ディレクトリ構成

```
.
├── classification/
│   ├── data/
│   │   ├── dataset.py             # multi-label dataset、`--image_root` でパス書換に対応
│   │   ├── label_definitions.py   # ラベル定義（Ⅰ/Ⅱ/Ⅲ/Ⅳ・主桁 等）
│   │   └── extract_labels.py      # 所見テキストからラベル抽出
│   ├── models/
│   │   ├── clip_finetuned_knn.py  # 提案手法（CLIP-FT 特徴の kNN 多数決）
│   │   ├── clip_classifier.py     # CLIP backbone + 教師ありヘッド
│   │   ├── clip_zeroshot.py       # text-feature 内積（zero-shot）
│   │   ├── resnet50.py / vit.py 等  # ImageNet 事前学習ベースライン
│   │   └── gpt4o.py / qwen3vl.py / internvl.py / llama.py  # zero-shot VLM
│   ├── train.py                   # 教師あり学習エントリポイント
│   ├── train_clip.sh              # CLIP ファインチューニング起動
│   ├── evaluate.py                # macro/micro/weighted F1 + Balanced Acc + per-class P/R
│   ├── retrieval_eval.py          # R@k + 属性ベース AttrMatch/mAP/NDCG
│   ├── plot_confusion_matrices.py # Blues + カラーバー付き混同行列
│   ├── predict_top1.py            # weighted 系の top-1 公平再推論
│   ├── run_all_classifiers.sh     # ResNet50/ViT 全構成の逐次実行
│   ├── run_clip_textdot.sh        # text-feature 内積の評価
│   └── retrain_clip_short.sh      # CLIP の短期間ファインチューニング
├── Pretreatment/                  # xROAD 生データから CSV を作るスクリプト
├── eval_retrieval.py / eval_retrieval_baseline.py  # スタンドアロン retrieval 評価
├── pick_best_ckpt.py              # results.jsonl から最良 epoch を選択
├── pyproject.toml                 # uv プロジェクト設定
└── README.md
```

## 実験設計上の注意

- 現在の train/val 分割は **同一橋梁の別画像が train と val の両方に含まれる**ことが分かっており（val 2,679 件のうち 2,395 件 = 89.4% が train と橋梁を共有）、「未知の橋梁への汎化」を評価していません。橋梁ID単位 (Leave-One-Bridge-Out) の再分割が必要です。
- 公開チェックポイントは `epoch_5.pt`（val_loss = 2.5242）です。中間 epoch は容量節約のため削除しています。
- 旧バージョンの README に掲載されていた「平均 Exact Match Accuracy ≈ 0.38」は、`BridgeInspectionDataset` の壊れ画像サイレント代替バグと、単一ラベルカテゴリへの BCE 損失誤適用に由来する数値でした。本リポジトリのコードは両バグを修正済みで、上記の数値は修正後の再学習・再評価結果です。

## 引用

論文準備中。

## ライセンス

コード: MIT License。データセット本体は配布しません（上記「データ」セクション参照）。
