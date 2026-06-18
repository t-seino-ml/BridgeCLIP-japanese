# マルチラベル分類 比較実験

道路橋点検画像に対して7種類のモデルで**マルチラベル分類**の比較実験を行う。

## 分類対象ラベル

| カテゴリ | クラス数 | 内容 |
|---------|---------|------|
| 健全度判定 (kenzenudo) | 4 | Ⅰ / Ⅱ / Ⅲ / Ⅳ |
| 対策区分 (taisaku) | 9 | A / B / C1 / C2 / E1 / E2 / M / S1 / S2 |
| 損傷種類 (damage_type) | 15 | ひびわれ / 剥離・鉄筋露出 / 腐食 など |
| 損傷部位 (damage_loc) | 20 | 主桁 / 横桁 / 床版 など |

ラベルは `data/train_clean.csv` のテキストから正規表現で自動抽出する。

---

## 実験モデル一覧

| モデル | 方式 | 備考 |
|--------|------|------|
| ResNet50 (linear_probe) | 事前学習＋線形ヘッド学習 | バックボーン凍結 |
| ResNet50 (finetune) | 全体ファインチューニング | |
| ViT-B/16 (linear_probe) | 事前学習＋線形ヘッド学習 | バックボーン凍結 |
| ViT-B/16 (finetune) | 全体ファインチューニング | |
| Base CLIP (ViT-B/32) | ゼロショット | LAION事前学習、fine-tuningなし |
| GPT-4o | ゼロショット | OpenAI API |
| Llama 3.2 Vision | ゼロショット | Ollama ローカル |
| InternVL3.5 | ゼロショット | HuggingFace ローカル |
| Qwen3-VL | ゼロショット | HuggingFace ローカル |

> **linear_probe（事前学習ゼロショット相当）**: バックボーンの重みをこのタスクのデータで一切更新せず、
> 分類ヘッドのみを学習する。事前学習の特徴表現をそのまま使う点でゼロショットに近い設定。

---

## ディレクトリ構成

```
classification/
├── data/
│   ├── label_definitions.py   # ラベル定義・表記ゆれ正規化マップ
│   ├── extract_labels.py      # テキストからラベルを抽出してCSV生成
│   └── dataset.py             # PyTorch Dataset（マルチホットベクトル）
├── models/
│   ├── resnet50.py            # ResNet50 (linear_probe / finetune)
│   ├── vit.py                 # ViT-B/16 (linear_probe / finetune)
│   ├── clip_zeroshot.py       # Base CLIP ゼロショット
│   ├── gpt4o.py               # GPT-4o API
│   ├── llama.py               # Llama (Ollama)
│   ├── internvl.py            # InternVL3.5 (HuggingFace)
│   └── qwen3vl.py             # Qwen3-VL (HuggingFace)
├── prompts.py                 # 全VLMで共通のプロンプト（一元管理）
├── train.py                   # ResNet50/ViT 学習スクリプト
├── evaluate.py                # 評価メトリクス（F1, mAP, Accuracy）
├── run_vlm.py                 # VLMゼロショット 一括実行
└── results/                   # 予測CSV・評価JSONの保存先
```

---

## 実行手順

### Step 1: ラベルをテキストから抽出する

```bash
# 検証セット
python -m classification.data.extract_labels \
    --input  data/val_clean.csv \
    --output classification/results/labeled_val.csv \
    --report

# 学習セット
python -m classification.data.extract_labels \
    --input  data/train_clean.csv \
    --output classification/results/labeled_train.csv
```

### Step 2: ResNet50 / ViT の学習

```bash
# ResNet50 linear_probe（バックボーン凍結）
python -m classification.train \
    --model resnet50 --mode linear_probe \
    --train_csv classification/results/labeled_train.csv \
    --val_csv   classification/results/labeled_val.csv \
    --out_dir   classification/results/resnet50_linear_probe \
    --epochs 20 --lr 1e-3 --batch_size 64

# ResNet50 finetune（全体学習）
python -m classification.train \
    --model resnet50 --mode finetune \
    --train_csv classification/results/labeled_train.csv \
    --val_csv   classification/results/labeled_val.csv \
    --out_dir   classification/results/resnet50_finetune \
    --epochs 30 --lr 1e-4 --batch_size 32

# ViT linear_probe
python -m classification.train \
    --model vit --mode linear_probe \
    --train_csv classification/results/labeled_train.csv \
    --val_csv   classification/results/labeled_val.csv \
    --out_dir   classification/results/vit_linear_probe \
    --epochs 20 --lr 1e-3 --batch_size 64

# ViT finetune
python -m classification.train \
    --model vit --mode finetune \
    --train_csv classification/results/labeled_train.csv \
    --val_csv   classification/results/labeled_val.csv \
    --out_dir   classification/results/vit_finetune \
    --epochs 30 --lr 5e-5 --batch_size 32
```

学習済みモデルで予測CSVを生成する場合（evaluate.py で比較するため）:
```bash
python -m classification.train --predict \
    --model resnet50 --mode finetune \
    --ckpt  classification/results/resnet50_finetune/best_model.pt \
    --val_csv  classification/results/labeled_val.csv \
    --out_csv  classification/results/resnet50_finetune_preds.csv
```

### Step 3: VLMゼロショット分類

```bash
# 事前準備
export OPENAI_API_KEY="sk-..."          # GPT-4o用
ollama pull llama3.2-vision             # Llama用
ollama serve &                          # Ollamaサーバー起動

# 全VLMを実行（最初の200件）
python -m classification.run_vlm \
    --csv    classification/results/labeled_val.csv \
    --out    classification/results \
    --n      200 \
    --models clip gpt4o llama internvl qwen3vl

# CLIP のみ（全件）
python -m classification.run_vlm \
    --csv    classification/results/labeled_val.csv \
    --out    classification/results \
    --models clip
```

### Step 4: 評価・比較

```bash
# 全モデルを一括比較
python -m classification.evaluate \
    --compare_dir classification/results \
    --out         classification/results/comparison.json

# 単一モデルの詳細評価
python -m classification.evaluate \
    --pred classification/results/clip_zeroshot_preds.csv \
    --out  classification/results/clip_zeroshot_metrics.json
```

---

## 評価メトリクス

| メトリクス | 説明 |
|-----------|------|
| Macro-F1 | クラス均等重み付けF1（クラス不均衡に頑健） |
| Micro-F1 | 全サンプル×クラスの統合F1 |
| Weighted-F1 | サンプル数重み付けF1 |
| mAP | mean Average Precision（マルチラベル向け） |
| Exact Match Accuracy | 全カテゴリが完全一致した割合 |

---

## プロンプトの統一

全VLMで使用するプロンプトは `prompts.py` で一元管理しています。
`SYSTEM_PROMPT` と `USER_PROMPT` を変更すると全モデルに反映されます。

出力形式（JSON）:
```json
{
  "健全度判定": "Ⅱ",
  "対策区分": "C1",
  "損傷種類": ["ひびわれ", "漏水・遊離石灰"],
  "損傷部位": ["主桁", "床版"]
}
```

---

## 依存ライブラリ

```bash
# 共通
pip install torch torchvision scikit-learn pandas tqdm pillow

# CLIP
pip install open_clip_torch

# GPT-4o
pip install openai

# Ollama（Llama用）
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2-vision

# InternVL3.5
pip install transformers accelerate einops timm

# Qwen3-VL
pip install transformers accelerate qwen-vl-utils
```
