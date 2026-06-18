#!/usr/bin/env bash
# ResNet50 / ViT を 3 モード × 2 バックボーン = 6 通りで再学習し、予測CSVを書き出す。
#
# Python インタプリタは PYTHON 環境変数で上書き可能（既定 python3）。
#
# 変更点（前回学習からの差分）:
#   1) kenzenudo / taisaku は CrossEntropyLoss、damage_type / damage_loc は BCEWithLogitsLoss
#   2) ViT は VIT_TRAIN_TRANSFORM / VIT_TRANSFORM を使う
#   3) weighted 系の pos_weight を上限クリップ（既定 10）
#   4) CSV の image 列を --image_root で実体ディレクトリに差し替え
#
# Usage:
#   IMAGE_ROOT=/path/to/images bash classification/run_all_classifiers.sh
#
# 環境変数:
#   IMAGE_ROOT    画像ディレクトリ（必須）
#   TRAIN_CSV     学習CSV     （既定: classification/results/unified_train_local.csv）
#   VAL_CSV       検証CSV     （既定: classification/results/unified_val_local.csv）
#   EPOCHS        エポック数  （既定: 30 — 100 は 6 モデルだと長すぎ。必要なら上げる）
#   BS_RESNET     ResNet バッチ（既定: 64）
#   BS_VIT        ViT  バッチ（既定: 32）
#   LR_RESNET_FT  ResNet finetune lr（既定: 1e-4）
#   LR_RESNET_LP  ResNet linear_probe lr（既定: 1e-3）
#   LR_VIT_FT     ViT finetune lr（既定: 1e-5）  ← ViT 崩壊対策で小さめ
#   LR_VIT_LP     ViT linear_probe lr（既定: 5e-4）
#   NUM_WORKERS   DataLoader workers（既定: 8）
#
# 出力:
#   classification/results/<model_name>/best_model.pt
#   classification/results/<model_name>_preds.csv

set -euo pipefail

PYTHON="${PYTHON:-python3}"
# 複数語の PYTHON（例: "uv run python"）に対応するため配列化
read -r -a PYTHON_CMD <<< "$PYTHON"

: "${IMAGE_ROOT:?IMAGE_ROOT must be set (e.g. /path/to/images)}"
TRAIN_CSV="${TRAIN_CSV:-classification/results/unified_train_local.csv}"
VAL_CSV="${VAL_CSV:-classification/results/unified_val_local.csv}"
EPOCHS="${EPOCHS:-30}"
BS_RESNET="${BS_RESNET:-64}"
BS_VIT="${BS_VIT:-32}"
LR_RESNET_FT="${LR_RESNET_FT:-1e-4}"
LR_RESNET_LP="${LR_RESNET_LP:-1e-3}"
LR_VIT_FT="${LR_VIT_FT:-1e-5}"
LR_VIT_LP="${LR_VIT_LP:-5e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
POS_WEIGHT_CLIP="${POS_WEIGHT_CLIP:-10.0}"

OUT_DIR="classification/results"
mkdir -p "$OUT_DIR"

echo "[config] IMAGE_ROOT=$IMAGE_ROOT  EPOCHS=$EPOCHS"
echo "[config] TRAIN_CSV=$TRAIN_CSV  VAL_CSV=$VAL_CSV"

train_one() {
  local model="$1" mode="$2" out_sub="$3" lr="$4" bs="$5"
  local out_dir="$OUT_DIR/$out_sub"
  echo "------------------------------------------------------------"
  echo "[train] model=$model mode=$mode out=$out_dir lr=$lr bs=$bs"
  echo "------------------------------------------------------------"
  "${PYTHON_CMD[@]}" -m classification.train \
    --model "$model" --mode "$mode" \
    --train_csv "$TRAIN_CSV" \
    --val_csv   "$VAL_CSV"   \
    --out_dir   "$out_dir"   \
    --epochs "$EPOCHS" --lr "$lr" --batch_size "$bs" \
    --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT" \
    --pos_weight_clip "$POS_WEIGHT_CLIP"
}

predict_one() {
  local model="$1" mode="$2" out_sub="$3" bs="$4"
  local out_dir="$OUT_DIR/$out_sub"
  local out_csv="$OUT_DIR/${out_sub}_preds.csv"
  echo "[predict] $out_sub -> $out_csv"
  "${PYTHON_CMD[@]}" -m classification.train \
    --predict \
    --model "$model" --mode "$mode" \
    --val_csv "$VAL_CSV" \
    --ckpt    "$out_dir/best_model.pt" \
    --out_csv "$out_csv" \
    --batch_size "$bs" --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT"
}

run_pair() {
  local model="$1" mode="$2" out_sub="$3" lr="$4" bs="$5"
  train_one   "$model" "$mode" "$out_sub" "$lr" "$bs"
  predict_one "$model" "$mode" "$out_sub" "$bs"
}

# ── ResNet50 系 ──
run_pair resnet50          linear_probe resnet50_linear_probe        "$LR_RESNET_LP" "$BS_RESNET"
run_pair resnet50          finetune     resnet50_finetune            "$LR_RESNET_FT" "$BS_RESNET"
run_pair resnet50_weighted finetune     resnet50_weighted_finetune   "$LR_RESNET_FT" "$BS_RESNET"

# ── ViT 系 ──
run_pair vit               linear_probe vit_linear_probe             "$LR_VIT_LP"    "$BS_VIT"
run_pair vit               finetune     vit_finetune                 "$LR_VIT_FT"    "$BS_VIT"
run_pair vit_weighted      finetune     vit_weighted_finetune        "$LR_VIT_FT"    "$BS_VIT"

# ── 評価 (JA) ──
echo "============================================================"
echo "[evaluate] re-running evaluate.py on all regenerated JA preds"
echo "============================================================"
"${PYTHON_CMD[@]}" - <<'PY'
import json
from pathlib import Path
from classification.evaluate import evaluate_model

ja_models = [
    'clip_base_knn',
    'clip_finetuned_knn',
    'resnet50_finetune',
    'resnet50_linear_probe',
    'resnet50_weighted_finetune',
    'vit_finetune',
    'vit_linear_probe',
    'vit_weighted_finetune',
]
all_results = {}
for name in ja_models:
    p = Path(f'classification/results/{name}_preds.csv')
    if not p.exists():
        print(f'  [skip] {p} not found')
        continue
    res = evaluate_model(str(p))
    all_results[name] = res
    Path(f'classification/results/{name}_metrics.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8'
    )
Path('classification/results/comparison_ja.json').write_text(
    json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8'
)
print('saved comparison_ja.json')
PY

echo "DONE."
