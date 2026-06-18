#!/bin/bash
# ファインチューニングCLIP 学習スクリプト（GPU は CUDA_VISIBLE_DEVICES で指定、デフォルト 0）
# unified_train_local.csv / unified_val_local.csv を使用
#
# 使い方:
#   bash classification/train_clip.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python -m open_clip_train.main \
  --train-data "${ROOT_DIR}/classification/results/unified_train_user.csv" \
  --val-data   "${ROOT_DIR}/classification/results/unified_val_user.csv" \
  --dataset-type csv \
  --csv-separator "," \
  --csv-img-key image \
  --csv-caption-key text \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --batch-size 128 \
  --epochs 100 \
  --lr 1e-4 \
  --wd 0.1 \
  --warmup 1000 \
  --precision amp \
  --workers 8 \
  --report-to tensorboard \
  --save-frequency 1 \
  --zeroshot-frequency 0 \
  --logs "${ROOT_DIR}/logs_classification" \
  --name "bridgeclip_vitb32_unified"
