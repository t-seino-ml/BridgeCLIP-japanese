#!/bin/bash
# ベースCLIP k近傍分類スクリプト（GPU2使用、学習なし）
#
# 使い方:
#   bash classification/run_clip_base_knn.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CUDA_VISIBLE_DEVICES=2 uv run python -m classification.models.clip_finetuned_knn \
    --train_csv "${ROOT_DIR}/classification/results/unified_train_user.csv" \
    --val_csv   "${ROOT_DIR}/classification/results/unified_val_user.csv" \
    --out       "${ROOT_DIR}/classification/results/clip_base_knn_preds.csv" \
    --k 10
