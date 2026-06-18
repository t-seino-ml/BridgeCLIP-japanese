#!/usr/bin/env bash
# EN データスケーリング sweep（指定 1 GPU で逐次実行）
# Phase 2 と並列に走らせるための単独 GPU 用バリアント。
#
# Usage:
#   GPU=2 ROOT=$(pwd) bash run_sweep_single_gpu_en.sh

set -euo pipefail

GPU="${GPU:-2}"
ROOT="${ROOT:-$(pwd)}"
TRAIN_DIR="${ROOT}/train_subsets_user_en"
VAL_CSV="${ROOT}/val_item_data_en/val_clean_base.csv"
LOG_DIR="${ROOT}/logs_sweep_en"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"

COMMON_ARGS=(
  --val-data "${VAL_CSV}"
  --dataset-type csv
  --csv-separator ","
  --csv-img-key image
  --csv-caption-key text
  --model ViT-B-32
  --pretrained laion2b_s34b_b79k
  --batch-size 128
  --epochs 10
  --lr 1e-4
  --wd 0.1
  --warmup 1000
  --precision amp
  --workers 8
  --report-to tensorboard
  --save-frequency 1
  --zeroshot-frequency 0
  --logs "${LOG_DIR}"
)

echo "[config] GPU=$GPU  ROOT=$ROOT  TS=$TS"
echo "[config] TRAIN_DIR=$TRAIN_DIR  VAL=$VAL_CSV"

# 小さい順から処理（早く完了するものから）
SUBSETS=(
  train_10k_base.csv
  train_20k_base.csv
  train_30k_base.csv
  train_40k_base.csv
  train_50k_base.csv
  train_60k_base.csv
  train_70k_base.csv
  train_80k_base.csv
  train_90k_base.csv
  train_100k_base.csv
  train_110k_base.csv
  train_120k_base.csv
  train_clean_base.csv
)

for csv_name in "${SUBSETS[@]}"; do
  csv="${TRAIN_DIR}/${csv_name}"
  [[ -f "${csv}" ]] || { echo "[SKIP] ${csv} not found"; continue; }
  base="$(basename "${csv}" .csv)"
  name="bridgeclip_vitb32_en_${base}_${TS}"
  log="${LOG_DIR}/${name}.log"
  if [[ -d "${LOG_DIR}/${name}/checkpoints" ]]; then
    echo "[SKIP-DONE] ${name} already has checkpoints/"
    continue
  fi
  echo "[GPU ${GPU}] START ${name} (${csv_name})"
  CUDA_VISIBLE_DEVICES="${GPU}" \
    uv run python -m open_clip_train.main \
      --train-data "${csv}" \
      --name "${name}" \
      "${COMMON_ARGS[@]}" \
      > "${log}" 2>&1
  echo "[GPU ${GPU}] DONE  ${name}"
done

echo "[ALL DONE] sweep finished. logs in ${LOG_DIR}"
