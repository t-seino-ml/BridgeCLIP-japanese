#!/usr/bin/env bash
# EN データスケーリング sweep（GPU 0/1/3、GPU 2 スキップ）
# 既存 `run_sweep_4gpu_queue_en.sh` の 3 GPU 版。
#
# Usage:
#   ROOT=$(pwd) bash run_sweep_3gpu_queue_en.sh

set -euo pipefail

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

FIFO="$(mktemp -u)"
mkfifo "${FIFO}"
trap 'rm -f "${FIFO}"' EXIT

(
  ls -1 "${TRAIN_DIR}"/train_*_base.csv | sort
) > "${FIFO}" &

worker () {
  local gpu="$1"
  while read -r csv; do
    [[ -f "${csv}" ]] || continue
    base="$(basename "${csv}" .csv)"
    name="bridgeclip_vitb32_en_${base}_${TS}"
    echo "[GPU ${gpu}] START ${name} (${csv})"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      uv run python -m open_clip_train.main \
        --train-data "${csv}" \
        --name "${name}" \
        "${COMMON_ARGS[@]}" \
        > "${LOG_DIR}/${name}.log" 2>&1
    echo "[GPU ${gpu}] DONE  ${name}"
  done < "${FIFO}"
}

# GPU 0/1/3 のみ
worker 0 &
worker 1 &
worker 3 &

wait
echo "[ALL DONE] ${LOG_DIR}"
