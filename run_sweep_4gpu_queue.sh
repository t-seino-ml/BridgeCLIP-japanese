#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
#TRAIN_DIR="${ROOT}/train_subsets"
TRAIN_DIR="${ROOT}/train_subsets_item"
#VAL_CSV="${ROOT}/val_data/val_clean.csv"
VAL_CSV="${ROOT}/val_item_data/val_clean_base.csv"
#LOG_DIR="${ROOT}/logs_sweep"
LOG_DIR="${ROOT}/logs_sweep_item"
mkdir -p "${LOG_DIR}"

# 既存run名衝突を避けるためタイムスタンプを付ける
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

# キュー（FIFO）を作る
FIFO="$(mktemp -u)"
mkfifo "${FIFO}"
trap 'rm -f "${FIFO}"' EXIT

# CSV一覧をFIFOへ流し込む（sortで順番固定）
(
  #ls -1 "${TRAIN_DIR}"/train_*.csv | sort
  ls -1 "${TRAIN_DIR}"/train_*base.csv | sort
) > "${FIFO}" &

worker () {
  local gpu="$1"
  while read -r csv; do
    [[ -f "${csv}" ]] || continue
    base="$(basename "${csv}" .csv)"   # train_10k 等
    name="roadclip_vitb32_${base}_${TS}"

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

# GPU 0..3 でワーカーを起動
worker 0 &
worker 1 &
worker 2 &
worker 3 &

wait
echo "[ALL DONE] ${LOG_DIR}"
