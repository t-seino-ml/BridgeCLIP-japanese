#!/usr/bin/env bash
# EN CLIP backbone + supervised head 並列実行（GPU 0/1/3、GPU 2 スキップ）。
#   レーンA (GPU0): clip_base_LP + clip_ft_LP（軽量）
#   レーンB (GPU1): clip_ft_FT
#   レーンC (GPU3): clip_ft_weighted_FT
#
# Usage:
#   IMAGE_ROOT=$(pwd)/images CLIP_CKPT_DIR=logs_classification_en/clip_en_v2/checkpoints \
#     bash classification/run_clip_classifier_parallel_en_3gpu.sh

set -euo pipefail
export LABEL_LANG=en

PYTHON="${PYTHON:-python3}"
read -r -a PYTHON_CMD <<< "$PYTHON"
: "${IMAGE_ROOT:?IMAGE_ROOT must be set}"
TRAIN_CSV="${TRAIN_CSV:-classification/results/unified_train_user_en.csv}"
VAL_CSV="${VAL_CSV:-classification/results/unified_val_user_en.csv}"
EPOCHS="${EPOCHS:-15}"
BS="${BS:-64}"
LR_LP="${LR_LP:-5e-4}"
LR_FT="${LR_FT:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-6}"
POS_WEIGHT_CLIP="${POS_WEIGHT_CLIP:-10.0}"
CLIP_CKPT_DIR="${CLIP_CKPT_DIR:-logs_classification_en/clip_en_v2/checkpoints}"
GPU_A="${GPU_A:-0}"; GPU_B="${GPU_B:-1}"; GPU_C="${GPU_C:-3}"

OUT_DIR="classification/results_en"
LOG_DIR="$OUT_DIR/clip_classifier_logs_3gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[config] LABEL_LANG=en  LOG_DIR=$LOG_DIR  EPOCHS=$EPOCHS  GPUs=$GPU_A,$GPU_B,$GPU_C"
echo "[config] CLIP_CKPT_DIR=$CLIP_CKPT_DIR"

train_and_predict() {
  local gpu="$1" model="$2" mode="$3" out_sub="$4" lr="$5"
  local out_dir="$OUT_DIR/$out_sub"
  local out_csv="$OUT_DIR/${out_sub}_en_preds.csv"
  local extra=()
  if [[ "$model" == "clip_ft" || "$model" == "clip_ft_weighted" ]]; then
    extra=(--clip_ckpt_dir "$CLIP_CKPT_DIR")
  fi
  echo "[GPU $gpu] start $out_sub"
  CUDA_VISIBLE_DEVICES="$gpu" LABEL_LANG=en "${PYTHON_CMD[@]}" -m classification.train \
    --model "$model" --mode "$mode" \
    --train_csv "$TRAIN_CSV" --val_csv "$VAL_CSV" \
    --out_dir "$out_dir" \
    --epochs "$EPOCHS" --lr "$lr" --batch_size "$BS" \
    --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT" \
    --pos_weight_clip "$POS_WEIGHT_CLIP" \
    "${extra[@]}"

  CUDA_VISIBLE_DEVICES="$gpu" LABEL_LANG=en "${PYTHON_CMD[@]}" -m classification.train --predict \
    --model "$model" --mode "$mode" \
    --val_csv "$VAL_CSV" \
    --ckpt "$out_dir/best_model.pt" \
    --out_csv "$out_csv" \
    --batch_size "$BS" --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT" \
    "${extra[@]}"
  echo "[GPU $gpu] done  $out_sub"
}

laneA() {
  train_and_predict "$GPU_A" clip_base linear_probe clip_base_linear_probe "$LR_LP"
  train_and_predict "$GPU_A" clip_ft   linear_probe clip_ft_linear_probe   "$LR_LP"
}
laneB() { train_and_predict "$GPU_B" clip_ft          finetune clip_ft_finetune          "$LR_FT"; }
laneC() { train_and_predict "$GPU_C" clip_ft_weighted finetune clip_ft_weighted_finetune "$LR_FT"; }

laneA > "$LOG_DIR/laneA_clip_lp.log" 2>&1 &  PA=$!
laneB > "$LOG_DIR/laneB_clip_ft.log" 2>&1 &  PB=$!
laneC > "$LOG_DIR/laneC_clip_w_ft.log" 2>&1 &  PC=$!
echo "[launched] PIDs A=$PA B=$PB C=$PC"
echo "[tail logs] tail -f $LOG_DIR/lane*.log"

set +e
wait $PA; RA=$?
wait $PB; RB=$?
wait $PC; RC=$?
set -e
echo "[exit codes] laneA=$RA laneB=$RB laneC=$RC"
if [[ $RA -ne 0 || $RB -ne 0 || $RC -ne 0 ]]; then
  echo "[ERROR] Some lanes failed."; exit 1
fi

# 評価
LABEL_LANG=en "${PYTHON_CMD[@]}" - <<'PY'
import json
from pathlib import Path
from classification.evaluate_en import evaluate_model
for name in ['clip_base_linear_probe', 'clip_ft_linear_probe',
             'clip_ft_finetune', 'clip_ft_weighted_finetune']:
    p = Path(f'classification/results_en/{name}_en_preds.csv')
    if not p.exists():
        print(f'  [skip] {p}'); continue
    res = evaluate_model(str(p))
    Path(f'classification/results_en/{name}_en_metrics.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved: {name}_en_metrics.json')
PY

# top-1 補正
echo "[top-1] regenerate argmax-only preds for EN weighted variant"
CUDA_VISIBLE_DEVICES="$GPU_C" LABEL_LANG=en "${PYTHON_CMD[@]}" -m classification.predict_top1 \
  --val_csv "$VAL_CSV" \
  --image_root "$IMAGE_ROOT" \
  --out_dir "$OUT_DIR" \
  --models clip_ft_weighted:finetune:clip_ft_weighted_finetune/best_model.pt

echo "DONE."
