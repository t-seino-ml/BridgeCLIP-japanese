#!/usr/bin/env bash
# EN 版: ResNet50/ViT を 3 モード × 2 backbone = 6 通りで並列学習し、予測 CSV を生成。
# 既存 `run_all_classifiers_parallel.sh` の EN 同等版。LABEL_LANG=en を強制する。
#
# Usage:
#   IMAGE_ROOT=$(pwd)/images bash classification/run_all_classifiers_parallel_en.sh

set -euo pipefail

export LABEL_LANG=en

PYTHON="${PYTHON:-python3}"
read -r -a PYTHON_CMD <<< "$PYTHON"
: "${IMAGE_ROOT:?IMAGE_ROOT must be set}"
TRAIN_CSV="${TRAIN_CSV:-classification/results/unified_train_user_en.csv}"
VAL_CSV="${VAL_CSV:-classification/results/unified_val_user_en.csv}"
EPOCHS="${EPOCHS:-30}"
BS_RESNET="${BS_RESNET:-64}"
BS_VIT="${BS_VIT:-32}"
LR_RESNET_FT="${LR_RESNET_FT:-1e-4}"
LR_RESNET_LP="${LR_RESNET_LP:-1e-3}"
LR_VIT_FT="${LR_VIT_FT:-1e-5}"
LR_VIT_LP="${LR_VIT_LP:-5e-4}"
NUM_WORKERS="${NUM_WORKERS:-6}"
POS_WEIGHT_CLIP="${POS_WEIGHT_CLIP:-10.0}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"

OUT_DIR="classification/results_en"
LOG_DIR="$OUT_DIR/parallel_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[config] LABEL_LANG=en  LOG_DIR=$LOG_DIR  IMAGE_ROOT=$IMAGE_ROOT  EPOCHS=$EPOCHS"
echo "[config] TRAIN_CSV=$TRAIN_CSV  VAL_CSV=$VAL_CSV"

train_and_predict() {
  local gpu="$1" model="$2" mode="$3" out_sub="$4" lr="$5" bs="$6"
  local out_dir="$OUT_DIR/$out_sub"
  local out_csv="$OUT_DIR/${out_sub}_en_preds.csv"
  echo "[GPU $gpu] start $out_sub"
  CUDA_VISIBLE_DEVICES="$gpu" LABEL_LANG=en "${PYTHON_CMD[@]}" -m classification.train \
    --model "$model" --mode "$mode" \
    --train_csv "$TRAIN_CSV" --val_csv "$VAL_CSV" \
    --out_dir "$out_dir" \
    --epochs "$EPOCHS" --lr "$lr" --batch_size "$bs" \
    --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT" \
    --pos_weight_clip "$POS_WEIGHT_CLIP"

  CUDA_VISIBLE_DEVICES="$gpu" LABEL_LANG=en "${PYTHON_CMD[@]}" -m classification.train --predict \
    --model "$model" --mode "$mode" \
    --val_csv "$VAL_CSV" \
    --ckpt "$out_dir/best_model.pt" \
    --out_csv "$out_csv" \
    --batch_size "$bs" --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT"
  echo "[GPU $gpu] done  $out_sub"
}

lane0() {
  train_and_predict "$GPU0" resnet50          linear_probe resnet50_linear_probe        "$LR_RESNET_LP" "$BS_RESNET"
  train_and_predict "$GPU0" resnet50          finetune     resnet50_finetune            "$LR_RESNET_FT" "$BS_RESNET"
}
lane1() { train_and_predict "$GPU1" resnet50_weighted finetune     resnet50_weighted_finetune   "$LR_RESNET_FT" "$BS_RESNET"; }
lane2() {
  train_and_predict "$GPU2" vit               linear_probe vit_linear_probe             "$LR_VIT_LP"    "$BS_VIT"
  train_and_predict "$GPU2" vit               finetune     vit_finetune                 "$LR_VIT_FT"    "$BS_VIT"
}
lane3() { train_and_predict "$GPU3" vit_weighted      finetune     vit_weighted_finetune        "$LR_VIT_FT"    "$BS_VIT"; }

lane0 > "$LOG_DIR/lane0.log" 2>&1 &  PID0=$!
lane1 > "$LOG_DIR/lane1.log" 2>&1 &  PID1=$!
lane2 > "$LOG_DIR/lane2.log" 2>&1 &  PID2=$!
lane3 > "$LOG_DIR/lane3.log" 2>&1 &  PID3=$!

echo "[launched] PIDs: $PID0 $PID1 $PID2 $PID3"
echo "[tail logs] tail -f $LOG_DIR/lane*.log"

set +e
wait $PID0; RC0=$?
wait $PID1; RC1=$?
wait $PID2; RC2=$?
wait $PID3; RC3=$?
set -e
echo "[exit codes] lane0=$RC0 lane1=$RC1 lane2=$RC2 lane3=$RC3"
if [[ $RC0 -ne 0 || $RC1 -ne 0 || $RC2 -ne 0 || $RC3 -ne 0 ]]; then
  echo "[ERROR] Some lanes failed. Check $LOG_DIR/lane*.log"
  exit 1
fi

# 評価（EN）
echo "============================================================"
echo "[evaluate] EN preds → evaluate_en.py"
echo "============================================================"
LABEL_LANG=en "${PYTHON_CMD[@]}" - <<'PY'
import json
from pathlib import Path
from classification.evaluate_en import evaluate_model

models = [
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
for name in models:
    p = Path(f'classification/results_en/{name}_en_preds.csv')
    if not p.exists():
        print(f'  [skip] {p}'); continue
    res = evaluate_model(str(p))
    all_results[name] = res
    Path(f'classification/results_en/{name}_en_metrics.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
Path('classification/results_en/comparison_en.json').write_text(
    json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved comparison_en.json')
PY

echo "DONE."
