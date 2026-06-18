#!/usr/bin/env bash
# EN ResNet50/ViT 6 構成 + 3 GPU 並列（GPU 0/1/3、GPU 2 はスキップ）。
# バランス: 3 レーン × 各 ~120 分目安。
#
# Usage:
#   IMAGE_ROOT=$(pwd)/images bash classification/run_all_classifiers_parallel_en_3gpu.sh

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
GPU_A="${GPU_A:-0}"  # ResNet50 系
GPU_B="${GPU_B:-1}"  # ResNet50 weighted + ViT LP
GPU_C="${GPU_C:-3}"  # ViT FT + ViT weighted FT

OUT_DIR="classification/results_en"
LOG_DIR="$OUT_DIR/parallel_logs_3gpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[config] LABEL_LANG=en  LOG_DIR=$LOG_DIR  GPUs=$GPU_A,$GPU_B,$GPU_C"
echo "[config] TRAIN_CSV=$TRAIN_CSV  VAL_CSV=$VAL_CSV  EPOCHS=$EPOCHS"

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

# 3 レーン構成（合計時間がほぼ等しいよう ResNet/ViT を混ぜる）
laneA() {
  # GPU_A: ResNet50 LP + ResNet50 FT （~30 + ~60 = ~90 分）
  train_and_predict "$GPU_A" resnet50          linear_probe resnet50_linear_probe        "$LR_RESNET_LP" "$BS_RESNET"
  train_and_predict "$GPU_A" resnet50          finetune     resnet50_finetune            "$LR_RESNET_FT" "$BS_RESNET"
}
laneB() {
  # GPU_B: ResNet50 weighted_FT + ViT LP （~60 + ~30 = ~90 分）
  train_and_predict "$GPU_B" resnet50_weighted finetune     resnet50_weighted_finetune   "$LR_RESNET_FT" "$BS_RESNET"
  train_and_predict "$GPU_B" vit               linear_probe vit_linear_probe             "$LR_VIT_LP"    "$BS_VIT"
}
laneC() {
  # GPU_C: ViT FT + ViT weighted_FT （~90 + ~90 = ~180 分、最長レーン）
  train_and_predict "$GPU_C" vit               finetune     vit_finetune                 "$LR_VIT_FT"    "$BS_VIT"
  train_and_predict "$GPU_C" vit_weighted      finetune     vit_weighted_finetune        "$LR_VIT_FT"    "$BS_VIT"
}

laneA > "$LOG_DIR/laneA_resnet.log" 2>&1 &  PA=$!
laneB > "$LOG_DIR/laneB_mixed.log"  2>&1 &  PB=$!
laneC > "$LOG_DIR/laneC_vit.log"    2>&1 &  PC=$!
echo "[launched] PIDs A=$PA B=$PB C=$PC"
echo "[tail logs] tail -f $LOG_DIR/lane*.log"

set +e
wait $PA; RA=$?
wait $PB; RB=$?
wait $PC; RC=$?
set -e
echo "[exit codes] laneA=$RA laneB=$RB laneC=$RC"
if [[ $RA -ne 0 || $RB -ne 0 || $RC -ne 0 ]]; then
  echo "[ERROR] Some lanes failed. Check $LOG_DIR/lane*.log"; exit 1
fi

# 評価 (EN)
LABEL_LANG=en "${PYTHON_CMD[@]}" - <<'PY'
import json
from pathlib import Path
from classification.evaluate_en import evaluate_model
models = [
    'clip_base_knn', 'clip_finetuned_knn',
    'resnet50_finetune', 'resnet50_linear_probe', 'resnet50_weighted_finetune',
    'vit_finetune', 'vit_linear_probe', 'vit_weighted_finetune',
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
