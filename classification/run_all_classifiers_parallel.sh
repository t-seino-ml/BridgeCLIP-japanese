#!/usr/bin/env bash
# 6 モデルを 4 GPU に振り分けて並列実行する。
#
# 振り分け（学習時間目安）:
#   GPU 0: resnet50_linear_probe (~30min)  + resnet50_finetune (~60min)         = 90min
#   GPU 1: resnet50_weighted_finetune (~60min)                                  = 60min
#   GPU 2: vit_linear_probe (~30min)        + vit_finetune (~90min)             = 120min
#   GPU 3: vit_weighted_finetune (~90min)                                       = 90min
#
# Usage:
#   IMAGE_ROOT=$(pwd)/images bash classification/run_all_classifiers_parallel.sh
#
# 環境変数は run_all_classifiers.sh と同じ。さらに:
#   GPU0..GPU3   各レーンに割り当てるGPU index（既定 0/1/2/3）

set -euo pipefail

PYTHON="${PYTHON:-python3}"
# 複数語の PYTHON（例: "uv run python"）に対応するため配列化
read -r -a PYTHON_CMD <<< "$PYTHON"
: "${IMAGE_ROOT:?IMAGE_ROOT must be set}"
TRAIN_CSV="${TRAIN_CSV:-classification/results/unified_train_user.csv}"
VAL_CSV="${VAL_CSV:-classification/results/unified_val_user.csv}"
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

OUT_DIR="classification/results"
LOG_DIR="$OUT_DIR/parallel_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[config] LOG_DIR=$LOG_DIR  IMAGE_ROOT=$IMAGE_ROOT  EPOCHS=$EPOCHS"
echo "[config] GPUs: lane0=$GPU0 lane1=$GPU1 lane2=$GPU2 lane3=$GPU3"

train_and_predict() {
  # args: GPU_ID model mode out_sub lr bs
  local gpu="$1" model="$2" mode="$3" out_sub="$4" lr="$5" bs="$6"
  local out_dir="$OUT_DIR/$out_sub"
  local out_csv="$OUT_DIR/${out_sub}_preds.csv"
  echo "[GPU $gpu] start $out_sub"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m classification.train \
    --model "$model" --mode "$mode" \
    --train_csv "$TRAIN_CSV" --val_csv "$VAL_CSV" \
    --out_dir "$out_dir" \
    --epochs "$EPOCHS" --lr "$lr" --batch_size "$bs" \
    --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT" \
    --pos_weight_clip "$POS_WEIGHT_CLIP"

  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m classification.train --predict \
    --model "$model" --mode "$mode" \
    --val_csv "$VAL_CSV" \
    --ckpt "$out_dir/best_model.pt" \
    --out_csv "$out_csv" \
    --batch_size "$bs" --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT"
  echo "[GPU $gpu] done  $out_sub"
}

# ── 各レーン（GPU）で逐次実行する関数 ──
lane0() {
  train_and_predict "$GPU0" resnet50          linear_probe resnet50_linear_probe        "$LR_RESNET_LP" "$BS_RESNET"
  train_and_predict "$GPU0" resnet50          finetune     resnet50_finetune            "$LR_RESNET_FT" "$BS_RESNET"
}
lane1() {
  train_and_predict "$GPU1" resnet50_weighted finetune     resnet50_weighted_finetune   "$LR_RESNET_FT" "$BS_RESNET"
}
lane2() {
  train_and_predict "$GPU2" vit               linear_probe vit_linear_probe             "$LR_VIT_LP"    "$BS_VIT"
  train_and_predict "$GPU2" vit               finetune     vit_finetune                 "$LR_VIT_FT"    "$BS_VIT"
}
lane3() {
  train_and_predict "$GPU3" vit_weighted      finetune     vit_weighted_finetune        "$LR_VIT_FT"    "$BS_VIT"
}

# 4 レーンを並列起動
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
