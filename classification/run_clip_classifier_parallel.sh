#!/usr/bin/env bash
# C-F: CLIP backbone + 4 ヘッド分類 (教師あり学習)
#   C: clip_base + linear_probe
#   D: clip_ft   + linear_probe   ← レビュアーコメント1の linear probe 版
#   E: clip_ft   + finetune       ← レビュアーコメント1の本命
#   F: clip_ft_weighted + finetune (weighted BCE)
#
# 4 GPU に 1 構成ずつ並列。
#
# Usage:
#   IMAGE_ROOT=$(pwd)/images bash classification/run_clip_classifier_parallel.sh

set -euo pipefail
PYTHON="${PYTHON:-python3}"
read -r -a PYTHON_CMD <<< "$PYTHON"

: "${IMAGE_ROOT:?IMAGE_ROOT must be set}"
TRAIN_CSV="${TRAIN_CSV:-classification/results/unified_train_user.csv}"
VAL_CSV="${VAL_CSV:-classification/results/unified_val_user.csv}"
EPOCHS="${EPOCHS:-15}"        # CLIP backbone は小さいので少なめでよい
BS="${BS:-64}"
LR_LP="${LR_LP:-5e-4}"        # linear_probe lr
LR_FT="${LR_FT:-1e-5}"        # finetune lr（ViT 同様低めに）
NUM_WORKERS="${NUM_WORKERS:-6}"
POS_WEIGHT_CLIP="${POS_WEIGHT_CLIP:-10.0}"
CLIP_CKPT_DIR="${CLIP_CKPT_DIR:-logs_classification/bridgeclip_vitb32_unified/checkpoints}"
GPU0="${GPU0:-0}"; GPU1="${GPU1:-1}"; GPU2="${GPU2:-2}"; GPU3="${GPU3:-3}"

OUT_DIR="classification/results"
LOG_DIR="$OUT_DIR/clip_classifier_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[config] LOG_DIR=$LOG_DIR  IMAGE_ROOT=$IMAGE_ROOT  EPOCHS=$EPOCHS"
echo "[config] CLIP_CKPT_DIR=$CLIP_CKPT_DIR"

train_and_predict() {
  local gpu="$1" model="$2" mode="$3" out_sub="$4" lr="$5"
  local out_dir="$OUT_DIR/$out_sub"
  local out_csv="$OUT_DIR/${out_sub}_preds.csv"
  local extra=()
  if [[ "$model" == "clip_ft" || "$model" == "clip_ft_weighted" ]]; then
    extra=(--clip_ckpt_dir "$CLIP_CKPT_DIR")
  fi
  echo "[GPU $gpu] start $out_sub"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m classification.train \
    --model "$model" --mode "$mode" \
    --train_csv "$TRAIN_CSV" --val_csv "$VAL_CSV" \
    --out_dir "$out_dir" \
    --epochs "$EPOCHS" --lr "$lr" --batch_size "$BS" \
    --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT" \
    --pos_weight_clip "$POS_WEIGHT_CLIP" \
    "${extra[@]}"

  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m classification.train --predict \
    --model "$model" --mode "$mode" \
    --val_csv "$VAL_CSV" \
    --ckpt "$out_dir/best_model.pt" \
    --out_csv "$out_csv" \
    --batch_size "$BS" --num_workers "$NUM_WORKERS" \
    --image_root "$IMAGE_ROOT" \
    "${extra[@]}"
  echo "[GPU $gpu] done  $out_sub"
}

# 各レーン関数
lane0() { train_and_predict "$GPU0" clip_base         linear_probe clip_base_linear_probe         "$LR_LP"; }   # C
lane1() { train_and_predict "$GPU1" clip_ft           linear_probe clip_ft_linear_probe           "$LR_LP"; }   # D
lane2() { train_and_predict "$GPU2" clip_ft           finetune     clip_ft_finetune               "$LR_FT"; }   # E
lane3() { train_and_predict "$GPU3" clip_ft_weighted  finetune     clip_ft_weighted_finetune      "$LR_FT"; }   # F

lane0 > "$LOG_DIR/lane0_clip_base_lp.log"           2>&1 &  PID0=$!
lane1 > "$LOG_DIR/lane1_clip_ft_lp.log"             2>&1 &  PID1=$!
lane2 > "$LOG_DIR/lane2_clip_ft_ft.log"             2>&1 &  PID2=$!
lane3 > "$LOG_DIR/lane3_clip_ft_weighted_ft.log"    2>&1 &  PID3=$!

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

# 評価
echo "============================================================"
echo "[evaluate] CLIP classifier variants"
echo "============================================================"
"${PYTHON_CMD[@]}" - <<'PY'
import json
from pathlib import Path
from classification.evaluate import evaluate_model

names = [
    'clip_base_linear_probe',
    'clip_ft_linear_probe',
    'clip_ft_finetune',
    'clip_ft_weighted_finetune',
]
for name in names:
    p = Path(f'classification/results/{name}_preds.csv')
    if not p.exists():
        print(f'  [skip] {p}'); continue
    res = evaluate_model(str(p))
    out = Path(f'classification/results/{name}_metrics.json')
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved: {out}')
PY

# top-1 補正（weighted 系のみ過剰予測する可能性 → argmax 版も作る）
echo "============================================================"
echo "[top-1] regenerate argmax-only preds for weighted variant"
echo "============================================================"
CUDA_VISIBLE_DEVICES="$GPU3" "${PYTHON_CMD[@]}" -m classification.predict_top1 \
  --val_csv "$VAL_CSV" \
  --image_root "$IMAGE_ROOT" \
  --models clip_ft_weighted:finetune:clip_ft_weighted_finetune/best_model.pt

echo "DONE."
