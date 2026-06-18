#!/usr/bin/env bash
# A + B: text-dot-product 分類
#   A: CLIP-base (laion2b) + text-dot
#   B: CLIP-FT  (epoch 5)   + text-dot
#
# Usage:
#   IMAGE_ROOT=$(pwd)/images bash classification/run_clip_textdot.sh

set -euo pipefail
PYTHON="${PYTHON:-python3}"
read -r -a PYTHON_CMD <<< "$PYTHON"

: "${IMAGE_ROOT:?IMAGE_ROOT must be set}"
VAL_CSV="${VAL_CSV:-classification/results/unified_val_user.csv}"
GPU="${GPU:-0}"
THRESHOLD="${THRESHOLD:-0.3}"
BS="${BS:-64}"
CLIP_CKPT_DIR="${CLIP_CKPT_DIR:-logs_classification/bridgeclip_vitb32_unified/checkpoints}"

OUT_DIR="classification/results"
mkdir -p "$OUT_DIR"

# A: ベース CLIP の text-dot（JA）
echo "============================================================"
echo "[A] CLIP-base text-dot-product (JA)"
echo "============================================================"
CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON_CMD[@]}" -m classification.models.clip_zeroshot \
  --csv "$VAL_CSV" \
  --out "$OUT_DIR/clip_base_textdot_preds.csv" \
  --batch_size "$BS" --threshold "$THRESHOLD" \
  --image_root "$IMAGE_ROOT"

# B: CLIP-FT (epoch 5) の text-dot（JA）
echo "============================================================"
echo "[B] CLIP-FT text-dot-product (JA, best ckpt)"
echo "============================================================"
CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON_CMD[@]}" -m classification.models.clip_zeroshot \
  --csv "$VAL_CSV" \
  --out "$OUT_DIR/clip_ft_textdot_preds.csv" \
  --batch_size "$BS" --threshold "$THRESHOLD" \
  --ckpt_dir "$CLIP_CKPT_DIR" \
  --image_root "$IMAGE_ROOT"

# 評価
echo "============================================================"
echo "[eval] text-dot-product predictions"
echo "============================================================"
"${PYTHON_CMD[@]}" - <<'PY'
import json
from pathlib import Path
from classification.evaluate import evaluate_model

for name in ['clip_base_textdot', 'clip_ft_textdot']:
    p = Path(f'classification/results/{name}_preds.csv')
    if not p.exists():
        print(f'  [skip] {p}'); continue
    res = evaluate_model(str(p))
    out = Path(f'classification/results/{name}_metrics.json')
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved: {out}')
PY

echo "DONE."
