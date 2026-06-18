#!/usr/bin/env bash
# EN 版: A + B の text-dot-product 分類 (CLIP-base / CLIP-FT × text-dot)
#
# Usage:
#   IMAGE_ROOT=$(pwd)/images bash classification/run_clip_textdot_en.sh

set -euo pipefail
export LABEL_LANG=en

PYTHON="${PYTHON:-python3}"
read -r -a PYTHON_CMD <<< "$PYTHON"
: "${IMAGE_ROOT:?IMAGE_ROOT must be set}"
VAL_CSV="${VAL_CSV:-classification/results/unified_val_user_en.csv}"
GPU="${GPU:-0}"
THRESHOLD="${THRESHOLD:-0.3}"
BS="${BS:-64}"
# EN CLIP-FT の ckpt dir（再学習後に決まる）。事前に存在しない場合 B はスキップ。
CLIP_CKPT_DIR="${CLIP_CKPT_DIR:-logs_classification_en/clip_en_v2/checkpoints}"

OUT_DIR="classification/results_en"
mkdir -p "$OUT_DIR"

# A: ベース CLIP の text-dot (EN)
echo "============================================================"
echo "[A] CLIP-base text-dot-product (EN)"
echo "============================================================"
CUDA_VISIBLE_DEVICES="$GPU" LABEL_LANG=en "${PYTHON_CMD[@]}" -m classification.models.clip_zeroshot \
  --csv "$VAL_CSV" \
  --out "$OUT_DIR/clip_base_textdot_en_preds.csv" \
  --batch_size "$BS" --threshold "$THRESHOLD" \
  --image_root "$IMAGE_ROOT"

# B: CLIP-FT (EN best) の text-dot
if [[ -d "$CLIP_CKPT_DIR" ]]; then
  echo "============================================================"
  echo "[B] CLIP-FT text-dot-product (EN, best ckpt)"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU" LABEL_LANG=en "${PYTHON_CMD[@]}" -m classification.models.clip_zeroshot \
    --csv "$VAL_CSV" \
    --out "$OUT_DIR/clip_ft_textdot_en_preds.csv" \
    --batch_size "$BS" --threshold "$THRESHOLD" \
    --ckpt_dir "$CLIP_CKPT_DIR" \
    --image_root "$IMAGE_ROOT"
else
  echo "[B skipped] $CLIP_CKPT_DIR not found. Train EN CLIP-FT first (retrain_clip_short_en.sh)."
fi

# 評価
LABEL_LANG=en "${PYTHON_CMD[@]}" - <<'PY'
import json
from pathlib import Path
from classification.evaluate_en import evaluate_model
for name in ['clip_base_textdot', 'clip_ft_textdot']:
    p = Path(f'classification/results_en/{name}_en_preds.csv')
    if not p.exists():
        print(f'  [skip] {p}'); continue
    res = evaluate_model(str(p))
    out = Path(f'classification/results_en/{name}_en_metrics.json')
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved: {out}')
PY

echo "DONE."
