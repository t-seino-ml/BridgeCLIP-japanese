#!/bin/bash
# EN CLIP-FT 再学習（10 epoch、save-frequency 1）
#
# JA で完成した `retrain_clip_short.sh` の EN 同等版。
# 完了後 `logs_classification_en/clip_en_v2/checkpoints/epoch_{1..10}.pt` が並ぶ。
# best ckpt は results.jsonl から val_loss 最小を find_best_checkpoint で自動採用。
#
# Usage:
#   bash classification/retrain_clip_short_en.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

GPU="${GPU:-3}"
NAME="${NAME:-clip_en_v2}"
EPOCHS="${EPOCHS:-10}"

echo "[config] GPU=$GPU  NAME=$NAME  EPOCHS=$EPOCHS"

CUDA_VISIBLE_DEVICES="$GPU" uv run python -m open_clip_train.main \
  --train-data "${ROOT_DIR}/classification/results/unified_train_user_en.csv" \
  --val-data   "${ROOT_DIR}/classification/results/unified_val_user_en.csv" \
  --dataset-type csv \
  --csv-separator "," \
  --csv-img-key image \
  --csv-caption-key text \
  --model ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --batch-size 128 \
  --epochs "$EPOCHS" \
  --lr 1e-4 \
  --wd 0.1 \
  --warmup 1000 \
  --precision amp \
  --workers 8 \
  --report-to tensorboard \
  --save-frequency 1 \
  --zeroshot-frequency 0 \
  --logs "${ROOT_DIR}/logs_classification_en" \
  --name "$NAME"

echo "[DONE] EN CLIP-FT ckpts at ${ROOT_DIR}/logs_classification_en/${NAME}/checkpoints/"
echo
echo "次のステップ: best epoch 確認"
echo "  uv run python - <<'PY'"
echo "  import json; from pathlib import Path"
echo "  ck = Path('logs_classification_en/${NAME}/checkpoints')"
echo "  ent=[]"
echo "  with open(ck/'results.jsonl') as f:"
echo "      for l in f:"
echo "          d=json.loads(l); ent.append((d['clip_val_loss'], d['epoch']))"
echo "  ent.sort()"
echo "  for l,e in ent[:5]: print(f'epoch {e}: val_loss={l:.4f}')"
echo "  PY"
