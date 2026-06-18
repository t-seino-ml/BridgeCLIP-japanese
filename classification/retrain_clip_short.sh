#!/bin/bash
# CLIP ファインチューニング再学習（10 epoch・early-stop 想定）
#
# 目的: 削除された epoch 5 best モデルを復元する。
# 既存 `bridgeclip_vitb32_unified` は触らず、別 name で出力する。
#
# 出力: logs_classification/bridgeclip_vitb32_unified_v2/
#         checkpoints/epoch_1..10.pt
#         checkpoints/results.jsonl
#
# 完了後、retrieval_eval.py / clip_finetuned_knn.py が
# results.jsonl で val_loss 最小の epoch を自動採用する。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# GPU 選択（既定: GPU 0、CUDA_VISIBLE_DEVICES と等価）
GPU="${GPU:-0}"
NAME="${NAME:-bridgeclip_vitb32_unified_v2}"
EPOCHS="${EPOCHS:-10}"

echo "[config] GPU=$GPU  NAME=$NAME  EPOCHS=$EPOCHS"

CUDA_VISIBLE_DEVICES="$GPU" uv run python -m open_clip_train.main \
  --train-data "${ROOT_DIR}/classification/results/unified_train_user.csv" \
  --val-data   "${ROOT_DIR}/classification/results/unified_val_user.csv" \
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
  --logs "${ROOT_DIR}/logs_classification" \
  --name "$NAME"

echo "[DONE] checkpoints at ${ROOT_DIR}/logs_classification/${NAME}/checkpoints/"
echo
echo "次のステップ: best epoch を確認"
echo "  uv run python - <<'PY'"
echo "  import json"
echo "  from pathlib import Path"
echo "  ck = Path('logs_classification/${NAME}/checkpoints')"
echo "  ent = []"
echo "  with open(ck/'results.jsonl') as f:"
echo "      for line in f:"
echo "          d = json.loads(line)"
echo "          ent.append((d['clip_val_loss'], d['epoch']))"
echo "  ent.sort()"
echo "  for l,e in ent[:5]:"
echo "      print(f'epoch {e}: val_loss={l:.4f}')"
echo "  PY"
