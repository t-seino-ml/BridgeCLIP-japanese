#!/usr/bin/env bash
# EN 用 train_subsets を作成（10k〜120k + clean）。
# unified_train_user_en.csv (130930 件) からランダムサンプルする。
#
# Output: train_subsets_user_en/train_{10k,...,120k,clean}_base.csv
#         val_item_data_en/val_clean_base.csv
#
# Note: image_root に依存しないように image 列はそのまま保持（学習時に --image_root で上書き想定）。

set -euo pipefail

PYTHON="${PYTHON:-python3}"
read -r -a PYTHON_CMD <<< "$PYTHON"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_CSV_FULL="${ROOT}/classification/results/unified_train_user_en.csv"
VAL_CSV_FULL="${ROOT}/classification/results/unified_val_user_en.csv"
OUT_TRAIN_DIR="${ROOT}/train_subsets_user_en"
OUT_VAL_DIR="${ROOT}/val_item_data_en"
mkdir -p "$OUT_TRAIN_DIR" "$OUT_VAL_DIR"

# val は base 形式（image,text のみ）で保存
"${PYTHON_CMD[@]}" - <<PY
import pandas as pd
df = pd.read_csv("$VAL_CSV_FULL")
df[['image','text']].to_csv("$OUT_VAL_DIR/val_clean_base.csv", index=False)
print(f"saved val_clean_base.csv: {len(df)} rows")
PY

# train サブセット作成
for n in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 110000 120000; do
  label="$((n / 1000))k"
  out="$OUT_TRAIN_DIR/train_${label}_base.csv"
  "${PYTHON_CMD[@]}" - <<PY
import pandas as pd
df = pd.read_csv("$TRAIN_CSV_FULL")
sub = df[['image','text']].sample(n=$n, random_state=42).reset_index(drop=True)
sub.to_csv("$out", index=False)
print(f"saved $out: {len(sub)} rows")
PY
done

# clean (全量)
"${PYTHON_CMD[@]}" - <<PY
import pandas as pd
df = pd.read_csv("$TRAIN_CSV_FULL")
df[['image','text']].to_csv("$OUT_TRAIN_DIR/train_clean_base.csv", index=False)
print(f"saved train_clean_base.csv: {len(df)} rows")
PY

echo "DONE."
