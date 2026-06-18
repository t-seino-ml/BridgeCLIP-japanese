#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
#LOGS="${ROOT}/logs_sweep"
#VAL="${ROOT}/val_data/val_clean.csv"
#OUTDIR="${ROOT}/eval_results_best"
LOGS="${ROOT}/logs_sweep_item"
VAL="${ROOT}/val_item_data/val_clean_base.csv"
OUTDIR="${ROOT}/eval_results_best_item"
mkdir -p "${OUTDIR}"

# --- best ckpt一覧作成 ---
mapfile -t RUNS < <(find "${LOGS}" -maxdepth 1 -mindepth 1 -type d -print | sort)
RUNS=($(for r in "${RUNS[@]}"; do [[ -d "$r/checkpoints" ]] && echo "$r"; done))

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "[ERR] No run dirs with checkpoints under ${LOGS}"
  exit 1
fi

BEST_TSV="${OUTDIR}/best_ckpts.tsv"
: > "${BEST_TSV}"
for r in "${RUNS[@]}"; do
  uv run python pick_best_ckpt.py --run-dir "$r" >> "${BEST_TSV}"
done

echo "[INFO] best list: ${BEST_TSV}"
head -n 5 "${BEST_TSV}" || true

# --- GPUプール（ロックで管理）---
POOL_DIR="${OUTDIR}/_gpu_pool"
mkdir -p "${POOL_DIR}"
POOL_FILE="${POOL_DIR}/pool.txt"
LOCK_FILE="${POOL_DIR}/lock"

# 初期化（毎回作り直す）
printf "0\n1\n2\n3\n" > "${POOL_FILE}"

acquire_gpu () {
  local gpu=""
  while [[ -z "${gpu}" ]]; do
    # flock が無ければ util-linux を入れる必要あり（普通は入ってる）
    gpu="$(flock -x "${LOCK_FILE}" bash -c 'if [[ -s "'"${POOL_FILE}"'" ]]; then head -n 1 "'"${POOL_FILE}"'"; sed -i "1d" "'"${POOL_FILE}"'"; fi')"
    [[ -z "${gpu}" ]] && sleep 1
  done
  echo "${gpu}"
}

release_gpu () {
  local gpu="$1"
  flock -x "${LOCK_FILE}" bash -c 'echo "'"${gpu}"'" >> "'"${POOL_FILE}"'"'
}

run_one () {
  local run_name="$1"
  local ckpt="$2"
  local best_ep="$3"
  local best_loss="$4"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[SKIP] ${run_name} ckpt missing: ${ckpt}"
    return 0
  fi

  local gpu
  gpu="$(acquire_gpu)"
  local out_json="${OUTDIR}/${run_name}.json"
  local out_log="${OUTDIR}/${run_name}.log"

  echo "[GPU ${gpu}] START ${run_name} ckpt=$(basename "${ckpt}") best_ep=${best_ep} best_loss=${best_loss}"
  (
    set -e
    CUDA_VISIBLE_DEVICES="${gpu}" \
      uv run python eval_retrieval.py \
        --ckpt "${ckpt}" \
        --val-csv "${VAL}" \
        --out-json "${out_json}" \
        --batch-size 256 \
        --num-workers 8 \
      > "${out_log}" 2>&1
  )
  echo "[GPU ${gpu}] DONE  ${run_name}"
  release_gpu "${gpu}"
}

# --- 最大4並列で回す（GPUプール方式なので安全）---
pids=()
while IFS=$'\t' read -r run_name ckpt best_ep best_loss; do
  # 空行ガード
  [[ -z "${run_name}" ]] && continue
  run_one "${run_name}" "${ckpt}" "${best_ep}" "${best_loss}" &
  pids+=($!)
done < "${BEST_TSV}"

# wait all
fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

echo "[ALL DONE] ${OUTDIR}"
exit "${fail}"
