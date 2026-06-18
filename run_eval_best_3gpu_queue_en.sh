#!/usr/bin/env bash
# EN sweep 評価（GPU 0/1/3、GPU 2 スキップ）
# 各 sweep run の best epoch を pick_best_ckpt.py で選び、eval_retrieval.py で評価。
#
# Usage:
#   ROOT=$(pwd) bash run_eval_best_3gpu_queue_en.sh

set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
LOGS="${ROOT}/logs_sweep_en"
VAL="${ROOT}/val_item_data_en/val_clean_base.csv"
OUTDIR="${ROOT}/eval_results_best_en_sweep"
mkdir -p "${OUTDIR}"

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

# GPU プール（GPU 0/1/3 のみ）
POOL_DIR="${OUTDIR}/_gpu_pool"
mkdir -p "${POOL_DIR}"
POOL_FILE="${POOL_DIR}/pool.txt"
LOCK_FILE="${POOL_DIR}/lock"
printf "0\n1\n3\n" > "${POOL_FILE}"

acquire_gpu () {
  local gpu=""
  while [[ -z "${gpu}" ]]; do
    gpu="$(flock -x "${LOCK_FILE}" bash -c 'if [[ -s "'"${POOL_FILE}"'" ]]; then head -n 1 "'"${POOL_FILE}"'"; sed -i "1d" "'"${POOL_FILE}"'"; fi')"
    [[ -z "${gpu}" ]] && sleep 1
  done
  echo "${gpu}"
}
release_gpu () {
  flock -x "${LOCK_FILE}" bash -c 'echo "'"$1"'" >> "'"${POOL_FILE}"'"'
}

run_one () {
  local run_name="$1" ckpt="$2" best_ep="$3" best_loss="$4"
  [[ -f "${ckpt}" ]] || { echo "[SKIP] ${run_name} ckpt missing"; return 0; }
  local gpu; gpu="$(acquire_gpu)"
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

pids=()
while IFS=$'\t' read -r run_name ckpt best_ep best_loss; do
  [[ -z "${run_name}" ]] && continue
  run_one "${run_name}" "${ckpt}" "${best_ep}" "${best_loss}" &
  pids+=($!)
done < "${BEST_TSV}"

fail=0
for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done

echo "[ALL DONE] ${OUTDIR}"
exit "${fail}"
