#!/usr/bin/env bash
set -euo pipefail

WM_TYPE="${1:-Tree-Ring}"
DATASET_ID="${2:-coco}"
FIXED_KEY="${3:-0}"
OUTPUT_DIR="${4:-outputs_fixedkey_k0}"
SHARED_CLEAN_DIR="${5:-}"
MAX_TRIALS="${MAX_TRIALS:-1000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/src"

if [[ -z "${SHARED_CLEAN_DIR}" ]]; then
  SHARED_CLEAN_DIR="../outputs/${DATASET_ID}/${WM_TYPE}/img_pil"
fi

echo "[1/2] Generate fixed-key watermarked images"
python generate_fixed_key_fast.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --fixed_key "${FIXED_KEY}" \
  --output_dir "${OUTPUT_DIR}" \
  --shared_clean_dir "${SHARED_CLEAN_DIR}"

echo "[2/2] Detect fixed-key clean case"
python detect_avg_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --only_clean \
  --max_trials "${MAX_TRIALS}" \
  --no_save_inverted

echo
echo "Fixed-key outputs:"
echo "  ${REPO_ROOT}/${OUTPUT_DIR}/${DATASET_ID}/${WM_TYPE}"
