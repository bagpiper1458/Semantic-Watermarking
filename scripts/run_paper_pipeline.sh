#!/usr/bin/env bash
set -euo pipefail

WM_TYPE="${1:-HSQR}"
DATASET_ID="${2:-coco}"
OUTPUT_DIR="${3:-outputs}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}/src"

echo "[1/4] Generate images"
python generate.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}"

echo "[2/4] Compute image quality metrics"
python metric.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}"

echo "[3/4] Run diffusion regeneration attack"
python diff_attack/diff_wm_attack.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}"

echo "[4/4] Detect and identify"
python detect.py \
  --wm_type "${WM_TYPE}" \
  --dataset_id "${DATASET_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --no_save_inverted

echo
echo "Experiment outputs:"
echo "  ${REPO_ROOT}/${OUTPUT_DIR}/${DATASET_ID}/${WM_TYPE}"
