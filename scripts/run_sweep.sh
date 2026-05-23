#!/bin/bash
set -euo pipefail

SWEEP_FILE="${1:?usage: run_sweep.sh <sweep.yaml>}"
: "${MAIN_DIR:?MAIN_DIR must be set}"

SWEEP_NAME="$(basename "${SWEEP_FILE}" .yaml)"
GEN_DIR="${MAIN_DIR}/args/_generated"
mkdir -p "${GEN_DIR}" "${MAIN_DIR}/logs"

MANIFEST="${GEN_DIR}/${SWEEP_NAME}_manifest.txt"
python3 "${MAIN_DIR}/scripts/expand_sweep.py" "${SWEEP_FILE}" "${GEN_DIR}" > "${MANIFEST}"

N=$(wc -l < "${MANIFEST}")
echo "Submitting 1 sbatch job with ${N} sequential runs from ${SWEEP_FILE}"

sbatch \
  --job-name="apertus-${SWEEP_NAME}" \
  --output="${MAIN_DIR}/logs/%j_${SWEEP_NAME}.out" \
  --error="${MAIN_DIR}/logs/%j_${SWEEP_NAME}.err" \
  "${MAIN_DIR}/scripts/run_sbatch.sh" "${MANIFEST}"
