#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljs/4DGC"
SCENE="${1:-}"

if [[ -z "${SCENE}" ]]; then
  echo "Usage: $0 <scene_name>"
  exit 1
fi

LAMBDAS=(0.0005 0.001 0.002 0.004 0.008)

for lam in "${LAMBDAS[@]}"; do
  tag="l$(echo "${lam}" | tr '.' 'p')"
  exp="${SCENE}_${tag}"
  echo "========== ${SCENE} | lambda_rd_base=${lam} | exp=${exp} =========="

  source "${ROOT}/.venv310/bin/activate"
  python "${ROOT}/scripts/n3dv_make_cfg.py" \
    --root "${ROOT}" \
    --scene "${SCENE}" \
    --exp_name "${exp}" \
    --lambda_rd_base "${lam}"

  bash "${ROOT}/scripts/run_n3dv_pipeline.sh" "${SCENE}" "${exp}"
done
