#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljs/4DGC"
SCENES=(coffee_martini cook_spinach cut_roasted_beef flame_steak sear_steak)

for scene in "${SCENES[@]}"; do
  echo "==================== ${scene} ===================="
  if [[ ! -d "${ROOT}/dataset/${scene}" ]]; then
    "${ROOT}/scripts/download_n3dv_scene.sh" "${scene}"
  fi
  "${ROOT}/scripts/prepare_n3dv_scene.sh" "${scene}"
  "${ROOT}/scripts/run_n3dv_pipeline.sh" "${scene}"
done
