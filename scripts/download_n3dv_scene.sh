#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljs/4DGC"
SCENE="${1:-}"

if [[ -z "${SCENE}" ]]; then
  echo "Usage: $0 <scene_name>"
  echo "Supported: coffee_martini cook_spinach cut_roasted_beef flame_steak sear_steak"
  exit 1
fi

case "${SCENE}" in
  coffee_martini|cook_spinach|cut_roasted_beef|flame_steak|sear_steak)
    ZIP_NAME="${SCENE}.zip"
    ;;
  flame_salmon)
    echo "flame_salmon is split in release assets (multiple parts). Download manually first, then run prepare_n3dv_scene.sh."
    exit 1
    ;;
  *)
    echo "Unsupported scene: ${SCENE}"
    exit 1
    ;;
esac

URL="https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/${ZIP_NAME}"
ARCHIVE_DIR="${ROOT}/dataset/_archives"
mkdir -p "${ARCHIVE_DIR}" "${ROOT}/dataset"

ZIP_PATH="${ARCHIVE_DIR}/${ZIP_NAME}"

echo "Downloading ${URL}"
wget -c "${URL}" -O "${ZIP_PATH}"

echo "Unzipping ${ZIP_PATH} -> ${ROOT}/dataset"
unzip -oq "${ZIP_PATH}" -d "${ROOT}/dataset"

echo "Done: ${ROOT}/dataset/${SCENE}"
