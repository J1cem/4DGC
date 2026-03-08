#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljs/4DGC"
SCENE=""
ZIP_PATH=""
OVERWRITE=0
CAMERA_MODE="SINGLE"
CAMERA_MODEL="SIMPLE_PINHOLE"

usage() {
  cat <<EOF
Usage: $0 <scene_name> [--zip /path/to/<scene>.zip] [--overwrite] [--camera-mode SINGLE|PER_IMAGE] [--camera-model SIMPLE_PINHOLE|PINHOLE]
Example:
  $0 coffee_martini --zip /data/coffee_martini.zip --overwrite
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SCENE="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip)
      ZIP_PATH="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --camera-mode)
      CAMERA_MODE="$2"
      shift 2
      ;;
    --camera-model)
      CAMERA_MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

SCENE_DIR="${ROOT}/dataset/${SCENE}"

if [[ -n "${ZIP_PATH}" ]]; then
  if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "Zip not found: ${ZIP_PATH}"
    exit 1
  fi
  mkdir -p "${ROOT}/dataset"
  echo "[0/3] Unzip ${ZIP_PATH} -> ${ROOT}/dataset"
  unzip -oq "${ZIP_PATH}" -d "${ROOT}/dataset"
fi

if [[ ! -d "${SCENE_DIR}" ]]; then
  echo "Scene dir not found: ${SCENE_DIR}"
  echo "Put N3DV scene at dataset/${SCENE} or pass --zip"
  exit 1
fi

if [[ ! -d "${ROOT}/.venv310" ]]; then
  echo "Missing venv: ${ROOT}/.venv310"
  exit 1
fi

. "${ROOT}/.venv310/bin/activate"

if [[ ${OVERWRITE} -eq 1 ]]; then
  PREP_OVERWRITE="--overwrite"
else
  PREP_OVERWRITE=""
fi

echo "[1/3] Extract multi-view videos -> colmap_x/images"
if compgen -G "${SCENE_DIR}/colmap_*" > /dev/null && [[ ${OVERWRITE} -eq 0 ]]; then
  echo "skip extraction (found existing colmap_*; use --overwrite to rebuild)"
else
  python "${ROOT}/scripts/prepare_mv_videos.py" --scene_dir "${SCENE_DIR}" ${PREP_OVERWRITE}
fi

echo "[2/3] Reconstruct colmap_0/sparse/0"
if [[ ${OVERWRITE} -eq 1 || ! -f "${SCENE_DIR}/colmap_0/sparse/0/images.bin" ]]; then
  python "${ROOT}/scripts/reconstruct_colmap0.py" \
    --scene_dir "${SCENE_DIR}" \
    --frame 0 \
    --camera_mode "${CAMERA_MODE}" \
    --camera_model "${CAMERA_MODEL}" \
    --device cpu
else
  echo "skip reconstruction (found sparse/0/images.bin)"
fi

echo "[3/3] Build paper-aligned cfg"
python "${ROOT}/scripts/n3dv_make_cfg.py" --root "${ROOT}" --scene "${SCENE}"

echo "Done."
