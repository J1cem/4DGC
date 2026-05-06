#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ljs/4DGC"
SCENE="${1:-}"
EXP_NAME="${2:-${SCENE}}"

if [[ -z "${SCENE}" ]]; then
  echo "Usage: $0 <scene_name> [exp_name]"
  exit 1
fi

SCENE_DIR="${ROOT}/dataset/${SCENE}"
INIT_OUT="${ROOT}/outputs/${EXP_NAME}_init"
FRAME_OUT="${ROOT}/outputs/${EXP_NAME}_frames"
MEM_OUT="${ROOT}/mem/${EXP_NAME}.pth"
CFG="${ROOT}/test/${EXP_NAME}/cfg_args.json"

if [[ ! -d "${SCENE_DIR}" ]]; then
  echo "Scene not found: ${SCENE_DIR}"
  echo "Run scripts/prepare_n3dv_scene.sh ${SCENE} first."
  exit 1
fi

if [[ ! -f "${SCENE_DIR}/colmap_0/sparse/0/images.bin" ]]; then
  echo "Missing COLMAP sparse model: ${SCENE_DIR}/colmap_0/sparse/0"
  echo "Run scripts/prepare_n3dv_scene.sh ${SCENE} --overwrite first."
  exit 1
fi

. "${ROOT}/.venv310/bin/activate"
export LD_LIBRARY_PATH="${ROOT}/.venv310/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

mkdir -p "${ROOT}/outputs" "${ROOT}/mem" "${ROOT}/test/${EXP_NAME}"

if [[ ! -f "${CFG}" ]]; then
  python "${ROOT}/scripts/n3dv_make_cfg.py" --root "${ROOT}" --scene "${SCENE}" --exp_name "${EXP_NAME}"
fi

LAMBDA_RD_BASE="$(python - <<PY
import json
c=json.load(open('${CFG}','r'))
print(float(c.get('lambda_rd_base', 0.0)))
PY
)"
Q_VAL="$(python - <<PY
import json
c=json.load(open('${CFG}','r'))
print(int(c.get('q', 1)))
PY
)"
FRAME_END="$(python - <<PY
import json
c=json.load(open('${CFG}','r'))
print(int(c.get('frame_end', 300)))
PY
)"
FINAL_ITER="$(python - <<PY
import json
c=json.load(open('${CFG}','r'))
print(int(c.get('iterations', 400)))
PY
)"

INIT_CKPT="${INIT_OUT}/chkpnt_latest.pth"
if [[ ! -f "${INIT_OUT}/point_cloud/iteration_10000/point_cloud.ply" ]]; then
  echo "[1/3] train_initial (${SCENE}, exp=${EXP_NAME}) ..."
  if [[ -f "${INIT_CKPT}" ]]; then
    echo "  resume from ${INIT_CKPT}"
    python "${ROOT}/train_initial.py" \
      -s "${SCENE_DIR}/colmap_0" \
      -o "${INIT_OUT}" \
      -m "${INIT_OUT}" \
      --eval \
      -r 2 \
      --lambda_rd_base "${LAMBDA_RD_BASE}" \
      --start_checkpoint "${INIT_CKPT}"
  else
    python "${ROOT}/train_initial.py" \
      -s "${SCENE_DIR}/colmap_0" \
      -o "${INIT_OUT}" \
      -m "${INIT_OUT}" \
      --eval \
      -r 2 \
      --lambda_rd_base "${LAMBDA_RD_BASE}"
  fi
else
  echo "[1/3] skip train_initial (found iteration_10000)"
fi

if [[ ! -f "${MEM_OUT}" ]]; then
  echo "[2/3] Motion Grid warmup (${SCENE}, exp=${EXP_NAME}) ..."
  python "${ROOT}/scripts/Motion_Grid_warmup.py" \
    --pcd_path "${INIT_OUT}/point_cloud/iteration_10000/point_cloud.ply" \
    --q "${Q_VAL}" \
    --output_path "${MEM_OUT}"
else
  echo "[2/3] skip warmup (found ${MEM_OUT})"
fi

RESUME_START=1
if [[ -d "${FRAME_OUT}" ]]; then
  for d in "${FRAME_OUT}"/colmap_*; do
    [[ -d "${d}" ]] || continue
    name="$(basename "${d}")"
    idx="${name#colmap_}"
    case "${idx}" in
      ''|*[!0-9]*) continue ;;
    esac
    ply="${d}/point_cloud/iteration_${FINAL_ITER}/point_cloud.ply"
    if [[ -f "${ply}" && "${idx}" -ge "${RESUME_START}" ]]; then
      RESUME_START=$((idx + 1))
    fi
  done
fi

if [[ "${RESUME_START}" -ge "${FRAME_END}" ]]; then
  echo "[3/3] skip train_frames (already finished up to frame index $((FRAME_END-1)))"
  exit 0
fi

MODEL_IN="${INIT_OUT}"
if [[ "${RESUME_START}" -gt 1 ]]; then
  MODEL_IN="${FRAME_OUT}/colmap_$((RESUME_START-1))"
fi

echo "[3/3] train_frames (${SCENE}, exp=${EXP_NAME}) ... (resume frame_start=${RESUME_START}, model=${MODEL_IN})"
python "${ROOT}/train_frames.py" \
  --read_config \
  --config_path "${CFG}" \
  -o "${FRAME_OUT}" \
  -m "${MODEL_IN}" \
  -v "${SCENE_DIR}" \
  --image images \
  --frame_start "${RESUME_START}" \
  --frame_end "${FRAME_END}"
