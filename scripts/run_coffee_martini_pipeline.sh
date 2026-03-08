#!/usr/bin/env bash
set -eu

ROOT="/home/ljs/4DGC"
SCENE="${ROOT}/dataset/coffee_martini"
INIT_OUT="${ROOT}/outputs/coffee_martini_init"
FRAME_OUT="${ROOT}/outputs/coffee_martini_frames"
MEM_OUT="${ROOT}/mem/coffee_martini.pth"
CFG="${ROOT}/test/coffee_martini/cfg_args.json"

. "${ROOT}/.venv310/bin/activate"
export LD_LIBRARY_PATH="${ROOT}/.venv310/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"

mkdir -p "${ROOT}/outputs" "${ROOT}/mem"

INIT_CKPT="${INIT_OUT}/chkpnt_latest.pth"
if [ ! -f "${INIT_OUT}/point_cloud/iteration_10000/point_cloud.ply" ]; then
  echo "[1/3] train_initial ..."
  if [ -f "${INIT_CKPT}" ]; then
    echo "  resume from ${INIT_CKPT}"
    python "${ROOT}/train_initial.py" \
      -s "${SCENE}/colmap_0" \
      -o "${INIT_OUT}" \
      -m "${INIT_OUT}" \
      --eval \
      -r 2 \
      --start_checkpoint "${INIT_CKPT}"
  else
    python "${ROOT}/train_initial.py" \
      -s "${SCENE}/colmap_0" \
      -o "${INIT_OUT}" \
      -m "${INIT_OUT}" \
      --eval \
      -r 2
  fi
else
  echo "[1/3] skip train_initial (found iteration_10000)"
fi

if [ ! -f "${MEM_OUT}" ]; then
  echo "[2/3] Motion Grid warmup ..."
  python "${ROOT}/scripts/Motion_Grid_warmup.py" \
    --pcd_path "${INIT_OUT}/point_cloud/iteration_10000/point_cloud.ply" \
    --q 1 \
    --output_path "${MEM_OUT}"
else
  echo "[2/3] skip warmup (found ${MEM_OUT})"
fi

FRAME_END="$(python -c "import json; c=json.load(open('${CFG}','r')); print(int(c.get('frame_end', 300)))")"
FINAL_ITER="$(python -c "import json; c=json.load(open('${CFG}','r')); print(int(c.get('iterations', 400)) + int(c.get('iterations_s2', 100)))")"

RESUME_START=1
if [ -d "${FRAME_OUT}" ]; then
  for d in "${FRAME_OUT}"/colmap_*; do
    [ -d "${d}" ] || continue
    name="$(basename "${d}")"
    idx="${name#colmap_}"
    case "${idx}" in
      ''|*[!0-9]*) continue ;;
    esac
    ply="${d}/point_cloud/iteration_${FINAL_ITER}/point_cloud.ply"
    if [ -f "${ply}" ] && [ "${idx}" -ge "${RESUME_START}" ]; then
      RESUME_START=$((idx + 1))
    fi
  done
fi

if [ "${RESUME_START}" -ge "${FRAME_END}" ]; then
  echo "[3/3] skip train_frames (already finished up to frame index $((FRAME_END-1)))"
  exit 0
fi

MODEL_IN="${INIT_OUT}"
if [ "${RESUME_START}" -gt 1 ]; then
  MODEL_IN="${FRAME_OUT}/colmap_$((RESUME_START-1))"
fi

echo "[3/3] train_frames ... (resume frame_start=${RESUME_START}, model=${MODEL_IN})"
python "${ROOT}/train_frames.py" \
  --read_config \
  --config_path "${CFG}" \
  -o "${FRAME_OUT}" \
  -m "${MODEL_IN}" \
  -v "${SCENE}" \
  --image images \
  --frame_start "${RESUME_START}" \
  --frame_end "${FRAME_END}"
