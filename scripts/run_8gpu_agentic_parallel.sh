#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$ROOT_DIR/scripts"
LOCAL_ENV="$ROOT_DIR/config/local.env"

if [[ -f "$LOCAL_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$LOCAL_ENV"
  set +a
fi

MODE="${1:-smoke}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_NAME="${RUN_NAME:-agentic_pipeline_v21_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_NAME}"
INPUT_RESULTS="${INPUT_RESULTS:-${MOT_INPUT_RESULTS:-$ROOT_DIR/data/eval_samples_v7_reference.json}}"
VL_MODEL_ARG="${VL_MODEL:-${MOT_VL_MODEL:-}}"
VIDEO_DIRS_ARG="${VIDEO_DIRS:-}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "$MODE" == "full" ]]; then
  DEFAULT_MAX_ROUNDS=3
  DEFAULT_GRID_MAX_FRAMES=128
  DEFAULT_VL_NFRAMES=0
  DEFAULT_N_PER_TYPE=10
else
  DEFAULT_MAX_ROUNDS=1
  DEFAULT_GRID_MAX_FRAMES=32
  DEFAULT_VL_NFRAMES=8
  DEFAULT_N_PER_TYPE=1
fi

MAX_ROUNDS="${MAX_ROUNDS:-$DEFAULT_MAX_ROUNDS}"
GRID_MAX_FRAMES="${GRID_MAX_FRAMES:-$DEFAULT_GRID_MAX_FRAMES}"
VL_NFRAMES="${VL_NFRAMES:-$DEFAULT_VL_NFRAMES}"
N_PER_TYPE="${N_PER_TYPE:-$DEFAULT_N_PER_TYPE}"

if [[ ! -f "$INPUT_RESULTS" ]]; then
  echo "Input manifest not found: $INPUT_RESULTS" >&2
  echo "Set INPUT_RESULTS/MOT_INPUT_RESULTS or place data/eval_samples_v7_reference.json in the repo." >&2
  exit 1
fi

if [[ -e "$OUTPUT_DIR" && "$OVERWRITE" != "1" ]]; then
  echo "Output directory already exists: $OUTPUT_DIR" >&2
  echo "Set OUTPUT_DIR to a new path or OVERWRITE=1 to reuse it." >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
if [[ "$OVERWRITE" == "1" ]]; then
  echo "Cleaning previous run artifacts under: $OUTPUT_DIR"
  find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1     \( -name 'gpu*.log' -o -name 'gpu[0-9]*' -o -name 'parsed_summary.json' \)     -exec rm -rf {} +
fi

if command -v rocm-smi >/dev/null 2>&1; then
  echo "ROCm GPUs:"
  rocm-smi -i
fi

echo "Running mode: $MODE"
echo "NUM_GPUS=$NUM_GPUS, MAX_ROUNDS=$MAX_ROUNDS, GRID_MAX_FRAMES=$GRID_MAX_FRAMES, VL_NFRAMES=$VL_NFRAMES, N_PER_TYPE=$N_PER_TYPE"
echo "INPUT_RESULTS=$INPUT_RESULTS"
echo "OUTPUT_DIR=$OUTPUT_DIR"

COMMON_ARGS=(
  --num_gpus "$NUM_GPUS"
  --n_per_type "$N_PER_TYPE"
  --max_rounds "$MAX_ROUNDS"
  --grid_max_frames "$GRID_MAX_FRAMES"
  --vl-nframes "$VL_NFRAMES"
  --input_results "$INPUT_RESULTS"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$MODE" == "full" ]]; then
  COMMON_ARGS+=(--full)
fi
if [[ -n "$VL_MODEL_ARG" ]]; then
  COMMON_ARGS+=(--vl-model "$VL_MODEL_ARG")
fi

VIDEO_ARGS=()
if [[ -n "$VIDEO_DIRS_ARG" ]]; then
  OLDIFS="$IFS"
  IFS=':' read -r -a VIDEO_DIR_ARRAY <<< "$VIDEO_DIRS_ARG"
  IFS="$OLDIFS"
  for dir_path in "${VIDEO_DIR_ARRAY[@]}"; do
    [[ -n "$dir_path" ]] && VIDEO_ARGS+=(--video-dir "$dir_path")
  done
fi

PIDS=()
for gid in $(seq 0 $((NUM_GPUS - 1))); do
  LOG_PATH="$OUTPUT_DIR/gpu${gid}.log"
  (
    CUDA_VISIBLE_DEVICES="$gid"     python3 "$SCRIPTS_DIR/grid64_agentic_pipeline.py"       --gpu_id "$gid"       --device "cuda:0"       "${COMMON_ARGS[@]}"       "${VIDEO_ARGS[@]}"
  ) > "$LOG_PATH" 2>&1 &
  PIDS+=("$!")
done

echo "Waiting for all GPU workers..."
wait "${PIDS[@]}"

echo "Parsing overall results from logs..."
python3 "$SCRIPTS_DIR/overall_results_parser.py" --base "$OUTPUT_DIR" --num_gpus "$NUM_GPUS"
