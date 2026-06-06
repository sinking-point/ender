#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/replay_submission_in_kaggle_docker.sh BUNDLE RECORD [options]

Arguments:
  BUNDLE   Submission bundle directory or .tar.gz created by package_kaggle_submission.py
  RECORD   Saved Kaggle episode JSON to replay

Options:
  --image IMAGE        Docker image to use (default: gcr.io/kaggle-images/python)
  --out-dir DIR        Output directory for per-seat jsonl logs
  --step-limit N       Replay at most N steps
  --seats LIST         Comma-separated seat indices (default: 0,1,2,3)
  --docker-arg ARG     Extra docker run arg; may be passed multiple times
  --no-pull            Skip docker pull

This launches one Docker container per seat with the packaged submission mounted
read-only and network disabled, then replays the saved episode through the
submission's main.py in a persistent worker process.
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

BUNDLE=$1
RECORD=$2
shift 2

IMAGE="gcr.io/kaggle-images/python"
OUT_DIR=""
STEP_LIMIT=""
SEATS="0,1,2,3"
NO_PULL=0
DOCKER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE=$2
      shift 2
      ;;
    --out-dir)
      OUT_DIR=$2
      shift 2
      ;;
    --step-limit)
      STEP_LIMIT=$2
      shift 2
      ;;
    --seats)
      SEATS=$2
      shift 2
      ;;
    --docker-arg)
      DOCKER_ARGS+=("$2")
      shift 2
      ;;
    --no-pull)
      NO_PULL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUNDLE_ABS=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$BUNDLE")
RECORD_ABS=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$RECORD")
RECORD_DIR=$(dirname "$RECORD_ABS")
RECORD_NAME=$(basename "$RECORD_ABS")

if [[ -z "$OUT_DIR" ]]; then
  base=$(basename "$RECORD_ABS")
  base=${base%.json}
  OUT_DIR="$ROOT_DIR/records/docker-replay-$base"
fi
OUT_DIR=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_DIR")
mkdir -p "$OUT_DIR"

TMP_ROOT=$(mktemp -d /tmp/orbit-wars-kaggle-docker.XXXXXX)
cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

BUNDLE_DIR="$TMP_ROOT/bundle"
mkdir -p "$BUNDLE_DIR"
if [[ -d "$BUNDLE_ABS" ]]; then
  cp -R "$BUNDLE_ABS"/. "$BUNDLE_DIR"/
else
  tar -xzf "$BUNDLE_ABS" -C "$BUNDLE_DIR"
fi

if [[ $NO_PULL -eq 0 ]]; then
  docker pull "$IMAGE"
fi

IFS=',' read -r -a SEAT_LIST <<< "$SEATS"
PIDS=()
declare -A LOG_PATHS
declare -A ERR_PATHS
for seat in "${SEAT_LIST[@]}"; do
  log_path="$OUT_DIR/seat-${seat}.jsonl"
  err_path="$OUT_DIR/seat-${seat}.err"
  : >"$log_path"
  : >"$err_path"
  LOG_PATHS["$seat"]="$log_path"
  ERR_PATHS["$seat"]="$err_path"
  cmd=(
    docker run --rm
    --network none
    -v "$BUNDLE_DIR:/bundle:ro"
    -v "$ROOT_DIR:/workspace:ro"
    -v "$RECORD_DIR:/record:ro"
    -w /workspace
  )
  if [[ ${#DOCKER_ARGS[@]} -gt 0 ]]; then
    cmd+=("${DOCKER_ARGS[@]}")
  fi
  cmd+=(
    "$IMAGE"
    /usr/bin/python3 /workspace/scripts/replay_submission_worker.py
    --bundle-dir /bundle
    --record "/record/$RECORD_NAME"
    --seat "$seat"
  )
  if [[ -n "$STEP_LIMIT" ]]; then
    cmd+=(--step-limit "$STEP_LIMIT")
  fi
  "${cmd[@]}" >"$log_path" 2>"$err_path" &
  PIDS+=($!)
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

for seat in "${SEAT_LIST[@]}"; do
  log_path=${LOG_PATHS["$seat"]}
  err_path=${ERR_PATHS["$seat"]}
  row_count=$(wc -l <"$log_path")
  if [[ "$row_count" -eq 0 ]]; then
    status=1
    echo "seat-$seat: 0 rows written" >&2
    if [[ -s "$err_path" ]]; then
      echo "seat-$seat stderr:" >&2
      sed -n '1,20p' "$err_path" >&2
    fi
  else
    echo "seat-$seat: $row_count rows"
  fi
done

echo "Wrote logs to $OUT_DIR"
exit "$status"
