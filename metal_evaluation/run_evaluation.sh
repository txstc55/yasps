#!/usr/bin/env bash

set -euo pipefail

REPOSITORY_ROOT="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd
)"
ARTIFACT_ROOT="$REPOSITORY_ROOT/metal_evaluation"
YASPS_PATH="$REPOSITORY_ROOT/yasps"

run_rendered() {
  local variant="$1"
  local example_directory="$2"
  local script="$3"
  shift 3
  local output="$ARTIFACT_ROOT/$variant/evaluation"
  mkdir -p "$output/frames"
  (
    cd "$example_directory"
    /usr/bin/time -p env \
      YASPS_BACKEND=metal \
      YASPS_NUM_FRAMES=24 \
      YASPS_OFF_SCREEN=1 \
      PYVISTA_OFF_SCREEN=true \
      YASPS_RENDER_WIDTH=960 \
      YASPS_RENDER_HEIGHT=540 \
      YASPS_FRAME_DIRECTORY="$output/frames" \
      YASPS_METAL_TIMING_JSON="$output/kernel_timings.json" \
      PYTHONPATH="$YASPS_PATH" \
      python "$script" "$@" >"$output/run.log" 2>&1
  )
  (
    cd "$output/frames"
    ffmpeg -y -loglevel error -framerate 12 \
      -i frame_%04d.png -c:v libx264 -preset slow -crf 18 \
      -pix_fmt yuv420p "../$variant.mp4"
  )
}

run_rendered \
  one_bunny_partial_abd \
  "$REPOSITORY_ROOT/examples/one_bunny_partial_abd" \
  one_bunny_partial_abd.py

run_rendered \
  one_bunny_partial_abd_separate_jacobian \
  "$REPOSITORY_ROOT/examples/one_bunny_partial_abd_separate_jacobian" \
  one_bunny_partial_abd.py

run_rendered \
  dropping_in_container \
  "$REPOSITORY_ROOT/examples/dropping_in_container" \
  dropping_in_container.py \
  --num-bunnies 1

run_rendered \
  dropping_in_container_mixed \
  "$REPOSITORY_ROOT/examples/dropping_in_container_mixed" \
  dropping_in_container.py \
  --num-bunnies 2

run_rendered \
  dropping_in_container_mixed_separation \
  "$REPOSITORY_ROOT/examples/dropping_in_container_mixed_separation" \
  dropping_in_container.py \
  --num-bunnies 2

NO_SAVE_OUTPUT="$ARTIFACT_ROOT/dropping_in_container_no_save/evaluation"
mkdir -p "$NO_SAVE_OUTPUT"
(
  cd "$REPOSITORY_ROOT/examples/dropping_in_container"
  /usr/bin/time -p env \
    YASPS_BACKEND=metal \
    YASPS_NUM_FRAMES=24 \
    YASPS_METAL_TIMING_JSON="$NO_SAVE_OUTPUT/kernel_timings.json" \
    PYTHONPATH="$YASPS_PATH" \
    python dropping_in_container_no_save.py \
      --num-bunnies 1 >"$NO_SAVE_OUTPUT/run.log" 2>&1
)

python "$ARTIFACT_ROOT/summarize.py"
