#!/usr/bin/env bash
set -uo pipefail
root="$PWD"
python="$root/.venv/bin/python"
status=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
for pr in 8883 8884; do
  cd "$root/source-$pr"
  export PYTHONPATH="$PWD"
  git rev-parse HEAD > "$root/results/head-$pr.txt"
  "$python" -m pytest tests/entrypoints/openai_api/test_serving_speech.py \
    -m 'core_model and cpu' --run-level core_model -ra \
    --junitxml="$root/results/speech-$pr.xml" \
    2>&1 | tee "$root/results/speech-$pr.log"
  rc=${PIPESTATUS[0]}
  printf '%s\n' "$rc" > "$root/results/speech-$pr.exit"
  if [ "$rc" -ne 0 ]; then status=1; fi
  "$python" -m pytest \
    tests/e2e/features/comfyui/test_h3_all_in_one_workflow.py \
    tests/e2e/features/comfyui/test_h3_music_video_workflow.py \
    tests/e2e/features/comfyui/test_h3_timeline_guides_workflow.py \
    -m 'core_model and cpu' --run-level core_model -ra \
    --junitxml="$root/results/comfy-$pr.xml" \
    2>&1 | tee "$root/results/comfy-$pr.log"
  rc=${PIPESTATUS[0]}
  printf '%s\n' "$rc" > "$root/results/comfy-$pr.exit"
  if [ "$rc" -ne 0 ]; then status=1; fi
  git status --porcelain > "$root/results/worktree-$pr.txt"
done
exit "$status"
