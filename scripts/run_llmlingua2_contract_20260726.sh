#!/usr/bin/env bash
set -euo pipefail

model_label="${1:?model label: qwen25 or mistral02}"
repo=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
artifact_parent=$(cd "$repo/../.." && pwd)
data_root=${PROOFKV_DATA_ROOT:-$artifact_parent/proofkv_data/frozen_workspace}
base=${OUTPUT_ROOT:-$artifact_parent/proofkv_runs}
selection=${SELECTION_MANIFEST:-$data_root/gpu_results_20260724/acl_main_longbench_structured_20260724/selection_manifest_v10_complement_n47.json}
compressor=/workspace/models/llmlingua-2-xlm-roberta-large-meetingbank-ebaba9b
out="$base/llmlingua2_contract_20260726/formal_${model_label}_n46"

case "$model_label" in
  qwen25)
    model=/workspace/models/Qwen2.5-7B-Instruct-a09a354
    revision=a09a35458c702b33eeacc393d103063234e8bc28
    ;;
  mistral02)
    model=/workspace/models/Mistral-7B-Instruct-v0.2-63a8b08
    revision=63a8b081895390a26e140280378bc85ec8bce07a
    ;;
  *)
    echo "unknown model label: $model_label" >&2
    exit 2
    ;;
esac

exec /venv/main/bin/python scripts/run_llmlingua2_longbench_20260726.py \
  --selection-manifest "$selection" \
  --model "$model" \
  --model-label "$model_label" \
  --model-revision "$revision" \
  --compressor-model "$compressor" \
  --compressor-revision ebaba9b0e874dadd3003ffcff828e4397e568089 \
  --capacities 128,256,512 \
  --sample-start-per-task 1 \
  --max-samples-per-task 46 \
  --out-dir "$out"
