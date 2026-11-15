#!/bin/bash
set -euo pipefail

model_label=${1:?model label required: qwen25 or mistral02}
repo=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
artifact_parent=$(cd "$repo/../.." && pwd)
data_root=${PROOFKV_DATA_ROOT:-$artifact_parent/proofkv_data/frozen_workspace}
runs_root=${OUTPUT_ROOT:-$artifact_parent/proofkv_runs}
root=$runs_root/fair_contract_external_20260726
selection=${SELECTION_MANIFEST:-$data_root/gpu_results_20260724/acl_main_longbench_structured_20260724/selection_manifest_v10_complement_n47.json}
scorer=${SCORER_CHECKPOINT:-$repo/checkpoint/proofkv_scorer_a090}

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
    echo "unsupported model label: $model_label" >&2
    exit 2
    ;;
esac

mkdir -p "$root"
cd "$repo"
export PYTHONPATH=.:scripts
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

exec /venv/main/bin/python scripts/run_longbench_structured_upgraded_v10.py \
  --selection-manifest "$selection" \
  --model "$model" \
  --model-label "$model_label" \
  --model-revision "$revision" \
  --scorer-checkpoint "$scorer" \
  --methods learned_root_topk,learned_and_or_control_anchor75_aux \
  --capacities 64,128,256,512 \
  --sample-start-per-task 1 \
  --max-samples-per-task 46 \
  --attn-implementation sdpa \
  --learned-dense-core-count 4 \
  --learned-edge-batch-size 256 \
  --out-dir "$root/formal_${model_label}_n47"
