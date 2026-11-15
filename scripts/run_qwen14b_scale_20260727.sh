#!/bin/bash
set -euo pipefail

mode=${1:?mode required: smoke, proof_pilot, proof_formal, or ll2_formal}
repo=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
artifact_parent=$(cd "$repo/../.." && pwd)
data_root=${PROOFKV_DATA_ROOT:-$artifact_parent/proofkv_data/frozen_workspace}
runs_root=${OUTPUT_ROOT:-$artifact_parent/proofkv_runs}
root=$runs_root/qwen14b_scale_20260727
source_selection=${SELECTION_MANIFEST:-$data_root/gpu_results_20260724/acl_main_longbench_structured_20260724/selection_manifest_v10_complement_n47.json}
selection=$root/selection_manifest_qwen14b_n47.json
model=/workspace/models/Qwen2.5-14B-Instruct-cf98f3b
revision=cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
scorer=${SCORER_CHECKPOINT:-$repo/checkpoint/proofkv_scorer_a090}
compressor=/workspace/models/llmlingua-2-xlm-roberta-large-meetingbank-ebaba9b

mkdir -p "$root"
cd "$repo"
export PYTHONPATH=.:scripts
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

if [[ ! -f "$selection" ]]; then
  /venv/main/bin/python scripts/prepare_qwen14b_scale_manifest_20260727.py \
    --source-manifest "$source_selection" \
    --model "$model" \
    --out "$selection"
fi

common=(
  /venv/main/bin/python scripts/run_longbench_structured_upgraded_v10.py
  --selection-manifest "$selection"
  --model "$model"
  --model-label qwen25_14b
  --model-revision "$revision"
  --scorer-checkpoint "$scorer"
  --methods full,learned_root_topk,learned_and_or_control_anchor75_aux
  --capacities 128,256,512
  --attn-implementation sdpa
  --learned-dense-core-count 4
  --learned-edge-batch-size 256
)

case "$mode" in
  smoke)
    exec "${common[@]}" \
      --tasks hotpotqa \
      --max-samples-per-task 1 \
      --out-dir "$root/smoke_qwen25_14b"
    ;;
  proof_pilot)
    exec "${common[@]}" \
      --sample-start-per-task 1 \
      --max-samples-per-task 5 \
      --out-dir "$root/formal_proof_qwen25_14b_n47"
    ;;
  proof_formal)
    exec "${common[@]}" \
      --sample-start-per-task 1 \
      --max-samples-per-task 46 \
      --out-dir "$root/formal_proof_qwen25_14b_n47"
    ;;
  ll2_formal)
    exec /venv/main/bin/python scripts/run_llmlingua2_longbench_20260726.py \
      --selection-manifest "$selection" \
      --model "$model" \
      --model-label qwen25_14b \
      --model-revision "$revision" \
      --compressor-model "$compressor" \
      --compressor-revision ebaba9b0e874dadd3003ffcff828e4397e568089 \
      --capacities 128,256,512 \
      --sample-start-per-task 1 \
      --max-samples-per-task 46 \
      --attn-implementation sdpa \
      --out-dir "$root/formal_ll2_qwen25_14b_n47"
    ;;
  *)
    echo "unsupported mode: $mode" >&2
    exit 2
    ;;
esac
