#!/bin/bash
set -euo pipefail

task=${1:?task required}
condition=${2:?condition required}
repo=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
artifact_parent=$(cd "$repo/../.." && pwd)
runs_root=${OUTPUT_ROOT:-$artifact_parent/proofkv_runs}
root=$runs_root/acl_main_longbench_structured_20260724/evidencepack_budget_pareto_V25
scorer=${SCORER_CHECKPOINT:-$repo/checkpoint/proofkv_scorer_a090}

case "$task" in
  musique)
    data=/workspace/ircot/processed_data/musique/test_subsampled.jsonl
    ;;
  2wiki)
    data=/workspace/ircot/processed_data/2wikimultihopqa/test_subsampled.jsonl
    ;;
  *)
    echo "unsupported task: $task" >&2
    exit 2
    ;;
esac

case "$condition" in
  full)
    budget=0.08
    methods=full
    out="$root/formal_${task}_qwen25_n64_offset368_full"
    ;;
  b04|b08|b12)
    code=${condition#b}
    budget=$(awk -v code="$code" 'BEGIN { printf "%.2f", code / 100 }')
    methods=entity_graph_adaptive,learned_root_topk,learned_and_or
    out="$root/formal_${task}_qwen25_n64_offset368_target8000_budget${code}"
    ;;
  *)
    echo "unsupported condition: $condition" >&2
    exit 2
    ;;
esac

cd "$repo"
PYTHONPATH=.:scripts HF_HOME=/workspace/.cache/huggingface /venv/main/bin/python \
  scripts/run_hotpot_compact_qa.py \
  --dataset "$task" \
  --data-file "$data" \
  --split validation \
  --unlock-locked-test \
  --model /workspace/models/Qwen2.5-7B-Instruct-a09a354 \
  --sample-offset 368 \
  --max-samples 64 \
  --budget "$budget" \
  --methods "$methods" \
  --max-new-tokens 24 \
  --seed 17 \
  --packing-seed 1907 \
  --target-prompt-tokens 8000 \
  --attn-implementation sdpa \
  --full-prefill-chunk-size 2048 \
  --compact-prefill-chunk-size 512 \
  --learned-dense-core-count 4 \
  --learned-edge-batch-size 256 \
  --scorer-checkpoint "$scorer" \
  --out-dir "$out"
