#!/bin/bash
set -euo pipefail

task=${1:?task required}
target=${2:?target prompt tokens required}
repo=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
artifact_parent=$(cd "$repo/../.." && pwd)
runs_root=${OUTPUT_ROOT:-$artifact_parent/proofkv_runs}
root=$runs_root/acl_main_longbench_structured_20260724/evidencepack_risk_controller_V22_formal
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

case "$target" in
  8000)
    methods=full,bm25_topk,reranker_topk,entity_graph_adaptive,learned_root_topk,learned_and_or
    ;;
  16000|32000)
    methods=full,entity_graph_adaptive,learned_root_topk,learned_and_or
    ;;
  *)
    echo "unsupported target: $target" >&2
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
  --sample-offset 240 \
  --max-samples 128 \
  --budget 0.08 \
  --methods "$methods" \
  --max-new-tokens 24 \
  --seed 17 \
  --packing-seed 1907 \
  --target-prompt-tokens "$target" \
  --attn-implementation sdpa \
  --full-prefill-chunk-size 2048 \
  --compact-prefill-chunk-size 512 \
  --learned-dense-core-count 4 \
  --learned-edge-batch-size 256 \
  --scorer-checkpoint "$scorer" \
  --out-dir "$root/formal_${task}_qwen25_n128_offset240_target${target}"
