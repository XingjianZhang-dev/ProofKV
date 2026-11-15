# Follow-up analyses

Code for the analyses that were run after the main experiments: the compressor comparison, link removal across
datasets, iterative retrieval, the decomposition of the gap to the oracle, the budget sweep, and related
distractors. Their records are in `results/followup/` and their statistics are recomputed by
`analysis/followup_statistics.py`.

All scripts import the entry points in `scripts/` unchanged, so prompts, budgets, selection, and decoding are the
ones used by the main experiments. Run them from the repository root. The commands below assume

```bash
export IRCOT=data/ircot/processed_data            # IRCoT processed data (docs/DATA.md)
export PROOFKV_IRCOT=$IRCOT
export DECODER=Qwen/Qwen2.5-7B-Instruct           # or a local copy
export REV=a09a35458c702b33eeacc393d103063234e8bc28
export FINAL=checkpoints/proofkv_scorer_final
export V6=checkpoints/proofkv_scorer_musique_replay_v6
export PROOFKV_OUT=runs
```

## Premise test: do token-level compressors keep the chain?

```bash
python experiments/followup/premise_test/compressors.py --data-file $IRCOT/musique/test_subsampled.jsonl \
    --model $DECODER --model-revision $REV --out-dir runs/premise_test/compressors
python experiments/followup/premise_test/chain_and_oracles.py --data-file $IRCOT/musique/test_subsampled.jsonl \
    --scorer-checkpoint $FINAL --model $DECODER --model-revision $REV --out-dir runs/premise_test/chain_and_oracles
```

## One link removed at equal bytes

```bash
COMMON="--scorer-checkpoint $FINAL --model $DECODER --model-revision $REV --seed 17 --budget-tokens 512 --max-new-tokens 24"
RUN=experiments/followup/link_interventions/run_link_interventions.py

# identifiers of the HotpotQA validation questions used elsewhere in the paper (excluded from the paragraph-level run)
python - <<'EOF'
import json, random
from datasets import load_dataset
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
order = list(range(len(ds))); random.Random(17).shuffle(order)
with open("runs/hotpot_validation_pos0_2048.jsonl", "w") as f:
    for i in order[:2048]:
        f.write(json.dumps({"id": ds[i]["id"]}) + "\n")
EOF

python $RUN --dataset musique --data-file $IRCOT/musique/test_subsampled.jsonl \
    --sample-offset 128 --max-samples 128 --exclude-positions 0:128 $COMMON --out-dir runs/link_interventions/musique_pos128_256
python $RUN --dataset musique --data-file $IRCOT/musique/test_subsampled.jsonl \
    --sample-offset 256 --max-samples 244 --exclude-positions 0:256 $COMMON --out-dir runs/link_interventions/musique_pos256_500
python $RUN --dataset hotpotqa_hf --sample-offset 2048 --max-samples 128 --exclude-positions 0:2048 \
    $COMMON --out-dir runs/link_interventions/hotpotqa_sentence
python $RUN --dataset hotpotqa_ircot --data-file $IRCOT/hotpotqa/test_subsampled.jsonl \
    --sample-offset 0 --max-samples 128 --skip-excluded --exclude-id-files runs/hotpot_validation_pos0_2048.jsonl \
    $COMMON --out-dir runs/link_interventions/hotpotqa_paragraph
python $RUN --dataset 2wiki --data-file $IRCOT/2wikimultihopqa/test_subsampled.jsonl \
    --sample-offset 0 --max-samples 128 --exclude-positions 128:432 \
    --exclude-id-files $IRCOT/2wikimultihopqa/dev_subsampled.jsonl $COMMON --out-dir runs/link_interventions/2wikimultihopqa

# the MuSiQue selections re-decoded under three attention implementations
python experiments/followup/link_interventions/attention_backends.py \
    --routes runs/link_interventions/musique_pos128_256/routes.jsonl --out-dir runs/link_interventions/attention_backends
```

## Held-out selector test: gap decomposition and iterative retrieval

Both scripts are selector-only; the decoder is loaded for its tokenizer.

```bash
python experiments/followup/heldout_selector/oracle_gap.py --split test \
    --data-file $IRCOT/musique/test_subsampled.jsonl --official-dev data/musique/musique_ans_v1.0_dev.jsonl \
    --scorer-checkpoint $V6 --model $DECODER --max-samples 1000 --budgets 256,512 --out-dir runs/heldout_selector/oracle_gap
python experiments/followup/heldout_selector/iterative_selector.py --split test \
    --data-file $IRCOT/musique/test_subsampled.jsonl --scorer-checkpoint $V6 --model $DECODER \
    --max-samples 1000 --budgets 256,512 --ks 1,2,4 --max-hops 4 --out-dir runs/heldout_selector/iterative_retrieval
```

`iterative_retrieval.py` is the retriever itself: at each hop it appends the retrieved spans to the query and
re-scores all spans with ProofKV's start head, so the encoder is the same as ProofKV's.

## Budget sweep

`run_iterative_matrix.py` adds the iterative-retrieval methods to `scripts/run_hotpot_compact_qa.py` and runs it
unchanged.

```bash
M=learned_topk_paths,learned_iterative_retrieval_k1_h4_hoporder_tokenfill,learned_iterative_retrieval_k1_h4_tokenfill,learned_root_topk_tokenfill
for B in 64 96 128 192 256 384 512; do
  python experiments/followup/heldout_selector/run_iterative_matrix.py --dataset musique \
      --data-file $IRCOT/musique/test_subsampled.jsonl --split validation --unlock-locked-test \
      --model $DECODER --scorer-checkpoint $V6 --max-samples 128 --sample-offset 0 --seed 17 \
      --budget-tokens $B --max-new-tokens 24 --attn-implementation sdpa \
      --learned-dense-core-count 4 --learned-edge-batch-size 256 --methods $M \
      --out-dir runs/budget_regime/matrix_musique_v6_n128/fixed$B
done
python experiments/followup/budget_regime/chain_features.py     # gold-chain size and fit per question and budget
```

## Related distractors

```bash
for POOL in HARD_TFIDF SAME_ENTITY; do for T in 8000 16000 32000; do
  d=runs/related_distractors/$POOL/hotpotqa_target$T; mkdir -p $d
  python experiments/followup/related_distractors/run_related_matrix.py $POOL $d/packed_contexts.jsonl \
      --dataset hotpotqa --split validation --model $DECODER --scorer-checkpoint $FINAL \
      --max-samples 128 --sample-offset 0 --seed 17 --budget 0.08 --target-prompt-tokens $T --packing-seed 911 \
      --max-new-tokens 24 --attn-implementation sdpa --full-prefill-chunk-size 2048 --compact-prefill-chunk-size 512 \
      --learned-dense-core-count 4 --learned-edge-batch-size 256 \
      --methods full,entity_graph_adaptive,learned_topk_paths,oracle_adaptive --out-dir $d
done; done
python experiments/followup/related_distractors/audit_packing.py   # utilization, answer leakage, duplicates
```

## Statistics

```bash
python analysis/followup_statistics.py results/followup results/followup/summary.json
```

Paired differences use a 10,000-sample bootstrap for the 95% interval and a 100,000-replicate sign-flip test, with
the seeds fixed in the script.
