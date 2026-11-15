# Frozen Qwen2.5-14B scale-transfer protocol

Frozen on 2026-07-27 before any Qwen2.5-14B QA prediction was generated.

## Question

Does ProofKV's typed-composition advantage survive when the downstream decoder
is scaled from 7B to 14B, without retraining or adapting the frozen evidence
scorer? This experiment is a decoder-scale transfer test, not a new selector
development stage.

## Immutable inputs

- Decoder: `Qwen/Qwen2.5-14B-Instruct`, revision
  `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
- Frozen LongBench selection:
  `selection_manifest_v10_complement_n47.json`, with formal indices 1--46.
  Index 0 is reserved for technical smoke testing.
- Tasks: Qasper, MultiFieldQA-en, HotpotQA, 2WikiMultiHopQA, and MuSiQue.
- Frozen scorer:
  `relation_scorer_soup_v3b_v8_a090/checkpoint`. It is not retrained,
  calibrated, or thresholded for the 14B decoder.
- Total compact-prefill capacities: 128, 256, and 512 decoder tokens.
- Methods:
  - Full context;
  - `learned_root_topk`, which uses the same frozen scorer but discards typed
    edges;
  - `learned_and_or_control_anchor75_aux`, the frozen ProofKV compiler;
  - official LLMLingua-2 under exactly matched retained decoder tokens.
- Generation prompts, answer normalization, maximum output lengths, physical
  compiler, and scorer settings are identical to the locked 7B contract.

The Qwen2.5-14B tokenizer is recomputed over every frozen example before the
run. The derived manifest must preserve every example ID and source hash. It
records whether its token counts are identical to the locked Qwen2.5-7B
tokenizer; no count is copied without this verification.

## Precommitted endpoints

1. Primary scale-transfer endpoint: equal-weight mean paired QA-F1 difference
   between ProofKV and same-scorer independent root ranking over all five tasks
   and all three capacities.
2. Matched visible-text endpoint: the corresponding paired difference between
   ProofKV and official LLMLingua-2, with zero per-example retained-token
   difference.
3. Full-context retention at 512 tokens, reported as ProofKV F1 divided by Full
   F1 and as an absolute difference. This is descriptive and is not required to
   exceed Full.
4. Selector/compressor time, generation time, retained KV bytes, and peak CUDA
   memory are retained in the raw rows.

Inference clusters paired examples by task and capacity, uses 50,000 bootstrap
replicates, and uses exact cluster sign randomization when feasible. The two
primary compact-method endpoints receive Holm correction.

## Staged execution and stopping rule

The common implementation tests must pass before GPU execution. Index 0 may be
used only for technical loading and tokenization checks. Formal output is
append-only and resumable. After the first five formal examples per task, an
interim audit may stop the remaining run only for implementation failure,
capacity violation, non-finite output, or if both precommitted compact-method
differences are non-positive. No method, prompt, scorer, capacity, task, or
example may be changed after formal output begins.

Negative task cells remain in the aggregate. The result enters the paper only
under the scale-transfer contract stated here; it is never used to claim
universal task or model scaling.
