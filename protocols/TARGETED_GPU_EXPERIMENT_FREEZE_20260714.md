# Targeted GPU Experiment Freeze — 2026-07-14

This file was frozen before any formal result from the two targeted groups was
opened. The accepted selector is the alpha-0.90 typed cross-encoder checkpoint.

## Group 1 — selector-inclusive serving

- Dataset: deterministically packed HotpotQA validation, seed 17, packing seed
  911, budget 0.08.
- Scales: 8k with 16 requests per batch; 32k with 4 requests per batch.
- Output lengths: 256 and 1024 fixed decode tokens.
- Timing: two warmups and five measured repeats, rotating/reversing method order.
- Modes:
  - `full`: full prompt vLLM prefill and decode.
  - `precompiled`: route already compiled; vLLM prefill and decode only.
  - `shared_index`: re-run the query-specific typed scorer and proof selector,
    reusing only the query-agnostic Evidence Index, then vLLM prefill and decode.
  - `cold`: rebuild the query-agnostic index, re-run the query-specific typed
    scorer and proof selector, then vLLM prefill and decode.
- Forbidden reuse: a query-specific overlay, root/terminal scores, learned typed
  edges, proof paths, or retained-token route may not be reused in `cold` or
  `shared_index` measured timing.
- Integrity gates: cold/shared/precompiled token routes must be bit-identical for
  every request; no vLLM request may be marked corrupted; final prompt token is
  always retained.
- The precompiled result is not called end-to-end. Cold and shared-index results
  are reported even if their speedup is negative.

## Group 2 — natural topology-specific downstream causality

- Dataset: official IRCoT processed MuSiQue `test_subsampled.jsonl` (500 rows).
- Prior downstream exclusion: all 128 IDs used by
  `locked_v6_qwen25_musique_test_matrix_n128/fixed512/results.jsonl`.
- Formal slice: deterministic seed-17 shuffled positions 128:256, exactly 128
  IDs. The overlap with prior downstream IDs must be zero.
- Claim boundary: this is a disjoint post-hoc downstream causal slice, not an
  untouched locked test, because aggregate selector metrics over all 500 rows
  were observed previously.
- Model: Qwen2.5-7B-Instruct, greedy short-answer decoding, 24 new-token cap.
- Every intervention retains exactly 512 prompt tokens per example (or the full
  prompt if shorter), so physical BF16 KV bytes are matched within example.
- Methods:
  - `proof_core`: accepted typed AND/OR proof core.
  - `relevance_same_bytes`: learned root relevance at identical retained bytes.
  - `edge_shuffled`: preserve learned edge types/confidences but shuffle targets.
  - `random_edges`: randomize both edge targets and types.
  - `broken_middle`: remove the middle node of each selected proof (the terminal
    bridge target for a two-span proof) and forbid it from neutral padding.
  - `oracle_proof`: gold supporting paragraphs under the identical byte budget.
- Interventions preserve learned root and terminal scores. Padding needed for
  exact byte equality is query-agnostic and does not use gold labels.
- Frozen primary gate: `proof_core - edge_shuffled >= +0.03` answer F1 and the
  paired bootstrap two-sided 95% confidence interval lower bound is above zero.
- Secondary comparisons (`relevance_same_bytes`, `random_edges`,
  `broken_middle`) and the oracle sanity ceiling are reported regardless of sign.
- Formal IDs are not used for architecture repair or threshold selection after
  results are opened.

## Audit artifacts

Every group writes an immutable run manifest containing selected IDs, hashes,
arguments, timing boundaries, row counts, corruption checks, and paired results.
The experiment queue continues after one cell fails so a local software failure
cannot silently discard the other targeted group.
