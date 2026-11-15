# Frozen LLMLingua-2 contract protocol

Frozen before any LLMLingua-2 QA output was generated on 2026-07-26.

## Purpose

This experiment compares ProofKV with a prompt compressor under the contract
they genuinely share: both may inspect the visible input text before decoding,
both execute prefill only over the resulting compact prompt, and both are
charged for selector/compressor time, generation time, retained KV bytes, and
peak CUDA memory.

## Immutable inputs

- Frozen LongBench selection:
  `selection_manifest_v10_complement_n47.json`
  (`c7cda95a9b7a42c70428aba9b973e732de56242c1f7b04e6322ff60546fc6b76`).
- Formal examples: indices 1--46 of Qasper, MultiFieldQA-en, HotpotQA,
  2WikiMultiHopQA, and MuSiQue. Index 0 remains excluded because it was opened
  for technical smoke testing of the other formal comparison.
- Decoders:
  Qwen2.5-7B-Instruct revision
  `a09a35458c702b33eeacc393d103063234e8bc28` and
  Mistral-7B-Instruct-v0.2 revision
  `63a8b081895390a26e140280378bc85ec8bce07a`.
- Compressor: official `llmlingua==0.2.2` with
  `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` revision
  `ebaba9b0e874dadd3003ffcff828e4397e568089`.
- Total prefill-KV capacities: 128, 256, and 512 tokens, including the
  uncompressed question and answer prefix.

LLMLingua-2 receives only the context. The official question and answer prefix
are appended unchanged. Its compressed text is retokenized by the evaluated
decoder. If the official target produces fewer decoder tokens than the budget,
the target is doubled, using only token counts, until the output reaches the
budget; the resulting text is then deterministically trimmed from the end to
the exact total-KV cap. No task-specific parser, gold support, answer, decoder
attention, hidden state, or logit is available to the compressor.

## Endpoints

1. Equal-weight task--model--capacity QA-F1 for LLMLingua-2 and ProofKV.
2. Paired ProofKV-minus-LLMLingua-2 QA-F1 over both models, all five tasks,
   and all three capacities, with 50,000 paired clustered bootstrap and sign
   randomization replicates.
3. Selector-inclusive latency, generation latency, retained KV bytes, and peak
   CUDA memory. The paper reports these measurements without claiming that
   different front-end model sizes are identical.

Formal rows are append-only and resumable. No compressor option, task,
capacity, prompt, example, or postprocessing rule may be changed after output
begins.

## Budget-normalization amendment before formal V2

The first technical matrix exposed an official-API boundary on very long
Qasper inputs: at sub-1% target rates, LLMLingua-2 sometimes emitted zero
context tokens, leaving a nominal 128-token cell with only 21--39 physical
tokens. Those rows are retained under `failed_budget_underfill_v1_*` and are
excluded from inference. Before formal V2 began, a deterministic length-only
normalizer was frozen: double the compressor target until the decoder-tokenized
prompt reaches the requested cap, then trim to that cap. This repairs the
resource contract without consulting answers or QA scores.

## Wall-time sharding amendment before tail-task output

Before any formal 2WikiMultiHopQA or MuSiQue LLMLingua-2 row was generated,
those two tail tasks were assigned to one additional process per decoder. The
shards use the unchanged runner, model revisions, examples, capacities,
compressor, prompts, budget normalizer, and decoding configuration. Their
selection manifests are literal task-subsets of the locked manifest; no row is
reselected.

The canonical result is formed by a deterministic, score-blind merge. Run
directories have a declared priority, the first occurrence of a duplicate
model--task--ID--capacity key is retained without inspecting its prediction or
score, and the merge fails unless all 690 rows per decoder are present. This
amendment changes only wall time and is frozen before starting either tail
shard.

Before the Mistral tail shard started, the task allocation was narrowed to
MuSiQue only because the Mistral primary process was progressing faster than
the Qwen primary process. The primary Mistral process remains responsible for
2WikiMultiHopQA. This scheduling choice used only completed row counts, was
made before the primary process entered 2WikiMultiHopQA, and did not inspect
predictions or scores. The Qwen tail shard remains 2WikiMultiHopQA+MuSiQue.

After the Qwen tail shard completed, the remaining Mistral MuSiQue process was
the only GPU job. It was stopped after 11 complete examples plus the 128-token
row of the twelfth example. Before inspecting any MuSiQue score, the remaining
locked positions 12--29 and 30--46 were assigned to two disjoint sample shards
(18 and 17 examples). The already written twelfth-example 128-token row is a
score-blind duplicate resolved by the same declared merge priority. This final
sharding amendment changes wall time only.
