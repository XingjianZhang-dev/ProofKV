# Frozen fair-contract and parser-free external protocol

Frozen before the formal runs on 2026-07-26.

## Question

Does typed complete-proof selection outperform independent-span ranking when
both methods use the same trained scorer, training data, decoder, selected-token
budget, prompts, and frozen test examples? Does the result survive on datasets
that were not used to train the scorer and without a task-specific parser?

## Immutable inputs

- Selection:
  `selection_manifest_v10_complement_n47.json`
  (`c7cda95a9b7a42c70428aba9b973e732de56242c1f7b04e6322ff60546fc6b76`).
- Formal subset: indices 1--46 of every selected task. Index 0 is excluded
  because it was opened for the technical model/checkpoint smoke test.
- Models:
  Qwen2.5-7B-Instruct revision
  `a09a35458c702b33eeacc393d103063234e8bc28`; Mistral-7B-Instruct-v0.2
  revision `63a8b081895390a26e140280378bc85ec8bce07a`.
- Scorer: the previously frozen
  `relation_scorer_soup_v3b_v8_a090/checkpoint`.
- Capacities: 64, 128, 256, and 512 total KV tokens.
- Tasks: Qasper, MultiFieldQA-en, HotpotQA, 2WikiMultiHopQA, and MuSiQue.
- Methods:
  `learned_root_topk` (independent spans ordered only by the shared scorer's
  root probabilities) and `learned_and_or_control_anchor75_aux` (the
  previously development-selected ProofKV route).

The independent-span control deliberately computes the complete shared scorer
output and then discards typed edges. Thus model capacity, learned
representations, candidate spans, training supervision, and semantic front-end
cost are matched; only set construction differs.

## Precommitted endpoints

1. Primary contract endpoint: equal-weight mean paired QA-F1 difference,
   ProofKV minus independent spans, over both models, all five tasks, and
   capacities 128/256/512.
2. Parser-free external endpoint: the same statistic restricted to Qasper and
   MultiFieldQA-en. Neither dataset was used to train the scorer, whose natural
   training families are HotpotQA, 2WikiMultiHopQA, and MuSiQue.
3. Resource endpoint: retained-token equality and selector-inclusive time,
   reported separately from generation time.

Inference uses paired example bootstrap and paired sign randomization with
50,000 replicates, clustered by model, task, and capacity. Holm correction is
applied to the two confirmatory endpoints. A positive confirmatory result
requires a positive point estimate, a 95% interval strictly above zero, and
Holm-adjusted two-sided p below 0.05.

## No-tuning rule

Formal predictions are append-only and resumable. No threshold, scorer,
candidate count, anchor fraction, task parser, prompt, capacity, or example may
be changed after formal output begins. Negative cells remain in the audit and
paper-facing aggregate.

## Training-boundary audit amendment

Before confirmatory analysis and before the run reached any 2Wiki formal row,
the checkpoint manifest was re-audited. It records 8,000 HotpotQA, 4,000
2WikiMultiHopQA, 4,000 MuSiQue, and 4,000 controlled-topology training
examples. The original text incorrectly described 2WikiMultiHopQA as external.
The parser-free external endpoint was therefore narrowed, outcome-blind with
respect to 2Wiki, to Qasper and MultiFieldQA-en. The all-task same-scorer
endpoint and every executed row remain unchanged.
