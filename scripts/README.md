# Scripts

The entry points that produced the results of the paper, grouped by experiment. A filled marker (●) is a script you
run; the others are imported by it or audit and analyze its output. Every Python script takes `--help`
(`run_vllm_paged_suite.py` once `vllm` and `pynvml` are installed). The exact arguments
of each reported run are in the `run_manifest.json` beside its records under `results/paper/records/`.

File names are the ones recorded in those manifests, including the dates and version tags of the runs, so a record
can be traced to the script that wrote it. [`docs/REPRODUCE.md`](../docs/REPRODUCE.md) maps every table and figure to
its script and translates the method names used in code into the names used in the paper.

## Start here

Five scripts carry the method and the main results. Everything else in this directory supports one specific table
or figure of the paper.

| Script | Use it to |
|---|---|
| `train_typed_relation_scorer.py` | train the scorer |
| `eval_proofcache_selector.py` | measure complete-chain recall of a selector (the held-out MuSiQue test) |
| `run_hotpot_compact_qa.py` | answer questions end to end with compact prefill at a fixed budget |
| `run_longbench_structured_upgraded_v10.py` | run the five-task suite |
| `run_natural_topology_causality.py` | remove one link at equal bytes |

## Scorer training and checkpoints

| | Script | What it does |
|---|---|---|
| ● | `train_typed_relation_scorer.py` | Train the scorer (start, end, and link heads) for one replay stage. |
| ● | `interpolate_relation_checkpoints.py` | Interpolate two checkpoints; the final checkpoint uses weight 0.90. |
|  | `eval_relation_scorer_components.py` | Per-head accuracy of a checkpoint. |
|  | `train_cached_relation_scorer.py` | Optional bi-encoder scorer with cacheable span embeddings (the `cached_*` methods). Not used for any result in the paper. |
|  | `distill_cached_relation_scorer.py` | Distillation of that optional scorer from the cross-encoder. Not used for any result in the paper. |

## Selector evaluation without decoding

| | Script | What it does |
|---|---|---|
| ● | `eval_proofcache_selector.py` | Complete-chain recall of every selector at exact budgets; runs the held-out MuSiQue selector test. |
| ● | `eval_topology_gate.py` | Generated chains of controlled depth, with intact, shuffled, and replaced links. |
|  | `eval_natural_route_coverage.py` | Prompt construction, budget accounting, and gold-chain coverage; imported by the generation scripts. |
|  | `audit_hotpot_spans.py` | Audit of HotpotQA span boundaries and budgets under a decoder's tokenizer. |

## Fixed-budget generation on HotpotQA, 2WikiMultiHopQA, and MuSiQue

| | Script | What it does |
|---|---|---|
| ● | `run_hotpot_compact_qa.py` | End-to-end QA with compact prefill for every selector, at exact token or ratio budgets; also builds the packed contexts (`--target-prompt-tokens`). |
| ● | `run_natural_topology_causality.py` | Equal-byte link interventions: intact chains, relevance ranking, shuffled link targets, random links, middle span removed, oracle. |
| ● | `run_kvpress_natural_qa.py` | KVPress eviction baselines on the same questions and budgets. |
|  | `audit_targeted_gpu_experiments.py` | Paired bootstrap intervals and sign-flip tests for the intervention and serving groups. |

## Five-task suite (LongBench structured QA)

| | Script | What it does |
|---|---|---|
| ● | `freeze_longbench_structured_selection.py` | Fix the question selection for all tasks and decoders before any run. |
| ● | `run_longbench_structured_upgraded_v10.py` | ProofKV and the same-scorer selectors on the suite at exact KV caps. |
|  | `run_longbench_structured_common.py` | Shared runner of the suite. |
|  | `run_hotpot_compact_qa_upgraded_v10.py` | Compact-prefill QA driver used by the suite and the RULER runs. |
|  | `audit_longbench_structured_common.py` | Budget and score audit of a suite run. |
|  | `analyze_longbench_upgraded_v10_statistics.py` | Paired inference for the suite. |
|  | `analyze_longbench_v10_efficiency.py` | Physical accounting (tokens, KV bytes) from suite rows. |
|  | `prepare_longbench_v10_tail_shards.py` | Split a run into per-task shards for scheduling. |
|  | `merge_longbench_v10_tail_shards.py` | Merge the shards with exact duplicate checks. |

## Baselines on the five-task suite

| | Script | What it does |
|---|---|---|
| ● | `run_official_longbench_v10_baseline.py` | KV-eviction baselines in their own setting on the suite. |
|  | `freeze_official_longbench_v10_protocol.py` | Fix the baseline comparison before any output exists. |
|  | `audit_official_longbench_v10_smoke_preflight.py` | Preflight check of the baseline runs. |
|  | `audit_official_longbench_v10_baselines.py` | Integrity audit and paired inference for the baselines. |
| ● | `chain_official_longbench_v10_baselines.sh` | Runs the four steps above in order. |
|  | `chain_official_longbench_v10_baselines_state_reset_r1.sh` | The same chain with decoder state reset between methods. |
|  | `analyze_transparent_baselines_20260726.py` | Builds the baseline comparison of the main five-task table. |
| ● | `run_fair_contract_external_20260726.sh` | Same-scorer comparison and the two tasks unseen in training. |
|  | `analyze_fair_contract_external_20260726.py` | Paired endpoints of that comparison. |
|  | `prepare_llmlingua2_shards_20260726.py` | Task-subset manifests for the LLMLingua-2 runs. |
|  | `run_llmlingua2_longbench_20260726.py` | LLMLingua-2 at the token count ProofKV retains on each example. |
| ● | `run_llmlingua2_contract_20260726.sh` | Launches the LLMLingua-2 comparison. |
|  | `analyze_llmlingua2_contract_20260726.py` | Paired endpoints of the LLMLingua-2 comparison. |
|  | `prepare_qwen14b_scale_manifest_20260727.py` | Bind Qwen2.5-14B token counts to the fixed selection. |
| ● | `run_qwen14b_scale_20260727.sh` | Qwen2.5-14B transfer with the scorer and thresholds unchanged. |
|  | `analyze_qwen14b_scale_20260727.py` | Paired endpoints of the 14B transfer. |

## Routing

| | Script | What it does |
|---|---|---|
| ● | `run_evidencepack_risk_controller_V22_formal.sh` | Margin router across context lengths on the first held-out interval. |
|  | `audit_evidencepack_risk_controller_V22_formal.py` | Audit and statistics of that run. |
| ● | `run_evidencepack_budget_pareto_V25.sh` | Budget sweep of the router on the second held-out interval. |
|  | `audit_evidencepack_budget_pareto_V25.py` | Audit and statistics of the budget sweep. |
|  | `calibrate_longbench_risk_controller.py` | Calibrate the confidence-only gate. |
|  | `audit_longbench_risk_formal.py` | Audit of the confidence-only gate run. |
|  | `analyze_longbench_risk_formal_statistics.py` | Paired inference for the confidence-only gate. |
|  | `calibrate_longbench_utility_router.py` | Calibrate a multi-candidate utility router. |
|  | `calibrate_longbench_domain_utility_router.py` | Calibrate its domain-stable variant. |
|  | `audit_longbench_utility_formal.py` | Audit of the utility-routed run. |
|  | `analyze_longbench_utility_formal_statistics.py` | Paired inference for the utility-routed run. |
|  | `evaluate_longbench_confirmation_gate.py` | Apply a fixed gate to the confirmation slice. |

## RULER stress test

| | Script | What it does |
|---|---|---|
| ● | `prepare_official_ruler_v10_split.py` | Generate and fix the RULER split with the NVIDIA generator. |
|  | `freeze_official_ruler_v10_protocol.py` | Fix the protocol before any output exists. |
|  | `freeze_official_ruler_v10_controller.py` | Fix the routes from development runs only. |
| ● | `run_official_ruler_upgraded_v10.py` | ProofKV on the fixed RULER rows. |
|  | `official_ruler_v10_common.py` | Helpers shared by the RULER scripts. |
|  | `audit_official_ruler_v10.py` | Structural and semantic audit of the runs. |
|  | `audit_official_ruler_v10_protocol_binding.py` | Check that runs used the fixed implementation. |
|  | `audit_official_ruler_metric_parity.py` | Replay every row with RULER's own string metric. |
|  | `audit_official_ruler_extremes.py` | Audit of the lowest budgets. |
|  | `analyze_official_ruler_v10_statistics.py` | Paired task-stratified inference. |
|  | `analyze_official_ruler_v10_efficiency.py` | Physical accounting at RULER capacities. |
|  | `eval_official_ruler_mini.py` | Small RULER check with the NVIDIA generators. |
|  | `eval_deployable_ruler_lite.py` | Query-agnostic guidepost selectors on a local synthetic RULER-style task. |

## Index reuse, compact prefill, serving, and precision

| | Script | What it does |
|---|---|---|
| ● | `eval_multiquery_reuse.py` | One Evidence Index reused across several questions per document. |
|  | `analyze_reuse_q_20260726.py` | Break-even number of questions per document. |
| ● | `run_physical_kv_replay.py` | Compact prefill against gathering KV from a full prefill. |
|  | `benchmark_physical_path.py` | Timing of the compact-prefill path in an independent process. |
|  | `audit_physical_matrix_quality.py` | Audit of the timing matrices. |
| ● | `run_vllm_paged_suite.py` | End-to-end serving on the vLLM paged-KV backend. |
|  | `analyze_vllm_paged_suite.py` | Speedups and intervals of the serving runs. |
| ● | `run_tiered_kv_precision_pilot.py` | BF16 selection with an INT8 or INT4 auxiliary tier. |
|  | `analyze_tiered_precision_holdout.py` | Held-out confirmation of the INT8 tier. |

The follow-up analyses (compressor comparison, link removal across datasets, iterative retrieval, gap decomposition,
related distractors) live in [`experiments/followup/`](../experiments/followup/) and reuse these scripts unchanged.
