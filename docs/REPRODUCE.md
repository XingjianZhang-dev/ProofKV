# Reproducing the paper

Everything below runs from the repository root.

## Without a GPU

The repository holds the result records behind every table and figure, so the numbers of the paper can be checked
and its tables and figures regenerated on a laptop (Python 3.10 or later with NumPy and Matplotlib):

```bash
python analysis/verify_release.py
```

This recomputes the follow-up statistics from the per-example records, regenerates all 51 generated tables, compares
them byte for byte with `paper/tables/`, and checks that every record named by
`results/paper/canonical_results.json` is present. The individual steps are:

```bash
# follow-up analyses: per-example records -> statistics -> tables
python analysis/followup_statistics.py results/followup results/followup/summary.json
python analysis/make_followup_tables.py results/followup/summary.json paper/tables results/paper/canonical_results.json

# original evaluation: records -> tables
python analysis/make_paper_tables.py --out paper/tables

# all figures
python analysis/make_figures.py --paper results/paper/canonical_results.json \
    --routing results/paper/routing_summary.json --followup results/followup/summary.json --out paper/figures
```

The figures use Liberation Sans; Matplotlib falls back to its default font if it is not installed.

## Tables

Tables are LaTeX fragments (`tabular` environments) in `paper/tables/`, exactly as the paper includes them.

| Paper | File | Written by | Records |
|---|---|---|---|
| Table 1 | `paper/tables/checkpoints.tex` | written by hand (method settings) | none |
| Table 2 | `paper/tables/oracle_gap.tex` | `analysis/make_followup_tables.py` | `results/followup/heldout_selector/oracle_gap/` |
| Table 3 | `paper/tables/contract_transparent.tex` | `analysis/paper_tables/baseline_tables.py` | `results/paper/records/gpu_results_20260726/`, `results/paper/reuse_q_projection.json` |
| Table 4 | `paper/tables/packed_all.tex` | `analysis/make_followup_tables.py` | `results/followup/related_distractors/`, `results/paper/canonical_results.json` |
| Table 5 | `paper/tables/link_musique.tex` | `analysis/make_followup_tables.py` | `results/followup/link_interventions/` |
| Table 6 | `paper/tables/candidate_rules.tex` | written by hand (method settings) | none |
| Table 7 | `paper/tables/training_hparams.tex` | written by hand (method settings) | none |
| Table 8 | `paper/tables/training_stages.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 9 | `paper/tables/component_metrics.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 10 | `paper/tables/interpolation.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 11 | `paper/tables/matrix_hotpot.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 12 | `paper/tables/matrix_2wiki.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 13 | `paper/tables/matrix_musique.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 14 | `paper/tables/selector.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 15 | `paper/tables/hop_topology.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 16 | `paper/tables/topology.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 17 | `paper/tables/synthetic_topology.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 18 | `paper/tables/dense_core.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 19 | `paper/tables/premise_test.tex` | `analysis/make_followup_tables.py` | `results/followup/premise_test/` |
| Table 20 | `paper/tables/link_replication.tex` | `analysis/make_followup_tables.py` | `results/followup/link_interventions/` |
| Table 21 | `paper/tables/link_backends.tex` | `analysis/make_followup_tables.py` | `results/followup/link_interventions/attention_backends/` |
| Table 22 | `paper/tables/iterative_selector.tex` | `analysis/make_followup_tables.py` | `results/followup/heldout_selector/iterative_retrieval/` |
| Table 23 | `paper/tables/iterative_regime.tex` | `analysis/make_followup_tables.py` | `results/followup/budget_regime/` |
| Table 24 | worked examples in the paper's appendix | assembled by hand | `results/followup/heldout_selector/oracle_gap/` |
| Table 25 | `paper/tables/kvpress_matrix.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 26 | `paper/tables/kvzip_full.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 27 | `paper/tables/headkv_matrix.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 28 | `paper/tables/baseline_matrix.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 29 | `paper/tables/contract_endpoints.tex` | `analysis/paper_tables/baseline_tables.py` | `results/paper/records/gpu_results_20260726/`, `results/paper/reuse_q_projection.json` |
| Table 30 | `paper/tables/v10_external.tex` | `analysis/paper_tables/five_task_tables.py` | `results/paper/records/gpu_results_20260724/` |
| Table 31 | `paper/tables/v10_official_boundary.tex` | `analysis/paper_tables/five_task_tables.py` | `results/paper/records/gpu_results_20260724/` |
| Table 32 | `paper/tables/support_coverage.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 33 | `paper/tables/packing_seeds.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 34 | `paper/tables/packed_protocol.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 35 | `paper/tables/matrix_mistral02.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 36 | `paper/tables/matrix_mistral03.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 37 | `paper/tables/matrix_qwen3.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 38 | `paper/tables/qwen14b_scale_transfer.tex` | `analysis/paper_tables/qwen14b_table.py` | `results/paper/records/gpu_results_20260727/` |
| Table 39 | `paper/tables/context_halos.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 40 | `paper/tables/precision.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 41 | `paper/tables/v10_efficiency.tex` | `analysis/paper_tables/five_task_tables.py` | `results/paper/records/gpu_results_20260724/` |
| Table 42 | `paper/tables/serving.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 43 | `paper/tables/reuse_q.tex` | `analysis/paper_tables/baseline_tables.py` | `results/paper/records/gpu_results_20260726/`, `results/paper/reuse_q_projection.json` |
| Table 44 | `paper/tables/vllm_backend.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 45 | `paper/tables/route_frontier.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 46 | `paper/tables/multiquery.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 47 | `paper/tables/v22_risk_length.tex` | `analysis/paper_tables/routing_tables.py` | `results/paper/records/gpu_results_20260724/` |
| Table 48 | `paper/tables/v25_budget_pareto.tex` | `analysis/paper_tables/routing_tables.py` | `results/paper/records/gpu_results_20260724/` |
| Table 49 | `paper/tables/failure_gates.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 50 | `paper/tables/longbench.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 51 | `paper/tables/statistical_ledger.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 52 | `paper/tables/crossmodel_statistics.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 53 | `paper/tables/evaluation_inventory.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 54 | `paper/tables/policy_hyperparameters.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |
| Table 55 | `paper/tables/training_negatives.tex` | `analysis/paper_tables/main_tables.py` | `results/paper/canonical_results.json` |

Confidence intervals and tests are computed from unrounded per-example values with the seeds recorded in the
analysis scripts, so reruns give the same digits.

## Figures

`analysis/make_figures.py` draws every figure from `results/paper/canonical_results.json`,
`results/paper/routing_summary.json`, and `results/followup/summary.json`. Figure 1 is a schematic drawn by the same
script.

| Paper | File |
|---|---|
| Figure 1 | `paper/figures/overview.pdf` |
| Figure 2 | `paper/figures/evidence.pdf` |
| Figure 3 | `paper/figures/budget_regime.pdf` |
| Figure 4 | `paper/figures/checkpoint_selection.pdf` |
| Figure 5 | `paper/figures/downstream_landscape.pdf` |
| Figure 6 | `paper/figures/topology_depth.pdf` |
| Figure 7 | `paper/figures/packed_scaling.pdf` |
| Figure 8 | `paper/figures/compiler_diagnostics.pdf` |
| Figure 9 | `paper/figures/serving.pdf` |
| Figure 10 | `paper/figures/reuse_and_gate.pdf` |
| Figure 11 | `paper/figures/routing_frontier.pdf` |

## Re-running the experiments

Decoding needs one GPU; the paper used an NVIDIA A100 (80 GB) and an H100 NVL (94 GB). It also needs the
public decoders at the revisions in `models/public_models.json`, the datasets in [`DATA.md`](DATA.md), and the
checkpoints in [`../checkpoints/README.md`](../checkpoints/README.md). Each record directory under
`results/paper/records/` keeps the `run_manifest.json` of its run with the exact arguments, the selected question
identifiers, and the digests of the code and data it used. Paths that start with `/workspace/` in those manifests are
the paths on the machine that ran the experiment.

| Experiment | Entry point | Analysis |
|---|---|---|
| Scorer training and checkpoint interpolation | `scripts/train_typed_relation_scorer.py`, `scripts/interpolate_relation_checkpoints.py` | `scripts/eval_relation_scorer_components.py` |
| Held-out selector test | `scripts/eval_proofcache_selector.py` | `experiments/followup/heldout_selector/` |
| Fixed-budget generation on HotpotQA, 2WikiMultiHopQA, MuSiQue; packed contexts | `scripts/run_hotpot_compact_qa.py` | `scripts/audit_targeted_gpu_experiments.py` |
| Five-task suite | `scripts/run_longbench_structured_upgraded_v10.py` | `scripts/analyze_longbench_upgraded_v10_statistics.py` |
| KV-eviction baselines in their own setting | `scripts/run_official_longbench_v10_baseline.py`, `scripts/run_kvpress_natural_qa.py` | `scripts/analyze_transparent_baselines_20260726.py` |
| Same-scorer comparison and unseen tasks | `scripts/run_fair_contract_external_20260726.sh` | `scripts/analyze_fair_contract_external_20260726.py` |
| LLMLingua-2 at identical token counts | `scripts/run_llmlingua2_contract_20260726.sh` | `scripts/analyze_llmlingua2_contract_20260726.py` |
| Qwen2.5-14B transfer | `scripts/run_qwen14b_scale_20260727.sh` | `scripts/analyze_qwen14b_scale_20260727.py` |
| Equal-byte link interventions | `scripts/run_natural_topology_causality.py` | `experiments/followup/link_interventions/` |
| Margin routing | `scripts/run_evidencepack_risk_controller_V22_formal.sh`, `scripts/run_evidencepack_budget_pareto_V25.sh` | `scripts/audit_evidencepack_risk_controller_V22_formal.py`, `scripts/audit_evidencepack_budget_pareto_V25.py` |
| RULER stress test | `scripts/run_official_ruler_upgraded_v10.py` | `scripts/analyze_official_ruler_v10_statistics.py` |
| Index reuse across questions | `scripts/eval_multiquery_reuse.py` | `scripts/analyze_reuse_q_20260726.py` |
| Compact prefill and paged serving | `scripts/run_physical_kv_replay.py`, `scripts/run_vllm_paged_suite.py` | `scripts/audit_physical_matrix_quality.py`, `scripts/analyze_vllm_paged_suite.py` |
| Mixed-precision KV | `scripts/run_tiered_kv_precision_pilot.py` | `scripts/analyze_tiered_precision_holdout.py` |

The follow-up analyses (compressor comparison, link removal across datasets, iterative retrieval, gap decomposition,
related distractors) have their own commands in
[`../experiments/followup/README.md`](../experiments/followup/README.md).

## Names in code and records

Scripts and records keep the identifiers that were in use when the experiments ran. The paper uses these names:

| In code and records | In the paper |
|---|---|
| `learned_topk_paths` | ProofKV, chain union |
| `learned_and_or`, `learned_and_or_aux` | ProofKV, AND/OR selection (without and with auxiliary fill) |
| `learned_single_path` | Single chain |
| `learned_root_topk` | Same scorer, independent ranking ("Root") |
| `learned_iterative_retrieval_k*_h*` | Iterative retrieval with the same encoder |
| `reranker_proxy` | Hybrid reranker (reciprocal-rank fusion of BM25 and hashed-embedding ranks) |
| `graph_similarity_proxy` | Similarity graph |
| `entity_graph_adaptive` | Entity graph |
| `oracle_adaptive`, `oracle_proof` | Oracle (gold chain if it fits) |
| `proof_core` | Selected chains, intact |
| `relevance_same_bytes` | Relevance ranking at the same bytes |
| `edge_shuffled`, `random_edges`, `broken_middle` | Link targets shuffled, random links, middle span removed |
| `root`, `terminal`, `relation`/`edge` (heads and scores) | Start, end, and link |
| `proofcache` (module and file names) | The ProofKV evidence index and selector |
| `evidencepack`, `risk_controller` | Margin router |
| `a090`, `accepted_soup_a090` | Final checkpoint |
| `v6`, `locked_v6` | MuSiQue-replay checkpoint and the held-out MuSiQue test |
| `v10`, `longbench_structured` | Five-task suite |
| `fair_contract`, `contract` | Comparison under one shared interface and budget |
