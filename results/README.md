# Results

| Path | Contents |
|---|---|
| `paper/canonical_results.json` | Every number of the original evaluation, with the record file each one was read from (`source`) |
| `paper/records/` | The records named by `canonical_results.json`: per-example predictions and scores (`results.jsonl`), aggregate metrics, statistical audits, selection manifests, and run manifests. Directory names are those of the runs |
| `paper/routing_summary.json`, `paper/reuse_q_projection.json` | Aggregates of the margin-routing and index-reuse experiments used by the figures and tables |
| `paper/reference/` | Metrics of the original MuSiQue link-intervention run, used by the attention-backend check |
| `followup/` | Per-example records of the follow-up analyses (`results.jsonl` and `run_info.json` per run) |
| `followup/summary.json` | Every follow-up statistic of the paper, recomputed from those records by `analysis/followup_statistics.py` |

Follow-up record directories and the paper:

| Directory | Analysis in the paper |
|---|---|
| `followup/premise_test/compressors/` | Gold-span coverage and F1 of SnapKV, PyramidKV, H2O-style eviction, and StreamingLLM |
| `followup/premise_test/chain_and_oracles/` | The same slice with ProofKV selections, span-level baselines, and gold-informed references |
| `followup/link_interventions/` | One link removed at equal bytes on MuSiQue, HotpotQA (sentence and paragraph spans), and 2WikiMultiHopQA; `attention_backends/` repeats MuSiQue under three attention implementations |
| `followup/heldout_selector/oracle_gap/` | Decomposition of the gap to the oracle on the 500 held-out MuSiQue questions |
| `followup/heldout_selector/iterative_retrieval/` | Iterative retrieval with the same encoder on the same 500 questions |
| `followup/budget_regime/` | MuSiQue generation at seven budgets and the per-question chain features |
| `followup/related_distractors/` | HotpotQA packed with topically related paragraphs |

`python analysis/verify_release.py` checks that these records reproduce the committed summary and tables.
