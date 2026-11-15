"""Camera-ready wording for the generated tables.

The generators in this directory write the tables with the method and column names used while the experiments were
run. This module maps those names to the terminology of the paper. It changes labels only; every number is written by
the generators. The two rows added to the held-out selector table are filled from results/followup/summary.json.
"""
from __future__ import annotations

RENAMED_FILES = {"V22_risk_length.tex": "v22_risk_length.tex", "V25_budget_pareto.tex": "v25_budget_pareto.tex"}

EDITS = {
    "contract_transparent.tex": [("Model & KV & Full & Stream & Snap & Pyr. & Chunk & LL-2 & Root & \\proofkv \\\\",
                                  "Model & KV cap & Full & StreamingLLM & SnapKV & PyramidKV & ChunkKV & LLMLingua-2 & Root & \\proofkv \\\\")],
    "contract_endpoints.tex": [("Confirmatory endpoint", "Paired endpoint"),
                               ("Same scorer: typed proof $-$ root top-$k$", "Same scorer: \\proofkv $-$ Root"),
                               ("Parser-free external transfer", "Unseen tasks (Qasper, MultiFieldQA): \\proofkv $-$ Root"),
                               ("Exact-token: \\proofkv $-$ LL-2", "Identical tokens: \\proofkv $-$ LLMLingua-2")],
    "failure_gates.tex": [("Branch & Held-out gate statistic & Decision", "Candidate component & Held-out gate statistic & Adopted"),
                          ("Risk, fixed 8\\% & coverage 0.0\\%; error 0.0\\%; mean KV 100.0\\% & Fail", "Confidence-only gate, fixed 8\\% & coverage 0.0\\%; error 0.0\\%; mean KV 100.0\\% & No"),
                          ("Risk, adaptive & coverage 3.3\\%; error 14.7\\%; mean KV 97.2\\% & Fail", "Confidence-only gate, adaptive & coverage 3.3\\%; error 14.7\\%; mean KV 97.2\\% & No"),
                          ("Aux-cap repair & $\\Delta$F1 -0.70; $p=0.773$ & Fail", "Auxiliary-cap fill & $\\Delta$F1 $-0.70$; $p=0.773$ & No"),
                          ("Layer residual & $\\Delta$F1 +0.00 vs. matched shared core & Fail", "Layer-specific residual & $\\Delta$F1 $+0.00$ vs.\\ matched shared core & No")],
    "evaluation_inventory.tex": [("Evaluation block & Population & Conditions & Scientific role", "Evaluation block & Population & Conditions & Role"),
                                 ("Locked MuSiQue selector & 500 questions & 2 exact budgets & Primary PCR claim", "Held-out MuSiQue selector & 500 questions & 2 exact budgets & Complete-chain retention"),
                                 ("Controlled topology & 512 examples & 7 hop strata & Structural mechanism", "Generated chains & 512 examples & 7 depth strata & Depth behavior (development)"),
                                 ("Natural topology & 128 questions & 6 equal-byte routes & Causal downstream mechanism", "Equal-byte interventions & 128--372 questions & 6 equal-byte selections & Mechanism"),
                                 ("Risk/LongBench diagnostics & 2,048 / 128 & held-out gates & Scope and excluded branches", "Gate and LongBench~v2 diagnostics & 2,048 / 128 & held-out gates & Scope and design choices")],
    "statistical_ledger.tex": [("Locked MuSiQue PCR@256: paths--reranker", "Held-out MuSiQue PCR@256: chains--reranker"),
                               ("Locked MuSiQue PCR@512: paths--reranker", "Held-out MuSiQue PCR@512: chains--reranker"),
                               ("Hotpot F1@128: paths--CompressKV", "HotpotQA F1@128: chains--CompressKV"),
                               ("Hotpot F1@256: AND/OR+aux--CompressKV", "HotpotQA F1@256: AND/OR+aux--CompressKV"),
                               ("Hotpot F1@512: paths--CompressKV", "HotpotQA F1@512: chains--CompressKV"),
                               ("Packed 8k, three seeds: paths--Full", "Packed 8k, three seeds: chains--Full"),
                               ("Natural topology: core--shuffled", "MuSiQue interventions: chains--targets shuffled"),
                               ("Natural topology: core--random", "MuSiQue interventions: chains--random links"),
                               ("Natural topology: core--broken", "MuSiQue interventions: chains--middle span removed")],
    "synthetic_topology.tex": [("Typed paths &", "Chain union &"), ("AND/OR proofs &", "AND/OR selection &"), ("Broken proof &", "One span replaced &"),
                               ("Edge-shuffled &", "Link targets shuffled &"), ("Random edges &", "Random links &"), ("Gold recall & Route precision", "Gold recall & Precision")],
    "v22_risk_length.tex": [("Target & Controller F1 & $\\Delta$AND--OR & $\\Delta$Full", "Length & Router F1 & $\\Delta$ chains only & $\\Delta$ Full")],
    "v25_budget_pareto.tex": [("Budget & Retained & F1 & $\\Delta$Entity & $\\Delta$Root & $\\Delta$AND--OR", "Budget & Retained & F1 & $\\Delta$ Entity & $\\Delta$ Root & $\\Delta$ chains only")],
    "hop_topology.tex": [("Hops & $n$ & Intact & Shuffled & Broken & Oracle", "Links & $n$ & Intact & Shuffled & Replaced & Oracle")],
    "selector.tex": [("Reranker & 3.8 & 10.4 \\\\", "Hybrid reranker & 3.8 & 10.4 \\\\"),
                     ("Typed single path & 15.2 & 31.6 \\\\", "{FOLLOWUP_ROWS}Single chain & 15.2 & 31.6 \\\\"),
                     ("\\proofkv{} paths & 16.8 & 46.0", "\\proofkv{} chain union & 16.8 & 46.0"),
                     ("Oracle proof & 24.8 & 79.4", "Oracle (gold chain if it fits) & 24.8 & 79.4")],
    "topology.tex": [("512-token route & PCR & EM & F1", "512-token selection & PCR & EM & F1"),
                     ("Typed proof core &", "Selected chains (intact) &"), ("Relevance, same bytes &", "Relevance ranking, same bytes &"),
                     ("Edge-shuffled &", "Link targets shuffled &"), ("Random edges &", "Random links &"), ("Broken middle &", "Middle span removed &"),
                     ("Oracle proof &", "Oracle &")],
    "policy_hyperparameters.tex": [("Component & Frozen value", "Component & Value"),
                                   ("Dense edge core & 4 highest-confidence spans", "Dense link core & 4 highest-scoring spans"),
                                   ("Root frontier & 16 roots", "Start spans opened & 16"),
                                   ("Edge batch & 256 pairs", "Link batch & 256 pairs; kept if not \\textsc{none} and $e\\ge0.35$"),
                                   ("Path search & beam 8; at most 8 hops", "Chain search & beam 8; at most 8 links; at least 2 spans; end-span floor 0.20"),
                                   ("Proof set & at most 8 proofs", "Proof set & at most 8 proofs"),
                                   ("Length prior & $\\exp(-0.04)$ per hop", "Score priors & $\\gamma=\\exp(-0.04)$ per link; contradiction links $\\times0.55$ unless the question is negated"),
                                   ("Boundary context & 8 tokens per retained side", "Boundary halo & optional; $h{=}0$ in all accuracy results, $h{=}8$ on the fidelity probe"),
                                   ("Precision & BF16 proof core; optional INT8 auxiliary", "Precision & BF16 selection; optional INT8 auxiliary tier")],
    "longbench.tex": [("\\proofkv{} paths &", "\\proofkv{} chain union &")],
    "training_stages.tex": [("v8 Hotpot boost", "v8 HotpotQA-focused")],
}

def followup_rows(summary: dict) -> str:
    """Rows of the held-out selector table that come from the follow-up records."""
    pcr = summary["heldout_selector"]["pcr"]
    same, iterative = pcr["same_scorer_independent"], pcr["iterative_hop_order"]
    return (f"Same scorer, independent & {same['256']:.1f} & {same['512']:.1f} \\\\\n"
            f"Iterative retrieval (same encoder) & {iterative['256']:.1f} & {iterative['512']:.1f} \\\\\n")


def apply(table_dir, summary: dict) -> int:
    """Rename and relabel the generated tables in place; return the number of tables changed."""
    from pathlib import Path
    table_dir = Path(table_dir)
    for old, new in RENAMED_FILES.items():
        if (table_dir / old).exists():
            (table_dir / old).rename(table_dir / new)
    rows = followup_rows(summary)
    for name, pairs in EDITS.items():
        path = table_dir / name
        text = path.read_text()
        for old, new in pairs:
            if old not in text:
                raise SystemExit(f"{name}: pattern not found: {old}")
            text = text.replace(old, new.replace("{FOLLOWUP_ROWS}", rows))
        path.write_text(text)
    return len(EDITS)
