#!/usr/bin/env python3
"""Generate LaTeX tables directly from the canonical result ledger."""

from __future__ import annotations

import json
from pathlib import Path


import os

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results" / "paper" / "canonical_results.json"
OUT = Path(os.environ.get("PROOFKV_TABLE_DIR", REPO / "paper" / "tables"))


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}"


def row_for(rows: list[dict], **criteria: object) -> dict:
    hits = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in criteria.items())
    ]
    if len(hits) != 1:
        raise AssertionError(f"Expected one row for {criteria}, found {len(hits)}")
    return hits[0]


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def packed_scale(data: dict) -> None:
    rows = data["packed_hotpot_scale"]["rows"]
    body: list[str] = []
    for target in (8000, 16000, 32000):
        full = row_for(rows, target_prompt_tokens=target, method="full")
        entity = row_for(
            rows, target_prompt_tokens=target, method="entity_graph_adaptive"
        )
        proof = row_for(
            rows, target_prompt_tokens=target, method="learned_topk_paths"
        )
        oracle = row_for(
            rows, target_prompt_tokens=target, method="oracle_adaptive"
        )
        body.append(
            f"{target // 1000}k & {pct(full['answer_f1'])} & "
            f"{pct(entity['answer_f1'])} / {pct(entity['retained_ratio'])} & "
            f"\\textbf{{{pct(proof['answer_f1'])}}} / "
            f"\\textbf{{{pct(proof['retained_ratio'])}}} & "
            f"{100 * (proof['answer_f1'] - full['answer_f1']):+.1f} & "
            f"{pct(oracle['answer_f1'])} \\\\"
        )
    write(
        "packed_scale.tex",
        [
            "\\begin{tabular}{@{}lccccc@{}}",
            "\\toprule",
            "Context & Full F1 & Entity F1 / KV\\% & \\proofkv{} F1 / KV\\% "
            "& $\\Delta$Full & Oracle F1 \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def fixed_budget(data: dict) -> None:
    main = data["hotpot_fixed_budget"]["rows"]
    baselines = data["hotpot_fixed_budget"]["baseline_rows"]
    body: list[str] = []
    for budget in (128, 256, 512):
        official = row_for(
            baselines,
            budget_tokens=budget,
            family="official_compresskv",
            method="compresskv",
        )
        entity = row_for(main, budget_tokens=budget, method="entity_graph_adaptive")
        paths = row_for(main, budget_tokens=budget, method="learned_topk_paths")
        andor = row_for(main, budget_tokens=budget, method="learned_and_or_aux")
        body.append(
            f"{budget} & {pct(entity['answer_f1'])} & "
            f"{pct(official['answer_f1'])} & {pct(paths['answer_f1'])} & "
            f"{pct(andor['answer_f1'])} \\\\"
        )
    write(
        "fixed_budget.tex",
        [
            "\\begin{tabular}{@{}lcccc@{}}",
            "\\toprule",
            "Tokens & Entity & CompressKV & \\proofkv{} paths & "
            "\\proofkv{} AND/OR+aux \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def baseline_matrix(data: dict) -> None:
    main = data["hotpot_fixed_budget"]["rows"]
    baselines = data["hotpot_fixed_budget"]["baseline_rows"]
    methods = [
        ("kvpress", "streamingllm", "StreamingLLM"),
        ("h2o_style", "observed_attention_h2o_style", "H$_2$O-style"),
        ("kvpress", "snapkv", "SnapKV"),
        ("kvpress", "pyramidkv", "PyramidKV"),
        ("kvpress", "chunkkv_snapkv", "ChunkKV"),
    ]
    body: list[str] = []
    for family, method, label in methods:
        values = []
        for budget in (128, 256):
            row = row_for(
                baselines,
                budget_tokens=budget,
                family=family,
                method=method,
            )
            values.append(pct(row["answer_f1"]))
        body.append(f"{label} & {values[0]} & {values[1]} \\\\")

    official_values = []
    entity_values = []
    paths_values = []
    andor_values = []
    for budget in (128, 256, 512):
        official_values.append(
            pct(
                row_for(
                    baselines,
                    budget_tokens=budget,
                    family="official_compresskv",
                    method="compresskv",
                )["answer_f1"]
            )
        )
        entity_values.append(
            pct(
                row_for(
                    main, budget_tokens=budget, method="entity_graph_adaptive"
                )["answer_f1"]
            )
        )
        paths_values.append(
            pct(
                row_for(main, budget_tokens=budget, method="learned_topk_paths")[
                    "answer_f1"
                ]
            )
        )
        andor_values.append(
            pct(
                row_for(main, budget_tokens=budget, method="learned_and_or_aux")[
                    "answer_f1"
                ]
            )
        )
    body.extend(
        [
            "\\midrule",
            f"CompressKV (official) & \\textbf{{{official_values[0]}}} & "
            f"{official_values[1]} \\\\",
            f"Entity graph & {entity_values[0]} & {entity_values[1]} \\\\",
            f"\\proofkv{{}} paths & {paths_values[0]} & {paths_values[1]} \\\\",
            f"\\proofkv{{}} AND/OR+aux & {andor_values[0]} & "
            f"\\textbf{{{andor_values[1]}}} \\\\",
            "\\addlinespace[1pt]",
            "\\multicolumn{3}{@{}l}{\\footnotesize 512 tok.: "
            f"CompressKV {official_values[2]}; Entity {entity_values[2]}; "
            f"\\proofkv{{}} \\textbf{{{paths_values[2]}}}; "
            f"AND/OR+aux {andor_values[2]}}} \\\\",
        ]
    )
    write(
        "baseline_matrix.tex",
        [
            "\\begin{tabular}{@{}lrr@{}}",
            "\\toprule",
            "Method & F1@128 & F1@256 \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def support_coverage(data: dict) -> None:
    rows = data["hotpot_support_coverage"]["rows"]
    body = []
    for budget in (128, 256, 512):
        row = row_for(rows, budget_tokens=budget)
        body.append(
            f"{budget} & {row['complete_n']}/{row['n']} "
            f"({pct(row['complete_rate'])}\\%) & "
            f"{pct(row['f1_complete'])} & {pct(row['f1_incomplete'])} & "
            f"{100 * row['f1_gap']:+.1f} \\\\"
        )
    write(
        "support_coverage.tex",
        [
            "\\begin{tabular}{@{}lcccc@{}}",
            "\\toprule",
            "Tokens & Complete support & F1 complete & F1 incomplete & $\\Delta$ \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def kvzip_contract(data: dict) -> None:
    rows = data["hotpot_fixed_budget"]["baseline_rows"]
    body = []
    for budget in (128, 256, 512):
        row = row_for(
            rows,
            budget_tokens=budget,
            family="official_kvzip",
            method=f"kvzip_fixed{budget}",
        )
        body.append(
            f"{budget} & {pct(row['answer_f1'])} & "
            f"{pct(row['logical_retained_ratio'])} & "
            f"{pct(row['mean_physical_tensor_ratio_to_full'])} \\\\"
        )
    write(
        "kvzip_contract.tex",
        [
            "\\begin{tabular}{@{}lccc@{}}",
            "\\toprule",
            "Logical tokens & F1 & Logical KV\\% & Physical tensor\\% \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def cross_model(data: dict) -> None:
    qwen = data["hotpot_fixed_budget"]["rows"]
    other = data["cross_model_hotpot"]["rows"]
    specs = [
        ("Qwen2.5-7B", qwen, None),
        ("Mistral-7B-v0.3", other, "Mistral-7B-Instruct-v0.3"),
        ("Qwen3-8B", other, "Qwen3-8B"),
    ]
    body: list[str] = []
    for model, rows, model_name in specs:
        for index, budget in enumerate((128, 256, 512)):
            criteria = {"budget_tokens": budget}
            if model_name is not None:
                criteria["model_name"] = model_name
            entity = row_for(rows, method="entity_graph_adaptive", **criteria)
            proof = row_for(rows, method="learned_topk_paths", **criteria)
            prefix = f"\\multirow{{3}}{{*}}{{{model}}}" if index == 0 else ""
            body.append(
                f"{prefix} & {budget} & {pct(entity['answer_f1'])} & "
                f"\\textbf{{{pct(proof['answer_f1'])}}} & "
                f"{100 * (proof['answer_f1'] - entity['answer_f1']):+.1f} \\\\"
            )
        if model != specs[-1][0]:
            body.append("\\addlinespace[1pt]")
    write(
        "cross_model.tex",
        [
            "\\begin{tabular}{@{}llccc@{}}",
            "\\toprule",
            "Model & Tokens & Entity F1 & \\proofkv{} F1 & $\\Delta$ \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def selector(data: dict) -> None:
    rows = data["locked_musique_selector"]["rows"]
    methods = [
        ("bm25", "BM25"),
        ("reranker_proxy", "Reranker"),
        ("graph_similarity_proxy", "Similarity graph"),
        ("learned_single_path", "Typed single path"),
        ("learned_topk_paths", "\\proofkv{} paths"),
        ("learned_and_or", "\\proofkv{} AND/OR"),
        ("oracle_proof", "Oracle proof"),
    ]
    body: list[str] = []
    for method, label in methods:
        b256 = row_for(rows, method=method, budget_spec="256")
        b512 = row_for(rows, method=method, budget_spec="512")
        body.append(f"{label} & {pct(b256['pcr'])} & {pct(b512['pcr'])} \\\\")
    write(
        "selector.tex",
        [
            "\\begin{tabular}{@{}lcc@{}}",
            "\\toprule",
            "Selector & PCR@256 & PCR@512 \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def topology(data: dict) -> None:
    rows = data["natural_topology_downstream"]["records"]
    methods = [
        ("proof_core", "Typed proof core"),
        ("relevance_same_bytes", "Relevance, same bytes"),
        ("edge_shuffled", "Edge-shuffled"),
        ("random_edges", "Random edges"),
        ("broken_middle", "Broken middle"),
        ("oracle_proof", "Oracle proof"),
    ]
    body: list[str] = []
    for method, label in methods:
        row = row_for(rows, method=method)
        body.append(
            f"{label} & {pct(row['gold_proof_complete'])} & "
            f"{pct(row['answer_em'])} & {pct(row['answer_f1'])} \\\\"
        )
    write(
        "topology.tex",
        [
            "\\begin{tabular}{@{}lccc@{}}",
            "\\toprule",
            "512-token route & PCR & EM & F1 \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def synthetic_topology(data: dict) -> None:
    rows = data["synthetic_topology"]["records"]
    methods = [
        ("learned_topk_paths", "Typed paths"),
        ("learned_and_or", "AND/OR proofs"),
        ("broken_proof_same_count", "Broken proof"),
        ("edge_shuffled", "Edge-shuffled"),
        ("random_edges", "Random edges"),
        ("oracle", "Oracle"),
    ]
    body: list[str] = []
    for method, label in methods:
        row = row_for(rows, method=method)
        body.append(
            f"{label} & {pct(row['pcr'])} & {pct(row['gold_recall'])} & "
            f"{pct(row['route_precision'])} \\\\"
        )
    write(
        "synthetic_topology.tex",
        [
            "\\begin{tabular}{@{}lccc@{}}",
            "\\toprule",
            "256-token selector & PCR & Gold recall & Route precision \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def serving(data: dict) -> None:
    audit = data["targeted_independent_audit"]["group1"]
    records: dict[tuple[str, int], dict[str, dict]] = {}
    for group in audit:
        path = group["path"]
        context = "8k" if "8k" in path else ("32k" if "32k" in path else "30k")
        for row in group["comparisons"]:
            records.setdefault((context, row["output_tokens"]), {})[
                row["mode"]
            ] = row
    body: list[str] = []
    for context, output in (("8k", 256), ("8k", 1024), ("32k", 256), ("30k", 1024)):
        group = records[(context, output)]
        body.append(
            f"{context} / {output} & "
            f"{group['precompiled']['paired_speedup']['mean']:.2f}$\\times$ & "
            f"\\textbf{{{group['shared_index']['paired_speedup']['mean']:.2f}$\\times$}} & "
            f"{group['cold']['paired_speedup']['mean']:.2f}$\\times$ \\\\"
        )
    write(
        "serving.tex",
        [
            "\\begin{tabular}{@{}lccc@{}}",
            "\\toprule",
            "Context / output & Precompiled & Reusable E2E & Cold E2E \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def diagnostics(data: dict) -> None:
    tier = data["tiered_precision"]["summaries"][0]
    longbench = data["longbench_V2"]
    lbr = {row["method"]: row for row in longbench["records"]}
    contextual = data["contextualization_gap"]["sample_records"]
    multi0 = row_for(contextual, task_type="multi_hop", repair_halo=0)
    multi8 = row_for(contextual, task_type="multi_hop", repair_halo=8)
    write(
        "diagnostics.tex",
        [
            "\\begin{tabular}{@{}p{0.25\\columnwidth}p{0.66\\columnwidth}@{}}",
            "\\toprule",
            "Diagnostic & Result \\\\",
            "\\midrule",
            "INT8 auxiliary & "
            f"{tier['mean_storage_reduction_factor']:.2f}$\\times$ additional storage "
            f"reduction; {pct(tier['exact_generation_match_rate'])}\\% exact generation "
            f"match; $\\Delta$F1={tier['answer_f1_delta']:.3f} ($n={tier['n']}$). \\\\",
            "Boundary repair & "
            f"Multi-hop correctness {pct(multi0['compact_correctness'])}$\\rightarrow$"
            f"{pct(multi8['compact_correctness'])} with an 8-token halo "
            f"($n={multi8['n']}$). \\\\",
            "LongBench V2 & "
            f"Full {pct(lbr['full']['accuracy'])}, \\proofkv{{}} "
            f"{pct(lbr['learned_topk_paths']['accuracy'])}; "
            f"{longbench['truncated_samples']}/{longbench['max_samples']} inputs truncated. \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def downstream_matrix_table(data: dict, key: str, output: str) -> None:
    rows = data["downstream_matrices"][key]["rows"]
    fixed_methods = [
        ("compass_fixed", "Relevance"),
        ("entity_graph_adaptive", "Entity"),
        ("learned_topk_paths", "\\proofkv{} paths"),
        ("learned_and_or_aux", "AND/OR+aux"),
        ("oracle_adaptive", "Oracle"),
    ]
    body: list[str] = []
    for budget_name, label in (
        ("fixed128", "128 tok."),
        ("fixed256", "256 tok."),
        ("fixed512", "512 tok."),
    ):
        budget_rows = [row for row in rows if row["budget_name"] == budget_name]
        cells: list[str] = []
        for method, _ in fixed_methods:
            cells.append(pct(row_for(budget_rows, method=method)["answer_f1"]))
        proof = row_for(budget_rows, method="learned_topk_paths")
        body.append(
            f"{label} & {' & '.join(cells)} & {pct(proof['retained_ratio'])} \\\\"
        )
    body.extend(
        [
            "\\addlinespace[2pt]",
            "\\multicolumn{7}{@{}l}{\\textit{Ratio budgets: all six executed routes}} \\\\",
            "Budget & Full & Relevance & Entity & \\proofkv{} paths & "
            "AND/OR+aux & Oracle \\\\",
            "\\midrule",
        ]
    )
    for budget_name, label in (
        ("ratio004", "4\\%"),
        ("ratio008", "8\\%"),
        ("ratio016", "16\\%"),
    ):
        budget_rows = [row for row in rows if row["budget_name"] == budget_name]
        full = row_for(budget_rows, method="full")
        relevance = row_for(budget_rows, method="compass_fixed")
        entity = row_for(budget_rows, method="entity_graph_adaptive")
        proof = row_for(budget_rows, method="learned_topk_paths")
        andor = row_for(budget_rows, method="learned_and_or_aux")
        oracle = row_for(budget_rows, method="oracle_adaptive")
        body.append(
            f"{label} & {pct(full['answer_f1'])} & "
            f"{pct(relevance['answer_f1'])} & {pct(entity['answer_f1'])} & "
            f"{pct(proof['answer_f1'])} & {pct(andor['answer_f1'])} & "
            f"{pct(oracle['answer_f1'])} \\\\"
        )
    write(
        output,
        [
            "\\begin{tabular}{@{}lcccccc@{}}",
            "\\toprule",
            "Budget & Relevance & Entity & \\proofkv{} paths & AND/OR+aux & "
            "Oracle & Path KV\\% \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def interpolation(data: dict) -> None:
    rows = data["checkpoint_interpolation"]["rows"]
    body: list[str] = []
    for alpha in (0.70, 0.78, 0.84, 0.90, 0.95):
        hotpot_selector = row_for(
            rows, alpha=alpha, gate="hotpot_selector", method="learned_topk_paths"
        )
        musique = row_for(
            rows,
            alpha=alpha,
            gate="musique_dev_selector",
            method="learned_topk_paths",
        )
        downstream = [
            row
            for row in rows
            if row["alpha"] == alpha
            and row["gate"] == "hotpot_downstream"
            and row["method"] == "learned_topk_paths"
        ]
        f1 = pct(downstream[0]["answer_f1"]) if downstream else "gate-excluded"
        prefix = "\\textbf{" if alpha == 0.90 else ""
        suffix = "}" if alpha == 0.90 else ""
        body.append(
            f"{prefix}{alpha:.2f}{suffix} & "
            f"{prefix}{pct(hotpot_selector['pcr'])}{suffix} & "
            f"{prefix}{pct(musique['pcr'])}{suffix} & "
            f"{prefix}{f1}{suffix} \\\\"
        )
    write(
        "interpolation.tex",
        [
            "\\begin{tabular}{@{}lccc@{}}",
            "\\toprule",
            "$\\alpha$ (v8 weight) & Hotpot PCR@8\\% & MuSiQue PCR@512 & "
            "Hotpot F1@8\\% \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def training_stages(data: dict) -> None:
    body: list[str] = []
    for row in data["training_and_components"]["training_runs"]:
        samples = (
            row["hotpot_samples"]
            + row["two_wiki_samples"]
            + row["musique_samples"]
            + row["synthetic_samples"]
        )
        body.append(
            f"{row['stage']} & {samples:,} & {row['training_items']:,} & "
            f"{row['steps']:,} & {row['loss']:.3f} & "
            f"{row['elapsed_seconds'] / 60:.1f} \\\\"
        )
    write(
        "training_stages.tex",
        [
            "\\begin{tabular}{@{}lrrrrr@{}}",
            "\\toprule",
            "Stage & Examples & Items & Steps & Loss & Minutes \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def component_metrics(data: dict) -> None:
    datasets = data["training_and_components"]["component_metrics"]
    body: list[str] = []
    for dataset_key, label in (
        ("hotpot_validation", "HotpotQA validation"),
        ("neutral_topology_ood", "Neutral topology OOD"),
    ):
        metrics = datasets[dataset_key]
        for index, component in enumerate(("root", "terminal", "edge")):
            row = metrics[component]
            prefix = f"\\multirow{{3}}{{*}}{{{label}}}" if index == 0 else ""
            body.append(
                f"{prefix} & {component.title()} & {row['n']:,} & "
                f"{pct(row['f1'])} & {pct(row['auroc'])} & "
                f"{pct(row['false_positive_rate'])} \\\\"
            )
        if dataset_key == "hotpot_validation":
            body.append("\\addlinespace[1pt]")
    write(
        "component_metrics.tex",
        [
            "\\begin{tabular}{@{}llrrrr@{}}",
            "\\toprule",
            "Set & Head & $n$ & F1 & AUROC & FPR \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def hop_topology(data: dict) -> None:
    rows = data["topology_full_matrix"]["records"]
    body: list[str] = []
    for hops in ("1", "2", "3", "4", "5", "6", "8"):
        intact = row_for(
            rows,
            axis="hops",
            axis_value=hops,
            budget_spec="512",
            method="learned_and_or",
        )
        shuffled = row_for(
            rows,
            axis="hops",
            axis_value=hops,
            budget_spec="512",
            method="edge_shuffled",
        )
        broken = row_for(
            rows,
            axis="hops",
            axis_value=hops,
            budget_spec="512",
            method="broken_proof_same_count",
        )
        oracle = row_for(
            rows,
            axis="hops",
            axis_value=hops,
            budget_spec="512",
            method="oracle",
        )
        body.append(
            f"{hops} & {intact['n']} & \\textbf{{{pct(intact['pcr'])}}} & "
            f"{pct(shuffled['pcr'])} & {pct(broken['pcr'])} & "
            f"{pct(oracle['pcr'])} \\\\"
        )
    write(
        "hop_topology.tex",
        [
            "\\begin{tabular}{@{}rrrrrr@{}}",
            "\\toprule",
            "Hops & $n$ & Intact & Shuffled & Broken & Oracle \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def dense_core(data: dict) -> None:
    rows = data["dense_core_sweep"]["rows"]
    body: list[str] = []
    for core in (0, 4, 8, 16):
        hotpot = row_for(
            rows,
            dense_core=core,
            task="hotpot",
            method="learned_topk_paths",
            budget_spec="0.0800",
        )
        topology = row_for(
            rows,
            dense_core=core,
            task="topology",
            method="learned_and_or",
            axis="overall",
            budget_spec="256",
        )
        body.append(
            f"{core} & {pct(hotpot['pcr'])} & {pct(hotpot['route_precision'])} & "
            f"{pct(topology['pcr'])} & {pct(topology['route_precision'])} \\\\"
        )
    write(
        "dense_core.tex",
        [
            "\\begin{tabular}{@{}rcccc@{}}",
            "\\toprule",
            "Dense core & Hotpot PCR & Hotpot precision & Topology PCR & "
            "Topology precision \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def kvpress_matrix(data: dict) -> None:
    rows = data["kvpress_full_matrix"]["rows"]
    methods = [
        ("streamingllm", "StreamingLLM"),
        ("snapkv", "SnapKV"),
        ("pyramidkv", "PyramidKV"),
        ("chunkkv_snapkv", "ChunkKV"),
    ]
    body: list[str] = []
    for method, label in methods:
        cells: list[str] = []
        ratios: list[str] = []
        for budget in ("ratio004", "ratio008", "ratio016"):
            row = row_for(rows, budget_name=budget, method=method)
            cells.append(pct(row["answer_f1"]))
            ratios.append(pct(row["actual_retained_ratio"]))
        body.append(
            f"{label} & {cells[0]} & {cells[1]} & {cells[2]} & "
            f"{ratios[0]}/{ratios[1]}/{ratios[2]} \\\\"
        )
    write(
        "kvpress_matrix.tex",
        [
            "\\begin{tabular}{@{}lcccc@{}}",
            "\\toprule",
            "Method & F1@4\\% & F1@8\\% & F1@16\\% & Actual KV\\% (4/8/16) \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def kvzip_full(data: dict) -> None:
    rows = data["official_kvzip_full_matrix"]["records"]
    body: list[str] = []
    for method, label in (
        ("full", "Full"),
        ("kvzip_fixed128", "Fixed 128"),
        ("kvzip_fixed256", "Fixed 256"),
        ("kvzip_fixed512", "Fixed 512"),
        ("kvzip_ratio004", "Ratio 4\\%"),
        ("kvzip_ratio008", "Ratio 8\\%"),
        ("kvzip_ratio016", "Ratio 16\\%"),
    ):
        row = row_for(rows, method=method)
        body.append(
            f"{label} & {pct(row['answer_f1'])} & "
            f"{pct(row['logical_retained_ratio'])} & "
            f"{pct(row['mean_physical_tensor_ratio_to_full'])} & "
            f"{row['offline_index_seconds']:.3f} \\\\"
        )
    write(
        "kvzip_full.tex",
        [
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Setting & F1 & Logical KV\\% & Physical tensor\\% & Index s \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def headkv_matrix(data: dict) -> None:
    rows = data["official_headkv_matrix"]["rows"]
    body: list[str] = []
    for budget_name, label in (
        ("fixed32", "32"),
        ("fixed32_beta10_tuned", "32, tuned $\\beta$"),
        ("fixed64", "64"),
        ("fixed128", "128"),
        ("fixed256", "256"),
        ("fixed512", "512"),
    ):
        row = row_for(rows, budget_name=budget_name)
        body.append(
            f"{label} & {row['beta']:.3g} & {pct(row['answer_f1'])} & "
            f"{row['native_gqa_equivalent_tokens_mean']:.0f} & "
            f"{pct(row['actual_kv_ratio_mean'])} \\\\"
        )
    write(
        "headkv_matrix.tex",
        [
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Nominal tokens/head & $\\beta$ & F1 & GQA-equivalent tokens & "
            "Actual KV\\% \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def packing_seeds(data: dict) -> None:
    body: list[str] = []
    for summary in data["packed_hotpot_seed_replication"]["summaries"]:
        methods = summary["methods"]
        proof = methods["learned_topk_paths"]
        full = methods["full"]
        entity = methods["entity_graph_adaptive"]
        body.append(
            f"{summary['label'].replace('seed', '')} & {pct(full['answer_f1'])} & "
            f"{pct(entity['answer_f1'])} & \\textbf{{{pct(proof['answer_f1'])}}} & "
            f"{100 * (proof['answer_f1'] - full['answer_f1']):+.1f} & "
            f"{pct(proof['retained_ratio'])} \\\\"
        )
    write(
        "packing_seeds.tex",
        [
            "\\begin{tabular}{@{}rrrrrr@{}}",
            "\\toprule",
            "Seed & Full F1 & Entity F1 & \\proofkv{} F1 & $\\Delta$Full & KV\\% \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def packed_protocol(data: dict) -> None:
    body: list[str] = []
    for row in data["packed_protocol"]["audit"]["summaries"]:
        body.append(
            f"{row['target_prompt_tokens'] // 1000}k & {row['n']} & "
            f"{100 * row['utilization_min']:.2f} & "
            f"{100 * row['utilization_median']:.2f} & "
            f"{100 * row['utilization_max']:.2f} & "
            f"{row['samples_with_issues']} \\\\"
        )
    write(
        "packed_protocol.tex",
        [
            "\\begin{tabular}{@{}rrrrrr@{}}",
            "\\toprule",
            "Target & $n$ & Min util.\\% & Median\\% & Max\\% & Issues \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def context_halos(data: dict) -> None:
    body: list[str] = []
    for task, label in (("exact_retrieval", "Exact retrieval"), ("multi_hop", "Multi-hop")):
        for index, halo in enumerate((0, 8, 16)):
            row = row_for(
                data["contextualization_gap"]["sample_records"],
                task_type=task,
                repair_halo=halo,
            )
            prefix = f"\\multirow{{3}}{{*}}{{{label}}}" if index == 0 else ""
            body.append(
                f"{prefix} & {halo} & {pct(row['retained_ratio'])} & "
                f"{row['first_token_kl']:.4f} & "
                f"{pct(row['top_token_agreement'])} & "
                f"{pct(row['compact_correctness'])} \\\\"
            )
        if task == "exact_retrieval":
            body.append("\\addlinespace[1pt]")
    write(
        "context_halos.tex",
        [
            "\\begin{tabular}{@{}llrrrr@{}}",
            "\\toprule",
            "Task & Halo & KV\\% & First-token KL & Top-1 agree & Accuracy \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def multiquery(data: dict) -> None:
    body: list[str] = []
    for row in data["multiquery"]["records"]:
        body.append(
            f"{row['queries']} & {row['amortized_offline_index_ms']:.1f} & "
            f"{row['mean_overlay_ms']:.1f} & "
            f"{pct(row['mean_pairwise_overlay_jaccard'])} & "
            f"{pct(row['union_context_ratio'])} & "
            f"{row['mean_incremental_context_tokens']:.1f} \\\\"
        )
    write(
        "multiquery.tex",
        [
            "\\begin{tabular}{@{}rrrrrr@{}}",
            "\\toprule",
            "Queries & Index ms/q & Overlay ms/q & Pair Jaccard\\% & Union\\% & "
            "New tokens/q \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def precision_table(data: dict) -> None:
    body: list[str] = []
    for row in data["tiered_precision_pilot"]["summaries"]:
        body.append(
            f"Pilot & {row['precision'].upper()} & {row['n']} & "
            f"{row['mean_storage_reduction_factor']:.2f}$\\times$ & "
            f"{pct(row['exact_generation_match_rate'])} & "
            f"{100 * row['answer_f1_delta']:+.2f} & "
            f"{'Pass' if row['passed'] else 'Fail'} \\\\"
        )
    row = data["tiered_precision"]["summaries"][0]
    body.append(
        f"Holdout & INT8 & {row['n']} & "
        f"{row['mean_storage_reduction_factor']:.2f}$\\times$ & "
        f"{pct(row['exact_generation_match_rate'])} & "
        f"{100 * row['answer_f1_delta']:+.2f} & Pass \\\\"
    )
    write(
        "precision.tex",
        [
            "\\begin{tabular}{@{}llrrrrl@{}}",
            "\\toprule",
            "Split & Precision & $n$ & Storage reduction & Exact gen.\\% & "
            "$\\Delta$F1 & Gate \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def failure_gates(data: dict) -> None:
    fixed = data["risk_controller"]["fixed_8pct"]
    adaptive = data["risk_controller"]["adaptive"]["test"]
    aux = data["auxiliary_holdout"]
    layer = data["layer_specific_residual"]
    body = [
        f"Risk, fixed 8\\% & coverage {pct(fixed['test_coverage'])}\\%; "
        f"error {pct(fixed['test_selective_error'])}\\%; "
        f"mean KV {pct(fixed['mean_effective_kv_ratio'])}\\% & Fail \\\\",
        f"Risk, adaptive & coverage {pct(adaptive['coverage'])}\\%; "
        f"error {pct(adaptive['selective_error'])}\\%; "
        f"mean KV {pct(adaptive['mean_kv_ratio'])}\\% & Fail \\\\",
        f"Aux-cap repair & $\\Delta$F1 "
        f"{100 * aux['repair_minus_core_f1']:+.2f}; "
        f"$p={aux['cluster_sign_flip_p_two_sided']:.3f}$ & Fail \\\\",
        f"Layer residual & $\\Delta$F1 "
        f"{100 * layer['comparisons']['minus_shared_core_aux']:+.2f} "
        "vs. matched shared core & Fail \\\\",
    ]
    write(
        "failure_gates.tex",
        [
            "\\begin{tabular}{@{}lp{0.64\\textwidth}l@{}}",
            "\\toprule",
            "Branch & Held-out gate statistic & Decision \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def longbench(data: dict) -> None:
    body: list[str] = []
    for method, label in (
        ("full", "Full"),
        ("entity_graph_adaptive", "Entity graph"),
        ("learned_topk_paths", "\\proofkv{} paths"),
        ("learned_and_or", "\\proofkv{} AND/OR"),
    ):
        row = row_for(data["longbench_V2"]["records"], method=method)
        body.append(
            f"{label} & {pct(row['accuracy'])} & {pct(row['mean_retained_ratio'])} & "
            f"{row['mean_selection_seconds']:.2f} & "
            f"{row['mean_generation_seconds']:.2f} \\\\"
        )
    write(
        "longbench.tex",
        [
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Method & Accuracy & KV\\% & Selection s & Generation s \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def vllm_backend(data: dict) -> None:
    stats = data["vllm_paged_backend"]
    body: list[str] = []
    for scenario, output in (("throughput_out256", 256), ("throughput_out1024", 1024)):
        item = stats["throughput_statistics"][scenario]
        speed = item["paired_full_wall_over_route_wall"]
        body.append(
            f"{output} & {speed['n']} & {speed['mean']:.3f}$\\times$ & "
            f"[{speed['ci95'][0]:.3f}, {speed['ci95'][1]:.3f}] & "
            f"{item['full_over_route_mean_active_paged_bytes']:.2f}$\\times$ \\\\"
        )
    write(
        "vllm_backend.tex",
        [
            "\\begin{tabular}{@{}rrrrr@{}}",
            "\\toprule",
            "Output & Batches & Throughput speedup & 95\\% CI & KV-byte reduction \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def route_frontier(data: dict) -> None:
    rows = [
        row
        for row in data["physical_route_frontier"]["comparisons"]
        if row["primary"]
    ]
    body: list[str] = []
    for row in sorted(rows, key=lambda item: (item["output_tokens"], item["context_length_requested"])):
        body.append(
            f"{row['context_length_requested'] // 1000}k & "
            f"{row['output_tokens']} & {row['batch_size']} & "
            f"{row['prefill_speedup_vs_gather']:.2f}$\\times$ & "
            f"{row['total_speedup_vs_gather']:.3f}$\\times$ & "
            f"{row['decode_throughput_ratio_vs_gather']:.3f} & "
            f"{row['peak_memory_ratio_vs_gather']:.3f} \\\\"
        )
    write(
        "route_frontier.tex",
        [
            "\\begin{tabular}{@{}rrrrrrr@{}}",
            "\\toprule",
            "Context & Output & Batch & Prefill & Total & Decode ratio & Peak ratio \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def pvalue(value: float) -> str:
    if value < 0.001:
        return f"{value:.1e}".replace("e-0", "e-")
    return f"{value:.3f}"


def statistical_ledger(data: dict) -> None:
    rows: list[str] = []

    locked = data["locked_musique_selector"]["statistics"]
    for budget in ("256", "512"):
        stat = next(
            row
            for row in locked
            if f"v6_{budget}_topk_vs_reranker" in row["source"]
        )
        rows.append(
            f"Locked MuSiQue PCR@{budget}: paths--reranker & {stat['n_paired']} & "
            f"{100 * stat['delta']:+.2f} & "
            f"[{100 * stat['ci95'][0]:+.2f}, {100 * stat['ci95'][1]:+.2f}] & "
            f"{pvalue(stat['p_two_sided'])} \\\\"
        )

    fixed = data["hotpot_fixed_budget"]["statistics"]
    specs = [
        ("fixed128_learned_topk", "Hotpot F1@128: paths--CompressKV"),
        ("fixed256_learned_and_or", "Hotpot F1@256: AND/OR+aux--CompressKV"),
        ("fixed512_learned_topk", "Hotpot F1@512: paths--CompressKV"),
    ]
    for marker, label in specs:
        stat = next(row for row in fixed if marker in row["source"] and "compresskv" in row["source"])
        rows.append(
            f"{label} & {stat['n_paired']} & {100 * stat['delta']:+.2f} & "
            f"[{100 * stat['ci95'][0]:+.2f}, {100 * stat['ci95'][1]:+.2f}] & "
            f"{pvalue(stat['p_two_sided'])} \\\\"
        )

    seed = next(
        row
        for row in data["packed_hotpot_seed_replication"]["comparisons"]
        if row["method_b"] == "full"
    )
    rows.append(
        f"Packed 8k, three seeds: paths--Full & {seed['n_question_clusters']} & "
        f"{100 * seed['mean_difference_a_minus_b']:+.2f} & "
        f"[{100 * seed['cluster_bootstrap_ci95'][0]:+.2f}, "
        f"{100 * seed['cluster_bootstrap_ci95'][1]:+.2f}] & "
        f"{pvalue(seed['cluster_sign_flip_p_two_sided'])} \\\\"
    )

    topology = data["targeted_independent_audit"]["group2"]["comparisons"]
    for key, label in (
        ("proof_core_vs_edge_shuffled", "Natural topology: core--shuffled"),
        ("proof_core_vs_random_edges", "Natural topology: core--random"),
        ("proof_core_vs_broken_middle", "Natural topology: core--broken"),
    ):
        stat = topology[key]
        rows.append(
            f"{label} & 128 & {100 * stat['mean']:+.2f} & "
            f"[{100 * stat['ci95_low']:+.2f}, {100 * stat['ci95_high']:+.2f}] & "
            f"{pvalue(stat['paired_sign_flip_p_one_sided'])} \\\\"
        )
    write(
        "statistical_ledger.tex",
        [
            "\\begin{tabular}{@{}p{0.50\\textwidth}rrrr@{}}",
            "\\toprule",
            "Paired comparison & $n$ & $\\Delta$ points & 95\\% CI & $p$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def crossmodel_statistics(data: dict) -> None:
    stats = data["cross_model_hotpot"]["statistics"]
    rows: list[str] = []
    for model_marker, label in (("mistral", "Mistral-v0.3"), ("qwen3", "Qwen3-8B")):
        for budget in (128, 256, 512):
            stat = next(
                row
                for row in stats
                if f"{model_marker}_fixed{budget}_learned_topk" in row["source"]
            )
            rows.append(
                f"{label} & {budget} & {100 * stat['delta']:+.2f} & "
                f"[{100 * stat['ci95'][0]:+.2f}, {100 * stat['ci95'][1]:+.2f}] & "
                f"{pvalue(stat['p_two_sided'])} \\\\"
            )
    write(
        "crossmodel_statistics.tex",
        [
            "\\begin{tabular}{@{}lrrrr@{}}",
            "\\toprule",
            "Decoder & Tokens & Paths--entity F1 & 95\\% CI & $p$ \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def training_negatives(data: dict) -> None:
    runs = data["training_and_components"]["training_runs"]
    rows: list[str] = []
    for stage in ("v3b mixed replay", "v8 Hotpot boost"):
        row = next(item for item in runs if item["stage"] == stage)
        counts = row["negative_counts"]
        rows.append(
            f"{stage} & {counts['edge:positive']:,} & "
            f"{counts['edge:reversed_gold_edge']:,} & "
            f"{counts['edge:cross_proof_shortcut']:,} & "
            f"{counts['edge:missing_or_wrong_middle']:,} & "
            f"{counts['edge:correct_endpoint_wrong_source']:,} \\\\"
        )
    write(
        "training_negatives.tex",
        [
            "\\begin{tabular}{@{}lrrrrr@{}}",
            "\\toprule",
            "Stage & Positive & Reversed & Cross-proof & Wrong middle & Wrong source \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def policy_hyperparameters() -> None:
    write(
        "policy_hyperparameters.tex",
        [
            "\\begin{tabular}{@{}p{0.30\\columnwidth}p{0.62\\columnwidth}@{}}",
            "\\toprule",
            "Component & Frozen value \\\\",
            "\\midrule",
            "Span candidates & 32 query-conditioned candidates \\\\",
            "Dense edge core & 4 highest-confidence spans \\\\",
            "Root frontier & 16 roots \\\\",
            "Edge batch & 256 pairs \\\\",
            "Path search & beam 8; at most 8 hops \\\\",
            "Proof set & at most 8 proofs \\\\",
            "Subset compiler & beam 128 over proof unions \\\\",
            "Length prior & $\\exp(-0.04)$ per hop \\\\",
            "Boundary context & 8 tokens per retained side \\\\",
            "Pinned positions & 4 sink tokens and recent tail \\\\",
            "Page size & 16 KV tokens \\\\",
            "Precision & BF16 proof core; optional INT8 auxiliary \\\\",
            "Decoding & greedy; deterministic seed 17 \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def evaluation_inventory() -> None:
    write(
        "evaluation_inventory.tex",
        [
            "\\begin{tabular}{@{}p{0.28\\textwidth}p{0.14\\textwidth}p{0.12\\textwidth}p{0.34\\textwidth}@{}}",
            "\\toprule",
            "Evaluation block & Population & Conditions & Scientific role \\\\",
            "\\midrule",
            "Locked MuSiQue selector & 500 questions & 2 exact budgets & Primary PCR claim \\\\",
            "HotpotQA fixed/ratio & 128 questions & 6 budgets, 4 decoders & Downstream and decoder transfer \\\\",
            "2Wiki development & 100 questions & 6 budgets & Explicit-path development transfer \\\\",
            "MuSiQue generation & 128 questions & 6 budgets & Hard downstream transfer \\\\",
            "Packed Hotpot scale & 128 questions & 8k/16k/32k & Context scaling and proof size \\\\",
            "Packing replication & 128 clusters & 3 seeds & Question-clustered robustness \\\\",
            "Controlled topology & 512 examples & 7 hop strata & Structural mechanism \\\\",
            "Natural topology & 128 questions & 6 equal-byte routes & Causal downstream mechanism \\\\",
            "Selector-inclusive H100 & 4 regimes & 3 modes, 80 batches & Reuse and end-to-end latency \\\\",
            "Paged vLLM & 736 requests & 43 batch records & Physical backend and active bytes \\\\",
            "Compiler diagnostics & 32--64 examples & halo and precision gates & Fidelity of physical route \\\\",
            "Risk/LongBench diagnostics & 2,048 / 128 & held-out gates & Scope and excluded branches \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    packed_scale(data)
    fixed_budget(data)
    baseline_matrix(data)
    support_coverage(data)
    kvzip_contract(data)
    cross_model(data)
    selector(data)
    topology(data)
    synthetic_topology(data)
    serving(data)
    diagnostics(data)
    downstream_matrix_table(data, "hotpot_qwen25", "matrix_hotpot.tex")
    downstream_matrix_table(data, "twowiki_qwen25", "matrix_2wiki.tex")
    downstream_matrix_table(data, "musique_qwen25", "matrix_musique.tex")
    downstream_matrix_table(data, "hotpot_mistral02", "matrix_mistral02.tex")
    downstream_matrix_table(data, "hotpot_mistral03", "matrix_mistral03.tex")
    downstream_matrix_table(data, "hotpot_qwen3", "matrix_qwen3.tex")
    interpolation(data)
    training_stages(data)
    component_metrics(data)
    hop_topology(data)
    dense_core(data)
    kvpress_matrix(data)
    kvzip_full(data)
    headkv_matrix(data)
    packing_seeds(data)
    packed_protocol(data)
    context_halos(data)
    multiquery(data)
    precision_table(data)
    failure_gates(data)
    longbench(data)
    vllm_backend(data)
    route_frontier(data)
    statistical_ledger(data)
    crossmodel_statistics(data)
    training_negatives(data)
    policy_hyperparameters()
    evaluation_inventory()


if __name__ == "__main__":
    main()
