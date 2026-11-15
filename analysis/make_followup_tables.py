#!/usr/bin/env python3
"""Write the LaTeX tables of the follow-up analyses from results/followup/summary.json.

Usage: python analysis/make_followup_tables.py results/followup/summary.json paper/tables results/paper/canonical_results.json
"""
import json
import sys
from pathlib import Path

summary = json.load(open(sys.argv[1]))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
canonical = json.load(open(sys.argv[3]))


def ci(s, digits=1):
    return f"${s['delta']:+.{digits}f}$ & $[{s['lo']:.{digits}f},{s['hi']:.{digits}f}]$"


def write(name, lines):
    (out / name).write_text("\n".join(lines) + "\n")


# ---- premise test
prem = summary["premise_test"]
rows = [("SnapKV", "snapkv"), ("PyramidKV", "pyramidkv"), ("H$_2$O-style", "observed_attention_h2o_style"), ("StreamingLLM", "streamingllm")]
lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"512 tokens & Spans touched & Gold tokens & PCR & F1 \\", r"\midrule"]
for label, key in rows:
    c = prem["compressors"][key]
    lines.append(f"{label} & {c['all_gold_spans_touched']:.1f} & {c['gold_token_recall']:.1f} & {c['chain_complete']:.1f} & {c['f1']:.1f} \\\\")
c = prem["chain"]["proof_core"]
lines += [r"\midrule", f"\\proofkv{{}} chains & -- & {c['gold_token_recall']:.1f} & \\textbf{{{c['chain_complete']:.1f}}} & \\textbf{{{c['f1']:.1f}}} \\\\",
          r"\bottomrule", r"\end{tabular}"]
write("premise_test.tex", lines)

# ---- link interventions across datasets
li = summary["link_interventions"]
order = [("musique_slice", "MuSiQue, original slice", "paragraph"), ("musique_pooled", "MuSiQue, all 372", "paragraph"),
         ("hotpot_paragraph", "HotpotQA", "paragraph"), ("hotpot_sentence", "HotpotQA", "sentence"), ("twowiki", "2WikiMultiHopQA", "paragraph")]
lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Dataset & Span unit & $n$ & Chains F1 & One link removed & Loss & 95\% CI \\", r"\midrule"]
for key, label, unit in order:
    b = li[key]
    s = b["loss"]["broken_middle"]
    lines.append(f"{label} & {unit} & {b['n']} & {b['f1']['proof_core']:.1f} & {b['f1']['broken_middle']:.1f} & ${s['delta']:.1f}$ & $[{s['lo']:.1f},{s['hi']:.1f}]$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("link_replication.tex", lines)

lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r"MuSiQue intervention & $n{=}128$ & $n{=}372$ & 95\% CI ($n{=}372$) \\", r"\midrule"]
for arm, label in (("edge_shuffled", "Link targets shuffled"), ("random_edges", "Random links"), ("broken_middle", "Middle span removed")):
    a, b = li["musique_slice"]["loss"][arm], li["musique_pooled"]["loss"][arm]
    lines.append(f"{label} & ${a['delta']:.1f}$ & ${b['delta']:.1f}$ & $[{b['lo']:.1f},{b['hi']:.1f}]$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("link_musique.tex", lines)

bk = li["attention_backends"]
lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Attention backend & Shuffled & Random & Removed \\", r"\midrule"]
for key, label in (("sdpa_flash", "SDPA, flash kernel"), ("sdpa_math", "SDPA, math kernel"), ("eager", "Eager attention")):
    lines.append(f"{label} & {bk[key]['edge_shuffled']:.1f} & {bk[key]['random_edges']:.1f} & {bk[key]['broken_middle']:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("link_backends.tex", lines)

# ---- iterative retrieval: selector and generation
hs = summary["heldout_selector"]
lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{All} & \multicolumn{3}{c}{PCR@512 by depth} \\", r"\cmidrule(lr){2-3}\cmidrule(l){4-6}",
         r"Selector & @256 & @512 & 2 hops & 3 hops & 4 hops \\", r"\midrule"]
for key, label in (("same_scorer_independent", "Same scorer, independent"), ("iterative_hop_order", "Iterative retrieval"), ("chains", "\\proofkv{} chains"), ("oracle", "Oracle")):
    h = hs["by_hop_512"][key]
    lines.append(f"{label} & {hs['pcr'][key]['256']:.1f} & {hs['pcr'][key]['512']:.1f} & {h['2hop']['pcr']:.1f} & {h['3hop']['pcr']:.1f} & {h['4hop']['pcr']:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("iterative_selector.tex", lines)

br = summary["budget_regime"]["budgets"]
lines = [r"\begin{tabular}{@{}rrrrrrr@{}}", r"\toprule",
         r"Tokens & Chain fits & Independent & Iterative & \proofkv{} & Gain over iterative & 95\% CI \\", r"\midrule"]
for b in ("64", "96", "128", "192", "256", "384", "512"):
    c = br[b]
    g = c["gain"]["iterative_hop_order"]
    lines.append(f"{b} & {c['gold_fits_pct']:.0f}\\% & {c['f1']['same_scorer_independent']:.1f} & {c['f1']['iterative_hop_order']:.1f} & {c['f1']['chains']:.1f} & ${g['delta']:+.1f}$ & $[{g['lo']:.1f},{g['hi']:.1f}]$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("iterative_regime.tex", lines)

# ---- oracle gap
gap = hs["gap_512"]
labels = [("iii_a_edge_mis_scored", "Gold link under-scored by the scorer"), ("iii_d_search_root_or_terminal_gate", "Start or end span excluded by the search gate"),
          ("iii_e_found_but_not_selected", "Complete chain found but outranked"), ("ii_candidate_relation_miss", "Gold link absent from the candidate graph"),
          ("iii_d_search_beam_pruned", "Pruned by the beam"), ("iii_b_wrong_direction", "Link predicted in the wrong direction")]
lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule", r"Held-out MuSiQue test, 512 tokens ($n{=}500$) & Questions (\%) \\", r"\midrule",
         f"Complete gold chain retained & {gap['complete']:.1f} \\\\", r"\addlinespace[2pt]"]
total = 0.0
for key, label in labels:
    lines.append(f"{label} & {gap.get(key, 0.0):.1f} \\\\")
    total += gap.get(key, 0.0)
lines += ["Wrong relation label & 0.0 \\\\", r"\addlinespace[2pt]", f"Gold chain does not fit the budget & {gap['iv_budget_infeasible']:.1f} \\\\", r"\bottomrule", r"\end{tabular}"]
write("oracle_gap.tex", lines)
assert abs(total - 33.4) < 1e-6 and abs(gap["complete"] + total + gap["iv_budget_infeasible"] - 100.0) < 1e-6

# ---- packed contexts: unrelated (original runs) and related distractors (follow-up runs)
rd = summary["related_distractors"]
packed = canonical["packed_hotpot_scale"]["rows"]


def packed_row(target, method):
    hits = [r for r in packed if r["target_prompt_tokens"] == target and r["method"] == method]
    assert len(hits) == 1
    return hits[0]


lines = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r"Added documents & Length & Full & Entity graph & \proofkv{} & KV\% & Oracle & $\Delta$ Full & 95\% CI \\", r"\midrule"]
for k, target in enumerate((8000, 16000, 32000)):
    full, ent, ours, orc = (100 * packed_row(target, m)["answer_f1"] for m in ("full", "entity_graph_adaptive", "learned_topk_paths", "oracle_adaptive"))
    kv = 100 * packed_row(target, "learned_topk_paths")["retained_ratio"]
    lines.append(f"{'Unrelated' if k == 0 else ''} & {target // 1000}k & {full:.1f} & {ent:.1f} & \\textbf{{{ours:.1f}}} & {kv:.1f} & {orc:.1f} & ${ours - full:+.1f}$ & -- \\\\")
for pool, label in (("tfidf_hard", "Highest TF-IDF similarity"), ("same_entity", "Same entities")):
    lines.append(r"\addlinespace[2pt]")
    for k, target in enumerate(("8000", "16000", "32000")):
        c = rd[pool][target]
        d = c["chains_minus_full"]
        lines.append(f"{label if k == 0 else ''} & {int(target) // 1000}k & {c['full_f1']:.1f} & {c['entity_f1']:.1f} & \\textbf{{{c['chains_f1']:.1f}}} & {c['chains_kv_pct']:.1f} & {c['oracle_f1']:.1f} & ${d['delta']:+.1f}$ & $[{d['lo']:.1f},{d['hi']:.1f}]$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("packed_all.tex", lines)
print("wrote follow-up tables to", out)
