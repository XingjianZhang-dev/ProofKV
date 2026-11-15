#!/usr/bin/env python
"""Driver for topically related distractor packing.  Injects a candidate ORDER into the frozen packer and runs the
frozen matrix script's main() unchanged (same 128 questions, targets, 8% ratio budget, decoder, selector policy,
packing seed, answer-leak / duplicate-title rules, position shuffle, exact-target trimming).

Only the distractor sampling changes: the frozen packer walks the pool in a seeded random order; here it walks the pool
in a per-question ranked order:
  HARD_TFIDF   pool paragraphs ranked by unigram+bigram TF-IDF cosine similarity to the question (HotpotQA's own
               distractor construction, applied to the pooled validation paragraphs); every candidate has a score, so the
               walk never falls back to random order.
  SAME_ENTITY  pool paragraphs ranked by the number of gold-document title entities (capitalised title tokens, no
               stopwords) they share, title matches weighted 2, text mentions 1; ties (score 0 = unrelated) keep the
               frozen seeded random order, so the walk falls back to the unrelated protocol once related paragraphs
               are exhausted.  The number of related vs fallback pages admitted is recorded per example.
Packed contexts are dumped to packed_contexts.jsonl for the independent audit (audit_packing.py).
"""
from __future__ import annotations
import copy, json, math, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
IMPL = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(IMPL / "scripts"))
import run_hotpot_compact_qa as rq

POOL_KIND = None; DUMP = None
STOP = set("a an and are as at be by for from had has have he her his how in is it its of on or she that the their them they this to was were what when where which who why with".split())


def toks(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


def grams(text):
    t = toks(text); return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]


class TfidfIndex:
    def __init__(self, pool):
        self.df = Counter(); self.doc_tf = []; self.norm = []
        for entry in pool:
            g = Counter(grams(entry["title"] + " " + " ".join(entry["sentences"]))); self.doc_tf.append(g); self.df.update(g.keys())
        self.n = len(pool); self.idf = {t: math.log((self.n + 1) / (df + 1)) + 1 for t, df in self.df.items()}
        self.inv = defaultdict(list)
        for i, g in enumerate(self.doc_tf):
            w = {t: (1 + math.log(c)) * self.idf[t] for t, c in g.items()}; nrm = math.sqrt(sum(v * v for v in w.values())) or 1.0; self.norm.append(nrm)
            for t, v in w.items(): self.inv[t].append((i, v / nrm))

    def scores(self, question):
        q = Counter(grams(question)); qw = {t: (1 + math.log(c)) * self.idf.get(t, math.log(self.n + 1) + 1) for t, c in q.items()}
        qn = math.sqrt(sum(v * v for v in qw.values())) or 1.0; s = defaultdict(float)
        for t, v in qw.items():
            for i, dv in self.inv.get(t, ()): s[i] += dv * v / qn
        return s


def entity_tokens(title):
    return {t for t in re.findall(r"[A-Z][A-Za-z0-9\-]+", title) if t.lower() not in STOP and len(t) > 2}


_original_pack = rq.pack_source_to_prompt_target
_original_pool = rq.build_natural_distractor_pool
_index = {}


def build_pool(dataset):
    pool = _original_pool(dataset)
    if POOL_KIND == "HARD_TFIDF":
        _index["tfidf"] = TfidfIndex(pool)
    else:
        _index["title_ents"] = [entity_tokens(e["title"]) for e in pool]
        _index["text_lower"] = [(" " + " ".join(e["sentences"]) + " ").lower() for e in pool]
    return pool


def pack(source, tokenizer, pool, target_prompt_tokens, packing_seed, user_only_chat):
    packed = copy.deepcopy(source)
    original_titles = [str(t) for t in packed["context"]["title"]]
    original_pages = [list(map(str, s)) for s in packed["context"]["sentences"]]
    packed["context"] = {"title": list(original_titles), "sentences": original_pages}
    base_prompt, _, _ = rq.make_prompt(packed, tokenizer, user_only_chat)
    base_tokens = len(tokenizer(base_prompt, add_special_tokens=False)["input_ids"])
    if base_tokens >= target_prompt_tokens:
        return packed, {"target_prompt_tokens": target_prompt_tokens, "base_prompt_tokens": base_tokens, "actual_prompt_tokens": base_tokens, "added_pages": 0,
                        "answer_leak_candidates_skipped": 0, "base_exceeds_target": base_tokens > target_prompt_tokens, "pool_kind": POOL_KIND}
    rng = random.Random(rq._stable_sample_seed(str(source["id"]), packing_seed))
    candidate_order = list(range(len(pool))); rng.shuffle(candidate_order)          # frozen seeded order = tie-break / fallback
    # ---- the only change: rank candidates by relatedness to this question / its gold documents
    if POOL_KIND == "HARD_TFIDF":
        score = _index["tfidf"].scores(str(source["question"]))
    else:
        gold_titles = set(source["supporting_facts"]["title"]); ents = set().union(*(entity_tokens(t) for t in gold_titles)) if gold_titles else set()
        ents_lower = {e.lower() for e in ents}; score = {}
        for i in candidate_order:
            s = 2 * len(_index["title_ents"][i] & ents) + sum(1 for e in ents_lower if f" {e} " in _index["text_lower"][i] or f" {e}," in _index["text_lower"][i] or f" {e}." in _index["text_lower"][i])
            if s: score[i] = s
    rank = {i: r for r, i in enumerate(candidate_order)}
    candidate_order = sorted(candidate_order, key=lambda i: (-score.get(i, 0.0), rank[i]))
    used_titles = set(original_titles); answer = rq.normalize(str(source.get("answer", "")))
    answer_pattern = re.compile(rf"(?:^|\s){re.escape(answer)}(?:\s|$)") if len(answer) >= 2 else None
    skipped = 0; estimated = base_tokens; added_titles = []; added_scores = []
    for ci in candidate_order:
        cand = pool[ci]; title = cand["title"]
        if cand["source_id"] == str(source["id"]) or title in used_titles:
            continue
        paragraph = " ".join(cand["sentences"])
        if answer_pattern is not None and answer_pattern.search(rq.normalize(paragraph)):
            skipped += 1; continue
        fragment = "".join(f"[{title}] {s}\n" for s in cand["sentences"])
        ft = len(tokenizer(fragment, add_special_tokens=False)["input_ids"])
        if estimated + ft > target_prompt_tokens - 16:
            continue
        packed["context"]["title"].append(title); packed["context"]["sentences"].append(list(cand["sentences"]))
        added_titles.append(title); added_scores.append(float(score.get(ci, 0.0))); used_titles.add(title); estimated += ft
        if estimated >= target_prompt_tokens - 64:
            break
    pages = list(zip(packed["context"]["title"], packed["context"]["sentences"])); rng.shuffle(pages)
    packed["context"]["title"] = [t for t, _ in pages]; packed["context"]["sentences"] = [s for _, s in pages]
    prompt, _, _ = rq.make_prompt(packed, tokenizer, user_only_chat); actual = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    while actual > target_prompt_tokens and added_titles:
        rt = added_titles.pop(); added_scores.pop(); ri = packed["context"]["title"].index(rt)
        packed["context"]["title"].pop(ri); packed["context"]["sentences"].pop(ri)
        prompt, _, _ = rq.make_prompt(packed, tokenizer, user_only_chat); actual = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    meta = {"target_prompt_tokens": target_prompt_tokens, "base_prompt_tokens": base_tokens, "actual_prompt_tokens": actual, "added_pages": len(added_titles),
            "answer_leak_candidates_skipped": skipped, "base_exceeds_target": False, "pool_kind": POOL_KIND,
            "related_pages_admitted": sum(s > 0 for s in added_scores), "fallback_unrelated_pages": sum(s <= 0 for s in added_scores),
            "mean_relatedness_score": (sum(added_scores) / len(added_scores)) if added_scores else 0.0}
    if DUMP is not None:
        with open(DUMP, "a") as f:
            f.write(json.dumps({"id": str(source["id"]), "question": str(source["question"]), "answer": str(source.get("answer", "")), "target_prompt_tokens": target_prompt_tokens,
                                "pool_kind": POOL_KIND, "original_titles": original_titles, "supporting_titles": list(source["supporting_facts"]["title"]),
                                "context": packed["context"], "packing": meta}) + "\n")
    return packed, meta


if __name__ == "__main__":
    POOL_KIND = sys.argv[1]; DUMP = sys.argv[2]; sys.argv = [sys.argv[0]] + sys.argv[3:]
    assert POOL_KIND in ("HARD_TFIDF", "SAME_ENTITY")
    rq.build_natural_distractor_pool = build_pool; rq.pack_source_to_prompt_target = pack
    rq.main()
