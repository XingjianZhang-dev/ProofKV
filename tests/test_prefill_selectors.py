from proofkv.prefill_selectors import (
    _node_ids,
    _plan_from_spans,
    _var_assignment_chain_spans,
    block_halo_guideposts,
    compass_prefill_guideposts,
)


def _word_offsets(text: str):
    offsets = []
    cursor = 0
    for part in text.split():
        start = text.index(part, cursor)
        end = start + len(part)
        offsets.append((start, end))
        cursor = end
    return offsets


def _line_token_span(text: str, offsets, prefix: str):
    start = text.index(prefix)
    end = text.find("\n", start)
    if end < 0:
        end = len(text)
    hits = [i for i, (s, e) in enumerate(offsets) if e > start and s < end]
    return min(hits), max(hits) + 1


def _covered(plan, span):
    return set(range(span[0], span[1])).issubset(set(plan.retained_indices))


def _chain_prompt(hops: int = 5, filler_words: int = 12):
    nodes = [f"NODE_ALPHA_{i}" for i in range(hops)]
    facts = []
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            facts.append(f"Fact: {node} points to {nodes[i + 1]}.")
        else:
            facts.append(f"Fact: {node} resolves to number 777.")
    filler = " ".join(f"filler{i}" for i in range(filler_words))
    body = []
    for fact in facts:
        body.append(filler)
        body.append(fact)
    text = (
        "Follow the pointer chain through the notes.\n"
        + "\n".join(body)
        + f"\nQuestion: Starting from {nodes[0]}, what number is reached? Answer with only the number.\nAnswer:"
    )
    return text, facts


def _var_tracking_prompt(filler_words: int = 16):
    filler = " ".join(f"vtfill{i}" for i in range(filler_words))
    target = [
        "VAR ALPHA = 12345",
        "VAR BRAVO = VAR ALPHA",
        "VAR CHARLIE = VAR BRAVO",
        "VAR DELTA = VAR CHARLIE",
        "VAR ECHO = VAR DELTA",
    ]
    distractors = [
        "VAR OTHER = 99999",
        "VAR SPARE = VAR OTHER",
    ]
    body = []
    for line in [distractors[0], target[0], target[1], distractors[1], target[2], target[3], target[4]]:
        body.append(filler)
        body.append(line)
    text = (
        "Memorize and track the chain(s) of variable assignment hidden in the following text.\n"
        + "\n".join(body)
        + "\nQuestion: Find all variables that are assigned the value 12345 in the text above."
        + " Answer: According to the chain(s) of variable assignment in the text above, 5 variables are assigned the value 12345, they are:"
    )
    return text, target, distractors


def test_block_halo_respects_guidepost_budget():
    plan = block_halo_guideposts(200, guidepost_budget=24, sink_size=4, recent_size=16, mode="uniform")
    assert len(plan.guidepost_indices) <= 24
    assert all(4 <= i < 184 for i in plan.guidepost_indices)
    assert plan.retained_ratio <= (4 + 16 + 24) / 200


def test_zero_guidepost_budget_keeps_only_mandatory_stream_tokens():
    plan = _plan_from_spans(
        seq_len=100,
        spans=[(20, 40)],
        guidepost_budget=0,
        sink_size=4,
        recent_size=16,
    )
    assert plan.guidepost_indices == []
    assert plan.retained_indices == list(range(4)) + list(range(84, 100))


def test_compass_prefill_selects_query_key_span():
    text = (
        "neutral filler words before the useful line "
        "K12345 stores value 998877 in the ledger "
        "more neutral filler words "
        "Question: What value is stored for K12345?"
    )
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="What value is stored for K12345?",
        token_offsets=offsets,
        guidepost_budget=8,
        sink_size=1,
        recent_size=4,
        block_size=4,
        halo=1,
    )
    key_token = next(i for i, (s, e) in enumerate(offsets) if text[s:e] == "K12345")
    assert key_token in plan.guidepost_indices


def test_node_identifiers_are_extracted_intact():
    text = "Fact: NODE_AB12C_0 points to NODE_ZYX99_1."
    assert _node_ids(text) == ["NODE_AB12C_0", "NODE_ZYX99_1"]


def test_compass_prefill_node_chain_selects_two_hop_facts():
    text, facts = _chain_prompt(hops=2)
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="Question: Starting from NODE_ALPHA_0, what number is reached?",
        token_offsets=offsets,
        guidepost_budget=32,
        sink_size=1,
        recent_size=6,
        block_size=4,
        halo=1,
    )
    assert all(_covered(plan, _line_token_span(text, offsets, fact)) for fact in facts)


def test_compass_prefill_node_chain_selects_five_hop_facts_and_resolver():
    text, facts = _chain_prompt(hops=5)
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="Question: Starting from NODE_ALPHA_0, what number is reached?",
        token_offsets=offsets,
        guidepost_budget=80,
        sink_size=1,
        recent_size=6,
        block_size=4,
        halo=1,
    )
    spans = [_line_token_span(text, offsets, fact) for fact in facts]
    assert all(_covered(plan, span) for span in spans)
    assert _covered(plan, spans[-1])


def test_compass_prefill_node_chain_spans_preserve_fact_order():
    text, facts = _chain_prompt(hops=5)
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="Question: Starting from NODE_ALPHA_0, what number is reached?",
        token_offsets=offsets,
        guidepost_budget=80,
        sink_size=1,
        recent_size=6,
        block_size=4,
        halo=1,
    )
    starts = []
    for fact in facts:
        fact_span = _line_token_span(text, offsets, fact)
        span_idx = next(i for i, span in enumerate(plan.spans) if span[0] <= fact_span[0] and span[1] >= fact_span[1])
        starts.append(span_idx)
    assert starts == sorted(starts)


def test_compass_prefill_node_chain_keeps_retained_ratio_near_budget():
    text, _ = _chain_prompt(hops=5, filler_words=180)
    offsets = _word_offsets(text)
    budget = 80
    sink = 4
    recent = 32
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="Question: Starting from NODE_ALPHA_0, what number is reached?",
        token_offsets=offsets,
        guidepost_budget=budget,
        sink_size=sink,
        recent_size=recent,
        block_size=8,
        halo=1,
    )
    assert len(plan.guidepost_indices) <= budget
    assert plan.retained_ratio <= (sink + recent + budget) / len(offsets)


def test_compass_prefill_exact_retrieval_selects_key_after_node_chain_update():
    text = (
        "neutral filler words before the useful line "
        "K999 stores value 123456 in the ledger "
        "more neutral filler words "
        "Question: What value is stored for K999?"
    )
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="What value is stored for K999?",
        token_offsets=offsets,
        guidepost_budget=8,
        sink_size=1,
        recent_size=4,
        block_size=4,
        halo=1,
    )
    key_token = next(i for i, (s, e) in enumerate(offsets) if text[s:e] == "K999")
    assert key_token in plan.guidepost_indices


def test_var_assignment_chain_spans_extract_official_style_lines():
    text, target, _ = _var_tracking_prompt()
    offsets = _word_offsets(text)
    spans = _var_assignment_chain_spans(
        text,
        offsets,
        "Question: Find all variables that are assigned the value 12345 in the text above.",
        start=1,
        end=len(offsets) - 8,
        halo=0,
        max_hops=5,
    )
    target_spans = [_line_token_span(text, offsets, line) for line in target]
    assert len(spans) == len(target_spans)
    assert all(span in spans for span in target_spans)


def test_compass_prefill_var_tracking_selects_query_relevant_chain():
    text, target, _ = _var_tracking_prompt()
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="Question: Find all variables that are assigned the value 12345 in the text above.",
        token_offsets=offsets,
        guidepost_budget=40,
        sink_size=1,
        recent_size=8,
        block_size=8,
        halo=1,
    )
    assert all(_covered(plan, _line_token_span(text, offsets, line)) for line in target)


def test_compass_prefill_var_tracking_does_not_select_all_distractors():
    text, _, distractors = _var_tracking_prompt()
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="Question: Find all variables that are assigned the value 12345 in the text above.",
        token_offsets=offsets,
        guidepost_budget=40,
        sink_size=1,
        recent_size=8,
        block_size=8,
        halo=1,
    )
    distractor_spans = [_line_token_span(text, offsets, line) for line in distractors]
    assert not all(_covered(plan, span) for span in distractor_spans)


def test_compass_prefill_var_tracking_span_order_is_stable():
    text, target, _ = _var_tracking_prompt()
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="Question: Find all variables that are assigned the value 12345 in the text above.",
        token_offsets=offsets,
        guidepost_budget=40,
        sink_size=1,
        recent_size=8,
        block_size=8,
        halo=1,
    )
    span_indices = []
    for line in target:
        fact_span = _line_token_span(text, offsets, line)
        span_indices.append(next(i for i, span in enumerate(plan.spans) if span[0] <= fact_span[0] and span[1] >= fact_span[1]))
    assert span_indices == sorted(span_indices)


def test_compass_prefill_var_tracking_respects_budget_cap():
    text, _, _ = _var_tracking_prompt(filler_words=80)
    offsets = _word_offsets(text)
    budget = 40
    sink = 4
    recent = 16
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="Question: Find all variables that are assigned the value 12345 in the text above.",
        token_offsets=offsets,
        guidepost_budget=budget,
        sink_size=sink,
        recent_size=recent,
        block_size=8,
        halo=1,
    )
    assert len(plan.guidepost_indices) <= budget
    assert plan.retained_ratio <= (sink + recent + budget) / len(offsets)


def _natural_qa_prompt():
    sentences = [
        "Alice went to the kitchen.",
        "Sandra went to the garden.",
        "Mary went to the hallway.",
        "John went to the office.",
        "Mary went to the bathroom.",
        "Daniel went to the bedroom.",
        "Sandra went to the kitchen.",
        "John went to the garden.",
    ]
    text = (
        "Answer the question using the story.\n"
        + "\n".join(sentences)
        + "\n\nQuestion: Where is Mary? \nAnswer:"
    )
    return text, sentences


def test_compass_prefill_natural_qa_selects_queried_entity_sentence():
    text, sentences = _natural_qa_prompt()
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="\n\nQuestion: Where is Mary? \nAnswer:",
        token_offsets=offsets,
        guidepost_budget=22,
        sink_size=1,
        recent_size=4,
        block_size=4,
        halo=1,
    )
    assert any(_covered(plan, _line_token_span(text, offsets, sentence)) for sentence in sentences if "Mary" in sentence)


def test_compass_prefill_natural_qa_keeps_last_entity_mention():
    text, _ = _natural_qa_prompt()
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="\n\nQuestion: Where is Mary? \nAnswer:",
        token_offsets=offsets,
        guidepost_budget=22,
        sink_size=1,
        recent_size=4,
        block_size=4,
        halo=1,
    )
    assert _covered(plan, _line_token_span(text, offsets, "Mary went to the bathroom."))


def test_compass_prefill_natural_qa_does_not_keep_all_distractors_and_respects_budget():
    text, sentences = _natural_qa_prompt()
    offsets = _word_offsets(text)
    budget = 22
    sink = 1
    recent = 4
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="\n\nQuestion: Where is Mary? \nAnswer:",
        token_offsets=offsets,
        guidepost_budget=budget,
        sink_size=sink,
        recent_size=recent,
        block_size=4,
        halo=1,
    )
    distractors = [sentence for sentence in sentences if "Mary" not in sentence]
    assert not all(_covered(plan, _line_token_span(text, offsets, sentence)) for sentence in distractors)
    assert len(plan.guidepost_indices) <= budget
    assert plan.retained_ratio <= (sink + recent + budget) / len(offsets)


def test_compass_prefill_natural_qa_span_order_is_stable():
    text, sentences = _natural_qa_prompt()
    offsets = _word_offsets(text)
    plan = compass_prefill_guideposts(
        prompt=text,
        query_text="\n\nQuestion: Where is Mary? \nAnswer:",
        token_offsets=offsets,
        guidepost_budget=22,
        sink_size=1,
        recent_size=4,
        block_size=4,
        halo=1,
    )
    starts = []
    for sentence in sentences:
        if "Mary" not in sentence:
            continue
        sent_span = _line_token_span(text, offsets, sentence)
        if any(span[0] <= sent_span[0] and span[1] >= sent_span[1] for span in plan.spans):
            starts.append(sent_span[0])
    assert starts == sorted(starts)
