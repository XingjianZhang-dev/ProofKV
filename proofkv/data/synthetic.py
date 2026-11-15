from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence


@dataclass(frozen=True)
class SyntheticSample:
    prompt: str
    answer: str
    task_type: str
    metadata: Dict[str, object]


def _rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _word(r: random.Random, n: int = 6) -> str:
    alphabet = string.ascii_lowercase
    return "".join(r.choice(alphabet) for _ in range(n))


def _filler(r: random.Random, tokens: int) -> str:
    # Plain words tokenize reasonably across LLaMA/Mistral/Qwen tokenizers and
    # are deterministic under seed.
    return " ".join(_word(r, r.randint(4, 8)) for _ in range(max(0, tokens)))


def exact_key_retrieval(seed: int, filler_tokens: int = 2048, depth: float = 0.5) -> SyntheticSample:
    r = _rng(seed)
    key_id = r.randint(100, 999)
    value = str(r.randint(10000, 99999))
    before = int(filler_tokens * max(0.0, min(1.0, depth)))
    after = filler_tokens - before
    prompt = (
        "You are reading a long audit log. Remember exact access codes.\n"
        + _filler(r, before)
        + f"\nRecord: access key K-{key_id} has code {value}.\n"
        + _filler(r, after)
        + f"\nQuestion: What is the code for access key K-{key_id}? Answer with only the code.\nAnswer:"
    )
    return SyntheticSample(prompt, value, "exact_retrieval", {"key": key_id, "depth": depth, "filler_tokens": filler_tokens})


def multi_hop_tracing(seed: int, filler_tokens: int = 2048, hops: int = 3) -> SyntheticSample:
    r = _rng(seed)
    hops = max(2, int(hops))
    entities = [f"NODE_{_word(r, 5).upper()}_{i}" for i in range(hops + 1)]
    final_value = str(r.randint(100, 999))
    facts = []
    for i in range(hops):
        target = entities[i + 1] if i + 1 < hops else final_value
        if i + 1 < hops:
            facts.append(f"Fact: {entities[i]} points to {target}.")
        else:
            facts.append(f"Fact: {entities[i]} resolves to number {target}.")
    # Interleave facts through the context to require global navigation.
    chunks = [_filler(r, filler_tokens // (hops + 1)) for _ in range(hops + 1)]
    body = []
    for i, fact in enumerate(facts):
        body.append(chunks[i])
        body.append(fact)
    body.append(chunks[-1])
    prompt = (
        "Follow the pointer chain through the notes.\n"
        + "\n".join(body)
        + f"\nQuestion: Starting from {entities[0]}, what number is reached? Answer with only the number.\nAnswer:"
    )
    return SyntheticSample(prompt, final_value, "multi_hop", {"hops": hops, "filler_tokens": filler_tokens})


def distributed_aggregation(seed: int, filler_tokens: int = 2048, n_evidence: int = 9) -> SyntheticSample:
    r = _rng(seed)
    attrs = ["amber", "violet", "silver", "cobalt"]
    winning = r.choice(attrs)
    votes = [winning] * (n_evidence // 2 + 1)
    while len(votes) < n_evidence:
        votes.append(r.choice([a for a in attrs if a != winning]))
    r.shuffle(votes)
    chunks = [_filler(r, max(1, filler_tokens // (n_evidence + 1))) for _ in range(n_evidence + 1)]
    body = []
    for i, vote in enumerate(votes):
        body.append(chunks[i])
        body.append(f"Marked evidence block {i}: selected attribute = {vote}.")
    body.append(chunks[-1])
    prompt = (
        "Several marked evidence blocks vote for attributes. Aggregate only marked evidence blocks.\n"
        + "\n".join(body)
        + "\nQuestion: Which selected attribute appears most often? Answer with only the attribute.\nAnswer:"
    )
    return SyntheticSample(prompt, winning, "aggregation", {"n_evidence": n_evidence, "filler_tokens": filler_tokens})


def local_continuation(seed: int, filler_tokens: int = 2048) -> SyntheticSample:
    r = _rng(seed)
    topic = r.choice(["weather", "finance", "robotics", "biology"])
    phrase = r.choice(["therefore", "in summary", "as a result", "the next step"])
    prompt = (
        _filler(r, filler_tokens)
        + f"\nFinal paragraph about {topic}: The argument is now local and stylistic. {phrase}"
    )
    # This task is used for route calibration, not exact scoring.
    return SyntheticSample(prompt, phrase, "local_continuation", {"topic": topic, "filler_tokens": filler_tokens})


def make_calibration_suite(
    n_per_task: int = 8,
    filler_tokens: int = 2048,
    seed: int = 0,
    include_local: bool = True,
) -> List[SyntheticSample]:
    samples: List[SyntheticSample] = []
    for i in range(n_per_task):
        samples.append(exact_key_retrieval(seed + 10_000 + i, filler_tokens, depth=(i + 1) / (n_per_task + 1)))
        samples.append(multi_hop_tracing(seed + 20_000 + i, filler_tokens, hops=2 + (i % 3)))
        samples.append(distributed_aggregation(seed + 30_000 + i, filler_tokens, n_evidence=7 + 2 * (i % 3)))
        if include_local:
            samples.append(local_continuation(seed + 40_000 + i, filler_tokens // 2))
    return samples


def make_ruler_lite_suite(
    n_per_task: int = 16,
    filler_tokens: int = 4096,
    seed: int = 1,
) -> List[SyntheticSample]:
    samples: List[SyntheticSample] = []
    for i in range(n_per_task):
        samples.append(exact_key_retrieval(seed + 1000 + i, filler_tokens, depth=(i % 11) / 10.0))
        samples.append(multi_hop_tracing(seed + 2000 + i, filler_tokens, hops=2 + (i % 4)))
        samples.append(distributed_aggregation(seed + 3000 + i, filler_tokens, n_evidence=9 + 2 * (i % 4)))
    return samples
