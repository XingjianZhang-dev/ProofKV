from __future__ import annotations

import torch

from run_llmlingua2_longbench_20260726 import (
    compress_to_decoder_budget,
    fit_compressed_context,
)


class _FakeTokenizer:
    chat_template = None
    bos_token = ""

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        return_tensors=None,
    ):
        tokens = text.split()
        ids = list(range(len(tokens)))
        output = {"input_ids": ids}
        if return_offsets_mapping:
            offsets = []
            cursor = 0
            for token in tokens:
                start = text.find(token, cursor)
                offsets.append((start, start + len(token)))
                cursor = start + len(token)
            output["offset_mapping"] = offsets
        if return_tensors == "pt":
            output = {
                key: torch.tensor([value])
                for key, value in output.items()
            }
        return output

    def decode(
        self,
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join(f"t{index}" for index in ids)


def test_fitted_prompt_never_exceeds_total_capacity() -> None:
    tokenizer = _FakeTokenizer()
    _, input_ids, context_tokens, question_tokens, total = (
        fit_compressed_context(
            tokenizer,
            "a b c d e f g h",
            "q one",
            "answer",
            6,
        )
    )
    assert total == int(input_ids.shape[1])
    assert total <= 6
    assert context_tokens + question_tokens == total


class _FakeCompressor:
    def compress_prompt(self, context, target_token, **kwargs):
        kept = 0 if target_token < 8 else target_token
        text = " ".join(f"w{index}" for index in range(kept))
        return {
            "compressed_prompt": text,
            "origin_tokens": 100,
            "compressed_tokens": kept,
        }


def test_budget_normalization_retries_underfilled_output() -> None:
    result = compress_to_decoder_budget(
        _FakeCompressor(),
        _FakeTokenizer(),
        "unused",
        "q",
        "a",
        capacity=10,
        initial_target=2,
    )
    _, _, _, _, _, actual_tokens, attempts = result
    assert actual_tokens == 10
    assert [row["compressor_target_tokens"] for row in attempts] == [
        2,
        4,
        8,
    ]
