from __future__ import annotations

from analyze_llmlingua2_contract_20260726 import (
    CAPACITIES,
    LLMLINGUA_METHOD,
    MODELS,
)
from analyze_fair_contract_external_20260726 import ROOT_METHOD


def test_formal_contract_constants() -> None:
    assert CAPACITIES == (128, 256, 512)
    assert MODELS == ("qwen25", "mistral02")
    assert LLMLINGUA_METHOD == "llmlingua2_official"
    assert ROOT_METHOD == "learned_root_topk"
