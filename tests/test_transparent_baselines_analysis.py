import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_transparent_baselines_20260726.py"
SPEC = importlib.util.spec_from_file_location("transparent_analysis", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_latex_table_bolds_proof_when_best_compact() -> None:
    rows = []
    for model in MODULE.MODELS:
        for capacity in MODULE.CAPACITIES:
            values = {
                method: 10.0 + index
                for index, method in enumerate(MODULE.DISPLAY_METHODS)
            }
            values["full"] = 30.0
            rows.append(
                {
                    "model": model,
                    "capacity": capacity,
                    "f1": values,
                }
            )
    table = MODULE.latex_table(rows)
    assert "Qwen2.5--7B" in table
    assert "Mistral--7B v0.2" in table
    assert r"\textbf{17.0}" in table
    assert table.count(r"\textbf{17.0}") == 6


def test_add_rows_rejects_capacity_overrun() -> None:
    destination = {}
    row = {
        "task": "qasper",
        "id": "sample",
        "capacity": 128,
        "method": MODULE.PROOF_METHOD,
        "score": 0.5,
        "actual_retained_tokens": 129,
    }
    try:
        MODULE.add_rows(
            destination,
            "qwen25",
            [row],
            {MODULE.PROOF_METHOD},
        )
    except ValueError as error:
        assert "invalid retention" in str(error)
    else:
        raise AssertionError("capacity overrun was accepted")
