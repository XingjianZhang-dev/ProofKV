from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_longbench_confirmation_gate import confirmation_summaries


def _statistics(task_delta: float = 0.5) -> dict:
    runs = []
    for model in ("qwen25", "mistral02"):
        capacities = []
        for capacity in (64, 128, 256, 512):
            capacities.append(
                {
                    "capacity": capacity,
                    "utility_delta_vs_fallback": task_delta,
                    "proof_routes": 6,
                    "utility_rows": 60,
                    "task_breakdown": [
                        {
                            "task": task,
                            "utility_delta_vs_fallback": task_delta,
                        }
                        for task in (
                            "narrativeqa",
                            "qasper",
                            "multifieldqa_en",
                            "hotpotqa",
                            "2wikimqa",
                            "musique",
                        )
                    ],
                }
            )
        runs.append(
            {
                "model": model,
                "selection_start_per_task": 28,
                "samples_per_task": 10,
                "capacities": capacities,
            }
        )
    return {
        "analysis_role": "confirmation",
        "excluded_from_final_effect_estimates": True,
        "primary_inference": {
            "paired_task_macro_delta": task_delta,
        },
        "runs": runs,
    }


def test_confirmation_summaries_preserve_equal_axes() -> None:
    summaries = confirmation_summaries(_statistics())
    assert summaries["global_task_macro_delta_f1"] == 0.5
    assert set(summaries["model_macro_delta_f1"]) == {
        "qwen25",
        "mistral02",
    }
    assert set(summaries["capacity_macro_delta_f1"]) == {
        "64",
        "128",
        "256",
        "512",
    }
    assert len(summaries["task_macro_delta_f1"]) == 6
    assert summaries["overall_proof_route_coverage"] == 0.1
