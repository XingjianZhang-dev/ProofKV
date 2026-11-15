from __future__ import annotations

from analyze_reuse_q_20260726 import projected_reuse_rows


def test_q1_matches_cold_and_reuse_amortizes_index() -> None:
    payload = {
        "protocol": {"samples": 4},
        "records": [
            {
                "output_tokens": 256,
                "mode": "full",
                "total_seconds_p50": 10.0,
            },
            {
                "output_tokens": 256,
                "mode": "shared_index",
                "total_seconds_p50": 6.0,
            },
            {
                "output_tokens": 256,
                "mode": "cold",
                "total_seconds_p50": 9.0,
            },
        ],
    }
    rows = projected_reuse_rows(payload, "metrics.json")
    by_q = {
        row["requests_per_document"]: row
        for row in rows
    }
    assert by_q[1]["proofkv_total_seconds"] == 9.0
    assert by_q[1]["q1_matches_measured_cold"]
    assert by_q[2]["proofkv_total_seconds"] == 15.0
    assert by_q[8]["full_over_proofkv_speedup"] > by_q[2][
        "full_over_proofkv_speedup"
    ]
