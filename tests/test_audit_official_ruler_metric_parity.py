from __future__ import annotations

from scripts.audit_official_ruler_metric_parity import official_row_score


def test_official_row_score_matches_string_match_all() -> None:
    assert official_row_score("answer: Alpha and BETA", ["alpha", "beta"]) == 1.0
    assert official_row_score("Alpha only", ["alpha", "beta"]) == 0.5
    assert official_row_score("neither", ["alpha", "beta"]) == 0.0


def test_official_row_score_matches_postprocessing() -> None:
    assert official_row_score("\x00 Alpha \t", ["alpha"]) == 1.0
    assert official_row_score("A  B", ["A B"]) == 0.0
