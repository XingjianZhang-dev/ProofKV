import torch

from proofkv.config import CompassConfig, Route
from proofkv.routing import project_query_to_kv_groups, project_to_budget, route_errors_to_table


def test_route_errors_to_cheapest_sufficient():
    cfg = CompassConfig(calibration_threshold=0.9)
    stream = torch.tensor([[0.95, 0.2, 0.1]])
    nav = torch.tensor([[0.99, 0.95, 0.2]])
    table = route_errors_to_table(stream, nav, cfg)
    assert table.routes.tolist() == [[int(Route.STREAM), int(Route.NAV), int(Route.FULL)]]


def test_query_to_kv_projection_prefers_more_memory():
    routes = torch.tensor([[2, 1, 0, 2]])
    out = project_query_to_kv_groups(routes, num_kv_heads=2)
    assert out.tolist() == [[1, 0]]


def test_project_to_budget_downgrades():
    cfg = CompassConfig(sink_size=1, recent_size=1, guidepost_count=1)
    stream = torch.ones(1, 4)
    nav = torch.ones(1, 4)
    table = route_errors_to_table(torch.zeros(1, 4), torch.zeros(1, 4), cfg, threshold=0.9)
    projected = project_to_budget(table, cfg, seq_len=100, target_ratio=0.1, streaming_mass=stream, navigation_mass=nav)
    assert projected.memory_ratio(cfg, 100) <= 0.1


def test_project_to_budget_continues_to_streaming_when_needed():
    cfg = CompassConfig(sink_size=1, recent_size=1, guidepost_count=1)
    stream = torch.ones(1, 4)
    nav = torch.ones(1, 4)
    table = route_errors_to_table(torch.zeros(1, 4), torch.zeros(1, 4), cfg, threshold=0.9)
    projected = project_to_budget(table, cfg, seq_len=100, target_ratio=0.02, streaming_mass=stream, navigation_mass=nav)
    assert projected.memory_ratio(cfg, 100) <= 0.02
    assert projected.routes.tolist() == [[int(Route.STREAM)] * 4]


def test_project_to_budget_raises_when_target_is_unachievable():
    cfg = CompassConfig(sink_size=1, recent_size=1, guidepost_count=1)
    stream = torch.ones(1, 4)
    nav = torch.ones(1, 4)
    table = route_errors_to_table(torch.zeros(1, 4), torch.zeros(1, 4), cfg, threshold=0.9)
    try:
        project_to_budget(table, cfg, seq_len=100, target_ratio=0.01, streaming_mass=stream, navigation_mass=nav)
    except ValueError as exc:
        assert "minimum achievable ratio" in str(exc)
    else:
        raise AssertionError("project_to_budget should reject unreachable budgets")
