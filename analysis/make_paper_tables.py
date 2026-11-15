#!/usr/bin/env python
"""Regenerate the tables of the original evaluation from the released records.

Runs the five generators in analysis/paper_tables/ on results/paper/, applies the camera-ready wording
(analysis/paper_tables/camera_ready_labels.py), and writes the tables that appear in the paper to the output directory.

    python analysis/make_paper_tables.py [--out paper/tables]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER_TABLES = (
    "baseline_matrix.tex",
    "component_metrics.tex",
    "context_halos.tex",
    "contract_endpoints.tex",
    "contract_transparent.tex",
    "crossmodel_statistics.tex",
    "dense_core.tex",
    "evaluation_inventory.tex",
    "failure_gates.tex",
    "headkv_matrix.tex",
    "hop_topology.tex",
    "interpolation.tex",
    "kvpress_matrix.tex",
    "kvzip_full.tex",
    "longbench.tex",
    "matrix_2wiki.tex",
    "matrix_hotpot.tex",
    "matrix_mistral02.tex",
    "matrix_mistral03.tex",
    "matrix_musique.tex",
    "matrix_qwen3.tex",
    "multiquery.tex",
    "packed_protocol.tex",
    "packing_seeds.tex",
    "policy_hyperparameters.tex",
    "precision.tex",
    "qwen14b_scale_transfer.tex",
    "reuse_q.tex",
    "route_frontier.tex",
    "selector.tex",
    "serving.tex",
    "statistical_ledger.tex",
    "support_coverage.tex",
    "synthetic_topology.tex",
    "topology.tex",
    "training_negatives.tex",
    "training_stages.tex",
    "v10_efficiency.tex",
    "v10_external.tex",
    "v10_official_boundary.tex",
    "v22_risk_length.tex",
    "v25_budget_pareto.tex",
    "vllm_backend.tex",
)
GENERATORS = ("main_tables", "five_task_tables", "baseline_tables", "routing_tables", "qwen14b_table")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_tables import camera_ready_labels  # noqa: E402


def tables_used_by_paper() -> set[str]:
    """Tables of the original evaluation that appear in the paper (the generators also write a few that do not)."""
    return set(PAPER_TABLES)


def generate(out_dir: Path) -> list[str]:
    summary = json.loads((REPO / "results" / "followup" / "summary.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, PROOFKV_TABLE_DIR=tmp, MPLBACKEND="Agg")
        for generator in GENERATORS:
            script = Path(__file__).resolve().parent / "paper_tables" / f"{generator}.py"
            subprocess.run([sys.executable, str(script)], check=True, env=env, stdout=subprocess.DEVNULL)
        camera_ready_labels.apply(tmp, summary)
        used = tables_used_by_paper()
        written = sorted(p.name for p in Path(tmp).glob("*.tex") if p.name in used)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in written:
            shutil.copyfile(Path(tmp) / name, out_dir / name)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "paper" / "tables")
    args = parser.parse_args()
    names = generate(args.out)
    print(f"wrote {len(names)} tables to {args.out}")
