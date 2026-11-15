#!/usr/bin/env python3
"""Generate the risk-routing tables and quality--budget figure from frozen audits."""

from __future__ import annotations

import argparse
import os
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPARATORS = (
    "entity_graph_adaptive",
    "learned_root_topk",
    "learned_and_or",
    "full",
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    workspace = repo / "results" / "paper" / "records"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=(
            workspace
            / "gpu_results_20260724"
            / "acl_main_longbench_structured_20260724"
        ),
    )
    parser.add_argument("--table-dir", type=Path, default=Path(os.environ.get("PROOFKV_TABLE_DIR", repo / "paper" / "tables")))
    parser.add_argument("--figure-dir", type=Path, default=Path(os.environ.get("PROOFKV_TABLE_DIR", repo / "paper" / "tables")))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_source(path: Path, experiment_root: Path) -> str:
    """Record a bundle-relative source path instead of a machine-local path."""
    return str(
        Path("gpu_results_20260724")
        / experiment_root.name
        / path.relative_to(experiment_root)
    )


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def validate_V22(protocol: dict[str, Any], audit: dict[str, Any]) -> None:
    if (
        protocol.get("status") != "FROZEN"
        or float(protocol.get("threshold", -1)) != 0.17
        or audit.get("status") != "REJECT"
    ):
        raise ValueError("unexpected V22 protocol or audit state")
    checks = audit.get("checks", {})
    if checks.get("minimum_cell_vs_full") is not False or not all(
        value is True
        for key, value in checks.items()
        if key != "minimum_cell_vs_full"
    ):
        raise ValueError("V22 must fail only the uniform per-cell guard")
    for comparator in COMPARATORS:
        inference = audit["paired_inference"][comparator]
        if (
            float(inference["macro_delta_points"]) <= 0
            or float(inference["ci95_points"][0]) <= 0
            or float(inference["holm_adjusted_p"]) >= 0.05
        ):
            raise ValueError(f"V22 aggregate comparison is not positive: {comparator}")


def validate_V25(protocol: dict[str, Any], audit: dict[str, Any]) -> None:
    if (
        protocol.get("status") != "FROZEN"
        or audit.get("status") != "PASS"
        or audit.get("mode") != "final"
        or not all(audit.get("checks", {}).values())
    ):
        raise ValueError("V25 quality--budget evidence is not complete")
    budgets = [float(value) for value in protocol["budgets"]]
    if budgets != [0.04, 0.08, 0.12]:
        raise ValueError("unexpected V25 budget grid")
    for budget in budgets:
        row = audit["budget_macro"][str(budget)]
        if min(row["delta_vs_each_compact_points"].values()) <= 0:
            raise ValueError(f"non-positive compact comparison at budget {budget}")
    entity = audit["paired_curve_inference"]["entity_graph_adaptive"]
    if (
        float(entity["ci95_points"][0]) <= 0
        or float(entity["holm_adjusted_p"]) >= 0.05
    ):
        raise ValueError("V25 curve-level Entity comparison is not significant")


def make_tables(
    table_dir: Path,
    V22: dict[str, Any],
    V25: dict[str, Any],
) -> dict[str, Any]:
    budgets = [0.04, 0.08, 0.12]
    budget_rows = []
    budget_lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Budget & Retained & F1 & $\Delta$Entity & $\Delta$Root & $\Delta$AND--OR \\",
        r"\midrule",
    ]
    for budget in budgets:
        row = V25["budget_macro"][str(budget)]
        deltas = row["delta_vs_each_compact_points"]
        record = {
            "budget_percent": 100 * budget,
            "retained_percent": 100 * float(row["controller_retained_ratio"]),
            "controller_f1": 100 * float(row["controller_f1"]),
            "delta_entity": float(deltas["entity_graph_adaptive"]),
            "delta_root": float(deltas["learned_root_topk"]),
            "delta_and_or": float(deltas["learned_and_or"]),
        }
        budget_rows.append(record)
        budget_lines.append(
            f"{record['budget_percent']:.0f}\\% & {record['retained_percent']:.2f}\\% & "
            f"{record['controller_f1']:.2f} & {record['delta_entity']:+.2f} & "
            f"{record['delta_root']:+.2f} & {record['delta_and_or']:+.2f} \\\\"
        )
    budget_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_lines(table_dir / "V25_budget_pareto.tex", budget_lines)

    length_rows = []
    length_lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"Target & Controller F1 & $\Delta$AND--OR & $\Delta$Full \\",
        r"\midrule",
    ]
    for target in (8000, 16000, 32000):
        cells = [
            row
            for row in V22["cell_metrics"].values()
            if int(row["target_prompt_tokens"]) == target
        ]
        if len(cells) != 2:
            raise ValueError(f"V22 target {target}: expected two task cells")
        record = {
            "target_tokens": target,
            "controller_f1": 100
            * sum(float(row["controller_f1"]) for row in cells)
            / len(cells),
            "delta_and_or": sum(
                float(row["delta_vs_each_method_points"]["learned_and_or"])
                for row in cells
            )
            / len(cells),
            "delta_full": sum(
                float(row["delta_vs_each_method_points"]["full"]) for row in cells
            )
            / len(cells),
        }
        length_rows.append(record)
        length_lines.append(
            f"{target // 1000}k & {record['controller_f1']:.2f} & "
            f"{record['delta_and_or']:+.2f} & {record['delta_full']:+.2f} \\\\"
        )
    length_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_lines(table_dir / "V22_risk_length.tex", length_lines)
    return {"budget_rows": budget_rows, "length_rows": length_rows}


def make_figure(figure_dir: Path, budget_rows: list[dict[str, Any]], V25: dict[str, Any]) -> None:
    """Ordered dots plus grouped bars.

    Analytical question: does the frozen predecode controller produce a useful
    quality--memory frontier, and are its gains an artifact of one fixed expert?
    Takeaway: F1 rises monotonically from 4% to 12%, while every task-macro
    comparison against the three fixed compact experts remains positive.
    """

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )
    blue = "#245B84"
    blue_mid = "#4F86AA"
    blue_open = "#AFC9DA"
    gold = "#C89B2C"
    ink = "#202832"
    grey = "#7B858F"
    grid = "#D9DEE3"

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.22))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.23, wspace=0.31)

    budgets = np.array([row["budget_percent"] for row in budget_rows])
    retained = np.array([row["retained_percent"] for row in budget_rows])
    f1 = np.array([row["controller_f1"] for row in budget_rows])
    curve_full_delta = float(V25["paired_curve_inference"]["full"]["macro_delta_points"])
    full_f1 = float(np.mean(f1) - curve_full_delta)

    ax = axes[0]
    ax.plot(
        budgets,
        f1,
        color=blue,
        marker="o",
        markersize=5.0,
        linewidth=1.65,
        markeredgecolor=ink,
        markeredgewidth=0.5,
        label="Risk-routed EvidencePack",
        zorder=3,
    )
    ax.axhline(
        full_f1,
        color=grey,
        linestyle=(0, (3, 2)),
        linewidth=1.15,
        label="Full context (100% retained)",
        zorder=1,
    )
    for x, y in zip(budgets, f1):
        ax.annotate(
            f"{y:.1f}",
            (x, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            color=ink,
            fontsize=6.8,
            fontweight="bold",
        )
    ax.annotate(
        f"{full_f1:.1f}",
        (12.15, full_f1),
        ha="left",
        va="center",
        fontsize=6.5,
        color=grey,
    )
    ax.set_xticks(
        budgets,
        [f"{b:.0f}%\n({r:.1f}% kept)" for b, r in zip(budgets, retained)],
    )
    ax.set_xlim(2.8, 13.3)
    ax.set_ylim(28, 43)
    ax.set_ylabel("Answer F1 (%)")
    ax.set_xlabel("Requested memory budget (actual retained)")
    ax.set_title("(a) Quality across frozen budgets", loc="left", fontweight="bold")
    ax.grid(axis="y", color=grid, linewidth=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", handlelength=2.6)

    ax = axes[1]
    width = 0.22
    positions = np.arange(len(budgets))
    series = [
        ("Entity graph", "delta_entity", blue, ""),
        ("Root ranking", "delta_root", gold, "//"),
        ("AND--OR", "delta_and_or", blue_open, ".."),
    ]
    for index, (label, key, color, hatch) in enumerate(series):
        values = np.array([row[key] for row in budget_rows])
        bars = ax.bar(
            positions + (index - 1) * width,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor=ink,
            linewidth=0.45,
            hatch=hatch,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.25,
                f"+{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.0,
                color=ink,
            )
    ax.axhline(0, color=ink, linewidth=0.7)
    ax.set_xticks(positions, [f"{budget:.0f}%" for budget in budgets])
    ax.set_ylim(0, 11.4)
    ax.set_ylabel("Controller gain (F1 points)")
    ax.set_xlabel("Requested memory budget")
    ax.set_title("(b) Gain over fixed compact experts", loc="left", fontweight="bold")
    ax.grid(axis="y", color=grid, linewidth=0.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.8,
    )

    for axis in axes:
        axis.tick_params(length=2.5, width=0.6, color=ink)
        axis.spines["left"].set_color(ink)
        axis.spines["bottom"].set_color(ink)
    fig.savefig(
        figure_dir / "evidencepack_pareto.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": "EvidencePack quality-budget Pareto curve",
            "Creator": "ProofKV table generator",
        },
    )
    fig.savefig(
        figure_dir / "evidencepack_pareto.svg",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": "EvidencePack quality-budget Pareto curve",
            "Creator": "ProofKV table generator",
        },
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.experiment_root.resolve()
    V22_protocol_path = root / "evidencepack_risk_controller_V22_formal/frozen_protocol.json"
    V22_audit_path = root / "evidencepack_risk_controller_V22_formal/formal_audit.json"
    V25_protocol_path = root / "evidencepack_budget_pareto_V25/frozen_protocol.json"
    V25_audit_path = root / "evidencepack_budget_pareto_V25/formal_audit.json"
    closure_path = root / "audits/acl_main_architecture_matched_evidence_closure_V26_final.json"
    V22_protocol = load_json(V22_protocol_path)
    V22 = load_json(V22_audit_path)
    V25_protocol = load_json(V25_protocol_path)
    V25 = load_json(V25_audit_path)
    closure = load_json(closure_path)
    validate_V22(V22_protocol, V22)
    validate_V25(V25_protocol, V25)
    if not (
        closure.get("gpu_evidence_complete") is True
        and closure.get("positive_core_claims_passed") is True
        and closure.get("paper_experiment_loop_closed") is True
        and not closure.get("remaining_required_components")
        and not closure.get("artifact_issues")
    ):
        raise ValueError("final architecture-matched closure audit did not pass")

    tables = make_tables(args.table_dir, V22, V25)
    make_figure(args.figure_dir, tables["budget_rows"], V25)
    summary = {
        "status": "PASS",
        "sources": {
            "V22_protocol": portable_source(V22_protocol_path, root),
            "V22_protocol_sha256": sha256(V22_protocol_path),
            "V22_audit": portable_source(V22_audit_path, root),
            "V22_audit_sha256": sha256(V22_audit_path),
            "V25_protocol": portable_source(V25_protocol_path, root),
            "V25_protocol_sha256": sha256(V25_protocol_path),
            "V25_audit": portable_source(V25_audit_path, root),
            "V25_audit_sha256": sha256(V25_audit_path),
            "closure": portable_source(closure_path, root),
            "closure_sha256": sha256(closure_path),
        },
        "V22": {
            "paired_inference": V22["paired_inference"],
            "target_macro_delta_vs_primary_points": V22[
                "target_macro_delta_vs_primary_points"
            ],
            "mean_selected_retained_ratio": V22["mean_selected_retained_ratio"],
            "only_failed_uniform_cell_guard": True,
        },
        "V25": {
            "scientific_outcome": V25["scientific_outcome"],
            "paired_curve_inference": V25["paired_curve_inference"],
            "budget_rows": tables["budget_rows"],
            "length_rows": tables["length_rows"],
        },
    }
    summary_path = args.table_dir / "evidencepack_generated_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "summary": str(summary_path),
                "figure": str(args.figure_dir / "evidencepack_pareto.pdf"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
