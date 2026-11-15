#!/usr/bin/env python3
"""Generate publication tables directly from audited ProofKV-v10 artifacts."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Any


MODEL_NAMES = {
    "qwen25": r"Qwen2.5--7B",
    "mistral02": r"Mistral--7B v0.2",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    workspace = repo / "results" / "paper" / "records"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        default=str(
            workspace
            / "gpu_results_20260724"
            / "acl_main_longbench_structured_20260724"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(os.environ.get("PROOFKV_TABLE_DIR", repo / "paper" / "tables")),
    )
    parser.add_argument(
        "--require-external",
        action="store_true",
        help=(
            "Fail unless the completed official-RULER and official-code "
            "baseline audits are available."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def portable_source(path: Path, experiment_root: Path) -> str:
    """Record a bundle-relative source path instead of a machine-local path."""
    return str(
        Path("gpu_results_20260724")
        / experiment_root.name
        / path.relative_to(experiment_root)
    )


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def format_p(value: float) -> str:
    if value < 0.001:
        return r"p<.001"
    return "p=" + f"{value:.3f}".lstrip("0")


def make_primary_table(
    statistics: dict[str, Any],
    audit: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    if audit.get("status") != "PASS":
        raise ValueError("v10 primary audit did not pass")
    if statistics.get("analysis_role") != "formal":
        raise ValueError("v10 statistics are not formal")
    if statistics.get("excluded_from_final_effect_estimates") is not False:
        raise ValueError("v10 statistics are excluded from final estimates")
    if int(statistics.get("bootstrap_replicates", 0)) != 50_000:
        raise ValueError("v10 bootstrap count is not 50,000")
    if int(statistics.get("permutation_replicates", 0)) != 50_000:
        raise ValueError("v10 permutation count is not 50,000")
    robustness = {
        (str(row["model"]), int(row["capacity"])): row
        for row in statistics["capacity_robustness_inference"]
    }
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & KV & Frozen physical & \proofkv & $\Delta$ & Proof route \\",
        r"\midrule",
    ]
    records = []
    for run_index, run in enumerate(statistics["runs"]):
        model = str(run["model"])
        if model not in MODEL_NAMES:
            raise ValueError(f"unknown model label: {model}")
        if run_index:
            lines.append(r"\midrule")
        for capacity_index, capacity_row in enumerate(run["capacities"]):
            capacity = int(capacity_row["capacity"])
            inference = robustness[(model, capacity)]
            delta = float(capacity_row["utility_delta_vs_fallback"])
            if abs(delta - float(inference["paired_task_macro_delta"])) > 1e-9:
                raise ValueError(f"{model}:{capacity}: delta mismatch")
            holm_p = float(inference["holm_adjusted_p"])
            model_cell = MODEL_NAMES[model] if capacity_index == 0 else ""
            utility = float(capacity_row["utility_score"])
            fallback = float(
                capacity_row["calibration_selected_physical_score"]
            )
            route = 100 * float(capacity_row["proof_route_coverage"])
            utility_cell = (
                rf"\textbf{{{utility:.1f}}}"
                if delta > 0 and holm_p < 0.05
                else f"{utility:.1f}"
            )
            delta_cell = (
                rf"\textbf{{+{delta:.1f}}}$^{{*}}$"
                if delta > 0 and holm_p < 0.05
                else f"{delta:+.1f}"
            )
            lines.append(
                f"{model_cell} & {capacity} & {fallback:.1f} & "
                f"{utility_cell} & {delta_cell} & {route:.0f}\\% \\\\"
            )
            records.append(
                {
                    "model": model,
                    "capacity": capacity,
                    "frozen_physical": fallback,
                    "proofkv": utility,
                    "delta": delta,
                    "proof_route_percent": route,
                    "holm_p": holm_p,
                }
            )
    primary = statistics["primary_inference"]
    delta = float(primary["paired_task_macro_delta"])
    low = float(primary["bootstrap_95_ci_low"])
    high = float(primary["bootstrap_95_ci_high"])
    p_value = float(primary["two_sided_sign_flip_p"])
    lines.extend(
        [
            r"\midrule",
            (
                r"\multicolumn{6}{l}{\footnotesize "
                rf"Primary endpoint: $\Delta={delta:+.2f}$ QA--F1, "
                rf"95\% CI $[{low:.2f},{high:.2f}]$, "
                rf"${format_p(p_value)}$; "
                rf"{int(primary['pairs'])} paired examples.}} \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\vspace{1pt}\par\scriptsize $^{*}$Holm-adjusted "
                r"$p<.05$ within the eight model--capacity comparisons."
            ),
        ]
    )
    write(out_dir / "v10_primary.tex", lines)
    return {
        "records": records,
        "primary": primary,
    }


def make_efficiency_table(
    efficiency: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    if efficiency.get("status") != "PASS":
        raise ValueError("v10 physical-efficiency audit did not pass")
    if efficiency.get("analysis_role") != (
        "descriptive_postformal_physical_efficiency"
    ):
        raise ValueError("unexpected efficiency analysis role")
    fallback = efficiency["utility_vs_frozen_physical_fallback"][
        "equal_weighted_task_macro"
    ]["macro"]
    full = efficiency["utility_vs_full_cache"][
        "equal_weighted_task_macro"
    ]["macro"]
    comparisons = [
        ("Frozen physical", fallback),
        ("Full KV", full),
    ]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Reference & $\Delta$QA--F1 & $\Delta$KV bytes & $\Delta$time & $\Delta$peak \\",
        r"\midrule",
    ]
    records = []
    for label, row in comparisons:
        score = float(row["qa_f1_delta"])
        kv = float(row["actual_kv_bytes_percent_change_of_means"])
        total_time = float(row["total_seconds_percent_change_of_means"])
        peak = float(row["peak_cuda_memory_percent_change_of_means"])
        lines.append(
            f"{label} & {score:+.2f} & {kv:+.3f}\\% & "
            f"{total_time:+.2f}\\% & {peak:+.2f}\\% \\\\"
        )
        records.append(
            {
                "reference": label,
                "qa_f1_delta": score,
                "kv_bytes_percent": kv,
                "offline_total_time_percent": total_time,
                "peak_cuda_memory_percent": peak,
            }
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write(out_dir / "v10_efficiency.tex", lines)
    return {"status": "PASS", "records": records}


def external_result_sentence(
    *,
    subject: str,
    target: float,
    reference: float,
    delta: float,
    ci_low: float,
    ci_high: float,
    p_value: float,
) -> str:
    significant = ci_low > 0.0 and p_value < 0.05
    significantly_negative = ci_high < 0.0 and p_value < 0.05
    if significant:
        relation = "an improvement of"
        magnitude = f"{delta:.1f}"
    elif significantly_negative:
        relation = "a deficit of"
        magnitude = f"{abs(delta):.1f}"
    else:
        relation = "a paired change of"
        magnitude = f"{delta:+.1f}"
    return (
        f"{subject}, \\proofkv reaches {target:.1f} versus {reference:.1f}, "
        f"{relation} {magnitude} points "
        rf"(95\% CI $[{ci_low:.1f},{ci_high:.1f}]$, "
        rf"${format_p(p_value)}$)."
    )


def make_external_table(
    ruler: dict[str, Any],
    official: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    if (
        ruler.get("protocol")
        != "proofkv_v10_official_ruler_paired_inference_v1"
        or int(ruler.get("bootstrap_replicates", 0)) != 50_000
        or int(ruler.get("permutation_replicates", 0)) != 50_000
    ):
        raise ValueError("official-RULER inference contract differs")
    if (
        official.get("status") != "PASS"
        or official.get("protocol")
        != "frozen_model_compatible_official_longbench_v10_audit_and_paired_inference"
        or official.get("capacities") != [64, 128, 256, 512]
        or int(official.get("bootstrap_replicates", 0)) != 50_000
        or int(official.get("permutation_replicates", 0)) != 50_000
    ):
        raise ValueError("official-code baseline audit did not pass")
    ruler_cells = [
        row
        for run in ruler["runs"]
        for row in run["capacities"]
    ]
    if len(ruler_cells) != 8:
        raise ValueError("official-RULER must contain two models x four capacities")
    official_cells = list(official["comparisons"])
    if len(official_cells) != 8:
        raise ValueError("official-code audit must contain eight comparisons")

    ruler_reference = sum(
        float(row["frozen_fallback_score"]) for row in ruler_cells
    ) / len(ruler_cells)
    ruler_target = sum(
        float(row["v10_score"]) for row in ruler_cells
    ) / len(ruler_cells)
    ruler_primary = ruler["primary_inference"]
    official_reference = sum(
        float(row["official_score"]) for row in official_cells
    ) / len(official_cells)
    official_target = sum(
        float(row["target_score"]) for row in official_cells
    ) / len(official_cells)
    official_primary = official["primary_inference"]
    records = [
        {
            "evaluation": "Official RULER",
            "reference": ruler_reference,
            "proofkv": ruler_target,
            "delta": float(ruler_primary["paired_task_macro_delta"]),
            "ci_low": float(ruler_primary["bootstrap_95_ci_low"]),
            "ci_high": float(ruler_primary["bootstrap_95_ci_high"]),
            "p": float(ruler_primary["two_sided_sign_flip_p"]),
        },
        {
            "evaluation": "Official method code",
            "reference": official_reference,
            "proofkv": official_target,
            "delta": float(official_primary["delta_score_points"]),
            "ci_low": float(official_primary["ci95_low"]),
            "ci_high": float(official_primary["ci95_high"]),
            "p": float(official_primary["permutation_p_two_sided"]),
        },
    ]
    # RULER is architecture matched: it changes the generator while preserving
    # the frozen controller and physical interface.  The official-code test
    # instead substitutes a generic selector for ProofKV's EvidencePack and is
    # therefore emitted as a scoped boundary table for the appendix.
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Evaluation & Reference & \proofkv & $\Delta$ & 95\% CI \\",
        r"\midrule",
    ]
    for row in records[:1]:
        significant = row["ci_low"] > 0.0 and row["p"] < 0.05
        proofkv = (
            rf"\textbf{{{row['proofkv']:.1f}}}"
            if significant
            else f"{row['proofkv']:.1f}"
        )
        delta = (
            rf"\textbf{{{row['delta']:+.1f}}}"
            if significant
            else f"{row['delta']:+.1f}"
        )
        lines.append(
            f"{row['evaluation']} & {row['reference']:.1f} & "
            f"{proofkv} & {delta} & "
            rf"$[{row['ci_low']:.1f},{row['ci_high']:.1f}]$ \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write(out_dir / "v10_external.tex", lines)
    boundary = records[1]
    boundary_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Selector substitution & Official & Target & $\Delta$ & 95\% CI \\",
        r"\midrule",
        (
            f"Generic direct selector & {boundary['reference']:.1f} & "
            f"{boundary['proofkv']:.1f} & {boundary['delta']:+.1f} & "
            rf"$[{boundary['ci_low']:.1f},{boundary['ci_high']:.1f}]$ \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write(out_dir / "v10_official_boundary.tex", boundary_lines)
    claims = [
        external_result_sentence(
            subject="On the official RULER generator",
            target=records[0]["proofkv"],
            reference=records[0]["reference"],
            delta=records[0]["delta"],
            ci_low=records[0]["ci_low"],
            ci_high=records[0]["ci_high"],
            p_value=records[0]["p"],
        ),
        (
            "This architecture-matched shift changes the data generator while "
            "retaining the frozen controller and physical interface. A separate "
            "generic direct-selector substitution is reported as a scope "
            "boundary in Appendix~\\ref{app:official-boundary}."
        ),
    ]
    write(out_dir / "v10_external_claims.tex", [" ".join(claims)])
    return {
        "status": "PASS",
        "records": records,
        "main_record": records[0],
        "boundary_record": boundary,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.experiment_root)
    audits = root / "audits"
    out_dir = Path(args.out_dir)
    statistics_path = audits / "formal_primary_v10_statistics.json"
    audit_path = audits / "formal_primary_v10_both_complete.json"
    efficiency_path = audits / "formal_primary_v10_physical_efficiency.json"
    ruler_path = root / "official_ruler_v10" / "formal_statistics_v10.json"
    official_path = (
        root
        / "official_baselines_v10"
        / "formal_audit_statistics_state_reset_r1.json"
    )
    statistics = load_json(statistics_path)
    audit = load_json(audit_path)
    efficiency = load_json(efficiency_path)
    payload = {
        "status": "PASS",
        "sources": {
            "statistics": portable_source(statistics_path, root),
            "audit": portable_source(audit_path, root),
            "efficiency": portable_source(efficiency_path, root),
        },
        "primary_table": make_primary_table(
            statistics,
            audit,
            out_dir,
        ),
        "efficiency_table": make_efficiency_table(
            efficiency,
            out_dir,
        ),
    }
    if ruler_path.is_file() and official_path.is_file():
        payload["sources"]["ruler_statistics"] = portable_source(ruler_path, root)
        payload["sources"]["official_baseline_audit"] = portable_source(
            official_path, root
        )
        payload["external_table"] = make_external_table(
            load_json(ruler_path),
            load_json(official_path),
            out_dir,
        )
    elif args.require_external:
        missing = [
            str(path)
            for path in (ruler_path, official_path)
            if not path.is_file()
        ]
        raise FileNotFoundError(
            "required external artifacts are missing: " + ", ".join(missing)
        )
    else:
        payload["external_table"] = {
            "status": "PENDING",
            "missing": [
                str(path)
                for path in (ruler_path, official_path)
                if not path.is_file()
            ],
        }
    summary = out_dir / "v10_generated_summary.json"
    summary.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "summary": str(summary)}))


if __name__ == "__main__":
    main()
