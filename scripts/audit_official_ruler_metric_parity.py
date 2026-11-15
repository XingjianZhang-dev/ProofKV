#!/usr/bin/env python
"""Replay every stored RULER row with NVIDIA RULER's exact string metric."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_RULER_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_row_score(prediction: str, references: list[str]) -> float:
    """Match RULER postprocess_pred followed by string_match_all."""

    prediction = str(prediction).strip()
    prediction = re.sub(r"[\x00-\x1f]", "\n", prediction).strip()
    if not references:
        raise ValueError("RULER row has no references")
    return sum(
        1.0 if str(reference).lower() in prediction.lower() else 0.0
        for reference in references
    ) / len(references)


def git_commit(repo: Path) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo.resolve()}",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def audit(official_repo: Path, run_dirs: list[Path]) -> dict[str, Any]:
    commit = git_commit(official_repo)
    if commit != EXPECTED_RULER_COMMIT:
        raise ValueError(f"official RULER commit differs: {commit}")
    metric_path = official_repo / "scripts/eval/synthetic/constants.py"
    evaluate_path = official_repo / "scripts/eval/evaluate.py"
    if not metric_path.is_file() or not evaluate_path.is_file():
        raise FileNotFoundError("official RULER evaluation source is missing")

    run_records = []
    total_rows = 0
    for run_dir in sorted(run_dirs, key=lambda path: str(path.resolve())):
        results_path = run_dir / "results.jsonl"
        if not results_path.is_file():
            raise FileNotFoundError(results_path)
        rows = [
            json.loads(line)
            for line in results_path.read_text().splitlines()
            if line.strip()
        ]
        mismatches = []
        for row in rows:
            observed = float(row["score"])
            replayed = official_row_score(
                str(row["prediction"]),
                [str(value) for value in row["answers"]],
            )
            if abs(observed - replayed) > 1.0e-12:
                mismatches.append(
                    {
                        "id": str(row["id"]),
                        "method": str(row["method"]),
                        "capacity": row["capacity"],
                        "observed": observed,
                        "official_replay": replayed,
                    }
                )
        if mismatches:
            raise ValueError(
                f"{run_dir}: {len(mismatches)} official-metric mismatches; "
                f"first={mismatches[0]}"
            )
        total_rows += len(rows)
        run_records.append(
            {
                "run_dir": str(run_dir.resolve()),
                "rows": len(rows),
                "official_metric_mismatches": 0,
                "results_sha256": sha256_file(results_path),
            }
        )
    return {
        "status": "PASS",
        "protocol": "proofkv_v10_official_ruler_metric_parity_v1",
        "official_repo": str(official_repo.resolve()),
        "official_commit": commit,
        "official_metric_source_sha256": sha256_file(metric_path),
        "official_postprocess_source_sha256": sha256_file(evaluate_path),
        "runs": run_records,
        "rows": total_rows,
        "official_metric_mismatches": 0,
        "audit_implementation_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    args = parse_args()
    payload = audit(
        Path(args.official_repo),
        [Path(value) for value in args.run_dir],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["rows"],
                "official_metric_mismatches": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
