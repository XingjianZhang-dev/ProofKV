#!/usr/bin/env python
"""Check that the committed results, tables, and checkpoints agree with one another.

1. Recompute results/followup/summary.json from the per-example records and compare it with the committed file.
2. Regenerate every generated table of the paper and compare it byte for byte with paper/tables/.
3. Check that every source path named in results/paper/canonical_results.json exists under results/paper/records/.
4. If the scorer checkpoints have been unpacked into checkpoints/, check their SHA-256 digests.

    python analysis/verify_release.py

No GPU, model, or dataset download is needed. Exit status is 0 when every check passes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "analysis"
sys.path.insert(0, str(ANALYSIS))
import make_paper_tables  # noqa: E402

# Tables written by hand: they state settings of the method, not results.
STATIC_TABLES = {"candidate_rules.tex", "checkpoints.tex", "training_hparams.tex"}

CHECKPOINT_DIGESTS = {
    "proofkv_scorer_final": {
        "encoder/model.safetensors": "774e7f692bcea6ae945fe4d0eeb90045f9e9d2371d28d76a4287b348c082cd64",
        "heads.pt": "e95a0075813bf2d87d02327960107f7e6f31b6ff8b546d541a859f8a851e7385",
        "proofcache_scorer.json": "815538590808393fd6a119b88a725553efc6929b4c599889bf23c7f65708e41f",
    },
    "proofkv_scorer_musique_replay_v6": {
        "encoder/model.safetensors": "c64c04878baac749343b69607c8b9a157024694ad82fcd30de029e63e2135fb4",
        "heads.pt": "f29ed4a274bbd7b462344dd19cc088fa815940fb38d4f8a02b2ab0c45c6185b2",
        "proofcache_scorer.json": "d6df5faa16394b2600239f5d2b91984b9c87825a734cd77906982a0a72fd86e2",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_followup_summary(tmp: Path) -> list[str]:
    out = tmp / "summary.json"
    subprocess.run([sys.executable, str(ANALYSIS / "followup_statistics.py"), str(REPO / "results" / "followup"), str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    committed = json.loads((REPO / "results" / "followup" / "summary.json").read_text())
    return [] if json.loads(out.read_text()) == committed else ["results/followup/summary.json is not reproduced by followup_statistics.py"]


def check_tables(tmp: Path) -> tuple[list[str], int]:
    out = tmp / "tables"
    out.mkdir()
    subprocess.run([sys.executable, str(ANALYSIS / "make_followup_tables.py"), str(tmp / "summary.json"), str(out),
                    str(REPO / "results" / "paper" / "canonical_results.json")], check=True, stdout=subprocess.DEVNULL)
    make_paper_tables.generate(out)
    problems = []
    generated = {p.name for p in out.glob("*.tex")}
    committed = {p.name for p in (REPO / "paper" / "tables").glob("*.tex")}
    for name in sorted(generated):
        if name not in committed:
            problems.append(f"paper/tables/{name} is missing")
        elif (out / name).read_bytes() != (REPO / "paper" / "tables" / name).read_bytes():
            problems.append(f"paper/tables/{name} differs from the regenerated table")
    for name in sorted(committed - generated - STATIC_TABLES):
        problems.append(f"paper/tables/{name} is neither generated nor listed as static")
    for name in sorted(make_paper_tables.tables_used_by_paper() - committed):
        problems.append(f"paper/tables/{name} is expected but missing")
    return problems, len(generated)


def check_record_sources() -> tuple[list[str], int]:
    records = REPO / "results" / "paper" / "records"
    sources: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "source" and isinstance(value, str):
                    sources.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads((REPO / "results" / "paper" / "canonical_results.json").read_text()))
    missing = sorted(s for s in sources if not (records / s).exists())
    return [f"results/paper/records/{s} is missing" for s in missing], len(sources)


def check_checkpoints() -> tuple[list[str], list[str]]:
    problems, present = [], []
    for name, files in CHECKPOINT_DIGESTS.items():
        root = REPO / "checkpoints" / name
        if not root.is_dir():
            continue
        present.append(name)
        for rel, expected in files.items():
            path = root / rel
            if not path.is_file():
                problems.append(f"checkpoints/{name}/{rel} is missing")
            elif sha256(path) != expected:
                problems.append(f"checkpoints/{name}/{rel} has an unexpected SHA-256 digest")
    return problems, present


def main() -> int:
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        found = check_followup_summary(tmp)
        problems += found
        print(f"[{'ok' if not found else 'FAIL'}] follow-up statistics recomputed from per-example records")
        found, count = check_tables(tmp)
        problems += found
        print(f"[{'ok' if not found else 'FAIL'}] {count} generated tables identical to paper/tables "
              f"({len(STATIC_TABLES)} more are static method settings)")
    found, count = check_record_sources()
    problems += found
    print(f"[{'ok' if not found else 'FAIL'}] {count} source records named by canonical_results.json are present")
    found, present = check_checkpoints()
    problems += found
    if present:
        print(f"[{'ok' if not found else 'FAIL'}] checkpoint digests: {', '.join(present)}")
    else:
        print("[skip] no checkpoints unpacked in checkpoints/ (see checkpoints/README.md)")
    for line in problems:
        print("  -", line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
