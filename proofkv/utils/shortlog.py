from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def short_print(payload: Dict[str, Any]) -> None:
    """One-line JSON output for fast agent iteration."""

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def write_tiny_summary(path: str | Path, payload: Dict[str, Any], max_lines: int = 12) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# ProofKV run summary", ""]
    for i, (k, v) in enumerate(payload.items()):
        if i >= max_lines:
            break
        lines.append(f"- **{k}**: `{v}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
