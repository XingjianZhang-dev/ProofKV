import json
from pathlib import Path


def test_task_subset_preserves_selected_rows(tmp_path: Path) -> None:
    source = {
        "protocol": "locked",
        "tasks": ["a", "b"],
        "selections": {
            "a": {"selected": [{"id": "a0"}]},
            "b": {"selected": [{"id": "b0"}]},
        },
    }
    source_path = tmp_path / "source.json"
    output_path = tmp_path / "out.json"
    source_path.write_text(json.dumps(source))

    import subprocess
    import sys

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "prepare_llmlingua2_shards_20260726.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--selection-manifest",
            str(source_path),
            "--tasks",
            "b",
            "--out",
            str(output_path),
        ],
        check=True,
    )
    output = json.loads(output_path.read_text())
    assert output["tasks"] == ["b"]
    assert output["selections"] == {"b": source["selections"]["b"]}
