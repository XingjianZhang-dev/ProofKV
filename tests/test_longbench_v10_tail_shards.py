from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_longbench_v10_tail_shards import (  # noqa: E402
    expected_keys,
    main as merge_main,
    validate_duplicate,
)
from prepare_longbench_v10_tail_shards import parse_shards  # noqa: E402


def test_tail_shard_declarations_are_explicit_and_nonempty() -> None:
    assert parse_shards(
        [
            "mistral02_2wikimqa=2wikimqa",
            "mistral02_musique=musique",
        ]
    ) == {
        "mistral02_2wikimqa": ["2wikimqa"],
        "mistral02_musique": ["musique"],
    }

    with pytest.raises(ValueError, match="invalid shard"):
        parse_shards(["mistral02_musique="])


def test_tail_merge_expected_keyspace_has_61_configs_per_example() -> None:
    selection = {
        "tasks": ["task_a"],
        "selections": {
            "task_a": {
                "selected": [
                    {"id": "a"},
                    {"id": "b"},
                ]
            }
        },
    }

    keys, task_order, id_order = expected_keys(selection)

    assert len(keys) == 2 * 61
    assert task_order == {"task_a": 0}
    assert id_order == {"task_a:a": 0, "task_a:b": 1}
    assert ("task_a", "a", "full", "full") in keys
    assert (
        "task_a",
        "a",
        "confirmation_stable_proofkv_v10",
        "512",
    ) in keys


def test_tail_merge_duplicate_requires_exact_scientific_replay() -> None:
    row = {
        "answers": ["answer"],
        "prediction": "answer",
        "score": 1.0,
        "prompt_tokens": 1_000,
        "question_tokens": 20,
        "context_tokens": 980,
        "actual_retained_tokens": 128,
        "actual_retained_ratio": 0.128,
        "actual_kv_bytes": 65_536,
        "seconds": 2.0,
    }
    replay = {**row, "seconds": 3.0}

    validate_duplicate(("task", "id", "method", "128"), row, replay)

    replay["prediction"] = "different"
    with pytest.raises(ValueError, match="prediction"):
        validate_duplicate(("task", "id", "method", "128"), row, replay)


def test_tail_merge_end_to_end_preserves_exact_keyspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = {
        "protocol": "test",
        "tasks": ["task_a"],
        "max_samples_per_task": 2,
        "selections": {
            "task_a": {
                "selected": [
                    {"id": "a"},
                    {"id": "b"},
                ]
            }
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "parent_selection_manifest_sha256": selection_sha256,
                "scientific_selection_unchanged": True,
                "shards": [
                    {
                        "name": "shard",
                        "tasks": ["task_a"],
                    }
                ],
            }
        )
    )
    keys, _, _ = expected_keys(selection)

    def make_row(key: tuple[str, str, str, str]) -> dict[str, object]:
        task, source_id, method, capacity = key
        retained = 1_000 if capacity == "full" else int(capacity)
        return {
            "task": task,
            "id": source_id,
            "method": method,
            "capacity": capacity if capacity == "full" else int(capacity),
            "answers": ["answer"],
            "prediction": "answer",
            "score": 1.0,
            "prompt_tokens": 1_000,
            "question_tokens": 20,
            "context_tokens": 980,
            "actual_retained_tokens": retained,
            "actual_retained_ratio": retained / 1_000,
            "actual_kv_bytes": retained * 512,
            "seconds": 1.0,
        }

    original_dir = tmp_path / "original"
    original_dir.mkdir()
    original_rows = [
        make_row(key) for key in keys if key[1] == "a"
    ]
    (original_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in original_rows)
    )
    run_contract = {
        "selection_manifest_sha256": selection_sha256,
        "implementation_sha256": "implementation",
        "scorer_checkpoint_sha256": {"scorer": "hash"},
        "risk_controller_sha256": "controller",
    }
    (original_dir / "run_manifest.json").write_text(
        json.dumps({**run_contract, "arguments": {}})
    )

    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    shard_rows = [make_row(key) for key in keys]
    (shard_dir / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in shard_rows)
    )
    (shard_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": "mistral02",
                "n_per_task": 2,
                "tasks": ["task_a"],
                "selection_manifest_sha256": selection_sha256,
            }
        )
    )
    (shard_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                **run_contract,
                "arguments": {"tasks": "task_a"},
            }
        )
    )
    merge_manifest = tmp_path / "merge.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_longbench_v10_tail_shards.py",
            "--selection-manifest",
            str(selection_path),
            "--shard-index",
            str(index_path),
            "--model-label",
            "mistral02",
            "--original-run-dir",
            str(original_dir),
            "--shard-run-dir",
            str(shard_dir),
            "--out",
            str(merge_manifest),
        ],
    )

    merge_main()

    merged_rows = [
        json.loads(line)
        for line in (original_dir / "results.jsonl").read_text().splitlines()
    ]
    audit = json.loads(merge_manifest.read_text())
    assert len(merged_rows) == len(keys) == 122
    assert audit["status"] == "PASS"
    assert audit["duplicate_rows_exactly_replayed"] == 61
    assert audit["merged_unique_keys"] == 122
    assert sum(
        1
        for line in (
            original_dir / "results_pre_tail_merge.jsonl"
        ).read_text().splitlines()
        if line
    ) == 61
