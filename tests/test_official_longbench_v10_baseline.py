from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("torch")

from run_official_longbench_v10_baseline import (
    allocate_layers,
    headkv_internal_capacity,
)
from audit_official_longbench_v10_baselines import (
    CAPACITIES,
    CONFIGURATIONS,
    PRIMARY_ENDPOINT,
    PRIMARY_WEIGHTING,
    capacity_vector_sha256,
    main as audit_main,
    combine_paired_matrices,
    holm_adjust,
    paired_inference,
    sha256_file,
)


def test_compresskv_layer_allocation_is_exact_and_bounded() -> None:
    capacities = allocate_layers(
        [1.0, 2.0, 3.0, 4.0],
        total_budget=4 * 128,
        minimum=32,
        maximum=256,
    )

    assert sum(capacities) == 512
    assert min(capacities) >= 32
    assert max(capacities) <= 256
    assert capacities == sorted(capacities)


def test_headkv_capacity_is_conservative_in_native_gqa_units() -> None:
    internal = headkv_internal_capacity(
        target_native_tokens=128,
        num_attention_heads=32,
        num_key_value_heads=8,
        window_size=8,
    )

    assert internal == 31
    assert internal * 4 < 128
    assert internal > 8


def test_official_baseline_paired_inference_uses_score_point_units() -> None:
    matrix = np.asarray([[2.0, 4.0], [6.0, 8.0]])

    result = paired_inference(
        matrix,
        bootstrap_replicates=2_000,
        permutation_replicates=2_000,
        rng=np.random.default_rng(7),
    )

    assert result["delta_score_points"] == pytest.approx(5.0)
    assert result["ci95_low"] <= 5.0 <= result["ci95_high"]
    assert 0.0 < result["permutation_p_two_sided"] <= 1.0


def test_official_baseline_holm_adjustment_is_monotone() -> None:
    records = [
        {"permutation_p_two_sided": 0.01},
        {"permutation_p_two_sided": 0.04},
        {"permutation_p_two_sided": 0.03},
    ]

    holm_adjust(records)

    ordered = sorted(records, key=lambda value: value["permutation_p_two_sided"])
    assert [value["holm_p"] for value in ordered] == pytest.approx(
        [0.03, 0.06, 0.06]
    )


def test_official_baseline_combined_endpoint_is_block_balanced() -> None:
    matrices = [
        np.asarray([[1.0, 3.0], [5.0, 7.0]]),
        np.asarray([[3.0, 5.0], [7.0, 9.0]]),
        np.asarray([[5.0, 7.0], [9.0, 11.0]]),
    ]

    combined = combine_paired_matrices(matrices)

    np.testing.assert_allclose(
        combined,
        np.asarray([[3.0, 5.0], [7.0, 9.0]]),
    )


def test_official_baseline_combined_endpoint_rejects_shape_drift() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        combine_paired_matrices(
            [
                np.zeros((2, 3)),
                np.zeros((2, 4)),
            ]
        )


def test_official_baseline_full_audit_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = {
        "tasks": ["task_a"],
        "max_samples_per_task": 1,
        "selections": {
            "task_a": {
                "selected": [{"id": "example_a"}],
            }
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))
    selection_sha256 = sha256_file(selection_path)

    official_repo = tmp_path / "official_repo"
    qwen_source = official_repo / "methods" / "compresskv"
    mistral_source = official_repo / "headkv"
    qwen_source.mkdir(parents=True)
    mistral_source.mkdir(parents=True)
    (qwen_source / "implementation.py").write_text("QWEN = True\n")
    (mistral_source / "implementation.py").write_text("MISTRAL = True\n")

    def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    run_dirs = {}
    target_dirs = {}
    for model in ("qwen25", "mistral02"):
        method = CONFIGURATIONS[model]["method"]
        run_dir = tmp_path / f"official_{model}"
        target_dir = tmp_path / f"target_{model}"
        run_dir.mkdir()
        target_dir.mkdir()
        run_dirs[model] = run_dir
        target_dirs[model] = target_dir
        official_rows = []
        target_rows = []
        capacity_configurations = {}
        for capacity in CAPACITIES:
            retained = float(capacity)
            layer_capacities = [capacity]
            capacity_configurations[str(capacity)] = {
                "layer_capacities": layer_capacities,
            }
            common = {
                "task": "task_a",
                "id": "example_a",
                "capacity": capacity,
                "answers": ["answer"],
                "prediction": "answer",
                "prompt_tokens": 1_000,
                "actual_retained_tokens": retained,
                "actual_retained_ratio": retained / 1_000,
                "actual_kv_bytes": (
                    retained
                    * CONFIGURATIONS[model]["kv_bytes_per_native_token"]
                ),
                "seconds": 1.0,
                "generation_seconds": 1.0,
                "selection_seconds": 0.0,
            }
            official_row = {
                **common,
                "method": method,
                "score": 1.0,
            }
            if method == "official_compresskv":
                official_row.update(
                    {
                        "runtime_capacity_cluster_count": len(
                            layer_capacities
                        ),
                        "runtime_layer_capacities_sha256": (
                            capacity_vector_sha256(layer_capacities)
                        ),
                    }
                )
            official_rows.append(official_row)
            target_rows.append(
                {
                    "task": "task_a",
                    "id": "example_a",
                    "capacity": capacity,
                    "method": "confirmation_stable_proofkv_v10",
                    "score": 0.8,
                }
            )
        official_results = run_dir / "results.jsonl"
        write_jsonl(official_results, official_rows)
        write_jsonl(target_dir / "results.jsonl", target_rows)
        source_root = (
            qwen_source if model == "qwen25" else mistral_source
        )
        source_hashes = {
            str(path.relative_to(source_root)): sha256_file(path)
            for path in source_root.rglob("*.py")
        }
        manifest = {
            "arguments": {
                "official_repo": str(official_repo),
                "method": method,
                "model_label": model,
            },
            "selection_manifest_sha256": selection_sha256,
            "official_commit": CONFIGURATIONS[model]["commit"],
            "environment": {
                "packages": CONFIGURATIONS[model]["packages"],
            },
            "official_source_sha256": source_hashes,
            "official_auxiliary_sha256": {},
            "implementation_sha256": {
                "/synthetic/run_official_longbench_v10_baseline.py": (
                    "RUNNER_HASH_PLACEHOLDER"
                )
            },
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest))
        metrics = {
            "status": "PASS",
            "method": method,
            "model": model,
            "rows": 4,
            "n_per_task": 1,
            "selection_manifest_sha256": selection_sha256,
            "official_commit": CONFIGURATIONS[model]["commit"],
            "results_sha256": sha256_file(official_results),
            "capacities": list(CAPACITIES),
            "capacity_configurations": capacity_configurations,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics))

    artifact_paths = {}
    for name in (
        "controller",
        "ruler_audit",
        "runner",
        "chain",
        "preflight_implementation",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}))
        artifact_paths[name] = path
    runner_hash = sha256_file(artifact_paths["runner"])
    for run_dir in run_dirs.values():
        manifest_path = run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["implementation_sha256"][
            "/synthetic/run_official_longbench_v10_baseline.py"
        ] = runner_hash
        manifest_path.write_text(json.dumps(manifest))
    preflight_path = tmp_path / "preflight_audit.json"
    preflight_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "protocol": (
                    "official_longbench_v10_current_code_smoke_preflight_v1"
                ),
                "runner_implementation_sha256": runner_hash,
                "runs": [{"rows": 20}, {"rows": 20}],
            }
        )
    )
    audit_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "audit_official_longbench_v10_baselines.py"
    )
    protocol = {
        "status": "FROZEN",
        "protocol": "official_longbench_v10_model_compatible_formal_v1",
        "selection_manifest_sha256": selection_sha256,
        "audit_implementation_sha256": sha256_file(audit_path),
        "precommitted_primary_endpoint": PRIMARY_ENDPOINT,
        "primary_equal_weighting": PRIMARY_WEIGHTING,
        "formal_outputs_unobserved_at_freeze": True,
        "controller": str(artifact_paths["controller"]),
        "controller_sha256": sha256_file(artifact_paths["controller"]),
        "ruler_audit": str(artifact_paths["ruler_audit"]),
        "ruler_audit_sha256": sha256_file(artifact_paths["ruler_audit"]),
        "runner": str(artifact_paths["runner"]),
        "runner_implementation_sha256": sha256_file(
            artifact_paths["runner"]
        ),
        "chain": str(artifact_paths["chain"]),
        "chain_implementation_sha256": sha256_file(
            artifact_paths["chain"]
        ),
        "preflight_audit": str(preflight_path),
        "preflight_audit_sha256": sha256_file(preflight_path),
        "preflight_audit_implementation": str(
            artifact_paths["preflight_implementation"]
        ),
        "preflight_audit_implementation_sha256": sha256_file(
            artifact_paths["preflight_implementation"]
        ),
        "models": {
            model: {
                "official_method": values["method"],
                "official_commit": values["commit"],
            }
            for model, values in CONFIGURATIONS.items()
        },
        "capacities": list(CAPACITIES),
        "bootstrap_replicates": 1_000,
        "permutation_replicates": 1_000,
        "seed": 7,
        "formal_output_directories": [
            str(run_dirs["qwen25"].resolve()),
            str(run_dirs["mistral02"].resolve()),
        ],
    }
    protocol_path = tmp_path / "formal_protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    out = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_official_longbench_v10_baselines.py",
            "--selection-manifest",
            str(selection_path),
            "--formal-protocol",
            str(protocol_path),
            "--qwen-run-dir",
            str(run_dirs["qwen25"]),
            "--mistral-run-dir",
            str(run_dirs["mistral02"]),
            "--qwen-v10-run-dir",
            str(target_dirs["qwen25"]),
            "--mistral-v10-run-dir",
            str(target_dirs["mistral02"]),
            "--bootstrap-replicates",
            "1000",
            "--permutation-replicates",
            "1000",
            "--seed",
            "7",
            "--out",
            str(out),
        ],
    )

    audit_main()

    payload = json.loads(out.read_text())
    assert payload["status"] == "PASS"
    assert payload["formal_protocol_sha256"] == sha256_file(protocol_path)
    assert len(payload["run_audits"]) == 2
    assert len(payload["comparisons"]) == 8
    assert payload["primary_inference"]["delta_score_points"] == pytest.approx(
        -20.0
    )
    assert payload["primary_inference"]["task_groups"] == 1
    assert payload["primary_inference"]["samples_per_task"] == 1
