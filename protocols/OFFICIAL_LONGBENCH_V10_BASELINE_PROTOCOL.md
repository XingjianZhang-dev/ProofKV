# Official LongBench-v10 Baseline Protocol

This protocol tests whether the ProofKV-v10 result survives comparison with
model-compatible upstream implementations rather than common-interface
reimplementations.

## Frozen compatibility pairs

- Qwen2.5-7B-Instruct:
  official CompressKV commit
  `12f920fbecba67e16806a6d8eac0ad389ddcad6f`.
- Mistral-7B-Instruct-v0.2:
  official HeadKV/ReasonKV commit
  `0862a0955fe82e9ff611d59541918e02c5def625`.

Each method runs in its pinned Transformers environment and uses the same
five-task, 47-example-per-task selection, prompt tokens, decoder revision, and
64/128/256/512 native-equivalent KV capacities as ProofKV-v10.

## Fail-closed execution order

1. Wait for the untouched Official-RULER formal audit.
2. Run fresh, current-runner 20-row smokes for both compatibility pairs.
3. Independently audit the smoke rows:
   runtime runner hashes, upstream commits and sources, package and CUDA
   extension hashes, row identities, nonempty/nonzero sanity, QA-F1 replay,
   retained-token limits, exact model-specific KV bytes, and timing
   decomposition.
4. Freeze the formal protocol while both 940-row formal output directories are
   still unobserved. The freeze binds the selection, controller, RULER audit,
   current smoke audit, runner, formal auditor, and chain.
5. Run both formal official baselines.
6. Wait for the complete ProofKV-v10 61-configuration matrix.
7. Independently audit both 940-row runs and compute the precommitted
   equal-weight two-model/four-capacity paired endpoint with 50,000 bootstrap
   and 50,000 sign-flip replicates. Apply Holm correction to the eight
   model-capacity secondary tests.

The chain is:

```bash
bash scripts/chain_official_longbench_v10_baselines.sh
```

Supervisor owns the chain, the two preflights, and the two formal runs. Formal
acceptance requires:

- `official_baselines_v10/current_code_smoke_preflight_audit_v10.json`:
  `status=PASS`, two runs, 40 rows total;
- `official_baselines_v10/formal_protocol_v10.json`: `status=FROZEN`;
- `official_baselines_v10/formal_qwen25/results.jsonl`: 940 rows;
- `official_baselines_v10/formal_mistral02/results.jsonl`: 940 rows;
- `official_baselines_v10/formal_audit_statistics.json`: `status=PASS`,
  eight comparisons, 50,000 bootstrap and 50,000 permutation replicates.

Earlier smoke directories are historical diagnostics only. They cannot satisfy
the current-runner preflight gate.
