# Official RULER v10 protocol

This extension tests the same physical ProofKV-v10 implementation used by the
non-truncated LongBench matrix on official NVIDIA RULER data.

## Locked design

- Source: NVIDIA/RULER commit
  `38da79d79519ef87aa46ae804f838e1eab7f86d7`.
- Tasks: NIAH and variable tracking.
- Models: Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.2.
- Total-prefill KV capacities: 128, 256, 512, and 1024 tokens.
- Physical baselines: StreamingLLM, SnapKV, PyramidKV, and ChunkKV.
- Development: 10 examples per task, shared across models.
- Formal: 47 untouched examples per task, shared across models.
- Development and formal generator/selection seeds differ; selected ID overlap
  is required to be zero.

The lower capacity starts at 128 because the official variable-tracking
question and answer prefix occupy 60 tokens. The common physical runner keeps
at least five context tokens, so a 64-token total-prefill budget is infeasible
without changing the official prompt or violating the capacity contract.

## Frozen inputs

- Development manifest SHA-256:
  `8172e7638cf7174503ad11ea81d72806c663e7509e52c88859b376aa828692e9`.
- Formal manifest SHA-256:
  `5d71da9bddafbef4e9f90e5b9ef0e28bc8ae027afe0b9ff72a0b7672cfdcd050`.

`scripts/prepare_official_ruler_v10_split.py` regenerates these manifests and
checks the RULER commit. `scripts/run_official_ruler_upgraded_v10.py` adapts the
v10 physical runner without modifying its frozen LongBench implementation.

## Prospective route freeze

The development matrix contains Full, all four physical baselines, the
25/50/75 Control Anchor candidates, and a RULER chain adapter. The adapter
exposes the repository's existing prompt-visible NIAH/variable-assignment
compiler through the same physical v10 decode path; it does not inspect
answers or generated tokens. For each model–task–capacity cell:

1. choose the highest-development-score physical fallback;
2. admit a Control Anchor or the prompt-visible chain adapter only when its
   paired mean gain is at least 0.02 and its 10th-percentile bootstrap lower
   bound is nonnegative;
3. otherwise retain the physical fallback.

`scripts/freeze_official_ruler_v10_controller.py` performs this decision and
refuses to run if any formal result row already exists.
`scripts/freeze_official_ruler_v10_protocol.py` then freezes the controller,
formal manifest, implementation hashes, audit code, and inference settings
before either formal model is started.

## Completion gates

The formal run is complete only when:

1. both models contain 1,974 rows and a completed metrics artifact;
2. `scripts/audit_official_ruler_v10.py` replays every official score, checks
   every configuration crossing and route, and reports no capacity violation;
3. `scripts/audit_official_ruler_v10_protocol_binding.py` verifies every
   implementation hash in the pre-output protocol against disk and proves that
   both runtime manifests used the same hash-matching implementation subset;
4. `scripts/audit_official_ruler_metric_parity.py` replays every stored score
   using the exact upstream `postprocess_pred` and `string_match_all` semantics
   at the frozen NVIDIA RULER commit;
5. `scripts/analyze_official_ruler_v10_statistics.py` produces the paired
   task-stratified 50k-bootstrap/50k-sign-flip endpoint and Holm-adjusted
   capacity family;
6. `scripts/analyze_official_ruler_v10_efficiency.py` produces a two-model
   physical-efficiency audit with the prospectively frozen RULER capacities
   128/256/512/1024, while retaining the hash and all accounting checks of the
   unchanged generic LongBench efficiency analyzer.

The generic analyzer was frozen before RULER output and encodes LongBench's
64/128/256/512 capacity tuple. It is therefore left immutable. The independent
RULER wrapper specializes only that declared tuple, calls the frozen analyzer,
then fail-closes unless both 1,974-row runs and exactly the four RULER
capacities are present. Its SHA-256 is
`2f96359bb8ad9bad9d4aa3ec43c2ea6ce338a22d39fc901915cbb82b56ada43f`;
the supplemental audit chain SHA-256 is
`9e3291bb80b32d7d805e13dac9a6474df47893164ab5e8d845e3308ae98cefc4`.
The supplemental chain also stops any development child processes that a
failed generic-analyzer autorestart may have launched before the parent chain
was stopped; this scheduling cleanup cannot modify formal rows.
The output protocol is
`proofkv_v10_official_ruler_physical_efficiency_audit_v1`.

`scripts/chain_official_ruler_v10.sh` may execute concurrently with the
independent LongBench full matrix to reduce wall-clock cost. Within RULER, the
stage ordering is strict: no formal output can precede completion of
development, controller freezing, and formal-protocol freezing.

## Final audited result

Both formal runs completed with 1,974 rows. The precommitted equal-weighted
two-model endpoint is +39.8138 official score points versus the
development-frozen physical fallback, with a 95% task-stratified bootstrap
interval [38.0053, 41.5160] and a 50k two-sided sign-flip p-value below
2e-5. Exact upstream metric replay reports zero mismatches across all 3,948
rows.

- Formal integrity audit SHA-256:
  `14d53f0bd7f4bf0dc6fe439335d185f7af99e49f7755b03218225d6a7be54b3e`.
- Formal statistics SHA-256:
  `dc5c118663524266112cce4f83b738e4f4a0a3fc6c4f8723393d1eea9fb28326`.
- Protocol-binding audit SHA-256:
  `c029b33f5eea6c8b0935f66566f3f2a902b3c865f94e9eb6177d77acf4a3eb0b`.
- Exact-upstream metric audit SHA-256:
  `1b60b313412729e32789b402886c31c07dd5f84f60c968ec4ac0bacd8533b608`.
- RULER-specialized physical-efficiency audit SHA-256:
  `82b5a3e94ca2c60c10f884aeb7d212df994f1432b96dc7f4a3834e4b2dfb8603`.
