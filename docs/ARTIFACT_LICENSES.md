# Licenses of the artifacts used and released

This page records the licenses visible at the pinned upstream artifacts. Upstream terms remain controlling: this
repository does not relicense public datasets, public model weights, or third-party source code.

## Released with this repository

| Artifact | License |
|---|---|
| ProofKV code, scripts, documentation, scorer checkpoints, result records, and analysis outputs | MIT License (`LICENSE`) |

## Public datasets and benchmark generators

| Artifact | Upstream license | In this repository |
|---|---|---|
| HotpotQA dataset | CC BY-SA 4.0 | Not redistributed |
| HotpotQA code | Apache License 2.0 | Referenced only |
| 2WikiMultiHopQA | Apache License 2.0 | Not redistributed |
| MuSiQue | CC BY 4.0 | Not redistributed |
| LongBench and LongBench v2 | MIT License | Row identifiers and model outputs only |
| NVIDIA RULER generator | Apache License 2.0 | Pinned source with its upstream license |

The per-example records hold row identifiers, reference answers, predictions, scores, token counts, timings, and
selection diagnostics. They do not hold the benchmark contexts.

## Public models

| Artifact | License | In this repository |
|---|---|---|
| Qwen2.5-7B-Instruct | Apache License 2.0 | Revision and file digests |
| Qwen2.5-14B-Instruct | Apache License 2.0 | Revision and file digests |
| Qwen3-8B | Apache License 2.0 | Revision and file digests |
| Mistral-7B-Instruct-v0.2 | Apache License 2.0 | Revision and file digests |
| Mistral-7B-Instruct-v0.3 | Apache License 2.0 | Revision and file digests |
| LLMLingua-2 XLM-RoBERTa (MeetingBank) | MIT License | Revision and file digests |
| MiniLM-L12-H384-uncased | MIT License | Initialization of the released scorer |

Exact revisions, file sizes, SHA-256 digests, and download commands are in `models/public_models.json`.

## Baseline implementations

| Artifact | License at the pinned revision | In this repository |
|---|---|---|
| NVIDIA KVPress | Apache License 2.0 | Pinned source with its upstream license |
| PyramidKV | MIT License | Pinned commit only |
| KVzip | MIT License | Pinned commit only |
| H2O | No license file at the pinned revision | Pinned commit only |
| HeadKV | No license file at the pinned revision | Pinned commit only |
| CompressKV | No license file at the pinned revision | Pinned commit only |

For repositories without a license file at the pinned commit, this repository distributes no upstream source: it
records the commit and contains only the runner and compatibility code written for ProofKV.
