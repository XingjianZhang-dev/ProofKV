# Data

ProofKV is trained and evaluated on public benchmarks. None of them is redistributed here; the records in
`results/` hold question identifiers, reference answers, predictions, and scores, not the contexts.

## Sources

| Dataset | Source | Used for |
|---|---|---|
| HotpotQA (distractor setting) | Hugging Face `hotpotqa/hotpot_qa`, configuration `distractor` | Training (train split); development, generation matrices, packed contexts, and sentence-level link interventions (validation split) |
| 2WikiMultiHopQA | Training set from the Hugging Face mirror named in `scripts/eval_proofcache_selector.py`; `2wikimultihopqa/dev_subsampled.jsonl` and `test_subsampled.jsonl` from the IRCoT processed data | Training; generation matrix (100 development questions); link interventions (test subsample) |
| MuSiQue | `musique/train.jsonl`, `dev_subsampled.jsonl`, and `test_subsampled.jsonl` from the IRCoT processed data; `musique_ans_v1.0_dev.jsonl` from the official MuSiQue v1.0 release for question decompositions | Training; development; the held-out selector test (all 500 questions of `test_subsampled.jsonl`); generation matrices and link interventions |
| HotpotQA, paragraph level | `hotpotqa/test_subsampled.jsonl` from the IRCoT processed data | Paragraph-level link interventions |
| LongBench | `hotpotqa`, `2wikimqa`, `musique`, `qasper`, `multifieldqa_en` task files | Five-task suite |
| LongBench v2 | Official release | Scope diagnostic |
| RULER | Generator in `third_party/NVIDIA_RULER` | Stress test |

The IRCoT processed data is the `processed_data` archive distributed with the
[IRCoT repository](https://github.com/StonyBrookNLP/ircot). The scripts take its location through `--data-file`, and
the follow-up scripts also read the environment variable `PROOFKV_IRCOT` (default `data/ircot/processed_data`).
The held-out MuSiQue file `musique/test_subsampled.jsonl` has SHA-256
`380b4c567a9d8168e0be36e9ccd12953c285edfb77dd2fd0ecef64b6f8e7da31`.

## Question slices

Every slice is a contiguous range of one seeded shuffle (seed 17) of the source file, so slices of the same file
never overlap unless stated. The identifiers of the selected questions are stored with each run (`selected_ids` in
`run_info.json` or `run_manifest.json`).

| Experiment | File | Slice |
|---|---|---|
| Held-out selector test, gap decomposition, iterative retrieval | MuSiQue `test_subsampled.jsonl` | All 500 questions |
| MuSiQue generation matrices and budget sweep | MuSiQue `test_subsampled.jsonl` | Positions 0 to 127 |
| Compressor comparison; equal-byte link interventions | MuSiQue `test_subsampled.jsonl` | Positions 128 to 255 |
| Link interventions, extension | MuSiQue `test_subsampled.jsonl` | Positions 256 to 499 |
| HotpotQA generation matrices, packed contexts, related distractors | HotpotQA validation | Positions 0 to 127 |
| Link interventions, sentence spans | HotpotQA validation | Positions 2048 to 2175 |
| Link interventions, paragraph spans | IRCoT HotpotQA `test_subsampled.jsonl` | First 128 questions outside validation positions 0 to 2047 |
| Link interventions on 2WikiMultiHopQA | IRCoT 2Wiki `test_subsampled.jsonl` | Positions 0 to 127 |
| 2WikiMultiHopQA generation matrix | IRCoT 2Wiki `dev_subsampled.jsonl` | 100 questions |
| Five-task suite | LongBench task files | 46 questions per task, fixed before any run; manifests in `results/paper/records/gpu_results_20260724/` |

Scorer training uses only training splits and generated chains. HotpotQA validation and MuSiQue development
questions were used to choose the checkpoint interpolation and the selection policy; the held-out MuSiQue selector
result is a single evaluation after both were fixed.

## Packed contexts

Packed HotpotQA contexts add paragraphs from other validation questions until the prompt reaches 8k, 16k, or 32k
tokens (packing seed 911 unless a table says otherwise). `scripts/run_hotpot_compact_qa.py` builds them with
`--target-prompt-tokens`; `experiments/followup/related_distractors/run_related_matrix.py` replaces the random
order of added paragraphs with a ranking by TF-IDF similarity to the question or by shared title entities.
