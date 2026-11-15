# Third-party code

The experiments use the following upstream repositories at fixed commits. Their original licenses apply.

| Artifact | Upstream | Commit |
|---|---|---|
| KVPress | `https://github.com/NVIDIA/kvpress.git` | `6d965557a5b9f0201a2301b23c454473dd681d0d` |
| NVIDIA RULER | `https://github.com/NVIDIA/RULER` | `38da79d79519ef87aa46ae804f838e1eab7f86d7` |
| CompressKV | `https://github.com/TUDa-HWAI/CompressKV.git` | `12f920fbecba67e16806a6d8eac0ad389ddcad6f` |
| H2O | `https://github.com/FMInference/H2O.git` | `ac75c2a8a9e76832b2a4139b9363373b56336bfb` |
| HeadKV | `https://github.com/FYYFU/HeadKV.git` | `0862a0955fe82e9ff611d59541918e02c5def625` |
| PyramidKV | `https://github.com/Zefan-Cai/PyramidKV.git` | `94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0` |
| KVzip | `https://github.com/snu-mllab/KVzip.git` | `5d8472975d286e3da6b52178af39d18904330c0d` |

`third_party/` contains the pinned KVPress and NVIDIA RULER source trees (without `.git` metadata and `.gitattributes`), because the
ProofKV runners import them directly, together with their upstream license files. The other baselines are restored
from the table above. The local compatibility changes are in `third_party/patches/`, and the run manifests in
`results/paper/records/` record the hashes of the implementation each baseline run used.

In the RULER tree, `scripts/data/synthetic/json/english_words.json` is stored upstream with Git LFS and is kept
here as its pointer file; fetch it from the upstream repository at the pinned commit before regenerating RULER data.
