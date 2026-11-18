# Scorer checkpoints

Both checkpoints are on [Hugging Face](https://huggingface.co/XingjianZhangDev/ProofKV-scorer). From the repository
root,

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('XingjianZhangDev/ProofKV-scorer', local_dir='checkpoints', allow_patterns=['proofkv_scorer_*/*'])"
python analysis/verify_release.py      # also checks the digests below
```

They are also attached to the [release page](https://github.com/XingjianZhang-dev/ProofKV/releases). Download the
archives into this directory and unpack them:

```bash
cd checkpoints
tar -xzf proofkv_scorer_final.tar.gz
tar -xzf proofkv_scorer_musique_replay_v6.tar.gz
cd .. && python analysis/verify_release.py      # also checks the digests below
```

Each checkpoint is a MiniLM-L12-H384 cross-encoder (`encoder/`, about 133 MB) with the start, end, and link heads
(`heads.pt`), its tokenizer, and the training metadata (`proofcache_scorer.json`). Load one with

```python
from proofkv.proofcache.model import TypedRelationScorer
scorer, tokenizer = TypedRelationScorer.load_checkpoint("checkpoints/proofkv_scorer_final", device="cuda")
```

## Which checkpoint produced which result

| Checkpoint | Training | Results in the paper |
|---|---|---|
| `proofkv_scorer_final` | Interpolation of the mixed-replay (v3b) and HotpotQA-focused (v8) stages with weight 0.90 on the latter | Five-task suite, Qwen2.5-14B transfer, packed contexts and related distractors, equal-byte link interventions, compressor comparison, routing, serving, decoder transfer |
| `proofkv_scorer_musique_replay_v6` | MuSiQue-replay stage (v6) | Held-out MuSiQue selector test and its analyses (gap decomposition, iterative retrieval), MuSiQue generation matrices and budget sweep |

The mixed-replay development checkpoint (v3b) is used only for two development analyses in the appendix (the depth
curve on generated chains and the dense-core sweep); their records are in `results/paper/records/`.

## SHA-256 digests

| File | `proofkv_scorer_final` | `proofkv_scorer_musique_replay_v6` |
|---|---|---|
| `encoder/model.safetensors` | `774e7f692bcea6ae945fe4d0eeb90045f9e9d2371d28d76a4287b348c082cd64` | `c64c04878baac749343b69607c8b9a157024694ad82fcd30de029e63e2135fb4` |
| `heads.pt` | `e95a0075813bf2d87d02327960107f7e6f31b6ff8b546d541a859f8a851e7385` | `f29ed4a274bbd7b462344dd19cc088fa815940fb38d4f8a02b2ab0c45c6185b2` |
| `proofcache_scorer.json` | `815538590808393fd6a119b88a725553efc6929b4c599889bf23c7f65708e41f` | `d6df5faa16394b2600239f5d2b91984b9c87825a734cd77906982a0a72fd86e2` |

Model card: [`docs/MODEL_CARD.md`](../docs/MODEL_CARD.md).
