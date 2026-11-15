# Model card: ProofKV evidence scorer

## Description

The ProofKV scorer is a 12-layer, 384-hidden cross-encoder initialized from `microsoft/MiniLM-L12-H384-uncased`
(about 33M parameters) with three heads: a start head and an end head that score a span against the question, and a
nine-way link head that scores an ordered pair of spans (eight link labels and `none`). ProofKV composes evidence
chains from these three scores; see Section 4 and Appendix B of the paper.

Two checkpoints are released (`checkpoints/README.md` lists digests and the results each one produced):

- **`proofkv_scorer_final`** is a linear interpolation of all parameters of two training stages, the mixed-replay
  stage (v3b) and the HotpotQA-focused stage (v8), with weight 0.90 on the latter. The interpolation weight and the
  selection policy were chosen on HotpotQA validation and MuSiQue development questions.
- **`proofkv_scorer_musique_replay_v6`** is the MuSiQue-replay stage. It was evaluated once on the 500 held-out
  MuSiQue test questions; neither the checkpoint nor its policy changed afterwards.

## Training

Training uses the training splits of HotpotQA, 2WikiMultiHopQA, and MuSiQue and generated evidence chains, in a
sequence of replay stages (Appendix C). Every stage uses AdamW with learning rate 2e-5, weight decay 0.01, batch
size 48, gradient clipping 1.0, and maximum input length 320. The loss of each head adds a positive-evidence term
(weight 0.5), a budget term (0.05), and a calibration term (0.2) to its classification loss.

The last stage behind the final checkpoint saw 730,723 training decisions from 8,000 HotpotQA, 4,000
2WikiMultiHopQA, 4,000 MuSiQue, and 4,000 generated examples (one epoch, 3,807 steps, seed 307). The MuSiQue-replay
stage saw 922,211 decisions from 4,000 HotpotQA, 4,000 2WikiMultiHopQA, 8,000 MuSiQue, and 4,000 generated examples
(one epoch, 9,607 steps, seed 101). Negatives include wrong start spans with high lexical overlap, wrong or missing
middle spans, reversed links, and shortcuts across chains; the counts are in each `proofcache_scorer.json`.

## Intended use

Reproducing the paper, and research on evidence selection and KV-memory construction for English question answering
over documents whose answer is determined by a short chain of spans.

The scorer selects what a decoder gets to read. A wrong selection removes evidence, so applications where an error
is costly should show the selected evidence and keep a route back to the full context, as the margin router in the
paper does.

## Evaluation

The paper evaluates the checkpoints on English multi-hop and document QA (HotpotQA, 2WikiMultiHopQA, MuSiQue, and
the LongBench versions of these plus Qasper and MultiFieldQA-en) with 7B to 14B open decoders. Other languages,
dialogue, code, summarization, and free-form generation are outside this evaluation.

## Data and privacy

No private or user data and no new human annotation were collected. The scorer inherits the properties of MiniLM
and of the public QA benchmarks it was trained on.

## License

MIT, as for the rest of this repository. `microsoft/MiniLM-L12-H384-uncased` is also released under the MIT License.
