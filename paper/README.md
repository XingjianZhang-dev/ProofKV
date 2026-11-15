# Paper

- `proofkv_eacl2027.pdf`: the camera-ready paper (Findings of EACL 2027).
- `tables/`: every table of the paper as a LaTeX `tabular` fragment. 51 of them are written by the scripts in
  `analysis/` from the records in `results/`; three (`candidate_rules`, `checkpoints`, `training_hparams`) state
  settings of the method and are written by hand.
- `figures/`: every figure of the paper, drawn by `analysis/make_figures.py`.

`python analysis/verify_release.py` regenerates the tables and compares them with this directory byte for byte.
[`docs/REPRODUCE.md`](../docs/REPRODUCE.md) maps each table and figure to its number in the paper.
