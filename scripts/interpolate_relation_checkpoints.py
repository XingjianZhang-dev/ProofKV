#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear model soup for compatible typed-relation checkpoints.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--right-weight", type=float, required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def interpolate(left: Any, right: Any, alpha: float) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError("checkpoint state keys differ")
        return {key: interpolate(left[key], right[key], alpha) for key in left}
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("checkpoint tensor shape/dtype mismatch")
        if left.is_floating_point():
            return left.mul(1.0 - alpha).add(right, alpha=alpha)
        return right.clone()
    if left != right:
        raise ValueError("non-tensor checkpoint values differ")
    return right


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.right_weight <= 1.0:
        raise ValueError("right-weight must lie in [0, 1]")
    left, right, out = Path(args.left), Path(args.right), Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    for directory in ("tokenizer", "encoder"):
        shutil.copytree(right / directory, out / directory)
    encoder_file = "model.safetensors"
    left_encoder = load_file(str(left / "encoder" / encoder_file), device="cpu")
    right_encoder = load_file(str(right / "encoder" / encoder_file), device="cpu")
    mixed_encoder = interpolate(left_encoder, right_encoder, args.right_weight)
    (out / "encoder" / encoder_file).unlink()
    save_file(mixed_encoder, str(out / "encoder" / encoder_file))

    left_heads = torch.load(left / "heads.pt", map_location="cpu", weights_only=True)
    right_heads = torch.load(right / "heads.pt", map_location="cpu", weights_only=True)
    torch.save(interpolate(left_heads, right_heads, args.right_weight), out / "heads.pt")

    metadata = json.loads((right / "proofcache_scorer.json").read_text())
    metadata["interpolation"] = {
        "left": str(left),
        "right": str(right),
        "right_weight": args.right_weight,
    }
    (out / "proofcache_scorer.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"out": str(out), "right_weight": args.right_weight}))


if __name__ == "__main__":
    main()
