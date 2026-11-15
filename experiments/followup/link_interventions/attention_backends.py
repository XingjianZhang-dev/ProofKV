#!/usr/bin/env python
"""Attention-backend check of the MuSiQue equal-byte link interventions.

Re-decodes the six frozen arms' exact routes (the routes.jsonl written by run_link_interventions.py) on the
128-example topology slice through the production decode path (production_compact_decode) under each
attention backend, then reports per-backend absolute F1 and the paired differences (10k bootstrap CI,
100k one-sided sign-flip p) that carry the claim.

The default SDPA kernel is IDENTIFIED with the torch profiler (CUDA kernel names of one prefill + one
decode step), not assumed to be flash.
"""
from __future__ import annotations
import argparse, json, random, statistics, sys, time
from pathlib import Path
IMPL = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(IMPL / "scripts"))
import torch
from datasets import load_dataset
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.profiler import profile, ProfilerActivity
from proofkv.eval.generation import load_model_tokenizer
from run_hotpot_compact_qa import answer_scores, extract_short_answer, make_prompt, normalize_ircot_source, production_compact_decode
from run_natural_topology_causality import METHODS, bootstrap_delta, stable_seed, file_sha256
from run_physical_kv_replay import _release_cuda

MODEL = __import__("os").environ.get("PROOFKV_DECODER", "Qwen/Qwen2.5-7B-Instruct")
DATA = __import__("os").path.join(__import__("os").environ.get("PROOFKV_IRCOT", "data/ircot/processed_data"), "musique/test_subsampled.jsonl")
CONFIGS = [
    ("sdpa_default", "sdpa", None),
    ("sdpa_flash", "sdpa", SDPBackend.FLASH_ATTENTION),
    ("sdpa_efficient", "sdpa", SDPBackend.EFFICIENT_ATTENTION),
    ("sdpa_cudnn", "sdpa", SDPBackend.CUDNN_ATTENTION),
    ("sdpa_math", "sdpa", SDPBackend.MATH),
    ("eager", "eager", None),
]


def sign_flip(values, samples, seed):
    observed = statistics.fmean(values); rng = random.Random(seed); one = two = 0
    for _ in range(samples):
        v = statistics.fmean(x if rng.random() < 0.5 else -x for x in values)
        one += v >= observed - 1e-15; two += abs(v) >= abs(observed) - 1e-15
    return {"p_one_sided_greater": (one + 1) / (samples + 1), "p_two_sided": (two + 1) / (samples + 1)}


def kernel_names(model, tokenizer, input_ids, retained, ctx_factory):
    names = set()
    with ctx_factory():
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            text, cache, _, _, _ = production_compact_decode(model, tokenizer, input_ids, retained, int(input_ids.shape[1]), 2, 512)
            del cache
    for evt in prof.key_averages():
        n = evt.key.lower()
        if any(tag in n for tag in ("flash", "fmha", "mem_eff", "cutlass", "cudnn", "sdp", "attention", "softmax")):
            names.add(evt.key[:160])
    _release_cuda()
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--routes", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=10000); ap.add_argument("--permutation-samples", type=int, default=100000)
    ap.add_argument("--intervention-seed", type=int, default=42823318)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    routes = {}
    for line in open(args.routes):
        r = json.loads(line); routes[r["id"]] = r["routes"]
    ids = list(routes)
    dataset = load_dataset("json", data_files={"v": DATA}, split="v")
    by_id = {}
    for i in range(len(dataset)):
        src = normalize_ircot_source(dict(dataset[i])); by_id[str(src["id"])] = src
    manifest = {"experiment_id": "link_interventions/attention_backends", "routes_file": args.routes, "routes_sha256": file_sha256(Path(args.routes)),
                "n_examples": len(ids), "methods": list(METHODS), "model": MODEL, "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0), "configs": [c[0] for c in CONFIGS],
                "sdp_flags": {"flash": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                              "math": torch.backends.cuda.math_sdp_enabled(), "cudnn": torch.backends.cuda.cudnn_sdp_enabled()},
                "decode_path": "production_compact_decode (identical to the runs), greedy, 24 new tokens, chunk 512",
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {}; rows_by_config = {}
    for name, attn, backend in CONFIGS:
        model, tokenizer = load_model_tokenizer(MODEL, dtype="bfloat16", attn_implementation=attn)
        device = next(model.parameters()).device
        ctx_factory = (lambda b=backend: sdpa_kernel(b)) if backend is not None else (lambda: torch.no_grad())
        entry = {"attn_implementation": attn, "forced_backend": str(backend) if backend else None}
        # kernel identification on the first example's proof_core route
        src0 = by_id[ids[0]]; p0, _, _ = make_prompt(src0, tokenizer)
        ii0 = tokenizer(p0, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        try:
            entry["cuda_kernels_observed"] = kernel_names(model, tokenizer, ii0, routes[ids[0]]["proof_core"], ctx_factory)
        except Exception as exc:  # backend not eligible for this shape
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            summary[name] = entry; print(name, "UNAVAILABLE:", entry["error"], flush=True)
            del model; _release_cuda(); torch.cuda.empty_cache(); continue
        rows = []; t0 = time.time()
        for k, sid in enumerate(ids):
            src = by_id[sid]; prompt, _, _ = make_prompt(src, tokenizer)
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            for arm in METHODS:
                with ctx_factory():
                    text, cache, _, _, _ = production_compact_decode(model, tokenizer, input_ids, routes[sid][arm], int(input_ids.shape[1]), 24, 512)
                del cache; _release_cuda()
                short = extract_short_answer(text); em, f1, _ = answer_scores(short, src["answer"])
                rows.append({"id": sid, "method": arm, "config": name, "prediction": text, "short_prediction": short, "answer_f1": f1, "answer_em": em})
            if (k + 1) % 32 == 0: print(f"  {name} {k+1}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
        with open(out / f"rows_{name}.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        rows_by_config[name] = {(r["id"], r["method"]): r for r in rows}
        entry["f1_by_arm"] = {arm: 100 * statistics.fmean(r["answer_f1"] for r in rows if r["method"] == arm) for arm in METHODS}
        entry["em_by_arm"] = {arm: 100 * statistics.fmean(r["answer_em"] for r in rows if r["method"] == arm) for arm in METHODS}
        entry["seconds"] = time.time() - t0
        comps = {}
        for opp in ("relevance_same_bytes", "edge_shuffled", "random_edges", "broken_middle"):
            left = [rows_by_config[name][(sid, "proof_core")]["answer_f1"] for sid in ids]
            right = [rows_by_config[name][(sid, opp)]["answer_f1"] for sid in ids]
            c = bootstrap_delta(left, right, args.bootstrap_samples, stable_seed(opp, args.intervention_seed))
            c.update(sign_flip([a - b for a, b in zip(left, right)], args.permutation_samples, 82000 + sum(map(ord, opp))))
            comps[f"proof_core_vs_{opp}"] = c
        left = [rows_by_config[name][(sid, "oracle_proof")]["answer_f1"] for sid in ids]; right = [rows_by_config[name][(sid, "proof_core")]["answer_f1"] for sid in ids]
        c = bootstrap_delta(left, right, args.bootstrap_samples, args.intervention_seed + 99)
        c.update(sign_flip([a - b for a, b in zip(left, right)], args.permutation_samples, 82099)); comps["oracle_proof_vs_proof_core"] = c
        entry["paired_f1"] = comps
        if "sdpa_default" in rows_by_config and name != "sdpa_default":
            base = rows_by_config["sdpa_default"]
            entry["prediction_flips_vs_sdpa_default_by_arm"] = {arm: sum(base[(sid, arm)]["prediction"] != rows_by_config[name][(sid, arm)]["prediction"] for sid in ids) for arm in METHODS}
        summary[name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k in ("f1_by_arm", "prediction_flips_vs_sdpa_default_by_arm", "cuda_kernels_observed")}, indent=1), flush=True)
        del model; _release_cuda(); torch.cuda.empty_cache()
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps({"protocol": manifest, "backends": summary}, indent=2) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
