from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch

from proofkv.config import CompassConfig, Route
from proofkv.controllers import CompassKVController
from proofkv.patch import patch_model
from proofkv.routing import RouteTable, make_uniform_route_table


@dataclass
class LoadedLLM:
    model: Any
    tokenizer: Any
    controller: Optional[CompassKVController]


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name}")


def load_model_tokenizer(
    model_name_or_path: str,
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    device_map: str = "auto",
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install proofkv[llm] or requirements.txt to run LLM experiments") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype_from_name(dtype),
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def prepare_model_with_method(
    model: Any,
    method: str,
    cfg: CompassConfig,
    route_table: Optional[RouteTable] = None,
) -> Optional[CompassKVController]:
    method = method.lower()
    if method == "full":
        return None
    # Uniform baselines reuse the same patch, only changing routes/selectors.
    if route_table is None:
        # Infer layers/heads from config if possible.
        layers = getattr(model.config, "num_hidden_layers")
        heads = getattr(model.config, "num_attention_heads")
        if method in {"streaming", "stream"}:
            route_table = make_uniform_route_table(layers, heads, Route.STREAM, {"method": method})
        elif method in {"navigation", "nav", "compass", "compasskv", "uniform_nav", "random_nav", "extended_recent"}:
            route_table = make_uniform_route_table(layers, heads, Route.NAV, {"method": method})
        else:
            raise ValueError(f"Unknown method {method}")
    cfg = CompassConfig.from_dict(cfg.to_dict())
    if method == "uniform_nav":
        cfg.selector = "uniform"
    elif method == "random_nav":
        cfg.selector = "random"
    elif method == "extended_recent":
        cfg.selector = "extended_recent"
    controller = CompassKVController(cfg, route_table)
    patch_model(model, controller)
    return controller


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-5)
        out = model.generate(**gen_kwargs)
    gen = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()
