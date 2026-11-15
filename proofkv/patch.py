from __future__ import annotations

import inspect
import math
import types
from typing import Any, Callable, Iterable, Optional, Tuple

import torch

from .attention import routed_attention_reference
from .controllers import CompassKVController


def _import_apply_rotary() -> Callable[..., Tuple[torch.Tensor, torch.Tensor]]:
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen2.modeling_qwen2",
    ]
    for name in candidates:
        try:
            module = __import__(name, fromlist=["apply_rotary_pos_emb"])
            return getattr(module, "apply_rotary_pos_emb")
        except Exception:
            continue
    raise ImportError("Could not import apply_rotary_pos_emb from transformers LLaMA/Mistral/Qwen2 modules")


def _apply_rotary(apply_fn: Callable[..., Any], q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    try:
        return apply_fn(q, k, cos, sin)
    except TypeError:
        # Qwen2/Llama variants sometimes expose unsqueeze_dim.
        return apply_fn(q, k, cos, sin, unsqueeze_dim=1)


def _shape_projection(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    bsz, q_len, _ = x.shape
    return x.view(bsz, q_len, num_heads, head_dim).transpose(1, 2).contiguous()


def _get_num_heads(module: torch.nn.Module) -> int:
    for attr in ("num_heads", "num_attention_heads"):
        if hasattr(module, attr):
            return int(getattr(module, attr))
    cfg = getattr(module, "config", None)
    if cfg is not None:
        for attr in ("num_attention_heads", "num_heads"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
    raise AttributeError(f"{module.__class__.__name__} has no num_heads attribute")


def _get_num_kv_heads(module: torch.nn.Module) -> int:
    for attr in ("num_key_value_heads", "num_kv_heads"):
        if hasattr(module, attr):
            return int(getattr(module, attr))
    cfg = getattr(module, "config", None)
    if cfg is not None:
        for attr in ("num_key_value_heads", "num_kv_heads"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
    return _get_num_heads(module)


def _get_head_dim(module: torch.nn.Module) -> int:
    if hasattr(module, "head_dim"):
        return int(getattr(module, "head_dim"))
    cfg = getattr(module, "config", None)
    if cfg is not None and hasattr(cfg, "head_dim"):
        return int(getattr(cfg, "head_dim"))
    hidden_size = int(getattr(module, "hidden_size", getattr(cfg, "hidden_size", 0)))
    if hidden_size <= 0:
        raise AttributeError(f"{module.__class__.__name__} has no hidden_size/head_dim attribute")
    return hidden_size // _get_num_heads(module)


def _build_compass_forward(module: torch.nn.Module, controller: CompassKVController) -> Callable[..., Any]:
    apply_rotary_pos_emb = _import_apply_rotary()
    original_params = set(inspect.signature(module.forward).parameters)
    uses_legacy_cache = "past_key_value" in original_params or "use_cache" in original_params

    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        bsz, q_len, _ = hidden_states.size()
        cache = past_key_value if past_key_value is not None else past_key_values
        # Accuracy experiments route only decode/probe calls by default.  For
        # prefill, delegate to the original optimized HF attention instead of
        # the Python reference implementation; otherwise long prompts would be
        # unreasonably slow.
        if not controller.should_route(q_len):
            common_kwargs = dict(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if "past_key_values" in original_params:
                common_kwargs["past_key_values"] = cache
            else:
                common_kwargs["past_key_value"] = cache
            out = self._proofkv_original_forward(**common_kwargs)  # type: ignore[attr-defined]
            controller.record_call(q_len, q_len, routed=False)
            return out

        num_heads = _get_num_heads(self)
        num_kv_heads = _get_num_kv_heads(self)
        head_dim = _get_head_dim(self)

        query_states = _shape_projection(self.q_proj(hidden_states), num_heads, head_dim)
        key_states = _shape_projection(self.k_proj(hidden_states), num_kv_heads, head_dim)
        value_states = _shape_projection(self.v_proj(hidden_states), num_kv_heads, head_dim)

        if position_embeddings is None:
            if not hasattr(self, "rotary_emb"):
                raise AttributeError("Patched attention module needs rotary_emb or position_embeddings")
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = _apply_rotary(apply_rotary_pos_emb, query_states, key_states, cos, sin)

        if cache is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            try:
                key_states, value_states = cache.update(key_states, value_states, self.layer_idx, cache_kwargs)
            except TypeError:
                key_states, value_states = cache.update(key_states, value_states, self.layer_idx)

        routes = controller.get_query_head_routes(
            int(getattr(self, "layer_idx", 0)),
            num_query_heads=num_heads,
            num_kv_heads=num_kv_heads,
            device=query_states.device,
        )
        attn_output, info = routed_attention_reference(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            routes=routes,
            cfg=controller.cfg,
            attention_mask=attention_mask,
            guidepost_indices=controller.fixed_guidepost_map(
                bsz,
                num_heads,
                key_states.shape[-2],
                query_states.device,
            ),
            output_attentions=output_attentions,
        )
        attn_weights = info.attention_probs if output_attentions else None
        controller.record_call(q_len, key_states.shape[-2], routed=True, routes=routes, kept_counts=info.kept_counts)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, num_heads * head_dim)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        if uses_legacy_cache:
            return attn_output, attn_weights, cache
        return attn_output, attn_weights

    return forward


def is_supported_attention_module(module: torch.nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    has_proj = all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj"))
    supported_name = any(key in name for key in ("llamaattention", "mistralattention", "qwen2attention"))
    return has_proj and supported_name


def patch_model(model: torch.nn.Module, controller: CompassKVController) -> int:
    """Patch supported Hugging Face decoder attention modules in-place.

    Returns the number of attention modules patched.  The original ``forward`` is
    stored as ``_proofkv_original_forward`` so ``unpatch_model`` can restore
    it.  This is a reference implementation: it masks attention after QK logits
    are formed, so accuracy experiments are valid but GPU memory is not reduced
    until a physical cache/kernel path is added.
    """

    patched = 0
    for module in model.modules():
        if not is_supported_attention_module(module):
            continue
        if hasattr(module, "_proofkv_original_forward"):
            continue
        module._proofkv_original_forward = module.forward  # type: ignore[attr-defined]
        module._proofkv_controller = controller  # type: ignore[attr-defined]
        module.forward = types.MethodType(_build_compass_forward(module, controller), module)  # type: ignore[method-assign]
        patched += 1
    if patched == 0:
        raise RuntimeError(
            "No supported attention modules were patched. Supported classes include LlamaAttention, MistralAttention, and Qwen2Attention."
        )
    return patched


def unpatch_model(model: torch.nn.Module) -> int:
    restored = 0
    for module in model.modules():
        original = getattr(module, "_proofkv_original_forward", None)
        if original is None:
            continue
        module.forward = original  # type: ignore[method-assign]
        delattr(module, "_proofkv_original_forward")
        if hasattr(module, "_proofkv_controller"):
            delattr(module, "_proofkv_controller")
        restored += 1
    return restored
