# Targeted GPU Protocol Amendment — model-window correction

This amendment was written after the 32k vLLM configuration smoke failed and
before any 32k/30k measured batch was generated.

Qwen2.5-7B-Instruct declares a native maximum position length of 32,768. A prompt
near 32k plus 1,024 generated tokens cannot be measured without setting
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and extrapolating RoPE beyond the declared
window. That would create an avoidable correctness risk and is forbidden here.

The frozen scale protocol is therefore corrected as follows:

- 32k packed prompt, output 256, batch 4, five repeats.
- 30k packed prompt, output 1024, batch 4, five repeats.
- `max_model_len=32768` in both cells; no long-window override.
- All other modes, timing boundaries, scorer settings, route-identity checks,
  warmups, and integrity gates remain unchanged.

The failed 32k+1024 configuration log remains in the audit tree. It contains no
measured batches and is not used in any aggregate.
