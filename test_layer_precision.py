#!/usr/bin/env python3
"""Hook into model layers to log numerical stats during a forward pass.
This reveals which layer type introduces the most error."""
import torch
import torch_npu
import json
import os

from vllm import LLM, SamplingParams
from vllm.config import VllmConfig, set_current_vllm_config

# We'll add hooks to log per-layer statistics
hook_stats = {}

def make_hook(name):
    def hook(module, inp, outp):
        if isinstance(outp, torch.Tensor):
            s = {
                'mean': outp.float().mean().item(),
                'std': outp.float().std().item(),
                'max': outp.float().max().item(),
                'min': outp.float().min().item(),
                'has_nan': torch.isnan(outp).any().item(),
                'has_inf': torch.isinf(outp).any().item(),
                'shape': list(outp.shape),
            }
            hook_stats[name] = s
            # Print immediately for layer 0, 1, 29, 30, 58, 59
            layer_num = None
            if '.layers.' in name:
                try:
                    layer_num = int(name.split('.layers.')[1].split('.')[0])
                except:
                    pass
            if layer_num in (0, 1, 29, 30, 58, 59) or layer_num is None:
                print(f"  [{name}] mean={s['mean']:.6f} std={s['std']:.6f} "
                      f"max={s['max']:.4f} min={s['min']:.4f} "
                      f"NaN={s['has_nan']} Inf={s['has_inf']} shape={s['shape']}")
        return outp
    return hook


if __name__ == "__main__":
    model_path = "/data/gemma4/gemma-4-31b-it"

    print("Loading model...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        enforce_eager=True,  # No CUDA graphs for clean testing
        tensor_parallel_size=1,
    )

    # Hook key attention and RoPE layers
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    for name, module in model.named_modules():
        if 'self_attn' in name or 'rotary_emb' in name:
            module.register_forward_hook(make_hook(name))

    print("\nRunning forward pass with thinking...")
    prompt = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nWhat is 2+2? Explain step by step.<turn|>\n<|turn>model\n"

    outputs = llm.generate(
        [prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=64,
            skip_special_tokens=False,
        ),
    )

    print(f"\nOutput: {outputs[0].outputs[0].text[:200]}")

    # Show stats for first and last few layers
    print("\n=== Layer attention stats ===")
    for layer_num in range(60):
        for suffix in ['self_attn', 'self_attn.rotary_emb']:
            key = f"model.layers.{layer_num}.{suffix}"
            if key in hook_stats:
                s = hook_stats[key]
                print(f"  L{layer_num:02d} {suffix}: mean={s['mean']:.6f} std={s['std']:.6f} "
                      f"NaN={s['has_nan']} Inf={s['has_inf']}")
