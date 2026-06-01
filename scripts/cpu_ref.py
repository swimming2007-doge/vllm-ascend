#!/usr/bin/env python3
"""CPU reference forward pass for Gemma4 using pure HuggingFace transformers.

Saves per-layer hidden states, final logits, and token-101 analysis so they
can be compared against an Ascend NPU run.

Usage:
    python scripts/cpu_ref.py \
        --model /data/gemma4/gemma-4-31b-it \
        --prompt "What is 15 + 27?" \
        --output /tmp/cpu_ref
"""

import argparse
import json
import os
import sys
import time

import torch

torch.set_grad_enabled(False)


def load_model(model_path: str):
    """Load Gemma4 on CPU via HF transformers, return (text_model, lm_head, tokenizer, config)."""
    from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration

    print(f"Loading config from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model on CPU (bf16, low_cpu_mem_usage=True) ...")
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Gemma4ForConditionalGeneration -> Gemma4Model (multimodal) -> Gemma4TextModel (language)
    text_model = model.model.language_model  # Gemma4TextModel
    lm_head = model.lm_head
    text_config = text_model.config  # Gemma4TextConfig

    print(f"  Hidden size: {text_config.hidden_size}")
    print(f"  Num layers: {text_config.num_hidden_layers}")
    print(f"  Vocab size: {text_config.vocab_size}")
    print(f"  Final softcap: {text_config.final_logit_softcapping}")

    return text_model, lm_head, tokenizer, text_config


def run_forward(text_model, lm_head, tokenizer, text_config, prompt: str, output_dir: str):
    """Run a forward pass and save all intermediate tensors."""
    os.makedirs(output_dir, exist_ok=True)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"\nPrompt: {prompt!r}")
    print(f"  Tokenized length: {seq_len}")
    print(f"  Token IDs: {input_ids.tolist()}")

    # Save metadata
    meta = {
        "prompt": prompt,
        "seq_len": seq_len,
        "token_ids": input_ids.tolist(),
        "hidden_size": text_config.hidden_size,
        "num_layers": text_config.num_hidden_layers,
        "vocab_size": text_config.vocab_size,
        "softcap": text_config.final_logit_softcapping,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Forward pass
    print("Running forward pass on CPU ...")
    t0 = time.time()
    with torch.inference_mode():
        outputs = text_model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    print(f"  Forward done in {time.time() - t0:.1f}s")

    # outputs.hidden_states is a tuple: (embed_out, L0_out, L1_out, ..., L59_out, final_norm_out)
    # Length should be num_layers + 2 = 62 (embed + 60 layers + final norm)
    hidden_states = outputs.hidden_states
    print(f"  Hidden states tuple length: {len(hidden_states)}")

    # Save per-layer hidden states
    print("Saving per-layer hidden states ...")
    stats_all = {}
    for i, hs in enumerate(hidden_states):
        if i == 0:
            name = "embed"
        elif i <= text_config.num_hidden_layers:
            name = f"layer_{i - 1:03d}"
        else:
            name = "final_norm"

        tensor = hs.cpu()
        torch.save(tensor, os.path.join(output_dir, f"{name}_hidden.pt"))
        f32 = tensor.float()
        stats_all[name] = {
            "shape": list(tensor.shape),
            "mean": float(f32.mean()),
            "std": float(f32.std()),
            "min": float(f32.min()),
            "max": float(f32.max()),
        }

    # Compute logits from the final hidden state (last token only)
    print("Computing logits ...")
    final_hidden = hidden_states[-1][:, -1:, :]  # [1, 1, hidden_size]
    raw_logits = lm_head(final_hidden)  # [1, 1, vocab_size]

    # Apply final logit softcapping (matching vLLM LogitsProcessor)
    sc = text_config.final_logit_softcapping
    logits = torch.tanh(raw_logits / sc) * sc

    # Save
    torch.save(final_hidden.cpu(), os.path.join(output_dir, "hs_final.pt"))
    torch.save(logits.cpu(), os.path.join(output_dir, "logits.pt"))

    # Logits stats
    l = logits[0, 0].float()
    top5_vals, top5_ids = l.topk(5)
    tok101 = float(l[101])
    rank101 = int((l > tok101).sum().item())

    logit_stats = {
        "token_101_logit": tok101,
        "token_101_rank": rank101,
        "top5_ids": top5_ids.tolist(),
        "top5_vals": [round(float(v), 4) for v in top5_vals.tolist()],
        "logit_mean": float(l.mean()),
        "logit_std": float(l.std()),
    }
    stats_all["logits"] = logit_stats

    # Also save the last-token hidden state stats
    f32_hs = final_hidden.float()
    stats_all["hs_final"] = {
        "shape": list(final_hidden.shape),
        "mean": float(f32_hs.mean()),
        "std": float(f32_hs.std()),
        "min": float(f32_hs.min()),
        "max": float(f32_hs.max()),
    }

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats_all, f, indent=2)

    print(f"\n=== Results ===")
    print(f"Token 101 (<channel|>) logit: {tok101:.4f}  (rank {rank101}/{text_config.vocab_size})")
    print(f"Top 5: {list(zip(top5_ids.tolist(), [round(float(v), 2) for v in top5_vals.tolist()]))}")
    print(f"\nAll data saved to {output_dir}/")
    print(f"Files: {sorted(os.listdir(output_dir))}")


def main():
    parser = argparse.ArgumentParser(description="CPU reference forward pass for Gemma4")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", required=True, help="Input prompt text")
    parser.add_argument("--output", required=True, help="Output directory for saved tensors")
    args = parser.parse_args()

    text_model, lm_head, tokenizer, text_config = load_model(args.model)
    run_forward(text_model, lm_head, tokenizer, text_config, args.prompt, args.output)


if __name__ == "__main__":
    main()
