#!/usr/bin/env python3
"""
Diagnose Gemma4 thinking loop in graph mode.

Tests the same prompt against a running server to check whether
the model generates <channel|> (end-of-thinking token) properly.
Also checks if reasoning_content is present in the response.

Usage:
  python scripts/diagnose_thinking_loop.py --port 8831

The script reports:
  1. Whether <channel|> appeared in the raw output
  2. Whether reasoning was extracted by the parser
  3. Whether thinking actually happened (vs. being silently disabled)
"""

import argparse
import json
import time
from openai import OpenAI

# The same GPQA physics question used in test_eager_vs_graph.py
PROMPT = {
    "id": "gpqa_physics",
    "messages": [
        {"role": "user", "content": (
            "Among the following exoplanets, which one has the highest density?\n\n"
            "a) An Earth-mass and Earth-radius planet.\n"
            "b) A planet with 2 Earth masses and a density of approximately 5.5 g/cm^3.\n"
            "c) A planet with the same composition as Earth but 5 times more massive than Earth.\n"
            "d) A planet with the same composition as Earth but half the mass of Earth.\n\n"
            "Think step by step and explain your reasoning."
        )}
    ],
    "max_tokens": 2048,
}


def test_config(port: int, config_name: str, extra_body: dict, model_name: str = "gemma-4-26b-a4b-it"):
    """Send a single request and analyze the output."""
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")

    t0 = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=PROMPT["messages"],
        max_tokens=PROMPT["max_tokens"],
        temperature=0,
        extra_body=extra_body,
    )
    elapsed = time.time() - t0

    choice = response.choices[0]
    content = choice.message.content or ""
    reasoning = (getattr(choice.message, "reasoning", None)
                 or getattr(choice.message, "reasoning_content", None)
                 or "")
    finish = choice.finish_reason
    tokens = response.usage.completion_tokens if response.usage else 0

    # Check for thinking markers in content+reasoning
    has_think_start = "<|channel>" in content or "<|channel>" in reasoning
    has_think_end = "<channel|>" in content or "<channel|>" in reasoning
    has_reasoning = len(reasoning) > 0  # reasoning extracted by vLLM parser

    thinking_works = has_reasoning
    thinking_loops = has_think_start and not has_think_end and not has_reasoning
    thinking_disabled = not has_think_start and not has_reasoning

    status = "THINKING_OK" if thinking_works else \
             "THINKING_LOOP" if thinking_loops else \
             "THINKING_OFF" if thinking_disabled else \
             "UNCLEAR"

    print(f"  [{config_name}]")
    print(f"    tokens={tokens}  finish={finish}  elapsed={elapsed:.1f}s")
    print(f"    content_len={len(content)}  reasoning_len={len(reasoning)}")
    print(f"    has_think_start={has_think_start}  has_think_end={has_think_end}")
    print(f"    has_reasoning={has_reasoning}")
    print(f"    status={status}")
    if content:
        print(f"    content[:200]: {content[:200]}")
    if reasoning:
        print(f"    reasoning[:200]: {reasoning[:200]}")
    print()

    return {
        "config": config_name,
        "extra_body": extra_body,
        "tokens": tokens,
        "finish": finish,
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "has_think_start": has_think_start,
        "has_think_end": has_think_end,
        "has_reasoning": has_reasoning,
        "status": status,
        "elapsed": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Gemma4 thinking loop in graph mode")
    parser.add_argument("--port", type=int, required=True,
                        help="vLLM server port")
    parser.add_argument("--model", type=str,
                        default="gemma-4-26b-a4b-it",
                        help="Model name")
    args = parser.parse_args()

    configs = [
        # Config 1: enable_thinking=True (expected to loop in FULL_DECODE_ONLY)
        ("enable_thinking=True", {
            "chat_template_kwargs": {"enable_thinking": True}
        }),
        # Config 2: thinking: enabled (what previous session said "works")
        ("thinking:{type:enabled}", {
            "thinking": {"type": "enabled"}
        }),
        # Config 3: Both (verify interaction)
        ("both", {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking": {"type": "enabled"},
        }),
        # Config 4: No thinking at all (baseline)
        ("no_thinking", {}),
    ]

    print(f"Testing against port {args.port}, model={args.model}")
    print(f"Server timeout: 120s per request")
    print("=" * 60)

    results = []
    for name, extra_body in configs:
        try:
            result = test_config(args.port, name, extra_body, args.model)
            results.append(result)
        except Exception as e:
            print(f"  [{name}] ERROR: {e}\n")
            results.append({"config": name, "error": str(e)})

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print(f"{'Config':<30} {'Tokens':<8} {'Finish':<8} {'Status':<15}")
    print("-" * 60)
    for r in results:
        if "error" in r:
            print(f"{r['config']:<30} {'ERROR':<8} {r['error'][:40]}")
        else:
            print(f"{r['config']:<30} {r['tokens']:<8} {r['finish']:<8} {r['status']:<15}")

    print()
    print("KEY:")
    print("  THINKING_OK   = reasoning extracted, <channel|> present in output")
    print("  THINKING_LOOP = thinking started but never ended (<channel|> missing)")
    print("  THINKING_OFF  = no thinking markers at all (thinking disabled)")
    print()
    print("RECOMMENDATION:")
    print("  If enable_thinking=True gives THINKING_LOOP but thinking:{type:enabled}")
    print("  gives THINKING_OFF with content, then 'thinking' from extra_body is")
    print("  NOT enabling thinking for Gemma4 (template doesn't check it).")
    print("  The model is giving direct answers without CoT reasoning.")
    print()
    print("  To fix: need to debug why graph mode suppresses <channel|> logit.")
    print("  Try disabling fuse_qknorm_rope via ascend_compilation_config.")


if __name__ == "__main__":
    main()
