#!/usr/bin/env python3
"""
Quick test: compare chat_template_kwargs vs thinking in extra_body
for Gemma4 thinking mode.

Usage:
  python scripts/test_thinking_fix.py --port 8026
"""

import argparse, json, time
from openai import OpenAI

PROMPTS = [
    # Should trigger deep thinking
    {
        "id": "gpqa",
        "messages": [{"role": "user", "content": (
            "Among the following exoplanets, which one has the highest density?\n\n"
            "a) An Earth-mass and Earth-radius planet.\n"
            "b) A planet with 2 Earth masses and a density of approximately 5.5 g/cm^3.\n"
            "c) A planet with the same composition as Earth but 5 times more massive than Earth.\n"
            "d) A planet with the same composition as Earth but half the mass of Earth.\n\n"
            "Think step by step and explain your reasoning."
        )}],
        "max_tokens": 2048,
    },
    # Simple question - minimal thinking needed
    {
        "id": "simple",
        "messages": [{"role": "user", "content": "What is 15 + 27?"}],
        "max_tokens": 256,
    },
]

def test_config(client, model, name, extra_body):
    results = []
    for prompt in PROMPTS:
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=prompt["messages"],
                max_tokens=prompt["max_tokens"],
                temperature=0,
                extra_body=extra_body,
            )
            elapsed = time.time() - t0
            choice = resp.choices[0]
            msg = choice.message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning", None) or \
                        getattr(msg, "reasoning_content", None) or ""
            usage = resp.usage

            has_start = "<|channel>" in content or "<|channel>" in reasoning
            has_end = "<channel|>" in content or "<channel|>" in reasoning
            has_reasoning = len(reasoning) > 0

            result = {
                "id": prompt["id"],
                "config": name,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "finish": choice.finish_reason,
                "content_len": len(content),
                "reasoning_len": len(reasoning),
                "has_think_start": has_start,
                "has_think_end": has_end,
                "has_reasoning": has_reasoning,
                "elapsed": round(elapsed, 1),
                "content_preview": content[:200] if content else "",
                "reasoning_preview": reasoning[:200] if reasoning else "",
            }
            results.append(result)

            status = "OK" if (has_reasoning and has_end) else \
                     "LOOP?" if (has_start and not has_end) else \
                     "NO_THINK" if (not has_start and not has_reasoning) else \
                     "MIXED"
            print(f"  [{name}][{prompt['id']}] tokens={result['completion_tokens']} "
                  f"finish={result['finish']} think_end={has_end} reasoning={has_reasoning} "
                  f"status={status} time={elapsed:.1f}s")
        except Exception as e:
            results.append({"id": prompt["id"], "config": name, "error": str(e)})
            print(f"  [{name}][{prompt['id']}] ERROR: {e}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=str, default="gemma-4-26b-a4b-it")
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="not-needed")

    configs = [
        ("enable_thinking", {"chat_template_kwargs": {"enable_thinking": True}}),
        ("thinking_enabled", {"thinking": {"type": "enabled"}}),
        ("baseline", {}),
    ]

    print("=" * 70)
    print(f"Testing thinking mode behavior on port {args.port}")
    print("=" * 70)

    all_results = {}
    for name, extra in configs:
        print(f"\n--- {name} ---")
        all_results[name] = test_config(client, args.model, name, extra)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Config':<22} {'Q':<8} {'Tokens':<8} {'Finish':<8} {'ThinkEnd':<10} {'HasReason':<10} {'Time':<8}")
    print("-" * 70)
    for name, results in all_results.items():
        for r in results:
            if "error" in r:
                print(f"{name:<22} {r['id']:<8} ERROR: {r['error'][:30]}")
            else:
                print(f"{name:<22} {r['id']:<8} {r['completion_tokens']:<8} "
                      f"{r['finish']:<8} {str(r['has_think_end']):<10} "
                      f"{str(r['has_reasoning']):<10} {r['elapsed']:<8}")

    print()
    print("Takeaway:")
    print("  - If 'thinking_enabled' gives no thinking markers + normal stop:")
    print("    thinking is being silently DISABLED")
    print("  - If 'enable_thinking' gives many tokens + length finish + no thinking end:")
    print("    thinking mode is LOOPING (model can't generate <channel|>)")

if __name__ == "__main__":
    main()
