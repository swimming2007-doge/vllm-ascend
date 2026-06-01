#!/usr/bin/env python3
"""
Test if logit_bias on <channel|> (token 101) fixes the thinking loop.

Usage:
  python scripts/test_logit_bias.py --port 8026
"""

import argparse, time
from openai import OpenAI

PROMPT = {
    "messages": [{"role": "user", "content": (
        "Among the following exoplanets, which one has the highest density?\n\n"
        "a) An Earth-mass and Earth-radius planet.\n"
        "b) A planet with 2 Earth masses and a density of approximately 5.5 g/cm^3.\n"
        "c) A planet with the same composition as Earth but 5 times more massive than Earth.\n"
        "d) A planet with the same composition as Earth but half the mass of Earth.\n\n"
        "Think step by step and explain your reasoning."
    )}],
    "max_tokens": 2048,
}


def test(client, model, name, extra_body):
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model, messages=PROMPT["messages"],
            max_tokens=PROMPT["max_tokens"], temperature=0,
            extra_body=extra_body,
        )
        elapsed = time.time() - t0
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        usage = resp.usage

        has_end = "<channel|>" in content or "<channel|>" in reasoning
        tokens = usage.completion_tokens if usage else 0

        print(f"[{name}] tokens={tokens} finish={choice.finish_reason} "
              f"has_<channel|>={has_end} content_len={len(content)} "
              f"reasoning_len={len(reasoning)} time={elapsed:.1f}s")
        if content:
            print(f"  content[:300]: {content[:300]}")
        if reasoning:
            print(f"  reasoning[:200]: {reasoning[:200]}")
        print()
        return {"name": name, "tokens": tokens, "finish": choice.finish_reason,
                "has_end": has_end, "content_len": len(content),
                "reasoning_len": len(reasoning)}
    except Exception as e:
        print(f"[{name}] ERROR: {e}\n")
        return {"name": name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=str, default="gemma-4-26b-a4b-it")
    args = parser.parse_args()
    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="not-needed")

    configs = [
        # Baseline: enable_thinking=True (expected to loop)
        ("enable_thinking (baseline)", {
            "chat_template_kwargs": {"enable_thinking": True}}),
        # logit_bias on <channel|> token=101 with different strengths
        ("+logit_bias=1", {
            "chat_template_kwargs": {"enable_thinking": True},
            "logit_bias": {"101": 1}}),
        ("+logit_bias=5", {
            "chat_template_kwargs": {"enable_thinking": True},
            "logit_bias": {"101": 5}}),
        ("+logit_bias=10", {
            "chat_template_kwargs": {"enable_thinking": True},
            "logit_bias": {"101": 10}}),
        ("+logit_bias=20", {
            "chat_template_kwargs": {"enable_thinking": True},
            "logit_bias": {"101": 20}}),
        # Also bias <|channel> (start, token=100) to prevent early thinking
        ("+both_bias=10", {
            "chat_template_kwargs": {"enable_thinking": True},
            "logit_bias": {"101": 10, "100": -5}}),
    ]

    print("=" * 70)
    for name, extra in configs:
        test(client, args.model, name, extra)


if __name__ == "__main__":
    main()
