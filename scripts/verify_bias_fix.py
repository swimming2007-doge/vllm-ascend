#!/usr/bin/env python3
"""Verify logit_bias=1 actually works: does reasoning get extracted?"""
import argparse, time
from openai import OpenAI

PROMPT = {"messages": [{"role": "user", "content": (
    "Among the following exoplanets, which one has the highest density?\n\n"
    "a) An Earth-mass and Earth-radius planet.\n"
    "b) A planet with 2 Earth masses and a density of approximately 5.5 g/cm^3.\n"
    "c) A planet with the same composition as Earth but 5 times more massive than Earth.\n"
    "d) A planet with the same composition as Earth but half the mass of Earth.\n\n"
    "Think step by step and explain your reasoning."
)}]}

def test(client, model, name, extra_body, max_tokens=4096):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model, messages=PROMPT["messages"],
        max_tokens=max_tokens, temperature=0, extra_body=extra_body,
    )
    elapsed = time.time() - t0
    msg = resp.choices[0].message
    content = msg.content or ""
    # vLLM uses 'reasoning' field (not 'reasoning_content')
    reasoning = getattr(msg, "reasoning", None) or ""
    usage = resp.usage

    has_end = "<channel|>" in content or "<channel|>" in reasoning
    tokens = usage.completion_tokens if usage else 0

    print(f"[{name}] tokens={tokens} finish={resp.choices[0].finish_reason}")
    print(f"  reasoning_len={len(reasoning)}  content_len={len(content)}")
    print(f"  has_<channel|>_in_content={('<channel|>' in content)}")
    print(f"  has_<channel|>_in_reasoning={('<channel|>' in reasoning)}")
    if reasoning:
        print(f"  reasoning[:300]: {reasoning[:300]}")
        print(f"  reasoning[-200:]: {reasoning[-200:]}")
    if content:
        print(f"  content[:300]: {content[:300]}")
    print(f"  time={elapsed:.1f}s")
    print()
    return tokens, resp.choices[0].finish_reason, len(reasoning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=str, default="gemma-4-26b-a4b-it")
    args = parser.parse_args()
    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="not-needed")

    configs = [
        ("baseline", {"chat_template_kwargs": {"enable_thinking": True}}),
        ("logit_bias=1", {"chat_template_kwargs": {"enable_thinking": True}, "logit_bias": {"101": 1}}),
        ("logit_bias=2", {"chat_template_kwargs": {"enable_thinking": True}, "logit_bias": {"101": 2}}),
        ("logit_bias=3", {"chat_template_kwargs": {"enable_thinking": True}, "logit_bias": {"101": 3}}),
    ]

    for name, extra in configs:
        test(client, args.model, name, extra, max_tokens=4096)


if __name__ == "__main__":
    main()
