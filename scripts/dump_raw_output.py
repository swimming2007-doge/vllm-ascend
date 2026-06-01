#!/usr/bin/env python3
"""Quick raw output dump: check exact token behavior with skip_special_tokens=False."""
import argparse, json, time
from openai import OpenAI

PROMPT = "What is 2+2? Explain step by step."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=str, default="gemma-4-26b-a4b-it")
    args = parser.parse_args()
    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="not-needed")

    for name, extra in [
        ("enable_thinking=True", {"chat_template_kwargs": {"enable_thinking": True}}),
        ("thinking:enabled", {"thinking": {"type": "enabled"}}),
        ("no_thinking", {}),
    ]:
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=512,
                temperature=0,
                extra_body=extra,
            )
            c = resp.choices[0]
            msg = c.message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            usage = resp.usage

            print(f"[{name}]")
            print(f"  tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
            print(f"  finish: {c.finish_reason}")
            print(f"  content_len: {len(content)}")
            print(f"  reasoning_len: {len(reasoning)}")
            print(f"  has_<|channel>: {'<|channel>' in content}")
            print(f"  has_<channel|>: {'<channel|>' in content}")
            if content:
                # Show first 500 chars with repr for special chars
                snippet = content[:500]
                print(f"  content[:500]: {snippet}")
            if reasoning:
                print(f"  reasoning[:300]: {reasoning[:300]}")
            # Check all message fields
            model_dump = msg.model_dump() if hasattr(msg, 'model_dump') else {}
            print(f"  message keys: {list(model_dump.keys())}")
            print()
        except Exception as e:
            print(f"[{name}] ERROR: {e}\n")

if __name__ == "__main__":
    main()
