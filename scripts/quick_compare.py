#!/usr/bin/env python3
"""Send the same prompt to a running vLLM server and capture first-token logits.

Usage:
  python quick_compare.py --port 8826 --prompt "What is 2+2?" --max-tokens 1
"""
import argparse
import json
import sys
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="not-needed")

    response = client.completions.create(
        model="gemma-4-26b-a4b-it",
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=0,
        logprobs=20,
    )

    choice = response.choices[0]
    result = {
        "text": choice.text,
        "finish_reason": choice.finish_reason,
        "logprobs": choice.logprobs.model_dump() if choice.logprobs else None,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
