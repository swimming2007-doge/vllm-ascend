#!/usr/bin/env python3
"""Send a single GPQA question to a running vLLM server in thinking mode."""
import argparse
import json
import time
from openai import OpenAI

PROMPT = """Answer the following multiple choice question. Keep your reasoning concise, under 10 sentences. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.

Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?

A) 10^-9 eV
B) 10^-8 eV
C) 10^-4 eV
D) 10^-11 eV"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="not-needed")

    t0 = time.time()
    response = client.chat.completions.create(
        model="gemma-4-26b-a4b-it",
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=args.max_tokens,
        temperature=0,
        extra_body={"thinking": {"type": "enabled"}},
    )
    elapsed = time.time() - t0

    choice = response.choices[0]
    content = choice.message.content or ""
    reason = getattr(choice.message, "reasoning_content", "") or ""

    usage = response.usage
    print(f"=== RESULT ===")
    print(f"latency: {elapsed:.1f}s")
    print(f"prompt_tokens: {usage.prompt_tokens}, completion_tokens: {usage.completion_tokens}")
    print(f"reasoning_tokens: {getattr(usage, 'completion_tokens_details', None)}")
    print(f"output (len={len(content)}): [{content[:500]}]")
    if reason:
        print(f"reasoning (len={len(reason)}): [{reason[:300]}...]" if len(reason) > 300 else f"reasoning (len={len(reason)}): [{reason}]")
    print(f"finish_reason: {choice.finish_reason}")


if __name__ == "__main__":
    main()
