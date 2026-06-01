#!/usr/bin/env python3
"""
Compare token-101 (<channel|>) logprobs between PIECEWISE and FULL_DECODE_ONLY.

Usage:
  # Step 1: Run on FULL_DECODE_ONLY server (current)
  python scripts/compare_logits.py --port 8026 --output /tmp/fdo_logprobs.json

  # Step 2: Restart server with PIECEWISE, then:
  python scripts/compare_logits.py --port 8026 --output /tmp/pw_logprobs.json

  # Step 3: Compare
  python scripts/compare_logits.py --compare /tmp/fdo_logprobs.json /tmp/pw_logprobs.json
"""
import argparse, json, time
from openai import OpenAI

# A prompt complex enough to trigger thinking, simple enough to finish quickly
PROMPT = {
    "messages": [{"role": "user", "content": (
        "A particle has energy E=5eV and lifetime tau=1ns. "
        "What is the minimum uncertainty in its energy? "
        "Use the energy-time uncertainty principle: Delta_E * tau >= hbar/2. "
        "Think step by step."
    )}],
}

KEY_TOKENS = {
    "<|channel>": 100,   # start thinking
    "<channel|>": 101,   # end thinking
    "<|think|>": 98,     # think token
    "<|turn>": 105,      # start turn
    "<turn|>": 106,      # end turn
}


def collect_logprobs(port: int, model: str, max_tokens: int = 512):
    """Send a thinking prompt and collect per-step logprobs for key tokens."""
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=PROMPT["messages"],
        max_tokens=max_tokens,
        temperature=0,
        logprobs=True,
        top_logprobs=20,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    elapsed = time.time() - t0

    choice = resp.choices[0]
    content = choice.message.content or ""
    reasoning = getattr(choice.message, "reasoning", None) or ""
    finish = choice.finish_reason

    # Extract per-token logprobs
    logprobs_data = choice.logprobs
    steps = []
    if logprobs_data and logprobs_data.content:
        for i, token_info in enumerate(logprobs_data.content):
            token_str = token_info.token
            token_id = token_info.token if isinstance(token_info.token, int) else -1
            top_logprobs = token_info.top_logprobs or []

            # Extract logprob for key tokens from top_logprobs
            key_logprobs = {}
            for entry in top_logprobs:
                # entry is ChatCompletionTokenLogprob with .token (str) and .logprob (float)
                try:
                    t = entry.token
                    lp = entry.logprob
                    # Try to match token string to key tokens
                    for name, tid in KEY_TOKENS.items():
                        if t == name:
                            key_logprobs[name] = round(lp, 6)
                except Exception:
                    pass

            steps.append({
                "step": i,
                "token": str(token_str),
                "key_logprobs": key_logprobs,
            })

    # Summary
    token_101_steps = []
    token_100_steps = []
    for s in steps:
        if "<channel|>" in s["key_logprobs"]:
            token_101_steps.append((s["step"], s["key_logprobs"]["<channel|>"]))
        if "<|channel>" in s["key_logprobs"]:
            token_100_steps.append((s["step"], s["key_logprobs"]["<|channel>"]))

    print(f"  tokens={len(steps)} finish={finish} time={elapsed:.1f}s")
    print(f"  content_len={len(content)} reasoning_len={len(reasoning)}")
    print(f"  token_101 (<channel|>) in top20 at steps: {token_101_steps}")
    print(f"  token_100 (<|channel>) in top20 at steps: {token_100_steps}")
    if content:
        print(f"  content[:300]: {content[:300]}")

    return {
        "port": port,
        "completion_tokens": len(steps),
        "finish": finish,
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "elapsed": round(elapsed, 1),
        "steps": steps,
        "token_101_in_top20": token_101_steps,
        "token_100_in_top20": token_100_steps,
    }


def compare_results(fdo_path: str, pw_path: str):
    """Compare two logprob dumps."""
    with open(fdo_path) as f:
        fdo = json.load(f)
    with open(pw_path) as f:
        pw = json.load(f)

    print("=" * 70)
    print("COMPARISON: FULL_DECODE_ONLY vs PIECEWISE")
    print("=" * 70)
    print(f"FDO: {fdo['completion_tokens']} tokens, finish={fdo['finish']}, "
          f"content={fdo['content_len']}, reasoning={fdo['reasoning_len']}")
    print(f"PW:  {pw['completion_tokens']} tokens, finish={pw['finish']}, "
          f"content={pw['content_len']}, reasoning={pw['reasoning_len']}")

    # Compare token 101 appearance
    fdo_101 = {s: lp for s, lp in fdo["token_101_in_top20"]}
    pw_101 = {s: lp for s, lp in pw["token_101_in_top20"]}

    if fdo_101 and pw_101:
        print(f"\nToken 101 (<channel|>) appeared in both:")
        fdo_first = min(fdo_101.keys())
        pw_first = min(pw_101.keys())
        print(f"  FDO first appearance: step {fdo_first}, logprob={fdo_101[fdo_first]}")
        print(f"  PW  first appearance: step {pw_first}, logprob={pw_101[pw_first]}")
    elif fdo_101:
        print(f"\nToken 101 appeared ONLY in FDO (steps: {list(fdo_101.keys())})")
    elif pw_101:
        print(f"\nToken 101 appeared ONLY in PW (steps: {list(pw_101.keys())})")
    else:
        print(f"\nToken 101 NEVER appeared in top20 for either mode")

    # Check divergence point: compare top-1 token at each step
    fdo_steps = {s["step"]: s for s in fdo["steps"]}
    pw_steps = {s["step"]: s for s in pw["steps"]}
    diverged_at = None
    for step in sorted(set(fdo_steps) & set(pw_steps)):
        fdo_tok = fdo_steps[step]["token"]
        pw_tok = pw_steps[step]["token"]
        if fdo_tok != pw_tok:
            diverged_at = step
            print(f"\nFirst token divergence at step {step}:")
            print(f"  FDO: '{fdo_tok}'")
            print(f"  PW:  '{pw_tok}'")
            # Show key token logprobs at this step
            if "key_logprobs" in fdo_steps[step]:
                print(f"  FDO key logprobs: {fdo_steps[step]['key_logprobs']}")
            if "key_logprobs" in pw_steps[step]:
                print(f"  PW  key logprobs: {pw_steps[step]['key_logprobs']}")
            break

    if diverged_at is None:
        # Check if both have same tokens but different lengths
        min_len = min(len(fdo["steps"]), len(pw["steps"]))
        print(f"\nAll {min_len} shared steps have identical top-1 tokens")
        if len(fdo["steps"]) != len(pw["steps"]):
            print(f"  FDO generated {len(fdo['steps']) - min_len} more tokens")
            print(f"  PW  generated {len(pw['steps']) - min_len} more tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--model", type=str, default="gemma-4-26b-a4b-it")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, help="Save results to JSON")
    parser.add_argument("--compare", nargs=2, metavar=("FDO_JSON", "PW_JSON"),
                        help="Compare two saved results")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.port:
        result = collect_logprobs(args.port, args.model, args.max_tokens)
        if args.output:
            # Convert to serializable format
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Saved to {args.output}")
    else:
        parser.error("Need --port or --compare")


if __name__ == "__main__":
    main()
