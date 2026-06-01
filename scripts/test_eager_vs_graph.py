#!/usr/bin/env python3
"""
Send identical test prompts to PIECEWISE and FULL_DECODE_ONLY vLLM servers,
collect outputs, timings, and dump data for comparison.

Usage:
  # Terminal 1: start PIECEWISE server (good mode, 88%+)
  export VLLM_ASCEND_DUMP_DIR=/tmp/vllm_dump/PIECEWISE
  vllm serve /data/gemma4/gemma-4-31b-it -tp 4 \
      --host 127.0.0.1 --port 8831 ...
  python scripts/test_eager_vs_graph.py --port 8831 --mode piecewise

  # Terminal 2: start FULL_DECODE_ONLY server (bad mode)
  export VLLM_ASCEND_DUMP_DIR=/tmp/vllm_dump/FULL_DECODE_ONLY
  vllm serve /data/gemma4/gemma-4-31b-it -tp 4 \
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' ...
  python scripts/test_eager_vs_graph.py --port 8831 --mode fulldecode

  # Compare
  python scripts/test_eager_vs_graph.py --compare /tmp/vllm_test_output/piecewise /tmp/vllm_test_output/fulldecode
"""

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI

# ── Gemma 4 key tokens for thinking model ──────────────────────
#   think_token: <|think|>
#   soc_token:   <|channel>     (start of channel/thinking)
#   eoc_token:   <channel|>     (end of channel/thinking)

# ── Test prompts (varying thinking depth) ───────────────────────
TEST_PROMPTS = [
    # 1. GPQA Physics — requires deep thinking, evaluates <channel|> correctness
    {
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
        "expect_thinking": True,
        "ground_truth": "c",
    },
    # 2. Multi-step math — triggers long thinking chain
    {
        "id": "math_1",
        "messages": [
            {"role": "user", "content": "请一步步推导: 证明根号2是无理数"}
        ],
        "max_tokens": 1024,
        "expect_thinking": True,
    },
    # 3. Arithmetic with explicit request for step-by-step
    {
        "id": "math_2",
        "messages": [
            {"role": "user", "content": "A store sells apples at $3 each. "
             "If I buy 5 apples on Monday and 3 more on Tuesday, "
             "how much did I spend total? Think step by step."}
        ],
        "max_tokens": 512,
        "expect_thinking": True,
    },
    # 4. Simple factual — should NOT trigger deep thinking
    {
        "id": "factual_1",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 128,
        "expect_thinking": False,
    },
    # 5. Code generation
    {
        "id": "code_1",
        "messages": [
            {"role": "user", "content": "Write a Python function to check "
             "if a string is a palindrome. Include explanation."}
        ],
        "max_tokens": 512,
        "expect_thinking": True,
    },
    # 6. Logic puzzle — needs multi-step reasoning
    {
        "id": "logic_1",
        "messages": [
            {"role": "user", "content": "Alice is taller than Bob. "
             "Bob is taller than Charlie. David is shorter than Bob "
             "but taller than Charlie. Who is the tallest and who is the shortest?"}
        ],
        "max_tokens": 256,
        "expect_thinking": True,
    },
    # 7. Short question — minimal thinking
    {
        "id": "simple_1",
        "messages": [
            {"role": "user", "content": "What is 15 + 27?"}
        ],
        "max_tokens": 64,
        "expect_thinking": False,
    },
]


def run_tests(port: int, mode: str, output_dir: str, num_repeats: int = 1):
    """Send all test prompts and collect results."""
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed")
    outdir = Path(output_dir) / mode
    outdir.mkdir(parents=True, exist_ok=True)
    results = []

    for idx, prompt in enumerate(TEST_PROMPTS):
        for rep in range(num_repeats):
            rid = f"{prompt['id']}_r{rep}" if num_repeats > 1 else prompt["id"]
            print(f"\n{'='*60}")
            print(f"[{mode}] {rid}  max_tokens={prompt['max_tokens']}")
            print(f"  prompt: {prompt['messages'][0]['content'][:100]}...")

            t0 = time.time()
            try:
                response = client.chat.completions.create(
                    model="gemma-4-31b-it",
                    messages=prompt["messages"],
                    max_tokens=prompt["max_tokens"],
                    temperature=0,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True}
                    },
                )
                elapsed = time.time() - t0

                choice = response.choices[0]
                msg = choice.message
                text = msg.content or ""
                reasoning = getattr(msg, "reasoning", None) or ""
                finish_reason = choice.finish_reason

                # Check for thinking in reasoning field (vLLM strips tags)
                has_think = len(reasoning) > 0
                has_think_start = "<|channel>" in text or "<|channel>" in reasoning
                has_think_end = "<channel|>" in text or "<channel|>" in reasoning
                has_channel = has_think_start or has_think_end or has_think

                # Check ground truth for GPQA questions
                gt = prompt.get("ground_truth")
                answer_correct = None
                if gt:
                    # Rough check: see if the output contains the correct answer letter
                    answer_correct = gt.lower() in (text + reasoning).lower()

                result = {
                    "id": rid,
                    "mode": mode,
                    "prompt": prompt["messages"][0]["content"],
                    "output_text": text,
                    "reasoning_text": reasoning[:500],  # first 500 chars of thinking
                    "reasoning_length": len(reasoning),
                    "output_length": len(text),
                    "finish_reason": finish_reason,
                    "has_think": has_think,
                    "has_think_start": has_think_start,
                    "has_think_end": has_think_end,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "elapsed_sec": round(elapsed, 3),
                    "tokens_per_sec": round(
                        response.usage.completion_tokens / elapsed, 1
                    ) if response.usage and elapsed > 0 else 0,
                    "expect_thinking": prompt["expect_thinking"],
                    "ground_truth": gt,
                    "answer_correct": answer_correct,
                }

                print(f"  tokens: {result['completion_tokens']}  "
                      f"finish: {finish_reason}  "
                      f"think: {has_think} (reasoning={len(reasoning)} chars)  "
                      f"{result['tokens_per_sec']} tok/s")
                if gt:
                    print(f"  ground_truth: {gt}  match: {answer_correct}")

                # Warn if thinking was expected but not present
                if prompt["expect_thinking"] and not has_think:
                    print(f"  *** WARNING: expected thinking but none detected!")
                # Warn if thinking didn't close (potential loop)
                if has_think_start and not has_think_end:
                    print(f"  *** WARNING: thinking started but never ended (LOOP?)")

            except Exception as e:
                elapsed = time.time() - t0
                result = {
                    "id": rid,
                    "mode": mode,
                    "error": str(e),
                    "elapsed_sec": round(elapsed, 3),
                }
                print(f"  ERROR: {e}")

            results.append(result)

    # Save results
    results_path = outdir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_path}")

    # Quick summary
    errors = [r for r in results if "error" in r]
    no_think_end = [r for r in results
                    if r.get("has_think_start") and not r.get("has_think_end")]
    print(f"  Total: {len(results)}  Errors: {len(errors)}  "
          f"Think-not-closed: {len(no_think_end)}")
    if no_think_end:
        print("  THINK-LOOP candidates:")
        for r in no_think_end:
            print(f"    - {r['id']}: {r['output_text'][:200]}...")

    return results


def compare_results(piecewise_dir: str, fulldecode_dir: str):
    """Compare PIECEWISE vs FULL_DECODE_ONLY test results."""
    pw_path = Path(piecewise_dir) / "test_results.json"
    fd_path = Path(fulldecode_dir) / "test_results.json"

    if not pw_path.exists():
        print(f"PIECEWISE results not found: {pw_path}")
        return
    if not fd_path.exists():
        print(f"FULL_DECODE_ONLY results not found: {fd_path}")
        return

    pw_data = json.loads(pw_path.read_text())
    fd_data = json.loads(fd_path.read_text())

    pw_by_id = {r["id"]: r for r in pw_data}
    fd_by_id = {r["id"]: r for r in fd_data}

    print("\n" + "=" * 70)
    print("PIECEWISE vs FULL_DECODE_ONLY COMPARISON")
    print("=" * 70)

    divergences = 0
    for rid in sorted(set(pw_by_id) | set(fd_by_id)):
        pw = pw_by_id.get(rid, {})
        fd = fd_by_id.get(rid, {})
        pw_err = pw.get("error")
        fd_err = fd.get("error")

        print(f"\n[{rid}]")
        if pw_err or fd_err:
            print(f"  piecewise error:       {pw_err}")
            print(f"  full_decode_only error: {fd_err}")
            divergences += 1
            continue

        pw_text = pw.get("output_text", "")
        fd_text = fd.get("output_text", "")
        pw_reason = pw.get("reasoning_length", 0)
        fd_reason = fd.get("reasoning_length", 0)
        pw_think_end = pw.get("has_think_end", False)
        fd_think_end = fd.get("has_think_end", False)
        pw_finish = pw.get("finish_reason")
        fd_finish = fd.get("finish_reason")
        pw_correct = pw.get("answer_correct")
        fd_correct = fd.get("answer_correct")
        gt = pw.get("ground_truth")

        issues = []
        if pw_text != fd_text:
            issues.append(f"TEXT DIFFERS (piecewise={len(pw_text)} chars, fulldecode={len(fd_text)} chars)")
        if pw_finish != fd_finish:
            issues.append(f"FINISH_REASON differs: piecewise={pw_finish} fulldecode={fd_finish}")
        if pw_reason != fd_reason:
            issues.append(f"REASONING LENGTH differs: piecewise={pw_reason} fulldecode={fd_reason}")
        if pw_think_end != fd_think_end:
            if fd_think_end and not pw_think_end:
                issues.append("think_end: piecewise=NO fulldecode=YES")
            else:
                issues.append("*** CRITICAL: think_end: piecewise=YES fulldecode=NO (LOOP!) ***")
                divergences += 1
        if gt:
            if pw_correct != fd_correct:
                issues.append(f"*** CRITICAL: ANSWER differs: piecewise={pw_correct} fulldecode={fd_correct} (GT={gt}) ***")
                divergences += 1
            else:
                issues.append(f"answer: both {'correct' if pw_correct else 'incorrect'} (GT={gt})")

        if issues:
            for issue in issues:
                print(f"  {issue}")
            print(f"  piecewise text[:300]: {pw_text[:300]}")
            print(f"  fulldecode text[:300]: {fd_text[:300]}")
            divergences += 1
        else:
            print(f"  IDENTICAL (finish={pw_finish}, {pw.get('completion_tokens', 0)} tokens)")

    print(f"\n{'='*70}")
    print(f"Total divergences: {divergences}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test PIECEWISE vs FULL_DECODE_ONLY for Gemma 4 thinking model")
    parser.add_argument("--port", type=int, default=8831, help="vLLM server port")
    parser.add_argument("--mode", choices=["piecewise", "fulldecode"], required=False,
                        help="Which mode the server is running in")
    parser.add_argument("--output-dir", default="/tmp/vllm_test_output",
                        help="Directory to save test results")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Number of times to repeat each prompt")
    parser.add_argument("--compare", nargs=2, metavar=("PIECEWISE_DIR", "FULLDECODE_DIR"),
                        help="Compare existing results instead of running tests")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.mode:
        run_tests(args.port, args.mode, args.output_dir, args.repeat)
    else:
        parser.error("Either --mode or --compare is required")
