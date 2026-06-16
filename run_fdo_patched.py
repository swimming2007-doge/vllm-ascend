#!/usr/bin/env python3
"""
Wrapper: patch adjust_cudagraph_sizes_for_spec_decode then run vllm serve.

MUST be run as __main__ so that vllm's multiprocessing (spawn) can
re-import this module in worker processes without side effects.
"""
import sys
import os

# ── Apply the patch FIRST (before vllm imports).  This runs every
#    time this module is imported (main process + spawned workers).
_sys_path_0 = os.path.dirname(os.path.abspath(__file__))
if _sys_path_0 not in sys.path:
    sys.path.insert(0, _sys_path_0)
import patch_fdo_adjust  # noqa: E402


def _main():
    """Entry point — only executed when run as __main__, not by workers."""
    # Use both_fdo mode: two separate FDO graphs (target 60 layers + draft 4 layers),
    # each with its own capture, keeping total task groups per graph under CANN limit.
    os.environ["VLLM_ASCEND_MTP_MODE"] = "both_fdo"
    # Ascend NPU device selection uses ASCEND_RT_VISIBLE_DEVICES, not CUDA_VISIBLE_DEVICES
    if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"

    sys.argv = [
        "vllm", "serve",
        "/data/gemma4/gemma-4-31b-it",
        "--port", "18999",
        "--speculative-config",
        '{"method":"mtp","model":"/data/gemma4/gemma-4-31b-it-assistant","num_speculative_tokens":5}',
        "--tensor-parallel-size", "2",
        "--compilation-config", '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--trust-remote-code",
        "--max-model-len", "16384",
        "--max-num-batched-tokens", "16384",
        "--max-num-seqs", "32",
        "--gpu_memory_utilization", "0.70",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "gemma4",
        "--reasoning-parser", "gemma4",
        "--enable-prefix-caching",
    ]
    from vllm.entrypoints.cli.main import main
    sys.exit(main())


if __name__ == "__main__":
    _main()
