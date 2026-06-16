#!/bin/bash
# Run vllm serve with FDO mode, patching out the spec-decode capture-size
# adjustment that introduces PA/FIA-incompatible batch sizes on Ascend 910B4.
cd /home/wumeng/vllm-ascend/.worktrees/mtp-sequential-loop-debug

export ASCEND_RT_VISIBLE_DEVICES=2,3
export VLLM_ASCEND_MTP_MODE=both_fdo
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Pre-import the patch before vllm loads
python3 -c "import patch_fdo_adjust" || exit 1

# Run vllm serve
exec vllm serve /data/gemma4/gemma-4-31b-it --port 18999 \
  --speculative-config '{"method":"mtp","model":"/data/gemma4/gemma-4-31b-it-assistant","num_speculative_tokens":5}' \
  --tensor-parallel-size 2 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --max-model-len 16384 --max-num-batched-tokens 16384 --max-num-seqs 32 \
  --gpu_memory_utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --enable-prefix-caching \
  2>&1 | tee /home/wumeng/logs/60_fdo_patched.log
