#!/bin/bash
# One-shot eager vs graph comparison for Gemma 4 thinking model.
# Usage: bash scripts/run_comparison.sh
#
# This script:
#  1. Starts eager-mode server on port 8831
#  2. Sends test prompts, waits for completion
#  3. Kills eager server
#  4. Starts graph-mode (PIECEWISE) server on port 8831
#  5. Sends same test prompts
#  6. Kills graph server
#  7. Runs comparison

set -e

MODEL="/data/gemma4/gemma-4-31b-it"
CHAT_TEMPLATE="${MODEL}/chat_template.jinja"
DUMP_BASE="/tmp/vllm_dump"
TEST_OUTPUT="/tmp/vllm_test_output"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/test_eager_vs_graph.py"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_dumps.py"

EAGER_PORT=8831
GRAPH_PORT=8832  # use different port to avoid conflict

# Cleanup from previous runs
rm -rf "${DUMP_BASE}" "${TEST_OUTPUT}"

# ── Step 1: Start eager server ──────────────────────────────────
echo "=== Starting EAGER server on port ${EAGER_PORT} ==="
mkdir -p "${DUMP_BASE}/eager"
VLLM_ASCEND_DUMP_DIR="${DUMP_BASE}/eager" \
  nohup vllm serve "${MODEL}" \
    --enforce-eager \
    -tp 4 \
    --host 127.0.0.1 --port ${EAGER_PORT} \
    --served-model-name gemma-4-31b-it \
    --trust-remote-code \
    --max-num-batched-tokens 32768 \
    --max-model-len 131072 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --chat-template "${CHAT_TEMPLATE}" \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser gemma4 \
    > /tmp/vllm_eager.log 2>&1 &

EAGER_PID=$!
echo "Eager server PID: ${EAGER_PID}"

# Wait for server to be ready
echo "Waiting for eager server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${EAGER_PORT}/health" > /dev/null 2>&1; then
        echo "Eager server ready after ${i}s"
        break
    fi
    sleep 1
done

# ── Step 2: Run tests against eager server ──────────────────────
echo ""
echo "=== Running tests against EAGER server ==="
python3 "${TEST_SCRIPT}" --port ${EAGER_PORT} --mode eager --output-dir "${TEST_OUTPUT}"

# ── Step 3: Kill eager server ───────────────────────────────────
echo ""
echo "=== Killing eager server ==="
kill ${EAGER_PID} 2>/dev/null || true
wait ${EAGER_PID} 2>/dev/null || true
sleep 3

# ── Step 4: Start graph server ──────────────────────────────────
echo ""
echo "=== Starting GRAPH server on port ${GRAPH_PORT} ==="
mkdir -p "${DUMP_BASE}/graph"
VLLM_ASCEND_DUMP_DIR="${DUMP_BASE}/graph" \
  nohup vllm serve "${MODEL}" \
    -tp 4 \
    --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
    --host 127.0.0.1 --port ${GRAPH_PORT} \
    --served-model-name gemma-4-31b-it \
    --trust-remote-code \
    --max-num-batched-tokens 32768 \
    --max-model-len 131072 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --chat-template "${CHAT_TEMPLATE}" \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser gemma4 \
    > /tmp/vllm_graph.log 2>&1 &

GRAPH_PID=$!
echo "Graph server PID: ${GRAPH_PID}"

echo "Waiting for graph server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${GRAPH_PORT}/health" > /dev/null 2>&1; then
        echo "Graph server ready after ${i}s"
        break
    fi
    sleep 1
done

# ── Step 5: Run tests against graph server ──────────────────────
echo ""
echo "=== Running tests against GRAPH server ==="
python3 "${TEST_SCRIPT}" --port ${GRAPH_PORT} --mode graph --output-dir "${TEST_OUTPUT}"

# ── Step 6: Kill graph server ───────────────────────────────────
echo ""
echo "=== Killing graph server ==="
kill ${GRAPH_PID} 2>/dev/null || true
wait ${GRAPH_PID} 2>/dev/null || true

# ── Step 7: Compare results ─────────────────────────────────────
echo ""
echo "=== Comparing text outputs ==="
python3 "${TEST_SCRIPT}" --compare "${TEST_OUTPUT}/eager" "${TEST_OUTPUT}/graph"

echo ""
echo "=== Comparing internal tensor dumps ==="
python3 "${COMPARE_SCRIPT}" "${DUMP_BASE}/eager" "${DUMP_BASE}/graph"

echo ""
echo "=== Done ==="
echo "Text results:  ${TEST_OUTPUT}/eager/test_results.json"
echo "Tensor dumps:  ${DUMP_BASE}/eager/  and  ${DUMP_BASE}/graph/"
