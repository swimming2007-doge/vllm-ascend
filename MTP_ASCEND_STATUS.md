# Gemma4 31B MTP on Ascend NPU — Status Summary for vLLM 0.23.0 Migration

**Date**: 2026-06-29
**Current base commit**: `64fce64b` (vLLM 0.21.0 based)

---

## 1. Working Configuration

### Mode: `draft_eager` (Target FDO + Draft eager)

| Component | Mode | Status |
|-----------|------|--------|
| Target model | FULL_DECODE_ONLY (FDO graph) | ✅ Working |
| Draft model | Eager (torch.compile only) | ✅ Working |
| Acceptance rate (K=3) | TP2: ~49%, TP4: ~58% | ✅ Good |
| Throughput (TP4, 4k→1.5k) | 38.5 tok/s output | ✅ |

### Mode: `both_fdo` (Target FDO + Draft FDO) — SHELVED

| Issue | Detail |
|-------|--------|
| Root cause | Draft ACLGraphWrapper created with `runtime_mode=FULL` but `forward_context.cudagraph_runtime_mode` is `PIECEWISE` → wrapper always falls through to eager |
| Nested capture | Target PIECEWISE capturing → `_outer_capturing=True` → Draft skips capture → attention ops never store graph params |
| Hang | Replay path: `event.wait()` inside captured graph never signals because `update_graph_params` records events on wrong stream during nested capture |
| Status | **SHELVED** — requires coordinated changes across `acl_graph.py`, `model_runner_v1.py`, `attention_v1.py` |

### Mode: `target_eager` — Not tested

---

## 2. Key Code Paths (Ascend-specific)

### Files to port/adapt for vLLM 0.23.0

| File | Role | Critical Lines |
|------|------|----------------|
| `vllm_ascend/compilation/acl_graph.py` | **ACLGraphWrapper** — NPU graph capture/replay dispatch | L111-224 `__call__`, L152-158 nested capture guard, L236-268 `update_full_graph_params` |
| `vllm_ascend/worker/model_runner_v1.py` | **Mode dispatch** — MTP mode selection, ACLGraphWrapper creation | L3486-3516 MTP mode env var + wrapper, L2022-2033 target event record, L3171-3177 cudagraph_runtime_mode |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | **Draft proposer** — Draft ACLGraphWrapper, `_propose()`, `_update_full_graph_params` | L352-383 load_model, L568-911 `_propose`, L601-604 FULL gate, L2045-2064 `_update_full_graph_params` |
| `vllm_ascend/spec_decode/gemma4_proposer.py` | **Gemma4-specific** — KV sharing setup, draft model loading | L325-403 `load_model` |
| `vllm_ascend/attention/attention_v1.py` | **Attention backend** — Graph param storage, PA block_table update | L522-604 draft path, L572-604 PA path |
| `vllm_ascend/platform.py` | **Platform config** — CUDAGraphMode defaults, FULL_DECODE_ONLY downgrade | L325-420 compilation config post-processing |
| `vllm_ascend/models/gemma4_mtp.py` | **Draft model** — 4-layer Gemma4MTPModel | KV sharing layers |

### Env vars

```bash
VLLM_ASCEND_MTP_MODE=draft_eager  # draft_eager | both_fdo | target_eager
# default is target_eager
```

### CLI args

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'  # Required for Target FDO
```

---

## 3. Critical Ascend-specific Behaviors

### 3.1 NPUGraph.replay() is async
- `torch.npu.NPUGraph.replay()` launches work asynchronously
- **Post-replay `stream.synchronize()` DEADLOCKS** (commits `2eb30a16`, `90e4a4ad`)
- Use `event.record()` + `event.wait()` for inter-model sync instead

### 3.2 Nested NPU graph capture corrupts outer workspace
- Comment at `acl_graph.py:141-158` documents this
- Target PIECEWISE capture → `_outer_capturing=True` → Draft skips capture entirely
- Draft runs eagerly (backed by torch.compile, not NPU graphs)

### 3.3 FULL_DECODE_ONLY must be set explicitly
- Default `cudagraph_mode` is `PIECEWISE` (platform.py L378-380 downgrades FULL_AND_PIECEWISE)
- Even with `_mtp_target_fdo=True`, the compilation config stays PIECEWISE unless overridden
- `ACLGraphWrapper(runtime_mode=FULL)` is bypassed when `forward_context` says PIECEWISE

### 3.4 Target→Draft KV sync (Gemma4 specific)
- Target FDO graph writes KV cache asynchronously on NPU stream
- After target forward, `event.record()` → `drafter._target_done_event` (L2029-2033)
- Draft must call `_sync_wait_target_events()` before reading KV cache
- **Gemma4-specific**: `_update_full_graph_params` must be called BEFORE `run_draft()` (commit `aba0370e`, lost in reset)

---

## 4. Working Fixes (committed, still in branch)

| Commit | Description |
|--------|-------------|
| `2f245f5c` | Fix: `residual=hidden_states` in Gemma4MTPDecoderLayer.forward |
| `cf05109c` | Fix: route draft KV reads to correct per-group block_table |
| `2eb30a16` | feat: always sync target KV cache writes before draft FDO reads |
| `d28e1630` | feat: add NPU event-based sync gap diagnostics |
| `b8875992` | chore: remove FIA-UPDATE debug prints |

---

## 5. Known Issues

| Issue | Severity | Notes |
|-------|----------|-------|
| both_fdo hang | Critical | Shelved — requires FULL mode dispatch fix |
| 0% acceptance with wrong draft model | High | Must use `/data/gemma4/gemma-4-31b-it-assistant` (NOT `-w8a8-compressed`) |
| Draft KV sharing returns zeros for first K steps | Medium | Only observed in diagnostic mode, acceptance rate OK in practice |
| NPU OOM with 4k prefill at high concurrency | Medium | Mitigated by `max_num_seqs=2`, `max_num_batched_tokens=8192` |
| 4k prefill + TP4 requires ≥0.75 gpu_mem for headroom | Low | |

---

## 6. Migration Notes for vLLM 0.23.0

### Likely affected areas

1. **vLLM V1 API changes** — `vllm/v1/engine/`, `AsyncLLM`, `EngineCore` interfaces may have changed
2. **CompilationConfig** — `cudagraph_mode` enum, `compile_sizes`, `cudagraph_capture_sizes` APIs
3. **SpeculativeConfig** — `method="mtp"` config format, `num_speculative_tokens`
4. **Model registry** — `DeepseekV4ForCausalLM`, `DeepSeekV4MTPModel` overwrite warnings
5. **Platform plugin** — `vllm_ascend:register` entry point
6. **ACL Graph** — `torch.npu.NPUGraph`, `torch.npu.is_current_stream_capturing()` APIs

### Priority order

1. Port `acl_graph.py` ACLGraphWrapper — core FDO mechanism
2. Port `model_runner_v1.py` MTP mode dispatch + compile config
3. Port `llm_base_proposer.py` draft proposer
4. Port `attention_v1.py` graph param storage
5. Port `platform.py` compilation config post-processing
6. Port `gemma4_proposer.py` + `gemma4_mtp.py` KV sharing
