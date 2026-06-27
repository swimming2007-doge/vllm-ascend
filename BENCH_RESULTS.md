# MTP Acceptance Rate Benchmark Results

**Date**: 2026-06-26  
**Commit**: 225bf65e  
**Model**: Gemma4 31B, K=4, eager mode, TP=2  
**Protocol**: 4 concurrent × 10 rounds × 200 tokens/req, 100/100 OK  
**Log**: `/home/wumeng/logs/31b_MTP_k4_btfix_v2.log`

## Weighted Acceptance Rate

| Pos0 | Pos1 | Pos2 | Pos3 | Avg |
|------|------|------|------|-----|
| 68.0% | 44.5% | 29.6% | 19.9% | 40.5% |

## Key Fixes Applied

1. **cf05109c** — Block table routing: draft KV reads now use per-group block_table
   via `_kv_share_gid` + `_per_group_bt_ref`
2. **2f245f5c** — Residual flow: restored `residual = hidden_states` at top of
   `Gemma4MTPDecoderLayer.forward()` (was lost during debug cleanup)
3. **225bf65e** — Debug cleanup: all diagnostic logging removed from 6 files

## Comparison vs Pre-Fix Baseline (K=5 FDO)

| Pre-fix | Pos0 49.5% | Pos1 22.2% | Pos2 9.9% | Pos3 5.1% | Avg 17.9% |
| Post-fix | 68.0% | 44.5% | 29.6% | 19.9% | 40.5% |

Improvement: **+22.6pp** (2.3× baseline)

---

## K=3, Target FDO + Draft Eager, TP=2

**Date**: 2026-06-27  
**Commit**: b8875992  
**Model**: Gemma4 31B, K=3, Target FDO (FULL_DECODE_ONLY), Draft eager, TP=2  
**Config**: `VLLM_ASCEND_MTP_MODE=draft_eager`, `cudagraph_mode=FULL_DECODE_ONLY`, no capture size limit  
**Protocol**: 4 concurrent × 10 rounds × 200 tokens/req, 100/100 OK  
**Log**: `/home/wumeng/logs/31b_MTP_k3_draft_eager.log`

## Throughput & Latency

| Metric | K=4 eager (baseline) | K=3 FDO target | Change |
|--------|:---:|:---:|:---:|
| Throughput | 16.0 tok/s | **25.3 tok/s** | **+58%** |
| TTFT | ~800ms | **~510ms** | **-36%** |
| Accepted throughput | ~16 tok/s | ~35 tok/s | +119% |
| Drafted throughput | ~45 tok/s | ~78 tok/s | +73% |

## Weighted Acceptance Rate (K=3)

| Pos0 | Pos1 | Pos2 | Avg |
|:---:|:---:|:---:|:---:|
| 66% | 45% | 31% | **48%** |

## Code Changes

1. **4585eec2** — Event sync enabled for target FDO KV cache writes, draft FDO hardcoded off
2. **b8875992** — Cleaned [FIA-UPDATE] debug prints from attention_v1.py
3. **9ed19929** — Guard `synchronize()` during NPU graph capture in `_run_merged_draft`
4. K=3 (was K=4), auto capture sizes (was [5])

