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
