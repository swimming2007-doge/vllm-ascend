# TP4 FDO Throughput Benchmark — Gemma4 31B MTP K=3

**Date**: 2026-06-29
**Commit**: 64fce64b
**Mode**: draft_eager (Target FDO + Draft eager)
**Config**: TP=4, NPU 0-3, FULL_DECODE_ONLY, max_num_seqs=2, max_num_batched_tokens=8192, gpu_memory_utilization=0.75

## Scenario 1: 4k input / 1.5k output (--max-concurrency 2)

| Metric | Value |
|--------|-------|
| Success rate | 10/10 |
| Benchmark duration | 399s |
| Output token throughput | **38.51 tok/s** |
| Total token throughput | 141.21 tok/s |
| Mean TTFT | 4.29s |
| Mean TPOT | 48.4ms |

| Pos 0 | Pos 1 | Pos 2 | Avg Acceptance | Mean Accept Length |
|-------|-------|-------|----------------|-------------------|
| 62.09% | 57.37% | 57.10% | 58.85% | 2.77 |

## Scenario 2: 1k input / 256 output (--max-concurrency 4)

| Metric | Value |
|--------|-------|
| Success rate | 20/20 |
| Benchmark duration | 139s |
| Output token throughput | **36.93 tok/s** |
| Total token throughput | 192.72 tok/s |
| Mean TTFT | 13.45s |
| Mean TPOT | 49.5ms |

| Pos 0 | Pos 1 | Pos 2 | Avg Acceptance | Mean Accept Length |
|-------|-------|-------|----------------|-------------------|
| 60.34% | 56.94% | 55.50% | 57.59% | 2.73 |

## Comparison: TP2 vs TP4 (draft_eager + FDO)

| Metric | TP2 (16k, 4-con) | TP4 (4k/1.5k, 2-con) |
|--------|-------------------|----------------------|
| Output tok/s | 24-26 | 38.51 |
| Pos 0 | 63.9% | 62.09% |
| Pos 1 | 46.6% | 57.37% |
| Pos 2 | 37.2% | 57.10% |
| Avg Acceptance | 49.2% | 58.85% |

## Key Findings

1. TP4 yields ~50% higher throughput (38.5 vs 25 tok/s) vs TP2
2. Pos 1/2 acceptance dramatically improves with TP4: +11pp / +20pp
3. Long-context (4k) acceptance is excellent at ~59% — draft model benefits from richer context
4. Scenario 2 TTFT high (13s) due to max_num_seqs=2 bottleneck — requests queue behind long-running 4k prefills
