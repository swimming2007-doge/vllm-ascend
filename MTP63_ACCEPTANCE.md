# Gemma4 MTP 接受率验证 — 63.8%

日期: 2026-07-08

## 测试结果

| 项 | 值 |
|---|---|
| 分支/commit | `pr-10643` @ `03625020` |
| target | `gemma-4-31B-it`（instruct） |
| draft | `gemma-4-31B-it-assistant` |
| k | 3 |
| 模式 | FULL_DECODE_ONLY (FDO) |
| NPU | 0，端口 8000 |
| 请求 | 30 × 128 tokens，全部成功 |
| drafts | 1321 |
| draft_tokens | 3963 |
| accepted | 2528 |
| **接受率** | **63.8%** |
| 平均每次接受 | 1.91 / 3.0 |
| pos0 / pos1 / pos2 | 82.1% / 55.7% / 53.6% |
| wall time | 79.2s |
| 吞吐 | 48.5 tokens/s |

与 2026-07-03 基线逐位吻合（1321 drafts / 3963 draft_tokens / 2528 accepted，pos0=82% pos1=56% pos2=54%）。

## 说明

`pr-10643 @ 03625020` 这个 commit 早就存在、早就 push 过了（在 `MTP-vllm23` 分支上，作者是 `swimming2007-doge <791025341@qq.com>`）。工作树是干净的，没有新改动。本次提交仅为补一份接受率验证记录（markdown），不改动任何代码。

## 环境

- torch 2.10.0+cpu / torch_npu 2.10.0 / CANN 9.1.0.B050 / vLLM 0.23.0
- NPU: Ascend950PR（A5），单卡 0
- target/draft 权重: `/home/wumeng/google/gemma-4-31B-it`、`/home/wumeng/google/gemma-4-31B-it-assistant`
- 启动: `start_vllm_server.py` + `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` + `--gpu-memory-utilization 0.85` + `--max-model-len 2048`
- 测量脚本: `/home/wumeng/log/run_acc_verify.py`（30 请求 × 128 tokens，temperature=0，走 `/metrics` delta）
- 日志: `/home/wumeng/log/vllm_mtp63_test.log`、`/home/wumeng/log/acc_mtp63_result.txt`
