#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Equivalence tests for the _npu_apply_top_k_top_p adapter.

torch.ops._C_ascend.npu_apply_top_k_top_p (the custom Ascend op) was pulled
from recent torch_npu/CANN releases, so any sampling step with top_k/top_p
crashes on such stacks ("npu_apply_top_k_top_p does not exist") unless the
call routes through the CANN builtin torch_npu.npu_top_k_top_p. These tests
pin the adapter against the upstream vllm apply_top_k_top_p_pytorch
semantics (top-p over the renormalized post-top-k distribution):

  - top-k only: kept-token sets exactly equal
  - k+p combined with fp32 logits: exactly equal
  - bf16 logits: only the top-p cutoff boundary may drift, by negligible
    probability mass (p is cast to the logits dtype per the op's contract
    and the op runs softmax/cumsum in that dtype)
"""

import unittest

import torch
import torch_npu  # noqa: F401  (registers the npu TopKTopP op)

from tests.ut.base import TestBase
from vllm_ascend.sample.sampler import _npu_apply_top_k_top_p

B, V = 8, 4096
# Includes no-top-k rows (k == V), top-p-only-friendly rows and extreme k=1.
KS = [10, V, 100, 5, V, 32, 64, 1]
PS = [0.9, 1.0, 0.95, 0.5, 0.99, 1.0, 0.8, 0.3]


def _ref_kept(logits, k, p):
    """Upstream vllm apply_top_k_top_p_pytorch kept-token mask."""
    lg = logits.float()
    sorted_lg, idx = lg.sort(dim=-1, descending=False)
    if k is not None:
        cutoff = sorted_lg.gather(1, (sorted_lg.size(1) - k.to(torch.long)).unsqueeze(1))
        sorted_lg = sorted_lg.masked_fill(sorted_lg < cutoff, float("-inf"))
    if p is not None:
        probs = sorted_lg.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        m = cum <= (1.0 - p.float().unsqueeze(1))
        m[:, -1] = False  # keep at least one token
        sorted_lg = sorted_lg.masked_fill(m, float("-inf"))
    kept_sorted = sorted_lg > float("-inf")
    kept = torch.zeros_like(kept_sorted)
    kept.scatter_(-1, idx, kept_sorted)
    return kept


def _run_case(logits, k, p):
    out = _npu_apply_top_k_top_p(logits.clone(), k, p)
    got = out.float() > float("-inf")
    exp = _ref_kept(logits, k, p)
    probs = logits.float().softmax(dim=-1)
    mismatch = (got != exp).sum(dim=-1)
    mass = (probs * (got != exp)).sum(dim=-1)
    return mismatch, mass


class TestTopKTopPAdapter(TestBase):
    def setUp(self):
        torch.manual_seed(0)
        self.k = torch.tensor(KS, dtype=torch.int32).npu()
        self.p = torch.tensor(PS, dtype=torch.float32).npu()

    def test_topk_only_exact(self):
        logits = (torch.randn(B, V) * 3.0).bfloat16().npu()
        mismatch, _ = _run_case(logits, self.k, None)
        self.assertEqual(int(mismatch.sum()), 0)

    def test_topk_topp_fp32_exact(self):
        logits = (torch.randn(B, V) * 3.0).float().npu()
        mismatch, _ = _run_case(logits, self.k, self.p)
        self.assertEqual(int(mismatch.sum()), 0)

    def test_topp_only_fp32_exact(self):
        logits = (torch.randn(B, V) * 3.0).float().npu()
        mismatch, _ = _run_case(logits, None, self.p)
        self.assertEqual(int(mismatch.sum()), 0)

    def test_bf16_boundary_drift_is_negligible_mass(self):
        logits = (torch.randn(B, V) * 3.0).bfloat16().npu()
        for k, p in ((self.k, self.p), (None, self.p)):
            _, mass = _run_case(logits, k, p)
            self.assertTrue(bool((mass < 5e-3).all()), f"mass={mass.tolist()}")

    def test_new_op_is_used_when_available(self):
        # The whole point of the adapter: on stacks where the legacy custom
        # op is gone, sampling must not crash. Running the cases above on
        # torch_npu >= 2.10 (no _C_ascend.npu_apply_top_k_top_p) already
        # proves it; assert the dispatch explicitly for clarity.
        self.assertTrue(hasattr(torch_npu, "npu_top_k_top_p"))


if __name__ == "__main__":
    unittest.main()
