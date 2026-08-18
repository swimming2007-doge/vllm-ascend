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

"""Position-guard regression tests for the Triton RoPE kernels.

Backstory (gemma-4-31b MTP, logs/31b_MTP_0725.log): the draft model's
Q-only RoPE (``gemma4_q_only_rope`` -> ``rope_forward_triton`` ->
``_triton_rope``) indexed ``cos_sin_cache`` with the raw position value.
Spec-decode placeholder/padded rows can carry out-of-range positions
(-1 dummy slots), and the resulting cos/sin row load is an MTE access at
an OOB DDR address that kills the device stream:

    EZ9999: fftsplus aicore error, error code = 0x800000
    The DDR address of the MTE instruction is out of range

Depending on where the fault lands it either crashes the worker
(507011/507035 at the next synchronize) or silently rotates the row with
out-of-cache garbage (observed as 0% MTP acceptance on some requests).

The kernels now clamp the row index into ``[0, num_pos_rows - 1]``. These
tests pin that behaviour down:

* ``neg``    : -1 placeholder              -> reads cache row 0
* ``above``  : position beyond cache rows  -> reads the last cache row
* ``valid``  : in-range positions          -> numerics unchanged vs reference

Both entry points (``rope_forward_triton`` incl. the empty-key Q-only
shape used by the Gemma4 MTP draft, and ``rope_forward_triton_siso``),
both position dtypes, and neox/interleaved styles are covered.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.rope_q_only import gemma4_q_only_rope
from vllm_ascend.ops.triton.rope import rope_forward_triton, rope_forward_triton_siso
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

CACHE_ROWS = 4096
NUM_TOKENS = 33  # odd on purpose: exercises the row-block tail loop
BAD_ROWS = (3, 17, 30)


def _ref_rope(x, cache, positions, rope_dim, is_neox_style, clamp_to=None):
    """Torch reference: rotate ``x`` [T, H, D] against cache rows."""
    pos = positions.long()
    if clamp_to is not None:
        pos = pos.clamp(0, clamp_to)
    half = rope_dim // 2
    cos = cache[pos, :half].float()[:, None, :]
    sin = cache[pos, half:].float()[:, None, :]
    head_dim = x.shape[-1]
    if is_neox_style:
        x1, x2 = x[..., :half].float(), x[..., half:rope_dim].float()
        rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
        rot = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)
    if rope_dim < head_dim:
        rot = torch.cat((rot, x[..., rope_dim:].float()), dim=-1)
    return rot.to(x.dtype)


def _inputs(num_heads, head_dim, rope_dim, pos_dtype, device):
    torch.manual_seed(0)
    q = (0.5 * torch.randn(NUM_TOKENS, num_heads, head_dim, dtype=torch.bfloat16)).to(device)
    cache = torch.randn(CACHE_ROWS, rope_dim, dtype=torch.float32, device=device)
    positions = torch.arange(NUM_TOKENS, dtype=pos_dtype, device=device)
    return q, cache, positions


@pytest.fixture(scope="module", autouse=True)
def _init_triton_device_properties():
    init_device_properties_triton()


@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_heads,head_dim,rope_dim,neox", [
    (8, 256, 256, True),    # gemma-4 MTP draft sliding-attention layer
    (4, 512, 512, True),    # draft full-attention (global head dim) layer
    (8, 128, 64, True),     # partial rotary (rope_dim < head_dim)
    (8, 128, 128, False),   # interleaved (non-neox) style
])
@pytest.mark.parametrize("case", ["neg", "above", "valid"])
def test_rope_forward_triton_position_guard(case, num_heads, head_dim, rope_dim, neox, pos_dtype):
    device = torch.device("npu:0")
    q, cache, positions = _inputs(num_heads, head_dim, rope_dim, pos_dtype, device)
    clamp_to = None
    if case == "neg":
        positions[list(BAD_ROWS)] = -1
        clamp_to = CACHE_ROWS - 1
    elif case == "above":
        positions[list(BAD_ROWS)] = CACHE_ROWS + 10_000_000
        clamp_to = CACHE_ROWS - 1

    k = torch.empty(NUM_TOKENS, 0, head_dim, dtype=q.dtype, device=device)
    out, _ = rope_forward_triton(
        q.clone(), k, cos_sin_cache=cache, positions=positions,
        rope_dim=rope_dim, is_neox_style=neox,
    )
    torch.npu.synchronize()  # surface any aicore fault here

    want = _ref_rope(q, cache, positions, rope_dim, neox, clamp_to=clamp_to)
    torch.testing.assert_close(out.float(), want.float(), atol=2e-2, rtol=2e-2)


def test_rope_forward_triton_siso_position_guard():
    device = torch.device("npu:0")
    num_heads, head_dim, rope_dim = 8, 256, 256
    q, cache, positions = _inputs(num_heads, head_dim, rope_dim, torch.int32, device)
    positions[list(BAD_ROWS)] = -1

    out = rope_forward_triton_siso(
        q.clone(), cos_sin_cache=cache, positions=positions,
        rope_dim=rope_dim, is_neox_style=True,
    )
    torch.npu.synchronize()

    want = _ref_rope(q, cache, positions, rope_dim, True, clamp_to=CACHE_ROWS - 1)
    torch.testing.assert_close(out.float(), want.float(), atol=2e-2, rtol=2e-2)


def test_gemma4_q_only_rope_production_entry():
    """The exact entry the Gemma4 MTP draft model hits at runtime."""
    device = torch.device("npu:0")
    num_heads, head_dim = 8, 256
    q, cache, positions = _inputs(num_heads, head_dim, head_dim, torch.int32, device)
    positions[BAD_ROWS[0]] = -1  # spec-decode dummy slot

    out = gemma4_q_only_rope(
        positions, q.flatten(-2, -1).contiguous().clone(), cache, head_dim, head_dim, True
    ).view(NUM_TOKENS, num_heads, head_dim)
    torch.npu.synchronize()

    want = _ref_rope(q, cache, positions, head_dim, True, clamp_to=CACHE_ROWS - 1)
    torch.testing.assert_close(out.float(), want.float(), atol=2e-2, rtol=2e-2)
