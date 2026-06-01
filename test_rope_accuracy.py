#!/usr/bin/env python3
"""Compare Ascend RoPE vs PyTorch reference for Gemma4 layer types.
Matches the ACTUAL code path in AscendRotaryEmbedding.forward_oot
and AscendGemma4RotaryEmbedding.forward_oot."""
import torch
import torch_npu
import json

dtype = torch.bfloat16


def compute_cos_sin_cache_gemma4(head_size, rotary_dim, max_pos, base):
    """Replicate Gemma4RotaryEmbedding._compute_inv_freq + base class cache."""
    rope_angles = rotary_dim // 2
    nope_angles = (head_size // 2) - rope_angles
    freq_exponents = torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_size
    inv_freq = 1.0 / (base ** freq_exponents)
    if nope_angles > 0:
        inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles, dtype=torch.float)])

    # Base class cos_sin_cache construction
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, 2*len(inv_freq)] = [max_pos, head_size]
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    # cos_sin_cache = interleaved: [cos0, sin0, cos1, sin1, ...]
    cache = torch.stack((cos, sin), dim=-1).reshape(max_pos, -1)
    # cache: [max_pos, 2*head_size] for Gemma4 (head_size includes nope padding)
    return cache


def compute_cos_sin_cache_standard(head_size, rotary_dim, max_pos, base):
    """Replicate standard RotaryEmbedding._compute_inv_freq + base class cache."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    cache = torch.stack((cos, sin), dim=-1).reshape(max_pos, -1)
    return cache


def apply_rope_cpu(x, positions, cos_sin_cache, rotary_dim, is_neox=True):
    """Reference RoPE - matches RotaryEmbedding.forward_native exactly.
    x: [num_tokens, num_heads, head_size]
    """
    # Only first rotary_dim dimensions get rotated
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    # Get cos/sin from cache
    # cache shape: [max_pos, 2*head_dim_in_cache]
    # We need the first rotary_dim entries
    cache_per_pos = cos_sin_cache.shape[1] // 2  # half the interleaved pairs
    cos_cache = cos_sin_cache[:, 0:cache_per_pos:2]  # every even index is cos
    # Actually cos_sin_cache is interleaved as:
    # [cos0, sin0, cos1, sin1, ..., cosN, sinN]
    # So at position p: cache[p, 0]=cos0, cache[p,1]=sin0, cache[p,2]=cos1, ...
    cos = torch.zeros(x.shape[0], rotary_dim, dtype=dtype)
    sin = torch.zeros(x.shape[0], rotary_dim, dtype=dtype)
    for i in range(rotary_dim):
        cos[:, i] = cos_sin_cache[positions, 2*i]
        sin[:, i] = cos_sin_cache[positions, 2*i+1]

    if is_neox:
        x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]
        x_rotated = torch.cat([-x2, x1], dim=-1)
    else:
        half = rotary_dim // 2
        x1, x2 = x_rot[..., :half], x_rot[..., half:]
        x_rotated = torch.cat([-x2, x1], dim=-1)

    x_rot_out = x_rot * cos.unsqueeze(1) + x_rotated * sin.unsqueeze(1)
    return torch.cat([x_rot_out, x_pass], dim=-1)


def apply_rope_npu(x, positions, cos_sin_cache, head_size, rotary_dim, is_neox=True):
    """Apply RoPE on NPU, matching AscendRotaryEmbedding.forward_oot exactly."""
    x_npu = x.clone().npu()
    pos_npu = positions.clone().npu()
    cache_npu = cos_sin_cache.clone().npu()

    num_tokens = x.shape[0]

    if rotary_dim < head_size:
        # Partial rotation path (matches non-310p rope_forward_oot)
        x_n = x_npu.view(num_tokens, -1, head_size)
        q_rot = x_n[..., :rotary_dim].contiguous().view(num_tokens, -1)
        q_pass = x_n[..., rotary_dim:]

        torch_npu._npu_rotary_embedding(
            pos_npu, q_rot, q_rot.clone(),
            rotary_dim, cache_npu, is_neox,
        )
        q_rot = q_rot.view(num_tokens, -1, rotary_dim)
        x_out = torch.cat((q_rot, q_pass), dim=-1).view(x.shape)
    else:
        # Full rotation path
        x_flat = x_npu.contiguous().view(num_tokens, -1)
        torch_npu._npu_rotary_embedding(
            pos_npu, x_flat, x_flat.clone(),
            head_size, cache_npu, is_neox,
        )
        x_out = x_flat.view_as(x_npu)

    return x_out.cpu()


def test_rope(name, head_size, rotary_dim, max_pos, base, is_neox,
              num_tokens=2048, num_heads=32, gemma4_cache=False):
    """Compare CPU vs NPU RoPE output."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  head_size={head_size}, rotary_dim={rotary_dim}, base={base}")
    print(f"  num_tokens={num_tokens}, num_heads={num_heads}")

    if gemma4_cache:
        cache = compute_cos_sin_cache_gemma4(head_size, rotary_dim, max_pos, base)
    else:
        cache = compute_cos_sin_cache_standard(head_size, rotary_dim, max_pos, base)

    print(f"  cache shape: {cache.shape}")

    torch.manual_seed(42)
    x = torch.randn(num_tokens, num_heads, head_size, dtype=dtype)
    positions = torch.arange(num_tokens, dtype=torch.long)

    # CPU reference
    ref = apply_rope_cpu(x, positions, cache, rotary_dim, is_neox)

    # NPU
    npu_out = apply_rope_npu(x, positions, cache, head_size, rotary_dim, is_neox)

    # Compare
    ref_f = ref.float()
    npu_f = npu_out.float()
    diff = (ref_f - npu_f).abs()

    print(f"  Max error:  {diff.max().item():.10f}")
    print(f"  Mean error: {diff.mean().item():.10f}")
    print(f"  Median error: {diff.median().item():.10f}")
    print(f"  Has NaN: {torch.isnan(npu_f).any().item()}")

    rel_err = (diff.norm() / max(ref_f.norm(), 1e-8)).item()
    print(f"  Relative L2 error: {rel_err:.10f}")

    for atol in [1e-1, 1e-2, 1e-3]:
        close = torch.allclose(ref_f, npu_f, atol=atol, rtol=1e-1)
        print(f"  allclose atol={atol}: {close}")

    return torch.allclose(ref_f, npu_f, atol=1e-2, rtol=1e-1)


if __name__ == "__main__":
    with open('/data/gemma4/gemma-4-31b-it/config.json') as f:
        c = json.load(f)
    tc = c['text_config']
    max_pos = tc['max_position_embeddings']

    print("=" * 60)
    print("Gemma4 31b RoPE: NPU vs CPU Reference (corrected)")

    all_pass = True

    # Test 1: Sliding-256d, full rotation (rotary_dim=256, no partial)
    all_pass &= test_rope(
        "Sliding-256d (standard, full rot)",
        head_size=256, rotary_dim=256, max_pos=max_pos, base=10000.0,
        is_neox=True, gemma4_cache=False,
    )

    # Test 2: Global-512d, partial rotation (rotary_dim=128, head_size=512)
    # This is the Gemma4 proportional RoPE case
    all_pass &= test_rope(
        "Global-512d (proportional, p_r_f=0.25, partial rot)",
        head_size=512, rotary_dim=128, max_pos=max_pos, base=1000000.0,
        is_neox=True, gemma4_cache=True,
    )

    # Test 3: Long sequence (thinking mode)
    all_pass &= test_rope(
        "Global-512d LONG (16K tokens)",
        head_size=512, rotary_dim=128, max_pos=max_pos, base=1000000.0,
        is_neox=True, gemma4_cache=True, num_tokens=16384, num_heads=4,
    )

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL PASS - RoPE is correct")
    else:
        print("FAILURES - RoPE has issues!")
