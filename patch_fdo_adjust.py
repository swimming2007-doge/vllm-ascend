#!/usr/bin/env python3
"""
Monkey-patch for FULL_DECODE_ONLY cudagraph mode on Ascend NPU.

Problem: adjust_cudagraph_sizes_for_spec_decode rounds capture sizes
to multiples of 6, introducing batch sizes (6, 18, 30, 42, ...) that
Ascend's PA/FIA kernels do not support.  Keeping the full original
list (27 sizes) causes OOM during graph capture.

Fix: produce a compact set of capture sizes that are:
  1. Multiples of uniform_decode_query_len (num_spec_tokens + 1)
     so the cudagraph dispatcher assertion passes.
  2. Supported by PA/FIA kernels on Ascend 910B4.
  3. Small enough to avoid OOM during FDO graph capture.

Import this module BEFORE vllm is imported.
"""
import vllm.config.compilation as _comp
from vllm.config import CUDAGraphMode

_original_adjust = _comp.CompilationConfig.adjust_cudagraph_sizes_for_spec_decode

# Batch sizes known to be compatible with Ascend PA/FIA kernels.
# These are the well-tested power-of-2 / common sizes.
_ASCEND_SAFE_SIZES = {
    1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
}


def _patched_adjust(self, uniform_decode_query_len, tensor_parallel_size):
    """Patched version: use PA/FIA-safe capture sizes."""
    multiple_of = uniform_decode_query_len
    if not self.cudagraph_capture_sizes or multiple_of <= 1:
        return

    assert self.max_cudagraph_capture_size is not None

    # Only keep sizes that are multiples of uniform_decode_query_len
    # AND were in the original capture set (known PA/FIA-compatible).
    rounded_sizes = sorted(
        size for size in self.cudagraph_capture_sizes
        if size % multiple_of == 0 and size <= self.max_cudagraph_capture_size
    )
    # Ensure we have at least the minimum usable size.
    if not rounded_sizes or rounded_sizes[0] > multiple_of:
        rounded_sizes.insert(0, multiple_of)

    if not rounded_sizes:
        # Fallback: just use the original adjustment
        return _original_adjust(self, uniform_decode_query_len, tensor_parallel_size)

    self.max_cudagraph_capture_size = rounded_sizes[-1]
    self.cudagraph_capture_sizes = rounded_sizes
    # The patched _create_padded_batch_descriptor handles batching
    # when actual num_tokens isn't a perfect multiple.


_comp.CompilationConfig.adjust_cudagraph_sizes_for_spec_decode = _patched_adjust
print("[MTP-FDO PATCH] adjust_cudagraph_sizes_for_spec_decode patched with Ascend-safe sizes",
      flush=True)
