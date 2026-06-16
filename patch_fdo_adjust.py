#!/usr/bin/env python3
"""
Monkey-patch for FULL_DECODE_ONLY cudagraph mode on Ascend NPU.

The `adjust_cudagraph_sizes_for_spec_decode` in vllm rounds capture
sizes to multiples of (num_speculative_tokens + 1), producing batch
sizes (e.g. 6, 18, 30, 42) that Ascend's PA/FIA kernels do not
support.  Patching the function to a no-op so that FULL_DECODE_ONLY
uses the original, PA/FIA-compatible capture sizes.

Import this module BEFORE vllm is imported to ensure the patch
takes effect.
"""
import vllm.config.compilation as _comp


_original_adjust = _comp.CompilationConfig.adjust_cudagraph_sizes_for_spec_decode


def _patched_adjust(self, uniform_decode_query_len, tensor_parallel_size):
    """No-op patch: use original capture sizes on Ascend NPU.

    The original function rounds cudagraph_capture_sizes to multiples
    of uniform_decode_query_len (num_spec_tokens + 1).  On Ascend 910B4
    the PA and FIA kernels only support specific batch sizes (the ones
    auto-computed without MTP rounding).  Skipping the rounding lets
    FDO mode capture graphs with sizes that the NPU kernels can handle.
    """
    # Still set max_cudagraph_capture_size if not explicitly set
    if self.max_cudagraph_capture_size is None:
        # Keep the original behaviour but don't round
        pass
    return


_comp.CompilationConfig.adjust_cudagraph_sizes_for_spec_decode = _patched_adjust
print("[MTP-FDO PATCH] adjust_cudagraph_sizes_for_spec_decode monkey-patched to no-op for Ascend NPU",
      flush=True)
