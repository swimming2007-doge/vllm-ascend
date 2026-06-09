# SPDX-License-Identifier: Apache-2.0
"""Fix BOS token for Gemma4 models.

Gemma4's tokenizer_config.json sets ``add_bos_token: false``, which
prevents the HF tokenizer from prepending <bos> (token id 2) during
``encode()`` even with ``add_special_tokens=True``.

The chat template works around this by including ``<bos>`` as literal
text, but raw completions (``/v1/completions``) do NOT add BOS.
Without the BOS token the model produces repetitive/garbled output
(e.g. "France is France is France is...").

This patch sets ``add_bos_token = True`` on the tokenizer after it
has been wrapped by vLLM's ``get_cached_tokenizer``, so both chat
and completion endpoints produce correct results.
"""

from vllm.logger import init_logger
from vllm.tokenizers.hf import get_cached_tokenizer as _orig_get_cached_tokenizer

logger = init_logger(__name__)

# Model types whose tokenizers need BOS but ship with add_bos_token=False.
_NEEDS_BOS_FIX = frozenset({"gemma4", "gemma4_text"})

_ALREADY_PATCHED = False


def _patch_gemma4_bos():
    global _ALREADY_PATCHED
    if _ALREADY_PATCHED:
        return
    _ALREADY_PATCHED = True

    def _patched_get_cached_tokenizer(tokenizer):
        cached = _orig_get_cached_tokenizer(tokenizer)
        try:
            add_bos = getattr(cached, "add_bos_token", None)
            model_type = getattr(cached, "model_type", None)
            if (
                add_bos is False
                and getattr(cached, "bos_token_id", None) is not None
                and (
                    model_type is None
                    or model_type in _NEEDS_BOS_FIX
                )
            ):
                cached.add_bos_token = True
                logger.info(
                    "gemma4_bos patch: set add_bos_token=True on "
                    "%s (model_type=%s).",
                    type(cached).__name__,
                    model_type,
                )
        except Exception:
            pass
        return cached

    import vllm.tokenizers.hf as _hf

    _hf.get_cached_tokenizer = _patched_get_cached_tokenizer


_patch_gemma4_bos()
