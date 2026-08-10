"""Attention compatibility helpers for Wan-Animate-2.

Upstream currently hard-requires the optional flash-attn extension for its
reference and cross-attention paths. PyTorch SDPA already selects fused CUDA
kernels on Ada/Blackwell, so use it as a portable fallback while retaining the
model's native compiled FlexAttention path for in-context generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa_varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Match upstream's ``flash_attention`` contract with fused torch SDPA."""
    del deterministic
    if window_size != (-1, -1):
        raise ValueError("the SDPA fallback only supports full attention")
    if q_scale is not None:
        q = q * q_scale

    outputs: list[torch.Tensor] = []
    batch = q.shape[0]
    for index in range(batch):
        q_len = int(q_lens[index]) if q_lens is not None else q.shape[1]
        k_len = int(k_lens[index]) if k_lens is not None else k.shape[1]
        qi = q[index : index + 1, :q_len].to(dtype).transpose(1, 2)
        ki = k[index : index + 1, :k_len].to(dtype).transpose(1, 2)
        vi = v[index : index + 1, :k_len].to(dtype).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            qi,
            ki,
            vi,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        ).transpose(1, 2)
        if q_len < q.shape[1]:
            out = F.pad(out, (0, 0, 0, 0, 0, q.shape[1] - q_len))
        outputs.append(out)
    return torch.cat(outputs, dim=0).to(q.dtype)


def install_attention_fallback() -> bool:
    """Install the fallback only when FlashAttention is unavailable."""
    from diffusers.models.transformers import transformer_wan_animate_2 as module

    if module.FLASH_VER is not None:
        return False
    module.flash_attention = sdpa_varlen_attention
    return True
