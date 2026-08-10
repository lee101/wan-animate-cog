import torch

from attention import sdpa_varlen_attention


def test_sdpa_fallback_preserves_upstream_shape_and_zero_pads_short_queries():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 5, 2, 8)
    v = torch.randn(1, 5, 2, 8)
    output = sdpa_varlen_attention(
        q,
        k,
        v,
        q_lens=torch.tensor([3]),
        k_lens=torch.tensor([4]),
        dtype=torch.float32,
    )
    assert output.shape == q.shape
    assert torch.count_nonzero(output[:, 3:]) == 0
    assert torch.isfinite(output[:, :3]).all()
