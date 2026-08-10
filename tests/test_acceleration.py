from __future__ import annotations

from types import SimpleNamespace

import torch

from acceleration import ConservativeTaylorCache, PromptEmbeddingCache


def test_prompt_cache_reuses_cpu_copy_and_respects_capacity():
    calls = []

    def encode(prompt, device=None, dtype=None, max_sequence_length=512):
        calls.append(prompt)
        return torch.full((1, 2), float(len(calls)), dtype=dtype or torch.float32, device=device)

    pipeline = SimpleNamespace(_get_t5_prompt_embeds=encode)
    PromptEmbeddingCache(pipeline, capacity=1)
    first = pipeline._get_t5_prompt_embeds("same", device="cpu", dtype=torch.float32)
    second = pipeline._get_t5_prompt_embeds("same", device="cpu", dtype=torch.float32)
    assert torch.equal(first, second)
    assert calls == ["same"]

    pipeline._get_t5_prompt_embeds("different", device="cpu", dtype=torch.float32)
    pipeline._get_t5_prompt_embeds("same", device="cpu", dtype=torch.float32)
    assert calls == ["same", "different", "same"]


class LinearTransformer:
    def __init__(self):
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append((kwargs.get("method"), float(kwargs.get("t", torch.tensor([0.0]))[0])))
        if kwargs.get("method") == "forward_ref":
            return None
        value = kwargs["t"].float().reshape(1, 1)
        return [value]


def test_taylor_cache_never_skips_reference_and_reanchors_after_one_prediction():
    transformer = LinearTransformer()
    cache = ConservativeTaylorCache(transformer, threshold=0.001, warmup=3)
    cache.enabled = True

    assert transformer.forward(method="forward_ref", t=torch.tensor([10.0])) is None
    for timestep in (10.0, 9.0, 8.0):
        result = transformer.forward(method="forward_gen", t=torch.tensor([timestep]))
        assert float(result[0][0, 0]) == timestep

    predicted = transformer.forward(method="forward_gen", t=torch.tensor([7.0]))
    assert float(predicted[0][0, 0]) == 7.0
    assert cache.stats.predicted_forwards == 1

    # A prediction always forces a real forward on the next denoise step.
    transformer.forward(method="forward_gen", t=torch.tensor([6.0]))
    assert cache.stats.actual_forwards == 4
    assert len(transformer.calls) == 5  # one ref + four real generation calls


def test_taylor_cache_resets_when_a_new_segment_restarts_timesteps():
    transformer = LinearTransformer()
    cache = ConservativeTaylorCache(transformer, threshold=0.001, warmup=3)
    cache.enabled = True
    for timestep in (10.0, 9.0, 8.0):
        transformer.forward(method="forward_gen", t=torch.tensor([timestep]))
    assert cache.stats.actual_forwards == 3
    transformer.forward(method="forward_gen", t=torch.tensor([10.0]))
    assert cache.stats.actual_forwards == 1
    assert cache.stats.predicted_forwards == 0
