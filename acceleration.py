"""Conservative, request-scoped accelerations for Wan-Animate-2."""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass
from typing import Any

import torch


class PromptEmbeddingCache:
    """Small CPU LRU for repeat prompts, especially the fixed reference caption."""

    def __init__(self, pipeline: Any, capacity: int = 16):
        self.pipeline = pipeline
        self.capacity = max(1, capacity)
        self._original = pipeline._get_t5_prompt_embeds
        self._cache: collections.OrderedDict[tuple[Any, ...], torch.Tensor] = collections.OrderedDict()
        self._lock = threading.Lock()
        pipeline._get_t5_prompt_embeds = self._get

    def _get(self, prompt, device=None, dtype=None, max_sequence_length=512):
        prompts = (prompt,) if isinstance(prompt, str) else tuple(prompt)
        key = (prompts, str(dtype), int(max_sequence_length))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
        if cached is not None:
            return cached.to(device=device, dtype=dtype, non_blocking=True)

        value = self._original(
            prompt,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
        )
        stored = value.detach().to(device="cpu", copy=True)
        with self._lock:
            self._cache[key] = stored
            self._cache.move_to_end(key)
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return value


@dataclass
class TaylorStats:
    actual_forwards: int = 0
    predicted_forwards: int = 0
    calibration_checks: int = 0
    last_relative_error: float | None = None


class ConservativeTaylorCache:
    """One-step, confidence-gated output prediction for explicit A/B testing.

    The reference pass is never cached because it builds per-request K/V state.
    Generation uses two real timestep/output anchors, validates the extrapolator
    on real calls, and permits at most one predicted call before re-anchoring.
    It is disabled by default until video-level identity/motion evaluation passes.
    """

    def __init__(self, transformer: Any, threshold: float = 0.015, warmup: int = 4):
        self.transformer = transformer
        self.threshold = float(threshold)
        self.warmup = max(3, int(warmup))
        self._original = transformer.forward
        transformer.forward = self._forward
        transformer._wan_taylor_cache = self
        self.enabled = False
        self.reset()

    def reset(self) -> None:
        self.anchors: list[tuple[float, torch.Tensor]] = []
        self.force_anchor = False
        self.stats = TaylorStats()

    @staticmethod
    def _tensor(result: Any) -> torch.Tensor:
        if isinstance(result, list) and result and isinstance(result[0], torch.Tensor):
            return result[0]
        if isinstance(result, tuple) and result and isinstance(result[0], torch.Tensor):
            return result[0]
        if isinstance(result, torch.Tensor):
            return result
        raise TypeError("unsupported transformer output for Taylor cache")

    @staticmethod
    def _wrap_like(result: Any, tensor: torch.Tensor) -> Any:
        if isinstance(result, list):
            return [tensor]
        if isinstance(result, tuple):
            return (tensor,)
        return tensor

    @staticmethod
    def _timestep(kwargs: dict[str, Any]) -> float:
        value = kwargs.get("t")
        if isinstance(value, torch.Tensor):
            return float(value.detach().flatten()[0].cpu())
        return float(value)

    def _predict(self, timestep: float) -> torch.Tensor | None:
        if len(self.anchors) < 2:
            return None
        t0, y0 = self.anchors[-2]
        t1, y1 = self.anchors[-1]
        if t1 == t0:
            return None
        return y1 + (y1 - y0) * ((timestep - t1) / (t1 - t0))

    def _forward(self, *args, **kwargs):
        if not self.enabled or kwargs.get("method") != "forward_gen" or kwargs.get("is_uncondtion", False):
            return self._original(*args, **kwargs)

        timestep = self._timestep(kwargs)
        if self.anchors and timestep >= self.anchors[-1][0]:
            # A new segment restarts the scheduler at high noise.
            self.reset()
            self.enabled = True

        candidate = self._predict(timestep)
        can_skip = (
            candidate is not None
            and self.stats.actual_forwards >= self.warmup
            and self.stats.last_relative_error is not None
            and self.stats.last_relative_error <= self.threshold
            and not self.force_anchor
        )
        if can_skip:
            self.stats.predicted_forwards += 1
            self.force_anchor = True
            return [candidate]

        result = self._original(*args, **kwargs)
        actual = self._tensor(result).detach()
        self.stats.actual_forwards += 1
        if candidate is not None and candidate.shape == actual.shape:
            error = (candidate - actual).abs().mean() / actual.abs().mean().clamp_min(1e-6)
            self.stats.last_relative_error = float(error.cpu())
            self.stats.calibration_checks += 1
        else:
            self.stats.last_relative_error = None
        self.anchors.append((timestep, actual.clone()))
        self.anchors = self.anchors[-2:]
        self.force_anchor = False
        return result

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "actual_forwards": self.stats.actual_forwards,
            "predicted_forwards": self.stats.predicted_forwards,
            "calibration_checks": self.stats.calibration_checks,
            "last_relative_error": self.stats.last_relative_error,
        }


def install_accelerations(pipeline: Any) -> ConservativeTaylorCache:
    PromptEmbeddingCache(pipeline)
    cache = ConservativeTaylorCache(pipeline.transformer)
    return cache
