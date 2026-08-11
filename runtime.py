"""Persistent Wan-Animate-2 inference runtime."""

from __future__ import annotations

import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from acceleration import install_accelerations
from attention import install_attention_fallback
from media import encode_video, mux_driving_audio, normalize_driving_video, temporary_directory

MODEL_ID = os.getenv("WAN_MODEL_ID", "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers")
MODEL_REVISION = os.getenv("WAN_MODEL_REVISION", "36b185201c469c756601cb0779f6597dda1d6c01")

QUALITY_AREAS = {
    "preview": (640, 480),
    "balanced": (800, 640),
    "high": (1280, 720),
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _quantization_config(mode: str):
    if mode == "bf16":
        return None
    from diffusers import TorchAoConfig
    from diffusers.quantizers import PipelineQuantizationConfig

    if mode == "fp8":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
        from torchao.quantization.granularity import PerRow

        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    elif mode == "int8":
        from torchao.quantization import Int8WeightOnlyConfig

        config = Int8WeightOnlyConfig()
    else:
        raise ValueError("WAN_QUANT must be fp8, int8, or bf16")
    return PipelineQuantizationConfig(quant_mapping={"transformer": TorchAoConfig(config)})


@dataclass
class GenerationResult:
    path: Path
    metrics: dict[str, Any]


class WanAnimateRuntime:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Wan-Animate-2 requires a CUDA GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.device = torch.device("cuda")
        requested_quant = os.getenv("WAN_QUANT", "fp8").strip().lower()
        major, minor = torch.cuda.get_device_capability()
        if requested_quant == "fp8" and (major, minor) < (8, 9):
            requested_quant = "int8"
        self.quant = requested_quant
        self.attention_fallback = install_attention_fallback()

        from diffusers import WanAnimate2Pipeline

        cache_dir = os.getenv("WAN_MODEL_CACHE") or os.getenv("HF_HOME")
        self.pipeline = WanAnimate2Pipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            quantization_config=_quantization_config(self.quant),
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        cpu_offload = _env_bool("WAN_CPU_OFFLOAD", self.quant == "bf16")
        if cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)
        if hasattr(self.pipeline.vae, "enable_slicing"):
            self.pipeline.vae.enable_slicing()
        if hasattr(self.pipeline.vae, "enable_tiling"):
            self.pipeline.vae.enable_tiling()

        if _env_bool("WAN_COMPILE", False):
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            self.pipeline.transformer.compile_repeated_blocks(fullgraph=False)

        self.taylor = install_accelerations(self.pipeline)
        self._lock = threading.Lock()

    def generate(
        self,
        *,
        image: Path,
        driving_video: Path,
        prompt: str,
        quality: str = "preview",
        max_seconds: float = 5.0,
        fps: int = 24,
        frames_per_segment: int = 37,
        steps: int = 10,
        seed: int | None = None,
        preserve_audio: bool = True,
        cgtaylor: bool = False,
        cgtaylor_threshold: float = 0.015,
    ) -> GenerationResult:
        if quality not in QUALITY_AREAS:
            raise ValueError(f"quality must be one of {', '.join(QUALITY_AREAS)}")
        if not (1.0 <= float(max_seconds) <= 15.0):
            raise ValueError("max_seconds must be between 1 and 15")
        if fps not in {12, 16, 24, 30}:
            raise ValueError("fps must be 12, 16, 24, or 30")
        if not (17 <= frames_per_segment <= 81) or (frames_per_segment - 1) % 4:
            raise ValueError("frames_per_segment must be 4n+1 between 17 and 81")
        if not (6 <= steps <= 20):
            raise ValueError("steps must be between 6 and 20; distilled quality baseline is 10")
        if not prompt.strip():
            raise ValueError("prompt is required")
        if len(prompt) > 4000:
            raise ValueError("prompt exceeds 4000 characters")
        if image.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("image exceeds 32 MiB")

        selected_seed = seed if seed is not None else random.SystemRandom().randint(0, 2**31 - 1)
        width, height = QUALITY_AREAS[quality]

        with self._lock, temporary_directory() as temp_name:
            temp = Path(temp_name)
            normalized = temp / "driving.mp4"
            source_info = normalize_driving_video(driving_video, normalized, fps, max_seconds)
            reference = Image.open(image)
            if reference.width * reference.height > 40_000_000:
                raise ValueError("image exceeds the 40 megapixel budget")
            reference = reference.convert("RGB")
            output_path = temp / "wan-animate-2.mp4"
            muxed_path = temp / "wan-animate-2-audio.mp4"

            self.taylor.reset()
            self.taylor.threshold = float(cgtaylor_threshold)
            self.taylor.enabled = bool(cgtaylor)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.no_grad():
                output = self.pipeline(
                    image=reference,
                    driving_video=str(normalized),
                    prompt=prompt,
                    height=height,
                    width=width,
                    clip_len=frames_per_segment,
                    first_num=1,
                    fps=fps,
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                    sample_shift=5.0,
                    flow_solver="euler",
                    seed=selected_seed,
                    generator=torch.Generator(device=self.device).manual_seed(selected_seed),
                    output_type="np",
                )
            inference_seconds = time.perf_counter() - started
            encode_video(output.frames[0], output_path, fps=fps, crf=18)
            final_path = mux_driving_audio(output_path, normalized, muxed_path) if preserve_audio else output_path

            # Cog requires the returned file to outlive this request's temp context.
            fd, result_name = tempfile.mkstemp(
                prefix=f"wan-animate-{selected_seed}-",
                suffix=".mp4",
                dir=os.getenv("TMPDIR") or None,
            )
            os.close(fd)
            result_path = Path(result_name)
            result_path.write_bytes(final_path.read_bytes())
            metrics = {
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "seed": selected_seed,
                "quality": quality,
                "fps": fps,
                "steps": steps,
                "frames_per_segment": frames_per_segment,
                "input_seconds": round(float(source_info["duration"]), 3),
                "inference_seconds": round(inference_seconds, 3),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "quant": self.quant,
                "attention": "torch-sdpa" if self.attention_fallback else "flash-attn",
                "cgtaylor": self.taylor.as_dict(),
            }
            return GenerationResult(result_path, metrics)
