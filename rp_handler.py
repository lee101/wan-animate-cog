"""RunPod Serverless entry point using the same persistent runtime as Cog."""

from __future__ import annotations

from pathlib import Path

import runpod

from media import MAX_IMAGE_BYTES, MAX_VIDEO_BYTES, materialize_input, media_artifact, temporary_directory
from runtime import WanAnimateRuntime

_runtime: WanAnimateRuntime | None = None


def _bool_input(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError("boolean inputs must be true or false")


def get_runtime() -> WanAnimateRuntime:
    global _runtime
    if _runtime is None:
        _runtime = WanAnimateRuntime()
    return _runtime


def handler(event):
    values = event.get("input") or {}
    if not isinstance(values, dict):
        raise ValueError("input must be an object")
    with temporary_directory("wan-animate-input-") as temp_name:
        temp = Path(temp_name)
        image = materialize_input(str(values.get("image") or ""), temp, "image", MAX_IMAGE_BYTES)
        driving = materialize_input(
            str(values.get("driving_video") or ""), temp, "driving_video", MAX_VIDEO_BYTES
        )
        result = get_runtime().generate(
            image=image,
            driving_video=driving,
            prompt=str(values.get("prompt") or ""),
            quality=str(values.get("quality") or "preview"),
            max_seconds=float(values.get("max_seconds", 5.0)),
            fps=int(values.get("fps", 24)),
            frames_per_segment=int(values.get("frames_per_segment", 37)),
            steps=int(values.get("steps", 10)),
            seed=int(values["seed"]) if values.get("seed") is not None else None,
            preserve_audio=_bool_input(values.get("preserve_audio"), True),
            cgtaylor=_bool_input(values.get("cgtaylor"), False),
            cgtaylor_threshold=float(values.get("cgtaylor_threshold", 0.015)),
        )
        try:
            return {"outputs": [media_artifact(result.path)], "metrics": result.metrics}
        finally:
            result.path.unlink(missing_ok=True)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
