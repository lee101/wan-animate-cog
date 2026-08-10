from cog import BaseRunner, Input, Path

from runtime import WanAnimateRuntime


class Runner(BaseRunner):
    def setup(self) -> None:
        self.runtime = WanAnimateRuntime()

    def run(
        self,
        image: Path = Input(description="Reference character image"),
        driving_video: Path = Input(description="Raw driving video; motion, expression, and optional audio are preserved"),
        prompt: str = Input(description="Character appearance and background caption; objective descriptions work best"),
        quality: str = Input(default="preview", choices=["preview", "balanced", "high"]),
        max_seconds: float = Input(default=5.0, ge=1.0, le=15.0),
        fps: int = Input(default=24, choices=[12, 16, 24, 30]),
        frames_per_segment: int = Input(
            default=37,
            ge=17,
            le=81,
            description="Must be 4n+1. 37 lowers activation memory; 81 can improve long-segment continuity.",
        ),
        steps: int = Input(default=10, ge=6, le=20, description="10 is the distilled quality baseline"),
        seed: int | None = Input(default=None, ge=0, le=2147483647),
        preserve_audio: bool = Input(default=True, description="Mux driving-video audio into the generated MP4"),
        cgtaylor: bool = Input(
            default=False,
            description="Experimental conservative denoiser prediction; keep off for exact distilled-model quality",
        ),
        cgtaylor_threshold: float = Input(default=0.015, ge=0.001, le=0.05),
    ) -> Path:
        result = self.runtime.generate(
            image=image,
            driving_video=driving_video,
            prompt=prompt,
            quality=quality,
            max_seconds=max_seconds,
            fps=fps,
            frames_per_segment=frames_per_segment,
            steps=steps,
            seed=seed,
            preserve_audio=preserve_audio,
            cgtaylor=cgtaylor,
            cgtaylor_threshold=cgtaylor_threshold,
        )
        print(result.metrics, flush=True)
        return Path(result.path)
