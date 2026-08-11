# Wan Animate 2 Cog

An Apache-2.0 Cog and RunPod Serverless adapter for
[`Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers).
It animates a reference character from a raw driving video—no pose or face preprocessing stage.

The runtime is quality-first:

- the upstream distilled 10-step Euler trajectory with CFG disabled is the default;
- TorchAO dynamic FP8 reduces the 30.5 GiB BF16 transformer to an Ada-friendly footprint;
- the remaining encoders and VAE stay BF16;
- native Wan-Animate-2 reference K/V caching is retained;
- repeated prompt embeddings are cached across warm requests;
- PyTorch fused SDPA replaces the optional FlashAttention extension when unavailable, while upstream's compiled sparse FlexAttention remains active;
- VAE slicing/tiling and persistent Hugging Face/TorchInductor caches reduce memory and cold-start work;
- experimental confidence-gated Taylor prediction is available but **off by default** until identity, motion, and frame-quality A/B tests pass.

The default deployment target is one 48 GiB Ada GPU (L40/L40S/RTX 6000 Ada), not an H100. BF16 mode requires CPU offload or a larger GPU.

## Inputs

| Field | Default | Notes |
| --- | --- | --- |
| `image` | required | Reference character image |
| `driving_video` | required | Raw motion/expression video, up to 15 seconds |
| `prompt` | required | Objective character appearance and background caption |
| `quality` | `preview` | `preview`, `balanced`, or `high`; reference aspect ratio is preserved |
| `max_seconds` | `5` | Trims longer driving videos |
| `fps` | `24` | 12, 16, 24, or 30 |
| `frames_per_segment` | `37` | Must be 4n+1; 37 keeps activation memory bounded |
| `steps` | `10` | Upstream distilled quality baseline |
| `preserve_audio` | `true` | Muxes driving audio into the output |
| `cgtaylor` | `false` | Experimental one-step prediction with forced re-anchoring |

## Build and run

```bash
cog build -t wan-animate-cog
cog run \
  -i image=@examples/official-demo1/reference.png \
  -i driving_video=@examples/official-demo1/template.mp4 \
  -i prompt="$(cat examples/official-demo1/prompt.txt)" \
  -i quality=preview -i max_seconds=3 -i seed=42
```

Fetch the pinned upstream Apache-2.0 example first:

```bash
python scripts/fetch_demo.py
```

The same checksummed reference set and provenance manifest are published at
[`wan-animate-2/references/manifest.json`](https://manifoldgenstatic.manifoldgen.com/wan-animate-2/references/manifest.json)
for serverless tests and human review.

RunPod Serverless uses the same image with:

```text
python -u /src/rp_handler.py
```

Recommended environment:

```text
WAN_QUANT=fp8
WAN_CPU_OFFLOAD=true
WAN_MODEL_CACHE=/runpod-volume/huggingface
HF_HOME=/runpod-volume/huggingface
TORCHINDUCTOR_CACHE_DIR=/runpod-volume/torchinductor
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Set workers min to 0, max to 2, and idle timeout to 30 seconds. A persistent network volume is strongly recommended because the pinned model snapshot is about 42.8 GiB.

## Tests

```bash
python -m pytest
ruff check .
```

GPU validation should compare cache-off and cache-on output with the same source image, video, prompt, and seed. Do not call a cache profile quality-preserving based only on latency; inspect identity, hands/feet, motion timing, and every output frame.

## Dataset and media policy

The suggested Full Body TikTok Dancing Kaggle dataset is CC BY-NC-ND 4.0 and is a segmentation-image dataset, not a clean collection of redistributable raw driving videos. This repository intentionally does not download or republish it for a commercial app. `examples.json` pins the upstream Apache-2.0 demo instead. Add further references only with explicit commercial redistribution rights and attribution.

## License

Adapter code is Apache-2.0. The Wan-Animate-2 code and checkpoint are Apache-2.0; input media remains subject to its own license and consent requirements.
