import json
from pathlib import Path

from runtime import MODEL_ID, MODEL_REVISION, QUALITY_AREAS


def test_model_and_demo_assets_are_revision_pinned():
    assert MODEL_ID == "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers"
    assert len(MODEL_REVISION) == 40
    manifest = json.loads((Path(__file__).parents[1] / "examples.json").read_text())
    assert len(manifest["sources"][0]["revision"]) == 40
    assert manifest["sources"][0]["license"] == "Apache-2.0"
    assert "CC BY-NC-ND" in manifest["excluded_sources"][0]["reason"]


def test_quality_areas_are_cuda_friendly_multiples_of_16():
    for width, height in QUALITY_AREAS.values():
        assert width % 16 == 0
        assert height % 16 == 0
