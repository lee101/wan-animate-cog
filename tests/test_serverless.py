from __future__ import annotations

import base64
from pathlib import Path

import rp_handler
from runtime import GenerationResult


def data_url(mime: str, value: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(value).decode()}"


def test_serverless_boolean_parsing():
    assert rp_handler._bool_input("false", True) is False
    assert rp_handler._bool_input("YES", False) is True


def test_serverless_handler_uses_same_fields_and_returns_video(monkeypatch, tmp_path: Path):
    class Runtime:
        def generate(self, **kwargs):
            assert kwargs["prompt"] == "dance"
            assert kwargs["quality"] == "preview"
            assert kwargs["steps"] == 10
            assert kwargs["image"].exists()
            assert kwargs["driving_video"].exists()
            output = tmp_path / "result.mp4"
            output.write_bytes(b"video")
            return GenerationResult(output, {"seed": 42})

    monkeypatch.setattr(rp_handler, "get_runtime", lambda: Runtime())
    result = rp_handler.handler(
        {
            "input": {
                "image": data_url("image/png", b"image"),
                "driving_video": data_url("video/mp4", b"video"),
                "prompt": "dance",
                "seed": 42,
            }
        }
    )
    assert result["outputs"][0]["content_type"] == "video/mp4"
    assert result["outputs"][0]["data"] == "dmlkZW8="
    assert result["metrics"] == {"seed": 42}
