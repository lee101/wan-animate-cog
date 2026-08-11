from pathlib import Path

import numpy as np

from media import encode_video, media_artifact, normalize_driving_video, probe_video, temporary_directory


def test_encode_and_normalize_short_video(tmp_path: Path):
    source = tmp_path / "source.mp4"
    encode_video(
        [
            np.zeros((64, 96, 3), dtype=np.uint8),
            np.full((64, 96, 3), 180, dtype=np.uint8),
            np.full((64, 96, 3), 255, dtype=np.uint8),
        ],
        source,
        fps=12,
    )
    assert probe_video(source)["duration"] > 0

    normalized = tmp_path / "normalized.mp4"
    info = normalize_driving_video(source, normalized, fps=24, max_seconds=1)
    assert normalized.exists()
    assert info["width"] == 96
    assert info["height"] == 64


def test_media_artifact_is_typed_base64(tmp_path: Path):
    output = tmp_path / "result.mp4"
    output.write_bytes(b"video")
    assert media_artifact(output) == {
        "filename": "result.mp4",
        "data": "dmlkZW8=",
        "content_type": "video/mp4",
    }


def test_temporary_directory_creates_configured_root(tmp_path: Path, monkeypatch):
    configured = tmp_path / "network-volume" / "tmp"
    monkeypatch.setenv("TMPDIR", str(configured))
    with temporary_directory("serverless-") as generated:
        assert Path(generated).parent == configured
        assert Path(generated).is_dir()
    assert configured.is_dir()
