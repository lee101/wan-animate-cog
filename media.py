"""Bounded media IO shared by Cog and RunPod Serverless."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import json
import mimetypes
import os
import socket
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import av
import numpy as np

MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_VIDEO_BYTES = 256 * 1024 * 1024


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown ffmpeg error"
        raise RuntimeError(detail)


def probe_video(path: Path) -> dict[str, float | int | bool]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("driving_video is not a readable video")
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("driving_video has no video stream")
    duration = float((payload.get("format") or {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError("driving_video duration is unavailable")
    return {
        "duration": duration,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "has_audio": any(stream.get("codec_type") == "audio" for stream in streams),
    }


def normalize_driving_video(source: Path, destination: Path, fps: int, max_seconds: float) -> dict[str, float | int | bool]:
    if source.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError(f"driving_video exceeds {MAX_VIDEO_BYTES // (1024 * 1024)} MiB")
    info = probe_video(source)
    if int(info["width"]) * int(info["height"]) > 4096 * 4096:
        raise ValueError("driving_video dimensions exceed the 4096x4096 pixel budget")
    duration = min(float(info["duration"]), float(max_seconds))
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"fps={fps}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(destination),
        ]
    )
    normalized = probe_video(destination)
    normalized["source_duration"] = float(info["duration"])
    return normalized


def encode_video(frames: Iterable, destination: Path, fps: int, crf: int = 18) -> None:
    arrays = []
    for frame in frames:
        if hasattr(frame, "convert"):
            frame = np.asarray(frame.convert("RGB"))
        else:
            frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
        arrays.append(frame)
    if not arrays:
        raise RuntimeError("model produced no frames")
    height, width = arrays[0].shape[:2]
    container = av.open(str(destination), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "fast"}
    for array in arrays:
        for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def mux_driving_audio(video: Path, driving_video: Path, destination: Path) -> Path:
    info = probe_video(driving_video)
    if not info["has_audio"]:
        return video
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-i",
            str(driving_video),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(destination),
        ]
    )
    return destination


def _validate_public_https(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("media URLs must use public HTTPS")
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or 443):
        if not ipaddress.ip_address(result[4][0]).is_global:
            raise ValueError("media URL resolves to a private or reserved address")


class _PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_https(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def materialize_input(value: str, directory: Path, name: str, max_bytes: int) -> Path:
    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError(f"{name} data URL must be base64 encoded")
        try:
            data = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ValueError(f"{name} contains invalid base64") from exc
        if len(data) > max_bytes:
            raise ValueError(f"{name} exceeds {max_bytes} bytes")
        mime = header[5:].split(";", 1)[0]
        suffix = mimetypes.guess_extension(mime) or (".mp4" if name == "driving_video" else ".png")
        path = directory / f"{name}{suffix}"
        path.write_bytes(data)
        return path

    _validate_public_https(value)
    request = urllib.request.Request(value, headers={"User-Agent": "wan-animate-cog/0.1"})
    opener = urllib.request.build_opener(_PublicRedirectHandler())
    with opener.open(request, timeout=120) as response:
        length = int(response.headers.get("Content-Length", "0") or 0)
        if length > max_bytes:
            raise ValueError(f"{name} exceeds {max_bytes} bytes")
        data = response.read(max_bytes + 1)
        final_url = response.geturl()
    _validate_public_https(final_url)
    if len(data) > max_bytes:
        raise ValueError(f"{name} exceeds {max_bytes} bytes")
    suffix = Path(urllib.parse.urlparse(final_url).path).suffix or (".mp4" if name == "driving_video" else ".png")
    path = directory / f"{name}{suffix}"
    path.write_bytes(data)
    return path


def media_artifact(path: Path) -> dict[str, str]:
    return {
        "filename": path.name,
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        "content_type": "video/mp4",
    }


def temporary_directory(prefix: str = "wan-animate-"):
    configured = os.getenv("TMPDIR")
    if configured:
        Path(configured).mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=configured or None)
