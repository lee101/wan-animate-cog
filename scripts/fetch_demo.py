#!/usr/bin/env python3
"""Fetch the pinned Apache-2.0 upstream demo without committing media blobs."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "examples.json").read_text())
DESTINATION = ROOT / "examples" / "official-demo1"


def main() -> None:
    source = MANIFEST["sources"][0]
    revision = source["revision"]
    base = f"https://raw.githubusercontent.com/Wan-Video/Wan-Animate-2/{revision}/examples/demo1"
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, expected in source["files"].items():
        request = urllib.request.Request(f"{base}/{name}", headers={"User-Agent": "wan-animate-cog/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise SystemExit(f"checksum mismatch for {name}: {actual}")
        destination = DESTINATION / name
        destination.write_bytes(data)
        print(destination)


if __name__ == "__main__":
    main()
