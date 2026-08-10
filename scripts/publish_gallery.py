#!/usr/bin/env python3
"""Publish generated demos and a review manifest to an S3-compatible bucket."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
from datetime import UTC, datetime
from pathlib import Path


def client():
    import boto3

    account = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT", f"https://{account}.r2.cloudflarestorage.com"),
        aws_access_key_id=os.environ["CLOUDFLARE_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["CLOUDFLARE_R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--prefix", default="wan-animate-2/gallery")
    parser.add_argument("--source-url", help="Upstream source shared by these assets")
    parser.add_argument("--source-license", help="SPDX identifier for the source assets")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bucket = os.environ["R2_BUCKET"]
    public_host = os.environ["R2_PUBLIC_HOST"].rstrip("/")
    if "://" not in public_host:
        public_host = "https://" + public_host
    rows = []
    s3 = None if args.dry_run else client()
    for path in args.paths:
        if not path.is_file():
            raise SystemExit(f"not a file: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        key = f"{args.prefix}/{digest[:12]}-{path.name}"
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if s3:
            s3.upload_file(
                str(path),
                bucket,
                key,
                ExtraArgs={"ContentType": content_type, "CacheControl": "public, max-age=31536000, immutable"},
            )
        row = {"name": path.name, "key": key, "url": f"{public_host}/{key}", "sha256": digest}
        if args.source_url:
            row["source_url"] = args.source_url
        if args.source_license:
            row["source_license"] = args.source_license
        rows.append(row)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
        "license_note": "Only publish source media whose license permits this use; generated outputs require human review.",
        "assets": rows,
    }
    manifest_data = json.dumps(manifest, indent=2).encode()
    manifest_key = f"{args.prefix}/manifest.json"
    if s3:
        s3.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=manifest_data,
            ContentType="application/json",
            CacheControl="public, max-age=60",
        )
    print(json.dumps({**manifest, "manifest_url": f"{public_host}/{manifest_key}"}, indent=2))


if __name__ == "__main__":
    main()
