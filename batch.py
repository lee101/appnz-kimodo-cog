from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from motion_cog.manifest import load_batch_manifest
from motion_cog.service import MotionService


def upload_file(path: Path, url: str) -> None:
    request = urllib.request.Request(
        url,
        data=path.read_bytes(),
        method="PUT",
        headers={"content-type": "application/zip"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        if response.status not in {200, 201, 204}:
            raise RuntimeError(f"artifact upload failed with HTTP {response.status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a bounded Kimodo motion batch")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="/workspace/kimodo-motions.zip")
    parser.add_argument("--upload-url", default="")
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1)
    args = parser.parse_args()

    items = load_batch_manifest(args.manifest)
    service = MotionService()
    output = service.generate_archive(
        items,
        archive_path=args.output,
        diffusion_steps=args.diffusion_steps,
        num_samples=args.num_samples,
    )
    if args.upload_url:
        upload_file(output, args.upload_url)
    print(json.dumps({"status": "succeeded", "output": str(output), "motions": len(items)}))


if __name__ == "__main__":
    main()

