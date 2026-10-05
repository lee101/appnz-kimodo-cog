from __future__ import annotations

import base64
from pathlib import Path

from motion_cog.manifest import validate_generation_inputs
from predict import Predictor

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
        _predictor.setup()
    return _predictor


def handler(job, predictor=None):
    inputs = job.get("input") if isinstance(job, dict) else None
    if not isinstance(inputs, dict):
        raise ValueError("job.input must be an object")
    inputs = validate_generation_inputs(inputs)
    output = Path((predictor or get_predictor()).predict(**inputs))
    payload = output.read_bytes()
    return {
        "file": "data:application/zip;base64," + base64.b64encode(payload).decode(),
        "contentType": "application/zip",
        "bytes": len(payload),
    }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
