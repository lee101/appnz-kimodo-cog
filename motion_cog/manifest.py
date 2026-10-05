from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

MAX_BATCH_ITEMS = 64
MAX_DURATION_SECONDS = 10.0
SLUG_RE = re.compile(r"[^a-z0-9]+")


def validate_generation_inputs(inputs: dict) -> dict:
    allowed = {"prompt", "duration", "num_samples", "seed", "diffusion_steps"}
    if not isinstance(inputs, dict) or set(inputs) - allowed:
        raise ValueError("unsupported motion inputs")
    prompt = inputs.get("prompt")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 800:
        raise ValueError("prompt must contain 1-800 characters")
    duration = inputs.get("duration", 4.0)
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or not 1 <= duration <= 10):
        raise ValueError("duration must be 1-10 finite seconds")
    result = {"prompt": prompt.strip(), "duration": float(duration)}
    for name, default, minimum, maximum in [("num_samples", 1, 1, 4),
                                           ("seed", 41, 0, 2_147_483_647),
                                           ("diffusion_steps", 100, 20, 150)]:
        value = inputs.get(name, default)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
        result[name] = value
    return result


@dataclass(frozen=True)
class BatchItem:
    id: str
    name: str
    prompt: str
    duration: float
    seed: int
    tags: tuple[str, ...] = ()
    loop: bool = False


def slugify(value: str) -> str:
    return SLUG_RE.sub("-", value.lower()).strip("-")[:64]


def parse_batch_manifest(value: dict) -> list[BatchItem]:
    raw_items = value.get("motions")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("manifest.motions must be a non-empty array")
    if len(raw_items) > MAX_BATCH_ITEMS:
        raise ValueError(f"batch exceeds {MAX_BATCH_ITEMS} motions")

    items: list[BatchItem] = []
    seen: set[str] = set()
    for offset, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"motions[{offset}] must be an object")
        prompt = str(raw.get("prompt", "")).strip()
        if not prompt or len(prompt) > 800:
            raise ValueError(f"motions[{offset}].prompt must contain 1-800 characters")
        name = str(raw.get("name", "")).strip() or prompt[:80]
        item_id = slugify(str(raw.get("id", "")).strip() or name)
        if not item_id or item_id in seen:
            raise ValueError(f"motions[{offset}].id is empty or duplicated")
        duration = float(raw.get("duration", 4.0))
        if not 1.0 <= duration <= MAX_DURATION_SECONDS:
            raise ValueError(f"motions[{offset}].duration must be 1-{MAX_DURATION_SECONDS} seconds")
        seed = int(raw.get("seed", 41 + offset))
        if not 0 <= seed <= 2_147_483_647:
            raise ValueError(f"motions[{offset}].seed is outside the supported range")
        tags_value = raw.get("tags", [])
        if not isinstance(tags_value, list):
            raise ValueError(f"motions[{offset}].tags must be an array")
        tags = tuple(dict.fromkeys(slugify(str(tag)) for tag in tags_value if slugify(str(tag))))[:12]
        items.append(
            BatchItem(
                id=item_id,
                name=name[:100],
                prompt=prompt,
                duration=duration,
                seed=seed,
                tags=tags,
                loop=bool(raw.get("loop", False)),
            )
        )
        seen.add(item_id)
    return items


def load_batch_manifest(path: str | Path) -> list[BatchItem]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return parse_batch_manifest(value)
