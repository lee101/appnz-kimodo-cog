import json

import pytest

from motion_cog.encoder import normalize_tokenizer_metadata
from motion_cog.manifest import load_batch_manifest, parse_batch_manifest, slugify


def test_slugify_is_asset_safe():
    assert slugify("  Warm Laugh! #2 ") == "warm-laugh-2"


def test_repository_manifest_is_valid():
    items = load_batch_manifest("manifests/avatar-core.json")
    assert len(items) == 30
    assert len({item.id for item in items}) == len(items)
    assert {"idle", "walk", "run", "swim", "talk", "attach"} <= {
        tag for item in items for tag in item.tags
    }


def test_manifest_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicated"):
        parse_batch_manifest(
            {
                "motions": [
                    {"id": "same", "prompt": "first"},
                    {"id": "same", "prompt": "second"},
                ]
            }
        )


def test_manifest_roundtrip_shape(tmp_path):
    source = {"motions": [{"id": "wave", "name": "Wave", "prompt": "A person waves.", "tags": ["wave"]}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(source))
    item = load_batch_manifest(path)[0]
    assert item.name == "Wave"
    assert item.duration == 4.0


def test_transformers_five_tokenizer_metadata_is_normalized(tmp_path):
    config = tmp_path / "tokenizer_config.json"
    config.write_text(json.dumps({"tokenizer_class": "TokenizersBackend", "model_max_length": 42}))
    normalize_tokenizer_metadata(tmp_path)
    assert json.loads(config.read_text()) == {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 42,
    }
