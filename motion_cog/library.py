from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

DEFAULT_ARCHIVE = Path(__file__).resolve().parents[1] / 'outputs' / 'kimodo-avatar-core.zip'


class MotionLibrary:
    """Serve existing generated candidates without loading weights or renting a GPU."""

    def __init__(self, archive_path: Path = DEFAULT_ARCHIVE) -> None:
        self.assets: list[dict] = []
        self.files: dict[str, bytes] = {}
        with zipfile.ZipFile(archive_path) as archive:
            if sum(item.file_size for item in archive.infolist()) > 256 << 20:
                raise ValueError('motion archive exceeds 256 MiB')
            index = json.loads(archive.read('index.json'))
            if index.get('version') != 1 or not 1 <= len(index.get('assets', [])) <= 200:
                raise ValueError('invalid motion index')
            seen: set[str] = set()
            for asset in index['assets']:
                motion_id = asset.get('id', '')
                if not re.fullmatch(r'[a-z0-9-]{1,100}', motion_id) or motion_id in seen:
                    raise ValueError('invalid or duplicate motion ID')
                seen.add(motion_id)
                metadata = json.loads(archive.read(asset['metadataPath']))
                if metadata.get('skeleton') != 'SOMA-77' or metadata.get('restPose') != 'standard-t':
                    raise ValueError('only standard-T SOMA-77 motions are supported')
                variant = metadata['variants'][0]
                payload = archive.read(asset['assetPath'])
                if len(payload) > 4 << 20 or not payload.startswith(b'HIERARCHY'):
                    raise ValueError('invalid or oversized BVH')
                digest = hashlib.sha256(payload).hexdigest()
                path = f'/motions/kimodo/{motion_id}-{digest[:16]}.bvh'
                self.files[path] = payload
                self.assets.append({
                    **{key: value for key, value in asset.items()
                       if key not in {'assetPath', 'metadataPath', 'assetUrl'}},
                    'assetUrl': path, 'sha256': digest, 'model': metadata['model'],
                    'skeleton': metadata['skeleton'], 'restPose': metadata['restPose'],
                    'seed': metadata['seed'], 'fps': variant['fps'], 'frames': variant['frames'],
                    'duration': (variant['frames'] - 1) / variant['fps'],
                    'provenance': 'kimodo-generated-3d', 'rig': 'humanoid',
                    'loop': False, 'requestedLoop': bool(metadata.get('loop')),
                    'reviewRequired': True,
                })

    def index(self, query: str = '') -> dict:
        terms = query.lower().split()
        assets = [asset for asset in self.assets if all(
            term in ' '.join([asset['id'], asset['name'], asset['sourcePrompt'], *asset['tags']]).lower()
            for term in terms
        )]
        return {'version': 1, 'generator': 'Kimodo-SOMA-RP-v1.1', 'count': len(assets),
                'newGenerationEnabled': False, 'assets': assets}

    def reuse(self, request: dict) -> dict:
        if request.get('provider', 'kimodo-library') != 'kimodo-library':
            raise NotImplementedError('Fresh generation uses the existing Kimodo Cog, not NVIDIA ACE')
        if set(request) - {'provider', 'asset_id', 'rig_type', 'loop', 'prompt'}:
            raise ValueError('unsupported reuse parameters; existing clips are immutable')
        if request.get('rig_type', 'biped') not in {'biped', 'humanoid'}:
            raise ValueError('Kimodo is humanoid motion, not quadruped wolf capture')
        if request.get('loop', False):
            raise ValueError('Looping requires seam and target-avatar contact review')
        asset = next((asset for asset in self.assets if asset['id'] == request.get('asset_id')), None)
        if asset is None:
            raise LookupError('Select an asset_id from /v1/animations/library')
        if request.get('prompt') and request['prompt'].strip() != asset['sourcePrompt']:
            raise ValueError('A new prompt cannot silently reuse a different generated action')
        return {'id': asset['sha256'], 'status': 'completed', 'cached': True, 'asset': asset,
                'cost': {'new_generation': False, 'gpu_seconds': 0},
                'limitations': ['in-place VRM preview requires retargeting', 'review feet and loop seams']}

    def export(self, destination: Path) -> None:
        for path, payload in self.files.items():
            target = destination / path.removeprefix('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        target = destination / 'motions' / 'kimodo' / 'index.json'
        target.write_text(json.dumps(self.index(), indent=2) + '\n', encoding='utf-8')
