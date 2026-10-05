from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
import zipfile
from http.server import HTTPServer

import pytest

from library_server import make_handler
from motion_cog.library import MotionLibrary


@pytest.fixture
def library(tmp_path):
    path = tmp_path / 'motions.zip'
    asset = {'id': 'kimodo-wave', 'name': 'Friendly wave', 'sourcePrompt': 'A person waves.',
             'tags': ['wave', 'greeting'], 'assetPath': 'wave/wave.bvh',
             'metadataPath': 'wave/motion.json', 'license': 'NVIDIA Open Model License'}
    metadata = {'skeleton': 'SOMA-77', 'restPose': 'standard-t', 'model': 'Kimodo-SOMA-RP-v1.1',
                'seed': 304, 'loop': True, 'variants': [{'frames': 120, 'fps': 30}]}
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('index.json', json.dumps({'version': 1, 'assets': [asset]}))
        archive.writestr('wave/motion.json', json.dumps(metadata))
        archive.writestr('wave/wave.bvh', 'HIERARCHY\nfixture\n')
    return MotionLibrary(path)


def test_cached_library_is_searchable_and_preserves_provenance(library):
    asset = library.index('friendly wave')['assets'][0]
    assert asset['seed'] == 304
    assert asset['provenance'] == 'kimodo-generated-3d'
    assert asset['loop'] is False
    assert asset['requestedLoop'] is True
    assert library.index('wolf')['count'] == 0
    payload = library.files[asset['assetUrl']]
    assert hashlib.sha256(payload).hexdigest() == asset['sha256']


def test_reuse_is_stable_free_and_never_substitutes_a_new_prompt(library):
    request = {'asset_id': 'kimodo-wave'}
    first = library.reuse(request)
    assert library.reuse(request) == first
    assert first['cost'] == {'new_generation': False, 'gpu_seconds': 0}
    assert first['status'] == 'completed'
    for extra in [{'loop': True}, {'rig_type': 'quadruped'}, {'prompt': 'A wolf bites.'}, {'duration': 4}]:
        with pytest.raises(ValueError):
            library.reuse({**request, **extra})
    with pytest.raises(NotImplementedError):
        library.reuse({**request, 'provider': 'nvidia-ace'})
    with pytest.raises(LookupError):
        library.reuse({'asset_id': 'missing'})


def test_static_export_matches_service_payloads(library, tmp_path):
    library.export(tmp_path)
    index = json.loads((tmp_path / 'motions/kimodo/index.json').read_text())
    assert index == library.index()
    for asset in index['assets']:
        payload = (tmp_path / asset['assetUrl'].removeprefix('/')).read_bytes()
        assert payload == library.files[asset['assetUrl']]


def test_http_health_cached_generation_bounds_and_conditional_assets(library):
    server = HTTPServer(('127.0.0.1', 0), make_handler(library))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'

    def post(body):
        request = urllib.request.Request(base + '/v1/animations/generations', data=body,
                                         headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(request, timeout=3)

    try:
        with urllib.request.urlopen(base + '/health', timeout=3) as response:
            assert json.load(response)['gpu_loaded'] is False
        with post(b'{"asset_id":"kimodo-wave"}') as response:
            assert json.load(response)['cached'] is True
        for payload, status in [(b'[]', 400), (b'{}', 404), (b'x' * 65537, 413),
                                (b'{"provider":"nvidia-ace"}', 501)]:
            with pytest.raises(urllib.error.HTTPError) as error:
                post(payload)
            assert error.value.code == status
        asset = library.assets[0]
        with urllib.request.urlopen(base + asset['assetUrl'], timeout=3) as response:
            etag = response.headers['ETag']
            assert response.read() == library.files[asset['assetUrl']]
        request = urllib.request.Request(base + asset['assetUrl'], headers={'If-None-Match': etag})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=3)
        assert error.value.code == 304
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
