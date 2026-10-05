import pytest

import runpod_handler
from motion_cog.manifest import validate_generation_inputs


@pytest.mark.parametrize('extra', [
    {'duration': float('nan')}, {'duration': float('inf')}, {'duration': 10000},
    {'duration': True}, {'num_samples': 100}, {'num_samples': 1.5},
    {'diffusion_steps': 100000}, {'diffusion_steps': False}, {'seed': -1},
    {'seed': 1.2}, {'prompt': None}, {'prompt': 12}, {'prompt': ' '}, {'unknown': True},
])
def test_invalid_requests_do_not_load_the_gpu_predictor(extra, monkeypatch):
    def forbidden():
        pytest.fail('invalid request attempted a paid GPU cold load')

    monkeypatch.setattr(runpod_handler, 'get_predictor', forbidden)
    with pytest.raises(ValueError):
        runpod_handler.handler({'input': {'prompt': 'A person waves.', **extra}})


def test_bounded_defaults_and_whitespace():
    assert validate_generation_inputs({'prompt': ' A person waves. '}) == {
        'prompt': 'A person waves.', 'duration': 4.0, 'num_samples': 1,
        'seed': 41, 'diffusion_steps': 100,
    }
