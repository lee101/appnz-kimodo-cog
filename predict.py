from __future__ import annotations

from pathlib import Path as FSPath

try:
    from cog import BasePredictor, Input, Path
except ImportError:
    class BasePredictor:
        pass

    def Input(default=None, **_kwargs):
        return default

    Path = FSPath

from motion_cog.manifest import BatchItem, validate_generation_inputs
from motion_cog.service import MotionService


class Predictor(BasePredictor):
    def setup(self):
        self.service = MotionService()

    def predict(
        self,
        prompt: str = Input(description="Natural-language description of one motion"),
        duration: float = Input(default=4.0, ge=1.0, le=10.0),
        num_samples: int = Input(default=1, ge=1, le=4),
        seed: int = Input(default=41, ge=0, le=2_147_483_647),
        diffusion_steps: int = Input(default=100, ge=20, le=150),
    ) -> Path:
        inputs = validate_generation_inputs({"prompt": prompt, "duration": duration,
                                             "num_samples": num_samples, "seed": seed,
                                             "diffusion_steps": diffusion_steps})
        item = BatchItem(
            id="generated-motion",
            name="Generated motion",
            prompt=inputs["prompt"],
            duration=inputs["duration"],
            seed=inputs["seed"],
            tags=("generated",),
        )
        return Path(
            self.service.generate_archive(
                [item],
                num_samples=num_samples,
                diffusion_steps=diffusion_steps,
            )
        )
