from __future__ import annotations

import json
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

from .encoder import QuantizedLLM2VecEncoder
from .manifest import BatchItem

DEFAULT_MODEL = "Kimodo-SOMA-RP-v1.1"
DEFAULT_ENCODER = "matbee/kimodo-llm2vec-nf4"


class MotionService:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        encoder_repo: str = DEFAULT_ENCODER,
        device: str = "cuda:0",
    ) -> None:
        import torch
        from kimodo import load_model

        if not torch.cuda.is_available():
            raise RuntimeError("Kimodo Cog requires a CUDA GPU")
        self.device = device
        self.model_name = model_name
        self.encoder_repo = encoder_repo
        encoder = QuantizedLLM2VecEncoder.from_hub(encoder_repo, device=device)
        self.model, self.resolved_model = load_model(
            model_name,
            device=device,
            return_resolved_name=True,
            text_encoder=encoder,
        )

    def generate_item(
        self,
        item: BatchItem,
        output_dir: Path,
        *,
        num_samples: int = 1,
        diffusion_steps: int = 100,
    ) -> list[dict]:
        import torch
        from kimodo.exports.bvh import save_motion_bvh
        from kimodo.exports.motion_io import save_kimodo_npz
        from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots
        from kimodo.tools import seed_everything

        seed_everything(item.seed)
        frames = int(round(item.duration * float(self.model.fps)))
        started = time.time()
        with torch.inference_mode():
            output = self.model(
                [item.prompt],
                [frames],
                constraint_lst=[],
                num_denoising_steps=diffusion_steps,
                num_samples=num_samples,
                multi_prompt=True,
                num_transition_frames=5,
                post_processing=True,
                return_numpy=True,
            )
        item_dir = output_dir / item.id
        item_dir.mkdir(parents=True, exist_ok=False)
        skeleton = self.model.skeleton
        if isinstance(skeleton, SOMASkeleton30):
            skeleton = skeleton.somaskel77.to(self.device)

        variants: list[dict] = []
        for sample in range(num_samples):
            suffix = f"-{sample + 1:02d}" if num_samples > 1 else ""
            stem = f"{item.id}{suffix}"
            single = {
                key: (
                    value[sample]
                    if hasattr(value, "shape") and len(value.shape) and value.shape[0] == num_samples
                    else value
                )
                for key, value in output.items()
            }
            npz_path = item_dir / f"{stem}.npz"
            bvh_path = item_dir / f"{stem}.bvh"
            save_kimodo_npz(str(npz_path), single)
            joints_pos = torch.from_numpy(output["posed_joints"][sample]).to(self.device)
            joints_rot = torch.from_numpy(output["global_rot_mats"][sample]).to(self.device)
            local_rotations = global_rots_to_local_rots(joints_rot, skeleton)
            root_positions = joints_pos[:, skeleton.root_idx, :]
            save_motion_bvh(
                str(bvh_path),
                local_rotations,
                root_positions,
                skeleton=skeleton,
                fps=self.model.fps,
                standard_tpose=True,
            )
            variants.append(
                {
                    "id": stem,
                    "bvh": bvh_path.name,
                    "npz": npz_path.name,
                    "frames": frames,
                    "fps": float(self.model.fps),
                }
            )

        metadata = {
            **asdict(item),
            "tags": list(item.tags),
            "model": self.model_name,
            "resolvedModel": self.resolved_model,
            "textEncoder": self.encoder_repo,
            "diffusionSteps": diffusion_steps,
            "generationSeconds": round(time.time() - started, 3),
            "skeleton": "SOMA-77",
            "restPose": "standard-t",
            "variants": variants,
        }
        (item_dir / "motion.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return variants

    def generate_archive(
        self,
        items: list[BatchItem],
        *,
        num_samples: int = 1,
        diffusion_steps: int = 100,
        archive_path: str | Path | None = None,
    ) -> Path:
        if not 1 <= num_samples <= 4:
            raise ValueError("num_samples must be 1-4")
        if not 20 <= diffusion_steps <= 150:
            raise ValueError("diffusion_steps must be 20-150")
        work = Path(tempfile.mkdtemp(prefix="appnz-kimodo-"))
        output_dir = work / "motions"
        output_dir.mkdir()
        try:
            for item in items:
                self.generate_item(
                    item,
                    output_dir,
                    num_samples=num_samples,
                    diffusion_steps=diffusion_steps,
                )
            index = build_index(items, output_dir)
            (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
            destination = Path(archive_path) if archive_path else work / "kimodo-motions.zip"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(output_dir.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(output_dir))
            if archive_path:
                shutil.rmtree(work)
            return destination
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise


def build_index(items: list[BatchItem], output_dir: Path) -> dict:
    assets = []
    for item in items:
        metadata = json.loads((output_dir / item.id / "motion.json").read_text(encoding="utf-8"))
        assets.append(
            {
                "id": f"kimodo-{item.id}",
                "name": item.name,
                "kind": "motion",
                "description": item.prompt,
                "tags": [*item.tags, "kimodo", "soma77"],
                "format": "standard-T BVH + Kimodo NPZ",
                "license": "Generated with Kimodo SOMA-RP under NVIDIA Open Model License",
                "sourceUrl": "https://github.com/lee101/appnz-kimodo-cog",
                "runtime": "gpu",
                "sourcePrompt": item.prompt,
                "assetPath": f"{item.id}/{metadata['variants'][0]['bvh']}",
                "metadataPath": f"{item.id}/motion.json",
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
    return {"version": 1, "generator": DEFAULT_MODEL, "assets": assets}
