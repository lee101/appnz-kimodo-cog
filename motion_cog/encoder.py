from __future__ import annotations

import os

import numpy as np


class QuantizedLLM2VecEncoder:
    """Kimodo-compatible wrapper for the public pre-quantized LLM2Vec export."""

    def __init__(self, base_path: str, adapter_path: str, device: str = "cuda:0") -> None:
        import torch
        from kimodo.model.llm2vec.llm2vec import LLM2Vec

        self._device = device
        self.llm_dim = 4096
        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_path,
            peft_model_name_or_path=adapter_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_hub(cls, repo_id: str, device: str = "cuda:0") -> "QuantizedLLM2VecEncoder":
        from huggingface_hub import snapshot_download

        base_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=os.getenv("HF_HOME") or None,
        )
        return cls(base_path, os.path.join(base_path, "supervised_adapter"), device=device)

    def __call__(self, text: list[str] | str):
        import torch

        is_string = isinstance(text, str)
        texts = [text] if is_string else text
        with torch.inference_mode():
            encoded = self.model.encode(
                texts,
                batch_size=1,
                show_progress_bar=False,
                device=self._device,
            )
        if encoded.shape[-1] != self.llm_dim:
            raise RuntimeError(f"unexpected text embedding width {encoded.shape[-1]}")
        encoded = torch.as_tensor(np.asarray(encoded)[:, None], device=self._device)
        lengths: list[int] | int = [1] * len(texts)
        if is_string:
            return encoded[0], 1
        return encoded, lengths

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.model.device

