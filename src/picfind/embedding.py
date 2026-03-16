from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor, CLIPModel, CLIPProcessor


class ModelBundle:
    def __init__(self, clip_model_name: str, caption_model_name: str) -> None:
        self.clip_model_name = clip_model_name
        self.caption_model_name = caption_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def clip_processor(self) -> CLIPProcessor:
        return CLIPProcessor.from_pretrained(self.clip_model_name, use_fast=False)

    @cached_property
    def clip_model(self) -> CLIPModel:
        model = _load_model(CLIPModel, self.clip_model_name)
        model.to(self.device)
        model.eval()
        return model

    @cached_property
    def caption_processor(self) -> BlipProcessor:
        return BlipProcessor.from_pretrained(self.caption_model_name, use_fast=False)

    @cached_property
    def caption_model(self) -> BlipForConditionalGeneration:
        model = _load_model(BlipForConditionalGeneration, self.caption_model_name)
        model.to(self.device)
        model.eval()
        return model

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        return _normalize(features)

    def text_embedding(self, text: str) -> np.ndarray:
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        return _normalize(features)

    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.caption_processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            output = self.caption_model.generate(**inputs, max_new_tokens=30)
        return self.caption_processor.decode(output[0], skip_special_tokens=True).strip()

    def device_summary(self) -> str:
        if self.device != "cuda":
            return "cpu"
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"cuda ({device_name}, {total_vram_gb:.1f} GB VRAM)"


def _load_model(model_class: type[Any], model_name: str) -> Any:
    try:
        return model_class.from_pretrained(model_name, use_safetensors=True)
    except OSError:
        pass
    except ValueError as error:
        _raise_if_torch_too_old(error)
        raise

    try:
        return model_class.from_pretrained(model_name)
    except ValueError as error:
        _raise_if_torch_too_old(error)
        raise


def _raise_if_torch_too_old(error: ValueError) -> None:
    message = str(error)
    if "require users to upgrade torch to at least v2.6" not in message:
        return
    raise RuntimeError(
        "현재 설치된 torch 버전이 너무 낮아서 Hugging Face 모델을 로드할 수 없습니다. "
        "torch 2.6 이상으로 업그레이드하세요. 예: "
        "pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    ) from error


def _normalize(value: Any) -> np.ndarray:
    tensor = _as_tensor(value)
    array = tensor.detach().cpu().numpy().astype(np.float32)[0]
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "image_embeds") and value.image_embeds is not None:
        return value.image_embeds
    if hasattr(value, "text_embeds") and value.text_embeds is not None:
        return value.text_embeds
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    if hasattr(value, "last_hidden_state") and value.last_hidden_state is not None:
        return value.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported embedding output type: {type(value)!r}")
