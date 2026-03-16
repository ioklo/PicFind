from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, CLIPModel, CLIPProcessor


class ModelBundle:
    def __init__(self, clip_model_name: str, caption_model_name: str, caption_prompt: str) -> None:
        self.clip_model_name = clip_model_name
        self.caption_model_name = caption_model_name
        self.caption_prompt = caption_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

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
    def caption_processor(self) -> Any:
        _ensure_supported_transformers_version_for_florence()
        return AutoProcessor.from_pretrained(self.caption_model_name, trust_remote_code=True)

    @cached_property
    def caption_model(self) -> Any:
        _ensure_supported_transformers_version_for_florence()
        model = _load_florence_model(self.caption_model_name, self.torch_dtype)
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
        inputs = self.caption_processor(
            text=self.caption_prompt,
            images=image,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device, self.torch_dtype) if hasattr(value, "dtype") and value.dtype.is_floating_point else value.to(self.device)
            for key, value in inputs.items()
        }
        with torch.no_grad():
            output = self.caption_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                do_sample=False,
                num_beams=3,
            )
        generated_text = self.caption_processor.batch_decode(output, skip_special_tokens=False)[0]
        parsed = self.caption_processor.post_process_generation(
            generated_text,
            task=self.caption_prompt,
            image_size=(image.width, image.height),
        )
        caption = parsed.get(self.caption_prompt, "") if isinstance(parsed, dict) else ""
        return str(caption).strip()

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


def _load_florence_model(model_name: str, torch_dtype: torch.dtype) -> Any:
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            use_safetensors=True,
        )
    except OSError:
        pass
    except ValueError as error:
        _raise_if_torch_too_old(error)
        raise

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    except ValueError as error:
        _raise_if_torch_too_old(error)
        raise


def _ensure_supported_transformers_version_for_florence() -> None:
    version = transformers.__version__
    major_minor = tuple(int(part) for part in version.split(".")[:2])
    if major_minor < (4, 49) or major_minor >= (4, 50):
        raise RuntimeError(
            "현재 설치된 transformers 버전은 Florence-2와 호환되지 않습니다. "
            f"현재 버전: {version}. `pip install \"transformers>=4.49,<4.50\"` 로 맞추세요."
        )


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
