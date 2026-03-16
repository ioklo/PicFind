from __future__ import annotations

import re
from functools import cached_property

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


_HANGUL_RE = re.compile(r"[가-힣]")


class MultilingualTextEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name, device=self.device)

    def encode(self, text: str) -> np.ndarray:
        vector = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return vector.astype(np.float32)


def contains_hangul(text: str) -> bool:
    return bool(_HANGUL_RE.search(text))
