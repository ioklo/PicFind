from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = Path("data") / "multilingual.db"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_CAPTION_MODEL = "microsoft/Florence-2-base-ft"
DEFAULT_MULTILINGUAL_TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
DEFAULT_CAPTION_PROMPT = "<MORE_DETAILED_CAPTION>"
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}


@dataclass(slots=True)
class Settings:
    db_path: Path = DEFAULT_DB_PATH
    clip_model: str = DEFAULT_CLIP_MODEL
    caption_model: str = DEFAULT_CAPTION_MODEL
    multilingual_text_model: str = DEFAULT_MULTILINGUAL_TEXT_MODEL
    caption_prompt: str = DEFAULT_CAPTION_PROMPT
    enable_captioning: bool = True
    enable_multilingual_text: bool = True

    def ensure_parent_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
