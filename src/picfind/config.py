from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DB_PATH = Path("data") / "picfind.db"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
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
    enable_captioning: bool = True

    def ensure_parent_dirs(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
