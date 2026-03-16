from __future__ import annotations

from pathlib import Path
from typing import Iterator

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from . import db
from .config import IMAGE_EXTENSIONS, Settings
from .embedding import ModelBundle
from .image_io import register_heif_opener  # noqa: F401
from .models import ImageRecord


FLUSH_EVERY = 25


def discover_images(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def index_directory(source: Path, settings: Settings) -> dict[str, int]:
    settings.ensure_parent_dirs()
    models = ModelBundle(settings.clip_model, settings.caption_model, settings.caption_prompt)
    pending_records: list[ImageRecord] = []
    indexed = 0
    skipped = 0
    failed = 0
    interrupted = False

    image_paths = list(discover_images(source))
    total = len(image_paths)

    with db.connect(settings.db_path) as connection:
        try:
            with tqdm(image_paths, total=total, desc="Indexing", unit="image") as progress:
                for path in progress:
                    stat = path.stat()
                    existing = db.get_existing_file_state(connection, path)
                    if existing and existing["size_bytes"] == stat.st_size and existing["modified_time"] == stat.st_mtime:
                        skipped += 1
                        _update_progress(progress, indexed, skipped, failed)
                        continue

                    try:
                        with Image.open(path) as image:
                            rgb_image = image.convert("RGB")
                            caption = ""
                            if settings.enable_captioning:
                                caption = models.generate_caption(rgb_image)
                            embedding = models.image_embedding(rgb_image).tobytes()
                    except (OSError, UnidentifiedImageError):
                        failed += 1
                        _update_progress(progress, indexed, skipped, failed)
                        continue

                    pending_records.append(
                        ImageRecord(
                            path=path,
                            size_bytes=stat.st_size,
                            modified_time=stat.st_mtime,
                            caption=caption,
                            embedding=embedding,
                        )
                    )

                    if len(pending_records) >= FLUSH_EVERY:
                        indexed += db.upsert_images(connection, pending_records)
                        pending_records.clear()

                    _update_progress(progress, indexed + len(pending_records), skipped, failed)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if pending_records:
                indexed += db.upsert_images(connection, pending_records)
                pending_records.clear()

    return {
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "total": total,
        "interrupted": int(interrupted),
        "flush_every": FLUSH_EVERY,
    }


def _update_progress(progress: tqdm, indexed: int, skipped: int, failed: int) -> None:
    progress.set_postfix(indexed=indexed, skipped=skipped, failed=failed)
