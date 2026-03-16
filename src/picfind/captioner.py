from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from . import db
from .config import Settings
from .embedding import ModelBundle
from .image_io import register_heif_opener  # noqa: F401


FLUSH_EVERY = 25


def generate_captions(settings: Settings, overwrite: bool = False) -> dict[str, int]:
    settings.ensure_parent_dirs()
    models = ModelBundle(settings.clip_model, settings.caption_model, settings.caption_prompt)
    pending_rows: list[tuple[Path, str]] = []
    updated = 0
    skipped = 0
    failed = 0
    interrupted = False
    cleared = 0

    with db.connect(settings.db_path) as connection:
        if overwrite:
            cleared = db.clear_captions(connection)
        targets = db.iter_caption_targets(connection, skip_existing=not overwrite)
        total = len(targets)

        try:
            with tqdm(targets, total=total, desc="Captioning", unit="image") as progress:
                for path, existing_caption in progress:
                    if not overwrite and existing_caption:
                        skipped += 1
                        _update_progress(progress, updated, skipped, failed)
                        continue

                    try:
                        with Image.open(path) as image:
                            rgb_image = image.convert("RGB")
                            caption = models.generate_caption(rgb_image)
                    except (FileNotFoundError, OSError, UnidentifiedImageError):
                        failed += 1
                        _update_progress(progress, updated, skipped, failed)
                        continue

                    pending_rows.append((path, caption))
                    if len(pending_rows) >= FLUSH_EVERY:
                        updated += db.update_captions(connection, pending_rows)
                        pending_rows.clear()

                    _update_progress(progress, updated + len(pending_rows), skipped, failed)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if pending_rows:
                updated += db.update_captions(connection, pending_rows)
                pending_rows.clear()

    return {
        "total": total,
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
        "interrupted": int(interrupted),
        "flush_every": FLUSH_EVERY,
        "cleared": cleared,
    }


def _update_progress(progress: tqdm, updated: int, skipped: int, failed: int) -> None:
    progress.set_postfix(updated=updated, skipped=skipped, failed=failed)
