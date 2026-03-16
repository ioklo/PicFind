from __future__ import annotations

from pathlib import Path

import numpy as np

from . import db
from .config import Settings
from .embedding import ModelBundle
from .models import SearchResult


def search_images(query: str, settings: Settings, limit: int = 5) -> list[SearchResult]:
    with db.connect(settings.db_path) as connection:
        matrix, metadata = db.load_search_matrix(connection)

    if matrix.size == 0:
        return []

    models = ModelBundle(settings.clip_model, settings.caption_model)
    query_vector = models.text_embedding(query)
    scores = matrix @ query_vector
    top_indices = np.argsort(scores)[::-1][:limit]

    results: list[SearchResult] = []
    for index in top_indices:
        path, caption = metadata[int(index)]
        results.append(
            SearchResult(
                path=Path(path),
                caption=caption,
                score=float(scores[int(index)]),
            )
        )
    return results
