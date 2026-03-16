from __future__ import annotations

import numpy as np

from . import db
from .config import DEFAULT_CLIP_MODEL, Settings
from .embedding import ModelBundle
from .models import SearchResponse, SearchResult
from .query_text import MultilingualTextEmbedder, contains_hangul


def search_images(query: str, settings: Settings, limit: int = 5) -> SearchResponse:
    original_query = query.strip()
    effective_query = original_query
    embedding_mode = "clip"

    if (
        settings.enable_multilingual_text
        and contains_hangul(original_query)
        and settings.clip_model == DEFAULT_CLIP_MODEL
    ):
        embedder = MultilingualTextEmbedder(settings.multilingual_text_model)
        query_vector = embedder.encode(original_query)
        embedding_mode = "multilingual_clip_text"
    else:
        models = ModelBundle(settings.clip_model, settings.caption_model)
        query_vector = models.text_embedding(effective_query)

    with db.connect(settings.db_path) as connection:
        matrix, metadata = db.load_search_matrix(connection)

    if matrix.size == 0:
        return SearchResponse(
            original_query=original_query,
            effective_query=effective_query,
            embedding_mode=embedding_mode,
            results=[],
        )

    scores = matrix @ query_vector
    top_indices = np.argsort(scores)[::-1][:limit]

    results: list[SearchResult] = []
    for index in top_indices:
        path, caption = metadata[int(index)]
        results.append(
            SearchResult(
                path=path,
                caption=caption,
                score=float(scores[int(index)]),
            )
        )

    return SearchResponse(
        original_query=original_query,
        effective_query=effective_query,
        embedding_mode=embedding_mode,
        results=results,
    )
