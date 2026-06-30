"""
reranker.py — Cross-encoder re-ranking of retrieved candidates.

Vector / hybrid search is optimized for recall (cast a wide net). A
cross-encoder jointly encodes (query, candidate) pairs and is much more
precise at judging actual relevance — but too slow to run over a whole
corpus, so we only use it to re-sort a small candidate set (e.g. top 20)
down to the final top-k (e.g. 4) handed to the LLM.

Model is loaded lazily and cached at module level so it survives across
requests within one process (loading it per-call would be far too slow).
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None  # lazy singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"[rerank] Loading cross-encoder {_MODEL_NAME} (CPU)…")
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(
    query: str,
    docs: list[Document],
    top_n: int = 4,
) -> list[Document]:
    """
    Re-scores `docs` against `query` with a cross-encoder and returns the
    top_n highest-scoring documents, sorted best-first.

    Degrades gracefully: if sentence-transformers isn't installed or the
    model fails to load, falls back to returning the first top_n docs
    unchanged (i.e. whatever order the upstream retriever produced).
    """
    if not docs:
        return []
    if len(docs) <= top_n:
        return docs

    try:
        model = _get_model()
    except Exception as e:
        logger.warning(f"[rerank] Cross-encoder unavailable ({e}); skipping rerank")
        return docs[:top_n]

    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.warning(f"[rerank] Prediction failed ({e}); skipping rerank")
        return docs[:top_n]

    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
    logger.info(
        f"[rerank] {len(docs)} candidates → top {top_n} "
        f"(best score={ranked[0][1]:.3f}, worst kept={ranked[top_n-1][1]:.3f})"
    )
    return [d for d, _ in ranked[:top_n]]
