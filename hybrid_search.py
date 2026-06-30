"""
hybrid_search.py — BM25 keyword search + vector similarity search,
fused with Reciprocal Rank Fusion (RRF).

Pure vector search misses exact terms (names, numbers, jargon). BM25 catches
those. RRF combines both rankings without needing to calibrate score scales.

Usage
-----
    index = BM25Index.build(chunk_docs)          # once, at indexing time
    index.save(f"./bm25_cache/{video_id}.pkl")    # persist alongside Chroma
    ...
    index = BM25Index.load(f"./bm25_cache/{video_id}.pkl")
    docs = hybrid_search(query, vectorstore, index, k=10)
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЇїІіЄєҐґ0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi that keeps the source Documents
    alongside the index so we can return real Document objects from search."""

    def __init__(self, docs: list[Document]):
        from rank_bm25 import BM25Okapi  # local import keeps it an optional dep

        self.docs = docs
        self._tokenized = [_tokenize(d.page_content) for d in docs]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    @classmethod
    def build(cls, docs: list[Document]) -> "BM25Index":
        # Don't index the special summary doc — it isn't a retrievable chunk.
        chunk_docs = [d for d in docs if d.metadata.get("type") != "summary"]
        logger.info(f"[bm25] Building index over {len(chunk_docs)} chunks")
        return cls(chunk_docs)

    def search(self, query: str, k: int = 10) -> list[tuple[Document, float]]:
        if not self._bm25 or not self.docs:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: -scores[i])[:k]
        return [(self.docs[i], float(scores[i])) for i in ranked]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.docs, f)

    @classmethod
    def load(cls, path: str) -> Optional["BM25Index"]:
        p = Path(path)
        if not p.exists():
            return None
        with open(p, "rb") as f:
            docs = pickle.load(f)
        return cls(docs)


def _doc_key(d: Document) -> str:
    """Stable identity for RRF fusion — chunk_idx + video_id is unique enough."""
    return f"{d.metadata.get('video_id')}::{d.metadata.get('chunk_idx')}"


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Standard RRF: score(d) = sum(1 / (k + rank_in_list)) across all lists
    the doc appears in. k=60 is the commonly used constant from the original
    RRF paper — it dampens the influence of any single very-high rank.
    """
    scores: dict[str, float] = {}
    lookup: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            lookup.setdefault(key, doc)

    ordered_keys = sorted(scores, key=scores.get, reverse=True)
    return [lookup[key] for key in ordered_keys]


def hybrid_search(
    query: str,
    vectorstore: Chroma,
    bm25_index: Optional[BM25Index],
    k: int = 10,
    vector_k: Optional[int] = None,
    bm25_k: Optional[int] = None,
) -> list[Document]:
    """
    Runs vector similarity search and BM25 search in parallel, fuses with RRF,
    and returns the top-k fused results.

    Falls back to pure vector search if no BM25 index is available (e.g. it
    failed to build, or rank_bm25 isn't installed) — hybrid degrades gracefully.
    """
    vector_k = vector_k or max(k, 10)
    bm25_k = bm25_k or max(k, 10)

    vector_hits = vectorstore.similarity_search(query, k=vector_k)

    if bm25_index is None:
        logger.info("[hybrid] No BM25 index — falling back to vector-only search")
        return vector_hits[:k]

    bm25_hits = [d for d, _ in bm25_index.search(query, k=bm25_k)]

    fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
    logger.info(
        f"[hybrid] vector={len(vector_hits)} bm25={len(bm25_hits)} "
        f"fused={len(fused)} → returning top {k}"
    )
    return fused[:k]
