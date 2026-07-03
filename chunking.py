"""
chunking.py — Pluggable chunking strategies for YouTube RAG bot.

Three strategies (set CHUNK_STRATEGY in .env):
  sentence   — sentence-boundary-aware grouping (~300 tokens, cleaner retrieval)
  timestamp  — transcript-segment-aware; stores start_time in metadata for deep links
  semantic   — embedding-distance topic-shift detection (most accurate, slowest)

All strategies return List[Document] and are drop-in replacements for the old
split_into_chunks() + index_transcript() pair.

FIXED vs. original:
  - `build_documents`'s default argument was
        config: ChunkingConfig = field(default_factory=ChunkingConfig)
    `field()` is a dataclass-field descriptor, not a value — using it as a
    plain function default doesn't construct a ChunkingConfig; it leaves the
    parameter bound to a Field object, which would blow up the moment any
    code tried to read e.g. `config.strategy` without explicitly passing
    `config=`. Fixed to `config: Optional[ChunkingConfig] = None` with a
    `config = config or ChunkingConfig()` inside the function body.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# ── Tokenizer ──────────────────────────────────────────────────────────────────

def _get_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = _get_encoder()
    return len(enc.encode(text)) if enc else len(text) // 4


# ── Transcript segment dataclass ───────────────────────────────────────────────

@dataclass
class Segment:
    """One raw segment returned by youtube-transcript-api."""
    text: str
    start: float          # seconds from video start
    duration: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


def segments_from_fetched(fetched) -> list[Segment]:
    """Convert youtube-transcript-api FetchedTranscript → List[Segment]."""
    return [Segment(text=e.text, start=e.start, duration=e.duration) for e in fetched]


def format_timestamp(seconds: float) -> str:
    """Convert float seconds → 'HH:MM:SS' or 'MM:SS' string."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def youtube_deep_link(video_id: str, start_seconds: float) -> str:
    return f"https://youtu.be/{video_id}?t={int(start_seconds)}"


# ── Strategy 1: Sentence-aware chunking ───────────────────────────────────────

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _split_into_sentences(text: str) -> list[str]:
    """Naive but fast sentence splitter (handles Mr./Dr. reasonably well)."""
    protected = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|avg)\.\s',
                       r'\1<DOT> ', text)
    parts = _SENTENCE_END.split(protected)
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


def sentence_aware_chunks(
    text: str,
    video_id: str,
    lang: str,
    chunk_tokens: int = 300,
    overlap_sentences: int = 1,
) -> list[Document]:
    """
    Groups whole sentences into windows of <= chunk_tokens tokens.
    Adjacent chunks share `overlap_sentences` sentences for context continuity.
    """
    sentences = _split_into_sentences(text)
    chunks: list[Document] = []
    i = 0
    chunk_idx = 0

    while i < len(sentences):
        window: list[str] = []
        token_count = 0

        for j in range(i, len(sentences)):
            s_tokens = count_tokens(sentences[j])
            if token_count + s_tokens > chunk_tokens and window:
                break
            window.append(sentences[j])
            token_count += s_tokens

        chunk_text = " ".join(window)
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "video_id":    video_id,
                "lang":        lang,
                "chunk_idx":   chunk_idx,
                "strategy":    "sentence",
                "type":        "chunk",
            },
        ))

        advance = max(1, len(window) - overlap_sentences)
        i += advance
        chunk_idx += 1

    logger.info(f"[sentence] {len(chunks)} chunks from {len(sentences)} sentences")
    return chunks


# ── Strategy 2: Timestamp-aware chunking ──────────────────────────────────────

def timestamp_aware_chunks(
    segments: list[Segment],
    video_id: str,
    lang: str,
    chunk_tokens: int = 300,
    overlap_tokens: int = 30,
) -> list[Document]:
    """
    Groups raw transcript segments into chunks while preserving start_time.
    """
    chunks: list[Document] = []
    i = 0
    prev_tail = ""
    chunk_idx = 0

    while i < len(segments):
        window_segs: list[Segment] = []
        token_count = count_tokens(prev_tail)

        for j in range(i, len(segments)):
            s_tokens = count_tokens(segments[j].text)
            if token_count + s_tokens > chunk_tokens and window_segs:
                break
            window_segs.append(segments[j])
            token_count += s_tokens

        chunk_text = (prev_tail + " " + " ".join(s.text for s in window_segs)).strip()
        start_sec   = window_segs[0].start
        end_sec     = window_segs[-1].end

        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "video_id":       video_id,
                "lang":           lang,
                "chunk_idx":      chunk_idx,
                "strategy":       "timestamp",
                "type":           "chunk",
                "start_time":     start_sec,
                "end_time":       end_sec,
                "timestamp_label": format_timestamp(start_sec),
                "deep_link":      youtube_deep_link(video_id, start_sec),
            },
        ))

        full_text = " ".join(s.text for s in window_segs)
        enc = _get_encoder()
        if enc:
            toks = enc.encode(full_text)
            prev_tail = enc.decode(toks[-overlap_tokens:]) if len(toks) > overlap_tokens else full_text
        else:
            prev_tail = full_text[-(overlap_tokens * 4):]

        i += len(window_segs)
        chunk_idx += 1

    logger.info(f"[timestamp] {len(chunks)} chunks from {len(segments)} segments")
    return chunks


# ── Strategy 3: Semantic chunking ─────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    return dot / (mag_a * mag_b + 1e-10)


def semantic_chunks(
    text: str,
    video_id: str,
    lang: str,
    embeddings: Embeddings,
    chunk_tokens: int = 300,
    similarity_threshold: float = 0.75,
    min_sentences_per_chunk: int = 3,
) -> list[Document]:
    """
    Embeds every sentence, computes pairwise cosine similarity between adjacent
    sentences, and splits at topic-shift valleys (similarity < threshold).
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    logger.info(f"[semantic] Embedding {len(sentences)} sentences — may take a moment…")
    sentence_embeddings = embeddings.embed_documents(sentences)

    similarities = [
        _cosine_similarity(sentence_embeddings[k], sentence_embeddings[k + 1])
        for k in range(len(sentences) - 1)
    ]

    split_indices: set[int] = {0}
    for k, sim in enumerate(similarities):
        if sim < similarity_threshold:
            split_indices.add(k + 1)
    split_indices.add(len(sentences))
    splits = sorted(split_indices)

    raw_groups: list[list[str]] = []
    for a, b in zip(splits, splits[1:]):
        raw_groups.append(sentences[a:b])

    merged_groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for group in raw_groups:
        group_tokens = sum(count_tokens(s) for s in group)

        if (current and
                len(current) >= min_sentences_per_chunk and
                current_tokens + group_tokens > chunk_tokens):
            merged_groups.append(current)
            current = list(group)
            current_tokens = group_tokens
        else:
            current.extend(group)
            current_tokens += group_tokens

    if current:
        merged_groups.append(current)

    docs = [
        Document(
            page_content=" ".join(grp),
            metadata={
                "video_id":  video_id,
                "lang":      lang,
                "chunk_idx": idx,
                "strategy":  "semantic",
                "type":      "chunk",
            },
        )
        for idx, grp in enumerate(merged_groups)
    ]

    logger.info(f"[semantic] {len(docs)} topic-coherent chunks")
    return docs


# ── Unified entry point ────────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    strategy: str = "timestamp"       # "sentence" | "timestamp" | "semantic"
    chunk_tokens: int = 300
    overlap_tokens: int = 30          # used by timestamp strategy
    overlap_sentences: int = 1        # used by sentence strategy
    similarity_threshold: float = 0.75  # used by semantic strategy
    min_sentences_per_chunk: int = 3  # used by semantic strategy


def build_documents(
    *,
    video_id: str,
    lang: str,
    text: str,                                  # full plain text (always required)
    segments: Optional[list[Segment]] = None,   # required for timestamp strategy
    embeddings: Optional[Embeddings] = None,    # required for semantic strategy
    config: Optional[ChunkingConfig] = None,
) -> list[Document]:
    """
    Unified entry point. Returns a list of LangChain Documents ready for ChromaDB.
    """
    config = config or ChunkingConfig()
    strategy = config.strategy.lower()

    if strategy == "sentence":
        return sentence_aware_chunks(
            text=text,
            video_id=video_id,
            lang=lang,
            chunk_tokens=config.chunk_tokens,
            overlap_sentences=config.overlap_sentences,
        )

    elif strategy == "timestamp":
        if segments is None:
            logger.warning(
                "[timestamp] No segments provided — falling back to sentence strategy."
            )
            return sentence_aware_chunks(
                text=text, video_id=video_id, lang=lang,
                chunk_tokens=config.chunk_tokens,
                overlap_sentences=config.overlap_sentences,
            )
        return timestamp_aware_chunks(
            segments=segments,
            video_id=video_id,
            lang=lang,
            chunk_tokens=config.chunk_tokens,
            overlap_tokens=config.overlap_tokens,
        )

    elif strategy == "semantic":
        if embeddings is None:
            raise ValueError(
                "[semantic] An `embeddings` model must be supplied for semantic chunking."
            )
        return semantic_chunks(
            text=text,
            video_id=video_id,
            lang=lang,
            embeddings=embeddings,
            chunk_tokens=config.chunk_tokens,
            similarity_threshold=config.similarity_threshold,
            min_sentences_per_chunk=config.min_sentences_per_chunk,
        )

    else:
        raise ValueError(
            f"Unknown CHUNK_STRATEGY='{config.strategy}'. "
            "Choose from: sentence | timestamp | semantic"
        )
