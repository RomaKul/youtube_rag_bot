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
  - `timestamp_aware_chunks` used to window over raw transcript *segments*
    directly. YouTube segments are cut on arbitrary time boundaries, not
    sentence boundaries, so a sentence could end up split across two chunks,
    and the token-based `overlap_sentences` tail could cut a sentence in half
    too. It now first reconstructs whole sentences from the segments
    (keeping each sentence's start/end time), windows over *sentences*, and
    overlaps by whole sentences (`overlap_sentences`, default 1) — matching
    the sentence strategy's behavior.
  - `restore_punctuation` used to instantiate a brand-new PunctuationModel()
    (loads a transformer model from disk) on *every call*. It's now a
    lazily-created module-level singleton, since with the fix below it can
    be called more than once per transcript.
  - Root fix, now applied to ALL THREE strategies: some auto-generated
    transcripts have long runs of text with no sentence-ending punctuation
    at all, so the naive `_split_into_sentences` regex finds no boundaries
    and returns the whole block as a single "sentence." That's a problem in
    two ways:
      * `semantic_chunks` hands that oversized string straight to
        `embeddings.embed_documents(...)`, which crashed against Cohere's
        embedding API with:
            ValueError: The Cohere embedding API does not support texts
            longer than 2048 characters.
      * `sentence`/`timestamp` chunking window by *token count*
        (`chunk_tokens`), so a single oversized "sentence" larger than the
        window size ends up as its own oversized chunk regardless of the
        configured `chunk_tokens`.
    Fix: `_split_into_sentences_with_spans_safe()` walks the naive sentence
    list and, for any piece longer than `EMBEDDING_TEXT_MAX_LENGTH` chars,
    runs it through `restore_punctuation()` and re-splits it. Because
    punctuation restoration only inserts punctuation/capitalization and
    never adds, removes, splits, or reorders words, the re-split pieces are
    realigned to their original character offsets by *word count* rather
    than by literal substring search (which would fail — the punctuated
    text is no longer a verbatim substring of the source). If restoration
    still doesn't produce a usable split (no breakpoints found), it falls
    back to a hard character-length wrap so nothing oversized ever survives.
    `timestamp_aware_chunks` uses the span-aware version directly so
    resplit pieces still map to correct start/end timestamps; `sentence`
    and `semantic` chunking use the text-only wrapper.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import tiktoken
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from deepmultilingualpunctuation import PunctuationModel

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


# ── Sentence splitting (with safe long-block handling) ────────────────────────

# Hard cap on how long a single "sentence" is allowed to be before we force a
# re-split. 2048 originates from Cohere's embed API limit, but it's a
# reasonable general guard for sentence/timestamp chunking too — no chunking
# strategy should ever treat a multi-thousand-character punctuation-less blob
# as one atomic unit.
EMBEDDING_TEXT_MAX_LENGTH = 2048

# ── Punctuation restoration (cached — loading the model is expensive) ─────────

_punct_model: Optional[PunctuationModel] = None


def _get_punctuation_model() -> PunctuationModel:
    global _punct_model
    if _punct_model is None:
        _punct_model = PunctuationModel()
    return _punct_model


def restore_punctuation(text: str) -> str:
    """Load once (cached singleton), reuse across calls."""
    return _get_punctuation_model().restore_punctuation(text)

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
_WORD = re.compile(r'\S+')


def _split_into_sentences(text: str) -> list[str]:
    """Naive but fast sentence splitter (handles Mr./Dr. reasonably well)."""
    protected = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|approx|avg)\.\s',
                       r'\1<DOT> ', text)
    parts = _SENTENCE_END.split(protected)
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def _restore_and_resplit_with_spans(
    text: str, max_length: int
) -> list[tuple[str, int, int]]:
    """
    `text` is an over-length block with no usable sentence-ending punctuation
    (offsets returned are relative to `text` itself). Restores punctuation
    and re-splits; since restoration only inserts punctuation/capitalization
    and never adds/removes/reorders words, the resulting pieces are realigned
    to `text`'s original character offsets by WORD COUNT rather than literal
    substring search (the punctuated text is no longer a verbatim substring
    of `text`, so `.find()` would not reliably work).

    Falls back to a hard character-length wrap if punctuation restoration
    finds no usable breakpoints either, so an oversized block never survives
    unsplit.
    """
    punctuated = restore_punctuation(text)
    sub_sentences = _split_into_sentences(punctuated)

    if len(sub_sentences) <= 1:
        spans = []
        for k in range(0, len(text), max_length):
            piece = text[k:k + max_length]
            if piece.strip():
                spans.append((piece, k, k + len(piece)))
        return spans

    word_spans = _word_spans(text)
    result: list[tuple[str, int, int]] = []
    word_idx = 0

    for sub in sub_sentences:
        n_words = len(sub.split())
        if n_words == 0:
            continue
        piece_spans = word_spans[word_idx:word_idx + n_words]
        if not piece_spans:
            break
        start, end = piece_spans[0][0], piece_spans[-1][1]
        result.append((sub, start, end))
        word_idx += n_words

    # Alignment drift safety net: if word counts didn't line up exactly and
    # some trailing words are unaccounted for, fold them into a final piece
    # rather than silently dropping text.
    if word_idx < len(word_spans):
        start = word_spans[word_idx][0]
        end = word_spans[-1][1]
        result.append((text[start:end], start, end))

    return result


def _split_into_sentences_with_spans_safe(
    text: str, max_length: int = EMBEDDING_TEXT_MAX_LENGTH
) -> list[tuple[str, int, int]]:
    """
    Sentence split that also returns each sentence's (start, end) character
    offset into `text`, guarding against transcripts with little or no
    punctuation (common in auto-generated captions) where the naive splitter
    would otherwise return one giant unsplit block.
    """
    naive = _split_into_sentences(text)
    result: list[tuple[str, int, int]] = []
    search_from = 0

    for sent in naive:
        idx = text.find(sent, search_from)
        if idx == -1:
            idx = search_from  # defensive fallback; shouldn't normally trigger
        start, end = idx, idx + len(sent)
        search_from = end

        if end - start <= max_length:
            result.append((sent, start, end))
        else:
            logger.warning(
                f"[chunking] {end - start}-char block exceeds {max_length}-char "
                "limit and has no usable punctuation — restoring punctuation "
                "and re-splitting."
            )
            for sub_text, local_start, local_end in _restore_and_resplit_with_spans(
                text[start:end], max_length
            ):
                result.append((sub_text, start + local_start, start + local_end))

    return result


def _split_into_sentences_safe(
    text: str, max_length: int = EMBEDDING_TEXT_MAX_LENGTH
) -> list[str]:
    """Text-only convenience wrapper around _split_into_sentences_with_spans_safe."""
    return [s for s, _, _ in _split_into_sentences_with_spans_safe(text, max_length)]


# ── Strategy 1: Sentence-aware chunking ───────────────────────────────────────


def sentence_aware_chunks(
    text: str,
    video_id: str,
    lang: str,
    chunk_tokens: int = 300,
    overlap_sentences: int = 1,
    split_sentences: bool = True,
) -> list[Document]:
    """
    split_sentences=True:  groups whole sentences into windows of <= chunk_tokens
    tokens, sharing `overlap_sentences` sentences between adjacent chunks.
 
    split_sentences=False: ignores sentence boundaries entirely and slices the
    text on raw token boundaries instead, so every chunk (but possibly the
    last) is exactly chunk_tokens tokens. The whole text is encoded ONCE and
    sliced from the token-id list directly — no per-sentence or per-word
    calls to a tokenizer, which is what made the previous version either
    ignore chunk_tokens (one giant chunk) or be slow on long transcripts.
    `overlap_sentences` is reused here as an overlap in TOKENS (not
    sentences) between adjacent windows, since there are no sentences in
    this mode.
    """
    if split_sentences:
        return _sentence_windows(text, video_id, lang, chunk_tokens, overlap_sentences)
    return _token_windows(text, video_id, lang, chunk_tokens, overlap_tokens=overlap_sentences)
 
 
def _sentence_windows(
    text: str, video_id: str, lang: str, chunk_tokens: int, overlap_sentences: int,
) -> list[Document]:
    sentences = _split_into_sentences_safe(text)
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
                "video_id": video_id, "lang": lang, "chunk_idx": chunk_idx,
                "strategy": "sentence", "type": "chunk",
            },
        ))
 
        advance = max(1, len(window) - overlap_sentences)
        i += advance
        chunk_idx += 1
 
    logger.info(f"[sentence] {len(chunks)} chunks from {len(sentences)} sentences")
    return chunks

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    tiktoken = None
    _ENCODING = None
 
def _token_windows(
    text: str, video_id: str, lang: str, chunk_tokens: int, overlap_tokens: int,
) -> list[Document]:
    if not text.strip():
        return []
 
    if _ENCODING is not None:
        token_ids = _ENCODING.encode(text)
        decode = _ENCODING.decode
    else:
        # No tiktoken available — fall back to whitespace tokens as the unit.
        # Still O(n) total (one split, no repeated count_tokens calls), just
        # a coarser approximation of "tokens" than a real BPE tokenizer.
        token_ids = text.split()
        decode = " ".join
 
    step = max(1, chunk_tokens - overlap_tokens)
    chunks: list[Document] = []
    chunk_idx = 0
 
    for start in range(0, len(token_ids), step):
        window = token_ids[start:start + chunk_tokens]
        if not window:
            break
        chunk_text = decode(window).strip()
        if chunk_text:
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "video_id": video_id, "lang": lang, "chunk_idx": chunk_idx,
                    "strategy": "sentence", "type": "chunk",
                },
            ))
            chunk_idx += 1
        if start + chunk_tokens >= len(token_ids):
            break
 
    logger.info(f"[sentence/no-split] {len(chunks)} chunks from {len(token_ids)} tokens")
    return chunks


# ── Strategy 2: Timestamp-aware chunking ──────────────────────────────────────

def _segments_with_offsets(segments: list[Segment]) -> tuple[str, list[tuple[int, int, Segment]]]:
    """
    Joins segment texts into one string (space-separated) and records, for each
    segment, the [start_char, end_char) span it occupies in that joined string.
    Lets us map a sentence's character position back to the segment(s) — and
    therefore the timestamps — it came from.
    """
    full_text = ""
    offsets: list[tuple[int, int, Segment]] = []
    for seg in segments:
        if full_text:
            full_text += " "
        start = len(full_text)
        full_text += seg.text
        offsets.append((start, len(full_text), seg))
    return full_text, offsets


def _segment_at(offsets: list[tuple[int, int, Segment]], pos: int) -> Optional[Segment]:
    for start, end, seg in offsets:
        if start <= pos < end:
            return seg
    return offsets[-1][2] if offsets else None


def _sentences_with_timestamps(segments: list[Segment]) -> list[tuple[str, float, float]]:
    """
    Reconstructs whole sentences from raw (often mid-sentence) transcript
    segments and tags each sentence with the start/end time of the segment(s)
    it spans, so a sentence is never later split across two chunks.

    Uses the span-aware safe splitter so a long punctuation-less block still
    gets broken up (instead of becoming one oversized chunk) while keeping
    correct timestamp offsets for the re-split pieces.
    """
    full_text, offsets = _segments_with_offsets(segments)
    sentence_spans = _split_into_sentences_with_spans_safe(full_text)

    result: list[tuple[str, float, float]] = []
    for sent, start_pos, end_pos in sentence_spans:
        start_seg = _segment_at(offsets, start_pos)
        end_seg = _segment_at(offsets, max(start_pos, end_pos - 1))
        start_time = start_seg.start if start_seg else 0.0
        end_time = end_seg.end if end_seg else start_time

        result.append((sent, start_time, end_time))

    return result


def timestamp_aware_chunks(
    segments: list[Segment],
    video_id: str,
    lang: str,
    chunk_tokens: int = 300,
    overlap_sentences: int = 1,
) -> list[Document]:
    """
    Groups whole sentences (reconstructed from the raw transcript segments)
    into windows of <= chunk_tokens tokens, so a sentence is never split
    between two chunks. Preserves start_time/end_time metadata for deep links.
    Adjacent chunks share `overlap_sentences` sentences for context continuity.
    """
    sentence_data = _sentences_with_timestamps(segments)
    chunks: list[Document] = []
    i = 0
    chunk_idx = 0

    while i < len(sentence_data):
        window: list[tuple[str, float, float]] = []
        token_count = 0

        for j in range(i, len(sentence_data)):
            s_tokens = count_tokens(sentence_data[j][0])
            if token_count + s_tokens > chunk_tokens and window:
                break
            window.append(sentence_data[j])
            token_count += s_tokens

        chunk_text = " ".join(s[0] for s in window)
        start_sec = window[0][1]
        end_sec = window[-1][2]

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

        advance = max(1, len(window) - overlap_sentences)
        i += advance
        chunk_idx += 1

    logger.info(f"[timestamp] {len(chunks)} chunks from {len(sentence_data)} sentences")
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

    Uses `_split_into_sentences_safe` rather than the raw splitter: some
    auto-generated transcripts have long stretches with no sentence-ending
    punctuation at all, which would otherwise produce a single oversized
    "sentence" that blows past embedding-API length limits (e.g. Cohere's
    2048-character cap).
    """
    sentences = _split_into_sentences_safe(text)
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
    overlap_sentences: int = 1        # used by sentence & timestamp strategies
    similarity_threshold: float = 0.75  # used by semantic strategy
    split_sentences: bool = True  # used by sentence strategy
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
            split_sentences=config.split_sentences
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
            overlap_sentences=config.overlap_sentences,
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