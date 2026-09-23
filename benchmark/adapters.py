"""
Adapter wiring the benchmark harness to youtube_rag_bot.

Wired against the real code you shared: app.bots.bot_local +
app.rag.rag_graph (build_routed_rag_graph, RAGState, CANDIDATE_K,
SIMILARITY_K, hybrid_search, rerank).

Resolved, now that rag_graph.py is in hand
-------------------------------------------
- GRAPH I/O CONTRACT: agent.invoke() takes the full RAGState dict, exactly
  as shown in receive_question(). No more guessing at input/output keys.
- ROUTER ON/OFF (E3): classify() only calls the router's LLM at all when
  prev_chunks is truthy — `... if prev_chunks else (Route.FROM_DB, ...)`.
  So handing it an always-empty prev_chunks_fn is not a proxy, it's the
  real "no router" path: zero routing LLM calls, always fresh retrieval.
- RETRIEVAL MODE (E2): CANDIDATE_K / SIMILARITY_K are plain module
  globals in rag_graph.py, read at call time inside the retrieve node's
  closure — so setting rag_graph.CANDIDATE_K / .SIMILARITY_K before a run
  controls them without touching your source. Same trick for the
  retrieval function itself: hybrid_search and rerank were imported by
  name into rag_graph's namespace, so reassigning rag_graph.hybrid_search
  / rag_graph.rerank switches what retrieve() calls. Used to implement
  "vector only" (rerank turned into a pass-through slice) and "hybrid, no
  rerank" cleanly.
- Confirmed: the off_topic branch in the module docstring's diagram isn't
  actually wired into route_after_classify — the graph is binary
  (from_context / from_db). Off-topic questions are handled inside
  generate() via the system prompt's "answer using general knowledge"
  instruction, not via a separate route. Worth stating exactly this way
  in the article rather than the three-way diagram.

Still open (send hybrid_search.py to close these)
--------------------------------------------------
- TODO(BM25): "bm25 only" for E2 needs BM25Index's real search method.
  This adapter tries a few likely method names defensively and raises a
  clear error if none exist — I don't have hybrid_search.py to confirm
  the actual name.
- TODO(usage tracking is best-effort): to get total tokens per question
  (classify + rewrite + generate calls) without editing rag_graph.py, the
  llm object is wrapped so every .invoke() is logged. This assumes
  router.classify_question (which we don't have) also just calls
  llm.invoke(...) like every node in rag_graph.py does, and that nothing
  does isinstance(llm, BaseChatModel) checks against it. If classify_question
  breaks with this wrapper, send router.py and I'll adjust.
- Confirmed real inconsistency (unchanged from before): your .env has
  OVERLAP_SENTENCES=1, bot_local.py reads OVERLAP_SENTENCES (default 1) —
  a different unit, not a rename. Still mapped through as-is below.
- bot_local.py's SIMILARITY_K env var turns out to be dead code — the
  retrieve node ignores it and always reads rag_graph.CANDIDATE_K /
  .SIMILARITY_K instead. Dropped it from apply_config() below in favor of
  patching those two module attributes directly.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from configs import RunConfig


# Make `import app...` work regardless of whether `pip install -e .` was run
# and regardless of the working directory run_benchmark.py is launched from.
# This repo uses a src-layout (src/app/...); benchmark/ sits next to src/, so
# the package root is always ../src relative to this file.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


_bot = None            # app.bots.bot_local, imported/reloaded by apply_config
_llm_cache: dict = {}   # (provider, model) -> _UsageTrackingLLM, built once per config
_real_hybrid_search = None   # captured once, restored for "hybrid"/"hybrid_rerank"
_real_rerank = None


def _bot_module():
    global _bot
    if _bot is None:
        from app.bots import bot_local as bot
        _bot = bot
    return _bot


class _UsageTrackingLLM:
    """Wraps an LLM so every .invoke() call this turn is logged, letting us
    sum tokens across classify + rewrite_query + generate without editing
    rag_graph.py. See TODO(usage tracking) in the module docstring."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list = []

    def invoke(self, *args, **kwargs):
        resp = self._inner.invoke(*args, **kwargs)
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            self.calls.append(usage)
        return resp

    def reset(self):
        self.calls = []

    def __getattr__(self, name):
        return getattr(self._inner, name)


@dataclass
class AnswerResult:
    answer: str
    contexts: list[str]
    route: str = "from_db"
    llm_in_tokens: int = 0
    llm_out_tokens: int = 0
    embed_tokens: int = 0
    rerank_queries: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# 1. Apply a configuration
# ---------------------------------------------------------------------------
def apply_config(cfg: RunConfig) -> None:
    os.environ["PROVIDER"] = cfg.provider
    os.environ["CHUNK_STRATEGY"] = cfg.chunk_strategy
    os.environ["CHUNK_TOKENS"] = str(cfg.chunk_tokens)
    os.environ["OVERLAP_SENTENCES"] = str(cfg.overlap_sentences)   # see docstring note
    os.environ["SIMILARITY_THR"] = str(cfg.similarity_thr)

    if cfg.provider == "bedrock":
        os.environ["BEDROCK_LLM_MODEL"] = cfg.llm_model
        os.environ["BEDROCK_EMBED_MODEL"] = cfg.embed_model
    else:
        os.environ["OLLAMA_MODEL"] = cfg.llm_model
        os.environ["OLLAMA_EMBED"] = cfg.embed_model

    bot = _bot_module()
    importlib.reload(bot)

    key = (cfg.provider, cfg.llm_model)
    if key not in _llm_cache:
        _llm_cache[key] = _UsageTrackingLLM(bot.build_llm())

    _configure_retrieval(cfg)


def _llm_for(cfg: RunConfig) -> _UsageTrackingLLM:
    return _llm_cache[(cfg.provider, cfg.llm_model)]


def _bm25_only_search(query, vectorstore, bm25_index, k):
    if bm25_index is None:
        return []
    for method in ("search", "query", "get_top_k", "retrieve"):
        fn = getattr(bm25_index, method, None)
        if fn:
            return fn(query, k)
    raise NotImplementedError(
        "BM25Index exposes none of (search/query/get_top_k/retrieve) — "
        "send hybrid_search.py so this can call the real method name."
    )


def _configure_retrieval(cfg: RunConfig) -> None:
    """Point rag_graph's CANDIDATE_K/SIMILARITY_K and its hybrid_search/
    rerank names at whatever this config's retrieval mode needs. Safe to
    call every apply_config — always sets an explicit state, never assumes
    what the previous config left behind."""
    import app.rag.rag_graph as rag_graph

    global _real_hybrid_search, _real_rerank
    if _real_hybrid_search is None:
        _real_hybrid_search = rag_graph.hybrid_search
        _real_rerank = rag_graph.rerank

    rag_graph.SIMILARITY_K = cfg.similarity_k
    rag_graph.RETRIEVAL_TOP_K = cfg.retrieval_top_k

    def _pass_through_rerank(query, candidates, top_n):
        return candidates[:top_n]

    if cfg.retrieval == "vector":
        rag_graph.hybrid_search = lambda query, vs, bm25, k: vs.similarity_search(query, k=k)
        rag_graph.rerank = _pass_through_rerank
    elif cfg.retrieval == "bm25":
        rag_graph.hybrid_search = _bm25_only_search
        rag_graph.rerank = _pass_through_rerank
    elif cfg.retrieval == "hybrid":
        rag_graph.hybrid_search = _real_hybrid_search
        rag_graph.rerank = _pass_through_rerank
    else:   # "hybrid_rerank" — real system behaviour, untouched
        rag_graph.hybrid_search = _real_hybrid_search
        rag_graph.rerank = _real_rerank


# ---------------------------------------------------------------------------
# 2. Index a video
# ---------------------------------------------------------------------------
def index_video(video_url: str, cfg: RunConfig) -> dict[str, Any]:
    bot = _bot_module()
    video_id = bot.extract_video_id(video_url)
    if not video_id:
        raise ValueError(f"could not extract a video id from {video_url!r}")

    t0 = time.perf_counter()
    transcript_text, segments, kind, lang, status = bot.get_transcript(video_id)
    llm = _llm_for(cfg)

    vs, n_chunks, summary = bot.index_transcript(
        video_id, transcript_text, lang, llm, segments=segments,
    )

    return {
        "video_id": video_id,
        "language": lang,
        "transcript_kind": kind,
        "n_chunks": n_chunks,
        "transcript_tokens": bot.count_tokens(transcript_text),
        "index_seconds": round(time.perf_counter() - t0, 2),
    }


def drop_index(video_url: str, cfg: RunConfig) -> None:
    bot = _bot_module()
    video_id = bot.extract_video_id(video_url)
    if not video_id:
        return
    try:
        bot.init_vectorstore(video_id).delete_collection()
    except Exception as exc:
        print(f"drop_index: could not delete Chroma collection: {exc}")
    try:
        os.remove(bot.bm25_cache_path(video_id))
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# 3. Answer one question
# ---------------------------------------------------------------------------
def _to_qa_pairs(history: list[dict]) -> list[tuple[str, str]]:
    """RAGState.history is list[tuple[question, answer]]; the runner hands us
    list[{"role": ..., "content": ...}]. Pair them up."""
    pairs, pending_q = [], None
    for msg in history:
        if msg["role"] == "user":
            pending_q = msg["content"]
        elif msg["role"] == "assistant" and pending_q is not None:
            pairs.append((pending_q, msg["content"]))
            pending_q = None
    return pairs


def answer_question(
    question: str,
    video_url: str,
    cfg: RunConfig,
    history: list[dict],
    prev_context: list[str],
) -> AnswerResult:
    bot = _bot_module()
    video_id = bot.extract_video_id(video_url)
    llm = _llm_for(cfg)
    llm.reset()

    # Real "no router" ablation: classify() only calls the router's LLM when
    # prev_chunks is truthy, so an always-empty prev_chunks_fn forces
    # Route.FROM_DB with zero routing calls — not a proxy, the actual path.
    if not cfg.router_enabled:
        prev_docs: list[Document] = []
        qa_history: list[tuple[str, str]] = []
    else:
        prev_docs = [Document(page_content=t, metadata={}) for t in (prev_context or [])]
        qa_history = _to_qa_pairs(history)

    vs = bot.init_vectorstore(video_id)
    bm25 = bot.load_bm25_index(video_id)

    agent = bot.build_routed_rag_graph(
        vs, llm,
        prev_chunks_fn=lambda: prev_docs,
        history_fn=lambda: qa_history,
        bm25_index_fn=lambda: bm25,
    )

    t0 = time.perf_counter()
    try:
        result = agent.invoke({
            "question": question,
            "search_query": "",
            "context": "",
            "answer": "",
            "video_id": video_id,
            "route": "",
            "retrieved_docs": [],
            "history": qa_history,
        })
    except Exception as exc:
        return AnswerResult(answer="", contexts=[], error=repr(exc),
                            total_ms=(time.perf_counter() - t0) * 1000)
    total_ms = (time.perf_counter() - t0) * 1000

    answer = result.get("answer", "")
    route = result.get("route", "from_db")
    contexts = [d.page_content for d in result.get("retrieved_docs", [])]

    in_tok = sum(u.get("input_tokens", 0) for u in llm.calls)
    out_tok = sum(u.get("output_tokens", 0) for u in llm.calls)
    if not llm.calls:   # provider didn't attach usage_metadata; fall back
        in_tok = bot.count_tokens(question + "\n".join(contexts))
        out_tok = bot.count_tokens(answer)

    return AnswerResult(
        answer=answer,
        contexts=contexts,
        route=route,
        llm_in_tokens=in_tok,
        llm_out_tokens=out_tok,
        rerank_queries=1 if cfg.retrieval == "hybrid_rerank" else 0,
        total_ms=total_ms,
    )


# ---------------------------------------------------------------------------
# 4. Judge model for RAGAS
# ---------------------------------------------------------------------------
def get_judge():
    from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    region = os.getenv("AWS_REGION", "us-east-1")
    llm = ChatBedrockConverse(model=os.getenv("JUDGE_LLM", "amazon.nova-pro-v1:0"),
                              region_name=region, temperature=0)
    emb = BedrockEmbeddings(model_id=os.getenv("JUDGE_EMBED", "cohere.embed-multilingual-v3"),
                            region_name=region)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)