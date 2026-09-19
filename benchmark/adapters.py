"""
Adapter wiring the benchmark harness to youtube_rag_bot (app.bots.bot_local).

Wired directly from the bot_local.py you shared:
  - app.bots.bot_local: PROVIDER, CHUNK_STRATEGY, build_llm, build_embeddings,
    extract_video_id, get_transcript, index_transcript, init_vectorstore,
    load_bm25_index, bm25_cache_path, count_tokens
  - app.rag.rag_graph.build_routed_rag_graph

Three things are NOT exposed by the code you shared, so they're stubbed with a
clearly marked TODO. Each needs either a one-line contract check on your side,
or a small addition to rag_graph.py / hybrid_search.py. I don't have those
files, only the README's description of them, so I can't wire these for real.

  TODO(1) RETRIEVAL MODE (vector / bm25 / hybrid / hybrid_rerank) for E2.
          build_routed_rag_graph has no parameter for this — per the README it
          always does hybrid+rerank internally. See the comment inside
          _build_agent() for the minimal patch to rag_graph.py that would let
          this adapter actually switch modes.

  TODO(2) ROUTER ON/OFF for E3.
          There's no explicit switch to bypass the router node. As a proxy,
          when cfg.router_enabled is False this adapter wipes history and
          prev_context before every question, which forces the router to
          always pick "from_db" — this isolates the routing *benefit* even
          without a real bypass, but it is not the same as skipping the
          router node's own LLM call. If you add a real `router_enabled=`
          kwarg to build_routed_rag_graph, wire it in below instead.

  TODO(3) GRAPH I/O CONTRACT. I don't have rag_graph.py, so invoke here
          guesses the state shape: input {"question": ...}, output read
          defensively from a few likely key names for the answer, the route,
          and the retrieved chunks. If your graph uses different keys, fix
          the three marked spots below — everything else in the harness is
          unaffected by this.

Also flagging two things that don't match earlier messages, so they don't
silently produce wrong numbers:
  - Your .env used OVERLAP_TOKENS=30, but this bot_local.py reads
    OVERLAP_SENTANCES (default 1) — a different unit, not just a rename.
    I'm setting OVERLAP_SENTANCES from cfg here; overlap_tokens in configs.py
    is being reinterpreted as a sentence count. Rename the field in
    configs.py once you confirm which one the shipped code actually uses.
  - SIMILARITY_K here looks like it's just the vector retriever's k (default
    4), not the reranker's 20-candidates input the README describes. If
    there's no separate "pull 20, keep 4" parameter anywhere in
    hybrid_search.py/reranker.py, E2's hybrid_rerank run and the plain hybrid
    run may end up identical — worth checking once you send those files.
"""

from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

from configs import RunConfig


_bot = None            # app.bots.bot_local, imported/reloaded by apply_config
_llm_cache: dict = {}   # (provider, model) -> llm instance, built once per config


def _bot_module():
    global _bot
    if _bot is None:
        from app.bots import bot_local as bot
        _bot = bot
    return _bot


@dataclass
class Retrieved:
    text: str
    score: float = 0.0
    start_sec: float | None = None
    meta: dict = field(default_factory=dict)


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
    """
    bot_local.py reads its settings into module-level constants (PROVIDER,
    CHUNK_STRATEGY, ...) via os.getenv() at import time. Set the env vars,
    then reload the module so those constants are recomputed for this run.
    """
    os.environ["PROVIDER"] = cfg.provider
    os.environ["CHUNK_STRATEGY"] = cfg.chunk_strategy
    os.environ["CHUNK_TOKENS"] = str(cfg.chunk_tokens)
    os.environ["OVERLAP_SENTANCES"] = str(cfg.overlap_tokens)   # see module docstring
    os.environ["SIMILARITY_THR"] = str(cfg.similarity_thr)
    os.environ["SIMILARITY_K"] = str(cfg.candidates_k)

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
        _llm_cache[key] = bot.build_llm()


def _llm_for(cfg: RunConfig):
    return _llm_cache[(cfg.provider, cfg.llm_model)]


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
        vs = bot.init_vectorstore(video_id)
        vs.delete_collection()   # langchain_chroma: drops this Chroma collection
    except Exception as exc:
        print(f"drop_index: could not delete Chroma collection: {exc}")

    try:
        os.remove(bot.bm25_cache_path(video_id))
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# 3. Answer one question
# ---------------------------------------------------------------------------
def _build_agent(video_id: str, cfg: RunConfig, prev_context: list[str], history: list[dict]):
    bot = _bot_module()
    vs = bot.init_vectorstore(video_id)
    bm25 = bot.load_bm25_index(video_id)
    llm = _llm_for(cfg)

    # TODO(1) RETRIEVAL MODE — minimal patch to rag_graph.py that would make
    # this real (adjust names to whatever hybrid_search.py/reranker.py use):
    #
    #   def build_routed_rag_graph(vs, llm, prev_chunks_fn, history_fn,
    #                               bm25_index_fn, retrieval_mode="hybrid_rerank"):
    #       ...
    #       if retrieval_mode == "vector":
    #           docs = vs.similarity_search(query, k=final_k)
    #       elif retrieval_mode == "bm25":
    #           docs = bm25_index_fn().search(query, k=final_k)
    #       else:  # "hybrid" or "hybrid_rerank"
    #           docs = reciprocal_rank_fusion([vector_hits, bm25_hits], k=60)
    #           if retrieval_mode == "hybrid_rerank":
    #               docs = reranker.rerank(query, docs, top_k=final_k)
    #
    # Once that parameter exists, uncomment:
    # kwargs["retrieval_mode"] = cfg.retrieval
    kwargs = dict(
        prev_chunks_fn=lambda: prev_context or None,
        history_fn=lambda: history or [],
        bm25_index_fn=lambda: bm25,
    )

    # TODO(2) ROUTER ON/OFF — real switch, once it exists:
    # kwargs["router_enabled"] = cfg.router_enabled

    return bot.build_routed_rag_graph(vs, llm, **kwargs)


def _extract_contexts(state: dict) -> list[str]:
    # TODO(3): confirm the real key name from rag_graph.py.
    for key in ("contexts", "retrieved_chunks", "chunks", "documents", "docs"):
        val = state.get(key)
        if val:
            texts = []
            for d in val:
                if isinstance(d, dict):
                    texts.append(d.get("text") or d.get("page_content", ""))
                else:
                    texts.append(getattr(d, "page_content", str(d)))
            return [t for t in texts if t]
    return []


def answer_question(
    question: str,
    video_url: str,
    cfg: RunConfig,
    history: list[dict],
    prev_context: list[str],
) -> AnswerResult:
    bot = _bot_module()
    video_id = bot.extract_video_id(video_url)

    # router ablation proxy — see TODO(2) in the module docstring
    if not cfg.router_enabled:
        history, prev_context = [], []

    t0 = time.perf_counter()
    try:
        agent = _build_agent(video_id, cfg, prev_context, history)

        # TODO(3): confirm the input key the graph expects.
        result = agent.invoke({"question": question})

        answer = result.get("answer") or result.get("response") or ""
        route = result.get("route") or result.get("routing_decision") or "from_db"
        contexts = _extract_contexts(result)

    except Exception as exc:
        return AnswerResult(answer="", contexts=[], error=repr(exc),
                            total_ms=(time.perf_counter() - t0) * 1000)

    total_ms = (time.perf_counter() - t0) * 1000

    # Token usage: prefer provider-reported counts if the graph surfaces them;
    # otherwise fall back to counting with bot_local's own tokenizer. The
    # fallback is fine for relative comparisons between configs, but real
    # counts are better for the article's absolute cost figures.
    usage = None
    for key in ("usage_metadata", "usage", "llm_usage"):
        if result.get(key):
            usage = result[key]
            break

    if usage:
        in_tok = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        out_tok = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    else:
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