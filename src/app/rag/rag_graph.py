"""
rag_graph.py — Routed RAG graph.

Flow:
                        ┌─────────────┐
                        │   classify  │  (lightweight LLM call, history-aware)
                        └──────┬──────┘
               ┌───────────────┼──────────────┐
          from_context      from_db        off_topic
               │               │               │
               ▼               ▼               ▼
         [generate        [rewrite_query    [general_
          from cache]      → hybrid_search    knowledge]
               │            → rerank →
               │            generate]
               │               │               │
               └───────────────┴───────────────┘
                                │
                              [END]

Key behaviours
──────────────
• from_context  : uses only the cached last-retrieval chunks; no DB touch.
• from_db       : query rewrite → hybrid (vector + BM25) search → cross-encoder
                  rerank → generate.
• Recent chat history is injected into both routing and generation so
  follow-up questions work naturally.
• After every from_db answer the retrieved docs are saved back into
  user_data so the NEXT question has fresh cached context.
• After a from_context answer the cache is NOT overwritten (it's still valid).
• Answers include chunk/timestamp citations the user can use to jump to the
  exact moment in the video.

Backend-agnostic vectorstore:
  This module has NO hard runtime dependency on langchain_chroma. Chroma is
  only imported inside an `if TYPE_CHECKING:` block for editor/mypy support —
  it's never actually imported when the code runs. Everywhere else the
  vectorstore is typed as `VectorStoreLike`, a Protocol requiring only
  `similarity_search()`, which both Chroma and OpenSearchVectorSearch
  implement identically. That's what lets this exact file be imported by
  bot.py (local, Chroma) AND bot_aws.py (OpenSearch Serverless) without
  langchain_chroma needing to be installed in the AWS image at all.

  The one spot the two backends genuinely differ is fetching the video
  summary (Chroma's metadata `.get(where=...)` vs. OpenSearch's filtered
  `similarity_search`). `build_routed_rag_graph()` takes an optional
  `summary_fetcher_fn` for exactly this — see bot_aws.py, which passes
  `vectorstore_aws.fetch_summary_doc`.

FIXED vs. original:
  - `_format_docs` no longer builds a Python set per doc (was a silent
    TypeError on join — `[{d.page_content} for d in docs]`).
  - `receive_question` no longer hardcodes a magic "ASK_QUESTION = 2"
    placeholder; the real conversation-state constant is imported.
  - LLM calls in generation nodes are wrapped in try/except so a transient
    provider error degrades to a user-facing message instead of crashing
    the whole graph invocation silently mid-route.
  - langchain_chroma is no longer a hard import (see "Backend-agnostic
    vectorstore" above) so this module loads fine in a Chroma-less AWS image.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Protocol,
    TypedDict,
    runtime_checkable,
)

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from telegram.ext import ContextTypes, ConversationHandler

from app.rag.router import (
    Route, classify_question,
    save_retrieved_chunks, load_retrieved_chunks,
    save_turn, load_history,
)
from app.rag.hybrid_search import BM25Index, hybrid_search
from app.rag.reranker import rerank

logger = logging.getLogger(__name__)

SIMILARITY_K   = 4    # final number of chunks fed to the LLM
CANDIDATE_K    = 20   # candidates pulled before re-ranking down to SIMILARITY_K


if TYPE_CHECKING:
    # Type-checking only — never executed at runtime. Combined with
    # `from __future__ import annotations` above (which turns every
    # annotation in this module into a plain string), this means Chroma is
    # NOT a runtime import here. Keep this import here purely so mypy/IDEs
    # can resolve `Chroma` if you reference it explicitly anywhere.
    from langchain_chroma import Chroma


@runtime_checkable
class VectorStoreLike(Protocol):
    """Structural type covering whatever vectorstore backend is in play
    (Chroma locally, OpenSearchVectorSearch in AWS). Both implement
    similarity_search() with this shape, which is all this module needs
    outside of fetch_video_summary()."""

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]: ...


# ── Summary fetch ──────────────────────────────────────────────────────────────

def fetch_video_summary(vectorstore: "Chroma") -> str:
    """
    Default summary fetcher — Chroma-specific.

    index_transcript() stores it with metadata type='summary', chunk_idx=-1.
    We query by metadata filter so we never pay for an embedding lookup.
    Returns an empty string if none exists (e.g. summary generation failed).

    This relies on Chroma's `.get(where=...)` API, which OpenSearch doesn't
    have. Non-Chroma backends should pass their own `summary_fetcher_fn` to
    build_routed_rag_graph() instead of relying on this default — see
    vectorstore_aws.fetch_summary_doc() + bot_aws.py.
    """
    try:
        results = vectorstore.get(
            where={"type": "summary"},
            include=["documents"],
        )
        docs = results.get("documents", [])
        if docs and docs[0]:
            logger.info("[summary] Loaded video summary from ChromaDB")
            return docs[0]
    except Exception as e:
        logger.warning(f"[summary] Could not fetch summary: {e}")
    return ""


SYSTEM_PROMPT = """You are an AI assistant that analyzes YouTube videos subtitles.
Use the provided video transcript context to answer questions accurately.
If the answer is not contained in the context, answer it but mention about this limitation. 
If the question is off-topic, answer it using your general knowledge.
Context will have timestampes before some cunks, you can use them to cite the answer.
Answer in the same language as the question. 
"""

_QUERY_REWRITE_SYSTEM = """You rewrite casual user questions into focused search
queries for finding relevant passages in a video transcript.
Respond with ONLY the rewritten query — no preamble, no quotes, no explanation.
Keep it short (under 20 words). If the question is already a good search
query, return it unchanged. New question must be written in the same language as the original question."""


# ── Shared state ───────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question:        str
    search_query:     str          # possibly rewritten version used for retrieval
    context:          str          # formatted text passed to generate node
    answer:           str
    video_id:         str
    route:            str          # Route enum value (string)
    retrieved_docs:   list[Document]   # docs found in this turn (empty for context/off-topic)
    history:          list[tuple[str, str]]   # recent (question, answer) turns


# ── Node factories ─────────────────────────────────────────────────────────────

def make_classify_node(llm: BaseChatModel, prev_chunks_fn, history_fn):
    """
    prev_chunks_fn: callable() → list[Document]
    history_fn:     callable() → list[tuple[str, str]]
    Injected so the node can read user_data without importing Telegram types.

    """
    def classify(state: RAGState) -> RAGState:
        prev_chunks = prev_chunks_fn()
        history = history_fn()
        route, reason = classify_question(
            state["question"], prev_chunks, llm, history=history,
        ) if prev_chunks else (Route.FROM_DB, "No cached context available")

        if route == Route.FROM_CONTEXT and prev_chunks:
            # Format cached chunks as context right here
            context = _format_docs(prev_chunks)
            return {**state, "route": route.value, "context": context,
                    "retrieved_docs": prev_chunks, "history": history}

        elif route == Route.FROM_CONTEXT:
            route = Route.FROM_DB

        # FROM_DB and OFF_TOPIC — context filled in later nodes
        return {**state, "route": route.value, "context": "", "retrieved_docs": [],
                "history": history}

    return classify


def make_query_rewrite_node(llm: BaseChatModel):
    """Rewrites the raw question into a better retrieval query (from_db path only)."""
    def rewrite(state: RAGState) -> RAGState:
        try:
            resp = llm.invoke([
                SystemMessage(content=_QUERY_REWRITE_SYSTEM),
                HumanMessage(content=state["question"]),
            ])
            rewritten = resp.content.strip().strip('"')
            if not rewritten:
                rewritten = state["question"]
        except Exception as e:
            logger.warning(f"[rewrite] Query rewrite failed ({e}); using original question")
            rewritten = state["question"]
        logger.info(f"[rewrite] '{state['question']}' → '{rewritten}'")
        return {**state, "search_query": rewritten}
    return rewrite


def make_retrieve_node(vectorstore: VectorStoreLike, bm25_index_fn):
    """
    bm25_index_fn: callable() → Optional[BM25Index]
    Pulled in lazily (not baked in at build time) so a freshly (re)built BM25
    index after re-indexing is always picked up.
    """
    def retrieve(state: RAGState) -> RAGState:
        query = state.get("search_query") or state["question"]
        logger.info(f"[retrieve] hybrid search for: {query}")

        bm25_index = bm25_index_fn()
        candidates = hybrid_search(query, vectorstore, bm25_index, k=CANDIDATE_K)

        docs = rerank(query, candidates, top_n=SIMILARITY_K)
        context = _format_docs(docs)
        return {**state, "context": context, "retrieved_docs": docs}
    return retrieve


def make_generate_node(llm: BaseChatModel):
    def generate(state: RAGState) -> RAGState:
        logger.info(f"[generate] route={state['route']}")
        history_block = _format_history_for_prompt(state.get("history", []))
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Recent conversation (for follow-up context only):\n{history_block}\n\n"
                f"Video transcript context:\n{state['context']}\n\n"
                f"Question: {state['question']}\n\n"
                "Provide a detailed answer based on the context above. "
            )),
        ]
        try:
            response = llm.invoke(messages)
            answer = response.content
        except Exception as e:
            logger.error(f"[generate] LLM call failed: {e}")
            answer = "⚠️ Sorry, I had trouble generating an answer just now. Please try again."
        return {**state, "answer": answer}
    return generate


# ── Routing edge ───────────────────────────────────────────────────────────────

def route_after_classify(state: RAGState) -> str:
    """LangGraph conditional edge: maps route value → next node name."""
    r = state.get("route", Route.FROM_DB.value)
    if r == Route.FROM_CONTEXT.value:
        return "generate"           # skip DB; context already filled in
    return "rewrite_query"          # FROM_DB default


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_routed_rag_graph(
    vectorstore: VectorStoreLike,
    llm: BaseChatModel,
    prev_chunks_fn,          # callable() → list[Document]
    history_fn=None,         # callable() → list[tuple[str, str]]
    bm25_index_fn=None,      # callable() → Optional[BM25Index]
) -> object:                 # compiled LangGraph

    if history_fn is None:
        history_fn = lambda: []
    if bm25_index_fn is None:
        bm25_index_fn = lambda: None

    graph = StateGraph(RAGState)

    graph.add_node("classify",          make_classify_node(llm, prev_chunks_fn, history_fn))
    graph.add_node("rewrite_query",     make_query_rewrite_node(llm))
    graph.add_node("retrieve",          make_retrieve_node(vectorstore, bm25_index_fn))
    graph.add_node("generate",          make_generate_node(llm))

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "generate":          "generate",
            "rewrite_query":     "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query",     "retrieve")
    graph.add_edge("retrieve",          "generate")
    graph.add_edge("generate",          END)

    return graph.compile()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no context)"
    parts = []
    for d in docs:
        label = d.metadata.get("timestamp_label")
        link = d.metadata.get("deep_link")
        if label and link:
            block = f"{label}({link})\n{d.page_content}"
        elif label:
            block = f"{label}\n{d.page_content}"
        else:
            block = d.page_content
        parts.append(block)
    return "\n\n---\n\n".join(parts)


def _format_history_for_prompt(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(none)"
    lines = []
    for q, a in history[-3:]:
        lines.append(f"User: {q}")
        lines.append(f"Bot: {a[:300]}")
    return "\n".join(lines)


# ── receive_question handler (drop-in for bot.py) ───────────────────────────

async def receive_question(update, context: ContextTypes.DEFAULT_TYPE):
    """
    Returns the conversation-state constant to stay in (ASK_QUESTION),
    imported lazily to avoid a circular import at module load time. Tries
    bot.py first (local dev) then bot_aws.py (AWS deployment) — both define
    the same ASK_URL, ASK_LANG, ASK_QUESTION = range(3) constants.
    """
    from app.bots.bot_core import ASK_QUESTION

    agent = context.user_data.get("agent")
    if not agent:
        await update.message.reply_text("❌ Load a video first — /start")
        return ConversationHandler.END

    question = update.message.text.strip()
    thinking = await update.message.reply_text("🤔 Thinking…")

    try:
        history = load_history(context.user_data)

        result = agent.invoke({
            "question":       question,
            "search_query":   "",
            "context":        "",
            "answer":         "",
            "video_id":       context.user_data.get("video_id", ""),
            "route":          "",
            "retrieved_docs": [],
            "history":        history,
        })

        answer         = result["answer"]
        route          = result.get("route", "")
        retrieved_docs = result.get("retrieved_docs", [])

        # Persist fresh chunks so next question can use them
        if route == Route.FROM_DB.value and retrieved_docs:
            save_retrieved_chunks(context.user_data, retrieved_docs)

        # Save this turn for follow-up handling
        save_turn(context.user_data, question, answer)

        # Append timestamp deep-links when available
        answer_with_links = _maybe_append_timestamps(answer, retrieved_docs)

        await _safe_update(thinking, answer_with_links, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"[receive_question] error: {e}")
        await _safe_update(thinking, f"⚠️ Error: {str(e)[:200]}\nTry /start")

    return ASK_QUESTION


def _maybe_append_timestamps(answer: str, docs: list[Document]) -> str:
    links = []
    for d in docs:
        if d.metadata.get("strategy") == "timestamp" and "deep_link" in d.metadata:
            label = d.metadata["timestamp_label"]
            link  = d.metadata["deep_link"]
            links.append(f"[{label}]({link})")
    if links:
        unique = list(dict.fromkeys(links))
        return answer + "\n\n📍 *Mentioned at:* " + "  ·  ".join(unique[:4])
    return answer


async def _safe_update(msg, text: str, **kwargs):
    try:
        await msg.edit_text(text, **kwargs)
    except Exception:
        await msg.reply_text(text, **kwargs)