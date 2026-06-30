"""
router.py — Question routing for YouTube RAG bot.

For every incoming question, a lightweight LLM call classifies it into one of
three routes BEFORE the expensive vector-DB search is considered:

  Route.FROM_CONTEXT  — answerable from the chunks already shown to the user
                        (last retrieval stored in user_data); skip DB search.
  Route.FROM_DB       — question is about the video but needs a fresh DB search.

Design decisions (per spec):
  - Only the *last* retrieval's chunks are kept as "memory" (in user_data).
  - Routing uses one small LLM call (fast, cheap) before the main generation.
  - DB search always backs up FROM_CONTEXT: if the context-only answer looks
    thin, the graph falls through to DB retrieval automatically.
  - Off-topic questions are answered normally; a short note is appended.
  - Recent chat history (last few turns) is shown to the classifier so
    follow-ups like "tell me more about that" route correctly instead of
    looking like off-topic noise.

FIXED vs. original:
  - `prev_context` join no longer crashes (was joining Document objects
    instead of their .page_content strings).
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── Route enum ─────────────────────────────────────────────────────────────────

class Route(str, Enum):
    FROM_CONTEXT = "from_context"   # answer from cached chunks
    FROM_DB      = "from_db"        # need a fresh vector-DB search
    OFF_TOPIC    = "off_topic"      # unrelated to the video


# ── Classifier prompt ──────────────────────────────────────────────────────────

_ROUTER_SYSTEM = """You are a routing assistant for a YouTube video Q&A bot.

Your ONLY job is to classify a user question into exactly one of three categories.
Respond with a single JSON object — nothing else, no markdown, no explanation.

Categories:
  "from_context" — The question can be fully answered using ONLY the provided
                   [PREVIOUS CONTEXT]. Choose this when the context already
                   contains the relevant information.
  "from_db"      — The question is about the video but the [PREVIOUS CONTEXT]
                   does not contain enough information to answer it. A fresh
                   search of the video transcript database is needed.

Use [RECENT CHAT HISTORY] to resolve follow-up questions ("tell me more about
that", "what else did he say?", "why?") — these refer to the previous turn's
topic, NOT a brand-new unrelated subject. A short follow-up referring back to
something already discussed should usually be "from_context" (if the previous
context still covers it) or "from_db" (if it needs more detail).

JSON format (respond with ONLY this):
{"route": "<from_context|from_db>", "reason": "<one short sentence>"}"""

_ROUTER_USER = """\
[RECENT CHAT HISTORY]
{history}

[PREVIOUS CONTEXT]
{prev_context}

[USER QUESTION]
{question}

Classify the question."""


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(no prior turns)"
    lines = []
    for q, a in history[-5:]:
        lines.append(f"User: {q}")
        lines.append(f"Bot: {a[:200]}")
    return "\n".join(lines)


def classify_question(
    question: str,
    prev_chunks: list[Document],
    llm: BaseChatModel,
    history: Optional[list[tuple[str, str]]] = None,
) -> tuple[Route, str]:
    """
    Calls the LLM classifier and returns (Route, reason_string).

    video_summary: the pre-generated summary fetched from ChromaDB
                   (type='summary', chunk_idx=-1). Used as the topic hint
                   so the classifier knows what the video is about from
                   question #1, not just after the first retrieval.

    history: list of (question, answer) tuples from recent turns, used so
             follow-up questions classify correctly.

    Falls back to Route.FROM_DB on any parse failure so the bot never
    silently drops a valid question.
    """

    messages = [
        SystemMessage(content=_ROUTER_SYSTEM),
        HumanMessage(content=_ROUTER_USER.format(
            history=_format_history(history or []),
            prev_context="\n---\n".join(d.page_content for d in prev_chunks) or "(none)",
            question=question,
        )),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

        data   = json.loads(raw)
        route  = Route(data["route"])
        reason = data.get("reason", "")
        logger.info(f"[router] → {route.value}  ({reason})")
        return route, reason

    except Exception as e:
        logger.warning(f"[router] Classification failed ({e}), defaulting to FROM_DB")
        return Route.FROM_DB, "parse error — defaulting to DB search"


# ── Context memory helpers (stored in user_data) ───────────────────────────────

CONTEXT_KEY = "last_retrieved_chunks"   # key used in telegram ContextTypes
HISTORY_KEY = "chat_history"
MAX_HISTORY_TURNS = 5


def save_retrieved_chunks(user_data: dict, docs: list[Document]) -> None:
    """Store the most-recently-retrieved chunks (overwrites previous)."""
    user_data[CONTEXT_KEY] = docs


def load_retrieved_chunks(user_data: dict) -> list[Document]:
    """Return the cached chunks, or an empty list if none yet."""
    return user_data.get(CONTEXT_KEY, [])


def clear_retrieved_chunks(user_data: dict) -> None:
    user_data.pop(CONTEXT_KEY, None)


def save_turn(user_data: dict, question: str, answer: str,
              max_turns: int = MAX_HISTORY_TURNS) -> None:
    """Append a (question, answer) turn to the rolling history, trimmed to max_turns."""
    history: list[tuple[str, str]] = user_data.setdefault(HISTORY_KEY, [])
    history.append((question, answer))
    user_data[HISTORY_KEY] = history[-max_turns:]


def load_history(user_data: dict) -> list[tuple[str, str]]:
    return user_data.get(HISTORY_KEY, [])


def clear_history(user_data: dict) -> None:
    user_data.pop(HISTORY_KEY, None)

