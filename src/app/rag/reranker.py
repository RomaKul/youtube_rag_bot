"""
reranker.py — Cross-encoder / Bedrock re-ranking of retrieved candidates.

Vector / hybrid search is optimized for recall (cast a wide net). A
reranker jointly scores (query, candidate) pairs and is much more precise
at judging actual relevance — but too slow/costly to run over a whole
corpus, so we only use it to re-sort a small candidate set (e.g. top 20)
down to the final top-k (e.g. 4) handed to the LLM.

Two backends are supported, selected via PROVIDER in .env (same variable
used elsewhere in the project for LLM/embeddings):

  - PROVIDER=bedrock → uses AWS Bedrock's managed rerank model, configured
    via BEDROCK_RERANK_MODEL in .env (e.g. cohere.rerank-v3-5:0).
  - PROVIDER=ollama (or anything else) → uses a local cross-encoder model
    (sentence-transformers), loaded lazily and cached at module level so
    it survives across requests within one process.

Both paths degrade gracefully on failure: if the configured backend is
unavailable for any reason, we fall back to returning the first top_n
docs unchanged (i.e. whatever order the upstream retriever produced).
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
PROVIDER             = os.getenv("PROVIDER", "ollama").lower()  # "ollama" | "bedrock"
AWS_REGION           = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_RERANK_MODEL = os.getenv("BEDROCK_RERANK_MODEL", "cohere.rerank-v3-5:0")

_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None          # lazy singleton (cross-encoder)
_bedrock_client = None  # lazy singleton (boto3 bedrock-agent-runtime client)


# ── Local cross-encoder backend ─────────────────────────────────────────────
def _get_cross_encoder():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"[rerank] Loading cross-encoder {_CROSS_ENCODER_MODEL_NAME} (CPU)…")
        _model = CrossEncoder(_CROSS_ENCODER_MODEL_NAME)
    return _model


def _rerank_cross_encoder(query: str, docs: list[Document], top_n: int) -> list[Document]:
    try:
        model = _get_cross_encoder()
    except Exception as e:
        logger.warning(f"[rerank] Cross-encoder unavailable ({e}); skipping rerank")
        return docs[:top_n]

    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.warning(f"[rerank] Cross-encoder prediction failed ({e}); skipping rerank")
        return docs[:top_n]

    ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
    logger.info(
        f"[rerank] (cross-encoder) {len(docs)} candidates → top {top_n} "
        f"(best score={ranked[0][1]:.3f}, worst kept={ranked[top_n-1][1]:.3f})"
    )
    return [d for d, _ in ranked[:top_n]]


# ── AWS Bedrock backend ──────────────────────────────────────────────────────
def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        logger.info(
            f"[rerank] Creating Bedrock Agent Runtime client "
            f"(region={AWS_REGION}, model={BEDROCK_RERANK_MODEL})…"
        )
        _bedrock_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
    return _bedrock_client


def _bedrock_model_arn() -> str:
    # Bedrock's rerank API expects a full model ARN, not just the model id.
    if BEDROCK_RERANK_MODEL.startswith("arn:"):
        return BEDROCK_RERANK_MODEL
    return f"arn:aws:bedrock:{AWS_REGION}::foundation-model/{BEDROCK_RERANK_MODEL}"


def _rerank_bedrock(query: str, docs: list[Document], top_n: int) -> list[Document]:
    try:
        client = _get_bedrock_client()
    except Exception as e:
        logger.warning(f"[rerank] Bedrock client unavailable ({e}); skipping rerank")
        return docs[:top_n]

    text_sources = [
        {
            "type": "INLINE",
            "inlineDocumentSource": {
                "type": "TEXT",
                "textDocument": {"text": d.page_content},
            },
        }
        for d in docs
    ]

    try:
        response = client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=text_sources,
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": min(top_n, len(docs)),
                    "modelConfiguration": {
                        "modelArn": _bedrock_model_arn(),
                    },
                },
            },
        )
    except Exception as e:
        logger.warning(f"[rerank] Bedrock rerank call failed ({e}); skipping rerank")
        return docs[:top_n]

    try:
        results = response["results"]  # each: {"index": int, "relevanceScore": float}
        ranked = [(docs[r["index"]], r["relevanceScore"]) for r in results]
    except Exception as e:
        logger.warning(f"[rerank] Unexpected Bedrock rerank response shape ({e}); skipping rerank")
        return docs[:top_n]

    if not ranked:
        return docs[:top_n]

    logger.info(
        f"[rerank] (bedrock/{BEDROCK_RERANK_MODEL}) {len(docs)} candidates → top {len(ranked)} "
        f"(best score={ranked[0][1]:.3f}, worst kept={ranked[-1][1]:.3f})"
    )
    return [d for d, _ in ranked[:top_n]]


# ── Public API ───────────────────────────────────────────────────────────────
def rerank(
    query: str,
    docs: list[Document],
    top_n: int = 4,
) -> list[Document]:
    """
    Re-scores `docs` against `query` and returns the top_n highest-scoring
    documents, sorted best-first.

    Backend is chosen by PROVIDER (.env):
      - "bedrock" → AWS Bedrock rerank model (BEDROCK_RERANK_MODEL)
      - otherwise → local cross-encoder

    Degrades gracefully on any failure: falls back to returning the first
    top_n docs unchanged (i.e. whatever order the upstream retriever
    produced).
    """
    if not docs:
        return []
    if len(docs) <= top_n:
        return docs

    if PROVIDER == "bedrock":
        return _rerank_bedrock(query, docs, top_n)
    return _rerank_cross_encoder(query, docs, top_n)