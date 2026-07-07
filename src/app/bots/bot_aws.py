"""
bot_aws.py — AWS-deployed variant of bot.py.

Run THIS file in AWS (ECS/Fargate/EC2), not bot.py. The local machine runs
local_watcher.py separately and is the only thing that talks to YouTube
directly.

Differences from bot.py:
  - get_transcript(): checks S3 first; if missing, enqueues an SQS job for
    the local watcher and polls S3 until the result lands (with a timeout).
  - Vector store: OpenSearch Serverless (vectorstore_aws.py) instead of
    local Chroma.
  - BM25 doc set: stored/loaded from S3 (s3_transcript_store.py) instead of
    a local pickle file, since ECS/Fargate containers don't have persistent
    local disk across deploys/restarts.
  - Everything else (chunking, routing, RAG graph, reranking) is unchanged —
    those modules don't care where the bytes came from. In fact rag_graph.py
    and hybrid_search.py have no runtime dependency on langchain_chroma at
    all, so `langchain-chroma` does NOT need to be in this image's
    requirements.txt — see the note at the bottom of this file.

Required new env vars (on top of bot.py's):
  S3_BUCKET                e.g. "youtube-rag-bot"
  SQS_REQUEST_QUEUE_URL    transcript-request queue (bot → local watcher)
  SQS_RESULT_QUEUE_URL     optional; speeds up the wait vs. pure S3 polling
  OPENSEARCH_ENDPOINT      OpenSearch Serverless collection host

CHANGED: dropped manual language selection entirely, mirroring bot.py. The
ASK_LANG conversation step and SUPPORTED_LANGS keyboard are gone —
local_watcher.py now auto-detects the best available transcript (preferring
a manually created one, falling back to auto-generated), so the user only
ever needs to send a URL. `lang` is still threaded through as the
auto-detected language code (used for chunk metadata + status messages),
it's just no longer something the user picks.
"""

import os
import re
import time
import logging
from typing import Optional
from dotenv import load_dotenv

from telegram import Update, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, filters, ContextTypes
)

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from app.rag.chunking import ChunkingConfig, Segment, build_documents, count_tokens
from app.rag.router import load_retrieved_chunks, load_history
from app.rag.hybrid_search import BM25Index
from app.rag.rag_graph import build_routed_rag_graph, receive_question

import app.storage.s3_transcript_store as s3store
import app.storage.sqs_transcript_queue as queue
import app.storage.vectorstore_aws as vs_aws

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────
PROVIDER       = os.getenv("PROVIDER", "bedrock").lower()  # bedrock is the natural choice in AWS
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
SIMILARITY_K   = int(os.getenv("SIMILARITY_K", 4))

AWS_REGION           = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_LLM_MODEL    = os.getenv("BEDROCK_LLM_MODEL", "amazon.nova-lite-v1:0")
BEDROCK_EMBED_MODEL  = os.getenv("BEDROCK_EMBED_MODEL", "cohere.embed-multilingual-v3")

CHUNK_STRATEGY   = os.getenv("CHUNK_STRATEGY", "timestamp")
CHUNK_TOKENS     = int(os.getenv("CHUNK_TOKENS",    300))
OVERLAP_SENTANCES   = int(os.getenv("OVERLAP_SENTANCES",   1))
SIMILARITY_THR   = float(os.getenv("SIMILARITY_THR", 0.75))

TRANSCRIPT_WAIT_TIMEOUT_S = int(os.getenv("TRANSCRIPT_WAIT_TIMEOUT_S", 90))
TRANSCRIPT_POLL_INTERVAL_S = int(os.getenv("TRANSCRIPT_POLL_INTERVAL_S", 4))

# Conversation states (language selection removed — handled automatically)
ASK_URL, ASK_QUESTION = range(2)


# ── Provider factory (bedrock only — ollama doesn't make sense to run in AWS) ──
def build_llm() -> BaseChatModel:
    from langchain_aws import ChatBedrockConverse
    logger.info(f"🟠 LLM: Bedrock / {BEDROCK_LLM_MODEL}")
    return ChatBedrockConverse(
        model=BEDROCK_LLM_MODEL, region_name=AWS_REGION,
        temperature=0.3, max_tokens=1024,
    )


def build_embeddings() -> Embeddings:
    from langchain_aws import BedrockEmbeddings
    logger.info(f"🟠 Embeddings: Bedrock / {BEDROCK_EMBED_MODEL}")
    return BedrockEmbeddings(model_id=BEDROCK_EMBED_MODEL, region_name=AWS_REGION)


def extract_video_id(url: str) -> Optional[str]:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


# ── Transcript: S3-first, SQS-to-local fallback ────────────────────────────────
def get_transcript_via_s3_and_local_worker(video_id: str) -> tuple[str, list[Segment], str]:
    """
    1. Check S3 — if already fetched (by this or any prior request), use it.
    2. Otherwise enqueue an SQS job for local_watcher.py and poll S3 until
       the transcript appears or TRANSCRIPT_WAIT_TIMEOUT_S elapses.

    No language is requested — local_watcher.py auto-detects the best
    available transcript for the video. Returns (text, segments, lang_code).
    """
    cached = s3store.get_transcript(video_id)
    if cached:
        text, seg_dicts, lang_code = cached
        segments = [Segment(**d) for d in seg_dicts]
        return text, segments, lang_code

    queue.enqueue_transcript_request(video_id)

    deadline = time.time() + TRANSCRIPT_WAIT_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(TRANSCRIPT_POLL_INTERVAL_S)
        cached = s3store.get_transcript(video_id)
        if cached:
            text, seg_dicts, lang_code = cached
            segments = [Segment(**d) for d in seg_dicts]
            return text, segments, lang_code

    raise TimeoutError(
        f"No transcript appeared in S3 within {TRANSCRIPT_WAIT_TIMEOUT_S}s. "
        "Is local_watcher.py running on your machine?"
    )


# ── Vector store (OpenSearch Serverless) ───────────────────────────────────────
def collection_name(video_id: str) -> str:
    return f"video-{video_id}-{PROVIDER}-{CHUNK_STRATEGY}".lower()


SUMMARY_PROMPT = """You are given a YouTube video transcript.
Write a concise summary (5-7 sentences) covering:
- The main topic and purpose of the video
- Key points or arguments made
- Any notable conclusions
- Provide answer in the same language as the transcript.

Transcript (may be truncated):
{transcript}"""


def index_transcript_aws(video_id: str, text: str, lang: str, llm: BaseChatModel,
                          segments: list[Segment]) -> tuple[object, int]:
    embeddings = build_embeddings()
    index_name = collection_name(video_id)

    already_indexed = vs_aws.index_exists(index_name)
    vstore = vs_aws.init_vectorstore(index_name, embeddings)

    if already_indexed:
        logger.info(f"♻️ OpenSearch index {index_name} already exists — reusing")
        return vstore, -1  # unknown count without an extra query; fine for the status message

    cfg = ChunkingConfig(
        strategy=CHUNK_STRATEGY, chunk_tokens=CHUNK_TOKENS,
        overlap_sentences=OVERLAP_SENTANCES, similarity_threshold=SIMILARITY_THR,
    )
    chunk_docs = build_documents(
        video_id=video_id, lang=lang, text=text, segments=segments,
        embeddings=embeddings if CHUNK_STRATEGY == "semantic" else None,
        config=cfg,
    )

    preview_tokens = count_tokens(text)
    preview = text[: int(len(text) * min(1.0, 3000 / max(preview_tokens, 1)))]
    try:
        summary = llm.invoke([HumanMessage(content=SUMMARY_PROMPT.format(transcript=preview))]).content
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        summary = ""

    if summary:
        chunk_docs.append(Document(
            page_content=summary,
            metadata={"video_id": video_id, "lang": lang, "chunk_idx": -1, "type": "summary"},
        ))

    vstore.add_documents(chunk_docs)
    logger.info(f"📦 Stored {len(chunk_docs)} docs in OpenSearch index {index_name}")

    try:
        s3store.put_bm25_docs(index_name, chunk_docs)
        logger.info(f"🔎 BM25 doc set uploaded to S3 for {index_name}")
    except Exception as e:
        logger.warning(f"⚠️ BM25 upload failed ({e}); hybrid search falls back to vector-only")

    return vstore, len(chunk_docs)


def load_bm25_index(video_id: str) -> Optional[BM25Index]:
    docs = s3store.get_bm25_docs(collection_name(video_id))
    if docs is None:
        return None
    return BM25Index(docs)


# ── Telegram helpers / handlers (mirrors bot.py) ───────────────────────────────
async def safe_update(msg, text: str, **kwargs):
    try:
        await msg.edit_text(text, **kwargs)
    except Exception:
        await msg.reply_text(text, **kwargs)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "👋 Hi! I analyze YouTube videos using their transcripts.\n\n📎 Send me a YouTube video link:",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ASK_URL


async def receive_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    url = update.message.text.strip()
    video_id = extract_video_id(url)
    if not video_id:
        await update.message.reply_text(
            "❌ Couldn't recognize a YouTube link.\nFormat: https://youtube.com/watch?v=XXXXXXXXXXX"
        )
        return ASK_URL
    context.user_data["video_id"] = video_id

    status_msg = await update.message.reply_text(
        f"✅ Video found: `{video_id}`\n⏳ Checking S3 / requesting from local fetcher if needed…",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardRemove(),
    )

    try:
        text, segments, lang = get_transcript_via_s3_and_local_worker(video_id)

        total_tokens = count_tokens(text)
        await safe_update(
            status_msg,
            f"✅ Transcript ready (language: {lang or 'unknown'})\n📊 Tokens: {total_tokens}\n⏳ Indexing…"
        )

        llm = context.bot_data["llm"]
        vstore, n_chunks = index_transcript_aws(video_id, text, lang, llm, segments)

        def get_cached_chunks():
            return load_retrieved_chunks(context.user_data)

        def get_history():
            return load_history(context.user_data)

        def get_bm25():
            return load_bm25_index(video_id)

        context.user_data["agent"] = build_routed_rag_graph(
            vstore, llm,
            prev_chunks_fn=get_cached_chunks,
            history_fn=get_history,
            bm25_index_fn=get_bm25,
        )

        chunk_line = f"📦 {n_chunks} chunks" if n_chunks >= 0 else "📦 Using existing index"
        await safe_update(
            status_msg,
            f"✅ Video indexed!\n🌐 Transcript language: {lang or 'unknown'}\n{chunk_line}\n"
            f"🤖 Provider: AWS Bedrock\n"
            f"☁️ Vector store: OpenSearch Serverless\n"
            f"🔎 Retrieval: hybrid (vector + BM25) + cross-encoder rerank\n\n"
            f"💬 Ask me anything about the video.\nNew video → /start  |  Exit → /cancel"
        )
        return ASK_QUESTION

    except TimeoutError as e:
        await safe_update(status_msg, f"⚠️ {e}\nTry again in a bit, or /start.")
        return ASK_URL
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        await safe_update(status_msg, f"⚠️ Error: {str(e)[:200]}\nTry /start")
        return ASK_URL


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("👋 Session ended. For a new video — /start", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *How I work:*\n\n"
        "1️⃣ /start — let's begin\n2️⃣ Send a YouTube video link\n"
        "3️⃣ I auto-detect the best transcript (preferring the original, "
        "non auto-generated one, in whatever language is available) via a "
        "local worker (YouTube blocks cloud IPs)\n"
        "4️⃣ I store chunks in OpenSearch Serverless + a BM25 index in S3\n"
        "5️⃣ I answer your questions using hybrid retrieval + reranking (RAG)\n\n"
        "🟠 *Provider:* AWS Bedrock\n\n"
        "📌 *Commands:*\n/start — new video\n/cancel — end session\n/help — this message",
        parse_mode="Markdown",
    )


def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("❌ TELEGRAM_TOKEN not found in .env")
    for required in ("S3_BUCKET", "SQS_REQUEST_QUEUE_URL", "OPENSEARCH_ENDPOINT"):
        if not os.getenv(required):
            raise ValueError(f"❌ {required} not set — required for the AWS deployment")

    logger.info("🚀 Starting AWS-hosted bot")
    llm = build_llm()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.bot_data["llm"] = llm

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_URL:      [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_url)],
            ASK_QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_question)],
        },
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("start", start)],
    )
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", help_command))

    logger.info("✅ Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()