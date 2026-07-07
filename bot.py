"""
YouTube RAG Telegram Bot
Supports two providers via PROVIDER= in .env:
  - ollama   → local Ollama (dev / no cost)
  - bedrock  → AWS Bedrock (production / pay-per-token)

Stack: LangGraph + ChromaDB + BM25 (hybrid search) + cross-encoder rerank
       + youtube-transcript-api

FIXED / CHANGED vs. original:
  - Removed unused `List` import and the dead top-level SYSTEM_PROMPT
    (the real one lives in rag_graph.py and is the one actually used).
  - Collection name now also keys on CHUNK_STRATEGY, so re-indexing a video
    with a different strategy doesn't silently mix incompatible chunks into
    one Chroma collection.
  - Builds and persists a BM25 index alongside the Chroma vectorstore at
    indexing time (hybrid_search.BM25Index), used by rag_graph for hybrid
    retrieval.
  - Passes a history_fn and bm25_index_fn into build_routed_rag_graph so the
    RAG graph can do conversation-aware routing + hybrid search.
  - Removed manual language selection entirely. get_transcript() now lists
    all available transcripts for the video and picks a manually created
    one if available (falling back to auto-generated only if necessary),
    so the user never needs to specify a language. This also removes the
    ASK_LANG conversation step.
"""

import os
import re
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
from langchain_chroma import Chroma
from langchain_core.documents import Document

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# Chunking utilities
from chunking import (
    ChunkingConfig, Segment, build_documents,
    segments_from_fetched, count_tokens,
)

# Router, hybrid search, and routed graph
from router import load_retrieved_chunks, load_history
from hybrid_search import BM25Index
from rag_graph import build_routed_rag_graph, receive_question

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────
PROVIDER       = os.getenv("PROVIDER", "ollama").lower()  # "ollama" | "bedrock"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHROMA_DIR     = os.getenv("CHROMA_DIR", "./chroma_db")
BM25_DIR       = os.getenv("BM25_DIR", "./bm25_cache")
SIMILARITY_K   = int(os.getenv("SIMILARITY_K", 4))

# Ollama settings (used when PROVIDER=ollama)
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "gemma3:4b")
OLLAMA_EMBED    = os.getenv("OLLAMA_EMBED",    "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Bedrock settings (used when PROVIDER=bedrock)
AWS_REGION           = os.getenv("AWS_REGION",        "us-east-1")
BEDROCK_LLM_MODEL     = os.getenv("BEDROCK_LLM_MODEL", "amazon.nova-lite-v1:0")
BEDROCK_EMBED_MODEL   = os.getenv("BEDROCK_EMBED_MODEL", "cohere.embed-multilingual-v3")

# Chunking (configurable via .env)
CHUNK_STRATEGY   = os.getenv("CHUNK_STRATEGY", "timestamp")   # sentence | timestamp | semantic
CHUNK_TOKENS     = int(os.getenv("CHUNK_TOKENS",    300))
OVERLAP_TOKENS   = int(os.getenv("OVERLAP_TOKENS",   30))
SIMILARITY_THR   = float(os.getenv("SIMILARITY_THR", 0.75))   # semantic only

# Conversation states (language selection removed — handled automatically)
ASK_URL, ASK_QUESTION = range(2)


# ── Provider factory ────────────────────────────────────────────────────────────
def build_llm() -> BaseChatModel:
    """Returns the correct LLM based on PROVIDER."""
    if PROVIDER == "bedrock":
        from langchain_aws import ChatBedrockConverse
        logger.info(f"🟠 LLM: Bedrock / {BEDROCK_LLM_MODEL}")
        return ChatBedrockConverse(
            model=BEDROCK_LLM_MODEL,
            region_name=AWS_REGION,
            temperature=0.3,
            max_tokens=256,
        )
    else:
        from langchain_ollama import ChatOllama
        logger.info(f"🟢 LLM: Ollama / {OLLAMA_MODEL}")
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_predict=256,
            num_thread=os.cpu_count(),
            keep_alive="30m"
        )


def build_embeddings() -> Embeddings:
    """Returns the correct embedding model based on PROVIDER."""
    if PROVIDER == "bedrock":
        from langchain_aws import BedrockEmbeddings
        logger.info(f"🟠 Embeddings: Bedrock / {BEDROCK_EMBED_MODEL}")
        return BedrockEmbeddings(
            model_id=BEDROCK_EMBED_MODEL,
            region_name=AWS_REGION,
        )
    else:
        from langchain_ollama import OllamaEmbeddings
        logger.info(f"🟢 Embeddings: Ollama / {OLLAMA_EMBED}")
        return OllamaEmbeddings(
            model=OLLAMA_EMBED,
            base_url=OLLAMA_BASE_URL,
        )


# ── YouTube transcript ──────────────────────────────────────────────────────────
def extract_video_id(url: str) -> Optional[str]:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return match.group(1) if match else None


def get_transcript(video_id: str) -> tuple[str, list[Segment], str, str]:
    """
    Downloads a transcript for the video, automatically choosing a language.
    Prefers a manually created transcript (i.e. provided by the uploader,
    not auto-generated) if one exists in any language; otherwise falls back
    to an auto-generated transcript.

    Returns (full_text, segments, status_message, lang_code).
    """
    api = YouTubeTranscriptApi()

    try:
        transcript_list = list(api.list(video_id))
    except TranscriptsDisabled:
        raise TranscriptsDisabled(video_id)

    if not transcript_list:
        raise NoTranscriptFound(video_id, [], {})

    manual = [t for t in transcript_list if not t.is_generated]
    chosen = manual[0] if manual else transcript_list[0]

    fetched   = chosen.fetch()
    segments  = segments_from_fetched(fetched)
    full_text = " ".join(s.text for s in segments)

    kind = "manual" if not chosen.is_generated else "auto-generated"
    status = (
        f"✅ Transcript loaded ({len(segments)} segments) — "
        f"language: {chosen.language} [{chosen.language_code}], {kind}"
    )
    return full_text, segments, status, chosen.language_code


# ── Vector store ───────────────────────────────────────────────────────────────
def collection_name(video_id: str) -> str:
    # Keyed on provider AND chunk strategy so mismatched re-indexing can't
    # silently mix incompatible chunk shapes/metadata in one collection.
    return f"video_{video_id}_{PROVIDER}_{CHUNK_STRATEGY}"


def init_vectorstore(video_id: str) -> Chroma:
    return Chroma(
        collection_name=collection_name(video_id),
        embedding_function=build_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def bm25_cache_path(video_id: str) -> str:
    return os.path.join(BM25_DIR, f"{collection_name(video_id)}.pkl")


SUMMARY_PROMPT = """You are given a YouTube video transcript.
Write a concise summary 300 words covering:
- The main topic and purpose of the video
- Key points or arguments made
- Any notable conclusions
- Provide answer in the same language as the transcript.

Transcript (may be truncated):
{transcript}"""


def generate_summary(text: str, llm: BaseChatModel) -> str:
    """Generates a short summary of the full transcript using the LLM."""
    # Truncate by tokens (not raw characters) to stay within context limits.
    enc_tokens = count_tokens(text)
    if enc_tokens > 3000:
        # crude proportional character truncation based on measured token count
        ratio = 3000 / enc_tokens
        preview = text[: int(len(text) * ratio)]
    else:
        preview = text
    try:
        response = llm.invoke([
            HumanMessage(content=SUMMARY_PROMPT.format(transcript=preview))
        ])
        return response.content
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        return ""


def index_transcript(
    video_id: str,
    text: str,
    lang: str,
    llm: BaseChatModel,
    segments: list[Segment] | None = None,
) -> tuple[Chroma, int]:
    """
    Chunks the transcript using the selected strategy, upserts into ChromaDB,
    and builds/persists a BM25 keyword index for hybrid search.
    Also generates and stores a full-video summary as a special document.
    """
    cfg = ChunkingConfig(
        strategy=CHUNK_STRATEGY,
        chunk_tokens=CHUNK_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
        similarity_threshold=SIMILARITY_THR,
    )

    embeddings = build_embeddings()

    chunk_docs = build_documents(
        video_id=video_id,
        lang=lang,
        text=text,
        segments=segments,           # None is safe; timestamp falls back to sentence
        embeddings=embeddings if CHUNK_STRATEGY == "semantic" else None,
        config=cfg,
    )

    summary = generate_summary(text, llm)
    if summary:
        chunk_docs.append(Document(
            page_content=summary,
            metadata={"video_id": video_id, "lang": lang,
                        "chunk_idx": -1, "type": "summary"},
        ))

    vs = init_vectorstore(video_id)
    vs.add_documents(chunk_docs)
    n_docs = len(chunk_docs)
    logger.info(f"📦 Stored {n_docs} docs for {video_id} [{CHUNK_STRATEGY}]")

    # Build + persist BM25 index for hybrid search over the SAME chunk set
    try:
        bm25 = BM25Index.build(chunk_docs)
        bm25.save(bm25_cache_path(video_id))
        logger.info(f"🔎 BM25 index built and cached for {video_id}")
    except Exception as e:
        logger.warning(f"⚠️ BM25 index build failed ({e}); hybrid search will fall back to vector-only")

    return vs, n_docs


def load_bm25_index(video_id: str) -> Optional[BM25Index]:
    try:
        return BM25Index.load(bm25_cache_path(video_id))
    except Exception as e:
        logger.warning(f"⚠️ Could not load BM25 index for {video_id}: {e}")
        return None


# ── Telegram helpers ───────────────────────────────────────────────────────────
async def safe_update(msg, text: str, **kwargs):
    """edit_text with automatic fallback to reply_text on failure."""
    try:
        await msg.edit_text(text, **kwargs)
    except Exception:
        await msg.reply_text(text, **kwargs)


def _build_agent(context: ContextTypes.DEFAULT_TYPE, vs: Chroma, video_id: str):
    """Wires up the routed RAG graph with closures over fresh user_data."""
    llm = context.bot_data["llm"]

    def get_cached_chunks():
        return load_retrieved_chunks(context.user_data)

    def get_history():
        return load_history(context.user_data)

    def get_bm25():
        return load_bm25_index(video_id)

    return build_routed_rag_graph(
        vs, llm,
        prev_chunks_fn=get_cached_chunks,
        history_fn=get_history,
        bm25_index_fn=get_bm25,
    )


# ── Conversation handlers ──────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "👋 Hi! I analyze YouTube videos using their transcripts.\n\n"
        "📎 Send me a YouTube video link:",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ASK_URL


async def receive_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    url      = update.message.text.strip()
    video_id = extract_video_id(url)
    if not video_id:
        await update.message.reply_text(
            "❌ Couldn't recognize a YouTube link.\n"
            "Format: https://youtube.com/watch?v=XXXXXXXXXXX"
        )
        return ASK_URL

    context.user_data["video_id"] = video_id

    # ── Check if this video is already indexed ──────────────────────────
    vs = init_vectorstore(video_id)
    existing_ids = vs.get().get("ids", [])
    if existing_ids:
        # Already have chunks → skip transcript download entirely
        n_chunks = len(existing_ids)
        context.user_data["agent"] = _build_agent(context, vs, video_id)

        await update.message.reply_text(
            f"✅ Video already indexed – using existing data.\n"
            f"📦 {n_chunks} chunks\n"
            f"🤖 Provider: {'AWS Bedrock' if PROVIDER == 'bedrock' else 'Ollama (local)'}\n"
            f"✂️ Chunking: {CHUNK_STRATEGY}  ({CHUNK_TOKENS} tok, {OVERLAP_TOKENS} overlap)\n"
            f"🔎 Retrieval: hybrid (vector + BM25) + cross-encoder rerank\n\n"
            f"💬 Ask me anything about the video.\n"
            f"New video → /start  |  Exit → /cancel",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ASK_QUESTION

    # ── Not indexed yet → download & index ───────────────────────────────
    status_msg = await update.message.reply_text(
        "⏳ Downloading transcript (auto-detecting best available language)...",
        reply_markup=ReplyKeyboardRemove(),
    )

    try:
        transcript_text, segments, status, lang = get_transcript(video_id)
        total_tokens = count_tokens(transcript_text)

        await safe_update(
            status_msg,
            f"{status}\n📊 Tokens: {total_tokens}\n⏳ Indexing into vector store..."
        )

        llm = context.bot_data["llm"]
        vectorstore, n_chunks = index_transcript(
            video_id, transcript_text, lang, llm, segments=segments
        )

        context.user_data["agent"] = _build_agent(context, vectorstore, video_id)

        await safe_update(
            status_msg,
            f"✅ Video indexed!\n"
            f"🌐 Transcript language: {lang}\n"
            f"📦 {n_chunks} chunks (~{CHUNK_TOKENS} tokens, {OVERLAP_TOKENS}-token overlap)\n"
            f"🤖 Provider: {'AWS Bedrock' if PROVIDER == 'bedrock' else 'Ollama (local)'}\n"
            f"✂️ Chunking: {CHUNK_STRATEGY}  ({CHUNK_TOKENS} tok, {OVERLAP_TOKENS} overlap)\n"
            f"🔎 Retrieval: hybrid (vector + BM25) + cross-encoder rerank\n\n"
            f"💬 Ask me anything about the video.\n"
            f"New video → /start  |  Exit → /cancel"
        )
        return ASK_QUESTION

    except TranscriptsDisabled:
        await safe_update(status_msg,
            "❌ Transcripts are disabled for this video. Try another or /start")
        return ASK_URL

    except NoTranscriptFound:
        await safe_update(status_msg,
            "❌ No transcript is available for this video at all.\n"
            "Check the video page for available captions.\n"
            "Try /start with a different video.")
        return ASK_URL

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        await safe_update(status_msg, f"⚠️ Error: {str(e)[:200]}\nTry /start")
        return ASK_URL


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "👋 Session ended. For a new video — /start",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    provider_line = (
        "🟠 *Provider:* AWS Bedrock"
        if PROVIDER == "bedrock"
        else "🟢 *Provider:* Ollama (local)"
    )
    await update.message.reply_text(
        "🤖 *How I work:*\n\n"
        "1️⃣ /start — let's begin\n"
        "2️⃣ Send a YouTube video link\n"
        "3️⃣ I auto-detect the best transcript (preferring the original, "
        "non auto-generated one, in whatever language is available)\n"
        "4️⃣ I split the transcript using the configured chunking strategy\n"
        "5️⃣ I store chunks in ChromaDB + a BM25 keyword index\n"
        "6️⃣ I answer your questions using hybrid retrieval + reranking (RAG)\n\n"
        f"{provider_line}\n\n"
        "📌 *Commands:*\n"
        "/start — new video\n"
        "/cancel — end session\n"
        "/help — this message",
        parse_mode="Markdown",
    )


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("❌ TELEGRAM_TOKEN not found in .env")
    if PROVIDER not in ("ollama", "bedrock"):
        raise ValueError(f"❌ Unknown PROVIDER='{PROVIDER}'. Use 'ollama' or 'bedrock'.")

    os.makedirs(BM25_DIR, exist_ok=True)

    logger.info(f"🚀 Starting bot. Provider: {PROVIDER.upper()}")

    llm = build_llm()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.bot_data["llm"] = llm

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_URL:      [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_url)],
            ASK_QUESTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_question)],
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            CommandHandler("start",  start),
        ],
    )
    app.add_handler(conv_handler)
    app.add_handler(CommandHandler("help", help_command))

    logger.info("✅ Bot started!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()