"""
YouTube RAG Telegram Bot
Supports two providers via PROVIDER= in .env:
  - ollama   → local Ollama (dev / no cost)
  - bedrock  → AWS Bedrock (production / pay-per-token)

Stack: LangGraph + ChromaDB + youtube-transcript-api
"""

import os
import re
import logging
from typing import TypedDict, Optional, List
from dotenv import load_dotenv

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, filters, ContextTypes
)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

import tiktoken
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

# Import chunking utilities
from chunking import (
    ChunkingConfig, Segment, build_documents,
    segments_from_fetched, format_timestamp, youtube_deep_link,
    _count_tokens,                  # moved from bot.py
)

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
SIMILARITY_K   = int(os.getenv("SIMILARITY_K", 4))

# Ollama settings (used when PROVIDER=ollama)
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "gemma3:4b")
OLLAMA_EMBED    = os.getenv("OLLAMA_EMBED",    "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Bedrock settings (used when PROVIDER=bedrock)
AWS_REGION        = os.getenv("AWS_REGION",        "us-east-1")
BEDROCK_LLM_MODEL = os.getenv("BEDROCK_LLM_MODEL", "amazon.nova-lite-v1:0")
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL",
                                "cohere.embed-multilingual-v3")

# Chunking (now configurable via .env)
CHUNK_STRATEGY   = os.getenv("CHUNK_STRATEGY", "timestamp")   # sentence | timestamp | semantic
CHUNK_TOKENS     = int(os.getenv("CHUNK_TOKENS",    300))
OVERLAP_TOKENS   = int(os.getenv("OVERLAP_TOKENS",   30))
SIMILARITY_THR   = float(os.getenv("SIMILARITY_THR", 0.75))   # semantic only

# Conversation states
ASK_URL, ASK_LANG, ASK_QUESTION = range(3)

SUPPORTED_LANGS = {
    "uk": "🇺🇦 Ukrainian",
    "en": "🇬🇧 English",
    "de": "🇩🇪 German",
    "fr": "🇫🇷 French",
    "pl": "🇵🇱 Polish",
    "es": "🇪🇸 Spanish",
    "ru": "🇷🇺 Russian",
}

SYSTEM_PROMPT = """You are an AI assistant that analyzes YouTube videos.
Use the provided video transcript context to answer accurately.
If the answer is not contained in the context, honestly say so.
When citing specific parts, refer to them by their timestamp labels (e.g., [Timestamp: 01:23]) if available, otherwise by chunk number.
Reference specific moments from the video when possible."""


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
            max_tokens=1024,
        )
    else:
        from langchain_ollama import ChatOllama
        logger.info(f"🟢 LLM: Ollama / {OLLAMA_MODEL}")
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_predict=1024,
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


def get_transcript(video_id: str, lang: str) -> tuple[str, list[Segment], str]:
    """Downloads transcript. Returns (full_text, segments, status_message)."""
    api = YouTubeTranscriptApi()
    try:
        fetched   = api.fetch(video_id, languages=[lang])
        segments  = segments_from_fetched(fetched)
        full_text = " ".join(s.text for s in segments)
        return full_text, segments, f"✅ Transcript loaded ({len(segments)} segments)"

    except NoTranscriptFound:
        try:
            for t in api.list(video_id):
                fetched   = t.fetch()
                segments  = segments_from_fetched(fetched)
                full_text = " ".join(s.text for s in segments)
                return full_text, segments, (
                    f"⚠️ No '{lang}' transcript. Using: {t.language_code}"
                )
        except Exception as e:
            raise NoTranscriptFound(video_id, [lang]) from e

    except TranscriptsDisabled:
        raise TranscriptsDisabled(video_id)


# ── Vector store ───────────────────────────────────────────────────────────────
def init_vectorstore(video_id: str) -> Chroma:
    embed_tag = PROVIDER # "ollama" or "bedrock"
    return Chroma(
        collection_name=f"video_{video_id}_{embed_tag}",
        embedding_function=build_embeddings(),
        persist_directory=CHROMA_DIR,
    )


SUMMARY_PROMPT = """You are given a YouTube video transcript.
Write a concise summary (5-7 sentences) covering:
- The main topic and purpose of the video
- Key points or arguments made
- Any notable conclusions

Transcript (may be truncated):
{transcript}"""


def generate_summary(text: str, llm: BaseChatModel) -> str:
    """Generates a short summary of the full transcript using the LLM."""
    # Use first ~3000 tokens worth of characters to stay within context limits
    preview = text[:3000]
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
    Chunks the transcript using the selected strategy and upserts into ChromaDB.
    Also generates and stores a full‑video summary as a special document.
    """
    cfg = ChunkingConfig(
        strategy=CHUNK_STRATEGY,
        chunk_tokens=CHUNK_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
        similarity_threshold=SIMILARITY_THR,
    )

    embeddings = build_embeddings()   # reuse existing factory

    chunk_docs = build_documents(
        video_id=video_id,
        lang=lang,
        text=text,
        segments=segments,           # None is safe; timestamp falls back to sentence
        embeddings=embeddings if CHUNK_STRATEGY == "semantic" else None,
        config=cfg,
    )

    # Summary doc (unchanged logic)
    summary = generate_summary(text, llm)
    if summary:
        chunk_docs.append(Document(
            page_content=summary,
            metadata={"video_id": video_id, "lang": lang,
                      "chunk_idx": -1, "type": "summary"},
        ))

    vs = init_vectorstore(video_id)
    existing = vs.get()
    if existing["ids"]:
        vs.delete(existing["ids"])
    vs.add_documents(chunk_docs)
    logger.info(f"📦 Stored {len(chunk_docs)} docs for {video_id} [{CHUNK_STRATEGY}]")
    return vs, len(chunk_docs)


# ── LangGraph ──────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:       str
    context:        str
    answer:         str
    video_id:       str
    retrieved_docs: list   # new — populated by retrieve node


def make_retrieve_node(vectorstore: Chroma):
    def retrieve(state: RAGState) -> RAGState:
        logger.info(f"🔍 Retrieving context for: {state['question']}")
        docs = vectorstore.similarity_search(state["question"], k=SIMILARITY_K)
        # Build context with appropriate labels (timestamp if available)
        context_parts = []
        for d in docs:
            if "timestamp_label" in d.metadata:
                label = f"[Timestamp: {d.metadata['timestamp_label']}]"
            else:
                label = f"[Chunk {d.metadata.get('chunk_idx', '?')}]"
            context_parts.append(f"{label}\n{d.page_content}")
        context = "\n\n---\n\n".join(context_parts)
        return {**state, "context": context, "retrieved_docs": docs}
    return retrieve


def make_generate_node(llm: BaseChatModel):
    def generate(state: RAGState) -> RAGState:
        logger.info("✍️ Generating answer...")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Video transcript context:\n{state['context']}\n\n"
                f"Question: {state['question']}\n\n"
                "Provide a detailed answer based on the context above. "
                "Answer in the same language the question is written in."
            )),
        ]
        response = llm.invoke(messages)
        return {**state, "answer": response.content}
    return generate


def build_rag_graph(vectorstore: Chroma, llm: BaseChatModel):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve_node(vectorstore))
    graph.add_node("generate", make_generate_node(llm))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# ── Telegram helpers ───────────────────────────────────────────────────────────
async def safe_update(msg, text: str, **kwargs):
    """edit_text with automatic fallback to reply_text on failure."""
    try:
        await msg.edit_text(text, **kwargs)
    except Exception:
        await msg.reply_text(text, **kwargs)


# ── Timestamp citation helper ──────────────────────────────────────────────────
def _append_timestamp_citations(answer: str, docs: list) -> str:
    """Appends '📍 Mentioned at: X:XX, Y:YY' if chunks carry timestamps."""
    links = []
    for d in docs:
        if d.metadata.get("strategy") == "timestamp" and "deep_link" in d.metadata:
            label = d.metadata["timestamp_label"]
            link  = d.metadata["deep_link"]
            links.append(f"[{label}]({link})")
    if links:
        unique = list(dict.fromkeys(links))   # deduplicate, preserve order
        return answer + "\n\n📍 *Mentioned at:* " + "  ·  ".join(unique[:4])
    return answer


# ── Conversation handlers ──────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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
    keyboard = [[f"{code} — {name}"] for code, name in SUPPORTED_LANGS.items()]
    await update.message.reply_text(
        f"✅ Video found: `{video_id}`\n\n🌐 Specify the transcript language:",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return ASK_LANG


async def receive_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    lang     = update.message.text.strip()[:2].lower()
    video_id = context.user_data.get("video_id")

    if lang not in SUPPORTED_LANGS:
        await update.message.reply_text("❌ Please choose a language from the keyboard.")
        return ASK_LANG

    context.user_data["lang"] = lang
    status_msg = await update.message.reply_text(
        "⏳ Downloading transcript...",
        reply_markup=ReplyKeyboardRemove(),
    )

    try:
        transcript_text, segments, status = get_transcript(video_id, lang)
        total_tokens = _count_tokens(transcript_text)

        await safe_update(
            status_msg,
            f"{status}\n📊 Tokens: {total_tokens}\n⏳ Indexing into vector store..."
        )

        llm = context.bot_data["llm"]
        vectorstore, n_chunks = index_transcript(
            video_id, transcript_text, lang, llm, segments=segments
        )
        context.user_data["agent"] = build_rag_graph(vectorstore, llm)

        await safe_update(
            status_msg,
            f"✅ Video indexed!\n"
            f"📦 {n_chunks} chunks (~{CHUNK_TOKENS} tokens, {OVERLAP_TOKENS}-token overlap)\n"
            f"🤖 Provider: {'AWS Bedrock' if PROVIDER == 'bedrock' else 'Ollama (local)'}\n"
            f"✂️ Chunking: {CHUNK_STRATEGY}  ({CHUNK_TOKENS} tok, {OVERLAP_TOKENS} overlap)\n\n"
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
            f"❌ No '{lang}' transcript found.\n"
            "Check the video page for available captions.\n"
            "Try /start with a different language.")
        return ASK_URL

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        await safe_update(status_msg, f"⚠️ Error: {str(e)[:200]}\nTry /start")
        return ASK_URL


async def receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    agent = context.user_data.get("agent")
    if not agent:
        await update.message.reply_text("❌ Load a video first — /start")
        return ConversationHandler.END

    thinking = await update.message.reply_text("🤔 Searching the transcript...")
    try:
        result = agent.invoke({
            "question": update.message.text.strip(),
            "context":  "",
            "answer":   "",
            "video_id": context.user_data.get("video_id", ""),
        })
        final_answer = result["answer"]
        # Append timestamp citations if using timestamp strategy
        if CHUNK_STRATEGY == "timestamp":
            retrieved_docs = result.get("retrieved_docs", [])
            final_answer = _append_timestamp_citations(final_answer, retrieved_docs)
        await safe_update(thinking, final_answer, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Agent error: {e}")
        provider_hint = (
            "Make sure Ollama is running:\n`ollama serve`"
            if PROVIDER == "ollama"
            else "Check your AWS credentials and region in .env"
        )
        await safe_update(thinking, f"⚠️ Generation error.\n{provider_hint}",
                          parse_mode="Markdown")

    return ASK_QUESTION


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
        "3️⃣ Choose the transcript language\n"
        "4️⃣ I split the transcript using the configured chunking strategy\n"
        "5️⃣ I store them in a ChromaDB vector database\n"
        "6️⃣ I answer your questions using RAG\n\n"
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

    logger.info(f"🚀 Starting bot. Provider: {PROVIDER.upper()}")

    llm = build_llm()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.bot_data["llm"] = llm

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_URL:      [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_url)],
            ASK_LANG:     [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_lang)],
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