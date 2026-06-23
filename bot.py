"""
YouTube RAG Telegram Bot
Stack: Ollama (gemma3:4b) + LangGraph + ChromaDB + youtube-transcript-api
"""

import os
import re
import logging
from typing import TypedDict, Optional
from dotenv import load_dotenv

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ConversationHandler, filters, ContextTypes
)

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document

from langgraph.graph import StateGraph, START, END

import tiktoken
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR      = os.getenv("CHROMA_DIR", "./chroma_db")

CHUNK_TOKENS    = 300
OVERLAP_RATIO   = 0.10   # 10% overlap
OVERLAP_TOKENS  = int(CHUNK_TOKENS * OVERLAP_RATIO)  # = 30 tokens

# ConversationHandler states
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
Reference specific moments from the video when possible."""


# ── Utilities ──────────────────────────────────────────────────────────────────
def extract_video_id(url: str) -> Optional[str]:
    """Extracts video_id from various YouTube URL formats."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _get_encoder():
    """Returns a tiktoken encoder, or None if unavailable."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Counts the number of tokens in the text."""
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text))
    # Fallback: ~4 characters = 1 token (GPT rule of thumb)
    return len(text) // 4


def split_into_chunks(text: str, chunk_size: int = CHUNK_TOKENS,
                      overlap: int = OVERLAP_TOKENS) -> list[str]:
    """
    Splits text into chunks of chunk_size tokens with `overlap` tokens
    of overlap between consecutive chunks.
    Uses tiktoken if available, otherwise falls back to a character-based method.
    """
    enc = _get_encoder()

    if enc:
        # Precise counting via tiktoken
        tokens = enc.encode(text)
        chunks = []
        start  = 0
        while start < len(tokens):
            end        = min(start + chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            chunks.append(chunk_text)
            if end == len(tokens):
                break
            start = end - overlap
        return chunks

    else:
        # Fallback: split by characters (~4 chars = 1 token)
        char_size    = chunk_size * 4
        char_overlap = overlap * 4
        chunks       = []
        start        = 0
        while start < len(text):
            end = min(start + char_size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - char_overlap
        return chunks


def get_transcript(video_id: str, lang: str) -> tuple[str, str]:
    """
    Downloads the transcript from YouTube.
    Returns (text, status_message).
    """
    api = YouTubeTranscriptApi()
    try:
        fetched  = api.fetch(video_id, languages=[lang])
        snippets = [entry.text for entry in fetched]
        return " ".join(snippets), f"✅ Transcript loaded ({len(snippets)} segments)"

    except NoTranscriptFound:
        # Try to fall back to any available transcript
        try:
            transcript_list = api.list(video_id)
            for t in transcript_list:
                fetched  = t.fetch()
                snippets = [entry.text for entry in fetched]
                return " ".join(snippets), (
                    f"⚠️ No transcript found for language '{lang}'. "
                    f"Used instead: {t.language_code}"
                )
        except Exception as e:
            raise NoTranscriptFound(video_id, [lang]) from e

    except TranscriptsDisabled:
        raise TranscriptsDisabled(video_id)


# ── Vector store ───────────────────────────────────────────────────────────────
def init_vectorstore(video_id: str) -> Chroma:
    """Initializes or loads the vector store for a specific video."""
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    vectorstore = Chroma(
        collection_name=f"video_{video_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def index_transcript(video_id: str, transcript_text: str, lang: str) -> tuple[Chroma, int]:
    """
    Splits the transcript into chunks and writes them to ChromaDB.
    Returns (vectorstore, number_of_chunks).
    """
    chunks = split_into_chunks(transcript_text)

    docs = [
        Document(
            page_content=chunk,
            metadata={
                "video_id":  video_id,
                "lang":      lang,
                "chunk_idx": i,
                "total":     len(chunks),
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    vectorstore = init_vectorstore(video_id)

    # If the video was already indexed, clear the old collection
    existing = vectorstore.get()
    if existing["ids"]:
        vectorstore.delete(existing["ids"])
        logger.info(f"🗑️ Cleared old collection for {video_id}")

    vectorstore.add_documents(docs)
    logger.info(f"📦 Stored {len(docs)} chunks for video {video_id}")

    return vectorstore, len(docs)


# ── LangGraph state and nodes ────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:    str
    context:     str
    answer:      str
    video_id:    str


def make_retrieve_node(vectorstore: Chroma):
    def retrieve(state: RAGState) -> RAGState:
        logger.info(f"🔍 Searching context for: {state['question']}")
        docs = vectorstore.similarity_search(state["question"], k=4)
        context = "\n\n---\n\n".join(
            f"[Chunk {d.metadata.get('chunk_idx', '?')}]\n{d.page_content}"
            for d in docs
        )
        return {**state, "context": context}
    return retrieve


def make_generate_node(llm: ChatOllama):
    def generate(state: RAGState) -> RAGState:
        logger.info("✍️ Generating answer...")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Video transcript context:\n{state['context']}\n\n"
                f"Question: {state['question']}\n\n"
                "Provide a detailed answer based on the context above."
            )),
        ]
        response = llm.invoke(messages)
        return {**state, "answer": response.content}
    return generate


def build_rag_graph(vectorstore: Chroma, llm: ChatOllama):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve_node(vectorstore))
    graph.add_node("generate", make_generate_node(llm))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# ── Telegram ConversationHandler ─────────────────────────────────────────────────
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
            "Send it in the format: https://youtube.com/watch?v=XXXXXXXXXXX"
        )
        return ASK_URL

    context.user_data["video_id"] = video_id
    context.user_data["url"]      = url

    # Language selection keyboard
    keyboard = [[f"{code} — {name}"] for code, name in SUPPORTED_LANGS.items()]

    await update.message.reply_text(
        f"✅ Video found: `{video_id}`\n\n"
        "🌐 Specify the transcript language:",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return ASK_LANG


async def safe_update(msg, text: str, **kwargs):
    """
    Safely updates a message.
    If edit_text fails (BadRequest) — sends a new message instead.
    """
    try:
        await msg.edit_text(text, **kwargs)
    except Exception:
        await msg.reply_text(text, **kwargs)


async def receive_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text     = update.message.text.strip()
    video_id = context.user_data.get("video_id")

    lang = text[:2].lower()
    if lang not in SUPPORTED_LANGS:
        await update.message.reply_text("❌ Please choose a language from the keyboard below.")
        return ASK_LANG

    context.user_data["lang"] = lang

    status_msg = await update.message.reply_text(
        "⏳ Downloading transcript...",
        reply_markup=ReplyKeyboardRemove(),
    )

    try:
        transcript_text, status = get_transcript(video_id, lang)
        total_tokens = count_tokens(transcript_text)

        await safe_update(
            status_msg,
            f"{status}\n"
            f"📊 Tokens: {total_tokens}\n"
            f"⏳ Splitting into chunks and indexing..."
        )

        llm = context.bot_data["llm"]
        vectorstore, n_chunks = index_transcript(video_id, transcript_text, lang)
        agent = build_rag_graph(vectorstore, llm)

        context.user_data["agent"]    = agent
        context.user_data["n_chunks"] = n_chunks

        chunk_info = (
            f"📦 Chunks: {n_chunks} "
            f"(~{CHUNK_TOKENS} tokens, {OVERLAP_TOKENS}-token overlap)"
        )

        await safe_update(
            status_msg,
            f"✅ Video indexed!\n"
            f"{chunk_info}\n\n"
            f"💬 Now ask me anything about the video.\n"
            f"For a new video — /start\n"
            f"To exit — /cancel"
        )
        return ASK_QUESTION

    except TranscriptsDisabled:
        await safe_update(
            status_msg,
            "❌ Transcripts are disabled for this video.\n"
            "Try a different video or /start"
        )
        return ASK_URL

    except NoTranscriptFound:
        await safe_update(
            status_msg,
            f"❌ No transcript found for language '{lang}' on this video.\n"
            f"Check the video page for available languages.\n"
            f"Try /start and pick a different language.",
        )
        return ASK_URL

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        await safe_update(
            status_msg,
            f"⚠️ Error: {str(e)[:200]}\n"
            "Try /start"
        )
        return ASK_URL


async def receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    question = update.message.text.strip()
    agent    = context.user_data.get("agent")

    if not agent:
        await update.message.reply_text("❌ Load a video first — /start")
        return ConversationHandler.END

    thinking = await update.message.reply_text("🤔 Searching the transcript...")

    try:
        result = agent.invoke({
            "question": question,
            "context":  "",
            "answer":   "",
            "video_id": context.user_data.get("video_id", ""),
        })
        await safe_update(thinking, result["answer"])

    except Exception as e:
        logger.error(f"Agent error: {e}")
        await safe_update(
            thinking,
            "⚠️ Generation error. Make sure Ollama is running:\n`ollama serve`",
            parse_mode="Markdown",
        )

    return ASK_QUESTION   # stay in the question-answering state


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "👋 Session ended. For a new video — /start",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *How I work:*\n\n"
        "1️⃣ /start — let's begin\n"
        "2️⃣ Send a YouTube video link\n"
        "3️⃣ Choose the transcript language\n"
        "4️⃣ I download the transcript and split it into 300-token chunks (10% overlap)\n"
        "5️⃣ I store them in a ChromaDB vector database\n"
        "6️⃣ I answer your questions about the video\n\n"
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

    logger.info(f"🚀 Starting bot. LLM: {OLLAMA_MODEL}, Embeddings: {EMBED_MODEL}")

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,
        num_predict=1024,
    )

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
