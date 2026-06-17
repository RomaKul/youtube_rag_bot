"""
Україномовний YouTube RAG Telegram-бот
Стек: Ollama (gemma3:4b) + LangGraph + ChromaDB + youtube-transcript-api
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

# ── Константи ──────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR      = os.getenv("CHROMA_DIR", "./chroma_db")

CHUNK_TOKENS    = 300
OVERLAP_RATIO   = 0.10   # 10% оверлеп
OVERLAP_TOKENS  = int(CHUNK_TOKENS * OVERLAP_RATIO)  # = 30 токенів

# Стани ConversationHandler
ASK_URL, ASK_LANG, ASK_QUESTION = range(3)

SUPPORTED_LANGS = {
    "uk": "🇺🇦 Українська",
    "en": "🇬🇧 Англійська",
    "de": "🇩🇪 Німецька",
    "fr": "🇫🇷 Французька",
    "pl": "🇵🇱 Польська",
    "es": "🇪🇸 Іспанська",
    "ru": "🇷🇺 Російська",
}

SYSTEM_PROMPT = """Ти україномовний AI-асистент що аналізує YouTube відео.
Відповідай ВИКЛЮЧНО українською мовою.
Використовуй наданий контекст із субтитрів відео для точних відповідей.
Якщо відповідь не міститься в контексті — чесно скажи про це.
Посилайся на конкретні моменти відео якщо це можливо."""


# ── Утиліти ────────────────────────────────────────────────────────────────────
def extract_video_id(url: str) -> Optional[str]:
    """Витягує video_id з різних форматів YouTube URL."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _get_encoder():
    """Повертає tiktoken encoder або None якщо недоступний."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Підраховує кількість токенів у тексті."""
    enc = _get_encoder()
    if enc:
        return len(enc.encode(text))
    # Fallback: ~4 символи = 1 токен (правило GPT)
    return len(text) // 4


def split_into_chunks(text: str, chunk_size: int = CHUNK_TOKENS,
                      overlap: int = OVERLAP_TOKENS) -> list[str]:
    """
    Розбиває текст на чанки по chunk_size токенів з overlap токенів перекриття.
    Використовує tiktoken якщо доступний, інакше — символьний метод.
    """
    enc = _get_encoder()

    if enc:
        # Точний підрахунок через tiktoken
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
        # Fallback: ділимо по символах (~4 символи = 1 токен)
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
    Завантажує субтитри з YouTube.
    Повертає (текст, повідомлення про статус).
    """
    api = YouTubeTranscriptApi()
    try:
        fetched  = api.fetch(video_id, languages=[lang])
        snippets = [entry.text for entry in fetched]
        return " ".join(snippets), f"✅ Субтитри завантажено ({len(snippets)} фраз)"

    except NoTranscriptFound:
        # Спробуємо отримати авто-згенеровані субтитри
        try:
            transcript_list = api.list(video_id)
            # Шукаємо будь-який доступний субтитр
            for t in transcript_list:
                fetched  = t.fetch()
                snippets = [entry.text for entry in fetched]
                return " ".join(snippets), (
                    f"⚠️ Субтитри мовою '{lang}' не знайдені. "
                    f"Використано: {t.language_code}"
                )
        except Exception as e:
            raise NoTranscriptFound(video_id, [lang]) from e

    except TranscriptsDisabled:
        raise TranscriptsDisabled(video_id)


# ── Векторна база ──────────────────────────────────────────────────────────────
def init_vectorstore(video_id: str) -> Chroma:
    """Ініціалізує або завантажує векторну базу для конкретного відео."""
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
    Розбиває субтитри на чанки і записує у ChromaDB.
    Повертає (vectorstore, кількість чанків).
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

    # Якщо відео вже проіндексовано — очищаємо стару колекцію
    existing = vectorstore.get()
    if existing["ids"]:
        vectorstore.delete(existing["ids"])
        logger.info(f"🗑️ Очищено стару колекцію для {video_id}")

    vectorstore.add_documents(docs)
    logger.info(f"📦 Збережено {len(docs)} чанків для відео {video_id}")

    return vectorstore, len(docs)


# ── LangGraph стан і вузли ─────────────────────────────────────────────────────
class RAGState(TypedDict):
    question:    str
    context:     str
    answer:      str
    video_id:    str


def make_retrieve_node(vectorstore: Chroma):
    def retrieve(state: RAGState) -> RAGState:
        logger.info(f"🔍 Пошук контексту для: {state['question']}")
        docs = vectorstore.similarity_search(state["question"], k=4)
        context = "\n\n---\n\n".join(
            f"[Чанк {d.metadata.get('chunk_idx', '?')}]\n{d.page_content}"
            for d in docs
        )
        return {**state, "context": context}
    return retrieve


def make_generate_node(llm: ChatOllama):
    def generate(state: RAGState) -> RAGState:
        logger.info("✍️ Генерація відповіді...")
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Контекст із субтитрів відео:\n{state['context']}\n\n"
                f"Питання: {state['question']}\n\n"
                "Дай розгорнуту відповідь українською на основі контексту."
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


# ── Telegram ConversationHandler ───────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "👋 Привіт! Я аналізую YouTube відео через субтитри.\n\n"
        "📎 Надішли посилання на YouTube відео:",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ASK_URL


async def receive_url(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    url      = update.message.text.strip()
    video_id = extract_video_id(url)

    if not video_id:
        await update.message.reply_text(
            "❌ Не вдалося розпізнати YouTube посилання.\n"
            "Надішли у форматі: https://youtube.com/watch?v=XXXXXXXXXXX"
        )
        return ASK_URL

    context.user_data["video_id"] = video_id
    context.user_data["url"]      = url

    # Клавіатура вибору мови
    lang_buttons = [[code, name] for code, name in SUPPORTED_LANGS.items()]
    keyboard = [[f"{code} — {name}"] for code, name in SUPPORTED_LANGS.items()]

    await update.message.reply_text(
        f"✅ Відео знайдено: `{video_id}`\n\n"
        "🌐 Вкажи мову субтитрів відео:",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return ASK_LANG


async def receive_lang(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text     = update.message.text.strip()
    video_id = context.user_data.get("video_id")

    # Витягуємо код мови (перші 2 символи)
    lang = text[:2].lower()
    if lang not in SUPPORTED_LANGS:
        await update.message.reply_text("❌ Вибери мову з клавіатури нижче.")
        return ASK_LANG

    context.user_data["lang"] = lang

    status_msg = await update.message.reply_text(
        "⏳ Завантажую субтитри...",
        reply_markup=ReplyKeyboardRemove(),
    )

    try:
        transcript_text, status = get_transcript(video_id, lang)
        total_tokens = count_tokens(transcript_text)

        await status_msg.edit_text(
            f"{status}\n"
            f"📊 Токенів: {total_tokens}\n"
            f"⏳ Розбиваю на чанки та індексую..."
        )

        llm         = context.bot_data["llm"]
        vectorstore, n_chunks = index_transcript(video_id, transcript_text, lang)
        agent       = build_rag_graph(vectorstore, llm)

        context.user_data["agent"]    = agent
        context.user_data["n_chunks"] = n_chunks

        chunk_info = (
            f"📦 Чанків: {n_chunks} "
            f"(~{CHUNK_TOKENS} токенів, оверлеп {OVERLAP_TOKENS} токенів)"
        )

        await status_msg.edit_text(
            f"✅ Відео проіндексовано!\n"
            f"{chunk_info}\n\n"
            f"💬 Тепер задавай питання про відео.\n"
            f"Для нового відео — /start\n"
            f"Для виходу — /cancel"
        )
        return ASK_QUESTION

    except TranscriptsDisabled:
        await status_msg.edit_text(
            "❌ Субтитри вимкнені для цього відео.\n"
            "Спробуй інше відео або /start"
        )
        return ASK_URL

    except NoTranscriptFound:
        await status_msg.edit_text(
            f"❌ Субтитри мовою `{lang}` не знайдені для цього відео.\n"
            f"Доступні мови можна переглянути на сторінці відео.\n"
            f"Спробуй /start і вкажи іншу мову.",
            parse_mode="Markdown",
        )
        return ASK_URL

    except Exception as e:
        logger.error(f"Помилка індексації: {e}")
        await status_msg.edit_text(
            f"⚠️ Помилка: {str(e)[:200]}\n"
            "Спробуй /start"
        )
        return ASK_URL


async def receive_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    question = update.message.text.strip()
    agent    = context.user_data.get("agent")

    if not agent:
        await update.message.reply_text("❌ Спочатку завантаж відео — /start")
        return ConversationHandler.END

    thinking = await update.message.reply_text("🤔 Шукаю відповідь у субтитрах...")

    try:
        result = agent.invoke({
            "question": question,
            "context":  "",
            "answer":   "",
            "video_id": context.user_data.get("video_id", ""),
        })
        await thinking.edit_text(result["answer"])

    except Exception as e:
        logger.error(f"Помилка агента: {e}")
        await thinking.edit_text(
            "⚠️ Помилка генерації. Переконайся що Ollama запущена:\n`ollama serve`",
            parse_mode="Markdown",
        )

    return ASK_QUESTION   # залишаємось у стані питань


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "👋 Сесію завершено. Для нового відео — /start",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Як я працюю:*\n\n"
        "1️⃣ /start — починаємо\n"
        "2️⃣ Надсилаєш посилання на YouTube відео\n"
        "3️⃣ Вибираєш мову субтитрів\n"
        "4️⃣ Я завантажую субтитри та розбиваю на чанки по 300 токенів (оверлеп 10%)\n"
        "5️⃣ Зберігаю у векторну базу ChromaDB\n"
        "6️⃣ Відповідаю на твої питання про відео\n\n"
        "📌 *Команди:*\n"
        "/start — нове відео\n"
        "/cancel — завершити сесію\n"
        "/help — ця довідка",
        parse_mode="Markdown",
    )


# ── Запуск ─────────────────────────────────────────────────────────────────────
def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("❌ TELEGRAM_TOKEN не знайдено в .env")

    logger.info(f"🚀 Запуск бота. LLM: {OLLAMA_MODEL}, Embeddings: {EMBED_MODEL}")

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

    logger.info("✅ Бот запущено!")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
