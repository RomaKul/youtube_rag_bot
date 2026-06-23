# 🎬 YouTube RAG Telegram Bot

Analyzes YouTube videos via their transcripts. Ask questions — the bot answers based on the video's content.

## Stack
- **Ollama + gemma3:4b** — local LLM (no API, free)
- **Ollama + nomic-embed-text** — local embeddings
- **ChromaDB** — local vector database
- **LangGraph** — RAG pipeline orchestration
- **youtube-transcript-api** — transcript downloading

---

## 🚀 Step-by-step setup

### Step 1 — Install Ollama and models

```bash
# Download Ollama from https://ollama.com

# Model for answering (~3 GB)
ollama pull gemma3:4b

# Model for embeddings (~274 MB)
ollama pull nomic-embed-text
```

### Step 2 — Create a Telegram bot

1. Find @BotFather on Telegram
2. Send `/newbot` and follow the instructions
3. Copy the token you receive

### Step 3 — Configure the project

```bash
cp .env.example .env
# Paste your TELEGRAM_TOKEN into .env
```

### Step 4 — Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
# or venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Step 5 — Run

```bash
# Terminal 1
ollama serve

# Terminal 2
python bot.py
```

---

## 💬 Usage flow

```
User: /start
Bot: Send me a YouTube video link

User: https://youtube.com/watch?v=XXXXXXX
Bot: Specify the transcript language [uk / en / de ...]

User: en — English
Bot: ✅ Loaded 847 segments, 12 chunks (300 tokens, 30 overlap)

User: What is this video about?
Bot: [answer based on the transcript]

User: /start   ← new video
```

---

## ⚙️ Chunking parameters

| Parameter | Value |
|---|---|
| Chunk size | 300 tokens |
| Overlap | 30 tokens (10%) |
| Tokenizer | tiktoken cl100k_base (or character-based fallback) |
| Retrieval | Top-4 semantically similar chunks |

---

## 📁 Project structure

```
ua_rag_bot/
├── bot.py           # main file
├── requirements.txt
├── .env             # your secrets
├── .env.example     # template
├── README.md
└── chroma_db/       # vector store (created automatically)
```

---

## 🔧 Troubleshooting

**Ollama isn't responding**
```bash
curl http://localhost:11434/api/tags
ollama serve   # start it if not running
```

**Transcript not found**
- Check that captions are enabled on the video's page
- Try a different language (auto-generated English is often available)

**Slow responses on CPU**
- Normal: 5-15 seconds per response
- Lighter alternative: `OLLAMA_MODEL=llama3.2:3b`

**YouTube blocks the request (403 / PoToken errors)**
- This is a YouTube anti-bot measure, not a bug in this code
- Try upgrading: `pip install --upgrade youtube-transcript-api`
- Or pass browser cookies to the API (see library docs for `cookies` support)
