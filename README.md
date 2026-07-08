# 🎬 YouTube RAG Telegram Bot

Analyzes YouTube videos via transcripts. Supports two deployment modes via a single `.env` switch. For `bot_aws.py` default is bedrock

| Mode | `PROVIDER=ollama` | `PROVIDER=bedrock` | `bot_aws.py` |
|---|---|---|---|
| **Where it runs** | Your local machine | our local machine | AWS EC2 (t3.micro) |
| **LLM** | gemma3:4b via Ollama | Amazon Nova Lite | Amazon Nova Lite |
| **Embeddings** | nomic-embed-text | Cohere Embed Multilingual | Cohere Embed Multilingual |
| **Cost** | Free | ~$0 from credits, then pay-per-use |  ~$0 from credits, then pay-per-use | 
| **RAM needed** | 4+ GB (for Ollama) | 1 GB is enough | 1 GB (EC2 t3.micro is enough) |
| **Best for** | Local dev & testing | Local dev (faster) | Production deployment |

---

## 🛠️ Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
# or: venv\Scripts\activate  # Windows

# installs the app package itself (editable) + shared/base deps
pip install -e .
```

Then install the extra deps for the mode you're running:

```bash
# local (Ollama) mode
pip install -r requirements/local.txt

# AWS (Bedrock) mode
pip install -r requirements/aws.txt
```

---

## 🟢 Mode 1: Local (Ollama)

### Step 1 — Install Ollama and pull models

```bash
# Download from https://ollama.com
ollama pull gemma3:4b           # ~3 GB, LLM
ollama pull nomic-embed-text    # ~274 MB, embeddings
```

### Step 2 — Create a Telegram bot via @BotFather

### Step 3 — Configure .env

```bash
cp .env.example .env
```

Edit `.env`:
```
TELEGRAM_TOKEN=your_token
PROVIDER=ollama
```

### Step 4 — Run

```bash
# Terminal 1
ollama serve

# Terminal 2
python -m app.bots.bot_local
```

---

## 🟠 Mode 2: Local with AWS Bedrock

### Step 1 — Create AWS account and enable Bedrock models

1. Go to https://aws.amazon.com and create an account
2. Open **Amazon Bedrock → Model access** in your AWS console
3. Request access to:
   - `Amazon Nova Lite`
   - `Cohere Embed Multilingual v3`
4. Wait for approval (usually instant)

### Step 2 — Create IAM credentials

1. Go to **IAM → Users → Create user**
2. Attach policy: `AmazonBedrockFullAccess`
3. Create **Access Key** → copy `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

> **On EC2:** attach an IAM Role instead of using keys — safer and no credentials in .env

### Step 3 — Configure .env

```
TELEGRAM_TOKEN=your_token
PROVIDER=bedrock

AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

BEDROCK_LLM_MODEL=amazon.nova-lite-v1:0
BEDROCK_EMBED_MODEL=cohere.embed-multilingual-v3
```

## 🟠 Mode 3: AWS Bedrock (Production)

Refer to [AWS Deployment Guide](infra/AWS_DEPLOYMENT.md)

---

## 💰 Bedrock cost estimate

| Action | Tokens | Cost |
|---|---|---|
| Index 1 hour video transcript | ~10K tokens | ~$0.001 |
| Answer 1 question (in + out) | ~2K tokens | ~$0.0003 |
| 100 questions/day for a month | ~6M tokens | ~$0.90/month |

The $200 AWS credit covers ~200,000 questions before you pay anything.

---

## 💬 Bot usage

```
/start → send YouTube link → choose language → ask questions
/cancel → end session
/help → show info including active provider
```

---

## 📁 Project structure

```
youtube_rag_bot/
├── README.md
├── pyproject.toml          # editable install: pip install -e .
├── .env                    # your secrets (never commit this)
├── .env.example            # template
├── .gitignore
│
├── requirements/
│   ├── base.txt            # shared deps
│   ├── local.txt           # Ollama-mode extras
│   └── aws.txt             # Bedrock-mode extras
│
├── infra/
│   ├── Dockerfile.cloud
│   └── .dockerignore
│
├── src/
│   └── app/
│       ├── bots/
│       │   ├── bot_local.py    # entrypoint: PROVIDER=ollama / PROVIDER=bedrock
│       │   └── bot_aws.py      # entrypoint: aws
│       │
│       ├── rag/                # shared by both bots
│       │   ├── rag_graph.py    # langgraph structure
│       │   ├── router.py       # off_topic / from_db / from_context routing
│       │   ├── chunking.py     # transcript chunking logic
│       │   ├── hybrid_search.py# vector + BM25 retrieval
│       │   └── reranker.py     # 20 chunks → top 4
│       │
│       ├── storage/
│       │   ├── s3_transcript_store.py   # S3 get/put
│       │   ├── vectorstore_aws.py       # OpenSearch get/put
│       │   └── sqs_transcript_queue.py  # SQS queue
│       │
│       └── watcher/
│           └── local_watcher.py   # queue → S3 writer (used by bot_aws)
│
├── tests/
│   └── test_vectorstore_aws.py
│
└── data/                    # gitignored, local caches
    ├── chroma_db/
    └── bm25_cache/
```

---

## 🔧 Troubleshooting

**Ollama not responding**
```bash
curl http://localhost:11434/api/tags
ollama serve
```

**Bedrock AccessDeniedException**
- Check IAM policy includes `AmazonBedrockFullAccess`
- Check model access is approved in Bedrock console
- Check `AWS_REGION` matches where you enabled the models

**YouTube 403 / PoToken error**
- Upgrade: `pip install --upgrade youtube-transcript-api`
- Or pass browser cookies — see library docs

**Slow on CPU (Ollama)**
- Normal: 5–15 sec/response
- Switch to lighter model: `OLLAMA_MODEL=llama3.2:3b`

**`ModuleNotFoundError: No module named 'app'`**
- Make sure you ran `pip install -e .` from the project root (the folder containing `pyproject.toml`) while your venv was active.
- Re-run `pip install -e .` only after adding new subpackages/folders under `src/app/` — everyday edits to existing files take effect immediately without reinstalling.