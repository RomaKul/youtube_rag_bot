# 🎬 YouTube RAG Telegram Bot

Analyzes YouTube videos via transcripts. Supports two deployment modes via a single `.env` switch.

| Mode | `PROVIDER=ollama` | `PROVIDER=bedrock` |
|---|---|---|
| **Where it runs** | Your local machine | AWS EC2 (t3.micro) |
| **LLM** | gemma3:4b via Ollama | Amazon Nova Lite |
| **Embeddings** | nomic-embed-text | Cohere Embed Multilingual |
| **Cost** | Free | ~$0 from credits, then pay-per-use |
| **RAM needed** | 4+ GB (for Ollama) | 1 GB (EC2 t3.micro is enough) |
| **Best for** | Local dev & testing | Production deployment |

---

## 🛠️ Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
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
python bot.py
```

---

## 🟠 Mode 2: AWS Bedrock (Production)

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

### Step 4 — Deploy to EC2 (Free Tier)

```bash
# 1. Launch EC2 t3.micro with Amazon Linux 2023 or Ubuntu 24.04
#    (t3.micro = 1 GB RAM, enough since no Ollama)

# 2. SSH into your instance
ssh -i key.pem ec2-user@your-ec2-ip

# 3. Install Python
sudo apt update && sudo apt install -y python3-pip python3-venv git

# 4. Clone / upload your project
git clone https://github.com/you/ua_rag_bot.git
cd ua_rag_bot

# 5. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Create .env with PROVIDER=bedrock
nano .env

# 7. Run with auto-restart
pip install supervisor
# or simply use: nohup python bot.py &
python bot.py
```

### Keep the bot alive with systemd

```bash
sudo nano /etc/systemd/system/ytbot.service
```

```ini
[Unit]
Description=YouTube RAG Telegram Bot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/ua_rag_bot
ExecStart=/home/ubuntu/ua_rag_bot/venv/bin/python bot.py
Restart=always
RestartSec=10
EnvironmentFile=/home/ubuntu/ua_rag_bot/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ytbot
sudo systemctl start ytbot
sudo systemctl status ytbot
```

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
ua_rag_bot/
├── bot.py            # main file — all logic
├── requirements.txt
├── .env              # your secrets (never commit this)
├── .env.example      # template
├── README.md
└── chroma_db/        # vector store (auto-created)
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
