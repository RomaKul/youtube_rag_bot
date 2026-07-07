# AWS Cloud Deployment Guide

## Architecture

YouTube blocks transcript-API requests from most cloud-provider IP ranges (AWS/GCP/Azure),
so the bot runs in AWS but transcript *fetching* must happen from your local machine.

```
┌──────────────────────┐        SQS        ┌────────────────────────────┐
│   YOUR LOCAL PC      │ ─────────────────► │   AWS ECS / Fargate        │
│                      │                    │                            │
│  app/watcher/        │        S3          │  app/bots/bot_aws.py       │
│  local_watcher.py    │ ◄────────────────► │                            │
│                      │                    │  ┌─────────────────────┐   │
│  Fetches YouTube     │                    │  │ OpenSearch Serverless│   │
│  transcripts         │                    │  │ (vector index)      │   │
│  (not IP-blocked)    │                    │  └─────────────────────┘   │
│                      │                    │                            │
└──────────────────────┘                    │  S3 (transcripts + BM25)   │
                                            └────────────────────────────┘
```

**Flow:**
1. User sends a YouTube link to the Telegram bot (running in AWS)
2. Bot checks S3 — if transcript already cached, uses it directly
3. If not cached, bot pushes a job to SQS and waits (up to 90s)
4. Your local `app/watcher/local_watcher.py` picks up the SQS job, fetches transcript, uploads to S3
5. Bot gets the transcript from S3, chunks it, indexes into OpenSearch + BM25
6. Bot answers questions via hybrid search + cross-encoder rerank + LLM

---

## Prerequisites

Install on your local machine:

```bash
# AWS CLI
pip install awscli

# Docker Desktop — https://docs.docker.com/desktop/

# Verify
aws --version
docker --version

# Configure AWS credentials
aws configure
# AWS Access Key ID:     <your key>
# AWS Secret Access Key: <your secret>
# Default region name:   us-east-1
# Default output format: json

# Save your account ID — you'll need it throughout
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo $AWS_ACCOUNT_ID
```

---

## Step 1 — AWS Resources (one-time setup)

### 1.1 S3 Bucket

```bash
aws s3 mb s3://youtube-rag-bot-${AWS_ACCOUNT_ID} --region us-east-1

# Block all public access
aws s3api put-public-access-block \
  --bucket youtube-rag-bot-${AWS_ACCOUNT_ID} \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

### 1.2 SQS Queues

```bash
# Request queue: bot → local watcher
aws sqs create-queue \
  --queue-name yt-rag-transcript-requests \
  --region us-east-1

# Result queue: local watcher → bot (optional but recommended)
aws sqs create-queue \
  --queue-name yt-rag-transcript-results \
  --region us-east-1
```

Save the `QueueUrl` values from the output — you'll need them in the `.env`.

### 1.3 OpenSearch Serverless

Do this in the **AWS Console** (not CLI — the serverless policies are complex to script):

1. Go to **OpenSearch Service → Serverless → Collections → Create collection**
2. Name: `yt-rag-bot`, Type: **Vector search**
3. Create an **Encryption policy** (AWS-owned key is fine):
   - Name: `yt-rag-encryption`
   - Resource: `Collection/yt-rag-bot`
4. Create a **Network policy**:
   - Name: `yt-rag-network`
   - Type: Public (simplest to start; restrict to VPC later)
   - Resource: `Collection/yt-rag-bot`
5. Create a **Data access policy** — this is the critical one:
   - Name: `yt-rag-data-access`
   - Add a rule with these principals:

   ```json
   [
     "arn:aws:iam::<ACCOUNT_ID>:role/yt-rag-bot-task-role",
     "arn:aws:iam::<ACCOUNT_ID>:user/<your-iam-user>"
   ]
   ```

   Give them these permissions on `Collection/yt-rag-bot` and `Index/yt-rag-bot/*`:
   - `aoss:CreateIndex`
   - `aoss:DescribeIndex`
   - `aoss:ReadDocument`
   - `aoss:WriteDocument`
   - `aoss:UpdateIndex`
   - `aoss:DeleteIndex`

6. Note the **Collection endpoint** — looks like:
   `abc123xyz.us-east-1.aoss.amazonaws.com`

> ⚠️ **If you skip the Data access policy or use the wrong role ARN, the bot will
> get 403 errors when trying to read/write vectors, even though IAM permissions
> look correct. OpenSearch Serverless uses its own access control layer on top
> of IAM.**

### 1.4 Bedrock Model Access

In the AWS Console:
1. Go to **Amazon Bedrock → Model access**
2. Request access to:
   - `Amazon Nova Lite` (LLM)
   - `Cohere Embed Multilingual v3` (embeddings)
3. Approval is usually instant for these models

### 1.5 IAM Roles

#### ECS Task Execution Role (pulls images from ECR, reads secrets)

```bash
# Create role
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach AWS managed policies
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
```

#### ECS Task Role (what the running bot container can do)

```bash
# Create role
aws iam create-role \
  --role-name yt-rag-bot-task-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ecs-tasks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach permissions policy
aws iam put-role-policy \
  --role-name yt-rag-bot-task-role \
  --policy-name yt-rag-bot-policy \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"s3:GetObject\",\"s3:PutObject\",\"s3:HeadObject\",\"s3:ListBucket\"],
        \"Resource\": [
          \"arn:aws:s3:::youtube-rag-bot-${AWS_ACCOUNT_ID}\",
          \"arn:aws:s3:::youtube-rag-bot-${AWS_ACCOUNT_ID}/*\"
        ]
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"sqs:SendMessage\",\"sqs:ReceiveMessage\",\"sqs:DeleteMessage\",\"sqs:GetQueueAttributes\"],
        \"Resource\": \"arn:aws:sqs:us-east-1:${AWS_ACCOUNT_ID}:yt-rag-*\"
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"bedrock:InvokeModel\",\"bedrock:InvokeModelWithResponseStream\"],
        \"Resource\": \"*\"
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"aoss:APIAccessAll\"],
        \"Resource\": \"arn:aws:aoss:us-east-1:${AWS_ACCOUNT_ID}:collection/*\"
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"logs:CreateLogGroup\",\"logs:CreateLogStream\",\"logs:PutLogEvents\"],
        \"Resource\": \"*\"
      }
    ]
  }"
```

> ⚠️ **`aoss:APIAccessAll` in IAM is necessary but not sufficient.** The bot's
> role ARN must also appear in the OpenSearch Serverless **Data access policy**
> (Step 1.3 above). Both must be in place — either one alone causes 403 errors.

#### Verify roles exist

```bash
aws iam get-role --role-name ecsTaskExecutionRole --query Role.Arn --output text
aws iam get-role --role-name yt-rag-bot-task-role --query Role.Arn --output text
aws iam list-attached-role-policies --role-name ecsTaskExecutionRole
```

### 1.6 Telegram Token in Secrets Manager

```bash
aws secretsmanager create-secret \
  --name yt-rag-bot/telegram-token \
  --secret-string "YOUR_BOT_TOKEN_HERE"
```

### 1.7 ECR Repository

```bash
aws ecr create-repository \
  --repository-name youtube-rag-bot \
  --region us-east-1

# Note the repositoryUri from output:
# 123456789.dkr.ecr.us-east-1.amazonaws.com/youtube-rag-bot
```

### 1.8 ECS Cluster and CloudWatch Logs

```bash
aws ecs create-cluster --cluster-name youtube-rag-bot-cluster

aws logs create-log-group --log-group-name /ecs/youtube-rag-bot
```

---

## Step 2 — Build and Push Docker Image

From the **project root** (`youtube-rag-bot/`, the folder containing `pyproject.toml` and `src/`):

```bash
export ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/youtube-rag-bot"

# Login to ECR
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Build (Dockerfile installs pyproject.toml + src/ and runs `python -m app.bots.bot_aws`)
docker build -f infra/Dockerfile.cloud -t youtube-rag-bot:latest .

# Tag and push
docker tag youtube-rag-bot:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest
```

---

## Step 3 — ECS Task Definition

Save as `infra/ecs-task-definition.json` — replace all `<PLACEHOLDERS>`:

```json
{
  "family": "youtube-rag-bot",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/yt-rag-bot-task-role",
  "containerDefinitions": [
    {
      "name": "bot",
      "image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/youtube-rag-bot:latest",
      "essential": true,
      "environment": [
        {"name": "PROVIDER",               "value": "bedrock"},
        {"name": "AWS_REGION",             "value": "us-east-1"},
        {"name": "S3_BUCKET",              "value": "youtube-rag-bot-<ACCOUNT_ID>"},
        {"name": "SQS_REQUEST_QUEUE_URL",  "value": "https://sqs.us-east-1.amazonaws.com/<ACCOUNT_ID>/yt-rag-transcript-requests"},
        {"name": "SQS_RESULT_QUEUE_URL",   "value": "https://sqs.us-east-1.amazonaws.com/<ACCOUNT_ID>/yt-rag-transcript-results"},
        {"name": "OPENSEARCH_ENDPOINT",    "value": "<YOUR_COLLECTION>.us-east-1.aoss.amazonaws.com"},
        {"name": "BEDROCK_LLM_MODEL",      "value": "amazon.nova-lite-v1:0"},
        {"name": "BEDROCK_EMBED_MODEL",    "value": "cohere.embed-multilingual-v3"},
        {"name": "CHUNK_STRATEGY",         "value": "timestamp"},
        {"name": "CHUNK_TOKENS",           "value": "300"},
        {"name": "OVERLAP_SENTANCES",         "value": "1"},
        {"name": "TRANSCRIPT_WAIT_TIMEOUT_S", "value": "90"}
      ],
      "secrets": [
        {
          "name": "TELEGRAM_TOKEN",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<ACCOUNT_ID>:secret:yt-rag-bot/telegram-token"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/youtube-rag-bot",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "bot"
        }
      }
    }
  ]
}
```

Register it:

```bash
aws ecs register-task-definition \
  --cli-input-json file://infra/ecs-task-definition.json
```

---

## Step 4 — Launch ECS Service

```bash
# Get default VPC and subnet
VPC_ID=$(aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)

SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" \
  --query "Subnets[0].SubnetId" --output text)

# Create security group (only outbound needed — bot uses long-polling, not webhook)
SG_ID=$(aws ec2 create-security-group \
  --group-name yt-rag-bot-sg \
  --description "YouTube RAG Bot" \
  --vpc-id $VPC_ID \
  --query "GroupId" --output text)

# Allow all outbound (default), no inbound needed
echo "VPC: $VPC_ID  Subnet: $SUBNET_ID  SG: $SG_ID"

# Create ECS service
aws ecs create-service \
  --cluster youtube-rag-bot-cluster \
  --service-name youtube-rag-bot \
  --task-definition youtube-rag-bot \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[$SUBNET_ID],
    securityGroups=[$SG_ID],
    assignPublicIp=ENABLED
  }"
```

Check it started:

```bash
aws ecs describe-services \
  --cluster youtube-rag-bot-cluster \
  --services youtube-rag-bot \
  --query "services[0].{Status:status,Running:runningCount,Desired:desiredCount}"
```

---

## Step 5 — Local Watcher Setup

The watcher runs **on your machine only** — it is never deployed to AWS.

### Environment

Create `.env.watcher` in the project root:

```env
AWS_REGION=us-east-1
S3_BUCKET=youtube-rag-bot-<ACCOUNT_ID>
SQS_REQUEST_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<ACCOUNT_ID>/yt-rag-transcript-requests
SQS_RESULT_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/<ACCOUNT_ID>/yt-rag-transcript-results
```

IAM permissions needed for your local credentials (keep these narrow):

```bash
aws iam put-user-policy \
  --user-name <your-iam-username> \
  --policy-name yt-rag-watcher-policy \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"sqs:ReceiveMessage\",\"sqs:DeleteMessage\",\"sqs:GetQueueAttributes\"],
        \"Resource\": \"arn:aws:sqs:us-east-1:${AWS_ACCOUNT_ID}:yt-rag-transcript-requests\"
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"sqs:SendMessage\"],
        \"Resource\": \"arn:aws:sqs:us-east-1:${AWS_ACCOUNT_ID}:yt-rag-transcript-results\"
      },
      {
        \"Effect\": \"Allow\",
        \"Action\": [\"s3:PutObject\",\"s3:HeadObject\"],
        \"Resource\": \"arn:aws:s3:::youtube-rag-bot-${AWS_ACCOUNT_ID}/transcripts/*\"
      }
    ]
  }"
```

### Run manually

```bash
cd youtube-rag-bot/
source venv/bin/activate

pip install -e .
pip install -r requirements/aws.txt   # boto3, youtube-transcript-api, etc.

python -m app.watcher.local_watcher
```

### Run automatically on startup

**macOS** — LaunchAgent:

```xml
<!-- Save to: ~/Library/LaunchAgents/com.ytragbot.watcher.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.ytragbot.watcher</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/yourname/youtube-rag-bot/venv/bin/python</string>
    <string>-m</string>
    <string>app.watcher.local_watcher</string>
  </array>
  <key>WorkingDirectory</key>
  <string>/Users/yourname/youtube-rag-bot</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>AWS_REGION</key><string>us-east-1</string>
    <key>S3_BUCKET</key><string>youtube-rag-bot-<ACCOUNT_ID></string>
    <key>SQS_REQUEST_QUEUE_URL</key>
    <string>https://sqs.us-east-1.amazonaws.com/<ACCOUNT_ID>/yt-rag-transcript-requests</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/ytragbot-watcher.log</string>
  <key>StandardErrorPath</key><string>/tmp/ytragbot-watcher-err.log</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.ytragbot.watcher.plist
launchctl start com.ytragbot.watcher

# Check it's running
launchctl list | grep ytragbot
tail -f /tmp/ytragbot-watcher.log
```

**Linux** — systemd:

```ini
# /etc/systemd/system/ytragbot-watcher.service
[Unit]
Description=YouTube RAG Bot Transcript Watcher
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/youtube-rag-bot
ExecStart=/home/youruser/youtube-rag-bot/venv/bin/python -m app.watcher.local_watcher
Restart=always
RestartSec=5
EnvironmentFile=/home/youruser/youtube-rag-bot/.env.watcher

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ytragbot-watcher
sudo systemctl start ytragbot-watcher
sudo systemctl status ytragbot-watcher
```

---

## Step 6 — Deploying Updates

Every code change is three commands. Save as `deploy.sh` in the project root:

```bash
#!/bin/bash
set -e

export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/youtube-rag-bot"

echo "🔨 Building image..."
docker build -f infra/Dockerfile.cloud -t youtube-rag-bot:latest .

echo "🔐 Logging into ECR..."
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

echo "📤 Pushing image..."
docker tag youtube-rag-bot:latest ${ECR_URI}:latest
docker push ${ECR_URI}:latest

echo "🚀 Forcing ECS redeployment..."
aws ecs update-service \
  --cluster youtube-rag-bot-cluster \
  --service youtube-rag-bot \
  --force-new-deployment \
  --query "service.{Status:status,Desired:desiredCount}" \
  --output table

echo "✅ Done. New task will be running in ~30 seconds."
```

```bash
chmod +x deploy.sh
./deploy.sh
```

---

## Viewing Logs

```bash
# Live logs (Ctrl+C to stop)
aws logs tail /ecs/youtube-rag-bot --follow

# Last 100 lines
aws logs tail /ecs/youtube-rag-bot --since 1h

# Or in the AWS Console:
# CloudWatch → Log groups → /ecs/youtube-rag-bot
```

---

## Troubleshooting

### Bot task keeps restarting (ECS shows STOPPED)

```bash
# See the stop reason
aws ecs describe-tasks \
  --cluster youtube-rag-bot-cluster \
  --tasks $(aws ecs list-tasks \
    --cluster youtube-rag-bot-cluster \
    --query "taskArns[0]" --output text) \
  --query "tasks[0].{StopCode:stopCode,StopReason:stoppedReason}"

# Then check logs for the Python traceback
aws logs tail /ecs/youtube-rag-bot --since 10m
```

### 403 errors on OpenSearch

Two separate access-control layers both must be configured:

| Layer | Where | What to check |
|---|---|---|
| IAM | Role policy | `aoss:APIAccessAll` on `arn:aws:aoss:::collection/*` |
| OpenSearch | Data access policy | Task role ARN listed as principal with `aoss:CreateIndex`, `aoss:ReadDocument`, `aoss:WriteDocument` |

Check the data access policy:
```
AWS Console → OpenSearch Serverless → Security → Data access policies
```
The principal must be:
```json
"arn:aws:iam::<ACCOUNT_ID>:role/yt-rag-bot-task-role"
```

### Transcript never arrives (bot waits 90s then times out)

```bash
# Check watcher is running locally
launchctl list | grep ytragbot          # macOS
systemctl status ytragbot-watcher       # Linux

# Check if the SQS message was actually sent
aws sqs get-queue-attributes \
  --queue-url https://sqs.us-east-1.amazonaws.com/<ACCOUNT_ID>/yt-rag-transcript-requests \
  --attribute-names ApproximateNumberOfMessages

# Check watcher logs
tail -50 /tmp/ytragbot-watcher.log      # macOS
journalctl -u ytragbot-watcher -n 50   # Linux
```

### Bedrock returns AccessDeniedException

```bash
# Verify model access is enabled (must be done in Console, not CLI)
# Console → Amazon Bedrock → Model access
# Both Nova Lite and Cohere Embed Multilingual must show "Access granted"

# Verify IAM policy on task role
aws iam get-role-policy \
  --role-name yt-rag-bot-task-role \
  --policy-name yt-rag-bot-policy
```

### Task starts but bot doesn't respond

```bash
# Verify TELEGRAM_TOKEN secret is correct
aws secretsmanager get-secret-value \
  --secret-id yt-rag-bot/telegram-token \
  --query SecretString --output text

# Check the token works
TOKEN="your_token"
curl "https://api.telegram.org/bot${TOKEN}/getMe"
```

### `ModuleNotFoundError: No module named 'app'` (inside the container)

Means `pip install -e .` / `pip install .` didn't run in the image, or `pyproject.toml` / `src/` weren't copied before that step. Check `infra/Dockerfile.cloud` copies both and runs the install before the `CMD`.

---

## Estimated Monthly Cost

| Service | Config | $/month |
|---|---|---|
| ECS Fargate | 0.5 vCPU / 1 GB, 24/7 | ~$14 |
| S3 | < 5 GB (transcripts + BM25) | < $0.50 |
| SQS | thousands of messages | < $0.01 |
| CloudWatch Logs | < 1 GB/month | ~$0.50 |
| Bedrock Nova Lite | ~1M tokens/month | ~$2 |
| Bedrock Cohere Embed | ~10M tokens/month | ~$1 |
| **OpenSearch Serverless** | **min. 2 OCU** | **~$350** |
| **Total** | | **~$368** |

> **OpenSearch Serverless minimum cost (~$350/month) dominates.** If this is too
> expensive for a personal or small-scale bot, consider these alternatives —
> only `src/app/storage/vectorstore_aws.py` needs to change, nothing else in the codebase:
>
> - **pgvector on RDS Aurora Serverless v2** — ~$30/month, proper vector search,
>   scales to zero when idle
> - **Chroma on EBS** — mount an EBS volume to the Fargate task for persistent
>   storage; free beyond the EBS cost (~$1–3/month for 20 GB). Simplest swap.
> - **Pinecone** (serverless tier) — free up to 2M vectors, then pay-per-use

---

## Environment Variables Reference

### Cloud bot (`app.bots.bot_aws` / ECS task definition)

| Variable | Required | Example | Description |
|---|---|---|---|
| `TELEGRAM_TOKEN` | ✅ | via Secrets Manager | Bot token from @BotFather |
| `PROVIDER` | ✅ | `bedrock` | Always `bedrock` in cloud |
| `AWS_REGION` | ✅ | `us-east-1` | AWS region |
| `S3_BUCKET` | ✅ | `youtube-rag-bot-123456789` | S3 bucket name |
| `SQS_REQUEST_QUEUE_URL` | ✅ | `https://sqs...` | Transcript request queue |
| `SQS_RESULT_QUEUE_URL` | | `https://sqs...` | Result notification queue |
| `OPENSEARCH_ENDPOINT` | ✅ | `abc.us-east-1.aoss.amazonaws.com` | Collection host (no https://) |
| `BEDROCK_LLM_MODEL` | | `amazon.nova-lite-v1:0` | LLM model ID |
| `BEDROCK_EMBED_MODEL` | | `cohere.embed-multilingual-v3` | Embedding model ID |
| `CHUNK_STRATEGY` | | `timestamp` | `timestamp`, `sentence`, or `semantic` |
| `CHUNK_TOKENS` | | `300` | Target tokens per chunk |
| `OVERLAP_SENTANCES` | | `1` | Overlap between chunks |
| `TRANSCRIPT_WAIT_TIMEOUT_S` | | `90` | Seconds to wait for local watcher |
| `USER_QUOTA_BYTES` | | `1073741824` | Per-user storage quota (default 1 GB) |

### Local watcher (`app.watcher.local_watcher`, `.env.watcher` on your machine)

| Variable | Required | Example |
|---|---|---|
| `AWS_REGION` | ✅ | `us-east-1` |
| `S3_BUCKET` | ✅ | `youtube-rag-bot-123456789` |
| `SQS_REQUEST_QUEUE_URL` | ✅ | `https://sqs.us-east-1.amazonaws.com/.../yt-rag-transcript-requests` |
| `SQS_RESULT_QUEUE_URL` | | `https://sqs.us-east-1.amazonaws.com/.../yt-rag-transcript-results` |