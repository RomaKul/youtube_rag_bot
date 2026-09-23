"""
scripts/manual_test.py — exercise the real bot logic against hardcoded
YouTube links, with zero Telegram involved: no bot token, no polling, no
network call to Telegram's API.

How this works: bot_core.receive_url and bot_core.receive_question are
plain async functions of the shape (update, context) -> state. They only
ever touch update.message.text, update.message.reply_text(/edit_text),
and context.user_data / context.bot_data. This file fakes those three
things and calls the *actual* handler functions unmodified, so a passing
run here means the real bot would have done the same thing.

Usage:
    BOT_BACKEND=local python scripts/manual_test.py     # Chroma, PROVIDER=ollama|bedrock
    BOT_BACKEND=aws   python scripts/manual_test.py     # OpenSearch + S3/SQS

Edit VIDEO_URLS and QUESTIONS below before running.
"""

import asyncio
import os

from app.bots.bot_core import start, receive_url, receive_question, ASK_QUESTION
from app.bots.local_backend import LocalBackend
from app.bots.aws_backend import AwsBackend

# ── Edit these ───────────────────────────────────────────────────────────
VIDEO_URLS = [
    "https://www.youtube.com/watch?v=e1tkFsFOBHA",
    # "https://youtu.be/YYYYYYYYYYY",
]

QUESTIONS = [
    "What is this video about?",
    "Summarize the main points in 3 bullets.",
]
# ─────────────────────────────────────────────────────────────────────────


class FakeMessage:
    """Stands in for telegram.Message. Just prints instead of hitting the API."""

    def __init__(self, text: str):
        self.text = text

    async def reply_text(self, text: str, **kwargs):
        print(f"[bot] {text}")
        return FakeMessage(text)  # so status_msg.edit_text(...) still works below

    async def edit_text(self, text: str, **kwargs):
        print(f"[bot:edit] {text}")


class FakeUpdate:
    """Stands in for telegram.Update. Only .message is ever touched by our handlers."""

    def __init__(self, text: str):
        self.message = FakeMessage(text)


class FakeContext:
    """Stands in for telegram.ext.ContextTypes.DEFAULT_TYPE."""

    def __init__(self, bot_data: dict):
        self.user_data: dict = {}
        self.bot_data = bot_data


def _make_backend():
    name = os.getenv("BOT_BACKEND", "local").lower()
    if name == "aws":
        print("Backend: AWS (OpenSearch + S3/SQS)")
        return AwsBackend()
    print("Backend: local (Chroma)")
    return LocalBackend()


async def run_one(video_url: str, bot_data: dict):
    context = FakeContext(bot_data)

    print(f"\n{'=' * 70}\n{video_url}\n{'=' * 70}")
    await start(FakeUpdate("/start"), context)

    state = await receive_url(FakeUpdate(video_url), context)
    if state != ASK_QUESTION:
        print("⚠️  Indexing did not reach ASK_QUESTION — skipping questions for this video.")
        return

    for q in QUESTIONS:
        print(f"\n> {q}")
        await receive_question(FakeUpdate(q), context)


async def main():
    backend = _make_backend()
    bot_data = {"backend": backend, "llm": backend.build_llm()}

    if not VIDEO_URLS or "XXXXXXXXXXX" in VIDEO_URLS[0]:
        raise SystemExit("Edit VIDEO_URLS at the top of this file with real links first.")

    for url in VIDEO_URLS:
        await run_one(url, bot_data)


if __name__ == "__main__":
    asyncio.run(main())