"""
local_watcher.py — Runs on YOUR machine (laptop / home server), NOT in AWS.

YouTube blocks transcript-fetch requests from most cloud-provider IP ranges.
This script is the workaround: it long-polls an SQS queue for transcript
requests pushed by the AWS-hosted bot, fetches the transcript locally (your
home/office IP is not blocked), and uploads the result to S3. The bot then
picks it up from S3 and continues indexing — no other local dependency
needed once a transcript is cached.

Run it as a long-lived background process:
    python local_watcher.py
    # or, for resilience: a systemd service / cron @reboot / pm2 / nohup

Requires the same AWS credentials as the bot (an IAM user/role with
sqs:ReceiveMessage, sqs:DeleteMessage, s3:PutObject on the relevant
queue/bucket) — configure via `aws configure` or env vars, same as boto3
normally expects. This script does NOT need youtube-transcript-api on AWS;
only here.
"""

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv

load_dotenv()

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

from chunking import segments_from_fetched, Segment
from s3_transcript_store import put_transcript, transcript_exists
from sqs_transcript_queue import (
    receive_transcript_requests, ack_transcript_request, notify_result,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("local_watcher")


def fetch_transcript_locally(video_id: str, lang: str) -> tuple[str, list[Segment], str]:
    """Same logic as bot.py's get_transcript — duplicated here deliberately so
    this script has zero dependency on the AWS-side bot module."""
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=[lang])
        segments = segments_from_fetched(fetched)
        full_text = " ".join(s.text for s in segments)
        return full_text, segments, "exact_lang"
    except NoTranscriptFound:
        for t in api.list(video_id):
            fetched = t.fetch()
            segments = segments_from_fetched(fetched)
            full_text = " ".join(s.text for s in segments)
            return full_text, segments, f"fallback_{t.language_code}"
        raise


def handle_job(video_id: str, lang: str) -> None:
    if transcript_exists(video_id, lang):
        logger.info(f"[watcher] {video_id}/{lang} already in S3 — skipping fetch")
        notify_result(video_id, lang, status="ok")
        return

    logger.info(f"[watcher] Fetching {video_id}/{lang} locally…")
    try:
        text, segments, note = fetch_transcript_locally(video_id, lang)
    except TranscriptsDisabled:
        logger.warning(f"[watcher] Transcripts disabled for {video_id}")
        notify_result(video_id, lang, status="error", error="transcripts_disabled")
        return
    except NoTranscriptFound:
        logger.warning(f"[watcher] No transcript found for {video_id}/{lang}")
        notify_result(video_id, lang, status="error", error="no_transcript")
        return
    except Exception as e:
        logger.error(f"[watcher] Unexpected error fetching {video_id}: {e}")
        notify_result(video_id, lang, status="error", error=str(e)[:200])
        return

    put_transcript(video_id, lang, text, segments)
    logger.info(f"[watcher] Done: {video_id}/{lang} ({note}, {len(text)} chars)")
    notify_result(video_id, lang, status="ok")


def main():
    logger.info("🟢 Local transcript watcher started. Polling SQS…")
    while True:
        try:
            jobs = receive_transcript_requests(max_messages=5, wait_seconds=20)
            for job in jobs:
                handle_job(job["video_id"], job["lang"])
                ack_transcript_request(job["_receipt_handle"])
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
            break
        except Exception as e:
            logger.error(f"[watcher] Loop error: {e}")
            time.sleep(5)  # back off before retrying so a persistent SQS/network
                            # error doesn't spin in a hot loop


if __name__ == "__main__":
    main()
