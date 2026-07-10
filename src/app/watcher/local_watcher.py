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

CHANGED: no more `lang` parameter anywhere. fetch_transcript_locally() now
auto-detects the best available transcript exactly like bot.py's
get_transcript() does — it lists everything available for the video and
prefers a manually created transcript over an auto-generated one, in
whichever language happens to exist, instead of requiring the caller to
request a specific language up front.
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

from app.rag.chunking import segments_from_fetched, Segment
from app.storage.s3_transcript_store import put_transcript, transcript_exists
from app.storage.sqs_transcript_queue import (
    receive_transcript_requests, ack_transcript_request, notify_result,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("local_watcher")


def fetch_transcript_locally(video_id: str) -> tuple[str, list[Segment], str, str]:
    """Auto-detects and fetches the best available transcript for a video —
    same logic as bot.py's get_transcript(), duplicated here deliberately so
    this script has zero dependency on the AWS-side bot module.

    Lists every transcript available for the video and prefers a manually
    created one (i.e. provided by the uploader, not auto-generated) in
    whichever language exists; falls back to the first auto-generated one
    otherwise. No language needs to be requested or configured.

    Returns (full_text, segments, kind, lang_code, status).
    """
    api = YouTubeTranscriptApi()

    transcript_list = list(api.list(video_id))
    if not transcript_list:
        raise NoTranscriptFound(video_id, [], {})

    manual = [t for t in transcript_list if not t.is_generated]
    chosen = manual[0] if manual else transcript_list[0]

    fetched = chosen.fetch()
    segments = segments_from_fetched(fetched)
    full_text = " ".join(s.text for s in segments)

    kind = "manual" if not chosen.is_generated else "auto-generated"
    status = (
        f"✅ Transcript loaded ({len(segments)} segments) — "
        f"language: {chosen.language} [{chosen.language_code}], {kind}"
    )    
    return full_text, segments, kind, chosen.language_code, status


def handle_job(video_id: str) -> None:
    if transcript_exists(video_id):
        logger.info(f"[watcher] {video_id} already in S3 — skipping fetch")
        notify_result(video_id, status="ok")
        return

    logger.info(f"[watcher] Fetching {video_id} locally (auto-detecting language)…")
    try:
        text, segments, kind, lang_code = fetch_transcript_locally(video_id)
    except TranscriptsDisabled:
        logger.warning(f"[watcher] Transcripts disabled for {video_id}")
        notify_result(video_id, status="error", error="transcripts_disabled")
        return
    except NoTranscriptFound:
        logger.warning(f"[watcher] No transcript found for {video_id}")
        notify_result(video_id, status="error", error="no_transcript")
        return
    except Exception as e:
        logger.error(f"[watcher] Unexpected error fetching {video_id}: {e}")
        notify_result(video_id, status="error", error=str(e)[:200])
        return

    put_transcript(video_id, text, segments, lang_code=lang_code)
    logger.info(f"[watcher] Done: {video_id} ({kind}, lang={lang_code}, {len(text)} chars)")
    notify_result(video_id, status="ok")


def main():
    logger.info("🟢 Local transcript watcher started. Polling SQS…")
    while True:
        try:
            jobs = receive_transcript_requests(max_messages=5, wait_seconds=20)
            for job in jobs:
                handle_job(job["video_id"])
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