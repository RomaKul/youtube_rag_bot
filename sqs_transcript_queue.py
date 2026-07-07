"""
sqs_transcript_queue.py — SQS request/result queues connecting the AWS bot
to the local transcript-fetching watcher.

Two queues (create both in AWS first, e.g. via the console or terraform):
    youtube-rag-transcript-requests   — bot → local watcher
    youtube-rag-transcript-results    — local watcher → bot (optional;
                                          polling S3 directly also works and
                                          is what bot_aws.py does by default,
                                          but the result queue lets the bot
                                          avoid tight polling loops if you
                                          prefer)

CHANGED: dropped `lang` from every message payload. The local watcher now
auto-detects the transcript language itself (youtube_transcript_api picks
the best available transcript), so a request is just "get me this video's
transcript" — no language to negotiate.
"""

from __future__ import annotations

import json
import logging
import os

import boto3

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
REQUEST_QUEUE_URL = os.getenv("SQS_REQUEST_QUEUE_URL")
RESULT_QUEUE_URL = os.getenv("SQS_RESULT_QUEUE_URL")  # optional

_sqs = boto3.client("sqs", region_name=AWS_REGION)


def enqueue_transcript_request(video_id: str) -> None:
    if not REQUEST_QUEUE_URL:
        raise RuntimeError("SQS_REQUEST_QUEUE_URL not set in environment")
    _sqs.send_message(
        QueueUrl=REQUEST_QUEUE_URL,
        MessageBody=json.dumps({"video_id": video_id}),
    )
    logger.info(f"[sqs] Enqueued transcript request: {video_id}")


def receive_transcript_requests(max_messages: int = 1, wait_seconds: int = 20) -> list[dict]:
    """Long-polls the request queue. Returns list of {video_id, _receipt_handle}."""
    if not REQUEST_QUEUE_URL:
        raise RuntimeError("SQS_REQUEST_QUEUE_URL not set in environment")
    resp = _sqs.receive_message(
        QueueUrl=REQUEST_QUEUE_URL,
        MaxNumberOfMessages=max_messages,
        WaitTimeSeconds=wait_seconds,
    )
    jobs = []
    for msg in resp.get("Messages", []):
        body = json.loads(msg["Body"])
        body["_receipt_handle"] = msg["ReceiptHandle"]
        jobs.append(body)
    return jobs


def ack_transcript_request(receipt_handle: str) -> None:
    _sqs.delete_message(QueueUrl=REQUEST_QUEUE_URL, ReceiptHandle=receipt_handle)


def notify_result(video_id: str, status: str, error: str = "") -> None:
    """Optional: push a small completion event so the bot doesn't have to poll S3."""
    if not RESULT_QUEUE_URL:
        return
    _sqs.send_message(
        QueueUrl=RESULT_QUEUE_URL,
        MessageBody=json.dumps({"video_id": video_id, "status": status, "error": error}),
    )