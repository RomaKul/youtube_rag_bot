"""
s3_transcript_store.py — S3-backed storage for transcripts and BM25 indexes.

Used by both the AWS-side bot (read path) and the local watcher (write path).

Layout in the bucket:
    transcripts/{video_id}.json     — {"text": ..., "segments": [...], "lang_code": ...}
    bm25/{collection_name}.pkl      — pickled BM25Index.docs

Both sides only need boto3 + this module — no shared DB connection required.

CHANGED: dropped the `lang` parameter everywhere. Transcript language is no
longer chosen by the caller — it's auto-detected by youtube_transcript_api
(see local_watcher.py's fetch_transcript_locally), so there's only ever one
cached transcript per video_id. The detected language code is still kept,
but purely as informational metadata inside the stored JSON payload — it's
no longer part of the S3 key.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import asdict
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3_BUCKET = os.getenv("S3_BUCKET", "youtube-rag-bot")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

_s3 = boto3.client("s3", region_name=AWS_REGION)


def _transcript_key(video_id: str) -> str:
    return f"transcripts/{video_id}.json"


def _bm25_key(collection_name: str) -> str:
    return f"bm25/{collection_name}.pkl"


# ── Transcripts ──────────────────────────────────────────────────────────────

def transcript_exists(video_id: str) -> bool:
    try:
        _s3.head_object(Bucket=S3_BUCKET, Key=_transcript_key(video_id))
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def put_transcript(video_id: str, text: str, segments: list, lang_code: str = "") -> None:
    """segments: list[Segment] dataclass instances from chunking.py.

    lang_code is whatever youtube_transcript_api auto-detected/returned —
    stored for information/display purposes only, not used as a lookup key.
    """
    payload = {
        "text": text,
        "segments": [asdict(s) for s in segments] if segments else [],
        "lang_code": lang_code,
    }
    _s3.put_object(
        Bucket=S3_BUCKET,
        Key=_transcript_key(video_id),
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info(f"[s3] Uploaded transcript {video_id} ({len(text)} chars, lang={lang_code or 'unknown'})")


def get_transcript(video_id: str) -> Optional[tuple[str, list, str]]:
    """Returns (text, segments_as_dicts, lang_code) or None if not yet present."""
    try:
        obj = _s3.get_object(Bucket=S3_BUCKET, Key=_transcript_key(video_id))
        payload = json.loads(obj["Body"].read())
        return payload["text"], payload.get("segments", []), payload.get("lang_code", "")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


# ── BM25 index ───────────────────────────────────────────────────────────────

def put_bm25_docs(collection_name: str, docs: list) -> None:
    """docs: list[langchain_core.documents.Document]"""
    body = pickle.dumps(docs)
    _s3.put_object(Bucket=S3_BUCKET, Key=_bm25_key(collection_name), Body=body)
    logger.info(f"[s3] Uploaded BM25 doc set for {collection_name} ({len(docs)} docs)")


def get_bm25_docs(collection_name: str) -> Optional[list]:
    try:
        obj = _s3.get_object(Bucket=S3_BUCKET, Key=_bm25_key(collection_name))
        return pickle.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise