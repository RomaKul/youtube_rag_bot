"""
test_vectorstore_aws.py — run this locally to sanity-check vectorstore_aws.py
against your real AOSS collection before wiring it into bot.py.

It does NOT spin up OpenSearch locally — AOSS is a managed cloud service,
so "local" here just means "run from your machine, talking to AWS over the
network." This script uses a fake, deterministic embedding function so you
don't need an OpenAI/Bedrock key just to test connectivity + indexing.

Usage:
    export OPENSEARCH_ENDPOINT="xxxx.us-east-1.aoss.amazonaws.com"
    export AWS_REGION="us-east-1"          # optional, defaults to us-east-1
    # make sure your AWS credentials are configured (aws configure / SSO / env vars)

    pip install boto3 opensearch-py requests-aws4auth langchain-community langchain-core

    python test_vectorstore_aws.py

It will:
  1. Verify env vars + AWS credentials are present.
  2. Create/reuse a throwaway test index and add a few documents.
  3. Run a similarity_search and print results.
  4. Run fetch_summary_doc against a doc tagged metadata.type == "summary".
  5. Clean up by deleting the test index (so re-runs start fresh).

Exits non-zero on failure so you can wire it into CI if useful.

Place this file in the same directory as vectorstore_aws.py.
"""

from __future__ import annotations

import logging
import sys
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import vectorstore_aws as vs_mod

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_vectorstore_aws")


class FakeEmbeddings(Embeddings):
    """
    Deterministic, dependency-free stand-in for a real embedding model.
    Hashes each token into a fixed-size vector so that similar text produces
    similar vectors — good enough to prove indexing/search works end-to-end
    without needing an OpenAI/Bedrock API key for a connectivity test.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            idx = hash(token) % self.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def check_prerequisites() -> None:
    if not vs_mod.OPENSEARCH_ENDPOINT:
        logger.error("OPENSEARCH_ENDPOINT is not set. export it and re-run.")
        sys.exit(1)

    import boto3
    creds = boto3.Session().get_credentials()
    if creds is None:
        logger.error(
            "No AWS credentials found. Run `aws configure`, `aws sso login`, "
            "or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN."
        )
        sys.exit(1)

    logger.info("OPENSEARCH_ENDPOINT=%s", vs_mod.OPENSEARCH_ENDPOINT)
    logger.info("AWS_REGION=%s", vs_mod.AWS_REGION)
    logger.info("AWS credentials found for access key: %s...", creds.access_key[:6])


def main() -> None:
    check_prerequisites()

    index_name = f"test_vectorstore_aws_{uuid.uuid4().hex[:8]}"
    embeddings = FakeEmbeddings()

    logger.info("Using throwaway index: %s", index_name)

    # 1. index_exists should be False before we've added anything
    exists_before = vs_mod.index_exists(index_name)
    logger.info("index_exists before add_documents: %s (expected False)", exists_before)
    assert exists_before is False, "expected index to not exist yet"

    # 2. init_vectorstore + add_documents (this is what creates the index,
    #    with the AOSS-safe faiss/is_aoss defaults from AOSSVectorSearch)
    vstore = vs_mod.init_vectorstore(index_name, embeddings)

    docs = [
        Document(
            page_content="This video is a summary about cats and dogs living together.",
            metadata={"type": "summary", "video_id": "abc123"},
        ),
        Document(
            page_content="Transcript chunk: the speaker discusses machine learning basics.",
            metadata={"type": "chunk", "video_id": "abc123"},
        ),
        Document(
            page_content="Transcript chunk: another section about neural networks.",
            metadata={"type": "chunk", "video_id": "abc123"},
        ),
    ]

    logger.info("Adding %d documents...", len(docs))
    ids = vstore.add_documents(docs)
    logger.info("add_documents returned %d ids", len(ids))
    assert len(ids) == len(docs), "expected one id per document"

    # 3. index_exists should be True now
    exists_after = vs_mod.index_exists(index_name)
    logger.info("index_exists after add_documents: %s (expected True)", exists_after)
    assert exists_after is True, "expected index to exist after add_documents"

    # 4. basic similarity_search
    logger.info("Running similarity_search for 'machine learning'...")
    results = vstore.similarity_search("machine learning", k=2)
    for r in results:
        logger.info("  hit: %.60s... metadata=%s", r.page_content, r.metadata)
    assert len(results) > 0, "expected at least one similarity_search result"

    # 5. fetch_summary_doc (exercises the approximate_search + efficient_filter path)
    logger.info("Running fetch_summary_doc...")
    summary = vs_mod.fetch_summary_doc(vstore)
    logger.info("fetch_summary_doc returned: %r", summary)
    assert summary, "expected fetch_summary_doc to return the summary doc's content"
    assert "cats and dogs" in summary, "expected the summary-tagged doc to be returned"

    logger.info("ALL CHECKS PASSED")

    # 6. cleanup so repeated runs don't leave junk indices behind
    cleanup(index_name)


def cleanup(index_name: str) -> None:
    from opensearchpy import OpenSearch, RequestsHttpConnection

    client = OpenSearch(
        hosts=[{"host": vs_mod.OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=vs_mod._build_auth(),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=vs_mod.CLIENT_TIMEOUT_SECONDS,
        max_retries=vs_mod.CLIENT_MAX_RETRIES,
        retry_on_timeout=True,
    )
    try:
        client.indices.delete(index=index_name)
        logger.info("Deleted test index: %s", index_name)
    except Exception as e:
        logger.warning("Could not delete test index %s: %s", index_name, e)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("CHECK FAILED: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during test run: %s", e)
        sys.exit(1)