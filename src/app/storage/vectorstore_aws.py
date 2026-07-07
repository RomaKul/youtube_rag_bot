"""
vectorstore_aws.py — Amazon OpenSearch Serverless (AOSS, vector engine) adapter.

Drop-in replacement for the local `Chroma` usage in bot.py / rag_graph.py.
Returns a vectorstore that implements the same `similarity_search` /
`add_documents`-style surface the rest of the codebase already calls — so
rag_graph.py's `hybrid_search()`, `_format_docs()`, etc. need no changes.

--------------------------------------------------------------------------
Why this broke with:
  illegal_argument_exception: Field parameter 'engine' is not supported

AWS now offers two flavors of OpenSearch Serverless vector collection:
  - "Classic" collections — the original kind, where you choose the k-NN
    engine yourself (nmslib/faiss) via the index mapping's "method.engine".
  - "NextGen" collections — now the default when you create a new
    collection through the console's unified creation flow. NextGen manages
    the vector engine internally and REJECTS the "engine" field entirely —
    not just unsupported *values* of it, the key itself is disallowed.

`langchain-community`'s OpenSearchVectorSearch always writes "engine" into
the index mapping it auto-creates on first add_texts/add_documents call
(defaulting to "nmslib" if you don't override it). On a NextGen collection
that create-index call is rejected outright with exactly this error,
regardless of what engine value you pass — so simply switching to
engine="faiss" (the fix for Classic collections) does not help here.

The fix: pre-create the index ourselves, before langchain gets a chance to,
using a mapping that omits "engine". langchain's add_texts/add_documents
only auto-creates the index if it doesn't already exist, so once we've
created it ourselves it never sends its own (broken, for NextGen) mapping.
We try WITH "engine" first (works on Classic collections) and, only if the
server rejects that specific field, retry WITHOUT it (NextGen) — so this
works against either collection type without you needing to know which one
you have.

On top of that, AOSS has one more hard constraint most tutorials miss:
bulk indexing must NOT send a custom "_id" — AOSS rejects it. This is
controlled by langchain's `is_aoss` flag, which has to be passed on every
add_texts/add_documents call, not just once at construction. Since bot.py
calls `vs.add_documents(docs)` with no extra kwargs, we return a thin
subclass that injects that (and pre-creates the index) automatically, so no
other file in the codebase needs to change.
--------------------------------------------------------------------------

Prerequisites (one-time AWS setup, console or IaC):
  1. Create an OpenSearch Serverless **collection** with type "VECTORSEARCH".
  2. Attach a data-access policy granting your bot's IAM role
     aoss:APIAccessAll on the collection.
  3. Set OPENSEARCH_ENDPOINT in your environment (the collection's host,
     without the https:// prefix, e.g. "xxxx.us-east-1.aoss.amazonaws.com").
  4. Index creation now happens explicitly (see _ensure_index below) the
     first time you call add_documents/add_texts for a given index_name.
     Keep AOSS_SPACE_TYPE/AOSS_EF_CONSTRUCTION/AOSS_M constant for a given
     index once created — OpenSearch won't let you redefine a field's
     method in place.

Each video gets its own OpenSearch *index* (mirroring the old per-video
Chroma collection), named the same way bot.py already names collections.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, List, Optional
from dotenv import load_dotenv

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import RequestError
from requests_aws4auth import AWS4Auth
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

load_dotenv()
logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT")  # e.g. "xxxx.us-east-1.aoss.amazonaws.com"
OPENSEARCH_SERVICE = "aoss"  # OpenSearch Serverless; use "es" for a managed cluster instead

# AOSS vector engine constraints — keep these consistent for the life of an
# index. Changing them for an index that already exists will raise mapping
# errors, since OpenSearch won't let you redefine a field's method in place.
AOSS_ENGINE = "faiss"       # only used on Classic collections; NextGen ignores/rejects this
AOSS_SPACE_TYPE = "l2"      # match to your embedding model / similarity choice
AOSS_EF_CONSTRUCTION = 512
AOSS_M = 16
DEFAULT_VECTOR_FIELD = "vector_field"
DEFAULT_TEXT_FIELD = "text"


def _build_auth():
    creds = boto3.Session().get_credentials()
    return AWS4Auth(
        region=AWS_REGION,
        service=OPENSEARCH_SERVICE,
        refreshable_credentials=creds,
    )


# AOSS can be slow on cold index creation / first bulk write (compute spins
# up under the hood). opensearch-py's default 10s read timeout is too short
# for that — bump it and let the client retry transient timeouts.
CLIENT_TIMEOUT_SECONDS = int(os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "60"))
CLIENT_MAX_RETRIES = 3


def _get_client() -> OpenSearch:
    if not OPENSEARCH_ENDPOINT:
        raise RuntimeError("OPENSEARCH_ENDPOINT not set in environment")
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=_build_auth(),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=CLIENT_TIMEOUT_SECONDS,
        max_retries=CLIENT_MAX_RETRIES,
        retry_on_timeout=True,
    )


def _index_mapping(dimension: int, vector_field: str, text_field: str, include_engine: bool) -> dict:
    method: dict = {
        "name": "hnsw",
        "space_type": AOSS_SPACE_TYPE,
        "parameters": {"ef_construction": AOSS_EF_CONSTRUCTION, "m": AOSS_M},
    }
    if include_engine:
        method["engine"] = AOSS_ENGINE
    return {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                vector_field: {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": method,
                },
                text_field: {"type": "text"},
                "metadata": {"type": "object"},
            }
        },
    }


def _is_engine_unsupported_error(err: Exception) -> bool:
    msg = str(err)
    return "engine" in msg and "not supported" in msg


def _ensure_index(
    client: OpenSearch,
    index_name: str,
    dimension: int,
    vector_field: str = DEFAULT_VECTOR_FIELD,
    text_field: str = DEFAULT_TEXT_FIELD,
) -> None:
    """
    Create the index ourselves (if it doesn't already exist) with a mapping
    that works for either AOSS collection type, so langchain never gets a
    chance to auto-create it with a mapping that might be rejected.

    Tries WITH "engine" first (required on Classic collections), and falls
    back to omitting it if the server specifically rejects that field
    (NextGen collections).
    """
    if client.indices.exists(index=index_name):
        return

    try:
        client.indices.create(
            index=index_name,
            body=_index_mapping(dimension, vector_field, text_field, include_engine=True),
        )
        logger.info("[aoss] created index %r with explicit engine=%r (Classic collection)", index_name, AOSS_ENGINE)
    except RequestError as e:
        if not _is_engine_unsupported_error(e):
            raise
        logger.info(
            "[aoss] collection rejected the 'engine' mapping field (NextGen collection) — "
            "retrying index creation without it"
        )
        client.indices.create(
            index=index_name,
            body=_index_mapping(dimension, vector_field, text_field, include_engine=False),
        )
        logger.info("[aoss] created index %r without an explicit engine (NextGen collection)", index_name)


class AOSSVectorSearch(OpenSearchVectorSearch):
    """
    OpenSearchVectorSearch subclass that transparently enforces AOSS-safe
    behavior on every add_texts/add_documents call, so existing callers in
    bot.py / rag_graph.py don't need to know or care they're targeting
    Serverless, or which AOSS collection generation they're on:
      - pre-creates the index with an engine-optional mapping (see above)
      - sets is_aoss=True so bulk ingestion omits "_id" (AOSS requirement)
      - drops any caller-supplied ids, which AOSS's bulk API rejects
    """

    def _dimension_from(self, texts: Iterable[str]) -> int:
        texts = list(texts)
        probe = texts[0] if texts else "dimension probe"
        return len(self.embedding_function.embed_query(probe))

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        vector_field = kwargs.get("vector_field", DEFAULT_VECTOR_FIELD)
        text_field = kwargs.get("text_field", DEFAULT_TEXT_FIELD)

        _ensure_index(
            self.client,
            self.index_name,
            dimension=self._dimension_from(texts),
            vector_field=vector_field,
            text_field=text_field,
        )

        kwargs.setdefault("is_aoss", True)
        # AOSS rejects a custom "_id" on bulk index requests — drop it and
        # let OpenSearch assign one, rather than letting the request 400.
        if ids is not None:
            logger.debug("[aoss] dropping caller-supplied ids; not supported on AOSS bulk index")
            ids = None
        return super().add_texts(
            texts, metadatas=metadatas, ids=ids, bulk_size=bulk_size, **kwargs
        )

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        vector_field = kwargs.get("vector_field", DEFAULT_VECTOR_FIELD)
        text_field = kwargs.get("text_field", DEFAULT_TEXT_FIELD)

        _ensure_index(
            self.client,
            self.index_name,
            dimension=self._dimension_from([d.page_content for d in documents]),
            vector_field=vector_field,
            text_field=text_field,
        )

        kwargs.setdefault("is_aoss", True)
        return super().add_documents(documents, **kwargs)


def init_vectorstore(index_name: str, embeddings: Embeddings) -> AOSSVectorSearch:
    """
    index_name: same naming convention bot.py already uses for Chroma
                collections, e.g. f"video_{video_id}_{provider}_{strategy}"
                (OpenSearch index names must be lowercase — already true
                here since video_id/provider/strategy are lowercase).
    """
    if not OPENSEARCH_ENDPOINT:
        raise RuntimeError("OPENSEARCH_ENDPOINT not set in environment")

    auth = _build_auth()

    return AOSSVectorSearch(
        index_name=index_name,
        embedding_function=embeddings,
        opensearch_url=f"https://{OPENSEARCH_ENDPOINT}",
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        is_aoss=True,
        bulk_size=500,
        timeout=CLIENT_TIMEOUT_SECONDS,
        max_retries=CLIENT_MAX_RETRIES,
        retry_on_timeout=True,
    )


def index_exists(index_name: str) -> bool:
    """
    Cheap existence check so bot.py can decide whether to re-index, mirroring
    the old `vs.get(); if existing["ids"]:` pattern used with Chroma.
    """
    client = _get_client()
    try:
        return client.indices.exists(index=index_name)
    except Exception as e:
        logger.warning(f"[opensearch] index_exists check failed: {e}")
        return False


def fetch_summary_doc(vs: AOSSVectorSearch) -> str:
    """
    Equivalent of rag_graph.fetch_video_summary() but against OpenSearch.

    IMPORTANT: this must use approximate k-NN ("approximate_search"), NOT
    "script_scoring" — AOSS indices are always created with the approximate
    method mapping (see _index_mapping above), and script_scoring expects a
    plain knn_vector field with no "method" block at all. Querying an
    approximate-mapped field with script_scoring is what originally produced
    the "Field parameter 'engine' is not supported" error (a mismatched
    mapping, on top of the NextGen issue this file now also handles).
    """
    try:
        results = vs.similarity_search(
            query="",
            k=1,
            search_type="approximate_search",
            # efficient_filter (not pre_filter) is the AOSS-supported way to
            # narrow an approximate k-NN query by metadata.
            efficient_filter={"bool": {"filter": [{"term": {"metadata.type": "summary"}}]}},
        )
        if results:
            return results[0].page_content
    except Exception as e:
        logger.warning(f"[opensearch] summary fetch failed: {e}")
    return ""