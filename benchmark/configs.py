"""
Experiment matrix for the article benchmark.

12 runs, grouped into three experiments:

  E1 (runs 1-4)  - chunking strategy           (retrieval fixed: hybrid+rerank, provider: bedrock)
  E2 (runs 5-8)  - retrieval scheme            (chunking fixed: winner of E1)
  E3 (runs 9-12) - provider x router ablation  (chunking + retrieval fixed: winners of E1/E2)

E2 and E3 inherit the winning settings from the previous experiment. Run the
experiments in order; after E1 finishes, put the winning chunking config name
into E1_WINNER below, and likewise for E2_WINNER.
"""

from dataclasses import dataclass, asdict, field
from typing import Literal


@dataclass
class RunConfig:
    run_id: str
    experiment: str          # "E1" | "E2" | "E3"
    label: str               # short human label used in the article tables

    # --- chunking -------------------------------------------------------
    chunk_strategy: Literal["timestamp", "sentence", "semantic"] = "timestamp"
    chunk_tokens: int = 300
    overlap_tokens: int = 30
    similarity_thr: float = 0.75

    # --- retrieval ------------------------------------------------------
    # "vector" | "bm25" | "hybrid" | "hybrid_rerank"
    retrieval: str = "hybrid_rerank"
    candidates_k: int = 20   # chunks pulled by each retriever before fusion
    final_k: int = 4         # chunks that actually reach the LLM

    # --- generation -----------------------------------------------------
    provider: Literal["bedrock", "ollama"] = "bedrock"
    llm_model: str = "amazon.nova-lite-v1:0"
    embed_model: str = "cohere.embed-multilingual-v3"
    rerank_model: str = "cohere.rerank-v3-5:0"

    # --- agent ----------------------------------------------------------
    router_enabled: bool = True
    history_turns: int = 5

    notes: str = ""

    def as_row(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fill these in after each experiment so the next one inherits the winner.
# ---------------------------------------------------------------------------
E1_WINNER = dict(chunk_strategy="timestamp", chunk_tokens=300, overlap_tokens=30)
E2_WINNER = dict(retrieval="hybrid_rerank", candidates_k=20, final_k=4)


# --- Experiment 1: chunking strategy ---------------------------------------
E1 = [
    RunConfig("E1-1", "E1", "timestamp 300/30",
              chunk_strategy="timestamp", chunk_tokens=300, overlap_tokens=30,
              notes="baseline, current production default"),
    RunConfig("E1-2", "E1", "timestamp 600/60",
              chunk_strategy="timestamp", chunk_tokens=600, overlap_tokens=60,
              notes="larger time window"),
    RunConfig("E1-3", "E1", "sentence 300/30",
              chunk_strategy="sentence", chunk_tokens=300, overlap_tokens=30,
              notes="sentence-aware split, ignores timecodes"),
    RunConfig("E1-4", "E1", "semantic 300 (thr 0.75)",
              chunk_strategy="semantic", chunk_tokens=300, overlap_tokens=0,
              similarity_thr=0.75,
              notes="embedding-similarity merging"),
]

# --- Experiment 2: retrieval scheme ----------------------------------------
E2 = [
    RunConfig("E2-1", "E2", "dense only",
              retrieval="vector", candidates_k=20, final_k=4, **E1_WINNER),
    RunConfig("E2-2", "E2", "BM25 only",
              retrieval="bm25", candidates_k=20, final_k=4, **E1_WINNER),
    RunConfig("E2-3", "E2", "hybrid RRF (k=60)",
              retrieval="hybrid", candidates_k=20, final_k=4, **E1_WINNER),
    RunConfig("E2-4", "E2", "hybrid RRF + rerank",
              retrieval="hybrid_rerank", candidates_k=20, final_k=4, **E1_WINNER),
]

# --- Experiment 3: provider x router ---------------------------------------
E3 = [
    RunConfig("E3-1", "E3", "Bedrock Nova Lite + router",
              provider="bedrock", llm_model="amazon.nova-lite-v1:0",
              embed_model="cohere.embed-multilingual-v3",
              router_enabled=True, **E1_WINNER, **E2_WINNER),
    RunConfig("E3-2", "E3", "Bedrock Nova Lite, no router",
              provider="bedrock", llm_model="amazon.nova-lite-v1:0",
              embed_model="cohere.embed-multilingual-v3",
              router_enabled=False, **E1_WINNER, **E2_WINNER),
    RunConfig("E3-3", "E3", "Ollama gemma3:4b + router",
              provider="ollama", llm_model="gemma3:4b",
              embed_model="nomic-embed-text", rerank_model="",
              router_enabled=True, **E1_WINNER, **E2_WINNER),
    RunConfig("E3-4", "E3", "Ollama gemma3:4b, no router",
              provider="ollama", llm_model="gemma3:4b",
              embed_model="nomic-embed-text", rerank_model="",
              router_enabled=False, **E1_WINNER, **E2_WINNER),
]

ALL_RUNS = E1 + E2 + E3


def get_runs(experiments: list[str] | None = None) -> list[RunConfig]:
    if not experiments:
        return ALL_RUNS
    return [r for r in ALL_RUNS if r.experiment in experiments]


# ---------------------------------------------------------------------------
# Pricing, USD. VERIFY THESE AGAINST THE AWS PRICING PAGE ON THE DAY YOU RUN
# THE EXPERIMENT and cite the access date in the article.
# ---------------------------------------------------------------------------
PRICES = {
    "amazon.nova-lite-v1:0":        {"in_per_1k": 0.00006,  "out_per_1k": 0.00024},
    "amazon.nova-micro-v1:0":       {"in_per_1k": 0.000035, "out_per_1k": 0.00014},
    "amazon.nova-pro-v1:0":         {"in_per_1k": 0.0008,   "out_per_1k": 0.0032},
    "cohere.embed-multilingual-v3": {"in_per_1k": 0.0001,   "out_per_1k": 0.0},
    "cohere.rerank-v3-5:0":         {"per_query": 0.002},
    # local models cost nothing per token; report wall-clock and RAM instead
    "gemma3:4b":                    {"in_per_1k": 0.0, "out_per_1k": 0.0},
    "nomic-embed-text":             {"in_per_1k": 0.0, "out_per_1k": 0.0},
}


def token_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    p = PRICES.get(model)
    if not p or "in_per_1k" not in p:
        return 0.0
    return (in_tokens / 1000) * p["in_per_1k"] + (out_tokens / 1000) * p["out_per_1k"]


def rerank_cost(model: str, n_queries: int) -> float:
    p = PRICES.get(model) or {}
    return p.get("per_query", 0.0) * n_queries
