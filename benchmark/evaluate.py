#!/usr/bin/env python
"""
Score the raw answers with RAGAS and produce the article tables.

    python evaluate.py results/raw_20260918-120000.jsonl

Writes next to the input file:
    scored_<stamp>.csv      every question with its four RAGAS metrics
    table_E1_<stamp>.csv    chunking comparison        -> Table 3 of the article
    table_E2_<stamp>.csv    retrieval comparison       -> Table 4
    table_E3_<stamp>.csv    provider x router ablation -> Table 5

Metrics (all in [0,1], higher is better):
    faithfulness      - is the answer supported by the retrieved contexts
    answer_relevancy  - does the answer address the question
    context_precision - how much of the retrieved context is useful
    context_recall    - did retrieval find everything the ground truth needs
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_raw(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(rows: list[dict], max_workers: int = 4, timeout: int = 180,
          max_retries: int = 10, sample: int | None = None) -> pd.DataFrame:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (faithfulness, answer_relevancy,
                               context_precision, context_recall)
    from ragas.run_config import RunConfig
    import adapters

    usable = [r for r in rows if not r["error"] and r["answer"]]
    skipped = len(rows) - len(usable)
    if skipped:
        print(f"skipping {skipped} failed answers")
    if sample:
        usable = usable[:sample]
        print(f"--sample set: scoring only the first {len(usable)} rows")

    ds = Dataset.from_dict({
        "question":     [r["question"] for r in usable],
        "answer":       [r["answer"] for r in usable],
        "contexts":     [r["contexts"] or [""] for r in usable],
        "ground_truth": [r["ground_truth"] for r in usable],
    })

    # RAGAS fans out many concurrent LLM calls per row per metric (faithfulness
    # alone does a statement-extraction call plus a verification call; context
    # precision does roughly one call per retrieved chunk). At the default
    # max_workers=16 this routinely throttles a single Bedrock judge model,
    # and every throttle becomes a retry-with-backoff, which is what a wall of
    # TimeoutError and a ballooning ETA usually mean — not that the judge is
    # actually 4 hours of real work. Lower concurrency, longer per-job
    # patience, more retries: slower ceiling, far fewer failures in practice.
    run_config = RunConfig(timeout=timeout, max_retries=max_retries,
                           max_wait=90, max_workers=max_workers)

    judge_llm, judge_emb = adapters.get_judge()
    eval_kwargs = dict(
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm, embeddings=judge_emb,
        run_config=run_config,
    )
    try:
        # raise_exceptions=False keeps partial results (NaN for the failed
        # row/metric) instead of losing the whole batch to one bad row.
        # Dropped automatically if your installed ragas version doesn't
        # accept this kwarg (the flag's name/presence has moved around
        # across ragas releases).
        result = evaluate(ds, raise_exceptions=False, **eval_kwargs)
    except TypeError:
        result = evaluate(ds, **eval_kwargs)

    scored = result.to_pandas()
    meta = pd.DataFrame([{k: r[k] for k in
                          ("run_id", "experiment", "label", "repeat", "video_id",
                           "video_lang", "question_id", "question_type", "route",
                           "total_ms", "retrieval_ms", "llm_in_tokens",
                           "llm_out_tokens", "cost_usd")} for r in usable])
    return pd.concat([meta.reset_index(drop=True),
                      scored[METRIC_COLS].reset_index(drop=True)], axis=1)


def article_table(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    sel = df[df["experiment"] == experiment]
    if sel.empty:
        return sel
    g = sel.groupby(["run_id", "label"], as_index=False).agg(
        faithfulness=("faithfulness", "mean"),
        answer_relevancy=("answer_relevancy", "mean"),
        context_precision=("context_precision", "mean"),
        context_recall=("context_recall", "mean"),
        latency_s=("total_ms", lambda s: s.mean() / 1000),
        latency_sd=("total_ms", lambda s: s.std(ddof=1) / 1000 if len(s) > 1 else 0.0),
        tokens_in=("llm_in_tokens", "mean"),
        cost_per_1000_q=("cost_usd", lambda s: s.mean() * 1000),
        from_context_share=("route", lambda s: (s == "from_context").mean()),
        n=("question_id", "count"),
    )
    return g.round(4).sort_values("run_id")


def per_question_type(df: pd.DataFrame) -> pd.DataFrame:
    """Useful for the discussion: multi-hop and follow-up questions behave differently."""
    return (df.groupby(["experiment", "question_type"], as_index=False)[METRIC_COLS]
              .mean().round(4))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("--max-workers", type=int, default=4,
                    help="concurrent RAGAS jobs; lower this first if you see "
                         "TimeoutError / throttling (default: 4, RAGAS default is 16)")
    ap.add_argument("--timeout", type=int, default=180,
                    help="seconds RAGAS waits per job before retrying (default: 180)")
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--sample", type=int, default=None,
                    help="score only the first N rows — use this to smoke-test "
                         "before committing to a full run")
    args = ap.parse_args()

    stamp = args.raw.stem.replace("raw_", "")
    out = args.raw.parent

    rows = load_raw(args.raw)
    df = score(rows, max_workers=args.max_workers, timeout=args.timeout,
              max_retries=args.max_retries, sample=args.sample)
    suffix = f"{stamp}_sample{args.sample}" if args.sample else stamp
    df.to_csv(out / f"scored_{suffix}.csv", index=False, encoding="utf-8")

    for exp in ("E1", "E2", "E3"):
        t = article_table(df, exp)
        if not t.empty:
            t.to_csv(out / f"table_{exp}_{suffix}.csv", index=False, encoding="utf-8")
            print(f"\n=== {exp} ===")
            print(t.to_string(index=False))

    pq = per_question_type(df)
    pq.to_csv(out / f"by_question_type_{suffix}.csv", index=False, encoding="utf-8")
    print("\n=== by question type ===")
    print(pq.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())