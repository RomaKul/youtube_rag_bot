#!/usr/bin/env python
"""
Run the experiment matrix and write raw results.

    python run_benchmark.py --questions questions.json --out results/
    python run_benchmark.py --experiments E1 --repeats 3
    python run_benchmark.py --dry-run          # check wiring without spending money

Outputs (per invocation):
    results/raw_<timestamp>.jsonl   one line per answered question
    results/runs_<timestamp>.csv    one line per (run x video), latency + cost
    results/index_<timestamp>.csv   indexing stats per (run x video)

Then:  python evaluate.py results/raw_<timestamp>.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from configs import get_runs, token_cost, rerank_cost, RunConfig
import adapters


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    videos = data["videos"]
    for v in videos:
        for i, q in enumerate(v["questions"]):
            q.setdefault("id", f"{v['video_id']}-q{i+1}")
            q.setdefault("type", "factual")
            q.setdefault("follow_up_to", None)
    return videos


def run_one_video(cfg: RunConfig, video: dict, repeat: int, dry: bool) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    url = video["url"]

    if dry:
        index_stats = {"n_chunks": 0, "index_seconds": 0.0}
    else:
        index_stats = adapters.index_video(url, cfg)

    history: list[dict] = []
    prev_context: list[str] = []

    for q in video["questions"]:
        # a question marked as a follow-up keeps the dialogue state; anything
        # else starts a fresh session, so the router ablation stays honest
        if not q.get("follow_up_to"):
            history, prev_context = [], []

        t0 = time.perf_counter()
        try:
            if dry:
                res = adapters.AnswerResult(answer="(dry run)", contexts=[], route="disabled")
            else:
                res = adapters.answer_question(
                    question=q["question"], video_url=url, cfg=cfg,
                    history=history[-cfg.history_turns * 2:], prev_context=prev_context,
                )
        except Exception as exc:                      # a failed run must not kill the matrix
            traceback.print_exc()
            res = adapters.AnswerResult(answer="", contexts=[], error=repr(exc))
        wall_ms = (time.perf_counter() - t0) * 1000

        cost = (token_cost(cfg.llm_model, res.llm_in_tokens, res.llm_out_tokens)
                + token_cost(cfg.embed_model, res.embed_tokens, 0)
                + rerank_cost(cfg.rerank_model, res.rerank_queries))

        rows.append({
            "run_id": cfg.run_id, "experiment": cfg.experiment, "label": cfg.label,
            "repeat": repeat,
            "video_id": video["video_id"], "video_lang": video.get("language"),
            "video_minutes": video.get("minutes"),
            "question_id": q["id"], "question_type": q["type"],
            "question": q["question"], "ground_truth": q.get("ground_truth", ""),
            "answer": res.answer, "contexts": res.contexts,
            "route": res.route,
            "llm_in_tokens": res.llm_in_tokens, "llm_out_tokens": res.llm_out_tokens,
            "embed_tokens": res.embed_tokens,
            "retrieval_ms": round(res.retrieval_ms, 1),
            "generation_ms": round(res.generation_ms, 1),
            "total_ms": round(res.total_ms or wall_ms, 1),
            "cost_usd": round(cost, 8),
            "error": res.error,
            **{f"cfg_{k}": v for k, v in cfg.as_row().items()
               if k not in ("run_id", "experiment", "label", "notes")},
        })

        history += [{"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": res.answer}]
        if res.contexts:
            prev_context = res.contexts

    if not dry:
        adapters.drop_index(url, cfg)

    index_row = {"run_id": cfg.run_id, "video_id": video["video_id"], **index_stats}
    return rows, index_row


def summarise(rows: list[dict]) -> list[dict]:
    """One line per (run_id, video_id): the numbers that go into the article tables."""
    out = []
    keys = sorted({(r["run_id"], r["video_id"]) for r in rows})
    for run_id, video_id in keys:
        sel = [r for r in rows if r["run_id"] == run_id and r["video_id"] == video_id]
        ok = [r for r in sel if not r["error"]]
        lat = [r["total_ms"] for r in ok] or [0]
        n_ctx = sum(1 for r in ok if r["route"] == "from_context")
        out.append({
            "run_id": run_id, "label": sel[0]["label"], "experiment": sel[0]["experiment"],
            "video_id": video_id, "n_questions": len(sel), "n_errors": len(sel) - len(ok),
            "latency_mean_s": round(statistics.mean(lat) / 1000, 3),
            "latency_p95_s": round(sorted(lat)[max(0, int(len(lat) * 0.95) - 1)] / 1000, 3),
            "retrieval_mean_ms": round(statistics.mean([r["retrieval_ms"] for r in ok] or [0]), 1),
            "tokens_in_mean": round(statistics.mean([r["llm_in_tokens"] for r in ok] or [0]), 1),
            "tokens_out_mean": round(statistics.mean([r["llm_out_tokens"] for r in ok] or [0]), 1),
            "cost_per_1000_q_usd": round(sum(r["cost_usd"] for r in ok) / max(len(ok), 1) * 1000, 4),
            "from_context_share": round(n_ctx / max(len(ok), 1), 3),
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=Path("questions.json"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--experiments", nargs="*", default=None, help="E1 E2 E3")
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run ids, e.g. E1-1 E1-3")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the whole matrix N times; report mean and stdev in the article")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    videos = load_questions(args.questions)

    runs = get_runs(args.experiments)
    if args.runs:
        runs = [r for r in runs if r.run_id in args.runs]
    if not runs:
        print("no runs selected", file=sys.stderr)
        return 1

    total = len(runs) * len(videos) * args.repeats
    print(f"{len(runs)} configs x {len(videos)} videos x {args.repeats} repeats = {total} indexings")

    all_rows, index_rows = [], []
    done = 0
    for repeat in range(1, args.repeats + 1):
        for cfg in runs:
            adapters.apply_config(cfg)
            for video in videos:
                done += 1
                print(f"[{done}/{total}] {cfg.run_id} {cfg.label} | {video['video_id']}", flush=True)
                rows, idx = run_one_video(cfg, video, repeat, args.dry_run)
                all_rows += rows
                index_rows.append({"repeat": repeat, **idx})

    raw_path = args.out / f"raw_{stamp}.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_csv(args.out / f"runs_{stamp}.csv", summarise(all_rows))
    write_csv(args.out / f"index_{stamp}.csv", index_rows)

    n_err = sum(1 for r in all_rows if r["error"])
    print(f"\nwrote {raw_path} ({len(all_rows)} answers, {n_err} errors)")
    print(f"next: python evaluate.py {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
