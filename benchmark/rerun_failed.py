"""Re-score only the rows where faithfulness/context_precision came back NaN,
then merge the recovered values back into the full scored CSV.

    python rerun_failed.py scored_20260924.csv raw_20260924.jsonl
    python rerun_failed.py scored_20260924.csv raw_20260924.jsonl --run-id E1-4
"""
import argparse
import json
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("scored", type=Path)
ap.add_argument("raw", type=Path)
ap.add_argument("--run-id", default=None,
                help="only retry NaN rows belonging to this run_id (e.g. E1-4). "
                     "Omit to retry NaN rows across every config in the file.")
args = ap.parse_args()

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

scored = pd.read_csv(args.scored)
if args.run_id and args.run_id not in set(scored["run_id"]):
    print(f"⚠️  run_id {args.run_id!r} not found in {args.scored.name} — "
          f"available: {sorted(scored['run_id'].unique())}")

subset_df = scored[scored["run_id"] == args.run_id] if args.run_id else scored
failed_mask = subset_df[METRIC_COLS].isna().any(axis=1)
failed_ids = set(subset_df.loc[failed_mask, "question_id"])

scope = f"run_id={args.run_id}" if args.run_id else "all runs"
print(f"{len(failed_ids)} / {len(subset_df)} rows have at least one NaN metric ({scope})")

raw_rows = [json.loads(l) for l in args.raw.read_text(encoding="utf-8").splitlines() if l.strip()]
subset = [r for r in raw_rows
         if r["question_id"] in failed_ids and (not args.run_id or r["run_id"] == args.run_id)]

tag = f"{args.raw.stem}_{args.run_id}" if args.run_id else args.raw.stem
subset_path = args.raw.with_name(f"retry_{tag}.jsonl")
subset_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in subset), encoding="utf-8")
print(f"wrote {len(subset)} rows to {subset_path}")
print(f"\nnow run:\n  python evaluate.py {subset_path} --max-workers 2 --timeout 300")
print(f"\nthen merge the resulting scored_retry_{tag}.csv back with:")
print(f"  python merge_retry.py {args.scored} scored_retry_{tag}.csv")