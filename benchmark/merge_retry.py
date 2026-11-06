"""Merge recovered (retry) metric values back into the original scored CSV,
filling only the cells that were NaN — never overwriting a value that
already scored successfully.

    python merge_retry.py scored_20260924.csv scored_retry_20260924.csv
"""
import sys
import pandas as pd
from pathlib import Path

orig_path, retry_path = Path(sys.argv[1]), Path(sys.argv[2])
METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

orig = pd.read_csv(orig_path).set_index("question_id")
retry = pd.read_csv(retry_path).set_index("question_id")

filled = 0
for qid in retry.index:
    if qid not in orig.index:
        continue
    for col in METRIC_COLS:
        if pd.isna(orig.at[qid, col]) and not pd.isna(retry.at[qid, col]):
            orig.at[qid, col] = retry.at[qid, col]
            filled += 1

still_nan = orig[METRIC_COLS].isna().sum().sum()
out_path = orig_path.with_name(orig_path.stem + "_merged.csv")
orig.reset_index().to_csv(out_path, index=False, encoding="utf-8")
print(f"filled {filled} cells; {still_nan} still NaN after merge")
print(f"wrote {out_path}")
