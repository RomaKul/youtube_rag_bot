# Benchmark harness for the article

Measures 12 configurations of the YouTube RAG bot on a fixed question set and
produces the tables for the Results section.

```
benchmark/
├── configs.py                 12 run configurations + pricing
├── adapters.py                ← the only file you edit
├── run_benchmark.py           runs the matrix, writes raw answers + latency/cost
├── evaluate.py                RAGAS scoring, writes the article tables
├── questions.example.json     test-set template
└── make_questions_prompt.md   how to build the ground truth quickly
```

## Install

```bash
pip install ragas datasets pandas langchain-aws
```

Drop the folder at the repository root, next to `src/`, so `import app...` works
inside `adapters.py`.

## Order of work

1. **Build the question set.** Follow `make_questions_prompt.md`. Copy
   `questions.example.json` → `questions.json` and fill in three video URLs and
   30 verified pairs. Budget ~1.5 hours.

2. **Wire `adapters.py`.** Four functions: `apply_config`, `index_video`,
   `answer_question`, `drop_index`. Everything else is done.

3. **Check the wiring without spending anything:**

   ```bash
   python run_benchmark.py --dry-run
   ```

4. **Run one config end to end** before committing to the full matrix:

   ```bash
   python run_benchmark.py --runs E1-1
   ```

5. **Experiment 1 — chunking:**

   ```bash
   python run_benchmark.py --experiments E1 --repeats 3
   python evaluate.py results/raw_<stamp>.jsonl
   ```

   Put the winner into `E1_WINNER` in `configs.py`.

6. **Experiment 2 — retrieval:** same, `--experiments E2`. Put the winner into
   `E2_WINNER`.

7. **Experiment 3 — provider and router:** `--experiments E3`. Start `ollama
   serve` first, and run E3-3/E3-4 on the same machine as E3-1/E3-2 or the
   latency comparison is meaningless. Record the machine's CPU, RAM and whether
   a GPU was used — that goes in Materials and Methods.

## `--repeats`

LLM answers vary between identical runs. `--repeats 3` gives you a standard
deviation per cell, which `evaluate.py` reports as `latency_sd`. Reviewers of
this kind of paper increasingly ask for it; three repeats is the cheapest way to
have an answer.

## Cost

One full pass is 12 configs × 30 questions = 360 answers, plus RAGAS scoring
(roughly four judge calls per answer). On Nova Lite plus a Nova Pro judge this
is a few dollars at most. With `--repeats 3`, still under ten. Verify the
prices in `configs.py` against the AWS pricing page on the day you run it, and
cite the access date in the article.
