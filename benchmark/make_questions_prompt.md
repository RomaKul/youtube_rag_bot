# Prompt for building the ground-truth set

Do this once per video. Fetch the full transcript first (`youtube-transcript-api`,
same source your bot uses — the questions must be answerable from exactly the text
the system will see, including its ASR errors). Paste the transcript and this prompt
into a strong model, then read every generated pair yourself and fix it. Unverified
generated ground truth is the one thing a reviewer will attack hardest, so the
manual pass is not optional.

---

You are helping build an evaluation set for a retrieval-augmented question
answering system over YouTube transcripts.

Below is the full auto-generated transcript of one video, with timecodes.

Produce exactly 10 question–answer pairs as a JSON array. Use these types and
counts:

- 3 × `factual` — answerable from a single short passage; prefer questions whose
  answer is a specific number, name, date or term.
- 2 × `summary` — require condensing a substantial part of the video.
- 2 × `multi_hop` — require combining two passages that are at least ten minutes
  apart in the video. State both timecodes in `evidence`.
- 2 × `follow_up` — short, elliptical continuations ("say more about that",
  "why?") that only make sense after the preceding question. The first must
  follow the last `multi_hop` question, the second must follow the first
  `follow_up`.
- 1 × `off_topic` — plausible-sounding but not discussed in the video at all.

For every pair return:

```json
{
  "id": "<VIDEO_ID>-q<N>",
  "type": "<one of the types above>",
  "follow_up_to": "<id of the preceding question, or null>",
  "question": "<the question, in the language of the video>",
  "ground_truth": "<a complete answer, 1-3 sentences, using only the transcript>",
  "evidence": "<the transcript fragment(s) and timecode(s) that support it>"
}
```

Rules:
- Never use knowledge from outside the transcript.
- Do not paraphrase the transcript so heavily that the wording of the question
  gives away which chunk contains the answer — that biases lexical retrieval.
- For the `off_topic` item, `ground_truth` must state that the video does not
  cover the subject.
- Return only the JSON array, nothing else.

TRANSCRIPT:
<paste here>

---

## Your manual pass — check each pair for

1. Is the ground truth actually in the transcript? (Open the timecode.)
2. Is the factual question really single-hop, and the multi-hop really two-hop?
3. Does the follow-up genuinely make no sense standalone? If it does make sense
   standalone, the router test is worthless.
4. Is the wording natural — the way a Telegram user would type it, not the way a
   dataset builder would?

Then paste the verified arrays into `questions.json` under the right video.
