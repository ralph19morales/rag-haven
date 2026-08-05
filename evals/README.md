# evals/

Measurements of the **running system**, as opposed to `tests/`, which checks the
code. The distinction matters and decides where a new check belongs:

| | `tests/` | `evals/` |
|---|---|---|
| Needs the corpus indexed | no | **yes** |
| Needs the embedding / reranker models | no | **yes** |
| Needs vLLM running | no | **yes** (except `retrieval_authority.py`, unless HyDE fires) |
| Runtime | milliseconds | seconds to minutes, uses the GPU |
| Deterministic | yes | no — sampled generation |
| Run when | every change | before/after anything touching retrieval order, prompts or the LLM config |

Because they are slow, non-deterministic and compete with real users for the
GPU, these are **not** part of the test suite and nothing runs them
automatically.

## The three

**`retrieval_authority.py`** — for seven questions whose controlling instrument
is known independently, does that instrument reach the top_k? Two cases are
deliberately case-law-governed so that a change which merely drags statutes
upward is seen to break them. Also reports context breadth and whether a statute
and its IRR arrive together.

**`answer_quality.py`** — inspects the ANSWER, not the clock. Leaked prompt
labels, back-to-back repeated fragments (the speculative-decoding signature),
and whether anything was cited at all. Both defects it checks for shipped past
every latency and liveness measurement in the project.

**`bench_llm.py`** — TTFT and decode throughput on real prompts. Takes token
counts from the server's `usage` block; **never** count SSE chunks, which
under-reports badly whenever speculative decoding is on.

## Using them

```bash
python evals/retrieval_authority.py before
# ...make the change...
python evals/retrieval_authority.py after
```

Sweep a knob instead of guessing a value — this is how the current defaults for
`RERANK_PROTECT_TOP` and `MAX_CHUNKS_PER_FAMILY` were chosen:

```bash
for n in 0 2 3 4; do
  MAX_CHUNKS_PER_FAMILY=$n python evals/retrieval_authority.py "cap=$n" | tail -1
done
```

Reference numbers for this corpus sit in each script's docstring and beside the
constants they justify in `ragmed/config.py`. If you change the corpus or the
embedding model, those numbers are void — re-measure before trusting any of
them.
