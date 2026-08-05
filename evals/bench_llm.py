"""Benchmark generation on a REAL Haven prompt. Run identically before/after a
vLLM config change; compare TTFT and decode tok/s."""
import json, pathlib, sys, time, statistics
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ragmed import config, rag, retriever, embeddings, rerank

LABEL = sys.argv[1] if len(sys.argv) > 1 else "run"
OUT = str(pathlib.Path(__file__).parent / f"bench_{LABEL}.json")

QUESTIONS = [
    "What are the grounds for revoking a physicians certificate of registration?",
    "Is there a senior citizen discount on hospital bills?",
    "The hospital won't release my father's body until we pay the bill. Can they do that?",
]

embeddings.warmup(); rerank.warmup()

# Build the real prompts once, so every config benchmarks the SAME token stream
# and differences are the server's, not retrieval's.
prompts = []
for q in QUESTIONS:
    chunks = retriever.retrieve(q)
    prompts.append((q, rag.build_prompt(q, chunks)))
print(f"built {len(prompts)} prompts; sizes: "
      f"{[len(p)//4 for _, p in prompts]} approx tokens")

from openai import OpenAI
client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                timeout=config.LLM_TIMEOUT)


def one(prompt, seed):
    """Time one generation. Token count comes from the server's usage block,
    NOT from counting SSE chunks: with speculative decoding several accepted
    tokens can arrive in a single chunk, so chunk-counting silently
    under-reports throughput and made spec decode look like a regression."""
    t = time.perf_counter(); first = None; chunks = 0; toks = None
    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "system", "content": rag.SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        temperature=config.LLM_TEMPERATURE, top_p=config.LLM_TOP_P,
        max_tokens=config.LLM_MAX_TOKENS, seed=seed, stream=True,
        stream_options={"include_usage": True},
        extra_body={"top_k": config.LLM_TOP_K,
                    "chat_template_kwargs": {"enable_thinking": False}},
    )
    for ch in stream:
        if getattr(ch, "usage", None):
            toks = ch.usage.completion_tokens
        if ch.choices and ch.choices[0].delta.content:
            if first is None:
                first = (time.perf_counter() - t) * 1000
            chunks += 1
    total = (time.perf_counter() - t) * 1000
    n = toks if toks else chunks
    return dict(ttft_ms=first, total_ms=total, tokens=n, chunks=chunks,
                decode_tps=n / max(total - first, 1) * 1000)


one(prompts[0][1], 1)   # warm the server
rows = []
for q, p in prompts:
    for seed in (11, 22):
        r = one(p, seed); r["q"] = q[:40]
        rows.append(r)
        print(f"  {r['ttft_ms']:7.0f} ms TTFT  {r['tokens']:4d} tok "
              f"({r['chunks']:4d} chunks)  {r['decode_tps']:5.1f} tok/s  "
              f"{r['total_ms']:7.0f} ms total  {r['q']}")

summary = dict(
    label=LABEL,
    ttft_ms=statistics.median(r["ttft_ms"] for r in rows),
    decode_tps=statistics.median(r["decode_tps"] for r in rows),
    total_ms=statistics.median(r["total_ms"] for r in rows),
    rows=rows,
)
print(f"\n[{LABEL}] median TTFT {summary['ttft_ms']:.0f} ms | "
      f"decode {summary['decode_tps']:.1f} tok/s | total {summary['total_ms']:.0f} ms")
json.dump(summary, open(OUT, "w"), indent=1)
