"""Local LLM wrapper (vLLM, via its OpenAI-compatible API)."""
from __future__ import annotations

from . import config


def _client():
    from openai import OpenAI

    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        timeout=config.LLM_TIMEOUT,
    )


def is_available() -> tuple[bool, str]:
    """Check the vLLM server is up and serving the configured model."""
    try:
        client = _client()
        models = client.models.list()
        names = {m.id for m in models.data}
        wanted = config.LLM_MODEL
        if wanted not in names:
            return False, (
                f"vLLM is running but model '{wanted}' is not loaded. "
                f"Loaded: {', '.join(sorted(names)) or '(none)'}"
            )
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, (
            f"Cannot reach vLLM at {config.LLM_BASE_URL} ({e}). "
            "Is the vllm container running?"
        )


def generate(system: str, prompt: str, stream: bool = False,
             max_tokens: int | None = None, seed: bool = False):
    """Generate a completion. Returns a string, or a generator if stream=True.

    max_tokens caps the reply — used by short auxiliary calls like HyDE
    drafting, where a rambling answer wastes time and dilutes the embedding it
    is meant to produce. Defaults to LLM_MAX_TOKENS when not given.

    seed=True pins sampling to config.LLM_SEED, making the call reproducible.
    Used by the calls that FEED RETRIEVAL (the HyDE draft, the follow-up
    rewrite) so the same question searches for the same thing twice — see
    LLM_SEED. The answer itself is left unseeded."""
    client = _client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    kwargs = dict(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=config.LLM_TEMPERATURE,
        top_p=config.LLM_TOP_P,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
        extra_body={
            "top_k": config.LLM_TOP_K,
            "chat_template_kwargs": {"enable_thinking": config.LLM_ENABLE_THINKING},
        },
    )
    if seed and config.LLM_SEED >= 0:
        kwargs["seed"] = config.LLM_SEED

    if stream:
        def _gen():
            for chunk in client.chat.completions.create(stream=True, **kwargs):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _gen()

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content
