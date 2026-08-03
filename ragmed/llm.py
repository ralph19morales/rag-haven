"""Local LLM wrapper (Ollama)."""
from __future__ import annotations

from . import config


def _client():
    import ollama

    return ollama.Client(host=config.OLLAMA_HOST)


def is_available() -> tuple[bool, str]:
    """Check the Ollama server is up and the configured model is pulled."""
    try:
        client = _client()
        models = client.list().get("models", [])
        names = {m.get("model", m.get("name", "")) for m in models}
        # Match with or without an explicit :latest tag.
        wanted = config.LLM_MODEL
        ok = any(n == wanted or n.split(":")[0] == wanted.split(":")[0]
                 for n in names)
        if not ok:
            return False, (
                f"Ollama is running but model '{wanted}' is not pulled. "
                f"Run:  ollama pull {wanted}"
            )
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, (
            f"Cannot reach Ollama at {config.OLLAMA_HOST} ({e}). "
            "Is 'ollama serve' running?"
        )


def generate(system: str, prompt: str, stream: bool = False,
             max_tokens: int | None = None):
    """Generate a completion. Returns a string, or a generator if stream=True.

    max_tokens caps the reply (Ollama's num_predict) — used by short auxiliary
    calls like HyDE drafting, where a rambling answer wastes time and dilutes
    the embedding it is meant to produce."""
    client = _client()
    options = {
        "temperature": config.LLM_TEMPERATURE,
        "num_ctx": config.LLM_NUM_CTX,
    }
    if max_tokens:
        options["num_predict"] = max_tokens
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    # Keep the model resident between questions. Ollama evicts after 5 minutes
    # idle by default, and reloading a 9 GB model off disk costs far more than
    # the question itself — a pure latency win with no effect on the answer.
    # Sent per request so this works against any Ollama server without asking
    # the user to reconfigure the service.
    ka = config.OLLAMA_KEEP_ALIVE

    if stream:
        def _gen():
            for part in client.chat(
                model=config.LLM_MODEL, messages=messages,
                options=options, stream=True, keep_alive=ka,
            ):
                yield part["message"]["content"]
        return _gen()

    resp = client.chat(
        model=config.LLM_MODEL, messages=messages, options=options,
        keep_alive=ka,
    )
    return resp["message"]["content"]
