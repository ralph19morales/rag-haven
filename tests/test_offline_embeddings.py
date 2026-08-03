"""Tests for the offline fallback in ragmed.embeddings.

Why this exists: sentence-transformers contacts huggingface.co on load even
when the model is already cached, so a DNS hiccup hard-crashes a live query in
a system whose whole point is running locally.

Note what this asserts. A first version of the fallback set HF_HUB_OFFLINE=1
and retried; a test that only checked "the env var was set" PASSED while the
fallback did not work at all — huggingface_hub reads that variable into a
module constant at import time, so a late change is ignored. These tests assert
the mechanism that actually matters: the retry is handed a local DIRECTORY.

Run:  .venv/Scripts/python.exe tests/test_offline_embeddings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sentence_transformers as st  # noqa: E402

from ragmed import config, embeddings  # noqa: E402

NETWORK_DOWN = OSError("[WinError 10061] target machine actively refused it")


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def run() -> int:
    r = []
    real = st.SentenceTransformer
    try:
        # --- the hub is unreachable; the cached snapshot must be used --------
        seen: list[str] = []

        class FailsOnce:
            def __new__(cls, name_or_path, *a, **k):
                seen.append(str(name_or_path))
                if len(seen) == 1:
                    raise NETWORK_DOWN
                return real(name_or_path, *a, **k)

        st.SentenceTransformer = FailsOnce
        embeddings._model.cache_clear()
        model = embeddings._model()

        r.append(check("a model is still returned when the hub is down",
                       model is not None))
        r.append(check("it retried exactly once", len(seen) == 2,
                       f"(attempts: {len(seen)})"))
        r.append(check("first attempt used the model id",
                       seen[0] == config.EMBED_MODEL))
        # The point of the whole fix: a filesystem path bypasses hub lookups.
        r.append(check("retry was handed a local directory that exists",
                       bool(seen[1]) and Path(seen[1]).is_dir(),
                       f"({seen[1][:58]}...)" if len(seen) > 1 else ""))
        r.append(check("retry path is NOT the bare model id",
                       seen[1] != config.EMBED_MODEL))
        r.append(check("the returned model actually embeds",
                       len(embeddings.embed_query("test")) == 768))

        # --- the happy path must not pay for any of this --------------------
        calls: list[str] = []

        class Works:
            def __new__(cls, name_or_path, *a, **k):
                calls.append(str(name_or_path))
                return real(name_or_path, *a, **k)

        st.SentenceTransformer = Works
        embeddings._model.cache_clear()
        embeddings._model()
        r.append(check("when the hub is reachable there is no second load",
                       len(calls) == 1, f"(attempts: {len(calls)})"))
    finally:
        st.SentenceTransformer = real
        embeddings._model.cache_clear()

    failed = r.count(False)
    print(f"\n{len(r) - failed}/{len(r)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
