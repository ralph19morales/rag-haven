"""Local query-metrics logging, read by dashboard.py.

Every question answered through rag.answer() appends one JSON line to
data/metrics.jsonl — timings, retrieval stats, and outcome. Local-only, same
as everything else this project writes (data/ is git-ignored, nothing here is
transmitted). Logging is best-effort: a failure to write a line must never
break an answer, and a corrupt line (e.g. a write interrupted mid-flush) must
never break reading the rest back.
"""
from __future__ import annotations

import json
import time

from . import config


def log_query(**fields) -> None:
    if not config.METRICS_ENABLED:
        return
    record = {"ts": time.time(), **fields}
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with config.METRICS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - metrics must never break an answer
        pass


def load(limit: int | None = None) -> list[dict]:
    """Read logged query records, oldest first. `limit` keeps only the most
    recent N. Tolerant of a corrupt/partial trailing line."""
    if not config.METRICS_PATH.exists():
        return []
    with config.METRICS_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    if limit:
        lines = lines[-limit:]
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
