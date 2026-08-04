"""Ops dashboard for the Philippine medical-law RAG system.

Run with:  streamlit run dashboard.py --server.port 8502

WHO THIS IS FOR: whoever operates this box, not the end user — the opposite
audience from app.py. It surfaces the things that fail silently otherwise: is
the LLM server actually reachable, are the CPU-pinned models actually on CPU
(the exact bug fixed in HAVEN_VLLM_MIGRATION.md §9 — this dashboard exists
partly to catch a regression of it before a user does), is the index populated,
and — once queries have been logged — how the system is actually performing.

Colours and type come from .streamlit/config.toml, same as app.py, so the two
apps read as one system. This app additionally leans on the theme's
`chartCategoricalColors` (a deliberately chosen, fixed-order palette) for every
chart — no ad hoc colors are introduced here.

REFRESH: manual only, via the button — deliberately, after two automatic
approaches didn't hold up. `st.fragment(run_every=...)` looked right but its
auto-rerun depends on a browser-side timer scheduled over Streamlit's
websocket protocol that this environment had no way to verify end-to-end. A
follow-up attempt with a plain HTML `<meta http-equiv="refresh">` tag worked
mechanically but reloaded the visible page every few seconds, which read as
broken rather than "live." A query is logged only once its answer has
*finished* generating (up to ~a minute — see README.md's Performance
section), so don't expect a question still in progress in Haven to appear
here before then; that is normal, not a bug.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from ragmed import config, embeddings, llm, metrics, ocr, rerank, vectorstore

st.set_page_config(
    page_title="Haven — Ops Dashboard",
    page_icon=":material/monitoring:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Probes. Each is cached briefly so a page of widgets doesn't mean a page of
# redundant subprocess calls / model loads on every fragment tick — but the
# TTLs stay short because this is a health dashboard, not a report: a stale
# "OK" is worse than a slightly-too-frequent check.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10, show_spinner=False)
def llm_health() -> tuple[bool, str, float]:
    t0 = time.perf_counter()
    ok, msg = llm.is_available()
    return ok, msg, (time.perf_counter() - t0) * 1000


@st.cache_data(ttl=5, show_spinner=False)
def gpu_status() -> dict | None:
    """None means nvidia-smi isn't available on this box — a real state, not
    an error: not every deployment has an NVIDIA GPU to query."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    name, total, used, free, util, temp = (p.strip() for p in out.stdout.split(","))
    return {
        "name": name, "total_mib": float(total), "used_mib": float(used),
        "free_mib": float(free), "util_pct": float(util), "temp_c": float(temp),
    }


@st.cache_resource(show_spinner="Loading embedding model…")
def embeddings_device() -> str:
    return str(embeddings._model().device)


@st.cache_resource(show_spinner="Loading reranker model…")
def reranker_device() -> str | None:
    """None = disabled by config. Raises if enabled but the model won't load
    — is_available() catches that for cli.py status, but here we want the
    actual device the *next* successful load lands on, so we surface load
    failure as its own state instead of masking it."""
    if not config.RERANK_ENABLED:
        return None
    return str(rerank._model().model.device)


@st.cache_data(ttl=30, show_spinner=False)
def index_stats() -> dict:
    collection = vectorstore.get_collection()
    n = vectorstore.count(collection)
    families: dict[str, int] = {}
    if n:
        _, _, metas = vectorstore.all_documents(collection)
        for m in metas:
            src = m.get("source", "")
            parts = src.split("/")
            family = parts[1] if src.startswith("_fetched/") and len(parts) > 1 else parts[0]
            families[family or "other"] = families.get(family or "other", 0) + 1
    return {"chunks": n, "families": families}


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} PB"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Haven — Ops Dashboard")
st.caption("Health and query metrics for the RAG engine. Not the user-facing app — see `streamlit run app.py` for that.")

top_l, top_r = st.columns([5, 1])
with top_l:
    st.caption(f":material/schedule: Last updated {time.strftime('%H:%M:%S')} · "
               "click Refresh to re-check. A query only appears below once its "
               "answer has *finished* generating — that can take up to a "
               "minute (see Performance in README.md), so a question still in "
               "progress in Haven will not show up here yet.")
with top_r:
    if st.button(":material/refresh: Refresh now", width="stretch"):
        st.cache_data.clear()
        st.rerun()

st.divider()


# ---------------------------------------------------------------------------
# Live content — plain top-level code, re-executed on every page load
# (currently only the manual Refresh button; no auto-refresh — a tried
# meta-refresh reload was visually disruptive and was removed).
# ---------------------------------------------------------------------------

def _live_body() -> None:
    llm_ok, llm_msg, llm_latency = llm_health()
    gpu = gpu_status()
    idx = index_stats()
    emb_dev = embeddings_device()
    try:
        rer_dev = reranker_device()
        rer_error = None
    except Exception as e:  # noqa: BLE001 - want to display, not crash the page
        rer_dev, rer_error = None, str(e)

    ocr_ok = ocr.is_available()

    # A pinning regression is the one failure mode this dashboard specifically
    # exists to catch (see module docstring) — surface it above the fold, not
    # buried in a detail panel.
    pinning_ok = emb_dev == "cpu" and (rer_dev in (None, "cpu"))
    overall_ok = llm_ok and idx["chunks"] > 0 and pinning_ok

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Overall", "Healthy" if overall_ok else "Attention needed")
        (st.success if overall_ok else st.error)(" ", icon=":material/check_circle:" if overall_ok else ":material/error:")
    with s2:
        st.metric("LLM server", "Up" if llm_ok else "Down", f"{llm_latency:.0f} ms" if llm_ok else None)
        if not llm_ok:
            st.caption(llm_msg)
    with s3:
        if gpu:
            st.metric("GPU VRAM", f"{gpu['used_mib']/1024:.1f} / {gpu['total_mib']/1024:.1f} GB")
        else:
            st.metric("GPU VRAM", "n/a")
    with s4:
        st.metric("Chunks indexed", f"{idx['chunks']:,}")
    with s5:
        label = "OK" if pinning_ok else "REGRESSION"
        st.metric("CPU pinning", label)
        if not pinning_ok:
            st.caption("Embeddings or reranker loaded onto GPU — see HAVEN_VLLM_MIGRATION.md §9")

    st.divider()

    # -----------------------------------------------------------------------
    # Compute health / Index & corpus — side by side, detail behind the summary
    # -----------------------------------------------------------------------

    col_compute, col_index = st.columns(2)

    with col_compute:
        st.subheader("Compute")

        if gpu:
            used_frac = gpu["used_mib"] / gpu["total_mib"] if gpu["total_mib"] else 0
            st.progress(min(used_frac, 1.0),
                        text=f"{gpu['name']} — {gpu['used_mib']/1024:.1f} GB used / "
                             f"{gpu['total_mib']/1024:.1f} GB total, {gpu['util_pct']:.0f}% util, "
                             f"{gpu['temp_c']:.0f}°C")
            st.caption(
                "Running near-full is expected, not a warning: vLLM is configured "
                "with `--gpu-memory-utilization 0.93` by design (see "
                "HAVEN_VLLM_MIGRATION.md §5). What matters is the row below."
            )
        else:
            st.info("nvidia-smi not found on this machine — GPU metrics unavailable.")

        device_rows = [
            {"Component": "Embeddings (bge-base)", "Device": emb_dev,
             "Expected": "cpu", "OK": emb_dev == "cpu"},
            {"Component": "Reranker (cross-encoder)",
             "Device": "disabled" if rer_dev is None and rer_error is None else (rer_dev or f"error: {rer_error}"),
             "Expected": "cpu (or disabled)",
             "OK": rer_dev in (None, "cpu") if rer_error is None else False},
        ]
        st.dataframe(pd.DataFrame(device_rows), hide_index=True, width="stretch")

        st.caption(
            ("OCR (Tesseract): available" if ocr_ok else
             f"OCR (Tesseract): {ocr.unavailable_reason()}" if config.OCR_ENABLED else
             "OCR: disabled (OCR_ENABLED=false)")
        )

    with col_index:
        st.subheader("Index & corpus")

        families = idx["families"]
        if families:
            fam_df = (pd.DataFrame({"family": list(families.keys()), "chunks": list(families.values())})
                       .sort_values("chunks", ascending=False).set_index("family"))
            st.bar_chart(fam_df, y="chunks", horizontal=True, height=220)
        else:
            st.info("Index is empty — run `python cli.py ingest`.")

        disk_rows = [
            {"Path": "data/chroma/ (vector store)", "Size": _human(_dir_size(config.CHROMA_DIR))},
            {"Path": "data/bm25.pkl (lexical index)", "Size": _human(_dir_size(config.BM25_PATH))},
            {"Path": "corpus/ (source documents)", "Size": _human(_dir_size(config.CORPUS_DIR))},
        ]
        st.dataframe(pd.DataFrame(disk_rows), hide_index=True, width="stretch")

        total, used, free = shutil.disk_usage(config.PROJECT_ROOT)
        st.caption(f"Host disk: {_human(used)} used / {_human(total)} total ({free/total:.0%} free)")

    st.divider()

    # -----------------------------------------------------------------------
    # Query metrics — from ragmed/metrics.py's log. Read fresh every fragment
    # tick (not cached) so a query logged seconds ago shows up on the next
    # auto-refresh. Empty until questions have actually been asked through
    # rag.answer() (CLI ask/chat or app.py).
    # -----------------------------------------------------------------------

    st.subheader("Query metrics")

    records = metrics.load()
    if not records:
        st.info(
            "No queries logged yet. Ask something via `python cli.py ask \"…\"`, "
            "`cli.py chat`, or the Haven app — this section fills in from there."
        )
        return

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    df["errored"] = df["error"].notna()

    span_hours = (df["time"].max() - df["time"].min()).total_seconds() / 3600
    bucket = "h" if span_hours <= 48 else "D"
    df["bucket"] = df["time"].dt.floor(bucket)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total queries", f"{len(df):,}")
    m2.metric("Median latency", f"{df['total_ms'].median()/1000:.1f} s")
    m3.metric("HyDE fire rate", f"{df['hyde_fired'].mean():.0%}")
    err_rate = df["errored"].mean()
    m4.metric("Error rate", f"{err_rate:.0%}",
              delta=None if err_rate == 0 else "check recent errors below",
              delta_color="inverse")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Queries per " + ("hour" if bucket == "h" else "day"))
        vol = df.groupby("bucket").size().rename("queries").to_frame()
        st.bar_chart(vol, y="queries", height=240)
    with c2:
        st.caption("Latency by stage (ms, median per bucket)")
        lat = df.groupby("bucket")[["retrieval_ms", "generation_ms"]].median()
        st.line_chart(lat, height=240)

    st.caption("Recent queries")
    show_cols = ["time", "question", "num_chunks", "hyde_fired", "total_ms", "error"]
    recent = df.sort_values("time", ascending=False)[[c for c in show_cols if c in df.columns]].head(25)
    if "question" not in df.columns or df["question"].isna().all():
        recent = recent.drop(columns=[c for c in ["question"] if c in recent.columns])
        st.caption("(question text not logged — METRICS_LOG_QUESTIONS=false)")
    st.dataframe(recent, hide_index=True, width="stretch")


_live_body()
