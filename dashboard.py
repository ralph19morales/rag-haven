"""Ops dashboard for the Philippine medical-law RAG system.

Run with:  streamlit run dashboard.py --server.port 8502

WHO THIS IS FOR: whoever operates this box, not the end user — the opposite
audience from app.py. It surfaces the things that fail silently otherwise, grouped by the kind of
failure each one catches:

  LIVENESS     is the LLM server reachable, is the index populated, what is
               the GPU and the host doing, is anything queueing in the engine.
  REGRESSION   are the CPU-pinned models still on CPU (the exact bug fixed in
               HAVEN_VLLM_MIGRATION.md §9), and does the LIVE engine config
               still satisfy this deployment's invariants.
  DRIFT        is the corpus newer than the index — nothing re-indexes
               automatically, so an un-ingested document is invisible.
  OUTPUT       does the system still produce a correct-looking ANSWER, checked
               on demand by the canary at the foot of the page.

That last category is the newest and the reason the others are not enough. The
worst regression this project has had — speculative decoding duplicating
prompt fragments into answers, including a mangled G.R. number — kept every
liveness probe green. Latency was fine, the GPU was fine, the index was fine,
and the product was broken. A dashboard that only measures whether the machine
is busy will report a healthy system right up until a client reads the output.

PRESENTATION: this app does NOT use the shared law-library theme from
.streamlit/config.toml — it overrides it with a heads-up display (`ui/hud.py`).
That is a deliberate split from app.py, not drift. Haven is read by a worried
patient and stays quiet; this is read at a glance by whoever runs the box, to
answer "is anything wrong", and a status console is the one surface where a
glowing readout is the right register. The override is safe because this runs
as its own app on its own port, so it cannot leak into Haven.

The one rule the HUD does not get to break: a state must never be carried by
motion or glow alone. Every reading here is legible with animation disabled and
in plain text — the chrome decorates a value, it never encodes one.

REFRESH: manual by default, with opt-in live polling — the third attempt at
this, and the first that holds up. The history matters, because two plausible
approaches were tried and reverted:

  1. `st.fragment(run_every=...)` — right mechanism, but its auto-rerun depends
     on a browser-side timer scheduled over Streamlit's websocket protocol, and
     at the time there was no way to verify it end-to-end.
  2. `<meta http-equiv="refresh">` — worked mechanically and was wrong in
     practice: it reloaded the whole visible page every few seconds, which read
     as broken rather than live, and would now also wipe the canary's output.

The current version returns to (1), because a fragment is precisely the fix for
what made (2) unusable: Streamlit clears and redraws the FRAGMENT's elements
and persists the rest of the app, so there is no page flash, scroll position
holds, and anything rendered outside the fragment survives.

Two things keep it honest. It is **opt-in and defaults to off** — this page
shares a GPU with real users and should not poll forever on a spare monitor.
And the browser timer itself still cannot be verified from this environment;
what IS verified is everything around it (fragment wiring, the interval control,
the sample buffer, and that the canary sits outside the fragment so it never
auto-fires). To confirm the timer in a browser: switch Live on and watch the
sample count in the "live traces" panel climb without touching anything.

The tick interval interacts with the probe TTLs rather than overriding them: at
a 5s tick the cheap probes re-read every tick while the LLM handshake (10s) and
index scan (30s) coalesce. The tick sets how often the page can change; the
TTLs set how often the box is actually touched.

A query is logged only once its answer has
*finished* generating (up to ~a minute — see README.md's Performance
section), so don't expect a question still in progress in Haven to appear
here before then; that is normal, not a bug.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from ragmed import config, embeddings, llm, metrics, ocr, rerank, vectorstore
from ui import hud

st.set_page_config(
    page_title="HAVEN // OPS",
    page_icon=":material/radar:",
    layout="wide",
)
st.html(hud.hud_css())


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


@st.cache_data(ttl=5, show_spinner=False)
def host_status() -> dict:
    """CPU / memory / load from /proc.

    Read directly rather than via psutil: this project is deliberately
    dependency-light, /proc is stable, and the embedder, reranker and Chroma
    all live in HOST ram — so this is not an idle nice-to-have. If the box
    starts swapping, retrieval slows down in a way that looks like a model
    problem. Returns {} off Linux instead of raising: a missing panel is a
    better failure than a dead dashboard."""
    out: dict = {}
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, _, v = line.partition(":")
                mem[k] = float(v.strip().split()[0]) * 1024  # kB -> bytes
        total = mem.get("MemTotal", 0.0)
        avail = mem.get("MemAvailable", 0.0)
        out["mem_total"] = total
        out["mem_used"] = total - avail
        out["mem_frac"] = (total - avail) / total if total else 0.0
        sw_t = mem.get("SwapTotal", 0.0)
        out["swap_used"] = sw_t - mem.get("SwapFree", 0.0)
        out["swap_total"] = sw_t
        with open("/proc/loadavg") as f:
            l1, l5, l15, *_ = f.read().split()
        out["load1"], out["load5"], out["load15"] = float(l1), float(l5), float(l15)
        out["cpus"] = os.cpu_count() or 1
        out["load_frac"] = out["load1"] / out["cpus"]
    except (OSError, ValueError, IndexError):
        return {}
    return out


@st.cache_data(ttl=5, show_spinner=False)
def vllm_engine() -> dict:
    """Engine-internal counters from vLLM's Prometheus endpoint.

    `nvidia-smi` says the GPU is busy; this says what the ENGINE is doing —
    how full the KV cache is, whether requests are queueing, and whether the
    prefix cache is actually being hit. That last one is not cosmetic: prefix
    caching is one of the two optimisations still in place, and a config change
    that silently disabled it would otherwise be invisible until someone
    re-benchmarked by hand."""
    try:
        r = httpx.get(config.LLM_BASE_URL.replace("/v1", "") + "/metrics",
                      timeout=4.0)
        r.raise_for_status()
    except Exception:  # noqa: BLE001 - server down is a normal state here
        return {}
    vals: dict[str, float] = {}
    for line in r.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, rest = line.partition("{")
        if not rest:
            name, _, v = line.partition(" ")
        else:
            _, _, v = rest.partition("} ")
        try:
            vals[name.strip()] = float(v)
        except ValueError:
            continue

    def g(k, default=0.0):
        return vals.get(k, default)

    hits, queries = g("vllm:prefix_cache_hits_total"), g("vllm:prefix_cache_queries_total")
    lat_sum = g("vllm:e2e_request_latency_seconds_sum")
    lat_n = g("vllm:e2e_request_latency_seconds_count")
    itl_sum = g("vllm:inter_token_latency_seconds_sum")
    itl_n = g("vllm:inter_token_latency_seconds_count")
    return {
        "kv_usage": g("vllm:kv_cache_usage_perc"),
        "running": g("vllm:num_requests_running"),
        "waiting": g("vllm:num_requests_waiting"),
        "preemptions": g("vllm:num_preemptions_total"),
        "prefix_hit_rate": (hits / queries) if queries else None,
        "prompt_tokens": g("vllm:prompt_tokens_total"),
        "generation_tokens": g("vllm:generation_tokens_total"),
        "requests": lat_n,
        "mean_e2e_s": (lat_sum / lat_n) if lat_n else None,
        # Inter-token latency is the honest decode-speed number: it excludes
        # prefill, so it is not diluted by prompt length the way tok/s is.
        "tok_per_s": (1.0 / (itl_sum / itl_n)) if itl_n and itl_sum else None,
    }


# The invariants this deployment must hold. Each was established by an incident
# or a measurement, and each is silently breakable by editing one docker flag or
# one env var — which is exactly why they are asserted here rather than trusted.
def _config_invariants(engine_args: str) -> list[tuple[str, str, bool, str]]:
    """(check, actual, ok, why) rows."""
    rows = []
    spec_off = "--speculative-config" not in engine_args
    rows.append((
        "speculative decoding OFF", "absent" if spec_off else "ENABLED", spec_off,
        "n-gram spec decode is 1.7x faster and duplicated prompt fragments into "
        "answers, including a mangled G.R. number. Reverted deliberately.",
    ))
    pref_on = "--enable-prefix-caching" in engine_args
    rows.append((
        "prefix caching ON", "on" if pref_on else "OFF", pref_on,
        "The ~480-token system prompt is identical every request. Off, TTFT "
        "roughly doubles.",
    ))
    rows.append((
        "thinking mode OFF", str(config.LLM_ENABLE_THINKING).lower(),
        config.LLM_ENABLE_THINKING is False,
        "Qwen3.6 spends the token budget on chain-of-thought and can return "
        "empty content.",
    ))
    rows.append((
        "temperature > 0", f"{config.LLM_TEMPERATURE}", config.LLM_TEMPERATURE > 0,
        "Greedy decoding makes Qwen3 loop on repeated tokens.",
    ))
    rows.append((
        "retrieval seeded", str(config.LLM_SEED), config.LLM_SEED >= 0,
        "Unseeded HyDE/rewrite calls make the retrieved passages themselves "
        "random — the same question returned 2-3 of the same 10 sources.",
    ))
    return rows


@st.cache_data(ttl=30, show_spinner=False)
def engine_args() -> str:
    """The vLLM container's actual launch flags. '' when docker isn't reachable
    — the invariants that read config.py still apply, the flag-based ones can't
    be judged and say so."""
    try:
        out = subprocess.run(
            ["docker", "inspect", "vllm", "--format", "{{join .Args \" \"}}"],
            capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


@st.cache_data(ttl=30, show_spinner=False)
def index_freshness() -> dict:
    """Is the index older than the corpus it was built from?

    Nothing re-indexes automatically (see CLAUDE.md), so a document added and
    never ingested is invisible: retrieval simply never returns it, and the
    answer says the corpus does not cover it. That reads as a coverage gap
    rather than an operational mistake, which is why it belongs on this page."""
    newest, newest_name = 0.0, ""
    if config.CORPUS_DIR.exists():
        for f in config.CORPUS_DIR.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                m = f.stat().st_mtime
                if m > newest:
                    newest, newest_name = m, str(f.relative_to(config.CORPUS_DIR))
    built = 0.0
    if config.CHROMA_DIR.exists():
        built = max((f.stat().st_mtime for f in config.CHROMA_DIR.rglob("*")
                     if f.is_file()), default=0.0)
    return {"corpus_newest": newest, "corpus_newest_name": newest_name,
            "index_built": built, "stale": bool(newest and built and newest > built)}


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

# The live controls sit OUTSIDE the fragment on purpose: a widget inside a
# fragment only reruns that fragment, so the toggle could never rebuild the
# fragment with a different interval.
ctl_l, ctl_m, ctl_r = st.columns([3, 1, 1], vertical_alignment="bottom")
with ctl_m:
    _live = st.toggle("Live", value=False,
                      help="Re-poll every few seconds. Only this page's panels "
                           "redraw — the page itself is not reloaded.")
with ctl_r:
    _interval = st.selectbox("Interval", ["5s", "10s", "30s"], index=0,
                             disabled=not _live, label_visibility="collapsed")

top_l, top_r = st.columns([5, 1])
with top_l:
    st.caption("Health and query metrics for the RAG engine — not the "
               "user-facing app (`streamlit run app.py` for that). A query "
               "appears below only once its answer has *finished* generating, "
               "which can take up to a minute (see Performance in README.md); "
               "a question still in progress in Haven will not show up here "
               "yet. That is normal, not a stalled dashboard.")
with top_r:
    if st.button(":material/refresh: Re-scan", width="stretch"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Live content — plain top-level code, re-executed on every page load
# (currently only the manual Refresh button; no auto-refresh — a tried
# meta-refresh reload was visually disruptive and was removed).
# ---------------------------------------------------------------------------

def _live_body() -> None:
    llm_ok, llm_msg, llm_latency = llm_health()
    gpu = gpu_status()
    idx = index_stats()
    host = host_status()
    eng = vllm_engine()
    args = engine_args()
    invariants = _config_invariants(args) if args else _config_invariants("")
    fresh = index_freshness()
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
    # Config drift counts toward the overall verdict. Today's worst regression
    # (speculative decoding corrupting answers) was a green dashboard with a
    # broken product — every probe was healthy because none of them looked at
    # whether the engine was configured the way this deployment requires.
    config_ok = all(ok for _, _, ok, _ in invariants)
    overall_ok = llm_ok and idx["chunks"] > 0 and pinning_ok and config_ok

    st.html(hud.banner(
        "SYSTEM NOMINAL" if overall_ok else "ATTENTION REQUIRED",
        "ok" if overall_ok else "alert",
        f"SCAN {time.strftime('%H:%M:%S')}   ·   {config.LLM_MODEL}",
    ))

    core_col, stat_col = st.columns([1, 2], gap="medium")

    with core_col:
        if gpu:
            frac = gpu["used_mib"] / gpu["total_mib"] if gpu["total_mib"] else 0
            st.html(hud.reactor(
                frac,
                f"{gpu['used_mib']/1024:.1f}",
                f"/ {gpu['total_mib']/1024:.0f} GIB",
                "vram core",
                # Near-full is the DESIGNED state here (--gpu-memory-utilization
                # 0.93), so it must not read as an alarm. Only the genuinely
                # unexpected — a card with nothing on it — gets a warn colour.
                "ok" if frac > 0.25 else "warn",
            ))
        else:
            st.html(hud.reactor(0.0, "n/a", "NO GPU", "vram core", "warn"))

        # Fills the space under the core, and earns it: this is the element
        # that answers "is anything wrong" from across the room, before any
        # label has been read. The words under it carry the same verdict, so
        # the colour is a shortcut and never the only signal.
        st.html(hud.neutron(
            overall_ok,
            "all systems nominal" if overall_ok else "fault detected",
        ))

    with stat_col:
        st.html(hud.panel("subsystems", hud.rows([
            ("llm link",
             f"ONLINE · {llm_latency:.0f} ms" if llm_ok else "OFFLINE",
             "ok" if llm_ok else "alert"),
            ("index",
             f"{idx['chunks']:,} passages" if idx["chunks"] else "EMPTY",
             "ok" if idx["chunks"] else "alert"),
            ("cpu pinning",
             "LOCKED" if pinning_ok else "REGRESSION",
             "ok" if pinning_ok else "alert"),
            ("embeddings", emb_dev.upper(), "ok" if emb_dev == "cpu" else "alert"),
            ("reranker",
             ("DISABLED" if rer_dev is None and rer_error is None
              else (rer_dev.upper() if rer_dev else "LOAD FAILED")),
             "ok" if rer_error is None and rer_dev in (None, "cpu") else "alert"),
            ("ocr", "READY" if ocr_ok else
             ("UNAVAILABLE" if config.OCR_ENABLED else "DISABLED"),
             "ok" if ocr_ok else ("warn" if config.OCR_ENABLED else "ok")),
        ]), scan=True))

        if gpu:
            st.html(hud.panel("gpu", hud.rows([
                ("device", gpu["name"], "ok"),
                ("utilisation", f"{gpu['util_pct']:.0f} %", "ok"),
                ("temperature", f"{gpu['temp_c']:.0f} °C",
                 "ok" if gpu["temp_c"] < 80 else "warn"),
            ]) + hud.segbar(gpu["util_pct"] / 100)))

    # --- Configuration invariants -------------------------------------------
    # The most valuable panel here, and the newest. Every other probe answers
    # "is it running"; this answers "is it running the way it must". The bug
    # that shipped today passed every liveness check.
    inv_rows = [(name, actual, "ok" if ok else "alert")
                for name, actual, ok, _ in invariants]
    st.html(hud.panel(
        "configuration invariants" + ("" if args else " · engine flags unreadable"),
        hud.rows(inv_rows)))
    for name, actual, ok, why in invariants:
        if not ok:
            st.error(f"INVARIANT VIOLATED — {name} (found: {actual}). {why}",
                     icon=":material/error:")
    if not args:
        st.caption("`docker inspect vllm` unavailable, so flag-based invariants "
                   "could not be checked — the config.py ones above still apply.")

    # --- Host + engine ------------------------------------------------------
    h_col, e_col = st.columns(2, gap="medium")
    with h_col:
        if host:
            mem_state = "ok" if host["mem_frac"] < 0.9 else "warn"
            load_state = "ok" if host["load_frac"] < 1.0 else "warn"
            swap_state = "ok" if host["swap_used"] < 1 << 30 else "warn"
            st.html(hud.panel("host", hud.rows([
                ("memory",
                 f"{_human(host['mem_used'])} / {_human(host['mem_total'])}",
                 mem_state),
                ("load 1m / 5m / 15m",
                 f"{host['load1']:.2f} / {host['load5']:.2f} / {host['load15']:.2f}",
                 load_state),
                ("cpu cores", f"{host['cpus']}", "ok"),
                ("swap in use", _human(host["swap_used"]), swap_state),
            ]) + hud.segbar(host["mem_frac"], mem_state)))
            st.caption("Embeddings, reranker and Chroma all live in host RAM — "
                       "if this box starts swapping, retrieval slows in a way "
                       "that looks like a model problem.")
        else:
            st.html(hud.panel("host", hud.rows(
                [("proc filesystem", "UNAVAILABLE", "warn")])))

    with e_col:
        if eng:
            hit = eng["prefix_hit_rate"]
            st.html(hud.panel("engine", hud.rows([
                ("kv cache in use", f"{eng['kv_usage'] * 100:.1f} %",
                 "ok" if eng["kv_usage"] < 0.9 else "warn"),
                ("requests running", f"{eng['running']:.0f}", "ok"),
                ("requests waiting", f"{eng['waiting']:.0f}",
                 "ok" if eng["waiting"] == 0 else "warn"),
                ("prefix cache hits",
                 "n/a" if hit is None else f"{hit:.0%}",
                 "ok" if (hit is None or hit > 0) else "warn"),
                ("decode rate",
                 "n/a" if not eng["tok_per_s"] else f"{eng['tok_per_s']:.1f} tok/s",
                 "ok"),
                ("preemptions", f"{eng['preemptions']:.0f}",
                 "ok" if eng["preemptions"] == 0 else "warn"),
                ("tokens in / out",
                 f"{eng['prompt_tokens']:,.0f} / {eng['generation_tokens']:,.0f}",
                 "ok"),
            ])))
            st.caption("Straight from vLLM's own /metrics — engine-internal "
                       "state that nvidia-smi cannot see. Decode rate is "
                       "derived from inter-token latency, so unlike tok/s it "
                       "isn't diluted by prompt length.")
        else:
            st.html(hud.panel("engine", hud.rows(
                [("vllm /metrics", "UNREACHABLE", "alert")])))

    # --- Rolling traces -----------------------------------------------------
    # A number that re-prints the same value tells you nothing about whether it
    # is moving. Each tick appends one sample to a ring buffer in session state,
    # so live mode shows SHAPE — a KV cache filling, a load spike, VRAM held
    # flat by vLLM's reservation — which is the thing a spot reading can't give.
    # Session-scoped and capped: this is a live view, not a metrics store, and
    # data/metrics.jsonl already covers durable history.
    hist = st.session_state.setdefault("hv_hist", [])
    hist.append({
        "gpu_util": gpu["util_pct"] if gpu else None,
        "kv": eng.get("kv_usage", 0.0) * 100 if eng else None,
        "mem": host.get("mem_frac", 0.0) * 100 if host else None,
        "waiting": eng.get("waiting") if eng else None,
    })
    del hist[:-120]          # ~10 min at a 5s tick

    def _series(k):
        return [r[k] for r in hist if r.get(k) is not None]

    traces = ""
    if _series("gpu_util"):
        traces += hud.sparkline(_series("gpu_util"), "gpu utilisation",
                                f"{hist[-1]['gpu_util']:.0f} %")
    if _series("kv"):
        traces += hud.sparkline(_series("kv"), "kv cache",
                                f"{hist[-1]['kv']:.1f} %")
    if _series("mem"):
        traces += hud.sparkline(_series("mem"), "host memory",
                                f"{hist[-1]['mem']:.0f} %")
    if traces:
        st.html(hud.panel(f"live traces · {len(hist)} samples", traces))
        if len(hist) < 3:
            st.caption("Traces build up as the page re-polls — turn on **Live** "
                       "above, or keep pressing Re-scan.")

    # --- Index freshness ----------------------------------------------------
    if fresh["stale"]:
        st.warning(
            f"Corpus is newer than the index — `{fresh['corpus_newest_name']}` "
            f"changed at {time.strftime('%Y-%m-%d %H:%M', time.localtime(fresh['corpus_newest']))}, "
            f"after the index was last written "
            f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(fresh['index_built']))}). "
            "Nothing re-indexes automatically: run `python cli.py ingest`. Until "
            "you do, those documents cannot be retrieved and answers will read "
            "as though the corpus does not cover them.",
            icon=":material/sync_problem:")

    # The failure this dashboard exists to catch gets a full-width, plain-text
    # alert — never a colour change alone.
    if not pinning_ok:
        st.error("CPU PINNING REGRESSION — the embedding or reranker model has "
                 "loaded onto the GPU. vLLM holds ~93% of VRAM by design, so "
                 "the next load will hit a hard OutOfMemoryError mid-query. "
                 "See HAVEN_VLLM_MIGRATION.md §9.", icon=":material/error:")
    if not llm_ok:
        st.error(llm_msg, icon=":material/error:")

    # -----------------------------------------------------------------------
    # Compute health / Index & corpus — side by side, detail behind the summary
    # -----------------------------------------------------------------------

    col_compute, col_index = st.columns(2)

    with col_compute:
        st.subheader("Compute")
        st.caption(
            "VRAM sitting near-full is expected, not a warning: vLLM is "
            "configured with `--gpu-memory-utilization 0.93` by design (see "
            "HAVEN_VLLM_MIGRATION.md §5). What matters is the table below — "
            "which device each CPU-bound model actually landed on."
        )
        if not gpu:
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

    err_rate = df["errored"].mean()
    recent_first = df.sort_values("time")

    t1, t2 = st.columns(2, gap="medium")
    with t1:
        st.html(hud.panel("telemetry", (
            hud.sparkline(recent_first["total_ms"].tolist(), "total",
                          f"{df['total_ms'].median()/1000:.1f} s median")
            + hud.sparkline(recent_first["retrieval_ms"].tolist(), "retrieval",
                            f"{df['retrieval_ms'].median()/1000:.1f} s median")
            + hud.sparkline(recent_first["generation_ms"].tolist(), "generation",
                            f"{df['generation_ms'].median()/1000:.1f} s median")
        )))
    with t2:
        st.html(hud.panel("counters", hud.rows([
            ("queries logged", f"{len(df):,}", "ok"),
            ("median latency", f"{df['total_ms'].median()/1000:.1f} s", "ok"),
            ("hyde fire rate", f"{df['hyde_fired'].mean():.0%}", "ok"),
            ("error rate", f"{err_rate:.0%}",
             "ok" if err_rate == 0 else "alert"),
            ("chunks per answer", f"{df['num_chunks'].median():.0f}", "ok"),
        ]) + hud.segbar(1.0 - err_rate, "ok" if err_rate == 0 else "alert")))
        st.caption("Success rate across all logged queries.")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Queries per " + ("hour" if bucket == "h" else "day"))
        vol = df.groupby("bucket").size().rename("queries").to_frame()
        st.bar_chart(vol, y="queries", height=240)
    with c2:
        st.caption("Latency by stage (ms, median per bucket)")
        lat = df.groupby("bucket")[["retrieval_ms", "generation_ms"]].median()
        st.line_chart(lat, height=240)

    # Percentiles, not just the median: the median said 15s all through the
    # week the p99 was a minute. A tail is what a client actually notices.
    p = df["total_ms"].quantile([0.5, 0.9, 0.99]) / 1000
    st.html(hud.panel("latency distribution", hud.rows([
        ("p50", f"{p[0.5]:.1f} s", "ok"),
        ("p90", f"{p[0.9]:.1f} s", "ok" if p[0.9] < 45 else "warn"),
        ("p99", f"{p[0.99]:.1f} s", "ok" if p[0.99] < 60 else "warn"),
        ("slowest logged", f"{df['total_ms'].max()/1000:.1f} s",
         "ok" if df["total_ms"].max() / 1000 < 90 else "warn"),
    ])))

    st.caption("Recent queries")
    show_cols = ["time", "question", "num_chunks", "hyde_fired", "total_ms", "error"]
    recent = df.sort_values("time", ascending=False)[[c for c in show_cols if c in df.columns]].head(25)
    if "question" not in df.columns or df["question"].isna().all():
        recent = recent.drop(columns=[c for c in ["question"] if c in recent.columns])
        st.caption("(question text not logged — METRICS_LOG_QUESTIONS=false)")
    st.dataframe(recent, hide_index=True, width="stretch")


def _canary() -> None:
    """Ask one known question and inspect the ANSWER, not the clock.

    Every other probe on this page is a liveness check, and today proved that
    a system can pass all of them while producing broken output: speculative
    decoding kept latency healthy and quietly duplicated prompt fragments into
    answers, and the prompt's own labels were being emitted as citations that
    resolve to nothing. Both were found by a human reading a reply, which is
    not a monitoring strategy.

    So this checks the three properties that actually failed:
      * no prompt-internal passage labels leaked into the answer;
      * no fragment is repeated back-to-back (the spec-decode signature);
      * the answer cites something with a real identifier.

    On demand, never automatic — it costs a full generation on the shared GPU,
    and a dashboard that quietly competes with users for the model is worse
    than one that makes you press a button."""
    st.subheader("Answer-quality canary")
    st.caption(
        "Liveness checks cannot see a corrupted answer. This asks one known "
        "question and inspects the reply for the two defects that have "
        "actually shipped: leaked passage labels, and repeated fragments. "
        "Costs one generation (~25s) on the same GPU users share."
    )
    if not st.button(":material/biotech: Run canary", width="content"):
        return

    from ragmed import rag

    q = "What are the grounds for revoking a physician's certificate of registration?"
    with st.spinner("Asking the canary question…"):
        t0 = time.perf_counter()
        try:
            ans = rag.answer(q, stream=False)
        except Exception as e:  # noqa: BLE001 - report, don't kill the page
            st.error(f"Canary could not complete: {e}", icon=":material/error:")
            return
        elapsed = (time.perf_counter() - t0) * 1000

    text = ans.text
    labels = re.findall(r"\[Context\s*\d+\]|\bPASSAGE\s+\d+\b", text, re.I)
    # A 1-4 word fragment repeated back-to-back, separated only by quotes or
    # punctuation \u2014 the shape a mis-accepted speculative draft leaves behind.
    frag_re = r"""\b([A-Za-z][\w']*(?:\s+[\w']+){0,3})\b[\s"'\u2019,.;:]{1,6}\1\b"""
    frags = [m.group(1) for m in re.finditer(frag_re, text, re.I)]
    cited = re.findall(r"(?:Republic Act No\.|G\.R\. No\.|Sec\.|Section|Article)\s*\S+",
                       text)

    checks = [
        ("no leaked passage labels", f"{len(labels)} found", not labels),
        ("no repeated fragments", f"{len(frags)} found", not frags),
        ("answer cites authority", f"{len(cited)} citations", bool(cited)),
        ("sources returned", f"{len(ans.sources)}", bool(ans.sources)),
    ]
    st.html(hud.panel(f"canary · {elapsed/1000:.1f} s", hud.rows(
        [(n, v, "ok" if ok else "alert") for n, v, ok in checks])))

    for n, v, ok in checks:
        if not ok:
            st.error(f"CANARY FAILED — {n} ({v}). See the answer below.",
                     icon=":material/error:")
    if all(ok for _, _, ok in checks):
        st.success("All canary checks passed.", icon=":material/check_circle:")
    with st.expander("Canary answer in full"):
        st.markdown(text)


# Auto-refresh, third attempt — and the first that holds up.
#
# The two earlier tries are recorded in the module docstring: `st.fragment`
# could not be verified end-to-end at the time, and a `<meta http-equiv=refresh>`
# reloaded the whole visible page every few seconds, which read as broken rather
# than live. A fragment fixes precisely that second failure — Streamlit clears
# and redraws the FRAGMENT's elements and persists the rest of the app, so the
# page does not flash, scroll position holds, and the canary's output below is
# not wiped every tick.
#
# It is opt-in and defaults to off. This page shares a GPU with real users, and
# a dashboard left open on a spare monitor should not poll forever by default.
#
# Note the interval interacts with the probe TTLs above rather than overriding
# them: at a 5s tick the cheap probes (GPU, host, engine) genuinely re-read,
# while the LLM handshake (10s) and index scan (30s) still coalesce. That is
# the intended behaviour — the tick rate sets how often the page can change,
# the TTLs set how often the box is actually touched.
if _live:
    st.fragment(_live_body, run_every=_interval)()
else:
    _live_body()

st.divider()
# Deliberately OUTSIDE the fragment: the canary costs a full generation on the
# shared GPU, and anything inside would fire on every tick.
_canary()
