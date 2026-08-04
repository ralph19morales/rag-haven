"""Streamlit web app for the Philippine medical-law RAG system.

Run with:  streamlit run app.py

WHO THIS IS FOR: ordinary people trying to find out where they stand — a
patient, a relative, a nurse — not the engineer who built the index. That
decision drives most of what follows. The retrieval machinery (embedding model,
BM25, reranker, chunk scores) is real and worth knowing about, but it is not
what a worried person needs on screen, so it lives behind "How this works"
instead of in a sidebar of model names.

Two things are deliberately unmissable rather than tucked away:
  * that this is not legal advice, stated in the hero AND under every answer;
  * where each answer came from, in plain language, because an answer nobody
    can check is worth less than no answer.

Colours and type come from .streamlit/config.toml — this app honours the
reader's light/dark preference, unlike dashboard.py which forces its own dark
ground. Custom markup lives in `ui/`: the hero mark (`hero.py`), the waiting
animation (`thinking.py`), and the page chrome and source cards (`chrome.py`).

`chrome.py` is a deliberately quiet relative of the ops dashboard's heads-up
display (`ui/hud.py`). It borrows that screen's structure — corner brackets,
fading hairline rules, uppercase letterspaced micro-labels for metadata — and
none of its voltage: no dark ground, no high-chroma accent, no scanlines, no
glow on text, nothing that rotates. The reason is the audience. The dashboard is
scanned from across a room to see whether anything is broken; this page is read
slowly by someone who is worried, and it carries a disclaimer that has to be
believed. Micro-labels are therefore used ONLY for metadata — never for the
answer and never for the disclaimer, because a legal caveat set in tracked-out
capitals reads as decoration and gets skipped.
"""
from __future__ import annotations

import streamlit as st

from ragmed import config, embeddings, llm, rag, rerank, vectorstore
from ui import chrome
from ui.hero import scales_svg
from ui.thinking import thinking_mark

ASSISTANT = "Haven"

st.set_page_config(
    page_title=f"{ASSISTANT} — Philippine medical law",
    page_icon=":material/balance:",
    layout="centered",
)

# Haven opens the conversation rather than leaving a blank prompt. People who
# arrive worried do not know what this can answer or what to call their problem,
# so the greeting states the scope in plain words and invites them to use their
# own. It is rendered as a chat message but deliberately NOT stored in
# `messages`: it is not a turn, and keeping it out means the hero, the example
# situations and the greeting all disappear together the moment a real
# conversation starts.
GREETING = (
    f"Hello, I'm **{ASSISTANT}**.\n\n"
    "I can look up what Philippine law actually says about your rights as a "
    "patient — hospital bills, emergency treatment, consent, discounts you may "
    "be entitled to, or a doctor's duty of care.\n\n"
    "Tell me what's happening in your own words, and I'll show you the law it "
    "comes from. **How can I help?**"
)

# Real situations people arrive with, in the words they would actually use —
# not "grounds for revocation of a certificate of registration". Each one is
# answerable from the indexed corpus.
COMMON_QUESTIONS = [
    "The hospital won't release my father's body until we pay the bill. Can they do that?",
    "Can a hospital refuse to treat me in an emergency because I can't pay a deposit?",
    "Am I entitled to a senior citizen or PWD discount on my hospital bill?",
    "My doctor never explained the risks before my operation. What was I owed?",
]


@st.cache_resource
def get_collection():
    return vectorstore.get_collection()


@st.cache_resource(show_spinner=False)
def warm_up() -> bool:
    """Load the embedding and reranking models when the app starts.

    Both are lazy and process-cached, so without this the first person to ask a
    question waits ~6s for them on top of their answer — the worst possible
    moment to pay it, and the reason the first entry in metrics.jsonl showed
    ~19s of "retrieval" for a question that actually retrieves in about 1s.
    st.cache_resource makes this run once per server, not once per rerun."""
    embeddings.warmup()
    rerank.warmup()
    return True


@st.cache_data(ttl=30, show_spinner=False)
def llm_status() -> tuple[bool, str]:
    return llm.is_available()


def theme_colors() -> tuple[str, str, str]:
    """Stroke colours for the hero mark, matched to the active theme."""
    dark = getattr(st.context.theme, "type", "light") == "dark"
    if dark:
        return "#4d5a70", "#a3b6e2", "#39445a"
    return "#aab4c2", "#38476b", "#c7cfda"


def render_sources(sources: list[dict], expanded: bool = False) -> None:
    """Where the answer came from, named the way the law names itself.

    The relevance score is deliberately not shown. It is meaningful when tuning
    retrieval and meaningless to someone asking whether they can take their
    father home — and a number beside a legal citation invites false confidence.

    Rendered BELOW the answer. This used to be above it, and the reason it was
    is worth recording because it expired rather than being wrong: generation
    took about a minute, the passages were known the moment retrieval finished,
    and putting them first gave someone a real thing to read during the wait.
    Generation is now several times faster and the wait is held by an animated
    mark instead (ui/thinking.py), so the answer can sit where a reader expects
    it and the sources can support it from underneath.

    `expanded` must match between the live turn and the replay from history.
    Streamlit re-renders the whole conversation after each turn, so an expander
    that is open live and closed in history visibly collapses the instant the
    answer completes — which is exactly when the reader is deciding whether to
    check it.
    """
    if not sources:
        return
    with st.expander(f"Where this comes from ({len(sources)})",
                     icon=":material/menu_book:", expanded=expanded):
        st.caption(
            "The assistant was only allowed to read these passages. Open the "
            "law itself before you rely on any of it."
        )
        # One html block for the whole list rather than a widget per source:
        # the cards are pure presentation, and building them as Streamlit
        # containers meant a bordered box inside a bordered expander inside a
        # chat bubble — three nested frames competing for the same edge.
        st.html(chrome.label(f"retrieved passages · {len(sources)}") + "".join(
            chrome.source_card(i, s["law"], s["section"], s["source"])
            for i, s in enumerate(sources, 1)
        ))


# --- State -----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

ok, msg = llm_status()
warm_up()
stroke, accent, faint = theme_colors()
st.html(chrome.chrome_css(dark=getattr(st.context.theme, "type", "light") == "dark"))


# --- Hero ------------------------------------------------------------------
if not st.session_state.messages:
    st.html(scales_svg(stroke, accent, faint))
    st.title(ASSISTANT, text_alignment="center")
    st.markdown(
        "Answers about Philippine medical law, drawn from the law itself.",
        text_alignment="center",
    )
    st.html(chrome.hero_meta(vectorstore.count(get_collection())))
    st.warning(
        "**Haven does not give legal advice.** It searches the law and quotes "
        "it back to you. It cannot weigh the facts of your situation, and it "
        "is no substitute for talking to a lawyer.",
        icon=":material/gavel:",
    )
else:
    st.title(ASSISTANT)

if not ok:
    st.error(
        f"{ASSISTANT} isn't running on this computer right now, so questions "
        "can't be answered yet.",
        icon=":material/error:",
    )
    with st.expander("What this means"):
        st.caption(msg)


# --- Conversation ----------------------------------------------------------
for turn, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            # Keyed by turn INDEX, not id(m): Streamlit keys must be stable
            # across reruns, and object ids are recycled — two turns could
            # collide, or one could change key on every rerun.
            with st.container(key=f"hv-answer-{turn}"):
                st.markdown(m["content"])
        else:
            st.markdown(m["content"])
        render_sources(m.get("sources", []))
        if m["role"] == "assistant":
            st.caption(
                ":material/info: Information, not legal advice — check the "
                "sources above."
            )

question = st.chat_input("Tell Haven what's happening…")

# An empty screen is an invitation to act. Most people do not know what this
# can answer, so Haven opens, then offers real situations to start from.
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(GREETING)
    st.html(chrome.label("or start with a common situation"))
    # Keyed so chrome.py can scope its CSS to these buttons alone — Streamlit
    # exposes the key as a `.st-key-…` class, which is the supported way to
    # style a specific widget rather than every button on the page.
    with st.container(key="hv-situations"):
        for i, q in enumerate(COMMON_QUESTIONS):
            if st.button(q, key=f"common_{i}", width="stretch"):
                question = q

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not ok:
        st.stop()

    # Everything before this turn. Haven uses it to work out what a follow-up
    # refers to; it is never treated as a source (see ragmed/conversation.py).
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        # One placeholder holds the whole turn: the waiting mark, then the
        # answer that replaces it. Writing both through `stage` is what makes
        # the transition a swap rather than a spinner disappearing and text
        # appearing somewhere else.
        answer_box = st.container(key="hv-answer-live")
        stage = answer_box.empty()
        stage.html(thinking_mark(stroke, accent, faint, "Searching the law…"))

        gen, chunks = rag.answer(question, top_k=config.TOP_K, stream=True,
                                 history=history)

        # `gen` is lazy: retrieval has finished, but the model has not read the
        # passages yet. That is a second, distinct wait, so the mark keeps
        # running and says which one we are in rather than freezing on the
        # first phase.
        sources = [{
            "law": c.metadata.get("law", ""),
            "section": c.metadata.get("section", ""),
            "source": c.metadata.get("source", ""),
        } for c in chunks]
        if sources:
            stage.html(thinking_mark(
                stroke, accent, faint,
                f"Reading {len(sources)} passages and writing your answer…"))

        buf = ""
        for token in gen:
            buf += token
            stage.markdown(buf)      # the first token clears the mark

        # An answer that came back empty would otherwise leave the scales
        # swinging for ever, since nothing ever overwrites them.
        if not buf:
            stage.empty()
            st.caption(
                ":material/error: No answer came back. The model may have "
                "stopped early — try asking again."
            )

        render_sources(sources)
        st.caption(
            ":material/info: Information, not legal advice — check the "
            "sources above."
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": buf, "sources": sources}
    )
    st.rerun()


# --- Sidebar: plain language first, machinery last -------------------------
with st.sidebar:
    st.markdown(f"### About {ASSISTANT}")
    st.markdown(
        f"{ASSISTANT} searches the text of Philippine laws, health regulations "
        "and Supreme Court decisions, then answers using only what it found. "
        "Every answer shows the passages behind it."
    )
    st.markdown(
        "It runs entirely on this computer. Nothing you type is sent anywhere."
    )

    if st.session_state.messages:
        if st.button("Ask something else", icon=":material/refresh:",
                     width="stretch"):
            st.session_state.messages = []
            st.rerun()

    with st.expander("What it can't do", icon=":material/error:"):
        st.markdown(
            "- It can't give legal advice or tell you what to do.\n"
            "- It only knows the documents indexed on this computer, so it "
            "will miss anything not yet added.\n"
            "- It can be confidently wrong. Always open the source it cites."
        )

    with st.expander("How this works", icon=":material/settings:"):
        st.caption(
            f"{vectorstore.count(get_collection()):,} passages indexed. "
            "Each question is matched against them by meaning and by keyword, "
            "the best matches are re-ranked for relevance, and a language "
            "model writes the answer from those passages alone."
        )
        st.caption(
            f"Language model: {config.LLM_MODEL} · "
            f"Search: {config.EMBED_MODEL} + BM25 · "
            f"Re-ranking: {config.RERANK_MODEL if config.RERANK_ENABLED else 'off'}"
        )
