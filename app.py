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

Colours and type come from .streamlit/config.toml. The only custom markup is
the animated hero mark in ui/hero.py, which Streamlit has no native equivalent
for.
"""
from __future__ import annotations

import streamlit as st

from ragmed import config, embeddings, llm, rag, rerank, vectorstore
from ui.hero import scales_svg

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

    Rendered ABOVE the answer, always. On this hardware the model spends about a
    minute reading before it writes a word, and the passages are known the
    instant retrieval finishes — so they are shown first, giving someone a real
    thing to read during the wait instead of a spinner. Keeping the position
    identical live and in history means nothing jumps when the turn completes.
    """
    if not sources:
        return
    with st.expander(f"Where this comes from ({len(sources)})",
                     icon=":material/menu_book:", expanded=expanded):
        st.caption(
            "The assistant was only allowed to read these passages. Open the "
            "law itself before you rely on any of it."
        )
        for i, s in enumerate(sources, 1):
            with st.container(border=True):
                title = s["law"] or "Untitled document"
                st.markdown(f"**{i}. {title}**")
                if s["section"]:
                    st.markdown(s["section"])
                st.caption(s["source"])


# --- State -----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

ok, msg = llm_status()
warm_up()
stroke, accent, faint = theme_colors()


# --- Hero ------------------------------------------------------------------
if not st.session_state.messages:
    st.html(scales_svg(stroke, accent, faint))
    st.title(ASSISTANT, text_alignment="center")
    st.markdown(
        "Answers about Philippine medical law, drawn from the law itself.",
        text_alignment="center",
    )
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
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render_sources(m.get("sources", []))
        st.markdown(m["content"])
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
    st.markdown("##### Or start with a common situation")
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
        with st.spinner("Searching the law…"):
            gen, chunks = rag.answer(question, top_k=config.TOP_K, stream=True,
                                     history=history)

        # `gen` is lazy: retrieval has finished but the model has not started
        # reading yet, so the passages can go on screen now rather than a
        # minute from now when the first token arrives.
        sources = [{
            "law": c.metadata.get("law", ""),
            "section": c.metadata.get("section", ""),
            "source": c.metadata.get("source", ""),
        } for c in chunks]
        render_sources(sources, expanded=True)

        placeholder = st.empty()
        if sources:
            placeholder.caption(
                ":material/hourglass_top: Reading these passages and writing "
                "your answer — this can take a minute on this computer."
            )
        buf = ""
        for token in gen:
            buf += token
            placeholder.markdown(buf)

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
