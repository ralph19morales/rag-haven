"""Streamlit web app for the Philippine medical-law RAG system.

Run with:  streamlit run app.py

WHO THIS IS FOR: ordinary people trying to find out where they stand — a
patient, a relative, a nurse — not the engineer who built the index. That
decision drives most of what follows, and it is the reason this page states
almost nothing about its own machinery.

WHAT IS DELIBERATELY NOT ON SCREEN, and where it went. Assume a stranger opens
this. None of the following changes what they should do next, so none of it is
shown: the language model's name, the embedding and reranker models, how many
passages are indexed, how many were retrieved for this answer, relevance
scores, and the filename each passage came from. An operator wants every one of
them — `cli.py status` and `dashboard.py` have them all. On this page they read
either as noise or, worse, as precision that invites someone to defer to the
answer instead of checking it. The same logic governs failures: when the model
is unreachable the reader is told plainly and told what to do, not shown the
transport error.

Two things are deliberately unmissable rather than tucked away:
  * that this is not legal advice, stated in the hero AND under every answer;
  * where each answer came from — named as the law names itself, so it can be
    carried to a lawyer or searched for — because an answer nobody can check is
    worth less than no answer.

PRESENTATION. Colours and type come from .streamlit/config.toml; Haven commits
to one dark ground rather than following the reader's light/dark preference, so
nothing here branches on theme. Custom markup lives in `ui/`: the animated mark
(`robot.py`), and the page chrome, advisory panel and source cards
(`chrome.py`).

This page is a heads-up display, the same register as the ops dashboard — cyan
on near-black, scanlines, corner brackets, glow, monospace — and `ui/chrome.py`
is built on `ui/hud.py`'s stylesheet rather than beside it, so the two cannot
drift. That is a change of intent: the two surfaces used to be deliberately
unlike each other, on the reasoning that a legal answer rendered in sci-fi
chrome reads as a toy and takes its disclaimer down with it.

That risk is real and did not go away with the palette, so it is handled
structurally instead of chromatically, and these are the parts not to trade away
for atmosphere:
  * the not-legal-advice line is an ALERT-weight panel (`chrome.advisory`), not
    ambient cyan — on a console the accent is the colour of everything that is
    merely working, and the eye learns to skip it;
  * the caveat sentence is never set in tracked-out capitals. Capitals are for
    micro-labels and headings. A legal caveat styled as a HUD label reads as
    decoration and gets skipped;
  * every state the mark shows in colour is also stated in words next to it.
"""
from __future__ import annotations

import streamlit as st

from ragmed import config, embeddings, llm, rag, rerank
from ui import chrome
from ui.robot import robot_hero, robot_thinking

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
    "The hospital won't release my relative's body until we pay the bill. Can they do that?",
    "Can a hospital refuse to treat me in an emergency because I can't pay a deposit?",
    "Am I entitled to a senior citizen or PWD discount on my hospital bill?",
    "My doctor never explained the risks before my operation. What was I owed?",
]


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


def render_sources(sources: list[dict], expanded: bool = False) -> None:
    """Where the answer came from, named the way the law names itself.

    The relevance score is deliberately not shown. It is meaningful when tuning
    retrieval and meaningless to someone asking whether they can take their
    relative home — and a number beside a legal citation invites false confidence.

    Rendered BELOW the answer. This used to be above it, and the reason it was
    is worth recording because it expired rather than being wrong: generation
    took about a minute, the passages were known the moment retrieval finished,
    and putting them first gave someone a real thing to read during the wait.
    Generation is now several times faster and the wait is held by an animated
    mark instead (ui/robot.py), so the answer can sit where a reader expects it
    and the sources can support it from underneath.

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
        #
        # No "retrieved passages · N" micro-label above them any more: the
        # expander this sits inside is already titled "Where this comes from",
        # and "retrieved passages" is the system's word for it, not a reader's.
        st.html("".join(
            chrome.source_card(i, s["law"], s["section"])
            for i, s in enumerate(sources, 1)
        ))


# --- State -----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# The chrome goes first, before anything that has to be styled by it — which
# now includes the boot panel immediately below.
st.html(chrome.chrome_css())

# --- Boot ------------------------------------------------------------------
# The embedder and reranker load on first use: about six seconds on CPU, paid
# by whoever opens the page first, and until now paid against a BLANK page. A
# blank page reads as broken rather than as busy, which is the worst possible
# first impression for a page someone arrived at worried. The nucleus holds
# that time instead, and is cleared the moment the work finishes.
#
# Gated on session_state, not on the cache. `warm_up` is cached per SERVER, so
# a later visitor pays nothing: the placeholder is written and cleared inside a
# single script run and never paints. The flag exists only to stop it
# reappearing on every rerun of a session that has already started up.
boot = st.empty()
if "booted" not in st.session_state:
    boot.html(chrome.booting())
ok, msg = llm_status()
warm_up()
boot.empty()
st.session_state.booted = True


# --- Unreachable model -----------------------------------------------------
# Stated BEFORE the mark, not after it. The mark turns red in this state, and a
# colour must never be the first or only thing that says something is wrong —
# the sentence has to have been read already by the time the reader reaches the
# drawing. `msg` (the transport error, with a URL and port in it) is
# deliberately not printed: see the module docstring.
if not ok:
    st.error(
        f"{ASSISTANT} isn't running on this computer right now, so questions "
        "can't be answered yet.",
        icon=":material/error:",
    )
    st.caption(
        "Nothing you typed has been lost. Try again in a few minutes, or ask "
        "whoever set this up to check it."
    )


# --- Hero ------------------------------------------------------------------
if not st.session_state.messages:
    st.html(robot_hero(alert=not ok))
    st.title(ASSISTANT, text_alignment="center")
    st.markdown(
        "Answers about Philippine medical law, drawn from the law itself.",
        text_alignment="center",
    )
    st.html(chrome.hero_meta())
    # The one panel on this page that is NOT in the ambient accent. It is the
    # sentence that has to be believed, and on a console everything cyan reads
    # as "working normally" within a few seconds of looking at it.
    st.html(chrome.advisory(
        "<b>Haven does not give legal advice.</b> It searches the law and "
        "quotes it back to you. It cannot weigh the facts of your situation, "
        "and it is no substitute for talking to a lawyer."
    ))
else:
    st.title(ASSISTANT)


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
        stage.html(robot_thinking("Searching the law…"))

        gen, chunks = rag.answer(question, top_k=config.TOP_K, stream=True,
                                 history=history)

        # `gen` is lazy: retrieval has finished, but the model has not read the
        # passages yet. That is a second, distinct wait, so the mark keeps
        # running and says which one we are in rather than freezing on the
        # first phase.
        sources = [{
            "law": c.metadata.get("law", ""),
            "section": c.metadata.get("section", ""),
        } for c in chunks]
        if sources:
            stage.html(robot_thinking(
                "Reading the law and writing your answer…"))

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

    # "How this works" is kept because trust is the thing this page trades in,
    # but it is now the PLAIN answer only. It used to end with a line of model
    # identifiers and an indexed-passage count — the sort of detail that reads
    # as reassuring precision to whoever built the index and as noise, or worse
    # as a reason to defer, to the person this app is for. None of it is
    # actionable: nobody chooses whether to trust a legal answer on the
    # strength of a reranker's name. It lives in `cli.py status` and the ops
    # dashboard, which is where an operator will look for it.
    with st.expander("How this works", icon=":material/settings:"):
        st.markdown(
            "It searches the actual text of Philippine laws, health "
            "regulations and Supreme Court rulings for the passages closest "
            "to your question, then writes an answer using only those "
            "passages — and shows you which ones it used."
        )
        st.markdown(
            "It cannot use anything outside them, which is why it will "
            "sometimes tell you it doesn't know."
        )
