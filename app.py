"""
app.py — Context-Aware Q&A Assistant (Streamlit + LangChain LCEL)

Run with:  streamlit run app.py

Demonstrates:
- LCEL chain composition (prompt | model | parser)
- Streaming responses
- Token-budgeted conversation history with auto-summarisation
- Dynamic system prompt based on user role
- Structured output (each answer is a validated Pydantic object)
- One-line model backend swap (see config.py)
"""

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

import config
from prompts import build_chat_prompt, QAAnswer
from context_manager import manage_context, count_tokens, extract_text

st.set_page_config(page_title="Context-Aware Q&A Assistant", page_icon="💬")

# ---------------------------------------------------------------------
# SESSION STATE — this is our "message state" (LangChain concept #4)
# ---------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list[HumanMessage | AIMessage], full raw log
if "summary" not in st.session_state:
    st.session_state.summary = ""  # rolling summary of trimmed-away turns
if "last_turn_tokens" not in st.session_state:
    st.session_state.last_turn_tokens = 0

# ---------------------------------------------------------------------
# SIDEBAR — role selector (session metadata) + diagnostics
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Session settings")
    role = st.selectbox("Answering for a…", ["developer", "manager", "student"])

    st.divider()
    st.subheader("Context engineering stats")
    st.write(f"**Backend:** {config.MODEL_BACKEND} ({config.MODEL_NAMES[config.MODEL_BACKEND]})")
    st.write(f"**Token budget:** {config.MAX_HISTORY_TOKENS}")
    st.write(f"**Last turn tokens sent:** {st.session_state.last_turn_tokens}")
    if st.session_state.summary:
        st.write("**Rolling summary (older turns):**")
        st.caption(st.session_state.summary)

    if st.button("Reset conversation"):
        st.session_state.history = []
        st.session_state.summary = ""
        st.session_state.last_turn_tokens = 0
        st.rerun()

st.title("💬 Context-Aware Q&A Assistant")

# Render existing chat
for msg in st.session_state.history:
    role_label = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role_label):
        st.write(msg.content)

question = st.chat_input("Ask something…")

if question:
    with st.chat_message("user"):
        st.write(question)

    # Build the model once (config.py is the single switch point)
    model = config.get_chat_model(streaming=True)

    # --- CONTEXT ENGINEERING STEP ---
    # Decide what history actually goes into the prompt: verbatim
    # recent turns, plus a rolling summary of anything trimmed.
    trimmed_history, updated_summary, history_tokens = manage_context(
        st.session_state.history, model, st.session_state.summary
    )
    st.session_state.summary = updated_summary

    # --- LCEL CHAIN ---
    # prompt | model | structured_output  — this IS the pipe syntax.
    prompt = build_chat_prompt(role=role, extra_context=updated_summary)
    structured_model = model.with_structured_output(QAAnswer)
    chain = prompt | structured_model

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        # NOTE: with_structured_output disables token-by-token streaming
        # for most backends (the model must finish the JSON object first).
        # To demonstrate BOTH structured output and live streaming, we
        # stream a plain-text pass for the UI, then run structured output
        # for the validated answer + follow-up shown below the reply.
        plain_chain = prompt | model
        for chunk in plain_chain.stream(
            {"question": question, "history": trimmed_history}
        ):
            streamed_text += extract_text(chunk.content)
            placeholder.markdown(streamed_text + "▌")
        placeholder.markdown(streamed_text)

        structured: QAAnswer = chain.invoke(
            {"question": question, "history": trimmed_history}
        )
        st.caption(
            f"Confidence: **{structured.confidence}** · "
            f"Follow-up idea: _{structured.follow_up_question}_"
        )

    # --- UPDATE STATE ---
    st.session_state.history.append(HumanMessage(content=question))
    st.session_state.history.append(AIMessage(content=streamed_text))
    st.session_state.last_turn_tokens = history_tokens + count_tokens(question)
