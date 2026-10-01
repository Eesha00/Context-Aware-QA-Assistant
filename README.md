# Context-Aware Q&A Assistant (Phase 2 project)

A Streamlit chat app built on LangChain's LCEL (`|`) syntax that
demonstrates context engineering: deciding what actually goes into
the model's context window on every turn.

## Setup

```bash
cd context_qa_assistant
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# edit .env: set MODEL_BACKEND and the matching API key
streamlit run app.py
```

If using Ollama, install it separately (ollama.com) and run
`ollama pull llama3.2:1b` before starting the app.

For Gemini: grab a free key from https://aistudio.google.com/app/apikey
and put it in `.env` as `GOOGLE_API_KEY`.

## File map

| File | Role |
|---|---|
| `config.py` | The ONE place that picks the model backend (OpenAI / Anthropic / Ollama). Change `MODEL_BACKEND` and everything downstream keeps working. |
| `prompts.py` | Dynamic system prompt (changes by user role) + the `QAAnswer` structured-output schema. |
| `context_manager.py` | Token counting (tiktoken) and the summarise-when-over-budget logic. |
| `app.py` | Streamlit UI + the actual LCEL chain (`prompt \| model \| parser`) + streaming. |

## How each Phase 2 topic shows up in the code

**LCEL / `|` syntax** — `app.py`: `chain = prompt | structured_model` and
`plain_chain = prompt | model`. Both are runnables, so both get
`.stream()` and `.invoke()` for free without extra code.

**Chat models, swappable backend** — `config.get_chat_model()`. The
rest of the app only ever calls this function, never a specific
SDK, so swapping backend is a one-line env var change, not a code
change.

**Prompt templates, dynamic prompts** — `prompts.build_chat_prompt()`
builds a different system message depending on the selected role
(developer / manager / student) — this is "dynamic prompts based on
session metadata."

**Structured output** — `QAAnswer` (a Pydantic model) is passed to
`model.with_structured_output(QAAnswer)`, so the model's reply is
parsed and validated into `.answer`, `.confidence`,
`.follow_up_question` instead of raw text you'd have to regex out.

**Message state / conversation history** — `st.session_state.history`
is a plain list of `HumanMessage` / `AIMessage` objects — LangChain's
message state pattern, kept in Streamlit's session instead of a
database for this mini-project.

**Context window budgeting** — `context_manager.manage_context()`:
counts tokens with tiktoken, and once `MAX_HISTORY_TOKENS` is
exceeded, older turns get summarised into a rolling note (via a
call back to the same model) while the last `KEEP_LAST_N_MESSAGES`
stay verbatim. The summary is injected into the *system* prompt via
`prompts.build_system_prompt(extra_context=...)`.

**Placement of context** — Notice the prompt order in
`build_chat_prompt`: system instructions + role + summary → recent
verbatim history → new question. This ordering matters: instructions
first, examples/retrieved context next, most-recent/most-relevant
content closest to the actual question.

**Streaming** — `app.py` iterates `plain_chain.stream(...)` and
updates a placeholder live, the standard Streamlit-streaming pattern.

**Token count in sidebar** — `st.session_state.last_turn_tokens`,
shown live in the sidebar along with the current backend and budget.

## A note on structured output + streaming together

Most providers can't stream *and* guarantee valid structured JSON in
the same call — the model has to finish the object before it's valid
JSON. This app works around it pedagogically: one pass streams plain
text for the live UI feel, a second (non-streamed) call gets the
validated `QAAnswer` object for the confidence/follow-up line under
the reply. In a real product you'd usually pick one or the other,
not both, per turn.

## Things worth trying as extensions

- Swap `tiktoken`'s `cl100k_base` counting for a provider-native
  token counter once you're firmly on one backend.
- Persist `st.session_state.history` to disk/DB so conversations
  survive a page refresh.
- Add a real retriever (vector store) and inject retrieved chunks
  into `extra_context` alongside the summary — that's Phase 3
  territory (RAG).
