# Context-Aware Q&A Assistant (Phase 2 project)

A Streamlit chat app built on LangChain's LCEL (`|`) syntax. The focus of this project is context engineering: deciding what actually goes into the model's context window on every turn, instead of just dumping the whole chat history into every request.

## Setup

```bash
cd context_qa_assistant
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# edit .env: set MODEL_BACKEND and the matching API key
streamlit run app.py
```

If you're using Ollama, install it separately (ollama.com) and run `ollama pull llama3.2:1b` before starting the app.

For Gemini, grab a free key from https://aistudio.google.com/app/apikey and put it in `.env` as `GOOGLE_API_KEY`.

## File map

| File | Role |
|---|---|
| `config.py` | The one place that picks the model backend (OpenAI, Anthropic, Gemini, or Ollama). Change `MODEL_BACKEND` in `.env` and everything downstream keeps working. |
| `prompts.py` | Dynamic system prompt (changes by user role) plus the `QAAnswer` structured output schema. |
| `context_manager.py` | Token counting (tiktoken) and the summarize-when-over-budget logic. |
| `app.py` | Streamlit UI, the LCEL chain (`prompt \| model \| parser`), and streaming. |

## How each topic shows up in the code

**LCEL syntax.** In `app.py`: `chain = prompt | structured_model` and `plain_chain = prompt | model`. Both are runnables, so both get `.stream()` and `.invoke()` without extra code.

**Swappable chat model backend.** `config.get_chat_model()`. The rest of the app only ever calls this function, never a specific SDK directly, so swapping backend is a one line env var change.

**Prompt templates and dynamic prompts.** `prompts.build_chat_prompt()` builds a different system message depending on the selected role (developer, manager, or student).

**Structured output.** `QAAnswer` is a Pydantic model passed to `model.with_structured_output(QAAnswer)`, so the reply gets parsed and validated into `.answer`, `.confidence`, and `.follow_up_question` instead of raw text that would need regex to pull apart.

**Message state and conversation history.** `st.session_state.history` is a list of `HumanMessage` and `AIMessage` objects, kept in Streamlit's session for this mini project.

**Context window budgeting.** `context_manager.manage_context()` counts tokens with tiktoken, and once `MAX_HISTORY_TOKENS` is exceeded, older turns get summarized into a rolling note while the last `KEEP_LAST_N_MESSAGES` stay verbatim. That summary gets injected into the system prompt through `prompts.build_system_prompt(extra_context=...)`.

**Placement of context.** In `build_chat_prompt`, the order is: system instructions and role and summary, then recent verbatim history, then the new question. Instructions first, older context next, the actual question last and closest to where the model answers.

**Streaming.** `app.py` iterates `plain_chain.stream(...)` and updates a placeholder live, the standard Streamlit streaming pattern.

**Token count in sidebar.** `st.session_state.last_turn_tokens`, shown in the sidebar next to the current backend and budget.

## A note on structured output and streaming together

Most providers can't stream and guarantee valid structured JSON in the same call, since the model has to finish the object before it's valid JSON. This app works around it by running one streamed plain text pass for the live UI, then a second non-streamed call that gets the validated `QAAnswer` object for the confidence and follow up line shown under the reply. In a real product you'd usually pick one approach per turn, not both.
